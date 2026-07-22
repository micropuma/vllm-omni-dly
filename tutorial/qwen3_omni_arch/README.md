# Qwen3-Omni-MoE 实现架构与推理优化分析

本文档以 Qwen3-Omni-MoE 为例，系统梳理 vLLM-Omni 中一个**生产级 Omni 模型**从模型实现、Pipeline 编排、Stage 间通信到部署配置的完整架构，并逐一解释每个模块对推理性能的影响。

通过阅读本文档，可以理解：

- 一个 Omni 模型在 vLLM-Omni 中由哪些模块组成，各模块职责是什么
- `max_num_seqs > 1`、CUDA graph、`async_chunk`、流式输出等能力是如何在源码层面实现的
- 如何以 Qwen3-Omni-MoE 为参考，优化其他 Omni 模型（如 MiniCPM-o 4.5）

---

## 总览：4 层架构

Qwen3-Omni-MoE 的实现分布在 4 层目录，20+ 个核心文件中：

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 4: 部署配置  (deploy/qwen3_omni_moe.yaml)              │
│   max_num_seqs=64, async_chunk=true, enforce_eager=false      │
├──────────────────────────────────────────────────────────────┤
│ Layer 3: Stage 间通信  (omni_connectors/)                     │
│   OmniChunkTransferAdapter + SharedMemoryConnector            │
├──────────────────────────────────────────────────────────────┤
│ Layer 2: Pipeline & Stage Bridge  (pipeline.py + processors)  │
│   3-stage 拓扑 + thinker2talker + talker2code2wav 6 个函数    │
├──────────────────────────────────────────────────────────────┤
│ Layer 1: 模型实现  (models/qwen3_omni/)                       │
│   Thinker / Talker / Code2Wav + 统一入口                      │
└──────────────────────────────────────────────────────────────┘
```

关键源码路径（相对于仓库根目录，GitHub 可点击跳转）：

| 层级 | 路径 |
|---|---|
| Layer 1 | [`vllm_omni/model_executor/models/qwen3_omni/`](../../vllm_omni/model_executor/models/qwen3_omni/) |
| Layer 2 | [`vllm_omni/model_executor/models/qwen3_omni/pipeline.py`](../../vllm_omni/model_executor/models/qwen3_omni/pipeline.py) |
| Layer 2 | [`vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py) |
| Layer 2 | [`vllm_omni/model_executor/stage_input_processors/qwen3_tts.py`](../../vllm_omni/model_executor/stage_input_processors/qwen3_tts.py) |
| Layer 3 | [`vllm_omni/distributed/omni_connectors/connectors/shm_connector.py`](../../vllm_omni/distributed/omni_connectors/connectors/shm_connector.py) |
| Layer 3 | [`vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py) |
| Layer 4 | [`vllm_omni/deploy/qwen3_omni_moe.yaml`](../../vllm_omni/deploy/qwen3_omni_moe.yaml) |

---

## Chapter 0: 从一条命令到多 Stage 协同运行的完整链路

```
vllm omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni
```

这一条命令背后，vLLM-Omni 经历了 **10 个阶段**，横跨 CLI → 配置解析 → Stage 编排 → 进程启动 → 通信器装配：

```
CLI 解析 → AsyncOmniEngine → 注册表查 Pipeline →
→ PipelineConfig + DeployYAML 合并成 StageConfig
→ 计算 Replica 布局 → 构建 VllmConfig → 派生 EngineCore 子进程
→ 建立 ZMQ 连接 → 装配 StagePool + Connector → Orchestrator 启动事件循环
```

本章以 Qwen3-Omni-MoE 为实例，逐阶段拆解。

---

### 阶段 1: CLI → AsyncOmniEngine

**入口**：[`cli/serve.py:86`](../../vllm_omni/entrypoints/cli/serve.py#L86)

```python
class OmniServeCommand:
    def cmd(self, args):
        uvloop.run(omni_run_server(args))
```

`omni_run_server()` → `build_async_omni_from_stage_config()` → `AsyncOmniEngine(model=..., **kwargs)`。

[`AsyncOmniEngine.__init__`](../../vllm_omni/engine/async_omni_engine.py#L213) 是整个初始化的总控制器，它依次执行：

```python
def __init__(self, model, **kwargs):
    # ① 解析 endpoint 限制
    self.endpoint_restrictions = StageConfigFactory.get_pipeline_endpoint_restrictions(...)

    # ② 解析所有 stage 配置（Pipeline + Deploy YAML 合并）
    self.config_path, self.stage_configs = self._resolve_stage_configs(model, kwargs)

    # ③ 拉起 orchestrator 后台线程
    self.orchestrator_thread = threading.Thread(
        target=self._bootstrap_orchestrator, daemon=True
    )
    self.orchestrator_thread.start()
```

---

### 阶段 2: HF Config 加载 + model_type 推断

**文件**：[`config_factory.py:107`](../../vllm_omni/config/config_factory.py#L107)

`StageConfigFactory.try_infer_model_type()` 加载模型的 `config.json`，读取 `model_type`。支持多种回退策略：

1. `AutoConfig.from_pretrained(model).model_type`（标准）
2. 直接读 `config.json` 文件（HF 报错时的回退）
3. 读 `model_index.json`（Diffusers 格式）
4. 用模型路径的 basename 匹配 pipeline key（最后手段）

对于 Qwen3-Omni，返回 `"qwen3_omni_moe"`。

---

### 阶段 3: 注册表查 PipelineConfig

**文件**：[`pipeline_registry.py:102-147`](../../vllm_omni/config/pipeline_registry.py#L102)

所有模型的 Pipeline 在一个全局字典中注册：

```python
OMNI_PIPELINES: dict[str, PipelineConfig | PipelineResolverFunc] = {
    "minicpmo_4_5":  MINICPMO_4_5_PIPELINE,        # 静态配置
    "qwen3_omni_moe": resolve_qwen3_omni_pipeline,  # 动态 resolver 函数
    ...
}
```

**为什么 value 有两种类型？**

- **静态 `PipelineConfig`**：模型只有一种拓扑（如 MiniCPM-o 固定 2 stage），直接用实例
- **resolver 函数**：同一个 `model_type` 有多个变体（如 Qwen3-Omni Instruct vs Captioner），需要根据 HF config 动态选择

[`resolve_pipeline_config()`](../../vllm_omni/config/pipeline_registry.py#L172) 分派：

```python
def resolve_pipeline_config(model_type, hf_config=None):
    pipeline = OMNI_PIPELINES[model_type]
    return pipeline(hf_config) if callable(pipeline) else pipeline
```

对于 `"qwen3_omni_moe"`，`pipeline` 是 `resolve_qwen3_omni_pipeline` 函数：

```python
@pipeline_cfg_resolver(config_type=Qwen3OmniMoeConfig)
def resolve_qwen3_omni_pipeline(hf_config):
    if not hf_config.enable_audio_output:
        return QWEN3_OMNI_THINKER_ONLY_PIPELINE  # Captioner: 1 stage
    return QWEN3_OMNI_PIPELINE                    # Instruct: 3 stage
```

`@pipeline_cfg_resolver` 是一个**装饰器工厂**（[源码 `stage_config.py:33`](../../vllm_omni/config/stage_config.py#L33)），自动加上 `isinstance(hf_config, Qwen3OmniMoeConfig)` 类型守卫：类型不匹配时静默返回 `None`，避免错误 config 传入导致 `AttributeError`。

**两种变体的结果**：

| 模型变体 | `enable_audio_output` | 返回的 Pipeline |
|---|---|---|
| Instruct | `True` | `QWEN3_OMNI_PIPELINE`（3 stage：thinker+talker+code2wav） |
| Captioner | `False` | `QWEN3_OMNI_THINKER_ONLY_PIPELINE`（1 stage：thinker only） |

---

### 阶段 4: PipelineConfig + DeployYAML → StageConfig（核心合并）

**文件**：[`merge_pipeline_deploy`](../../vllm_omni/config/stage_config.py#L834)

这是整个配置系统的核心。它把三种来源的信息合并成每个 stage 的最终运行配置：

```
PipelineConfig    (pipeline.py — 声明"有几个 stage、每个 stage 是什么类型")
    +
DeployConfig      (deploy/*.yaml — 声明"每个 stage 用几张 GPU、max_num_seqs 多少")
    +
CLI overrides     (--stage-overrides '{"0.max_num_seqs":8}')
    ↓
list[StageConfig] (每个 stage 一份，包含 engine_args + runtime 所有字段)
```

合并逻辑对每个 `StagePipelineConfig`（来自 pipeline）执行：

```
For each StagePipelineConfig (ps):

  ① _resolve_execution_mode()
     LLM_AR → (StageType.LLM, "ar")
     LLM_GENERATION → (StageType.LLM, "generation")
     DIFFUSION → (StageType.DIFFUSION, None)

  ② _select_processor_funcs()
     根据 async_chunk 挑选正确的 inter-stage 处理函数：
     async_chunk=true  → async_chunk_process_next_stage_input_func
     async_chunk=false → custom_process_next_stage_input_func

  ③ _build_engine_args()
     三层覆盖（由低到高优先级）：
       PipelineConfig 全局默认 (trust_remote_code, dtype, quantization...)
         → 对应 stage 的 StageDeployConfig 值 (max_num_seqs, gpu_memory_utilization, devices...)
           → engine_extras 透传

  ④ _resolve_scheduler()
     LLM_AR + async_scheduling → OmniARAsyncScheduler
     LLM_AR + no_async         → OmniARScheduler
     LLM_GENERATION             → OmniGenerationScheduler
     DIFFUSION                  → None (diffusion 独立调度)

  ⑤ _build_extras()
     合并 default_sampling_params（YAML）+ sampling_constraints（pipeline，高优先级）
     吸附 output_connectors / input_connectors / prompt_expand_func

  ⑥ 创建 StageConfig 实例
```

生成的 `StageConfig` 实例被 [`to_omegaconf()`](../../vllm_omni/config/stage_config.py#L939) 序列化为 OmegaConf 字典，供引擎初始化使用。

---

### 阶段 5: Deploy YAML 加载（继承 + 平台覆盖）

**文件**：[`load_deploy_config`](../../vllm_omni/config/stage_config.py#L605)

Deploy YAML 支持两个重要特性：

**特性 A: `base_config` 继承**
```yaml
base_config: qwen3_omni_moe.yaml  # 继承默认配置
stages:
  - stage_id: 1
    max_num_seqs: 8               # 只覆盖 stage 1 的 max_num_seqs
```

[`resolve_deploy_yaml()`](../../vllm_omni/config/stage_config.py#L579) 递归加载 base_config，按 `stage_id` deep-merge。

**特性 B: 平台覆盖**
```yaml
platforms:
  npu:
    stages:
      - stage_id: 0
        devices: "0,1,2,3"       # NPU 上 thinker 用 4 卡
  rocm:
    stages:
      - stage_id: 2
        enforce_eager: true       # ROCm 上 code2wav 不走 CUDA graph
```

[`_apply_platform_overrides()`](../../vllm_omni/config/stage_config.py#L660) 在合并前应用当前平台的覆盖。

加载结果是一个 [`DeployConfig`](../../vllm_omni/config/stage_config.py#L338)，包含 `async_chunk`、`connectors`、`platforms`、`edges` 和 `list[StageDeployConfig]`。

---

### 阶段 6: 构建 VllmConfig + 计算 Replica 布局

回到 [`AsyncOmniEngine._bootstrap_orchestrator()`](../../vllm_omni/engine/async_omni_engine.py#L422)，启动 orchestrator 线程后调用 [`_initialize_stages()`](../../vllm_omni/engine/async_omni_engine.py#L364)，进入 [`StageRuntime.initialize()`](../../vllm_omni/engine/stage_runtime.py#L230)：

```
StageRuntime.initialize()
  ├── _prepare_stage_plans()
  │     ├── compute_replica_layout()       # 决定每个 stage 几个副本、落在哪些 GPU
  │     ├── load_omni_transfer_config_for_model()  # 读取 connectors: / edges: 配置
  │     └── _build_logical_stage_init_plans()
  │           For each stage:
  │             ├── extract_stage_metadata()     # model_stage, engine_output_type...
  │             ├── get_stage_connector_spec()   # async_chunk 的 connector 配置
  │             ├── resolve_omni_kv_config()     # AR↔DiT KV 传输配置
  │             ├── build_engine_args_dict()     # 组装 OmniEngineArgs
  │             └── build_vllm_config()          # 创建 VllmConfig + executor_class
  │           For each replica:
  │             生成 ReplicaInitPlan(metadata, device, vllm_config, executor_class)
  │
  ├── _initialize_stage_replicas()
  │     (按 GPU 分组并行初始化，同 GPU 内串行)
  │     └── _initialize_local_llm_replica()
  │           ├── launch_stage_replica()    # 派生 EngineCore 子进程
  │           └── make_async_mp_client()    # 建立 ZMQ 连接到子进程
  │
  └── _finalize_initialized_stages()
        └── _assemble_stage_pools()
              └── StagePool(stage_idx, clients, output_processor, vllm_config)
```

**关键决策**：

1. **Replica 布局**：`compute_replica_layout()` 根据每个 stage 声明的 `devices` 和 `data_parallel_size` 计算需要多少个 EngineCore 副本，分布到哪些 GPU 上
2. **VllmConfig 构建**：`build_vllm_config()` 从 `OmniEngineArgs` 创建，这是 vLLM 原生 `VllmConfig` 的超集（包含 `model_stage`、`hf_config_name`、`stage_id`、`engine_output_type` 等 omni 扩展字段）
3. **Connector spec 注入**：`get_stage_connector_spec()` 从 deploy YAML 的 `connectors:` 段读取 connector 类型和参数，注入到 engine args 中

---

### 阶段 7: EngineCore 子进程派生 + ZMQ 连接

[`launch_stage_replica()`](../../vllm_omni/engine/stage_runtime.py#L568) 通过 vLLM 的 executor 框架派生 EngineCore 子进程。子进程内：

- 根据 `vllm_config.model_config.model_stage` 初始化对应的模型子模块（thinker/talker/code2wav）
- 创建 Scheduler（`OmniARAsyncScheduler` 等）
- 加载权重
- 创建 ZMQ socket 暴露通信端点

父进程通过 [`make_async_mp_client()`](../../vllm_omni/engine/stage_runtime.py#L591) 建立 ZMQ 连接，返回 [`StageEngineCoreClientBase`](../../vllm_omni/engine/stage_engine_core_client.py) 实例。

每个 EngineCore 子进程独立运行：
```
GPU 0: EngineCore(stage_id=0, model_stage="thinker")
GPU 1: EngineCore(stage_id=1, model_stage="talker")   ← 同 GPU 不同进程
GPU 1: EngineCore(stage_id=2, model_stage="code2wav")  ← 同 GPU 不同进程
```

---

### 阶段 8: StagePool + Connector 装配 + Orchestrator 启动

[`_assemble_stage_pools()`](../../vllm_omni/engine/stage_runtime.py#L683) 聚合每个 stage 的多个 EngineCore 客户端：

```python
StagePool(
    stage_idx=0,
    clients=[client_0, ...],                       # 该 stage 的所有 ZMQ 客户端
    output_processor=MultimodalOutputProcessor(),   # 输出格式转换器
    stage_vllm_config=thinker_vllm_config,
)
```

三个 `StagePool` 被传入 [`Orchestrator.__init__`](../../vllm_omni/engine/orchestrator.py#L212)，Orchestrator 建立两个核心 asyncio 协程：

```python
Orchestrator.run():
    asyncio.gather(
        self.request_handler(),              # 接收用户请求 → 分发到 Stage 0
        self.orchestration_output_handler(), # 收集各 stage 输出 → 转发下游 / 返回客户端
    )
```

**Connector 的装配时机**：

- **Plan time**（阶段 6）：`load_omni_transfer_config_for_model()` 解析 deploy YAML 的 `connectors:` 和 `edges:` 段，转为 `ConnectorSpec`
- **Replica init time**：`get_stage_connector_spec()` 提取 connector spec 注入 engine args，EngineCore 子进程据此创建 `OmniChunkTransferAdapter` + connector 实例
- **Runtime**：Orchestrator 通过 `StagePool` 的 client ZMQ 通道收发数据，connector 在后台线程中负责 stage 间的 chunk 传输

对于 Qwen3-Omni-MoE 的默认配置：

```
                SharedMemoryConnector
                      /dev/shm
    ┌─────────┐  ┌──────────────┐  ┌─────────┐
    │ Stage 0 │─▶│ ChunkTransfer │─▶│ Stage 1 │  thinker → talker
    │ thinker │  │   Adapter     │  │ talker  │  (async_chunk)
    └─────────┘  └──────────────┘  └────┬────┘
                                        │
                   SharedMemoryConnector│
                       /dev/shm         │
                                ┌──────────────┐  ┌──────────┐
                                │ ChunkTransfer │─▶│ Stage 2  │ talker → code2wav
                                │   Adapter     │  │ code2wav │ (async_chunk)
                                └──────────────┘  └──────────┘
```

---

### 完整 10 阶段总览

```
vllm omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni
  │
  │  阶段 1  CLI 解析 → AsyncOmniEngine.__init__
  │  阶段 2  HF Config 加载 → model_type="qwen3_omni_moe"
  │  阶段 3  注册表查找 → resolve_qwen3_omni_pipeline(hf_config)
  │           enable_audio_output=True → QWEN3_OMNI_PIPELINE (3 stages)
  │  阶段 4  PipelineConfig + deploy/qwen3_omni_moe.yaml → merge_pipeline_deploy()
  │           三层覆盖 + 调度器解析 + connector 吸附 → list[StageConfig]
  │  阶段 5  StageConfig.to_omegaconf() → OmegaConf 字典
  │  阶段 6  StageRuntime._prepare_stage_plans()
  │           Replica 布局 + VllmConfig 构建 + connector spec 注入
  │  阶段 7  StageRuntime._initialize_stage_replicas()
  │           派生 3 个 EngineCore 子进程 + 建立 ZMQ 连接
  │  阶段 8  StageRuntime._finalize_initialized_stages()
  │           装配 StagePool × 3 + MultimodalOutputProcessor
  │  阶段 9  Orchestrator(stage_pools, queues, ...)
  │  阶段 10 Orchestrator.run() → request_handler + output_handler 并发运行
  │
  ▼
  Ready to serve
```

---

### 源码索引

| 阶段 | 关键函数 | 文件与行号 |
|---|---|---|
| 1 | `AsyncOmniEngine.__init__` | [`async_omni_engine.py:213`](../../vllm_omni/engine/async_omni_engine.py#L213) |
| 2 | `StageConfigFactory.try_infer_model_type` | [`config_factory.py:107`](../../vllm_omni/config/config_factory.py#L107) |
| 3 | `resolve_pipeline_config` | [`pipeline_registry.py:172`](../../vllm_omni/config/pipeline_registry.py#L172) |
| 3 | `resolve_qwen3_omni_pipeline` | [`pipeline.py:98`](../../vllm_omni/model_executor/models/qwen3_omni/pipeline.py#L98) |
| 4 | `merge_pipeline_deploy` | [`stage_config.py:834`](../../vllm_omni/config/stage_config.py#L834) |
| 4 | `_build_engine_args` | [`stage_config.py:760`](../../vllm_omni/config/stage_config.py#L760) |
| 4 | `_resolve_scheduler` | [`stage_config.py:181`](../../vllm_omni/config/stage_config.py#L181) |
| 5 | `load_deploy_config` | [`stage_config.py:605`](../../vllm_omni/config/stage_config.py#L605) |
| 5 | `_apply_platform_overrides` | [`stage_config.py:660`](../../vllm_omni/config/stage_config.py#L660) |
| 5 | `StageConfig.to_omegaconf` | [`stage_config.py:939`](../../vllm_omni/config/stage_config.py#L939) |
| 6 | `StageRuntime._prepare_stage_plans` | [`stage_runtime.py:275`](../../vllm_omni/engine/stage_runtime.py#L275) |
| 6 | `build_vllm_config` | [`stage_init_utils.py:748`](../../vllm_omni/engine/stage_init_utils.py#L748) |
| 6 | `load_omni_transfer_config_for_model` | [`stage_init_utils.py:1030`](../../vllm_omni/engine/stage_init_utils.py#L1030) |
| 7 | `launch_stage_replica` | [`stage_runtime.py:568`](../../vllm_omni/engine/stage_runtime.py#L568) |
| 7 | `make_async_mp_client` | [`stage_runtime.py:591`](../../vllm_omni/engine/stage_runtime.py#L591) |
| 8 | `_assemble_stage_pools` | [`stage_runtime.py:683`](../../vllm_omni/engine/stage_runtime.py#L683) |
| 9 | `Orchestrator.__init__` | [`orchestrator.py:212`](../../vllm_omni/engine/orchestrator.py#L212) |
| 10 | `Orchestrator.run` | [`orchestrator.py:330`](../../vllm_omni/engine/orchestrator.py#L330) |

---

## Layer 1: 模型实现层（7 个源文件）

目录：[`vllm_omni/model_executor/models/qwen3_omni/`](../../vllm_omni/model_executor/models/qwen3_omni/)

| 文件 | 行数 | 职责 |
|---|---|---|
| [`__init__.py`](../../vllm_omni/model_executor/models/qwen3_omni/__init__.py) | 9 | 空包标记（有意不做重导出） |
| [`pipeline.py`](../../vllm_omni/model_executor/models/qwen3_omni/pipeline.py) | 111 | 3-stage pipeline 拓扑定义 |
| [`qwen3_omni.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py) | 1392 | **统一入口**，dispatch 到 thinker/talker/code2wav |
| [`qwen3_omni_moe_thinker.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py) | 1802 | Thinker：多模态理解 + 文本生成 |
| [`qwen3_omni_moe_talker.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py) | 393 | Talker：文本嵌入 → RVQ 音频编码 |
| [`qwen3_omni_code2wav.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py) | 360 | Code2Wav：RVQ 编码 → 24kHz 波形 |
| [`qwen3_omni_moe_code_predictor_mtp.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_code_predictor_mtp.py) | 30 | Code Predictor 薄封装（MTP） |

### 1.1 统一入口：[`Qwen3OmniMoeForConditionalGeneration`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L81)

**继承链**：
```
nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE,
SupportsRealtime, CustomProcessMixin
→ Qwen3OmniMoeForConditionalGeneration
```

根据 `model_stage` 分发到三个子模块（[`__init__` L104-L213](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L104)）：

```python
if self.model_stage == "thinker":
    self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(...)
elif self.model_stage == "talker":
    self.talker = Qwen3OmniMoeTalkerForConditionalGeneration(...)
elif self.model_stage == "code2wav":
    self.code2wav = Qwen3OmniMoeCode2Wav(...)
```

[`forward()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L347) 中的关键分支：

```python
# thinker（L374-L404）
hidden_states, captured_layer_dict = self.thinker(...)
# 捕获 layer 0（word embedding）和 layer 24（深层特征）的 hidden states
return OmniOutput(
    multimodal_outputs={"hidden_states": {"layers": captured_layer_dict}, "embed": ...}
)

# talker（L407-L423）
return self.talker.forward(inputs_embeds=inputs_embeds, ...)

# code2wav（L462-L470）
# runtime_additional_information 是 list[dict]，逐元素遍历，不取 [0]
for info in runtime_additional_information:
    left_context_size.append(info.get("meta", {}).get("left_context_size"))
return generate_audio(codes, left_context_size, seq_token_counts)
```

**推理影响**：

- **`SupportsRealtime`**：[`buffer_realtime_audio()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L225) 做分段实时音频输入（5 秒/段），TTFP 大幅降低
- **`realtime_max_tokens = 64`**（[L102](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L102)）：控制每次编码的音频分片大小
- **`eager_omni_postprocess_before_async_output = True`**（talker，[L163](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L163)）：`hidden_states.last` 留在 GPU 上，避免 D2H 传输
- **`gpu_resident_buffer_keys`**（[L161](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L161)）：标记中间结果留在 GPU 上，减少 CPU-GPU 传输
- **per-request 的 `runtime_additional_information`**（[L462-L470](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L462)）：作为 `list[dict]` 逐元素处理（非 `[0]` 索引），天然支持 `max_num_seqs > 1`

### 1.2 Thinker：多模态理解

**文件**：[`qwen3_omni_moe_thinker.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py)（1802 行，12 个类，4 个子模块）

#### 子模块 A：视觉编码器 [`Qwen3Omni_VisionTransformer`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py#L139)

```python
class Qwen3Omni_VisionTransformer(_Qwen3Omni_VisionTransformer):
    def forward(self, hidden_states, grid_thw, ...):
        # patch_embed → RoPE → cu_seqlens → MultiBlock attention → DeepStack merge
```

**推理影响**：
- RoPE 位置编码直接在 GPU 上计算，不经过 CPU
- `cu_seqlens` 支持变长 batch 图块：一次 forward 处理多张不同分辨率的图
- DeepStack 机制：中间层 hidden states 被收集并注入 LLM 对应层，提供多尺度视觉特征

#### 子模块 B：音频编码器 [`Qwen3OmniMoeAudioEncoder`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py#L325)

```python
class Qwen3OmniMoeAudioEncoder(_Qwen3OmniMoeAudioEncoder):
    def forward(self, input_features, feature_lens, aftercnn_lens, ...):
        # 1. 分 window 处理（chunk_size = n_window * 2）
        # 2. 每个 window 走 3 层 Conv2d（stride 2×3 = 8x 下采样）
        # 3. 正弦位置编码 → Transformer 层 → 最终投影
```

**推理影响**：
- 音频分 window 处理：长音频不会 OOM
- 尾部 window 独立处理：避免 padding 计算浪费
- `conv_chunksize` 子分片：CNN 阶段的额外内部分片，防止大 batch 长音频 OOM

#### 子模块 C：LLM 模型 [`Qwen3MoeLLMModel`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py#L523)

```python
class Qwen3MoeLLMModel(_Qwen3MoeLLMModel):
    def forward(self, ..., deepstack_input_embeds, capture_layer_indices):
        for layer_idx, layer in enumerate(self.layers):
            # 每层注入对应深度的 deepstack 视觉特征
            if layer_idx in deepstack_input_embeds:
                hidden_states += deepstack_input_embeds[layer_idx]
            # 捕获指定层的 hidden states（给 talker 用）
            if layer_idx in capture_layer_indices:
                captured[layer_idx] = hidden_states.clone()
```

**推理影响**：
- `capture_layer_indices=[0, 24]`：捕获词嵌入层和第 24 层输出，作为 talker 的输入条件（thinker→talker 核心数据流）
- DeepStack 注入：视觉多尺度特征在各层注入，而非仅在输入端拼接，提升视觉理解质量

#### 子模块 D：量化分量分离（[L1164-L1190](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py#L1164)）

```python
quantization_config = ComponentQuantizationConfig(
    component_quant={
        "audio_tower": None,       # 音频编码器保持 BF16
        "visual": None,            # 视觉编码器保持 BF16
        "language_model": "fp8",   # 仅 LLM 量化
    }
)
```

**推理影响**：
- 编码器对精度敏感，保持 BF16；LLM 参数多，FP8 量化节省一半显存
- 显著降低显存压力，允许更大 `max_model_len` 或更大 batch
- 这是解决编码器内存压力问题的标准方案

### 1.3 Talker：文本 → RVQ 音频编码

**文件**：[`qwen3_omni_moe_talker.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py)（[L32](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py#L32)）

```
Qwen3OmniMoeTalkerForConditionalGeneration(nn.Module, SupportsPP)
  ├── text_projection  : ResizeMLP  # thinker emb → talker dim
  ├── hidden_projection: ResizeMLP  # thinker hid → talker dim
  ├── codec_head       : Linear     # talker hidden → 音频 codec vocab
  ├── language_model   : Qwen3MoeLLMModel（替换了 text embedding）
  └── code_predictor   : CodePredictorWrapper  # RVQ layers 1-15
```

**[`code_predictor_forward()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py#L122)**：MTP 一次出 16 层 RVQ 编码：

```python
for i in range(seq_len):
    layer0_code = result_codes[:, 0, i]
    codec_emb = self.codec_embed(layer0_code)
    _, result_codes[:, :, i], summed = self.code_predictor(codec_emb, ...)
    summed_embeddings[:, i, :] = summed
```

**推理影响**：
- **无 KV cache**：talker 每次做 re-prefill，因为序列通常较短，re-prefill 比维护 KV cache 更经济
- **MTP（Multi-Token Prediction）**：code_predictor 一次出 16 层 RVQ 编码（非逐层自回归），大幅减少生成步数
- **ResizeMLP 投影**：thinker 和 talker 的 hidden size 不同（如 2048 → 4096），通过 SiLU+Linear×2 映射（[`Qwen3OmniMoeTalkerResizeMLP`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py#L324)）

### 1.4 Thinker→Talker 投影（在 [`qwen3_omni.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py) 中）

**[`talker_preprocess()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L689)** — 区分 prefill/decode：
```python
# prefill 路径：thinker hidden states(0,24) + embeds → 投影 → talker embeds
# decode 路径：thinker 每步 decode embedding → text_projection → 缓存 trailing_text
```

**[`_thinker_to_talker_prefill()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L932)** — 完整投影管线：
```
thinker prompt chat template →
  user segments:      hidden_projection(hidden_states)
  assistant segments: text_projection(embeddings)
  TTS tokens:         特殊 speaker embedding
→ 拼接成 talker prompt embeddings
```

**推理影响**：prefill 时完整投影整段提示，decode 时只投影单 token，大幅降低每一步的计算量。

### 1.5 Code2Wav：RVQ 编码 → 24kHz 波形

**文件**：[`qwen3_omni_code2wav.py`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py)（[L35](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py#L35)）

```
Qwen3OmniMoeCode2Wav(nn.Module)
  1. Code Embedding: [B, 16, T] → [B, T, D]（平均 16 层 RVQ）
  2. Pre-Transformer: 滑动窗口 attention → 时序上下文
  3. Upsampling: ConvNeXt blocks × 多级上采样（~1280x）
  4. Decoder: SnakeBeta 激活 + 残差单元 → 24kHz 波形
```

**[`enable_cudagraph()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py#L137)** — CUDA Graph 支持：
```python
def enable_cudagraph(self):
    self._cudagraph_wrapper = CUDAGraphDecoderWrapper(self)
    self._cudagraph_wrapper.warmup(warmup_sizes,
                                    codec_chunk_frames,
                                    codec_left_context_frames)
```

**[`chunked_decode()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py#L216)** — 自动选择 CUDA graph 路径：
```python
if self._cudagraph_enabled and self._cudagraph_wrapper is not None:
    return self._cudagraph_wrapper.chunked_decode_with_cudagraph(codes, seq_token_counts)
```

**[`chunked_decode_streaming()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py#L276)** — async_chunk 流式解码，含尾部补偿：
```python
# 补回上一 chunk 因因果卷积丢失的 tail 样本（~23ms）
start = max(0, left_context_size[idx] * total_upsample - tail)
```

**推理影响**：
- 纯 feed-forward 模型天然适合 CUDA graph，预热后直接 trace 执行
- `chunked_decode` 分块解码长序列避免 OOM
- `chunked_decode_streaming` 配合 async_chunk：每收到一批 talker codes 就立刻解码一段音频，实现流式输出
- 尾部补偿消除 chunk 边界的 ~23ms 音频缺口

---

## Layer 2: Pipeline 与 Stage Bridge

### 2.1 Pipeline 拓扑

**文件**：[`pipeline.py`](../../vllm_omni/model_executor/models/qwen3_omni/pipeline.py)（[L22](../../vllm_omni/model_executor/models/qwen3_omni/pipeline.py#L22)）

```python
QWEN3_OMNI_PIPELINE = PipelineConfig(
    stages=(
        StagePipelineConfig(stage_id=0, model_stage="thinker",
            execution_type=LLM_AR,
            engine_output_type="latent",
            custom_process_next_stage_input_func="thinker2talker_full_payload",
            async_chunk_process_next_stage_input_func="thinker2talker_async_chunk",
        ),
        StagePipelineConfig(stage_id=1, model_stage="talker",
            execution_type=LLM_AR, input_sources=(0,),
            engine_output_type="latent",
            custom_process_next_stage_input_func="talker2code2wav_full_payload",
            async_chunk_process_next_stage_input_func="talker2code2wav_async_chunk",
            sampling_constraints={"stop_token_ids": [2150]},
        ),
        StagePipelineConfig(stage_id=2, model_stage="code2wav",
            execution_type=LLM_GENERATION, input_sources=(1,),
            engine_output_type="audio",
        ),
    ),
)
```

**推理影响**：
- 每个 stage 定义**双路径**函数：`custom_process_next_stage_input_func`（全量同步）和 `async_chunk_process_next_stage_input_func`（分块异步），运行时根据 `async_chunk` 配置二选一
- Stage 2 的 `LLM_GENERATION`（非 `LLM_AR`）告诉 scheduler 这是一次性 forward 而非逐 token 生成，更好地规划显存和调度
- `stop_token_ids=[2150]` 强制 talker 在 codec EOS 停止，防止无限生成

### 2.2 Stage Input Processor：6 个跨 Stage 函数

**文件**：[`stage_input_processors/qwen3_omni.py`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py)

| 函数 | 行号 | 方向 | 模式 | 作用 |
|---|---|---|---|---|
| [`thinker2talker_async_chunk`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L434) | L434 | thinker→talker | async_chunk | 分 chunk 传递 hidden states |
| [`thinker2talker_full_payload`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L530) | L530 | thinker→talker | full payload | 一次性传递全部 hidden states |
| [`thinker2talker_token_only`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L616) | L616 | thinker→talker | sync placeholder | 为 talker 分配 KV cache 槽位 |
| [`talker2code2wav_async_chunk`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L679) | L679 | talker→code2wav | async_chunk | 分 chunk 传递 codec codes |
| [`talker2code2wav_full_payload`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L757) | L757 | talker→code2wav | full payload | 一次性传递全部 codes |

**流式状态类**（同文件）：

| 类 | 行号 | 作用 |
|---|---|---|
| [`_Thinker2TalkerStreamingState`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L347) | L347 | 跟踪 thinker→talker 的 token 增量 |
| [`_Qwen3OmniStreamingState`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L354) | L354 | 聚合 thinker+talker 的完整流式状态 |

**[`thinker2talker_async_chunk`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L434) 延迟发送策略**：

```python
def thinker2talker_async_chunk(transfer_manager, multimodal_output, request, is_finished):
    if chunk_id == 0:
        # 缓存 prefill embeddings + hidden_states
        payload = OmniPayloadStruct(
            embed=EmbeddingsStruct(prefill=thinker_emb, tts_bos=..., tts_eos=..., tts_pad=...),
            hidden_states=HiddenStatesStruct(output=thinker_hid),
            ids=IdsStruct(all=all_token_ids, prompt=prompt_token_ids),
            meta=MetaStruct(finished=...),
        )
        transfer_manager.request_payload[request_id] = to_dict(payload)
        return None  # 延迟发送，等第一个 decode token
    else:
        # 只发当前 decode token 的 embedding
        return OmniPayloadStruct(
            embed=EmbeddingsStruct(decode=thinker_emb.detach().cpu()),
            meta=MetaStruct(finished=is_finished),
        )
```

**推理影响**：
- chunk 0 不立刻发送，等 chunk 1 的 decode token 到达后合并 prefill+decode，talker 一次性拿到完整条件
- 支持 **resumable 实时音频流**（[`_construct_thinker2talker_streaming_input_async_chunk`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L264)）：新音频分片到达时触发增量 prefill

### 2.3 Stage 内辅助函数

**[`make_omni_output()`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L484)** — talker 的 batch 处理（[L527-L534](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L527)）：

```python
# 从 model_intermediate_buffer（或 runtime_additional_information）遍历全部请求
info_dicts = kwargs.get("model_intermediate_buffer")
if info_dicts is None:
    info_dicts = kwargs.get("runtime_additional_information")
# 逐请求合并
code_predictor_codes = [info.get("codes", {}).get("audio") for info in info_dicts]
audio_codes = torch.cat(code_predictor_codes, dim=0)
span_len = audio_codes.shape[0]
talker_hidden = talker_hidden[:span_len]  # 按实际 batch 裁切
```

**推理影响**：逐元素迭代 + `torch.cat` 合并，而非取 `[0]`，是支持 `max_num_seqs > 1` 的关键。

---

## Layer 3: Stage 间通信层

### 3.1 SharedMemoryConnector

**文件**：[`shm_connector.py`](../../vllm_omni/distributed/omni_connectors/connectors/shm_connector.py)（[L17](../../vllm_omni/distributed/omni_connectors/connectors/shm_connector.py#L17)）

```python
class SharedMemoryConnector(OmniConnectorBase):
    def __init__(self, config):
        self.threshold = int(config.get("shm_threshold_bytes", 65536))  # 64KB 阈值

    def put(self, from_stage, to_stage, put_key, data):
        payload = self.serialize_obj(data)
        if size >= self.threshold:
            # 大 payload → POSIX 共享内存（/dev/shm） + 文件锁
            meta = shm_write_bytes(payload, name=put_key)
            metadata = {"shm": meta, "size": size}
        else:
            # 小 payload → 内联序列化，零 SHM 创建/销毁开销
            metadata = {"inline_bytes": payload, "size": size}
        return True, size, metadata
```

**推理影响**：
- **64KB 阈值**（[L34](../../vllm_omni/distributed/omni_connectors/connectors/shm_connector.py#L34)）：decode 的单 token embedding（~16KB）走内联路径，prefill 的完整 hidden states（可达几十 MB）走 SHM
- **文件锁**（[`put` L59](../../vllm_omni/distributed/omni_connectors/connectors/shm_connector.py#L59)）：保证多进程并发安全的 key 级写入
- **零网络开销**：同机通信用 `/dev/shm`（POSIX 共享内存），延迟 < 10μs

### 3.2 OmniChunkTransferAdapter

**文件**：[`chunk_transfer_adapter.py`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py)（[L23](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L23)）

这是 async_chunk 的调度核心，管理 per-request 的 chunk 发送/接收生命周期：

```python
class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    put_req_chunk: dict[str, int]          # 每个请求已发送的 chunk 序号
    get_req_chunk: dict[str, int]          # 每个请求已接收的 chunk 序号
    finished_requests: set[str]            # 已完成的请求
    request_payload: dict[str, dict]       # 缓存的 prefill payload
    code_prompt_token_ids: dict[str, list] # 累积的 codec frames
    _active_window: int                     # 活跃流窗口上限
```

**核心方法**：

| 方法 | 行号 | 职责 |
|---|---|---|
| [`save_async()`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L147) | L147 | 每步 decode 后将 chunk 入队，唤醒后台发送线程 |
| [`_send_single_request()`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L294) | L294 | 后台线程：调 `custom_process_next_stage_input_func` → `connector.put()` |
| [`load_async()`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L124) | L124 | 下游 stage 注册等待 chunk |
| [`_poll_single_request()`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L195) | L195 | 后台轮询接收，写入 `request.additional_information` |
| [`process_pending_chunks()`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L443) | L443 | Scheduler 集成：恢复 `WAITING_FOR_CHUNK` 请求为 running |
| [`finish_requests()`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L746) | L746 | 清理完成/abort 请求的 chunk 状态 |

**推理影响**：
- **跨 stage 时间重叠**：thinker 每生成一个 token 就通过后台线程异步发给 talker，E2E 延迟 = max(thinker, talker+code2wav) 而非 sum()
- **背压控制**：`_active_window` 和 chunk queue 限制活跃流数量，防止显存爆炸
- **容错**：zombie cleanup、preempt 检查、finish marker 保证中途 abort 不残留状态

### 3.3 其他 Connector 类型

[`omni_connectors/connectors/`](../../vllm_omni/distributed/omni_connectors/connectors/) 目录下提供了跨机器通信的连接器：

| Connector | 文件 | 传输方式 | 适用场景 |
|---|---|---|---|
| `SharedMemoryConnector` | [`shm_connector.py`](../../vllm_omni/distributed/omni_connectors/connectors/shm_connector.py) | POSIX 共享内存 | 同机多 GPU |
| `MooncakeStoreConnector` | [`mooncake_store_connector.py`](../../vllm_omni/distributed/omni_connectors/connectors/mooncake_store_connector.py) | TCP (分布式 KV store) | 跨机（非 RDMA） |
| `MooncakeTransferEngineConnector` | [`mooncake_transfer_engine_connector.py`](../../vllm_omni/distributed/omni_connectors/connectors/mooncake_transfer_engine_connector.py) | RDMA (Mooncake TransferEngine) | 跨机（RDMA 网卡） |
| `MoriTransferEngineConnector` | [`mori_transfer_engine_connector.py`](../../vllm_omni/distributed/omni_connectors/connectors/mori_transfer_engine_connector.py) | RDMA 或 XGMI (Mori IOEngine) | 跨机 RDMA / AMD GPU 直连 |

---

## Layer 4: 部署配置

**文件**：[`vllm_omni/deploy/qwen3_omni_moe.yaml`](../../vllm_omni/deploy/qwen3_omni_moe.yaml)

```yaml
async_chunk: true  # 开启跨 stage 流式 chunk 传递

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      initial_codec_chunk_frames: 4   # 首个 chunk 只等 4 帧就生成音频 → 极低 TTFP
      codec_chunk_frames: 25           # 后续每 25 帧发一个 chunk
      codec_left_context_frames: 25   # 每个 chunk 带 25 帧左上下文

stages:
  - stage_id: 0  # Thinker
    max_num_seqs: 64
    max_num_batched_tokens: 32768
    gpu_memory_utilization: 0.9
    devices: "0"            # GPU 0 独占
    # enforce_eager 不设 → 默认 false → CUDA graph 开启

  - stage_id: 1  # Talker
    max_num_seqs: 64
    max_num_batched_tokens: 32768
    gpu_memory_utilization: 0.6   # 只占 60%，留空间给 code2wav
    devices: "1"            # 与 code2wav 共享 GPU 1

  - stage_id: 2  # Code2Wav
    max_num_seqs: 64
    max_num_batched_tokens: 65536
    gpu_memory_utilization: 0.1   # 只占 10%（模型极小）
    enforce_eager: false    # CUDA graph 明确开启
    devices: "1"            # 与 talker 共享 GPU 1
```

默认 2 GPU 布局：

```text
GPU 0: Thinker（最大模型，占 90% 显存）
GPU 1: Talker（占 60% 显存）+ Code2Wav（占 10% 显存）
```

**推理影响**：
- **`initial_codec_chunk_frames: 4`**：第一个音频 chunk 只要 4 帧 codec frames，极低 TTFP
- **`codec_left_context_frames: 25`**：每个流式 chunk 带 25 帧历史上下文，避免边界音频失真
- **2 GPU 布局**：充分利用显存，不浪费

---

## 推理性能影响总结

按用户可感知的指标维度：

| 指标 | 关键模块 | 效果 |
|---|---|---|
| **TTFT**（首 token 延迟） | [`Thinker`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py) 多模态编码 + CUDA graph + `max_num_seqs=64` | 编码器在 GPU 上并行批量处理 |
| **TTFP**（首音频延迟） | [`thinker2talker_async_chunk`](../../vllm_omni/model_executor/stage_input_processors/qwen3_omni.py#L434) + `initial_codec_chunk_frames=4` | thinker 刚开始 decode 就触发 code2wav |
| **吞吐** | `max_num_seqs=64` + [`OmniChunkTransferAdapter`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py) | 64 路并发，active_window 背压控制 |
| **显存** | [`ComponentQuantizationConfig`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py#L1164)（编码器 BF16，LLM FP8）| LLM 量化省一半显存，编码器精度不损失 |
| **流式输出** | [`SupportsRealtime`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L225) + [`chunked_decode_streaming`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py#L276) | 客户端分段接收音频（5 秒/段） |
| **E2E 延迟** | `enforce_eager=false` + `async_chunk` 跨 stage 重叠 | 三个阶段时间重叠，非串行求和 |
| **容错** | [`OmniChunkTransferAdapter`](../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py) zombie cleanup + preempt 检查 | abort 请求不残留 chunk 状态 |

---

## 对比：Qwen3-Omni-MoE vs MiniCPM-o 4.5

| 能力 | Qwen3-Omni-MoE | MiniCPM-o 4.5 |
|---|---|---|
| CUDA Graph | `enforce_eager=false`（3 个 stage） | `enforce_eager=true`（全部） |
| Continuous Batching | `max_num_seqs=64` | `max_num_seqs=1` |
| async_chunk | `true` | `false` |
| 流式音频输出 | [`SupportsRealtime`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L81) | 无 |
| `runtime_additional_information` | 遍历 `list[dict]`（[L462-L470](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py#L462)） | `[0]` 硬编码 |
| Stage 数 | 3（Code2Wav 独立，可 CG） | 2（TTS+Token2Wav 融合） |
| 编码器量化分离 | [`ComponentQuantizationConfig`](../../vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py#L1164) | 无 |

Qwen3-Omni-MoE 的每一处实现都对应 MiniCPM-o 4.5 RFC（[#5069](https://github.com/vllm-project/vllm-omni/issues/5069)）提出的一个待优化方向，是学习 vLLM-Omni 架构优化的最佳参考模型。

---

## 相关文件完整索引

```
vllm_omni/model_executor/models/qwen3_omni/
  ├── __init__.py                                 # 空包
  ├── pipeline.py                                 # Pipeline 拓扑
  ├── qwen3_omni.py                               # 统一入口 + 投影 + MTP
  ├── qwen3_omni_moe_thinker.py                   # Thinker（1802 行）
  ├── qwen3_omni_moe_talker.py                    # Talker
  ├── qwen3_omni_code2wav.py                     # Code2Wav + CUDA graph
  └── qwen3_omni_moe_code_predictor_mtp.py       # Code Predictor 封装

vllm_omni/model_executor/stage_input_processors/
  ├── qwen3_omni.py                               # thinker↔talker（5 函数 + 2 状态类）
  └── qwen3_tts.py                                # talker↔code2wav

vllm_omni/distributed/omni_connectors/
  ├── connectors/shm_connector.py                  # 共享内存连接器
  └── transfer_adapter/chunk_transfer_adapter.py   # async_chunk 调度核心

vllm_omni/deploy/
  └── qwen3_omni_moe.yaml                         # 默认部署配置
```
