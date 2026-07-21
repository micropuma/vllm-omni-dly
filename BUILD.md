# vLLM-Omni 源码编译指南 (RTX 3090 + CUDA 13.2)

## 环境信息

| 项目 | 详情 |
|------|------|
| 操作系统 | Ubuntu / Linux 6.8.0-106-generic |
| Python | 3.12.3 |
| GPU | 3x NVIDIA GeForce RTX 3090 (24 GB) |
| 驱动版本 | 590.48.01 |
| CUDA 工具包 | 13.2 (nvcc 13.2.51) |
| CUDA_HOME | /usr/local/cuda -> /usr/local/cuda-13.2 |
| Rust | rustc 1.93.1 (2026-02-11) |

## 安装结果

| 包 | 版本 |
|----|------|
| torch | 2.13.0+cu132 |
| torchvision | 0.28.0+cu132 |
| triton | 3.7.1 |
| vllm | 0.24.1.dev0+gee0da84ab (v0.24.0 tag) |
| vllm-omni | 0.1.dev2228+g109942465 (editable install) |
| flashinfer-python | 0.6.12 |
| flashinfer-cubin | 0.6.12 |

## 完整编译步骤

### 1. 创建虚拟环境

```bash
cd /mnt/home/douliyang/mlsys/vllm-omini/vllm-omni-dly
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装 PyTorch (CUDA 13.2)

> **重要**：PyTorch 必须从 cu132 索引安装，系统 CUDA 13.2 对应 PyTorch 的 cu132 构建。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

> torchaudio 在 cu132 中暂无预编译包，不影响 vllm-omni 功能。

验证 CUDA 可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
# 应输出: True 13.2 NVIDIA GeForce RTX 3090
```

### 3. 安装编译工具

```bash
pip install cmake ninja packaging setuptools-rust jinja2 setuptools-scm
```

Rust 已安装 (`rustc 1.93.1`)，如没有需安装：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 4. 安装 vLLM 依赖

vllm-omni 需要 vllm **v0.24.x**（非 main 分支 v0.21.x）。原因是 vllm-omni 依赖了 `vllm.entrypoints.serve.utils.*` 等模块，这些模块在 v0.21.x 中尚未引入。

```bash
# 切到 vllm 仓库，检出 v0.24.0 tag
cd /mnt/home/douliyang/mlsys/vllm
git fetch --tags
git checkout v0.24.0
```

vllm 的 `requirements/cuda.txt` 固定了 `torch==2.11.0`，而我们安装了 `torch 2.13.0+cu132`。需要注释掉 torch 相关行，否则 pip 会尝试降级 torch：

```bash
# 编辑 requirements/cuda.txt，注释掉这三行：
#   torch==2.11.0
#   torchaudio==2.11.0
#   torchvision==0.26.0
```

安装 vllm 的其他依赖（不含 torch）：

```bash
cd /mnt/home/douliyang/mlsys/vllm-omini/vllm-omni-dly
source .venv/bin/activate

# 安装 vllm CUDA 依赖（已注释掉 torch）
pip install -r /mnt/home/douliyang/mlsys/vllm/requirements/cuda.txt
```

> **注意**：vllm 0.24.0 相比 0.21.x 有几个依赖版本升级：
> - flashinfer-python/cubin: 0.6.11.post2 -> 0.6.12
> - nvidia-cudnn-frontend: >=1.13.0,<1.19.0 -> >=1.19.1
> - fastsafetensors: >=0.2.2 -> >=0.3.2
> - nvidia-cutlass-dsl: 4.5.0 -> 4.5.2
> - 新增 humming-kernels[cu13]==0.1.6

### 5. 编译 vLLM C++/Rust/CUDA 扩展

```bash
cd /mnt/home/douliyang/mlsys/vllm-omini/vllm-omni-dly
source .venv/bin/activate

CUDA_HOME=/usr/local/cuda \
MAX_JOBS=64 \
CMAKE_BUILD_PARALLEL_LEVEL=64 \
CARGO_BUILD_JOBS=64 \
pip install --no-build-isolation --no-deps /mnt/home/douliyang/mlsys/vllm
```

关键环境变量：
- `--no-build-isolation`: 使用已安装的 torch（避免 pip 隔离环境中安装 torch 2.11.0）
- `--no-deps`: 跳过依赖安装（已在上一步手动安装）
- `MAX_JOBS=64`: NVCC 并行编译线程数
- `CMAKE_BUILD_PARALLEL_LEVEL=64`: CMake 并行编译级别
- `CARGO_BUILD_JOBS=64`: Rust cargo 并行编译线程数
- `CUDA_HOME`: 必须指向系统 CUDA 安装路径

编译产物：wheel 文件约 521 MB，包含 CUDA kernel、C++ 扩展、Rust 前端二进制。

### 6. 安装 vLLM-Omni (editable)

```bash
cd /mnt/home/douliyang/mlsys/vllm-omini/vllm-omni-dly
source .venv/bin/activate
pip install -e .
```

`setup.py` 会：
1. 自动检测 CUDA 后端（通过 `torch.version.cuda`）
2. 加载 `requirements/cuda.txt`（包含 common.txt + CUDA 特定依赖）
3. 以 editable 模式安装，代码修改即时生效

### 7. 验证安装

```bash
# 验证 vLLM
python -c "import vllm; print(vllm.__version__)"

# 验证 vLLM-Omni
python -c "import vllm_omni; print('OK')"

# 验证 CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期输出：
```
vllm: 0.24.1.dev0+gee0da84ab.d20260721
vllm_omni imported successfully
True NVIDIA GeForce RTX 3090
```

预期的 WARNING（非致命）：
- `Failed to import from vllm._qutlass_C`: 预编译 QuTlass 扩展缺失，不影响基本推理
- `mismatched major/minor versions` (vLLM-Omni 0.1 vs vLLM 0.24): 版本号命名策略差异，不影响功能

### 8. 恢复 vLLM 仓库

```bash
cd /mnt/home/douliyang/mlsys/vllm
git checkout -- requirements/cuda.txt
git checkout main
```

## 故障排查

### 常见问题

1. **`No module named 'vllm.entrypoints.serve.utils'`**
   - 原因：vllm 版本太旧 (< v0.24.0)
   - 解决：`git checkout v0.24.0` 或更新的 tag

2. **`ERROR: No matching distribution found for torch`**
   - 原因：PyTorch 索引 URL 不对，CUDA 13.2 对应 `cu132` 不是 `cu131`
   - 解决：使用 `--index-url https://download.pytorch.org/whl/cu132`

3. **pip 尝试降级 torch 到 2.11.0**
   - 原因：vllm 的 `requirements/cuda.txt` 固定了 torch==2.11.0
   - 解决：注释掉 torch 行，使用 `--no-build-isolation --no-deps`

4. **`ModuleNotFoundError: No module named 'setuptools_scm'`**
   - 原因：使用 `--no-build-isolation` 时需要手动安装 build 依赖
   - 解决：`pip install setuptools-scm`

5. **Rust 编译失败**
   - 确保 `rustc >= 1.85`（vllm 0.24 需要较新的 Rust 工具链）
   - 更新 rust：`rustup update`

### 编译时间参考

- vLLM wheel 构建：5-15 分钟（取决于 CPU 核心数和网络）
- 主要耗时：Rust 依赖下载和编译、CUDA kernel 编译（`nvcc`）
- 使用 `MAX_JOBS=64` 可显著加速，但内存消耗更大
