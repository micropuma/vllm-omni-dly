# BAGEL 单卡图生图

该示例运行 BAGEL 的原生两阶段拓扑，但两段均放在同一张 GPU：

```text
输入图片 + 编辑指令 -> Thinker (AR) -> DiT (diffusion) -> 输出图片
```

默认参数是仓库性能用例覆盖的单卡路径：512x512、20 diffusion steps、
`cfg_parallel_size=1`。

```bash
CUDA_VISIBLE_DEVICES=0 python tutorial/single_gpu_bagel_img2img/run.py
```

替换输入图片与编辑指令：

```bash
CUDA_VISIBLE_DEVICES=0 python tutorial/single_gpu_bagel_img2img/run.py \
  --image /path/to/input.jpg \
  --prompt "将这张照片变成中国水墨画，保留主体构图" \
  --output tutorial/single_gpu_bagel_img2img/output/result.png
```

`vllm_omni/deploy/bagel.yaml` 已将 `Thinker` 与 `DiT` 的 `devices` 都设为
`"0"`。BAGEL 的 512x512 图生图单卡性能用例峰值约 33GB，建议使用至少
40GB 显存的单张 GPU；24GB 卡可加 `--cpu-offload` 或
`--layerwise-offload` 尝试，但不属于已验证配置。

模型首次运行会下载 `ByteDance-Seed/BAGEL-7B-MoT`，也可通过 `--model`
指定本地检查点。
