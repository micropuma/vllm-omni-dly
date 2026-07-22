# vLLM-Omni Tutorial

每个 tutorial 都放在独立目录中，命令可以直接从仓库根目录执行。

| 目录 | 模型 | 输入 / 输出 | 展示重点 |
| --- | --- | --- | --- |
| [`vlm_chat/`](vlm_chat/) | Qwen2.5-VL-3B-Instruct | 文本、图片 -> 文本 | vLLM 原生单阶段 VLM 与 OpenAI API |
| [`two_stage_tts_cosyvoice3/`](two_stage_tts_cosyvoice3/) | Fun-CosyVoice3-0.5B-2512 | 文本、参考语音 -> 语音 | CosyVoice3 声音克隆的两阶段 TTS |
| [`three_stage_audio_omni/`](three_stage_audio_omni/) | Qwen3-Omni-30B-A3B-Instruct | 真实 WAV、文本 -> 文本、语音 | Thinker -> Talker -> Code2Wav 三阶段 Omni 推理 |
| [`single_gpu_bagel_img2img/`](single_gpu_bagel_img2img/) | BAGEL-7B-MoT | 图片、编辑指令 -> 图片 | 单张 GPU 上的 Thinker -> DiT 两阶段图生图 |

| [minicpmo45_dual_5090/](minicpmo45_dual_5090/) | MiniCPM-o-4_5 | 文本、图片或音频 -> 文本、语音 | RTX 5090 x2 的源码构建与 Thinker -> Talker -> Token2Wav |

## 环境

```bash
source .venv/bin/activate
```

## 1. 图片问答

```bash
python tutorial/vlm_chat/offline_inference.py
```

在线 API：

```bash
bash tutorial/vlm_chat/serve.sh
python tutorial/vlm_chat/client_chat.py
```

## 2. CosyVoice3 两阶段 TTS

```bash
CUDA_VISIBLE_DEVICES=3 python tutorial/two_stage_tts_cosyvoice3/run.py \
  --text "你好，欢迎使用语音合成系统。"
```

CosyVoice3 的模型拓扑固定为：

```text
Talker (AR LLM) -> Code2Wav (Flow Matching)
```

它没有 Thinker stage，因此这个示例不将两阶段 TTS 称为三阶段 Omni。
输出写入 `tutorial/two_stage_tts_cosyvoice3/output/`。

## 3. 真实语音输入的三阶段 Omni

```bash
CUDA_VISIBLE_DEVICES=0,1 python tutorial/three_stage_audio_omni/audio_to_audio.py
```

示例默认使用 `three_stage_audio_omni/assets/input.wav`。这是随示例提供的真实语音 WAV；可以替换为任意本地 WAV、MP3 或 FLAC：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tutorial/three_stage_audio_omni/audio_to_audio.py \
  --audio /path/to/your.wav \
  --question "请转写这段语音，并用中文语音简短回答。"
```

此示例显式校验 `Omni.num_stages == 3`，并请求两类最终结果：

```text
audio input -> Thinker (语音理解、文本回答) -> Talker (语音 token) -> Code2Wav (24 kHz WAV)
```

Thinker 的文本与 Code2Wav 的 WAV 都会写入 `tutorial/three_stage_audio_omni/output/`。默认采用仓库内 `vllm_omni/deploy/qwen3_omni_moe.yaml`，该配置为 Qwen3-Omni 的三阶段部署，默认需要两张足够显存的 GPU；可通过 `--model` 与 `--deploy-config` 替换为本地模型和对应部署配置。

## 4. 单卡多阶段图生图

```bash
CUDA_VISIBLE_DEVICES=0 python tutorial/single_gpu_bagel_img2img/run.py
```

BAGEL 的两个 stage 都位于 `GPU 0`：

```text
input image + instruction -> Thinker (AR) -> DiT (diffusion) -> output image
```

默认以 512x512、20 steps 运行，并使用仓库已覆盖的单卡图生图拓扑。此路径的峰值显存约 33GB，建议单张 40GB 及以上显卡；24GB 显卡可以使用 `--cpu-offload` 或 `--layerwise-offload` 试运行，但不保证成功。详见 [单卡 BAGEL 图生图](single_gpu_bagel_img2img/README.md)。

## vLLM 与 vLLM-Omni

```text
from vllm import LLM          # 单阶段 AR 推理，例如 Qwen2.5-VL
from vllm_omni import Omni    # 跨阶段编排，例如 TTS 与 Omni 音频生成
```
