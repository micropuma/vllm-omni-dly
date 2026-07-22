# Run And Test MiniCPM-o 4.5

本页在双 RTX 5090 上用本地模型 /models/MiniCPM-o-4_5 启动服务，验证文本与语音输出。先完成 [BUILD_FROM_SOURCE.md](BUILD_FROM_SOURCE.md)。

## 1. 用 ModelScope 下载模型

```bash
cd /openbayes/home/workspace/vllm-omni-dly
source .venv/bin/activate
mkdir -p /models
uv pip install modelscope==1.38.1

python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download(
    "OpenBMB/MiniCPM-o-4_5",
    local_dir="/models/MiniCPM-o-4_5",
))
PY
```

下载完成前会出现 .incomplete 文件。所有文件完成改名后才可启动：

```bash
ls -lh /models/MiniCPM-o-4_5/model-*.safetensors
find /models/MiniCPM-o-4_5 -name '*.incomplete'
```

预期有四个 model-0000x-of-00004.safetensors，第二条命令没有输出。

## 2. 32 GiB 卡的显存限制

模型原始配置公布 40960 上下文。它在启动期为多模态 encoder/KV cache 做大规模预分配，单张 32 GiB 5090 无法稳定启动。已验证的覆盖配置：

```json
{
  "0": {
    "max_model_len": 2048,
    "max_num_batched_tokens": 2048,
    "limit_mm_per_prompt": {"video": 0, "image": 1}
  }
}
```

该配置保留单图与音频输入，禁用最大视频输入预分配。它适合交互演示，不适合 40k context 或视频推理。

## 3. 启动双卡服务

```bash
cd /openbayes/home/workspace/vllm-omni-dly
mkdir -p /openbayes/home/workspace/logs

HTTP_PROXY=http://alchemist-experience:7890 \
HTTPS_PROXY=http://alchemist-experience:7890 \
http_proxy=http://alchemist-experience:7890 \
https_proxy=http://alchemist-experience:7890 \
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nohup .venv/bin/python -m vllm_omni.entrypoints.cli.main serve \
  /models/MiniCPM-o-4_5 \
  --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --stage-overrides '{"0":{"max_model_len":2048,"max_num_batched_tokens":2048,"limit_mm_per_prompt":{"video":0,"image":1}}}' \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8099 \
  > /openbayes/home/workspace/logs/minicpmo45.log 2>&1 &
```

```bash
curl -i http://127.0.0.1:8099/health
tail -f /openbayes/home/workspace/logs/minicpmo45.log
```

成功日志包含：

    Stage 0 initialized
    Loaded Token2wav from /models/MiniCPM-o-4_5/assets/token2wav
    Stage 1 initialized
    Starting vLLM API server 0 on http://0.0.0.0:8099

显存参考：GPU 0 Thinker 约 23.2 GiB；GPU 1 Talker + Token2Wav 约 5.6 GiB。

## 4. 文本请求

```bash
curl -sS http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "用一句中文介绍你自己。"}],
    "modalities": ["text"],
    "max_tokens": 64
  }'
```

## 5. 文字加语音端到端请求

chat_template_kwargs 必须处于 JSON 根层级，不能放在 extra_body 中，否则不会启用 MiniCPM-o 的 TTS 模板。

```bash
curl -sS http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "请用一句中文介绍你自己。"}],
    "modalities": ["text", "audio"],
    "max_tokens": 64,
    "chat_template_kwargs": {"use_tts_template": true}
  }' > /openbayes/home/workspace/logs/minicpmo45-tts-response.json
```

成功响应含文本 choice 和 message.audio.data choice；audio.data 是 WAV 的 base64 内容。提取音频：

```bash
python - <<'PY'
import base64
import json

response_path = "/openbayes/home/workspace/logs/minicpmo45-tts-response.json"
output_path = "/openbayes/home/workspace/logs/minicpmo45-tts.wav"
raw = open(response_path).read()
response = json.loads(raw[raw.index("{"):])
audio = next(
    c["message"]["audio"]["data"]
    for c in response["choices"]
    if c["message"].get("audio")
)
open(output_path, "wb").write(base64.b64decode(audio))
print(output_path)
PY
```

本机实测：

    sample rate: 24000 Hz
    channels: 1
    samples: 242880
    duration: 10.12 s
    OmniTiming: Stage 0 = 0.73 s, Stage 1 = 4.80 s

这证明请求不仅经过 Thinker，也执行了 Talker 和 Token2Wav。

停止服务：

```bash
pkill -f '[v]llm_omni.entrypoints.cli.main serve /models/MiniCPM-o-4_5'
```
