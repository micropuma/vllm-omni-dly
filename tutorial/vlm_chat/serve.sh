#!/bin/bash
# 启动 Qwen2.5-VL-3B OpenAI 兼容 API 服务
# 用法: bash tutorial/serve.sh
#
# 服务启动后:
#   - API 地址:  http://localhost:8000/v1
#   - Swagger:   http://localhost:8000/docs
#   - 健康检查:  http://localhost:8000/health

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/.venv/bin/activate"

MODEL_PATH="/mnt/home/douliyang/mlsys/vllm-omini/models/Qwen2.5-VL-3B-Instruct"

echo "========================================="
echo "  vLLM-Omni: Qwen2.5-VL-3B API 服务"
echo "========================================="
echo ""
echo "  模型: $MODEL_PATH"
echo "  API:  http://localhost:8000/v1"
echo "  Docs: http://localhost:8000/docs"
echo ""

# 使用 vllm serve 命令 (vllm-omni 完全兼容)
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt image=3
