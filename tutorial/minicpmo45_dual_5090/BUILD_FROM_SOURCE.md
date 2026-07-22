# Build vLLM-Omni From Source

本记录使用官方兼容组合：vLLM v0.25.0 + 当前 vLLM-Omni 源码，面向 RTX 5090 x2 / CUDA 13.0。它取代旧 BUILD.md 中 v0.24.0 / RTX 3090 的环境记录。

## 前提

    GPU: RTX 5090 x2
    CUDA toolkit: 13.0, CUDA_HOME=/usr/local/cuda
    Python: 3.12
    workspace: /openbayes/home/workspace

```bash
nvidia-smi
nvcc --version
uv --version
```

在 OpenBayes 访问 PyPI、GitHub 或 ModelScope 时设置代理：

```bash
export HTTP_PROXY=http://alchemist-experience:7890
export HTTPS_PROXY=http://alchemist-experience:7890
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
```

## 1. 创建虚拟环境和依赖基线

```bash
cd /openbayes/home/workspace
git clone https://github.com/vllm-project/vllm-omni.git vllm-omni-dly
cd vllm-omni-dly

uv venv --python 3.12 --seed
source .venv/bin/activate

# 先取得 v0.25.0 对齐的 PyTorch 和 Python 依赖。
# 下一节用源码构建版本覆盖 vLLM 本身。
uv pip install vllm==0.25.0 --torch-backend=auto
```

克隆并固定 vLLM 源码。不要使用 v0.24.0：本次验证的 Omni 源码和官方安装路线以 v0.25.0 为基线。

```bash
cd /openbayes/home/workspace
git clone https://github.com/vllm-project/vllm.git
cd vllm
git fetch --tags
git checkout v0.25.0
```

## 2. 安装构建工具

```bash
cd /openbayes/home/workspace/vllm-omni-dly
source .venv/bin/activate
uv pip install setuptools-rust setuptools-scm cmake ninja packaging

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY
```

本机输出为 torch 2.11.0+cu130、CUDA 13.0 和 cuda available=True。

## 3. 编译并安装 vLLM 源码

```bash
cd /openbayes/home/workspace/vllm

CUDA_HOME=/usr/local/cuda \
MAX_JOBS=16 \
CMAKE_BUILD_PARALLEL_LEVEL=16 \
CARGO_BUILD_JOBS=16 \
uv pip install --no-build-isolation --no-deps --reinstall .
```

- --no-build-isolation：使用当前虚拟环境中的 PyTorch/CUDA 组合。
- --no-deps：保留 vLLM wheel 建立的依赖基线，不重新解析或降级 PyTorch。
- 三个 jobs 参数控制 NVCC、CMake 和 Cargo 并行度。

这台机器约 78 GiB RAM。实际测试中 jobs=64 被系统以 exit code 137 终止，原因是编译并发导致内存耗尽。jobs=16 已成功完成构建，是本环境的稳定值；内存更大的机器可从 24 或 32 逐步增加。

```bash
cd /openbayes/home/workspace/vllm-omni-dly
python - <<'PY'
import torch, vllm
print("vllm:", vllm.__version__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY
```

预期至少显示 vllm 0.25.0 和 cuda=True。

## 4. 安装当前 vLLM-Omni 源码

```bash
cd /openbayes/home/workspace/vllm-omni-dly
uv pip install -e '.[minicpmo,demo]'

python - <<'PY'
import vllm
import vllm_omni
from stepaudio2 import Token2wav
print("vllm:", vllm.__version__)
print("vllm_omni: OK")
print("Token2wav: OK")
PY
```

当前 vLLM-Omni 版本名为 0.1.dev...，会对 vLLM 0.25.0 输出 major/minor 名称不一致 warning。这是版本名策略 warning；下一页的双卡文本和 TTS 请求已在该组合上验证成功。

继续：[RUN_AND_TEST.md](RUN_AND_TEST.md)。
