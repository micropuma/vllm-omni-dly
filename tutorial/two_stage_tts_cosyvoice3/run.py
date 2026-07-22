"""
TTS 示例: Fun-CosyVoice3-0.5B (语音合成 + 声音克隆)
==================================================
使用 vLLM-Omni 的多阶段流水线运行 CosyVoice3 TTS 模型。
演示 vLLM-Omni 相比 vLLM 的核心能力: 多阶段 Orchestrator。

模型: FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (0.5B 参数)
流水线: Talker (AR LLM) → Code2Wav (Flow Matching)

注意: CosyVoice3 的原生拓扑只有这两个 stage，没有 Thinker。需要
语音理解 + Thinker → Talker → Code2Wav 三阶段流程时，请运行同级的
``three_stage_audio_omni`` 示例。
显存: ~4 GB

用法:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=3 python tutorial/two_stage_tts_cosyvoice3/run.py
"""

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

# === cuDNN 修复: 确保子进程找到正确版本的 cuDNN ===
# PyTorch 2.13+cu132 依赖 cuDNN 9.20，但系统可能加载旧版本。
# 此处预先将 nvidia-cudnn-cu13 包的 lib 目录加到 LD_LIBRARY_PATH。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENV_ROOT = str(_REPO_ROOT)
_CUDNN_LIB = os.path.join(_VENV_ROOT, ".venv", "lib", "python3.12", "site-packages", "nvidia", "cudnn", "lib")
if os.path.isdir(_CUDNN_LIB):
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    if _CUDNN_LIB not in _existing:
        os.environ["LD_LIBRARY_PATH"] = f"{_CUDNN_LIB}:{_existing}".strip(":")

# === 路径配置 ===
MODEL_PATH = str(_REPO_ROOT.parent / "models" / "Fun-CosyVoice3-0.5B-2512")
TOKENIZER_PATH = os.path.join(MODEL_PATH, "CosyVoice-BlankEN")
DEPLOY_CONFIG = os.path.join(os.path.dirname(__file__), "deploy.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
ZERO_SHOT_PROMPT_URL = (
    "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"
)


def ensure_ref_audio() -> str:
    """下载参考音频用于声音克隆."""
    dest = os.path.join(OUTPUT_DIR, "zero_shot_prompt.wav")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        print(f"下载参考音频到 {dest}...")
        urllib.request.urlretrieve(ZERO_SHOT_PROMPT_URL, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 TTS with voice cloning")
    parser.add_argument(
        "--text",
        type=str,
        default="你好，欢迎使用 vLLM Omni 的语音合成功能。这是一个支持声音克隆的多阶段推理系统。",
        help="要合成的文本",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM-Omni Tutorial: CosyVoice3 TTS (两阶段流水线)")
    print("=" * 60)

    # Step 1: 确保参考音频存在
    print("\n[1/4] 准备参考音频...")
    ref_audio_path = ensure_ref_audio()

    # Step 2: 初始化 Omni 引擎 (多阶段流水线)
    print(f"\n[2/4] 初始化 Omni 多阶段引擎...")
    print(f"  模型: {MODEL_PATH}")
    print(f"  部署配置: {DEPLOY_CONFIG}")
    print(f"  流水线: Talker (AR LLM) → Code2Wav (Flow Matching)")
    omni = Omni(
        model=MODEL_PATH,
        deploy_config=DEPLOY_CONFIG,
        tokenizer=TOKENIZER_PATH,
        log_stats=True,
    )

    # Step 3: 准备输入
    print(f"\n[3/4] 准备合成: {args.text}")
    audio_signal, sr = load_audio(ref_audio_path, sr=None)
    audio_data = (audio_signal.astype(np.float32), sr)

    prompts = {
        "prompt": args.text,
        "multi_modal_data": {"audio": audio_data},
        "mm_processor_kwargs": {
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后，能够做的比我还好呦!",
            "sample_rate": sr,
        },
    }

    # 计算 min/max token 长度
    sampling_cfg = {"top_p": 0.8, "top_k": 25, "eos_token_id": 6561 + 1}
    config = CosyVoice3Config()
    tokenizer = get_qwen_tokenizer(
        token_path=TOKENIZER_PATH,
        skip_special_tokens=config.skip_special_tokens,
        version=config.version,
    )
    _, text_token_len = extract_text_token(args.text, tokenizer, config.allowed_special)
    base_len = int(text_token_len)
    min_len = int(base_len * config.min_token_text_ratio)
    max_len = int(base_len * config.max_token_text_ratio)

    # 每个 stage 的 SamplingParams
    gpt_sampling = SamplingParams(
        temperature=1.0,
        top_p=sampling_cfg["top_p"],
        top_k=sampling_cfg["top_k"],
        repetition_penalty=2.0,
        min_tokens=min_len,
        max_tokens=max_len,
        stop_token_ids=[sampling_cfg["eos_token_id"]],
        detokenize=False,
    )
    s2mel_sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=2.0,
        max_tokens=256,
        detokenize=False,
    )

    # Step 4: 生成语音
    print(f"\n[4/4] 生成语音...")
    outputs = list(omni.generate(prompts, sampling_params_list=[gpt_sampling, s2mel_sampling]))

    for i, output in enumerate(outputs):
        ro = output.request_output
        if ro is None:
            print("未获取到 request_output。")
            continue

        mm = getattr(ro, "multimodal_output", None)
        if not mm and ro.outputs:
            mm = getattr(ro.outputs[0], "multimodal_output", None)

        if mm and "audio" in mm:
            audio_out = mm["audio"]
            print(f"  音频张量形状: {audio_out.shape}")
            out_path = os.path.join(OUTPUT_DIR, f"cosyvoice3_output_{i}.wav")
            sf.write(out_path, audio_out.cpu().numpy().squeeze(), 24000)
            print(f"  语音已保存到: {out_path}")
        else:
            print("未获取到音频输出。")

    omni.close()
    print("\n" + "=" * 60)
    print("TTS 推理完成! CosyVoice3 两阶段流水线运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    main()
