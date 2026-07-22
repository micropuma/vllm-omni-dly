"""Three-stage audio-in, audio-out inference with Qwen3-Omni.

The request deliberately includes a real WAV input.  vLLM-Omni routes it
through the model's full topology:

    Thinker (audio understanding + text) -> Talker (speech tokens) -> Code2Wav

Run from the repository root:

    CUDA_VISIBLE_DEVICES=0,1 python tutorial/three_stage_audio_omni/audio_to_audio.py
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_AUDIO = EXAMPLE_DIR / "assets" / "input.wav"
DEFAULT_DEPLOY_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful Chinese voice assistant. Understand the user's audio, "
    "then answer concisely in Chinese."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Omni three-stage audio conversation")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen3-Omni model ID or local checkpoint path")
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=DEFAULT_DEPLOY_CONFIG,
        help="Three-stage deployment YAML (Thinker, Talker, Code2Wav)",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help="Input WAV/MP3/FLAC. Defaults to this example's real speech WAV.",
    )
    parser.add_argument(
        "--question",
        default="请听这段语音，先概括其内容，再用自然的中文语音回答。",
        help="Question for the Thinker stage.",
    )
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_DIR / "output")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    return parser.parse_args()


def build_prompt(question: str, audio_path: Path, sampling_rate: int) -> dict:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio input does not exist: {audio_path}")

    waveform, sample_rate = load_audio(str(audio_path), sr=sampling_rate)
    audio_data = (waveform.astype(np.float32), sample_rate)
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"audio": audio_data},
        # Request both final outputs: Thinker text and Code2Wav waveform.
        "modalities": ["text", "audio"],
    }


def sampling_params() -> list[SamplingParams]:
    return [
        SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=512,
            repetition_penalty=1.05,
            detokenize=True,
        ),
        SamplingParams(
            temperature=0.9,
            top_k=50,
            max_tokens=4096,
            repetition_penalty=1.05,
            stop_token_ids=[2150],
            detokenize=False,
        ),
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=65536,
            repetition_penalty=1.1,
            detokenize=True,
        ),
    ]


def audio_to_numpy(audio: object) -> np.ndarray:
    if isinstance(audio, list):
        return np.concatenate([audio_to_numpy(chunk) for chunk in audio])
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def main() -> None:
    args = parse_args()
    if not args.deploy_config.is_file():
        raise FileNotFoundError(f"Deploy config does not exist: {args.deploy_config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(args.question, args.audio, args.sampling_rate)

    print("=" * 72)
    print("vLLM-Omni Tutorial: Qwen3-Omni audio input / three-stage output")
    print("=" * 72)
    print(f"Input audio: {args.audio}")
    print("Pipeline: Thinker (understand) -> Talker (speech tokens) -> Code2Wav")

    omni = Omni(
        model=args.model,
        deploy_config=str(args.deploy_config),
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": 1},
        log_stats=True,
    )
    try:
        if omni.num_stages != 3:
            raise RuntimeError(
                f"Expected a three-stage Qwen3-Omni pipeline, got {omni.num_stages} stages. "
                f"Use {DEFAULT_DEPLOY_CONFIG}."
            )

        print("Verified stages: 0=Thinker, 1=Talker, 2=Code2Wav")
        text_seen = False
        audio_seen = False
        for stage_output in omni.generate(prompt, sampling_params_list=sampling_params()):
            request_output = stage_output.request_output
            if request_output is None or not request_output.outputs:
                continue

            if stage_output.final_output_type == "text":
                text = request_output.outputs[0].text.strip()
                text_path = args.output_dir / f"{request_output.request_id}.txt"
                text_path.write_text(text + "\n", encoding="utf-8")
                print(f"[Thinker] {text}")
                print(f"[Thinker] Text saved to: {text_path}")
                text_seen = True
            elif stage_output.final_output_type == "audio":
                multimodal = request_output.outputs[0].multimodal_output
                waveform = audio_to_numpy(multimodal["audio"])
                audio_path = args.output_dir / f"{request_output.request_id}.wav"
                sf.write(audio_path, waveform, samplerate=24000, format="WAV")
                print(f"[Code2Wav] Audio saved to: {audio_path}")
                audio_seen = True

        if not (text_seen and audio_seen):
            raise RuntimeError("The request did not return both Thinker text and Code2Wav audio outputs.")
        print("Completed: the real audio input traversed all three vLLM-Omni stages.")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
