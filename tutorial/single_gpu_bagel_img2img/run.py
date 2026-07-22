"""Single-GPU, two-stage BAGEL image-to-image tutorial.

The default BAGEL topology keeps both stages on one GPU:

    Thinker (AR multimodal planning) -> DiT (diffusion image generation)

The bundled vLLM-Omni performance configuration exercises this exact 512x512
image-to-image path on one H100. Run from the repository root:

    CUDA_VISIBLE_DEVICES=0 python tutorial/single_gpu_bagel_img2img/run.py
"""

import argparse
import copy
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    get_extra_body_params,
    get_model_class_name,
    should_init_extra_args_for_non_diffusion_stages,
)
from vllm_omni.platforms import current_omni_platform


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
DEFAULT_DEPLOY_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "bagel.yaml"
DEFAULT_IMAGE = EXAMPLE_DIR / "assets" / "airplane.jpeg"


def clone_sampling_params(params: object) -> object:
    if hasattr(params, "clone"):
        try:
            return params.clone()
        except Exception:
            pass
    return copy.deepcopy(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BAGEL single-GPU, two-stage image editing")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="BAGEL model ID or local checkpoint path")
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input JPG/PNG image")
    parser.add_argument(
        "--prompt",
        default="Turn this airplane photo into a detailed watercolor travel poster at sunset.",
        help="Instruction for the image edit",
    )
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality, watermark")
    parser.add_argument("--output", type=Path, default=EXAMPLE_DIR / "output" / "edited_airplane.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.5)
    parser.add_argument("--cpu-offload", action="store_true", help="Trade speed for lower GPU memory use")
    parser.add_argument("--layerwise-offload", action="store_true", help="Offload DiT blocks between operations")
    return parser.parse_args()


def build_sampling_params(omni: Omni, args: argparse.Namespace, model_class_name: str | None) -> list:
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    diffusion_params = OmniDiffusionSamplingParams(
        generator=generator,
        true_cfg_scale=args.cfg_text_scale,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
    )
    extra_args = {
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "negative_prompt": args.negative_prompt,
    }
    apply_declared_extra_args(diffusion_params, get_extra_body_params(model_class_name), extra_args)

    params_list = [clone_sampling_params(params) for params in omni.default_sampling_params_list]
    if len(params_list) != 2:
        raise RuntimeError(f"Expected two BAGEL stages, got {len(params_list)}.")

    diffusion_replaced = False
    init_non_diffusion = should_init_extra_args_for_non_diffusion_stages(model_class_name)
    for index, params in enumerate(params_list):
        if isinstance(params, OmniDiffusionSamplingParams):
            merged_extra = dict(getattr(params, "extra_args", {}) or {})
            merged_extra.update(diffusion_params.extra_args)
            diffusion_params.extra_args = merged_extra
            params_list[index] = diffusion_params
            diffusion_replaced = True
        elif init_non_diffusion and hasattr(params, "extra_args") and params.extra_args is None:
            params.extra_args = {}

    if not diffusion_replaced:
        raise RuntimeError("The BAGEL deployment did not expose a diffusion stage.")
    return params_list


def extract_image(outputs: list) -> Image.Image:
    for output in outputs:
        images = getattr(output, "images", None)
        if images:
            return images[0]
        request_output = getattr(output, "request_output", None)
        images = getattr(request_output, "images", None) if request_output is not None else None
        if images:
            return images[0]
    raise RuntimeError("BAGEL returned no image output.")


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")
    if not args.deploy_config.is_file():
        raise FileNotFoundError(f"Deploy config does not exist: {args.deploy_config}")
    if args.width <= 0 or args.height <= 0 or args.steps <= 0:
        raise ValueError("--width, --height, and --steps must be positive.")

    input_image = Image.open(args.image).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("vLLM-Omni Tutorial: BAGEL single-GPU image-to-image")
    print("=" * 72)
    print(f"Input image: {args.image}")
    print("Pipeline: GPU 0 / Thinker (AR) -> DiT (diffusion)")
    print(f"Output: {args.output}")

    omni = Omni(
        model=args.model,
        deploy_config=str(args.deploy_config),
        enable_cpu_offload=args.cpu_offload,
        enable_layerwise_offload=args.layerwise_offload,
    )
    try:
        if omni.num_stages != 2:
            raise RuntimeError(f"Expected Thinker -> DiT, got {omni.num_stages} stages.")
        model_class_name = get_model_class_name(omni)
        if model_class_name != "BagelPipeline":
            raise RuntimeError(f"Expected BagelPipeline, got {model_class_name!r}.")

        prompt = build_image_to_image_prompt(
            model_class_name=model_class_name,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            input_image=input_image,
            height=args.height,
            width=args.width,
        )
        sampling_params_list = build_sampling_params(omni, args, model_class_name)
        image = extract_image(omni.generate(prompt, sampling_params_list=sampling_params_list))
        image.save(args.output)
        print("Verified stages: 0=Thinker, 1=DiT, both on GPU 0")
        print(f"Edited image saved to: {args.output}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
