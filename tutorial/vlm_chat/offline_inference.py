"""
离线推理示例: Qwen2.5-VL-3B-Instruct
====================================
使用 vLLM 的 LLM.chat() API 进行本地多模态推理。
vLLM-Omni 完全兼容 vLLM API。

用法:
    source .venv/bin/activate
    python tutorial/offline_inference.py
"""

import base64
import os
from io import BytesIO

import requests
from PIL import Image
from vllm import LLM, SamplingParams

# === 配置 ===
MODEL_PATH = "/mnt/home/douliyang/mlsys/vllm-omini/models/Qwen2.5-VL-3B-Instruct"
TEST_IMAGE = os.path.join(os.path.dirname(__file__), "assets", "test_image.jpg")


def encode_image(image_path: str) -> str:
    """将图片编码为 base64 data URI."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def ensure_test_image() -> str:
    """确保测试图片存在，优先下载，失败则生成简单测试图片."""
    if os.path.exists(TEST_IMAGE):
        return TEST_IMAGE

    # 尝试从网络下载
    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg"
        img = Image.open(BytesIO(requests.get(url, timeout=10).content))
        img.save(TEST_IMAGE)
        print(f"测试图片已下载到: {TEST_IMAGE}\n")
        return TEST_IMAGE
    except Exception:
        pass

    # Fallback: 本地生成简单测试图片
    print("下载图片失败，生成本地测试图片...")
    from PIL import ImageDraw as _ImageDraw
    img = Image.new("RGB", (640, 480), color=(135, 206, 235))
    draw = _ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], fill=(255, 0, 0))
    draw.ellipse([350, 150, 550, 350], fill=(0, 255, 0))
    draw.text((200, 350), "Test Image", fill=(0, 0, 0))
    img.save(TEST_IMAGE)
    print(f"本地测试图片已生成: {TEST_IMAGE}\n")
    return TEST_IMAGE


def main():
    print("=" * 60)
    print("vLLM-Omni Tutorial: Qwen2.5-VL-3B-Instruct")
    print("=" * 60)

    # Step 1: 加载模型
    print("\n[1/3] 加载模型...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.40,
        limit_mm_per_prompt={"image": 3},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    # Step 2: 纯文本对话
    print("\n[2/3] 纯文本对话...")
    outputs = llm.chat(
        messages=[
            {"role": "user", "content": "请用中文介绍一下你自己，你是哪个版本的模型？"}
        ],
        sampling_params=sampling_params,
    )
    print(f"    Q: 请用中文介绍一下你自己")
    print(f"    A: {outputs[0].outputs[0].text.strip()}")

    # Step 3: 多模态对话 (图片理解)
    print("\n[3/3] 多模态对话 (图片理解)...")
    ensure_test_image()

    outputs = llm.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用中文详细描述这张图片里的内容。"},
                    {"type": "image_url", "image_url": {"url": encode_image(TEST_IMAGE)}},
                ],
            }
        ],
        sampling_params=sampling_params,
    )
    print(f"    Q: 请用中文详细描述这张图片里的内容。")
    print(f"    A: {outputs[0].outputs[0].text.strip()}")

    print("\n" + "=" * 60)
    print("推理完成! Qwen2.5-VL-3B 运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    main()
