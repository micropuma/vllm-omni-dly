"""
客户端调用示例: 通过 OpenAI 兼容 API 调用 Qwen2.5-VL-3B 服务
============================================================
先启动服务 (bash tutorial/serve.sh)，再运行此脚本。

用法:
    source .venv/bin/activate
    python tutorial/client_chat.py
"""

import base64
import os
from openai import OpenAI

# === 配置 ===
API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"


def encode_image(image_path: str) -> str:
    """将本地图片编码为 base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat_text():
    """纯文本对话."""
    client = OpenAI(base_url=API_BASE, api_key="not-needed")
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "用一句话介绍人工智能。"}],
        max_tokens=128,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def chat_vision(image_path: str):
    """多模态视觉对话 (文本 + 图片)."""
    client = OpenAI(base_url=API_BASE, api_key="not-needed")

    # 编码图片
    image_b64 = encode_image(image_path)
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请用中文详细描述这张图片。"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        max_tokens=256,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def main():
    print("=" * 60)
    print("vLLM-Omni Tutorial: OpenAI API 客户端")
    print("=" * 60)

    # 测试 1: 纯文本
    print("\n[1] 纯文本对话...")
    try:
        reply = chat_text()
        print(f"    Q: 用一句话介绍人工智能。")
        print(f"    A: {reply}")
    except Exception as e:
        print(f"    [错误] {e}")
        print("    => 请确认服务已启动: bash tutorial/serve.sh")

    # 测试 2: 图片理解
    print("\n[2] 图片理解...")
    image_path = os.path.join(os.path.dirname(__file__), "assets", "test_image.jpg")
    if os.path.exists(image_path):
        try:
            reply = chat_vision(image_path)
            print(f"    Q: 描述这张图片。")
            print(f"    A: {reply}")
        except Exception as e:
            print(f"    [错误] {e}")
    else:
        print(f"    => 请先运行 offline_inference.py 生成测试图片")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
