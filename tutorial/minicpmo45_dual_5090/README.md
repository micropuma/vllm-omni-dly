# MiniCPM-o 4.5 on RTX 5090 x2

这是在 OpenBayes 双 RTX 5090（每张 32 GiB）上实际跑通 MiniCPM-o 4.5 的记录。

    GPU 0: Thinker，多模态理解与文本生成
    GPU 1: Talker，语音 token -> Token2Wav，24 kHz waveform

vLLM-Omni 日志将前两部分显示为两个 vLLM engine stage；Token2Wav 以内嵌组件运行在 Stage 1。完整输出链路：

    text / image / audio -> Thinker -> Talker -> Token2Wav -> WAV

- [BUILD_FROM_SOURCE.md](BUILD_FROM_SOURCE.md)：源码编译 vLLM v0.25.0 和安装当前 vLLM-Omni。
- [RUN_AND_TEST.md](RUN_AND_TEST.md)：下载模型、启动双卡服务、验证文本与 TTS。

| 项目 | 已验证值 |
| --- | --- |
| GPU | RTX 5090 x2，32 GiB / GPU |
| CUDA | 13.0 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| vLLM | v0.25.0，源码编译 |
| 模型 | ModelScope OpenBMB/MiniCPM-o-4_5 |

验证请求生成了 24 kHz、单声道、242,880 samples（10.12 秒）的 WAV。远端产物：

    /openbayes/home/workspace/logs/minicpmo45-tts-response.json
    /openbayes/home/workspace/logs/minicpmo45-tts.wav

MiniCPM-o 4.5 是多模态理解与语音生成模型，不是图生图模型。图生图请使用 [single_gpu_bagel_img2img](../single_gpu_bagel_img2img/README.md)。
