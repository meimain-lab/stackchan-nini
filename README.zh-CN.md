[English](README.md) | 中文 | [日本語](README.ja.md)

> [!IMPORTANT]
> **仅限非商业使用。** 固件包含 Emotion Ball 的眼睛轮廓数据，该数据只允许非商业使用。你可以出于个人、教学等非商业目的使用、修改和分享本项目，但不能出售本项目或运行它的硬件。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md#emotion-ball-eye-ring-data-aora-bot-emotion-ball-community-license)。

![Groki Bot 像素风头图：CRT 复古电脑造型的桌面机器人，屏幕上是白色圆脸](docs/images/banner.png)
<!-- demo-video
     演示视频后补。发布后把嵌入代码（iframe 或 video）放在本注释块下方。
-->

# Groki Bot

Groki Bot 是一台开源桌面机器人（M5Stack CoreS3 加 Stack-chan 底座），接上 Grok Bot 应用就能和它直接对话；固件、电脑端网关和 Grok Bot 联动都在这一个仓库里。

## 特性

- 原生语音（仅 macOS）：直接和 Grok Bot 应用里的智能体语音交谈，复用 Mac 上应用的现有登录状态，不用另配 xAI 语音 API 密钥。
- 免触碰语音唤醒：随时呼叫「Hi Grok」即可开启对话，由电脑端网关轻量本地算法实时检测唤醒词。
- 动态人脸追踪：让机器人转头注视你的位置，结合 Mac 摄像头识别人脸并实时驱动双轴舵机底座转向。
- 灵动表情反馈：机器人自带表情引擎，有 15 种表情，会眨眼。
- 网页一键刷机与 OTA：免去本地搭建交叉编译环境，通过 Chrome 或 Edge 浏览器网页即可一键烧录固件，并支持 Wi-Fi 设备端 OTA 升级。

## 快速开始：通过 Grok Bot 原生语音对话

1. 烧录固件与配置网络：在桌面端 Chrome 或 Edge 打开网页刷机工具烧录固件，并在配网页面完成 Wi-Fi 连接。
2. 安装 Mac 网关依赖：在已登录 Grok Bot 桌面应用的 Mac 上克隆本仓库，进入 `gateway/` 目录同步依赖。
   ```sh
   cd gateway && uv sync --all-extras
   ```
3. 切换到 Grok 语音：默认后端是 Gemini，需要在 `gateway/.env` 里写 `STACKCHAN_VOICE_BACKEND=gbot` 和你的 agent 名字、ID；保持 Grok Bot 桌面应用已登录，另开终端常驻运行登录桥，启动网关，并在蓝牙设置页把机器人指向电脑的网关地址。
4. 呼叫唤醒词开始对话：对机器人说出 "Hi Grok" 即可开始语音交流，详细的环境变量配置、登录桥常驻命令与切回方法请参阅[配置 Grok Bot 原生语音](docs/quick-start.zh-CN.md#配置-grok-bot-原生语音)。

## 文档

- [快速开始与配置](docs/quick-start.zh-CN.md)
- [其他语音后端](docs/voice-backends.zh-CN.md)
- [功能](docs/features.zh-CN.md)
- [硬件](docs/hardware.zh-CN.md)
- [从源码编译](docs/build.zh-CN.md)
- [常见问题](docs/troubleshooting.zh-CN.md)
- [架构](docs/architecture.zh-CN.md)
- [仓库结构](docs/repository.zh-CN.md)
- [网关](gateway/README.zh-CN.md)

## 授权

[BSL-1.0](LICENSE) | [Gateway (MIT)](gateway/LICENSE) | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | [docs/licenses.html](docs/licenses.html) | [repository](docs/repository.zh-CN.md)

第三方许可和署名见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
