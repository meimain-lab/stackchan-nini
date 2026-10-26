"""DeepSeek voice backend: Qwen ASR → DeepSeek → ElevenLabs on the device audio transport."""

from __future__ import annotations

import os

from .deepseek_pipeline_bridge import DeepSeekPipelineBridge
from .gemini_voice_proxy import GeminiVoiceProxy


class DeepSeekVoiceProxy(GeminiVoiceProxy):
    def _voice_api_key(self):
        return os.getenv("DEEPSEEK_API_KEY") or None

    def _voice_requires_key(self):
        return True

    def _voice_requires_device(self):
        return False

    def _voice_hello(self):
        return {"provider": "deepseek_pipeline"}

    def _speak_waits_for_first_audio(self):
        return True

    def _voice_tts_mode(self):
        return "native"

    def _make_bridge(self, esp32, **kwargs):
        return (self.bridge_factory or DeepSeekPipelineBridge)(
            esp32=esp32,
            tool_executor=self.voice_tool_executor,
            tool_executor_factory=self.voice_tool_executor_factory,
            tool_extensions=self.voice_tool_extensions,
            **kwargs,
        )
