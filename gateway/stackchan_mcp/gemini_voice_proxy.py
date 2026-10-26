"""Gemini Live voice backend that replaces the xiaozhi CloudProxy.

The device-facing protocol is identical to CloudProxy on purpose:

  * Device → gateway: 16 kHz Opus binary frames (microphone capture).
  * Gateway → device: 24 kHz Opus binary frames bracketed by
    ``tts.start`` / ``tts.stop`` JSON, optionally preceded by
    ``llm.emotion`` and ``tts.sentence_start``.

What changes is the upstream: instead of forwarding everything to
``api.tenclass.net``, the proxy decodes the 16 kHz Opus into PCM, ships
it to Gemini Live, encodes the 24 kHz PCM reply back into Opus, and
streams it down to the device. Gemini's function calls are routed to
the same ESP32 manager / USB transport the rest of the gateway uses
through :class:`GeminiLiveBridge`.

This module owns no audio I/O — it sits between ``esp32_client``
(device WebSocket) and ``gemini_live_bridge`` (Gemini Live session).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import struct
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .cloud_proxy import CloudConnectionInfo
from .debug_status import DebugStatus, get_debug_status
from .device_emotion import (
    DEAD_LED_RGB,
    IDLE_EMOTION,
    LISTENING_EMOTION,
    VOICE_ERROR_EMOTION,
    VOICE_ERROR_FACE,
    WAKE_CONFIRM_EMOTION,
    face_to_device_emotion,
)
from .edge_tts_provider import EdgeTTSProvider
from .gemini_live_bridge import GeminiLiveBridge, manual_vad_enabled
from .opus_codec import StackchanOpusCodec
from .session_keepalive import SessionKeepalive, keepalive_interval_from_env
from .wake_gate import LISTENING_LED_RGB, WakeGate, create_wake_gate_from_env

logger = logging.getLogger(__name__)


SendToDevice = Callable[[str | bytes], Awaitable[None]]


# Marker stamped into ``server_hello`` so other components (e.g. orchestrator
# debug logs) can tell which backend is live without sniffing class names.
GEMINI_HELLO_MARKER = {"provider": "gemini_live"}

WAKE_CHIME_DURATION_MS = 180
WAKE_CHIME_FREQUENCY_HZ = 880.0
WAKE_CHIME_SAMPLE_RATE = 24_000


def synthesize_wake_chime_pcm(
    *,
    sample_rate: int = WAKE_CHIME_SAMPLE_RATE,
    duration_ms: int = WAKE_CHIME_DURATION_MS,
    frequency_hz: float = WAKE_CHIME_FREQUENCY_HZ,
) -> bytes:
    """Return a short 24 kHz int16 mono beep for wake confirmation."""
    sample_count = max(1, int(sample_rate * duration_ms / 1000))
    fade = max(1, int(sample_rate * 0.01))
    samples: list[int] = []
    for index in range(sample_count):
        value = math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
        if index < fade:
            value *= index / fade
        elif index >= sample_count - fade:
            value *= (sample_count - index) / fade
        samples.append(int(max(-32767, min(32767, value * 0.35 * 32767))))
    return struct.pack(f"<{sample_count}h", *samples)


@dataclass
class GeminiVoiceProxy:
    """Replace xiaozhi cloud with a Gemini Live session.

    Exposes the same surface as :class:`stackchan_mcp.cloud_proxy.CloudProxy`
    so ``esp32_client`` can swap backends through the ``cloud_proxy_factory``
    seam without learning about Gemini.
    """

    esp32_ref: Callable[[], Any] | None = None
    voice_bridge: Any | None = None
    usb_transport: Any | None = None
    bridge_factory: Callable[..., GeminiLiveBridge] | None = None
    voice_tool_executor: Any = None
    voice_tool_executor_factory: Any = None
    voice_tool_extensions: tuple = ()
    codec_factory: Callable[[], StackchanOpusCodec] | None = None
    edge_tts_provider_factory: Callable[..., EdgeTTSProvider] | None = None
    wake_gate_factory: Callable[[], WakeGate | None] | None = None
    on_head_command: Callable[[], None] | None = None
    on_device_state: Callable[[str], None] | None = None
    set_face_tracking: Callable[[bool], Awaitable[dict[str, Any]]] | None = None
    debug_status: DebugStatus | None = None

    info: CloudConnectionInfo | None = None
    server_hello: dict[str, Any] | None = None

    _bridge: GeminiLiveBridge | None = None
    _codec: StackchanOpusCodec | None = None
    _edge_tts: EdgeTTSProvider | None = None
    _wake_gate: WakeGate | None = None
    _wake_gate_failures: int = 0
    _wake_gate_permanently_failed: bool = False
    _wake_gate_next_rebuild_at: float = 0.0
    _wake_gate_rebuild_max: int = 3
    _keepalive: SessionKeepalive | None = None
    _send_to_device: SendToDevice | None = None
    _device_hello: dict[str, Any] | None = None
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _tts_active: bool = False
    _tts_generation: int = 0
    _speak_text_active: bool = False
    _speak_text_frames: int = 0
    _tts_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    _ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    _stopped: bool = False
    _reported_dead: bool = False
    # TTS could not be reached; the red x-eyes face stays up until speech works or a wake.
    _voice_error: bool = False
    # Firmware applies tts.start on its main task. A tiny guard delay keeps the
    # first binary Opus frame from racing in while the device is still Listening.
    device_tts_start_delay_s: float = field(
        default_factory=lambda: max(
            0.0,
            float(os.getenv("STACKCHAN_GEMINI_TTS_START_DELAY_SECONDS", "0.08")),
        )
    )
    wake_confirm_hold_s: float = field(
        default_factory=lambda: max(
            0.0,
            float(os.getenv("STACKCHAN_WAKE_CONFIRM_HOLD_S", "0.3")),
        )
    )
    device_frame_interval_s: float = field(
        default_factory=lambda: max(
            0.0,
            float(os.getenv("STACKCHAN_GEMINI_DEVICE_FRAME_INTERVAL_SECONDS", "0.0")),
        )
    )
    # Gemini streams 24 kHz PCM in bursty chunks separated by jitter. Encoding
    # and forwarding each chunk straight to the device drains its playback
    # buffer the moment a gap shows up. Hold the first burst until we have
    # this many ms of PCM, then release in one shot — gives the firmware a
    # cushion before the next burst arrives. Set to 0 to disable.
    device_prebuffer_ms: float = field(
        default_factory=lambda: max(
            0.0,
            float(os.getenv("STACKCHAN_GEMINI_PREBUFFER_MS", "200")),
        )
    )
    _pcm_prebuffer: bytearray = field(default_factory=bytearray)
    _prebuffer_satisfied: bool = False
    # Even past the prebuffer threshold, Gemini keeps streaming PCM in bursty
    # gulps. Feeding the device buffer at Gemini's irregular pace makes the
    # speaker underrun mid-sentence. A drain task pulls encoded frames off
    # a queue at the same cadence as the 60 ms frame length so the firmware
    # sees an even rate regardless of upstream jitter.
    device_frame_pace_ms: float = field(
        default_factory=lambda: max(
            0.0,
            float(os.getenv("STACKCHAN_GEMINI_FRAME_PACE_MS", "60")),
        )
    )
    # Network-jitter cushion: at the very start of each turn, fire the first
    # ~burst_ms of frames back-to-back (skipping the pace metronome) so the
    # firmware decode queue fills fast instead of staying starved at ~one frame
    # ahead. After the burst the drain settles back into device_frame_pace_ms.
    # Set to 0 to disable. Capped to 500 ms of frames inside _drain so a long
    # reply can't overflow the firmware's ~40-frame decode queue.
    device_burst_ms: float = field(
        default_factory=lambda: max(
            0.0,
            float(os.getenv("STACKCHAN_GEMINI_BURST_MS", "400")),
        )
    )
    _frame_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _drain_task: asyncio.Task | None = None
    _drain_stop: asyncio.Event = field(default_factory=asyncio.Event)
    # Per-turn upstream receive stats: how Gemini actually delivered audio.
    # Logged at end_tts — separates "upstream generated haltingly" (nothing
    # downstream can fix) from "delivery pacing" (the drain's job).
    _turn_rx_first: float | None = None
    _turn_rx_last: float | None = None
    _turn_rx_bytes: int = 0
    _turn_rx_stalls: int = 0
    _turn_rx_max_gap: float = 0.0

    @property
    def _status(self) -> DebugStatus:
        return self.debug_status or get_debug_status()

    @property
    def connected(self) -> bool:
        """True when Gemini Live has handshook and a device socket is wired.

        During a reconnect (Gemini Live session naturally hit its 15 min cap)
        bridge.running stays True but the inner session is gone. Check
        bridge._connected_event so callers don't try to send audio into
        a half-open channel.
        """
        bridge = self._bridge
        if not self._accepts_device_audio():
            return False
        if not bridge.running:
            return False
        connected_event = getattr(bridge, "_connected_event", None)
        if connected_event is not None and not connected_event.is_set():
            return False
        return True

    def _accepts_device_audio(self) -> bool:
        """Device mic frames stay accepted while the proxy is wired.

        Live reconnect clears ``connected``, but gated audio must still reach
        the session cache instead of being dropped at this entry.
        """
        if self._stopped:
            return False
        return (
            self._bridge is not None
            and self._codec is not None
            and self._send_to_device is not None
            and self._ready_event.is_set()
        )

    @property
    def running(self) -> bool:
        bridge = self._bridge
        return bridge is not None and bridge.running

    def _voice_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def _voice_requires_key(self) -> bool:
        return True

    def _voice_requires_device(self) -> bool:
        return True

    def _voice_hello(self) -> dict[str, Any]:
        return dict(GEMINI_HELLO_MARKER)

    def _voice_tts_mode(self) -> str:
        return os.getenv("STACKCHAN_GEMINI_TTS", "native").strip().lower()

    def _make_bridge(self, esp32: Any, **kwargs: Any) -> Any:
        return (self.bridge_factory or GeminiLiveBridge)(esp32, **kwargs)

    def _speak_waits_for_first_audio(self) -> bool:
        return False

    async def _send_text_prompt(self, prompt: str) -> None:
        bridge = self._bridge
        injector = getattr(bridge, "inject_user_text", None)
        if injector is not None:
            if await injector(prompt) is False:
                raise RuntimeError("voice text channel unavailable")
            return
        session = getattr(bridge, "_session", None)
        if session is None:
            raise RuntimeError("voice text channel unavailable")
        await session.send_realtime_input(text=prompt)

    async def start(
        self,
        info: CloudConnectionInfo,
        device_hello: dict[str, Any],
        send_to_device: SendToDevice,
    ) -> bool:
        """Open the Gemini Live session for this device connection.

        Returns False (without raising) when no API key is configured so the
        device can still get the local MCP loop. The xiaozhi backend has the
        same "best-effort upstream" contract.
        """
        self.info = info
        self._send_to_device = send_to_device
        self._device_hello = device_hello
        self._stopped = False

        api_key = self._voice_api_key()
        if self._voice_requires_key() and not api_key:
            logger.warning(
                "Gemini voice backend disabled: GEMINI_API_KEY / GOOGLE_API_KEY not set; "
                "set STACKCHAN_VOICE_BACKEND=xiaozhi to use the xiaozhi cloud instead",
            )
            self._ready_event.set()  # unblock waiters with connected=False
            return False

        esp32 = self.esp32_ref() if self.esp32_ref is not None else None
        if esp32 is None and self._voice_requires_device():
            logger.error(
                "Gemini voice backend: no esp32 reference; refusing to start (function "
                "calling has nothing to dispatch to)",
            )
            self._ready_event.set()
            return False

        self._codec = (self.codec_factory or StackchanOpusCodec)()
        edge_factory = self.edge_tts_provider_factory or EdgeTTSProvider
        self._edge_tts = edge_factory(
            codec=self._codec,
            send_to_device=send_to_device,
            send_lock=self._send_lock,
            tts_start_delay_s=self.device_tts_start_delay_s,
            frame_interval_s=self.device_frame_interval_s,
            on_device_state=self.on_device_state,
        )
        # AUDIO path: Gemini emits native 24 kHz PCM, we Opus-encode straight
        # to the device. Bypasses edge-tts entirely (saves ~2-3 s of first-
        # phrase latency). We still ask the bridge for "TEXT" — internally
        # that means AUDIO + output_audio_transcription, which gives us
        # transcripts for logging at no latency cost. The transcription is
        # routed to _log_gemini_transcript (not _handle_gemini_text, which
        # would re-trigger edge-tts).
        #
        # edge-tts stays available as self._edge_tts for the MCP "say" tool
        # and as a deliberate fallback path.
        # STACKCHAN_GEMINI_TTS=edge is the bandwidth-emergency lever: Gemini
        # streams TEXT only (needs a 2.x Live model via STACKCHAN_GEMINI_MODEL,
        # 3.1 rejects TEXT) and edge-tts synthesizes locally-fetched audio.
        # Trades ~2-3 s of first-phrase latency for ~1000x less downstream
        # traffic on the Gemini socket — the native 24 kHz PCM stream needs
        # ~500 kbps, which a congested cross-border evening link cannot carry.
        tts_mode = self._voice_tts_mode()
        on_text = (
            self._handle_gemini_text
            if tts_mode == "edge"
            else self._log_gemini_transcript
        )
        if tts_mode == "edge":
            logger.info("Gemini TTS mode: edge (text-only Live + edge-tts)")
        self._wake_gate = (
            self.wake_gate_factory()
            if self.wake_gate_factory is not None
            else create_wake_gate_from_env()
        )
        gate = self._wake_gate
        if gate is not None:
            gate.is_tts_active = lambda: self._tts_active
        self._status.on_wake_gate_configured(
            available=bool(gate is not None and gate.available),
            state=gate.state.value if gate is not None else "DORMANT",
        )
        self._bridge = self._make_bridge(
            esp32,
            api_key=api_key,
            response_modality="TEXT",
            on_audio=self._handle_gemini_audio,
            on_text=on_text,
            on_turn_complete=self.end_tts,
            on_interrupt=self._abort_in_flight_tts,
            on_end_conversation=self._handle_bridge_end_conversation,
            on_session_dead=self._handle_session_dead,
            usb_transport=self.usb_transport,
            on_head_command=self.on_head_command,
            wake_gate_state_getter=self._wake_gate_state,
            set_face_tracking=self.set_face_tracking,
            debug_status=self._status,
        )
        if hasattr(self._bridge, "on_voice_error"):
            self._bridge.on_voice_error = self._handle_voice_error
        try:
            await self._bridge.start()
        except Exception as exc:
            logger.warning("Gemini Live bridge start failed: %s", exc)
            self._ready_event.set()
            return False

        ok = await self._bridge.wait_connected(timeout=10.0)
        if not ok:
            logger.warning("Gemini Live session did not connect within 10 s")
            self._ready_event.set()
            return False

        # Synthesize a hello-like surface so callers that inspect server_hello
        # can branch on provider. The xiaozhi backend gets this for free from
        # the cloud, we have to forge one here.
        self.server_hello = self._voice_hello()
        self._ready_event.set()
        self._start_keepalive()
        logger.info("Gemini voice backend ready (device=%s)", info.device_id or "?")
        return True

    def _start_keepalive(self) -> None:
        """Optional experimental silent-frame keepalive (off by default; see the session_keepalive module)."""
        gate = self._wake_gate
        if gate is None:
            self._status.on_keepalive_disabled("wake_gate_absent")
            return
        interval = keepalive_interval_from_env()
        if interval <= 0:
            self._status.on_keepalive_disabled("interval_disabled")
            return
        if not gate.available:
            self._status.on_keepalive_disabled("wake_gate_unavailable")
            return
        self._keepalive = SessionKeepalive(
            send_silence=self._send_keepalive_silence,
            should_send=self._keepalive_should_send,
            interval_s=interval,
            on_sent=self._status.on_keepalive_sent,
        )
        self._keepalive.start()
        if self._keepalive.running:
            self._status.on_keepalive_started()
        else:
            self._status.on_keepalive_disabled("not_started")

    def _wake_gate_state(self) -> str:
        gate = self._wake_gate
        if gate is None:
            return "DORMANT"
        return gate.state.value

    def _keepalive_should_send(self) -> bool:
        gate = self._wake_gate
        if gate is None:
            self._status.on_keepalive_skipped("wake_gate_absent")
            return False
        if not gate.available:
            self._status.on_keepalive_skipped("wake_gate_unavailable")
            return False
        if gate.is_listening:
            self._status.on_keepalive_skipped("listening")
            return False
        if not self.connected:
            self._status.on_keepalive_skipped("disconnected")
            return False
        return True

    async def _send_keepalive_silence(self, pcm: bytes) -> None:
        bridge = self._bridge
        if bridge is None:
            return
        sender = getattr(bridge, "send_keepalive_audio", None)
        if sender is None:
            await bridge.send_audio(pcm)
            return
        await sender(pcm)

    async def stop(self) -> None:
        self._stopped = True
        self._tts_generation += 1
        if self._keepalive is not None:
            await self._keepalive.stop()
            self._keepalive = None
            self._status.on_keepalive_stopped()
        await self._cancel_tts_tasks()
        await self._stop_drain_task(cancel=True)
        if self._bridge is not None:
            try:
                await self._bridge.stop()
            except Exception as exc:
                logger.warning("Gemini Live bridge stop failed: %s", exc)
        self._bridge = None
        self._codec = None
        self._edge_tts = None
        self._wake_gate = None
        self._ready_event.clear()

    async def ensure_connected(self) -> bool:
        if self._stopped:
            return False
        if self.connected:
            return True
        # Caller can re-issue start() if it kept the args. For now, just
        # report whatever state we landed in.
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            return False
        return self.connected

    # ---- ingress from device ------------------------------------------------

    async def send_device_json(self, message: dict[str, Any]) -> bool:
        """Handle a JSON envelope coming from the device.

        Most xiaozhi-flavoured JSON (``listen``, ``goodbye``, ``abort``) is
        useless to Gemini Live — VAD lives inside the API. The proxy still
        observes ``abort`` to clear any in-flight encoder buffer so the
        next utterance starts clean.
        """
        if not self.connected:
            return False
        msg_type = message.get("type", "")
        if msg_type == "abort":
            interrupt = getattr(self._bridge, "interrupt", None)
            if interrupt is not None:
                await interrupt()
            else:
                await self._abort_in_flight_tts()
        elif msg_type == "listen":
            state = message.get("state", "")
            if state == "stop":
                # Mark end-of-user-turn so Gemini commits the utterance.
                bridge = self._bridge
                if bridge is not None:
                    try:
                        await bridge.send_audio_stream_end()
                    except Exception as exc:
                        logger.debug("send_audio_stream_end ignored: %s", exc)
        return True

    async def send_device_binary(self, data: bytes) -> bool:
        """Forward one device Opus frame to Gemini Live as 16 kHz PCM."""
        if not self._accepts_device_audio() or self._bridge is None or self._codec is None:
            return False
        if not data:
            return True
        pcm = self._codec.decode_device_frame(data)
        if not pcm:
            return False
        try:
            await self._forward_gated_device_pcm(pcm)
            return True
        except Exception as exc:
            logger.warning("Gemini send_audio failed: %s", exc)
            return False

    def _configure_wake_gate(self, gate: WakeGate | None) -> None:
        if gate is not None:
            gate.is_tts_active = lambda: self._tts_active
        self._wake_gate = gate

    def _maybe_rebuild_wake_gate(self) -> WakeGate | None:
        if self._wake_gate_permanently_failed:
            return None
        if self._wake_gate is not None:
            return self._wake_gate
        now = time.monotonic()
        if now < self._wake_gate_next_rebuild_at:
            return None
        factory = self.wake_gate_factory or create_wake_gate_from_env
        try:
            gate = factory()
        except Exception as exc:
            self._wake_gate_failures += 1
            logger.warning("wake gate rebuild failed (%s): %s", self._wake_gate_failures, exc)
            if self._wake_gate_failures >= self._wake_gate_rebuild_max:
                self._wake_gate_permanently_failed = True
                self._status.on_wake_gate_unavailable()
                return None
            delay = min(30.0, 2.0 ** self._wake_gate_failures)
            self._wake_gate_next_rebuild_at = now + delay
            return None
        self._wake_gate_failures = 0
        self._configure_wake_gate(gate)
        self._status.on_wake_gate_configured(
            available=bool(gate is not None and gate.available),
            state=gate.state.value if gate is not None else "DORMANT",
        )
        return gate

    def _handle_wake_gate_process_error(self, exc: Exception) -> None:
        logger.warning("wake word gate failed; scheduling rebuild: %s", exc)
        self._wake_gate = None
        self._wake_gate_failures += 1
        if self._wake_gate_failures >= self._wake_gate_rebuild_max:
            self._wake_gate_permanently_failed = True
            self._status.on_wake_gate_unavailable()
            return
        delay = min(30.0, 2.0 ** self._wake_gate_failures)
        self._wake_gate_next_rebuild_at = time.monotonic() + delay

    async def _signal_manual_vad_start(self, bridge: GeminiLiveBridge) -> None:
        if not manual_vad_enabled():
            return
        sender = getattr(bridge, "send_activity_start", None)
        if callable(sender):
            try:
                await sender()
            except Exception as exc:
                logger.warning("manual VAD activity_start failed: %s", exc)

    async def _signal_manual_vad_end(self, bridge: GeminiLiveBridge) -> None:
        if not manual_vad_enabled():
            return
        sender = getattr(bridge, "send_activity_end", None)
        if callable(sender):
            try:
                await sender()
            except Exception as exc:
                logger.warning("manual VAD activity_end failed: %s", exc)

    def _listen_mode(self) -> str:
        from .console_settings import get_console_settings  # noqa: PLC0415

        return get_console_settings().get("listen_mode")

    async def _on_woke(self, bridge: Any) -> None:
        """Wake-up feedback: listening light, confirm face, chime."""
        self._reported_dead = False
        self._voice_error = False
        await self._signal_manual_vad_start(bridge)
        await self._set_leds_safe(*LISTENING_LED_RGB)
        await self._send_device_emotion_safe(WAKE_CONFIRM_EMOTION)
        if self.wake_confirm_hold_s > 0:
            await asyncio.sleep(self.wake_confirm_hold_s)
        await self._send_device_emotion_safe(LISTENING_EMOTION)
        self._signal_wake_chime()

    async def _on_wake_closed(self, bridge: Any) -> None:
        await self._signal_manual_vad_end(bridge)
        if self._voice_error:
            await self._set_leds_safe(*DEAD_LED_RGB)
            await self._send_device_emotion_safe(VOICE_ERROR_EMOTION)
        elif not self._tts_active:
            await self._set_leds_safe(0, 0, 0)
            await self._send_device_emotion_safe(IDLE_EMOTION)
        clear_context = getattr(bridge, "clear_conversation_context", None)
        if callable(clear_context):
            result = clear_context()
            if inspect.isawaitable(result):
                await result

    async def wake_now(self) -> bool:
        """Console "wake now": open the listening window as if the keyword was heard."""
        bridge = self._bridge
        if bridge is None or not self.connected:
            return False
        gate = self._maybe_rebuild_wake_gate()
        if gate is not None and gate.force_wake():
            self._status.on_wake_woke()
        await self._on_woke(bridge)
        return True

    async def close_listening(self) -> bool:
        """Close an open listening window (mode switched to off / always)."""
        gate = self._wake_gate
        bridge = self._bridge
        if gate is None or bridge is None or not gate.close():
            return False
        self._status.on_wake_closed()
        await self._on_wake_closed(bridge)
        return True

    def apply_console_settings(self, changed: dict[str, Any]) -> None:
        """Runtime-apply console changes that live on the wake gate."""
        if {"kws_threshold", "kws_score"} & changed.keys():
            # Rebuilt with the new sensitivity on the next audio frame.
            self._wake_gate = None
            self._wake_gate_failures = 0
            self._wake_gate_permanently_failed = False
            self._wake_gate_next_rebuild_at = 0.0
        if "wake_idle_s" in changed and self._wake_gate is not None:
            self._wake_gate.idle_s = float(changed["wake_idle_s"])
        if changed.get("listen_mode") in ("off", "always"):
            asyncio.get_running_loop().create_task(self.close_listening())

    async def _forward_gated_device_pcm(self, pcm: bytes) -> None:
        """Apply the wake gate before forwarding decoded device PCM to Gemini."""
        bridge = self._bridge
        if bridge is None:
            return
        mode = self._listen_mode()
        if mode == "off":
            # Console "off": not listening at all, not even for the wake word.
            return
        if mode == "always":
            # Console "always awake": every frame goes to speech recognition.
            await bridge.send_audio(pcm)
            return
        gate = self._maybe_rebuild_wake_gate()
        if gate is None:
            # While the gate is backing off a rebuild or has failed for good, pass the microphone through so the robot is never deaf.
            await bridge.send_audio(pcm)
            return
        try:
            decision = gate.process(pcm)
        except Exception as exc:
            self._handle_wake_gate_process_error(exc)
            await bridge.send_audio(pcm)
            return
        if decision.closed:
            self._status.on_wake_closed()
        if decision.woke:
            self._status.on_wake_woke()
        if decision.closed:
            await self._on_wake_closed(bridge)
        if decision.woke:
            await self._on_woke(bridge)
        for chunk in decision.forward_pcm:
            await bridge.send_audio(chunk)

    async def _handle_voice_error(self, failed: bool) -> None:
        """Voice output failed (or works again): show it on the face, never silently."""
        if failed == self._voice_error:
            return
        self._voice_error = failed
        if not failed:
            logger.info("voice output works again; clearing the voice-error face")
            return
        logger.warning("voice output unavailable; showing the voice-error face")
        await self._set_leds_safe(*DEAD_LED_RGB)
        await self._send_device_emotion_safe(VOICE_ERROR_EMOTION)

    async def _handle_session_dead(self) -> None:
        """Session reconnect exceeded the audio-cache TTL: close listening and report that the session could not be resumed."""
        self._reported_dead = True
        try:
            gate = self._wake_gate
            if gate is not None:
                closed = gate.close()
                if closed:
                    self._status.on_wake_closed()
        except Exception:
            logger.exception("session-dead wake gate close failed")
        try:
            await self._abort_in_flight_tts()
        except Exception:
            logger.exception("session-dead abort tts failed")
        try:
            await self._set_leds_safe(*DEAD_LED_RGB)
        except Exception:
            logger.exception("session-dead red LED failed")

    async def _handle_bridge_end_conversation(self) -> None:
        """Close the wake window when Gemini explicitly ends the conversation."""
        gate = self._wake_gate
        if gate is None:
            return
        closed = gate.close()
        if closed:
            self._status.on_wake_closed()
            bridge = self._bridge
            if bridge is not None:
                await self._signal_manual_vad_end(bridge)
            if not self._tts_active:
                await self._set_leds_safe(0, 0, 0)
                await self._send_device_emotion_safe(IDLE_EMOTION)

    async def try_handle_device_mcp(self, message: dict[str, Any]) -> bool:
        """Device MCP responses go back to the gateway, not to us.

        The xiaozhi backend remapped cloud-originated MCP request IDs so
        responses could be routed back upstream. With Gemini Live, function
        calls are issued by the gateway via :class:`GeminiLiveBridge`, which
        already owns the call/response loop — there's nothing to route here.
        """
        return False

    # ---- egress to device ---------------------------------------------------

    async def speak_text(
        self,
        text: str,
        *,
        session_id: str,
        prompt_audio_frames: list[bytes],
        emotion: str | None = None,
    ) -> dict[str, Any]:
        """Ask Gemini to speak ``text`` verbatim and stream the audio back.

        Same shape as CloudProxy.speak_text so the MCP ``say`` tool can call
        either backend through the same wrapper. ``prompt_audio_frames`` is
        ignored here — xiaozhi needed it for the listen/audio dance, Gemini
        accepts text directly.
        """
        if not self.connected or self._bridge is None:
            return {
                "ok": False,
                "text": text,
                "emotion": emotion,
                "frames_sent": 0,
                "provider": self._voice_hello()["provider"],
                "error": "gemini live is not connected",
            }

        # Optional emotion hint: drop the same {"type":"llm","emotion":...}
        # marker xiaozhi uses so the avatar reacts before the speech lands.
        if emotion:
            await self._send_device_emotion_safe(face_to_device_emotion(emotion))

        # Gemini 3.1 Flash Live rejects send_client_content for live updates
        # (only allowed for seeding initial history). Use send_realtime_input
        # which works on both 2.5 and 3.1. The "复述" prompt is a hint; the
        # model may rephrase, which is fine for a desktop pet — perfect
        # word-for-word TTS is what tts/voicevox is for.
        prompt = (
            "Say the following text aloud in your own voice, as written, briefly and naturally. "
            "Do not add anything and do not call tools:\n"
            + text
        )
        self._speak_text_active = True
        self._speak_text_frames = 0
        try:
            await self._send_text_prompt(prompt)
        except Exception as exc:
            self._speak_text_active = False
            return {
                "ok": False,
                "text": text,
                "emotion": emotion,
                "frames_sent": 0,
                "provider": self._voice_hello()["provider"],
                "error": f"gemini send failed: {exc}",
            }

        # The receive loop in GeminiLiveBridge will drive our on_audio
        # callback which writes the Opus frames to the device. Wait for
        # the tts.stop to flip the flag.
        timeout = float(os.getenv("GEMINI_SPEAK_TIMEOUT_SECONDS", "30"))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while (self._tts_active or self._codec_has_data() or
                   (self._speak_waits_for_first_audio() and self._speak_text_frames == 0)):
                if loop.time() > deadline:
                    break
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        finally:
            self._speak_text_active = False

        frames_sent = self._speak_text_frames
        return {
            "ok": frames_sent > 0,
            "text": text,
            "emotion": emotion,
            "frames_sent": frames_sent,
            "provider": self._voice_hello()["provider"],
            "error": None if frames_sent > 0 else "no audio frames delivered",
        }

    def _codec_has_data(self) -> bool:
        codec = self._codec
        return codec is not None and bool(codec._encode_buffer)

    async def _handle_gemini_text(self, text: str) -> None:
        """Legacy path: route Gemini text to edge-tts.

        Kept so the unit tests around edge-tts dispatch keep working and so
        the proxy can be switched back to text-only mode in an emergency by
        reverting the bridge_factory call to ``on_text=self._handle_gemini_text``.
        Not used by the default AUDIO path.
        """
        self._status.record_transcript(text)
        provider = self._edge_tts
        if provider is None:
            return

        async def speak_in_background() -> None:
            result = await provider.speak_text(text)
            if not result.get("ok"):
                logger.warning("edge-tts failed: %s", result.get("error"))

        task = asyncio.create_task(speak_in_background(), name="edge-tts-speak")
        self._tts_tasks.add(task)
        task.add_done_callback(self._on_tts_task_done)

    async def _log_gemini_transcript(self, text: str) -> None:
        """Default on_text in AUDIO mode: log the transcript, do not synthesize.

        Gemini's ``output_audio_transcription`` gives us a free running
        transcript of what the model just said in audio. Useful for debugging
        and observability — without it, a Gemini-side stutter or wrong word
        looks like a mysterious noise from the speaker. We never feed this
        text to a TTS engine in AUDIO mode; the speaker is already saying
        the words via native PCM.
        """
        if text:
            logger.info("gemini transcript: %s", text)
            self._status.record_transcript(text)

    def _on_tts_task_done(self, task: asyncio.Task[None]) -> None:
        self._tts_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("edge-tts background task failed")

    async def _cancel_tts_tasks(self) -> None:
        if not self._tts_tasks:
            return
        tasks = list(self._tts_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tts_tasks.difference_update(tasks)

    async def _stop_drain_task(self, cancel: bool = False) -> None:
        """Tear down the drain task and discard any leftover frames.

        ``cancel=True`` is the hard path used by abort/stop; ``cancel=False``
        just signals stop and waits (used when end_tts joined the queue).
        """
        task = self._drain_task
        self._drain_task = None
        if task is None or task.done():
            self._frame_queue = asyncio.Queue()
            self._drain_stop = asyncio.Event()
            return
        self._drain_stop.set()
        if cancel:
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception) as exc:
            logger.debug("drain task teardown: %s", exc)
        # Fresh queue + event so the next turn starts clean.
        self._frame_queue = asyncio.Queue()
        self._drain_stop = asyncio.Event()

    # ---- private: Gemini audio sink ----------------------------------------

    async def _handle_gemini_audio(self, pcm_24khz: bytes) -> None:
        """Called by GeminiLiveBridge.on_audio for every 24 kHz PCM chunk.

        Encodes into 60 ms Opus frames and bracket-streams them down the
        device WebSocket. Two pieces of state mean we can run end-to-end
        without a separate orchestrator:

          - First non-empty chunk emits ``tts.start`` + ``tts.sentence_start``.
          - When the codec drains and Gemini stops sending audio, the
            outer turn-complete handler emits ``tts.stop``.

        ``tts.stop`` itself is sent from :meth:`on_turn_complete` because
        Gemini's audio frames don't always land before turn_complete.
        """
        if (self._stopped or self._send_to_device is None
                or self._codec is None or not pcm_24khz):
            return
        now = asyncio.get_running_loop().time()
        if self._turn_rx_first is None:
            self._turn_rx_first = now
        elif self._turn_rx_last is not None:
            gap = now - self._turn_rx_last
            if gap > 0.24:  # device-side decode-queue tolerance
                self._turn_rx_stalls += 1
                self._turn_rx_max_gap = max(self._turn_rx_max_gap, gap)
        self._turn_rx_last = now
        self._turn_rx_bytes += len(pcm_24khz)
        # 24 kHz int16 mono = 48 bytes per ms.
        threshold = int(self.device_prebuffer_ms * 48)
        if threshold > 0 and not self._prebuffer_satisfied:
            self._pcm_prebuffer.extend(pcm_24khz)
            if len(self._pcm_prebuffer) < threshold:
                return
            burst = bytes(self._pcm_prebuffer)
            self._pcm_prebuffer.clear()
            self._prebuffer_satisfied = True
            frames = self._codec.encode_pcm_24k(burst)
        else:
            frames = self._codec.encode_pcm_24k(pcm_24khz)
        if not frames:
            return
        if not await self._queue_tts_frames(frames):
            return

    def _ensure_drain_task(self) -> None:
        """Spin up the paced drain task at the start of each TTS turn."""
        if self._drain_task is not None and not self._drain_task.done():
            return
        # Fresh stop event per turn; the queue is reused (it's been drained
        # to zero outstanding task_done by the previous end_tts).
        self._drain_stop = asyncio.Event()
        try:
            self._drain_task = asyncio.create_task(self._drain())
        except RuntimeError:
            # No running loop — would only happen in pathological teardown.
            self._drain_task = None

    async def _drain(self) -> None:
        """Ship queued Opus frames at a steady cadence.

        Runs from the first frame of a turn until end_tts signals stop AND
        the queue has been emptied. Sending exceptions are logged and the
        frame is dropped, but the loop keeps going so a single bad write
        doesn't strand the rest of the turn.
        """
        pace_s = max(self.device_frame_pace_ms / 1000.0, 0.0)
        loop = asyncio.get_running_loop()
        # Cushion-target pacing. The device-side buffer level is estimated
        # open-loop: cushion = audio-ms shipped − wall-clock elapsed since the
        # turn's first frame. Below target → send back-to-back (this covers
        # both the initial burst-preheat AND catch-up after a mid-turn Gemini
        # stall); at target → one frame per pace interval. The old metronome
        # (next_send_time += pace) could only ever fall behind: a source stall
        # of T ms permanently ate T ms of device cushion, so long replies got
        # progressively closer to underrun. Target is quantised to whole
        # frames to preserve the ceil(burst/pace) preheat contract, and capped
        # at 500 ms so a long reply can't overflow the firmware's ~40-frame
        # decode queue.
        target_ahead_s = pace_s
        if pace_s > 0 and self.device_burst_ms > 0:
            target_ahead_s = max(
                min(
                    math.ceil(self.device_burst_ms / self.device_frame_pace_ms),
                    math.ceil(500.0 / self.device_frame_pace_ms),
                )
                * pace_s,
                pace_s,
            )
        logger.info(
            "drain cushion target=%.0fms pace=%.0fms",
            target_ahead_s * 1000,
            pace_s * 1000,
        )
        turn_t0: float | None = None
        audio_sent_s = 0.0
        while True:
            if self._drain_stop.is_set() and self._frame_queue.empty():
                return
            try:
                frame = await asyncio.wait_for(
                    self._frame_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            if pace_s > 0:
                now = loop.time()
                if turn_t0 is None:
                    turn_t0 = now
                cushion_s = audio_sent_s - (now - turn_t0)
                if cushion_s < 0:
                    # Device already underran — the past can't be refilled.
                    # Re-anchor so the catch-up burst rebuilds from zero
                    # instead of dumping the whole deficit at once.
                    turn_t0 = now - audio_sent_s
                    cushion_s = 0.0
                if cushion_s + pace_s > target_ahead_s:
                    await asyncio.sleep(cushion_s + pace_s - target_ahead_s)
            try:
                if self._send_to_device is not None:
                    async with self._send_lock:
                        await self._send_to_device(frame)
                    if self._speak_text_active:
                        self._speak_text_frames += 1
            except Exception as exc:
                logger.warning("device send opus frame failed: %s", exc)
            finally:
                self._frame_queue.task_done()
            audio_sent_s += pace_s

    async def _queue_tts_frames(self, frames: list[bytes]) -> bool:
        """Start TTS if needed, honor the start-to-first-frame delay, then queue frames."""
        if not frames or self._stopped:
            return False
        generation = self._tts_generation
        async with self._send_lock:
            if generation != self._tts_generation:
                return False
            if not self._tts_active:
                if not await self._begin_tts():
                    return False
                if self.device_tts_start_delay_s > 0:
                    await asyncio.sleep(self.device_tts_start_delay_s)
        if generation != self._tts_generation or self._stopped:
            return False
        self._ensure_drain_task()
        for frame in frames:
            await self._frame_queue.put(frame)
        return True

    async def _begin_tts(self) -> bool:
        """Emit the start envelope just before the first Opus frame."""
        assert self._send_to_device is not None
        try:
            await self._send_to_device(json.dumps({"type": "tts", "state": "start"}))
            await self._send_to_device(
                json.dumps({"type": "tts", "state": "sentence_start", "text": ""})
            )
        except Exception as exc:
            logger.warning("tts.start envelope failed: %s", exc)
            return False
        self._tts_active = True
        self._status.on_tts_state(True)
        # Switch the avatar to a "talking" expression so the face doesn't freeze
        # while Gemini speaks (firmware stops the idle loop in Speaking state).
        # Fire-and-forget for the same reason as the LED dispatch below.
        self._signal_speaking_avatar(self._speaking_avatar_face())
        # Light the base ring soft blue while Gemini speaks. Fire-and-forget so
        # a slow / failing LED call never holds up the Opus stream.
        self._signal_speaking_leds(0, 80, 180)
        return True

    def _signal_wake_chime(self) -> None:
        """Play the wake beep without blocking the listening window."""
        try:
            task = asyncio.create_task(self._play_wake_chime())
        except RuntimeError:
            return
        self._tts_tasks.add(task)
        task.add_done_callback(self._tts_tasks.discard)

    async def _play_wake_chime(self) -> None:
        """Send a short beep through the existing TTS envelope and Opus frames.

        Failures are logged only. This path uses the same tts.start-to-first-frame
        delay as Gemini speech so firmware can leave Listening before audio lands.
        """
        if (
            self._tts_active
            or self._reported_dead
            or self._send_to_device is None
            or self._codec is None
        ):
            return
        try:
            await self._handle_gemini_audio(synthesize_wake_chime_pcm())
            await self.end_tts()
        except Exception as exc:
            logger.warning("wake chime failed: %s", exc)

    def _signal_speaking_leds(self, r: int, g: int, b: int) -> None:
        """Schedule a non-blocking LED update. Failures are swallowed."""
        try:
            task = asyncio.create_task(self._set_leds_safe(r, g, b))
        except RuntimeError:
            return  # no running loop — nothing we can do
        self._tts_tasks.add(task)
        task.add_done_callback(self._tts_tasks.discard)

    async def _set_leds_safe(self, r: int, g: int, b: int) -> None:
        """USB-first, WS-fallback LED dispatch. Never raises."""
        args = {"r": r, "g": g, "b": b}
        logger.info("status LED dispatch requested rgb=(%s,%s,%s)", r, g, b)
        usb = self.usb_transport
        if usb is not None and getattr(usb, "connected", False):
            logger.info("status LED dispatch branch=usb tool=self.led.set_all")
            try:
                await usb.call_tool("self.led.set_all", args)
                logger.info("status LED dispatch branch=usb ok")
                return
            except Exception as exc:
                logger.info(
                    "status LED dispatch branch=usb failed error=%s; trying WS",
                    exc,
                )
        elif usb is None:
            logger.info("status LED dispatch branch=usb skipped reason=unavailable")
        else:
            logger.info("status LED dispatch branch=usb skipped reason=disconnected")

        if self.esp32_ref is None:
            logger.info(
                "status LED dispatch branch=ws skipped reason=esp32_ref_missing"
            )
            return
        try:
            esp32 = self.esp32_ref()
        except Exception as exc:
            logger.info("status LED dispatch branch=ws failed error=%s", exc)
            return
        if esp32 is None:
            logger.info("status LED dispatch branch=ws skipped reason=esp32_ref_none")
            return
        if not getattr(esp32, "device_connected", False):
            logger.info(
                "status LED dispatch branch=ws skipped reason=device_disconnected"
            )
            return

        mcp_supported = self._esp32_mcp_supported(esp32)
        send_led = getattr(esp32, "send_led", None)
        if callable(send_led):
            logger.info(
                "status LED dispatch branch=ws_direct mcp_supported=%s",
                mcp_supported,
            )
            try:
                async with self._send_lock:
                    _result, error = await send_led(
                        r,
                        g,
                        b,
                        notify_activity=False,
                    )
            except Exception as exc:
                logger.info("status LED dispatch branch=ws_direct failed error=%s", exc)
            else:
                if not error:
                    logger.info("status LED dispatch branch=ws_direct ok")
                    return
                logger.info(
                    "status LED dispatch branch=ws_direct failed error=%s",
                    error,
                )
        else:
            logger.info(
                "status LED dispatch branch=ws_direct skipped reason=unavailable "
                "mcp_supported=%s",
                mcp_supported,
            )

        if not mcp_supported:
            logger.info(
                "status LED dispatch branch=ws_mcp skipped reason=mcp_unsupported"
            )
            return

        logger.info("status LED dispatch branch=ws_mcp tool=self.led.set_all")
        try:
            async with self._send_lock:
                await esp32.call_tool("self.led.set_all", args)
        except Exception as exc:
            logger.info("status LED dispatch branch=ws_mcp failed error=%s", exc)
        else:
            logger.info("status LED dispatch branch=ws_mcp ok")

    def _signal_speaking_avatar(self, face: str) -> None:
        """Schedule a non-blocking avatar update. Failures are swallowed."""
        try:
            task = asyncio.create_task(self._set_avatar_safe(face))
        except RuntimeError:
            return  # no running loop — nothing we can do
        self._tts_tasks.add(task)
        task.add_done_callback(self._tts_tasks.discard)

    async def _set_avatar_safe(self, face: str) -> None:
        """USB-first, WS-fallback avatar dispatch. Never raises."""
        args = {"face": face}
        usb = self.usb_transport
        if usb is not None and getattr(usb, "connected", False):
            try:
                await usb.call_tool("self.display.set_avatar", args)
                return
            except Exception as exc:
                logger.debug("avatar via USB failed: %s; trying WS", exc)
        esp32 = self.esp32_ref() if self.esp32_ref is not None else None
        if esp32 is not None and getattr(esp32, "device_connected", False):
            if not self._esp32_mcp_supported(esp32):
                await self._send_device_emotion_safe(face_to_device_emotion(face))
                return
            try:
                async with self._send_lock:
                    await esp32.call_tool("self.display.set_avatar", args)
            except Exception as exc:
                logger.debug("avatar via WS failed: %s", exc)

    @staticmethod
    def _esp32_mcp_supported(esp32: Any) -> bool:
        connection = getattr(esp32, "connection", None)
        if connection is None:
            return True
        return bool(getattr(connection, "mcp_supported", True))

    async def _send_device_emotion_safe(self, emotion: str) -> None:
        """Best-effort ``llm.emotion`` expression dispatch. Never raises."""
        esp32 = self.esp32_ref() if self.esp32_ref is not None else None
        if esp32 is not None and getattr(esp32, "device_connected", False):
            send_emotion = getattr(esp32, "send_emotion", None)
            if callable(send_emotion):
                try:
                    async with self._send_lock:
                        _result, error = await send_emotion(
                            emotion,
                            notify_activity=False,
                        )
                    if error:
                        logger.debug("emotion via ESP32 manager failed: %s", error)
                    else:
                        return
                except Exception as exc:
                    logger.debug("emotion via ESP32 manager failed: %s", exc)
        if self._send_to_device is None:
            return
        try:
            async with self._send_lock:
                await self._send_to_device(
                    json.dumps(
                        {"type": "llm", "emotion": emotion},
                        ensure_ascii=False,
                    )
                )
        except Exception as exc:
            logger.debug("emotion hint dropped: %s", exc)

    def _speaking_avatar_face(self) -> str:
        bridge = self._bridge
        face = getattr(bridge, "current_turn_emotion_face", None)
        return str(face or "happy")

    def _clear_turn_emotion_face(self) -> None:
        bridge = self._bridge
        if bridge is None:
            return
        clear = getattr(bridge, "clear_turn_emotion_face", None)
        if callable(clear):
            clear()
            return
        if hasattr(bridge, "current_turn_emotion_face"):
            setattr(bridge, "current_turn_emotion_face", None)

    async def end_tts(self) -> None:
        """Drain leftover PCM and send the closing ``tts.stop`` envelope.

        Public so the bridge / orchestrator can call it on turn_complete.
        Idempotent: a second call without intervening start is a no-op.
        """
        generation = self._tts_generation
        # Short turns may finish before the prebuffer threshold trips. Encode
        # whatever PCM we held back and push it onto the queue so the drain
        # task can ship it at the steady cadence.
        if self._pcm_prebuffer and self._codec is not None and self._send_to_device is not None:
            burst = bytes(self._pcm_prebuffer)
            self._pcm_prebuffer.clear()
            self._prebuffer_satisfied = True
            frames = self._codec.encode_pcm_24k(burst)
            if frames:
                await self._queue_tts_frames(frames)

        if self._turn_rx_first is not None and self._turn_rx_last is not None:
            audio_ms = self._turn_rx_bytes / 48.0
            wall_ms = (self._turn_rx_last - self._turn_rx_first) * 1000.0
            logger.info(
                "turn rx stats: audio=%.0fms wall=%.0fms stalls>240ms=%d max_gap=%.0fms",
                audio_ms,
                wall_ms,
                self._turn_rx_stalls,
                self._turn_rx_max_gap * 1000.0,
            )
        self._turn_rx_first = None
        self._turn_rx_last = None
        self._turn_rx_bytes = 0
        self._turn_rx_stalls = 0
        self._turn_rx_max_gap = 0.0

        if not self._tts_active:
            self._clear_turn_emotion_face()
            return

        # Push codec tail through the drain so it stays in cadence with the
        # rest of the turn instead of double-tapping the device.
        if self._codec is not None:
            tail = self._codec.flush()
            for frame in tail:
                await self._frame_queue.put(frame)

        # Wait for the drain to ship every queued frame, then stop it cleanly.
        if self._drain_task is not None and not self._drain_task.done():
            try:
                await self._frame_queue.join()
            except Exception as exc:
                logger.warning("frame queue join failed: %s", exc)
        if generation != self._tts_generation:
            return
        await self._stop_drain_task(cancel=False)

        async with self._send_lock:
            if generation != self._tts_generation:
                return
            try:
                await self._send_to_device(  # type: ignore[misc]
                    json.dumps({"type": "tts", "state": "stop"})
                )
            except Exception as exc:
                logger.warning("tts.stop envelope failed: %s", exc)
            self._tts_active = False
            self._status.on_tts_state(False)
            self._prebuffer_satisfied = False
            self._pcm_prebuffer.clear()
            self._signal_speaking_avatar(self._post_tts_face())
            self._signal_speaking_leds(*self._post_tts_led_rgb())
            self._clear_turn_emotion_face()

    async def _abort_in_flight_tts(self) -> None:
        """Drop in-flight TTS and tell the device to leave Speaking state.

        Without the final ``tts.stop`` envelope the firmware stays in
        ``kDeviceStateSpeaking`` until the next ``tts.start`` lands, which
        means the next utterance is dropped silently (the orchestrator
        notes this in ``audio_stream.send_tts_state``). Always pair the
        codec reset with a stop envelope so the device returns to idle.
        """
        self._tts_generation += 1
        was_active = self._tts_active
        if self._codec is not None:
            self._codec.reset_encode_buffer()
        await self._cancel_tts_tasks()
        await self._stop_drain_task(cancel=True)
        if self._edge_tts is not None:
            try:
                await self._edge_tts.abort()
            except Exception:
                logger.exception("edge-tts abort failed")
        self._tts_active = False
        self._status.on_tts_state(False)
        self._prebuffer_satisfied = False
        self._pcm_prebuffer.clear()
        if was_active and self._send_to_device is not None:
            try:
                async with self._send_lock:
                    await self._send_to_device(
                        json.dumps({"type": "tts", "state": "stop"})
                    )
            except Exception as exc:
                logger.debug("abort tts.stop drop: %s", exc)
            self._signal_speaking_leds(*self._post_tts_led_rgb())
        self._clear_turn_emotion_face()

    def _post_tts_face(self) -> str:
        """Back to the listening face if the wake window is still open."""
        if self._voice_error:
            return VOICE_ERROR_FACE
        gate = self._wake_gate
        if not self._reported_dead and gate is not None and gate.is_listening:
            return "listening"
        return "idle"

    def _post_tts_led_rgb(self) -> tuple[int, int, int]:
        if self._reported_dead or self._voice_error:
            return DEAD_LED_RGB
        gate = self._wake_gate
        if gate is not None and gate.is_listening:
            return LISTENING_LED_RGB
        return (0, 0, 0)
