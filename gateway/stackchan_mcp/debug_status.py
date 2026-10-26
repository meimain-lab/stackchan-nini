"""Central gateway runtime-status state.

gateway / esp32_client / gemini_voice_proxy / gemini_live_bridge call the
event methods here at key moments; the capture server's ``GET /debug/status``
and ``GET /debug/panel`` only render a read-only snapshot and never guess.

Design constraints:

* Every method is synchronous and only called from the gateway's single event
  loop, so no locks are needed.
* Timestamps are ``time.time()`` (unix epoch seconds); the panel formats them
  as local time.
* The two ``recent`` queues keep the latest 10 entries.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RECENT_LIMIT = 10

_USAGE_COUNT_FIELDS = (
    "prompt_token_count",
    "cached_content_token_count",
    "response_token_count",
    "tool_use_prompt_token_count",
    "thoughts_token_count",
    "total_token_count",
)
_USAGE_DETAIL_FIELDS = (
    "prompt_tokens_details",
    "cache_tokens_details",
    "response_tokens_details",
    "tool_use_prompt_tokens_details",
)


def _field_value(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _enum_value(value: Any) -> Any:
    if value is None:
        return None
    return getattr(value, "value", value)


def _serialize_token_details(details: Any) -> list[dict[str, Any]]:
    if not details:
        return []
    out: list[dict[str, Any]] = []
    for item in details:
        out.append(
            {
                "modality": _enum_value(_field_value(item, "modality")),
                "token_count": _field_value(item, "token_count"),
            }
        )
    return out


def _serialize_usage_metadata(usage_metadata: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {
        name: _field_value(usage_metadata, name) for name in _USAGE_COUNT_FIELDS
    }
    for name in _USAGE_DETAIL_FIELDS:
        usage[name] = _serialize_token_details(_field_value(usage_metadata, name))
    usage["traffic_type"] = _enum_value(_field_value(usage_metadata, "traffic_type"))
    return usage


@dataclass
class DebugStatus:
    """Gateway-wide runtime status. Each field is written only by the event method of the component that owns it."""

    clock: Callable[[], float] = time.time

    # ---- device (ESP32 WebSocket connection)----
    device_connected_flag: bool = False
    device_id: str | None = None
    device_ip: str | None = None
    device_connected_since: float | None = None
    device_last_disconnect_at: float | None = None
    device_disconnect_count: int = 0

    # ---- gemini (Live session)----
    gemini_generation: int = 0
    gemini_running: bool = False
    gemini_connected_flag: bool = False
    gemini_session_count: int = 0
    gemini_connected_since: float | None = None
    gemini_last_error: dict[str, Any] | None = None
    gemini_reconnect_1008_count: int = 0
    gemini_reconnect_receive_stall_count: int = 0
    gemini_has_resumption_handle: bool = False
    gemini_last_reconnect_used_handle: bool | None = None
    gemini_resumption_handle_updated_at: float | None = None
    gemini_active_drops: int = 0
    gemini_last_active_drop_at: float | None = None
    gemini_keepalive_count: int = 0
    gemini_last_keepalive_at: float | None = None
    gemini_tool_call_cancel_count: int = 0
    gemini_last_tool_call_cancel_at: float | None = None
    gemini_keepalive_running: bool = False
    gemini_keepalive_disabled_reason: str | None = None
    gemini_last_keepalive_skip_reason: str | None = None
    gemini_last_keepalive_skip_at: float | None = None
    gemini_token_usage: dict[str, Any] | None = None

    # ---- wake_gate (wake word gate)----
    wake_available: bool = False
    wake_state: str = "DORMANT"
    wake_last_wake_at: float | None = None
    wake_count: int = 0
    wake_close_count: int = 0

    # ---- audio ----
    last_device_audio_at: float | None = None
    tts_active: bool = False
    _tts_sources: set[str] = field(default_factory=set, repr=False)

    # ---- face tracking ----
    # Set by the gateway; returns the "face_tracking" section of the snapshot.
    face_tracking_provider: Callable[[], dict[str, Any]] | None = field(default=None, repr=False)

    # ---- recent ----
    recent_tool_calls: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=RECENT_LIMIT)
    )
    recent_transcripts: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=RECENT_LIMIT)
    )

    # ---- device events ----

    def on_device_connected(self, device_id: str | None = None, ip: str | None = None) -> None:
        self.device_connected_flag = True
        self.device_id = device_id
        self.device_ip = ip
        self.device_connected_since = self.clock()

    def on_device_disconnected(self) -> None:
        self.device_connected_flag = False
        self.device_connected_since = None
        self.device_last_disconnect_at = self.clock()
        self.device_disconnect_count += 1

    # ---- gemini events ----

    def next_gemini_generation(self) -> int:
        """Allocate a generation id for a new Live bridge instance."""
        self.gemini_generation += 1
        return self.gemini_generation

    def _is_current_generation(self, generation: int | None) -> bool:
        return generation is None or generation == self.gemini_generation

    def on_gemini_started(self, generation: int | None = None) -> None:
        if not self._is_current_generation(generation):
            return
        self.gemini_running = True

    def on_gemini_connected(
        self,
        session_count: int,
        *,
        used_resumption_handle: bool = False,
        generation: int | None = None,
    ) -> None:
        if not self._is_current_generation(generation):
            return
        self.gemini_connected_flag = True
        self.gemini_session_count = session_count
        self.gemini_connected_since = self.clock()
        self.gemini_last_reconnect_used_handle = used_resumption_handle

    def on_resumption_handle_updated(self) -> None:
        self.gemini_has_resumption_handle = True
        self.gemini_resumption_handle_updated_at = self.clock()

    def on_resumption_handle_cleared(self) -> None:
        self.gemini_has_resumption_handle = False

    def on_gemini_receive_stall_reconnect(self) -> None:
        self.gemini_reconnect_receive_stall_count += 1

    def on_gemini_disconnected(
        self,
        *,
        error: str | None = None,
        code: int | None = None,
        was_connected: bool = True,
        generation: int | None = None,
    ) -> bool:
        """Record the end of a Gemini session; return whether it dropped an
        active conversation.

        Active means the wake gate is LISTENING or the device is playing TTS.
        Both signals are kept by their own components, so the decision is made
        here and callers need not fetch them.
        """
        now = self.clock()
        if not self._is_current_generation(generation):
            if code == 1008:
                self.gemini_reconnect_1008_count += 1
            return False
        self.gemini_connected_flag = False
        self.gemini_connected_since = None
        if error:
            self.gemini_last_error = {"message": error, "at": now}
        if code == 1008:
            self.gemini_reconnect_1008_count += 1
        active = was_connected and (self.wake_state == "LISTENING" or self.tts_active)
        if active:
            self.gemini_active_drops += 1
            self.gemini_last_active_drop_at = now
        return active

    def on_gemini_stopped(self, generation: int | None = None) -> None:
        if not self._is_current_generation(generation):
            return
        self.gemini_running = False
        self.gemini_connected_flag = False
        self.gemini_connected_since = None

    def on_keepalive_sent(self) -> None:
        self.gemini_keepalive_count += 1
        self.gemini_last_keepalive_at = self.clock()
        self.gemini_last_keepalive_skip_reason = None
        self.gemini_last_keepalive_skip_at = None

    def on_keepalive_started(self) -> None:
        self.gemini_keepalive_running = True
        self.gemini_keepalive_disabled_reason = None

    def on_keepalive_stopped(self) -> None:
        self.gemini_keepalive_running = False

    def on_keepalive_disabled(self, reason: str) -> None:
        self.gemini_keepalive_running = False
        self.gemini_keepalive_disabled_reason = reason

    def on_keepalive_skipped(self, reason: str) -> None:
        self.gemini_last_keepalive_skip_reason = reason
        self.gemini_last_keepalive_skip_at = self.clock()

    def record_usage_metadata(self, usage_metadata: Any) -> None:
        usage = _serialize_usage_metadata(usage_metadata)
        usage["updated_at"] = self.clock()
        self.gemini_token_usage = usage

    def on_tool_call_cancelled(self) -> None:
        self.gemini_tool_call_cancel_count += 1
        self.gemini_last_tool_call_cancel_at = self.clock()

    # ---- wake_gate events ----

    def on_wake_gate_configured(self, *, available: bool, state: str = "DORMANT") -> None:
        self.wake_available = available
        self.wake_state = state

    def on_wake_woke(self) -> None:
        self.wake_state = "LISTENING"
        self.wake_last_wake_at = self.clock()
        self.wake_count += 1

    def on_wake_closed(self) -> None:
        self.wake_state = "DORMANT"
        self.wake_close_count += 1

    def on_wake_gate_unavailable(self) -> None:
        self.wake_available = False
        self.wake_state = "UNAVAILABLE"

    # ---- audio events ----

    def on_device_audio(self) -> None:
        self.last_device_audio_at = self.clock()

    def on_tts_state(self, active: bool, *, source: str = "gemini") -> None:
        """Merge independent speakers so one backend cannot clear another's TTS."""
        if active:
            self._tts_sources.add(source)
        else:
            self._tts_sources.discard(source)
        self.tts_active = bool(self._tts_sources)

    # ---- recent events ----

    def record_tool_call(self, name: str, ok: bool, error: str | None = None) -> None:
        entry: dict[str, Any] = {"name": name, "ok": ok, "at": self.clock()}
        if error:
            entry["error"] = error
        self.recent_tool_calls.append(entry)

    def record_transcript(self, text: str) -> None:
        if not text:
            return
        self.recent_transcripts.append({"text": text, "at": self.clock()})

    # ---- snapshot ----

    def snapshot(self) -> dict[str, Any]:
        """Build the JSON for ``GET /debug/status``. recent lists newest first."""
        return {
            "generated_at": self.clock(),
            "device": {
                "connected": self.device_connected_flag,
                "device_id": self.device_id,
                # Peer address of the device WebSocket: its LAN IP on the current network.
                "ip": self.device_ip,
                "connected_since": self.device_connected_since,
                "last_disconnect_at": self.device_last_disconnect_at,
                "disconnect_count": self.device_disconnect_count,
            },
            "gemini": {
                "generation": self.gemini_generation,
                "running": self.gemini_running,
                "connected": self.gemini_connected_flag,
                "session_count": self.gemini_session_count,
                "connected_since": self.gemini_connected_since,
                "last_error": self.gemini_last_error,
                "reconnect_1008_count": self.gemini_reconnect_1008_count,
                "reconnect_receive_stall_count": self.gemini_reconnect_receive_stall_count,
                "has_resumption_handle": self.gemini_has_resumption_handle,
                "last_reconnect_used_handle": self.gemini_last_reconnect_used_handle,
                "resumption_handle_updated_at": self.gemini_resumption_handle_updated_at,
                "active_drops": self.gemini_active_drops,
                "last_active_drop_at": self.gemini_last_active_drop_at,
                "keepalive_count": self.gemini_keepalive_count,
                "last_keepalive_at": self.gemini_last_keepalive_at,
                "tool_call_cancel_count": self.gemini_tool_call_cancel_count,
                "last_tool_call_cancel_at": self.gemini_last_tool_call_cancel_at,
                "keepalive_running": self.gemini_keepalive_running,
                "keepalive_disabled_reason": self.gemini_keepalive_disabled_reason,
                "last_keepalive_skip_reason": self.gemini_last_keepalive_skip_reason,
                "last_keepalive_skip_at": self.gemini_last_keepalive_skip_at,
                "token_usage": self.gemini_token_usage,
            },
            "wake_gate": {
                "state": self.wake_state,
                "available": self.wake_available,
                "last_wake_at": self.wake_last_wake_at,
                "wake_count": self.wake_count,
                "close_count": self.wake_close_count,
            },
            "audio": {
                "last_device_audio_at": self.last_device_audio_at,
                "tts_active": self.tts_active,
            },
            "face_tracking": (
                self.face_tracking_provider() if self.face_tracking_provider is not None else None
            ),
            "recent": {
                "tool_calls": list(reversed(self.recent_tool_calls)),
                "transcripts": list(reversed(self.recent_transcripts)),
            },
        }


_status: DebugStatus | None = None


def get_debug_status() -> DebugStatus:
    """Process-wide singleton. Tests can inject their own DebugStatus instance."""
    global _status
    if _status is None:
        _status = DebugStatus()
    return _status


def reset_debug_status() -> None:
    """For tests: drop the singleton; the next get creates a new one."""
    global _status
    _status = None
