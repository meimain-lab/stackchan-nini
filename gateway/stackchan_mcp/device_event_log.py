"""JSONL log of every frame the gateway sends to the robot.

Enabled by ``STACKCHAN_DEVICE_EVENT_LOG=/path/to/device-events.jsonl``; a no-op
otherwise. JSON frames (head, led, llm emotion, tts state, ...) are written
whole, one object per line. Binary Opus frames would be one line per 60 ms,
so they are summed into one ``audio`` record per burst, closed by the next
JSON frame or after a short gap. Readers such as the monitor page tail the file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OPUS_FRAME_MS = 60
AUDIO_BURST_GAP_S = 1.0
# Face tracking sends ~10 head frames a second; keep the file bounded.
MAX_LOG_BYTES = 20 * 1024 * 1024


class DeviceEventLog:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._burst: dict[str, Any] | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def record(self, payload: bytes | str) -> None:
        if self._path is None:
            return
        now = time.time()
        try:
            if isinstance(payload, (bytes, bytearray)):
                self._add_audio(len(payload), now)
                return
            self._close_burst()
            try:
                message: Any = json.loads(payload)
            except ValueError:
                message = {"raw": payload[:500]}
            self._write({"ts": now, "dir": "out", "kind": "json", "message": message})
        except Exception as exc:  # logging must never break the audio path
            logger.debug("device event log failed: %s", exc)

    def _add_audio(self, size: int, now: float) -> None:
        burst = self._burst
        if burst is not None and now - burst["last"] > AUDIO_BURST_GAP_S:
            self._close_burst()
            burst = None
        if burst is None:
            self._burst = burst = {"start": now, "last": now, "frames": 0, "bytes": 0}
        burst["frames"] += 1
        burst["bytes"] += size
        burst["last"] = now

    def _close_burst(self) -> None:
        burst, self._burst = self._burst, None
        if burst is None:
            return
        self._write({
            "ts": burst["start"], "dir": "out", "kind": "audio",
            "frames": burst["frames"], "bytes": burst["bytes"],
            "audio_ms": burst["frames"] * OPUS_FRAME_MS,
            "wall_ms": round((burst["last"] - burst["start"]) * 1000),
        })

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            try:
                if self._path.stat().st_size > MAX_LOG_BYTES:
                    self._path.replace(self._path.with_name(self._path.name + ".1"))
            except FileNotFoundError:
                pass
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


_default: DeviceEventLog | None = None


def get_device_event_log() -> DeviceEventLog:
    global _default
    if _default is None:
        raw = os.getenv("STACKCHAN_DEVICE_EVENT_LOG", "").strip()
        _default = DeviceEventLog(Path(raw).expanduser() if raw else None)
    return _default


def logged_sender(
    send: Callable[[bytes | str], Awaitable[None]],
    log: DeviceEventLog | None = None,
) -> Callable[[bytes | str], Awaitable[None]]:
    """Wrap a websocket ``send`` so each outgoing frame is also logged."""
    log = log or get_device_event_log()
    if not log.enabled:
        return send

    async def send_and_log(payload: bytes | str) -> None:
        await send(payload)
        log.record(payload)

    return send_and_log
