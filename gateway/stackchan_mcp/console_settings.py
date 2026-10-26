"""Settings edited from the Stack-chan console, persisted as JSON.

Defaults come from the environment (``gateway/.env``) so nothing changes until
the user edits a value. ``update`` validates every field against ``SCHEMA``,
writes the file atomically and tells listeners which keys changed, so the
gateway can apply a change at runtime (stop the camera, rebuild the wake gate,
...) without a restart. Secrets never live here; API keys stay in ``.env``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Field:
    kind: str  # "bool" | "int" | "float" | "str" | "choice"
    default: Callable[[], Any]
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    max_len: int = 4000
    nullable: bool = False


SCHEMA: dict[str, Field] = {
    # Wake & listening
    "listen_mode": Field("choice", lambda: "wake_word", choices=("wake_word", "always", "off")),
    "kws_threshold": Field("float", lambda: _env_float("STACKCHAN_KWS_THRESHOLD", 0.05), 0.01, 0.6),
    "kws_score": Field("float", lambda: _env_float("STACKCHAN_KWS_SCORE", 7.0), 1.0, 20.0),
    "wake_idle_s": Field("float", lambda: _env_float("STACKCHAN_WAKE_IDLE_S", 30.0), 5.0, 300.0),
    "vad_silence_ms": Field("int", lambda: int(_env_float("STACKCHAN_ASR_VAD_SILENCE_MS", 1000)), 300, 3000),
    "turn_grace_ms": Field("int", lambda: int(_env_float("STACKCHAN_TURN_GRACE_MS", 900)), 0, 3000),
    # Camera & motion
    "face_tracking": Field("bool", lambda: _env_bool("STACKCHAN_FACE_TRACKER_AUTOSTART", True)),
    "head_follow": Field("bool", lambda: _env_bool("STACKCHAN_HEAD_FOLLOW_DEFAULT", True)),
    # Reply style
    # short: brief by default, long only on request; detailed: always full; brief: always one or two sentences.
    "reply_style": Field("choice", lambda: "short", choices=("short", "detailed", "brief")),
    "thinking": Field("choice", lambda: os.getenv("STACKCHAN_DEEPSEEK_THINKING", "auto").strip().lower() or "auto",
                      choices=("auto", "on", "off")),
    "robot_name": Field("str", lambda: os.getenv("STACKCHAN_ROBOT_NAME", "妮妮"), max_len=20),
    # Empty means "use the built-in persona".
    "persona": Field("str", lambda: "", max_len=4000),
    # Words the speech recogniser should expect (names, games, products), one string.
    "asr_hotwords": Field("str", lambda: os.getenv(
        "STACKCHAN_ASR_HOTWORDS", "英雄联盟 王者荣耀 羊了个羊 DeepSeek ElevenLabs Stack-chan Claude Mac"), max_len=1000),
    # Voice
    "tts_engine": Field("choice", lambda: "elevenlabs", choices=("elevenlabs", "edge")),
    "eleven_voice_id": Field("str", lambda: os.getenv("STACKCHAN_TTS_VOICE_ID", ""), max_len=64),
    # The Text to Dialogue WebSocket that eleven_v4_turbo needs accepts only
    # voice_settings.stability (first message) plus the language_code, seed and
    # apply_text_normalization query parameters. None = ElevenLabs default.
    "eleven_stability": Field("float", lambda: None, 0.0, 1.0, nullable=True),
    "eleven_language": Field("choice", lambda: "", choices=("", "zh", "en", "ja", "yue")),
    "eleven_seed": Field("int", lambda: None, 1, 4294967295, nullable=True),
    "eleven_text_normalization": Field("choice", lambda: "auto", choices=("auto", "on", "off")),
    "edge_voice": Field("str", lambda: os.getenv("STACKCHAN_EDGE_TTS_VOICE", "zh-CN-XiaoyiNeural"), max_len=64),
    # Provider adapters (backend only for now; not exposed in the console yet).
    "llm_provider": Field("choice", lambda: "deepseek", choices=("deepseek",)),
    "llm_model": Field("str", lambda: os.getenv("STACKCHAN_DEEPSEEK_MODEL", "deepseek-flash"), max_len=64),
    "asr_provider": Field("choice", lambda: "qwen_realtime", choices=("qwen_realtime",)),
    "tts_model": Field("str", lambda: os.getenv("STACKCHAN_TTS_MODEL", "eleven_v4_turbo"), max_len=64),
}


class SettingsError(ValueError):
    pass


def _coerce(key: str, field: Field, value: Any) -> Any:
    if value is None:
        if field.nullable:
            return None
        raise SettingsError(f"{key} 不能为空")
    try:
        if field.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if field.kind in ("int", "float"):
            number = int(value) if field.kind == "int" else float(value)
            if field.minimum is not None and number < field.minimum:
                raise SettingsError(f"{key} 不能小于 {field.minimum}")
            if field.maximum is not None and number > field.maximum:
                raise SettingsError(f"{key} 不能大于 {field.maximum}")
            return number
        if field.kind == "choice":
            text = str(value).strip()
            if text not in field.choices:
                raise SettingsError(f"{key} 只能是 {', '.join(field.choices)}")
            return text
        text = str(value)
        if len(text) > field.max_len:
            raise SettingsError(f"{key} 太长（最多 {field.max_len} 字）")
        return text.strip() if key != "persona" else text
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SettingsError):
            raise
        raise SettingsError(f"{key} 的值无效：{value!r}") from exc


class ConsoleSettings:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        # Only values the user set; everything else follows the env defaults live.
        self._explicit: dict[str, Any] = {}
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._load()

    def _load(self) -> None:
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning("console settings unreadable, using defaults: %s", exc)
            return
        for key, value in (saved or {}).items():
            field = SCHEMA.get(key)
            if field is None:
                continue
            try:
                self._explicit[key] = _coerce(key, field, value)
            except SettingsError as exc:
                logger.warning("ignoring saved console setting %s: %s", key, exc)

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._explicit:
                return self._explicit[key]
        return SCHEMA[key].default()

    def snapshot(self) -> dict[str, Any]:
        return {key: self.get(key) for key in SCHEMA}

    def add_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Validate, persist and broadcast ``changes``; returns the keys that changed."""
        coerced: dict[str, Any] = {}
        for key, value in changes.items():
            field = SCHEMA.get(key)
            if field is None:
                raise SettingsError(f"未知设置项：{key}")
            coerced[key] = _coerce(key, field, value)
        current = self.snapshot()
        changed = {k: v for k, v in coerced.items() if current.get(k) != v}
        with self._lock:
            # A value the user chose sticks even when it equals today's default,
            # so a later .env change cannot silently override it.
            newly_explicit = any(self._explicit.get(k, object()) != v for k, v in coerced.items())
            self._explicit.update(coerced)
            if newly_explicit:
                self._save_locked()
        if changed:
            logger.info("console settings changed: %s", ", ".join(f"{k}={v!r}"[:80] for k, v in changed.items()))
            for listener in list(self._listeners):
                try:
                    listener(changed)
                except Exception:
                    logger.exception("console settings listener failed")
        return changed

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Store only explicit values, so later .env edits still apply to the rest.
        data = dict(self._explicit)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)


_default: ConsoleSettings | None = None


def get_console_settings() -> ConsoleSettings:
    global _default
    if _default is None:
        raw = os.getenv("STACKCHAN_CONSOLE_SETTINGS", "").strip()
        path = Path(raw).expanduser() if raw else Path(__file__).resolve().parents[1] / "data" / "console-settings.json"
        _default = ConsoleSettings(path)
    return _default


def reset_console_settings_for_tests(path: Path | None = None) -> ConsoleSettings | None:
    """Tests: point the singleton at a temp file (or clear it)."""
    global _default
    _default = ConsoleSettings(path) if path is not None else None
    return _default
