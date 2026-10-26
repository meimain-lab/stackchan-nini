"""Wake word gate for the device microphone stream sent to Gemini.

The device streams 16 kHz PCM to the gateway continuously. Until the
configured wake word is detected, audio is only consumed locally; after a
wake the gate opens a bounded listening window and returns the PCM frames
that may be forwarded to Gemini Live.

Who may set the LEDs (together with gemini_live_bridge / gemini_voice_proxy):

* LISTENING: the proxy lights the listening cyan (0,180,180) on wake; the
  bridge's set_all_leds is deferred.
* TTS active: proxy._begin_tts lights the speaking blue (0,80,180); closing
  the gate must not turn it off; the bridge's set_all_leds is deferred.
* DORMANT: the gateway holds no status light, so idle behaviour and the
  model's set_all_leds may write normally.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# Default wake word "Hi Grok". sherpa-onnx does open-vocabulary keyword
# spotting: keywords are written in the phonemes of the model's tokens.txt, so
# changing the phrase needs no retraining. A few vowel variants cover accents.
WAKE_PHRASE = "hi grok"
WAKE_KEYWORD = "HH AY1 G R AA1 K @hi_grok"
WAKE_KEYWORD_VARIANTS = (
    "HH AY1 G R OW1 K @hi_grok_ow",
    "HH AY1 G R AO1 K @hi_grok_ao",
    "HH AY1 G R AH1 K @hi_grok_ah",
)
# Optional wake word "Hey Groki": set STACKCHAN_WAKE_PHRASE=hey groki. It has
# three vowel variants as well.
HEY_GROKI_WAKE_PHRASE = "hey groki"
HEY_GROKI_WAKE_KEYWORD = "HH EY1 G R OW1 K IY0 @hey_groki"
HEY_GROKI_WAKE_KEYWORD_VARIANTS = (
    "HH EY1 G R AA1 K IY0 @hey_groki_aa",
    "HH EY1 G R AO1 K IY0 @hey_groki_ao",
    "HH EY1 G R AH1 K IY0 @hey_groki_ah",
)
KWS_MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_PREROLL_S = 2.0
DEFAULT_WAKE_IDLE_S = 30.0
DEFAULT_WAKE_MAX_LISTENING_S = 45.0
DEFAULT_ACTIVITY_RMS = 500.0
DEFAULT_KWS_RESET_INTERVAL_S = 60.0
DEFAULT_KWS_RESET_DEFER_MAX_S = 15.0
DEFAULT_TTS_STUCK_MAX_S = 300.0
LISTENING_LED_RGB = (0, 180, 180)


class WakeGateState(str, Enum):
    """Gateway-side microphone gate state."""

    DORMANT = "DORMANT"
    LISTENING = "LISTENING"


class KeywordSpotter(Protocol):
    """Minimal interface shared by the state machine and tests."""

    @property
    def available(self) -> bool: ...

    def detect(self, pcm_16khz: bytes) -> bool: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class WakeGateResult:
    """Gate decision for one decoded device PCM frame."""

    forward_pcm: tuple[bytes, ...] = ()
    woke: bool = False
    closed: bool = False
    bypassed: bool = False


class WakeWordUnavailable(RuntimeError):
    """Raised when the optional KWS runtime or model files are not available."""


class SherpaOnnxKeywordSpotter:
    """sherpa-onnx KeywordSpotter wrapper for the robot's wake phrase."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        keyword: str = WAKE_KEYWORD,
        phrase: str = WAKE_PHRASE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_threads: int = 2,
        provider: str = "cpu",
        keywords_score: float = 7.0,
        keywords_threshold: float = 0.05,
    ) -> None:
        try:
            import numpy as np  # noqa: PLC0415
            import sherpa_onnx  # noqa: PLC0415
        except (ImportError, OSError) as exc:
            raise WakeWordUnavailable(
                f"sherpa-onnx unavailable; install gateway[wakeword] and make sure the native library loads to use the wake word: {exc}"
            ) from exc

        self._np = np
        self._sample_rate = sample_rate
        self._phrase = phrase
        self._keyword = keyword
        self._model_dir = _resolve_model_dir(model_dir)
        self._tokens = _require_file(self._model_dir / "tokens.txt")
        self._encoder = _find_onnx(self._model_dir, "encoder")
        self._decoder = _find_onnx(self._model_dir, "decoder")
        self._joiner = _find_onnx(self._model_dir, "joiner")
        keywords_file = self._ensure_keywords_file()
        # sherpa-onnx separates per-stream keywords with "/".
        self._stream_keyword = (
            "" if keywords_file.name == "stackchan_keywords.txt" else keyword.replace("|", "/")
        )
        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(self._tokens),
                encoder=str(self._encoder),
                decoder=str(self._decoder),
                joiner=str(self._joiner),
                num_threads=num_threads,
                keywords_file=str(keywords_file),
                provider=provider,
                keywords_score=keywords_score,
                keywords_threshold=keywords_threshold,
            )
            self._stream = self._spotter.create_stream(self._stream_keyword)
        except Exception as exc:  # pragma: no cover - needs the native runtime
            raise WakeWordUnavailable(f"sherpa-onnx KWS initialisation failed: {exc}") from exc
        logger.info(
            "KWS ready model=%s dir=%s phrase=%r keyword=%r score=%.2f threshold=%.2f",
            self._model_dir.name,
            self._model_dir,
            self._phrase,
            self._keyword,
            keywords_score,
            keywords_threshold,
        )

    @property
    def available(self) -> bool:
        return True

    def detect(self, pcm_16khz: bytes) -> bool:
        sample_count = len(pcm_16khz) // 2
        if sample_count <= 0:
            return False
        trimmed = pcm_16khz[: sample_count * 2]
        samples = self._np.frombuffer(trimmed, dtype="<i2").astype(self._np.float32)
        samples /= 32768.0
        self._stream.accept_waveform(self._sample_rate, samples)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
        result = str(self._spotter.get_result(self._stream) or "")
        if not result:
            return False
        self._spotter.reset_stream(self._stream)
        return self._result_matches(result)

    def _result_matches(self, result: str) -> bool:
        if self._phrase in result:
            return True
        compact_result = "".join(ch.lower() for ch in result if ch.isalnum())
        compact_phrase = "".join(ch.lower() for ch in self._phrase if ch.isalnum())
        return bool(compact_phrase) and compact_phrase in compact_result

    def reset(self) -> None:
        self._spotter.reset_stream(self._stream)

    def _ensure_keywords_file(self) -> Path:
        path = self._model_dir / "stackchan_keywords.txt"
        wanted = self._keyword.strip()
        body = keywords_file_body(wanted)
        try:
            if not path.exists() or path.read_text(encoding="utf-8") != body:
                path.write_text(body, encoding="utf-8")
            return path
        except OSError as exc:
            fallback = self._model_dir / "keywords.txt"
            if fallback.is_file():
                logger.warning(
                    "cannot write the custom wake word file %s; using the model's keywords.txt and adding the keyword to the stream instead: %s",
                    path,
                    exc,
                )
                return fallback
            raise WakeWordUnavailable(f"cannot write the wake word file {path}: {exc}") from exc


@dataclass
class WakeGate:
    """Two-state wake gate with a pre-roll buffer and an RMS-based idle close."""

    kws: KeywordSpotter | None
    enabled: bool = True
    sample_rate: int = DEFAULT_SAMPLE_RATE
    preroll_s: float = DEFAULT_PREROLL_S
    idle_s: float = DEFAULT_WAKE_IDLE_S
    max_listening_s: float = DEFAULT_WAKE_MAX_LISTENING_S
    activity_rms_threshold: float = DEFAULT_ACTIVITY_RMS
    kws_reset_interval_s: float = DEFAULT_KWS_RESET_INTERVAL_S
    kws_reset_defer_max_s: float = DEFAULT_KWS_RESET_DEFER_MAX_S
    tts_stuck_max_s: float = DEFAULT_TTS_STUCK_MAX_S
    is_tts_active: Callable[[], bool] = field(default=lambda: False)
    clock: Callable[[], float] = time.monotonic
    state: WakeGateState = WakeGateState.DORMANT
    _preroll: deque[bytes] = field(default_factory=deque, init=False)
    _preroll_bytes: int = field(default=0, init=False)
    _last_voice_at: float | None = field(default=None, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _last_kws_reset_at: float | None = field(default=None, init=False)
    _tts_active_since: float | None = field(default=None, init=False)
    _listening_window_anchor: float | None = field(default=None, init=False)
    _kws_reset_due_since: float | None = field(default=None, init=False)
    _tts_stuck_latched: bool = field(default=False, init=False)

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.kws is not None and self.kws.available)

    @property
    def is_listening(self) -> bool:
        return self.state == WakeGateState.LISTENING

    @property
    def preroll_limit_bytes(self) -> int:
        return max(0, int(self.sample_rate * 2 * self.preroll_s))

    def process(self, pcm_16khz: bytes) -> WakeGateResult:
        """Process one PCM frame and return the frames that may be forwarded upstream."""
        if not self.available:
            return WakeGateResult(forward_pcm=(pcm_16khz,), bypassed=True)
        if not pcm_16khz:
            return WakeGateResult()

        now = self.clock()
        self._track_tts_activity(now)
        closed = False
        if self.is_listening and (
            self._is_idle(now) or self._is_past_max_listening_window(now)
        ):
            self.close()
            closed = True

        if self.state == WakeGateState.DORMANT:
            self._append_preroll(pcm_16khz)
            self._maybe_reset_kws_in_dormant(pcm_16khz, now)
            if not self.kws.detect(pcm_16khz):
                return WakeGateResult(closed=closed)
            self.state = WakeGateState.LISTENING
            self._last_voice_at = now
            self._opened_at = now
            self._listening_window_anchor = now
            forward = tuple(self._preroll)
            self._clear_preroll()
            return WakeGateResult(forward_pcm=forward, woke=True, closed=closed)

        if pcm_rms_int16(pcm_16khz) >= self.activity_rms_threshold:
            self._last_voice_at = now
        return WakeGateResult(forward_pcm=(pcm_16khz,), closed=closed)

    def force_wake(self) -> bool:
        """Open the listening window without the keyword (console "wake now")."""
        if not self.available:
            return False
        now = self.clock()
        self.state = WakeGateState.LISTENING
        self._last_voice_at = now
        self._opened_at = now
        self._listening_window_anchor = now
        # The pre-roll is old room audio, not part of what the user is about to say.
        self._clear_preroll()
        return True

    def close(self) -> bool:
        """Go back to DORMANT and drop the buffered conversation audio."""
        was_listening = self.is_listening
        self.state = WakeGateState.DORMANT
        self._last_voice_at = None
        self._opened_at = None
        self._listening_window_anchor = None
        self._clear_preroll()
        if self.kws is not None:
            self.kws.reset()
            self._last_kws_reset_at = self.clock()
        return was_listening

    def reset(self) -> None:
        self.close()

    def _is_tts_active(self, now: float) -> bool:
        try:
            active = bool(self.is_tts_active())
        except Exception:
            logger.debug("wake gate TTS getter failed; treating as inactive")
            active = False
        if not active:
            self._tts_stuck_latched = False
            return False
        if self._tts_stuck_latched:
            return False
        if (
            self._tts_active_since is not None
            and self.tts_stuck_max_s > 0
            and now - self._tts_active_since >= self.tts_stuck_max_s
        ):
            logger.warning(
                "TTS active signal stuck for %.0fs (>= %.0fs); "
                "treating as inactive to avoid permanent listen lock",
                now - self._tts_active_since,
                self.tts_stuck_max_s,
            )
            self._tts_stuck_latched = True
            return False
        return True

    def _track_tts_activity(self, now: float) -> None:
        if self._is_tts_active(now):
            if self._tts_active_since is None:
                self._tts_active_since = now
            return
        if self._tts_active_since is not None:
            was_stuck = self._tts_stuck_latched
            # The maximum listening window restarts when TTS ends (including the stuck-TTS fallback).
            self._listening_window_anchor = now
            self._tts_active_since = None
            if not was_stuck:
                # Normal end: treat now as new activity, so the user gets a full idle_s window to answer.
                self._last_voice_at = now

    def _is_idle(self, now: float) -> bool:
        if self._last_voice_at is None or self.idle_s <= 0:
            return False
        if self._is_tts_active(now):
            return False
        return now - self._last_voice_at >= self.idle_s

    def _is_past_max_listening_window(self, now: float) -> bool:
        anchor = self._listening_window_anchor or self._opened_at
        if anchor is None or self.max_listening_s <= 0:
            return False
        if self._is_tts_active(now):
            return False
        return now - anchor >= self.max_listening_s

    def _append_preroll(self, pcm: bytes) -> None:
        limit = self.preroll_limit_bytes
        if limit <= 0:
            return
        self._preroll.append(pcm)
        self._preroll_bytes += len(pcm)
        while self._preroll and self._preroll_bytes > limit:
            old = self._preroll.popleft()
            self._preroll_bytes -= len(old)

    def _clear_preroll(self) -> None:
        self._preroll.clear()
        self._preroll_bytes = 0

    def _maybe_reset_kws_in_dormant(self, pcm: bytes, now: float) -> None:
        """Reset the KWS stream on a fixed period while DORMANT, so detection
        does not degrade over time.

        The reset does not wait for a quiet room: with a TV or conversation
        going on it still happens when due. Side effect: a reset drops a
        half-recognised wake word. When the current frame's RMS suggests
        someone may be saying the wake word, the reset is postponed briefly;
        the total postponement is capped so "maybe a hit" cannot turn into
        never resetting.
        """
        if self.kws is None or self.kws_reset_interval_s <= 0:
            return
        if self._last_kws_reset_at is None:
            self._last_kws_reset_at = now
            return
        if now - self._last_kws_reset_at < self.kws_reset_interval_s:
            self._kws_reset_due_since = None
            return
        if self._kws_reset_due_since is None:
            self._kws_reset_due_since = now
        if self._should_defer_kws_reset(pcm, now):
            return
        self.kws.reset()
        self._last_kws_reset_at = now
        self._kws_reset_due_since = None

    def _should_defer_kws_reset(self, pcm: bytes, now: float) -> bool:
        if self.kws_reset_defer_max_s <= 0:
            return False
        due_since = self._kws_reset_due_since
        if due_since is None:
            return False
        if now - due_since >= self.kws_reset_defer_max_s:
            return False
        # Postpone only for high-energy speech frames (possibly the wake word);
        # a room that is loud all the time is reset once the cap is reached.
        return pcm_rms_int16(pcm) >= self.activity_rms_threshold


def pcm_rms_int16(pcm: bytes) -> float:
    """Return the RMS energy of little-endian signed 16-bit mono PCM."""
    sample_count = len(pcm) // 2
    if sample_count <= 0:
        return 0.0
    total = 0
    for (sample,) in struct.iter_unpack("<h", pcm[: sample_count * 2]):
        total += sample * sample
    return math.sqrt(total / sample_count)


def create_wake_gate_from_env(
    *,
    kws: KeywordSpotter | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> WakeGate | None:
    """Create the production wake gate; None means pass-through mode."""
    if _env_disabled("STACKCHAN_WAKE_WORD"):
        logger.info("wake word gate disabled by STACKCHAN_WAKE_WORD=0")
        return None
    # Console settings default to the same env values, and win once edited.
    from .console_settings import get_console_settings  # noqa: PLC0415

    settings = get_console_settings()
    try:
        spotter = kws or SherpaOnnxKeywordSpotter(
            model_dir=_model_dir_from_env(),
            keyword=_keyword_from_env(),
            phrase=_phrase_from_env(),
            keywords_score=float(settings.get("kws_score")),
            keywords_threshold=float(settings.get("kws_threshold")),
        )
    except WakeWordUnavailable as exc:
        logger.warning("wake word gate unavailable; passing device audio through: %s", exc)
        return None
    return WakeGate(
        kws=spotter,
        idle_s=float(settings.get("wake_idle_s")),
        max_listening_s=_float_env(
            "STACKCHAN_WAKE_MAX_LISTENING_S",
            DEFAULT_WAKE_MAX_LISTENING_S,
        ),
        activity_rms_threshold=_float_env(
            "STACKCHAN_WAKE_RMS_THRESHOLD",
            DEFAULT_ACTIVITY_RMS,
        ),
        kws_reset_interval_s=_float_env(
            "STACKCHAN_KWS_RESET_INTERVAL_S",
            DEFAULT_KWS_RESET_INTERVAL_S,
        ),
        clock=clock,
    )


def model_download_commands(target_dir: str | Path | None = None) -> str:
    """Return the shell commands to install the current KWS model by hand."""
    base = Path(target_dir) if target_dir is not None else _default_model_root()
    archive = f"{KWS_MODEL_NAME}.tar.bz2"
    url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{archive}"
    return "\n".join(
        [
            f"mkdir -p {base}",
            f"cd {base}",
            f"curl -L -O {url}",
            f"tar xf {archive}",
            f"rm {archive}",
            f"printf '%s\\n' '{WAKE_KEYWORD}' > {KWS_MODEL_NAME}/stackchan_keywords.txt",
        ]
    )


def keywords_file_body(keyword: str) -> str:
    """Keywords file content: the configured keyword, plus its vowel variants for a built-in wake word.

    A custom keyword may list pronunciation variants separated by "|", e.g.
    "D IY1 F AO1 R S IY1 @d4c|D IY1 F AO1 S IY1 @d4c"; each becomes its own line.
    """
    wanted = keyword.strip()
    variants: tuple[str, ...] = ()
    if wanted == WAKE_KEYWORD:
        variants = WAKE_KEYWORD_VARIANTS
    elif wanted == HEY_GROKI_WAKE_KEYWORD:
        variants = HEY_GROKI_WAKE_KEYWORD_VARIANTS
    lines = [part.strip() for part in wanted.split("|") if part.strip()]
    return "".join(f"{line}\n" for line in (*lines, *variants))


def _looks_like_kws_dir(path: Path) -> bool:
    return (path / "tokens.txt").is_file() and any(path.glob("encoder-*.onnx"))


def _resolve_model_dir(model_dir: str | Path | None) -> Path:
    root = Path(model_dir) if model_dir is not None else _model_dir_from_env()
    if _looks_like_kws_dir(root):
        return root
    named = root / KWS_MODEL_NAME
    if _looks_like_kws_dir(named):
        return named
    raise WakeWordUnavailable(
        f"model directory not found: {named}; and the path is not a KWS directory itself (needs tokens.txt + encoder-*.onnx)"
    )


def _find_onnx(model_dir: Path, prefix: str) -> Path:
    cands = [p for p in model_dir.glob(f"{prefix}-*.onnx") if p.is_file()]
    direct = model_dir / f"{prefix}.onnx"
    if direct.is_file():
        cands.append(direct)
    if not cands:
        raise WakeWordUnavailable(f"model file missing: {model_dir}/{prefix}*.onnx")
    fp32 = [p for p in cands if "int8" not in p.name]
    pool = fp32 or cands
    preferred = [p for p in pool if "chunk-16-left-64" in p.name]
    return sorted(preferred or pool)[0]


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise WakeWordUnavailable(f"model file missing: {path}")
    return path


def _model_dir_from_env() -> Path:
    value = os.getenv("STACKCHAN_KWS_MODEL_DIR")
    return Path(value).expanduser() if value else _default_model_root()


def _default_model_root() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "kws"


def _phrase_from_env() -> str:
    for name in ("WAKE_PHRASE", "STACKCHAN_WAKE_PHRASE"):
        raw = os.getenv(name)
        if raw and raw.strip():
            return raw.strip()
    return WAKE_PHRASE


def _keyword_from_env() -> str:
    for name in ("STACKCHAN_WAKE_KEYWORD", "WAKE_KEYWORD"):
        raw = os.getenv(name)
        if raw and raw.strip():
            return raw.strip()
    phrase = _phrase_from_env()
    if phrase.lower() == HEY_GROKI_WAKE_PHRASE:
        return HEY_GROKI_WAKE_KEYWORD
    return WAKE_KEYWORD


def _env_disabled(name: str) -> bool:
    value = os.getenv(name, "1").strip().lower()
    return value in {"0", "false", "no", "off"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using %.3f", name, raw, default)
        return default


def flatten_pcm(chunks: Iterable[bytes]) -> bytes:
    """Small helpers for tests and diagnostics."""
    return b"".join(chunks)
