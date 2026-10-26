from __future__ import annotations

import struct
import sys
from unittest.mock import MagicMock

from stackchan_mcp.wake_gate import (
    WAKE_KEYWORD,
    SherpaOnnxKeywordSpotter,
    WakeGate,
    WakeGateState,
    create_wake_gate_from_env,
    flatten_pcm,
    model_download_commands,
    pcm_rms_int16,
)


class FakeKeywordSpotter:
    def __init__(self, hits: list[bool] | None = None, *, available: bool = True) -> None:
        self.hits = list(hits or [])
        self.available = available
        self.seen: list[bytes] = []
        self.reset_count = 0

    def detect(self, pcm_16khz: bytes) -> bool:
        self.seen.append(pcm_16khz)
        return self.hits.pop(0) if self.hits else False

    def reset(self) -> None:
        self.reset_count += 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _pcm(value: int, samples: int = 160) -> bytes:
    return struct.pack("<" + "h" * samples, *([value] * samples))


def test_dormant_blocks_until_keyword_and_replays_preroll() -> None:
    kws = FakeKeywordSpotter([False, False, True])
    gate = WakeGate(kws=kws, sample_rate=10, preroll_s=0.6)
    first = _pcm(100, samples=2)  # 4 bytes
    second = _pcm(200, samples=2)
    third = _pcm(300, samples=2)

    assert gate.process(first).forward_pcm == ()
    assert gate.process(second).forward_pcm == ()
    result = gate.process(third)

    assert result.woke is True
    assert gate.state == WakeGateState.LISTENING
    # 10 Hz * 2 bytes * 0.6 s = 12 bytes: all three frames fit in the pre-roll.
    assert result.forward_pcm == (first, second, third)
    assert flatten_pcm(result.forward_pcm) == first + second + third


def test_preroll_buffer_drops_oldest_frames() -> None:
    kws = FakeKeywordSpotter([False, False, True])
    gate = WakeGate(kws=kws, sample_rate=10, preroll_s=0.4)
    first = _pcm(100, samples=2)  # 4 bytes
    second = _pcm(200, samples=2)
    third = _pcm(300, samples=2)

    gate.process(first)
    gate.process(second)
    result = gate.process(third)

    # 10 Hz * 2 bytes * 0.4 s = 8 bytes: only the last two frames are kept.
    assert result.forward_pcm == (second, third)


def test_listening_forwards_audio_and_refreshes_on_real_voice() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True])
    gate = WakeGate(
        kws=kws,
        idle_s=30.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )
    wake = _pcm(1200)
    voice = _pcm(1300)

    gate.process(wake)
    clock.advance(20.0)
    result = gate.process(voice)
    clock.advance(20.0)
    still_open = gate.process(_pcm(100))

    assert result.forward_pcm == (voice,)
    assert still_open.forward_pcm == (_pcm(100),)
    assert gate.state == WakeGateState.LISTENING


def test_listening_closes_after_max_window_even_when_audio_stays_loud() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True, False])
    gate = WakeGate(
        kws=kws,
        idle_s=30.0,
        max_listening_s=2.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(1200))
    clock.advance(1.5)
    still_open = gate.process(_pcm(1200))
    clock.advance(0.6)
    closed = gate.process(_pcm(1200))

    assert still_open.forward_pcm == (_pcm(1200),)
    assert closed.closed is True
    assert closed.forward_pcm == ()
    assert gate.state == WakeGateState.DORMANT


def test_background_noise_does_not_refresh_idle_timer() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True, False])
    gate = WakeGate(
        kws=kws,
        idle_s=30.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(1200))
    clock.advance(20.0)
    noise = gate.process(_pcm(50))
    clock.advance(11.0)
    closed = gate.process(_pcm(50))

    assert noise.forward_pcm == (_pcm(50),)
    assert closed.closed is True
    assert closed.forward_pcm == ()
    assert gate.state == WakeGateState.DORMANT


def test_speech_after_idle_timeout_requires_new_wake_word() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True, False])
    gate = WakeGate(
        kws=kws,
        idle_s=30.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(1200))
    clock.advance(31.0)
    result = gate.process(_pcm(1400))

    assert result.closed is True
    assert result.forward_pcm == ()
    assert gate.state == WakeGateState.DORMANT


def test_disabled_or_unavailable_gate_bypasses_audio() -> None:
    pcm = _pcm(100)
    disabled = WakeGate(kws=FakeKeywordSpotter([True]), enabled=False)
    unavailable = WakeGate(kws=FakeKeywordSpotter([True], available=False))

    assert disabled.process(pcm).forward_pcm == (pcm,)
    assert disabled.process(pcm).bypassed is True
    assert unavailable.process(pcm).forward_pcm == (pcm,)
    assert unavailable.process(pcm).bypassed is True


def test_dormant_periodic_reset_on_quiet_timeout() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([False])
    gate = WakeGate(
        kws=kws,
        kws_reset_interval_s=60.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(50))
    assert kws.reset_count == 0

    clock.advance(61.0)
    gate.process(_pcm(50))

    assert kws.reset_count == 1


def test_dormant_resets_even_when_loud() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([False])
    gate = WakeGate(
        kws=kws,
        kws_reset_interval_s=60.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(50))
    clock.advance(61.0)
    gate.process(_pcm(1200))
    clock.advance(16.0)
    gate.process(_pcm(1200))

    assert kws.reset_count == 1


def test_dormant_defers_reset_briefly_during_suspected_wake() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([False])
    gate = WakeGate(
        kws=kws,
        kws_reset_interval_s=60.0,
        kws_reset_defer_max_s=15.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(50))
    clock.advance(61.0)
    gate.process(_pcm(1200))
    assert kws.reset_count == 0

    clock.advance(5.0)
    gate.process(_pcm(50))
    assert kws.reset_count == 1


def test_dormant_forces_reset_after_defer_cap_despite_loud_audio() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([False])
    gate = WakeGate(
        kws=kws,
        kws_reset_interval_s=60.0,
        kws_reset_defer_max_s=10.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(50))
    clock.advance(61.0)
    for _ in range(5):
        gate.process(_pcm(1200))
        clock.advance(3.0)

    assert kws.reset_count == 1


def test_listening_does_not_close_while_tts_active() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True, False])
    tts_active = {"value": True}
    gate = WakeGate(
        kws=kws,
        idle_s=5.0,
        max_listening_s=10.0,
        activity_rms_threshold=500.0,
        is_tts_active=lambda: tts_active["value"],
        clock=clock,
    )

    gate.process(_pcm(1200))
    clock.advance(20.0)
    still_open = gate.process(_pcm(50))

    assert still_open.closed is False
    assert gate.state == WakeGateState.LISTENING


def test_listening_idle_window_restarts_when_tts_ends_after_long_reply() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True, False])
    tts_active = {"value": True}
    gate = WakeGate(
        kws=kws,
        idle_s=30.0,
        max_listening_s=120.0,
        activity_rms_threshold=500.0,
        is_tts_active=lambda: tts_active["value"],
        clock=clock,
    )

    gate.process(_pcm(1200))
    clock.advance(40.0)
    assert gate.process(_pcm(50)).closed is False

    tts_active["value"] = False
    assert gate.process(_pcm(50)).closed is False

    clock.advance(15.0)
    user_pcm = _pcm(1300)
    result = gate.process(user_pcm)

    assert result.closed is False
    assert result.forward_pcm == (user_pcm,)
    assert gate.state == WakeGateState.LISTENING


def test_listening_closes_after_tts_stuck_timeout() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True, False])
    gate = WakeGate(
        kws=kws,
        idle_s=5.0,
        max_listening_s=100.0,
        tts_stuck_max_s=10.0,
        is_tts_active=lambda: True,
        clock=clock,
    )

    gate.process(_pcm(1200))
    clock.advance(11.0)
    closed = gate.process(_pcm(50))

    assert closed.closed is True
    assert gate.state == WakeGateState.DORMANT


def test_max_listening_window_resets_after_tts_ends() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([True, False])
    tts_active = {"value": True}
    gate = WakeGate(
        kws=kws,
        idle_s=100.0,
        max_listening_s=10.0,
        is_tts_active=lambda: tts_active["value"],
        clock=clock,
    )

    gate.process(_pcm(1200))
    clock.advance(8.0)
    still_open = gate.process(_pcm(50))
    assert still_open.closed is False

    tts_active["value"] = False
    gate.process(_pcm(50))
    clock.advance(11.0)
    closed = gate.process(_pcm(50))

    assert closed.closed is True


def test_dormant_reset_then_wake_still_works() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([False, False, True])
    gate = WakeGate(
        kws=kws,
        kws_reset_interval_s=60.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )

    gate.process(_pcm(50))
    clock.advance(61.0)
    gate.process(_pcm(50))
    assert kws.reset_count == 1

    result = gate.process(_pcm(300))

    assert result.woke is True
    assert gate.state == WakeGateState.LISTENING


def test_dormant_periodic_reset_preserves_preroll() -> None:
    clock = FakeClock()
    kws = FakeKeywordSpotter([False, False, True])
    gate = WakeGate(
        kws=kws,
        sample_rate=10,
        preroll_s=0.6,
        kws_reset_interval_s=60.0,
        activity_rms_threshold=500.0,
        clock=clock,
    )
    first = _pcm(100, samples=2)
    second = _pcm(200, samples=2)

    gate.process(first)
    clock.advance(61.0)
    gate.process(second)
    assert kws.reset_count == 1

    result = gate.process(_pcm(300, samples=2))

    assert result.woke is True
    assert result.forward_pcm == (first, second, _pcm(300, samples=2))


def test_close_resets_state_and_keyword_spotter() -> None:
    kws = FakeKeywordSpotter([True])
    gate = WakeGate(kws=kws)

    gate.process(_pcm(1000))
    assert gate.close() is True

    assert gate.state == WakeGateState.DORMANT
    assert kws.reset_count == 1


def test_pcm_rms_int16_uses_sample_energy() -> None:
    pcm = _pcm(1000, samples=3)
    assert pcm_rms_int16(pcm) == 1000.0


def test_create_wake_gate_returns_none_when_model_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("STACKCHAN_WAKE_WORD", raising=False)
    monkeypatch.setenv("STACKCHAN_KWS_MODEL_DIR", "/tmp/stackchan-missing-kws-model")

    assert create_wake_gate_from_env() is None


def test_create_wake_gate_returns_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("STACKCHAN_WAKE_WORD", "0")

    assert create_wake_gate_from_env(kws=FakeKeywordSpotter([True])) is None


def test_model_download_commands_include_keyword_and_model_name() -> None:
    commands = model_download_commands("gateway/models/kws")

    assert "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20" in commands
    assert WAKE_KEYWORD in commands


def test_sherpa_onnx_keyword_spotter_passes_keywords_score_and_threshold(monkeypatch, tmp_path) -> None:
    """keywords_score / keywords_threshold must be passed through to sherpa_onnx.KeywordSpotter."""
    mock_np = MagicMock()
    mock_sherpa = MagicMock()
    mock_spotter = MagicMock()
    mock_stream = MagicMock()
    mock_spotter.create_stream.return_value = mock_stream
    mock_sherpa.KeywordSpotter.return_value = mock_spotter

    monkeypatch.setitem(sys.modules, "numpy", mock_np)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mock_sherpa)

    # The model is not in the repo: empty files with the right names in a temp dir; the sherpa constructor is mocked, so nothing native loads
    model_dir = tmp_path / "kws-model"
    model_dir.mkdir()
    for name in (
        "tokens.txt",
        "encoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "joiner-epoch-13-avg-2-chunk-16-left-64.onnx",
    ):
        (model_dir / name).write_text("", encoding="utf-8")
    SherpaOnnxKeywordSpotter(
        model_dir=model_dir,
        keywords_score=3.14,
        keywords_threshold=0.07,
    )

    call = mock_sherpa.KeywordSpotter.call_args
    assert call.kwargs["keywords_score"] == 3.14
    assert call.kwargs["keywords_threshold"] == 0.07


def test_create_wake_gate_from_env_passes_kws_reset_interval(monkeypatch) -> None:
    monkeypatch.delenv("STACKCHAN_WAKE_WORD", raising=False)
    monkeypatch.setenv("STACKCHAN_KWS_RESET_INTERVAL_S", "120")

    gate = create_wake_gate_from_env(kws=FakeKeywordSpotter([False]))
    assert gate is not None
    assert gate.kws_reset_interval_s == 120.0


def test_create_wake_gate_from_env_passes_kws_env_vars(monkeypatch) -> None:
    """STACKCHAN_KWS_SCORE / STACKCHAN_KWS_THRESHOLD must reach the Sherpa constructor through _float_env."""
    monkeypatch.delenv("STACKCHAN_WAKE_WORD", raising=False)
    monkeypatch.setenv("STACKCHAN_KWS_SCORE", "4.2")
    monkeypatch.setenv("STACKCHAN_KWS_THRESHOLD", "0.09")

    mock_ctor = MagicMock()
    fake_kws = MagicMock()
    fake_kws.available = True
    mock_ctor.return_value = fake_kws

    import stackchan_mcp.wake_gate as wg  # noqa: PLC0415

    monkeypatch.setattr(wg, "SherpaOnnxKeywordSpotter", mock_ctor)

    gate = create_wake_gate_from_env()
    assert gate is not None

    call = mock_ctor.call_args
    assert call.kwargs["keywords_score"] == 4.2
    assert call.kwargs["keywords_threshold"] == 0.09


def _fake_kws_model(monkeypatch, tmp_path):
    mock_sherpa = MagicMock()
    monkeypatch.setitem(sys.modules, "numpy", MagicMock())
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mock_sherpa)
    model_dir = tmp_path / "kws-model"
    model_dir.mkdir()
    for name in (
        "tokens.txt",
        "encoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        "joiner-epoch-13-avg-2-chunk-16-left-64.onnx",
    ):
        (model_dir / name).write_text("", encoding="utf-8")
    return model_dir


def _clear_wake_env(monkeypatch) -> None:
    for name in ("WAKE_PHRASE", "STACKCHAN_WAKE_PHRASE", "WAKE_KEYWORD", "STACKCHAN_WAKE_KEYWORD"):
        monkeypatch.delenv(name, raising=False)


def test_default_wake_word_is_hi_grok(monkeypatch, tmp_path) -> None:
    from stackchan_mcp import wake_gate

    _clear_wake_env(monkeypatch)
    assert wake_gate._phrase_from_env() == "hi grok"
    assert wake_gate._keyword_from_env() == "HH AY1 G R AA1 K @hi_grok"

    model_dir = _fake_kws_model(monkeypatch, tmp_path)
    spotter = SherpaOnnxKeywordSpotter(
        model_dir=model_dir,
        keyword=wake_gate._keyword_from_env(),
        phrase=wake_gate._phrase_from_env(),
    )
    lines = (model_dir / "stackchan_keywords.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "HH AY1 G R AA1 K @hi_grok",
        "HH AY1 G R OW1 K @hi_grok_ow",
        "HH AY1 G R AO1 K @hi_grok_ao",
        "HH AY1 G R AH1 K @hi_grok_ah",
    ]
    # Every variant tag counts as the wake word.
    for tag in ("hi_grok", "hi_grok_ow", "hi_grok_ao", "hi_grok_ah"):
        assert spotter._result_matches(tag)
    assert not spotter._result_matches("hey_groki")


def test_hey_groki_phrase_switches_to_the_optional_keyword(monkeypatch, tmp_path) -> None:
    from stackchan_mcp import wake_gate

    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("STACKCHAN_WAKE_PHRASE", "Hey Groki")
    assert wake_gate._keyword_from_env() == "HH EY1 G R OW1 K IY0 @hey_groki"

    model_dir = _fake_kws_model(monkeypatch, tmp_path)
    spotter = SherpaOnnxKeywordSpotter(
        model_dir=model_dir,
        keyword=wake_gate._keyword_from_env(),
        phrase=wake_gate._phrase_from_env(),
    )
    lines = (model_dir / "stackchan_keywords.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "HH EY1 G R OW1 K IY0 @hey_groki",
        "HH EY1 G R AA1 K IY0 @hey_groki_aa",
        "HH EY1 G R AO1 K IY0 @hey_groki_ao",
        "HH EY1 G R AH1 K IY0 @hey_groki_ah",
    ]
    for tag in ("hey_groki", "hey_groki_aa", "hey_groki_ao", "hey_groki_ah"):
        assert spotter._result_matches(tag)
    assert not spotter._result_matches("hi_grok")


def test_custom_wake_keyword_is_written_alone(monkeypatch) -> None:
    from stackchan_mcp.wake_gate import keywords_file_body

    assert keywords_file_body("HH EH1 L OW0 @hello") == "HH EH1 L OW0 @hello\n"


def test_custom_wake_keyword_variants_are_split_into_lines() -> None:
    from stackchan_mcp.wake_gate import keywords_file_body

    body = keywords_file_body(" D IY1 F AO1 R S IY1 @d4c | D IY1 F AO1 S IY1 @d4c |")
    assert body == "D IY1 F AO1 R S IY1 @d4c\nD IY1 F AO1 S IY1 @d4c\n"
