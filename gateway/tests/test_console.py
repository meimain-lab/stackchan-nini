# SPDX-License-Identifier: MIT
"""Console settings, local-only control API, listen modes and wake-now."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from stackchan_mcp import console_control
from stackchan_mcp.console_settings import ConsoleSettings, SettingsError, get_console_settings
from stackchan_mcp.wake_gate import WakeGate, WakeGateState


# ---- settings store -------------------------------------------------------------


def test_settings_validate_persist_and_only_store_explicit_values(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    store = ConsoleSettings(path)
    monkeypatch.setenv("STACKCHAN_FACE_TRACKER_AUTOSTART", "1")
    monkeypatch.setenv("STACKCHAN_ROBOT_NAME", "小妮")
    assert store.get("robot_name") == "小妮"          # unset keys follow env live
    changed = store.update({"vad_silence_ms": "1200", "face_tracking": False})
    assert changed == {"vad_silence_ms": 1200, "face_tracking": False}
    assert json.loads(path.read_text()) == {"vad_silence_ms": 1200, "face_tracking": False}
    assert ConsoleSettings(path).get("vad_silence_ms") == 1200
    with pytest.raises(SettingsError):
        store.update({"vad_silence_ms": 50})
    with pytest.raises(SettingsError):
        store.update({"listen_mode": "sometimes"})
    with pytest.raises(SettingsError):
        store.update({"no_such_setting": 1})
    assert store.update({"vad_silence_ms": 1200}) == {}  # unchanged value is a no-op
    # An explicit choice equal to the current default still sticks.
    store.update({"head_follow": True})
    monkeypatch.setenv("STACKCHAN_HEAD_FOLLOW_DEFAULT", "0")
    assert store.get("head_follow") is True


def test_listeners_hear_only_real_changes(tmp_path):
    store = ConsoleSettings(tmp_path / "s.json")
    seen = []
    store.add_listener(seen.append)
    store.update({"head_follow": False})
    store.update({"head_follow": False})
    assert seen == [{"head_follow": False}]


# ---- control API ----------------------------------------------------------------


class FakeGateway:
    def __init__(self):
        self.woke = 0
        self.said = []

    def console_runtime_status(self):
        return {"device": {"connected": True}}

    async def console_wake_now(self):
        self.woke += 1
        return True

    async def console_say(self, text):
        self.said.append(text)
        return True


async def _client(gateway):
    app = web.Application()
    console_control.register_console_routes(app, gateway)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_control_api_reads_updates_and_rejects_bad_values():
    gateway = FakeGateway()
    client = await _client(gateway)
    try:
        body = await (await client.get("/control/settings")).json()
        assert body["settings"]["listen_mode"] == "wake_word"
        assert body["schema"]["vad_silence_ms"]["min"] == 300
        resp = await client.post("/control/settings", json={"listen_mode": "always"})
        assert (await resp.json())["changed"] == {"listen_mode": "always"}
        assert get_console_settings().get("listen_mode") == "always"
        bad = await client.post("/control/settings", json={"kws_threshold": 5})
        assert bad.status == 400
        assert (await (await client.post("/control/wake")).json())["ok"] is True
        await client.post("/control/say", json={"text": "试听一下"})
        assert gateway.woke == 1 and gateway.said == ["试听一下"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_control_routes_refuse_requests_from_other_machines():
    async def handler(_request):
        return web.json_response({"ok": True})

    guarded = console_control._local_only(handler)
    denied = await guarded(SimpleNamespace(remote="192.168.8.67"))
    allowed = await guarded(SimpleNamespace(remote="127.0.0.1"))
    assert denied.status == 403 and allowed.status == 200


@pytest.mark.asyncio
async def test_memory_can_be_listed_and_cleared(tmp_path, monkeypatch):
    from stackchan_mcp.deepseek_pipeline_bridge import MemoryStore

    db = tmp_path / "memory.sqlite"
    monkeypatch.setenv("STACKCHAN_MEMORY_DB", str(db))
    store = MemoryStore(db)
    store.add("user", "你好")
    store.add("assistant", "你好呀")
    store.close()
    client = await _client(FakeGateway())
    try:
        listed = await (await client.get("/control/memory?limit=10")).json()
        assert listed["total"] == 2 and listed["turns"][-1]["content"] == "你好呀"
        cleared = await (await client.delete("/control/memory")).json()
        assert cleared["deleted"] == 2
        assert (await (await client.get("/control/memory")).json())["total"] == 0
    finally:
        await client.close()


# ---- wake gate & listen modes ---------------------------------------------------------


class Spotter:
    available = True

    def __init__(self, hits=()):
        self.hits = list(hits)

    def detect(self, _pcm):
        return self.hits.pop(0) if self.hits else False

    def reset(self):
        pass


def test_force_wake_opens_listening_without_old_preroll():
    gate = WakeGate(kws=Spotter())
    gate.process(b"\x01\x00" * 160)   # dormant: buffered as pre-roll
    assert gate.force_wake() is True
    assert gate.state == WakeGateState.LISTENING
    assert gate.process(b"\x02\x00" * 160).forward_pcm == (b"\x02\x00" * 160,)


def _proxy_with_bridge():
    from stackchan_mcp.gemini_voice_proxy import GeminiVoiceProxy

    proxy = GeminiVoiceProxy(wake_confirm_hold_s=0.0)
    bridge = SimpleNamespace(send_audio=AsyncMock(), running=True)
    proxy._bridge = bridge
    proxy._wake_gate = WakeGate(kws=Spotter())
    return proxy, bridge


@pytest.mark.asyncio
async def test_listen_mode_off_drops_audio_and_always_bypasses_the_wake_word():
    proxy, bridge = _proxy_with_bridge()
    settings = get_console_settings()
    settings.update({"listen_mode": "off"})
    await proxy._forward_gated_device_pcm(b"\x01\x00" * 160)
    bridge.send_audio.assert_not_awaited()
    settings.update({"listen_mode": "always"})
    await proxy._forward_gated_device_pcm(b"\x02\x00" * 160)
    bridge.send_audio.assert_awaited_once_with(b"\x02\x00" * 160)
    settings.update({"listen_mode": "wake_word"})
    await proxy._forward_gated_device_pcm(b"\x03\x00" * 160)   # no keyword: stays dormant
    assert bridge.send_audio.await_count == 1


@pytest.mark.asyncio
async def test_wake_now_opens_the_gate_and_plays_wake_feedback(monkeypatch):
    proxy, bridge = _proxy_with_bridge()
    monkeypatch.setattr(type(proxy), "connected", property(lambda self: True))
    proxy._on_woke = AsyncMock()
    assert await proxy.wake_now() is True
    assert proxy._wake_gate.state == WakeGateState.LISTENING
    proxy._on_woke.assert_awaited_once_with(bridge)
    await proxy._forward_gated_device_pcm(b"\x04\x00" * 160)
    bridge.send_audio.assert_awaited_with(b"\x04\x00" * 160)


def test_changing_sensitivity_rebuilds_the_wake_gate_and_idle_applies_live():
    proxy, _bridge = _proxy_with_bridge()
    proxy.apply_console_settings({"wake_idle_s": 60.0})
    assert proxy._wake_gate.idle_s == 60.0
    proxy.apply_console_settings({"kws_threshold": 0.2})
    assert proxy._wake_gate is None   # rebuilt with new sensitivity on the next frame


# ---- bridge reads console settings -----------------------------------------------------


def _bare_bridge(monkeypatch, tmp_path):
    from stackchan_mcp.deepseek_pipeline_bridge import DeepSeekPipelineBridge, MemoryStore

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    return DeepSeekPipelineBridge(
        on_audio=AsyncMock(), on_turn_complete=AsyncMock(), on_interrupt=AsyncMock(),
        on_session_dead=AsyncMock(), memory=MemoryStore(tmp_path / "m.sqlite"),
    )


def test_brief_style_name_and_custom_persona_shape_the_prompt(monkeypatch, tmp_path):
    bridge = _bare_bridge(monkeypatch, tmp_path)
    settings = get_console_settings()
    settings.update({"reply_style": "brief", "robot_name": "小星"})
    prompt = bridge._system_prompt()
    assert "小星" in prompt and "简短" in prompt and "妮妮，一个" not in prompt
    settings.update({"persona": "你是一只会说话的猫。"})
    assert bridge._system_prompt().startswith("你是一只会说话的猫。")


def test_elevenlabs_options_go_where_the_dialogue_socket_accepts_them(monkeypatch, tmp_path):
    bridge = _bare_bridge(monkeypatch, tmp_path)
    get_console_settings().update({
        "eleven_language": "zh", "eleven_seed": 42, "eleven_text_normalization": "on",
        "tts_model": "eleven_v4_turbo",
    })
    query = parse_qs(bridge._elevenlabs_query())
    assert query["language_code"] == ["zh"] and query["seed"] == ["42"]
    assert query["apply_text_normalization"] == ["on"] and query["model_id"] == ["eleven_v4_turbo"]


@pytest.mark.asyncio
async def test_edge_engine_setting_skips_elevenlabs(monkeypatch, tmp_path):
    bridge = _bare_bridge(monkeypatch, tmp_path)
    bridge._eleven_key = "fixture"
    get_console_settings().update({"tts_engine": "edge", "eleven_voice_id": "abc"})
    called = []

    async def eleven(_text):
        called.append("eleven")
        yield b"x"

    async def edge(_text):
        called.append("edge")
        yield b"\x00\x00"

    bridge._elevenlabs = eleven
    bridge._edge_tts = edge
    chunks = [c async for c in bridge._synthesize("你好")]
    assert called == ["edge"] and chunks == [b"\x00\x00"]


# ---- listening face ------------------------------------------------------------------


def test_wake_and_follow_up_window_use_the_firmware_listening_face():
    from stackchan_mcp.device_emotion import LISTENING_EMOTION, face_to_device_emotion

    assert LISTENING_EMOTION == "listening"          # firmware Expression::Listening
    assert face_to_device_emotion("listening") == "listening"
    proxy, _bridge = _proxy_with_bridge()
    assert proxy._post_tts_face() == "idle"           # dormant gate after a reply
    proxy._wake_gate.force_wake()
    assert proxy._post_tts_face() == "listening"      # window still open


@pytest.mark.asyncio
async def test_voice_error_face_stays_until_speech_works_or_a_wake():
    from stackchan_mcp.device_emotion import DEAD_LED_RGB, face_to_device_emotion

    assert face_to_device_emotion("voice_error") == "dizzy"   # firmware Expression::Dizzy, red x eyes
    proxy, bridge = _proxy_with_bridge()
    proxy._set_leds_safe = AsyncMock()
    proxy._send_device_emotion_safe = AsyncMock()
    bridge.clear_conversation_context = None
    await proxy._handle_voice_error(True)
    proxy._send_device_emotion_safe.assert_awaited_once_with("dizzy")
    proxy._set_leds_safe.assert_awaited_once_with(*DEAD_LED_RGB)
    proxy._wake_gate.force_wake()
    assert proxy._post_tts_face() == "voice_error"
    assert proxy._post_tts_led_rgb() == DEAD_LED_RGB
    await proxy._on_wake_closed(bridge)                       # dormant again: still red
    proxy._send_device_emotion_safe.assert_awaited_with("dizzy")
    await proxy._handle_voice_error(False)
    assert proxy._post_tts_face() == "listening"              # gate still open in this test


# ---- ElevenLabs: one dialogue socket per reply ------------------------------------------


class FakeDialogueSocket:
    """Records client messages; answers with one audio chunk per input, then is_final."""

    def __init__(self):
        self.sent = []
        self._out = None

    async def __aenter__(self):
        import asyncio

        self._out = asyncio.Queue()
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, raw):
        import base64

        message = json.loads(raw)
        self.sent.append(message)
        if "inputs" in message:
            await self._out.put(json.dumps({"audio": base64.b64encode(b"\x01\x00").decode()}))
        if message.get("close_socket"):
            await self._out.put(json.dumps({"is_final": True}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._out.get()


def _eleven_bridge(monkeypatch, tmp_path, socket=None, fail=False):
    import websockets

    bridge = _bare_bridge(monkeypatch, tmp_path)
    bridge._eleven_key = "fixture"
    get_console_settings().update({"tts_engine": "elevenlabs", "eleven_voice_id": "voice-x"})

    def connect(*_args, **_kwargs):
        if fail:
            raise OSError("network down")
        return socket

    monkeypatch.setattr(websockets, "connect", connect)
    edge_texts = []

    async def edge(text):
        edge_texts.append(text)
        yield b"\x02\x00"

    bridge._edge_tts = edge
    return bridge, edge_texts


@pytest.mark.asyncio
async def test_whole_reply_streams_into_one_socket_without_per_sentence_flush(monkeypatch, tmp_path):
    import asyncio

    socket = FakeDialogueSocket()
    bridge, edge_texts = _eleven_bridge(monkeypatch, tmp_path, socket)
    queue = asyncio.Queue()
    for piece in ["[happy]看呢，我一直都在这儿呀。", "你刚才说腿上的电脑有点热，", "要不要我帮你查查？", None]:
        queue.put_nowait(piece)
    await bridge._speak_queue(queue)
    texts = [m["inputs"][0]["text"] for m in socket.sent if "inputs" in m]
    assert texts == ["看呢，我一直都在这儿呀。", "你刚才说腿上的电脑有点热，", "要不要我帮你查查？"]
    assert socket.sent[0] == {"voices": ["voice-x"]}
    # Only the opening sentence is flushed (fast first sound); the rest streams as one segment.
    kinds = [next(iter(m)) for m in socket.sent[1:]]
    assert kinds == ["inputs", "flush", "inputs", "inputs", "close_socket"]
    assert socket.sent[-1] == {"close_socket": True}
    assert edge_texts == []
    bridge._on_turn_complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_pause_in_the_reply_flushes_the_short_filler(monkeypatch, tmp_path):
    import asyncio

    from stackchan_mcp import deepseek_pipeline_bridge as dpb

    monkeypatch.setattr(dpb, "ELEVEN_PAUSE_FLUSH_S", 0.05)
    socket = FakeDialogueSocket()
    bridge, _edge = _eleven_bridge(monkeypatch, tmp_path, socket)
    queue = asyncio.Queue()
    queue.put_nowait("[curious]我查一下哦。")

    async def later():
        await asyncio.sleep(0.2)          # the web search
        queue.put_nowait("今天上海晴，最高二十三度。")
        queue.put_nowait(None)

    feeder = asyncio.create_task(later())
    await bridge._speak_queue(queue)
    await feeder
    kinds = [next(iter(m)) for m in socket.sent[1:]]
    # The 6-char filler is below the opening threshold, so the pause flushes it;
    # the answer's first sentence then gets the opening flush.
    assert kinds == ["inputs", "flush", "inputs", "flush", "close_socket"]


@pytest.mark.asyncio
async def test_socket_failure_shows_the_voice_error_face_instead_of_another_voice(monkeypatch, tmp_path):
    import asyncio

    bridge, edge_texts = _eleven_bridge(monkeypatch, tmp_path, fail=True)
    reports = []

    async def on_voice_error(failed):
        reports.append(failed)

    bridge.on_voice_error = on_voice_error
    queue = asyncio.Queue()
    for piece in ["第一句。", "第二句。", None]:
        queue.put_nowait(piece)
    await asyncio.wait_for(bridge._speak_queue(queue), 2)
    assert edge_texts == [] and reports == [True]
    assert queue.empty()                               # the unspoken rest was drained
    bridge._on_turn_complete.assert_not_awaited()      # nothing played, no tts.stop


@pytest.mark.asyncio
async def test_a_failed_socket_is_retried_once_and_clears_the_error_face(monkeypatch, tmp_path):
    import asyncio

    import websockets

    socket = FakeDialogueSocket()
    bridge, edge_texts = _eleven_bridge(monkeypatch, tmp_path, socket)
    calls = []

    def flaky_connect(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionResetError("reset by peer")
        return socket

    monkeypatch.setattr(websockets, "connect", flaky_connect)
    reports = []

    async def on_voice_error(failed):
        reports.append(failed)

    bridge.on_voice_error = on_voice_error
    queue = asyncio.Queue()
    for piece in ["第一句话说完了。", "第二句。", None]:
        queue.put_nowait(piece)
    await bridge._speak_queue(queue)
    assert len(calls) == 2 and edge_texts == [] and reports == [False]
    texts = [m["inputs"][0]["text"] for m in socket.sent if "inputs" in m]
    assert texts == ["第一句话说完了。", "第二句。"]


class RejectingDialogueSocket(FakeDialogueSocket):
    """Takes the whole reply, then errors (e.g. an unknown voice id) instead of audio."""

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if message.get("close_socket"):
            await self._out.put(json.dumps({"error": "voice_not_found", "message": "voice not found"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("second_ok", [True, False])
async def test_a_socket_rejected_after_the_whole_reply_does_not_hang(monkeypatch, tmp_path, second_ok):
    # The first socket took the reply's None terminator; the retry must not wait for it again.
    import asyncio

    import websockets

    sockets = [RejectingDialogueSocket(), FakeDialogueSocket() if second_ok else RejectingDialogueSocket()]
    bridge, edge_texts = _eleven_bridge(monkeypatch, tmp_path, sockets[0])
    monkeypatch.setattr(websockets, "connect", lambda *_a, **_k: sockets.pop(0))
    reports = []

    async def on_voice_error(failed):
        reports.append(failed)

    bridge.on_voice_error = on_voice_error
    queue = asyncio.Queue()
    for piece in ["这是一次声音故障测试。", None]:
        queue.put_nowait(piece)
    await asyncio.wait_for(bridge._speak_queue(queue), 2)
    assert reports == [not second_ok] and edge_texts == []


@pytest.mark.asyncio
async def test_quota_check_never_blocks_a_turn(monkeypatch):
    import asyncio

    from stackchan_mcp.deepseek_pipeline_bridge import QuotaGuard

    guard = QuotaGuard("fixture", 0.8)
    slow = asyncio.Event()

    async def refresh():
        await slow.wait()

    guard._refresh = refresh
    assert await asyncio.wait_for(guard.over_quota(), timeout=0.05) is False
    slow.set()
