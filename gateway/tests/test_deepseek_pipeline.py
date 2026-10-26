# SPDX-License-Identifier: MIT
"""DeepSeek ASR → LLM → TTS bridge with fake cloud services (no network)."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from stackchan_mcp.deepseek_pipeline_bridge import (
    DeepSeekPipelineBridge,
    MemoryStore,
    SentenceSplitter,
    clean_for_speech,
    openai_tools,
    split_mood,
)


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tool_delta(index, call_id, name, arguments):
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class FakeLLM:
    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        return FakeStream(self.rounds.pop(0))


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def declarations(self):
        return [
            {"name": "move_head", "description": "head", "parameters": {"type": "object", "properties": {}}},
            {"name": "express_emotion", "description": "mood", "parameters": {"type": "object", "properties": {}}},
            {"name": "self.tracking.start", "description": "track", "parameters": {"type": "object", "properties": {}}},
            {"name": "end_conversation", "description": "bye", "parameters": {"type": "object", "properties": {}}},
        ]

    async def execute(self, name, args):
        self.calls.append((name, args))
        return {"ok": True}

    async def observe_user_input(self, text, finished=False):
        return None


class FakeASR:
    def __init__(self):
        self.sent = []
        self._closed = asyncio.Event()

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def close(self):
        self._closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration


def _bridge(tmp_path, llm, spoken, monkeypatch, executor=None, asr_connect=None):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")

    async def tts(text, _voice):
        spoken.append(text)
        yield b"\x01\x00" * 240

    audio = AsyncMock()
    done = AsyncMock()
    bridge = DeepSeekPipelineBridge(
        on_audio=audio,
        on_turn_complete=done,
        on_interrupt=AsyncMock(),
        on_session_dead=AsyncMock(),
        llm_client=llm,
        tts_synthesize=tts,
        tool_executor=executor or FakeExecutor(),
        memory=MemoryStore(tmp_path / "memory.sqlite"),
        asr_connect=asr_connect or AsyncMock(return_value=FakeASR()),
    )
    return bridge, audio, done


def test_split_mood_and_clean_for_speech():
    assert split_mood("[happy] 你好") == ("happy", "你好")
    assert split_mood("[emotion:sad]唉") == ("sad", "唉")
    assert split_mood("[shouting] hi") == (None, "hi")
    assert split_mood("没有标签") == (None, "没有标签")
    assert clean_for_speech("**你好** [laughs] 😀") == "你好"


def test_sentence_splitter_cuts_on_punctuation_and_long_clauses():
    splitter = SentenceSplitter(soft_break_after=8)
    assert splitter.feed("你好呀！今天") == ["你好呀！"]
    assert splitter.feed("天气真的特别好，适合出去") == ["今天天气真的特别好，"]
    assert splitter.flush() == ["适合出去"]


def test_openai_tools_drop_emotion_tool_and_alias_dotted_names():
    tools, aliases = openai_tools(FakeExecutor().declarations())
    names = [t["function"]["name"] for t in tools]
    assert "express_emotion" not in names
    assert aliases["self_tracking_start"] == "self.tracking.start"
    assert tools[0]["type"] == "function"


@pytest.mark.asyncio
async def test_turn_speaks_reply_sets_mood_and_remembers(tmp_path, monkeypatch):
    llm = FakeLLM([_chunk("[ha"), _chunk("ppy]你好呀！"), _chunk("今天怎么样？")])
    spoken = []
    executor = FakeExecutor()
    bridge, audio, done = _bridge(tmp_path, llm, spoken, monkeypatch, executor)
    await bridge.start()
    await bridge._start_turn("你好")
    await bridge._turn_task
    await asyncio.sleep(0)

    assert spoken == ["你好呀！", "今天怎么样？"]
    assert audio.await_count == 2
    done.assert_awaited_once()
    assert ("express_emotion", {"mood": "happy"}) in executor.calls
    request = llm.requests[0]
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["model"] == "deepseek-flash"
    assert bridge._memory.recent(10)[-2:] == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀！今天怎么样？"},
    ]
    await bridge.stop()


@pytest.mark.asyncio
async def test_tool_call_round_runs_tool_then_speaks(tmp_path, monkeypatch):
    llm = FakeLLM(
        [_chunk(tool_calls=[_tool_delta(0, "call_1", "move_head", '{"yaw": 0, "pitch": 20}')])],
        [_chunk("[happy]点好头啦！")],
    )
    spoken = []
    executor = FakeExecutor()
    bridge, _audio, done = _bridge(tmp_path, llm, spoken, monkeypatch, executor)
    await bridge.start()
    await bridge._start_turn("点点头")
    await bridge._turn_task

    assert ("move_head", {"yaw": 0, "pitch": 20}) in executor.calls
    assert spoken == ["点好头啦！"]
    second = llm.requests[1]["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"ok": True})}
    done.assert_awaited_once()
    await bridge.stop()


@pytest.mark.asyncio
async def test_end_conversation_tool_closes_after_goodbye(tmp_path, monkeypatch):
    llm = FakeLLM(
        [_chunk("[happy]拜拜！"), _chunk(tool_calls=[_tool_delta(0, "c", "end_conversation", "{}")])],
    )
    end = AsyncMock()
    bridge, _audio, _done = _bridge(tmp_path, llm, [], monkeypatch)
    bridge._on_end_conversation = end
    await bridge.start()
    await bridge._start_turn("再见")
    await bridge._turn_task
    end.assert_awaited_once()
    assert len(llm.requests) == 1
    await bridge.stop()


@pytest.mark.asyncio
async def test_asr_is_warm_and_microphone_is_dropped_while_speaking(tmp_path, monkeypatch):
    asr = FakeASR()
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch, asr_connect=AsyncMock(return_value=asr))
    await bridge.start()
    await asyncio.sleep(0.01)  # keeper connects before any wake word
    assert bridge._asr_ws is asr
    bridge._speaking = True
    await bridge.send_audio(b"\x00\x00" * 160)
    assert asr.sent == []
    bridge._speaking = False
    await bridge.send_audio(b"\x01\x00" * 160)
    assert [e["type"] for e in asr.sent] == ["input_audio_buffer.append"]
    await bridge.stop()


@pytest.mark.asyncio
async def test_speech_during_connect_is_buffered_then_flushed_in_order(tmp_path, monkeypatch):
    asr = FakeASR()
    gate = asyncio.Event()

    async def slow_connect():
        await gate.wait()
        return asr

    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch, asr_connect=slow_connect)
    await bridge.start()
    await bridge.send_audio(b"\x01\x00")
    await bridge.send_audio(b"\x02\x00")
    gate.set()
    await asyncio.sleep(0.01)
    payloads = [e["audio"] for e in asr.sent]
    assert payloads == [base64.b64encode(b"\x01\x00").decode(), base64.b64encode(b"\x02\x00").decode()]
    await bridge.stop()


@pytest.mark.asyncio
async def test_unreachable_asr_is_announced_once_after_waiting(tmp_path, monkeypatch):
    import stackchan_mcp.deepseek_pipeline_bridge as module

    monkeypatch.setattr(module, "ASR_ANNOUNCE_AFTER_S", 0.02)
    spoken = []
    connect = AsyncMock(side_effect=OSError("proxy refused"))
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), spoken, monkeypatch, asr_connect=connect)
    await bridge.start()
    await bridge.send_audio(b"\x00\x00" * 160)
    assert spoken == []  # a short wait is not announced
    await asyncio.sleep(0.03)
    await bridge.send_audio(b"\x00\x00" * 160)
    await bridge._turn_task
    await bridge.send_audio(b"\x00\x00" * 160)
    assert spoken == ["我现在连不上语音识别，等一下再叫我吧。"]
    await bridge.stop()


@pytest.mark.asyncio
async def test_start_requires_deepseek_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    bridge = DeepSeekPipelineBridge(
        on_audio=AsyncMock(), on_turn_complete=AsyncMock(), on_interrupt=AsyncMock(),
        on_session_dead=AsyncMock(), llm_client=FakeLLM(), tool_executor=FakeExecutor(),
        memory=MemoryStore(tmp_path / "m.sqlite"),
    )
    with pytest.raises(RuntimeError):
        await bridge.start()


def test_gateway_selects_deepseek_proxy(monkeypatch):
    from stackchan_mcp.deepseek_voice_proxy import DeepSeekVoiceProxy
    from stackchan_mcp.gateway import Gateway

    monkeypatch.setenv("STACKCHAN_VOICE_BACKEND", "deepseek")
    gateway = Gateway.__new__(Gateway)
    gateway.voice_input_bridge = None
    gateway.usb_transport = None
    gateway.idle_gate = SimpleNamespace(notify_head_command=None, notify_device_state=None)
    gateway._set_face_tracking_from_voice = None
    assert isinstance(gateway._make_voice_proxy(), DeepSeekVoiceProxy)



@pytest.mark.asyncio
async def test_punctuation_only_transcript_does_not_start_a_turn(tmp_path, monkeypatch):
    llm = FakeLLM()
    bridge, _audio, _done = _bridge(tmp_path, llm, [], monkeypatch)
    await bridge.start()
    for noise in ("。", " ？！ ", "..."):
        await bridge._start_turn(noise)
    assert bridge._turn_task is None
    assert llm.requests == []
    await bridge.stop()


@pytest.mark.asyncio
async def test_name_reminder_sits_between_history_and_question(tmp_path, monkeypatch):
    llm = FakeLLM([_chunk("[happy]我叫妮妮。")])
    bridge, _audio, _done = _bridge(tmp_path, llm, [], monkeypatch)
    bridge._memory.add("assistant", "我叫D4C呀")
    await bridge.start()
    await bridge._start_turn("你叫什么名字？")
    await bridge._turn_task
    messages = llm.requests[0]["messages"]
    assert messages[-2]["role"] == "system" and "妮妮" in messages[-2]["content"]
    assert messages[-1] == {"role": "user", "content": "你叫什么名字？"}
    assert messages[-3] == {"role": "assistant", "content": "我叫D4C呀"}
    await bridge.stop()


@pytest.mark.asyncio
async def test_web_search_tool_says_filler_then_answers_from_result(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fixture")
    llm = FakeLLM(
        [_chunk(tool_calls=[_tool_delta(0, "s1", "web_search", '{"query": "上海今天天气"}')])],
        [_chunk("[happy]上海今天阴天，18 到 23 度。")],
    )
    spoken = []
    bridge, _audio, _done = _bridge(tmp_path, llm, spoken, monkeypatch)
    searched = []

    async def fake_search(query):
        searched.append(query)
        return {"ok": True, "query": query, "result": "上海阴，18-23℃"}

    bridge._web_search = fake_search
    await bridge.start()
    assert "web_search" in [t["function"]["name"] for t in bridge._tools]
    await bridge._start_turn("上海今天天气怎么样？")
    await bridge._turn_task
    assert searched == ["上海今天天气"]
    assert spoken == ["我查一下哦。", "上海今天阴天，18 到 23 度。"]
    tool_message = llm.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and "18-23" in tool_message["content"]
    await bridge.stop()


def test_persona_mentions_model_and_search(tmp_path, monkeypatch):
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)
    prompt = bridge._system_prompt()
    assert "妮妮" in prompt and "DeepSeek V4.1 Flash" in prompt and "web_search" in prompt


def _reasoning_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, tool_calls=None, reasoning_content=text))])


@pytest.mark.parametrize("question, expected", [
    ("你好呀", False),
    ("今天天气怎么样", False),
    ("三十二乘以十七等于多少", True),
    ("为什么天是蓝的", False),          # everyday "why": fast path
    ("帮我想个周末计划", False),
    ("你给我一点建议吧", False),
    ("帮我算一下每个月要还多少", True),
    ("对比一下这两款手机的优缺点", True),
    ("一步一步推导一下这个公式", True),
])
def test_thinking_turns_on_for_reasoning_questions(tmp_path, monkeypatch, question, expected):
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)
    assert bridge._wants_thinking(question) is expected


@pytest.mark.asyncio
async def test_thinking_turn_says_filler_and_echoes_reasoning_across_tool_rounds(tmp_path, monkeypatch):
    llm = FakeLLM(
        [_reasoning_chunk("先查一下。"), _chunk(tool_calls=[_tool_delta(0, "s1", "web_search", '{"query": "x"}')])],
        [_reasoning_chunk("结果够了。"), _chunk("[neutral]答案是 42。")],
    )
    spoken = []
    bridge, _audio, _done = _bridge(tmp_path, llm, spoken, monkeypatch)

    async def fake_search(query):
        return {"ok": True, "results": []}

    bridge._web_search = fake_search
    await bridge.start()
    await bridge._start_turn("帮我仔细想想为什么答案是 42")
    await bridge._turn_task
    assert spoken[0] == "我想想。"
    assert spoken[-1] == "答案是 42。"
    assert "我查一下哦。" not in spoken  # one filler per turn is enough
    first, second = llm.requests
    assert first["reasoning_effort"] == "high" and "extra_body" not in first
    assistant = second["messages"][-2]
    assert assistant["role"] == "assistant" and assistant["reasoning_content"] == "先查一下。"
    await bridge.stop()


@pytest.mark.asyncio
async def test_deepseek_search_results_are_ranked_and_trimmed(tmp_path, monkeypatch):
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)
    calls = []

    async def fake_tool(url, payload, timeout):
        calls.append((url, payload))
        return {"results": [
            {"title": "low", "rerank_score": 0.5, "summary": "a", "url": "https://a", "site_name": "A"},
            {"title": "high", "rerank_score": 2.1, "summary": "b" * 2000, "url": "https://b", "site_name": "B"},
        ]}

    bridge._deepseek_tool = fake_tool
    bridge._current_question = "上海今天天气怎么样？"
    result = await bridge._web_search("上海 天气")
    assert calls[0][1]["ori_question"] == "上海今天天气怎么样？"
    assert [r["title"] for r in result["results"]] == ["high", "low"]
    assert len(result["results"][0]["summary"]) == 800


@pytest.mark.asyncio
async def test_search_falls_back_to_qwen_when_deepseek_fails(tmp_path, monkeypatch):
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)

    async def broken(url, payload, timeout):
        raise RuntimeError("HTTP 503")

    async def qwen(query, started):
        return {"ok": True, "query": query, "result": "qwen answer"}

    bridge._deepseek_tool = broken
    bridge._qwen_search = qwen
    assert (await bridge._web_search("anything"))["result"] == "qwen answer"


@pytest.mark.asyncio
async def test_device_abort_stops_the_turn_and_drops_queued_audio(tmp_path, monkeypatch):
    gate = asyncio.Event()

    class SlowLLM(FakeLLM):
        async def _create(self, **kwargs):
            await gate.wait()
            return FakeStream([_chunk("[happy]很长很长的回答。")])

    bridge, _audio, _done = _bridge(tmp_path, SlowLLM(), [], monkeypatch)
    await bridge.start()
    await bridge._start_turn("给我讲个很长的故事")
    await asyncio.sleep(0)
    await bridge.interrupt()
    assert bridge._turn_task is None
    bridge._on_interrupt.assert_awaited_once()
    await bridge.interrupt()  # nothing in flight: no second abort to the proxy
    bridge._on_interrupt.assert_awaited_once()
    await bridge.stop()


def test_riddle_style_questions_use_thinking(tmp_path, monkeypatch):
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)
    assert bridge._wants_thinking("出个脑筋急转弯：三个开关怎么知道哪个控制哪盏灯")


@pytest.mark.asyncio
async def test_each_chat_request_is_logged_as_an_api_call(tmp_path, monkeypatch, caplog):
    import logging

    usage_chunk = SimpleNamespace(choices=[], model="deepseek-flash",
                                  usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 50, "completion_tokens": 9}))
    llm = FakeLLM([_chunk("[happy]你好！"), usage_chunk])
    bridge, _audio, _done = _bridge(tmp_path, llm, [], monkeypatch)
    await bridge.start()
    with caplog.at_level(logging.INFO, logger="stackchan_mcp.deepseek_pipeline_bridge"):
        await bridge._start_turn("你好")
        await bridge._turn_task
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("API call:")]
    assert any("provider=deepseek" in l and "purpose=chat" in l and "status=ok" in l
               and "prompt_tokens=50" in l and "thinking=off" in l for l in lines), lines
    assert llm.requests[0]["stream_options"] == {"include_usage": True}
    await bridge.stop()


from stackchan_mcp.deepseek_pipeline_bridge import is_filler_only, strip_wake_phrase, wake_aliases


@pytest.mark.parametrize("heard, expected", [
    ("你好，妮妮。今天天气怎么样？", "今天天气怎么样？"),
    ("你好倪妮。", ""),
    ("妮妮，帮我查一下", "帮我查一下"),
    ("你好吗？", "你好吗？"),
    ("妮妮你好", "你好"),
])
def test_wake_phrase_is_stripped_from_transcripts(heard, expected):
    assert strip_wake_phrase(heard, wake_aliases("你好妮妮", "妮妮")) == expected


def test_filler_only_detection():
    assert is_filler_only("嗯。") and is_filler_only("。") and is_filler_only("呃，嗯")
    assert not is_filler_only("嗯，帮我查天气")


def _grace(bridge, ms=20):
    bridge._turn_grace_ms = ms


@pytest.mark.asyncio
async def test_wake_word_only_transcript_does_not_reply(tmp_path, monkeypatch):
    monkeypatch.setenv("STACKCHAN_WAKE_PHRASE", "你好妮妮")
    llm = FakeLLM()
    bridge, _audio, _done = _bridge(tmp_path, llm, [], monkeypatch)
    _grace(bridge)
    await bridge.start()
    await bridge._on_transcript("你好，妮妮。")
    await bridge._on_transcript("嗯。")
    await asyncio.sleep(0.05)
    assert bridge._turn_task is None and llm.requests == []
    await bridge.stop()


@pytest.mark.asyncio
async def test_segments_within_the_grace_window_become_one_question(tmp_path, monkeypatch):
    monkeypatch.setenv("STACKCHAN_WAKE_PHRASE", "你好妮妮")
    llm = FakeLLM([_chunk("[happy]先别急。")], [_chunk("[happy]好的。")])
    bridge, _audio, _done = _bridge(tmp_path, llm, [], monkeypatch)
    _grace(bridge, 40)
    await bridge.start()
    await bridge._on_transcript("你好，妮妮。我是说。")
    assert bridge._turn_task is not None      # DeepSeek starts at once...
    bridge._on_speech_started()               # ...but the user is still talking
    await asyncio.sleep(0.06)
    assert not bridge._may_speak.is_set() and _audio.await_count == 0   # so its voice is held
    await bridge._on_transcript("帮我查一下宁波的天气。")
    await asyncio.sleep(0.08)
    await bridge._turn_task
    assert llm.requests[-1]["messages"][-1] == {"role": "user", "content": "我是说。帮我查一下宁波的天气。"}
    assert bridge._memory.recent(10)[0]["content"] == "我是说。帮我查一下宁波的天气。"   # held turn not remembered
    await bridge.stop()


@pytest.mark.asyncio
async def test_talking_before_the_reply_plays_restarts_with_the_full_question(tmp_path, monkeypatch):
    gate = asyncio.Event()

    class GatedLLM(FakeLLM):
        async def _create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                await gate.wait()
            return FakeStream([_chunk("[happy]明白了。")])

    llm = GatedLLM()
    spoken = []
    bridge, _audio, _done = _bridge(tmp_path, llm, spoken, monkeypatch)
    _grace(bridge)
    await bridge.start()
    await bridge._start_turn("给我推荐")
    await asyncio.sleep(0)
    bridge._on_speech_started()
    await bridge._on_transcript("一部电影。")
    await asyncio.sleep(0.05)
    await bridge._turn_task
    assert llm.requests[-1]["messages"][-1]["content"] == "给我推荐一部电影。"
    assert spoken == ["明白了。"]
    rows = bridge._memory.recent(10)
    assert rows == [{"role": "user", "content": "给我推荐一部电影。"},
                    {"role": "assistant", "content": "明白了。"}]
    await bridge.stop()


@pytest.mark.asyncio
async def test_unplayed_reply_waits_while_user_talks_then_resumes_on_filler(tmp_path, monkeypatch):
    llm = FakeLLM([_chunk("[happy]好呀。")])
    spoken = []
    bridge, _audio, _done = _bridge(tmp_path, llm, spoken, monkeypatch)
    await bridge.start()
    bridge._may_speak.clear()            # as if speech started just before the reply
    bridge._turn_task = asyncio.create_task(bridge._run_turn("讲个笑话"))
    await asyncio.sleep(0.02)
    assert spoken == []                  # held while the user is talking
    await bridge._on_transcript("嗯。")   # only a filler: let the reply play
    await bridge._turn_task
    assert spoken == ["好呀。"]
    await bridge.stop()


# ---- understanding tag, ASR context, prompt caching -----------------------------------


def test_reply_head_waits_for_both_tags_and_hides_the_understanding():
    from stackchan_mcp.deepseek_pipeline_bridge import clean_for_speech, parse_reply_head

    assert parse_reply_head("[happy]")[0] is False                       # an understanding tag may follow
    assert parse_reply_head("[happy][理解:说说章")[0] is False            # tag still open
    ready, mood, understood, rest = parse_reply_head("[happy][理解:说说章鱼]好，讲章鱼。")
    assert (ready, mood, understood, rest) == (True, "happy", "说说章鱼", "好，讲章鱼。")
    assert parse_reply_head("[neutral]好的。") == (True, "neutral", None, "好的。")
    assert parse_reply_head("[happy]", final=True) == (True, "happy", None, "")
    assert clean_for_speech("嗯[理解:这是一句很长很长很长很长很长很长的理解内容]好") == "嗯好"


@pytest.mark.asyncio
async def test_understanding_is_logged_remembered_and_not_spoken(tmp_path, monkeypatch, caplog):
    import logging

    llm = FakeLLM([_chunk("[happy][理解:"), _chunk("说说章鱼]"), _chunk("好，讲章鱼。")])
    spoken = []
    bridge, _audio, _done = _bridge(tmp_path, llm, spoken, monkeypatch)
    await bridge.start()
    with caplog.at_level(logging.INFO, logger="stackchan_mcp.deepseek_pipeline_bridge"):
        await bridge._start_turn("四斗的章鱼。")
        await bridge._turn_task
    assert "".join(spoken) == "好，讲章鱼。"
    assert "understood as: heard=四斗的章鱼。 understood=说说章鱼" in caplog.text
    assert bridge._memory.recent(2)[0]["content"] == "四斗的章鱼。（妮妮理解为：说说章鱼）"
    await bridge.stop()


def test_asr_context_is_a_bare_word_list_without_the_last_reply(tmp_path, monkeypatch):
    # Qwen ASR echoed corpus sentences (and the previous reply) back as "user said".
    from stackchan_mcp.console_settings import get_console_settings

    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)
    get_console_settings().update({"asr_hotwords": "英雄联盟 羊了个羊"})
    bridge._last_reply = "章鱼有三颗心脏。"
    corpus = bridge._asr_session_update()["session"]["input_audio_transcription"]["corpus"]["text"]
    assert corpus == "妮妮 你好妮妮 英雄联盟 羊了个羊"


@pytest.mark.parametrize("transcript, echo", [
    ("对，就是这个意思。协议到二零二七年年底，到时候还要重新谈。", True),    # the reply, read back
    ("妮妮 你好妮妮 英雄联盟 羊了个羊", True),                                   # the corpus, read back
    ("那二零二七年以后呢？", False),                                              # a real follow-up
    ("好的。", False),
    ("嗯，那换一个热闹点的。", False),                                            # quotes a short phrase of the reply
])
def test_transcripts_that_repeat_the_reply_or_context_are_dropped(tmp_path, monkeypatch, transcript, echo):
    from stackchan_mcp.console_settings import get_console_settings

    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)
    get_console_settings().update({"asr_hotwords": "英雄联盟 羊了个羊"})
    bridge._last_reply = "对，就是这个意思。协议到二零二七年年底，到时候还要重新谈，能不能续上得看那时候的谈判。下次换一个热闹点的。"
    assert bridge._is_echo(transcript) is echo


@pytest.mark.asyncio
async def test_settings_change_reconnects_asr_instead_of_a_mid_session_update(tmp_path, monkeypatch):
    bridge, _audio, _done = _bridge(tmp_path, FakeLLM(), [], monkeypatch)
    ws = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
    bridge._asr_ws = ws
    bridge._on_settings_changed({"asr_hotwords": "王者荣耀"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    ws.send.assert_not_awaited()        # Qwen closes the socket (1007) on a second session.update
    ws.close.assert_awaited_once()
    assert bridge._asr_ws is None


@pytest.mark.asyncio
async def test_system_prompt_stays_identical_so_the_prefix_cache_hits(tmp_path, monkeypatch):
    llm = FakeLLM([_chunk("[happy]好。")])
    bridge, _audio, _done = _bridge(tmp_path, llm, [], monkeypatch)
    first = bridge._system_prompt()
    monkeypatch.setattr("time.strftime", lambda *_a, **_k: "2099-01-01 00:00 Friday")
    assert bridge._system_prompt() == first and "现在时间" not in first
    await bridge.start()
    await bridge._start_turn("几点了")
    await bridge._turn_task
    assert "2099-01-01 00:00" in llm.requests[0]["messages"][-2]["content"]   # the clock rides at the end
    await bridge.stop()
