"""ASR → DeepSeek → TTS voice pipeline behind the GeminiLiveBridge surface.

DeepSeek has no realtime voice API, so this bridge chains three services:

* Qwen3-ASR-Flash-Realtime (DashScope WebSocket) turns the wake-gated device
  PCM into text; its server VAD decides when the user has finished a sentence.
* DeepSeek (OpenAI-compatible Chat Completions, thinking disabled) writes a
  short spoken reply that starts with an ``[mood]`` tag and may call the
  shared voice tools (head, face, LEDs, end_conversation).
* ElevenLabs ``eleven_v4_turbo`` over the Text to Dialogue WebSocket speaks it
  as 24 kHz PCM, one dialogue socket per reply; edge-tts takes over when
  ElevenLabs fails or the plan's character quota passes the fallback ratio.

The proxy (:class:`GeminiVoiceProxy`) keeps owning the wake word, chimes,
LEDs and Opus encoding; it only sees ``send_audio`` going in and ``on_audio``
coming out. The first version is half-duplex: microphone audio is dropped
while a reply is being spoken, so the robot never transcribes its own voice.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import difflib
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .console_settings import ConsoleSettings, get_console_settings

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DASHSCOPE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_ASR_MODEL = "qwen3-asr-flash-realtime"
QWEN_ASR_URL = f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={QWEN_ASR_MODEL}"
ELEVENLABS_API = "https://api.elevenlabs.io"
ELEVENLABS_DIALOGUE_WS = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
# Flush buffered text when the reply stream pauses this long (fillers before a
# tool call); keep the socket alive under its 20 s inactivity timeout.
ELEVEN_PAUSE_FLUSH_S = 0.8
ELEVEN_KEEPALIVE_S = 8.0
# The reply's opening sentence (at least this many characters) is flushed alone
# for a fast first sound; the rest of the reply streams as one segment.
ELEVEN_OPENING_MIN_CHARS = 8
ECHO_MIN_RUN_CHARS = 12
EDGE_TTS_VOICE = "zh-CN-XiaoyiNeural"
OUTPUT_RATE = 24_000
MOODS = ("excited", "happy", "neutral", "tired", "sad", "curious", "apologetic")
# Shared tools that make sense for a spoken DeepSeek turn. express_emotion is
# replaced by the reply's leading [mood] tag (no extra model round trip), and
# the agent hand-off tools need CLIs this setup does not install.
EXCLUDED_TOOLS = {"express_emotion", "ask_claude", "ask_grokbot"}
MAX_TOOL_ROUNDS = 3
SEARCH_FILLER = "[curious]我查一下哦。"
FETCH_FILLER = "[curious]我打开看看。"
THINK_FILLER = "[curious]我想想。"
DEEPSEEK_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
DEEPSEEK_FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"
SEARCH_RESULTS_FOR_MODEL = 5
FETCH_TIMEOUT_S = 15.0
FETCH_MAX_CHARS = 6000
# Questions that read like reasoning, maths, planning or analysis get DeepSeek's
# thinking mode; chit-chat and look-ups stay on the fast path.
# Only explicit reasoning work: everyday "why / any advice / explain" questions
# stay on the fast path, since thinking adds seconds before the first word.
DEEP_THINKING_HINTS = re.compile(
    r"推理|证明|推导|算一下|算算|帮我算|计算一下|数学题|应用题|逻辑题|谜题|脑筋急转弯|"
    r"帮我规划|帮我计划|做个计划|制定.{0,4}(?:计划|方案)|对比一下|比较一下|权衡|利弊|优缺点|"
    r"一步一步|仔细想想|好好想想|深入分析|详细分析|"
    r"[0-9零一二三四五六七八九十百千万]+\s*(?:乘以?|除以|加上?|减去?|[×÷*/+\-^])\s*[0-9零一二三四五六七八九十百千万]"
)
ASR_MAX_BACKOFF_S = 30.0
HOLD_REPLY_MAX_S = 6.0
ASR_OPEN_TIMEOUT_S = 6.0
# Speech captured while ASR is (re)connecting is buffered, not dropped.
ASR_BUFFER_MAX_BYTES = 16_000 * 2 * 10
# Only tell the user ASR is unreachable once speech has waited this long.
ASR_ANNOUNCE_AFTER_S = 6.0
SENTENCE_END = re.compile(r"[。！？!?…~\n]")
SOFT_BREAK = re.compile(r"[，,；;、]")
MOOD_TAG = re.compile(r"^\s*\[(?:emotion:)?\s*([a-zA-Z_]+)\s*\]\s*")
# Hidden "[理解:…]" tag: DeepSeek's reading of a misrecognised transcript.
UNDERSTOOD_TAG = re.compile(r"^\s*\[\s*理解\s*[:：]\s*([^\]]{0,240})\]\s*")

DEFAULT_PERSONA = """你是妮妮，一个住在用户桌上的小机器人（Stack-chan），是用户聪明又贴心的桌边伙伴。
你的名字叫妮妮。之前的对话里你可能自称 D4C，那是旧名字，不要再用；用户叫你「你好妮妮」时就是在叫你。
你的大脑是 {model_label}，知识面很广，能推理、算数、解释概念、写东西、出主意。用户问你用的什么模型时，如实告诉他。
你说的话会被合成语音念出来，所以：
- 用口语化的简体中文，像面对面聊天；不用列表、标题、Markdown、表情符号、代码块，也不要念网址。
- 默认简短：闲聊、简单问题和一般的提问，说两到四句，一共不超过 {long_chars} 个字（念出来十几秒），先说最重要的，其余的等用户追问再说。
- 只有用户明确要你展开时（讲个故事、详细说说、展开讲讲、一步一步教我），才讲长一些，最多三百字左右；讲完问一句要不要接着讲。
- 用户可以点一下你的屏幕或摸一下你的头来打断你。
用户的话来自语音识别，可能有同音错字、漏字或断句不对：
- 结合发音和前面的对话去理解用户真正想说什么，不要被错字带偏，也不要指出用户的错字。
- 识别成「心姨」「宁宁」「倪妮」「腻腻」之类的，都是在叫你的名字。
- 关键的词拿不准、而且猜错会答非所问时，先用一句话确认（「你是说……吗？」），不要悄悄乱猜。
- 只有当你按发音把识别结果纠正成了别的意思时，才在情绪标签后面紧接着写 [理解:你理解的完整句子]，这个标签不会被念出来；识别没问题就不要写。
- 每次回复的最开头写一个情绪标签，从 [excited] [happy] [neutral] [tired] [sad] [curious] [apologetic] 里选一个，标签后面直接接要说的话。
你会用工具：
- 天气、新闻、价格、比分、最新的产品或模型这类需要实时信息的问题，或者你不确定的事实，先调用 web_search 查，再根据结果回答；不要说自己不能上网。
- 搜索结果的摘要不够时，可以用 web_fetch 打开最相关的那个网页读原文。
- 用户没说城市却问天气时，先问他在哪个城市。
- 回答里提到信息来源时，说网站名字就好，不要念网址。
- 需要点头、摇头、转头、换表情或亮灯时调用对应工具；用户说再见或不想聊了就调用 end_conversation。
- 查过还不确定就坦白说，不要编造。"""
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web (DeepSeek search). Returns ranked results with title, site, "
            "date, url and a long summary. Use for weather, news, prices, scores, recent products "
            "or models, and any time-sensitive or uncertain fact. Put the city or date into the "
            "query when it matters; keep the query under 128 characters."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up, in Chinese"}},
            "required": ["query"],
        },
    },
}
WEB_FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Open one web page (usually a url from web_search results) and read its main text "
            "as Markdown. Use when the search summary is not enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "http(s) URL to read"}},
            "required": ["url"],
        },
    },
}


def log_api_call(provider: str, model: str, purpose: str, status: str, took: float, **detail: Any) -> None:
    """One uniform line per external API call; the monitor page lists these."""
    extra = "".join(f" {key}={value}" for key, value in detail.items() if value not in (None, ""))
    logger.info("API call: provider=%s model=%s purpose=%s status=%s took=%.2fs%s",
                provider, model, purpose, status, took, extra)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _tag_still_open(text: str) -> bool:
    """True while the reply head may still be inside (or before) a leading tag."""
    stripped = text.lstrip()
    if not stripped:
        return True
    if not stripped.startswith("[") or "]" in stripped:
        return False
    if stripped.startswith(("[理", "[ 理")) or len(stripped) < 3:
        return len(stripped) < 240
    return len(stripped) < 24


def parse_reply_head(head: str, *, final: bool = False) -> tuple[bool, str | None, str | None, str]:
    """Split the leading ``[mood]`` and optional ``[理解:…]`` tags off a reply.

    Returns (ready, mood, understood, rest). While streaming, ``ready`` is False
    until both tags are known to be complete, so no tag text reaches the TTS.
    """
    if not final and _tag_still_open(head):
        return False, None, None, head
    mood, rest = split_mood(head)
    if not final and _tag_still_open(rest):
        return False, None, None, head
    understood = None
    match = UNDERSTOOD_TAG.match(rest)
    if match:
        understood = match.group(1).strip() or None
        rest = rest[match.end():]
    return True, mood, understood, rest


def split_mood(text: str) -> tuple[str | None, str]:
    """Strip a leading ``[mood]`` tag; return (mood or None, remaining text)."""
    match = MOOD_TAG.match(text)
    if not match:
        return None, text
    mood = match.group(1).lower()
    return (mood if mood in MOODS else None), text[match.end():]


_PUNCT = "\\s，,。．.！!？?、；;：:…~～\"'“”‘’（）()"
FILLER_ONLY = re.compile(r"^(?:嗯|啊|呃|哦|噢|额|唔|哎|诶|欸|喔|哈|嘿|呀)+$")
# Common ASR spellings of the robot's name, so "你好，倪妮。" is still the wake word.
NAME_HOMOPHONES = {"妮妮": ("妮妮", "倪妮", "尼尼", "泥泥", "霓霓", "妮尼", "倪倪", "腻腻", "呢呢")}


def wake_aliases(wake_phrase: str, robot_name: str) -> list[str]:
    """Transcript spellings that mean "the wake word", longest first."""
    names = NAME_HOMOPHONES.get(robot_name, (robot_name,))
    aliases = {wake_phrase} if wake_phrase else set()
    for name in names:
        aliases.update({f"你好{name}", f"您好{name}", f"嗨{name}", f"嘿{name}", name})
    return sorted((a for a in aliases if a), key=len, reverse=True)


def strip_wake_phrase(text: str, aliases: list[str]) -> str:
    """Drop a leading wake word (punctuation allowed between its characters)."""
    gap = f"[{_PUNCT}]*"
    for alias in aliases:
        pattern = "^" + gap + "".join(re.escape(ch) + gap for ch in alias)
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return text[match.end():]
    return text


def is_filler_only(text: str) -> bool:
    """True for empty / punctuation-only / pure hesitation transcripts ("嗯。")."""
    compact = re.sub(f"[{_PUNCT}]", "", text)
    return not compact or bool(FILLER_ONLY.match(compact))


def has_speech_content(text: str) -> bool:
    """True when a transcript holds at least one letter, digit or CJK character."""
    return any(ch.isalnum() for ch in text)


def clean_for_speech(text: str) -> str:
    """Remove markup the TTS would read aloud (tags, Markdown, emoji)."""
    text = re.sub(r"\[\s*理解\s*[:：][^\]]*\]", "", text)
    text = re.sub(r"\[[^\]]{1,24}\]", "", text)
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    return text.strip()


class SentenceSplitter:
    """Cut a streaming reply into speakable pieces as early as is natural."""

    def __init__(self, soft_break_after: int = 16) -> None:
        self._buffer = ""
        self._soft_break_after = soft_break_after

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        pieces: list[str] = []
        while True:
            match = SENTENCE_END.search(self._buffer)
            if match is None and len(self._buffer) >= self._soft_break_after:
                match = SOFT_BREAK.search(self._buffer, self._soft_break_after // 2)
            if match is None:
                break
            piece, self._buffer = self._buffer[: match.end()], self._buffer[match.end():]
            if piece.strip():
                pieces.append(piece)
        return pieces

    def flush(self) -> list[str]:
        piece, self._buffer = self._buffer, ""
        return [piece] if piece.strip() else []


def openai_tools(declarations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Map shared voice-tool declarations to Chat Completions tools.

    Returns the tool list and an alias → original-name map (DeepSeek, like
    OpenAI, rejects dots in function names, e.g. ``self.tracking.start``).
    """
    tools: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for declaration in declarations:
        name = declaration["name"]
        if name in EXCLUDED_TOOLS:
            continue
        alias = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        if alias in aliases:
            raise ValueError(f"voice tool alias collision: {alias}")
        aliases[alias] = name
        tools.append({
            "type": "function",
            "function": {
                "name": alias,
                "description": declaration.get("description", ""),
                "parameters": declaration.get("parameters") or {"type": "object", "properties": {}},
            },
        })
    return tools, aliases


class MemoryStore:
    """Conversation turns persisted in SQLite so the robot remembers across restarts."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, "
            "role TEXT NOT NULL, content TEXT NOT NULL)"
        )
        self._db.commit()

    def add(self, role: str, content: str) -> None:
        self._db.execute(
            "INSERT INTO turns (created_at, role, content) VALUES (?, ?, ?)",
            (time.time(), role, content),
        )
        self._db.commit()

    def recent(self, limit: int) -> list[dict[str, str]]:
        rows = self._db.execute(
            "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def close(self) -> None:
        self._db.close()


class QuotaGuard:
    """Decide between ElevenLabs and edge-tts from the plan's character usage."""

    def __init__(self, api_key: str, ratio: float, refresh_s: float = 300.0) -> None:
        self._api_key = api_key
        self._ratio = ratio
        self._refresh_s = refresh_s
        self._checked_at = 0.0
        self._over = False

    async def over_quota(self) -> bool:
        """Answer from the last check at once; refresh in the background when stale.

        The check is an HTTP round trip, and a turn waiting on it before its
        first word was a needless second or two of silence.
        """
        if time.monotonic() - self._checked_at >= self._refresh_s:
            self._checked_at = time.monotonic()
            task = asyncio.create_task(self._refresh())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return self._over

    async def _refresh(self) -> None:
        started = time.monotonic()
        try:
            import aiohttp  # noqa: PLC0415

            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(
                headers={"xi-api-key": self._api_key}, timeout=timeout
            ) as session:
                async with session.get(f"{ELEVENLABS_API}/v1/user/subscription") as resp:
                    data = await resp.json()
            used, limit = data.get("character_count"), data.get("character_limit")
            if isinstance(used, int) and isinstance(limit, int) and limit > 0:
                self._over = used / limit >= self._ratio
                logger.info("ElevenLabs usage %d/%d (%.1f%%)", used, limit, 100 * used / limit)
            log_api_call("elevenlabs", "user/subscription", "quota", "ok", time.monotonic() - started,
                         used=used, limit=limit)
        except Exception as exc:
            logger.warning("ElevenLabs quota check failed (keeping last answer): %s", exc)
            log_api_call("elevenlabs", "user/subscription", "quota", "failed", time.monotonic() - started,
                         error=type(exc).__name__)


class DeepSeekPipelineBridge:
    """Qwen ASR → DeepSeek → ElevenLabs, exposed through the voice-bridge interface."""

    def __init__(
        self,
        *,
        on_audio: Callable[[bytes], Awaitable[None]],
        on_turn_complete: Callable[[], Awaitable[None]],
        on_interrupt: Callable[[], Awaitable[None]],
        on_session_dead: Callable[[], Awaitable[None]],
        esp32: Any = None,
        api_key: str | None = None,
        response_modality: str | None = None,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_end_conversation: Callable[[], Awaitable[None]] | None = None,
        usb_transport: Any = None,
        on_head_command: Callable[[], None] | None = None,
        wake_gate_state_getter: Callable[[], str] | None = None,
        set_face_tracking: Any = None,
        debug_status: Any = None,
        tool_executor: Any = None,
        tool_executor_factory: Any = None,
        tool_extensions: tuple = (),
        llm_client: Any = None,
        tts_synthesize: Callable[[str, str | None], Any] | None = None,
        asr_connect: Callable[[], Awaitable[Any]] | None = None,
        memory: MemoryStore | None = None,
        settings: ConsoleSettings | None = None,
    ) -> None:
        del response_modality  # audio always comes from our own TTS
        self._settings = settings or get_console_settings()
        # Tests (and callers) may pin a value without touching the settings file.
        self._overrides: dict[str, Any] = {}
        self._on_audio = on_audio
        self._on_turn_complete = on_turn_complete
        self._on_interrupt = on_interrupt
        self._on_session_dead = on_session_dead
        self._on_text = on_text
        self._on_end_conversation = on_end_conversation
        # Set by the voice proxy: called with True when a reply could not be voiced
        # (ElevenLabs unreachable), False once speech works again.
        self.on_voice_error: Callable[[bool], Awaitable[None]] | None = None
        self._voice_failed = False
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._model_label = os.getenv("STACKCHAN_MODEL_LABEL", "DeepSeek V4.1 Flash")
        # ~4.5 spoken Chinese characters per second: 80 characters is under 20 seconds.
        self._long_chars = _env_int("STACKCHAN_REPLY_LONG_CHARS", 80)
        self._search_model = os.getenv("STACKCHAN_SEARCH_MODEL", "qwen-flash")
        self._search_client: Any = None
        self._last_served: str | None = None
        self._last_usage: dict[str, Any] | None = None
        self._max_chars = _env_int("STACKCHAN_REPLY_MAX_CHARS", 40)
        self._history_limit = _env_int("STACKCHAN_MEMORY_TURNS", 40)
        self._reasoning_effort = os.getenv("STACKCHAN_DEEPSEEK_REASONING_EFFORT", "high")
        self._current_question = ""
        self._wake_phrase = os.getenv("STACKCHAN_WAKE_PHRASE", "").strip()
        self._pending_parts: list[str] = []
        self._turn_timer: asyncio.Task[None] | None = None
        self._turn_question = ""
        self._turn_understood: str | None = None
        # Cleared while the user speaks over a reply that has not started playing.
        self._may_speak = asyncio.Event()
        self._may_speak.set()
        self._hold_timer: asyncio.Task[None] | None = None
        self._dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
        self._eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._quota = (
            QuotaGuard(self._eleven_key, _env_float("STACKCHAN_TTS_QUOTA_FALLBACK", 0.8))
            if self._eleven_key else None
        )
        self._llm = llm_client
        self._tts_synthesize = tts_synthesize
        self._asr_connect = asr_connect or self._connect_qwen_asr
        self._memory = memory
        self._executor = tool_executor
        self._executor_factory = tool_executor_factory
        self._executor_options = dict(
            esp32=esp32,
            usb_transport=usb_transport,
            on_head_command=on_head_command,
            wake_gate_state_getter=wake_gate_state_getter,
            set_face_tracking=set_face_tracking,
            on_end_conversation=on_end_conversation,
            debug_status=debug_status,
            extensions=tool_extensions,
            announce_result=self._announce_result,
        )
        self._tools: list[dict[str, Any]] = []
        self._tool_aliases: dict[str, str] = {}
        self._connected_event = asyncio.Event()
        self._asr_ws: Any = None
        self._asr_reader: asyncio.Task[None] | None = None
        self._asr_lock = asyncio.Lock()
        self._asr_keeper: asyncio.Task[None] | None = None
        self._asr_poke = asyncio.Event()
        self._asr_dropped = asyncio.Event()
        self._asr_buffer: collections.deque[bytes] = collections.deque()
        self._asr_buffer_bytes = 0
        self._asr_buffered_since: float | None = None
        self._asr_failure_announced = False
        self._utterance_started: float | None = None
        self._last_reply = ""
        self._asr_stale = False
        self._last_speech_stopped: float | None = None
        self._last_asr_done: float | None = None
        # Per-turn latency stamps (monotonic seconds), logged at the first sound.
        self._timing: dict[str, float | None] | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._speaking = False
        self._end_requested = False
        self.running = False

    # ---- console settings ----------------------------------------------------

    def _setting(self, key: str) -> Any:
        return self._overrides[key] if key in self._overrides else self._settings.get(key)

    def _pin(key: str, doc: str):  # noqa: N805 - property factory
        def getter(self):
            return self._setting(key)

        def setter(self, value):
            self._overrides[key] = value

        return property(getter, setter, doc=doc)

    _model = _pin("llm_model", "DeepSeek model name")
    _robot_name = _pin("robot_name", "the robot's name")
    _thinking_mode = _pin("thinking", "auto / on / off")
    _turn_grace_ms = _pin("turn_grace_ms", "wait after a sentence before replying")
    _vad_silence_ms = _pin("vad_silence_ms", "ASR end-of-speech silence")
    _voice_id = _pin("eleven_voice_id", "ElevenLabs voice")
    _tts_model = _pin("tts_model", "ElevenLabs model")
    del _pin

    @property
    def _wake_aliases(self) -> list[str]:
        return wake_aliases(self._wake_phrase, self._robot_name)

    def _on_settings_changed(self, changed: dict[str, Any]) -> None:
        if changed.keys() & {"vad_silence_ms", "asr_hotwords", "robot_name"} and self._asr_ws is not None:
            # Qwen ASR rejects session.update once a session has started (close
            # 1007), so new settings apply by reconnecting, between turns.
            self._asr_stale = True
            if not self._turn_in_flight():
                asyncio.get_running_loop().create_task(self._refresh_stale_asr())

    def _asr_context(self) -> str:
        """Context biasing for Qwen ASR: a bare word list, never sentences.

        Qwen ASR echoes corpus sentences back as the transcript of quiet or
        short audio, so no prose and no previous reply goes in here.
        """
        words = [self._robot_name, f"你好{self._robot_name}"]
        words += (self._setting("asr_hotwords") or "").split()
        return " ".join(dict.fromkeys(w for w in words if w))

    def _asr_session_update(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm",
                "sample_rate": 16000,
                "input_audio_transcription": {"language": "zh", "corpus": {"text": self._asr_context()}},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.2,
                    "silence_duration_ms": int(self._vad_silence_ms),
                },
            },
        }

    # ---- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if not self._api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        if self._llm is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            self._llm = AsyncOpenAI(api_key=self._api_key, base_url=DEEPSEEK_BASE_URL)
        if self._memory is None:
            db = os.getenv("STACKCHAN_MEMORY_DB") or str(
                Path(__file__).resolve().parents[1] / "data" / "memory.sqlite"
            )
            self._memory = MemoryStore(Path(db).expanduser())
        if self._executor is None:
            from .voice_tools import VoiceToolExecutor  # noqa: PLC0415

            self._executor = (self._executor_factory or VoiceToolExecutor)(**self._executor_options)
        self._tools, self._tool_aliases = openai_tools(self._executor.declarations())
        self._tools.extend([WEB_SEARCH_TOOL, WEB_FETCH_TOOL])
        self._tool_aliases["web_search"] = "web_search"
        self._tool_aliases["web_fetch"] = "web_fetch"
        logger.info(
            "DeepSeek pipeline ready: model=%s tts=%s voice=%s tools=%s",
            self._model, self._tts_model, self._voice_id or "-", ",".join(sorted(self._tool_aliases)),
        )
        self.running = True
        self._connected_event.set()
        self._settings.add_listener(self._on_settings_changed)
        # Keep the ASR socket warm so a wake word never waits on a handshake.
        self._asr_keeper = asyncio.create_task(self._keep_asr_warm())

    async def wait_connected(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    async def stop(self) -> None:
        self.running = False
        self._connected_event.clear()
        self._settings.remove_listener(self._on_settings_changed)
        self._clear_pending_turn()
        await self._cancel_turn()
        keeper, self._asr_keeper = self._asr_keeper, None
        if keeper is not None:
            keeper.cancel()
            try:
                await keeper
            except (asyncio.CancelledError, Exception):
                pass
        await self._close_asr()
        close = getattr(self._executor, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    # ---- audio in --------------------------------------------------------------

    async def send_audio(self, pcm_16khz: bytes) -> None:
        """Wake-gated 16 kHz PCM from the device; dropped while the robot speaks."""
        if not self.running or self._speaking or not pcm_16khz:
            return
        ws = self._asr_ws
        if ws is None:
            await self._buffer_audio(pcm_16khz)
            return
        await self._send_pcm(ws, pcm_16khz)

    async def _send_pcm(self, ws: Any, pcm_16khz: bytes) -> bool:
        try:
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_16khz).decode(),
            }))
            return True
        except Exception as exc:
            logger.warning("Qwen ASR send failed, reconnecting: %s", exc)
            await self._close_asr()
            return False

    async def _buffer_audio(self, pcm_16khz: bytes) -> None:
        if self._asr_buffered_since is None:
            self._asr_buffered_since = time.monotonic()
        self._asr_buffer.append(pcm_16khz)
        self._asr_buffer_bytes += len(pcm_16khz)
        while self._asr_buffer_bytes > ASR_BUFFER_MAX_BYTES:
            self._asr_buffer_bytes -= len(self._asr_buffer.popleft())
        self._asr_poke.set()  # retry now instead of waiting out the back-off
        waited = time.monotonic() - self._asr_buffered_since
        if waited >= ASR_ANNOUNCE_AFTER_S and not self._asr_failure_announced:
            self._asr_failure_announced = True
            self._clear_audio_buffer()
            logger.warning("Qwen ASR still unreachable after %.1fs of speech", waited)
            await self.inject_user_text("我现在连不上语音识别，等一下再叫我吧。")

    def _clear_audio_buffer(self) -> None:
        self._asr_buffer.clear()
        self._asr_buffer_bytes = 0
        self._asr_buffered_since = None

    async def send_keepalive_audio(self, pcm_16khz: bytes) -> None:
        # Nothing to keep alive here: the ASR keeper holds its own socket open.
        del pcm_16khz

    async def send_audio_stream_end(self) -> None:
        return None

    async def clear_conversation_context(self) -> None:
        """Wake window closed: drop unsent audio; the warm ASR socket stays open."""
        self._asr_failure_announced = False
        self._clear_audio_buffer()
        self._clear_pending_turn()

    async def interrupt(self) -> None:
        """Device barge-in (screen tap / head touch sends ``abort``).

        The CoreS3 mic is off while the speaker plays, so a wake word cannot
        interrupt a reply; the firmware sends ``abort`` instead. Stop the turn
        and also drop audio the proxy already queued for the device.
        """
        busy = self._speaking or (self._turn_task is not None and not self._turn_task.done())
        await self._cancel_turn()
        if busy:
            logger.info("reply interrupted by the device (abort)")
            await self._on_interrupt()

    async def inject_user_text(self, text: str) -> bool:
        """MCP ``say``: speak ``text`` as written, without asking the LLM."""
        if not self.running:
            return False
        await self._cancel_turn()
        self._turn_task = asyncio.create_task(self._speak_verbatim(text))
        return True

    # ---- Qwen ASR -----------------------------------------------------------------

    async def _connect_qwen_asr(self) -> Any:
        import websockets  # noqa: PLC0415

        if not self._dashscope_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        headers = {"Authorization": f"Bearer {self._dashscope_key}"}
        try:
            ws = await websockets.connect(
                QWEN_ASR_URL, additional_headers=headers, max_size=None,
                open_timeout=ASR_OPEN_TIMEOUT_S, proxy=None,
            )
        except Exception as exc:
            logger.info("Qwen ASR direct connect failed (%s); trying the system proxy", exc)
            ws = await websockets.connect(
                QWEN_ASR_URL, additional_headers=headers, max_size=None,
                open_timeout=ASR_OPEN_TIMEOUT_S,
            )
        await ws.send(json.dumps(self._asr_session_update()))
        logger.info("Qwen ASR session set: silence=%d ms context=%s",
                    int(self._vad_silence_ms), self._asr_context())
        return ws

    async def _keep_asr_warm(self) -> None:
        delay = 1.0
        while self.running:
            if self._asr_ws is None:
                if await self._connect_asr_once():
                    delay = 1.0
                    continue
                self._asr_poke.clear()
                try:
                    await asyncio.wait_for(self._asr_poke.wait(), delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, ASR_MAX_BACKOFF_S)
                continue
            self._asr_dropped.clear()
            await self._asr_dropped.wait()

    async def _connect_asr_once(self) -> bool:
        async with self._asr_lock:
            if self._asr_ws is not None:
                return True
            started = time.monotonic()
            try:
                ws = await self._asr_connect()
            except Exception as exc:
                logger.warning("Qwen ASR connect failed: %s: %s", type(exc).__name__, exc)
                log_api_call("dashscope", QWEN_ASR_MODEL, "asr-connect", "failed", time.monotonic() - started,
                             error=type(exc).__name__)
                return False
            # Flush speech captured while connecting, in order, before going live.
            while self._asr_buffer:
                chunk = self._asr_buffer.popleft()
                self._asr_buffer_bytes -= len(chunk)
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))
            self._asr_buffered_since = None
            self._asr_ws = ws
            self._asr_reader = asyncio.create_task(self._read_asr(ws))
            logger.info("Qwen ASR connected in %.2fs", time.monotonic() - started)
            log_api_call("dashscope", QWEN_ASR_MODEL, "asr-connect", "ok", time.monotonic() - started)
            return True

    async def _refresh_stale_asr(self) -> None:
        if self._asr_stale and self._asr_ws is not None:
            self._asr_stale = False
            logger.info("Qwen ASR settings changed; reconnecting to apply them")
            await self._close_asr()

    async def _close_asr(self) -> None:
        ws, self._asr_ws = self._asr_ws, None
        reader, self._asr_reader = self._asr_reader, None
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        self._asr_dropped.set()

    async def _read_asr(self, ws: Any) -> None:
        try:
            async for raw in ws:
                event = json.loads(raw)
                kind = event.get("type", "")
                if kind == "input_audio_buffer.speech_stopped":
                    self._last_speech_stopped = time.monotonic()
                elif kind == "input_audio_buffer.speech_started":
                    self._utterance_started = time.monotonic()
                    self._on_speech_started()
                elif kind == "conversation.item.input_audio_transcription.completed":
                    text = (event.get("transcript") or "").strip()
                    took = time.monotonic() - self._utterance_started if self._utterance_started else 0.0
                    self._utterance_started = None
                    self._last_asr_done = time.monotonic()
                    log_api_call("dashscope", QWEN_ASR_MODEL, "asr", "ok" if text else "empty", took,
                                 chars=len(text))
                    await self._on_transcript(text)
                elif kind == "error":
                    logger.warning("Qwen ASR error: %s", event.get("error"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info("Qwen ASR stream closed: %s", exc)
        finally:
            if self._asr_ws is ws:
                self._asr_ws = None
                self._asr_dropped.set()

    # ---- turn taking ------------------------------------------------------------------

    def _turn_in_flight(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    def _on_speech_started(self) -> None:
        """The user is (still) talking: don't reply yet, and hold an unplayed reply."""
        if self._turn_timer is not None:
            self._turn_timer.cancel()
            self._turn_timer = None
        if self._turn_in_flight() and not self._speaking:
            self._may_speak.clear()
            if self._hold_timer is not None:
                self._hold_timer.cancel()
            self._hold_timer = asyncio.create_task(self._release_hold_later())

    async def _release_hold_later(self) -> None:
        # Never hold a reply forever if the ASR never finishes the utterance.
        await asyncio.sleep(HOLD_REPLY_MAX_S)
        self._may_speak.set()

    def _release_hold(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.cancel()
            self._hold_timer = None
        self._may_speak.set()

    async def _on_transcript(self, text: str) -> None:
        """One finished ASR segment: strip the wake word, merge segments, then reply."""
        cleaned = strip_wake_phrase(text, self._wake_aliases).strip()
        if self._is_echo(cleaned):
            logger.info("ignoring transcript that repeats the reply or ASR context: %s", text)
            return
        if self._turn_in_flight():
            if self._speaking:
                logger.info("dropping transcript while a reply is in progress: %s", text)
                return
            if is_filler_only(cleaned):
                self._release_hold()
                return
            # The reply has not started playing: fold the new words into the question.
            logger.info("user kept talking before the reply started; restarting with the full question")
            previous = self._turn_question
            await self._cancel_turn()
            self._release_hold()
            self._pending_parts = [previous] if previous else []
        if is_filler_only(cleaned):
            if text.strip():
                logger.info("ignoring wake word / filler only transcript: %r", text)
            if self._pending_parts:
                await self._schedule_turn()
            return
        self._pending_parts.append(cleaned)
        await self._schedule_turn()

    def _is_echo(self, text: str) -> bool:
        """A transcript that is mostly the last reply or the ASR context is not the user."""
        compact = re.sub(r"[\s，。！？、,.!?…~：:；;「」“”\"']", "", text)
        if len(compact) < 8:
            return False
        for source in (self._last_reply, self._asr_context()):
            ref = re.sub(r"[\s，。！？、,.!?…~：:；;「」“”\"']", "", source)
            if not ref:
                continue
            match = difflib.SequenceMatcher(None, compact, ref, autojunk=False).find_longest_match(
                0, len(compact), 0, len(ref))
            # A real follow-up may quote a short phrase of the reply ("换一个热闹点的");
            # ASR echoes copy whole clauses, so require a long run as well.
            if match.size >= max(ECHO_MIN_RUN_CHARS, 0.6 * len(compact)):
                return True
        return False

    async def _schedule_turn(self) -> None:
        """Start DeepSeek on the finished segment now; hold its voice for the grace window.

        Thinking overlaps the grace window instead of following it. If the user
        keeps talking, ``_on_speech_started`` extends the hold and the next
        transcript restarts the turn with the merged question.
        """
        question = "".join(self._pending_parts).strip()
        self._pending_parts = []
        if not question:
            return
        self._may_speak.clear()
        if self._hold_timer is not None:
            self._hold_timer.cancel()
        self._hold_timer = asyncio.create_task(self._release_after_grace())
        await self._start_turn(question)

    async def _emit_audio(self, pcm: bytes) -> None:
        timing = self._timing
        if timing is not None and timing.get("audio") is None:
            timing["audio"] = now = time.monotonic()
            self._log_turn_timing(timing, now)
        await self._on_audio(pcm)

    @staticmethod
    def _log_turn_timing(t: dict[str, float | None], now: float) -> None:
        def gap(a: str, b: str) -> str:
            return f"{t[b] - t[a]:.2f}s" if t.get(a) is not None and t.get(b) is not None else "-"

        origin = t.get("speech_end") or t.get("asr") or t["start"]
        logger.info(
            "turn timing: speech_end_to_asr=%s asr_to_start=%s start_to_first_token=%s "
            "first_token_to_audio=%s total=%.2fs",
            gap("speech_end", "asr"), gap("asr", "start"), gap("start", "token"), gap("token", "audio"),
            now - origin,  # type: ignore[operator]
        )

    async def _release_after_grace(self) -> None:
        await asyncio.sleep(self._turn_grace_ms / 1000)
        self._hold_timer = None
        if self._turn_in_flight() and not self._speaking:
            # The user is done: show that the robot heard it and is thinking.
            self._show_face("thinking")
        self._may_speak.set()

    def _show_face(self, face: str) -> None:
        if self._executor is None:
            return
        task = asyncio.create_task(self._executor.execute("set_avatar", {"face": face}))
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    def _clear_pending_turn(self) -> None:
        if self._turn_timer is not None:
            self._turn_timer.cancel()
            self._turn_timer = None
        self._pending_parts = []
        self._release_hold()

    # ---- turns ----------------------------------------------------------------------

    async def _start_turn(self, user_text: str) -> None:
        if not has_speech_content(user_text):
            # Qwen sometimes emits a bare "。" for noise; it is not a turn.
            logger.info("ignoring empty transcript: %r", user_text)
            return
        if self._speaking or (self._turn_task is not None and not self._turn_task.done()):
            logger.info("dropping transcript while a reply is in progress: %s", user_text)
            return
        logger.info("user said: %s", user_text)
        observe = getattr(self._executor, "observe_user_input", None)
        if callable(observe):
            await observe(user_text, True)
        self._turn_task = asyncio.create_task(self._run_turn(user_text))

    async def _cancel_turn(self) -> None:
        task, self._turn_task = self._turn_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._speaking = False

    def _system_prompt(self) -> str:
        custom = (self._setting("persona") or "").strip()
        persona_file = os.getenv("STACKCHAN_DEEPSEEK_PERSONA_FILE")
        if custom:
            # Console-written persona: used as written (no {placeholders}).
            prompt = custom
        else:
            persona = DEFAULT_PERSONA
            if persona_file and Path(persona_file).expanduser().is_file():
                persona = Path(persona_file).expanduser().read_text(encoding="utf-8")
            prompt = persona.format(
                max_chars=self._max_chars, long_chars=self._long_chars, model_label=self._model_label,
            ).replace("妮妮", self._robot_name)
        style = self._setting("reply_style")
        if style == "brief":
            prompt += f"\n- 现在用户想要简短的回答：无论什么问题，都尽量一两句说完，{self._max_chars} 字左右，除非用户明确要你展开。"
        elif style == "detailed":
            prompt += "\n- 现在用户想要完整的回答：解释、讲故事、出主意时可以讲完整，先说结论再展开，最多三百字左右。"
        # No clock here: the system prompt must stay byte-identical between turns
        # so DeepSeek's prefix cache covers it and the history after it.
        return prompt

    def _wants_thinking(self, user_text: str) -> bool:
        if self._thinking_mode in ("on", "always", "1", "true"):
            return True
        if self._thinking_mode in ("off", "never", "0", "false"):
            return False
        return bool(DEEP_THINKING_HINTS.search(user_text))

    async def _run_turn(self, user_text: str) -> None:
        assert self._memory is not None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            *self._memory.recent(self._history_limit),
            # Recent history outweighs the system prompt (old turns still say
            # "我叫D4C"), so restate the name and tag rule right before the question.
            {"role": "system", "content": (
                f"提醒：你叫{self._robot_name}，大脑是 {self._model_label}；回复开头要写情绪标签；"
                "需要实时信息就用 web_search，摘要不够再用 web_fetch。"
                f"现在时间：{time.strftime('%Y-%m-%d %H:%M %A')}。"
            )},
            {"role": "user", "content": user_text},
        ]
        self._turn_question = user_text
        self._current_question = user_text
        self._turn_understood = None
        self._timing = {"speech_end": self._last_speech_stopped, "asr": self._last_asr_done,
                        "start": time.monotonic(), "token": None, "audio": None}
        self._last_speech_stopped = self._last_asr_done = None
        self._end_requested = False
        thinking = self._wants_thinking(user_text)
        spoken: list[str] = []
        fillers: set[str] = set()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak_queue(queue))
        if thinking:
            # Thinking adds a second or more before the first word; say so.
            fillers.add(THINK_FILLER)
            await queue.put(THINK_FILLER)
        try:
            for _ in range(MAX_TOOL_ROUNDS + 1):
                tool_calls, reasoning = await self._stream_reply(messages, queue, spoken, thinking)
                if not tool_calls:
                    break
                assistant: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": tool_calls}
                if reasoning:
                    # Thinking mode needs its reasoning echoed back across tool rounds.
                    assistant["reasoning_content"] = reasoning
                messages.append(assistant)
                names = {c["function"]["name"] for c in tool_calls}
                filler = SEARCH_FILLER if "web_search" in names else FETCH_FILLER if "web_fetch" in names else None
                if filler and not fillers and not "".join(spoken).strip():
                    # A search or fetch takes a few seconds; say so instead of going
                    # quiet, but only once per turn (no "我想想" then "我查一下").
                    fillers.add(filler)
                    await queue.put(filler)
                for call in tool_calls:
                    result = await self._run_tool(call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                if self._end_requested and spoken:
                    break
        except asyncio.CancelledError:
            heard_any = self._speaking
            speaker.cancel()
            partial = "".join(spoken).strip()
            if partial and heard_any:
                # Interrupted mid-reply: remember what was actually said. A turn
                # started early and cancelled before speaking is not a turn.
                self._memory.add("user", self._remembered_question(user_text))
                self._memory.add("assistant", partial)
            raise
        except Exception as exc:
            logger.exception("DeepSeek turn failed: %s", exc)
            await queue.put("[apologetic] 我刚才走神了，你再说一遍好吗？")
        finally:
            await queue.put(None)
        await speaker
        reply = "".join(spoken).strip()
        if reply:
            self._memory.add("user", self._remembered_question(user_text))
            self._memory.add("assistant", reply)
            self._last_reply = reply
            if self._on_text is not None:
                await self._on_text(reply)
        if self._asr_stale:
            await self._refresh_stale_asr()
        if self._end_requested and self._on_end_conversation is not None:
            await self._on_end_conversation()

    async def _stream_reply(
        self,
        messages: list[dict[str, Any]],
        queue: asyncio.Queue[str | None],
        spoken: list[str],
        thinking: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """Stream one completion and log it as an API call."""
        started = time.monotonic()
        self._last_served = self._last_usage = None
        status = "ok"
        try:
            calls, reasoning = await self._stream_reply_inner(messages, queue, spoken, thinking)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status = f"failed:{type(exc).__name__}"
            raise
        finally:
            usage = self._last_usage or {}
            log_api_call(
                "deepseek", self._last_served or self._model, "chat", status, time.monotonic() - started,
                requested=self._model, thinking="on" if thinking else "off",
                prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            )
        return calls, reasoning

    async def _stream_reply_inner(
        self,
        messages: list[dict[str, Any]],
        queue: asyncio.Queue[str | None],
        spoken: list[str],
        thinking: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """Stream one completion; queue speakable pieces, return (tool calls, reasoning)."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "max_tokens": 4000 if thinking else 1500,
            "stream_options": {"include_usage": True},
        }
        if thinking:
            kwargs["reasoning_effort"] = self._reasoning_effort
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if self._tools:
            kwargs["tools"] = self._tools
        started = time.monotonic()
        stream = await self._llm.chat.completions.create(**kwargs)
        splitter = SentenceSplitter()
        calls: dict[int, dict[str, Any]] = {}
        first = True
        mood_checked = False
        head = ""
        served_logged = False
        reasoning_parts: list[str] = []
        async for chunk in stream:
            if not served_logged and getattr(chunk, "model", None):
                # The API only returns the alias; the fingerprint changes when
                # DeepSeek swaps the backend build behind it.
                logger.info(
                    "DeepSeek served model=%s fingerprint=%s",
                    chunk.model, getattr(chunk, "system_fingerprint", None) or "-",
                )
                served_logged = True
                self._last_served = chunk.model
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                self._last_usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            for tc in getattr(delta, "tool_calls", None) or []:
                entry = calls.setdefault(tc.index, {
                    "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                })
                if tc.id:
                    entry["id"] = tc.id
                if tc.function and tc.function.name:
                    entry["function"]["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    entry["function"]["arguments"] += tc.function.arguments
            thought = getattr(delta, "reasoning_content", None) or ""
            if thought:
                reasoning_parts.append(thought)
            text = getattr(delta, "content", None) or ""
            if not text:
                continue
            if first:
                logger.info("DeepSeek first token after %.2fs", time.monotonic() - started)
                if self._timing is not None and self._timing.get("token") is None:
                    self._timing["token"] = time.monotonic()
                if reasoning_parts:
                    logger.info(
                        "DeepSeek thinking: effort=%s reasoning_chars=%d",
                        self._reasoning_effort, sum(len(p) for p in reasoning_parts),
                    )
                first = False
            if not mood_checked:
                head += text
                ready, mood, understood, text = parse_reply_head(head)
                if not ready:
                    continue
                mood_checked = True
                self._apply_reply_head(mood, understood)
            spoken.append(text)
            for piece in splitter.feed(text):
                await queue.put(piece)
        if not mood_checked and head:
            _ready, mood, understood, rest = parse_reply_head(head, final=True)
            self._apply_reply_head(mood, understood)
            spoken.append(rest)
            splitter.feed(rest)
        for piece in splitter.flush():
            await queue.put(piece)
        return [calls[i] for i in sorted(calls)], "".join(reasoning_parts)

    def _remembered_question(self, user_text: str) -> str:
        if self._turn_understood:
            return f"{user_text}（{self._robot_name}理解为：{self._turn_understood}）"
        return user_text

    def _apply_reply_head(self, mood: str | None, understood: str | None) -> None:
        if mood:
            self._express(mood)
        if understood and understood != self._turn_question:
            self._turn_understood = understood
            logger.info("understood as: heard=%s understood=%s", self._turn_question, understood)

    def _express(self, mood: str) -> None:
        if self._executor is None:
            return
        task = asyncio.create_task(self._executor.execute("express_emotion", {"mood": mood}))
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _run_tool(self, call: dict[str, Any]) -> dict[str, Any]:
        alias = call["function"]["name"]
        name = self._tool_aliases.get(alias, alias)
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "invalid JSON arguments"}
        if name == "web_search":
            return await self._web_search(str(args.get("query", "")))
        if name == "web_fetch":
            return await self._web_fetch(str(args.get("url", "")))
        if name == "end_conversation":
            self._end_requested = True
            return {"ok": True}
        logger.info("tool %s %s", name, args)
        try:
            return await self._executor.execute(name, args)
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            return {"ok": False, "error": str(exc)}

    async def _deepseek_tool(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        import aiohttp  # noqa: PLC0415

        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as session:
            async with session.post(url, json=payload) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {str(body)[:200]}")
                return body

    async def _web_search(self, query: str) -> dict[str, Any]:
        """Live web search: DeepSeek's search API first, Qwen's built-in search as fallback."""
        query = query.strip()[:128]
        if not query:
            return {"ok": False, "error": "empty query"}
        started = time.monotonic()
        try:
            body = await self._deepseek_tool(DEEPSEEK_SEARCH_URL, {
                "query": query,
                "count": 8,
                "ori_question": (self._current_question or query)[:128],
            }, timeout=10.0)
            ranked = sorted(body.get("results") or [], key=lambda r: r.get("rerank_score") or 0, reverse=True)
            results = [{
                "title": r.get("title"),
                "site": r.get("site_name"),
                "date": r.get("date"),
                "url": r.get("url"),
                "summary": (r.get("summary") or r.get("snippet") or "")[:800],
            } for r in ranked[:SEARCH_RESULTS_FOR_MODEL]]
            if results:
                titles = " | ".join(f"{r['title']}（{r['site'] or '?'}）" for r in results[:3])
                logger.info("web search ok: %.2fs source=deepseek query=%s answer=%s",
                            time.monotonic() - started, query, titles)
                log_api_call("deepseek", "search", "web_search", "ok", time.monotonic() - started,
                             results=len(results))
                return {"ok": True, "query": query, "results": results}
            logger.info("DeepSeek search returned nothing for %s; trying Qwen", query)
            log_api_call("deepseek", "search", "web_search", "empty", time.monotonic() - started)
        except Exception as exc:
            logger.warning("DeepSeek search failed, trying Qwen: query=%s error=%s", query, exc)
            log_api_call("deepseek", "search", "web_search", "failed", time.monotonic() - started,
                         error=type(exc).__name__)
        return await self._qwen_search(query, started)

    async def _qwen_search(self, query: str, started: float) -> dict[str, Any]:
        if not self._dashscope_key:
            return {"ok": False, "error": "search unavailable"}
        if self._search_client is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            self._search_client = AsyncOpenAI(api_key=self._dashscope_key, base_url=DASHSCOPE_COMPAT_URL)
        try:
            response = await self._search_client.chat.completions.create(
                model=self._search_model,
                messages=[
                    {"role": "system", "content": "用中文回答，给出关键事实、数字和日期，300字以内，不要附网址。"},
                    {"role": "user", "content": query},
                ],
                max_tokens=600,
                extra_body={"enable_search": True, "search_options": {"forced_search": True}},
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("web search failed: query=%s error=%s", query, exc)
            log_api_call("dashscope", self._search_model, "web_search", "failed", time.monotonic() - started,
                         error=type(exc).__name__)
            return {"ok": False, "error": str(exc)}
        logger.info("web search ok: %.2fs source=qwen query=%s answer=%s", time.monotonic() - started, query, answer)
        log_api_call("dashscope", self._search_model, "web_search", "ok", time.monotonic() - started,
                     enable_search="on")
        return {"ok": True, "query": query, "result": answer}

    async def _web_fetch(self, url: str) -> dict[str, Any]:
        """Read one page as Markdown with DeepSeek's fetch API (truncated for the model)."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "error": "url must start with http:// or https://"}
        started = time.monotonic()
        try:
            body = await self._deepseek_tool(DEEPSEEK_FETCH_URL, {"url": url}, timeout=FETCH_TIMEOUT_S)
        except Exception as exc:
            logger.warning("web fetch failed: %.2fs url=%s error=%s", time.monotonic() - started, url, exc)
            log_api_call("deepseek", "fetch", "web_fetch", "failed", time.monotonic() - started,
                         error=type(exc).__name__)
            return {"ok": False, "error": f"page could not be read: {exc}"}
        content = body.get("content") or ""
        logger.info("web fetch ok: %.2fs url=%s title=%s chars=%d",
                    time.monotonic() - started, url, body.get("title") or "-", len(content))
        log_api_call("deepseek", "fetch", "web_fetch", "ok", time.monotonic() - started, chars=len(content))
        return {
            "ok": True,
            "title": body.get("title"),
            "content": content[:FETCH_MAX_CHARS],
            "truncated": len(content) > FETCH_MAX_CHARS,
        }

    async def _announce_result(self, text: str) -> bool:
        return await self.inject_user_text(text)

    # ---- speech out -------------------------------------------------------------------

    async def _speak_verbatim(self, text: str) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak_queue(queue))
        for piece in SentenceSplitter().feed(text + "\n"):
            await queue.put(piece)
        await queue.put(None)
        await speaker

    async def _speak_queue(self, queue: asyncio.Queue[str | None]) -> None:
        """Speak queued pieces in order; emits on_turn_complete once at the end.

        ElevenLabs gets the whole reply on one dialogue socket so it can keep
        prosody and voice consistent across sentences; other engines speak
        piece by piece.
        """
        produced = False
        try:
            first = await self._next_spoken_piece(queue)
            if first is None:
                return
            if not self._speaking:
                await self._may_speak.wait()
            self._speaking = True
            self._voice_failed = False
            if await self._use_elevenlabs():
                produced = await self._speak_stream_elevenlabs(first, queue)
            else:
                produced = await self._speak_pieces([first], queue)
            # Before on_turn_complete, so the post-reply face is already the error face.
            await self._report_voice_error(self._voice_failed or not produced)
        finally:
            if produced:
                await self._on_turn_complete()
            self._speaking = False

    async def _report_voice_error(self, failed: bool) -> None:
        if self.on_voice_error is None:
            return
        try:
            await self.on_voice_error(failed)
        except Exception as exc:
            logger.warning("voice error report failed: %s", exc)

    async def _drain_unspoken(self, queue: asyncio.Queue[str | None]) -> None:
        """Consume the rest of the reply so the turn can finish without a voice."""
        while await queue.get() is not None:
            pass

    @staticmethod
    async def _next_spoken_piece(queue: asyncio.Queue[str | None]) -> str | None:
        while True:
            piece = await queue.get()
            if piece is None:
                return None
            text = clean_for_speech(piece)
            if text:
                return text

    async def _speak_pieces(
        self, pending: list[str], queue: asyncio.Queue[str | None] | None
    ) -> bool:
        """Synthesize each piece separately (edge-tts engine setting, injected engines)."""
        produced = False
        while True:
            if pending:
                text = pending.pop(0)
            elif queue is not None:
                text = await self._next_spoken_piece(queue)
                if text is None:
                    break
            else:
                break
            async for pcm in self._synthesize(text):
                produced = True
                await self._emit_audio(pcm)
        return produced

    async def _use_elevenlabs(self) -> bool:
        if self._tts_synthesize is not None:
            return False
        if not (self._eleven_key and self._voice_id) or self._setting("tts_engine") != "elevenlabs":
            return False
        if self._quota is not None and await self._quota.over_quota():
            logger.info("ElevenLabs quota past the fallback ratio; using edge-tts")
            return False
        return True

    async def _speak_stream_elevenlabs(self, first: str, queue: asyncio.Queue[str | None]) -> bool:
        """Stream one reply into a single ElevenLabs dialogue socket.

        Pieces are sent as consecutive inputs of the same turn, so the server
        buffers ~40 characters before generating and keeps one prosody segment
        instead of voicing each short sentence in isolation. Flushes are sent
        only after the opening sentence (fast first sound) and when the text
        stream pauses (e.g. a filler before a web search). A socket that fails
        is retried once; if that fails too the reply is not voiced and the
        voice-error face goes up (no robotic fallback voice).
        """
        started = time.monotonic()
        # queue_done: the reply's None terminator was taken; it is never put back for a retry.
        state = {"first_audio": None, "audio_bytes": 0, "finished": False, "queue_done": False}
        texts: list[str] = []          # every piece taken from the queue, in order
        replay = [first]               # pieces the next socket must send before the queue
        for attempt in range(2):
            try:
                await self._elevenlabs_session(replay, queue, texts, state, started)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                heard = state["audio_bytes"] > 0
                if attempt == 0 and (not heard or not state["finished"]):
                    logger.warning("ElevenLabs socket failed (%s: %s); retrying once", type(exc).__name__, exc)
                    # Nothing heard: say it all again. Something heard: we cannot tell how
                    # much, so only the pieces not yet taken from the queue follow.
                    # (A socket that failed before its sender ran never recorded the replay.)
                    replay = (list(texts) or replay) if not heard else []
                    if not heard:
                        texts.clear()
                    state["finished"] = False
                    continue
                logger.warning("ElevenLabs TTS failed after a retry; the rest of the reply is not voiced: %s", exc)
                log_api_call("elevenlabs", self._tts_model, "tts", "failed", time.monotonic() - started,
                             voice=self._voice_id, chars=sum(map(len, texts)), error=type(exc).__name__)
                self._voice_failed = True
                if not state["queue_done"]:
                    await self._drain_unspoken(queue)
                return heard
        text = "".join(texts)
        if not state["audio_bytes"]:
            logger.warning("ElevenLabs TTS returned no audio; the reply is not voiced: text=%s", text)
            log_api_call("elevenlabs", self._tts_model, "tts", "empty", time.monotonic() - started,
                         voice=self._voice_id, chars=len(text))
            self._voice_failed = True
            return False
        first_audio = state["first_audio"] or 0.0
        logger.info(
            "ElevenLabs TTS ok: model=%s chars=%d pieces=%d first_audio=%.2fs total=%.2fs audio=%.1fs text=%s",
            self._tts_model, len(text), len(texts), first_audio, time.monotonic() - started,
            state["audio_bytes"] / (OUTPUT_RATE * 2), text,
        )
        log_api_call("elevenlabs", self._tts_model, "tts", "ok", time.monotonic() - started,
                     voice=self._voice_id, chars=len(text), pieces=len(texts),
                     first_audio=f"{first_audio:.2f}s")
        return True

    async def _elevenlabs_session(
        self, replay: list[str], queue: asyncio.Queue[str | None], texts: list[str],
        state: dict[str, Any], started: float,
    ) -> None:
        """One dialogue socket: send ``replay`` then the queue, play audio until is_final."""
        import websockets  # noqa: PLC0415

        async def sender(ws) -> None:
            # The opening sentence is flushed at once so the first sound comes
            # quickly; everything after it streams unflushed as one segment.
            socket_chars = 0
            opening = True

            async def send_piece(text: str) -> bool:
                nonlocal opening, socket_chars
                await ws.send(json.dumps({"inputs": [{"text": text, "voice_id": self._voice_id}]}))
                socket_chars += len(text)
                if opening and socket_chars >= ELEVEN_OPENING_MIN_CHARS:
                    await ws.send(json.dumps({"flush": True}))
                    opening = False
                    return False
                return True

            # A non-empty replay always starts from an empty ``texts`` (first socket,
            # or a retry where nothing was heard), so record it before sending.
            texts.extend(replay)
            dirty = False
            for text in replay:
                dirty = await send_piece(text)
            idle = 0.0
            while not state["queue_done"]:
                try:
                    piece = await asyncio.wait_for(queue.get(), timeout=ELEVEN_PAUSE_FLUSH_S)
                except asyncio.TimeoutError:
                    idle += ELEVEN_PAUSE_FLUSH_S
                    if dirty:
                        await ws.send(json.dumps({"flush": True}))
                        dirty, idle = False, 0.0
                    elif idle >= ELEVEN_KEEPALIVE_S:
                        await ws.send(json.dumps({"keep_alive": True}))
                        idle = 0.0
                    continue
                if piece is None:
                    state["queue_done"] = True
                    break
                text = clean_for_speech(piece)
                if not text:
                    continue
                texts.append(text)
                dirty, idle = await send_piece(text), 0.0
            await ws.send(json.dumps({"close_socket": True}))
            state["finished"] = True

        send_task: asyncio.Task[None] | None = None
        try:
            url = f"{ELEVENLABS_DIALOGUE_WS}?{self._elevenlabs_query()}"
            async with websockets.connect(
                url, additional_headers={"xi-api-key": self._eleven_key}, max_size=None
            ) as ws:
                await ws.send(json.dumps(self._elevenlabs_init()))
                send_task = asyncio.create_task(sender(ws))
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("error"):
                        raise RuntimeError(str(message.get("message") or message["error"]))
                    if message.get("audio"):
                        pcm = base64.b64decode(message["audio"])
                        if state["first_audio"] is None:
                            state["first_audio"] = time.monotonic() - started
                        state["audio_bytes"] += len(pcm)
                        await self._emit_audio(pcm)
                    if message.get("is_final") or message.get("isFinal"):
                        break
                if send_task.done() and send_task.exception() is not None:
                    raise send_task.exception()  # type: ignore[misc]
                if not state["finished"]:
                    raise RuntimeError("dialogue socket closed before the reply was sent")
        finally:
            if send_task is not None and not send_task.done():
                send_task.cancel()

    def _elevenlabs_init(self) -> dict[str, Any]:
        init: dict[str, Any] = {"voices": [self._voice_id]}
        stability = self._setting("eleven_stability")
        if stability is not None:
            # The dialogue socket accepts only stability, on the first message.
            init["voice_settings"] = {"stability": float(stability)}
        return init

    async def _synthesize(self, text: str):
        if self._tts_synthesize is not None:
            async for pcm in self._tts_synthesize(text, None):
                yield pcm
            return
        use_eleven = (bool(self._eleven_key and self._voice_id)
                      and self._setting("tts_engine") == "elevenlabs")
        if use_eleven and self._quota is not None and await self._quota.over_quota():
            logger.info("ElevenLabs quota past the fallback ratio; using edge-tts")
            use_eleven = False
        if use_eleven:
            started = time.monotonic()
            first_audio: float | None = None
            audio_bytes = 0
            try:
                async for pcm in self._elevenlabs(text):
                    if first_audio is None:
                        first_audio = time.monotonic() - started
                    audio_bytes += len(pcm)
                    yield pcm
                if audio_bytes:
                    logger.info(
                        "ElevenLabs TTS ok: model=%s chars=%d first_audio=%.2fs total=%.2fs audio=%.1fs text=%s",
                        self._tts_model, len(text), first_audio or 0.0, time.monotonic() - started,
                        audio_bytes / (OUTPUT_RATE * 2), text,
                    )
                    log_api_call("elevenlabs", self._tts_model, "tts", "ok", time.monotonic() - started,
                                 voice=self._voice_id, chars=len(text), first_audio=f"{first_audio or 0.0:.2f}s")
                    return
                logger.warning("ElevenLabs TTS returned no audio, falling back to edge-tts: text=%s", text)
                log_api_call("elevenlabs", self._tts_model, "tts", "empty", time.monotonic() - started,
                             voice=self._voice_id, chars=len(text))
            except Exception as exc:
                logger.warning("ElevenLabs TTS failed, falling back to edge-tts: %s", exc)
                log_api_call("elevenlabs", self._tts_model, "tts", "failed", time.monotonic() - started,
                             voice=self._voice_id, error=type(exc).__name__)
        started = time.monotonic()
        audio_bytes = 0
        async for pcm in self._edge_tts(text):
            audio_bytes += len(pcm)
            yield pcm
        logger.info(
            "edge-tts TTS %s: chars=%d total=%.2fs audio=%.1fs text=%s",
            "ok" if audio_bytes else "returned no audio", len(text),
            time.monotonic() - started, audio_bytes / (OUTPUT_RATE * 2), text,
        )
        log_api_call("microsoft", "edge-tts", "tts", "ok" if audio_bytes else "empty", time.monotonic() - started,
                     voice=self._setting("edge_voice") or EDGE_TTS_VOICE, chars=len(text))

    def _elevenlabs_query(self) -> str:
        from urllib.parse import urlencode  # noqa: PLC0415

        params: dict[str, Any] = {"model_id": self._tts_model, "output_format": f"pcm_{OUTPUT_RATE}"}
        if self._setting("eleven_language"):
            params["language_code"] = self._setting("eleven_language")
        if self._setting("eleven_seed") is not None:
            params["seed"] = int(self._setting("eleven_seed"))
        if self._setting("eleven_text_normalization") != "auto":
            params["apply_text_normalization"] = self._setting("eleven_text_normalization")
        return urlencode(params)

    async def _elevenlabs(self, text: str):
        import websockets  # noqa: PLC0415

        url = f"{ELEVENLABS_DIALOGUE_WS}?{self._elevenlabs_query()}"
        async with websockets.connect(
            url, additional_headers={"xi-api-key": self._eleven_key}, max_size=None
        ) as ws:
            await ws.send(json.dumps(self._elevenlabs_init()))
            await ws.send(json.dumps({"inputs": [{"text": text, "voice_id": self._voice_id}]}))
            await ws.send(json.dumps({"flush": True}))
            await ws.send(json.dumps({"close_socket": True}))
            async for raw in ws:
                message = json.loads(raw)
                if message.get("error"):
                    raise RuntimeError(str(message["error"]))
                if message.get("audio"):
                    yield base64.b64decode(message["audio"])
                if message.get("is_final") or message.get("isFinal"):
                    break

    async def _edge_tts(self, text: str):
        import edge_tts  # noqa: PLC0415
        import miniaudio  # noqa: PLC0415

        mp3 = bytearray()
        voice = self._setting("edge_voice") or EDGE_TTS_VOICE
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk.get("type") == "audio":
                mp3 += chunk["data"]
        if not mp3:
            return
        decoded = miniaudio.decode(
            bytes(mp3), output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1, sample_rate=OUTPUT_RATE,
        )
        yield decoded.samples.tobytes()
