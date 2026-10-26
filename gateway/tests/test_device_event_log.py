# SPDX-License-Identifier: MIT
"""JSONL log of frames sent to the robot."""

from __future__ import annotations

import json

import pytest

from stackchan_mcp import device_event_log as module
from stackchan_mcp.device_event_log import DeviceEventLog, logged_sender


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_json_frames_are_logged_whole_and_audio_is_summed_per_burst(tmp_path):
    path = tmp_path / "events.jsonl"
    log = DeviceEventLog(path)
    log.record(json.dumps({"type": "tts", "state": "start"}))
    for _ in range(5):
        log.record(b"\x00" * 120)
    log.record(json.dumps({"type": "head", "yaw": 10, "pitch": 5, "speed": 300}))

    records = _records(path)
    assert [r["kind"] for r in records] == ["json", "audio", "json"]
    assert records[0]["message"] == {"type": "tts", "state": "start"}
    assert records[1]["frames"] == 5 and records[1]["bytes"] == 600 and records[1]["audio_ms"] == 300
    assert records[2]["message"]["type"] == "head"


def test_audio_gap_starts_a_new_burst(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    log = DeviceEventLog(path)
    clock = iter([100.0, 100.06, 102.0, 102.06])
    monkeypatch.setattr(module.time, "time", lambda: next(clock))
    for _ in range(4):
        log.record(b"\x01")
    log._close_burst()
    assert [r["frames"] for r in _records(path)] == [2, 2]


def test_disabled_log_writes_nothing_and_sender_is_unwrapped(tmp_path):
    async def send(_payload):
        return None

    log = DeviceEventLog(None)
    log.record("{}")
    assert logged_sender(send, log) is send
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_logged_sender_sends_then_records(tmp_path):
    sent = []

    async def send(payload):
        sent.append(payload)

    path = tmp_path / "events.jsonl"
    wrapped = logged_sender(send, DeviceEventLog(path))
    await wrapped('{"type":"led","r":1,"g":2,"b":3}')
    assert sent == ['{"type":"led","r":1,"g":2,"b":3}']
    assert _records(path)[0]["message"]["type"] == "led"
