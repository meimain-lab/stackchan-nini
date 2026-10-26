"""Local-only control API for the Stack-chan console (monitor page).

Mounted on the gateway's capture app (port 8766). That app listens on all
interfaces so the device can upload captures, so every route here refuses
requests that do not come from this Mac.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from .console_settings import SCHEMA, SettingsError, get_console_settings

if TYPE_CHECKING:  # pragma: no cover
    from .gateway import Gateway

logger = logging.getLogger(__name__)

LOCAL_ADDRESSES = {"127.0.0.1", "::1", "localhost"}
ELEVENLABS_API = "https://api.elevenlabs.io"


def _local_only(handler):
    async def guarded(request: web.Request) -> web.StreamResponse:
        if request.remote not in LOCAL_ADDRESSES:
            return web.json_response({"error": "console control is local-only"}, status=403)
        return await handler(request)

    return guarded


def _memory_db() -> Path:
    raw = os.getenv("STACKCHAN_MEMORY_DB", "").strip()
    return Path(raw).expanduser() if raw else Path(__file__).resolve().parents[1] / "data" / "memory.sqlite"


def register_console_routes(app: web.Application, gateway: Gateway) -> None:
    settings = get_console_settings()

    async def get_settings(_request: web.Request) -> web.Response:
        return web.json_response({
            "settings": settings.snapshot(),
            "schema": {
                key: {
                    "kind": f.kind, "min": f.minimum, "max": f.maximum,
                    "choices": list(f.choices), "nullable": f.nullable,
                }
                for key, f in SCHEMA.items()
            },
            "runtime": gateway.console_runtime_status(),
        })

    async def post_settings(request: web.Request) -> web.Response:
        try:
            changes = await request.json()
            if not isinstance(changes, dict):
                raise SettingsError("请求体必须是 JSON 对象")
            changed = settings.update(changes)
        except SettingsError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except ValueError:
            return web.json_response({"error": "请求体不是合法的 JSON"}, status=400)
        return web.json_response({"changed": changed, "settings": settings.snapshot(),
                                  "runtime": gateway.console_runtime_status()})

    async def post_wake(_request: web.Request) -> web.Response:
        ok = await gateway.console_wake_now()
        return web.json_response({"ok": ok, "error": None if ok else "Stack-chan 还没连上 gateway"})

    async def post_say(request: web.Request) -> web.Response:
        body = await request.json()
        text = str(body.get("text") or "我是妮妮，这是现在的声音设置。").strip()[:200]
        ok = await gateway.console_say(text)
        return web.json_response({"ok": ok, "error": None if ok else "Stack-chan 还没连上 gateway"})

    async def get_memory(request: web.Request) -> web.Response:
        limit = max(1, min(200, int(request.query.get("limit", "40"))))
        path = _memory_db()
        if not path.exists():
            return web.json_response({"turns": [], "total": 0})
        with sqlite3.connect(str(path)) as db:
            total = db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            rows = db.execute(
                "SELECT created_at, role, content FROM turns ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        turns = [{"time": time.strftime("%m-%d %H:%M", time.localtime(t)), "role": r, "content": c}
                 for t, r, c in reversed(rows)]
        return web.json_response({"turns": turns, "total": total})

    async def delete_memory(_request: web.Request) -> web.Response:
        path = _memory_db()
        deleted = 0
        if path.exists():
            with sqlite3.connect(str(path)) as db:
                deleted = db.execute("DELETE FROM turns").rowcount
        logger.info("console cleared conversation memory (%d turns)", deleted)
        return web.json_response({"deleted": deleted})

    async def get_voices(_request: web.Request) -> web.Response:
        import aiohttp  # noqa: PLC0415

        key = os.getenv("ELEVENLABS_API_KEY", "")
        if not key:
            return web.json_response({"error": "没有配置 ELEVENLABS_API_KEY"}, status=400)
        async with aiohttp.ClientSession(headers={"xi-api-key": key},
                                         timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(f"{ELEVENLABS_API}/v1/voices") as resp:
                data: dict[str, Any] = await resp.json()
                if resp.status != 200:
                    return web.json_response({"error": f"ElevenLabs 返回 {resp.status}"}, status=502)
        voices = [{
            "voice_id": v.get("voice_id"),
            "name": v.get("name"),
            "category": v.get("category"),
            "language": (v.get("labels") or {}).get("language"),
        } for v in data.get("voices") or []]
        return web.json_response({"voices": voices})

    routes = [
        ("GET", "/control/settings", get_settings),
        ("POST", "/control/settings", post_settings),
        ("POST", "/control/wake", post_wake),
        ("POST", "/control/say", post_say),
        ("GET", "/control/memory", get_memory),
        ("DELETE", "/control/memory", delete_memory),
        ("GET", "/control/elevenlabs/voices", get_voices),
    ]
    for method, path, handler in routes:
        app.router.add_route(method, path, _local_only(handler))
