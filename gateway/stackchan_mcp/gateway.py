"""Two-faced gateway: bridges MCP client (stdio MCP) and ESP32 (WebSocket MCP).

MCP client sees a standard MCP server via stdio.
ESP32 sees a WebSocket server that sends MCP client requests.
This module orchestrates both sides.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time

from aiohttp import web
from aiohttp.web_log import AccessLogger

from .capture_server import create_capture_app
from .console_control import register_console_routes
from .console_settings import get_console_settings
from .cloud_proxy import CloudProxy, DEFAULT_CLOUD_URL
from .device_emotion import IDLE_EMOTION, LISTENING_EMOTION, face_to_device_emotion
from .demo_reactions import DemoReactions
from .debug_status import get_debug_status
from .esp32_client import ESP32Manager
from .face_tracker import FaceTrackerSupervisor
from .deepseek_voice_proxy import DeepSeekVoiceProxy
from .gemini_voice_proxy import GeminiVoiceProxy
from .gbot_voice_proxy import GbotVoiceProxy
from .idle_behavior import IdleBehavior
from .idle_gate import IdleGate
from .mac_control import resolve_claude_bin
from .touch_bridge import TouchEventBridge
from .tracking_bridge import TrackingBridge
from .tts import synthesize_and_send
from .usb_transport import UsbTransport
from .voice_input_bridge import VoiceInputBridge

logger = logging.getLogger(__name__)


class TrackQuietAccessLogger(AccessLogger):
    """Keep noisy /track polling out of INFO logs."""

    @property
    def enabled(self) -> bool:
        return self.logger.isEnabledFor(logging.INFO) or self.logger.isEnabledFor(
            logging.DEBUG
        )

    def log(self, request, response, time) -> None:
        try:
            fmt_info = self._format_line(request, response, time)
            values = []
            extra = {}
            for key, value in fmt_info:
                values.append(value)
                if key.__class__ is str:
                    extra[key] = value
                else:
                    k1, k2 = key
                    dct = extra.get(k1, {})
                    dct[k2] = value
                    extra[k1] = dct
            message = self._log_format % tuple(values)
            if request.path == "/track":
                self.logger.debug(message, extra=extra)
            else:
                self.logger.info(message, extra=extra)
        except Exception:
            self.logger.exception("Error in logging")


def head_follow_default_enabled() -> bool:
    """Whether the head follows faces when the gateway starts.

    Set STACKCHAN_HEAD_FOLLOW_DEFAULT=0 when the robot has no head servos or
    should stay still; "look at me" turns following on later.
    """
    value = os.getenv("STACKCHAN_HEAD_FOLLOW_DEFAULT", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


class Gateway:
    """Main gateway orchestrator.

    Holds the ESP32 manager and provides the bridge between
    the stdio MCP server (MCP client side) and the ESP32 device.

    Also runs an HTTP capture server for receiving photos from ESP32.
    """

    def __init__(self):
        self.voice_input_bridge = VoiceInputBridge()
        # USB Serial/JTAG transport for low-latency control traffic.
        # Initialised before esp32 so the voice backend factory can hand it
        # to GeminiVoiceProxy without a forward reference. With the cable in,
        # TrackingBridge / hooks / Gemini function calls send set_head_angles
        # through here at ~10 ms instead of ~240 ms over WS.
        # Opt-in with STACKCHAN_USB_TRANSPORT=1: it opens /dev/cu.usbmodem*
        # exclusively, which blocks flashing and serial monitors, and the
        # Groki Bot firmware does not speak this USB protocol.
        # STACKCHAN_USB_DISABLE=1 still forces it off.
        self.usb_transport: UsbTransport | None
        usb_enabled = os.getenv("STACKCHAN_USB_TRANSPORT", "0").strip().lower() in {"1", "true", "yes", "on"}
        usb_disabled = os.getenv("STACKCHAN_USB_DISABLE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if usb_enabled and not usb_disabled:
            self.usb_transport = UsbTransport()
        else:
            self.usb_transport = None
            logger.info("USB transport off (set STACKCHAN_USB_TRANSPORT=1 to enable)")

        self.esp32 = ESP32Manager(
            cloud_proxy_factory=self._make_voice_proxy,
            usb_transport=self.usb_transport,
            local_tool_handler=self._handle_external_local_tool,
        )
        self.demo_reactions = DemoReactions(
            self.esp32,
            usb_transport=self.usb_transport,
        )
        self.idle_gate = IdleGate(
            self.esp32,
            usb_transport=self.usb_transport,
        )
        self.touch_bridge = TouchEventBridge(
            self.esp32,
            poll_interval_s=float(os.getenv("STACKCHAN_TOUCH_POLL_INTERVAL_S", "0.1")),
            on_touch_activity=self.idle_gate.notify_touch,
        )
        self.touch_bridge.add_listener(self._handle_touch_event)
        self.idle_behavior = IdleBehavior(
            self.esp32,
            usb_transport=self.usb_transport,
        )
        self.tracking_bridge = TrackingBridge(
            self.esp32,
            usb_transport=self.usb_transport,
            on_face_detected=self._notify_face_detected,
            on_auto_head_command=self.idle_gate.notify_auto_head_command,
            # The head holds still while the robot speaks or listens.
            is_tts_active=lambda: get_debug_status().tts_active,
            is_listening=lambda: bool(getattr(
                getattr(self._active_voice_proxy(), "_wake_gate", None), "is_listening", False,
            )),
        )
        # The face tracker runs as long as the gateway does; head follow is on
        # by default and "look at me" / "stop looking" toggle it.
        self.tracking_bridge.enabled = bool(get_console_settings().get("head_follow"))
        self.face_tracker = FaceTrackerSupervisor()
        self._last_face_at: float | None = None
        self._last_detection: dict | None = None
        self._detection_count = 0
        self.esp32.on_activity = self.idle_behavior.notify_activity
        self.esp32.on_head_command = self.idle_gate.notify_head_command
        self.esp32.on_device_state = self.idle_gate.notify_device_state
        self.esp32.on_wake_detected = self._run_wake_response
        self.esp32.on_end_conversation = self._handle_end_conversation
        self._running = False
        self._http_runner: web.AppRunner | None = None
        get_console_settings().add_listener(self._apply_console_settings)

    def _notify_face_detected(self) -> None:
        self._last_face_at = time.time()
        self.idle_gate.notify_face_detected()
        self.face_tracker.notify_face_reported()

    # ---- console control -------------------------------------------------------

    def console_runtime_status(self) -> dict:
        status = get_debug_status().snapshot()
        return {
            "device": status.get("device"),
            "wake_gate": status.get("wake_gate"),
            "face_tracking": self.face_tracking_status(),
        }

    async def console_wake_now(self) -> bool:
        proxy = self._active_voice_proxy()
        wake = getattr(proxy, "wake_now", None)
        return bool(await wake()) if callable(wake) else False

    async def console_say(self, text: str) -> bool:
        proxy = self._active_voice_proxy()
        bridge = getattr(proxy, "_bridge", None)
        inject = getattr(bridge, "inject_user_text", None)
        return bool(await inject(text)) if callable(inject) else False

    def _apply_console_settings(self, changed: dict) -> None:
        loop = asyncio.get_running_loop()
        if "face_tracking" in changed:
            if changed["face_tracking"]:
                self.face_tracker.start(f"http://127.0.0.1:{self.capture_port}/track", force=True)
            else:
                # Stops the tracker process, which releases the Mac camera.
                loop.create_task(self.face_tracker.stop())
        if "head_follow" in changed:
            self.tracking_bridge.enabled = bool(changed["head_follow"])
        proxy = self._active_voice_proxy()
        apply = getattr(proxy, "apply_console_settings", None)
        if callable(apply):
            apply(changed)

    def face_tracking_status(self) -> dict:
        """The "face_tracking" section of /debug/status."""
        return {
            "tracker_running": self.face_tracker.running,
            "tracker_pid": self.face_tracker.pid,
            "face_reported": self.face_tracker.available,
            "last_face_at": self._last_face_at,
            "head_follow": self.tracking_bridge.enabled,
            # idle: following; working: the robot is speaking; quiet: it is
            # listening after the wake word. The head holds still in both.
            "mode": self.tracking_bridge.mode,
            # Only the face box ever leaves the tracker; frames stay in its memory.
            "last_detection": self._last_detection,
            "detections": self._detection_count,
        }

    def _active_voice_proxy(self):
        connection = getattr(self.esp32, "connection", None)
        return getattr(connection, "cloud_proxy", None)

    @property
    def capture_port(self) -> int:
        return int(os.getenv("CAPTURE_PORT", "8766"))

    async def start_face_tracking(self) -> dict:
        """Turn head follow on ("look at me"). The face tracker keeps running either way."""
        return self._set_head_follow(True)

    async def stop_face_tracking(self) -> dict:
        """Turn head follow off ("stop looking"). Face detection keeps running."""
        return self._set_head_follow(False)

    def _set_head_follow(self, enabled: bool) -> dict:
        self.tracking_bridge.enabled = enabled
        result: dict = {
            "ok": True,
            "enabled": enabled,
            "face_detector_running": self.face_tracker.running,
        }
        if enabled and not self.face_tracker.running:
            result["warning"] = "face tracker is not running; head will not follow"
        return result

    async def _handle_external_local_tool(self, name: str, arguments: dict) -> dict:
        """Handle gateway-local tools called through the external command socket."""
        if name == "set_face_tracking":
            if not isinstance(arguments.get("enabled"), bool):
                return {"ok": False, "error": "enabled must be a boolean"}
            return await (self.start_face_tracking() if arguments["enabled"] else self.stop_face_tracking())
        if name == "set_voice_mode":
            return self.voice_input_bridge.set_enabled(
                bool(arguments.get("enabled")),
                surface=str(arguments.get("surface") or "") or None,
            )
        if name == "say":
            try:
                return await synthesize_and_send(arguments, gateway=self)
            except (ValueError, NotImplementedError, RuntimeError) as exc:
                return {"error": str(exc)}
        if name == "gemini_say":
            return await self._gemini_say(arguments)
        raise RuntimeError(f"Unknown local tool: {name}")

    async def _gemini_say(self, arguments: dict) -> dict:
        """Speak ``text`` in the robot's live Gemini voice (STACKCHAN_GEMINI_VOICE, default Kore).

        Reuses the existing GeminiVoiceProxy.speak_text path so a proactive
        notifier (e.g. a Claude Code Stop hook) can make the robot talk in its
        own voice instead of the separate Qwen3/VOICEVOX TTS used by ``say``.
        """
        text = str(arguments.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "text is required"}
        connection = getattr(self.esp32, "connection", None)
        proxy = getattr(connection, "cloud_proxy", None)
        speak = getattr(proxy, "speak_text", None)
        if speak is None:
            return {"ok": False, "error": "no active gemini voice session"}
        return await speak(
            text,
            session_id=str(arguments.get("session_id") or "hook-notify"),
            prompt_audio_frames=[],
            emotion=str(arguments.get("emotion")) if arguments.get("emotion") else None,
        )

    @property
    def voice_backend(self) -> str:
        """Voice backend name (``gemini`` default, ``xiaozhi`` to fall back)."""
        return os.getenv("STACKCHAN_VOICE_BACKEND", "gemini").strip().lower()

    def _make_voice_proxy(self):
        """Per-connection voice backend factory.

        Called every time a device WebSocket comes up. ``self.esp32`` and
        ``self.usb_transport`` are deferenced lazily so the lambda doesn't
        need a forward reference at __init__ time.
        """
        backend = self.voice_backend
        if backend in {"gemini", "gbot", "deepseek"}:
            proxy_class = {
                "gemini": GeminiVoiceProxy,
                "gbot": GbotVoiceProxy,
                "deepseek": DeepSeekVoiceProxy,
            }[backend]
            return proxy_class(
                esp32_ref=lambda: self.esp32,
                voice_bridge=self.voice_input_bridge,
                usb_transport=self.usb_transport,
                on_head_command=self.idle_gate.notify_head_command,
                on_device_state=self.idle_gate.notify_device_state,
                set_face_tracking=self._set_face_tracking_from_voice,
            )
        if backend == "xiaozhi":
            return CloudProxy(voice_bridge=self.voice_input_bridge)
        logger.warning(
            "Unknown STACKCHAN_VOICE_BACKEND=%r; defaulting to gemini",
            backend,
        )
        return GeminiVoiceProxy(
            esp32_ref=lambda: self.esp32,
            voice_bridge=self.voice_input_bridge,
            usb_transport=self.usb_transport,
            on_head_command=self.idle_gate.notify_head_command,
            on_device_state=self.idle_gate.notify_device_state,
            set_face_tracking=self._set_face_tracking_from_voice,
        )

    async def _set_face_tracking_from_voice(self, enabled: bool) -> dict:
        return await self._handle_external_local_tool("set_face_tracking", {"enabled": enabled})

    @property
    def vision_url(self) -> str:
        """URL for ESP32 to POST captured photos to.

        VISION_URL can be set to a complete public capture URL for remote
        access setups such as Tailscale Funnel. Otherwise VISION_HOST should
        be the LAN IP of the host running this gateway, as seen from the ESP32
        (e.g. something like 192.168.x.y on a typical home network). Falls
        back to "127.0.0.1" with a warning if unset; in that case the ESP32
        will not be able to reach the capture endpoint over the network.
        """
        explicit_url = os.getenv("VISION_URL")
        if explicit_url:
            return explicit_url

        host = os.getenv("VISION_HOST")
        if not host:
            logger.warning(
                "VISION_URL/VISION_HOST not set; defaulting to 127.0.0.1. "
                "ESP32 will not reach the capture endpoint unless "
                "VISION_HOST is set to this host's LAN IP or VISION_URL is "
                "set to a full capture URL."
            )
            host = "127.0.0.1"
        port = self.capture_port
        return f"http://{host}:{port}/capture"

    @property
    def vision_token(self) -> str:
        """Bearer token expected by the capture endpoint.

        VISION_TOKEN can be set separately. By default, reuse the ESP32
        WebSocket token so remote capture uploads are protected whenever the
        gateway itself is protected.
        """
        return (
            os.getenv("VISION_TOKEN")
            or os.getenv("STACKCHAN_TOKEN")
            or os.getenv("BEARER_TOKEN")
            or ""
        )

    @property
    def voice_proxy_url(self) -> str:
        """xiaozhi cloud WebSocket URL used by the voice proxy."""
        return os.getenv("XIAOZHI_CLOUD_URL", DEFAULT_CLOUD_URL)

    async def start(self) -> None:
        """Start the ESP32 WebSocket server and HTTP capture server."""
        host = os.getenv("HOST", "0.0.0.0")
        ws_port = int(os.getenv("WS_PORT", os.getenv("PORT", "8765")))
        capture_port = self.capture_port
        logger.info("claude CLI resolved to %s", resolve_claude_bin())

        # Start WebSocket server for ESP32
        await self.esp32.start(
            host,
            ws_port,
            vision_url=self.vision_url,
            vision_token=self.vision_token,
        )
        await self.touch_bridge.start()
        await self.idle_behavior.start()

        # USB transport is best-effort: if /dev/cu.usbmodem* isn't there yet
        # (cable unplugged, device booting), start() returns False and the
        # reader thread keeps retrying in the background. TrackingBridge
        # falls back to WS until USB comes online.
        # macOS /dev/cu.usbmodem* is exclusive — log a
        # visible reminder so flashing/monitoring doesn't silently collide.
        if self.usb_transport is not None:
            await self.usb_transport.start(wait_connect=False)
            logger.warning(
                "USB transport holds /dev/cu.usbmodem* exclusively. "
                "Stop the gateway or unset STACKCHAN_USB_TRANSPORT before "
                "running idf.py flash / monitor / screen.",
            )

        await self.idle_gate.start()

        # Start HTTP capture server (also hosts /track for face tracking)
        get_debug_status().face_tracking_provider = self.face_tracking_status
        app = create_capture_app(
            capture_token=self.vision_token,
            inject_text_handler=self._inject_debug_text,
        )
        app.router.add_post("/track", self._handle_track)
        register_console_routes(app, self)
        app.router.add_post("/demo/intro", self._handle_demo_intro)
        app.router.add_get("/demo/intro", self._handle_demo_intro)
        app.router.add_post("/demo/arm", self._handle_demo_arm)
        app.router.add_get("/demo/arm", self._handle_demo_arm)
        self._http_runner = web.AppRunner(
            app,
            access_log_class=TrackQuietAccessLogger,
        )
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, host, capture_port)
        await site.start()

        self._running = True
        logger.info(
            "Gateway started: WS on %s:%d, capture on %s:%d, vision_url=%s, voice_backend=%s",
            host, ws_port, host, capture_port, self.vision_url,
            self.voice_proxy_url if self.voice_backend == "xiaozhi" else self.voice_backend,
        )
        # /track is up; now launch the face tracker that posts to it, unless
        # the console switched the camera off.
        if get_console_settings().get("face_tracking"):
            self.face_tracker.start(f"http://127.0.0.1:{capture_port}/track")
        else:
            logger.info("face tracker off (console setting); the Mac camera stays unused")

    async def _handle_track(self, request: web.Request) -> web.Response:
        """Forward face-detection JSON from Vision Tracker to TrackingBridge.

        Co-located with /capture so the live-view server only needs one
        URL to reach the gateway. Fail-safe: any handler exception returns
        HTTP 200 with {"ok": False} so the upstream tracker keeps streaming
        rather than tearing the pipeline down.
        """
        try:
            detection = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)
        if isinstance(detection, dict):
            self._detection_count += 1
            self._last_detection = {
                key: detection.get(key)
                for key in ("x", "y", "width", "height", "confidence", "timestamp")
            }
        try:
            moved = await self.tracking_bridge.handle_detection(detection)
        except Exception as exc:
            logger.warning("tracking_bridge handle_detection failed: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)})
        status = self.esp32.get_status()
        if (
            status.get("connected")
            and status.get("initialized")
            and self.demo_reactions.maybe_face_entered(detection)
        ):
            self.demo_reactions.spawn_face_reaction()
        return web.json_response({"ok": True, "moved": moved})

    async def _handle_demo_intro(self, _request: web.Request) -> web.Response:
        """Manual trigger for the scripted self-introduction (demo reactions)."""
        self.demo_reactions.spawn_intro()
        return web.json_response({"ok": True})

    async def _handle_demo_arm(self, _request: web.Request) -> web.Response:
        """Arm the next face detection for the scripted demo reactions."""
        self.demo_reactions.arm_face_entry()
        return web.json_response({"ok": True})

    async def _inject_debug_text(self, text: str) -> bool:
        """Inject a plain user text turn into the active Gemini Live session."""
        connection = getattr(self.esp32, "connection", None)
        proxy = getattr(connection, "cloud_proxy", None)
        bridge = getattr(proxy, "_bridge", None)
        inject = getattr(bridge, "inject_user_text", None)
        if inject is None:
            return False
        result = inject(text)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    def _handle_touch_event(self, event) -> None:
        """React to a debounced head tap without involving Gemini/Claude."""
        if self.demo_reactions.maybe_head_tap(event):
            self.demo_reactions.spawn_touch_reaction()

    async def _run_wake_response(self) -> None:
        """Wake word response: head and blue LED in parallel.

        TTS removed (fix-gemini-one-round): speaking out a wake greeting via
        edge-tts opens AudioOut on the device, which mutes the microphone for
        the duration of the playback. The user's first sentence after the
        wake word would be silently dropped. We rely on Gemini Live's own
        audio reply to acknowledge the wake.
        """
        await asyncio.gather(
            self._set_wake_head(),
            self._set_wake_led_blue(),
        )

    async def _set_wake_head(self) -> None:
        await self._call_visual_tool(
            "self.robot.set_head_angles",
            {"yaw": 0, "pitch": 0, "speed": 700},
        )

    async def _set_wake_led_blue(self) -> None:
        await self._call_visual_tool(
            "self.led.set_all",
            {"r": 0, "g": 100, "b": 255},
        )

    async def _handle_end_conversation(self) -> dict[str, bool]:
        """End the current voice turn and return visuals to the baseline."""
        await asyncio.gather(
            self._set_conversation_led_off(),
            self._set_wake_head(),
            self._clear_current_gemini_context(),
        )
        return {"ok": True}

    async def _set_conversation_led_off(self) -> None:
        await self._call_visual_tool(
            "self.led.set_all",
            {"r": 0, "g": 0, "b": 0},
        )

    async def _clear_current_gemini_context(self) -> None:
        connection = getattr(self.esp32, "connection", None)
        proxy = getattr(connection, "cloud_proxy", None)
        bridge = getattr(proxy, "_bridge", None)
        clear_context = getattr(bridge, "clear_conversation_context", None)
        if clear_context is None:
            return
        result = clear_context()
        if inspect.isawaitable(result):
            await result

    async def _call_visual_tool(self, name: str, arguments: dict) -> None:
        usb = self.usb_transport
        if usb is not None and getattr(usb, "connected", False):
            try:
                await usb.call_tool(name, arguments, timeout_s=0.08)
                return
            except Exception as exc:
                logger.debug("wake response USB %s failed: %s", name, exc)
        result, error = await self.esp32.call_tool(name, arguments)
        if error:
            logger.warning("wake response tool %s failed: %s", name, error)
            await self._fallback_visual_emotion(name, arguments, error)

    async def _fallback_visual_emotion(
        self,
        name: str,
        arguments: dict,
        error: dict,
    ) -> None:
        message = str(error.get("message", ""))
        if "features.mcp=false" not in message:
            return
        emotion: str | None = None
        if name == "self.led.set_all":
            emotion = (
                IDLE_EMOTION
                if not any(int(arguments.get(channel, 0)) for channel in ("r", "g", "b"))
                else LISTENING_EMOTION
            )
        elif name == "self.display.set_avatar":
            emotion = face_to_device_emotion(str(arguments.get("face", "")))
        if emotion is None:
            return
        send_emotion = getattr(self.esp32, "send_emotion", None)
        if not callable(send_emotion):
            return
        _result, send_error = await send_emotion(emotion, notify_activity=False)
        if send_error:
            logger.debug("wake response emotion fallback failed: %s", send_error)

    async def stop(self) -> None:
        """Stop the gateway."""
        self._running = False
        await self.face_tracker.stop()
        await self.demo_reactions.stop()
        if self._http_runner:
            await self._http_runner.cleanup()
            self._http_runner = None
        await self.idle_gate.stop()
        await self.idle_behavior.stop()
        await self.tracking_bridge.stop()
        await self.touch_bridge.stop()
        if self.usb_transport is not None:
            await self.usb_transport.stop()
        await self.esp32.stop()
        logger.info("Gateway stopped")


# Singleton gateway instance, shared between stdio server and ESP32 manager
_gateway: Gateway | None = None


def get_gateway() -> Gateway:
    """Get or create the singleton gateway."""
    global _gateway
    if _gateway is None:
        _gateway = Gateway()
    return _gateway
