# SPDX-FileCopyrightText: 2026 sefuzhou770801-hub
# SPDX-License-Identifier: MIT
"""Keep the Mac face tracker running for as long as the gateway runs.

The tracker (``tools/vision-tracker``) posts a detection to ``/track`` only
while it sees a face; ``TrackingBridge`` turns those detections into head
motion. The tracker starts with the gateway and is restarted with backoff if
it exits. It never saves or serves camera frames; ``tools/vision-tracker/
live-view`` is a separate debugging page and is not started here.

A missing binary or a failed launch is logged once and the gateway keeps
running without face tracking. Set ``STACKCHAN_FACE_TRACKER_AUTOSTART=0`` on
machines without a camera, ``STACKCHAN_FACE_TRACKER_BIN`` to run a binary
from somewhere else, and ``STACKCHAN_FACE_TRACKER_CAMERA`` to pick a camera by
(part of) its name instead of the tracker's default order.

The tracker only counts as available once the running process has reported
at least one face; a process that is alive but waiting for camera permission
(or looking at a covered camera) is running but not available.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TRACKER_FPS = 8
TRACKER_BINARY_NAME = "groki-vision-tracker"
# The tracker exits with this code when the camera is missing or not authorized.
CAMERA_UNAVAILABLE_EXIT_CODE = 2


def default_tracker_binary() -> Path:
    override = os.getenv("STACKCHAN_FACE_TRACKER_BIN")
    if override and override.strip():
        return Path(override.strip()).expanduser()
    return (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "vision-tracker"
        / ".build"
        / "release"
        / TRACKER_BINARY_NAME
    )


def tracker_arguments(endpoint: str, fps: int) -> list[str]:
    args = ["--endpoint", endpoint, "--fps", str(fps)]
    camera = os.getenv("STACKCHAN_FACE_TRACKER_CAMERA", "").strip()
    if camera:
        args += ["--camera", camera]
    return args


def autostart_enabled() -> bool:
    value = os.getenv("STACKCHAN_FACE_TRACKER_AUTOSTART", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


class FaceTrackerSupervisor:
    """Run one tracker child process and restart it when it exits."""

    def __init__(
        self,
        *,
        fps: int = DEFAULT_TRACKER_FPS,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 300.0,
        stable_run_s: float = 30.0,
    ) -> None:
        self.fps = fps
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.stable_run_s = stable_run_s
        self._child: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._face_reported = False

    @property
    def running(self) -> bool:
        """True while a tracker process is running."""
        child = self._child
        return child is not None and child.returncode is None

    @property
    def available(self) -> bool:
        """True once the running process has reported a face since it started."""
        return self.running and self._face_reported

    @property
    def pid(self) -> int | None:
        return self._child.pid if self.running else None

    def notify_face_reported(self) -> None:
        self._face_reported = True

    def start(self, endpoint: str, *, force: bool = False) -> None:
        """Launch the tracker; ``force`` skips the env autostart check (console switch)."""
        if self._task is not None and not self._task.done():
            return
        if not force and not autostart_enabled():
            logger.info("face tracker autostart disabled by STACKCHAN_FACE_TRACKER_AUTOSTART")
            return
        self._task = asyncio.create_task(
            self._supervise(endpoint),
            name="stackchan-face-tracker",
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        child = self._child
        self._child = None
        if child is None:
            return
        if child.returncode is None:
            child.terminate()
            try:
                await asyncio.wait_for(child.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                child.kill()
                await child.wait()
        else:
            await child.wait()

    async def _supervise(self, endpoint: str) -> None:
        loop = asyncio.get_running_loop()
        backoff = self.initial_backoff_s
        while True:
            binary = default_tracker_binary()
            if not binary.is_file():
                logger.warning(
                    "face tracker unavailable: executable not found at %s "
                    "(build with: cd tools/vision-tracker && swift build -c release); "
                    "the head will not follow faces",
                    binary,
                )
                return
            try:
                child = await asyncio.create_subprocess_exec(
                    str(binary), *tracker_arguments(endpoint, self.fps),
                )
            except OSError as exc:
                logger.warning(
                    "face tracker failed to start (%s): %s; the head will not follow faces",
                    binary,
                    exc,
                )
                return
            self._child = child
            self._face_reported = False
            started_at = loop.time()
            logger.info("face tracker started pid=%s endpoint=%s", child.pid, endpoint)
            returncode = await child.wait()
            # Without a camera or camera permission the tracker may wait a long
            # time before it exits; that does not count as a stable run.
            if (
                returncode != CAMERA_UNAVAILABLE_EXIT_CODE
                and loop.time() - started_at >= self.stable_run_s
            ):
                backoff = self.initial_backoff_s
            logger.warning(
                "face tracker exited returncode=%s; restarting in %.1fs",
                returncode,
                backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)
