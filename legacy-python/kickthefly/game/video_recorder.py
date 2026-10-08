"""Video recording of any length: MP4 (or WebM) streamed to ffmpeg, or an animated GIF when ffmpeg isn't installed.

Frames are paced by the wall clock, not by the game's frame rate: at 30 fps the recorder takes a frame whenever one is
due, duplicating the last frame if the game drew slower and skipping frames if it drew faster, so the video always
plays back at real speed. The ffmpeg pipe is fed from a worker thread so a slow encoder never stalls the game; if it
falls far behind, frames are dropped rather than blocking.

The GIF fallback keeps its frames in memory, so it is downscaled (at most GIF_MAX_W wide, GIF_FPS frames per second)
and capped at GIF_MAX_S seconds, after which it stops taking frames and says so on the badge.
"""
from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import pygame

log = logging.getLogger(__name__)

VIDEO_FPS = 30
GIF_FPS = 12
GIF_MAX_W = 480
GIF_MAX_S = 60.0
MAX_DUPLICATES = 2 * VIDEO_FPS      # a longer stall (loading a save, a hitch) is not padded out beyond 2 s


def ffmpeg_available() -> bool:
    """True if an ffmpeg executable is on PATH."""
    return bool(shutil.which("ffmpeg"))


class VideoRecorder:
    """Streams frames to ffmpeg through stdin, or collects a downscaled animated GIF."""

    def __init__(self) -> None:
        self.is_recording: bool = False
        self.output_path: Path | None = None
        self.start_time: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.fps: int = VIDEO_FPS
        self.use_ffmpeg: bool = False
        self.gif_full: bool = False
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[bytes | None] | None = None
        self._worker: threading.Thread | None = None
        self._gif_frames: list = []
        self._gif_size: tuple[int, int] = (0, 0)
        self._frames_written: int = 0
        self._dropped: int = 0
        self._error: str | None = None

    def start(self, output_path: str | Path, width: int = 1280, height: int = 760, fps: int = VIDEO_FPS) -> Path:
        """Start recording to output_path. A path without a suffix gets .mp4 (or .gif without ffmpeg); an .mp4, .webm,
        .mkv or .mov path becomes .gif when ffmpeg is missing."""
        if self.is_recording:
            assert self.output_path is not None
            return self.output_path
        has_ffmpeg = ffmpeg_available()
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.suffix:
            target = target.with_suffix(".mp4" if has_ffmpeg else ".gif")

        self.width = max(2, width - (width % 2))           # H.264 and VP9 need even dimensions
        self.height = max(2, height - (height % 2))
        self.fps = max(1, int(fps))
        self.start_time = time.perf_counter()
        self._frames_written = 0
        self._dropped = 0
        self._error = None
        self.gif_full = False

        ext = target.suffix.lower().lstrip(".")
        self.use_ffmpeg = has_ffmpeg and ext in ("mp4", "webm", "mkv", "mov")
        if self.use_ffmpeg:
            self.output_path = target
            if ext == "webm":
                vcodec = ["-c:v", "libvpx-vp9", "-b:v", "2M", "-deadline", "realtime", "-cpu-used", "8"]
            else:
                vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"]
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
                   *vcodec, "-pix_fmt", "yuv420p", str(self.output_path)]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
                self._queue = queue.Queue(maxsize=3 * self.fps)
                self._worker = threading.Thread(target=self._pipe_worker, name="video-recorder", daemon=True)
                self._worker.start()
            except OSError as e:
                log.warning("ffmpeg failed to start (%s); recording a GIF instead", e)
                self.use_ffmpeg = False
        if not self.use_ffmpeg:
            self.output_path = target if ext == "gif" else target.with_suffix(".gif")
            self._gif_frames = []
            s = min(1.0, GIF_MAX_W / self.width)
            self._gif_size = (max(2, int(self.width * s)), max(2, int(self.height * s)))
        self.is_recording = True
        return self.output_path

    def _pipe_worker(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(item)
            except (BrokenPipeError, OSError, AssertionError) as e:
                self._error = str(e) or "ffmpeg closed its input"
                break

    # --- frames -------------------------------------------------------------------------------------------------------
    def frames_due(self, now: float | None = None) -> int:
        """How many frames the video needs now to stay in step with the wall clock (0: skip this game frame)."""
        if not self.is_recording or self.gif_full:
            return 0
        fps = self.fps if self.use_ffmpeg else GIF_FPS
        due = int(self.elapsed(now) * fps) + 1 - self._frames_written
        return max(0, min(due, MAX_DUPLICATES))

    def capture(self, grab: Callable[[], tuple[bytes, tuple[int, int]]], now: float | None = None) -> None:
        """Call once per drawn game frame. grab() returns (raw RGB bytes, (w, h)) and is only called when a frame is
        due, so the (GPU) readback is skipped on frames the video doesn't need."""
        n = self.frames_due(now)
        if n:
            raw, size = grab()
            self.write_frame_bytes(raw, size, repeat=n)

    def write_frame_bytes(self, raw_rgb: bytes, size: tuple[int, int] | None = None, repeat: int = 1) -> None:
        """Append one frame (repeat times) of raw RGB bytes. size defaults to the recording size; a frame of another
        size (the window was resized mid-recording) is scaled to fit."""
        if not self.is_recording:
            return
        size = size or (self.width, self.height)
        if self.use_ffmpeg:
            if size != (self.width, self.height):
                raw_rgb = self._scale(raw_rgb, size, (self.width, self.height))
            assert self._queue is not None
            for _ in range(max(1, repeat)):
                try:
                    self._queue.put_nowait(raw_rgb)
                except queue.Full:
                    self._dropped += 1
                self._frames_written += 1
        else:
            if len(self._gif_frames) >= int(GIF_MAX_S * GIF_FPS):
                self.gif_full = True
                return
            from PIL import Image
            img = Image.frombytes("RGB", size, raw_rgb)
            if img.size != self._gif_size:
                img = img.resize(self._gif_size, Image.BILINEAR)
            frame = img.convert("P", palette=Image.ADAPTIVE, colors=128)   # 1 byte a pixel while it waits
            for _ in range(max(1, repeat)):
                self._gif_frames.append(frame)
                self._frames_written += 1

    def write_frame_surface(self, surf: pygame.Surface, now: float | None = None) -> None:
        """Record a pygame surface (the 2D game), paced by the wall clock like capture()."""
        self.capture(lambda: (pygame.image.tobytes(surf, "RGB"), surf.get_size()), now)

    @staticmethod
    def _scale(raw: bytes, size: tuple[int, int], to: tuple[int, int]) -> bytes:
        surf = pygame.image.frombytes(raw, size, "RGB")
        return pygame.image.tobytes(pygame.transform.smoothscale(surf, to), "RGB")

    # --- status -------------------------------------------------------------------------------------------------------
    def elapsed(self, now: float | None = None) -> float:
        if not self.is_recording:
            return 0.0
        return max(0.0, (time.perf_counter() if now is None else now) - self.start_time)

    def badge_text(self, now: float | None = None) -> str:
        """The HUD badge, e.g. '[REC 01:23]', or '[GIF FULL 01:00]' once the GIF fallback hit its cap."""
        m, s = divmod(int(self.elapsed(now)), 60)
        return f"[{'GIF FULL' if self.gif_full else 'REC'} {m:02d}:{s:02d}]"

    def stop(self) -> Path | None:
        """Finish the file and return its path, or None if nothing usable was written."""
        if not self.is_recording:
            return None
        self.is_recording = False
        out = self.output_path
        if self.use_ffmpeg and self._proc:
            if self._queue is not None:
                self._queue.put(None)
            if self._worker is not None:
                self._worker.join(timeout=30.0)
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                _, err = self._proc.communicate(timeout=60.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                err = b"timed out"
            code = self._proc.returncode
            self._proc = self._queue = self._worker = None
            if self._dropped:
                log.info("video: dropped %d of %d frames (encoder behind)", self._dropped, self._frames_written)
            if code != 0 or self._error:
                log.error("ffmpeg failed (exit %s): %s", code, (err or b"").decode(errors="replace").strip()
                          or self._error)
                return None
        elif self._gif_frames:
            try:
                frames = self._gif_frames
                frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / GIF_FPS), loop=0,
                               optimize=False)
            except Exception as e:
                log.error("saving the GIF failed: %s", e)
                out = None
            self._gif_frames = []
        else:
            out = None
        return out if out is not None and out.exists() else None
