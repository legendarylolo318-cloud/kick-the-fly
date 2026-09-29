"""Logging, system diagnostics and crash reports.

A crash writes KickTheFly-crash.txt next to the exe or AppImage (when that folder is writable) and always into the
per-user state folder (paths.py). The report starts with what helps reproduce it: app version, seed, OS and version,
the Linux session type (Wayland/X11) and SDL video driver, the GPU's OpenGL vendor, renderer and version strings and
the driver version.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from kickthefly.core.version import __version__

log = logging.getLogger("kickthefly")
CRASH_NAME = "KickTheFly-crash.txt"
PREVIOUS_LOG = "kickthefly.previous.log"

# filled in as the game starts up (seed, video driver, GL strings, 2D/3D)
info: dict[str, str] = {}


def setup_logging(state_dir: Path | None = None, verbose: bool = False) -> None:
    if log.handlers:
        return
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    if sys.stderr is not None:                        # a windowed exe has no stderr
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        log.addHandler(h)
    if state_dir is not None:
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            cur = state_dir / "kickthefly.log"
            if cur.exists() and cur.stat().st_size:          # keep the last session's log (the one that may have
                os.replace(cur, state_dir / PREVIOUS_LOG)     # crashed) for Report a bug, instead of overwriting it
            fh = logging.FileHandler(cur, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError:
            pass
    log.propagate = False


def os_description() -> str:
    if sys.platform == "win32":
        try:
            v = sys.getwindowsversion()
            rel, ver, _, _ = platform.win32_ver()
            return f"Windows {rel} (version {ver}, build {v.build})"
        except Exception:
            return f"Windows {platform.version()}"
    if sys.platform.startswith("linux"):
        name = "Linux"
        try:
            for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
                if line.startswith("PRETTY_NAME="):
                    name = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return f"{name}, kernel {platform.release()}"
    return f"{platform.system()} {platform.release()}"


def session_description() -> str:
    if not sys.platform.startswith("linux"):
        return ""
    e = os.environ
    return (f"XDG_SESSION_TYPE={e.get('XDG_SESSION_TYPE', '')} WAYLAND_DISPLAY={e.get('WAYLAND_DISPLAY', '')} "
            f"DISPLAY={e.get('DISPLAY', '')} SDL_VIDEODRIVER={e.get('SDL_VIDEODRIVER', '')}")


def record_gl(ctx) -> None:
    """Remember the GL strings from a moderngl context (called once the 3D window is up)."""
    try:
        i = ctx.info
        info["gl_vendor"] = str(i.get("GL_VENDOR", ""))
        info["gl_renderer"] = str(i.get("GL_RENDERER", ""))
        info["gl_version"] = str(i.get("GL_VERSION", ""))
    except Exception as e:
        info["gl_error"] = str(e)


def driver_version() -> str:
    """Best effort: the GL version string usually carries it (Mesa x.y, NVIDIA 5xx); plus what the OS reports."""
    parts = []
    if info.get("gl_version"):
        parts.append(f"from GL_VERSION: {info['gl_version']}")
    if sys.platform.startswith("linux"):
        nv = Path("/sys/module/nvidia/version")
        try:
            if nv.exists():
                parts.append(f"nvidia kernel module {nv.read_text().strip()}")
        except OSError:
            pass
    elif sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' driver ' + $_.DriverVersion }"],
                capture_output=True, text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            parts += [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        except Exception:
            pass
    return "; ".join(parts) or "unknown"


def record_backend(backend: str, device: str = "") -> None:
    """Remember the compute backend and device name for diagnostics and crash reporting."""
    info["sim_backend"] = str(backend)
    if device:
        info["sim_device"] = str(device)


def report_text(exc_text: str) -> str:
    lines = [
        f"Kick the Fly {__version__} crash report, {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"seed: {info.get('seed', 'unknown')}",
        f"mode: {info.get('mode', 'unknown')}",
        f"sim backend: {info.get('sim_backend', 'unknown')}",
        f"sim device: {info.get('sim_device', 'unknown')}",
        f"OS: {os_description()}",
        f"python: {sys.version.split()[0]}  frozen: {bool(getattr(sys, 'frozen', False))}",
    ]
    if sys.platform.startswith("linux"):
        lines.append(f"session: {session_description()}")
    lines += [
        f"SDL video driver: {info.get('video_driver', 'unknown')}",
        f"GPU (GL_RENDERER): {info.get('gl_renderer', 'not created (2D or failed before the window)')}",
        f"GL_VENDOR: {info.get('gl_vendor', '')}",
        f"GL_VERSION: {info.get('gl_version', '')}",
        f"driver: {driver_version()}",
    ]
    lines += [f"{k}: {v}" for k, v in info.items()
              if k not in ("seed", "mode", "sim_backend", "sim_device", "video_driver", "gl_renderer", "gl_vendor", "gl_version")]
    return "\n".join(lines) + "\n\n" + exc_text


def beside_executable() -> Path | None:
    """The folder of the exe or AppImage the player launched (not PyInstaller's temp dir or the AppImage mount)."""
    if os.environ.get("APPIMAGE"):
        return Path(os.environ["APPIMAGE"]).resolve().parent
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def write_crash_report(exc_text: str | None = None, state_dir: Path | None = None) -> list[Path]:
    exc_text = exc_text or traceback.format_exc()
    text = report_text(exc_text)
    written = []
    targets = [beside_executable()]
    if state_dir is None:
        try:
            from kickthefly.core import paths
            state_dir = paths.get().state_dir
        except Exception:
            state_dir = None
    targets.append(state_dir)
    for d in targets:
        if d is None:
            continue
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / CRASH_NAME).write_text(text, encoding="utf-8")
            written.append(d / CRASH_NAME)
        except OSError:
            continue
    return written


class DebugCrash(RuntimeError):
    """The test crash KICK_THE_FLY_DEBUG_CRASH asks for (tests and QA of the crash screen and the bug report)."""


def maybe_debug_crash(seconds_running: float) -> None:
    """KICK_THE_FLY_DEBUG_CRASH=SECONDS: raise a test crash from the game loop after that long. Debug only; unset in
    normal use, where this costs one environment lookup per frame."""
    v = os.environ.get("KICK_THE_FLY_DEBUG_CRASH")
    if v:
        try:
            after = float(v)
        except ValueError:
            return
        if seconds_running >= after:
            raise DebugCrash(f"test crash requested by KICK_THE_FLY_DEBUG_CRASH={v}")


def handle_crash(exc_text: str | None = None) -> list[Path]:
    """Write the crash report, log where it went and, in a windowed run, show the crash screen (ui/crashscreen.py) with
    "Report a bug" on it. Headless runs (validation, CI) only get the report. Never raises."""
    written: list[Path] = []
    try:
        written = write_crash_report(exc_text)
        log.error("crashed; report written to %s", ", ".join(map(str, written)) or "nowhere (no writable folder)")
    except Exception:
        pass
    if info.get("mode") in ("2d", "3d") and not os.environ.get("KICK_THE_FLY_NO_CRASH_SCREEN"):
        try:
            from kickthefly.ui import crashscreen
            crashscreen.show(written)
        except Exception:
            log.exception("the crash screen itself failed")
    return written
