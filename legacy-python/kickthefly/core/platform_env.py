"""OS-specific startup: the Linux display backend, Windows DPI awareness, and consoles for headless runs.

Linux: native Wayland is preferred when WAYLAND_DISPLAY is set, with an automatic fall back to X11/XWayland if SDL
can't open a Wayland display. --backend wayland|x11 (or the Graphics setting, applied on restart) overrides the choice.
An SDL_VIDEODRIVER already set in the environment always wins.

Windows: the process is made per-monitor DPI aware before the window exists, so 125%/150% displays render at native
resolution instead of being bitmap-stretched (blurry) by Windows.
"""
from __future__ import annotations

import os
import sys

from kickthefly.core.crash import info, log

BACKENDS = ("auto", "wayland", "x11")


def choose_backend(cli: str | None, configured: str | None, env: dict | None = None,
                   platform: str | None = None) -> str | None:
    """The SDL_VIDEODRIVER to try first, or None to let SDL decide. Only meaningful on Linux."""
    env = os.environ if env is None else env
    if not (platform or sys.platform).startswith("linux"):
        return None
    if env.get("SDL_VIDEODRIVER"):
        return env["SDL_VIDEODRIVER"]
    for choice in (cli, configured):
        if choice in ("wayland", "x11"):
            return choice
    return "wayland" if env.get("WAYLAND_DISPLAY") else None


def init_video(cli_backend: str | None = None, configured: str | None = None) -> str:
    """Initialize SDL video with the chosen backend, falling back to X11 if Wayland fails. Returns the driver name."""
    import pygame

    user_forced = bool(os.environ.get("SDL_VIDEODRIVER"))
    want = choose_backend(cli_backend, configured)
    if want and not user_forced:
        os.environ["SDL_VIDEODRIVER"] = want
    try:
        pygame.display.init()
    except pygame.error as e:
        if want == "wayland" and not user_forced:
            log.warning("native Wayland display failed (%s); falling back to X11/XWayland", e)
            pygame.display.quit()
            os.environ["SDL_VIDEODRIVER"] = "x11"
            pygame.display.init()
        else:
            raise
    driver = pygame.display.get_driver()
    info["video_driver"] = driver
    log.info("video driver: %s", driver)
    return driver


def reset_to_x11() -> bool:
    """Called when a GL window can't be made on Wayland: retry on XWayland. False if already on X11."""
    import pygame

    if os.environ.get("SDL_VIDEODRIVER") != "wayland" or not os.environ.get("DISPLAY"):
        return False
    log.warning("OpenGL window failed on Wayland; retrying on X11/XWayland")
    pygame.display.quit()
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    info["video_driver"] = pygame.display.get_driver()
    return True


def windows_dpi_aware() -> float:
    """Per-monitor DPI awareness (v2 where available). Returns the primary display's scale (1.0 = 100%)."""
    if sys.platform != "win32":
        return 1.0
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")    # SDL >= 2.24 applies it too
    scale = 1.0
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            ok = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))   # PER_MONITOR_AWARE_V2
        except AttributeError:
            ok = 0
        if not ok:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)               # PROCESS_PER_MONITOR_DPI_AWARE
            except Exception:
                user32.SetProcessDPIAware()
        try:
            scale = user32.GetDpiForSystem() / 96.0
        except AttributeError:
            pass
    except Exception as e:
        log.warning("could not set DPI awareness: %s", e)
    info["dpi_scale"] = f"{scale:.2f}"
    return scale


def attach_console() -> None:
    """The exe is a windowed app with no console. For --headless, borrow the console it was started from, if any."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        import ctypes

        if ctypes.windll.kernel32.AttachConsole(-1):                        # ATTACH_PARENT_PROCESS
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except Exception:
        pass
    for name in ("stdout", "stderr"):                                        # no console at all: don't crash on print
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))


def no_display_available() -> bool:
    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")


def available_memory_mb() -> float | None:
    """Physical memory the OS could give us now, in MB (Linux MemAvailable, Windows ullAvailPhys), or None if unknown."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024.0
        elif sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullAvailPhys / (1024.0 * 1024.0)
    except (OSError, ValueError, AttributeError):
        pass
    return None
