"""Where Kick the Fly keeps its files on each OS.

  Linux (XDG Base Directory spec)
    config           $XDG_CONFIG_HOME/kickthefly/config.toml       (~/.config/kickthefly)
    training memory  $XDG_DATA_HOME/kickthefly/memory              (~/.local/share/kickthefly/memory)
    save states      $XDG_DATA_HOME/kickthefly/saves
    screenshots/GIFs <xdg-user-dir PICTURES>/Kick the Fly          (~/Pictures/Kick the Fly)
    crash reports    $XDG_STATE_HOME/kickthefly                    (~/.local/state/kickthefly)
  Windows (Known Folders, so redirected and OneDrive folders work)
    config           %APPDATA%\\Kick the Fly\\config.toml
    training memory  Documents\\Kick the Fly\\memory                 (unchanged since v2.2)
    save states      Documents\\Kick the Fly\\saves
    screenshots/GIFs Pictures\\Kick the Fly                          (unchanged)
    crash reports    %LOCALAPPDATA%\\Kick the Fly

Older versions used ~/Documents/Kick the Fly/memory and ~/Pictures/Kick the Fly literally on every OS. On Linux that
data is copied (never moved or deleted) into the XDG locations on first launch. On Windows nothing is ever moved: if
the training memory only exists at the old literal path (e.g. Documents is now redirected to OneDrive), the game keeps
using the old folder so the fly remembers what it learned.

KICK_THE_FLY_HOME puts everything under one folder (portable installs, tests, CI). KICK_THE_FLY_MEMORY still overrides
just the memory folder, as before.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

XDG_NAME = "kickthefly"
WIN_NAME = "Kick the Fly"
MEMORY_FILES = ("fly-memory.npz", "training-log.json")

# Known Folder IDs (KNOWNFOLDERID) from ShlObj_core.h
FOLDERID = {
    "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "Pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "RoamingAppData": "{3EB685DB-65F9-4CF6-A03A-E3EF65729F3D}",
    "LocalAppData": "{F1B32785-6FBA-4FCF-9D55-7B8E7F157091}",
}


@dataclass
class AppPaths:
    config_dir: Path
    memory_dir: Path
    saves_dir: Path
    pictures_dir: Path
    state_dir: Path
    data_dir: Path
    legacy_memory_dirs: list[Path] = field(default_factory=list)
    legacy_pictures_dirs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"


def windows_known_folder(name: str) -> Path | None:
    """SHGetKnownFolderPath: follows folder redirection (OneDrive, network Documents). None off Windows or on error."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        import uuid

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", wintypes.BYTE * 8)]

        u = uuid.UUID(FOLDERID[name])
        guid = GUID(u.fields[0], u.fields[1], u.fields[2], (wintypes.BYTE * 8).from_buffer_copy(u.bytes[8:]))
        out = ctypes.c_wchar_p()
        shell32 = ctypes.windll.shell32
        shell32.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE,
                                                 ctypes.POINTER(ctypes.c_wchar_p)]
        if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) != 0:
            return None
        try:
            return Path(out.value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(out)
    except Exception:
        return None


def _xdg_base(env: dict, var: str, default: Path) -> Path:
    v = env.get(var, "")
    return Path(v) if v and os.path.isabs(v) else default     # the spec says relative values must be ignored


def xdg_user_dir(kind: str, env: dict, home: Path, run_tool: bool = True) -> Path:
    """xdg-user-dir PICTURES (and friends): the tool if present, else ~/.config/user-dirs.dirs, else ~/Pictures."""
    fallback = home / kind.capitalize()
    if run_tool and shutil.which("xdg-user-dir"):
        try:
            out = subprocess.run(["xdg-user-dir", kind], capture_output=True, text=True, timeout=2).stdout.strip()
            if out and Path(out) != home:                       # it prints $HOME when the dir isn't configured
                return Path(out)
        except (OSError, subprocess.SubprocessError):
            pass
    cfg = _xdg_base(env, "XDG_CONFIG_HOME", home / ".config") / "user-dirs.dirs"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"XDG_{kind}_DIR="):
                val = line.split("=", 1)[1].strip().strip('"').replace("$HOME", str(home))
                if val and Path(val) != home:
                    return Path(val)
    except OSError:
        pass
    return fallback


def resolve(platform: str | None = None, env: dict | None = None, home: Path | None = None,
            known_folder=windows_known_folder, cwd: Path | None = None, run_tool: bool = True) -> AppPaths:
    """Pure path resolution (no directories created). Arguments exist so tests can resolve other OSes' layouts."""
    platform = platform or sys.platform
    env = dict(os.environ) if env is None else env
    home = Path(home) if home is not None else Path.home()
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    legacy_mem = [home / "Documents" / WIN_NAME / "memory", cwd / "Kick the Fly memory"]
    legacy_pics = [home / "Pictures" / WIN_NAME, cwd / "Kick the Fly saves"]

    root = env.get("KICK_THE_FLY_HOME")
    if root:
        r = Path(root)
        p = AppPaths(config_dir=r / "config", data_dir=r / "data", memory_dir=r / "data" / "memory",
                     saves_dir=r / "data" / "saves", pictures_dir=r / "pictures", state_dir=r / "state")
    elif platform == "win32":
        def kf(name, fallback):
            got = known_folder(name)
            return Path(got) if got else fallback

        docs = kf("Documents", home / "Documents")
        appdata = kf("RoamingAppData", Path(env.get("APPDATA") or home / "AppData" / "Roaming"))
        local = kf("LocalAppData", Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local"))
        pics = kf("Pictures", home / "Pictures")
        base = docs / WIN_NAME
        p = AppPaths(config_dir=appdata / WIN_NAME, data_dir=base, memory_dir=base / "memory", saves_dir=base / "saves",
                     pictures_dir=pics / WIN_NAME, state_dir=local / WIN_NAME)
        # never move Windows data: if the memory only exists where older versions put it, keep using that folder
        if not (p.memory_dir / MEMORY_FILES[0]).exists():
            for old in legacy_mem:
                if old != p.memory_dir and (old / MEMORY_FILES[0]).exists():
                    p.notes.append(f"using existing training memory in {old}")
                    p.memory_dir = old
                    break
    elif platform == "darwin":
        sup = home / "Library" / "Application Support" / WIN_NAME
        p = AppPaths(config_dir=sup, data_dir=sup, memory_dir=sup / "memory", saves_dir=sup / "saves",
                     pictures_dir=home / "Pictures" / WIN_NAME, state_dir=home / "Library" / "Logs" / WIN_NAME)
    else:
        data = _xdg_base(env, "XDG_DATA_HOME", home / ".local" / "share") / XDG_NAME
        p = AppPaths(config_dir=_xdg_base(env, "XDG_CONFIG_HOME", home / ".config") / XDG_NAME, data_dir=data,
                     memory_dir=data / "memory", saves_dir=data / "saves",
                     pictures_dir=xdg_user_dir("PICTURES", env, home, run_tool) / WIN_NAME,
                     state_dir=_xdg_base(env, "XDG_STATE_HOME", home / ".local" / "state") / XDG_NAME)
    if env.get("KICK_THE_FLY_MEMORY"):
        p.memory_dir = Path(env["KICK_THE_FLY_MEMORY"])
    p.legacy_memory_dirs = [d for d in legacy_mem if d != p.memory_dir]
    p.legacy_pictures_dirs = [d for d in legacy_pics if d != p.pictures_dir]
    return p


def migrate_linux(p: AppPaths) -> list[str]:
    """Copy memory and screenshots from pre-2.6 locations into the XDG ones, once. Originals are left in place."""
    marker = p.data_dir / ".migrated-from-legacy"
    if marker.exists():
        return []
    done: list[str] = []
    try:
        if not (p.memory_dir / MEMORY_FILES[0]).exists():
            for old in p.legacy_memory_dirs:
                if (old / MEMORY_FILES[0]).is_file():
                    p.memory_dir.mkdir(parents=True, exist_ok=True)
                    for name in MEMORY_FILES:
                        if (old / name).is_file():
                            shutil.copy2(old / name, p.memory_dir / name)
                    done.append(f"copied training memory {old} -> {p.memory_dir}")
                    break
        for old in p.legacy_pictures_dirs:
            if old.is_dir() and old.resolve() != p.pictures_dir.resolve():
                files = [f for f in old.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".gif")]
                if files:
                    p.pictures_dir.mkdir(parents=True, exist_ok=True)
                    n = 0
                    for f in files:
                        if not (p.pictures_dir / f.name).exists():
                            shutil.copy2(f, p.pictures_dir / f.name)
                            n += 1
                    done.append(f"copied {n} screenshots/GIFs {old} -> {p.pictures_dir}")
        p.data_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n") + "\n".join(done) + "\n", encoding="utf-8")
    except OSError as e:
        done.append(f"migration skipped: {e}")
    return done


_cached: AppPaths | None = None


def get() -> AppPaths:
    """The resolved paths for this run (migration runs once, the first time they're asked for on Linux)."""
    global _cached
    if _cached is None:
        _cached = resolve()
        if sys.platform.startswith("linux") and not os.environ.get("KICK_THE_FLY_HOME") \
                and not os.environ.get("KICK_THE_FLY_MEMORY"):
            _cached.notes += migrate_linux(_cached)
    return _cached


def reset_cache() -> None:
    global _cached
    _cached = None


def ensure_dir(preferred: Path, *fallbacks: Path) -> Path:
    """The first of these folders that can be created, else the current directory."""
    for d in (preferred, *fallbacks):
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except OSError:
            continue
    return Path.cwd()
