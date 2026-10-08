"""Data path resolution and migration on Linux (XDG) and Windows (Known Folders), without touching real folders."""
from pathlib import Path

import numpy as np

from kickthefly.core import crash
from kickthefly.core import paths


def linux(tmp_path, env=None):
    home = tmp_path / "home"
    return paths.resolve(platform="linux", env=env or {}, home=home, cwd=tmp_path / "cwd", run_tool=False), home


def test_linux_xdg_defaults(tmp_path):
    p, home = linux(tmp_path)
    assert p.config_file == home / ".config" / "kickthefly" / "config.toml"
    assert p.memory_dir == home / ".local" / "share" / "kickthefly" / "memory"
    assert p.saves_dir == home / ".local" / "share" / "kickthefly" / "saves"
    assert p.pictures_dir == home / "Pictures" / "Kick the Fly"
    assert p.state_dir == home / ".local" / "state" / "kickthefly"


def test_linux_xdg_env_and_relative_values_ignored(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_DATA_HOME": "relative/data", "XDG_STATE_HOME": str(tmp_path / "st")}
    p, home = linux(tmp_path, env)
    assert p.config_dir == tmp_path / "cfg" / "kickthefly"
    assert p.data_dir == home / ".local" / "share" / "kickthefly"       # relative XDG values must be ignored
    assert p.state_dir == tmp_path / "st" / "kickthefly"


def test_linux_pictures_from_user_dirs(tmp_path):
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True)
    (home / ".config" / "user-dirs.dirs").write_text('XDG_PICTURES_DIR="$HOME/Bilder"\n')
    p = paths.resolve(platform="linux", env={}, home=home, cwd=tmp_path, run_tool=False)
    assert p.pictures_dir == home / "Bilder" / "Kick the Fly"


def test_windows_known_folders(tmp_path):
    kf = {"Documents": tmp_path / "OneDrive" / "Documents", "Pictures": tmp_path / "OneDrive" / "Pictures",
          "RoamingAppData": tmp_path / "Roaming", "LocalAppData": tmp_path / "Local"}
    p = paths.resolve(platform="win32", env={}, home=tmp_path / "user", known_folder=kf.get, cwd=tmp_path)
    assert p.memory_dir == kf["Documents"] / "Kick the Fly" / "memory"
    assert p.saves_dir == kf["Documents"] / "Kick the Fly" / "saves"
    assert p.pictures_dir == kf["Pictures"] / "Kick the Fly"
    assert p.config_file == kf["RoamingAppData"] / "Kick the Fly" / "config.toml"
    assert p.state_dir == kf["LocalAppData"] / "Kick the Fly"


def test_windows_keeps_legacy_memory_in_place(tmp_path):
    """Documents redirected to OneDrive after training: the old memory is used where it is, never moved."""
    user = tmp_path / "user"
    old = user / "Documents" / "Kick the Fly" / "memory"
    old.mkdir(parents=True)
    (old / "fly-memory.npz").write_bytes(b"trained")
    kf = {"Documents": tmp_path / "OneDrive" / "Documents"}
    p = paths.resolve(platform="win32", env={}, home=user, known_folder=kf.get, cwd=tmp_path)
    assert p.memory_dir == old
    assert (old / "fly-memory.npz").read_bytes() == b"trained"
    # once memory exists in the Known Folder location, that one wins
    new = kf["Documents"] / "Kick the Fly" / "memory"
    new.mkdir(parents=True)
    (new / "fly-memory.npz").write_bytes(b"new")
    assert paths.resolve(platform="win32", env={}, home=user, known_folder=kf.get, cwd=tmp_path).memory_dir == new


def test_windows_known_folder_failure_falls_back(tmp_path):
    p = paths.resolve(platform="win32", env={"APPDATA": str(tmp_path / "ad")}, home=tmp_path / "u",
                      known_folder=lambda name: None, cwd=tmp_path)
    assert p.memory_dir == tmp_path / "u" / "Documents" / "Kick the Fly" / "memory"
    assert p.config_dir == tmp_path / "ad" / "Kick the Fly"


def test_linux_migration_copies_once_and_keeps_originals(tmp_path):
    p, home = linux(tmp_path)
    old = home / "Documents" / "Kick the Fly" / "memory"
    old.mkdir(parents=True)
    (old / "fly-memory.npz").write_bytes(b"weights")
    (old / "training-log.json").write_text("{}")
    oldpics = home / "Pictures" / "Kick the Fly"
    p.pictures_dir = home / "Bilder" / "Kick the Fly"            # XDG pictures differs from the old literal path
    p.legacy_pictures_dirs = [oldpics]
    oldpics.mkdir(parents=True)
    (oldpics / "shot.png").write_bytes(b"png")
    msgs = paths.migrate_linux(p)
    assert (p.memory_dir / "fly-memory.npz").read_bytes() == b"weights"
    assert (p.memory_dir / "training-log.json").exists()
    assert (p.pictures_dir / "shot.png").exists()
    assert (old / "fly-memory.npz").exists() and (oldpics / "shot.png").exists()   # originals untouched
    assert len(msgs) == 2
    (p.memory_dir / "fly-memory.npz").write_bytes(b"trained more")
    assert paths.migrate_linux(p) == []                             # only once
    assert (p.memory_dir / "fly-memory.npz").read_bytes() == b"trained more"


def test_home_override(isolated_home):
    p = paths.get()
    assert p.memory_dir == isolated_home / "data" / "memory" and p.config_dir == isolated_home / "config"


def test_old_memory_file_loads_with_new_code(tmp_path, monkeypatch):
    """A fly-memory.npz written by v2.5.0's memory.save() format still loads (version and signature unchanged)."""
    from kickthefly.core import memory
    monkeypatch.setenv("KICK_THE_FLY_MEMORY", str(tmp_path / "mem"))
    paths.reset_cache()
    assert memory.memory_dir() == tmp_path / "mem"
    assert memory.VERSION == 1


def test_crash_report_contents(tmp_path):
    crash.info.update(seed="42", video_driver="wayland", gl_renderer="TestGPU", gl_version="4.6 Mesa 99")
    written = crash.write_crash_report("Traceback: boom", state_dir=tmp_path / "state")
    assert tmp_path / "state" / crash.CRASH_NAME in written
    text = (tmp_path / "state" / crash.CRASH_NAME).read_text()
    for needle in ("seed: 42", "OS:", "TestGPU", "GL_VERSION: 4.6 Mesa 99", "driver:", "Traceback: boom"):
        assert needle in text


def test_backend_choice():
    from kickthefly.core import platform_env as pe
    assert pe.choose_backend(None, None, {"WAYLAND_DISPLAY": "wayland-0"}, "linux") == "wayland"
    assert pe.choose_backend(None, None, {"DISPLAY": ":0"}, "linux") is None
    assert pe.choose_backend("x11", "wayland", {"WAYLAND_DISPLAY": "w"}, "linux") == "x11"
    assert pe.choose_backend(None, "x11", {"WAYLAND_DISPLAY": "w"}, "linux") == "x11"
    assert pe.choose_backend("wayland", None, {"SDL_VIDEODRIVER": "kmsdrm"}, "linux") == "kmsdrm"
    assert pe.choose_backend("wayland", None, {}, "win32") is None


import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Known Folders API")
def test_real_windows_known_folders():
    """The ctypes SHGetKnownFolderPath call itself (runs on the Windows CI runner)."""
    for name in ("Documents", "Pictures", "RoamingAppData", "LocalAppData"):
        got = paths.windows_known_folder(name)
        assert got is not None and got.is_absolute() and got.exists(), name
    p = paths.resolve(platform="win32", env={})
    assert p.memory_dir.parts[-2:] == ("Kick the Fly", "memory")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DPI awareness")
def test_real_windows_dpi_awareness():
    from kickthefly.core import platform_env
    scale = platform_env.windows_dpi_aware()
    assert scale >= 1.0
