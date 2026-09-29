"""--selftest and the in-game self-test (2.13): every check, the verdicts, the JSON, the exit codes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import ROOT, needs_pack
from kickthefly.core import selftest
from kickthefly.core.selftest import FAIL, PASS, WARN, Check


def checks(*statuses):
    return [Check(f"c{i}", f"check {i}", s, "detail", "fix it" if s != PASS else "") for i, s in enumerate(statuses)]


def test_exit_codes_zero_all_pass_one_any_fail_three_warnings_only():
    assert selftest.summarize(checks(PASS, PASS))["exit_code"] == 0
    assert selftest.summarize(checks(PASS, WARN))["exit_code"] == 3
    assert selftest.summarize(checks(WARN, FAIL, PASS))["exit_code"] == 1
    assert selftest.summarize([])["exit_code"] == 0


def test_the_text_says_what_to_fix_and_the_json_has_the_same_content():
    rep = selftest.summarize(checks(PASS, WARN, FAIL), app_version="9.9")
    text = selftest.to_text(rep)
    assert "[PASS] check 0" in text and "[WARN] check 1" in text and "[FAIL] check 2" in text
    assert text.count("fix: fix it") == 2 and "Exit code 1" in text
    back = json.loads(json.dumps(rep))
    assert back["counts"] == {"PASS": 1, "WARN": 1, "FAIL": 1} and len(back["checks"]) == 3
    assert {"id", "name", "status", "detail", "fix", "seconds"} <= set(back["checks"][0])


def test_a_check_that_raises_becomes_a_fail_not_a_crash(monkeypatch):
    from kickthefly.sim import brainpack

    def boom(brain="adult"):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(brainpack, "find", boom)
    c = selftest.check_brainpack_adult()
    assert c.status == FAIL and "disk on fire" in c.detail and c.fix


def test_brain_pack_checks(tmp_path, monkeypatch):
    import zipfile

    from kickthefly.sim import brainpack

    monkeypatch.setattr(brainpack, "find", lambda brain="adult": None)
    a, l = selftest.check_brainpack_adult(), selftest.check_brainpack_larva()
    assert a.status == FAIL and "kick_brain.npz" in a.detail and "brainpack build" in a.fix
    assert l.status == PASS and "optional" in l.detail, "the larva pack is optional"
    bad = tmp_path / "kick_brain.npz"
    bad.write_bytes(b"this is not a zip file")
    monkeypatch.setattr(brainpack, "find", lambda brain="adult": bad)
    c = selftest.check_brainpack_adult()
    assert c.status == FAIL and "damaged" in c.fix
    good = tmp_path / "kick_brain.npz"
    np.savez(good, n=np.array(12), type=np.array(["a"] * 12))
    c = selftest.check_brainpack_adult()
    assert c.status == WARN and "12 neurons" in c.detail, "the wrong neuron count is a warning"
    (tmp_path / "kick_brain.npz.sha256").write_text("0" * 64 + "  kick_brain.npz\n")
    c = selftest.check_brainpack_adult()
    assert c.status == FAIL and "does not match the published" in c.detail
    digest = selftest._sha256(good)
    (tmp_path / "kick_brain.npz.sha256").write_text(digest + "  kick_brain.npz\n")
    monkeypatch.setattr(selftest, "ADULT_NEURONS", 12)
    c = selftest.check_brainpack_adult()
    assert c.status == PASS and "matches the published checksum" in c.detail and c.data["sha256"] == digest
    (tmp_path / "kick_brain.npz.sha256").unlink()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  kick_brain.npz\n")
    assert selftest._published_checksum(good) == digest


def test_ffmpeg_audio_disk_and_memory_verdicts(monkeypatch):
    monkeypatch.setattr(selftest.shutil, "which", lambda name: None)
    c = selftest.check_ffmpeg()
    assert c.status == WARN and "GIF" in c.fix
    monkeypatch.setattr(selftest.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert selftest.check_ffmpeg().status == PASS
    import pygame

    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)
    monkeypatch.setattr(pygame.mixer, "init", lambda *a, **k: (_ for _ in ()).throw(pygame.error("no card")))
    c = selftest.check_audio()
    assert c.status == WARN and "silently" in c.fix
    from kickthefly.core import platform_env

    monkeypatch.setattr(platform_env, "available_memory_mb", lambda: 500.0)
    mem = next(c for c in selftest.check_resources() if c.id == "memory")
    assert mem.status == FAIL and mem.fix
    monkeypatch.setattr(platform_env, "available_memory_mb", lambda: 1500 + 400.0)
    assert next(c for c in selftest.check_resources() if c.id == "memory").status == WARN
    monkeypatch.setattr(platform_env, "available_memory_mb", lambda: 16000.0)
    res = {c.id: c for c in selftest.check_resources()}
    assert res["memory"].status == PASS and res["memory"].data["flies_fit"] > 10
    monkeypatch.setattr(selftest, "_free_disk_mb", lambda d: 50.0)
    assert next(c for c in selftest.check_resources() if c.id == "disk").status == FAIL
    monkeypatch.setattr(selftest, "_free_disk_mb", lambda d: 1000.0)
    assert next(c for c in selftest.check_resources() if c.id == "disk").status == WARN
    monkeypatch.setattr(selftest, "_free_disk_mb", lambda d: 90_000.0)
    assert next(c for c in selftest.check_resources() if c.id == "disk").status == PASS


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_folders_fail_with_a_fix(tmp_path, monkeypatch):
    from kickthefly.core import paths

    home = tmp_path / "ro"
    home.mkdir()
    home.chmod(0o500)
    try:
        monkeypatch.setenv("KICK_THE_FLY_HOME", str(home / "sub"))
        paths.reset_cache()
        cs = {c.id: c for c in selftest.check_folders()}
        assert cs["dir_config"].status == FAIL and "KICK_THE_FLY_HOME" in cs["dir_config"].fix
        assert cs["dir_screenshots"].status == WARN
    finally:
        home.chmod(0o700)
        paths.reset_cache()


def test_the_display_check_reports_the_session(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    c = selftest.check_display()
    assert c.status in (PASS, WARN) and (sys.platform != "linux" or c.data.get("session") == "none")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-9")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    c = selftest.check_display()
    if sys.platform.startswith("linux"):
        assert c.status == PASS and "wayland" in c.detail


def test_gpu_verdict_is_statistical_and_cpu_verdict_is_exact():
    ref = dict(spikes=1000, blocks=list(range(100)))
    ok, msg = selftest._statistical(ref, dict(spikes=1010, blocks=[b + 1 for b in range(100)]))
    assert ok and "correlate" in msg
    bad, _ = selftest._statistical(ref, dict(spikes=1500, blocks=list(range(100))))
    assert not bad
    bad, _ = selftest._statistical(ref, dict(spikes=1000, blocks=list(reversed(range(100)))))
    assert not bad
    assert selftest.CPU_EXACT == ("cpu", "numba", "torch-cpu") and "gl" in selftest.GPU_BACKENDS


def test_a_backend_that_cannot_run_is_reported_by_the_child_process_without_hurting_us():
    r = selftest._in_child("not-a-backend", 5)
    assert "error" in r


@needs_pack
def test_cpu_side_backends_match_spike_for_spike_in_the_selftest():
    from kickthefly.sim.connectome import backends

    ref = selftest.run_backend("cpu", 60)
    assert ref["spikes"] > 0 and len(ref["sha"]) == 64
    for name in ("numba", "torch-cpu"):
        if name in backends.detect_available_backends():
            assert selftest.run_backend(name, 60)["sha"] == ref["sha"], name
    rep = selftest.check_backends(["cpu"], steps=60)
    assert rep[0].id == "backend_cpu" and rep[0].status == PASS


@needs_pack
def test_the_smoke_protocol_runs_and_drives_the_detectors():
    c = selftest.check_smoke(seconds=1.5)
    assert c.status == PASS and c.data["looming_detectors"] > 0


@needs_pack
def test_selftest_from_the_command_line_writes_json_and_exits_with_the_verdict(tmp_path):
    out = tmp_path / "st.json"
    env = dict(os.environ, KICK_THE_FLY_HOME=str(tmp_path / "home"), KICK_THE_FLY_OFFLINE="1", SDL_AUDIODRIVER="dummy")
    p = subprocess.run([sys.executable, str(ROOT / "kick_the_fly.py"), "--selftest", "--sim-backend", "cpu", "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert out.exists(), p.stdout[-2000:] + p.stderr[-2000:]
    rep = json.loads(out.read_text())
    assert rep["format"] == "kick-the-fly-selftest" and rep["exit_code"] == p.returncode
    assert p.returncode in (0, 3), "only warnings are allowed on a healthy source checkout"
    ids = {c["id"] for c in rep["checks"]}
    assert {"build", "brainpack_adult", "backend_cpu", "opengl", "audio", "ffmpeg", "dir_config", "disk", "memory", "display", "smoke"} <= ids
    assert "self-test" in p.stdout and "passed" in p.stdout
