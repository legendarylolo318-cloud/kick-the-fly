"""Tests for arbitrary-duration video recording (MP4/WebM with ffmpeg and GIF fallback)."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from kickthefly.core import config, simcore
from kickthefly.game import kick_the_fly as k2
from kickthefly.game.video_recorder import VideoRecorder, ffmpeg_available


def test_cli_record_video_args():
    args = k2.parse_args(["--record-video"])
    assert args.record_video == "default"

    args = k2.parse_args(["--record-video", "my_recording.webm"])
    assert args.record_video == "my_recording.webm"

    args = k2.parse_args([])
    assert args.record_video is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required for MP4 recording test")
def test_video_recorder_mp4(tmp_path: Path):
    rec = VideoRecorder()
    out_file = tmp_path / "test_run.mp4"
    res_path = rec.start(out_file, width=320, height=240, fps=30)
    assert res_path == out_file
    assert rec.is_recording
    assert rec.use_ffmpeg
    assert rec.badge_text().startswith("[REC 00:")

    # Stream 10 black/white alternating frames
    frame_a = b"\x00" * (320 * 240 * 3)
    frame_b = b"\xff" * (320 * 240 * 3)
    for i in range(10):
        rec.write_frame_bytes(frame_a if i % 2 == 0 else frame_b)

    final_path = rec.stop()
    assert final_path == out_file
    assert not rec.is_recording
    assert out_file.exists()
    assert out_file.stat().st_size > 500


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required for WebM recording test")
def test_video_recorder_webm(tmp_path: Path):
    rec = VideoRecorder()
    out_file = tmp_path / "test_run.webm"
    res_path = rec.start(out_file, width=320, height=240, fps=30)
    assert res_path == out_file
    assert rec.is_recording
    assert rec.use_ffmpeg

    frame = b"\x80" * (320 * 240 * 3)
    for _ in range(10):
        rec.write_frame_bytes(frame)

    final_path = rec.stop()
    assert final_path == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 500


def test_video_recorder_gif_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("kickthefly.game.video_recorder.ffmpeg_available", lambda: False)

    rec = VideoRecorder()
    out_file = tmp_path / "test_fallback.mp4"
    res_path = rec.start(out_file, width=160, height=120, fps=15)
    assert res_path.suffix == ".gif"
    assert rec.is_recording
    assert not rec.use_ffmpeg

    frame = b"\x50" * (160 * 120 * 3)
    for _ in range(6):
        rec.write_frame_bytes(frame)

    final_path = rec.stop()
    assert final_path.suffix == ".gif"
    assert final_path.exists()
    assert final_path.stat().st_size > 100


def test_game_video_recording_lifecycle(tmp_path: Path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.Surface((640, 360))
    cfg = config.Config(None)

    brain = simcore.new_brain(seed=0, warmup=10)
    view = SimpleNamespace(
        calm=np.zeros(brain.n, np.float32),
        set_palette=lambda name: None,
        firing=0,
        hot_firing=0,
        legend=((255, 0, 0), (0, 0, 255)),
        sparkle=True,
        render=lambda *a, **kw: pygame.Surface((10, 10)),
        yaw=0.0,
        pitch=0.0,
        pan_x=0.0,
        pan_y=0.0,
        zoom=1.0,
        view_mode="neuron",
    )
    game = k2.Game(screen, brain, view, *simcore.pack()[:2:2], cfg=cfg)
    game.view_stop = True

    assert not game.video_recording

    target = tmp_path / "game_video.mp4"
    game.toggle_video_recording(target)
    assert game.video_recording

    # Capture 5 frames
    for _ in range(5):
        game.capture_video_frame()

    # Toggle off to stop and save
    game.toggle_video_recording()
    assert not game.video_recording

    # Verify saved video
    saved_files = list(tmp_path.glob("game_video.*"))
    assert len(saved_files) == 1
    assert saved_files[0].stat().st_size > 100


def test_game_hotkey_toggle():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.Surface((640, 360))
    cfg = config.Config(None)
    brain = simcore.new_brain(seed=0, warmup=10)
    view = SimpleNamespace(
        calm=np.zeros(brain.n, np.float32),
        set_palette=lambda name: None,
        firing=0,
        hot_firing=0,
        legend=((255, 0, 0), (0, 0, 255)),
        sparkle=True,
        render=lambda *a, **kw: pygame.Surface((10, 10)),
        yaw=0.0,
        pitch=0.0,
        pan_x=0.0,
        pan_y=0.0,
        zoom=1.0,
        view_mode="neuron",
    )
    game = k2.Game(screen, brain, view, *simcore.pack()[:2:2], cfg=cfg)
    game.view_stop = True

    # Press Shift+R to toggle video recording on
    ev_press = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=pygame.KMOD_SHIFT, unicode="R")
    game.handle(ev_press, 0.0)
    assert game.video_recording

    # Shift+R again stops it; plain R is left to "reset the fly" (and respawn in the duel)
    ev_stop = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=pygame.KMOD_SHIFT, unicode="R")
    game.handle(ev_stop, 1.0)
    assert not game.video_recording


def test_recorder_paces_by_wall_clock(tmp_path: Path, monkeypatch):
    """A game drawing at 10 fps still gives a 30 fps video that lasts as long as the recording (frames repeated),
    and one drawing at 120 fps doesn't speed the video up (frames skipped)."""
    monkeypatch.setattr("kickthefly.game.video_recorder.ffmpeg_available", lambda: False)
    rec = VideoRecorder()
    rec.start(tmp_path / "slow.gif", width=64, height=48)
    t0 = rec.start_time
    frame = (b"\x10" * (64 * 48 * 3), (64, 48))
    grabs = []
    for i in range(20):                                      # 2 s at 10 fps
        rec.capture(lambda: grabs.append(1) or frame, now=t0 + i / 10)
    assert len(grabs) == 20
    from kickthefly.game.video_recorder import GIF_FPS
    assert abs(len(rec._gif_frames) - 1.9 * GIF_FPS) <= 2
    rec.stop()

    rec = VideoRecorder()
    rec.start(tmp_path / "fast.gif", width=64, height=48)
    t0 = rec.start_time
    grabs = []
    for i in range(240):                                     # 2 s at 120 fps
        rec.capture(lambda: grabs.append(1) or frame, now=t0 + i / 120)
    assert abs(len(rec._gif_frames) - 2 * GIF_FPS) <= 2
    assert len(grabs) < 240                                  # the readback is skipped when no frame is due
    rec.stop()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_recorder_survives_resize(tmp_path: Path):
    rec = VideoRecorder()
    out = rec.start(tmp_path / "resize.mp4", width=320, height=240)
    rec.write_frame_bytes(b"\x00" * (320 * 240 * 3))
    rec.write_frame_bytes(b"\x80" * (640 * 360 * 3), size=(640, 360))    # the window grew mid-recording
    assert rec.stop() == out and out.stat().st_size > 500


def test_gif_fallback_is_capped(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("kickthefly.game.video_recorder.ffmpeg_available", lambda: False)
    from kickthefly.game import video_recorder as vr
    monkeypatch.setattr(vr, "GIF_MAX_S", 0.5)
    rec = VideoRecorder()
    rec.start(tmp_path / "cap.gif", width=64, height=48)
    for _ in range(40):
        rec.write_frame_bytes(b"\x20" * (64 * 48 * 3))
    assert rec.gif_full and len(rec._gif_frames) == int(0.5 * vr.GIF_FPS)
    assert "GIF FULL" in rec.badge_text()
    assert rec.stop().exists()
