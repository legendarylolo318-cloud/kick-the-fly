"""Tests for Time-lapse export (speed-up recording 2x-20x to GIF/MP4)."""
import os
import shutil
import time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pygame
from kickthefly.core import config
from kickthefly.game import kick_the_fly as k2
from kickthefly.core import simcore


def test_timelapse_config():
    c = config.Config(None)
    assert c["graphics.timelapse_speedup"] in (2, 5, 10, 20)
    assert c["graphics.timelapse_format"] in ("mp4", "gif")
    assert c["graphics.timelapse_target"] in ("brain", "room")
    assert c.action_for("l") == "timelapse"
    # Check that timelapse_speedup is tagged as GAME_RULE
    s = config.BY_KEY["graphics.timelapse_speedup"]
    assert s is not None
    assert s.tag == config.GAME_RULE


def test_timelapse_capture_and_save(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.Surface((960, 540))
    cfg = config.Config(None)
    cfg.set("graphics.timelapse_speedup", 5)
    cfg.set("graphics.timelapse_format", "gif")
    cfg.set("graphics.timelapse_target", "room")

    brain = simcore.new_brain(seed=0, warmup=10)
    view = SimpleNamespace(
        calm=np.zeros(brain.n, np.float32), set_palette=lambda name: None, firing=0, hot_firing=0,
        legend=((255, 0, 0), (0, 0, 255)), sparkle=True,
        render=lambda *a, **kw: pygame.Surface((10, 10)),
        yaw=0.0, pitch=0.0, pan_x=0.0, pan_y=0.0, zoom=1.0, view_mode="neuron"
    )
    game = k2.Game(screen, brain, view, *simcore.pack()[:2:2], cfg=cfg)
    game.view_stop = True
    game._save_dir = lambda: tmp_path

    assert not game.timelapse_recording
    game.toggle_timelapse()
    assert game.timelapse_recording
    assert len(game.timelapse_frames) == 0

    # Capture 6 frames
    for _ in range(6):
        game.capture_timelapse_frame()
    assert len(game.timelapse_frames) == 6
    w, h = game.timelapse_size
    assert len(game.timelapse_frames[0]) == w * h * 3

    # Stop recording - this should invoke save_timelapse
    game.toggle_timelapse()
    assert not game.timelapse_recording

    # Wait briefly for worker thread to finish export
    for _ in range(30):
        time.sleep(0.1)
        gifs = list(tmp_path.glob("*.gif"))
        if gifs and gifs[0].stat().st_size > 0:
            break
    gifs = list(tmp_path.glob("*.gif"))
    assert len(gifs) == 1
    assert gifs[0].stat().st_size > 100


def test_timelapse_save_mp4(tmp_path):
    if not shutil.which("ffmpeg"):
        return
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.Surface((960, 540))
    cfg = config.Config(None)
    cfg.set("graphics.timelapse_speedup", 10)
    cfg.set("graphics.timelapse_format", "mp4")
    cfg.set("graphics.timelapse_target", "room")

    brain = simcore.new_brain(seed=0, warmup=10)
    view = SimpleNamespace(
        calm=np.zeros(brain.n, np.float32), set_palette=lambda name: None, firing=0, hot_firing=0,
        legend=((255, 0, 0), (0, 0, 255)), sparkle=True,
        render=lambda *a, **kw: pygame.Surface((10, 10)),
        yaw=0.0, pitch=0.0, pan_x=0.0, pan_y=0.0, zoom=1.0, view_mode="neuron"
    )
    game = k2.Game(screen, brain, view, *simcore.pack()[:2:2], cfg=cfg)
    game.view_stop = True
    game._save_dir = lambda: tmp_path

    game.toggle_timelapse()
    for _ in range(8):
        game.capture_timelapse_frame()
    game.toggle_timelapse()

    for _ in range(40):
        time.sleep(0.1)
        mp4s = list(tmp_path.glob("*.mp4"))
        if mp4s and mp4s[0].stat().st_size > 0:
            break
    mp4s = list(tmp_path.glob("*.mp4"))
    assert len(mp4s) == 1
    assert mp4s[0].stat().st_size > 100
