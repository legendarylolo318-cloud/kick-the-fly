"""Tests for Screenshot mode: FreeCamera, photo mode controls, clean capture, DOF, and shutter sound."""
import numpy as np
from kickthefly.core import config
from kickthefly.game import kick_the_fly as k2
from kickthefly.game import kick3d


def test_photo_config():
    c = config.Config(None)
    assert c["graphics.photo_scale"] in (1, 2, 4)
    assert c["graphics.clean_capture"] is True
    assert 0.0 <= c["graphics.photo_dof"] <= 1.0
    assert 0.1 <= c["graphics.photo_focus"] <= 15.0
    assert c.action_for("f10") == "photo_mode"
    assert k2.REACTION_SOURCE["PHOTO MODE"] == "rule"
    assert k2.POPUP_SOURCE["PHOTO MODE"] == "rule"


def test_shutter_sound_synthesized():
    snd = k2.Sound()
    if snd.ok:
        assert "shutter" in snd.fx
        assert "click" in snd.fx


def test_free_camera_movement():
    cam = kick3d.FreeCamera(pos=(0.0, 1.5, 0.0), yaw=0.0, pitch=0.0)
    assert np.allclose(cam.eye, [0.0, 1.5, 0.0])
    f, r, u = cam.basis()
    assert np.allclose(f, [1.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(r, [0.0, 0.0, 1.0], atol=1e-3)
    assert np.allclose(u, [0.0, 1.0, 0.0], atol=1e-3)

    # Move forward with 'w'
    cam.update(0.1, {"w": True}, (0, 0))
    assert cam.pos[0] > 0.0

    # Move up with 'up' (Space / E)
    y_before = cam.pos[1]
    cam.update(0.1, {"up": True}, (0, 0))
    assert cam.pos[1] > y_before

    # Room bounds clamping: fly very far
    for _ in range(50):
        cam.update(0.2, {"w": True, "up": True, "sprint": True}, (0, 0))
    assert -kick3d.RX <= cam.pos[0] <= kick3d.RX
    assert 0.05 <= cam.pos[1] <= kick3d.RY
    assert -kick3d.RZ <= cam.pos[2] <= kick3d.RZ
