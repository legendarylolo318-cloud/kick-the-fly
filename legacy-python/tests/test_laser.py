"""Tests for the targeted optogenetics laser module."""
import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.lab.laser import LaserState, BASE_STIM_CURRENT, BASE_SILENCE_CURRENT


def test_laser_state_defaults_and_toggles():
    ls = LaserState()
    assert ls.target_type == "dnp01"
    assert ls.mode == "activate"
    assert ls.intensity == 1.0
    assert ls.trigger_mode == "hold"
    assert ls.current_value() == pytest.approx(BASE_STIM_CURRENT)

    # Toggle mode to silence
    ls.toggle_mode()
    assert ls.mode == "silence"
    assert ls.current_value() == pytest.approx(BASE_SILENCE_CURRENT)

    # Intensity scaling
    ls.set_intensity(2.0)
    assert ls.current_value() == pytest.approx(BASE_SILENCE_CURRENT * 2.0)

    # Intensity clipping
    ls.set_intensity(5.0)
    assert ls.intensity == 3.0
    ls.set_intensity(0.01)
    assert ls.intensity == 0.1


def test_laser_trigger_hold_vs_pulse():
    ls = LaserState(trigger_mode="hold")
    assert not ls.is_active(now=1.0)
    ls.trigger_press(now=1.0)
    assert ls.is_active(now=1.0)
    assert ls.is_active(now=2.0)
    ls.trigger_release()
    assert not ls.is_active(now=2.1)

    # Pulse mode
    ls.set_trigger_mode("pulse")
    ls.pulse_duration = 0.20
    assert not ls.is_active(now=3.0)
    ls.trigger_press(now=3.0)
    assert ls.is_active(now=3.10)
    assert ls.is_active(now=3.19)
    assert not ls.is_active(now=3.25)


@needs_pack
def test_laser_resolve_target_and_apply():
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=100)
    ls = LaserState(target_type="dnp01", mode="activate", intensity=1.5)

    rows = ls.resolve_target_rows(br)
    assert len(rows) > 0, "dnp01 should resolve to at least 1 neuron"

    # Test applying when not firing or not hitting
    cur = ls.apply(br, now=1.0, is_hitting=False)
    assert cur == 0.0
    assert np.all(br.override[rows] == 0.0)

    # Press trigger and hit fly
    ls.trigger_press(now=1.0)
    cur = ls.apply(br, now=1.0, is_hitting=True)
    expected_cur = BASE_STIM_CURRENT * 1.5
    assert cur == pytest.approx(expected_cur)
    assert np.all(br.override[rows] == pytest.approx(expected_cur))
    assert br.surgery is True

    # Move beam off fly: override cleared
    cur = ls.apply(br, now=1.05, is_hitting=False)
    assert cur == 0.0
    assert np.all(br.override[rows] == 0.0)
    assert br.surgery is False

    # Switch to silence and hit fly
    ls.set_mode("silence")
    ls.set_intensity(1.0)
    cur = ls.apply(br, now=1.1, is_hitting=True)
    assert cur == pytest.approx(BASE_SILENCE_CURRENT)
    assert np.all(br.override[rows] == pytest.approx(BASE_SILENCE_CURRENT))

    # Clear completely
    ls.clear(br)
    assert np.all(br.override[rows] == 0.0)
    assert not ls.is_active(now=1.1)


@needs_pack
def test_laser_multi_fly():
    from kickthefly.core import simcore

    br1 = simcore.new_brain(seed=101)
    br2 = simcore.new_brain(seed=102)
    ls = LaserState(target_type="dnp01", mode="activate", intensity=1.0)
    rows1 = ls.resolve_target_rows(br1)
    rows2 = ls.resolve_target_rows(br2)

    ls.trigger_press(now=1.0)
    # Fly 1 is hit, fly 2 is not
    ls.apply(br1, now=1.0, is_hitting=True)
    ls.apply(br2, now=1.0, is_hitting=False)
    assert np.all(br1.override[rows1] == pytest.approx(BASE_STIM_CURRENT))
    assert np.all(br2.override[rows2] == 0.0)

    # Beam moves to Fly 2, fly 1 is not hit
    ls.apply(br1, now=1.1, is_hitting=False)
    ls.apply(br2, now=1.1, is_hitting=True)
    assert np.all(br1.override[rows1] == 0.0)
    assert np.all(br2.override[rows2] == pytest.approx(BASE_STIM_CURRENT))

    # Beam moves off all flies
    ls.apply(br1, now=1.2, is_hitting=False)
    ls.apply(br2, now=1.2, is_hitting=False)
    assert np.all(br1.override[rows1] == 0.0)
    assert np.all(br2.override[rows2] == 0.0)


@needs_pack
def test_laser_3d_integration():
    from kickthefly.game import kick3d, kick_the_fly as k2
    from kickthefly.core import config
    import pygame
    pygame.init()

    state = {"seed": 3}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")
    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game = kick3d.Game3D(hud, state["brain"], state["view"], state["graph"], state["weights"], cfg=config.Config(None))

    laser_idx = k2.TOOL_NAMES.index("laser")
    game.tool = laser_idx
    assert hasattr(game, "laser_state")
    assert not game.laser_state.is_active(now=1.0)

    # Use tool 3D
    game.use_tool3d(now=1.0)
    assert game.torching is True
    assert game.laser_state.firing is True
    assert game.laser_state.is_active(now=1.0)

    # Update 3D
    keys = dict(w=0, s=0, a=0, d=0, sprint=0, crouch=0, up=0, down=0)
    game.update3d(1.0, 1 / 60, keys, (0, 0))

    # Mouse button up release
    ev = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(100, 100))
    game.handle3d(ev, 1.05, lambda p: p)
    assert game.torching is False
    assert game.laser_state.firing is False

    # Update 3D after release clears overrides
    game.update3d(1.1, 1 / 60, keys, (0, 0))
    for slot in game.flies:
        assert len(getattr(slot.brain, "_laser_rows", [])) == 0


