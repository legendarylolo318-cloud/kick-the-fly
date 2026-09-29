"""Gamepad support (kickthefly/game/gamepad.py): bindings, reading a pad, and what it does in the 3D game."""
import os

import numpy as np
import pygame
import pytest

from conftest import needs_pack
from kickthefly.core import config


class FakePad:
    """Stands in for pygame.joystick.Joystick: axes and buttons set by the test."""

    def __init__(self, axes=6, buttons=11):
        self.axes, self.buttons = [0.0] * axes, [0] * buttons
        if axes > 5:
            self.axes[5] = -1.0                               # a trigger at rest

    def get_numaxes(self): return len(self.axes)
    def get_numbuttons(self): return len(self.buttons)
    def get_axis(self, i): return self.axes[i]
    def get_button(self, i): return self.buttons[i]
    def get_name(self): return "Fake Pad"


class FakeController:
    """Stands in for pygame._sdl2.controller.Controller: SDL's named inputs, ints like the real API."""

    def __init__(self):
        self.axes, self.buttons = [0] * 6, [0] * 15

    def get_axis(self, i): return self.axes[i]
    def get_button(self, i): return self.buttons[i]


def _pad(cfg=None):
    """A raw joystick SDL has no mapping for: names fall back to the Xbox numbering."""
    from kickthefly.game import gamepad

    pad = gamepad.Gamepad(cfg or config.Config(None))
    fake = FakePad()
    pad.pads = {0: gamepad.Pad(fake)}
    return pad, fake


def test_bindings_roundtrip_swap_and_reset(tmp_path):
    cfg = config.Config(tmp_path / "config.toml")
    assert cfg.pad["move_x"] == "leftx" and cfg.pad["use"] == "righttrigger"
    ok, msg = cfg.bind_pad("tool_next", "leftshoulder")     # taken by tool_prev: swapped
    assert ok and cfg.pad["tool_next"] == "leftshoulder" and cfg.pad["tool_prev"] == "rightshoulder" and "swapped" in msg
    assert cfg.bind_pad("up", "button12")[0] and cfg.pad["up"] == "button12"        # raw, for an unknown pad
    assert not cfg.bind_pad("use", "wiggle")[0]
    assert cfg.bind_pad("sprint", "")[0] and cfg.pad["sprint"] == ""
    cfg.save()
    again = config.Config.load(tmp_path / "config.toml")
    assert again.pad == cfg.pad and again.keys == cfg.keys   # the keyboard bindings are untouched
    again.reset_tab("Controls")
    assert again.pad == {a: b for a, _, b in config.PAD_ACTIONS}


def test_reading_sticks_buttons_and_the_wheel():
    from kickthefly.game import gamepad

    pad, fake = _pad()
    keys, look, down = pad.poll()
    assert not any(keys.values()) and look == (0.0, 0.0) and not down           # at rest, trigger at -1
    fake.axes[1], fake.axes[3] = -0.9, 0.1                                      # forward; look within the dead zone
    fake.buttons[5] = 1
    keys, look, down = pad.poll()
    assert keys["w"] and not keys["s"] and look == (0.0, 0.0) and down == {"tool_next"}
    assert pad.poll()[2] == set()                                               # held, not pressed again
    fake.axes[5] = 1.0
    assert "use" in pad.poll()[2]                                               # the trigger past half travel
    fake.axes[3], fake.axes[4] = 1.0, 0.0                                       # right stick right: 3 o'clock
    assert pad.wheel(12) == 3
    assert gamepad.capture_binding(pygame.event.Event(pygame.JOYBUTTONDOWN, button=7, joy=0, instance_id=0)) == "button7"
    assert gamepad.capture_binding(pygame.event.Event(pygame.JOYAXISMOTION, axis=2, value=0.3, joy=0, instance_id=0)) is None
    assert gamepad.capture_binding(pygame.event.Event(pygame.CONTROLLERBUTTONDOWN, button=10, instance_id=0)) == "rightshoulder"
    assert gamepad.capture_binding(pygame.event.Event(pygame.CONTROLLERAXISMOTION, axis=5, value=32000, instance_id=0)) == "righttrigger"
    cfg = config.Config(None)
    cfg.set("controls.gamepad", False)
    off, fake2 = _pad(cfg)
    fake2.axes[1] = -1.0
    assert off.poll() == ({}, (0.0, 0.0), set())                                # switched off in Settings


@needs_pack
def test_pad_in_the_3d_game(monkeypatch):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode((64, 64))
    from kickthefly.game import kick3d
    from kickthefly.game import kick_the_fly as k2

    state = {"seed": 4}
    k2.load_brain(state)
    game = kick3d.Game3D(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), state["brain"], state["view"],
                         state["graph"], state["weights"], cfg=config.Config(None))
    fake = FakePad()
    game.pad.pads = {0: kick3d.gamepad.Pad(fake)}
    game.tool = 0
    fake.buttons[5] = 1                                       # RB: next tool of the loadout (2.13: not of every tool)
    game.pad_tick(1 / 60, game.clock.now)
    assert game.tool_name() == game.loadout.tools[1]
    fake.buttons[5] = 0
    fake.axes[5] = 1.0                                        # trigger while not looking: starts looking
    game.pad_tick(1 / 60, game.clock.now)
    assert game.look
    fake.axes[5] = -1.0
    fake.axes[3] = 1.0                                        # right stick: looks right, through update_player
    yaw = game.player.yaw
    keys, rel = game.pad_tick(1 / 60, game.clock.now)
    game.update_player(1 / 60, dict(game.held(pygame.key.get_pressed()), **keys), rel)
    assert game.player.yaw > yaw
    expected = game.cfg["controls.pad_look_speed"] / 60                # radians in one frame at full tilt
    assert game.player.yaw - yaw == pytest.approx(expected, rel=0.05)
    fake.axes[3] = 0.0
    fake.buttons[3] = 1                                       # hold Y: the wheel; point down (6 o'clock of 12)
    fake.axes[4] = 1.0
    game.pad_tick(1 / 60, game.clock.now)
    assert game.pad.wheel_open
    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game._draw_tool_wheel(hud)
    fake.buttons[3] = 0
    fake.axes[4] = 0.0
    game.pad_tick(1 / 60, game.clock.now)
    wheel_tools = game._wheel_tools()                         # every tool the mode allows, not only the hotbar's
    assert not game.pad.wheel_open and game.tool_name() == wheel_tools[len(wheel_tools) // 2]
    fake.buttons[7] = 1                                       # Start: the pause menu
    game.pad_tick(1 / 60, game.clock.now)
    assert game.menu.open


@needs_pack
def test_pad_x_opens_the_loadout_editor_and_the_default_binding_is_free():
    """2.13: a pad-only player can reach the loadout editor (the tutorial's last step) without the Q key."""
    assert config.Config(None).pad["loadout"] == "x" and len({b for b in config.Config(None).pad.values() if b}) == \
        len([b for b in config.Config(None).pad.values() if b]), "no two pad actions share a button"
    pygame.init()
    from kickthefly.game import kick3d
    from kickthefly.game import kick_the_fly as k2

    state = {"seed": 4}
    k2.load_brain(state)
    game = kick3d.Game3D(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), state["brain"], state["view"],
                         state["graph"], state["weights"], cfg=config.Config(None))
    fake = FakePad()
    game.pad.pads = {0: kick3d.gamepad.Pad(fake)}
    fake.buttons[2] = 1                                       # X
    game.pad_tick(1 / 60, game.clock.now)
    assert game.menu.open and game.menu.screen == "loadout"
    game.view_stop = True


def test_keyboards_listed_as_joysticks_are_not_pads():
    """SDL lists a keyboard's media keys ("... System Control": 1 axis, 19 buttons) as a joystick."""
    from kickthefly.game import gamepad

    kb = FakePad(axes=1, buttons=19)
    assert not gamepad.is_gamepad(99, kb)
    assert gamepad.is_gamepad(99, FakePad())



def test_known_controllers_use_sdl_names_whatever_their_raw_numbers():
    """A Switch Pro's right stick is raw axes 2/3 and its R is raw button 10; through SDL's controller API it is just
    "rightx"/"righty" and "rightshoulder", so the defaults work on it (and on Xbox and PlayStation pads) unchanged."""
    from kickthefly.game import gamepad

    pad = gamepad.Gamepad(config.Config(None))
    raw = FakePad()                                         # raw axis 3 = what an Xbox pad calls right X; ignored here
    ctrl = FakeController()
    pad.pads = {0: gamepad.Pad(raw, ctrl)}
    raw.axes[3] = 1.0
    ctrl.axes[2] = 32767                                    # SDL: right stick X, full right
    ctrl.axes[5] = 30000                                    # SDL: right trigger (ZR), 0..32767
    ctrl.buttons[10] = 1                                    # SDL: right shoulder (R)
    keys, look, down = pad.poll()
    assert look[0] == pytest.approx(1.0) and look[1] == 0.0
    assert {"use", "tool_next"} <= down
    assert gamepad.pretty("start") == "start / +" and gamepad.pretty("") == "unbound"


def test_a_just_connected_pad_is_ignored_while_it_settles(monkeypatch):
    """Right after connecting, a Switch Pro reported both sticks tilted for ~0.2 s: that must not move or turn you."""
    from kickthefly.game import gamepad

    clock = [100.0]
    monkeypatch.setattr(gamepad.time, "monotonic", lambda: clock[0])
    pad = gamepad.Gamepad(config.Config(None))
    fake = FakePad()
    fake.axes[1], fake.axes[3] = -1.0, 1.0
    pad.pads = {0: gamepad.Pad(fake, None, gamepad.SETTLE_S)}
    assert pad.poll()[:2] == (dict(w=False, s=False, a=False, d=False, sprint=False, crouch=False, up=False,
                                   down_fly=False), (0.0, 0.0))
    clock[0] += gamepad.SETTLE_S + 0.01
    keys, look, _ = pad.poll()
    assert keys["w"] and look[0] > 0.9
