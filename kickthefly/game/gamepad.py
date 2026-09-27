"""Gamepad support for the 3D game: SDL game controllers through pygame, on top of (never instead of) keyboard and mouse.

Left stick walks, right stick looks, the right trigger uses the tool in your hand, the bumpers cycle tools and holding
the tool-wheel button opens a wheel you point at with the right stick. Start opens the menu. Stick input is merged into
the same movement and look the keyboard and mouse produce, so nothing about them changes.

Controllers SDL knows (Xbox, PlayStation, Switch Pro and the rest of its controller database) are read through SDL's
GameController API, which names every input the same way whatever the pad: "rightx", "righttrigger", "rightshoulder",
"start"... (button names follow the labels printed on the pad, so "y" is the button marked Y). Raw joystick numbers
differ between pads and drivers (a Switch Pro's right stick is raw axes 2/3, an Xbox pad's 3/4), which is why the
defaults use these names. Bindings are rebindable in Settings > Controls (click, then press the button or push the
stick). A pad SDL doesn't know is read as a raw joystick: the names then fall back to the usual Xbox numbering, and a
binding can also be a raw "axisN" or "buttonN".
"""
from __future__ import annotations

import math
import time

import pygame

from kickthefly.core.config import PAD_ACTIONS, PAD_AXES, PAD_BUTTONS

PAD_LABEL = {a: label for a, label, _ in PAD_ACTIONS}
STICKS = ("move_x", "move_y", "look_x", "look_y")
BUTTONS = tuple(a for a, _, _ in PAD_ACTIONS if a not in STICKS)
# a raw joystick SDL has no mapping for: the names on the usual Xbox (XInput) numbering
RAW_FALLBACK = {"leftx": ("axis", 0), "lefty": ("axis", 1), "lefttrigger": ("axis", 2), "rightx": ("axis", 3),
                "righty": ("axis", 4), "righttrigger": ("axis", 5), "a": ("button", 0), "b": ("button", 1),
                "x": ("button", 2), "y": ("button", 3), "leftshoulder": ("button", 4), "rightshoulder": ("button", 5),
                "back": ("button", 6), "start": ("button", 7), "guide": ("button", 8), "leftstick": ("button", 9),
                "rightstick": ("button", 10)}

try:
    from pygame._sdl2 import controller as _sdl_controller
except Exception:                                           # pragma: no cover - an SDL/pygame without the API
    _sdl_controller = None


def parse(binding: str) -> tuple[str, object] | None:
    """("std", name) for a standard controller name, ("axis", n) / ("button", n) for a raw one."""
    if binding in PAD_AXES or binding in PAD_BUTTONS:
        return "std", binding
    for kind in ("axis", "button"):
        if binding.startswith(kind) and binding[len(kind):].isdigit():
            return kind, int(binding[len(kind):])
    return None


def pretty(binding: str) -> str:
    """How a binding reads in Settings."""
    names = {"leftx": "left stick X", "lefty": "left stick Y", "rightx": "right stick X", "righty": "right stick Y",
             "lefttrigger": "left trigger", "righttrigger": "right trigger", "leftshoulder": "left bumper",
             "rightshoulder": "right bumper", "leftstick": "left stick click", "rightstick": "right stick click",
             "back": "back / select / -", "start": "start / +", "guide": "home / guide"}
    if not binding:
        return "unbound"
    return names.get(binding, binding.upper() if binding in PAD_BUTTONS else binding)


SETTLE_S = 0.3        # a Switch Pro reported both sticks tilted for ~0.2 s after connecting: ignore that stale state


class Pad:
    """One connected device: the raw joystick, and SDL's controller view of it when SDL knows the pad."""

    def __init__(self, joystick, ctrl=None, settle_s: float = 0.0):
        self.joystick, self.ctrl = joystick, ctrl
        self.ready_at = time.monotonic() + settle_s

    def read(self, b: tuple[str, object]) -> float:
        if time.monotonic() < self.ready_at:
            return 0.0
        kind, n = b
        j, c = self.joystick, self.ctrl
        if kind == "std":
            if c is not None:
                if n in PAD_AXES:
                    return max(-1.0, c.get_axis(PAD_AXES.index(n)) / 32767.0)   # sticks -1..1, triggers 0..1
                return float(c.get_button(PAD_BUTTONS.index(n)))
            kind, n = RAW_FALLBACK.get(n, ("", -1))
        if kind == "axis" and 0 <= n < j.get_numaxes():
            return j.get_axis(n)
        if kind == "button" and 0 <= n < j.get_numbuttons():
            return float(j.get_button(n))
        return 0.0


class Gamepad:
    """Every connected pad, read each frame. Hot-plugging works (JOYDEVICEADDED / JOYDEVICEREMOVED)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pads: dict[int, Pad] = {}
        self.was: dict[str, bool] = {a: False for a in BUTTONS}
        self.wheel_open = False
        self.wheel_pick: int | None = None
        try:
            pygame.joystick.init()
            if _sdl_controller is not None:
                _sdl_controller.init()
            for i in range(pygame.joystick.get_count()):
                self._add(i)
        except pygame.error:
            pass

    def _add(self, index: int) -> None:
        try:
            j = pygame.joystick.Joystick(index)
            if not is_gamepad(index, j):
                return
            ctrl = None
            if _sdl_controller is not None and _sdl_controller.is_controller(index):
                ctrl = _sdl_controller.Controller(index)
            self.pads[j.get_instance_id()] = Pad(j, ctrl, SETTLE_S)
        except pygame.error:
            pass

    @property
    def connected(self) -> bool:
        return bool(self.pads) and bool(self.cfg["controls.gamepad"])

    def name(self) -> str:
        return next((p.joystick.get_name() for p in self.pads.values()), "")

    def is_controller(self, instance_id: int) -> bool:
        p = self.pads.get(instance_id)
        return p is not None and p.ctrl is not None

    def device_event(self, ev) -> bool:
        if ev.type == pygame.JOYDEVICEADDED:
            self._add(ev.device_index)
            return True
        if ev.type == pygame.JOYDEVICEREMOVED:
            self.pads.pop(ev.instance_id, None)
            return True
        return False

    # --- reading ----------------------------------------------------------------------------------------------------
    def _value(self, action: str) -> float:
        b = parse(self.cfg.pad.get(action, ""))
        if b is None:
            return 0.0
        best = 0.0
        for p in self.pads.values():
            try:
                v = p.read(b)
            except pygame.error:
                continue
            if abs(v) > abs(best):
                best = v
        return best

    def stick(self, ax: str, ay: str) -> tuple[float, float]:
        """A stick with the dead zone taken out, each axis -1..1."""
        x, y = self._value(ax), self._value(ay)
        dz = float(self.cfg["controls.pad_deadzone"])
        m = math.hypot(x, y)
        if m < dz:
            return 0.0, 0.0
        k = min(1.0, (m - dz) / (1 - dz)) / m
        return x * k, y * k

    def pressed(self, action: str) -> bool:
        return self._value(action) > 0.5

    def poll(self) -> tuple[dict, dict, set]:
        """(movement keys to OR into the keyboard's, look (x, y) in -1..1, button actions that went down)."""
        if not self.connected:
            self.was = {a: False for a in BUTTONS}
            return {}, (0.0, 0.0), set()
        mx, my = self.stick("move_x", "move_y")
        keys = dict(w=my < -0.35, s=my > 0.35, a=mx < -0.35, d=mx > 0.35, sprint=self.pressed("sprint"),
                    crouch=self.pressed("crouch"), up=self.pressed("up"), down_fly=self.pressed("crouch"))
        down = set()
        for a in BUTTONS:
            now = self.pressed(a)
            if now and not self.was[a]:
                down.add(a)
            self.was[a] = now
        return keys, self.stick("look_x", "look_y"), down

    def released(self, action: str) -> bool:
        """True once the button bound to `action` is up (after poll())."""
        return not self.was.get(action, False)

    def wheel(self, n_tools: int) -> int | None:
        """While the wheel is open: the tool the right stick points at (0 at the top, clockwise), or None."""
        x, y = self.stick("look_x", "look_y")
        if math.hypot(x, y) < 0.5:
            return self.wheel_pick
        ang = math.atan2(x, -y) % (2 * math.pi)
        self.wheel_pick = int(round(ang / (2 * math.pi) * n_tools)) % n_tools
        return self.wheel_pick


def is_gamepad(index: int, j) -> bool:
    """Only real gamepads: SDL also lists things like a keyboard's "System Control" media keys as joysticks (1 axis,
    19 buttons), which would otherwise count as a pad. SDL's controller database decides when it knows the device;
    otherwise it needs two sticks' worth of axes and six buttons."""
    try:
        if _sdl_controller is not None:
            _sdl_controller.init()
            if _sdl_controller.is_controller(index):
                return True
    except Exception:
        pass
    return j.get_numaxes() >= 4 and j.get_numbuttons() >= 6


def capture_binding(ev, pad: "Gamepad | None" = None) -> str | None:
    """A binding from an event, for Settings > Controls: a controller's named button or axis (Xbox, PlayStation, Switch
    Pro... all give the same names), or, for a pad SDL doesn't know, the raw joystick button or axis."""
    if ev.type == pygame.CONTROLLERBUTTONDOWN and 0 <= ev.button < len(PAD_BUTTONS):
        return PAD_BUTTONS[ev.button]
    if ev.type == pygame.CONTROLLERAXISMOTION and 0 <= ev.axis < len(PAD_AXES) and abs(ev.value) > 0.7 * 32767:
        return PAD_AXES[ev.axis]
    if ev.type in (pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION):
        if pad is not None and pad.is_controller(getattr(ev, "instance_id", -1)):
            return None                                     # the same press arrives as a CONTROLLER event
        if ev.type == pygame.JOYBUTTONDOWN:
            return f"button{ev.button}"
        if abs(ev.value) > 0.7:
            return f"axis{ev.axis}"
    return None
