"""Gamepad support for the 3D game: SDL joysticks through pygame, on top of (never instead of) keyboard and mouse.

Left stick walks, right stick looks, the right trigger uses the tool in your hand, the bumpers cycle tools and holding
the tool-wheel button opens a wheel you point at with the right stick. Start opens the menu. Every binding is
rebindable in Settings > Controls (click, then press the button or push the stick), because the raw axis and button
numbers differ between controllers and drivers; the defaults are an Xbox-style pad on SDL. Stick input is merged into
the same movement and look the keyboard and mouse produce, so nothing about them changes.

Bindings are strings: "axisN" (a stick axis, signed), "buttonN", or "" (unbound). An axis bound to a button-like
action counts as pressed past half travel (triggers rest at -1 and go to +1, so that works for them too).
"""
from __future__ import annotations

import math

import pygame

from kickthefly.core.config import PAD_ACTIONS

PAD_LABEL = {a: label for a, label, _ in PAD_ACTIONS}
STICKS = ("move_x", "move_y", "look_x", "look_y")
BUTTONS = tuple(a for a, _, _ in PAD_ACTIONS if a not in STICKS)


def parse(binding: str) -> tuple[str, int] | None:
    for kind in ("axis", "button"):
        if binding.startswith(kind) and binding[len(kind):].isdigit():
            return kind, int(binding[len(kind):])
    return None


class Gamepad:
    """Every connected pad, read each frame. Hot-plugging works (JOYDEVICEADDED / JOYDEVICEREMOVED)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pads: dict[int, "pygame.joystick.JoystickType"] = {}
        self.was: dict[str, bool] = {a: False for a in BUTTONS}
        self.wheel_open = False
        self.wheel_pick: int | None = None
        try:
            pygame.joystick.init()
            for i in range(pygame.joystick.get_count()):
                self._add(i)
        except pygame.error:
            pass

    def _add(self, index: int) -> None:
        try:
            j = pygame.joystick.Joystick(index)
            if is_gamepad(index, j):
                self.pads[j.get_instance_id()] = j
        except pygame.error:
            pass

    @property
    def connected(self) -> bool:
        return bool(self.pads) and bool(self.cfg["controls.gamepad"])

    def name(self) -> str:
        return next((j.get_name() for j in self.pads.values()), "")

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
        kind, n = b
        best = 0.0
        for j in self.pads.values():
            try:
                if kind == "axis" and n < j.get_numaxes():
                    v = j.get_axis(n)
                elif kind == "button" and n < j.get_numbuttons():
                    v = float(j.get_button(n))
                else:
                    continue
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
        from pygame._sdl2 import controller

        controller.init()
        if controller.is_controller(index):
            return True
    except Exception:
        pass
    return j.get_numaxes() >= 4 and j.get_numbuttons() >= 6


def capture_binding(ev) -> str | None:
    """A binding from a raw joystick event, for Settings > Controls: a button press, or a stick/trigger pushed far."""
    if ev.type == pygame.JOYBUTTONDOWN:
        return f"button{ev.button}"
    if ev.type == pygame.JOYAXISMOTION and abs(ev.value) > 0.7:
        return f"axis{ev.axis}"
    return None
