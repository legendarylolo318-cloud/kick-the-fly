"""Game time: pause, slow motion (0.1x, 0.25x, 0.5x, 1x) and single steps, for the room and the brain together.

The game's physics are tuned per frame at 60 Hz, so slow motion does not shrink the physics step. Instead the room
advances in whole 1/60 s ticks: at 0.25x one tick every fourth frame, with the fly's body drawn interpolated between
ticks so it still moves smoothly. Every game timer (stuns, flights, sprays) reads this clock, not the wall clock, and
each brain runs at the same scale on its own thread, so a spike and the reaction it causes stay lined up.

A single step (while paused) advances one tick: 1/60 s of room time and the matching ~3.3 brain steps (5 ms each).
"""
from __future__ import annotations

import time

SPEEDS = (0.1, 0.25, 0.5, 1.0)
TICK = 1.0 / 60.0


class SimClock:
    def __init__(self, start: float | None = None):
        self.now = time.perf_counter() if start is None else start    # virtual seconds (same origin as perf_counter)
        self.scale = 1.0
        self.user_paused = False          # Z / the time controls
        self.menu_paused = False          # the pause menu, save/load dialogs
        self._acc = 0.0                   # fraction of a tick owed to the room
        self._last_real: float | None = None
        self.pending_steps = 0
        self.alpha = 1.0                  # interpolation between the last two room ticks, for drawing
        self.fixed_per_frame = True       # the frame cap is 60: exactly one room tick per frame at 1x
        self.brain_dt = 0.005             # seconds per brain step
        self.brain_debt = 0.0

    @property
    def paused(self) -> bool:
        return self.user_paused or self.menu_paused

    def frame(self, real_now: float | None = None) -> list[float]:
        """Call once per rendered frame. Returns the dt of each room tick to run now; advance `now` by each before
        running that tick. At 1x with a 60 fps cap that is one tick per frame (how the game always ran); otherwise
        ticks are paced at 60 Hz of game time, so the physics speed doesn't depend on the frame rate."""
        real_now = time.perf_counter() if real_now is None else real_now
        real_dt = 0.0 if self._last_real is None else min(0.1, real_now - self._last_real)
        self._last_real = real_now
        self.alpha = 1.0
        if self.menu_paused:
            return []
        if self.user_paused:
            n, self.pending_steps = self.pending_steps, 0
            self.brain_debt += n * TICK / self.brain_dt
            return [TICK] * n
        if self.scale >= 1.0 and self.fixed_per_frame:
            self._acc = 0.0
            return [real_dt]
        self._acc += real_dt * self.scale / TICK
        n = min(int(self._acc), 3)
        self._acc = min(self._acc - n, 1.0)
        self.alpha = self._acc
        return [TICK] * n

    def take_brain_steps(self) -> int:
        """Whole brain steps owed for single-stepped ticks (fed to each Brain.request_steps)."""
        n = int(self.brain_debt)
        self.brain_debt -= n
        return n

    def step(self) -> None:
        if self.user_paused:
            self.pending_steps += 1

    def slower(self) -> float:
        i = SPEEDS.index(self.scale) if self.scale in SPEEDS else len(SPEEDS) - 1
        self.scale = SPEEDS[max(0, i - 1)]
        return self.scale

    def faster(self) -> float:
        i = SPEEDS.index(self.scale) if self.scale in SPEEDS else len(SPEEDS) - 1
        self.scale = SPEEDS[min(len(SPEEDS) - 1, i + 1)]
        return self.scale

    def brain_speed(self) -> float:
        """How fast brain threads should run relative to real time (0 while paused; steps are fed separately)."""
        return 0.0 if self.paused else self.scale

    def label(self) -> str:
        if self.menu_paused:
            return ""
        if self.user_paused:
            return "PAUSED  (Z resume, . step)"
        if self.scale < 1.0:
            return f"SLOW-MO {self.scale:g}x  ([ ] speed)"
        return ""
