"""Rain, gusts and storms for the open field and the orchard (3.0 day 3). Plain logic: no pygame, no OpenGL, no brain.

The 3D game, the Python API and the tests all use this one module. It decides what the weather does each frame; the game
turns that into pokes on the fly's real sensory neurons exactly as it does for tools, wind and sunlight.

What is what:
  CONNECTOME   what the weather drives, all through groups the game already drives:
                 raindrop hits   the fly's touch neurons by body part (head BM_*/JO-*, body SNta*, legs SNpp*, wings WG*),
                                 as a tool hit does (Brain.poke)
                 wet air         the humidity receptor neurons (HRN), as the pool arena's water does
                 gusts           the Johnston's organ wind neurons JO-C/E, through the existing wind -> JO transduction
                                 (outdoors.wind_drive): a gust is only a change in the wind speed that goes into it
                 lightning       the photoreceptors R1-R8 (the "light" group), as sunlight does
  GAME RULE    everything that is not a neuron: how often a drop hits and how hard, which part it hits (the table below),
               when a fly's wings count as wet (it fills up with rain and dries off), where gusts and flashes come from, how
               long they last and how strong they are, the thunder delay, the storm's preset (what "storm on" means), and
               how the scene darkens and flashes. None of it is measured. Real raindrops are not modelled: a "hit" is a
               touch pulse of the strength given here.
  MODEL        nothing is predicted from this module.

Defaults keep every existing outdoor result: rain 0, gusts 0, storm off. The Lab parameters are weather.rain (0-1),
weather.gust_hz (gusts a second, 0-0.5) and weather.storm (0 or 1).

    from kickthefly.game import weather
    w = weather.Weather(seed=1)
    w.update(dt, rain=0.5, gust_hz=0.1, storm=True)
    w.wind(base_speed=3.0, base_dir=180.0)      # (speed, direction) with the gusts added
    w.hits(dt, exposed=True)                    # [(region, side, strength)] raindrops that hit this frame
"""
from __future__ import annotations

import math

import numpy as np

# --- GAME RULES ----------------------------------------------------------------------------------------------------
HITS_PER_S = 6.0                 # raindrop hits a second at rain 1.0 on a fly in the open
# Which part a drop hits: the share of the fly's area seen from above (wings are the biggest part, legs the smallest).
PART_WEIGHTS = (("wing", 0.35), ("body", 0.40), ("head", 0.15), ("legs", 0.10))
HIT_STRENGTH = (0.15, 0.35)      # touch pulse strength of one drop, scaled by the rain
HUMID_POKE = 0.2                 # humidity neurons: strength at rain 1.0, every HUMID_EVERY_S
HUMID_EVERY_S = 0.5
WET_FILL_S = 20.0                # seconds of full rain until the wings are wet (they can't fly while wet, as in the pool)
WET_DRY_S = 30.0                 # seconds the filling takes to dry off out of the rain
WET_FOR_S = 3.0                  # fly.wet while soaked: the pool's value (kick3d: fly.wet = 3.0)
GUST_LEN_S = (1.2, 2.6)          # a gust's duration
GUST_AMP = (2.0, 6.0)            # a gust's extra wind speed, m/s
GUST_TURN_DEG = 25.0             # how far a gust's direction wanders from the steady wind's
STORM_RAIN, STORM_GUST_HZ, STORM_EXTRA_WIND = 0.7, 0.2, 3.0   # what "storm on" means: at least this much of each
FLASH_EVERY_S = (5.0, 14.0)      # lightning, only in a storm
FLASH_PATTERN = ((0.00, 1.00, 0.07), (0.12, 0.60, 0.05), (0.25, 0.85, 0.10))      # (start s, brightness, length s)
THUNDER_DELAY_S = (1.0, 3.0)
REDUCED_FLASH_RISE_S = 0.6       # reduced flashing: the screen brightens over this long instead of flashing
REDUCED_FLASH_PEAK = 0.25        # and no brighter than this
DARKEN_STORM = 0.45              # how much a storm darkens the scene (the sky and the sun)
WIND_FULL_RAIN = 1.0


def effective(rain: float, gust_hz: float, storm: bool) -> tuple[float, float]:
    """(rain, gust rate) after the storm preset: a storm is at least STORM_RAIN of rain and STORM_GUST_HZ of gusts."""
    rain = float(np.clip(rain, 0.0, 1.0))
    gust_hz = float(np.clip(gust_hz, 0.0, 0.5))
    if storm:
        rain, gust_hz = max(rain, STORM_RAIN), max(gust_hz, STORM_GUST_HZ)
    return rain, gust_hz


class Weather:
    """The weather's state: gusts under way, the next lightning, thunder on its way. Deterministic in its seed."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.gusts: list[tuple[float, float, float, float]] = []      # (start, length, amplitude, turn in degrees)
        self.next_flash = float(self.rng.uniform(*FLASH_EVERY_S))
        self.flash_at: float | None = None                            # when the current lightning started
        self.thunder: list[float] = []                                # times the thunder arrives
        self.rain = 0.0
        self.storm = False
        self._humid = 0.0

    # --- time ------------------------------------------------------------------------------------------------------
    def update(self, dt: float, rain: float = 0.0, gust_hz: float = 0.0, storm: bool = False) -> None:
        self.rain, gust_hz = effective(rain, gust_hz, storm)
        self.storm = bool(storm)
        self.t += dt
        if gust_hz > 0 and self.rng.random() < gust_hz * dt:
            self.gusts.append((self.t, float(self.rng.uniform(*GUST_LEN_S)), float(self.rng.uniform(*GUST_AMP)),
                               float(self.rng.uniform(-GUST_TURN_DEG, GUST_TURN_DEG))))
        self.gusts = [g for g in self.gusts if self.t < g[0] + g[1]]
        if storm and self.t >= self.next_flash:
            self.flash_at = self.t
            self.next_flash = self.t + float(self.rng.uniform(*FLASH_EVERY_S))
            self.thunder.append(self.t + float(self.rng.uniform(*THUNDER_DELAY_S)))
        elif not storm:
            self.next_flash = max(self.next_flash, self.t + 1.0)
        # the flash stays "on" until both the lightning and the reduced-flashing swell are over (the swell outlasts the flash)
        if self.flash_at is not None and self.t - self.flash_at > max(FLASH_PATTERN[-1][0] + FLASH_PATTERN[-1][2],
                                                                       2 * REDUCED_FLASH_RISE_S):
            self.flash_at = None

    # --- what it does --------------------------------------------------------------------------------------------------
    def wind(self, base_speed: float, base_dir: float) -> tuple[float, float]:
        """The wind with the gusts in it: (speed m/s, comes-from direction in degrees). A gust is a raised-cosine bump."""
        extra, turn = 0.0, 0.0
        for start, length, amp, tdeg in self.gusts:
            f = math.sin(math.pi * min(max((self.t - start) / length, 0.0), 1.0)) ** 2
            extra += amp * f
            turn += tdeg * f
        bonus = STORM_EXTRA_WIND if self.storm else 0.0
        return max(0.0, float(base_speed) + extra + bonus), (float(base_dir) + turn) % 360.0

    def hits(self, dt: float, exposed: bool = True) -> list[tuple[str, str | None, float]]:
        """The raindrops that hit a fly this frame, as (region, side, strength) for Brain.poke. At most 3 a frame."""
        if not exposed or self.rain <= 0:
            return []
        out = []
        for _ in range(3):
            if self.rng.random() < HITS_PER_S * self.rain * dt / 3:
                x = self.rng.random()
                acc, part = 0.0, PART_WEIGHTS[-1][0]
                for name, w in PART_WEIGHTS:
                    acc += w
                    if x < acc:
                        part = name
                        break
                side = None if part in ("body", "head") else ("L" if self.rng.random() < 0.5 else "R")
                s = float(self.rng.uniform(*HIT_STRENGTH)) * (0.5 + 0.5 * self.rain)
                out.append((part, side, s))
        return out

    def humid(self, dt: float) -> float:
        """Strength for a poke of the humidity neurons, or 0 when it is not due (every HUMID_EVERY_S while raining)."""
        self._humid += dt
        if self.rain > 0 and self._humid >= HUMID_EVERY_S:
            self._humid = 0.0
            return HUMID_POKE * self.rain
        return 0.0

    def lightning(self) -> float:
        """The lightning's brightness now, 0 to 1 (what the photoreceptors are driven by)."""
        if self.flash_at is None:
            return 0.0
        e = self.t - self.flash_at
        for start, b, length in FLASH_PATTERN:
            if start <= e < start + length:
                return b
        return 0.0

    def screen_flash(self, reduced: bool = False) -> float:
        """How much the scene brightens on screen. Normally the lightning itself. With reduced flashing it is a single
        slow swell no brighter than REDUCED_FLASH_PEAK, so nothing strobes."""
        if not reduced:
            return self.lightning()
        if self.flash_at is None:
            return 0.0
        e = self.t - self.flash_at
        span = REDUCED_FLASH_RISE_S
        if e < span:
            return REDUCED_FLASH_PEAK * (e / span)
        return max(0.0, REDUCED_FLASH_PEAK * (1.0 - (e - span) / span))

    def thunder_due(self) -> int:
        """How many thunderclaps have arrived since the last call."""
        due = [x for x in self.thunder if x <= self.t]
        self.thunder = [x for x in self.thunder if x > self.t]
        return len(due)

    def darkness(self) -> float:
        """0 to DARKEN_STORM: how much the scene darkens (with the rain; a storm is at least STORM_RAIN of rain)."""
        return DARKEN_STORM * float(np.clip(self.rain / max(STORM_RAIN, 1e-6), 0.0, 1.0)) * (1.0 if self.storm else 0.6)


def soak(level: float, dt: float, rain: float, wing_hits: int = 0) -> float:
    """The wing-wetness of one fly, 0 to 1: it fills with rain and dries out of it. At 1 the wings count as wet (the game
    then sets fly.wet = WET_FOR_S and the fly can't take off, as after the pool). GAME RULE."""
    if rain > 0:
        level += dt * rain / WET_FILL_S * (1.0 + 0.5 * wing_hits)
    else:
        level -= dt / WET_DRY_S
    return float(np.clip(level, 0.0, 1.0))
