"""The two outdoor arenas, Open field and Orchard: their space, their weather, and the orchard's fruit.

Everything here is plain logic with no OpenGL, so the headless probes and the tests use exactly the code the 3D game
does. kick3d.py draws it and moves bodies around in it.

What comes from the connectome and what is a game rule:

  wind   CONNECTOME: steady ambient wind drives the real wind-sensing antennal neurons JO-C/E (the "wind" group,
         203 on the left antenna, 132 on the right), the same neurons the fan arena drives.
         GAME RULE: the transduction. The share of the drive each antenna gets is the cosine of where the wind comes
         from relative to the fly's heading, and the strength scales with the wind speed. Real Johnston's organs are
         displaced by the air and encode direction through that deflection; that mechanics isn't modelled.
  sun    CONNECTOME: sunlight drives the photoreceptors R1-R8 (the "light" group), as the lamp arena does.
         GAME RULE: how bright (from the sun's elevation) and how the two eyes split it (from its azimuth relative to
         the heading). No image is formed: this is luminance, not vision.
  fruit  CONNECTOME: feeding drives the sugar-pathway taste neurons and the PAM dopamine reward neurons exactly as
         the sugar tool does (and fermented fruit exactly as the alcohol tool does), and heals the fly.
         GAME RULE: the trees, the fruit, how much each fruit holds, regrowth, and the fly flying to a fruit, landing
         and feeding. The fly does not forage through its own circuitry: the game steers it to the fruit.
  motion CONNECTOME: the direction-selective T4/T5 neurons (T4a/T5a front-to-back, T4b/T5b back-to-front) and
         everything after them, down to the steering DNs DNa01/DNa02 (validated: optomotor_turning).
         GAME RULE: emd_motion_drive, the stage that turns a yaw rotation into T4/T5 current. Used by the Lab
         (validation, protocols); the game loop doesn't feed it the fly's view.
  day/night CONNECTOME: the photoreceptors R1-R8 and the morning clock neurons l-LNv/s-LNv that daylight drives, and
         everything downstream of them, including the dorsal fan-shaped body (FB6/FB7) the SLEEP readout watches.
         GAME RULE: the cycle itself (outdoor.day_s, off by default; noon +60 deg, midnight -30 deg), light reaching
         the LNvs directly (real ones see it through the H-B eyelet and CRY), the scene darkening, and the dFB
         threshold that means sleep. In testing a whole day never moved dFB past ~1.5x, so the fly doesn't sleep on
         its own.
  space  GAME RULE: the ground, the sky, the rocks, grass and trees, and where the fly counts as lost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

OUTDOOR = ("field", "orchard")
WIND_FULL = 6.0                     # m/s at which the antennae are driven fully (game rule)


@dataclass(frozen=True)
class WorldSpec:
    """How big an arena is. rx/rz bound you and anything thrown; fly_* bound the fly, which outdoors can leave."""

    name: str
    rx: float
    ry: float
    rz: float
    fly_rx: float
    fly_ry: float
    fly_rz: float
    lost_radius: float | None           # a fly this far from you (horizontally) or this high is out of sight
    colliders: tuple = ()
    ground: tuple = (0.24, 0.2, 0.17)
    sky: tuple = (0.42, 0.44, 0.5)


ROOM = WorldSpec("room", 4.2, 3.0, 3.6, 4.2, 3.0, 3.6, None)
FIELD = WorldSpec("field", 30.0, 60.0, 30.0, 200.0, 80.0, 200.0, 26.0,
                  ground=(0.3, 0.33, 0.22), sky=(0.55, 0.66, 0.8))
ORCHARD = WorldSpec("orchard", 16.0, 40.0, 16.0, 150.0, 60.0, 150.0, 22.0,
                    ground=(0.3, 0.3, 0.2), sky=(0.55, 0.66, 0.8))


def spec(arena: str) -> WorldSpec:
    return {"field": FIELD, "orchard": ORCHARD}.get(arena, ROOM)


# --- weather: wind and sun ----------------------------------------------------------------------------------------
def wind_vector(wind_dir_deg: float, speed: float) -> np.ndarray:
    """The air's velocity in the world's x/z plane. wind_dir_deg is where the wind comes FROM (0 = +x, 90 = +z)."""
    a = math.radians(wind_dir_deg)
    return -np.array([math.cos(a), 0.0, math.sin(a)]) * float(speed)


def wind_drive(yaw: float, wind_dir_deg: float, speed: float) -> tuple[float, float]:
    """(left antenna, right antenna) drive, 0-1, for a fly facing `yaw` in a steady wind. GAME RULE transduction.

    The side the wind comes from gets more: right = total * (1 + sideness) / 2, where sideness is the cosine between
    the wind's source direction and the fly's right side (kick3d.body_axes). A head-on or tail wind drives both
    antennae equally.
    """
    total = float(np.clip(speed / WIND_FULL, 0.0, 1.0))
    a = math.radians(wind_dir_deg)
    source = (math.cos(a), math.sin(a))
    side = (-math.sin(yaw), math.cos(yaw))              # the fly's right, as kick3d.body_axes defines it
    sideness = source[0] * side[0] + source[1] * side[1]
    return total * (1 - sideness) / 2, total * (1 + sideness) / 2


def sun_direction(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector pointing from the ground toward the sun."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])


def sun_light(yaw: float, az_deg: float, el_deg: float) -> tuple[float, float]:
    """(left eye, right eye) photoreceptor drive from the sun. GAME RULE: brightness from elevation (0 below the
    horizon), and up to a 35% split between the eyes from where the sun sits relative to the heading."""
    el = math.radians(el_deg)
    total = float(np.clip(0.15 + 0.85 * math.sin(el), 0.0, 1.0)) if el > 0 else 0.0
    az = math.radians(az_deg)
    side = (-math.sin(yaw), math.cos(yaw))
    sideness = (math.cos(az) * side[0] + math.sin(az) * side[1]) * math.cos(el)
    return total * (1 - 0.35 * sideness), total * (1 + 0.35 * sideness)


# --- optomotor: Reichardt / EMD motion detection ------------------------------------------------------------------
def emd_motion_drive(yaw_rate: float, gain: float = 1.0) -> dict[str, float]:
    """Reichardt/EMD motion detection stage feeding directional visual neurons T4/T5.
    GAME RULE transduction.
    yaw_rate: angular velocity of the visual field in rad/s (positive = visual world rotating right / clockwise).
    Returns activation drive [0, 1] for progressive (T4a/T5a) and regressive (T4b/T5b) visual channels for left and right eyes.
    """
    resp = float(np.tanh(yaw_rate * gain))
    # Rightward visual motion: right eye sees progressive motion (front-to-back), left eye sees regressive motion (back-to-front)
    # Leftward visual motion: left eye sees progressive motion, right eye sees regressive motion
    return {
        "prog_r": max(0.0, resp),
        "regr_l": max(0.0, resp),
        "prog_l": max(0.0, -resp),
        "regr_r": max(0.0, -resp),
    }


# --- circadian cycle and sleep state ------------------------------------------------------------------------------
def diurnal_cycle(time_seconds: float, day_length_s: float = 600.0) -> tuple[float, float]:
    """Computes (azimuth_deg, elevation_deg) along an orbital 24-hr day/night cycle.
    GAME RULE: 360-degree diurnal sun path.
    Elevation > 0 is day (peak +60 deg at noon), < 0 is night (nadir -30 deg at midnight).
    """
    phase = (time_seconds % day_length_s) / day_length_s * 2 * math.pi
    az_deg = (math.degrees(phase) + 180.0) % 360.0
    el_deg = 45.0 * math.sin(phase - math.pi / 2) + 15.0
    return az_deg, el_deg


def circadian_clock_drive(el_deg: float) -> dict[str, float]:
    """Day/night drive to the clock neurons (GAME RULE). Daylight (el_deg > 0) drives the morning cells l-LNv and s-LNv
    directly, full at 60 deg; the game uses only "morning_cells". "evening_cells" is the complement, kept for Lab
    scripts; nothing in the game drives the evening cells (LNd, DN1), since darkness doesn't excite them in flies."""
    day_fraction = float(np.clip(el_deg / 60.0, 0.0, 1.0)) if el_deg > 0 else 0.0
    return {
        "light_level": day_fraction,
        "morning_cells": day_fraction,
        "evening_cells": 1.0 - day_fraction,
    }


def sun_now(params: dict, t: float, day_s: float = 0.0) -> tuple[float, float]:
    """(azimuth, elevation) in degrees: the Lab's fixed sun, or the day/night cycle at game time t (GAME RULE) when
    the Lab's outdoor.day_s, or failing that day_s (Settings > Brain > Day/night cycle), is above 0."""
    day = float(params.get("outdoor.day_s", 0.0)) or float(day_s)
    if day > 0:
        return diurnal_cycle(t, day)
    return float(params.get("outdoor.sun_az", 135.0)), float(params.get("outdoor.sun_el", 45.0))


def dusk(el_deg: float) -> float:
    """How much daylight the scene has, 0 (night, sun 10 deg below the horizon) to 1 (sun 15 deg up or higher)."""
    return float(np.clip((el_deg + 10.0) / 25.0, 0.0, 1.0))


def dfb_sleep_state(dfb_level: float, threshold: float = 2.0) -> bool:
    """Evaluates whether elevated dorsal fan-shaped body (FB6/FB7) activity puts the fly into quiet sleep.
    GAME RULE readout.
    """
    return bool(dfb_level > threshold)


# --- scenery ------------------------------------------------------------------------------------------------------
def scenery(arena: str, seed: int = 7) -> dict:
    """Rocks and grass tufts (both arenas) and trees (orchard), placed from a fixed seed so every run matches."""
    rng = np.random.default_rng(seed)
    w = spec(arena)
    out: dict = dict(rocks=[], tufts=[], trees=[])
    if arena not in OUTDOOR:
        return out
    n_rocks, n_tufts = (26, 900) if arena == "field" else (10, 600)
    for _ in range(n_rocks):
        p = rng.uniform((-w.rx + 2, -w.rz + 2), (w.rx - 2, w.rz - 2))
        if np.hypot(*p) < 3.0:                          # keep the spawn point clear
            continue
        s = float(rng.uniform(0.15, 0.7))
        out["rocks"].append((np.array([p[0], s * 0.35, p[1]]),
                             np.array([s, s * rng.uniform(0.45, 0.8), s * rng.uniform(0.7, 1.2)]),
                             float(rng.uniform(0, math.pi)), float(rng.uniform(0.38, 0.52))))
    for _ in range(n_tufts):
        p = rng.uniform((-w.rx, -w.rz), (w.rx, w.rz))
        h = float(rng.uniform(0.08, 0.22))
        out["tufts"].append((np.array([p[0], h / 2, p[1]]), h, float(rng.uniform(0.75, 1.1))))
    if arena == "orchard":
        for gx in range(-2, 3):
            for gz in range(-2, 3):
                if gx == 0 and gz == 0:
                    continue                            # a clearing where you start
                jitter = rng.uniform(-0.8, 0.8, 2)
                out["trees"].append(dict(pos=np.array([gx * 5.5 + jitter[0], 0.0, gz * 5.5 + jitter[1]]),
                                         height=float(rng.uniform(2.6, 3.4)), crown=float(rng.uniform(1.3, 1.7))))
    return out


def tree_colliders(trees) -> tuple:
    """Trunks as boxes, so you and the walking fly go around them."""
    out = []
    for t in trees:
        x, _, z = t["pos"]
        out.append((np.array([x - 0.18, 0.0, z - 0.18]), np.array([x + 0.18, t["height"], z + 0.18])))
    return tuple(out)


# --- the orchard's fruit ------------------------------------------------------------------------------------------
DEFAULT_FEEDS = 4
DEFAULT_REGROW_S = 75.0
DEFAULT_CAP = 4
SLOTS_PER_TREE = 6
FEED_BOUT_S = 1.5                   # how long one feed lasts; a fruit holds `feeds` of these (game rule)


@dataclass
class Fruit:
    tree: int
    pos: np.ndarray
    feeds_max: int
    feeds_left: int = 0
    fermented: bool = False
    regrow_at: float | None = None      # set while the fruit is gone
    fell_at: float | None = None        # when it dropped, for the falling animation
    eater: object = None                # the fly slot feeding on it right now, if any

    @property
    def ripe(self) -> bool:
        return self.feeds_left > 0

    @property
    def fullness(self) -> float:
        return self.feeds_left / max(1, self.feeds_max)


@dataclass
class Orchard:
    """Every fruit slot on every tree, and the rules for eating it down and growing it back. All GAME RULES.

    feeds     how many feeding bouts one fruit supports before it drops
    regrow_s  mean time for a dropped fruit's slot to grow a new one; each regrowth is jittered +-25% so the orchard
              never refills in one burst. Slots above the cap stay dormant and only start their clock when a fruit
              on the same tree is emptied, so nothing refills sooner than 0.75 x regrow_s.
    cap       most fruit one tree carries at a time
    ferment   share of new fruit that are fermented (fed from, they act exactly like the alcohol tool)
    """

    trees: list
    feeds: int = DEFAULT_FEEDS
    regrow_s: float = DEFAULT_REGROW_S
    cap: int = DEFAULT_CAP
    ferment: float = 0.2
    seed: int = 0
    fruit: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.fruit = []
        for ti, t in enumerate(self.trees):
            top = t["pos"] + np.array([0.0, t["height"] - 0.3, 0.0])
            for k in range(SLOTS_PER_TREE):
                a = 2 * math.pi * k / SLOTS_PER_TREE + self.rng.uniform(-0.3, 0.3)
                r = t["crown"] * self.rng.uniform(0.55, 0.85)
                pos = top + np.array([r * math.cos(a), -self.rng.uniform(0.35, 0.9), r * math.sin(a)])
                self.fruit.append(Fruit(ti, pos, self.feeds))
        for ti in range(len(self.trees)):              # start with each tree at its cap, the rest staggered
            slots = [f for f in self.fruit if f.tree == ti]
            order = self.rng.permutation(len(slots))
            for j, idx in enumerate(order):
                f = slots[idx]
                if j < self.cap:
                    self._grow(f)
                else:
                    f.regrow_at = None                  # dormant: its clock starts when the tree loses a fruit

    def _grow(self, f: Fruit) -> None:
        f.feeds_max = self.feeds
        f.feeds_left = self.feeds
        f.fermented = bool(self.rng.random() < self.ferment)
        f.regrow_at = f.fell_at = None
        f.eater = None

    def on_tree(self, tree: int) -> int:
        return sum(1 for f in self.fruit if f.tree == tree and f.ripe)

    def step(self, now: float) -> list[Fruit]:
        """Regrow what is due, respecting each tree's cap and at most one new fruit per tree per call."""
        grown = []
        busy = set()
        for f in sorted((f for f in self.fruit if not f.ripe and f.regrow_at is not None), key=lambda f: f.regrow_at):
            if f.regrow_at > now or f.tree in busy:
                continue
            if self.on_tree(f.tree) >= self.cap:
                f.regrow_at = None                      # full tree: go dormant until it loses a fruit
                continue
            self._grow(f)
            busy.add(f.tree)
            grown.append(f)
        return grown

    def feed(self, f: Fruit, now: float) -> bool:
        """One feeding bout finished on this fruit. Returns True if that emptied it (it drops and starts regrowing)."""
        if not f.ripe:
            return False
        f.feeds_left -= 1
        if f.feeds_left > 0:
            return False
        f.fell_at = now
        f.eater = None
        f.regrow_at = now + self.regrow_s * self.rng.uniform(0.75, 1.25)
        for other in self.fruit:                        # the tree has room again: its dormant slots start growing
            if other.tree == f.tree and not other.ripe and other.regrow_at is None:
                other.regrow_at = now + self.regrow_s * self.rng.uniform(0.75, 1.25)
        return True

    def nearest_ripe(self, pos, exclude_taken: bool = True) -> Fruit | None:
        best, bd = None, float("inf")
        for f in self.fruit:
            if not f.ripe or (exclude_taken and f.eater is not None):
                continue
            d = float(np.linalg.norm(f.pos - pos))
            if d < bd:
                best, bd = f, d
        return best

    def set_params(self, feeds: int | None = None, regrow_s: float | None = None, cap: int | None = None) -> None:
        """Lab changes apply to fruit that grow from now on; fruit already hanging keep what they had."""
        if feeds is not None:
            self.feeds = max(1, int(feeds))
        if regrow_s is not None:
            self.regrow_s = max(1.0, float(regrow_s))
        if cap is not None:
            self.cap = max(1, min(SLOTS_PER_TREE, int(cap)))

    def counts(self) -> dict:
        ripe = [f for f in self.fruit if f.ripe]
        return dict(ripe=len(ripe), fermented=sum(f.fermented for f in ripe), slots=len(self.fruit),
                    regrowing=sum(1 for f in self.fruit if not f.ripe))

    # save states: regrowth is stored as time remaining, since the clock a save is loaded into is not the same one
    def state(self, now: float) -> dict:
        return dict(feeds=self.feeds, regrow_s=self.regrow_s, cap=self.cap,
                    fruit=[dict(left=f.feeds_left, max=f.feeds_max, fermented=f.fermented,
                                regrow_in=None if f.regrow_at is None else f.regrow_at - now) for f in self.fruit])

    def load_state(self, st: dict, now: float) -> None:
        self.set_params(st.get("feeds"), st.get("regrow_s"), st.get("cap"))
        for f, r in zip(self.fruit, st.get("fruit", [])):
            f.feeds_left, f.feeds_max = int(r["left"]), int(r["max"])
            f.fermented = bool(r["fermented"])
            f.regrow_at = None if r["regrow_in"] is None else now + float(r["regrow_in"])
            f.eater = f.fell_at = None
