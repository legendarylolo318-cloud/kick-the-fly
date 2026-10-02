"""The kitchen arena (3.0 day 3): a countertop with a fruit bowl, a vinegar trap, a sink, a stove burner and a cook.
Plain logic: no pygame, no OpenGL, no brain. kick3d.py draws it and calls it; the Lab tests and the playthrough bot use the
same code.

The room's floor is the countertop. Everything below is placed on it, in the 3D room's metres (the room is 8.4 m x 7.2 m, the
fly is half a metre long), in the same coordinates as outdoors.py.

What is what, per part:
  fruit bowl    CONNECTOME: feeding drives the sugar-pathway taste neurons and the PAM reward neurons exactly as sugar and the
                orchard's fruit do (the same Orchard code, with one "tree" that is the bowl); fermented fruit as alcohol does,
                its scent on DM1/DM2/DP1m. GAME RULE: the bowl, how much each fruit holds, regrowth, the fly flying to the
                nearest ripe fruit and landing. The Lab's orchard.* parameters apply to the bowl too.
  vinegar trap  CONNECTOME: the scent is poked onto the fermentation glomeruli DM1, DM2 and DP1m (the "alcohol" scent group,
                the same neurons the alcohol tool and fermented fruit drive), and whether the fly is attracted is read from
                those neurons' own firing against their calm rate: the game does not send it anywhere on a timer.
                GAME RULE: the jar, how far the smell reaches, how strongly it falls off with distance, how much firing counts
                as "smelling it", the trip the fly then makes to the mouth, and the trap physics: a fly that hovers over the
                mouth falls in and is stuck, drowning slowly, and can't get out (an immortal fly escapes after a while).
  sink          CONNECTOME: the water drives the humidity neurons (HRN) and wet-body touch, as the pool does. GAME RULE:
                the basin (a rectangle of the counter with water 0.3 m deep), the floating, wet wings and drowning, which
                are the pool arena's rules limited to the basin.
  stove burner  CONNECTOME: heat drives the heat sensors, as the lamp does. GAME RULE: where the burner is, how far its heat
                reaches, and that touching it hurts and pushes the fly off (the lamp's rule).
  cook          CONNECTOME: the swat is something that grows in the fly's view, so it goes through the existing looming
                transduction (LPLC2/LC4 -> DNp01), and a hit fires the touch neurons. GAME RULE: that there is a cook, when
                he swats (every 10-20 s), where (at where a fly was when he started to swing), how fast, and how hard it
                hits. Nothing tells the fly a swat is coming: a fly that sees it grow in time dodges, one that does not
                is hit.
Nothing here is measured.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RX, RY, RZ = 4.2, 3.0, 3.6

# --- GAME RULES: where things are and how big (the counter is y = 0) ----------------------------------------------------------
BOWL_POS = np.array([-2.4, 0.0, 1.6])
BOWL_R, BOWL_H = 0.75, 0.30
TRAP_POS = np.array([2.3, 0.0, -1.9])
TRAP_R, TRAP_H, TRAP_MOUTH = 0.42, 0.80, 0.30
SINK = (2.0, 3.9, 1.4, 3.3)                    # x0, x1, z0, z1 of the basin
SINK_WATER = 0.30                              # the water's surface above the counter (the pool's is 0.6)
BURNER_POS = np.array([-0.5, 0.0, -2.5])
BURNER_R, BURNER_HEAT_RANGE = 0.5, 1.1
COOK_Z = -RZ + 0.55

# the vinegar trap's smell and what the fly does about it (GAME RULES)
SCENT_RANGE = 3.4                              # m from the jar's mouth within which the smell reaches the fly
SCENT_POKE = 0.35                              # strength of the poke at the jar; falls off linearly to 0 at SCENT_RANGE
ATTRACT_RATIO, ATTRACT_MIN_HZ = 2.5, 6.0       # DM1/DM2/DP1m firing (Hz per neuron) that counts as smelling it: x its calm rate
SCENT_WARM_S = 2.0                             # a fly needs this long of calm to know its own baseline before it can follow a smell
HOVER_TO_FALL_S = 0.4                          # hovering over the mouth this long and it falls in
TRAP_DROWN = 0.03                              # health lost a frame inside the trap
TRAP_IMMORTAL_ESCAPE_S = 8.0

# the cook (GAME RULES)
COOK_EVERY_S = (10.0, 20.0)
COOK_FIRST_S = (6.0, 10.0)
COOK_WINDUP_S, COOK_SWAT_S, COOK_RECOVER_S = 0.7, 0.9, 1.6   # the first version swung in 0.28 s with an ease-in: the fly saw it only in its last 3 frames
COOK_REACH = 0.55                              # a fly within this far (horizontally) of the impact point, and low, is hit
COOK_HIT_HEIGHT = 0.95
COOK_SWATTER_R = 0.5                           # looming radius of the swatter and the forearm behind it (the first version used 0.34)
COOK_HAND_HEIGHT = 2.0
COOK_DAMAGE = 38.0
COOK_SPEED = 0.6                               # m/s he paces along the counter


def colliders() -> list:
    """Boxes you and the walking fly go around: the jar (its top is the mouth)."""
    x, _, z = TRAP_POS
    return [(np.array([x - TRAP_R, 0.0, z - TRAP_R]), np.array([x + TRAP_R, TRAP_H, z + TRAP_R])),
            (np.array([-RX, 0.0, -3.35]), np.array([-RX + 0.8, 2.1, -1.95]))]                   # the fridge


# --- the fruit bowl ------------------------------------------------------------------------------------------------------------
def bowl_orchard(seed: int = 0, feeds: int | None = None, regrow_s: float | None = None, cap: int | None = None):
    """The orchard's fruit code with one "tree": the bowl. Its six slots are fruit sitting in the bowl."""
    from kickthefly.game import outdoors

    tree = dict(pos=BOWL_POS.copy(), height=BOWL_H + 0.3, crown=0.3)
    o = outdoors.Orchard([tree], feeds=outdoors.DEFAULT_FEEDS if feeds is None else int(feeds),
                         regrow_s=outdoors.DEFAULT_REGROW_S if regrow_s is None else float(regrow_s),
                         cap=min(outdoors.SLOTS_PER_TREE, outdoors.DEFAULT_CAP if cap is None else int(cap)), seed=seed)
    rng = np.random.default_rng(seed + 11)
    for k, f in enumerate(o.fruit):
        a = 2 * math.pi * k / len(o.fruit) + rng.uniform(-0.25, 0.25)
        r = BOWL_R * rng.uniform(0.25, 0.6)
        f.pos = BOWL_POS + np.array([r * math.cos(a), 0.2 + 0.05 * (k % 2), r * math.sin(a)])
    return o


# --- the vinegar trap ------------------------------------------------------------------------------------------------------------
def mouth() -> np.ndarray:
    return TRAP_POS + np.array([0.0, TRAP_H + 0.05, 0.0])


def scent_strength(head) -> float:
    """The poke strength of the vinegar's smell at a point: SCENT_POKE at the jar, 0 at SCENT_RANGE (horizontal distance
    from the mouth, plus a little for height). GAME RULE."""
    d = float(np.linalg.norm(np.asarray(head, float) - mouth()))
    return SCENT_POKE * max(0.0, 1.0 - d / SCENT_RANGE)


def over_mouth(thorax) -> bool:
    t = np.asarray(thorax, float)
    return bool(math.hypot(t[0] - TRAP_POS[0], t[2] - TRAP_POS[2]) <= TRAP_MOUTH and TRAP_H - 0.05 <= t[1] <= TRAP_H + 0.55)


@dataclass
class ScentTracker:
    """Reads one fly's DM1/DM2/DP1m firing and decides whether it is smelling the vinegar: the rate must be at least
    ATTRACT_RATIO x its own calm rate and ATTRACT_MIN_HZ. The calm rate is a slow average taken only while no vinegar smell
    has reached the fly for 3 s (so the smell can't raise its own baseline), and the fly needs SCENT_WARM_S of that first."""

    base: float | None = None
    rate: float = 0.0
    since_poke: float = 99.0
    calm_time: float = 0.0
    k: float = 0.02

    def update(self, dt: float, hz: float, poked: bool) -> None:
        self.rate = hz
        self.since_poke = 0.0 if poked else self.since_poke + dt
        if self.since_poke > 3.0:
            self.calm_time += dt
            self.base = hz if self.base is None else self.base + (hz - self.base) * self.k

    @property
    def smelling(self) -> bool:
        return (self.base is not None and self.calm_time >= SCENT_WARM_S
                and self.rate >= max(ATTRACT_MIN_HZ, ATTRACT_RATIO * self.base))


def group_rate_hz(spike_idx: np.ndarray, rows_mask: np.ndarray, n_rows: int, window_s: float) -> float:
    """Firing of a neuron set in Hz per neuron from the spike indices of the last window (neurodex.window_spikes)."""
    if n_rows <= 0 or window_s <= 0:
        return 0.0
    return float(rows_mask[spike_idx].sum()) / n_rows / window_s


# --- the sink --------------------------------------------------------------------------------------------------------------------
def in_sink(p) -> bool:
    p = np.asarray(p, float)
    return bool(SINK[0] <= p[0] <= SINK[1] and SINK[2] <= p[2] <= SINK[3])


def sink_submerged(points: np.ndarray) -> np.ndarray:
    """Which body points are under the basin's water."""
    pts = np.asarray(points, float)
    return (pts[:, 0] >= SINK[0]) & (pts[:, 0] <= SINK[1]) & (pts[:, 2] >= SINK[2]) & (pts[:, 2] <= SINK[3]) & (pts[:, 1] < SINK_WATER)


# --- the burner -------------------------------------------------------------------------------------------------------------------
def burner_heat(head) -> tuple[float, bool]:
    """(heat drive 0-1 at a point, touching the burner). GAME RULE: linear falloff to BURNER_HEAT_RANGE; touching is being
    over the ring and within 0.2 m of the counter."""
    h = np.asarray(head, float)
    horiz = math.hypot(h[0] - BURNER_POS[0], h[2] - BURNER_POS[2])
    dist = float(np.linalg.norm(h - (BURNER_POS + (0.0, 0.05, 0.0))))
    heat = float(np.clip(1.0 - dist / BURNER_HEAT_RANGE, 0.0, 1.0))
    return heat, bool(horiz <= BURNER_R and h[1] <= 0.2)


# --- the cook ---------------------------------------------------------------------------------------------------------------------
@dataclass
class CookEvent:
    kind: str                       # "windup" | "swat" | "impact"
    at: np.ndarray | None = None


@dataclass
class Cook:
    """Paces along the back of the counter and every 10-20 s swings a swatter at where a fly was. GAME RULE."""

    rng: np.random.Generator
    x: float = 0.0
    state: str = "idle"
    t: float = 0.0
    wait: float = 0.0
    goal_x: float = 0.0
    aim: np.ndarray | None = None
    hand0: np.ndarray | None = None
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    swats: int = 0

    def __post_init__(self):
        self.wait = float(self.rng.uniform(*COOK_FIRST_S))
        self.goal_x = float(self.rng.uniform(-2.5, 2.5))
        self.pos = np.array([self.x, COOK_HAND_HEIGHT * 0.6, COOK_Z + 0.4])

    def step(self, dt: float, targets: list) -> list:
        """targets: thorax positions (np arrays) of living flies. Returns what happened this step."""
        ev: list = []
        self.t += dt
        if self.state == "idle":
            if abs(self.goal_x - self.x) < 0.05:
                self.goal_x = float(self.rng.uniform(-2.5, 2.5))
            self.x += math.copysign(min(COOK_SPEED * dt, abs(self.goal_x - self.x)), self.goal_x - self.x)
            self.pos = np.array([self.x, COOK_HAND_HEIGHT * 0.6, COOK_Z + 0.4])
            if self.t >= self.wait and targets:
                near = min(targets, key=lambda p: abs(float(p[0]) - self.x) + abs(float(p[2]) - COOK_Z))
                self.aim = np.array([float(near[0]), 0.12, float(near[2])])      # aimed once, at where the fly is now
                self.hand0 = np.array([self.x, COOK_HAND_HEIGHT, COOK_Z + 0.5])
                self.state, self.t = "windup", 0.0
                ev.append(CookEvent("windup", self.aim.copy()))
        elif self.state == "windup":
            f = min(1.0, self.t / COOK_WINDUP_S)
            self.pos = np.array([self.x, COOK_HAND_HEIGHT * (0.6 + 0.4 * f), COOK_Z + 0.4 + 0.1 * f])
            if self.t >= COOK_WINDUP_S:
                self.state, self.t = "swat", 0.0
                ev.append(CookEvent("swat", self.aim.copy()))
        elif self.state == "swat":
            e = min(1.0, self.t / COOK_SWAT_S)
            ease = e                                                            # a steady swing, so its growth is visible early
            self.pos = self.hand0 * (1 - ease) + self.aim * ease + np.array([0.0, 0.25 * math.sin(math.pi * e), 0.0])
            if self.t >= COOK_SWAT_S:
                self.state, self.t = "recover", 0.0
                self.swats += 1
                ev.append(CookEvent("impact", self.aim.copy()))
        elif self.state == "recover":
            self.pos = self.aim + np.array([0.0, 0.1 + 0.9 * min(1.0, self.t / COOK_RECOVER_S), 0.0])
            if self.t >= COOK_RECOVER_S:
                self.x = float(np.clip(self.aim[0], -3.0, 3.0))
                self.state, self.t = "idle", 0.0
                self.wait = float(self.rng.uniform(*COOK_EVERY_S))
        return ev

    def threats(self) -> list:
        """What the fly's eyes can see grow: the swatter, from the start of the swing to the impact."""
        if self.state in ("swat",):
            return [(("cook", "swatter"), self.pos.copy(), COOK_SWATTER_R)]
        return []

    def hit(self, fly_thorax) -> bool:
        """Is a fly at this point hit by the swat that just landed?"""
        if self.aim is None:
            return False
        t = np.asarray(fly_thorax, float)
        return bool(math.hypot(t[0] - self.aim[0], t[2] - self.aim[2]) <= COOK_REACH and t[1] <= COOK_HIT_HEIGHT)


@dataclass
class KitchenState:
    """What the running kitchen remembers: the cook, each fly's vinegar tracker, who is hovering over the mouth and who is in."""

    cook: Cook
    trackers: dict = field(default_factory=dict)
    hover: dict = field(default_factory=dict)             # fly key -> seconds hovering over the mouth
    trapped: dict = field(default_factory=dict)           # fly key -> seconds inside the trap
    jitter: dict = field(default_factory=dict)

    def tracker(self, key) -> ScentTracker:
        return self.trackers.setdefault(key, ScentTracker())

    def release(self, key) -> None:
        self.trapped.pop(key, None)
        self.hover.pop(key, None)

    def inside_point(self, key, t: float) -> np.ndarray:
        """Where a trapped fly is held: down in the jar, bobbing a little (it is struggling)."""
        j = self.jitter.setdefault(key, np.random.default_rng(abs(hash(key)) % (1 << 31)).uniform(-0.12, 0.12, 2))
        return TRAP_POS + np.array([j[0], 0.3 + 0.04 * math.sin(t * 6.0), j[1]])
