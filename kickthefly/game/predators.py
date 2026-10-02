"""Predators (3.0 day 3): the frog, the dragonfly and the mantis. Plain logic, no pygame, no OpenGL, no brain.

The 3D game, the 2D game, the Lab assay and the tests all use this one engine, so a predator behaves the same
everywhere. A predator is a small state machine that moves a body and, when it strikes, a striking part (the tongue,
the claw). It reports where those parts are each frame; the game turns them into looming exactly as it does for every
other object.

What comes from the connectome and what is a game rule:

  detection  CONNECTOME + the game's existing transduction. The fly "sees" a predator only as an object that grows in
             its view, through the same code path as the swatter, the spider and you (angular expansion speed above
             LOOM_MIN drives the real looming detectors LPLC2/LC4, which drive the giant fiber DNp01; DNp01 above
             THRESH["escape"] x calm makes the fly dodge). Nothing here tells the fly a predator is near. The
             transduction threshold is itself a game rule, so "sneak up slowly and it won't notice" is a consequence
             of that rule, not a measurement.
  capture    CONNECTOME for what it drives: a capture fires the fly's real touch neurons by body part (head, body,
             legs, wings), as tool hits do. GAME RULE: that a capture happens, and how hard it is.
  the AI     GAME RULE, all of it: when a predator waits, hops, stalks, pursues or strikes, how fast, how far it
             reaches, how long a strike takes, how many strikes it makes, and when it gives up. These are numbers chosen
             for play. Real frogs, dragonflies and mantises are not measured here, and no published value is used
             or implied. The only thing the three have in common with their real namesakes is the shape of the idea:
             a frog strikes with a very fast tongue, a dragonfly chases flying prey from above, a mantis creeps up and
             then strikes.

Units are metres and seconds in the 3D game's scale, where the fly itself is large (its thorax sits 0.38 m up and its
body is about half a metre long), so the predators are scaled up to be bigger than it. The 2D game converts at 0.006 m a pixel. Time advances in fixed 1/60 s steps so the same seed gives the same predator. The vertical axis is y.

    from kickthefly.game import predators as pr
    p = pr.Predator("frog", position, np.random.default_rng(7), pr.Bounds(-4, 4, 0, 3, -3.6, 3.6))
    events = p.step(pr.DT, [pr.Target(1, fly_xyz, flying=False)])
    p.threats()                      # [(key, position, radius)] for the game's looming measure
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DT = 1.0 / 60.0
THORAX_Y = 0.38              # how high the game's standing fly's thorax is (kick3d.STAND3), m
KINDS = ("frog", "dragonfly", "mantis")


@dataclass(frozen=True)
class Spec:
    """One predator's numbers. Every field is a GAME RULE (see the module docstring)."""

    kind: str
    label: str
    body_r: float          # looming radius of the body, m
    tip_r: float           # looming radius of the striking part (0: it has none), m
    reach: float           # how far the tongue / claw reaches from the body, m
    trigger: float         # how close a target must be (horizontally for the frog and mantis) before it aims, m
    aim_s: float           # a pause before the strike, during which the predator is still
    strike_s: float        # how long the strike takes to reach its aim point, s
    retract_s: float
    recover_s: float       # after a miss, how long before it can strike again
    max_strikes: int
    capture_r: float       # the striking part grabs a fly this close to it, m
    speed: float           # m/s: hopping (frog), stalking (mantis), pursuit (dragonfly)
    note: str


SPECS = {
    "frog": Spec("frog", "FROG", 0.45, 0.12, 1.8, 1.5, 0.30, 0.07, 0.12, 1.8, 3, 0.28, 1.4,
                 "sits still, hops closer, then a tongue that takes 0.07 s to reach where the fly was when it aimed"),
    "dragonfly": Spec("dragonfly", "DRAGONFLY", 0.20, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1, 0.30, 4.0,
                      "patrols high up and chases a flying fly from above; a walking fly is not prey"),
    "mantis": Spec("mantis", "MANTIS", 0.35, 0.09, 0.9, 0.8, 0.20, 0.06, 0.15, 2.0, 2, 0.22, 0.10,
                   "creeps up at 10 cm/s, freezing now and then, then strikes with a forelimb in 0.06 s"),
}
TELL_FREEZE_S = (0.4, 0.9)       # the mantis's stalk: it freezes for this long every STALK_RUN_S of walking
STALK_RUN_S = 2.0
HOP_S, HOP_LEN, HOP_EVERY_S = 0.35, 1.0, 1.5          # the frog's hop: how long, how far, how often
DRAGONFLY_GIVE_UP_S = 30.0
DRAGONFLY_GROUND_PATIENCE_S = 1.0                      # a target that lands this long stops being prey
DRAGONFLY_TURN = 5.0                                   # rad/s
LEAVE_S = 3.0                                          # how long a predator takes to get out of the arena after it gives up
CARRY_S = 1.2                                          # how long a captured fly is held before it is eaten

# What a capture fires: (region, side, strength) on the fly's real touch groups. GAME RULE: which parts, and how hard.
CAPTURE_TOUCH = {
    "frog": (("body", None, 1.0), ("head", None, 0.6), ("legs", "L", 0.5), ("legs", "R", 0.5)),
    "dragonfly": (("legs", "L", 0.9), ("legs", "R", 0.9), ("wing", None, 0.9), ("body", None, 0.8)),
    "mantis": (("body", None, 1.0), ("legs", "L", 0.9), ("legs", "R", 0.9)),
}


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float            # the floor
    ymax: float            # the ceiling (or how high the predator may go)
    zmin: float
    zmax: float

    def clip(self, p: np.ndarray, margin: float = 0.3) -> np.ndarray:
        lo = np.array([self.xmin + margin, self.ymin, self.zmin + margin])
        hi = np.array([self.xmax - margin, self.ymax - margin, self.zmax - margin])
        return np.minimum(np.maximum(p, lo), hi)


@dataclass
class Target:
    """What a predator needs to know about a fly: where its thorax is, whether it is in the air, whether it is alive."""

    id: object
    p: np.ndarray
    flying: bool = False
    alive: bool = True


@dataclass
class Event:
    kind: str                       # "aim" | "strike" | "capture" | "miss" | "leave" | "gone"
    target: object = None
    touch: tuple = ()
    at: np.ndarray | None = None


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


class Predator:
    """One predator. Call step() every 1/60 s with the fly targets; read threats() for the looming measure."""

    def __init__(self, kind: str, position, rng: np.random.Generator, bounds: Bounds):
        if kind not in SPECS:
            raise ValueError(f"unknown predator {kind!r}; use one of {', '.join(KINDS)}")
        self.kind, self.spec, self.rng, self.bounds = kind, SPECS[kind], rng, bounds
        self.p = np.array(position, float)
        self.state = "patrol" if kind == "dragonfly" else "lurk"
        self.t = 0.0                    # time in this state
        self.age = 0.0
        self.strikes = 0
        self.tip: np.ndarray | None = None
        self._tip_full = np.zeros(3)
        self.aim_at: np.ndarray | None = None
        self.mouth = self.p.copy()
        self.target = None
        self.carry = None               # (target id, time left) while a captured fly is held
        self.heading = _unit(np.array([rng.uniform(-1, 1), 0.0, rng.uniform(-1, 1)]) + 1e-6)
        self.face = self.heading.copy()   # where it looks: toward the nearest fly (the games draw it that way)
        self.anchor = self.p.copy()
        self.vel = np.zeros(3)
        self._since_hop = 0.0
        self._stalk_t = 0.0
        self._freeze_left = 0.0
        self._ground_t = 0.0
        self._prev: dict = {}
        self.done = False

    # --- what the game reads -----------------------------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return not self.done

    def threats(self) -> list:
        """(key, position, radius) of what the fly could see grow: the body, and the striking part while it strikes."""
        if self.done:
            return []
        out = [((self.kind, "body"), self.p + (0.0, self.spec.body_r * 0.6, 0.0) if self.kind != "dragonfly" else self.p.copy(),
                self.spec.body_r)]
        if self.tip is not None and self.spec.tip_r > 0 and self.state in ("strike", "retract", "eat"):
            out.append(((self.kind, "tip"), self.tip.copy(), self.spec.tip_r))
        return out

    def grab_point(self) -> np.ndarray:
        """Where a captured fly is held."""
        return (self.tip if self.tip is not None else self.p).copy()

    # --- the AI --------------------------------------------------------------------------------------------------------
    def step(self, dt: float, targets: list) -> list:
        ev: list = []
        if self.done:
            return ev
        self.age += dt
        self.t += dt
        live = [t for t in targets if t.alive]
        for t in live:
            self._prev[t.id] = np.asarray(t.p, float).copy()
        if live:
            near, _ = self._nearest(live)
            _, h = self._horizontal(near)
            if float(np.linalg.norm(h)) > 1e-6:
                self.face = _unit(h)
        getattr(self, "_" + self.kind)(dt, live, ev)
        self.p = self.bounds.clip(self.p, margin=0.0) if self.state != "leave" else self.p
        return ev

    def _set(self, state: str) -> None:
        self.state, self.t = state, 0.0

    def _nearest(self, live, flying_only: bool = False):
        best, bd = None, 1e9
        for t in live:
            if flying_only and not t.flying:
                continue
            d = float(np.linalg.norm(np.asarray(t.p, float) - self.p))
            if d < bd:
                best, bd = t, d
        return best, bd

    def _horizontal(self, t) -> tuple[float, np.ndarray]:
        d = np.asarray(t.p, float) - self.p
        h = np.array([d[0], 0.0, d[2]])
        return float(np.linalg.norm(h)), h

    # frog and mantis share the strike machinery
    def _begin_aim(self, t, ev) -> None:
        self.target = t.id
        self._set("aim")
        ev.append(Event("aim", t.id))

    def _strike_phase(self, dt: float, live, ev) -> bool:
        """aim -> strike -> retract -> recover, for the two predators that strike with a part. True while handled."""
        sp = self.spec
        if self.state == "aim":
            tgt = next((t for t in live if t.id == self.target), None)
            if self.t >= sp.aim_s:
                if tgt is None:
                    self._set("lurk" if self.kind == "frog" else "stalk")
                    return True
                self.aim_at = np.asarray(tgt.p, float).copy()          # it aims once, then commits: no tracking
                self.mouth = self.p + (0.0, sp.body_r * 0.7, 0.0) + self.face * sp.body_r * 0.8
                self.tip = self.mouth.copy()
                self._set("strike")
                ev.append(Event("strike", self.target, at=self.aim_at.copy()))
            return True
        if self.state == "strike":
            f = min(1.0, self.t / sp.strike_s)
            aim_vec = self.aim_at - self.mouth
            reach_vec = _unit(aim_vec) * min(sp.reach, float(np.linalg.norm(aim_vec)) + 0.05)
            self.tip = self.mouth + reach_vec * f
            self._tip_full = self.tip.copy()
            hit = self._capture_check(live, ev)
            if hit:
                return True
            if f >= 1.0:
                self._set("retract")
            return True
        if self.state == "retract":
            f = max(0.0, 1.0 - self.t / sp.retract_s)
            self.tip = self.mouth + (self._tip_full - self.mouth) * f
            if self.t >= sp.retract_s:
                self.tip = None
                self.strikes += 1
                ev.append(Event("miss", self.target))
                if self.strikes >= sp.max_strikes:
                    self._set("leave")
                    ev.append(Event("leave"))
                else:
                    self._set("recover")
            return True
        if self.state == "recover":
            if self.t >= sp.recover_s:
                self._set("lurk" if self.kind == "frog" else "stalk")
            return True
        if self.state == "eat":
            self.tip = self.mouth + (self.tip - self.mouth) * max(0.0, 1.0 - dt / CARRY_S)
            if self.t >= CARRY_S:
                self.tip = None
                self._set("leave")
                ev.append(Event("leave"))
            return True
        return False

    def _capture_check(self, live, ev) -> bool:
        """The striking part grabs any fly within capture_r of it. Returns True once something was caught."""
        sp = self.spec
        for t in live:
            if self.kind == "dragonfly" and not t.flying:
                continue
            if float(np.linalg.norm(np.asarray(t.p, float) - self.tip)) <= sp.capture_r:
                self._tip_full = self.tip.copy()
                self.target = t.id
                self.carry = (t.id, CARRY_S)
                self._set("eat")
                ev.append(Event("capture", t.id, touch=CAPTURE_TOUCH[self.kind], at=self.tip.copy()))
                return True
        return False

    def _frog(self, dt: float, live, ev) -> None:
        sp = self.spec
        if self.state in ("aim", "strike", "retract", "recover", "eat"):
            self._strike_phase(dt, live, ev)
            return
        if self.state == "leave":
            self.p = self.p + self.heading * 1.6 * dt
            if self.t >= LEAVE_S:
                self.done = True
                ev.append(Event("gone"))
            return
        if not live:
            if self.t > 8.0:
                self._set("leave")
                ev.append(Event("leave"))
            return
        t, _ = self._nearest(live)
        dh, h = self._horizontal(t)
        height = float(np.asarray(t.p, float)[1] - self.p[1])
        self._since_hop += dt
        if self.state == "hop":
            self.p = self.p + _unit(h) * (HOP_LEN / HOP_S) * dt
            if self.t >= HOP_S:
                self._set("lurk")
                self._since_hop = 0.0
            return
        if dh <= sp.trigger and height <= sp.reach * 0.9:
            self._begin_aim(t, ev)
        elif self._since_hop >= HOP_EVERY_S and dh > sp.trigger * 0.6:
            self._set("hop")
        if self.t > 25.0 and self.state == "lurk":                 # nothing came near for a long time
            self._set("leave")
            ev.append(Event("leave"))

    def _mantis(self, dt: float, live, ev) -> None:
        sp = self.spec
        if self.state in ("aim", "strike", "retract", "recover", "eat"):
            self._strike_phase(dt, live, ev)
            return
        if self.state == "leave":
            self.p = self.p + self.heading * sp.speed * 3.0 * dt
            if self.t >= LEAVE_S * 2:
                self.done = True
                ev.append(Event("gone"))
            return
        if self.state == "lurk":
            self._set("stalk")
        if not live:
            if self.t > 15.0:
                self._set("leave")
                ev.append(Event("leave"))
            return
        t, _ = self._nearest(live)
        dh, h = self._horizontal(t)
        height = abs(float(np.asarray(t.p, float)[1] - self.p[1]))
        if dh <= sp.trigger and height <= sp.reach:
            self._begin_aim(t, ev)
            return
        if self._freeze_left > 0.0:                                 # frozen: no motion, so no looming
            self._freeze_left -= dt
            return
        self.p = self.p + _unit(h) * sp.speed * dt
        self._stalk_t += dt
        if self._stalk_t >= STALK_RUN_S:
            self._stalk_t = 0.0
            self._freeze_left = float(self.rng.uniform(*TELL_FREEZE_S))

    def _dragonfly(self, dt: float, live, ev) -> None:
        sp = self.spec
        if self.state == "leave":
            self.p = self.p + np.array([0.0, 1.0, 0.0]) * 2.5 * dt + self.heading * 1.0 * dt
            if self.t >= LEAVE_S:
                self.done = True
                ev.append(Event("gone"))
            return
        if self.state == "eat":
            self.tip = self.p.copy()
            if self.t >= CARRY_S:
                self.tip = None
                self._set("leave")
                ev.append(Event("leave"))
            return
        if self.age >= DRAGONFLY_GIVE_UP_S:
            self._set("leave")
            ev.append(Event("leave"))
            return
        prey, d = self._nearest(live, flying_only=True)
        if self.state == "patrol":
            ang = self.age * 0.9
            goal = self.anchor + np.array([math.cos(ang) * 1.6, math.sin(self.age * 1.7) * 0.12, math.sin(ang) * 1.6])
            self.p = self.p + _unit(goal - self.p) * min(2.0 * dt, float(np.linalg.norm(goal - self.p)))
            if prey is not None and d <= sp.trigger:
                self.target = prey.id
                self._set("pursue")
            return
        # pursue: steer toward where the fly will be, at a limited turn rate, and grab it within capture_r
        tgt = next((t for t in live if t.id == self.target and t.flying), None)
        if tgt is None:
            if any(t.id == self.target for t in live):
                self._ground_t += dt
                if self._ground_t >= DRAGONFLY_GROUND_PATIENCE_S:
                    self._ground_t = 0.0
                    self._set("patrol")
            else:
                self._set("patrol")
            return
        self._ground_t = 0.0
        tp = np.asarray(tgt.p, float)
        prev = self._prev.get(tgt.id, tp)
        fly_v = (tp - prev) / dt if dt > 0 and self.t > dt else np.zeros(3)
        lead = min(0.35, float(np.linalg.norm(tp - self.p)) / max(sp.speed, 1e-6))
        want = _unit(tp + fly_v * lead - self.p)
        cur = _unit(self.vel) if float(np.linalg.norm(self.vel)) > 1e-6 else want
        cosang = float(np.clip(np.dot(cur, want), -1.0, 1.0))
        ang = math.acos(cosang)
        k = 1.0 if ang < 1e-6 else min(1.0, DRAGONFLY_TURN * dt / ang)
        dirn = _unit(cur * (1 - k) + want * k)
        self.vel = dirn * sp.speed
        self.p = self.p + self.vel * dt
        if float(np.linalg.norm(tp - self.p)) <= sp.capture_r:
            self.tip = self.p.copy()
            self.carry = (tgt.id, CARRY_S)
            self._set("eat")
            ev.append(Event("capture", tgt.id, touch=CAPTURE_TOUCH["dragonfly"], at=self.p.copy()))


# --- placing a predator ---------------------------------------------------------------------------------------------------
def spawn_position(kind: str, bounds: Bounds, rng: np.random.Generator, near=None, altitude: float | None = None) -> np.ndarray:
    """Where a predator starts. GAME RULE. `near` is a fly's thorax: the frog and mantis start about a metre from it, on the
    floor; the dragonfly starts at altitude above it."""
    sp = SPECS[kind]
    n = np.asarray(near, float) if near is not None else np.array([(bounds.xmin + bounds.xmax) / 2, bounds.ymin,
                                                                   (bounds.zmin + bounds.zmax) / 2])
    if kind == "dragonfly":
        top = bounds.ymax - 0.5 if altitude is None else altitude
        p = np.array([n[0] + rng.uniform(-1.5, 1.5), top, n[2] + rng.uniform(-1.5, 1.5)])
    else:
        r = rng.uniform(2.0, 2.8) if kind == "frog" else rng.uniform(1.4, 2.0)
        a = rng.uniform(0, 2 * math.pi)
        p = np.array([n[0] + math.cos(a) * r, bounds.ymin, n[2] + math.sin(a) * r])
    return bounds.clip(p, margin=0.5)


# --- the headless trace the Lab assay and the tests use ----------------------------------------------------------------------
def trace(kind: str, seed: int, fly_pos=None, flying: bool | None = None, max_s: float = 40.0,
          bounds: Bounds | None = None) -> dict:
    """Run one predator against one fly that stays where it is (it has not noticed anything), and record what the fly's
    eyes would see: per 1/60 s frame, the list of (key, position, radius) threats; plus when the capture happens.
    Deterministic in `seed`. This is the assay's input: lab/predators.py feeds it to the real brain."""
    bounds = bounds or Bounds(-4.2, 4.2, 0.0, 3.0, -3.6, 3.6)
    rng = np.random.default_rng(seed)
    flying = (kind == "dragonfly") if flying is None else flying
    fly = np.array(fly_pos if fly_pos is not None else ([0.0, 1.5, 0.0] if flying else [0.0, THORAX_Y, 0.0]), float)
    p = Predator(kind, spawn_position(kind, bounds, rng, near=fly), rng, bounds)
    frames, head = [], fly + (0.0, 0.01, 0.0)
    capture_frame, aim_frame, strike_frame = None, None, None
    n = int(max_s / DT)
    target = Target("fly", fly, flying=flying)
    for i in range(n):
        ev = p.step(DT, [target])
        frames.append(p.threats())
        for e in ev:
            if e.kind == "aim" and aim_frame is None:
                aim_frame = i
            elif e.kind == "strike" and strike_frame is None:
                strike_frame = i
            elif e.kind == "capture":
                capture_frame = i
        if capture_frame is not None or p.done:
            break
    return dict(kind=kind, seed=seed, head=head, frames=frames, capture_frame=capture_frame, aim_frame=aim_frame,
                strike_frame=strike_frame, dt=DT)
