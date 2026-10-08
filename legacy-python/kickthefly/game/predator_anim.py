"""Predator animation (3.1.0 task 1): plain logic shared by the spider, the mantis and the games' drawing. No pygame, no OpenGL.

GAME RULE, all of it: how the legs step, how long a wind-up lasts. None of it reaches the brain; the fly sees a predator only as an object
growing in its view (the existing looming pipeline), and a bite fires the fly's touch neurons exactly as before.

  two_bone_ik   the knee of a two-segment leg whose foot is on a target
  LegRig        a set of IK legs with planted feet: a foot stays where it was put until the body has walked away from it, then it steps
                (alternating groups, an arc), so a walking animal never slides its feet
  SpiderCycle   walk -> windup -> strike -> recover, the spider's bite as a cycle a fly can dodge (the frog's hop works the same way:
                predators.Predator._frog)
"""
from __future__ import annotations

import math

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def ease(x: float) -> float:
    """Smoothstep, clamped to 0..1."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def two_bone_ik(root, target, l1: float, l2: float, bend) -> tuple[np.ndarray, np.ndarray]:
    """(knee, foot) for a leg of segments l1, l2 from `root` toward `target`. A target out of reach (or too close) is clamped, so the
    foot is the point actually reached. `bend` is the side the knee points to."""
    root, target = np.asarray(root, float), np.asarray(target, float)
    d = target - root
    dist = float(np.linalg.norm(d))
    lo, hi = abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6
    if dist < 1e-9:
        d, dist = UP * -1.0, 1.0
    dirn = d / dist
    dist = min(max(dist, lo), hi)
    foot = root + dirn * dist
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    b = np.asarray(bend, float)
    b = b - dirn * float(np.dot(b, dirn))
    b = _unit(b) if float(np.linalg.norm(b)) > 1e-9 else _unit(np.cross(dirn, [1.0, 0.0, 0.0]))
    return root + dirn * a + b * h, foot


class LegRig:
    """IK legs with planted feet. Offsets are in the body frame (forward, up, side); feet rest on the ground plane y = `ground`.

    `groups` assigns each leg to a gait group; a leg may start a step only when no leg of another group is mid-step, which gives the
    alternating tetrapod gait of a spider or the diagonal walk of a mantis. A planted foot never moves in the world."""

    def __init__(self, hips, rests, l1: float, l2: float, groups, step_len: float, step_s: float = 0.18, lift: float = 0.04):
        self.hips = np.asarray(hips, float)
        self.rests = np.asarray(rests, float)
        self.l1, self.l2 = l1, l2
        self.groups = list(groups)
        self.step_len, self.step_s, self.lift = step_len, step_s, lift
        self.n = len(self.hips)
        self.feet: list[np.ndarray | None] = [None] * self.n
        self._from = [np.zeros(3)] * self.n
        self._to = [np.zeros(3)] * self.n
        self._t = [-1.0] * self.n                      # < 0 planted, else the progress of the step in seconds

    @staticmethod
    def _world(pos, fwd, off, ground=None):
        side = np.cross(fwd, UP)
        w = pos + fwd * off[0] + side * off[2] + UP * off[1]
        if ground is not None:
            w[1] = ground
        return w

    def stepping(self) -> int:
        return sum(1 for t in self._t if t >= 0.0)

    def update(self, pos, fwd, dt: float, moving: bool = True, ground: float | None = 0.0, overrides: dict | None = None,
               hang: float = 0.0):
        """Advance the feet and return [(hip, knee, foot)] in world space. `ground` None: the animal is off the ground (dangling): the
        feet hang `hang` below their hips and nothing is planted. `overrides` {leg: world point}: that foot is held there (a raised
        foreleg in a wind-up)."""
        pos = np.asarray(pos, float)
        fwd = _unit(np.array([fwd[0], 0.0, fwd[2]]) if len(fwd) == 3 else np.array([fwd[0], 0.0, 0.0]))
        if float(np.linalg.norm(fwd)) < 1e-9:
            fwd = np.array([0.0, 0.0, 1.0])
        side = np.cross(fwd, UP)
        out = []
        for i in range(self.n):
            hip = self._world(pos, fwd, self.hips[i])
            rest = self._world(pos, fwd, self.rests[i], ground if ground is not None else None)
            if overrides and i in overrides:
                self.feet[i], self._t[i] = np.asarray(overrides[i], float).copy(), -1.0
            elif ground is None:
                rest = hip + fwd * (self.rests[i][0] - self.hips[i][0]) + side * (self.rests[i][2] - self.hips[i][2]) - UP * hang
                self.feet[i], self._t[i] = rest, -1.0
            elif self.feet[i] is None:
                self.feet[i] = rest.copy()
            elif self._t[i] >= 0.0:
                self._t[i] += dt
                k = ease(self._t[i] / self.step_s)
                tgt = self._to[i]
                self.feet[i] = self._from[i] + (tgt - self._from[i]) * k + UP * (self.lift * math.sin(math.pi * k))
                if self._t[i] >= self.step_s:
                    self.feet[i], self._t[i] = tgt.copy(), -1.0
            else:
                far = float(np.linalg.norm((self.feet[i] - rest)[[0, 2]]))
                busy = any(self._t[j] >= 0.0 and self.groups[j] != self.groups[i] for j in range(self.n))
                if far > self.step_len and not busy:
                    lead = fwd * (self.step_len * 0.6) if moving else 0.0
                    self._from[i], self._to[i], self._t[i] = self.feet[i].copy(), rest + lead, 0.0
            foot = self.feet[i]
            out.append((hip, *two_bone_ik(hip, foot, self.l1, self.l2, UP + side * (0.0 if self.hips[i][2] == 0 else
                                                                                    0.35 * math.copysign(1.0, self.hips[i][2])))))
        return out


# --- the spider's bite --------------------------------------------------------------------------------------------------------
WINDUP_S, STRIKE_S, SPIDER_RECOVER_S = 0.28, 0.08, 0.40       # GAME RULE: 0.76 s from the first sign of a bite to the next, as 0.7 s before
BITE_TOLERANCE = 1.45                                           # the bite lands if the fly is still within this many reaches at the strike


class SpiderCycle:
    """walk -> windup -> strike -> recover. A spider in reach rears and raises its front legs (windup, nothing else moves), lunges
    (strike) and bites at the END of the lunge if the fly is still within BITE_TOLERANCE reaches; otherwise it misses. It walks only in
    the `walk` state. Timings are GAME RULE."""

    def __init__(self):
        self.state, self.t = "walk", 0.0

    @property
    def walking(self) -> bool:
        return self.state == "walk"

    def _set(self, s: str) -> None:
        self.state, self.t = s, 0.0

    def step(self, dt: float, dist: float, reach: float) -> list[str]:
        """Advance by dt; `dist` to the fly, `reach` the spider's striking distance (any one unit). Events: 'windup', 'bite', 'miss'."""
        ev: list[str] = []
        self.t += dt
        if self.state == "walk":
            if dist <= reach:
                self._set("windup")
                ev.append("windup")
        elif self.state == "windup":
            if self.t >= WINDUP_S:
                self._set("strike")
        elif self.state == "strike":
            if self.t >= STRIKE_S:
                ev.append("bite" if dist <= reach * BITE_TOLERANCE else "miss")
                self._set("recover")
        elif self.state == "recover":
            if self.t >= SPIDER_RECOVER_S:
                self._set("walk")
        return ev

    def pose(self) -> dict:
        """crouch 0..1 (body lowered, rearing), raise 0..1 (front legs up), lunge 0..1 (body thrown forward)."""
        s, t = self.state, self.t
        if s == "windup":
            k = ease(t / WINDUP_S)
            return dict(crouch=k, raise_=k, lunge=0.0)
        if s == "strike":
            k = ease(t / STRIKE_S)
            return dict(crouch=1.0 - k, raise_=1.0 - 0.6 * k, lunge=k)
        if s == "recover":
            k = ease(t / SPIDER_RECOVER_S)
            return dict(crouch=0.0, raise_=0.4 * (1.0 - k), lunge=1.0 - k)
        return dict(crouch=0.0, raise_=0.0, lunge=0.0)


def spider_rig(u: float = 1.0) -> LegRig:
    """Eight legs, four a side, in a tetrapod gait. `u` scales every length (1 for the 3D game's metres, about 150 for the 2D game's pixels)."""
    hips, rests, groups = [], [], []
    for sgn in (-1, 1):
        for k, (hf, rf) in enumerate(zip((0.07, 0.025, -0.025, -0.07), (0.17, 0.07, -0.07, -0.17))):
            hips.append((hf * u, 0.0, 0.04 * sgn * u))
            rests.append((rf * u, 0.0, 0.22 * sgn * u))
            groups.append((k + (sgn > 0)) % 2)
    return LegRig(hips, rests, 0.17 * u, 0.17 * u, groups, step_len=0.07 * u, step_s=0.14, lift=0.05 * u)
