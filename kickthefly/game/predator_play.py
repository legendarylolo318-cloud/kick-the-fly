"""Predators in the game (3.0 day 3): the adapter between the engine (predators.py) and the 2D and 3D games.

The engine works in metres with y up. The 3D game is already in metres; the 2D game is in pixels with y down, so this
adapter converts (0.006 m a pixel, the same scale the 3D room uses). One class serves both games, so a frog does the same
thing in either.

What it does each frame: builds the engine's targets from the live flies, steps every predator, and acts on what the engine
reports. A capture fires the fly's REAL touch neurons by body part through `brain.poke` (head, body, legs, wings, as the tools
do), hurts it, and holds it at the predator's mouth; the predator then carries it off and eats it. An immortal fly (I) is let go
instead of eaten, as it breaks out of the spider's silk. Detection is not here at all: the games add `threats()` to the
objects the fly's eyes measure, so the existing looming transduction (LPLC2/LC4 -> DNp01) does the noticing.
"""
from __future__ import annotations

import random

import numpy as np

from kickthefly.game import predators as pr

S = 0.006                       # metres per 2D pixel (kick3d.S)
LABEL = {"frog": "A FROG!", "dragonfly": "A DRAGONFLY!", "mantis": "A MANTIS!"}
MAX_AT_ONCE = 3                 # one of each at most
CAPTURE_DAMAGE = 55.0           # GAME RULE: health lost at the grab; the rest is the eating


class PredatorPlay:
    def __init__(self, game, three_d: bool, floor_px: float = 0.0, ceil_px: float = 0.0, play_w: float = 0.0):
        self.g = game
        self.three_d = three_d
        self.floor, self.ceil, self.play_w = floor_px, ceil_px, play_w
        self.list: list[pr.Predator] = []
        self.held: dict = {}                                   # slot -> predator that is holding it
        self.rng = np.random.default_rng()

    # --- coordinates -------------------------------------------------------------------------------------------------
    def to_m(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        if self.three_d:
            return p.copy()
        return np.array([(p[0] - self.play_w / 2) * S, (self.floor - p[1]) * S, 0.0])

    def to_game(self, m) -> np.ndarray:
        m = np.asarray(m, float)
        if self.three_d:
            return m.copy()
        return np.array([m[0] / S + self.play_w / 2, self.floor - m[1] / S])

    def scale(self, r: float) -> float:
        return r if self.three_d else r / S

    def bounds(self) -> pr.Bounds:
        if not self.three_d:
            return pr.Bounds(-self.play_w / 2 * S, self.play_w / 2 * S, 0.0, (self.floor - self.ceil) * S, 0.0, 0.0)
        from kickthefly.game import kick3d
        from kickthefly.game import outdoors

        w = outdoors.spec(getattr(self.g, "world", "room"))
        ymax = 3.0 if w.lost_radius is None else 8.0
        return pr.Bounds(-w.rx, w.rx, 0.0, ymax, -w.rz, w.rz)

    # --- spawning and clearing -----------------------------------------------------------------------------------------
    def has(self, kind: str) -> bool:
        return any(p.kind == kind for p in self.list)

    def spawn(self, kind: str, at) -> bool:
        """Put a predator down at a point (game units: the click on the floor). One of each kind at a time."""
        if kind not in pr.SPECS or self.has(kind) or len(self.list) >= MAX_AT_ONCE:
            return False
        if not any(self._targetable(s) for s in self.g.flies):
            return False
        b = self.bounds()
        m = self.to_m(at)
        if kind == "dragonfly":
            m = np.array([m[0], b.ymax - 0.35, m[2]])
        else:
            m[1] = 0.0
        if not self.three_d:
            m[2] = 0.0
        p = pr.Predator(kind, b.clip(m, margin=0.25) if self.three_d else np.array([float(np.clip(m[0], b.xmin + .25, b.xmax - .25)), m[1], 0.0]),
                        np.random.default_rng(self.rng.integers(1 << 31)), b)
        self.list.append(p)
        self.g.popup(self._popup_at(p), LABEL[kind], (200, 220, 160))
        return True

    def clear(self) -> None:
        for slot in list(self.held):
            self._release(slot)
        self.list.clear()

    @staticmethod
    def _targetable(slot) -> bool:
        f = slot.fly
        return not (f.dead or f.dissolved_at is not None or f.shattered_at is not None)

    def _popup_at(self, p: pr.Predator):
        g = self.to_game(p.p)
        return g + (0.0, 0.3, 0.0) if self.three_d else g + (0.0, -40.0)

    # --- the engine's view of the game ---------------------------------------------------------------------------------
    def _targets(self) -> list:
        from kickthefly.game import kick_the_fly as k2

        out = []
        for slot in self.g.flies:
            f = slot.fly
            out.append(pr.Target(slot, self.to_m(f.p[k2.THX]), flying=bool(f.flying), alive=self._targetable(slot)))
        return out

    def threats(self) -> list:
        """(key, position, radius) in game units, for the looming measure (`_threats` in both games)."""
        out = []
        for p in self.list:
            for key, pos, r in p.threats():
                out.append((key, self.to_game(pos), self.scale(r)))
        return out

    def pin_for(self, slot):
        """Where a held fly is pinned (game units), or None."""
        p = self.held.get(slot)
        return None if p is None else self.to_game(p.grab_point())

    # --- each frame ------------------------------------------------------------------------------------------------------
    def step(self, now: float) -> None:
        if not self.list:
            return
        targets = self._targets()
        for p in list(self.list):
            for ev in p.step(pr.DT, targets):
                self._event(p, ev, now)
            if p.done:
                self.list.remove(p)
        for slot, p in list(self.held.items()):
            if p not in self.list or slot not in self.g.flies or not self._targetable(slot):
                self._release(slot)

    def _event(self, p: pr.Predator, ev: pr.Event, now: float) -> None:
        g, slot = self.g, ev.target
        if ev.kind == "capture" and slot is not None:
            self._capture(p, slot, ev, now)
        elif ev.kind == "leave":
            for s, holder in list(self.held.items()):
                if holder is p:
                    self._finish(p, s)
        elif ev.kind == "strike":
            g.sound.play("whack", 0.5)

    def _capture(self, p: pr.Predator, slot, ev: pr.Event, now: float) -> None:
        from kickthefly.game import kick_the_fly as k2

        g, fly = self.g, slot.fly
        for region, side, strength in ev.touch:                # CONNECTOME: the fly's own touch neurons, by body part
            slot.brain.poke(region, side, strength)
        if self.three_d:
            fly.last_hit = self.to_game(ev.at).copy()
        else:
            fly.last_hit_x = float(self.to_game(ev.at)[0])
        fly.hurt = 1.0
        g.damage(slot, CAPTURE_DAMAGE, f"a {p.kind}")
        g.sound.play("chomp")
        g.popup(self._popup_at(p), random.choice({"frog": ("SNAP!", "ZAP!", "GULP!"), "dragonfly": ("GRABBED!", "SNATCH!"),
                                                  "mantis": ("STRIKE!", "SNAP!")}[p.kind]), (230, 120, 120))
        fly.grabbed = k2.THX
        self.held[slot] = p
        g.note(f"CAUGHT   by a {p.kind}")

    def _finish(self, p: pr.Predator, slot) -> None:
        """The predator has carried the fly off and eats it. An immortal fly is let go."""
        g = self.g
        if getattr(g, "immortal", False):
            self._release(slot)
            g.popup(self._popup_at(p), "BROKE FREE!", (255, 225, 120), force=True)
            return
        g.damage(slot, 1000.0, f"eaten by a {p.kind}")
        self._release(slot)

    def _release(self, slot) -> None:
        self.held.pop(slot, None)
        f = slot.fly
        if f.grabbed is not None and not getattr(f, "wrapped", False):
            f.grabbed = None


# --- drawing: cheap primitives only, like the spider's ----------------------------------------------------------------------
# Each animal is drawn from a small design (frog body 0.16 m, mantis 0.12 m, dragonfly 0.10 m) multiplied by how much bigger
# its game body is, so the art and the looming radius always agree.
ART_R = {"frog": 0.16, "mantis": 0.12, "dragonfly": 0.10}


def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def draw3d(play: PredatorPlay, rd, now: float) -> None:
    """The predators in the 3D scene. A handful of spheres and cylinders each; nothing here touches the brain."""
    import math

    from kickthefly.game.render3d import P_NONE, segment, trs

    for p in play.list:
        sc = p.spec.body_r / ART_R[p.kind]
        x, y, z = p.p
        face = p.face
        side = np.array([-face[2], 0.0, face[0]])
        up = np.array([0.0, 1.0, 0.0])

        def sph(c, r3, col, *a):
            rd.add("sphere", trs(c, None, tuple(v * sc for v in r3)), col, *a)

        def cyl(a, b, r, col, *rest):
            rd.add("cylinder", segment(a, b, r * sc), col, *rest)

        if p.kind == "frog":
            g1, g2 = (0.25, 0.5, 0.2), (0.35, 0.62, 0.25)
            body = np.array([x, y + 0.08 * sc, z])
            sph(body, (0.15, 0.09, 0.17), g1)
            sph(body + face * 0.14 * sc + up * 0.04 * sc, (0.1, 0.07, 0.1), g2)
            for sg in (-1, 1):
                eye = body + face * 0.17 * sc + side * 0.06 * sc * sg + up * 0.1 * sc
                sph(eye, (0.03, 0.03, 0.03), (0.95, 0.85, 0.3))
                sph(eye + face * 0.02 * sc + up * 0.005 * sc, (0.014, 0.014, 0.014), (0.05, 0.05, 0.05))
                hip = body - face * 0.1 * sc + side * 0.12 * sc * sg
                cyl(hip, hip - face * 0.12 * sc + side * 0.06 * sc * sg - up * 0.05 * sc, 0.025, g1)
            if p.tip is not None:
                rd.add("cylinder", segment(p.mouth, p.tip, 0.03), (0.9, 0.35, 0.4), P_NONE)
                rd.add("sphere", trs(p.tip, None, (p.spec.tip_r,) * 3), (0.95, 0.45, 0.5), P_NONE)
        elif p.kind == "mantis":
            col, dark = (0.45, 0.62, 0.22), (0.3, 0.45, 0.15)
            base = np.array([x, y + 0.12 * sc, z])
            neck = base + face * 0.06 * sc + up * 0.16 * sc
            cyl(base - face * 0.22 * sc - up * 0.02 * sc, base + up * 0.02 * sc, 0.03, col)
            cyl(base + up * 0.02 * sc, neck, 0.022, col)
            sph(neck + face * 0.03 * sc + up * 0.03 * sc, (0.035, 0.03, 0.03), dark)
            for sg in (-1, 1):
                sph(neck + face * 0.045 * sc + side * 0.022 * sc * sg + up * 0.045 * sc, (0.012, 0.012, 0.012), (0.1, 0.1, 0.08))
                leg = base - face * 0.1 * sc + side * 0.04 * sc * sg
                cyl(leg, leg + side * 0.07 * sc * sg - up * 0.12 * sc, 0.007, dark)
                arm = neck + side * 0.025 * sc * sg
                if p.tip is not None and p.state in ("strike", "retract", "eat"):
                    rd.add("cylinder", segment(arm, p.tip, 0.02), col)
                    rd.add("sphere", trs(p.tip, None, (p.spec.tip_r,) * 3), dark, P_NONE)
                else:
                    elbow = arm + face * 0.05 * sc + up * 0.04 * sc
                    cyl(arm, elbow, 0.01, col)
                    cyl(elbow, elbow + face * 0.03 * sc - up * 0.07 * sc, 0.008, dark)
        else:                                                   # dragonfly
            dirn = _norm(p.vel) if float(np.linalg.norm(p.vel)) > 1e-3 else face
            body = np.array([x, y, z])
            blue = (0.15, 0.45, 0.75)
            cyl(body - dirn * 0.2 * sc, body + dirn * 0.05 * sc, 0.012, blue)
            sph(body + dirn * 0.06 * sc, (0.028, 0.026, 0.028), (0.1, 0.6, 0.35))
            sd = np.array([-dirn[2], 0.0, dirn[0]])
            flap = 0.25 + 0.2 * math.sin(now * 55)
            for sg in (-1, 1):
                for off, ln in ((0.02, 0.17), (-0.03, 0.15)):
                    a = body + dirn * off * sc
                    rd.add("cylinder", segment(a, a + sd * ln * sc * sg + up * flap * ln * 0.6 * sc, 0.006 * sc),
                           (0.85, 0.9, 0.95, 0.55), P_NONE)
        rd.add("sphere", trs(np.array([x, 0.005, z]), None, (0.14 * sc, 0.004, 0.14 * sc)), (0.0, 0.0, 0.0, 0.25), P_NONE)


def draw2d(play: PredatorPlay, surf, now: float) -> None:
    """The predators in the 2D game, side-on. Simple shapes from the game's own drawing helpers."""
    import math

    from pygame import gfxdraw

    from kickthefly.game import kick_the_fly as k2

    for p in play.list:
        sc = p.spec.body_r / ART_R[p.kind]
        gx, gy = play.to_game(p.p)
        d = 1 if p.face[0] >= 0 else -1                           # which way it faces

        def P(dx, dy):
            return (gx + d * dx * sc, gy + dy * sc)

        if p.kind == "frog":
            k2.aacircle(surf, P(0, -14), 22 * sc, (62, 128, 52))
            k2.aacircle(surf, P(18, -22), 12 * sc, (84, 150, 64))
            k2.aacircle(surf, P(22, -33), 5 * sc, (240, 220, 80))
            k2.aacircle(surf, P(23, -33), 2 * sc, (20, 20, 20))
            k2.thick_line(surf, P(-12, -4), P(-26, 0), 5 * sc, (50, 108, 42))
            if p.tip is not None:
                mx, my = play.to_game(p.mouth)
                tx, ty = play.to_game(p.tip)
                k2.thick_line(surf, (mx, my), (tx, ty), 5, (230, 90, 100))
                k2.aacircle(surf, (tx, ty), max(7, p.spec.tip_r / 0.006), (245, 120, 130))
        elif p.kind == "mantis":
            col, dark = (116, 158, 56), (78, 116, 38)
            k2.thick_line(surf, P(-30, -16), P(0, -22), 7 * sc, col)
            k2.thick_line(surf, P(0, -22), P(8, -52), 5 * sc, col)
            k2.aacircle(surf, P(11, -58), 7 * sc, dark)
            k2.aacircle(surf, P(15, -61), 2 * sc, (20, 20, 16))
            for k in (-1, 1):
                k2.thick_line(surf, P(-12, -20), P(-12 + k * 7, 0), 2.5 * sc, dark)
            if p.tip is not None and p.state in ("strike", "retract", "eat"):
                tx, ty = play.to_game(p.tip)
                k2.thick_line(surf, P(8, -50), (tx, ty), 5, col)
                k2.aacircle(surf, (tx, ty), max(5, p.spec.tip_r / 0.006), dark)
            else:
                k2.thick_line(surf, P(8, -50), P(22, -44), 4 * sc, col)
                k2.thick_line(surf, P(22, -44), P(26, -28), 3 * sc, dark)
        else:
            dd = 1 if (p.vel[0] >= 0 if float(np.linalg.norm(p.vel)) > 1e-3 else p.face[0] >= 0) else -1
            k2.thick_line(surf, (gx - dd * 34 * sc, gy), (gx + dd * 8 * sc, gy), 4 * sc, (38, 116, 190))
            k2.aacircle(surf, (gx + dd * 12 * sc, gy), 6 * sc, (30, 150, 90))
            flap = (10 + 8 * math.sin(now * 40)) * sc
            for off in (-2, 8):
                gfxdraw.filled_ellipse(surf, int(gx - dd * off * sc), int(gy - flap), int(22 * sc), int(4 * sc), (200, 215, 235, 150))
                gfxdraw.filled_ellipse(surf, int(gx - dd * off * sc), int(gy + flap * 0.3), int(22 * sc), int(3 * sc), (200, 215, 235, 110))
        gfxdraw.filled_ellipse(surf, int(gx), int(k2.FLOOR + 1), int(26 * sc), 4, (0, 0, 0, 70))
