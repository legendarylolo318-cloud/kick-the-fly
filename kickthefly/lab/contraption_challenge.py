"""The contraption builder's challenge class (3.1.0 task 13): the editor, the run and the report. The machine itself is kickthefly/game/contraption.py.

Esc > Challenges > Contraption, in the 3D game and in --2d. The editor is a side-view bench drawn on the HUD in both games; a run puts the machine in the world
(2D: on the arena, side-on; 3D: standing along the room's x axis, in the plane z = 0) and fires the parts' effects through the game's own code:

  * Tool, Sugar and the marble's or a domino's strike on the fly call the game's tool code at the part's position (`use_tool` in 2D, `use_tool3d` with the aim and the tool tip pointed
    at the part in 3D), so a swat, a bomb, a zap, a cVA puff, sugar, alcohol, fruit, a decoy, a spider and a flick drive the neurons they drive in play. Nothing is re-implemented here.
  * A fan drives the Johnston's organ wind neurons and a lamp the photoreceptors with `Brain.poke("wind" | "light")`, the regions the fan and lamp arenas use.

The fan does not push the fly's body, and the lamp does not make it walk to the light; those are the arenas' rules and a contraption leaves them out so that each part does one thing.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from kickthefly.game import contraption as ct
from kickthefly.lab.challenges import ACCENT, AMBER, BAD, DIM, GOOD, INK, LABEL, TEXT, Button, Challenge, load_scores, record_score
from kickthefly.lab.puppet_challenge import event_name, is_airborne

S = 0.006                                   # metres per pixel in the 2D game (the same scale Puppeteer uses)
AFTERMATH_S = 3.0                           # after the machine stops, the run goes on this long so the fly's reaction can be seen
CH_COLORS = ((190, 190, 200), (235, 90, 80), (90, 150, 245), (110, 205, 120), (240, 190, 70), (190, 120, 230))
TOOL_ABBR = {"flick": "FL", "swatter": "SW", "bomb": "BM", "zapper": "ZP", "cva": "cV", "alcohol": "AL", "fruit": "FR", "decoy": "DC", "spider": "SP"}
PROPS = {   # kind -> [(field, label, formatter)]
    "marble": [("t", "Delay", lambda v: f"{v / 10:.1f} s")],
    "ramp": [("a", "Slope", lambda v: f"{v} deg"), ("n", "Length", lambda v: f"{v} cm")],
    "spring": [("a", "Launch angle", lambda v: f"{v} deg")],
    "domino": [("n", "Height", lambda v: f"{v} cm")],
    "fan": [("a", "Direction", lambda v: f"{v} deg"), ("n", "Reach", lambda v: f"{v} cm"), ("ch", "Channel", lambda v: "start" if v == 0 else str(v))],
    "lamp": [("ch", "Channel", lambda v: "start" if v == 0 else str(v)), ("t", "Lit for", lambda v: f"{v / 10:.0f} s")],
    "sugar": [("ch", "Channel", lambda v: "start" if v == 0 else str(v))],
    "tool": [("tool", "Tool", str), ("ch", "Channel", lambda v: "start" if v == 0 else str(v)), ("c", "Shots", str)],
    "button": [("ch", "Sends", lambda v: str(v))],
    "timer": [("t", "Rings at", lambda v: f"{v / 10:.1f} s"), ("ev", "Again every", lambda v: "once" if v == 0 else f"{v / 10:.1f} s"), ("c", "Times", str),
              ("ch", "Sends", lambda v: str(v))],
    "fly": [("a", "Faces", lambda v: "left" if v >= 90 else "right"), ("c", "Tethered", lambda v: "yes" if v else "no")],
}


# --- the game's world, both ways ------------------------------------------------------------------------------------------------------------------
def fly_body(game, slot) -> tuple[float, float, float, float]:
    """(x, y, z, radius) of the fly's thorax in metres on the bench's axes: x along it, y up, z toward you."""
    from kickthefly.game import kick_the_fly as k2

    t = slot.fly.p[k2.THX]
    if getattr(game, "three_d", False):
        return float(t[0]), float(t[1]), float(t[2]), 0.16
    return float((t[0] - k2.PLAY_W / 2) * S), float((k2.FLOOR - t[1]) * S), 0.0, 0.16


def place_fly(game, slot, x: float, facing_right: bool) -> None:
    """Put the focused fly at x on the floor, facing the way given, standing still (the same fly object the slot had is replaced, as Puppeteer does)."""
    from kickthefly.game import kick_the_fly as k2

    if getattr(game, "three_d", False):
        from kickthefly.game import kick3d

        slot.fly = kick3d.Fly3D((x, 0.0), yaw=0.0 if facing_right else math.pi)
        slot.fly.yaw_target = slot.fly.yaw
    else:
        slot.fly = k2.Fly(k2.PLAY_W / 2 + x / S)
        slot.fly.facing = 1 if facing_right else -1
    slot.reset_episode()


def fire_tool(game, name: str, x: float, y: float, now: float) -> None:
    """Fire one of the game's tools at (x, y) on the bench, through the game's own code. In 3D the tool code reads where you aim and where the tool is, so for the length of the call
    (and a moment after, for the swatter's swing) the game is told the part is the tool and the fly is what it aims at."""
    from kickthefly.game import kick_the_fly as k2

    saved = game.tool
    game.tool = k2.TOOL_NAMES.index(name)
    try:
        if not getattr(game, "three_d", False):
            game.use_tool((k2.PLAY_W / 2 + x / S, k2.FLOOR - y / S), now)
            return
        pos = np.array([x, y, 0.0])
        d = np.array([0.0, -1.0, 0.0])
        if name in ("flick", "swatter", "zapper"):                        # aimed at the fly they are for
            live = [s for s in game.flies if not s.fly.dead]
            if live:
                to = live[min(game.focus, len(live) - 1)].fly.p[k2.THX] - pos
                n = float(np.linalg.norm(to))
                if n > 1e-6:
                    d = to / n
        eye = pos - d * (1.05 if name == "swatter" else 0.05)
        game.aim = lambda: (eye, d)
        game.tool_tip = lambda: pos.copy()
        game._ctr_aim_until = now + 0.4
        game.use_tool3d(now)
    finally:
        game.tool = saved


def tether(game, slot, x: float) -> None:
    """Hold the fly at x on the floor, along the bench and across it (GAME RULE): its body is slid back each frame, so it can rear, jump and flail in place, and its brain
    reacts as it would. Only the position is held: every reaction is still the brain's."""
    from kickthefly.game import kick_the_fly as k2

    fly = slot.fly
    if getattr(game, "three_d", False):
        t = fly.p[k2.THX]
        shift = np.array([x - t[0], 0.0, 0.0 - t[2]])
        fly.p += shift
        fly.prev += shift
        fly.anchor = np.array([x, 0.0])
    else:
        shift = np.array([k2.PLAY_W / 2 + x / S - fly.p[k2.THX][0], 0.0])
        fly.p += shift
        fly.prev += shift
        fly.anchor_x = k2.PLAY_W / 2 + x / S


def release_aim(game) -> None:
    for a in ("aim", "tool_tip"):
        if a in game.__dict__:
            del game.__dict__[a]
    game._ctr_aim_until = 0.0


# --- drawing (shared by the editor, the 2D arena and the previews) --------------------------------------------------------------------------------
def _font(size: int):
    return pygame.font.Font(None, size)


def part_dist(p: ct.Part, x: float, y: float) -> float:
    """How far (m) a bench point is from a part, for picking."""
    px, py = p.x / 100, p.y / 100
    if p.k == "ramp":
        ax, ay, bx, by = ct.Machine._ramp_ends(p)
        sx, sy = bx - ax, by - ay
        u = max(0.0, min(1.0, ((x - ax) * sx + (y - ay) * sy) / (sx * sx + sy * sy)))
        return math.hypot(x - (ax + u * sx), y - (ay + u * sy))
    if p.k == "domino":
        return math.hypot(x - px, max(0.0, abs(y - (py + p.n / 200)) - p.n / 200))
    return math.hypot(x - px, y - py - 0.03)


def draw_scene(surf, build: ct.Build, T, u: float, snap: dict | None = None, sel: int | None = None, font=None) -> None:
    """Draw every part. T(x_m, y_m) -> pixel; u = pixels per metre. `snap` (Machine.snapshot) animates the marbles, the dominoes, the fans and the lamps."""
    font = font or _font(16)
    fired = (snap or {}).get("fired", {})
    on_fans = set((snap or {}).get("fans", ()))
    on_lamps = set((snap or {}).get("lamps", ()))
    dom_angle = {}
    if snap:
        di = [i for i, p in enumerate(build.parts) if p.k == "domino"]
        for i, d in zip(di, snap["dominoes"]):
            dom_angle[i] = d[3]
    for i, p in enumerate(build.parts):
        x, y = p.x / 100, p.y / 100
        cx, cy = T(x, y)
        col_sel = (255, 235, 140)
        if p.k == "ramp":
            ax, ay, bx, by = ct.Machine._ramp_ends(p)
            pygame.draw.line(surf, (160, 128, 96), T(ax, ay), T(bx, by), max(3, int(0.03 * u)))
            if sel == i:
                pygame.draw.line(surf, col_sel, T(ax, ay), T(bx, by), 1)
        elif p.k == "spring":
            w = 0.2 * u
            pygame.draw.rect(surf, (90, 96, 112), (cx - w / 2, cy - 0.03 * u, w, 0.03 * u))
            pts = [(cx + (6 if k % 2 else -6) * u / 140, cy - 0.03 * u - k * 0.012 * u) for k in range(7)]
            pygame.draw.lines(surf, (200, 205, 215), False, pts, 2)
            a = math.radians(p.a)
            pygame.draw.line(surf, (120, 200, 150), (cx, cy - 0.1 * u), (cx + math.sin(a) * 0.2 * u, cy - 0.1 * u - math.cos(a) * 0.2 * u), 2)
        elif p.k == "domino":
            h = p.n / 100
            ang = dom_angle.get(i, 0.0)
            s = 1.0 if ang >= 0 else -1.0
            if ang == 0.0 and snap is None:
                s = 1.0
            phi = abs(ang)
            pivx, pivy = x + s * 0.02, y
            pts = []
            for (lu, lv) in ((-0.02 - s * 0.02, 0.0), (0.02 - s * 0.02, 0.0), (0.02 - s * 0.02, h), (-0.02 - s * 0.02, h)):
                pts.append(T(pivx + lu * math.cos(phi) + s * lv * math.sin(phi), pivy + lv * math.cos(phi) - s * lu * math.sin(phi)))
            pygame.draw.polygon(surf, (235, 235, 240), pts)
            pygame.draw.polygon(surf, (60, 64, 76), pts, 1)
        elif p.k == "fan":
            a = math.radians(p.a)
            pygame.draw.circle(surf, (110, 118, 136), (cx, cy), int(0.07 * u))
            for k in range(3):
                b = a + math.pi / 2 + k * 2.1 + (snap["t"] * 14 if i in on_fans else 0)
                pygame.draw.line(surf, (190, 200, 220), (cx, cy), (cx + math.cos(b) * 0.065 * u, cy - math.sin(b) * 0.065 * u), 3)
            if i in on_fans or snap is None:
                n = 3 if i in on_fans else 1
                for k in range(n):
                    off = (-0.12 + 0.12 * k) if n > 1 else 0.0
                    sx, sy = x + math.cos(a) * 0.12 - math.sin(a) * off, y + math.sin(a) * 0.12 + math.cos(a) * off
                    ex, ey = sx + math.cos(a) * min(0.5, p.n / 100 * 0.5), sy + math.sin(a) * min(0.5, p.n / 100 * 0.5)
                    pygame.draw.line(surf, (150, 210, 255) if i in on_fans else (80, 100, 125), T(sx, sy), T(ex, ey), 2)
        elif p.k == "lamp":
            lit = i in on_lamps
            if lit:
                glow = pygame.Surface((int(0.6 * u), int(0.6 * u)), pygame.SRCALPHA)
                pygame.draw.circle(glow, (255, 235, 150, 70), (glow.get_width() // 2, glow.get_height() // 2), glow.get_width() // 2)
                surf.blit(glow, (cx - glow.get_width() // 2, cy - glow.get_height() // 2))
            pygame.draw.circle(surf, (255, 240, 170) if lit else (150, 140, 90), (cx, cy), max(5, int(0.06 * u)))
            pygame.draw.line(surf, (90, 90, 100), (cx, cy - 0.06 * u), (cx, cy - 0.16 * u), 2)
        elif p.k == "sugar":
            r = max(5, int(0.05 * u))
            pygame.draw.rect(surf, (250, 250, 255), (cx - r, cy - 2 * r, 2 * r, 2 * r))
            for gx, gy in ((-0.4, -0.3), (0.3, -0.6), (0.1, -1.4)):
                pygame.draw.circle(surf, (225, 225, 235), (int(cx + gx * r), int(cy + gy * r)), 2)
        elif p.k == "tool":
            r = max(8, int(0.07 * u))
            pygame.draw.rect(surf, (70, 62, 96), (cx - r, cy - 2 * r, 2 * r, 2 * r), border_radius=4)
            pygame.draw.rect(surf, (150, 140, 190), (cx - r, cy - 2 * r, 2 * r, 2 * r), 1, border_radius=4)
            surf.blit(font.render(TOOL_ABBR.get(p.tool, "?"), True, (240, 235, 255)), (cx - r + 3, cy - 2 * r + 4))
            if p.c > 1:
                surf.blit(font.render(f"x{p.c}", True, (200, 195, 220)), (cx - r, cy + 2))
        elif p.k == "button":
            on = i in fired
            pygame.draw.rect(surf, (80, 190, 110) if on else (200, 70, 70), (cx - 0.08 * u, cy - 0.025 * u, 0.16 * u, 0.025 * u), border_radius=3)
            pygame.draw.rect(surf, (60, 64, 76), (cx - 0.1 * u, cy - 0.006 * u, 0.2 * u, 0.012 * u))
        elif p.k == "timer":
            r = max(8, int(0.07 * u))
            pygame.draw.circle(surf, (210, 215, 225), (cx, cy), r)
            pygame.draw.circle(surf, (60, 64, 76), (cx, cy), r, 2)
            pygame.draw.line(surf, (60, 64, 76), (cx, cy), (cx, cy - r * 0.7), 2)
            pygame.draw.line(surf, (60, 64, 76), (cx, cy), (cx + r * 0.45, cy), 2)
            surf.blit(font.render(f"{p.t / 10:.1f}s", True, (200, 205, 215)), (cx - r, cy + r + 1))
        elif p.k == "fly":
            d = -1 if p.a >= 90 else 1
            pygame.draw.ellipse(surf, (60, 58, 66), (cx - 0.11 * u, cy - 0.13 * u, 0.22 * u, 0.1 * u))
            pygame.draw.circle(surf, (80, 76, 86), (int(cx + d * 0.12 * u), int(cy - 0.1 * u)), max(3, int(0.04 * u)))
            pygame.draw.ellipse(surf, (180, 200, 230), (cx - 0.1 * u, cy - 0.2 * u, 0.12 * u, 0.06 * u), 1)
        elif p.k == "marble":
            pygame.draw.circle(surf, (205, 205, 220), (cx, cy), max(5, int(ct.BALL_R * u)))
            pygame.draw.circle(surf, (120, 120, 140), (cx, cy), max(5, int(ct.BALL_R * u)), 1)
            if p.t:
                surf.blit(font.render(f"{p.t / 10:.1f}s", True, (200, 205, 215)), (cx + 8, cy - 8))
        if p.k in ("fan", "lamp", "sugar", "tool", "button", "timer"):
            pygame.draw.circle(surf, CH_COLORS[p.ch % len(CH_COLORS)], (cx + int(0.09 * u), cy - int(0.2 * u)), 5)
            pygame.draw.circle(surf, (20, 22, 30), (cx + int(0.09 * u), cy - int(0.2 * u)), 5, 1)
        if sel == i:
            pygame.draw.circle(surf, col_sel, (cx, cy - int(0.04 * u)), int(0.16 * u), 1)
    if snap:
        for bx, by in snap["balls"]:
            c = T(bx, by)
            pygame.draw.circle(surf, (225, 225, 240), c, max(5, int(ct.BALL_R * u)))
            pygame.draw.circle(surf, (90, 92, 110), c, max(5, int(ct.BALL_R * u)), 1)


class Contraption(Challenge):
    key = "contraption"

    def __init__(self, game):
        super().__init__(game)
        self.build = ct.Build("my contraption")
        self.state = "build"                     # build -> run -> report
        self.pick: str | None = None             # the palette's part to place (None = select and move)
        self.sel: int | None = None
        self.msg, self.msg_bad = "", False
        self.slot_mode = "load"                  # what a slot button does: load | save
        self.machine: ct.Machine | None = None
        self.last = 0.0
        self.t_end: float | None = None
        self.reactions: list[tuple[float, str]] = []
        self.report: dict | None = None
        self.runs = 0
        self.canvas = pygame.Rect(0, 0, 10, 10)
        self.buttons = []
        self._frame = 0
        self.fly_start: tuple[float, bool] = (0.9, True)             # (x, tethered)
        game.note_hooks.append(self._on_note)

    # --- the game's hooks ------------------------------------------------------------------------------------------------------------------------
    @property
    def overlay(self) -> bool:
        return self.state in ("build", "report")

    @property
    def speed(self) -> float:
        return 1.0

    def end(self) -> None:
        g = self.game
        if self._on_note in g.note_hooks:
            g.note_hooks.remove(self._on_note)
        release_aim(g)
        super().end()

    def _on_note(self, text: str) -> None:
        if self.state == "run":
            name = event_name(text)
            if name and self.machine is not None:
                self.reactions.append((self.machine.t, name))

    def _available(self, tool: str) -> bool:
        from kickthefly.core import loadout as lo

        g = self.game
        return lo.available(tool, lab=bool(g.cfg.lab), larva=bool(getattr(g, "is_larva", False)))

    def _say(self, text: str, bad: bool = False) -> None:
        self.msg, self.msg_bad = text, bad

    # --- editing ---------------------------------------------------------------------------------------------------------------------------------
    def place(self, wx: float, wy: float) -> None:
        part = ct.new_part(self.pick, wx * 100, wy * 100)
        if part.k == "tool" and not self._available(part.tool):
            part.tool = next((t for t in ct.TOOLS if self._available(t)), part.tool)
        try:
            self.build.add(part)
        except ct.BuildError as e:
            self._say(str(e), True)
            return
        self.sel = len(self.build.parts) - 1
        self._say("")

    def pick_at(self, wx: float, wy: float) -> int | None:
        best, bi = 0.12, None
        for i, p in enumerate(self.build.parts):
            d = part_dist(p, wx, wy)
            if d < best:
                best, bi = d, i
        return bi

    def canvas_click(self, pos) -> None:
        r = self.canvas
        u = r.w / (2 * ct.XMAX)
        wx, wy = (pos[0] - r.centerx) / u, (r.bottom - pos[1]) / u
        if self.pick:
            self.place(wx, wy)
            return
        i = self.pick_at(wx, wy)
        if i is not None:
            self.sel = i
        elif self.sel is not None:                           # clicking an empty place moves the selected part there
            p = self.build.parts[self.sel]
            p.x, p.y = ct.clamp_position(p.k, wx * 100, wy * 100)

    def delete_selected(self) -> None:
        if self.sel is not None and self.sel < len(self.build.parts):
            del self.build.parts[self.sel]
            self.sel = None

    def clear(self) -> None:
        self.build = ct.Build(self.build.name)
        self.sel = None
        self._say("")

    def load_example(self, name: str) -> None:
        self.build = ct.Build.from_json(ct.EXAMPLES[name].to_json())
        self.sel = None
        self._say(f"Loaded the example '{name}'.")

    def slot_press(self, i: int, slots: list) -> None:
        if self.slot_mode == "save":
            if ct.save_slot(i, self.build):
                self._say(f"Saved in slot {i + 1}.")
            else:
                self._say("Nothing to save: place a part first.", True)
        elif slots[i] is None:
            self._say(f"Slot {i + 1} is empty.", True)
        else:
            self.build = ct.Build.from_json(slots[i])
            self.sel = None
            self._say(f"Loaded slot {i + 1}.")

    def copy_code(self) -> None:
        from kickthefly.core import clipboard

        try:
            code = self.build.code()
        except ct.BuildError as e:
            self._say(str(e), True)
            return
        self._say("Code copied." if clipboard.put_text(code) else "Could not reach the clipboard.", False)

    def paste_code(self) -> None:
        from kickthefly.core import clipboard

        text = clipboard.get_text()
        if not text:
            self._say("The clipboard has no text.", True)
            return
        try:
            self.build = ct.Build.from_code(text, self._available)
        except ct.BuildError as e:
            self._say(str(e), True)
            return
        self.sel = None
        self._say(f"Loaded '{self.build.name}' from the code.")

    # --- running ---------------------------------------------------------------------------------------------------------------------------------
    def begin_run(self) -> None:
        bad = self.build.problems(self._available)
        if bad:
            self._say(bad[0], True)
            return
        if not [p for p in self.build.parts if p.k != "fly"]:
            self._say("Place at least one part besides the fly.", True)
            return
        g = self.game
        slot = self.slot
        mark = self.build.fly_mark()
        x, right = (mark.x / 100, mark.a < 90) if mark else (0.9, False)
        self.fly_start = (x, mark.c != 0 if mark else True)
        release_aim(g)
        place_fly(g, slot, x, right)
        self.machine = ct.Machine(self.build)
        self.state, self.last, self.t_end = "run", g.clock.now, None
        self.reactions, self.report, self._frame = [], None, 0
        self._say("")

    def stop_run(self) -> None:
        if self.state == "run":
            self._finish()

    def _finish(self) -> None:
        m = self.machine
        slot = self.slot
        fired = []
        for e in (m.log if m else []):
            if e.kind == "fire" and e.ch != 0:
                fired.append((e.t, f"channel {e.ch} signalled by {e.by}"))
            elif e.kind == "tool":
                fired.append((e.t, f"{e.tool} at {e.x:.2f} m, {e.y:.2f} m" + (f" ({e.by})" if e.by else "")))
            elif e.kind == "on":
                fired.append((e.t, f"the {e.tool} came on at {e.x:.2f} m" + (f" ({e.by})" if e.by else "")))
            elif e.kind == "hit":
                fired.append((e.t, f"a {e.by} met the fly"))
        self.report = {"fired": fired[:12], "reactions": self.reactions[:10], "seconds": m.t if m else 0.0,
                       "died": bool(slot.fly.dead), "parts_fired": len(m.fired) if m else 0}
        self.runs += 1
        record_score("contraption", float(load_scores().get("contraption", 0)) + 1.0, "high")
        release_aim(self.game)
        self.state = "report"

    def update(self, now: float) -> None:
        g = self.game
        if getattr(g, "_ctr_aim_until", 0.0) and now > g._ctr_aim_until:
            release_aim(g)
        if self.state != "run" or self.machine is None:
            return
        dt = min(max(now - self.last, 0.0), 0.05)
        self.last = now
        m = self.machine
        slot = self.slot
        self._frame += 1
        if dt > 0 and not m.finished:
            for _ in range(max(1, int(round(dt / ct.DT)))):
                body = fly_body(g, slot)
                for e in m.step(ct.DT, body):
                    if e.kind == "tool":
                        fire_tool(g, e.tool, e.x, e.y, now)
                if m.finished:
                    break
        if self.fly_start[1] and not slot.fly.dead and slot.fly.grabbed is None:
            tether(g, slot, self.fly_start[0])
        body = fly_body(g, slot)
        if not slot.fly.dead and self._frame % 3 == 0:
            w = m.wind_on(body[0], body[1], body[2])
            if w > 0.02:
                slot.brain.poke("wind", None, min(1.0, w * 1.4))            # Johnston's organ, as the fan arena drives it
            light = m.light_on(body[0], body[1], body[2])
            if light > 0.02:
                slot.brain.poke("light", None, light, recruit=0.25 * light)   # photoreceptors, as the lamp arena drives them
        if m.finished:
            if self.t_end is None:
                self.t_end = now
            elif now - self.t_end > AFTERMATH_S:
                self._finish()

    # --- clicks ----------------------------------------------------------------------------------------------------------------------------------
    def click(self, pos) -> bool:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(pos):
                self.game.sound.play("click")
                b.action()
                return True
        if self.state == "build" and self.canvas.collidepoint(pos):
            self.canvas_click(pos)
            return True
        return self.overlay

    # --- drawing ---------------------------------------------------------------------------------------------------------------------------------
    def draw(self, surf, now, mouse) -> None:
        self.buttons = []
        if self.state == "build":
            self._draw_editor(surf, mouse)
        elif self.state == "report":
            self._draw_report(surf, mouse)
        else:
            self._draw_run(surf, mouse)
        for b in self.buttons:
            b.draw(self.game, surf, mouse)

    def _fit(self, text: str, width: int) -> str:
        f = self.game.f_small
        if f.size(text)[0] <= width:
            return text
        while text and f.size(text + "...")[0] > width:
            text = text[:-1]
        return text.rstrip() + "..."

    def _tag_line(self, surf, x: int, y: int) -> None:
        g = self.game
        for label, col in (("CONNECTOME", (110, 205, 150)), ("GAME RULE", (235, 190, 80)), ("MODEL PREDICTION", (140, 170, 255))):
            w = g.f_small.size(label)[0] + 14
            pygame.draw.rect(surf, (28, 32, 42), (x, y, w, 20), border_radius=10)
            pygame.draw.rect(surf, col, (x, y, w, 20), 1, border_radius=10)
            g._text(surf, label, (x + 7, y + 3), col, g.f_small)
            x += w + 8

    def _draw_editor(self, surf, mouse) -> None:
        g = self.game
        panel = self.panel(surf, "Contraption", "Place parts, wire them with channels, and run it against the fly.", h=640)
        self._tag_line(surf, panel.right - 376, panel.y + 20)
        lx = panel.x + 14
        top = panel.y + 108
        # palette
        mode_sel = self.pick is None
        self.buttons.append(Button((lx, top, 118, 22), "Select / move", lambda: setattr(self, "pick", None), style="primary" if mode_sel else "normal"))
        for i, k in enumerate(ct.KINDS):
            self.buttons.append(Button((lx, top + 25 * (i + 1), 118, 22), ct.LABEL[k], (lambda k=k: setattr(self, "pick", k)), style="primary" if self.pick == k else "normal"))
        # canvas
        self.canvas = pygame.Rect(panel.x + 142, top, panel.w - 142 - 14, int((panel.w - 142 - 14) * ct.YMAX / (2 * ct.XMAX)))
        r = self.canvas
        u = r.w / (2 * ct.XMAX)
        pygame.draw.rect(surf, (14, 17, 24), r)
        for gx in range(-2, 3):
            px = int(r.centerx + gx * u)
            pygame.draw.line(surf, (30, 36, 48), (px, r.y), (px, r.bottom))
            if gx:
                g._text(surf, f"{gx} m", (px - 12, r.bottom - 16), (80, 90, 110), g.f_small)
        for gy in range(1, 3):
            py = int(r.bottom - gy * u)
            pygame.draw.line(surf, (30, 36, 48), (r.x, py), (r.right, py))
        pygame.draw.line(surf, (110, 96, 80), (r.x, r.bottom - 1), (r.right, r.bottom - 1), 3)
        pygame.draw.rect(surf, (52, 60, 76), r, 1)
        T = lambda x, y: (int(r.centerx + x * u), int(r.bottom - y * u))     # noqa: E731
        prev = surf.get_clip()
        surf.set_clip(r)
        draw_scene(surf, self.build, T, u, None, self.sel, g.f_small)
        if self.pick and r.collidepoint(mouse):
            wx, wy = (mouse[0] - r.centerx) / u, (r.bottom - mouse[1]) / u
            gx, gy = ct.clamp_position(self.pick, wx * 100, wy * 100)
            ghost = ct.Build("g", [ct.new_part(self.pick, gx, gy)])
            gs = pygame.Surface(r.size, pygame.SRCALPHA)
            draw_scene(gs, ghost, lambda x, y: (int(r.w / 2 + x * u), int(r.h - y * u)), u, None, None, g.f_small)
            gs.set_alpha(120)
            surf.blit(gs, r.topleft)
        surf.set_clip(prev)
        # the selected part's properties
        py = top + 25 * (len(ct.KINDS) + 1) + 6
        g._text(surf, "SELECTED PART", (lx, py), LABEL, g.f_small)
        py += 18
        if self.sel is not None and self.sel < len(self.build.parts):
            p = self.build.parts[self.sel]
            g._text(surf, ct.LABEL[p.k], (lx, py), INK, g.f_bold)
            py += 20
            for fld, label, fmt in PROPS[p.k]:
                g._text(surf, f"{label}: {fmt(getattr(p, fld))}", (lx, py), TEXT, g.f_small)
                py += 14
                self.buttons.append(Button((lx, py, 56, 18), "-", (lambda p=p, f=fld: ct.step_field(p, f, -1))))
                self.buttons.append(Button((lx + 62, py, 56, 18), "+", (lambda p=p, f=fld: ct.step_field(p, f, 1))))
                py += 22
            self.buttons.append(Button((lx, py, 118, 22), "Delete part", self.delete_selected, style="danger"))
        else:
            g._text(surf, "click a part", (lx, py), DIM, g.f_small)
        # the rows under the canvas
        y = r.bottom + 8
        x = r.x
        n_parts = len([p for p in self.build.parts if p.k != "fly"])
        g._text(surf, f"{n_parts}/{ct.MAX_PARTS} parts", (x, y + 3), LABEL, g.f_small)
        if self.pick:
            g._text(surf, self._fit(f"Place {ct.LABEL[self.pick].lower()}: {ct.HELP[self.pick]}", r.w - 96), (x + 96, y + 3), TEXT, g.f_small)
        elif self.msg:
            g._text(surf, self._fit(self.msg, r.w - 96), (x + 96, y + 3), BAD if self.msg_bad else GOOD, g.f_small)
        else:
            g._text(surf, self._fit("Channel 0 goes off at the start; a button or timer sends 1-5 (the dot's colour).", r.w - 96), (x + 96, y + 3), DIM, g.f_small)
        y += 24
        self.buttons.append(Button((x, y, 110, 32), "RUN", self.begin_run, style="primary"))
        self.buttons.append(Button((x + 118, y, 80, 32), "Clear", self.clear))
        ex = x + 206
        for name in ct.EXAMPLES:
            w = g.f_small.size(name)[0] + 22
            self.buttons.append(Button((ex, y, w, 32), name, (lambda n=name: self.load_example(n))))
            ex += w + 6
        self.buttons.append(Button((r.right - 80, y, 80, 32), "Close", self.end))
        y += 38
        slots = ct.load_slots()
        g._text(surf, "SLOTS", (x, y + 7), LABEL, g.f_small)
        sx = x + 50
        self.buttons.append(Button((sx, y, 62, 28), "Save" if self.slot_mode == "save" else "Load", lambda: setattr(self, "slot_mode", "load" if self.slot_mode == "save" else "save"),
                                   style="primary" if self.slot_mode == "save" else "normal"))
        sx += 68
        for i in range(ct.SLOTS):
            lab = str(i + 1) + ("" if slots[i] is None else "*")
            self.buttons.append(Button((sx, y, 34, 28), lab, (lambda i=i, s=slots: self.slot_press(i, s))))
            sx += 38
        self.buttons.append(Button((sx + 8, y, 100, 28), "Copy code", self.copy_code, enabled=bool(self.build.parts)))
        self.buttons.append(Button((sx + 114, y, 100, 28), "Paste code", self.paste_code))

    def _draw_run(self, surf, mouse) -> None:
        g = self.game
        from kickthefly.game import kick_the_fly as k2

        m = self.machine
        bar = pygame.Rect(258, 58, 400, 62)
        card = pygame.Surface(bar.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 215), card.get_rect(), border_radius=10)
        surf.blit(card, bar)
        g._text(surf, "CONTRAPTION", (bar.x + 12, bar.y + 8), AMBER, g.f_bold)
        g._text(surf, f"{m.t:4.1f} s" + ("   finished, watching the fly" if m.finished else ""), (bar.x + 150, bar.y + 10), INK, g.f_text)
        self.buttons += [Button((bar.right - 160, bar.y + 6, 70, 24), "Stop", self.stop_run), Button((bar.right - 82, bar.y + 6, 70, 24), "Edit", self._to_editor)]
        self._tag_line(surf, bar.x + 12, bar.y + 36)
        if not getattr(g, "three_d", False):
            T = lambda x, y: (int(k2.PLAY_W / 2 + x / S), int(k2.FLOOR - y / S))   # noqa: E731
            draw_scene(surf, self.build, T, 1.0 / S, m.snapshot(), None, g.f_small)

    def _to_editor(self) -> None:
        release_aim(self.game)
        self.state = "build"

    def _draw_report(self, surf, mouse) -> None:
        g = self.game
        panel = self.panel(surf, "Contraption ran", "What went off, and what the fly did about it.", h=520)
        self._tag_line(surf, panel.right - 376, panel.y + 20)
        r = self.report or {}
        x, y = panel.x + 24, panel.y + 100
        g._text(surf, f"{r.get('parts_fired', 0)} of {len(self.build.parts)} parts went off in {r.get('seconds', 0):.1f} s.   MODEL PREDICTION below.", (x, y), TEXT, g.f_text)
        y += 30
        g._text(surf, "WHAT WENT OFF   (GAME RULE: the machine; CONNECTOME: what each tool drives)", (x, y), LABEL, g.f_small)
        y += 20
        for t, line in r.get("fired", []) or [(0.0, "nothing went off")]:
            g._text(surf, f"{t:5.1f} s   {line}", (x + 8, y), INK, g.f_small)
            y += 17
        y += 12
        g._text(surf, "WHAT THE FLY DID   (MODEL PREDICTION: the connectome's own reactions)", (x, y), LABEL, g.f_small)
        y += 20
        react = r.get("reactions", [])
        if r.get("died"):
            g._text(surf, "The fly did not survive.", (x + 8, y), BAD, g.f_small)
            y += 17
        for t, name in react or [(0.0, "no reaction the game reads")]:
            g._text(surf, f"{t:5.1f} s   {name}", (x + 8, y), ACCENT, g.f_small)
            y += 17
        by = panel.bottom - 56
        self.buttons += [Button((x, by, 150, 40), "Run again", self.begin_run, style="primary"), Button((x + 160, by, 150, 40), "Edit", self._to_editor),
                         Button((panel.right - 170, by, 140, 40), "Close", self.end)]

    # --- the 3D room ---------------------------------------------------------------------------------------------------------------------------
    def draw_world3d(self, rd, now: float) -> None:
        """The machine standing in the room along the x axis at z = 0 (drawn only while it runs)."""
        if self.state != "run" or self.machine is None:
            return
        from kickthefly.game.render3d import P_NONE, segment, trs

        snap = self.machine.snapshot()
        fired = snap["fired"]
        W = 0.06                                                            # the bench's thickness in 3D (metres): every part is this deep
        for i, p in enumerate(self.build.parts):
            x, y = p.x / 100, p.y / 100
            if p.k == "ramp":
                ax, ay, bx, by = ct.Machine._ramp_ends(p)
                rd.add("cylinder", segment((ax, ay, 0.0), (bx, by, 0.0), 0.02), (0.63, 0.5, 0.38), P_NONE)
            elif p.k == "spring":
                rd.add("cube", trs((x, y + 0.015, 0.0), None, (0.2, 0.03, W)), (0.35, 0.38, 0.45), P_NONE)
                rd.add("cylinder", segment((x, y + 0.03, 0.0), (x, y + 0.11, 0.0), 0.012), (0.8, 0.82, 0.86), P_NONE)
            elif p.k == "fan":
                a = math.radians(p.a)
                on = i in snap["fans"]
                rd.add("cylinder", segment((x, y, -0.03), (x, y, 0.03), 0.08), (0.43, 0.47, 0.54), P_NONE)
                for k in range(3):
                    b = a + math.pi / 2 + k * 2.1 + (snap["t"] * 14 if on else 0)
                    rd.add("cylinder", segment((x, y, 0.035), (x + math.cos(b) * 0.075, y + math.sin(b) * 0.075, 0.035), 0.006), (0.76, 0.8, 0.88), P_NONE)
                if on:
                    for k in range(3):
                        off = 0.1 * (k - 1)
                        sx, sy = x + math.cos(a) * 0.14 - math.sin(a) * off, y + math.sin(a) * 0.14 + math.cos(a) * off
                        L = min(0.5, p.n / 100 * 0.5)
                        rd.add("cylinder", segment((sx, sy, 0.0), (sx + math.cos(a) * L, sy + math.sin(a) * L, 0.0), 0.004), (0.6, 0.82, 1.0, 0.7), P_NONE, 0.6)
            elif p.k == "lamp":
                lit = i in snap["lamps"]
                rd.add("cylinder", segment((x, y + 0.06, 0.0), (x, y + 0.2, 0.0), 0.006), (0.35, 0.35, 0.4), P_NONE)
                rd.add("sphere", trs((x, y, 0.0), None, (0.05, 0.05, 0.05)), (1.0, 0.95, 0.65) if lit else (0.6, 0.55, 0.35), P_NONE, 1.0 if lit else 0.0)
            elif p.k == "sugar":
                rd.add("cube", trs((x, y + 0.03, 0.0), None, (0.06, 0.06, 0.06)), (0.97, 0.97, 1.0), P_NONE)
            elif p.k == "tool":
                rd.add("cube", trs((x, y + 0.05, 0.0), None, (0.1, 0.1, 0.1)), (0.27, 0.24, 0.38), P_NONE)
                rd.add("sphere", trs((x, y + 0.12, 0.0), None, (0.025, 0.025, 0.025)), (0.95, 0.75, 0.3), P_NONE, 0.8)
            elif p.k == "button":
                rd.add("cube", trs((x, y + 0.006, 0.0), None, (0.2, 0.012, 0.12)), (0.25, 0.26, 0.3), P_NONE)
                pressed = i in fired
                rd.add("cube", trs((x, y + (0.012 if pressed else 0.022), 0.0), None, (0.15, 0.02 if not pressed else 0.008, 0.1)), (0.3, 0.75, 0.42) if pressed else (0.8, 0.27, 0.27), P_NONE)
            elif p.k == "timer":
                rd.add("cylinder", segment((x, y, -0.03), (x, y, 0.03), 0.07), (0.82, 0.84, 0.88), P_NONE)
            elif p.k == "fly":
                rd.add("cylinder", segment((x, 0.0, 0.0), (x, 0.005, 0.0), 0.12), (0.5, 0.8, 1.0, 0.5), P_NONE)
            elif p.k == "marble":
                pass
            if p.k in ("fan", "lamp", "sugar", "tool", "button", "timer"):
                c = CH_COLORS[p.ch % len(CH_COLORS)]
                rd.add("sphere", trs((x + 0.09, y + 0.22, 0.0), None, (0.03, 0.03, 0.03)), (c[0] / 255, c[1] / 255, c[2] / 255), P_NONE, 0.5)
        for (dx, dy, h, ang) in snap["dominoes"]:
            s = 1.0 if ang >= 0 else -1.0
            phi = abs(ang)
            rot = np.array([[math.cos(phi), -s * math.sin(phi), 0.0], [s * math.sin(phi), math.cos(phi), 0.0], [0.0, 0.0, 1.0]])
            # the domino pivots about its lower edge on the side it falls toward
            pivot = np.array([dx + s * 0.02, dy, 0.0])
            centre_local = np.array([-s * 0.02, h / 2, 0.0])
            rd.add("cube", trs(pivot + rot @ centre_local, rot, (0.04, h, 0.12)), (0.93, 0.93, 0.95), P_NONE)
        for bx, by in snap["balls"]:
            rd.add("sphere", trs((bx, by, 0.0), None, (ct.BALL_R * 1.05,) * 3), (0.85, 0.86, 0.93), P_NONE)
