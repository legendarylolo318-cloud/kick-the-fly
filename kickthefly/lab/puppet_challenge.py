"""Puppeteer mode's challenge class (3.1.0 task 10): the Challenge the game runs; the levels and the goal logic are kickthefly/game/puppeteer.py.

It plugs into the same hooks every Challenge uses (update, on_reaction, click, draw), plus three small ones in the game: `note_hooks` (every reaction the brain makes
passes through Game.note), `puppet_active` (nothing of yours may poke the fly; the only tool is the laser) and `select_slot` (the hotbar keys choose a palette type).
Works in the 3D game and in --2d: the fly's position and heading come from its own particles in both.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from kickthefly.game import puppeteer as pz
from kickthefly.lab.challenges import (ACCENT, AMBER, BAD, DIM, GOOD, INK, LABEL, TEXT, Button, Challenge, draw_stars, load_scores, record_score)

KNOWN_EVENTS = ("BACK UP", "TAKE OFF", "FLY AWAY", "TURN R", "TURN L", "WALK", "RUN", "KICK", "SONG", "GROOM", "SLEEP", "DODGE", "PROBOSCIS", "AVOID")
NO_TARGET = "no target"            # matches no neuron (an empty string would match every untyped one)
S = 0.006


def event_name(text: str) -> str | None:
    """The reaction a Game.note line reports: its first words, if they are one of the reactions (WALK, BACK UP, TURN R, ...)."""
    t = text.strip()
    for name in sorted(KNOWN_EVENTS, key=len, reverse=True):
        if t == name or t.startswith(name + " "):
            return name
    return None


def fly_state(game, slot) -> tuple[float, float, float, float]:
    """(x, z, hx, hz): where the fly is on the floor in metres and the unit vector it faces, in the 3D game and in the 2D one. Whether it is in
    the air is `is_airborne`."""
    from kickthefly.game import kick_the_fly as k2

    fly = slot.fly
    if getattr(game, "three_d", False):
        t, h = fly.p[k2.THX], fly.p[k2.HEAD] - fly.p[k2.ABD]
        v = np.array([h[0], h[2]])
        n = float(np.hypot(*v)) or 1.0
        return float(t[0]), float(t[2]), float(v[0] / n), float(v[1] / n)
    return float((fly.p[k2.THX][0] - k2.PLAY_W / 2) * S), 0.0, float(fly.facing), 0.0


def is_airborne(game, slot) -> bool:
    fly = slot.fly
    return bool(fly.escape_until > game.clock.now and getattr(fly, "grabbed", None) is None)


def place_fly(game, slot, level: pz.Level) -> None:
    """Put the focused fly at the level's start, facing the level's way, standing still. The same fly object is replaced in the same slot."""
    from kickthefly.game import kick_the_fly as k2

    if getattr(game, "three_d", False):
        from kickthefly.game import kick3d

        yaw = 0.0 if level.facing > 0 else math.pi
        slot.fly = kick3d.Fly3D((level.start_x, 0.0), yaw=yaw)
        slot.fly.yaw_target = slot.fly.yaw
    else:
        slot.fly = k2.Fly(k2.PLAY_W / 2 + level.start_x / S)
        slot.fly.facing = level.facing
    slot.reset_episode()
    slot.brain.clear_overrides()
    slot.brain.clear_current("puppet")


class Puppeteer(Challenge):
    key = "puppeteer"

    def __init__(self, game):
        super().__init__(game)
        self.state = "select"                    # select -> play -> won
        self.level: pz.Level | None = None
        self.tracker: pz.Tracker | None = None
        self.actions = self.hints = 0
        self.effect = "activate"                 # activate | silence
        self.how = "laser"                       # laser | latch
        self.target: str | None = None
        self.latched: dict[str, str] = {}
        self.hint_shown = False
        self.result: dict | None = None
        self.msg = ""
        self._was_firing = False
        self._sig = None
        self._loom_until = 0.0
        self.buttons = []
        self._snap: dict = {}                    # the brain as it was when you opened Puppeteer: every level starts from it, so what an
        self._meta = None                        # earlier level did to the brain (a held command, a flight) never leaks into the next
        self._snap_brain = None
        self._take_snapshot()
        game.puppet_active = True                # nothing of yours may touch the fly (Game.hit), and the laser is the only tool
        game.note_hooks.append(self._on_note)
        self._lock_tool()

    def _take_snapshot(self) -> None:
        from kickthefly.core import savestate

        br = self.slot.brain
        self._snap = {}
        with br.step_lock:
            self._meta = savestate.brain_state(br, "p_", self._snap)
        self._snap_brain = br

    def _restore_snapshot(self) -> None:
        from kickthefly.core import savestate

        br = self.slot.brain
        if self._meta is None or br is not self._snap_brain:
            self._take_snapshot()
            return
        with br.step_lock:
            savestate.restore_brain(br, self._meta, self._snap, "p_")

    # --- the game's hooks ------------------------------------------------------------------------------------------------------
    @property
    def overlay(self) -> bool:                   # the level list and the result card take the mouse; playing leaves it to the laser
        return self.state in ("select", "won")

    @property
    def speed(self) -> float:
        return 1.0

    def _lock_tool(self) -> None:
        from kickthefly.game.kick_the_fly import TOOL_NAMES

        self.game.tool = TOOL_NAMES.index("laser")
        self.game.torching = False

    def end(self) -> None:
        g = self.game
        g.puppet_active = False
        if self._on_note in g.note_hooks:
            g.note_hooks.remove(self._on_note)
        for slot in g.flies:
            slot.brain.clear_current("puppet")
        try:
            g.laser_state.set_target("dnp01")
        except Exception:
            pass
        super().end()

    def _on_note(self, text: str) -> None:
        if self.state != "play" or self.tracker is None:
            return
        name = event_name(text)
        if name:
            self.tracker.event(name, self.game.clock.now)

    # --- starting a level --------------------------------------------------------------------------------------------------------
    def begin(self, level: pz.Level) -> None:
        g, slot = self.game, self.slot
        self.level, self.state = level, "play"
        self.tracker = pz.Tracker(level.goal)
        self.actions = self.hints = 0
        self.hint_shown, self.result, self.msg = False, None, ""
        self.latched = {}
        place_fly(g, slot, level)
        self._restore_snapshot()
        g.type_ops = {}
        g.surgery_modes = [0] * len(g.surgery_modes)
        g._apply_surgery()
        self.target = None
        g.laser_state.set_target(NO_TARGET)
        g.laser_state.set_mode("activate")
        g.laser_state.set_trigger_mode("hold")
        self.effect, self.how = "activate", "laser"
        self._lock_tool()
        self.tracker.start(g.clock.now)
        self._sig = self._signature()
        self._was_firing = False
        self._loom_until = 0.0                    # a shadow still being sent from the last level must not follow you into this one
        g.laser_state.trigger_release()
        g.note(f"PUPPETEER {level.title}: {level.brief}")

    def _signature(self):
        g = self.game
        return (tuple(g.surgery_modes), tuple(sorted(g.type_ops.items())))

    def retry(self) -> None:
        if self.level is not None:
            self.begin(self.level)

    def next_level(self) -> None:
        if self.level is None:
            return
        i = pz.LEVELS.index(self.level)
        if i + 1 < len(pz.LEVELS):
            self.begin(pz.LEVELS[i + 1])
        else:
            self.state = "select"

    # --- every frame ------------------------------------------------------------------------------------------------------------
    def update(self, now: float) -> None:
        g = self.game
        if self.state != "play" or self.tracker is None:
            return
        self._lock_tool()
        slot = self.slot
        # the score counts what you do: a laser pulse fired at a chosen type (hit or miss), a latch or a surgery change
        ls = g.laser_state
        if ls.firing and not self._was_firing and ls.target_type != NO_TARGET:
            self.actions += 1
        self._was_firing = bool(ls.firing)
        sig = self._signature()
        if sig != self._sig:
            self.actions += 1
            self._sig = sig
        x, z, hx, hz = fly_state(g, slot)
        self.tracker.position(x, z, hx, hz, now, is_airborne(g, slot))
        if self.tracker.loom_due(now):
            self.tracker.loom_fired(now)
            self._loom_until = now + pz.LOOM_FOR_S
            g.note("SHADOW   a looming shadow")
        if now < self._loom_until:
            slot.brain.poke("loom", None, 1.0, recruit=0.6)
        self.tracker.tick(now)
        if self.tracker.message:
            self.msg, self.tracker.message = self.tracker.message, ""
        if self.tracker.done:
            self._win(now)

    def _win(self, now: float) -> None:
        total = self.actions + self.hints
        n = pz.stars_for(total, self.level.par)
        new_best = record_score(pz.score_key(self.level.id), total, "low")
        record_score("puppeteer", pz.completed_levels(load_scores()), "high")
        self.state = "won"
        self.result = dict(actions=self.actions, hints=self.hints, total=total, stars=n, best=new_best)
        for slot in self.game.flies:
            slot.brain.clear_current("puppet")
        self.game.sound.play("yum", 0.6)

    # --- the player's choices ------------------------------------------------------------------------------------------------------------
    def pick_chip(self, i: int) -> bool:
        """A palette type (hotbar keys 1-9, 0 choose the first ten). In latch mode this also latches or releases it."""
        if self.state != "play" or not 0 <= i < len(pz.PALETTE):
            return False
        label, spec = pz.PALETTE[i]
        self.target = spec
        self.game.laser_state.set_target(spec)
        if self.how == "latch":
            self._toggle_latch(spec)
        self.game.sound.play("click")
        return True

    def _toggle_latch(self, spec: str) -> None:
        g = self.game
        slot = self.slot
        rows = g.laser_state.resolve_target_rows(slot.brain)
        cur = self.latched.get(spec)
        if cur == self.effect:
            self.latched.pop(spec)
        else:
            self.latched[spec] = self.effect
        self.actions += 1
        self._apply_latches()

    def _apply_latches(self) -> None:
        from kickthefly.lab.laser import BASE_SILENCE_CURRENT, BASE_STIM_CURRENT

        slot = self.slot
        arr = np.zeros(slot.brain.n, np.float32)
        ls = self.game.laser_state
        saved = ls.target_type
        for spec, eff in self.latched.items():
            ls.target_type = spec
            rows = ls.resolve_target_rows(slot.brain)
            arr[rows] += BASE_STIM_CURRENT if eff == "activate" else BASE_SILENCE_CURRENT
        ls.target_type = saved
        if np.any(arr):
            slot.brain.set_current("puppet", np.flatnonzero(arr), arr[arr != 0])
        else:
            slot.brain.clear_current("puppet")

    def set_effect(self, effect: str) -> None:
        self.effect = effect
        self.game.laser_state.set_mode(effect)

    def set_how(self, how: str) -> None:
        self.how = how

    def show_hint(self) -> None:
        if not self.hint_shown and self.level is not None:
            self.hint_shown = True
            self.hints += 1

    def solution_text(self) -> str:
        parts = []
        for a in self.level.solution:
            verb = {"activate": "activate", "silence": "silence"}[a["mode"]]
            parts.append(f"{a['how']}: {verb} {pz.LABEL_OF.get(a['target'], a['target'])}")
        return "; ".join(parts)

    # --- clicks -----------------------------------------------------------------------------------------------------------------------------
    def click(self, pos) -> bool:
        for b in self.buttons:
            if b.enabled and b.rect.collidepoint(pos):
                self.game.sound.play("click")
                b.action()
                return True
        return self.overlay

    # --- drawing ----------------------------------------------------------------------------------------------------------------------------
    def draw(self, surf, now, mouse) -> None:
        from kickthefly.game import kick_the_fly as k2

        g = self.game
        self.buttons = []
        if self.state == "select":
            self._draw_select(surf, mouse)
            return
        lv = self.level
        if self.state == "won":                           # the result card alone: its buttons are the only ones to click
            self._draw_won(surf, mouse)
            for b in self.buttons:
                b.draw(g, surf, mouse)
            return
        box = pygame.Rect(258, 58, 400, 156)
        card = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 225), card.get_rect(), border_radius=10)
        surf.blit(card, box)
        g._text(surf, lv.title.upper(), (box.x + 12, box.y + 8), AMBER, g.f_bold)
        g._text(surf, "PUPPETEER", (box.x + 12 + g.f_bold.size(lv.title.upper())[0] + 12, box.y + 11), DIM, g.f_small)
        self.buttons += [Button((box.right - 150, box.y + 6, 66, 24), "Levels", lambda: setattr(self, "state", "select")),
                         Button((box.right - 78, box.y + 6, 66, 24), "Quit", self.end)]
        y = self._para(surf, lv.brief, box.x + 12, box.y + 34, box.w - 24)
        prog = self.tracker.describe() if self.tracker else ""
        total = self.actions + self.hints
        g._text(surf, f"actions {total}   par {lv.par}" + (f"   {prog}" if prog else ""), (box.x + 12, box.y + 62), INK, g.f_text)
        if self.msg:
            g._text(surf, self.msg, (box.x + 12, box.y + 84), ACCENT, g.f_small)
        self.buttons.append(Button((box.x + 12, box.y + 106, 70, 24), "Retry", self.retry))
        self.buttons.append(Button((box.x + 90, box.y + 106, 70, 24), "Hint", self.show_hint, enabled=not self.hint_shown))
        if self.hint_shown:
            self._para(surf, f"{lv.hint} ({self.solution_text()})", box.x + 12, box.y + 134, box.w - 24)
        self._draw_goal_marker(surf)
        self._draw_palette(surf, mouse)
        for b in self.buttons:
            b.draw(g, surf, mouse)

    def _draw_palette(self, surf, mouse) -> None:
        from kickthefly.game import kick_the_fly as k2

        g = self.game
        W = k2.PLAY_W
        y0 = k2.H - 118 if k2.H > 400 else 300
        panel = pygame.Rect(12, y0, W - 24, 112)
        card = pygame.Surface(panel.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 215), card.get_rect(), border_radius=10)
        surf.blit(card, panel)
        g._text(surf, "WHICH NEURONS? pick a type (keys 1-9, 0), then aim the laser at the fly and hold the button", (panel.x + 12, panel.y + 6), LABEL, g.f_small)
        x, y = panel.x + 12, panel.y + 26
        for i, (label, spec) in enumerate(pz.PALETTE):
            w = max(70, g.f_small.size(label)[0] + 30)
            if x + w > panel.right - 8:
                x, y = panel.x + 12, y + 28
            sel = spec == self.target
            on = spec in self.latched
            b = Button((x, y, w, 24), f"{(i + 1) % 10 if i < 10 else ''} {label}".strip(), (lambda k=i: self.pick_chip(k)), style="primary" if sel else "normal")
            self.buttons.append(b)
            if on:
                pygame.draw.rect(surf, AMBER if self.latched[spec] == "activate" else (90, 170, 255), (x, y + 25, w, 3), border_radius=2)
            x += w + 6
        bx = panel.right - 330
        by = panel.bottom - 32
        self.buttons += [Button((bx, by, 100, 24), "Activate", lambda: self.set_effect("activate"), style="primary" if self.effect == "activate" else "normal"),
                         Button((bx + 106, by, 100, 24), "Silence", lambda: self.set_effect("silence"), style="primary" if self.effect == "silence" else "normal"),
                         Button((bx + 220, by, 54, 24), "Laser", lambda: self.set_how("laser"), style="primary" if self.how == "laser" else "normal"),
                         Button((bx + 278, by, 52, 24), "Latch", lambda: self.set_how("latch"), style="primary" if self.how == "latch" else "normal")]

    def _draw_goal_marker(self, surf) -> None:
        from kickthefly.game import kick_the_fly as k2

        t = self.tracker
        if t is None or self.game.three_d or t.cur["kind"] != "travel" or t.origin is None:
            return
        ox, oz, ux, uz = t.origin
        gx = k2.PLAY_W / 2 + (ox + ux * t.cur["m"]) / S
        fy = k2.FLOOR
        pygame.draw.line(surf, (255, 220, 120), (gx, fy), (gx, fy - 70), 3)
        pygame.draw.polygon(surf, (255, 190, 60), [(gx, fy - 70), (gx + 28, fy - 60), (gx, fy - 50)])
        pygame.draw.ellipse(surf, (255, 220, 120), (gx - 24, fy - 5, 48, 10), 2)

    def draw_world3d(self, rd, now: float) -> None:
        """The goal post in the 3D room."""
        from kickthefly.game.render3d import P_NONE, segment, trs

        t = self.tracker
        if t is None or t.cur["kind"] != "travel" or t.origin is None:
            return
        ox, oz, ux, uz = t.origin
        gx, gz = ox + ux * t.cur["m"], oz + uz * t.cur["m"]
        rd.add("cylinder", segment((gx, 0.0, gz), (gx, 0.7, gz), 0.012), (1.0, 0.85, 0.4), P_NONE)
        rd.add("sphere", trs(np.array([gx, 0.72, gz]), None, (0.05, 0.05, 0.05)), (1.0, 0.75, 0.2), P_NONE)
        rd.add("sphere", trs(np.array([gx, 0.004, gz]), None, (0.22, 0.004, 0.22)), (1.0, 0.85, 0.4, 0.6), P_NONE)

    def _draw_won(self, surf, mouse) -> None:
        from kickthefly.game import kick_the_fly as k2

        g, lv, r = self.game, self.level, self.result
        w, h = 560, 360
        box = pygame.Rect(k2.PLAY_W // 2 - w // 2, 110, w, h)
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 240), card.get_rect(), border_radius=14)
        surf.blit(card, box)
        g._text(surf, f"{lv.title.upper()}: DONE", (box.x + 18, box.y + 12), GOOD, g.f_head)
        draw_stars(surf, (box.right - 90, box.y + 28), r["stars"], 12)
        g._text(surf, f"{r['actions']} actions + {r['hints']} hint(s) = {r['total']}   par {lv.par}" + ("   new best!" if r["best"] else ""),
                (box.x + 18, box.y + 50), INK, g.f_text)
        y = box.y + 82
        for tag, text in lv.why:
            col = (60, 170, 220) if tag == "CONNECTOME" else (220, 150, 50)
            g._text(surf, tag, (box.x + 18, y), col, g.f_small)
            y = self._para(surf, text, box.x + 130, y, w - 150)
            y += 6
        nxt = pz.LEVELS.index(lv) + 1 < len(pz.LEVELS)
        self.buttons += [Button((box.x + 18, box.bottom - 46, 110, 32), "Retry", self.retry),
                         Button((box.x + 136, box.bottom - 46, 110, 32), "Levels", lambda: setattr(self, "state", "select")),
                         Button((box.right - 150, box.bottom - 46, 132, 32), "Next level" if nxt else "All done", self.next_level, style="primary")]

    def _para(self, surf, text: str, x: int, y: int, width: int) -> int:
        g = self.game
        line = ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if g.f_small.size(trial)[0] > width and line:
                g._text(surf, line, (x, y), TEXT, g.f_small)
                y += g.f_small.get_linesize()
                line = word
            else:
                line = trial
        if line:
            g._text(surf, line, (x, y), TEXT, g.f_small)
            y += g.f_small.get_linesize()
        return y

    def _draw_select(self, surf, mouse) -> None:
        from kickthefly.game import kick_the_fly as k2

        g = self.game
        scores = load_scores()
        w, h = 700, 54 + 44 * len(pz.LEVELS) + 56
        box = pygame.Rect(k2.PLAY_W // 2 - w // 2, 40, w, h)
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (10, 12, 18, 240), card.get_rect(), border_radius=14)
        surf.blit(card, box)
        g._text(surf, "PUPPETEER", (box.x + 18, box.y + 10), AMBER, g.f_head)
        g._text(surf, "You cannot touch the fly. Steer it only with the laser and by switching real neurons on and off.", (box.x + 18, box.y + 40), LABEL, g.f_small)
        y = box.y + 66
        for lv in pz.LEVELS:
            best = scores.get(pz.score_key(lv.id))
            row = pygame.Rect(box.x + 14, y, w - 28, 38)
            pygame.draw.rect(surf, (28, 32, 42), row, border_radius=8)
            g._text(surf, lv.title, (row.x + 12, row.y + 4), INK, g.f_bold)
            g._text(surf, lv.brief, (row.x + 12, row.y + 21), LABEL, g.f_small)
            if best is not None:
                draw_stars(surf, (row.right - 190, row.centery), pz.stars_for(int(best), lv.par), 8)
                g._text(surf, f"best {int(best)} / par {lv.par}", (row.right - 140, row.y + 11), AMBER, g.f_small)
            else:
                g._text(surf, f"par {lv.par}", (row.right - 140, row.y + 11), DIM, g.f_small)
            self.buttons.append(Button((row.right - 66, row.y + 4, 58, 30), "Play", (lambda L=lv: self.begin(L)), style="primary"))
            y += 44
        self.buttons.append(Button((box.right - 98, box.bottom - 42, 84, 30), "Quit", self.end))
        for b in self.buttons:
            b.draw(g, surf, mouse)
