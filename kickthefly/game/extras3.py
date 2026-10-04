"""The 3.0 features as seen by the running game (2D and 3D share this): Neurodex tracking and toasts, the kill cam, and
the Neuron of the Day card. One `Extras` per Game, created at the end of Game.__init__ and called from a few hooks:

    tick(now)              every frame, from update(): discovery checks, kill cam sampling and playback clock
    on_die(slot)           from Game._die: freezes the kill cam buffer into a replay offer
    draw(surf, now)        from Game.draw_science_card (both games call it): toasts, the kill cam, the launch card
    handle_event(ev)       first thing in handle / handle3d: keys, clicks and the pad-free parts of the three features
    view_rates()           what Game._view_loop renders while the kill cam plays (the recorded frame, not the live brain)
    panel_fast(br)         the brain panel's group rates while the kill cam plays

See kickthefly/core/neurodex.py, killcam.py and neuron_of_day.py for what is CONNECTOME, GAME RULE or LITERATURE; this
module only wires them to the screen, and draws the tags next to the things they tag.

Not done here on purpose: nothing in this module steps, drives or changes a brain. Discovery reads the simulator's rate
estimate, the kill cam reads frames it copied, and the card's Try it goes through the game's own surgery / laser.
"""
from __future__ import annotations

import datetime as _dt
import threading
import time

import numpy as np
import pygame

from kickthefly.core import killcam as kc
from kickthefly.core import neurodex as nd
from kickthefly.core import neuron_of_day as notd

TOAST_S = 4.0
TOAST_MAX = 3            # 3.0 release review: a burst of discoveries (a new arena, a strong stimulus) stacked dozens of toasts over the screen
NOTD_S = 14.0
OFFER_S = 12.0                     # how long the "kill cam ready" prompt stays when the autopsy isn't up
SPEEDS = (0.1, 0.25, 0.5, 1.0)


class Extras:
    def __init__(self, game):
        self.g = game
        self.async_build = True                      # tests build the Neurodex table in the calling thread
        # --- Neurodex
        self.progress: nd.Progress | None = None
        self.table: nd.TypeTable | None = None
        self.table_state = "idle"                    # idle | building | ready | none | error
        self.table_error = ""
        self.tracker: nd.Tracker | None = None
        self.toasts: list[tuple[str, float]] = []
        self._last_check: dict = {}
        self.selected: str | None = None             # the entry the panel shows
        # --- kill cam
        self.buf: kc.Buffer | None = None
        self.buf_brain = None
        self.offer: dict | None = None               # {"replay", "slot", "summary", "t", "cause"}
        self.player: kc.Player | None = None
        self.kc_speed = kc.SPEED
        self.kc_saving = False
        self.kc_rects: dict = {}
        self._kc_wall = time.perf_counter()
        self.kc_prompt_until = 0.0
        # --- Neuron of the Day
        self.notd: dict | None = None
        self.notd_t0 = 0.0
        self.notd_rects: dict = {}
        self.notd_note = ""
        self._notd_pending = False

    # ---------------------------------------------------------------------------------------------------------------------
    @property
    def brain_name(self) -> str:
        return "larva" if getattr(self.g, "is_larva", False) else "adult"

    @property
    def calm_fx(self) -> bool:
        return bool(self.g.cfg["access.reduced_flashing"])

    # --- Neurodex: table, progress, discovery -----------------------------------------------------------------------------------
    def ensure_progress(self) -> nd.Progress:
        if self.progress is None:
            self.progress = nd.Progress()
            for w in self.progress.warnings:
                from kickthefly.core.crash import log
                log.warning(w)
        return self.progress

    def ensure_table(self) -> None:
        """Start building the type table (a few seconds on the real pack) if nobody has. Never blocks the game."""
        if self.table_state != "idle":
            return
        self.table_state = "building"
        name = self.brain_name
        prog = self.ensure_progress()                    # review: made here, on the game thread, before the worker starts.
        # The worker used to call ensure_progress() itself, racing the game thread (opening the Neurodex, a discovery), so two
        # Progress objects could exist: the tracker recorded into one and the panel showed and saved the other.

        def work():
            try:
                t = nd.table(name)
                if t is None:
                    self.table_state = "none"
                    return
                self.table = t
                self.tracker = nd.Tracker(t, prog)
                self.table_state = "ready"
            except Exception as e:                      # the Neurodex is never worth a crash
                from kickthefly.core.crash import log
                log.exception("Neurodex table failed")
                self.table_error = f"{type(e).__name__}: {e}"
                self.table_state = "error"

        if self.async_build:
            threading.Thread(target=work, name="neurodex-table", daemon=True).start()
        else:
            work()

    def dex_tick(self) -> None:
        g = self.g
        if not g.cfg["brain.neurodex"]:
            return
        self.ensure_table()
        tr = self.tracker
        if tr is None:
            return
        br = g.brain                                    # the fly the brain panel shows
        if br.dead:
            return
        last = self._last_check.get(id(br))
        if last is not None and br.steps < last:
            last = None                                 # 3.0 day 2 decisions: the step counter went back (a new fly, a loaded
                                                        # save): without this no check ran until it caught up with the old count
        if last is not None and br.steps - last < nd.CHECK_STEPS:
            return
        if br.steps == last:
            return                                      # time is paused
        self._last_check[id(br)] = br.steps
        calm = (br.sedation == 0 and not br.surgery and not br.driving and br.steps - br.last_poke > 400)
        driven = bool(br.surgery or br.driving or getattr(g.laser_state, "firing", False))
        spikes, steps = nd.window_spikes(br.sim.activity)
        found = tr.observe(id(br), spikes, br.dt, calm, driven, window_steps=steps)
        for name in found:
            self.on_discovery(name, driven)

    def on_discovery(self, name: str, driven: bool) -> None:
        g = self.g
        prog = self.ensure_progress()
        rec = prog.types(self.brain_name).get(name, {})
        how = {"stimulated": " (by stimulation)", "rest": " (at rest)"}.get(rec.get("how"), "")
        self.toasts.append((f"NEURODEX  {name} discovered{how}", time.perf_counter()))
        g.note(f"NEURODEX  {name} discovered{how}  x{rec.get('x', 0):.1f} calm", source="rule")
        try:
            g.sound.play("click", 0.6)
        except Exception:
            pass
        prog.save()

    def discovered_count(self) -> tuple[int, int]:
        prog = self.ensure_progress()
        total = len(self.table) if self.table is not None else 0
        return prog.n_discovered(self.brain_name), total

    # --- kill cam: sampling, offer, playback -------------------------------------------------------------------------------------
    def killcam_enabled(self) -> bool:
        return bool(self.g.cfg["brain.killcam"])

    def kc_feed(self) -> None:
        g = self.g
        if not self.killcam_enabled() or self.player is not None:
            return
        br = g.brain
        if br.dead:
            return
        if self.buf is None or self.buf_brain is not br or self.buf.n != br.n:
            self.buf, self.buf_brain = kc.Buffer(br.n), br
        if self.buf.due(br.steps):                      # the rates are only read on the frames that store a sample
            self.buf.push(br.steps, br.sim.activity.rates(), {"fast": br.fast.copy()})

    def on_die(self, slot) -> None:
        """A fly died. If it is the one being sampled, keep its last seconds for the offer."""
        if self.buf is None or slot.brain is not self.buf_brain or not self.killcam_enabled():
            return
        rep = self.buf.freeze()
        self.buf.reset()
        if rep is None:
            return
        br = slot.brain
        self.offer = {"replay": rep, "slot": slot, "summary": rep.summary(br.types, br.superclass), "t": time.perf_counter(),
                      "base": br.base.copy(), "cause": getattr(self.g, "killed_by", None)}
        self.kc_prompt_until = time.perf_counter() + OFFER_S

    def kc_available(self) -> bool:
        return self.offer is not None and self.player is None

    def kc_start(self, record: bool = False) -> bool:
        if self.offer is None:
            return False
        speed = kc.SPEED_REDUCED if self.calm_fx else self.kc_speed
        self.player = kc.Player(self.offer["replay"], speed)
        self._kc_wall = time.perf_counter()
        self.kc_saving = False
        if self.g.menu.open:
            self.g.menu.close()
        if record:
            self.kc_save()
        return True

    def kc_stop(self) -> None:
        if self.kc_saving:
            self.g.toggle_video_recording()             # stop and write the file
            self.kc_saving = False
        self.player = None

    def kc_save(self) -> None:
        """Play it again from the start, recorded by the game's own recorder (MP4 with ffmpeg, else a GIF)."""
        if self.player is None or self.kc_saving or self.g.video_recording:
            return
        self.player.restart()
        self._kc_wall = time.perf_counter()
        self.g.toggle_video_recording()
        self.kc_saving = True

    def kc_playing(self) -> bool:
        return self.player is not None

    def kc_frame(self) -> int:
        return self.player.frame if self.player else 0

    def view_rates(self):
        """(rates per step, spiked) for the brain view while the kill cam plays, else None (use the live brain)."""
        p = self.player
        if p is None:
            return None
        return p.replay.rates_per_step(p.frame), np.zeros(0, np.int64)

    def panel_fast(self, br):
        """(fast, base) group vectors for the brain panel: the recorded ones while the kill cam plays, else live."""
        p = self.player
        if p is None or self.offer is None:
            return br.fast, br.base
        ex = p.replay.extras[p.frame]
        if not ex or "fast" not in ex:
            return br.fast, br.base
        return ex["fast"], self.offer["base"]

    # --- Neuron of the Day --------------------------------------------------------------------------------------------------------
    def maybe_show_notd(self) -> None:
        """Once per launch, from the run loop's first-run notices (never from Game.__init__, so tests and headless runs
        don't get a card)."""
        g = self.g
        if not g.cfg["brain.neuron_of_day"]:
            return
        if getattr(g, "tutorial", None) is not None or g.menu.open:
            return
        self.ensure_table()
        if self.async_build is False or self.table is not None:
            self._make_notd()
        else:
            self._notd_pending = True

    def _make_notd(self) -> None:
        self._notd_pending = False
        if self.table is None or self.notd is not None:
            return
        c = notd.card(_dt.date.today(), self.table, lab=self.g.cfg.lab)
        if c is not None:
            self.notd, self.notd_t0 = c, time.perf_counter()

    def notd_try(self) -> None:
        c = self.notd
        if c is None:
            return
        self.g.note(notd.apply(c, self.g), source="rule")
        self.notd = None

    def notd_turn_off(self) -> None:
        self.g.set_setting("brain.neuron_of_day", False)
        self.notd = None
        self.g.note("NEURON OF THE DAY off (Settings > Brain turns it back on)", source="rule")

    # --- per frame --------------------------------------------------------------------------------------------------------------------
    def tick(self, now: float) -> None:
        wall = time.perf_counter()
        if getattr(self, "_notd_pending", False) and self.table is not None:
            self._make_notd()
        if self.notd is not None and wall - self.notd_t0 > NOTD_S:
            self.notd = None
        self.toasts = [(t, t0) for t, t0 in self.toasts if wall - t0 < TOAST_S]
        if self.player is not None:
            self.player.advance(wall - self._kc_wall)
            if self.player.done and (not self.kc_saving or self.player.t >= self.player.replay.seconds):
                self.kc_stop()
        self._kc_wall = wall
        try:
            self.dex_tick()
            self.kc_feed()
            self.g.live.tick()
        except Exception:
            from kickthefly.core.crash import log
            log.exception("3.0 tick failed")

    # --- events -------------------------------------------------------------------------------------------------------------------------
    def handle_event(self, ev) -> bool:
        g = self.g
        if g.menu.open:
            return False
        if ev.type == pygame.KEYDOWN:
            name = pygame.key.name(ev.key)
            if self.player is not None:                 # the kill cam owns the keyboard while it plays
                if ev.key in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER) \
                        or "killcam" in g.cfg.actions_for(name):
                    self.kc_stop()
                elif ev.key == pygame.K_s:
                    self.kc_save()
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    self.kc_speed = SPEEDS[ev.key - pygame.K_1]
                    self.player.speed = self.kc_speed
                return True
            acts = g.cfg.actions_for(name)
            if "profiler" in acts:                       # 3.1.0 task 5: off at every launch
                from kickthefly.core.profiler import PROF
                g.note("PROFILER  on (F3 again to hide)" if PROF.toggle() else "PROFILER  off")
                return True
            if "killcam" in acts and self.kc_available():
                self.kc_start()
                return True
            if "neurodex" in acts and (not getattr(g, "three_d", False) or not getattr(g, "look", False)):
                self.open_neurodex()
                return True
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            pos = ev.pos
            if self.player is not None:
                for key, r in self.kc_rects.items():
                    if r.collidepoint(pos):
                        {"skip": self.kc_stop, "save": self.kc_save, "speed": self.kc_cycle_speed}[key]()
                        return True
                return True
            if self.notd is not None:
                for key, r in self.notd_rects.items():
                    if r.collidepoint(pos):
                        {"try": self.notd_try, "close": lambda: setattr(self, "notd", None), "off": self.notd_turn_off}[key]()
                        return True
        return False

    def kc_cycle_speed(self) -> None:
        i = SPEEDS.index(min(SPEEDS, key=lambda s: abs(s - self.kc_speed)))
        self.kc_speed = SPEEDS[(i + 1) % len(SPEEDS)]
        if self.player is not None:
            self.player.speed = self.kc_speed

    def open_neurodex(self) -> None:
        self.ensure_table()
        self.ensure_progress()
        self.g.open_menu("neurodex")

    def pad_event(self, down) -> bool:
        """Gamepad (3D): called once per frame with the buttons that went down. True if they were used."""
        if self.player is not None:
            # review: these are the pad's ACTION names (config.PAD_ACTIONS); "b"/"a" were button names and never matched, so
            # B (bound to crouch) did not skip it as the docs say. Start is handled before this (it sends Esc).
            if down & {"killcam", "crouch", "up", "use"}:
                self.kc_stop()
            return True
        if "killcam" in down and self.kc_available():
            self.kc_start()
            return True
        if "neurodex" in down:
            self.open_neurodex()
            return True
        return False

    # --- drawing -----------------------------------------------------------------------------------------------------------------------
    def draw_profiler(self, surf) -> None:
        """The frame profiler (core/profiler.py): FPS, the sections' milliseconds, a frame-time strip, the engine and the brain view."""
        from kickthefly.core.profiler import PROF
        g = self.g
        try:
            be = g.flies[0].brain.sim.backend
            backend = f"{be.name} ({be.device})"
        except Exception:
            backend = ""
        view = getattr(getattr(g, "view", None), "engine_note", "") or ("CPU (sparse matrices)" if getattr(g, "view", None) is not None else "")
        lines = PROF.lines(backend, view.split(" (")[0]) + ["GAME RULE: timings only; no neuron reads this"]
        f = g.f_small
        w = 440
        h = 12 + 20 * len(lines) + 54
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (8, 10, 16, 215), card.get_rect(), border_radius=8)
        pygame.draw.rect(card, (70, 80, 100, 180), card.get_rect(), 1, border_radius=8)
        for i, ln in enumerate(lines):
            card.blit(f.render(ln, True, (200, 235, 210) if i == 0 else (222, 228, 238)), (10, 8 + 20 * i))
        y0 = 12 + 20 * len(lines)                         # the last frames, one column each; the lines mark 16.7 ms and 33 ms
        fr = list(PROF.frames)[-(w - 20):]
        top = 40.0
        for ms, col in ((1000 / 60, (60, 140, 90)), (1000 / 30, (150, 120, 60))):
            yy = y0 + 44 - int(44 * min(ms, top) / top)
            pygame.draw.line(card, col, (10, yy), (w - 10, yy))
        for k, ms in enumerate(fr):
            hh = int(44 * min(ms, top) / top)
            pygame.draw.line(card, (110, 190, 255) if ms < 1000 / 30 else (255, 150, 90), (10 + k, y0 + 44), (10 + k, y0 + 44 - hh))
        surf.blit(card, (8, 8))

    def draw(self, surf, now: float) -> None:
        g = self.g
        wall = time.perf_counter()
        from kickthefly.core.profiler import PROF
        if PROF.on:
            self.draw_profiler(surf)
        if self.player is not None:
            self.draw_killcam(surf, wall)
            return
        y = 8
        shown = self.toasts[-TOAST_MAX:]                    # the newest few; the rest are in the reactions log and the Neurodex
        for text, t0 in shown:
            rise = 0.6 if self.calm_fx else 0.15             # Reduced flashing: a slower fade-in
            a = int(255 * min(1.0, (TOAST_S - (wall - t0)) / 0.6, (wall - t0) / rise + (0.0 if self.calm_fx else 0.2)))
            self._banner(surf, text, y, (90, 200, 120), a)
            y += 34
        more = len(self.toasts) - len(shown)
        if more > 0:
            from kickthefly.core.i18n import tr as _tr
            key = (g.cfg.keys.get("neurodex") or "").upper()
            self._banner(surf, _tr("+{n} more discovered (Neurodex: {key})", n=more, key=key or "Esc"), y, (90, 200, 120), 200)
            y += 34
        if self.offer is not None and g.report is None and wall < self.kc_prompt_until:
            key = (g.cfg.keys.get("killcam") or "").upper() or "(unbound: Settings > Controls)"
            self._banner(surf, f"KILL CAM ready: press {key} to watch its last {self.offer['replay'].seconds:.0f} s",
                         y, (255, 176, 64), 235)
            y += 34
        if self.notd is not None and g.report is None and not g.menu.open:
            self.draw_notd(surf, wall)

    def _banner(self, surf, text, y, col, alpha=255) -> None:
        g = self.g
        img = g.f_bold.render(text, True, (240, 243, 248))
        w = min(surf.get_width() - 20, img.get_width() + 28)
        cx = min(surf.get_width(), 890) // 2
        r = pygame.Rect(cx - w // 2, y, w, 28)
        alpha = max(0, min(255, int(alpha)))
        card = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (14, 18, 26, int(alpha * 0.9)), card.get_rect(), border_radius=10)
        pygame.draw.rect(card, (*col, alpha), card.get_rect(), 2, border_radius=10)
        surf.blit(card, r)
        surf.blit(img, (r.x + 14, r.y + 4))

    def draw_notd(self, surf, wall: float) -> None:
        g = self.g
        c = self.notd
        base = g.science_rect()
        left = max(base.x, 256)                          # clear of the HUD cards down the left side
        r = pygame.Rect(left, base.y - 172, min(base.w, 880 - left), 162)
        card = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (14, 20, 30, 240), card.get_rect(), border_radius=14)
        pygame.draw.rect(card, (86, 214, 255, 255), card.get_rect(), 2, border_radius=14)
        surf.blit(card, r)
        label = c["type"] + (f"  (+{c['also']} related types)" if c["also"] else "")
        g._text(surf, "NEURON OF THE DAY", (r.x + 16, r.y + 10), (86, 214, 255), g.f_small)
        g._text(surf, label, (r.x + 16, r.y + 26), (240, 243, 248), g.f_head)
        lines = self._wrap(c["text"], g.f_text, r.w - 32)[:3]
        for i, ln in enumerate(lines):
            g._text(surf, ln, (r.x + 16, r.y + 54 + i * 18), (205, 212, 224), g.f_text)
        g._text(surf, c["cite"], (r.x + 16, r.bottom - 50), (130, 142, 160), g.f_small)
        self.notd_rects = {}
        bx = r.x + 16
        p = c["plan"]
        if p["action"] != "none":
            look3d = getattr(g, "three_d", False) and getattr(g, "look", False)
            tr_ = pygame.Rect(bx, r.bottom - 30, 92, 24)
            col = (40, 110, 150) if not look3d else (44, 50, 64)
            pygame.draw.rect(surf, col, tr_, border_radius=8)
            g._text(surf, "Try it", tr_.center, (240, 243, 248), g.f_small, "center")
            self.notd_rects["try"] = tr_
            hint = "Tab frees the mouse to click" if look3d else p["text"].split(". ")[0] + "."
            room = r.right - 178 - (tr_.right + 10)
            while hint and g.f_small.size(hint)[0] > room:
                hint = hint[:-2]
            g._text(surf, hint, (tr_.right + 10, tr_.y + 5), (130, 142, 160), g.f_small)
        off = pygame.Rect(r.right - 168, r.bottom - 30, 152, 24)
        pygame.draw.rect(surf, (44, 50, 64), off, border_radius=8)
        g._text(surf, "Don't show again", off.center, (205, 212, 224), g.f_small, "center")
        self.notd_rects["off"] = off
        x = pygame.Rect(r.right - 30, r.y + 6, 24, 22)
        g._text(surf, "x", x.center, (130, 142, 160), g.f_bold, "center")
        self.notd_rects["close"] = x
        for tag, col in (("LITERATURE", (120, 200, 140)),):
            g._text(surf, tag, (r.right - 110, r.y + 12), col, g.f_small)

    @staticmethod
    def _wrap(text, font, width):
        lines, line = [], ""
        for w in text.split():
            trial = (line + " " + w).strip()
            if font.size(trial)[0] > width and line:
                lines.append(line)
                line = w
            else:
                line = trial
        return lines + [line]

    # kill cam overlay -----------------------------------------------------------------------------------------------------------------
    # Ring and trace colors. The rank number is drawn beside every ring, so color is never the only cue; the colorblind and
    # high-contrast palettes (Settings > Accessibility) switch to colors that stay apart without red versus green.
    PALETTES = {
        "default": ((255, 150, 60), (86, 214, 255), (130, 230, 130), (240, 110, 200), (250, 220, 90), (170, 140, 255)),
        "blue-yellow": ((255, 204, 20), (60, 120, 255), (255, 255, 255), (120, 170, 255), (255, 235, 130), (40, 70, 190)),
        "high-contrast": ((255, 31, 217), (255, 255, 255), (255, 31, 217), (255, 255, 255), (255, 31, 217), (255, 255, 255)),
    }

    @property
    def RISER_COLORS(self):
        return self.PALETTES.get(self.g.cfg["access.palette"], self.PALETTES["default"])

    def draw_killcam(self, surf, wall: float) -> None:
        g = self.g
        p, off = self.player, self.offer
        rep, summ = off["replay"], off["summary"]
        W = min(surf.get_width(), 890)
        H = surf.get_height()
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((6, 7, 10, 222))
        surf.blit(veil, (0, 0))
        card = pygame.Rect(28, 26, W - 56, min(H - 52, 690))
        pygame.draw.rect(surf, (20, 22, 28), card, border_radius=16)
        pygame.draw.rect(surf, (52, 60, 76), card, 1, border_radius=16)
        x, y = card.x + 24, card.y + 16
        g._text(surf, "KILL CAM", (x, y), (240, 243, 248), g.f_title)
        by = f"   killed by {off['cause']}" if off.get("cause") else ""
        g._text(surf, f"the last {rep.seconds:.1f} s before it died, at {p.speed:g}x{by}", (x + 2, y + 44),
                (130, 142, 160), g.f_text)
        g._text(surf, "CONNECTOME", (card.right - 130, y + 6), (60, 170, 220), g.f_small)
        g._text(surf, "each neuron's own firing, as the brain panel showed it", (card.right - 24, y + 22),
                (130, 142, 160), g.f_small, "topright")
        # timeline
        tl = pygame.Rect(x, y + 84, card.w - 48, 150)
        pygame.draw.rect(surf, (12, 14, 20), tl, border_radius=8)
        risers = summ["risers"]
        top = risers[:6]
        if top:
            tr = rep.traces([r["index"] for r in top])
            ymax = max(10.0, float(tr.max()) * 1.05)
            for k, (r, col) in enumerate(zip(top, self.RISER_COLORS)):
                pts = [(tl.x + 6 + i * (tl.w - 12) / max(1, rep.n_frames - 1),
                        tl.bottom - 6 - float(v) / ymax * (tl.h - 12)) for i, v in enumerate(tr[k])]
                if len(pts) > 1:
                    pygame.draw.lines(surf, col, False, pts, 2)
            g._text(surf, f"{ymax:.0f} spikes/s", (tl.x + 8, tl.y + 4), (130, 142, 160), g.f_small)
        else:
            g._text(surf, "no neuron rose by "
                    f"{kc.MIN_RISE_HZ:g} spikes/s or more before death", tl.center, (130, 142, 160), g.f_text, "center")
        px = tl.x + 6 + p.progress * (tl.w - 12)
        pygame.draw.line(surf, (240, 243, 248), (px, tl.y + 2), (px, tl.bottom - 2), 2)
        g._text(surf, "death", (tl.right - 6, tl.bottom + 4), (208, 59, 59), g.f_small, "topright")
        g._text(surf, f"-{rep.seconds - p.t:0.1f} s", (px, tl.bottom + 4), (205, 212, 224), g.f_small, "midtop")
        # the list
        ly = tl.bottom + 34
        g._text(surf, f"NEURONS THAT ROSE MOST (last {kc.TAIL_S:g} s vs the first {kc.BASE_S:g} s of the window)", (x, ly),
                (130, 142, 160), g.f_small)
        ly += 20
        for k, r in enumerate(risers):
            col = self.RISER_COLORS[k % len(self.RISER_COLORS)]
            pygame.draw.circle(surf, col, (x + 8, ly + 9), 5)
            g._text(surf, f"{k + 1:>2}  {r['type']}", (x + 22, ly), (240, 243, 248), g.f_text)
            g._text(surf, r.get("superclass", ""), (x + 250, ly), (130, 142, 160), g.f_small)
            g._text(surf, f"{r['before_hz']:5.1f} > {r['after_hz']:5.1f} spikes/s   (+{r['rise_hz']:.1f})",
                    (card.right - 24, ly), (205, 212, 224), g.f_text, "topright")
            ly += 22
            if ly > card.bottom - 92:
                break
        if not risers:
            g._text(surf, "Nothing rose that much: the drive that killed it came from outside the recorded window.",
                    (x, ly), (205, 212, 224), g.f_text)
        # buttons
        self.kc_rects = {}
        by_ = card.bottom - 52
        for key, label, w in (("skip", "SKIP  (Esc / Space)", 190), ("save", "SAVE VIDEO  (S)", 170),
                              ("speed", f"SPEED {p.speed:g}x  (1-4)", 170)):
            r = pygame.Rect(x + sum(w_ + 10 for k_, w_ in (("skip", 190), ("save", 170), ("speed", 170))[:["skip", "save", "speed"].index(key)]), by_, w, 38)
            pygame.draw.rect(surf, (44, 50, 64) if key != "skip" else (40, 110, 150), r, border_radius=10)
            if key == "save" and self.kc_saving:                 # review: "[recording]" ran past the button's edge
                label = "RECORDING..."
            g._text(surf, label, r.center, (240, 243, 248), g.f_small, "center")
            self.kc_rects[key] = r
        g._text(surf, "The highlighted neurons ring on the brain panel. Nothing here changes the fly.",
                (x, by_ - 22), (130, 142, 160), g.f_small)

    def draw_killcam_rings(self, surf, rect: pygame.Rect, key: str, now: float) -> None:
        """Rings on the brain view at the highlighted neurons' positions (called by the panel and big-view draw)."""
        if self.player is None or self.offer is None:
            return
        view = self.g.view
        pix = view.spark_pix.get(key)
        if pix is None:
            return
        w, _ = __import__("kickthefly.game.kick_the_fly", fromlist=["VIEW_SIZES"]).VIEW_SIZES[key]
        pulse = 0.0 if self.calm_fx else 1.5 * (1 + np.sin(now * 5.0))
        for k, r in enumerate(self.offer["summary"]["risers"]):
            p_ = int(pix[r["index"]])
            if p_ < 0:
                continue
            cx, cy = rect.x + p_ % w, rect.y + p_ // w
            col = self.RISER_COLORS[k % len(self.RISER_COLORS)]
            pygame.draw.circle(surf, col, (cx, cy), int(7 + pulse), 2)
            if key == "big" or k < 6:
                self.g._text(surf, str(k + 1), (cx + 9, cy - 9), col, self.g.f_small)
