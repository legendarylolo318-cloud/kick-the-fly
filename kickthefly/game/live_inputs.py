"""Microphone and Streamer mode in the running game (3.0 day 3): the switches, the per-frame tick, the on-screen indicators, the
vote tally, and the menu page (Esc > Mic and streamer).

Both features are OPT-IN and OFF AT EVERY LAUNCH: the switches here live in memory only, never in config.toml, so a mic or a
network connection can't switch itself on from an old session. What is stored (Settings > Brain) is only tuning that does nothing
by itself: the microphone's sensitivity, which chat commands count, the vote window, the cooldown and the minimum votes.

While either is on, a red pill says so on screen, in 2D and in 3D, and the pause menu shows it too:
  MIC ON       the microphone is being analysed (nothing is saved or sent)
  TWITCH CHAT  host, port and channel of the connection (read-only, anonymous)

What is what: see core/mic.py (microphone -> JO-A/B: CONNECTOME neurons, GAME RULE transduction) and core/streamer.py (every
rule is a GAME RULE). This file adds only wiring: it applies the microphone's current to every fly's JO-A/B neurons, runs the
winning vote through the same game actions a player uses (select a tool, change the arena, toggle a surgery), and draws.
"""
from __future__ import annotations

import time

import numpy as np
import pygame

from kickthefly.core import mic as micmod
from kickthefly.core import netguard, streamer

RED = (235, 70, 70)
PILL_TOP = 90                       # the red pills' top: below the video REC badge and the "saved" line
PILL_MIN_X = 256                    # never left of this: the status card (x 10, 236 wide) and the panels under it
PILL_PAD = 36                       # the dot and the margins
STALE_S = 0.5                       # no new sound for this long and the microphone's drive goes to zero (GAME RULE)


class LiveInputs:
    def __init__(self, game):
        self.g = game
        self.mic = micmod.Mic()
        self.mic_on = False
        self.mic_error = ""
        self.reading = micmod.Reading()
        self._last_t = -1.0
        self._rows: dict = {}
        self.chat: streamer.TwitchChat | None = None
        self.chat_factory = streamer.TwitchChat            # the playthrough bot and the tests swap in a fake connection
        self.stream_on = False
        self.stream_error = ""
        self.channel = ""
        self.board = streamer.VoteBoard()
        self.history: list[str] = []
        self._opts_at = 0.0
        self.jo_firing = (0.0, 0.0, 0, 0)            # (JO-A Hz, JO-B Hz, JO-A neurons firing, JO-B neurons firing)
        import atexit
        atexit.register(self.stop_all)                # the microphone and the connection never outlive the game

    # --- the switches (session only) ---------------------------------------------------------------------------------
    def set_mic(self, on: bool) -> bool:
        """Turn the microphone on or off. Returns whether it is on afterwards; on failure says why, in the menu."""
        if on == self.mic_on:
            return self.mic_on
        if not on:
            self.mic.stop()
            for slot in self.g.flies:
                micmod.release(slot.brain)
            self.mic_on, self.reading, self.mic_error = False, micmod.Reading(), ""
            self.g.note("MIC      off")
            return False
        ok, why = micmod.availability()
        if not ok:
            self.mic_error = why
            self._flash(why)
            return False
        self.mic.set_sensitivity(float(self.g.cfg.get("brain.mic_sensitivity", 1.0)))
        try:
            self.mic.start()
        except micmod.MicError as e:
            self.mic_error = str(e)
            self._flash(str(e))
            return False
        self.mic_on, self.mic_error = True, ""
        self.g.note("MIC      on: sound drives the Johnston's organ JO-A/B neurons [GAME RULE]; nothing is saved or sent", source="rule")
        return True

    def set_stream(self, on: bool, channel: str | None = None) -> bool:
        if on == self.stream_on:
            return self.stream_on
        if not on:
            if self.chat is not None:
                self.chat.stop()
            self.chat, self.stream_on, self.stream_error = None, False, ""
            self.board.reset()
            self.g.note("STREAM   off")
            return False
        try:
            chan = streamer.clean_channel(self.channel if channel is None else channel)
            self.channel = chan
            self.chat = self.chat_factory(chan)
            self.chat.start()
        except (streamer.StreamError, netguard.NetworkBlocked) as e:
            self.stream_error = str(e)
            self._flash(str(e))
            self.chat = None
            return False
        self.board.reset()
        self.stream_on, self.stream_error = True, ""
        self.g.note(f"STREAM   reading #{chan} on {streamer.HOST} (read-only, anonymous): viewers vote with !tool, !arena, !surgery",
                    source="rule")
        return True

    def stop_all(self) -> None:
        """Called when the game quits: close the microphone and the connection."""
        if self.mic_on:
            self.set_mic(False)
        if self.stream_on:
            self.set_stream(False)

    def _flash(self, text: str) -> None:
        menu = getattr(self.g, "menu", None)
        if menu is not None:
            from kickthefly.ui import menu as menu_ui

            menu.flash(text, menu_ui.AMBER)

    # --- every frame ---------------------------------------------------------------------------------------------------
    def tick(self) -> None:
        if self.mic_on:
            self._mic_tick()
        if self.stream_on:
            self._stream_tick()

    def _mic_tick(self) -> None:
        if not self.mic.active:
            self.set_mic(False)
            return
        r = self.mic.latest()
        now = time.monotonic()
        if r.t != self._last_t:
            self._fresh_at = now
        elif now - getattr(self, "_fresh_at", now) > STALE_S and (r.drive_a or r.drive_b):
            # 3.0 day 3 review: a device that stops delivering (unplugged, or a capture error) left its last sound on JO-A/B for
            # good. No new sound for STALE_S: the drive goes to zero; the switch and its pill stay as the player left them.
            r = micmod.Reading(t=r.t)
            self._rows_applied = None
        self.reading = r
        key = (r.t, r.drive_a, r.drive_b)
        if key == getattr(self, "_rows_applied", None) and all(id(s.brain) in self._rows for s in self.g.flies):
            return
        self._last_t, self._rows_applied = r.t, key
        for slot in self.g.flies:
            rows = self._rows.get(id(slot.brain))
            if rows is None:
                rows = self._rows[id(slot.brain)] = micmod.jo_rows(slot.brain)
            micmod.apply(slot.brain, r, rows)
        self._measure_jo()

    def _measure_jo(self) -> None:
        """How the focused fly's JO-A and JO-B neurons are firing right now (the hum test's readout)."""
        from kickthefly.core import neurodex as nd

        br = self.g.brain
        a, b = self._rows.get(id(br)) or micmod.jo_rows(br)
        if not len(a) and not len(b):
            self.jo_firing = (0.0, 0.0, 0, 0)
            return
        spikes, steps = nd.window_spikes(br.sim.activity)
        win = max(1, steps) * br.dt

        def one(rows):
            if not len(rows):
                return 0.0, 0
            mask = np.zeros(br.n, bool)
            mask[rows] = True
            return float(mask[spikes].sum()) / len(rows) / win, int(np.unique(spikes[mask[spikes]]).size)

        (ha, na), (hb, nb) = one(a), one(b)
        self.jo_firing = (ha, hb, na, nb)

    def options(self) -> dict:
        from kickthefly.game import kick_the_fly as k2

        g = self.g
        return {"tool": {streamer.slug(t): t.upper() for t in getattr(g, "loadout", None) and g.loadout.tools or ()},
                "arena": {a: a for a in k2.ARENAS if g.three_d or a not in k2.THREE_D_ONLY},
                "surgery": {streamer.slug(label): label.split(" (")[0] for label, _ in k2.SURGERY}}

    def rules(self) -> streamer.Rules:
        c = self.g.cfg
        allow = {n for n in streamer.COMMANDS if c.get(f"stream.allow_{n}", n == "tool")}
        return streamer.Rules(float(c.get("stream.window_s", streamer.WINDOW_S)), float(c.get("stream.cooldown_s", streamer.COOLDOWN_S)),
                              int(c.get("stream.min_votes", streamer.MIN_VOTES)), frozenset(allow))

    def _stream_tick(self) -> None:
        chat = self.chat
        if chat is None:
            self.set_stream(False)
            return
        if chat.state == "error":
            self.stream_error = chat.error
            self._flash(chat.error)
            self.set_stream(False)
            self.stream_error = chat.error
            return
        now = time.monotonic()
        if now - self._opts_at > 1.0:
            self._opts_at = now
            self.board.set_rules(self.rules())
            self.board.set_options(self.options())
        for user, text in chat.drain():
            self.board.submit(user, text, now)
        res = self.board.tick(now)
        if res is not None:
            self._execute(res)

    def _execute(self, res: streamer.Result) -> None:
        """The winning vote does what a player's click would, through the game's own actions."""
        from kickthefly.game import kick_the_fly as k2

        g = self.g
        if res.command not in self.rules().allow:                         # the streamer's allowlist as it is now, not when the vote opened
            return
        label = f"CHAT VOTED  {res.command} {res.label} ({res.votes} of {res.total})"
        if res.command == "tool":
            names = [t for t in g.loadout.tools if streamer.slug(t) == res.slug]
            if not names:
                return
            idx = next((i for i, t in enumerate(k2.TOOLS) if t[0] == names[0]), None)
            if idx is None:
                return
            g.select_tool(names[0])
        elif res.command == "arena":
            if res.slug not in k2.ARENAS or (not g.three_d and res.slug in k2.THREE_D_ONLY):
                return
            g.set_setting("brain.arena", res.slug, save=False)              # not saved: a vote is for now
        elif res.command == "surgery":
            k = next((i for i, (lab, _) in enumerate(k2.SURGERY) if streamer.slug(lab) == res.slug), None)
            if k is None:
                return
            g._set_surgery(k, 0 if g.surgery_modes[k] else -1)              # silence it, or put it back
        g.note(label, source="rule")
        head = g.flies[min(g.focus, len(g.flies) - 1)].fly.p[k2.HEAD]
        g.popup(head + ((0.0, 0.7, 0.0) if g.three_d else (0.0, -70.0)), label, (190, 170, 255), force=True)
        self.history = ([f"{res.command} {res.label}: {res.votes} of {res.total} votes"] + self.history)[:4]

    # --- on screen -----------------------------------------------------------------------------------------------------
    def indicators(self) -> list[str]:
        out = []
        if self.mic_on:
            out.append("MIC ON - listening (nothing is saved or sent)")
        if self.stream_on and self.chat is not None:
            st = {"connecting": "connecting to", "live": "reading"}.get(self.chat.state, self.chat.state)
            out.append(f"TWITCH CHAT - {st} {self.chat.host}:{self.chat.port} #{self.chat.channel} (read-only)")
        return out

    def draw(self, surf, font, play_w: int) -> None:
        """The red pills (always) and the vote tally (while streaming)."""
        # 3.0 day 3 review: below the REC badge and the "saved" line (both centred at y 62-86), right of the status card, the dot
        # drawn as a circle (the HUD's fallback font on Linux has no U+25CF and showed a missing-glyph box), and the text cut to fit
        y = PILL_TOP
        max_w = play_w - 14 - PILL_MIN_X
        for text in self.indicators():
            img = font.render(text, True, (255, 235, 235))
            while img.get_width() + PILL_PAD > max_w and len(text) > 12:   # a long channel name never runs over the status card
                text = text[:-4].rstrip() + "..."
                img = font.render(text, True, (255, 235, 235))
            r = pygame.Rect(0, y, img.get_width() + PILL_PAD, img.get_height() + 10)
            r.centerx = play_w // 2
            if r.x < PILL_MIN_X:
                r.x = PILL_MIN_X
            pygame.draw.rect(surf, (120, 20, 24), r, border_radius=r.h // 2)
            pygame.draw.rect(surf, RED, r, 2, border_radius=r.h // 2)
            pygame.draw.circle(surf, (255, 90, 90), (r.x + 15, r.centery), max(4, r.h // 5))
            surf.blit(img, (r.x + 27, r.y + 5))
            y = r.bottom + 6
        if self.stream_on:
            self._draw_tally(surf, font, play_w, y + 2)

    def _draw_tally(self, surf, font, play_w: int, top: int = 64) -> None:
        b = self.board
        rows = b.tally()[:5]
        lh = font.get_linesize() + 4                       # rows follow the font, so Larger text still fits
        w, x, y = max(300, int(font.size("tool SWATTER  9999")[0] * 1.5)), 0, top
        x = play_w - w - 12
        h = 44 + lh * max(1, len(rows)) + (lh - 2) * len(self.history[:2])
        pygame.draw.rect(surf, (10, 12, 18, 215) if surf.get_flags() & pygame.SRCALPHA else (10, 12, 18), (x, y, w, h), border_radius=10)
        pygame.draw.rect(surf, (150, 130, 255), (x, y, w, h), 1, border_radius=10)
        title = {"idle": "CHAT VOTE: !tool NAME", "open": "CHAT VOTE", "cooldown": "VOTE COOLDOWN"}[b.state]
        surf.blit(font.render(title, True, (225, 220, 255)), (x + 10, y + 8))
        if b.state != "idle":
            t = font.render(f"{b.remaining():.0f} s", True, (255, 220, 140))
            surf.blit(t, (x + w - t.get_width() - 10, y + 8))
        yy = y + 8 + lh + 4
        top = max([r[3] for r in rows] + [1])
        for command, _slug, label, n in rows:
            pygame.draw.rect(surf, (60, 52, 110), (x + 10, yy + 2, int((w - 20) * n / top), lh - 6), border_radius=4)
            surf.blit(font.render(f"{command} {label}", True, (240, 240, 255)), (x + 14, yy))
            c = font.render(str(n), True, (255, 255, 255))
            surf.blit(c, (x + w - c.get_width() - 14, yy))
            yy += lh
        if not rows:
            surf.blit(font.render("no votes yet" if b.state != "cooldown" else "no votes counted", True, (170, 170, 190)), (x + 14, yy))
            yy += lh
        for line in self.history[:2]:
            surf.blit(font.render("last: " + line, True, (170, 200, 170)), (x + 10, yy))
            yy += lh - 2


# --- the menu page --------------------------------------------------------------------------------------------------------------------
def page(m, surf, rect, mouse) -> None:
    from kickthefly.lab import labtoolkit
    from kickthefly.ui import menu as ui

    m._surf = surf
    host = m.host
    live = host.live
    cfg = host.cfg
    y = labtoolkit._title(m, surf, rect, "MICROPHONE AND STREAMER MODE",
                          "Two opt-in inputs. Both are off every time the game starts, both show a red indicator on screen while they "
                          "are on, and neither is ever on during validation, assays, protocols or tests.",
                          [("GAME RULE", "every rule here is a game rule"), ("CONNECTOME", "mic: the JO-A/B neurons it drives")])
    x = rect.x + 24
    # 3.0 day 3 review: the label column follows the font (with Larger text "Viewers may vote" ran into "!tool"), and the
    # streamer's status has its own line (it ran off the panel's right edge, even at the normal size)
    lw = max(150, max(m.f_text.size(t)[0] for t in ("Microphone", "Sensitivity", "Streamer mode", "Viewers may vote",
                                                    "Vote window")) + 16)
    c = x + lw
    # --- microphone -------------------------------------------------------------------------------------------------------------
    m.text(surf, "MICROPHONE  ->  JOHNSTON'S ORGAN (JO-A and JO-B)", (x, y), ui.INK, m.f_text)
    y += 28
    ok, why = micmod.availability()
    m.text(surf, "Microphone", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    m.toggle(surf, (c, y, 50, 28), live.mic_on, lambda v: live.set_mic(bool(v)), id="live_mic", enabled=ok or live.mic_on,
             tip="Off every time the game starts. Sound is analysed in memory and thrown away: nothing is recorded, saved or sent.")
    if live.mic_on:
        m.text(surf, "ON: listening. Nothing is saved or sent.", (c + 120, y + 14), (255, 140, 140), m.f_small, "midleft")
    else:
        m.text(surf, live.mic_error or why, (c + 120, y + 14), ui.BAD if live.mic_error or not ok else ui.LABEL, m.f_small, "midleft")
    y += 36
    m.text(surf, "Sensitivity", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (c, y, 240, 28), float(cfg.get("brain.mic_sensitivity", 1.0)), micmod.SENSITIVITY[0], micmod.SENSITIVITY[1], 0.1,
             "{:.1f}x", lambda v: (cfg.set("brain.mic_sensitivity", float(v)), live.mic.set_sensitivity(float(v))), lambda: None,
             id="live_sens", tip="How loud a band must be to drive its neurons fully (game rule). Saved; the switch is not.")
    y += 40
    r = live.reading
    box = pygame.Rect(x, y, rect.w - 48, 104)
    pygame.draw.rect(surf, (12, 14, 20), box, border_radius=8)
    big = f"{r.peak_hz:.0f} Hz" if r.peak_hz else "- Hz"
    m.text(surf, "Hum test", (box.x + 12, box.y + 8), ui.LABEL, m.f_small)
    m.text(surf, big, (box.x + 12, box.y + 30), ui.INK, m.f_head)
    bins = r.bins or (0.0,) * 24
    bw = (box.w - 380) / 24
    for i, v in enumerate(bins):
        h = int(8 + 70 * v)
        pygame.draw.rect(surf, ui.ACCENT, (box.x + 190 + i * bw, box.bottom - 10 - h, max(2, bw - 2), h))
    ha, hb, na, nb = live.jo_firing
    m.text(surf, f"JO-A drive {r.drive_a / micmod.MAX_CURRENT:4.0%}   {na} of 50 neurons firing", (box.right - 180, box.y + 12), ui.TEXT, m.f_small, "midtop")
    m.text(surf, f"JO-B drive {r.drive_b / micmod.MAX_CURRENT:4.0%}   {nb} of 88 neurons firing", (box.right - 180, box.y + 36), ui.TEXT, m.f_small, "midtop")
    m.text(surf, "B prefers below ~100 Hz, A higher (LITERATURE); the split is a game rule", (box.right - 180, box.y + 66), ui.LABEL, m.f_small, "midtop")
    y = box.bottom + 8
    m.button(surf, (x, y, max(330, m.f_text.size("Run the hum demo (no microphone)")[0] + 40), 34), "Run the hum demo (no microphone)", lambda: _open_demo(m), id="live_demo",
             tip="A synthetic 200 Hz hum, steady and in 35 ms pulses, through the same analysis, onto real JO-A/B: does it reach the courtship pathway?")
    y += 48
    # --- streamer mode ------------------------------------------------------------------------------------------------------------
    m.text(surf, "STREAMER MODE (Twitch chat, read-only)", (x, y), ui.INK, m.f_text)
    y += 28
    m.text(surf, "Streamer mode", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    nok, nwhy = netguard.allowed()
    m.toggle(surf, (c, y, 50, 28), live.stream_on, lambda v: live.set_stream(bool(v)), id="live_stream", enabled=nok or live.stream_on,
             tip="Off every time the game starts. Connects to irc.chat.twitch.tv (port 6697) anonymously and only reads: it never logs in, "
                 "never sends, never stores a token. The connection is shown on screen while it is open.")
    m.text(surf, "channel", (c + 130, y + 14), ui.TEXT, m.f_small, "midleft")
    m.text_field(surf, (c + 195, y, 220, 28), live.channel, lambda v: setattr(live, "channel", v.strip()), id="live_chan", limit=26,
                 tip="A Twitch channel name, for example  mychannel . Letters, digits and underscores only.")
    y += 34
    if live.stream_on and live.chat is not None:
        status, col = f"{live.chat.state}: {live.chat.describe()}   commands read: {live.chat.commands_read}", (255, 140, 140)
    else:
        status, col = live.stream_error or ("" if nok else nwhy), ui.BAD if live.stream_error else ui.LABEL
    if status:
        m.wrapped(surf, status, (c, y), rect.right - 24 - c, col, m.f_small, 2)
    y += 2 * m.f_small.get_linesize() + 4
    m.text(surf, "Viewers may vote", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    cx = c
    for name, default in (("tool", True), ("arena", False), ("surgery", False)):
        m.text(surf, "!" + name, (cx, y + 14), ui.TEXT, m.f_small, "midleft")
        m.toggle(surf, (cx + 90, y, 50, 28), bool(cfg.get(f"stream.allow_{name}", default)),
                 lambda v, n=name: cfg.set(f"stream.allow_{n}", bool(v)), id=f"live_allow_{name}",
                 tip=f"Whether viewers can change the {name} by voting. You choose; the default is tools only.")
        cx += 210
    y += 36
    m.text(surf, "Vote window", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (c, y, 160, 28), float(cfg.get("stream.window_s", 20.0)), 5, 120, 5, "{:.0f} s",
             lambda v: cfg.set("stream.window_s", float(v)), lambda: None, id="live_win")
    sx = c + 180
    m.text(surf, "cooldown", (sx, y + 14), ui.TEXT, m.f_small, "midleft")
    sx += max(80, m.f_small.size("cooldown")[0] + 14)
    m.slider(surf, (sx, y, 160, 28), float(cfg.get("stream.cooldown_s", 30.0)), 5, 300, 5, "{:.0f} s",
             lambda v: cfg.set("stream.cooldown_s", float(v)), lambda: None, id="live_cool")
    sx += 180
    m.text(surf, "min votes", (sx, y + 14), ui.TEXT, m.f_small, "midleft")
    sx += max(80, m.f_small.size("min votes")[0] + 14)
    m.slider(surf, (sx, y, 140, 28), float(cfg.get("stream.min_votes", 2)), 1, 50, 1, "{:.0f}",
             lambda v: cfg.set("stream.min_votes", int(v)), lambda: None, id="live_min")
    y += 38
    m.wrapped(surf, "Only chat lines that start with ! are read, for one frame, to be counted. A viewer's name is hashed with a random salt and "
                    "held only for the round (and the 2 s flood limit); the screen shows counts, never names. YouTube is not supported (it can't be done without storing a key).",
              (x, y), rect.w - 60, ui.LABEL, m.f_small, 3)
    labtoolkit._back(m, rect, "live")


def _open_demo(m) -> None:
    from kickthefly.lab import lab

    lab._state(m).kind = "hum_demo"
    m.show("lab_assays")
