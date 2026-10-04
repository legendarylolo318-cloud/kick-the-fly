"""Fly arcade page (3.0 day 4): the fly tournament and fly racing, in the pause menu (Esc > Fly arcade).

Everything heavy (the personality cards, the duels, the race) runs in a background job with a progress bar and a Cancel button, in
worker processes, never on the game thread. Nothing here uses the network or the microphone, and points are in-game points only:
no money, no purchase (core/points.py). The physics and the tags (CONNECTOME / GAME RULE / MODEL PREDICTION) are in game/flyduel.py,
game/flyrace.py, lab/tournament.py and lab/racing.py; the page repeats them where a number is shown.
"""
from __future__ import annotations

import math
import time

import pygame

from kickthefly.core.i18n import tr
from kickthefly.ui import menu as ui
from kickthefly.ui.bgjob import BgJob, draw_progress

SERIES = {"default": ((57, 135, 229), (217, 89, 38)), "blue-yellow": ((30, 100, 200), (240, 190, 20)),
          "high-contrast": ((255, 255, 255), (255, 200, 0))}


class ArcadeState:
    def __init__(self):
        self.tab = "tournament"
        self.size = 8
        self.base_seed = 2000
        self.match_s = 20.0
        self.favorite: int | None = None
        self.job: BgJob | None = None
        self.bracket: dict | None = None
        self.sel_match: tuple[int, int] | None = None
        self.replay_t = 0.0
        self.playing = True
        self.last = time.perf_counter()
        self.lanes = 5
        self.race_base = 3000
        self.field: list[dict] | None = None          # measured cards of the race field
        self.odds: list[dict] | None = None
        self.bet_fly: int | None = None
        self.stake = 10
        self.race: dict | None = None
        self.bet_result: list[dict] | None = None
        self.error = ""
        self.wallet = None


def state(m) -> ArcadeState:
    st = getattr(m.host, "_arcade_state", None)
    if st is None:
        st = m.host._arcade_state = ArcadeState()
    return st


def _colors(m):
    return SERIES.get(m.host.cfg["access.palette"], SERIES["default"])


def _workers() -> int:
    from kickthefly.lab import labjobs

    return max(1, min(2, labjobs.default_workers()))        # each worker holds a brain (about 1 GB): two keeps a small machine usable


def _backend(m):
    return None


# --- jobs --------------------------------------------------------------------------------------------------------------------------
def _start_tournament(m, st: ArcadeState) -> None:
    from kickthefly.lab import tournament

    seeds = [st.base_seed + i for i in range(st.size)]
    mode = _mode(m)

    def work(job: BgJob):
        return tournament.run_bracket(seeds, seconds=st.match_s, mode=mode, favorite=st.favorite if st.favorite in seeds else None,
                                      workers=_workers(), cancel=job.cancel,
                                      progress=lambda d, n, label: job.update(d / max(1, n), label))

    st.bracket, st.sel_match, st.error = None, None, ""
    st.job = BgJob("Tournament", work).start()


MEASURE_PLAY = "Measuring the flies in play"


def _play_slots(m) -> list:
    return [sl for sl in (getattr(m.host, "flies", None) or []) if hasattr(sl, "card_settings")]


def _start_measure_play(m, st: ArcadeState) -> None:
    """Measure the personality cards of the flies in play (3.0 day 4 review, decision A): each with the individuality mode, sigma
    and LIF parameters its brain really runs with, in worker processes; the cards go into the card cache and the flies show them."""
    from kickthefly.lab import tournament

    groups: dict = {}
    for sl in _play_slots(m):
        mode, sigma, params = sl.card_settings()
        groups.setdefault((mode, sigma, tuple(sorted(params.items()))), []).append(int(sl.seed))

    def work(job: BgJob):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        out, total = {}, sum(len(v) for v in groups.values())
        pool = ProcessPoolExecutor(max_workers=_workers(), mp_context=multiprocessing.get_context("spawn"))  # not the game's process
        try:
            for (mode, sigma, params), seeds in groups.items():
                got = tournament.measure_cards(seeds, mode, None, 2, pool,
                                               lambda d, n, label: job.update((len(out) + d) / max(1, total), label), job.cancel,
                                               sigma=sigma, params=dict(params))
                out.update(got)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return out

    st.error = ""
    st.job = BgJob(MEASURE_PLAY, work).start()


def _mode(m) -> str:
    mode = str(m.host.cfg["brain.individuality"]) if hasattr(m.host, "cfg") else "subtle"
    return mode if mode in ("off", "subtle", "strong") else "subtle"


def _start_field(m, st: ArcadeState) -> None:
    from kickthefly.lab import racing

    seeds = [st.race_base + i for i in range(st.lanes)]
    mode = _mode(m)

    def work(job: BgJob):
        cards = racing.measure_field(seeds, mode, None, lambda d, n, label: job.update(d / max(1, n), label), job.cancel, workers=_workers())
        return cards

    st.race, st.bet_result, st.error, st.bet_fly = None, None, "", None
    st.field, st.odds = None, None
    st.job = BgJob("Looking at the field", work).start()


def _start_race(m, st: ArcadeState) -> None:
    from kickthefly.lab import racing

    seeds = [c["seed"] for c in st.field]
    mode = _mode(m)
    cards = st.field

    def work(job: BgJob):
        return racing.run_race(seeds, mode=mode, repeats=1, workers=_workers(), race_seed=st.race_base, card_list=cards, cancel=job.cancel,
                               progress=lambda d, n, label: job.update(d / max(1, n), label))

    st.race, st.bet_result, st.error = None, None, ""
    st.job = BgJob("Race", work).start()


def _collect(m, st: ArcadeState) -> None:
    """Take a finished job's result into the page's state (on the game thread, in the page's draw)."""
    job = st.job
    if job is None or job.running:
        return
    st.job = None
    if job.error:
        st.error = job.error
        return
    if job.cancelled or job.result is None:
        return
    if job.label == "Tournament":
        st.bracket = job.result
        st.replay_t, st.playing = 0.0, True
        champion = job.result["champion"]
        st.sel_match = (len(job.result["rounds"]) - 1, 0)
        if st.favorite is not None:
            m.flash(tr("Your favorite won the tournament!") if champion == st.favorite else tr("Your favorite lost. Fly {n} won.").format(n=champion),
                    ui.GOOD if champion == st.favorite else ui.AMBER)
    elif job.label == MEASURE_PLAY:
        for sl in _play_slots(m):
            sl.refresh_card()
        m.flash(tr("Measured {n} personality card(s): the flies in play show them now.").format(n=len(job.result)), ui.GOOD)
    elif job.label == "Looking at the field":
        from kickthefly.lab import racing

        st.field = job.result
        st.odds = racing.odds_table(job.result)
    elif job.label == "Race":
        from kickthefly.core import points
        from kickthefly.lab import racing

        st.race = job.result
        st.replay_t, st.playing = 0.0, True
        if st.bet_fly is not None:
            if st.wallet is None:
                st.wallet = points.Wallet()
            try:
                st.bet_result = racing.settle_bets(job.result, [dict(fly=st.bet_fly, stake=st.stake)], st.wallet)
            except ValueError as e:
                st.error = str(e)


# --- drawing helpers ----------------------------------------------------------------------------------------------------------------
def _tag(m, surf, x, y, name: str) -> int:
    col = ui.TAG_COLORS.get(name, (150, 120, 220))
    img = m.f_small.render(name, True, (10, 12, 16))
    r = pygame.Rect(x, y, img.get_width() + 12, img.get_height() + 4)
    pygame.draw.rect(surf, col, r, border_radius=5)
    surf.blit(img, (r.x + 6, r.y + 2))
    return r.right + 6


def _draw_duel(m, surf, rect: pygame.Rect, frames: list, t: float, names: tuple[str, str], cols) -> None:
    pygame.draw.rect(surf, (12, 15, 20), rect, border_radius=8)
    pygame.draw.rect(surf, ui.BORDER, rect, 1, border_radius=8)
    if not frames:
        return
    i = min(len(frames) - 1, max(0, int(t * 10)))
    fr = frames[i]
    half = 6.0
    px = lambda x, y: (int(rect.centerx + x / half * (rect.w / 2 - 14)), int(rect.centery - y / half * (rect.h / 2 - 14)))   # noqa: E731
    for k, (x, y, yaw, hp) in enumerate(((fr[1], fr[2], fr[3], fr[4]), (fr[5], fr[6], fr[7], fr[8]))):
        c = px(x, y)
        alive = hp > 0
        pygame.draw.circle(surf, cols[k] if alive else (70, 70, 70), c, 11)
        tip = (c[0] + int(math.cos(yaw) * 20), c[1] - int(math.sin(yaw) * 20))
        pygame.draw.line(surf, (240, 240, 240), c, tip, 3)
        bar = pygame.Rect(rect.x + 10 + k * (rect.w // 2), rect.y + 8, rect.w // 2 - 20, 8)
        pygame.draw.rect(surf, (40, 44, 54), bar, border_radius=4)
        pygame.draw.rect(surf, cols[k], (bar.x, bar.y, int(bar.w * hp / 100.0), bar.h), border_radius=4)
        m.text(surf, f"{names[k]}  {hp:.0f}", (bar.x, bar.bottom + 2), ui.TEXT, m.f_small)
    for (x, y, owner) in fr[9]:
        pygame.draw.circle(surf, cols[owner], px(x, y), 3)


def _advance(st: ArcadeState, length: float) -> None:
    now = time.perf_counter()
    if st.playing:
        st.replay_t = (st.replay_t + (now - st.last)) % max(0.5, length)
    st.last = now


def _bar(m, surf, y, rect, st):
    if st.job is not None and st.job.running:
        draw_progress(m, surf, pygame.Rect(rect.x + 24, y, rect.w - 48, 12), st.job, ui)
        return True
    return False


# --- the page ---------------------------------------------------------------------------------------------------------------------
def page(m: ui.Menu, surf, rect, mouse) -> None:
    st = state(m)
    _collect(m, st)
    m.text(surf, tr("FLY ARCADE"), (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    from kickthefly.core import points

    if st.wallet is None:
        st.wallet = points.Wallet()
    m.text(surf, tr("Points: {n}  (game points only: no money, nothing to buy)").format(n=st.wallet.points), (rect.right - 24, rect.y + 22),
           ui.AMBER, m.f_small, "topright")
    x = rect.x + 24
    for key, label in (("tournament", tr("Tournament")), ("race", tr("Racing"))):
        m.button(surf, (x, rect.y + 54, 150, 32), label, (lambda k=key: setattr(st, "tab", k)), id=("arcade_tab", key), active=st.tab == key)
        x += 158
    slots = _play_slots(m)
    busy = st.job is not None and st.job.running
    unmeasured = sum(1 for sl in slots if (getattr(sl, "personality", None) or {}).get("measured") is not True)
    m.button(surf, (x + 10, rect.y + 54, m.bw(tr("Measure the flies in play ({n})").format(n=unmeasured), 300), 32), tr("Measure the flies in play ({n})").format(n=unmeasured), lambda: _start_measure_play(m, st),
             id="arcade_measure_play", enabled=bool(slots) and unmeasured > 0 and not busy,
             tip=tr("Reads each fly's personality card from its own brain (looming latency, sugar response, steering), about 10 s of compute "
                    "per fly in the background. Until then a fly shows 'card not measured': the game never shows numbers it did not measure."))
    body = pygame.Rect(rect.x + 16, rect.y + 96, rect.w - 32, rect.h - 96 - 76)
    key = "arcade_" + st.tab
    off = int(m.scroll.get(key, 0))
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    y = body.y + 4 - off
    y = (_tournament(m, surf, body, y, st) if st.tab == "tournament" else _racing(m, surf, body, y, st))
    m.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    if st.error:
        m.text(surf, st.error, (rect.x + 24, rect.bottom - 84), ui.BAD, m.f_small)
    if not _bar(m, surf, rect.bottom - 44, rect, st):
        pass
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back"), m.back, style="primary", id=("arcade", "back"))


def _tournament(m, surf, body, y: int, st: ArcadeState) -> int:
    cols = _colors(m)
    busy = st.job is not None and st.job.running
    x = body.x + 8
    y = m.wrapped(surf, tr("A bracket of flies, each its own individuality seed. Both sides of every duel are brains: nothing about who wins is scripted."),
                  (x, y), body.w - 24, ui.LABEL, m.f_small, max_lines=4) + 6
    for n in (4, 8, 16):
        m.button(surf, (x, y, 70, 32), str(n), (lambda n=n: setattr(st, "size", n) or setattr(st, "favorite", None)), id=("arc_size", n),
                 active=st.size == n, enabled=not busy, tip=tr("How many flies in the bracket."))
        x += 78
    m.slider(surf, (x + 10, y, 250, 32), st.match_s, 10, 40, 5, "{:.0f} s duels", lambda v: setattr(st, "match_s", float(v)), lambda: None,
             id="arc_len", enabled=not busy, tip=tr("How long a duel lasts if nobody is knocked out. More health at the end wins."))
    m.button(surf, (x + 280, y, 130, 32), tr("New flies"), lambda: setattr(st, "base_seed", st.base_seed + st.size) or setattr(st, "favorite", None),
             id="arc_new", enabled=not busy, tip=tr("Pick a different set of individuality seeds."))
    rw = max(190, m.f_bold.size(tr("Run tournament"))[0] + 32)
    if x + 420 + rw > body.right - 8:                       # a narrow menu at larger text: the Run button goes on its own row
        y += 40
        x = body.x + 8 - 420
    m.button(surf, (x + 420, y, rw, 32), tr("Run tournament"), lambda: _start_tournament(m, st), style="primary", id="arc_run",
             enabled=not busy, tip=tr("Measures each fly's personality card, then plays the bracket. Minutes: it runs in the background."))
    y += 44
    x = body.x + 8
    xx = _tag(m, surf, x, y, "CONNECTOME")
    xx = _tag(m, surf, xx, y, "GAME RULE")
    xx = _tag(m, surf, xx, y, "MODEL PREDICTION")
    y = max(y + 30, m.wrapped(surf, tr("brains steer and shoot (CONNECTOME); the arena and blaster (GAME RULE); who wins (MODEL PREDICTION)"),
                              (xx + 6, y + 2), body.right - xx - 16, ui.LABEL, m.f_small, max_lines=4) + 6)
    seeds = [st.base_seed + i for i in range(st.size)]
    m.text(surf, tr("Pick your favorite (click a fly):"), (x, y), ui.INK, m.f_bold)
    y += 26
    cards = (st.bracket or {}).get("cards", {})
    labels = []
    for s in seeds:
        c = cards.get(s) or cards.get(str(s))
        labels.append(("FAV  " if st.favorite == s else "") + (f"{s}  {c['title']}" if c else f"{tr('Fly')} {s}"))
    bfont = m.f_bold                                        # 3.0 day 4 review: wide enough for the label in every text size (buttons use f_bold)
    cw = max(220, max(bfont.size(t)[0] for t in labels) + 24)
    cw = min(cw, body.w - 16)
    per_row = max(1, (body.w - 16) // (cw + 8))
    for i, s in enumerate(seeds):
        cx, cy = x + (i % per_row) * (cw + 8), y + (i // per_row) * 40
        c = cards.get(s) or cards.get(str(s))
        m.button(surf, (cx, cy, cw, 34), labels[i], (lambda s=s: setattr(st, "favorite", s)), id=("arc_fav", s),
                 active=st.favorite == s, enabled=not busy,
                 tip=(c["summary"] + f" · looming latency {c['loom_latency_s']:.2f} s, sugar x{c['sugar_ratio']:.2f}, steering R/L {c['turning_ratio']:.2f}"
                      " (measured from this fly's brain)") if c else tr("Its personality card is measured when the tournament runs."))
    y += ((len(seeds) + per_row - 1) // per_row) * 40 + 8
    b = st.bracket
    if b is None:
        m.text(surf, tr("No tournament yet."), (x, y), ui.LABEL, m.f_text)
        return y + 30
    # the bracket
    m.text(surf, tr("Bracket (click a match to replay it)"), (x, y), ui.INK, m.f_bold)
    y += 26
    colw = max(150, (body.w - 24) // max(1, len(b["rounds"])))
    top = y
    for ri, rnd in enumerate(b["rounds"]):
        for mi, mt in enumerate(rnd):
            span = 2 ** ri
            my = top + int((mi + 0.5) * span * 50 - 22)
            box = pygame.Rect(x + ri * colw, my, colw - 14, 44)
            sel = st.sel_match == (ri, mi)
            pygame.draw.rect(surf, (36, 44, 60) if sel else (26, 30, 40), box, border_radius=6)
            pygame.draw.rect(surf, ui.ACCENT if sel else ui.BORDER, box, 1, border_radius=6)
            for k, f in enumerate((mt["a"], mt["b"])):
                won = mt["winner"] == f
                m.text(surf, f"{'FAV ' if b.get('favorite') == f else ''}{f}  " + (tr("won") if won else ""), (box.x + 8, box.y + 4 + k * 18),
                       ui.GOOD if won else ui.TEXT, m.f_small)
            m._register(box, "button", id=("arc_match", ri, mi), click=(lambda ri=ri, mi=mi: (setattr(st, "sel_match", (ri, mi)), setattr(st, "replay_t", 0.0))),
                        enabled=True, tip=f"{mt['decided_by']}; hits {mt['hits'][str(mt['a'])]}-{mt['hits'][str(mt['b'])]}"
                        + ("; coin toss" if mt["coin_toss"] else ""))
    y = top + int(len(b["rounds"][0]) * 50) + 12
    # the replay
    if st.sel_match is not None:
        ri, mi = st.sel_match
        mt = b["rounds"][ri][mi]
        m.text(surf, f"{tr('Replay')}: {mt['a']} v {mt['b']}  ->  {mt['winner']}  ({mt['decided_by']}{', ' + tr('coin toss') if mt['coin_toss'] else ''})",
               (x, y), ui.INK, m.f_bold)
        y += 26
        side = min(330, body.w // 2)
        arena = pygame.Rect(x, y, side, side)
        length = len(mt["frames"]) / 10.0
        _advance(st, length)
        _draw_duel(m, surf, arena, mt["frames"], st.replay_t, (str(mt["a"]), str(mt["b"])), cols)
        m.button(surf, (arena.right + 16, y, 120, 32), tr("Pause") if st.playing else tr("Play"), lambda: setattr(st, "playing", not st.playing),
                 id="arc_play")
        m.slider(surf, (arena.right + 16, y + 40, 300, 30), st.replay_t, 0, max(0.5, length), 0.1, "{:.1f} s", lambda v: setattr(st, "replay_t", float(v)),
                 lambda: None, id="arc_scrub")
        ev = [e for e in mt["events"] if e[0] in ("hit", "dodge", "run")]
        ey = m.wrapped(surf, tr("Events:") + " " + ", ".join(f"{e[0]} {e[2]}@{e[1]:.1f}s" for e in ev[:6]), (arena.right + 16, y + 80),
                       body.right - arena.right - 32, ui.LABEL, m.f_small, max_lines=4)
        m.text(surf, f"{tr('shots')} {mt['shots'][str(mt['a'])]}-{mt['shots'][str(mt['b'])]}  ·  max DNp35 level "
                     f"{mt['max_levels'][str(mt['a'])]['fire']:.1f} / {mt['max_levels'][str(mt['b'])]['fire']:.1f}", (arena.right + 16, ey + 2), ui.LABEL, m.f_small)
        y += side + 14
    # the champion
    ch = b["champion"]
    c = b["cards"].get(ch) or b["cards"].get(str(ch))
    m.text(surf, f"{tr('Champion')}: fly {ch}  {c['title'] if c else ''}  ({tr('rounds won')} {b['rounds_won'].get(ch, b['rounds_won'].get(str(ch), 0))})",
           (x, y), ui.AMBER, m.f_head)
    y += 38
    if c:
        m.text(surf, c["summary"], (x, y), ui.TEXT, m.f_small)
        y += 22
    m.text(surf, tr("Which neurons fired before the champion's landed shots (a correlation: nothing was silenced; the touch neurons lead because a hit fly also fires them):"),
           (x, y), ui.LABEL, m.f_small)
    y += 22
    for r in b.get("champion_drivers", [])[:6]:
        m.text(surf, f"{r['type']:<14} x{r['ratio']:.1f}  ({r['hz_before_hits']:.1f} vs {r['hz_overall']:.1f} Hz, {r['neurons']} neurons)", (x + 8, y), ui.TEXT, m.f_small)
        y += 18
    ro = b.get("champion_readouts", [])
    if ro:
        y += 4
        m.text(surf, tr("Neurons the duel's own rules read:") + "  " + "  ".join(f"{r['type']} x{r['ratio']:.1f}" for r in ro[:8]), (x, y), ui.TEXT, m.f_small)
        y += 22
    return y + 6


def _racing(m, surf, body, y: int, st: ArcadeState) -> int:
    from kickthefly.core import points

    cols = _colors(m)
    busy = st.job is not None and st.job.running
    x = body.x + 8
    y = m.wrapped(surf, tr("Flies race down a track with sugar and fruit lures, each through its own brain. Bet points on a fly; the odds come from its personality card."),
                  (x, y), body.w - 24, ui.LABEL, m.f_small, max_lines=4) + 6
    m.slider(surf, (x, y, 260, 32), st.lanes, 3, 8, 1, "{:.0f} lanes", lambda v: setattr(st, "lanes", int(v)) or setattr(st, "field", None), lambda: None,
             id="race_lanes", enabled=not busy)
    m.button(surf, (x + 280, y, 130, 32), tr("New field"), lambda: (setattr(st, "race_base", st.race_base + st.lanes), setattr(st, "field", None),
                                                                    setattr(st, "odds", None), setattr(st, "race", None)),
             id="race_new", enabled=not busy)
    m.button(surf, (x + 420, y, 200, 32), tr("Look at the field"), lambda: _start_field(m, st), id="race_look", enabled=not busy,
             tip=tr("Measures each fly's personality card (a few seconds each) and sets the odds."))
    y += 44
    xx = _tag(m, surf, x, y, "CONNECTOME")
    xx = _tag(m, surf, xx, y, "GAME RULE")
    xx = _tag(m, surf, xx, y, "MODEL PREDICTION")
    y = max(y + 30, m.wrapped(surf, tr("walking neurons set the speed (CONNECTOME); track, lures, odds, points (GAME RULE); the finish (MODEL PREDICTION)"),
                              (xx + 6, y + 2), body.right - xx - 16, ui.LABEL, m.f_small, max_lines=4) + 6)
    if st.odds is None:
        m.text(surf, tr("Look at the field to see the odds."), (x, y), ui.LABEL, m.f_text)
        return y + 30
    y = m.wrapped(surf, tr("Odds (form = sugar response and calm walking drive, from the measured card). Click a fly to bet on it:"), (x, y),
                  body.w - 24, ui.INK, m.f_bold, max_lines=3) + 4
    for o in st.odds:
        row = pygame.Rect(x, y, body.w - 24, 32)
        c = next((c for c in st.field if c["seed"] == o["fly"]), None)
        m.button(surf, row, f"{'BET  ' if st.bet_fly == o['fly'] else ''}fly {o['fly']}  {o['title']}   x{o['decimal_odds']:g}   ({o['p_win'] * 100:.0f}%)",
                 (lambda f=o["fly"]: setattr(st, "bet_fly", f)), id=("race_bet", o["fly"]), active=st.bet_fly == o["fly"], enabled=not busy,
                 tip=(c["summary"] + f" · sugar x{c['sugar_ratio']:.2f}, calm walking drive {c['walk_level_calm']:.2f}") if c else None)
        y += 36
    for s_ in points.STAKES:
        m.button(surf, (x, y, 80, 32), f"{s_}", (lambda s_=s_: setattr(st, "stake", s_)), id=("race_stake", s_), active=st.stake == s_,
                 enabled=not busy and s_ <= st.wallet.points, tip=tr("Points to stake (game points only)."))
        x += 88
    m.button(surf, (x + 10, y, 170, 32), tr("Start race"), lambda: _start_race(m, st), style="primary", id="race_go", enabled=not busy,
             tip=tr("Runs every lane in the background. Betting is optional; no money is involved, ever."))
    y += 44
    x = body.x + 8
    r = st.race
    if r is None:
        return y
    winner = r["winner"]
    m.text(surf, tr("Finishing order:") + " " + "  >  ".join(f"{s}" for s in r["orders"][0]), (x, y), ui.AMBER, m.f_bold)
    y += 26
    if st.bet_result:
        b = st.bet_result[0]
        m.text(surf, (tr("You won") if b["won"] else tr("You lost")) + f" {abs(b['delta'])} " + tr("points on fly {f} at x{o:g}.").format(f=b["fly"], o=b["odds"]),
               (x, y), ui.GOOD if b["won"] else ui.BAD, m.f_text)
        y += 26
    length = max((l["trace"][-1][0] for l in r["runs"]["0"].values()), default=1.0)
    _advance(st, length + 1.0)
    track = pygame.Rect(x, y, body.w - 40, 34 * len(r["seeds"]) + 12)
    pygame.draw.rect(surf, (12, 15, 20), track, border_radius=8)
    for i, s in enumerate(r["seeds"]):
        lane = r["runs"]["0"][str(s)]
        ly = track.y + 8 + i * 34
        pygame.draw.line(surf, (50, 56, 68), (track.x + 70, ly + 14), (track.right - 20, ly + 14), 2)
        for (lx, kind) in r["lures"]:
            cx = track.x + 70 + int(lx / r["track_m"] * (track.w - 90))
            pygame.draw.circle(surf, (240, 200, 80) if kind == "sugar" else (220, 90, 90), (cx, ly + 14), 6)
        pos = lane["distance"] if lane["finish_s"] is None else r["track_m"]
        for (tt, xx_) in lane["trace"]:
            if tt >= st.replay_t:
                break
            pos = xx_
        else:
            pos = r["track_m"] if lane["finish_s"] is not None else lane["trace"][-1][1]
        px = track.x + 70 + int(pos / r["track_m"] * (track.w - 90))
        pygame.draw.circle(surf, cols[i % 2] if s != winner else ui.AMBER, (px, ly + 14), 10)
        m.text(surf, f"{s}", (track.x + 8, ly + 6), ui.TEXT, m.f_small)
    y += track.h + 10
    y = m.wrapped(surf, tr("MODEL PREDICTION. Points only. The headless race assay asks whether individuality predicts the finish: --headless --race"),
                  (x, y), body.w - 24, ui.LABEL, m.f_small, max_lines=3)
    return y + 8


def pad_nav(host, down) -> bool:
    """Gamepad while the Fly arcade is open (3D). `down` holds the action names whose buttons just went down, so these are the pad's own
    bindings: the bumpers (next / previous tool) pick the favorite or the fly to bet on, the kill cam and big-view buttons change the
    bracket size or the number of lanes, the trigger (use) runs the tournament, or looks at the field and then starts the race, the
    Neurodex button switches between Tournament and Racing, and B (crouch) or Start closes the page. True if a button was used."""
    m = host.menu
    if m.screen != "arcade":
        return False
    st = state(m)
    busy = st.job is not None and st.job.running
    if down & {"crouch", "menu"}:
        m.back()
        return True
    used = False
    if "neurodex" in down:
        st.tab = "race" if st.tab == "tournament" else "tournament"
        used = True
    if busy:
        return used
    step = (1 if "tool_next" in down else -1) if down & {"tool_next", "tool_prev"} else 0
    if step and st.tab == "tournament":
        seeds = [st.base_seed + i for i in range(st.size)]
        cur = seeds.index(st.favorite) if st.favorite in seeds else (-1 if step > 0 else 0)
        st.favorite = seeds[(cur + step) % len(seeds)]
        used = True
    elif step and st.odds:
        flies = [o["fly"] for o in st.odds]
        cur = flies.index(st.bet_fly) if st.bet_fly in flies else (-1 if step > 0 else 0)
        st.bet_fly = flies[(cur + step) % len(flies)]
        used = True
    if down & {"killcam", "big_view"}:
        d = 1 if "killcam" in down else -1
        if st.tab == "tournament":
            sizes = (4, 8, 16)
            st.size, st.favorite = sizes[(sizes.index(st.size) + d) % 3], None
        else:
            st.lanes, st.field, st.odds = int(min(8, max(3, st.lanes + d))), None, None
        used = True
    if "use" in down:
        if st.tab == "tournament":
            _start_tournament(m, st)
        elif st.field is None:
            _start_field(m, st)
        else:
            _start_race(m, st)
        used = True
    return used
