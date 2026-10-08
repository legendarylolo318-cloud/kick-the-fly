"""Lesion battle tab of the Fly arcade (3.1.0 task 12): two players, one screen, each does surgery on their own fly within a budget, then the flies fight.

The engine, the rules and the tags are in game/lesion_battle.py; the duel itself is game/flyduel.py (both sides are brains). This is the page: hot-seat setup (the second
player chooses while the first player's picks are hidden), saved loadouts, share codes (an ordinary surgery code), the fight as a background job, and the replay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pygame

from kickthefly.core.i18n import tr
from kickthefly.game import lesion_battle as lb
from kickthefly.ui import menu as ui


@dataclass
class BattleState:
    phase: str = "setup1"            # setup1 -> pass -> setup2 -> ready -> (job) -> result
    los: list = field(default_factory=lambda: [lb.Loadout("player 1"), lb.Loadout("player 2")])
    effect: int = lb.SILENCE
    search: str = ""
    rounds: int = 1
    seconds: float = 20.0
    base_seed: int = 5000
    sizes: dict | None = None
    result: dict | None = None
    sel_round: int = 0
    replay_t: float = 0.0
    playing: bool = True
    last: float = field(default_factory=time.perf_counter)
    msg: str = ""
    msg_bad: bool = False
    paste_note: str = ""

    @property
    def player(self) -> int:
        return 1 if self.phase == "setup2" else 0

    def reset(self) -> None:
        self.phase, self.result, self.msg = "setup1", None, ""
        self.los = [lb.Loadout("player 1"), lb.Loadout("player 2")]
        self.search = ""


def bstate(st) -> BattleState:
    if getattr(st, "bt", None) is None:
        st.bt = BattleState()
    return st.bt


def _sizes(m, bt: BattleState) -> dict:
    if bt.sizes is None:
        br = getattr(m.host, "brain", None)
        bt.sizes = lb.type_sizes(br.types) if br is not None else {}
    return bt.sizes


def _say(bt: BattleState, text: str, bad: bool = False) -> None:
    bt.msg, bt.msg_bad = text, bad


def candidates(sizes: dict, query: str, limit: int = 14) -> list[tuple[str, int]]:
    """The cell types a pick may name that match the search (exact first, then starts-with, then contains), within the size cap."""
    q = query.strip().lower()
    ok = [(t, n) for t, n in sizes.items() if n <= lb.MAX_TYPE_NEURONS]
    if not q:
        return [(t, sizes[t]) for t in lb.SUGGESTED if t in sizes and sizes[t] <= lb.MAX_TYPE_NEURONS][:limit]
    rank = lambda tn: (tn[0].lower() != q, not tn[0].lower().startswith(q), tn[0].lower())       # noqa: E731
    return sorted((x for x in ok if q in x[0].lower()), key=rank)[:limit]


def _add(bt: BattleState, sizes: dict, t: str) -> None:
    lo = bt.los[bt.player]
    try:
        lo.add(t, bt.effect, sizes)
        _say(bt, "")
    except lb.LoadoutError as e:
        _say(bt, str(e), True)


def _done(bt: BattleState) -> None:
    bt.phase = "pass" if bt.phase == "setup1" else "ready"
    bt.search, bt.msg = "", ""


def _copy(bt: BattleState) -> None:
    from kickthefly.core import clipboard

    lo = bt.los[bt.player]
    try:
        code = lo.code()
    except Exception as e:
        _say(bt, str(e), True)
        return
    _say(bt, tr("Code copied: {c}").format(c=code[:30] + "...") if clipboard.put_text(code) else tr("Could not reach the clipboard; the code is shown below."), False)
    bt.paste_note = code


def _paste(m, bt: BattleState, sizes: dict) -> None:
    from kickthefly.core import clipboard

    text = clipboard.get_text()
    if not text:
        _say(bt, tr("The clipboard has no text."), True)
        return
    try:
        lo = lb.Loadout.from_code(text, sizes, name=bt.los[bt.player].name)
    except lb.LoadoutError as e:
        _say(bt, str(e), True)
        return
    bt.los[bt.player] = lo
    _say(bt, tr("Loaded {n} picks from the code.").format(n=lo.used()))


def start_fight(m, st, bt: BattleState) -> None:
    from kickthefly.core import config
    from kickthefly.ui.bgjob import BgJob

    lo1, lo2 = bt.los
    rounds, seconds = bt.rounds, bt.seconds
    s1, s2 = bt.base_seed, bt.base_seed + 1
    mode = str(m.host.cfg["brain.individuality"]) if hasattr(m.host, "cfg") else "subtle"
    mode = mode if mode in ("off", "subtle", "strong") else "subtle"

    def work(job: BgJob):
        return lb.fight(s1, s2, lo1, lo2, rounds=rounds, seconds=seconds, mode=mode, battle_seed=bt.base_seed, workers=2,
                        cancel=job.cancel, progress=lambda d, n: job.update(d / max(1, n), f"round {d} of {n}"))

    bt.result, bt.msg = None, ""
    bt.phase = "fighting"
    st.job = BgJob("Battle", work).start()


def collect(m, st, job) -> None:
    """The finished battle job's result into the page (the arcade page's _collect calls this)."""
    bt = bstate(st)
    if job.error:
        st.error = job.error
        bt.phase = "ready"
        return
    if job.cancelled or job.result is None:
        bt.phase = "ready"
        return
    bt.result, bt.phase = job.result, "result"
    bt.sel_round, bt.replay_t, bt.playing = 0, 0.0, True


def draw(m, surf, body: pygame.Rect, y: int, st, tag, draw_duel, advance, colors) -> int:
    """The tab's content; returns the y below it. (tag/draw_duel/advance/colors are the arcade page's own helpers.)"""
    bt = bstate(st)
    busy = st.job is not None and st.job.running
    x = body.x + 8
    sizes = _sizes(m, bt)
    y = m.wrapped(surf, tr("Each player does surgery on their own fly, within a budget of {n} cell types (each silenced or stimulated, up to {c} neurons), "
                           "then the two flies fight. Hot-seat: the second player chooses while the first player's picks are hidden.").format(n=lb.BUDGET, c=lb.MAX_TYPE_NEURONS),
                  (x, y), body.w - 24, ui.LABEL, m.f_small, max_lines=4) + 6
    xx = tag(m, surf, x, y, "CONNECTOME")
    xx = tag(m, surf, xx, y, "GAME RULE")
    xx = tag(m, surf, xx, y, "MODEL PREDICTION")
    y = max(y + 30, m.wrapped(surf, tr("what the surgery does (CONNECTOME); the budget and the arena (GAME RULE); who wins (MODEL PREDICTION)"),
                              (xx + 6, y + 2), body.right - xx - 16, ui.LABEL, m.f_small, max_lines=3) + 6)
    if bt.phase in ("setup1", "setup2"):
        return _setup(m, surf, body, y, st, bt, sizes)
    if bt.phase == "pass":
        m.text(surf, tr("PLAYER 1 IS DONE."), (x, y), ui.AMBER, m.f_head)
        y += 44
        y = m.wrapped(surf, tr("Player 1: look away. Hand over to player 2, who now operates on their own fly. Player 1's picks are not shown again until the fight."),
                      (x, y), body.w - 24, ui.TEXT, m.f_text, max_lines=4) + 10
        m.button(surf, (x, y, 260, 40), tr("Player 2 is ready"), lambda: setattr(bt, "phase", "setup2"), style="primary", id="bt_pass")
        return y + 52
    if bt.phase in ("ready", "fighting"):
        m.text(surf, tr("READY TO FIGHT"), (x, y), ui.AMBER, m.f_head)
        y += 44
        for i, lo in enumerate(bt.los):
            m.text(surf, tr("Player {n}: {k} of {b} picks, {t} neurons (hidden until the fight)").format(n=i + 1, k=lo.used(), b=lb.BUDGET, t=lo.neurons(sizes)),
                   (x, y), ui.TEXT, m.f_text)
            y += 26
        y += 6
        for r in lb.ROUNDS:
            m.button(surf, (x, y, 130, 32), tr("{n} round(s)").format(n=r) if r == 1 else tr("best of {n}").format(n=r), (lambda r=r: setattr(bt, "rounds", r)),
                     id=("bt_rounds", r), active=bt.rounds == r, enabled=not busy)
            x += 138
        m.slider(surf, (x + 6, y, 240, 32), bt.seconds, 10, 40, 5, "{:.0f} s rounds", lambda v: setattr(bt, "seconds", float(v)), lambda: None, id="bt_len", enabled=not busy)
        m.button(surf, (x + 260, y, 190, 32), tr("New flies"), lambda: setattr(bt, "base_seed", bt.base_seed + 2), id="bt_new", enabled=not busy,
                 tip=tr("Pick a different pair of individuality seeds. The same two flies are used in every round."))
        y += 44
        x = body.x + 8
        m.button(surf, (x, y, 220, 44), tr("FIGHT"), lambda: start_fight(m, st, bt), style="primary", id="bt_fight", enabled=not busy,
                 tip=tr("Runs the duel in the background (about a minute a round)."))
        m.button(surf, (x + 232, y, 190, 44), tr("Start again"), bt.reset, id="bt_reset", enabled=not busy)
        return y + 56
    return _result(m, surf, body, y, st, bt, draw_duel, advance, colors)


def _setup(m, surf, body, y: int, st, bt: BattleState, sizes: dict) -> int:
    lo = bt.los[bt.player]
    x = body.x + 8
    m.text(surf, tr("PLAYER {n}: operate on your fly").format(n=bt.player + 1), (x, y), ui.AMBER, m.f_head)
    y += 42
    m.text(surf, tr("{k} of {b} picks used").format(k=lo.used(), b=lb.BUDGET), (x, y), ui.INK, m.f_bold)
    m.segmented(surf, (x + 200, y - 4, 260, 32), [tr("Silence"), tr("Stimulate")], 0 if bt.effect == lb.SILENCE else 1,
                lambda i: setattr(bt, "effect", lb.SILENCE if i == 0 else lb.STIMULATE), id="bt_effect",
                tip=tr("Silencing holds every neuron of the type quiet; stimulating drives it, as the game's brain surgery does."))
    y += 38
    for t, mode in list(lo.picks.items()):
        r = pygame.Rect(x, y, 330, 28)
        pygame.draw.rect(surf, (28, 32, 42), r, border_radius=6)
        m.text(surf, f"{t}  {lb.MODE_WORD[mode]}  ({sizes.get(t, 0)} neurons)", (r.x + 10, r.y + 5), (255, 150, 90) if mode > 0 else (110, 190, 255), m.f_small)
        m.button(surf, (r.right + 8, y, 80, 28), tr("Remove"), (lambda t=t: lo.remove(t)), id=("bt_rm", t), font=m.f_small)
        y += 32
    if not lo.picks:
        m.text(surf, tr("No picks yet: with none, your fly goes in as it is."), (x, y), ui.LABEL, m.f_small)
        y += 24
    y += 6
    m.text(surf, tr("Find a cell type:"), (x, y + 4), ui.TEXT, m.f_small)
    m.text_field(surf, (x + 150, y, 260, 30), bt.search, lambda v: setattr(bt, "search", v), id="bt_search", limit=24,
                 tip=tr("Type part of a name (DNp, LC, MDN, PPL...) and press Enter. Types over {c} neurons are not offered.").format(c=lb.MAX_TYPE_NEURONS))
    y += 40
    cands = candidates(sizes, bt.search)
    if not bt.search.strip():
        m.text(surf, tr("The types the duel itself reads (a hint, not an answer):"), (x, y), ui.LABEL, m.f_small)
        y += 20
    elif not cands:
        m.text(surf, tr("No cell type of at most {c} neurons matches.").format(c=lb.MAX_TYPE_NEURONS), (x, y), ui.LABEL, m.f_small)
        y += 22
    y = m.flow_buttons(surf, x, y, body.right - 16, [(f"{t} ({n})", (lambda t=t: _add(bt, sizes, t)), ("bt_cand", t), t in lo.picks, None) for t, n in cands]) + 8
    if bt.msg:
        m.text(surf, bt.msg, (x, y), ui.BAD if bt.msg_bad else ui.GOOD, m.f_small)
        y += 22
    y += 4
    m.button(surf, (x, y, 190, 40), tr("Done"), lambda: _done(bt), style="primary", id="bt_done",
             tip=tr("Hands the screen over (player 1) or readies the fight (player 2). With no picks, the fly fights as it is."))
    bx = x + 202
    m.button(surf, (bx, y, 140, 40), tr("Copy code"), lambda: _copy(bt), id="bt_copy", enabled=bool(lo.picks),
             tip=tr("A surgery code (KTF1-SRG-...): the same one Esc > Share makes. Anyone can Paste it for their own fly."))
    m.button(surf, (bx + 150, y, 140, 40), tr("Paste code"), lambda: _paste(m, bt, sizes), id="bt_paste")
    m.button(surf, (bx + 300, y, 130, 40), tr("Save"), lambda: _save(bt, lo), id="bt_save", enabled=bool(lo.picks), tip=tr("Saves this loadout under its name."))
    y += 50
    saved = lb.load_saved()
    if saved:
        m.text(surf, tr("Saved loadouts:"), (x, y + 4), ui.LABEL, m.f_small)
        y = m.flow_buttons(surf, x + 130, y, body.right - 16, [(f"{n} ({len(p)})", (lambda n=n, p=p: _load(bt, n, p, sizes)), ("bt_load", n), False, None)
                                                                for n, p in saved.items()]) + 6
    if bt.paste_note:
        y = m.wrapped(surf, bt.paste_note, (x, y), body.w - 24, ui.DIM, m.f_small, max_lines=3) + 4
    return y


def _save(bt: BattleState, lo: lb.Loadout) -> None:
    lo.name = f"loadout {len(lb.load_saved()) + 1}" if lo.name in ("", "my fly", "player 1", "player 2") else lo.name
    _say(bt, tr("Saved as {n}.").format(n=lo.name) if lb.save_loadout(lo) else tr("The saved list is full ({n}); delete one first.").format(n=lb.MAX_SAVED), False)


def _load(bt: BattleState, name: str, picks: dict, sizes: dict) -> None:
    lo = lb.Loadout(name, dict(picks))
    bad = lo.problems(sizes)
    if bad:
        _say(bt, "; ".join(bad[:2]), True)
        return
    bt.los[bt.player] = lo
    _say(bt, tr("Loaded {n}.").format(n=name))


def _result(m, surf, body, y: int, st, bt: BattleState, draw_duel, advance, colors) -> int:
    r = bt.result
    x = body.x + 8
    m.text(surf, lb.headline(r), (x, y), ui.AMBER, m.f_head)
    y += 42
    names = ("Player 1", "Player 2")
    for i, side in enumerate(("A", "B")):
        picks = r["loadouts"][side]
        m.text(surf, f"{names[i]}: " + (", ".join(f"{t} {lb.MODE_WORD[mo]}" for t, mo in picks.items()) or "no surgery"), (x, y), ui.TEXT, m.f_text)
        y += 24
    y += 6
    for k, rd in enumerate(r["rounds"]):
        w = rd["winner"]
        label = tr("Round {n}: {w}").format(n=k + 1, w=(names[0] if w == "A" else names[1] if w == "B" else tr("draw")))
        m.button(surf, (x + k * 190, y, 180, 30), label, (lambda k=k: (setattr(bt, "sel_round", k), setattr(bt, "replay_t", 0.0))), id=("bt_round", k),
                 active=bt.sel_round == k)
    y += 42
    rd = r["rounds"][min(bt.sel_round, len(r["rounds"]) - 1)]
    side = min(330, body.w // 2)
    arena = pygame.Rect(x, y, side, side)
    length = len(rd["frames"]) / 10.0
    advance(bt, length)                                         # BattleState keeps its own replay clock (playing, replay_t, last)
    draw_duel(m, surf, arena, rd["frames"], bt.replay_t, names, colors)
    m.button(surf, (arena.right + 16, y, 120, 32), tr("Pause") if bt.playing else tr("Play"), lambda: setattr(bt, "playing", not bt.playing), id="bt_play")
    m.slider(surf, (arena.right + 16, y + 40, 300, 30), bt.replay_t, 0, max(0.5, length), 0.1, "{:.1f} s", lambda v: setattr(bt, "replay_t", float(v)),
             lambda: None, id="bt_scrub")
    ty = y + 84
    m.text(surf, tr("shots {a}-{b}   hits {c}-{d}   health {e:.0f}-{f:.0f}").format(a=rd["shots"]["A"], b=rd["shots"]["B"], c=rd["hits"]["A"], d=rd["hits"]["B"],
                                                                                    e=rd["hp"]["A"], f=rd["hp"]["B"]), (arena.right + 16, ty), ui.TEXT, m.f_small)
    ty += 22
    ml = rd["max_levels"]
    m.text(surf, tr("What the surgery did (the highest level each fly's own readout reached, x its calm rate):"), (arena.right + 16, ty), ui.LABEL, m.f_small)
    ty += 20
    for n, label in (("fire", "shooting DNp35/DNpe052"), ("escape", "dodge DNp01"), ("run", "run (body-touch DNs)"), ("walk", "walk DNp09")):
        m.text(surf, f"{label:<26} P1 x{ml['A'][n]:.1f}   P2 x{ml['B'][n]:.1f}", (arena.right + 16, ty), ui.TEXT, m.f_small)
        ty += 18
    y += side + 14
    m.button(surf, (x, y, 230, 40), tr("Rematch (same surgery)"), lambda: (setattr(bt, "phase", "ready"), setattr(bt, "base_seed", bt.base_seed)), id="bt_rematch",
             tip=tr("Back to the fight screen with the same two flies and the same surgery; round noise differs only if you change the seeds."))
    m.button(surf, (x + 242, y, 200, 40), tr("New battle"), bt.reset, style="primary", id="bt_newbattle")
    y += 52
    return m.wrapped(surf, tr("MODEL PREDICTION: who wins is what this model does with these two individuals, these rules and this noise. Not a measurement of real flies."),
                     (x, y), body.w - 24, ui.LABEL, m.f_small, max_lines=3) + 8
