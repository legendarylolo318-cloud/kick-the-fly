#!/usr/bin/env python3
"""Screenshots of the pygame Fly arcade page (ui/arcade_ui.py) with a fixed fixture for the Rust port to be compared with:
python make_arcade_reference.py [OUT_DIR]. Writes arcade_fixture.json (the data the Rust host answers with) and page_arcade_*.png."""
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_ui_reference as R  # noqa: E402
from kickthefly.lab import racing  # noqa: E402
from kickthefly.ui import arcade_ui  # noqa: E402


def card(s):
    lat = 0.1 + 0.037 * (s % 7)
    sugar = 1.0 + 0.23 * (s % 5)
    turn = 0.8 + 0.07 * (s % 6)
    walk = 1.0 + 0.11 * (s % 4)
    bold = "Bold" if lat >= 0.3 else "Skittish" if lat <= 0.2 else "Alert"
    steer = "Right-turner" if turn >= 1.12 else "Left-turner" if turn <= 0.89 else "Straight-walker"
    return dict(seed=s, mode="subtle", loom_latency_s=lat, sugar_ratio=sugar, turning_ratio=turn, turning_log_ratio=math.log(turn), walk_level_calm=walk,
                title=f"{bold} {steer}", summary=f"{bold} · {steer} · Steady feeder", traits=[bold, steer, "Steady feeder"], measured=True)


def frames(a, b, n=60):
    out = []
    for i in range(n):
        t = i / 10
        ang = i / n * 2.2
        ax, ay = -3 + 1.2 * math.cos(ang) + i * 0.02, 1.5 * math.sin(ang)
        bx, by = 3 - 1.2 * math.cos(ang + 1) - i * 0.02, -1.2 * math.sin(ang + 0.5)
        pell = [[round(ax + 0.5 * k, 2), round(ay + 0.3 * k, 2), 0] for k in range(i % 4)] + [[round(bx - 0.4 * k, 2), round(by, 2), 1] for k in range(i % 3)]
        out.append([round(t, 3), ax, ay, ang, max(0.0, 100.0 - i * 0.6), bx, by, math.pi - ang, max(0.0, 100.0 - i * 0.9), pell])
    return out


seeds = list(range(2000, 2008))
cards = {str(s): card(s) for s in seeds}
rounds, fighters, won = [], list(seeds), {str(s): 0 for s in seeds}
for ri in range(3):
    ms = []
    for i in range(0, len(fighters), 2):
        a, b = fighters[i], fighters[i + 1]
        w = a if (a * 7 + b * 3 + ri) % 3 else b
        won[str(w)] += 1
        ms.append(dict(a=a, b=b, winner=w, coin_toss=(ri == 0 and i == 2), attempts=1, decided_by="knockout" if (a + ri) % 2 else "health at the time limit",
                       hp={str(a): 41.0, str(b): 0.0}, shots={str(a): 9 + ri, str(b): 6}, hits={str(a): 6 + ri, str(b): 2}, seconds=14.5, knockout=True,
                       max_levels={str(a): dict(fire=4.5 + ri, escape=2.1, run=1.0, walk=3.0), str(b): dict(fire=3.8, escape=1.0, run=1.5, walk=2.0)},
                       events=[["shot", 1.1, str(a)], ["hit", 1.6, str(a)], ["dodge", 2.4, str(b)], ["run", 3.0, str(b)], ["hit", 4.4, str(a)], ["shot", 5.0, str(b)], ["hit", 6.0, str(b)]],
                       frames=frames(a, b), match_seed=7))
    rounds.append(ms)
    fighters = [m["winner"] for m in ms]
champ = fighters[0]
bracket = dict(kind="tournament", size=8, seeds=seeds, individuality="subtle", match_seconds=20.0, bracket_seed=0, favorite=2003, favorite_won=False,
               favorite_rounds_won=1, cards=cards, rounds=rounds, champion=champ, rounds_won=won,
               champion_drivers=[dict(type=t, neurons=n, hz_before_hits=h, hz_overall=o, ratio=h / o) for t, n, h, o in
                                 (("SNta13", 12, 18.2, 3.1), ("DNp35", 2, 22.0, 5.0), ("LC10", 140, 9.5, 4.0), ("DNa01", 4, 12.0, 6.5), ("KCg", 800, 6.0, 4.1),
                                  ("DNp01", 2, 8.2, 5.9), ("MBON01", 6, 7.0, 6.1))],
               champion_readouts=[dict(type=t, neurons=2, hz_before_hits=h, hz_overall=o, ratio=h / o) for t, h, o in (("DNp35", 22.0, 5.0), ("DNa01", 12.0, 6.5), ("LC10", 9.5, 4.0))])
rseeds = [3000, 3001, 3002, 3003, 3004]
field = [card(s) for s in rseeds]
odds = racing.odds_table(field)
runs = {}
for i, s in enumerate(rseeds):
    fin = 22.0 + 3.5 * i if i != 4 else None
    runs[str(s)] = dict(finish_s=fin, distance=8.0 if fin else 5.2, seconds=fin or 90.0, lures_touched=[0, 1], max_walk_level=2.1, rule_version=2,
                        trace=[[round(t * 0.5, 2), round(min(8.0, t * 0.5 * (8.0 / (fin or 60.0))), 3)] for t in range(0, int((fin or 40.0) * 2) + 1, 5)])
race = dict(kind="race", seeds=rseeds, individuality="subtle", race_seed=3000, repeats=1, track_m=8.0, lures=[[2.0, "sugar"], [4.0, "fruit"], [6.0, "sugar"]],
            cards={str(c["seed"]): c for c in field}, runs={"0": runs}, orders=[[3000, 3001, 3002, 3003, 3004]], winner=3000, speed_rule_version=2)
bet_result = [dict(fly=3002, stake=10, odds=odds[2]["decimal_odds"], won=False, delta=-10, points=90)]
fx = dict(points=100, slots=0, unmeasured=0, job=None, error="", palette="default", bracket_serial=1, bracket=bracket, race_serial=1,
          field=field, odds=odds, race=race, bet_result=bet_result)

def prep_for(tab, scroll, **extra):
    def prep(m):
        m.pages["arcade"] = arcade_ui.page
        st = arcade_ui.state(m)
        st.tab, st.bracket, st.field, st.odds, st.race, st.bet_result = tab, bracket, field, odds, race, bet_result
        st.favorite, st.sel_match, st.bet_fly, st.playing, st.replay_t = 2003, (2, 0), 3002, False, 0.0
        st.wallet = SimpleNamespace(points=100)
        st.error = ""
        for k, v in extra.items():
            setattr(st, k, v)
        m.scroll[f"arcade_{tab}"] = scroll
    return prep


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "ui_reference")
    out.mkdir(parents=True, exist_ok=True)
    (out / "arcade_fixture.json").write_text(json.dumps(fx))
    R.INSTALL = []
    for name, tab, scroll in (("tournament_top", "tournament", 0), ("tournament_replay", "tournament", 330), ("tournament_end", "tournament", 560),
                              ("race_top", "race", 0), ("race_lanes", "race", 100)):
        R.render("arcade", "Graphics", out / f"page_arcade_{name}.png", prep=prep_for(tab, scroll))
    print("wrote", out)


# --- the lesion battle tab -----------------------------------------------------------------------------------------------------------
def battle_fixture():
    from kickthefly.game import lesion_battle as lb

    sizes = {"DNp35": 2, "DNpe052": 2, "DNa01": 4, "DNa02": 4, "DNp01": 2, "DNp09": 2, "LC10a": 120, "LC10b": 90, "LC11": 33, "LPLC2": 280, "LC4": 70,
             "MDN": 4, "DNg02_a": 28, "PPL1": 130, "PAM": 311, "KCg": 1800, "DNpe010": 2, "DNp06": 2, "DNp07": 2, "LC12": 300, "LC16": 450, "LC17": 900,
             "MBON01": 2, "DNg12_a": 15}
    types = [t for t, n in sizes.items() for _ in range(n)]
    saved = {"my best": {"DNp35": -1, "MDN": 1}, "sleepy": {"DNa01": -1}}
    a = {"DNp35": -1, "MDN": 1}
    b = {"LC10a": 1}

    def rd(w, hp_a, hp_b):
        return dict(winner=w, frames=frames(1, 2), shots={"A": 8, "B": 5}, hits={"A": 3, "B": 2}, hp={"A": hp_a, "B": hp_b},
                    max_levels={"A": dict(fire=4.1, escape=2.2, run=1.0, walk=3.0), "B": dict(fire=3.4, escape=1.1, run=1.8, walk=2.2)})

    result = dict(winner="A", wins={"A": 2, "B": 1}, rounds=[rd("A", 55.0, 0.0), rd("B", 0.0, 37.0), rd("A", 82.0, 0.0)], rounds_asked=3,
                  seeds=dict(A=5000, B=5001), seconds_each=20.0, loadouts=dict(A=a, B=b), individuality="subtle", tag="MODEL PREDICTION")
    return lb, sizes, types, saved, a, b, result


def battle_prep(phase, los, search="", rounds=1, seconds=20.0, msg="", bad=False, result=None):
    lb, sizes, types, saved, *_ = battle_fixture()
    lb.load_saved = lambda: dict(saved)

    def prep(m):
        from kickthefly.ui import battle_ui

        m.pages["arcade"] = arcade_ui.page
        m.host.brain = SimpleNamespace(types=types)
        st = arcade_ui.state(m)
        st.tab, st.wallet, st.error = "battle", SimpleNamespace(points=100), ""
        bt = battle_ui.bstate(st)
        bt.phase, bt.search, bt.rounds, bt.seconds = phase, search, rounds, seconds
        bt.los = [lb.Loadout("player 1", dict(los[0])), lb.Loadout("player 2", dict(los[1]))]
        bt.msg, bt.msg_bad, bt.playing, bt.replay_t, bt.result, bt.sel_round = msg, bad, False, 0.0, result, 1
        m.scroll["arcade_battle"] = 0
    return prep


def write_battle(out):
    lb, sizes, types, saved, a, b, result = battle_fixture()
    fx2 = dict(sizes=sizes, saved=[[n, [[t, m] for t, m in p.items()]] for n, p in saved.items()],
               result=dict(serial=1, winner=result["winner"], wins=result["wins"], rounds=result["rounds"],
                           loadouts={k: [[t, m] for t, m in v.items()] for k, v in result["loadouts"].items()}))
    (out / "arcade_battle_fixture.json").write_text(json.dumps(fx2))
    cases = (("battle_setup1", battle_prep("setup1", (a, {}), search="DN")),
             ("battle_setup2", battle_prep("setup2", ({"DNp35": -1}, {"LC10a": 1, "LC10b": -1, "LPLC2": 1}), search="LC",
                                           msg="the budget is 3 picks: take one out first", bad=True)),
             ("battle_pass", battle_prep("pass", (a, {}))),
             ("battle_ready", battle_prep("ready", (a, b), rounds=3, seconds=25.0)),
             ("battle_result", battle_prep("result", (a, b), result=result)))
    for name, prep in cases:
        R.render("arcade", "Graphics", out / f"page_arcade_{name}.png", prep=prep)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "ui_reference")
    write_battle(out)
