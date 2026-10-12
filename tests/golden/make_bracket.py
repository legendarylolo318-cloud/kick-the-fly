"""Writes tests/golden/bracket.json: tournament.run_bracket and racing.{odds_table, run_race, analyze} with a deterministic `play` function
in place of brains. The Rust fly_sim::tournament::run_bracket / fly_sim::race::{odds_table, run_race, analyze} must give the same."""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.lab import racing, tournament  # noqa: E402


def card(s, mode):
    return dict(seed=s, mode=mode, loom_latency_s=0.1 + 0.01 * (s % 7), sugar_ratio=1.0 + 0.1 * (s % 5), turning_log_ratio=0.02 * (s % 3 - 1),
                walk_level_calm=1.0 + 0.05 * (s % 4), title=f"Card {s}", summary="x", traits=[], measured=True)


TYPES = np.array(["DNp35", "DNa01", "KCg", "LC10"])


def play(t):
    if t[0] == "card":
        return card(t[1], t[2])
    a, b, bseed, attempt, seconds, mode, backend, drivers = t
    x = (a * 31 + b * 17 + attempt * 7) % 10
    winner = None if x == 0 else (a if x % 2 else b)
    hp = {str(a): 100.0 - x, str(b): 100.0 - 2 * x}
    res = dict(winner=None if winner is None else str(winner), hp=hp, shots={str(a): x + 3, str(b): x + 1}, hits={str(a): x % 3, str(b): x % 2},
               seconds=seconds - x, knockout=bool(x % 4 == 1), max_levels={str(a): dict(fire=1.5, escape=2.0, run=0.5, walk=3.0),
                                                                             str(b): dict(fire=2.5, escape=1.0, run=1.5, walk=2.0)},
               events=[["shot", 1.0 + x, str(a)], ["hit", 2.0, str(b)]], frames=[[0.0, 1.0, 2.0, 0.5, 100.0, -1.0, -2.0, 1.5, 100.0, [[0.5, 0.5, 0]]]],
               steps=1000, attempt=attempt, match_seed=(a * 1000 + b + attempt) & 0x7FFFFFFF)
    if drivers:
        def part(k):
            return dict(types=TYPES, hit=np.array([60.0 + k + x, 30.0, 8.0 + x, 100.0]), base=np.array([200.0 + k, 150.0, 400.0, 900.0]),
                        n=np.array([2, 4, 30, 10]), hit_windows=3 + x % 3, steps=1000, dt=0.005)
        res["_drivers"] = {str(a): part(0), str(b): part(1)}
    return res


out = {}
cases = []
for seeds, favorite in (([1, 2, 3, 4], 3), ([10, 11, 12, 13, 14, 15, 16, 17], None), ([5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], 24)):
    b = tournament.run_bracket(seeds, seconds=20.0, mode="subtle", favorite=favorite, bracket_seed=2, play=play)
    for k in ("created", "seconds"):
        b.pop(k)
    cases.append(b)
out["brackets"] = cases

# races
races = []
cards = [card(s, "subtle") for s in (3000, 3001, 3002, 3003, 3004)]
out["odds"] = racing.odds_table(cards)
for race_seed, seeds in ((1, [3000, 3001, 3002]), (2, [3003, 3004, 3005, 3006])):
    def lane(t):
        if t[0] == "card":
            return card(t[1], t[2])
        seed, repeat, rs, mode, backend, cap = t
        fin = 20.0 + (seed * 7 + repeat * 13) % 11
        did = (seed + repeat) % 4 != 0
        return seed, repeat, dict(finish_s=fin if did else None, distance=8.0 if did else 3.0 + (seed % 3), trace=[[0.0, 0.0], [1.0, 1.0]],
                                  lures_touched=[0], max_walk_level=2.0, seconds=fin if did else 90.0, rule_version=2)
    r = racing.run_race(seeds, mode="subtle", repeats=2, race_seed=race_seed, play=lane)
    r.pop("created"), r.pop("seconds")
    races.append(r)
out["races"] = races
out["race_analysis"] = racing.analyze(races)
Path(__file__).with_name("bracket.json").write_text(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
print(len(cases), out["brackets"][0]["champion"], out["race_analysis"]["tests"])
