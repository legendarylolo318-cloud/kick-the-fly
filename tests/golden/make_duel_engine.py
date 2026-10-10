"""Writes tests/golden/duel_engine.json: game/flyduel.run_duel on deterministic fake brains (so the engine, its numpy random numbers and
its driver analysis are compared without any brain noise). The Rust fly_sim::duel must give identical frames, events and tables."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.game import flyduel  # noqa: E402

TYPES = ["DNp35", "DNa01", "DNa02", "LC10", "LC11", "DNp01", "SNta13", "KCg", "MBON01", "DNp09"]


class Fake:
    dt = 0.005

    def __init__(self, k, n=240):
        self.k, self.n, self.s, self.hurt = k, n, 0, -10**9
        self.types = np.array([TYPES[i % len(TYPES)] for i in range(n)])
        self.sim = types.SimpleNamespace(spikes=np.zeros(n, bool))
        self.log = []

    def poke(self, region, side, strength, recruit=None):
        self.log.append([region, side, float(strength), recruit is not None])
        if region == "body":
            self.hurt = self.s

    def _step(self):
        self.s += 1
        self.sim.spikes = ((self.s * 7 + np.arange(self.n) * 13 + self.k) % 11) == 0

    def _tri(self, p, ph=0):
        return ((self.s + ph) % p) / p

    def level(self, name):
        if name == "fire":
            return 0.5 + 6.0 * self._tri(97 + 3 * self.k, 5 * self.k)
        if name == "escape":
            return 0.5 + 7.0 * self._tri(61 + self.k, 11)
        if name == "run":
            return 0.5 + 4.0 * self._tri(43, 3 * self.k) + (3.0 if self.s - self.hurt < 100 else 0.0)
        if name == "walk":
            return 0.5 + 7.0 * self._tri(151 - 4 * self.k)
        return 1.0

    def hz(self, name):
        if name == "turn_r":
            return 6.0 + 4.0 * self._tri(37 + self.k, 2)
        if name == "turn_l":
            return 1.0 + 0.5 * self._tri(53 + self.k, 7)
        return 0.0


out = []
for seed, secs in ((11, 8.0), (4242, 12.0), (7, 5.0), (91, 20.0), (303, 20.0), (555, 20.0)):
    a, b = Fake(0), Fake(1)
    r = flyduel.run_duel(a, b, seed=seed, seconds=secs, names=("A", "B"), drivers=True)
    d = r.pop("_drivers")
    sums = {k: flyduel.combine_drivers([v]) for k, v in d.items()}
    jd = {k: dict(hit_windows=v["hit_windows"], steps=v["steps"], types=list(v["types"]), hit=list(map(float, v["hit"])),
                  base=list(map(float, v["base"])), n=list(map(int, v["n"]))) for k, v in d.items()}
    out.append(dict(seed=seed, seconds=secs, result=r, drivers=jd,
                    top={k: flyduel.top_drivers(v, min_base=1, min_hit=1, min_windows=1) for k, v in sums.items()},
                    top_default={k: flyduel.top_drivers(v) for k, v in sums.items()},
                    named={k: flyduel.named_drivers(v) for k, v in sums.items()},
                    pokes=[a.log, b.log]))
    print(seed, r["winner"], r["hp"], r["shots"], r["hits"], len(r["frames"]), len(r["events"]))
Path(__file__).with_name("duel_engine.json").write_text(json.dumps(out, default=float))
