"""Writes tests/golden/stats.json: scipy's binomtest, spearmanr and wilcoxon on fixed inputs (what the tournament and race analyses call)."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

rng = np.random.default_rng(5)
out = dict(binom=[], spearman=[], wilcoxon=[])
for k, n in ((0, 1), (1, 1), (5, 7), (13, 15), (7, 15), (30, 60), (22, 60), (0, 10), (10, 10), (45, 100)):
    out["binom"].append(dict(k=k, n=n, p=float(stats.binomtest(k, n, 0.5).pvalue)))
for n in (5, 8, 16, 48):
    for _ in range(3):
        x = rng.normal(size=n)
        y = 0.4 * x + rng.normal(size=n)
        if _ == 2:
            x = np.round(x * 2) / 2
            y = np.round(y)
        r = stats.spearmanr(x, y)
        out["spearman"].append(dict(x=x.tolist(), y=y.tolist(), rho=float(r.statistic), p=float(r.pvalue)))
for n in (3, 6, 12, 30, 60):
    for _ in range(2):
        d = rng.normal(-0.05, 0.2, size=n)
        if _ == 1:
            d = np.round(d * 10) / 10
        try:
            p = float(stats.wilcoxon(d, alternative="less", zero_method="wilcox").pvalue)
        except ValueError:
            p = 1.0
        out["wilcoxon"].append(dict(d=d.tolist(), p=p))
Path(__file__).with_name("stats.json").write_text(json.dumps(out))
print(out["binom"][:3], out["wilcoxon"][0]["p"])

# --- tournament.analyze on a synthetic bracket set ---------------------------------------------------------------------------------
import os, sys, hashlib  # noqa: E402,E401
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.lab import tournament  # noqa: E402

r2 = np.random.default_rng(11)
brackets = []
for bi in range(3):
    seeds = [100 * bi + i for i in range(8)]
    cards = {s: dict(loom_latency_s=float(r2.uniform(0.05, 0.5)), sugar_ratio=float(r2.uniform(1, 3)),
                     turning_log_ratio=float(r2.normal(0, 0.3)), title="t") for s in seeds}
    fighters, rounds, won = list(seeds), [], {s: 0 for s in seeds}
    while len(fighters) > 1:
        ms = []
        for i in range(0, len(fighters), 2):
            a, b = fighters[i], fighters[i + 1]
            coin = bool(r2.uniform() < 0.1)
            w = a if (cards[a]["sugar_ratio"] + r2.normal(0, 1) > cards[b]["sugar_ratio"]) else b
            won[w] += 1
            ms.append(dict(a=a, b=b, winner=w, coin_toss=coin, knockout=bool(r2.uniform() < 0.5)))
        rounds.append(ms)
        fighters = [m["winner"] for m in ms]
    brackets.append(dict(seeds=seeds, cards=cards, rounds=rounds, rounds_won=won, individuality="subtle"))
an = tournament.analyze(brackets)
out["analysis"] = dict(brackets=brackets, result=an)
Path(__file__).with_name("stats.json").write_text(json.dumps(out, default=float))
print({k: an[k] for k in ("flies", "decided_matches", "coin_tosses", "knockouts", "predicts_winning")})
