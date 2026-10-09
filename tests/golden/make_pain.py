"""Writes tests/golden/pain.json: the pain index of `Brain.pain_parts` / `pain_index` (kick_the_fly.py) evaluated with NumPy for random group
rates, using the group sizes of tests/golden/groups.json. Run: python3 tests/golden/make_pain.py"""
import json
import numpy as np

sizes = json.load(open("tests/golden/groups.json"))["sizes"]
sc = np.load("data/kick_brain.npz")["superclass"].astype(str)
sizes = dict(sizes, ascending=int((sc == "ascending_neuron").sum()), descending=int(np.isin(sc, ("descending_neuron", "descending_neuron_tbc")).sum()))
names = list(sizes)
TOUCH = ("head", "body", "legs", "wing")
W = np.array([0.55, 0.25, 0.55, 0.20, 0.30])
col = {n: i for i, n in enumerate(names)}
g_size = np.array([sizes[n] for n in names], float)
rng = np.random.default_rng(5)
cases = []
for level in (0, 1, 2):
    for k in range(6):
        fast = (rng.random(len(names)) * rng.choice([2, 30, 80])).astype(np.float32)
        base = (rng.random(len(names)) * 6).astype(np.float32)
        rates = fast[None, :]

        def above(ns, full):
            idx = [col[n] for n in ns]
            sz = g_size[idx]
            return ((rates[:, idx] - base[idx]) * sz).sum(1) / sz.sum() / full

        touch = above(list(TOUCH) + (["body_extra"] if level else []), 35.0)
        thermal = np.maximum(above(["heat"], 30.0), above(["cold"], 30.0))
        chemical = (above(["smell", "taste"], 1.0) - 8.0) / 27.0
        d = col["descending"]
        alarm = (rates[:, d] / max(float(base[d]), 1.0) - 1.06) / 0.3
        a = col["ascending"]
        relay = (rates[:, a] / max(float(base[a]), 1.0) - 1.06) / 0.35 if level >= 2 else np.zeros(1)
        parts = np.clip(np.stack([touch, thermal, chemical, alarm, relay], 1), 0, 1)[0]
        idx = list(TOUCH) + ["heat", "cold", "smell", "taste"] + (["body_extra"] if level >= 1 else []) + (["ascending"] if level >= 2 else [])
        cases.append(dict(level=level, fast={n: float(fast[col[n]]) for n in names}, base={n: float(base[col[n]]) for n in names},
                          parts=[float(x) for x in parts], index=float(min(100.0, 100.0 * (parts @ W))),
                          neurons=int(sum(sizes[n] for n in idx))))
json.dump(cases, open("tests/golden/pain.json", "w"), indent=0)
print(len(cases), "cases")
