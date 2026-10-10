"""Writes tests/golden/autopsy.json: `Game._autopsy` (kick_the_fly.py) on a scripted run: calm, a touch drive, calm, death, 3.2 s of flat-line.
The Rust `fly_sim::autopsy` must give the same rows, series and pain. Run from the repo root with legacy-python/data/kick_brain.npz."""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.core import simcore  # noqa: E402
from kickthefly.game import kick_the_fly as k  # noqa: E402

br = simcore.new_brain(seed=3, memory=False, warmup=600)
rows_head = br.sense[("head", None)]
rows_body = br.sense[("body", None)]
touch = np.concatenate([rows_head, rows_body])
simcore.step(br, 1500)
simcore.drive(br, touch, 0.3)
simcore.step(br, 600)
simcore.undrive(br, touch)
simcore.step(br, 400)
br.kill()
simcore.step(br, 800)

ds = br.death_sample
base = br.death_base
last = br.history(ds - 100, ds)
after = br.history(ds + 50, ds + 150)
names = ["whole brain"] + [n for n, _ in k.POPS] + ["head", "body", "legs", "wing", "heat", "cold", "smell", "taste", "reward", "jump", "run", "fly"]
rows = []
for n in names:
    i = br.col[n]
    b = float(base[i])
    f = float(last[:, i].mean()) if len(last) else 0.0
    a = float(after[:, i].mean()) if len(after) else 0.0
    rows.append(dict(name=n, base=b, final=f, after=a, ratio=(f + 0.05) / (b + 0.05)))
tl = br.history(ds - 1000, ds + 150)
start = max(ds - 1000, br.hist_n - k.HIST, 0)
touch_idx = [br.col[n] for n in ("head", "body", "legs", "wing")]
tsz = br.g_size[touch_idx]
series = [
    tl[:, br.col["whole brain"]] / max(float(base[br.col["whole brain"]]), 1e-3),
    (tl[:, touch_idx] * tsz).sum(1) / tsz.sum() / max(float((base[touch_idx] * tsz).sum() / tsz.sum()), 1e-3),
    tl[:, br.col["descending"]] / max(float(base[br.col["descending"]]), 1e-3),
]
pain = br.pain_index(br.hold_alarm(br.pain_parts(tl, base)))
last_pain = float(pain[max(0, ds - start - 100):ds - start].mean())
out = dict(ds=int(ds), hist_n=int(br.hist_n), pain_level=int(br.pain_level), rows=rows, n=int(len(tl)), death_at=int(ds - start),
           series=[[float(x) for x in s] for s in series], pain=[float(x) for x in pain], pain_last=last_pain)
Path(__file__).with_name("autopsy.json").write_text(json.dumps(out))
print(ds, len(tl), last_pain, max(r["ratio"] for r in rows[1:]))
