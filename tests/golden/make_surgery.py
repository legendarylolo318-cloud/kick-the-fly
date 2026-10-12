"""Writes tests/golden/surgery.json: the neuron rows of every entry of the brain surgery screen (`kick_the_fly.SURGERY`, `Game._surgery_rows`),
from the real Python code on the real pack: count and SHA-256 of the row list per entry. Run from the repo root with legacy-python/data/kick_brain.npz."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.core import simcore  # noqa: E402
from kickthefly.game import kick_the_fly as k  # noqa: E402
from kickthefly.lab import lesions  # noqa: E402

br = simcore.new_brain(seed=1, memory=False, warmup=0)
g = br.graph


def rows_of(spec):          # Game._surgery_rows
    kind, names = spec
    if kind == "group":
        return np.flatnonzero(np.isin(br.det_id, [br.col[n] for n in names]))
    if kind == "pop":
        pops = [n for n, _ in k.POPS]
        return np.flatnonzero(np.isin(br.pop_id, [pops.index(n) for n in names]))
    if kind == "type":
        return np.flatnonzero(np.isin(br.types, names))
    if kind == "prefix":
        m = np.zeros(br.n, bool)
        for pre in names:
            m |= np.char.startswith(br.types, pre)
        return np.flatnonzero(m)
    if kind == "hemifield_visual":
        return lesions.hemifield_visual_rows(g, side=names)
    if kind == "hemisphere":
        return lesions.hemisphere_rows(g, side=names)
    return np.arange(br.n)


out = []
for label, spec in k.SURGERY:
    r = rows_of(spec).astype(np.int64)
    out.append(dict(label=label, n=int(len(r)), sha=hashlib.sha256(r.tobytes()).hexdigest(), first=[int(x) for x in r[:5]]))
Path(__file__).with_name("surgery.json").write_text(json.dumps(out, indent=1))
print(len(out), [o["n"] for o in out])
