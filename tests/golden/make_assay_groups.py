"""Writes tests/golden/assay_groups.json: the `sweet` and `bitter` taste-neuron sets of `lab/assays.py:groups` (ranked by one- and two-synapse
reach onto the sugar and bitter pathways), from the real Python code. Run from the repo root with legacy-python/data/kick_brain.npz in place."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.core import simcore  # noqa: E402
from kickthefly.lab import assays  # noqa: E402

br = simcore.new_brain(seed=1, memory=False, warmup=0)
g = assays.groups(br)
out = {k: [int(x) for x in g[k]] for k in ("sweet", "bitter", "head_gust", "mn9", "sugar_pn", "bitter_sel")}
out["SWEET_N"] = assays.SWEET_N
Path(__file__).with_name("assay_groups.json").write_text(json.dumps(out))
print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()})
