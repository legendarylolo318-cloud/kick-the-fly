"""Golden for fly_sim::{lesions,diffmode}: hemifield rows, and two diffs (a hemifield lesion under looming, a threshold under sugar)."""
import json, sys
sys.path.insert(0, "legacy-python")
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab import diffmode, lesions
from kickthefly.sim.wiring import Wiring

g, _, soma = simcore.pack()
out = {}
vis = lesions.hemifield_visual_rows(g, "L")
hemi = lesions.hemisphere_rows(g, "L", soma=soma)
out["vis_L"] = dict(n=int(len(vis)), sum=int(vis.sum()))
out["hemi_L"] = dict(n=int(len(hemi)), sum=int(hemi.sum()))
out["vis_R"] = dict(n=int(len(lesions.hemifield_visual_rows(g, "R"))))
a = dict(wiring=Wiring(), lesion_rows=None, label="Fly A (Unperturbed)")
out["looming"] = diffmode.run_connectome_diff(a, dict(wiring=Wiring(), lesion_rows=vis, label="Fly B (hemi_vis)"), stimulus="looming", steps=200, seed=1000)
out["sugar"] = diffmode.run_connectome_diff(a, dict(wiring=Wiring(min_synapses=10), lesion_rows=None, label="Fly B (thresh_10)"), stimulus="sugar", steps=200, seed=1000)
for k in ("looming", "sugar"):
    out[k]["created"] = "GOLDEN"; out[k]["seconds"] = 0
json.dump(out, open("tests/golden/diff.json", "w"), default=str)
print(out["looming"]["readouts"], out["vis_L"], out["hemi_L"])
