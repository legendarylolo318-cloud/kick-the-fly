"""Golden for fly_sim::patchclamp: protocol maths, an isolated run, an embedded run on a fresh brain, an I-F curve and the CSVs."""
import json, sys, io
sys.path.insert(0, "legacy-python")
from pathlib import Path
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab import patchclamp as pc
from kickthefly.sim.connectome.sim import LIFParams

out = {}
proto = pc.ClampProtocol([0.0, 0.1], duration_ms=100, pre_ms=50, post_ms=50, repeats=2, gap_ms=25, holding=0.01)
out["protocol"] = dict(total=proto.total_steps(), windows=proto.sweep_windows(), sched_sum=float(proto.schedule().sum()), n=len(proto.schedule()))
iso = pc.isolated(proto.schedule(), LIFParams(), seed=3)
out["isolated"] = dict(v_sum=float(iso.v.astype(np.float64).sum()), v_last=float(iso.v[-1]), spikes=int(iso.spikes.sum()),
                       spike_idx=[int(i) for i in np.flatnonzero(iso.spikes)][:30], v_head=[float(x) for x in iso.v[:12]])
br = simcore.new_brain(seed=1000)
row = pc.pick_neuron(br, "type:DNp01", 0)
emb = pc.run_current_clamp(br, row, pc.ClampProtocol([0.0, 0.5], duration_ms=100, pre_ms=50, post_ms=50, gap_ms=0), "embedded")
out["row"] = int(row)
out["embedded"] = dict(v_sum=float(emb.v.astype(np.float64).sum()), spikes=int(emb.spikes.sum()), spike_idx=[int(i) for i in np.flatnonzero(emb.spikes)],
                       meta=emb.meta["neuron"])
curve = pc.if_curve(None, None, [0.0, 0.05, 0.1, 0.5], 200.0, 3, "isolated", 5, LIFParams())
out["curve"] = {k: v for k, v in curve.items() if k != "recording"}
out["curve"]["sweeps"] = {str(k): v for k, v in curve["sweeps"].items()}
d = Path("tests/golden/patch_files"); d.mkdir(exist_ok=True)
pc.export_csv(emb, d / "trace.csv"); pc.export_if_csv(curve, d / "curve.csv")
json.dump(out, open("tests/golden/patch.json", "w"), default=str)
print(out["protocol"]["total"], out["isolated"]["spikes"], out["embedded"]["spikes"], out["curve"]["rate_hz"])
