"""Golden for fly_sim::imaging: Poisson draws, time constants, kernel, ROIs, two recordings on the real brain, the CSV."""
import json, sys
sys.path.insert(0, "legacy-python")
from pathlib import Path
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab import imaging as im

out = {}
rng = np.random.default_rng(5)
lams = np.array([0.0, 0.5, 3.0, 9.9, 10.0, 12.5, 50.0, 1000.0, 1e5, 4.0, 7.7])
out["poisson"] = [[int(x) for x in rng.poisson(lams)] for _ in range(40)]
out["lams"] = [float(x) for x in lams]
out["tau"] = {k: list(im.time_constants(i)) for k, i in im.INDICATORS.items()}
out["kernel6f"] = [float(x) for x in im.kernel(im.INDICATORS["gcamp6f"], 5.0)]
br = simcore.new_brain(seed=1000)
regions = im.rois_by_region(br)
out["region_names"] = list(regions)
out["region_sizes"] = [int(len(v)) for v in regions.values()]
specs = im.rois_from_specs(br, ["type:DNp01", "prefix:KC"])
out["spec_names"] = list(specs)
out["spec_sizes"] = [int(len(v)) for v in specs.values()]
pn = im.rois_from_specs(br, ["DNp01"], per_neuron=True)
out["pn_names"] = list(pn)


def rec(rois, ind, fps, seconds, seed, **kw):
    s = im.ImagingSession(br.n, rois, ind, fps, seed=seed, **kw)
    s.prime_from_activity(br.sim.activity)
    for _ in range(int(round(seconds / im.DT_S))):
        br._step()
        s.push(np.flatnonzero(br.sim.spikes))
    r = s.result()
    delta = s.neuron_delta()
    full = np.zeros(br.n, np.float32); full[s.tracked] = np.maximum(delta, 0.0) * np.float32(1.5)
    return s, r, dict(t=r.t_s.tolist(), true=r.true_dff.tolist(), dff=r.dff.tolist(), photons=r.photons.tolist(),
                      frame_steps=s.frame_steps, fps=s.fps, tracked=int(len(s.tracked)), full_sum=float(full.astype(np.float64).sum()),
                      full_nz=int((full > 0).sum()), base_sum=float(s.base.astype(np.float64).sum()),
                      roi_true=s.roi_dff_true().tolist())


s1, r1, out["a"] = rec(regions, "gcamp6s", 20.0, 2.0, 7)
br = simcore.new_brain(seed=1000)
s2, r2, out["b"] = rec(im.rois_from_specs(br, ["type:DNp01", "prefix:KC"]), "6f", 10.0, 3.0, 3, shot_noise=False, baseline="zero")
d = Path("tests/golden/imaging_files"); d.mkdir(exist_ok=True)
im.export_csv(r2, d / "roi.csv")
json.dump(out, open("tests/golden/imaging.json", "w"))
print(out["tau"], len(out["a"]["t"]), out["a"]["tracked"], out["b"]["tracked"])
