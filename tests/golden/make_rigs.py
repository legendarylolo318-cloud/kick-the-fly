"""Golden for fly_sim::rigs: per-fly rig measurements and scene runs on the real pack, the analysis and the assay summary text."""
import json, sys
sys.path.insert(0, "legacy-python")
import numpy as np
from kickthefly.lab import rigassay as ra

out = {}
# individuality "off": with "subtle" the per-neuron gains differ in the last bit between NumPy's and libm's exp, and a brain amplifies that
M = "off"
out["tethered"] = ra.tethered_fly(1000, M)
out["ball"] = ra.ball_fly(1000, M)
out["buridan"] = ra.buridan_fly(1000, M, seconds=20.0)
out["fourfield"] = ra.fourfield_fly(1000, M, seconds=20.0)
scenes = {}
for name, rig, seed, opt in [
    ("tethered_closed", "tethered", 1001, dict(mode="closed", omega=2.0, seconds=9.0, gain=1.0)),
    ("tethered_open", "tethered", 1001, dict(mode="open", omega=3.0, seconds=9.0)),
    ("ball_bar", "ball", 1002, dict(seconds=10.0, scene="bar", bar_deg=-90.0)),
    ("ball_pano", "ball", 1002, dict(seconds=10.0, scene="panorama", omega=1.0, mode="open")),
    ("buridan", "buridan", 1003, dict(seconds=15.0)),
    ("fourfield", "fourfield", 1004, dict(seconds=15.0, mode="odor")),
]:
    r = ra.scene_run(rig, seed, M, **opt)
    a = np.asarray(r["trace"], float)
    scenes[name] = dict(rig=rig, seed=seed, opt=opt, summary=r["summary"], seconds=r["seconds"], n=len(a), trace_sum=a.sum(axis=0).tolist(),
                        first=a[:3].tolist(), last=a[-3:].tolist(), params=r["params"], geometry=r["geometry"])
out["scenes"] = scenes
# analysis and the summary text on synthetic flies
rng = np.random.default_rng(3)
fake = {
    "tethered": lambda s: dict(seed=s, d_directional=float(4 + rng.normal()), compensation_mean=float(0.12 + 0.05 * rng.normal())),
    "ball": lambda s: dict(seed=s, compensation_mean=float(0.11 + 0.05 * rng.normal()), bar_error_deg_visible=float(12 + rng.normal()),
                           bar_error_deg_hidden=float(40 + 5 * rng.normal())),
    "buridan": lambda s: dict(seed=s, stripes=dict(deviation_deg=float(20 + 3 * rng.normal()), transits=int(rng.integers(4, 9))),
                              none=dict(deviation_deg=float(44 + 3 * rng.normal()), transits=int(rng.integers(0, 3)))),
    "fourfield": lambda s: dict(seed=s, odor=dict(pi=float(0.02 * rng.normal()), speed_odor=0.3, speed_air=0.31),
                                sham=dict(pi=float(0.02 * rng.normal()))),
}
an = {}
for rig in ra.rigs.RIGS:
    flies = [fake[rig](1000 + i) for i in range(8)]
    an[rig] = dict(flies=flies, analysis=ra.analyze(rig, flies))
out["analysis"] = an
res = ra.run_assay("buridan", tuple(range(1000, 1006)), play=lambda t: fake["buridan"](t[1]))
out["summary_text"] = ra.summary(res)
out["assay"] = res
json.dump(out, open("tests/golden/rigs.json", "w"), default=str)
print(out["tethered"]["d_directional"], out["scenes"]["buridan"]["summary"])
