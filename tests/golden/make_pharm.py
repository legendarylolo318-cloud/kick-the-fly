"""Golden for fly_sim::pharmacology: counts by confidence level, the wiring a drug makes, and a short dose-response."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.core import simcore
from kickthefly.lab import pharmacology as ph

g = simcore.pack()[0]
out = {}
for name, dose, low, cut in (("picrotoxin", 0.5, True, 0.7), ("cholinergic", 0.25, False, 0.7), ("gabaa_agonist", 1.0, False, 0.9), ("glucl", 0.5, True, 0.7)):
    out[f"{name}:{dose}:{low}:{cut}"] = ph.affected(g, name, dose, low, cut)
w = ph.wiring_for({"picrotoxin": 0.5, "glucl": 0.5, "gabaa_agonist": 1.0}, False, 0.8)
out["wiring"] = dict(min_synapses=w.min_synapses, nt_scales=[list(x) for x in w.nt_scales], nt_min_conf=w.nt_min_conf)
out["dose_response"] = ph.dose_response("picrotoxin", doses=(0.0, 0.5, 1.0), seeds=(1000,), steps=100)
json.dump(out, open("tests/golden/pharm.json", "w"), default=str)
print(out["wiring"], [r["mean_hz"] for r in out["dose_response"]["rows"]])
