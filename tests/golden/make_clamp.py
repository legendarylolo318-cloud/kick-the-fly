"""Golden for fly_sim::clamp: a looming reference replayed with a synapse threshold and with a DNp01 lesion, and a sugar one with no inhibition."""
import json, sys
sys.path.insert(0, "legacy-python")
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab import clamp
from kickthefly.sim.wiring import Wiring

g = simcore.pack()[0]
types = g.type.astype(str)
out = {}
ref = clamp.record_reference_run("looming", steps=200, seed=1000)
out["ref_region_rates"] = ref["region_rates"]
out["thresh"] = clamp.run_neural_clamp(ref, wiring=Wiring(min_synapses=10), seed=1000)
out["lesion"] = clamp.run_neural_clamp(ref, lesion_rows=np.flatnonzero(types == "DNp01"), seed=1000)
ref2 = clamp.record_reference_run("sugar", steps=200, seed=1000)
out["inhib"] = clamp.run_neural_clamp(ref2, wiring=Wiring(inhibition_scale=0.0), seed=1000)
for k in ("thresh", "lesion", "inhib"):
    out[k]["created"] = "GOLDEN"; out[k]["seconds"] = 0
json.dump(out, open("tests/golden/clamp.json", "w"))
print(out["thresh"]["readouts"])
