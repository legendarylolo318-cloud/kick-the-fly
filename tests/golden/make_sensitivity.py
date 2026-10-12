"""Golden for fly_sim::sensitivity: noise at two values and a synapse threshold on two behaviors, 3 seeds."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.lab import sensitivity, validation

res = sensitivity.run(params=["noise_std", "min_synapses"], tests=["looming_escape", "sugar_feeding"], seeds=validation.SEEDS[:3],
                      values={"noise_std": [0.0375, 0.075], "min_synapses": [5]}, workers=1)
res["created"] = "GOLDEN"
res["seconds"] = 0
for c in res["cells"]:
    c["seconds"] = 0
json.dump(res, open("tests/golden/sensitivity.json", "w"))
print(sensitivity.summary(res))
