"""Golden for fly_sim::criticalpath: a behavior (shortlist top 4, 3 seeds) and an assay with one named type."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.core import simcore
from kickthefly.lab import criticalpath

out = {}
br = simcore.new_brain(seed=1000, warmup=0)
out["shortlist"] = criticalpath.shortlist(br, "looming_escape", 12)
out["sugar_shortlist"] = criticalpath.shortlist(br, "sugar_feeding", 8)
del br
out["behavior"] = criticalpath.run("looming_escape", seeds=(1000, 1001, 1002), top=4, workers=1)
out["assay"] = criticalpath.run("sugar", seeds=(1000, 1001), types=["MN9"], workers=1)
for k in ("behavior", "assay"):
    out[k]["created"] = "GOLDEN"; out[k]["seconds"] = 0
json.dump(out, open("tests/golden/critical.json", "w"))
print(criticalpath.summary(out["behavior"]))
