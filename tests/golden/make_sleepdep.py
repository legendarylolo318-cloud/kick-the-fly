"""Golden for fly_sim::sleepdep: the assay on seeds 1000-1002 (one worker; a few minutes)."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.lab import sleepdep

res = sleepdep.run([1000, 1001, 1002], workers=1)
res["created"] = "GOLDEN"
res["seconds"] = 0
json.dump(res, open("tests/golden/sleepdep.json", "w"))
print(sleepdep.summary(res))
