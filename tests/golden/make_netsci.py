"""Golden for fly_sim::netsci: the whole analysis on the real pack (a few minutes). Run from the repo root with the Python venv."""
import json, sys
sys.path.insert(0, "legacy-python")
import numpy as np
from kickthefly.lab import netsci

res = netsci.compute("adult", cache=False, force=True)
res["created"] = "GOLDEN"
res["seconds"] = 0
json.dump(res, open("tests/golden/netsci.json", "w"), allow_nan=True)
print(netsci.summary(res))
