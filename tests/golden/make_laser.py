"""Golden for fly_sim::laser::resolve_rows: row counts and sums for a range of target specs on the real pack."""
import json, sys
sys.path.insert(0, "legacy-python")
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab.laser import LaserState, DEFAULT_TARGETS

br = simcore.new_brain(seed=1000, warmup=0)
out = {}
for t in list(DEFAULT_TARGETS) + ["type:LC4,LPLC2", "prefix:DNa", "side:R:DNa02", "side:l:LC10", "xyz", "line:SS00727", "p1", "  MN9 "]:
    ls = LaserState(target_type=t)
    rows = ls.resolve_target_rows(br)
    out[t] = dict(n=int(len(rows)), sum=int(np.sum(rows)) if len(rows) else 0)
json.dump(out, open("tests/golden/laser.json", "w"))
print(out)
