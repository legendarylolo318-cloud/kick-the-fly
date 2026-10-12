"""Golden for fly_sim::genetics on the real pack."""
import json, sys
sys.path.insert(0, "legacy-python")
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab import genetics as G

g = simcore.pack()[0]
types = g.type
out = dict(coverage=G.coverage(types))
out["describe"] = {n: G.describe(n, types) for n in ("SS00727", "ss00727", "SS02299")}
out["search"] = {q: [l.name for l in G.search(q)] for q in ("DNp01", "ss0", "LC10", "zzz")}
rows = G.rows_of_lines(types, "line:SS00727")
out["rows"] = dict(n=int(len(rows)), sum=int(rows.sum()))
out["lines_for_type"] = [l.name for l in G.lines_for_type("DNp01")][:10]
json.dump(out, open("tests/golden/genetics.json", "w"))
print(out["coverage"], out["rows"])
