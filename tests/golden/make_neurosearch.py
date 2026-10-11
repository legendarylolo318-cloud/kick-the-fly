"""Golden for fly_sim::neurosearch: search queries and top_paths on the real pack (lab/neurosearch.py)."""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, "legacy-python")
from kickthefly.core import simcore
from kickthefly.lab import neurosearch as ns

br = simcore.new_brain(seed=0, memory=False, warmup=0)
queries = ["DNp01", "dnp01", "LPLC2", "lc4", "KC", "720575940", "123", "P1", "GF", "xyzzy", " ", "pC1", "_L", "AVLP727m"]
bid = int(br.body_id[int(np.flatnonzero(br.types == "DNp01")[0])]) if hasattr(br, "types") else 0
queries.append(str(bid))
out = {"queries": [[q, ns.search(br, q), ns.search(br, q, limit=3)] for q in queries]}
W, Wc = br.sim.W_csr, br.sim.W_csc
rng = np.random.default_rng(5)
pairs = []
dn = int(np.flatnonzero(br.types == "DNp01")[0])
a, b = W.indptr[dn], W.indptr[dn + 1]
ins = W.indices[a:b]
pairs += [(int(s), dn) for s in ins[:3]]
for _ in range(40):
    s = int(rng.integers(br.n))
    outs = Wc.indices[Wc.indptr[s]:Wc.indptr[s + 1]]
    if not len(outs):
        continue
    m = int(rng.choice(outs))
    o2 = Wc.indices[Wc.indptr[m]:Wc.indptr[m + 1]]
    if not len(o2):
        continue
    m2 = int(rng.choice(o2))
    o3 = Wc.indices[Wc.indptr[m2]:Wc.indptr[m2 + 1]]
    pairs.append((s, m2))
    if len(o3):
        pairs.append((s, int(rng.choice(o3))))
    if len(pairs) > 24:
        break
pairs.append((dn, dn))
out["paths"] = [dict(src=s, dst=d, k=5, paths=ns.top_paths(W, s, d, k=5, Wc=Wc)) for s, d in pairs]
json.dump(out, open("tests/golden/neurosearch.json", "w"), default=float)
print(len(out["paths"]), sum(1 for p in out["paths"] if p["paths"]))
