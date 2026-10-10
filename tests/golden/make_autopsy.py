"""Golden for fly_sim::failure_autopsy: three failing tests on the real pack."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.lab import failure_autopsy as fa
from kickthefly.core import simcore
out = {}
for tid in ("mdn_backward", "grooming_hierarchy", "mb_extinction"):
    a = fa.autopsy(tid)
    a["created"] = "GOLDEN"
    out[tid] = a
    (open(f"tests/golden/autopsy_{tid}.md", "w")).write(fa.to_markdown(a))
json.dump(out, open("tests/golden/autopsy.json", "w"), default=str)
print({k: [s["verdict"]["text"] for s in v["sections"]] for k, v in out.items()})
