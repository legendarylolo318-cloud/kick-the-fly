"""Golden for fly_sim::screens: a small activation screen and a small knockout screen (singles and batches) on the real pack."""
import json, sys, shutil
sys.path.insert(0, "legacy-python")
from pathlib import Path
from kickthefly.lab import screens

root = Path("tests/golden/screens_files")
shutil.rmtree(root, ignore_errors=True)
seeds = (4000, 4001)
types = ["DNp01", "MDN", "DNp09", "DNg62", "MN9", "LPLC2"]
act = screens.run_activation(root / "act", seeds=seeds, types=types, controls=2, workers=1, steps=400, stream=None)
ko = screens.run_knockout(root / "ko", behaviors=["looming_escape", "sugar_feeding"], seeds=seeds, top=4, batch_size=2, controls=1, workers=1, stream=None)
json.dump(dict(act=act, ko=ko), open("tests/golden/screens.json", "w"), default=str)
print(act["n_types"], act["n_called"], [r["type"] + ":" + r["behavior_call"] for r in act["rows"]][:6])
print(len(ko["rows"]), ko["summary"])
