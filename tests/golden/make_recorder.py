"""Golden for ktf_data::recorder: a Recorder on a real brain (individuality off, seed 7): 300 steps, events and kinematics, saved."""
import json, sys, shutil
sys.path.insert(0, "legacy-python")
from pathlib import Path
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab import assays, recorder

d = Path("tests/golden/recorder_files")
shutil.rmtree(d, ignore_errors=True)
br = simcore.new_brain(seed=7)
g = assays.groups(br)
rec = recorder.Recorder(br, {"dn": g["dn"], "dnp01": g["dnp01"], "loom": g["loom"], "empty": np.zeros(0, np.int64)}).start()
for k in range(300):
    if k == 50:
        br.poke("head", "R", 0.6)
    if k == 100:
        rec.log_kinematics(0.5, 0.25, 0.0, 0.1, 100.0, "walk", "kitchen")
    if k == 200:
        rec.log_event("note", "JUMP", "JUMP  head touch", 0.0)
    br._step()
rec.stop()
files = rec.save(d / "recording", {"recorded_live": True})
print([f.name for f in files], len(rec.rows), rec.arrays()["spike_steps"].shape)
json.dump(dict(brain_steps=int(br.steps), n_rows=len(rec.rows)), open("tests/golden/recorder.json", "w"))
