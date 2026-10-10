"""Writes tests/golden/assay_jobs.json: lab/labjobs.run_sync (the Lab's assays page) for T-maze, looming and sugar over two flies, with and
without a surgery, on the real pack. The Rust fly_sim::assay_jobs must give the same summaries and comparisons."""
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.lab import labjobs  # noqa: E402


def clean(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        return clean(o.item())
    return o


out = {}
seeds = [2000, 2001]
out["tmaze"] = labjobs.run_sync("tmaze", seeds, workers=1)
out["looming"] = labjobs.run_sync("looming", seeds, options=dict(speeds=(1.4, 5.5), approaches=2), workers=1)
out["sugar"] = labjobs.run_sync("sugar", seeds, options=dict(doses=(0.0, 0.2, 1.0), repeats=2), workers=1)
out["looming_surgery"] = labjobs.run_sync("looming", seeds, options=dict(speeds=(1.4, 5.5), approaches=2), surgery={"type:DNp01": -1}, workers=1)
for k in out:
    out[k].pop("flies", None)
    out[k].pop("control_flies", None)
Path(__file__).with_name("assay_jobs.json").write_text(json.dumps(clean(out)))
print({k: v["treated"].get("metric") for k, v in out.items()})
