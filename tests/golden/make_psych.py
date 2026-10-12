"""Golden for fly_sim::psychometrics: a short sweep of each target (2 x values, 2 flies) and its three files."""
import json, sys
sys.path.insert(0, "legacy-python")
from pathlib import Path
from kickthefly.lab import psychometrics as P

out = {}
d = Path("tests/golden/psych_files"); d.mkdir(exist_ok=True)
for target, xs in (("looming", [1.0, 5.5]), ("sugar", [0.0, 0.4]), ("tmaze", [1, 2])):
    r = P.run_sweep(target, x_values=xs, n_flies=2, base_seed=1000)
    r["created"] = "GOLDEN"; r["seconds"] = 0
    r["caption"] = f"Kick the Fly v{r['app_version']} | Target: {r['spec']['label']} | Seeds: 1000..1001 (n=2) | Default LIF sim (dt=5ms, tau=20ms) | Generated: GOLDEN"
    P.save_all_formats(r, f"psych_{target}", d)
    out[target] = r
json.dump(out, open("tests/golden/psych.json", "w"))
print({k: [p["primary"] for p in v["points"]] for k, v in out.items()})
