"""Golden for fly_sim::thermogenetics: the escape assay on 2 seeds (3 temperatures), its summary, and the effector curves."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.lab import labjobs, thermogenetics as tg

temps = [22.0, 28.0, 32.0]
flies = [labjobs.assay_task("thermo_escape", s, dict(temps=temps), None, None) for s in (1000, 1001)]
out = dict(flies=flies, summary=labjobs.summarize("thermo_escape", flies),
           fractions={e: [tg.EFFECTORS[e].target_fraction(t) for t in (20, 25, 26.5, 28, 29, 30, 35)] for e in tg.EFFECTORS},
           arena=[tg.arena_temperature(t) for t in (-2, -1, 0, 0.3, 1)])
json.dump(out, open("tests/golden/thermo.json", "w"), default=str)
print(out["summary"]["rows"][1]["escapes"], out["arena"])
