"""Golden for game::lab_extra::hum_demo_fly: the hum demo on the real pack (individuality off)."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.lab import audio

r = audio.demo_fly(1000, conditions=("silence", "pulses_200_ipi35", "steady_50"), seconds=0.6)
json.dump(r, open("tests/golden/hum.json", "w"), default=float)
print({c: round(v["p1"]["ratio"], 3) for c, v in r["conditions"].items()})
