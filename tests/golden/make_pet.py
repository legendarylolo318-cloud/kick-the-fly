"""Golden for ktf_data::pet: a scripted pet life through core/pet.py with a fake clock and a scripted random()."""
import json, sys, tempfile, time, random
from pathlib import Path
sys.path.insert(0, "legacy-python")
from kickthefly.core import pet as P

clock = [1_700_000_000.0]
time.time = lambda: clock[0]
rolls = iter([0.9, 0.001, 0.2, 0.003, 0.7, 0.004] * 100)
random.random = lambda: next(rolls)

d = Path(tempfile.mkdtemp())
m = P.PetManager(data_dir=d)
steps = []
def snap(tag):
    steps.append(dict(tag=tag, hunger=m.hunger, sleep=m.sleep_pressure, mood=m.mood, sleeping=m.is_sleeping, dead=m.is_dead,
                      welcome=m.welcome_message, events=[e["event_type"] + ":" + e["description"] for e in m.timeline]))

m.create_new_pet(seed=42, brain_type="adult"); snap("new")
clock[0] += 3 * 3600; m.load_or_create(); snap("away3h")
m.feed(0.2, "sugar"); snap("fed")
clock[0] += 30 * 60; m.load_or_create(); snap("away30m")
clock[0] += 10 * 3600; m.load_or_create(); snap("away10h")
m.real_stakes = True; m.hunger = 1.0
clock[0] += 3 * 86400; m.load_or_create(); snap("away3d_stakes")
m2 = P.PetManager(data_dir=Path(tempfile.mkdtemp())); m2.create_new_pet(seed=1)
m2.hunger, m2.sleep_pressure, m2.mood = 0.4, 0.88, 0.5
seq = []
for i in range(12):
    m2.update_live(dt=0.5 + i, is_night=bool(i % 5 == 4), pam_rate=2.0 + i * 0.1, ppl1_rate=1.0)
    seq.append([m2.hunger, m2.sleep_pressure, m2.mood, m2.is_sleeping])
m2.record_injury("a test"); m2.record_training("tmaze")
json.dump(dict(steps=steps, live=seq, live_rolls=[0.9, 0.001, 0.2, 0.003, 0.7, 0.004] * 2,
               after=dict(mood=m2.mood, events=[e["event_type"] for e in m2.timeline]),
               mult=[m2.get_sugar_gain_multiplier(), m2.get_pam_reward_multiplier(), m2.get_dfb_sleep_drive()]),
          open("tests/golden/pet.json", "w"), indent=1)
print("ok", len(steps))
