"""Run with the v2.13.1 checkout: make real 2.13.1 artifacts into ~/ktf_final/compat/a2131."""
import os, sys, tempfile, shutil, subprocess
from pathlib import Path
OUT = Path.home() / "ktf_final/compat/a2131"
shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
os.environ["SDL_VIDEODRIVER"] = "offscreen"; os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["KICK_THE_FLY_HOME"] = str(OUT / "home")
sys.path.insert(0, os.getcwd())
from kickthefly.core import config, version
print("version", version.__version__)
# 1. config.toml with player changes
cfg = config.Config(OUT / "config.toml")
cfg.set("brain.seed", 12); cfg.set("brain.mode", "lab"); cfg.set("access.larger_text", True); cfg.set("access.palette", "blue-yellow")
cfg.bind("arena", "e"); cfg.first_run["tutorial_done"] = True; cfg.dirty = True
assert cfg.save(); print("config written")
# 2. pet
from kickthefly.core import pet
pm = pet.PetManager(OUT / "pet"); pm.create_new_pet(seed=3); pm.feed(0.3); pm.record_training("tmaze"); pm.save()
print("pet:", sorted(p.name for p in (OUT / "pet").iterdir()))
# 3. save state from a real 2D game after a hit
from kickthefly.lab import playthrough as pt
rig = pt.Rig(False, "cpu", seed=5, lab=True)
g = rig.game
rig.frames(120)
from kickthefly.core import savestate
p = savestate.save_game(g, OUT / "state.ktfsave")
print("save state:", p, p.stat().st_size)
rig.close() if hasattr(rig, "close") else None
