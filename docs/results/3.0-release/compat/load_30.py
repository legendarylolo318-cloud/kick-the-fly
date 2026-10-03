import os, sys, tempfile, shutil, subprocess, yaml
from pathlib import Path
A = Path.home() / "ktf_final/compat/a2131"
os.environ["SDL_VIDEODRIVER"] = "offscreen"; os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["KICK_THE_FLY_HOME"] = tempfile.mkdtemp(prefix="ktf-compat-")
sys.path.insert(0, "/home/lolo/kick-the-fly")
from kickthefly.core import config, version, pet, savestate
print("loading with", version.__version__)
work = Path(tempfile.mkdtemp()); shutil.copy(A / "config.toml", work / "config.toml")
cfg = config.Config.load(work / "config.toml")
print("config: warnings", cfg.warnings, "| seed", cfg["brain.seed"], "mode", cfg["brain.mode"], "larger", cfg["access.larger_text"], "palette", cfg["access.palette"],
      "| tutorial_done", cfg.first_run.get("tutorial_done"), "whatsnew_seen", cfg.first_run.get("whatsnew_3_0_seen"), "| conflicts", cfg.conflicts())
pd = work / "pet"; shutil.copytree(A / "pet", pd)
pm = pet.PetManager(pd); pm.load_or_create()
print("pet: seed", getattr(pm, "seed", None), "| state keys ok:", pm.exists(), "| quarantined:", [p.name for p in pd.iterdir()])
from kickthefly.lab import playthrough as pt
rig = pt.Rig(False, "cpu", seed=5, lab=True)
g = rig.game
meta = savestate.read_meta(A / "state.ktfsave")
print("save state: version", meta.get("version"), "| compatible:", savestate.compatible(meta, g) or "yes")
out = savestate.load_game(g, A / "state.ktfsave")
rig.frames(30)
print("save state loaded; game alive, flies", len(g.flies) if hasattr(g, "flies") else 1, "brain steps", g.brain.steps)
from kickthefly.lab import protocol
bad = []
for f in sorted((A / "protocols").glob("*.yaml")):
    try:
        protocol.check(yaml.safe_load(f.read_text()), f.name)
    except Exception as e:
        bad.append((f.name, repr(e)))
print("protocols:", len(list((A / "protocols").glob("*.yaml"))), "checked; refused:", bad)
