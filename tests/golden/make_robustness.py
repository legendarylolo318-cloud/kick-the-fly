"""Golden for fly_sim::{wiring,robustness}: small runs of every analysis on the real pack."""
import json, sys
sys.path.insert(0, "legacy-python")
import hashlib
import numpy as np
from kickthefly.core import simcore
from kickthefly.lab import robustness, validation
from kickthefly.sim import wiring as W

g = simcore.pack()[0]
out = {}
out["threshold_stats_5"] = W.threshold_stats(5)
out["confidence_stats_05"] = W.confidence_stats(g, 0.5)
out["inhibition_stats_05"] = W.inhibition_stats(g, 0.5)
fl = W.random_flip(g, 0.5, 0.5, 3)
out["random_flip"] = dict(n=len(fl), first=list(fl[:20]), sum=int(sum(fl)))
seeds = validation.SEEDS[:3]
inc = ("looming_escape", "sugar_feeding")
out["sweep"] = robustness.threshold_sweep(thresholds=(1, 5), seeds=seeds, include=inc, workers=1)
out["flips"] = robustness.signflip_trials(cutoff=0.7, share=0.5, trials=2, seeds=seeds, include=inc, workers=1)
out["inhib"] = robustness.inhibition_report(scale=0.5, seeds=(1000,), steps=200)
for k in ("sweep", "flips", "inhib"):
    out[k]["created"] = "GOLDEN"; out[k]["seconds"] = 0
json.dump(out, open("tests/golden/robustness.json", "w"))
print(robustness.summary(out["sweep"])); print(robustness.summary(out["flips"]))
