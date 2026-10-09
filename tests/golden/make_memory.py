"""Writes tests/golden/memory.json: a scripted training run on the real pack with the Python `core/memory.py` (isolated, untrained start),
so crates/fly_sim/tests/memory_golden.rs can check the Rust plasticity against it.
Run from the repo root with legacy-python/data/kick_brain.npz in place:  python3 tests/golden/make_memory.py

The scenario drives neurons directly (no random pokes), so both sides see the same spikes: settle 1200 steps, then 6 fear trials on smell A
(KCs 0..699 at 0.35 for 120 steps, plus the PPL1 punishment neurons at 0.8 for 100 steps, rest 130), a test of smell B (other KCs, 120 steps),
4 liking trials on A (PAM reward neurons at 0.8: the reversal), a test of A. After each trial: memory of A (fear, like), weight sum,
share weakened, the approach MBONs' response."""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.core import simcore  # noqa: E402

br = simcore.new_brain(seed=5, memory=True, isolated_memory=True, warmup=600)
mem = br.memory
kc = mem.kc
A, B = kc[:700], kc[1500:2200]
types = np.asarray(br.graph.type).astype(str)
ppl1 = np.array([r for r in mem.dan if types[r].startswith("PPL1")])
pam = np.array([r for r in mem.dan if not types[r].startswith("PPL1")])


def run(steps, drives, observe=None):
    for rows, amp in drives:
        simcore.drive(br, rows, amp)
    mbon = []
    for i in range(steps):
        br._step()
        if observe and i % 3 == 0:
            mem.observe(observe, br.sim.activity.rates())
        if i >= steps - 40:
            mbon.append(mem.mbon_response(br.sim.activity.rates()))
    for rows, _ in drives:
        simcore.undrive(br, rows)
    return float(np.mean(mbon))


checks = []


def check(label, scent, mbon_hz):
    fear, like = mem.memory_of(scent)
    checks.append(dict(label=label, scent=scent, fear=float(fear), like=float(like), w_sum=float(mem.w.sum(dtype=np.float64)),
                       w_min=float(mem.w.min()), weakened=mem.weakened_share(), mbon_hz=mbon_hz, updates=mem.updates,
                       templates=sorted(mem.templates)))


run(1200, [])
for t in range(6):
    run(120, [(A, 0.35)], "A")
    hz = run(100, [(A, 0.35), (ppl1, 0.8)], "A")
    run(130, [])
    check(f"fear{t + 1}", "A", hz)
hz = run(120, [(B, 0.35)], "B")
check("test_B", "B", hz)
for t in range(4):
    run(120, [(A, 0.35)], "A")
    hz = run(100, [(A, 0.35), (pam, 0.8)], "A")
    run(130, [])
    check(f"like{t + 1}", "A", hz)
hz = run(120, [(A, 0.35)], "A")
check("test_A", "A", hz)
out = dict(seed=5, n_kc=len(kc), n_mbon=len(mem.mbon), n_syn=len(mem.w0), signature=[float(x) for x in mem.signature],
           punish_comps=int(mem.punish_comp.sum()), checks=checks, w_sample=[float(x) for x in mem.w[::997][:60]],
           kc_calm_sample=[float(x) for x in mem.kc_calm[::97][:40]], dan_calm_sample=[float(x) for x in mem.dan_calm[:40]],
           template_A_sample=[float(x) for x in mem.templates["A"][::97][:40]])
Path(__file__).with_name("memory.json").write_text(json.dumps(out, indent=1))
for c in checks:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k != "templates"})
