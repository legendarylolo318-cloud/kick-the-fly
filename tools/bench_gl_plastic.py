"""Issue #2: what the gl backend uploads when the mushroom body learns, before and after, and which way is faster.

    python tools/bench_gl_plastic.py [--pairings 10] [--repeat 5]

Runs a real conditioning session (odor + shock, the T-maze assay's own presentation) on a gl-backed brain, records
every set of KC -> MBON synapses a learning update changed, then replays those sets through each upload path and times
it up to ctx.finish(), so the GPU's side of the copy is counted too:
  full     the old path: the whole W_csr.data buffer (41 MB) on every update
  runs     GLBackend.PLASTIC_UPLOAD = "runs": merged contiguous runs, one buffer.write(offset=...) each
  scatter  GLBackend.PLASTIC_UPLOAD = "scatter": (index, value) pairs and a compute shader
Not imported by the game.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairings", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from kickthefly.core import simcore
    from kickthefly.lab import assays

    br = simcore.new_brain(seed=a.seed, backend="gl")
    be = br.sim.backend
    if be.name != "gl":
        print(f"gl backend unavailable here (got {be.name}); nothing to measure")
        return 1
    sets = []
    orig = be.on_weights_changed

    def record(positions=None):
        if positions is not None:
            sets.append(np.asarray(positions).copy())
        return orig(positions)

    be.on_weights_changed = record
    t0 = time.perf_counter()
    assays.rest(br, 500)
    for _ in range(a.pairings):
        assays.present(br, ["odor_a"], 240, shock=True)
        assays.rest(br, 160)
    be.on_weights_changed = orig
    run_s = time.perf_counter() - t0
    steps = 500 + a.pairings * 400
    sizes = np.array([len(x) for x in sets])
    fear, _ = br.memory.memory_of("odor_a")
    print(f"{a.pairings} pairings, {steps} steps ({steps * 0.005:.0f} s brain time) in {run_s:.1f} s; "
          f"{len(sets)} learning updates changed weights; fear of the trained odor {fear:.2f}")
    print(f"changed synapses per update: median {int(np.median(sizes))}, max {sizes.max()} of {len(br.memory.w)}")

    def timed(fn):
        with be.ctx:
            be.ctx.finish()
            t = time.perf_counter()
            fn()
            be.ctx.finish()
            return time.perf_counter() - t

    data = br.sim.W_csr.data
    full_b = np.ascontiguousarray(data, dtype=np.float32).nbytes
    res = {}
    res["full"] = [timed(lambda: be.buf_values.write(np.ascontiguousarray(data, dtype=np.float32))) for _ in range(a.repeat)]
    for mode in ("runs", "scatter"):
        be.PLASTIC_UPLOAD = mode
        times, nbytes = [], 0
        for _ in range(a.repeat):
            for pos in sets:
                before = be.upload_stats["part_bytes"]
                times.append(timed(lambda: be.on_weights_changed(pos)))
                nbytes = be.upload_stats["part_bytes"] - before
        res[mode] = times
        be.upload_stats[f"{mode}_last_bytes"] = nbytes
    worst = np.sort(br.memory.csr_pos)
    for mode in ("runs", "scatter"):
        be.PLASTIC_UPLOAD = mode
        res[mode + "_all"] = [timed(lambda: be.on_weights_changed(worst)) for _ in range(a.repeat)]
    be.PLASTIC_UPLOAD = type(be).PLASTIC_UPLOAD
    per_update = lambda k: 1000 * float(np.median(res[k]))
    n_upd = len(sets)
    print(f"\n{'path':10s} {'bytes/update':>14s} {'ms/update (median)':>20s} {'all 41,495 plastic':>20s}")
    print(f"{'full':10s} {full_b:>14,d} {per_update('full'):>20.3f} {'-':>20s}")
    for mode in ("runs", "scatter"):
        b = _bytes(be, sets, mode)
        print(f"{mode:10s} {int(np.median(b)):>14,d} {per_update(mode):>20.3f} {per_update(mode + '_all'):>17.3f} ms")
    print(f"\nlearning updates in this session: {n_upd}; bytes uploaded: full {full_b * n_upd / 1e6:,.0f} MB, "
          f"runs {sum(_bytes(be, sets, 'runs')) / 1e6:.2f} MB, scatter {sum(_bytes(be, sets, 'scatter')) / 1e6:.2f} MB")
    return 0


def _bytes(be, sets, mode):
    out = []
    for pos in sets:
        if mode == "scatter":
            out.append(8 * len(pos))
        else:
            brk = np.flatnonzero(np.diff(pos) > be.RUN_GAP)
            starts = np.concatenate(([pos[0]], pos[brk + 1]))
            ends = np.concatenate((pos[brk], [pos[-1]])) + 1
            out.append(4 * int((ends - starts).sum()))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
