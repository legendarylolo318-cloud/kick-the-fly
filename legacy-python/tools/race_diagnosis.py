"""Why is a fly's race time repeatable even with individuality OFF? (3.0 day 4 review, decision B). EXPLORATION SEEDS ONLY (0-299).

    python tools/race_diagnosis.py [--seeds 0-15] [--workers 3] [--out FILE.json]

The race assay (lab/racing.py) found R1 repeatability rho 0.96 with individuality 'subtle' and 0.92 with it 'off', where every fly
has the same brain. Each run of a fly is built by tournament.build_fighter: new_brain(seed) warms the brain up for 600 steps with the
seed's own noise, THEN the lane's noise is reseeded. So both runs of a fly start from the same post-warm-up state. This script asks
what in that state carries the fly's speed, with individuality off (identical wiring), each fly run twice with new lane noise:

  A  as the assay does it: both runs start from the seed's own post-warm-up state
  B  a fresh warm-up per run: new_brain(seed, warmup=0), reseed to the lane noise, THEN warm up. Nothing of the seed is left
  C  as A, but the synaptic gain (the slow gain controller's one number, sim.gain) is set to the mean over the seeds after warm-up
  D  as A, but the calm baselines (Brain.base and the fast rates the levels are read against) are set to the means over the seeds

and records, per seed, the post-warm-up gain, the walking group's calm baseline and its fast rate, against the finishing time.
Spearman rho of run 1 vs run 2 for each condition. Analysis only: nothing here changes the game, the assay or a criterion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _state(br) -> dict:
    i = br.col["walk"]
    return dict(gain=float(br.sim.gain), walk_base=float(br.base[i]), walk_fast=float(br.fast[i]),
                whole_brain_hz=float(br.fast[br.col["whole brain"]]))


def _lane(args) -> dict:
    cond, seed, repeat, common = args
    from kickthefly.core import simcore
    from kickthefly.game import flyrace
    from kickthefly.lab import racing

    noise = racing.lane_noise(0, seed, repeat)
    if cond == "B":
        br = simcore.new_brain(seed=seed, individuality="off", warmup=0)
        br.reseed(noise)
        br.warmup(600)
    else:
        br = simcore.new_brain(seed=seed, individuality="off")
        st0 = _state(br)
        br.reseed(noise)
        if cond == "C":
            br.sim.gain = float(common["gain"])
        elif cond == "D":
            br.base[:] = np.asarray(common["base"])
            br.fast[:] = np.asarray(common["fast"])
    st = _state(br)
    r = flyrace.run_lane(br)
    t = racing._effective_time(r, flyrace.TRACK_LENGTH)
    return dict(cond=cond, seed=seed, repeat=repeat, time=t, finished=r["finish_s"] is not None, state=st,
                state_before_override=st0 if cond != "B" else None)


def _warm(seed) -> dict:
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=seed, individuality="off")
    return dict(seed=seed, **_state(br), base=br.base.tolist(), fast=br.fast.tolist())


def main() -> int:
    from scipy import stats

    from kickthefly.lab import headless

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-15")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--conditions", default="ABCD")
    ap.add_argument("--out")
    a = ap.parse_args()
    seeds = headless.parse_seeds(a.seeds, range(16))
    assert all(0 <= s <= 299 for s in seeds), "exploration seeds only (0-299)"
    import multiprocessing

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        warm = [f.result() for f in [ex.submit(_warm, s) for s in seeds]]
        common = dict(gain=float(np.mean([w["gain"] for w in warm])), base=np.mean([w["base"] for w in warm], 0).tolist(),
                      fast=np.mean([w["fast"] for w in warm], 0).tolist())
        tasks = [(c, s, r, common) for c in a.conditions for s in seeds for r in (0, 1)]
        res = []
        for f in as_completed([ex.submit(_lane, t) for t in tasks]):
            res.append(f.result())
            print(f"  {len(res)}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    out = dict(seeds=seeds, seconds=round(time.time() - t0, 1), common_gain=common["gain"],
               warm=[{k: v for k, v in w.items() if k not in ("base", "fast")} for w in warm], conditions={})
    for c in a.conditions:
        t1 = [next(r["time"] for r in res if r["cond"] == c and r["seed"] == s and r["repeat"] == 0) for s in seeds]
        t2 = [next(r["time"] for r in res if r["cond"] == c and r["seed"] == s and r["repeat"] == 1) for s in seeds]
        rho, p = stats.spearmanr(t1, t2)
        out["conditions"][c] = dict(rho=float(rho), p_two_sided=float(p), times_run1=t1, times_run2=t2,
                                    unfinished=sum(1 for r in res if r["cond"] == c and not r["finished"]))
    mean_a = [np.mean([r["time"] for r in res if r["cond"] == "A" and r["seed"] == s]) for s in seeds] if "A" in a.conditions else None
    if mean_a is not None:
        for k in ("gain", "walk_base", "walk_fast", "whole_brain_hz"):
            xs = [w[k] for w in warm]
            rho, p = stats.spearmanr(xs, mean_a)
            out.setdefault("state_vs_time_A", {})[k] = dict(rho=float(rho), p_two_sided=float(p), values=xs)
    text = json.dumps(dict(out, lanes=res), indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(f"seeds {seeds[0]}-{seeds[-1]} (n = {len(seeds)}), individuality off, {out['seconds']:.0f} s")
    for c, d in out["conditions"].items():
        print(f"  {c}: run 1 vs run 2 rho = {d['rho']:+.2f} (p = {d['p_two_sided']:.3g}), unfinished lanes {d['unfinished']}")
    for k, d in out.get("state_vs_time_A", {}).items():
        print(f"  post-warm-up {k} vs mean finishing time (A): rho = {d['rho']:+.2f} (p = {d['p_two_sided']:.3g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
