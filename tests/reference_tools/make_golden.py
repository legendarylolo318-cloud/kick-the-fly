#!/usr/bin/env python3
"""Dump reference outputs of the Python simulator for the Rust port's golden tests.

    python legacy-python/make_golden.py [--pack data/kick_brain.npz] [--out tests/golden]

Runs the Python LIF simulator (kickthefly/sim/connectome/sim.py, the NumPy reference backend) with fixed seeds through a set of
stimulus scenarios and writes, per scenario, what the Rust port has to reproduce:

  * the scenario itself (seed, steps, the stimulus schedule with explicit neuron rows),
  * for every step: the number of spikes, a SHA-256 of the sorted spike indices (so the WHOLE raster is checked, not a sample),
    the gain controller's value, and the spikes of a monitored subset of neurons (explicit indices),
  * at the end: the membrane potentials of the monitored neurons (float32 bit patterns) and checksums of the full state.

It also writes `manifest.json` (versions, pack SHA-256, the LIF parameters, the stated tolerances) and `rng.json` (the first values of
NumPy's PCG64 / ziggurat streams that the sim uses, a hash of the noise bank, the individuality gains) so that a failure can be traced to
the random generator, to the pack loader or to the step itself.

The scenarios drive the SIMULATOR (`LIFSim.step(sensory_input)`), exactly as `Brain._step` does, with the currents `Brain` would build
(STIM_AMP 0.5 for stimuli, -0.4 * ramp for the death/sedation inhibition, SURGERY_CURRENT for surgery). Nothing in the game layer
(pygame, groups, plasticity) is involved, which is why this script needs only numpy and scipy.

The script looks for the `kickthefly` package next to itself (after the Python tree is moved into legacy-python/) and one level up
(while it still lives in the repo root), so it works before and after the move.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent):
    if (cand / "kickthefly" / "sim").is_dir():
        sys.path.insert(0, str(cand))
        PY_ROOT = cand
        break
else:  # pragma: no cover
    raise SystemExit("cannot find the kickthefly package next to or above this script")
REPO = HERE.parent

os.environ["KICK_THE_FLY_SIM_BACKEND"] = "cpu"          # the NumPy reference backend; Numba and torch-cpu are bit-exact with it
os.environ.setdefault("KICK_THE_FLY_SIM_DTYPE", "float32")
os.environ.setdefault("KICK_THE_FLY_INDIVIDUALITY", "off")

import numpy as np  # noqa: E402

STIM_AMP = 0.5                     # game/kick_the_fly.py
SURGERY_SILENCE = -0.6             # SURGERY_CURRENT[-1]
INHIB = -0.4                       # Brain._inhib
TOL = {
    "spikes": "bit-exact: per-step SHA-256 of the full spike set must match for every step",
    "gain": "relative 1e-12 (f64; Rust exp() may differ from NumPy's by 1 ulp)",
    "v": "bit-exact float32 for the monitored neurons",
    "v_sum": "relative 1e-9 (summation order differs; the per-neuron values above are exact)",
}


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def f32_bits(x) -> list[int]:
    return [int(b) for b in np.asarray(x, np.float32).view(np.uint32)]


def rows_of(g, spec: str) -> np.ndarray:
    """Subset of core/simcore.py:rows_of that the scenarios use (no Brain object needed)."""
    types = np.asarray(g.type).astype(str)
    sc = np.asarray(g.superclass).astype(str)
    kind, _, arg = spec.partition(":")
    if kind == "type":
        return np.flatnonzero(np.isin(types, arg.split(",")))
    if kind == "prefix":
        m = np.zeros(len(types), bool)
        for p in arg.split(","):
            m |= np.char.startswith(types, p)
        return np.flatnonzero(m)
    if kind == "superclass":
        return np.flatnonzero(np.isin(sc, arg.split(",")))
    raise ValueError(spec)


def scenarios(g) -> list[dict]:
    rng = np.random.default_rng(20261008)
    n = len(g.type)
    mon = np.unique(np.concatenate([
        rows_of(g, "type:DNp01,DNp09,MDN,DNa01,DNa02"), rows_of(g, "type:LPLC2")[:40], rows_of(g, "prefix:PAM")[:40],
        rows_of(g, "type:MN9"), rng.choice(n, 384, replace=False)]))
    touch_pool = rows_of(g, "prefix:BM_,JO-")
    touch_rows = np.sort(rng.choice(touch_pool, size=max(1, int(0.3 * len(touch_pool))), replace=False))
    surg = rows_of(g, "type:LC4")
    return [
        dict(name="calm_seed1000", seed=1000, steps=400, inputs=[]),
        dict(name="calm_seed7", seed=7, steps=300, inputs=[]),
        dict(name="loom_drive", seed=1000, steps=360,
             inputs=[dict(start=160, end=360, op="set", amp=STIM_AMP, spec="type:LPLC2,LC4")]),
        dict(name="touch_pulses", seed=1001, steps=320,
             inputs=[dict(start=s, end=s + 8, op="set", amp=STIM_AMP, rows=touch_rows.tolist(), spec="rows") for s in (100, 180, 260)]),
        dict(name="reward_and_surgery", seed=42, steps=320,
             inputs=[dict(start=0, end=320, op="add", amp=SURGERY_SILENCE, spec="type:LC4"),
                     dict(start=100, end=320, op="set", amp=STIM_AMP, spec="prefix:PAM"),
                     dict(start=200, end=320, op="add", amp=STIM_AMP, spec="type:MN9")]),
        dict(name="death_ramp", seed=3, steps=520, params_at={150: {"gain_adapt": 0.0}},
             inputs=[dict(start=150, end=520, op="ramp_all", amp=INHIB, ramp_steps=300)]),
    ], mon


def sens_for(sc: dict, t: int, n: int, resolved: list[np.ndarray]):
    """The sensory vector Brain._step would hand to LIFSim.step at step t, or None."""
    cur = None
    for inp, rows in zip(sc["inputs"], resolved):
        if not (inp["start"] <= t < inp["end"]):
            continue
        if cur is None:
            cur = np.zeros(n, np.float32)
        if inp["op"] == "set":
            cur[rows] = np.float32(inp["amp"])
        elif inp["op"] == "add":
            cur[rows] += np.float32(inp["amp"])
        elif inp["op"] == "ramp_all":
            ramp = min(1.0, max(0.0, (t - inp["start"]) / inp["ramp_steps"]))
            cur = np.full(n, np.float32(INHIB), np.float32) * np.float32(ramp)
        else:
            raise ValueError(inp["op"])
    return cur


def run_scenario(sc: dict, W, g, mon: np.ndarray, LIFParams, LIFSim) -> dict:
    n = W.shape[0]
    p = LIFParams()
    sim = LIFSim(None, p, seed=sc["seed"], W_in=W)
    resolved = []
    for inp in sc["inputs"]:
        if inp["op"] == "ramp_all":
            resolved.append(np.arange(n))
        elif inp.get("rows") is not None:
            resolved.append(np.asarray(inp["rows"], np.int64))
        else:
            resolved.append(rows_of(g, inp["spec"]))
            inp["n_rows"] = int(len(resolved[-1]))
            inp["rows_sha"] = sha(resolved[-1].astype("<i8"))
    steps = []
    for t in range(sc["steps"]):
        for k, v in sc.get("params_at", {}).get(t, {}).items():
            setattr(sim.p, k, v)
        sens = sens_for(sc, t, n, resolved)
        spikes = sim.step(sens)
        idx = np.flatnonzero(spikes)
        steps.append(dict(n=int(len(idx)), sha=sha(idx.astype("<u4")), gain=float(sim.gain),
                          mon=np.intersect1d(idx, mon).tolist()))
    return dict(
        name=sc["name"], seed=sc["seed"], steps=sc["steps"], inputs=sc["inputs"],
        params_at={str(k): v for k, v in sc.get("params_at", {}).items()},
        per_step=steps,
        end=dict(v_mon_bits=f32_bits(sim.v[mon]), v_sha=sha(sim.v), refr_sha=sha(sim.refr), v_sum=float(np.sum(sim.v, dtype=np.float64)),
                 spike_total=int(sim.spike_total), gain=float(sim.gain),
                 rate_ema_sha=sha(sim.activity.rates()), rate_ema_sum=float(np.sum(sim.activity.rates(), dtype=np.float64))),
    )


def rng_section(n: int, noise_std: float) -> dict:
    out = {"seeds": {}}
    for seed in (0, 3, 7, 42, 1000, 1001):
        r = np.random.default_rng(seed)
        bank = (r.standard_normal(n * 16, dtype=np.float32) * np.float32(noise_std))
        offs = [int(r.integers(0, bank.size - n)) for _ in range(32)]
        out["seeds"][str(seed)] = dict(
            bank_first_bits=f32_bits(bank[:16]), bank_last_bits=f32_bits(bank[-16:]), bank_sha=sha(bank),
            offsets=offs)
    r = np.random.default_rng(1000)
    out["u64_first"] = [int(x) for x in r.bit_generator.random_raw(8)]
    r = np.random.default_rng(1000)
    out["normal_f64_first_bits"] = [int(x) for x in r.standard_normal(16).view(np.uint64)]
    # individuality gains (core/individuality.py:compute_fly_gains), sigma 0.05 and 0.15
    from kickthefly.core import individuality as ind
    out["individuality"] = {}
    for mode in ("subtle", "strong"):
        pre, post = ind.compute_fly_gains(1000, n, mode)
        out["individuality"][mode] = dict(pre_first_bits=f32_bits(pre[:16]), post_first_bits=f32_bits(post[:16]),
                                          pre_sha=sha(pre), post_sha=sha(post))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", type=Path, default=None, help="kick_brain.npz (default: data/ in the repo, or KICK_THE_FLY_PACK)")
    ap.add_argument("--out", type=Path, default=REPO / "tests" / "golden")
    args = ap.parse_args()
    pack = args.pack or Path(os.environ.get("KICK_THE_FLY_PACK", REPO / "data" / "kick_brain.npz"))
    if not pack.exists():
        raise SystemExit(f"brain pack not found: {pack}")

    import scipy
    from kickthefly.sim import brainpack
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    g, W, _ = brainpack.load(pack)
    n = W.shape[0]
    args.out.mkdir(parents=True, exist_ok=True)
    scs, mon = scenarios(g)
    p = LIFParams()
    manifest = dict(
        format="ktf-golden", version=1, numpy=np.__version__, scipy=scipy.__version__, python=sys.version.split()[0],
        pack=dict(name=pack.name, sha256=hashlib.sha256(pack.read_bytes()).hexdigest(), n=int(n), nnz=int(W.nnz),
                  w_data_sha=sha(W.data), indptr_sha=sha(W.indptr), indices_sha=sha(W.indices),
                  type_sha=hashlib.sha256("\n".join(np.asarray(g.type).astype(str)).encode()).hexdigest()[:16]),
        params={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(p).items()
                if k in ("dt_ms", "tau_ms", "v_thresh", "v_reset", "refractory_steps", "syn_gain", "ext_gain", "bias", "noise_std",
                         "target_rate_hz", "gain_adapt", "gain_bounds", "sparse_path_max_active", "dtype")},
        monitor=mon.tolist(), scenarios=[s["name"] for s in scs], tolerances=TOL,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "rng.json").write_text(json.dumps(rng_section(n, p.noise_std), indent=1))
    for sc in scs:
        res = run_scenario(sc, W, g, mon, LIFParams, LIFSim)
        (args.out / f"{sc['name']}.json").write_text(json.dumps(res, separators=(",", ":")))
        tot = sum(s["n"] for s in res["per_step"])
        print(f"[golden] {sc['name']:22s} seed {sc['seed']:5d} steps {sc['steps']:4d} mean spikes/step {tot / sc['steps']:8.1f} "
              f"final gain {res['end']['gain']:.4f}")
    print(f"[golden] wrote {len(scs)} scenarios + manifest.json + rng.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
