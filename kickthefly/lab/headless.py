"""Headless runs: no window, no sound, no display needed (SSH, CI, a Windows console).

    KickTheFly --headless --validate [--out results.json] [--workers N] [--seeds 1000-1009]
    KickTheFly --headless --protocol experiment.yaml [--out folder]

The exe and the AppImage accept the same flags. On Windows the exe borrows the console it was started from for its
output; from cmd use `start /wait KickTheFly.exe --headless ...` (or check the files written to --out).
Exit codes: 0 success; 1 a validation result differs from the expected pass/fail (only with --strict); 2 bad input.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from kickthefly.core.crash import log


def prepare() -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"          # nothing in a headless run may open a window or an audio device
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    from kickthefly.core import platform_env

    platform_env.attach_console()


def parse_seeds(text: str | None, default):
    if not text:
        return tuple(default)
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return tuple(out)


def run_validate(args) -> int:
    from kickthefly.lab import validation

    seeds = parse_seeds(args.seeds, validation.SEEDS)
    t0 = time.time()

    def progress(done, total, label):
        print(f"  {done}/{total} ({time.time() - t0:.0f}s)", flush=True)

    res = validation.run(seeds=seeds, workers=args.workers, progress=progress)
    out = Path(args.out) if args.out else validation.local_path()
    if out.suffix.lower() != ".json":
        out = out / validation.RESULTS_NAME
    validation.save_results(res, out)
    print(validation.summary(res))
    print(f"results written to {out}")
    if args.strict:
        wrong = [t["id"] for t in res["tests"] if t["passed"] != validation.EXPECTED.get(t["id"])]
        if wrong:
            print(f"differs from the expected results: {', '.join(wrong)}")
            return 1
    return 0


def run_threshold_sweep(args) -> int:
    """--threshold-sweep: which validated behaviors survive dropping weakly reconstructed connections."""
    from kickthefly.lab import recorder, robustness, validation

    seeds = parse_seeds(args.seeds, validation.SEEDS)
    thresholds = tuple(args.thresholds) if getattr(args, "thresholds", None) else robustness.DEFAULT_THRESHOLDS
    t0 = time.time()
    res = robustness.threshold_sweep(thresholds=thresholds, seeds=seeds, workers=args.workers,
                                     progress=lambda d, n, label: print(f"  {d}/{n} {label} "
                                                                        f"({time.time() - t0:.0f}s)", flush=True))
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-threshold"
    robustness.save(res, folder)
    print(robustness.summary(res))
    print(f"results written to {folder}")
    return 0


def run_signflip(args) -> int:
    """--signflip-test: which validated behaviors survive flipping uncertain neurotransmitter signs."""
    from kickthefly.lab import recorder, robustness, validation

    seeds = parse_seeds(args.seeds, validation.SEEDS)
    t0 = time.time()
    res = robustness.signflip_trials(
        cutoff=getattr(args, "flip_confidence", None) or robustness.DEFAULT_CUTOFF,
        share=getattr(args, "flip_share", None) or robustness.DEFAULT_SHARE,
        trials=getattr(args, "trials", None) or robustness.DEFAULT_TRIALS,
        seeds=seeds, workers=args.workers,
        progress=lambda d, n, label: print(f"  {d}/{n} {label} ({time.time() - t0:.0f}s)", flush=True))
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-signflip"
    robustness.save(res, folder)
    print(robustness.summary(res))
    print(f"results written to {folder}")
    return 0


def run_critical_path(args) -> int:
    """--critical-path TARGET: silence each candidate cell type in turn and rank them by effect size."""
    from kickthefly.lab import criticalpath, labjobs, recorder, validation

    target = args.critical_path
    if target not in validation.BY_ID and target not in labjobs.ASSAYS:
        print(f"error: unknown target {target!r}; use one of "
              f"{sorted(validation.BY_ID) + list(labjobs.ASSAYS)}", file=sys.stderr)
        return 2
    seeds = parse_seeds(args.seeds, validation.SEEDS)
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"critical-path-{target}"
    if folder.exists() and not getattr(args, "resume", False) and (folder / "critical_path_progress.json").exists():
        print(f"note: {folder} holds an unfinished run; pass --resume to continue it, or --out elsewhere to start "
              f"fresh", file=sys.stderr)
    t0 = time.time()
    last = [0.0]

    def progress(done, total, label):
        if time.time() - last[0] > 2 or done == total:      # a type takes seconds; don't spam a line per fly
            last[0] = time.time()
            print(f"  {done}/{total} {label} ({time.time() - t0:.0f}s)", flush=True)

    res = criticalpath.run(target, seeds=seeds, top=getattr(args, "top", None) or criticalpath.DEFAULT_TOP,
                           workers=args.workers, progress=progress,
                           resume=folder if getattr(args, "resume", False) else folder,
                           types=getattr(args, "types", None))
    criticalpath.save(res, folder)
    print(criticalpath.summary(res))
    print(f"results written to {folder}")
    return 0


def audit_asymmetry(seconds: float = 5.0, seed: int = 0, mirror: bool = False) -> dict:
    """Audit bilateral asymmetry between left and right hemibrains:
    - Measures baseline turning bias with no input over a calm run
    - Compares L vs R synapse in/out counts and firing rates for key cell types:
      DNa01, DNa02, LC10, LPLC2, LC4, DNp01.
    """
    import numpy as np
    from kickthefly.core import simcore
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    g, W, _ = simcore.pack()
    if mirror:
        W = simcore.symmetrize_weights(g, W)
    inst = g.instance.astype(str)
    types = g.type.astype(str)
    W_csr = W.tocsr()
    W_csc = W.tocsc()

    sim = LIFSim(None, LIFParams(), W_in=W, seed=seed)
    for _ in range(100):
        sim.step()

    steps = int(round(seconds / 0.005))
    spikes = np.zeros(g.n, dtype=np.int32)
    for _ in range(steps):
        sim.step()
        spikes += sim.spikes
    rates = spikes / float(seconds)

    key_types = ["DNa01", "DNa02", "LC10", "LPLC2", "LC4", "DNp01"]
    records = []
    for t in key_types:
        tm = np.char.startswith(types, t)
        l_idx = np.flatnonzero(tm & np.char.endswith(inst, "_L"))
        r_idx = np.flatnonzero(tm & np.char.endswith(inst, "_R"))
        l_n, r_n = len(l_idx), len(r_idx)
        l_in = float(sum(W_csr[i].nnz for i in l_idx) / l_n) if l_n else 0.0
        r_in = float(sum(W_csr[i].nnz for i in r_idx) / r_n) if r_n else 0.0
        l_out = float(sum(W_csc[:, i].nnz for i in l_idx) / l_n) if l_n else 0.0
        r_out = float(sum(W_csc[:, i].nnz for i in r_idx) / r_n) if r_n else 0.0
        l_hz = float(rates[l_idx].mean()) if l_n else 0.0
        r_hz = float(rates[r_idx].mean()) if r_n else 0.0
        records.append({
            "type": t,
            "l_count": l_n, "r_count": r_n,
            "l_in_syn": round(l_in, 1), "r_in_syn": round(r_in, 1),
            "l_out_syn": round(l_out, 1), "r_out_syn": round(r_out, 1),
            "l_rate_hz": round(l_hz, 2), "r_rate_hz": round(r_hz, 2),
            "diff_rate_hz": round(r_hz - l_hz, 2),
        })

    dna_r = rates[np.flatnonzero(np.isin(types, ("DNa01", "DNa02")) & np.char.endswith(inst, "_R"))].mean()
    dna_l = rates[np.flatnonzero(np.isin(types, ("DNa01", "DNa02")) & np.char.endswith(inst, "_L"))].mean()
    turn_bias = float(dna_r - dna_l)

    return {
        "seconds": seconds,
        "seed": seed,
        "mirror": mirror,
        "turning_bias_hz": round(turn_bias, 3),
        "turning_direction": "right" if turn_bias > 0.05 else "left" if turn_bias < -0.05 else "neutral",
        "records": records,
    }


def format_asymmetry_report(res: dict) -> str:
    lines = [
        f"LEFT/RIGHT ASYMMETRY AUDIT ({res['seconds']:.1f} s calm run, seed={res['seed']}, mirror={res['mirror']})",
        "-" * 88,
        f"{'Cell Type':<9} {'Count L/R':<11} {'In-Syn L/R':<15} {'Out-Syn L/R':<15} {'Firing Rate L/R':<18} {'Diff (R-L)':<10}",
        "-" * 88,
    ]
    for r in res["records"]:
        counts = f"{r['l_count']}/{r['r_count']}"
        in_s = f"{r['l_in_syn']:.0f}/{r['r_in_syn']:.0f}"
        out_s = f"{r['l_out_syn']:.0f}/{r['r_out_syn']:.0f}"
        rates_s = f"{r['l_rate_hz']:.1f}/{r['r_rate_hz']:.1f} Hz"
        diff_s = f"{r['diff_rate_hz']:+.2f} Hz"
        lines.append(f"{r['type']:<9} {counts:<11} {in_s:<15} {out_s:<15} {rates_s:<18} {diff_s:<10}")
    lines.append("-" * 88)
    bias = res["turning_bias_hz"]
    direction = res["turning_direction"]
    lines.append(f"Baseline Steering Bias (DNa01/02 R - L): {bias:+.2f} Hz ({direction} turn bias)")
    if res["mirror"]:
        lines.append("Note: Mirror-averaged weights active [GAME RULE: data modification].")
    else:
        lines.append("Note: Raw connectome weights. Asymmetries stem from both biology and EM reconstruction depth.")
    return "\n".join(lines)


def run_audit_asymmetry(args) -> int:
    import json
    mirror = getattr(args, "mirror_weights", False)
    seed = getattr(args, "seed", None) or 0
    res = audit_asymmetry(seconds=5.0, seed=seed, mirror=mirror)
    print(format_asymmetry_report(res))
    if getattr(args, "out", None):
        out_path = Path(args.out)
        if out_path.is_dir() or not out_path.suffix:
            out_path = out_path / "asymmetry_audit.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))
        print(f"audit results written to {out_path}")
    return 0


def run_benchmark(args) -> int:
    from kickthefly.lab import benchmark
    flies = args.flies if getattr(args, "flies", None) else (1, 8, 16)
    seconds = getattr(args, "seconds", None) or 5.0
    backend = os.environ.get("KICK_THE_FLY_SIM_BACKEND", "auto")
    res = benchmark.run_benchmark(fly_counts=tuple(flies), seconds=seconds, backend=backend)
    print(benchmark.format_benchmark_report(res))
    out_p = benchmark.save_benchmark_results(res, getattr(args, "out", None))
    print(f"Results written to {out_p}")
    return 0


def run_headless_replay(args) -> int:
    """--headless --replay FILE --out DIR: re-exports the recording."""
    from kickthefly.core import replay, simcore
    from kickthefly.lab import recorder
    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"error: replay file not found: {replay_path}", file=sys.stderr)
        return 2
    try:
        player = replay.ReplayPlayer.load(replay_path)
    except replay.ReplayError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else recorder.exports_dir() / f"replay-{replay_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    br = simcore.new_brain(seed=player.seed)
    why = replay.check_compatibility(player.meta, replay.pack_signature(br))
    if why:
        print(f"error: incompatible replay: {why}", file=sys.stderr)
        return 2

    rec = recorder.Recorder(br)
    for step in range(player.total_steps + 1):
        for ev in player.events_at(step):
            if ev["type"] == "poke":
                br.poke(ev["sense"], ev.get("target"), ev.get("val", 0.5))
        br._step()
        rec.step(br)
    rec.export_csv(out_dir / "replay_spikes.csv")
    print(f"Replay re-exported to {out_dir}")
    return 0


def main(args) -> int:
    prepare()
    from kickthefly.sim.connectome import backends
    choice = getattr(args, "sim_backend", None) or getattr(args, "backend", None)
    if choice in backends.BACKEND_NAMES:                # inherited by the validation/assay worker processes
        os.environ["KICK_THE_FLY_SIM_BACKEND"] = choice
    if getattr(args, "dtype", None):
        os.environ["KICK_THE_FLY_SIM_DTYPE"] = args.dtype
    log.info("headless run (simulation backend: %s)", os.environ.get("KICK_THE_FLY_SIM_BACKEND", "auto"))
    try:
        if getattr(args, "replay", None):
            return run_headless_replay(args)
        if getattr(args, "benchmark", False):
            return run_benchmark(args)
        if getattr(args, "audit_asymmetry", False):
            return run_audit_asymmetry(args)
        if getattr(args, "threshold_sweep", False):
            return run_threshold_sweep(args)
        if getattr(args, "signflip_test", False):
            return run_signflip(args)
        if getattr(args, "critical_path", None):
            return run_critical_path(args)
        if args.validate:
            return run_validate(args)
        if args.protocol:
            from kickthefly.lab import protocol

            return protocol.run_file(Path(args.protocol), Path(args.out) if args.out else None, workers=args.workers,
                                     nwb=bool(getattr(args, "nwb", False)))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("nothing to do: use --validate, --protocol FILE, --replay FILE, --audit-asymmetry, --benchmark, "
          "--threshold-sweep, --signflip-test or --critical-path TARGET", file=sys.stderr)
    return 2


