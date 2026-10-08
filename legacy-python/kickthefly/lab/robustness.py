"""How much do this simulation's results depend on the reconstruction being right?

Two manipulations, one question. Both re-run the validation suite's behaviors on a changed connectome and report
which of them still pass:

  threshold_sweep    drop every connection reconstructed with fewer than N synapses, for a range of N, and report
                     the threshold at which each validated behavior breaks. A behavior that survives a high
                     threshold does not rest on weak, possibly spurious contacts.
  signflip_trials    flip the sign of neurons whose neurotransmitter prediction the dataset is least sure about,
                     over repeated randomized trials, and report how often each behavior survives, with a survival
                     rate and its 95% Wilson interval and the unperturbed run of the same seeds as the control.

Both use the validation suite's own pass criteria (validation.py), unchanged, so "breaks" means the same thing here
as on the validation dashboard. The runs are lockstep and seeded, so the same sweep gives the same answer twice.

The connectome supplies the synapse counts and the neurotransmitter predictions with their confidences. Which
thresholds to try, how many flip trials to run and how many seeds to average over are this game's choices, and the
brain pack is already filtered to connections of at least 3 synapses by the loader, so a threshold of 1-3 changes
nothing at all.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from kickthefly.sim.wiring import Wiring

# The behaviors worth asking about: the four that pass validation on the unmodified connectome. The two that fail
# there are still run when asked for, but "breaks at" means nothing for a behavior that never worked.
DEFAULT_TESTS = ("looming_escape", "sugar_feeding", "antenna_grooming_circuit", "mb_conditioning")
DEFAULT_THRESHOLDS = (1, 4, 5, 6, 8, 10)
MIN_PACK_SYNAPSES = 3            # kickthefly/sim/connectome/loader.py MIN_WEIGHT


def _results_by_id(res: dict) -> dict:
    out = {}
    for t in res["tests"]:
        m = t["measured"]
        out[t["id"]] = dict(
            passed=bool(t["passed"]), name=t["name"],
            effect=float(m.get("drive_ratio_mean", m.get("pi_mean", 0.0))),
            control=float(m.get("control_ratio_mean", m.get("control_pi_mean", 0.0))),
            p_value=float(m.get("p_value", 1.0)),
            # the ratio's denominator has a floor, so a brain that has gone quieter overall shows a bigger ratio
            # without the pathway having got any stronger: the rates say which of the two happened
            base_hz=float(m.get("readout_base_hz", float("nan"))),
            driven_hz=float(m.get("readout_driven_hz", float("nan"))),
            metric="drive/baseline ratio" if "drive_ratio_mean" in m else "performance index")
    return out


def run_behaviors(wiring: Wiring, seeds, include=DEFAULT_TESTS, workers: int | None = None, progress=None) -> dict:
    """The validation suite's behaviors, re-run on a changed connectome, with its own pass criteria."""
    from kickthefly.lab import validation

    res = validation.run(seeds=tuple(seeds), workers=workers, progress=progress, include=set(include), wiring=wiring)
    return _results_by_id(res)


def threshold_sweep(thresholds=DEFAULT_THRESHOLDS, seeds=None, include=DEFAULT_TESTS,
                    workers: int | None = None, progress=None) -> dict:
    """Run the behaviors at each minimum-synapse threshold and report where each one breaks."""
    from kickthefly.lab import validation
    from kickthefly.sim import wiring as wiring_mod

    # The validation seeds by default: the pass criteria are validation's, and a one-sided Wilcoxon over fewer than
    # seven seeds cannot reach p < 0.01 however large the effect is, so a short sweep would report false breaks.
    seeds = tuple(seeds) if seeds else validation.SEEDS
    thresholds = sorted({max(1, int(t)) for t in thresholds})
    t0 = time.time()
    steps = []
    for i, n in enumerate(thresholds):
        def step_progress(done, total, label, i=i, n=n):
            if progress:
                progress(i, len(thresholds), f"threshold {n}: {done}/{total}")

        stats = wiring_mod.threshold_stats(n)
        res = run_behaviors(Wiring(min_synapses=n), seeds, include, workers, step_progress)
        steps.append(dict(threshold=n, stats=stats, behaviors=res))
    breaks = {}
    for test_id in include:
        passed_at = [s["threshold"] for s in steps if s["behaviors"].get(test_id, {}).get("passed")]
        failed_at = [s["threshold"] for s in steps if test_id in s["behaviors"]
                     and not s["behaviors"][test_id]["passed"]]
        first_fail = None
        for s in steps:                                  # the lowest threshold at which it fails and never recovers
            if test_id in s["behaviors"] and not s["behaviors"][test_id]["passed"]:
                later = [x for x in steps if x["threshold"] >= s["threshold"] and test_id in x["behaviors"]]
                if all(not x["behaviors"][test_id]["passed"] for x in later):
                    first_fail = s["threshold"]
                    break
        breaks[test_id] = dict(breaks_at=first_fail, survived=passed_at, failed=failed_at,
                               name=steps[0]["behaviors"].get(test_id, {}).get("name", test_id) if steps else test_id)
    return dict(kind="threshold_sweep", created=time.strftime("%Y-%m-%d %H:%M:%S"),
                seconds=round(time.time() - t0, 1), seeds=list(seeds), thresholds=thresholds,
                pack_min_synapses=MIN_PACK_SYNAPSES, steps=steps, breaks=breaks)


# --- sign flips --------------------------------------------------------------------------------------------------
# The pathway behaviors by default: each trial re-runs them over every seed, and the T-maze costs about ten times as
# much per seed, so including it turns a two-minute stress test into half an hour. The Lab screen offers it anyway.
FLIP_TESTS = ("looming_escape", "sugar_feeding", "antenna_grooming_circuit")
DEFAULT_TRIALS = 5
DEFAULT_CUTOFF = 0.7
DEFAULT_SHARE = 0.5


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k of n trials; it behaves at 0/n and n/n, where a normal interval does not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def signflip_trials(cutoff: float = DEFAULT_CUTOFF, share: float = DEFAULT_SHARE, trials: int = DEFAULT_TRIALS,
                    seeds=None, include=FLIP_TESTS, workers: int | None = None, progress=None,
                    trial_seed: int = 0) -> dict:
    """Flip the sign of uncertain neurons, repeatedly, and report which behaviors survive it.

    Each trial flips a different random half (or `share`) of the neurons whose transmitter prediction the dataset is
    less than `cutoff` sure about, then re-runs the behaviors with the validation suite's own pass criteria. The
    unperturbed run of the same seeds is the control.
    """
    from kickthefly.core import simcore
    from kickthefly.lab import labstats, validation
    from kickthefly.sim import wiring as wiring_mod

    seeds = tuple(seeds) if seeds else validation.SEEDS
    g = simcore.pack()[0]
    stats = wiring_mod.confidence_stats(g, cutoff)
    t0 = time.time()
    total = trials + 1

    def sub(i, label):
        def p(done, n, _label):
            if progress:
                progress(i, total, f"{label}: {done}/{n}")
        return p

    control = run_behaviors(Wiring(), seeds, include, workers, sub(0, "unperturbed control"))
    runs = []
    for t in range(trials):
        rows = wiring_mod.random_flip(g, cutoff, share, trial_seed + t)
        res = run_behaviors(Wiring(flip_rows=rows), seeds, include, workers, sub(t + 1, f"trial {t + 1}"))
        runs.append(dict(trial=t + 1, flipped=len(rows), behaviors=res))
    per_behavior = {}
    for test_id in include:
        passed = [r for r in runs if r["behaviors"].get(test_id, {}).get("passed")]
        effects = [r["behaviors"][test_id]["effect"] for r in runs if test_id in r["behaviors"]]
        lo, hi = _wilson(len(passed), len(runs))
        ci = labstats.mean_ci(effects)
        base = control.get(test_id, {})
        per_behavior[test_id] = dict(
            name=base.get("name", test_id), metric=base.get("metric", ""),
            control_effect=base.get("effect"), control_passed=base.get("passed"),
            survived=len(passed), trials=len(runs), survival=len(passed) / max(1, len(runs)),
            survival_ci=[lo, hi], effect_mean=ci["mean"], effect_ci=[ci["lo"], ci["hi"]], effect_sd=ci["sd"],
            effects=effects,
            verdict=("survives" if len(passed) == len(runs) else
                     "breaks" if len(passed) == 0 else "sometimes breaks"))
    return dict(kind="signflip_trials", created=time.strftime("%Y-%m-%d %H:%M:%S"),
                seconds=round(time.time() - t0, 1), seeds=list(seeds), cutoff=cutoff, share=share,
                trials=trials, trial_seed=trial_seed, candidates=stats, control=control, runs=runs,
                behaviors=per_behavior)


def flips_csv(res: dict, path: Path) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["behavior", "trial", "neurons_flipped", "metric", "effect", "control_effect",
                    "readout_base_hz", "readout_driven_hz", "p_value", "passed"])
        rows = [(run["trial"], run["flipped"], run["behaviors"]) for run in res["runs"]]
        rows.append((0, 0, res["control"]))                         # trial 0 is the unperturbed control
        for trial, flipped, behaviors in rows:
            for test_id, b in behaviors.items():
                ctrl = res["control"].get(test_id, {})
                w.writerow([test_id, trial, flipped, b["metric"], f"{b['effect']:.4f}",
                            f"{ctrl.get('effect', float('nan')):.4f}", f"{b.get('base_hz', float('nan')):.4f}",
                            f"{b.get('driven_hz', float('nan')):.4f}", f"{b['p_value']:.5f}",
                            "pass" if b["passed"] else "fail"])
    return path


def sweep_csv(res: dict, path: Path) -> Path:
    """One row per behavior and threshold: what the effect was and whether it still passed."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["behavior", "threshold", "connections_dropped", "connections_dropped_share", "neurons_cut_off",
                    "metric", "effect", "control", "readout_base_hz", "readout_driven_hz", "p_value", "passed"])
        for step in res["steps"]:
            st = step["stats"]
            for test_id, b in step["behaviors"].items():
                w.writerow([test_id, step["threshold"], st["connections_dropped"],
                            f"{st['connections_dropped_share']:.4f}", st["neurons_cut_off"], b["metric"],
                            f"{b['effect']:.4f}", f"{b['control']:.4f}", f"{b.get('base_hz', float('nan')):.4f}",
                            f"{b.get('driven_hz', float('nan')):.4f}", f"{b['p_value']:.5f}",
                            "pass" if b["passed"] else "fail"])
    return path


def flip_summary(res: dict) -> str:
    c = res["candidates"]
    lines = [f"Sign-flip stress test: {res['trials']} trials, each flipping {res['share']:.0%} of the "
             f"{c['candidates']:,} neurons whose transmitter the dataset is under {res['cutoff']:.2f} sure of "
             f"({c['unknown_confidence']:,} of them have no confidence at all), seeds {res['seeds'][0]}-"
             f"{res['seeds'][-1]}, {res['seconds']}s"]
    for test_id, b in res["behaviors"].items():
        lines.append(f"  {b['name']}: {b['verdict']} ({b['survived']}/{b['trials']} trials, "
                     f"95% CI {b['survival_ci'][0]:.0%}-{b['survival_ci'][1]:.0%}; effect "
                     f"{b['effect_mean']:.2f} vs {b['control_effect']:.2f} unperturbed)")
    return "\n".join(lines)


def summary(res: dict) -> str:
    if res.get("kind") == "signflip_trials":
        return flip_summary(res)
    lines = [f"Minimum-synapse threshold sweep, seeds {res['seeds']}, {res['seconds']}s",
             f"(the brain pack already drops connections below {res['pack_min_synapses']} synapses, so thresholds "
             f"of 1-{res['pack_min_synapses']} change nothing)"]
    for test_id, b in res["breaks"].items():
        where = f"breaks at >= {b['breaks_at']} synapses" if b["breaks_at"] else "survives every threshold tried"
        last = res["steps"][-1]["behaviors"].get(test_id, {})
        first = res["steps"][0]["behaviors"].get(test_id, {})
        rates = ""
        if last.get("driven_hz") == last.get("driven_hz"):      # not NaN: a pathway test, so rates exist
            rates = (f"; readout {first['driven_hz']:.1f} -> {last['driven_hz']:.1f} spikes/s driven, "
                     f"{first['base_hz']:.1f} -> {last['base_hz']:.1f} at rest")
        lines.append(f"  {b['name']}: {where}{rates}")
    return "\n".join(lines)


def measure_firing_distribution(wiring: Wiring | None = None, seeds=(1000,), steps: int = 200) -> dict:
    """Measure firing rate distribution across the whole connectome under a given wiring."""
    import numpy as np
    from kickthefly.core import simcore

    w = wiring or Wiring()
    all_rates = []
    for s in seeds:
        br = simcore.new_brain(seed=s, warmup=150, wiring=w)
        rec = simcore.step(br, steps, record=np.arange(br.n))
        rates_hz = rec.sum(axis=0) / (steps * 0.005)
        all_rates.append(rates_hz)

    mean_rates = np.mean(all_rates, axis=0) if len(all_rates) > 1 else all_rates[0]
    bins = [0, 2, 5, 10, 20, 40, 100]
    hist, _ = np.histogram(mean_rates, bins=bins)
    bin_labels = ["<2 Hz", "2-5 Hz", "5-10 Hz", "10-20 Hz", "20-40 Hz", ">40 Hz"]
    shares = (hist / len(mean_rates)).tolist()

    return dict(
        mean=float(np.mean(mean_rates)),
        median=float(np.median(mean_rates)),
        p95=float(np.percentile(mean_rates, 95)),
        max=float(np.max(mean_rates)),
        active_share=float(np.count_nonzero(mean_rates > 0.1) / len(mean_rates)),
        histogram=hist.tolist(),
        bin_labels=bin_labels,
        bin_edges=bins,
        bin_shares=shares,
        n_neurons=int(len(mean_rates)),
    )


def inhibition_report(scale: float = 0.0, seeds=(1000,), steps: int = 200) -> dict:
    """Compare firing rate distributions before (scale=1.0) and after (scale=scale) an inhibition block."""
    import time
    from kickthefly.core import simcore
    from kickthefly.sim import wiring as wiring_mod

    g, _, _ = simcore.pack()
    stats = wiring_mod.inhibition_stats(g, scale)
    t0 = time.time()
    dist_before = measure_firing_distribution(Wiring(inhibition_scale=1.0), seeds=seeds, steps=steps)
    dist_after = measure_firing_distribution(Wiring(inhibition_scale=scale), seeds=seeds, steps=steps)

    return dict(
        kind="inhibition_block",
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        seconds=round(time.time() - t0, 1),
        scale=float(scale),
        severity=float(stats["severity"]),
        stats=stats,
        before=dist_before,
        after=dist_after,
        seeds=list(seeds),
    )


def inhibition_csv(res: dict, path: Path) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "before_100pct_inhibition", f"after_scale_{res['scale']:.2f}"])
        w.writerow(["mean_hz", f"{res['before']['mean']:.2f}", f"{res['after']['mean']:.2f}"])
        w.writerow(["median_hz", f"{res['before']['median']:.2f}", f"{res['after']['median']:.2f}"])
        w.writerow(["p95_hz", f"{res['before']['p95']:.2f}", f"{res['after']['p95']:.2f}"])
        w.writerow(["max_hz", f"{res['before']['max']:.2f}", f"{res['after']['max']:.2f}"])
        w.writerow(["active_neurons_share", f"{res['before']['active_share']:.4f}", f"{res['after']['active_share']:.4f}"])
        w.writerow([])
        w.writerow(["firing_rate_bin", "before_count", "before_share", "after_count", "after_share"])
        for lbl, bc, bs, ac, as_ in zip(res["before"]["bin_labels"],
                                        res["before"]["histogram"], res["before"]["bin_shares"],
                                        res["after"]["histogram"], res["after"]["bin_shares"]):
            w.writerow([lbl, bc, f"{bs:.4f}", ac, f"{as_:.4f}"])
    return path


def save(res: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{res['kind']}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    if res["kind"] == "threshold_sweep":
        sweep_csv(res, folder / "threshold_sweep.csv")
    elif res["kind"] == "signflip_trials":
        flips_csv(res, folder / "signflip_trials.csv")
    elif res["kind"] == "inhibition_block":
        inhibition_csv(res, folder / "inhibition_block.csv")
    return folder
