"""Connectome diff mode: compare two flies with different configurations from the same seed and inputs.

Runs Fly A (control/reference) and Fly B (perturbed) in lockstep, diffing activity region by region,
reusing the autopsy comparison format (diverging bars) and tracking the timeline when they diverge.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kickthefly.core import simcore
from kickthefly.sim.wiring import Wiring


def run_connectome_diff(
    config_a: dict | None = None,
    config_b: dict | None = None,
    stimulus: str = "looming",
    steps: int = 200,
    seed: int = 1000,
) -> dict:
    """Run two flies with different configurations from the exact same seed and inputs.

    config_a / config_b: dict with optional keys:
      - 'wiring': Wiring object
      - 'lesion_rows': 1D array of neuron rows to silence
      - 'label': string name
    """
    t0 = time.time()
    g, _, _ = simcore.pack()
    cfg_a = config_a or {"wiring": Wiring(), "lesion_rows": None, "label": "Fly A (Unperturbed control)"}
    cfg_b = config_b or {"wiring": Wiring(), "lesion_rows": None, "label": "Fly B (Modified)"}

    # Initialize both brains identically
    br_a = simcore.new_brain(seed=seed, warmup=150, wiring=cfg_a.get("wiring"))
    if cfg_a.get("lesion_rows") is not None and len(cfg_a["lesion_rows"]):
        br_a.set_override(cfg_a["lesion_rows"], -1)

    br_b = simcore.new_brain(seed=seed, warmup=150, wiring=cfg_b.get("wiring"))
    if cfg_b.get("lesion_rows") is not None and len(cfg_b["lesion_rows"]):
        br_b.set_override(cfg_b["lesion_rows"], -1)

    # Get regions
    regions = np.asarray(getattr(br_a, "region", np.full(br_a.n, "unassigned"))).astype(str)
    unique_regions = sorted(set(regions))
    reg_masks = {reg: np.flatnonzero(regions == reg) for reg in unique_regions if np.any(regions == reg)}

    # Stimulus setup
    if stimulus == "looming":
        stim_rows = br_a.sense.get(("loom", None), np.zeros(0, np.int64))
        simcore.drive(br_a, stim_rows, amp=0.5)
        simcore.drive(br_b, stim_rows, amp=0.5)
    elif stimulus == "sugar":
        stim_rows = br_a.sense.get(("sweet", None), np.zeros(0, np.int64))
        simcore.drive(br_a, stim_rows, amp=0.5)
        simcore.drive(br_b, stim_rows, amp=0.5)
    elif stimulus == "antenna":
        stim_rows = br_a.sense.get(("head", None), np.zeros(0, np.int64))
        simcore.drive(br_a, stim_rows, amp=0.5)
        simcore.drive(br_b, stim_rows, amp=0.5)
    else:
        stim_rows = None

    # Step both brains synchronously and record timeline
    timeline_a = {reg: [] for reg in reg_masks}
    timeline_b = {reg: [] for reg in reg_masks}
    divergence_history = []
    diverge_step = None
    diverge_threshold = 1.5  # spikes/s divergence threshold across regions

    bin_size = 5  # 25 ms bins for timeline smoothing
    n_bins = steps // bin_size
    
    for b in range(n_bins):
        sp_a_bin = np.zeros(br_a.n, np.int32)
        sp_b_bin = np.zeros(br_b.n, np.int32)
        for _ in range(bin_size):
            br_a._step()
            br_b._step()
            sp_a_bin += br_a.sim.spikes.astype(np.int32)
            sp_b_bin += br_b.sim.spikes.astype(np.int32)

        dt_bin = bin_size * 0.005
        div_sum = 0.0
        for reg, r_idx in reg_masks.items():
            rate_a = float(sp_a_bin[r_idx].mean() / dt_bin)
            rate_b = float(sp_b_bin[r_idx].mean() / dt_bin)
            timeline_a[reg].append(rate_a)
            timeline_b[reg].append(rate_b)
            div_sum += abs(rate_b - rate_a)

        mean_div = div_sum / max(1, len(reg_masks))
        divergence_history.append(mean_div)
        if diverge_step is None and mean_div > diverge_threshold:
            diverge_step = b * bin_size

    if stim_rows is not None:
        simcore.undrive(br_a, stim_rows)
        simcore.undrive(br_b, stim_rows)

    # Autopsy-style comparison rows: rate A, rate B, diff, ratio, log2 ratio
    rows = []
    for reg in unique_regions:
        if reg not in reg_masks:
            continue
        vals_a = timeline_a[reg]
        vals_b = timeline_b[reg]
        mean_a = float(np.mean(vals_a)) if vals_a else 0.0
        mean_b = float(np.mean(vals_b)) if vals_b else 0.0
        diff = mean_b - mean_a
        ratio = (mean_b + 0.05) / (mean_a + 0.05)
        lr = float(math.log2(max(ratio, 1e-4)))
        rows.append({
            "name": reg,
            "rate_a": mean_a,
            "rate_b": mean_b,
            "diff": diff,
            "ratio": ratio,
            "log2_ratio": lr,
            "divergence": abs(diff),
        })

    # Sort by absolute divergence descending
    rows.sort(key=lambda r: r["divergence"], reverse=True)
    top = rows[0] if rows else None

    # Key readouts
    types = g.type.astype(str)
    readouts = {}
    for r_name, r_type in (("giant_fiber_DNp01", "DNp01"), ("sugar_MN9", "MN9"),
                           ("steering_DNa01_02", ("DNa01", "DNa02"))):
        m = np.isin(types, r_type) if isinstance(r_type, tuple) else types == r_type
        if np.any(m):
            hz_a = float(br_a.sim.activity.rates()[m].mean() / 0.005)
            hz_b = float(br_b.sim.activity.rates()[m].mean() / 0.005)
            readouts[r_name] = dict(rate_a=hz_a, rate_b=hz_b, diff=hz_b - hz_a)

    return dict(
        kind="connectome_diff",
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        seconds=round(time.time() - t0, 2),
        stimulus=stimulus,
        steps=int(steps),
        seed=int(seed),
        label_a=cfg_a.get("label", "Fly A"),
        label_b=cfg_b.get("label", "Fly B"),
        diverge_step=diverge_step,
        diverge_time_s=float(diverge_step * 0.005) if diverge_step is not None else None,
        divergence_timeline=divergence_history,
        timeline_bin_ms=bin_size * 5.0,
        rows=rows,
        top_divergence=top,
        readouts=readouts,
    )


def diff_csv(res: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", f"rate_{res['label_a']}_hz", f"rate_{res['label_b']}_hz", "diff_hz", "ratio"])
        for r in res["rows"]:
            w.writerow([r["name"], f"{r['rate_a']:.2f}", f"{r['rate_b']:.2f}", f"{r['diff']:+.2f}", f"{r['ratio']:.2f}"])
        w.writerow([])
        w.writerow(["divergence_timeline_time_ms", "mean_regional_divergence_hz"])
        bin_ms = res.get("timeline_bin_ms", 25.0)
        for i, div in enumerate(res.get("divergence_timeline", [])):
            w.writerow([f"{i * bin_ms:.1f}", f"{div:.3f}"])
    return path


def save(res: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "connectome_diff.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    diff_csv(res, folder / "connectome_diff.csv")
    return folder
