"""Neural Clamp: record a run's exact spike train and replay it into a modified connectome.

DYNAMIC CLAMP NOTICE:
Forced spikes override the network's own state and break feedback loops such as proprioception.
This is dynamic clamping, not an autonomous run. Clamping isolates connectome wiring effects
(lesions, thresholds, sign flips, inhibition blocks) from behavioral input variation.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kickthefly.core import simcore
from kickthefly.sim.wiring import Wiring

if TYPE_CHECKING:
    from kickthefly.lab.recorder import Recorder

DYNAMIC_CLAMP_NOTICE = (
    "DYNAMIC CLAMP: Forced spikes override the network's own state and break feedback loops "
    "such as proprioception. This is dynamic clamping, not an autonomous run."
)


def record_reference_run(stimulus: str = "looming", steps: int = 200, seed: int = 1000) -> dict:
    """Record free-running reference spike train with run context (stimuli, kinematics)."""
    from kickthefly.lab import recorder

    g, _, _ = simcore.pack()
    br = simcore.new_brain(seed=seed, warmup=150)
    
    # Watch all neurons grouped by broad region/functional group
    groups = {
        "looming (LPLC2, LC4)": br.sense.get(("loom", None), np.zeros(0, np.int64)),
        "giant fiber (DNp01)": np.flatnonzero(g.type.astype(str) == "DNp01"),
        "steering (DNa01, DNa02)": np.flatnonzero(np.isin(g.type.astype(str), ("DNa01", "DNa02"))),
        "sugar (taste)": br.sense.get(("sweet", None), np.zeros(0, np.int64)),
        "proboscis (MN9)": np.flatnonzero(g.type.astype(str) == "MN9"),
    }
    rec = recorder.Recorder(br, groups).start()
    
    # Apply stimulus if requested
    if stimulus == "looming":
        simcore.drive(br, br.sense[("loom", None)], amp=0.5)
        rec.log_event("stimulus", "loom", "both", 0.5)
    elif stimulus == "sugar":
        simcore.drive(br, br.sense[("sweet", None)], amp=0.5)
        rec.log_event("stimulus", "sweet", "both", 0.5)
    elif stimulus == "antenna":
        simcore.drive(br, br.sense[("head", None)], amp=0.5)
        rec.log_event("stimulus", "head", "both", 0.5)
    
    # Step synchronously
    all_spikes = simcore.step(br, steps, record=np.arange(br.n))
    
    if stimulus in ("looming", "sugar", "antenna"):
        simcore.undrive(br, br.sense[("loom" if stimulus == "looming" else "sweet" if stimulus == "sugar" else "head", None)])
        
    rec.stop()
    arrays = rec.arrays()
    ctx = rec.context()
    
    # Per-region firing rates
    regions = np.asarray(getattr(br, "region", np.full(br.n, "unassigned"))).astype(str)
    unique_regions = sorted(set(regions))
    region_rates = {}
    for reg in unique_regions:
        m = regions == reg
        if np.any(m):
            region_rates[reg] = float(all_spikes[:, m].mean() / 0.005)

    return dict(
        kind="reference_run",
        stimulus=stimulus,
        steps=int(steps),
        seed=int(seed),
        spikes=all_spikes,
        spike_steps=arrays["spike_steps"],
        spike_index=arrays["spike_index"],
        recorded_rows=rec.rows,
        events=ctx["events"],
        kinematics=ctx["kinematics"],
        region_rates=region_rates,
        notice=DYNAMIC_CLAMP_NOTICE,
    )


def run_neural_clamp(
    ref_run: dict,
    wiring: Wiring | None = None,
    lesion_rows: np.ndarray | None = None,
    clamp_rows: np.ndarray | None = None,
    seed: int = 1000,
) -> dict:
    """Replay a recorded spike train into a modified brain under dynamic clamp."""
    t0 = time.time()
    g, _, _ = simcore.pack()
    w = wiring or Wiring()
    steps = ref_run["steps"]
    ref_spikes = ref_run["spikes"]
    
    # Build modified brain
    br_mod = simcore.new_brain(seed=seed, warmup=150, wiring=w)
    if lesion_rows is not None and len(lesion_rows):
        br_mod.set_override(lesion_rows, -1)
        
    # By default, clamp input/sensory neurons that received stimulus, or all recorded neurons
    if clamp_rows is None:
        stim = ref_run.get("stimulus")
        if stim == "looming":
            clamp_rows = br_mod.sense.get(("loom", None), np.zeros(0, np.int64))
        elif stim == "sugar":
            clamp_rows = br_mod.sense.get(("sweet", None), np.zeros(0, np.int64))
        elif stim == "antenna":
            clamp_rows = br_mod.sense.get(("head", None), np.zeros(0, np.int64))
        else:
            clamp_rows = np.arange(br_mod.n, dtype=np.int64)

    clamp_set = np.asarray(clamp_rows, np.int64)
    clamped_spikes = np.zeros((steps, br_mod.n), bool)
    
    # Step loop under dynamic clamp
    for t in range(steps):
        # Force spikes from reference run on clamped neurons
        forced = clamp_set[ref_spikes[t, clamp_set]]
        br_mod.sim.spikes[clamp_set] = False
        br_mod.sim.spikes[forced] = True
        br_mod.sim.v[forced] = br_mod.sim.p.v_reset
        br_mod.sim.refr[forced] = br_mod.sim.p.refractory_steps
        if getattr(br_mod.sim, "backend", None) is not None:
            br_mod.sim.backend.sync_from_host()
        
        # Advance simulation step (override drive applied if surgery active)
        drv = br_mod.override if br_mod.surgery else None
        sp = br_mod.sim.step(drv)
        # Ensure clamped neurons reflect forced spikes
        sp[clamp_set] = False
        sp[forced] = True
        clamped_spikes[t] = sp

    # Compute region firing rates under clamp
    regions = np.asarray(getattr(br_mod, "region", np.full(br_mod.n, "unassigned"))).astype(str)
    unique_regions = sorted(set(regions))
    clamped_region_rates = {}
    diff_region_rates = {}
    for reg in unique_regions:
        m = regions == reg
        if np.any(m):
            c_rate = float(clamped_spikes[:, m].mean() / 0.005)
            f_rate = ref_run["region_rates"].get(reg, 0.0)
            clamped_region_rates[reg] = c_rate
            diff_region_rates[reg] = c_rate - f_rate

    # Check key readouts
    types = g.type.astype(str)
    readouts = {}
    for r_name, r_type in (("giant_fiber_DNp01", "DNp01"), ("sugar_MN9", "MN9"),
                           ("steering_DNa01_02", ("DNa01", "DNa02"))):
        m = np.isin(types, r_type) if isinstance(r_type, tuple) else types == r_type
        if np.any(m):
            f_hz = float(ref_spikes[:, m].mean() / 0.005)
            c_hz = float(clamped_spikes[:, m].mean() / 0.005)
            readouts[r_name] = dict(free_hz=f_hz, clamped_hz=c_hz, diff_hz=c_hz - f_hz)

    return dict(
        kind="neural_clamp",
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        seconds=round(time.time() - t0, 2),
        stimulus=ref_run.get("stimulus"),
        steps=int(steps),
        clamped_neurons=len(clamp_set),
        lesioned_neurons=len(lesion_rows) if lesion_rows is not None else 0,
        wiring=w.as_dict(),
        notice=DYNAMIC_CLAMP_NOTICE,
        free_region_rates=ref_run["region_rates"],
        clamped_region_rates=clamped_region_rates,
        diff_region_rates=diff_region_rates,
        readouts=readouts,
    )


def clamp_csv(res: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# " + DYNAMIC_CLAMP_NOTICE])
        w.writerow(["region", "free_running_hz", "clamped_hz", "diff_hz"])
        for reg in sorted(res["free_region_rates"]):
            f = res["free_region_rates"].get(reg, 0.0)
            c = res["clamped_region_rates"].get(reg, 0.0)
            d = res["diff_region_rates"].get(reg, 0.0)
            w.writerow([reg, f"{f:.2f}", f"{c:.2f}", f"{d:+.2f}"])
        w.writerow([])
        w.writerow(["readout", "free_running_hz", "clamped_hz", "diff_hz"])
        for r_name, r_vals in res.get("readouts", {}).items():
            w.writerow([r_name, f"{r_vals['free_hz']:.2f}", f"{r_vals['clamped_hz']:.2f}", f"{r_vals['diff_hz']:+.2f}"])
    return path


def save(res: dict, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "neural_clamp.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    clamp_csv(res, folder / "neural_clamp_diff.csv")
    return folder
