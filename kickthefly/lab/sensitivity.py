"""Sensitivity analysis (3.0 day 4): how much does each validated behavior depend on the simulation's own parameters?

The LIF model has a handful of numbers that were chosen when the sim was built, not measured: the membrane noise, the tonic
drive, the firing-rate target the slow gain controller steers toward, the sensory input gain, how fast that controller adapts,
and (a property of the data, not the neuron model) the minimum synapse count a connection needs to be kept. This module varies
each one across a documented range, one at a time, and re-runs every validated behavior with the validation suite's own code
and pass criteria (validation.py: drive ratio >= 1.5 and above the control in a one-sided Wilcoxon test at p < 0.01 over the
seeds; T-maze PI >= 0.5, |unpaired PI| <= 0.25 and paired > unpaired). Nothing about a criterion is changed here, and the
baseline column (every parameter at its default) is validation's own result for the same seeds.

CONNECTOME vs GAME RULE vs MODEL PREDICTION
  CONNECTOME        the pathways being tested (the synapses between the drive set and the readout set) and the synapse counts
                    the threshold prunes.
  GAME RULE         the parameter ranges below (this game's choice of how far to look), and the pass criteria (ours, from
                    validation.py, not from the papers).
  MODEL PREDICTION  every cell of the heatmap: "this behavior still passes when the noise is doubled" is a statement about the
                    model, not about flies.

This is analysis only. The defaults are NOT changed, retuned or re-centred on anything this reports; a behavior that is fragile
to a parameter is a finding about the model, and one that is robust is not evidence the default is right.

Method notes
  - One parameter at a time, the others at their defaults. Interactions are not explored.
  - The seeds are the validation seeds (1000-1009) by default. A one-sided Wilcoxon over fewer than 7 seeds cannot reach p < 0.01
    however large the effect, so a run with fewer is marked underpowered and its FAILs mean only "cannot pass".
  - The gain controller (target rate, adaptation) acts during the 600-step warm-up as well as the test, so a changed value is
    settled into before the stimulus starts.
  - Resumable: every finished (parameter, value) cell is written to sensitivity_progress.json as it finishes; --resume skips them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from kickthefly.lab import validation

# parameter id, label, kind, default, tested values (the default is the baseline column), tag, note. Every tested value lies
# inside the Lab > Parameters range of that parameter (lab.PARAMS), except the synapse threshold, which is the robustness
# sweep's own range (lab/robustness.py) and is a property of the data: the pack already drops connections of fewer than 3.
PARAMETERS = (
    dict(id="noise_std", label="Membrane noise", kind="lif", default=0.05, values=(0.025, 0.0375, 0.075, 0.10), fmt="{:g}",
         note="Random current on every neuron each 5 ms step. The code notes that 0.10 drowned vision and 0.03 was too quiet."),
    dict(id="bias", label="Tonic drive", kind="lif", default=0.20, values=(0.15, 0.175, 0.225, 0.25), fmt="{:g}",
         note="Constant input to every neuron. At 0.20 a neuron rests at 80% of threshold; 0.25 puts it at threshold."),
    dict(id="target_rate_hz", label="Target rate (Hz)", kind="lif", default=5.0, values=(2.5, 3.5, 7.5, 10.0), fmt="{:g}",
         note="The whole-brain firing rate the slow gain controller steers toward."),
    dict(id="ext_gain", label="Sensory input gain", kind="lif", default=4.0, values=(2.0, 3.0, 5.0, 6.0), fmt="{:g}",
         note="How strongly driven neurons are pushed: the stimulus strength of every pathway test."),
    dict(id="gain_adapt", label="Gain adaptation rate", kind="lif", default=0.002, values=(0.0, 0.001, 0.004, 0.008), fmt="{:g}",
         note="How fast the gain controller reacts; 0 freezes the synaptic gain at its starting value."),
    dict(id="min_synapses", label="Synapse threshold", kind="wiring", default=3, values=(4, 5, 6, 8, 10), fmt="{:g}",
         note="Keep only connections with at least this many synapses (the pack already keeps those with 3 or more)."),
)
BY_ID = {p["id"]: p for p in PARAMETERS}
BASELINE = "baseline"
UNDERPOWERED_BELOW = 7           # 2**-n < 0.01 needs n >= 7: the smallest n at which a one-sided Wilcoxon can reach p < 0.01


class SensitivityError(ValueError):
    pass


def validated_behaviors() -> tuple[str, ...]:
    """The behaviors validation reproduces (EXPECTED pass), in the validation suite's order. The ones that fail on the unmodified
    connectome are left out: 'stops passing' means nothing for a behavior that never passed."""
    ids = {t["id"] for t in validation.TESTS}
    return tuple(t["id"] for t in validation.TESTS if validation.EXPECTED.get(t["id"]) and t["id"] in ids)


def parameter_ids() -> tuple[str, ...]:
    return tuple(p["id"] for p in PARAMETERS)


def cells(params=None, values=None) -> list[tuple[str, float | None]]:
    """(parameter, value) cells to run, the baseline first. values: {parameter: [values]} replaces the documented range."""
    params = tuple(params) if params else parameter_ids()
    for p in params:
        if p not in BY_ID:
            raise SensitivityError(f"unknown parameter {p!r}; use one of {', '.join(parameter_ids())}")
    out: list[tuple[str, float | None]] = [(BASELINE, None)]
    for p in params:
        vals = (values or {}).get(p) or BY_ID[p]["values"]
        out += [(p, float(v)) for v in vals]
    return out


def cell_key(param: str, value) -> str:
    return BASELINE if param == BASELINE else f"{param}={float(value):g}"


def _config(seeds, tests, cell_list) -> dict:
    return dict(seeds=list(seeds), tests=list(tests), cells=[cell_key(*c) for c in cell_list])


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _run_args(param: str, value):
    """(lab params, wiring) for one cell. The LIF parameters go in as Lab parameters, the synapse threshold as a connectome
    change, both through the paths validation already takes."""
    if param == BASELINE:
        return None, None
    if BY_ID[param]["kind"] == "wiring":
        from kickthefly.sim.wiring import Wiring

        return None, Wiring(min_synapses=int(round(value)))
    return {param: float(value)}, None


def _effect_size(t: dict) -> dict:
    """The headline number, its control and a standardised paired effect (Cohen's d_z of drive minus control over the seeds)."""
    m, per = t["measured"], t.get("per_seed") or []
    if "drive_ratio_mean" in m:
        a = [p["drive"]["ratio"] for p in per]
        b = [p["control"]["ratio"] for p in per]
        effect, control, metric = m["drive_ratio_mean"], m["control_ratio_mean"], "drive/baseline ratio"
    else:                                                    # the T-maze: PI paired vs unpaired
        a = [p["pi"] for p in per]
        b = [p["control_pi"] for p in per]
        effect, control, metric = m["pi_mean"], m["control_pi_mean"], "performance index"
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    dz = float(np.mean(d) / sd) if sd > 1e-12 else (0.0 if np.allclose(d, 0) else float("inf"))
    return dict(passed=bool(t["passed"]), effect=float(effect), control=float(control), metric=metric,
                effect_size_dz=dz, p_value=float(m.get("p_value", 1.0)), n=int(m.get("n", len(per))))


def run_cell(param: str, value, seeds, tests, workers: int | None = None, progress=None) -> dict:
    """One (parameter, value) cell: the validation suite's code on the chosen behaviors, with that one change."""
    lab_params, wiring = _run_args(param, value)
    res = validation.run(seeds=tuple(seeds), workers=workers, progress=progress, include=set(tests), wiring=wiring,
                         params=lab_params)
    by = {t["id"]: _effect_size(t) for t in res["tests"]}
    return dict(key=cell_key(param, value), parameter=param, value=None if value is None else float(value), behaviors=by,
                backend=res.get("backend"), seconds=res.get("seconds"))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


PROGRESS_NAME = "sensitivity_progress.json"


def run(params=None, tests=None, seeds=None, values=None, workers: int | None = None, folder: Path | str | None = None,
        resume: bool = False, progress=None) -> dict:
    """Run the analysis. progress(done_cells, total_cells, label) after every cell and as a cell's seeds finish.

    folder: where sensitivity_progress.json is kept (one line per finished cell). With resume=True, cells already in it (for
    the same seeds, tests and cell list) are not run again; a progress file for a different configuration is refused.
    """
    seeds = tuple(seeds) if seeds else validation.SEEDS
    tests = tuple(tests) if tests else validated_behaviors()
    for t in tests:
        if t not in validation.BY_ID or t not in {x["id"] for x in validation.TESTS}:
            raise SensitivityError(f"unknown adult behavior {t!r}")
    cell_list = cells(params, values)
    cfg = _config(seeds, tests, cell_list)
    chash = _config_hash(cfg)
    done: dict[str, dict] = {}
    path = Path(folder) / PROGRESS_NAME if folder else None
    if path is not None and resume and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("config_hash") != chash:
            raise SensitivityError(f"{path} was made for a different run (other seeds, behaviors or cells); use another --out, "
                                   "or run without --resume to start over")
        done = dict(saved.get("cells", {}))
    t0 = time.time()
    total = len(cell_list)
    for i, (p, v) in enumerate(cell_list):
        key = cell_key(p, v)
        if key in done:
            if progress:
                progress(len(done), total, f"{key} (already done)")
            continue

        def inner(d, n, label, key=key):
            if progress:
                progress(len(done), total, f"{key}: {d}/{n} {label}")

        done[key] = run_cell(p, v, seeds, tests, workers=workers, progress=inner)
        if path is not None:
            _atomic_write(path, json.dumps(dict(config=cfg, config_hash=chash, cells=done), indent=1))
        if progress:
            progress(len(done), total, f"{key} done")
    ordered = [done[cell_key(p, v)] for p, v in cell_list]
    return dict(kind="sensitivity", created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
                seeds=list(seeds), tests=list(tests), underpowered=len(seeds) < UNDERPOWERED_BELOW,
                criteria=dict(ratio_min=validation.RATIO_MIN, p_max=validation.P_MAX, pi_min=validation.PI_MIN,
                              control_pi_max=validation.CONTROL_PI_MAX),
                parameters=[dict(id=p["id"], label=p["label"], kind=p["kind"], default=p["default"], note=p["note"],
                                 values=[float(x) for x in ((values or {}).get(p["id"]) or p["values"])])
                            for p in PARAMETERS if (not params or p["id"] in params)],
                cells=ordered, config_hash=chash,
                tags=dict(pathways="CONNECTOME", ranges_and_criteria="GAME RULE", every_cell="MODEL PREDICTION"))


# --- reading a result --------------------------------------------------------------------------------------------------
def baseline_of(res: dict) -> dict:
    return next(c for c in res["cells"] if c["parameter"] == BASELINE)["behaviors"]


def rows(res: dict) -> list[dict]:
    """The long table: one row per (parameter, value, behavior)."""
    base = baseline_of(res)
    out = []
    for c in res["cells"]:
        for b, r in c["behaviors"].items():
            out.append(dict(parameter=c["parameter"], value="" if c["value"] is None else c["value"],
                            is_default=c["parameter"] == BASELINE, behavior=b, passed=r["passed"], metric=r["metric"],
                            effect=r["effect"], control=r["control"], effect_size_dz=r["effect_size_dz"],
                            p_value=r["p_value"], n_seeds=r["n"],
                            effect_vs_baseline=r["effect"] - base[b]["effect"] if b in base else ""))
    return out


def robustness_table(res: dict) -> list[dict]:
    """Per parameter and behavior: how many of the tested values still pass, and the first value (going away from the
    default in each direction) at which it stops."""
    out = []
    for p in res["parameters"]:
        cs = [c for c in res["cells"] if c["parameter"] == p["id"]]
        for b in res["tests"]:
            below = sorted([c for c in cs if c["value"] < p["default"]], key=lambda c: -c["value"])
            above = sorted([c for c in cs if c["value"] > p["default"]], key=lambda c: c["value"])

            def first_fail(seq):
                for c in seq:
                    if b in c["behaviors"] and not c["behaviors"][b]["passed"]:
                        return c["value"]
                return None

            n_pass = sum(1 for c in cs if c["behaviors"].get(b, {}).get("passed"))
            out.append(dict(parameter=p["id"], behavior=b, values_tested=len(cs), values_passing=n_pass,
                            first_fail_below=first_fail(below), first_fail_above=first_fail(above)))
    return out


def summary(res: dict) -> str:
    base = baseline_of(res)
    lines = [f"SENSITIVITY ANALYSIS: {len(res['tests'])} validated behaviors x {len(res['parameters'])} parameters, seeds "
             f"{res['seeds'][0]}-{res['seeds'][-1]} (n = {len(res['seeds'])}), {res['seconds']:.0f} s",
             "analysis only: the defaults are not changed by anything below; every cell is a MODEL PREDICTION"]
    if res.get("underpowered"):
        lines.append(f"UNDERPOWERED: fewer than {UNDERPOWERED_BELOW} seeds cannot reach p < {validation.P_MAX}, so a FAIL here only "
                     "means 'cannot pass at this n'")
    lines.append("baseline (all defaults): " + ", ".join(f"{b} {'PASS' if r['passed'] else 'FAIL'}" for b, r in base.items()))
    lines.append("")
    for p in res["parameters"]:
        lines.append(f"{p['label']} [{p['id']}], default {p['default']:g}")
        for c in [c for c in res["cells"] if c["parameter"] == p["id"]]:
            marks = " ".join(f"{b.split('_')[0][:6]}:{'P' if r['passed'] else 'F'}{r['effect']:.2f}"
                             for b, r in c["behaviors"].items())
            lines.append(f"  {c['value']:>8g}  {marks}")
    fragile = [(r["parameter"], r["behavior"], r["values_passing"], r["values_tested"]) for r in robustness_table(res)
               if r["values_passing"] < r["values_tested"]]
    lines.append("")
    lines.append(f"{len(fragile)} of {len(res['parameters']) * len(res['tests'])} parameter x behavior pairs fail at one or more "
                 "tested values")
    return "\n".join(lines)


# --- export ------------------------------------------------------------------------------------------------------------
PALETTES = {                      # (pass, fail, pass text, fail text, empty); blue/orange stays apart with red-green colorblindness
    "default": ((46, 125, 50), (183, 28, 28), (255, 255, 255), (255, 255, 255), (60, 60, 60)),
    "blue-yellow": ((30, 100, 200), (240, 190, 20), (255, 255, 255), (20, 20, 20), (90, 90, 90)),
    "high-contrast": ((255, 255, 255), (0, 0, 0), (0, 0, 0), (255, 255, 255), (128, 128, 128)),
}


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(x) for x in c)


def heatmap_cells(res: dict) -> tuple[list[str], list[tuple[str, dict]]]:
    """(behavior columns, [(row label, {behavior: result})]) in the order the heatmap draws them."""
    grid = []
    for c in res["cells"]:
        label = "baseline (defaults)" if c["parameter"] == BASELINE else f"{BY_ID[c['parameter']]['label']} = {c['value']:g}"
        grid.append((label, c["behaviors"]))
    return list(res["tests"]), grid


def heatmap_svg(res: dict, palette: str = "default") -> str:
    """A self-contained SVG (no scripts, no external requests): rows = parameter value, columns = behavior, each cell its PASS
    or FAIL and its effect (the drive ratio, or the PI) with the cell shaded by how far it is above the pass line."""
    ok, bad, ok_t, bad_t, _ = PALETTES.get(palette, PALETTES["default"])
    cols, grid = heatmap_cells(res)
    cw, rh, lw, top = 74, 22, 220, 150
    w, h = lw + cw * len(cols) + 20, top + rh * len(grid) + 60
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11" '
           f'role="img" aria-label="Sensitivity heatmap: parameter value by validated behavior, pass or fail with effect size">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           f'<text x="8" y="18" font-size="14" font-weight="bold">Sensitivity: parameter value x validated behavior</text>',
           f'<text x="8" y="36">each cell: PASS/FAIL (validation\'s own criteria) and the effect (drive ratio, or the T-maze PI); '
           f'seeds {res["seeds"][0]}-{res["seeds"][-1]}, n = {len(res["seeds"])}</text>',
           '<text x="8" y="52">MODEL PREDICTION. Analysis only: the defaults are not changed by this.</text>']
    if res.get("underpowered"):
        out.append(f'<text x="8" y="68" fill="#b71c1c">UNDERPOWERED: fewer than {UNDERPOWERED_BELOW} seeds cannot reach p &lt; 0.01</text>')
    for j, b in enumerate(cols):
        x = lw + j * cw + cw / 2
        out.append(f'<text transform="translate({x},{top - 6}) rotate(-50)" text-anchor="start">{b}</text>')
    for i, (label, by) in enumerate(grid):
        y = top + i * rh
        out.append(f'<text x="8" y="{y + 15}">{label}</text>')
        for j, b in enumerate(cols):
            r = by.get(b)
            x = lw + j * cw
            if r is None:
                out.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{rh - 2}" fill="#dddddd"/>')
                continue
            fill, txt = (ok, ok_t) if r["passed"] else (bad, bad_t)
            out.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{rh - 2}" fill="{_hex(fill)}" stroke="#222" stroke-width="0.5"/>')
            out.append(f'<text x="{x + cw / 2 - 1}" y="{y + 15}" text-anchor="middle" fill="{_hex(txt)}">'
                       f'{"PASS" if r["passed"] else "FAIL"} {r["effect"]:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def save(res: dict, folder: Path | str) -> list[Path]:
    """sensitivity.json (everything), sensitivity.csv (the long table), sensitivity_matrix.csv (a parameter value per row, a
    behavior per column), sensitivity_robustness.csv and sensitivity_heatmap.svg."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    p = folder / "sensitivity.json"
    _atomic_write(p, json.dumps(res, indent=1))
    paths.append(p)
    table = rows(res)
    p = folder / "sensitivity.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["parameter"])
        w.writeheader()
        w.writerows(table)
    paths.append(p)
    p = folder / "sensitivity_matrix.csv"
    cols, grid = heatmap_cells(res)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter_value"] + cols)
        for label, by in grid:
            w.writerow([label] + [f"{'PASS' if by[b]['passed'] else 'FAIL'} {by[b]['effect']:.3f} (dz {by[b]['effect_size_dz']:.2f})"
                                  if b in by else "" for b in cols])
    paths.append(p)
    p = folder / "sensitivity_robustness.csv"
    rt = robustness_table(res)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rt[0]) if rt else ["parameter"])
        w.writeheader()
        w.writerows(rt)
    paths.append(p)
    p = folder / "sensitivity_heatmap.svg"
    _atomic_write(p, heatmap_svg(res))
    paths.append(p)
    return paths
