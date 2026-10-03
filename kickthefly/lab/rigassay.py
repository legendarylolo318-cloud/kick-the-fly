"""Classic behavior rigs: the headless protocols, the assay and its PRE-REGISTERED criteria (3.0 day 5). The physics are game/rigs.py's; read
its docstring for what is CONNECTOME, GAME RULE and MODEL PREDICTION.

Written and committed BEFORE the first run on any held-out seed (the commit time is the audit trail). Design choices (durations, the definitions
of stripe deviation, a transit and the preference index, the effect thresholds below) were made looking only at exploration seeds 0-9, individuality
'subtle'; no held-out seed (1000-1009) has been run through any of these rigs when this text was committed.

Design. One brain per fly and rig, seeds 1000-1009 (n = 10), individuality 'subtle' (the default) and the control 'off'. Every criterion is a
paired comparison within each fly. A criterion PASSES when the mean of the per-fly effect is at least the stated minimum AND the one-sided Wilcoxon
signed-rank p is below 0.01 (the validation suite's own bar; at n = 10 the smallest p is 1/1024, reported as p < 0.001). Only the default mode's
result counts as the rig's result; the 'off' control is reported next to it, and where the two differ the report says so.

  Tethered flight simulator (open loop: 2 s calm panorama then 4 s of rotation, at +3, -3, +1 and -1 rad/s; closed loop: 2 s calm then 10 s of a
  constant 1 rad/s rotation, +1 and -1, the panorama also rotated by the fly's own yaw at gain 1)
    T1  open loop direction. d = DS(+3) - DS(-3), where DS(w) is the mean right-minus-left DNa01/02 rate (Hz) in the last 3.5 s of the
        rotation minus the mean in the calm 1.5 s before it. PASS: mean d >= 3.0 Hz and p < 0.01 (d > 0).
    T2  closed loop compensation. c = (mean yaw rate over the last 5 s - mean yaw rate in the calm 1.5 s) / w, averaged over w = +1 and -1: the
        share of the imposed rotation the fly cancels. PASS: mean c >= 0.10 and p < 0.01 (c > 0).
  Fly on a ball (VR; 2 s calm then 14 s of drift +1 and -1 rad/s on a distant panorama, closed loop, while walking; a closed-loop bar at +90 and
  -90 degrees for 20 s, visible and hidden)
    BL1 as T2 on the ball: mean c >= 0.10 and p < 0.01.
    BL2 bar fixation. e = mean |bar bearing| over the last 5 s of 20 s (deg), averaged over the two offsets; the effect is e(hidden) - e(visible).
        PASS: mean effect >= 10 deg and p < 0.01 and the mean e(visible) < 30 deg.
  Buridan's paradigm (platform radius 0.5 m, stripes east and west 1.5 m from the centre, 120 s, start heading drawn from the seed; stripes versus none)
    B1  stripe deviation (the median over the run, after 10 s, of the angle between the direction of travel and the east-west axis, folded to 0-90
        deg; 45 is chance). Effect: deviation(none) - deviation(stripes). PASS: mean >= 10 deg and p < 0.01.
    B2  transits (the east-west position reaches beyond 0.8 radius on one side and then the other; each switch counts once). Effect: transits(stripes) -
        transits(none). PASS: mean >= 3 and p < 0.01.
  Four-field olfactory arena (1 m square, the odor 'fruit' in quadrants 0 and 2, air in 1 and 3, 120 s, after 10 s counted; odor versus the sham, the same
  arena and start heading with no odor ever delivered)
    F1  preference index PI = (time in odor - time in air) / (their sum). Effect: PI(odor) - PI(sham). PASS: mean >= 0.15 and p < 0.01 (the fly prefers the
        odor quadrants). Reported, not a criterion: the speed in odor minus in air (a kinesis).

What validation predicts (docs/validation.md), written before the runs. optomotor_turning PASSES (rightward rotation excites the right steering neurons
x2.72, the left pair x1.09), so T1, T2 and BL1 are expected to pass: the rotation reaches DNa01/02 through T4/T5 and the yaw is read from DNa01/02, the only
GAME RULEs in the loop being the EMD stage and the yaw gain. BL2 and B1 and B2 depend on the duel's LC10 rule (LC10 poked on the side an object is on),
which no validation test checks; a probe on exploration seeds 0 and 1 found it steers the right way (right-side LC10 drive moves right-minus-left by
+8.6 to +9.7 Hz, left by -8.1 to -10.2), so they are expected to pass, with a deviation far tighter than real flies'. F1 has no mechanism that gives an odor a
valence: nothing in the validation suite connects an olfactory input to steering or walking, and the E-PG compass tests FAIL, so no rig here requires a
heading memory. F1 is expected to FAIL (no preference). Whatever the held-out runs show is reported, including a miss.
"""
from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from kickthefly.game import rigs

DEFAULT_MODE = "subtle"
CONTROL_MODE = "off"
VALIDATION_SEEDS = tuple(range(1000, 1010))
P_MAX = 0.01

# --- design (fixed from exploration seeds 0-9) -----------------------------------------------------------------------------------
TETHER_PRE_S, TETHER_STIM_S = 2.0, 4.0              # open loop: 2 s of calm panorama, then 4 s of rotation
TETHER_OMEGAS = (3.0, -3.0, 1.0, -1.0)              # rad/s, rightward positive; the first two are the criterion
TETHER_CLOSED_S, TETHER_CLOSED_OMEGAS = 10.0, (1.0, -1.0)
SETTLE_S = 0.5                                      # skipped at the start of a stimulus before its response is averaged
LAST_S = 5.0                                        # the closed-loop and bar-fixation measures use the last 5 s
BALL_DRIFT_S, BALL_BAR_S = 16.0, 20.0
BAR_OFFSETS_DEG = (90.0, -90.0)
BURIDAN_S, FOURFIELD_S = 120.0, 120.0
ODOR = "fruit"


def _wilcoxon_greater(a, b=None) -> float:
    from scipy import stats

    d = np.asarray(a, float) if b is None else np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)


def _new(seed: int, mode: str, backend: str | None):
    from kickthefly.core import simcore

    return simcore.new_brain(seed=int(seed), individuality=mode, backend=backend)


def _win(a: np.ndarray, t0: float, t1: float, col: str) -> np.ndarray:
    m = (a[:, 0] > t0) & (a[:, 0] <= t1)
    return a[m, rigs.COLS.index(col)]


# --- per-fly measurements (one brain each, in a worker process) ---------------------------------------------------------------
def tethered_fly(seed: int, mode: str = DEFAULT_MODE, backend: str | None = None, cancel=None, progress=None, br=None) -> dict:
    br = br or _new(seed, mode, backend)
    ds, dR, dL = {}, {}, {}
    for om in TETHER_OMEGAS:                                     # open loop
        r = rigs.run_tethered(br, [(TETHER_PRE_S, 0.0), (TETHER_STIM_S, om)], "open", cancel=cancel)
        a = r["_full"]
        pre = _win(a, 0.5, TETHER_PRE_S, "steer_hz").mean()
        stim = _win(a, TETHER_PRE_S + SETTLE_S, TETHER_PRE_S + TETHER_STIM_S, "steer_hz").mean()
        ds[om] = float(stim - pre)
        dR[om] = float(_win(a, TETHER_PRE_S + SETTLE_S, 99, "turn_r").mean() - _win(a, 0.5, TETHER_PRE_S, "turn_r").mean())
        dL[om] = float(_win(a, TETHER_PRE_S + SETTLE_S, 99, "turn_l").mean() - _win(a, 0.5, TETHER_PRE_S, "turn_l").mean())
        if progress:
            progress(0.15 * (1 + list(TETHER_OMEGAS).index(om)))
    comp, slip = {}, {}
    for om in TETHER_CLOSED_OMEGAS:                              # closed loop
        r = rigs.run_tethered(br, [(TETHER_PRE_S, 0.0), (TETHER_CLOSED_S, om)], "closed", cancel=cancel)
        a = r["_full"]
        end = TETHER_PRE_S + TETHER_CLOSED_S
        yaw_pre = _win(a, 0.5, TETHER_PRE_S, "yaw_rate").mean()
        yaw_end = _win(a, end - LAST_S, end, "yaw_rate").mean()
        comp[om] = float((yaw_end - yaw_pre) / om)                # fraction of the imposed rotation the fly cancels
        slip[om] = float(np.abs(_win(a, end - LAST_S, end, "slip")).mean())
    return dict(seed=int(seed), individuality=mode, delta_steer_hz={str(k): v for k, v in ds.items()},
                delta_R_hz={str(k): v for k, v in dR.items()}, delta_L_hz={str(k): v for k, v in dL.items()},
                d_directional=float(ds[3.0] - ds[-3.0]), compensation={str(k): v for k, v in comp.items()},
                compensation_mean=float(np.mean(list(comp.values()))), closed_slip={str(k): v for k, v in slip.items()})


def ball_fly(seed: int, mode: str = DEFAULT_MODE, backend: str | None = None, cancel=None, progress=None, br=None) -> dict:
    br = br or _new(seed, mode, backend)
    comp, fwd = {}, []
    for om in TETHER_CLOSED_OMEGAS:                              # panorama drift in closed loop, while walking
        r = rigs.run_ball(br, "panorama", "closed", seconds=TETHER_PRE_S + BALL_DRIFT_S,
                          omega_ext=lambda t, om=om: om if t > TETHER_PRE_S else 0.0, cancel=cancel)
        a = r["_full"]
        end = TETHER_PRE_S + BALL_DRIFT_S
        comp[om] = float((_win(a, end - LAST_S, end, "yaw_rate").mean() - _win(a, 0.5, TETHER_PRE_S, "yaw_rate").mean()) / om)
        fwd.append(float(_win(a, 0.5, end, "speed").mean()))
    err = {"visible": [], "hidden": []}
    for vis in (True, False):                                    # a bar in closed loop: how far from straight ahead does the fly end up?
        for off in BAR_OFFSETS_DEG:
            r = rigs.run_ball(br, "bar", "closed", seconds=BALL_BAR_S, bar_offset=math.radians(off), bar_visible=vis, cancel=cancel)
            a = r["_full"]
            beta = _win(a, BALL_BAR_S - LAST_S, BALL_BAR_S, "aux")
            err["visible" if vis else "hidden"].append(float(np.degrees(np.abs(beta)).mean()))
            fwd.append(float(_win(a, 0.5, BALL_BAR_S, "speed").mean()))
    return dict(seed=int(seed), individuality=mode, compensation={str(k): v for k, v in comp.items()}, compensation_mean=float(np.mean(list(comp.values()))),
                bar_error_deg_visible=float(np.mean(err["visible"])), bar_error_deg_hidden=float(np.mean(err["hidden"])),
                bar_error_deg_by_offset=dict(visible=err["visible"], hidden=err["hidden"]), forward_speed_mean=float(np.mean(fwd)))


def _start_heading(seed: int) -> float:
    return float(np.random.default_rng(int(seed) + 770).uniform(-math.pi, math.pi))


def buridan_fly(seed: int, mode: str = DEFAULT_MODE, backend: str | None = None, cancel=None, progress=None, seconds: float = BURIDAN_S) -> dict:
    out = {}
    for key, on in (("stripes", True), ("none", False)):
        br = _new(seed, mode, backend)
        r = rigs.run_buridan(br, seconds, stripes=on, start_heading=_start_heading(seed), cancel=cancel)
        f = r["_full"]
        out[key] = dict(deviation_deg=rigs.stripe_deviation(f), transits=rigs.transits(f), edge_contacts=int(f[-1, rigs.COLS.index("aux")]),
                        speed=float(f[:, rigs.COLS.index("speed")].mean()), path=float(np.hypot(np.diff(f[:, 6]), np.diff(f[:, 7])).sum()))
        if progress:
            progress(0.5 if on else 1.0)
    return dict(seed=int(seed), individuality=mode, **out)


def fourfield_fly(seed: int, mode: str = DEFAULT_MODE, backend: str | None = None, cancel=None, progress=None, seconds: float = FOURFIELD_S) -> dict:
    out = {}
    for key, dl in (("odor", True), ("sham", False)):
        br = _new(seed, mode, backend)
        r = rigs.run_fourfield(br, seconds, odor=ODOR, deliver=dl, start_heading=_start_heading(seed), cancel=cancel)
        out[key] = rigs.preference_index(r["_full"], r["dt"])
        if progress:
            progress(0.5 if dl else 1.0)
    return dict(seed=int(seed), individuality=mode, **out)


FLY = dict(tethered=tethered_fly, ball=ball_fly, buridan=buridan_fly, fourfield=fourfield_fly)


def _task(args: tuple) -> dict:
    rig, seed, mode, backend = args
    return FLY[rig](seed, mode, backend)


def run_flies(rig: str, seeds, mode: str = DEFAULT_MODE, workers: int = 1, backend: str | None = None, progress=None, cancel=None, play=None) -> list[dict]:
    """The per-fly measurements of one rig. play: (task) -> fly dict, replacing the brains (tests)."""
    if rig not in FLY:
        raise ValueError(f"unknown rig {rig!r} (choose from {', '.join(FLY)})")
    tasks = [(rig, int(s), mode, backend) for s in seeds]
    out: dict = {}
    t0 = time.time()

    def got(res):
        out[res["seed"]] = res
        if progress:
            progress(len(out), len(tasks), f"{rig}, {mode}: fly {res['seed']} done ({time.time() - t0:.0f} s)")

    if play is not None:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            got(play(t))
    elif workers <= 1:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            got(FLY[rig](t[1], mode, backend, cancel=cancel))
    else:
        import multiprocessing

        ex = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        try:
            futs = [ex.submit(_task, t) for t in tasks]
            for f in as_completed(futs):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")
                got(f.result())
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    return [out[int(s)] for s in seeds]


# --- one scene run: what Lab > Behavior rigs, --rig and the rig protocol run (and what they export) --------------------------------
SCENES = {
    "tethered": dict(mode=("open", "closed"), default=dict(mode="open", omega=3.0, seconds=14.0, gain=1.0)),
    "ball": dict(mode=("closed", "open"), default=dict(mode="closed", scene="bar", seconds=30.0, omega=0.0, bar_deg=90.0)),
    "buridan": dict(mode=("stripes", "none"), default=dict(mode="stripes", seconds=60.0)),
    "fourfield": dict(mode=("odor", "sham"), default=dict(mode="odor", seconds=60.0)),
}
OPT_RANGES = (("seconds", 5.0, 600.0), ("omega", -10.0, 10.0), ("gain", 0.0, 3.0), ("bar_deg", -180.0, 180.0))
REC_GROUPS = ("steer_R", "steer_L", "walk_DNp09", "T4T5_progressive", "T4T5_regressive", "LC10")


def tethered_schedule(omega: float, seconds: float) -> list[tuple[float, float]]:
    """A calm panorama, the rotation to the right, a rest, the rotation to the left, a rest (seconds split 1:3:1:3:1... as it fits)."""
    s = max(5.0, float(seconds))
    u = s / 9.0
    return [(u, 0.0), (3 * u, float(omega)), (u, 0.0), (3 * u, -float(omega)), (u, 0.0)]


def rec_groups(br) -> dict[str, np.ndarray]:
    from kickthefly.lab import assays

    g = assays.groups(br)
    lc10 = np.flatnonzero(np.char.startswith(br.types, "LC10"))
    return {"steer_R": g["dna_steer_r"], "steer_L": g["dna_steer_l"], "walk_DNp09": g["dnp09"],
            "T4T5_progressive": np.concatenate([g["t45_prog_r"], g["t45_prog_l"]]),
            "T4T5_regressive": np.concatenate([g["t45_regr_r"], g["t45_regr_l"]]), "LC10": lc10}


def scene_run(rig: str, seed: int = 0, individuality: str = DEFAULT_MODE, backend: str | None = None, folder: Path | None = None,
              progress=None, cancel=None, br=None, **opt) -> dict:
    """One fly in one rig, for the Lab page, --rig and the rig protocol. opt overrides SCENES[rig]['default'] (mode, seconds, omega, gain,
    scene, bar_deg). With `folder` the run is recorded in the Lab's own format (recorder.py: spikes, rates, group rates, events, kinematics,
    metadata) plus rig_trace.csv and rig_summary.json. Returns the trace (decimated), the summary and the tags."""
    from kickthefly.lab import recorder

    if rig not in SCENES:
        raise ValueError(f"unknown rig {rig!r} (choose from {', '.join(SCENES)})")
    p = dict(SCENES[rig]["default"], **{k: v for k, v in opt.items() if v is not None})
    if p["mode"] not in SCENES[rig]["mode"]:
        raise ValueError(f"{rig}: mode must be one of {', '.join(SCENES[rig]['mode'])}")
    for key, lo, hi in OPT_RANGES:                                 # the protocol checker's ranges, for the API and the page too
        if key in p and not (isinstance(p[key], (int, float)) and not isinstance(p[key], bool) and math.isfinite(p[key]) and lo <= p[key] <= hi):
            raise ValueError(f"{rig}: {key} must be a number from {lo:g} to {hi:g}")
    if p.get("scene", "bar") not in ("panorama", "bar"):
        raise ValueError(f"{rig}: scene must be panorama or bar")
    br = br or _new(seed, individuality, backend)
    rec = None
    if folder is not None:
        rec = recorder.Recorder(br, rec_groups(br)).start()
    sd = int(seed)
    try:
        if rig == "tethered":
            r = rigs.run_tethered(br, tethered_schedule(p["omega"], p["seconds"]), p["mode"], gain=float(p["gain"]), cancel=cancel, progress=progress)
            f = r["_full"]
            summ = dict(response_right_hz=float(_win(f, 0, 99, "steer_hz")[(_win(f, 0, 99, "omega_ext") > 0)].mean()),
                        response_left_hz=float(_win(f, 0, 99, "steer_hz")[(_win(f, 0, 99, "omega_ext") < 0)].mean()),
                        calm_hz=float(_win(f, 0, 99, "steer_hz")[(_win(f, 0, 99, "omega_ext") == 0)].mean()),
                        mean_abs_slip=float(np.abs(_win(f, 0, 99, "slip")).mean()))
        elif rig == "ball":
            om = float(p["omega"])
            r = rigs.run_ball(br, p["scene"], p["mode"], seconds=float(p["seconds"]), omega_ext=om, bar_offset=math.radians(float(p["bar_deg"])),
                              cancel=cancel, progress=progress)
            f = r["_full"]
            last = float(p["seconds"]) - LAST_S
            summ = dict(forward_speed_mean=float(f[:, rigs.COLS.index("speed")].mean()), end_x=float(f[-1, 6]), end_y=float(f[-1, 7]),
                        mean_abs_yaw_rate=float(np.abs(f[:, rigs.COLS.index("yaw_rate")]).mean()),
                        final_bar_error_deg=float(np.degrees(np.abs(_win(f, last, 1e9, "aux")).mean())) if p["scene"] == "bar" else None)
        elif rig == "buridan":
            r = rigs.run_buridan(br, float(p["seconds"]), stripes=p["mode"] == "stripes", start_heading=_start_heading(sd), cancel=cancel, progress=progress)
            f = r["_full"]
            summ = dict(stripe_deviation_deg=rigs.stripe_deviation(f), transits=rigs.transits(f), edge_contacts=int(f[-1, rigs.COLS.index("aux")]),
                        mean_speed=float(f[:, rigs.COLS.index("speed")].mean()))
        else:
            r = rigs.run_fourfield(br, float(p["seconds"]), odor=ODOR, deliver=p["mode"] == "odor", start_heading=_start_heading(sd), cancel=cancel, progress=progress)
            f = r["_full"]
            summ = rigs.preference_index(f, r["dt"])
    finally:
        if rec is not None:
            rec.stop()
    out = dict(kind=f"rig_{rig}", rig=rig, seed=sd, individuality=individuality, params=p, cols=r["cols"], trace=r["trace"], seconds=r["seconds"], dt=r["dt"],
               summary=summ, tags=rigs.RIG_TAGS[rig], title=rigs.RIG_TITLE[rig], rule_version=rigs.RULE_VERSION,
               geometry={k: r[k] for k in r if k in ("platform_radius", "stripe_distance", "stripe_bearings_deg", "field_half", "odor_quadrants", "bar_offset")})
    if folder is not None:
        out["files"] = export_scene(out, folder, rec, br)
    return out


def export_scene(res: dict, folder: Path, rec, br) -> list[str]:
    """The recorder's files (-spikes.csv, -rates.csv, -group-rates.csv, -events.csv, -kinematics.csv, .npz, -metadata.json) for this run,
    plus rig_trace.csv (the decimated trace, every column) and rig_summary.json."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{res['rig']}-seed{res['seed']}"
    meta = dict(rig=res["rig"], params=res["params"], seed=res["seed"], individuality=res["individuality"], tags=res["tags"],
                rule_version=res["rule_version"], summary=res["summary"])
    files = [p.name for p in rec.save(stem, meta)] if rec is not None else []
    p = folder / f"{res['rig']}-seed{res['seed']}-rig_trace.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(res["cols"])
        w.writerows(res["trace"])
    files.append(p.name)
    p = folder / f"{res['rig']}-seed{res['seed']}-rig_summary.json"
    p.write_text(json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=1, default=str), encoding="utf-8")
    files.append(p.name)
    return files


# --- the pre-registered analysis -----------------------------------------------------------------------------------------------------
def _crit(cid: str, label: str, diffs, min_effect: float, unit: str, extra: dict | None = None) -> dict:
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    p = _wilcoxon_greater(d)
    mean = float(d.mean()) if len(d) else float("nan")
    return dict(id=cid, label=label, n=int(len(d)), mean_effect=mean, sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0, min_effect=min_effect, unit=unit,
                p=p, passed=bool(len(d) and mean >= min_effect and p < P_MAX), **(extra or {}))


def analyze(rig: str, flies: list[dict]) -> dict:
    """The criteria of one rig on its per-fly measurements (see the module docstring for what each says and why)."""
    f = flies
    if rig == "tethered":
        crit = [_crit("T1", "open loop: right-minus-left steering after a rightward minus a leftward panorama rotation (3 rad/s), Hz",
                      [x["d_directional"] for x in f], 3.0, "Hz"),
                _crit("T2", "closed loop: share of an imposed 1 rad/s rotation the fly cancels (yaw rate / rotation, calm baseline removed)",
                      [x["compensation_mean"] for x in f], 0.10, "fraction")]
    elif rig == "ball":
        crit = [_crit("BL1", "closed loop, walking: share of an imposed 1 rad/s drift of the VR panorama the fly cancels",
                      [x["compensation_mean"] for x in f], 0.10, "fraction"),
                _crit("BL2", "closed-loop bar: heading error (deg) with the bar hidden minus with it visible, last 5 s of 20 s",
                      [x["bar_error_deg_hidden"] - x["bar_error_deg_visible"] for x in f], 10.0, "deg",
                      dict(visible_mean_deg=float(np.mean([x["bar_error_deg_visible"] for x in f])),
                           visible_below_30=bool(np.mean([x["bar_error_deg_visible"] for x in f]) < 30.0)))]
        crit[1]["passed"] = bool(crit[1]["passed"] and crit[1]["visible_below_30"])
    elif rig == "buridan":
        crit = [_crit("B1", "stripe deviation (deg) without stripes minus with stripes (median over the run, 0 = along the stripe axis)",
                      [x["none"]["deviation_deg"] - x["stripes"]["deviation_deg"] for x in f], 10.0, "deg"),
                _crit("B2", "transits between the stripes with stripes minus without", [x["stripes"]["transits"] - x["none"]["transits"] for x in f], 3.0, "transits")]
    elif rig == "fourfield":
        crit = [_crit("F1", "preference index with the odor minus the sham (same arena and start, no odor), positive = prefers the odor quadrants",
                      [x["odor"]["pi"] - x["sham"]["pi"] for x in f], 0.15, "PI")]
    else:
        raise ValueError(rig)
    extra = {}
    if rig == "fourfield":
        so = [x["odor"]["speed_odor"] - x["odor"]["speed_air"] for x in f]
        extra["speed_odor_minus_air_m_s"] = dict(mean=float(np.nanmean(so)), p_greater=_wilcoxon_greater(so), note="reported, not a criterion (a kinesis would show here)")
    return dict(rig=rig, n=len(f), criteria=crit, all_passed=all(c["passed"] for c in crit), extra=extra)


def run_assay(rig: str, seeds=VALIDATION_SEEDS, modes=(DEFAULT_MODE, CONTROL_MODE), workers: int = 1, backend: str | None = None, progress=None, cancel=None,
              play=None) -> dict:
    """The assay: the rig's per-fly measurements and criteria for the default individuality mode and for the off control."""
    t0 = time.time()
    out = {}
    for mode in modes:
        flies = run_flies(rig, seeds, mode, workers, backend, progress, cancel, play)
        out[mode] = dict(flies=flies, analysis=analyze(rig, flies))
    return dict(kind=f"rig_assay_{rig}", rig=rig, created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1), seeds=[int(s) for s in seeds],
                modes=list(modes), tags=rigs.RIG_TAGS[rig], title=rigs.RIG_TITLE[rig], rule_version=rigs.RULE_VERSION, results=out)


def summary(res: dict) -> str:
    from kickthefly.lab import labstats

    lines = [f"{res['title'].upper()} ASSAY, {len(res['seeds'])} flies (seeds {res['seeds'][0]}-{res['seeds'][-1]})"]
    for mode, r in res["results"].items():
        lines.append(f"  individuality {mode}{'  (the control)' if mode == CONTROL_MODE else ''}:")
        for c in r["analysis"]["criteria"]:
            lines.append(f"    {c['id']} {'PASS' if c['passed'] else 'FAIL'}: {c['label']}: {c['mean_effect']:+.3g} {c['unit']} "
                         f"(needs >= {c['min_effect']:g}), {labstats.fmt_p(c['p'])}, n = {c['n']}")
        for k, v in r["analysis"].get("extra", {}).items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def save(res: dict, folder: Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"rig_assay_{res['rig']}.json"
    p.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    out = [p]
    p = folder / f"rig_assay_{res['rig']}_criteria.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["individuality", "id", "label", "n", "mean_effect", "unit", "min_effect", "p", "passed"])
        for mode, r in res["results"].items():
            for c in r["analysis"]["criteria"]:
                w.writerow([mode, c["id"], c["label"], c["n"], c["mean_effect"], c["unit"], c["min_effect"], c["p"], int(c["passed"])])
    out.append(p)
    return out


def main(args) -> int:
    """--headless --rig NAME [--seeds A-B] [--individuality MODE] [--out DIR]: one fly (the first seed) in the rig, recorded in the Lab's format.
    --headless --rig-assay NAME [--seeds A-B] [--workers N] [--out DIR]: the pre-registered assay, default mode and the off control."""
    from kickthefly.lab import headless, recorder

    assay = getattr(args, "rig_assay", None)
    name = assay or getattr(args, "rig", None)
    if name not in SCENES:
        print(f"error: unknown rig {name!r}; choose from {', '.join(SCENES)}", flush=True)
        return 2
    mode = getattr(args, "individuality", None) or DEFAULT_MODE
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-rig-{name}"
    t0 = time.time()
    if assay:
        seeds = headless.parse_seeds(args.seeds, VALIDATION_SEEDS)
        res = run_assay(assay, seeds, workers=max(1, args.workers or 1), progress=lambda d, n, label: print(f"  {d}/{n} {label}", flush=True))
        save(res, folder)
        print(summary(res))
        print(f"results written to {folder} ({time.time() - t0:.0f} s)")
        return 0
    seeds = headless.parse_seeds(args.seeds, (0,))
    be = getattr(args, "sim_backend", None)
    res = scene_run(name, seeds[0], mode, folder=folder, backend=None if be in (None, "auto") else be)
    print(f"{res['title']}: seed {res['seed']}, individuality {mode}, {res['seconds']:g} s simulated")
    for k, v in res["summary"].items():
        print(f"  {k}: {v}")
    print(f"files written to {folder}: {', '.join(res['files'])}")
    return 0
