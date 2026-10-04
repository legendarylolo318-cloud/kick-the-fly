"""Sleep-deprivation assay (3.0 day 4): keep a fly awake through a stretch of the night, then measure its rebound sleep against an
undisturbed control of the same seed. Uses the day/night cycle (outdoors.diurnal_cycle) and the dFB readout (FB6/FB7).

WHAT IS WHAT (the assay's output prints the same table)
  GAME RULE      SLEEP PRESSURE, a number between 0 and 1: it rises while the fly is awake and falls while it sleeps (the pet's
                 rule, core/pet.py: recovery twice as fast as the build-up), here on a compressed clock. Also game rules: the
                 compressed day, the disturbances (what they are and how often), that a disturbance wakes the fly and holds it
                 awake, what counts as sleep (dFB above THRESH["sleep"] for a second without a break), and the current that
                 pressure becomes: DFB_CURRENT_AT_FULL * pressure on every FB6/FB7 neuron.
  CONNECTOME     what that current does: the dFB neurons' firing, read as a multiple of their own calm rate (the same readout the
                 game's SLEEP uses), through their real synapses, individuality and noise; daylight reaching the photoreceptors and
                 the morning clock neurons l-LNv/s-LNv; the touch neurons a disturbance fires. The dFB response curve (current ->
                 firing) is the connectome's and is not a rule.
  MODEL PREDICTION  the rebound itself. A rebound is EXPECTED from the pressure rule (a fly that was kept awake has more pressure to
                 discharge): what the connectome adds is how much current the dFB needs before it crosses the sleep threshold, and
                 that is what the dFB columns report. This assay does not show that the fly's brain has a sleep homeostat; it shows
                 that a game-rule homeostat acting through the real dFB neurons produces a rebound, and how large.

Pre-registered criteria (written before the run; seeds 1000-1009; the constants below were fixed beforehand and chosen on exploration
seeds 11 and 12 for the dFB response only (current 0.04 gave 2.8-2.9x, 0.08 gave 4.6-4.7x); nothing was tuned to make a criterion pass):
  S1 rebound: sleep in the recovery window, deprived minus control, one-sided Wilcoxon signed-rank over the seeds, p < 0.01 and mean > 0.
  S2 a manipulation check (expected to pass by construction of the rule, which holds a touched fly awake for longer than the gap to the
     next touch; it exists to catch a broken device): the deprived flies slept at most 25% as long as their controls during the window.
  S3 the dFB readout: mean dFB level at the end of the deprivation window, deprived above control, one-sided Wilcoxon p < 0.01.
With n = 10 an exact one-sided Wilcoxon cannot go below p = 0.00098; that floor is reported as p < 0.001.
"""
from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# --- GAME RULE constants -------------------------------------------------------------------------------------------------------
DAY_S = 300.0                         # a compressed day (the cycle of outdoors.diurnal_cycle); night is about 39% of it
START_PHASE = 0.80                    # the run starts at this fraction of the day: dusk
SETTLE_S, DEPRIVE_S, RECOVER_S = 10.0, 45.0, 60.0
START_PRESSURE = 0.10                 # the pet's own starting value
RISE_PER_S = 1.0 / 60.0               # pressure per second awake (the pet's 16 h awake, compressed)
FALL_PER_S = 1.0 / 30.0               # per second asleep (the pet's 8 h of sleep, compressed: twice as fast)
DFB_CURRENT_AT_FULL = 0.08            # current on each FB6/FB7 neuron at pressure 1 (x ext_gain inside the sim)
SLEEP_AFTER_S = 1.0                   # dFB above threshold this long without a break = asleep
DISTURB_EVERY_S, WAKE_HOLD_S = 3.0, 2.0   # a touch every 3 s holds the fly awake for 2 s: the 1 s gap is shorter than a sleep bout (SLEEP_AFTER_S)
DISTURB_POKE = 0.6                    # touch strength of a disturbance on the legs and the body
TICK_STEPS = 4
CRITERIA = dict(p_max=0.01, deprived_share_max=0.25)
TAGS = dict(sleep_pressure_and_disturbance="GAME RULE", dfb_firing_and_daylight="CONNECTOME", rebound="MODEL PREDICTION")


def _phase_times() -> tuple[float, float, float]:
    t0 = START_PHASE * DAY_S
    return t0, t0 + SETTLE_S, t0 + SETTLE_S + DEPRIVE_S


def run_fly(seed: int, disturb: bool, mode: str = "off", backend: str | None = None, deprive_s: float = DEPRIVE_S,
            recover_s: float = RECOVER_S, settle_s: float = SETTLE_S, brain=None, progress=None, cancel=None, params=None) -> dict:
    """One fly through settle -> deprivation window -> recovery window. disturb=False is its undisturbed control."""
    from kickthefly.core import simcore
    from kickthefly.game import kick_the_fly as k
    from kickthefly.game import outdoors

    br = brain or simcore.new_brain(seed=int(seed), individuality=mode, backend=backend, params=params)
    rows = br.sense[("sleep", None)]
    dt = TICK_STEPS * float(br.dt)
    t0 = START_PHASE * DAY_S
    total = settle_s + deprive_s + recover_s
    ticks = int(round(total / dt))
    pressure, above_since, asleep, wake_until, next_disturb = START_PRESSURE, None, False, 0.0, settle_s
    sleep_s = dict(settle=0.0, deprivation=0.0, recovery=0.0)
    dfb_last, first_sleep_after = [], None
    peak = pressure
    trace = []
    for tick in range(ticks):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        t = tick * dt
        if settle_s <= t < settle_s + deprive_s:
            phase = "deprivation"
        elif t < settle_s:
            phase = "settle"
        else:
            phase = "recovery"
        az, el = outdoors.diurnal_cycle(t0 + t, DAY_S)                       # the cycle: daylight reaches the eyes and the clock
        left, right = outdoors.sun_light(0.0, az, el)
        if left > 0.02:
            br.poke("light", "L", left, recruit=0.25 * left)
        if right > 0.02:
            br.poke("light", "R", right, recruit=0.25 * right)
        clock = outdoors.circadian_clock_drive(el)["morning_cells"]
        if clock > 0.02:
            br.poke("clock", None, clock)
        if disturb and phase == "deprivation" and t >= next_disturb:         # the deprivation device: touch, then forced awake
            br.poke("legs", None, DISTURB_POKE)
            br.poke("body", None, DISTURB_POKE)
            wake_until = t + WAKE_HOLD_S
            next_disturb += DISTURB_EVERY_S
            asleep, above_since = False, None
        br.set_current("sleep_pressure", rows, DFB_CURRENT_AT_FULL * pressure)   # sleep pressure drives the real dFB neurons
        for _ in range(TICK_STEPS):
            br._step()
        lvl = float(br.level("sleep"))
        if lvl > k.THRESH["sleep"] and t >= wake_until:
            if above_since is None:
                above_since = t
            if not asleep and t - above_since >= SLEEP_AFTER_S:
                asleep = True
        else:
            asleep, above_since = False, None
        if asleep:
            pressure = max(0.0, pressure - FALL_PER_S * dt)
            sleep_s[phase] += dt
            if phase == "recovery" and first_sleep_after is None:
                first_sleep_after = t - (settle_s + deprive_s)
        else:
            pressure = min(1.0, pressure + RISE_PER_S * dt)
        peak = max(peak, pressure)
        if phase == "deprivation" and t >= settle_s + deprive_s - 5.0:
            dfb_last.append(lvl)
        if tick % 25 == 0:
            trace.append([round(t, 2), round(pressure, 3), round(lvl, 2), int(asleep)])
        if progress and tick % 100 == 0:
            progress(tick / ticks)
    br.clear_current("sleep_pressure")
    return dict(seed=int(seed), disturbed=bool(disturb), sleep_s={k_: round(v, 2) for k_, v in sleep_s.items()},
                pressure_end_of_deprivation=_at(trace, settle_s + deprive_s), peak_pressure=round(peak, 3),
                dfb_level_end_of_deprivation=float(np.mean(dfb_last)) if dfb_last else float("nan"),
                first_sleep_after_deprivation_s=first_sleep_after, trace=trace)


def _at(trace, t) -> float:
    best = min(trace, key=lambda r: abs(r[0] - t))
    return best[1]


def _task(args: tuple) -> dict:
    return run_fly(*args)


def fly_pair(seed: int, mode: str = "off", backend: str | None = None, params=None, deprive_s: float = DEPRIVE_S,
             recover_s: float = RECOVER_S, **kw) -> dict:
    """Both conditions for one seed (the Lab assay and protocol path); the control and deprived flies share the seed."""
    kw = dict(kw, params=params, deprive_s=deprive_s, recover_s=recover_s)
    return dict(seed=int(seed), deprive_s=deprive_s, recover_s=recover_s, individuality=mode,
                control=run_fly(seed, False, mode, backend, **kw), deprived=run_fly(seed, True, mode, backend, **kw))


def summarize(flies: list[dict]) -> dict:
    """The assay's statistics over fly_pair results (the labjobs path): the same criteria and aggregation as run()."""
    seeds = [f["seed"] for f in flies]
    res = aggregate(seeds, [f["control"] for f in flies], [f["deprived"] for f in flies], flies[0]["individuality"],
                    flies[0]["deprive_s"], flies[0]["recover_s"], 0.0)
    return dict(metric="rebound sleep, deprived minus control (s) - GAME RULE pressure through the CONNECTOME dFB; MODEL PREDICTION",
                criteria=res["criteria"], mean=res["mean"], all_passed=res["all_passed"], components=[list(c) for c in COMPONENTS],
                per_fly=[float(d["sleep_s"]["recovery"] - c["sleep_s"]["recovery"]) for c, d in zip((f["control"] for f in flies), (f["deprived"] for f in flies))])


def run(seeds, mode: str = "off", workers: int = 1, backend: str | None = None, deprive_s: float = DEPRIVE_S, recover_s: float = RECOVER_S,
        progress=None, cancel=None, play=None) -> dict:
    """The assay over the seeds. play: a function (args) -> run_fly result, replacing the real brains (tests)."""
    seeds = [int(s) for s in seeds]
    tasks = [(s, d, mode, backend, deprive_s, recover_s) for s in seeds for d in (False, True)]
    t0 = time.time()
    results: dict[tuple[int, bool], dict] = {}
    if play is not None:
        for t in tasks:
            results[(t[0], t[1])] = play(t)
    elif workers > 1:
        import multiprocessing

        ex = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        try:                                   # 3.0 day 4 review: leaving a `with` block waited for every queued fly after Cancel
            futs = {ex.submit(_task, t): t for t in tasks}
            for f in as_completed(futs):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")
                r = f.result()
                results[(r["seed"], r["disturbed"])] = r
                if progress:
                    progress(len(results), len(tasks), f"fly {r['seed']} {'deprived' if r['disturbed'] else 'control'}")
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    else:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            r = _task(t)
            results[(r["seed"], r["disturbed"])] = r
            if progress:
                progress(len(results), len(tasks), f"fly {r['seed']} {'deprived' if r['disturbed'] else 'control'}")
    ctrl = [results[(s, False)] for s in seeds]
    dep = [results[(s, True)] for s in seeds]
    return aggregate(seeds, ctrl, dep, mode, deprive_s, recover_s, time.time() - t0)


def aggregate(seeds, ctrl, dep, mode, deprive_s, recover_s, seconds) -> dict:
    from kickthefly.lab import labstats

    rebound_c = np.array([r["sleep_s"]["recovery"] for r in ctrl])
    rebound_d = np.array([r["sleep_s"]["recovery"] for r in dep])
    sd_c = np.array([r["sleep_s"]["deprivation"] for r in ctrl])
    sd_d = np.array([r["sleep_s"]["deprivation"] for r in dep])
    dfb_c = np.array([r["dfb_level_end_of_deprivation"] for r in ctrl])
    dfb_d = np.array([r["dfb_level_end_of_deprivation"] for r in dep])
    p1 = _wilcoxon_greater(rebound_d, rebound_c)
    p3 = _wilcoxon_greater(dfb_d, dfb_c)
    share = float(sd_d.mean() / sd_c.mean()) if sd_c.mean() > 0 else float("nan")
    diff = rebound_d - rebound_c
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    crit = dict(
        S1=dict(label="rebound sleep, deprived - control (s)", mean_difference=float(diff.mean()), dz=float(diff.mean() / sd) if sd > 1e-12 else float("nan"),
                p=p1, passed=bool(p1 < CRITERIA["p_max"] and diff.mean() > 0)),
        S2=dict(label="deprived sleep as a share of control sleep during the window", share=share,
                passed=bool(share == share and share <= CRITERIA["deprived_share_max"])),
        S3=dict(label="dFB level at the end of the window, deprived - control", mean_difference=float(np.nanmean(dfb_d - dfb_c)), p=p3,
                passed=bool(p3 < CRITERIA["p_max"] and np.nanmean(dfb_d - dfb_c) > 0)))
    return dict(kind="sleep_deprivation", created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(seconds, 1), seeds=seeds,
                individuality=mode, timeline=dict(day_s=DAY_S, start_phase=START_PHASE, settle_s=SETTLE_S, deprive_s=deprive_s, recover_s=recover_s),
                rules=dict(rise_per_s=RISE_PER_S, fall_per_s=FALL_PER_S, dfb_current_at_full=DFB_CURRENT_AT_FULL, sleep_after_s=SLEEP_AFTER_S,
                           disturb_every_s=DISTURB_EVERY_S, wake_hold_s=WAKE_HOLD_S, start_pressure=START_PRESSURE),
                control=ctrl, deprived=dep,
                mean=dict(recovery_sleep_control=float(rebound_c.mean()), recovery_sleep_deprived=float(rebound_d.mean()),
                          deprivation_sleep_control=float(sd_c.mean()), deprivation_sleep_deprived=float(sd_d.mean()),
                          dfb_control=float(np.nanmean(dfb_c)), dfb_deprived=float(np.nanmean(dfb_d)),
                          pressure_control=float(np.mean([r["pressure_end_of_deprivation"] for r in ctrl])),
                          pressure_deprived=float(np.mean([r["pressure_end_of_deprivation"] for r in dep]))),
                criteria=crit, all_passed=all(c["passed"] for c in crit.values()), tags=TAGS,
                components=COMPONENTS, p_text=labstats.fmt_p(p1))


COMPONENTS = (
    ("sleep pressure (rises awake, falls asleep)", "GAME RULE"),
    ("the disturbances and the forced wake", "GAME RULE"),
    ("what counts as sleep: dFB above THRESH['sleep'] for 1 s", "GAME RULE"),
    ("pressure -> current on every FB6/FB7 neuron", "GAME RULE"),
    ("dFB (FB6/FB7) firing in response to that current", "CONNECTOME readout"),
    ("daylight on the photoreceptors and the l-LNv/s-LNv clock neurons", "CONNECTOME (the drive is a game rule)"),
    ("the rebound itself", "MODEL PREDICTION"),
)


def _wilcoxon_greater(a, b) -> float:
    from scipy import stats

    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)


def summary(res: dict) -> str:
    from kickthefly.lab import labstats

    m, c, tl = res["mean"], res["criteria"], res["timeline"]
    lines = [f"SLEEP DEPRIVATION ASSAY, {len(res['seeds'])} flies (seeds {res['seeds'][0]}-{res['seeds'][-1]}), paired: each seed is its own control",
             f"  timeline: day {tl['day_s']:g} s compressed, from {tl['start_phase']:.0%} of the day (dusk): {tl['settle_s']:g} s settle, {tl['deprive_s']:g} s "
             f"deprivation (a touch every {res['rules']['disturb_every_s']:g} s), then {tl['recover_s']:g} s undisturbed recovery",
             f"  sleep during the window: control {m['deprivation_sleep_control']:.1f} s, deprived {m['deprivation_sleep_deprived']:.1f} s",
             f"  sleep pressure at the end of the window (GAME RULE): control {m['pressure_control']:.2f}, deprived {m['pressure_deprived']:.2f}",
             f"  dFB level at the end of the window (CONNECTOME readout, x calm): control {m['dfb_control']:.2f}, deprived {m['dfb_deprived']:.2f}",
             f"  sleep in the recovery window: control {m['recovery_sleep_control']:.1f} s, deprived {m['recovery_sleep_deprived']:.1f} s"]
    for cid, x in c.items():
        extra = (f", mean {x['mean_difference']:+.2f}" + (f", dz {x['dz']:.2f}" if "dz" in x else "") + f", {labstats.fmt_p(x['p'])}") if "p" in x else f", share {x['share']:.2f}"
        lines.append(f"  {cid} {'PASS' if x['passed'] else 'FAIL'}: {x['label']}{extra}")
    lines.append("  what is what: " + "; ".join(f"{a} = {b}" for a, b in COMPONENTS))
    lines.append("  A rebound is expected from the pressure rule; this shows it works through the real dFB neurons, not that the brain has a homeostat.")
    return "\n".join(lines)


def save(res: dict, folder: Path | str) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "sleep_deprivation.json"
    slim = dict(res)
    p.write_text(json.dumps(slim, indent=1), encoding="utf-8")
    q = folder / "sleep_deprivation.csv"
    with q.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "condition", "sleep_settle_s", "sleep_deprivation_s", "sleep_recovery_s", "pressure_end_of_deprivation",
                    "dfb_level_end_of_deprivation", "first_sleep_after_deprivation_s", "tag_pressure", "tag_dfb"])
        for r in res["control"] + res["deprived"]:
            w.writerow([r["seed"], "deprived" if r["disturbed"] else "control", r["sleep_s"]["settle"], r["sleep_s"]["deprivation"], r["sleep_s"]["recovery"],
                        r["pressure_end_of_deprivation"], r["dfb_level_end_of_deprivation"], r["first_sleep_after_deprivation_s"], "GAME RULE",
                        "CONNECTOME readout"])
    return [p, q]


def main(args) -> int:
    from kickthefly.lab import headless, recorder

    seeds = headless.parse_seeds(args.seeds, range(1000, 1010))
    mode = getattr(args, "individuality", None) or "off"
    t0 = time.time()
    res = run(seeds, mode=mode, workers=max(1, args.workers or 1),
              progress=lambda d, n, label: print(f"  {d}/{n} {label} ({time.time() - t0:.0f}s)", flush=True))
    folder = Path(args.out) if args.out else recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-sleep-deprivation"
    save(res, folder)
    print(summary(res))
    print(f"results written to {folder}")
    return 0 if res["all_passed"] else 1
