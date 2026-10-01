"""Predator escape assay (3.0 day 3): how often does a fly escape each predator, with a 95% confidence interval.

For each predator (frog, dragonfly, mantis) the engine in game/predators.py runs one attack against a fly that stays where
it is. What the fly's eyes would see (the body and the striking part, as growing circles) is fed to the REAL brain through
the game's own looming transduction (the same formula as assays.looming_fly and the game's `_vision`), and the fly has
"escaped" if its giant fiber DNp01 crossed THRESH["escape"] x calm before the capture. Headless and deterministic: the same
seed gives the same attack, the same brain noise and the same result.

What is what:
  CONNECTOME      LPLC2/LC4 -> DNp01 and everything the simulation does between them.
  GAME RULE       the predators' behaviour (game/predators.py), the loom transduction threshold, and "escaped = DNp01 above
                  THRESH['escape'] before the capture". The fly does not move in the assay, so an "escape" is the brain's
                  decision to dodge, not a flight path that does or does not clear the tongue.
  MODEL PREDICTION  the probabilities this prints. They are what this model of the fly does with this model of the
                  predator, not measurements of any real animal, and they move if a game rule moves.

Pre-registered before the first run (fixed here, on exploration seeds 0-299; scored once on validation seeds 1000-1009):
  P1  the mantis's creep never shows the fly an object expanding faster than LOOM_MIN. This is geometry, checked per trace,
      with no brain: it holds or it doesn't.
  P2  the mantis creep does not make the fly escape: in at least 9 of 10 flies (seeds 1000-1009) DNp01 stays below the escape
      threshold until the strike begins.
  Nothing else is a criterion. The other probabilities are reported, not judged: a frog or dragonfly "should" escape or not as
  the model decides.
"""
from __future__ import annotations

import math

import numpy as np

from kickthefly.core import simcore
from kickthefly.game import predators as pr

CRITERIA = (
    "P1  the mantis's approach, before it strikes, never expands faster than LOOM_MIN (geometry, per trace)",
    "P2  in at least 9 of 10 flies (seeds 1000-1009) DNp01 stays below the escape threshold during the mantis's creep, until the strike begins",
)
TRIALS_PER_KIND = 3
WARM_FRAMES = 120                    # frames of brain time simulated before the first visible loom (2 s)
SETTLE_STEPS = 400


def loom_rates(tr: dict) -> list[float]:
    """The game's loom measure for each frame of a trace: the largest angular expansion speed (rad/s) of any threat."""
    head = np.asarray(tr["head"], float)
    prev: dict = {}
    out = []
    for frame in tr["frames"]:
        seen, best = {}, 0.0
        for key, pos, r in frame:
            d = max(float(np.linalg.norm(np.asarray(pos, float) - head)), r + 0.02)
            theta = 2 * math.atan(r / d)
            seen[key] = theta
            if key in prev and (theta - prev[key]) * 60.0 > best:
                best = (theta - prev[key]) * 60.0
        prev = seen
        out.append(best)
    return out


def escape_trial(br, tr: dict) -> dict:
    """One attack against one brain. Returns whether DNp01 crossed the escape threshold before the capture, when, and
    whether it did so before the strike began (the mantis's 'did it notice the creep' question)."""
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import assays

    rates = loom_rates(tr)
    n = len(rates)
    end = tr["capture_frame"] if tr["capture_frame"] is not None else n - 1
    strike = tr["strike_frame"] if tr["strike_frame"] is not None else end
    first = next((i for i, r in enumerate(rates) if r > k.LOOM_MIN), end)
    start = max(0, first - WARM_FRAMES)                 # nothing is delivered before the first visible loom: a calm brain
    escaped_at, escaped_before_strike, peak, tick = None, False, 0.0, 0
    for i in range(start, end + 1):
        strength = float(np.clip((rates[i] - k.LOOM_MIN) / k.LOOM_FULL, 0, 1))
        if strength > 0:
            br.poke("loom", None, strength, recruit=0.6 * strength)
        for _ in range(assays._tick_steps(tick)):
            br._step()
        tick += 1
        lvl = br.level("escape")
        peak = max(peak, lvl)
        if escaped_at is None and lvl > k.THRESH["escape"]:
            escaped_at = i
            escaped_before_strike = i < strike
            break
    captured = tr["capture_frame"] is not None
    return dict(kind=tr["kind"], escaped=escaped_at is not None, captured=captured,
                latency_s=None if escaped_at is None else (escaped_at - first) / 60.0,
                lead_s=None if escaped_at is None else (end - escaped_at) / 60.0,
                noticed_before_strike=bool(escaped_before_strike), peak_level=float(peak),
                max_loom_before_strike=float(max(rates[:strike + 1], default=0.0)),
                strike_loom_peak=float(max(rates[strike:end + 1], default=0.0)), frames=end + 1)


def escape_fly(seed: int, kinds=pr.KINDS, trials: int = TRIALS_PER_KIND, surgery: dict | None = None,
               params: dict | None = None, brain=None, wiring=None) -> dict:
    """One fly (seed) faces each predator `trials` times. `surgery` as in the other assays."""
    from kickthefly.lab import assays

    kinds = tuple(kinds)
    bad = [k for k in kinds if k not in pr.SPECS]
    if bad:
        raise ValueError(f"unknown predator {bad[0]!r}; use one of {', '.join(pr.KINDS)}")
    br = brain or simcore.new_brain(seed=seed, params=params, wiring=wiring)
    assays.apply_surgery(br, surgery)
    assays.rest(br, SETTLE_STEPS)
    out = {}
    for kind in kinds:
        rows = []
        for j in range(int(trials)):
            tr = pr.trace(kind, seed * 1000 + j)
            rows.append(escape_trial(br, tr))
            assays.rest(br, SETTLE_STEPS)
        out[kind] = rows
    return dict(seed=seed, kinds=list(kinds), trials=out)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson 95% interval for k of n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(flies: list[dict]) -> dict:
    """Escape probability by predator with a Wilson 95% CI over all trials, plus the per-fly spread."""
    from kickthefly.lab import labstats

    rows = []
    for kind in flies[0]["kinds"]:
        trials = [t for f in flies for t in f["trials"][kind]]
        k, n = sum(t["escaped"] for t in trials), len(trials)
        lo, hi = wilson(k, n)
        per_fly = [float(np.mean([t["escaped"] for t in f["trials"][kind]])) for f in flies]
        lead = [t["lead_s"] for t in trials if t["escaped"] and t["lead_s"] is not None]
        rows.append(dict(predator=kind, escapes=int(k), trials=int(n), escape_probability=k / n if n else float("nan"),
                         ci95=[lo, hi], per_fly=labstats.mean_ci(per_fly),
                         lead_s=labstats.mean_ci(lead) if lead else None,
                         noticed_before_strike=int(sum(t["noticed_before_strike"] for t in trials)),
                         max_loom_before_strike=float(max(t["max_loom_before_strike"] for t in trials))))
    return dict(metric="escape probability by predator (Wilson 95% CI over trials) - MODEL PREDICTION", rows=rows,
                per_fly=[float(np.mean([np.mean([t["escaped"] for t in f["trials"][k]]) for k in f["kinds"]]))
                         for f in flies])


def verdict(flies: list[dict]) -> dict:
    """Score the pre-registered criteria (CRITERIA). Meaningful for ten flies, seeds 1000-1009."""
    from kickthefly.game import kick_the_fly as k

    mantis = [f for f in flies if "mantis" in f["trials"]]
    if not mantis:
        return dict(P1=None, P2=None, passed=None, flies=len(flies))
    # P1 comes from the traces, not from the brain: the largest rate the fly's eyes see before the strike frame
    p1 = all(t["max_loom_before_strike"] <= k.LOOM_MIN for f in mantis for t in f["trials"]["mantis"])
    calm = sum(1 for f in mantis if not any(t["noticed_before_strike"] for t in f["trials"]["mantis"]))
    p2 = calm >= math.ceil(0.9 * len(mantis))
    return dict(P1=bool(p1), P2=bool(p2), flies_that_did_not_notice_the_creep=int(calm), flies=len(mantis),
                passed=bool(p1 and p2))
