"""Fly racing (3.0 day 4): flies race down a track through their own brains and their own individuality.

CONNECTOME   the walking command neurons' firing (DNp09, "P9") sets how fast a fly goes; the smell of a sugar or fruit lure ahead
             drives the real olfactory neurons of that scent (the same scent pokes the game's tools use), and touching a lure drives
             the sugar-pathway taste neurons and the PAM reward neurons exactly as sugar and ripe fruit do in the game. What those
             neurons then do to the walking neurons is the wiring's, and each fly's own individuality gains and noise are its own.
GAME RULE    the track (8 m, three lures), how walking drive becomes speed (below), the smell range and contact distance, the pulse
             strengths (the game's own: scent 0.3, taste 0.5, reward 0.5), the time cap, the lanes (flies never touch or block one
             another, so every lane is simulated on its own), and everything about betting (core/points.py, lab/racing.py: points
             only, no money, nothing to buy).
MODEL PREDICTION  the finishing order, and whether individuality predicts it.

Speed rule (GAME RULE, version 2 since the 3.0 day 4 review): speed = V_MAX * clip(walk level / THRESH["walk"], 0, 1), where the walk
level is DNp09's firing divided by ONE FIXED reference rate, WALK_REF_HZ, the same for every fly, and THRESH["walk"] is the game's own
walking threshold (3.0x): a fly firing at the reference rate goes a third as fast as one whose walking command is three times above it.
Version 1 divided by each fly's own calm baseline instead, which the brain estimates during its 3 s warm-up from its seed's noise: those
baselines range 1.75-6.43 Hz between seeds, so a fly's speed was mostly its warm-up's accident, repeatable even with identical brains
(docs/racing.md). WALK_REF_HZ is the mean of that baseline (floored at the game's 2 Hz, as Brain.level does) over exploration seeds 0-15
with individuality off (tools/race_diagnosis.py); it was fixed from those before any held-out run of version 2. Nothing is tuned to
make a race close or an order predictable.

The engine takes a brain-like object (poke, level, _step, dt) so it can be tested without the real pack.
"""
from __future__ import annotations

import numpy as np

TRACK_LENGTH = 8.0                    # m
V_MAX = 0.6                           # m/s at a walking level at or above the game's threshold
LURES = ((2.0, "sugar"), (4.0, "fruit"), (6.0, "sugar"))      # (distance along the track, kind)
SMELL_RANGE, CONTACT = 1.5, 0.15      # m
SCENT_POKE, TASTE_POKE, REWARD_POKE = 0.3, 0.5, 0.5          # the game's own pulse strengths for a scent, a sip, a reward
TICK_STEPS = 4                        # brain steps (5 ms) per engine tick: 20 ms
TIME_CAP_S = 90.0
TRACE_EVERY = 10                      # a trace sample every 10 ticks (5 per second)
MAX_LANES = 8
RULE_VERSION = 2                      # 1: level against each fly's own warm-up baseline; 2: against WALK_REF_HZ (see the docstring)
WALK_REF_HZ = 4.32                    # GAME RULE: the fixed calm DNp09 reference, exploration seeds 0-15 (tools/race_diagnosis.py)


def walk_level(brain) -> float:
    """DNp09's firing (Hz per neuron, the brain's own fast rate) as a multiple of the fixed reference WALK_REF_HZ (speed rule version 2)."""
    return float(brain.hz("walk")) / WALK_REF_HZ


def speed_of(level: float, threshold: float) -> float:
    return V_MAX * float(np.clip(level / threshold, 0.0, 1.0))


def run_lane(brain, lures=LURES, cap_s: float = TIME_CAP_S, length: float = TRACK_LENGTH, cancel=None, progress=None) -> dict:
    """One fly down the track. Returns its finishing time (None if it did not finish by the cap), the distance, a trace of
    (time, position) samples and which lures it touched."""
    from kickthefly.game import kick_the_fly as k

    dt = TICK_STEPS * float(brain.dt)
    x, t = 0.0, 0.0
    trace, touched, max_level = [[0.0, 0.0]], set(), 0.0
    finish = None
    ticks = int(round(cap_s / dt))
    for tick in range(ticks):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        for i, (lx, kind) in enumerate(lures):
            d = lx - x
            if abs(d) <= CONTACT:
                brain.poke("taste", None, TASTE_POKE)
                brain.poke("reward", None, REWARD_POKE)
                touched.add(i)
            elif 0 < d <= SMELL_RANGE:                # only a lure ahead is smelled, stronger the closer it is
                brain.poke("scent", kind, SCENT_POKE * (1.0 - d / SMELL_RANGE) + 0.05)
        for _ in range(TICK_STEPS):
            brain._step()
        lvl = walk_level(brain)
        max_level = max(max_level, lvl)
        x += speed_of(lvl, k.THRESH["walk"]) * dt
        t += dt
        if tick % TRACE_EVERY == 0:
            trace.append([round(t, 2), round(min(x, length), 3)])
        if progress and tick % 100 == 0:
            progress(tick / ticks)
        if x >= length:
            finish = round(t, 3)
            trace.append([round(t, 2), length])
            break
    return dict(finish_s=finish, distance=round(min(x, length), 3), trace=trace, lures_touched=sorted(touched),
                max_walk_level=round(max_level, 3), seconds=round(t, 3), rule_version=RULE_VERSION)


def order(results: dict[int, dict]) -> list[int]:
    """Finishing order of {fly: lane result}: finished flies by time, then those that did not finish by how far they got."""
    done = sorted((s for s, r in results.items() if r["finish_s"] is not None), key=lambda s: results[s]["finish_s"])
    rest = sorted((s for s, r in results.items() if r["finish_s"] is None), key=lambda s: -results[s]["distance"])
    return done + rest
