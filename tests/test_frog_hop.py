"""4.0 task 0, the frog's hop (GAME RULE): locomotion is a series of ballistic leaps, never a slide. Plain engine tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kickthefly.game import predators as pr

BOUNDS = pr.Bounds(-6.0, 6.0, 0.0, 3.0, -6.0, 6.0)
GROUND = {"idle", "track", "crouch", "leap", "land", "recover", "aim", "strike", "retract", "regroup", "eat"}
NEXT = {                                  # every transition the frog may make
    "idle": {"track", "aim", "crouch"}, "track": {"crouch", "aim"}, "crouch": {"leap"}, "leap": {"airborne"},
    "airborne": {"land"}, "land": {"recover"}, "recover": {"idle"},
    "aim": {"strike", "idle"}, "strike": {"retract", "eat"}, "retract": {"regroup", "crouch"}, "regroup": {"idle"},
    "eat": {"crouch"},
}


def _frog(fly_xz=(4.0, 0.0), start=(0.0, 0.0), seed=1):
    rng = np.random.default_rng(seed)
    p = pr.Predator("frog", np.array([start[0], 0.0, start[1]]), rng, BOUNDS)
    t = pr.Target("f", np.array([fly_xz[0], pr.THORAX_Y, fly_xz[1]]), flying=False)
    return p, t


def _trace(p, t, seconds=20.0, fly_fn=None):
    for i in range(int(seconds / pr.DT)):
        if fly_fn:
            fly_fn(i, t)
        prev_state, prev_p = p.state, p.p.copy()
        p.step(pr.DT, [t])
        yield (i, prev_state, p.state, prev_p, p.p.copy())
        if p.done:
            break


def test_gravity_is_the_games_gravity():
    from kickthefly.game import kick3d
    assert pr.GRAVITY == pytest.approx(kick3d.GRAV * 3600.0)       # per frame^2 at the 60 Hz tick


def test_the_arc_lands_where_it_was_aimed_and_peaks_at_the_fixed_apex():
    p, t = _frog(fly_xz=(5.0, 0.0))
    apex, landed, aim = 0.0, None, None
    for r in _trace(p, t, 6.0):
        if r[1] == "crouch" and r[2] == "leap":
            aim = p._land_at.copy()
        apex = max(apex, r[4][1])
        if r[1] == "airborne" and r[2] == "land":
            landed = r[4]
            break
    assert aim is not None and landed is not None
    assert np.linalg.norm(landed - aim) < 0.02
    assert landed[1] == 0.0
    assert apex == pytest.approx(pr.LEAP_APEX, rel=0.03)
    expect = min(5.0 - pr.STANDOFF, pr.LEAP_MAX)
    assert aim[0] == pytest.approx(expect, abs=0.02)


def test_the_integrated_arc_is_ballistic():
    p, t = _frog(fly_xz=(3.0, 0.0))
    for r in _trace(p, t, 6.0):
        if r[2] == "airborne":
            ta = p.t
            assert p.p[1] == pytest.approx((p._v0[1] * ta - 0.5 * pr.GRAVITY * ta * ta), abs=1e-9)
    assert pr.LEAP_T == pytest.approx(2 * math.sqrt(2 * pr.LEAP_APEX / pr.GRAVITY))


def test_a_fly_that_moves_during_the_flight_is_not_followed():
    p, t = _frog(fly_xz=(5.0, 0.0))
    seen = {}

    def dodge(i, tgt):
        if p.state == "airborne":
            tgt.p = np.array([5.0, pr.THORAX_Y, 3.0])
    for r in _trace(p, t, 6.0, dodge):
        if r[1] == "crouch" and r[2] == "leap":
            seen["aim"] = p._land_at.copy()
        if r[1] == "airborne" and r[2] == "land":
            seen["end"] = r[4]
            break
    assert np.linalg.norm(seen["end"] - seen["aim"]) < 0.02


def test_every_transition_is_valid_and_the_hop_states_all_occur():
    p, t = _frog(fly_xz=(5.0, 1.0))
    states = {"idle"}
    for r in _trace(p, t, 40.0):
        if r[1] != r[2]:
            assert r[2] in NEXT[r[1]], f"{r[1]} -> {r[2]}"
            states.add(r[2])
    assert {"track", "crouch", "leap", "airborne", "land", "recover", "aim", "strike"} <= states


def test_nothing_slides_the_frog_moves_only_while_airborne_and_on_the_arc():
    for fly in ((5.0, 1.0), (-4.0, -3.0)):
        p, t = _frog(fly_xz=fly)
        for r in _trace(p, t, 40.0):
            moved = float(np.linalg.norm(r[4] - r[3]))
            if r[1] != "airborne":
                assert moved < 1e-12, f"moved {moved:.4f} m in {r[1]}"
            else:
                assert moved <= pr.LEAP_MAX / pr.LEAP_T * pr.DT * 1.5 + 0.1


def test_leaving_is_hops_too():
    p, t = _frog(fly_xz=(0.6, 0.0))
    t.alive = False
    rows = list(_trace(p, t, 30.0))
    assert p.done and p.leaving
    air = sum(1 for r in rows if r[2] == "airborne" and r[1] != "airborne")
    assert air >= 2
    assert all(float(np.linalg.norm(r[4] - r[3])) < 1e-12 for r in rows if r[1] != "airborne")


def test_the_pose_is_bounded_and_squashes_before_it_stretches():
    p, t = _frog(fly_xz=(5.0, 0.0))
    squash_t = stretch_t = None
    for r in _trace(p, t, 6.0):
        ps = p.pose()
        assert 0 <= ps["squash"] <= 1 and 0 <= ps["stretch"] <= 1 and 0 <= ps["hind"] <= 1 and 0 <= ps["tuck"] <= 1
        assert 0 <= ps["reach"] <= 1 and -math.pi / 2 <= ps["pitch"] <= math.pi / 2
        if ps["squash"] > 0.9 and squash_t is None:
            squash_t = r[0]
        if ps["stretch"] > 0.9 and stretch_t is None:
            stretch_t = r[0]
    assert squash_t is not None and stretch_t is not None and squash_t < stretch_t


def test_the_leaping_frog_is_seen_only_as_a_growing_body():
    p, t = _frog(fly_xz=(5.0, 0.0))
    for r in _trace(p, t, 3.0):
        keys = {k for k, _, _ in p.threats()}
        assert keys <= {("frog", "body"), ("frog", "tip")}          # no other channel to the fly: the looming pipeline only


def test_the_tongue_overshoots_but_stays_near_reach():
    p, t = _frog(fly_xz=(1.0, 0.0), start=(0.0, 0.0))
    far = 0.0
    for r in _trace(p, t, 6.0):
        if p.tip is not None and p.state == "strike":
            far = max(far, float(np.linalg.norm(p.tip - p.mouth)))
    assert 0 < far <= p.spec.reach * (1 + pr.TONGUE_OVERSHOOT) + 0.06


def test_the_same_seed_hops_the_same_way():
    a = [tuple(r[4]) for r in _trace(*_frog(), seconds=10.0)]
    b = [tuple(r[4]) for r in _trace(*_frog(), seconds=10.0)]
    assert a == b


@pytest.mark.parametrize("three_d", [False, True])
def test_the_frog_draws_in_every_state_and_a_leaping_frog_has_a_smaller_shadow(synthetic_pack, three_d):
    import pygame

    import test_predators as tp
    from kickthefly.game import kick_the_fly as k2, predator_play

    class Rec:
        def __init__(self):
            self.rows = []

        def add(self, kind, m, col, *rest):
            self.rows.append((kind, np.asarray(m), col))

        def particle(self, *a, **k):
            pass

    g = tp._game(three_d)
    slot = g.flies[0]
    at = np.array([float(slot.fly.p[k2.THX][0]) + 0.6, 0.0, float(slot.fly.p[k2.THX][2])]) if three_d else \
        np.array([float(slot.fly.p[k2.THX][0]) + 140.0, float(k2.FLOOR)])
    assert g.preds.spawn("frog", at)
    frog = g.preds.list[0]
    seen, shadow = set(), {}
    for i in range(60 * 14):
        g.preds.step(i * pr.DT)
        if not g.preds.list:
            break
        if frog.state not in seen or i % 15 == 0:
            seen.add(frog.state)
            if three_d:
                rd = Rec()
                predator_play.draw3d(g.preds, rd, i * pr.DT)
                sh = [m for k, m, c in rd.rows if len(c) == 4 and c[3] <= 0.25 and c[:3] == (0.0, 0.0, 0.0)]
                assert sh
                shadow[frog.state] = min(shadow.get(frog.state, 1.0), sh[-1][0, 0])
            else:
                predator_play.draw2d(g.preds, pygame.Surface((k2.W, k2.H)), i * pr.DT)
    assert {"crouch", "airborne", "land"} <= seen
    if three_d:
        assert shadow["airborne"] < shadow["idle"], "the shadow shrinks with height"
