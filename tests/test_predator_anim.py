"""3.1.0 task 1, predator animation (GAME RULE): the IK legs and planted feet, the spider's windup/strike/recover cycle, the mantis and
dragonfly poses. The engine pieces are plain logic and tested directly; the games' drawing runs on the synthetic pack."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kickthefly.game import predator_anim as pa
from kickthefly.game import predators as pr

BOUNDS = pr.Bounds(-6.0, 6.0, 0.0, 3.0, -6.0, 6.0)


def test_two_bone_ik_keeps_segment_lengths_and_reaches_the_target():
    rng = np.random.default_rng(0)
    for _ in range(200):
        root = rng.uniform(-1, 1, 3)
        tgt = root + rng.uniform(-0.25, 0.25, 3)
        knee, foot = pa.two_bone_ik(root, tgt, 0.17, 0.15, (0, 1, 0))
        assert np.linalg.norm(knee - root) == pytest.approx(0.17, abs=1e-6)
        assert np.linalg.norm(foot - knee) == pytest.approx(0.15, abs=1e-6)
        if 0.02 < np.linalg.norm(tgt - root) < 0.31:
            assert np.linalg.norm(foot - tgt) < 1e-6
    knee, foot = pa.two_bone_ik((0, 0, 0), (5, 0, 0), 0.17, 0.15, (0, 1, 0))      # out of reach: stretched toward it, not detached
    assert np.linalg.norm(foot - knee) == pytest.approx(0.15, abs=1e-6) and foot[0] < 0.33 and foot[1] == pytest.approx(0, abs=1e-6)


def test_the_knee_bends_upward():
    knee, foot = pa.two_bone_ik((0, 0.12, 0), (0.15, 0, 0), 0.17, 0.17, (0, 1, 0))
    assert knee[1] > 0.12


def _walk(rig, seconds=6.0, speed=0.3, dt=1 / 60):
    pos = np.array([0.0, 0.12, 0.0])
    fwd = np.array([0.0, 0.0, 1.0])
    prev = None
    for i in range(int(seconds / dt)):
        pos = pos + fwd * speed * dt
        legs = rig.update(pos, fwd, dt, moving=True, ground=0.0)
        yield i, pos.copy(), legs, [t >= 0.0 for t in rig._t], prev
        prev = [f[2].copy() for f in legs]


def test_planted_feet_never_slide_and_the_legs_stay_connected():
    rig = pa.spider_rig(1.0)
    last_t = [False] * rig.n
    last = None
    steps = 0
    for i, pos, legs, stepping, prev in _walk(rig):
        for k, (hip, knee, foot) in enumerate(legs):
            assert np.linalg.norm(knee - hip) == pytest.approx(0.17, abs=1e-6)
            assert np.linalg.norm(foot - knee) == pytest.approx(0.17, abs=1e-6)
            assert np.linalg.norm(foot - hip) <= 0.34 + 1e-9
            if prev is not None and not stepping[k] and not last_t[k] and i > 2:
                assert np.linalg.norm(foot - prev[k]) < 1e-9, f"leg {k} slid on frame {i}"
            assert foot[1] >= -1e-9
        steps += sum(1 for k in range(rig.n) if stepping[k] and not last_t[k])
        groups_up = {rig.groups[k] for k in range(rig.n) if stepping[k]}
        assert len(groups_up) <= 1, "two gait groups stepping at once"
        last_t = stepping
    assert steps >= 20, "a walking spider must actually step"


def test_a_standing_rig_does_not_step():
    rig = pa.spider_rig(1.0)
    pos, fwd = np.array([0.0, 0.12, 0.0]), np.array([0.0, 0.0, 1.0])
    for _ in range(120):
        rig.update(pos, fwd, 1 / 60, moving=False)
    assert rig.stepping() == 0


def test_the_spider_cycle_is_walk_windup_strike_recover_and_bites_at_the_end_of_the_lunge():
    c = pa.SpiderCycle()
    seq, bite_t, t = ["walk"], None, 0.0
    for i in range(240):
        ev = c.step(1 / 60, 0.1, 0.2)
        t += 1 / 60
        if seq[-1] != c.state:
            seq.append(c.state)
        if "bite" in ev:
            bite_t = t if bite_t is None else bite_t
    assert seq[:5] == ["walk", "windup", "strike", "recover", "walk"]
    assert bite_t == pytest.approx(pa.WINDUP_S + pa.STRIKE_S, abs=0.03)
    assert pa.WINDUP_S + pa.STRIKE_S + pa.SPIDER_RECOVER_S == pytest.approx(0.76)


def test_a_fly_that_dodges_the_windup_is_missed_and_the_spider_never_walks_mid_cycle():
    c = pa.SpiderCycle()
    events, walk_in_cycle = [], 0
    for i in range(120):
        dist = 0.1 if c.state in ("walk", "windup") and i < 12 else 0.5
        events += c.step(1 / 60, dist, 0.2)
        walk_in_cycle += c.walking and c.t > 0 and i < 40 and i > 14
    assert "miss" in events and "bite" not in events
    assert not walk_in_cycle


def test_the_pose_runs_zero_to_one_and_back():
    c = pa.SpiderCycle()
    peak = 0.0
    for _ in range(120):
        c.step(1 / 60, 0.1, 0.2)
        ps = c.pose()
        assert all(0.0 <= ps[k] <= 1.0 for k in ps)
        peak = max(peak, ps["raise_"], ps["lunge"])
    assert peak == pytest.approx(1.0, abs=0.02)
    assert pa.SpiderCycle().pose() == dict(crouch=0.0, raise_=0.0, lunge=0.0)


def test_the_mantis_rears_and_cocks_in_its_windup_and_sways_only_while_stalking():
    rng = np.random.default_rng(3)
    fly = np.array([2.0, pr.THORAX_Y, 0.0])
    m = pr.Predator("mantis", np.array([0.5, 0.0, 0.0]), rng, BOUNDS)
    t = pr.Target("f", fly, flying=False)
    seen_aim = seen_sway = False
    for i in range(60 * 90):
        m.step(pr.DT, [t])
        ps = m.pose()
        assert set(ps) == {"sway", "rear", "cock", "walking"}
        assert all(0.0 <= ps[k] <= 1.0 for k in ("rear", "cock")) and -1 <= ps["sway"] <= 1
        if m.state == "aim":
            seen_aim = True
            assert ps["rear"] == pytest.approx(min(1.0, pa.ease(m.t / m.spec.aim_s)))
            assert not ps["walking"]
        if m._freeze_left > 0:
            assert ps["sway"] == 0.0 and not ps["walking"]
        seen_sway |= abs(ps["sway"]) > 0.5
        if m.done:
            break
    assert seen_aim and seen_sway


def test_the_mantis_does_not_move_in_its_windup_or_recovery():
    rng = np.random.default_rng(3)
    m = pr.Predator("mantis", np.array([0.3, 0.0, 0.0]), rng, BOUNDS)
    t = pr.Target("f", np.array([1.0, pr.THORAX_Y, 0.0]), flying=False)
    for _ in range(60 * 20):
        before = m.p.copy()
        m.step(pr.DT, [t])
        if m.state in ("aim", "strike", "retract", "recover", "eat"):
            assert np.allclose(m.p, before)
        if m.done:
            break


def test_the_dragonfly_banks_into_turns_and_throws_its_legs_forward_to_grab():
    rng = np.random.default_rng(5)
    d = pr.Predator("dragonfly", np.array([0.0, 2.6, 0.0]), rng, BOUNDS)
    t = pr.Target("f", np.array([2.0, 1.5, 0.0]), flying=True)
    banks, reach = [], []
    for i in range(60 * 20):
        if d.state == "pursue" and i % 7 == 0:
            t.p = np.array([2.0 + math.sin(i / 25.0) * 2.0, 1.5, math.cos(i / 25.0) * 2.0])      # a dodging fly
        d.step(pr.DT, [t])
        ps = d.pose()
        assert -0.7 <= ps["bank"] <= 0.7 and 0 <= ps["reach"] <= 1 and 0 <= ps["tuck"] <= 1 and -math.pi / 2 <= ps["pitch"] <= math.pi / 2
        banks.append(ps["bank"])
        reach.append(ps["reach"])
        if d.state == "eat":
            assert ps["tuck"] == 1.0
        if d.done:
            break
    assert max(abs(b) for b in banks) > 0.05
    d2 = pr.Predator("dragonfly", np.array([0.0, 2.6, 0.0]), np.random.default_rng(5), BOUNDS)
    t2 = pr.Target("f", np.array([2.0, 1.5, 0.0]), flying=True)
    reach = []
    for _ in range(60 * 10):
        d2.step(pr.DT, [t2])
        reach.append(d2.pose()["reach"])
        if d2.state == "eat":
            break
    assert d2.state == "eat" and max(reach) > 0.6, "the legs are thrown forward as it closes on a fly it will catch"


def test_a_patrolling_dragonfly_has_no_bank_or_reach():
    d = pr.Predator("dragonfly", np.array([0.0, 2.6, 0.0]), np.random.default_rng(0), BOUNDS)
    d.step(pr.DT, [])
    ps = d.pose()
    assert ps["bank"] == 0.0 and ps["reach"] == 0.0


# --- the games -----------------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _free(request):
    yield
    if "synthetic_pack" in request.fixturenames:
        import gc
        import threading

        import test_extras3 as t3
        while t3._GAMES:
            g = t3._GAMES.pop()
            g.view_stop = True
            for slot in getattr(g, "flies", []):
                slot.brain.stop()
        for th in threading.enumerate():
            if th.name == "brain-view":
                th.join(timeout=2.0)
        gc.collect()


def test_the_3d_spider_winds_up_before_it_bites_and_stands_still_while_it_does(synthetic_pack):
    import test_extras3 as t3
    from kickthefly.game import kick3d
    from kickthefly.game import kick_the_fly as k2

    g = t3.make_game(True)
    fly = g.flies[0].fly
    thx = np.asarray(fly.p[k2.THX], float)
    g.spider3 = g.spider = dict(p=thx + np.array([0.9, -0.2, 0.0]), state="hunt", bite_at=0.0, bites=0, anchor=thx.copy())
    sp = g.spider3
    bites_at, first_windup = None, None
    for i in range(60 * 8):
        before = sp["p"].copy()
        n = sp["bites"]
        g._spider3d(i / 60.0)
        cyc = sp["cycle"]
        if cyc.state in ("windup", "strike", "recover"):
            first_windup = i if first_windup is None else first_windup
            assert np.allclose(sp["p"], before), f"the spider moved in {cyc.state}"
        if sp["bites"] > n and bites_at is None:
            bites_at = i
        if bites_at is not None:
            break
    assert first_windup is not None and bites_at is not None
    assert (bites_at - first_windup) / 60.0 == pytest.approx(pa.WINDUP_S + pa.STRIKE_S, abs=0.05)
    assert kick3d.SpiderCycle is pa.SpiderCycle


@pytest.mark.parametrize("three_d", [False, True])
def test_every_predator_draws_in_every_state_and_the_spider_in_each_phase(synthetic_pack, three_d):
    import pygame
    import test_predators as tp
    from kickthefly.game import kick_the_fly as k2, predator_play

    class Rec:
        n = 0

        def add(self, *a, **k):
            self.n += 1

        def particle(self, *a, **k):
            pass

    g = tp._game(three_d)
    slot = g.flies[0]
    thx = np.asarray(slot.fly.p[k2.THX], float)
    for kind in pr.KINDS:
        at = np.array([float(thx[0]) + 0.6, 0.0, float(thx[2])]) if three_d else np.array([float(thx[0]) + 140.0, float(k2.FLOOR)])
        g.preds.spawn(kind, at)
    sp_pos = thx + np.array([0.5, -0.2, 0.0]) if three_d else thx + np.array([60.0, 10.0])
    sp = dict(p=sp_pos.copy(), state="hunt", bite_at=0.0, bites=0, anchor=sp_pos.copy())
    sp["cycle"] = pa.SpiderCycle()
    for ph in ("walk", "windup", "strike", "recover"):
        sp["cycle"].state, sp["cycle"].t = ph, 0.05
        if three_d:
            predator_play.draw_spider3d(sp, Rec(), 1.0)
        else:
            predator_play.draw_spider2d(sp, pygame.Surface((k2.W, k2.H)), 1.0)
    sp["state"] = "drop"
    if three_d:
        predator_play.draw_spider3d(sp, Rec(), 1.0)
    else:
        predator_play.draw_spider2d(sp, pygame.Surface((k2.W, k2.H)), 1.0)
    for i in range(60 * 30):
        g.preds.step(i / 60.0)
        if i % 10 == 0 and g.preds.list:
            if three_d:
                predator_play.draw3d(g.preds, Rec(), i / 60.0)
            else:
                predator_play.draw2d(g.preds, pygame.Surface((k2.W, k2.H)), i / 60.0)
