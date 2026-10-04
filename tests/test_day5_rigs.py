"""3.0 day 5, classic behavior rigs: the physics (game/rigs.py) on scripted brains, the measures, and real-code smoke tests on the SYNTHETIC
pack (plumbing, not biology)."""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from kickthefly.game import flyrace, rigs


class FakeBrain:
    """DNa01/02 R and L rates, DNp09's rate and the pokes the engine makes. `turn` maps (brain, slip) to R - L Hz."""

    def __init__(self, walk_level=1.5, turn_hz=0.0, follow=0.0):
        self.walk_level, self.turn_hz, self.follow = walk_level, turn_hz, follow
        self.dt, self.pokes, self.slip = 0.005, [], 0.0

    def poke(self, region, side, strength, recruit=None):
        self.pokes.append((region, side, round(strength, 3), None if recruit is None else round(recruit, 3)))

    def _step(self):
        pass

    def hz(self, name):
        if name == "walk":
            return self.walk_level * flyrace.WALK_REF_HZ
        d = self.turn_hz + self.follow * self.slip
        return {"turn_r": 4.0 + max(d, 0), "turn_l": 4.0 + max(-d, 0)}[name]

    def level(self, name):
        return self.walk_level


class FakeTransducer:
    def __init__(self):
        self.slips, self.released = [], 0

    def drive(self, br, slip):
        self.slips.append(slip)
        br.slip = slip

    def release(self, br):
        self.released += 1


# --- the steering rule --------------------------------------------------------------------------------------------------------------
def test_steerer_has_the_duels_dead_zone_gain_and_cap():
    for hz, expect in ((0.0, 0.0), (1.0, 0.0), (-1.4, 0.0), (4.0, (4.0 - 1.5) * rigs.YAW_GAIN), (-4.0, -(4.0 - 1.5) * rigs.YAW_GAIN), (500.0, rigs.YAW_MAX)):
        b, st = FakeBrain(turn_hz=hz), rigs.Steerer()
        y = 0.0
        for _ in range(60):                                   # the smoothing needs a few ticks to settle
            y = st.update(b)
        assert y == pytest.approx(expect, abs=1e-6), hz
    assert rigs.YAW_GAIN == pytest.approx(0.24) and rigs.YAW_MAX == pytest.approx(2.1) and rigs.YAW_DEADZONE_HZ == 1.5


def test_walking_speed_is_the_races_rule():
    from kickthefly.game import kick_the_fly as k

    assert rigs.walk_speed(FakeBrain(1.5)) == pytest.approx(flyrace.speed_of(1.5, k.THRESH["walk"]))
    assert rigs.walk_speed(FakeBrain(0.0)) == 0.0 and rigs.walk_speed(FakeBrain(9.0)) == flyrace.V_MAX


# --- 1. tethered ----------------------------------------------------------------------------------------------------------------------
def test_open_loop_panorama_follows_the_schedule_and_the_fly_does_not_move():
    tr = FakeTransducer()
    r = rigs.run_tethered(FakeBrain(), [(1.0, 0.0), (2.0, 3.0), (1.0, -3.0)], "open", transducer=tr)
    a = r["_full"]
    assert r["ticks"] == 200 and a[:, rigs.COLS.index("x")].max() == 0 and a[:, rigs.COLS.index("y")].max() == 0
    assert set(np.unique(a[:, rigs.COLS.index("slip")])) == {0.0, 3.0, -3.0} and tr.released == 1
    assert list(np.unique(a[:, rigs.COLS.index("aux")])) == [0, 1, 2]


def test_closed_loop_subtracts_the_flys_own_yaw_so_it_can_null_the_rotation():
    # a fly whose steering follows the slip: closed loop leaves less slip than open loop, in the direction of the stimulus
    open_ = rigs.run_tethered(FakeBrain(follow=3.0), [(6.0, 1.0)], "open", transducer=FakeTransducer())["_full"]
    closed = rigs.run_tethered(FakeBrain(follow=3.0), [(6.0, 1.0)], "closed", transducer=FakeTransducer())["_full"]
    c = rigs.COLS
    assert open_[-1, c.index("slip")] == 1.0
    assert 0 < closed[-1, c.index("slip")] < 1.0 and closed[-1, c.index("yaw_rate")] > 0
    assert closed[-1, c.index("slip")] == pytest.approx(1.0 - closed[-1, c.index("yaw_rate")], abs=1e-9)      # gain 1: slip = omega - yaw rate
    half = rigs.run_tethered(FakeBrain(follow=3.0), [(6.0, 1.0)], "closed", gain=0.0, transducer=FakeTransducer())["_full"]
    assert half[-1, c.index("slip")] == 1.0                                                                        # gain 0 is open loop


def test_tethered_heading_integrates_yaw_and_wraps():
    r = rigs.run_tethered(FakeBrain(turn_hz=40.0), [(10.0, 0.0)], "open", transducer=FakeTransducer())["_full"]
    psi = r[:, rigs.COLS.index("psi")]
    assert np.all(np.abs(psi) <= math.pi + 1e-9) and psi.min() < 0 < psi.max()                              # 2.1 rad/s for 10 s wrapped past pi


def test_tethered_rejects_unknown_mode_and_cancel_stops_it():
    with pytest.raises(ValueError):
        rigs.run_tethered(FakeBrain(), [(1, 0)], "sideways", transducer=FakeTransducer())
    ev = threading.Event()
    ev.set()
    tr = FakeTransducer()
    with pytest.raises(RuntimeError, match="cancelled"):
        rigs.run_tethered(FakeBrain(), [(5, 0)], "open", cancel=ev, transducer=tr)
    assert tr.released == 1                                                                                   # the drive is let go even on cancel


# --- 2. ball --------------------------------------------------------------------------------------------------------------------------
def test_ball_integrates_forward_speed_and_turning_in_closed_loop_only():
    b = FakeBrain(walk_level=3.0, turn_hz=0.0)
    r = rigs.run_ball(b, "panorama", "closed", seconds=5.0, transducer=FakeTransducer())["_full"]
    assert r[-1, rigs.COLS.index("y")] == pytest.approx(flyrace.V_MAX * 5.0, rel=0.02) and abs(r[-1, rigs.COLS.index("x")]) < 1e-6
    o = rigs.run_ball(FakeBrain(walk_level=3.0), "panorama", "open", seconds=2.0, transducer=FakeTransducer())["_full"]
    assert o[-1, rigs.COLS.index("y")] == 0.0                                                                # open loop: the VR does not move


def test_ball_panorama_slip_is_drift_minus_turning_when_closed_and_the_drift_alone_when_open():
    tr = FakeTransducer()
    rigs.run_ball(FakeBrain(follow=2.0), "panorama", "closed", seconds=4.0, omega_ext=1.0, transducer=tr)
    assert tr.slips[0] == 1.0 and tr.slips[-1] < 1.0
    tr2 = FakeTransducer()
    rigs.run_ball(FakeBrain(follow=2.0), "panorama", "open", seconds=4.0, omega_ext=1.0, transducer=tr2)
    assert set(tr2.slips) == {1.0}


def test_ball_bar_is_tracked_on_its_side_and_hidden_bar_is_not():
    b = FakeBrain()
    rigs.run_ball(b, "bar", "closed", seconds=0.5, bar_offset=math.radians(90), transducer=FakeTransducer())
    assert b.pokes and {p[1] for p in b.pokes if p[0] == "track"} == {"R"}
    b2 = FakeBrain()
    rigs.run_ball(b2, "bar", "closed", seconds=0.5, bar_offset=-math.radians(90), transducer=FakeTransducer())
    assert {p[1] for p in b2.pokes if p[0] == "track"} == {"L"}
    b3 = FakeBrain()
    r = rigs.run_ball(b3, "bar", "closed", seconds=0.5, bar_offset=math.radians(90), bar_visible=False, transducer=FakeTransducer())
    assert not [p for p in b3.pokes if p[0] == "track"] and r["bar_visible"] is False
    assert r["_full"][0, rigs.COLS.index("aux")] == pytest.approx(math.radians(90), abs=0.01)                # the bearing is still recorded


def test_a_bar_dead_ahead_or_out_of_view_is_not_tracked():
    b = FakeBrain()
    assert rigs.track_object(b, 0.0, 3.0) == 0 and rigs.track_object(b, math.radians(170), 3.0) == 0 and rigs.track_object(b, 1.0, 99.0) == 0
    assert not b.pokes
    assert rigs.track_object(b, math.radians(60), 3.0) > 0 and b.pokes[0][:2] == ("track", "R")


def test_ball_rejects_unknown_scene():
    with pytest.raises(ValueError):
        rigs.run_ball(FakeBrain(), "ceiling", "closed", transducer=FakeTransducer())


# --- 3. Buridan -----------------------------------------------------------------------------------------------------------------------
def test_buridan_fly_stays_on_the_platform_and_is_reflected_at_its_edge():
    r = rigs.run_buridan(FakeBrain(walk_level=3.0), seconds=40.0, stripes=False, start_heading=0.3)["_full"]
    rad = np.hypot(r[:, rigs.COLS.index("x")], r[:, rigs.COLS.index("y")])
    assert rad.max() <= rigs.PLATFORM_RADIUS and r[-1, rigs.COLS.index("aux")] >= 3                           # it reached the edge several times


def test_reflection_is_specular_and_ignores_a_fly_already_heading_in():
    pos, psi, hit = rigs._reflect_circle(np.array([0.6, 0.0]), math.radians(90), 0.5)          # at the east edge heading east
    assert hit and abs(math.degrees(rigs.wrap(psi - math.radians(270)))) < 1e-6 and np.hypot(*pos) < 0.5
    pos, psi, hit = rigs._reflect_circle(np.array([0.6, 0.0]), math.radians(270), 0.5)
    assert not hit and psi == pytest.approx(math.radians(270))
    assert rigs._reflect_circle(np.array([0.1, 0.1]), 1.0, 0.5)[2] is False


def test_buridan_tracks_only_the_stripe_nearer_the_heading_and_not_at_all_without_stripes():
    b = FakeBrain()
    rigs.run_buridan(b, seconds=0.4, stripes=True, start_heading=math.radians(60))                 # east stripe is 30 degrees to the right
    assert {p[1] for p in b.pokes if p[0] == "track"} == {"R"}
    b2 = FakeBrain()
    rigs.run_buridan(b2, seconds=0.4, stripes=True, start_heading=math.radians(120))               # west stripe is the nearer one, to the left? 150 deg away: east is 30 left
    assert {p[1] for p in b2.pokes if p[0] == "track"} == {"L"}
    b3 = FakeBrain()
    rigs.run_buridan(b3, seconds=0.4, stripes=False, start_heading=math.radians(60))
    assert not [p for p in b3.pokes if p[0] == "track"]


def test_stripe_deviation_and_transits_on_known_paths():
    c = rigs.COLS
    n = 600

    def path(xs, ys):
        a = np.zeros((n, len(c)))
        a[:, 0] = np.linspace(11, 60, n)
        a[:, c.index("x")], a[:, c.index("y")] = xs, ys
        return a

    ew = path(0.48 * np.sin(np.linspace(0, 6 * math.pi, n)), np.zeros(n))                   # along the stripe axis
    assert rigs.stripe_deviation(ew) < 1.0 and rigs.transits(ew) == 5
    ns = path(np.zeros(n), 0.48 * np.sin(np.linspace(0, 6 * math.pi, n)))                   # across it
    assert rigs.stripe_deviation(ns) == pytest.approx(90.0, abs=1.0) and rigs.transits(ns) == 0
    still = path(np.zeros(n), np.zeros(n))
    assert math.isnan(rigs.stripe_deviation(still))
    # a fly that touches one edge and turns back has made no transit
    back = path(0.45 * np.abs(np.sin(np.linspace(0, 6 * math.pi, n))), np.zeros(n))
    assert rigs.transits(back) == 0


# --- 4. four-field --------------------------------------------------------------------------------------------------------------------
def test_quadrants_are_clockwise_from_the_top_right():
    assert [rigs.quadrant(*p) for p in ((.1, .1), (.1, -.1), (-.1, -.1), (-.1, .1))] == [0, 1, 2, 3]


def test_odor_is_delivered_only_in_the_odor_quadrants_and_only_when_asked():
    b = FakeBrain(walk_level=3.0)
    r = rigs.run_fourfield(b, seconds=30.0, deliver=True)["_full"]
    q = r[:, rigs.COLS.index("aux")].astype(int)
    scents = [p for p in b.pokes if p[0] == "scent"]
    assert scents and {p[1] for p in scents} == {"fruit"} and {p[2] for p in scents} == {rigs.SCENT_POKE}
    assert len(scents) == int(np.isin(q, rigs.ODOR_QUADRANTS).sum())                                      # one pulse per tick spent in an odor quadrant
    b2 = FakeBrain(walk_level=3.0)
    rigs.run_fourfield(b2, seconds=30.0, deliver=False)
    assert not [p for p in b2.pokes if p[0] == "scent"]


def test_the_square_arena_keeps_the_fly_inside():
    r = rigs.run_fourfield(FakeBrain(walk_level=3.0), seconds=60.0, deliver=False, start_heading=1.0)["_full"]
    assert np.abs(r[:, rigs.COLS.index("x")]).max() <= rigs.FIELD_HALF and np.abs(r[:, rigs.COLS.index("y")]).max() <= rigs.FIELD_HALF


def test_preference_index_definition():
    c = rigs.COLS
    a = np.zeros((1000, len(c)))
    a[:, 0] = np.arange(1, 1001) * 0.02
    q = np.where(np.arange(1000) % 4 == 0, 1, 0)                                                             # a quarter air, the rest odor quadrant 0
    a[:, c.index("aux")] = q
    pi = rigs.preference_index(a, 0.02, settle_s=0.0)
    assert pi["pi"] == pytest.approx((750 - 250) / 1000, abs=0.01)
    a[:, c.index("aux")] = 1                                                                                 # always in air: PI -1
    assert rigs.preference_index(a, 0.02, settle_s=0.0)["pi"] == -1.0
    assert rigs.preference_index(a, 0.02, settle_s=1e9)["pi"] == 0.0                                          # nothing after the settle: no preference, no crash


# --- recording ------------------------------------------------------------------------------------------------------------------------
def test_a_recorder_on_the_brain_receives_the_rigs_kinematics():
    class Rec:
        def __init__(self):
            self.rows = []

        def log_kinematics(self, *a):
            self.rows.append(a)

    b = FakeBrain(walk_level=2.0)
    b.recorder = Rec()
    rigs.run_buridan(b, seconds=2.0, stripes=True)
    assert len(b.recorder.rows) == 20 and b.recorder.rows[0][-1] == "buridan" and b.recorder.rows[0][-2] == "stripes"
    b.recorder.rows.clear()
    rigs.run_fourfield(b, seconds=1.0)
    assert b.recorder.rows and b.recorder.rows[0][-1] == "fourfield"


def test_the_trace_is_decimated_and_the_full_array_is_a_private_key():
    r = rigs.run_buridan(FakeBrain(), seconds=2.0, stripes=False)
    assert len(r["trace"]) == 20 and r["cols"] == list(rigs.COLS) and "_full" in r and r["ticks"] == 100 and r["rule_version"] == rigs.RULE_VERSION
