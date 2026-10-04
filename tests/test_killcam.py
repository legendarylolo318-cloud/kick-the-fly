"""Kill cam (3.0): the rate buffer, the risers, the playback clock. Pure numpy, no brain needed."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.core import killcam as kc


def test_rate_encoding_is_accurate_where_it_matters():
    hz = np.array([0.0, 0.5, 2.0, 5.0, 10.0, 40.0, 100.0, 255.0, 400.0], np.float32)
    back = kc.decode_hz(kc.encode_rates(hz * kc.DT))
    assert back[0] == 0 and back[-1] == kc.MAX_HZ                      # clipped, not wrapped
    rel = np.abs(back[1:-2] - hz[1:-2]) / hz[1:-2]
    assert rel.max() < 0.12 and abs(back[2] - 2.0) < 0.15 and abs(back[4] - 10.0) < 0.3


def test_buffer_keeps_one_sample_per_sample_steps_and_the_last_window():
    b = kc.Buffer(4)
    r = np.zeros(4)
    stored = [b.push(s, r) for s in range(0, 200)]
    assert sum(stored) == len(range(0, 200, kc.SAMPLE_STEPS))
    assert len(b) == 25 and len(b.q) == b.cap == 150                      # 25 samples so far, room for 150
    for s in range(200, 200 + 160 * kc.SAMPLE_STEPS):
        b.push(s, r)
    assert len(b) == b.cap                                               # a ring: it never grows
    rep = b.freeze()
    assert rep.n_frames == b.cap and np.all(np.diff(rep.steps) >= kc.SAMPLE_STEPS)       # oldest first
    assert abs(rep.seconds - kc.WINDOW_S) < 0.05


def test_a_paused_brain_adds_nothing_and_a_backwards_step_clears():
    b = kc.Buffer(3)
    r = np.zeros(3)
    assert b.push(100, r) and not b.push(100, r) and not b.push(103, r) and b.push(108, r)
    assert len(b) == 2
    assert b.push(5, r) and len(b) == 1                                  # a save was loaded: start again


def test_too_few_samples_give_no_replay():
    b = kc.Buffer(3)
    for s in range(0, 5 * kc.SAMPLE_STEPS, kc.SAMPLE_STEPS):
        b.push(s, np.zeros(3))
    assert b.freeze() is None


def _ramped(n_neurons=50, frames=150, risers=None, calm=2.0):
    """A replay where chosen neurons ramp up over the last second and the rest sit at calm."""
    hz = np.full((frames, n_neurons), calm, np.float32)
    for i, peak in (risers or {}).items():
        hz[-30:, i] = np.linspace(calm, peak, 30)
    q = kc.encode_rates(hz * kc.DT)
    return kc.Replay(q, np.arange(frames) * kc.SAMPLE_STEPS)


def test_top_risers_are_the_biggest_rises_in_order():
    rep = _ramped(risers={7: 90.0, 3: 60.0, 20: 40.0, 11: 8.0})
    got = rep.top_risers()
    assert [r["index"] for r in got][:3] == [7, 3, 20]
    assert got[0]["rise_hz"] > got[1]["rise_hz"] > got[2]["rise_hz"] > 0
    assert 11 not in [r["index"] for r in got]                           # rose under MIN_RISE_HZ over the tail's mean
    assert all(r["rise_hz"] >= kc.MIN_RISE_HZ for r in got)
    assert got[0]["before_hz"] == pytest.approx(2.0, abs=0.3)


def test_risers_are_deterministic_and_ties_break_by_index():
    rep = _ramped(risers={9: 50.0, 4: 50.0, 30: 50.0})
    assert [r["index"] for r in rep.top_risers()] == [4, 9, 30]
    assert rep.top_risers() == rep.top_risers()


def test_no_riser_means_an_empty_list_not_noise():
    assert _ramped().top_risers() == []
    assert len(_ramped(risers={i: 50.0 for i in range(30)}).top_risers(k=12)) == 12


def test_a_neuron_that_was_loud_then_fell_silent_is_not_a_riser():
    hz = np.full((150, 5), 2.0, np.float32)
    hz[:40, 1] = 100.0
    hz[-30:, 1] = 0.0
    rep = kc.Replay(kc.encode_rates(hz * kc.DT), np.arange(150))
    assert rep.top_risers() == []


def test_summary_names_the_types():
    types = np.array(["A"] * 10 + ["B"] * 10 + [""] * 30)
    rep = _ramped(risers={1: 80.0, 2: 70.0, 12: 60.0, 40: 50.0})
    s = rep.summary(types, np.array(["sc"] * 50))
    assert [r["type"] for r in s["risers"]] == ["A", "A", "B", "(no type)"]
    assert s["types"][0] == ("A", 2) and s["risers"][0]["superclass"] == "sc"


def test_traces_and_frames_match_the_source():
    rep = _ramped(risers={5: 100.0})
    tr = rep.traces([5, 6])
    assert tr.shape == (2, 150) and tr[0, -1] > 90 and tr[1, -1] < 3
    assert rep.rates_per_step(149)[5] == pytest.approx(100.0 * kc.DT, rel=0.1)
    assert rep.hz(10**6).shape == (50,)                                  # out-of-range frames clamp


def test_player_runs_in_slow_motion_and_stops_at_the_moment_of_death():
    rep = _ramped()
    p = kc.Player(rep, speed=0.25)
    assert p.frame == 0 and not p.done
    assert p.wall_seconds == pytest.approx(rep.seconds * 4)
    p.advance(1.0)                                                       # 1 s of wall time = 0.25 s of brain time
    assert p.t == pytest.approx(0.25) and p.frame == int(0.25 / (kc.SAMPLE_STEPS * kc.DT))
    for _ in range(200):
        p.advance(0.5)
    assert p.done and not p.skipped and p.frame == rep.n_frames - 1 and p.progress == 1.0
    p.restart(speed=1.0)
    assert p.t == 0 and not p.done and p.speed == 1.0


def test_skip_ends_at_once_and_advancing_after_done_is_harmless():
    p = kc.Player(_ramped())
    p.advance(0.1)
    p.skip()
    t = p.t
    p.advance(5)
    assert p.done and p.skipped and p.t == t
    p.advance(-3)
    assert p.t == t


def test_reduced_flashing_default_is_slower():
    assert kc.SPEED_REDUCED < kc.SPEED <= 0.5


def test_replay_time_comes_from_step_numbers_not_frame_counts():
    """Frames land a few steps late when a frame boundary falls in between: 10 steps apart here, not 8."""
    frames = 120
    rep = kc.Replay(kc.encode_rates(np.zeros((frames, 4)) + 0.01), np.arange(frames) * 10)
    assert rep.seconds == pytest.approx((frames - 1) * 10 * kc.DT)
    assert rep.frame_at(0.0) == 0 and rep.frame_at(0.049) == 0 and rep.frame_at(0.05) == 1
    assert rep.frame_at(1000.0) == frames - 1
    p = kc.Player(rep, speed=0.25)
    p.advance(4.0)                                                       # 1.0 s of brain time
    assert p.t == pytest.approx(1.0) and p.frame == 20


def test_freeze_keeps_at_most_the_window_even_with_widely_spaced_samples():
    b = kc.Buffer(2)
    for i in range(150):
        b.push(i * 13, np.zeros(2))                                      # 13 steps apart: 150 frames span 9.75 s
    rep = b.freeze()
    assert rep.seconds <= kc.WINDOW_S + 1e-6 and rep.n_frames < 150 and rep.steps[-1] == 149 * 13


def test_risers_windows_follow_time_when_spacing_is_uneven():
    frames, hz = 100, np.full((100, 3), 2.0, np.float32)
    hz[-20:, 1] = 80.0
    steps = np.cumsum(np.r_[0, np.full(99, 12)])
    rep = kc.Replay(kc.encode_rates(hz * kc.DT), steps)
    got = rep.top_risers()
    assert [r["index"] for r in got] == [1] and got[0]["rise_hz"] > 60
