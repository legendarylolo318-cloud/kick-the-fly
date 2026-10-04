"""3.1.0 task 3, the simulation off the render thread. The game's Brain has run on its own thread at 200 steps/s (times the game speed)
since 1.x; these tests pin that down: it advances while the main thread is busy, slow motion and pause scale it, and the same seed
steps to the same spikes whether a thread or a caller steps it. Synthetic pack (plumbing, not biology)."""
from __future__ import annotations

import time

import numpy as np
import pytest

import test_extras3 as t3


@pytest.fixture(autouse=True)
def _free():
    yield
    import gc
    import threading

    while t3._GAMES:
        g = t3._GAMES.pop()
        g.view_stop = True
        for slot in getattr(g, "flies", []):
            slot.brain.stop()
    for th in threading.enumerate():
        if th.name == "brain-view":
            th.join(timeout=2.0)
    gc.collect()


def _start(g):
    br = g.brain
    br._stop = False
    br.start()
    return br


def test_the_brain_advances_while_the_main_thread_is_busy_at_200_steps_a_second(synthetic_pack):
    g = t3.make_game(False)
    br = _start(g)
    time.sleep(0.2)
    s0, t0 = br.steps, time.perf_counter()
    end = t0 + 1.0
    x = 0
    while time.perf_counter() < end:          # the "render thread" is busy and never yields on purpose
        x += sum(range(2000))
    rate = (br.steps - s0) / (time.perf_counter() - t0)
    br.stop()
    assert 150 <= rate <= 215, f"{rate:.0f} steps/s"


def test_a_busy_render_thread_cannot_starve_the_brain_of_the_gil(synthetic_pack):
    """Regression: at CPython's default 5 ms switch interval a main thread that holds the GIL left the brain 27 of its 200 steps/s."""
    import sys

    old = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.005)
        g = t3.make_game(False)
        _start(g)
        assert sys.getswitchinterval() <= 0.0002 + 1e-9
    finally:
        sys.setswitchinterval(old)


def test_slow_motion_and_pause_scale_the_thread(synthetic_pack):
    g = t3.make_game(False)
    br = _start(g)
    time.sleep(0.2)
    out = {}
    for name, speed in (("half", 0.5), ("paused", 0.0)):
        br.speed = speed
        time.sleep(0.15)
        s0, t0 = br.steps, time.perf_counter()
        time.sleep(1.0)
        out[name] = (br.steps - s0) / (time.perf_counter() - t0)
    br.stop()
    assert 70 <= out["half"] <= 115, out
    assert out["paused"] == 0, out


def test_the_same_seed_steps_to_the_same_spikes_on_a_thread_and_by_hand(synthetic_pack):
    a = t3.make_game(False)
    b = t3.make_game(False)
    for g in (a, b):                                          # one brain per game: same seed, same pack
        g.brain.speed = 1.0
    ra, rb = [], []
    # the thread can only be compared with a hand-stepped brain through the number of steps it took, so count and replay them
    br = _start(a)
    time.sleep(0.5)
    br.stop()
    time.sleep(0.05)
    n = br.steps
    assert n > 20
    start_a = a.brain.sim.spike_total
    assert start_a >= 0
    # hand-step a twin to the same step count, then both must agree on the next steps exactly
    twin = b.brain
    while twin.steps < n:
        with twin.step_lock:
            twin._step()
    for _ in range(30):
        with br.step_lock:
            br._step()
        with twin.step_lock:
            twin._step()
        ra.append(np.asarray(br.sim.spikes).copy())
        rb.append(np.asarray(twin.sim.spikes).copy())
    assert twin.steps == br.steps
    assert all(np.array_equal(x, y) for x, y in zip(ra, rb))
