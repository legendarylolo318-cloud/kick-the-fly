"""The brains in their own process (core/brainproc.py, 3.1.0 review): the same brain, the same spikes, and the game's view of it works."""
import numpy as np
import pytest

from kickthefly.core import brainproc as bp


@pytest.fixture(scope="module")
def proc(synthetic_pack_module):
    p = bp.BrainProcess(auto_policy="exact")
    yield p
    p.close()


@pytest.fixture(scope="module")
def synthetic_pack_module(tmp_path_factory):
    """The real pack when it is here (the identity check means most with it), else the test suite's synthetic one."""
    from kickthefly.sim import brainpack
    if brainpack.find() is None:
        pytest.skip("needs a brain pack")
    yield


@pytest.fixture(scope="module")
def template():
    from kickthefly.core import simcore
    g, W, _ = simcore.pack()
    return bp.make_template(g, W), g, W


def _local(g, W, seed, backend="cpu"):
    from kickthefly.core import memory
    from kickthefly.game import kick_the_fly as k2
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim
    p = LIFParams()
    p.backend = backend
    sim = LIFSim(None, p, W_in=W, seed=seed)
    br = k2.Brain(g, sim, seed=seed)
    br.graph = g
    if getattr(g, "dan_mbon", None) is not None:
        br.memory = memory.Memory(g, sim)
        br.memory.save = lambda: None
    br.warmup(600)
    return br


@pytest.mark.parametrize("seed", [1000, 1001])
def test_the_same_seed_gives_the_same_spikes_in_either_process(proc, template, seed):
    tmpl, g, W = template
    rb = proc.new_brain(tmpl, dict(seed=seed, backend="cpu", warmup=600))
    try:
        remote = rb.run("kickthefly.core.brainproc", "_spike_digest")
        local = bp._spike_digest(_local(g, W, seed))
        assert remote == local, "a brain in the brain process stepped differently from the same brain in this process"
    finally:
        rb.stop()


def test_the_game_reads_and_drives_a_remote_brain(proc, template):
    import time
    tmpl, g, W = template
    rb = proc.new_brain(tmpl, dict(seed=7, backend="cpu", warmup=100))
    try:
        rb.start()
        t_end = time.monotonic() + 10
        while rb.steps < 400 and time.monotonic() < t_end:            # its own thread, at real time
            time.sleep(0.05)
        assert rb.steps >= 400 and rb.hist_n > 0 and np.all(np.isfinite(rb.fast)) and not rb.dead
        assert rb.level("walk") >= 0 and rb.history(0, rb.hist_n).shape[1] == len(rb.names)
        rows = np.flatnonzero(rb.types == rb.types[0])[:3]
        from kickthefly.game import kick_the_fly as k2
        rb.set_override(rows, -1)                                      # surgery reaches the brain process
        assert rb.run("kickthefly.core.brainproc", "_override_at", rows.tolist()) == [float(np.float32(k2.SURGERY_CURRENT[-1]))] * len(rows)
        assert rb.surgery
        rb.override[rows] = 0.0                                        # an in-place write (the laser does that) too
        time.sleep(0.2)
        assert rb.run("kickthefly.core.brainproc", "_override_at", rows.tolist()) == [0.0] * len(rows)
        rb.sedation = 0.5                                              # a forwarded setting, kept here as well
        assert rb.sedation == 0.5 and rb.run("kickthefly.core.brainproc", "_get", "sedation") == 0.5
        rb.kill()
        t_end = time.monotonic() + 5
        while not rb.dead and time.monotonic() < t_end:
            time.sleep(0.02)
        assert rb.dead
        assert len(rb.sim.activity.rates()) == rb.n                     # the brain view's data, published while asked for
        assert rb.memory is None or isinstance(rb.memory.memory_of("sugar"), tuple)
        assert getattr(rb, "_no_such_thing", "default") == "default"   # a missing attribute is an AttributeError, asked once
    finally:
        rb.stop()
    assert rb._stop and getattr(rb, "anything", None) is None             # a stopped brain answers here, never from the process


def test_a_save_state_round_trip_through_the_brain_process(proc, template):
    tmpl, g, W = template
    rb = proc.new_brain(tmpl, dict(seed=3, backend="cpu", warmup=200))
    try:
        meta, arrays = rb.run("kickthefly.core.savestate", "capture_locked", "f0_b_")
        assert arrays and meta
        before = rb.run("kickthefly.core.brainproc", "_spike_digest", 50)
        rb.run("kickthefly.core.savestate", "restore_locked", meta, arrays, "f0_b_")
        rb.refresh_mirrors()
        again = rb.run("kickthefly.core.brainproc", "_spike_digest", 50)
        assert before["sha256"] == again["sha256"], "restoring the save state did not put the brain back where it was"
    finally:
        rb.stop()
