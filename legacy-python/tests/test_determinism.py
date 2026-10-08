"""Seeded, lockstep runs replay exactly, and save states restore a brain bit-for-bit."""
import hashlib

import numpy as np
import pytest

from conftest import needs_pack

pytestmark = needs_pack


def schedule(br, t):
    """A fixed stimulus schedule: touches, a smell, looming and a shock."""
    if t % 50 == 10:
        br.poke("head", None, 0.8)
    if 100 <= t < 300 and t % 20 == 0:
        br.poke("scent", "swatter", 0.5)
        br.poke("punish", None, 1.0)
    if t % 70 == 35:
        br.poke("loom", None, 0.6, recruit=0.4)


def run(br, n, start=0):
    h = hashlib.sha256()
    for t in range(start, start + n):
        schedule(br, t)
        br._step()
        h.update(np.packbits(br.sim.spikes).tobytes())
    return h.hexdigest()


def test_same_seed_same_inputs_same_spikes():
    from kickthefly.core import simcore
    a = simcore.new_brain(seed=7, warmup=200)
    b = simcore.new_brain(seed=7, warmup=200)
    assert run(a, 400) == run(b, 400)
    assert np.array_equal(a.sim.v, b.sim.v) and np.array_equal(a.memory.w, b.memory.w)
    c = simcore.new_brain(seed=8, warmup=200)
    assert run(c, 400) != run(simcore.new_brain(seed=7, warmup=200), 400)


def test_brain_state_round_trip_is_exact():
    from kickthefly.core import savestate
    from kickthefly.core import simcore
    a = simcore.new_brain(seed=3, warmup=200)
    run(a, 300)
    a.poke("legs", "L", 1.0)                                  # a stimulus still being delivered at save time
    arrays = {}
    meta = savestate.brain_state(a, "x_", arrays)
    after_a = run(a, 300, start=300)
    b = simcore.new_brain(seed=99, warmup=50)                 # a different brain, then restored
    savestate.restore_brain(b, meta, arrays, "x_")
    after_b = run(b, 300, start=300)
    assert after_a == after_b
    assert np.array_equal(a.memory.w, b.memory.w)
    assert np.array_equal(a.sim.W_csr.data, b.sim.W_csr.data)


@pytest.fixture
def game2d():
    import pygame
    from types import SimpleNamespace

    from kickthefly.game import kick_the_fly as k
    from kickthefly.core import simcore

    pygame.init()
    screen = pygame.display.set_mode((k.W, k.H))
    br = simcore.new_brain(seed=0, warmup=200)
    view = SimpleNamespace(calm=np.zeros(br.n, np.float32), set_palette=lambda name: None, firing=0, hot_firing=0,
                           legend=((255, 0, 0), (0, 0, 255)), sparkle=True,
                           render=lambda *a, **kw: pygame.Surface((10, 10)))
    games = []

    def make(seed=0):
        g = k.Game(screen, simcore.new_brain(seed=seed, warmup=200) if games else br, view, *simcore.pack()[:2:2], cfg=None)
        g.graph, g.weights = simcore.pack()[0], simcore.pack()[1]
        g.view_stop = True
        games.append(g)
        return g

    yield make
    for g in games:
        g.view_stop = True


def game_run(g, ticks=240):
    """Lockstep: 10 brain steps per 3 ticks (5 ms steps, 60 Hz ticks), scripted clicks with tools."""
    now = 1000.0
    h = hashlib.sha256()
    for t in range(ticks):
        now += 1 / 60
        g.clock.now = now
        if t in (30, 90):
            g.tool = 2 if t == 30 else 1
            g.use_tool((float(g.fly.p[1, 0]), float(g.fly.p[1, 1])), now)
        g.update(now, (400, 300))
        for _ in range(4 if t % 3 == 0 else 3):
            g.brain._step()
        h.update(np.round(g.fly.p, 6).tobytes())
        h.update(np.packbits(g.brain.sim.spikes).tobytes())
    return h.hexdigest()


def test_game_lockstep_replays(game2d):
    a = game2d(0)
    ha = game_run(a)
    b = game2d(0)
    b.new_fly()
    hb = game_run(b)
    assert ha == hb


def test_save_file_round_trip_and_rejections(game2d, tmp_path):
    import json
    import zipfile

    from kickthefly.core import savestate
    g = game2d(0)
    game_run(g, 60)
    g.arena_i = 3
    g.set_lab_param("thresh.escape", 5.5)
    path = savestate.save_game(g, tmp_path / "s.ktfsave")
    meta = savestate.read_meta(path)
    assert meta["format_version"] == savestate.FORMAT_VERSION and meta["mode"] == "2d"
    import io
    with zipfile.ZipFile(path) as zf:                        # every array header says little-endian: portable
        with zipfile.ZipFile(io.BytesIO(zf.read("arrays.npz"))) as inner:
            for name in inner.namelist():
                with inner.open(name) as fh:
                    version = np.lib.format.read_magic(fh)
                    read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                            else np.lib.format.read_array_header_2_0)
                    shape, fortran, dtype = read(fh)
                    assert dtype.str[0] in "<|", (name, dtype.str)
    v_saved, p_saved = g.brain.sim.v.copy(), g.fly.p.copy()
    game_run(g, 30)
    g.arena_i = 0
    g.set_lab_param("thresh.escape", 4.0)
    savestate.load_game(g, path)
    assert np.array_equal(g.brain.sim.v, v_saved) and np.array_equal(g.fly.p, p_saved)
    assert g.arena_i == 3 and g.lab_params["thresh.escape"] == 5.5

    def rewrite(**changes):
        bad = tmp_path / f"bad{len(changes)}{list(changes)[0]}.ktfsave"
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(bad, "w") as dst:
            m = json.loads(src.read("meta.json"))
            m.update(changes)
            dst.writestr("meta.json", json.dumps(m))
            dst.writestr("arrays.npz", src.read("arrays.npz"))
        return bad

    for changes, word in (({"format_version": 99}, "newer"), ({"mode": "3d"}, "3D"),
                          ({"signature": {"n_neurons": 5, "synapses": 1}}, "brain pack")):
        with pytest.raises(savestate.SaveError, match=word):
            savestate.load_game(g, rewrite(**changes))
    junk = tmp_path / "junk.ktfsave"
    junk.write_bytes(b"not a zip")
    with pytest.raises(savestate.SaveError):
        savestate.read_meta(junk)
    g.arena_i = 1
    assert g.load_state(junk, wait=True) is False and g.arena_i == 1     # a bad file changes nothing
