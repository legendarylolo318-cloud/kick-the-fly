"""3.1.0 task 4, the brain view on the GPU: the same picture as the CPU view (neuron and region modes, panel and big, a turned camera),
the same picking table, a clean fall back to the CPU, and the setting. Needs OpenGL 4.3 offscreen (software GL is fine); skips without."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from kickthefly.game import kick_the_fly as k2
from kickthefly.game import gpu_brainview as gv


@pytest.fixture
def views(synthetic_pack):
    from kickthefly.core import simcore

    pygame.init()
    g, W, soma = simcore.pack()
    hot = np.random.default_rng(0).random(g.n) < 0.05
    cpu = k2.BrainView(soma, W, hot, regions=getattr(g, "region", None))
    gpu = gv.GPUBrainView(soma, W, hot, regions=getattr(g, "region", None))
    gpu.pref = "gpu"
    try:
        gpu._gpu = gv._Worker()
        gpu._gpu.call(gpu._upload_static)
    except Exception as e:
        pytest.skip(f"no offscreen OpenGL 4.3 here ({e})")
    yield cpu, gpu, g
    if gpu._gpu is not None:
        gpu._gpu.close()


def _inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    rates = (0.025 * (1 + 3 * (rng.random(n) < 0.05) * rng.random(n))).astype(np.float32)
    spiked = np.flatnonzero(rng.random(n) < 0.05)
    return rates, spiked


def _img(s):
    return np.frombuffer(pygame.image.tobytes(s, "RGB"), np.uint8).reshape(s.get_height(), s.get_width(), 3).astype(int)


def _same(a, b):
    d = np.abs(_img(a) - _img(b))
    assert d.mean() < 0.05, f"mean difference {d.mean():.3f}"
    assert (d.max(2) > 3).mean() < 0.002, "more than 0.2% of pixels differ"


@pytest.mark.parametrize("mode", ["neuron", "region"])
@pytest.mark.parametrize("key", ["panel", "big"])
def test_the_gpu_view_draws_the_cpu_picture(views, mode, key):
    cpu, gpu, g = views
    cpu.view_mode = gpu.view_mode = mode
    rates, spiked = _inputs(g.n)
    a = cpu.render(key, rates, spiked, 0.0, False)
    b = gpu.render(key, rates, spiked, 0.0, False)
    assert gpu.on_gpu
    _same(a, b)
    assert gpu.firing == cpu.firing and gpu.hot_firing == cpu.hot_firing
    assert np.allclose(gpu.region_rates, cpu.region_rates)


def test_a_turned_camera_gives_the_same_picture_and_the_same_picking_table(views):
    cpu, gpu, g = views
    rates, spiked = _inputs(g.n, 3)
    for v in (cpu, gpu):
        v.set_camera(35.0, -20.0, 300.0, -200.0, 1.4)
    assert gpu.M["big"] is None, "the GPU view builds no sparse matrix for a turned camera"
    assert np.array_equal(cpu.spark_pix["big"], gpu.spark_pix["big"]), "the neuron inspector must pick the same neurons"
    assert gpu.gain["big"] == pytest.approx(cpu.gain["big"], rel=1e-4)
    _same(cpu.render("big", rates, spiked, 0.0, False), gpu.render("big", rates, spiked, 0.0, False))
    for v in (cpu, gpu):
        v.set_preset("front")
    assert np.array_equal(cpu.spark_pix["big"], gpu.spark_pix["big"])
    _same(cpu.render("big", rates, spiked, 0.0, False), gpu.render("big", rates, spiked, 0.0, False))
    for preset in ("side", "top", "front"):
        for v in (cpu, gpu):
            v.set_preset(preset)
        assert np.array_equal(cpu.spark_pix["big"], gpu.spark_pix["big"]), preset


def test_the_region_structure_is_cached_per_camera(views):
    cpu, gpu, g = views
    cpu.view_mode = gpu.view_mode = "region"
    rates, spiked = _inputs(g.n)
    gpu.render("panel", rates, spiked, 0.0, False)
    n = len(gpu._region_base_cache)
    gpu.render("panel", rates, spiked, 0.0, False)
    assert len(gpu._region_base_cache) == n >= 1


def test_a_palette_change_reaches_the_gpu(views):
    cpu, gpu, g = views
    rates, spiked = _inputs(g.n)
    for v in (cpu, gpu):
        v.set_palette("high-contrast")
    _same(cpu.render("panel", rates, spiked, 0.0, False), gpu.render("panel", rates, spiked, 0.0, False))


def test_sparkles_can_be_turned_off(views):
    cpu, gpu, g = views
    rates, spiked = _inputs(g.n)
    cpu.sparkle = gpu.sparkle = False
    _same(cpu.render("panel", rates, spiked, 0.0, False), gpu.render("panel", rates, spiked, 0.0, False))


def test_a_gpu_failure_falls_back_to_the_cpu_view_with_the_same_picture(views):
    cpu, gpu, g = views
    rates, spiked = _inputs(g.n)
    gpu.set_camera(20.0, 10.0, 0.0, 0.0, 1.2)
    cpu.set_camera(20.0, 10.0, 0.0, 0.0, 1.2)

    def boom(*a, **k):
        raise RuntimeError("driver lost")

    gpu._draw = boom
    out = gpu.render("big", rates, spiked, 0.0, False)       # fails inside, falls back, still returns a picture
    assert not gpu.on_gpu and gpu._gpu_failed and "CPU" in gpu.engine_note
    _same(cpu.render("big", rates, spiked, 0.0, False), out)
    assert gpu.M["big"] is not None, "the CPU path rebuilt its matrix for the current camera"
    for preset in ("side", "front"):
        gpu.set_preset(preset)
        gpu.render("big", rates, spiked, 0.0, False)


def test_the_cpu_preference_never_starts_the_gpu(synthetic_pack, monkeypatch):
    from kickthefly.core import simcore

    g, W, soma = simcore.pack()
    monkeypatch.setattr(gv, "_Worker", lambda: pytest.fail("a CPU view must not make a GL context"))
    v = gv.GPUBrainView(soma, W, np.zeros(g.n, bool))
    v.pref = "cpu"
    rates, spiked = _inputs(g.n)
    v.render("panel", rates, spiked, 0.0, False)
    assert not v.on_gpu and v._gpu is None


def test_no_gl_at_all_is_a_quiet_cpu_view(synthetic_pack, monkeypatch):
    from kickthefly.core import simcore

    g, W, soma = simcore.pack()

    def nope():
        raise RuntimeError("no EGL")

    monkeypatch.setattr(gv, "_Worker", nope)
    v = gv.GPUBrainView(soma, W, np.zeros(g.n, bool))
    v.pref = "gpu"
    rates, spiked = _inputs(g.n)
    img = v.render("panel", rates, spiked, 0.0, False)
    assert img.get_size() == k2.VIEW_SIZES["panel"] and v._gpu_failed and "CPU" in v.engine_note


def test_make_view_honours_the_environment(synthetic_pack, monkeypatch):
    from kickthefly.core import simcore

    g, W, soma = simcore.pack()
    monkeypatch.setenv("KICK_THE_FLY_BRAINVIEW", "cpu")
    assert type(gv.make_view(soma, W, np.zeros(g.n, bool))) is k2.BrainView
    monkeypatch.delenv("KICK_THE_FLY_BRAINVIEW")
    from kickthefly.sim.connectome import backends
    monkeypatch.setattr(backends, "gpu_capable", lambda refresh=False: (True, "a GPU"))
    assert isinstance(gv.make_view(soma, W, np.zeros(g.n, bool)), gv.GPUBrainView)
    monkeypatch.setattr(backends, "gpu_capable", lambda refresh=False: (False, "llvmpipe (a software renderer, not a GPU)"))
    v = gv.make_view(soma, W, np.zeros(g.n, bool))          # no GPU: the CPU view from the start, not a failover in its first frame
    assert type(v) is k2.BrainView and "no GPU" in v.engine_note


def test_the_setting_exists_and_the_game_applies_it():
    from kickthefly.core import config

    s = next(x for x in config.SETTINGS if x.key == "graphics.brain_view")
    assert s.default == "auto" and set(s.options) == {"auto", "gpu", "cpu"}
    assert config.Config(None)["graphics.brain_view"] == "auto"
