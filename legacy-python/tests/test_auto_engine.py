"""3.1.0 task 2, GPU by default: what 'auto' means under each policy, the fallback on any error, software renderers not counting as a
GPU, the engine panel in Settings. No GPU needed: the probes are patched."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.sim.connectome import backends as bk

FULL = {"cpu": "CPU", "numba": "Numba", "gl": "OpenGL 4.6: Real GPU", "torch-cpu": "t", "torch-cuda": "CUDA: x"}


@pytest.fixture(autouse=True)
def _restore():
    old, cache = bk.auto_policy(), bk._gpu_cache
    yield
    bk.set_auto_policy(old)
    bk._gpu_cache = cache


def _gpu(monkeypatch, ok=True, dev="OpenGL 4.6: AMD Radeon"):
    monkeypatch.setattr(bk, "_gl_compute_available", lambda: (ok, dev))
    bk._gpu_cache = None


def test_the_default_policy_is_exact_and_never_picks_gl(monkeypatch):
    _gpu(monkeypatch)
    bk.set_auto_policy("exact")
    assert bk.auto_chain(FULL) == ["torch-cuda", "numba", "cpu"]


def test_the_fastest_policy_tries_gl_then_torch_gpu_then_numba_then_numpy(monkeypatch):
    _gpu(monkeypatch)
    bk.set_auto_policy("fastest")
    assert bk.auto_chain(FULL) == ["gl", "torch-cuda", "numba", "cpu"]
    assert bk.auto_chain({"cpu": "c"}) == ["cpu"]
    assert bk.auto_chain({"cpu": "c", "numba": "n"}) == ["numba", "cpu"]


@pytest.mark.parametrize("dev", ["OpenGL 4.5: llvmpipe (LLVM 17)", "OpenGL 4.3: softpipe", "OpenGL 4.6: Microsoft Basic Render Driver", "OpenGL 4.6: lavapipe"])
def test_a_software_renderer_is_not_a_gpu(monkeypatch, dev):
    _gpu(monkeypatch, True, dev)
    bk.set_auto_policy("fastest")
    assert not bk.gpu_capable()[0]
    assert "gl" not in bk.auto_chain(FULL)


def test_no_compute_support_is_no_gpu(monkeypatch):
    _gpu(monkeypatch, False, "OpenGL 4.1 < 4.3 (Compute shaders not supported)")
    bk.set_auto_policy("fastest")
    assert "gl" not in bk.auto_chain(FULL)


class _Boom:
    def __init__(self, *a, **k):
        raise RuntimeError("driver crash")


class _Ok:
    name = "ok"

    def __init__(self, sim, *a):
        self.sim = sim

    def setup(self):
        pass


def test_a_failing_gl_falls_back_to_the_next_engine(monkeypatch):
    _gpu(monkeypatch)
    monkeypatch.setattr(bk, "_moderngl_available", True)
    monkeypatch.setattr(bk, "GLBackend", _Boom)
    monkeypatch.setattr(bk, "_torch_gpu_kind", lambda: None)
    monkeypatch.setattr(bk, "_numba_available", True)
    monkeypatch.setattr(bk, "NumbaBackend", _Ok)
    bk.set_auto_policy("fastest")
    b = bk.create_backend(object(), "auto")
    assert isinstance(b, _Ok), "gl failed, so Numba"
    monkeypatch.setattr(bk, "NumbaBackend", _Boom)
    monkeypatch.setattr(bk, "CPUBackend", _Ok)
    assert isinstance(bk.create_backend(object(), "auto"), _Ok), "everything failed, so NumPy"


def test_the_exact_policy_does_not_even_try_gl(monkeypatch):
    _gpu(monkeypatch)
    monkeypatch.setattr(bk, "_moderngl_available", True)
    monkeypatch.setattr(bk, "GLBackend", lambda *a: pytest.fail("headless 'auto' must not use gl"))
    monkeypatch.setattr(bk, "_torch_gpu_kind", lambda: None)
    monkeypatch.setattr(bk, "_numba_available", False)
    monkeypatch.setattr(bk, "CPUBackend", _Ok)
    bk.set_auto_policy("exact")
    assert isinstance(bk.create_backend(object(), "auto"), _Ok)


def test_an_explicit_choice_still_wins_over_the_policy(monkeypatch):
    monkeypatch.setattr(bk, "CPUBackend", _Ok)
    bk.set_auto_policy("fastest")
    assert isinstance(bk.create_backend(object(), "cpu"), _Ok)


def test_the_fly_cap_follows_the_same_chain(monkeypatch):
    from kickthefly.game import kick_the_fly as k2

    monkeypatch.setattr(bk, "detect_available_backends", lambda: FULL)
    _gpu(monkeypatch)
    bk.set_auto_policy("fastest")
    assert k2.get_max_flies("auto") == 32 and k2.get_max_flies("gl") == 32
    bk.set_auto_policy("exact")
    monkeypatch.setattr(bk, "detect_available_backends", lambda: {"cpu": "c"})
    assert k2.get_max_flies("auto") == 16


def test_an_unknown_policy_is_exact():
    bk.set_auto_policy("turbo")
    assert bk.auto_policy() == "exact"


def test_the_engine_panel_names_the_engine_the_choice_and_the_chain(monkeypatch):
    from kickthefly.ui.menu import Menu

    monkeypatch.setattr(bk, "detect_available_backends", lambda: FULL)
    _gpu(monkeypatch)
    bk.set_auto_policy("fastest")

    class Be:
        name, device = "cpu", "CPU (NumPy)"

    class Slot:
        brain = type("B", (), {"sim": type("S", (), {"backend": Be()})()})()

    class Host:
        flies = [Slot()]
        cfg = {"brain.backend": "gl"}

    rows = []

    class Fake:
        f_small = None

        def text(self, surf, txt, pos, col, font, anchor):
            rows.append(txt)

    y = Menu._engine_info(Fake(), None, type("R", (), {"x": 0})(), 0, Host())
    text = "\n".join(rows)
    assert "Engine in use: cpu" in text and "You chose gl" in text and "gl > torch-cuda > numba > cpu" in text and y > 0
    Host.flies = []
    rows.clear()
    Menu._engine_info(Fake(), None, type("R", (), {"x": 0})(), 0, Host())
    assert "none yet" in "\n".join(rows)


def test_a_stale_departure_does_not_empty_a_slot_a_new_brain_took():
    """3.1.0: a garbage-collected gl brain posts its departure to the group's thread; the slot may have been reused by then. The
    departure must name its own admission, or it removes the new brain, the group closes and the new brain falls back to the CPU
    (what happened in the 3D game's first seconds on gl)."""
    from kickthefly.core import simcore
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    if "gl" not in bk.detect_available_backends():
        pytest.skip("no OpenGL 4.3 compute here")
    _, W, _ = simcore.pack("adult")
    sim = LIFSim(None, LIFParams(backend="gl", individuality="off"), W_in=W, seed=1)
    try:
        sim.step(None)
        be = sim.backend
        g, slot, serial = be._group, be._slot, be._serial
        assert serial is not None
        g.call(g._remove, slot, serial + 12345)         # somebody else's admission: ignored
        assert not g.closed and g.member_count == 1
        for _ in range(3):
            sim.step(None)
        assert sim.backend.name == "gl", "the brain must still be on the GPU"
    finally:
        sim.backend.close()


def test_a_join_that_arrives_after_the_group_closed_is_refused():
    from kickthefly.core import simcore
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    if "gl" not in bk.detect_available_backends():
        pytest.skip("no OpenGL 4.3 compute here")
    _, W, _ = simcore.pack("adult")
    sim = LIFSim(None, LIFParams(backend="gl", individuality="off"), W_in=W, seed=1)
    other = LIFSim(None, LIFParams(backend="cpu", individuality="off"), W_in=W, seed=2)
    try:
        sim.step(None)
        g = sim.backend._group
        g.close()
        assert g._add(other.backend) is None            # runs before any GL call, so this thread may make it
    finally:
        getattr(sim.backend, "close", lambda: None)()
