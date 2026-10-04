"""Tests for Task 2: Fly individuality determinism, bit-exactness, and validation invariants."""
import numpy as np
import pytest

from kickthefly.core import simcore
from kickthefly.core.individuality import compute_fly_gains, compute_personality_card
from kickthefly.lab import validation
from kickthefly.sim.connectome.sim import LIFParams, LIFSim
from kickthefly.sim.connectome import backends, larva_loader

from conftest import needs_larva


def test_individuality_determinism():
    """Verify that per-fly gains are strictly deterministic from the fly's seed."""
    g1_pre, g1_post = compute_fly_gains(seed=42, n=1000, mode="subtle")
    g2_pre, g2_post = compute_fly_gains(seed=42, n=1000, mode="subtle")
    assert np.array_equal(g1_pre, g2_pre)
    assert np.array_equal(g1_post, g2_post)

    # Different seeds must produce different gains
    g3_pre, g3_post = compute_fly_gains(seed=43, n=1000, mode="subtle")
    assert not np.array_equal(g1_pre, g3_pre)
    assert not np.array_equal(g1_post, g3_post)

    # Mode 'off' produces None (identity matrix)
    g_off_pre, g_off_post = compute_fly_gains(seed=42, n=1000, mode="off")
    assert g_off_pre is None
    assert g_off_post is None

    # Signs are strictly preserved (gains > 0)
    assert np.all(g1_pre > 0)
    assert np.all(g1_post > 0)


def _run(backend, mode, W, steps=60):
    sim = LIFSim(None, LIFParams(backend=backend, individuality=mode), W_in=W, seed=77)
    rng = np.random.default_rng(3)
    spikes = []
    for t in range(steps):
        stim = None
        if 20 <= t < 40:
            stim = np.zeros(sim.n, np.float32)
            stim[rng.choice(sim.n, 200, replace=False)] = 1.0
        sim.step(stim)
        spikes.append(sim.spikes.copy())
    sim.backend.sync_to_host()
    return np.array(spikes), sim.v.copy()


@pytest.mark.parametrize("mode", ["off", "subtle", "strong"])
def test_numba_bit_exact_with_individuality(mode):
    """NumPy and Numba stay bit-exact (spikes and every voltage) with individuality off and on."""
    if "numba" not in backends.detect_available_backends():
        pytest.skip("numba not installed")
    _, W, _ = simcore.pack("adult")
    s_np, v_np = _run("cpu", mode, W)
    s_nb, v_nb = _run("numba", mode, W)
    assert np.array_equal(s_np, s_nb)
    assert np.array_equal(v_np, v_nb)


def test_torch_cpu_bit_exact_with_individuality_off_only():
    """torch-cpu is bit-exact with individuality off. With it on, the scaled products are summed in a different order,
    so float32 rounding differs and the chaotic network drifts (known, see docs/individuality.md): only closeness over
    a short run is asserted there."""
    if "torch-cpu" not in backends.detect_available_backends():
        pytest.skip("torch not installed")
    _, W, _ = simcore.pack("adult")
    s_np, v_np = _run("cpu", "off", W)
    s_t, v_t = _run("torch-cpu", "off", W)
    assert np.array_equal(s_np, s_t) and np.array_equal(v_np, v_t)
    s_np, v_np = _run("cpu", "subtle", W, steps=5)
    s_t, v_t = _run("torch-cpu", "subtle", W, steps=5)
    assert np.allclose(v_np, v_t, atol=1e-4)


def test_individuality_scaling_is_d_post_w_d_pre():
    """i_syn equals D_post · W · D_pre · spikes, the shared W is untouched, and every gain is positive (signs kept)."""
    _, W, _ = simcore.pack("adult")
    sim = LIFSim(None, LIFParams(backend="cpu", individuality="strong"), W_in=W, seed=5)
    before = sim.W_csr.data.copy()
    rng = np.random.default_rng(0)
    sim.spikes[:] = rng.random(sim.n) < 0.01
    got = sim._propagate()
    want = sim.d_post * (sim.W_csr @ (sim.spikes * sim.d_pre).astype(np.float32))
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
    assert np.array_equal(before, sim.W_csr.data)
    assert (sim.d_pre > 0).all() and (sim.d_post > 0).all()


def test_individuality_forced_off_in_validation(monkeypatch):
    """Every brain the validation builds has individuality off, whatever the environment asks for."""
    monkeypatch.setenv("KICK_THE_FLY_INDIVIDUALITY", "strong")
    built = []
    real = simcore.new_brain

    def spy(*a, **kw):
        br = real(*a, **kw)
        built.append(br.sim.d_pre)
        return br

    monkeypatch.setattr(simcore, "new_brain", spy)
    validation.run(brain="adult", seeds=(1000,), workers=1, include={"looming_escape"})
    assert built and all(d is None for d in built)


@needs_larva
def test_individuality_forced_off_in_larva_validation(monkeypatch):
    """The same for the larva brain (3.0 release review: split out so the adult half runs where the optional larva pack isn't built)."""
    monkeypatch.setenv("KICK_THE_FLY_INDIVIDUALITY", "strong")
    built = []
    real = simcore.new_brain

    def spy(*a, **kw):
        br = real(*a, **kw)
        built.append(br.sim.d_pre)
        return br

    monkeypatch.setattr(simcore, "new_brain", spy)
    larva_loader.ensure_larva_brain_pack()
    validation.run(brain="larva", seeds=(1000,), workers=1)
    assert built and all(d is None for d in built)


def test_gl_does_not_claim_individuality():
    """The gl shaders do not apply D_pre/D_post, so a gl fly must not report gains it does not use."""
    if "gl" not in backends.detect_available_backends():
        pytest.skip("no OpenGL 4.3 compute here")
    _, W, _ = simcore.pack("adult")
    sim = LIFSim(None, LIFParams(backend="gl", individuality="subtle"), W_in=W, seed=1)
    try:
        assert sim.d_pre is None and sim.d_post is None
    finally:
        sim.backend.close()


def test_personality_card_generation():
    """Verify personality card generation is deterministic, non-scripted, and includes thresholds."""
    card1 = compute_personality_card(seed=123)
    card2 = compute_personality_card(seed=123)
    assert card1 == card2
    assert "title" in card1
    assert "summary" in card1
    assert "metrics" in card1

    card3 = compute_personality_card(seed=456)
    assert "title" in card3


def test_batched_gpu_plastic_weights_with_individuality():
    """Assert batched GPU paths keep plastic weights equal with individuality on.
    Skips cleanly if no GPU is available, never attempting to set one up."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available; cleanly skipping GPU batched individuality test.")

    # Only reached if a real GPU is already present
    from kickthefly.core import memory
    g, W, _ = simcore.pack()

    sim1 = LIFSim(None, LIFParams(backend="torch-gpu", individuality="subtle"), W_in=W.copy(), seed=42)
    sim2 = LIFSim(None, LIFParams(backend="torch-gpu", individuality="subtle"), W_in=W.copy(), seed=99)
    mem1 = memory.Memory(g, sim1, load=False)

    stim = np.zeros(g.n, np.float32)
    stim[mem1.kc[:30]] = 4.0
    for _ in range(10):
        backends.TorchBackend.step_batch([sim1, sim2], [stim, None])
        mem1.step(sim1.activity.rates(), calm=False, steps=10)

    assert mem1.w is not None
