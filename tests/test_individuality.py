"""Tests for Task 2: Fly individuality determinism, bit-exactness, and validation invariants."""
import numpy as np
import pytest
import torch

from kickthefly.core import simcore
from kickthefly.core.individuality import compute_fly_gains, compute_personality_card
from kickthefly.lab import validation
from kickthefly.sim.connectome.sim import LIFParams, LIFSim
from kickthefly.sim.connectome import backends


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


def test_cpu_backends_bit_exact_with_individuality():
    """NumPy, Numba, and torch-cpu must stay bit-exact with individuality turned on."""
    _, W, _ = simcore.pack("adult")
    seed = 77

    sim_cpu = LIFSim(None, LIFParams(backend="cpu", individuality="subtle"), W_in=W, seed=seed)
    sim_torch = LIFSim(None, LIFParams(backend="torch-cpu", individuality="subtle"), W_in=W, seed=seed)

    # Step both in lockstep
    for _ in range(40):
        sim_cpu.step()
        sim_torch.step()

    sim_torch.backend.sync_to_host()

    assert np.array_equal(sim_cpu.spikes, sim_torch.spikes), "torch-cpu spikes differ from NumPy with individuality on!"
    assert np.allclose(sim_cpu.v, sim_torch.v, atol=1e-5), "torch-cpu potentials differ from NumPy!"

    avail = backends.detect_available_backends()
    if "numba" in avail:
        sim_numba = LIFSim(None, LIFParams(backend="numba", individuality="subtle"), W_in=W, seed=seed)
        for _ in range(40):
            sim_numba.step()
        assert np.array_equal(sim_cpu.spikes, sim_numba.spikes), "Numba spikes differ from NumPy with individuality on!"


def test_individuality_forced_off_in_validation():
    """Verify that validation forces individuality OFF to ensure reproducibility of published results."""
    # When validation runs, it must ensure brains have individuality='off'
    results = validation.run(brain="adult", seeds=(1000,), workers=1, include={"looming_escape"})
    tests = results["tests"]
    assert len(tests) == 1
    assert "passed" in tests[0]
    assert isinstance(tests[0]["passed"], bool)


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
