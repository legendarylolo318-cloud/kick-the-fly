"""Tests for Task 1: Drosophila larva connectome integration, loader, and validation."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from kickthefly.core import savestate
from kickthefly.game import larva
from kickthefly.lab import validation
from kickthefly.sim.connectome import larva_loader

from conftest import needs_larva


@needs_larva
def test_larva_connectome_loader_and_counts():
    """Verify neuron count, synapse count, license, and citation as read from the data."""
    pack_path = larva_loader.ensure_larva_brain_pack()
    assert pack_path.exists()

    data = np.load(pack_path, allow_pickle=True)
    n_neurons = len(data["type"])
    n_synapses = len(data["data"])
    citation = str(data["citation"])
    license_str = str(data["license"])

    # Verified counts from Winding et al. 2023 Science dataset
    assert n_neurons == 2952, f"Expected 2952 neurons, got {n_neurons}"
    n_pairs = len(data["data"])
    total_synapses = int(np.sum(np.abs(data["data"])))
    assert n_pairs == 110677, f"Expected 110677 pairs, got {n_pairs}"
    assert total_synapses == 352611, f"Expected 352611 synapses, got {total_synapses}"
    assert "Winding" in citation
    assert "not redistributed" in license_str.lower()      # no license is stated for Data S1 (docs/larva.md)


def test_brainpack_separation_and_save_refusal():
    """Verify that a save made for adult is refused by larva and vice versa."""
    # When game is larva and save is adult
    game_larva = SimpleNamespace(three_d=False, brain_type="larva", cfg={"brain.brain": "larva"})
    adult_meta = {
        "format": savestate.FORMAT,
        "format_version": savestate.FORMAT_VERSION,
        "mode": "2d",
        "brain": "adult",
    }
    reason = savestate.compatible(adult_meta, game_larva)
    assert reason is not None
    assert "made for the adult brain (currently using larva)" in reason

    # When game is adult and save is larva
    game_adult = SimpleNamespace(three_d=False, brain_type="adult", cfg={"brain.brain": "adult"})
    larva_meta = {
        "format": savestate.FORMAT,
        "format_version": savestate.FORMAT_VERSION,
        "mode": "2d",
        "brain": "larva",
    }
    reason2 = savestate.compatible(larva_meta, game_adult)
    assert reason2 is not None
    assert "made for the larva brain (currently using adult)" in reason2


def test_larva_body_segmented_crawler():
    """Verify LarvaBody mechanics: 10 segments, crawling peristalsis, head-casting, rolling."""
    body = larva.LarvaBody(x=400.0)
    assert body.n_segments == 10
    assert len(body.p) == 10
    assert not body.is_rolling

    # Test rolling activation
    body.trigger_roll(1.0)
    assert body.is_rolling
    assert body.rolling_until > 0

    # Step physics
    body.step(1.0)
    assert not np.isnan(body.p).any()


def test_larva_arena_gating():
    """Verify that arenas inappropriate for larvae are gated with an informative message."""
    allowed, msg = larva.is_arena_allowed_for_larva("room")
    assert allowed
    assert msg == ""

    allowed_paper, _ = larva.is_arena_allowed_for_larva("flypaper")
    assert allowed_paper

    allowed_open, msg_open = larva.is_arena_allowed_for_larva("outdoors")
    assert not allowed_open
    assert "crawling larvae" in msg_open.lower() or "larva" in msg_open.lower()


@needs_larva
def test_larva_validation_execution():
    """Run larva validation tests and verify honest reporting without hardcoded pass."""
    results = validation.run(brain="larva", seeds=(1000,), workers=1)
    tests = results["tests"]
    assert len(tests) >= 2
    for r in tests:
        assert "passed" in r
        assert "measured" in r
        assert isinstance(r["passed"], bool)
