"""Tests for deterministic replay files (.ktfreplay).

Verifies bit-exact spike reproduction on NumPy/CPU backends and compatibility
validation against mismatched brain packs or versions.
"""
from pathlib import Path
import numpy as np
import pytest

from kickthefly.core import replay, simcore
from kickthefly.core.replay import ReplayError, ReplayRecorder, ReplayPlayer


def test_replay_record_and_round_trip(tmp_path):
    sig = {"n_neurons": 166700, "synapses": 10272125, "memory_signature": [1.0, 2.0]}
    rec = ReplayRecorder(
        seed=1001,
        backend="cpu",
        dtype="float32",
        signature=sig,
        arena="room",
        settings={"brain.immortal": True},
    )
    rec.record(0, "tool_select", tool="cva")
    rec.record(5, "poke", sense="scent", target="cva", val=0.8)
    rec.record(10, "sim_speed", speed=0.5)
    rec.record(20, "end")

    path = tmp_path / "test.ktfreplay"
    rec.save(path)
    assert path.exists()

    player = ReplayPlayer.load(path, expected_sig=sig)
    assert player.seed == 1001
    assert player.backend == "cpu"
    assert player.arena == "room"
    assert len(player.events_at(0)) == 1
    assert player.events_at(0)[0]["tool"] == "cva"
    assert len(player.events_at(5)) == 1
    assert player.events_at(5)[0]["val"] == 0.8
    assert player.is_finished(20)


def test_replay_mismatch_rejection(tmp_path):
    sig = {"n_neurons": 100, "synapses": 500, "memory_signature": [1.0]}
    rec = ReplayRecorder(seed=1, backend="cpu", dtype="float32", signature=sig, arena="room")
    path = tmp_path / "mismatch.ktfreplay"
    rec.save(path)

    # Different neuron count
    with pytest.raises(ReplayError, match="different brain pack"):
        ReplayPlayer.load(path, expected_sig={"n_neurons": 105, "synapses": 500, "memory_signature": [1.0]})

    # Different synapses
    with pytest.raises(ReplayError, match="different brain pack"):
        ReplayPlayer.load(path, expected_sig={"n_neurons": 100, "synapses": 505, "memory_signature": [1.0]})


def test_replay_spike_reproducibility(tmp_path):
    """NumPy/CPU backend must reproduce spike-for-spike deterministically."""
    from kickthefly.core import paths

    seed = 1005
    br1 = simcore.new_brain(seed=seed)
    sig = replay.pack_signature(br1)

    rec = ReplayRecorder(seed=seed, backend="cpu", dtype="float32", signature=sig, arena="room")

    # Scripted events
    rec.record(5, "poke", sense="head", target=None, val=0.7)
    rec.record(15, "poke", sense="sweet", target=None, val=0.9)
    rec.record(30, "poke", sense="wind", target=None, val=0.5)

    spikes_run1 = []
    for step in range(40):
        for ev in rec.events:
            if ev["step"] == step:
                if ev["type"] == "poke":
                    br1.poke(ev["sense"], ev["target"], ev["val"])
        br1._step()
        spikes_run1.append(br1.sim.spikes.copy())

    path = tmp_path / "session.ktfreplay"
    rec.save(path)

    # Replay session with fresh brain
    player = ReplayPlayer.load(path, expected_sig=sig)
    br2 = simcore.new_brain(seed=player.seed)

    spikes_run2 = []
    for step in range(40):
        for ev in player.events_at(step):
            if ev["type"] == "poke":
                br2.poke(ev["sense"], ev["target"], ev["val"])
        br2._step()
        spikes_run2.append(br2.sim.spikes.copy())

    # Bit-exact check on all spikes across all 40 steps
    for step, (s1, s2) in enumerate(zip(spikes_run1, spikes_run2)):
        assert np.array_equal(s1, s2), f"Spikes diverged at step {step}!"
