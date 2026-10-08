"""Deterministic replay files (.ktfreplay): the file round trip, refusing another brain pack, and a recorded protocol
run replaying to the same spikes on every bit-exact backend (and a changed input being caught)."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import needs_pack
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


@needs_pack
def test_same_seed_and_pokes_give_the_same_spikes(tmp_path):
    """The determinism replays rest on: two brains with the same seed, poked the same way at the same steps (the pokes
    read back from a saved replay file), fire the same spikes. This doesn't play a replay; the tests below do."""
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


PROTOCOL = """
name: replay_check
seed: 1003
warmup_s: 0.2
duration_s: 1.5
surgery: {dnp01: -1}
stimuli:
  - {at_s: 0.1, for_s: 0.4, target: loom, mode: poke, strength: 0.7}
  - {at_s: 0.3, for_s: 0.5, target: sweet, mode: drive, amp: 0.5}
  - {at_s: 0.9, for_s: 0.3, target: "prefix:JO-", mode: poke, strength: 0.9}
recordings:
  - {name: mn9, neurons: mn9}
"""


def _record(tmp_path, backend="cpu"):
    import os
    from kickthefly.lab import protocol

    os.environ["KICK_THE_FLY_SIM_BACKEND"] = backend
    try:
        spec = tmp_path / "replay_check.yaml"
        spec.write_text(PROTOCOL)
        dest = tmp_path / "check.ktfreplay"
        assert protocol.record_replay(spec, dest, tmp_path / "out") == 0
    finally:
        os.environ.pop("KICK_THE_FLY_SIM_BACKEND", None)
    return dest


def _play(path, backend, out):
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import headless

    return headless.main(k.parse_args(["--headless", "--replay", str(path), "--out", str(out), "--backend", backend]))


@needs_pack
def test_recorded_protocol_replays_to_the_same_spikes_on_bit_exact_backends(tmp_path):
    """A protocol run with pokes (drawing from the brain's RNG), a held drive and surgery, recorded on NumPy, replays
    to the recording's spikes on NumPy, Numba and torch-cpu (whichever are installed)."""
    from kickthefly.sim.connectome import backends

    dest = _record(tmp_path)
    rp = ReplayPlayer.load(dest, expected_sig=None)
    kinds = {ev["type"] for ev in rp.meta["events"]}
    assert {"sense", "poke", "drive"} <= kinds, kinds
    assert rp.settings["surgery"] == {"dnp01": -1} and rp.spike_sha256
    avail = backends.detect_available_backends()
    for name in [b for b in replay.BIT_EXACT_BACKENDS if b in avail]:
        out = tmp_path / f"play-{name}"
        assert _play(dest, name, out) == 0, name
        summary = json.loads((out / "replay_summary.json").read_text())
        assert summary["backend"] == name and summary["identical"], summary
        spikes = np.load(out / "spikes.npz")
        assert len(spikes["neuron"]) > 0


@needs_pack
def test_replay_refuses_another_brain_pack(tmp_path):
    """Same neuron and synapse counts, different pack: the SHA-256 tells them apart and the replay is refused."""
    dest = _record(tmp_path)
    with zipfile.ZipFile(dest) as zf:
        meta = json.loads(zf.read("replay.json"))
    meta["signature"]["pack_sha256"] = "0" * 64
    other = tmp_path / "other.ktfreplay"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("replay.json", json.dumps(meta))
    br = simcore.new_brain(seed=1, warmup=0)
    with pytest.raises(ReplayError, match="different brain pack"):
        ReplayPlayer.load(other, expected_sig=replay.pack_signature(br))
    assert _play(other, "cpu", tmp_path / "refused") == 2


@needs_pack
def test_replay_with_a_changed_input_is_caught(tmp_path):
    """A replay whose inputs don't match its recorded spikes fails on a bit-exact backend (exit 1), so the comparison
    really compares."""
    dest = _record(tmp_path)
    with zipfile.ZipFile(dest) as zf:
        meta = json.loads(zf.read("replay.json"))
    poke = next(ev for ev in meta["events"] if ev["type"] == "poke")
    poke["strength"] = 0.2
    bad = tmp_path / "changed.ktfreplay"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("replay.json", json.dumps(meta))
    assert _play(bad, "cpu", tmp_path / "changed") == 1
