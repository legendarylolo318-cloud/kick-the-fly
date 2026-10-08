"""Existing players' data keeps working: a training memory saved by v2.5.0 loads into this version unchanged."""
import shutil

import numpy as np

from conftest import ROOT, needs_pack

FIXTURE = ROOT / "tests" / "fixtures"


@needs_pack
def test_v250_training_memory_loads(tmp_path, monkeypatch):
    """fly-memory.npz was written by v2.5.0's memory.py (git tag v2.5.0) after fear training on the swatter smell."""
    from kickthefly.core import paths
    from kickthefly.core import simcore

    folder = tmp_path / "Documents" / "Kick the Fly" / "memory"
    folder.mkdir(parents=True)
    shutil.copy(FIXTURE / "fly-memory-v2.5.0.npz", folder / "fly-memory.npz")
    monkeypatch.setenv("KICK_THE_FLY_MEMORY", str(folder))
    paths.reset_cache()
    br = simcore.new_brain(seed=0, warmup=0, isolated_memory=False)
    expected = np.load(FIXTURE / "fly-memory-v2.5.0-expected-w.npy")
    assert np.array_equal(br.memory.w, expected), "the old memory was not loaded (signature or format changed)"
    assert br.memory.weakened_share() > 0
    fear, like = br.memory.memory_of("swatter")
    assert fear > 0.2 and like < 0.01
    kc_mbon = br.sim.W_csr.data[br.memory.csr_pos]            # and it is live in the simulation's synapses
    assert np.array_equal(kc_mbon, expected)


def test_windows_memory_location_is_unchanged(tmp_path):
    """Where v2.2-v2.5 put training memory on Windows is still where this version looks."""
    from kickthefly.core import paths
    user = tmp_path / "user"
    docs = user / "Documents"
    p = paths.resolve(platform="win32", env={}, home=user, known_folder={"Documents": docs}.get, cwd=tmp_path)
    assert p.memory_dir == docs / "Kick the Fly" / "memory"
    assert p.pictures_dir.name == "Kick the Fly"
