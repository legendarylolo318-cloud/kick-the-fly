import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("KICK_THE_FLY_OFFLINE", "1")            # never contact neuPrint from the tests (or CI)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test writes config, memory, saves and pictures under a temp folder, never the real user folders."""
    monkeypatch.setenv("KICK_THE_FLY_HOME", str(tmp_path / "ktf-home"))
    monkeypatch.delenv("KICK_THE_FLY_MEMORY", raising=False)
    from kickthefly.core import paths
    paths.reset_cache()
    yield tmp_path / "ktf-home"
    paths.reset_cache()


def brain_pack():
    from kickthefly.sim import brainpack
    return brainpack.find()


needs_pack = pytest.mark.skipif(brain_pack() is None, reason="brain pack data/kick_brain.npz not built")


@pytest.fixture(scope="session")
def _synthetic_pack_file(tmp_path_factory):
    import synthetic_pack

    return synthetic_pack.build(tmp_path_factory.mktemp("synthetic-pack") / synthetic_pack.PACK_NAME)


@pytest.fixture
def synthetic_pack(_synthetic_pack_file, monkeypatch):
    """Point brainpack.find at the tiny SYNTHETIC pack (tests/synthetic_pack.py: random wiring, real type names) for one
    test. For plumbing tests only; it is not the connectome and proves nothing about biology."""
    from kickthefly.core import neurodex, simcore
    from kickthefly.sim import brainpack

    monkeypatch.setattr(brainpack, "find", lambda brain="adult": _synthetic_pack_file if brain == "adult" else None)
    neurodex.reset_cache()
    simcore.pack.cache_clear()                      # simcore.pack is cached per process: never let one pack leak into another test
    yield _synthetic_pack_file
    neurodex.reset_cache()
    simcore.pack.cache_clear()
