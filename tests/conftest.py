import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Forced, not defaulted (3.0 day 2 review): a shell that exports SDL_VIDEODRIVER=wayland or x11 made the tests open real, empty
# windows on the desktop. No test needs a visible window; KTF_TEST_SDL_VIDEODRIVER / KTF_TEST_SDL_AUDIODRIVER pick another
# driver on purpose (e.g. offscreen).
os.environ["SDL_VIDEODRIVER"] = os.environ.get("KTF_TEST_SDL_VIDEODRIVER", "dummy")
os.environ["SDL_AUDIODRIVER"] = os.environ.get("KTF_TEST_SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("KICK_THE_FLY_OFFLINE", "1")            # never contact neuPrint from the tests (or CI)
os.environ["KTF_NO_NETWORK"] = "1"                            # 3.0 day 3: no network feature (Streamer mode) may connect in a test; forced, not defaulted


def _no_network_connect():
    """3.0 day 3 review: belt and braces under KTF_NO_NETWORK. Any connect() to an internet address in the test process raises,
    so a test (or code under test) that tries to reach Twitch, neuPrint or anywhere else fails loudly instead of connecting.
    Unix sockets (socketpair, the fake chat servers) are unaffected."""
    import socket

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError(f"the test suite never opens a network connection (tried {address!r})")

    def connect(self, address):
        guard(self, address)
        return real_connect(self, address)

    def connect_ex(self, address):
        guard(self, address)
        return real_connect_ex(self, address)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex


_no_network_connect()


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
