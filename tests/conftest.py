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


# --- memory report per test file (3.0 day 4, a follow-up of the day 3 review) -------------------------------------------------------
# Off unless KTF_MEM_REPORT=path.json is set (tools/run_tests.py --mem-report does that). After each test module the process is
# garbage-collected and its resident size recorded: a file whose growth is large and never comes back is a leak, and the running
# total shows whether the suite as a whole is climbing. Costs nothing when off.
_MEM_ROWS: list[dict] = []


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576
    except (OSError, ValueError):
        return 0.0


@pytest.fixture(scope="module", autouse=True)
def _ktf_memory_report(request):
    path = os.environ.get("KTF_MEM_REPORT")
    if not path:
        yield
        return
    import gc

    gc.collect()
    before = _rss_mb()
    yield
    shutdown_live_games()                          # what the file left running is freed first (3.0 day 4 review), then measured
    gc.collect()
    after = _rss_mb()
    _MEM_ROWS.append(dict(file=str(Path(str(request.fspath)).relative_to(ROOT)), before_mb=round(before), after_mb=round(after),
                          growth_mb=round(after - before)))
    try:
        import json

        Path(path).write_text(json.dumps(_MEM_ROWS, indent=1))
    except OSError:
        pass


# defined after the memory report on purpose: a later module fixture is torn down first, so the games are freed before it measures
def shutdown_live_games() -> int:
    """Shut down every Game still alive (its brain-view thread holds it and all its brains until view_stop is set). Returns how many.
    3.0 day 4 review: test files that built whole games and never stopped them kept 1-1.8 GB each after they finished."""
    k2 = sys.modules.get("kickthefly.game.kick_the_fly")
    if k2 is None:
        return 0
    games = list(k2.Game.LIVE)
    for g in games:
        g.shutdown()
    import gc

    gc.collect()
    return len(games)


@pytest.fixture(scope="module", autouse=True)
def _ktf_free_games_after_each_file():
    """After each test file (not each test: some files share one game across their tests through a module fixture, which this
    fixture outlives), every game the file built is shut down so its memory can go back."""
    yield
    shutdown_live_games()


_RULE_GLOBALS = (("kickthefly.game.kick_the_fly", ("LOOM_MIN", "LOOM_FULL")),
                 ("kickthefly.core.pet", ("HUNGER_RATE_PER_SEC", "SLEEP_RATE_PER_SEC", "SLEEP_RECOVERY_RATE", "HUNGER_SUGAR_GAIN",
                                          "HUNGER_PAM_GAIN", "SLEEP_DFB_DRIVE", "MAX_CATCHUP_SECONDS")))


def snapshot_rule_globals() -> list:
    """The module-level game rules Lab parameters write (lab.apply_rules: THRESH, the looming cut-offs, individuality sigmas, pet rates)."""
    snap = []
    k2 = sys.modules.get("kickthefly.game.kick_the_fly")
    if k2 is not None:
        snap.append((k2.THRESH, dict(k2.THRESH)))
    ind = sys.modules.get("kickthefly.core.individuality")
    if ind is not None:
        snap.append((ind.SIGMAS, dict(ind.SIGMAS)))
    for mod_name, names in _RULE_GLOBALS:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            snap.append((mod, {n: getattr(mod, n) for n in names if hasattr(mod, n)}))
    return snap


def restore_rule_globals(snap: list) -> None:
    for target, saved in snap:
        if isinstance(target, dict):
            target.clear()
            target.update(saved)
        else:
            for n, v in saved.items():
                setattr(target, n, v)


@pytest.fixture(autouse=True)
def _ktf_restore_rule_globals():
    """3.0 day 4 review: a test that set a Lab rule parameter (test_determinism sets thresh.escape to 5.5) left the module-global THRESH
    changed for every later test in the process, so results depended on which files shared a chunk."""
    had = "kickthefly.game.kick_the_fly" in sys.modules
    snap = snapshot_rule_globals()
    yield
    restore_rule_globals(snap)
    if not had and "kickthefly.game.kick_the_fly" in sys.modules:   # first imported during this test: nothing was snapshotted
        from kickthefly.lab import lab

        lab.apply_rules(lab.DEFAULTS)
