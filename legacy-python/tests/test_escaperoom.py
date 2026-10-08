"""Tests for Escape-Room arena (Play mode).

Verifies multi-hazard emergent navigation, speedrun timer, and
tamper-evident verification code generation/validation.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from kickthefly.game import kick_the_fly as k2
from kickthefly.game.speedrun import make_speedrun_code, verify_speedrun_code


def test_speedrun_code_generation_and_verification():
    seed = 42
    run_time = 14.58
    code = make_speedrun_code(seed, run_time)
    assert code.startswith("KTF-42-01458-")

    res = verify_speedrun_code(code)
    assert res["valid"] is True
    assert res["seed"] == 42
    assert abs(res["time_seconds"] - 14.58) < 1e-4
    assert res["formatted_time"] == "14.58s"


def test_speedrun_code_tampering_detection():
    seed = 100
    run_time = 8.20
    code = make_speedrun_code(seed, run_time)

    # Tampered seed
    tampered_seed = code.replace("100", "101", 1)
    res = verify_speedrun_code(tampered_seed)
    assert res["valid"] is False

    # Tampered time
    tampered_time = code.replace("00820", "00500")
    res = verify_speedrun_code(tampered_time)
    assert res["valid"] is False

    # Corrupt string
    res = verify_speedrun_code("INVALID-CODE")
    assert res["valid"] is False


def test_escaperoom_arena_registered():
    assert "escaperoom" in k2.ARENAS
    assert hasattr(k2, "SUGAR_GOAL")


def test_escaperoom_flypaper_sticking():
    fly = k2.Fly(400.0)
    fly.arena = "escaperoom"
    # Flypaper in escaperoom is between x = 280.0 and 560.0
    fly.p[:, 0] = 350.0
    fly.p[:, 1] = k2.FLOOR - 1.0
    contacts = fly.step(1.0, (0, 0))
    assert len(fly.stuck) > 0


def test_escaperoom_speedrun_completion(tmp_path, monkeypatch):
    class MockBrain:
        def __init__(self):
            self.dead = False
            self.pokes = []
        def poke(self, name, side=None, strength=1.0, recruit=0.0):
            self.pokes.append((name, side, strength))

    class MockSound:
        def __init__(self):
            self.played = []
        def play(self, name):
            self.played.append(name)

    game = SimpleNamespace()
    game.arena_i = k2.ARENAS.index("escaperoom")
    game.frame = 6
    game.streaks = []
    game.cfg = {"brain.seed": 42}
    game.escaperoom_start_t = 10.0
    game.escaperoom_seed = 42
    game.escaperoom_completed = False
    game.escaperoom_finish_t = 0.0
    game.escaperoom_code = ""
    game.sound = MockSound()
    game.notes = []
    game.note = lambda msg, **kw: game.notes.append(msg)
    game.damage = lambda *a: None

    fly = k2.Fly(k2.SUGAR_GOAL[0])
    fly.p[:, 0] = k2.SUGAR_GOAL[0]
    fly.p[:, 1] = k2.SUGAR_GOAL[1]
    fly.dead_at = None
    slot = SimpleNamespace(fly=fly, brain=MockBrain())

    now = 18.5
    # Call _environment_one on game
    k2.Game._environment_one(game, slot, "escaperoom", now, (0, 0))

    assert game.escaperoom_completed is True
    assert game.escaperoom_finish_t == now
    assert game.escaperoom_code.startswith("KTF-42-")
    res = verify_speedrun_code(game.escaperoom_code)
    assert res["valid"] is True
    assert abs(res["time_seconds"] - 8.5) < 0.05


def test_escaperoom_reset():
    game = SimpleNamespace()
    game.cfg = {"brain.seed": 123}
    fly = k2.Fly(500.0)
    fly.stuck[0] = np.array([300.0, 500.0])
    slot = SimpleNamespace(fly=fly)
    game.flies = [slot]

    k2.Game.reset_escaperoom(game, now=5.0, seed=77)
    assert game.escaperoom_start_t == 5.0
    assert game.escaperoom_seed == 77
    assert game.escaperoom_completed is False
    assert len(fly.stuck) == 0
    # Fly should be repositioned near x = 130
    assert abs(fly.p[k2.HEAD, 0] - 130.0) < 1e-4


def test_escaperoom_timer_starts_at_the_game_clock():
    """Starting in the escape room (config or --arena) resets it without a time; the timer used to start at 0 on a
    clock that doesn't, and showed the machine's uptime (e.g. 7841 s) as the run time."""
    game = SimpleNamespace(cfg={"brain.seed": 1}, clock=SimpleNamespace(now=7841.0))
    game.flies = [SimpleNamespace(fly=k2.Fly(500.0))]
    k2.Game.reset_escaperoom(game)
    assert game.escaperoom_start_t == 7841.0


def test_game_started_in_the_escaperoom_times_from_now():
    """The same bug through the real constructor: a game started in the escape room (saved arena or --arena) showed
    thousands of seconds on its speedrun timer, because __init__ set the start to 0 after new_fly() had run."""
    import pygame
    pygame.init()
    from kickthefly.core.config import Config
    from kickthefly.core.simcore import pack
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim
    g, W, soma = pack()
    br = k2.Brain(g, LIFSim(None, LIFParams(backend="cpu"), W_in=W, seed=1), seed=1)
    cfg = Config()
    cfg.set("brain.arena", "escaperoom")
    game = k2.Game(None, br, k2.BrainView(soma, W, np.zeros(g.n, bool)), graph=g, weights=W, cfg=cfg)
    assert k2.ARENAS[game.arena_i] == "escaperoom"
    assert 0 <= game.clock.now - game.escaperoom_start_t < 5.0
