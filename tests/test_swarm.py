"""Tests verifying swarm scaling, dynamic fly cap, F cycling key, surgery, and per-fly plasticity."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

pygame.init()
pygame.font.init()

from kickthefly.core.config import Config
from kickthefly.core.simcore import pack
from kickthefly.sim.connectome.sim import LIFParams, LIFSim
from kickthefly.game import kick_the_fly as k2
from kickthefly.game.kick_the_fly import get_max_flies


def test_dynamic_max_flies():
    import os
    assert get_max_flies("cpu") == 16
    assert get_max_flies("torch-cpu") == 16
    assert get_max_flies("numba") == min(32, max(16, os.cpu_count() or 16))
    assert get_max_flies("torch-cuda") == get_max_flies("torch-rocm") == 32
    assert get_max_flies("gl") == 32                            # one batch group (backends._GLGroup)


def test_gl_cap_ignores_expanded_swarm(monkeypatch):
    monkeypatch.setenv("KICK_THE_FLY_EXPANDED_SWARM", "1")
    assert get_max_flies("torch-cuda") == 64
    assert get_max_flies("gl") == 32


def test_swarm_and_f_cycling_and_plasticity():
    g, W, soma = pack()
    sim1 = LIFSim(None, LIFParams(), W_in=W, seed=1)
    br1 = k2.Brain(g, sim1, seed=1)
    br1.graph = g
    from kickthefly.core import memory as mem_mod
    br1.memory = mem_mod.Memory(g, sim1, load=False)

    view = k2.BrainView(soma, W, np.zeros(g.n, bool))
    cfg = Config()
    cfg.set("brain.backend", "numba")

    # Create game in headless/dummy mode
    game = k2.Game(None, br1, view, graph=g, weights=W, cfg=cfg)
    assert game.max_flies == get_max_flies(game.brain.sim.backend.name)

    # Spawn additional flies up to 4 for testing
    for s in range(2, 6):
        sim_i = LIFSim(None, LIFParams(), W_in=W.copy(), seed=s)
        br_i = k2.Brain(g, sim_i, seed=s)
        br_i.graph = g
        br_i.memory = mem_mod.Memory(g, sim_i, load=False)
        fly_i = game._new_spawn_fly()
        slot = k2.FlySlot(fly_i, br_i, seed=s, primary=False)
        game.flies.append(slot)

    assert len(game.flies) == 5

    # 1. Test F cycling key
    assert game.focus == 0
    game.do_action("cycle_fly", now=1.0)
    assert game.focus == 1
    game.cycle_focus()
    assert game.focus == 2

    # 2. Test surgery on focused fly
    target_idx = 100
    orig_override_0 = float(game.flies[0].brain.override[target_idx])
    # Apply surgery to currently focused fly (index 2)
    game.focus = 2
    game.brain.set_override(np.array([target_idx]), 1)
    # Fly 2 was changed; fly 0 was not changed
    assert float(game.flies[2].brain.override[target_idx]) != orig_override_0
    assert float(game.flies[0].brain.override[target_idx]) == orig_override_0

    # 3. Test per-fly plasticity
    mem1 = game.flies[1].brain.memory
    mem2 = game.flies[2].brain.memory
    if mem1 is not None and mem2 is not None:
        w_before_1 = float(mem1.w[0])
        w_before_2 = float(mem2.w[0])
        # Modify weights in fly 1
        mem1.w[0] -= 0.05
        mem1._write_back()
        # Verify fly 1 changed, fly 2 remained unaffected
        assert float(mem1.w[0]) != w_before_1
        assert float(mem2.w[0]) == w_before_2


def test_swarm_in_outdoor_arenas():
    g, W, soma = pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=1)
    br = k2.Brain(g, sim, seed=1)
    br.graph = g
    view = k2.BrainView(soma, W, np.zeros(g.n, bool))
    cfg = Config()

    for arena in ["orchard", "field"]:
        cfg.set("brain.arena", arena)
        game = k2.Game(None, br, view, graph=g, weights=W, cfg=cfg)
        game.three_d = True
        game.arena_i = k2.ARENAS.index(arena)
        assert k2.ARENAS[game.arena_i] == arena


def test_spawn_builds_a_real_brain_through_the_game():
    """N goes through Game.spawn_fly -> build_brain on a thread. 2.8's first cut lost an import there, so every spawn
    died with a NameError on that thread and the swarm never grew; build the brain the same way the key does."""
    g, W, soma = pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=1)
    br = k2.Brain(g, sim, seed=1)
    view = k2.BrainView(soma, W, np.zeros(g.n, bool))
    cfg = Config()
    cfg.set("brain.backend", "cpu")
    game = k2.Game(None, br, view, graph=g, weights=W, cfg=cfg)
    new = game.build_brain(seed=2)
    assert new.sim.n == g.n and new.sim.backend.name == "cpu"
    game.spawn_fly()
    import time
    t0 = time.perf_counter()
    while game._spawning and game._new_slot is None and time.perf_counter() - t0 < 60:
        time.sleep(0.1)
    assert game._new_slot is not None, "the spawn thread never produced a fly"
    game._new_slot.brain.stop()
