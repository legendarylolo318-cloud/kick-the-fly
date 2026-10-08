"""Big brain view: neuron search and the path tracer (kickthefly/lab/neurosearch.py)."""
import itertools

import numpy as np
import pygame
import pytest
import scipy.sparse as sp

from conftest import needs_pack
from kickthefly.lab import neurosearch as ns


def test_top_paths_match_brute_force():
    rng = np.random.default_rng(3)
    for trial in range(15):
        n = 22
        W = sp.random(n, n, density=0.18, random_state=trial, format="csr") * rng.choice([-1, 1], (n, n))
        W = sp.csr_matrix(W)
        W.setdiag(0)
        W.eliminate_zeros()
        D = W.toarray()
        want = []
        for hops in (1, 2, 3):
            for mid in itertools.permutations(range(2, n), hops - 1):
                nodes = [0, *mid, 1]
                ws = [D[b, a] for a, b in zip(nodes, nodes[1:])]
                if all(ws):
                    want.append((float(np.prod(np.abs(ws))), nodes, int(np.sign(np.prod(ws)))))
        want.sort(key=lambda x: -x[0])
        got = ns.top_paths(W, 0, 1, k=5)
        assert [round(p["strength"], 12) for p in got] == [round(s, 12) for s, _, _ in want[:5]]
        for p in got:                                 # each path is real, with its weights and sign
            assert all(D[b, a] == pytest.approx(w) for (a, b), w in zip(zip(p["nodes"], p["nodes"][1:]), p["weights"]))
            assert p["sign"] == int(np.sign(np.prod(p["weights"])))
    assert ns.top_paths(W, 4, 4) == []


@pytest.fixture(scope="module")
def brain():
    from kickthefly.core import simcore
    return simcore.new_brain(seed=2, warmup=0)


@needs_pack
def test_search_by_type_instance_and_body_id(brain):
    rows = ns.search(brain, "dnp01")
    assert rows and all(brain.types[i] == "DNp01" for i in rows)
    assert all(brain.types[i] == "pIP10" for i in ns.search(brain, "pIP10"))
    i = rows[0]
    assert ns.search(brain, str(int(brain.body_id[i])))[0] == i
    assert ns.search(brain, brain.instance[i])[0] in rows
    assert ns.search(brain, "") == [] and len(ns.search(brain, "KC", limit=5)) == 5


@needs_pack
def test_real_paths_are_in_the_connectome(brain):
    src, dst = ns.search(brain, "pIP10")[0], ns.search(brain, "ps1 MN")[0]
    paths = ns.top_paths(brain.sim.W_csr, src, dst, k=3, Wc=brain.sim.W_csc)
    assert paths and all(2 <= len(p["nodes"]) - 1 <= 3 for p in paths)
    W = brain.sim.W_csr
    for p in paths:
        for (a, b), w in zip(zip(p["nodes"], p["nodes"][1:]), p["weights"]):
            assert W[b, a] == pytest.approx(w)
    assert [p["strength"] for p in paths] == sorted((p["strength"] for p in paths), reverse=True)


@needs_pack
def test_big_view_search_and_path_tracer_render(brain):
    from kickthefly.core import config, simcore
    from kickthefly.game import kick_the_fly as k2

    pygame.init()
    g, W, soma = simcore.pack()
    view = k2.BrainView(soma, W, np.zeros(g.n, bool))
    game = k2.Game(pygame.Surface((k2.W, k2.H)), brain, view, graph=g, weights=W, cfg=config.Config(None))
    game.big_view = True
    surf = pygame.Surface((k2.W, k2.H))
    game._draw_big_view(surf)
    box = game.search_box_rect()
    assert game.search_click(box.center) and game.nsearch["active"]
    for ch in "pIP10":
        assert game.search_key(pygame.event.Event(pygame.TEXTINPUT, text=ch))
    assert game.nsearch["results"] and brain.types[game.nsearch["results"][0]] == "pIP10"
    assert game.search_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))
    assert game.inspect["type"] == "pIP10" and not game.nsearch["active"]
    game.set_path_end("from", game.inspect["i"])
    game.set_path_end("to", ns.search(brain, "ps1 MN")[0])
    assert game.paths
    game._draw_big_view(surf)                        # draws the paths, the list and the search box without error
    assert game.path_clear_rect is not None
    assert game.search_click(game.path_clear_rect.center) and not game.paths
