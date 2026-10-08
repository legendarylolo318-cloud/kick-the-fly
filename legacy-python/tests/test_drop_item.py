"""Tests for drop_item identity removal and regression coverage for multi-item removal."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.game import kick_the_fly as k2
from kickthefly.game.kick_the_fly import drop_item


def test_drop_item_removes_by_identity_not_equality():
    """list.remove() compares dicts with ==, which compares numpy arrays element-wise and
    raises ValueError: 'The truth value of an array with more than one element is ambiguous'
    as soon as it has to compare against an earlier element. drop_item removes by object identity."""
    d1 = {"p": np.array([1.0, 2.0, 3.0]), "v": np.array([0.0, 0.0, 0.0]), "left": 1.0, "landed": False}
    d2 = {"p": np.array([4.0, 5.0, 6.0]), "v": np.array([0.0, 0.0, 0.0]), "left": 0.5, "landed": False}
    d3 = {"p": np.array([7.0, 8.0, 9.0]), "v": np.array([0.0, 0.0, 0.0]), "left": 0.2, "landed": False}

    items = [d1, d2, d3]

    # Verify root cause: list.remove on the second item raises ValueError
    with pytest.raises(ValueError, match="The truth value of an array with more than one element is ambiguous"):
        items.remove(d2)

    # drop_item removes the second item by identity cleanly
    drop_item(items, d2)
    assert len(items) == 2
    assert items[0] is d1
    assert items[1] is d3

    # Removing the first item works as well
    drop_item(items, d1)
    assert len(items) == 1
    assert items[0] is d3

    # Removing the last item
    drop_item(items, d3)
    assert len(items) == 0

    # Removing an item not in the list is a no-op
    drop_item(items, d1)
    assert len(items) == 0


def test_drop_item_handles_identical_array_values_distinct_objects():
    """Two distinct dicts holding identical numpy array values are distinguished by identity."""
    d1 = {"p": np.array([1.0, 2.0]), "left": 1.0}
    d2 = {"p": np.array([1.0, 2.0]), "left": 1.0}
    items = [d1, d2]

    drop_item(items, d2)
    assert len(items) == 1
    assert items[0] is d1


@needs_pack
def test_3d_sugar_eating_two_piles_second_eaten_first():
    """In the 3D game, when two sugar piles exist and the fly finishes the second pile,
    removing it must not crash on numpy array equality comparison."""
    import pygame
    from kickthefly.core import config
    from kickthefly.game import kick3d

    pygame.init()
    state = {"seed": 3}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")

    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game = kick3d.Game3D(hud, state["brain"], state["view"], state["graph"], state["weights"], cfg=config.Config(None))

    # Pile 1 is farther away and full
    s1 = dict(p=np.array([2.0, 0.035, 2.0]), v=np.array([0.0, 0.0, 0.0]), left=1.0, landed=True)

    # Pile 2 is right at the fly's head and almost finished
    fly = game.flies[0].fly
    head_pos = fly.p[k2.HEAD].copy()
    head_pos[1] = 0.035
    s2 = dict(p=head_pos, v=np.array([0.0, 0.0, 0.0]), left=1 / 240, landed=True)

    game.sugars3 = [s1, s2]

    # Prepare fly state on the floor so it eats
    fly.p[k2.THX, 1] = 0.035
    fly.eating_until = 0.0
    fly.escape_until = 0.0
    fly.stun_until = 0.0

    # Before the fix, _sugar3d would crash at self.sugars3.remove(s2) with ValueError
    game._sugar3d(now=1.0)

    # The second pile is consumed and removed; pile 1 remains
    assert len(game.sugars3) == 1
    assert game.sugars3[0] is s1


@needs_pack
def test_2d_sugar_eating_two_piles_second_eaten_first():
    """In the 2D game, when two sugar piles exist and the fly finishes the second pile,
    removing it must not crash on numpy array equality comparison."""
    import pygame
    from kickthefly.core import config

    pygame.init()
    state = {"seed": 3}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")

    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game = k2.Game(hud, state["brain"], state["view"], state["graph"], state["weights"], cfg=config.Config(None))

    fly = game.flies[0].fly
    s1 = dict(p=np.array([fly.p[k2.THX, 0] + 200.0, float(k2.FLOOR - 8)]), v=0.0, left=1.0)
    s2 = dict(p=np.array([fly.p[k2.HEAD, 0], float(k2.FLOOR - 8)]), v=0.0, left=1 / 240)

    game.sugars = [s1, s2]
    fly.eating_until = 0.0
    fly.escape_until = 0.0
    fly.stun_until = 0.0

    game._sugar(now=1.0)

    assert len(game.sugars) == 1
    assert game.sugars[0] is s1

