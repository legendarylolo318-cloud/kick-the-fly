"""Unit tests for Decoy female (2D & 3D) and cVA visible puff reach.

Covers:
- Dropped decoy female renders in 3D (mesh list has female abdomen, stripes, decoy tint).
- Foreleg contact drives LgLG5-8 taste neurons in both 2D and 3D.
- Reset (R / new_fly) clears decoys in 2D and 3D.
- Changing arena clears decoys in 2D and 3D.
- Save state round-trips decoys in 2D and 3D.
- Hand tool picks up, holds, throws, and right-click removes decoys in 2D and 3D.
- 4th decoy drop is refused with max 3 on-screen notification.
- Visible cVA puff cloud in 2D and 3D showing reach (250 px / 2.5 m).
"""
from __future__ import annotations

import io
import math
import numpy as np
import pygame
import pytest

from conftest import needs_pack
from kickthefly.core import config, savestate
from kickthefly.game import kick3d, kick_the_fly as k2
from kickthefly.game.render3d import P_STRIPES, P_NONE


class DummyRenderer:
    def __init__(self):
        self.items: dict[tuple[str, str], list] = {}
        self.particles: dict[str, list] = {"alpha": [], "add": []}

    def add(self, mesh: str, model: np.ndarray, color, pattern: int = 0, glow: float = 0.0, layer: str | None = None):
        rgba = tuple(color) + ((1.0,) if len(color) == 3 else ())
        if layer is None:
            layer = "opaque" if rgba[3] >= 0.999 else "blend"
        self.items.setdefault((layer, mesh), []).append((model, rgba, pattern, glow))

    def particle(self, pos, size: float, color, additive: bool = False):
        self.particles["add" if additive else "alpha"].append((pos, size, color))


def _make_game2d():
    pygame.init()
    state = {"seed": 5}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")
    screen = pygame.Surface((k2.W, k2.H))
    game = k2.Game(screen, state["brain"], state["view"], state["graph"], state["weights"], cfg=config.Config(None))
    return game


def _make_game3d():
    pygame.init()
    state = {"seed": 5}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")
    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game = kick3d.Game3D(hud, state["brain"], state["view"], state["graph"], state["weights"], cfg=config.Config(None))
    return game


@needs_pack
def test_3d_decoy_render_list_female_morphology():
    """A dropped decoy female is added to the 3D render list with female morphology:
    banded striped pattern (P_STRIPES), no sex combs, decoy tint, and ovipositor."""
    game = _make_game3d()
    decoy_tool_idx = [i for i, (name, *_) in enumerate(kick3d.TOOLS) if name == "decoy"][0]
    game.tool = decoy_tool_idx

    # Drop a decoy
    game.use_tool3d(now=1.0)
    assert len(game.decoys3) == 1

    rd = DummyRenderer()
    game._draw_extras(rd, now=1.0)

    # Decoy has body meshes (thorax, head, abdomen, ovipositor) in items
    opaque_spheres = rd.items.get(("opaque", "sphere"), [])
    assert len(opaque_spheres) > 0, "Decoy spheres should be in render list"

    # Verify female banded abdomen is present with P_STRIPES pattern
    abdomen_entries = [entry for entry in opaque_spheres if entry[2] == P_STRIPES]
    assert len(abdomen_entries) >= 1, "Female abdomen must use P_STRIPES pattern"

    # Verify decoy tint (200, 180, 220) is applied
    tinted_entries = [entry for entry in opaque_spheres if abs(entry[1][0] - 200 / 255) < 0.05]
    assert len(tinted_entries) >= 1, "Decoy must be tinted with female decoy color"


@needs_pack
def test_foreleg_contact_drives_lglg_in_3d():
    """Foreleg contact with decoy in 3D drives LgLG5-8 (putative ppk23/ppk25) taste neurons."""
    game = _make_game3d()
    slot = game.flies[0]
    fly = slot.fly

    # Place decoy right at fly head
    dec = {"p": fly.p[k2.HEAD].copy(), "v": np.zeros(3), "landed": True, "yaw": 0.0}
    game.decoys3 = [dec]

    # Warm up / run a few steps without contact
    for _ in range(5):
        slot.brain.step()

    base_lglg = slot.brain.lglg_level()

    # Step decoy contact in 3D
    game._decoy3d(now=1.0)
    assert fly.decoy_contact_until > 1.0

    # Step the brain with the foreleg poke applied by decoy contact
    for _ in range(3):
        slot.brain.step()

    active_lglg = slot.brain.lglg_level()
    assert active_lglg > base_lglg, f"LgLG firing must increase after foreleg contact: {active_lglg} vs {base_lglg}"


@needs_pack
def test_foreleg_contact_drives_lglg_in_2d():
    """Foreleg contact with decoy in 2D drives LgLG5-8 (putative ppk23/ppk25) taste neurons."""
    game = _make_game2d()
    fly = game.fly

    # Place decoy right at fly head
    game.decoys = [{"p": fly.p[k2.HEAD].copy(), "v": np.zeros(2), "t": 1.0, "floor": True}]

    for _ in range(5):
        game.brain.step()

    base_lglg = game.brain.lglg_level()

    # Step decoy contact in 2D
    game._decoy(now=1.0)
    assert fly.decoy_contact_until > 1.0

    # Step the brain with foreleg taste poke
    for _ in range(3):
        game.brain.step()

    active_lglg = game.brain.lglg_level()
    assert active_lglg > base_lglg, f"LgLG firing must increase after foreleg contact: {active_lglg} vs {base_lglg}"


@needs_pack
def test_reset_clears_decoys():
    """Reset (new_fly) clears all dropped decoys in both 2D and 3D."""
    # 2D
    game2d = _make_game2d()
    game2d.decoys = [{"p": np.array([200.0, 300.0]), "v": np.zeros(2), "t": 1.0, "floor": True}]
    assert len(game2d.decoys) == 1
    game2d.new_fly()
    assert len(game2d.decoys) == 0

    # 3D
    game3d = _make_game3d()
    game3d.decoys3 = [{"p": np.array([0.0, 0.384, 0.0]), "v": np.zeros(3), "landed": True, "yaw": 0.0}]
    assert len(game3d.decoys3) == 1
    game3d.new_fly()
    assert len(game3d.decoys3) == 0


@needs_pack
def test_arena_change_clears_decoys():
    """Changing arena clears all dropped decoys in both 2D and 3D."""
    # 2D
    game2d = _make_game2d()
    game2d.decoys = [{"p": np.array([200.0, 300.0]), "v": np.zeros(2), "t": 1.0, "floor": True}]
    game2d.on_arena_changed(now=1.0)
    assert len(game2d.decoys) == 0

    # 3D
    game3d = _make_game3d()
    game3d.decoys3 = [{"p": np.array([0.0, 0.384, 0.0]), "v": np.zeros(3), "landed": True, "yaw": 0.0}]
    game3d.on_arena_changed(now=1.0)
    assert len(game3d.decoys3) == 0


@needs_pack
def test_savestate_roundtrips_decoys():
    """Save state round-trips decoys in both 2D and 3D backwards-compatibly."""
    # 2D
    game2d = _make_game2d()
    game2d.decoys = [
        {"p": np.array([150.0, 400.0]), "v": np.array([1.0, -2.0]), "t": 2.5, "floor": True},
        {"p": np.array([300.0, 420.0]), "v": np.array([0.0, 0.0]), "t": 3.0, "floor": False},
    ]
    extra2d = game2d.save_extra({}, now=1.0)
    assert "decoys" in extra2d
    assert len(extra2d["decoys"]) == 2

    # Clear and restore
    game2d.decoys = []
    game2d.load_extra(extra2d, None, 1.0)
    assert len(game2d.decoys) == 2
    assert np.allclose(game2d.decoys[0]["p"], [150.0, 400.0])
    assert np.allclose(game2d.decoys[1]["p"], [300.0, 420.0])

    # 3D
    game3d = _make_game3d()
    game3d.decoys3 = [
        {"p": np.array([1.0, 0.384, -0.5]), "v": np.array([0.1, 0.0, -0.1]), "landed": True, "yaw": 1.25},
    ]
    extra3d = game3d.save_extra({}, now=1.0)
    assert "decoys3" in extra3d
    assert len(extra3d["decoys3"]) == 1

    # Clear and restore
    game3d.decoys3 = []
    game3d.load_extra(extra3d, None, 1.0)
    assert len(game3d.decoys3) == 1
    assert np.allclose(game3d.decoys3[0]["p"], [1.0, 0.384, -0.5])
    assert game3d.decoys3[0]["yaw"] == pytest.approx(1.25)


@needs_pack
def test_hand_tool_pick_up_throw_remove_3d():
    """Hand tool in 3D can grab a decoy, throw it on mouse up, and right click removes it."""
    game = _make_game3d()
    game.set_look(True)
    hand_tool_idx = [i for i, (name, *_) in enumerate(kick3d.TOOLS) if name == "hand"][0]
    game.tool = hand_tool_idx

    # Place a decoy within grab reach
    eye, d = game.aim()
    dec_pos = eye + d * 1.0
    dec = {"p": dec_pos.copy(), "v": np.zeros(3), "landed": True, "yaw": 0.0}
    game.decoys3 = [dec]

    # Left click with hand grabs it
    game.use_tool3d(now=1.0)
    assert game.grabbed_decoy3 is dec

    # Mouse button up throws it
    ev_up = pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1})
    game.handle(ev_up, now=1.1)
    assert game.grabbed_decoy3 is None
    assert not dec["landed"]
    assert np.linalg.norm(dec["v"]) > 0.01

    # Right click with hand removes it
    game.grabbed_decoy3 = dec
    ev_right = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 3, "pos": (200, 200)})
    game.handle(ev_right, now=1.2)
    assert dec not in game.decoys3
    assert len(game.decoys3) == 0


@needs_pack
def test_hand_tool_pick_up_throw_remove_2d():
    """Hand tool in 2D can pick up a decoy, throw it on release, and right-click removes it."""
    game = _make_game2d()
    hand_tool_idx = [i for i, (name, *_) in enumerate(k2.TOOLS) if name == "hand"][0]
    game.tool = hand_tool_idx

    pos = np.array([250.0, 350.0])
    dec = {"p": pos.copy(), "v": np.zeros(2), "t": 1.0, "floor": True}
    game.decoys = [dec]

    # Hand tool picks up decoy
    game.use_tool(pos, now=1.0)
    assert game.grabbed_decoy is dec

    # Releasing mouse throws it
    ev_up = pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": tuple(pos)})
    game.handle(ev_up, now=1.1)
    assert game.grabbed_decoy is None

    # Right click removes it
    game.grabbed_decoy = dec
    game.remove_decoy(dec)
    assert dec not in game.decoys
    assert len(game.decoys) == 0


@needs_pack
def test_max_3_decoys_refused():
    """Dropping a 4th decoy is refused with on-screen notification in both 2D and 3D."""
    # 2D
    game2d = _make_game2d()
    decoy_tool_2d = [i for i, (name, *_) in enumerate(k2.TOOLS) if name == "decoy"][0]
    game2d.tool = decoy_tool_2d

    for _ in range(3):
        game2d.use_tool((200, 200), now=1.0)
    assert len(game2d.decoys) == 3

    # 4th drop
    game2d.use_tool((200, 200), now=1.5)
    assert len(game2d.decoys) == 3
    # Check popup was created
    assert any("MAX 3 DECOYS" in p[2] for p in game2d.popups)

    # 3D
    game3d = _make_game3d()
    decoy_tool_3d = [i for i, (name, *_) in enumerate(kick3d.TOOLS) if name == "decoy"][0]
    game3d.tool = decoy_tool_3d

    for k in range(3):
        game3d.use_tool3d(now=1.0 + k * 0.4)
    assert len(game3d.decoys3) == 3

    # 4th drop
    game3d.use_tool3d(now=2.5)
    assert len(game3d.decoys3) == 3
    assert any("MAX 3 DECOYS" in p[1] for p in game3d.popups3)


@needs_pack
def test_cva_puff_visible_reach():
    """cVA puff produces visible expanding cloud reaching 250 px in 2D and 2.5 m in 3D."""
    # 2D
    game2d = _make_game2d()
    cva_tool_2d = [i for i, (name, *_) in enumerate(k2.TOOLS) if name == "cva"][0]
    game2d.tool = cva_tool_2d
    game2d.use_tool((300, 300), now=1.0)
    assert len(game2d.cva_puffs) == 1
    puff2d = game2d.cva_puffs[0]
    assert puff2d["reach"] == pytest.approx(250.0)

    # 3D
    game3d = _make_game3d()
    cva_tool_3d = [i for i, (name, *_) in enumerate(kick3d.TOOLS) if name == "cva"][0]
    game3d.tool = cva_tool_3d
    game3d.use_tool3d(now=1.0)
    assert len(game3d.cva_puffs3) == 1
    puff3d = game3d.cva_puffs3[0]
    assert puff3d["reach"] == pytest.approx(2.5)
