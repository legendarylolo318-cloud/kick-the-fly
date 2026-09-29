"""Tool loadouts (2.13): the catalog, the presets, the hotbar, the wheel, the editor, config migration, replays, the API.

The first half needs no brain pack. The second half drives the real 2D and 3D games (marked needs_pack)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pygame
import pytest

from conftest import needs_pack
from kickthefly.core import config
from kickthefly.core import loadout as lo
from kickthefly.game import kick_the_fly as k2


# --- the catalog and the presets ------------------------------------------------------------------------------------------
def test_catalog_covers_exactly_the_games_tools():
    assert lo.TOOL_NAMES == k2.TOOL_NAMES == tuple(t[0] for t in k2.TOOLS)
    assert {t.category for t in lo.CATALOG} <= set(lo.CATEGORIES)
    assert set(lo.CATEGORIES) == {t.category for t in lo.CATALOG}, "every category has a tool"
    for t in lo.CATALOG:
        assert t.desc and t.neurons and t.tag in (lo.CONNECTOME, lo.GAME_RULE), t.name
        assert t.larva or t.larva_note, f"{t.name} is hidden in larva mode without a reason"


def test_presets_are_the_documented_ones():
    p = lo.PRESETS
    assert p["base"] == ("hand", "swatter", "torch", "freeze", "sugar")
    assert set(p["chaos"]) == {"hand", "bomb", "torch", "cleaner", "zapper", "spider", "alcohol"}
    assert set(p["chemist"]) == {"hand", "cleaner", "alcohol", "cva", "sugar", "freeze"}
    assert p["lab"] == p["all"] == lo.TOOL_NAMES and "laser" in p["lab"]
    assert set(p["pet"]) == set(p["base"]) | {"fruit"}
    assert not {"bomb", "cleaner", "zapper"} & set(p["pet"])
    for name in ("base", "chaos", "chemist", "pet"):
        assert "laser" not in p[name]


@pytest.mark.parametrize("preset", lo.CHOICES)
@pytest.mark.parametrize("mode", ["play", "lab", "pet"])
@pytest.mark.parametrize("larva", [False, True])
def test_every_preset_in_every_mode_is_valid(preset, mode, larva):
    cfg = config.Config(None)
    cfg.set("brain.mode", mode)
    cfg.set("controls.loadout_preset", preset)
    cfg.custom_loadout(["sugar", "laser", "bomb", "sugar", "nonsense"])
    cfg.set("brain.brain", "larva" if larva else "adult")
    ld = lo.resolve(cfg)
    assert ld.tools[0] == "hand" and len(set(ld.tools)) == len(ld.tools)
    assert all(lo.available(t, lab=cfg.lab, larva=larva) for t in ld.tools)
    assert ("laser" in ld.tools) <= (mode == "lab") and ("laser" not in ld.tools if larva else True)
    assert ld.n_pages == max(1, -(-len(ld.tools) // lo.PAGE_SIZE))


def test_automatic_preset_follows_the_mode():
    cfg = config.Config(None)
    assert lo.resolve(cfg).tools == list(lo.PRESETS["base"])
    cfg.set("brain.mode", "lab")
    assert lo.resolve(cfg).tools == list(lo.TOOL_NAMES)
    cfg.set("brain.mode", "pet")
    assert lo.resolve(cfg).tools == list(lo.PRESETS["pet"])


def test_larva_hides_tools_without_a_larval_mapping():
    larva = set(lo.available_tools(lab=True, larva=True))
    assert larva == {t.name for t in lo.CATALOG if t.larva}
    assert not larva & {"bomb", "spider", "alcohol", "cva", "decoy", "laser"}
    assert {"hand", "swatter", "torch", "freeze", "sugar", "cleaner"} <= larva


def test_the_hand_cannot_be_removed_or_moved():
    ld = lo.Loadout(["swatter", "sugar", "hand"])
    assert ld.tools == ["hand", "swatter", "sugar"]
    assert not ld.unequip("hand") and not ld.move("hand", 2)
    assert ld.move("sugar", 0) and ld.tools == ["hand", "sugar", "swatter"]       # nothing goes ahead of the hand
    assert ld.unequip("sugar") and "sugar" not in ld
    assert not ld.equip("laser") and ld.equip("bomb") and ld.tools[-1] == "bomb"
    assert not ld.equip("bomb"), "no duplicates"


def test_pages_only_turn_when_the_loadout_is_longer_than_ten():
    short = lo.Loadout(lo.PRESETS["base"])
    assert short.n_pages == 1 and not short.turn_page(1) and short.page == 0
    long = lo.Loadout(lo.TOOL_NAMES, lab=True)                                     # 15 tools
    assert long.n_pages == 2 and len(long.page_tools(0)) == 10 and len(long.page_tools(1)) == 5
    assert long.slot_tool(0) == "hand" and long.slot_tool(9) == long.tools[9]
    assert long.turn_page(1) and long.slot_tool(0) == long.tools[10] and long.slot_tool(5) is None
    assert long.turn_page(1) and long.page == 0, "the pages wrap"
    long.show("fruit")
    assert long.page == 1 and long.slot_of("fruit") == long.tools.index("fruit") - 10


def test_neighbor_wraps_and_starts_from_the_front_for_a_tool_off_the_hotbar():
    ld = lo.Loadout(lo.PRESETS["base"])
    assert ld.neighbor("sugar", 1) == "hand" and ld.neighbor("hand", -1) == "sugar"
    assert ld.neighbor("cva", 1) == "hand" and ld.neighbor("cva", -1) == "sugar"


# --- config: settings, keys, saved loadouts, migration ----------------------------------------------------------------------
def test_slot_keys_default_to_the_number_row_and_are_rebindable():
    c = config.Config(None)
    assert [c.keys[f"slot{i}"] for i in range(1, 11)] == list("1234567890")
    assert c.keys["page_prev"] == "-" and c.keys["page_next"] == "=" and c.keys["loadout"] == "q"
    assert "escape" in config.RESERVED_KEYS and "1" not in config.RESERVED_KEYS
    ok, _ = c.bind("slot1", "escape")
    assert not ok, "Esc stays reserved"
    ok, msg = c.bind("training", "1")
    assert ok and "swapped" in msg and c.keys["slot1"] == "t"                     # a taken key swaps, as before
    assert not c.conflicts()


def test_the_wheel_key_does_not_collide_with_any_default_binding():
    c = config.Config(None)
    assert not c.conflicts()
    assert c.keys["tool_wheel"] == "`" and c.keys["free_mouse"] == "tab", "Tab keeps freeing the mouse"


def test_new_settings_have_tips_and_labels():
    s = config.BY_KEY["controls.loadout_preset"]
    assert s.tip and s.label and s.options == lo.CHOICES and len(s.labels) == len(s.options)
    assert "Help" in config.TABS


def test_config_round_trips_the_custom_loadout_and_five_saved_ones(tmp_path):
    path = tmp_path / "config.toml"
    c = config.Config(path)
    c.set("controls.loadout_preset", "custom")
    c.custom_loadout(["hand", "bomb", "sugar"])
    for i in range(5):
        assert c.save_loadout(f"Mine {i}", ["hand", "torch"])
    assert not c.save_loadout("One too many", ["hand"]), "at most five"
    assert c.save_loadout("Mine 2", ["hand", "spider"]), "the same name replaces"
    c.first_run["tutorial_done"] = True
    assert c.save()
    d = config.Config.load(path)
    assert d["controls.loadout_preset"] == "custom" and d.loadout["custom"] == ["hand", "bomb", "sugar"]
    assert [i["name"] for i in d.loadout["saved"]] == [f"Mine {i}" for i in range(5)]
    assert d.loadout["saved"][2]["tools"] == ["hand", "spider"]
    assert d.first_run["tutorial_done"] is True and d.migrated_from is None
    d.delete_loadout("Mine 0")
    assert len(d.loadout["saved"]) == 4


def test_bad_loadout_sections_are_ignored(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('schema_version = 3\n[loadout]\ncustom = ["hand", 5, "warp drive", "sugar"]\n'
                    '[[loadout.saved]]\nname = ""\ntools = ["hand"]\n[[loadout.saved]]\nname = "ok"\ntools = "nope"\n'
                    '[first_run]\ntutorial_done = "yes"\n')
    c = config.Config.load(path)
    assert c.loadout["custom"] == ["hand", "sugar"]
    assert [i["name"] for i in c.loadout["saved"]] == ["ok"] and c.loadout["saved"][0]["tools"] == []
    assert c.first_run["tutorial_done"] is False


OLD_CONFIG = """# Kick the Fly 2.11.0 settings.
schema_version = 2

[graphics]
fullscreen = true

[brain]
mode = "lab"
arena = "fan"

[controls]
fov = 90.0

[keys]
forward = "w"
surgery = "o"
"""


def test_a_pre_2_12_config_migrates_to_all_with_a_one_time_notice(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(OLD_CONFIG)
    c = config.Config.load(path)
    assert c.migrated_from == 2 and c["controls.loadout_preset"] == "all"
    assert c.first_run["loadout_notice"] is True and c.first_run["tutorial_done"] is True
    assert any("migrated to schema 3" in w for w in c.warnings)
    assert c["graphics.fullscreen"] is True and c["brain.arena"] == "fan" and c["controls.fov"] == 90.0, "the rest is kept"
    assert c.keys["surgery"] == "o" and c.keys["slot1"] == "1"
    assert lo.resolve(c).tools == list(lo.available_tools(lab=True, larva=False))
    # the popup is shown once: the game clears the flag and saves; loading again does not migrate again
    c.first_run["loadout_notice"] = False
    assert c.save()
    d = config.Config.load(path)
    assert d.migrated_from is None and d["controls.loadout_preset"] == "all" and not d.first_run["loadout_notice"]
    assert "schema_version = 3" in path.read_text()


def test_a_1_x_config_without_a_schema_version_migrates_too(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[brain]\nscience_popups = true\n')
    c = config.Config.load(path)
    assert c.migrated_from == 1 and c["controls.loadout_preset"] == "all" and c["brain.science_popups"] is False


def test_an_old_config_that_already_chose_a_preset_keeps_it(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('schema_version = 2\n[controls]\nloadout_preset = "chaos"\n')
    c = config.Config.load(path)
    assert c["controls.loadout_preset"] == "chaos"


def test_a_fresh_install_gets_base_and_the_tutorial(tmp_path):
    c = config.Config.load(tmp_path / "nope" / "config.toml")
    assert c["controls.loadout_preset"] == "auto" and lo.resolve(c).tools == list(lo.PRESETS["base"])
    assert c.first_run == {"tutorial_done": False, "loadout_notice": False} and c.migrated_from is None


# --- replays and the Python API -----------------------------------------------------------------------------------------------
@needs_pack
def test_tool_events_are_recorded_and_replay_exactly(tmp_path):
    from kickthefly.core import replay, simcore

    br = simcore.new_brain(seed=8, warmup=50)
    rec = replay.ReplayRecorder(seed=8, backend=br.sim.backend.name, dtype="float32", signature={}, arena="room")
    rec.attach(br)
    rec.record(0, "tool", name="torch")
    for i in range(60):
        if i == 20:
            rec.record(20, "tool", name="sugar")
            br.poke("taste", None, 0.7)
        br.step()
    rec.detach()
    path = rec.save(tmp_path / "t.ktfreplay")
    player = replay.ReplayPlayer.load(path)
    assert player.tools() == ["torch", "sugar"]
    assert player.play(simcore.new_brain(seed=8, warmup=50)) == rec.spike_sha256


@needs_pack
def test_fly_loadout_in_the_python_api():
    from kickthefly import Fly

    fly = Fly(seed=3, warmup_s=0.2)
    assert fly.loadout.tools == list(lo.PRESETS["base"]) and fly.loadout.preset == "base"
    fly.set_loadout("chaos")
    assert fly.loadout.tools[0] == "hand" and "bomb" in fly.loadout.tools
    fly.set_loadout(["swatter", "cva"])
    assert fly.loadout.tools == ["hand", "swatter", "cva"] and fly.loadout.preset == "custom"
    with pytest.raises(ValueError):
        fly.set_loadout("no-such-preset")
    fly.use_tool("sugar")
    assert fly.tool == "sugar"
    with pytest.raises(ValueError):
        fly.use_tool("laser")                    # Lab-only


# --- the games ---------------------------------------------------------------------------------------------------------------
_GAMES: list = []


@pytest.fixture(autouse=True)
def _free_games():
    yield
    import gc
    import threading
    while _GAMES:
        g = _GAMES.pop()
        g.view_stop = True
        for slot in getattr(g, "flies", []):
            slot.brain.stop()
    for th in threading.enumerate():
        if th.name == "brain-view":
            th.join(timeout=2.0)
    gc.collect()


def _game(three_d: bool, mode: str = "play", **settings):
    pygame.init()
    state = {"seed": 5}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")
    cfg = config.Config(None)
    cfg.set("brain.mode", mode)
    for k, v in settings.items():
        cfg.set(k.replace("__", "."), v)
    if three_d:
        from kickthefly.game import kick3d
        g = kick3d.Game3D(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), state["brain"], state["view"], state["graph"],
                          state["weights"], cfg=cfg)
    else:
        g = k2.Game(pygame.Surface((k2.W, k2.H)), state["brain"], state["view"], state["graph"], state["weights"], cfg=cfg)
    _GAMES.append(g)
    return g


def key(k, typ=pygame.KEYDOWN, mod=0):
    return pygame.event.Event(typ, key=k, mod=mod, unicode="", scancode=0)


def press(g, k):
    if g.three_d:
        g.handle3d(key(k), 1.0, lambda p: p)
    else:
        g.handle(key(k), 1.0)


@needs_pack
@pytest.mark.parametrize("three_d", [False, True])
def test_number_keys_pick_hotbar_slots_and_unused_keys_do_nothing(three_d):
    g = _game(three_d)
    assert g.loadout.tools == list(lo.PRESETS["base"]) and g.tool_name() == "hand"
    for i, name in enumerate(lo.PRESETS["base"]):
        press(g, pygame.K_1 + i)
        assert g.tool_name() == name
    press(g, pygame.K_9)
    assert g.tool_name() == "sugar", "slot 9 is empty in Base"
    press(g, pygame.K_EQUALS)
    press(g, pygame.K_MINUS)
    assert g.loadout.page == 0 and g.tool_name() == "sugar", "- and = do nothing with 10 tools or fewer"


@needs_pack
def test_a_long_loadout_pages_with_minus_and_equals_and_the_number_row_follows_the_page():
    g = _game(True, "lab")
    assert len(g.loadout) == 15 and g.loadout.n_pages == 2
    press(g, pygame.K_0)
    assert g.tool_name() == g.loadout.tools[9]
    press(g, pygame.K_EQUALS)
    assert g.loadout.page == 1
    press(g, pygame.K_1)
    assert g.tool_name() == g.loadout.tools[10]
    press(g, pygame.K_0)
    assert g.tool_name() == g.loadout.tools[10], "page 2 has only five tools: slot 10 is empty"
    press(g, pygame.K_MINUS)
    assert g.loadout.page == 0


@needs_pack
@pytest.mark.parametrize("three_d", [False, True])
def test_rebound_slot_keys_work_and_the_mouse_wheel_walks_the_loadout(three_d):
    g = _game(three_d)
    g.cfg.bind("slot1", "z")
    g.cfg.bind("slot2", "1")
    press(g, pygame.K_z)
    assert g.tool_name() == "hand"
    press(g, pygame.K_1)
    assert g.tool_name() == "swatter"
    g.select_tool("sugar")
    if three_d:
        g.look = True
    ev = pygame.event.Event(pygame.MOUSEWHEEL, y=-1, x=0, pos=(10, 10))
    (g.handle3d(ev, 1.0, lambda p: p) if three_d else g.handle(ev, 1.0))
    assert g.tool_name() == "hand", "wheel down wraps to the first tool"
    ev = pygame.event.Event(pygame.MOUSEWHEEL, y=1, x=0, pos=(10, 10))
    (g.handle3d(ev, 1.0, lambda p: p) if three_d else g.handle(ev, 1.0))
    assert g.tool_name() == "sugar"


@needs_pack
def test_the_wheel_reaches_tools_that_are_not_on_the_hotbar_and_escape_cancels():
    from kickthefly.ui import loadout_ui

    g = _game(False)
    assert "bomb" not in g.loadout and "bomb" in g._wheel_tools() and "laser" not in g._wheel_tools()
    n = len(g._wheel_tools())
    assert loadout_ui.wheel_pick((0, -100), n) == 0 and loadout_ui.wheel_pick((0, 100), n) == n // 2 or n % 2
    assert loadout_ui.wheel_pick((5, 5), n) is None, "the middle picks nothing"
    press(g, pygame.K_BACKQUOTE)
    assert g.kwheel_open
    i = g._wheel_tools().index("bomb")
    ang = i / n * 2 * np.pi
    g.kwheel_vec = [100 * np.sin(ang), -100 * np.cos(ang)]
    g.draw(1.0, (400, 300))                                         # the wheel draws
    g.handle(key(pygame.K_BACKQUOTE, pygame.KEYUP), 1.0)
    assert not g.kwheel_open and g.tool_name() == "bomb"
    press(g, pygame.K_BACKQUOTE)
    g.kwheel_vec = [0, -100]
    g.handle(key(pygame.K_ESCAPE), 1.0)
    assert not g.kwheel_open and g.tool_name() == "bomb", "Esc cancels the wheel without picking"


@needs_pack
def test_the_wheel_is_reachable_in_3d_and_the_mouse_points_instead_of_looking():
    g = _game(True)
    g.look = True
    press(g, pygame.K_BACKQUOTE)
    assert g.kwheel_open
    yaw = g.player.yaw
    g.update_player(1 / 60, {}, (40.0, 0.0))
    assert g.player.yaw == yaw and g.kwheel_vec[0] > 0, "mouse movement points, the view stays put"
    g.handle3d(key(pygame.K_BACKQUOTE, pygame.KEYUP), 1.0, lambda p: p)
    assert not g.kwheel_open
    g.kwheel_open = True
    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    g._draw_tool_wheel(hud)


@needs_pack
def test_the_wheel_and_editor_key_are_ignored_while_a_menu_or_overlay_is_open():
    g = _game(False)
    g.open_menu("pause")
    g.open_wheel()
    assert not g.kwheel_open
    g.menu.close()
    g.help_open = True
    g.open_wheel()
    assert not g.kwheel_open


@needs_pack
def test_laser_only_in_lab_and_leaving_lab_puts_it_down():
    g = _game(True, "lab")
    assert g.select_tool("laser") and g.tool_name() == "laser"
    g.set_setting("brain.mode", "play", save=False)
    assert g.tool_name() == "hand" and "laser" not in g.loadout and not g.select_tool("laser")


@needs_pack
def test_larva_brain_hides_tools_in_the_game_loadout():
    g = _game(False)
    g.brain_type = "larva"
    g.refresh_loadout()
    assert all(lo.BY_NAME[t].larva for t in g.loadout.tools)
    assert not g.select_tool("bomb") and g.select_tool("sugar")


@needs_pack
def test_the_editor_equips_reorders_saves_and_never_drops_the_hand():
    from kickthefly.ui import loadout_ui

    g = _game(False)
    g.open_loadout_editor()
    assert g.menu.screen == "loadout" and g.clock.menu_paused
    screen = pygame.Surface((k2.W, k2.H))
    g.menu.draw(screen, (300, 300), 1.0)                            # the page draws without raising
    m = g.menu
    loadout_ui.toggle(m, "bomb")
    assert "bomb" in g.loadout and g.cfg["controls.loadout_preset"] == "custom"
    assert g.cfg.loadout["custom"][-1] == "bomb"
    loadout_ui.toggle(m, "hand")
    assert g.loadout.tools[0] == "hand" and g.loadout.tools.count("hand") == 1
    loadout_ui.drop_on_slot(m, ("tool", "bomb", 5), 1)
    assert g.loadout.tools[1] == "bomb"
    loadout_ui.drop_on_slot(m, ("tool", "spider", None), 2)
    assert g.loadout.tools[2] == "spider"
    loadout_ui.drop_off(m, ("tool", "hand", 0))
    assert "hand" in g.loadout
    loadout_ui.drop_off(m, ("tool", "spider", 2))
    assert "spider" not in g.loadout
    loadout_ui.save_current(m, "Mine")
    assert g.cfg.loadout["saved"][0]["name"] == "Mine"
    loadout_ui.choose_preset(m, "chemist")
    assert g.loadout.tools == [t for t in lo.PRESETS["chemist"] if t != "hand"][:0] + list(lo.PRESETS["chemist"])
    loadout_ui.load_saved(m, g.cfg.loadout["saved"][0])
    assert g.loadout.tools[:2] == ["hand", "bomb"]
    m.lo_from = "chaos"
    loadout_ui.reset_to_preset(m)
    assert g.loadout.tools == list(lo.PRESETS["chaos"])
    m.handle(key(pygame.K_ESCAPE), (0, 0))
    assert not m.open and not g.clock.menu_paused, "Esc closes the editor and the game resumes"


@needs_pack
def test_the_editor_page_handles_a_drag_from_the_hotbar_to_the_grid_and_back():
    g = _game(False)
    g.open_loadout_editor()
    screen = pygame.Surface((k2.W, k2.H))
    m = g.menu
    m.draw(screen, (0, 0), 1.0)
    src = next(r for r, kind, d in m.hits if kind == "dragsrc" and d["id"] == ("lo-slot", 4))       # sugar
    down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=src.center)
    m.handle(down, src.center)
    m.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(src.centerx, 400), rel=(0, 0), buttons=(1, 0, 0)), (src.centerx, 400))
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(src.centerx, 400)), (src.centerx, 400))
    assert "sugar" not in g.loadout, "dragging a slot off the hotbar removes it"
    m.draw(screen, (0, 0), 1.0)
    card, clip = next((r, d["clip"]) for r, kind, d in m.hits if kind == "dragsrc" and d["id"] == ("lo-card", "sugar"))
    m.scroll["loadout"] = m.content_h["loadout"]                    # the Reward cards are down the page: scroll to them
    m.draw(screen, (0, 0), 1.0)
    card = next(r for r, kind, d in m.hits if kind == "dragsrc" and d["id"] == ("lo-card", "sugar"))
    assert clip.contains(card), "the card is visible after scrolling"
    slot = next(r for r, kind, d in m.hits if kind == "drop" and d["id"] == ("lo-drop", 1))
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=card.center), card.center)
    m.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=slot.center, rel=(0, 0), buttons=(1, 0, 0)), slot.center)
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=slot.center), slot.center)
    assert g.loadout.tools[1] == "sugar", "dragging a card onto a slot puts it there"
    m.scroll["loadout"] = 0
    m.draw(screen, (0, 0), 1.0)
    card = next(r for r, kind, d in m.hits if kind == "dragsrc" and d["id"] == ("lo-card", "bomb"))
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=card.center), card.center)
    m.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=card.center), card.center)
    assert "bomb" in g.loadout, "a plain click on a card equips it"


@needs_pack
def test_the_q_key_opens_the_editor_in_2d_and_3d_but_not_in_photo_mode():
    for three_d in (False, True):
        g = _game(three_d)
        press(g, pygame.K_q)
        assert g.menu.screen == "loadout"
    g.menu.close()
    g.photo_mode = True
    press(g, pygame.K_q)
    assert not g.menu.open, "Q flies the photo camera down there"


@needs_pack
@pytest.mark.parametrize("three_d", [False, True])
def test_the_hotbar_draws_pages_and_clicks(three_d):
    g = _game(three_d, "lab")
    surf = g.screen if not three_d else g.screen
    g._draw_toolbar(surf)
    assert len(g.tool_rects) == 10 and len(g.page_rects) == 2
    (r_next, d) = g.page_rects[1]
    assert g.hotbar_click(r_next.center) and g.loadout.page == 1
    g._draw_toolbar(surf)
    assert len(g.tool_rects) == 5
    assert g.hotbar_click(g.tool_rects[2].center) and g.tool_name() == g.loadout.tools[12]


@needs_pack
def test_load_state_still_restores_the_tool_by_index(tmp_path):
    from kickthefly.core import savestate

    g = _game(False)
    g.select_tool("sugar")
    savestate.save_game(g, tmp_path / "s.ktfsave")
    g.select_tool("hand")
    savestate.load_game(g, tmp_path / "s.ktfsave")
    assert g.tool_name() == "sugar"


@needs_pack
@pytest.mark.parametrize("three_d", [False, True])
def test_fruit_is_eaten_exactly_like_sugar(three_d):
    g = _game(three_d, "lab")
    g.select_tool("fruit")
    if three_d:
        g.use_tool3d(g.clock.now + 1)
        items = g.sugars3
    else:
        g.use_tool((300, 300), g.clock.now + 1)
        items = g.sugars
    assert len(items) == 1 and items[0]["fruit"] is True
    assert g.tool_uses == 1 and g.last_tool_used == "fruit"


# --- the localization catalog ---------------------------------------------------------------------------------------------------
def test_new_ui_strings_are_in_the_catalogs():
    import importlib.util

    spec = importlib.util.spec_from_file_location("i18n_sync", Path(__file__).resolve().parent.parent / "tools" / "i18n_sync.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0, "run: python tools/i18n_sync.py"


def test_tr_is_used_for_the_editors_user_facing_text():
    """No bare string literal is drawn by the 2.13 pages: every text the pages draw is a tr() call, a variable, or an
    f-string of variables."""
    root = Path(__file__).resolve().parent.parent / "kickthefly" / "ui"
    for name in ("loadout_ui.py", "tutorial.py", "help_ui.py", "crashscreen.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("text", "wrapped", "button"):
                args = node.args[2:3] if node.func.attr in ("text", "wrapped", "button") else []
                for a in args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        assert not re.search(r"[A-Za-z]{3,}", a.value) or a.value in ("x", "<", ">"), \
                            f"{name}:{node.lineno} draws {a.value!r} without tr()"


@needs_pack
def test_loading_a_save_made_with_a_tool_this_mode_does_not_allow_puts_it_down(tmp_path):
    from kickthefly.core import savestate

    g = _game(False, "lab")
    g.select_tool("laser")
    savestate.save_game(g, tmp_path / "lab.ktfsave")
    g.set_setting("brain.mode", "play", save=False)
    savestate.load_game(g, tmp_path / "lab.ktfsave")
    assert g.tool_name() == "hand" and "laser" not in g.loadout


@needs_pack
def test_the_wheel_can_be_pointed_at_with_a_free_mouse_in_3d():
    g = _game(True)
    g.look = False
    press(g, pygame.K_BACKQUOTE)
    assert g.kwheel_open
    from kickthefly.ui import loadout_ui

    cx, cy = loadout_ui.wheel_center(g)
    g.wheel_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(cx, cy - 120), rel=(0, 0), buttons=(0, 0, 0)))
    assert g.kwheel_vec == [0, -120]
    g.handle3d(key(pygame.K_BACKQUOTE, pygame.KEYUP), 1.0, lambda p: p)
    assert g.tool_name() == g._wheel_tools()[0]


# --- regressions found in QA (Opus) ---------------------------------------------------------------------------------------
def test_an_old_config_keeps_its_own_keys_when_a_new_actions_default_collides(tmp_path):
    """A pre-2.13 config with Training on Q (the editor's new default) and Mute on ` (the wheel's): before the fix both
    keys stayed bound twice (the conflict fix reset the new action to its default, the same key), so the editor and the
    wheel were unreachable and Settings showed a conflict. Now the player's keys win and the new actions start unbound."""
    path = tmp_path / "config.toml"
    path.write_text('schema_version = 2\n[keys]\ntraining = "q"\nmute = "`"\n[gamepad]\nbig_view = "x"\n')
    c = config.Config.load(path)
    assert c.keys["training"] == "q" and c.keys["mute"] == "`" and c.pad["big_view"] == "x"
    assert c.keys["loadout"] == "" and c.keys["tool_wheel"] == "" and c.pad["loadout"] == ""
    assert not c.conflicts() and c.action_for("q") == "training" and c.action_for("") is None
    assert any("unbound" in w for w in c.warnings)
    assert c.save()
    again = config.Config.load(path)
    assert again.keys["loadout"] == "" and again.pad["loadout"] == "" and not again.conflicts() and not again.warnings
    ok, _ = again.bind("loadout", "f9")
    assert ok and again.keys["loadout"] == "f9" and not again.conflicts()


def test_texts_name_the_players_key_or_say_where_to_bind_it():
    from kickthefly.ui import loadout_ui

    c = config.Config(None)
    assert loadout_ui.key_label(c, "loadout") == "Q"
    c.keys["loadout"] = ""
    assert "Settings > Controls" in loadout_ui.key_label(c, "loadout")


def test_a_control_character_in_a_loadout_name_never_makes_config_toml_unreadable(tmp_path):
    """A hand-edited name with a TOML escape (\\n, \\t) used to be written back raw: the next launch couldn't parse the
    file and every setting fell back to defaults (fullscreen lost here)."""
    path = tmp_path / "config.toml"
    path.write_text('schema_version = 3\n[graphics]\nfullscreen = true\n[[loadout.saved]]\nname = "two\\nlines\\ttab"\n'
                    'tools = ["hand"]\n')
    c = config.Config.load(path)
    assert c.loadout["saved"][0]["name"] == "two lines tab"
    assert config._toml_value("a\nb\x07c\x7f") == '"a\\u000ab\\u0007c\\u007f"'
    c.save_loadout("raw\ttab", ["hand"])          # even if something hands save_loadout a control character
    assert c.save()
    again = config.Config.load(path)
    assert again["graphics.fullscreen"] is True and not again.warnings and not (tmp_path / "config.toml.bad").exists()
    assert [i["name"] for i in again.loadout["saved"]] == ["two lines tab", "raw tab"]


@needs_pack
@pytest.mark.parametrize("three_d", [False, True])
def test_esc_on_the_tool_wheel_cancels_it_without_opening_the_pause_menu(three_d):
    """Found in QA: the pause menu's Esc handling ran before the wheel saw the key, so Esc cancelled the wheel and also
    opened the pause menu."""
    g = _game(three_d)
    g.select_tool("sugar")
    press(g, pygame.K_BACKQUOTE)
    g.kwheel_vec = [0, -120]
    press(g, pygame.K_ESCAPE)
    assert not g.kwheel_open and not g.menu.open and g.tool_name() == "sugar"
    press(g, pygame.K_ESCAPE)
    assert g.menu.open and g.menu.screen == "pause", "with the wheel closed, Esc opens the menu as before"


@needs_pack
@pytest.mark.parametrize("three_d", [False, True])
def test_a_digit_rebound_to_the_big_view_still_closes_it(three_d):
    """2.13 made the digits rebindable: binding slot 3 to B swaps the big view onto 3. In the big view, 1/2/3/0 pick
    its camera, so 3 opened the view and then only turned it to the top camera; it must close it again."""
    g = _game(three_d)
    ok, msg = g.cfg.bind("slot3", "b")
    assert ok and g.cfg.keys["big_view"] == "3", msg
    press(g, pygame.K_3)
    assert g.big_view
    press(g, pygame.K_3)
    assert not g.big_view, "the big view's own key closes it"
    press(g, pygame.K_3)
    press(g, pygame.K_2)
    assert g.big_view and g.view.preset == "side", "the other digits still pick the camera"
