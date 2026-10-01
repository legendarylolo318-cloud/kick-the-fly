"""3.0 day 2 in the running game: the five Lab screens (draw and click), the inspector's Patch button, live thermogenetics and the
imaging view in the game loop, line surgery, the laser on a line. Synthetic pack: this checks the wiring of the UI, not biology."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from kickthefly.game import kick_the_fly as k2
from test_extras3 import _free_games, make_game  # noqa: F401  (the autouse fixture that stops the games' threads)


@pytest.fixture
def game(synthetic_pack):
    g = make_game(mode="lab")
    from kickthefly.sim import wiring as W

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    yield g
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


def draw_menu(g, page):
    g.menu.show(page)
    surf = pygame.Surface((k2.W, k2.H))
    g.menu.mouse = (5, 5)
    g.menu.draw(surf, (5, 5), time.perf_counter())
    return surf


def hit(g, ident):
    for rect, kind, d in g.menu.hits:
        if d.get("id") == ident:
            return d
    raise AssertionError(f"no control {ident!r} on {g.menu.screen}: {[d.get('id') for _, _, d in g.menu.hits][:40]}")


def click(g, ident):
    d = hit(g, ident)
    d["click"]()
    return d


def redraw(g):
    surf = pygame.Surface((k2.W, k2.H))
    g.menu.draw(surf, (5, 5), time.perf_counter())
    return surf


def test_the_lab_hub_lists_the_five_new_screens(game):
    labels = [label for label, page, _ in game.lab_pages()]
    for want in ("Genetic toolkit", "Thermogenetics", "Patch clamp", "Calcium imaging", "Pharmacology"):
        assert want in labels
    assert all(page in game.menu.pages for _, page, _ in game.lab_pages())
    draw_menu(game, "lab")


@pytest.mark.parametrize("page", ["lab_genetics", "lab_thermo", "lab_patch", "lab_imaging", "lab_pharm"])
def test_every_new_screen_draws_with_its_tags(game, page):
    draw_menu(game, page)
    assert game.menu.screen == page
    assert any(kind == "button" and d.get("id") for _, kind, d in game.menu.hits)                        # something is clickable
    back = [d for _, kind, d in game.menu.hits if kind == "button" and isinstance(d.get("id"), tuple) and d["id"][1] == "back"]
    assert back                                                                                          # every screen can be left


# --- genetic toolkit ---------------------------------------------------------------------------------------------------------------------
def test_genetics_search_select_and_surgery_on_a_line(game):
    draw_menu(game, "lab_genetics")
    hit(game, "gen_query")["commit"]("SS00727")
    redraw(game)
    st = game.menu.toolkit
    assert st.results[0].name == "SS00727" and st.selected == "SS00727"
    redraw(game)
    click(game, ("gen_op", -1))
    assert game.type_ops.get("DNp01") == -1                                   # the line's cell type is silenced, like the inspector does
    assert game.brain.override[game.brain.types == "DNp01"].max() < 0
    redraw(game)
    click(game, ("gen_op", 0))
    assert "DNp01" not in game.type_ops and not game.brain.override.any()
    redraw(game)
    click(game, "gen_laser")
    assert game.laser_state.target_type == "line:SS00727" and game.laser_state.mode == "activate"
    rows = game.laser_state.resolve_target_rows(game.brain)
    assert set(rows) == set(np.flatnonzero(game.brain.types == "DNp01"))
    redraw(game)
    click(game, "gen_trp")
    assert game.thermo_live.expressions[-1] == dict(effector="trpa1", target="line:SS00727", strength=1.0)


def test_a_line_with_no_matching_type_selects_nothing_and_says_so(game):
    draw_menu(game, "lab_genetics")
    from kickthefly.lab import genetics

    name = next(ln.name for ln in genetics.table().values() if not genetics.match(ln, game.brain.types).neurons)
    hit(game, "gen_query")["commit"](name)
    redraw(game)
    redraw(game)
    assert game.menu.toolkit.selected == name
    d = hit(game, ("gen_op", -1))
    assert d["enabled"] is False                                              # the buttons are off: nothing to silence


# --- thermogenetics ----------------------------------------------------------------------------------------------------------------------
def test_thermo_page_adds_an_expression_and_the_game_loop_applies_it(game):
    draw_menu(game, "lab_thermo")
    tl = game.thermo_live
    tl.kinetics = "steady"
    hit(game, "th_target")["commit"]("type:DNp01")
    redraw(game)
    click(game, "th_add")
    assert tl.expressions == [dict(effector="trpa1", target="type:DNp01", strength=1.0)]
    tl.temperature_c = 34.0
    for _ in range(6):
        game.frame += 1
        game._lab_tick()
    assert game.brain.injecting and set(np.flatnonzero(game.brain.inject)) == set(np.flatnonzero(game.brain.types == "DNp01"))
    redraw(game)                                                              # the row shows its activation
    tl.temperature_c = 20.0
    for _ in range(6):
        game.frame += 1
        game._lab_tick()
    assert not game.brain.injecting
    click(game, "th_clear")
    assert not tl.expressions and not game.brain.injecting


def test_a_bad_target_is_reported_and_dropped_not_a_crash(game):
    tl = game.thermo_live
    tl.add("trpa1", "type:NoSuchType")
    tl.temperature_c = 34.0
    for _ in range(6):
        game.frame += 1
        game._lab_tick()
    assert not tl.expressions and not game.brain.injecting


def test_the_thermo_arena_sets_the_temperature_under_each_fly(game):
    game.arena_i = k2.ARENAS.index("thermo")
    slot = game.flies[0]
    game._thermo_tick(slot, -1.0, True)
    assert slot.arena_temp_c == 15.0
    game._thermo_tick(slot, 1.0, True)
    assert slot.arena_temp_c == 35.0
    tl = game.thermo_live
    tl.add("trpa1", "type:DNp01")
    tl.kinetics, tl.source = "steady", "arena"
    game.frame = 3
    game._lab_tick()
    assert game.brain.injecting                                               # 35 C at the hot wall
    game._thermo_tick(slot, -1.0, True)
    game.frame = 6
    game._lab_tick()
    assert not game.brain.injecting                                           # 15 C at the cold wall


def test_the_assay_button_opens_the_assays_page_on_the_thermo_assay(game):
    draw_menu(game, "lab_thermo")
    click(game, "th_assay")
    assert game.menu.screen == "lab_assays"
    from kickthefly.lab import lab

    assert lab._state(game.menu).kind == "thermo_escape"
    redraw(game)


def test_assay_result_draws_on_the_assays_page(game):
    from kickthefly.lab import labjobs, lab

    res = labjobs.run_sync("thermo_escape", [0, 1], {"temps": (22.0, 34.0)}, None, None, workers=1)
    st = lab._state(game.menu)
    st.kind, st.result = "thermo_escape", res
    draw_menu(game, "lab_assays")
    surf = redraw(game)
    assert surf.get_bounding_rect().w > 0


# --- patch clamp -------------------------------------------------------------------------------------------------------------------------
def test_the_inspector_patch_button_opens_the_patch_page_on_that_neuron(game):
    row = int(np.flatnonzero(game.brain.types == "MDN")[0])
    game.inspect = game._neuron_info(row)
    game.big_view = True
    surf = pygame.Surface((k2.W, k2.H))
    game._draw_big_view(surf)
    assert game.patch_button is not None and game.patch_button[1] == row       # only in Lab mode
    game.patch_neuron(row)
    assert game.menu.screen == "lab_patch" and game.patch_row == row
    redraw(game)
    assert game.menu.toolkit.p_row == row and game.menu.toolkit.p_type == "MDN" and game.patch_row is None


def test_the_patch_button_is_not_offered_outside_lab_mode(synthetic_pack):
    g = make_game(mode="play")
    g.inspect = g._neuron_info(0)
    g.big_view = True
    g._draw_big_view(pygame.Surface((k2.W, k2.H)))
    assert g.patch_button is None


def test_patch_page_runs_steps_isolated_and_records_an_if_curve_and_exports(game, tmp_path):
    draw_menu(game, "lab_patch")
    hit(game, "pc_type")["commit"]("type:DNp01")
    redraw(game)
    click(game, "pc_pick")
    redraw(game)
    st = game.menu.toolkit
    assert st.p_row is not None and game.brain.types[st.p_row] == "DNp01"
    st.p_mode = 1                                                             # isolated
    redraw(game)
    st.p_amp = 0.2
    click(game, "pc_run")
    assert st.p_rec is not None and st.p_rec.mode == "isolated" and st.p_rec.spikes.sum() > 0
    redraw(game)
    click(game, "pc_if")
    st.p_job["thread"].join(timeout=60)
    redraw(game)
    assert st.p_curve is not None and st.p_curve["mode"] == "isolated" and st.p_curve["rate_hz"][-1] > st.p_curve["rate_hz"][0]
    redraw(game)
    click(game, "pc_csv")
    from kickthefly.lab import recorder

    files = list(recorder.exports_dir().rglob("patch_*.csv"))
    assert {f.name for f in files} == {"patch_trace.csv", "patch_if_curve.csv"}


def test_patch_live_trace_and_embedded_steps_on_the_running_brain(game):
    draw_menu(game, "lab_patch")
    hit(game, "pc_type")["commit"]("type:MDN")
    redraw(game)
    click(game, "pc_pick")
    redraw(game)
    st = game.menu.toolkit
    st.p_live = True
    st.p_amp = 0.3
    redraw(game)                                                              # attaches the electrode
    assert st.p_electrode is not None and game.brain.probe is not None
    click(game, "pc_run")
    for _ in range(int(0.1 / 0.005) * 5):
        with game.brain.step_lock:
            game.brain._step()
    redraw(game)
    v, sp, cur = st.p_electrode.window()
    assert len(v) > 0 and cur.max() == pytest.approx(0.3)
    st.p_live = False
    from kickthefly.lab import labtoolkit

    labtoolkit._detach(st)
    assert game.brain.probe is None and not game.brain.injecting


# --- imaging -----------------------------------------------------------------------------------------------------------------------------
def test_imaging_mode_changes_the_brain_view_and_exports(game, tmp_path):
    draw_menu(game, "lab_imaging")
    click(game, "im_on")
    assert game.imaging_live.on and game.imaging_live.session is not None
    game.imaging_live.keep_frames = True
    for _ in range(6):
        for _ in range(60):
            with game.brain.step_lock:
                game.brain._step()
        game.imaging_live.feed(game.brain)
        rates = game.imaging_live.view_rates(game.view, False)
        surf = game.view.render("panel", rates, np.zeros(0, np.int64), 0.0, False)
        game.imaging_live.recolor(game.view, surf, game.cfg["access.palette"], False)
    redraw(game)
    assert game.imaging_live.frames and game.imaging_live.session.t
    click(game, "im_csv")
    from kickthefly.lab import recorder

    assert list(recorder.exports_dir().rglob("imaging_roi.csv"))
    click(game, "im_tiff")
    assert list(recorder.exports_dir().rglob("imaging_view.tif"))
    click(game, "im_on")
    assert not game.imaging_live.on


def test_the_view_thread_path_with_imaging_on_does_not_crash_and_labels_the_view(game):
    game.imaging_live.set_on(True, game.brain, game.graph)
    br = game.brain
    for _ in range(40):
        with br.step_lock:
            br._step()
    img = game.imaging_live
    img.feed(br)
    rates = img.view_rates(game.view, bool(game.cfg["access.reduced_flashing"]))
    assert rates is not None
    surf = pygame.Surface((k2.W, k2.H))
    game._hud_overlay(surf, pygame.Rect(10, 10, 356, 193), 0.0, True)           # draws the IMAGING (MODEL) badge; must not raise
    game.big_view = True
    game._draw_big_view(surf)


def test_reduced_flashing_smooths_and_shot_noise_is_not_shown(game):
    game.cfg.set("access.reduced_flashing", True)
    game.imaging_live.set_on(True, game.brain, game.graph)
    for _ in range(300):
        with game.brain.step_lock:
            game.brain._step()
    game.imaging_live.feed(game.brain)
    game.imaging_live.view_rates(game.view, True)
    a = game.imaging_live.smooth.copy()
    game.imaging_live.view_rates(game.view, True)
    assert game.imaging_live.smooth is not None and a.shape == game.imaging_live.smooth.shape
    draw_menu(game, "lab_imaging")                                              # plots the noise-free trace under reduced flashing


# --- pharmacology ------------------------------------------------------------------------------------------------------------------------
def test_pharmacology_page_applies_a_drug_and_washes_it_out(game):
    draw_menu(game, "lab_pharm")
    st = game.menu.toolkit
    st.d_drug, st.d_dose = "cholinergic", 0.5
    redraw(game)
    applied = []
    game.set_wiring = lambda w, note=True: applied.append(w)                   # the threaded rebuild is tested with wiring itself
    click(game, "ph_apply")
    assert applied and dict(applied[-1].nt_scales) == {"acetylcholine": 0.5} and applied[-1].nt_min_conf == 0.0
    game.wiring = applied[-1]
    redraw(game)
    assert hit(game, "ph_apply")["enabled"] is False                            # already applied
    st.d_low = False
    redraw(game)
    click(game, "ph_apply")
    assert applied[-1].nt_min_conf == st.d_cut
    click(game, "ph_wash")
    assert applied[-1].is_identity


def test_pharmacology_picrotoxin_is_the_old_inhibition_block_and_the_robustness_tab_is_gone(game):
    from kickthefly.lab import labwiring

    assert "inhibition" not in [k for k, _, _ in labwiring.TABS]
    draw_menu(game, "lab_pharm")
    st = game.menu.toolkit
    st.d_drug, st.d_dose = "picrotoxin", 0.6
    redraw(game)
    applied = []
    game.set_wiring = lambda w, note=True: applied.append(w)
    click(game, "ph_apply")
    from kickthefly.sim import wiring as W

    g = game.graph
    i_old, m_old = W.modified_entries(g, W.Wiring(inhibition_scale=0.4))
    i_new, m_new = W.modified_entries(g, applied[-1])
    assert np.array_equal(i_old, i_new) and np.allclose(m_old, m_new)


def test_save_state_keeps_a_drug_in_its_wiring(game, tmp_path):
    """A save made with a drug on loads back with the drug (Wiring.as_dict carries it), and an old save's dict still loads."""
    from kickthefly.sim.wiring import Wiring
    from kickthefly.lab import pharmacology as ph

    w = ph.wiring_for({"cholinergic": 0.3}, False, 0.8)
    assert Wiring.from_dict(w.as_dict()) == w
    assert Wiring.from_dict({"min_synapses": 1, "flipped_neurons": 0, "inhibition_scale": 1.0, "label": "x"}).is_identity
