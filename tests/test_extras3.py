"""3.0 in the running games (2D and 3D): Neurodex discovery and saving, the kill cam, the Neuron of the Day card, the
menus, keys, gamepad buttons and the settings. Runs on the synthetic pack: it checks the wiring, not any biology."""
from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path

import numpy as np
import pygame
import pytest

from kickthefly.core import config, killcam
from kickthefly.core import neurodex as nd
from kickthefly.core import neuron_of_day as notd
from kickthefly.game import kick_the_fly as k2

_GAMES: list = []


@pytest.fixture(autouse=True)
def _free_games():
    yield
    import gc
    while _GAMES:
        g = _GAMES.pop()
        g.view_stop = True
        for slot in getattr(g, "flies", []):
            slot.brain.stop()
    for th in threading.enumerate():
        if th.name == "brain-view":
            th.join(timeout=2.0)
    gc.collect()


def make_game(three_d=False, mode="play", **settings):
    pygame.init()
    state = {"seed": 5}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")
    cfg = config.Config(None)
    cfg.set("brain.mode", mode)
    for key, v in settings.items():
        cfg.set(key.replace("__", "."), v)
    if three_d:
        from kickthefly.game import kick3d
        g = kick3d.Game3D(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), state["brain"], state["view"], state["graph"],
                          state["weights"], cfg=cfg)
    else:
        g = k2.Game(pygame.Surface((k2.W, k2.H)), state["brain"], state["view"], state["graph"], state["weights"], cfg=cfg)
    _GAMES.append(g)
    g.x3.async_build = False
    g.brain.stop()                                   # the tests step the brain themselves, so they are deterministic
    time.sleep(0.15)
    return g


def run(g, steps, every=10):
    br = g.brain
    for i in range(steps):
        with br.step_lock:
            br._step()
        if i % every == 0:
            g.x3.dex_tick()
            g.x3.kc_feed()


@pytest.fixture
def game(synthetic_pack):
    return make_game()


def key(k, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode="", scancode=0)


def click(pos):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)


# --- Neurodex in play ------------------------------------------------------------------------------------------------------
def test_discovery_needs_settling_and_a_real_rise(game):
    run(game, nd.SETTLE_CHECKS * nd.CHECK_STEPS + 200)
    assert game.x3.table_state == "ready"
    assert game.x3.progress.types("adult") == {}                     # calm: nothing is discovered (3.0 Day 2 decision)
    from kickthefly.core import simcore
    simcore.drive(game.brain, game.x3.table.rows("LPLC2"), 0.5)       # the validation suite's activation current
    # (a 60-neuron type: the synthetic pack rests near 17 Hz, too high for a 2-neuron type to be significant there)
    run(game, 600)
    rec = game.x3.progress.types("adult")
    assert rec["LPLC2"]["how"] == "stimulated"


def test_discoveries_are_saved_next_to_the_training_memory(game, isolated_home):
    run(game, nd.SETTLE_CHECKS * nd.CHECK_STEPS + 100)
    from kickthefly.core import simcore
    simcore.drive(game.brain, game.x3.table.rows("LPLC2"), 0.5)
    run(game, 600)
    path = nd.progress_path()
    assert path.exists() and path.parent.name == "memory"
    again = nd.Progress(path)
    assert again.discovered("adult", "LPLC2")


def test_discovery_can_be_switched_off(synthetic_pack):
    g = make_game(brain__neurodex=False)
    run(g, nd.SETTLE_CHECKS * nd.CHECK_STEPS + 100)
    g.type_ops["MDN"] = 1
    g._apply_surgery()
    run(g, 600)
    assert g.x3.table_state == "idle" and g.x3.progress is None


def test_a_dead_fly_discovers_nothing_more(game):
    run(game, nd.SETTLE_CHECKS * nd.CHECK_STEPS + 100)
    game._die(game.flies[game.focus], game.clock.now)
    n = game.x3.progress.n_discovered("adult")
    game.type_ops["MDN"] = 1
    game._apply_surgery()
    run(game, 400)
    assert game.x3.progress.n_discovered("adult") == n


def test_toast_and_reaction_log_on_discovery(game):
    assert game.x3.ensure_progress().mark("adult", "DNp01", 5.0, "stimulated")
    game.x3.on_discovery("DNp01", True)
    assert game.x3.toasts and "DNp01" in game.x3.toasts[0][0] and "stimulation" in game.x3.toasts[0][0]
    assert any("NEURODEX" in msg for _, msg, _ in game.log)


# --- kill cam -------------------------------------------------------------------------------------------------------------
def die_with_history(g, stim="DNp01"):
    run(g, 700)
    g.type_ops[stim] = 1
    g._apply_surgery()
    run(g, 500)
    g._die(g.flies[g.focus], g.clock.now)


def test_death_offers_a_kill_cam_with_the_top_riser(game):
    die_with_history(game)
    off = game.x3.offer
    assert off is not None and 100 <= off["replay"].n_frames <= 150
    assert killcam.WINDOW_S - 0.3 <= off["replay"].seconds <= killcam.WINDOW_S
    top = off["summary"]["risers"][0]
    assert top["type"] == "DNp01" and top["rise_hz"] > 20
    assert game.x3.kc_available()


def test_the_kill_cam_plays_in_slow_motion_and_shows_recorded_frames(game):
    die_with_history(game)
    run(game, 500)                                                 # the dead brain flatlines while the tests keep stepping it
    assert game.x3.kc_start()
    p = game.x3.player
    assert p.speed == killcam.SPEED and game.x3.kc_playing() and game._overlay_open()
    p.t = p.replay.seconds * 0.9
    rates, spiked = game.x3.view_rates()
    assert rates.shape == (game.brain.n,) and len(spiked) == 0
    # the panel reads the recorded group rates, not the flatlined live ones
    fast, base = game.x3.panel_fast(game.brain)
    assert fast is not game.brain.fast and base is not game.brain.base
    i = game.brain.col["escape"]
    assert fast[i] > game.brain.fast[i] * 2                       # recorded DNp01 firing vs the dead brain now
    game.x3.kc_stop()
    assert game.x3.view_rates() is None and not game._overlay_open()


def test_kill_cam_is_skippable_by_key_click_and_pad(game):
    die_with_history(game)
    for how in ("esc", "space", "click", "pad"):
        game.x3.kc_start()
        assert game.x3.kc_playing()
        if how == "esc":
            assert game.x3.handle_event(key(pygame.K_ESCAPE))
        elif how == "space":
            assert game.x3.handle_event(key(pygame.K_SPACE))
        elif how == "click":
            game.draw(time.perf_counter(), (10, 10))
            assert game.x3.handle_event(click(game.x3.kc_rects["skip"].center))
        else:
            assert game.x3.pad_event({"killcam"})
        assert not game.x3.kc_playing(), how


def test_kill_cam_ends_by_itself(game):
    die_with_history(game)
    game.x3.kc_start()
    game.x3.player.speed = 1000.0
    time.sleep(0.05)
    game.x3.tick(0)
    game.x3.tick(0)
    assert not game.x3.kc_playing()


def test_kill_cam_can_be_switched_off(synthetic_pack):
    g = make_game(brain__killcam=False)
    die_with_history(g)
    assert g.x3.offer is None and not g.x3.kc_available()


def test_reduced_flashing_slows_it_and_steadies_the_rings(synthetic_pack):
    g = make_game(access__reduced_flashing=True)
    die_with_history(g)
    g.x3.kc_start()
    assert g.x3.player.speed == killcam.SPEED_REDUCED
    g.draw(time.perf_counter(), (10, 10))                          # drawing it must not raise
    surf = pygame.Surface((400, 300))
    g.x3.draw_killcam_rings(surf, pygame.Rect(0, 0, 356, 193), "panel", 1.0)


def test_kill_cam_draws_in_every_state_and_at_larger_text(synthetic_pack):
    for big in (False, True):
        g = make_game(access__larger_text=big)
        die_with_history(g)
        g.x3.kc_start()
        g.draw(time.perf_counter(), (10, 10))
        g.big_view = True
        g.draw(time.perf_counter(), (10, 10))
        g.x3.kc_stop()
        g.draw(time.perf_counter(), (10, 10))


def test_the_autopsy_card_offers_the_kill_cam(game):
    die_with_history(game)
    game.report = game._autopsy(game.flies[game.focus], game.clock.now)
    game.draw(time.perf_counter(), (10, 10))
    kill = [(r, w) for r, w in game.save_rects if w == "killcam"]
    assert kill
    game.handle(click(kill[0][0].center), time.perf_counter())
    assert game.x3.kc_playing()


def test_saving_the_kill_cam_uses_the_game_recorder(game, isolated_home):
    die_with_history(game)
    game.x3.kc_start()
    game.x3.kc_save()
    assert game.x3.kc_saving and game.video_recording
    game.draw(time.perf_counter(), (10, 10))
    game.capture_video_frame()
    game.x3.kc_stop()
    assert not game.video_recording and not game.x3.kc_saving


def test_the_buffer_follows_the_fly_being_watched(game):
    run(game, 400)
    b1 = game.x3.buf
    assert b1 is not None and len(b1) > 0
    game.x3.buf_brain = object()                                   # the focus moved to another brain
    game.x3.kc_feed()
    assert game.x3.buf is not b1


# --- Neuron of the Day -----------------------------------------------------------------------------------------------------
def test_the_launch_card_shows_once_a_curated_type_and_has_its_own_switch(game):
    game.x3.maybe_show_notd()
    c = game.x3.notd
    assert c is not None and nd.fact_for(c["type"]) and c["cite"]
    game.draw(time.perf_counter(), (10, 10))
    assert {"try", "off", "close"} <= set(game.x3.notd_rects)
    assert game.cfg["brain.neuron_of_day"] is True                 # default on
    game.x3.handle_event(click(game.x3.notd_rects["off"].center))
    assert game.x3.notd is None and game.cfg["brain.neuron_of_day"] is False
    game.x3.maybe_show_notd()
    assert game.x3.notd is None


def test_the_card_is_separate_from_the_science_cards(game):
    game.cfg.set("brain.science_popups", False)
    game.x3.maybe_show_notd()
    assert game.x3.notd is not None and game.science_card is None
    game.cfg.set("brain.neuron_of_day", False)
    game.x3.notd = None
    game.cfg.set("brain.science_popups", True)
    game.x3.maybe_show_notd()
    assert game.x3.notd is None


def test_try_it_sets_up_the_experiment(game):
    game.x3.maybe_show_notd()
    c = game.x3.notd
    game.draw(time.perf_counter(), (10, 10))
    game.x3.handle_event(click(game.x3.notd_rects["try"].center))
    assert game.x3.notd is None
    assert set(game.type_ops) == set(c["plan"]["types"]) and all(v == 1 for v in game.type_ops.values())


def test_try_it_in_lab_mode_aims_the_laser(synthetic_pack):
    g = make_game(mode="lab")
    g.x3.ensure_table()
    facts = {f.id: f for f in nd.facts()}
    g.x3.notd = {"plan": notd.plan(facts["mdn"], g.x3.table, lab=True)}
    g.x3.notd_try()
    assert g.laser_state.target_type == "MDN" and g.laser_state.mode == "activate" and g.x3.notd is None


def test_no_card_in_the_tutorial_or_a_menu(synthetic_pack):
    g = make_game()
    g.menu.show("pause")
    g.x3.maybe_show_notd()
    assert g.x3.notd is None


def test_first_run_notices_show_the_card_after_onboarding(synthetic_pack):
    g = make_game()
    g.cfg.first_run["tutorial_done"] = True
    g.cfg.first_run["loadout_notice"] = False
    g.show_first_run_notices()
    assert g.x3.notd is not None


# --- keys, menus, pad --------------------------------------------------------------------------------------------------------
def test_d_opens_the_neurodex_in_2d_and_is_rebindable(game):
    assert game.cfg.keys["neurodex"] == "d"
    assert game.handle(key(pygame.K_d), time.perf_counter())
    assert game.menu.open and game.menu.screen == "neurodex"
    game.menu.close()
    game.cfg.bind("neurodex", "j")
    game.cfg.bind("recall", "z")                                  # any other key swap must still work
    assert not game.x3.handle_event(key(pygame.K_d))
    assert game.x3.handle_event(key(pygame.K_j)) and game.menu.screen == "neurodex"


def test_d_is_still_walk_right_in_the_3d_game_while_looking(synthetic_pack):
    g = make_game(three_d=True)
    g.look = True
    assert not g.x3.handle_event(key(pygame.K_d)) and not g.menu.open
    g.look = False
    assert g.x3.handle_event(key(pygame.K_d)) and g.menu.screen == "neurodex"
    assert "right" in g.cfg.actions_for("d") and g.cfg.conflicts() == {}


def test_pause_menu_has_neurodex_and_share_and_they_open(game):
    game.menu.show("pause")
    surf = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game.menu.draw(surf, (0, 0))
    ids = {h[2]["id"] for h in game.menu.hits if isinstance(h[2].get("id"), tuple)}
    assert ("pause", "neurodex") in ids and ("pause", "share") in ids and ("pause", "quit") in ids
    game.menu_action("neurodex")
    assert game.menu.screen == "neurodex"
    game.menu_action("share")
    assert game.menu.screen == "share"


def test_every_new_screen_draws_in_every_mode(synthetic_pack):
    for mode in ("play", "lab", "pet"):
        for big in (False, True):
            g = make_game(mode=mode, access__larger_text=big)
            surf = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
            for screen in ("neurodex", "share", "pause"):
                g.menu.show(screen)
                g.menu.draw(surf, (300, 300))
                g.menu.close()
            g.x3.ensure_progress().mark("adult", "MDN", 4.0)
            g.menu.show("neurodex")
            g.menu._dex_state = None
            g.menu.draw(surf, (300, 300))
            g.menu._dex_state.selected = "MDN"
            g.menu.draw(surf, (300, 300))
            g.menu._dex_state.selected = "LPLC2"               # undiscovered: only "???"
            g.menu.draw(surf, (300, 300))
            g.menu.close()


def test_the_neurodex_hides_everything_about_an_undiscovered_type(game):
    from kickthefly.ui import neurodex_ui

    game.x3.ensure_table()
    game.menu.show("neurodex")
    st = neurodex_ui._state(game.menu)
    st.selected = "LPLC2"
    texts = []
    orig = game.menu.text
    game.menu.text = lambda surf, s, *a, **kw: (texts.append(str(s)), orig(surf, s, *a, **kw))[1]
    origw = game.menu.wrapped
    game.menu.wrapped = lambda surf, s, *a, **kw: (texts.append(str(s)), origw(surf, s, *a, **kw))[1]
    game.menu.draw(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), (0, 0))
    joined = " ".join(texts)
    assert "???" in joined and "LPLC2" not in joined and "Klapoetke" not in joined and "visual_projection" not in joined


def test_pad_buttons_open_browse_and_close_the_neurodex(synthetic_pack):
    from kickthefly.ui import neurodex_ui

    g = make_game(three_d=True)
    assert g.x3.pad_event({"neurodex"}) and g.menu.screen == "neurodex"
    g.x3.ensure_progress()
    for n in ("MDN", "DNp01", "LPLC2"):
        g.x3.progress.mark("adult", n, 4.0)
    g.menu.draw(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), (0, 0))
    assert neurodex_ui.pad_nav(g, {"tool_next"}) and neurodex_ui._state(g.menu).selected in ("DNp01", "LPLC2", "MDN")
    first = neurodex_ui._state(g.menu).selected
    neurodex_ui.pad_nav(g, {"tool_next"})
    assert neurodex_ui._state(g.menu).selected != first
    region = neurodex_ui._state(g.menu).region
    neurodex_ui.pad_nav(g, {"killcam"})
    assert neurodex_ui._state(g.menu).region != region
    assert neurodex_ui.pad_nav(g, {"crouch"}) and not g.menu.open


def test_the_pad_default_bindings_do_not_clash(synthetic_pack):
    cfg = config.Config(None)
    used = [b for b in cfg.pad.values() if b]
    assert len(used) == len(set(used)) and cfg.pad["neurodex"] == "dpup" and cfg.pad["killcam"] == "dpdown"


def test_the_overlays_never_change_the_brain(game):
    """Drawing the card, the toasts and the kill cam, and discovering, leave the brain's state untouched."""
    run(game, 300)
    before = (game.brain.sim.v.copy(), game.brain.override.copy(), game.brain.steps)
    game.x3.maybe_show_notd()
    game.x3.toasts.append(("x", time.perf_counter()))
    game.draw(time.perf_counter(), (10, 10))
    after = (game.brain.sim.v, game.brain.override, game.brain.steps)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1]) and before[2] == after[2]


# --- the Python API ------------------------------------------------------------------------------------------------------------
def test_api_collect_discovers_by_stimulation_and_never_touches_the_players_dex(synthetic_pack, isolated_home):
    from kickthefly import Fly

    fly = Fly(seed=3, warmup_s=0.5)
    prog = fly.collect()
    fly.step(6.0)
    assert prog.types("adult") == {}                              # a calm fly discovers nothing (3.0 Day 2 decision)
    fly.drive("type:LPLC2", amp=0.5)                              # the validation suite's activation current
    fly.step(3.0)
    assert ("LPLC2", "stimulated") in [(n, how) for _, n, how in fly.discoveries]
    assert prog.path.parent != nd.progress_path().parent and not nd.progress_path().exists()
    e = fly.neurodex("LPLC2")
    assert e["discovered"] and e["curated"]["doi"] == "10.1038/nature24626" and fly.neurodex("Nope") is None


def test_api_killcam(synthetic_pack):
    from kickthefly import Fly

    fly = Fly(seed=4, warmup_s=0.5).killcam()
    fly.step(3.0)
    fly.stimulate("type:MDN")
    fly.step(3.0)
    fly.kill()
    rep = fly.killcam_replay()
    assert rep is not None and rep.top_risers()[0]["index"] in fly.neurons("type:MDN")


def test_the_new_model_assumption_cards_fit_their_two_lines_and_carry_a_tag():
    """The Model Assumptions page shows two lines of each text; a longer one is cut with an ellipsis."""
    from kickthefly.lab import lab

    new = [a for a in lab.ASSUMPTIONS if "(3.0)" in a[0]]
    assert len(new) == 5
    for title, cat, sim, bio, docs in new:
        assert cat in ("GAME RULE", "DATASET") and docs
        assert len(sim) <= 250 and len(bio) <= 250, title


def test_kill_cam_ring_colors_follow_the_colorblind_palettes(synthetic_pack):
    from kickthefly.game import extras3

    seen = {}
    for pal in ("default", "blue-yellow", "high-contrast"):
        g = make_game(access__palette=pal)
        seen[pal] = g.x3.RISER_COLORS
        assert len(seen[pal]) >= 6
    assert seen["blue-yellow"] != seen["default"] and seen["high-contrast"] != seen["default"]
    # no red/orange next to green in the blue-yellow set: every color is blue-ish, yellow-ish or white
    for r, gr, b in seen["blue-yellow"]:
        assert b >= 120 or (r >= 200 and gr >= 180) 
    assert set(extras3.Extras.PALETTES) == {"default", "blue-yellow", "high-contrast"}


def test_the_buffer_skips_reading_rates_between_samples():
    b = killcam.Buffer(3)
    assert b.due(0)
    b.push(0, np.zeros(3))
    assert not b.due(5) and b.due(8) and b.due(-1)


# --- review (Day 1) ---------------------------------------------------------------------------------------------------------
def test_the_table_worker_uses_the_progress_the_game_thread_made(synthetic_pack):
    """The worker used to create its own Progress, racing the game thread's; the tracker and the panel could then hold two."""
    g = make_game()
    g.x3.async_build = True
    g.x3.ensure_table()
    prog_now = g.x3.progress
    assert prog_now is not None                                  # made on the game thread, before the worker starts
    t_end = time.time() + 30
    while g.x3.table_state == "building" and time.time() < t_end:
        time.sleep(0.02)
    assert g.x3.table_state == "ready" and g.x3.tracker.prog is prog_now is g.x3.progress


def test_gamepad_b_skips_the_kill_cam(game):
    die_with_history(game)
    for button in ("crouch", "up", "use", "killcam"):             # B, A, the trigger, the kill cam button
        game.x3.kc_start()
        assert game.x3.pad_event({button}) and not game.x3.kc_playing(), button


def test_the_tutorial_does_not_cover_or_steal_keys_from_the_kill_cam(game):
    """Review: seen in a real 3D frame, the first-launch tutorial drew over the kill cam and took Enter / Backspace first."""
    die_with_history(game)
    game.start_tutorial()
    drawn = []
    game.tutorial.draw = lambda surf: drawn.append(1)
    game.x3.kc_start()
    game.draw(time.perf_counter(), (10, 10))
    assert not drawn
    game.handle(key(pygame.K_RETURN), time.perf_counter())
    assert not game.x3.kc_playing()                                # Enter skipped the kill cam, not a tutorial step
