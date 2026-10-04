"""3.1.0 task 10, Puppeteer mode (game/puppeteer.py, lab/puppet_challenge.py): the levels as data, the goal logic with no game in it, the game's hooks (you cannot
hit the fly; the laser is the only tool), scoring, and (on the real brain) that every level is solvable at its par in the 3D game and in --2d and that wrong
answers do not win."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pygame
import pytest

from conftest import needs_pack
from kickthefly.game import puppeteer as pz
from kickthefly.lab import challenges
from kickthefly.lab import puppet_challenge as pc
from kickthefly.lab.laser import LaserState


# --- the levels are data ---------------------------------------------------------------------------------------------------------------
def test_ten_levels_with_a_par_a_solution_and_a_circuit_note():
    assert len(pz.LEVELS) == 10 and len({lv.id for lv in pz.LEVELS}) == 10
    specs = {spec for _, spec in pz.PALETTE}
    for lv in pz.LEVELS:
        assert lv.title and lv.brief and lv.hint and lv.goal["kind"] in ("event", "travel", "survive_loom", "sequence")
        assert lv.par == len(lv.solution) >= 1, "par is what the author's own solution needed"
        for a in lv.solution:
            assert a["target"] in specs and a["mode"] in ("activate", "silence") and a["how"] in ("laser", "latch")
        tags = {t for t, _ in lv.why}
        assert tags == {"CONNECTOME", "GAME RULE"}, f"{lv.id} must say which part is the connectome and which a rule"
        assert all(len(text) > 40 for _, text in lv.why)
    assert pz.BY_ID["detour"].goal["kind"] == "sequence" and len(pz.BY_ID["detour"].goal["steps"]) == 2


def test_the_palette_has_decoys_and_every_solution_is_in_it():
    answers = {a["target"] for lv in pz.LEVELS for a in lv.solution}
    assert len(pz.PALETTE) >= 14 and answers < {s for _, s in pz.PALETTE}
    assert len({s for _, s in pz.PALETTE}) == len(pz.PALETTE)
    assert {s for _, s in pz.PALETTE} - answers, "some palette types are never the answer"


def test_stars_and_progress():
    assert [pz.stars_for(n, 2) for n in (1, 2, 3, 4, 5, 9)] == [3, 3, 2, 2, 1, 1]
    assert pz.completed_levels({"puppet_first_steps": 1, "puppet_detour": 2, "tmaze": 9}) == 2 and pz.completed_levels({}) == 0


# --- the goal logic, with no game ----------------------------------------------------------------------------------------------------------
def test_an_event_goal_wins_on_its_event_only():
    t = pz.Tracker(dict(kind="event", event="WALK"))
    t.start(0.0)
    t.event("TURN R", 1.0)
    t.event("BACK UP", 1.0)
    assert not t.done
    t.event("WALK", 2.0)
    assert t.done


def test_travel_is_what_the_fly_walked_along_the_way_it_faces():
    t = pz.Tracker(dict(kind="travel", m=0.9))
    t.start(0.0)
    t.position(0.0, 0.0, 1.0, 0.0, 0.1)
    t.position(0.45, 0.0, 1.0, 0.0, 0.2)
    assert not t.done and t.progress == pytest.approx(0.45)
    t.position(0.45, 0.0, 0.0, 1.0, 0.3)                 # it turned on the spot: that is no distance
    assert t.progress == pytest.approx(0.45)
    t.position(0.45, 0.46, 0.0, 1.0, 0.4)                # and walked on the new way
    assert t.done and t.progress == pytest.approx(0.91)
    b = pz.Tracker(dict(kind="travel", m=-0.5))
    b.start(0.0)
    b.position(1.0, 1.0, 0.0, 1.0, 0.0)
    b.position(1.0, 0.6, 0.0, 1.0, 0.1)                  # moved against the way it faces: backward
    assert not b.done and b.progress == pytest.approx(-0.4)
    b.position(1.0, 0.45, 0.0, 1.0, 0.2)
    assert b.done and b.progress == pytest.approx(-0.55)
    assert "back" in pz.Tracker(dict(kind="travel", m=-0.5)).describe()


def test_a_wall_that_turns_the_fly_round_does_not_make_walking_into_backing_up():
    t = pz.Tracker(dict(kind="travel", m=-0.5))
    t.start(0.0)
    t.position(0.0, 0.0, 1.0, 0.0, 0.0)
    for k in range(1, 40):                                # walks right to the wall...
        t.position(0.05 * k, 0.0, 1.0, 0.0, 0.1 * k)
    for k in range(1, 80):                                # ...turns round and walks back left, now facing left
        t.position(2.0 - 0.05 * k, 0.0, -1.0, 0.0, 4.0 + 0.1 * k)
    assert not t.done and t.progress > 1.0, "walking is forward whichever way it faces"
    w = pz.Tracker(dict(kind="travel", m=0.9))
    w.start(0.0)
    w.position(0.0, 0.0, 1.0, 0.0, 0.0)
    for k in range(1, 40):                                # backs up (moves against the way it faces)
        w.position(-0.05 * k, 0.0, 1.0, 0.0, 0.1 * k)
    for k in range(1, 80):                                # at the wall it turns, and keeps backing up (now moving right while facing left)
        w.position(-2.0 + 0.05 * k, 0.0, -1.0, 0.0, 4.0 + 0.1 * k)
    assert not w.done and w.progress < -1.0


def test_flying_is_not_walking():
    t = pz.Tracker(dict(kind="travel", m=0.9))
    t.start(0.0)
    t.position(0.0, 0.0, 1.0, 0.0, 0.0)
    for k in range(1, 40):
        t.position(0.05 * k, 0.0, 1.0, 0.0, 0.1 * k, airborne=True)
    assert t.progress == 0.0 and not t.done


def test_a_teleport_is_not_distance():
    t = pz.Tracker(dict(kind="travel", m=0.9))
    t.start(0.0)
    t.position(0.0, 0.0, 1.0, 0.0, 0.0)
    t.position(3.0, 0.0, 1.0, 0.0, 0.1)
    assert t.progress == 0.0 and not t.done


def test_a_sequence_needs_its_steps_in_order_and_measures_the_second_from_the_first():
    t = pz.Tracker(pz.BY_ID["detour"].goal)
    t.start(0.0)
    t.position(0.0, 0.0, -1.0, 0.0, 0.1)
    t.position(-0.6, 0.0, -1.0, 0.0, 0.2)                # walking before the turn does nothing (the first step is an event)
    assert t.stage == 0 and not t.done and t.origin is None
    t.event("TURN R", 1.0)
    assert t.stage == 1
    t.position(2.0, 0.0, 1.0, 0.0, 1.1)                  # the second leg starts here, facing +x
    t.position(2.3, 0.0, 1.0, 0.0, 1.2)
    assert not t.done
    t.position(2.55, 0.0, 1.0, 0.0, 1.3)
    assert t.done


def test_looms_must_pass_unseen_and_a_dodge_resets_the_count():
    g = pz.BY_ID["nerves_of_steel"].goal
    t = pz.Tracker(g)
    t.start(0.0)
    assert not t.loom_due(0.5) and t.loom_due(2.1)
    t.loom_fired(2.1)
    assert not t.loom_due(2.2)                             # one at a time
    t.tick(2.1 + pz.LOOM_WINDOW_S + 0.01)
    assert t.looms_survived == 1 and t.last_loom is None
    now = 6.0
    assert t.loom_due(now)
    t.loom_fired(now)
    t.event("DODGE", now + 0.5)
    t.tick(now + 0.6)
    assert t.looms_survived == 0 and "saw" in t.message
    for k in range(3):
        now = t.next_loom + 0.01
        assert t.loom_due(now)
        t.loom_fired(now)
        t.tick(now + pz.LOOM_WINDOW_S + 0.01)
    assert t.looms_survived == 3 and t.done
    t2 = pz.Tracker(g)
    t2.start(0.0)
    t2.event("DODGE", 1.0)                                  # a dodge with no shadow out is not counted against you
    assert t2.dodged is False


def test_game_notes_become_reaction_names():
    f = pc.event_name
    assert f("WALK     DNp09 x4.1") == "WALK" and f("BACK UP  MDN x6.0") == "BACK UP" and f("TURN R   DNa01/02 R-L +3.2") == "TURN R"
    assert f("TAKE OFF DNg02 x1.62") == "TAKE OFF" and f("SONG     ps1 wing MNs x2.0") == "SONG" and f("DODGE    giant fiber DNp01 x7.0") == "DODGE"
    assert f("WALKING nonsense") is None and f("SHADOW   a looming shadow") is None and f("") is None


# --- the laser's target specs -------------------------------------------------------------------------------------------------------------
def test_side_and_prefix_specs_pick_the_right_cells(synthetic_pack):
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=1, warmup=0, memory=False)
    ls = LaserState()
    inst = np.array([str(x) for x in br.instance])
    ls.set_target("side:R:DNa01,DNa02")
    rows = ls.resolve_target_rows(br)
    assert len(rows) and all(inst[r].endswith("_R") and br.types[r] in ("DNa01", "DNa02") for r in rows)
    ls.set_target("side:L:DNa01,DNa02")
    left = ls.resolve_target_rows(br)
    assert len(left) and not set(rows) & set(left) and all(inst[r].endswith("_L") for r in left)
    ls.set_target("prefix:JO-C,JO-E")
    assert {str(br.types[r]) for r in ls.resolve_target_rows(br)} == {"JO-C", "JO-E"}
    ls.set_target("type:LPLC2,LC4")
    assert {str(br.types[r]) for r in ls.resolve_target_rows(br)} == {"LPLC2", "LC4"}
    ls.set_target(pc.NO_TARGET)
    assert len(ls.resolve_target_rows(br)) == 0
    ls.set_target("dnp01")                                    # the old plain names still work
    assert len(ls.resolve_target_rows(br)) == 2


# --- the game's hooks -----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(params=[False, True], ids=["2d", "3d"])
def game(request, synthetic_pack, tmp_path, monkeypatch):
    import test_extras3 as t3

    monkeypatch.setattr(challenges, "scores_path", lambda: tmp_path / "scores.json")
    g = t3.make_game(request.param)
    yield g
    for slot in g.flies:
        slot.brain.stop()


def _start(g):
    g.start_challenge("puppeteer")
    return g.challenge


def test_puppeteer_locks_the_game_and_ending_it_gives_it_back(game):
    ch = _start(game)
    assert game.puppet_active and ch._on_note in game.note_hooks and ch.overlay and ch.state == "select"
    assert game.select_tool("hand") is False and game.select_tool("laser") is False
    ch.end()
    assert not game.puppet_active and game.note_hooks == [] and game.challenge is None
    assert game.select_tool("hand") is True


def test_nothing_of_yours_can_poke_the_fly(game):
    slot = game.flies[0]
    game.hit(slot, 1, 1.0)
    assert slot.pending_hits, "a normal hit pokes the touch neurons"
    slot.pending_hits.clear()
    ch = _start(game)
    game.hit(slot, 1, 1.0)
    assert not slot.pending_hits, "Puppeteer: no hit reaches the fly"
    ch.end()
    game.hit(slot, 1, 1.0)
    assert slot.pending_hits


def test_in_puppeteer_nothing_is_smelled(game):
    slot = game.flies[0]
    slot.scent_now = "swatter"
    ch = _start(game)
    game._scents(slot, 0.0, (slot.fly.p[0] if not game.three_d else (0, 0)))
    assert slot.scent_now is None and not slot.sugar_scent
    ch.end()


def test_a_level_places_the_fly_and_starts_the_goal(game):
    ch = _start(game)
    lv = pz.BY_ID["about_face"]
    ch.begin(lv)
    assert ch.state == "play" and not ch.overlay and ch.tracker is not None and ch.actions == 0
    x, z, hx, hz = pc.fly_state(game, game.flies[0])
    assert (hx < 0) == (lv.facing < 0) and abs(hx) > 0.9, "it faces the level's way"
    assert abs(x - lv.start_x) < 0.15
    assert game.laser_state.target_type == pc.NO_TARGET and game.tool_name() == "laser"
    ch.end()


def test_a_reaction_wins_the_level_scores_it_and_the_card_explains(game, tmp_path):
    ch = _start(game)
    lv = pz.BY_ID["first_steps"]
    ch.begin(lv)
    game.note("WALK     DNp09 x4.0")
    ch.update(game.clock.now + 0.1)
    assert ch.state == "won" and ch.overlay
    assert ch.result["total"] == 0 and ch.result["stars"] == 3
    scores = challenges.load_scores()
    assert pz.score_key(lv.id) in scores and scores["puppeteer"] == 1
    surf = pygame.Surface((1280, 760), pygame.SRCALPHA)
    ch.draw(surf, game.clock.now, (0, 0))
    assert ch.buttons and any(b.label in ("Next level", "All done") for b in ch.buttons)
    ch.next_level()
    assert ch.level.id == "about_face" and ch.state == "play"
    ch.retry()
    assert ch.level.id == "about_face" and ch.actions == 0
    ch.end()


def test_actions_count_pulses_latches_surgery_and_hints(game):
    ch = _start(game)
    ch.begin(pz.BY_ID["serenade"])
    ls = game.laser_state
    ch.update(0.1)
    assert ch.actions == 0
    ls.firing = True                                       # a pulse with no type chosen is not an action
    ch.update(0.2)
    ls.firing = False
    ch.update(0.3)
    assert ch.actions == 0
    ch.pick_chip(5)                                        # pIP10
    assert ls.target_type == "type:pIP10"
    ls.firing = True
    ch.update(0.4)
    ch.update(0.5)                                         # held: still one pulse
    ls.firing = False
    ch.update(0.6)
    assert ch.actions == 1
    game.type_ops["LC4"] = 1                               # a surgery change through the ordinary menu
    ch.update(0.7)
    assert ch.actions == 2
    ch.show_hint()
    ch.show_hint()
    assert ch.hints == 1
    ch.end()


def test_latching_holds_a_type_on_through_a_current_and_releasing_clears_it(game):
    ch = _start(game)
    ch.begin(pz.BY_ID["long_walk"])
    br = game.flies[0].brain
    ch.set_how("latch")
    idx = [s for _, s in pz.PALETTE].index("type:DNp09")
    ch.pick_chip(idx)
    rows = np.flatnonzero(br.types == "DNp09")
    assert br.injecting and np.all(br.inject[rows] > 0) and ch.actions == 1 and ch.latched == {"type:DNp09": "activate"}
    ch.pick_chip(idx)
    assert not br.injecting and ch.latched == {} and ch.actions == 2
    ch.set_effect("silence")
    ch.pick_chip(idx)
    assert br.injecting and np.all(br.inject[rows] < 0)
    ch.begin(pz.BY_ID["long_walk"])
    assert not br.injecting and ch.latched == {}, "a new level starts with nothing held"
    ch.end()
    assert not br.injecting


def test_the_hotbar_keys_choose_palette_types_while_it_runs(game):
    ch = _start(game)
    assert game.select_slot(0) is False                     # in the level list the keys do nothing
    ch.begin(pz.BY_ID["lift_off"])
    assert game.select_slot(4) and ch.target == "type:DNg02".replace("type:", "prefix:")
    assert game.select_slot(99) is False
    ch.end()


def test_every_state_draws_and_the_list_clicks_start_a_level(game):
    ch = _start(game)
    surf = pygame.Surface((1280, 760), pygame.SRCALPHA)
    ch.draw(surf, 0.0, (0, 0))
    play = [b for b in ch.buttons if b.label == "Play"]
    assert len(play) == 10
    assert ch.click(play[3].rect.center) and ch.level.id == "reverse_gear" and ch.state == "play"
    ch.draw(surf, 0.1, (0, 0))
    assert any(b.label == "Hint" for b in ch.buttons)
    ch.show_hint()
    ch.draw(surf, 0.2, (0, 0))
    ch.tracker.done = True
    ch.update(game.clock.now + 1.0)
    ch.draw(surf, 0.3, (0, 0))
    ch.end()


def test_the_loom_level_sends_shadows_and_poke_the_loom_neurons(game):
    ch = _start(game)
    ch.begin(pz.BY_ID["nerves_of_steel"])
    br = game.flies[0].brain
    t0 = game.clock.now
    ch.update(t0 + 2.2)
    assert ch.tracker.last_loom is not None and ch._loom_until > t0 + 2.2
    assert br._pending.get(("loom", None)) is not None, "the shadow goes through the game's own looming drive"
    ch.end()


def test_the_challenges_page_lists_it_and_the_class_loads_lazily(synthetic_pack):
    keys = [k for k, *_ in challenges.INFO]
    assert "puppeteer" in keys and "puppeteer" in challenges.CLASSES and challenges.CLASSES["puppeteer"] is pc.Puppeteer
    assert challenges.stars("puppeteer", 10) == 3 and challenges.stars("puppeteer", 0) == 0
    with pytest.raises(KeyError):
        challenges.CLASSES["nope"]


# --- the real brain: every level is solvable at par, in 3D and 2D, and wrong answers do not win ---------------------------------------------------------
def _solve(rig, ch, lv, limit=30.0):
    g = rig.game
    for a in lv.solution:
        idx = [s for _, s in pz.PALETTE].index(a["target"])
        ch.set_effect(a["mode"])
        ch.set_how(a["how"])
        ch.pick_chip(idx)
        if a["how"] == "laser":
            for _ in range(a["pulses"]):
                if rig.three_d:
                    rig.face_fly()
                    g.use_tool3d(g.clock.now)
                else:
                    g.use_tool(tuple(rig.slot.fly.p[rig.k2.HEAD]), g.clock.now)
                stage = ch.tracker.stage
                t = 0.0
                while t < 2.5 and ch.state == "play" and ch.tracker.stage == stage:
                    rig.seconds(0.1)
                    t += 0.1
                rig.release()
    t = 0.0
    while ch.state == "play" and t < limit:
        rig.seconds(0.25)
        t += 0.25


@pytest.fixture(scope="module", params=[False, True], ids=["2d", "3d"])
def rig(request, tmp_path_factory):
    from kickthefly.lab import playthrough as pt

    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
    r = pt.Rig(request.param, "cpu", lab=False)
    old = challenges.scores_path
    challenges.scores_path = lambda: tmp_path_factory.mktemp("sc") / "scores.json"
    r.game.start_challenge("puppeteer")
    yield r
    challenges.scores_path = old
    r.close()


@needs_pack
@pytest.mark.parametrize("level_id", [lv.id for lv in pz.LEVELS])
def test_every_level_is_solvable_at_par_on_the_real_brain(rig, level_id):
    ch = rig.game.challenge
    lv = pz.BY_ID[level_id]
    rig.seconds(1.0)
    ch.begin(lv)
    rig.seconds(1.0)
    assert ch.state == "play", f"{level_id} was won before anything was done"
    _solve(rig, ch, lv)
    assert ch.state == "won", f"{level_id}: {ch.tracker.describe()}"
    assert ch.actions == lv.par and ch.result["stars"] == 3
    assert not rig.game.flies[0].fly.dead


WRONG = (("first_steps", "type:MDN", "activate"), ("about_face", "side:L:DNa01,DNa02", "activate"), ("lift_off", "type:DNp09", "activate"),
         ("serenade", "type:DNp09", "activate"), ("clean_antennae", "type:MN9", "activate"), ("sleep_tight", "type:DNp09", "activate"),
         ("long_walk", "type:MDN", "activate"), ("reverse_gear", "type:DNp09", "activate"))


@needs_pack
@pytest.mark.parametrize("level_id,spec,mode", WRONG, ids=[w[0] for w in WRONG])
def test_a_wrong_neuron_does_not_win(rig, level_id, spec, mode):
    ch = rig.game.challenge
    g = rig.game
    rig.seconds(1.0)
    ch.begin(pz.BY_ID[level_id])
    rig.seconds(0.5)
    ch.set_effect(mode)
    ch.set_how("latch")
    ch.pick_chip([s for _, s in pz.PALETTE].index(spec))
    rig.seconds(6.0)
    assert ch.state == "play", f"{spec} must not solve {level_id}"
    ch.retry()
