"""The outdoor arenas: Open field and Orchard.

The orchard's rules and the weather transduction are plain logic (kickthefly/game/outdoors.py) and are tested
directly. The rest runs a real Game3D without an OpenGL window, stepping its brain in lockstep, so it checks the
same code the game runs: bounds, flight with no ceiling, getting lost and recalled, feeding driving the real reward
neurons, and save states.
"""
import os

import numpy as np
import pytest

from conftest import needs_pack


# --- the orchard's rules --------------------------------------------------------------------------------------------
def _orchard(**kw):
    from kickthefly.game import outdoors

    return outdoors.Orchard(outdoors.scenery("orchard")["trees"], seed=1, **kw)


def test_orchard_starts_at_the_cap():
    from kickthefly.game import outdoors

    o = _orchard(cap=3)
    for tree in range(len(o.trees)):
        assert o.on_tree(tree) == 3
    assert o.counts()["slots"] == len(o.trees) * outdoors.SLOTS_PER_TREE


def test_a_fruit_is_eaten_down_then_drops():
    o = _orchard(feeds=3)
    f = o.nearest_ripe(np.zeros(3))
    assert f.fullness == 1.0
    assert not o.feed(f, 10.0) and f.fullness == pytest.approx(2 / 3)
    assert not o.feed(f, 11.0)
    assert o.feed(f, 12.0), "the third feed empties a 3-feed fruit"
    assert not f.ripe and f.fell_at == 12.0 and f.regrow_at is not None
    assert not o.feed(f, 13.0), "an empty fruit can't be fed from"


def test_regrowth_is_staggered_and_capped():
    o = _orchard(feeds=1, regrow_s=60.0, cap=4)
    tree = 0
    mine = [f for f in o.fruit if f.tree == tree and f.ripe]
    for f in mine:                                      # strip the tree bare at t = 100
        o.feed(f, 100.0)
    assert o.on_tree(tree) == 0
    grown_at = []
    t = 100.0
    while t < 400.0:
        t += 0.5
        for f in o.step(t):
            if f.tree == tree:
                grown_at.append(t)
        assert o.on_tree(tree) <= 4, "never more than the cap on one tree"
    assert len(grown_at) >= 4, "the tree grows back"
    assert len(set(grown_at)) == len(grown_at), "never two fruits on one tree in the same instant"
    assert min(grown_at) >= 100.0 + 0.75 * 60.0, "nothing regrows sooner than 0.75 x regrow_s"
    assert grown_at[0] != grown_at[-1]


def test_orchard_params_apply_to_new_fruit_and_save():
    o = _orchard(feeds=4)
    f = o.nearest_ripe(np.zeros(3))
    o.set_params(feeds=2, regrow_s=30, cap=2)
    assert f.feeds_max == 4, "fruit already hanging keep what they had"
    st = o.state(now=50.0)
    o2 = _orchard(feeds=4)
    o2.load_state(st, now=1000.0)
    assert (o2.feeds, o2.regrow_s, o2.cap) == (2, 30.0, 2)
    assert [x.feeds_left for x in o2.fruit] == [x.feeds_left for x in o.fruit]


def test_sun_light_follows_elevation_and_side():
    from kickthefly.game import outdoors

    assert outdoors.sun_light(0.0, 0.0, -5.0) == (0.0, 0.0)
    high = sum(outdoors.sun_light(0.0, 90.0, 70.0))
    low = sum(outdoors.sun_light(0.0, 90.0, 10.0))
    assert high > low
    left, right = outdoors.sun_light(0.0, 90.0, 20.0)      # sun on the fly's right
    assert right > left


@needs_pack
def test_the_alcohol_scent_shares_glomeruli_with_another_tool():
    """Documented in the README: alcohol uses the real fermentation glomeruli DM1/DM2/DP1m, and DM2 and DP1m are
    also in the zapper's scent set, so mushroom-body training on one partly generalises to the other."""
    from kickthefly.core import simcore
    from kickthefly.game import kick_the_fly as k

    br = simcore.new_brain(seed=0, warmup=0, memory=False)

    def glomeruli(tool):
        return {str(t).replace("ORN_", "") for t in br.types[br.sense[("scent", tool)]]}

    assert glomeruli("alcohol") == {"DM1", "DM2", "DP1m"}
    shared = {tool: glomeruli("alcohol") & glomeruli(tool) for tool in k.TOOL_NAMES if tool not in ("alcohol", "laser")}
    assert {t: g for t, g in shared.items() if g} == {"zapper": {"DM2", "DP1m"}}


# --- the 3D game, headless ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def game3d(tmp_path_factory):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame

    pygame.init()
    pygame.display.set_mode((64, 64))
    from kickthefly.core import config
    from kickthefly.game import kick3d
    from kickthefly.game import kick_the_fly as k2

    import random

    random.seed(3)                                           # the game's own randomness (flight targets, jitter)
    np.random.seed(3)                                        # and the ragdoll's (flailing, tremor)
    state = {"seed": 3}
    k2.load_brain(state)
    assert "error" not in state, state.get("error")
    game = kick3d.Game3D(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), state["brain"], state["view"],
                         state["graph"], state["weights"], cfg=config.Config(None))
    yield game
    game.set_setting("brain.arena", "room", save=False)


KEYS = dict(w=0, s=0, a=0, d=0, sprint=0, crouch=0, up=0, down=0)


class _Notes:
    """Every note the game logs while the block runs (game.log itself only keeps the last 7)."""

    def __init__(self, game):
        self.game, self.notes = game, []

    def __enter__(self):
        self._orig = self.game.note

        def note(text, source=None):
            self.notes.append(text)
            return self._orig(text, source)

        self.game.note = note
        return self.notes

    def __exit__(self, *exc):
        self.game.note = self._orig


def _ground(game, fly) -> None:
    """Put the fly down where it is, not flying, stunned or in an escape's refractory window."""
    from kickthefly.game import kick3d

    game._move_fly(fly, fly.p[kick3d.THX, [0, 2]])
    now = game.clock.now
    fly.escape_ready = fly.stun_until = fly.walk_until = fly.eating_until = now


def k2_thresh(name: str) -> float:
    from kickthefly.game import kick_the_fly as k2

    return k2.THRESH[name]


def _run(game, seconds: float) -> None:
    for _ in range(int(seconds * 60)):
        for slot in game.flies:
            for _ in range(3):                         # 200 brain steps a second against 60 game ticks
                slot.brain._step()
        game.clock.now += 1 / 60
        game.update3d(game.clock.now, 1 / 60, KEYS, (0, 0))


@needs_pack
def test_field_is_big_and_has_no_ceiling(game3d):
    from kickthefly.core import simcore
    from kickthefly.game import kick3d

    game3d.set_setting("brain.arena", "field", save=False)
    assert kick3d.RX >= 30 and kick3d.FLY_RY >= 60, "the field is much larger than the room, and open above"
    game3d.lab_params["field.wind_speed"] = 0.0              # no wind: nothing else launches it
    fly, br = game3d.fly, game3d.brain
    _run(game3d, 3)
    _ground(game3d, fly)                                      # on the ground, free and ready to launch
    rows = simcore.rows_of(br, "jump")                       # the real head-touch escape DNs
    with _Notes(game3d) as notes:
        simcore.drive(br, rows, 0.6)
        top, peak = 0.0, 0.0
        for _ in range(60):
            _run(game3d, 0.1)
            top = max(top, float(fly.p[kick3d.THX, 1]))
            peak = max(peak, br.level("jump"))
        simcore.undrive(br, rows)
        _run(game3d, 2)
    assert peak > k2_thresh("jump"), f"driving the head-touch DNs should read above threshold ({peak:.2f})"
    assert any(n.startswith("FLY AWAY") for n in notes), notes
    # no ceiling: where a given escape goes is random (a game rule), so test the bound itself. Indoors this target
    # would be clamped at the 3 m ceiling; outdoors the fly has to be able to reach it.
    game3d._fly_to(fly, game3d.clock.now, fly.p[kick3d.THX] + np.array([0.0, 6.0, 0.0]), 8.0)
    for _ in range(60):
        _run(game3d, 0.1)
        top = max(top, float(fly.p[kick3d.THX, 1]))
    assert top > 3.5, f"a flight 6 m up should pass where the room's ceiling was (max {top:.2f} m)"


@needs_pack
def test_dng02_take_off_reads_outdoors(game3d):
    from kickthefly.core import simcore

    game3d.set_setting("brain.arena", "field", save=False)
    game3d.lab_params["field.wind_speed"] = 0.0
    fly, br = game3d.fly, game3d.brain
    _run(game3d, 3)
    _ground(game3d, fly)                                      # on the ground, free and ready to launch
    rows = simcore.rows_of(br, "fly")                        # the 29 DNg02 wing-power descending neurons
    with _Notes(game3d) as notes:
        simcore.drive(br, rows, 0.6)
        peak, flew = 0.0, False
        for _ in range(30):
            _run(game3d, 0.1)
            peak = max(peak, br.level("fly"))
            flew = flew or game3d.clock.now < fly.escape_until
        simcore.undrive(br, rows)
    assert peak > k2_thresh("fly"), f"DNg02 should read above its take-off threshold ({peak:.2f})"
    # the reaction chain gives a giant-fiber or head-touch escape priority over a DNg02 take-off, so either launch
    # counts; what matters is that the fly leaves the ground with DNg02 high and nothing above it stops it
    assert flew and any(n.startswith(("TAKE OFF DNg02", "FLY AWAY", "DODGE")) for n in notes), notes


@needs_pack
def test_a_fly_that_leaves_is_lost_and_can_be_recalled(game3d):
    from kickthefly.game import kick3d, outdoors

    game3d.set_setting("brain.arena", "field", save=False)
    slot = game3d.flies[0]
    far = outdoors.FIELD.lost_radius + 5
    game3d._move_fly(slot.fly, (far, 0.0))
    _run(game3d, 0.2)
    assert slot.lost and any("LOST" in n for _, n, _ in game3d.log)
    game3d.do_action("recall", game3d.clock.now)
    th = slot.fly.p[kick3d.THX]
    assert not slot.lost
    assert np.hypot(th[0] - game3d.player.pos[0], th[2] - game3d.player.pos[1]) < 6.0


@needs_pack
def test_orchard_feeding_drives_the_real_reward_neurons(game3d):
    import random

    from kickthefly.game import kick3d

    # Earlier tests in this module run the game for a wall-clock duration, so both the random stream and where they
    # leave the fly depend on the machine's speed. Seed the generators and put the fly in a known state next to a
    # ripe fruit, or this passes or fails with the load.
    random.seed(17)
    np.random.seed(17)
    game3d.set_setting("brain.arena", "orchard", save=False)
    assert game3d.orchard is not None and kick3d.COLLIDERS, "trees are obstacles"
    slot, br = game3d.flies[0], game3d.brain
    fly = slot.fly
    fly.wet = fly.frost = fly.melt = fly.venom = fly.char = fly.soak = 0.0
    fly.stuck.clear()
    fly.escape_until = fly.stun_until = fly.walk_until = 0.0
    fly.health, fly.dead_at, fly.perch = kick3d.MAX_HEALTH, None, None
    slot.lost, slot.fruit, slot.landed = False, None, False
    slot.fruit_ready = 0.0
    ripe = next(f for f in game3d.orchard.fruit if f.ripe)
    game3d._move_fly(fly, (float(ripe.pos[0]) + 0.4, float(ripe.pos[2]) + 0.4))
    pam = br.sense[("reward", None)]
    fed = False
    before = after = None
    for _ in range(180):                                     # 90 s of game time: it has to find a fruit and land
        was = getattr(slot, "landed", False)
        spikes = 0
        steps = 0
        for _ in range(30):
            for _ in range(3):
                br._step()
                spikes += int(np.count_nonzero(br.sim.spikes[pam]))
                steps += 1
            game3d.clock.now += 1 / 60
            game3d.update3d(game3d.clock.now, 1 / 60, KEYS, (0, 0))
        rate = spikes / len(pam) / (steps * 0.005)
        if was and getattr(slot, "landed", False):
            after = rate
            fed = True
            break
        before = rate if before is None else 0.8 * before + 0.2 * rate
    assert fed, "the fly never landed on a fruit in 90 s of game time"
    assert after > 1.5 * before, f"feeding should drive PAM like sugar does ({before:.1f} -> {after:.1f} Hz)"
    assert slot.fruit is not None and slot.fruit.eater is slot


@needs_pack
def test_real_vs_rule_tags_for_the_orchard():
    from kickthefly.game import kick_the_fly as k

    for text in ("TO FRUIT flies to a fruit", "EATING   fruit: taste + PAM reward", "LOST     flew out of sight",
                 "RECALL   1 lost fly called back"):
        assert k.reaction_source(text) == "rule", text


@needs_pack
def test_arena_and_orchard_survive_a_save(game3d, tmp_path):
    from kickthefly.core import savestate

    game3d.set_setting("brain.arena", "orchard", save=False)
    f = game3d.orchard.nearest_ripe(np.zeros(3))
    game3d.orchard.feed(f, game3d.clock.now)
    left = [x.feeds_left for x in game3d.orchard.fruit]
    path = savestate.save_game(game3d, tmp_path / "orchard.ktfsave")
    meta = savestate.read_meta(path)
    assert meta["arena"] == "orchard"
    game3d.set_setting("brain.arena", "room", save=False)
    savestate.load_game(game3d, path)
    from kickthefly.game import kick_the_fly as k2

    assert k2.ARENAS[game3d.arena_i] == "orchard" and game3d.cfg["brain.arena"] == "orchard"
    assert [x.feeds_left for x in game3d.orchard.fruit] == left


def test_the_2d_game_stays_indoors():
    from kickthefly.game import kick_the_fly as k

    class Host:
        three_d = False
        is_larva = False                       # apply_setting reads it since 2.11

        def __init__(self):
            from kickthefly.core import config
            self.cfg = config.Config(None)
            self.notes = []
            self.arena_i = 0

        def note(self, text, source=None):
            self.notes.append(text)

    h = Host()
    h.cfg.set("brain.arena", "orchard")
    k.Game.apply_setting(h, "brain.arena")
    assert k.ARENAS[h.arena_i] == "room"
    assert any("needs the 3D game" in n for n in h.notes)


@needs_pack
def test_orchard_assay_is_reproducible():
    from kickthefly.lab import assays

    a = assays.orchard_fly(5, feeds=2, regrow_s=20, cap=2, duration_s=12)
    b = assays.orchard_fly(5, feeds=2, regrow_s=20, cap=2, duration_s=12)
    assert a["feeds"] == b["feeds"] and a["mn9_feed_hz"] == b["mn9_feed_hz"]
    assert a["feeds_per_fruit"] == 2 and a["cap"] == 2 and a["feeds"] > 0
    assert a["fruit_emptied"] >= a["feeds"] // 2 - 1
