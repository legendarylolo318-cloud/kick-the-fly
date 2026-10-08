"""3.0 day 3, Predators: the engine (game/predators.py), its adapter in the 2D and 3D games (game/predator_play.py), the Lab
assay (lab/predators.py), and the catalog. The engine is plain logic and tested directly; the games run on the synthetic pack
(plumbing, not biology); the assay's brain tests need the real pack."""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.game import predators as pr

BOUNDS = pr.Bounds(-4.2, 4.2, 0.0, 3.0, -3.6, 3.6)


def _run(kind, seed=0, fly=None, flying=None, seconds=60.0, fly_fn=None):
    rng = np.random.default_rng(seed)
    flying = (kind == "dragonfly") if flying is None else flying
    fly = np.array(fly if fly is not None else ([0.0, 1.5, 0.0] if flying else [0.0, pr.THORAX_Y, 0.0]), float)
    p = pr.Predator(kind, pr.spawn_position(kind, BOUNDS, rng, near=fly), rng, BOUNDS)
    target = pr.Target("f", fly.copy(), flying=flying)
    events = []
    for i in range(int(seconds / pr.DT)):
        if fly_fn is not None:
            fly_fn(i, target)
        for e in p.step(pr.DT, [target]):
            events.append((i, e))
        if p.done:
            break
    return p, events


# --- the engine --------------------------------------------------------------------------------------------------------------
def test_the_same_seed_gives_the_same_attack():
    a, b = pr.trace("frog", 3), pr.trace("frog", 3)
    assert a["capture_frame"] == b["capture_frame"] and len(a["frames"]) == len(b["frames"])
    assert all(np.allclose(x[0][1], y[0][1]) for x, y in zip(a["frames"], b["frames"]))


@pytest.mark.parametrize("kind", pr.KINDS)
def test_every_predator_catches_a_fly_that_does_not_move(kind):
    p, ev = _run(kind)
    caught = [e for _, e in ev if e.kind == "capture"]
    assert caught and caught[0].target == "f" and caught[0].touch == pr.CAPTURE_TOUCH[kind]
    assert {r for r, _, _ in caught[0].touch} <= {"head", "body", "legs", "wing"}, "a capture fires real touch groups"


def test_the_frog_strike_takes_its_stated_time_and_commits_to_where_the_fly_was():
    p, ev = _run("frog", fly_fn=None)
    strike = next(i for i, e in ev if e.kind == "strike")
    cap = next(i for i, e in ev if e.kind == "capture")
    assert (cap - strike) * pr.DT <= pr.SPECS["frog"].strike_s + 2 * pr.DT
    # a fly that moves away the moment the tongue is out is missed: the frog does not track
    rng = np.random.default_rng(1)
    fly = np.array([0.0, pr.THORAX_Y, 0.0])
    frog = pr.Predator("frog", pr.spawn_position("frog", BOUNDS, rng, near=fly), rng, BOUNDS)
    t = pr.Target("f", fly.copy())
    missed, captured = False, False
    for i in range(int(40 / pr.DT)):
        evs = frog.step(pr.DT, [t])
        if any(e.kind == "strike" for e in evs):
            t.p = t.p + np.array([0.0, 0.0, 3.0])            # gone before the tongue arrives
        missed |= any(e.kind == "miss" for e in evs)
        captured |= any(e.kind == "capture" for e in evs)
        if frog.done:
            break
    assert missed and not captured


def test_the_mantis_creep_never_looms_but_the_strike_does():
    from kickthefly.lab import predators as lp

    for seed in range(5):
        tr = pr.trace("mantis", seed)
        rates = lp.loom_rates(tr)
        s = tr["strike_frame"]
        assert max(rates[:s + 1]) < 1.5, "the creep stays below LOOM_MIN (game rule): sneaking up slowly works"
        assert max(rates[s:]) > 8.0, "the strike expands fast"


def test_the_dragonfly_ignores_a_fly_that_is_walking():
    p, ev = _run("dragonfly", flying=False, seconds=40.0)
    assert not [e for _, e in ev if e.kind == "capture"]
    assert p.done, "it gives up and leaves"
    p2, ev2 = _run("dragonfly", flying=True, seconds=10.0)
    assert [e for _, e in ev2 if e.kind == "capture"]


def test_a_predator_leaves_when_nothing_is_left_to_hunt():
    rng = np.random.default_rng(0)
    p = pr.Predator("mantis", [1.0, 0.0, 0.0], rng, BOUNDS)
    gone = False
    for _ in range(int(120 / pr.DT)):
        gone |= any(e.kind == "gone" for e in p.step(pr.DT, []))
        if p.done:
            break
    assert gone and p.done


def test_a_dead_fly_is_not_prey():
    p, ev = _run("frog", seconds=5.0, fly_fn=lambda i, t: setattr(t, "alive", False))
    assert not [e for _, e in ev if e.kind in ("aim", "strike", "capture")]


def test_threats_are_a_body_and_while_striking_a_tip():
    rng = np.random.default_rng(2)
    fly = np.array([0.0, pr.THORAX_Y, 0.0])
    frog = pr.Predator("frog", pr.spawn_position("frog", BOUNDS, rng, near=fly), rng, BOUNDS)
    t = pr.Target("f", fly)
    seen_tip = False
    for _ in range(int(20 / pr.DT)):
        frog.step(pr.DT, [t])
        if frog.done:
            break
        keys = {k for k, _, _ in frog.threats()}
        assert ("frog", "body") in keys
        seen_tip |= ("frog", "tip") in keys
    assert seen_tip and frog.threats() == []


def test_unknown_predators_are_refused():
    with pytest.raises(ValueError, match="frog"):
        pr.Predator("walrus", [0, 0, 0], np.random.default_rng(0), BOUNDS)


def test_the_predators_are_in_the_catalog_as_game_rule_creatures_with_a_loom_probe():
    from kickthefly.core import loadout as lo
    from kickthefly.game import kick_the_fly as k2

    for kind in pr.KINDS:
        t = lo.BY_NAME[kind]
        assert t.category == "Creatures" and t.tag == lo.GAME_RULE and t.probes == (lo.LOOM,) and not t.larva
        assert kind in k2.TOOL_NAMES and kind in [x[0] for x in k2.TOOLS]
    assert lo.TOOL_NAMES[:15] == ("hand", "flick", "swatter", "bomb", "torch", "cleaner", "zapper", "freeze", "spider", "sugar",
                                  "alcohol", "laser", "cva", "decoy", "fruit"), "the old tools keep their number-key order"


# --- the games ---------------------------------------------------------------------------------------------------------------
def _game(three_d):
    import test_extras3 as t3

    return t3.make_game(three_d)


@pytest.fixture(autouse=True)
def _free(request):
    yield
    if "synthetic_pack" in request.fixturenames:
        import test_extras3 as t3
        import gc
        import threading
        while t3._GAMES:
            g = t3._GAMES.pop()
            g.view_stop = True
            for slot in getattr(g, "flies", []):
                slot.brain.stop()
        for th in threading.enumerate():
            if th.name == "brain-view":
                th.join(timeout=2.0)
        gc.collect()


@pytest.mark.parametrize("three_d", [False, True])
def test_the_games_convert_between_pixels_and_metres(synthetic_pack, three_d):
    g = _game(three_d)
    for pt in ([300.0, 400.0], [0.5, 0.2, -0.3]) if not three_d else ([0.5, 0.2, -0.3],):
        pt = np.array(pt)
        back = g.preds.to_game(g.preds.to_m(pt))
        assert np.allclose(back[:len(pt)] if three_d else back, pt[:2] if not three_d else pt, atol=1e-6)


@pytest.mark.parametrize("three_d", [False, True])
@pytest.mark.parametrize("kind", pr.KINDS)
def test_a_predator_spawns_once_loops_through_the_loom_measure_and_catches(synthetic_pack, three_d, kind):
    from kickthefly.game import kick_the_fly as k2

    g = _game(three_d)
    slot = g.flies[0]
    pokes = []
    real = slot.brain.poke
    slot.brain.poke = lambda region, side, strength, recruit=None: (pokes.append((region, side)), real(region, side, strength, recruit))[1]
    if three_d:
        at = np.array([slot.fly.p[k2.THX][0] + 0.6, 0.0, slot.fly.p[k2.THX][2]])
    else:
        at = np.array([float(slot.fly.p[k2.THX][0]) + 140.0, float(k2.FLOOR)])
    assert g.preds.spawn(kind, at) and not g.preds.spawn(kind, at), "one of each kind at a time"
    keys = {k for k, _, _ in g.preds.threats()}
    assert (kind, "body") in keys, "the predator is something the fly's eyes can see grow"
    caught = False
    for i in range(int(90 / pr.DT)):
        if kind == "dragonfly":
            slot.fly.escape_until = 1e9               # an airborne fly: the dragonfly only hunts those
            slot.fly.grabbed = None if not g.preds.held else slot.fly.grabbed
        g.preds.step(i * pr.DT)
        if g.preds.held:
            caught = True
            break
    assert caught, "a fly that stays put is caught"
    regions = {r for r, _ in pokes}
    assert regions & {"body", "legs", "wing", "head"}, "a capture drives the fly's real touch neurons"
    assert slot.fly.grabbed == k2.THX and g.preds.pin_for(slot) is not None
    assert slot.pending_damage >= 50 and "a " + kind in slot.damage_src


@pytest.mark.parametrize("three_d", [False, True])
def test_an_immortal_fly_is_let_go_not_eaten(synthetic_pack, three_d):
    from kickthefly.game import kick_the_fly as k2

    g = _game(three_d)
    g.immortal = True
    slot = g.flies[0]
    at = np.array([float(slot.fly.p[k2.THX][0]) + 0.6, 0.0, float(slot.fly.p[k2.THX][2])]) if three_d else \
        np.array([float(slot.fly.p[k2.THX][0]) + 140.0, float(k2.FLOOR)])
    assert g.preds.spawn("frog", at)
    for i in range(int(120 / pr.DT)):
        g.preds.step(i * pr.DT)
        if not g.preds.list:
            break
    assert not g.preds.list and not g.preds.held
    assert slot.damage_src != "eaten by a frog", "an immortal fly is released"


@pytest.mark.parametrize("three_d", [False, True])
def test_the_predators_draw_without_error(synthetic_pack, three_d):
    import pygame

    from kickthefly.game import kick_the_fly as k2, predator_play

    g = _game(three_d)
    slot = g.flies[0]
    for kind in pr.KINDS:
        at = np.array([float(slot.fly.p[k2.THX][0]) + 0.6, 0.0, float(slot.fly.p[k2.THX][2])]) if three_d else \
            np.array([float(slot.fly.p[k2.THX][0]) + 140.0, float(k2.FLOOR)])
        g.preds.spawn(kind, at)
    for i in range(240):
        g.preds.step(i * pr.DT)
    if not three_d:
        surf = pygame.Surface((k2.W, k2.H))
        predator_play.draw2d(g.preds, surf, 1.0)
    else:
        class Rec:
            n = 0

            def add(self, *a, **k):
                self.n += 1

            def particle(self, *a, **k):
                self.n += 1

        rd = Rec()
        predator_play.draw3d(g.preds, rd, 1.0)
        assert rd.n > 0 or not g.preds.list


# --- the Lab assay -----------------------------------------------------------------------------------------------------------------
def test_the_assay_is_registered_and_its_protocol_options_are_checked():
    from kickthefly.lab import labjobs, protocol

    assert "predator_escape" in labjobs.ASSAYS and labjobs.ASSAY_LABEL["predator_escape"]
    protocol.check(dict(name="a", seed=1000, flies=2, assay="predator_escape", assay_options=dict(kinds=["frog"], trials=2)))
    for bad in (dict(nope=1), dict(kinds=[]), dict(kinds=["walrus"]), dict(trials=0), dict(trials=True), dict(trials=99),
                dict(kinds="frog")):
        with pytest.raises(protocol.ProtocolError):
            protocol.check(dict(name="a", seed=1000, flies=2, assay="predator_escape", assay_options=bad))


def test_wilson_interval_matches_known_values():
    from kickthefly.lab import predators as lp

    lo, hi = lp.wilson(0, 30)
    assert lo == 0.0 and hi == pytest.approx(0.1135, abs=1e-3)
    lo, hi = lp.wilson(15, 30)
    assert lo == pytest.approx(0.3321, abs=1e-3) and hi == pytest.approx(0.6679, abs=1e-3)
    assert all(math.isnan(x) for x in lp.wilson(0, 0))


def test_the_summary_and_headline_from_fake_flies():
    from kickthefly.lab import labjobs

    def trial(esc):
        return dict(escaped=esc, captured=True, latency_s=0.2 if esc else None, lead_s=0.1 if esc else None,
                    noticed_before_strike=False, peak_level=1.0, max_loom_before_strike=0.1, strike_loom_peak=9.0, frames=10)

    flies = [dict(seed=i, kinds=["frog", "mantis"], trials={"frog": [trial(i % 2 == 0)] * 2, "mantis": [trial(False)] * 2})
             for i in range(4)]
    s = labjobs.summarize("predator_escape", flies)
    frog = next(r for r in s["rows"] if r["predator"] == "frog")
    assert frog["escapes"] == 4 and frog["trials"] == 8 and frog["ci95"][0] < 0.5 < frog["ci95"][1]
    assert labjobs.headline("predator_escape", flies[0]) == pytest.approx(0.5)


@needs_pack
def test_the_assay_runs_on_the_real_brain_and_the_mantis_creep_is_not_noticed():
    from kickthefly.lab import predators as lp

    f = lp.escape_fly(0, trials=1)
    assert set(f["trials"]) == set(pr.KINDS)
    m = f["trials"]["mantis"][0]
    assert m["max_loom_before_strike"] < 1.5 and not m["noticed_before_strike"]
    assert f["trials"]["dragonfly"][0]["captured"] and f["trials"]["frog"][0]["captured"]
    assert lp.escape_fly(0, trials=1)["trials"]["frog"][0]["peak_level"] == f["trials"]["frog"][0]["peak_level"], "deterministic"
