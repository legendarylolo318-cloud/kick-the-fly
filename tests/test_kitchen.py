"""3.0 day 3, Kitchen arena: the rules (game/kitchen.py) and the 3D game's use of them. The game tests run on the synthetic pack:
they check the wiring (which neurons get poked, when; what happens to a fly), not any biology."""
from __future__ import annotations

import gc
import math
import threading

import numpy as np
import pytest

from kickthefly.game import kitchen as kt


# --- the rules ---------------------------------------------------------------------------------------------------------------
def test_everything_is_inside_the_room_and_nothing_overlaps():
    for p in (kt.BOWL_POS, kt.TRAP_POS, kt.BURNER_POS):
        assert abs(p[0]) < kt.RX - 0.6 and abs(p[2]) < kt.RZ - 0.6
    x0, x1, z0, z1 = kt.SINK
    assert -kt.RX < x0 < x1 < kt.RX and -kt.RZ < z0 < z1 < kt.RZ
    for p, r in ((kt.BOWL_POS, kt.BOWL_R), (kt.TRAP_POS, kt.TRAP_R), (kt.BURNER_POS, kt.BURNER_R)):
        assert not (x0 - r < p[0] < x1 + r and z0 - r < p[2] < z1 + r), "nothing sits in the sink"
    assert np.linalg.norm(kt.BOWL_POS - kt.TRAP_POS) > kt.BOWL_R + kt.TRAP_R + 1.0
    assert np.linalg.norm(kt.BURNER_POS - kt.TRAP_POS) > kt.BURNER_R + kt.TRAP_R + 1.0


def test_the_bowl_is_the_orchards_fruit_code_with_the_fruit_in_the_bowl():
    o = kt.bowl_orchard(seed=1)
    assert len(o.fruit) == 6 and o.counts()["ripe"] == 4, "the orchard's six slots and its default cap of four"
    for f in o.fruit:
        d = math.hypot(f.pos[0] - kt.BOWL_POS[0], f.pos[2] - kt.BOWL_POS[2])
        assert d < kt.BOWL_R and 0.1 < f.pos[1] < 0.4
    f = o.nearest_ripe(kt.BOWL_POS + (0.0, 1.0, 0.0))
    for _ in range(o.feeds - 1):
        assert not o.feed(f, 10.0)
    assert o.feed(f, 11.0) and not f.ripe, "eaten down, it drops: the orchard's rule"
    assert [x.pos for x in kt.bowl_orchard(seed=1).fruit][0].tolist() == [x.pos for x in kt.bowl_orchard(seed=1).fruit][0].tolist()


def test_the_smell_falls_off_with_distance_and_is_zero_out_of_range():
    m = kt.mouth()
    assert kt.scent_strength(m) == pytest.approx(kt.SCENT_POKE)
    assert 0 < kt.scent_strength(m + (kt.SCENT_RANGE / 2, 0, 0)) < kt.SCENT_POKE
    assert kt.scent_strength(m + (kt.SCENT_RANGE + 0.1, 0, 0)) == 0.0


def test_a_fly_over_the_mouth_is_over_it_and_one_beside_it_or_far_above_is_not():
    top = kt.TRAP_POS + (0.0, kt.TRAP_H + 0.1, 0.0)
    assert kt.over_mouth(top)
    assert not kt.over_mouth(top + (kt.TRAP_MOUTH + 0.1, 0, 0))
    assert not kt.over_mouth(top + (0, 1.5, 0))
    assert not kt.over_mouth(kt.TRAP_POS + (0.0, 0.1, 0.0)), "walking on the counter beside the jar"


def test_a_fly_follows_the_smell_only_when_its_own_neurons_fire_for_it():
    tr = kt.ScentTracker()
    assert not tr.smelling, "no baseline yet"
    for _ in range(int(kt.SCENT_WARM_S * 60 / 6) + 5):             # calm: 2 Hz per neuron, no smell
        tr.update(6 / 60, 2.0, False)
    assert tr.base == pytest.approx(2.0) and not tr.smelling
    tr.update(6 / 60, 3.0, True)
    assert not tr.smelling, "3 Hz is not 2.5x its calm rate (and under the floor)"
    tr.update(6 / 60, 12.0, True)
    assert tr.smelling, "12 Hz is far above its calm rate"
    base = tr.base
    for _ in range(300):
        tr.update(6 / 60, 12.0, True)
    assert tr.base == base, "the smell can't raise its own baseline"
    cold = kt.ScentTracker()
    cold.update(0.1, 50.0, True)
    assert not cold.smelling, "a fly that has never been calm can't tell"


def test_the_sink_water_and_the_burner_are_where_they_say():
    pts = np.array([[2.5, 0.1, 2.0], [2.5, 0.5, 2.0], [0.0, 0.1, 0.0]])
    assert kt.sink_submerged(pts).tolist() == [True, False, False]
    assert kt.in_sink(pts[0]) and not kt.in_sink(pts[2])
    heat, touching = kt.burner_heat(kt.BURNER_POS + (0.0, 0.1, 0.0))
    assert heat > 0.8 and touching
    heat2, touching2 = kt.burner_heat(kt.BURNER_POS + (0.0, 1.0, 0.0))
    assert 0 < heat2 < heat and not touching2
    assert kt.burner_heat(kt.BURNER_POS + (kt.BURNER_HEAT_RANGE + 0.5, 0.3, 0.0))[0] == 0.0


def _cook_run(seed, seconds, fly=(0.5, 0.38, 0.0)):
    c = kt.Cook(np.random.default_rng(seed))
    ev = []
    for i in range(int(seconds * 60)):
        for e in c.step(1 / 60, [np.array(fly)]):
            ev.append((i / 60, e))
    return c, ev


def test_the_cook_swats_every_ten_to_twenty_seconds_at_where_the_fly_was():
    c, ev = _cook_run(1, 120.0)
    impacts = [(t, e) for t, e in ev if e.kind == "impact"]
    assert 5 <= len(impacts) <= 11
    assert kt.COOK_FIRST_S[0] <= [t for t, e in ev if e.kind == "windup"][0] <= kt.COOK_FIRST_S[1] + 0.1
    gaps = np.diff([t for t, e in impacts])
    assert gaps.min() >= kt.COOK_EVERY_S[0] + kt.COOK_RECOVER_S - 0.2 and gaps.max() <= kt.COOK_EVERY_S[1] + kt.COOK_WINDUP_S + kt.COOK_SWAT_S + kt.COOK_RECOVER_S + 0.5
    assert all(np.allclose(e.at[[0, 2]], [0.5, 0.0]) for _, e in impacts), "aimed at the fly's place when he started"
    c2, ev2 = _cook_run(1, 120.0)
    assert [t for t, _ in ev] == [t for t, _ in ev2], "deterministic"


def test_a_fly_that_has_moved_is_missed_and_one_still_there_is_hit():
    c, ev = _cook_run(3, 30.0)
    at = next(e.at for _, e in ev if e.kind == "impact")
    assert c.hit(at + (0.0, 0.38, 0.0))
    assert not c.hit(at + (kt.COOK_REACH + 0.1, 0.38, 0.0))
    assert not c.hit(at + (0.0, 2.0, 0.0)), "flying high above it"


def test_the_swatter_is_a_looming_threat_only_while_it_swings_and_it_expands_fast():
    from kickthefly.game import kick_the_fly as k2

    c = kt.Cook(np.random.default_rng(4))
    fly = np.array([0.5, 0.38, 0.0])
    head = fly + (0.05, 0.0, 0.0)
    prev, peak, seen = None, 0.0, 0
    for _ in range(int(40 * 60)):
        c.step(1 / 60, [fly])
        th = c.threats()
        if not th:
            prev = None
            continue
        seen += 1
        _, pos, r = th[0]
        d = max(float(np.linalg.norm(pos - head)), r + 0.02)
        theta = 2 * math.atan(r / d)
        if prev is not None:
            peak = max(peak, (theta - prev) * 60)
        prev = theta
    assert seen > 0 and peak > k2.LOOM_MIN + 1.0, f"a swat grows at {peak:.1f} rad/s: something the fly can see coming"
    assert c.threats() == [] or c.state == "swat"


# --- the 3D game --------------------------------------------------------------------------------------------------------------
@pytest.fixture
def g3(synthetic_pack):
    import test_extras3 as t3

    g = t3.make_game(True)
    g.set_setting("brain.arena", "kitchen", save=False)
    yield g
    while t3._GAMES:
        x = t3._GAMES.pop()
        x.view_stop = True
        for slot in getattr(x, "flies", []):
            slot.brain.stop()
    for th in threading.enumerate():
        if th.name == "brain-view":
            th.join(timeout=2.0)
    gc.collect()


def _record(slot):
    pokes = []
    real = slot.brain.poke
    slot.brain.poke = lambda region, side, strength, recruit=None: (pokes.append((region, side, round(float(strength), 3))), real(region, side, strength, recruit))[1]
    return pokes


def _frames(g, seconds):
    t0 = g.clock.now
    for i in range(int(seconds * 60)):
        g.frame += 1
        g._environment(t0 + (i + 1) / 60)
        for slot in g.flies:                                   # the trap's pin, as update3d does it
            pin = g.kitchen_pin_for(slot)
            if pin is not None:
                slot.fly.grabbed = 1
    g.clock.now = t0 + seconds


def _place(fly, xyz):
    from kickthefly.game import kick3d

    off = np.asarray(xyz, float) - fly.p[kick3d.THX]
    fly.p += off
    fly.prev = fly.p.copy()
    fly.hover = fly.p[kick3d.THX].copy()


def test_the_kitchen_is_an_arena_with_a_bowl_a_cook_and_colliders(g3):
    from kickthefly.game import kick3d, kick_the_fly as k2

    assert "kitchen" in k2.ARENAS and k2.ARENAS[-1] == "kitchen", "arenas are only ever appended"
    assert g3.kitchen is not None and g3.orchard is not None and len(g3.orchard.fruit) == 6
    assert len(kick3d.COLLIDERS) == len(kt.colliders()), "the jar and the fridge, not the living room's couch"
    g3.set_setting("brain.arena", "room", save=False)
    assert g3.kitchen is None and g3.orchard is None and len(kick3d.COLLIDERS) == len(kick3d.ROOM_COLLIDERS)


def test_the_2d_game_keeps_the_kitchen_out_and_says_why(synthetic_pack):
    import test_extras3 as t3

    g = t3.make_game(False)
    try:
        g.log.clear()
        g.set_setting("brain.arena", "kitchen", save=False, force=True)
        from kickthefly.game import kick_the_fly as k2

        assert k2.ARENAS[g.arena_i] == "room"
        assert any("needs the 3D game" in t for _, t, _ in g.log)
    finally:
        for slot in g.flies:
            slot.brain.stop()
        t3._GAMES.clear()


def test_the_vinegar_smells_on_the_fermentation_glomeruli_near_the_jar_and_not_far_away(g3):
    slot = g3.flies[0]
    pokes = _record(slot)
    _place(slot.fly, kt.TRAP_POS + (1.0, 0.38, 0.0))
    _frames(g3, 2)
    near = [p for p in pokes if p[:2] == ("scent", "alcohol")]
    assert near and all(0 < p[2] <= kt.SCENT_POKE for p in near)
    pokes.clear()
    _place(slot.fly, np.array([-3.0, 0.38, 3.0]))
    _frames(g3, 2)
    assert not [p for p in pokes if p[0] == "scent"]


def test_hovering_over_the_mouth_drops_a_fly_into_the_trap_where_it_is_stuck_and_drowns(g3):
    slot = g3.flies[0]
    fly = slot.fly
    from kickthefly.game import kick3d

    _place(fly, kt.TRAP_POS + (0.0, kt.TRAP_H + 0.15, 0.0))
    pokes = _record(slot)
    _frames(g3, 1.0)
    assert id(slot) in g3.kitchen.trapped and fly.grabbed == kick3d.THX
    assert g3.kitchen_pin_for(slot) is not None
    _frames(g3, 3)
    assert {"humid", "legs"} <= {p[0] for p in pokes}, "struggling in the vinegar: wet and leg touch"
    assert slot.pending_damage > 0 and "vinegar trap" in slot.damage_src
    assert fly.wet > 0


def test_a_fly_that_only_passes_over_the_mouth_briefly_is_not_trapped(g3):
    slot = g3.flies[0]
    _place(slot.fly, kt.TRAP_POS + (0.0, kt.TRAP_H + 0.15, 0.0))
    _frames(g3, kt.HOVER_TO_FALL_S * 0.5)
    _place(slot.fly, kt.TRAP_POS + (1.5, 0.38, 0.0))
    _frames(g3, 2)
    assert id(slot) not in g3.kitchen.trapped


def test_an_immortal_fly_escapes_the_trap_after_a_while(g3):
    g3.immortal = True
    slot = g3.flies[0]
    calls = []
    real = slot.fly.escape
    slot.fly.escape = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    _place(slot.fly, kt.TRAP_POS + (0.0, kt.TRAP_H + 0.15, 0.0))
    _frames(g3, 1.0)
    assert id(slot) in g3.kitchen.trapped and not calls
    _frames(g3, kt.TRAP_IMMORTAL_ESCAPE_S)
    assert calls, "released after the escape time, and it flies off (this test's fly doesn't move, so it can fall in again)"
    assert g3.kitchen.trapped.get(id(slot), 0.0) < kt.TRAP_IMMORTAL_ESCAPE_S


def test_a_fly_whose_vinegar_neurons_fire_flies_to_the_jar_and_one_whose_do_not_stays(g3):
    slot = g3.flies[0]
    _place(slot.fly, kt.TRAP_POS + (-2.0, 0.38, 0.0))
    g3._alcohol_hz = lambda s: 2.0
    _frames(g3, 6)                                              # calm baseline of 2 Hz, with a poke in range
    assert slot.fly.fly_target is not None
    before = slot.fly.fly_target.copy()
    # in range the smell is poked every frame, so the baseline is only learned out of range: put it there first
    _place(slot.fly, np.array([-3.0, 0.38, 3.0]))
    _frames(g3, 6)
    tr = g3.kitchen.tracker(id(slot))
    assert tr.base == pytest.approx(2.0, abs=0.3) and not tr.smelling
    _place(slot.fly, kt.TRAP_POS + (-2.0, 0.38, 0.0))
    g3._alcohol_hz = lambda s: 2.0
    _frames(g3, 3)
    assert np.allclose(slot.fly.fly_target, before) or slot.fly.escape_until <= g3.clock.now, "calm neurons: no trip"
    g3._alcohol_hz = lambda s: 30.0
    _frames(g3, 2)
    assert np.allclose(slot.fly.fly_target, kt.mouth() + (0.0, 0.15, 0.0)), "the neurons fire: it flies to the mouth"
    assert slot.fly.perch == "vinegar" and np.allclose(slot.trap_trip, slot.fly.fly_target), "and it holds over the mouth, not wandering off"
    slot.fly.fly_target = slot.fly.fly_target + 5.0                             # its escape neurons sent it elsewhere: the hold is let go
    _frames(g3, 0.1)
    assert slot.fly.perch is None and slot.trap_trip is None


def test_the_sink_floats_a_fly_and_wets_its_wings(g3):
    slot = g3.flies[0]
    pokes = _record(slot)
    _place(slot.fly, np.array([3.0, 0.2, 2.3]))
    _frames(g3, 1.0)
    assert slot.fly.water_mask().any(), "its body is under the basin's water"
    assert slot.fly.wet > 0 and {"humid", "body"} <= {p[0] for p in pokes}, "wet wings, and the humidity neurons"
    pokes.clear()
    _place(slot.fly, np.array([0.0, 0.2, 0.0]))                 # the same place on the dry counter: no water, no humidity
    slot.fly.wet = 0.0
    _frames(g3, 1.0)
    assert not slot.fly.water_mask().any() and "humid" not in {p[0] for p in pokes}


def test_the_burner_drives_the_heat_sensors_and_burns_what_touches_it(g3):
    slot = g3.flies[0]
    pokes = _record(slot)
    _place(slot.fly, kt.BURNER_POS + (0.5, 0.38, 0.0))
    _frames(g3, 1.0)
    assert "heat" in {p[0] for p in pokes}
    assert slot.pending_damage == 0, "near, not touching: no damage"
    _place(slot.fly, kt.BURNER_POS + (0.0, 0.1, 0.0))
    _frames(g3, 1.0)
    assert slot.pending_damage > 0 and "burner" in slot.damage_src


def test_the_cooks_swat_hits_a_fly_that_stays_and_misses_one_that_moved(g3):
    slot = g3.flies[0]
    _place(slot.fly, np.array([0.0, 0.38, -1.0]))
    pokes = _record(slot)
    k = g3.kitchen
    k.cook.wait = 0.0
    seen_threat = False
    for i in range(int(4 * 60)):
        g3.frame += 1
        g3._environment(g3.clock.now + (i + 1) / 60)
        seen_threat |= bool([t for t in g3._threats(slot, 0.0) if t[0] == ("cook", "swatter")])
    assert seen_threat, "the swatter is one of the objects the fly's eyes measure"
    assert slot.pending_damage >= kt.COOK_DAMAGE and "swatter" in slot.damage_src
    assert slot.pending_hits, "the hit is queued on the fly's touch neurons, as every tool hit is"
    slot.pending_damage = 0.0
    k.cook.wait = 0.0
    k.cook.state, k.cook.t = "idle", 0.0
    for i in range(int(0.9 * 60)):
        g3.frame += 1
        g3._environment(g3.clock.now + (i + 1) / 60)
        if k.cook.state == "swat":
            break
    _place(slot.fly, np.array([-3.0, 0.38, 2.5]))              # it moved away between the aim and the impact
    for i in range(int(2 * 60)):
        g3.frame += 1
        g3._environment(g3.clock.now + (i + 1) / 60)
    assert slot.pending_damage < kt.COOK_DAMAGE


def test_the_kitchen_draws_without_error_and_the_hud_describes_it(g3):
    class Rec:
        n = 0

        def add(self, *a, **k):
            self.n += 1

        def particle(self, *a, **k):
            self.n += 1

    rd = Rec()
    g3.draw_world(rd, 1.0)
    assert rd.n > 60
    assert any("fruit" in s for s in g3.arena_status()) and any("cook swats" in s for s in g3.arena_status())


def test_the_fruit_bowl_state_is_saved_and_restored(g3, tmp_path):
    from kickthefly.core import savestate

    f = g3.orchard.nearest_ripe(kt.BOWL_POS)
    g3.orchard.feed(f, g3.clock.now)
    left = [x.feeds_left for x in g3.orchard.fruit]
    p = tmp_path / "k.ktfsave"
    savestate.save_game(g3, p)
    g3.set_setting("brain.arena", "room", save=False)
    savestate.load_game(g3, p)
    from kickthefly.game import kick_the_fly as k2

    assert k2.ARENAS[g3.arena_i] == "kitchen" and g3.kitchen is not None
    assert [x.feeds_left for x in g3.orchard.fruit] == left


def test_a_fly_that_is_gone_leaves_nothing_in_the_traps_books(g3):
    slot = g3.flies[0]
    _place(slot.fly, kt.TRAP_POS + (0.0, kt.TRAP_H + 0.15, 0.0))
    _frames(g3, 1.0)
    assert id(slot) in g3.kitchen.trapped and "1 in the trap" in " ".join(g3.arena_status())
    g3.flies.clear()                                              # as R (reset to one fresh fly) does
    g3.flies.append(type(slot)(slot.fly, slot.brain) if False else slot)
    g3.flies.clear()
    _frames(g3, 0.1)
    assert g3.kitchen.trapped == {} and g3.kitchen.trackers == {} and g3.kitchen.hover == {}
