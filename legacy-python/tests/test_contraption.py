"""3.1.0 task 13, the contraption builder (game/contraption.py, lab/contraption_challenge.py): the build rules, the machine's physics and wiring, share codes and save
slots, and (on the real brain, in the 3D game and in --2d) that a running contraption fires the game's own tools at the fly."""
from __future__ import annotations

import math

import pygame
import pytest

from conftest import needs_pack
from kickthefly.core import sharecode
from kickthefly.game import contraption as ct

P = ct._P


def B(*parts, name="t"):
    return ct.Build(name, list(parts))


# --- the rules -----------------------------------------------------------------------------------------------------------------------------
def test_a_build_respects_the_part_marble_and_fly_limits():
    b = ct.Build()
    for i in range(ct.MAX_MARBLES):
        b.add(ct.new_part("marble", i * 10, 100))
    with pytest.raises(ct.BuildError, match="marbles"):
        b.add(ct.new_part("marble", 0, 100))
    for i in range(ct.MAX_PARTS - ct.MAX_MARBLES):
        b.add(ct.new_part("ramp", 0, 50 + i))
    with pytest.raises(ct.BuildError, match="at most 24"):
        b.add(ct.new_part("domino", 0, 0))
    b.add(ct.new_part("fly", 10, 0))
    b.add(ct.new_part("fly", 90, 0))
    assert b.count("fly") == 1 and b.fly_mark().x == 90, "the fly mark is one part: placing it again moves it (and it is not a slot taken)"


def test_new_parts_snap_to_the_grid_and_stay_on_the_bench():
    p = ct.new_part("ramp", 123.0, 87.0)
    assert (p.x, p.y) == (125, 85)
    p = ct.new_part("ramp", 9999, 9999)
    assert p.x <= 230 and p.y <= 300
    p = ct.new_part("domino", -9999, -50)
    assert p.x >= -230 and p.y == 0
    with pytest.raises(ct.BuildError):
        ct.new_part("trebuchet", 0, 0)


def test_editing_a_field_stays_in_its_range_and_wraps_the_fan_angle():
    r = ct.defaults("ramp")
    for _ in range(40):
        ct.step_field(r, "a", 1)
    assert r.a == 60
    for _ in range(40):
        ct.step_field(r, "n", -1)
    assert r.n == 20 and not ct.step_field(r, "n", -1)
    f = ct.defaults("fan")
    ct.step_field(f, "a", -1)
    assert f.a == 345
    t = ct.defaults("tool")
    seen = []
    for _ in range(len(ct.TOOLS)):
        ct.step_field(t, "tool", 1)
        seen.append(t.tool)
    assert set(seen) == set(ct.TOOLS) and t.tool == "swatter"
    assert not ct.step_field(ct.defaults("sugar"), "n", 1), "a field the part does not have changes nothing"


def test_check_says_what_is_wrong_with_a_build():
    ok = {"name": "x", "parts": [{"k": "marble", "x": 0, "y": 100}]}
    assert ct.check(ok) == []
    assert ct.check([]) and ct.check({"parts": []}) and ct.check({"name": "x"})
    assert "unknown fields" in ct.check({**ok, "owner": 1})[0]
    assert "not a part" in ct.check({"parts": [{"k": "rocket"}]})[0]
    assert "whole number" in ct.check({"parts": [{"k": "marble", "x": 1.5, "y": 0}]})[0]
    assert "whole number" in ct.check({"parts": [{"k": "marble", "x": True, "y": 0}]})[0]
    assert "off the bench" in ct.check({"parts": [{"k": "marble", "x": 999, "y": 0}]})[0]
    assert "from 20 to 200" in ct.check({"parts": [{"k": "ramp", "x": 0, "y": 0, "n": 500}]})[0]
    assert "not a tool" in ct.check({"parts": [{"k": "tool", "x": 0, "y": 0, "tool": "laser"}]})[0]
    assert "unknown fields" in ct.check({"parts": [{"k": "marble", "x": 0, "y": 0, "evil": 1}]})[0]
    assert ct.check({"parts": [{"k": "marble", "x": 0, "y": 0}] * 4})
    assert ct.check({"parts": [{"k": "fly", "x": 0, "y": 0}] * 2})
    assert ct.check({"name": "n" * 30, "parts": [{"k": "marble", "x": 0, "y": 0}]})
    no_zapper = lambda t: t != "zapper"                                          # noqa: E731
    assert ct.check({"parts": [{"k": "tool", "x": 0, "y": 0, "tool": "zapper"}]}, no_zapper)
    assert ct.check({"parts": [{"k": "tool", "x": 0, "y": 0, "tool": "swatter"}]}, no_zapper) == []


def test_a_build_survives_json_without_its_defaults():
    b = ct.EXAMPLES["Domino swat"]
    d = b.to_json()
    assert all("ch" not in r for r in d["parts"] if r["k"] == "marble"), "defaults are left out so a code stays small"
    back = ct.Build.from_json(d)
    assert back.to_json() == d and [p.k for p in back.parts] == [p.k for p in b.parts]
    with pytest.raises(ct.BuildError):
        ct.Build.from_json({"parts": [{"k": "ramp", "x": 0, "y": 0, "n": 9999}]})


# --- the machine --------------------------------------------------------------------------------------------------------------------------
def test_the_machine_is_deterministic():
    a, _ = ct.simulate(ct.EXAMPLES["Domino swat"])
    b, _ = ct.simulate(ct.EXAMPLES["Domino swat"])
    assert [(e.t, e.kind, e.tool, e.x, e.y) for e in a] == [(e.t, e.kind, e.tool, e.x, e.y) for e in b]


def test_a_marble_rolls_down_a_ramp_and_comes_to_rest_on_the_floor():
    ev, m = ct.simulate(B(P("marble", -200, 100), P("ramp", -205, 60, a=-10, n=120)))
    b = m.balls[0]
    assert b.y == pytest.approx(ct.BALL_R, abs=1e-6) and b.x > -0.5 and m.finished and m.t < ct.MAX_RUN_S
    assert not [e for e in ev if e.kind == "tool"]


def test_a_ramp_is_one_sided_and_a_marble_off_every_part_falls_to_the_floor():
    _, m = ct.simulate(B(P("marble", 0, 150)))
    assert m.balls[0].x == pytest.approx(0.0, abs=1e-6) and m.balls[0].y == pytest.approx(ct.BALL_R, abs=1e-6)
    _, m2 = ct.simulate(B(P("marble", 0, 20), P("ramp", -50, 60, a=0, n=100)))     # under the ramp: it never goes through to the top
    assert m2.balls[0].y < 0.2


def test_a_spring_throws_a_marble_up_and_sideways():
    m = ct.Machine(B(P("marble", 0, 100), P("spring", 0, 0, a=30)))
    top, x_after = 0.0, 0.0
    for _ in range(int(2.5 / ct.DT)):
        m.step()
        top = max(top, m.balls[0].y) if m.balls and m.t > 0.2 else top
    assert top > 0.5, "launched up (it fell from 1.0 m, then rose again)"
    assert m.balls[0].x > 0.5, "and to the side the angle says"
    assert 1 in m.fired or 1 in {i for i in m.fired}


def test_a_fan_pushes_a_marble_only_while_it_is_on():
    fan = B(P("marble", -100, 10), P("fan", -200, 4, a=0, n=200, ch=0))
    _, m = ct.simulate(fan, seconds=1.5)
    assert m.balls[0].x > -1.0, "the fan on from the start pushes the marble along the floor"
    off = B(P("marble", -100, 10), P("fan", -200, 4, a=0, n=200, ch=3))
    _, m2 = ct.simulate(off, seconds=1.5)
    assert m2.balls[0].x == pytest.approx(-1.0, abs=0.02), "a fan on a channel nothing sends stays off"


def test_a_domino_chain_falls_in_order_and_presses_the_button_once():
    b = B(P("marble", -200, 60), P("ramp", -205, 30, a=-10, n=100), P("domino", -80, 0, n=30), P("domino", -55, 0, n=30), P("domino", -30, 0, n=30),
          P("button", 0, 0, ch=2), P("tool", 100, 30, ch=2, tool="bomb"))
    ev, m = ct.simulate(b)
    presses = [e for e in ev if e.kind == "fire" and e.ch == 2]
    assert len(presses) == 1 and presses[0].by == "domino"
    assert [d.state for d in m.dominoes] == ["fallen"] * 3
    tips = sorted(d.t0 for d in m.dominoes)
    assert tips == [d.t0 for d in m.dominoes], "left to right, each after the one before"
    shots = [e for e in ev if e.kind == "tool"]
    assert len(shots) == 1 and shots[0].tool == "bomb" and shots[0].t == pytest.approx(presses[0].t, abs=0.01)


def test_a_domino_far_from_the_next_does_not_reach_it():
    _, m = ct.simulate(B(P("marble", -200, 30), P("ramp", -205, 20, a=-10, n=100), P("domino", -80, 0, n=20), P("domino", 100, 0, n=20)))
    assert [d.state for d in m.dominoes] == ["fallen", "standing"]


def test_timers_ring_at_their_times_and_repeat():
    ev, m = ct.simulate(B(P("timer", 0, 150, t=10, ev=5, c=3, ch=2), P("tool", 0, 30, ch=2, tool="cva")))
    rings = [e.t for e in ev if e.kind == "fire" and e.ch == 2]
    assert rings == pytest.approx([1.0, 1.5, 2.0], abs=0.01)
    assert [e.tool for e in ev if e.kind == "tool"] == ["cva"] * 3


def test_a_tool_part_bursts_and_channel_zero_goes_off_at_the_start():
    ev, _ = ct.simulate(B(P("tool", 50, 30, ch=0, tool="flick", c=3), P("sugar", -50, 30, ch=0)))
    tools = [(round(e.t, 2), e.tool) for e in ev if e.kind == "tool"]
    assert tools[0][0] == 0.0 and sorted(tools) == [(0.0, "flick"), (0.0, "sugar"), (0.3, "flick"), (0.6, "flick")]


def test_nothing_is_wired_to_a_channel_nobody_sends():
    ev, m = ct.simulate(B(P("tool", 50, 30, ch=4, tool="bomb"), P("timer", 0, 150, t=10, ch=2)))
    assert not [e for e in ev if e.kind == "tool"] and m.finished


def test_fans_and_lamps_say_what_they_do_to_a_point_and_stop_after_their_time():
    m = ct.Machine(B(P("fan", -100, 20, a=0, n=150, ch=0), P("lamp", -100, 120, ch=0, t=20)))
    m.step()
    assert m.wind_on(0.0, 0.2) > 0.1 and m.wind_on(-1.5, 0.2) == 0.0, "behind the fan: nothing"
    assert m.wind_on(0.0, 0.2) > m.wind_on(0.4, 0.2), "weaker with distance"
    assert m.wind_on(0.0, 0.2, z=2.0) == 0.0, "a fly far off the bench's plane feels nothing"
    assert m.wind_on(0.0, 2.0) == 0.0, "above the stream"
    assert m.light_on(-1.0, 1.2) > m.light_on(1.0, 0.2) > 0.0
    for _ in range(int(2.2 / ct.DT)):
        m.step()
    assert m.light_on(-1.0, 1.2) == 0.0 and m.wind_on(0.0, 0.2) > 0.1, "the lamp is lit for 2 s, the fan for 4"
    for _ in range(int(2.2 / ct.DT)):
        m.step()
    assert m.wind_on(0.0, 0.2) == 0.0


def test_a_marble_that_meets_the_fly_flicks_it():
    fly = (0.4, 0.05, 0.0, 0.16)
    ev, _ = ct.simulate(B(P("marble", -200, 60), P("ramp", -205, 30, a=-10, n=100)), fly=fly)
    hits = [e for e in ev if e.kind == "hit"]
    assert len(hits) >= 1 and hits[0].by == "marble"
    assert any(e.kind == "tool" and e.tool == "flick" for e in ev)
    ev2, _ = ct.simulate(B(P("marble", -200, 60), P("ramp", -205, 30, a=-10, n=100)), fly=(0.4, 0.05, 1.5, 0.16))
    assert not [e for e in ev2 if e.kind == "hit"], "a fly off the bench's plane is not in the way"


def test_a_falling_domino_hits_a_fly_in_reach():
    ev, _ = ct.simulate(B(P("marble", -200, 30), P("ramp", -205, 20, a=-10, n=100), P("domino", -80, 0, n=30)), fly=(-0.55, 0.1, 0.0, 0.16))
    assert any(e.kind == "hit" and e.by == "domino" for e in ev)


def test_every_run_ends_within_the_limit():
    ev, m = ct.simulate(B(P("timer", 0, 150, t=600, ch=1), P("lamp", 0, 100, ch=1)))
    assert m.finished and m.t <= ct.MAX_RUN_S + 1.0


def test_the_examples_do_what_their_names_say():
    ev, _ = ct.simulate(ct.EXAMPLES["Domino swat"])
    assert [e.tool for e in ev if e.kind == "tool"] == ["swatter"] and any(e.kind == "fire" and e.by == "domino" for e in ev)
    ev, _ = ct.simulate(ct.EXAMPLES["Fan and lamp"])
    assert {e.tool for e in ev if e.kind == "on"} == {"fan", "lamp"} and [round(e.t) for e in ev if e.kind == "on"] == [2, 4]
    ev, _ = ct.simulate(ct.EXAMPLES["Bait and bomb"])
    tools = [(e.tool, round(e.t, 1)) for e in ev if e.kind == "tool"]
    assert tools[0] == ("sugar", 0.0) and tools[1][0] == "bomb" and 1.0 < tools[1][1] < 4.0


# --- share codes and slots ---------------------------------------------------------------------------------------------------------------------
def test_a_build_is_a_share_code_and_back():
    b = ct.EXAMPLES["Bait and bomb"]
    code = b.code()
    assert code.startswith("KTF1-CTN-") and len(code) <= sharecode.MAX_CODE_CHARS
    back = ct.Build.from_code(code)
    assert back.to_json() == b.to_json()
    assert ct.Build.from_code(" ".join(code[i:i + 9] for i in range(0, len(code), 9))).to_json() == b.to_json(), "spaces and breaks are ignored"
    with pytest.raises(ct.BuildError, match="nothing to share"):
        ct.Build().code()


def test_codes_that_are_not_a_contraption_or_are_damaged_are_refused():
    with pytest.raises(ct.BuildError, match="surgery"):
        ct.Build.from_code(sharecode.encode("surgery", {"types": {"LC4": 1}}))
    code = ct.EXAMPLES["Domino swat"].code()
    with pytest.raises(ct.BuildError):
        ct.Build.from_code(code[:-2] + ("00" if not code.endswith("00") else "11"))
    with pytest.raises(ct.BuildError):
        ct.Build.from_code("hello")
    with pytest.raises(sharecode.ShareError):
        sharecode.encode("contraption", {"parts": [{"k": "rocket"}]})


def test_the_share_machinery_validates_previews_and_saves_a_contraption(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "saved_path", lambda: tmp_path / "contraptions.json")
    code = sharecode.decode(ct.EXAMPLES["Domino swat"].code())
    ctx = sharecode.Context(tools={"swatter", "bomb"})
    assert sharecode.validate(code, ctx) is None
    assert sharecode.validate(code, sharecode.Context(tools={"bomb"})) is not None, "a tool this install lacks is refused"
    pv = sharecode.preview(code, ctx)
    assert "swatter" in " ".join(pv.lines) and "first free" in " ".join(pv.lines) and not pv.warnings
    assert sharecode.preview(code, sharecode.Context(tools={"swatter"}, contraption_slots_used=8)).warnings
    out = sharecode.apply(code, object())
    assert "slot 1" in out[0] and ct.load_slots()[0] == ct.EXAMPLES["Domino swat"].to_json()
    assert "slot 2" in sharecode.apply(code, object())[0]
    for _ in range(6):
        sharecode.apply(code, object())
    assert "full" in sharecode.apply(code, object())[0], "a full set of slots refuses and overwrites nothing"


def test_existing_kinds_keep_their_bytes():
    assert [sharecode.KIND_BYTE[k] for k in ("surgery", "loadout", "protocol", "challenge", "lab")] == [1, 2, 3, 4, 5]
    assert sharecode.KIND_BYTE["contraption"] == 6


def test_save_slots_roundtrip_and_ignore_damage(tmp_path, monkeypatch):
    path = tmp_path / "contraptions.json"
    monkeypatch.setattr(ct, "saved_path", lambda: path)
    assert ct.load_slots() == [None] * 8
    assert ct.save_slot(2, ct.EXAMPLES["Fan and lamp"]) and not ct.save_slot(9, ct.EXAMPLES["Fan and lamp"]) and not ct.save_slot(0, ct.Build())
    assert ct.load_slots()[2]["name"] == "Fan and lamp"
    path.write_text("{oops", encoding="utf-8")
    assert ct.load_slots() == [None] * 8
    path.write_text('[null, {"parts": [{"k": "rocket"}]}, {"name": "ok", "parts": [{"k": "marble", "x": 0, "y": 100}]}]', encoding="utf-8")
    got = ct.load_slots()
    assert got[0] is None and got[1] is None and got[2]["name"] == "ok"


# --- in the game (the real brain) --------------------------------------------------------------------------------------------------------------------
def _rig(three_d):
    from kickthefly.lab.playthrough import Rig

    return Rig(three_d, "cpu", seed=5)


def _press(ch, label):
    for b in ch.buttons:
        if b.label == label and b.enabled:
            b.action()
            return True
    return False


def _draw(ch, g, mouse=(0, 0)):
    surf = pygame.Surface((1280, 760))
    ch.draw(surf, g.clock.now, mouse)
    return surf


@needs_pack
@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_the_editor_places_selects_moves_edits_and_deletes(three_d):
    r = _rig(three_d)
    try:
        g = r.game
        g.start_challenge("contraption")
        ch = g.challenge
        assert ch.key == "contraption" and ch.overlay
        _draw(ch, g)
        assert _press(ch, "Ramp") and ch.pick == "ramp"
        c = ch.canvas
        ch.click((c.centerx, c.bottom - 120))
        assert ch.build.count("ramp") == 1 and ch.sel == 0
        _draw(ch, g)
        assert _press(ch, "+"), "the selected part's fields have buttons"
        _draw(ch, g)
        _press(ch, "Select / move")
        ch.click((c.centerx + 150, c.bottom - 60))
        p = ch.build.parts[0]
        assert p.x > 50, "clicking an empty place moves the selected part there"
        ch.click((c.centerx + int(p.x / 100 * c.w / 4.8), c.bottom - int(p.y / 100 * c.w / 4.8)))
        assert ch.sel == 0
        _draw(ch, g)
        assert _press(ch, "Delete part") and not ch.build.parts
        ch.pick = "marble"
        for i in range(4):
            ch.click((c.x + 40 + 30 * i, c.y + 60))
        assert ch.build.count("marble") == 3 and ch.msg_bad and "marbles" in ch.msg
        assert surf_has_pixels(_draw(ch, g))
        assert not ch.build.problems()
        ch.clear()
        assert not ch.build.parts
    finally:
        r.close()


def surf_has_pixels(surf):
    return surf.get_bounding_rect().w > 100


@needs_pack
@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_run_refuses_an_empty_build_and_an_unavailable_tool(three_d):
    r = _rig(three_d)
    try:
        g = r.game
        g.start_challenge("contraption")
        ch = g.challenge
        ch.begin_run()
        assert ch.state == "build" and ch.msg_bad
        ch.build = ct.Build("x", [P("fly", 100, 0)])
        ch.begin_run()
        assert ch.state == "build" and "besides the fly" in ch.msg
        ch.build = ct.Build("x", [P("tool", 0, 30, ch=0, tool="swatter")])
        g.cfg.set("brain.mode", "play")
        ch._available = lambda t: t != "swatter"
        ch.begin_run()
        assert ch.state == "build" and "not available" in ch.msg
    finally:
        r.close()


@needs_pack
@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_a_running_contraption_swats_the_fly_through_the_games_own_tool(three_d):
    r = _rig(three_d)
    try:
        g = r.game
        r.seconds(2.0)
        g.start_challenge("contraption")
        ch = g.challenge
        ch.load_example("Domino swat")
        ch.begin_run()
        assert ch.state == "run" and not ch.overlay
        for _ in range(60 * 14):
            r.frames(1)
            if ch.state != "run":
                break
        assert ch.state == "report" and ch.overlay
        rep = ch.report
        kinds = [line for _, line in rep["fired"]]
        assert any("swatter" in k for k in kinds) and any("domino" in k for k in kinds)
        t_swat = next(t for t, k in rep["fired"] if "swatter" in k)
        assert 2.5 < t_swat < 3.5
        react = [(t, n) for t, n in rep["reactions"] if t > t_swat]
        assert react and react[0][0] - t_swat < 1.0 and react[0][1] in ("FLY AWAY", "TAKE OFF", "DODGE", "KICK", "RUN"), react
        assert not rep["died"]
        assert surf_has_pixels(_draw(ch, g))
    finally:
        r.close()


@needs_pack
@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_the_tether_holds_the_fly_and_without_it_the_fly_is_free(three_d):
    from kickthefly.lab import contraption_challenge as cc

    r = _rig(three_d)
    try:
        g = r.game
        g.start_challenge("contraption")
        ch = g.challenge
        for tethered, held in ((1, True), (0, False)):
            ch.build = ct.Build("x", [P("fly", 100, 0, c=tethered), P("timer", 0, 150, t=500, ch=1)])
            ch.begin_run()
            x0 = cc.fly_body(g, ch.slot)[0]
            cc.fire_tool(g, "bomb", x0 - 0.3, 0.1, g.clock.now)                  # something that throws it about
            r.seconds(5.0)
            x1, y1, z1, _ = cc.fly_body(g, ch.slot)
            moved = abs(x1 - x0) + abs(z1)
            assert (moved < 0.06) == held, (tethered, moved)
            g.bombs.clear() if hasattr(g, "bombs") else None
            ch.stop_run()
    finally:
        r.close()


@needs_pack
def test_a_fan_and_a_lamp_drive_the_wind_and_light_neurons():
    r = _rig(False)
    try:
        g = r.game
        r.seconds(2.0)
        g.start_challenge("contraption")
        ch = g.challenge
        ch.build = ct.Build("x", [P("fan", 60, 30, a=0, n=200, ch=0), P("fly", 100, 0, a=0)])
        calls = []
        br = ch.slot.brain
        real = br.poke
        br.poke = lambda region, side, s, recruit=None: (calls.append((region, round(float(s), 2))), real(region, side, s, recruit))[1]
        ch.begin_run()
        ch.slot.brain.poke = br.poke
        for _ in range(60):
            r.frames(1)
        assert any(c[0] == "wind" and c[1] > 0.2 for c in calls), calls[:5]
        assert not any(c[0] == "light" for c in calls)
        ch.stop_run()
        ch.build = ct.Build("x", [P("lamp", 60, 80, ch=0, t=50), P("fly", 100, 0)])
        calls.clear()
        ch.begin_run()
        for _ in range(60):
            r.frames(1)
        assert any(c[0] == "light" for c in calls) and not any(c[0] == "wind" for c in calls)
    finally:
        r.close()


@needs_pack
def test_the_challenge_is_listed_and_lazy_loaded():
    from kickthefly.lab import challenges
    from kickthefly.lab import contraption_challenge as cc

    assert "contraption" in [k for k, *_ in challenges.INFO] and "contraption" in challenges.CLASSES and challenges.CLASSES["contraption"] is cc.Contraption
    assert challenges.stars("contraption", 7) == 0


@needs_pack
@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_ending_the_challenge_gives_the_games_aim_back(three_d):
    r = _rig(three_d)
    try:
        g = r.game
        g.start_challenge("contraption")
        ch = g.challenge
        ch.load_example("Domino swat")
        ch.begin_run()
        from kickthefly.lab import contraption_challenge as cc

        cc.fire_tool(g, "swatter", 0.5, 0.4, g.clock.now)
        if three_d:
            assert "aim" in g.__dict__
        ch.end()
        assert "aim" not in g.__dict__ and "tool_tip" not in g.__dict__ and g.challenge is None
        assert ch._on_note not in g.note_hooks
    finally:
        r.close()
