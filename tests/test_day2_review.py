"""3.0 day 2 review (Opus): regression tests for the bugs found by attacking Sonnet's day 2 toolkit. Each test failed on
sonnet/3.0-day2 (4745422). Synthetic pack where a brain is needed: plumbing, not biology."""
from __future__ import annotations

import time

import numpy as np
import pygame
import pytest

from kickthefly.game import kick_the_fly as k2
from test_extras3 import _free_games, make_game  # noqa: F401  (the autouse fixture that stops the games' threads)


@pytest.fixture
def game(synthetic_pack):
    from kickthefly.sim import wiring as W

    g = make_game(mode="lab")
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    yield g
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


@pytest.fixture
def fly(synthetic_pack):
    from kickthefly import Fly
    from kickthefly.sim import wiring as W

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    yield Fly(seed=3, warmup_s=0.2, learn=False)
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


def _draw(g, page):
    if g.menu.screen != page:
        g.menu.show(page)
    surf = pygame.Surface((k2.W, k2.H))
    g.menu.mouse = (5, 5)
    g.menu.draw(surf, (5, 5), time.perf_counter())
    return surf


def _stimulus_protocol(**extra):
    return dict(name="t", seed=1000, flies=1, warmup_s=0.1, duration_s=0.2, stimuli=[], recordings=[], **extra)


# --- Brain.currents: the game thread must never zero the array the brain thread is reading --------------------------------------------
def test_setting_a_current_swaps_in_a_new_array_and_never_zeroes_the_one_a_step_may_be_reading(game):
    br = game.brain
    br.set_current("a", [0, 1], 0.5)
    seen = br.inject
    snapshot = seen.copy()
    br.set_current("b", [2], 0.25)                 # the game thread (Lab tick) while the brain thread may be mid-step
    br.clear_current("a")
    assert np.array_equal(seen, snapshot), "an array a running step held was changed in place (it read a half-built sum)"
    assert br.inject[2] == pytest.approx(0.25) and br.inject[0] == 0 and br.injecting
    br.clear_current("b")
    assert not br.injecting and not np.any(br.inject)


def test_a_probe_that_raises_is_removed_and_does_not_kill_the_brain(game):
    br = game.brain

    def bad(_b):
        raise IndexError("row 999999 is out of range")

    br.probe = bad
    br._step()                                     # must not raise: on the brain thread an exception ends the loop for good
    assert br.probe is None


# --- live thermogenetics ------------------------------------------------------------------------------------------------------------
def test_live_thermo_state_is_tied_to_the_brain_object_not_its_id(game):
    """A respawned fly or a loaded save gets a new Brain; CPython can give it the dead brain's id(). The old per-brain state
    (its 'already written' current) must not be reused, or the new brain silently gets no current."""
    from types import SimpleNamespace

    tl = game.thermo_live
    tl.kinetics = "steady"
    tl.temperature_c = 34.0
    tl.add("trpa1", "type:DNp01")
    from kickthefly.core import simcore

    a = game.brain
    tl.tick([SimpleNamespace(brain=a)])
    assert a.injecting
    b = simcore.new_brain(seed=7)
    for key in list(tl.per_brain):                 # what id() reuse looks like: the new brain finds the old brain's entry
        tl.per_brain[id(b)] = tl.per_brain.pop(key)
    tl.tick([SimpleNamespace(brain=b)])
    assert b.injecting, "the new brain never got the TrpA1 current"
    assert len(tl.per_brain) == 1


def test_live_thermo_forgets_brains_that_left_the_game(game):
    from types import SimpleNamespace

    tl = game.thermo_live
    tl.kinetics = "steady"
    tl.add("trpa1", "type:DNp01")
    tl.tick([SimpleNamespace(brain=game.brain)])
    tl.tick([])                                    # the fly died or was removed
    assert not tl.per_brain


def test_a_bad_thermo_target_is_refused_and_the_good_expressions_stay(game):
    from kickthefly.lab import labtoolkit

    tl = game.thermo_live
    tl.kinetics = "steady"
    tl.temperature_c = 34.0
    tl.add("trpa1", "type:DNp01")
    st = labtoolkit._st(game.menu)
    st.t_target, st.t_effector = "type:NoSuchType", 0
    labtoolkit._add_expr(st, tl, game)             # the page's Add button
    assert [e["target"] for e in tl.expressions] == ["type:DNp01"]
    assert "NoSuchType" in st.note
    tl.expressions.append(dict(effector="trpa1", target="type:AlsoMissing", strength=1.0))   # e.g. a brain with other types
    for _ in range(6):
        game.frame += 1
        game._lab_tick()
    assert [e["target"] for e in tl.expressions] == ["type:DNp01"] and game.brain.injecting


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), -1.0, 4.0])
def test_thermo_strength_must_be_a_finite_number_in_range(strength):
    from kickthefly.lab import thermogenetics as tg

    with pytest.raises(tg.ThermoError):
        tg.Expression("trpa1", "type:DNp01", strength)
    with pytest.raises(tg.ThermoError):
        tg.check_spec(dict(expression=[dict(effector="trpa1", target="type:DNp01", strength=strength)]))


# --- protocols: only ProtocolError comes out of check(), whatever the file says ------------------------------------------------------
@pytest.mark.parametrize("block", [
    dict(thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")], temperature_c=[dict(at_s="x", c=30)])),
    dict(thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")], temperature_c=[dict(at_s=0, c=None)])),
    dict(thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")], temperature_c=[dict(at_s=float("nan"), c=30)])),
    dict(thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")] * 1000)),
    dict(thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")],
                             temperature_c=[dict(at_s=i, c=30) for i in range(100_000)])),
    dict(thermogenetics=dict(expression=[dict(effector="trpa1", target=["type:DNp01"])])),
    dict(imaging=dict(f0_photons=-5)),
    dict(imaging=dict(f0_photons="many")),
    dict(imaging=dict(f0_photons=float("inf"))),
    dict(imaging=dict(dff_per_spike=float("nan"))),
    dict(imaging=dict(shot_noise="no")),
    dict(imaging=dict(tiff="yes")),
    dict(imaging=dict(rois=["type:DNp01"] * 5000)),
])
def test_hostile_day2_protocol_blocks_raise_protocol_error(block):
    from kickthefly.lab import protocol

    with pytest.raises(protocol.ProtocolError):
        protocol.check(_stimulus_protocol(**block))


@pytest.mark.parametrize("pa", [
    dict(neuron="type:DNp01", amplitudes=[0.0, float("nan")]),
    dict(neuron="type:DNp01", amplitudes=[float("nan")]),
    dict(neuron=["type:DNp01"]),
    dict(neuron="type:DNp01", amplitudes="0.1"),
])
def test_hostile_patch_protocols_raise_protocol_error(pa):
    from kickthefly.lab import protocol

    with pytest.raises(protocol.ProtocolError):
        protocol.check(dict(name="p", seed=1000, patch=pa))


@pytest.mark.parametrize("extra", [dict(surgery={"type:DNp01": -1}), dict(recordings=[dict(name="gf", neurons="type:DNp01")]),
                                   dict(control=True)])
def test_a_patch_protocol_refuses_keys_it_would_silently_ignore(extra):
    from kickthefly.lab import protocol

    with pytest.raises(protocol.ProtocolError, match="patch"):
        protocol.check(dict(name="p", seed=1000, patch=dict(neuron="type:DNp01"), **extra))


def test_patch_protocol_amplitudes_written_as_strings_are_numbers_after_check(synthetic_pack, tmp_path):
    """'0.1' passed ClampProtocol's float(), then the summary looked the string up among float keys after the whole run."""
    from kickthefly.lab import protocol

    p = protocol.check(dict(name="p", seed=1000, patch=dict(neuron="type:MDN", amplitudes=["0", "0.2"], duration_ms=50, repeats=1,
                                                             warmup_s=0.05)))
    assert p["patch"]["amplitudes"] == [0.0, 0.2]
    folder = protocol.run(p, tmp_path)
    assert (folder / "summary.json").exists()


@pytest.mark.parametrize("opts", [dict(brain=1), dict(temps=[]), dict(temps=[22, "hot"]), dict(temps=[5.0]), dict(effector_name="gfp"),
                                  dict(target=3), dict(temps=list(range(10, 45)) * 10)])
def test_thermo_escape_assay_options_are_checked(opts):
    from kickthefly.lab import protocol

    with pytest.raises(protocol.ProtocolError):
        protocol.check(dict(name="a", seed=1000, flies=2, assay="thermo_escape", assay_options=opts))


def test_a_classroom_protocol_refuses_day2_blocks_too():
    from kickthefly.lab import protocol

    with pytest.raises(protocol.ProtocolError, match="drug"):
        protocol.check(dict(name="c", classroom=True, steps=[], drug=dict(doses={"picrotoxin": 0.5})))


# --- patch clamp ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [dict(amplitudes=[float("nan")]), dict(amplitudes=[0.1], holding=float("nan")),
                                dict(amplitudes=[float("-inf")])])
def test_clamp_protocol_refuses_non_finite_currents(kw):
    from kickthefly.lab import patchclamp as pc

    with pytest.raises(pc.PatchError):
        pc.ClampProtocol(**kw)


def test_a_nan_current_never_reaches_the_brain(fly):
    from kickthefly.lab import patchclamp as pc

    row = int(fly.neurons("type:MDN")[0])
    with pytest.raises(pc.PatchError):
        fly.patch(row, [0.0, float("nan")], duration_ms=50, repeats=1)
    ele = pc.LiveElectrode(fly.brain, row).attach()
    with pytest.raises(pc.PatchError):
        ele.set_hold(float("nan"))
    ele.detach()
    fly.step(0.1)
    assert np.isfinite(fly.brain.sim.v).all()


def test_starting_a_live_protocol_never_exposes_half_set_state_to_the_brain_thread(fly):
    """run() set the schedule before the sample buffers existed: a brain step landing in between raised AttributeError on the
    brain thread (which ends the brain loop). Simulated here by stepping exactly when the schedule appears."""
    from kickthefly.lab import patchclamp as pc

    br = fly.brain
    row = int(fly.neurons("type:MDN")[0])

    class Racy(pc.LiveElectrode):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "_sched" and value is not None and getattr(self, "attached", False):
                self._push(br)                    # the brain thread steps right now

    ele = Racy(br, row)
    ele.attach()
    ele.attached = True
    ele.run(pc.ClampProtocol([0.2], duration_ms=20, pre_ms=0, post_ms=0, gap_ms=0))
    for _ in range(10):
        br._step()
    assert ele.done and len(ele.recording.v) == 4
    ele.detach()


def test_leaving_the_patch_page_takes_the_electrode_off_the_brain(game):
    from kickthefly.lab import labtoolkit

    _draw(game, "lab_patch")
    st = labtoolkit._st(game.menu)
    st.p_type = "type:MDN"
    labtoolkit._pick(st, game)
    st.p_live, st.p_hold = True, 0.3
    _draw(game, "lab_patch")
    labtoolkit._hold(st, 0.3)
    assert game.brain.probe is not None and game.brain.injecting
    game.menu.back()                               # to the Lab hub; the game will run again once the menu closes
    game.frame += 1
    game._lab_tick()
    assert game.brain.probe is None and not game.brain.injecting and st.p_electrode is None


def test_patch_page_survives_a_row_from_another_brain(game):
    from kickthefly.lab import labtoolkit

    st = labtoolkit._st(game.menu)
    st.p_row = game.brain.n + 10                    # chosen on a bigger brain, then the brain changed
    _draw(game, "lab_patch")
    assert getattr(game.menu, "_page_error", None) is None or game.menu._page_error[0] != "lab_patch"
    assert st.p_row is None and "outside" in st.note


# --- imaging --------------------------------------------------------------------------------------------------------------------------
def test_kept_frames_match_imaging_frames_after_the_buffer_fills(game):
    """recolor() appended a rendered frame whenever len(session.t) > len(frames): once the 300-frame deque was full and the
    session held more, every render was kept, so the TIFF / NWB ImageSeries no longer matched the imaging frame times."""
    il = game.imaging_live
    il.fps, il.keep_frames = 20.0, True
    il.set_on(True, game.brain, game.graph)
    surf = pygame.Surface((8, 8))
    s = il.session
    for i in range(400):                           # 400 imaging frames, one render each
        for _ in range(s.frame_steps):
            s.push(np.zeros(0, np.int64))
        surf.fill((i % 256, 0, 0))
        il.recolor(game.view, surf, "default", False)
    last = il.frames[-1].copy()
    surf.fill((0, 0, 200))
    for _ in range(5):                             # five renders with no new imaging frame: nothing new to keep
        il.recolor(game.view, surf, "default", False)
    assert np.array_equal(il.frames[-1], last)


def test_imaging_frames_of_different_sizes_still_export(game, tmp_path):
    """The view renders at the panel size or the big-view size (B); kept frames of both sizes broke np.stack in the NWB export."""
    from kickthefly.lab import imaging

    il = game.imaging_live
    il.keep_frames = True
    il.set_on(True, game.brain, game.graph)
    s = il.session
    for size in ((20, 10), (20, 10), (40, 30)):
        for _ in range(s.frame_steps):
            s.push(np.zeros(0, np.int64))
        il.recolor(game.view, pygame.Surface(size), "default", False)
    res = il.result()
    from kickthefly.lab import nwbexport

    imaging.export_tiff(res.frames, tmp_path / "v.tif", res.meta)
    if nwbexport.available() is None:
        imaging.export_nwb(res, tmp_path / "v.nwb")
    assert len({f.shape for f in res.frames}) == 1


def test_imaging_keeps_running_after_the_step_count_goes_back(synthetic_pack):
    """A save state replaces the brain's step count. A guard, not a bug fix: this already worked on day 2 (feed_from_activity
    takes the new count before it returns)."""
    from types import SimpleNamespace

    from kickthefly.lab import imaging

    raster = [np.array([0])] * 400
    act = SimpleNamespace(steps=10_000, raster=lambda: raster)
    s = imaging.ImagingSession(4, {"a": np.array([0, 1])}, fps=20, shot_noise=False)
    s.feed_from_activity(act)
    act.steps = 10_100
    s.feed_from_activity(act)
    n = len(s.t)
    act.steps = 500                                # loaded an older save
    s.feed_from_activity(act)
    act.steps = 600
    s.feed_from_activity(act)
    assert len(s.t) > n


def test_a_huge_roi_list_is_refused_before_resolving_every_spec(synthetic_pack, monkeypatch):
    from kickthefly.core import simcore
    from kickthefly.lab import imaging

    br = simcore.new_brain(seed=1)
    calls = []
    real = simcore.rows_of
    monkeypatch.setattr(simcore, "rows_of", lambda b, s: calls.append(s) or real(b, s))
    with pytest.raises(imaging.ImagingError):
        imaging.rois_from_specs(br, ["type:DNp01"] * 100_000)
    assert len(calls) <= imaging.MAX_ROIS + 1


def test_imaging_shorter_than_one_frame_is_a_clear_error(fly):
    from kickthefly.lab import imaging

    with pytest.raises(imaging.ImagingError, match="frame"):
        fly.image(0.01)
    fly.express("trpa1", "type:DNp01")
    with pytest.raises(imaging.ImagingError, match="frame"):
        fly.image(0.01)
    with pytest.raises(imaging.ImagingError):
        fly.image(-1.0)


@pytest.mark.parametrize("kw", [dict(f0_photons=float("inf")), dict(dff_per_spike=float("inf")), dict(baseline_tau_s=float("nan")),
                                dict(baseline_tau_s=-1.0), dict(dff_cap=float("nan"))])
def test_imaging_session_refuses_non_finite_parameters(kw):
    from kickthefly.lab import imaging

    with pytest.raises(imaging.ImagingError):
        imaging.ImagingSession(4, {"a": np.array([0])}, **kw)


# --- the Python API -------------------------------------------------------------------------------------------------------------------
def test_a_refused_dose_does_not_poison_later_drug_calls(fly):
    from kickthefly.lab import pharmacology as ph

    with pytest.raises(ph.PharmError):
        fly.drug("picrotoxin", 2.0)
    out = fly.drug("cholinergic", 0.5)              # raised 'dose must be between 0 and 1' forever before the fix
    assert out is not None
    assert fly._doses == {"cholinergic": 0.5}
    with pytest.raises(ph.PharmError):
        fly.drug("cholinergic", 0.5, cut=float("nan"))
    assert fly._doses == {"cholinergic": 0.5}


def test_wiring_from_a_damaged_save_does_not_crash(synthetic_pack):
    from kickthefly.sim.wiring import Wiring

    assert Wiring.from_dict(dict(nt_scales=[["gaba", 0.5]])).nt_scales == ()
    assert Wiring.from_dict(dict(nt_scales={"gaba": "x", "acetylcholine": 0.5})).nt_scales == (("acetylcholine", 0.5),)
    assert Wiring.from_dict(dict(nt_scales={"gaba": float("nan")})).nt_scales == ()
    assert Wiring.from_dict(dict(nt_min_conf="high")).nt_min_conf == 0.0


def test_conftest_forces_the_dummy_video_driver():
    """Your shell may export SDL_VIDEODRIVER=wayland; a test must never open a real window."""
    import os

    assert os.environ["SDL_VIDEODRIVER"] in ("dummy", "offscreen")


# --- the playthrough bot's Neurodex check (pre-existing FAIL on the real pack) ---------------------------------------------------------
@pytest.fixture
def rig(synthetic_pack):
    import tempfile
    from pathlib import Path

    from kickthefly.lab import playthrough as pt

    r = pt.Rig(False, "cpu")
    r.game.x3.async_build = False
    for slot in r.game.flies:
        slot.brain.stop()
    time.sleep(0.15)
    r.take_snapshot(Path(tempfile.mkdtemp(prefix="ktf-snap-")) / "snap.ktfsave")
    yield r
    r.close()


def test_neurodex_check_judges_only_its_own_calm_window(rig):
    """In --playthrough all the check ran after dozens of tool legs and judged every type discovered since the start; a type
    found in play was (correctly) tagged 'play' and failed it."""
    from kickthefly.lab import playthrough as pt

    x3 = rig.game.x3
    x3.ensure_table()
    prog = x3.ensure_progress()
    prog.mark(x3.brain_name, "AN01B002", 3.0, "play")    # an earlier leg's discovery (the type the real run reported)
    res = pt.Result(id="t", group="extra", brain="adult")
    pt.guarded(res, pt.extra_neurodex, rig, res)
    assert res.status == pt.PASS, res.failures
    assert res.metrics["discovered_before"] >= 1


def test_neurodex_check_keeps_the_hand_off_the_fly_while_calm(rig, monkeypatch):
    """The bot's hand sat on the fly's head, so the fly smelled the tool every frame: never calm by the game's own rule."""
    from kickthefly.core import simcore
    from kickthefly.lab import playthrough as pt

    scents, driving = [], []
    real_poke, real_drive = k2.Brain.poke, simcore.drive
    monkeypatch.setattr(k2.Brain, "poke", lambda self, region, *a, **kw: (scents.append(region) if region == "scent" and not driving
                                                                         else None) or real_poke(self, region, *a, **kw))
    monkeypatch.setattr(simcore, "drive", lambda *a, **kw: driving.append(1) or real_drive(*a, **kw))
    res = pt.Result(id="t", group="extra", brain="adult")
    pt.guarded(res, pt.extra_neurodex, rig, res)
    assert res.status == pt.PASS, res.failures
    assert not scents, f"{len(scents)} scent pokes during the calm window"


def test_imaging_is_not_on_until_its_session_exists(game, monkeypatch):
    """Seen in a real 3D frame on the real pack: Lab > Imaging's toggle never turned imaging on. set_on() set `on` first, the view
    thread found no session while it was being built, raised and switched imaging off; start() then cleared the error."""
    from kickthefly.lab import livelab

    il = game.imaging_live
    seen = []
    real = livelab.ImagingLive.start

    def slow_start(self, br, graph=None):
        seen.append((self.on, self.session))         # what the view thread can see while the session is being built
        real(self, br, graph)

    monkeypatch.setattr(livelab.ImagingLive, "start", slow_start)
    il.set_on(True, game.brain, game.graph)
    assert seen == [(False, None)]
    assert il.on and il.session is not None and il.error == ""


def test_imaging_mode_does_not_show_firing_counts(game):
    """Seen in a real 3D frame: '50,776 firing / 3,127 pain' over the imaging view; those count the dF/F pixels the view was fed."""
    texts = []
    real = game._text
    game._text = lambda surf, text, *a, **kw: texts.append(str(text)) or real(surf, text, *a, **kw)
    game.view.firing, game.view.hot_firing = 50776, 3127
    surf = pygame.Surface((k2.W, k2.H))
    game.imaging_live.set_on(True, game.brain, game.graph)
    game._hud_overlay(surf, pygame.Rect(0, 0, 300, 200), 0.0, small=True)
    game.big_view = True
    game._draw_big_view(surf)
    assert not any("firing" in t and "50,776" in t for t in texts) and not any("3,127 pain" in t for t in texts)
    assert any("IMAGING (MODEL)" in t for t in texts)
    assert not any("pain-sensing neurons firing" in t for t in texts)        # the firing legend's colors aren't on screen


def test_the_inspector_patch_button_is_on_screen_and_clickable(game):
    """Seen in real 2D and 3D frames: the button was drawn at card.x + 312, past the 308-px card and the big view's clip; only
    'PAT' showed. Now a row of its own, inside the view, and a real click (down + up) opens Lab > Patch."""
    row = int(np.flatnonzero(game.brain.types == "MDN")[0])
    game.inspect = game._neuron_info(row)
    game.big_view = True
    game._draw_big_view(pygame.Surface((k2.W, k2.H)))
    r, i = game.patch_button
    assert game.big_rect.contains(r) and i == row
    assert not any(r.colliderect(p[0]) for p in game.path_buttons)
    pos = r.center
    game.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1), 0.0)
    game.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1), 0.0)
    assert game.menu.screen == "lab_patch" and game.patch_row == row


def test_thermo_page_shows_what_the_flies_sense_in_the_arena(game):
    texts = []
    real = game.menu.text
    game.menu.text = lambda surf, text, *a, **kw: texts.append(str(text)) or real(surf, text, *a, **kw)
    game.thermo_live.source = "arena"
    _draw(game, "lab_thermo")
    assert any("slider applies" in t for t in texts)
    game.arena_i = k2.ARENAS.index("thermo")
    game.flies[0].arena_temp_c = 31.25
    texts.clear()
    _draw(game, "lab_thermo")
    assert any("31.2 C" in t for t in texts), [t for t in texts if " C" in t]
