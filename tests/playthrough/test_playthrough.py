"""The playthrough bot: its machinery without a brain, and a small slice of the real run with one."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import needs_pack
from kickthefly.core import loadout as lo
from kickthefly.lab import playthrough as pt


# --- machinery, no brain pack ---------------------------------------------------------------------------------------------
def fake_brain(n=2000, rows=200):
    br = SimpleNamespace(sense={("head", None): np.arange(rows), ("heat", None): np.arange(rows, 2 * rows)},
                         sim=SimpleNamespace(spikes=np.zeros(n, bool)))
    return br


def drive(probe, br, calm_rate, later_rate, rows, calm_steps=200, later_steps=100, seed=0):
    rng = np.random.default_rng(seed)
    for step in range(calm_steps + later_steps):
        rate = calm_rate if step < calm_steps else later_rate
        br.sim.spikes[:] = False
        br.sim.spikes[rows] = rng.random(len(rows)) < rate * pt.DT
        probe.sample()
    return probe.summarize(calm_steps)


def test_a_group_that_fires_far_above_its_own_noise_counts_as_fired():
    br = fake_brain()
    probe = pt.Probe(br, ((("head", None),),))
    (p,) = drive(probe, br, 2.0, 60.0, br.sense[("head", None)])
    assert p["fired"] and p["peak_hz"] > 40 and p["baseline_hz"] < 5


def test_a_group_that_does_not_change_did_not_fire_and_the_criterion_scales_with_the_groups_noise():
    false_positives = 0
    for seed in range(40):                                  # no change at all: the criterion should almost never fire
        br = fake_brain()
        probe = pt.Probe(br, ((("head", None),),))
        (p,) = drive(probe, br, 30.0, 30.0, br.sense[("head", None)], seed=seed)
        assert p["needed_hz"] > p["baseline_hz"]
        false_positives += p["fired"]
    assert false_positives <= 2, f"{false_positives} of 40 null runs counted as a response"
    br = fake_brain()
    probe = pt.Probe(br, ((("head", None),),))
    (quiet,) = drive(probe, br, 1.0, 1.0, br.sense[("head", None)])
    br = fake_brain()
    probe = pt.Probe(br, ((("head", None),),))
    (busy,) = drive(probe, br, 30.0, 30.0, br.sense[("head", None)])
    assert busy["baseline_sd_hz"] > quiet["baseline_sd_hz"], "a busier group has more noise, so it needs a bigger rise"
    br = fake_brain()
    probe = pt.Probe(br, ((("head", None),),))
    (doubled,) = drive(probe, br, 30.0, 60.0, br.sense[("head", None)])
    assert doubled["fired"], "doubling a busy group's rate is a response"


def test_probes_pool_keys_and_skip_groups_a_brain_does_not_have():
    br = fake_brain()
    probe = pt.Probe(br, ((("head", None), ("heat", None)), (("loom", None),)))
    assert [name for name, _ in probe.parts] == ["head+heat"] and probe.empty == ["loom"]
    assert len(probe.parts[0][1]) == 400


def test_a_failed_expectation_marks_the_result_and_an_exception_keeps_its_traceback():
    r = pt.Result(id="x", group="g")
    assert r.expect(True, "fine") and r.status == pt.PASS
    assert not r.expect(False, "broken") and r.status == pt.FAIL and r.failures == ["broken"]
    r2 = pt.Result(id="y", group="g")

    def boom():
        raise ValueError("kaput")

    pt.guarded(r2, boom)
    assert r2.status == pt.FAIL and "ValueError: kaput" in r2.failures[0] and "Traceback" in r2.traceback and r2.seconds >= 0


def test_the_report_has_json_and_a_markdown_table_and_the_exit_verdict(tmp_path):
    rep = pt.Report(dict(app_version="2.13.0", created="now", backend="cpu", brains="adult", quick=True))
    rep.add(pt.Result(id="a", group="brain", status=pt.PASS, seconds=1.0))
    rep.add(pt.Result(id="b", group="gate", status=pt.GATED, notes=['gate message: "Lab mode only"']))
    assert not rep.failed()
    bad = rep.add(pt.Result(id="c|d", group="game3d", status=pt.FAIL, failures=["nope"], traceback="Traceback: x"))
    assert rep.failed() and rep.counts() == {"pass": 1, "fail": 1, "skip": 0, "gated": 1}
    j, m = rep.write(tmp_path)
    data = json.loads(j.read_text())
    assert data["counts"]["fail"] == 1 and data["results"][2]["traceback"] == "Traceback: x"
    md = m.read_text()
    assert "| combo | group | verdict | seconds |" in md and "| a | brain | PASS |" in md and "c/d" in md and "## Tracebacks" in md


def test_gated_combos_are_asserted_with_the_message_a_player_gets():
    rs = pt.gate_results(["adult", "larva"], ["room", "pool", "field", "orchard"], list(lo.TOOL_NAMES))
    assert rs and all(r.status == pt.GATED for r in rs), [r.failures for r in rs if r.status != pt.GATED]
    ids = {r.id for r in rs}
    assert "gate:play:*:laser" in ids and "gate:larva:*:bomb" in ids and "gate:larva:*:decoy" in ids
    assert any(r.id.startswith("gate:larva:field:") for r in rs), "larva x an outdoor arena is gated"
    laser = next(r for r in rs if r.id == "gate:play:*:laser")
    assert "Lab mode only" in laser.notes[0]
    assert not any(r.id == "gate:larva:*:sugar" for r in rs), "sugar has a larval mapping: not gated"


def test_every_documented_probe_names_a_real_sense_key_format():
    for t in lo.CATALOG:
        for probe in t.probes:
            assert probe and all(isinstance(k, tuple) and len(k) == 2 for k in probe), t.name
    assert lo.BY_NAME["laser"].probes == (), "the laser is checked by aiming at cell types, not by a fixed group"


# --- the real thing, a slice --------------------------------------------------------------------------------------------------
@needs_pack
def test_a_slice_of_the_bot_runs_clean_on_both_brains_and_writes_its_report(tmp_path):
    from kickthefly.sim import brainpack

    brains = ("adult", "larva") if brainpack.find(brain="larva") else ("adult",)
    rep = pt.run(brains, tmp_path, quick=True, tools=["hand", "sugar", "torch", "laser", "bomb"], arenas=["room"],
                 progress=lambda s: None)
    bad = [(r.id, r.failures) for r in rep.results if r.status == pt.FAIL]
    assert not bad, bad
    ids = {r.id for r in rep.results}
    assert {"brain:adult:hand", "brain:adult:torch", "game3d:adult:room:sugar", "game3d:adult:room:torch",
            "extra:loadouts", "extra:pet-catch-up", "extra:surgery", "extra:individuality"} <= ids
    assert next(r for r in rep.results if r.id == "brain:adult:torch").metrics["replay_identical"] is True
    assert next(r for r in rep.results if r.id == "game3d:adult:room:torch").metrics["died"] is True, "torch: death and autopsy"
    assert next(r for r in rep.results if r.id == "game3d:adult:room:sugar").metrics["save_load_restored"] is True
    assert (tmp_path / "playthrough.json").exists() and "| combo |" in (tmp_path / "playthrough.md").read_text()
    if "larva" in brains:
        assert {"brain:larva:hand", "brain:larva:sugar", "gate:larva:*:bomb"} <= ids
    render = next(r for r in rep.results if r.id == "extra:render")
    assert render.status in (pt.PASS, pt.SKIP), "with no OpenGL the render checks are skipped, never failed"


@needs_pack
def test_main_returns_the_verdict_as_an_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "run", lambda *a, **k: _rep(pt.FAIL))
    assert pt.main(SimpleNamespace(playthrough="adult", out=str(tmp_path), sim_backend=None, playthrough_quick=True)) == 1
    monkeypatch.setattr(pt, "run", lambda *a, **k: _rep(pt.GATED))
    assert pt.main(SimpleNamespace(playthrough="all", out=str(tmp_path), sim_backend="cpu", playthrough_quick=False)) == 0


def _rep(status):
    rep = pt.Report({})
    rep.add(pt.Result(id="x", group="g", status=status))
    return rep


def test_no_opengl_marks_the_render_check_skipped_not_passed(monkeypatch):
    moderngl = pytest.importorskip("moderngl")
    monkeypatch.setattr(moderngl, "create_context", lambda **k: (_ for _ in ()).throw(RuntimeError("no GL (simulated)")))
    r = pt.Result(id="extra:render", group="extra")
    pt.guarded(r, pt.extra_render, None, r)
    assert r.status == pt.SKIP and "render checks skipped" in r.notes[0]


def test_the_2d_spider_drops_like_the_3d_one():
    """The 2D spider drops on its thread at the 3D spider's speed and to its height (kick3d._spider3d: 0.04 m a frame
    down to 0.12 m above the floor), converted with the games' shared scale of kick3d.S meters per 2D pixel."""
    from kickthefly.game import kick3d
    from kickthefly.game import kick_the_fly as k2

    assert k2.SPIDER_DROP * kick3d.S == pytest.approx(0.04)
    assert (k2.FLOOR - k2.SPIDER_DROP_TO) * kick3d.S == pytest.approx(0.12)


@needs_pack
def test_the_2d_spider_is_a_looming_threat_while_it_drops(tmp_path):
    """The one full-playthrough failure of 2.13 (game2d:adult:flypaper:spider): the 2D spider appeared at the ceiling
    already crawling, so a fly stuck to flypaper, which can't move its head toward it, saw it grow at ~4 rad/s at most
    and LPLC2/LC4 barely rose. The 3D spider drops on its thread first and looms while it drops; the 2D one now does too."""
    from kickthefly.game import kick_the_fly as k2

    rig = pt.Rig(False, "cpu")
    try:
        g = rig.game
        rig.set_arena("flypaper")
        rig.use("spider")
        sp = g.spider
        assert sp["state"] == "drop" and sp["p"][1] == k2.CEIL + 4.0
        slot = rig.slot
        y0 = sp["p"][1]
        g._spider(g.clock.now)
        assert sp["p"][1] == pytest.approx(y0 + k2.SPIDER_DROP)
        assert any(key == "spider" for key, _, _ in g._threats(slot, g.clock.now, (0, 0))), "it looms while it drops"
        for _ in range(200):
            if sp["state"] != "drop":
                break
            g._spider(g.clock.now)
        assert sp["state"] == "hunt" and sp["p"][1] == pytest.approx(k2.SPIDER_DROP_TO)
    finally:
        rig.close()


@needs_pack
def test_a_decoy_the_fly_never_touched_is_inconclusive_not_a_failure(tmp_path):
    """Under the 3D lamp the fly hovers at the light (~1.75 m up) and the dropped decoy lands on the floor, so there is
    no foreleg contact and LgLG5-8 stay at ~3 Hz (57-64 Hz everywhere the fly touches it). That is the "effect never
    reached the fly" case the bot already reports as a skip for food it never ate; it failed the full run instead."""
    rig = pt.Rig(True, "cpu")
    try:
        rig.take_snapshot(tmp_path / "snapshot3d.ktfsave")
        lamp = pt.game_leg(rig, "lamp", "decoy", 0.0, tmp_path, save_load=False)
        room = pt.game_leg(rig, "room", "decoy", 0.0, tmp_path, save_load=False)
    finally:
        rig.close()
    assert lamp.metrics["engaged"] is False, "the fly at the lamp never touched the decoy on the floor"
    assert lamp.status != pt.FAIL, (lamp.failures, lamp.notes)           # a skip, or a pass if the noisy group spiked anyway
    if lamp.status == pt.SKIP:
        assert any("never reached the fly" in n for n in lamp.notes)
    assert room.metrics["engaged"] is True and room.status == pt.PASS, (room.failures, room.metrics["probes"])
