"""3.0 day 5, Lab > Behavior rigs (hub and four scenes) and Mini-papers: they draw before anything is run and with results, in every accessibility
palette and at larger text, their controls are registered with the menu (mouse and gamepad share the hit list), a run is a background job collected by its
own label, and nothing starts by itself."""
from __future__ import annotations

import threading
import time

import numpy as np
import pygame
import pytest

from test_day5_rigs import FakeBrain, FakeTransducer
from test_labpages import StubHost

from kickthefly.game import rigs
from kickthefly.ui.bgjob import BgJob


@pytest.fixture
def menu(isolated_home):
    pygame.init()
    pygame.display.set_mode((1280, 760))
    from kickthefly.core import config
    from kickthefly.lab import lab
    from kickthefly.ui import menu as ui

    host = StubHost()
    host.cfg = config.Config(None)
    m = ui.Menu(host)
    lab.install(m)
    m.fonts()
    m.mouse = (0, 0)
    yield m
    pygame.display.quit()


def draw(m, page, scroll=0, size=(1280, 760)):
    surf = pygame.Surface(size)
    m.hits = []
    m.screen = page
    m.scroll[page] = scroll
    m.pages[page](m, surf, pygame.Rect(40, 40, size[0] - 80, size[1] - 80), (0, 0))
    return surf


def hit(m, ident):
    for rect, kind, data in m.hits:
        if data.get("id") == ident:
            return data
    raise AssertionError(f"no registered control {ident!r}; have {[d.get('id') for _, _, d in m.hits][:40]}")


@pytest.fixture
def fake_transducer(monkeypatch):
    monkeypatch.setattr(rigs.Transducer, "drive", lambda self, br, slip: None)
    monkeypatch.setattr(rigs.Transducer, "release", lambda self, br: None)


def fake_run(rig, seconds=12.0, **opt):
    from kickthefly.lab import rigassay

    return rigassay.scene_run(rig, 3, "subtle", br=FakeBrain(walk_level=2.0, follow=1.5), seconds=seconds, **opt)


@pytest.mark.parametrize("palette", ["default", "blue-yellow", "high-contrast"])
@pytest.mark.parametrize("larger", [False, True])
def test_every_rig_page_draws_before_and_after_a_run_in_every_palette_and_text_size(menu, fake_transducer, palette, larger):
    from kickthefly.lab import labrigs

    menu.host.cfg.set("access.palette", palette)
    menu.host.cfg.set("access.larger_text", larger)
    menu.fonts()
    draw(menu, "lab_rigs")
    s = labrigs.st(menu)
    for rig, page in labrigs.PAGES.items():
        s.res.pop(rig, None)
        draw(menu, page)
        for mode in __import__("kickthefly.lab.rigassay", fromlist=["SCENES"]).SCENES[rig]["mode"]:
            s.res[rig] = fake_run(rig, mode=mode)
            for t in (0.0, 5.0, 99.0):
                s.t, s.playing = t, False
                draw(menu, page)
    draw(menu, "lab_rigs")


def test_rig_scenes_stay_inside_a_phone_width_window(menu, fake_transducer):
    from kickthefly.lab import labrigs

    s = labrigs.st(menu)
    for rig, page in labrigs.PAGES.items():
        s.res[rig] = fake_run(rig)
        s.playing = False
        draw(menu, page, size=(448, 906))
        draw(menu, page, scroll=400, size=(448, 906))


def test_controls_are_registered_for_the_pad_and_change_the_rigs_condition(menu):
    from kickthefly.lab import labrigs

    draw(menu, "lab_rig_buridan")
    ids = [d.get("id") for _, kind, d in menu.hits if kind in ("button", "slider")]
    for want in ((("rig_mode", "buridan"), 0), (("rig_mode", "buridan"), 1), ("rig_run", "buridan"), ("rig_rec", "buridan"), ("lab_rig_buridan", "back")):
        assert want in ids, want
    hit(menu, (("rig_mode", "buridan"), 1))["click"]()
    assert labrigs.st(menu).p["buridan"]["mode"] == "none"
    draw(menu, "lab_rig_ball")
    hit(menu, (("rig_scene", "ball"), 1))["click"]()
    draw(menu, "lab_rig_ball")
    assert labrigs.st(menu).p["ball"]["scene"] == "panorama"
    assert any(d.get("id") == ("rig_sl", "ball", "omega") for _, _, d in menu.hits), "the drift slider shows in the panorama scene"


def test_a_run_is_a_background_job_collected_by_its_rig_and_nothing_starts_by_itself(menu, monkeypatch):
    from kickthefly.lab import labrigs, rigassay

    labrigs.install(menu)
    monkeypatch.setattr(rigassay, "scene_run", lambda rig, *a, **k: pytest.fail("a page ran a rig on its own"))
    for page in labrigs.PAGES.values():
        draw(menu, page)
    s = labrigs.st(menu)
    assert s.job is None
    got = []

    def fake(rig, seed, individuality, folder=None, progress=None, cancel=None, br=None, **opt):
        got.append((rig, seed, individuality, folder, opt))
        return dict(rig=rig, seed=seed, individuality=individuality, params=dict(opt), cols=list(rigs.COLS), trace=[], seconds=1.0, summary={"x": 1.0}, tags={}, title="t", geometry={})

    monkeypatch.setattr(rigassay, "scene_run", fake)
    labrigs.start(menu, "fourfield")
    s.job.thread.join(5)
    draw(menu, "lab_rig_tethered")                    # a different rig's page open: the result still goes to fourfield
    assert "fourfield" in s.res and "tethered" not in s.res and s.job is None, s.error
    assert got[0][0] == "fourfield" and got[0][3] is None, "not recorded unless asked"
    s.record = True
    labrigs.start(menu, "ball")
    s.job.thread.join(5)
    assert got[1][3] is not None and "rig-ball" in str(got[1][3])


def test_a_failed_rig_job_shows_its_error_and_cancel_stops_it(menu, monkeypatch):
    from kickthefly.lab import labrigs, rigassay

    s = labrigs.st(menu)

    def boom(*a, **k):
        raise RuntimeError("worker died")

    monkeypatch.setattr(rigassay, "scene_run", boom)
    labrigs.start(menu, "buridan")
    s.job.thread.join(5)
    draw(menu, "lab_rig_buridan")
    assert "worker died" in s.error and "buridan" not in s.res
    started = threading.Event()

    def slow(rig, seed, individuality, folder=None, progress=None, cancel=None, br=None, **opt):
        started.set()
        for _ in range(500):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            time.sleep(0.01)
        return {}

    monkeypatch.setattr(rigassay, "scene_run", slow)
    labrigs.start(menu, "ball")
    started.wait(2)
    s.job.stop()
    s.job.thread.join(3)
    assert s.job.cancelled and not s.job.running


def test_rig_pad_nav_runs_changes_condition_and_goes_back(menu, monkeypatch):
    from kickthefly.lab import labrigs

    host = menu.host
    host.menu = menu
    menu.screen = "lab_rig_tethered"
    s = labrigs.st(menu)
    called = []
    monkeypatch.setattr(labrigs, "start", lambda m, rig: called.append(rig))
    assert labrigs.pad_nav(host, {"tool_next"}) and s.p["tethered"]["mode"] == "closed"
    assert labrigs.pad_nav(host, {"killcam"}) and s.p["tethered"]["seconds"] == 24.0
    assert labrigs.pad_nav(host, {"use"}) and called == ["tethered"]
    menu.screen = "lab"
    assert not labrigs.pad_nav(host, {"use"})


# --- mini-papers ----------------------------------------------------------------------------------------------------------------------
def fake_paper_result(pid, full=False, reproduced=True, source="live"):
    from kickthefly.lab import minipapers

    p = minipapers.PAPERS[pid]

    def play(paper, seeds):
        out = {}
        for q in paper.questions:
            if q.test == "mb_conditioning":
                m = dict(pi_mean=1.0 if reproduced else 0.0, pi_sd=0.0, control_pi_mean=-0.03, control_pi_sd=0.1, fear_cs_plus=0.7, fear_cs_minus=0.05, n=len(seeds), p_value=0.001)
                out[q.test] = dict(measured=m, passed=reproduced, per_seed=[dict(seed=s, pi=m["pi_mean"], control_pi=-0.03) for s in seeds])
            elif q.test.startswith("rig:"):
                continue
            else:
                d, c = (4.0, 0.9) if reproduced else (0.9, 0.9)
                m = dict(drive_ratio_mean=d, drive_ratio_sd=0.5, control_ratio_mean=c, control_ratio_sd=0.2, p_value=0.001, n=len(seeds))
                out[q.test] = dict(measured=m, passed=reproduced, per_seed=[dict(seed=s, drive=dict(ratio=d + 0.1 * i), control=dict(ratio=c)) for i, s in enumerate(seeds)])
        return out

    if pid == "colomb_2012":
        import kickthefly.lab.rigassay as ra

        flies = [dict(seed=s, stripes=dict(deviation_deg=3.0, transits=17), none=dict(deviation_deg=40.0, transits=7)) for s in seeds_for(full)]
        monkey = lambda *a, **k: flies
        orig = ra.run_flies
        ra.run_flies = monkey
        try:
            return minipapers.run_paper(pid, seeds_for(full), workers=1)
        finally:
            ra.run_flies = orig
    return minipapers.run_paper(pid, seeds_for(full), play=play)


def seeds_for(full):
    from kickthefly.lab import minipapers

    return minipapers.FULL_SEEDS if full else minipapers.QUICK_SEEDS


@pytest.mark.parametrize("palette", ["default", "blue-yellow", "high-contrast"])
@pytest.mark.parametrize("larger", [False, True])
def test_every_mini_paper_step_draws_with_and_without_results_in_every_palette_and_text_size(menu, palette, larger):
    from kickthefly.lab import minipapers
    from kickthefly.ui import minipaper_ui

    menu.host.cfg.set("access.palette", palette)
    menu.host.cfg.set("access.larger_text", larger)
    menu.fonts()
    draw(menu, "lab_minipapers")
    s = minipaper_ui.st(menu)
    for pid in minipapers.ORDER:
        minipaper_ui.open_paper(menu, pid)
        s.results.pop(pid, None)
        for with_result in (False, True):
            if with_result:
                s.results[pid] = fake_paper_result(pid, full=larger)
                s.answers[pid] = {q.test: q.expected for q in minipapers.PAPERS[pid].questions}
            for i in range(s.sess.total_steps):
                s.sess.goto_step(i)
                draw(menu, "lab_minipapers")
                draw(menu, "lab_minipapers", scroll=300, size=(448, 906))


def test_a_failed_paper_says_so_and_why(menu):
    from kickthefly.lab import minipapers

    res = fake_paper_result("hampel_2015", reproduced=False)
    rows = minipapers.compare(res, {"antenna_grooming_circuit": "up", "adn_grooming_motor": "up"})
    assert [r["model_reproduces_paper"] for r in rows] == [False, False]
    text = minipapers.render(res, {"antenna_grooming_circuit": "up"})
    assert "DOES NOT REPRODUCE" in text and "no legs" in text and "What this model cannot check" in text


def test_the_run_waits_for_a_hypothesis_and_the_plot_and_compare_wait_for_a_run(menu, monkeypatch):
    from kickthefly.lab import minipapers
    from kickthefly.ui import minipaper_ui

    monkeypatch.setattr(minipapers, "run_paper", lambda *a, **k: pytest.fail("a page ran an experiment on its own"))
    draw(menu, "lab_minipapers")
    hit(menu, ("mp_paper", "von_reyn_2014"))["click"]()
    s = minipaper_ui.st(menu)
    s.sess.goto_step(2)
    draw(menu, "lab_minipapers")
    assert hit(menu, ("mp_run", "von_reyn_2014")).get("enabled") is False
    s.sess.goto_step(1)
    draw(menu, "lab_minipapers")
    hit(menu, ("mp_opt", "von_reyn_2014", "looming_escape", "up"))["click"]()
    s.sess.goto_step(2)
    draw(menu, "lab_minipapers")
    assert hit(menu, ("mp_run", "von_reyn_2014")).get("enabled") is True
    for step in (3, 4):
        s.sess.goto_step(step)
        draw(menu, "lab_minipapers")                          # no result yet: a note, not a crash
    assert s.job is None


def test_a_mini_paper_run_is_a_job_collected_by_its_paper(menu, monkeypatch):
    from kickthefly.lab import minipapers
    from kickthefly.ui import minipaper_ui

    s = minipaper_ui.st(menu)
    called = []
    made = {pid: fake_paper_result(pid) for pid in ("tully_quinn_1985", "shiu_2024")}      # built before run_paper is replaced

    def fake(pid, seeds, workers=1, progress=None, cancel=None, play=None):
        called.append((pid, tuple(seeds), workers))
        return made[pid]

    monkeypatch.setattr(minipapers, "run_paper", fake)
    minipaper_ui.open_paper(menu, "tully_quinn_1985")
    minipaper_ui.start(menu, "tully_quinn_1985")
    s.job.thread.join(5)
    s.paper = "shiu_2024"                                      # another paper open when it finishes
    draw(menu, "lab_minipapers")
    assert "tully_quinn_1985" in s.results and "shiu_2024" not in s.results
    assert called[0][1] == minipapers.QUICK_SEEDS
    s.full = True
    minipaper_ui.start(menu, "shiu_2024")
    s.job.thread.join(5)
    assert called[1][1] == minipapers.FULL_SEEDS


def test_a_quick_buridan_run_is_judged_on_the_effects_not_on_a_p_value_it_cannot_reach(monkeypatch):
    import kickthefly.lab.rigassay as ra
    from kickthefly.lab import minipapers

    good = lambda seeds: [dict(seed=s, stripes=dict(deviation_deg=3.0, transits=18), none=dict(deviation_deg=42.0 + s, transits=7)) for s in seeds]
    monkeypatch.setattr(ra, "run_flies", lambda rig, seeds, *a, **k: good(seeds))
    quick = minipapers.run_paper("colomb_2012", minipapers.QUICK_SEEDS)
    v = quick["questions"][0]["verdict"]
    assert v["reproduced"] and "too few flies" in v["basis"] and not quick["full"]
    full = minipapers.run_paper("colomb_2012", minipapers.FULL_SEEDS)
    assert full["full"] and full["questions"][0]["verdict"]["reproduced"] and "p < 0.01" in full["questions"][0]["verdict"]["basis"]
    weak = lambda seeds: [dict(seed=s, stripes=dict(deviation_deg=40.0, transits=7), none=dict(deviation_deg=42.0, transits=7)) for s in seeds]
    monkeypatch.setattr(ra, "run_flies", lambda rig, seeds, *a, **k: weak(seeds))
    assert not minipapers.run_paper("colomb_2012", minipapers.QUICK_SEEDS)["questions"][0]["verdict"]["reproduced"]


def test_the_larva_paper_shows_recorded_numbers_when_the_pack_is_missing(menu, synthetic_pack):
    from kickthefly.lab import minipapers

    ok, why = minipapers.available(minipapers.PAPERS["ohyama_2015"])
    assert not ok and "larva" in why
    res = minipapers.run_paper("ohyama_2015")
    assert res["source"] == "recorded" and res["full"] and all(not q["verdict"]["reproduced"] for q in res["questions"])
    assert "recorded" in minipapers.render(res)


def test_an_adult_paper_without_the_adult_pack_says_why_instead_of_crashing(menu, monkeypatch):
    from kickthefly.lab import minipapers
    from kickthefly.ui import minipaper_ui

    monkeypatch.setattr(minipapers, "available", lambda p: (False, "the adult brain pack is not built"))
    with pytest.raises(FileNotFoundError, match="not built"):
        minipapers.run_paper("von_reyn_2014")
    s = minipaper_ui.st(menu)
    minipaper_ui._run_or_record(menu, minipapers.PAPERS["von_reyn_2014"], s)
    assert "not built" in s.error and s.job is None
    minipaper_ui.open_paper(menu, "von_reyn_2014")
    s.sess.goto_step(2)
    draw(menu, "lab_minipapers")                                 # and the page shows the error line without raising


def test_mini_paper_pad_nav(menu, monkeypatch):
    from kickthefly.ui import minipaper_ui

    host = menu.host
    host.menu = menu
    menu.screen = "lab_minipapers"
    s = minipaper_ui.st(menu)
    assert not minipaper_ui.pad_nav(host, {"use"}), "nothing to do on the list"
    minipaper_ui.open_paper(menu, "hampel_2015")
    assert minipaper_ui.pad_nav(host, {"tool_next"}) and s.sess.step_idx == 1
    assert minipaper_ui.pad_nav(host, {"killcam"}) and s.answers["hampel_2015"]["antenna_grooming_circuit"] == "up"
    assert minipaper_ui.pad_nav(host, {"killcam"}) and s.answers["hampel_2015"]["antenna_grooming_circuit"] == "same"
    assert minipaper_ui.pad_nav(host, {"use"}) and s.q_cursor == 1 and s.sess.step_idx == 1, "use moves to the paper's second question"
    assert minipaper_ui.pad_nav(host, {"big_view"}) and s.answers["hampel_2015"]["adn_grooming_motor"] == "same"
    assert minipaper_ui.pad_nav(host, {"use"}) and s.sess.step_idx == 2 and s.q_cursor == 0
    assert minipaper_ui.pad_nav(host, {"crouch"}) and s.paper is None


def test_the_five_curated_lectures_are_still_five_and_papers_are_lecture_protocols(menu):
    from kickthefly.lab import classroom, minipapers

    assert len(classroom.CURATED_LECTURES) == 5
    for pid in minipapers.ORDER:
        proto = classroom.lecture(f"paper_{pid}")
        assert proto is not None and len(proto.steps) == 5 and all(st_.citation for st_ in proto.steps)
        sess = classroom.ClassroomSession(f"paper_{pid}")
        assert sess.protocol is proto and sess.next_step() and sess.step_idx == 1


def test_the_pause_menu_has_mini_papers_and_the_lab_hub_lists_rigs_and_papers(menu):
    from kickthefly.ui import menu as ui

    menu.host.menu_action = lambda name: menu.show("lab_minipapers") if name == "minipapers" else None
    menu.screen = "pause"
    surf = pygame.Surface((1280, 760))
    menu.hits = []
    menu._page_pause(surf, pygame.Rect(240, 40, 800, 680))
    ids = [d.get("id") for _, kind, d in menu.hits if kind == "button"]
    assert ("pause", "minipapers") in ids and ("pause", "arcade") in ids
    hit(menu, ("pause", "minipapers"))["click"]()
    assert menu.screen == "lab_minipapers"
    labs = [p for _, p, _ in menu.host.lab_pages()]
    assert "lab_rigs" in labs and "lab_minipapers" in labs
