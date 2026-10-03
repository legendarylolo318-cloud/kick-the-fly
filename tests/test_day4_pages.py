"""3.0 day 4, the Fly arcade page and the three Lab pages (Network science, Sleep deprivation, Sensitivity analysis): they draw with
results and without, in every accessibility palette and with larger text, their buttons are registered with the menu (the
mouse and the gamepad both use the registered hit list), a background job reports progress, errors and cancellation, and the pages
never start work by themselves."""
from __future__ import annotations

import threading
import time

import numpy as np
import pygame
import pytest

from test_labpages import StubHost
from kickthefly.ui.bgjob import BgJob


@pytest.fixture
def menu(isolated_home):
    pygame.init()
    pygame.display.set_mode((1280, 760))
    from kickthefly.core import config
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import lab
    from kickthefly.ui import arcade_ui, menu as ui

    host = StubHost()
    host.cfg = config.Config(None)
    m = ui.Menu(host)
    lab.install(m)
    m.pages["arcade"] = arcade_ui.page
    m.fonts()
    m.mouse = (0, 0)
    yield m
    pygame.display.quit()


def draw(m, page, scroll=0):
    surf = pygame.Surface((1280, 760))
    m.hits = []
    m.scroll[page] = scroll
    m.pages[page](m, surf, pygame.Rect(40, 40, 1200, 680), (0, 0))
    return surf


def hit(m, ident):
    for rect, kind, data in m.hits:
        if data.get("id") == ident:
            return data
    raise AssertionError(f"no registered control {ident!r}; have {[d.get('id') for _, _, d in m.hits][:40]}")


def fake_card(s):
    return dict(seed=s, mode="subtle", loom_latency_s=0.12, loom_crossed=True, sugar_ratio=2.0, turning_ratio=1.0, turning_log_ratio=0.0, walk_level_calm=1.0,
                title="Alert Steady", summary="Alert · Steady", traits=[], measured=True)


def fake_bracket(size=4):
    from kickthefly.lab import tournament as tn

    def play(task):
        if task[0] == "card":
            return fake_card(task[1])
        a, b = task[0], task[1]
        frames = [[i / 10, -3 + i * 0.1, 0, 0.2, 100, 3, 0, 3.0, max(0, 100 - i), [[0.0, 0.0, 0]]] for i in range(30)]
        return dict(winner=str(max(a, b)), hp={str(a): 20.0, str(b): 70.0}, shots={str(a): 3, str(b): 4}, hits={str(a): 2, str(b): 5}, seconds=3.0,
                    knockout=False, max_levels={str(a): dict(fire=4, escape=1, run=1, walk=1), str(b): dict(fire=5, escape=1, run=1, walk=1)},
                    events=[["hit", 1.0, str(b)], ["dodge", 1.4, str(a)]], frames=frames, steps=600, attempt=0, match_seed=1)

    b = tn.run_bracket(list(range(2000, 2000 + size)), play=play, favorite=2001, drivers=False)
    b["champion_drivers"] = [dict(type="DNp35", neurons=2, hz_before_hits=30.0, hz_overall=5.0, ratio=6.0)]
    b["champion_readouts"] = [dict(type="DNp35", neurons=2, hz_before_hits=30.0, hz_overall=5.0, ratio=6.0)]
    return b


def fake_race():
    seeds = [3000, 3001, 3002]
    lanes = {str(s): dict(finish_s=10.0 + i, distance=8.0, trace=[[0, 0], [5.0, 4.0], [10.0 + i, 8.0]], lures_touched=[0], max_walk_level=2.0, seconds=10.0 + i)
             for i, s in enumerate(seeds)}
    return dict(seeds=seeds, track_m=8.0, lures=[[2.0, "sugar"], [4.0, "fruit"]], runs={"0": lanes}, orders=[seeds], winner=seeds[0],
                odds={str(s): dict(p_win=1 / 3, decimal_odds=2.7, form=0.0) for s in seeds})


@pytest.mark.parametrize("palette", ["default", "blue-yellow", "high-contrast"])
@pytest.mark.parametrize("larger", [False, True])
def test_every_page_draws_with_results_in_every_palette_and_text_size(menu, palette, larger):
    from kickthefly.lab import labday4, netsci, sensitivity
    from kickthefly.ui import arcade_ui

    menu.host.cfg.set("access.palette", palette)
    menu.host.cfg.set("access.larger_text", larger)
    menu.fonts()
    st = arcade_ui.state(menu)
    st.bracket, st.sel_match = fake_bracket(4), (0, 0)
    st.replay_t = 1.0
    for tab in ("tournament", "race"):
        st.tab = tab
        st.field = [fake_card(s) for s in (3000, 3001, 3002)]
        from kickthefly.lab import racing

        st.odds, st.bet_fly, st.race = racing.odds_table(st.field), 3001, fake_race()
        st.bet_result = [dict(fly=3001, stake=10, odds=2.7, won=False, delta=-10, points=90)]
        draw(menu, "arcade")
    d = labday4._st(menu)
    from test_day4_sensitivity import fake_cell

    cells = [fake_cell(sensitivity.BASELINE, None, [1, 2, 3], ["looming_escape", "sugar_feeding"])] + \
            [fake_cell("noise_std", v, [1, 2, 3], ["looming_escape", "sugar_feeding"]) for v in (0.025, 0.075)]
    d.sens = dict(seeds=[1, 2, 3], tests=["looming_escape", "sugar_feeding"], underpowered=True, seconds=1.0, cells=cells,
                  parameters=[dict(id="noise_std", label="Membrane noise", kind="lif", default=0.05, note="", values=[0.025, 0.075])])
    draw(menu, "lab_sensitivity")
    from test_day4_sleepdep import fake_play
    from kickthefly.lab import sleepdep

    d.sleep = sleepdep.run(list(range(1000, 1010)), play=fake_play)
    draw(menu, "lab_sleepdep")


def test_the_pages_draw_before_anything_has_been_run(menu):
    for page in ("arcade", "lab_netsci", "lab_sleepdep", "lab_sensitivity"):
        draw(menu, page)
    from kickthefly.ui import arcade_ui

    arcade_ui.state(menu).tab = "race"
    draw(menu, "arcade")


def test_network_science_page_draws_a_real_result_and_exports_it(menu, synthetic_pack, tmp_path):
    from kickthefly.lab import labday4, netsci

    d = labday4._st(menu)
    d.net["adult"] = netsci.compute("adult", nulls=2, wedges=3000, seed=1)
    draw(menu, "lab_netsci")
    hit(menu, "ns_csv")["click"]()
    assert "CSV files" in d.msg or d.error == "", (d.msg, d.error)
    from kickthefly.lab import recorder

    assert list((recorder.exports_dir() / "netsci-adult").glob("*.csv"))


def test_the_netsci_page_does_not_compute_until_asked(menu, monkeypatch):
    from kickthefly.lab import netsci

    monkeypatch.setattr(netsci, "compute", lambda *a, **k: pytest.fail("the page computed on its own"))
    draw(menu, "lab_netsci")
    from kickthefly.lab import labday4

    assert labday4._st(menu).job is None


def test_the_arcade_buttons_are_registered_so_the_gamepad_can_reach_them(menu):
    from kickthefly.ui import arcade_ui

    draw(menu, "arcade")
    ids = [d.get("id") for _, kind, d in menu.hits if kind == "button"]
    for want in ("arc_run", "arc_new", ("arc_size", 4), ("arc_size", 8), ("arc_size", 16), ("arcade_tab", "race"), ("arcade", "back")):
        assert want in ids, want
    assert all(d.get("enabled", True) is not None for _, _, d in menu.hits)
    arcade_ui.state(menu).tab = "race"
    draw(menu, "arcade")
    ids = [d.get("id") for _, kind, d in menu.hits if kind == "button"]
    assert "race_look" in ids and "race_new" in ids


def test_picking_a_favorite_a_size_and_a_new_set_of_flies(menu):
    from kickthefly.ui import arcade_ui

    st = arcade_ui.state(menu)
    draw(menu, "arcade")
    hit(menu, ("arc_fav", st.base_seed + 2))["click"]()
    assert st.favorite == st.base_seed + 2
    hit(menu, ("arc_size", 4))["click"]()
    assert st.size == 4 and st.favorite is None, "a new size clears a favorite that may not be in the bracket"
    draw(menu, "arcade")
    hit(menu, "arc_new")["click"]()
    assert st.base_seed == 2004


def test_a_finished_bracket_job_is_taken_in_by_the_page_and_the_favorite_is_told(menu):
    from kickthefly.ui import arcade_ui

    st = arcade_ui.state(menu)
    st.favorite = 2001
    b = fake_bracket(4)
    st.job = BgJob("Tournament", lambda job: b).start()
    st.job.thread.join(2)
    draw(menu, "arcade")
    assert st.bracket is b and st.job is None and st.sel_match == (1, 0)
    assert menu.message and "favorite" in menu.message[0], "the player is told how their favorite did"


def test_a_failed_job_shows_its_error_and_leaves_no_result(menu):
    from kickthefly.ui import arcade_ui

    st = arcade_ui.state(menu)

    def boom(job):
        raise RuntimeError("worker died")

    st.job = BgJob("Tournament", boom).start()
    st.job.thread.join(2)
    draw(menu, "arcade")
    assert "worker died" in st.error and st.bracket is None


def test_a_bet_is_settled_in_game_points_when_the_race_comes_back(menu, isolated_home):
    from kickthefly.core import points
    from kickthefly.ui import arcade_ui

    st = arcade_ui.state(menu)
    st.bet_fly, st.stake = 3001, 10
    race = fake_race()
    race["winner"] = 3001
    st.job = BgJob("Race", lambda job: race).start()
    st.job.thread.join(2)
    draw(menu, "arcade")
    assert st.bet_result and st.bet_result[0]["won"] and points.Wallet().points == points.START_POINTS + st.bet_result[0]["delta"] > points.START_POINTS


def test_the_page_says_points_are_not_money(menu):
    import inspect

    from kickthefly.ui import arcade_ui

    assert "no money" in inspect.getsource(arcade_ui) and "Points are not money" in open("kickthefly/ui/menu.py").read()


# --- the background job ----------------------------------------------------------------------------------------------------------------
def test_bgjob_reports_progress_a_result_and_finishes():
    def work(job, n):
        for i in range(n):
            job.update((i + 1) / n, f"step {i + 1}")
        return n * 2

    j = BgJob("t", work, n=4).start()
    j.thread.join(2)
    assert j.result == 8 and j.frac == 1.0 and j.note == "step 4" and not j.running and j.error is None and not j.cancelled


def test_bgjob_cancel_stops_it_and_an_error_is_kept_as_text():
    started = threading.Event()

    def slow(job):
        started.set()
        while not job.cancel.is_set():
            time.sleep(0.01)
        raise RuntimeError("cancelled")

    j = BgJob("t", slow).start()
    started.wait(2)
    j.stop()
    j.thread.join(2)
    assert j.cancelled and j.error is None and j.result is None

    def bad(job):
        raise ValueError("nope")

    k = BgJob("t", bad).start()
    k.thread.join(2)
    assert k.error == "ValueError: nope" and not k.cancelled


def test_the_heavy_work_never_runs_on_the_calling_thread():
    main = threading.get_ident()
    seen = []
    j = BgJob("t", lambda job: seen.append(threading.get_ident())).start()
    j.thread.join(2)
    assert seen and seen[0] != main


# --- the gamepad (the pad's own bindings reach the arcade; the menus are otherwise mouse-driven) --------------------------------------
def test_the_gamepad_can_pick_a_favorite_change_the_size_and_run_the_tournament_and_close_the_page(menu, monkeypatch):
    from types import SimpleNamespace

    from kickthefly.ui import arcade_ui

    host = SimpleNamespace(menu=menu)
    st = arcade_ui.state(menu)
    menu.screen = "other"
    assert arcade_ui.pad_nav(host, {"use"}) is False, "only while the arcade page is open"
    menu.screen = "arcade"
    assert arcade_ui.pad_nav(host, {"tool_next"}) and st.favorite == st.base_seed
    assert arcade_ui.pad_nav(host, {"tool_next"}) and st.favorite == st.base_seed + 1
    assert arcade_ui.pad_nav(host, {"tool_prev"}) and st.favorite == st.base_seed
    assert arcade_ui.pad_nav(host, {"killcam"}) and st.size == 16 and st.favorite is None
    assert arcade_ui.pad_nav(host, {"big_view"}) and st.size == 8
    started = []
    monkeypatch.setattr(arcade_ui, "_start_tournament", lambda m, s: started.append("t"))
    monkeypatch.setattr(arcade_ui, "_start_field", lambda m, s: started.append("f"))
    monkeypatch.setattr(arcade_ui, "_start_race", lambda m, s: started.append("r"))
    assert arcade_ui.pad_nav(host, {"use"}) and started == ["t"]
    assert arcade_ui.pad_nav(host, {"neurodex"}) and st.tab == "race"
    assert arcade_ui.pad_nav(host, {"use"}) and started == ["t", "f"]
    st.field = [fake_card(s) for s in (1, 2, 3)]
    from kickthefly.lab import racing

    st.odds = racing.odds_table(st.field)
    assert arcade_ui.pad_nav(host, {"tool_next"}) and st.bet_fly == 1
    assert arcade_ui.pad_nav(host, {"use"}) and started == ["t", "f", "r"]
    assert arcade_ui.pad_nav(host, {"killcam"}) and st.lanes == 6 and st.field is None and st.odds is None
    closed = []
    monkeypatch.setattr(menu, "back", lambda: closed.append(1))
    assert arcade_ui.pad_nav(host, {"crouch"}) and closed == [1]


def test_a_running_job_ignores_everything_but_the_tab_switch_and_close(menu):
    from types import SimpleNamespace

    from kickthefly.ui import arcade_ui

    host = SimpleNamespace(menu=menu)
    st = arcade_ui.state(menu)
    menu.screen = "arcade"
    ev = threading.Event()
    st.job = BgJob("Tournament", lambda job: ev.wait(2)).start()
    try:
        assert arcade_ui.pad_nav(host, {"use"}) is False and arcade_ui.pad_nav(host, {"tool_next"}) is False and st.favorite is None
        assert arcade_ui.pad_nav(host, {"neurodex"}) is True
    finally:
        ev.set()
        st.job.thread.join(3)


def test_the_netsci_page_looks_for_its_cache_once_not_every_frame(menu, monkeypatch):
    from kickthefly.lab import netsci

    calls = []
    monkeypatch.setattr(netsci, "load_cached", lambda *a, **k: calls.append(1))
    for _ in range(5):
        draw(menu, "lab_netsci")
    assert len(calls) == 1
