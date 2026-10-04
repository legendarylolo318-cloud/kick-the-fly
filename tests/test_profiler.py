"""3.1.0 task 5, the profiler overlay and the frame profile of `--benchmark`."""
from __future__ import annotations

import json
import time

import pytest

from conftest import needs_pack
from kickthefly.core import config
from kickthefly.core import profiler as pf


@pytest.fixture
def prof():
    p = pf.Profiler()
    yield p


def test_it_is_off_by_default_and_records_nothing_when_off(prof):
    assert pf.PROF.on is False
    with prof.section("render"):
        time.sleep(0.002)
    prof.add("sim", 5.0)
    prof.sim_busy(10.0)
    prof.end_frame()
    assert prof.stats()["render"] == (0.0, 0.0, 0.0) and not prof.frames
    assert prof.section("render") is prof.section("ui"), "off: one shared do-nothing context"


def test_sections_nest_and_the_inner_time_leaves_the_outer(prof):
    prof.set(True)
    prof.end_frame()
    with prof.section("render"):
        time.sleep(0.02)
        with prof.section("ui"):
            time.sleep(0.01)
    prof.end_frame()
    s = prof.stats()
    assert 8 <= s["ui"][0] <= 40
    assert 15 <= s["render"][0] <= 60, "render holds only its own 20 ms, not the nested 10"
    assert s["render"][0] + s["ui"][0] <= s["frame"][0] + 1.0


def test_frames_fps_and_percentiles(prof):
    prof.set(True)
    for i in range(30):
        prof.end_frame()
        time.sleep(0.004)
    st = prof.stats()
    assert len(prof.frames) == 29 and 2.5 < st["frame"][0] < 40 and 20 < st["fps"] < 400
    assert st["frame"][2] >= st["frame"][1] >= 0


def test_sim_busy_is_the_growth_of_the_brains_step_time(prof):
    prof.set(True)
    prof.sim_busy(1.0)
    prof.end_frame()
    prof.sim_busy(1.012)
    prof.end_frame()
    assert prof.stats()["sim"][0] == pytest.approx(12.0, abs=0.01)
    prof.sim_busy(0.5)                         # a counter that went backwards (a new brain) is not a negative frame
    prof.end_frame()
    assert prof.hist["sim"][-1] == 0.0


def test_turning_it_on_starts_clean_and_the_window_is_bounded(prof):
    prof.set(True)
    for _ in range(pf.WINDOW + 50):
        prof.add("render", 1.0)
        prof.end_frame()
    assert len(prof.frames) == pf.WINDOW and len(prof.hist["render"]) == pf.WINDOW
    prof.set(False)
    prof.set(True)
    assert not prof.frames and not prof.hist["render"]


def test_an_open_section_cannot_leak_across_frames(prof):
    prof.set(True)
    prof.end_frame()
    sec = prof.section("render")
    sec.__enter__()                           # never exited: an exception in the middle of a frame
    prof.end_frame()
    assert prof._stack == []


def test_lines_name_every_section_the_engine_and_the_view(prof):
    prof.set(True)
    prof.end_frame()
    prof.end_frame()
    text = "\n".join(prof.lines("gl (OpenGL 4.6)", "GPU"))
    for k in pf.ORDER:
        assert k in text
    assert "FPS" in text and "gl (OpenGL 4.6)" in text and "GPU" in text


def test_the_key_is_rebindable_defaults_to_f3_and_conflicts_with_nothing():
    assert ("profiler", "Profiler overlay", "f3") in config.ACTIONS
    cfg = config.Config(None)
    assert cfg.keys["profiler"] == "f3" and cfg.conflicts() == {}
    ok, _ = cfg.bind("profiler", "f9")
    assert ok and cfg.actions_for("f9") == ["profiler"] and cfg.actions_for("f3") == []


@pytest.mark.parametrize("three_d", [False, True])
def test_the_key_toggles_the_overlay_and_it_draws(synthetic_pack, three_d):
    import pygame
    import test_extras3 as t3
    from kickthefly.game import kick_the_fly as k2

    g = t3.make_game(three_d)
    try:
        pf.PROF.set(False)
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F3, mod=0, unicode="")
        assert g.x3.handle_event(ev) and pf.PROF.on
        pf.PROF.end_frame()
        pf.PROF.end_frame()
        surf = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
        g.x3.draw(surf, 1.0)
        assert surf.get_at((20, 20))[3] > 0, "the overlay card is on the surface"
        assert g.x3.handle_event(ev) and not pf.PROF.on
        surf2 = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
        g.x3.draw(surf2, 1.0)
        assert surf2.get_at((20, 20))[3] == 0
    finally:
        pf.PROF.set(False)
        for slot in g.flies:
            slot.brain.stop()


def test_the_game_loop_costs_nothing_measurable_when_off():
    t0 = time.perf_counter()
    for _ in range(100_000):
        with pf.PROF.section("render"):
            pass
    assert (time.perf_counter() - t0) / 100_000 < 2e-6


def test_the_benchmark_hooks_write_the_scene_json(tmp_path, monkeypatch):
    out = tmp_path / "scene.json"
    monkeypatch.setenv("KICK_THE_FLY_PROFILE_JSON", str(out))
    pf.bench_start()
    assert pf.PROF.on
    pf.bench_warm_done()
    pf.PROF.add("render", 2.0)
    pf.PROF.end_frame()
    pf.PROF.add("render", 3.0)
    pf.PROF.end_frame()

    class Be:
        name, device = "cpu", "CPU (NumPy)"

    class Brain:
        n, steps_per_s = 1000, 200.0
        sim = type("S", (), {"backend": Be()})()

    class Game:
        flies = [type("F", (), {"brain": Brain()})()]
        view = None

    pf.bench_finish(Game(), "smoke ok")
    pf.PROF.set(False)
    row = json.loads(out.read_text())
    assert row["engine"] == "cpu" and row["steps_per_s"] == 200.0 and row["flies"] == 1
    assert set(pf.ORDER) | {"frame"} <= set(row) and row["status"] == "smoke ok" and row["view"] == "CPU"


def test_without_the_env_var_the_hooks_do_nothing(monkeypatch):
    monkeypatch.delenv("KICK_THE_FLY_PROFILE_JSON", raising=False)
    pf.PROF.set(False)
    pf.bench_start()
    assert not pf.PROF.on


def test_the_frame_table_has_a_row_per_scene_and_says_why_one_did_not_run():
    from kickthefly.lab import framebench as fb

    row = {"scene": "3D", "fps": 144.0, "frame": {"mean": 6.9}, "sim": {"mean": 3.0}, "physics": {"mean": 0.4}, "render": {"mean": 1.6},
           "ui": {"mean": 3.9}, "present": {"mean": 0.1}, "engine": "gl", "view": "GPU", "steps_per_s": 200}
    table = fb.format_table([row, {"scene": "2D", "error": "exit 1: no display"}])
    lines = table.splitlines()
    assert "physics" in lines[0] and lines[2].startswith("3D") and "144" in lines[2] and "GPU" in lines[2]
    assert "not run: exit 1: no display" in lines[3]


def test_the_benchmark_flag_prints_both_tables_and_can_skip_the_scenes(monkeypatch, capsys, tmp_path):
    from kickthefly.lab import benchmark, framebench, headless

    monkeypatch.setattr(benchmark, "run_benchmark", lambda **kw: {"fake": True})
    monkeypatch.setattr(benchmark, "format_benchmark_report", lambda r: "SIM TABLE")
    monkeypatch.setattr(benchmark, "save_benchmark_results", lambda r, out: tmp_path / "bench.json")
    calls = []
    monkeypatch.setattr(framebench, "run_scenes", lambda **kw: calls.append(kw) or [{"scene": "3D", "error": "x"}])
    args = type("A", (), {"flies": None, "seconds": 1.0, "out": None, "no_render_bench": False})()
    assert headless.run_benchmark(args) == 0
    out = capsys.readouterr().out
    assert "SIM TABLE" in out and "Frame profile" in out and calls
    calls.clear()
    args.no_render_bench = True
    headless.run_benchmark(args)
    assert not calls and "Frame profile" not in capsys.readouterr().out


def test_the_flag_parses():
    from kickthefly.game import kick_the_fly as k2

    src = open(k2.__file__, encoding="utf-8").read()
    assert '"--no-render-bench"' in src and '"--benchmark"' in src


@needs_pack
def test_a_real_2d_scene_runs_and_reports_every_section():
    from kickthefly.lab import framebench as fb

    row = fb.run_scene("2D", ["--2d"], {}, seconds=5.0, backend="cpu", cap=60)
    assert "error" not in row, row
    for k in pf.ORDER:
        assert row[k]["mean"] >= 0
    assert row["render"]["mean"] > 0 and row["frames"] > 5 and row["engine"] == "cpu" and row["fps"] > 0
