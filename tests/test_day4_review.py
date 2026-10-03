"""3.0 day 4 review (Opus): one regression test per bug found in Sonnet's day 4 features. Every test here failed on sonnet/3.0-day4
(898e20a) before its fix. Plumbing tests use the SYNTHETIC pack or fakes; nothing here is a biological result."""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import pytest

from test_day4_pages import menu  # noqa: F401  (the pages' fixture: a Menu on a stub host)


# --- points: a wallet can never go negative, never crash on a hand-edited file, and never pays a non-odds ------------------------
@pytest.mark.parametrize("text", ['{"points": -500}', '{"points": Infinity}', '{"points": -Infinity}', '{"points": NaN}',
                                  '{"points": 1e400}', '{"points": "12"}', '{"points": [3]}', '[1, 2]', '{"points": 3}'])
def test_a_hand_edited_wallet_never_loads_negative_or_crashes(tmp_path, text):
    from kickthefly.core import points

    p = tmp_path / "arcade_points.json"
    p.write_text(text, encoding="utf-8")
    w = points.Wallet(p)
    assert isinstance(w.points, int) and w.points >= points.TOP_UP_BELOW, (text, w.points)


@pytest.mark.parametrize("odds", [float("nan"), float("inf"), -3.0, 0.0, 0.5])
def test_a_wallet_refuses_odds_that_are_not_a_payout(tmp_path, odds):
    from kickthefly.core import points

    w = points.Wallet(tmp_path / "w.json")
    before = w.points
    with pytest.raises(ValueError):
        w.settle(10, odds, True)
    assert w.points == before and w.history == []


def test_a_stake_that_is_not_a_whole_positive_number_is_refused(tmp_path):
    from kickthefly.core import points

    w = points.Wallet(tmp_path / "w.json")
    for stake in (0, -5, 2.5, float("nan"), True):
        with pytest.raises(ValueError):
            w.settle(stake, 2.0, True)
    assert w.points == points.START_POINTS


# --- the three day 4 Lab pages share one background job: a result must land on its own page, whichever page is open -------------------
@pytest.mark.parametrize("open_page", ["lab_netsci", "lab_sensitivity"])
def test_a_finished_sleep_job_is_not_taken_in_by_another_page(menu, open_page):
    from test_day4_pages import draw
    from test_day4_sleepdep import fake_play
    from kickthefly.lab import labday4, sleepdep
    from kickthefly.ui.bgjob import BgJob

    st = labday4._st(menu)
    res = sleepdep.run(list(range(1000, 1004)), play=fake_play)
    st.job = BgJob("Sleep deprivation", lambda job: res).start()
    st.job.thread.join(2)
    draw(menu, open_page)                                   # before: KeyError 'brain' (netsci) or a sleep result shown as a heatmap
    assert st.sleep is res and st.net == {} and st.sens is None and st.job is None and not st.error
    draw(menu, "lab_sleepdep")


def test_a_finished_netsci_job_lands_on_the_netsci_page_even_if_the_sleep_page_is_open(menu):
    from test_day4_pages import draw
    from kickthefly.lab import labday4
    from kickthefly.ui.bgjob import BgJob

    st = labday4._st(menu)
    res = dict(brain="larva", kind="netsci")
    st.job = BgJob("Network science", lambda job: res).start()
    st.job.thread.join(2)
    draw(menu, "lab_sleepdep")
    assert st.net.get("larva") is res and st.sleep is None


# --- Cancel really cancels: no queued work keeps running for minutes after the player pressed Cancel -----------------------------
class FakePool:
    """A stand-in for ProcessPoolExecutor: one worker thread runs the queued tasks in order, skipping any whose Future was cancelled,
    as a real pool does. Records how many tasks ran and how it was shut down."""
    instances: list = []

    def __init__(self, max_workers=None, mp_context=None):
        import queue
        from concurrent.futures import Future

        self._Future, self.q, self.ran, self.shutdown_args = Future, queue.Queue(), 0, None
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()
        FakePool.instances.append(self)

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            f, fn, args = item
            if not f.set_running_or_notify_cancel():
                continue
            self.ran += 1
            try:
                f.set_result(fn(*args))
            except BaseException as e:                       # noqa: BLE001
                f.set_exception(e)

    def submit(self, fn, *args):
        f = self._Future()
        self.q.put((f, fn, args))
        return f

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_args = dict(wait=wait, cancel_futures=cancel_futures)
        if cancel_futures:
            import queue
            while True:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[0].cancel()
        self.q.put(None)
        if wait:
            self.t.join(5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown(wait=True)
        return False


@pytest.fixture
def fake_pool(monkeypatch):
    FakePool.instances = []
    return FakePool


def test_cancelling_the_sleep_assay_stops_the_queued_flies(monkeypatch, fake_pool):
    from test_day4_sleepdep import fake_play
    from kickthefly.lab import sleepdep

    monkeypatch.setattr(sleepdep, "ProcessPoolExecutor", fake_pool)
    monkeypatch.setattr(sleepdep, "_task", fake_play)
    cancel = threading.Event()
    with pytest.raises(RuntimeError, match="cancelled"):
        sleepdep.run(list(range(1000, 1010)), workers=2, cancel=cancel, progress=lambda d, n, label: cancel.set())
    assert fake_pool.instances[0].ran < 5, f"{fake_pool.instances[0].ran} of 20 flies ran after Cancel"


def test_cancelling_a_race_field_measurement_stops_the_queued_cards(monkeypatch, fake_pool):
    import concurrent.futures as cf
    from kickthefly.lab import racing, tournament

    monkeypatch.setattr(racing, "ProcessPoolExecutor", fake_pool)
    monkeypatch.setattr(tournament, "_card_task", lambda args: time.sleep(0.05) or dict(seed=args[0]))
    cancel = threading.Event()
    with pytest.raises(RuntimeError, match="cancelled"):
        racing.measure_field(list(range(3000, 3008)), workers=2, cancel=cancel, progress=lambda d, n, label: cancel.set())
    assert fake_pool.instances[0].ran < 4


def test_cancelling_a_tournament_stops_inside_a_round(fake_pool, monkeypatch):
    from kickthefly.lab import tournament

    cancel = threading.Event()
    played = []

    def play(task):
        if task[0] == "card":
            return dict(seed=task[1], title="t", summary="s", measured=True)
        played.append(task[:2])
        cancel.set()                                     # the player presses Cancel during the first match
        return dict(winner=str(task[0]), hp={str(task[0]): 50.0, str(task[1]): 40.0}, shots={}, hits={}, seconds=1.0, knockout=False,
                    max_levels={}, events=[], frames=[], steps=1, attempt=0, match_seed=1)

    with pytest.raises(RuntimeError, match="cancelled"):
        tournament.run_bracket(list(range(2000, 2016)), play=play, cancel=cancel, drivers=False)
    assert len(played) == 1, f"{len(played)} of the 8 first-round matches ran after Cancel"


def test_cancelling_the_sensitivity_grid_stops_between_cells_and_keeps_the_finished_ones(monkeypatch, tmp_path):
    from test_day4_sensitivity import fake_cell
    from kickthefly.lab import sensitivity as sv

    calls = []

    def run_cell(param, value, seeds, tests, workers=None, progress=None, cancel=None):
        calls.append(sv.cell_key(param, value))
        cancel_ev.set()
        return fake_cell(param, value, seeds, tests, workers, progress)

    cancel_ev = threading.Event()
    monkeypatch.setattr(sv, "run_cell", run_cell)
    with pytest.raises(RuntimeError, match="cancelled"):
        sv.run(params=["noise_std"], tests=["looming_escape"], seeds=(1, 2, 3), folder=tmp_path, cancel=cancel_ev)
    assert calls == [sv.BASELINE]
    saved = json.loads((tmp_path / sv.PROGRESS_NAME).read_text())
    assert list(saved["cells"]) == [sv.BASELINE], "the finished cell is kept for --resume"


def test_cancelling_validation_cancels_the_queued_seeds(monkeypatch, fake_pool):
    from kickthefly.lab import validation

    monkeypatch.setattr(validation, "ProcessPoolExecutor", fake_pool)
    monkeypatch.setattr(validation, "_pathway_seed", lambda s, *a: time.sleep(0.05) or {"_backend": ("cpu", "cpu")})
    cancel = threading.Event()
    with pytest.raises(RuntimeError, match="cancelled"):
        validation.run(seeds=tuple(range(1000, 1010)), workers=2, include={"looming_escape"}, cancel=cancel,
                       progress=lambda d, n, label: cancel.set())
    assert fake_pool.instances[0].ran < 5


def test_the_sensitivity_page_hands_its_cancel_button_to_the_run(menu, monkeypatch):
    from kickthefly.lab import labday4, sensitivity as sv

    got = {}

    def run(**kw):
        got.update(kw)
        raise RuntimeError("cancelled")

    monkeypatch.setattr(sv, "run", run)
    st = labday4._st(menu)
    labday4._start_sens(st)
    st.job.thread.join(5)
    assert got.get("cancel") is st.job.cancel if st.job else False


# --- validation.run(params=...): what ran is what is recorded, and a test that cannot take the parameters says so ------------------
def test_a_run_with_lab_parameters_records_the_parameters_it_ran_with(synthetic_pack):
    from kickthefly.lab import lab, validation

    res = validation.run(seeds=(1, 2), workers=1, include={"looming_escape"}, params={"noise_std": 0.06})
    assert res["lab_params"]["noise_std"] == pytest.approx(0.06), "before: the defaults were recorded whatever ran"
    assert res["lab_params"]["bias"] == lab.DEFAULTS["bias"]
    plain = validation.run(seeds=(1, 2), workers=1, include={"looming_escape"})
    assert plain["lab_params"] == dict(lab.DEFAULTS), "a plain run records exactly what it always did"


@pytest.mark.parametrize("test_id", ["mb_extinction", "mb_second_order", "epg_compass", "epg_compass_wind"])
def test_parameters_are_refused_for_a_test_that_would_silently_ignore_them(test_id, monkeypatch):
    from kickthefly.lab import compass, sensitivity as sv, validation

    def ran(*a, **k):
        raise AssertionError("ran with the parameters silently ignored")

    for mod, name in ((validation, "_extinction_seed"), (validation, "_second_order_seed"), (compass, "probe_epg_compass"),
                      (compass, "probe_epg_wind")):
        monkeypatch.setattr(mod, name, ran)

    with pytest.raises(ValueError, match="parameters"):
        validation.run(seeds=(1, 2), workers=1, include={test_id}, params={"noise_std": 0.06})
    with pytest.raises(sv.SensitivityError):
        sv.run(params=["noise_std"], tests=[test_id], seeds=(1, 2, 3))
