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


# --- network science: the rich-club null keeps the degrees the coefficient is defined on; an enrichment over zero is undefined -------
def test_the_rich_club_null_keeps_every_neurons_undirected_degree(synthetic_pack, monkeypatch):
    from kickthefly.lab import netsci

    seen = []
    real = netsci.rich_club

    def spy(edges, deg, ks):
        seen.append(np.asarray(deg).copy())
        return real(edges, deg, ks)

    monkeypatch.setattr(netsci, "rich_club", spy)
    res = netsci.compute("adult", nulls=2, wedges=2000, seed=1, cache=False)
    assert res["reciprocity"]["reciprocal_pairs"] > 0, "the test needs reciprocal pairs (they are what the directed swaps split)"
    assert len(seen) == 3
    for d in seen[1:]:
        assert np.array_equal(d, seen[0]), "phi(k) and phi_null(k) must count neurons of the same skeleton degree"


def test_undirected_swaps_keep_degrees_and_make_no_loop_or_repeat():
    from kickthefly.lab import netsci

    rng = np.random.default_rng(3)
    n = 60
    a = rng.random((n, n)) < 0.15
    a = np.triu(a | a.T, 1)
    lo, hi = np.nonzero(a)
    nlo, nhi = netsci.rewire_undirected(lo.astype(np.int64), hi.astype(np.int64), n, np.random.default_rng(0))
    deg = lambda x, y: np.bincount(np.concatenate([x, y]), minlength=n)          # noqa: E731
    assert np.array_equal(deg(lo, hi), deg(nlo, nhi))
    assert np.all(nlo != nhi) and len(np.unique(np.minimum(nlo, nhi) * n + np.maximum(nlo, nhi))) == len(nlo)
    assert len(set(zip(lo.tolist(), hi.tolist())) & set(zip(np.minimum(nlo, nhi).tolist(), np.maximum(nlo, nhi).tolist()))) < 0.6 * len(lo)


def test_a_motif_the_null_never_produced_has_an_undefined_enrichment_with_its_reason(synthetic_pack, monkeypatch, tmp_path):
    from kickthefly.lab import netsci

    real = netsci.motif_estimate
    calls = [0]

    def est(*a, **k):
        r = real(*a, **k)
        calls[0] += 1
        if calls[0] > 1:                                   # every null graph: none of the all-mutual triad
            r["counts"][12] = 0.0
        return r

    monkeypatch.setattr(netsci, "motif_estimate", est)
    res = netsci.compute("adult", nulls=2, wedges=2000, seed=1, cache=False)
    m = res["motifs"][12]
    assert m["motif"] == "300" and m["enrichment"] is None and m["z"] is None
    assert "undefined" in m["enrichment_note"] and "2 null graphs" in m["enrichment_note"]
    text = netsci.summary(res)
    assert "nan" not in text.lower() and "undefined" in text
    netsci.export_csv(res, tmp_path)
    rows = (tmp_path / "netsci_adult_motifs.csv").read_text().splitlines()
    assert rows[0].endswith(",note") and "undefined" in rows[13] and "nan" not in rows[13].lower()
    json.dumps(m, allow_nan=False)                           # the cached/exported JSON has no NaN for this motif any more


def test_the_netsci_page_draws_an_undefined_enrichment(menu, synthetic_pack, monkeypatch):
    from test_day4_pages import draw
    from kickthefly.lab import labday4, netsci

    res = netsci.compute("adult", nulls=2, wedges=2000, seed=1, cache=False)
    res["motifs"][12] = dict(res["motifs"][12], enrichment=None, z=None, enrichment_note="undefined: none of the 2 null graphs had this motif")
    st = labday4._st(menu)
    st.net["adult"], st.cached["adult"] = res, True
    draw(menu, "lab_netsci")


# --- decision A: the game shows only MEASURED personality cards, labelled; a seed-drawn card is never shown as a measurement --------
def test_a_card_without_readouts_says_it_was_drawn_from_the_seed_not_measured():
    from kickthefly.core.individuality import compute_personality_card

    drawn = compute_personality_card(123)
    assert drawn["measured"] is False and "drawn from the seed" in drawn["source"]
    real = compute_personality_card(123, looming_latency=0.1, turning_ratio=1.0, sugar_ratio=2.0)
    assert real["measured"] is True and real["title"] == "Skittish Straight-walker"


def _slot(seed, mode, isolated=True):
    from kickthefly.core import simcore
    from kickthefly.game import kick_the_fly as k

    br = simcore.new_brain(seed=seed, individuality=mode, warmup=5)
    return k.FlySlot(k.Fly(300.0), br, seed=seed)


def test_a_fly_in_play_shows_card_not_measured_until_its_own_card_is_measured(isolated_home, synthetic_pack):
    from kickthefly.core import cards
    from kickthefly.lab import tournament

    slot = _slot(7, "subtle")
    assert slot.personality["measured"] is False and cards.label(slot.personality) == "card not measured"
    assert "Not measured yet" in slot.personality["summary"] and not slot.personality["traits"]
    card = tournament.measure_card(7, "subtle", warmup=100)
    assert cards.store(card) == 1
    slot.refresh_card()
    assert slot.personality["measured"] is True and slot.personality["loom_latency_s"] == card["loom_latency_s"]
    assert cards.label(slot.personality).endswith("(measured)")
    other = _slot(7, "off")                                   # same seed, other individuality: not the same fly, not its card
    assert other.personality["measured"] is False
    noisy = _slot(7, "subtle")
    noisy.brain.sim.p.noise_std = 0.09                         # a fly running other LIF parameters is not the fly that was measured
    noisy.refresh_card()
    assert noisy.personality["measured"] is False


def test_the_card_cache_survives_damage_and_ignores_another_version(isolated_home, synthetic_pack):
    from kickthefly.core import cards

    card = cards.stamp(dict(seed=3, mode="subtle", measured=True, title="Bold Right-turner", summary="s", traits=[], loom_latency_s=0.3))
    cards.store(card)
    assert cards.lookup(3, "subtle")["title"] == "Bold Right-turner"
    cards.cache_file().write_text("{broken", encoding="utf-8")
    assert cards.lookup(3, "subtle") is None and cards.for_fly(3, "subtle")["measured"] is False
    cards.cache_file().write_text(json.dumps(dict(version=999, cards={card["card_key"]: card})), encoding="utf-8")
    assert cards.lookup(3, "subtle") is None
    assert cards.store(dict(seed=3, mode="subtle", measured=False, title="drawn")) == 0, "an unmeasured card is never cached"


def test_measured_cards_from_a_tournament_reach_the_flies_in_play(isolated_home, synthetic_pack):
    from kickthefly.core import cards
    from kickthefly.lab import tournament

    got = tournament.measure_cards([11, 12], "subtle")
    assert {s: cards.lookup(s, "subtle")["loom_latency_s"] for s in (11, 12)} == {s: got[s]["loom_latency_s"] for s in (11, 12)}


OLD_PET = {"seed": 4242, "brain_type": "adult", "born_time": 1.0e9, "last_saved_time": 1.0e9, "hunger": 0.3, "sleep_pressure": 0.2,
           "is_sleeping": False, "mood": 0.6, "real_stakes": False, "is_dead": False, "timeline": [],
           "personality_card": {"seed": 4242, "mode": "subtle", "title": "Bold Right-turner", "summary": "Bold · Right-turner · Sugar lover",
                                "traits": ["Bold", "Right-turner", "Sugar lover"], "metrics": {"looming_latency": 0.31}}}


def test_an_old_pet_file_keeps_its_drawn_card_as_legacy_and_shows_card_not_measured(isolated_home, synthetic_pack, tmp_path):
    from kickthefly.core import cards, pet as pet_mod
    from kickthefly.game import kick_the_fly as k

    d = tmp_path / "pet"
    d.mkdir()
    (d / "pet.ktfsave").write_text(json.dumps(OLD_PET), encoding="utf-8")
    p = pet_mod.PetManager(d)
    p.load_or_create()
    assert p.personality_card == {} and p.legacy_personality_card["title"] == "Bold Right-turner"
    slot = _slot(7, "subtle")                                 # the pet fly: built from the brain seed (7), not pet.seed (4242)
    assert k.pet_card(p, slot) is None and cards.label(k.pet_card(p, slot)) == "card not measured"
    p.save()
    saved = json.loads((d / "pet.ktfsave").read_text())
    assert saved["legacy_personality_card"]["title"] == "Bold Right-turner", "the old card is never deleted"


def test_the_pet_shows_and_keeps_the_measured_card_of_its_own_fly(isolated_home, synthetic_pack, tmp_path):
    from kickthefly.core import cards, pet as pet_mod
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import tournament

    p = pet_mod.PetManager(tmp_path / "pet")
    p.create_new_pet(seed=4242)
    assert p.personality_card == {}, "a new pet starts without a drawn card"
    slot = _slot(7, "subtle")
    card = tournament.measure_card(7, "subtle", warmup=100)
    cards.store(card)
    slot.refresh_card()
    assert k.pet_card(p, slot)["card_key"] == card["card_key"] and p.personality_card["card_key"] == card["card_key"]
    p.save()
    cards.cache_file().unlink()                               # the cache is gone: the pet file still carries its fly's card
    again = pet_mod.PetManager(tmp_path / "pet")
    again.load_or_create()
    slot2 = _slot(7, "subtle")
    assert slot2.personality["measured"] is False
    assert k.pet_card(again, slot2)["loom_latency_s"] == card["loom_latency_s"] and slot2.personality["measured"] is True
    stranger = _slot(8, "subtle")                             # a pet file's card never goes to a different fly
    assert k.pet_card(again, stranger) is None


def test_the_arcade_measures_the_flies_in_play(menu, isolated_home, synthetic_pack, monkeypatch):
    from test_day4_pages import draw, hit
    from kickthefly.lab import tournament
    from kickthefly.ui import arcade_ui

    slots = [_slot(7, "subtle"), _slot(8, "subtle")]
    menu.host.flies = slots
    draw(menu, "arcade")
    assert hit(menu, "arcade_measure_play")["enabled"]
    seen = []

    def fake(seeds, mode, backend=None, workers=1, pool=None, progress=None, cancel=None, sigma=None, params=None):
        seen.append((tuple(seeds), mode, params))
        from kickthefly.core import cards
        out = {s: tournament.measure_card(s, mode, warmup=100, sigma=sigma, params=params) for s in seeds}
        cards.store(list(out.values()))
        return out

    monkeypatch.setattr(tournament, "measure_cards", fake)
    st = arcade_ui.state(menu)
    arcade_ui._start_measure_play(menu, st)
    st.job.thread.join(120)
    draw(menu, "arcade")
    assert seen and seen[0][0] == (7, 8) and seen[0][1] == "subtle" and seen[0][2]["noise_std"] == pytest.approx(0.05)
    assert all(s.personality["measured"] for s in slots) and not hit(menu, "arcade_measure_play")["enabled"]


# --- decision B: R4 holds the brain state fixed and varies only the individual (the analysis on fakes; the plumbing on the synthetic pack)
def _r4_play(speed_of_fly, speed_of_state, noise_sd=1.0):
    def play(task):
        race, state_seed, iseed, repeat, mode, backend, cap = task
        rng = np.random.default_rng(hash((iseed, repeat)) & 0xFFFF)
        t = 40.0 + speed_of_state(state_seed) + speed_of_fly(iseed, mode) + rng.normal(0, noise_sd)
        return race, iseed, repeat, dict(finish_s=t, distance=8.0, seconds=t)
    return play


def test_r4_design_is_eight_races_of_six_with_one_state_seed_each():
    from kickthefly.lab import racing

    d = racing.r4_design()
    assert len(d) == 8 and all(len(x["individuality_seeds"]) == 6 for x in d)
    assert [x["state_seed"] for x in d] == list(range(1000, 1008))
    assert sorted(i for x in d for i in x["individuality_seeds"]) == list(range(1000, 1048))


def test_r4_ignores_what_the_shared_brain_state_does_to_speed():
    from kickthefly.lab import racing

    state_only = _r4_play(lambda i, m: 0.0, lambda s: 10.0 * (s - 1000))           # the off control's situation in R1: state, not individual
    sub = racing.run_r4("subtle", play=state_only)
    off = racing.run_r4("off", play=state_only)
    an = racing.analyze_r4(sub, off, n_perm=2000, n_boot=2000)
    assert abs(an["r_within_subtle"]) < 0.35 and not an["passed"]
    from scipy import stats
    pooled = stats.spearmanr([f["times"][0] for f in sub["flies"]], [f["times"][1] for f in sub["flies"]])[0]
    assert pooled > 0.9, "R1's pooled statistic would call this repeatable: R4's within-race one does not"


def test_r4_finds_a_fly_speed_that_comes_from_its_individuality_and_not_from_the_off_control():
    from kickthefly.lab import racing

    rng = np.random.default_rng(5)
    own = {i: rng.normal(0, 4.0) for i in range(1000, 1048)}
    play = _r4_play(lambda i, m: own[i] if m != "off" else 0.0, lambda s: 3.0 * (s - 1000))
    an = racing.analyze_r4(racing.run_r4("subtle", play=play), racing.run_r4("off", play=play), n_perm=2000, n_boot=2000)
    assert an["passed"] and an["r_within_subtle"] > 0.6 and an["perm_p_subtle"] < 0.01 and an["ci_difference"][0] > 0
    assert abs(an["r_within_off"]) < 0.4
    text = racing.summary_r4(an)
    assert "R4 PASS" in text and "nan" not in text


def test_r4_files_and_the_headless_flag(tmp_path, monkeypatch, capsys):
    import argparse
    from kickthefly.lab import racing

    play = _r4_play(lambda i, m: 0.0, lambda s: 0.0)
    real = racing.run_r4
    monkeypatch.setattr(racing, "run_r4", lambda mode, **kw: real(mode, play=play))
    assert racing.main_r4(argparse.Namespace(workers=1, out=str(tmp_path))) == 0
    assert {p.name for p in tmp_path.iterdir()} == {"race_r4.json", "race_r4_flies.csv", "race_r4_analysis.csv"}
    assert "R4 FAIL" in capsys.readouterr().out
    assert len((tmp_path / "race_r4_flies.csv").read_text().splitlines()) == 1 + 96


def test_the_individuality_seed_can_differ_from_the_state_seed_and_defaults_to_it(synthetic_pack):
    from kickthefly.core import individuality, simcore

    a = simcore.new_brain(seed=5, individuality="subtle", warmup=0)
    b = simcore.new_brain(seed=5, individuality="subtle", individuality_seed=9, warmup=0)
    n = a.sim.n
    assert np.array_equal(a.sim.d_pre, individuality.compute_fly_gains(5, n, "subtle")[0])
    assert np.array_equal(b.sim.d_pre, individuality.compute_fly_gains(9, n, "subtle")[0])
    assert np.array_equal(a.sim._noise, b.sim._noise), "the noise (state) seed is still 5"
    c = simcore.new_brain(seed=5, individuality="off", individuality_seed=9, warmup=0)
    assert c.sim.d_pre is None


# --- the test suite's memory: a game's brain-view thread kept the game and every brain alive after its test file ended ----------
def test_a_built_game_is_freed_by_the_suites_shutdown_after_its_file(synthetic_pack):
    import gc
    import weakref

    import pygame
    from conftest import shutdown_live_games
    from kickthefly.core import config, memory as mem_mod, simcore
    from kickthefly.game import kick_the_fly as k2
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    pygame.init()
    g, W, soma = simcore.pack()
    sim = LIFSim(None, LIFParams(), W_in=W, seed=1)
    br = k2.Brain(g, sim, seed=1)
    br.graph = g
    br.memory = mem_mod.Memory(g, sim, load=False)
    game = k2.Game(None, br, k2.BrainView(soma, W, np.zeros(g.n, bool)), graph=g, weights=W, cfg=config.Config())
    ref_game, ref_brain = weakref.ref(game), weakref.ref(br)
    del game, br, sim
    gc.collect()
    assert ref_game() is not None, "while its brain-view thread runs, the game is alive (this is what kept 1-1.8 GB per file)"
    assert shutdown_live_games() >= 1
    gc.collect()
    assert ref_game() is None and ref_brain() is None, "after the shutdown the game and its brain can be freed"
    assert not [t for t in threading.enumerate() if t.name == "brain-view" and t.is_alive()]


def test_a_lab_rule_a_test_changes_is_put_back_for_the_next_test():
    from conftest import restore_rule_globals, snapshot_rule_globals
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import lab

    lab.apply_rules(lab.DEFAULTS)
    snap = snapshot_rule_globals()
    lab.apply_rules({**lab.DEFAULTS, "thresh.escape": 5.5, "loom_min": 2.5})
    assert k.THRESH["escape"] == 5.5 and k.LOOM_MIN == 2.5
    restore_rule_globals(snap)
    assert k.THRESH["escape"] == 4.0 and k.LOOM_MIN == 1.5


def test_a_headless_run_that_switches_the_network_off_does_not_leak_into_the_next_test():
    from conftest import restore_rule_globals, snapshot_rule_globals
    from kickthefly.core import netguard

    netguard.enable()
    snap = snapshot_rule_globals()
    netguard.disable("a headless run")
    assert netguard._disabled
    restore_rule_globals(snap)
    assert netguard._disabled is None


# --- test time: tools/run_tests.py shards the long files so no chunk waits on one file ---------------------------------------------
def test_long_files_are_sharded_and_every_test_runs_exactly_once():
    import importlib.util
    from pathlib import Path

    from conftest import shard_keep

    spec = importlib.util.spec_from_file_location("run_tests", Path(__file__).resolve().parents[1] / "tools" / "run_tests.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    durations = {"a.py": 600.0, "b.py": 270.0, "c.py": 30.0, "d.py": 20.0}
    items, est = rt.plan_shards(list(durations), durations, 4)
    assert sum(1 for i in items if i.startswith("a.py")) > 1 and "c.py" in items
    assert abs(sum(est.values()) - sum(durations.values())) < 1e-6
    chunks = rt.deal(items, est, 4)
    assert max(sum(est[i] for i in c) for c in chunks) < 600.0, "the longest chunk is shorter than the longest file"
    seen = [0] * 37
    for c in chunks:
        files, shards = rt.chunk_args(c)
        if "a.py" in files:
            for i, k in enumerate(shard_keep(37, shards["a.py"]["n"], shards["a.py"]["keep"])):
                seen[i] += k
    assert seen == [1] * 37, "each of the file's tests runs in exactly one chunk"


# --- the race's speed rule version 2: a fly's speed no longer depends on the calm baseline its warm-up happened to set -------------
def test_a_flys_race_does_not_depend_on_the_calm_baseline_its_warm_up_left(synthetic_pack):
    from kickthefly.core import simcore
    from kickthefly.game import flyrace

    def race(base_scale):
        br = simcore.new_brain(seed=3, individuality="off", warmup=50)
        br.base[:] = br.base * base_scale                   # the same brain, as if its warm-up had set another baseline
        br.reseed(11)
        return flyrace.run_lane(br, cap_s=4.0)

    a, b = race(1.0), race(3.0)
    assert a["distance"] == b["distance"] and a["max_walk_level"] == b["max_walk_level"]
    assert a["rule_version"] == 2
