"""3.0 day 4, Fly racing: the lane engine (game/flyrace.py) on a scripted brain, the odds and the bet (lab/racing.py, core/points.py:
points only), the pre-registered analysis, and a real-code smoke test on the SYNTHETIC pack (plumbing, not biology)."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.core import points
from kickthefly.game import flyrace
from kickthefly.lab import racing


class FakeBrain:
    def __init__(self, level=3.0):
        self.level_walk, self.dt, self.pokes = level, 0.005, []

    def poke(self, region, side, strength, recruit=None):
        self.pokes.append((region, side, round(strength, 3)))

    def _step(self):
        pass

    def level(self, name):
        return self.level_walk if name == "walk" else 1.0

    def hz(self, name):                                  # speed rule version 2 reads DNp09's rate against the fixed reference
        return self.level_walk * flyrace.WALK_REF_HZ if name == "walk" else flyrace.WALK_REF_HZ


def test_speed_rule_is_linear_up_to_the_games_walk_threshold_and_capped():
    from kickthefly.game import kick_the_fly as k

    assert flyrace.speed_of(k.THRESH["walk"], k.THRESH["walk"]) == flyrace.V_MAX
    assert flyrace.speed_of(1.0, 3.0) == pytest.approx(flyrace.V_MAX / 3) and flyrace.speed_of(99, 3.0) == flyrace.V_MAX and flyrace.speed_of(-1, 3.0) == 0


def test_a_fly_at_full_walking_drive_finishes_in_track_over_vmax():
    r = flyrace.run_lane(FakeBrain(3.0))
    assert r["finish_s"] == pytest.approx(flyrace.TRACK_LENGTH / flyrace.V_MAX, abs=0.1) and r["distance"] == flyrace.TRACK_LENGTH
    assert r["trace"][0] == [0.0, 0.0] and r["trace"][-1][1] == flyrace.TRACK_LENGTH and r["lures_touched"] == [0, 1, 2]
    slow = flyrace.run_lane(FakeBrain(1.5))
    assert slow["finish_s"] == pytest.approx(2 * r["finish_s"], rel=0.02)


def test_a_fly_that_does_not_move_does_not_finish_and_the_cap_ends_the_race():
    r = flyrace.run_lane(FakeBrain(0.0), cap_s=3.0)
    assert r["finish_s"] is None and r["distance"] == 0.0 and r["seconds"] == pytest.approx(3.0, abs=0.05)


def test_lures_are_smelled_ahead_and_tasted_on_contact_with_the_games_own_pulses():
    b = FakeBrain(3.0)
    flyrace.run_lane(b)
    scents = [p for p in b.pokes if p[0] == "scent"]
    assert {p[1] for p in scents} == {"sugar", "fruit"} and max(p[2] for p in scents) <= flyrace.SCENT_POKE + 0.05 + 1e-6
    assert any(p[0] == "taste" and p[2] == flyrace.TASTE_POKE for p in b.pokes) and any(p[0] == "reward" and p[2] == flyrace.REWARD_POKE for p in b.pokes)


def test_a_lure_behind_the_fly_is_not_smelled():
    b = FakeBrain(3.0)
    flyrace.run_lane(b, lures=((0.5, "sugar"),), length=4.0)
    first_past = next(i for i, p in enumerate(b.pokes) if p[0] == "taste")
    # after the fly has passed the lure at 0.5 m there are no more scent pokes
    assert all(p[0] != "scent" for p in b.pokes[first_past + 40:])


def test_finishing_order_puts_finishers_first_by_time_then_the_rest_by_distance():
    res = {1: dict(finish_s=10.0, distance=8), 2: dict(finish_s=None, distance=6.0), 3: dict(finish_s=9.0, distance=8), 4: dict(finish_s=None, distance=7.0)}
    assert flyrace.order(res) == [3, 1, 4, 2]


def test_cancel_stops_a_lane():
    import threading

    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        flyrace.run_lane(FakeBrain(), cancel=ev)


# --- the odds and the points --------------------------------------------------------------------------------------------------------
def card(seed, sugar, walk):
    return dict(seed=seed, title=f"Fly {seed}", sugar_ratio=sugar, walk_level_calm=walk, loom_latency_s=0.2, turning_log_ratio=0.0, mode="subtle",
                summary="", traits=[], measured=True)


def test_the_odds_are_probabilities_that_rank_by_form_and_pay_a_fair_price_less_the_take():
    cards = [card(1, 1.5, 0.9), card(2, 2.0, 1.1), card(3, 2.5, 1.3), card(4, 1.8, 1.0)]
    p = racing.win_probabilities(cards)
    assert p.sum() == pytest.approx(1.0) and np.argmax(p) == 2 and np.argmin(p) == 0
    table = racing.odds_table(cards)
    assert table[2]["decimal_odds"] < table[0]["decimal_odds"] and all(t["decimal_odds"] >= points.MIN_ODDS for t in table)
    assert points.decimal_odds(0.5) == pytest.approx(1.8) and points.decimal_odds(0.9) == points.MIN_ODDS and points.decimal_odds(1e-9) > 1000
    assert sum(1 / t["decimal_odds"] for t in table) < 1.2, "the take is a modest margin, not a different game"


def test_a_field_with_identical_flies_gets_even_odds():
    p = racing.win_probabilities([card(i, 2.0, 1.0) for i in range(5)])
    assert np.allclose(p, 0.2)


def test_a_wallet_pays_stake_times_odds_on_a_win_and_takes_the_stake_on_a_loss(tmp_path):
    w = points.Wallet(tmp_path / "w.json")
    assert w.points == points.START_POINTS
    assert w.settle(10, 3.0, True) == 20 and w.points == 120
    assert w.settle(25, 3.0, False) == -25 and w.points == 95
    again = points.Wallet(tmp_path / "w.json")
    assert again.points == 95 and len(again.history) == 2, "points persist"


def test_a_wallet_refuses_a_bet_it_cannot_cover_and_tops_up_for_free_when_nearly_empty(tmp_path):
    w = points.Wallet(tmp_path / "w.json")
    with pytest.raises(ValueError):
        w.settle(1000, 2.0, True)
    w.points = 8
    w.settle(5, 2.0, False)
    assert w.points == points.TOP_UP_TO and w.history[-1]["label"] == "free top-up"
    assert not w.can_bet(0) and not w.can_bet(-3) and w.can_bet(5)


def test_a_damaged_wallet_file_starts_over_and_a_read_only_folder_does_not_crash(tmp_path):
    p = tmp_path / "w.json"
    p.write_text("{not json")
    assert points.Wallet(p).points == points.START_POINTS
    w = points.Wallet(tmp_path / "missing" / "dir" / "w.json")
    w.settle(5, 2.0, True)                                 # creates the folder; and if it could not, nothing raises


def test_there_is_no_money_and_no_network_anywhere_in_the_arcade_code():
    import inspect

    from kickthefly.core import points as pts
    from kickthefly.game import flyduel, flyrace as fr
    from kickthefly.lab import racing as rc, tournament as tnm

    for mod in (pts, fr, rc, tnm, flyduel):
        src = inspect.getsource(mod)
        for word in ("import socket", "urllib", "requests", "http.client", "websocket", "stripe", "paypal", "checkout", "payment_method"):
            assert word not in src, f"{mod.__name__} mentions {word!r}"
    assert "not money" in pts.__doc__ and "no code path" in pts.__doc__


# --- a race, with a fake lane player -------------------------------------------------------------------------------------------------
def make_play(speed_of_seed):
    def play(task):
        if task[0] == "card":
            s = task[1]
            return card(s, 1.5 + 0.1 * (s % 7), 1.0 + 0.05 * (s % 5))
        seed, repeat = task[0], task[1]
        t = 8.0 / speed_of_seed(seed, repeat)
        return seed, repeat, dict(finish_s=round(t, 3), distance=8.0, trace=[[0, 0], [round(t, 2), 8.0]], lures_touched=[0], max_walk_level=2.0, seconds=t)
    return play


def test_a_race_orders_the_field_and_computes_odds_for_it():
    seeds = [11, 12, 13, 14]
    race = racing.run_race(seeds, repeats=2, play=make_play(lambda s, r: 0.1 * s), cap_s=60.0)
    assert race["orders"][0] == [14, 13, 12, 11] and race["winner"] == 14 and set(race["odds"]) == {"11", "12", "13", "14"}
    assert abs(sum(v["p_win"] for v in race["odds"].values()) - 1) < 1e-9 and race["tags"]["order_and_prediction"] == "MODEL PREDICTION"


def test_a_race_needs_two_to_eight_distinct_lanes():
    for bad in ([1], list(range(9)), [1, 1, 2]):
        with pytest.raises(ValueError):
            racing.run_race(bad, play=make_play(lambda s, r: 1.0))


def test_a_bet_is_settled_against_the_winner_in_points(tmp_path):
    race = racing.run_race([11, 12, 13], play=make_play(lambda s, r: 0.1 * s))
    w = points.Wallet(tmp_path / "w.json")
    out = racing.settle_bets(race, [dict(fly=13, stake=10), dict(fly=11, stake=5)], w)
    assert out[0]["won"] and not out[1]["won"] and out[1]["delta"] == -5
    assert w.points == points.START_POINTS + out[0]["delta"] - 5 and out[0]["delta"] == round(10 * race["odds"]["13"]["decimal_odds"]) - 10


# --- the analysis -------------------------------------------------------------------------------------------------------------------------
def _races(speed, n_races=6, lanes=6, seed0=100):
    races = []
    for i in range(n_races):
        seeds = list(range(seed0 + i * lanes, seed0 + (i + 1) * lanes))
        races.append(racing.run_race(seeds, repeats=2, play=make_play(speed), race_seed=i, cap_s=60.0))
    return races


def test_a_fly_that_is_just_as_fast_every_time_is_repeatable_and_the_form_score_finds_it_when_it_matters():
    # speed depends on the fly (its seed) only: repeatable. It is also what the form score is built from (sugar 1.5 + 0.1 (s % 7), walk 1.0 + 0.05 (s % 5))
    an = racing.analyze(_races(lambda s, r: 0.2 + 0.1 * (s % 7) + 0.05 * (s % 5)))
    r1 = next(t for t in an["tests"] if t["id"] == "R1")
    r2 = next(t for t in an["tests"] if t["id"] == "R2")
    assert r1["rho"] > 0.99 and r1["significant"] and r2["rho"] < -0.5 and r2["significant"] and an["individuality_predicts_finish"]


def test_a_race_decided_by_noise_alone_shows_no_repeatability():
    rng = np.random.default_rng(9)
    noisy = {}

    def speed(s, r):
        return noisy.setdefault((s, r), 0.3 + rng.random())

    an = racing.analyze(_races(speed, n_races=8))
    r1 = next(t for t in an["tests"] if t["id"] == "R1")
    assert abs(r1["rho"]) < 0.5 and not r1["significant"]


def test_the_secondary_trait_table_and_the_summary_and_files(tmp_path):
    races = _races(lambda s, r: 0.2 + 0.1 * (s % 7))
    an = racing.analyze(races)
    assert {x["trait"] for x in an["secondary"]} == set(racing.TRAITS)
    text = racing.summary(races, an)
    assert "DOES INDIVIDUALITY PREDICT THE FINISH" in text and "Points only" in text and "R1" in text and "R3" in text
    files = racing.save(races, an, tmp_path)
    assert {f.name for f in files} == {"race.json", "race_flies.csv", "race_analysis.csv"}
    assert len((tmp_path / "race_flies.csv").read_text().splitlines()) == 1 + 36


def test_an_unfinished_fly_keeps_its_place_by_its_average_speed():
    assert racing._effective_time(dict(finish_s=None, seconds=40.0, distance=4.0), 8.0) == pytest.approx(80.0)
    assert racing._effective_time(dict(finish_s=12.5, seconds=12.5, distance=8.0), 8.0) == 12.5


# --- the real code on the synthetic pack (plumbing) -------------------------------------------------------------------------------------
def test_a_real_lane_runs_and_is_deterministic_for_a_seed(synthetic_pack):
    from kickthefly.lab import tournament as tn

    a = flyrace.run_lane(tn.build_fighter(5, "subtle", 3), cap_s=4.0)
    b = flyrace.run_lane(tn.build_fighter(5, "subtle", 3), cap_s=4.0)
    assert a == b and a["seconds"] == pytest.approx(4.0, abs=0.05) and a["distance"] >= 0
    c = flyrace.run_lane(tn.build_fighter(5, "subtle", 4), cap_s=4.0)
    assert c["max_walk_level"] >= 0
