"""3.0 day 4, Fly tournament: the duel engine (game/flyduel.py) on scripted fake brains, the bracket logic (lab/tournament.py) with
a fake match player, the pre-registered analysis, and a real-code smoke test on the SYNTHETIC pack (plumbing, not biology)."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.game import flyduel
from kickthefly.lab import tournament as tn


class FakeBrain:
    """The brain interface the duel uses (poke, level, hz, _step, sim.spikes, types, n, dt), scripted: `levels` are the group levels
    it reports; it turns toward whichever side it was last poked with a tracking stimulus (what the real wiring does with LC10)."""

    def __init__(self, fire=0.0, escape=0.0, run=0.0, walk=0.0, aim=True, n=5):
        self.levels = dict(fire=fire, escape=escape, run=run, walk=walk)
        self.aim, self.n, self.dt = aim, n, 0.005
        self.types = np.array(["DNp35", "DNa01", "LC10a", "SNta01", "x"])[:n]
        self.sim = type("S", (), {"spikes": np.zeros(n, bool)})()
        self.pokes, self.step, self.track = [], 0, ("", -99)

    def poke(self, region, side, strength, recruit=None):
        self.pokes.append((region, side, strength))
        if region == "track":
            self.track = (side, self.step)

    def _step(self):
        self.step += 1
        self.sim.spikes = np.zeros(self.n, bool)
        self.sim.spikes[self.step % self.n] = True

    def level(self, name):
        return self.levels.get(name, 0.0)

    def hz(self, name):
        if name in ("turn_r", "turn_l") and self.aim and self.step - self.track[1] < 12:
            return 12.0 if (name == "turn_r") == (self.track[0] == "R") else 0.0
        return 0.0


def duel(a, b, seed=0, seconds=12.0, **kw):
    return flyduel.run_duel(a, b, seed=seed, seconds=seconds, names=("A", "B"), **kw)


# --- the engine ---------------------------------------------------------------------------------------------------------------------
def test_a_fly_that_shoots_and_aims_beats_one_that_does_nothing():
    r = duel(FakeBrain(fire=5.0), FakeBrain())
    assert r["winner"] == "A" and r["hp"]["B"] < 100 and r["hits"]["A"] >= 1 and r["hits"]["B"] == 0 and r["shots"]["B"] == 0
    assert any(e[0] == "hit" and e[2] == "A" for e in r["events"]) and r["frames"], "no hit or no replay frames"


def test_the_match_is_deterministic_for_a_seed_and_the_seed_changes_the_start():
    a, b = duel(FakeBrain(fire=5.0), FakeBrain(fire=5.0), seed=4), duel(FakeBrain(fire=5.0), FakeBrain(fire=5.0), seed=4)
    assert a["hp"] == b["hp"] and a["events"] == b["events"] and a["frames"] == b["frames"]
    c = duel(FakeBrain(fire=5.0), FakeBrain(fire=5.0), seed=5)
    assert c["frames"][0] != a["frames"][0], "a new seed should start the flies facing differently"


def test_two_flies_that_do_nothing_draw():
    r = duel(FakeBrain(), FakeBrain(), seconds=3.0)
    assert r["winner"] is None and r["hp"] == {"A": 100.0, "B": 100.0} and not r["knockout"]


def test_the_match_ends_at_a_knockout():
    r = duel(FakeBrain(fire=5.0), FakeBrain(), seconds=60.0)
    assert r["knockout"] and r["hp"]["B"] == 0 and r["seconds"] < 60.0 and r["winner"] == "A"
    hits = r["hits"]["A"]
    assert hits == int(np.ceil(100.0 / flyduel.PELLET_DAMAGE))


def test_a_fly_whose_giant_fiber_fires_dodges_incoming_pellets():
    r = duel(FakeBrain(fire=5.0), FakeBrain(escape=9.0), seconds=10.0)
    assert any(e[0] == "dodge" and e[2] == "B" for e in r["events"])
    r2 = duel(FakeBrain(fire=5.0), FakeBrain(), seconds=10.0)
    assert r["hits"]["A"] < r2["hits"]["A"], "dodging should avoid some hits"


def test_a_hurt_fly_runs_when_its_body_touch_group_says_so_and_a_hit_pokes_the_touch_neurons():
    b = FakeBrain(run=9.0)
    r = duel(FakeBrain(fire=5.0), b, seconds=8.0)
    assert any(e[0] == "run" and e[2] == "B" for e in r["events"])
    assert any(p[0] == "body" for p in b.pokes) and any(p[0] == "punish" for p in b.pokes)


def test_the_shooter_is_rewarded_and_the_one_hit_is_punished_as_in_the_duel_against_you():
    a, b = FakeBrain(fire=5.0), FakeBrain()
    duel(a, b, seconds=6.0)
    assert any(p[0] == "reward" for p in a.pokes) and not any(p[0] == "reward" for p in b.pokes)


def test_what_a_fly_sees_of_the_other_is_the_duels_own_tracking_and_small_object_input():
    a, b = FakeBrain(), FakeBrain()
    duel(a, b, seconds=2.0)
    regions = {p[0] for p in a.pokes}
    assert "track" in regions and {p[1] for p in a.pokes if p[0] == "track"} <= {"L", "R"}


def test_looming_pellets_poke_the_looming_detectors_only_when_incoming():
    a, b = FakeBrain(fire=5.0), FakeBrain()
    duel(a, b, seconds=6.0)
    assert any(p[0] == "loom" for p in b.pokes) and not any(p[0] == "loom" for p in a.pokes)


def test_fliers_stay_inside_the_arena():
    r = duel(FakeBrain(fire=5.0, walk=9.0, run=9.0, escape=9.0), FakeBrain(fire=5.0, walk=9.0), seconds=10.0)
    for fr in r["frames"]:
        assert abs(fr[1]) <= flyduel.ARENA_HALF + 1e-9 and abs(fr[2]) <= flyduel.ARENA_HALF + 1e-9 and abs(fr[5]) <= flyduel.ARENA_HALF + 1e-9


def test_the_drivers_report_counts_spikes_before_landed_shots_only():
    a, b = FakeBrain(fire=5.0), FakeBrain()
    r = duel(a, b, seconds=8.0, drivers=True)
    d = r["_drivers"]["A"]
    assert d["hit_windows"] == r["hits"]["A"] and d["steps"] == r["steps"] and d["base"].sum() == r["steps"]
    # the windows are 40 steps long, one spike per step in the fake: every landed shot's window holds 40 spikes
    assert d["hit"].sum() <= d["hit_windows"] * flyduel.WINDOW_STEPS and d["hit"].sum() >= (d["hit_windows"] - 1) * flyduel.WINDOW_STEPS
    both = flyduel.combine_drivers([d, d])
    assert both["hit_windows"] == 2 * d["hit_windows"] and both["steps"] == 2 * d["steps"]


def test_top_drivers_ranks_by_the_ratio_and_ignores_types_with_too_little_data():
    sums = dict(types=np.array(["a", "b", "c", "d"]), hit=np.array([100.0, 40.0, 2.0, 500.0]), base=np.array([1000.0, 1000.0, 1000.0, 20.0]),
                n=np.array([10, 10, 10, 10]), hit_windows=10, steps=10_000, dt=0.005)
    rows = flyduel.top_drivers(sums)
    assert [r["type"] for r in rows] == ["a", "b"], "d has too few overall spikes and c too few before shots"
    assert rows[0]["ratio"] == pytest.approx((100 / (10 * 0.2 * 10)) / (1000 / (10_000 * 0.005 * 10)))
    assert flyduel.top_drivers(dict(sums, hit_windows=1)) == [] and flyduel.top_drivers(None) == []
    ro = flyduel.named_drivers(dict(sums, types=np.array(["DNp35", "x", "y", "z"])))
    assert ro and ro[0]["type"] == "DNp35"


# --- the bracket --------------------------------------------------------------------------------------------------------------------
def make_play(winner_of=lambda a, b, attempt: a, calls=None):
    def play(task):
        if task[0] == "card":
            s = task[1]
            return dict(seed=s, mode="subtle", loom_latency_s=0.1 + 0.001 * s, loom_crossed=True, sugar_ratio=2.0 + 0.01 * s, turning_ratio=1.0,
                        turning_log_ratio=0.001 * s, walk_level_calm=1.0, title=f"Fly {s}", summary="x", traits=[], measured=True)
        a, b, bseed, attempt = task[0], task[1], task[2], task[3]
        if calls is not None:
            calls.append((a, b, attempt))
        w = winner_of(a, b, attempt)
        return dict(winner=None if w is None else str(w), hp={str(a): 50.0, str(b): 50.0 if w is None else 10.0}, shots={str(a): 1, str(b): 1},
                    hits={str(a): 1, str(b): 1}, seconds=5.0, knockout=False, max_levels={str(a): {}, str(b): {}}, events=[], frames=[], steps=1,
                    attempt=attempt, match_seed=attempt)
    return play


@pytest.mark.parametrize("size", [4, 8, 16])
def test_a_bracket_has_size_minus_one_matches_and_one_champion(size):
    seeds = list(range(100, 100 + size))
    b = tn.run_bracket(seeds, play=make_play(lambda a, b, t: max(a, b)), drivers=False)
    assert [len(r) for r in b["rounds"]] == [size // 2 // 2 ** i for i in range(len(b["rounds"]))] and sum(len(r) for r in b["rounds"]) == size - 1
    assert b["champion"] == max(seeds) and b["rounds_won"][b["champion"]] == len(b["rounds"])
    assert sum(b["rounds_won"].values()) == size - 1
    first = b["rounds"][0]
    assert [(m["a"], m["b"]) for m in first] == list(zip(seeds[::2], seeds[1::2])), "seeds are paired in the order given"


def test_a_bracket_refuses_other_sizes_repeated_seeds_and_a_favorite_who_is_not_in_it():
    with pytest.raises(tn.TournamentError):
        tn.run_bracket([1, 2, 3], play=make_play())
    with pytest.raises(tn.TournamentError):
        tn.run_bracket([1, 1, 2, 3], play=make_play())
    with pytest.raises(tn.TournamentError):
        tn.run_bracket([1, 2, 3, 4], favorite=9, play=make_play())


def test_the_favorite_is_tracked():
    b = tn.run_bracket([1, 2, 3, 4], favorite=3, play=make_play(lambda a, b, t: max(a, b)), drivers=False)
    assert b["favorite_won"] is False and b["favorite_rounds_won"] == 0 and b["champion"] == 4          # 3 lost to 4 in round 1
    b = tn.run_bracket([1, 2, 3, 4], favorite=2, play=make_play(lambda a, b, t: max(a, b)), drivers=False)
    assert b["favorite_won"] is False and b["favorite_rounds_won"] == 1
    b = tn.run_bracket([1, 2, 3, 4], favorite=4, play=make_play(lambda a, b, t: max(a, b)), drivers=False)
    assert b["favorite_won"] is True and b["favorite_rounds_won"] == 2


def test_a_draw_is_rematched_twice_then_a_seeded_coin_toss_that_says_so():
    calls = []
    b = tn.run_bracket([1, 2, 3, 4], play=make_play(lambda a, b, t: None, calls), drivers=False, bracket_seed=7)
    m = b["rounds"][0][0]
    assert m["coin_toss"] and m["attempts"] == 1 + flyduel.TIEBREAK_REMATCHES and m["decided_by"] == "draw"
    assert [c for c in calls if c[:2] == (1, 2)] == [(1, 2, 0), (1, 2, 1), (1, 2, 2)]
    b2 = tn.run_bracket([1, 2, 3, 4], play=make_play(lambda a, b, t: None), drivers=False, bracket_seed=7)
    assert [m["winner"] for r in b["rounds"] for m in r] == [m["winner"] for r in b2["rounds"] for m in r], "the coin is seeded"
    b3 = tn.run_bracket([1, 2, 3, 4], play=make_play(lambda a, b, t: a if t == 1 else None), drivers=False)
    m = b3["rounds"][0][0]
    assert not m["coin_toss"] and m["attempts"] == 2 and m["winner"] == 1


def test_a_rematch_has_different_noise_but_the_same_two_flies():
    a, b = tn.match_noise(0, 1, 2, 0), tn.match_noise(0, 1, 2, 1)
    assert a != b and tn.match_noise(0, 1, 2, 0) == a and tn.match_noise(1, 1, 2, 0) != a


def test_cancel_stops_the_bracket_between_matches():
    import threading

    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        tn.run_bracket([1, 2, 3, 4], play=make_play(), cancel=ev, drivers=False)


# --- the analysis -------------------------------------------------------------------------------------------------------------------
def test_holm_correction():
    assert tn.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert tn.holm([0.5]) == [0.5] and all(p <= 1 for p in tn.holm([0.9, 0.95, 0.99]))


def test_a_trait_that_always_wins_is_found_and_one_that_does_not_matter_is_not():
    brackets = []
    for k in range(4):                                    # four brackets of 16: 60 decided matches
        seeds = list(range(1000 + 16 * k, 1016 + 16 * k))
        # the fly with the higher sugar ratio (here: the higher seed) always wins
        brackets.append(tn.run_bracket(seeds, play=make_play(lambda a, b, t: max(a, b)), drivers=False))
    an = tn.analyze(brackets)
    sugar = next(t for t in an["tests"] if t["trait"] == "sugar_ratio")
    assert sugar["higher_trait_wins"] == sugar["matches"] == 60 and sugar["significant"] and sugar["p_holm"] < 1e-6 and an["predicts_winning"]
    assert an["decided_matches"] == 60 and an["coin_tosses"] == 0
    rng = np.random.default_rng(3)
    random_brackets = [tn.run_bracket(list(range(2000 + 16 * k, 2016 + 16 * k)), play=make_play(lambda a, b, t, r=rng: a if r.random() < 0.5 else b), drivers=False)
                       for k in range(4)]
    an2 = tn.analyze(random_brackets)
    assert not an2["predicts_winning"] or min(t["p_holm"] for t in an2["tests"]) > 0.001


def test_coin_tosses_are_left_out_of_the_prediction_test():
    b = tn.run_bracket([1, 2, 3, 4], play=make_play(lambda a, b, t: None), drivers=False)
    an = tn.analyze([b])
    assert an["decided_matches"] == 0 and an["coin_tosses"] == 3 and all(t["matches"] == 0 for t in an["tests"])


def test_the_summary_and_files_say_what_was_measured(tmp_path):
    b = tn.run_bracket([1, 2, 3, 4], favorite=2, play=make_play(lambda a, b, t: max(a, b)), drivers=False)
    an = tn.analyze([b])
    text = tn.summary([b], an)
    assert "DOES PERSONALITY PREDICT WINNING" in text and "MODEL PREDICTION" in text and "your favorite 2" in text
    files = tn.save([b], an, tmp_path)
    assert {f.name for f in files} == {"tournament.json", "tournament_flies.csv", "tournament_matches.csv", "tournament_analysis.csv"}
    import json

    data = json.loads((tmp_path / "tournament.json").read_text())
    assert data["brackets"][0]["tags"]["winner_and_drivers"] == "MODEL PREDICTION" and data["analysis"]["criteria"]


def test_p_values_at_the_floor_print_as_less_than_0_001():
    from kickthefly.lab import labstats

    assert labstats.fmt_p(1 / 1024) == "p < 0.001" and labstats.fmt_p(0.0195) == "p = 0.019"


# --- the real code on the synthetic pack (plumbing) ------------------------------------------------------------------------------------
def test_a_real_card_is_measured_from_the_brain_and_is_repeatable(synthetic_pack):
    c1, c2 = tn.measure_card(3, "subtle", warmup=100), tn.measure_card(3, "subtle", warmup=100)
    assert c1["measured"] and c1 == c2 and c1["title"] and c1["loom_latency_s"] <= tn.LATENCY_CAP_S and c1["sugar_ratio"] > 0 and c1["turning_ratio"] > 0
    assert set(tn.TRAITS) <= set(c1) and "walk_level_calm" in c1 and c1["tmaze_pi"].startswith("not measured")


def test_two_real_brains_can_duel_and_replay_frames_are_recorded(synthetic_pack):
    a = tn.build_fighter(5, "subtle", 11)
    b = tn.build_fighter(6, "subtle", 12)
    r = flyduel.run_duel(a, b, seed=1, seconds=1.0, names=("5", "6"), drivers=True)
    assert r["steps"] == 200 and r["winner"] in ("5", "6", None) and len(r["frames"]) == 10 and set(r["_drivers"]) == {"5", "6"}
