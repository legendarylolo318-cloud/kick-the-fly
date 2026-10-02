"""3.0 day 4, Sleep deprivation assay (lab/sleepdep.py): the game-rule dynamics on a scripted brain whose dFB level follows the
current it is given (so what is the rule and what is the readout can be told apart), the paired statistics, the Lab-job and protocol
paths, and a real-code smoke test on the SYNTHETIC pack (plumbing, not biology)."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.lab import sleepdep as sd


class FakeBrain:
    """dFB level = 1 + GAIN * (the mean current on the sleep neurons): the CONNECTOME readout, scripted. Everything else is the rule."""

    GAIN = 40.0

    def __init__(self):
        self.dt = 0.005
        self.sense = {("sleep", None): np.arange(10)}
        self.pokes, self.currents, self.steps = [], {}, 0
        self.times = []

    def poke(self, region, side, strength, recruit=None):
        self.pokes.append((self.steps * self.dt, region, strength))

    def set_current(self, name, rows, values):
        self.currents[name] = float(values)

    def clear_current(self, name):
        self.currents.pop(name, None)

    def _step(self):
        self.steps += 1

    def level(self, name):
        return 1.0 + self.GAIN * self.currents.get("sleep_pressure", 0.0) if name == "sleep" else 1.0


def run(disturb, **kw):
    b = FakeBrain()
    r = sd.run_fly(1, disturb, brain=b, **kw)
    return r, b


def test_a_control_fly_sleeps_once_its_pressure_has_driven_the_dfb_over_the_threshold():
    r, b = run(False)
    assert r["sleep_s"]["deprivation"] > 0 and r["sleep_s"]["recovery"] > 0 and 0 < r["peak_pressure"] <= 1.0


def test_disturbances_come_every_few_seconds_only_in_the_deprivation_window_and_never_in_a_control():
    r, b = run(True)
    t0, t1 = sd.SETTLE_S, sd.SETTLE_S + sd.DEPRIVE_S
    touches = sorted({round(t, 2) for t, region, _ in b.pokes if region == "legs"})
    assert touches and all(t0 - 0.05 <= t < t1 for t in touches)
    assert np.allclose(np.diff(touches), sd.DISTURB_EVERY_S, atol=0.05)
    assert {region for _, region, _ in b.pokes} >= {"legs", "body", "light", "clock"} or {"legs", "body"} <= {region for _, region, _ in b.pokes}
    _, c = run(False)
    assert not any(region in ("legs", "body") for _, region, _ in c.pokes)


def test_deprivation_keeps_the_fly_awake_and_builds_pressure_the_control_discharges():
    ctrl, _ = run(False)
    dep, _ = run(True)
    assert dep["sleep_s"]["deprivation"] <= 0.25 * max(ctrl["sleep_s"]["deprivation"], 1e-9) + 1e-9
    assert dep["pressure_end_of_deprivation"] > ctrl["pressure_end_of_deprivation"]
    assert dep["dfb_level_end_of_deprivation"] > ctrl["dfb_level_end_of_deprivation"], "more pressure drives the dFB harder"


def test_the_deprived_fly_sleeps_more_in_the_recovery_window_than_its_control():
    ctrl, _ = run(False)
    dep, _ = run(True)
    assert dep["sleep_s"]["recovery"] > ctrl["sleep_s"]["recovery"]
    assert dep["first_sleep_after_deprivation_s"] is not None and dep["first_sleep_after_deprivation_s"] <= sd.SLEEP_AFTER_S + 0.1


def test_pressure_stays_between_zero_and_one_and_falls_only_asleep_and_rises_only_awake():
    r, _ = run(False)
    trace = r["trace"]
    assert all(0.0 <= p <= 1.0 for _, p, _, _ in trace)
    for (t0, p0, _, s0), (t1, p1, _, s1) in zip(trace, trace[1:]):
        if s0 == 1 and s1 == 1:
            assert p1 < p0 + 1e-9
        if s0 == 0 and s1 == 0:
            assert p1 > p0 - 1e-9


def test_without_a_dfb_response_there_is_no_sleep_at_all():
    class Deaf(FakeBrain):
        GAIN = 0.0

    b = Deaf()
    r = sd.run_fly(1, True, brain=b)
    assert sum(r["sleep_s"].values()) == 0 and r["peak_pressure"] > 0.5, "the rule alone cannot make the fly sleep: the dFB readout decides"


def test_the_constants_that_define_the_rule_are_the_documented_ones():
    assert sd.RISE_PER_S == 1 / 60 and sd.FALL_PER_S == 1 / 30 and sd.FALL_PER_S == 2 * sd.RISE_PER_S, "recovery twice as fast as build-up, as the pet's"
    assert sd.DFB_CURRENT_AT_FULL == 0.08 and sd.SLEEP_AFTER_S == 1.0 and sd.START_PRESSURE == 0.10
    doc = sd.__doc__
    for word in ("GAME RULE", "CONNECTOME", "MODEL PREDICTION", "SLEEP PRESSURE", "p < 0.001"):
        assert word in doc
    assert {t for _, t in sd.COMPONENTS} >= {"GAME RULE", "CONNECTOME readout", "MODEL PREDICTION"}


# --- the assay over seeds -----------------------------------------------------------------------------------------------------------
def fake_play(args):
    seed, disturbed = args[0], args[1]
    r, _ = run(disturbed)
    r["seed"] = seed
    return r


def test_the_paired_assay_scores_its_three_pre_registered_criteria():
    res = sd.run(list(range(1000, 1010)), play=fake_play)
    assert set(res["criteria"]) == {"S1", "S2", "S3"} and res["all_passed"]
    s1 = res["criteria"]["S1"]
    assert s1["p"] == pytest.approx(1 / 1024) and s1["mean_difference"] > 0 and res["p_text"] == "p < 0.001"
    assert res["criteria"]["S2"]["share"] <= 0.25 and res["criteria"]["S3"]["passed"]
    assert res["tags"] == dict(sleep_pressure_and_disturbance="GAME RULE", dfb_firing_and_daylight="CONNECTOME", rebound="MODEL PREDICTION")


def test_the_assay_reports_a_fail_honestly_when_there_is_no_rebound():
    def flat(args):
        r = fake_play(args)
        r["sleep_s"]["recovery"] = 7.0
        return r

    res = sd.run(list(range(1000, 1010)), play=flat)
    assert not res["criteria"]["S1"]["passed"] and not res["all_passed"] and res["criteria"]["S1"]["p"] == 1.0


def test_summary_says_which_parts_are_game_rule_and_which_are_the_connectome():
    res = sd.run(list(range(1000, 1010)), play=fake_play)
    text = sd.summary(res)
    assert "GAME RULE" in text and "CONNECTOME" in text and "p < 0.001" in text and "expected from the pressure rule" in text
    assert "S1 PASS" in text and "paired" in text


def test_files_and_the_labjobs_and_protocol_paths(tmp_path):
    from kickthefly.lab import labjobs, protocol

    res = sd.run(list(range(1000, 1007)), play=fake_play)
    files = sd.save(res, tmp_path)
    assert {f.name for f in files} == {"sleep_deprivation.json", "sleep_deprivation.csv"}
    header = (tmp_path / "sleep_deprivation.csv").read_text().splitlines()[0]
    assert "tag_pressure" in header and "tag_dfb" in header
    assert "sleep_deprivation" in labjobs.ASSAYS and labjobs.ASSAY_LABEL["sleep_deprivation"] and labjobs.HEADLINE_LABEL["sleep_deprivation"]
    flies = [dict(seed=s, deprive_s=45.0, recover_s=60.0, individuality="off", control=fake_play((s, False)), deprived=fake_play((s, True))) for s in range(1000, 1010)]
    summary = labjobs.summarize("sleep_deprivation", flies)
    assert summary["all_passed"] and len(summary["per_fly"]) == 10 and all(x > 0 for x in summary["per_fly"])
    assert labjobs.headline("sleep_deprivation", flies[0]) == flies[0]["deprived"]["sleep_s"]["recovery"] - flies[0]["control"]["sleep_s"]["recovery"]
    ok = dict(assay="sleep_deprivation", assay_options=dict(deprive_s=30, recover_s=40, mode="subtle"), seeds=[1, 2], name="x")
    protocol._check_sleep_deprivation(ok, "p")
    for bad in (dict(assay_options=dict(deprive_s=1)), dict(assay_options=dict(recover_s=999)), dict(assay_options=dict(oops=1)),
                dict(assay_options=dict(mode="loud")), dict(surgery={"loom": -1}), dict(assay_options=[1])):
        with pytest.raises(protocol.ProtocolError):
            protocol._check_sleep_deprivation(dict(assay="sleep_deprivation", **bad), "p")
    with pytest.raises(ValueError):
        labjobs.assay_task("sleep_deprivation", 1, {}, {"loom": -1}, None)


# --- the real code on the synthetic pack (plumbing) -----------------------------------------------------------------------------------------
def test_a_real_pair_runs_and_is_deterministic(synthetic_pack):
    a = sd.fly_pair(3, deprive_s=6.0, recover_s=6.0, settle_s=2.0)
    b = sd.fly_pair(3, deprive_s=6.0, recover_s=6.0, settle_s=2.0)
    assert a["control"]["sleep_s"] == b["control"]["sleep_s"] and a["deprived"]["trace"] == b["deprived"]["trace"]
    assert a["deprived"]["pressure_end_of_deprivation"] >= a["control"]["pressure_end_of_deprivation"] - 1e-9
    assert a["control"]["disturbed"] is False and a["deprived"]["disturbed"] is True
