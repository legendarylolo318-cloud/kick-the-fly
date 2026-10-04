"""3.0 day 2: thermogenetics (TrpA1, shibire-ts) and the DNp01 escape-vs-temperature assay. Plumbing and arithmetic on the synthetic
pack (no biology is read off it); the real-pack assay run is reported in docs/thermogenetics.md."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.lab import thermogenetics as tg


# --- the effector curves: what the papers give, and what is a game rule --------------------------------------------------------------
def test_trpa1_turns_on_at_the_cited_temperature_and_is_off_below():
    ef = tg.effector("TrpA1")
    assert ef.onset_c == 25.0 and ef.kind == "activate"            # Pulver et al. 2009: tonic spiking at about 25 C
    assert ef.target_fraction(21.0) == 0.0 and ef.target_fraction(25.0) == 0.0
    assert ef.target_fraction(27.0) == pytest.approx(0.5) and ef.target_fraction(29.0) == 1.0 and ef.target_fraction(40.0) == 1.0
    assert "Pulver" in tg.CITATIONS["trpa1"]


def test_shibire_ts_is_restrictive_at_30_c_and_slow_like_the_paper():
    ef = tg.effector("shibire-ts")
    assert ef.kind == "silence" and ef.full_c == 30.0               # Kitamoto 2001: motionless within 2 min at 30 C
    assert ef.target_fraction(22.0) == 0.0 and ef.target_fraction(30.0) == 1.0
    assert ef.tau_on_s * 3 <= 120.0 + 1e-9                          # ~3 time constants fit in "within 2 min"
    assert ef.tau_off_s * 3 <= 60.0 + 1e-9                          # and in "about 1 min" to recover
    assert ef.tau_on_s > tg.effector("trpa1").tau_on_s
    assert "Kitamoto" in tg.CITATIONS["shibire"]


def test_every_number_is_tagged_either_cited_or_a_game_rule():
    for ef in tg.EFFECTORS.values():
        assert ef.cited and "game rule" in ef.rule
    assert tg.ACTIVATE_CURRENT == 0.5 and tg.SILENCE_CURRENT == -0.6   # validation's drive current and surgery's silencing current


def test_aliases_and_errors():
    for name in ("trpa1", "dTrpA1", "shibire", "shi-ts", "SHIBIRE_TS"):
        assert tg.effector(name)
    with pytest.raises(tg.ThermoError, match="unknown effector"):
        tg.effector("chrimson")


def test_arena_temperature_is_15_to_35_c_linear():
    assert tg.arena_temperature(-1) == 15.0 and tg.arena_temperature(1) == 35.0 and tg.arena_temperature(0) == 25.0
    assert tg.arena_temperature(-5) == 15.0 and tg.arena_temperature(3) == 35.0     # clamped at the walls


# --- on a brain ------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def br(synthetic_pack):
    from kickthefly.core import simcore

    return simcore.new_brain(seed=3, warmup=100)


def test_expression_reaches_only_the_chosen_neurons_and_clears(br):
    dn = np.flatnonzero(br.types == "DNp01")
    th = tg.Thermogenetics([tg.Expression("trpa1", "type:DNp01")], "steady")
    th.update(br, 32.0, 0.05)
    assert br.injecting and set(np.flatnonzero(br.inject)) == set(dn)
    assert np.allclose(br.inject[dn], tg.ACTIVATE_CURRENT)
    th.update(br, 20.0, 0.05)
    assert not br.injecting and not np.any(br.inject)               # below the onset the current is gone
    th.update(br, 32.0, 0.05)
    th.clear(br)
    assert not br.injecting and "thermo" not in br.currents


def test_kinetics_are_first_order_and_shibire_is_slower(br):
    a = tg.Thermogenetics([tg.Expression("trpa1", "type:DNp01")], "real")
    b = tg.Thermogenetics([tg.Expression("shibire", "type:MDN")], "real")
    for _ in range(20):                                              # 1 s at 35 C in 50 ms updates
        a.update(br, 35.0, 0.05)
        b.update(br, 35.0, 0.05)
    assert a.activation[0] == pytest.approx(1 - np.exp(-1.0 / 1.0), abs=1e-6)
    assert b.activation[0] == pytest.approx(1 - np.exp(-1.0 / 40.0), abs=1e-6)
    for _ in range(200):                                             # 10 s more: TrpA1 is on, shibire-ts is not yet
        a.update(br, 35.0, 0.05)
        b.update(br, 35.0, 0.05)
    assert a.activation[0] > 0.9999 and b.activation[0] < 0.25
    for _ in range(200):                                             # cooled: TrpA1 falls with its 1 s constant
        a.update(br, 20.0, 0.05)
    assert a.activation[0] == 0.0


def test_time_scale_compresses_the_kinetics(br):
    slow = tg.Thermogenetics([tg.Expression("shibire", "type:MDN")], "real", 1.0)
    fast = tg.Thermogenetics([tg.Expression("shibire", "type:MDN")], "real", 40.0)
    slow.update(br, 35.0, 0.05)
    fast.update(br, 35.0, 0.05)
    assert fast.activation[0] > 10 * slow.activation[0]


def test_two_expressions_in_the_same_cells_add(br):
    th = tg.Thermogenetics([tg.Expression("trpa1", "type:DNp01"), tg.Expression("shibire", "type:DNp01")], "steady")
    th.update(br, 36.0, 0.05)
    row = int(np.flatnonzero(br.types == "DNp01")[0])
    assert br.inject[row] == pytest.approx(tg.ACTIVATE_CURRENT + tg.SILENCE_CURRENT, abs=1e-6)


def test_a_driver_line_can_carry_the_effector(br):
    th = tg.Thermogenetics([tg.Expression("trpa1", "line:SS00727")], "steady")
    th.update(br, 33.0, 0.05)
    assert set(np.flatnonzero(br.inject)) == set(np.flatnonzero(br.types == "DNp01"))


def test_bad_targets_are_reported_not_ignored(br):
    with pytest.raises(tg.ThermoError, match="unknown neuron spec|selects no neurons"):
        tg.Thermogenetics([tg.Expression("trpa1", "nonsense")], "steady").update(br, 30.0, 0.05)
    with pytest.raises(tg.ThermoError, match="selects no neurons"):
        tg.Thermogenetics([tg.Expression("trpa1", "type:NoSuchType")], "steady").update(br, 30.0, 0.05)


def test_nothing_changes_while_no_expression_is_set(synthetic_pack):
    """The Brain.currents plumbing is inert: identical spikes with and without an (empty) thermogenetics object and a probe."""
    from kickthefly.core import simcore

    a, b = simcore.new_brain(seed=9, warmup=50), simcore.new_brain(seed=9, warmup=50)
    th = tg.Thermogenetics([], "real")
    b.probe = lambda br: None
    for i in range(300):
        if i % 10 == 0:
            th.update(b, 35.0, 0.05)
        a._step()
        b._step()
    assert np.array_equal(a.sim.v, b.sim.v) and not b.injecting


def test_the_thermo_current_does_not_fake_a_calm_baseline(br):
    """While a current is injected the brain is not calm, so the baseline that 'level' is measured against doesn't learn it."""
    base = br.base.copy()
    th = tg.Thermogenetics([tg.Expression("trpa1", "type:DNp01")], "steady")
    th.update(br, 35.0, 0.05)
    for _ in range(600):
        br._step()
    assert np.array_equal(br.base, base)


# --- protocols ---------------------------------------------------------------------------------------------------------------------------
def test_spec_validation():
    ok = tg.check_spec({"expression": [{"effector": "trpa1", "target": "type:DNp01"}], "temperature_c": 30})
    assert ok["temperature_c"] == 30
    for bad, msg in (({}, "expression must be a list"),
                     ({"expression": [{"effector": "x", "target": "a"}]}, "unknown effector"),
                     ({"expression": [{"effector": "trpa1"}]}, "needs effector and target"),
                     ({"expression": [{"effector": "trpa1", "target": "t"}], "temperature_c": 99}, "within"),
                     ({"expression": [{"effector": "trpa1", "target": "t"}], "kinetics": "fast"}, "kinetics"),
                     ({"expression": [{"effector": "trpa1", "target": "t"}], "time_scale": 0}, "time_scale"),
                     ({"expression": [{"effector": "trpa1", "target": "t"}], "what": 1}, "unknown keys")):
        with pytest.raises(tg.ThermoError, match=msg):
            tg.check_spec(bad)


def test_temperature_schedule_holds_until_the_next_point():
    spec = {"temperature_c": [{"at_s": 0, "c": 22}, {"at_s": 2, "c": 32}]}
    assert tg.temperature_at(spec, 0.0) == 22 and tg.temperature_at(spec, 1.99) == 22 and tg.temperature_at(spec, 2.0) == 32
    assert tg.temperature_at({"temperature_c": 28}, 10.0) == 28


def _protocol(tmp_path, temp, expression=True, seeds=(0,)):
    from kickthefly.lab import protocol

    p = dict(name="th", seed=seeds[0], flies=len(seeds), warmup_s=0.5, duration_s=1.5,
             stimuli=[], recordings=[dict(name="gf", neurons="type:DNp01")])
    if expression:
        p["thermogenetics"] = dict(expression=[dict(effector="trpa1", target="type:DNp01")], temperature_c=temp, kinetics="steady")
    p = protocol.check(p)
    out = tmp_path / f"t{temp}{expression}"
    out.mkdir()
    return protocol.run_seed(p, seeds[0], None, out, "run")


def test_a_protocol_drives_the_expressing_cells_only_above_the_threshold(synthetic_pack, tmp_path):
    cold = _protocol(tmp_path, 20.0)["gf"]
    hot = _protocol(tmp_path, 34.0)["gf"]
    control = _protocol(tmp_path, 34.0, expression=False)["gf"]
    assert hot > 4 * cold                                           # expressing cells fire much harder when it is warm (plumbing check)
    assert control < hot / 4                                        # the same temperature does nothing without the channel


def test_protocol_thermogenetics_is_refused_in_assays_and_replays(tmp_path):
    from kickthefly.lab import protocol

    with pytest.raises(protocol.ProtocolError, match="stimulus protocols only"):
        protocol.check(dict(name="x", assay="sugar", thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")])))
    with pytest.raises(protocol.ProtocolError, match="effector"):
        protocol.check(dict(name="x", thermogenetics=dict(expression=[dict(effector="nope", target="a")])))


# --- the assay ---------------------------------------------------------------------------------------------------------------------------
def test_assay_criteria_are_written_down_before_any_result():
    assert len(tg.ASSAY_CRITERIA) == 3 and tg.ASSAY_TEMPS == (18.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0)
    assert all(c.startswith("C") for c in tg.ASSAY_CRITERIA)


def _summary(rates, control=None, n=10):
    rows = []
    for t, r in zip(tg.ASSAY_TEMPS, rates):
        c = 0 if control is None else control[list(tg.ASSAY_TEMPS).index(t)]
        from kickthefly.lab import labstats

        rows.append(dict(temperature_c=t, escapes=r, control_escapes=c, flies=n, escape_rate=r / n, control_rate=c / n,
                         fisher_p=labstats.fisher(r, n, c, n)))
    return dict(rows=rows)


def test_the_verdict_scores_the_three_criteria_as_written():
    good = tg.verdict(_summary([0, 0, 0, 0, 3, 10, 10, 10]))
    assert good["C1"] and good["C2"] and good["C3"] and good["passed"]
    early = tg.verdict(_summary([5, 6, 7, 8, 9, 10, 10, 10]))          # escapes below the onset: C1 fails
    assert not early["C1"] and not early["passed"]
    weak = tg.verdict(_summary([0, 0, 0, 0, 0, 6, 7, 8]))              # not reliable when fully on: C2 fails
    assert not weak["C2"] and not weak["passed"]
    leaky_control = tg.verdict(_summary([0, 0, 0, 0, 3, 10, 10, 10], control=[0, 0, 0, 0, 0, 3, 3, 3]))
    assert not leaky_control["C2"]
    noisy = tg.verdict(_summary([0, 10, 0, 10, 0, 10, 0, 10]))
    assert not noisy["C3"]


def test_escape_fly_runs_and_summarizes(synthetic_pack):
    fly = tg.escape_fly(0, temps=(22.0, 34.0))
    assert set(fly["trials"]) == {22.0, 34.0}
    t = fly["trials"][34.0]
    assert set(t) == {"expressing", "control"} and t["expressing"]["dnp01_hz"] > t["control"]["dnp01_hz"]
    summ = tg.summarize([fly, tg.escape_fly(1, temps=(22.0, 34.0))])
    assert [r["temperature_c"] for r in summ["rows"]] == [22.0, 34.0] and len(summ["per_fly"]) == 2


def test_the_assay_is_a_lab_assay_and_survives_json(synthetic_pack):
    from kickthefly.lab import labjobs

    assert "thermo_escape" in labjobs.ASSAYS and labjobs.ASSAY_LABEL["thermo_escape"]
    r = labjobs.run_sync("thermo_escape", [0], {"temps": (22.0, 34.0)}, None, None, workers=1)
    import json

    back = json.loads(json.dumps(r, default=float))
    assert tg.summarize(back["flies"])["rows"][0]["temperature_c"] == 22.0      # float keys become strings in JSON and still work
    assert labjobs.headline("thermo_escape", r["flies"][0]) >= 0.0


def test_live_state_follows_the_arena_and_the_slider(synthetic_pack):
    from types import SimpleNamespace

    from kickthefly.core import simcore
    from kickthefly.lab import livelab

    br = simcore.new_brain(seed=2, warmup=50)
    slot = SimpleNamespace(brain=br, arena_temp_c=34.0)
    tl = livelab.ThermoLive()
    tl.add("trpa1", "type:DNp01")
    tl.kinetics = "steady"
    tl.temperature_c = 20.0
    tl.tick([slot], arena_is_thermo=True)                            # the slider is the source: cold
    assert not br.injecting
    tl.source = "arena"
    tl.tick([slot], arena_is_thermo=True)                            # the arena says 34 C under this fly
    assert br.injecting
    tl.tick([slot], arena_is_thermo=False)                           # outside the thermo arena the slider is used again
    assert not br.injecting
    tl.clear([br])
    assert not br.injecting and not tl.per_brain
