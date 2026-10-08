"""3.0 day 4, Sensitivity analysis (lab/sensitivity.py): the grid, how a parameter reaches the brain through validation's own code
(and that doing so changes nothing for validation itself), resume, exports and the heatmap, on the SYNTHETIC pack (plumbing, not
biology) and with a fake cell runner for the bookkeeping."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kickthefly.lab import lab, sensitivity as sv, validation


def fake_cell(param, value, seeds, tests, workers=None, progress=None):
    """A cell whose looming behavior passes unless the noise is above 0.07 and whose T-maze fails whenever anything is changed."""
    behaviors = {}
    for t in tests:
        passed = not (t == "looming_escape" and param == "noise_std" and value is not None and value > 0.07)
        passed = passed and not (t == "mb_conditioning" and param != sv.BASELINE)
        behaviors[t] = dict(passed=passed, effect=3.0 if passed else 1.1, control=0.9, metric="drive/baseline ratio", effect_size_dz=2.0 if passed else 0.2,
                            p_value=0.001 if passed else 0.4, n=len(seeds))
    if progress:
        progress(len(seeds), len(seeds), "fake")
    return dict(key=sv.cell_key(param, value), parameter=param, value=value, behaviors=behaviors, backend="cpu", seconds=0.1)


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def run_cell(param, value, seeds, tests, workers=None, progress=None):
        calls.append(sv.cell_key(param, value))
        return fake_cell(param, value, seeds, tests, workers, progress)

    monkeypatch.setattr(sv, "run_cell", run_cell)
    return calls


# --- the grid -----------------------------------------------------------------------------------------------------------------------
def test_every_lif_value_lies_inside_the_lab_parameter_range_and_the_default_is_the_labs():
    for p in sv.PARAMETERS:
        if p["kind"] == "lif":
            spec = lab.BY_NAME[p["id"]]
            assert p["default"] == pytest.approx(spec[3]), p["id"]
            assert all(spec[4] <= v <= spec[5] for v in p["values"]), (p["id"], p["values"], spec[4:6])
            assert p["default"] not in p["values"], "the default is the baseline column, not a tested value"
    assert {p["id"] for p in sv.PARAMETERS} == {"noise_std", "bias", "target_rate_hz", "ext_gain", "gain_adapt", "min_synapses"}, "the six parameters the brief names"


def test_the_synapse_threshold_range_is_the_robustness_sweeps_and_above_the_packs_own_floor():
    from kickthefly.lab import robustness

    p = sv.BY_ID["min_synapses"]
    assert p["kind"] == "wiring" and p["default"] == robustness.MIN_PACK_SYNAPSES and min(p["values"]) > p["default"]
    assert set(p["values"]) <= set(robustness.DEFAULT_THRESHOLDS)


def test_the_validated_behaviors_are_exactly_the_twelve_validation_reproduces():
    ids = sv.validated_behaviors()
    assert len(ids) == 12 and all(validation.EXPECTED[i] for i in ids) and "looming_escape" in ids and "mdn_backward" not in ids
    assert all(i in {t["id"] for t in validation.TESTS} for i in ids), "the larva tests are not part of this analysis"


def test_cells_start_with_the_baseline_and_one_per_value_and_unknown_parameters_are_refused():
    cs = sv.cells(["noise_std", "ext_gain"], {"noise_std": [0.03]})
    assert cs[0] == (sv.BASELINE, None) and cs[1:] == [("noise_std", 0.03), *[("ext_gain", float(v)) for v in sv.BY_ID["ext_gain"]["values"]]]
    with pytest.raises(sv.SensitivityError):
        sv.cells(["nope"])
    assert len(sv.cells()) == 1 + sum(len(p["values"]) for p in sv.PARAMETERS)


def test_a_cell_runs_one_change_only():
    assert sv._run_args(sv.BASELINE, None) == (None, None)
    params, wiring = sv._run_args("noise_std", 0.075)
    assert params == {"noise_std": 0.075} and wiring is None
    params, wiring = sv._run_args("min_synapses", 6)
    assert params is None and wiring.min_synapses == 6 and wiring.flip_rows == () and wiring.inhibition_scale == 1.0


# --- validation's own code, with parameters ------------------------------------------------------------------------------------------
def test_validation_defaults_are_untouched_and_params_reach_the_brain(synthetic_pack, monkeypatch):
    from kickthefly.core import simcore

    seen = []
    real = simcore.new_brain

    def spy(*a, **k):
        seen.append(k.get("params"))
        return real(*a, **k)

    monkeypatch.setattr(simcore, "new_brain", spy)
    validation._pathway_seed(1, None, None, {"looming_escape"})
    validation._pathway_seed(1, None, {"noise_std": 0.075}, {"looming_escape"})
    assert seen == [None, {"noise_std": 0.075}]


def test_a_subset_run_gives_the_same_numbers_as_the_full_run_for_the_tests_it_kept(synthetic_pack):
    full = validation._pathway_seed(2, None, None, None)
    for keep in ({"looming_escape"}, {"optomotor_turning"}, {"sugar_feeding", "da1pn_to_lh_asp"}):
        sub = validation._pathway_seed(2, None, None, keep)
        assert set(sub) - {"_backend"} == keep
        for t in keep:
            assert sub[t] == full[t], f"{t}: skipping other tests changed its result (the control draw or the noise stream moved)"


def test_a_changed_parameter_reaches_the_sim_and_changes_the_result(synthetic_pack):
    base = validation._pathway_seed(2, None, None, {"looming_escape"})
    noisy = validation._pathway_seed(2, None, {"noise_std": 0.10}, {"looming_escape"})
    assert base["looming_escape"] != noisy["looming_escape"]
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=1, params={"ext_gain": 2.0, "gain_adapt": 0.0}, warmup=10)
    assert br.sim.p.ext_gain == 2.0 and br.sim.p.gain_adapt == 0.0


def test_the_baseline_cell_is_validations_own_result(synthetic_pack):
    seeds = (1000, 1001, 1002)
    cell = sv.run_cell(sv.BASELINE, None, seeds, ["looming_escape", "sugar_feeding"], workers=1)
    direct = validation.run(seeds=seeds, workers=1, include={"looming_escape", "sugar_feeding"})
    for t in direct["tests"]:
        m = t["measured"]
        got = cell["behaviors"][t["id"]]
        assert got["effect"] == m["drive_ratio_mean"] and got["control"] == m["control_ratio_mean"] and got["p_value"] == m["p_value"] and got["passed"] == t["passed"]


def test_effect_size_is_the_paired_standardised_difference():
    t = dict(passed=True, measured=dict(drive_ratio_mean=3.0, control_ratio_mean=1.0, p_value=0.001, n=4),
             per_seed=[dict(drive=dict(ratio=r), control=dict(ratio=c)) for r, c in ((3.0, 1.0), (3.5, 1.0), (2.5, 1.0), (3.0, 0.5))])
    e = sv._effect_size(t)
    d = np.array([2.0, 2.5, 1.5, 2.5])
    assert e["effect_size_dz"] == pytest.approx(d.mean() / d.std(ddof=1)) and e["effect"] == 3.0 and e["metric"] == "drive/baseline ratio"
    tm = dict(passed=False, measured=dict(pi_mean=0.1, control_pi_mean=0.0, p_value=0.4, n=2), per_seed=[dict(pi=0.2, control_pi=0.0), dict(pi=0.0, control_pi=0.0)])
    assert sv._effect_size(tm)["metric"] == "performance index"


# --- the run, resume, exports ---------------------------------------------------------------------------------------------------------
def test_a_run_covers_every_cell_and_every_behavior_and_marks_few_seeds_underpowered(fake, tmp_path):
    res = sv.run(params=["noise_std"], tests=["looming_escape", "sugar_feeding"], seeds=[1000, 1001, 1002], folder=tmp_path)
    assert res["underpowered"] and len(res["cells"]) == 1 + 4 and fake[0] == sv.BASELINE
    assert all(set(c["behaviors"]) == {"looming_escape", "sugar_feeding"} for c in res["cells"])
    full = sv.run(params=["noise_std"], tests=["looming_escape"], folder=tmp_path / "b")
    assert not full["underpowered"] and full["seeds"] == list(validation.SEEDS)
    assert full["criteria"]["p_max"] == validation.P_MAX and full["tags"]["every_cell"] == "MODEL PREDICTION"


def test_resume_skips_finished_cells_and_continues_after_a_crash(monkeypatch, tmp_path):
    done, state = [], dict(crashed=False)

    def flaky(param, value, seeds, tests, workers=None, progress=None):
        key = sv.cell_key(param, value)
        if key == "noise_std=0.075" and not state["crashed"]:
            state["crashed"] = True
            raise RuntimeError("worker died")
        done.append(key)
        return fake_cell(param, value, seeds, tests, workers, progress)

    monkeypatch.setattr(sv, "run_cell", flaky)
    kw = dict(params=["noise_std"], tests=["looming_escape"], seeds=[1000, 1001, 1002], folder=tmp_path)
    with pytest.raises(RuntimeError):
        sv.run(**kw)
    saved = json.loads((tmp_path / sv.PROGRESS_NAME).read_text())
    assert set(saved["cells"]) == {"baseline", "noise_std=0.025", "noise_std=0.0375"}, "every finished cell was written as it finished"
    done.clear()
    res = sv.run(**kw, resume=True)
    assert done == ["noise_std=0.075", "noise_std=0.1"], f"resume reran finished cells: {done}"
    assert [c["key"] for c in res["cells"]] == ["baseline", "noise_std=0.025", "noise_std=0.0375", "noise_std=0.075", "noise_std=0.1"]


def test_resume_refuses_a_progress_file_from_a_different_run_and_a_fresh_run_ignores_it(fake, tmp_path):
    sv.run(params=["noise_std"], tests=["looming_escape"], seeds=[1000, 1001, 1002], folder=tmp_path)
    with pytest.raises(sv.SensitivityError, match="different run"):
        sv.run(params=["noise_std"], tests=["sugar_feeding"], seeds=[1000, 1001, 1002], folder=tmp_path, resume=True)
    fake.clear()
    sv.run(params=["noise_std"], tests=["sugar_feeding"], seeds=[1000, 1001, 1002], folder=tmp_path)
    assert len(fake) == 5, "without --resume every cell runs again"


def test_progress_is_reported_per_cell(fake, tmp_path):
    seen = []
    sv.run(params=["bias"], tests=["looming_escape"], seeds=[1, 2, 3], folder=tmp_path, progress=lambda d, n, label: seen.append((d, n, label)))
    assert seen[-1][0] == seen[-1][1] == 5 and all(b[0] >= a[0] for a, b in zip(seen, seen[1:]))


def test_nothing_the_analysis_does_changes_a_default(fake, tmp_path):
    from kickthefly.sim.connectome.sim import LIFParams

    before = (dict(lab.DEFAULTS), LIFParams().noise_std, LIFParams().bias, LIFParams().ext_gain, dict(validation.EXPECTED))
    sv.run(params=["noise_std", "bias"], tests=["looming_escape"], seeds=[1, 2, 3], folder=tmp_path)
    after = (dict(lab.DEFAULTS), LIFParams().noise_std, LIFParams().bias, LIFParams().ext_gain, dict(validation.EXPECTED))
    assert before == after


def test_exports_are_csv_json_and_a_script_free_svg_heatmap(fake, tmp_path):
    res = sv.run(params=["noise_std", "min_synapses"], tests=["looming_escape", "mb_conditioning"], seeds=list(validation.SEEDS), folder=tmp_path / "w")
    files = sv.save(res, tmp_path / "out")
    assert {f.name for f in files} == {"sensitivity.json", "sensitivity.csv", "sensitivity_matrix.csv", "sensitivity_robustness.csv", "sensitivity_heatmap.svg"}
    long = (tmp_path / "out" / "sensitivity.csv").read_text().splitlines()
    assert long[0].startswith("parameter,value,is_default,behavior,passed") and len(long) == 1 + len(res["cells"]) * 2
    assert "baseline" in (tmp_path / "out" / "sensitivity_matrix.csv").read_text()
    root = ET.fromstring((tmp_path / "out" / "sensitivity_heatmap.svg").read_text())
    text = ET.tostring(root, encoding="unicode")
    assert "<script" not in text and "http://" not in text.replace("http://www.w3.org/2000/svg", "") and "MODEL PREDICTION" in text and "PASS" in text and "FAIL" in text
    for pal in sv.PALETTES:
        ET.fromstring(sv.heatmap_svg(res, pal))
    assert json.loads((tmp_path / "out" / "sensitivity.json").read_text())["tests"] == ["looming_escape", "mb_conditioning"]


def test_the_robustness_table_names_where_each_behavior_stops(fake, tmp_path):
    res = sv.run(params=["noise_std"], tests=["looming_escape", "mb_conditioning"], seeds=list(validation.SEEDS), folder=tmp_path)
    rt = {(r["parameter"], r["behavior"]): r for r in sv.robustness_table(res)}
    loom = rt[("noise_std", "looming_escape")]
    assert loom["values_tested"] == 4 and loom["values_passing"] == 2 and loom["first_fail_above"] == 0.075 and loom["first_fail_below"] is None
    assert rt[("noise_std", "mb_conditioning")]["values_passing"] == 0 and rt[("noise_std", "mb_conditioning")]["first_fail_below"] == 0.0375
    text = sv.summary(res)
    assert "analysis only" in text and "MODEL PREDICTION" in text and "defaults are not changed" in text


def test_the_headless_flag_runs_the_analysis_and_writes_its_files(fake, tmp_path, capsys):
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import headless

    args = k.parse_args(["--headless", "--sensitivity", "--sens-params", "noise_std", "--sens-tests", "looming_escape", "--sens-values", "noise_std=0.03,0.07",
                         "--seeds", "1000-1009", "--out", str(tmp_path / "o")])
    assert args.sensitivity and args.sens_values == ["noise_std=0.03,0.07"]
    assert headless.run_sensitivity(args) == 0
    assert (tmp_path / "o" / "sensitivity_heatmap.svg").exists() and "analysis only" in capsys.readouterr().out
    args.sens_values = ["noise_std=oops"]
    assert headless.run_sensitivity(args) == 2
