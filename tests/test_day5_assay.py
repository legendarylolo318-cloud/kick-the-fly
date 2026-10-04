"""3.0 day 5, the rig assay: the pre-registered criteria on scripted per-fly measures, the headless entry points, the rig protocol kind, the Lab-format
export, and real-code smoke tests on the SYNTHETIC pack (plumbing, not biology)."""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from kickthefly.game import rigs
from kickthefly.lab import protocol, rigassay

ROOT = Path(__file__).resolve().parent.parent


def tether(seed, d=8.0, c=0.4):
    return dict(seed=seed, d_directional=d + 0.1 * seed, compensation_mean=c + 0.01 * seed, delta_steer_hz={}, compensation={}, closed_slip={})


def ball(seed, comp=0.4, vis=3.0, hid=90.0):
    return dict(seed=seed, compensation_mean=comp + 0.01 * seed, bar_error_deg_visible=vis, bar_error_deg_hidden=hid + seed, forward_speed_mean=0.14)


def buridan(seed, dev_s=3.0, dev_n=42.0, tr_s=18, tr_n=7):
    return dict(seed=seed, stripes=dict(deviation_deg=dev_s, transits=tr_s + seed % 3), none=dict(deviation_deg=dev_n + seed, transits=tr_n))


def four(seed, odor=0.1, sham=0.0):
    return dict(seed=seed, odor=dict(pi=odor + 0.01 * seed, speed_odor=0.17, speed_air=0.16), sham=dict(pi=sham - 0.01 * seed))


def test_every_criterion_passes_on_the_expected_effects_and_reports_its_numbers():
    seeds = range(10)
    t = rigassay.analyze("tethered", [tether(s) for s in seeds])
    assert t["all_passed"] and [c["id"] for c in t["criteria"]] == ["T1", "T2"] and t["criteria"][0]["p"] < 0.001
    b = rigassay.analyze("ball", [ball(s) for s in seeds])
    assert b["all_passed"] and [c["id"] for c in b["criteria"]] == ["BL1", "BL2"] and b["criteria"][1]["visible_below_30"]
    u = rigassay.analyze("buridan", [buridan(s) for s in seeds])
    assert u["all_passed"] and [c["id"] for c in u["criteria"]] == ["B1", "B2"]
    f = rigassay.analyze("fourfield", [four(s, odor=0.4) for s in seeds])
    assert f["all_passed"] and f["criteria"][0]["id"] == "F1" and "speed_odor_minus_air_m_s" in f["extra"]


def test_a_criterion_needs_both_the_minimum_effect_and_p_below_001():
    seeds = range(10)
    small = rigassay.analyze("tethered", [tether(s, d=1.0) for s in seeds])                  # significant but under 3 Hz
    assert not small["criteria"][0]["passed"] and small["criteria"][0]["p"] < 0.01
    noisy = [tether(s, d=8.0) for s in seeds]
    for i, x in enumerate(noisy):
        x["d_directional"] = 9.0 if i % 2 == 0 else -9.0                                     # a big mean is not possible: sign flips, p stays high
    r = rigassay.analyze("tethered", noisy)
    assert not r["criteria"][0]["passed"]
    few = rigassay.analyze("tethered", [tether(s) for s in range(3)])                         # n = 3 cannot reach p < 0.01
    assert not few["criteria"][0]["passed"] and few["criteria"][0]["p"] >= 0.125


def test_the_olfactory_arena_fails_without_a_preference_and_the_bar_needs_the_visible_error_under_30_degrees():
    seeds = range(10)
    assert not rigassay.analyze("fourfield", [four(s, odor=0.0) for s in seeds])["all_passed"]
    loose = rigassay.analyze("ball", [ball(s, vis=40.0, hid=120.0) for s in seeds])
    assert loose["criteria"][1]["mean_effect"] > 10 and not loose["criteria"][1]["passed"], "a hidden-minus-visible gap is not fixation if the bar is still 40 degrees off"


def test_unknown_rigs_are_refused():
    with pytest.raises(ValueError):
        rigassay.analyze("treadmill", [])
    with pytest.raises(ValueError):
        rigassay.run_flies("treadmill", [1])
    with pytest.raises(ValueError):
        rigassay.scene_run("treadmill")
    with pytest.raises(ValueError):
        rigassay.scene_run("buridan", mode="odor")
    for bad in (dict(seconds=1e9), dict(seconds=float("nan")), dict(seconds=True), dict(seconds="9"), dict(omega=99), dict(gain=-1), dict(scene="ceiling")):
        with pytest.raises(ValueError):
            rigassay.scene_run("tethered", **bad)


def test_run_assay_runs_both_modes_summarises_and_saves(tmp_path):
    seen = []

    def play(task):
        rig, seed, mode, backend = task
        seen.append((rig, seed, mode))
        return dict(buridan(seed - 1000), seed=seed)

    res = rigassay.run_assay("buridan", range(1000, 1010), play=play)
    assert res["modes"] == ["subtle", "off"] and {m for _, _, m in seen} == {"subtle", "off"} and len(seen) == 20
    assert all(res["results"][m]["analysis"]["all_passed"] for m in ("subtle", "off"))
    text = rigassay.summary(res)
    assert "BURIDAN'S PARADIGM ASSAY" in text and "B1 PASS" in text and "the control" in text and "p < 0.001" in text
    files = rigassay.save(res, tmp_path)
    assert json.loads(files[0].read_text())["rig"] == "buridan" and "off,B2" in files[1].read_text().replace('"', "")


def test_cancel_stops_an_assay_before_the_next_fly():
    import threading

    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        rigassay.run_flies("tethered", [1, 2], play=lambda t: tether(1), cancel=ev)


def test_the_criteria_are_written_in_the_module_docstring_with_their_minimums():
    doc = rigassay.__doc__
    for want in ("PRE-REGISTERED", "BEFORE the first run on any held-out seed", "T1", "T2", "BL1", "BL2", "B1", "B2", "F1", "3.0 Hz", "0.10", "10 deg", "0.15", "expected to FAIL"):
        assert want in doc, want
    assert rigassay.P_MAX == 0.01 and rigassay.VALIDATION_SEEDS == tuple(range(1000, 1010)) and rigassay.CONTROL_MODE == "off"


# --- the protocol kind ------------------------------------------------------------------------------------------------------------------
def test_the_four_rig_protocols_validate_and_hostile_ones_are_refused():
    for name in rigs.RIGS:
        p = protocol.load(ROOT / "protocols" / f"rig_{name}.yaml")
        assert p["rig"]["name"] == name and p["seeds"] == [0]
    bad = [dict(rig=dict(name="treadmill")), dict(rig=dict(name="buridan", mode="walls")), dict(rig=dict(name="buridan", seconds=-1)),
           dict(rig=dict(name="buridan", seconds=True)), dict(rig=dict(name="buridan", extra=1)), dict(rig="buridan"),
           dict(rig=dict(name="ball", scene="ceiling")), dict(rig=dict(name="buridan", individuality="wild")),
           dict(rig=dict(name="buridan"), surgery={"type:DNp01": -1}), dict(rig=dict(name="buridan"), stimuli=[]), dict(rig=dict(name="buridan"), assay="tmaze"),
           dict(rig=dict(name="tethered", omega=float("inf"))), dict(rig=dict(name=["buridan"]))]
    for b in bad:
        with pytest.raises(protocol.ProtocolError):
            protocol.check(dict(name="x", **b))


def test_a_rig_protocol_runs_each_seed_through_scene_run_and_summarises(tmp_path, monkeypatch):
    calls = []

    def fake(rig, seed, individuality, folder=None, **opt):
        calls.append((rig, seed, individuality, opt))
        return dict(summary=dict(stripe_deviation_deg=3.0 + seed, transits=5))

    monkeypatch.setattr(rigassay, "scene_run", fake)
    p = protocol.check(dict(name="rig-x", seed=5, flies=2, rig=dict(name="buridan", mode="none", seconds=20, individuality="off")))
    folder = protocol.run(p, tmp_path)
    s = json.loads((folder / "summary.json").read_text())
    assert [c[:3] for c in calls] == [("buridan", 5, "off"), ("buridan", 6, "off")] and calls[0][3] == dict(mode="none", seconds=20.0)
    assert s["rig_summary"]["6"]["stripe_deviation_deg"] == 9.0 and s["tags"]["trajectory"] == "MODEL PREDICTION"


# --- real-code smoke tests on the synthetic pack ------------------------------------------------------------------------------------
@pytest.fixture
def small(synthetic_pack):
    from kickthefly.core import simcore

    return simcore.new_brain(seed=3, warmup=50, individuality="off")


def test_every_rig_runs_on_a_real_brain_object_and_gives_a_well_formed_result(small):
    out = {}
    for rig in rigs.RIGS:
        out[rig] = rigassay.scene_run(rig, 3, "off", br=small, seconds=6.0)
        r = out[rig]
        assert r["seconds"] == pytest.approx(6.0, abs=0.05) and r["cols"] == list(rigs.COLS) and len(r["trace"]) == 60 and r["tags"] and r["summary"]
        json.dumps(r, default=str)                                                           # a result is exportable as it is
    assert set(out["fourfield"]["summary"]) >= {"pi", "seconds_odor", "seconds_air"}


def test_a_recorded_scene_run_writes_the_labs_files_and_the_trace(small, tmp_path):
    r = rigassay.scene_run("buridan", 3, "off", br=small, folder=tmp_path / "x", seconds=5.0)
    names = sorted(p.name for p in (tmp_path / "x").iterdir())
    for tail in ("-spikes.csv", "-rates.csv", "-group-rates.csv", "-kinematics.csv", ".npz", "-metadata.json", "-rig_trace.csv", "-rig_summary.json"):
        assert any(n.endswith(tail) for n in names), (tail, names)
    meta = json.loads(next((tmp_path / "x").glob("*-metadata.json")).read_text())
    assert meta["rig"] == "buridan" and meta["params"]["mode"] == "stripes" and meta["tags"]["steering_and_walking_from_DNs"] == "CONNECTOME"
    rows = next((tmp_path / "x").glob("*-rig_trace.csv")).read_text().splitlines()
    assert rows[0].split(",") == list(rigs.COLS) and len(rows) == len(r["trace"]) + 1
    kin = next((tmp_path / "x").glob("*-kinematics.csv")).read_text().splitlines()
    assert kin[0].startswith("time_ms,x,y,z,speed") and "buridan" in kin[1]


def test_the_same_seed_and_settings_give_the_same_trace(synthetic_pack):
    a = rigassay.scene_run("fourfield", 4, "off", seconds=6.0, backend="cpu")
    b = rigassay.scene_run("fourfield", 4, "off", seconds=6.0, backend="cpu")
    assert a["trace"] == b["trace"]


def test_the_cli_entry_points_write_results(synthetic_pack, tmp_path, capsys):
    args = types.SimpleNamespace(rig="tethered", rig_assay=None, seeds="2", individuality="off", out=str(tmp_path / "one"), workers=1, sim_backend="cpu")
    assert rigassay.main(args) == 0
    out = capsys.readouterr().out
    assert "Tethered flight simulator: seed 2" in out and "files written" in out and list((tmp_path / "one").glob("*-rig_trace.csv"))
    bad = types.SimpleNamespace(rig="treadmill", rig_assay=None, seeds=None, individuality=None, out=None, workers=1)
    assert rigassay.main(bad) == 2


def test_the_headless_dispatcher_reaches_the_rig_and_mini_paper_flags(synthetic_pack, capsys, tmp_path):
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import headless

    assert headless.main(k.parse_args(["--headless", "--minipaper", "list"])) == 0
    out = capsys.readouterr().out
    assert "von_reyn_2014" in out and "colomb_2012" in out
    assert headless.main(k.parse_args(["--headless", "--minipaper", "no_such_paper"])) == 2
    assert headless.main(k.parse_args(["--headless", "--rig", "ball", "--seeds", "1", "--individuality", "off", "--out", str(tmp_path / "r")])) == 0
    assert list((tmp_path / "r").glob("ball-seed1-*"))
