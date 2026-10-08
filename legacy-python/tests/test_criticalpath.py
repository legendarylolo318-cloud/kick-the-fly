"""Critical path finder: the shortlist, the ranking, and that an interrupted run resumes."""
import json

import numpy as np

from conftest import needs_pack


@needs_pack
def test_shortlist_puts_the_driving_types_first():
    from kickthefly.core import simcore
    from kickthefly.lab import criticalpath

    br = simcore.new_brain(seed=0, warmup=0)
    picked = criticalpath.shortlist(br, "looming_escape", top=10)
    names = [c["type"] for c in picked]
    assert {"LC4", "LPLC2"} <= set(names), "the types being driven must be tried"
    assert names[:2] == ["LC4", "LPLC2"] or names[:2] == ["LPLC2", "LC4"]
    assert all(c["neurons"] > 0 for c in picked)
    assert all(picked[i]["influence"] >= picked[i + 1]["influence"] for i in range(len(picked) - 2))


@needs_pack
def test_silencing_the_driven_type_lowers_the_readout():
    """The lesion must survive the assay driving the same neurons, or it would measure nothing."""
    from kickthefly.lab import criticalpath

    plain = criticalpath.behavior_seed((1000, "looming_escape", None))
    cut = criticalpath.behavior_seed((1000, "looming_escape", "type:LC4,LPLC2"))
    assert cut["driven_hz"] < plain["driven_hz"] * 0.8
    assert plain["ratio"] > 2.0


@needs_pack
def test_drive_and_surgery_add_instead_of_overwriting():
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=1, warmup=0)
    rows = simcore.rows_of(br, "type:LC4")
    br.set_override(rows, -1)
    before = br.override[rows].copy()
    simcore.drive(br, rows, 0.5)
    assert np.array_equal(br.override[rows], before), "driving must not clear a lesion"
    assert np.all(br.drive_cur[rows] == 0.5) and br.driving
    simcore.undrive(br, rows)
    assert np.all(br.drive_cur[rows] == 0) and not br.driving
    assert np.array_equal(br.override[rows], before)


@needs_pack
def test_run_ranks_and_resumes(tmp_path):
    from kickthefly.lab import criticalpath

    seeds = (1000, 1001, 1002)
    types = ["LC4", "LPLC2"]
    res = criticalpath.run("looming_escape", seeds=seeds, workers=1, resume=tmp_path, types=types)
    assert [r["rank"] for r in res["ranked"]] == [1, 2]
    assert abs(res["ranked"][0]["effect_share"]) >= abs(res["ranked"][1]["effect_share"])
    assert res["ranked"][0]["effect_share"] < 0, "silencing a looming detector must lower the readout"
    assert res["control"]["mean"] > 0 and len(res["control"]["per_seed"]) == 3

    progress = tmp_path / "critical_path_progress.json"
    assert progress.exists()
    saved = json.loads(progress.read_text())
    assert set(saved["results"]) == set(types) and saved["target"] == "looming_escape"

    # a second run with the same target and seeds does no new work and gives the same numbers
    again = criticalpath.run("looming_escape", seeds=seeds, workers=1, resume=tmp_path, types=types)
    assert again["ranked"][0]["lesioned_mean"] == res["ranked"][0]["lesioned_mean"]
    assert again["seconds"] < 5, "a fully resumed run should not re-measure anything"

    # a different question starts over rather than mixing runs
    fresh = criticalpath._load_resume(tmp_path, "sugar_feeding", seeds)
    assert fresh == {}


@needs_pack
def test_csv_has_a_row_per_type(tmp_path):
    from kickthefly.lab import criticalpath

    res = criticalpath.run("looming_escape", seeds=(1000, 1001), workers=1, resume=tmp_path, types=["LC4"])
    path = criticalpath.to_csv(res, tmp_path / "out.csv")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("rank,cell_type")
    assert "LC4" in lines[1]
    assert "Critical path for looming_escape" in criticalpath.summary(res)
