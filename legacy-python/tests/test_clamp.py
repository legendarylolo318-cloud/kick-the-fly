"""Tests for Neural Clamp: isolating wiring effects from input variation via dynamic clamping."""

import numpy as np
import pytest

from conftest import needs_pack


@needs_pack
def test_neural_clamp_looming_gf_lesion(tmp_path):
    from kickthefly.core import simcore
    from kickthefly.lab import clamp

    # 1. Record reference looming run
    ref = clamp.record_reference_run(stimulus="looming", steps=100, seed=1000)
    assert ref["kind"] == "reference_run"
    assert ref["steps"] == 100
    assert "DYNAMIC CLAMP" in ref["notice"]
    assert len(ref["spikes"]) == 100
    assert "optic lobe" in ref["region_rates"] or len(ref["region_rates"]) > 0

    # 2. Replay with DNp01 lesioned
    g = simcore.pack()[0]
    types = g.type.astype(str)
    dnp01 = np.flatnonzero(types == "DNp01")

    res = clamp.run_neural_clamp(ref, lesion_rows=dnp01, seed=1000)
    assert res["kind"] == "neural_clamp"
    assert "DYNAMIC CLAMP" in res["notice"]
    assert res["lesioned_neurons"] == len(dnp01)

    # Giant fiber should be completely silenced in clamped run
    gf = res["readouts"]["giant_fiber_DNp01"]
    assert gf["free_hz"] > 10.0, f"Expected active free GF: {gf['free_hz']}"
    assert gf["clamped_hz"] == 0.0, f"Expected silenced clamped GF: {gf['clamped_hz']}"
    assert gf["diff_hz"] < -10.0

    # 3. Export round-trip
    folder = tmp_path / "clamp_export"
    clamp.save(res, folder)
    assert (folder / "neural_clamp.json").exists()
    assert (folder / "neural_clamp_diff.csv").exists()
    csv_text = (folder / "neural_clamp_diff.csv").read_text()
    assert "DYNAMIC CLAMP" in csv_text


@needs_pack
def test_neural_clamp_threshold_wiring():
    from kickthefly.lab import clamp
    from kickthefly.sim.wiring import Wiring

    ref = clamp.record_reference_run(stimulus="sugar", steps=80, seed=1001)
    res = clamp.run_neural_clamp(ref, wiring=Wiring(min_synapses=10), seed=1001)
    assert res["steps"] == 80
    assert res["wiring"]["min_synapses"] == 10
    assert "sugar_MN9" in res["readouts"]
