"""Tests for Connectome Diff Mode: region-by-region diff and divergence timeline."""

import numpy as np
import pytest

from conftest import needs_pack


@needs_pack
def test_connectome_diff_looming_gf_lesion(tmp_path):
    from kickthefly.core import simcore
    from kickthefly.lab import diffmode
    from kickthefly.sim.wiring import Wiring

    g = simcore.pack()[0]
    types = g.type.astype(str)
    dnp01 = np.flatnonzero(types == "DNp01")

    cfg_a = dict(wiring=Wiring(), lesion_rows=None, label="Control")
    cfg_b = dict(wiring=Wiring(), lesion_rows=dnp01, label="GF Lesion")

    res = diffmode.run_connectome_diff(cfg_a, cfg_b, stimulus="looming", steps=100, seed=1000)
    assert res["kind"] == "connectome_diff"
    assert res["steps"] == 100
    assert len(res["rows"]) > 0
    assert "divergence_timeline" in res
    assert len(res["divergence_timeline"]) > 0

    # Top divergence row
    top = res["top_divergence"]
    assert top is not None
    assert "rate_a" in top and "rate_b" in top and "diff" in top

    # Export check
    folder = tmp_path / "diff_export"
    diffmode.save(res, folder)
    assert (folder / "connectome_diff.json").exists()
    assert (folder / "connectome_diff.csv").exists()


@needs_pack
def test_connectome_diff_identical_brains_zero_divergence():
    from kickthefly.lab import diffmode
    from kickthefly.sim.wiring import Wiring

    cfg = dict(wiring=Wiring(), lesion_rows=None, label="Identical")
    res = diffmode.run_connectome_diff(cfg, cfg, stimulus="calm", steps=50, seed=1000)
    assert res["steps"] == 50
    # Two identical brains with same seed and inputs should have zero regional divergence
    for r in res["rows"]:
        assert abs(r["diff"]) < 1e-4, f"Expected 0 diff for identical brains: {r['diff']}"
