"""3.0 day 2 in the Python API (Fly): driver lines, thermogenetics, patch clamp, imaging, drugs. Synthetic pack: plumbing only."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.sim import wiring as W


@pytest.fixture
def fly(synthetic_pack):
    from kickthefly import Fly

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    yield Fly(seed=3, warmup_s=0.5, learn=False)
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


def test_line_and_neurons_by_line(fly):
    d = fly.line("SS00727")
    assert d["matched"] == {"DNp01": 2} and d["off_target"] == "some" and d["neurons"] == 2
    assert set(fly.neurons("line:SS00727")) == set(np.flatnonzero(fly.brain.types == "DNp01"))
    rec = fly.record({"gf": "line:SS00727"})
    fly.step(0.2)
    assert rec.rates()["gf"] >= 0.0


def test_express_temperature_and_step(fly):
    fly.express("trpa1", "line:SS00727")
    fly.temperature(20.0, kinetics="steady")
    rec = fly.record({"gf": "type:DNp01"})
    fly.step(1.0)
    cold = rec.rates()["gf"]
    fly.temperature(34.0)
    fly.step(1.0)
    hot = rec.rates(start_s=1.0)["gf"]
    assert hot > cold + 15                                                             # the synthetic DNp01 already fires at rest; plumbing check
    fly.unexpress()
    assert not fly.brain.injecting
    with pytest.raises(ValueError, match="within"):
        fly.temperature(90)
    with pytest.raises(ValueError, match="kinetics"):
        fly.temperature(25, kinetics="fast")


def test_express_accepts_an_array_of_rows(fly):
    rows = fly.neurons("type:MDN")
    fly.express("shibire", rows).temperature(35.0, kinetics="steady")
    fly.step(0.1)
    assert set(np.flatnonzero(fly.brain.inject)) == set(rows) and (fly.brain.inject[rows] < 0).all()


def test_patch_returns_an_if_curve_and_leaves_no_current(fly):
    c = fly.patch("type:MDN", [0.0, 0.3], duration_ms=200, repeats=2)
    assert c["mode"] == "embedded" and len(c["rate_hz"]) == 2 and "MODEL" in c["tag"] and not fly.brain.injecting
    iso = fly.patch("type:MDN", [0.0, 0.1, 0.5], duration_ms=200, repeats=1, mode="isolated")
    assert iso["rate_hz"][-1] > iso["rate_hz"][0]


def test_image_returns_roi_traces_and_works_with_thermogenetics(fly):
    r = fly.image(1.0, fps=20)
    assert r.dff.shape[0] == 20 and "Mushroom Body" in r.roi_names
    r2 = fly.image(1.0, rois=["type:DNp01"], indicator="gcamp6f")
    assert r2.roi_names == ["type:DNp01"] and r2.meta["indicator"].startswith("GCaMP6f")
    fly.express("trpa1", "type:DNp01").temperature(35.0, kinetics="steady")
    r3 = fly.image(1.0, rois=["type:DNp01"], shot_noise=False)
    assert r3.true_dff[-1, 0] > r3.true_dff[0, 0]                                      # driven neurons brighten


def test_drug_and_washout_are_exact(fly):
    before = fly.brain.sim.W_csr.data.copy()
    info = fly.drug("cholinergic", 0.5)
    assert info["changed"] > 0 and "acetylcholine" in info["label"]
    info2 = fly.drug("picrotoxin", 1.0)                                                # adds to the first
    assert "gaba" in info2["label"] and "acetylcholine" in info2["label"]
    fly.washout()
    assert np.array_equal(fly.brain.sim.W_csr.data, before)
    with pytest.raises(Exception, match="unknown drug"):
        fly.drug("octopamine", 0.5)


def test_kinetics_chosen_before_expressing_still_applies(fly):
    """Found by the playthrough bot: temperature(..., kinetics='steady') before express() was silently dropped."""
    fly.temperature(34.0, kinetics="steady")
    fly.express("trpa1", "type:DNp01")
    assert fly._thermo.kinetics == "steady"
    fly.step(0.1)
    assert fly.brain.injecting                                                          # instant: no 1 s ramp
    fly.temperature(20.0)
    fly.step(0.1)
    assert not fly.brain.injecting                                                      # and instant off
