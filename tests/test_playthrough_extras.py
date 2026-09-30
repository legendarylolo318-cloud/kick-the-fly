"""The playthrough bot's 3.0 checks, run on the synthetic pack (they check the wiring: the real run is --playthrough)."""
from __future__ import annotations

import pytest

from kickthefly.lab import playthrough as pt


@pytest.fixture
def rig(synthetic_pack):
    r = pt.Rig(False, "cpu")
    r.game.x3.async_build = False
    for slot in r.game.flies:
        slot.brain.stop()
    import time
    time.sleep(0.15)
    r.take_snapshot(_tmp_snapshot())
    yield r
    r.close()


def _tmp_snapshot():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="ktf-snap-")) / "snap.ktfsave"


def run(fn, *a):
    res = pt.Result(id="t", group="extra", brain="adult")
    pt.guarded(res, fn, *a, res)
    return res


def test_neurodex_extra_passes(rig):
    res = run(pt.extra_neurodex, rig)
    assert res.status == pt.PASS, res.failures
    assert res.metrics["stimulated_type"] in ("DNp01", "MDN", "DNp09")


def test_killcam_extra_passes(rig):
    res = run(pt.extra_killcam, rig)
    assert res.status == pt.PASS, res.failures
    assert res.metrics["risers"]


def test_share_extra_passes(rig):
    res = run(pt.extra_share, rig)
    assert res.status == pt.PASS, res.failures


def test_bundle_extra_passes(synthetic_pack, tmp_path, monkeypatch):
    monkeypatch.setenv("KICK_THE_FLY_SIM_BACKEND", "cpu")
    res = pt.Result(id="t", group="extra", brain="adult")
    pt.guarded(res, pt.extra_bundle, "cpu", res, tmp_path)
    assert res.status == pt.PASS, res.failures
    assert res.metrics["verdict"] == "MATCH (bit-exact)"
