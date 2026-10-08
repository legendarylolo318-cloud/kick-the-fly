"""Tests for the E-PG compass bump formation and persistence probe.

Verifies that the probe accurately evaluates the central complex ring network
and honestly reports the negative result (no persistent bump under raw LIF weights).
"""
import pytest
from conftest import needs_pack

pytestmark = [needs_pack]


def test_epg_compass_negative_validation():
    from kickthefly.lab import compass

    # Run with 2 seeds for fast test execution
    res = compass.probe_epg_compass(seeds=(1000, 1001), warmup=50)

    assert "passed" in res
    assert res["passed"] is False  # Must fail honestly without weight tuning
    assert res["n_epg"] == 46
    assert res["mean_contrast"] < compass.CONTRAST_THRESHOLD
    assert res["mean_persistence_ms"] < compass.PERSISTENCE_MIN_MS
    assert "finding" in res
    assert "untuned weights" in res["finding"]


def test_epg_wind_probe_uses_the_arena_transduction_and_reports_honestly():
    """The wind test drives the real JO-C/E neurons through outdoors.wind_drive and reports every number it
    measured, pass or fail. Two seeds here for speed; the validation suite runs all ten."""
    from kickthefly.game import outdoors
    from kickthefly.lab import compass

    res = compass.probe_epg_wind(seeds=(1000, 1001))
    assert res["n_glomeruli"] == 16 and res["wind_speed_m_s"] == outdoors.WIND_FULL
    assert res["passed"] is (res["mean_contrast"] >= compass.CONTRAST_THRESHOLD
                             and res["mean_persistence_ms"] >= compass.PERSISTENCE_MIN_MS)
    assert 0.0 < res["direction_tracking_p"] <= 1.0
    assert "No persistent bump formed." in res["finding"] or "A bump formed" in res["finding"]


def test_wind_transduction_is_lateralised():
    from kickthefly.game import outdoors

    left, right = outdoors.wind_drive(0.0, 90.0, 6.0)    # fly faces +x; wind from +z, which is its right side
    assert right > 0.99 and left < 0.01
    left, right = outdoors.wind_drive(0.0, 0.0, 6.0)     # head-on: both antennae alike
    assert abs(left - right) < 1e-9 and abs(left + right - 1.0) < 1e-9
    assert outdoors.wind_drive(0.0, 90.0, 0.0) == (0.0, 0.0)
