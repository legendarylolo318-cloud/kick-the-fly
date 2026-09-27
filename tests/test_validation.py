"""The validation suite as tests: every behavior must pass or fail exactly as it did in this release's held-out run.

A test that starts failing is a regression. A test that starts passing is good news, but the README, the dashboard
and the popups promise the old result, so it must be reviewed and validation.EXPECTED updated on purpose.
"""
import os

import pytest

from conftest import needs_pack

pytestmark = [needs_pack, pytest.mark.validation]


@pytest.fixture(scope="module")
def results():
    from kickthefly.lab import validation
    workers = int(os.environ.get("KTF_VALIDATION_WORKERS", "0")) or None
    res = validation.run(workers=workers)
    out = os.environ.get("KTF_VALIDATION_OUT")
    if out:
        from pathlib import Path
        validation.save_results(res, Path(out))
    print("\n" + validation.summary(res))
    return {t["id"]: t for t in res["tests"]}


@pytest.mark.parametrize("test_id", ["looming_escape", "mdn_backward", "sugar_feeding", "antenna_grooming_circuit",
                                     "adn_grooming_motor", "mb_conditioning", "epg_compass",
                                     "epg_compass_wind", "courtship_song", "bitter_grn_to_dng28",
                                     "co2_orn_to_pn", "hot_trn_to_vp2pn", "cold_trn_to_vp3pn",
                                     "grooming_hierarchy", "optomotor_turning", "p1_courtship_song"])
def test_matches_expected(results, test_id):
    from kickthefly.lab import validation
    t = results[test_id]
    assert t["passed"] == validation.EXPECTED[test_id], f"{t['name']}: {t['measured']}"


def test_popups_only_for_passing_tests(results):
    from kickthefly.lab import validation
    res = dict(n_neurons=1, synapses=2, lab_params={}, tests=list(results.values()))
    events = validation.passing_events(res, 1, 2)
    assert set(events) == {t["popup_event"] for t in results.values() if t["passed"] and t["popup_event"]}
    assert all(t["passed"] for t in events.values())
    assert validation.passing_events(res, 1, 3) == {}                  # another brain pack: no popups
    from kickthefly.lab import lab
    bad = dict(res, lab_params=dict(lab.DEFAULTS, noise_std=0.1))
    assert validation.passing_events(bad, 1, 2) == {}                  # modified parameters: no popups
