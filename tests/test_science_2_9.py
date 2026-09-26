"""Test coverage for gemini/science-2.9 additions:
- Courtship song (pIP10 -> ps1 MN, pulse song buzz audio)
- Male-male aggression (FruM/TK AVLP727m/pC1, multi-fly lunge)
- Optomotor (Reichardt/EMD motion drive, T4/T5 -> DNa01/DNa02)
- Thermotaxis arena ("thermo", TRN_VP2 hot / TRN_VP3 cold gradient)
- Circadian cycle and dFB sleep readout
- New validation test definitions and pass/fail statuses
- New YAML protocols in protocols/
"""
import numpy as np
import pytest
from conftest import needs_pack


@pytest.fixture(scope="module")
def brain():
    from kickthefly.core import simcore
    return simcore.new_brain(seed=100)


def test_courtship_song_groups_and_sound(brain):
    from kickthefly.lab import assays
    from kickthefly.game import kick_the_fly as k

    g = assays.groups(brain)
    assert "pip10" in g and len(g["pip10"]) == 2
    assert "ps1" in g and len(g["ps1"]) == 2
    assert "song_wm" in g and len(g["song_wm"]) > 0

    # Sound check
    snd = k.Sound()
    if snd.ok:
        assert "pulse_song" in snd.fx


def test_aggression_circuit_and_tag(brain):
    from kickthefly.game import kick_the_fly as k

    assert "aggression" in brain.names
    agg_idx = brain.col["aggression"]
    agg_rows = np.flatnonzero(brain.det_id == agg_idx)
    assert len(agg_rows) > 0  # AVLP727m + pC1 cluster
    assert k.REACTION_SOURCE["LUNGE"] == "rule"
    assert k.POPUP_SOURCE["LUNGE!"] == "rule"


def test_optomotor_emd_and_groups(brain):
    from kickthefly.lab import assays
    from kickthefly.game import outdoors

    g = assays.groups(brain)
    assert "optomotor_right" in g and len(g["optomotor_right"]) > 0
    assert "dna_steer_r" in g and len(g["dna_steer_r"]) > 0

    # Test EMD calculation
    right_yaw = outdoors.emd_motion_drive(1.5)
    assert right_yaw["prog_r"] > 0 and right_yaw["regr_l"] > 0
    assert right_yaw["prog_l"] == 0 and right_yaw["regr_r"] == 0

    left_yaw = outdoors.emd_motion_drive(-1.5)
    assert left_yaw["prog_l"] > 0 and left_yaw["regr_r"] > 0
    assert left_yaw["prog_r"] == 0 and left_yaw["regr_l"] == 0


def test_thermotaxis_arena_and_groups(brain):
    from kickthefly.lab import assays
    from kickthefly.game import kick_the_fly as k
    from kickthefly.core import config

    assert "thermo" in k.ARENAS
    assert "thermo" in config.BY_KEY["brain.arena"].options

    g = assays.groups(brain)
    assert "trn_vp2" in g and len(g["trn_vp2"]) == 7
    assert "vp2_pn" in g and len(g["vp2_pn"]) > 0
    assert "trn_vp3" in g and len(g["trn_vp3"]) == 7
    assert "vp3_pn" in g and len(g["vp3_pn"]) > 0


def test_circadian_cycle_and_sleep(brain):
    from kickthefly.game import outdoors

    az_noon, el_noon = outdoors.diurnal_cycle(300.0, day_length_s=600.0)
    assert el_noon > 0  # daylight (+60 deg)
    az_mid, el_mid = outdoors.diurnal_cycle(0.0, day_length_s=600.0)
    assert el_mid < 0  # nighttime (-30 deg)

    clk_day = outdoors.circadian_clock_drive(45.0)
    assert clk_day["morning_cells"] > 0
    clk_night = outdoors.circadian_clock_drive(-10.0)
    assert clk_night["morning_cells"] == 0
    assert clk_night["evening_cells"] == 1.0

    assert outdoors.dfb_sleep_state(2.5, threshold=2.0) is True
    assert outdoors.dfb_sleep_state(1.5, threshold=2.0) is False

    # Sleep detail group in brain
    assert "sleep" in brain.names
    slp_idx = brain.col["sleep"]
    slp_rows = np.flatnonzero(brain.det_id == slp_idx)
    assert len(slp_rows) > 0  # FB6 and FB7


def test_new_protocols_load():
    from pathlib import Path
    from kickthefly.lab import protocol

    new_protocols = [
        "courtship_song.yaml", "male_aggression.yaml", "optomotor.yaml",
        "thermotaxis.yaml", "bitter_avoidance.yaml", "co2_avoidance.yaml",
        "grooming_hierarchy.yaml", "day_night_cycle.yaml"
    ]
    for p_name in new_protocols:
        p_path = Path("protocols") / p_name
        assert p_path.exists(), f"Missing protocol {p_name}"
        data = protocol.load(p_path)
        assert data["name"] == p_name.replace(".yaml", "")
        assert len(data["seeds"]) == 5


def test_model_assumptions_registered():
    from kickthefly.lab import lab

    titles = [a[0] for a in lab.ASSUMPTIONS]
    assert any("courtship pulse-song" in t.lower() for t in titles)
    assert any("aggression lunge" in t.lower() for t in titles)
    assert any("reichardt" in t.lower() for t in titles)
    assert any("thermotaxis" in t.lower() for t in titles)
    assert any("circadian" in t.lower() for t in titles)
