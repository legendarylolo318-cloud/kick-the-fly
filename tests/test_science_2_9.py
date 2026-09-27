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
    assert len(g["p1"]) == 86 and np.char.startswith(brain.types[g["p1"]], "pC1").all()   # pMP-e/pMP4 synonym
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
    inst = np.asarray(brain.instance).astype(str)
    # a rightward rotation is front-to-back (T4a/T5a) on the right eye and back-to-front (T4b/T5b) on the left
    right = g["optomotor_right"]
    assert set(np.unique(brain.types[right])) == {"T4a", "T5a", "T4b", "T5b"}
    for rows, types, side in ((g["t45_prog_r"], ("T4a", "T5a"), "_R"), (g["t45_regr_l"], ("T4b", "T5b"), "_L")):
        assert np.isin(brain.types[rows], types).all() and np.char.endswith(inst[rows], side).all()
    assert set(right) == set(g["t45_prog_r"]) | set(g["t45_regr_l"])
    assert len(g["dna_steer_r"]) == 2 and np.char.endswith(inst[g["dna_steer_r"]], "_R").all()
    assert not np.isin(g["ol_intrinsic_not_t45"], right).any()
    # the EMD stage drives exactly the two channels a rightward rotation matches
    stage = assays.emd_stage(g, assays.OPTOMOTOR_YAW)
    driven = np.concatenate([r for r, a in stage if a > 0])
    assert set(driven) == set(right)

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
    # new arenas are only ever appended: a pre-2.7 save names its arena by index alone
    assert k.ARENAS[:8] == ("room", "fan", "flypaper", "pool", "lamp", "escaperoom", "field", "orchard")
    assert set(config.BY_KEY["brain.arena"].options) == set(k.ARENAS)
    assert k.thermo_gradient(-1.0) == (1.0, 0.0) and k.thermo_gradient(1.0) == (0.0, 1.0)
    assert k.thermo_gradient(0.0) == (0.0, 0.0) and k.thermo_gradient(-3.0) == (1.0, 0.0)

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
        "thermosensory_pathways.yaml", "bitter_grn_to_dng28.yaml", "co2_orn_to_pn.yaml",
        "grooming_hierarchy.yaml", "light_clock_dfb.yaml"
    ]
    for p_name in new_protocols:
        p_path = Path("protocols") / p_name
        assert p_path.exists(), f"Missing protocol {p_name}"
        data = protocol.load(p_path)
        assert data["name"] == p_name.replace(".yaml", "")
        assert len(data["seeds"]) == 5


def test_model_assumptions_registered():
    from kickthefly.lab import lab

    titles = [a[0].lower() for a in lab.ASSUMPTIONS]
    for needle in ("courtship song", "aggression lunge", "optomotor emd", "thermo arena", "day/night"):
        assert any(needle in t for t in titles), needle
    tags = {a[0]: a[1] for a in lab.ASSUMPTIONS}
    assert all(tags[a[0]] == "GAME RULE" for a in lab.ASSUMPTIONS if any(
        n in a[0].lower() for n in ("courtship song", "aggression lunge", "optomotor emd", "thermo arena", "day/night")))


def test_new_surgery_groups_are_appended():
    """Save states store surgery modes by position, so 2.9's groups go after every older one."""
    from kickthefly.game import kick_the_fly as k

    labels = [x[0] for x in k.SURGERY]
    assert labels.index("Every neuron") == len(labels) - 5
    assert labels[-4:] == ["Song command neuron (pIP10)", "Courtship neurons P1 (pC1, pMP-e/pMP4)",
                           "Aggression neurons TK-FruM (AVLP727m)", "Sleep neurons: dorsal fan-shaped body (FB6, FB7)"]


def test_new_thresholds_are_lab_game_rules():
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import lab

    for name in ("song", "aggression", "sleep", "co2", "hot_pn", "cold_pn"):
        p = lab.BY_NAME[f"thresh.{name}"]
        assert p[2] == "rule" and p[3] == k.THRESH[name]


def test_sun_follows_day_night_cycle():
    from kickthefly.game import outdoors

    fixed = {"outdoor.sun_az": 100.0, "outdoor.sun_el": 30.0}
    assert outdoors.sun_now(fixed, 123.0) == (100.0, 30.0)                  # off by default
    assert outdoors.sun_now(fixed, 0.0, day_s=600)[1] < 0                     # midnight
    assert outdoors.sun_now(dict(fixed, **{"outdoor.day_s": 600.0}), 300.0)[1] > 55   # noon; the Lab setting wins
    assert outdoors.dusk(45.0) == 1.0 and outdoors.dusk(-30.0) == 0.0
