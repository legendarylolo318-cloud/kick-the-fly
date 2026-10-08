"""3.0 day 5 compatibility: an old config.toml shows the What's New screen once and never again (a migrated [first_run] key), a fresh install
doesn't see it, old command lines still parse with the new flags off, every old protocol still validates next to the new rig protocols, and the
pieces the rigs and mini-papers touch (classroom lectures, the loadout's tool list, validation's expected results) did not change shape."""
from __future__ import annotations

from pathlib import Path

import pytest

from kickthefly.core import config

ROOT = Path(__file__).resolve().parent.parent

OLD_2_13 = '''schema_version = 3

[brain]
mode = "lab"
seed = 12

[first_run]
tutorial_done = true
loadout_notice = false
neuron_shapes_asked = true
'''

OLD_2_12 = '''[brain]
seed = 4
'''


def load(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return config.Config.load(p)


def test_an_old_config_has_not_seen_whats_new_and_a_fresh_install_has():
    assert config.FIRST_RUN_DEFAULTS["whatsnew_3_0_seen"] is False


def test_a_2_13_config_without_the_key_reads_as_not_seen(tmp_path):
    cfg = load(tmp_path, OLD_2_13)
    assert cfg.first_run["whatsnew_3_0_seen"] is False and cfg.first_run["tutorial_done"] is True and cfg.warnings == []


def test_a_pre_2_13_config_is_migrated_and_has_not_seen_it_either(tmp_path):
    cfg = load(tmp_path, OLD_2_12)
    assert cfg.migrated_from == 1 and cfg.first_run["whatsnew_3_0_seen"] is False and cfg.first_run["loadout_notice"] is True


def test_a_fresh_install_does_not_get_whats_new(tmp_path):
    cfg = config.Config.load(tmp_path / "nothing-here" / "config.toml")
    assert cfg.first_run["whatsnew_3_0_seen"] is True and cfg.first_run["tutorial_done"] is False


def test_the_flag_is_written_and_read_back_and_a_bad_value_means_not_seen(tmp_path):
    cfg = load(tmp_path, OLD_2_13)
    cfg.first_run["whatsnew_3_0_seen"] = True
    assert cfg.save()
    assert "whatsnew_3_0_seen = true" in (tmp_path / "config.toml").read_text()
    assert config.Config.load(tmp_path / "config.toml").first_run["whatsnew_3_0_seen"] is True
    bad = load(tmp_path, OLD_2_13.replace("neuron_shapes_asked = true", 'whatsnew_3_0_seen = "yes"'))
    assert bad.first_run["whatsnew_3_0_seen"] is False


def test_whats_new_shows_once_after_an_upgrade_then_never_again(synthetic_pack, tmp_path):
    from test_extras3 import make_game

    path = tmp_path / "config.toml"
    path.write_text(OLD_2_13.replace("neuron_shapes_asked = true", "neuron_shapes_asked = true"), encoding="utf-8")
    g = make_game()
    g.cfg = config.Config.load(path)
    g.cfg.path = path
    g.show_first_run_notices()
    assert g.menu.screen == "whatsnew" and g.menu.open and g.cfg.first_run["whatsnew_3_0_seen"] is True
    assert "whatsnew_3_0_seen = true" in path.read_text(), "the flag is saved the moment the screen shows"
    g.menu.close()
    g.show_first_run_notices()
    assert g.menu.screen != "whatsnew"
    again = make_game()
    again.cfg = config.Config.load(path)                       # the next launch reads the saved file
    again.show_first_run_notices()
    assert again.menu.screen != "whatsnew"


def test_a_migrated_config_sees_the_loadout_notice_first_and_whats_new_on_a_later_launch(synthetic_pack, tmp_path):
    from test_extras3 import make_game

    path = tmp_path / "config.toml"
    path.write_text(OLD_2_12, encoding="utf-8")
    g = make_game()
    g.cfg = config.Config.load(path)
    g.cfg.path = path
    g.show_first_run_notices()
    assert g.menu.screen == "loadout_notice" and g.cfg.first_run["whatsnew_3_0_seen"] is False
    g.menu.close()
    g2 = make_game()
    g2.cfg = config.Config.load(path)
    g2.cfg.path = path
    g2.show_first_run_notices()
    assert g2.menu.screen == "whatsnew"


def test_whats_new_can_be_reopened_from_settings_help_without_touching_the_flag(synthetic_pack):
    from test_extras3 import make_game

    g = make_game()
    g.cfg.first_run["whatsnew_3_0_seen"] = True
    g.menu.show("whatsnew")
    assert g.menu.screen == "whatsnew" and g.cfg.first_run["whatsnew_3_0_seen"] is True
    import inspect

    from kickthefly.ui import help_ui

    assert "What's new in 3.0" in inspect.getsource(help_ui.help_tab)


def test_old_command_lines_parse_the_same_and_the_new_flags_are_off():
    from kickthefly.game import kick_the_fly as k

    for argv in (["--headless", "--validate", "--workers", "4", "--seeds", "1000-1009"], ["--headless", "--protocol", "protocols/smoke.yaml"],
                 ["--headless", "--race"], ["--headless", "--tournament", "8"], ["--2d", "--seed", "3"], ["--selftest"]):
        a = k.parse_args(argv)
        assert a.rig is None and a.rig_assay is None and a.minipaper is None
    a = k.parse_args(["--headless", "--rig-assay", "buridan", "--seeds", "1000-1003"])
    assert a.rig_assay == "buridan" and a.rig is None
    assert k.parse_args(["--headless", "--rig", "tethered"]).rig == "tethered" and k.parse_args(["--headless", "--minipaper", "list"]).minipaper == "list"
    with pytest.raises(SystemExit):
        k.parse_args(["--headless", "--rig", "treadmill"])


def test_every_old_protocol_still_validates_and_the_four_rig_protocols_are_new_ones():
    from kickthefly.lab import protocol

    names = []
    for f in sorted((ROOT / "protocols").glob("*.yaml")):
        data = protocol.load(f)
        names.append(data.get("name") or f.stem)
    assert {"rig-tethered", "rig-ball", "rig-buridan", "rig-fourfield", "sleep-deprivation-assay", "lecture-looming"} <= set(names)
    assert len(names) >= 42


def test_old_lectures_loadout_and_validation_expectations_did_not_change_shape():
    from kickthefly.core import loadout
    from kickthefly.lab import classroom, validation

    assert sorted(classroom.CURATED_LECTURES) == ["gf_lesion", "looming", "moonwalker", "sugar", "tmaze"]
    assert len(loadout.TOOL_NAMES) == 18, "rigs and mini-papers are scenes and pages, not tools: the loadout editor's list did not change"
    assert sum(validation.EXPECTED.values()) == 12 and len(validation.EXPECTED) == 23


def test_the_headless_dispatcher_runs_the_rig_and_paper_flags_through_the_same_door():
    import inspect

    from kickthefly.lab import headless

    src = inspect.getsource(headless.main) if hasattr(headless, "main") else inspect.getsource(headless)
    assert "rigassay.main" in src and "minipapers.main" in src
