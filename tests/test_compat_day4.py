"""3.0 day 4 compatibility: old command lines, protocols, configs, validation results and loadouts still work, and the new pieces are
additive (nothing the old code read changed shape)."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_old_command_lines_parse_the_same_and_the_new_flags_are_off():
    from kickthefly.game import kick_the_fly as k

    for argv in (["--headless", "--validate", "--workers", "4", "--seeds", "1000-1009", "--out", "x.json"],
                 ["--headless", "--protocol", "protocols/smoke.yaml"], ["--headless", "--critical-path", "looming_escape", "--resume"],
                 ["--headless", "--threshold-sweep", "--thresholds", "4", "5"], ["--2d", "--seed", "3"], ["--selftest"]):
        a = k.parse_args(argv)
        assert not a.sensitivity and not a.tournament and not a.race and not a.netsci and not a.sleep_deprivation
        assert a.sens_params is None and a.lanes is None and a.races is None and a.match_seconds is None
    a = k.parse_args(["--headless", "--validate"])
    assert a.validate and a.seeds is None and a.out is None


def test_every_protocol_in_the_repo_still_validates_and_the_new_example_is_one_of_them():
    from kickthefly.lab import protocol

    names = []
    for f in sorted((ROOT / "protocols").glob("*.yaml")):
        data = protocol.load(f)
        names.append(data.get("name") or f.stem)
    assert "sleep-deprivation-assay" in names and len(names) >= 38


def test_the_validation_result_keeps_its_shape_and_defaults(synthetic_pack):
    from kickthefly.lab import lab, validation

    res = validation.run(seeds=(1000, 1001), workers=1, include={"looming_escape"})
    assert {"app_version", "brain", "backend", "device", "created", "seconds", "seeds", "workers", "n_neurons", "synapses", "wiring", "lab_params",
            "thresholds", "loom", "sweet_n", "tests"} <= set(res)
    assert res["wiring"] is None and res["lab_params"] == dict(lab.DEFAULTS), "a default run records the defaults and no parameter override"
    assert "params" not in res and "sensitivity" not in res


def test_the_expected_validation_outcomes_are_untouched():
    from kickthefly.lab import validation

    assert sum(validation.EXPECTED.values()) == 12 and len([k for k, v in validation.EXPECTED.items() if not v]) == 11
    assert validation.SEEDS == tuple(range(1000, 1010)) and validation.RATIO_MIN == 1.5 and validation.P_MAX == 0.01


def test_the_lab_parameters_and_thresholds_keep_their_defaults():
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import lab

    assert lab.DEFAULTS["noise_std"] == 0.05 and lab.DEFAULTS["bias"] == 0.20 and lab.DEFAULTS["target_rate_hz"] == 5.0
    assert lab.DEFAULTS["ext_gain"] == 4.0 and lab.DEFAULTS["gain_adapt"] == 0.002
    assert k.THRESH["sleep"] == 2.0 and k.THRESH["fire"] == 3.0 and k.THRESH["escape"] == 4.0 and k.THRESH["walk"] == 3.0


def test_an_old_config_without_any_new_key_loads_and_the_loadout_tools_are_unchanged(tmp_path):
    from kickthefly.core import config, loadout

    p = tmp_path / "config.toml"
    p.write_text('[brain]\nseed = 7\nmode = "lab"\n[access]\npalette = "blue-yellow"\n')
    cfg = config.Config.load(p)
    assert cfg["brain.seed"] == 7 and cfg["access.palette"] == "blue-yellow"
    assert len(loadout.TOOL_NAMES) == 18, "the tournament and racing are modes, not tools: the loadout editor's tool list did not change"


def test_the_pause_menu_still_has_every_old_entry_and_the_new_one():
    import inspect

    from kickthefly.ui import menu

    src = inspect.getsource(menu.Menu._page_pause)
    for entry in ("Resume", "Neurodex", "Settings", "Mic and streamer", "Share", "Save State", "Load State", "Quit", "Fly arcade"):
        assert f'"{entry}"' in src, entry


def test_the_arcade_wallet_file_is_new_and_nothing_else_reads_it(isolated_home):
    from kickthefly.core import paths, points

    assert not (paths.get().data_dir / "arcade_points.json").exists()
    points.Wallet().settle(5, 2.0, False)
    assert (paths.get().data_dir / "arcade_points.json").exists()
