"""3.0 keeps old files working: configs without the new settings and keys, players' own key and pad bindings, training memory,
protocols and replays. The old inputs are written out by hand here, as a 2.13 install would have saved them."""
from __future__ import annotations

from kickthefly.core import config


OLD_213 = '''schema_version = 3

[graphics]
fullscreen = false

[brain]
mode = "lab"
science_popups = true
seed = 12

[keys]
forward = "w"
back = "s"
left = "a"
right = "d"
arena = "e"
loadout = "q"

[gamepad]
use = "righttrigger"

[loadout]
custom = ["hand", "sugar"]

[first_run]
tutorial_done = true
loadout_notice = false
'''


def load(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return config.Config.load(p)


def test_a_2_13_config_loads_with_the_new_settings_at_their_defaults(tmp_path):
    cfg = load(tmp_path, OLD_213)
    assert cfg.warnings == []
    assert cfg["brain.mode"] == "lab" and cfg["brain.seed"] == 12 and cfg["brain.science_popups"] is True
    assert cfg["brain.neurodex"] is True and cfg["brain.killcam"] is True and cfg["brain.neuron_of_day"] is True
    assert cfg.loadout["custom"] == ["hand", "sugar"] and cfg.first_run["tutorial_done"] is True
    assert cfg.keys["right"] == "d" and cfg.keys["neurodex"] == "d" and cfg.keys["killcam"] == ";"
    assert cfg.conflicts() == {}, "D is walk-right and Neurodex on purpose, and that isn't a conflict"


def test_the_new_settings_survive_a_save_and_load(tmp_path):
    cfg = load(tmp_path, OLD_213)
    cfg.set("brain.neurodex", False)
    cfg.set("brain.neuron_of_day", False)
    cfg.bind("killcam", "x")
    assert cfg.save()
    again = config.Config.load(tmp_path / "config.toml")
    assert again["brain.neurodex"] is False and again["brain.neuron_of_day"] is False and again["brain.killcam"] is True
    assert again.keys["killcam"] == "x" and again.keys["neurodex"] == "d" and again.warnings == []


def test_a_player_who_put_another_action_on_d_keeps_it_and_the_neurodex_is_unbound(tmp_path):
    cfg = load(tmp_path, OLD_213.replace('arena = "e"', 'arena = "d"').replace('right = "d"', 'right = "l"'))
    assert cfg.keys["arena"] == "d" and cfg.keys["right"] == "l"
    assert cfg.keys["neurodex"] == ""
    assert any("Neurodex" in w and "unbound" in w for w in cfg.warnings)
    assert cfg.action_for("d") == "arena" and cfg.actions_for("d") == ["arena"]


def test_a_player_who_used_the_semicolon_keeps_it(tmp_path):
    cfg = load(tmp_path, OLD_213.replace('loadout = "q"', 'loadout = ";"'))
    assert cfg.keys["loadout"] == ";" and cfg.keys["killcam"] == ""
    assert any("Kill cam" in w for w in cfg.warnings)


def test_a_pad_button_the_player_already_used_is_not_taken(tmp_path):
    cfg = load(tmp_path, OLD_213.replace('use = "righttrigger"', 'use = "dpup"'))
    assert cfg.pad["use"] == "dpup" and cfg.pad["neurodex"] == ""
    assert cfg.pad["killcam"] == "dpdown"
    assert any("neurodex" in w for w in cfg.warnings)


def test_defaults_do_not_collide_on_the_pad_or_keyboard():
    cfg = config.Config(None)
    assert cfg.conflicts() == {}
    pad = [b for b in cfg.pad.values() if b]
    assert len(pad) == len(set(pad))


def test_rebinding_away_from_d_removes_the_overlap():
    cfg = config.Config(None)
    ok, _ = cfg.bind("neurodex", "f9")
    assert ok and cfg.actions_for("d") == ["right"] and cfg.keys["neurodex"] == "f9"


def test_the_progress_file_sits_beside_old_training_memory_and_ignores_it(isolated_home):
    from kickthefly.core import memory
    from kickthefly.core import neurodex as nd

    d = memory.memory_dir()
    (d / "fly-memory.npz").write_bytes(b"old memory bytes")
    (d / "training-log.json").write_text("{}", encoding="utf-8")
    prog = nd.Progress()
    prog.mark("adult", "DNp01")
    prog.save()
    assert (d / "fly-memory.npz").read_bytes() == b"old memory bytes" and (d / "training-log.json").read_text() == "{}"
    assert (d / "neurodex.json").exists()


def test_old_protocols_still_load_and_ignore_the_bundle_fields(tmp_path):
    from kickthefly.lab import protocol

    p = protocol.check({"name": "old", "seed": 1, "flies": 1, "duration_s": 1, "warmup_s": 0.1,
                        "stimuli": [{"at_s": 0.1, "target": "loom"}], "recordings": [{"name": "a", "neurons": "loom"}]})
    assert p["seeds"] == [1] and p["stimuli"][0]["target"] == "loom"
    import pathlib
    for f in sorted(pathlib.Path(__file__).resolve().parent.parent.glob("protocols/*.yaml")):
        protocol.load(f)                                           # every bundled example protocol still loads


def test_old_replay_files_and_the_replay_format_are_unchanged():
    from kickthefly.core import replay

    assert replay.FORMAT == "kick-the-fly-replay" and replay.FORMAT_VERSION == 1
