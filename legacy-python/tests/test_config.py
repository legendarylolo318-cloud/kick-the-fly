"""Settings file: defaults, round trip, corrupt and hostile files, clamping and key rebinding."""
from kickthefly.core import config


def test_defaults_and_round_trip(tmp_path):
    path = tmp_path / "config.toml"
    c = config.Config.load(path)
    assert c["brain.mode"] == "play" and c["graphics.fps_cap"] == 60 and not c.warnings
    c.set("brain.mode", "lab")
    c.set("audio.master", 0.35)
    c.set("brain.sim_speed", 0.25)
    c.set("brain.seed", 1234)
    c.bind("training", "f5")
    assert c.save()
    d = config.Config.load(path)
    assert d["brain.mode"] == "lab" and d["audio.master"] == 0.35 and d["brain.sim_speed"] == 0.25
    assert d["brain.seed"] == 1234 and d.keys["training"] == "f5" and d.lab and d.tags_on()


def test_corrupt_file_falls_back_with_warning(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("this is [not toml = = =\n\x00")
    c = config.Config.load(path)
    assert c["brain.mode"] == "play"
    assert c.warnings and "default" in c.warnings[0]
    assert (tmp_path / "config.toml.bad").exists() and not path.exists()


def test_bad_values_are_clamped_or_ignored(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[audio]\nmaster = 7.0\nsfx = "loud"\n[brain]\nmode = "godmode"\nsim_speed = 0.25\nunknown = 1\n'
                    '[graphics]\nfps_cap = 144\n[controls]\nfov = -5\n[keys]\ntraining = "escape"\nsurgery = "b"\n')
    c = config.Config.load(path)
    assert c["audio.master"] == 1.0              # clamped
    assert c["audio.sfx"] == 1.0                 # wrong type: default
    assert c["brain.mode"] == "play"             # not an option: default
    assert c["brain.sim_speed"] == 0.25 and c["graphics.fps_cap"] == 144 and c["controls.fov"] == 50
    assert c.keys["training"] == "t"             # reserved key refused
    assert not c.conflicts()                     # surgery=b collided with big_view=b: one was reset


def test_missing_file_is_defaults(tmp_path):
    c = config.Config.load(tmp_path / "nope" / "config.toml")
    assert c["graphics.panel_mode"] == "solid" and not c.warnings


def test_rebind_swaps_on_conflict():
    c = config.Config(None)
    ok, msg = c.bind("training", "o")            # o belongs to surgery
    assert ok and "swapped" in msg
    assert c.keys["training"] == "o" and c.keys["surgery"] == "t" and not c.conflicts()
    ok, msg = c.bind("training", "escape")
    assert not ok and c.keys["training"] == "o"
    assert c.action_for("t") == "surgery"


def test_reset_tab():
    c = config.Config(None)
    c.set("controls.fov", 90)
    c.bind("training", "f5")
    c.set("audio.master", 0.1)
    changed = c.reset_tab("Controls")
    assert "controls.fov" in changed and "keys" in changed and c.keys["training"] == "t"
    assert c["audio.master"] == 0.1              # other tabs untouched


def test_every_setting_has_a_tooltip_and_brain_tags():
    for s in config.SETTINGS:
        assert s.tip and s.tab in config.TABS
        assert s.default == config._coerce(s, s.default)
    for key in ("brain.pain_level", "brain.immortal", "brain.sim_speed", "brain.seed"):
        assert config.BY_KEY[key].tag in (config.CONNECTOME, config.GAME_RULE)


def test_backend_setting_hidden_on_windows():
    s = config.BY_KEY["graphics.backend"]
    assert not config.visible(s, True, "win32") and config.visible(s, True, "linux")


def test_default_keys_match_the_old_hotkeys():
    old = {"big_view": "b", "surgery": "o", "training": "t", "duel": "x", "arena": "e", "pain": "p", "immortal": "i",
           "mute": "m", "screenshot": "f12", "gif": "g", "panel": "v", "menu_size": "u", "fullscreen": "f11",
           "spawn": "n", "reset": "r", "help": "h", "free_mouse": "tab"}
    c = config.Config(None)
    for a, k in old.items():
        assert c.keys[a] == k
    assert not c.conflicts()


def test_science_popups_default_and_migration(tmp_path):
    # 1. Fresh config has cards off
    path = tmp_path / "fresh_config.toml"
    c_fresh = config.Config.load(path)
    assert c_fresh["brain.science_popups"] is False
    assert config.BY_KEY["brain.science_popups"].tip == "Show a short card when the fly does something real flies were shown to do."

    # 2. Old config (schema 1 / unversioned) with science_popups = true migrates to false once
    old_path = tmp_path / "old_config.toml"
    old_path.write_text('[brain]\nscience_popups = true\n')
    c_old = config.Config.load(old_path)
    assert c_old["brain.science_popups"] is False
    assert c_old.dirty is True
    assert any("migrated" in w for w in c_old.warnings)

    # 3. User who turns them on afterward keeps them on
    c_old.set("brain.science_popups", True)
    assert c_old.save()
    reloaded = config.Config.load(old_path)
    assert reloaded["brain.science_popups"] is True

