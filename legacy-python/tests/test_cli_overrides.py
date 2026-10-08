"""3.1.0 review: a command-line flag is in effect for the session but is never written to config.toml (--autopilot was saved, so every
later launch started in spectator mode with no tools)."""
from kickthefly.core import config


def test_a_command_line_flag_is_not_saved(tmp_path):
    path = tmp_path / "config.toml"
    cfg = config.Config.load(path)
    cfg.set("brain.seed", 7)
    cfg.set_for_session("brain.autopilot", True)
    cfg.set_for_session("brain.arena", "orchard")
    assert cfg["brain.autopilot"] is True and cfg["brain.arena"] == "orchard"
    assert cfg.save()
    again = config.Config.load(path)
    assert again["brain.autopilot"] is False and again["brain.arena"] == config.BY_KEY["brain.arena"].default
    assert again["brain.seed"] == 7                       # an ordinary change is still saved


def test_changing_the_setting_in_game_saves_the_players_choice(tmp_path):
    path = tmp_path / "config.toml"
    cfg = config.Config.load(path)
    cfg.set_for_session("brain.autopilot", True)
    cfg.set("brain.autopilot", False)                     # Y in the game
    cfg.set("brain.autopilot", True)                      # and on again: the player's own choice now
    cfg.save()
    assert config.Config.load(path)["brain.autopilot"] is True
