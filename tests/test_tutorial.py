"""The first-launch tutorial (2.13): once per install, skippable at any step, replayable, keyboard and pad, no flashing."""
from __future__ import annotations

import pygame
import pytest

from conftest import needs_pack
from kickthefly.core import config
from kickthefly.game import kick_the_fly as k2
from kickthefly.ui import tutorial

_GAMES: list = []


@pytest.fixture(autouse=True)
def _free_games():
    yield
    import gc
    import threading
    while _GAMES:
        g = _GAMES.pop()
        g.view_stop = True
        for slot in getattr(g, "flies", []):
            slot.brain.stop()
    for th in threading.enumerate():
        if th.name == "brain-view":
            th.join(timeout=2.0)
    gc.collect()


def game(three_d=False, cfg=None):
    pygame.init()
    state = {"seed": 5}
    k2.load_brain(state)
    cfg = cfg or config.Config(None)
    if three_d:
        from kickthefly.game import kick3d
        g = kick3d.Game3D(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), state["brain"], state["view"], state["graph"],
                          state["weights"], cfg=cfg)
    else:
        g = k2.Game(pygame.Surface((k2.W, k2.H)), state["brain"], state["view"], state["graph"], state["weights"], cfg=cfg)
    _GAMES.append(g)
    return g


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)


@needs_pack
def test_it_never_starts_by_itself_only_from_the_run_loop_and_only_once(tmp_path):
    path = tmp_path / "config.toml"
    cfg = config.Config(path)
    g = game(cfg=cfg)
    assert g.tutorial is None, "Game.__init__ must not start it (tests, headless runs and the bot never see it)"
    assert not cfg.first_run["tutorial_done"]
    g.show_first_run_notices()
    assert g.tutorial is not None and g.tutorial.active and g.tutorial.step == "move"
    g.tutorial.finish()
    assert cfg.first_run["tutorial_done"] and path.exists() and g.tutorial is None
    assert config.Config.load(path).first_run["tutorial_done"] is True, "the flag is in config.toml"
    g.show_first_run_notices()
    assert g.tutorial is None, "shown once per install"


@needs_pack
def test_a_migrated_config_gets_the_loadout_notice_not_the_tutorial(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("schema_version = 2\n")
    cfg = config.Config.load(path)
    g = game(cfg=cfg)
    g.show_first_run_notices()
    assert g.tutorial is None and g.menu.screen == "loadout_notice"
    assert not cfg.first_run["loadout_notice"] and not config.Config.load(path).first_run["loadout_notice"], "once"
    g.menu.draw(pygame.Surface((k2.W, k2.H)), (0, 0), 1.0)


@needs_pack
@pytest.mark.parametrize("three_d", [False, True])
def test_every_step_completes_and_the_flag_is_set(tmp_path, three_d, monkeypatch):
    cfg = config.Config(tmp_path / "config.toml")
    g = game(three_d, cfg)
    g.start_tutorial()
    t = g.tutorial
    surf = g.screen
    seen = []
    for expected in tutorial.STEPS:
        assert t.step == expected
        seen.append(expected)
        t.draw(surf)                                                         # every step draws
        if expected == "move":
            if three_d:
                g.player.pos[0] += 2.0
            else:
                t.travel = tutorial.MOVE_PIXELS + 1
        elif expected == "tool":
            g.tool_uses += 1
        elif expected == "brain":
            monkeypatch.setattr(tutorial.time, "perf_counter", lambda: t.t_step + tutorial.BRAIN_SECONDS + 1)
        elif expected == "sugar":
            assert g.tool_name() == "sugar", "the sugar step puts sugar in your hand"
            g.tool_uses += 1
            g.last_tool_used = "sugar"
        elif expected == "loadout":
            g.open_loadout_editor()
            t.poll()
            g.menu.close()
        t.poll() if t.active else None
        if expected != "loadout":
            monkeypatch.undo()
    assert seen == list(tutorial.STEPS) and not t.active and g.tutorial is None
    assert cfg.first_run["tutorial_done"]


@needs_pack
def test_it_can_be_skipped_at_every_step_by_key_button_or_pad(tmp_path):
    for how in ("backspace", "button", "pad"):
        for step in range(len(tutorial.STEPS)):
            cfg = config.Config(tmp_path / f"{how}{step}.toml")
            g = game(cfg=cfg)
            g.start_tutorial()
            t = g.tutorial
            for _ in range(step):
                t.next()
            t.draw(g.screen)
            if how == "backspace":
                assert g.handle(key(pygame.K_BACKSPACE), 1.0) is True
            elif how == "button":
                r = next(r for r, what in t.buttons if what == "skip")
                assert g.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=r.center), 1.0)
            else:
                assert t.pad_event(["big_view"])
            assert not t.active and g.tutorial is None and t.skipped
            assert cfg.first_run["tutorial_done"], "skipping counts: it is not shown again"
            g.view_stop = True


@needs_pack
def test_enter_and_the_next_button_and_pad_a_skip_a_step_only():
    g = game()
    g.start_tutorial()
    t = g.tutorial
    g.handle(key(pygame.K_RETURN), 1.0)
    assert t.step == "tool" and t.active
    t.draw(g.screen)
    r = next(r for r, what in t.buttons if what == "next")
    g.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=r.center), 1.0)
    assert t.step == "brain"
    assert t.pad_event(["up"]) and t.step == "sugar"
    assert not t.pad_event(["use"]), "other pad buttons are not the tutorial's"
    assert g.handle(key(pygame.K_a), 1.0) is True, "other keys reach the game"


@needs_pack
def test_it_can_be_replayed_and_a_replay_leaves_the_flag_alone(tmp_path):
    cfg = config.Config(tmp_path / "config.toml")
    cfg.first_run["tutorial_done"] = True
    g = game(cfg=cfg)
    g.menu.show("settings")
    g.menu.tab = "Help"
    g.menu.draw(pygame.Surface((k2.W, k2.H)), (0, 0), 1.0)
    replay_btn = next(d for r, kind, d in g.menu.hits if d["id"] == ("help-btn", "Replay the tutorial"))
    replay_btn["click"]()
    assert not g.menu.open and g.tutorial is not None and g.tutorial.replay
    g.tutorial.draw(g.screen)
    g.tutorial.finish(skipped=True)
    assert cfg.first_run["tutorial_done"] is True


@needs_pack
def test_the_highlight_does_not_pulse_with_reduced_flashing(monkeypatch):
    widths = {}
    real_rect = pygame.draw.rect

    def spy(surf, color, rect, width=0, *a, **k):
        if width and pygame.Rect(rect).size == g.view_rect.inflate(8, 8).size:
            widths.setdefault(flag, set()).add(width)
        return real_rect(surf, color, rect, width, *a, **k)

    for flag in (True, False):
        cfg = config.Config(None)
        cfg.set("access.reduced_flashing", flag)
        g = game(cfg=cfg)
        g.view_rect = pygame.Rect(900, 54, 360, 200)
        g.start_tutorial()
        g.tutorial.i = tutorial.STEPS.index("brain")
        monkeypatch.setattr(pygame.draw, "rect", spy)
        for t in (0.0, 0.4, 0.8, 1.2, 1.6):
            monkeypatch.setattr(tutorial.time, "perf_counter", lambda t=t: 100.0 + t)
            g.tutorial.t_step = 100.0
            g.tutorial.draw(g.screen)
        monkeypatch.undo()
    assert widths[True] == {3}, "reduced flashing: a steady outline"
    assert len(widths[False]) > 1, "otherwise it swells slowly (never a flash)"


@needs_pack
def test_larger_text_and_colorblind_palettes_apply_to_the_new_ui():
    from kickthefly.ui import loadout_ui

    for pal in ("default", "blue-yellow", "high-contrast"):
        cfg = config.Config(None)
        cfg.set("access.palette", pal)
        cfg.set("access.larger_text", True)
        g = game(cfg=cfg)
        assert loadout_ui.palette(cfg)["select"] == loadout_ui.PALETTES[pal]["select"]
        g.start_tutorial()
        g.tutorial.draw(g.screen)
        g.open_loadout_editor()
        g.menu.draw(g.screen, (0, 0), 1.0)
        assert g.menu.f_text.get_height() > 16, "larger text reaches the menu fonts"
        g.menu.close()
        g.view_stop = True
    tags = {loadout_ui.tag_color(cfg, t) for t in ("CONNECTOME", "GAME RULE")}
    assert len(tags) == 2, "the two tags stay distinguishable in every palette (and are labeled in words)"
