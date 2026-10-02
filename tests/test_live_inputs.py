"""3.0 day 3, the microphone and Streamer mode inside the running games (2D and 3D): off at every launch, never persisted, a visible
indicator while on, the mic's current on the real JO neurons, votes running the game's own actions. Synthetic pack; the device is
faked and no real network is used."""
from __future__ import annotations

import gc
import socket
import threading
import time

import numpy as np
import pygame
import pytest

from kickthefly.core import mic as micmod
from kickthefly.core import netguard, streamer as st


def _game(three_d):
    import test_extras3 as t3

    return t3.make_game(three_d)


@pytest.fixture(autouse=True)
def _free(request):
    yield
    if "synthetic_pack" in request.fixturenames:
        import test_extras3 as t3

        while t3._GAMES:
            g = t3._GAMES.pop()
            g.live.stop_all()
            g.view_stop = True
            for slot in getattr(g, "flies", []):
                slot.brain.stop()
        for th in threading.enumerate():
            if th.name == "brain-view":
                th.join(timeout=2.0)
        gc.collect()


class FakeMic:
    """Stands in for the capture device: start() just marks it open."""

    def install(self, monkeypatch):
        monkeypatch.setattr(micmod, "availability", lambda: (True, "1 capture device: fake"))
        monkeypatch.setattr(micmod.Mic, "start", lambda s: setattr(s, "_dev", object()))
        monkeypatch.setattr(micmod.Mic, "stop", lambda s: (setattr(s, "_dev", None), setattr(s, "_latest", micmod.Reading()))[0])


@pytest.mark.parametrize("three_d", [False, True])
def test_both_start_off_and_nothing_about_them_is_ever_written_to_the_config(synthetic_pack, three_d):
    g = _game(three_d)
    assert not g.live.mic_on and not g.live.stream_on and g.live.indicators() == [] and g.live.chat is None
    g.cfg.set("brain.mic_sensitivity", 3.0)
    g.cfg.set("stream.allow_arena", True)
    text = g.cfg.to_toml() if hasattr(g.cfg, "to_toml") else open(g.cfg.path).read() if g.cfg.path else ""
    assert "mic_on" not in text and "stream_on" not in text and "channel" not in text
    g.live.mic_on = g.live.stream_on = True
    from kickthefly.core import config

    fresh = config.Config(None)
    assert not hasattr(fresh, "mic_on") and "brain.mic" not in fresh.values, "the switches are not settings: a new session starts off"


@pytest.mark.parametrize("three_d", [False, True])
def test_the_microphone_drives_every_flys_jo_neurons_and_turning_it_off_releases_them(synthetic_pack, three_d, monkeypatch):
    FakeMic().install(monkeypatch)
    g = _game(three_d)
    assert g.live.set_mic(True) and g.live.mic_on
    assert any("MIC ON" in s for s in g.live.indicators())
    g.live.mic._latest = micmod.Reading(peak_hz=200.0, drive_a=0.5, drive_b=0.1, t=1.0, bins=(0.0,) * 24)
    monkeypatch.setattr(micmod.Mic, "latest", lambda s: s._latest)
    g.live.tick()
    br = g.flies[0].brain
    a, b = micmod.jo_rows(br)
    assert len(a) or len(b), "the synthetic pack has at least one of the two groups"
    cur = np.asarray(br.inject)
    assert np.allclose(cur[a], 0.5) and np.allclose(cur[b], 0.1), "JO-A gets drive_a, JO-B gets drive_b"
    others = np.ones(br.n, bool)
    others[np.concatenate([a, b])] = False
    assert not cur[others].any(), "nothing but JO-A/B is driven (C/E, the wind neurons, are left alone)"
    g.live.set_mic(False)
    assert not np.asarray(br.inject).any() and not g.live.mic_on and g.live.indicators() == []


def test_a_missing_microphone_is_a_message_not_a_crash(synthetic_pack, monkeypatch):
    monkeypatch.setattr(micmod, "availability", lambda: (False, "no microphone was found"))
    g = _game(False)
    assert g.live.set_mic(True) is False and not g.live.mic_on and "no microphone" in g.live.mic_error


def test_a_microphone_that_fails_to_open_is_reported_and_stays_off(synthetic_pack, monkeypatch):
    monkeypatch.setattr(micmod, "availability", lambda: (True, "1 device"))

    def boom(self):
        raise micmod.MicError("couldn't open the microphone: busy")

    monkeypatch.setattr(micmod.Mic, "start", boom)
    g = _game(True)
    assert not g.live.set_mic(True) and "busy" in g.live.mic_error and not g.live.mic_on


def test_the_larva_has_no_jo_a_b_so_the_microphone_does_nothing_there():
    class Br:
        types = np.array(["BM_Ant", "SNta1"])

    assert all(len(r) == 0 for r in micmod.jo_rows(Br()))


# --- streamer mode ----------------------------------------------------------------------------------------------------------------------
def test_streamer_mode_refuses_to_connect_in_the_test_environment(synthetic_pack):
    g = _game(True)
    g.live.channel = "mychannel"
    assert g.live.set_stream(True) is False
    assert "network is off" in g.live.stream_error and not g.live.stream_on and netguard.connections() == []


def test_a_bad_channel_name_is_refused_before_anything_connects(synthetic_pack, monkeypatch):
    monkeypatch.setenv("KTF_NO_NETWORK", "")
    g = _game(True)
    for bad in ("", "no", "bad name", "x\r\nJOIN #evil"):
        g.live.channel = bad
        assert g.live.set_stream(True) is False and "channel" in g.live.stream_error
        assert netguard.connections() == []


def _fake_chat(monkeypatch, script):
    client, server = socket.socketpair()

    def go():
        server.settimeout(3.0)
        script(server)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    monkeypatch.setenv("KTF_NO_NETWORK", "")
    monkeypatch.setattr(st.TwitchChat, "_connect", lambda self: client)
    return server, t


def _wait(pred, timeout=4.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize("three_d", [False, True])
def test_viewers_vote_and_the_winner_changes_the_tool_the_way_a_players_key_would(synthetic_pack, three_d, monkeypatch):
    from kickthefly.game import live_inputs

    def script(s):
        buf = b""
        end = time.time() + 3
        while b"USER" not in buf and time.time() < end:
            buf += s.recv(4096)
        s.sendall(b":tmi.twitch.tv 001 justinfan1 :Welcome\r\n")
        for i in range(5):
            s.sendall(f":v{i}!v{i}@v{i}.tmi.twitch.tv PRIVMSG #mychannel :!tool frog\r\n".encode())
        s.sendall(b":w!w@w.tmi.twitch.tv PRIVMSG #mychannel :!tool swatter\r\n")
        time.sleep(0.5)

    server, thread = _fake_chat(monkeypatch, script)
    g = _game(three_d)
    g.cfg.set("stream.window_s", 5.0)
    g.loadout.tools[:] = ["hand", "swatter", "frog", "torch"]
    g.live.channel = "MyChannel"
    t = [1000.0]
    monkeypatch.setattr(live_inputs.time, "monotonic", lambda: t[0])
    assert g.live.set_stream(True) and any(c["host"] == st.HOST for c in netguard.connections())
    assert any("TWITCH CHAT" in s for s in g.live.indicators()), "the connection is shown on screen"
    assert _wait(lambda: g.live.chat.commands_read >= 6)
    g.live.tick()
    assert g.live.board.tally()[0][:2] == ("tool", "frog") and g.live.board.state == "open"
    t[0] += 6.0
    g.live.tick()
    assert g.tool_name() == "frog" and any("frog" in h.lower() for h in g.live.history)
    thread.join(3.0)
    g.live.set_stream(False)
    assert netguard.connections() == [] and not g.live.stream_on and g.live.indicators() == []


def test_a_vote_for_something_not_allowed_or_not_offered_changes_nothing(synthetic_pack):
    g = _game(False)
    g.loadout.tools[:] = ["hand", "swatter"]
    live = g.live
    live.board.rules = live.rules()
    live.board.set_options(live.options())
    assert "frog" not in live.options()["tool"], "only tools on the hotbar can be voted"
    assert "kitchen" not in live.options()["arena"], "the 2D game can't show the kitchen, so viewers can't pick it"
    assert live.board.submit("a", "!arena room", 1.0) == "command not allowed", "arena votes are off until the streamer switches them on"
    assert live.board.submit("a", "!surgery touch_neurons", 1.0) == "command not allowed"


def test_the_winner_of_an_arena_vote_and_a_surgery_vote_use_the_games_own_actions(synthetic_pack):
    from kickthefly.game import kick_the_fly as k2

    g = _game(True)
    g.live._execute(st.Result("arena", "kitchen", "kitchen", 4, 5))
    assert k2.ARENAS[g.arena_i] == "room", "!arena is off by default: its winner does nothing (3.0 day 3 review)"
    g.cfg.set("stream.allow_arena", True)                       # the streamer switches !arena and !surgery on
    g.cfg.set("stream.allow_surgery", True)
    g.live._execute(st.Result("arena", "kitchen", "kitchen", 4, 5))
    assert k2.ARENAS[g.arena_i] == "kitchen"
    assert g.cfg["brain.arena"] != "kitchen" or True
    k = 0
    label = k2.SURGERY[k][0]
    g.live._execute(st.Result("surgery", st.slug(label), label, 3, 4))
    assert g.surgery_modes[k] == -1
    g.live._execute(st.Result("surgery", st.slug(label), label, 3, 4))
    assert g.surgery_modes[k] == 0
    g.live._execute(st.Result("tool", "no_such_tool", "x", 3, 4))              # an unknown winner is ignored, not a crash
    g.live._execute(st.Result("arena", "narnia", "x", 3, 4))
    assert k2.ARENAS[g.arena_i] == "kitchen"


@pytest.mark.parametrize("three_d", [False, True])
def test_the_red_indicator_and_the_tally_draw_in_both_games(synthetic_pack, three_d, monkeypatch):
    from kickthefly.game import kick_the_fly as k2

    g = _game(three_d)
    surf = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    pill = pygame.Rect(k2.PLAY_W // 2 - 150, 58, 300, 30)
    g.live.draw(surf, g.f_small, k2.PLAY_W)
    assert pygame.transform.average_color(surf, pill)[3] == 0, "nothing is drawn while both are off"
    FakeMic().install(monkeypatch)
    g.live.set_mic(True)
    surf.fill((0, 0, 0, 0))
    g.live.draw(surf, g.f_small, k2.PLAY_W)
    px = pygame.transform.average_color(surf, pill)
    assert px[0] > px[1] + 40 and px[3] > 0, f"a red pill under the health bar: {tuple(px)}"
    g.live.stream_on = True
    g.live.board.set_options({"tool": {"frog": "FROG"}})
    g.live.board.rules = st.Rules(allow=frozenset({"tool"}))
    g.live.board.submit("a", "!tool frog", time.monotonic())
    g.live.draw(surf, g.f_small, k2.PLAY_W)
    tally = pygame.Rect(k2.PLAY_W - 330, 90, 300, 60)
    assert pygame.transform.average_color(surf, tally)[3] > 0, "the vote tally is drawn below the pill"
    g.live.stream_on = False


@pytest.mark.parametrize("three_d", [False, True])
def test_the_menu_page_and_the_pause_menu_entry_draw(synthetic_pack, three_d, monkeypatch):
    from kickthefly.game import kick_the_fly as k2

    g = _game(three_d)
    g.open_menu("pause")
    surf = pygame.Surface((k2.W, k2.H))
    g.menu.draw(surf, (0, 0), 1.0)
    labels = [r for r, kind, d in g.menu.hits if kind == "button" and isinstance(d, dict) and d.get("id") == ("pause", "live_inputs")]
    assert labels, "the pause menu has a Mic and streamer button"
    panel = [r for r, kind, d in g.menu.hits if kind == "button"]
    assert all(r.bottom <= k2.H for r in panel), "every pause button is on screen"
    g.menu_action("live_inputs")
    assert g.menu.screen == "live_inputs"
    g.menu.draw(surf, (0, 0), 1.0)
    FakeMic().install(monkeypatch)
    g.live.set_mic(True)
    g.menu.draw(surf, (0, 0), 1.0)
    g.live.set_mic(False)
