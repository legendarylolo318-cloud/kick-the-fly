"""3.0 day 3 review (Opus): regression tests for the bugs found while testing the predators, weather, kitchen, microphone and
Streamer mode adversarially. Each test failed before its fix. Synthetic pack only for game plumbing; nothing here is a result,
and no real microphone or network is used."""
from __future__ import annotations

import gc
import threading

import numpy as np
import pygame
import pytest

from kickthefly.core import netguard
from kickthefly.core import streamer as st


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


def _game(three_d):
    import test_extras3 as t3

    return t3.make_game(three_d)


# --- Settings: every choice has a label (the kitchen had none, so Settings > Brain crashed while in the kitchen) --------------
def test_every_choice_setting_has_one_label_per_option():
    from kickthefly.core import config

    for s in config.SETTINGS:
        if s.kind == "choice" and s.labels:
            assert len(s.labels) == len(s.options), f"{s.key}: {len(s.options)} options, {len(s.labels)} labels"


def test_settings_brain_tab_draws_while_in_the_kitchen(synthetic_pack):
    from kickthefly.game import kick_the_fly as k2

    g = _game(True)
    g.set_setting("brain.arena", "kitchen", save=False)
    g.open_menu("settings")
    g.menu.tab = "Brain"
    g.menu._page_error = None
    g.menu.draw(pygame.Surface((k2.W, k2.H)), (0, 0), 1.0)
    assert getattr(g.menu, "_page_error", None) is None, f"Settings > Brain showed an error page: {g.menu._page_error}"  # labels[9]


# --- protocols: hostile day 3 blocks give a ProtocolError, never another exception --------------------------------------------
@pytest.mark.parametrize("block", [
    {"predator": {"kind": []}},
    {"predator": {"kind": {"a": 1}}},
    {"predator": {"kind": ["frog"]}},
    {"assay": "hum_demo", "assay_options": {"conditions": [{}, []]}},
    {"assay": "predator_escape", "assay_options": {"kinds": [[], {"x": 1}]}},
    {"assay": "predator_escape", "assay_options": []},
    {"assay": "hum_demo", "assay_options": "steady"},
])
def test_hostile_day3_protocol_blocks_are_refused_cleanly(block):
    from kickthefly.lab import protocol as P

    with pytest.raises(P.ProtocolError):
        P.check({"name": "x", "duration_s": 1, **block})


# --- Streamer mode: a vote can only ever do what the allowlist allows when it wins ---------------------------------------------
class _Chat:
    """A connection that is live at once and hands over whatever the test queues."""

    def __init__(self, channel):
        self.channel, self.host, self.port = channel, "fake", 0
        self.state, self.error, self.q, self.commands_read = "off", "", [], 0

    def start(self):
        self.state = "live"

    def stop(self):
        self.state = "off"

    def describe(self):
        return "fake"

    def drain(self):
        out, self.q = self.q, []
        return out


def test_a_vote_for_a_command_switched_off_mid_round_does_not_run(synthetic_pack, monkeypatch):
    from kickthefly.game import kick_the_fly as k2

    g = _game(True)
    live = g.live
    live.chat_factory = _Chat
    g.cfg.set("stream.allow_arena", True)
    g.cfg.set("stream.min_votes", 1)
    clock = [1000.0]
    monkeypatch.setattr(st.time, "monotonic", lambda: clock[0])
    import kickthefly.game.live_inputs as li

    monkeypatch.setattr(li.time, "monotonic", lambda: clock[0])
    assert live.set_stream(True, "somechannel")
    live.chat.q = [("viewer1", "!arena kitchen"), ("viewer2", "!arena kitchen")]
    live.tick()
    assert live.board.state == "open"
    g.cfg.set("stream.allow_arena", False)                           # the streamer switches !arena off while the round is open
    before = k2.ARENAS[g.arena_i]
    clock[0] += 2.0
    live.tick()                                                      # the rules refresh
    clock[0] += 60.0
    live.tick()                                                      # the round closes
    assert k2.ARENAS[g.arena_i] == before, "a vote for a command the streamer switched off changed the arena"


# --- the Twitch reader: no hidden second connection, no leaked socket, nothing injected into what it sends ----------------------
class _SlowServer:
    """A fake connection whose 'connect' blocks until released, as a slow network does."""

    def __init__(self):
        self.release = threading.Event()
        self.socks = []

    def factory(self):
        self.release.wait(5.0)
        a, b = __import__("socket").socketpair()
        self.socks.append((a, b))
        return a


def test_turning_streamer_mode_off_and_on_while_connecting_never_leaves_a_hidden_connection():
    srv = _SlowServer()
    chat = st.TwitchChat("somechannel", sock_factory=srv.factory)
    chat.start()
    chat.stop()                                                     # off while the first connect is still blocked
    chat.start()                                                    # and straight back on
    srv.release.set()
    deadline = __import__("time").monotonic() + 5.0
    while __import__("time").monotonic() < deadline and len(srv.socks) < 2:
        __import__("time").sleep(0.05)
    __import__("time").sleep(0.5)
    readers = [t for t in threading.enumerate() if t.name == "streamer-chat" and t.is_alive()]
    try:
        assert len(readers) <= 1, f"{len(readers)} chat reader threads are running for one switch"
        assert len(netguard.connections()) == len(readers), "every running reader is shown on screen, and only those"
    finally:
        chat.stop()
        for a, b in srv.socks:
            a.close()
            b.close()
    assert not netguard.connections()


def test_nothing_the_server_sends_can_put_a_second_command_on_the_wire():
    import socket
    import time

    a, b = socket.socketpair()
    chat = st.TwitchChat("somechannel", sock_factory=lambda: a)
    chat.start()
    sent = b""
    try:
        b.sendall(b":tmi.twitch.tv 001 x :Welcome\r\nPING :tmi\rJOIN #otherchannel\r\n")
        b.settimeout(0.2)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and b"PONG" not in sent:      # waits on the data, not a fixed sleep (load-proof)
            try:
                chunk = b.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            sent += chunk
        time.sleep(0.3)
        try:
            sent += b.recv(4096)
        except (socket.timeout, OSError):
            pass
    finally:
        chat.stop()
        b.close()
    assert b"PONG" in sent, f"the reader never answered the PING: {sent!r}"
    lines = [x for x in sent.split(b"\r\n") if x]
    assert not any(b"\r" in x or b"\n" in x for x in lines), f"a line break inside a line went out: {sent!r}"
    assert not any(x.startswith(b"JOIN #otherchannel") for x in lines), "the server's CR started a second command"


def test_a_failed_tls_handshake_closes_the_raw_socket(monkeypatch):
    import socket
    import ssl

    made = []

    class Raw:
        closed = False

        def close(self):
            self.closed = True

    def fake_create(addr, timeout=None):
        made.append(Raw())
        return made[-1]

    class Ctx:
        def wrap_socket(self, raw, server_hostname=None):
            raise ssl.SSLError("handshake failed")

    monkeypatch.setattr(socket, "create_connection", fake_create)
    monkeypatch.setattr(ssl, "create_default_context", lambda: Ctx())
    chat = st.TwitchChat("somechannel")
    with pytest.raises(ssl.SSLError):
        chat._connect()
    assert made and made[0].closed, "the TCP connection under a failed TLS handshake was left open"


# --- the neuPrint skeleton fetch goes through the same network switch (headless runs and tests never fetch) ----------------------
def test_the_skeleton_fetch_respects_the_network_switch(monkeypatch):
    from kickthefly.sim import morphology

    monkeypatch.delenv("KICK_THE_FLY_OFFLINE", raising=False)
    monkeypatch.setenv("KTF_NO_NETWORK", "1")
    assert not morphology.network_allowed(), "KTF_NO_NETWORK (set by every headless run and the tests) must stop the neuPrint fetch too"
    monkeypatch.setenv("KTF_NO_NETWORK", "")
    netguard.disable("a headless run")
    try:
        assert not morphology.network_allowed()
    finally:
        netguard.enable()


# --- predators: a held fly stays held; a click elsewhere does not pull it out of the frog's mouth ----------------------------------
def _catch(g, kind="frog"):
    from kickthefly.game import kick_the_fly as k2
    from kickthefly.game import predators as pr

    slot = g.flies[0]
    at = np.array([float(slot.fly.p[k2.THX][0]) + 0.6, 0.0, float(slot.fly.p[k2.THX][2])]) if g.three_d else \
        np.array([float(slot.fly.p[k2.THX][0]) + 140.0, float(k2.FLOOR)])
    assert g.preds.spawn(kind, at)
    i = 0
    while not g.preds.held and i < int(90 / pr.DT):
        g.preds.step(i * pr.DT)
        i += 1
    assert g.preds.held, "a fly that stays put is caught"
    return slot, i


@pytest.mark.parametrize("three_d", [False, True])
def test_a_mouse_release_does_not_free_a_fly_a_predator_is_holding(synthetic_pack, three_d):
    from kickthefly.game import kick_the_fly as k2
    from kickthefly.game import predators as pr

    g = _game(three_d)
    slot, i = _catch(g)
    slot.fly.grabbed = None                         # what any left mouse-up does to the focused fly (both games)
    g.preds.step((i + 1) * pr.DT)
    assert slot.fly.grabbed == k2.THX, "the frog let go because the player released a mouse button somewhere"


@pytest.mark.parametrize("three_d", [False, True])
def test_a_fly_in_one_predators_mouth_is_not_prey_for_another(synthetic_pack, three_d):
    g = _game(three_d)
    slot, i = _catch(g, "frog")
    holder = g.preds.held[slot]
    targets = g.preds._targets()
    assert not any(t.id is slot and t.alive for t in targets), "a fly being eaten was still a live target for the mantis"
    assert g.preds.held[slot] is holder


# --- the Python API: hear() takes the protocol block's ranges, attack() refuses what is not a predator's name --------------------
def test_the_api_refuses_a_hum_it_cannot_build_and_a_predator_that_is_not_a_name(synthetic_pack):
    from kickthefly import Fly

    fly = Fly(seed=1)
    for bad in (dict(seconds=1e6), dict(hz=float("nan")), dict(amp=5.0), dict(ipi_ms=1.0), dict(sensitivity=0.0), dict(hz=True)):
        with pytest.raises(ValueError):
            fly.hear(**bad)
    for bad in ([], {"a": 1}, "toad"):
        with pytest.raises(ValueError):
            fly.attack(bad)


# --- weather: the wet air reaches every fly, not only the first one --------------------------------------------------------------
def _second_fly(g):
    import time

    g.spawn_fly()
    t0 = time.perf_counter()
    while g._new_slot is None and time.perf_counter() - t0 < 60:
        time.sleep(0.05)
    g._poll_spawn()
    assert len(g.flies) == 2


def test_rain_drives_the_humidity_neurons_of_every_fly(synthetic_pack):
    g = _game(True)
    g.set_setting("brain.arena", "field", save=False)
    _second_fly(g)
    seen = {id(s): [] for s in g.flies}
    for s in g.flies:
        real = s.brain.poke
        s.brain.poke = (lambda real, out: lambda region, side, strength, recruit=None: (out.append(region), real(region, side, strength, recruit))[1])(real, seen[id(s)])
    g.lab_params["weather.rain"] = 1.0
    t0 = g.clock.now
    for i in range(int(5 * 60)):
        g.frame += 1
        g._environment(t0 + (i + 1) / 60)
    counts = [seen[id(s)].count("humid") for s in g.flies]
    assert counts[0] > 0 and counts[1] > 0, f"humidity pokes per fly: {counts} (the second fly's wet air was never felt)"
    assert counts[0] == counts[1], "the same wet air for both"


# --- the kitchen: leaving it lets go of a fly that was in the vinegar trap --------------------------------------------------------
@pytest.mark.parametrize("to", ["room", "field"])
def test_leaving_the_kitchen_frees_a_fly_from_the_trap(synthetic_pack, to):
    from kickthefly.game import kick3d

    g = _game(True)
    g.set_setting("brain.arena", "kitchen", save=False)
    slot = g.flies[0]
    g.kitchen.trapped[id(slot)] = 0.5
    slot.fly.grabbed = kick3d.THX                               # what the trap does every frame
    g.set_setting("brain.arena", to, save=False)
    assert g.kitchen is None
    assert slot.fly.grabbed is None, "the trap is gone but the fly is still held (pinned to the player's hand point)"


# --- the microphone: a device that stops delivering sound stops driving the neurons ---------------------------------------------
def test_a_microphone_that_goes_quiet_releases_the_jo_neurons(synthetic_pack, monkeypatch):
    from kickthefly.core import mic as micmod
    import kickthefly.game.live_inputs as li

    monkeypatch.setattr(micmod, "availability", lambda: (True, "1 capture device: fake"))
    monkeypatch.setattr(micmod.Mic, "start", lambda s: setattr(s, "_dev", object()))
    monkeypatch.setattr(micmod.Mic, "stop", lambda s: setattr(s, "_dev", None))
    clock = [100.0]
    monkeypatch.setattr(li.time, "monotonic", lambda: clock[0])
    g = _game(True)
    assert g.live.set_mic(True)
    g.live.mic._latest = micmod.Reading(peak_hz=200.0, drive_a=0.5, drive_b=0.5, t=1.0, bins=(0.0,) * 24)
    g.live.tick()
    br = g.flies[0].brain
    a, b = micmod.jo_rows(br)
    rows = np.concatenate([a, b]).astype(int)
    assert np.allclose(np.asarray(br.inject)[rows], 0.5)
    clock[0] += 2.0                                  # no new chunk for 2 s: unplugged, or the capture thread failed
    g.live.tick()
    assert np.allclose(np.asarray(br.inject)[rows], 0.0), "the last sound kept driving JO-A/B after the microphone went quiet"
    assert g.live.mic_on, "the switch and its red pill stay as the player left them"


# --- --record-replay refuses the audio block (its current is not a replay event: the replay did not reproduce the spikes) --------
def test_record_replay_refuses_a_synthetic_hum(tmp_path, capsys):
    from kickthefly.lab import protocol as P

    f = tmp_path / "hum.yaml"
    f.write_text("name: hum\nseeds: [1]\nduration_s: 1\naudio: {hz: 200, seconds: 0.5}\n")
    assert P.record_replay(f, tmp_path / "x.ktfreplay") == 2
    assert "audio" in capsys.readouterr().out
    assert not (tmp_path / "x.ktfreplay").exists()


# --- the predator assay: a giant-fiber crossing in the capture frame itself is not an escape --------------------------------------
class _Brain:
    """Stands in for a brain whose DNp01 crosses the escape threshold at one chosen frame (counted in level() calls)."""

    def __init__(self, cross_at_call):
        self.calls, self.cross = 0, cross_at_call

    def poke(self, *a, **k):
        pass

    def _step(self):
        pass

    def level(self, group):
        self.calls += 1
        return 9.0 if self.calls >= self.cross else 1.0


def _trace(n=20, capture=19, strike=17):
    from kickthefly.game import predators as pr

    head = np.array([0.0, 0.39, 0.0])
    frames = [[(("frog", "body"), head + (0, 0, 2.0 - 0.09 * i), 0.45)] for i in range(n)]
    return dict(kind="frog", seed=0, head=head, frames=frames, capture_frame=capture, aim_frame=strike - 2, strike_frame=strike, dt=pr.DT)


def test_a_crossing_in_the_capture_frame_is_a_capture_not_an_escape():
    from kickthefly.lab import predators as lp

    tr = _trace()
    rates = lp.loom_rates(tr)
    from kickthefly.game import kick_the_fly as k

    first = next(i for i, r in enumerate(rates) if r > k.LOOM_MIN)
    start = max(0, first - lp.WARM_FRAMES)
    at_capture = tr["capture_frame"] - start + 1                        # the level() call made in the capture frame
    r = lp.escape_trial(_Brain(at_capture), tr)
    assert not r["escaped"], "DNp01 crossed in the very frame the fly was caught: the game holds it; that is a capture (lead 0 s)"
    r = lp.escape_trial(_Brain(at_capture - 1), tr)                     # one frame earlier: a real escape with a lead
    assert r["escaped"] and r["lead_s"] > 0


def test_a_run_without_a_capture_still_scores_its_last_frame():
    from kickthefly.lab import predators as lp

    tr = _trace(capture=None)
    tr["capture_frame"] = None
    r = lp.escape_trial(_Brain(10 ** 9), tr)
    assert not r["escaped"] and not r["captured"]


def test_the_assay_summaries_carry_their_pre_registered_verdict():
    from kickthefly.lab import audio
    from kickthefly.lab import predators as lp

    cell = lambda r: dict(calm_hz=1.0, hz=r, ratio=r)  # noqa: E731
    fly = dict(conditions={c: dict(**{k: cell(1.5) for k in audio.READOUTS + ("jo_a", "jo_b")}, analysis=dict(peak_hz=200.0))
                           for c in ("pulses_200_ipi35", "silence")})
    s = audio.summarize([fly] * 10)
    assert s["verdict"] is not None and "H1" in s["verdict"] and s["criteria"] == list(audio.CRITERIA)
    t = dict(escaped=False, captured=True, latency_s=None, lead_s=None, noticed_before_strike=False, max_loom_before_strike=0.1)
    s = lp.summarize([dict(seed=i, kinds=["mantis"], trials={"mantis": [t]}) for i in range(10)])
    assert s["verdict"]["P1"] and s["verdict"]["P2"] and s["criteria"] == list(lp.CRITERIA)


# --- the hum demo protocol runs to the end and exports (it crashed in export: its summary had no per_fly) -------------------------
@pytest.mark.parametrize("conditions", [None, ["silence", "steady_200"]])
def test_the_hum_demo_protocol_exports_its_per_fly_scores(synthetic_pack, tmp_path, conditions):
    from kickthefly.lab import protocol as P

    opts = {"seconds": 0.3}
    if conditions:
        opts["conditions"] = conditions
    p = P.check({"name": "hum", "seeds": [1000, 1001], "assay": "hum_demo", "assay_options": opts})
    folder = P.run(p, tmp_path, workers=1)
    rows = (folder / "per_fly.csv").read_text().splitlines()
    assert len(rows) == 3, rows


# --- drawing: a zero-length segment (a predator's tongue the frame its strike begins) is a finite matrix ---------------------------
def test_a_zero_length_segment_gives_a_finite_model_matrix():
    import warnings

    from kickthefly.game.render3d import segment

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = segment((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 0.03)
    assert np.isfinite(m).all()


# --- the red pills: inside the free strip (not over the status card, not under the REC badge), the dot drawn, not a glyph ---------
@pytest.mark.parametrize("larger", [False, True])
def test_the_pills_stay_clear_of_the_status_card_and_the_rec_badge(synthetic_pack, larger):
    from kickthefly.game import kick_the_fly as k2
    from kickthefly.game import live_inputs as li

    g = _game(True)
    if larger:
        g.set_setting("access.larger_text", True, save=False)
    live = g.live
    live.mic_on, live.mic._dev = True, object()
    live.chat_factory = _Chat
    assert live.set_stream(True, "averyveryverylongchannelx")           # 25 characters: the longest Twitch allows
    rects, texts = [], []
    real_rect = pygame.draw.rect

    def rec(surf, col, r, *a, **k):
        rects.append(pygame.Rect(r))
        return real_rect(surf, col, r, *a, **k)

    class F:
        def __init__(self, f):
            self.f = f

        def render(self, t, *a):
            texts.append(t)
            return self.f.render(t, *a)

        def __getattr__(self, n):
            return getattr(self.f, n)

    li.pygame.draw.rect = rec
    try:
        live.draw(pygame.Surface((k2.W, k2.H), pygame.SRCALPHA), F(g.f_bold), k2.PLAY_W)
    finally:
        li.pygame.draw.rect = real_rect
    pills = [r for r in rects if r.y < li.PILL_TOP + 80 and r.x < k2.PLAY_W - 340]
    assert len(pills) >= 2
    for r in pills:
        assert r.x >= li.PILL_MIN_X, f"a pill at x={r.x} is drawn over the status card (it ends at {li.PILL_MIN_X - 10})"
        assert r.right <= k2.PLAY_W - 14 and r.y >= li.PILL_TOP
    assert not any("●" in t for t in texts), "the dot is a glyph many fallback fonts don't have (drawn as a box)"


# --- compatibility: E through every arena with a predator holding the fly; save and load in the kitchen ---------------------------
@pytest.mark.parametrize("three_d", [False, True])
def test_e_cycles_every_arena_twice_with_a_predator_and_draws_each(synthetic_pack, three_d):
    from kickthefly.game import kick_the_fly as k2

    g = _game(three_d)
    slot, _ = _catch(g, "frog")
    seen = []
    for _ in range(2 * len(k2.ARENAS)):
        g.do_action("arena", g.clock.now)
        seen.append(k2.ARENAS[g.arena_i])
        t0 = g.clock.now
        for i in range(10):
            g.frame += 1
            g._environment(t0 + (i + 1) / 60) if three_d else g._environment(t0 + (i + 1) / 60, (300, 300))
            g.preds.step(t0 + (i + 1) / 60)
        g.clock.now = t0 + 10 / 60
        if not three_d:
            g.draw(g.clock.now, (10, 10))                  # one clock, as in play (streaks are stamped with it)
    want = set(k2.ARENAS) if three_d else set(k2.ARENAS) - set(k2.THREE_D_ONLY)
    assert set(seen) == want, f"E reached {sorted(set(seen))}"
    if not three_d:
        assert "kitchen" not in seen
    held = g.preds.pin_for(slot)
    assert held is None or slot.fly.grabbed is not None, "a pin without a hold"
    assert slot.fly.grabbed is None or g.preds.held or g.kitchen is not None or slot.fly.wrapped, \
        "the fly is held by nothing after the arenas changed"


def test_save_in_the_kitchen_with_a_trapped_and_a_caught_fly_then_load(synthetic_pack, tmp_path):
    from kickthefly.core import savestate
    from kickthefly.game import kick3d, kick_the_fly as k2

    g = _game(True)
    g.set_setting("brain.arena", "kitchen", save=False)
    slot, _ = _catch(g, "mantis")
    g.kitchen.trapped[id(slot)] = 1.0
    p = tmp_path / "k.ktfsave"
    savestate.save_game(g, p)
    g.set_setting("brain.arena", "field", save=False)
    assert slot.fly.grabbed is None or g.preds.held, "leaving the kitchen let go of the trapped fly"
    savestate.load_game(g, p)
    assert k2.ARENAS[g.arena_i] == "kitchen" and g.kitchen is not None
    g.preds.step(g.clock.now)
    for s in g.flies:
        held_by_something = s in g.preds.held or id(s) in g.kitchen.trapped or s.fly.wrapped
        assert s.fly.grabbed is None or held_by_something, "a loaded fly is held by nothing"
    assert g.kitchen.trapped == {} or all(k in {id(s) for s in g.flies} for k in g.kitchen.trapped)


def test_the_suite_itself_cannot_open_a_network_connection():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="never opens a network connection"):
            s.connect(("127.0.0.1", 9))
    finally:
        s.close()


# --- follow-ups the user approved: no chat votes in Lab mode; the live inputs are in a recording's metadata --------------------------
def test_streamer_mode_is_refused_in_lab_mode_and_stops_when_lab_mode_starts(synthetic_pack):
    from kickthefly.game import live_inputs as li

    g = _game(True)
    live = g.live
    live.chat_factory = _Chat
    g.set_setting("brain.mode", "lab", save=False)
    assert live.set_stream(True, "somechannel") is False and live.stream_error == li.LAB_LOCK and live.chat is None
    g.set_setting("brain.mode", "play", save=False)
    assert live.set_stream(True, "somechannel")
    g.set_setting("brain.mode", "lab", save=False)                    # switching to Lab while streaming
    live.tick()
    assert not live.stream_on and live.chat is None and live.stream_error == li.LAB_LOCK


def test_a_recording_says_whether_the_microphone_was_on(synthetic_pack):
    from kickthefly.lab import recorder

    g = _game(True)
    assert recorder.metadata(g.brain, g)["live_inputs"]["microphone_on"] is False
    g.live.mic_on = True
    m = recorder.metadata(g.brain, g)["live_inputs"]
    assert m["microphone_on"] is True and m["mic_sensitivity"] == 1.0


def test_the_neuprint_download_is_opt_in_and_asked_once(synthetic_pack, monkeypatch):
    from kickthefly.core import config
    from kickthefly.sim import morphology

    monkeypatch.delenv("KICK_THE_FLY_OFFLINE", raising=False)
    monkeypatch.setenv("KTF_NO_NETWORK", "")
    assert config.Config(None)["brain.neuron_shapes"] is False, "off by default"
    morphology.set_opt_in(False)
    assert not morphology.network_allowed(), "no download without the player's yes"
    morphology.set_opt_in(True)
    try:
        assert morphology.network_allowed()
    finally:
        morphology.set_opt_in(False)
    g = _game(True)
    g.cfg.first_run.update(tutorial_done=True, whatsnew_3_0_seen=True, loadout_notice=False, neuron_shapes_asked=False)
    g.show_first_run_notices()
    assert g.menu.screen == "neuron_shapes_ask" and g.cfg.first_run["neuron_shapes_asked"]
    g.menu.draw(pygame.Surface((1280, 760)), (0, 0), 1.0)
    assert getattr(g.menu, "_page_error", None) is None
    g.menu.back()
    g.show_first_run_notices()
    assert g.menu.screen != "neuron_shapes_ask", "asked once"


# --- memory: a game that is gone is freed (the microphone's atexit hook kept every game, and its brains, alive until exit) --------
def test_a_finished_game_is_freed(synthetic_pack):
    import weakref

    import test_extras3 as t3

    g = _game(True)
    ref, brain_ref = weakref.ref(g), weakref.ref(g.flies[0].brain)
    t3._GAMES.remove(g)
    g.view_stop = True
    for slot in g.flies:
        slot.brain.stop()
    for th in threading.enumerate():
        if th.name == "brain-view":
            th.join(timeout=2.0)
    del g, slot
    gc.collect()
    assert ref() is None and brain_ref() is None, "something still holds the game (and its brains) after it is gone"
