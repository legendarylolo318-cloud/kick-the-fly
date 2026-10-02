"""3.0 day 3, Streamer mode: IRC parsing, the vote rules (core/streamer.py), the network guard (core/netguard.py), and the
connection against an in-memory fake server. No real network is ever used."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from kickthefly.core import netguard, streamer as st

OPTS = {"tool": {"frog": "FROG", "swatter": "SWAT"}, "arena": {"kitchen": "kitchen", "room": "room"}, "surgery": {"silence_x": "Silence X"}}


def board(allow=("tool",), window=20.0, cool=30.0, min_votes=2):
    t = [0.0]
    b = st.VoteBoard(st.Rules(window, cool, min_votes, frozenset(allow)), OPTS, clock=lambda: t[0])
    return b, t


# --- parsing ------------------------------------------------------------------------------------------------------------------------
def test_privmsg_lines_with_and_without_tags_are_read():
    plain = ":alice!alice@alice.tmi.twitch.tv PRIVMSG #chan :!tool frog\r\n"
    tagged = "@badge-info=;color=#FF0000;display-name=Alice :alice!alice@alice.tmi.twitch.tv PRIVMSG #chan :!tool frog"
    for line in (plain, tagged):
        assert st.parse_line(line) == ("msg", "alice", "chan", "!tool frog")
    assert st.parse_line("PING :tmi.twitch.tv\r\n") == ("ping", "tmi.twitch.tv")
    assert st.parse_line(":tmi.twitch.tv 001 justinfan1 :Welcome, GLHF!")[:2] == ("numeric", "001")
    assert st.parse_line(":tmi.twitch.tv NOTICE * :Login authentication failed")[0] == "notice"
    assert st.parse_line("") is None and st.parse_line("@only-tags") is None and st.parse_line(":x JOIN #c") is None


def test_commands_are_read_strictly():
    assert st.parse_command("!tool frog") == ("tool", "frog")
    assert st.parse_command("!TOOL  Frog ") == ("tool", "frog")
    assert st.parse_command("!surgery Silence X") == ("surgery", "silence_x")
    for bad in ("tool frog", "!tool", "!dance frog", "hello !tool frog", "!" + "x" * 300, "!tool\r\nJOIN #evil"[:10]):
        assert st.parse_command(bad) is None or st.parse_command(bad)[0] in st.COMMANDS


@pytest.mark.parametrize("name,ok", [("mychannel", True), ("#MyChannel", True), ("a_b_9", True), ("ab", False), ("x" * 26, False),
                                     ("bad name", False), ("chan\r\nJOIN #evil", False), ("", False), ("#", False), ("a.b", False)])
def test_channel_names_are_validated_so_nothing_can_add_a_second_irc_command(name, ok):
    if ok:
        assert st.clean_channel(name) == name.lstrip("#").lower()
    else:
        with pytest.raises(st.StreamError):
            st.clean_channel(name)


# --- the vote ------------------------------------------------------------------------------------------------------------------------
def test_a_round_opens_on_the_first_vote_counts_one_vote_each_and_picks_the_most():
    b, t = board()
    assert b.state == "idle" and b.tick(0.0) is None
    for i, u in enumerate("abcdef"):
        t[0] = 1.0 + i * 3.0
        assert b.submit(u, "!tool frog" if i < 4 else "!tool swatter", t[0]) == "counted"
    assert b.state == "open" and b.tally()[0] == ("tool", "frog", "FROG", 4)
    t[0] = 25.0
    r = b.tick(t[0])
    assert (r.command, r.slug, r.votes, r.total) == ("tool", "frog", 4, 6)
    assert b.state == "cooldown" and b.tally() == []


def test_a_later_vote_replaces_the_earlier_one_and_a_double_vote_does_not_count_twice():
    b, t = board()
    assert b.submit("a", "!tool frog", 1.0) == "counted"
    assert b.submit("a", "!tool swatter", 5.0) == "counted"
    assert b.tally() == [("tool", "swatter", "SWAT", 1)]
    assert b.submit("A", "!tool frog", 5.5) == "too fast", "names are case-insensitive: the same viewer"


def test_too_few_votes_win_nothing():
    b, t = board(min_votes=3)
    b.submit("a", "!tool frog", 1.0)
    b.submit("b", "!tool frog", 4.0)
    assert b.tick(30.0) is None and b.state == "cooldown"


def test_the_cooldown_ignores_votes_then_ends():
    b, t = board(cool=30.0)
    b.submit("a", "!tool frog", 1.0)
    b.submit("b", "!tool frog", 4.0)
    assert b.tick(25.0) is not None
    assert b.submit("c", "!tool frog", 30.0) == "cooldown" and b.tally() == []
    assert b.submit("c", "!tool frog", 56.0) == "counted" and b.state == "open"


def test_only_allowlisted_commands_and_offered_options_count():
    b, t = board(allow=("tool",))
    assert b.submit("a", "!arena kitchen", 1.0) == "command not allowed"
    assert b.submit("a", "!surgery silence_x", 1.0) == "command not allowed"
    assert b.submit("a", "!tool laser", 1.0) == "not an option"
    assert b.submit("a", "chat is fun", 1.0) == "not a command"
    assert b.state == "idle" and b.ignored == 3, "nothing opened a round"
    b2, _ = board(allow=("tool", "arena", "surgery"))
    assert b2.submit("a", "!arena kitchen", 1.0) == "counted"


def test_a_viewer_cannot_vote_faster_than_the_limit_and_the_channel_is_capped():
    b, t = board()
    assert b.submit("a", "!tool frog", 1.0) == "counted"
    assert b.submit("a", "!tool frog", 1.5) == "too fast"
    assert b.submit("a", "!tool frog", 3.2) == "counted"
    flood = [b.submit(f"user{i}", "!tool frog", 10.0) for i in range(60)]
    assert flood.count("channel rate limit") == 60 - st.MAX_PER_SECOND + 1 - 1 or "channel rate limit" in flood
    assert len(b._votes) <= st.MAX_PER_SECOND + 1


def test_no_names_are_kept_and_the_hashes_vanish_when_the_round_ends():
    b, t = board()
    b.submit("secretviewer", "!tool frog", 1.0)
    blob = repr(vars(b)).lower()
    assert "secretviewer" not in blob
    b.submit("other", "!tool frog", 5.0)
    assert b.tick(30.0) is not None
    assert not b._votes and all("secretviewer" not in repr(k).lower() for k in b._last_seen)


def test_the_tally_is_ordered_by_votes_then_by_who_got_there_first():
    b, t = board()
    b.submit("a", "!tool swatter", 1.0)
    b.submit("b", "!tool frog", 4.0)
    assert [r[1] for r in b.tally()] == ["swatter", "frog"]
    b.submit("c", "!tool frog", 7.0)
    assert [r[1] for r in b.tally()] == ["frog", "swatter"]


# --- the network guard ------------------------------------------------------------------------------------------------------------------
def test_the_guard_blocks_when_disabled_or_in_the_test_environment(monkeypatch):
    monkeypatch.delenv("KTF_NO_NETWORK", raising=False)
    netguard.enable()
    assert netguard.allowed() == (True, "")
    netguard.disable("a headless run")
    ok, why = netguard.allowed()
    assert not ok and "headless" in why
    with pytest.raises(netguard.NetworkBlocked):
        netguard.require("Streamer mode")
    netguard.enable()
    monkeypatch.setenv("KTF_NO_NETWORK", "1")
    assert not netguard.allowed()[0]


def test_the_test_suite_itself_runs_with_the_network_off():
    import os

    assert os.environ.get("KTF_NO_NETWORK") == "1"
    with pytest.raises(netguard.NetworkBlocked):
        st.TwitchChat("somechannel").start()
    assert netguard.connections() == []


# --- the connection against a fake server (an in-memory socket pair; no network) ---------------------------------------------------------
class FakeServer:
    def __init__(self):
        self.client, self.server = socket.socketpair()
        self.received = b""
        self.thread = None

    def run(self, script):
        def go():
            self.server.settimeout(3.0)
            script(self)
        self.thread = threading.Thread(target=go, daemon=True)
        self.thread.start()

    def read_until(self, token: bytes, timeout=3.0):
        end = time.time() + timeout
        while token not in self.received and time.time() < end:
            try:
                self.received += self.server.recv(4096)
            except socket.timeout:
                pass

    def say(self, text: str):
        self.server.sendall(text.encode() + b"\r\n")


def _wait(pred, timeout=4.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_it_logs_in_anonymously_joins_one_channel_reads_commands_and_answers_ping():
    fs = FakeServer()

    def script(s):
        s.read_until(b"USER")
        s.say(":tmi.twitch.tv 001 justinfan12345 :Welcome, GLHF!")
        s.read_until(b"JOIN #mychannel")
        s.say("PING :tmi.twitch.tv")
        s.say(":alice!alice@alice.tmi.twitch.tv PRIVMSG #mychannel :!tool frog")
        s.say(":bob!bob@bob.tmi.twitch.tv PRIVMSG #mychannel :just chatting, no command here")
        s.say(":carol!carol@carol.tmi.twitch.tv PRIVMSG #otherchannel :!tool frog")
        s.say(":dave!dave@dave.tmi.twitch.tv PRIVMSG #mychannel :!arena kitchen")
        s.read_until(b"PONG")

    fs.run(script)
    chat = st.TwitchChat("MyChannel", sock_factory=lambda: fs.client)
    chat.start()
    assert _wait(lambda: chat.commands_read >= 2)
    assert chat.state == "live" and any(c["host"] == st.HOST for c in netguard.connections()), "the connection is listed while open"
    got = chat.drain()
    assert got == [("alice", "!tool frog"), ("dave", "!arena kitchen")], "only ! lines of this channel; chit-chat is dropped on arrival"
    assert chat.drain() == []
    chat.stop()
    fs.thread.join(3.0)
    sent = fs.received.decode()
    assert sent.startswith("NICK justinfan") and "PASS" not in sent and "oauth" not in sent.lower(), "anonymous: no password, no token"
    assert "JOIN #mychannel" in sent and "PONG :tmi.twitch.tv" in sent
    assert "PRIVMSG" not in sent, "read-only: it never says anything"
    assert chat.state == "off" and netguard.connections() == []


def test_a_refused_login_is_reported_and_closes_the_connection():
    fs = FakeServer()
    fs.run(lambda s: (s.read_until(b"USER"), s.say(":tmi.twitch.tv NOTICE * :Login authentication failed")))
    chat = st.TwitchChat("mychannel", sock_factory=lambda: fs.client)
    chat.start()
    assert _wait(lambda: chat.state == "error")
    assert "refused the anonymous login" in chat.error
    assert netguard.connections() == []


def test_a_server_that_hangs_up_is_an_error_not_a_crash():
    fs = FakeServer()
    fs.run(lambda s: (s.read_until(b"USER"), s.server.close()))
    chat = st.TwitchChat("mychannel", sock_factory=lambda: fs.client)
    chat.start()
    assert _wait(lambda: chat.state == "error") and "closed" in chat.error


def test_the_chat_module_has_no_way_to_send_a_message():
    import inspect

    src = inspect.getsource(st)
    assert "PRIVMSG #" not in src.replace('"PRIVMSG"', "") and "PASS " not in src.replace("no PASS", "").replace("PASS,", "")


# --- the self-test (optional checks that never open the microphone or the network) ---------------------------------------------------------
def test_selftest_checks_for_the_microphone_and_the_network_are_optional_and_open_nothing(monkeypatch):
    from kickthefly.core import mic, selftest

    opened = []
    monkeypatch.setattr(mic.Mic, "start", lambda self: opened.append("mic"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: opened.append("net"))
    monkeypatch.setattr(mic, "devices", lambda: [])
    m, n, d = selftest.check_microphone(), selftest.check_network(), selftest.check_day3()
    assert m.status == n.status == d.status == selftest.PASS, "an optional feature that isn't there is not a warning"
    assert "optional" in m.name.lower() and "optional" in n.name.lower()
    assert "did not connect" in n.detail or "off here" in n.detail
    assert opened == []
    monkeypatch.setattr(mic, "devices", lambda: ["Built-in Audio"])
    assert "did not open it" in selftest.check_microphone().detail and opened == []
