"""Streamer mode (3.0 day 3): viewers of a Twitch channel vote on what the game does next. OPT-IN, off at every launch.

What it is: a read-only, anonymous reader of one Twitch channel's chat (the IRC interface) plus a vote counter. It reads chat
messages that start with ! and counts them; it never sends a chat message, never logs in, never asks for or stores a token or
a key, and never keeps who said what. YouTube is out of scope (it cannot be done without storing a key).

What is what (every rule here is a GAME RULE; nothing touches the connectome):
  commands   !tool NAME, !surgery NAME, !arena NAME. The streamer chooses which of the three commands count (Settings >
             Streamer: tool is on, arena and surgery are off until switched on) and the names a viewer can pick are only what
             the game offers now: the tools on the hotbar, the surgery list, the arenas this game can show.
  rounds     the first valid vote opens a round of WINDOW_S seconds; the option with the most votes wins if it has at least
             MIN_VOTES; then a cooldown of COOLDOWN_S seconds during which votes are ignored. One vote per viewer a round (a
             later vote replaces the earlier one).
  limits     a viewer's messages closer than USER_MIN_INTERVAL_S are ignored; the whole channel is capped at MAX_PER_SECOND
             messages a second; a message longer than MAX_MESSAGE is ignored; only whole-line commands in the allowlist are read.
  privacy    only messages that begin with ! are kept, for one frame, to be counted. A viewer's name is hashed with a random
             salt made at start-up and only the hashes of this round are held (to stop double votes), in memory, and dropped when the
             round ends (those of the last USER_MIN_INTERVAL_S stay for the flood limit); the screen shows counts, never names.
             Nothing is written to disk.

The connection (TwitchChat): TLS to irc.chat.twitch.tv port 6697, nickname justinfan<random digits>, no password, JOIN one
channel, answer PING. Twitch documents the IRC interface at https://dev.twitch.tv/docs/chat/irc/ ; the anonymous "justinfan"
read-only login is described in Twitch's developer forum threads (discuss.dev.twitch.com "Anonymous connection using justinfan"
and "Possible to connect to Twitch IRC anonymously?") but is NOT in the current official documentation, so Twitch may change or
withdraw it; if the server refuses, the game says so and nothing else happens. The connection is shown on screen the whole time it
is open (core/netguard.py) and is never opened during tests, --validate or any headless run.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import socket
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from kickthefly.core import netguard

HOST, PORT = "irc.chat.twitch.tv", 6697
COMMANDS = ("tool", "surgery", "arena")
WINDOW_S, COOLDOWN_S, MIN_VOTES = 20.0, 30.0, 2
USER_MIN_INTERVAL_S = 2.0
MAX_PER_SECOND = 30
MAX_MESSAGE = 200
CONNECT_TIMEOUT_S = 10.0
RECV_POLL_S = 0.25                 # how often the reader checks whether it was switched off
CHANNEL_RE = re.compile(r"^[a-z0-9_]{3,25}$")
SLUG_RE = re.compile(r"[^a-z0-9]+")


class StreamError(RuntimeError):
    """Streamer mode can't do that; the message says why in plain words."""


def clean_channel(name: str) -> str:
    """A Twitch channel login: 3-25 letters, digits or underscores (a leading # is allowed and dropped). Anything else is refused,
    so nothing a user types can add a second IRC command."""
    c = str(name or "").strip().lower().lstrip("#")
    if not CHANNEL_RE.match(c):
        raise StreamError("a channel name is 3 to 25 letters, digits or underscores (for example  mychannel)")
    return c


def slug(label: str) -> str:
    return SLUG_RE.sub("_", str(label).lower()).strip("_")


# --- reading a line of IRC ---------------------------------------------------------------------------------------------------------
def parse_line(line: str):
    """('ping', payload) | ('msg', user, channel, text) | ('numeric', code, rest) | ('notice', text) | None."""
    line = line.rstrip("\r\n")
    if not line:
        return None
    if line.startswith("@"):                                 # message tags: drop them
        sp = line.find(" ")
        if sp < 0:
            return None
        line = line[sp + 1:]
    if line.startswith("PING"):
        return ("ping", line[5:].lstrip(":") if len(line) > 5 else "")
    parts = line.split(" ", 3)
    if len(parts) >= 4 and parts[1] == "PRIVMSG" and parts[0].startswith(":"):
        user = parts[0][1:].split("!", 1)[0]
        text = parts[3][1:] if parts[3].startswith(":") else parts[3]
        return ("msg", user, parts[2].lstrip("#"), text)
    if len(parts) >= 2 and parts[1].isdigit():
        return ("numeric", parts[1], line)
    if len(parts) >= 2 and parts[1] == "NOTICE":
        return ("notice", line)
    return None


def parse_command(text: str):
    """('tool', 'frog') for '!tool frog'; None for anything that is not a one-word-argument command."""
    t = text.strip()
    if not t.startswith("!") or len(t) > MAX_MESSAGE:
        return None
    bits = t[1:].split()
    if len(bits) < 2 or bits[0].lower() not in COMMANDS:
        return None
    return bits[0].lower(), slug(" ".join(bits[1:]))


# --- the vote ------------------------------------------------------------------------------------------------------------------------
@dataclass
class Rules:
    window_s: float = WINDOW_S
    cooldown_s: float = COOLDOWN_S
    min_votes: int = MIN_VOTES
    allow: frozenset = frozenset({"tool"})


@dataclass
class Result:
    command: str
    slug: str
    label: str
    votes: int
    total: int


class VoteBoard:
    """Counts votes and runs the rounds. Pure logic with an injected clock, so every rule is tested without a network."""

    def __init__(self, rules: Rules | None = None, options: dict | None = None, clock=time.monotonic):
        self.rules = rules or Rules()
        self.options: dict[str, dict[str, str]] = options or {}        # command -> {slug: label}
        self.clock = clock
        self._salt = os.urandom(16)
        self.state = "idle"                                            # idle | open | cooldown
        self._round_end = 0.0
        self._cool_end = 0.0
        self._votes: dict[bytes, tuple[str, str, float]] = {}          # hashed viewer -> (command, slug, when)
        self._last_seen: dict[bytes, float] = {}
        self._sec = 0
        self._sec_n = 0
        self.ignored = 0

    def set_options(self, options: dict) -> None:
        self.options = options

    def set_rules(self, rules: Rules) -> None:
        """New rules from the streamer's settings. Votes already cast for a command that is no longer allowed are dropped, so a
        command switched off while a round is open can't win it (3.0 day 3 review)."""
        self.rules = rules
        for h in [h for h, (c, _, _) in self._votes.items() if c not in rules.allow]:
            del self._votes[h]

    def _h(self, user: str) -> bytes:
        return hashlib.sha256(self._salt + user.lower().encode("utf-8", "ignore")).digest()[:12]

    def submit(self, user: str, text: str, now: float | None = None) -> str:
        """Count one chat message. Returns why it did or didn't count (for the tests and the on-screen counters)."""
        now = self.clock() if now is None else now
        sec = int(now)
        if sec != self._sec:
            self._sec, self._sec_n = sec, 0
        self._sec_n += 1
        if self._sec_n > MAX_PER_SECOND:
            self.ignored += 1
            return "channel rate limit"
        cmd = parse_command(text)
        if cmd is None:
            return "not a command"
        command, arg = cmd
        if command not in self.rules.allow:
            self.ignored += 1
            return "command not allowed"
        if arg not in self.options.get(command, {}):
            self.ignored += 1
            return "not an option"
        if self.state == "cooldown":
            if now < self._cool_end:
                self.ignored += 1
                return "cooldown"
            self.state = "idle"
        h = self._h(user)
        last = self._last_seen.get(h)
        if last is not None and now - last < USER_MIN_INTERVAL_S:
            self.ignored += 1
            return "too fast"
        self._last_seen[h] = now
        if len(self._last_seen) > 5000:                                # bounded memory
            self._last_seen = {k: v for k, v in self._last_seen.items() if now - v < 60}
        if self.state == "idle":
            self.state, self._round_end = "open", now + self.rules.window_s
            self._votes.clear()
        self._votes[h] = (command, arg, now)
        return "counted"

    def tally(self) -> list[tuple[str, str, str, int]]:
        """(command, slug, label, votes), most votes first. No names."""
        counts: dict[tuple[str, str], int] = {}
        first: dict[tuple[str, str], float] = {}
        for command, arg, when in self._votes.values():
            k = (command, arg)
            counts[k] = counts.get(k, 0) + 1
            first[k] = min(first.get(k, when), when)
        rows = [(c, a, self.options.get(c, {}).get(a, a), n) for (c, a), n in counts.items()]
        return sorted(rows, key=lambda r: (-r[3], first[(r[0], r[1])]))

    def tick(self, now: float | None = None) -> Result | None:
        """Close the round when its window is over. Returns the winner (once) or None."""
        now = self.clock() if now is None else now
        if self.state == "cooldown" and now >= self._cool_end:
            self.state = "idle"
        if self.state != "open" or now < self._round_end:
            return None
        rows = self.tally()
        total = len(self._votes)
        self._votes.clear()
        # 3.0 day 3 review: the docstring promises a round's viewer hashes are dropped when it ends; the flood limit's own book kept
        # them for up to a minute. Only the last USER_MIN_INTERVAL_S, which that limit needs, are kept now.
        self._last_seen = {k: v for k, v in self._last_seen.items() if now - v < USER_MIN_INTERVAL_S}
        self.state, self._cool_end = "cooldown", now + self.rules.cooldown_s
        if rows and rows[0][3] >= self.rules.min_votes:
            c, a, label, n = rows[0]
            return Result(c, a, label, n, total)
        return None

    def remaining(self, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        if self.state == "open":
            return max(0.0, self._round_end - now)
        if self.state == "cooldown":
            return max(0.0, self._cool_end - now)
        return 0.0

    def reset(self) -> None:
        self._votes.clear()
        self._last_seen.clear()
        self.state = "idle"


# --- the connection ---------------------------------------------------------------------------------------------------------------------
class TwitchChat:
    """Reads one channel's chat, anonymously. start() opens the connection (only if netguard allows it); stop() closes it.
    drain() hands over the commands read since the last call: (viewer, text) for messages that start with !, nothing else."""

    def __init__(self, channel: str, host: str = HOST, port: int = PORT, tls: bool = True, sock_factory=None):
        self.channel = clean_channel(channel)
        self.host, self.port, self.tls = host, port, tls
        self._factory = sock_factory
        self.state = "off"                      # off | connecting | live | error
        self.error = ""
        self.nick = f"justinfan{random.randint(10000, 99999)}"
        self._q: deque = deque(maxlen=200)
        self._lock = threading.Lock()
        self._sock = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._conn_id: int | None = None
        self.lines_read = 0
        self.commands_read = 0
        self.since = 0.0

    def describe(self) -> str:
        return f"{self.host}:{self.port} #{self.channel} (read-only, anonymous)"

    def start(self) -> None:
        if self.state in ("connecting", "live"):
            return
        if self._factory is None:
            netguard.require("Streamer mode")
        # 3.0 day 3 review: each run has its own stop flag, socket and on-screen entry. With one shared flag, off-then-on while the
        # first connect was still blocked cleared the flag the old reader was waiting on, so two readers ran and the old one's
        # connection was no longer shown on screen.
        stop = threading.Event()
        self._stop = stop
        self.state, self.error = "connecting", ""
        cid = netguard.register("Twitch chat (read-only, anonymous)", self.host, self.port, f"#{self.channel}")
        self._conn_id = cid
        self._thread = threading.Thread(target=self._run, args=(stop, cid), name="streamer-chat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the reader to hang up and wait briefly for it. Only the reader thread touches its socket (an SSL socket must not
        be used from two threads at once); it notices within RECV_POLL_S, says QUIT, closes, and takes itself off the screen.
        A reader still stuck in connecting stays listed on screen until its connect gives up, which is the truth."""
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3.0)
        self.state, self.error = "off", ""
        self._conn_id = None

    def _finish(self, state: str, error: str) -> None:
        self.state, self.error = state, error

    def _connect(self):
        if self._factory is not None:
            return self._factory()
        raw = socket.create_connection((self.host, self.port), timeout=CONNECT_TIMEOUT_S)
        if not self.tls:
            return raw
        try:
            ctx = ssl.create_default_context()
            return ctx.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()                                         # 3.0 day 3 review: a failed handshake left the TCP connection open
            raise

    @staticmethod
    def _line(text: str) -> bytes:
        """One IRC line. Nothing may add a second command: CR, LF and NUL are refused (3.0 day 3 review: a PING payload with a CR
        in it was echoed into the PONG)."""
        if any(c in text for c in "\r\n\0"):
            raise StreamError("refused to send a line with a line break in it")
        return text.encode("utf-8") + b"\r\n"

    def _run(self, stop: threading.Event, cid: int) -> None:
        sock = None
        try:
            sock = self._connect()
            if stop.is_set():
                return
            self._sock = sock
            sock.settimeout(RECV_POLL_S)
            sock.sendall(self._line(f"NICK {self.nick}"))            # anonymous: no PASS, no token, nothing to store
            sock.sendall(self._line(f"USER {self.nick} 0 * :{self.nick}"))
            buf, joined, t0 = b"", False, time.monotonic()
            while not stop.is_set():
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    if not joined and time.monotonic() - t0 > CONNECT_TIMEOUT_S:
                        raise StreamError("Twitch did not answer (no welcome within 10 seconds)")
                    continue
                if not data:
                    raise StreamError("the connection was closed by the server")
                buf += data
                if len(buf) > 65536:
                    raise StreamError("the server sent a line that is far too long")
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    ev = parse_line(raw.decode("utf-8", "replace"))
                    if ev is None:
                        continue
                    self.lines_read += 1
                    if ev[0] == "ping":
                        payload = re.sub(r"[\r\n\0]", "", ev[1])
                        sock.sendall(self._line(f"PONG :{payload}" if payload else "PONG"))
                    elif ev[0] == "numeric" and ev[1] == "001" and not joined:
                        joined = True
                        sock.sendall(self._line(f"JOIN #{self.channel}"))
                        if not stop.is_set():
                            self.state, self.since = "live", time.time()
                    elif ev[0] == "numeric" and ev[1] in ("464", "465"):
                        raise StreamError("Twitch refused the anonymous login (it may no longer allow it)")
                    elif ev[0] == "notice" and ("authentication failed" in ev[1].lower() or "improperly formatted" in ev[1].lower()):
                        raise StreamError("Twitch refused the anonymous login (it may no longer allow it)")
                    elif ev[0] == "msg" and ev[3].startswith("!") and len(ev[3]) <= MAX_MESSAGE and ev[2].lower() == self.channel:
                        with self._lock:
                            self._q.append((ev[1], ev[3]))
                            self.commands_read += 1
            try:
                sock.sendall(b"QUIT\r\n")
            except Exception:
                pass
        except Exception as e:
            if not stop.is_set():
                self._finish("error", str(e) if isinstance(e, StreamError) else f"couldn't reach Twitch chat: {e}")
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            if self._sock is sock:
                self._sock = None
            netguard.unregister(cid)                                    # off the screen only once the socket is really closed

    def drain(self) -> list[tuple[str, str]]:
        with self._lock:
            out = list(self._q)
            self._q.clear()
        return out
