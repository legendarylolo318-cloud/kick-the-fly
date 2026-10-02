"""Where the game may use the network, and where it may not (3.0 day 3). No telemetry, ever: nothing here sends anything anywhere.

Streamer mode (core/streamer.py) is OFF at every launch and shows every connection it makes on screen. The other network use,
older than this module, is the brain view's download of ten neurons' EM skeletons from neuPrint (sim/morphology.py), once, into a
cache; KICK_THE_FLY_OFFLINE=1 turns it off and it also obeys this switch (3.0 day 3 review). Building the brain pack downloads the
dataset, only when you run the build. This module is the one switch the in-game network features go through:

  disable(reason)   no network feature may open a connection from now on (headless runs and --validate call this first)
  allowed()         (True, "") or (False, why not); also False when KTF_NO_NETWORK is set (the test suite sets it)
  require(what)     raise NetworkBlocked unless allowed(): the first line of anything that would connect
  register / unregister / connections()   the live list the UI shows: what is connected, to where, since when

Nothing is persisted: a connection is never reopened by itself at the next launch.
"""
from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_disabled: str | None = None
_open: dict[int, dict] = {}
_next_id = 0


class NetworkBlocked(RuntimeError):
    """A network feature tried to connect where connecting is not allowed."""


def disable(reason: str) -> None:
    global _disabled
    _disabled = reason


def enable() -> None:
    global _disabled
    _disabled = None


def allowed() -> tuple[bool, str]:
    if _disabled:
        return False, f"the network is off here ({_disabled})"
    if os.environ.get("KTF_NO_NETWORK"):
        return False, "the network is off here (KTF_NO_NETWORK is set; the test suite does this)"
    return True, ""


def require(what: str) -> None:
    ok, why = allowed()
    if not ok:
        raise NetworkBlocked(f"{what} was not started: {why}")


def register(what: str, host: str, port: int, note: str = "") -> int:
    global _next_id
    with _lock:
        _next_id += 1
        _open[_next_id] = dict(what=what, host=host, port=port, note=note, since=time.time())
        return _next_id


def unregister(cid: int) -> None:
    with _lock:
        _open.pop(cid, None)


def connections() -> list[dict]:
    """Every connection a feature has open right now (copies)."""
    with _lock:
        return [dict(v) for v in _open.values()]
