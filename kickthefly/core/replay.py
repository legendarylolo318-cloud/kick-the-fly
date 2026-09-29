"""Deterministic replay files (.ktfreplay).

A replay holds what it takes to run a brain again step for step: the seed, the Lab parameters and surgery it was
built with, the brain pack's SHA-256, and every input it received, at the step it arrived:
  sense     the neurons a stimulus key stands for (recorded the first time a key is used, and when it changes)
  poke      Brain.poke(region, side, strength, recruit): a game-style stimulus (it draws from the brain's own RNG,
            which the replay reproduces by poking in the same order)
  drive     changes to Brain.drive_cur (a held current: assays, protocol "drive" stimuli)
  override  changes to Brain.override (brain surgery's per-neuron current)
  tool      (2.12) which tool the run used, at the step it was taken in hand: informational, the tool's effect is
            already in the pokes, drives and overrides above, so playing it back does nothing and an older version
            that doesn't know the type skips it
It also holds the SHA-256 of every step's spikes, so a replay can say whether it reproduced the recording.

What records one (2.10): a headless protocol run, `--headless --protocol FILE --record-replay OUT.ktfreplay` (the
protocol's first seed, without surgery unless the protocol has some). The windowed game doesn't record or play
replays yet. Play one back with `--headless --replay FILE --out DIR [--backend NAME]`.

Reproducibility: NumPy (cpu), Numba and torch-cpu are bit-exact with each other (tests/test_backends.py), so a replay
on any of them gives the recording's spikes exactly, and a difference there is an error. GPU backends are held only to
a statistical tolerance, so on them a replay is reported, not judged.

The brain pack must be the same file, byte for byte (its SHA-256); a replay recorded with another pack is refused.
"""
from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kickthefly.core.version import __version__

FORMAT = "kick-the-fly-replay"
FORMAT_VERSION = 1
SUFFIX = ".ktfreplay"
BIT_EXACT_BACKENDS = ("cpu", "numba", "torch-cpu")


class ReplayError(Exception):
    """Raised when a replay file cannot be loaded or is incompatible."""


_pack_hash_cache: dict[tuple, str] = {}
_pack_hash_lock = threading.Lock()


def pack_sha256(path: Path | None = None) -> str | None:
    """SHA-256 of the brain pack file (cached by path, size and modification time)."""
    from kickthefly.sim import brainpack

    p = Path(path) if path else brainpack.find()
    if p is None or not p.exists():
        return None
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    with _pack_hash_lock:
        if key not in _pack_hash_cache:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8 << 20), b""):
                    h.update(chunk)
            _pack_hash_cache[key] = h.hexdigest()
        return _pack_hash_cache[key]


def pack_signature(game_or_brain) -> dict:
    """The brain pack a brain runs on: neuron and synapse counts, the mushroom body signature, and the pack's
    SHA-256 (the counts alone don't tell two packs apart)."""
    br = getattr(game_or_brain, "brain", game_or_brain)
    if hasattr(br, "flies") and br.flies:
        br = br.flies[0].brain
    W = getattr(getattr(br, "sim", None), "W_csr", None)
    mem = getattr(br, "memory", None)
    return dict(
        n_neurons=int(getattr(br, "n", 0)),
        synapses=int(getattr(W, "nnz", 0)) if W is not None else 0,
        memory_signature=[float(x) for x in mem.signature] if mem is not None and getattr(mem, "signature", None) is not None else None,
        pack_sha256=pack_sha256(),
    )


def check_compatibility(meta: dict, current_sig: dict) -> str | None:
    """None if this replay can run here, else a user-facing reason."""
    v = meta.get("format_version")
    if not isinstance(v, int) or v > FORMAT_VERSION:
        return f"Replay was recorded with a newer version ({meta.get('app_version', '?')}); update the game to play it."
    if v < 1:
        return "Unsupported replay file format."
    sig = meta.get("signature", {})
    if current_sig.get("n_neurons") and sig.get("n_neurons") != current_sig["n_neurons"]:
        return f"Recorded with a different brain pack ({sig.get('n_neurons')} neurons vs current {current_sig['n_neurons']})."
    if current_sig.get("synapses") and sig.get("synapses") != current_sig["synapses"]:
        return f"Recorded with a different brain pack ({sig.get('synapses')} synapses vs current {current_sig['synapses']})."
    mine, theirs = current_sig.get("pack_sha256"), sig.get("pack_sha256")
    if mine and not theirs:
        return "Recorded without a brain pack checksum, so it can't be matched to this brain pack."
    if mine and theirs != mine:
        return f"Recorded with a different brain pack (SHA-256 {theirs[:12]}... vs current {mine[:12]}...)."
    if current_sig.get("memory_signature") and sig.get("memory_signature"):
        if not np.allclose(sig["memory_signature"], current_sig["memory_signature"]):
            return "Recorded with a different mushroom body wiring."
    return None


def _spike_bytes(spikes: np.ndarray) -> bytes:
    return np.packbits(np.asarray(spikes, dtype=bool)).tobytes()


@dataclass
class ReplayRecorder:
    seed: int
    backend: str
    dtype: str
    signature: dict
    arena: str
    settings: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    initial_step: int = 0
    total_steps: int = 0
    spike_sha256: str | None = None

    def record(self, step: int, event_type: str, **payload: Any) -> None:
        """Record an input event at a specific simulation step."""
        ev = {"step": int(step), "type": str(event_type), **payload}
        self.events.append(ev)
        if step > self.total_steps:
            self.total_steps = step

    def attach(self, br) -> "ReplayRecorder":
        """Record every input this brain gets from now on, and the spikes of every step it takes."""
        self._br = br
        self._step0 = br.steps
        self._last = {"drive": br.drive_cur.copy(), "override": br.override.copy()}
        self._sense: dict[tuple, np.ndarray] = {}
        self._hash = hashlib.sha256()
        self._in_poke = False
        orig_poke, orig_step = br.poke, br._step

        def poke(region, side, strength, recruit=None):
            if self._in_poke:                        # a poke's own follow-up (more pain neurons): replayed by the poke
                return orig_poke(region, side, strength, recruit=recruit)
            s = br.steps - self._step0
            key = (region, side)
            rows = br.sense.get(key)
            if rows is not None and (key not in self._sense or not np.array_equal(self._sense[key], rows)):
                self.record(s, "sense", region=region, side=side, rows=[int(x) for x in rows])
                self._sense[key] = np.array(rows, copy=True)
            self.record(s, "poke", region=region, side=side, strength=float(strength),
                        recruit=None if recruit is None else float(recruit))
            self._in_poke = True
            try:
                return orig_poke(region, side, strength, recruit=recruit)
            finally:
                self._in_poke = False

        def step():
            s = br.steps - self._step0
            for name, cur in (("drive", br.drive_cur), ("override", br.override)):
                prev = self._last[name]
                idx = np.flatnonzero(cur != prev)
                if len(idx):
                    self.record(s, name, rows=idx.tolist(), values=[float(v) for v in cur[idx]])
                    prev[idx] = cur[idx]
            out = orig_step()
            self._hash.update(_spike_bytes(br.sim.spikes))
            self.total_steps = s + 1
            self.spike_sha256 = self._hash.hexdigest()
            return out

        br.poke, br._step = poke, step
        self._restore = (orig_poke, orig_step)
        return self

    def detach(self) -> None:
        br = getattr(self, "_br", None)
        if br is not None:
            br.poke, br._step = self._restore
            self._br = None

    def save(self, path: Path) -> Path:
        """Save the recording to a .ktfreplay (zip) file."""
        meta = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "app_version": __version__,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.system(),
            "seed": self.seed,
            "backend": self.backend,
            "dtype": self.dtype,
            "signature": self.signature,
            "arena": self.arena,
            "settings": self.settings,
            "total_steps": self.total_steps,
            "spike_sha256": self.spike_sha256,
            "events": self.events,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("replay.json", json.dumps(meta, indent=1))
        tmp.replace(path)
        return path


class ReplayPlayer:
    def __init__(self, meta: dict):
        self.meta = meta
        self.seed: int = meta["seed"]
        self.backend: str = meta["backend"]
        self.dtype: str = meta.get("dtype", "float32")
        self.signature: dict = meta.get("signature", {})
        self.arena: str = meta.get("arena", "room")
        self.settings: dict[str, Any] = meta.get("settings", {})
        self.total_steps: int = meta.get("total_steps", 0)
        self.spike_sha256: str | None = meta.get("spike_sha256")
        self.events_by_step: dict[int, list[dict[str, Any]]] = {}
        for ev in meta.get("events", []):
            self.events_by_step.setdefault(int(ev["step"]), []).append(ev)

    @classmethod
    def load(cls, path: Path, expected_sig: dict | None = None) -> "ReplayPlayer":
        """Load a .ktfreplay file, checking it against expected_sig (pack_signature of the brain it will run on)."""
        try:
            with zipfile.ZipFile(path) as zf:
                data = json.loads(zf.read("replay.json"))
        except Exception as e:
            raise ReplayError(f"Cannot read replay file ({type(e).__name__}): {e}") from None
        if data.get("format") != FORMAT:
            raise ReplayError("File is not a valid Kick the Fly replay.")
        if expected_sig is not None:
            why = check_compatibility(data, expected_sig)
            if why:
                raise ReplayError(why)
        return cls(data)

    def tools(self) -> list[str]:
        """The tools the recorded run used, in order (from its "tool" events)."""
        return [ev["name"] for ev in self.meta.get("events", []) if ev.get("type") == "tool"]

    def events_at(self, step: int) -> list[dict[str, Any]]:
        """Return all events recorded at a given simulation step."""
        return self.events_by_step.get(step, [])

    def is_finished(self, step: int) -> bool:
        """True if the replay reached its recorded end."""
        return step >= self.total_steps

    def new_brain(self, backend: str | None = None):
        """A brain built the way the recorded one was: seed, Lab parameters and surgery."""
        from kickthefly.core import simcore
        from kickthefly.lab import assays

        br = simcore.new_brain(seed=self.seed, params=self.settings.get("params") or None, backend=backend)
        assays.apply_surgery(br, self.settings.get("surgery") or None)
        return br

    def apply(self, br, step: int) -> None:
        for ev in self.events_at(step):
            t = ev["type"]
            if t == "sense":
                br.sense[(ev["region"], ev["side"])] = np.asarray(ev["rows"], dtype=np.int64)
            elif t == "poke":
                br.poke(ev["region"], ev["side"], ev["strength"], recruit=ev.get("recruit"))
            elif t in ("drive", "override"):
                arr = br.drive_cur if t == "drive" else br.override
                arr[np.asarray(ev["rows"], dtype=np.int64)] = np.asarray(ev["values"], dtype=arr.dtype)
                if t == "drive":
                    br.driving = bool(np.any(br.drive_cur))
                else:
                    br.surgery = bool(np.any(br.override))

    def play(self, br, on_step=None) -> str:
        """Run the recording on this brain. Returns the SHA-256 of its spikes, comparable with spike_sha256."""
        h = hashlib.sha256()
        for step in range(self.total_steps):
            self.apply(br, step)
            br._step()
            h.update(_spike_bytes(br.sim.spikes))
            if on_step is not None:
                on_step(step, br.sim.spikes)
        return h.hexdigest()
