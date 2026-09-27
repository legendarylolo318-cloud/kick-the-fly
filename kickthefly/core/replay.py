"""Deterministic replay framework (.ktfreplay).

Records and plays back simulation sessions with exact event timestamps and
connectome metadata.

Reproducibility guarantees:
- NumPy (cpu), Numba, and torch-cpu backends reproduce spike-for-spike deterministically
  given the same seed, brain pack, and recorded inputs.
- GPU backends (torch-cuda, torch-rocm, OpenGL compute) reproduce statistically
  due to floating-point atomic sum non-determinism across parallel thread scheduling.
"""
from __future__ import annotations

import json
import os
import platform
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kickthefly.core.version import __version__
from kickthefly.core.crash import log

FORMAT = "kick-the-fly-replay"
FORMAT_VERSION = 1
SUFFIX = ".ktfreplay"


class ReplayError(Exception):
    """Raised when a replay file cannot be loaded or is incompatible."""
    pass


def pack_signature(game_or_brain) -> dict:
    """Extract brain pack neuron count, synapse count, and memory signature."""
    br = getattr(game_or_brain, "brain", game_or_brain)
    if hasattr(br, "flies") and br.flies:
        br = br.flies[0].brain
    W = getattr(getattr(br, "sim", None), "W_csr", None)
    mem = getattr(br, "memory", None)
    return dict(
        n_neurons=int(getattr(br, "n", 0)),
        synapses=int(getattr(W, "nnz", 0)) if W is not None else 0,
        memory_signature=[float(x) for x in mem.signature] if mem is not None and getattr(mem, "signature", None) is not None else None,
    )


def check_compatibility(meta: dict, current_sig: dict) -> str | None:
    """Returns None if compatible, or a user-facing explanation if incompatible."""
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
    if current_sig.get("memory_signature") and sig.get("memory_signature"):
        if not np.allclose(sig["memory_signature"], current_sig["memory_signature"]):
            return "Recorded with a different mushroom body wiring."
    return None


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

    def record(self, step: int, event_type: str, **payload: Any) -> None:
        """Record an input event at a specific simulation step."""
        ev = {"step": int(step), "type": str(event_type), **payload}
        self.events.append(ev)
        if step > self.total_steps:
            self.total_steps = step

    def save(self, path: Path) -> Path:
        """Save recording to a .ktfreplay zip file."""
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
            "events": self.events,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("replay.json", json.dumps(meta, indent=2))
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
        self.events_by_step: dict[int, list[dict[str, Any]]] = {}
        for ev in meta.get("events", []):
            s = int(ev["step"])
            self.events_by_step.setdefault(s, []).append(ev)

    @classmethod
    def load(cls, path: Path, expected_sig: dict | None = None) -> "ReplayPlayer":
        """Load a .ktfreplay file, checking compatibility."""
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

    def events_at(self, step: int) -> list[dict[str, Any]]:
        """Return all events recorded at a given simulation step."""
        return self.events_by_step.get(step, [])

    def is_finished(self, step: int) -> bool:
        """True if the replay reached its recorded end."""
        return step >= self.total_steps
