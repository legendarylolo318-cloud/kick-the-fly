"""Brains without a window: build, drive and step LIF brains synchronously ("lockstep").

The game runs each brain on its own real-time thread. Assays, validation, protocols, save-state tests and repeated
trials instead step brains directly from the calling thread, so a run is exactly reproducible: the same seed and the
same stimulus schedule give the same spikes on the same machine.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@lru_cache(maxsize=4)
def pack(brain: str = "adult"):
    """(graph-like namespace, W_in, soma) from the bundled brain pack, loaded once per process."""
    from kickthefly.sim import brainpack

    path = brainpack.find(brain=brain)
    if path is None:
        raise FileNotFoundError(f"brain pack for '{brain}' not found (run 'python -m kickthefly.sim.brainpack build')")
    return brainpack.load(path)


_symmetric_weights_cache = None


def symmetrize_weights(g, weights):
    """Mirror-average synaptic weights across bilateral pairs. Clearly a game-rule data modification."""
    global _symmetric_weights_cache
    if _symmetric_weights_cache is not None and _symmetric_weights_cache[0] is weights:
        return _symmetric_weights_cache[1]
    inst = g.instance.astype(str)
    inst_to_idx = {name: idx for idx, name in enumerate(inst) if len(name) > 0}
    perm = np.arange(g.n, dtype=np.int32)
    for idx, name in enumerate(inst):
        if name.endswith("_L"):
            other = inst_to_idx.get(name[:-2] + "_R")
            if other is not None:
                r_name = inst[other]
                if r_name.endswith("_R") and inst_to_idx.get(r_name[:-2] + "_L") == idx:
                    perm[idx] = other
                    perm[other] = idx
    W_mirrored = weights[perm, :][:, perm]
    W_sym = (weights + W_mirrored) * 0.5
    W_sym.eliminate_zeros()
    _symmetric_weights_cache = (weights, W_sym)
    return W_sym


def new_brain(seed: int = 0, memory: bool = True, warmup: int = 600, params: dict | None = None,
              isolated_memory: bool = True, mirror_weights: bool = False, wiring=None, backend: str | None = None,
              brain: str = "adult", individuality: str = "off", individuality_sigma: float = 0.15):
    """A warmed-up Brain that is not running on a thread. isolated_memory: start from the untrained connectome and never
    read or write the player's saved training memory. wiring: a sim.wiring.Wiring applied before the warm-up, so the
    brain settles with the changed connectome rather than on top of a brain that settled without it."""
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import lab
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    g, W, _ = pack(brain=brain)
    if mirror_weights:
        W = symmetrize_weights(g, W)
    lif_params = LIFParams()
    if backend is not None:
        lif_params.backend = backend
    lif_params.individuality = individuality
    lif_params.individuality_sigma = individuality_sigma
    sim = LIFSim(None, lif_params, W_in=W, seed=seed)
    if params:
        lab.apply_to_sim(sim, params)
    br = k.Brain(g, sim, seed=seed)
    br.brain_type = brain
    if memory and getattr(g, "dan_mbon", None) is not None:
        from kickthefly.core import memory as mem_mod

        br.memory = mem_mod.Memory(g, sim, load=not isolated_memory)
        br.memory.save = lambda: None
    br.graph = g
    if wiring is not None and not wiring.is_identity:
        from kickthefly.sim import wiring as wiring_mod

        wiring_mod.apply(br, wiring, g)
    if warmup:
        br.warmup(warmup)
    return br


def step(br, n: int, record: np.ndarray | None = None) -> np.ndarray | None:
    """Advance n steps. record: neuron rows whose spikes to return as an (n, len(rows)) bool array."""
    out = None if record is None else np.zeros((n, len(record)), bool)
    for i in range(n):
        br._step()
        if out is not None:
            out[i] = br.sim.spikes[record]
    return out


def rows_of(br, spec) -> np.ndarray:
    """Neuron rows from a spec: a group name (e.g. 'loom', 'escape'), 'type:DNp01,MDN', 'prefix:KC', 'superclass:x'
    or 'rows:1,2,3'."""
    if isinstance(spec, (list, tuple, np.ndarray)):
        return np.asarray(spec, np.int64)
    spec = str(spec)
    if spec.startswith("type:"):
        return np.flatnonzero(np.isin(br.types, spec[5:].split(",")))
    if spec.startswith("prefix:"):
        m = np.zeros(br.n, bool)
        for p in spec[7:].split(","):
            m |= np.char.startswith(br.types, p)
        return np.flatnonzero(m)
    if spec.startswith("superclass:"):
        return np.flatnonzero(np.isin(br.superclass, spec[11:].split(",")))
    if spec.startswith("rows:"):
        return np.array([int(x) for x in spec[5:].split(",") if x], np.int64)
    if spec in br.col:
        i = br.col[spec]
        if i < br.n_det:
            return np.flatnonzero(br.det_id == i)
        if spec == "whole brain":
            return np.arange(br.n)
        return np.flatnonzero(br.pop_id == i - br.n_det)
    raise ValueError(f"unknown neuron spec {spec!r}")


def drive(br, rows: np.ndarray, amp: float = 0.5) -> None:
    """Hold these neurons driven (like brain surgery's ON, with a chosen current) until undrive().

    This is its own current, added to any surgery already in force rather than replacing it, so silencing a cell
    type and then driving it in an assay leaves it silenced (net negative) instead of quietly undoing the lesion.
    """
    br.drive_cur[rows] = amp
    br.driving = bool(np.any(br.drive_cur))


def undrive(br, rows: np.ndarray) -> None:
    br.drive_cur[rows] = 0
    br.driving = bool(np.any(br.drive_cur))
