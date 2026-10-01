"""Wiring manipulations: change the connectome the simulation runs on, reversibly, and say exactly what changed.

Everything here edits the synapse matrix itself rather than driving neurons, so it answers a different question from
brain surgery: not "what happens when this cell type is silenced" but "how much does this result depend on the
reconstruction being right".

  min_synapses     drop every connection reconstructed with fewer than N synapses. The EM reconstruction's weak
                   connections are the ones most likely to be noise: a 1-2 synapse contact can be a mis-assigned
                   fragment. A result that survives a threshold of 5 does not rest on them.
  flip_rows        flip the sign of a neuron's outgoing synapses (excitatory <-> inhibitory). MaleCNS v1.0 predicts
                   each neuron's neurotransmitter with a confidence; the low-confidence ones are where the sign in
                   this simulation could be wrong.
  inhibition_scale scale every synapse whose presynaptic neuron the dataset calls inhibitory (GABA or glutamate).
                   At 0 they are gone; at 1 nothing changes.
  nt_scales        (3.0 day 2, Lab > Pharmacology) scale the synapses of each presynaptic neurotransmitter separately:
                   ((transmitter, factor), ...). nt_min_conf > 0 leaves out neurons whose transmitter is a low-confidence
                   prediction (only measured ones, or predictions at least that sure, are scaled).

All three are reversible: apply() remembers the values it overwrote, so the next apply() or a clear() puts the brain
back exactly as it was, learned synapses included.

The manipulations use the dataset's own numbers (synapse counts, neurotransmitter predictions and their confidence).
How a severity slider maps onto them, and which neurons a bulk flip picks, are choices this game makes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

SIGN_OF = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
INHIBITORY = ("gaba", "glutamate")


@dataclass(frozen=True)
class Wiring:
    """One reversible change to the connectome. The default changes nothing."""

    min_synapses: int = 1
    flip_rows: tuple[int, ...] = field(default=())
    inhibition_scale: float = 1.0
    nt_scales: tuple = field(default=())             # ((transmitter, factor), ...), e.g. (("acetylcholine", 0.4),)
    nt_min_conf: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (self.min_synapses <= 1 and not self.flip_rows and self.inhibition_scale == 1.0
                and all(float(f) == 1.0 for _, f in self.nt_scales))

    def label(self) -> str:
        bits = []
        if self.min_synapses > 1:
            bits.append(f"drop <{self.min_synapses} synapses")
        if self.flip_rows:
            bits.append(f"{len(self.flip_rows):,} neurons sign-flipped")
        if self.inhibition_scale != 1.0:
            bits.append(f"inhibition x{self.inhibition_scale:g}")
        for nt, f in self.nt_scales:
            if float(f) != 1.0:
                bits.append(f"{nt} synapses x{float(f):g}" + (f" (confidence >= {self.nt_min_conf:g} only)" if self.nt_min_conf > 0 else ""))
        return ", ".join(bits) or "unmodified connectome"

    def as_dict(self) -> dict:
        """For metadata and exports: what was changed, without the (possibly huge) list of flipped neurons."""
        d = dict(min_synapses=int(self.min_synapses), flipped_neurons=len(self.flip_rows),
                 inhibition_scale=float(self.inhibition_scale), label=self.label())
        if self.nt_scales:                                # only when used, so older files and validation output are unchanged
            d["nt_scales"] = {str(nt): float(f) for nt, f in self.nt_scales}
            d["nt_min_conf"] = float(self.nt_min_conf)
        return d

    def with_rows(self, rows) -> "Wiring":
        return Wiring(self.min_synapses, tuple(int(r) for r in rows), self.inhibition_scale, self.nt_scales, self.nt_min_conf)

    @staticmethod
    def from_dict(d: dict | None, rows=()) -> "Wiring":
        """Rebuild from as_dict() plus the flipped rows, which save states keep as an array next to it."""
        if not d:
            return Wiring()
        return Wiring(int(d.get("min_synapses", 1)), tuple(int(r) for r in rows),
                      float(d.get("inhibition_scale", 1.0)), _nt_scales_from(d.get("nt_scales")),
                      _finite_or(d.get("nt_min_conf", 0.0), 0.0))


def _finite_or(v, default: float) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _nt_scales_from(nts) -> tuple:
    """nt_scales as saved ({transmitter: factor}). 3.0 day 2 review: a damaged save (a list, a text factor, NaN) crashed the load
    or put NaN into every synapse of that transmitter; anything that isn't a finite, non-negative factor is left out."""
    if not isinstance(nts, dict):
        return ()
    out = []
    for k, v in nts.items():
        f = _finite_or(v, -1.0)
        if f >= 0.0 and f != 1.0:
            out.append((str(k), f))
    return tuple(sorted(out))


@lru_cache(maxsize=1)
def synapse_counts() -> np.ndarray:
    """|synapses| for every entry of W, in the brain pack's CSR order (the same order as sim.W_csr.data).

    Read straight from the pack rather than divided back out of the rate-normalized weights, so the counts are the
    dataset's integers and not a rounding of them.
    """
    from kickthefly.sim import brainpack

    path = brainpack.find()
    if path is None:
        raise FileNotFoundError("brain pack kick_brain.npz not found")
    with np.load(path) as z:
        return np.abs(z["data"]).astype(np.int32)


@lru_cache(maxsize=1)
def edge_pre() -> np.ndarray:
    """The presynaptic neuron's row for every entry of W (W is [post, pre], so this is W.indices)."""
    from kickthefly.sim import brainpack

    path = brainpack.find()
    with np.load(path) as z:
        return z["indices"].astype(np.int32)


@lru_cache(maxsize=1)
def edge_post() -> np.ndarray:
    """The postsynaptic neuron's row for every entry of W."""
    from kickthefly.sim import brainpack

    path = brainpack.find()
    with np.load(path) as z:
        return np.repeat(np.arange(len(z["inv"]), dtype=np.int32), np.diff(z["indptr"]))


def threshold_stats(n: int) -> dict:
    """What a minimum-synapse threshold of n would remove, counted over the whole connectome."""
    counts = synapse_counts()
    pre, post = edge_pre(), edge_post()
    total = int(len(counts))
    drop = counts < max(1, int(n))
    n_drop = int(drop.sum())
    n_neurons = int(max(pre.max(), post.max())) + 1 if total else 0
    kept = ~drop
    in_before = np.bincount(post, minlength=n_neurons)
    in_after = np.bincount(post[kept], minlength=n_neurons)
    out_before = np.bincount(pre, minlength=n_neurons)
    out_after = np.bincount(pre[kept], minlength=n_neurons)
    touched = np.bincount(post[drop], minlength=n_neurons) + np.bincount(pre[drop], minlength=n_neurons)
    return dict(
        threshold=int(n), connections=total, connections_dropped=n_drop,
        connections_dropped_share=n_drop / max(1, total),
        synapses=int(counts.sum()), synapses_dropped=int(counts[drop].sum()),
        neurons=n_neurons, neurons_touched=int(np.count_nonzero(touched)),
        neurons_cut_off=int(np.count_nonzero((in_before > 0) & (in_after == 0))),
        neurons_silenced_output=int(np.count_nonzero((out_before > 0) & (out_after == 0))),
    )


# --- neurotransmitter predictions and their confidence -----------------------------------------------------------
def transmitters(g) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(nt, confidence, source) per neuron, as the brain pack stored them (see brainpack.neurotransmitters)."""
    nt = getattr(g, "nt", None)
    if nt is None:
        raise ValueError("this brain pack predates the neurotransmitter arrays; rebuild it with "
                         "`python -m kickthefly.sim.brainpack build`")
    n = len(nt)
    conf = getattr(g, "nt_conf", None)
    source = getattr(g, "nt_source", None)
    return (np.asarray(nt).astype(str),
            np.full(n, np.nan, np.float32) if conf is None else np.asarray(conf, np.float32),
            np.full(n, "unknown") if source is None else np.asarray(source).astype(str))


def signed_neurons(g) -> np.ndarray:
    """Neurons whose transmitter gives their synapses a sign. Flipping any other neuron would change nothing:
    dopamine, octopamine, serotonin and 'unclear' contacts carry sign 0 and are not in the matrix at all."""
    nt, _, _ = transmitters(g)
    return np.isin(nt, list(SIGN_OF))


def flip_candidates(g, max_conf: float) -> np.ndarray:
    """Signed neurons the dataset is less than `max_conf` sure about, plus those it gives no confidence for.

    A neuron with no confidence is the least certain of all, so it belongs in the set; the Lab screen reports it
    separately so the two are never confused.
    """
    _, conf, _ = transmitters(g)
    unknown = np.isnan(conf)
    return np.flatnonzero(signed_neurons(g) & (unknown | (conf < float(max_conf))))


def confidence_stats(g, max_conf: float) -> dict:
    nt, conf, source = transmitters(g)
    signed = signed_neurons(g)
    cand = flip_candidates(g, max_conf)
    unknown = int(np.count_nonzero(signed & np.isnan(conf)))
    return dict(cutoff=float(max_conf), neurons=int(len(nt)), signed=int(signed.sum()),
                candidates=int(len(cand)), unknown_confidence=unknown,
                measured=int(np.count_nonzero(source == "ground_truth")),
                excitatory=int(np.count_nonzero(nt == "acetylcholine")),
                inhibitory=int(np.count_nonzero(np.isin(nt, INHIBITORY))),
                median_confidence=float(np.nanmedian(conf[signed])) if signed.any() else float("nan"))


def random_flip(g, max_conf: float, share: float, seed: int) -> tuple[int, ...]:
    """One trial's flip set: each candidate flipped independently with probability `share`.

    Which of the uncertain neurons are actually wrong is not knowable, so a trial samples a possible world rather
    than flipping all of them; running many trials is what makes the answer mean something. The sampling is this
    game's choice, the candidate set is the dataset's.
    """
    cand = flip_candidates(g, max_conf)
    if share >= 1.0:
        return tuple(int(x) for x in cand)
    rng = np.random.default_rng(seed)
    return tuple(int(x) for x in cand[rng.random(len(cand)) < share])


def _inhibitory_edges(g) -> np.ndarray:
    """Entries of W whose presynaptic neuron the dataset calls inhibitory (GABA or glutamate)."""
    nt = getattr(g, "nt", None)
    if nt is None:
        raise ValueError("this brain pack has no neurotransmitter predictions; rebuild it with "
                         "`python -m kickthefly.sim.brainpack build`")
    inhib = np.isin(np.asarray(nt).astype(str), INHIBITORY)
    return inhib[edge_pre()]


def confidence_ok(g, min_conf: float) -> np.ndarray:
    """Neurons whose transmitter may be scaled when low-confidence predictions are left out: every measured one
    (source ground_truth) and every prediction with at least min_conf confidence. min_conf <= 0 accepts all."""
    nt, conf, source = transmitters(g)
    if min_conf <= 0:
        return np.ones(len(nt), bool)
    return (source == "ground_truth") | (np.nan_to_num(conf, nan=-1.0) >= float(min_conf))


def _transmitter_edges(g, nt_name: str, min_conf: float = 0.0) -> np.ndarray:
    """Entries of W whose presynaptic neuron has this predicted transmitter (and passes the confidence cut)."""
    nt, _, _ = transmitters(g)
    ok = (nt == nt_name) & confidence_ok(g, min_conf)
    return ok[edge_pre()]


def inhibition_stats(g, scale: float = 1.0) -> dict:
    """Statistics for scaling inhibitory (GABA/glutamate) synapses."""
    nt, conf, source = transmitters(g)
    inhib_edges = _inhibitory_edges(g)
    total_edges = len(synapse_counts())
    n_inhib_edges = int(inhib_edges.sum())
    is_inhib = np.isin(nt, INHIBITORY)
    n_gaba = int(np.count_nonzero(nt == "gaba"))
    n_glut = int(np.count_nonzero(nt == "glutamate"))
    n_inhib_neurons = int(is_inhib.sum())
    return dict(
        scale=float(scale),
        severity=float(np.clip(1.0 - scale, 0.0, 1.0)),
        total_connections=total_edges,
        inhibitory_connections=n_inhib_edges,
        inhibitory_share=n_inhib_edges / max(1, total_edges),
        inhibitory_neurons=n_inhib_neurons,
        gaba_neurons=n_gaba,
        glutamate_neurons=n_glut,
        total_neurons=int(len(nt)),
    )


def modified_entries(g, w: Wiring) -> tuple[np.ndarray, np.ndarray]:
    """(entry indices this wiring changes, multiplier to apply to each). Multiplying keeps learned weights learned."""
    counts = synapse_counts()
    mult = np.ones(len(counts), np.float32)
    if w.min_synapses > 1:
        mult[counts < w.min_synapses] = 0.0
    if w.flip_rows:
        flip = np.zeros(int(edge_pre().max()) + 1, bool)
        flip[np.asarray(w.flip_rows, np.int64)] = True
        mult[flip[edge_pre()]] *= -1.0
    if w.inhibition_scale != 1.0:
        mult[_inhibitory_edges(g)] *= np.float32(w.inhibition_scale)
    for nt_name, factor in w.nt_scales:
        if float(factor) != 1.0:
            mult[_transmitter_edges(g, nt_name, w.nt_min_conf)] *= np.float32(factor)
    idx = np.flatnonzero(mult != 1.0)
    return idx, mult[idx]


def apply(brain, w: Wiring, g=None) -> dict:
    """Put this wiring on a live brain (or any object with .sim), undoing whatever wiring was on it first.

    Returns what changed. Call under the brain's step lock if it is running on a thread.
    """
    sim = getattr(brain, "sim", brain)
    g = g if g is not None else getattr(brain, "graph", None)
    if g is None:
        from kickthefly.core import simcore
        g = simcore.pack()[0]
    _restore(sim, brain)
    if w.is_identity:
        sim.wiring = Wiring()
        return dict(changed=0, **Wiring().as_dict())
    idx, mult = modified_entries(g, w)
    saved = sim.W_csr.data[idx].copy()
    sim.W_csr.data[idx] = saved * mult
    sim.W_csc = sim.W_csr.tocsc()                       # structure is unchanged, so index maps stay valid
    sim._wiring_idx, sim._wiring_saved = idx, saved
    sim.wiring = w
    mem = getattr(brain, "memory", None)
    if mem is not None:                                 # keep learning from writing the change back out
        _apply_to_memory(mem, idx, mult)
    if hasattr(sim, "backend") and hasattr(sim.backend, "on_weights_changed"):
        sim.backend.on_weights_changed()
    return dict(changed=int(len(idx)), **w.as_dict())


def clear(brain) -> None:
    apply(brain, Wiring())


def _restore(sim, brain=None) -> None:
    idx = getattr(sim, "_wiring_idx", None)
    if idx is not None:
        sim.W_csr.data[idx] = sim._wiring_saved
        sim.W_csc = sim.W_csr.tocsc()
        sim._wiring_idx = sim._wiring_saved = None
        if hasattr(sim, "backend") and hasattr(sim.backend, "on_weights_changed"):
            sim.backend.on_weights_changed()
    mem = getattr(brain, "memory", None) if brain is not None else None
    if mem is not None and getattr(mem, "_wiring_saved", None) is not None:
        with mem.lock:
            pos, w0, ww = mem._wiring_saved
            mem.w0[pos], mem.w[pos] = w0, ww
            mem.sim.W_csr.data[mem.csr_pos] = mem.w
            mem.sim.W_csc.data[mem.csc_pos] = mem.w
            mem._wiring_saved = None
            if hasattr(sim, "backend") and hasattr(sim.backend, "on_weights_changed"):
                sim.backend.on_weights_changed()


def _apply_to_memory(mem, idx: np.ndarray, mult: np.ndarray) -> None:
    """The plastic KC -> MBON synapses are written back from memory.w each update; change those the same way, or the
    manipulation would quietly wear off on exactly the synapses learning touches."""
    with mem.lock:
        order = np.argsort(idx)
        hit = np.searchsorted(idx[order], mem.csr_pos)
        hit = np.clip(hit, 0, len(idx) - 1)
        match = idx[order][hit] == mem.csr_pos
        pos = np.flatnonzero(match)
        if not len(pos):
            mem._wiring_saved = None
            return
        m = mult[order][hit[pos]]
        mem._wiring_saved = (pos, mem.w0[pos].copy(), mem.w[pos].copy())
        mem.w0[pos] = mem.w0[pos] * m
        mem.w[pos] = mem.w[pos] * m
        mem.sim.W_csr.data[mem.csr_pos] = mem.w
        mem.sim.W_csc.data[mem.csc_pos] = mem.w
