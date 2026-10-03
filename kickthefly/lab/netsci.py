"""Network science of the brain pack (3.0 day 4): degree distributions, reciprocity, 3-node motifs against a sampled null model,
the rich-club coefficient, modularity and communities, and per-region summaries. Adult and larva.

CONNECTOME vs GAME RULE vs MODEL PREDICTION
  CONNECTOME        every number here is computed from the pack's synapse counts (data/kick_brain.npz, MaleCNS v1.0 filtered to
                    connections of 3+ synapses; kick_larva_brain.npz, Winding et al. 2023): nothing is simulated, no LIF
                    parameter, no game rule is involved. It describes the wiring diagram as this game ships it.
  GAME RULE         only the analysis choices: how many null graphs and wedge samples, the seed, the rich-club degree cut-offs,
                    the community-detection method and its stopping rule. They change how precisely a number is estimated, not
                    what the wiring is.
  MODEL PREDICTION  none. (Nothing here predicts behavior; an enrichment over a null model is a statement about the pack.)

Definitions (the same ones written into every export)
  - A connection is one ordered pair (pre, post) with a nonzero synapse count in the pack; a self-connection is dropped. Degrees
    count connections (binary) and, separately, synapses (weighted). The pack is a MaleCNS subset with a 3-synapse floor, and EM
    reconstruction coverage is uneven, so every number carries those limits; no power law is fitted or claimed.
  - Reciprocity: the share of connections whose reverse connection exists (binary), and sum(min(w_ij, w_ji)) / sum(w) (weighted).
  - Motifs: the 13 connected 3-node subgraphs of a directed graph. They are counted by SAMPLING connected triples (a centre and
    two of its neighbours in the undirected skeleton, each triple weighted by 1 / the number of wedges it contains, so every
    triple counts once) and scaling by the exact wedge count. Counts are estimates with a sampling error, not exact.
  - Null model: degree-preserving directed edge swaps (every neuron keeps its in- and out-degree; no self-loops or parallel
    edges), a few independent samples, the same estimator applied to each. The enrichment is real / null mean; z uses the
    spread of the null samples, which is rough with few of them.
  - Rich club: phi(k) = 2 E(>k) / (N(>k) (N(>k) - 1)) on the undirected skeleton, over the neurons whose skeleton degree exceeds
    k, normalised by the same quantity on the null graphs (rho = phi / phi_null; rho > 1 is a rich club beyond what degrees alone
    give). Its null is its own: degree-preserving UNDIRECTED swaps of the skeleton's edges, so every neuron keeps the skeleton
    degree phi(k) is defined on (analysis version 2, 3.0 day 4 review; version 1 took the skeleton of the directed null, which
    splits reciprocal pairs and so gives most neurons a different skeleton degree than they have).
  - An enrichment over a null mean of 0 (a motif none of the null graphs produced in the sampled wedges) is undefined, not
    infinite: it is reported as undefined with that reason, and its z as undefined too.
  - Communities: a Louvain-style modularity maximisation on the symmetrised synapse-weighted graph (A + A^T), run with
    synchronous moves on a random half of the neurons each sweep. It is a heuristic: the partition is not unique, and the
    reported Q is the exact modularity of the partition it found, not the maximum possible. NMI is the normalised mutual
    information between the communities and the pack's own region labels.

Everything is cached on disk next to the player's data, keyed by the pack's SHA-256 and the analysis settings, with a checksum of
the cached result: a changed pack, changed settings or a damaged file is recomputed, never trusted.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import threading
import time
from pathlib import Path

import numpy as np

ANALYSIS_VERSION = 2               # 2: the rich club's own undirected null; undefined enrichments (3.0 day 4 review)
DEFAULT_NULLS = 3
DEFAULT_WEDGES = 300_000
DEFAULT_SEED = 0
RICH_PERCENTILES = (50, 75, 90, 95, 98, 99)
SWEEPS_PER_LEVEL = 12


class Cancelled(Exception):
    pass


# --- loading -----------------------------------------------------------------------------------------------------------
def load_edges(brain: str = "adult") -> dict:
    """The pack's connections as arrays: src, dst (int64, no self-loops), syn (synapses, > 0), sign (+1 / -1), plus the neuron
    labels. Read from the pack file's own raw synapse counts (the in-game matrix is row-normalised for the simulation)."""
    from kickthefly.core import replay
    from kickthefly.sim import brainpack

    path = brainpack.find(brain=brain)
    if path is None:
        raise FileNotFoundError(f"brain pack for '{brain}' not found (run 'python -m kickthefly.sim.brainpack build')")
    z = np.load(path)
    n = len(z["inv"])
    indptr = z["indptr"]
    post = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))          # rows are the post neurons (W_in is [post, pre])
    pre = z["indices"].astype(np.int64)
    raw = z["data"].astype(np.float64)
    keep = (raw != 0) & (pre != post)
    region = z["region"].astype(str) if "region" in z else np.full(n, "unassigned")
    out = dict(n=n, src=pre[keep], dst=post[keep], syn=np.abs(raw[keep]), sign=np.sign(raw[keep]).astype(np.int8),
               region=region, superclass=z["superclass"].astype(str), type=z["type"].astype(str), brain=brain,
               pack_sha256=replay.pack_sha256(path), pack_name=Path(path).name)
    return out


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# --- degree and reciprocity --------------------------------------------------------------------------------------------
def _summary(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), max=float(x.max()), zero=int(np.count_nonzero(x == 0)))


def _log_hist(deg: np.ndarray) -> list[dict]:
    """Counts in log2 bins: degree 0, 1, 2-3, 4-7, ... (binning only; no distribution is fitted)."""
    edges = [0, 1]
    while edges[-1] <= deg.max():
        edges.append(edges[-1] * 2)
    h, _ = np.histogram(deg, bins=[-0.5, 0.5] + [e - 0.5 for e in edges[2:]] + [np.inf])
    out = [dict(low=0, high=0, count=int(h[0]))]
    for i, (a, b) in enumerate(zip(edges[1:-1], edges[2:])):
        out.append(dict(low=int(a), high=int(b - 1), count=int(h[i + 1])))
    return [r for r in out if r["count"] > 0 or r["low"] == 0]


def degrees(g: dict) -> dict:
    n = g["n"]
    indeg = np.bincount(g["dst"], minlength=n)
    outdeg = np.bincount(g["src"], minlength=n)
    insyn = np.bincount(g["dst"], weights=g["syn"], minlength=n)
    outsyn = np.bincount(g["src"], weights=g["syn"], minlength=n)
    types = g["type"]
    hubs = []
    for name, d in (("in", indeg), ("out", outdeg)):
        for i in np.argsort(-d)[:8]:
            hubs.append(dict(direction=name, row=int(i), type=str(types[i]), degree=int(d[i])))
    return dict(n=int(n), connections=int(len(g["src"])), synapses=float(g["syn"].sum()),
                density=float(len(g["src"]) / (n * (n - 1))),
                in_degree=dict(_summary(indeg), histogram=_log_hist(indeg)),
                out_degree=dict(_summary(outdeg), histogram=_log_hist(outdeg)),
                in_synapses=_summary(insyn), out_synapses=_summary(outsyn), hubs=hubs,
                in_out_degree_correlation=float(np.corrcoef(indeg, outdeg)[0, 1]))


def reciprocity(src, dst, syn, n) -> dict:
    key = src * n + dst
    order = np.argsort(key)
    sk, sw = key[order], syn[order]
    rk = dst * n + src
    pos = np.searchsorted(sk, rk)
    pos[pos >= len(sk)] = len(sk) - 1
    has = sk[pos] == rk
    wr = np.where(has, sw[pos], 0.0)
    return dict(binary=float(has.mean()), weighted=float(np.minimum(syn, wr).sum() / syn.sum()),
                reciprocal_pairs=int(has.sum() // 2))


# --- the null model: degree-preserving directed edge swaps --------------------------------------------------------------
def rewire(src: np.ndarray, dst: np.ndarray, n: int, rng: np.random.Generator, rounds: int = 10, cancel=None):
    """Degree-preserving null: every round pairs the edges at random and swaps their targets ((a->b), (c->d)) -> ((a->d), (c->b))
    when that makes no self-loop and no edge that exists or repeats. In- and out-degrees are exactly preserved."""
    src, dst = src.copy(), dst.copy()
    m = len(src)
    for _ in range(rounds):
        _check(cancel)
        perm = rng.permutation(m)
        half = m // 2
        i, j = perm[:half], perm[half:2 * half]
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        new1, new2 = a * n + d, c * n + b
        ok = (a != d) & (c != b)
        existing = np.sort(src * n + dst)
        for k in (new1, new2):
            pos = np.searchsorted(existing, k)
            pos[pos >= m] = m - 1
            ok &= existing[pos] != k
        both = np.concatenate([new1, new2])
        uniq, inv, cnt = np.unique(both, return_inverse=True, return_counts=True)
        dup = cnt[inv] > 1
        ok &= ~dup[:half] & ~dup[half:]
        dst[i[ok]], dst[j[ok]] = d[ok], b[ok]
    return src, dst


def rewire_undirected(lo: np.ndarray, hi: np.ndarray, n: int, rng: np.random.Generator, rounds: int = 10, cancel=None):
    """Degree-preserving null for an UNDIRECTED edge list (each pair once): every round pairs the edges at random and swaps one end
    ((a-b), (c-d)) -> ((a-d), (c-b)) when that makes no self-loop and no pair that exists or repeats. Every node's degree is exactly
    preserved. Used for the rich club, whose coefficient is defined on the skeleton's degrees."""
    a_, b_ = lo.copy(), hi.copy()
    m = len(a_)
    for _ in range(rounds):
        _check(cancel)
        flip = rng.random(m) < 0.5                     # which end of each edge is "first": both swap patterns get used
        a_, b_ = np.where(flip, b_, a_), np.where(flip, a_, b_)
        perm = rng.permutation(m)
        half = m // 2
        i, j = perm[:half], perm[half:2 * half]
        a, b, c, d = a_[i], b_[i], a_[j], b_[j]
        k1 = np.minimum(a, d) * n + np.maximum(a, d)
        k2 = np.minimum(c, b) * n + np.maximum(c, b)
        ok = (a != d) & (c != b)
        existing = np.sort(np.minimum(a_, b_) * n + np.maximum(a_, b_))
        for k in (k1, k2):
            pos = np.searchsorted(existing, k)
            pos[pos >= m] = m - 1
            ok &= existing[pos] != k
        both = np.concatenate([k1, k2])
        _, inv, cnt = np.unique(both, return_inverse=True, return_counts=True)
        dup = cnt[inv] > 1
        ok &= ~dup[:half] & ~dup[half:]
        b_[i[ok]], b_[j[ok]] = d[ok], b[ok]
    return np.minimum(a_, b_), np.maximum(a_, b_)


# --- the undirected skeleton ---------------------------------------------------------------------------------------------
def skeleton(src, dst, n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(indptr, indices, degree) of the undirected skeleton (each connected pair once, either direction or both)."""
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = np.unique(lo * n + hi)
    a, b = key // n, key % n
    u = np.concatenate([a, b])
    v = np.concatenate([b, a])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    deg = np.bincount(u, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(deg)])
    return indptr, v, deg


# --- 3-node motifs -----------------------------------------------------------------------------------------------------
# The 13 connected directed triads, each given as ordered edge lists on nodes 0, 1, 2. The names are descriptive (the standard
# M-A-N code first); they are structural and do not depend on a convention for the letters.
_PAIRS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))
MOTIFS = (
    ("021D", "out-star: one neuron sends to two others", ((1, 0), (1, 2))),
    ("021U", "in-star: two neurons send to one", ((0, 1), (2, 1))),
    ("021C", "chain: A -> B -> C", ((0, 1), (1, 2))),
    ("111a", "mutual pair, one of them also sends to a third", ((0, 1), (1, 0), (1, 2))),
    ("111b", "mutual pair, a third sends to one of them", ((0, 1), (1, 0), (2, 1))),
    ("030T", "feed-forward loop", ((0, 1), (1, 2), (0, 2))),
    ("030C", "3-cycle", ((0, 1), (1, 2), (2, 0))),
    ("201", "two mutual pairs sharing a neuron", ((0, 1), (1, 0), (1, 2), (2, 1))),
    ("120D", "a neuron sends to both of a mutual pair", ((1, 0), (1, 2), (0, 2), (2, 0))),
    ("120U", "both of a mutual pair send to a third", ((0, 1), (2, 1), (0, 2), (2, 0))),
    ("120C", "a chain closed by a mutual pair", ((0, 1), (1, 2), (0, 2), (2, 0))),
    ("210", "two mutual pairs and a one-way link", ((0, 1), (1, 0), (1, 2), (2, 1), (0, 2))),
    ("300", "all three pairs mutual", ((0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0))),
)


def _code(edges) -> int:
    return sum(1 << _PAIRS.index(e) for e in edges)


def _build_table() -> np.ndarray:
    """code (6 bits) -> motif index (0-12) or -1 for the triads that are not connected."""
    canon = {}
    for name_i, (_, _, edges) in enumerate(MOTIFS):
        codes = set()
        for perm in itertools.permutations(range(3)):
            codes.add(_code([(perm[a], perm[b]) for a, b in edges]))
        for c in codes:
            canon[c] = name_i
    t = np.full(64, -1, np.int8)
    for c, i in canon.items():
        t[c] = i
    return t


_TABLE = _build_table()


def _has(sorted_keys: np.ndarray, k: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_keys, k)
    pos[pos >= len(sorted_keys)] = len(sorted_keys) - 1
    return sorted_keys[pos] == k


def motif_estimate(src, dst, n, rng, wedges: int, sk=None, cancel=None) -> dict:
    """Estimated counts of the 13 connected triads from `wedges` sampled wedges (see the module docstring)."""
    indptr, ind, deg = sk if sk is not None else skeleton(src, dst, n)
    wc = deg.astype(np.float64) * (deg - 1) / 2
    total_wedges = float(wc.sum())
    p = wc / total_wedges
    centre = rng.choice(n, size=wedges, p=p)
    d = deg[centre]
    oa = (rng.random(wedges) * d).astype(np.int64)
    ob = (rng.random(wedges) * (d - 1)).astype(np.int64)
    ob = np.where(ob >= oa, ob + 1, ob)                     # a second neighbour distinct from the first
    a = ind[indptr[centre] + oa]
    b = ind[indptr[centre] + ob]
    keys = np.sort(src * n + dst)
    bits = np.zeros(wedges, np.int64)
    for bit, (x, y) in enumerate(((centre, a), (a, centre), (centre, b), (b, centre), (a, b), (b, a))):
        bits |= _has(keys, x * n + y).astype(np.int64) << bit
    cls = _TABLE[bits]
    closed = (bits >> 4) & 3 != 0                         # a and b are linked: a triangle, seen from all three of its wedges
    w = np.where(closed, 1.0 / 3.0, 1.0)
    est, se = [], []
    for k in range(len(MOTIFS)):
        sel = cls == k
        x = np.where(sel, w, 0.0)
        est.append(total_wedges * float(x.mean()))
        se.append(total_wedges * float(x.std(ddof=1) / np.sqrt(wedges)))
    return dict(counts=est, se=se, total_wedges=total_wedges,
                connected_triples=total_wedges * float(w.mean()), unclassified=int((cls < 0).sum()))


def motif_table(mot_real: dict, null_counts: list, wedges: int) -> list[dict]:
    """One row per motif: the estimate, its sampling error, the null mean and sd, the enrichment (real / null mean) and z. An
    enrichment or z whose null mean or spread is 0 is undefined: None, with the reason in `enrichment_note` (3.0 day 4 review:
    the all-mutual triad's enrichment was printed as nan)."""
    nc = np.asarray(null_counts, float) if len(null_counts) else np.zeros((0, len(MOTIFS)))
    rows = []
    for k, (name, desc, _) in enumerate(MOTIFS):
        mean = float(nc[:, k].mean()) if len(nc) else None
        sd = float(nc[:, k].std(ddof=1)) if len(nc) > 1 else None
        real = float(mot_real["counts"][k])
        sd_eff = float(np.hypot(sd, mot_real["se"][k])) if sd is not None else None
        note = ""
        if not len(nc):
            enr = z = None
            note = "undefined: no null graphs were made"
        elif mean <= 0:
            enr = z = None
            note = (f"undefined: none of the {len(nc)} null graphs had this motif in {int(wedges):,} sampled wedges each "
                    f"(the real graph has about {real:,.0f}); real / 0 has no value")
        else:
            enr = real / mean
            z = (real - mean) / sd_eff if sd_eff is not None and sd_eff > 0 else None
            if z is None:
                note = "z undefined: fewer than two null graphs, or no spread among them"
        rows.append(dict(motif=name, description=desc, count_estimate=real, sampling_se=float(mot_real["se"][k]), null_mean=mean,
                         null_sd=sd, enrichment=enr, z=z, enrichment_note=note))
    return rows


def fmt_enrichment(m: dict) -> str:
    return "undefined" if m.get("enrichment") is None else f"x{m['enrichment']:.2f}"


def fmt_z(m: dict) -> str:
    return "undefined" if m.get("z") is None else f"{m['z']:.1f}"


# --- rich club ---------------------------------------------------------------------------------------------------------
def rich_club(skel_edges: tuple[np.ndarray, np.ndarray], deg: np.ndarray, ks) -> list[dict]:
    a, b = skel_edges
    mind = np.minimum(deg[a], deg[b])
    out = []
    for k in ks:
        nk = int((deg > k).sum())
        ek = int((mind > k).sum())
        phi = 2.0 * ek / (nk * (nk - 1)) if nk > 1 else float("nan")
        out.append(dict(k=float(k), neurons=nk, edges=ek, phi=phi))
    return out


def _skel_edges(src, dst, n):
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    key = np.unique(lo * n + hi)
    return key // n, key % n


# --- communities -------------------------------------------------------------------------------------------------------
def louvain(src, dst, syn, n, rng, sweeps: int = SWEEPS_PER_LEVEL, cancel=None, progress=None) -> tuple[np.ndarray, float]:
    """(community label per neuron, modularity Q) on the symmetrised synapse-weighted graph. See the module docstring."""
    from scipy import sparse as sp

    A = sp.csr_array((np.concatenate([syn, syn]), (np.concatenate([src, dst]), np.concatenate([dst, src]))), shape=(n, n))
    A.sum_duplicates()
    m2 = float(A.sum())
    labels = np.arange(n)
    level = 0
    while True:
        _check(cancel)
        N = A.shape[0]
        k = np.asarray(A.sum(axis=1)).ravel()
        selfw = np.asarray(A.diagonal())                    # a neuron's self-loop (after aggregation) goes wherever it goes
        comm = np.arange(N)
        tot = k.copy()
        moved_any = False
        for sweep in range(sweeps):
            _check(cancel)
            H = sp.csr_array((np.ones(N), (np.arange(N), comm)), shape=(N, N))
            M = (A @ H).tocsr()
            M.sort_indices()
            r = np.repeat(np.arange(N), np.diff(M.indptr))
            c = M.indices
            w = M.data
            own = comm[r] == c
            ki = k[r]
            gain = w - np.where(own, selfw[r], 0.0) - ki * (tot[c] - np.where(own, ki, 0.0)) / m2
            # staying scores the weight to the own community without the self-loop (it is the same in every option), minus the expected share
            has_own = np.zeros(N, bool)
            stay = np.full(N, -np.inf)
            stay[r[own]] = gain[own]
            has_own[r[own]] = True
            stay = np.where(has_own, stay, -k * (tot[comm] - k) / m2)
            best = np.full(N, -np.inf)
            starts = M.indptr[:-1]
            nz = np.diff(M.indptr) > 0
            best[nz] = np.maximum.reduceat(gain, starts[nz])
            is_best = gain >= best[r] - 1e-12
            pick = np.full(N, -1, np.int64)
            idx = np.flatnonzero(is_best)
            pick[r[idx[::-1]]] = c[idx[::-1]]               # the first best neighbour community in index order
            move = (best > stay + 1e-12) & (pick >= 0) & (pick != comm) & (rng.random(N) < 0.5)
            if not move.any():
                if sweep == 0:
                    break
                continue
            mv = np.flatnonzero(move)
            np.add.at(tot, comm[mv], -k[mv])
            np.add.at(tot, pick[mv], k[mv])
            comm[mv] = pick[mv]
            moved_any = True
            if progress:
                progress(f"communities: level {level}, sweep {sweep + 1}, {len(np.unique(comm))} communities")
            if len(mv) < max(1, N // 2000):
                break
        if not moved_any:
            break
        uniq, comm = np.unique(comm, return_inverse=True)
        labels = comm[labels]
        H = sp.csr_array((np.ones(N), (np.arange(N), comm)), shape=(N, len(uniq)))
        A = (H.T @ A @ H).tocsr()
        level += 1
        if A.shape[0] == 1:
            break
    return labels, modularity(src, dst, syn, n, labels)


def modularity(src, dst, syn, n, labels) -> float:
    from scipy import sparse as sp

    m2 = 2.0 * float(syn.sum())
    k = np.bincount(src, weights=syn, minlength=n) + np.bincount(dst, weights=syn, minlength=n)
    same = labels[src] == labels[dst]
    inside = 2.0 * float(syn[same].sum())
    tot = np.bincount(labels, weights=k)
    return float(inside / m2 - float(np.sum((tot / m2) ** 2)))


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((len(ua), len(ub)))
    np.add.at(cont, (ia, ib), 1)
    p = cont / cont.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (pa[:, None] * pb[None, :])[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / max(1e-12, np.sqrt(ha * hb)) if ha > 0 and hb > 0 else 0.0


def community_summary(labels, g, top: int = 12) -> list[dict]:
    uniq, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    out = []
    for ci in np.argsort(-cnt)[:top]:
        members = np.flatnonzero(inv == ci)
        regs, rc = np.unique(g["region"][members], return_counts=True)
        sup, sc = np.unique(g["superclass"][members], return_counts=True)
        out.append(dict(community=int(ci), size=int(cnt[ci]), top_region=str(regs[np.argmax(rc)]),
                        top_region_share=float(rc.max() / len(members)), top_superclass=str(sup[np.argmax(sc)]),
                        top_superclass_share=float(sc.max() / len(members))))
    return out


# --- regions -----------------------------------------------------------------------------------------------------------
def region_summary(g: dict) -> list[dict]:
    n, src, dst, syn, sign = g["n"], g["src"], g["dst"], g["syn"], g["sign"]
    reg = g["region"]
    names, ridx = np.unique(reg, return_inverse=True)
    indeg = np.bincount(dst, minlength=n)
    outdeg = np.bincount(src, minlength=n)
    out = []
    same = ridx[src] == ridx[dst]
    for i, name in enumerate(names):
        members = ridx == i
        out_edges = ridx[src] == i
        in_edges = ridx[dst] == i
        osyn = float(syn[out_edges].sum())
        isyn = float(syn[in_edges].sum())
        out.append(dict(region=str(name), neurons=int(members.sum()), share_of_neurons=float(members.mean()),
                        mean_in_degree=float(indeg[members].mean()), mean_out_degree=float(outdeg[members].mean()),
                        out_synapses=osyn, in_synapses=isyn,
                        share_out_staying_inside=float(syn[out_edges & same].sum() / osyn) if osyn else float("nan"),
                        share_in_from_inside=float(syn[in_edges & same].sum() / isyn) if isyn else float("nan"),
                        inhibitory_share_of_out_synapses=float(syn[out_edges & (sign < 0)].sum() / osyn) if osyn else float("nan")))
    return sorted(out, key=lambda r: -r["neurons"])


# --- the whole analysis, cached ----------------------------------------------------------------------------------------
def settings(nulls: int = DEFAULT_NULLS, wedges: int = DEFAULT_WEDGES, seed: int = DEFAULT_SEED) -> dict:
    return dict(version=ANALYSIS_VERSION, nulls=int(nulls), wedges=int(wedges), seed=int(seed), rich_percentiles=list(RICH_PERCENTILES),
                sweeps_per_level=SWEEPS_PER_LEVEL)


def cache_path(brain: str, pack_sha: str | None, st: dict) -> Path:
    from kickthefly.core import paths

    h = hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest()[:10]
    return paths.ensure_dir(paths.get().data_dir / "cache") / f"netsci_{brain}_{(pack_sha or 'nopack')[:12]}_{h}.json"


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_cached(brain: str = "adult", st: dict | None = None) -> dict | None:
    """The cached analysis for this pack and these settings, or None (missing, a different pack, or failing its checksum)."""
    from kickthefly.core import replay
    from kickthefly.sim import brainpack

    st = st or settings()
    pack = brainpack.find(brain=brain)
    if pack is None:
        return None
    path = cache_path(brain, replay.pack_sha256(pack), st)
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        payload = wrapper["result"]
        if wrapper.get("sha256") != _digest(payload) or payload.get("pack_sha256") != replay.pack_sha256(pack):
            return None
        return payload
    except (OSError, ValueError, KeyError):
        return None


def compute(brain: str = "adult", nulls: int = DEFAULT_NULLS, wedges: int = DEFAULT_WEDGES, seed: int = DEFAULT_SEED,
            progress=None, cancel: threading.Event | None = None, force: bool = False, cache: bool = True) -> dict:
    """The full analysis. progress(fraction, label); cancel: a threading.Event that stops it between steps (Cancelled is raised).
    Cached on disk (see the module docstring); force=True recomputes. Heavy on the adult pack (minutes): run it off the game
    thread, as the Lab page does."""
    st = settings(nulls, wedges, seed)
    if cache and not force:
        hit = load_cached(brain, st)
        if hit is not None:
            if progress:
                progress(1.0, "loaded from the cache")
            hit = dict(hit, from_cache=True)
            return hit

    def step(frac, label):
        _check(cancel)
        if progress:
            progress(frac, label)

    t0 = time.time()
    step(0.0, "reading the pack")
    g = load_edges(brain)
    n, src, dst, syn = g["n"], g["src"], g["dst"], g["syn"]
    rng = np.random.default_rng(seed)
    step(0.04, "degrees and reciprocity")
    deg = degrees(g)
    rec = reciprocity(src, dst, syn, n)
    step(0.08, "the undirected skeleton")
    sk = skeleton(src, dst, n)
    skel_deg = sk[2]
    se = _skel_edges(src, dst, n)
    ks = sorted({float(np.percentile(skel_deg, p)) for p in RICH_PERCENTILES})
    rich_real = rich_club(se, skel_deg, ks)
    step(0.12, "sampling motifs in the real graph")
    mot_real = motif_estimate(src, dst, n, rng, wedges, sk=sk, cancel=cancel)
    null_counts, null_rich, null_rec = [], [], []
    rich_rng = np.random.default_rng(seed + 2)               # its own stream: the motif nulls are the same as with analysis version 1
    for i in range(nulls):
        base = 0.15 + 0.4 * i / max(1, nulls)
        step(base, f"null graph {i + 1}/{nulls}: swapping edges")
        ns, nd = rewire(src, dst, n, rng, cancel=cancel)
        step(base + 0.2 / max(1, nulls), f"null graph {i + 1}/{nulls}: motifs and rich club")
        nsk = skeleton(ns, nd, n)
        null_counts.append(motif_estimate(ns, nd, n, rng, wedges, sk=nsk, cancel=cancel)["counts"])
        null_rec.append(reciprocity(ns, nd, np.ones(len(ns)), n)["binary"])
        del ns, nd, nsk
        ulo, uhi = rewire_undirected(se[0], se[1], n, rich_rng, cancel=cancel)       # the rich club's own null (see the docstring)
        null_rich.append([r["phi"] for r in rich_club((ulo, uhi), skel_deg, ks)])
        del ulo, uhi
    motifs = motif_table(mot_real, null_counts, wedges)
    rich = []
    for i, r in enumerate(rich_real):
        pn = [row[i] for row in null_rich]
        phi_null = float(np.nanmean(pn)) if pn else float("nan")
        rich.append(dict(r, phi_null=phi_null, rho=r["phi"] / phi_null if phi_null and phi_null > 0 else float("nan")))
    rec_null = float(np.mean(null_rec)) if null_rec else float("nan")
    rec["null_binary_mean"] = rec_null
    rec["enrichment"] = rec["binary"] / rec_null if rec_null and rec_null > 0 else float("nan")
    step(0.62, "communities")
    labels, q = louvain(src, dst, syn, n, np.random.default_rng(seed + 1), cancel=cancel,
                        progress=lambda s: step(0.62 + 0.3 * 0.5, s))
    comm = dict(modularity=q, communities=int(len(np.unique(labels))), nmi_with_regions=nmi(labels, g["region"]),
                nmi_with_superclass=nmi(labels, g["superclass"]), largest=community_summary(labels, g))
    step(0.95, "regions")
    regions = region_summary(g)
    result = dict(kind="netsci", brain=brain, analysis_version=ANALYSIS_VERSION, settings=st, pack_sha256=g["pack_sha256"],
                  pack_name=g["pack_name"], created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
                  degrees=deg, reciprocity=rec, motifs=motifs, motif_meta=dict(total_wedges=mot_real["total_wedges"],
                                                                              connected_triples=mot_real["connected_triples"],
                                                                              wedges_sampled=int(wedges), null_graphs=int(nulls)),
                  rich_club=rich, communities=comm, regions=regions,
                  tags=dict(everything="CONNECTOME (computed from the pack; no simulation)",
                            analysis_choices="GAME RULE (nulls, samples, seed, cut-offs, community method)"))
    if cache:
        wrapper = dict(sha256=_digest(result), result=result)
        p = cache_path(brain, g["pack_sha256"], st)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(wrapper), encoding="utf-8")
        tmp.replace(p)
    step(1.0, "done")
    return result


# --- reading and exporting ---------------------------------------------------------------------------------------------
def summary(res: dict) -> str:
    d, r, c = res["degrees"], res["reciprocity"], res["communities"]
    lines = [f"NETWORK SCIENCE of the {res['brain']} pack ({res['pack_name']}, sha256 {res['pack_sha256'][:12]}...)",
             f"  {d['n']:,} neurons, {d['connections']:,} connections, {d['synapses']:,.0f} synapses, density {d['density']:.2e}",
             f"  in-degree mean {d['in_degree']['mean']:.1f} (median {d['in_degree']['median']:.0f}, max {d['in_degree']['max']:.0f}); "
             f"out-degree mean {d['out_degree']['mean']:.1f} (median {d['out_degree']['median']:.0f}, max {d['out_degree']['max']:.0f})",
             f"  reciprocity: {r['binary']:.3f} of connections have their reverse (degree-preserving null "
             f"{r['null_binary_mean']:.3f}, x{r['enrichment']:.2f}); synapse-weighted {r['weighted']:.3f}",
             f"  communities: {c['communities']:,}, modularity Q = {c['modularity']:.3f}, NMI with regions {c['nmi_with_regions']:.2f}",
             "  motifs (estimated by sampling; enrichment = real / degree-preserving null):"]
    for m in res["motifs"]:
        lines.append(f"    {m['motif']:5s} {m['count_estimate']:>14,.0f}  {fmt_enrichment(m):>9}  z {fmt_z(m):>9}  {m['description']}"
                     + (f"  [{m['enrichment_note']}]" if m.get("enrichment") is None and m.get("enrichment_note") else ""))
    lines.append("  rich club (rho = phi / phi_null):  " + "  ".join(f"k>{x['k']:.0f}: {x['rho']:.2f}" for x in res["rich_club"]))
    lines.append("  every number here is computed from the wiring: CONNECTOME. The analysis choices (nulls, samples) are GAME RULE.")
    return "\n".join(lines)


def export_csv(res: dict, folder: Path | str) -> list[Path]:
    """One CSV per table: degree histograms, summary, motifs, rich club, communities, regions, and the settings/definitions."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    brain = res["brain"]
    written = []

    def w(name, header, data):
        p = folder / f"netsci_{brain}_{name}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(header)
            cw.writerows(data)
        written.append(p)

    d = res["degrees"]
    w("degree_histogram", ["direction", "degree_low", "degree_high", "neurons"],
      [[k.split("_")[0], h["low"], h["high"], h["count"]] for k in ("in_degree", "out_degree") for h in d[k]["histogram"]])
    w("degree_summary", ["quantity", "mean", "median", "p90", "p99", "max", "neurons_with_zero"],
      [[k, *(d[k][x] for x in ("mean", "median", "p90", "p99", "max", "zero"))]
       for k in ("in_degree", "out_degree", "in_synapses", "out_synapses")])
    w("hubs", ["direction", "row", "type", "degree"], [[h["direction"], h["row"], h["type"], h["degree"]] for h in d["hubs"]])
    r = res["reciprocity"]
    w("reciprocity", ["binary", "weighted", "reciprocal_pairs", "null_binary_mean", "enrichment"],
      [[r["binary"], r["weighted"], r["reciprocal_pairs"], r["null_binary_mean"], r["enrichment"]]])
    w("motifs", ["motif", "description", "count_estimate", "sampling_se", "null_mean", "null_sd", "enrichment", "z", "note"],
      [[*("" if m[k] is None else m[k] for k in ("motif", "description", "count_estimate", "sampling_se", "null_mean", "null_sd",
                                                   "enrichment", "z")), m.get("enrichment_note", "")]
       for m in res["motifs"]])
    w("rich_club", ["k", "neurons_above_k", "edges_among_them", "phi", "phi_null", "rho"],
      [[x["k"], x["neurons"], x["edges"], x["phi"], x["phi_null"], x["rho"]] for x in res["rich_club"]])
    c = res["communities"]
    w("communities", ["rank", "size", "top_region", "top_region_share", "top_superclass", "top_superclass_share"],
      [[i + 1, x["size"], x["top_region"], x["top_region_share"], x["top_superclass"], x["top_superclass_share"]]
       for i, x in enumerate(c["largest"])])
    w("regions", ["region", "neurons", "share_of_neurons", "mean_in_degree", "mean_out_degree", "out_synapses", "in_synapses",
                  "share_out_staying_inside", "share_in_from_inside", "inhibitory_share_of_out_synapses"],
      [[x[k] for k in ("region", "neurons", "share_of_neurons", "mean_in_degree", "mean_out_degree", "out_synapses", "in_synapses",
                       "share_out_staying_inside", "share_in_from_inside", "inhibitory_share_of_out_synapses")]
       for x in res["regions"]])
    w("about", ["key", "value"],
      [["pack", res["pack_name"]], ["pack_sha256", res["pack_sha256"]], ["modularity", c["modularity"]],
       ["communities", c["communities"]], ["nmi_with_regions", c["nmi_with_regions"]], *res["settings"].items(),
       ["tag", "CONNECTOME for every number; analysis choices GAME RULE; no MODEL PREDICTION"],
       ["note", "motif counts are sampled estimates; the community partition is a heuristic, not a unique optimum"]])
    return written


def main(args) -> int:
    """--headless --netsci [adult|larva|both] [--out DIR]: compute (or load from the cache), print the summary, write the CSVs."""
    from kickthefly.lab import recorder

    which = ("adult", "larva") if args.netsci == "both" else (args.netsci,)
    t0 = time.time()
    last = [-1.0]

    def progress(frac, label):
        if frac - last[0] >= 0.02 or frac >= 1.0:
            last[0] = frac
            print(f"  {frac * 100:.0f}% {label} ({time.time() - t0:.0f}s)", flush=True)

    for brain in which:
        last[0] = -1.0
        try:
            res = compute(brain, progress=progress)
        except FileNotFoundError as e:
            print(f"error: {e}")
            return 2
        folder = Path(args.out) if args.out else recorder.exports_dir() / f"netsci-{brain}"
        files = export_csv(res, folder)
        print(summary(res))
        print(f"{len(files)} CSV files written to {folder}" + (" (from the cache)" if res.get("from_cache") else ""))
    return 0
