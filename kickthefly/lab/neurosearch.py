"""Find neurons and the strongest paths between them, for the big brain view (B) and for scripts.

    search(br, "DNp01")             neurons by body ID, type or instance
    top_paths(W, src, dst, k=5)     the k strongest paths from src to dst, up to 3 hops

Path strength is the product of |weight| along the path. A weight is the simulator's own: the signed synapse count
from pre to post divided by all of post's input synapses (sim.W_csr[post, pre]), so a path's strength is roughly the
share of the target's input that flows along it. The sign is the product of the synapses' signs (excitatory +,
inhibitory -). All connectome, nothing tuned.
"""
from __future__ import annotations

import heapq

import numpy as np


def search(br, query: str, limit: int = 12) -> list[int]:
    """Rows matching query, best first: an exact body ID, then an exact type, types starting with it, instances
    starting with it, then any type or instance containing it (case-insensitive)."""
    q = str(query).strip()
    if not q:
        return []
    out: list[int] = []
    body = getattr(br, "body_id", None)
    if q.isdigit() and body is not None:
        out += np.flatnonzero(np.asarray(body).astype(np.int64) == int(q)).tolist()
        if q.isdigit() and int(q) < br.n and not out:          # a row number, as the inspector shows (#123)
            out.append(int(q))
    lq = q.lower()
    types = np.char.lower(np.asarray(br.types).astype(str))
    inst = np.char.lower(np.asarray(getattr(br, "instance", np.full(br.n, ""))).astype(str))
    for mask in (types == lq, np.char.startswith(types, lq), np.char.startswith(inst, lq),
                 (np.char.find(types, lq) >= 0) | (np.char.find(inst, lq) >= 0)):
        for i in np.flatnonzero(mask):
            if len(out) >= limit:
                return out
            if int(i) not in out:
                out.append(int(i))
    return out[:limit]


def top_paths(W, src: int, dst: int, k: int = 5, max_hops: int = 3, Wc=None) -> list[dict]:
    """The k strongest paths src -> ... -> dst with at most max_hops synapses. W is [post, pre] CSR (sim.W_csr); Wc
    is the same matrix as CSC (sim.W_csc), built here if not given.

    Returns [{"nodes": [src, ..., dst], "weights": [w per hop], "strength": product of |w|, "sign": +1 or -1}],
    strongest first. Exhaustive over 1, 2 and 3 hops: every intermediate that connects is considered."""
    src, dst = int(src), int(dst)
    if src == dst:
        return []
    Wc = W.tocsc() if Wc is None else Wc
    best: list[tuple[float, int, tuple]] = []
    tie = 0

    def offer(nodes, weights):
        nonlocal tie
        s = float(np.prod(np.abs(weights)))
        if s <= 0:
            return
        tie += 1
        item = (s, tie, (tuple(nodes), tuple(float(w) for w in weights)))
        if len(best) < k:
            heapq.heappush(best, item)
        elif s > best[0][0]:
            heapq.heapreplace(best, item)

    def row(i):                                         # inputs of i: (pre rows, weights)
        a, b = W.indptr[i], W.indptr[i + 1]
        return W.indices[a:b], W.data[a:b]

    def col(j):                                         # outputs of j: (post rows, weights)
        a, b = Wc.indptr[j], Wc.indptr[j + 1]
        return Wc.indices[a:b], Wc.data[a:b]

    out_idx, out_w = col(src)                           # src -> a
    in_idx, in_w = row(dst)                             # b -> dst
    direct = in_w[in_idx == src]
    if len(direct):
        offer([src, dst], [direct[0]])
    if max_hops >= 2:                                   # src -> m -> dst
        w_to = dict(zip(out_idx.tolist(), out_w.tolist()))
        for m, w2 in zip(in_idx.tolist(), in_w.tolist()):
            w1 = w_to.get(m)
            if w1 is not None and m not in (src, dst):
                offer([src, m, dst], [w1, w2])
    if max_hops >= 3 and len(out_idx) and len(in_idx):  # src -> a -> b -> dst
        a_ok = out_idx[(out_idx != src) & (out_idx != dst)]
        b_ok = in_idx[(in_idx != src) & (in_idx != dst)]
        if len(a_ok) and len(b_ok):
            wa = dict(zip(out_idx.tolist(), out_w.tolist()))
            wb = dict(zip(in_idx.tolist(), in_w.tolist()))
            sub = W[b_ok][:, a_ok].tocoo()              # W[b, a]: a -> b
            # every 3-hop path, scored in full; only the strongest few go through the heap
            s1 = np.abs(np.array([wa[int(a)] for a in a_ok]))[sub.col]
            s3 = np.abs(np.array([wb[int(b)] for b in b_ok]))[sub.row]
            score = s1 * np.abs(sub.data) * s3
            keep = np.argsort(-score)[: max(k * 4, 20)]
            for t in keep:
                a, b = int(a_ok[sub.col[t]]), int(b_ok[sub.row[t]])
                if a != b:
                    offer([src, a, b, dst], [wa[a], float(sub.data[t]), wb[b]])
    ranked = sorted(best, key=lambda x: -x[0])
    return [dict(nodes=list(n), weights=list(w), strength=s, sign=int(np.sign(np.prod(w))) or 1)
            for s, _, (n, w) in ranked]
