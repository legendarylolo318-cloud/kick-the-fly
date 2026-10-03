"""One-off cross-check of lab/netsci.py against independent implementations (3.0 day 4 review). Needs networkx, which the game does not
use or ship: `pip install networkx` into your own environment first.

    python tools/netsci_crosscheck.py [--adult-sample N] [--out FILE.json]

What it compares, on the larva pack (whole) and on an induced subgraph of N random adult neurons (the whole adult graph is too large
for networkx's exact triad census):
  - motif counts: netsci.motif_estimate (sampled wedges) against networkx.triadic_census (exact), for the 13 connected triads;
  - the motif null: on one directed null graph made by netsci.rewire, every in- and out-degree is kept, and the sampled estimate
    matches networkx's exact census of that same graph;
  - rich club: netsci.rich_club's phi(k) against networkx.rich_club_coefficient(normalized=False) on the same skeleton, and the
    undirected null (netsci.rewire_undirected) keeps every skeleton degree;
  - modularity: netsci.modularity of netsci.louvain's partition against networkx.community.modularity of the same partition on the
    same symmetrised weighted graph (must agree exactly), and against networkx's own Louvain (a heuristic: both are reported);
  - NMI: netsci.nmi against a separate implementation written here from the definition (geometric-mean normalisation).
Nothing here is run by the game or the tests.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kickthefly.lab import netsci  # noqa: E402

NX_NAMES = {"021D": "021D", "021U": "021U", "021C": "021C", "111a": "111U", "111b": "111D", "030T": "030T", "030C": "030C",
            "201": "201", "120D": "120D", "120U": "120U", "120C": "120C", "210": "210", "300": "300"}


def nmi_reference(a, b) -> float:
    """NMI from the definition, with Counters (independent of netsci.nmi's numpy contingency table)."""
    n = len(a)
    ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
    mi = sum(c / n * math.log((c / n) / ((ca[x] / n) * (cb[y] / n))) for (x, y), c in cab.items())
    ha = -sum(c / n * math.log(c / n) for c in ca.values())
    hb = -sum(c / n * math.log(c / n) for c in cb.values())
    return mi / math.sqrt(ha * hb)


def compare(name: str, src, dst, syn, n, region, wedges: int, seed: int = 0) -> dict:
    import networkx as nx

    out = dict(graph=name, neurons=int(n), connections=int(len(src)))
    t0 = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    census = nx.triadic_census(G)
    rng = np.random.default_rng(seed)
    est = netsci.motif_estimate(src, dst, n, rng, wedges)
    rows = []
    for k, (mname, _, _) in enumerate(netsci.MOTIFS):
        exact = census[NX_NAMES[mname]]
        e, se = est["counts"][k], est["se"][k]
        rows.append(dict(motif=mname, networkx_name=NX_NAMES[mname], exact=int(exact), estimate=float(e), se=float(se),
                         z=float((e - exact) / se) if se > 0 else (0.0 if e == exact else float("inf"))))
    out["motifs"] = rows
    out["motif_max_abs_z"] = max(abs(r["z"]) for r in rows if math.isfinite(r["z"]))
    # the directed null: degrees kept, and the estimator agrees with an exact census on it too
    ns, nd = netsci.rewire(src, dst, n, np.random.default_rng(seed + 5))
    out["null_degrees_kept"] = bool(np.array_equal(np.bincount(ns, minlength=n), np.bincount(src, minlength=n))
                                    and np.array_equal(np.bincount(nd, minlength=n), np.bincount(dst, minlength=n)))
    out["null_self_loops"] = int(np.count_nonzero(ns == nd))
    out["null_repeats"] = int(len(ns) - len(np.unique(ns * n + nd)))
    H = nx.DiGraph()
    H.add_nodes_from(range(n))
    H.add_edges_from(zip(ns.tolist(), nd.tolist()))
    nc = nx.triadic_census(H)
    ne = netsci.motif_estimate(ns, nd, n, np.random.default_rng(seed + 6), wedges)
    out["null_motifs"] = [dict(motif=m[0], exact=int(nc[NX_NAMES[m[0]]]), estimate=float(ne["counts"][k]), se=float(ne["se"][k]))
                          for k, m in enumerate(netsci.MOTIFS)]
    out["null_reciprocity"] = netsci.reciprocity(ns, nd, np.ones(len(ns)), n)["binary"]
    # rich club on the skeleton
    lo, hi = netsci._skel_edges(src, dst, n)
    U = nx.Graph()
    U.add_nodes_from(range(n))
    U.add_edges_from(zip(lo.tolist(), hi.tolist()))
    deg = np.bincount(np.concatenate([lo, hi]), minlength=n)
    ks = sorted({float(np.percentile(deg, p)) for p in netsci.RICH_PERCENTILES})
    ours = netsci.rich_club((lo, hi), deg, ks)
    theirs = nx.rich_club_coefficient(U, normalized=False)
    out["rich_club"] = [dict(k=r["k"], phi=r["phi"], networkx_phi=float(theirs.get(int(r["k"]), float("nan")))) for r in ours]
    out["rich_club_max_abs_diff"] = max(abs(r["phi"] - r["networkx_phi"]) for r in out["rich_club"]
                                        if math.isfinite(r["phi"]) and math.isfinite(r["networkx_phi"]))
    ulo, uhi = netsci.rewire_undirected(lo, hi, n, np.random.default_rng(seed + 7))
    out["rich_null_degrees_kept"] = bool(np.array_equal(np.bincount(np.concatenate([ulo, uhi]), minlength=n), deg))
    old_null = netsci.skeleton(ns, nd, n)[2]
    out["directed_null_skeleton_degree_changed_neurons"] = int(np.count_nonzero(old_null != deg))
    # modularity of the same partition, and networkx's own Louvain for comparison
    labels, q = netsci.louvain(src, dst, syn, n, np.random.default_rng(seed + 1))
    W = nx.Graph()
    W.add_nodes_from(range(n))
    acc: dict = {}
    for s_, d_, w_ in zip(src.tolist(), dst.tolist(), syn.tolist()):
        key = (min(s_, d_), max(s_, d_))
        acc[key] = acc.get(key, 0.0) + w_
    W.add_weighted_edges_from((a, b, w) for (a, b), w in acc.items())
    parts = [set(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]
    out["modularity_ours"] = float(q)
    out["modularity_networkx_same_partition"] = float(nx.community.modularity(W, parts, weight="weight"))
    nxl = nx.community.louvain_communities(W, weight="weight", seed=seed)
    out["modularity_networkx_louvain"] = float(nx.community.modularity(W, nxl, weight="weight"))
    out["communities_ours"] = int(len(parts))
    out["communities_networkx_louvain"] = int(len(nxl))
    nxlab = np.zeros(n, np.int64)
    for i, c in enumerate(nxl):
        nxlab[list(c)] = i
    out["nmi_ours_vs_networkx_partition"] = float(netsci.nmi(labels, nxlab))
    out["nmi_with_regions_ours"] = float(netsci.nmi(labels, region))
    out["nmi_with_regions_reference"] = float(nmi_reference(labels.tolist(), list(region)))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adult-sample", type=int, default=4000)
    ap.add_argument("--wedges", type=int, default=1_000_000)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = []
    g = netsci.load_edges("larva")
    res.append(compare("larva (whole)", g["src"], g["dst"], g["syn"], g["n"], g["region"], a.wedges))
    if a.adult_sample:
        g = netsci.load_edges("adult")
        rng = np.random.default_rng(1)
        keep = np.sort(rng.choice(g["n"], size=a.adult_sample, replace=False))
        idx = np.full(g["n"], -1, np.int64)
        idx[keep] = np.arange(len(keep))
        m = (idx[g["src"]] >= 0) & (idx[g["dst"]] >= 0)
        res.append(compare(f"adult (induced subgraph of {a.adult_sample} random neurons, seed 1)", idx[g["src"][m]], idx[g["dst"][m]],
                           g["syn"][m], len(keep), g["region"][keep], a.wedges))
        del g
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    for r in res:
        print(f"{r['graph']}: {r['neurons']} neurons, {r['connections']} connections")
        print(f"  motifs: largest |estimate - exact| / se = {r['motif_max_abs_z']:.2f}")
        for m in r["motifs"]:
            print(f"    {m['motif']:5s} ({m['networkx_name']}) exact {m['exact']:>12,}  estimate {m['estimate']:>14,.1f} +- {m['se']:,.1f}")
        print(f"  directed null: degrees kept {r['null_degrees_kept']}, self-loops {r['null_self_loops']}, repeats {r['null_repeats']}, "
              f"reciprocity {r['null_reciprocity']:.4f}")
        print(f"  rich club phi: max |ours - networkx| = {r['rich_club_max_abs_diff']:.2e}; undirected null keeps skeleton degrees: "
              f"{r['rich_null_degrees_kept']}; the directed null changed the skeleton degree of {r['directed_null_skeleton_degree_changed_neurons']} neurons")
        print(f"  modularity of our partition: ours {r['modularity_ours']:.6f}, networkx {r['modularity_networkx_same_partition']:.6f}; "
              f"networkx Louvain {r['modularity_networkx_louvain']:.4f} ({r['communities_networkx_louvain']} communities vs our {r['communities_ours']})")
        print(f"  NMI with regions: ours {r['nmi_with_regions_ours']:.6f}, reference {r['nmi_with_regions_reference']:.6f}; "
              f"our partition vs networkx's {r['nmi_ours_vs_networkx_partition']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
