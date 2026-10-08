"""3.0 day 4, Network science (lab/netsci.py): the pure parts against small graphs whose answers are known, the null model's
invariants, the sampled motif estimate against an exact count, the community finder on a planted partition, and the cached analysis
end to end on the SYNTHETIC pack (plumbing, not biology)."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from kickthefly.lab import netsci

RNG = np.random.default_rng


def _random_graph(n=40, p=0.12, seed=0):
    rng = RNG(seed)
    a = rng.random((n, n)) < p
    np.fill_diagonal(a, False)
    src, dst = np.nonzero(a)
    return src.astype(np.int64), dst.astype(np.int64), n


# --- motif classes -----------------------------------------------------------------------------------------------------------------
def test_there_are_thirteen_connected_triad_classes_and_every_labelling_of_one_lands_in_it():
    assert len(netsci.MOTIFS) == 13 and len({m[0] for m in netsci.MOTIFS}) == 13
    assert sorted(set(netsci._TABLE[netsci._TABLE >= 0].tolist())) == list(range(13))
    for k, (_, _, edges) in enumerate(netsci.MOTIFS):
        for perm in itertools.permutations(range(3)):
            code = netsci._code([(perm[a], perm[b]) for a, b in edges])
            assert netsci._TABLE[code] == k
    # the triads that are not connected (no edges, or only one pair joined) are not counted
    assert netsci._TABLE[0] == -1 and netsci._TABLE[netsci._code([(0, 1)])] == -1 and netsci._TABLE[netsci._code([(0, 1), (1, 0)])] == -1


def _exact_counts(src, dst, n):
    adj = np.zeros((n, n), bool)
    adj[src, dst] = True
    out = np.zeros(13)
    for a, b, c in itertools.combinations(range(n), 3):
        code = 0
        for bit, (x, y) in enumerate(netsci._PAIRS):
            node = (a, b, c)
            code |= int(adj[node[x], node[y]]) << bit
        k = netsci._TABLE[code]
        if k >= 0:
            out[k] += 1
    return out


def test_the_sampled_motif_counts_match_an_exact_count_on_a_small_graph():
    src, dst, n = _random_graph(36, 0.14, 3)
    exact = _exact_counts(src, dst, n)
    est = netsci.motif_estimate(src, dst, n, RNG(5), wedges=400_000)
    assert est["unclassified"] == 0
    for k in range(13):
        if exact[k] >= 30:
            assert abs(est["counts"][k] - exact[k]) <= max(4 * est["se"][k], 0.06 * exact[k]), (netsci.MOTIFS[k][0], est["counts"][k], exact[k])
    assert abs(est["connected_triples"] - exact.sum()) <= 0.05 * exact.sum()


# --- degree, reciprocity, skeleton, rich club ----------------------------------------------------------------------------------------
def test_reciprocity_of_a_known_graph():
    src = np.array([0, 1, 1, 2, 3])
    dst = np.array([1, 0, 2, 1, 0])           # 0<->1, 1<->2 mutual; 3->0 one-way
    r = netsci.reciprocity(src, dst, np.array([4.0, 2.0, 1.0, 1.0, 5.0]), 4)
    assert r["binary"] == pytest.approx(4 / 5) and r["reciprocal_pairs"] == 2
    assert r["weighted"] == pytest.approx((2 + 2 + 1 + 1) / 13)


def test_degrees_count_connections_and_synapses_separately():
    g = dict(n=4, src=np.array([0, 0, 1]), dst=np.array([1, 2, 2]), syn=np.array([5.0, 3.0, 4.0]), type=np.array(list("abcd")))
    d = netsci.degrees(g)
    assert d["connections"] == 3 and d["synapses"] == 12 and d["in_degree"]["max"] == 2 and d["out_degree"]["max"] == 2
    assert d["in_synapses"]["max"] == 7 and d["in_degree"]["zero"] == 2 and d["density"] == pytest.approx(3 / 12)
    assert sum(h["count"] for h in d["in_degree"]["histogram"]) == 4


def test_the_rich_club_coefficient_of_a_clique_with_a_tail():
    clique = list(itertools.combinations(range(5), 2))
    edges = clique + [(4, 5), (5, 6), (6, 7)]
    src = np.array([a for a, b in edges])
    dst = np.array([b for a, b in edges])
    n = 8
    _, _, deg = netsci.skeleton(src, dst, n)
    rc = netsci.rich_club(netsci._skel_edges(src, dst, n), deg, [1, 3])
    assert rc[1]["neurons"] == 5 and rc[1]["edges"] == 10 and rc[1]["phi"] == pytest.approx(1.0)       # degree > 3: the clique
    assert rc[0]["neurons"] == 7 and rc[0]["edges"] == 12                                              # degree > 1 adds nodes 5, 6 and the edges 4-5, 5-6


# --- the null model ------------------------------------------------------------------------------------------------------------------
def test_the_null_model_keeps_every_degree_and_makes_no_self_loop_or_repeat():
    src, dst, n = _random_graph(60, 0.1, 1)
    ns, nd = netsci.rewire(src, dst, n, RNG(2), rounds=8)
    assert np.array_equal(np.bincount(src, minlength=n), np.bincount(ns, minlength=n))
    assert np.array_equal(np.bincount(dst, minlength=n), np.bincount(nd, minlength=n))
    assert not np.any(ns == nd) and len(set(zip(ns.tolist(), nd.tolist()))) == len(ns)
    assert (ns != src).sum() + (nd != dst).sum() > 0.5 * len(src), "the swaps did not move most edges"


def test_a_cancelled_run_stops_between_steps():
    import threading

    ev = threading.Event()
    ev.set()
    src, dst, n = _random_graph()
    with pytest.raises(netsci.Cancelled):
        netsci.rewire(src, dst, n, RNG(0), cancel=ev)


# --- communities ---------------------------------------------------------------------------------------------------------------------
def _dense_modularity(src, dst, syn, n, labels):
    A = np.zeros((n, n))
    np.add.at(A, (src, dst), syn)
    A = A + A.T
    m2 = A.sum()
    k = A.sum(1)
    same = labels[:, None] == labels[None, :]
    return float(((A - np.outer(k, k) / m2) * same).sum() / m2)


def test_modularity_matches_the_textbook_formula():
    src, dst, n = _random_graph(30, 0.2, 4)
    syn = RNG(1).integers(1, 9, len(src)).astype(float)
    labels = RNG(2).integers(0, 4, n)
    assert netsci.modularity(src, dst, syn, n, labels) == pytest.approx(_dense_modularity(src, dst, syn, n, labels))


def test_the_community_finder_recovers_a_planted_partition():
    rng = RNG(0)
    n, k = 160, 4
    truth = np.repeat(np.arange(k), n // k)
    prob = np.where(truth[:, None] == truth[None, :], 0.30, 0.01)
    a = (rng.random((n, n)) < prob) & ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(a)
    labels, q = netsci.louvain(src, dst, np.ones(len(src)), n, RNG(1))
    assert q > 0.5 and netsci.nmi(labels, truth) > 0.9
    assert q == pytest.approx(_dense_modularity(src, dst, np.ones(len(src)), n, labels))
    assert 3 <= len(np.unique(labels)) <= 8


def test_nmi_is_one_for_the_same_partition_and_near_zero_for_unrelated_ones():
    a = np.repeat(np.arange(5), 40)
    assert netsci.nmi(a, a) == pytest.approx(1.0)
    assert netsci.nmi(a, RNG(0).permutation(a)) < 0.1


# --- the whole analysis on the synthetic pack, cached --------------------------------------------------------------------------------
@pytest.fixture
def small(synthetic_pack):
    return dict(nulls=2, wedges=3000, seed=1)


def test_the_analysis_runs_caches_and_never_trusts_a_damaged_cache(small):
    res = netsci.compute("adult", **small)
    assert res["brain"] == "adult" and not res.get("from_cache") and len(res["motifs"]) == 13 and len(res["rich_club"]) >= 3
    assert res["degrees"]["n"] > 500 and res["communities"]["communities"] >= 2 and 0 <= res["reciprocity"]["binary"] <= 1
    assert sum(r["neurons"] for r in res["regions"]) == res["degrees"]["n"]
    again = netsci.compute("adult", **small)
    assert again["from_cache"] is True and again["communities"] == res["communities"]
    path = netsci.cache_path("adult", res["pack_sha256"], res["settings"])
    wrapper = json.loads(path.read_text())
    wrapper["result"]["degrees"]["connections"] += 1                    # one number changed: the checksum must catch it
    path.write_text(json.dumps(wrapper))
    assert netsci.load_cached("adult", res["settings"]) is None
    path.write_text("not json")
    assert netsci.load_cached("adult", res["settings"]) is None
    fresh = netsci.compute("adult", **small)
    assert fresh.get("from_cache") is not True and fresh["degrees"]["connections"] == res["degrees"]["connections"]


def test_other_settings_or_another_pack_are_a_different_cache_entry(small):
    a = netsci.compute("adult", **small)
    b = netsci.compute("adult", nulls=2, wedges=3500, seed=1)
    assert not b.get("from_cache") and netsci.cache_path("adult", a["pack_sha256"], a["settings"]) != netsci.cache_path("adult", a["pack_sha256"], b["settings"])
    assert netsci.cache_path("adult", "0" * 64, a["settings"]) != netsci.cache_path("adult", a["pack_sha256"], a["settings"])
    assert netsci.compute("adult", force=True, **small).get("from_cache") is not True


def test_the_cache_is_off_when_asked_and_leaves_no_file(small, tmp_path):
    from kickthefly.core import paths

    res = netsci.compute("adult", cache=False, **small)
    assert not list((paths.get().data_dir / "cache").glob("netsci_*.json"))
    assert res["settings"]["nulls"] == 2


def test_csv_export_writes_every_table_with_tags_and_the_pack_hash(small, tmp_path):
    res = netsci.compute("adult", **small)
    files = netsci.export_csv(res, tmp_path / "out")
    names = {f.name for f in files}
    assert {"netsci_adult_motifs.csv", "netsci_adult_degree_histogram.csv", "netsci_adult_regions.csv", "netsci_adult_rich_club.csv",
            "netsci_adult_communities.csv", "netsci_adult_reciprocity.csv", "netsci_adult_about.csv"} <= names
    about = (tmp_path / "out" / "netsci_adult_about.csv").read_text()
    assert res["pack_sha256"] in about and "CONNECTOME" in about and "GAME RULE" in about
    assert len((tmp_path / "out" / "netsci_adult_motifs.csv").read_text().splitlines()) == 14


def test_progress_is_reported_and_reaches_one(small):
    seen = []
    netsci.compute("adult", cache=False, progress=lambda f, label: seen.append(f), **small)
    assert seen[0] == 0.0 and seen[-1] == 1.0 and all(b >= a for a, b in zip(seen, seen[1:]))


def test_a_missing_pack_is_a_clear_error(monkeypatch):
    from kickthefly.sim import brainpack

    monkeypatch.setattr(brainpack, "find", lambda brain="adult": None)
    with pytest.raises(FileNotFoundError):
        netsci.load_edges("adult")
