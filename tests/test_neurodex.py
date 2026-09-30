"""Neurodex (3.0): the type table, the discovery rule, saved progress, the curated facts. Runs on the synthetic pack
(tests/synthetic_pack.py), so it needs no connectome; nothing here is a biological result."""
from __future__ import annotations

import json

import numpy as np
import pytest

from kickthefly.core import neurodex as nd


@pytest.fixture
def tab(synthetic_pack):
    t = nd.table("adult")
    assert t is not None
    return t


# --- the type table ------------------------------------------------------------------------------------------------------
def test_table_counts_match_the_pack(tab):
    from kickthefly.sim import brainpack

    z = np.load(brainpack.find("adult"))
    types = z["type"].astype(str)
    assert int(tab.count.sum()) == int((types != "").sum())
    for name in ("DNp01", "LPLC2", "PAM01", "KCg-m"):
        i = tab.index(name)
        assert tab.count[i] == int((types == name).sum())
        assert len(tab.rows(name)) == tab.count[i]
        assert set(types[tab.rows(name)]) == {name}


def test_entry_has_the_dataset_fields_and_tags(tab):
    e = nd.entry(tab, "LPLC2")
    assert e["superclass"] == "visual_projection" and e["region"] == "Optic Lobe"
    assert e["transmitter"] == "acetylcholine" and 0.3 <= e["transmitter_confidence"] <= 1.0
    assert e["tags"] == {"data": nd.CONNECTOME, "discovery": nd.GAME_RULE}
    assert e["inputs"] and e["outputs"]
    assert all(s > 0 and 0 < share <= 1 for _, s, share in e["inputs"] + e["outputs"])
    assert nd.entry(tab, "no such type") is None


def test_partner_counts_are_the_pack_synapse_counts(tab, synthetic_pack):
    """The top partner counts are exactly the summed absolute connection counts between the two types."""
    z = np.load(synthetic_pack)
    types = z["type"].astype(str)
    indptr, indices, data = z["indptr"], z["indices"], z["data"].astype(int)
    post = np.repeat(np.arange(len(types)), np.diff(indptr))
    name, (partner, n_syn, _) = "DNp01", nd.entry(tab, "DNp01")["inputs"][0]
    mask = (types[post] == name) & (types[indices] == partner)
    assert n_syn == int(np.abs(data[mask]).sum())
    name, (partner, n_syn, _) = "DNp01", nd.entry(tab, "DNp01")["outputs"][0]
    mask = (types[indices] == name) & (types[post] == partner)
    assert n_syn == int(np.abs(data[mask]).sum())


def test_chunked_partner_matrix_equals_one_pass(synthetic_pack):
    z = np.load(synthetic_pack)
    types = z["type"].astype(str)
    names, inv = np.unique(types, return_inverse=True)
    tid = inv.astype(np.int32)
    a = nd._partner_matrix(tid, len(names), z["indptr"], z["indices"], z["data"], chunk=1 << 30)
    b = nd._partner_matrix(tid, len(names), z["indptr"], z["indices"], z["data"], chunk=5000)
    assert (a != b).nnz == 0 and a.sum() == b.sum()


def test_region_progress_adds_up(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    prog.mark("adult", "DNp01")
    prog.mark("adult", "KCg-m")
    rows = nd.region_progress(tab, prog)
    allrow = rows[-1]
    assert allrow == ("(all)", 2, len(tab))
    assert sum(r[2] for r in rows[:-1]) == len(tab) and sum(r[1] for r in rows[:-1]) == 2


def test_larva_style_pack_without_transmitters_says_none(tmp_path):
    arrays = dict(type=np.array(["A", "A", "B"]), superclass=np.array(["x", "x", "y"]),
                  region=np.array(["r", "r", "r"]), body_id=np.array([1, 2, 3]),
                  indptr=np.array([0, 1, 2, 3]), indices=np.array([1, 2, 0]), data=np.array([3, -2, 5], np.int16))
    t = nd.build_table(arrays, "larva")
    e = nd.entry(t, "A")
    assert e["transmitter"] is None and e["transmitter_confidence"] is None
    assert nd.fact_for("A", "larva") is None                         # the literature table is about the adult fly
    assert e["outputs"][0][:2] == ("B", 5) and e["outputs"][1][:2] == ("A", 3)
    assert e["inputs"][0][:2] == ("A", 3) and e["inputs"][1][:2] == ("B", 2)


# --- the discovery rule --------------------------------------------------------------------------------------------------
def _rates(tab, n, **hz):
    """Per-neuron rate EMA (spikes per step, dt 5 ms) with the given types firing at the given Hz."""
    r = np.full(n, 1.0 * 0.005)
    for name, h in hz.items():
        r[tab.rows(name)] = h * 0.005
    return r


def test_a_type_is_discovered_only_after_settling_and_a_sustained_rise(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    quiet = _rates(tab, n)
    for _ in range(nd.SETTLE_CHECKS + 20):
        assert tr.observe("fly", quiet, 0.005, calm=True) == []
    loud = _rates(tab, n, DNp01=60.0)
    got = []
    for k in range(nd.DISCOVER_SUSTAIN - 1):                        # one check short of the sustain requirement
        got += tr.observe("fly", loud, 0.005, calm=False)
    assert got == []
    got += tr.observe("fly", loud, 0.005, calm=False)
    assert got == ["DNp01"] and prog.discovered("adult", "DNp01")
    assert tr.observe("fly", loud, 0.005, calm=False) == []          # once only


def test_nothing_is_discovered_while_settling_even_if_loud(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    loud = _rates(tab, len(tab.type_id), DNp01=80.0)
    for _ in range(nd.SETTLE_CHECKS - 1):
        assert tr.observe("fly", loud, 0.005, calm=True) == []
    assert prog.n_discovered("adult") == 0


def test_a_type_that_is_always_loud_is_not_a_discovery(tab, tmp_path):
    """The rule is relative to the type's own calm rate: a type that fires 30 Hz at rest never counts."""
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    always = _rates(tab, len(tab.type_id), ORN_DM1=30.0)
    for _ in range(nd.SETTLE_CHECKS + 50):
        tr.observe("fly", always, 0.005, calm=True)
    assert not prog.discovered("adult", "ORN_DM1")


def test_weak_rise_below_the_absolute_floor_is_not_a_discovery(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    quiet = np.full(n, 0.0)                                          # calm 0 Hz: the calm floor (2 Hz) applies
    for _ in range(nd.SETTLE_CHECKS + 5):
        tr.observe("fly", quiet, 0.005, calm=True)
    faint = _rates(tab, n, DNp01=4.0)                                # 4 Hz is 2x the floor but under DISCOVER_MIN_HZ
    for _ in range(10):
        tr.observe("fly", faint, 0.005, calm=False)
    assert not prog.discovered("adult", "DNp01")


def test_driven_discovery_is_tagged(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    for _ in range(nd.SETTLE_CHECKS + 5):
        tr.observe("fly", _rates(tab, n), 0.005, calm=True)
    for _ in range(nd.DISCOVER_SUSTAIN):
        tr.observe("fly", _rates(tab, n, MDN=50.0), 0.005, calm=False, driven=True)
    assert prog.types("adult")["MDN"]["how"] == "stimulated"


def test_each_brain_has_its_own_calm_baseline(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    for _ in range(nd.SETTLE_CHECKS + 5):
        tr.observe("a", _rates(tab, n), 0.005, calm=True)
    assert "a" in tr._state and "b" not in tr._state
    tr.observe("b", _rates(tab, n, DNp01=90.0), 0.005, calm=False)
    assert tr._state["b"]["n"] == 1 and not prog.discovered("adult", "DNp01")


# --- saved progress ------------------------------------------------------------------------------------------------------
def test_progress_round_trips_and_keeps_the_first_record(tmp_path):
    p = tmp_path / "dex.json"
    a = nd.Progress(p)
    assert a.mark("adult", "DNp01", 7.25, "play", when="2026-01-01T00:00:00")
    assert not a.mark("adult", "DNp01", 99, "stimulated")
    assert a.mark("larva", "X1")
    assert a.save()
    b = nd.Progress(p)
    assert b.types("adult") == {"DNp01": {"t": "2026-01-01T00:00:00", "x": 7.25, "how": "play"}}
    assert b.discovered("larva", "X1") and not b.discovered("adult", "X1")      # the larva has its own dex


def test_corrupt_progress_is_kept_aside_and_starts_empty(tmp_path):
    p = tmp_path / "dex.json"
    p.write_text("{not json", encoding="utf-8")
    prog = nd.Progress(p)
    assert prog.n_discovered("adult") == 0 and prog.warnings
    assert (tmp_path / "dex.json.bad").exists() and not p.exists()


def test_a_newer_progress_file_is_never_overwritten(tmp_path):
    p = tmp_path / "dex.json"
    p.write_text(json.dumps({"version": 99, "brains": {"adult": {"types": {"A": {}}}}}), encoding="utf-8")
    prog = nd.Progress(p)
    prog.mark("adult", "B")
    assert not prog.save()
    assert json.loads(p.read_text())["version"] == 99


def test_progress_ignores_junk_entries(tmp_path):
    p = tmp_path / "dex.json"
    p.write_text(json.dumps({"version": 1, "brains": {"adult": {"types": {"A": {"t": "x", "x": 2, "how": "play"},
                                                                          "B": 5, "C": None}}, "larva": 7}}), encoding="utf-8")
    prog = nd.Progress(p)
    assert list(prog.types("adult")) == ["A"] and prog.n_discovered("larva") == 0


def test_progress_lives_next_to_the_training_memory(isolated_home):
    from kickthefly.core import memory

    assert nd.progress_path().parent == memory.memory_dir()


def test_list_entries_hide_undiscovered_names_from_search(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    prog.mark("adult", "DNp01")
    assert nd.list_entries(tab, prog, query="dnp") == [tab.index("DNp01")]
    assert nd.list_entries(tab, prog, query="MDN") == []             # MDN exists but isn't discovered: search can't spoil it
    assert len(nd.list_entries(tab, prog, only_found=True)) == 1
    assert len(nd.list_entries(tab, prog)) == len(tab)


# --- curated facts -------------------------------------------------------------------------------------------------------
def test_the_shipped_facts_file_is_valid_and_about_thirty_types():
    fs = nd.load_facts()
    assert 25 <= len(fs) <= 40
    for f in fs:
        assert f.cite and f.doi.startswith("10.") and f.checked and len(f.text) <= nd.MAX_FACT_CHARS


def test_every_fact_has_a_checked_doi_shaped_citation_and_no_driver_line_names():
    for f in nd.load_facts():
        assert f.cite.count(",") >= 1 and any(ch.isdigit() for ch in f.cite)
        assert "GAL4" not in f.text and "Gal4" not in f.text and "split" not in f.text.lower()


def test_facts_cover_the_types_the_game_uses():
    need = {"DNp01", "LPLC2", "LC4", "MDN", "MN9", "DNg62", "DNge078", "PAM01", "PPL101", "KCg-m", "MBON01", "pIP10",
            "LC10a", "DNa02", "DNa01", "EPG"}
    missing = {t for t in need if nd.fact_for(t) is None}
    assert not missing, missing


def test_a_type_without_a_curated_fact_has_none(tab):
    assert nd.fact_for("Mi1") is None and nd.fact_for("CB_misc") is None
    assert nd.entry(tab, "Mi1")["curated"] is None
    assert nd.entry(tab, "LPLC2")["curated"]["doi"] == "10.1038/nature24626"


def test_prefix_facts_match_only_what_the_paper_covers():
    assert nd.fact_for("PAM07").id == "pam" and nd.fact_for("PPL103").id == "ppl1"
    assert nd.fact_for("pC1_1a").id == "p1"
    assert nd.fact_for("pC1_4a") is None                             # a pC1 type the game does not count as P1
    assert nd.fact_for("FB6A").id == "dorsal-fb" and nd.fact_for("FB5A") is None


def test_bad_facts_files_are_refused_with_the_entry_named(tmp_path):
    def write(body):
        p = tmp_path / "f.yaml"
        p.write_text(body, encoding="utf-8")
        return p

    with pytest.raises(nd.FactsError, match="needs its citation|required"):
        nd.load_facts(write("facts:\n  - {id: a, types: [X], text: hi}\n"))
    with pytest.raises(nd.FactsError, match="unknown keys"):
        nd.load_facts(write("facts:\n  - {id: a, types: [X], text: t, cite: c, doi: d, checked: k, extra: 1}\n"))
    with pytest.raises(nd.FactsError, match="duplicate"):
        nd.load_facts(write("facts:\n  - {id: a, types: [X], text: t, cite: c, doi: d, checked: k}\n"
                            "  - {id: a, types: [Y], text: t, cite: c, doi: d, checked: k}\n"))
    with pytest.raises(nd.FactsError, match="needs 'types' or 'prefix'"):
        nd.load_facts(write("facts:\n  - {id: a, text: t, cite: c, doi: d, checked: k}\n"))
    with pytest.raises(nd.FactsError, match="max"):
        nd.load_facts(write("facts:\n  - {id: a, types: [X], text: %s, cite: c, doi: d, checked: k}\n" % ("x" * 400)))


def test_skeleton_points_come_only_from_the_cache(tab, tmp_path, monkeypatch):
    from kickthefly.sim import morphology

    monkeypatch.setattr(morphology, "default_cache_dir", lambda: tmp_path)
    assert nd.skeleton_points(tab, "DNp01") is None                   # nothing cached, and nothing fetched
    row = int(tab.rows("DNp01")[0])
    swc = "\n".join(f"{i+1} 0 {i*10} {i*5} {i} 1 {i}" for i in range(1, 30))
    (tmp_path / f"{int(tab.body_id[row])}.swc").write_text(swc, encoding="utf-8")
    pts = nd.skeleton_points(tab, "DNp01", 20)
    assert pts is not None and pts.shape == (20, 3)
