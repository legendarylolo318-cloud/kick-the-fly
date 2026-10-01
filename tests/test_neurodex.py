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
def _rates(tab, n, base_hz: float = 0.0, **hz):
    """The spikes of one 150 ms window (neuron indices, one per spike) with the given types firing at the given Hz and every
    other neuron at base_hz. (Named _rates for history: Day 1's rule read rates; the review's reads counted spikes.)"""
    per = np.full(n, int(round(base_hz * nd.WINDOW_STEPS * 0.005)), np.int64)
    for name, h in hz.items():
        per[tab.rows(name)] = int(round(h * nd.WINDOW_STEPS * 0.005))
    return np.repeat(np.arange(n), per)


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
    always = _rates(tab, len(tab.type_id), ORN_DM1=40.0)
    for _ in range(nd.SETTLE_CHECKS + 50):
        tr.observe("fly", always, 0.005, calm=True)
    assert not prog.discovered("adult", "ORN_DM1")


def test_weak_rise_below_the_absolute_floor_is_not_a_discovery(tab, tmp_path):
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    quiet = _rates(tab, n)                                           # calm 0 Hz: the calm floor (2 Hz) applies
    for _ in range(nd.SETTLE_CHECKS + 5):
        tr.observe("fly", quiet, 0.005, calm=True)
    faint = _rates(tab, n, DNp01=40.0 / 9)                           # ~4.4 Hz: over 2x the floor but under DISCOVER_MIN_HZ
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


# --- the self-test ---------------------------------------------------------------------------------------------------------------
def test_selftest_checks_the_neurodex_facts(isolated_home, monkeypatch):
    from kickthefly.core import selftest

    c = selftest.check_neurodex()
    assert c.status == selftest.PASS and "curated facts" in c.detail
    monkeypatch.setattr(nd, "load_facts", lambda *a: (_ for _ in ()).throw(nd.FactsError("bad file")))
    c = selftest.check_neurodex()
    assert c.status == selftest.WARN and "bad file" in c.detail and c.fix


def test_selftest_warns_about_a_corrupt_progress_file(isolated_home):
    from kickthefly.core import selftest

    p = nd.progress_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{oops", encoding="utf-8")
    c = selftest.check_neurodex()
    assert c.status == selftest.WARN and "neurodex.json.bad" in c.fix


# --- the larva dex (its pack is built locally from Winding et al. 2023; these tests use it only when it is there) ------------------
def test_larval_transmitters_are_the_games_rule_so_no_transmitter_is_shown():
    arrays = dict(type=np.array(["KC", "KC", "LN"]), superclass=np.array(["mushroom_body"] * 2 + ["local_interneuron"]),
                  region=np.array(["Mushroom Body", "Mushroom Body", "Antennal Lobe"]), nt=np.array(["acetylcholine", "acetylcholine", "GABA"]),
                  nt_conf=np.array([0.8, 0.8, 0.8], np.float32), nt_source=np.array(["inferred"] * 3))
    t = nd.build_table(arrays, "larva", partners=False)
    for name in ("KC", "LN"):
        e = nd.entry(t, name)
        assert e["transmitter"] is None and e["transmitter_confidence"] is None
    # the same arrays as an adult pack's own calls would show
    arrays["nt_source"] = np.array(["predicted_nt"] * 3)
    assert nd.entry(nd.build_table(arrays, "adult", partners=False), "LN")["transmitter"] == "GABA"


def test_unassigned_neurons_are_not_a_larval_type():
    arrays = dict(type=np.array(["KC", "unassigned", "unassigned"]), superclass=np.array(["a", "b", "b"]))
    assert list(nd.build_table(arrays, "larva", partners=False).names) == ["KC"]
    assert list(nd.build_table(arrays, "adult", partners=False).names) == ["KC", "unassigned"]       # only the larva's label is special


def test_the_real_larva_pack_gives_its_own_dex():
    from kickthefly.sim import brainpack

    p = brainpack.find(brain="larva")
    if p is None:
        pytest.skip("the larva pack is built locally (docs/larva.md)")
    t = nd.table_from_pack(p, "larva")
    assert t.brain == "larva" and 10 <= len(t) <= 20 and "unassigned" not in set(map(str, t.names))
    assert {"KC", "MBON", "sensory"} <= set(map(str, t.names))
    assert all(nd.entry(t, str(n))["transmitter"] is None for n in t.names)
    assert all(nd.fact_for(str(n), "larva") is None for n in t.names)
    kc = nd.entry(t, "KC")
    assert kc["count"] == 144 and kc["superclass"] == "mushroom_body" and kc["inputs"] and kc["outputs"]


def test_larva_dex_discovers_by_stimulation_on_the_real_larval_brain(tmp_path, monkeypatch):
    from kickthefly.sim import brainpack

    if brainpack.find(brain="larva") is None:
        pytest.skip("the larva pack is built locally (docs/larva.md)")
    from kickthefly.core import simcore

    simcore.pack.cache_clear()
    try:
        br = simcore.new_brain(seed=11, brain="larva", memory=False, warmup=600)
        tab = nd.table_from_pack(brainpack.find(brain="larva"), "larva")
        prog = nd.Progress(tmp_path / "dex.json")
        tr = nd.Tracker(tab, prog)
        for i in range(1300):
            br._step()
            if i % 10 == 0:
                tr.observe("l", nd.window_spikes(br.sim.activity)[0], br.dt, True)
        assert prog.n_discovered("larva") == 0
        rows = tab.rows("KC")
        simcore.drive(br, rows, 0.5)
        for i in range(600):
            br._step()
            if i % 10 == 0:
                tr.observe("l", nd.window_spikes(br.sim.activity)[0], br.dt, False, True)
        assert prog.types("larva").get("KC", {}).get("how") == "stimulated"
        assert prog.n_discovered("adult") == 0                       # the adult list is separate
    finally:
        simcore.pack.cache_clear()


def test_every_curated_fact_matches_a_type_in_the_real_adult_pack():
    """Runs only where the real adult brain pack is built: the curated names must exist in the dataset's own type list, or the
    fact silently never shows. (The synthetic pack can't answer this.)"""
    from conftest import brain_pack
    from kickthefly.sim import brainpack

    if brain_pack() is None:
        pytest.skip("the adult brain pack is not built here")
    tab = nd.table_from_pack(brainpack.find("adult"), "adult")
    names = {str(n) for n in tab.names}
    unmatched = []
    for f in nd.facts():
        if not any(f.matches(n) for n in names):
            unmatched.append(f.id)                                     # review: JO-C/JO-E were exact names that match nothing
        for t in f.types:
            if t not in names:                                         # review: E-PG was listed, the pack only has EPG
                unmatched.append(f"{f.id}:{t}")
        for pre in f.prefix:
            if not any(n.startswith(pre) for n in names):
                unmatched.append(f"{f.id}:{pre}*")
    assert not unmatched, f"curated facts whose types are not in the pack: {unmatched}"


def test_a_fact_cites_the_paper_its_claim_was_checked_in():
    """Review: MBON14's 2-hour appetitive memory statement is in Aso et al. 2014 e04580, not e04577 (which Day 1 cited)."""
    f = next(f for f in nd.load_facts() if f.id == "mbon14")
    assert "2-hour" in f.text and f.doi == "10.7554/eLife.04580" and "e04580" in f.cite and "e04577" in f.cite
    jo = next(f for f in nd.load_facts() if f.id == "johnston-wind")
    assert jo.prefix == ("JO-C", "JO-E") and not jo.types
    assert nd.fact_for("JO-CM").id == "johnston-wind" and nd.fact_for("JO-EV3").id == "johnston-wind"
    assert nd.fact_for("JO-A1") is None and nd.fact_for("JO-B2") is None
    assert nd.fact_for("EPG").id == "epg" and nd.fact_for("EPGt") is None


def test_one_bad_value_in_the_progress_file_does_not_lose_the_rest(tmp_path):
    """Review: a non-numeric "x" (hand-edited, or a future format) crashed Progress() and so the Neurodex."""
    p = tmp_path / "dex.json"
    p.write_text(json.dumps({"version": 1, "brains": {"adult": {"types": {
        "A": {"t": "2026-01-01", "x": "abc", "how": "play"}, "B": {"x": None}, "C": {"x": 4.5, "how": "stimulated"},
        "D": {"x": float("nan"), "how": "<script>"}}}}}), encoding="utf-8")
    prog = nd.Progress(p)
    t = prog.types("adult")
    assert set(t) == {"A", "B", "C", "D"} and t["A"]["x"] == 0.0 and t["B"]["x"] == 0.0 and t["C"]["x"] == 4.5
    assert t["C"]["how"] == "stimulated" and t["D"]["how"] == "play" and t["D"]["x"] == 0.0 and p.exists()



# --- review (Day 1): the Poisson condition ------------------------------------------------------------------------------------------
def test_a_tiny_type_needs_more_evidence_than_a_big_one(tab, tmp_path):
    """The real pack's false discoveries were types of 1-4 neurons: 7 spikes from 2 neurons in 150 ms is >6 Hz and >3x
    the floor, but a plausible chance event at 2 Hz; the same rate from 40 neurons is not."""
    from scipy.stats import poisson

    lam_small = nd.DISCOVER_CALM_FLOOR_HZ * 2 * nd.WINDOW_STEPS * 0.005
    assert poisson.sf(7 - 1, lam_small) > nd.DISCOVER_ALPHA
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    for _ in range(nd.SETTLE_CHECKS + 5):
        tr.observe("f", _rates(tab, n), 0.005, calm=True)
    small = np.repeat(tab.rows("DNp01"), [4, 3])                     # 7 spikes, 2 neurons: 23 Hz mean
    big = np.repeat(tab.rows("ORN_DM1"), 4)                          # 30 neurons x 4 spikes: 27 Hz mean
    for _ in range(nd.DISCOVER_SUSTAIN + 2):
        tr.observe("f", np.concatenate([small, big]), 0.005, calm=False)
    assert not prog.discovered("adult", "DNp01") and prog.discovered("adult", "ORN_DM1")


def test_alpha_comes_from_the_stated_budget():
    assert nd.DISCOVER_ALPHA == pytest.approx(1 / (11_751 * 72_000 * nd.FALSE_ALARM_HOURS))


def test_a_driven_single_neuron_can_still_be_discovered():
    """A neuron driven every refractory cycle fires 10 times in 150 ms; at the 2 Hz floor that is far below alpha."""
    from scipy.stats import poisson

    assert poisson.sf(10 - 1, nd.DISCOVER_CALM_FLOOR_HZ * nd.WINDOW_STEPS * 0.005) < nd.DISCOVER_ALPHA


def test_window_spikes_reads_the_last_steps_of_the_raster():
    from kickthefly.sim.connectome.sim import ActivityBuffer

    a = ActivityBuffer(5)
    for k in range(40):
        s = np.zeros(5, bool)
        s[k % 5] = True
        a.push(s)
    idx, steps = nd.window_spikes(a, 30)
    assert steps == 30 and len(idx) == 30 and set(idx) == {0, 1, 2, 3, 4}
    empty, one = nd.window_spikes(ActivityBuffer(3))
    assert len(empty) == 0 and one == 1


# --- 3.0 Day 2 decision: a calm, untouched fly discovers nothing -----------------------------------------------------------------
def test_a_calm_fly_discovers_nothing_however_loud_a_type_gets(tab, tmp_path):
    """A spontaneous burst with nothing touching the fly (calm, not driven) is never a discovery, however long it lasts; the
    same burst counts once the fly is touched (play) and the type is still there to discover."""
    prog = nd.Progress(tmp_path / "dex.json")
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    for _ in range(nd.SETTLE_CHECKS + 20):
        tr.observe("fly", _rates(tab, n), 0.005, calm=True)
    loud = _rates(tab, n, DNp01=80.0)
    for _ in range(nd.DISCOVER_SUSTAIN * 3):               # short enough that the calm baseline has not absorbed the burst
        assert tr.observe("fly", loud, 0.005, calm=True) == []
    assert prog.n_discovered("adult") == 0
    got = []
    for _ in range(nd.DISCOVER_SUSTAIN):
        got += tr.observe("fly", loud, 0.005, calm=False)
    assert got == ["DNp01"] and prog.types("adult")["DNp01"]["how"] == "play"


def test_driven_counts_even_when_the_fly_is_calm_and_old_rest_entries_survive(tab, tmp_path):
    """Stimulation counts regardless of the calm flag (the API drives a calm fly), and a progress file written by a Day 1 build
    with an 'at rest' entry loads, keeps its tag and is not rediscovered."""
    path = tmp_path / "dex.json"
    old = nd.Progress(path)
    assert old.mark("adult", "ORN_VM5v", 4.0, "rest") and old.save()
    prog = nd.Progress(path)
    assert prog.types("adult")["ORN_VM5v"]["how"] == "rest"
    tr = nd.Tracker(tab, prog)
    n = len(tab.type_id)
    for _ in range(nd.SETTLE_CHECKS + 20):
        tr.observe("fly", _rates(tab, n), 0.005, calm=True)
    got = []
    for _ in range(nd.DISCOVER_SUSTAIN):
        got += tr.observe("fly", _rates(tab, n, MDN=50.0), 0.005, calm=True, driven=True)
    assert got == ["MDN"] and prog.types("adult")["MDN"]["how"] == "stimulated"
    assert prog.types("adult")["ORN_VM5v"]["how"] == "rest"
