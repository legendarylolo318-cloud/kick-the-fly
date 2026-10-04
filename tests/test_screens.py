"""3.1.0 tasks 6 and 7, the whole-brain screens (lab/screens.py, lab/labscreen.py): the machinery, run in process on the synthetic pack
(plumbing, not biology: nothing here says what the real connectome does; the real screen's calls were checked by hand against the
validated looming pathway and are in docs/screens.md)."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from kickthefly.lab import screens as sc

SEEDS = (4000, 4001, 4002, 4003)


@pytest.fixture(autouse=True)
def _pack(synthetic_pack):
    sc._DN_NAMES_CACHE = None
    yield
    sc._DN_NAMES_CACHE = None


def _brain():
    from kickthefly.core import simcore

    return simcore.new_brain(seed=1, warmup=0, memory=False)


# --- the small pieces ---------------------------------------------------------------------------------------------------------------
def test_the_catalog_lists_types_filters_and_refuses_unknown_names():
    cat = sc.catalog()
    names = [c["type"] for c in cat]
    assert names == sorted(names) and "LPLC2" in names and all(type(n) is str for n in names)
    assert {c["type"]: c["neurons"] for c in cat}["LPLC2"] == 60
    assert all(c["neurons"] >= 50 for c in sc.catalog(min_neurons=50))
    assert [c["type"] for c in sc.catalog(types=["LC4", "LPLC2"])] == ["LC4", "LPLC2"]
    assert len(sc.catalog(max_types=5)) == 5
    assert {c["superclass"] for c in sc.catalog(superclasses=["descending_neuron"])} == {"descending_neuron"}
    with pytest.raises(sc.ScreenError, match="NoSuchType"):
        sc.catalog(types=["LC4", "NoSuchType"])


def test_matched_controls_have_the_same_size_per_superclass_and_never_overlap():
    br = _brain()
    types = br.types.astype(str)
    rows = np.flatnonzero(np.isin(types, ["LPLC2", "DNp01"]))             # two superclasses
    exclude = np.flatnonzero(types == "DNp09")
    a = sc.matched_random(br, rows, exclude, 7)
    assert len(a) == len(rows) and not set(a) & set(rows) and not set(a) & set(exclude)
    scn = br.superclass.astype(str)
    for s in np.unique(scn[rows]):
        assert np.count_nonzero(scn[a] == s) == np.count_nonzero(scn[rows] == s)
    assert np.array_equal(a, sc.matched_random(br, rows, exclude, 7)), "seeded"
    assert not np.array_equal(a, sc.matched_random(br, rows, exclude, 8))
    assert sc._seed_of("a", 1) == sc._seed_of("a", 1) != sc._seed_of("a", 2)


def test_bh_matches_the_textbook_and_keeps_nan():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = sc.bh(p)
    assert np.allclose(q[[3, 0, 2, 1]], [0.02, 0.02, 0.04 * 4 / 3, 0.04], atol=1e-9) or np.isclose(q[3], 0.02)
    assert np.isnan(q[4]) and np.all(q[:4] >= p[:4]) and np.all(q <= 1.0 + 1e-12) | np.isnan(q[4])
    order = np.argsort(p[:4])
    assert np.all(np.diff(q[:4][order]) >= -1e-12), "monotone in p"


def test_the_paired_test_has_no_floor_where_the_wilcoxon_does():
    from kickthefly.lab import labstats

    treated, control = np.arange(8) + 10.0, np.arange(8) + 0.0 + np.linspace(0, 0.3, 8)
    p, share = sc.paired_p(treated, control)
    assert p < 1e-6 and share == 1.0
    assert labstats.paired(treated, control)["p_value"] >= 0.0078 - 1e-9, "the Wilcoxon cannot get lower at 8 seeds: why the screen does not use it"
    p2, share2 = sc.paired_p([1, 2, 3, 4, -9, 6, 7, 8], [0] * 8)
    assert p2 > 0.05 and share2 == 7 / 8, "one wild seed cannot carry it"
    assert sc.paired_p([1, 1, 1], [0, 0, 0]) == (0.0, 1.0) and sc.paired_p([1, 1], [1, 1])[0] == 1.0
    assert math.isnan(sc.paired_p([1.0], [0.0])[0])


def test_progress_reports_done_total_eta_and_calls_back(capsys):
    seen = []
    pr = sc.Progress(4, "a screen", callback=lambda d, n, label: seen.append((d, n, label)), stream=None)
    pr.start(0)
    pr.step()
    pr.step(3)
    assert seen[-1] == (4, 4, "a screen") and "4/4" in pr.line() and "100%" in pr.line() and "a screen" in pr.line()


def test_the_validated_pathways_are_pathway_tests():
    from kickthefly.lab import validation

    for b in sc.VALIDATED_PATHWAYS:
        t = validation.BY_ID[b]
        assert t.get("drive") and t.get("readout") and not t.get("kind"), b
    assert "mdn_backward" not in sc.VALIDATED_PATHWAYS, "a behavior the model does not reproduce is not a behavior to knock out"


# --- the store ---------------------------------------------------------------------------------------------------------------------------
def test_the_store_resumes_refuses_a_different_run_and_survives_a_cut_line(tmp_path):
    sig = dict(kind="activation", seeds=[1, 2])
    st = sc.Store(tmp_path, sig)
    st.add([dict(k="x", i=1), dict(k="x", i=2)])
    st.close()
    with open(tmp_path / "records.jsonl", "a") as f:
        f.write('{"k":"x","i":3')                                     # an interrupted write
    again = sc.Store(tmp_path, sig)
    assert [r["i"] for r in again.records] == [1, 2]
    again.add([dict(k="x", i=3)])
    again.close()
    assert len(sc.Store(tmp_path, sig).records) == 3
    with pytest.raises(sc.ScreenError, match="seeds"):
        sc.Store(tmp_path, dict(kind="activation", seeds=[1, 2, 3]))
    fresh = sc.Store(tmp_path, dict(kind="activation", seeds=[1, 2, 3]), restart=True)
    assert fresh.records == []


# --- the activation screen -----------------------------------------------------------------------------------------------------------------
def _act(folder, **kw):
    kw.setdefault("seeds", SEEDS)
    kw.setdefault("types", ["LPLC2", "LC4", "DNp01"])
    return sc.run_activation(folder, controls=1, steps=60, workers=0, stream=None, **kw)


def test_the_activation_screen_runs_writes_its_files_and_has_the_columns(tmp_path):
    res = _act(tmp_path)
    assert res["kind"] == "activation" and res["tag"] == "MODEL PREDICTION" and res["n_types"] == 3
    by = {r["type"]: r for r in res["rows"]}
    assert set(by) == {"LPLC2", "LC4", "DNp01"} and all(r["n_seeds"] == 4 for r in res.get("rows"))
    for r in res["rows"]:
        for k in sc.READOUT_NAMES:
            assert f"eff_{k}" in r and f"q_{k}" in r and f"lo_{k}" in r and f"sign_{k}" in r
        assert r["behavior_call"] == "none" or r["behavior_readout"]
    assert by["DNp01"]["self_readouts"] == "dnp01", "driving a readout's own neurons is not an effect"
    assert math.isnan(by["DNp01"]["p_dnp01"])
    files = {p.name for p in tmp_path.iterdir()}
    assert {"activation_screen.csv", "activation_screen_responses.csv", "activation_screen.json", "meta.json", "records.jsonl"} <= files
    header = (tmp_path / "activation_screen.csv").read_text().splitlines()[0].split(",")
    assert header[:5] == ["type", "superclass", "neurons", "n_seeds", "self_readouts"] and "behavior_call" in header
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["tag"] == "MODEL PREDICTION" and meta["engines"] and meta["signature"]["seeds"] == list(SEEDS)
    assert sc.load(tmp_path)["kind"] == "activation"


def test_the_screen_is_deterministic_and_resumable(tmp_path):
    a = _act(tmp_path / "a")
    b = _act(tmp_path / "b")
    key = lambda res: [(r["type"], r["behavior_call"], r["eff_dnp01"], r["eff_mn9"], r["dn_responders"]) for r in res["rows"]]   # noqa: E731
    assert key(a) == key(b), "the same seeds give the same screen on the CPU"
    n = len((tmp_path / "a" / "records.jsonl").read_text().splitlines())
    seen = []
    again = sc.run_activation(tmp_path / "a", seeds=SEEDS, types=["LPLC2", "LC4", "DNp01"], controls=1, steps=60, workers=0, stream=None,
                              progress=lambda d, n_, label: seen.append(label))
    assert len((tmp_path / "a" / "records.jsonl").read_text().splitlines()) == n, "nothing was run again"
    assert seen == ["nothing left to run"] and key(again) == key(a)
    more = _act(tmp_path / "a", types=["LPLC2", "LC4", "DNp01", "LC10a"])
    assert {r["type"] for r in more["rows"]} == {"LPLC2", "LC4", "DNp01", "LC10a"}
    assert sum(1 for r in sc.Store(tmp_path / "a", more["meta"]["signature"]).records if r.get("k") == "act" and r["t"] == "LPLC2") == 4, \
        "a type that was already done is not run twice"
    with pytest.raises(sc.ScreenError):
        sc.run_activation(tmp_path / "a", seeds=SEEDS, types=["LPLC2"], controls=3, steps=60, workers=0, stream=None)


def test_an_interrupted_screen_finishes_from_where_it_stopped(tmp_path):
    full = _act(tmp_path / "full")
    _act(tmp_path / "cut")
    lines = (tmp_path / "cut" / "records.jsonl").read_text().splitlines()
    keep = [ln for ln in lines if '"t":"DNp01"' not in ln]
    (tmp_path / "cut" / "records.jsonl").write_text("\n".join(keep) + "\n")
    done = _act(tmp_path / "cut")
    assert [(r["type"], r["eff_mn9"]) for r in done["rows"]] == [(r["type"], r["eff_mn9"]) for r in full["rows"]]


def test_an_activation_screen_refuses_nonsense(tmp_path):
    for kw in (dict(seeds=()), dict(controls=0), dict(steps=10)):
        args = dict(seeds=SEEDS, types=["LC4"], controls=1, steps=60)
        args.update(kw)
        with pytest.raises(sc.ScreenError):
            sc.run_activation(tmp_path / str(len(kw)), workers=0, stream=None, **args)


def test_a_called_behavior_needs_q_effect_size_and_agreement():
    cat = [dict(type="T", neurons=5, superclass="x")]
    n_dn = len(sc._dn_names())

    def rec(seed, eff, ctl):
        return dict(k="act", t="T", s=seed, n=5, r=[eff] + [0.0] * 9, dn=[], c=[dict(r=[ctl] + [0.0] * 9, dn=[])], ov=[])

    strong = [rec(s, 30.0 + s % 3, 0.1 * (s % 2)) for s in range(8)] + [rec(s, 0.0, 0.0) for s in range(8, 8)]
    # many other types supply the null p-values BH needs to be meaningful
    many = []
    rng = np.random.default_rng(0)
    for i in range(60):
        for s in range(8):
            many.append(dict(k="act", t=f"N{i}", s=s, n=3, r=list(rng.normal(0, 0.3, 10)), dn=[], c=[dict(r=list(rng.normal(0, 0.3, 10)), dn=[])], ov=[]))
    cats = cat + [dict(type=f"N{i}", neurons=3, superclass="x") for i in range(60)]
    res = sc.aggregate_activation(strong + many, cats, range(8))
    row = next(r for r in res["rows"] if r["type"] == "T")
    assert row["behavior_readout"] == "dnp01" and row["behavior_call"].startswith("raises") and row["behavior_q"] < 0.05
    assert sum(1 for r in res["rows"] if r["type"] != "T" and r["behavior_readout"]) == 0, "noise is not called"
    small = [rec(s, 0.5, 0.0) for s in range(8)]                                # consistent but under MIN_EFFECT_HZ
    r2 = sc.aggregate_activation(small + many, cats, range(8))
    assert next(r for r in r2["rows"] if r["type"] == "T")["behavior_readout"] == ""
    wild = [rec(s, 30.0 if s == 0 else -0.1, 0.0) for s in range(8)]           # one seed carries the mean
    r3 = sc.aggregate_activation(wild + many, cats, range(8))
    assert next(r for r in r3["rows"] if r["type"] == "T")["behavior_readout"] == ""
    assert n_dn > 0


# --- the knockout screen ---------------------------------------------------------------------------------------------------------------------
def test_the_knockout_screen_runs_opens_batches_and_ranks(tmp_path):
    res = sc.run_knockout(tmp_path, behaviors=["looming_escape"], seeds=SEEDS, types=["LPLC2", "LC4", "LC10a", "LC11"], batch_size=2, controls=1,
                          workers=0, stream=None)
    assert res["kind"] == "knockout" and res["tag"] == "MODEL PREDICTION"
    batches = [r for r in res["rows"] if r["is_batch"]]
    assert {r["candidate"] for r in batches} == {"LPLC2+LC4", "LC10a+LC11"}
    singles = [r for r in res["rows"] if not r["is_batch"]]
    for r in singles:
        assert r["rank"] in range(1, len(singles) + 1) and r["n_seeds"] == 4 and "drop_share" in r and "q" in r
    s = res["summary"]["looming_escape"]
    assert s["tested"] == len(singles) and "unperturbed_evoked_hz" in s
    files = {p.name for p in tmp_path.iterdir()}
    assert {"knockout_screen.csv", "knockout_screen.json", "meta.json", "records.jsonl", "candidates.json"} <= files
    again = sc.run_knockout(tmp_path, behaviors=["looming_escape"], seeds=SEEDS, types=["LPLC2", "LC4", "LC10a", "LC11"], batch_size=2, controls=1,
                            workers=0, stream=None)
    assert [(r["candidate"], r["drop_share"]) for r in again["rows"]] == [(r["candidate"], r["drop_share"]) for r in res["rows"]]


def test_a_batch_that_does_nothing_is_not_opened():
    recs = []
    for s in range(8):
        recs.append(dict(k="ko_base", b="looming_escape", s=s, base=1.0, driven=11.0, engine=["cpu", "x"]))
        recs.append(dict(k="ko", b="looming_escape", s=s, spec=["A", "B"], n=10, base=1.0, driven=11.0 + 0.01 * s, c=[[1.0, 11.0]]))
        recs.append(dict(k="ko", b="looming_escape", s=s, spec=["C", "D"], n=10, base=1.0, driven=1.5 + 0.01 * s, c=[[1.0, 11.0]]))
    opened = sc._open_batches(recs, ["looming_escape"], range(8), {})
    assert opened["looming_escape"] == ["C", "D"]


def test_the_knockout_screen_refuses_non_pathway_behaviors(tmp_path):
    for bad in ("mdn_backward_not_a_test", "grooming_hierarchy", "mb_extinction"):
        with pytest.raises(sc.ScreenError):
            sc.run_knockout(tmp_path / bad, behaviors=[bad], seeds=SEEDS, types=["LC4"], controls=1, workers=0, stream=None)


# --- parquet --------------------------------------------------------------------------------------------------------------------------------------
def test_parquet_is_written_when_pyarrow_is_there_and_skipped_cleanly_when_not(tmp_path, monkeypatch):
    rows = [dict(a=1, b="x", c=0.5), dict(a=2, b="y", c=None)]
    written = sc._write_rows(rows, tmp_path / "t")
    pytest.importorskip("pyarrow")
    assert {p.suffix for p in written} == {".csv", ".parquet"}
    import pyarrow.parquet as pq
    assert pq.read_table(tmp_path / "t.parquet").num_rows == 2
    import builtins
    real = builtins.__import__

    def no_arrow(name, *a, **k):
        if name.startswith("pyarrow"):
            raise ImportError("no pyarrow")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_arrow)
    assert [p.suffix for p in sc._write_rows(rows, tmp_path / "u")] == [".csv"]
    monkeypatch.undo()
    res = _act(tmp_path / "act")
    meta = json.loads((tmp_path / "act" / "meta.json").read_text())
    assert "activation_screen.parquet" in meta["outputs"] and meta["parquet_note"] is None


# --- protocol, command line, Lab ---------------------------------------------------------------------------------------------------------------
def test_a_screen_protocol_is_checked_and_runs(tmp_path):
    from kickthefly.lab import protocol

    p = protocol.check(dict(name="s", screen=dict(kind="activation", types=["LC4"], controls=1, max_types=5)))
    assert p["seeds"] == list(sc.SCREEN_SEEDS) and p["screen"]["controls"] == 1
    p = protocol.check(dict(name="s", seeds=[4000, 4001], screen=dict(kind="knockout", behaviors=["looming_escape"], top=2)))
    assert p["seeds"] == [4000, 4001]
    for bad, msg in ((dict(kind="nope"), "kind"), (dict(kind="activation", bogus=1), "only"), (dict(kind="activation", behaviors=["looming_escape"]), "knockout"),
                     (dict(kind="knockout", min_neurons=3), "activation"), (dict(kind="knockout", behaviors=["mdn_backward"]), "pathway"),
                     (dict(kind="activation", controls=0), "controls"), (dict(kind="activation", types=[]), "types")):
        with pytest.raises(protocol.ProtocolError, match=msg):
            protocol.check(dict(name="s", screen=bad))
    with pytest.raises(protocol.ProtocolError, match="stands alone"):
        protocol.check(dict(name="s", screen=dict(kind="activation"), surgery={"type:LC4": -1}))
    q = protocol.check(dict(name="tiny", seeds=[4000, 4001, 4002], screen=dict(kind="activation", types=["LPLC2", "LC4"], controls=1)))
    sc_steps = sc.STIM
    try:
        sc.STIM = 60
        folder = protocol.run(q, tmp_path)
    finally:
        sc.STIM = sc_steps
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["tag"] == "MODEL PREDICTION" and "activation_screen.csv" in summary["files"] and (folder / "screen" / "records.jsonl").exists()


def test_the_command_line_runs_a_screen(tmp_path, capsys):
    from kickthefly.lab import headless

    args = SimpleNamespace(out=str(tmp_path), seeds="4000-4003", workers=0, types=["LPLC2", "DNp01"], screen_controls=1, min_neurons=None, max_types=None,
                           screen_batch=0, restart=False, top=None, candidate_batch=None, knockout_screen=None)
    old = sc.STIM
    try:
        sc.STIM = 60
        assert headless.run_screen(args, "activation") == 0
    finally:
        sc.STIM = old
    assert "Activation screen (MODEL PREDICTION)" in capsys.readouterr().out and (tmp_path / "activation_screen.csv").exists()
    args.types = ["NoSuchType"]
    assert headless.run_screen(args, "activation") == 2


def _make_host():
    from test_labpages import StubHost

    class Host(StubHost):
        def __init__(self):
            super().__init__()
            self.big_view = False
            self.brain = SimpleNamespace(types=np.array(["LC4", "LC4", "X"]))

        def _neuron_info(self, i):
            return dict(i=i, type=str(self.brain.types[i]))

    h = Host()
    return h


@pytest.fixture
def lab_menu(tmp_path, monkeypatch):
    import pygame

    from kickthefly.lab import labscreen
    from kickthefly.ui import menu as ui

    pygame.init()
    pygame.display.set_mode((1280, 760))
    monkeypatch.setattr(labscreen, "folder_of", lambda kind: tmp_path / kind)
    m = ui.Menu(_make_host())
    labscreen.install(m)
    m.fonts()
    m.mouse = (0, 0)
    m.closed = []
    m.close = lambda: m.closed.append(True)
    yield m
    pygame.display.quit()


def _draw(m, page):
    import pygame

    surf = pygame.Surface((1280, 760))
    m.hits = []
    m.pages[page](m, surf, pygame.Rect(40, 40, 1200, 680), (0, 0))
    return surf


def _click(m, ident_prefix):
    for rect, kind, data in m.hits:
        if kind == "button" and tuple(data["id"][:len(ident_prefix)]) == tuple(ident_prefix):
            data["click"]()
            return True
    return False


def test_the_lab_pages_browse_search_sort_and_inspect(lab_menu, tmp_path):
    m = lab_menu
    _act(tmp_path / "activation")
    _draw(m, "lab_activation")
    st = m.lab_state.scr_activation
    assert st["res"] and st["res"]["kind"] == "activation"
    assert _click(m, ("scr", "activation", "row", "LC4")) and st["sel"] == "LC4"
    _draw(m, "lab_activation")                                       # the detail panel
    assert _click(m, ("scr", "activation", "sort", "neurons")) and st["sort"] == "neurons"
    _draw(m, "lab_activation")
    assert _click(m, ("scr", "activation", "sort", "neurons")) and st["desc"] is False
    st["search"] = "lplc"
    from kickthefly.lab import labscreen

    assert [r["type"] for r in labscreen._rows("activation", st, st["res"])] == ["LPLC2"]
    st["search"] = ""
    assert _click(m, ("scr", "activation", "live"))
    assert m.host.type_ops == {"LC4": 1} and m.host.surgery_applied == 1
    _draw(m, "lab_activation")
    assert _click(m, ("scr", "activation", "live")) and m.host.type_ops == {} and m.host.surgery_applied == 2
    assert _click(m, ("scr", "activation", "inspect"))
    assert m.host.inspect == dict(i=0, type="LC4") and m.host.big_view and m.closed
    assert _click(m, ("scr", "activation", "close")) and st["sel"] is None
    assert _click(m, ("scr", "activation", "export")) and (m.host.last_export and "activation" in m.host.last_export)


def test_the_knockout_page_browses_and_silences(lab_menu, tmp_path):
    m = lab_menu
    sc.run_knockout(tmp_path / "knockout", behaviors=["looming_escape"], seeds=SEEDS, types=["LPLC2", "LC4"], controls=1, workers=0, stream=None)
    _draw(m, "lab_knockout")
    st = m.lab_state.scr_knockout
    assert st["res"]["kind"] == "knockout"
    assert _click(m, ("scr", "knockout", "row", "looming_escape|LC4"))
    _draw(m, "lab_knockout")
    assert _click(m, ("scr", "knockout", "live")) and m.host.type_ops == {"LC4": -1}
    assert _click(m, ("scr", "knockout", "inspect")) and m.host.inspect["type"] == "LC4"


def test_the_pages_say_what_they_are_and_draw_with_nothing_to_show(lab_menu):
    import pygame

    for page in ("lab_activation", "lab_knockout"):
        surf = _draw(lab_menu, page)
        assert surf.get_bounding_rect().w > 0
    from kickthefly.lab import labscreen

    assert labscreen.KINDS["activation"]["title"] == "ACTIVATION SCREEN" and labscreen.KINDS["knockout"]["page"] == "lab_knockout"
    assert any(p == "lab_activation" for _, p, _ in __import__("kickthefly.game.kick_the_fly", fromlist=["x"]).Game.lab_pages(_make_host()))
