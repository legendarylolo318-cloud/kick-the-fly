"""3.1.0 task 8, the failure autopsy (lab/failure_autopsy.py, lab/labautopsy.py): READ-ONLY, so the first test is that it changes nothing. The
machinery runs on the synthetic pack and on hand-made matrices with known answers (plumbing, not biology)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from kickthefly.lab import failure_autopsy as fa


def _W(entries, n):
    """entries: [(post, pre, weight)] -> W[post, pre]."""
    r, c, v = zip(*entries)
    return sp.csr_array((v, (r, c)), shape=(n, n))


# --- the arithmetic ---------------------------------------------------------------------------------------------------------------------
def test_the_layer_budget_splits_walks_by_the_sign_of_the_walk():
    # 0 -> 2 directly (+0.2); 0 -> 1 (+0.5) then 1 -> 2 (-0.4): an inhibitory two-synapse walk; 0 -> 1 -> 3 (-0.3) -> 2 (-0.5): even, excitatory
    W = _W([(2, 0, 0.2), (1, 0, 0.5), (2, 1, -0.4), (3, 1, -0.3), (2, 3, -0.5)], 4)
    b = fa.layer_budget(W, np.array([0]), np.array([2]))
    assert b[0]["excitatory"] == pytest.approx(0.2) and b[0]["inhibitory"] == 0.0
    assert b[1]["inhibitory"] == pytest.approx(0.5 * 0.4) and b[1]["excitatory"] == 0.0
    assert b[2]["excitatory"] == pytest.approx(0.5 * 0.3 * 0.5) and b[2]["inhibitory"] == 0.0
    assert b[1]["net"] == pytest.approx(-0.2) and b[1]["inhibitory_share"] == pytest.approx(1.0)
    assert [x["hop"] for x in b] == [1, 2, 3]


def test_routes_rank_cell_type_chains_and_count_the_synapses():
    W = _W([(2, 0, 0.2), (1, 0, 0.5), (2, 1, -0.4), (3, 1, -0.3), (2, 3, -0.5)], 4)
    counts = sp.csr_array(W * 10)
    types = np.array(["in", "mid", "out", "late"])
    rt = fa.routes(W, counts, types, np.array([0]), np.array([2]))
    chains = {tuple(r["types"]): r for r in rt}
    assert () in chains and ("mid",) in chains and ("mid", "late") in chains
    assert chains[("mid",)]["inhibitory"] == pytest.approx(0.2) and chains[("mid",)]["net_sign"] == -1
    assert chains[()]["net_sign"] == 1 and chains[()]["edges"][0]["synapses"] == 2 and chains[()]["edges"][0]["excitatory"] == 2
    e = chains[("mid",)]["edges"]
    assert [x["synapses"] for x in e] == [5, 4] and e[1]["inhibitory"] == 4 and e[1]["excitatory"] == 0
    assert chains[("mid", "late")]["excitatory"] == pytest.approx(0.5 * 0.3 * 0.5)
    assert rt == sorted(rt, key=lambda r: -r["share"]) or len({r["hops"] for r in rt}) > 1, "strongest first, with every length represented"


def test_raw_counts_undo_the_normalization():
    counts = np.array([[0, 3, -2], [4, 0, 0], [0, -1, 0]], float)
    inv = 1.0 / np.abs(counts).sum(axis=1)                               # not the pack's definition: a row-normalized toy
    W = sp.csr_array(counts * inv[:, None])
    assert np.allclose(fa.raw_counts(W, inv).toarray(), counts)


def test_raw_counts_of_the_pack_are_the_integer_synapse_counts(synthetic_pack):
    from kickthefly.core import simcore
    from kickthefly.sim import brainpack

    _, W, _ = simcore.pack()
    z = np.load(brainpack.find("adult"))
    c = fa.raw_counts(W, z["inv"])
    assert np.allclose(c.data, np.round(c.data), atol=1e-2) and np.allclose(c.data, z["data"], atol=1e-2)


def test_the_verdict_names_where_the_ratio_falls_below_the_threshold():
    rt = dict(types=["a", "b"])
    rates = {k: dict(ratio=v) for k, v in {"__input__": 30.0, "a": 4.0, "b": 0.9, "__target__": 0.95}.items()}
    v = fa.verdict(rt, rates)
    assert v["stage"] == "a -> b" and "lost between a (x4.00) and b (x0.90)" in v["text"] and [x["ratio"] for x in v["ratios"]] == [30.0, 4.0, 0.9, 0.95]
    never = {k: dict(ratio=v) for k, v in {"__input__": 1.2, "a": 1.0, "b": 1.0, "__target__": 1.0}.items()}
    assert fa.verdict(rt, never)["stage"] == "input" and "never lifted" in fa.verdict(rt, never)["text"]
    fine = {k: dict(ratio=v) for k, v in {"__input__": 9.0, "a": 4.0, "b": 3.0, "__target__": 2.0}.items()}
    assert fa.verdict(rt, fine)["stage"] is None and "met 1.5x" in fa.verdict(rt, fine)["text"]
    assert fa.verdict(None, {})["stage"] is None and "nothing can carry" in fa.verdict(None, {})["text"]


def test_percentages_keep_digits_for_tiny_shares():
    assert fa.pct(0.0) == "0%" and fa.pct(0.0031) == "0.31%" and fa.pct(0.000012) == "0.0012%" and fa.pct(float("nan")) == "n/a"


# --- one autopsy, read-only --------------------------------------------------------------------------------------------------------------------
def test_an_autopsy_runs_and_leaves_the_weights_untouched(synthetic_pack):
    from kickthefly.core import simcore

    before = fa._digest(simcore.new_brain(seed=1, warmup=0, memory=False).sim.W_csr)
    a = fa.autopsy("mb_second_order", seeds=(1000, 1001))
    assert a["read_only"] is True and a["weights_digest"] == before == fa._digest(simcore.new_brain(seed=1, warmup=0, memory=False).sim.W_csr)
    assert a["tag"] == "MODEL PREDICTION" and a["test"]["id"] == "mb_second_order" and a["seeds"] == [1000, 1001]
    s = a["sections"][0]
    assert [b["hop"] for b in s["budget"]] == [1, 2, 3] and s["n_input"] > 0 and s["n_target"] > 0 and "text" in s["verdict"]
    for r in s["routes"]:
        assert len(r["stage_ratios"]) == len(r["types"]) + 2 and len(r["edges"]) == len(r["types"]) + 1
        for st in r["stage_ratios"]:
            assert st["ratio"] == pytest.approx(st["driven_hz"] / max(st["base_hz"], 0.5))
    json.dumps(a, default=str)                                      # serializable


def test_the_autopsy_never_writes_to_the_model():
    import inspect

    src = inspect.getsource(fa)
    for banned in ("set_override(", "W_csr.data =", "W_csr.data[", "apply_to_sim", "sim.p.", "lab_params", "wiring.apply"):
        assert banned not in src, f"the autopsy must not use {banned}"


def test_special_pathways_name_their_input_and_target(synthetic_pack):
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=1, warmup=0, memory=False)
    for tid, n_paths in (("epg_compass", 1), ("epg_compass_wind", 1), ("mb_extinction", 1), ("mb_second_order", 1), ("grooming_hierarchy", 3)):
        pws = fa.pathways(tid, br)
        assert len(pws) == n_paths and all({"label", "inp", "tgt", "inp_label", "tgt_label"} <= set(p) for p in pws)
    epg = fa.pathways("epg_compass", br)[0]
    assert len(epg["inp"]) == 4 and not set(epg["inp"]) & set(epg["tgt"])
    pw = fa.pathways("looming_escape", br)[0]                      # a plain pathway test: the test's own drive and readout
    assert pw["inp_label"].startswith("LPLC2")
    with pytest.raises(fa.AutopsyError, match="larva"):
        fa.pathways("larva_noci_to_goro_rolling", br)
    with pytest.raises(fa.AutopsyError):
        fa.special_pathways("not_a_test", br)


def test_an_empty_input_is_reported_not_crashed(synthetic_pack):
    a = fa.autopsy("p1_courtship_song", seeds=(1000,))             # the synthetic pack has no P1 types
    assert a["sections"][0].get("error") and "empty" in a["sections"][0]["error"]
    md = fa.to_markdown(a)
    assert "empty in this pack" in md


def test_failing_tests_come_from_a_validation_result():
    res = {"brain": "adult", "tests": [dict(id="looming_escape", name="x", passed=True), dict(id="mdn_backward", name="y", passed=False),
                                       dict(id="larva_noci_to_goro_rolling", name="z", passed=False)]}
    assert [t["id"] for t in fa.failing_tests(res)] == ["mdn_backward", "larva_noci_to_goro_rolling"]
    assert fa.failing_tests({"brain": "adult", "tests": []}) == []


def test_run_all_writes_a_page_per_failure_and_an_index(synthetic_pack, tmp_path):
    res = {"brain": "adult", "tests": [dict(id="mb_extinction", name="Mushroom body extinction", passed=False, criteria="PI must fall", measured=dict(p_value=1.0)),
                                       dict(id="mb_second_order", name="Second order", passed=False, criteria="c"),
                                       dict(id="larva_noci_to_goro_rolling", name="larva", passed=False),
                                       dict(id="looming_escape", name="looming", passed=True)]}
    out = fa.run_all(tmp_path, results=res, seeds=(1000,))
    assert [a["test"]["id"] for a in out] == ["mb_extinction", "mb_second_order", "larva_noci_to_goro_rolling"]
    assert "larva" in out[2]["error"] and "Not autopsied" in (tmp_path / "larva_noci_to_goro_rolling.md").read_text(), "a failure that cannot be autopsied still gets a page, saying why"
    names = {p.name for p in tmp_path.iterdir()}
    assert {"mb_extinction.md", "mb_extinction.json", "mb_second_order.md", "index.md"} <= names
    md = (tmp_path / "mb_extinction.md").read_text()
    assert md.startswith("# Autopsy: Mushroom body extinction") and "MODEL PREDICTION" in md and "CONNECTOME" in md and "Read-only" in md
    assert "### Where it fades" in md and "Weights digest" in md and "| synapses |" in md
    assert "mb_extinction.md" in (tmp_path / "index.md").read_text()
    only = fa.run_all(tmp_path / "one", results=res, only=["mb_second_order"], seeds=(1000,))
    assert [a["test"]["id"] for a in only] == ["mb_second_order"]


def test_the_command_line_prints_the_verdicts(synthetic_pack, tmp_path, capsys, monkeypatch):
    from kickthefly.lab import headless

    res = {"brain": "adult", "tests": [dict(id="mb_extinction", name="Mushroom body extinction", passed=False)]}
    monkeypatch.setattr(fa, "AUTOPSY_SEEDS", (1000,))
    monkeypatch.setattr(fa, "run_all", lambda folder, results=None, only=None, seeds=(1000,), progress=None: [
        dict(test=dict(id="x", name="X test"), sections=[dict(verdict=dict(text="the signal is lost between a and b"))], tag=fa.TAG)])
    args = SimpleNamespace(out=str(tmp_path), failure_autopsy=[])
    assert headless.run_failure_autopsy(args, results=res) == 0
    out = capsys.readouterr().out
    assert "X test" in out and "lost between a and b" in out and "read-only" in out


# --- the Lab page -----------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def lab_menu(tmp_path, monkeypatch, synthetic_pack):
    import pygame

    from kickthefly.lab import labautopsy, validation
    from kickthefly.ui import menu as ui

    pygame.init()
    pygame.display.set_mode((1280, 760))
    monkeypatch.setattr(labautopsy, "folder", lambda: tmp_path / "fa")
    res = {"brain": "adult", "tests": [dict(id="mb_extinction", name="Mushroom body extinction", passed=False, criteria="PI must fall"),
                                       dict(id="mb_second_order", name="Second order", passed=False, criteria="c")]}
    monkeypatch.setattr(validation, "load_results", lambda: (res, "test"))
    from test_labpages import StubHost

    m = ui.Menu(StubHost())
    labautopsy.install(m)
    m.fonts()
    m.mouse = (0, 0)
    yield m
    pygame.display.quit()


def _draw(m):
    import pygame

    surf = pygame.Surface((1280, 760))
    m.hits = []
    m.pages["lab_autopsy"](m, surf, pygame.Rect(40, 40, 1200, 680), (0, 0))
    return surf


def _click(m, ident):
    for rect, kind, data in m.hits:
        if kind == "button" and tuple(data["id"]) == tuple(ident):
            data["click"]()
            return True
    return False


def test_the_lab_page_lists_failures_runs_an_autopsy_and_shows_it(lab_menu, tmp_path, monkeypatch):
    m = lab_menu
    _draw(m)
    st = m.lab_state.fa
    assert [f["id"] for f in st["fails"]] == ["mb_extinction", "mb_second_order"] and st["sel"] == "mb_extinction"
    assert _click(m, ("fa", "row", "mb_second_order")) and st["sel"] == "mb_second_order"
    a = fa.autopsy("mb_second_order", seeds=(1000,))

    def fake(folder, results=None, only=None, seeds=(1000,), progress=None):
        progress(1, 1, "mb_second_order")
        return [a]

    monkeypatch.setattr(fa, "run_all", fake)
    _draw(m)
    assert _click(m, ("fa", "one"))
    st["job"]["thread"].join(5)
    _draw(m)
    assert "mb_second_order" in st["done"]
    _draw(m)                                                           # the document
    assert _click(m, ("fa", "export")) and m.host.last_export and "failure-autopsy" in m.host.last_export


def test_the_lab_page_without_a_validation_result_says_so(lab_menu, monkeypatch):
    from kickthefly.lab import validation

    monkeypatch.setattr(validation, "load_results", lambda: (None, "none"))
    surf = _draw(lab_menu)
    assert lab_menu.lab_state.fa["error"] and "validation" in lab_menu.lab_state.fa["error"].lower() and surf.get_bounding_rect().w > 0
