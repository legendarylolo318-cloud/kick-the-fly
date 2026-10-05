"""3.1.0 task 12, lesion battles (game/lesion_battle.py, ui/battle_ui.py): the budget rules, loadouts saved and shared as ordinary surgery codes, applying the surgery with the
game's own switch, the battle's bookkeeping, the hot-seat page, and (on the real brain) that a lesion really changes how a fly fights."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from conftest import needs_pack
from kickthefly.core import sharecode
from kickthefly.game import lesion_battle as lb

SIZES = {"DNp35": 2, "DNpe052": 2, "LC4": 126, "KCg-m": 1342, "MDN": 4, "LPLC2": 185, "AVLP727m": 6}


# --- the rules -----------------------------------------------------------------------------------------------------------------------------
def test_a_loadout_respects_the_budget_and_the_size_cap():
    lo = lb.Loadout("x")
    lo.add("DNp35", lb.SILENCE, SIZES)
    lo.add("LC4", lb.STIMULATE, SIZES)
    lo.add("MDN", lb.SILENCE, SIZES)
    assert lo.used() == 3 and lo.neurons(SIZES) == 132
    with pytest.raises(lb.LoadoutError, match="budget"):
        lo.add("LPLC2", lb.SILENCE, SIZES)
    lo.add("DNp35", lb.STIMULATE, SIZES)                      # changing a pick that is already in costs nothing
    assert lo.picks["DNp35"] == lb.STIMULATE and lo.used() == 3
    lo.remove("MDN")
    with pytest.raises(lb.LoadoutError, match="at most"):
        lo.add("KCg-m", lb.SILENCE, SIZES)
    with pytest.raises(lb.LoadoutError, match="not a cell type"):
        lo.add("NoSuchType", lb.SILENCE, SIZES)
    with pytest.raises(lb.LoadoutError, match="silenced"):
        lo.add("MDN", 0, SIZES)
    assert lo.problems(SIZES) == [] and "LC4 stimulated" in lo.describe()
    over = lb.Loadout("o", {t: -1 for t in ("DNp35", "DNpe052", "LC4", "MDN")})
    assert any("budget" in p for p in over.problems(SIZES))
    assert any("neurons" in p for p in lb.Loadout("k", {"KCg-m": -1}).problems(SIZES))
    assert lb.Loadout("e").describe() == "no surgery"


def test_type_sizes_counts_neurons_per_type():
    assert lb.type_sizes(np.array(["a", "a", "b", "", "a"])) == {"a": 3, "b": 1}


# --- sharing: an ordinary surgery code -------------------------------------------------------------------------------------------------------
def test_a_loadout_is_shared_as_the_existing_surgery_code():
    lo = lb.Loadout("duelist", {"DNp35": -1, "LC4": 1})
    code = lo.code()
    assert code.startswith("KTF1-SRG-")
    c = sharecode.decode(code)
    assert c.kind == "surgery" and c.payload == {"types": {"DNp35": -1, "LC4": 1}}
    ctx = sharecode.Context(type_names=set(SIZES), brain="adult")
    assert sharecode.validate(c, ctx) is None, "Esc > Share > Import would accept it as it is"
    back = lb.Loadout.from_code(code, SIZES)
    assert back.picks == lo.picks
    assert lb.Loadout.from_code(" ".join(code[i:i + 7] for i in range(0, len(code), 7)), SIZES).picks == lo.picks, "spaces and breaks are ignored"
    with pytest.raises(lb.LoadoutError, match="nothing to share"):
        lb.Loadout("e").code()


def test_codes_that_are_not_a_lesion_loadout_are_refused_with_a_reason():
    with pytest.raises(lb.LoadoutError):
        lb.Loadout.from_code("not a code at all", SIZES)
    code = sharecode.encode("loadout", {"name": "tools", "tools": ["hand", "swatter"]})
    with pytest.raises(lb.LoadoutError, match="loadout code"):
        lb.Loadout.from_code(code, SIZES)
    groups = sharecode.encode("surgery", {"groups": {"Touch: head": -1}, "types": {"LC4": 1}})
    with pytest.raises(lb.LoadoutError, match="groups"):
        lb.Loadout.from_code(groups, SIZES)
    big = sharecode.encode("surgery", {"types": {t: -1 for t in ("DNp35", "DNpe052", "LC4", "MDN")}})
    with pytest.raises(lb.LoadoutError, match="budget"):
        lb.Loadout.from_code(big, SIZES)
    unknown = sharecode.encode("surgery", {"types": {"Zzz9": -1}})
    with pytest.raises(lb.LoadoutError, match="not a cell type"):
        lb.Loadout.from_code(unknown, SIZES)
    huge = sharecode.encode("surgery", {"types": {"KCg-m": -1}})
    with pytest.raises(lb.LoadoutError, match="neurons"):
        lb.Loadout.from_code(huge, SIZES)
    damaged = sharecode.encode("surgery", {"types": {"LC4": 1}})
    with pytest.raises(lb.LoadoutError):
        lb.Loadout.from_code(damaged[:-3] + ("000" if not damaged.endswith("000") else "111"), SIZES)


# --- saved loadouts ---------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "saved_path", lambda: tmp_path / "lesion_loadouts.json")
    return tmp_path / "lesion_loadouts.json"


def test_loadouts_are_saved_overwritten_listed_and_deleted(saved):
    assert lb.load_saved() == {}
    assert lb.save_loadout(lb.Loadout("sniper", {"DNp35": 1}))
    assert lb.save_loadout(lb.Loadout("sniper", {"DNp35": 1, "LC4": -1}))
    assert lb.load_saved() == {"sniper": {"DNp35": 1, "LC4": -1}}
    for i in range(lb.MAX_SAVED - 1):
        assert lb.save_loadout(lb.Loadout(f"n{i}", {"MDN": -1}))
    assert not lb.save_loadout(lb.Loadout("one too many", {"MDN": -1})), "a full list refuses a new name"
    assert lb.save_loadout(lb.Loadout("sniper", {"MDN": 1})), "but still overwrites an existing one"
    lb.delete_saved("sniper")
    assert "sniper" not in lb.load_saved()
    lb.delete_saved("never was")


def test_a_damaged_saved_file_is_ignored(saved):
    saved.write_text("{not json", encoding="utf-8")
    assert lb.load_saved() == {}
    saved.write_text('{"ok": {"LC4": 1}, "bad": {"LC4": 7}, "worse": 5, "": {"LC4": 1}}', encoding="utf-8")
    got = lb.load_saved()
    assert got["ok"] == {"LC4": 1} and "bad" not in got and "worse" not in got


# --- applying the surgery ------------------------------------------------------------------------------------------------------------------
def test_the_surgery_is_the_games_own_switch(synthetic_pack):
    from kickthefly.core import simcore
    from kickthefly.game import kick_the_fly as k2

    br = simcore.new_brain(seed=1, warmup=0, memory=False)
    types = br.types.astype(str)
    n = lb.apply(br, lb.Loadout("x", {"LPLC2": lb.SILENCE, "LC4": lb.STIMULATE, "NoSuchType": lb.SILENCE}))
    assert n == int((types == "LPLC2").sum() + (types == "LC4").sum())
    assert np.all(br.override[types == "LPLC2"] == k2.SURGERY_CURRENT[-1]) and np.all(br.override[types == "LC4"] == k2.SURGERY_CURRENT[1])
    other = ~np.isin(types, ["LPLC2", "LC4"])
    assert not br.override[other].any() and br.surgery


# --- the battle's bookkeeping ------------------------------------------------------------------------------------------------------------------
def _fake_task(winners):
    calls = []

    def task(args):
        calls.append(args)
        rnd = args[-1]
        w = winners[rnd]
        return dict(winner=w, hp={"A": 50.0, "B": 60.0}, shots={"A": 3, "B": 4}, hits={"A": 1, "B": 2}, frames=[], events=[], max_levels={}, round=rnd, match_seed=rnd)
    return task, calls


def test_a_best_of_three_is_decided_by_the_majority_and_each_round_has_its_own_seed():
    task, calls = _fake_task(["A", "B", "A"])
    prog = []
    r = lb.fight(1, 2, lb.Loadout("a", {"LC4": -1}), lb.Loadout("b"), rounds=3, seconds=5, mode="off", task=task, workers=1, progress=lambda d, n: prog.append((d, n)))
    assert r["winner"] == "A" and r["wins"] == {"A": 2, "B": 1} and prog == [(1, 3), (2, 3), (3, 3)]
    assert [c[-1] for c in calls] == [0, 1, 2] and calls[0][2] == {"LC4": -1} and calls[0][3] == {}
    assert r["loadouts"] == {"A": {"LC4": -1}, "B": {}} and r["tag"] == "MODEL PREDICTION" and r["rounds_asked"] == 3
    assert lb.headline(r).startswith("Player 1 wins 2-1 with LC4 silenced")
    task2, _ = _fake_task(["B", None, None])
    r2 = lb.fight(1, 2, lb.Loadout("a"), lb.Loadout("b"), rounds=3, task=task2, workers=1)
    assert r2["winner"] == "B" and lb.headline(r2).startswith("Player 2 wins 1-0 with no surgery at all")
    task3, _ = _fake_task([None])
    r3 = lb.fight(1, 2, lb.Loadout("a"), lb.Loadout("b"), rounds=1, task=task3, workers=1)
    assert r3["winner"] is None and lb.headline(r3) == "A draw (0-0)."


def test_rounds_are_one_or_three_and_a_battle_can_be_cancelled():
    import threading

    with pytest.raises(lb.LoadoutError, match="rounds"):
        lb.fight(1, 2, lb.Loadout(), lb.Loadout(), rounds=2, task=lambda a: {})
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        lb.fight(1, 2, lb.Loadout(), lb.Loadout(), rounds=1, task=lambda a: {}, workers=1, cancel=ev)


def test_a_real_round_on_the_synthetic_pack_runs_end_to_end(synthetic_pack):
    r = lb.fight(7, 8, lb.Loadout("a", {"DNp35": -1}), lb.Loadout("b", {"LC4": 1}), rounds=1, seconds=2.0, mode="off", workers=1)
    rd = r["rounds"][0]
    assert set(rd["hp"]) == {"A", "B"} and rd["seconds"] <= 2.0 + 0.1 and "max_levels" in rd and isinstance(rd["frames"], list)
    assert r["winner"] in ("A", "B", None)


# --- on the real brain: a lesion really changes how a fly fights ----------------------------------------------------------------------------------
@needs_pack
def test_silencing_the_shooters_stops_the_shooting_and_the_battle_is_repeatable():
    a = lb.Loadout("a", {"DNp35": lb.SILENCE, "DNpe052": lb.SILENCE})
    b = lb.Loadout("b")
    r = lb.fight(5000, 5001, a, b, rounds=1, seconds=10.0, mode="subtle", workers=2)
    rd = r["rounds"][0]
    assert rd["shots"]["A"] == 0 and rd["max_levels"]["A"]["fire"] < 1.0 < rd["max_levels"]["B"]["fire"], "the surgery reached the readout the duel reads"
    assert r["winner"] == "B" and rd["hits"]["B"] >= 1
    again = lb.fight(5000, 5001, a, b, rounds=1, seconds=10.0, mode="subtle", workers=2)
    assert again["rounds"][0]["hp"] == rd["hp"] and again["rounds"][0]["shots"] == rd["shots"], "same seeds and surgery, same battle"
    even = lb.fight(5000, 5001, b, b, rounds=1, seconds=10.0, mode="subtle", workers=2)
    assert even["rounds"][0]["shots"]["A"] > 0 and even["rounds"][0]["shots"]["B"] > 0


# --- the hot-seat page ---------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def menu(synthetic_pack, saved):
    from kickthefly.core import config
    from kickthefly.ui import menu as ui

    from test_labpages import StubHost
    from kickthefly.core import simcore

    pygame.init()
    pygame.display.set_mode((1280, 760))
    host = StubHost()
    host.brain = simcore.new_brain(seed=1, warmup=0, memory=False)
    host.cfg = config.Config(None)
    m = ui.Menu(host)
    m.fonts()
    m.mouse = (0, 0)
    from kickthefly.ui import arcade_ui

    m.pages["arcade"] = arcade_ui.page
    yield m
    pygame.display.quit()


def _draw(m):
    surf = pygame.Surface((1280, 760))
    m.hits = []
    m.pages["arcade"](m, surf, pygame.Rect(40, 40, 1200, 680), (0, 0))
    return surf


def _norm(i):
    return tuple(i) if isinstance(i, (tuple, list)) else (i,)


def _click(m, ident):
    for rect, kind, data in m.hits:
        if kind == "button" and _norm(data["id"]) == _norm(ident):
            data["click"]()
            return True
    return False


def test_the_page_runs_a_hot_seat_battle_from_setup_to_result(menu, monkeypatch):
    from kickthefly.ui import arcade_ui, battle_ui

    st = arcade_ui.state(menu)
    st.tab = "battle"
    _draw(menu)
    bt = st.bt
    assert bt.phase == "setup1" and menu.host.brain is not None
    _click(menu, ("bt_cand", "DNp35"))
    _draw(menu)
    assert bt.los[0].picks == {"DNp35": lb.SILENCE}
    bt.effect = lb.STIMULATE
    _click(menu, ("bt_cand", "LPLC2"))
    _click(menu, ("bt_cand", "MDN"))
    _click(menu, ("bt_cand", "LC4"))
    assert bt.los[0].used() == 3 and "budget" in bt.msg and bt.msg_bad
    _draw(menu)
    assert _click(menu, ("bt_rm", "MDN")) and bt.los[0].used() == 2
    assert _click(menu, ("bt_done",)) and bt.phase == "pass"
    _draw(menu)
    assert _click(menu, ("bt_pass",)) and bt.phase == "setup2" and bt.player == 1
    _draw(menu)
    _click(menu, ("bt_cand", "LC4"))
    assert bt.los[1].picks == {"LC4": lb.STIMULATE} and bt.los[0].used() == 2, "player 2 builds their own fly"
    _draw(menu)
    assert _click(menu, ("bt_done",)) and bt.phase == "ready"
    surf = _draw(menu)
    assert surf.get_bounding_rect().w > 0
    assert _click(menu, ("bt_rounds", 3)) and bt.rounds == 3

    def fake_fight(s1, s2, lo1, lo2, rounds=1, **kw):
        frames = [[i * 0.1, -3.0 + i * 0.05, 0.0, 0.0, 100.0, 3.0 - i * 0.05, 0.0, math.pi, 90.0, []] for i in range(30)]
        rd = dict(winner="A", hp={"A": 80.0, "B": 0.0}, shots={"A": 5, "B": 2}, hits={"A": 3, "B": 1}, frames=frames, events=[], seconds=3.0,
                  max_levels={"A": dict(fire=3.0, escape=1.0, run=1.0, walk=1.0), "B": dict(fire=0.1, escape=1.0, run=1.0, walk=1.0)}, round=0)
        return dict(winner="A", wins={"A": 2, "B": 0}, rounds=[rd] * rounds, rounds_asked=rounds, seeds=dict(A=s1, B=s2), loadouts=dict(A=lo1.picks, B=lo2.picks),
                    seconds_each=3.0, individuality="off", took_s=1.0, tag="MODEL PREDICTION")

    import math
    monkeypatch.setattr(lb, "fight", fake_fight)
    assert _click(menu, ("bt_fight",))
    assert bt.phase == "fighting"
    st.job.thread.join(5)
    _draw(menu)
    assert bt.phase == "result" and bt.result["winner"] == "A"
    _draw(menu)
    assert _click(menu, ("bt_round", 1)) and bt.sel_round == 1
    surf = _draw(menu)
    assert lb.headline(bt.result).startswith("Player 1 wins 2-0 with DNp35 silenced")
    assert _click(menu, ("bt_rematch",)) and bt.phase == "ready"
    _draw(menu)
    assert _click(menu, ("bt_reset",)) and bt.phase == "setup1" and bt.los[0].used() == 0


def test_the_page_saves_loads_copies_and_pastes(menu, monkeypatch):
    from kickthefly.core import clipboard
    from kickthefly.ui import arcade_ui

    st = arcade_ui.state(menu)
    st.tab = "battle"
    _draw(menu)
    bt = st.bt
    _click(menu, ("bt_cand", "DNp35"))
    _draw(menu)
    assert _click(menu, ("bt_save",)) and lb.load_saved()
    _draw(menu)
    name = next(iter(lb.load_saved()))
    bt.los[0] = lb.Loadout("player 1")
    _draw(menu)
    assert _click(menu, ("bt_load", name)) and bt.los[0].picks == {"DNp35": lb.SILENCE}
    box = {}
    monkeypatch.setattr(clipboard, "put_text", lambda t: box.setdefault("t", t) is not None or True)
    _draw(menu)
    assert _click(menu, ("bt_copy",)) and box["t"].startswith("KTF1-SRG-") and "Code copied" in bt.msg
    bt.los[0] = lb.Loadout("player 1")
    monkeypatch.setattr(clipboard, "get_text", lambda: box["t"])
    _draw(menu)
    assert _click(menu, ("bt_paste",)) and bt.los[0].picks == {"DNp35": lb.SILENCE}
    monkeypatch.setattr(clipboard, "get_text", lambda: "garbage")
    assert _click(menu, ("bt_paste",)) and bt.msg_bad
    monkeypatch.setattr(clipboard, "get_text", lambda: None)
    assert _click(menu, ("bt_paste",)) and "no text" in bt.msg


def test_the_candidate_search_and_the_default_hints(synthetic_pack):
    from kickthefly.ui import battle_ui as bu

    assert [t for t, _ in bu.candidates(SIZES, "")] == [t for t in lb.SUGGESTED if t in SIZES and SIZES[t] <= lb.MAX_TYPE_NEURONS]
    assert [t for t, _ in bu.candidates(SIZES, "dnp")] == ["DNp35", "DNpe052"]
    assert [t for t, _ in bu.candidates(SIZES, "lc")] == ["LC4", "LPLC2"]
    assert bu.candidates(SIZES, "kcg") == [] and bu.candidates(SIZES, "zzz") == []
    assert bu.candidates(SIZES, "MDN")[0][0] == "MDN"
