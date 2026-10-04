"""Neuron of the Day (3.0): the pick, the card, Try it. Uses the synthetic pack's type names (plumbing only)."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from kickthefly.core import neurodex as nd
from kickthefly.core import neuron_of_day as notd


@pytest.fixture
def tab(synthetic_pack):
    return nd.table("adult")


def test_pick_is_deterministic_by_date(tab):
    d = dt.date(2026, 10, 1)
    assert notd.pick(d, tab) is notd.pick(d, tab) or notd.pick(d, tab).id == notd.pick(d, tab).id
    assert notd.pick(d, tab).id == notd.pick(dt.date(2026, 10, 1), tab).id


def test_every_curated_entry_comes_up_once_per_cycle(tab):
    n = len(notd.candidates(tab))
    assert n >= 15
    start = dt.date(2026, 1, 1).toordinal()
    start -= start % n                                                   # a cycle boundary
    ids = [notd.pick(dt.date.fromordinal(start + i), tab).id for i in range(n)]
    assert sorted(ids) == sorted(f.id for f in notd.candidates(tab))
    nxt = [notd.pick(dt.date.fromordinal(start + n + i), tab).id for i in range(n)]
    assert sorted(nxt) == sorted(ids) and nxt != ids                     # a different order next cycle


def test_only_curated_types_are_ever_picked(tab):
    for i in range(200):
        c = notd.card(dt.date(2026, 3, 1) + dt.timedelta(days=i), tab)
        assert c["text"] and c["cite"] and c["doi"].startswith("10.")
        assert nd.fact_for(c["type"]) is not None and all(nd.fact_for(t) for t in c["types"])


def test_larva_has_no_card(synthetic_pack):
    arrays = dict(type=__import__("numpy").array(["A", "B"]), superclass=__import__("numpy").array(["x", "y"]))
    assert notd.card(dt.date(2026, 10, 1), nd.build_table(arrays, "larva", partners=False)) is None


def test_card_content_and_tags(tab):
    c = notd.card(dt.date(2026, 10, 1), tab)
    assert c["type"] in c["types"] and c["tags"]["fact"] == nd.LITERATURE and c["tags"]["type pick, card, Try it"] == nd.GAME_RULE
    assert c["plan"]["action"] in ("surgery", "laser", "none") and c["plan"]["text"]


def test_plans_by_mode(tab):
    facts = {f.id: f for f in nd.facts()}
    p = notd.plan(facts["giant-fiber"], tab, lab=False)
    assert p == {"action": "surgery", "target": None, "types": ["DNp01"], "text": p["text"]} and "Replaces the surgery" in p["text"]
    p = notd.plan(facts["giant-fiber"], tab, lab=True)
    assert p["action"] == "laser" and p["target"] == "DNp01" and "Replaces your laser target" in p["text"]
    p = notd.plan(facts["pam"], tab, lab=True)
    assert p["action"] == "laser" and p["target"] == "PAM" and len(p["types"]) == 15
    p = notd.plan(facts["adn"], tab, lab=True)                           # two exact types: surgery even in Lab
    assert p["action"] == "surgery" and p["types"] == ["DNg62", "DNge078"]
    big = nd.Fact("x", "t", "c", "d", "k", prefix=("",))
    assert notd.plan(big, tab, lab=False)["action"] == "none"


class FakeGame:
    def __init__(self):
        self.type_ops, self.surgery_modes, self.applied, self.tool = {"OLD": -1}, [1, 0, -1], 0, None

    def _apply_surgery(self):
        self.applied += 1

    def select_tool(self, name):
        self.tool = name
        return True


def test_try_it_sets_surgery_and_replaces_the_old_one(tab):
    facts = {f.id: f for f in nd.facts()}
    g = FakeGame()
    c = {"plan": notd.plan(facts["adn"], tab, lab=False)}
    note = notd.apply(c, g)
    assert g.type_ops == {"DNg62": 1, "DNge078": 1} and g.surgery_modes == [0, 0, 0] and g.applied == 1
    assert "stimulating 2 type" in note


def test_try_it_aims_the_laser_in_lab(tab):
    facts = {f.id: f for f in nd.facts()}
    g = FakeGame()
    notd.apply({"plan": notd.plan(facts["mdn"], tab, lab=True)}, g)
    assert g.tool == "laser" and g.laser_state.target_type == "MDN" and g.laser_state.mode == "activate"
    assert g.type_ops == {"OLD": -1}                                      # the laser path leaves surgery alone
