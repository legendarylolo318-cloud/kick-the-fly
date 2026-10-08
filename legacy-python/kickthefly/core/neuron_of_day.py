"""Neuron of the Day (3.0): a small card at launch that shows one curated Neurodex type, one literature fact, and a
"Try it" button that sets up a one-click experiment with it.

No pygame here. Separate from the "Real flies do this too" science cards (Settings > Brain > Real-science popups):
its own setting, Settings > Brain > Neuron of the day (default ON), its own off switch on the card itself, and it never
appears in Lab mode's validation pages or in headless runs.

GAME RULE, all of it: the choice of type, the card, and what Try it sets up. The fact on the card is LITERATURE (the
curated table, kickthefly/data/neurodex_facts.yaml, with its citation); what the experiment then does is whatever the
connectome and the simulator do, read off the brain panel as always. Only curated types are ever picked, so the card
never shows a type without a checked fact and citation.

The pick is by date and nothing else: no clock reads beyond the date, no randomness beyond a fixed shuffle seeded by the
cycle number, no network, no account, no history of what you did. Every curated entry comes up once per cycle of N days
(N = how many there are in the brain you are running), in an order that differs from cycle to cycle, so the same fact
never repeats inside a cycle.

Try it: in Lab mode a type (or a prefix such as PAM) is aimed at with the optogenetics-style laser, activate mode, and
the laser tool is put in your hand; otherwise (Play mode, or several exact types) the type(s) are stimulated in brain
surgery. Either way it REPLACES whatever surgery or laser target you had, and the card says so before you click.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import random

from kickthefly.core import neurodex as nd

MAX_SURGERY_TYPES = 80           # a fact that matches more types than this is too broad to try in one click


def candidates(tab: nd.TypeTable) -> list[nd.Fact]:
    """The curated facts that match at least one type in this brain, sorted by id."""
    if tab.brain != "adult":
        return []
    out = [f for f in nd.facts() if matching_types(f, tab)]
    return sorted(out, key=lambda f: f.id)


def matching_types(fact: nd.Fact, tab: nd.TypeTable) -> list[str]:
    names = [str(n) for n in tab.names]
    return [n for n in names if fact.matches(n)]


def pick(day: _dt.date, tab: nd.TypeTable) -> nd.Fact | None:
    """The fact of the day. Deterministic in (date, candidates)."""
    cands = candidates(tab)
    if not cands:
        return None
    n = len(cands)
    days = day.toordinal()
    cycle, pos = divmod(days, n)
    seed = int(hashlib.sha256(f"kick-the-fly-notd-{n}-{cycle}".encode()).hexdigest()[:12], 16)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return cands[order[pos]]


def plan(fact: nd.Fact, tab: nd.TypeTable, lab: bool) -> dict:
    """What Try it will do, as data: {action: "laser"|"surgery", target, types, text}."""
    types = matching_types(fact, tab)
    if lab and (len(fact.types) == 1 and not fact.prefix):
        target = types[0]
        return {"action": "laser", "target": target, "types": [target],
                "text": f"Aims the laser at {target} (activate) and puts it in your hand. Replaces your laser target."}
    if lab and fact.prefix and not fact.types and len(fact.prefix) == 1:
        target = fact.prefix[0]
        return {"action": "laser", "target": target, "types": types,
                "text": f"Aims the laser at every {target}* type (activate) and puts it in your hand. "
                        "Replaces your laser target."}
    if len(types) > MAX_SURGERY_TYPES:
        return {"action": "none", "target": None, "types": types,
                "text": f"Matches {len(types)} types: too broad for a one-click experiment."}
    shown = ", ".join(types[:4]) + (f" and {len(types) - 4} more" if len(types) > 4 else "")
    return {"action": "surgery", "target": None, "types": types,
            "text": f"Stimulates {shown} in brain surgery. Replaces the surgery you have set."}


def card(day: _dt.date, tab: nd.TypeTable, lab: bool = False) -> dict | None:
    """Everything the launch card shows, or None (no curated type in this brain, e.g. the larva)."""
    f = pick(day, tab)
    if f is None:
        return None
    types = matching_types(f, tab)
    p = plan(f, tab, lab)
    return {"day": day.isoformat(), "fact_id": f.id, "types": types, "type": types[0], "also": len(types) - 1,
            "text": f.text, "cite": f.cite, "doi": f.doi, "game_note": f.game_note, "plan": p,
            "tags": {"type pick, card, Try it": nd.GAME_RULE, "fact": nd.LITERATURE}}


def apply(c: dict, game) -> str:
    """Set up the card's experiment on a game. Returns a one-line note. Never runs anything by itself: the laser or the
    surgery is on, the fly and the brain panel do the rest."""
    p = c["plan"]
    if p["action"] == "laser":
        from kickthefly.lab.laser import LaserState

        if not hasattr(game, "laser_state") or game.laser_state is None:
            game.laser_state = LaserState()
        game.laser_state.set_target(p["target"])
        game.laser_state.set_mode("activate")
        game.select_tool("laser")
        return f"TRY IT  laser on {p['target']} (activate): use the laser on the fly"
    if p["action"] == "surgery":
        game.type_ops = {t: 1 for t in p["types"]}
        game.surgery_modes = [0] * len(game.surgery_modes)
        game._apply_surgery()
        return f"TRY IT  stimulating {len(p['types'])} type(s) in brain surgery"
    return "TRY IT  nothing to set up for this one"
