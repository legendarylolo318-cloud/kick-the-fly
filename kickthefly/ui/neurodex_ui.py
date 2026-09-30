"""The Neurodex panel (Esc > Neurodex, or the Neurodex key, default D): progress by region, the list of discovered cell
types, and an entry for each.

What the tags mean on screen (the canonical map is the docstring of kickthefly/game/kick_the_fly.py):
  CONNECTOME   the numbers in an entry, read from the dataset (count, superclass, regions, predicted transmitter and the
               dataset's own confidence, strongest partner types by synapse count, and a cached EM skeleton if there is one)
  GAME RULE    "discovered" and the collection (kickthefly/core/neurodex.py says exactly when a type counts)
  LITERATURE   the one-line fact and citation on a curated type (kickthefly/data/neurodex_facts.yaml)

An undiscovered type shows only "???": no name, no data, so the collection stays a collection. The search box only finds
names you have discovered. Gamepad (3D): the Neurodex button (d-pad up by default) opens it; the bumpers move through the
list, d-pad down / Back change region, the trigger tries the Neuron of the Day, B or the Neurodex button closes it.
"""
from __future__ import annotations

import datetime as _dt
import time

import numpy as np
import pygame

from kickthefly.core import neurodex as nd
from kickthefly.core import neuron_of_day as notd
from kickthefly.core.i18n import tr
from kickthefly.ui import menu as ui

ROW_H = 26


def install(menu) -> None:
    menu.pages["neurodex"] = page_neurodex


class _State:
    def __init__(self):
        self.region = "(all)"
        self.query = ""
        self.only_found = True
        self.only_curated = False
        self.selected: str | None = None
        self.cache_key = None
        self.cache_list: list[int] = []
        self.skel_key = None
        self.skel = None
        self.wipe_armed = 0.0
        self.notd = None
        self.rows_key = None
        self.rows = []


def _state(m) -> _State:
    st = getattr(m, "_dex_state", None)
    if st is None:
        st = m._dex_state = _State()
    return st


def _truncate(font, text: str, width: int) -> str:
    if font.size(text)[0] <= width:
        return text
    while text and font.size(text + "…")[0] > width:
        text = text[:-1]
    return text + "…"


def page_neurodex(m, surf, rect, mouse) -> None:
    host = m.host
    x3 = host.x3
    x3.ensure_table()
    prog = x3.ensure_progress()
    st = _state(m)
    brain = x3.brain_name
    m.text(surf, tr("NEURODEX"), (rect.x + 24, rect.y + 14), ui.INK, m.f_head)
    back = lambda: (setattr(st, "selected", st.selected), m.back())         # noqa: E731
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back"), back, style="primary", id=("dex", "back"))
    tab = x3.table
    if tab is None:
        msg = {"building": tr("Building the Neurodex from the brain pack..."),
               "none": tr("No brain pack is built here, so there is no Neurodex to show."),
               "error": tr("The Neurodex could not be built") + f": {x3.table_error}"}.get(x3.table_state, "...")
        m.text(surf, msg, rect.center, ui.LABEL, m.f_text, "center")
        return
    got, total = prog.n_discovered(brain), len(tab)
    hdr = m.text(surf, f"{got:,} / {total:,} " + tr("cell types discovered") + f"   ({brain})", (rect.x + 24, rect.y + 46),
                 ui.LABEL, m.f_small)
    chip = m.chip(surf, (hdr.right + 14, rect.y + 46), "GAME RULE")
    m.text(surf, _truncate(m.f_small, tr("discovered = fires well above its calm rate while you play"),
                           rect.right - chip.right - 24), (chip.right + 8, rect.y + 46), ui.DIM, m.f_small)

    # --- left: regions with progress
    colx, top = rect.x + 24, rect.y + 76
    rkey = (got, brain)
    if getattr(st, "rows_key", None) != rkey:                   # the progress rows loop over every type: only when it changes
        st.rows_key, st.rows = rkey, nd.region_progress(tab, prog)
    rows = st.rows
    y = top
    m.text(surf, tr("REGIONS"), (colx, y), ui.LABEL, m.f_small)
    y += 20
    for reg, a, b in rows:
        sel = st.region == reg
        r = pygame.Rect(colx, y, 236, 30)
        m.button(surf, r, "", (lambda reg=reg: setattr(st, "region", reg)), id=("dex-reg", reg), active=sel,
                 tip=tr("Filter the list to this region. Regions are the coarse ones the game derives from the dataset's "
                        "class and soma-neuromere annotations."))
        m.text(surf, _truncate(m.f_small, reg if reg != "unassigned" else tr("unassigned"), 140), (r.x + 8, r.y + 3),
               ui.INK if sel else ui.TEXT, m.f_small)
        m.text(surf, f"{a}/{b}", (r.right - 8, r.y + 3), ui.LABEL, m.f_small, "topright")
        pygame.draw.rect(surf, (10, 12, 18), (r.x + 8, r.bottom - 9, r.w - 16, 4), border_radius=2)
        if b:
            pygame.draw.rect(surf, ui.ACCENT, (r.x + 8, r.bottom - 9, int((r.w - 16) * a / b), 4), border_radius=2)
        y += 34
        if y > rect.bottom - 190:
            break
    y = rect.bottom - 170
    m.toggle(surf, (colx, y, 72, 26), st.only_found, lambda v: setattr(st, "only_found", bool(v)), id="dex-found",
             tip=tr("Off lists every type in the region as ??? (nothing is revealed about an undiscovered type)."))
    m.text(surf, tr("Only discovered"), (colx + 82, y + 13), ui.TEXT, m.f_small, "midleft")
    m.toggle(surf, (colx, y + 32, 72, 26), st.only_curated, lambda v: setattr(st, "only_curated", bool(v)), id="dex-cur",
             tip=tr("Only discovered types that have a literature fact and citation."))
    m.text(surf, tr("Only with a fact"), (colx + 82, y + 45), ui.TEXT, m.f_small, "midleft")
    armed = time.perf_counter() - st.wipe_armed < 3.0

    def wipe():
        if armed:
            prog.wipe(brain)
            prog.save()
            x3.tracker and x3.tracker.sync()
            st.wipe_armed, st.selected = 0.0, None
        else:
            st.wipe_armed = time.perf_counter()

    m.button(surf, (colx, y + 66, 236, 30), tr("Click again to erase") if armed else tr("Reset Neurodex"), wipe,
             style="danger" if armed else "normal", id="dex-wipe",
             tip=tr("Forget every discovery for this brain. Training memory is not touched."))

    # --- middle: search and list
    mx, mw = rect.x + 280, 290
    m.text_field(surf, (mx, top, mw, 28), st.query, lambda v: setattr(st, "query", v), id="dex-q", limit=24,
                 tip=tr("Type part of a discovered type's name."))
    if not st.query and not (m.edit and m.edit.get("id") == "dex-q"):
        m.text(surf, tr("search discovered types"), (mx + 10, top + 14), ui.DIM, m.f_small, "midleft")
    key = (st.region, st.query, st.only_found, st.only_curated, got, brain)
    if st.cache_key != key:
        st.cache_key = key
        st.cache_list = nd.list_entries(tab, prog, st.region, st.query, st.only_found, st.only_curated)
    lst = st.cache_list
    body = pygame.Rect(mx, top + 38, mw, rect.bottom - top - 38 - 70)
    skey = "neurodex"
    off = int(m.scroll.get(skey, 0))
    m.content_h[skey] = max(0, len(lst) * ROW_H - body.h)
    pygame.draw.rect(surf, (12, 14, 20), body, border_radius=8)
    m.clip = body
    prev = surf.get_clip()
    surf.set_clip(body)
    known = prog.types(brain)
    first = off // ROW_H
    for k in range(first, min(len(lst), first + body.h // ROW_H + 2)):
        i = lst[k]
        name = str(tab.names[i])
        r = pygame.Rect(body.x + 2, body.y + k * ROW_H - off, body.w - 14, ROW_H - 2)
        found = name in known
        if found:
            cur = nd.fact_for(name, brain) is not None
            m.button(surf, r, "", (lambda n=name: setattr(st, "selected", n)), id=("dex-row", name),
                     active=st.selected == name)
            m.text(surf, _truncate(m.f_small, name, r.w - 50), (r.x + 8, r.y + 4), ui.INK, m.f_small)
            if cur:
                m.text(surf, "*", (r.right - 26, r.y + 3), ui.AMBER, m.f_small)
            m.text(surf, str(int(tab.count[i])), (r.right - 6, r.y + 4), ui.DIM, m.f_small, "topright")
        else:
            m.button(surf, r, "", (lambda n=name: setattr(st, "selected", n)), id=("dex-row", name),
                     active=st.selected == name)
            m.text(surf, "???", (r.x + 8, r.y + 4), ui.DIM, m.f_small)
            m.text(surf, tab.region[i], (r.right - 6, r.y + 4), ui.DIM, m.f_small, "topright")
    if not lst:
        m.text(surf, tr("Nothing here yet. Play, and it fills in.") if st.only_found else tr("Nothing matches."),
               body.center, ui.LABEL, m.f_small, "center")
    surf.set_clip(prev)
    m.clip = None

    # --- right: the entry
    dx, dw = rect.x + 588, rect.w - 588 - 24
    _draw_entry(m, surf, pygame.Rect(dx, top, dw, rect.bottom - top - 70), tab, prog, st, brain)

    # --- bottom: Neuron of the Day
    c = _notd_card(host, tab, st)
    if c is not None:
        y = rect.bottom - 52
        m.text(surf, tr("Neuron of the day") + f": {c['type']}" + (f" (+{c['also']})" if c["also"] else ""),
               (rect.x + 24, y + 4), ui.TEXT, m.f_small)
        m.text(surf, _truncate(m.f_small, c["plan"]["text"], 560), (rect.x + 24, y + 22), ui.DIM, m.f_small)
        m.button(surf, (rect.x + 600, y, 140, 38), tr("Try it"), lambda: _try(host, c, m), id="dex-try",
                 enabled=c["plan"]["action"] != "none",
                 tip=c["plan"]["text"] + " " + tr("It closes this screen so you can watch."))


def _notd_card(host, tab, st):
    if st.notd is None or st.notd[0] != _dt.date.today() or st.notd[2] != host.cfg.lab:
        st.notd = (_dt.date.today(), notd.card(_dt.date.today(), tab, lab=host.cfg.lab), host.cfg.lab)
    return st.notd[1]


def _try(host, card, m) -> None:
    host.note(notd.apply(card, host), source="rule")
    m.close()


def _draw_entry(m, surf, box: pygame.Rect, tab, prog, st, brain: str) -> None:
    name = st.selected
    if not name or tab.index(name) is None:
        m.text(surf, tr("Pick a type."), box.center, ui.DIM, m.f_text, "center")
        return
    e = nd.entry(tab, name, prog)
    x, y = box.x, box.y
    if not e["discovered"]:
        m.text(surf, "???", (x, y), ui.DIM, m.f_head)
        m.text(surf, e["region"], (x, y + 34), ui.LABEL, m.f_small)
        m.wrapped(surf, tr("Not discovered yet. Its neurons have to fire well above their calm rate while you play. "
                           "Nothing else is shown until they do."), (x, y + 62), box.w, ui.DIM, m.f_small, 4)
        return
    m.text(surf, name, (x, y), ui.INK, m.f_head)
    y += 34
    how = {"stimulated": tr("discovered by stimulating it"),
           "rest": tr("discovered at rest (a spontaneous burst, nothing touching the fly)")}.get(e["how"], tr("discovered in play"))
    c1 = m.chip(surf, (x, y), "GAME RULE")
    m.text(surf, f"{how}, {(e['when'] or '')[:10]}, x{(e['peak_x'] or 0):.1f} its calm rate", (c1.right + 8, y), ui.LABEL, m.f_small)
    y += 26
    c2 = m.chip(surf, (x, y), "CONNECTOME")
    m.text(surf, tr("from the dataset"), (c2.right + 8, y), ui.LABEL, m.f_small)
    y += 24
    lines = [f"{e['count']:,} " + (tr("neuron") if e["count"] == 1 else tr("neurons")) + f"   ·   {e['superclass'] or '?'}",
             tr("regions") + ": " + (", ".join(f"{r} ({c})" for r, c in e["regions"]) or "?")]
    if e["transmitter"]:
        c = e["transmitter_confidence"]
        meas = e["transmitter_measured_share"]
        t = f"{tr('transmitter')} ({tr('predicted')}): {e['transmitter']}"
        t += f", {tr('confidence')} {c:.2f}" if c is not None else ""
        t += f", {meas:.0%} {tr('measured')}" if meas else ""
        lines.append(t)
    else:
        lines.append(tr("transmitter") + ": " + tr("not in this dataset")
                     + (" (" + tr("the game assigns signs by a rule") + ")" if brain == "larva" else ""))
    for ln in lines:
        y = m.wrapped(surf, ln, (x, y), box.w, ui.TEXT, m.f_small, 2) + 2
    # partners
    half = box.w // 2 - 6
    for col, (label, rows_) in enumerate(((tr("TOP INPUTS"), e["inputs"]), (tr("TOP OUTPUTS"), e["outputs"]))):
        cx, cy = x + col * (half + 12), y + 4
        m.text(surf, label, (cx, cy), ui.LABEL, m.f_small)
        for k, (pn, syn, share) in enumerate(rows_[:5]):
            m.text(surf, _truncate(m.f_small, pn, half - 84), (cx, cy + 18 + k * 17), ui.TEXT, m.f_small)
            m.text(surf, f"{syn:,}  {share:.0%}", (cx + half, cy + 18 + k * 17), ui.DIM, m.f_small, "topright")
    y += 4 + 18 + 5 * 17 + 6
    m._register(pygame.Rect(x, y - 110, box.w, 110), "label", id=("dex-partner-tip",),
                tip=tr("Strongest partner types by synapse count in the game's brain pack. The share is of this type's "
                       "total input or output synapses. Connections below the loader's minimum weight, and unsigned "
                       "dopamine synapses, are not in the pack."))
    # skeleton
    if st.skel_key != name:
        st.skel_key, st.skel = name, nd.skeleton_points(tab, name)
    sk = pygame.Rect(x, y, 150, box.bottom - y - 4)
    if sk.h > 60:
        pygame.draw.rect(surf, (8, 10, 14), sk, border_radius=6)
        if st.skel is not None:
            pts = st.skel[:, :2].astype(np.float32)
            lo, hi = pts.min(0), pts.max(0)
            span = max(float((hi - lo).max()), 1.0)
            sc = (min(sk.w, sk.h) - 16) / span
            px = [(sk.x + 8 + (p[0] - lo[0]) * sc, sk.y + 8 + (p[1] - lo[1]) * sc) for p in pts]
            pygame.draw.lines(surf, ui.ACCENT, False, px, 2)
            m.text(surf, tr("EM skeleton (cached)"), (sk.x + 6, sk.bottom - 16), ui.DIM, m.f_small)
        else:
            m.wrapped(surf, tr("no skeleton cached for this type"), (sk.x + 8, sk.centery - 10), sk.w - 16, ui.DIM, m.f_small, 3)
    # the curated fact
    fx = x + 162
    fw = box.w - 162
    cur = e["curated"]
    if cur:
        m.chip(surf, (fx, y), "LITERATURE")
        yy = m.wrapped(surf, cur["text"], (fx, y + 22), fw, ui.TEXT, m.f_small, 6)
        yy = m.wrapped(surf, cur["cite"], (fx, yy + 2), fw, ui.LABEL, m.f_small, 3)
        yy = m.wrapped(surf, "doi:" + cur["doi"], (fx, yy), fw, ui.DIM, m.f_small, 1)
        if cur["game_note"]:
            m.wrapped(surf, tr("In the game") + ": " + cur["game_note"], (fx, yy + 4), fw, ui.AMBER, m.f_small, 4)
    else:
        m.wrapped(surf, tr("No curated fact for this type: data only."), (fx, y), fw, ui.DIM, m.f_small, 2)


def pad_nav(host, down) -> bool:
    """Gamepad while the Neurodex is open (3D). `down` holds the action names whose buttons just went down, so these are
    the pad's own bindings: the bumpers (next / previous tool) move through the list, the kill cam button (d-pad down by
    default) and the big-view button (Back) change region, the trigger tries the Neuron of the Day, and the Neurodex
    button or B (crouch) closes it. True if a button was used."""
    m = host.menu
    if m.screen != "neurodex":
        return False
    st = _state(m)
    if down & {"neurodex", "crouch", "menu"}:
        m.back()
        return True
    tab = host.x3.table
    if tab is None:
        return False
    prog = host.x3.ensure_progress()
    used = False
    regs = [r for r, _, _ in nd.region_progress(tab, prog)]
    if down & {"tool_next", "tool_prev"}:
        names = [str(tab.names[i]) for i in st.cache_list]
        if names:
            cur = names.index(st.selected) if st.selected in names else -1
            st.selected = names[(cur + (1 if "tool_next" in down else -1)) % len(names)]
            used = True
    if down & {"killcam", "big_view"}:
        i = regs.index(st.region) if st.region in regs else 0
        st.region = regs[(i + (1 if "killcam" in down else -1)) % len(regs)]
        used = True
    if "use" in down:
        c = _notd_card(host, tab, st)
        if c is not None and c["plan"]["action"] != "none":
            _try(host, c, m)
            used = True
    return used
