"""Lab screens for the 3.0 day 2 toolkit: Genetic toolkit, Thermogenetics, Patch clamp, Imaging and Pharmacology.

The logic is in genetics.py, thermogenetics.py, patchclamp.py, imaging.py and pharmacology.py (each says what is connectome,
literature, game rule and model); this file only draws it. Every screen carries its tag on screen: LITERATURE / CONNECTOME /
RULE / MODEL (and for the drugs, MODEL PREDICTION). Colors follow Settings > Accessibility; nothing here flashes.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pygame

from kickthefly.ui import menu as ui


def _st(m) -> SimpleNamespace:
    st = getattr(m, "toolkit", None)
    if st is None:
        st = m.toolkit = SimpleNamespace(
            query="DNp01", selected=None, results=[], note="",
            # thermogenetics
            t_effector=0, t_target="type:DNp01",
            # patch
            p_type="DNp01", p_index=0, p_mode=0, p_amp=0.1, p_dur=500, p_repeats=1, p_hold=0.0, p_live=False,
            p_rec=None, p_curve=None, p_job=None, p_electrode=None, p_row=None, p_last_frame=0.0,
            # imaging
            i_roi_text="", i_seconds=10, i_msg="",
            # pharmacology
            d_drug="picrotoxin", d_dose=0.5, d_low=True, d_cut=0.7, d_job=None, d_result=None, d_dr=None)
    return st


def _brain(host):
    br = host.brain
    if br is not None and getattr(br, "graph", None) is None and getattr(host, "graph", None) is not None:
        br.graph = host.graph                        # the brain's own labels (body id, region), as simcore.new_brain sets it
    return br


def _types(host):
    return _brain(host).types


def _title(m, surf, rect, title: str, sub: str, tags: list[tuple[str, str]]) -> int:
    """Heading, one line of description and the tag row (chips wrap to the next line rather than leave the panel)."""
    m.text(surf, title, (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = m.wrapped(surf, sub, (rect.x + 24, max(m.under_heading(rect), rect.y + 48)), rect.w - 60, ui.LABEL, m.f_small, 2) + 4
    x = rect.x + 24
    limit = rect.right - 24
    step = m.f_small.get_linesize() + 6              # 3.0 release review: was a fixed 22 px, so rows overlapped at larger text
    for chip, text in tags:
        w_chip = m.f_small.size(chip.upper())[0] + 12
        w_text = m.f_small.size(text)[0] + 6
        if x + w_chip + w_text > limit and x > rect.x + 24:
            x, y = rect.x + 24, y + step
        r = m.chip(surf, (x, y), chip)
        if r.right + 6 + w_text > limit:              # a text too long for what is left of the row wraps under itself
            yb = m.wrapped(surf, text, (r.right + 6, y), limit - r.right - 6, ui.LABEL, m.f_small, 3)
            x, y = rect.x + 24, max(y + step, yb + 4)
            continue
        t = m.text(surf, text, (r.right + 6, y), ui.LABEL, m.f_small)
        x = t.right + 18
    return y + step + 6


def _back(m, rect, key: str) -> None:
    m.button(surf_of(m), (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=(key, "back"))


def surf_of(m):
    return m._surf


def rect_top(plot) -> int:
    """Where a plot's legend goes: on the plot's top edge, to the right of its y-axis label."""
    return plot.y + 4


def _plot(m, surf, rect, series, xlim, ylim, xlabel="", ylabel="", hlines=(), markers=()):
    """A small line plot. series: [(points ndarray (n,2), color, width)]; hlines: [(y, color, label)]; markers: [(x, y, color)]."""
    pygame.draw.rect(surf, (12, 14, 20), rect, border_radius=8)
    plot = pygame.Rect(rect.x + 46, rect.y + 24, rect.w - 60, rect.h - 54)
    (x0, x1), (y0, y1) = xlim, ylim
    sx = plot.w / max(1e-9, x1 - x0)
    sy = plot.h / max(1e-9, y1 - y0)

    def px(x, y):
        return (plot.x + (x - x0) * sx, plot.bottom - (y - y0) * sy)

    for k in range(5):
        yy = y0 + (y1 - y0) * k / 4
        py = plot.bottom - (yy - y0) * sy
        pygame.draw.line(surf, (36, 40, 50), (plot.x, py), (plot.right, py))
        m.text(surf, f"{yy:.3g}", (plot.x - 6, py), ui.LABEL, m.f_small, "midright")
    for y, col, label in hlines:
        py = plot.bottom - (y - y0) * sy
        if plot.y <= py <= plot.bottom:
            pygame.draw.line(surf, col, (plot.x, py), (plot.right, py), 1)
            m.text(surf, label, (plot.right - 4, py - 2), col, m.f_small, "bottomright")
    prev = surf.get_clip()
    surf.set_clip(plot)
    for pts, col, width in series:
        if len(pts) > 1:
            pygame.draw.lines(surf, col, False, [px(x, y) for x, y in pts], width)
    for x, y, col in markers:
        q = px(x, y)
        pygame.draw.circle(surf, col, (int(q[0]), int(q[1])), 4)
    surf.set_clip(prev)
    if xlabel:
        m.text(surf, xlabel, (plot.centerx, rect.bottom - 4), ui.LABEL, m.f_small, "midbottom")
    if ylabel:
        m.text(surf, ylabel, (rect.x + 6, rect.y + 2), ui.LABEL, m.f_small)


# ==== Genetic toolkit ==============================================================================================================
def page_genetics(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import genetics

    m._surf = surf
    st, host = _st(m), m.host
    y = _title(m, surf, rect, "GENETIC TOOLKIT",
               "Choose neurons by split-GAL4 driver line instead of by cell type. Surgery, the laser and thermogenetics accept a line.",
               [("LITERATURE", "line -> cell types: Meissner et al. 2025, eLife (CC BY 4.0)"), ("CONNECTOME", "neuron counts"),
                ("RULE", "a cell-type name matches only if spelled identically")])
    try:
        types = _types(host)
        cov = genetics.coverage(types)
    except Exception as e:
        m.wrapped(surf, f"needs the adult brain pack and the driver-line table: {e}", (rect.x + 24, y + 10), rect.w - 48, ui.BAD, m.f_small, 4)
        _back(m, rect, "genetics")
        return
    y = m.wrapped(surf, f"{cov['lines']:,} adult split-GAL4 lines; {cov['lines_with_a_match']:,} name at least one cell type spelled like "
                        f"this connectome's. Not included: GAL4 (non-split) lines, for which no redistributable mapping was found.",
                  (rect.x + 24, y), rect.w - 60, ui.LABEL, m.f_small, 2) + 6
    m.text(surf, "Search (line or cell type)", (rect.x + 24, y + 12), ui.TEXT, m.f_text, "midleft")
    m.text_field(surf, (rect.x + 260, y, 240, 28), st.query, lambda v: _search(st, v), id="gen_query", limit=24,
                 tip="A line name such as SS00727, or a cell type such as DNp01. Enter to search.")
    m.button(surf, (rect.x + 510, y, 90, 28), "Search", lambda: _search(st, st.query), id="gen_go")
    y += 40
    left = pygame.Rect(rect.x + 24, y, 430, rect.bottom - y - 70)
    pygame.draw.rect(surf, (20, 23, 31), left, border_radius=8)
    yy = left.y + 6
    for ln in st.results[:16]:
        r = pygame.Rect(left.x + 4, yy, left.w - 8, 24)
        sel = st.selected == ln.name
        m.button(surf, r, f"{ln.name}   q{ln.quality}   {', '.join(ln.cell_types)[:40]}", (lambda n=ln.name: setattr(st, "selected", n)),
                 id=("gen_pick", ln.name), style="primary" if sel else "normal", font=m.f_small)
        yy += 26
    if not st.results:
        m.text(surf, "no lines found" if st.query else "type something to search", (left.x + 10, left.y + 10), ui.LABEL, m.f_small)
    x2 = left.right + 20
    w2 = rect.right - x2 - 24
    if st.selected:
        d = genetics.describe(st.selected, types)
        yy = y
        m.text(surf, d["line"], (x2, yy), ui.INK, m.f_head)
        yy += 34
        m.text(surf, f"line  ->  cell types  ->  neurons in this connectome: {d['neurons']:,}", (x2, yy), ui.TEXT, m.f_bold)
        yy += 24
        for t in d["cell_types"]:
            n = d["matched"].get(t)
            m.text(surf, f"{t}", (x2 + 8, yy), ui.INK if n else ui.LABEL, m.f_small)
            m.text(surf, f"{n:,} neurons" if n else "no identically spelled type in this connectome (not guessed)",
                   (x2 + 200, yy), ui.GOOD if n else ui.AMBER, m.f_small)
            yy += 18
        yy += 6
        col = {"minimal": ui.GOOD, "some": ui.AMBER, "yes": ui.BAD, "weak": ui.AMBER, "unstable": ui.BAD}.get(d["off_target"], ui.LABEL)
        m.text(surf, "Off-target expression (the source's own score)", (x2, yy), ui.TEXT, m.f_small)
        yy = m.wrapped(surf, d["off_target_text"], (x2, yy + 16), w2, col, m.f_small, 4) + 4
        m.text(surf, f"{d['cite'].replace(';', '')}  ·  doi:{d['doi']}  ·  sex difference: {d['sex_difference'] or 'n/a'}",
               (x2, yy), ui.LABEL, m.f_small)
        yy += 28
        ok = d["neurons"] > 0
        cur = {t: host.type_ops.get(t, 0) for t in d["matched"]}
        state = "silenced" if cur and all(v < 0 for v in cur.values()) else "stimulated" if cur and all(v > 0 for v in cur.values()) else "normal"
        m.text(surf, f"surgery on these types: {state}", (x2, yy), ui.LABEL, m.f_small)
        yy += 22
        for k, (label, mode) in enumerate((("Silence", -1), ("Stimulate", 1), ("Clear", 0))):
            m.button(surf, (x2 + k * 110, yy, 102, 30), label, (lambda md=mode: _line_surgery(host, st, d, md)), id=("gen_op", mode),
                     enabled=ok, tip="Brain surgery on the cell types this line labels (the same switches as the inspector).")
        yy += 38
        _, yy = m.button_row(surf, x2, yy, 30, [
            dict(label="Aim the laser at this line", click=lambda: _laser_line(host, d["line"], st), id="gen_laser", enabled=ok, w=210,
                 tip="Sets the Lab laser's target to this line (activate mode) and puts the laser in your hand."),
            dict(label="Express TrpA1 here", click=lambda: _thermo_line(host, "trpa1", d["line"], st), id="gen_trp", enabled=ok, w=210,
                 tip="Adds a thermogenetic expression: activates these neurons above the TrpA1 temperature."),
            dict(label="Express shibire-ts here", click=lambda: _thermo_line(host, "shibire", d["line"], st), id="gen_shi", enabled=ok, w=210,
                 tip="Adds a thermogenetic expression: silences these neurons above the shibire-ts temperature.")], right=rect.right - 24)
        yy -= 30
        yy += 40
        if st.note:
            m.text(surf, st.note, (x2, yy), ui.AMBER, m.f_small)
    m.wrapped(surf, "What this does not know: expression strength, what a line labels beyond its authors' score, developmental timing. "
                    "Nothing here is a measurement on this connectome.", (rect.x + 24, rect.bottom - 62), rect.w - 220, ui.LABEL, m.f_small, 2)
    _back(m, rect, "genetics")


def _search(st, q: str) -> None:
    from kickthefly.lab import genetics

    st.query = str(q).strip()
    st.results = genetics.search(st.query)
    if st.results:
        st.selected = st.results[0].name
    st.note = ""


def _line_surgery(host, st, d, mode: int) -> None:
    for t in d["matched"]:
        if mode == 0:
            host.type_ops.pop(t, None)
        else:
            host.type_ops[t] = mode
    host._apply_surgery()
    host.note(f"SURGERY  line {d['line']} ({', '.join(d['matched'])}): {'off' if mode < 0 else 'on' if mode > 0 else 'normal'}")
    st.note = f"line {d['line']}: {'silenced' if mode < 0 else 'stimulated' if mode > 0 else 'cleared'}"


def _laser_line(host, line: str, st) -> None:
    from kickthefly.lab.laser import LaserState

    if getattr(host, "laser_state", None) is None:
        host.laser_state = LaserState()
    host.laser_state.set_target(f"line:{line}")
    host.laser_state.set_mode("activate")
    host.select_tool("laser")
    st.note = f"laser aimed at line {line} (activate)"


def _thermo_line(host, effector: str, line: str, st) -> None:
    try:
        host.thermo_live.add(effector, f"line:{line}", brain=_brain(host))
        st.note = f"{effector} expressed in line {line}: set the temperature in Lab > Thermogenetics"
    except Exception as e:
        st.note = str(e)


# ==== Thermogenetics ===============================================================================================================
def page_thermo(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import thermogenetics as tg

    m._surf = surf
    st, host = _st(m), m.host
    tl = host.thermo_live
    y = _title(m, surf, rect, "THERMOGENETICS",
               "TrpA1 activates and shibire-ts silences the neurons that express them, above a threshold temperature.",
               [("LITERATURE", "onset ~25 C TrpA1 (Pulver 2009); shibire-ts 30 C (Kitamoto 2001)"),
                ("RULE", "the rest of the curve, kinetics and current size"), ("MODEL", "only expressing neurons respond")])
    x = rect.x + 24
    m.text(surf, "Temperature source", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    m.segmented(surf, (x + 200, y, 360, 28), ["Lab slider", "Thermo arena"], 0 if tl.source == "slider" else 1,
                lambda i: setattr(tl, "source", "slider" if i == 0 else "arena"), id="th_src",
                tip="Thermo arena: each fly senses the temperature where it stands, 15 C at the cold wall to 35 C at the hot wall "
                    "(a game rule). Elsewhere the slider applies.")
    y += 36
    m.text(surf, "Temperature", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (x + 200, y, 360, 28), tl.temperature_c, tg.TEMP_RANGE_C[0], tg.TEMP_RANGE_C[1], 0.5, "{:.1f} C",
             lambda v: setattr(tl, "temperature_c", float(v)), lambda: None, id="th_temp", enabled=tl.source == "slider",
             tip="Lab slider temperature. Nothing in the model changes with temperature except the effectors you add here.")
    if tl.source == "arena":                     # 3.0 day 2 review: the page showed the idle slider's value, not what flies sense
        from kickthefly.game import kick_the_fly as k2

        temps = [getattr(s_, "arena_temp_c", None) for s_ in host.flies] if k2.ARENAS[host.arena_i] == "thermo" else []
        temps = [t for t in temps if t is not None]
        m.text(surf, ("flies sense " + ", ".join(f"{t:.1f} C" for t in temps[:4]) + " (thermo arena, game rule)") if temps else
               "not in the thermo arena: the slider applies", (x + 580, y + 14), ui.AMBER, m.f_small, "midleft")
    y += 36
    m.text(surf, "Kinetics", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    m.segmented(surf, (x + 200, y, max(360, 2 * m.f_small.size("Realistic (game rule)")[0] + 40), max(28, m.f_small.get_linesize() + 10)),
                ["Realistic (game rule)", "Instant"], 0 if tl.kinetics == "real" else 1,
                lambda i: setattr(tl, "kinetics", "real" if i == 0 else "steady"), id="th_kin",
                tip="Realistic: TrpA1 follows the temperature in about 1 s, shibire-ts takes about 40 s to block and 20 s to recover "
                    "(read from Kitamoto 2001: paralysis within 2 min, recovery in about 1 min). Instant: the steady-state level.")
    y += 44
    m.text(surf, "Expression", (x, y), ui.INK, m.f_bold)
    y += 22
    rows = tl.status()
    for i, e in enumerate(tl.expressions):
        s = rows[i] if i < len(rows) else {}
        ef = tg.effector(e["effector"])
        m.text(surf, f"{ef.key}  ->  {e['target']}", (x, y + 12), ui.TEXT, m.f_small, "midleft")
        n = s.get("neurons", 0)
        m.text(surf, f"{n:,} neurons" if n else "(resolved when a fly is live)", (x + 330, y + 12), ui.LABEL, m.f_small, "midleft")
        a = float(s.get("activation", 0.0))
        pygame.draw.rect(surf, (30, 36, 48), (x + 480, y + 4, 160, 14), border_radius=7)
        pygame.draw.rect(surf, ui.AMBER if ef.kind == "activate" else (90, 150, 230), (x + 480, y + 4, max(3, int(160 * a)), 14), border_radius=7)
        m.text(surf, f"{a:.0%}", (x + 650, y + 12), ui.TEXT, m.f_small, "midleft")
        m.button(surf, (x + 700, y, m.bw("Remove", 70, m.f_small), max(24, m.f_small.get_linesize() + 6)), "Remove",
                 (lambda j=i: tl.remove(j, [s_.brain for s_ in host.flies])), id=("th_rm", i), font=m.f_small)
        y += 28
    if not tl.expressions:
        m.text(surf, "none yet", (x, y), ui.LABEL, m.f_small)
        y += 22
    y += 6
    seg_w = max(340, 2 * (m.f_small.size("shibire-ts (silence)")[0] + 28))    # 3.0 release review: was a fixed 340 (cut with wider fonts)
    m.segmented(surf, (x, y, seg_w, 28), ["TrpA1 (activate)", "shibire-ts (silence)"], st.t_effector, lambda i: setattr(st, "t_effector", i),
                id="th_eff")
    m.text_field(surf, (x + seg_w + 16, y, 240, 28), st.t_target, lambda v: setattr(st, "t_target", v), id="th_target", limit=40,
                 tip="A cell type (type:DNp01), prefix:KC, a group, or a driver line (line:SS00727). Lab > Genetic toolkit finds lines.")
    m.button_row(surf, x + seg_w + 266, y, 28, [dict(label="Add", click=lambda: _add_expr(st, tl, host), id="th_add", w=90),
                                       dict(label="Clear", click=lambda: tl.clear([s_.brain for s_ in host.flies]), id="th_clear",
                                            style="danger", w=70)], gap=8)
    y += 38
    if st.note:
        m.text(surf, st.note, (x, y), ui.AMBER, m.f_small)
        y += 20
    for e in tl.expressions[:1]:
        ef = tg.effector(e["effector"])
        m.wrapped(surf, f"{ef.label}: {ef.cited}. Game rules: {ef.rule}.", (x, y), rect.w - 60, ui.LABEL, m.f_small, 3)
        y += 52
    note = ("MODEL: temperature changes nothing else (no Q10). shibire-ts really blocks synaptic vesicle recycling at the terminal; "
                    "here it silences the neuron, a coarser intervention. Expression is all-or-nothing in the chosen neurons.")
    n_note = len(ui.Menu.fit_lines(m.f_small, note, rect.w - 220, 3)[0])        # anchored above the buttons whatever the text size
    m.wrapped(surf, note, (x, rect.bottom - 66 - n_note * m.f_small.get_linesize()), rect.w - 220, ui.LABEL, m.f_small, 3)
    m.button(surf, (x, rect.bottom - 58, max(300, m.f_bold.size("Open the escape-vs-temperature assay")[0] + 32), 42), "Open the escape-vs-temperature assay", lambda: _open_assay(m), id="th_assay",
             tip="Thermogenetic activation of DNp01: escape rate vs temperature, expressing flies against a no-expression control.")
    _back(m, rect, "thermo")


def _add_expr(st, tl, host) -> None:
    try:                                             # resolved on the live brain first: a typo is refused, not dropped later
        tl.add("trpa1" if st.t_effector == 0 else "shibire", st.t_target, brain=_brain(host))
        st.note = "added; it applies to every fly on the next frames"
    except Exception as e:
        st.note = str(e)


def _open_assay(m) -> None:
    from kickthefly.lab import lab

    lab._state(m).kind = "thermo_escape"
    m.show("lab_assays")


# ==== Patch clamp ===================================================================================================================
def page_patch(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import patchclamp as pc

    m._surf = surf
    st, host = _st(m), m.host
    y = _title(m, surf, rect, "VIRTUAL PATCH CLAMP",
               "Current-clamp one simulated neuron: membrane potential, spikes, threshold, current steps and an I-F curve.",
               [("MODEL", "a point-neuron LIF unit, NOT real electrophysiology; potential in model units, threshold = 1.0"),
                ("CONNECTOME", "embedded mode: the real synaptic input around the neuron")])
    br = _brain(host)
    if br is None:
        m.text(surf, "needs a running fly", (x0 := rect.x + 24, y + 6), ui.LABEL, m.f_small)
        _back(m, rect, "patch")
        return
    row = st.p_row if st.p_row is not None else None
    if getattr(host, "patch_row", None) is not None:                 # the inspector's Patch button
        row, st.p_row, host.patch_row = int(host.patch_row), int(host.patch_row), None
        _detach(st)
        if 0 <= row < br.n:
            st.p_type = str(br.types[row])
    if row is not None and not 0 <= row < br.n:                      # 3.0 day 2 review: chosen on another brain
        st.note = f"neuron {row} is outside this brain (0-{br.n - 1}); choose it again"
        row = st.p_row = None
        _detach(st)
    x = rect.x + 24
    m.text(surf, "Neuron", (x, y + 14), ui.TEXT, m.f_text, "midleft")
    m.text_field(surf, (x + 70, y, 170, 28), st.p_type, lambda v: setattr(st, "p_type", v), id="pc_type", limit=40,
                 tip="A cell type (DNp01), or any neuron spec: prefix:KC, line:SS00727.")
    m.text(surf, "#", (x + 250, y + 14), ui.LABEL, m.f_text, "midleft")
    m.number_field(surf, (x + 268, y, 54, 28), int(st.p_index), lambda v: setattr(st, "p_index", max(0, int(v))), id="pc_idx",
                   tip="Which neuron of that type, in connectome row order.")
    m.button(surf, (x + 332, y, 110, 28), "Patch it", lambda: _pick(st, host), id="pc_pick")
    ins = getattr(host, "inspect", None)
    if ins:
        m.button(surf, (x + 452, y, 210, 28), f"Use inspected ({str(ins['type'])[:10]})", lambda: _use_inspected(st, host),
                 id="pc_insp", font=m.f_small)
    sw_ = max(330, 2 * m.f_small.size("Embedded (live brain)")[0] + 40)
    if x + 680 + sw_ > rect.right - 24:                     # a narrow menu: the mode switch goes under the neuron row
        y += 36
        mx = x
    else:
        mx = x + 680
    m.segmented(surf, (mx, y, sw_, max(28, m.f_small.get_linesize() + 10)), ["Embedded (live brain)", "Isolated unit"], st.p_mode,
                lambda i: setattr(st, "p_mode", i), id="pc_mode",
                tip="Embedded: the neuron keeps all its synaptic input while you inject current. Isolated: the same LIF equation with "
                    "no synaptic input; every neuron gives the same curve because every neuron is the same unit.")
    y += 38
    if row is None:
        m.text(surf, "Choose a neuron above, or click a neuron in the big brain view (B) and press PATCH.", (x, y), ui.LABEL, m.f_small)
        _back(m, rect, "patch")
        return
    desc = pc.describe_neuron(br, row)
    m.text(surf, f"row {desc['row']}  ·  {desc['type']}  ·  {desc['instance']}  ·  body id {desc['body_id']}  ·  {desc['superclass']}",
           (x, y), ui.INK, m.f_small)
    y += 22
    # live trace
    plot = pygame.Rect(x, y, rect.w - 48, 190)
    ele = st.p_electrode
    if st.p_live and (ele is None or ele.row != row or ele.br is not br):
        _detach(st)
        ele = st.p_electrode = pc.LiveElectrode(br, row).attach()
        ele.set_hold(st.p_hold)
    if ele is not None:
        now = time.perf_counter()
        if now - st.p_last_frame > 0:                               # the menu pauses the game: ask the brain for real-time steps
            k = int(min(0.1, now - st.p_last_frame) / 0.005) if st.p_last_frame else 0
            st.p_last_frame = now
            if getattr(br, "step_requests", 0) == 0 and k:
                br.request_steps(k)
        v, sp, cur = ele.window()
        if len(v):
            tt = (np.arange(len(v)) - len(v)) * pc.DT_MS / 1000.0
            vmax = max(1.3, float(v.max()) * 1.05 if len(v) else 1.3)
            vmin = min(-0.1, float(v.min()) * 1.1)
            _plot(m, surf, plot, [(np.c_[tt, v], (120, 210, 255), 2)], (-ele.n * pc.DT_MS / 1000.0, 0), (vmin, vmax),
                  "seconds", "membrane potential (model units)", [(1.0, ui.AMBER, "threshold 1.0 (model)")],
                  [(tt[i], 1.15, (255, 210, 90)) for i in np.flatnonzero(sp)])
        if ele.done and getattr(ele, "recording", None) is not None:
            st.p_rec, ele.done = ele.recording, False
    elif st.p_rec is not None:
        rec = st.p_rec
        tt = rec.t_ms / 1000.0
        _plot(m, surf, plot, [(np.c_[tt, rec.v], (120, 210, 255), 2), (np.c_[tt, rec.current / max(1e-6, np.abs(rec.current).max()) * 0.4 + 1.6,
                                                                          ], (255, 160, 80), 1)],
              (0, float(tt[-1]) if len(tt) else 1), (min(-0.1, float(rec.v.min()) * 1.1), 2.1), "seconds",
              "membrane potential (model units); orange: injected current (scaled)",
              [(1.0, ui.AMBER, "threshold 1.0 (model)")], [(tt[i], 1.15, (255, 210, 90)) for i in np.flatnonzero(rec.spikes)])
    else:
        pygame.draw.rect(surf, (12, 14, 20), plot, border_radius=8)
        m.text(surf, "press Live to watch the potential, or Run steps", plot.center, ui.LABEL, m.f_small, "center")
    y = plot.bottom + 10
    m.text(surf, "Live", (x, y + 13), ui.TEXT, m.f_text, "midleft")
    m.toggle(surf, (x + 40, y, 50, 26), st.p_live, lambda v: (setattr(st, "p_live", v), None if v else _detach(st)), id="pc_live",
             tip="Watch this neuron live while the brain runs in real time.")
    m.text(surf, "hold", (x + 130, y + 13), ui.TEXT, m.f_small, "midleft")
    m.slider(surf, (x + 160, y, 150, 26), st.p_hold, -0.5, 0.5, 0.01, "{:.2f}", lambda v: _hold(st, v), lambda: None, id="pc_hold",
             tip="Constant holding current (the simulator's current units; 0.5 is the validation suite's activation current).")
    m.text(surf, "step", (x + 330, y + 13), ui.TEXT, m.f_small, "midleft")
    m.slider(surf, (x + 370, y, 150, 26), st.p_amp, -0.5, 1.0, 0.01, "{:.2f}", lambda v: setattr(st, "p_amp", float(v)), lambda: None,
             id="pc_amp")
    m.text(surf, "ms", (x + 530, y + 13), ui.TEXT, m.f_small, "midleft")
    m.number_field(surf, (x + 550, y, 64, 26), int(st.p_dur), lambda v: setattr(st, "p_dur", max(10, min(20000, int(v)))), id="pc_dur")
    m.text(surf, "x", (x + 624, y + 13), ui.TEXT, m.f_small, "midleft")
    m.number_field(surf, (x + 640, y, 44, 26), int(st.p_repeats), lambda v: setattr(st, "p_repeats", max(1, min(100, int(v)))), id="pc_rep")
    m.button(surf, (x + 694, y, m.bw("Run steps", 110), 26), "Run steps", lambda: _run_steps(st, host, br, row), id="pc_run", style="primary",
             tip="Current-clamp steps of this amplitude and duration, repeated; the sweeps are recorded for export.")
    y += 34
    busy = st.p_job is not None and st.p_job["thread"].is_alive()
    nwb = pc.nwb_available()
    m.button_row(surf, x, y, 30, [
        dict(label="Record I-F curve" if not busy else f"running… {st.p_job['label']}", click=lambda: _start_if(st, host, row), id="pc_if",
             enabled=not busy, w=210, tip="Firing rate against injected current, several sweeps per amplitude, in the chosen mode."),
        dict(label="Export CSV", click=lambda: _export_patch(st, m, "csv"), id="pc_csv", enabled=st.p_rec is not None or st.p_curve is not None, w=120),
        dict(label="Export NWB", click=lambda: _export_patch(st, m, "nwb"), id="pc_nwb", enabled=(st.p_rec is not None) and nwb is None, w=120,
             tip=nwb or "Neurodata Without Borders; the potential stays in model units.")])
    if st.p_job is not None and not st.p_job["thread"].is_alive() and st.p_job.get("result"):
        st.p_curve, st.p_job = st.p_job["result"], None
    if st.p_job is not None and st.p_job.get("error"):
        m.text(surf, st.p_job["error"], (x + 490, y + 14), ui.BAD, m.f_small, "midleft")
    y += 38
    if st.p_curve:
        c = st.p_curve
        pts = np.c_[c["amplitudes"], c["rate_hz"]]
        _plot(m, surf, pygame.Rect(x, y, 380, rect.bottom - y - 70), [(pts, (255, 160, 80), 2)],
              (min(c["amplitudes"]), max(c["amplitudes"])), (0, max(10.0, max(c["rate_hz"]) * 1.1)), "injected current (model units)",
              "spikes/s", markers=[(a, r, (255, 160, 80)) for a, r in zip(c["amplitudes"], c["rate_hz"])])
        m.wrapped(surf, f"I-F curve, {c['mode']} mode, {c['repeats']} sweeps x {c['duration_ms']:g} ms per amplitude. "
                        + ("Smallest current that made it fire: " + f"{c['rheobase']:g}" if c["rheobase"] is not None else "It never fired.")
                        + (" In isolation every neuron has this same curve: every unit in the model is identical." if c["mode"] == "isolated" else ""),
                 (x + 400, y + 6), rect.w - 480, ui.TEXT, m.f_small, 4)
    if st.note:
        m.text(surf, st.note, (x + 400, y + 90), ui.AMBER, m.f_small)
    _back(m, rect, "patch")


def _detach(st) -> None:
    if st.p_electrode is not None:
        st.p_electrode.detach()
        st.p_electrode = None


def _hold(st, v) -> None:
    from kickthefly.lab import patchclamp as pc

    try:
        st.p_hold = pc.check_current(v)
    except pc.PatchError as e:
        st.note = str(e)
        return
    if st.p_electrode is not None:
        st.p_electrode.set_hold(st.p_hold)


def leave_patch(m) -> None:
    """Take the live electrode off the brain (its probe and holding current) once Lab > Patch is no longer on screen: the
    game runs again when the menu closes, and a current left on a neuron would change the fly with nothing showing it
    (3.0 day 2 review)."""
    st = getattr(m, "toolkit", None)
    if st is not None and st.p_electrode is not None and m.screen != "lab_patch":
        _detach(st)
        st.p_live = False


def _pick(st, host) -> None:
    from kickthefly.lab import patchclamp as pc

    try:
        st.p_row = pc.pick_neuron(_brain(host), st.p_type, st.p_index)
        st.note = ""
        _detach(st)
    except pc.PatchError as e:
        st.note = str(e)


def _use_inspected(st, host) -> None:
    ins = host.inspect
    if ins:
        st.p_row, st.p_type = int(ins["i"]), str(ins["type"])
        _detach(st)


def _run_steps(st, host, br, row) -> None:
    from kickthefly.lab import patchclamp as pc

    try:
        proto = pc.ClampProtocol([st.p_amp], duration_ms=st.p_dur, repeats=st.p_repeats, holding=st.p_hold)
    except pc.PatchError as e:
        st.note = str(e)
        return
    if st.p_mode == 1:                                              # isolated: no brain needed, instant
        st.p_rec = pc.run_current_clamp(br, row, proto, "isolated", seed=int(br.seed), params=br.sim.p)
        st.note = "isolated unit: no synaptic input"
        return
    st.p_live = True
    if st.p_electrode is None or st.p_electrode.row != row:
        _detach(st)
        st.p_electrode = pc.LiveElectrode(br, row).attach()
    st.p_electrode.set_hold(st.p_hold)
    st.p_electrode.run(proto)
    st.note = f"running {proto.total_steps() * pc.DT_MS / 1000:.1f} s of steps on the live brain"


def _start_if(st, host, row) -> None:
    from kickthefly.core import simcore
    from kickthefly.lab import patchclamp as pc

    mode = "embedded" if st.p_mode == 0 else "isolated"
    amps = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    job = dict(label="starting", result=None, error=None)
    seed = int(_brain(host).seed)
    spec = st.p_type
    idx = int(st.p_index)

    def work():
        try:
            if mode == "isolated":
                br = _brain(host)
                job["result"] = pc.if_curve(br, row, amps, 500.0, 3, "isolated", seed, br.sim.p)
            else:
                job["label"] = "building a brain"
                br = simcore.new_brain(seed=seed)          # its own fresh brain: the live game is not disturbed
                r2 = pc.pick_neuron(br, spec, idx) if row is None else row
                job["label"] = "sweeping"
                job["result"] = pc.if_curve(br, r2, amps, 500.0, 3, "embedded", seed)
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="patch_if", daemon=True)
    job["thread"].start()
    st.p_job = job


def _export_patch(st, m, kind: str) -> None:
    from kickthefly.lab import patchclamp as pc
    from kickthefly.lab import recorder

    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-patch"
    try:
        if kind == "csv":
            if st.p_rec is not None:
                pc.export_csv(st.p_rec, folder / "patch_trace.csv")
            if st.p_curve is not None:
                pc.export_if_csv(st.p_curve, folder / "patch_if_curve.csv")
        else:
            pc.export_nwb(st.p_rec, folder / "patch.nwb")
        m.flash(f"saved to {folder.name} in exports", ui.GOOD)
    except Exception as e:
        m.flash(f"export failed: {e}", ui.BAD)


# ==== Imaging =======================================================================================================================
def page_imaging(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import imaging

    m._surf = surf
    st, host = _st(m), m.host
    il = host.imaging_live
    y = _title(m, surf, rect, "SIMULATED CALCIUM IMAGING",
               "The brain view colored by GCaMP dF/F: spikes convolved with an indicator kernel, plus photon shot noise.",
               [("MODEL", "a forward model on the simulation's spikes, not a measurement"),
                ("LITERATURE", "kernel speeds: see the indicator notes below"), ("CONNECTOME", "which neurons spike")])
    x = rect.x + 24
    # 3.0 release review: controls sat at fixed x up to x + 900 (past an 860 px menu), labels at fixed widths, and the MODEL note at a fixed
    # rect.bottom - 100 whatever was above it (it overlapped the indicator note). Rows now flow and the notes follow the content.
    rh = max(28, m.f_small.get_linesize() + 10)
    lw = max(m.f_text.size(t)[0] for t in ("Imaging mode", "Indicator", "Frame rate", "ROIs")) + 20
    right = rect.right - 24
    cur = [x + lw, y]

    def place(w: int) -> pygame.Rect:
        if cur[0] + w > right and cur[0] > x + lw:
            cur[0], cur[1] = x + lw, cur[1] + rh + 6
        r = pygame.Rect(cur[0], cur[1], w, rh)
        cur[0] += w + 14
        return r

    def row(label: str) -> None:
        nonlocal y
        y = cur[1] if cur[0] == x + lw else cur[1] + rh + 8
        cur[0], cur[1] = x + lw, y
        m.text(surf, label, (x, y + rh // 2), ui.TEXT, m.f_text, "midleft")

    row("Imaging mode")
    m.toggle(surf, place(100), il.on, lambda v: il.set_on(v, _brain(host), getattr(host, "graph", None)), id="im_on",
             tip="The brain view shows dF/F instead of firing. Press B for the big view.")
    if il.error:
        r = place(max(100, right - cur[0]))
        m.wrapped(surf, il.error, (r.x, r.y + 4), r.w, ui.BAD, m.f_small, 2)
    row("Indicator")
    keys = list(imaging.INDICATORS)
    labels = [imaging.INDICATORS[k].name.split(" (")[0] for k in keys]
    m.segmented(surf, place(min(right - (x + lw), max(420, len(keys) * (max(m.f_small.size(t)[0] for t in labels) + 24)))), labels,
                keys.index(il.indicator),
                lambda i: (setattr(il, "indicator", keys[i]), il.set_on(il.on, _brain(host), getattr(host, "graph", None)) if il.on else None),
                id="im_ind")
    row("Frame rate")
    m.slider(surf, place(200), il.fps, 5, 40, 5, "{:.0f} Hz", lambda v: setattr(il, "fps", float(v)), lambda: None, id="im_fps",
             tip="Applies when imaging mode is (re)started. The 5 ms simulation step limits it to 200 Hz.")
    r = place(m.f_small.size("photons/cell/frame")[0] + 8 + 170)
    m.text(surf, "photons/cell/frame", (r.x, r.centery), ui.TEXT, m.f_small, "midleft")
    m.slider(surf, (r.right - 170, r.y, 170, rh), il.f0_photons, 10, 1000, 10, "{:.0f}", lambda v: setattr(il, "f0_photons", float(v)),
             lambda: None, id="im_f0", tip="MODEL parameter: fewer photons, more shot noise.")
    r = place(m.f_small.size("shot noise")[0] + 8 + 100)
    m.text(surf, "shot noise", (r.x, r.centery), ui.TEXT, m.f_small, "midleft")
    m.toggle(surf, (r.right - 100, r.y, 100, rh), il.shot_noise, lambda v: setattr(il, "shot_noise", bool(v)), id="im_noise")
    row("ROIs")
    m.segmented(surf, place(max(230, 2 * m.f_small.size("Brain regions")[0] + 48)), ["Brain regions", "Custom"], 0 if il.roi_mode == "regions" else 1,
                lambda i: setattr(il, "roi_mode", "regions" if i == 0 else (st.i_roi_text or "type:DNp01")), id="im_roi")
    if il.roi_mode != "regions":
        m.text_field(surf, place(360), il.roi_mode, lambda v: setattr(il, "roi_mode", v.strip() or "regions"), id="im_spec",
                     limit=60, tip="Neuron specs separated by ;  for example  type:DNp01;prefix:KC;line:SS00727 . One ROI each.")
    r = place(m.f_small.size("keep frames")[0] + 8 + 100)
    m.text(surf, "keep frames", (r.x, r.centery), ui.TEXT, m.f_small, "midleft")
    m.toggle(surf, (r.right - 100, r.y, 100, rh), il.keep_frames, lambda v: setattr(il, "keep_frames", bool(v)), id="im_keep",
             tip="Keep the rendered frames (last ~15 s) so they can be exported as TIFF or NWB ImageSeries.")
    y = cur[1] + rh + 12
    ind = imaging.INDICATORS[il.indicator]
    ind_text = f"{ind.name}: {ind.source}. {ind.verified}"
    model_text = ("MODEL: each spike adds the same kernel; every neuron in an ROI is equally bright; F0 is a running mean (30 s), so a steady "
                  "rate reads as 0 and only changes show; no bleaching, motion, scattering or neuropil. dF/F per spike (0.2) and the "
                  "photon budget are game parameters.")
    lh_s = m.f_small.get_linesize()
    notes_h = (len(ui.Menu.fit_lines(m.f_small, ind_text, rect.w - 200, 6)[0]) + len(ui.Menu.fit_lines(m.f_small, model_text, rect.w - 200, 4)[0])) * lh_s + 8
    s = il.session
    plot = pygame.Rect(x, y, rect.w - 48, max(60, min(190, rect.bottom - 70 - notes_h - 46 - y)))
    if s is not None and len(s.t) > 2:
        reduced = bool(host.cfg["access.reduced_flashing"])
        data = np.array(s.true_dff if reduced else s.dff)
        t = np.array(s.t)
        t = t - t[-1]
        cols = [(255, 170, 60), (90, 200, 255), (120, 230, 120), (240, 110, 200), (250, 220, 90), (170, 140, 255), (255, 255, 255)]
        series = [(np.c_[t, data[:, k]], cols[k % len(cols)], 2) for k in range(min(data.shape[1], 7))]
        top = max(0.5, float(np.nanpercentile(data[:, :7], 99)) * 1.1)
        _plot(m, surf, plot, series, (float(t[0]), 0.0), (-0.1, top), "seconds", "dF/F (fraction)")
        lx = plot.x + 150
        for k in range(min(data.shape[1], 7)):
            m.text(surf, s.roi_names[k][:16], (lx, rect_top(plot)), cols[k % len(cols)], m.f_small)
            lx += 120
    else:
        pygame.draw.rect(surf, (12, 14, 20), plot, border_radius=8)
        m.text(surf, "turn Imaging mode on to see the ROI traces", plot.center, ui.LABEL, m.f_small, "center")
    y = plot.bottom + 8
    nwb = imaging_nwb_reason()
    xe, _ = m.button_row(surf, x, y, 30, [
        dict(label="Export CSV", click=lambda: _export_imaging(st, m, host, "csv"), id="im_csv", enabled=s is not None and len(s.t) > 1, w=130),
        dict(label="Export NWB", click=lambda: _export_imaging(st, m, host, "nwb"), id="im_nwb", enabled=s is not None and len(s.t) > 1 and nwb is None,
             w=130, tip=nwb or "RoiResponseSeries (dF/F and photons) and an ImageSeries."),
        dict(label="Export TIFF", click=lambda: _export_imaging(st, m, host, "tiff"), id="im_tiff", enabled=bool(il.frames), w=130,
             tip="A multi-page TIFF of the rendered view. Turn on 'keep frames' first.")])
    if st.i_msg:
        m.text(surf, st.i_msg, (xe + 10, y + 15), ui.AMBER, m.f_small, "midleft")
    y += 38
    y = m.wrapped(surf, ind_text, (x, y), rect.w - 200, ui.LABEL, m.f_small, 6) + 8      # clear of the Back button
    m.wrapped(surf, model_text, (x, y), rect.w - 200, ui.LABEL, m.f_small, 4)
    _back(m, rect, "imaging")


def imaging_nwb_reason():
    from kickthefly.lab import nwbexport

    return nwbexport.available()


def _export_imaging(st, m, host, kind: str) -> None:
    from kickthefly.lab import imaging, recorder

    res = host.imaging_live.result()
    if res is None:
        st.i_msg = "nothing recorded yet"
        return
    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-imaging"
    try:
        if kind == "csv":
            imaging.export_csv(res, folder / "imaging_roi.csv")
        elif kind == "nwb":
            imaging.export_nwb(res, folder / "imaging.nwb")
        else:
            imaging.export_tiff(res.frames, folder / "imaging_view.tif", res.meta)
        st.i_msg = f"saved to {folder.name}"
        m.flash(f"saved to {folder.name} in exports", ui.GOOD)
    except Exception as e:
        st.i_msg = f"export failed: {e}"


# ==== Pharmacology ===================================================================================================================
def page_pharm(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.core import simcore
    from kickthefly.lab import pharmacology as ph
    from kickthefly.sim.wiring import Wiring

    m._surf = surf
    st, host = _st(m), m.host
    y = _title(m, surf, rect, "PHARMACOLOGY",
               "Drugs scale synapses by the dataset's predicted transmitter. Picrotoxin moved here from Robustness.",
               [("MODEL", "PREDICTION: a synaptic scale only, no receptor or kinetic model"),
                ("CONNECTOME", "which synapses: transmitter predictions and their confidence")])
    x = rect.x + 24
    keys = list(ph.DRUGS)
    # 3.0 release review: the label column, the control widths and the table's columns and row step were fixed pixels; at larger text
    # "Low-confidence predictions" ran into its buttons and the table rows overlapped. All measured from the font now.
    fh = max(30, m.f_small.get_linesize() + 10)
    lw = max(m.f_text.size(t)[0] for t in ("Drug", "Dose", "Low-confidence predictions")) + 16
    avail = rect.right - 24 - (x + lw)
    m.text(surf, "Drug", (x, y + fh // 2), ui.TEXT, m.f_text, "midleft")
    y = m.flow_buttons(surf, x + lw, y, rect.right - 24, [(ph.DRUGS[k].label, (lambda k=k: setattr(st, "d_drug", k)), ("ph_drug", k), st.d_drug == k,
                                                           "Octopamine and dopamine modulation are left out: those neurons' synapses are not in the "
                                                           "simulated matrix and there is no existing gain to scale honestly.") for k in keys], h=fh) + 10
    m.text(surf, "Dose", (x, y + fh // 2), ui.TEXT, m.f_text, "midleft")
    m.slider(surf, (x + lw, y, min(320, avail), fh), st.d_dose, 0.0, 1.0, 0.05, "{:.0%}", lambda v: setattr(st, "d_dose", float(v)), lambda: None,
             id="ph_dose", tip="Blockers scale by (1 - dose). The GABA-A agonist scales by 1 + dose (x2 at full dose, a game rule).")
    scale = ph.scale_for(st.d_drug, st.d_dose)
    sx = x + lw + min(320, avail) + 20
    if sx + m.f_text.size(f"synapse weight scale x{scale:.2f}")[0] > rect.right - 24:     # no room beside the slider: under it
        y += fh + 4
        sx = x + lw
    m.text(surf, f"synapse weight scale x{scale:.2f}", (sx, y + fh // 2), ui.INK, m.f_text, "midleft")
    y += fh + 8
    m.text(surf, "Low-confidence predictions", (x, y + fh // 2), ui.TEXT, m.f_text, "midleft")
    sw = min(240, avail // 2)
    m.segmented(surf, (x + lw, y, sw, fh), ["Include", "Exclude"], 0 if st.d_low else 1, lambda i: setattr(st, "d_low", i == 0), id="ph_low",
                tip="Exclude leaves the drug off neurons whose transmitter is a prediction below the cut (and those with no confidence); "
                    "measured transmitters always count.")
    cx = x + lw + sw + 16
    m.text(surf, "cut", (cx, y + fh // 2), ui.TEXT, m.f_text, "midleft")
    cx += m.f_text.size("cut")[0] + 10
    m.slider(surf, (cx, y, max(160, min(200, rect.right - 24 - cx)), fh), st.d_cut, 0.5, 0.95, 0.05, "{:.2f}", lambda v: setattr(st, "d_cut", float(v)),
             lambda: None, id="ph_cut", enabled=not st.d_low, tip="A predicted transmitter below this confidence is 'low confidence' (a game rule).")
    y += fh + 12
    try:
        g = simcore.pack()[0]
        a = ph.affected(g, st.d_drug, st.d_dose, st.d_low, st.d_cut)
    except Exception as e:
        m.wrapped(surf, f"needs an adult brain pack with transmitter predictions: {e}", (x, y), rect.w - 48, ui.BAD, m.f_small, 3)
        _back(m, rect, "pharm")
        return
    heads = ["confidence level", "neurons", "connections", "synapses", "scaled?"]
    cols = [[r["level"] for r in a["rows"]], [f"{r['neurons']:,}" for r in a["rows"]], [f"{r['connections']:,}" for r in a["rows"]],
            [f"{r['synapses']:,}" for r in a["rows"]], ["yes", "no", "-"]]
    xs, cxx = [], x
    for h, vals in zip(heads, cols):
        xs.append(cxx)
        cxx += max(m.f_small.size(v)[0] for v in [h, *vals]) + 28
    rh = m.f_small.get_linesize()
    for hx, label in zip(xs, heads):
        m.text(surf, label, (hx, y), ui.LABEL, m.f_small)
    y += rh + 2
    for r in a["rows"]:
        col = ui.TEXT if r["applied"] else ui.LABEL
        m.text(surf, r["level"], (xs[0], y), col, m.f_small)
        m.text(surf, f"{r['neurons']:,}", (xs[1], y), col, m.f_small)
        m.text(surf, f"{r['connections']:,}", (xs[2], y), col, m.f_small)
        m.text(surf, f"{r['synapses']:,}", (xs[3], y), col, m.f_small)
        m.text(surf, "-" if not r["synapses"] else "yes" if r["synapses_applied"] > 0 else "no", (xs[4], y),
               ui.GOOD if r["synapses_applied"] else ui.LABEL, m.f_small)
        y += rh
    y = m.wrapped(surf, f"affected: {a['synapses_applied']:,} of {a['synapses']:,} synapses  ·  {a['connections_applied']:,} connections  ·  "
                        f"{a['neurons_applied']:,} presynaptic neurons", (x, y + 4), rect.w - 48, ui.INK, m.f_small, 2) - 16
    y += 28
    live = getattr(host, "wiring", None) or Wiring()
    new = ph.wiring_for({st.d_drug: st.d_dose}, st.d_low, st.d_cut, base=live) if st.d_dose else Wiring(live.min_synapses, live.flip_rows, live.inhibition_scale)
    applied = live == new
    job = st.d_job
    busy = job is not None and job["thread"].is_alive()
    apply_label = "Applied" if applied else "Apply to every fly"
    x_dr = x + m.bw(apply_label, 250) + 10 + m.bw("Wash out", 200) + 10          # the report button sits under this one
    x_after, _ = m.button_row(surf, x, y, 36, [
        dict(label=apply_label, click=lambda: host.set_wiring(new), id="ph_apply", style="primary", w=250,
             enabled=not applied and not getattr(host, "wiring_busy", ""),
             tip="Reversibly scales those synapses on the live flies (the wiring machinery of Lab > Robustness)."),
        dict(label="Wash out", click=lambda: host.set_wiring(Wiring(live.min_synapses, live.flip_rows, live.inhibition_scale)), id="ph_wash", w=200,
             enabled=bool(live.nt_scales) and not getattr(host, "wiring_busy", "")),
        dict(label="Measure dose-response" if not busy else "measuring…", click=lambda: _start_dr(st), id="ph_dr", enabled=not busy, w=250,
             tip="Whole-brain firing (Hz) at 0, 25, 50, 75 and 100% dose of this drug, seed 1000, 1 s each, on fresh brains.")])
    if job is not None and not busy:
        st.d_dr, st.d_job = job.get("result") or st.d_dr, None
    if live.nt_scales:
        m.wrapped(surf, f"live on the flies: {live.label()}", (x_after, y + 4), rect.right - 24 - x_after, ui.AMBER, m.f_small, 2)
    y += 46
    if st.d_drug == "picrotoxin" and st.d_low:
        from kickthefly.lab import labwiring

        wst = labwiring._st(m)
        j2 = wst.inhibition_job
        if j2 is not None and not j2["thread"].is_alive():
            wst.inhibition_result = j2.get("result") or wst.inhibition_result
            wst.inhibition_job = None
        if wst.inhibition_job is None:
            m.button(surf, (x_dr, y - 46 + 42, m.bw("Firing-rate distribution report", 250, m.f_small), max(26, m.f_small.get_linesize() + 6)),
                     "Firing-rate distribution report", lambda: labwiring._start_inhibition(m, wst, 1.0 - st.d_dose),
                     id="ph_report", font=m.f_small, tip="The original picrotoxin report: firing-rate histogram before and after, seed 1000.")
        else:
            m.text(surf, "report: " + wst.inhibition_job["label"], (x_dr, y - 46 + 50), ui.LABEL, m.f_small)
        if wst.inhibition_result:
            labwiring._draw_inhibition(m, surf, pygame.Rect(x, y + 6, rect.w - 48, rect.bottom - y - 80), wst, wst.inhibition_result)
            _back(m, rect, "pharm")
            return
    if st.d_dr:
        rows = st.d_dr["rows"]
        pts = np.c_[[r["dose"] for r in rows], [r["mean_hz"] for r in rows]]
        _plot(m, surf, pygame.Rect(x, y, 420, max(120, rect.bottom - y - 80)), [(pts, (255, 160, 80), 2)], (0, 1.0),
              (0, max(1.0, float(pts[:, 1].max()) * 1.1)), "dose", "mean firing (Hz)", markers=[(d, h, (255, 160, 80)) for d, h in pts])
        m.wrapped(surf, f"{ph.drug(st.d_dr['drug']).label}: mean whole-brain firing over {st.d_dr['steps'] * 0.005:.1f} s, seed "
                        f"{st.d_dr['seeds']}. The simulator's slow global gain pushes back on anything that changes overall synaptic strength, "
                        f"so this is the drug plus a little of that compensation.", (x + 440, y + 6), rect.w - 500, ui.LABEL, m.f_small, 6)
    _back(m, rect, "pharm")


def _start_dr(st) -> None:
    from kickthefly.lab import pharmacology as ph

    job = dict(result=None, error=None)
    drug, low, cut = st.d_drug, st.d_low, st.d_cut

    def work():
        try:
            job["result"] = ph.dose_response(drug, include_low_confidence=low, cut=cut, seeds=(1000,), steps=200)
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="pharm_dr", daemon=True)
    job["thread"].start()
    st.d_job = job


PAGES = {"lab_genetics": page_genetics, "lab_thermo": page_thermo, "lab_patch": page_patch, "lab_imaging": page_imaging,
         "lab_pharm": page_pharm}
