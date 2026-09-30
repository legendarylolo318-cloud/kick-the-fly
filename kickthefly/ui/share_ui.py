"""Esc > Share: make a share code for what you have set up, or paste one and see what it would change before it does.

A share code (kickthefly/core/sharecode.py) carries a brain surgery, a tool loadout, a protocol, a challenge setup or a set of
Lab parameters. GAME RULE, all of it: a container for settings the game already has. Import previews first and changes nothing
until you press Apply; a code that is damaged, from a newer version, or names something this version or this brain doesn't
have is refused with the reason. A code too big to copy is written to a .ktfshare file instead, and Import takes the file's
path as well.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pygame

from kickthefly.core import clipboard
from kickthefly.core import sharecode as sc
from kickthefly.core.i18n import tr
from kickthefly.ui import menu as ui

KINDS = ("surgery", "loadout", "lab", "protocol", "challenge")
KIND_NAMES = {"surgery": "Brain surgery", "loadout": "Tool loadout", "lab": "Lab parameters", "protocol": "Protocol",
              "challenge": "Challenge setup"}


def install(menu) -> None:
    menu.pages["share"] = page_share


class _State:
    def __init__(self):
        self.tab = "export"
        self.kind = "surgery"
        self.loadout_name = "Shared loadout"
        self.protocol_file = None
        self.challenge = "tmaze"
        self.text = ""
        self.parsed = None            # (text, Code | None, reason | None, Preview | None)
        self.last = ""
        self.code_cache = None


def _state(m) -> _State:
    st = getattr(m, "_share_state", None)
    if st is None:
        st = m._share_state = _State()
    return st


def make_context(host) -> sc.Context:
    """What this running game knows, for validating and previewing a code."""
    from kickthefly.core import loadout as lo
    from kickthefly.game import kick_the_fly as k
    from kickthefly.lab import challenges, lab

    x3 = host.x3
    if getattr(x3, "_type_names", None) is None or x3._type_names[0] is not host.brain:
        x3._type_names = (host.brain, set(np.unique(np.asarray(host.brain.types).astype(str))) - {""})
    arenas = {a for a in k.ARENAS if host.three_d or a not in k.OUTDOOR_ARENAS}
    return sc.Context(
        tools=set(lo.BY_NAME), lab_params={n: (p[4], p[5]) for n, p in lab.BY_NAME.items()},
        current_params={n: host.lab_params.get(n, lab.DEFAULTS[n]) for n in lab.DEFAULTS},
        surgery_labels={label for label, _ in k.SURGERY}, type_names=x3._type_names[1],
        challenges=set(challenges.CLASSES), arenas=arenas, current_surgery=sc.payload_for_surgery(host),
        current_loadout=list(host.cfg.loadout["custom"]), saved_loadouts=[i["name"] for i in host.cfg.loadout["saved"]],
        max_saved_loadouts=lo.MAX_SAVED, protocol_names={f.stem for f in lab.protocol_files()},
        larva=bool(host.is_larva), lab=bool(host.cfg.lab), brain=x3.brain_name)


def _payload(host, st: _State) -> tuple[dict | None, str | None]:
    """(payload, why not) for the selected kind from the game's current state."""
    from kickthefly.lab import challenges, lab

    kind = st.kind
    if kind == "surgery":
        p = sc.payload_for_surgery(host)
        return (p, None) if (p["groups"] or p["types"]) else (None, tr("No surgery is switched on."))
    if kind == "loadout":
        return {"name": st.loadout_name[:24], "tools": list(host.loadout.tools)}, None
    if kind == "lab":
        mod = lab.modified(host.lab_params)
        return ({"params": {k: float(v) for k, v in mod.items()}}, None) if mod else (None, tr("No Lab parameter is changed."))
    if kind == "protocol":
        files = lab.protocol_files()
        f = next((x for x in files if x.name == st.protocol_file), None)
        if f is None:
            return None, tr("Pick a protocol.")
        try:
            import yaml

            return {"protocol": yaml.safe_load(f.read_text(encoding="utf-8"))}, None
        except Exception as e:
            return None, f"{f.name}: {e}"
    if kind == "challenge":
        p = {"challenge": st.challenge, "seed": int(host.cfg["brain.seed"]), "arena": host.cfg["brain.arena"]}
        sg = sc.payload_for_surgery(host)
        if sg["groups"] or sg["types"]:
            p["surgery"] = sg
        mod = lab.modified(host.lab_params)
        if mod:
            p["params"] = {k: float(v) for k, v in mod.items()}
        return p, None
    return None, "?"


def _share_dir():
    from kickthefly.core import paths

    return paths.get().data_dir / "share"


def page_share(m, surf, rect, mouse) -> None:
    host = m.host
    st = _state(m)
    m.text(surf, tr("SHARE"), (rect.x + 24, rect.y + 14), ui.INK, m.f_head)
    m.chip(surf, (rect.x + 120, rect.y + 18), "GAME RULE")
    m.text(surf, tr("A code carries settings the game already has. Import shows what it would change first."),
           (rect.x + 222, rect.y + 20), ui.DIM, m.f_small)
    for i, (key, label) in enumerate((("export", tr("Make a code")), ("import", tr("Import a code")))):
        m.button(surf, (rect.x + 24 + i * 190, rect.y + 52, 180, 36), label, (lambda k=key: setattr(st, "tab", k)),
                 active=st.tab == key, id=("share-tab", key))
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), tr("Back"), m.back, style="primary", id=("share", "back"))
    body = pygame.Rect(rect.x + 24, rect.y + 100, rect.w - 48, rect.h - 100 - 70)
    (_page_export if st.tab == "export" else _page_import)(m, surf, body, st, host)


def _page_export(m, surf, body, st, host) -> None:
    x, y = body.x, body.y
    m.text(surf, tr("What to share"), (x, y), ui.LABEL, m.f_small)
    bw = 150
    for i, k in enumerate(KINDS):
        m.button(surf, (x + i * (bw + 8), y + 20, bw, 34), tr(KIND_NAMES[k]), (lambda k=k: setattr(st, "kind", k)),
                 active=st.kind == k, id=("share-kind", k))
    y += 70
    if st.kind == "loadout":
        m.text(surf, tr("Name"), (x, y + 5), ui.TEXT, m.f_small)
        m.text_field(surf, (x + 60, y, 260, 28), st.loadout_name, lambda v: setattr(st, "loadout_name", v.strip()[:24] or "Shared loadout"),
                     id="share-lname", limit=24)
        y += 38
    elif st.kind == "protocol":
        from kickthefly.lab import lab

        files = lab.protocol_files()
        m.text(surf, tr("Protocol file") + ":", (x, y + 5), ui.TEXT, m.f_small)
        for i, f in enumerate(files[:6]):
            col, row = i % 3, i // 3
            m.button(surf, (x + 110 + col * 250, y + row * 34, 240, 30), f.stem[:30],
                     (lambda f=f: setattr(st, "protocol_file", f.name)), active=st.protocol_file == f.name,
                     id=("share-proto", f.name))
        y += 72
    elif st.kind == "challenge":
        from kickthefly.lab import challenges

        m.text(surf, tr("Challenge") + ":", (x, y + 5), ui.TEXT, m.f_small)
        for i, key in enumerate(challenges.CLASSES):
            m.button(surf, (x + 100 + i * 140, y, 132, 30), key.replace("_", " "), (lambda key=key: setattr(st, "challenge", key)),
                     active=st.challenge == key, id=("share-chal", key))
        y += 38
        m.text(surf, tr("Includes the seed, arena, any surgery and any changed Lab parameters."), (x, y), ui.DIM, m.f_small)
        y += 22
    payload, why = _payload(host, st)
    box = pygame.Rect(x, y + 6, body.w, 150)
    pygame.draw.rect(surf, (12, 14, 20), box, border_radius=8)
    pygame.draw.rect(surf, ui.BORDER, box, 1, border_radius=8)
    code, err, too_big = None, why, False
    if payload is not None:
        ck = (st.kind, json.dumps(payload, sort_keys=True, default=str))
        if st.code_cache is None or st.code_cache[0] != ck:
            try:
                st.code_cache = (ck, sc.encode(st.kind, payload), None, False)
            except sc.CodeTooLarge as e:
                st.code_cache = (ck, None, str(e), True)
            except sc.ShareError as e:
                st.code_cache = (ck, None, str(e), False)
        _, code, err, too_big = st.code_cache
    if code:
        mono = pygame.font.SysFont("dejavusansmono,consolas,couriernew", 13)
        lines, line = [], ""
        for part in code.split("-"):
            if mono.size(line + part + "-")[0] > box.w - 24 and line:
                lines.append(line)
                line = ""
            line += part + "-"
        lines.append(line.rstrip("-"))
        for i, ln in enumerate(lines[:8]):
            m.text(surf, ln, (box.x + 12, box.y + 10 + i * 17), ui.INK, mono)
        m.text(surf, f"{len(code):,} " + tr("characters"), (box.right - 10, box.bottom - 18), ui.DIM, m.f_small, "topright")
    else:
        m.wrapped(surf, err or "", (box.x + 12, box.y + 12), box.w - 24, ui.AMBER if too_big else ui.LABEL, m.f_text, 4)
    by = box.bottom + 14
    m.button(surf, (x, by, 180, 40), tr("Copy code"), lambda: _copy(m, code), style="primary", id="share-copy",
             enabled=bool(code), tip=tr("Puts the code on your clipboard."))
    m.button(surf, (x + 190, by, 200, 40), tr("Save as a file"), lambda: _save_file(m, st, payload), id="share-file",
             enabled=payload is not None and err is None or too_big,
             tip=tr("Writes a .ktfshare file in your data folder's share folder (for a code too big to paste)."))
    if st.last:
        m.text(surf, st.last, (x, by + 52), ui.GOOD, m.f_small)


def _copy(m, code) -> None:
    if not code:
        return
    if clipboard.put_text(code):
        _state(m).last = tr("Copied. Paste it anywhere; Import reads it back.")
    else:
        _state(m).last = tr("Couldn't reach the clipboard here: select the code above, or use Save as a file.")


def _save_file(m, st, payload) -> None:
    if payload is None:
        return
    try:
        text = sc.encode(st.kind, payload, limit=None)
        p = sc.export_file(text, _share_dir(), f"{st.kind}-{time.strftime('%Y%m%d-%H%M%S')}")
        st.last = tr("Saved") + f": {p}"
    except sc.ShareError as e:
        st.last = str(e)


def _page_import(m, surf, body, st, host) -> None:
    x, y = body.x, body.y
    m.text(surf, tr("Paste a share code (or the path of a .ktfshare file)"), (x, y), ui.LABEL, m.f_small)
    m.text_field(surf, (x, y + 22, body.w - 130, 32), st.text, lambda v: setattr(st, "text", v), id="share-in", limit=6000,
                 compact=True, tip=tr("Click, then Ctrl+V. Spaces and line breaks are ignored."))
    m.button(surf, (body.right - 120, y + 22, 120, 32), tr("Paste"), lambda: _paste(st), id="share-paste",
             tip=tr("Reads your clipboard into the box."))
    y += 70
    text = st.text.strip()
    if st.parsed is None or st.parsed[0] != text:
        code = reason = pv = None
        if text:
            try:
                code = sc.read_any(text)
                reason = sc.validate(code, make_context(host))
                pv = sc.preview(code, make_context(host)) if reason is None else None
            except sc.ShareError as e:
                reason = str(e)
        st.parsed = (text, code, reason, pv)
    _, code, reason, pv = st.parsed
    if not text:
        m.text(surf, tr("Nothing pasted yet."), (x, y), ui.DIM, m.f_text)
        return
    if reason:
        m.chip(surf, (x, y), "REFUSED")
        m.wrapped(surf, reason, (x, y + 28), body.w, ui.BAD, m.f_text, 4)
        return
    m.text(surf, pv.title, (x, y), ui.INK, m.f_bold)
    m.text(surf, tr("Applying this will:"), (x, y + 26), ui.LABEL, m.f_small)
    yy = y + 46
    for ln in pv.lines[:10]:
        yy = m.wrapped(surf, "- " + ln, (x + 8, yy), body.w - 16, ui.TEXT, m.f_small, 2) + 1
    for w in pv.warnings:
        yy = m.wrapped(surf, w, (x + 8, yy + 2), body.w - 16, ui.AMBER, m.f_small, 3)
    m.button(surf, (x, body.bottom - 46, 200, 42), tr("Apply"), lambda: _apply(m, st, host, code), style="primary",
             id="share-apply", tip=tr("Does what the list above says, now."))
    if st.last:
        m.text(surf, st.last, (x + 214, body.bottom - 34), ui.GOOD, m.f_small)


def _paste(st) -> None:
    t = clipboard.get_text()
    st.text = "".join((t or "").split())[:6000]


def _apply(m, st, host, code) -> None:
    try:
        res = sc.apply(code, host)
    except Exception as e:                        # a code that validated but met a state it can't handle: say so, change nothing more
        from kickthefly.core.crash import log

        log.exception("share code apply failed")
        st.last = f"{tr('Could not apply it')}: {e}"
        return
    st.last = tr("Done") + ": " + "; ".join(res)
    st.text, st.parsed = "", None
    host.note("SHARE    imported " + sc.KIND_LABEL[code.kind].lower(), source="rule")
    if code.kind == "challenge":
        pass                                      # start_challenge closed the menu
