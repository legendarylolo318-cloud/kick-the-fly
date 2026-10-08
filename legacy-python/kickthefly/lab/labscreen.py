"""Lab pages for the whole-brain screens (3.1.0 tasks 6 and 7): run one, browse a finished one.

The run is lab/screens.py (the same one `--activation-screen` / `--knockout-screen` and a `screen:` protocol do); this is the front end:
a progress bar (the run is resumable, so closing the page or the game is safe), a table you can search and sort, and a detail panel for the
selected row with the numbers, the matched-control statistics, the descending neurons it moved, and two buttons that carry the row into
the game: put the perturbation on the live flies (the brain surgery switch), or open the neuron inspector on one of its neurons.
MODEL PREDICTION, shown on the page.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pygame

from kickthefly.ui import menu as ui

KINDS = {
    "activation": dict(title="ACTIVATION SCREEN", page="lab_activation", folder="activation-screen",
                       blurb="Hold each cell type driven in turn on held-out seeds, with matched random controls, and see which descending neurons "
                             "respond and which behavior readout moves. Search a type, sort by effect, click a row to see its numbers or inspect its neurons."),
    "knockout": dict(title="KNOCKOUT SCREEN", page="lab_knockout", folder="knockout-screen",
                     blurb="For each validated behavior, silence cell types one at a time (or in ranked batches first) and rank them by how much the "
                           "response drops against matched random lesions. Click a row to see its numbers or inspect its neurons."),
}
ACT_COLS = (("rank", "#", 8), ("type", "cell type", 44), ("neurons", "neurons", 170), ("behavior_call", "what it does to the readouts", 236),
            ("behavior_effect_hz", "change", 478), ("behavior_q", "q", 548), ("dn_responders", "DN types", 610))
KO_COLS = (("behavior", "behavior", 8), ("candidate", "cell type(s)", 190), ("drop_share", "drop", 400), ("q", "q", 470),
           ("neurons", "neurons", 540), ("rank", "rank", 610))


def _st(m, kind: str):
    from kickthefly.lab import lab

    st = lab._state(m)
    key = f"scr_{kind}"
    if not hasattr(st, key):
        setattr(st, key, dict(res=None, job=None, search="", sort=None, desc=True, sel=None, seeds=8, size=60, loaded=False, error=None,
                              flash=None))
    return getattr(st, key)


def folder_of(kind: str):
    from kickthefly.lab import recorder

    return recorder.exports_dir() / KINDS[kind]["folder"]


def page(kind: str):
    def _page(m: ui.Menu, surf, rect, mouse) -> None:
        _draw(m, surf, rect, kind)
    return _page


def _draw(m, surf, rect, kind: str) -> None:
    from kickthefly.lab import screens

    cfg, st, host = KINDS[kind], _st(m, kind), m.host
    ui.TAG_COLORS.setdefault(screens.TAG, (150, 120, 220))
    m.text(surf, cfg["title"], (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.chip(surf, (rect.x + 40 + m.f_head.size(cfg["title"])[0], rect.y + 22), screens.TAG)
    y = m.subtitle(surf, rect, cfg["blurb"])
    if not st["loaded"]:                                   # what an earlier session finished is there to browse
        st["loaded"] = True
        st["res"] = screens.load(folder_of(kind)) or None
        if st["res"] and st["res"].get("kind") != kind:
            st["res"] = None
    job = st["job"]
    if job is not None and not job["thread"].is_alive():
        if job.get("result"):
            st["res"], st["sel"] = job["result"], None
        st["error"] = job.get("error")
        st["job"] = job = None
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, y + 6, rect.w - 220, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, y + 6, max(8, int((rect.w - 220) * frac)), 12), border_radius=6)
        m.text(surf, f"{job['label'][:96]}  ·  {job['done']}/{job['total']} jobs  ·  {time.time() - job['t0']:.0f}s",
               (rect.x + 24, y + 24), ui.LABEL, m.f_small)
        m.button(surf, (rect.right - 180, y - 2, 156, 32), "Stop", lambda: job["stop"].set(), id=("scr", kind, "stop"),
                 tip="Stops after the running jobs finish. Everything finished is kept: run it again and it continues.")
        m.text(surf, "closing this page is safe: finished conditions are written as they go", (rect.x + 24, y + 42), ui.DIM, m.f_small)
        y += 60
    else:
        label = "Run it again (resumes)" if st["res"] else "Run the screen"
        w = m.bw(label, 190)
        m.button(surf, (rect.x + 24, y, w, 34), label, lambda: _start(m, kind, st), id=("scr", kind, "run"), style="primary",
                 tip="Runs on this machine's CPU processes (or the GPU's batched brains with --backend gl). Minutes for a small selection; "
                     "the whole connectome is hours: use the command line (--activation-screen / --knockout-screen) for that.")
        x = rect.x + 40 + w
        m.slider(surf, (x, y + 2, 190, 30), st["seeds"], 3, 8, 1, "{:.0f} seeds", lambda v: st.__setitem__("seeds", int(v)), lambda: None,
                 id=("scr", kind, "seeds"), tip="Held-out seeds 4000 upward. Fewer seeds run faster and detect less (eight is the default).")
        m.slider(surf, (x + 210, y + 2, 230, 30), st["size"], 10, 400 if kind == "activation" else 60, 10 if kind == "activation" else 5,
                 "{:.0f} types" if kind == "activation" else "{:.0f} cand.", lambda v: st.__setitem__("size", int(v)), lambda: None,
                 id=("scr", kind, "size"), tip=("How many cell types, in name order, a quick run covers." if kind == "activation" else
                                                "How many ranked candidate cell types to silence for each behavior."))
        y += 44
        if st["error"]:
            m.text(surf, f"error: {st['error']}", (rect.x + 24, y), ui.BAD, m.f_small)
            y += 20
    res = st["res"]
    area = pygame.Rect(rect.x + 16, y, rect.w - 32, rect.bottom - y - 66)
    if res:
        _table(m, surf, area, kind, st, res, host)
    else:
        m.text(surf, "No results yet. Run the screen, or point --out at a folder to browse an earlier run.", (area.x + 8, area.y + 6), ui.LABEL, m.f_small)
    if res:
        m.button(surf, (rect.x + 24, rect.bottom - 58, m.bw("Export", 150), 42), "Export", lambda: _export(m, kind, res), id=("scr", kind, "export"),
                 tip="Copies the CSV (and Parquet) files into a new folder in exports.")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("scr", kind, "back"))


# --- running -----------------------------------------------------------------------------------------------------------------------------
def _start(m, kind: str, st: dict) -> None:
    from kickthefly.lab import labjobs, screens

    stop = threading.Event()
    job = dict(done=0, total=1, label="starting", result=None, error=None, t0=time.time(), stop=stop)
    folder = folder_of(kind)
    seeds = screens.SCREEN_SEEDS[:int(st["seeds"])]
    quick = int(st["size"])

    def progress(d, n, label):
        job.update(done=d, total=n, label=label)
        if stop.is_set():
            raise KeyboardInterrupt

    def work():
        try:
            if kind == "activation":
                job["result"] = screens.run_activation(folder, seeds=seeds, max_types=quick, workers=labjobs.default_workers(), progress=progress,
                                                       stream=None)
            else:
                job["result"] = screens.run_knockout(folder, seeds=seeds, top=quick, workers=labjobs.default_workers(), progress=progress, stream=None)
        except KeyboardInterrupt:
            job["error"] = "stopped; what finished is kept (run it again to continue)"
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name=f"screen-{kind}", daemon=True)
    job["thread"].start()
    st["job"], st["error"] = job, None


# --- the table -----------------------------------------------------------------------------------------------------------------------------
def _rows(kind: str, st: dict, res: dict) -> list[dict]:
    rows = res["rows"] if kind == "activation" else [r for r in res["rows"]]
    q = st["search"].strip().lower()
    if q:
        keys = ("type", "behavior_call", "top_dn", "superclass") if kind == "activation" else ("candidate", "behavior")
        rows = [r for r in rows if any(q in str(r.get(k, "")).lower() for k in keys)]
    if st["sort"]:
        k = st["sort"]

        def key(r):
            v = r.get(k)
            return (v is None or (isinstance(v, float) and v != v), v if isinstance(v, (int, float)) else str(v or "").lower())
        rows = sorted(rows, key=key, reverse=st["desc"])
    return rows


def _fmt_q(q) -> str:
    return "" if q is None or q != q else (f"{q:.1e}" if q < 0.001 else f"{q:.3f}")


def _table(m, surf, area, kind: str, st: dict, res: dict, host) -> None:
    cols = ACT_COLS if kind == "activation" else KO_COLS
    rows = _rows(kind, st, res)
    detail_w = 340 if st["sel"] is not None else 0
    left = pygame.Rect(area.x, area.y, area.w - detail_w - (8 if detail_w else 0), area.h)
    head = (f"{res.get('n_types', len(rows)):,} types, seeds {res['seeds'][0]}-{res['seeds'][-1]}, {res.get('n_called', 0)} with a behavior call"
            if kind == "activation" else f"{len(res['behaviors'])} behaviors, seeds {res['seeds'][0]}-{res['seeds'][-1]}")
    m.text(surf, head + f"  ·  showing {len(rows):,}", (left.x + 8, left.y), ui.INK, m.f_small)
    box = pygame.Rect(left.right - 230, left.y - 4, 224, 26)
    m.text_field(surf, box, st["search"], lambda v: st.__setitem__("search", v), id=("scr", kind, "search"), limit=40,
                 tip="Filter by cell type or what it does. Type and press Enter.")
    if not st["search"]:
        m.text(surf, "search…", (box.x + 10, box.centery), ui.DIM, m.f_small, "midleft")
    y = left.y + 26
    for key, label, dx in cols:
        r = pygame.Rect(left.x + dx - 4, y - 2, m.f_small.size(label)[0] + 22, 20)
        arrow = (" v" if st["desc"] else " ^") if st["sort"] == key else ""
        m._register(r, "button", id=("scr", kind, "sort", key), click=lambda k=key: _sort(st, k), tip=f"Sort by {label}")
        m.text(surf, label + arrow, (left.x + dx, y), ui.AMBER if st["sort"] == key else ui.LABEL, m.f_small)
    y += 20
    view = pygame.Rect(left.x, y, left.w, left.bottom - y)
    key = f"lab_screen_{kind}"
    off = int(m.scroll.get(key, 0))
    prev = surf.get_clip()
    m.clip = view
    surf.set_clip(view)
    yy = y - off
    for r in rows:
        sel = st["sel"] is not None and _id(kind, r) == st["sel"]
        rr = pygame.Rect(left.x, yy - 2, left.w, 22)
        if yy + 22 >= view.y and yy <= view.bottom:
            over = m._register(rr, "button", id=("scr", kind, "row", _id(kind, r)), click=lambda r=r: st.__setitem__("sel", _id(kind, r)))
            if sel or over:
                pygame.draw.rect(surf, (44, 52, 68) if sel else ui.ROW_HOVER, rr, border_radius=4)
            _row(m, surf, left, yy, kind, r)
        yy += 22
    m.content_h[key] = max(0, yy + off - view.bottom + 8)
    surf.set_clip(prev)
    m.clip = None
    if detail_w:
        sel = next((r for r in (res["rows"]) if _id(kind, r) == st["sel"]), None)
        if sel is not None:
            _detail(m, surf, pygame.Rect(area.right - detail_w, area.y, detail_w, area.h), kind, st, res, sel, host)


def _id(kind: str, r: dict):
    return r["type"] if kind == "activation" else f"{r['behavior']}|{r['candidate']}"


def _sort(st: dict, k: str) -> None:
    st["desc"] = not st["desc"] if st["sort"] == k else k not in ("type", "candidate", "behavior", "rank", "behavior_call")
    st["sort"] = k


def _row(m, surf, left, yy, kind: str, r: dict) -> None:
    f = m.f_small
    if kind == "activation":
        called = bool(r["behavior_readout"])
        col = (ui.GOOD if r["behavior_effect_hz"] > 0 else ui.BAD) if called else ui.DIM
        vals = [(str(r["rank"]), ui.DIM), (r["type"], ui.TEXT if called else ui.LABEL), (f"{r['neurons']:,}", ui.LABEL),
                (r["behavior_call"][:33] + ("…" if len(r["behavior_call"]) > 33 else ""), col), (f"{r['behavior_effect_hz']:+.1f} Hz" if called else "", col), (_fmt_q(r.get("behavior_q")), ui.LABEL),
                (str(r.get("dn_responders", "")), ui.LABEL)]
        for (key, _, dx), (txt, c) in zip(ACT_COLS, vals):
            m.text(surf, txt, (left.x + dx, yy), c, f)
    else:
        strong = r["q"] is not None and r["q"] < 0.05 and (r["drop_share"] or 0) >= 0.1
        col = ui.BAD if strong else ui.TEXT
        vals = [(r["behavior"], ui.LABEL), (r["candidate"] + ("  (batch)" if r["is_batch"] else ""), col),
                (f"{r['drop_share']:+.0%}" if r["drop_share"] == r["drop_share"] else "", col), (_fmt_q(r["q"]), ui.LABEL), (f"{r['neurons']:,}", ui.LABEL),
                (str(r["rank"] or ""), ui.DIM)]
        for (key, _, dx), (txt, c) in zip(KO_COLS, vals):
            m.text(surf, txt[:30], (left.x + dx, yy), c, f)


# --- the detail panel (the "inspector" for a screen row) -----------------------------------------------------------------------------------
def _detail(m, surf, area, kind: str, st: dict, res: dict, r: dict, host) -> None:
    pygame.draw.rect(surf, (14, 17, 24), area, border_radius=10)
    pygame.draw.rect(surf, ui.BORDER, area, 1, border_radius=10)
    x, y = area.x + 12, area.y + 10
    f, fb = m.f_small, m.f_bold
    lh = f.get_linesize() + 2
    if kind == "activation":
        m.text(surf, r["type"], (x, y), ui.INK, fb)
        y += lh + 2
        m.text(surf, f"{r['neurons']:,} neurons · {r['superclass'] or 'unclassified'} · {r['n_seeds']} seeds", (x, y), ui.LABEL, f)
        y += lh
        m.text(surf, r["behavior_call"], (x, y), ui.GOOD if r["behavior_readout"] and r["behavior_effect_hz"] > 0 else ui.TEXT, f)
        y += lh + 4
        m.text(surf, "READOUTS vs MATCHED CONTROLS (Hz, CI, q)", (x, y), ui.DIM, f)
        y += lh
        ids = [x_["id"] for x_ in res["readouts"]]
        for rid in ids:
            e = r.get(f"eff_{rid}")
            if e is None:
                continue
            lo, hi, q = r.get(f"lo_{rid}"), r.get(f"hi_{rid}"), r.get(f"q_{rid}")
            own = rid in (r.get("self_readouts") or "").split(",")
            txt = f"{rid:<12}{e:+7.1f}  [{lo:+.1f}, {hi:+.1f}]" + ("   itself" if own else f"  {_fmt_q(q)}")
            m.text(surf, txt, (x, y), ui.DIM if own else (ui.TEXT if q is not None and q == q and q < 0.05 else ui.LABEL), f)
            y += lh - 3
        y += 4
        m.text(surf, f"DESCENDING TYPES MOVED ({r.get('dn_responders', 0)})", (x, y), ui.DIM, f)
        y += lh
        for n, d, z in ((r.get("_dn") or [])[:5] if r.get("_dn") else _parse_top(r)[:5]):
            m.text(surf, f"{n:<14}{d:+7.1f} Hz", (x, y), ui.TEXT, f)
            y += lh - 1
        types = [r["type"]]
        mode = +1
    else:
        m.text(surf, r["candidate"], (x, y), ui.INK, fb)
        y += lh + 2
        m.text(surf, f"{r['behavior']} · {r['neurons']:,} neurons · {r['n_seeds']} seeds", (x, y), ui.LABEL, f)
        y += lh + 4
        lines = [("unperturbed response", f"{r['unperturbed_evoked_hz']:.2f} Hz"), ("with this lesion", f"{r['lesion_evoked_hz']:.2f} Hz"),
                 ("matched random lesions", f"{r['control_evoked_hz']:.2f} Hz"),
                 ("drop vs matched", f"{r['drop_hz']:+.2f} Hz  [{r['drop_lo']:+.2f}, {r['drop_hi']:+.2f}]"),
                 ("as a share", f"{r['drop_share']:+.0%}"), ("q (screen-wide)", _fmt_q(r["q"])), ("seeds the same way", f"{r.get('sign_share', 0):.0%}")]
        for a, b in lines:
            m.text(surf, f"{a:<24}{b}", (x, y), ui.TEXT, f)
            y += lh - 1
        if r["is_batch"]:
            m.text(surf, "A batch: silenced together. Batches that cut the response", (x, y + 4), ui.DIM, f)
            m.text(surf, "were opened and their members tested one by one.", (x, y + 4 + lh - 2), ui.DIM, f)
            y += 2 * lh
        types = r["candidate"].split("+")
        mode = -1
    y = area.bottom - 78
    verb = "Activate on live flies" if mode > 0 else "Silence on live flies"
    live = bool(types) and all(getattr(host, "type_ops", {}).get(t) == mode for t in types)
    m.button(surf, (x, y, area.w - 24, 32), "undo on live flies" if live else verb, lambda: _toggle(host, types, mode), id=("scr", kind, "live"),
             active=live, tip="The brain surgery switch on every live fly, so you can watch what this row means in the game.")
    m.button(surf, (x, y + 38, area.w - 24, 32), "Inspect a neuron of this type", lambda: _inspect(m, host, types[0]), id=("scr", kind, "inspect"),
             tip="Closes the menu and opens the big brain view with the neuron inspector on one of its neurons.")
    closer = pygame.Rect(area.right - 30, area.y + 6, 24, 24)
    m._register(closer, "button", id=("scr", kind, "close"), click=lambda: st.__setitem__("sel", None), tip="Close this panel")
    m.text(surf, "x", closer.center, ui.LABEL, fb, "center")


def _parse_top(r: dict) -> list:
    out = []
    for part in (r.get("top_dn") or "").split("; "):
        try:
            n, rest = part.rsplit(" ", 1)
            out.append((n, float(rest.replace("Hz", "")), 0.0))
        except ValueError:
            continue
    return out


def _toggle(host, types: list[str], mode: int) -> None:
    ops = getattr(host, "type_ops", None)
    if ops is None:
        return
    on = all(ops.get(t) == mode for t in types)
    for t in types:
        if on:
            ops.pop(t, None)
        else:
            ops[t] = mode
    host._apply_surgery()
    host.note(f"SURGERY  {'+'.join(types)[:40]}: {'normal' if on else 'on' if mode > 0 else 'off'}")


def _inspect(m, host, cell_type: str) -> None:
    br = getattr(host, "brain", None)
    if br is None:
        return
    rows = np.flatnonzero(br.types == cell_type)
    if not len(rows):
        m.flash(f"no {cell_type} neurons in this brain", ui.BAD)
        return
    host.inspect = host._neuron_info(int(rows[0]))
    host.big_view = True
    m.close()


def _export(m, kind: str, res: dict) -> None:
    import shutil

    from kickthefly.lab import recorder

    dst = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{KINDS[kind]['folder']}"
    dst.mkdir(parents=True, exist_ok=True)
    src = folder_of(kind)
    for f in src.glob(f"{kind}_screen*") if src.exists() else []:
        shutil.copy2(f, dst / f.name)
    m.host.last_export = str(dst)
    m.flash(f"saved to {dst.name} in exports", ui.GOOD)


def install(menu) -> None:
    for kind, cfg in KINDS.items():
        menu.pages[cfg["page"]] = page(kind)
