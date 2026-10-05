"""Lab page: Failure autopsy (3.1.0 task 8). One page per failing validation test (lab/failure_autopsy.py): where the driven signal goes
and where it fades. Read-only: nothing on this page changes the model, and nothing suggests changing it. MODEL PREDICTION."""
from __future__ import annotations

import json
import threading
import time

import pygame

from kickthefly.ui import menu as ui


def _st(m):
    from kickthefly.lab import lab

    st = lab._state(m)
    if not hasattr(st, "fa"):
        st.fa = dict(fails=None, sel=None, done={}, job=None, error=None, flash=None)
    return st.fa


def folder():
    from kickthefly.lab import recorder

    return recorder.exports_dir() / "failure-autopsy"


def _load(st) -> None:
    from kickthefly.lab import failure_autopsy as fa, validation

    if st["fails"] is not None:
        return
    st["fails"], st["error"] = [], None
    try:
        res, _ = validation.load_results()
        st["fails"] = fa.failing_tests(res) if res else []
        if not res:
            st["error"] = "No validation result yet: run Lab > Validation (or --validate) first."
    except Exception as e:
        st["error"] = f"{type(e).__name__}: {e}"
    for f in st["fails"]:
        p = folder() / f"{f['id']}.json"
        if p.exists():
            try:
                st["done"][f["id"]] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
    if st["fails"] and st["sel"] is None:
        st["sel"] = st["fails"][0]["id"]


def page(m: ui.Menu, surf, rect, mouse) -> None:
    from kickthefly.lab import failure_autopsy as fa

    st = _st(m)
    ui.TAG_COLORS.setdefault(fa.TAG, (150, 120, 220))
    m.text(surf, "FAILURE AUTOPSY", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    m.chip(surf, (rect.x + 40 + m.f_head.size("FAILURE AUTOPSY")[0], rect.y + 22), fa.TAG)
    y = m.subtitle(surf, rect, "For each validation test that fails: the paths from its input to its target, how much of the target's input they can "
                               "supply, how much of it excites or inhibits, and what each stage did in the run. Read-only: it never changes or suggests "
                               "changing the model.")
    _load(st)
    job = st["job"]
    if job is not None and not job["thread"].is_alive():
        st["error"] = job.get("error")
        for a in job.get("result") or []:
            st["done"][a["test"]["id"]] = a
        st["job"] = job = None
    left = pygame.Rect(rect.x + 16, y, 330, rect.bottom - y - 66)
    right = pygame.Rect(left.right + 10, y, rect.right - left.right - 26, left.h)
    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (right.x, y, right.w - 8, 10), border_radius=5)
        pygame.draw.rect(surf, ui.AMBER, (right.x, y, max(8, int((right.w - 8) * frac)), 10), border_radius=5)
        m.text(surf, f"{job['label']}  ·  {job['done']}/{job['total']}  ·  {time.time() - job['t0']:.0f}s", (right.x, y + 14), ui.LABEL, m.f_small)
        right = pygame.Rect(right.x, y + 34, right.w, right.h - 34)
    _list(m, surf, left, st)
    if st["error"] and not st["fails"]:
        if hasattr(m, "wrapped"):                           # 3.1.0 review: one line ran past the panel at narrow widths
            m.wrapped(surf, st["error"], (right.x + 8, right.y + 8), right.w - 16, ui.BAD, m.f_small, 4)
        else:
            m.text(surf, st["error"], (right.x + 8, right.y + 8), ui.BAD, m.f_small)
    elif st["sel"] is not None:
        a = st["done"].get(st["sel"])
        if a is None:
            m.text(surf, "Not autopsied yet.", (right.x + 8, right.y + 8), ui.LABEL, m.f_small)
        else:
            _document(m, surf, right, a)
    if st["fails"]:
        w = m.bw("Autopsy this failure", 220)
        m.button(surf, (rect.x + 24, rect.bottom - 58, w, 42), "Autopsy this failure", lambda: _start(m, st, [st["sel"]]), id=("fa", "one"), style="primary",
                 enabled=job is None and st["sel"] is not None,
                 tip="Runs three seeds of the brain (about 15 s) and reads the wiring. Read-only.")
        w2 = m.bw("Autopsy all", 150)
        m.button(surf, (rect.x + 36 + w, rect.bottom - 58, w2, 42), "Autopsy all", lambda: _start(m, st, None), id=("fa", "all"), enabled=job is None)
        m.button(surf, (rect.x + 48 + w + w2, rect.bottom - 58, m.bw("Export", 130), 42), "Export", lambda: _export(m, st), id=("fa", "export"),
                 enabled=bool(st["done"]), tip="Copies each page (Markdown and JSON) into a new folder in exports.")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("fa", "back"))


def _list(m, surf, area, st) -> None:
    pygame.draw.rect(surf, (14, 17, 24), area, border_radius=10)
    y = area.y + 8
    m.text(surf, f"{len(st['fails'])} failing test(s)", (area.x + 10, y), ui.LABEL, m.f_small)
    y += 24
    for f in st["fails"]:
        r = pygame.Rect(area.x + 4, y - 2, area.w - 8, 44)
        sel = f["id"] == st["sel"]
        over = m._register(r, "button", id=("fa", "row", f["id"]), click=lambda i=f["id"]: st.__setitem__("sel", i), tip=f.get("name", f["id"]))
        if sel or over:
            pygame.draw.rect(surf, (44, 52, 68) if sel else ui.ROW_HOVER, r, border_radius=6)
        done = f["id"] in st["done"]
        m.text(surf, f["name"][:40], (area.x + 12, y), ui.TEXT if done else ui.LABEL, m.f_small)
        m.text(surf, "autopsied" if done else "not yet", (area.x + 12, y + 20), ui.GOOD if done else ui.DIM, m.f_small)
        y += 48


def _document(m, surf, area, a: dict) -> None:
    from kickthefly.lab import failure_autopsy as fa

    key = "lab_autopsy"
    off = int(m.scroll.get(key, 0))
    prev = surf.get_clip()
    m.clip = area
    surf.set_clip(area)
    x, y = area.x + 8, area.y + 4 - off
    f, fb = m.f_small, m.f_bold
    lh = f.get_linesize() + 3

    def line(text, col=ui.TEXT, font=f, dx=0, gap=0):
        nonlocal y
        y = m.wrapped(surf, text, (x + dx, y), area.w - 24 - dx, col, font, 6) + gap if hasattr(m, "wrapped") else y + lh

    t = a["test"]
    line(t["name"], ui.INK, fb, gap=2)
    if a.get("error"):
        line(a["error"], ui.BAD)
    else:
        line(f"Claim: {t['claim']}", ui.LABEL)
        r = a.get("result")
        if r:
            ms = ", ".join(f"{k} {v:.3g}" for k, v in (r.get("measured") or {}).items() if isinstance(v, (int, float)) and k in (
                "drive_ratio_mean", "control_ratio_mean", "mean_contrast", "mean_persistence_ms", "p_value", "n"))
            line(f"FAIL. {r.get('criteria', '')}   {ms}", ui.BAD, gap=2)
        for s in a["sections"]:
            y += 6
            line(s["label"], ui.AMBER, fb)
            if s.get("error"):
                line(s["error"], ui.BAD)
                continue
            line(f"input {s['input']} ({s['n_input']})   target {s['target']} ({s['n_target']})   direct synapses {s['direct_synapses']}", ui.LABEL)
            line("share of the target's input along walks of exactly h synapses", ui.DIM)
            for b in s["budget"]:
                total = b["total"] or 1e-12
                w = area.w - 470
                x0 = x + 160
                m.text(surf, f"{b['hop']} synapse{'s' if b['hop'] > 1 else ''}", (x, y), ui.TEXT, f)
                ew = int(w * b["excitatory"] / max(1e-12, max(bb["total"] for bb in s["budget"])))
                iw = int(w * b["inhibitory"] / max(1e-12, max(bb["total"] for bb in s["budget"])))
                pygame.draw.rect(surf, (60, 170, 110), (x0, y + 3, max(1, ew), 11))
                pygame.draw.rect(surf, (200, 80, 70), (x0 + max(1, ew), y + 3, max(1, iw), 11))
                m.text(surf, f"{fa.pct(b['excitatory'])} excites, {fa.pct(b['inhibitory'])} inhibits", (x0 + w + 8, y), ui.LABEL, f)
                y += lh
            y += 4
            line("strongest routes (green: stage ratio met the test's 1.5x, red: below it)", ui.DIM)
            for i, rt in enumerate(s["routes"], 1):
                chain = " > ".join(["input", *rt["types"], "target"])
                line(f"{i}. {chain}: {fa.pct(rt['share'])} of the target's input, net {'excites' if rt['net_sign'] > 0 else 'inhibits'}", ui.TEXT, gap=1)
                for st_, e in zip(rt["stage_ratios"][1:], rt["edges"]):
                    bw = int(min(1.0, st_["ratio"] / 4.0) * 110)
                    col = (60, 170, 110) if st_["ratio"] >= fa.RATIO_MIN else (200, 80, 70)
                    m.text(surf, f"{st_['stage'][:22]}", (x + 18, y), ui.LABEL, f)
                    pygame.draw.rect(surf, (30, 36, 48), (x + 180, y + 3, 110, 11))
                    pygame.draw.rect(surf, col, (x + 180, y + 3, max(2, bw), 11))
                    xl = x + 180 + int(110 * fa.RATIO_MIN / 4.0)
                    pygame.draw.line(surf, ui.AMBER, (xl, y + 1), (xl, y + 15))
                    m.text(surf, f"x{st_['ratio']:.2f}  {st_['base_hz']:.1f} to {st_['driven_hz']:.1f} Hz   {e['synapses']:,} syn ({e['excitatory']:,}+ {e['inhibitory']:,}-)",
                           (x + 300, y), ui.DIM, f)
                    y += lh
            line("Where it fades: " + s["verdict"]["text"], ui.GOOD if s["verdict"].get("stage") is None else ui.BAD, fb, gap=4)
        y += 6
        line("Read-only. " + a["method"], ui.DIM)
    m.content_h[key] = max(0, y + off - area.bottom + 8)
    surf.set_clip(prev)
    m.clip = None


def _start(m, st, only) -> None:
    from kickthefly.lab import failure_autopsy as fa

    job = dict(done=0, total=1, label="starting", error=None, result=None, t0=time.time())

    def work():
        try:
            job["result"] = fa.run_all(folder(), only=only, progress=lambda d, n, label: job.update(done=d, total=n, label=label))
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="failure-autopsy", daemon=True)
    job["thread"].start()
    st["job"], st["error"] = job, None


def _export(m, st) -> None:
    import shutil

    from kickthefly.lab import recorder

    dst = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-failure-autopsy"
    dst.mkdir(parents=True, exist_ok=True)
    for f in folder().glob("*") if folder().exists() else []:
        shutil.copy2(f, dst / f.name)
    m.host.last_export = str(dst)
    m.flash(f"saved to {dst.name} in exports", ui.GOOD)


def install(menu) -> None:
    menu.pages["lab_autopsy"] = page
