"""Settings > Help (2.12): replay the tutorial, run the self-test, report a bug. Also the pages behind those buttons.

Pages (menu.pages): "selftest" runs core/selftest.py on a background thread and lists each check with its verdict and
fix; "bugreport" is the review screen of core/bugreport.py: every item that would be included, shown in full, each
removable, and three ways out (clipboard, a file, a prefilled GitHub issue in the browser). Nothing is ever sent.
The same pages run on the crash screen (ui/crashscreen.py) with a small stand-in host.
"""
from __future__ import annotations

import threading

import pygame

from kickthefly.core import bugreport, selftest
from kickthefly.core.i18n import tr
from kickthefly.ui import menu as mu

STATUS_COLORS = {
    "default": {"PASS": (90, 200, 120), "WARN": (240, 180, 60), "FAIL": (230, 90, 80)},
    "blue-yellow": {"PASS": (70, 130, 240), "WARN": (255, 224, 70), "FAIL": (255, 255, 255)},
    "high-contrast": {"PASS": (255, 255, 255), "WARN": (255, 60, 220), "FAIL": (255, 60, 60)},
}


def status_color(cfg, status: str):
    return STATUS_COLORS.get(str(cfg.get("access.palette", "default")), STATUS_COLORS["default"]).get(status, mu.TEXT)


# --- the Help tab -------------------------------------------------------------------------------------------------
def help_tab(menu, surf, body, y: int) -> int:
    """The rows under Settings > Help. Returns the y below them."""
    host = menu.host
    rows = [
        (tr("Replay the tutorial"), tr("Play"),
         tr("The one-minute first-launch tutorial again: moving, using a tool, the brain panel, sugar and the loadout editor. Skippable at any step."),
         lambda: (menu.close(), host.start_tutorial(replay=True))),
        (tr("Tool loadout editor"), tr("Open"),
         tr("Choose which tools are on the hotbar (keys 1-9, 0), reorder them and save your own loadouts. Also on the Q key."),
         lambda: (menu.close(), host.open_loadout_editor())),
        (tr("Self-test"), tr("Run"),
         tr("Checks this install: brain packs, every compute backend, OpenGL, audio, ffmpeg, folders, disk, memory and a 10-second run. Says how to fix what it finds. Changes nothing."),
         lambda: menu.show("selftest")),
        (tr("Report a bug"), tr("Report"),
         tr("Collects the app version, OS, backend, self-test, the last 500 log lines and any crash report, shows you exactly what would be included, and lets you copy it or open a prefilled GitHub issue. Nothing is sent automatically."),
         lambda: menu.show("bugreport")),
    ]
    for label, btn, tip, click in rows:
        row = pygame.Rect(body.x, y, body.w, 44)
        if row.collidepoint(menu.mouse) and body.collidepoint(menu.mouse):
            pygame.draw.rect(surf, mu.ROW_HOVER, row, border_radius=8)
        menu.text(surf, label, (row.x + 12, row.centery), mu.TEXT, menu.f_text, "midleft")
        menu._register(pygame.Rect(row.x, row.y, 400, row.h), "label", id=("help", label), tip=tip)
        menu.button(surf, (row.x + 400, row.y + 7, 150, 30), btn, click, id=("help-btn", label), tip=tip)
        y += 48
    y += 6
    menu.wrapped(surf, tr("Kick the Fly never uploads anything and has no telemetry. Reports, self-tests and logs stay on "
                          "this computer until you copy or attach them yourself."),
                 (body.x + 12, y), body.w - 24, mu.LABEL, menu.f_small, max_lines=3)
    return y + 60


# --- self-test page -----------------------------------------------------------------------------------------------
def _start_selftest(menu) -> None:
    st = menu.__dict__.setdefault("st", {})
    if st.get("running"):
        return
    st.update(running=True, checks=[], report=None)

    def work():
        try:
            rep = selftest.run(progress=lambda c: st["checks"].append(dict(c.__dict__)))
            st["report"] = rep
        except Exception as e:                                  # never crash the menu from the worker
            st["checks"].append(dict(id="selftest", name="Self-test", status="FAIL", detail=f"{type(e).__name__}: {e}",
                                     fix="Report this with Report a bug.", seconds=0.0))
        finally:
            st["running"] = False

    threading.Thread(target=work, daemon=True, name="selftest").start()


def page_selftest(menu, surf, rect, mouse) -> None:
    cfg = menu.host.cfg
    st = menu.__dict__.setdefault("st", {})
    if not st.get("checks") and not st.get("running") and not st.get("started"):
        st["started"] = True
        _start_selftest(menu)
    menu.text(surf, tr("SELF-TEST"), (rect.x + 24, rect.y + 16), mu.INK, menu.f_head)
    checks = st.get("checks", [])
    running = st.get("running", False)
    rep = st.get("report")
    if running:
        menu.text(surf, tr("Running: {n} checks done...", n=len(checks)), (rect.right - 24, rect.y + 22), mu.AMBER,
                  menu.f_small, "topright")
    elif rep:
        n = rep["counts"]
        menu.text(surf, tr("{p} passed, {w} warnings, {f} failed", p=n["PASS"], w=n["WARN"], f=n["FAIL"]),
                  (rect.right - 24, rect.y + 22), status_color(cfg, "FAIL" if n["FAIL"] else "WARN" if n["WARN"] else "PASS"),
                  menu.f_small, "topright")
    body = pygame.Rect(rect.x + 16, rect.y + 56, rect.w - 32, rect.h - 56 - 70)
    key = "selftest"
    off = int(menu.scroll.get(key, 0))
    prev = surf.get_clip()
    menu.clip = body
    surf.set_clip(body)
    y = body.y - off
    for c in checks:
        col = status_color(cfg, c["status"])
        chip = pygame.Rect(body.x + 8, y + 4, 62, 22)
        pygame.draw.rect(surf, col, chip, border_radius=5)
        menu.text(surf, c["status"], chip.center, (12, 14, 18), menu.f_small, "center")
        menu.text(surf, c["name"], (body.x + 82, y + 3), mu.INK, menu.f_bold)
        y = menu.wrapped(surf, c["detail"], (body.x + 82, y + 26), body.w - 100, mu.TEXT, menu.f_small, max_lines=4)
        if c.get("fix"):
            y = menu.wrapped(surf, tr("Fix: {fix}", fix=c["fix"]), (body.x + 82, y + 2), body.w - 100, col, menu.f_small,
                             max_lines=4)
        y += 12
    menu.content_h[key] = max(0, y + off - body.bottom + 8)
    surf.set_clip(prev)
    menu.clip = None
    fy = rect.bottom - 58
    menu.button(surf, (rect.x + 24, fy, 170, 42), tr("Run again"), lambda: (st.update(started=True), _start_selftest(menu)),
                enabled=not running, id=("st", "again"), tip=tr("Run every check again."))
    menu.button(surf, (rect.x + 206, fy, 190, 42), tr("Report a bug"), lambda: menu.show("bugreport"), id=("st", "report"),
                tip=tr("Review what a bug report would include, with this self-test in it."))
    menu.button(surf, (rect.right - 164, fy, 140, 42), tr("Back"), menu.back, style="primary", id=("st", "back"))


# --- bug report page ----------------------------------------------------------------------------------------------
def _br(menu) -> dict:
    br = menu.__dict__.setdefault("br", {})
    if "items" not in br:
        br.update(items=bugreport.collect(getattr(selftest, "LAST", None)), sel=0, note="", saved=None, msg="")
    return br


def page_bugreport(menu, surf, rect, mouse) -> None:
    br = _br(menu)
    items = br["items"]
    menu.text(surf, tr("REPORT A BUG"), (rect.x + 24, rect.y + 16), mu.INK, menu.f_head)
    menu.text(surf, tr("Nothing is sent. This is exactly what would be included: switch items off to leave them out."),
              (rect.x + 24, rect.y + 48), mu.LABEL, menu.f_small)
    # what happened
    menu.text(surf, tr("What happened?"), (rect.x + 24, rect.y + 76), mu.TEXT, menu.f_small)
    menu.text_field(surf, (rect.x + 150, rect.y + 70, rect.w - 176, 30), br["note"],
                    lambda v: br.__setitem__("note", v), id=("br", "note"), limit=240,
                    tip=tr("A sentence about what you were doing. It goes into the report text below the items."))
    # item list
    left = pygame.Rect(rect.x + 16, rect.y + 112, 300, rect.h - 112 - 132)
    y = left.y
    for i, it in enumerate(items):
        row = pygame.Rect(left.x, y, left.w, 58)
        on_sel = i == br["sel"]
        pygame.draw.rect(surf, (30, 36, 50) if on_sel else (22, 26, 34), row, border_radius=8)
        pygame.draw.rect(surf, mu.ACCENT if on_sel else mu.BORDER, row, 1, border_radius=8)
        menu._register(row, "button", id=("br-row", i), click=lambda i=i: br.__setitem__("sel", i),
                       tip=tr("Show this item's exact text."))
        menu.text(surf, tr(it.title), (row.x + 10, row.y + 8), mu.INK if it.include else mu.DIM, menu.f_bold)
        menu.text(surf, it.size, (row.x + 10, row.y + 32), mu.LABEL, menu.f_small)
        menu.toggle(surf, (row.right - 100, row.y + 16, 90, 26), it.include,
                    lambda v, it=it: setattr(it, "include", v and bool(it.text.strip())), id=("br-inc", i),
                    enabled=bool(it.text.strip()),
                    tip=tr("Include this item in the report. It is empty when there is nothing to include."))
        y += 64
    # exact text
    view = pygame.Rect(left.right + 12, left.y, rect.right - 16 - left.right - 12, left.h)
    pygame.draw.rect(surf, (10, 12, 18), view, border_radius=8)
    pygame.draw.rect(surf, mu.BORDER, view, 1, border_radius=8)
    it = items[br["sel"]]
    text = it.text if it.text.strip() else tr("(nothing to show: no data for this item on this computer)")
    font = getattr(menu, "f_mono", None)
    if font is None:
        menu.f_mono = font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 12)
    lh = font.get_linesize()
    lines = text.splitlines() or [""]
    key = "bugreport"
    off = int(menu.scroll.get(key, 0))
    prev = surf.get_clip()
    inner = view.inflate(-12, -8)
    surf.set_clip(inner)
    menu.clip = view
    first = max(0, off // lh)
    for j in range(first, min(len(lines), first + inner.h // lh + 2)):
        surf.blit(font.render(lines[j][:220], True, mu.TEXT if it.include else mu.DIM), (inner.x, inner.y + j * lh - off))
    menu.content_h[key] = max(0, len(lines) * lh - inner.h + 8)
    surf.set_clip(prev)
    menu.clip = None
    if not it.include and it.text.strip():
        menu.text(surf, tr("left out"), (view.right - 10, view.y + 6), mu.AMBER, menu.f_small, "topright")
    # actions
    fy = rect.bottom - 116
    full = bugreport.clipboard_text(items, br["note"])
    title, body, attach = bugreport.compose_issue(items, br["note"], "the file it saves")
    url = bugreport.issue_url(title, body)
    menu.text(surf, tr("Report text: {n:,} characters. Opening the issue puts about {u:,} characters in the link; long items go to a file you attach.",
                       n=len(full), u=len(url)), (rect.x + 24, fy - 4), mu.LABEL, menu.f_small)
    if br.get("saved"):
        menu.text(surf, tr("Saved to {path}: attach this file to the issue.", path=br["saved"]), (rect.x + 24, fy + 16),
                  mu.GOOD, menu.f_small)
    elif br.get("msg"):
        menu.text(surf, br["msg"], (rect.x + 24, fy + 16), mu.GOOD, menu.f_small)
    by = rect.bottom - 62
    menu.button(surf, (rect.x + 24, by, 210, 42), tr("Copy to clipboard"), lambda: _copy(menu, br, full),
                id=("br", "copy"), tip=tr("Copy everything that is switched on. You paste it wherever you like."))
    menu.button(surf, (rect.x + 246, by, 250, 42), tr("Open GitHub issue"), lambda: _open(menu, br, attach, url),
                style="primary", id=("br", "open"),
                tip=tr("Opens your browser on a prefilled new-issue page. If items are long, they are saved to a file to attach by hand. Nothing is sent by this program."))
    menu.button(surf, (rect.x + 508, by, 180, 42), tr("Save to file"), lambda: _save(menu, br, full), id=("br", "save"),
                tip=tr("Save everything that is switched on as a text file."))
    menu.button(surf, (rect.right - 164, by, 140, 42), tr("Back"), lambda: (br.clear(), menu.back()), id=("br", "back"))


def _copy(menu, br, full) -> None:
    ok, msg = bugreport.copy_to_clipboard(full)
    br["msg"], br["saved"] = tr(msg), None
    if not ok:
        _save(menu, br, full)


def _save(menu, br, full) -> None:
    p = bugreport.save_attachment(full)
    br["saved"], br["msg"] = (str(p) if p else None), (tr("could not write the file") if p is None else "")


def _open(menu, br, attach, url) -> None:
    if attach:
        p = bugreport.save_attachment(attach)
        br["saved"] = str(p) if p else None
    ok = bugreport.open_in_browser(url)
    br["msg"] = tr("Opened your browser.") if ok else tr("Could not open a browser: copy the report instead.")


def install(menu) -> None:
    menu.pages["selftest"] = page_selftest
    menu.pages["bugreport"] = page_bugreport
