"""Lab screen for Connectome Diff Mode: diffing activity region by region between two flies.

Reuses autopsy comparison format (diverging bars) and includes a divergence timeline.
"""
from __future__ import annotations

import math
import threading
import time

import pygame

from kickthefly.ui import menu as ui

SCENARIOS = (
    ("looming", "Looming escape"),
    ("sugar", "Sugar feeding"),
    ("antenna", "Antennal touch"),
    ("calm", "Calm spontaneous"),
)

DIFF_MODS = (
    ("lesion_loom", "Silence looming (LPLC2, LC4)"),
    ("lesion_gf", "Silence giant fiber (DNp01)"),
    ("hemi_vis", "Silence left visual hemifield"),
    ("hemi_all", "Silence left hemisphere"),
    ("thresh_10", "Threshold: drop <10 synapses"),
    ("flips", "Flip uncertain signs (50% @ 0.5)"),
    ("inhibition_0", "Picrotoxin: 100% inhibition block"),
)


def _st(m):
    from kickthefly.lab import lab

    st = lab._state(m)
    if not hasattr(st, "diff_scenario"):
        st.diff_scenario = "looming"
        st.diff_mod = "lesion_loom"
        st.diff_job = None
        st.diff_result = None
    return st


def page(m: ui.Menu, surf, rect, mouse) -> None:
    st, host = _st(m), m.host
    m.text(surf, "CONNECTOME DIFF MODE", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = m.subtitle(surf, rect, "Run two flies with different configurations from the exact same seed and inputs, and diff their activity "
                               "region by region live.")
    # 3.0 release review: fixed button widths cut most labels and the second row overlapped its own label; the rows now flow from the font
    right = rect.right - 24
    y = m.flow_buttons(surf, rect.x + 24, y, right, [(label, (lambda k=key: setattr(st, "diff_scenario", k)), ("d_scen", key),
                                                      st.diff_scenario == key, None) for key, label in SCENARIOS], label="Shared Stimulus:") + 8
    y = m.flow_buttons(surf, rect.x + 24, y, right, [(label, (lambda k=key: setattr(st, "diff_mod", k)), ("d_mod", key),
                                                      st.diff_mod == key, None) for key, label in DIFF_MODS], label="Fly B Configuration:") - 28
    y += 40
    job = st.diff_job
    if job is not None and not job["thread"].is_alive():
        if job.get("result"):
            st.diff_result = job["result"]
        if job.get("error"):
            m.text(surf, f"error: {job['error']}", (rect.x + 24, y + 10), ui.BAD, m.f_small)
        st.diff_job = job = None

    if job is not None:
        frac = job["done"] / max(1, job["total"])
        pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, y + 10, rect.w - 300, 12), border_radius=6)
        pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, y + 10, max(8, int((rect.w - 300) * frac)), 12),
                         border_radius=6)
        m.text(surf, f"{job['label']} …", (rect.x + 24, y + 26), ui.LABEL, m.f_small)
    else:
        m.button(surf, (rect.x + 24, y, 260, 36), "Run Connectome Diff", lambda: _start(m, st),
                 id="run_diff", style="primary",
                 tip="Simulates Fly A (control) and Fly B (perturbed) with identical seed and inputs, measuring "
                     "divergence region by region.")

    res = st.diff_result
    body = pygame.Rect(rect.x + 24, y + 46, rect.w - 48, rect.bottom - y - 46 - 66)
    if res:
        _draw_diff_results(m, surf, body, res)
    else:
        m.text(surf, "Click 'Run Connectome Diff' to compare region-by-region activity and divergence timeline." if job is None else "",
               (body.x, body.y + 12), ui.LABEL, m.f_small)

    if res:
        m.button(surf, (rect.x + 24, rect.bottom - 58, 200, 42), "Export CSV + JSON",
                 lambda: _export(m, res), id="diff_export")
    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("diff", "back"))


def _start(m, st) -> None:
    from kickthefly.lab import diffmode, lesions
    from kickthefly.core import simcore
    from kickthefly.sim.wiring import Wiring, random_flip

    job = dict(done=0, total=1, label="running connectome diff", result=None, error=None)

    def work():
        try:
            g, _, soma = simcore.pack()
            types = g.type.astype(str)

            cfg_a = dict(wiring=Wiring(), lesion_rows=None, label="Fly A (Unperturbed)")
            cfg_b = dict(wiring=Wiring(), lesion_rows=None, label=f"Fly B ({st.diff_mod})")

            if st.diff_mod == "lesion_loom":
                cfg_b["lesion_rows"] = np.flatnonzero(np.isin(types, ("LPLC2", "LC4")))
            elif st.diff_mod == "lesion_gf":
                cfg_b["lesion_rows"] = np.flatnonzero(types == "DNp01")
            elif st.diff_mod == "hemi_vis":
                cfg_b["lesion_rows"] = lesions.hemifield_visual_rows(g, "L")
            elif st.diff_mod == "hemi_all":
                cfg_b["lesion_rows"] = lesions.hemisphere_rows(g, "L", soma=soma)
            elif st.diff_mod == "thresh_10":
                cfg_b["wiring"] = Wiring(min_synapses=10)
            elif st.diff_mod == "flips":
                cfg_b["wiring"] = Wiring(flip_rows=random_flip(g, 0.5, 0.5, seed=1))
            elif st.diff_mod == "inhibition_0":
                cfg_b["wiring"] = Wiring(inhibition_scale=0.0)

            res = diffmode.run_connectome_diff(cfg_a, cfg_b, stimulus=st.diff_scenario, steps=200, seed=1000)
            job["done"] = 1
            job["result"] = res
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"

    job["thread"] = threading.Thread(target=work, name="connectome_diff", daemon=True)
    job["thread"].start()
    st.diff_job = job


def _draw_diff_results(m, surf, area, res: dict) -> None:
    # 1. Divergence Timeline strip
    div_time = res.get("diverge_time_s")
    div_str = f"diverged at {div_time*1000:.0f} ms" if div_time is not None else "no major divergence"
    m.text(surf, f"DIVERGENCE TIMELINE: {div_str}  ·  seed {res['seed']}  ·  stimulus: {res['stimulus']}",
           (area.x, area.y), ui.INK, m.f_bold)

    timeline = res.get("divergence_timeline", [])
    tl_w, tl_h = min(area.w - 20, 680), 32
    tl_rect = pygame.Rect(area.x, area.y + 20, tl_w, tl_h)
    pygame.draw.rect(surf, (16, 20, 28), tl_rect, border_radius=6)
    pygame.draw.rect(surf, (40, 48, 64), tl_rect, 1, border_radius=6)

    if len(timeline) > 1:
        max_div = max(max(timeline), 1e-4)
        pts = []
        for i, val in enumerate(timeline):
            px = tl_rect.x + int(i * (tl_w - 4) / (len(timeline) - 1)) + 2
            py = tl_rect.bottom - int((val / max_div) * (tl_h - 6)) - 3
            pts.append((px, py))
        pygame.draw.lines(surf, ui.AMBER, False, pts, 2)
        if res.get("diverge_step") is not None:
            dx = tl_rect.x + int(res["diverge_step"] / (len(timeline) * 5) * tl_w)
            pygame.draw.line(surf, ui.BAD, (dx, tl_rect.y), (dx, tl_rect.bottom), 2)
            m.text(surf, "divergence", (dx + 4, tl_rect.y + 4), ui.BAD, m.f_small)

    y = tl_rect.bottom + 14

    # 2. Autopsy-style diverging bars
    m.text(surf, "Region-by-Region Activity Diff (Autopsy diverging bars: red = B > A, blue = B < A):",
           (area.x, y), ui.INK, m.f_bold)
    y += 20

    chart_w = area.w - 420
    cx0 = area.x + 190
    mid = cx0 + chart_w // 2

    # Column headers
    m.text(surf, "Neuropil Region", (area.x, y), ui.LABEL, m.f_small)
    m.text(surf, "1/4x", (mid - chart_w // 4, y), ui.LABEL, m.f_small, "midtop")
    m.text(surf, "same", (mid, y), ui.LABEL, m.f_small, "midtop")
    m.text(surf, "4x", (mid + chart_w // 4, y), ui.LABEL, m.f_small, "midtop")
    m.text(surf, "Fly A (Hz)", (cx0 + chart_w + 30, y), ui.LABEL, m.f_small)
    m.text(surf, "Fly B (Hz)", (cx0 + chart_w + 110, y), ui.LABEL, m.f_small)
    m.text(surf, "Diff Δ", (cx0 + chart_w + 190, y), ui.LABEL, m.f_small)
    y += 18

    pygame.draw.line(surf, (80, 80, 76), (mid, y - 2), (mid, y + 10 * 16), 1)

    for row in res.get("rows", [])[:9]:
        m.text(surf, row["name"][:22], (area.x, y), ui.TEXT, m.f_small)

        # Diverging bar
        lr = row["log2_ratio"]
        frac = max(-1.0, min(1.0, lr / 2.0))
        bl = int(abs(frac) * (chart_w // 2 - 4))
        col = (230, 100, 90) if frac > 0 else (60, 135, 230)
        if bl >= 2:
            rx = mid if frac > 0 else mid - bl
            pygame.draw.rect(surf, col, (rx, y + 2, bl, 11), border_radius=2)
        else:
            pygame.draw.circle(surf, (60, 60, 58), (mid, y + 7), 2)

        m.text(surf, f"{row['rate_a']:.1f}", (cx0 + chart_w + 30, y), ui.LABEL, m.f_small)
        m.text(surf, f"{row['rate_b']:.1f}", (cx0 + chart_w + 110, y), ui.INK, m.f_small)
        diff_col = ui.BAD if abs(row["diff"]) > 2.0 else ui.TEXT
        m.text(surf, f"{row['diff']:+.1f}", (cx0 + chart_w + 190, y), diff_col, m.f_small)
        y += 16


def _export(m, res: dict) -> None:
    from kickthefly.lab import diffmode, recorder

    folder = recorder.exports_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-connectome-diff"
    diffmode.save(res, folder)
    m.host.last_export = str(folder)
    m.flash(f"saved to {folder.name} in exports", ui.GOOD)
