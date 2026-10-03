"""Lab screen for Psychometric Curve Generator.

Plots psychometric curves with error bars across parameter sweeps and exports
publication-quality SVG, PDF, and CSV files.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pygame

from kickthefly.ui import menu as ui
from kickthefly.lab import psychometrics


def _st(m):
    from kickthefly.lab import lab

    st = lab._state(m)
    if not hasattr(st, "psych_target"):
        st.psych_target = "looming"
        st.psych_flies = 3
        st.psych_job = None
        st.psych_result = None
        st.psych_msg = None
    return st


def page(m: ui.Menu, surf, rect, mouse) -> None:
    st, host = _st(m), m.host
    m.text(surf, "PSYCHOMETRIC CURVE GENERATOR", (rect.x + 24, rect.y + 16), ui.INK, m.f_head)
    y = max(rect.y + 80, m.subtitle(surf, rect, "Sweep experimental parameters across headless trials with error bars. "
                                                "Exports publication-quality SVG, PDF, and CSV."))
    busy = st.psych_job is not None and st.psych_job["thread"].is_alive()

    # Target assay selector
    targets = list(psychometrics.SWEEP_PRESETS.keys())
    target_labels = [psychometrics.SWEEP_PRESETS[k]["label"] for k in targets]
    target_idx = targets.index(st.psych_target) if st.psych_target in targets else 0

    m.text(surf, "Assay target:", (rect.x + 24, y + 6), ui.TEXT, m.f_text)
    m.segmented(surf, (rect.x + 150, y, 620, 32),
                ["Looming speed", "Sugar dose", "T-maze pairings"],
                target_idx, lambda i: setattr(st, "psych_target", targets[i]),
                id="psych_target", enabled=not busy)
    y += 44

    # Flies slider
    m.text(surf, "Flies / point:", (rect.x + 24, y + 6), ui.TEXT, m.f_text)
    m.slider(surf, (rect.x + 150, y, 260, 32), st.psych_flies, 2, 8, 1, "{:.0f} flies",
             lambda v: setattr(st, "psych_flies", int(v)), lambda: None,
             id="psych_flies", enabled=not busy,
             tip="Number of seeds to run per point (more gives tighter error bars)")

    base_seed = int(host.cfg["brain.seed"]) if hasattr(host, "cfg") else 1000
    m.text(surf, f"Base seed: {base_seed}", (rect.x + 430, y + 16), ui.LABEL, m.f_small, "midleft")
    y += 44

    # Run / Status / Export
    def start_run():
        j = dict(done=0, total=1, result=None, error=None)

        def work():
            try:
                spec = psychometrics.SWEEP_PRESETS[st.psych_target]
                j["total"] = len(spec["x_values"]) * st.psych_flies
                params = getattr(host, "lab_params", None)
                res = psychometrics.run_sweep(
                    st.psych_target,
                    n_flies=st.psych_flies,
                    base_seed=base_seed,
                    progress=lambda d, t: j.update(done=d, total=t),
                    params=params,
                )
                j["result"] = res
            except Exception as e:
                j["error"] = f"{type(e).__name__}: {e}"

        j["thread"] = threading.Thread(target=work, name="psych_sweep", daemon=True)
        j["thread"].start()
        st.psych_job = j
        st.psych_msg = None

    if busy:
        j = st.psych_job
        frac = j["done"] / max(1, j["total"])
        bar_w = rect.w - 240
        pygame.draw.rect(surf, (30, 36, 48), (rect.x + 24, y, bar_w, 14), border_radius=7)
        pygame.draw.rect(surf, ui.AMBER, (rect.x + 24, y, max(8, int(bar_w * frac)), 14), border_radius=7)
        m.text(surf, f"Running trials: {j['done']} / {j['total']} ({frac:.0%})", (rect.x + 24, y + 22), ui.TEXT, m.f_small)
    else:
        if st.psych_job and st.psych_job.get("result") and st.psych_result is not st.psych_job["result"]:
            st.psych_result = st.psych_job["result"]
            st.psych_job = None
        m.button(surf, (rect.x + 24, y, 190, 38), "Run Sweep", start_run,
                 style="primary", id="psych_run",
                 tip="Runs the parameter sweep across headless brains in background.")

        if st.psych_result:
            def export_all():
                from kickthefly.core import paths
                folder = paths.get().data_dir / "exports"
                stamp = time.strftime("%Y%m%d_%H%M%S")
                fname = f"psych_{st.psych_target}_{stamp}"
                saved = psychometrics.save_all_formats(st.psych_result, fname, folder)
                st.psych_msg = f"Exported CSV, SVG & PDF to {saved['svg'].parent.name}/"

            m.button(surf, (rect.x + 230, y, 240, 38), "Export CSV, SVG & PDF", export_all,
                     id="psych_export_all", tip="Export publication-ready vector figures and CSV.")

        if st.psych_msg:
            m.text(surf, st.psych_msg, (rect.x + 490, y + 20), ui.GOOD, m.f_small, "midleft")

    y += 54

    # Render results chart and data summary
    if st.psych_result and not busy:
        res = st.psych_result
        spec = res["spec"]
        pts = res["points"]

        # Chart area (left half)
        chart_w = (rect.w - 60) // 2
        chart_rect = pygame.Rect(rect.x + 24, y, chart_w, rect.bottom - y - 70)
        pygame.draw.rect(surf, (20, 24, 34), chart_rect, border_radius=8)
        pygame.draw.rect(surf, (40, 48, 64), chart_rect, 1, border_radius=8)

        # Plot points inside chart
        p_l, p_r, p_t, p_b = chart_rect.x + 50, chart_rect.right - 20, chart_rect.y + 30, chart_rect.bottom - 40
        pw, ph = p_r - p_l, p_b - p_t
        xs = [pt["x"] for pt in pts]
        x_min, x_max = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1.0
        y_max = spec.get("y_max", 1.0)

        # Gridlines
        for i in range(5):
            y_val = i * y_max / 4
            py = p_b - int((y_val / y_max) * ph)
            pygame.draw.line(surf, (35, 42, 56), (p_l, py), (p_r, py), 1)
            m.text(surf, f"{y_val:.2f}", (p_l - 6, py), ui.LABEL, m.f_small, "midright")

        poly_pts = []
        for pt in pts:
            px = p_l + int((pt["x"] - x_min) / (x_max - x_min) * pw)
            pm = pt["primary"]["mean"]
            py = p_b - int(min(1.0, max(0.0, pm / y_max)) * ph)
            poly_pts.append((px, py))

            # Error bar
            ci = pt["primary"]["ci95"]
            y_hi = p_b - int(min(1.0, max(0.0, (pm + ci) / y_max)) * ph)
            y_lo = p_b - int(min(1.0, max(0.0, (pm - ci) / y_max)) * ph)
            pygame.draw.line(surf, (100, 160, 240), (px, y_lo), (px, y_hi), 2)
            pygame.draw.line(surf, (100, 160, 240), (px - 4, y_hi), (px + 4, y_hi), 2)
            pygame.draw.line(surf, (100, 160, 240), (px - 4, y_lo), (px + 4, y_lo), 2)

        if len(poly_pts) > 1:
            pygame.draw.lines(surf, (60, 140, 240), False, poly_pts, 2)
        for px, py in poly_pts:
            pygame.draw.circle(surf, (220, 240, 255), (px, py), 4)

        # Chart axes titles
        m.text(surf, spec["param_label"], (p_l + pw // 2, p_b + 20), ui.LABEL, m.f_small, "center")
        m.text(surf, spec["metric_primary_label"], (p_l, p_t - 14), ui.INK, m.f_bold, "midleft")

        # Table area (right half)
        tbl_x = chart_rect.right + 20
        tbl_w = rect.right - tbl_x - 24
        ty = y
        m.text(surf, "MEASURED VALUES", (tbl_x, ty), ui.INK, m.f_bold)
        ty += 24

        m.text(surf, f"{spec['param_name']}", (tbl_x, ty), ui.LABEL, m.f_small)
        m.text(surf, "Mean \u00b1 SEM", (tbl_x + 120, ty), ui.LABEL, m.f_small)
        m.text(surf, "95% CI", (tbl_x + 240, ty), ui.LABEL, m.f_small)
        ty += 18

        for pt in pts:
            p = pt["primary"]
            m.text(surf, f"{pt['x']:g}", (tbl_x, ty), ui.TEXT, m.f_small)
            m.text(surf, f"{p['mean']:.3f} \u00b1 {p['sem']:.3f}", (tbl_x + 120, ty), ui.INK, m.f_small)
            m.text(surf, f"[{p['mean'] - p['ci95']:.3f}, {p['mean'] + p['ci95']:.3f}]", (tbl_x + 240, ty), ui.LABEL, m.f_small)
            ty += 20

        # Caption box
        cap_y = rect.bottom - 56
        m.text(surf, res["caption"], (rect.x + 24, cap_y), (140, 150, 170), m.f_small)

    m.button(surf, (rect.right - 164, rect.bottom - 58, 140, 42), "Back", m.back, style="primary", id=("psych", "back"))
