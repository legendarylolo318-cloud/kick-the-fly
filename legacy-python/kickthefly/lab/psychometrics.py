"""Psychometric curve generator for Kick the Fly.

Sweeps experimental parameters across headless trials, computes mean and SEM/CI,
and produces publication-quality SVG, vector PDF, and CSV exports.

Targets:
1. Looming: approach speed vs escape probability and latency.
2. Sugar: concentration/dose vs proboscis extension response and MN9 ratio.
3. Conditioning: number of training pairings vs T-maze performance index.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from kickthefly.core import paths
from kickthefly.core.version import __version__
from kickthefly.lab import assays

SWEEP_PRESETS = {
    "looming": {
        "label": "Looming escape vs approach speed",
        "param_name": "approach_speed",
        "param_label": "Approach speed (m/s)",
        "x_values": [0.5, 1.0, 2.0, 3.5, 5.5, 8.0, 11.0],
        "metric_primary": "escape_prob",
        "metric_primary_label": "Escape probability",
        "metric_secondary": "latency_s",
        "metric_secondary_label": "Latency (s)",
        "y_max": 1.0,
    },
    "sugar": {
        "label": "Proboscis extension vs sugar concentration",
        "param_name": "sugar_concentration",
        "param_label": "Sugar concentration (fraction of sweet GRNs)",
        "x_values": [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
        "metric_primary": "response_rate",
        "metric_primary_label": "Proboscis extension rate",
        "metric_secondary": "mn9_ratio",
        "metric_secondary_label": "MN9 rate ratio (during/before)",
        "y_max": 1.0,
    },
    "tmaze": {
        "label": "Memory performance index vs conditioning pairings",
        "param_name": "pairings",
        "param_label": "Conditioning cycles (CS+ / shock pairings)",
        "x_values": [1, 2, 4, 6, 8, 12],
        "metric_primary": "pi",
        "metric_primary_label": "Performance index (PI)",
        "metric_secondary": None,
        "metric_secondary_label": None,
        "y_max": 1.0,
    },
}


def _calc_stats(vals: list[float]) -> dict:
    arr = np.array([v for v in vals if not math.isnan(v)], float)
    if len(arr) == 0:
        return dict(mean=float("nan"), sem=0.0, ci95=0.0, n=0)
    m = float(np.mean(arr))
    n = len(arr)
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem
    return dict(mean=m, sem=sem, ci95=ci95, n=n)


def run_sweep(
    target: str,
    x_values: list[float] | None = None,
    n_flies: int = 4,
    base_seed: int = 1000,
    progress: Callable[[int, int], None] | None = None,
    params: dict | None = None,
) -> dict:
    """Run psychometric curve sweep across N flies per point in lockstep."""
    if target not in SWEEP_PRESETS:
        raise ValueError(f"Unknown target '{target}', must be one of {list(SWEEP_PRESETS)}")

    spec = SWEEP_PRESETS[target]
    xs = list(x_values if x_values is not None else spec["x_values"])
    seeds = [base_seed + i for i in range(n_flies)]
    total_steps = len(xs) * n_flies
    done_steps = 0
    t0 = time.time()

    points = []
    if target == "looming":
        for x in xs:
            escaped_counts = []
            latencies = []
            for s in seeds:
                res = assays.looming_fly(s, speeds=[float(x)], approaches=2, params=params)
                trials = res["trials"][float(x)]
                esc_rate = np.mean([1.0 if t["escaped"] else 0.0 for t in trials])
                escaped_counts.append(esc_rate)
                trial_lats = [t["latency_s"] for t in trials if t["escaped"] and t["latency_s"] is not None]
                if trial_lats:
                    latencies.append(float(np.mean(trial_lats)))
                done_steps += 1
                if progress:
                    progress(done_steps, total_steps)

            st_prob = _calc_stats(escaped_counts)
            st_lat = _calc_stats(latencies)
            points.append({
                "x": float(x),
                "primary": st_prob,
                "secondary": st_lat,
            })

    elif target == "sugar":
        for x in xs:
            extensions = []
            ratios = []
            for s in seeds:
                res = assays.sugar_fly(s, doses=[float(x)], repeats=2, params=params)
                offers = res["offers"][float(x)]
                ext_rate = np.mean([1.0 if o["extended"] else 0.0 for o in offers])
                extensions.append(ext_rate)
                ratios.append(float(np.mean([o["ratio"] for o in offers])))
                done_steps += 1
                if progress:
                    progress(done_steps, total_steps)

            st_ext = _calc_stats(extensions)
            st_rat = _calc_stats(ratios)
            points.append({
                "x": float(x),
                "primary": st_ext,
                "secondary": st_rat,
            })

    elif target == "tmaze":
        for x in xs:
            pis = []
            cycles = int(x)
            for s in seeds:
                a = assays.tmaze_fly(s, "odor_a", paired=True, cycles=cycles, params=params)
                b = assays.tmaze_fly(s + 50_000, "odor_b", paired=True, cycles=cycles, params=params)
                pis.append(float((a["pi"] + b["pi"]) / 2))
                done_steps += 1
                if progress:
                    progress(done_steps, total_steps)

            st_pi = _calc_stats(pis)
            points.append({
                "x": float(x),
                "primary": st_pi,
                "secondary": None,
            })

    created = time.strftime("%Y-%m-%d %H:%M:%S")
    caption = (
        f"Kick the Fly v{__version__} | Target: {spec['label']} | "
        f"Seeds: {base_seed}..{base_seed + n_flies - 1} (n={n_flies}) | "
        f"Default LIF sim (dt=5ms, tau=20ms) | Generated: {created}"
    )

    return {
        "target": target,
        "spec": spec,
        "xs": xs,
        "n_flies": n_flies,
        "base_seed": base_seed,
        "seeds": seeds,
        "points": points,
        "seconds": round(time.time() - t0, 1),
        "caption": caption,
        "app_version": __version__,
        "created": created,
    }


def export_csv(result: dict, out_path: Path | str) -> Path:
    """Export psychometric sweep points to CSV."""
    import csv

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = result["spec"]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["# " + result["caption"]])
        writer.writerow([
            spec["param_name"],
            "n",
            f"{spec['metric_primary']}_mean",
            f"{spec['metric_primary']}_sem",
            f"{spec['metric_primary']}_ci95",
            *( [f"{spec['metric_secondary']}_mean", f"{spec['metric_secondary']}_sem"] if spec["metric_secondary"] else [] ),
        ])
        for pt in result["points"]:
            p = pt["primary"]
            row = [pt["x"], p["n"], f"{p['mean']:.4f}", f"{p['sem']:.4f}", f"{p['ci95']:.4f}"]
            if spec["metric_secondary"] and pt["secondary"]:
                s = pt["secondary"]
                row.extend([f"{s['mean']:.4f}", f"{s['sem']:.4f}"])
            writer.writerow(row)
    return path


def export_svg(result: dict, out_path: Path | str) -> Path:
    """Generate publication-ready vector SVG with error bars and metadata caption."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = result["spec"]
    pts = result["points"]

    W, H = 800, 560
    pad_l, pad_r, pad_t, pad_b = 90, 40, 70, 100
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b

    x_vals = [p["x"] for p in pts]
    x_min, x_max = min(x_vals), max(x_vals)
    if x_max == x_min:
        x_max += 1.0

    y_max = spec.get("y_max", 1.0)
    y_min = 0.0

    def tx(x):
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def ty(y):
        return pad_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;">',
        f'  <rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'  <text x="{pad_l}" y="36" font-size="18" font-weight="bold" fill="#111827">{spec["label"]}</text>',
        f'  <text x="{pad_l}" y="56" font-size="12" fill="#4b5563">n = {result["n_flies"]} flies per point (mean \u00b1 95% CI)</text>',
    ]

    # Gridlines & Y-axis ticks
    for i in range(6):
        y_val = y_min + i * (y_max - y_min) / 5
        py = ty(y_val)
        lines.append(f'  <line x1="{pad_l}" y1="{py}" x2="{pad_l + plot_w}" y2="{py}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'  <text x="{pad_l - 12}" y="{py + 4}" font-size="11" text-anchor="end" fill="#6b7280">{y_val:.2f}</text>')

    # X-axis ticks
    for pt in pts:
        px = tx(pt["x"])
        lines.append(f'  <line x1="{px}" y1="{pad_t + plot_h}" x2="{px}" y2="{pad_t + plot_h + 6}" stroke="#9ca3af" stroke-width="1.5"/>')
        lines.append(f'  <text x="{px}" y="{pad_t + plot_h + 20}" font-size="11" text-anchor="middle" fill="#6b7280">{pt["x"]:g}</text>')

    # Plot axes
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#374151" stroke-width="1.5"/>')
    lines.append(f'  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#374151" stroke-width="1.5"/>')

    # Axis titles
    lines.append(f'  <text x="{pad_l + plot_w / 2}" y="{pad_t + plot_h + 46}" font-size="13" font-weight="600" text-anchor="middle" fill="#1f2937">{spec["param_label"]}</text>')
    lines.append(f'  <text x="24" y="{pad_t + plot_h / 2}" font-size="13" font-weight="600" text-anchor="middle" fill="#1f2937" transform="rotate(-90 24 {pad_t + plot_h / 2})">{spec["metric_primary_label"]}</text>')

    # Connect curve
    poly_pts = []
    for pt in pts:
        px = tx(pt["x"])
        py = ty(pt["primary"]["mean"])
        poly_pts.append(f"{px:.1f},{py:.1f}")
    lines.append(f'  <polyline points="{" ".join(poly_pts)}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round"/>')

    # Data points and error bars
    for pt in pts:
        px = tx(pt["x"])
        pm = pt["primary"]["mean"]
        py = ty(pm)
        ci = pt["primary"]["ci95"]
        y_hi = ty(min(y_max, pm + ci))
        y_lo = ty(max(y_min, pm - ci))

        # Error bar
        lines.append(f'  <line x1="{px}" y1="{y_lo}" x2="{px}" y2="{y_hi}" stroke="#1d4ed8" stroke-width="1.5"/>')
        lines.append(f'  <line x1="{px - 4}" y1="{y_hi}" x2="{px + 4}" y2="{y_hi}" stroke="#1d4ed8" stroke-width="1.5"/>')
        lines.append(f'  <line x1="{px - 4}" y1="{y_lo}" x2="{px + 4}" y2="{y_lo}" stroke="#1d4ed8" stroke-width="1.5"/>')
        # Point circle
        lines.append(f'  <circle cx="{px}" cy="{py}" r="5" fill="#2563eb" stroke="#ffffff" stroke-width="1.5"/>')

    # Caption footer
    lines.append(f'  <line x1="{pad_l}" y1="{H - 34}" x2="{pad_l + plot_w}" y2="{H - 34}" stroke="#e5e7eb" stroke-width="1"/>')
    lines.append(f'  <text x="{pad_l}" y="{H - 18}" font-size="10" fill="#9ca3af">{result["caption"]}</text>')
    lines.append('</svg>')

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_pdf(result: dict, out_path: Path | str) -> Path:
    """Generate a clean, standalone vector PDF-1.4 document containing the figure."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = result["spec"]
    pts = result["points"]

    # 11 x 8.5 inches in points: 792 x 612 (landscape letter)
    PW, PH = 792, 612
    pad_l, pad_r, pad_t, pad_b = 80, 50, 80, 90
    plot_w = PW - pad_l - pad_r
    plot_h = PH - pad_t - pad_b

    x_vals = [p["x"] for p in pts]
    x_min, x_max = min(x_vals), max(x_vals)
    if x_max == x_min:
        x_max += 1.0

    y_max = spec.get("y_max", 1.0)
    y_min = 0.0

    def tx(x):
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def ty(y):
        return pad_b + (y - y_min) / (y_max - y_min) * plot_h

    # Build PDF graphics stream commands
    cmds = []
    # Background white
    cmds.append(f"1 1 1 rg 0 0 {PW} {PH} re f")

    # Grid lines (light gray)
    cmds.append("0.85 0.85 0.85 RG 0.75 w")
    for i in range(6):
        y_val = y_min + i * (y_max - y_min) / 5
        py = ty(y_val)
        cmds.append(f"{pad_l} {py:.1f} m {pad_l + plot_w} {py:.1f} l S")

    # X axis tick marks
    cmds.append("0.3 0.3 0.3 RG 1.5 w")
    for pt in pts:
        px = tx(pt["x"])
        cmds.append(f"{px:.1f} {pad_b} m {px:.1f} {pad_b - 6} l S")

    # Axes
    cmds.append(f"{pad_l} {pad_b + plot_h} m {pad_l} {pad_b} l {pad_l + plot_w} {pad_b} l S")

    # Polyline curve (blue)
    cmds.append("0.145 0.388 0.921 RG 2.5 w")
    poly_pts = [(tx(pt["x"]), ty(pt["primary"]["mean"])) for pt in pts]
    cmds.append(f"{poly_pts[0][0]:.1f} {poly_pts[0][1]:.1f} m")
    for px, py in poly_pts[1:]:
        cmds.append(f"{px:.1f} {py:.1f} l")
    cmds.append("S")

    # Error bars (darker blue)
    cmds.append("0.114 0.306 0.847 RG 1.5 w")
    for pt in pts:
        px = tx(pt["x"])
        pm = pt["primary"]["mean"]
        ci = pt["primary"]["ci95"]
        y_hi = ty(min(y_max, pm + ci))
        y_lo = ty(max(y_min, pm - ci))
        cmds.append(f"{px:.1f} {y_lo:.1f} m {px:.1f} {y_hi:.1f} l S")
        cmds.append(f"{px - 4:.1f} {y_hi:.1f} m {px + 4:.1f} {y_hi:.1f} l S")
        cmds.append(f"{px - 4:.1f} {y_lo:.1f} m {px + 4:.1f} {y_lo:.1f} l S")

    # Data point circles (filled circles)
    cmds.append("0.145 0.388 0.921 rg 1 1 1 RG 1.5 w")
    for px, py in poly_pts:
        r = 4.0
        # Bézier circle
        k = 0.552284749831 * r
        cmds.append(f"{px + r:.1f} {py:.1f} m "
                    f"{px + r:.1f} {py + k:.1f} {px + k:.1f} {py + r:.1f} {px:.1f} {py + r:.1f} c "
                    f"{px - k:.1f} {py + r:.1f} {px - r:.1f} {py + k:.1f} {px - r:.1f} {py:.1f} c "
                    f"{px - r:.1f} {py - k:.1f} {px - k:.1f} {py - r:.1f} {px:.1f} {py - r:.1f} c "
                    f"{px + k:.1f} {py - r:.1f} {px + r:.1f} {py - k:.1f} {px + r:.1f} {py:.1f} c B")

    # Text elements
    def clean_pdf_txt(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_cmds = ["BT"]
    # Title
    text_cmds.append("/F1 16 Tf 0.1 0.1 0.1 rg")
    text_cmds.append(f"{pad_l} {PH - 42} Td ({clean_pdf_txt(spec['label'])}) Tj")

    # Subtitle
    text_cmds.append("/F1 10 Tf 0.3 0.3 0.3 rg")
    text_cmds.append(f"0 -18 Td (n = {result['n_flies']} flies per point, mean +/- 95% CI) Tj")

    # X axis label
    text_cmds.append(f"/F1 12 Tf 0.15 0.15 0.15 rg")
    text_cmds.append(f"{pad_l + plot_w / 2 - 60} {pad_b - 36} Td ({clean_pdf_txt(spec['param_label'])}) Tj")

    # Y ticks labels
    text_cmds.append("/F1 9 Tf 0.4 0.4 0.4 rg")
    for i in range(6):
        y_val = y_min + i * (y_max - y_min) / 5
        py = ty(y_val)
        text_cmds.append(f"{pad_l - 30} {py - 3:.1f} Td ({y_val:.2f}) Tj 0 0 Td")

    # X ticks labels
    for pt in pts:
        px = tx(pt["x"])
        text_cmds.append(f"{px - 8:.1f} {pad_b - 18} Td ({pt['x']:g}) Tj 0 0 Td")

    # Footer metadata caption
    text_cmds.append("/F1 8 Tf 0.5 0.5 0.5 rg")
    text_cmds.append(f"{pad_l} 28 Td ({clean_pdf_txt(result['caption'])}) Tj")
    text_cmds.append("ET")

    content = "\n".join(cmds + text_cmds).encode("latin-1", "replace")

    # Construct standard PDF-1.4 file
    objects = []
    # 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3: Page
    objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PW} {PH}] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".encode("ascii"))
    # 4: Stream Content
    objects.append(f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream")
    # 5: Font
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    body = b"%PDF-1.4\n"
    xref_offsets = []
    for i, obj in enumerate(objects, 1):
        xref_offsets.append(len(body))
        body += f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    startxref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for off in xref_offsets:
        body += f"{off:010d} 00000 n \n".encode("ascii")

    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode("ascii")

    path.write_bytes(body)
    return path


def save_all_formats(result: dict, base_name: str, folder: Path | str | None = None) -> dict[str, Path]:
    """Saves CSV, SVG, and PDF to the exports folder."""
    out_dir = Path(folder) if folder is not None else paths.get().data_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / base_name
    csv_p = export_csv(result, stem.with_suffix(".csv"))
    svg_p = export_svg(result, stem.with_suffix(".svg"))
    pdf_p = export_pdf(result, stem.with_suffix(".pdf"))
    return dict(csv=csv_p, svg=svg_p, pdf=pdf_p)
