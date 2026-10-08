"""Tests for psychometric curve generator, SVG, PDF, and CSV exports."""
from pathlib import Path
import pytest

from conftest import needs_pack
from kickthefly.lab.psychometrics import _calc_stats, run_sweep, export_csv, export_svg, export_pdf, save_all_formats


def test_calc_stats():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    st = _calc_stats(vals)
    assert st["mean"] == pytest.approx(3.0)
    assert st["n"] == 5
    assert st["sem"] > 0
    assert st["ci95"] == pytest.approx(1.96 * st["sem"])

    empty = _calc_stats([])
    assert empty["n"] == 0


@needs_pack
def test_looming_sweep_and_exports(tmp_path):
    res = run_sweep("looming", x_values=[1.0, 5.0], n_flies=2, base_seed=100)
    assert res["target"] == "looming"
    assert len(res["points"]) == 2
    assert "caption" in res
    assert "Seeds: 100..101" in res["caption"]
    assert "v2." in res["caption"] or "v" in res["caption"]

    # Test CSV export
    csv_file = export_csv(res, tmp_path / "sweep.csv")
    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "# Kick the Fly" in content
    assert "escape_prob_mean" in content

    # Test SVG export
    svg_file = export_svg(res, tmp_path / "figure.svg")
    assert svg_file.exists()
    svg_text = svg_file.read_text(encoding="utf-8")
    assert "<svg" in svg_text and "</svg>" in svg_text
    assert "polyline" in svg_text
    assert "circle" in svg_text
    assert res["caption"] in svg_text

    # Test PDF export
    pdf_file = export_pdf(res, tmp_path / "figure.pdf")
    assert pdf_file.exists()
    pdf_bytes = pdf_file.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf_bytes
    assert b"/Helvetica" in pdf_bytes


@needs_pack
def test_sugar_and_tmaze_sweeps(tmp_path):
    # Quick sweep of sugar
    res_sugar = run_sweep("sugar", x_values=[0.0, 0.5], n_flies=2, base_seed=100)
    assert len(res_sugar["points"]) == 2
    assert res_sugar["points"][0]["primary"]["mean"] <= res_sugar["points"][1]["primary"]["mean"]

    # Quick sweep of tmaze
    res_tmaze = run_sweep("tmaze", x_values=[1, 4], n_flies=2, base_seed=100)
    assert len(res_tmaze["points"]) == 2

    # Test save_all_formats
    saved = save_all_formats(res_sugar, "sugar_sweep", tmp_path)
    assert saved["csv"].exists()
    assert saved["svg"].exists()
    assert saved["pdf"].exists()
