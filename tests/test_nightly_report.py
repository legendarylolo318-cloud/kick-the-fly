"""tools/nightly_report.py: the history, the flips, the static page and the issue text (no brain pack needed)."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("nightly_report", ROOT / "tools" / "nightly_report.py")
nr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nr)


def validation(passed: dict, ratio=5.0):
    return dict(app_version="2.13.0", backend="cpu", seconds=600.0,
                tests=[dict(id=i, passed=p, measured=dict(drive_ratio_mean=ratio)) for i, p in passed.items()])


def playthrough(fail=(), total=5):
    res = [dict(id=f"c{i}", status="fail" if f"c{i}" in fail else "pass", metrics=dict(sim_real_ratio=3.0 + i)) for i in range(total)]
    return dict(meta=dict(backend="cpu"), counts=dict(zip(("pass", "fail", "skip", "gated"), (total - len(fail), len(fail), 0, 2))),
                results=res)


def run(tmp_path, val, play, date, monkeypatch, commit="abcdef123456"):
    v = tmp_path / f"v{date}.json"
    v.write_text(json.dumps(val))
    p = tmp_path / "pt" / "playthrough.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(play))
    (p.parent / "playthrough.md").write_text("# report\n")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    nr.main(["--validation", str(v), "--playthrough", str(p), "--site", str(tmp_path / "site"),
             "--issue-body", str(tmp_path / "issue.md"), "--issue-title", str(tmp_path / "title.txt"),
             "--commit", commit, "--date", date, "--run-url", "https://example.invalid/run/1"])
    return json.loads((tmp_path / "site" / "history.json").read_text())["nights"]


def test_a_quiet_night_appends_history_and_opens_no_issue(tmp_path, monkeypatch, capsys):
    exp = nr.expected()
    same = {k: v for k, v in exp.items()}
    run(tmp_path, validation(same), playthrough(), "2026-01-01", monkeypatch)
    nights = run(tmp_path, validation(same), playthrough(), "2026-01-02", monkeypatch)
    assert [n["date"] for n in nights] == ["2026-01-01", "2026-01-02"]
    assert "open_issue=false" in capsys.readouterr().out and not (tmp_path / "issue.md").exists()
    assert nights[-1]["validation"]["passed"] == sum(same.values()) and nights[-1]["playthrough"]["median_sim_real"] == 5.0


def test_a_flipped_validation_result_opens_an_issue_naming_it(tmp_path, monkeypatch, capsys):
    exp = nr.expected()
    run(tmp_path, validation(dict(exp)), playthrough(), "2026-01-01", monkeypatch)
    flipped = dict(exp)
    victim = next(k for k, v in exp.items() if v)
    flipped[victim] = False
    run(tmp_path, validation(flipped), playthrough(), "2026-01-02", monkeypatch)
    assert "open_issue=true" in capsys.readouterr().out
    body = (tmp_path / "issue.md").read_text()
    assert f"{victim}: PASS -> FAIL" in body and "expected PASS, got FAIL" in body and "Nothing was tuned" in body
    assert "flipped" in (tmp_path / "title.txt").read_text()
    assert f"{victim}: PASS -&gt; FAIL" in (tmp_path / "site" / "index.html").read_text() or victim in (tmp_path / "site" / "index.html").read_text()


def test_a_playthrough_failure_opens_an_issue_and_a_rerun_replaces_the_same_night(tmp_path, monkeypatch, capsys):
    exp = nr.expected()
    run(tmp_path, validation(dict(exp)), playthrough(fail=("c2",)), "2026-01-01", monkeypatch)
    assert "open_issue=true" in capsys.readouterr().out and "`c2`" in (tmp_path / "issue.md").read_text()
    nights = run(tmp_path, validation(dict(exp)), playthrough(), "2026-01-01", monkeypatch)
    assert len(nights) == 1 and nights[0]["playthrough"]["failed"] == []


def test_the_page_is_static_with_charts_and_a_history_table(tmp_path, monkeypatch):
    exp = nr.expected()
    for d in ("2026-01-01", "2026-01-02", "2026-01-03"):
        run(tmp_path, validation(dict(exp), ratio=4.0 + int(d[-1])), playthrough(), d, monkeypatch)
    page = (tmp_path / "site" / "index.html").read_text()
    assert page.count("<svg") >= 3 + 1 and "<table>" in page and page.count("<tr><td>") == 3
    assert page.index("2026-01-03") < page.index("2026-01-01"), "newest first"
    assert not re.search(r"<script|<link|<iframe|@import|<img|src=|url\(", page), "no scripts, no external requests"
    assert not re.search(r'https?://', page), "no URL at all is loaded or linked from the page"
    assert (tmp_path / "site" / "validation.json").exists() and (tmp_path / "site" / "playthrough.md").exists()


def test_a_night_with_no_results_is_recorded_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert nr.main(["--site", str(tmp_path / "s"), "--date", "2026-02-01", "--commit", "x"]) == 0
    assert json.loads((tmp_path / "s" / "history.json").read_text())["nights"][0]["validation"] is None
