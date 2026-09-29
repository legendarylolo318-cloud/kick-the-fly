"""The workflow files and the review checklist (2.13): they parse, and say what CONTRIBUTING.md says they do."""
from __future__ import annotations

import yaml

from conftest import ROOT

WF = ROOT / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WF / name).read_text(encoding="utf-8"))


def triggers(d):
    return d.get("on") or d.get(True)             # PyYAML reads the bare key `on` as True


def test_every_workflow_parses():
    for f in WF.glob("*.yml"):
        d = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert d["jobs"], f.name


def test_the_claude_review_skips_cleanly_without_the_secret():
    d = load("claude-review.yml")
    job = d["jobs"]["review"]
    text = (WF / "claude-review.yml").read_text(encoding="utf-8")
    assert "pull_request" in triggers(d) and "ANTHROPIC_API_KEY" in text and "claude-sonnet" in text
    steps = job["steps"]
    check = next(s for s in steps if s.get("id") == "key")
    assert "present=false" in check["run"] and "notice" in check["run"], "a missing secret is a notice, not a failure"
    assert "exit 1" not in check["run"]
    gated = [s for s in steps if "claude-code-action" in s.get("uses", "") or s.get("uses", "").startswith("actions/checkout")]
    assert gated and all("steps.key.outputs.present == 'true'" in s.get("if", "") for s in gated)
    assert "claude-review.md" in text and d["permissions"]["contents"] == "read"


def test_the_checklist_covers_what_the_task_asked_for():
    md = (ROOT / ".github" / "claude-review.md").read_text(encoding="utf-8").lower()
    for needle in ("gpu", "pacman", "modprobe", "connectome", "game rule", "tuning", "validation", "migration", "localized",
                   "tr(", "sudo", "libgl", "mesa"):
        assert needle in md, needle


def test_nightly_runs_validation_and_the_playthrough_and_reports():
    d = load("nightly.yml")
    assert "schedule" in triggers(d) and "workflow_dispatch" in triggers(d)
    run = d["jobs"]["run"]
    assert run["runs-on"] == "ubuntu-22.04"
    text = (WF / "nightly.yml").read_text(encoding="utf-8")
    for needle in ("--validate", "--playthrough", "nightly_report.py", "gh-pages", "gh issue", "open_issue"):
        assert needle in text, needle
    assert d["permissions"]["issues"] == "write" and d["permissions"]["contents"] == "write"
    assert "secrets." not in text.replace("secrets.GITHUB_TOKEN", ""), "the nightly needs no secret"


def test_tests_and_release_require_the_selftest_and_the_playthrough():
    t = load("tests.yml")["jobs"]["selftest-playthrough"]
    text = (WF / "tests.yml").read_text(encoding="utf-8")
    assert "--selftest" in text and "--playthrough" in text and t["needs"] == "brainpack"
    r = load("release.yml")
    assert "selftest-playthrough" in r["jobs"]["release"]["needs"]
    rt = (WF / "release.yml").read_text(encoding="utf-8")
    assert "--selftest" in rt and "--playthrough adult" in rt and "-eq 3" in rt, "exit 0 and 3 pass, 1 does not"
    assert rt.count("--selftest") >= 4, "linux tests, windows tests, the AppImage and the exe"
    assert "--playthrough-quick" not in r["jobs"]["selftest-playthrough"]["steps"][-2]["run"], "a release runs the full bot"
