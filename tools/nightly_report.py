"""The nightly workflow's report: a history table and trend charts as one static HTML page, and the issue text.

    python tools/nightly_report.py --validation V.json --playthrough P.json --history site/history.json --site site \
        --issue-body issue.md --commit SHA

Reads the night's validation results (`--headless --validate --out`) and playthrough report (`--headless --playthrough
--out DIR`/playthrough.json), appends one entry to history.json, and writes into --site:
  index.html         the history table and inline-SVG trend charts. No scripts, no external requests, no trackers: the
                     page is one file with its CSS inline, and it works offline
  history.json       every night so far (kept to the last 400)
  validation.json, playthrough.md   the night's raw results, linked from the page
It writes --issue-body and prints `open_issue=true|false` (also appended to $GITHUB_OUTPUT) when a validation result
flipped relative to the previous night, or a validation result differs from the expected pass/fail, or the playthrough
failed. Standard library only.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import statistics
import sys
import time
from pathlib import Path

KEEP = 400


def load(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else None
    except (OSError, ValueError):
        return None


def expected() -> dict:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from kickthefly.lab.validation import EXPECTED
        return dict(EXPECTED)
    except Exception:
        return {}


def entry_from(validation: dict | None, playthrough: dict | None, commit: str, when: str | None = None) -> dict:
    e = dict(date=when or time.strftime("%Y-%m-%d"), commit=commit, validation=None, playthrough=None)
    if validation:
        tests = {t["id"]: bool(t["passed"]) for t in validation.get("tests", [])}
        ratios = {t["id"]: t.get("measured", {}).get("drive_ratio_mean") for t in validation.get("tests", [])
                  if isinstance(t.get("measured", {}).get("drive_ratio_mean"), (int, float))}
        e["validation"] = dict(app_version=validation.get("app_version"), backend=validation.get("backend"),
                               passed=sum(tests.values()), total=len(tests), tests=tests, ratios=ratios,
                               seconds=validation.get("seconds"))
    if playthrough:
        res = playthrough.get("results", [])
        ratios = [r["metrics"]["sim_real_ratio"] for r in res if isinstance(r.get("metrics", {}).get("sim_real_ratio"), (int, float))]
        e["playthrough"] = dict(counts=playthrough.get("counts", {}), failed=[r["id"] for r in res if r.get("status") == "fail"],
                                median_sim_real=round(statistics.median(ratios), 2) if ratios else None,
                                backend=playthrough.get("meta", {}).get("backend"))
    return e


def flips(prev: dict | None, cur: dict, exp: dict) -> tuple[list[str], list[str]]:
    """(tests whose result differs from the previous night, tests that differ from the expected pass/fail)."""
    v = cur.get("validation")
    if not v:
        return [], []
    changed, off = [], []
    pv = (prev or {}).get("validation")
    for tid, ok in v["tests"].items():
        if pv and tid in pv["tests"] and pv["tests"][tid] != ok:
            changed.append(f"{tid}: {'PASS' if pv['tests'][tid] else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")
        if tid in exp and exp[tid] != ok:
            off.append(f"{tid}: expected {'PASS' if exp[tid] else 'FAIL'}, got {'PASS' if ok else 'FAIL'}")
    return changed, off


# --- charts ---------------------------------------------------------------------------------------------------------------
def chart(title: str, values: list, labels: list[str], color: str = "#2a7fbf", height: int = 110, integer: bool = False) -> str:
    pts = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    w, h, pad = 360, height, 26
    if not pts:
        return f'<figure><figcaption>{html.escape(title)}</figcaption><p class="none">no data yet</p></figure>'
    lo, hi = min(v for _, v in pts), max(v for _, v in pts)
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    n = max(1, len(values) - 1)

    def xy(i, v):
        return pad + (w - 2 * pad) * (i / n if n else 0.5), h - pad + 4 - (h - pad - 10) * ((v - lo) / (hi - lo))

    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(i, v) for i, v in pts))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6"><title>{html.escape(labels[i])}: {v:g}</title></circle>'
                   for (i, v), (x, y) in zip(pts, (xy(i, v) for i, v in pts)))
    fmt = (lambda v: f"{v:.0f}") if integer else (lambda v: f"{v:.3g}")
    return (f'<figure><figcaption>{html.escape(title)}</figcaption>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">'
            f'<line x1="{pad}" y1="{h - pad + 4}" x2="{w - pad}" y2="{h - pad + 4}" class="axis"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/><g fill="{color}">{dots}</g>'
            f'<text x="2" y="12" class="t">{fmt(hi)}</text><text x="2" y="{h - pad + 4}" class="t">{fmt(lo)}</text>'
            f'<text x="{pad}" y="{h - 6}" class="t">{html.escape(labels[0])}</text>'
            f'<text x="{w - pad}" y="{h - 6}" class="t" text-anchor="end">{html.escape(labels[-1])}</text></svg></figure>')


CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1d23;--mute:#5b6472;--line:#d7dce3;--bad:#b3261e;--ok:#1b7f3b;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#12151a;--fg:#e6e9ee;--mute:#9aa3b0;--line:#2a303a;--bad:#ff8a80;--ok:#7fd99a;--warn:#f0c060}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.mute{color:var(--mute)}.bad{color:var(--bad);font-weight:600}.ok{color:var(--ok)}.warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
.charts{display:flex;flex-wrap:wrap;gap:14px}figure{margin:0;flex:1 1 340px;max-width:380px}figcaption{font-size:13px;color:var(--mute)}
svg{width:100%;height:auto}.axis{stroke:var(--line)}text.t{font-size:10px;fill:var(--mute)}.none{color:var(--mute);font-size:13px}
code{font-size:13px}
"""


def page(history: list[dict], exp: dict) -> str:
    cur = history[-1] if history else None
    rows = []
    prev = None
    for e in history:
        v, p = e.get("validation"), e.get("playthrough")
        ch, off = flips(prev, e, exp)
        vcell = "no run" if not v else f'{v["passed"]}/{v["total"]}' + (f' <span class="bad">flipped: {html.escape("; ".join(ch))}</span>' if ch else "") \
            + (f' <span class="warn">off expected: {html.escape("; ".join(off))}</span>' if off else "")
        if not p:
            pcell = "no run"
        else:
            c = p["counts"]
            pcell = f'{c.get("pass", 0)} pass, ' + (f'<span class="bad">{c.get("fail", 0)} FAIL</span>' if c.get("fail") else "0 fail") \
                + f', {c.get("gated", 0)} gated, {c.get("skip", 0)} skipped' + \
                (f' <span class="bad">{html.escape(", ".join(p["failed"][:4]))}</span>' if p["failed"] else "")
        sha = html.escape(str(e.get("commit", ""))[:8])
        rows.append(f'<tr><td>{html.escape(e["date"])}</td><td><code>{sha}</code></td><td>{vcell}</td><td>{pcell}</td>'
                    f'<td>{(p or {}).get("median_sim_real", "")}</td></tr>')
        prev = e
    labels = [e["date"][5:] for e in history] or ["-"]
    vs = [((e.get("validation") or {}).get("passed")) for e in history]
    fails = [((e.get("playthrough") or {}).get("counts", {}).get("fail")) for e in history]
    sim = [((e.get("playthrough") or {}).get("median_sim_real")) for e in history]
    per_test = {}
    for e in history:
        for tid in ((e.get("validation") or {}).get("ratios") or {}):
            per_test.setdefault(tid, [])
    small = []
    for tid in sorted(per_test):
        series = [((e.get("validation") or {}).get("ratios") or {}).get(tid) for e in history]
        small.append(chart(f"{tid}: drive ratio", series, labels, "#7a4fbf", 90))
    head = "no runs yet"
    if cur:
        v, p = cur.get("validation"), cur.get("playthrough")
        head = f'{html.escape(cur["date"])}: validation {v["passed"]}/{v["total"]} passed' if v else f'{html.escape(cur["date"])}'
        if p:
            head += f', playthrough {p["counts"].get("pass", 0)} passed, {p["counts"].get("fail", 0)} failed'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kick the Fly nightly</title><style>{CSS}</style></head><body><main>
<h1>Kick the Fly nightly</h1><p class="mute">Full validation and the playthrough bot on the CPU backend, every night on ubuntu-22.04.
This page is static: no scripts, no requests, no trackers. Latest: {head}.</p>
<h2>Trends</h2><div class="charts">
{chart("Validation tests passed", vs, labels, "#2a7fbf", integer=True)}
{chart("Playthrough failures", fails, labels, "#b3261e", integer=True)}
{chart("Playthrough median sim/real ratio", sim, labels, "#1b7f3b")}
</div>
<h2>Each validated pathway's drive ratio</h2><div class="charts">{''.join(small) or '<p class="none">no data yet</p>'}</div>
<h2>History</h2>
<table><thead><tr><th>Night</th><th>Commit</th><th>Validation</th><th>Playthrough</th><th>sim/real</th></tr></thead>
<tbody>{''.join(reversed(rows))}</tbody></table>
<p class="mute">Raw results of the latest night: <a href="validation.json">validation.json</a>, <a href="playthrough.md">playthrough.md</a>,
<a href="history.json">history.json</a>.</p></main></body></html>
"""


def issue_text(cur: dict, changed: list[str], off: list[str], failed: list[str], run_url: str) -> tuple[str, str]:
    parts = []
    if changed:
        parts.append("Validation results that flipped since the previous night:\n\n" + "\n".join(f"- {c}" for c in changed))
    if off:
        parts.append("Validation results that differ from the expected pass/fail (`kickthefly/lab/validation.py:EXPECTED`):\n\n"
                     + "\n".join(f"- {c}" for c in off))
    if failed:
        parts.append("Playthrough combos that failed:\n\n" + "\n".join(f"- `{c}`" for c in failed[:40])
                     + ("\n- ..." if len(failed) > 40 else ""))
    kind = "validation result flipped" if changed else "validation differs from expected" if off else "playthrough failed"
    body = (f"The nightly run of {cur['date']} (commit `{str(cur.get('commit', ''))[:8]}`) found a problem.\n\n" + "\n\n".join(parts) +
            f"\n\nRun: {run_url or '(no run URL)'}\n\nNothing was tuned or changed to make it pass. Open the run's artifacts "
            "for validation.json and playthrough.md (tracebacks and numbers).\n\n_Opened automatically by the nightly workflow; "
            "later failures add a comment here instead of a new issue._")
    return f"Nightly: {kind} ({cur['date']})", body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation")
    ap.add_argument("--playthrough")
    ap.add_argument("--history")
    ap.add_argument("--site", required=True)
    ap.add_argument("--issue-body")
    ap.add_argument("--issue-title")
    ap.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    ap.add_argument("--run-url", default="")
    ap.add_argument("--date")
    a = ap.parse_args(argv)
    site = Path(a.site)
    site.mkdir(parents=True, exist_ok=True)
    history = (load(a.history or site / "history.json") or {}).get("nights", [])
    val, play = load(a.validation), load(a.playthrough)
    cur = entry_from(val, play, a.commit, a.date)
    history = [h for h in history if h.get("date") != cur["date"]]              # a re-run replaces the same night
    prev = history[-1] if history else None
    exp = expected()
    changed, off = flips(prev, cur, exp)
    failed = (cur.get("playthrough") or {}).get("failed", [])
    history = (history + [cur])[-KEEP:]
    (site / "history.json").write_text(json.dumps({"nights": history}, indent=1), encoding="utf-8")
    (site / "index.html").write_text(page(history, exp), encoding="utf-8")
    if a.validation and Path(a.validation).exists():
        (site / "validation.json").write_text(Path(a.validation).read_text(encoding="utf-8"), encoding="utf-8")
    if a.playthrough and Path(a.playthrough).with_suffix(".md").exists():
        (site / "playthrough.md").write_text(Path(a.playthrough).with_suffix(".md").read_text(encoding="utf-8"), encoding="utf-8")
    problem = bool(changed or off or failed)
    if problem:
        title, body = issue_text(cur, changed, off, failed, a.run_url)
        if a.issue_body:
            Path(a.issue_body).write_text(body, encoding="utf-8")
        if a.issue_title:
            Path(a.issue_title).write_text(title, encoding="utf-8")
    line = f"open_issue={'true' if problem else 'false'}"
    print(line)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
