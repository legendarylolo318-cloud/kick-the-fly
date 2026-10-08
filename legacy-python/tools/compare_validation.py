#!/usr/bin/env python3
"""Compare two `--headless --validate --out FILE` results: the same tests, the expected PASS/FAIL count, and every number identical.

    python tools/compare_validation.py BASELINE.json NEW.json [--tests 21] [--passed 12]

Only the run's bookkeeping may differ (when it ran, how long it took, the app version, the worker count); anything else that differs is
printed and the exit code is 1. Used by .github/workflows/release-gate.yml against a run of the previous release tag (3.1.0 review).
"""
from __future__ import annotations

import argparse
import json
import sys

IGNORED = {"created", "seconds", "app_version", "workers"}


def flatten(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if not path and k in IGNORED:
                continue
            yield from flatten(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{path}[{i}]")
    else:
        yield path, x


def tests_of(res: dict) -> list:
    t = res.get("tests") or res.get("results") or []
    return list(t.values()) if isinstance(t, dict) else list(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--tests", type=int, default=21)
    ap.add_argument("--passed", type=int, default=12)
    a = ap.parse_args(argv)
    base = json.load(open(a.baseline, encoding="utf-8"))
    new = json.load(open(a.new, encoding="utf-8"))
    ok = True
    for name, res in (("baseline", base), ("new", new)):
        t = tests_of(res)
        n_pass = sum(1 for x in t if x.get("passed"))
        print(f"{name}: {len(t)} tests, {n_pass} PASS / {len(t) - n_pass} FAIL")
        if len(t) != a.tests or n_pass != a.passed:
            print(f"  expected {a.tests} tests, {a.passed} PASS / {a.tests - a.passed} FAIL")
            ok = False
    fb, fn = dict(flatten(base)), dict(flatten(new))
    diffs = sorted(k for k in fb.keys() | fn.keys() if fb.get(k, "<missing>") != fn.get(k, "<missing>"))
    numbers = sum(1 for v in fb.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    print(f"{numbers} numbers in the baseline; {len(diffs)} fields differ (ignoring {', '.join(sorted(IGNORED))})")
    for k in diffs[:40]:
        print(f"  {k}: {fb.get(k, '<missing>')!r} -> {fn.get(k, '<missing>')!r}")
    if diffs:
        ok = False
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
