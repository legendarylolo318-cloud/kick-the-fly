#!/usr/bin/env python3
"""Run the test suite in balanced chunks, a few at a time, with a memory guard (3.0 day 4, from the day 3 review's follow-ups).

Why: the whole suite in one process took ~83 min and, before the LiveInputs leak fix, grew to 20 GB; four hand-made chunks run two at a
time took ~15 min. This does that without hand-making them:

  python tools/run_tests.py                    # every test file, 4 chunks, 2 at a time
  python tools/run_tests.py --fast             # without the validation suite (tests/test_validation.py, ~17 min alone)
  python tools/run_tests.py --chunks 6 --parallel 3 --min-free-gb 3
  python tools/run_tests.py tests/test_day4_*.py -- -k duel      # a subset; arguments after -- go to pytest

How:
  - Files are sorted by their recorded duration (tests/.durations.json, written by this tool from each run's JUnit XML) and dealt into
    chunks longest-first (LPT), so the chunks finish together. A file with no recorded duration counts as the median.
  - `--parallel` chunks run at once. Before starting another, the tool waits until the machine has `--min-free-gb` of free memory
    (RAM plus swap-free is NOT counted: swapping makes the tests slower, not safe); it never kills a running chunk.
  - Each chunk's peak memory (its process tree, as the kernel reports it) is printed and recorded, so a file that grows is visible.
  - The exit code is 0 only if every chunk passed.

The tool changes nothing in the game and installs nothing; it needs only the standard library and pytest.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DURATIONS = ROOT / "tests" / ".durations.json"
VALIDATION_FILES = ("tests/test_validation.py",)


def available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")                                         # not Linux: no guard


def load_durations() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(DURATIONS.read_text()).items()}
    except (OSError, ValueError):
        return {}


def deal(files: list[str], durations: dict[str, float], chunks: int) -> list[list[str]]:
    """Longest-processing-time-first: each file goes to the chunk with the least total so far."""
    known = [durations[f] for f in files if f in durations]
    default = statistics.median(known) if known else 1.0
    out: list[list[str]] = [[] for _ in range(max(1, min(chunks, len(files))))]
    load = [0.0] * len(out)
    for f in sorted(files, key=lambda f: -durations.get(f, default)):
        i = load.index(min(load))
        out[i].append(f)
        load[i] += durations.get(f, default)
    return [c for c in out if c]


def junit_durations(path: Path) -> dict[str, float]:
    """Seconds per test file from a pytest JUnit XML (the classname holds the module path)."""
    out: dict[str, float] = {}
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return out
    for case in root.iter("testcase"):
        mod = case.get("classname", "").split(".")
        # tests.test_x.TestClass -> tests/test_x.py
        parts = [p for p in mod if not p[:1].isupper()]
        if not parts:
            continue
        f = "/".join(parts) + ".py"
        out[f] = out.get(f, 0.0) + float(case.get("time", 0.0))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="test files (default: tests/test_*.py)")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=2.5, help="wait for this much free memory before starting a chunk")
    ap.add_argument("--fast", action="store_true", help="leave out the validation suite")
    ap.add_argument("--only-validation", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="do not update tests/.durations.json")
    ap.add_argument("--mem-report", action="store_true",
                    help="record each file's memory growth (resident size after a garbage collection) and print the worst at the end")
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    else:
        extra = []
    args = ap.parse_args(argv)
    files = args.files or sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py"))
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "tests" / "playthrough").glob("test_*.py")] if not args.files else []
    if args.fast:
        files = [f for f in files if f not in VALIDATION_FILES]
    if args.only_validation:
        files = [f for f in files if f in VALIDATION_FILES]
    if not files:
        print("no test files")
        return 2
    durations = load_durations()
    chunks = deal(files, durations, args.chunks)
    print(f"{len(files)} files in {len(chunks)} chunks, {args.parallel} at a time; "
          f"estimated {max(sum(durations.get(f, 0) for f in c) for c in chunks) / 60:.0f} min for the longest chunk")
    tmp = Path(tempfile.mkdtemp(prefix="ktf-tests-"))
    pending = list(enumerate(chunks))
    running: dict[int, tuple[subprocess.Popen, float, Path]] = {}
    results: dict[int, dict] = {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.parallel and (not running or available_gb() >= args.min_free_gb):
            i, c = pending.pop(0)
            xml = tmp / f"chunk{i}.xml"
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *extra, *c]
            log = open(tmp / f"chunk{i}.log", "w")
            env = dict(os.environ, **({"KTF_MEM_REPORT": str(tmp / f"mem{i}.json")} if args.mem_report else {}))
            running[i] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env), time.time(), xml)
            print(f"[{time.time() - t0:5.0f}s] chunk {i} started: {len(c)} files", flush=True)
        for i, (p, started, xml) in list(running.items()):
            try:
                pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if pid == 0:
                continue
            del running[i]
            code = os.waitstatus_to_exitcode(status)
            p.returncode = code
            tail = (tmp / f"chunk{i}.log").read_text().strip().splitlines()[-1:] or [""]
            results[i] = dict(code=code, seconds=time.time() - started, peak_gb=ru.ru_maxrss / 1048576, summary=tail[0], xml=xml)
            print(f"[{time.time() - t0:5.0f}s] chunk {i} {'passed' if code == 0 else 'FAILED'} in {results[i]['seconds'] / 60:.1f} min, "
                  f"peak {results[i]['peak_gb']:.1f} GB: {tail[0]}", flush=True)
        time.sleep(1.0)
    new: dict[str, float] = {}
    for r in results.values():
        new.update(junit_durations(r["xml"]))
    if new and not args.no_record:
        merged = {**durations, **{k: round(v, 1) for k, v in new.items()}}
        DURATIONS.write_text(json.dumps(dict(sorted(merged.items())), indent=1) + "\n")
    if args.mem_report:
        rows = []
        for i in results:
            try:
                rows += [dict(r, chunk=i) for r in json.loads((tmp / f"mem{i}.json").read_text())]
            except (OSError, ValueError):
                pass
        rows.sort(key=lambda r: -r["growth_mb"])
        print("\nmemory growth per test file (MB resident after gc; a large number that never comes back is a leak):")
        for r in rows[:15]:
            print(f"  {r['growth_mb']:>6} MB  {r['file']}  ({r['before_mb']} -> {r['after_mb']}, chunk {r['chunk']})")
        (tmp / "memory.json").write_text(json.dumps(rows, indent=1))
    failed = [i for i, r in results.items() if r["code"] != 0]
    for i in failed:
        print(f"\n--- chunk {i} log ({tmp / f'chunk{i}.log'}) ---")
        print("\n".join((tmp / f"chunk{i}.log").read_text().splitlines()[-40:]))
    print(f"\n{len(results) - len(failed)}/{len(results)} chunks passed in {(time.time() - t0) / 60:.1f} min; logs in {tmp}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
