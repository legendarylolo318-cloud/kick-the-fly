"""Sim-time vs real-time for N independent fly brains, run the way the game runs them (one Brain thread per fly).

    python tools/bench_sim.py --flies 1 8 16 --seconds 5

Reports, per fly count: steps/s per brain when paced at real time (200 steps/s = 1.00x keeps up), uncapped
throughput, neurons/second, and memory footprint.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kickthefly.lab import benchmark  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Sim-time vs real-time benchmark for 1, 8, 16 flies.")
    ap.add_argument("--flies", type=int, nargs="+", default=[1, 8, 16], help="Number of flies to benchmark (default: 1 8 16)")
    ap.add_argument("--seconds", type=float, default=5.0, help="Measurement duration per run in seconds (default: 5.0)")
    ap.add_argument("--out", type=str, help="Where to save JSON benchmark results")
    args = ap.parse_args()

    res = benchmark.run_benchmark(fly_counts=tuple(args.flies), seconds=args.seconds)
    print(benchmark.format_benchmark_report(res))
    if args.out:
        out_p = benchmark.save_benchmark_results(res, args.out)
        print(f"Results written to {out_p}")
    else:
        benchmark.save_benchmark_results(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
