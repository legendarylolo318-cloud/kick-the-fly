"""Tests for simulation performance benchmark module."""

import json
from pathlib import Path
import pytest
from kickthefly.lab import benchmark


def test_system_info_and_memory():
    info = benchmark.get_system_info()
    assert "os" in info
    assert "cpu_count" in info
    assert info["cpu_count"] >= 1
    assert "cpu_model" in info

    mem = benchmark.get_memory_mb()
    assert isinstance(mem, float)
    assert mem >= 0.0


def test_run_benchmark_fast(tmp_path):
    # Short 0.05s benchmark on 1 fly to verify calculations
    res = benchmark.run_benchmark(fly_counts=(1,), seconds=0.05)
    assert "records" in res
    assert len(res["records"]) == 1
    rec = res["records"][0]
    assert rec["flies"] == 1
    assert "paced_steps_per_s" in rec
    assert "uncapped_steps_per_s" in rec
    assert "neurons_per_sec" in rec
    assert "synapses_per_sec" in rec
    assert "memory_mb" in rec

    report = benchmark.format_benchmark_report(res)
    assert "KICK THE FLY - SIMULATION PERFORMANCE BENCHMARK" in report
    assert "1" in report

    # Test save and load
    out_file = tmp_path / "test_bench.json"
    saved_p = benchmark.save_benchmark_results(res, out_file)
    assert saved_p.exists()
    loaded = benchmark.load_benchmark_results(out_file)
    assert loaded["timestamp"] == res["timestamp"]
    assert len(loaded["records"]) == 1
