"""Experiment bundles (3.0): create, verify, rerun (bit-exact and statistical), and every refusal.
Runs on the synthetic pack (tests/synthetic_pack.py): it checks the machinery, not any result."""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pytest

from kickthefly.lab import bundle, protocol

PROTO = {"name": "bundle-test", "seed": 7, "flies": 2, "warmup_s": 0.1, "duration_s": 0.6,
         "stimuli": [{"at_s": 0.1, "for_s": 0.3, "target": "loom", "strength": 0.8}],
         "recordings": [{"name": "loom", "neurons": "loom"}, {"name": "gf", "neurons": "dnp01"}]}


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """One protocol run and its bundle, shared by the tests (a run takes a few seconds)."""
    import synthetic_pack
    from kickthefly.core import neurodex
    from kickthefly.sim import brainpack

    root = tmp_path_factory.mktemp("bundle")
    pack = synthetic_pack.build(root / "kick_brain.npz")
    mp = pytest.MonkeyPatch()
    mp.setenv("KICK_THE_FLY_HOME", str(root / "home"))
    mp.setenv("KICK_THE_FLY_SIM_BACKEND", "cpu")
    mp.setattr(brainpack, "find", lambda brain="adult": pack)
    from kickthefly.core import paths
    paths.reset_cache()
    neurodex.reset_cache()
    p = protocol.check(dict(PROTO), "test")
    folder = protocol.run(p, root / "run", workers=1)
    z = bundle.create(folder, root / "exp.zip")
    yield dict(root=root, folder=folder, zip=z, pack=pack)
    mp.undo()
    paths.reset_cache()


@pytest.fixture
def env(made, monkeypatch):
    from kickthefly.core import paths
    from kickthefly.sim import brainpack

    monkeypatch.setenv("KICK_THE_FLY_HOME", str(made["root"] / "home"))
    monkeypatch.setenv("KICK_THE_FLY_SIM_BACKEND", "cpu")
    monkeypatch.setattr(brainpack, "find", lambda brain="adult": made["pack"])
    paths.reset_cache()
    return made


def rewrite(src: Path, dst: Path, edit_meta=None, edit_files=None, drop=()):
    """A copy of a bundle with edited metadata / files and a consistent crate (what a careless author would produce)."""
    with zipfile.ZipFile(src) as z:
        data = {n: z.read(n) for n in z.namelist()}
    meta = json.loads(data["metadata.json"])
    if edit_meta:
        edit_meta(meta)
    for n in drop:
        data.pop(n, None)
    for n, f in (edit_files or {}).items():
        data[n] = f(data[n])
    data.pop("metadata.json"), data.pop("ro-crate-metadata.json"), data.pop("README.txt", None)
    return bundle._write(dst, meta, data, None)


# --- what is in the bundle -----------------------------------------------------------------------------------------------
def test_bundle_holds_the_documented_parts(env):
    b = bundle.inspect(env["zip"])
    names = set(b.names())
    assert {"protocol.yaml", "metadata.json", "ro-crate-metadata.json", "README.txt", "results/summary.json"} <= names
    assert any(n.startswith("raw/") and n.endswith("-spikes.csv") for n in names)
    assert any(n.startswith("raw/") and n.endswith(".npz") for n in names)
    m = b.meta
    for key in ("app_version", "backend", "dtype", "seeds", "brain_pack_sha256", "parameters", "surgery", "individuality",
                "arena", "rerunnable", "flies", "platform", "python"):
        assert key in m, key
    assert m["seeds"] == [7, 8] and m["backend"] == "cpu" and m["dtype"] == "float32" and m["rerunnable"]
    from kickthefly.core import replay
    assert m["brain_pack_sha256"] == replay.pack_sha256()
    assert set(m["flies"]) == {"run-seed7", "run-seed8"} and all(len(f["spike_sha256"]) == 64 for f in m["flies"].values())
    assert b.protocol()["stimuli"][0]["target"] == "loom"
    assert b.verify() == []


def test_ro_crate_is_a_valid_looking_crate_with_hashes(env):
    b = bundle.inspect(env["zip"])
    c = b.crate
    assert c["@context"] == bundle.CRATE_CONTEXT
    g = {e["@id"]: e for e in c["@graph"]}
    desc = g["ro-crate-metadata.json"]
    assert desc["conformsTo"] == {"@id": bundle.CRATE_SPEC} and desc["about"] == {"@id": "./"}
    root = g["./"]
    assert root["@type"] == "Dataset" and root["name"] and root["datePublished"] and root["description"]
    assert {p["@id"] for p in root["hasPart"]} == {n for n in b.names() if n != "ro-crate-metadata.json"}
    for n in b.names():
        if n == "ro-crate-metadata.json":
            continue
        assert g[n]["@type"] == "File" and len(g[n]["sha256"]) == 64 and g[n]["contentSize"] == len(b.read(n))
    assert g["#malecns-v1.0"]["license"] == bundle.CC_BY_4            # the dataset's license, not one invented for the bundle
    assert "license" not in root
    assert g["#run"]["@type"] == "CreateAction" and g["#run"]["object"] == {"@id": "protocol.yaml"}


def test_bundle_bytes_do_not_depend_on_file_timestamps(env, tmp_path):
    a = json.loads(bundle.inspect(env["zip"]).read("metadata.json"))
    z1 = rewrite(env["zip"], tmp_path / "a.zip")
    time.sleep(1.1)
    z2 = rewrite(env["zip"], tmp_path / "b.zip")
    assert z1.read_bytes() == z2.read_bytes() and a["created"]


# --- rerun ------------------------------------------------------------------------------------------------------------------
def test_rerun_on_cpu_is_bit_exact(env, tmp_path):
    rep = bundle.rerun(env["zip"], tmp_path / "out", workers=1)
    assert rep["mode"] == "bit-exact" and rep["match"] and rep["verdict"] == "MATCH (bit-exact)"
    assert all(d["identical"] for d in rep["details"]) and len(rep["details"]) == 2
    assert (tmp_path / "out" / "rerun_report.json").exists()
    assert rep["recorded_backend"] == rep["rerun_backend"] == "cpu"


def test_environment_is_restored_after_a_rerun(env, tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("KICK_THE_FLY_SIM_DTYPE", "float32")
    monkeypatch.delenv("KICK_THE_FLY_SIM_BACKEND", raising=False)
    bundle.rerun(env["zip"], tmp_path / "out", workers=1)
    assert "KICK_THE_FLY_SIM_BACKEND" not in os.environ and os.environ["KICK_THE_FLY_SIM_DTYPE"] == "float32"


def test_a_changed_spike_fingerprint_is_a_mismatch(env, tmp_path):
    def bad(meta):
        meta["flies"]["run-seed8"]["spike_sha256"] = "0" * 64

    z = rewrite(env["zip"], tmp_path / "bad.zip", edit_meta=bad)
    rep = bundle.rerun(z, tmp_path / "out", workers=1)
    assert not rep["match"] and rep["verdict"].startswith("MISMATCH")
    assert [d["identical"] for d in rep["details"]].count(False) == 1
    assert "FAIL run-seed8 spikes" in bundle.format_report(rep)


def test_a_different_dtype_or_gpu_backend_is_judged_statistically(env, tmp_path):
    """Recorded on a GPU backend: spikes are not compared, the rates are (and the same run is of course consistent)."""
    z = rewrite(env["zip"], tmp_path / "gpu.zip", edit_meta=lambda m: m.update(backend="torch-cuda"))
    rep = bundle.rerun(z, tmp_path / "out", workers=1)
    assert rep["mode"] == "statistical" and rep["match"] and rep["verdict"].startswith("MATCH (statistically")
    assert all("ok" in d for d in rep["details"])


def test_statistical_mode_fails_outside_the_declared_tolerance(env, tmp_path):
    def shift(data):
        s = json.loads(data)
        for groups in s["mean_rate_hz"].values():
            for c in groups.values():
                c["mean"] = c["mean"] * 3 + 50.0
                c["lo"] = c["hi"] = float("nan")
        return json.dumps(s).encode()

    z = rewrite(env["zip"], tmp_path / "gpu2.zip", edit_meta=lambda m: m.update(backend="gl"),
                edit_files={"results/summary.json": shift})
    rep = bundle.rerun(z, tmp_path / "out", workers=1)
    assert rep["mode"] == "statistical" and not rep["match"] and rep["verdict"].startswith("DIFFERS")


def test_tolerance_rule_is_the_declared_one():
    assert bundle._ci_ok({"mean": 10.0, "lo": 9.0, "hi": 11.0, "n": 5}, 12.0)[0]          # 1 + 10% of 10 = 2
    assert not bundle._ci_ok({"mean": 10.0, "lo": 9.0, "hi": 11.0, "n": 5}, 12.5)[0]
    assert bundle._ci_ok({"mean": 10.0, "lo": float("nan"), "hi": float("nan"), "n": 1}, 13.0)[0]    # 25% + 0.5 Hz = 3
    assert not bundle._ci_ok({"mean": 10.0, "lo": float("nan"), "hi": float("nan"), "n": 1}, 13.6)[0]
    a = {"treated": {"pi": {"mean": 1.0, "lo": 0.8, "hi": 1.2, "n": 4}, "per_fly": [1, 2]}, "rows": [{"x": {"mean": 2.0, "n": 3, "lo": 1, "hi": 3}}]}
    b = {"treated": {"pi": {"mean": 1.1, "n": 4}}, "rows": [{"x": {"mean": 2.5, "n": 3}}]}
    got = list(bundle._walk_means(a, b))
    assert [p for p, _, _ in got] == ["/treated/pi", "/rows[0]/x"]


# --- refusals --------------------------------------------------------------------------------------------------------------------
def test_a_damaged_bundle_is_refused(env, tmp_path):
    z = tmp_path / "dmg.zip"
    with zipfile.ZipFile(env["zip"]) as src, zipfile.ZipFile(z, "w") as dst:
        for n in src.namelist():
            blob = src.read(n)
            if n == "protocol.yaml":
                blob = blob.replace(b"loom", b"LOOM")
            dst.writestr(n, blob)
    assert bundle.inspect(z).verify()
    with pytest.raises(bundle.BundleError, match="damaged.*protocol.yaml"):
        bundle.rerun(z, tmp_path / "out")


def test_a_different_brain_pack_is_refused(env, tmp_path):
    z = rewrite(env["zip"], tmp_path / "other.zip", edit_meta=lambda m: m.update(brain_pack_sha256="ab" * 32))
    with pytest.raises(bundle.BundleError, match="different brain pack"):
        bundle.rerun(z, tmp_path / "out")
    z = rewrite(env["zip"], tmp_path / "none.zip", edit_meta=lambda m: m.update(brain_pack_sha256=None))
    with pytest.raises(bundle.BundleError, match="doesn't record which brain pack"):
        bundle.rerun(z, tmp_path / "out")


def test_no_brain_pack_here_is_refused(env, tmp_path, monkeypatch):
    from kickthefly.sim import brainpack

    monkeypatch.setattr(brainpack, "find", lambda brain="adult": None)
    with pytest.raises(bundle.BundleError, match="no brain pack"):
        bundle.rerun(env["zip"], tmp_path / "out")


def test_a_newer_format_is_refused(env, tmp_path):
    z = rewrite(env["zip"], tmp_path / "new.zip", edit_meta=lambda m: m.update(format_version=2))
    with pytest.raises(bundle.BundleError, match="newer format"):
        bundle.inspect(z)


def test_not_a_bundle_is_refused(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"not a zip")
    with pytest.raises(bundle.BundleError, match="not a readable zip"):
        bundle.inspect(p)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(bundle.BundleError, match="metadata.json is missing"):
        bundle.inspect(p)


def test_zip_slip_names_are_refused(tmp_path):
    p = tmp_path / "evil.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("../evil.txt", "x")
        z.writestr("metadata.json", "{}")
        z.writestr("ro-crate-metadata.json", "{}")
    with pytest.raises(bundle.BundleError, match="unsafe file names"):
        bundle.inspect(p)


def test_a_live_recording_bundle_says_why_it_cannot_rerun(env, tmp_path):
    raw = tmp_path / "rec-spikes.csv"
    raw.write_text("time_ms,row\n1.0,3\n", encoding="utf-8")
    z = bundle.create_live([raw], {"arena": "room", "surgery": {}, "parameters": {}, "seeds": [0]}, tmp_path / "live.zip",
                           {"name": "live-session", "note": "recorded by hand"})
    b = bundle.inspect(z)
    assert b.meta["kind"] == "live-recording" and not b.meta["rerunnable"] and b.verify() == []
    with pytest.raises(bundle.BundleError, match="recorded live"):
        bundle.rerun(z, tmp_path / "out")


def test_a_run_from_before_spike_fingerprints_is_not_rerunnable(env, tmp_path):
    old = tmp_path / "old-run"
    import shutil

    shutil.copytree(env["folder"], old)
    for f in old.glob("*-metadata.json"):
        m = json.loads(f.read_text())
        m.pop("spike_sha256")
        f.write_text(json.dumps(m))
    z = bundle.create(old, tmp_path / "old.zip")
    assert not bundle.inspect(z).meta["rerunnable"]
    with pytest.raises(bundle.BundleError, match="before 3.0"):
        bundle.rerun(z, tmp_path / "out")


def test_not_a_run_folder_is_refused(tmp_path):
    with pytest.raises(bundle.BundleError, match="not a protocol run folder"):
        bundle.create(tmp_path, tmp_path / "x.zip")


def test_spike_fingerprint_is_order_free_and_sensitive():
    import numpy as np

    a = bundle.spike_sha256(np.array([1, 2, 2]), np.array([5, 7, 3]), 10)
    b = bundle.spike_sha256(np.array([2, 1, 2]), np.array([3, 5, 7]), 10)
    assert a == b
    assert a != bundle.spike_sha256(np.array([1, 2, 2]), np.array([5, 7, 4]), 10)
    assert a != bundle.spike_sha256(np.array([1, 2, 2]), np.array([5, 7, 3]), 11)


# --- the command line ---------------------------------------------------------------------------------------------------------
def test_cli_rerun_exit_codes(env, tmp_path, capsys):
    from types import SimpleNamespace

    ok = bundle.main(SimpleNamespace(rerun_bundle=str(env["zip"]), out=str(tmp_path / "o1"), backend=None, sim_backend=None,
                                     dtype=None, workers=1))
    assert ok == 0 and "MATCH (bit-exact)" in capsys.readouterr().out
    bad = rewrite(env["zip"], tmp_path / "b.zip", edit_meta=lambda m: m["flies"]["run-seed7"].update(spike_sha256="1" * 64))
    assert bundle.main(SimpleNamespace(rerun_bundle=str(bad), out=str(tmp_path / "o2"), backend=None, sim_backend=None,
                                       dtype=None, workers=1)) == 1
    assert bundle.main(SimpleNamespace(rerun_bundle=str(env["zip"]), out=None, backend=None, sim_backend=None, dtype=None,
                                       workers=1)) == 2
    assert bundle.main(SimpleNamespace(rerun_bundle=str(tmp_path / "missing.zip"), out=str(tmp_path / "o3"), backend=None,
                                       sim_backend=None, dtype=None, workers=1)) == 2


def test_protocol_cli_writes_a_bundle(env, tmp_path, capsys):
    f = tmp_path / "p.yaml"
    import yaml
    f.write_text(yaml.safe_dump(dict(PROTO, name="via-cli", flies=1)))
    rc = protocol.run_file(f, tmp_path / "runs", workers=1, bundle_to=tmp_path / "cli.zip")
    assert rc == 0 and bundle.inspect(tmp_path / "cli.zip").meta["protocol_name"] == "via-cli"
    assert "bundle written" in capsys.readouterr().out
