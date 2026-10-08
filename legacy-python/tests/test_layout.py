"""The 2.7 package layout keeps every entry point and every old file working."""
import pickle
import subprocess
import sys

import pytest

from conftest import ROOT


def test_root_shim_exposes_the_entry_point():
    """`python kick_the_fly.py ...` and `python -m kickthefly` must be the same program."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ktf_shim", ROOT / "kick_the_fly.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    from kickthefly.__main__ import main as pkg_main
    assert shim.main is pkg_main


def test_shim_docstring_points_at_the_canonical_one():
    text = (ROOT / "kick_the_fly.py").read_text(encoding="utf-8")
    assert "kickthefly/game/kick_the_fly.py" in text, "the shim must say where the canonical docstring lives"


def test_module_entry_point_runs():
    out = subprocess.run([sys.executable, "-m", "kickthefly", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "--headless" in out.stdout


def test_old_graph_pickles_still_load():
    """graph.pkl files written before the move name the module connectome.loader; they are 260 MB and a 1.1 GB
    download to rebuild, so they must keep loading."""
    import io
    import types

    from kickthefly.sim.connectome import loader

    stub = types.ModuleType("connectome.loader")
    stub.Dataset = loader.Dataset
    sys.modules["connectome"] = types.ModuleType("connectome")
    sys.modules["connectome.loader"] = stub
    was = loader.Dataset.__module__
    try:
        loader.Dataset.__module__ = "connectome.loader"          # pickled the way the pre-2.7 loader did
        legacy = pickle.dumps(loader.Dataset(key="k", filename="f", size=1, md5_b64="x"))
    finally:
        loader.Dataset.__module__ = was
        del sys.modules["connectome.loader"], sys.modules["connectome"]

    assert b"connectome.loader" in legacy
    with pytest.raises(ModuleNotFoundError):                     # a plain load would fail now
        pickle.loads(legacy)
    obj = loader._CompatUnpickler(io.BytesIO(legacy)).load()
    assert obj.filename == "f" and type(obj) is loader.Dataset


def test_no_loose_modules_in_the_repo_root():
    loose = sorted(p.name for p in ROOT.glob("*.py"))
    assert loose == ["kick_the_fly.py"], f"new modules belong in kickthefly/ or tools/, not the root: {loose}"
