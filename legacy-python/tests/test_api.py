"""The Python API: from kickthefly import Fly (kickthefly/lab/api.py)."""
import numpy as np

from conftest import needs_pack

pytestmark = needs_pack


def test_fly_drive_record_export(tmp_path):
    from kickthefly import Fly

    fly = Fly(seed=5, warmup_s=1.0)
    assert fly.n == 166_700 and fly.backend in ("cpu", "numba", "torch-cpu", "gl", "torch-cuda", "torch-rocm")
    gf = fly.neurons("dnp01")
    assert len(gf) and fly.describe(int(gf[0]))["type"] == "DNp01"
    rec = fly.record({"giant fiber": "dnp01", "looming": "loom"})
    fly.step(1.0)
    fly.drive("loom", amp=0.5)
    last = fly.step(1.0)
    fly.undrive()
    assert last.dtype == bool and last.shape == (fly.n,)
    assert not fly.brain.drive_cur.any()
    calm, driven = rec.rates(0.0, 1.0), rec.rates(1.0, 2.0)
    assert driven["giant fiber"] > 3 * max(calm["giant fiber"], 1.0)          # the validated looming pathway
    t, rows = rec.spikes()
    assert len(t) == len(rows) and (np.diff(t) >= 0).all() and np.isin(rows, np.concatenate(list(rec.groups.values()))).all()
    files = fly.export(tmp_path / "loom")
    names = {p.name for p in files}
    assert {"loom-spikes.csv", "loom-rates.csv", "loom-group-rates.csv"} <= names
    assert all(p.exists() for p in files)


def test_silence_blocks_the_pathway_and_runs_are_reproducible():
    from kickthefly import Fly

    def run(silence):
        fly = Fly(seed=6, warmup_s=1.0, backend="cpu")
        if silence:
            fly.silence("type:LPLC2,LC4")
        rec = fly.record("dnp01")
        fly.drive("loom")
        fly.step(0.5)
        return rec.rates()["dnp01"], rec.spikes()

    a, sa = run(False)
    b, sb = run(False)
    assert a == b and np.array_equal(sa[1], sb[1])                  # same seed, same calls, same spikes
    c, _ = run(True)
    assert c < a / 3                                                 # the drive is cancelled by the lesion
