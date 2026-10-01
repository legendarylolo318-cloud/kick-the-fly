"""3.0 day 2 protocols: the imaging block, the patch kind, the example files, and compatibility (old protocols unchanged, bundles
carry the new blocks and rerun bit-exactly). Synthetic pack: plumbing, not biology."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kickthefly.lab import protocol

ROOT = Path(__file__).resolve().parent.parent
NEW = ("genetics_line_silencing", "thermogenetic_dnp01", "thermogenetic_escape_assay", "patch_dnp01_if_curve", "imaging_looming",
       "drug_picrotoxin", "drug_cholinergic_block")


@pytest.mark.parametrize("name", NEW)
def test_every_example_protocol_loads_and_says_what_it_is(name):
    f = ROOT / "protocols" / f"{name}.yaml"
    p = protocol.load(f)
    head = f.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("#") and ("3.0 day 2" in head)
    assert any(k in p for k in ("thermogenetics", "drug", "imaging", "patch", "assay", "surgery"))
    assert any(f.name == q.name for q in __import__("kickthefly.lab.lab", fromlist=["x"]).protocol_files())      # offered in the Lab


def test_old_protocol_keys_and_defaults_are_unchanged():
    p = protocol.check({"name": "old", "seed": 4, "flies": 2, "stimuli": [{"at_s": 0.1, "target": "loom"}]})
    assert set(p) == {"name", "seed", "flies", "seeds", "warmup_s", "duration_s", "stimuli", "recordings"}
    assert p["seeds"] == [4, 5]


def test_unknown_keys_inside_the_new_blocks_are_refused():
    for bad, msg in ((dict(imaging=dict(nope=1)), "unknown keys"), (dict(imaging=dict(indicator="rcamp")), "indicator"),
                     (dict(imaging=dict(rois=[1, 2])), "rois"), (dict(imaging=dict(fps=0)), "fps"),
                     (dict(drug=dict(doses={"picrotoxin": 2})), "between 0 and 1"),
                     (dict(drug=dict(doses={"picrotoxin": .5}, include_low_confidence="yes")), "true or false")):
        with pytest.raises(protocol.ProtocolError, match=msg):
            protocol.check(dict(name="x", **bad))
    for bad, msg in ((dict(patch=dict()), "neuron"), (dict(patch=dict(neuron="type:X", mode="weird")), "mode"),
                     (dict(patch=dict(neuron="type:X", amplitudes=[99])), "limited"),
                     (dict(patch=dict(neuron="type:X"), stimuli=[]), "stands alone"),
                     (dict(patch=dict(neuron="type:X"), drug=dict(doses={"picrotoxin": .5})), "stands alone")):
        with pytest.raises(protocol.ProtocolError, match=msg):
            protocol.check(dict(name="x", **bad))


def test_the_imaging_block_writes_csv_nwb_and_tiff(synthetic_pack, tmp_path):
    from PIL import Image

    pytest.importorskip("pynwb")
    p = protocol.check(dict(name="im", seed=1, flies=1, warmup_s=0.3, duration_s=1.0, nwb=True,
                            stimuli=[dict(at_s=0.2, for_s=0.3, target="loom", strength=0.9)],
                            imaging=dict(indicator="gcamp6f", fps=10, rois=["type:LPLC2", "type:DNp01"], tiff=True),
                            recordings=[dict(name="gf", neurons="dnp01")]))
    out = tmp_path / "o"
    out.mkdir()
    protocol.run_seed(p, 1, None, out, "run")
    csv = (out / "run-seed1-imaging.csv").read_text(encoding="utf-8").splitlines()
    assert csv[0].startswith("# MODEL") and len(csv) == 3 + 10 and "dFF:type:LPLC2" in csv[2]
    with Image.open(out / "run-seed1-imaging.tif") as t:
        assert t.n_frames == 10 and t.size == (356, 193)
    assert (out / "run-seed1-imaging.nwb").exists()


def test_a_patch_protocol_writes_csv_and_a_summary_with_the_model_tag(synthetic_pack, tmp_path):
    p = protocol.check(dict(name="pc", seed=2, flies=2, patch=dict(neuron="type:MDN", mode="isolated", amplitudes=[0, 0.1, 0.5],
                                                                   duration_ms=200, repeats=2, warmup_s=0.2)))
    folder = protocol.run(p, tmp_path, workers=1)
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert s["tag"].startswith("MODEL") and "Not millivolts" in s["units"] and s["neuron"]["type"] == "MDN"
    rates = [s["rate_hz_by_current"][k]["mean"] for k in ("0", "0.1", "0.5")]
    assert rates[0] <= rates[1] <= rates[2]
    assert len(list(folder.glob("patch-seed*-if.csv"))) == 2 and len(list(folder.glob("patch-seed*-trace.csv"))) == 2
    with pytest.raises(protocol.ProtocolError, match="selects no neurons|unknown"):
        protocol.run_patch(protocol.check(dict(name="pc", patch=dict(neuron="type:Nope"))), tmp_path)


def test_the_cli_runs_a_patch_protocol(synthetic_pack, tmp_path, capsys):
    f = tmp_path / "p.yaml"
    f.write_text("name: cli-patch\nseed: 1\nflies: 1\npatch: {neuron: 'type:MDN', mode: isolated, amplitudes: [0, 0.2], duration_ms: 100, repeats: 1, warmup_s: 0.1}\n")
    assert protocol.run_file(f, tmp_path / "out") == 0
    out = capsys.readouterr().out
    assert "patch clamp (MODEL)" in out and "current" in out and "Not real electrophysiology" in out


def test_replays_refuse_the_new_blocks_and_bundles_carry_and_rerun_them(synthetic_pack, tmp_path):
    from kickthefly.lab import bundle

    p = protocol.check(dict(name="th", seed=3, flies=1, warmup_s=0.2, duration_s=1.0,
                            thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")], temperature_c=33, kinetics="steady"),
                            recordings=[dict(name="gf", neurons="dnp01")]))
    f = tmp_path / "th.yaml"
    import yaml

    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    assert protocol.record_replay(f, tmp_path / "x.ktfreplay", tmp_path) == 2          # refused with a reason, not silently wrong
    folder = protocol.run(p, tmp_path / "run", workers=1)
    z = bundle.create(folder, tmp_path / "b.zip")
    rep = bundle.rerun(z, tmp_path / "rerun", workers=1)
    assert rep["match"] and rep["mode"] == "bit-exact"
    assert "thermogenetics" in json.loads((folder / "protocol.json").read_text(encoding="utf-8"))


def test_a_drug_and_thermogenetics_compose_in_one_protocol(synthetic_pack, tmp_path):
    from kickthefly.sim import wiring as W

    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()
    p = protocol.check(dict(name="both", seed=1, flies=1, warmup_s=0.2, duration_s=0.5,
                            thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")], temperature_c=34, kinetics="steady"),
                            drug=dict(doses=dict(cholinergic=0.5)), recordings=[dict(name="gf", neurons="dnp01")]))
    out = tmp_path / "o"
    out.mkdir()
    assert protocol.run_seed(p, 1, None, out, "run")["gf"] > 0
    for f in (W.synapse_counts, W.edge_pre, W.edge_post):
        f.cache_clear()


# --- 3.0 Day 2 decisions: the F0 time constant is a setting, a protocol key and an API argument ---------------------------------------
def test_the_f0_time_constant_is_checked_and_reaches_the_session():
    from kickthefly.core import config

    for bad in (0, -5, "30", True, float("inf")):
        with pytest.raises(protocol.ProtocolError, match="f0_tau_s"):
            protocol.check(dict(name="x", imaging=dict(f0_tau_s=bad)))
    assert protocol.check(dict(name="x", imaging=dict(f0_tau_s=60)))["imaging"]["f0_tau_s"] == 60
    spec = {s.key: s for s in config.SETTINGS}
    assert spec["brain.imaging_f0_tau_s"].default == 30                     # the default is unchanged
    assert 5 in spec["brain.imaging_fps"].options and 20 in spec["brain.imaging_fps"].options
    assert spec["brain.imaging_f0_tau_s"].tag == config.GAME_RULE


def test_a_slower_f0_settles_later_than_a_faster_one():
    import numpy as np

    from kickthefly.lab import imaging as im

    out = {}
    for tau in (3.0, 30.0):
        rng = np.random.default_rng(0)
        s = im.ImagingSession(40, {"a": np.arange(40)}, "gcamp6s", 20.0, shot_noise=False, baseline_tau_s=tau)
        for _ in range(2000):                                               # 10 s of 10 Hz, then the rate triples for 5 s
            s.push(np.flatnonzero(rng.random(40) < 10 * 0.005))
        for _ in range(1000):
            s.push(np.flatnonzero(rng.random(40) < 30 * 0.005))
        out[tau] = float(np.array(s.true_dff)[-1, 0])
    assert out[30.0] > out[3.0]                                             # the short baseline has already absorbed the rise
