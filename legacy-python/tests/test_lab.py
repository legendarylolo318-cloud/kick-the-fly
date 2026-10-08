"""Protocols, recording/export, statistics and the brain smoke test."""
import json

import numpy as np
import pytest

from conftest import ROOT, needs_pack


# --- no brain pack needed ---------------------------------------------------------------------------------------------
def test_mean_ci_and_paired_stats():
    from kickthefly.lab import labstats
    c = labstats.mean_ci([1.0, 2.0, 3.0, 4.0])
    assert c["mean"] == 2.5 and c["n"] == 4 and c["lo"] < 2.5 < c["hi"]
    assert abs((c["hi"] - c["lo"]) / 2 - 3.182446 * np.std([1, 2, 3, 4], ddof=1) / 2) < 1e-4   # t(0.975, 3)
    assert np.isnan(labstats.mean_ci([5.0])["lo"])
    p = labstats.paired([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0])
    assert p["p_value"] < 0.05 and p["mean_difference"] == 4.5
    assert labstats.paired([1, 1], [1, 1])["p_value"] == 1.0
    assert labstats.fisher(10, 10, 0, 10) < 0.001


@pytest.mark.parametrize("text, message", [
    ("name: x\nflies: two\n", "flies"),
    ("name: x\nbogus: 1\n", "unknown keys"),
    ("name: x\nassay: optomotor\n", "assay must be"),
    ("name: x\nsurgery: {\"type:DNp01\": 0}\n", "silence"),
    ("name: x\nparams: {learning: 2}\n", "unknown params"),
    ("name: x\nstimuli: [{target: loom}]\n", "at_s"),
    ("- just\n- a list\n", "mapping"),
    ("name: [unclosed\n", "YAML"),
])
def test_protocol_errors(tmp_path, text, message):
    from kickthefly.lab import protocol
    f = tmp_path / "p.yaml"
    f.write_text(text)
    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.load(f)


def test_example_protocols_are_valid():
    from kickthefly.lab import protocol
    files = sorted((ROOT / "protocols").glob("*.yaml"))
    assert {f.name for f in files} >= {"smoke.yaml", "looming-giant-fiber.yaml", "kc-silencing-tmaze.yaml"}
    for f in files:
        p = protocol.load(f)
        assert p["seeds"]


# --- with the brain pack ---------------------------------------------------------------------------------------------
@needs_pack
def test_brain_pack_loads_and_runs_one_second():
    from kickthefly.core import simcore
    g, W, soma = simcore.pack()
    assert g.n == 166_700 and W.shape == (g.n, g.n) and W.nnz > 10_000_000
    br = simcore.new_brain(seed=0, warmup=0)
    spikes = 0
    for _ in range(200):                                       # 1 s of brain time
        br._step()
        spikes += int(br.sim.spikes.sum())
    assert br.steps == 200 and spikes > 0
    rate = spikes / g.n / 1.0
    assert 0.5 < rate < 50                                     # a live, non-saturated brain


@needs_pack
def test_giant_fiber_exceeds_baseline_when_looming_detectors_are_driven():
    from kickthefly.lab import assays
    from kickthefly.core import simcore
    br = simcore.new_brain(seed=5)
    g = assays.groups(br)
    r = assays.pathway_response(br, g["loom"], {"dnp01": g["dnp01"]}, pre=200, stim=200)["dnp01"]
    assert r[1] > 3 * max(r[0], 0.5), r


@needs_pack
def test_smoke_protocol_and_export_files(tmp_path):
    from kickthefly.lab import protocol
    p = protocol.load(ROOT / "protocols" / "smoke.yaml")
    folder = protocol.run(p, tmp_path, workers=1)
    names = {f.name for f in folder.iterdir()}
    assert {"summary.json", "protocol.json", "run-seed1-spikes.csv", "run-seed1-rates.csv", "run-seed1-group-rates.csv",
            "run-seed1.npz", "run-seed1-metadata.json"} <= names
    meta = json.loads((folder / "run-seed1-metadata.json").read_text())
    for key in ("app_version", "seed", "connectome", "brain_pack", "lif_params", "surgery_active", "recording"):
        assert key in meta
    z = np.load(folder / "run-seed1.npz")
    assert len(z["spike_steps"]) == meta["recording"]["spikes"] > 0
    assert set(np.unique(z["types"])) == {"DNp01", "LPLC2", "LC4"}
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["mean_rate_hz"]["run"]["giant_fiber"]["mean"] > 10
    lines = (folder / "run-seed1-spikes.csv").read_text().splitlines()
    assert lines[0] == "time_ms,row,body_id,type,instance,group" and len(lines) - 1 == meta["recording"]["spikes"]


@needs_pack
def test_protocol_with_surgery_runs_control(tmp_path):
    from kickthefly.lab import protocol
    p = protocol.check(dict(name="t", seed=3, flies=2, warmup_s=0.2, duration_s=0.6, surgery={"type:LPLC2,LC4": -1},
                            stimuli=[dict(at_s=0.1, for_s=0.4, target="loom", strength=0.9)],
                            recordings=[dict(name="gf", neurons="dnp01"), dict(name="lc", neurons="loom")]))
    folder = protocol.run(p, tmp_path)
    s = json.loads((folder / "summary.json").read_text())
    assert set(s["mean_rate_hz"]) == {"treated", "control"}
    assert s["mean_rate_hz"]["treated"]["lc"]["mean"] < 1 < s["mean_rate_hz"]["control"]["lc"]["mean"]
    assert "comparison" in s


@needs_pack
def test_lab_job_with_surgery_pairs_controls():
    from kickthefly.lab import labjobs
    res = labjobs.run_sync("sugar", [11, 12], options=dict(doses=(0.0, 1.0), repeats=1), surgery={"sweet": -1}, workers=1)
    assert res["control"] and res["comparison"]["overall"]["n"] == 2
    top_t = res["treated"]["rows"][-1]["mn9_ratio"]["mean"]
    top_c = res["control"]["rows"][-1]["mn9_ratio"]["mean"]
    assert top_c > top_t                                       # silencing the sugar-pathway GRNs removes the response


def test_model_assumptions_disclosure():
    from kickthefly.lab import lab
    from kickthefly.game import kick_the_fly as k2

    titles = [a[0].lower() for a in lab.ASSUMPTIONS]
    contents = " ".join(f"{a[0]} {a[2]} {a[3]} {a[4]}".lower() for a in lab.ASSUMPTIONS)

    assert any("raw synapse count" in t for t in titles)
    assert any("uniform synaptic efficacy" in t for t in titles)
    assert any("leaky integrate-and-fire" in t or "point neuron" in t for t in titles)
    assert any("no neurotransmitter" in t or "receptor kinetics" in t for t in titles)
    assert any("nmda" in t for t in titles)
    assert any("tonic" in t and "bias" in t for t in titles)
    assert "0.20" in contents
    assert any("gaussian" in t and "noise" in t for t in titles)
    assert "0.05" in contents
    assert any("asymmetry" in t for t in titles)
    assert "reconstruction" in contents or "artifact" in contents

    # Every assumption must cite a doc or code location
    for a in lab.ASSUMPTIONS:
        assert len(a[4]) > 5, f"Missing doc reference for assumption {a[0]}"

    # lab_assumptions page is registered and in lab_pages
    pages = [p[1] for p in k2.Game.lab_pages(None)]
    assert "lab_assumptions" in pages

