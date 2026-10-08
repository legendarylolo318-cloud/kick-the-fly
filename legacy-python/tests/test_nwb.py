"""NWB export: a recording written with pynwb reads back with the same spikes, context and metadata."""
import numpy as np
import pytest

from conftest import needs_pack

pynwb = pytest.importorskip("pynwb", reason="NWB export is optional (pip install pynwb)")


def _recorded_brain(seconds=0.5):
    """A short lockstep recording of a real brain, with stimuli, events and kinematics logged."""
    from kickthefly.core import simcore
    from kickthefly.lab import lab, recorder

    br = simcore.new_brain(seed=7, warmup=120)
    rows = {"giant_fiber": lab.resolve_group(br, "dnp01"), "looming": lab.resolve_group(br, "loom")}
    rec = recorder.Recorder(br, rows).start()
    steps = int(seconds / 0.005)
    for i in range(steps):
        if i == 10:
            br.poke("head", None, 0.8)
        if i == 20:
            rec.log_event("tool", "swatter", "x=100 y=200")
        if i == 22:
            rec.log_event("hit", "head", "swatter", 0.7)
        if i == 26:
            rec.log_event("note", "DODGE", "DODGE    giant fiber DNp01 x7.2 [real]")
        rec.log_kinematics(1.0 + i * 0.01, 2.0, 0.0, 0.5, 100.0 - i, "idle", "room")
        simcore.step(br, 1)
    rec.stop()
    return br, rec


@needs_pack
def test_nwb_round_trip(tmp_path):
    from kickthefly.lab import nwbexport, recorder

    br, rec = _recorded_brain()
    assert nwbexport.available() is None
    meta = recorder.metadata(br, None, dict(recorded_live=False))
    path = nwbexport.write(rec, tmp_path / "run.nwb", meta)
    assert path.exists() and path.stat().st_size > 0

    a = rec.arrays()
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        f = io.read()
        # units: one per recorded neuron, with the connectome's labels and every spike
        units = f.units.to_dataframe()
        assert len(units) == len(rec.rows)
        assert set(units.columns) >= {"spike_times", "row", "body_id", "cell_type", "instance", "superclass",
                                      "region", "recording_group", "mean_rate_hz"}
        assert sum(len(t) for t in units["spike_times"]) == len(a["spike_index"])
        assert "DNp01" in set(units["cell_type"])
        for t in units["spike_times"]:
            if len(t):
                assert np.allclose(np.asarray(t) / 0.005, np.round(np.asarray(t) / 0.005))  # exact 5 ms steps

        # rates, per group and per region
        rates = f.processing["ecephys"].data_interfaces
        assert "group_rate_giant_fiber" in rates and any(k.startswith("region_rate_") for k in rates)
        assert rates["group_rate_giant_fiber"].rate == pytest.approx(20.0)

        # stimuli and events kept their timing
        stim = f.intervals["stimuli"].to_dataframe()
        assert list(stim["sensory_group"])[0] == "head"
        assert stim["start_time"].iloc[0] == pytest.approx(10 * 0.005, abs=1e-6)
        ev = f.intervals["events"].to_dataframe()
        assert set(ev["kind"]) == {"tool", "hit", "note"}
        assert float(ev[ev["kind"] == "hit"]["value"].iloc[0]) == pytest.approx(0.7)

        # behaviour
        beh = f.processing["behavior"]
        pos = beh["Position"]["body_position"]
        assert pos.data.shape == (100, 3) and pos.timestamps[0] == 0.0
        assert beh["body_state"]["health"].data[0] == pytest.approx(100.0)

        # learned synapses before and after, and the subject/citation metadata
        w = f.scratch["kc_mbon_weights"].to_dataframe()
        assert len(w) > 10_000 and {"weight_connectome", "weight_before", "weight_after"} <= set(w.columns)
        assert np.allclose(w["weight_before"], w["weight_after"])          # nothing taught it in this run
        assert f.subject.species == "Drosophila melanogaster"
        assert "CC BY 4.0" in f.data_collection and "MaleCNS v1.0" in f.data_collection
        def scratch_text(name):
            d = f.scratch[name].data
            d = d[()] if hasattr(d, "shape") and d.shape == () else d
            return d.decode() if isinstance(d, bytes) else str(d)

        saved = scratch_text("kickthefly_metadata")
        assert '"app_version"' in saved and '"lif_params"' in saved and '"reaction_thresholds"' in saved
        assert '"connectome"' in saved and '"brain_pack"' in saved and '"seed"' in saved
        rules = scratch_text("game_rules")
        assert "from_the_connectome" in rules and "game_rules" in rules and "CC BY 4.0" in rules
        assert scratch_text("arena_and_mode").count("arena") == 1
        surg = f.scratch["surgery"].to_dataframe()
        assert {"target", "kind", "mode"} <= set(surg.columns)


@needs_pack
def test_nwb_inspector_finds_nothing_critical(tmp_path):
    """The file passes the NWB inspector's BEST_PRACTICE_VIOLATION bar and above, where the inspector is installed."""
    nwbinspector = pytest.importorskip("nwbinspector", reason="nwbinspector is a dev-only extra")
    from kickthefly.lab import nwbexport, recorder

    br, rec = _recorded_brain(0.2)
    path = nwbexport.write(rec, tmp_path / "run.nwb", recorder.metadata(br, None))
    bad = [m for m in nwbinspector.inspect_nwbfile(str(path))
           if m.importance.name in ("CRITICAL", "PYNWB_VALIDATION", "ERROR")]
    assert not bad, "\n".join(f"{m.importance.name}: {m.check_function_name}: {m.message}" for m in bad)


def test_export_degrades_without_pynwb(monkeypatch):
    """Without pynwb the Lab screen must say so instead of the CSV export breaking."""
    import builtins

    from kickthefly.lab import nwbexport

    real = builtins.__import__

    def no_pynwb(name, *a, **kw):
        if name == "pynwb":
            raise ImportError("no pynwb here")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_pynwb)
    why = nwbexport.available()
    assert why and "pip install pynwb" in why
