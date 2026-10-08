"""3.0 day 3 protocols: the weather, audio and predator blocks, the two new assay kinds, the example files, compatibility."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import needs_pack
from kickthefly.lab import protocol

ROOT = Path(__file__).resolve().parent.parent
NEW = ("predator_escape_assay", "mic_hum_demo", "weather_storm", "predator_mantis_creep", "mic_hum_pulses")


@pytest.mark.parametrize("name", NEW)
def test_the_example_protocols_load_and_check(name):
    p = protocol.load(ROOT / "protocols" / f"{name}.yaml")
    assert p["name"] and (any(k in p for k in ("weather", "audio", "predator", "assay")))


def test_old_protocols_are_unchanged_by_the_new_keys():
    p = protocol.check({"name": "old", "seed": 4, "flies": 2, "stimuli": [{"at_s": 0.1, "target": "loom"}]})
    assert set(p) == {"name", "seed", "flies", "seeds", "warmup_s", "duration_s", "stimuli", "recordings"} | {"recordings"}
    assert p["seeds"] == [4, 5]
    for f in sorted((ROOT / "protocols").glob("*.yaml")):
        if f.stem not in NEW:
            protocol.load(f)


def test_the_new_blocks_are_checked_and_normalised():
    ok = protocol.check(dict(name="x", weather={"rain": 1, "storm": True}, audio={"hz": 200}, predator={"kind": "frog"}))
    assert ok["weather"] == dict(rain=1.0, gust_hz=0.0, storm=True, wind_speed=0.0, wind_dir=180.0)
    assert ok["audio"]["hz"] == 200.0 and ok["audio"]["ipi_ms"] is None and ok["audio"]["seconds"] == 1.0
    assert ok["predator"] == dict(kind="frog", at_s=0.0)
    bad = [dict(weather={"rain": 2}), dict(weather={"rain": "wet"}), dict(weather={"snow": 1}), dict(weather={"storm": "yes"}),
           dict(weather={"wind_speed": float("inf")}), dict(audio={"hz": 5}), dict(audio={"device": "mic0"}), dict(audio={"amp": -1}),
           dict(audio={"seconds": 0}), dict(audio={"ipi_ms": 1}), dict(predator={"kind": "walrus"}), dict(predator={}),
           dict(predator={"kind": "frog", "speed": 9}), dict(weather=[1]), dict(audio="loud")]
    for b in bad:
        with pytest.raises(protocol.ProtocolError):
            protocol.check(dict(name="x", **b))


def test_the_new_blocks_cannot_ride_in_an_assay_or_patch_protocol():
    for blk in (dict(weather={"rain": 0.5}), dict(audio={"hz": 200}), dict(predator={"kind": "frog"})):
        with pytest.raises(protocol.ProtocolError, match="can't be used in an assay"):
            protocol.check(dict(name="x", assay="looming", seed=1, flies=2, **blk))
        with pytest.raises(protocol.ProtocolError, match="stands alone"):
            protocol.check(dict(name="x", patch={"neuron": "type:DNp01"}, **blk))


def test_the_audio_block_has_no_way_to_name_a_device():
    with pytest.raises(protocol.ProtocolError, match="never the microphone"):
        protocol.check(dict(name="x", audio={"device": "default"}))


def test_hum_demo_assay_options_are_checked():
    protocol.check(dict(name="a", seed=1000, flies=2, assay="hum_demo", assay_options=dict(conditions=["silence"], seconds=1.0)))
    for bad in (dict(nope=1), dict(conditions=[]), dict(conditions=["humming_bird"]), dict(seconds=0), dict(amp=2), dict(conditions="silence")):
        with pytest.raises(protocol.ProtocolError):
            protocol.check(dict(name="a", seed=1000, flies=2, assay="hum_demo", assay_options=bad))


@needs_pack
def test_the_weather_audio_and_predator_blocks_run_and_change_what_the_neurons_do(tmp_path):
    base = dict(name="b", seed=1000, flies=1, warmup_s=1, duration_s=6, recordings=[
        {"name": "wing", "neurons": "wing"}, {"name": "jo_a", "neurons": "prefix:JO-A"}, {"name": "loom", "neurons": "loom"}])
    calm = protocol.run_seed(protocol.check(dict(base)), 1000, None, tmp_path, "calm")
    wet = protocol.run_seed(protocol.check(dict(base, weather={"rain": 1.0})), 1000, None, tmp_path, "wet")
    hum = protocol.run_seed(protocol.check(dict(base, audio={"hz": 200, "at_s": 0.5, "seconds": 3, "ipi_ms": 35})), 1000, None, tmp_path, "hum")
    mantis = protocol.run_seed(protocol.check(dict(base, duration_s=14, predator={"kind": "mantis", "at_s": 0.5})), 1000, None, tmp_path, "mantis")
    assert wet["wing"] > 1.2 * calm["wing"], "rain fires the wing touch neurons"
    assert hum["jo_a"] > 5 * max(calm["jo_a"], 0.1), "the synthetic hum fires JO-A"
    assert mantis["loom"] > calm["loom"], "the strike reaches the looming detectors"
    again = protocol.run_seed(protocol.check(dict(base, weather={"rain": 1.0})), 1000, None, tmp_path, "wet2")
    assert again == wet, "deterministic"
