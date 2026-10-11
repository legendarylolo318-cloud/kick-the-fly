"""Golden for game::protocol_run: small protocols of each kind run by Python's protocol.run on the real pack (individuality off)."""
import json, sys, hashlib, tempfile
sys.path.insert(0, "legacy-python")
from pathlib import Path
from kickthefly.lab import protocol as pr

cases = {
    "smoke": dict(name="smoke", seed=1, flies=2, warmup_s=0.5, duration_s=1.0,
                  stimuli=[dict(at_s=0.3, for_s=0.5, target="loom", mode="drive", amp=0.5)],
                  recordings=[dict(name="giant_fiber", neurons="dnp01"), dict(name="looming_detectors", neurons="loom")]),
    "poke_surgery": dict(name="poke-surgery", seeds=[5, 6], warmup_s=0.5, duration_s=1.0, surgery={"type:DNp01": -1},
                         stimuli=[dict(at_s=0.2, for_s=0.4, target="loom", strength=0.8),
                                  dict(at_s=0.5, for_s=0.3, target="type:LC4", mode="drive", amp=0.5)],
                         recordings=[dict(name="giant_fiber", neurons="dnp01"), dict(name="mn9", neurons="mn9")]),
    "drug": dict(name="drug", seeds=[7], warmup_s=0.5, duration_s=0.6, drug=dict(doses=dict(picrotoxin=0.5)),
                 stimuli=[], recordings=[dict(name="dn", neurons="superclass:descending_neuron")]),
    "thermo": dict(name="thermo", seeds=[8], warmup_s=0.3, duration_s=0.8,
                   thermogenetics=dict(expression=[dict(effector="trpa1", target="type:DNp01")],
                                       temperature_c=[dict(at_s=0, c=22), dict(at_s=0.3, c=32)], kinetics="real"),
                   recordings=[dict(name="giant_fiber", neurons="dnp01")]),
    "imaging": dict(name="imaging", seeds=[9], warmup_s=0.3, duration_s=1.0,
                    imaging=dict(indicator="gcamp6f", fps=20, rois=["type:DNp01", "prefix:KC"]),
                    stimuli=[dict(at_s=0.2, for_s=0.4, target="loom", mode="drive", amp=0.5)],
                    recordings=[dict(name="giant_fiber", neurons="dnp01")]),
    "patch": dict(name="patch", seeds=[11, 12], patch=dict(neuron="type:DNp01", index=0, mode="embedded", amplitudes=[0.0, 0.05, 0.2],
                                                         duration_ms=200, repeats=2, warmup_s=0.5)),
    "rig": dict(name="rig", seeds=[3], rig=dict(name="buridan", seconds=5.0, individuality="off")),
    "assay": dict(name="assay", assay="looming", seeds=[20, 21], surgery={"type:DNp01": -1}, assay_options=dict(speeds=[3.0], approaches=1)),
}
out = {}
for name, p in cases.items():
    p = pr.check(p, name)
    folder = pr.run(p, Path(tempfile.mkdtemp()))
    summ = json.loads((folder / "summary.json").read_text())
    files = {}
    for f in sorted(folder.iterdir()):
        if f.suffix == ".csv":
            files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    meta = {f.name: json.loads(f.read_text()).get("spike_sha256") for f in folder.glob("*-metadata.json")}
    out[name] = dict(protocol=p, summary=summ, files=files, spike_sha=meta)
    print(name, "done", flush=True)
json.dump(out, open("tests/golden/protocol_runs.json", "w"), default=str)
