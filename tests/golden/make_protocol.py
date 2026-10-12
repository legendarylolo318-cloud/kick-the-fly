"""Golden for ktf_data::protocol: protocol.load of every bundled protocol, and check() on crafted bad protocols (the messages)."""
import json, sys
sys.path.insert(0, "legacy-python")
from pathlib import Path
from kickthefly.lab import protocol as pr

out = {"files": {}, "cases": []}
for f in sorted(Path("legacy-python/protocols").glob("*.yaml")):
    try:
        out["files"][f.name] = dict(ok=pr.load(f))
    except pr.ProtocolError as e:
        out["files"][f.name] = dict(error=str(e))
cases = [
    [],
    {"bogus": 1},
    {"seeds": ["a"]},
    {"seed": "x"},
    {"flies": 0},
    {"flies": 10**6},
    {"params": {"nope": 1}},
    {"params": 3},
    {"surgery": {"type:DNp01": 0}},
    {"nwb": "yes"},
    {"assay": "nope"},
    {"assay": "tmaze", "drug": {"doses": {"picrotoxin": 0.5}}},
    {"assay": "thermo_escape", "assay_options": {"temps": []}},
    {"assay": "thermo_escape", "assay_options": {"temps": [20, 30], "target": 3}},
    {"assay": "predator_escape", "assay_options": {"trials": 99}},
    {"assay": "hum_demo", "assay_options": {"conditions": ["x"]}},
    {"assay": "sleep_deprivation", "surgery": {"type:DNp01": -1}},
    {"assay": "sleep_deprivation", "assay_options": {"deprive_s": 1}},
    {"warmup_s": -1},
    {"duration_s": "x"},
    {"stimuli": [{"at_s": 1}]},
    {"stimuli": [{"at_s": 1, "target": "loom", "zzz": 1}]},
    {"stimuli": [{"at_s": 1, "target": "loom", "mode": "x"}]},
    {"recordings": [{"name": "x"}]},
    {"thermogenetics": {"expression": []}},
    {"thermogenetics": {"expression": [{"effector": "zzz", "target": "type:DNp01"}]}},
    {"thermogenetics": {"expression": [{"effector": "trpa1", "target": "type:DNp01"}], "temperature_c": 99}},
    {"thermogenetics": {"expression": [{"effector": "trpa1", "target": "type:DNp01"}], "kinetics": "x"}},
    {"thermogenetics": {"expression": [{"effector": "trpa1", "target": "type:DNp01"}], "time_scale": 0}},
    {"drug": {"doses": {}}},
    {"drug": {"doses": {"picrotoxin": 2}}},
    {"imaging": {"fps": 500}},
    {"imaging": {"indicator": "zzz"}},
    {"imaging": {"rois": []}},
    {"weather": {"rain": 2}},
    {"weather": {"zz": 1}},
    {"audio": {"hz": 1}},
    {"audio": {"sensitivity": 99}},
    {"predator": {"kind": "cat"}},
    {"patch": {"neuron": ""}},
    {"patch": {"neuron": "type:DNp01", "mode": "x"}},
    {"patch": {"neuron": "type:DNp01"}, "stimuli": []},
    {"patch": {"neuron": "type:DNp01", "amplitudes": [9]}},
    {"rig": {"name": "zz"}},
    {"rig": {"name": "buridan", "mode": "x"}},
    {"rig": {"name": "buridan", "seconds": 1}},
    {"rig": {"name": "buridan"}, "params": {}},
    {"screen": {"kind": "zz"}},
    {"screen": {"kind": "knockout", "behaviors": ["mb_conditioning"]}},
    {"screen": {"kind": "activation", "top": 3}},
    {"screen": {"kind": "activation", "controls": 0}},
    {"classroom": True, "drug": {"doses": {"picrotoxin": 0.5}}},
    {"name": "x", "classroom": True, "steps": []},
    {"seed": 5, "flies": 3, "duration_s": 1},
]
for c in cases:
    try:
        out["cases"].append(dict(input=c, ok=pr.check(c, "case")))
    except pr.ProtocolError as e:
        out["cases"].append(dict(input=c, error=str(e)))
json.dump(out, open("tests/golden/protocol.json", "w"), default=str)
print(sum("ok" in v for v in out["files"].values()), "files ok;", sum("error" in c for c in out["cases"]), "cases rejected of", len(cases))
