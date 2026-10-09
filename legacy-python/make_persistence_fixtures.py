#!/usr/bin/env python3
"""Dump real files and the settings schema of the Python game for the Rust port's persistence tests.

    python legacy-python/make_persistence_fixtures.py [--out tests/python_files] [--schema crates/ktf_data/data/schema.json]

Writes:
  * schema.json: SETTINGS, ACTIONS, PAD_ACTIONS, RESERVED_KEYS, OVERLAP_OK, tool names and limits (what the Rust Config is built from),
  * share/<name>.txt + share/<name>.json: one real code per kind (and a .ktfshare file) with the payload it decodes to,
  * share/refused.json: damaged / wrong codes with the reason text Python gives,
  * config/*.toml: config.toml files written by `Config.to_toml()` (defaults, changed values) and a few hand-made ones (schema 1,
    unknown keys, out-of-range values), each with `<name>.expect.json` = the values Python loads from it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kickthefly.core import config as C  # noqa: E402
from kickthefly.core import loadout as lo  # noqa: E402
from kickthefly.core import sharecode as sc  # noqa: E402


def schema() -> dict:
    return {
        "schema_version": C.SCHEMA_VERSION,
        "settings": [{"key": s.key, "tab": s.tab, "label": s.label, "kind": s.kind, "default": s.default, "tip": s.tip,
                      "options": list(s.options), "labels": list(s.labels), "lo": s.lo, "hi": s.hi, "step": s.step,
                      "tag": s.tag, "restart": s.restart, "only": s.only, "fmt": s.fmt} for s in C.SETTINGS],
        "actions": [list(a) for a in C.ACTIONS],
        "pad_actions": [list(a) for a in C.PAD_ACTIONS],
        "pad_axes": list(C.PAD_AXES), "pad_buttons": list(C.PAD_BUTTONS),
        "reserved_keys": sorted(C.RESERVED_KEYS),
        "overlap_ok": [sorted(p) for p in C.OVERLAP_OK],
        "first_run": C.FIRST_RUN_DEFAULTS,
        "tools": list(lo.BY_NAME), "max_saved_loadouts": lo.MAX_SAVED,
    }


def dump_cfg(cfg: C.Config) -> dict:
    return {"values": cfg.values, "keys": cfg.keys, "pad": cfg.pad, "loadout": cfg.loadout, "first_run": cfg.first_run,
            "migrated_from": cfg.migrated_from, "warnings": cfg.warnings}


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", default=str(root / "tests" / "python_files"))
    ap.add_argument("--schema", default=str(root / "crates" / "ktf_data" / "data" / "schema.json"))
    a = ap.parse_args()
    out, sch = Path(a.out), Path(a.schema)
    sch.parent.mkdir(parents=True, exist_ok=True)
    sch.write_text(json.dumps(schema(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- share codes
    share = out / "share"
    share.mkdir(parents=True, exist_ok=True)
    tools = list(lo.BY_NAME)[:6]
    payloads = {
        "surgery": {"groups": {"Silence loom detectors": -1}, "types": {"LC4": -1, "DNp01": 1}},
        "loadout": {"name": "My kit", "tools": tools},
        "lab": {"params": {"syn_gain": 5.5, "noise_std": 0.07}},
        "challenge": {"challenge": "tmaze", "seed": 42, "arena": "room", "surgery": {"types": {"LC4": -1}},
                      "params": {"bias": 0.25}},
        "protocol": {"protocol": {"name": "shared-loom", "assay": "looming", "seeds": [1, 2, 3], "surgery": {"LC4": -1}}},
        "contraption": {"name": "Domino run", "parts": [{"k": "marble", "x": -100, "y": 200}, {"k": "ramp", "x": 0, "y": 100, "a": 20},
                                                         {"k": "domino", "x": 80, "y": 60}, {"k": "fly", "x": 150, "y": 40}]},
        "unicode": None,
    }
    payloads["unicode"] = None
    del payloads["unicode"]
    for kind, p in payloads.items():
        text = sc.encode(kind, p)
        (share / f"{kind}.txt").write_text(text + "\n", encoding="ascii")
        (share / f"{kind}.json").write_text(json.dumps({"kind": kind, "payload": p}, sort_keys=True, indent=1) + "\n")
    big = {"name": "x", "tools": list(lo.BY_NAME)}
    big_p = {"protocol": {"name": "big", "notes": " ".join(f"word{i}" for i in range(900)), "assay": "looming"}}
    text = sc.encode("protocol", big_p, limit=None)
    (share / "big.ktfshare").write_text(text + "\n", encoding="ascii")
    (share / "big.json").write_text(json.dumps({"kind": "protocol", "payload": big_p}, sort_keys=True, indent=1) + "\n")
    good = sc.encode("loadout", payloads["loadout"])
    bad = {}
    for name, t in {"empty": "", "not_a_code": "hello world", "newer": good.replace("KTF1", "KTF9", 1),
                    "unknown_kind": good.replace("-LDT-", "-ZZZ-", 1), "short": "KTF1-LDT-0A",
                    "bad_char": good[:12] + "!" + good[13:], "checksum": good[:-1] + ("0" if good[-1] != "0" else "1"),
                    "kind_mismatch": good.replace("-LDT-", "-SRG-", 1)}.items():
        try:
            sc.decode(t)
            bad[name] = {"text": t, "error": None}
        except sc.ShareError as e:
            bad[name] = {"text": t, "error": str(e)}
    (share / "refused.json").write_text(json.dumps(bad, indent=1) + "\n")
    # tolerant input forms Python accepts: lowercase, spaces/newlines, O/I/L confusables
    (share / "messy.txt").write_text(payloads_text := sc.encode("lab", payloads["lab"]).lower().replace("-", "- \n") + "\n")

    # --- config.toml
    cfgdir = out / "config"
    cfgdir.mkdir(parents=True, exist_ok=True)
    cfg = C.Config(cfgdir / "defaults.toml")
    cfg.save()
    cfg2 = C.Config(cfgdir / "changed.toml")
    for k, v in {"graphics.fullscreen": True, "graphics.fps_cap": 144, "audio.master": 0.35, "brain.mode": "lab",
                 "brain.seed": 1234, "controls.fov": 95.0, "controls.invert_y": True, "access.language": "de"}.items():
        if k in C.BY_KEY:
            cfg2.set(k, v)
    cfg2.bind("big_view", "n")
    cfg2.bind("sprint", "left alt")
    cfg2.bind_pad("use", "lefttrigger")
    cfg2.custom_loadout(tools[:4])
    cfg2.save_loadout("Kit one", tools[:3])
    cfg2.save_loadout("Kit \"two\"", tools[1:5])
    cfg2.first_run["tutorial_done"] = True
    cfg2.save()
    hand = {
        "schema1.toml": 'schema_version = 1\n[graphics]\nfullscreen = true\n[brain]\nscience_popups = true\n',
        "unknown_and_range.toml": ('schema_version = 3\nfuture_thing = 1\n[graphics]\nfps_cap = 77\nresolution_scale = 9.0\nnew_key = "x"\n'
                                   '[audio]\nmaster = -4\n[keys]\nforward = "UP"\nsprint = "escape"\nfoo = "bar"\n[future]\nx = 1\n'),
        "double_bound.toml": 'schema_version = 3\n[keys]\nforward = "q"\nback = "q"\n',
        "wrong_types.toml": 'schema_version = 3\n[graphics]\nfullscreen = "yes"\nshow_fps = 3\n[brain]\nseed = 2.6\n[controls]\nfov = "wide"\n',
    }
    for n, t in hand.items():
        (cfgdir / n).write_text(t, encoding="utf-8")
    for p in sorted(cfgdir.glob("*.toml")):
        c = C.Config.load(p)
        (cfgdir / (p.stem + ".expect.json")).write_text(json.dumps(dump_cfg(c), indent=1, sort_keys=True) + "\n")
    print("wrote", sch, "and", out)


if __name__ == "__main__":
    main()
