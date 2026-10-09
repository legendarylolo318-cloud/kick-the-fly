//! The Rust readers against real files written by the Python game (`legacy-python/make_persistence_fixtures.py` →
//! `tests/python_files/`). Regenerate with that script if the Python formats ever change.

use std::path::PathBuf;

use ktf_data::config::Config;
use ktf_data::sharecode::{self, Context};
use serde_json::Value;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/python_files")
}
fn read(p: &str) -> String {
    std::fs::read_to_string(dir().join(p)).unwrap_or_else(|e| panic!("{p}: {e}"))
}
fn json(p: &str) -> Value {
    serde_json::from_str(&read(p)).unwrap()
}

#[test]
fn every_python_share_code_decodes_to_its_payload() {
    for kind in [
        "surgery",
        "loadout",
        "lab",
        "challenge",
        "protocol",
        "contraption",
    ] {
        let code = sharecode::decode(&read(&format!("share/{kind}.txt")))
            .unwrap_or_else(|e| panic!("{kind}: {e}"));
        let want = json(&format!("share/{kind}.json"));
        assert_eq!(code.kind, kind);
        assert_eq!(code.version, 1);
        assert_eq!(code.payload, want["payload"], "{kind}");
        assert_eq!(
            sharecode::check_payload(kind, &code.payload),
            None,
            "{kind}"
        );
    }
}

#[test]
fn a_ktfshare_file_of_any_length_reads_through_read_any() {
    let want = json("share/big.json");
    let path = dir().join("share/big.ktfshare");
    let code = sharecode::read_any(path.to_str().unwrap()).unwrap();
    assert_eq!(code.payload, want["payload"]);
    // and the text is longer than a chat message, which is why it was a file
    assert!(read("share/big.ktfshare").len() > sharecode::MAX_CODE_CHARS);
}

#[test]
fn messy_input_is_accepted_like_python_does() {
    // lower case, line breaks and spaces where the dashes were
    let code = sharecode::decode(&read("share/messy.txt")).unwrap();
    assert_eq!(code.kind, "lab");
    assert_eq!(code.payload, json("share/lab.json")["payload"]);
}

#[test]
fn refusals_carry_pythons_reasons() {
    let refused = json("share/refused.json");
    for (name, case) in refused.as_object().unwrap() {
        let text = case["text"].as_str().unwrap();
        match (sharecode::decode(text), case["error"].as_str()) {
            (Err(e), Some(want)) => assert_eq!(e.to_string(), want, "{name}"),
            (Ok(_), None) => {}
            (got, want) => panic!("{name}: got {got:?}, python said {want:?}"),
        }
    }
}

#[test]
fn rust_codes_round_trip_and_python_format_is_kept() {
    for kind in [
        "surgery",
        "loadout",
        "lab",
        "challenge",
        "protocol",
        "contraption",
    ] {
        let payload = json(&format!("share/{kind}.json"))["payload"].clone();
        let text = sharecode::encode(kind, &payload, Some(sharecode::MAX_CODE_CHARS)).unwrap();
        assert!(text.starts_with(&format!(
            "KTF1-{}-",
            match kind {
                "surgery" => "SRG",
                "loadout" => "LDT",
                "lab" => "LAB",
                "challenge" => "CHL",
                "protocol" => "PRT",
                _ => "CTN",
            }
        )));
        assert_eq!(sharecode::decode(&text).unwrap().payload, payload, "{kind}");
        // groups of five between dashes, the last may be shorter
        assert!(text.split('-').skip(2).all(|g| g.len() <= 5), "{text}");
    }
}

#[test]
fn non_ascii_payloads_are_escaped_like_python() {
    let p = serde_json::json!({"name": "Kät 🪰", "tools": ["hand"]});
    let b = '\\';
    let want = format!("{{\"name\":\"K{b}u00e4t {b}ud83e{b}udeb0\",\"tools\":[\"hand\"]}}");
    assert_eq!(String::from_utf8(sharecode::canonical(&p)).unwrap(), want);
}

#[test]
fn damaged_codes_never_decode() {
    let good = read("share/loadout.txt");
    let good = good.trim();
    for i in 8..good.len() {
        let c = good.as_bytes()[i];
        if c == b'-' {
            continue;
        }
        let mut s = good.as_bytes().to_vec();
        s[i] = if c == b'0' { b'1' } else { b'0' };
        assert!(
            sharecode::decode(std::str::from_utf8(&s).unwrap()).is_err(),
            "flip at {i} was accepted"
        );
    }
}

#[test]
fn validate_and_preview() {
    let code = sharecode::decode(&read("share/loadout.txt")).unwrap();
    let ctx = Context {
        tools: Some(vec!["hand".into()]),
        ..Default::default()
    };
    assert!(
        sharecode::validate(&code, &ctx)
            .unwrap()
            .starts_with("this loadout uses tools this version doesn't have")
    );
    let ctx = Context {
        tools: Some(ktf_data::config::schema().tools.clone()),
        max_saved_loadouts: 5,
        ..Default::default()
    };
    assert_eq!(sharecode::validate(&code, &ctx), None);
    let pv = sharecode::preview(&code, &ctx);
    assert_eq!(pv.title, "Tool loadout (code format 1)");
    assert_eq!(pv.lines.last().unwrap(), "Saves it as 'My kit'");

    let lab = sharecode::decode(&read("share/lab.txt")).unwrap();
    let ctx = Context {
        lab_params: Some(
            [
                ("syn_gain".to_string(), (1.0, 40.0)),
                ("noise_std".to_string(), (0.0, 0.05)),
            ]
            .into(),
        ),
        ..Default::default()
    };
    assert_eq!(
        sharecode::validate(&lab, &ctx).unwrap(),
        "Lab parameter 'noise_std' = 0.07 is outside 0 to 0.05"
    );
}

// --- config.toml ----------------------------------------------------------------------------------------------------------------

fn pairs(v: &Value) -> Vec<(String, String)> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect()
}
fn same_number_or_eq(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if a.is_number() && b.is_number() => x == y,
        _ => a == b,
    }
}

fn check_config(name: &str) {
    let want = json(&format!("config/{name}.expect.json"));
    let cfg = Config::from_toml(&read(&format!("config/{name}.toml"))).unwrap();
    for (k, v) in want["values"].as_object().unwrap() {
        assert!(
            same_number_or_eq(&cfg.values[k], v),
            "{name}: {k}: rust {:?}, python {v:?}",
            cfg.values[k]
        );
    }
    assert_eq!(
        cfg.values.len(),
        want["values"].as_object().unwrap().len(),
        "{name}"
    );
    let mut got = cfg.keys.clone();
    let mut exp = pairs(&want["keys"]);
    got.sort();
    exp.sort();
    assert_eq!(got, exp, "{name}: keys");
    let mut got = cfg.pad.clone();
    let mut exp = pairs(&want["pad"]);
    got.sort();
    exp.sort();
    assert_eq!(got, exp, "{name}: pad");
    let custom: Vec<String> = want["loadout"]["custom"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().into())
        .collect();
    assert_eq!(cfg.custom_loadout, custom, "{name}");
    let saved = want["loadout"]["saved"].as_array().unwrap();
    assert_eq!(cfg.saved_loadouts.len(), saved.len(), "{name}");
    for (a, b) in cfg.saved_loadouts.iter().zip(saved) {
        assert_eq!(a.name, b["name"].as_str().unwrap());
        let tools: Vec<&str> = b["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert_eq!(
            a.tools.iter().map(String::as_str).collect::<Vec<_>>(),
            tools
        );
    }
    for (k, v) in want["first_run"].as_object().unwrap() {
        assert_eq!(
            cfg.first_run[k],
            v.as_bool().unwrap(),
            "{name}: first_run.{k}"
        );
    }
    assert_eq!(cfg.migrated_from, want["migrated_from"].as_i64(), "{name}");
    assert_eq!(
        cfg.warnings.len(),
        want["warnings"].as_array().unwrap().len(),
        "{name}: {:?} vs {:?}",
        cfg.warnings,
        want["warnings"]
    );
}

#[test]
fn python_written_configs_load_to_the_same_values() {
    for name in [
        "defaults",
        "changed",
        "schema1",
        "unknown_and_range",
        "double_bound",
        "wrong_types",
    ] {
        check_config(name);
    }
}

#[test]
fn rewriting_a_python_config_gives_the_same_file() {
    for name in ["defaults", "changed"] {
        let text = read(&format!("config/{name}.toml"));
        let cfg = Config::from_toml(&text).unwrap();
        let ours = cfg.to_toml("X");
        // the first line carries the program version, which differs
        assert_eq!(
            ours.lines().skip(1).collect::<Vec<_>>(),
            text.lines().skip(1).collect::<Vec<_>>(),
            "{name}"
        );
    }
}

#[test]
fn defaults_match_python_defaults_and_have_75_settings() {
    let cfg = Config::new(None);
    let want = json("config/defaults.expect.json");
    assert_eq!(cfg.values.len(), 75);
    for (k, v) in want["values"].as_object().unwrap() {
        assert!(same_number_or_eq(&cfg.values[k], v), "{k}");
    }
}

#[test]
fn a_corrupt_file_is_kept_as_bad_and_defaults_load() {
    let tmp = std::env::temp_dir().join(format!("ktf-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let p = tmp.join("config.toml");
    std::fs::write(&p, "this is [not toml").unwrap();
    let cfg = Config::load(&p);
    assert_eq!(cfg.warnings.len(), 2);
    assert!(tmp.join("config.toml.bad").exists() && !p.exists());
    assert_eq!(cfg.values["graphics.fps_cap"], 60);
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn save_then_load_round_trips() {
    let tmp = std::env::temp_dir().join(format!("ktf-cfg2-{}", std::process::id()));
    let p = tmp.join("config.toml");
    let mut cfg = Config::new(Some(p.clone()));
    cfg.set("audio.master", &serde_json::json!(0.4)).unwrap();
    cfg.set("graphics.fps_cap", &serde_json::json!(144))
        .unwrap();
    assert!(cfg.set("graphics.fps_cap", &serde_json::json!(77)).is_err());
    assert!(cfg.bind("big_view", "n").0);
    assert!(!cfg.bind("big_view", "escape").0);
    assert!(cfg.save("t"));
    let back = Config::load(&p);
    assert_eq!(back.values["audio.master"], 0.4);
    assert_eq!(back.values["graphics.fps_cap"], 144);
    assert_eq!(back.key_of("big_view"), "n");
    std::fs::remove_dir_all(&tmp).ok();
}
