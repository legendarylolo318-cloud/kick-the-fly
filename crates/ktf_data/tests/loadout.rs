//! The loadout logic against what the Python module computed (`tests/python_files/loadout.json`).
use ktf_data::loadout::*;

fn case() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/python_files/loadout.json")).unwrap())
        .unwrap()
}
fn strs(v: &serde_json::Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

#[test]
fn every_preset_in_every_mode_matches_python() {
    let custom = strs(&serde_json::json!([
        "swatter",
        "laser",
        "bomb",
        "nope",
        "swatter",
        "frog",
        "mantis",
        "dragonfly",
        "fruit",
        "cva",
        "decoy",
        "alcohol"
    ]));
    let c = case();
    let builds = c["builds"].as_array().unwrap();
    assert_eq!(builds.len(), 8 * 3 * 2);
    for b in builds {
        let (mode, larva) = (b["mode"].as_str().unwrap(), b["larva"].as_bool().unwrap());
        let l = build(b["preset"].as_str().unwrap(), &custom, mode == "lab", larva, mode);
        assert_eq!(l.tools, strs(&b["tools"]), "{b}");
        assert_eq!(l.n_pages() as u64, b["pages"].as_u64().unwrap(), "{b}");
        assert_eq!(l.preset, b["name"].as_str().unwrap(), "{b}");
    }
}

#[test]
fn editing_operations_match_python() {
    let c = case();
    let ops = c["ops"].as_array().unwrap();
    let mut l = Loadout::new(&ktf_data::config::schema().loadout.tool_names, "custom", true, false);
    assert_eq!(l.tools, strs(&ops[0]["tools"]));
    for o in &ops[1..] {
        let name = o["op"].as_str().unwrap();
        let arg = &o["arg"];
        let result = match name {
            "turn_page" => serde_json::json!(l.turn_page(arg.as_i64().unwrap() as i32)),
            "unequip" => serde_json::json!(l.unequip(arg.as_str().unwrap())),
            "equip" => serde_json::json!(l.equip(arg.as_str().unwrap(), None)),
            "move" => serde_json::json!(l.move_to(arg[0].as_str().unwrap(), arg[1].as_u64().unwrap() as usize)),
            "show" => {
                l.show(arg.as_str().unwrap());
                serde_json::Value::Null
            }
            "neighbor" => serde_json::json!(l.neighbor(arg[0].as_str().unwrap(), arg[1].as_i64().unwrap() as i32)),
            other => panic!("{other}"),
        };
        assert_eq!(result, o["result"], "{name} {arg}");
        assert_eq!(l.tools, strs(&o["tools"]), "{name} {arg}");
        assert_eq!(l.page as u64, o["page"].as_u64().unwrap(), "{name} {arg}");
    }
}
