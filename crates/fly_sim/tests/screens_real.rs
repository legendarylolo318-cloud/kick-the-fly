//! The activation and knockout screens equal `lab/screens.py` (`tests/golden/make_screens.py`) on the real pack: records, rows, calls, CSVs.

use std::sync::Arc;

use fly_sim::pack;
use fly_sim::screens::{self as sc, ActivityOptions, KnockoutOptions};
use serde_json::Value;

fn close(a: &Value, b: &Value, rel: f64) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => (x - y).abs() <= rel * y.abs().max(1e-9) || (x - y).abs() < 1e-9,
        _ => a.is_null() && (b.is_null() || b.as_str() == Some("nan")) || a == b,
    }
}

#[test]
fn screens_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/screens.json")).unwrap().replace("NaN", "null")).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let tmp = std::env::temp_dir().join(format!("ktf-screens-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let seeds = vec![4000, 4001];
    let types: Vec<String> = ["DNp01", "MDN", "DNp09", "DNg62", "MN9", "LPLC2"].iter().map(|s| s.to_string()).collect();
    let o = ActivityOptions { seeds: seeds.clone(), types: Some(types), controls: 2, workers: 2, ..Default::default() };
    let act = sc::run_activation(&w, &l, &tmp.join("act"), &o, "2026-01-01 00:00:00", |_, _, _| true).unwrap();
    let want = &g["act"];
    assert_eq!(act["n_types"], want["n_types"]);
    assert_eq!(act["n_called"], want["n_called"]);
    assert_eq!(act["n_dn_types"], want["n_dn_types"]);
    for (a, b) in act["rows"].as_array().unwrap().iter().zip(want["rows"].as_array().unwrap()) {
        for (k, v) in b.as_object().unwrap() {
            if k == "_dn" {
                continue;
            }
            assert!(close(&a[k], v, 1e-6), "{} {k}: {} vs {v}", b["type"], a[k]);
        }
    }
    // the knockout screen, singles and opened batches
    let o = KnockoutOptions {
        behaviors: vec!["looming_escape".into(), "sugar_feeding".into()],
        seeds,
        top: 4,
        batch_size: 2,
        controls: 1,
        workers: 2,
        ..Default::default()
    };
    let ko = sc::run_knockout(&w, &l, &tmp.join("ko"), &o, "2026-01-01 00:00:00", |_, _, _| true).unwrap();
    let want = &g["ko"];
    assert_eq!(ko["rows"].as_array().unwrap().len(), want["rows"].as_array().unwrap().len());
    for (a, b) in ko["rows"].as_array().unwrap().iter().zip(want["rows"].as_array().unwrap()) {
        for (k, v) in b.as_object().unwrap() {
            assert!(close(&a[k], v, 1e-6), "{} {k}: {} vs {v}", b["candidate"], a[k]);
        }
    }
    for b in ["looming_escape", "sugar_feeding"] {
        assert_eq!(ko["summary"][b]["tested"], want["summary"][b]["tested"]);
        assert_eq!(ko["summary"][b]["significant"], want["summary"][b]["significant"]);
    }
    // the CSVs
    for (ours, theirs) in [
        ("act/activation_screen.csv", "act/activation_screen.csv"),
        ("ko/knockout_screen.csv", "ko/knockout_screen.csv"),
        ("act/activation_screen_responses.csv", "act/activation_screen_responses.csv"),
    ] {
        let a = std::fs::read_to_string(tmp.join(ours)).unwrap();
        let b = std::fs::read_to_string(format!("{root}/screens_files/{theirs}")).unwrap();
        assert_eq!(a.lines().next(), b.lines().next(), "{ours} header");
        assert_eq!(a.lines().count(), b.lines().count(), "{ours} lines");
    }
}
