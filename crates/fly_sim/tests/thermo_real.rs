//! The thermogenetic escape assay on the real pack equals `lab/thermogenetics.py` (`tests/golden/make_thermo.py`).

use std::sync::Arc;

use fly_sim::{assay_jobs, pack, thermogenetics};
use serde_json::{Value, json};

fn close(a: &Value, b: &Value, path: &str, bad: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                match x.get(k) {
                    Some(u) => close(u, v, &format!("{path}.{k}"), bad),
                    None => bad.push(format!("{path}.{k} missing")),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                bad.push(format!("{path}: len {} vs {}", x.len(), y.len()));
                return;
            }
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                close(u, v, &format!("{path}[{i}]"), bad);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() > 1e-5 * y.abs().max(1.0) {
                bad.push(format!("{path}: {x} vs {y}"));
            }
        }
        (x, y) => {
            if x != y && !(x.is_null() && y.is_null()) {
                bad.push(format!("{path}: {x} vs {y}"));
            }
        }
    }
}

#[test]
fn assay_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/thermo.json")).unwrap().replace("NaN", "null");
    let golden: Value = serde_json::from_str(&text).unwrap();
    for (e, vals) in golden["fractions"].as_object().unwrap() {
        let ef = thermogenetics::effector(e).unwrap();
        for (t, want) in [20.0, 25.0, 26.5, 28.0, 29.0, 30.0, 35.0].iter().zip(vals.as_array().unwrap()) {
            assert_eq!(ef.target_fraction(*t), want.as_f64().unwrap(), "{e} at {t}");
        }
    }
    for (t, want) in [-2.0, -1.0, 0.0, 0.3, 1.0].iter().zip(golden["arena"].as_array().unwrap()) {
        assert!((thermogenetics::arena_temperature(*t) - want.as_f64().unwrap()).abs() < 1e-12);
    }
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let opts = json!({"temps": [22.0, 28.0, 32.0]});
    let flies: Vec<Value> =
        [1000u64, 1001].iter().map(|&s| assay_jobs::assay_task(&w, &l, "thermo_escape", s, &opts, &[]).unwrap()).collect();
    let mut bad = Vec::new();
    close(&Value::Array(flies.clone()), &golden["flies"], "flies", &mut bad);
    close(&assay_jobs::summarize("thermo_escape", &flies), &golden["summary"], "summary", &mut bad);
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
}
