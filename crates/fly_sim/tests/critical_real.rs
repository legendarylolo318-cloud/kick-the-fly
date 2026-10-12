//! The critical path finder on the real pack equals `lab/criticalpath.py` (`tests/golden/make_critical.py`).

use std::sync::Arc;

use fly_sim::{criticalpath, pack};
use serde_json::Value;

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
fn runs_equal_python() {
    let Some(path) = pack::find_default() else { return };
    // Python wrote NaN for the types it was given by name (no influence); as JSON null here
    let text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/critical.json")).unwrap().replace("NaN", "null");
    let golden: Value = serde_json::from_str(&text).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let mut bad = Vec::new();
    let sl = Value::Array(criticalpath::shortlist(&w, &l, "looming_escape", 12));
    close(&sl, &golden["shortlist"], "shortlist", &mut bad);
    let sl = Value::Array(criticalpath::shortlist(&w, &l, "sugar_feeding", 8));
    close(&sl, &golden["sugar_shortlist"], "sugar_shortlist", &mut bad);
    let beh =
        criticalpath::run(&w, &l, "looming_escape", Some(&[1000, 1001, 1002]), 4, 3, None, None, "GOLDEN", &mut |_, _, _| true).unwrap();
    for k in ["control", "candidates", "ranked", "metric", "seeds"] {
        close(&beh[k], &golden["behavior"][k], &format!("behavior.{k}"), &mut bad);
    }
    let types = ["MN9".to_string()];
    let asy = criticalpath::run(&w, &l, "sugar", Some(&[1000, 1001]), 25, 2, None, Some(&types), "GOLDEN", &mut |_, _, _| true).unwrap();
    for k in ["control", "ranked", "metric"] {
        close(&asy[k], &golden["assay"][k], &format!("assay.{k}"), &mut bad);
    }
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
