//! `labjobs.run_sync` (the Lab's assays page) on the real pack against Python's (`tests/golden/make_assay_jobs.py`).

use std::sync::Arc;

use fly_sim::assay_jobs;
use fly_sim::pack::{self, Pack};
use serde_json::{Value, json};

fn close(path: &str, a: &Value, b: &Value, bad: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                match x.get(k) {
                    Some(m) => close(&format!("{path}.{k}"), m, v, bad),
                    None if v.is_null() => {}
                    None => bad.push(format!("{path}.{k} missing")),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                bad.push(format!("{path}: {} entries vs {}", x.len(), y.len()));
            }
            for (i, (m, v)) in x.iter().zip(y).enumerate() {
                close(&format!("{path}[{i}]"), m, v, bad);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() > 1e-5 * y.abs().max(1.0) {
                bad.push(format!("{path}: {x} vs {y}"));
            }
        }
        (Value::Null, Value::Null) => {}
        (x, y) if x == y => {}
        (x, y) => bad.push(format!("{path}: {x} vs {y}")),
    }
}

#[test]
fn assay_jobs_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let g: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/assay_jobs.json")).unwrap())
            .unwrap();
    let pk = Pack::load(path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let seeds = [2000u64, 2001];
    #[allow(clippy::type_complexity)]
    let cases: [(&str, &str, Value, Vec<(String, i8)>); 4] = [
        ("tmaze", "tmaze", json!({}), vec![]),
        ("looming", "looming", json!({"speeds": [1.4, 5.5], "approaches": 2}), vec![]),
        ("sugar", "sugar", json!({"doses": [0.0, 0.2, 1.0], "repeats": 2}), vec![]),
        ("looming_surgery", "looming", json!({"speeds": [1.4, 5.5], "approaches": 2}), vec![("type:DNp01".to_string(), -1)]),
    ];
    let mut bad = Vec::new();
    for (name, kind, options, surgery) in cases {
        let got = assay_jobs::run_sync(&w, &l, kind, &seeds, &options, &surgery, 2, |_, _| true).unwrap();
        let want = &g[name];
        for part in ["treated", "control", "comparison"] {
            if want.get(part).is_some() {
                close(&format!("{name}.{part}"), &got[part], &want[part], &mut bad);
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
