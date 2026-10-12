//! The failure autopsy of three failing tests on the real pack equals `lab/failure_autopsy.py` (`tests/golden/make_autopsy.py`).

use std::sync::Arc;

use fly_sim::npy::Npz;
use fly_sim::{failure_autopsy, pack};
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
            if (x - y).abs() > 1e-9 * y.abs().max(1e-3) {
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
fn autopsies_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let text = std::fs::read_to_string(format!("{root}/autopsy.json")).unwrap().replace("NaN", "null");
    let golden: Value = serde_json::from_str(&text).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let mut npz = Npz::read(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).unwrap();
    let inv = npz.take("inv").unwrap().into_f32().unwrap();
    let mut bad = Vec::new();
    for (id, want) in golden.as_object().unwrap() {
        let got = failure_autopsy::autopsy(&w, &l, &inv, id, &failure_autopsy::AUTOPSY_SEEDS, None, "GOLDEN", &mut |_, _, _| {}).unwrap();
        for k in ["test", "weights_digest", "sections", "method", "ratio_min"] {
            close(&got[k], &want[k], &format!("{id}.{k}"), &mut bad);
        }
        let md = std::fs::read_to_string(format!("{root}/autopsy_{id}.md")).unwrap();
        assert_eq!(failure_autopsy::to_markdown(&got), md.trim_end_matches('\n').to_string() + "\n", "{id} markdown");
    }
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
