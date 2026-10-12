//! The network-science analysis of the real pack equals `lab/netsci.py` (golden `tests/golden/netsci.json`, `make_netsci.py`).

use std::sync::atomic::AtomicBool;

use fly_sim::{netsci, pack};
use serde_json::Value;

fn close(a: &Value, b: &Value, path: &str, tol: f64, bad: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                match x.get(k) {
                    Some(u) => close(u, v, &format!("{path}.{k}"), tol, bad),
                    None => bad.push(format!("{path}.{k} missing")),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                bad.push(format!("{path}: length {} vs {}", x.len(), y.len()));
                return;
            }
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                close(u, v, &format!("{path}[{i}]"), tol, bad);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() > tol * y.abs().max(1.0) {
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
fn analysis_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/netsci.json")).unwrap()).unwrap();
    let g = netsci::load_edges(&path).unwrap();
    let sha = fly_sim::cards::pack_sha256(&path).unwrap();
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let cancel = AtomicBool::new(false);
    let got = netsci::analyse(&g, "adult", &name, &sha, "GOLDEN", (3, 300_000, 0), &cancel, &mut |_, _| {}).unwrap();
    let mut bad = Vec::new();
    for key in ["degrees", "reciprocity", "motifs", "motif_meta", "rich_club", "communities", "regions", "pack_sha256"] {
        close(&got[key], &golden[key], key, 1e-9, &mut bad);
    }
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
