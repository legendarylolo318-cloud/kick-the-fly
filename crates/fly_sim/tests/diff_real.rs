//! Hemifield rows and the connectome diff on the real pack equal `lab/lesions.py` and `lab/diffmode.py` (`tests/golden/make_diff.py`).

use std::sync::Arc;

use fly_sim::diffmode::{self, Config};
use fly_sim::wiring::{self, Wiring};
use fly_sim::{lesions, pack};
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
            if (x - y).abs() > 1e-9 * y.abs().max(1.0) {
                bad.push(format!("{path}: {x} vs {y}"));
            }
        }
        (x, y) => {
            if x != y {
                bad.push(format!("{path}: {x} vs {y}"));
            }
        }
    }
}

#[test]
fn diff_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/diff.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let counts = wiring::synapse_counts(&path).unwrap();
    let vis = lesions::hemifield_visual_rows(&l, 'L');
    let hemi = lesions::hemisphere_rows(&l, 'L');
    assert_eq!(vis.len() as u64, golden["vis_L"]["n"].as_u64().unwrap());
    assert_eq!(vis.iter().map(|&x| u64::from(x)).sum::<u64>(), golden["vis_L"]["sum"].as_u64().unwrap());
    assert_eq!(hemi.len() as u64, golden["hemi_L"]["n"].as_u64().unwrap());
    assert_eq!(hemi.iter().map(|&x| u64::from(x)).sum::<u64>(), golden["hemi_L"]["sum"].as_u64().unwrap());
    assert_eq!(lesions::hemifield_visual_rows(&l, 'R').len() as u64, golden["vis_R"]["n"].as_u64().unwrap());
    let a = Config { wiring: Wiring::default(), lesion_rows: None, label: "Fly A (Unperturbed)".into() };
    let b = Config { wiring: Wiring::default(), lesion_rows: Some(vis), label: "Fly B (hemi_vis)".into() };
    let mut bad = Vec::new();
    let res = diffmode::run_connectome_diff(&w, &l, &counts, &a, &b, "looming", 200, 1000, "GOLDEN");
    for k in ["stimulus", "steps", "diverge_step", "divergence_timeline", "rows", "top_divergence", "readouts", "label_b"] {
        close(&res[k], &golden["looming"][k], &format!("looming.{k}"), &mut bad);
    }
    let b = Config { wiring: Wiring { min_synapses: 10, ..Wiring::default() }, lesion_rows: None, label: "Fly B (thresh_10)".into() };
    let res = diffmode::run_connectome_diff(&w, &l, &counts, &a, &b, "sugar", 200, 1000, "GOLDEN");
    for k in ["diverge_step", "divergence_timeline", "rows", "readouts"] {
        close(&res[k], &golden["sugar"][k], &format!("sugar.{k}"), &mut bad);
    }
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
