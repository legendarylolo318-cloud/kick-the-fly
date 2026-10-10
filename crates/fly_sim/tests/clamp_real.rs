//! The neural clamp on the real pack equals `lab/clamp.py` (`tests/golden/make_clamp.py`).

use std::sync::Arc;

use fly_sim::wiring::{self, Wiring};
use fly_sim::{clamp, pack};
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
fn replays_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/clamp.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let counts = wiring::synapse_counts(&path).unwrap();
    let mut bad = Vec::new();
    let r1 = clamp::record_reference_run(&w, &l, "looming", 200, 1000);
    close(&Value::Object(r1.region_rates.clone()), &golden["ref_region_rates"], "ref_region_rates", &mut bad);
    let keys = [
        "stimulus",
        "steps",
        "clamped_neurons",
        "lesioned_neurons",
        "free_region_rates",
        "clamped_region_rates",
        "diff_region_rates",
        "readouts",
    ];
    let t = clamp::run_neural_clamp(&w, &l, &counts, &r1, &Wiring { min_synapses: 10, ..Wiring::default() }, None, 1000, "GOLDEN");
    for k in keys {
        close(&t[k], &golden["thresh"][k], &format!("thresh.{k}"), &mut bad);
    }
    let dn: Vec<u32> = (0..l.types.len() as u32).filter(|&i| l.types[i as usize] == "DNp01").collect();
    let t = clamp::run_neural_clamp(&w, &l, &counts, &r1, &Wiring::default(), Some(&dn), 1000, "GOLDEN");
    for k in keys {
        close(&t[k], &golden["lesion"][k], &format!("lesion.{k}"), &mut bad);
    }
    let r2 = clamp::record_reference_run(&w, &l, "sugar", 200, 1000);
    let t = clamp::run_neural_clamp(&w, &l, &counts, &r2, &Wiring { inhibition_scale: 0.0, ..Wiring::default() }, None, 1000, "GOLDEN");
    for k in keys {
        close(&t[k], &golden["inhib"][k], &format!("inhib.{k}"), &mut bad);
    }
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
