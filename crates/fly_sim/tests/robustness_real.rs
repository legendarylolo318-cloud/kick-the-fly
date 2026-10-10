//! Wiring statistics and the robustness analyses on the real pack equal `lab/robustness.py` (`tests/golden/make_robustness.py`).

use std::sync::Arc;

use fly_sim::robustness::{self, Ctx};
use fly_sim::{pack, validation, wiring};
use serde_json::Value;

fn same(a: &Value, b: &Value, path: &str, bad: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                match x.get(k) {
                    Some(u) => same(u, v, &format!("{path}.{k}"), bad),
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
                same(u, v, &format!("{path}[{i}]"), bad);
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            if (x - y).abs() > 1e-6 * y.abs().max(1.0) {
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
fn analyses_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/robustness.json")).unwrap())
            .unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let counts = wiring::synapse_counts(&path).unwrap();
    let c = Ctx { weights: &w, labels: &l, counts: &counts, workers: 3 };
    let mut bad = Vec::new();
    same(&wiring::threshold_stats(&w.csr, &counts, 5), &golden["threshold_stats_5"], "threshold_stats", &mut bad);
    same(&wiring::confidence_stats(&l, 0.5), &golden["confidence_stats_05"], "confidence_stats", &mut bad);
    same(&wiring::inhibition_stats(&w.csr, &l, 0.5), &golden["inhibition_stats_05"], "inhibition_stats", &mut bad);
    let fl = wiring::random_flip(&l, 0.5, 0.5, 3);
    assert_eq!(fl.len() as u64, golden["random_flip"]["n"].as_u64().unwrap());
    assert_eq!(fl.iter().map(|&x| u64::from(x)).sum::<u64>(), golden["random_flip"]["sum"].as_u64().unwrap());
    let seeds = validation::seeds();
    let inc = ["looming_escape", "sugar_feeding"];
    let sweep = robustness::threshold_sweep(&c, &[1, 5], Some(&seeds[..3]), &inc, "GOLDEN", &mut |_, _, _| true).unwrap();
    for k in ["steps", "breaks", "thresholds", "seeds"] {
        same(&sweep[k], &golden["sweep"][k], &format!("sweep.{k}"), &mut bad);
    }
    let flips = robustness::signflip_trials(&c, 0.7, 0.5, 2, Some(&seeds[..3]), &inc, "GOLDEN", &mut |_, _, _| true).unwrap();
    for k in ["control", "runs", "behaviors", "candidates"] {
        same(&flips[k], &golden["flips"][k], &format!("flips.{k}"), &mut bad);
    }
    let inh = robustness::inhibition_report(&c, 0.5, &[1000], 200, "GOLDEN");
    for k in ["before", "after", "stats", "scale", "severity"] {
        same(&inh[k], &golden["inhib"][k], &format!("inhib.{k}"), &mut bad);
    }
    assert!(bad.is_empty(), "{} differences:\n{}", bad.len(), bad.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
}
