//! The hum demo equals `lab/audio.py` (`tests/golden/make_hum.py`); the orchard and predator assays run and summarise.

use std::sync::Arc;

use fly_sim::pack;
use serde_json::{Value, json};

#[test]
fn hum_demo_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/hum.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let r = kick_the_fly::lab_extra::hum_demo_fly(
        &w,
        &l,
        1000,
        &json!({"conditions": ["silence", "pulses_200_ipi35", "steady_50"], "seconds": 0.6}),
    )
    .unwrap();
    assert_eq!(r["n_jo_a"], g["n_jo_a"]);
    for (cond, row) in g["conditions"].as_object().unwrap() {
        for (k, v) in row.as_object().unwrap() {
            for (f, x) in v.as_object().unwrap() {
                let (a, b) = (r["conditions"][cond][k][f].as_f64().unwrap(), x.as_f64().unwrap());
                if (a - b).abs() > 1e-4 * b.abs().max(1e-6) {
                    panic!("{cond}.{k}.{f}: {a} vs {b}");
                }
            }
        }
    }
}

#[test]
fn orchard_and_predator_assays_run() {
    let Some(path) = pack::find_default() else { return };
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    kick_the_fly::lab_extra::register();
    let o = kick_the_fly::lab_extra::orchard_fly(&w, &l, 1, &json!({"duration_s": 20.0}), &[]).unwrap();
    assert!(o["feeds"].as_u64().unwrap() >= 1);
    assert!(o["mn9_ratio"].as_f64().unwrap().is_finite());
    let p = kick_the_fly::lab_extra::predator_escape_fly(&w, &l, 1, &json!({"kinds": ["mantis"], "trials": 1}), &[]).unwrap();
    let s = kick_the_fly::lab_extra::predator_summarize(&[p]);
    assert_eq!(s["rows"][0]["predator"], "mantis");
}
