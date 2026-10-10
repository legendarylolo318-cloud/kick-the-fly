//! Two noise values and a synapse threshold on two behaviors, 3 seeds, equal `lab/sensitivity.py` (`tests/golden/make_sensitivity.py`).

use std::sync::Arc;

use fly_sim::{pack, sensitivity, validation, wiring};
use serde_json::Value;

#[test]
fn cells_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/sensitivity.json")).unwrap())
            .unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let counts = wiring::synapse_counts(&path).unwrap();
    let seeds = validation::seeds();
    let got = sensitivity::run(
        &w,
        &l,
        Some(&counts),
        Some(&["noise_std", "min_synapses"]),
        Some(&["looming_escape", "sugar_feeding"]),
        Some(&seeds[..3]),
        Some(&[("noise_std", vec![0.0375, 0.075]), ("min_synapses", vec![5.0])]),
        3,
        None,
        false,
        "GOLDEN",
        |_, _, _| true,
    )
    .unwrap();
    assert_eq!(got["config_hash"], golden["config_hash"]);
    let (gc, pc) = (got["cells"].as_array().unwrap(), golden["cells"].as_array().unwrap());
    assert_eq!(gc.len(), pc.len());
    for (g, p) in gc.iter().zip(pc) {
        assert_eq!(g["key"], p["key"]);
        for (b, pr) in p["behaviors"].as_object().unwrap() {
            let gr = &g["behaviors"][b];
            assert_eq!(gr["passed"], pr["passed"], "{} {b}", g["key"]);
            assert_eq!(gr["metric"], pr["metric"]);
            for k in ["effect", "control", "effect_size_dz", "p_value"] {
                let (a, e) = (gr[k].as_f64().unwrap(), pr[k].as_f64().unwrap());
                assert!((a - e).abs() < 1e-6 * e.abs().max(1.0), "{} {b} {k}: {a} vs {e}", g["key"]);
            }
        }
    }
}
