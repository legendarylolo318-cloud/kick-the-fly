//! Pharmacology on the real pack equals `lab/pharmacology.py` (`tests/golden/make_pharm.py`).

use std::sync::Arc;

use fly_sim::pack;
use fly_sim::pharmacology as ph;
use fly_sim::robustness::Ctx;
use fly_sim::wiring::{self, Wiring};
use serde_json::Value;

#[test]
fn drugs_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/pharm.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let syn = wiring::synapse_counts(&path).unwrap();
    for (key, want) in golden.as_object().unwrap() {
        if !key.contains(':') {
            continue;
        }
        let p: Vec<&str> = key.split(':').collect();
        let (dose, low, cut): (f64, bool, f64) = (p[1].parse().unwrap(), p[2] == "True", p[3].parse().unwrap());
        let got = ph::affected(&l, &w.csr, &syn, p[0], dose, low, cut).unwrap();
        assert_eq!(&got, want, "{key}");
    }
    let wr = ph::wiring_for(&[("picrotoxin", 0.5), ("glucl", 0.5), ("gabaa_agonist", 1.0)], false, 0.8, &Wiring::default()).unwrap();
    let want = &golden["wiring"];
    assert_eq!(wr.min_synapses, want["min_synapses"].as_i64().unwrap());
    assert_eq!(wr.nt_min_conf, want["nt_min_conf"].as_f64().unwrap());
    for ((nt, f), pw) in wr.nt_scales.iter().zip(want["nt_scales"].as_array().unwrap()) {
        assert_eq!(nt, pw[0].as_str().unwrap());
        assert_eq!(*f, pw[1].as_f64().unwrap());
    }
    let c = Ctx { weights: &w, labels: &l, counts: &syn, workers: 1 };
    let dr = ph::dose_response(&c, "picrotoxin", &[0.0, 0.5, 1.0], true, 0.7, &[1000], 100).unwrap();
    for (g, p) in dr["rows"].as_array().unwrap().iter().zip(golden["dose_response"]["rows"].as_array().unwrap()) {
        for k in ["scale", "mean_hz", "median_hz", "p95_hz"] {
            let (a, b) = (g[k].as_f64().unwrap(), p[k].as_f64().unwrap());
            assert!((a - b).abs() < 1e-9, "{k}: {a} vs {b}");
        }
    }
}
