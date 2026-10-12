//! The sleep-deprivation assay on seeds 1000-1002 equals `lab/sleepdep.py` (golden `tests/golden/sleepdep.json`).

use std::sync::Arc;

use fly_sim::{pack, sleepdep};
use serde_json::Value;

#[test]
fn assay_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let golden: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/sleepdep.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    // THRESH["sleep"] = 2.0
    let got = sleepdep::run(&w, &l, &[1000, 1001, 1002], "off", 3, 2.0, "GOLDEN", |_, _, _| true).unwrap();
    for side in ["control", "deprived"] {
        for (i, (g, p)) in got[side].as_array().unwrap().iter().zip(golden[side].as_array().unwrap()).enumerate() {
            for k in [
                "sleep_s",
                "pressure_end_of_deprivation",
                "peak_pressure",
                "dfb_level_end_of_deprivation",
                "first_sleep_after_deprivation_s",
            ] {
                if k == "sleep_s" {
                    for ph in ["settle", "deprivation", "recovery"] {
                        assert_eq!(g[k][ph], p[k][ph], "{side}[{i}].{k}.{ph}");
                    }
                } else {
                    let (a, b) = (g[k].as_f64(), p[k].as_f64());
                    match (a, b) {
                        (Some(a), Some(b)) => assert!((a - b).abs() < 1e-6, "{side}[{i}].{k}: {a} vs {b}"),
                        (None, None) => {}
                        _ => panic!("{side}[{i}].{k}: {a:?} vs {b:?}"),
                    }
                }
            }
        }
    }
    for k in ["recovery_sleep_control", "recovery_sleep_deprived", "dfb_control", "dfb_deprived"] {
        let (a, b) = (got["mean"][k].as_f64().unwrap(), golden["mean"][k].as_f64().unwrap());
        assert!((a - b).abs() < 1e-6, "{k}: {a} vs {b}");
    }
    for c in ["S1", "S2", "S3"] {
        assert_eq!(got["criteria"][c]["passed"], golden["criteria"][c]["passed"], "{c}");
    }
}
