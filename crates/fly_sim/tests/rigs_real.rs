//! The behavior rigs equal `game/rigs.py` + `lab/rigassay.py` (`tests/golden/make_rigs.py`): per-fly measurements and scene runs on the real
//! pack, the pre-registered analysis and the assay summary.

use std::sync::Arc;

use fly_sim::pack;
use fly_sim::rigs::{self, Ctx};
use serde_json::Value;

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-9) || (a - b).abs() < 1e-9
}

fn compare(a: &Value, b: &Value, path: &str, rel: f64) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                compare(&x[k], v, &format!("{path}.{k}"), rel);
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path} length");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                compare(p, q, &format!("{path}[{i}]"), rel);
            }
        }
        _ => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => {
                assert!(close(x, y, rel), "{path}: {x} vs {y}");
            }
            _ => assert!(a == b || (a.is_null() && b.is_null()), "{path}: {a} vs {b}"),
        },
    }
}

#[test]
fn analysis_and_summary_equal_python() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/rigs.json")).unwrap().replace("NaN", "null")).unwrap();
    for rig in rigs::RIGS {
        let flies: Vec<Value> = g["analysis"][rig]["flies"].as_array().unwrap().clone();
        let an = rigs::analyze(rig, &flies).unwrap();
        compare(&an, &g["analysis"][rig]["analysis"], rig, 1e-9);
    }
    assert_eq!(rigs::summary(&g["assay"]), g["summary_text"].as_str().unwrap());
}

#[test]
fn rigs_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/rigs.json")).unwrap().replace("NaN", "null")).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let c = Ctx { weights: &w, labels: &l, cancel: None };
    compare(&rigs::tethered_fly(&c, 1000, "off").unwrap(), &g["tethered"], "tethered", 1e-6);
    compare(&rigs::ball_fly(&c, 1000, "off").unwrap(), &g["ball"], "ball", 1e-6);
    compare(&rigs::buridan_fly(&c, 1000, "off", 20.0).unwrap(), &g["buridan"], "buridan", 1e-6);
    compare(&rigs::fourfield_fly(&c, 1000, "off", 20.0).unwrap(), &g["fourfield"], "fourfield", 1e-6);
    for (name, s) in g["scenes"].as_object().unwrap() {
        let r = rigs::scene_run(&c, s["rig"].as_str().unwrap(), s["seed"].as_u64().unwrap(), "off", &s["opt"]).unwrap();
        compare(&r["summary"], &s["summary"], &format!("{name}.summary"), 1e-6);
        assert_eq!(r["seconds"], s["seconds"], "{name} seconds");
        let tr: Vec<Vec<f64>> =
            r["trace"].as_array().unwrap().iter().map(|x| x.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect();
        assert_eq!(tr.len() as u64, s["n"].as_u64().unwrap(), "{name} rows");
        let mut sums = [0.0; 12];
        for r in &tr {
            for (a, b) in sums.iter_mut().zip(r) {
                *a += b;
            }
        }
        for (a, b) in sums.iter().zip(s["trace_sum"].as_array().unwrap()) {
            assert!(close(*a, b.as_f64().unwrap(), 1e-5), "{name} trace sum {a} vs {b}");
        }
        compare(&r["params"], &s["params"], &format!("{name}.params"), 1e-12);
    }
}
