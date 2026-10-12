//! The psychometric sweeps on the real pack, and the CSV/SVG/PDF writers, equal `lab/psychometrics.py` (`tests/golden/make_psych.py`).

use std::sync::Arc;

use fly_sim::cards::LIF_DEFAULTS;
use fly_sim::{pack, psychometrics};
use serde_json::Value;

#[test]
fn sweeps_and_files_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let text = std::fs::read_to_string(format!("{root}/psych.json")).unwrap().replace("NaN", "null");
    let golden: Value = serde_json::from_str(&text).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let tmp = std::env::temp_dir().join(format!("ktf-psych-{}", std::process::id()));
    for (target, xs) in [("looming", vec![1.0, 5.5]), ("sugar", vec![0.0, 0.4]), ("tmaze", vec![1.0, 2.0])] {
        let want = &golden[target];
        let version = want["app_version"].as_str().unwrap();
        let got = psychometrics::run_sweep(&w, &l, target, Some(&xs), 2, 1000, LIF_DEFAULTS, 2, version, "GOLDEN", |_, _| true).unwrap();
        for (g, p) in got["points"].as_array().unwrap().iter().zip(want["points"].as_array().unwrap()) {
            for part in ["primary", "secondary"] {
                for k in ["mean", "sem", "ci95", "n"] {
                    let (a, b) = (g[part][k].as_f64(), p[part][k].as_f64());
                    match (a, b) {
                        (Some(a), Some(b)) => assert!((a - b).abs() < 1e-9, "{target} {part}.{k}: {a} vs {b}"),
                        (None, None) => {}
                        _ => panic!("{target} {part}.{k}: {a:?} vs {b:?}"),
                    }
                }
            }
        }
        // the files, from Python's own result so only the writers are under test
        psychometrics::save_all_formats(want, &format!("psych_{target}"), &tmp).unwrap();
        for ext in ["csv", "svg", "pdf"] {
            let ours = std::fs::read(tmp.join(format!("psych_{target}.{ext}"))).unwrap();
            let theirs = std::fs::read(format!("{root}/psych_files/psych_{target}.{ext}")).unwrap();
            // Python's csv module writes \r\n, text files with \n; compare as text
            assert_eq!(String::from_utf8_lossy(&ours), String::from_utf8_lossy(&theirs), "{target}.{ext}");
        }
    }
    std::fs::remove_dir_all(tmp).ok();
}
