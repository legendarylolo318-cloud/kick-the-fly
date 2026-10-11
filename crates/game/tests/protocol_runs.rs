//! Protocol runs equal `lab/protocol.py:run` (`tests/golden/make_protocol_runs.py`) on the real pack, individuality off: summaries, the
//! recorder's CSVs byte for byte and every fly's spike fingerprint.

use std::sync::Arc;

use fly_sim::pack;
use kick_the_fly::protocol_run::{Env, run};
use serde_json::Value;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1e-9) || (a - b).abs() < 1e-9
}

fn compare(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in y {
                if k == "seconds" || k == "created" || k == "files" || k == "tag" || k == "text" {
                    continue;
                }
                assert!(x.contains_key(k), "{path}.{k} missing");
                compare(&x[k], v, &format!("{path}.{k}"));
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path} length");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                compare(p, q, &format!("{path}[{i}]"));
            }
        }
        _ => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => assert!(close(x, y), "{path}: {x} vs {y}"),
            _ => assert!(a == b || (a.is_null() && b.is_null()), "{path}: {a} vs {b}"),
        },
    }
}

#[test]
fn protocol_runs_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{root}/protocol_runs.json")).unwrap().replace("NaN", "null")).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let counts = fly_sim::wiring::synapse_counts(&path).ok().map(Arc::new);
    let tmp = std::env::temp_dir().join(format!("ktf-protorun-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    for (name, case) in g["protocol_runs_order"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>())
        .unwrap_or_else(|| g.as_object().unwrap().keys().cloned().collect::<Vec<_>>())
        .iter()
        .map(|n| (n, &g[n.as_str()]))
    {
        let env = Env { weights: &w, labels: &l, counts: counts.clone(), cancel: None };
        let folder = run(&env, &case["protocol"], &tmp.join(name), 2, |_, _, _| true, "2026-01-01 00:00:00")
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let summary: Value = serde_json::from_str(&std::fs::read_to_string(folder.join("summary.json")).unwrap()).unwrap();
        compare(&summary, &case["summary"], &format!("{name}.summary"));
        for (file, sha) in case["files"].as_object().unwrap() {
            if file.ends_with("-spikes.csv")
                || file.ends_with("-rates.csv")
                || file.ends_with("-group-rates.csv")
                || file.ends_with("-events.csv")
            {
                let got = fly_sim::util::sha256_hex(&std::fs::read(folder.join(file)).unwrap_or_else(|_| panic!("{name}: {file} missing")));
                assert_eq!(got, sha.as_str().unwrap(), "{name}: {file}");
            }
        }
        for (file, sha) in case["spike_sha"].as_object().unwrap() {
            if sha.is_null() {
                continue;
            }
            let m: Value = serde_json::from_str(&std::fs::read_to_string(folder.join(file)).unwrap()).unwrap();
            assert_eq!(m["spike_sha256"], *sha, "{name}: {file}");
        }
        eprintln!("{name}: ok");
    }
}
