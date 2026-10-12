//! The Lab's jobs on the real connectome through the same calls the pages make: a 2-fly sugar assay with a surgery (skipped without the pack).

use std::sync::Arc;
use std::time::{Duration, Instant};

use fly_sim::pack::{self, Pack};
use kick_the_fly::lab_play::LabRuns;
use kick_the_fly::sim_thread::Shared;
use serde_json::json;

#[test]
fn an_assay_with_a_surgery_runs_and_compares() {
    let Some(path) = pack::find_default() else {
        eprintln!("SKIPPED: no brain pack");
        return;
    };
    let p = Pack::load(path).unwrap();
    let shared = Shared { weights: Arc::new(p.weights), labels: Arc::new(p.labels) };
    let mut runs = LabRuns::default();
    runs.start_assay(shared, &json!({"kind": "sugar", "flies": 2, "base_seed": 1000, "surgery": {"type:MN9": -1}, "label": "MN9"}));
    let t = Instant::now();
    while runs.busy_assay() {
        runs.poll();
        assert!(t.elapsed() < Duration::from_secs(900), "the assay did not finish");
        std::thread::sleep(Duration::from_millis(100));
    }
    let q = runs.assay_query(1000);
    let r = &q["result"];
    assert_eq!(r["kind"], "sugar");
    assert_eq!(r["seeds"], json!([3000, 3001]));
    assert_eq!(r["treated"]["rows"].as_array().unwrap().len(), 7, "seven doses");
    assert!(r["comparison"]["overall"]["p_value"].is_number());
    // silencing MN9 leaves it no spikes, so its firing ratio is nil at every dose
    assert!(r["treated"]["rows"][6]["mn9_ratio"]["mean"].as_f64().unwrap() < 0.2, "{}", r["treated"]["rows"][6]);
    // an assay this build does not have says so
    runs.start_assay(
        Shared {
            weights: Arc::new(Pack::load(pack::find_default().unwrap()).unwrap().weights),
            labels: Arc::new(Pack::load(pack::find_default().unwrap()).unwrap().labels),
        },
        &json!({"kind": "orchard", "flies": 2}),
    );
    let t = Instant::now();
    loop {
        runs.poll();
        let q = runs.assay_query(1000);
        if let Some(e) = q["job"]["error"].as_str() {
            assert!(e.contains("not part of this build"), "{e}");
            break;
        }
        assert!(t.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(50));
    }
}
