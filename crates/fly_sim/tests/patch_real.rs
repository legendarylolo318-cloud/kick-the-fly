//! The patch clamp equals `lab/patchclamp.py` (`tests/golden/make_patch.py`): protocol, isolated unit, embedded electrode, I-F curve, CSVs.

use std::sync::Arc;

use fly_sim::cards::{Build, LIF_DEFAULTS, lif_params};
use fly_sim::pack;
use fly_sim::patchclamp::{self as pc, ClampProtocol};
use serde_json::Value;

#[test]
fn patch_equals_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_slice(&std::fs::read(format!("{root}/patch.json")).unwrap()).unwrap();
    let p = lif_params(&LIF_DEFAULTS);
    let proto = ClampProtocol {
        amplitudes: vec![0.0, 0.1],
        duration_ms: 100.0,
        pre_ms: 50.0,
        post_ms: 50.0,
        repeats: 2,
        gap_ms: 25.0,
        holding: 0.01,
    };
    assert_eq!(proto.total_steps() as u64, g["protocol"]["total"].as_u64().unwrap());
    assert_eq!(proto.schedule().len() as u64, g["protocol"]["n"].as_u64().unwrap());
    let win: Vec<(f64, u64, u64)> = proto.sweep_windows().iter().map(|w| (w.0, w.1 as u64, w.2 as u64)).collect();
    let want: Vec<(f64, u64, u64)> = g["protocol"]["windows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| (w[0].as_f64().unwrap(), w[1].as_u64().unwrap(), w[2].as_u64().unwrap()))
        .collect();
    assert_eq!(win, want);
    let iso = pc::isolated(&proto.schedule(), &p, 3, 400);
    let i = &g["isolated"];
    assert_eq!(iso.spikes.iter().filter(|s| **s).count() as u64, i["spikes"].as_u64().unwrap());
    let idx: Vec<u64> = iso.spikes.iter().enumerate().filter(|(_, s)| **s).map(|(k, _)| k as u64).take(30).collect();
    assert_eq!(idx, i["spike_idx"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<_>>());
    let vsum: f64 = iso.v.iter().map(|&x| f64::from(x)).sum();
    assert!((vsum - i["v_sum"].as_f64().unwrap()).abs() < 1e-6 * vsum.abs().max(1.0), "{vsum}");
    for (a, b) in iso.v.iter().zip(i["v_head"].as_array().unwrap()) {
        assert_eq!(f64::from(*a), b.as_f64().unwrap());
    }
    // the embedded electrode on a fresh brain
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let b = Build { weights: &w, labels: &l, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = b.brain_with(1000, None, 600, true);
    let row = pc::pick_neuron(&br, "type:DNp01", 0).unwrap();
    assert_eq!(u64::from(row), g["row"].as_u64().unwrap());
    let ep = ClampProtocol {
        amplitudes: vec![0.0, 0.5],
        duration_ms: 100.0,
        pre_ms: 50.0,
        post_ms: 50.0,
        repeats: 1,
        gap_ms: 0.0,
        holding: 0.0,
    };
    let rec = pc::run_current_clamp(&mut br, Some(row), &ep, "embedded", 0, &p).unwrap();
    let e = &g["embedded"];
    let idx: Vec<u64> = rec.spikes.iter().enumerate().filter(|(_, s)| **s).map(|(k, _)| k as u64).collect();
    assert_eq!(idx, e["spike_idx"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<_>>());
    let vsum: f64 = rec.v.iter().map(|&x| f64::from(x)).sum();
    assert!((vsum - e["v_sum"].as_f64().unwrap()).abs() < 1e-6 * vsum.abs().max(1.0), "{vsum}");
    assert_eq!(rec.meta["neuron"]["type"], e["meta"]["type"]);
    assert_eq!(rec.meta["neuron"]["body_id"], e["meta"]["body_id"]);
    // the I-F curve
    let c = pc::if_curve(&mut br, None, &[0.0, 0.05, 0.1, 0.5], 200.0, 3, "isolated", 5, &p).unwrap();
    let want = &g["curve"];
    for k in ["amplitudes", "rate_hz", "rate_sd", "mode", "duration_ms", "repeats", "rheobase"] {
        assert_eq!(c[k], want[k], "{k}");
    }
    // files
    let tmp = std::env::temp_dir().join(format!("ktf-patch-{}", std::process::id()));
    pc::export_csv(&rec, &tmp.join("trace.csv")).unwrap();
    pc::export_if_csv(&c, &tmp.join("curve.csv")).unwrap();
    for f in ["trace.csv", "curve.csv"] {
        let ours = std::fs::read_to_string(tmp.join(f)).unwrap().replace("\r\n", "\n");
        let theirs = std::fs::read_to_string(format!("{root}/patch_files/{f}")).unwrap().replace("\r\n", "\n");
        assert_eq!(ours, theirs, "{f}");
    }
    std::fs::remove_dir_all(tmp).ok();
}
