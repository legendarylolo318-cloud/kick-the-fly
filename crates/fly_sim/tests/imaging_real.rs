//! Simulated imaging equals `lab/imaging.py` (`tests/golden/make_imaging.py`): Poisson draws, kernel constants, ROIs, two recordings, CSV.

use std::sync::Arc;

use fly_sim::cards::{Build, LIF_DEFAULTS};
use fly_sim::imaging::{self as im, Params};
use fly_sim::pack;
use fly_sim::rng::Generator;
use serde_json::Value;

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn rows2(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(f).collect()).collect()
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-9)
}

#[test]
fn poisson_and_kernel_equal_python() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_slice(&std::fs::read(format!("{root}/imaging.json")).unwrap()).unwrap();
    let lams: Vec<f64> = g["lams"].as_array().unwrap().iter().map(f).collect();
    let mut rng = Generator::new(5);
    for row in g["poisson"].as_array().unwrap() {
        for (k, want) in row.as_array().unwrap().iter().enumerate() {
            assert_eq!(rng.poisson(lams[k]), want.as_i64().unwrap(), "lam {}", lams[k]);
        }
    }
    for ind in &im::INDICATORS {
        let (tr, td) = im::time_constants(ind).unwrap();
        let want = &g["tau"][ind.key];
        assert!(close(tr, f(&want[0]), 1e-7) && close(td, f(&want[1]), 1e-7), "{} {tr} {td}", ind.key);
    }
    let k = im::kernel(im::indicator("gcamp6f").unwrap(), 5.0, None, 1.0).unwrap();
    let want: Vec<f64> = g["kernel6f"].as_array().unwrap().iter().map(f).collect();
    assert_eq!(k.len(), want.len());
    for (a, b) in k.iter().zip(&want) {
        assert!(close(*a, *b, 1e-6) || (a - b).abs() < 1e-12);
    }
}

fn record(br: &mut fly_sim::brain::Brain, rois: &[(String, Vec<u32>)], p: &Params, seconds: f64) -> im::Session {
    let mut s = im::Session::new(br.n(), rois, p).unwrap();
    let raster: Vec<Vec<u32>> = br.sim.activity.raster().map(<[u32]>::to_vec).collect();
    let r: Vec<&[u32]> = raster.iter().map(Vec::as_slice).collect();
    s.prime(&r, br.sim.activity.steps);
    for _ in 0..(seconds / im::DT_S).round() as usize {
        let sp = br.step().to_vec();
        s.push(&sp);
    }
    s
}

fn check(s: &im::Session, w: &Value, tol: f64) {
    assert_eq!(s.frame_steps as u64, w["frame_steps"].as_u64().unwrap());
    assert_eq!(s.tracked.len() as u64, w["tracked"].as_u64().unwrap());
    let t: Vec<f64> = w["t"].as_array().unwrap().iter().map(f).collect();
    assert_eq!(s.t, t);
    for (name, got, want) in [("true", &s.true_dff, &w["true"]), ("dff", &s.dff, &w["dff"]), ("photons", &s.photons, &w["photons"])] {
        let want = rows2(want);
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            for (x, y) in a.iter().zip(b) {
                assert!(close(*x, *y, tol) || (x - y).abs() < 1e-9, "{name} frame {i}: {x} vs {y}");
            }
        }
    }
    let full = s.view_full(1.5, None);
    let sum: f64 = full.iter().map(|&x| f64::from(x)).sum();
    assert!(close(sum, f(&w["full_sum"]), 1e-4), "{sum} vs {}", w["full_sum"]);
    assert_eq!(full.iter().filter(|&&x| x > 0.0).count() as u64, w["full_nz"].as_u64().unwrap());
    let rt = s.roi_dff_true();
    for (a, b) in rt.iter().zip(w["roi_true"].as_array().unwrap()) {
        assert!(close(*a, f(b), tol) || (a - f(b)).abs() < 1e-9);
    }
}

#[test]
fn recordings_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden");
    let g: Value = serde_json::from_slice(&std::fs::read(format!("{root}/imaging.json")).unwrap()).unwrap();
    let pk = pack::Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let b = Build { weights: &w, labels: &l, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = b.brain_with(1000, None, 600, true);
    let regions = im::rois_by_region(&l.region).unwrap();
    let names: Vec<&str> = regions.iter().map(|r| r.0.as_str()).collect();
    let want: Vec<&str> = g["region_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, want);
    let sizes: Vec<u64> = regions.iter().map(|r| r.1.len() as u64).collect();
    assert_eq!(sizes, g["region_sizes"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect::<Vec<_>>());
    let p = Params { fps: 20.0, seed: 7, ..Default::default() };
    let s = record(&mut br, &regions, &p, 2.0);
    check(&s, &g["a"], 1e-5);
    // specs
    let brain_for_rows = b.brain_with(1000, None, 0, false);
    let rows = |spec: &str| brain_for_rows.rows_of(spec).map_err(|e| e.to_string());
    let no_group = |_: &str| false;
    let specs: Vec<String> = ["type:DNp01", "prefix:KC"].iter().map(|s| s.to_string()).collect();
    let rois = im::rois_from_specs(&specs, false, &rows, &no_group).unwrap();
    let pn = im::rois_from_specs(&["DNp01".to_string()], true, &rows, &no_group).unwrap();
    assert_eq!(
        pn.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
        g["pn_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>()
    );
    assert_eq!(
        rois.iter().map(|r| r.1.len() as u64).collect::<Vec<_>>(),
        g["spec_sizes"].as_array().unwrap().iter().take(2).map(|v| v.as_u64().unwrap()).collect::<Vec<_>>()
    );
    let mut br2 = b.brain_with(1000, None, 600, true);
    let p = Params { fps: 10.0, seed: 3, shot_noise: false, baseline: "zero".into(), indicator: "6f".into(), ..Default::default() };
    let s2 = record(&mut br2, &rois, &p, 3.0);
    check(&s2, &g["b"], 1e-5);
    let got = im::csv_text(&s2.result().unwrap());
    let want = std::fs::read_to_string(format!("{root}/imaging_files/roi.csv")).unwrap();
    assert_eq!(got, want);
}
