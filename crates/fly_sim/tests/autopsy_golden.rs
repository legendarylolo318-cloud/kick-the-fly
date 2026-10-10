//! `Game._autopsy` against Rust's on the scripted run of `tests/golden/make_autopsy.py` (calm, a touch drive, calm, death, flat-line).

use std::sync::Arc;

use fly_sim::Brain;
use fly_sim::autopsy;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{self, Pack};

#[test]
fn autopsy_matches_python() {
    let Some(path) = pack::find_default() else { return };
    let g: serde_json::Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/autopsy.json")).unwrap()).unwrap();
    let pk = Pack::load(path).unwrap();
    let mut b = Brain::new(LifSim::new(Arc::new(pk.weights), LifParams::default(), 3), Arc::new(pk.labels));
    b.warmup(600);
    let mut touch = b.groups_ref().sense("head", None).to_vec();
    touch.extend_from_slice(b.groups_ref().sense("body", None));
    let run = |b: &mut Brain, n: usize| (0..n).for_each(|_| _ = b.step());
    run(&mut b, 1500);
    b.drive(&touch, 0.3);
    run(&mut b, 600);
    b.undrive(&touch);
    run(&mut b, 400);
    b.kill();
    run(&mut b, 800);
    assert_eq!(b.death_sample as u64, g["ds"].as_u64().unwrap());
    assert_eq!(b.groups_ref().hist_n as u64, g["hist_n"].as_u64().unwrap());
    let a = autopsy::compute(b.groups_ref(), b.death_sample, &b.death_base, b.groups_ref().pain_level).unwrap();
    assert_eq!(a.n as u64, g["n"].as_u64().unwrap());
    assert_eq!(a.death_at as u64, g["death_at"].as_u64().unwrap());
    let close = |x: f32, y: f64, what: &str| assert!((f64::from(x) - y).abs() <= 1e-3 * y.abs() + 1e-4, "{what}: {x} vs {y}");
    for (r, w) in a.rows.iter().zip(g["rows"].as_array().unwrap()) {
        let name = w["name"].as_str().unwrap();
        close(r.base, w["base"].as_f64().unwrap(), &format!("{name} base"));
        close(r.final_, w["final"].as_f64().unwrap(), &format!("{name} final"));
        close(r.after, w["after"].as_f64().unwrap(), &format!("{name} after"));
        close(r.ratio, w["ratio"].as_f64().unwrap(), &format!("{name} ratio"));
    }
    for (s, w) in a.series.iter().zip(g["series"].as_array().unwrap()) {
        let w = w.as_array().unwrap();
        assert_eq!(s.1.len(), w.len());
        for (k, (x, y)) in s.1.iter().zip(w).enumerate() {
            close(*x, y.as_f64().unwrap(), &format!("{} [{k}]", s.0));
        }
    }
    let wp = g["pain"].as_array().unwrap();
    assert_eq!(a.pain.len(), wp.len());
    for (k, (x, y)) in a.pain.iter().zip(wp).enumerate() {
        close(*x, y.as_f64().unwrap(), &format!("pain[{k}]"));
    }
    close(a.pain_last, g["pain_last"].as_f64().unwrap(), "pain_last");
}
