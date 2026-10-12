//! Personality cards, a duel and a race lane on the real pack against Python's (`tests/golden/make_arcade.py`).

use std::sync::Arc;

use fly_sim::cards::{self, Build};
use fly_sim::pack::{self, Pack};
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/arcade.json")).unwrap()).unwrap()
}

#[test]
fn card_keys_and_pack_sha_equal_python() {
    let g = golden();
    for (k, c) in g["cards"].as_object().unwrap() {
        let (seed, mode) = k.split_once(':').unwrap();
        let key = cards::card_key(seed.parse().unwrap(), mode, None, &cards::LIF_DEFAULTS, g["pack_sha256"].as_str().unwrap());
        assert_eq!(key, c["card_key"].as_str().unwrap(), "{k}");
    }
    assert_eq!(cards::py_float(1e-5), "1e-05");
    assert_eq!(cards::py_float(0.002), "0.002");
    assert_eq!(cards::py_float(4.0), "4.0");
    assert_eq!(cards::py_float(1.5e16), "1.5e+16");
    assert_eq!(cards::py_float(123456.789), "123456.789");
}

#[test]
fn measured_cards_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let g = golden();
    let pk = Pack::load(&path).unwrap();
    let sha = cards::pack_sha256(&path).unwrap();
    assert_eq!(sha, g["pack_sha256"].as_str().unwrap());
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    for (k, c) in g["cards"].as_object().unwrap() {
        let (seed, mode) = k.split_once(':').unwrap();
        let b = Build { weights: &w, labels: &l, mode, sigma: None, lif: cards::LIF_DEFAULTS, wired: None };
        let got = cards::measure_card(&b, seed.parse().unwrap(), &sha, 600, || false).unwrap();
        let f = |n: &str| c[n].as_f64().unwrap();
        eprintln!(
            "{k}: latency {} (py {}), sugar {} (py {}), turning {} (py {}), walk {} (py {}), {}",
            got.loom_latency_s,
            f("loom_latency_s"),
            got.sugar_ratio,
            f("sugar_ratio"),
            got.turning_ratio,
            f("turning_ratio"),
            got.walk_level_calm,
            f("walk_level_calm"),
            got.title
        );
        assert_eq!(got.card_key, c["card_key"].as_str().unwrap(), "{k}");
        if mode == "strong" {
            // sigma 0.15: the gains' f32 exp differs from NumPy's SIMD exp in the last bit, and the network amplifies it (PORTING.md)
            assert_eq!(got.title, c["title"].as_str().unwrap(), "{k}");
            continue;
        }
        assert_eq!(got.title, c["title"].as_str().unwrap(), "{k}");
        assert!((got.loom_latency_s - f("loom_latency_s")).abs() < 1e-9, "{k} latency");
        assert!((got.sugar_ratio - f("sugar_ratio")).abs() < 1e-4 * f("sugar_ratio").abs().max(1.0), "{k} sugar");
        assert!((got.turning_ratio - f("turning_ratio")).abs() < 1e-4, "{k} turning");
        assert!((got.walk_level_calm - f("walk_level_calm")).abs() < 1e-4, "{k} walk");
    }
}

#[test]
fn duel_and_race_lane_on_the_real_pack_equal_python() {
    let Some(path) = pack::find_default() else { return };
    let g = golden();
    let pk = Pack::load(&path).unwrap();
    let (w, l) = (Arc::new(pk.weights), Arc::new(pk.labels));
    let b = Build { weights: &w, labels: &l, mode: "subtle", sigma: None, lif: cards::LIF_DEFAULTS, wired: None };
    let d = &g["duel"];
    let (sa, sb) = (d["a"].as_u64().unwrap(), d["b"].as_u64().unwrap());
    let (na, nb) = (d["noise"][0].as_u64().unwrap(), d["noise"][1].as_u64().unwrap());
    let (mut ba, mut bb) = (b.brain(sa, Some(na), 600), b.brain(sb, Some(nb), 600));
    let r = fly_sim::duel::run_duel(&mut ba, &mut bb, d["match_seed"].as_u64().unwrap(), 6.0, true, true, |_| true);
    let w = &d["result"];
    eprintln!(
        "duel: hp {:?} shots {:?} hits {:?} frames {} steps {}; python hp {} shots {} hits {}",
        r.hp,
        r.shots,
        r.hits,
        r.frames.len(),
        r.steps,
        w["hp"],
        w["shots"],
        w["hits"]
    );
    // the first 3.1 s of the match (12 events) are identical; after that a readout sitting on its threshold (the engine reads f32
    // levels, Python's float64 mean of them) can move one shot, so the later events and the totals are not compared (PORTING.md)
    let want: Vec<(String, f64, String)> = w["events"]
        .as_array()
        .unwrap()
        .iter()
        .take(12)
        .map(|e| (e[0].as_str().unwrap().to_string(), e[1].as_f64().unwrap(), e[2].as_str().unwrap().to_string()))
        .collect();
    let got: Vec<(String, f64, String)> =
        r.events.iter().take(12).map(|e| (e.kind.to_string(), e.t, ["2000", "2001"][e.who].to_string())).collect();
    assert_eq!(got, want);
    let n_early = r.frames.iter().take(30).count();
    for (g, wf) in r.frames.iter().zip(w["frames"].as_array().unwrap()).take(n_early) {
        assert!((g.pos[0][0] - wf[1].as_f64().unwrap()).abs() < 1e-5, "frame position");
    }
    // a race lane: individuality subtle, seed 3000, no reseed
    let mut lane_brain = b.brain(3000, None, 600);
    let lane = fly_sim::race::run_lane(&mut lane_brain, 14.0, fly_sim::race::TRACK_LENGTH, |_| true);
    let wl = &g["lane"];
    eprintln!(
        "lane: {:?} {} {:?} {} py {} {} {}",
        lane.finish_s, lane.distance, lane.lures_touched, lane.max_walk_level, wl["finish_s"], wl["distance"], wl["max_walk_level"]
    );
    assert_eq!(lane.finish_s, wl["finish_s"].as_f64());

    assert!((lane.distance - wl["distance"].as_f64().unwrap()).abs() < 0.1, "distance within 0.1 m (PORTING.md: scent poke populations)");
    assert_eq!(lane.max_walk_level, wl["max_walk_level"].as_f64().unwrap());
    assert_eq!(lane.trace.len(), wl["trace"].as_array().unwrap().len());
    assert_eq!(lane.lures_touched.len(), wl["lures_touched"].as_array().unwrap().len());
    assert!(fly_sim::race::lane_noise(7, 1, 2) > 0);
}
