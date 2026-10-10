//! `game/flyduel.run_duel` on deterministic fake brains (`tests/golden/make_duel_engine.py`): frames, events, health, the poke sequence
//! each brain received and the driver tables must be identical.

use fly_sim::duel::{self, Fighter, Side};
use serde_json::Value;

const TYPES: [&str; 10] = ["DNp35", "DNa01", "DNa02", "LC10", "LC11", "DNp01", "SNta13", "KCg", "MBON01", "DNp09"];

struct Fake {
    k: i64,
    s: i64,
    hurt: i64,
    spikes: Vec<u32>,
    types: Vec<String>,
    log: Vec<(String, Option<char>, f64, bool)>,
}

impl Fake {
    fn new(k: i64) -> Self {
        let n = 240;
        Self {
            k,
            s: 0,
            hurt: -1_000_000_000,
            spikes: vec![],
            types: (0..n).map(|i| TYPES[i % TYPES.len()].to_string()).collect(),
            log: vec![],
        }
    }
    fn tri(&self, p: i64, ph: i64) -> f64 {
        ((self.s + ph).rem_euclid(p)) as f64 / p as f64
    }
}

impl Fighter for Fake {
    fn level(&self, name: &str) -> f64 {
        let k = self.k;
        match name {
            "fire" => 0.5 + 6.0 * self.tri(97 + 3 * k, 5 * k),
            "escape" => 0.5 + 7.0 * self.tri(61 + k, 11),
            "run" => 0.5 + 4.0 * self.tri(43, 3 * k) + if self.s - self.hurt < 100 { 3.0 } else { 0.0 },
            "walk" => 0.5 + 7.0 * self.tri(151 - 4 * k, 0),
            _ => 1.0,
        }
    }
    fn hz(&self, name: &str) -> f64 {
        match name {
            "turn_r" => 6.0 + 4.0 * self.tri(37 + self.k, 2),
            "turn_l" => 1.0 + 0.5 * self.tri(53 + self.k, 7),
            _ => 0.0,
        }
    }
    fn poke(&mut self, region: &str, side: Option<char>, strength: f64, recruit: Option<f64>) {
        self.log.push((region.to_string(), side, strength, recruit.is_some()));
        if region == "body" {
            self.hurt = self.s;
        }
    }
    fn step(&mut self) -> &[u32] {
        self.s += 1;
        self.spikes = (0..240u32).filter(|i| (self.s * 7 + i64::from(*i) * 13 + self.k).rem_euclid(11) == 0).collect();
        &self.spikes
    }
    fn n(&self) -> usize {
        240
    }
    fn types(&self) -> &[String] {
        &self.types
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn rows_eq(got: &[duel::DriverRow], want: &Value, what: &str) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len(), "{what}: row count");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.kind, w["type"].as_str().unwrap(), "{what}");
        assert_eq!(u64::from(g.neurons), w["neurons"].as_u64().unwrap());
        for (a, b, n) in [(g.hz_before_hits, "hz_before_hits", 0), (g.hz_overall, "hz_overall", 1), (g.ratio, "ratio", 2)] {
            let b = f(&w[b]);
            assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "{what} {} col {n}: {a} vs {b}", g.kind);
        }
    }
}

#[test]
fn duel_engine_equals_python() {
    let g: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/duel_engine.json")).unwrap())
            .unwrap();
    for case in g.as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let (mut a, mut b) = (Fake::new(0), Fake::new(1));
        let r = duel::run_duel(&mut a, &mut b, seed, f(&case["seconds"]), true, true, |_| true);
        let w = &case["result"];
        let tag = format!("seed {seed}");
        let win = match w["winner"].as_str() {
            Some("A") => Some(Side::A),
            Some("B") => Some(Side::B),
            _ => None,
        };
        assert_eq!(r.winner, win, "{tag}");
        assert_eq!(r.hp, [f(&w["hp"]["A"]), f(&w["hp"]["B"])], "{tag}");
        assert_eq!(r.shots, [w["shots"]["A"].as_u64().unwrap() as u32, w["shots"]["B"].as_u64().unwrap() as u32], "{tag}");
        assert_eq!(r.hits, [w["hits"]["A"].as_u64().unwrap() as u32, w["hits"]["B"].as_u64().unwrap() as u32], "{tag}");
        assert_eq!(r.seconds, f(&w["seconds"]), "{tag}");
        assert_eq!(r.knockout, w["knockout"].as_bool().unwrap(), "{tag}");
        assert_eq!(r.steps, w["steps"].as_u64().unwrap(), "{tag}");
        for (i, who) in ["A", "B"].iter().enumerate() {
            let ml = &w["max_levels"][who];
            let got = r.max_levels[i];
            for (j, name) in ["fire", "escape", "run", "walk"].iter().enumerate() {
                assert_eq!(got[j], f(&ml[name]), "{tag} max level {who} {name}");
            }
        }
        let ev = w["events"].as_array().unwrap();
        assert_eq!(r.events.len(), ev.len(), "{tag}: events");
        for (e, we) in r.events.iter().zip(ev) {
            assert_eq!(e.kind, we[0].as_str().unwrap());
            assert_eq!(e.t, f(&we[1]));
            assert_eq!(["A", "B"][e.who], we[2].as_str().unwrap());
        }
        let fr = w["frames"].as_array().unwrap();
        assert_eq!(r.frames.len(), fr.len(), "{tag}: frames");
        for (i, (g, wf)) in r.frames.iter().zip(fr).enumerate() {
            let want = [f(&wf[0]), f(&wf[1]), f(&wf[2]), f(&wf[3]), f(&wf[4]), f(&wf[5]), f(&wf[6]), f(&wf[7]), f(&wf[8])];
            let got = [g.t, g.pos[0][0], g.pos[0][1], g.yaw[0], g.hp[0], g.pos[1][0], g.pos[1][1], g.yaw[1], g.hp[1]];
            assert_eq!(got, want, "{tag} frame {i}");
            let wp = wf[9].as_array().unwrap();
            assert_eq!(g.pellets.len(), wp.len(), "{tag} frame {i} pellets");
            for (p, q) in g.pellets.iter().zip(wp) {
                assert_eq!((p.0, p.1, p.2 as u64), (f(&q[0]), f(&q[1]), q[2].as_u64().unwrap()), "{tag} frame {i}");
            }
        }
        for (i, log) in [&a.log, &b.log].into_iter().enumerate() {
            let wl = case["pokes"][i].as_array().unwrap();
            assert_eq!(log.len(), wl.len(), "{tag}: pokes of {i}");
            for (p, q) in log.iter().zip(wl) {
                assert_eq!(p.0, q[0].as_str().unwrap());
                assert_eq!(p.1.map(|c| c.to_string()), q[1].as_str().map(str::to_string));
                assert_eq!(p.2, f(&q[2]), "{tag} poke strength {}", p.0);
                assert_eq!(p.3, q[3].as_bool().unwrap());
            }
        }
        // drivers
        let drv = r.drivers.as_ref().unwrap();
        for (i, who) in ["A", "B"].iter().enumerate() {
            let wd = &case["drivers"][who];
            assert_eq!(drv[i].hit_windows, wd["hit_windows"].as_u64().unwrap());
            assert_eq!(drv[i].steps, wd["steps"].as_u64().unwrap());
            let names: Vec<&str> = wd["types"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            assert_eq!(drv[i].types.iter().map(String::as_str).collect::<Vec<_>>(), names);
            for (k, v) in wd["hit"].as_array().unwrap().iter().enumerate() {
                assert_eq!(drv[i].hit[k], f(v), "{tag} hit {k}");
            }
            for (k, v) in wd["base"].as_array().unwrap().iter().enumerate() {
                assert_eq!(drv[i].base[k], f(v), "{tag} base {k}");
            }
            rows_eq(&duel::top_drivers(Some(&drv[i]), 10, 1.0, 1.0, 1), &case["top"][who], "top");
            rows_eq(&duel::top_drivers(Some(&drv[i]), 10, 50.0, 5.0, 3), &case["top_default"][who], "top_default");
            rows_eq(&duel::named_drivers(Some(&drv[i])), &case["named"][who], "named");
        }
    }
}

#[test]
fn rng_choice_without_replacement_equals_numpy() {
    let g: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/choice.json")).unwrap()).unwrap();
    for case in g.as_array().unwrap() {
        let mut rng = fly_sim::rng::Generator::new(case["seed"].as_u64().unwrap());
        for (call, want) in case["calls"].as_array().unwrap().iter().zip(case["results"].as_array().unwrap()) {
            let (n, k) = (call[0].as_u64().unwrap() as usize, call[1].as_u64().unwrap() as usize);
            let got = rng.choice_indices(n, k);
            let want: Vec<usize> = want.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(got, want, "seed {} choice({n}, {k})", case["seed"]);
        }
        assert_eq!(rng.integers(0, 1 << 30), case["after"].as_u64().unwrap(), "generator state after");
    }
}
