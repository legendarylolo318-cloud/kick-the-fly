//! `tournament.run_bracket`, `racing.{odds_table, run_race, analyze}` with a scripted `play` (`tests/golden/make_bracket.py`).

use fly_sim::duel::{DuelResult, Event, Frame, Side, TypeSums};
use fly_sim::race::{self, CardNumbers, LaneResult};
use fly_sim::tournament::{self, MatchPlayer};
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/bracket.json")).unwrap()).unwrap()
}

fn card(s: u64) -> CardNumbers {
    CardNumbers {
        loom_latency_s: 0.1 + 0.01 * (s % 7) as f64,
        sugar_ratio: 1.0 + 0.1 * (s % 5) as f64,
        turning_log_ratio: 0.02 * ((s % 3) as f64 - 1.0),
        walk_level_calm: 1.0 + 0.05 * (s % 4) as f64,
    }
}

struct Fake;

impl MatchPlayer for Fake {
    fn play(&mut self, a: u64, b: u64, attempt: u64, seconds: f64, drivers: bool) -> Result<(u64, DuelResult), String> {
        let x = (a * 31 + b * 17 + attempt * 7) % 10;
        let winner = if x == 0 {
            None
        } else if x % 2 == 1 {
            Some(Side::A)
        } else {
            Some(Side::B)
        };
        let xf = x as f64;
        let part = |k: f64| TypeSums {
            types: ["DNp35", "DNa01", "KCg", "LC10"].iter().map(|s| s.to_string()).collect(),
            hit: vec![60.0 + k + xf, 30.0, 8.0 + xf, 100.0],
            base: vec![200.0 + k, 150.0, 400.0, 900.0],
            n: vec![2.0, 4.0, 30.0, 10.0],
            hit_windows: 3 + x % 3,
            steps: 1000,
            dt: 0.005,
        };
        let r = DuelResult {
            winner,
            hp: [100.0 - xf, 100.0 - 2.0 * xf],
            shots: [x as u32 + 3, x as u32 + 1],
            hits: [(x % 3) as u32, (x % 2) as u32],
            seconds: seconds - xf,
            knockout: x % 4 == 1,
            max_levels: [[1.5, 2.0, 0.5, 3.0], [2.5, 1.0, 1.5, 2.0]],
            events: vec![Event { kind: "shot", t: 1.0 + xf, who: 0 }, Event { kind: "hit", t: 2.0, who: 1 }],
            frames: vec![Frame {
                t: 0.0,
                pos: [[1.0, 2.0], [-1.0, -2.0]],
                yaw: [0.5, 1.5],
                hp: [100.0, 100.0],
                pellets: vec![(0.5, 0.5, 0)],
            }],
            steps: 1000,
            drivers: drivers.then(|| [part(0.0), part(1.0)]),
        };
        Ok(((a * 1000 + b + attempt) & 0x7FFF_FFFF, r))
    }
}

#[test]
fn brackets_equal_python() {
    let g = golden();
    for case in g["brackets"].as_array().unwrap() {
        let seeds: Vec<u64> = case["seeds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let fav = case["favorite"].as_u64();
        let b = tournament::run_bracket(&seeds, 20.0, fav, 2, true, &mut Fake).unwrap();
        assert_eq!(b.champion, case["champion"].as_u64().unwrap());
        assert_eq!(b.favorite_won, case["favorite_won"].as_bool());
        assert_eq!(b.favorite_rounds_won.map(u64::from), case["favorite_rounds_won"].as_u64());
        for (s, w) in &b.rounds_won {
            assert_eq!(u64::from(*w), case["rounds_won"][s.to_string()].as_u64().unwrap(), "rounds won by {s}");
        }
        assert_eq!(b.rounds.len(), case["rounds"].as_array().unwrap().len());
        for (r, wr) in b.rounds.iter().zip(case["rounds"].as_array().unwrap()) {
            for (m, wm) in r.iter().zip(wr.as_array().unwrap()) {
                assert_eq!((m.a, m.b, m.winner), (wm["a"].as_u64().unwrap(), wm["b"].as_u64().unwrap(), wm["winner"].as_u64().unwrap()));
                assert_eq!(m.coin_toss, wm["coin_toss"].as_bool().unwrap());
                assert_eq!(m.attempts as u64, wm["attempts"].as_u64().unwrap());
                assert_eq!(m.decided_by, wm["decided_by"].as_str().unwrap());
                assert_eq!(m.match_seed, wm["match_seed"].as_u64().unwrap());
                let (a, bb) = (m.a.to_string(), m.b.to_string());
                assert_eq!(m.hp, [wm["hp"][&a].as_f64().unwrap(), wm["hp"][&bb].as_f64().unwrap()]);
                assert_eq!(m.shots[0] as u64, wm["shots"][&a].as_u64().unwrap());
                assert_eq!(m.hits[1] as u64, wm["hits"][&bb].as_u64().unwrap());
            }
        }
        for (rows, want) in [(&b.champion_drivers, &case["champion_drivers"]), (&b.champion_readouts, &case["champion_readouts"])] {
            let want = want.as_array().unwrap();
            assert_eq!(rows.len(), want.len());
            for (g, w) in rows.iter().zip(want) {
                assert_eq!(g.kind, w["type"].as_str().unwrap());
                assert_eq!(u64::from(g.neurons), w["neurons"].as_u64().unwrap());
                assert!((g.ratio - w["ratio"].as_f64().unwrap()).abs() < 1e-9);
                assert!((g.hz_before_hits - w["hz_before_hits"].as_f64().unwrap()).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn races_odds_and_analysis_equal_python() {
    let g = golden();
    let seeds5 = [3000u64, 3001, 3002, 3003, 3004];
    let cards5: Vec<CardNumbers> = seeds5.iter().map(|&s| card(s)).collect();
    for (o, w) in race::odds_table(&seeds5, &cards5).iter().zip(g["odds"].as_array().unwrap()) {
        assert_eq!(o.fly, w["fly"].as_u64().unwrap());
        assert!((o.p_win - w["p_win"].as_f64().unwrap()).abs() < 1e-12);
        assert!((o.form - w["form"].as_f64().unwrap()).abs() < 1e-12);
        assert_eq!(o.decimal_odds, w["decimal_odds"].as_f64().unwrap());
    }
    let mut outs = Vec::new();
    for r in g["races"].as_array().unwrap() {
        let seeds: Vec<u64> = r["seeds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let cards: Vec<CardNumbers> = seeds.iter().map(|&s| card(s)).collect();
        let mut lane = |seed: u64, repeat: usize| -> Result<LaneResult, String> {
            let fin = 20.0 + ((seed * 7 + repeat as u64 * 13) % 11) as f64;
            let did = !(seed + repeat as u64).is_multiple_of(4);
            Ok(LaneResult {
                finish_s: did.then_some(fin),
                distance: if did { 8.0 } else { 3.0 + (seed % 3) as f64 },
                trace: vec![[0.0, 0.0], [1.0, 1.0]],
                lures_touched: vec![0],
                max_walk_level: 2.0,
                seconds: if did { fin } else { 90.0 },
            })
        };
        let out = race::run_race(&seeds, 2, r["race_seed"].as_u64().unwrap(), &cards, &mut lane).unwrap();
        assert_eq!(out.winner, r["winner"].as_u64().unwrap());
        for (k, ord) in out.orders.iter().enumerate() {
            let want: Vec<u64> = r["orders"][k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            assert_eq!(*ord, want, "order of repeat {k}");
        }
        for (o, s) in out.odds.iter().zip(&seeds) {
            assert_eq!(o.decimal_odds, r["odds"][s.to_string()]["decimal_odds"].as_f64().unwrap());
        }
        outs.push((out, cards));
    }
    let an = race::analyze(&outs);
    let want = &g["race_analysis"];
    assert_eq!(an.flies as u64, want["flies"].as_u64().unwrap());
    assert_eq!(an.winners_were_favorite.map(|v| v as u64), want["winners_were_favorite"].as_u64());
    for (t, w) in an.tests.iter().zip(want["tests"].as_array().unwrap()) {
        assert_eq!(t.id, w["id"].as_str().unwrap());
        assert_eq!(t.n as u64, w["n"].as_u64().unwrap());
        for (x, y) in [(t.p, "p"), (t.p_holm, "p_holm")] {
            assert!((x - w[y].as_f64().unwrap()).abs() < 1e-9, "{} {y}: {x}", t.id);
        }
        if let (Some(r), Some(wr)) = (t.rho, w["rho"].as_f64()) {
            assert!((r - wr).abs() < 1e-12, "{} rho", t.id);
        }
        if let (Some(r), Some(wr)) = (t.mean_brier_diff, w["mean_brier_diff"].as_f64()) {
            assert!((r - wr).abs() < 1e-12);
        }
    }
    for (s, w) in an.secondary.iter().zip(want["secondary"].as_array().unwrap()) {
        assert_eq!(s.0, w["trait"].as_str().unwrap());
        let (wr, wp) = (w["rho"].as_f64(), w["p"].as_f64());
        if let Some(wr) = wr {
            assert!((s.1 - wr).abs() < 1e-12 && (s.2 - wp.unwrap()).abs() < 1e-9, "{}", s.0);
        } else {
            assert!(s.1.is_nan());
        }
    }
}
