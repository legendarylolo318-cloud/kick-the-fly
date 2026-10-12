//! scipy.stats against `fly_sim::stats` on `tests/golden/make_stats.py`'s inputs.

use fly_sim::stats;
use serde_json::Value;

fn vec(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn stats_equal_scipy() {
    let g: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/stats.json")).unwrap()).unwrap();
    for c in g["binom"].as_array().unwrap() {
        let (k, n) = (c["k"].as_u64().unwrap() as usize, c["n"].as_u64().unwrap() as usize);
        let (got, want) = (stats::binom_two_sided(k, n), c["p"].as_f64().unwrap());
        assert!((got - want).abs() < 1e-12, "binomtest({k},{n}): {got} vs {want}");
    }
    for c in g["spearman"].as_array().unwrap() {
        let (rho, p) = stats::spearman(&vec(&c["x"]), &vec(&c["y"]));
        assert!((rho - c["rho"].as_f64().unwrap()).abs() < 1e-12, "rho {rho}");
        assert!((p - c["p"].as_f64().unwrap()).abs() < 1e-9, "spearman p {p} vs {}", c["p"]);
    }
    for c in g["wilcoxon"].as_array().unwrap() {
        let (got, want) = (stats::wilcoxon_less(&vec(&c["d"])), c["p"].as_f64().unwrap());
        assert!((got - want).abs() < 1e-9, "wilcoxon n={}: {got} vs {want}", vec(&c["d"]).len());
    }
}

#[test]
fn tournament_analysis_and_noise_equal_python() {
    use fly_sim::tournament::{BracketIn, MatchIn, analyze, match_noise};
    use std::collections::HashMap;
    let g: Value =
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/stats.json")).unwrap()).unwrap();
    let a = &g["analysis"];
    let brackets: Vec<BracketIn> = a["brackets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| BracketIn {
            seeds: b["seeds"].as_array().unwrap().iter().map(|s| s.as_u64().unwrap()).collect(),
            traits: b["cards"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, c)| {
                    (
                        k.parse().unwrap(),
                        [
                            c["loom_latency_s"].as_f64().unwrap(),
                            c["sugar_ratio"].as_f64().unwrap(),
                            c["turning_log_ratio"].as_f64().unwrap(),
                        ],
                    )
                })
                .collect(),
            rounds: b["rounds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|m| MatchIn {
                            a: m["a"].as_u64().unwrap(),
                            b: m["b"].as_u64().unwrap(),
                            winner: m["winner"].as_u64().unwrap(),
                            coin_toss: m["coin_toss"].as_bool().unwrap(),
                            knockout: m["knockout"].as_bool().unwrap(),
                        })
                        .collect()
                })
                .collect(),
            rounds_won: b["rounds_won"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.parse().unwrap(), v.as_u64().unwrap() as u32))
                .collect::<HashMap<_, _>>(),
        })
        .collect();
    let got = analyze(&brackets);
    let want = &a["result"];
    assert_eq!(got.flies as u64, want["flies"].as_u64().unwrap());
    assert_eq!(got.decided_matches as u64, want["decided_matches"].as_u64().unwrap());
    assert_eq!(got.coin_tosses as u64, want["coin_tosses"].as_u64().unwrap());
    assert_eq!(got.knockouts as u64, want["knockouts"].as_u64().unwrap());
    assert_eq!(got.predicts_winning, want["predicts_winning"].as_bool().unwrap());
    for (t, w) in got.tests.iter().zip(want["tests"].as_array().unwrap()) {
        assert_eq!(t.trait_name, w["trait"].as_str().unwrap());
        assert_eq!(t.higher_trait_wins as u64, w["higher_trait_wins"].as_u64().unwrap());
        assert_eq!(t.matches as u64, w["matches"].as_u64().unwrap());
        for (x, y, n) in [(t.p, "p", 0), (t.p_holm, "p_holm", 1), (t.spearman_rho, "spearman_rho", 2), (t.spearman_p, "spearman_p", 3)] {
            let y = w[y].as_f64().unwrap();
            assert!((x - y).abs() < 1e-9, "{} col {n}: {x} vs {y}", t.trait_name);
        }
    }
    assert_eq!(match_noise(7, 1, 2, 1), (1033119759, 754526003, 411279734));
}
