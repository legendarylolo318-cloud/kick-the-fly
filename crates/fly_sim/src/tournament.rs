//! Tournament analysis and noise streams. Python: `lab/tournament.py` (`match_noise`, `analyze`, `holm`).
//!
//! Pre-registered analysis: over the matches decided by the brains (coin tosses excluded), how often does the fly with the higher trait
//! value win? Exact two-sided binomial test against 1/2, Holm-corrected over the three traits. Spearman of each trait with the rounds a
//! fly won is reported, uncorrected.

use std::collections::HashMap;

use crate::stats;
use crate::util::sha256_hex;

pub const TRAITS: [&str; 3] = ["loom_latency_s", "sugar_ratio", "turning_log_ratio"];
pub const TRAIT_LABEL: [&str; 3] = ["looming latency to dodge (s)", "sugar -> MN9 ratio", "log DNa01/02 right/left"];

/// `(match seed, noise of a, noise of b)`: deterministic from the pair and the attempt (a rematch is another attempt).
pub fn match_noise(bracket_seed: u64, a: u64, b: u64, attempt: u64) -> (u64, u64, u64) {
    let h = sha256_hex(format!("{bracket_seed}:{a}:{b}:{attempt}").as_bytes());
    let word = |i: usize| u64::from(u32::from_str_radix(&h[i * 8..i * 8 + 8], 16).unwrap_or(0).swap_bytes()) & 0x7FFF_FFFF;
    (word(0), word(1), word(2))
}

/// The seeded coin toss after a draw that survived every rematch: true means `a`.
pub fn coin(bracket_seed: u64, a: u64, b: u64) -> bool {
    let h = sha256_hex(format!("coin:{bracket_seed}:{a}:{b}").as_bytes());
    u8::from_str_radix(&h[..2], 16).unwrap_or(0).is_multiple_of(2)
}

#[derive(Debug, Clone)]
pub struct MatchIn {
    pub a: u64,
    pub b: u64,
    pub winner: u64,
    pub coin_toss: bool,
    pub knockout: bool,
}

#[derive(Debug, Clone)]
pub struct BracketIn {
    pub seeds: Vec<u64>,
    /// seed -> `[loom_latency_s, sugar_ratio, turning_log_ratio]`
    pub traits: HashMap<u64, [f64; 3]>,
    pub rounds: Vec<Vec<MatchIn>>,
    pub rounds_won: HashMap<u64, u32>,
}

#[derive(Debug, Clone)]
pub struct TraitTest {
    pub trait_name: &'static str,
    pub label: &'static str,
    pub matches: usize,
    pub higher_trait_wins: usize,
    pub win_share_of_higher: f64,
    pub p: f64,
    pub spearman_rho: f64,
    pub spearman_p: f64,
    pub p_holm: f64,
    pub significant: bool,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub brackets: usize,
    pub flies: usize,
    pub decided_matches: usize,
    pub coin_tosses: usize,
    pub knockouts: usize,
    pub tests: Vec<TraitTest>,
    pub predicts_winning: bool,
}

pub fn analyze(brackets: &[BracketIn]) -> Analysis {
    let mut flies: Vec<([f64; 3], f64)> = Vec::new();
    for b in brackets {
        for s in &b.seeds {
            flies.push((b.traits[s], f64::from(b.rounds_won[s])));
        }
    }
    let decided: Vec<(&BracketIn, &MatchIn)> =
        brackets.iter().flat_map(|b| b.rounds.iter().flatten().map(move |m| (b, m))).filter(|(_, m)| !m.coin_toss).collect();
    let mut tests = Vec::new();
    for (ti, t) in TRAITS.iter().enumerate() {
        let (mut higher_wins, mut n) = (0usize, 0usize);
        for (b, m) in &decided {
            let (va, vb) = (b.traits[&m.a][ti], b.traits[&m.b][ti]);
            if va == vb {
                continue;
            }
            n += 1;
            let higher = if va > vb { m.a } else { m.b };
            higher_wins += usize::from(m.winner == higher);
        }
        let p = if n > 0 { stats::binom_two_sided(higher_wins, n) } else { 1.0 };
        let xs: Vec<f64> = flies.iter().map(|f| f.0[ti]).collect();
        let ys: Vec<f64> = flies.iter().map(|f| f.1).collect();
        let distinct = |v: &[f64]| v.iter().any(|x| *x != v[0]);
        let (rho, rp) = if distinct(&xs) && distinct(&ys) { stats::spearman(&xs, &ys) } else { (f64::NAN, f64::NAN) };
        tests.push(TraitTest {
            trait_name: t,
            label: TRAIT_LABEL[ti],
            matches: n,
            higher_trait_wins: higher_wins,
            win_share_of_higher: if n > 0 { higher_wins as f64 / n as f64 } else { f64::NAN },
            p,
            spearman_rho: rho,
            spearman_p: rp,
            p_holm: 0.0,
            significant: false,
        });
    }
    let adj = stats::holm(&tests.iter().map(|t| t.p).collect::<Vec<_>>());
    for (t, ph) in tests.iter_mut().zip(adj) {
        t.p_holm = ph;
        t.significant = ph < 0.05;
    }
    Analysis {
        brackets: brackets.len(),
        flies: flies.len(),
        decided_matches: decided.len(),
        coin_tosses: brackets.iter().flat_map(|b| b.rounds.iter().flatten()).filter(|m| m.coin_toss).count(),
        knockouts: decided.iter().filter(|(_, m)| m.knockout).count(),
        predicts_winning: tests.iter().any(|t| t.significant),
        tests,
    }
}
