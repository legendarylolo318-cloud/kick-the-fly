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

// --- the bracket driver (`tournament.run_bracket`) ----------------------------------------------------------------------------------
use crate::duel::{DriverRow, DuelResult, Event, Frame, Side, TIEBREAK_REMATCHES, TypeSums, combine_drivers, named_drivers, top_drivers};

pub const SIZES: [usize; 3] = [4, 8, 16];

#[derive(Debug, Clone)]
pub struct MatchOut {
    pub a: u64,
    pub b: u64,
    pub winner: u64,
    pub coin_toss: bool,
    pub attempts: usize,
    pub decided_by: &'static str,
    pub hp: [f64; 2],
    pub shots: [u32; 2],
    pub hits: [u32; 2],
    pub seconds: f64,
    pub knockout: bool,
    pub max_levels: [[f64; 4]; 2],
    pub events: Vec<Event>,
    pub frames: Vec<Frame>,
    pub match_seed: u64,
}

#[derive(Debug, Clone)]
pub struct BracketOut {
    pub seeds: Vec<u64>,
    pub match_seconds: f64,
    pub bracket_seed: u64,
    pub favorite: Option<u64>,
    pub favorite_won: Option<bool>,
    pub favorite_rounds_won: Option<u32>,
    pub rounds: Vec<Vec<MatchOut>>,
    pub champion: u64,
    /// `(seed, rounds won)` in seed order.
    pub rounds_won: Vec<(u64, u32)>,
    pub champion_drivers: Vec<DriverRow>,
    pub champion_readouts: Vec<DriverRow>,
}

/// Plays one match: returns `(match seed, result)`. A rematch is another `attempt` (new noise). `Err` aborts (a cancel).
pub trait MatchPlayer {
    fn play(&mut self, a: u64, b: u64, attempt: u64, seconds: f64, drivers: bool) -> Result<(u64, DuelResult), String>;
}

/// Run a bracket of 4, 8 or 16 flies (seeds in the order given: 1 v 2, 3 v 4, ... then the winners in order). A draw is replayed twice
/// with new noise, then decided by a seeded coin toss that is labelled as one.
pub fn run_bracket(
    seeds: &[u64],
    seconds: f64,
    favorite: Option<u64>,
    bracket_seed: u64,
    drivers: bool,
    player: &mut dyn MatchPlayer,
) -> Result<BracketOut, String> {
    if !SIZES.contains(&seeds.len()) {
        return Err(format!("a bracket has 4, 8, 16 flies, not {}", seeds.len()));
    }
    let mut uniq = seeds.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() != seeds.len() {
        return Err("every fly needs its own individuality seed".into());
    }
    if favorite.is_some_and(|f| !seeds.contains(&f)) {
        return Err("the favorite must be one of the flies".into());
    }
    let mut wins: Vec<(u64, u32)> = seeds.iter().map(|&s| (s, 0)).collect();
    let mut by_seed: HashMap<u64, Vec<TypeSums>> = seeds.iter().map(|&s| (s, Vec::new())).collect();
    let mut fighters = seeds.to_vec();
    let mut rounds = Vec::new();
    while fighters.len() > 1 {
        let pairs: Vec<(u64, u64)> = fighters.chunks(2).map(|c| (c[0], c[1])).collect();
        let mut results: Vec<Vec<(u64, DuelResult)>> = vec![Vec::new(); pairs.len()];
        let mut pending: Vec<usize> = (0..pairs.len()).collect();
        for attempt in 0..=TIEBREAK_REMATCHES as u64 {
            let mut still = Vec::new();
            for &i in &pending {
                let r = player.play(pairs[i].0, pairs[i].1, attempt, seconds, drivers)?;
                if r.1.winner.is_none() {
                    still.push(i);
                }
                results[i].push(r);
            }
            pending = still;
            if pending.is_empty() {
                break;
            }
        }
        let mut matches = Vec::new();
        for (i, &(a, b)) in pairs.iter().enumerate() {
            let rs = &results[i];
            let (mseed, fin) = rs.last().expect("a match was played");
            let coin_toss = fin.winner.is_none();
            let winner = match fin.winner {
                Some(Side::A) => a,
                Some(Side::B) => b,
                None => {
                    if coin(bracket_seed, a, b) {
                        a
                    } else {
                        b
                    }
                }
            };
            if let Some(e) = wins.iter_mut().find(|e| e.0 == winner) {
                e.1 += 1;
            }
            for (_, r) in rs {
                if let Some(d) = &r.drivers {
                    if let Some(v) = by_seed.get_mut(&a) {
                        v.push(d[0].clone());
                    }
                    if let Some(v) = by_seed.get_mut(&b) {
                        v.push(d[1].clone());
                    }
                }
            }
            matches.push(MatchOut {
                a,
                b,
                winner,
                coin_toss,
                attempts: rs.len(),
                decided_by: if fin.winner.is_none() {
                    "draw"
                } else if fin.knockout {
                    "knockout"
                } else {
                    "health at the time limit"
                },
                hp: fin.hp,
                shots: fin.shots,
                hits: fin.hits,
                seconds: fin.seconds,
                knockout: fin.knockout,
                max_levels: fin.max_levels,
                events: fin.events.clone(),
                frames: fin.frames.clone(),
                match_seed: *mseed,
            });
        }
        fighters = matches.iter().map(|m| m.winner).collect();
        rounds.push(matches);
    }
    let champion = fighters[0];
    let sums = if drivers {
        let parts: Vec<&TypeSums> = by_seed[&champion].iter().collect();
        combine_drivers(&parts)
    } else {
        None
    };
    Ok(BracketOut {
        seeds: seeds.to_vec(),
        match_seconds: seconds,
        bracket_seed,
        favorite,
        favorite_won: favorite.map(|f| f == champion),
        favorite_rounds_won: favorite.and_then(|f| wins.iter().find(|e| e.0 == f).map(|e| e.1)),
        rounds,
        champion,
        rounds_won: wins,
        champion_drivers: if drivers { top_drivers(sums.as_ref(), 10, 50.0, 5.0, 3) } else { Vec::new() },
        champion_readouts: if drivers { named_drivers(sums.as_ref()) } else { Vec::new() },
    })
}
