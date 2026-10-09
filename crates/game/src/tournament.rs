//! Fly tournament: a single-elimination bracket of 4, 8 or 16 flies, each a different individuality seed, fighting 1v1 duels where both
//! sides are brains. Python: `lab/tournament.py`.
//!
//! GAME RULE: the arena, the blaster, the start, the bracket (seeds in the order given, paired 1-2, 3-4...), the match length, the
//! tie-break (two rematches with new noise, then a seeded coin toss that is labelled as one), the favorite. MODEL PREDICTION: who wins.
//! The bracket engine takes a [`MatchRunner`] so it is tested without brains; the game's runner builds the two brains and runs
//! `duel::run_duel`.

use fly_sim::util::sha256_hex;

use crate::duel::{DuelResult, Side, TIEBREAK_REMATCHES};

pub const SIZES: [usize; 3] = [4, 8, 16];

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub a: u32,
    pub b: u32,
    pub winner: u32,
    pub coin_toss: bool,
    pub attempts: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bracket {
    pub seeds: Vec<u32>,
    pub rounds: Vec<Vec<Match>>,
    pub champion: u32,
    pub wins: Vec<(u32, u32)>,
}

/// Plays one match: `(a, b, match seed, attempt)` -> result. A rematch is another attempt (new noise).
pub trait MatchRunner {
    fn run(&mut self, a: u32, b: u32, match_seed: u64, attempt: usize) -> DuelResult;
}

/// `(match seed, noise of a, noise of b)`: deterministic from the pair and the attempt.
pub fn match_noise(bracket_seed: u64, a: u32, b: u32, attempt: usize) -> (u64, u64, u64) {
    let h = sha256_hex(format!("{bracket_seed}:{a}:{b}:{attempt}").as_bytes());
    let word = |i: usize| u64::from(u32::from_str_radix(&h[i * 8..i * 8 + 8], 16).unwrap_or(0).swap_bytes()) & 0x7FFF_FFFF;
    (word(0), word(1), word(2))
}

/// The seeded coin toss after a draw that survived every rematch: true means `a`.
pub fn coin(bracket_seed: u64, a: u32, b: u32) -> bool {
    let h = sha256_hex(format!("coin:{bracket_seed}:{a}:{b}").as_bytes());
    u8::from_str_radix(&h[..2], 16).unwrap_or(0).is_multiple_of(2)
}

pub fn run_bracket(seeds: &[u32], bracket_seed: u64, runner: &mut impl MatchRunner) -> Result<Bracket, String> {
    if !SIZES.contains(&seeds.len()) {
        return Err(format!("a bracket has 4, 8 or 16 flies, not {}", seeds.len()));
    }
    let mut uniq = seeds.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() != seeds.len() {
        return Err("every fly needs its own individuality seed".into());
    }
    let mut fighters = seeds.to_vec();
    let mut wins: Vec<(u32, u32)> = seeds.iter().map(|&s| (s, 0)).collect();
    let mut rounds = Vec::new();
    while fighters.len() > 1 {
        let mut round = Vec::new();
        let mut next = Vec::new();
        for pair in fighters.chunks(2) {
            let (a, b) = (pair[0], pair[1]);
            let mut winner = None;
            let mut attempts = 0;
            for attempt in 0..=TIEBREAK_REMATCHES {
                attempts += 1;
                let (mseed, _, _) = match_noise(bracket_seed, a, b, attempt);
                let r = runner.run(a, b, mseed, attempt);
                if let Some(side) = r.winner {
                    winner = Some(if side == Side::A { a } else { b });
                    break;
                }
            }
            let coin_toss = winner.is_none();
            let w = winner.unwrap_or(if coin(bracket_seed, a, b) { a } else { b });
            if let Some(e) = wins.iter_mut().find(|e| e.0 == w) {
                e.1 += 1;
            }
            round.push(Match { a, b, winner: w, coin_toss, attempts });
            next.push(w);
        }
        rounds.push(round);
        fighters = next;
    }
    Ok(Bracket { seeds: seeds.to_vec(), rounds, champion: fighters[0], wins })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lower seed always wins, except pairs listed as draws.
    struct Fixed {
        draws: Vec<(u32, u32)>,
        calls: usize,
    }

    impl MatchRunner for Fixed {
        fn run(&mut self, a: u32, b: u32, _: u64, _: usize) -> DuelResult {
            self.calls += 1;
            let winner = if self.draws.contains(&(a, b)) { None } else if a < b { Some(Side::A) } else { Some(Side::B) };
            DuelResult { winner, hp: (0.0, 0.0), shots: (0, 0), hits: (0, 0), seconds: 1.0, knockout: false }
        }
    }

    #[test]
    fn a_bracket_pairs_in_order_and_plays_n_minus_one_matches() {
        let mut r = Fixed { draws: vec![], calls: 0 };
        let b = run_bracket(&[5, 2, 9, 7, 1, 8, 3, 4], 0, &mut r).unwrap();
        assert_eq!(r.calls, 7);
        assert_eq!((b.rounds.len(), b.rounds[0].len(), b.rounds[1].len(), b.rounds[2].len()), (3, 4, 2, 1));
        assert_eq!(b.rounds[0][0].winner, 2, "1st v 2nd: seeds 5 and 2");
        assert_eq!(b.champion, 1);
        assert_eq!(b.wins.iter().find(|e| e.0 == 1).unwrap().1, 3);
    }

    #[test]
    fn a_draw_is_replayed_twice_then_decided_by_a_labelled_coin_toss() {
        let mut r = Fixed { draws: vec![(1, 2)], calls: 0 };
        let b = run_bracket(&[1, 2, 3, 4], 7, &mut r).unwrap();
        let m = &b.rounds[0][0];
        assert!(m.coin_toss && m.attempts == 3);
        assert_eq!(m.winner, if coin(7, 1, 2) { 1 } else { 2 }, "the toss is seeded: the same bracket gives the same result");
        assert!(!b.rounds[0][1].coin_toss);
    }

    #[test]
    fn bad_brackets_are_refused_and_noise_is_deterministic() {
        let mut r = Fixed { draws: vec![], calls: 0 };
        assert!(run_bracket(&[1, 2, 3], 0, &mut r).is_err());
        assert!(run_bracket(&[1, 1, 2, 3], 0, &mut r).is_err());
        assert_eq!(match_noise(1, 2, 3, 0), match_noise(1, 2, 3, 0));
        assert_ne!(match_noise(1, 2, 3, 0), match_noise(1, 2, 3, 1));
    }
}
