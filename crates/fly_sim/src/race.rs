//! Fly racing: flies race down a track through their own brains. Python: `game/flyrace.py` (the lane), `lab/racing.py` (odds, lane noise).
//!
//! CONNECTOME: the walking command neurons' firing (DNp09) sets how fast a fly goes; the smell of a lure ahead drives the real olfactory
//! neurons, touching a lure drives the sugar-pathway taste and PAM reward neurons. GAME RULE: the track (8 m, three lures), how walking
//! drive becomes speed, the smell range, pulse strengths, the time cap, the lanes (flies never touch), the odds and everything about
//! betting. MODEL PREDICTION: the finishing order.
//!
//! Speed rule (version 2): `speed = V_MAX * clip(walk level / THRESH.walk, 0, 1)`, where the walk level is DNp09's firing over ONE FIXED
//! reference rate, `WALK_REF_HZ`, the same for every fly (version 1 divided by each fly's own calm baseline and was mostly warm-up luck).

use crate::duel::round_to;
use crate::util::sha256_hex;

pub const TRACK_LENGTH: f64 = 8.0;
pub const V_MAX: f64 = 0.6;
/// `(distance along the track, kind)`.
pub const LURES: [(f64, &str); 3] = [(2.0, "sugar"), (4.0, "fruit"), (6.0, "sugar")];
pub const SMELL_RANGE: f64 = 1.5;
pub const CONTACT: f64 = 0.15;
pub const SCENT_POKE: f64 = 0.3;
pub const TASTE_POKE: f64 = 0.5;
pub const REWARD_POKE: f64 = 0.5;
/// Brain steps (5 ms) per engine tick: 20 ms.
pub const TICK_STEPS: usize = 4;
pub const TIME_CAP_S: f64 = 90.0;
pub const TRACE_EVERY: usize = 10;
pub const MAX_LANES: usize = 8;
pub const RULE_VERSION: u32 = 2;
/// GAME RULE: the fixed calm DNp09 reference, exploration seeds 0-15.
pub const WALK_REF_HZ: f64 = 4.32;
pub const T_WALK: f64 = 3.0;
pub const BETA: f64 = 1.0;
pub const DEFAULT_LANES: usize = 6;

/// What a lane needs of a brain.
pub trait LaneBrain {
    fn poke(&mut self, region: &str, side: Option<char>, strength: f64);
    fn step(&mut self);
    fn walk_hz(&self) -> f64;
    fn dt(&self) -> f64 {
        0.005
    }
}

impl LaneBrain for crate::Brain {
    fn poke(&mut self, region: &str, side: Option<char>, strength: f64) {
        crate::Brain::poke(self, region, side, strength as f32);
    }
    fn step(&mut self) {
        crate::Brain::step(self);
    }
    fn walk_hz(&self) -> f64 {
        f64::from(self.hz("walk"))
    }
    fn dt(&self) -> f64 {
        self.sim.p.dt_ms / 1000.0
    }
}

pub fn walk_level(hz: f64) -> f64 {
    hz / WALK_REF_HZ
}

pub fn speed_of(level: f64, threshold: f64) -> f64 {
    V_MAX * (level / threshold).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaneResult {
    pub finish_s: Option<f64>,
    pub distance: f64,
    /// `[time, position]` samples.
    pub trace: Vec<[f64; 2]>,
    pub lures_touched: Vec<usize>,
    pub max_walk_level: f64,
    pub seconds: f64,
}

/// One fly down the track (`flyrace.run_lane`).
pub fn run_lane(brain: &mut dyn LaneBrain, cap_s: f64, length: f64, mut progress: impl FnMut(f64) -> bool) -> LaneResult {
    let dt = TICK_STEPS as f64 * brain.dt();
    let (mut x, mut t, mut max_level) = (0.0f64, 0.0f64, 0.0f64);
    let mut trace = vec![[0.0, 0.0]];
    let mut touched = [false; LURES.len()];
    let mut finish = None;
    let ticks = (cap_s / dt).round() as usize;
    for tick in 0..ticks {
        if tick % 100 == 0 && !progress(tick as f64 / ticks as f64) {
            break;
        }
        for (i, (lx, kind)) in LURES.iter().enumerate() {
            let d = lx - x;
            if d.abs() <= CONTACT {
                brain.poke("taste", None, TASTE_POKE);
                brain.poke("reward", None, REWARD_POKE);
                touched[i] = true;
            } else if 0.0 < d && d <= SMELL_RANGE {
                // only a lure ahead is smelled, stronger the closer it is
                brain.poke(&format!("scent:{kind}"), None, SCENT_POKE * (1.0 - d / SMELL_RANGE) + 0.05);
            }
        }
        for _ in 0..TICK_STEPS {
            brain.step();
        }
        let lvl = walk_level(brain.walk_hz());
        max_level = max_level.max(lvl);
        x += speed_of(lvl, T_WALK) * dt;
        t += dt;
        if tick % TRACE_EVERY == 0 {
            trace.push([round_to(t, 2), round_to(x.min(length), 3)]);
        }
        if x >= length {
            finish = Some(round_to(t, 3));
            trace.push([round_to(t, 2), length]);
            break;
        }
    }
    LaneResult {
        finish_s: finish,
        distance: round_to(x.min(length), 3),
        trace,
        lures_touched: (0..LURES.len()).filter(|&i| touched[i]).collect(),
        max_walk_level: round_to(max_level, 3),
        seconds: round_to(t, 3),
    }
}

/// Finishing order of the lanes: finished flies by time, then those that did not finish by how far they got. Ties keep lane order
/// (Python's `sorted` is stable on the dict's insertion order).
pub fn order(results: &[LaneResult]) -> Vec<usize> {
    let mut done: Vec<usize> = (0..results.len()).filter(|&i| results[i].finish_s.is_some()).collect();
    done.sort_by(|&a, &b| results[a].finish_s.partial_cmp(&results[b].finish_s).unwrap_or(std::cmp::Ordering::Equal));
    let mut rest: Vec<usize> = (0..results.len()).filter(|&i| results[i].finish_s.is_none()).collect();
    rest.sort_by(|&a, &b| results[b].distance.total_cmp(&results[a].distance));
    done.extend(rest);
    done
}

/// `racing.lane_noise`: the noise stream of one fly's run of a race.
pub fn lane_noise(race_seed: u64, fly: u64, repeat: u64) -> u64 {
    let h = sha256_hex(format!("{race_seed}:{fly}:{repeat}").as_bytes());
    u64::from(u32::from_str_radix(&h[..8], 16).unwrap_or(0).swap_bytes()) & 0x7FFF_FFFF
}

fn z(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    if sd > 1e-12 { x.iter().map(|v| (v - mean) / sd).collect() } else { vec![0.0; x.len()] }
}

/// GAME RULE: the mean of the within-field z-scores of the sugar ratio and the calm walking drive.
pub fn form_scores(sugar_ratio: &[f64], walk_level_calm: &[f64]) -> Vec<f64> {
    z(sugar_ratio).iter().zip(z(walk_level_calm)).map(|(a, b)| (a + b) / 2.0).collect()
}

/// `softmax(form * BETA)`.
pub fn win_probabilities(sugar_ratio: &[f64], walk_level_calm: &[f64]) -> Vec<f64> {
    let f: Vec<f64> = form_scores(sugar_ratio, walk_level_calm).iter().map(|v| v * BETA).collect();
    let mx = f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = f.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Runner {
        hz: f64,
        tastes: usize,
        smells: usize,
    }

    impl LaneBrain for Runner {
        fn poke(&mut self, region: &str, _: Option<char>, _: f64) {
            match region {
                "taste" => self.tastes += 1,
                r if r.starts_with("scent:") => self.smells += 1,
                _ => {}
            }
        }
        fn step(&mut self) {}
        fn walk_hz(&self) -> f64 {
            self.hz
        }
    }

    #[test]
    fn speed_rises_with_the_walk_level_and_caps_at_the_threshold() {
        assert_eq!(speed_of(0.0, 3.0), 0.0);
        assert!((speed_of(1.5, 3.0) - 0.3).abs() < 1e-12);
        assert_eq!(speed_of(9.0, 3.0), V_MAX);
    }

    #[test]
    fn a_fast_fly_finishes_touches_the_lures_and_beats_a_slow_one() {
        let mut fast = Runner { hz: WALK_REF_HZ * T_WALK, tastes: 0, smells: 0 };
        let mut slow = Runner { hz: WALK_REF_HZ, tastes: 0, smells: 0 };
        let a = run_lane(&mut fast, TIME_CAP_S, TRACK_LENGTH, |_| true);
        let b = run_lane(&mut slow, TIME_CAP_S, TRACK_LENGTH, |_| true);
        assert!((a.finish_s.unwrap() - TRACK_LENGTH / V_MAX).abs() < 0.2, "8 m at 0.6 m/s: {:?}", a.finish_s);
        assert_eq!(a.lures_touched, vec![0, 1, 2]);
        assert!(fast.smells > 0 && fast.tastes > 0);
        assert!(b.finish_s.is_none() || b.finish_s > a.finish_s);
        assert_eq!(order(&[b.clone(), a.clone()])[0], 1);
        assert_eq!(*a.trace.last().unwrap(), [round_to(a.finish_s.unwrap(), 2), TRACK_LENGTH]);
    }

    #[test]
    fn a_standing_fly_does_not_finish_and_ranks_by_distance() {
        let mut still = Runner { hz: 0.0, tastes: 0, smells: 0 };
        let r = run_lane(&mut still, 5.0, TRACK_LENGTH, |_| true);
        assert!(r.finish_s.is_none() && r.distance == 0.0);
        let far = LaneResult { distance: 3.0, ..r.clone() };
        assert_eq!(order(&[r, far]), vec![1, 0]);
    }

    #[test]
    fn odds_follow_form_and_sum_to_one() {
        let p = win_probabilities(&[1.0, 2.0, 3.0], &[1.0, 1.0, 2.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[2] > p[1] && p[1] > p[0]);
        assert_eq!(win_probabilities(&[2.0, 2.0], &[1.0, 1.0]), vec![0.5, 0.5]);
    }
}
