//! Fly racing: flies race down a track through their own brains. Python: `game/flyrace.py`.
//!
//! CONNECTOME: the walking command neurons' firing (DNp09) sets how fast a fly goes; the smell of a lure ahead drives the real olfactory
//! neurons, touching a lure drives the sugar-pathway taste and PAM reward neurons. GAME RULE: the track (8 m, three lures), how walking
//! drive becomes speed, the smell range, pulse strengths, the time cap, the lanes (flies never touch), and everything about betting.
//! MODEL PREDICTION: the finishing order.
//!
//! Speed rule (version 2): `speed = V_MAX * clip(walk level / THRESH.walk, 0, 1)`, where the walk level is DNp09's firing over ONE FIXED
//! reference rate, `WALK_REF_HZ`, the same for every fly (version 1 divided by each fly's own calm baseline and was mostly warm-up luck).

use crate::reactions::T_WALK;

pub const TRACK_LENGTH: f32 = 8.0;
pub const V_MAX: f32 = 0.6;
/// `(distance along the track, kind)`.
pub const LURES: [(f32, &str); 3] = [(2.0, "sugar"), (4.0, "fruit"), (6.0, "sugar")];
pub const SMELL_RANGE: f32 = 1.5;
pub const CONTACT: f32 = 0.15;
pub const SCENT_POKE: f32 = 0.3;
pub const TASTE_POKE: f32 = 0.5;
pub const REWARD_POKE: f32 = 0.5;
/// Brain steps (5 ms) per engine tick: 20 ms.
pub const TICK_STEPS: usize = 4;
pub const TIME_CAP_S: f32 = 90.0;
pub const TRACE_EVERY: usize = 10;
pub const MAX_LANES: usize = 8;
pub const RULE_VERSION: u32 = 2;
/// GAME RULE: the fixed calm DNp09 reference, exploration seeds 0-15.
pub const WALK_REF_HZ: f32 = 4.32;

/// What a lane needs of a brain.
pub trait RaceBrain {
    fn poke(&mut self, region: &str, side: Option<char>, strength: f32);
    fn step(&mut self, n: usize);
    fn walk_hz(&self) -> f32;
}

impl RaceBrain for fly_sim::Brain {
    fn poke(&mut self, region: &str, side: Option<char>, strength: f32) {
        fly_sim::Brain::poke(self, region, side, strength);
    }
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            fly_sim::Brain::step(self);
        }
    }
    fn walk_hz(&self) -> f32 {
        self.hz("walk")
    }
}

pub fn walk_level(hz: f32) -> f32 {
    hz / WALK_REF_HZ
}

pub fn speed_of(level: f32, threshold: f32) -> f32 {
    V_MAX * (level / threshold).clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
pub struct LaneResult {
    pub finish_s: Option<f32>,
    pub distance: f32,
    /// `(time, position)` samples.
    pub trace: Vec<(f32, f32)>,
    pub lures_touched: Vec<usize>,
    pub max_walk_level: f32,
    pub seconds: f32,
}

/// One fly down the track.
pub fn run_lane<B: RaceBrain>(brain: &mut B, cap_s: f32, length: f32) -> LaneResult {
    let dt = TICK_STEPS as f32 * 0.005;
    let (mut x, mut t, mut max_level) = (0.0f32, 0.0f32, 0.0f32);
    let mut trace = vec![(0.0, 0.0)];
    let mut touched: Vec<usize> = Vec::new();
    let mut finish = None;
    for tick in 0..(cap_s / dt).round() as usize {
        for (i, (lx, kind)) in LURES.iter().enumerate() {
            let d = lx - x;
            if d.abs() <= CONTACT {
                brain.poke("taste", None, TASTE_POKE);
                brain.poke("reward", None, REWARD_POKE);
                if !touched.contains(&i) {
                    touched.push(i);
                }
            } else if d > 0.0 && d <= SMELL_RANGE {
                // only a lure ahead is smelled, stronger the closer it is
                brain.poke(&format!("scent:{kind}"), None, SCENT_POKE * (1.0 - d / SMELL_RANGE) + 0.05);
            }
        }
        brain.step(TICK_STEPS);
        let lvl = walk_level(brain.walk_hz());
        max_level = max_level.max(lvl);
        x += speed_of(lvl, T_WALK) * dt;
        t += dt;
        if tick % TRACE_EVERY == 0 {
            trace.push((t, x.min(length)));
        }
        if x >= length {
            finish = Some(t);
            trace.push((t, length));
            break;
        }
    }
    touched.sort_unstable();
    LaneResult { finish_s: finish, distance: x.min(length), trace, lures_touched: touched, max_walk_level: max_level, seconds: t }
}

/// Finishing order of the lanes: finished flies by time, then those that did not finish by how far they got.
pub fn order(results: &[LaneResult]) -> Vec<usize> {
    let mut done: Vec<usize> = (0..results.len()).filter(|&i| results[i].finish_s.is_some()).collect();
    done.sort_by(|&a, &b| results[a].finish_s.partial_cmp(&results[b].finish_s).unwrap_or(std::cmp::Ordering::Equal));
    let mut rest: Vec<usize> = (0..results.len()).filter(|&i| results[i].finish_s.is_none()).collect();
    rest.sort_by(|&a, &b| results[b].distance.total_cmp(&results[a].distance));
    done.extend(rest);
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Runner {
        hz: f32,
        tastes: usize,
        smells: usize,
    }

    impl RaceBrain for Runner {
        fn poke(&mut self, region: &str, _: Option<char>, _: f32) {
            match region {
                "taste" => self.tastes += 1,
                r if r.starts_with("scent:") => self.smells += 1,
                _ => {}
            }
        }
        fn step(&mut self, _: usize) {}
        fn walk_hz(&self) -> f32 {
            self.hz
        }
    }

    #[test]
    fn speed_rises_with_the_walk_level_and_caps_at_the_threshold() {
        assert_eq!(speed_of(0.0, 3.0), 0.0);
        assert!((speed_of(1.5, 3.0) - 0.3).abs() < 1e-6);
        assert_eq!(speed_of(9.0, 3.0), V_MAX);
    }

    #[test]
    fn a_fast_fly_finishes_touches_the_lures_and_beats_a_slow_one() {
        let mut fast = Runner { hz: WALK_REF_HZ * T_WALK, tastes: 0, smells: 0 };
        let mut slow = Runner { hz: WALK_REF_HZ * 1.0, tastes: 0, smells: 0 };
        let a = run_lane(&mut fast, TIME_CAP_S, TRACK_LENGTH);
        let b = run_lane(&mut slow, TIME_CAP_S, TRACK_LENGTH);
        assert!((a.finish_s.unwrap() - TRACK_LENGTH / V_MAX).abs() < 0.2, "8 m at 0.6 m/s: {:?}", a.finish_s);
        assert_eq!(a.lures_touched, vec![0, 1, 2]);
        assert!(fast.smells > 0 && fast.tastes > 0);
        assert!(b.finish_s.is_none() || b.finish_s > a.finish_s);
        assert_eq!(order(&[b.clone(), a.clone()])[0], 1);
        assert!(*a.trace.last().unwrap() == (a.finish_s.unwrap(), TRACK_LENGTH));
    }

    #[test]
    fn a_standing_fly_does_not_finish_and_ranks_by_distance() {
        let mut still = Runner { hz: 0.0, tastes: 0, smells: 0 };
        let r = run_lane(&mut still, 5.0, TRACK_LENGTH);
        assert!(r.finish_s.is_none() && r.distance == 0.0);
        let far = LaneResult { distance: 3.0, ..r.clone() };
        assert_eq!(order(&[r, far]), vec![1, 0]);
    }
}
