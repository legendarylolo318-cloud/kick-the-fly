//! Procedural animation shared by the fly, the spider and the mantis: two-bone IK legs with planted feet, and the spider's bite cycle.
//! Python: `game/predator_anim.py`. Plain math, no engine: the plugins turn the joint positions into transforms.

use bevy::math::Vec3;

/// Smoothstep, clamped to 0..1.
pub fn ease(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn unit(v: Vec3) -> Vec3 {
    v.try_normalize().unwrap_or(Vec3::ZERO)
}

/// `(knee, foot)` for a leg of segments `l1`, `l2` from `root` toward `target`. A target out of reach (or too close) is clamped, so the
/// foot is the point actually reached. `bend` is the side the knee points to.
pub fn two_bone_ik(root: Vec3, target: Vec3, l1: f32, l2: f32, bend: Vec3) -> (Vec3, Vec3) {
    let mut d = target - root;
    let mut dist = d.length();
    let (lo, hi) = ((l1 - l2).abs() + 1e-6, l1 + l2 - 1e-6);
    if dist < 1e-9 {
        d = -Vec3::Y;
        dist = 1.0;
    }
    let dirn = d / dist;
    let dist = dist.clamp(lo, hi);
    let foot = root + dirn * dist;
    let a = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist);
    let h = (l1 * l1 - a * a).max(0.0).sqrt();
    let mut b = bend - dirn * bend.dot(dirn);
    b = if b.length() > 1e-9 { unit(b) } else { unit(dirn.cross(Vec3::X)) };
    (root + dirn * a + b * h, foot)
}

/// IK legs with planted feet. Offsets are in the body frame (x forward, y up, z side); feet rest on the ground plane `y = ground`.
///
/// `groups` assigns each leg to a gait group; a leg may start a step only when no leg of another group is mid-step, which gives the
/// alternating tetrapod gait of a spider or the diagonal walk of a mantis. A planted foot never moves in the world.
#[derive(Debug, Clone)]
pub struct LegRig {
    pub hips: Vec<Vec3>,
    pub rests: Vec<Vec3>,
    pub l1: f32,
    pub l2: f32,
    pub groups: Vec<u8>,
    pub step_len: f32,
    pub step_s: f32,
    pub lift: f32,
    feet: Vec<Option<Vec3>>,
    from: Vec<Vec3>,
    to: Vec<Vec3>,
    t: Vec<f32>,
}

/// One leg's joints in world space.
#[derive(Debug, Clone, Copy)]
pub struct LegPose {
    pub hip: Vec3,
    pub knee: Vec3,
    pub foot: Vec3,
}

impl LegRig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(hips: Vec<Vec3>, rests: Vec<Vec3>, l1: f32, l2: f32, groups: Vec<u8>, step_len: f32, step_s: f32, lift: f32) -> Self {
        let n = hips.len();
        Self { hips, rests, l1, l2, groups, step_len, step_s, lift, feet: vec![None; n], from: vec![Vec3::ZERO; n], to: vec![Vec3::ZERO; n], t: vec![-1.0; n] }
    }

    pub fn len(&self) -> usize {
        self.hips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hips.is_empty()
    }

    fn world(pos: Vec3, fwd: Vec3, off: Vec3, ground: Option<f32>) -> Vec3 {
        let side = fwd.cross(Vec3::Y);
        let mut w = pos + fwd * off.x + side * off.z + Vec3::Y * off.y;
        if let Some(g) = ground {
            w.y = g;
        }
        w
    }

    /// How many feet are mid-step.
    pub fn stepping(&self) -> usize {
        self.t.iter().filter(|&&t| t >= 0.0).count()
    }

    /// Advance the feet and return every leg's joints in world space.
    ///
    /// `ground` `None`: the animal is off the ground (dangling): the feet hang `hang` below their hips and nothing is planted.
    /// `overrides` `(leg, point)`: that foot is held there (a raised foreleg in a wind-up).
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, pos: Vec3, fwd: Vec3, dt: f32, moving: bool, ground: Option<f32>, overrides: &[(usize, Vec3)], hang: f32) -> Vec<LegPose> {
        let mut fwd = unit(Vec3::new(fwd.x, 0.0, fwd.z));
        if fwd == Vec3::ZERO {
            fwd = Vec3::Z;
        }
        let side = fwd.cross(Vec3::Y);
        let mut out = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            let hip = Self::world(pos, fwd, self.hips[i], None);
            let rest = Self::world(pos, fwd, self.rests[i], ground);
            if let Some((_, p)) = overrides.iter().find(|(l, _)| *l == i) {
                self.feet[i] = Some(*p);
                self.t[i] = -1.0;
            } else if ground.is_none() {
                let r = hip + fwd * (self.rests[i].x - self.hips[i].x) + side * (self.rests[i].z - self.hips[i].z) - Vec3::Y * hang;
                self.feet[i] = Some(r);
                self.t[i] = -1.0;
            } else if self.feet[i].is_none() {
                self.feet[i] = Some(rest);
            } else if self.t[i] >= 0.0 {
                self.t[i] += dt;
                let k = ease(self.t[i] / self.step_s);
                let tgt = self.to[i];
                self.feet[i] = Some(self.from[i] + (tgt - self.from[i]) * k + Vec3::Y * (self.lift * (std::f32::consts::PI * k).sin()));
                if self.t[i] >= self.step_s {
                    self.feet[i] = Some(tgt);
                    self.t[i] = -1.0;
                }
            } else {
                let f = self.feet[i].unwrap_or(rest);
                let far = Vec3::new(f.x - rest.x, 0.0, f.z - rest.z).length();
                let busy = (0..self.len()).any(|j| self.t[j] >= 0.0 && self.groups[j] != self.groups[i]);
                if far > self.step_len && !busy {
                    let lead = if moving { fwd * (self.step_len * 0.6) } else { Vec3::ZERO };
                    self.from[i] = f;
                    self.to[i] = rest + lead;
                    self.t[i] = 0.0;
                }
            }
            let foot = self.feet[i].unwrap_or(rest);
            let bend = Vec3::Y + side * if self.hips[i].z == 0.0 { 0.0 } else { 0.35 * self.hips[i].z.signum() };
            let (knee, foot) = two_bone_ik(hip, foot, self.l1, self.l2, bend);
            out.push(LegPose { hip, knee, foot });
        }
        out
    }
}

// --- the spider's bite ---------------------------------------------------------------------------------------------------------
/// GAME RULE: 0.76 s from the first sign of a bite to the next.
pub const WINDUP_S: f32 = 0.28;
pub const STRIKE_S: f32 = 0.08;
pub const SPIDER_RECOVER_S: f32 = 0.40;
/// The bite lands if the fly is still within this many reaches at the strike.
pub const BITE_TOLERANCE: f32 = 1.45;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiderState {
    Walk,
    Windup,
    Strike,
    Recover,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiderEvent {
    Windup,
    Bite,
    Miss,
}

#[derive(Debug, Clone, Copy)]
pub struct SpiderPose {
    pub crouch: f32,
    pub raise: f32,
    pub lunge: f32,
}

/// walk -> windup -> strike -> recover. A spider in reach rears and raises its front legs (windup, nothing else moves), lunges (strike)
/// and bites at the END of the lunge if the fly is still within [`BITE_TOLERANCE`] reaches; otherwise it misses.
#[derive(Debug, Clone)]
pub struct SpiderCycle {
    pub state: SpiderState,
    pub t: f32,
}

impl Default for SpiderCycle {
    fn default() -> Self {
        Self { state: SpiderState::Walk, t: 0.0 }
    }
}

impl SpiderCycle {
    pub fn walking(&self) -> bool {
        self.state == SpiderState::Walk
    }

    fn set(&mut self, s: SpiderState) {
        self.state = s;
        self.t = 0.0;
    }

    pub fn step(&mut self, dt: f32, dist: f32, reach: f32) -> Option<SpiderEvent> {
        self.t += dt;
        match self.state {
            SpiderState::Walk if dist <= reach => {
                self.set(SpiderState::Windup);
                return Some(SpiderEvent::Windup);
            }
            SpiderState::Windup if self.t >= WINDUP_S => self.set(SpiderState::Strike),
            SpiderState::Strike if self.t >= STRIKE_S => {
                let hit = dist <= reach * BITE_TOLERANCE;
                self.set(SpiderState::Recover);
                return Some(if hit { SpiderEvent::Bite } else { SpiderEvent::Miss });
            }
            SpiderState::Recover if self.t >= SPIDER_RECOVER_S => self.set(SpiderState::Walk),
            _ => {}
        }
        None
    }

    pub fn pose(&self) -> SpiderPose {
        match self.state {
            SpiderState::Windup => {
                let k = ease(self.t / WINDUP_S);
                SpiderPose { crouch: k, raise: k, lunge: 0.0 }
            }
            SpiderState::Strike => {
                let k = ease(self.t / STRIKE_S);
                SpiderPose { crouch: 1.0 - k, raise: 1.0 - 0.6 * k, lunge: k }
            }
            SpiderState::Recover => {
                let k = ease(self.t / SPIDER_RECOVER_S);
                SpiderPose { crouch: 0.0, raise: 0.4 * (1.0 - k), lunge: 1.0 - k }
            }
            SpiderState::Walk => SpiderPose { crouch: 0.0, raise: 0.0, lunge: 0.0 },
        }
    }
}

/// Eight legs, four a side, in a tetrapod gait. `u` scales every length (1 for the 3D game's metres).
pub fn spider_rig(u: f32) -> LegRig {
    let (mut hips, mut rests, mut groups) = (Vec::new(), Vec::new(), Vec::new());
    for sgn in [-1.0f32, 1.0] {
        for (k, (hf, rf)) in [0.07, 0.025, -0.025, -0.07].into_iter().zip([0.17, 0.07, -0.07, -0.17]).enumerate() {
            hips.push(Vec3::new(hf * u, 0.0, 0.04 * sgn * u));
            rests.push(Vec3::new(rf * u, 0.0, 0.22 * sgn * u));
            groups.push(((k + usize::from(sgn > 0.0)) % 2) as u8);
        }
    }
    LegRig::new(hips, rests, 0.17 * u, 0.17 * u, groups, 0.07 * u, 0.14, 0.05 * u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ik_reaches_or_clamps_and_keeps_segment_lengths() {
        let (knee, foot) = two_bone_ik(Vec3::ZERO, Vec3::new(0.2, -0.2, 0.0), 0.17, 0.17, Vec3::Y);
        assert!((knee.length() - 0.17).abs() < 1e-4);
        assert!(((foot - knee).length() - 0.17).abs() < 1e-4);
        assert!((foot - Vec3::new(0.2, -0.2, 0.0)).length() < 1e-4, "in reach: the foot is on the target");
        let (_, far) = two_bone_ik(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0), 0.17, 0.17, Vec3::Y);
        assert!(far.length() < 0.34, "out of reach is clamped");
    }

    #[test]
    fn planted_feet_do_not_slide_while_the_body_walks() {
        let mut rig = spider_rig(1.0);
        let mut pos = Vec3::ZERO;
        let mut prev = rig.update(pos, Vec3::X, 1.0 / 60.0, true, Some(0.0), &[], 0.0);
        let mut slid = 0.0f32;
        for _ in 0..240 {
            pos += Vec3::X * 0.5 / 60.0;
            let now = rig.update(pos, Vec3::X, 1.0 / 60.0, true, Some(0.0), &[], 0.0);
            for (a, b) in prev.iter().zip(&now) {
                // a foot that moved this frame must be one that is mid-step (lifted), never a planted one dragging
                if (a.foot - b.foot).length() > 0.05 && a.foot.y < 1e-4 && b.foot.y < 1e-4 {
                    slid += 1.0;
                }
            }
            prev = now;
        }
        assert_eq!(slid, 0.0, "a grounded foot never jumps along the floor");
        assert!(pos.x > 1.9);
        let feet_far = prev.iter().all(|l| (l.foot - l.hip).length() <= 0.34 + 1e-3);
        assert!(feet_far, "no leg is stretched past its reach");
    }

    #[test]
    fn spider_bites_in_range_and_misses_when_the_fly_moved_off() {
        let run = |dist_at_strike: f32| {
            let mut c = SpiderCycle::default();
            assert_eq!(c.step(1.0 / 60.0, 0.1, 0.2), Some(SpiderEvent::Windup));
            let mut ev = None;
            for _ in 0..200 {
                let d = if c.state == SpiderState::Strike { dist_at_strike } else { 0.1 };
                if let Some(e) = c.step(1.0 / 60.0, d, 0.2) {
                    ev = Some(e);
                    break;
                }
            }
            ev
        };
        assert_eq!(run(0.25), Some(SpiderEvent::Bite));
        assert_eq!(run(0.5), Some(SpiderEvent::Miss));
    }
}
