//! Fly vs fly duel: the 1v1 duel with a second brain in place of you. Nothing about who wins is scripted. Python: `game/flyduel.py`.
//!
//! CONNECTOME: what the neurons do with what they are given (turning is DNa01/02 right minus left, shooting is DNp35/DNpe052 above its
//! threshold, the dodge the giant fiber, the run the body-touch group). GAME RULE: the arena, the blaster, how a turning command becomes a
//! heading change, what a hit does, the match length. MODEL PREDICTION: who wins; a prediction of this model, not of real flies.
//!
//! The engine takes anything that looks like a brain ([`Fighter`]) so it can be tested without the real pack.

use std::f32::consts::{PI, TAU};

use bevy::math::Vec2;

use crate::reactions::{T_ESCAPE, T_FIRE, T_RUN, T_WALK};
use crate::rng::Rng;

pub const HP: f32 = 100.0;
/// Brain steps (5 ms) per engine tick: 20 ms.
pub const TICK_STEPS: usize = 4;
pub const ARENA_HALF: f32 = 6.0;
pub const START_APART: f32 = 6.0;
pub const START_YAW_JITTER: f32 = 1.2;
pub const FLY_RADIUS: f32 = 0.32;
pub const PELLET_SPEED: f32 = 9.0;
pub const PELLET_SPREAD: f32 = 0.035;
pub const PELLET_DAMAGE: f32 = 9.0;
pub const PELLET_LIFE_S: f32 = 1.6;
pub const PELLET_RADIUS: f32 = 0.10;
pub const FIRE_COOLDOWN: f32 = 0.45;
pub const REWARD_PULSE_S: f32 = 0.45;
pub const PUNISH_PULSE_S: f32 = 0.6;
pub const STEER_DEADZONE_HZ: f32 = 1.5;
pub const STEER_GAIN: f32 = 0.004;
pub const STEER_MAX: f32 = 0.035;
pub const TRACK_SCALE: f32 = 0.45;
pub const WALK_SPEED: f32 = 0.5;
pub const DODGE_HOP: f32 = 0.9;
pub const DODGE_COOLDOWN: f32 = 1.2;
pub const RUN_HOP: f32 = 0.7;
pub const RUN_COOLDOWN: f32 = 1.5;
pub const SEE_RANGE: f32 = 7.5;
pub const DEFAULT_SECONDS: f32 = 20.0;
pub const TIEBREAK_REMATCHES: usize = 2;
/// Rad/s of angular expansion: below `LOOM_MIN` nothing, `LOOM_MIN + LOOM_FULL` = full drive.
pub const LOOM_MIN: f32 = 1.5;
pub const LOOM_FULL: f32 = 8.0;

/// The part of a brain the duel needs.
pub trait Fighter {
    fn level(&self, name: &str) -> f32;
    fn hz(&self, name: &str) -> f32;
    fn poke(&mut self, region: &str, side: Option<char>, strength: f32, recruit: Option<f32>);
    /// Advance the brain by `n` 5 ms steps.
    fn step(&mut self, n: usize);
}

impl Fighter for fly_sim::Brain {
    fn level(&self, name: &str) -> f32 {
        fly_sim::Brain::level(self, name)
    }
    fn hz(&self, name: &str) -> f32 {
        fly_sim::Brain::hz(self, name)
    }
    fn poke(&mut self, region: &str, side: Option<char>, strength: f32, recruit: Option<f32>) {
        self.poke_recruit(region, side, strength, recruit);
    }
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            fly_sim::Brain::step(self);
        }
    }
}

fn wrap(a: f32) -> f32 {
    (a + PI).rem_euclid(TAU) - PI
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

#[derive(Debug, Clone)]
pub struct Duelist {
    pub pos: Vec2,
    pub yaw: f32,
    pub hp: f32,
    steer: f32,
    fire_ready: f32,
    dodge_ready: f32,
    run_ready: f32,
    reward_until: f32,
    punish_until: f32,
    hop: Vec2,
    pub shots: u32,
    pub hits: u32,
    pub max_fire: f32,
}

impl Duelist {
    fn new(pos: Vec2, yaw: f32) -> Self {
        Self { pos, yaw, hp: HP, steer: 0.0, fire_ready: 0.0, dodge_ready: 0.0, run_ready: 0.0, reward_until: 0.0, punish_until: 0.0, hop: Vec2::ZERO, shots: 0, hits: 0, max_fire: 0.0 }
    }

    pub fn fwd(&self) -> Vec2 {
        Vec2::new(self.yaw.cos(), self.yaw.sin())
    }
}

#[derive(Debug, Clone)]
pub struct Pellet {
    pub pos: Vec2,
    pub vel: Vec2,
    pub born: f32,
    pub owner: Side,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DuelResult {
    pub winner: Option<Side>,
    pub hp: (f32, f32),
    pub shots: (u32, u32),
    pub hits: (u32, u32),
    pub seconds: f32,
    pub knockout: bool,
}

/// What `me` sees of `other`: LC10 target tracking on the side the other is on, small-object detectors when it is dead ahead, plus
/// looming from the other's pellets coming at it (game rule: a 0.1 m pellet).
fn senses<F: Fighter>(br: &mut F, me: &Duelist, other: &Duelist, pellets: &[Pellet], me_side: Side) {
    let to = other.pos - me.pos;
    let dist = to.length();
    if dist <= SEE_RANGE && me.hp > 0.0 {
        let u = to / dist.max(1e-6);
        let f = me.fwd();
        let side_v = Vec2::new(-f.y, f.x);
        let (ahead, sideness) = (u.dot(f), u.dot(side_v));
        let near = (1.3 - dist / 7.0).clamp(0.3, 1.0);
        let s_side = (sideness.abs() / TRACK_SCALE).clamp(0.0, 1.0) * near;
        let key = if sideness > 0.0 { 'L' } else { 'R' }; // the 3D duel's +side is its right: this plane's +side is the left
        if ahead < 0.995 && s_side > 0.03 {
            br.poke("track", Some(key), 0.05, Some(0.15 + 0.45 * s_side));
        }
        if ahead > 0.985 && dist < 7.0 {
            let s = near * ((ahead - 0.985) / 0.012).clamp(0.0, 1.0);
            if s > 0.02 {
                let sides: &[char] = if sideness.abs() < 0.12 { &['L', 'R'] } else { std::slice::from_ref(&key) };
                for sd in sides {
                    br.poke("small", Some(*sd), 0.05, Some(0.2 + 0.4 * s));
                }
            }
        }
    }
    let mut best = 0.0f32;
    for p in pellets.iter().filter(|p| p.owner != me_side) {
        let rel = p.pos - me.pos;
        let d = rel.length();
        if !(0.3..=4.0).contains(&d) {
            continue;
        }
        let closing = -rel.dot(p.vel) / d;
        if closing > 0.0 {
            best = best.max(2.0 * PELLET_RADIUS * closing / (d * d));
        }
    }
    if best > 0.0 && me.hp > 0.0 {
        let strength = ((best - LOOM_MIN) / LOOM_FULL).clamp(0.0, 1.0);
        if strength > 0.0 {
            br.poke("loom", None, strength, Some(0.6 * strength));
        }
    }
}

/// Steering from DNa01/02 right minus left, shooting above the threshold, dodging on the giant fiber, running on the body-touch group.
#[allow(clippy::too_many_arguments)]
fn motor<F: Fighter>(br: &mut F, me: &mut Duelist, other: &Duelist, pellets: &mut Vec<Pellet>, me_side: Side, t: f32, dt: f32, rng: &mut Rng) {
    if me.hp <= 0.0 {
        return;
    }
    if t < me.reward_until {
        br.poke("reward", None, 0.8, None);
    }
    if t < me.punish_until {
        br.poke("punish", None, 1.0, None);
    }
    let turn = br.hz("turn_r") - br.hz("turn_l");
    me.steer += (turn - me.steer) * 0.35;
    let mag = (me.steer.abs() - STEER_DEADZONE_HZ).max(0.0);
    if mag > 0.0 {
        let dyaw = (mag * STEER_GAIN).min(STEER_MAX).copysign(me.steer) * (dt * 60.0);
        me.yaw = wrap(me.yaw - dyaw); // heading is counter-clockwise in this plane, so "right" is a negative step
    }
    let (fire, escape, run, walk) = (br.level("fire"), br.level("escape"), br.level("run"), br.level("walk"));
    me.max_fire = me.max_fire.max(fire);
    if fire > T_FIRE && t >= me.fire_ready {
        me.fire_ready = t + FIRE_COOLDOWN;
        let ang = me.yaw + rng.normal() * PELLET_SPREAD;
        pellets.push(Pellet { pos: me.pos + me.fwd() * 0.25, vel: Vec2::new(ang.cos(), ang.sin()) * PELLET_SPEED, born: t, owner: me_side });
        me.shots += 1;
    }
    let away = (me.pos - other.pos).normalize_or_zero();
    if escape > T_ESCAPE && t >= me.dodge_ready {
        let nearest = pellets.iter().filter(|p| p.owner != me_side && (p.pos - me.pos).length() < 4.0).min_by(|a, b| (a.pos - me.pos).length().total_cmp(&(b.pos - me.pos).length()));
        if let Some(p) = nearest {
            // a hop sideways to the nearest incoming pellet's line
            let dirp = p.vel / PELLET_SPEED;
            let mut perp = Vec2::new(-dirp.y, dirp.x);
            if perp.dot(me.pos - p.pos) < 0.0 {
                perp = -perp;
            }
            me.hop += perp * DODGE_HOP;
            me.dodge_ready = t + DODGE_COOLDOWN;
        }
    }
    if run > T_RUN && t >= me.run_ready && t < me.punish_until + 0.5 {
        me.hop += away * RUN_HOP; // hurt, and the body-touch group says run
        me.run_ready = t + RUN_COOLDOWN;
    }
    let mut step = Vec2::ZERO;
    if walk > T_WALK {
        step += me.fwd() * WALK_SPEED * dt;
    }
    if me.hop.length() > 1e-6 {
        let part = me.hop * (dt / 0.2).min(1.0); // hops play out over ~0.2 s
        step += part;
        me.hop -= part;
    }
    me.pos = (me.pos + step).clamp(Vec2::splat(-ARENA_HALF), Vec2::splat(ARENA_HALF));
}

fn move_pellets(pellets: &mut Vec<Pellet>, a: &mut Duelist, b: &mut Duelist, t: f32, dt: f32) {
    let mut keep = Vec::with_capacity(pellets.len());
    for mut p in pellets.drain(..) {
        p.pos += p.vel * dt;
        let (shooter, target) = if p.owner == Side::A { (&mut *a, &mut *b) } else { (&mut *b, &mut *a) };
        if target.hp > 0.0 && (p.pos - target.pos).length() < FLY_RADIUS {
            target.hp = (target.hp - PELLET_DAMAGE).max(0.0);
            shooter.hits += 1;
            shooter.reward_until = t + REWARD_PULSE_S;
            target.punish_until = t + PUNISH_PULSE_S;
            continue;
        }
        if p.pos.abs().max_element() > ARENA_HALF || t - p.born > PELLET_LIFE_S {
            continue;
        }
        keep.push(p);
    }
    *pellets = keep;
}

/// One match between two brains. The match seed sets the starting headings and the pellets' scatter; the brains' own noise is theirs.
/// The match ends when one health reaches 0 or the time is up (more health wins; equal health is a draw: the caller decides).
pub fn run_duel<F: Fighter>(ba: &mut F, bb: &mut F, seed: u64, seconds: f32) -> DuelResult {
    let mut rng = Rng::new(seed & 0x7FFF_FFFF);
    let mut a = Duelist::new(Vec2::new(-START_APART / 2.0, 0.0), rng.uniform(-START_YAW_JITTER, START_YAW_JITTER));
    let mut b = Duelist::new(Vec2::new(START_APART / 2.0, 0.0), PI + rng.uniform(-START_YAW_JITTER, START_YAW_JITTER));
    let mut pellets: Vec<Pellet> = Vec::new();
    let dt = TICK_STEPS as f32 * 0.005;
    let ticks = (seconds / dt).round() as usize;
    let mut done = 0;
    for tick in 0..ticks {
        let t = tick as f32 * dt;
        senses(ba, &a, &b, &pellets, Side::A);
        senses(bb, &b, &a, &pellets, Side::B);
        ba.step(TICK_STEPS);
        bb.step(TICK_STEPS);
        let (ca, cb) = (b.clone(), a.clone());
        motor(ba, &mut a, &ca, &mut pellets, Side::A, t, dt, &mut rng);
        motor(bb, &mut b, &cb, &mut pellets, Side::B, t, dt, &mut rng);
        move_pellets(&mut pellets, &mut a, &mut b, t, dt);
        done = tick + 1;
        if a.hp <= 0.0 || b.hp <= 0.0 {
            break;
        }
    }
    let winner = if a.hp > b.hp { Some(Side::A) } else if b.hp > a.hp { Some(Side::B) } else { None };
    DuelResult { winner, hp: (a.hp, b.hp), shots: (a.shots, b.shots), hits: (a.hits, b.hits), seconds: done as f32 * dt, knockout: a.hp <= 0.0 || b.hp <= 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A brain that always fires and turns toward the middle (positive right/left difference chosen per side).
    struct Gunner {
        fire: f32,
        turn_r: f32,
    }

    impl Fighter for Gunner {
        fn level(&self, name: &str) -> f32 {
            if name == "fire" { self.fire } else { 0.0 }
        }
        fn hz(&self, name: &str) -> f32 {
            if name == "turn_r" { self.turn_r } else { 0.0 }
        }
        fn poke(&mut self, _: &str, _: Option<char>, _: f32, _: Option<f32>) {}
        fn step(&mut self, _: usize) {}
    }

    #[test]
    fn a_fighter_that_cannot_shoot_loses_to_one_that_can_if_it_gets_hit() {
        let mut wins_a = 0;
        let mut hits = 0;
        for seed in 0..40 {
            let mut a = Gunner { fire: 5.0, turn_r: 0.0 };
            let mut b = Gunner { fire: 0.0, turn_r: 0.0 };
            let r = run_duel(&mut a, &mut b, seed, DEFAULT_SECONDS);
            assert_eq!(r.shots.1, 0);
            hits += r.hits.0;
            if r.winner == Some(Side::A) {
                wins_a += 1;
            }
            assert!(r.winner != Some(Side::B));
        }
        assert!(hits > 0, "a scattering gunner hits sometimes");
        assert!(wins_a > 0);
    }

    #[test]
    fn identical_idle_brains_draw_and_the_match_is_repeatable() {
        let run = |seed| run_duel(&mut Gunner { fire: 0.0, turn_r: 0.0 }, &mut Gunner { fire: 0.0, turn_r: 0.0 }, seed, 5.0);
        let r = run(3);
        assert_eq!(r.winner, None);
        assert_eq!(r, run(3));
        assert!((r.seconds - 5.0).abs() < 0.05 && !r.knockout);
    }

    #[test]
    fn steering_is_a_deadzone_plus_a_limited_turn_rate() {
        let mut g = Gunner { fire: 0.0, turn_r: 1.0 };
        let mut d = Duelist::new(Vec2::ZERO, 0.0);
        let other = Duelist::new(Vec2::new(5.0, 0.0), PI);
        let mut rng = Rng::new(1);
        let mut pellets = Vec::new();
        for _ in 0..100 {
            motor(&mut g, &mut d, &other, &mut pellets, Side::A, 0.0, 1.0 / 60.0, &mut rng);
        }
        assert_eq!(d.yaw, 0.0, "1 Hz of right-minus-left is inside the 1.5 Hz dead zone");
        g.turn_r = 12.0;
        for _ in 0..60 {
            motor(&mut g, &mut d, &other, &mut pellets, Side::A, 0.0, 1.0 / 60.0, &mut rng);
        }
        assert!(d.yaw < -0.5 && d.yaw > -2.2, "turned right (negative) at most 0.035 rad/frame: {}", d.yaw);
    }
}
