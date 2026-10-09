//! The kitchen arena: a countertop with a fruit bowl, a vinegar trap, a sink, a stove burner and a cook. Plain logic, no engine, no brain.
//! Python: `game/kitchen.py`. The room's floor is the countertop; everything is placed on it in the room's metres.
//!
//! CONNECTOME vs GAME RULE per part: the vinegar scent is poked onto the fermentation glomeruli DM1/DM2/DP1m and whether the fly is
//! attracted is read from those neurons' own firing against their calm rate (the game never sends it anywhere on a timer); the sink's water
//! drives the humidity neurons as the pool does; the burner's heat drives the heat sensors; the cook's swat is something that grows in the
//! fly's view and goes through the looming transduction (LPLC2/LC4 -> DNp01), a hit fires the touch neurons. The jar, the basin, the
//! burner, the cook's schedule and speed, where he swings and how hard it hits are GAME RULES. Nothing here is measured.

use std::collections::HashMap;
use std::f32::consts::PI;

use bevy::math::Vec3;

use crate::rng::Rng;
use crate::world::{Aabb, RX, RZ};

pub const BOWL_POS: Vec3 = Vec3::new(-2.4, 0.0, 1.6);
pub const BOWL_R: f32 = 0.75;
pub const BOWL_H: f32 = 0.30;
pub const TRAP_POS: Vec3 = Vec3::new(2.3, 0.0, -1.9);
pub const TRAP_R: f32 = 0.42;
pub const TRAP_H: f32 = 0.80;
pub const TRAP_MOUTH: f32 = 0.30;
/// x0, x1, z0, z1 of the basin.
pub const SINK: (f32, f32, f32, f32) = (2.0, 3.9, 1.4, 3.3);
/// The water's surface above the counter (the pool's is 0.6).
pub const SINK_WATER: f32 = 0.30;
pub const BURNER_POS: Vec3 = Vec3::new(-0.5, 0.0, -2.5);
pub const BURNER_R: f32 = 0.5;
pub const BURNER_HEAT_RANGE: f32 = 1.1;
pub const COOK_Z: f32 = -RZ + 0.55;

pub const SCENT_RANGE: f32 = 3.4;
pub const SCENT_POKE: f32 = 0.35;
/// DM1/DM2/DP1m firing that counts as smelling it: x its calm rate, and at least this many Hz.
pub const ATTRACT_RATIO: f32 = 2.5;
pub const ATTRACT_MIN_HZ: f32 = 6.0;
pub const SCENT_WARM_S: f32 = 2.0;
pub const HOVER_TO_FALL_S: f32 = 0.4;
pub const TRAP_DROWN: f32 = 0.03;
pub const TRAP_IMMORTAL_ESCAPE_S: f32 = 8.0;

pub const COOK_EVERY_S: (f32, f32) = (10.0, 20.0);
pub const COOK_FIRST_S: (f32, f32) = (6.0, 10.0);
pub const COOK_WINDUP_S: f32 = 0.7;
pub const COOK_SWAT_S: f32 = 0.9;
pub const COOK_RECOVER_S: f32 = 1.6;
/// A fly within this far (horizontally) of the impact point, and low, is hit.
pub const COOK_REACH: f32 = 0.55;
pub const COOK_HIT_HEIGHT: f32 = 0.95;
/// Looming radius of the swatter and the forearm behind it.
pub const COOK_SWATTER_R: f32 = 0.5;
pub const COOK_HAND_HEIGHT: f32 = 2.0;
pub const COOK_DAMAGE: f32 = 38.0;
pub const COOK_SPEED: f32 = 0.6;

/// Boxes you and the walking fly go around: the jar (its top is the mouth) and the fridge.
pub fn colliders() -> Vec<Aabb> {
    vec![
        Aabb { lo: Vec3::new(TRAP_POS.x - TRAP_R, 0.0, TRAP_POS.z - TRAP_R), hi: Vec3::new(TRAP_POS.x + TRAP_R, TRAP_H, TRAP_POS.z + TRAP_R) },
        Aabb { lo: Vec3::new(-RX, 0.0, -3.35), hi: Vec3::new(-RX + 0.8, 2.1, -1.95) },
    ]
}

pub fn mouth() -> Vec3 {
    TRAP_POS + Vec3::new(0.0, TRAP_H + 0.05, 0.0)
}

/// The poke strength of the vinegar's smell at a point: `SCENT_POKE` at the jar, 0 at `SCENT_RANGE`. GAME RULE.
pub fn scent_strength(head: Vec3) -> f32 {
    SCENT_POKE * (1.0 - (head - mouth()).length() / SCENT_RANGE).max(0.0)
}

pub fn over_mouth(thorax: Vec3) -> bool {
    (thorax.x - TRAP_POS.x).hypot(thorax.z - TRAP_POS.z) <= TRAP_MOUTH && (TRAP_H - 0.05..=TRAP_H + 0.55).contains(&thorax.y)
}

/// Reads one fly's DM1/DM2/DP1m firing and decides whether it is smelling the vinegar: the rate must be at least `ATTRACT_RATIO` x its own
/// calm rate and `ATTRACT_MIN_HZ`. The calm rate is a slow average taken only while no vinegar smell has reached the fly for 3 s (so the
/// smell can't raise its own baseline), and the fly needs `SCENT_WARM_S` of that first.
#[derive(Debug, Clone)]
pub struct ScentTracker {
    pub base: Option<f32>,
    pub rate: f32,
    pub since_poke: f32,
    pub calm_time: f32,
    pub k: f32,
}

impl Default for ScentTracker {
    fn default() -> Self {
        Self { base: None, rate: 0.0, since_poke: 99.0, calm_time: 0.0, k: 0.02 }
    }
}

impl ScentTracker {
    pub fn update(&mut self, dt: f32, hz: f32, poked: bool) {
        self.rate = hz;
        self.since_poke = if poked { 0.0 } else { self.since_poke + dt };
        if self.since_poke > 3.0 {
            self.calm_time += dt;
            self.base = Some(match self.base {
                None => hz,
                Some(b) => b + (hz - b) * self.k,
            });
        }
    }

    pub fn smelling(&self) -> bool {
        self.base.is_some_and(|b| self.calm_time >= SCENT_WARM_S && self.rate >= ATTRACT_MIN_HZ.max(ATTRACT_RATIO * b))
    }
}

pub fn in_sink(p: Vec3) -> bool {
    (SINK.0..=SINK.1).contains(&p.x) && (SINK.2..=SINK.3).contains(&p.z)
}

/// Which body points are under the basin's water.
pub fn sink_submerged(points: &[Vec3]) -> Vec<bool> {
    points.iter().map(|p| in_sink(*p) && p.y < SINK_WATER).collect()
}

/// `(heat drive 0-1 at a point, touching the burner)`. GAME RULE: linear falloff to `BURNER_HEAT_RANGE`; touching is being over the ring
/// and within 0.2 m of the counter.
pub fn burner_heat(head: Vec3) -> (f32, bool) {
    let horiz = (head.x - BURNER_POS.x).hypot(head.z - BURNER_POS.z);
    let dist = (head - (BURNER_POS + Vec3::new(0.0, 0.05, 0.0))).length();
    ((1.0 - dist / BURNER_HEAT_RANGE).clamp(0.0, 1.0), horiz <= BURNER_R && head.y <= 0.2)
}

#[derive(Debug, Clone, PartialEq)]
pub enum CookEvent {
    Windup(Vec3),
    Swat(Vec3),
    Impact(Vec3),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CookState {
    Idle,
    Windup,
    Swat,
    Recover,
}

/// Paces along the back of the counter and every 10-20 s swings a swatter at where a fly was. GAME RULE.
#[derive(Debug, Clone)]
pub struct Cook {
    rng: Rng,
    pub x: f32,
    pub state: CookState,
    pub t: f32,
    wait: f32,
    goal_x: f32,
    aim: Vec3,
    hand0: Vec3,
    pub pos: Vec3,
    pub swats: u32,
}

impl Cook {
    pub fn new(mut rng: Rng) -> Self {
        let wait = rng.range(COOK_FIRST_S);
        let goal_x = rng.uniform(-2.5, 2.5);
        Self { rng, x: 0.0, state: CookState::Idle, t: 0.0, wait, goal_x, aim: Vec3::ZERO, hand0: Vec3::ZERO, pos: Vec3::new(0.0, COOK_HAND_HEIGHT * 0.6, COOK_Z + 0.4), swats: 0 }
    }

    /// `targets`: thorax positions of living flies. Returns what happened this step.
    pub fn step(&mut self, dt: f32, targets: &[Vec3]) -> Vec<CookEvent> {
        let mut ev = Vec::new();
        self.t += dt;
        match self.state {
            CookState::Idle => {
                if (self.goal_x - self.x).abs() < 0.05 {
                    self.goal_x = self.rng.uniform(-2.5, 2.5);
                }
                self.x += (COOK_SPEED * dt).min((self.goal_x - self.x).abs()).copysign(self.goal_x - self.x);
                self.pos = Vec3::new(self.x, COOK_HAND_HEIGHT * 0.6, COOK_Z + 0.4);
                if self.t >= self.wait
                    && let Some(near) = targets.iter().min_by(|a, b| ((a.x - self.x).abs() + (a.z - COOK_Z).abs()).total_cmp(&((b.x - self.x).abs() + (b.z - COOK_Z).abs())))
                {
                    self.aim = Vec3::new(near.x, 0.12, near.z); // aimed once, at where the fly is now
                    self.hand0 = Vec3::new(self.x, COOK_HAND_HEIGHT, COOK_Z + 0.5);
                    self.state = CookState::Windup;
                    self.t = 0.0;
                    ev.push(CookEvent::Windup(self.aim));
                }
            }
            CookState::Windup => {
                let f = (self.t / COOK_WINDUP_S).min(1.0);
                self.pos = Vec3::new(self.x, COOK_HAND_HEIGHT * (0.6 + 0.4 * f), COOK_Z + 0.4 + 0.1 * f);
                if self.t >= COOK_WINDUP_S {
                    self.state = CookState::Swat;
                    self.t = 0.0;
                    ev.push(CookEvent::Swat(self.aim));
                }
            }
            CookState::Swat => {
                let e = (self.t / COOK_SWAT_S).min(1.0); // a steady swing, so its growth is visible early
                self.pos = self.hand0 * (1.0 - e) + self.aim * e + Vec3::new(0.0, 0.25 * (PI * e).sin(), 0.0);
                if self.t >= COOK_SWAT_S {
                    self.state = CookState::Recover;
                    self.t = 0.0;
                    self.swats += 1;
                    ev.push(CookEvent::Impact(self.aim));
                }
            }
            CookState::Recover => {
                self.pos = self.aim + Vec3::new(0.0, 0.1 + 0.9 * (self.t / COOK_RECOVER_S).min(1.0), 0.0);
                if self.t >= COOK_RECOVER_S {
                    self.x = self.aim.x.clamp(-3.0, 3.0);
                    self.state = CookState::Idle;
                    self.t = 0.0;
                    self.wait = self.rng.range(COOK_EVERY_S);
                }
            }
        }
        ev
    }

    /// What the fly's eyes can see grow: the swatter, from the start of the swing to the impact.
    pub fn threats(&self) -> Vec<((&'static str, &'static str), Vec3, f32)> {
        if self.state == CookState::Swat { vec![(("cook", "swatter"), self.pos, COOK_SWATTER_R)] } else { Vec::new() }
    }

    /// Is a fly at this point hit by the swat that just landed?
    pub fn hit(&self, thorax: Vec3) -> bool {
        (thorax.x - self.aim.x).hypot(thorax.z - self.aim.z) <= COOK_REACH && thorax.y <= COOK_HIT_HEIGHT
    }
}

/// What the running kitchen remembers: the cook, each fly's vinegar tracker, who is hovering over the mouth and who is in.
#[derive(Debug, Clone)]
pub struct KitchenState {
    pub cook: Cook,
    pub trackers: HashMap<u32, ScentTracker>,
    /// Fly key -> seconds hovering over the mouth.
    pub hover: HashMap<u32, f32>,
    /// Fly key -> seconds inside the trap.
    pub trapped: HashMap<u32, f32>,
}

impl KitchenState {
    pub fn new(seed: u64) -> Self {
        Self { cook: Cook::new(Rng::new(seed)), trackers: HashMap::new(), hover: HashMap::new(), trapped: HashMap::new() }
    }

    pub fn tracker(&mut self, key: u32) -> &mut ScentTracker {
        self.trackers.entry(key).or_default()
    }

    pub fn release(&mut self, key: u32) {
        self.trapped.remove(&key);
        self.hover.remove(&key);
    }

    /// Where a trapped fly is held: down in the jar, bobbing a little (it is struggling).
    pub fn inside_point(&self, key: u32, t: f32) -> Vec3 {
        let mut r = Rng::new(u64::from(key) + 99);
        TRAP_POS + Vec3::new(r.uniform(-0.12, 0.12), 0.3 + 0.04 * (t * 6.0).sin(), r.uniform(-0.12, 0.12))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scent_is_strongest_at_the_jar_and_gone_past_its_range() {
        assert!((scent_strength(mouth()) - SCENT_POKE).abs() < 1e-6);
        assert_eq!(scent_strength(mouth() + Vec3::new(SCENT_RANGE + 0.1, 0.0, 0.0)), 0.0);
        assert!(scent_strength(mouth() + Vec3::new(1.0, 0.0, 0.0)) < SCENT_POKE);
    }

    #[test]
    fn a_fly_only_smells_it_after_its_own_calm_baseline_is_known() {
        let mut t = ScentTracker::default();
        t.update(0.1, 20.0, true);
        assert!(!t.smelling(), "no baseline yet, and the smell cannot set its own baseline");
        for _ in 0..80 {
            t.update(0.1, 3.0, false);
        }
        assert!(!t.smelling(), "calm rate is not attraction");
        t.update(0.1, 12.0, true);
        assert!(t.smelling(), "4x its calm rate and over 6 Hz");
        t.update(0.1, 7.0, true);
        assert!(!t.smelling(), "needs 2.5x the baseline");
    }

    #[test]
    fn the_cook_winds_up_for_0_7_s_swings_for_0_9_s_and_hits_only_what_stayed_put() {
        let mut c = Cook::new(Rng::new(1));
        let fly = Vec3::new(0.5, 0.38, -2.0);
        let (mut windup, mut swat, mut impact) = (None, None, None);
        for i in 0..(40.0 * 60.0) as usize {
            for e in c.step(1.0 / 60.0, &[fly]) {
                let t = i as f32 / 60.0;
                match e {
                    CookEvent::Windup(_) => windup = windup.or(Some(t)),
                    CookEvent::Swat(_) => swat = swat.or(Some(t)),
                    CookEvent::Impact(at) => {
                        impact = impact.or(Some(t));
                        assert!(c.hit(fly), "a fly that stayed put is hit");
                        assert!(!c.hit(fly + Vec3::new(1.5, 0.0, 0.0)), "one that moved off is not");
                        assert!(!c.hit(fly + Vec3::new(0.0, 2.0, 0.0)), "one that flew up is not");
                        assert!((at.y - 0.12).abs() < 1e-6);
                    }
                }
            }
            if impact.is_some() {
                break;
            }
        }
        let (w, s, i) = (windup.unwrap(), swat.unwrap(), impact.unwrap());
        assert!(((s - w) - COOK_WINDUP_S).abs() < 0.03 && ((i - s) - COOK_SWAT_S).abs() < 0.03);
        assert!((COOK_FIRST_S.0 - 0.1..=COOK_FIRST_S.1 + 1.0).contains(&w), "first swing at 6-10 s: {w}");
        assert!(c.threats().is_empty(), "the swatter is only a looming threat during the swing");
    }

    #[test]
    fn burner_sink_and_trap_geometry() {
        assert!(burner_heat(BURNER_POS + Vec3::new(0.0, 0.1, 0.0)).1);
        assert!(!burner_heat(BURNER_POS + Vec3::new(0.0, 1.0, 0.0)).1);
        assert_eq!(burner_heat(BURNER_POS + Vec3::new(3.0, 0.0, 0.0)).0, 0.0);
        let pts = [Vec3::new(3.0, 0.1, 2.0), Vec3::new(3.0, 0.5, 2.0), Vec3::new(0.0, 0.1, 0.0)];
        assert_eq!(sink_submerged(&pts), vec![true, false, false]);
        assert!(over_mouth(TRAP_POS + Vec3::new(0.1, TRAP_H + 0.2, 0.0)));
        assert!(!over_mouth(TRAP_POS + Vec3::new(0.9, TRAP_H + 0.2, 0.0)));
    }
}
