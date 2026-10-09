//! The arenas' space and the outdoors' sun, wind and fruit. Plain logic, no engine. Python: `game/outdoors.py` and the constants of
//! `game/kick3d.py` (room size, furniture boxes, lamp/fan/flypaper placement).
//!
//! CONNECTOME vs GAME RULE (see the Python docstring): wind drives the Johnston's organ wind neurons JO-C/E, sunlight the photoreceptors,
//! fruit the sugar-pathway taste and PAM reward neurons; the transductions, the cycle of the day, the trees and the fruit economy are game
//! rules and nothing here is measured.

use std::f32::consts::{PI, TAU};

use bevy::math::Vec3;

use crate::rng::Rng;

/// `ARENAS`: new arenas are only ever appended (a save state names its arena by index).
pub const ARENAS: [&str; 10] = ["room", "fan", "flypaper", "pool", "lamp", "escaperoom", "field", "orchard", "thermo", "kitchen"];
pub const OUTDOOR: [&str; 2] = ["field", "orchard"];

pub fn is_outdoor(arena: &str) -> bool {
    OUTDOOR.contains(&arena)
}

/// Room half-width, height and half-depth, m.
pub const RX: f32 = 4.2;
pub const RY: f32 = 3.0;
pub const RZ: f32 = 3.6;
/// Metres per 2D pixel.
pub const S: f32 = 0.006;
pub const STAND3: f32 = 64.0 * S;
pub const WATER3: f32 = 0.6;
pub const LAMP3: Vec3 = Vec3::new(0.0, 2.2, -0.8);
pub const FAN3: Vec3 = Vec3::new(-RX + 0.45, 0.0, 0.6);
/// Flypaper x0, x1, z0, z1.
pub const PAPER3: (f32, f32, f32, f32) = (-1.6, 1.6, -1.9, 1.1);
/// Thermo arena: full cold at x = -3.9 m, full hot at +3.9 m (game rule).
pub const THERMO_HALF3: f32 = RX - 0.3;
pub const WIND_FULL: f32 = 6.0;

#[derive(Debug, Clone, Copy)]
pub struct WorldSpec {
    pub name: &'static str,
    pub rx: f32,
    pub ry: f32,
    pub rz: f32,
    pub fly_rx: f32,
    pub fly_ry: f32,
    pub fly_rz: f32,
    /// A fly this far from you (horizontally) or this high is out of sight.
    pub lost_radius: Option<f32>,
    pub ground: [f32; 3],
    pub sky: [f32; 3],
}

pub const ROOM: WorldSpec = WorldSpec { name: "room", rx: 4.2, ry: 3.0, rz: 3.6, fly_rx: 4.2, fly_ry: 3.0, fly_rz: 3.6, lost_radius: None, ground: [0.24, 0.2, 0.17], sky: [0.42, 0.44, 0.5] };
pub const FIELD: WorldSpec =
    WorldSpec { name: "field", rx: 30.0, ry: 60.0, rz: 30.0, fly_rx: 200.0, fly_ry: 80.0, fly_rz: 200.0, lost_radius: Some(26.0), ground: [0.3, 0.33, 0.22], sky: [0.55, 0.66, 0.8] };
pub const ORCHARD: WorldSpec =
    WorldSpec { name: "orchard", rx: 16.0, ry: 40.0, rz: 16.0, fly_rx: 150.0, fly_ry: 60.0, fly_rz: 150.0, lost_radius: Some(22.0), ground: [0.3, 0.3, 0.2], sky: [0.55, 0.66, 0.8] };

pub fn spec(arena: &str) -> WorldSpec {
    match arena {
        "field" => FIELD,
        "orchard" => ORCHARD,
        _ => ROOM,
    }
}

/// An axis-aligned furniture box (min corner, max corner).
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl Aabb {
    pub fn contains(&self, p: Vec3, margin: f32) -> bool {
        p.cmpgt(self.lo - Vec3::splat(margin)).all() && p.cmplt(self.hi + Vec3::splat(margin)).all()
    }
}

/// The room's furniture: couch, bookshelf, plant.
pub fn room_colliders() -> Vec<Aabb> {
    vec![
        Aabb { lo: Vec3::new(-3.5, 0.0, -RZ), hi: Vec3::new(-1.1, 0.62, -RZ + 1.0) },
        Aabb { lo: Vec3::new(RX - 0.45, 0.0, -1.9), hi: Vec3::new(RX, 2.05, -0.1) },
        Aabb { lo: Vec3::new(RX - 1.0, 0.0, RZ - 1.0), hi: Vec3::new(RX - 0.2, 0.55, RZ - 0.2) },
    ]
}

// --- weather: wind and sun -----------------------------------------------------------------------------------------------------
/// The air's velocity in the world's x/z plane. `wind_dir_deg` is where the wind comes FROM (0 = +x, 90 = +z).
pub fn wind_vector(wind_dir_deg: f32, speed: f32) -> Vec3 {
    let a = wind_dir_deg.to_radians();
    -Vec3::new(a.cos(), 0.0, a.sin()) * speed
}

/// `(left antenna, right antenna)` drive, 0-1, for a fly facing `yaw` in a steady wind. GAME RULE transduction.
pub fn wind_drive(yaw: f32, wind_dir_deg: f32, speed: f32) -> (f32, f32) {
    let total = (speed / WIND_FULL).clamp(0.0, 1.0);
    let a = wind_dir_deg.to_radians();
    let side = (-yaw.sin(), yaw.cos());
    let sideness = a.cos() * side.0 + a.sin() * side.1;
    (total * (1.0 - sideness) / 2.0, total * (1.0 + sideness) / 2.0)
}

/// Unit vector pointing from the ground toward the sun.
pub fn sun_direction(az_deg: f32, el_deg: f32) -> Vec3 {
    let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
    Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin())
}

/// `(left eye, right eye)` photoreceptor drive from the sun. GAME RULE: brightness from elevation, up to a 35% split between the eyes.
pub fn sun_light(yaw: f32, az_deg: f32, el_deg: f32) -> (f32, f32) {
    let el = el_deg.to_radians();
    let total = if el > 0.0 { (0.15 + 0.85 * el.sin()).clamp(0.0, 1.0) } else { 0.0 };
    let az = az_deg.to_radians();
    let side = (-yaw.sin(), yaw.cos());
    let sideness = (az.cos() * side.0 + az.sin() * side.1) * el.cos();
    (total * (1.0 - 0.35 * sideness), total * (1.0 + 0.35 * sideness))
}

/// `(azimuth, elevation)` in degrees along an orbital day/night cycle: peak +60 at noon, nadir -30 at midnight. GAME RULE.
pub fn diurnal_cycle(time_s: f32, day_length_s: f32) -> (f32, f32) {
    let phase = (time_s % day_length_s) / day_length_s * TAU;
    ((phase.to_degrees() + 180.0) % 360.0, 45.0 * (phase - PI / 2.0).sin() + 15.0)
}

/// Day/night drive to the morning clock neurons l-LNv and s-LNv (GAME RULE): full at 60 degrees.
pub fn circadian_clock_drive(el_deg: f32) -> f32 {
    if el_deg > 0.0 { (el_deg / 60.0).clamp(0.0, 1.0) } else { 0.0 }
}

/// The fixed sun of the Lab (azimuth 135, elevation 45) or the day/night cycle when `day_s` > 0.
pub fn sun_now(t: f32, day_s: f32) -> (f32, f32) {
    if day_s > 0.0 { diurnal_cycle(t, day_s) } else { (135.0, 45.0) }
}

/// How much daylight the scene has, 0 (night, sun 10 degrees below the horizon) to 1 (sun 15 degrees up or higher).
pub fn dusk(el_deg: f32) -> f32 {
    ((el_deg + 10.0) / 25.0).clamp(0.0, 1.0)
}

/// Elevated dorsal fan-shaped body (FB6/FB7) activity puts the fly into quiet sleep. GAME RULE readout.
pub fn dfb_sleep_state(level: f32, threshold: f32) -> bool {
    level > threshold
}

// --- scenery -------------------------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
pub struct Rock {
    pub pos: Vec3,
    pub scale: Vec3,
    pub yaw: f32,
    pub shade: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tuft {
    pub pos: Vec3,
    pub height: f32,
    pub shade: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tree {
    pub pos: Vec3,
    pub height: f32,
    pub crown: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Scenery {
    pub rocks: Vec<Rock>,
    pub tufts: Vec<Tuft>,
    pub trees: Vec<Tree>,
}

/// Rocks and grass tufts (both arenas) and trees (orchard), placed from a fixed seed so every run matches.
pub fn scenery(arena: &str, seed: u64) -> Scenery {
    let mut out = Scenery::default();
    if !is_outdoor(arena) {
        return out;
    }
    let mut rng = Rng::new(seed);
    let w = spec(arena);
    let (n_rocks, n_tufts) = if arena == "field" { (26, 900) } else { (10, 600) };
    for _ in 0..n_rocks {
        let p = (rng.uniform(-w.rx + 2.0, w.rx - 2.0), rng.uniform(-w.rz + 2.0, w.rz - 2.0));
        if p.0.hypot(p.1) < 3.0 {
            continue; // keep the spawn point clear
        }
        let s = rng.uniform(0.15, 0.7);
        out.rocks.push(Rock {
            pos: Vec3::new(p.0, s * 0.35, p.1),
            scale: Vec3::new(s, s * rng.uniform(0.45, 0.8), s * rng.uniform(0.7, 1.2)),
            yaw: rng.uniform(0.0, PI),
            shade: rng.uniform(0.38, 0.52),
        });
    }
    for _ in 0..n_tufts {
        let p = (rng.uniform(-w.rx, w.rx), rng.uniform(-w.rz, w.rz));
        let h = rng.uniform(0.08, 0.22);
        out.tufts.push(Tuft { pos: Vec3::new(p.0, h / 2.0, p.1), height: h, shade: rng.uniform(0.75, 1.1) });
    }
    if arena == "orchard" {
        for gx in -2..=2 {
            for gz in -2..=2 {
                if gx == 0 && gz == 0 {
                    continue; // a clearing where you start
                }
                let j = (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8));
                out.trees.push(Tree { pos: Vec3::new(gx as f32 * 5.5 + j.0, 0.0, gz as f32 * 5.5 + j.1), height: rng.uniform(2.6, 3.4), crown: rng.uniform(1.3, 1.7) });
            }
        }
    }
    out
}

/// Trunks as boxes, so you and the walking fly go around them.
pub fn tree_colliders(trees: &[Tree]) -> Vec<Aabb> {
    trees.iter().map(|t| Aabb { lo: Vec3::new(t.pos.x - 0.18, 0.0, t.pos.z - 0.18), hi: Vec3::new(t.pos.x + 0.18, t.height, t.pos.z + 0.18) }).collect()
}

// --- the orchard's fruit -------------------------------------------------------------------------------------------------------
pub const DEFAULT_FEEDS: u32 = 4;
pub const DEFAULT_REGROW_S: f32 = 75.0;
pub const DEFAULT_CAP: usize = 4;
pub const SLOTS_PER_TREE: usize = 6;
/// How long one feed lasts; a fruit holds `feeds` of these (game rule).
pub const FEED_BOUT_S: f32 = 1.5;

#[derive(Debug, Clone)]
pub struct Fruit {
    pub tree: usize,
    pub pos: Vec3,
    pub feeds_max: u32,
    pub feeds_left: u32,
    pub fermented: bool,
    /// Set while the fruit is gone.
    pub regrow_at: Option<f32>,
    /// When it dropped, for the falling animation.
    pub fell_at: Option<f32>,
    /// The fly feeding on it right now, if any.
    pub eater: Option<u32>,
}

impl Fruit {
    pub fn ripe(&self) -> bool {
        self.feeds_left > 0
    }

    pub fn fullness(&self) -> f32 {
        self.feeds_left as f32 / self.feeds_max.max(1) as f32
    }
}

/// Every fruit slot on every tree, and the rules for eating it down and growing it back. All GAME RULES.
#[derive(Debug, Clone)]
pub struct Orchard {
    pub fruit: Vec<Fruit>,
    pub feeds: u32,
    pub regrow_s: f32,
    pub cap: usize,
    /// Share of new fruit that are fermented (fed from, they act exactly like the alcohol tool).
    pub ferment: f32,
    rng: Rng,
}

impl Orchard {
    pub fn new(trees: &[Tree], feeds: u32, regrow_s: f32, cap: usize, seed: u64) -> Self {
        let mut o = Self { fruit: Vec::new(), feeds, regrow_s, cap, ferment: 0.2, rng: Rng::new(seed) };
        for (ti, t) in trees.iter().enumerate() {
            let top = t.pos + Vec3::new(0.0, t.height - 0.3, 0.0);
            for k in 0..SLOTS_PER_TREE {
                let a = TAU * k as f32 / SLOTS_PER_TREE as f32 + o.rng.uniform(-0.3, 0.3);
                let r = t.crown * o.rng.uniform(0.55, 0.85);
                let pos = top + Vec3::new(r * a.cos(), -o.rng.uniform(0.35, 0.9), r * a.sin());
                o.fruit.push(Fruit { tree: ti, pos, feeds_max: feeds, feeds_left: 0, fermented: false, regrow_at: None, fell_at: None, eater: None });
            }
        }
        for ti in 0..trees.len() {
            // start with each tree at its cap, the rest dormant until the tree loses a fruit
            let mut slots: Vec<usize> = (0..o.fruit.len()).filter(|&i| o.fruit[i].tree == ti).collect();
            o.rng.shuffle(&mut slots);
            for (j, idx) in slots.into_iter().enumerate() {
                if j < cap {
                    o.grow(idx);
                }
            }
        }
        o
    }

    fn grow(&mut self, i: usize) {
        let fermented = self.rng.f32() < self.ferment;
        let f = &mut self.fruit[i];
        f.feeds_max = self.feeds;
        f.feeds_left = self.feeds;
        f.fermented = fermented;
        f.regrow_at = None;
        f.fell_at = None;
        f.eater = None;
    }

    pub fn on_tree(&self, tree: usize) -> usize {
        self.fruit.iter().filter(|f| f.tree == tree && f.ripe()).count()
    }

    /// Regrow what is due, respecting each tree's cap and at most one new fruit per tree per call. Returns the indices that grew.
    pub fn step(&mut self, now: f32) -> Vec<usize> {
        let mut due: Vec<usize> = (0..self.fruit.len()).filter(|&i| !self.fruit[i].ripe() && self.fruit[i].regrow_at.is_some()).collect();
        due.sort_by(|&a, &b| self.fruit[a].regrow_at.partial_cmp(&self.fruit[b].regrow_at).unwrap_or(std::cmp::Ordering::Equal));
        let (mut grown, mut busy) = (Vec::new(), Vec::new());
        for i in due {
            let (at, tree) = (self.fruit[i].regrow_at.unwrap_or(f32::MAX), self.fruit[i].tree);
            if at > now || busy.contains(&tree) {
                continue;
            }
            if self.on_tree(tree) >= self.cap {
                self.fruit[i].regrow_at = None; // full tree: go dormant until it loses a fruit
                continue;
            }
            self.grow(i);
            busy.push(tree);
            grown.push(i);
        }
        grown
    }

    /// One feeding bout finished on this fruit. Returns true if that emptied it (it drops and starts regrowing).
    pub fn feed(&mut self, i: usize, now: f32) -> bool {
        if !self.fruit[i].ripe() {
            return false;
        }
        self.fruit[i].feeds_left -= 1;
        if self.fruit[i].feeds_left > 0 {
            return false;
        }
        let tree = self.fruit[i].tree;
        self.fruit[i].fell_at = Some(now);
        self.fruit[i].eater = None;
        let wait = self.regrow_s * self.rng.uniform(0.75, 1.25);
        self.fruit[i].regrow_at = Some(now + wait);
        for j in 0..self.fruit.len() {
            if j != i && self.fruit[j].tree == tree && !self.fruit[j].ripe() && self.fruit[j].regrow_at.is_none() {
                let w = self.regrow_s * self.rng.uniform(0.75, 1.25);
                self.fruit[j].regrow_at = Some(now + w); // the tree has room again: its dormant slots start growing
            }
        }
        true
    }

    pub fn nearest_ripe(&self, pos: Vec3, exclude_taken: bool) -> Option<usize> {
        (0..self.fruit.len())
            .filter(|&i| self.fruit[i].ripe() && !(exclude_taken && self.fruit[i].eater.is_some()))
            .min_by(|&a, &b| (self.fruit[a].pos - pos).length().total_cmp(&(self.fruit[b].pos - pos).length()))
    }

    pub fn ripe_count(&self) -> usize {
        self.fruit.iter().filter(|f| f.ripe()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sun_follows_the_day_and_the_wind_hits_the_side_it_comes_from() {
        let (_, noon) = diurnal_cycle(300.0, 600.0);
        let (_, midnight) = diurnal_cycle(0.0, 600.0);
        assert!((noon - 60.0).abs() < 1e-3 && (midnight + 30.0).abs() < 1e-3);
        assert_eq!(dusk(-10.0), 0.0);
        assert_eq!(dusk(15.0), 1.0);
        // wind from +z (90 degrees) onto a fly facing +x (yaw 0): its right side is +z, so the right antenna gets more
        let (l, r) = wind_drive(0.0, 90.0, 6.0);
        assert!(r > l && (l + r - 1.0).abs() < 1e-5);
        let (l, r) = wind_drive(0.0, 0.0, 6.0);
        assert!((l - r).abs() < 1e-5, "a head-on wind drives both equally");
    }

    #[test]
    fn scenery_is_repeatable_and_the_orchard_has_a_clearing() {
        let a = scenery("orchard", 7);
        let b = scenery("orchard", 7);
        assert_eq!(a.trees.len(), 24);
        assert_eq!(a.trees[3].pos, b.trees[3].pos);
        assert!(a.trees.iter().all(|t| t.pos.x.hypot(t.pos.z) > 3.0), "the start is clear");
        assert!(scenery("room", 7).trees.is_empty());
    }

    #[test]
    fn fruit_is_eaten_down_drops_and_grows_back_within_the_cap() {
        let trees = scenery("orchard", 7).trees;
        let mut o = Orchard::new(&trees[..2], 2, 10.0, 4, 1);
        assert_eq!(o.on_tree(0), 4);
        let i = o.fruit.iter().position(|f| f.tree == 0 && f.ripe()).unwrap();
        assert!(!o.feed(i, 0.0));
        assert!(o.feed(i, 1.0), "the second feed empties it");
        assert_eq!(o.on_tree(0), 3);
        assert!(o.step(5.0).is_empty(), "not before 0.75 x regrow_s");
        let mut t = 5.0;
        while o.on_tree(0) < 4 && t < 40.0 {
            o.step(t);
            t += 1.0;
        }
        assert_eq!(o.on_tree(0), 4, "it grew back, and no further than the cap");
        for _ in 0..50 {
            o.step(t);
            t += 1.0;
        }
        assert!(o.on_tree(0) <= 4);
    }
}
