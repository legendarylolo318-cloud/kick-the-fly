//! The predators and the spider in the game: the adapter between the engine (`predators.rs`) and the world, and their bodies.
//! Python: `game/predator_play.py` (adapter + drawing) and the spider code of `Game3D._spider3d` / `draw_spider3d`.
//!
//! What it does each tick: builds the engine's targets from the live flies, steps every predator, and acts on what the engine reports. A
//! capture fires the fly's REAL touch neurons by body part, hurts it (55), and holds it at the predator's mouth; the predator then carries
//! it off and eats it (an immortal fly is let go instead). Detection is not here at all: the predators report `threats()` to
//! [`Threats`], so the existing looming transduction (LPLC2/LC4 -> DNp01) does the noticing.
//!
//! The bodies are procedural: ellipsoids and jointed limbs placed each frame from the engine's pose functions (`frog_pose`, `mantis_pose`,
//! `dragonfly_pose`, `SpiderCycle::pose`), with IK legs on planted feet for the mantis and the spider. The frog's hop reads as a hop
//! because its pose is: crouch (squash) with the hind legs folded, the push-off (stretch, hind legs extended), the arc (tuck, then the
//! hands reach for the ground), the landing (squash again), recover.

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;

use bevy::prelude::*;

use crate::anim::{LegRig, SpiderCycle, SpiderEvent, spider_rig};
use crate::app::SimLink;
use crate::arena::WorldColliders;
use crate::config::{Arena, Settings};
use crate::feedback::{Notes, Popup, Sfx, Tag};
use crate::fly::{Fly, FlyDamage, FlyPart, Part};
use crate::predators::{self, Bounds, Event, EventKind, Predator, State, Target};
use crate::rng::Rng;
use crate::senses::Threats;
use crate::sim_thread::Command;
use crate::tools::SpawnPredator;
use crate::world;

pub const CAPTURE_DAMAGE: f32 = 55.0;
pub const MAX_AT_ONCE: usize = 3;
const SPIDER_TOP: f32 = 3.0;

fn label(kind: &str) -> &'static str {
    match kind {
        "frog" => "A FROG!",
        "dragonfly" => "A DRAGONFLY!",
        _ => "A MANTIS!",
    }
}

/// A primitive of a procedural body.
#[derive(Debug, Clone, Copy)]
pub enum Prim {
    /// An ellipsoid: centre, the direction its long axis points, `(length, height, width)`, colour.
    Ell { c: Vec3, fwd: Vec3, size: Vec3, col: [f32; 3] },
    /// A jointed limb segment (a cylinder with a rounded joint at `b`).
    Limb { a: Vec3, b: Vec3, r: f32, col: [f32; 3] },
    /// A translucent wing (an ellipsoid drawn flat and see-through).
    Wing { c: Vec3, fwd: Vec3, size: Vec3 },
    /// A bright unlit bead (an eye highlight, a lure).
    Bead { c: Vec3, r: f32, col: [f32; 3] },
}

fn unit(v: Vec3) -> Vec3 {
    v.try_normalize().unwrap_or(Vec3::ZERO)
}

/// Basis columns `(forward, up, side)` as a rotation: a unit sphere scaled `(length, height, width)` becomes an ellipsoid along `fwd`.
fn basis(fwd: Vec3, up: Vec3) -> Quat {
    let f = unit(fwd);
    let mut side = f.cross(up);
    if side.length() < 1e-6 {
        side = Vec3::Z;
    }
    let side = unit(side);
    let u = unit(side.cross(f));
    Quat::from_mat3(&Mat3::from_cols(f, u, side))
}

struct Frame {
    base: Vec3,
    face: Vec3,
    side: Vec3,
    sc: f32,
}

impl Frame {
    fn at(&self, f: f32, u: f32, s: f32) -> Vec3 {
        self.base + (self.face * f + Vec3::Y * u + self.side * s) * self.sc
    }
}

const ART_FROG: f32 = 0.16;
const ART_MANTIS: f32 = 0.12;
const ART_DRAGONFLY: f32 = 0.10;

/// The frog: body, head, eyes, front arms, hind legs folded/extended/tucked, the tongue (`predator_play.draw3d`, frog branch).
pub fn layout_frog(p: &Predator) -> Vec<Prim> {
    let sc = p.spec.body_r / ART_FROG;
    let face = unit(Vec3::new(p.face.x, 0.0, p.face.z)).try_normalize().unwrap_or(Vec3::X);
    let fr = Frame { base: p.p, face, side: Vec3::new(-face.z, 0.0, face.x), sc };
    let (g1, g2, belly, dark) = ([0.30, 0.52, 0.18], [0.38, 0.62, 0.22], [0.86, 0.84, 0.62], [0.16, 0.30, 0.10]);
    let ps = p.frog_pose();
    let (sq, st, hind, tuck, reach, pitch) = (ps.squash, ps.stretch, ps.hind, ps.tuck, ps.reach, ps.pitch);
    let airborne = p.state == State::Airborne;
    let lerp = |a: Vec3, b: Vec3, k: f32| a + (b - a) * k;
    let tilt = unit(face * pitch.cos() + Vec3::Y * pitch.sin());
    let mouth = p.p + Vec3::Y * p.spec.body_r * 0.7 + face * p.spec.body_r * 0.8;
    let head = mouth - face * 0.072 * sc + Vec3::Y * (0.012 - 0.02 * sq) * sc;
    let body = head - tilt * (0.125 - 0.012 * sq) * sc - Vec3::Y * (0.012 + 0.012 * sq) * sc;
    let bsize =
        Vec3::new(0.15 * (1.0 - 0.10 * sq + 0.35 * st), 0.075 * (1.0 - 0.35 * sq - 0.25 * st), 0.12 * (1.0 + 0.18 * sq - 0.15 * st));
    let mut out = Vec::new();
    let e = |c: Vec3, f: Vec3, size: Vec3, col: [f32; 3]| Prim::Ell { c, fwd: f, size: size * sc, col };
    out.push(e(body, tilt, bsize, g1));
    out.push(e(body - Vec3::Y * 0.025 * sc + tilt * 0.01 * sc, tilt, Vec3::new(bsize.x * 0.87, bsize.y * 0.73, bsize.z * 0.88), belly));
    out.push(e(head, face, Vec3::new(0.08, 0.05, 0.105), g2));
    out.push(e(mouth - face * 0.05 * sc - Vec3::Y * 0.014 * sc, face, Vec3::new(0.06, 0.022, 0.092), belly));
    out.push(e(mouth - face * 0.012 * sc, face, Vec3::new(0.012, 0.004, 0.07), [0.25, 0.12, 0.12]));
    for k in -1..=1 {
        let kf = k as f32;
        out.push(e(
            body + Vec3::Y * 0.07 * bsize.y / 0.075 * sc + fr.side * 0.05 * kf * sc - tilt * 0.03 * kf.abs() * sc,
            tilt,
            Vec3::new(0.025, 0.012, 0.022),
            dark,
        ));
    }
    let gy = p.p.y + 0.012 * sc;
    for sg in [-1.0f32, 1.0] {
        let eye = head + Vec3::Y * 0.048 * sc + fr.side * 0.055 * sg * sc - face * 0.005 * sc;
        out.push(Prim::Bead { c: eye, r: 0.032 * sc, col: [0.85, 0.72, 0.25] });
        out.push(e(eye + face * 0.022 * sc + fr.side * 0.008 * sg * sc, face, Vec3::new(0.012, 0.012, 0.022), [0.04, 0.04, 0.04]));
        let shoulder = body + tilt * 0.07 * sc + fr.side * 0.08 * sg * sc - Vec3::Y * 0.02 * sc;
        let hand = if airborne {
            lerp(fr.at(0.07, -0.01, 0.09 * sg), fr.at(0.27, -0.05, 0.10 * sg), reach)
        } else {
            let h = fr.at(0.17 + 0.10 * reach, 0.0, 0.11 * sg);
            Vec3::new(h.x, gy, h.z)
        };
        let elbow = (shoulder + hand) / 2.0 + fr.side * 0.02 * sg * sc;
        out.push(Prim::Limb { a: shoulder, b: elbow, r: 0.016 * sc, col: g1 });
        out.push(Prim::Limb { a: elbow, b: hand, r: 0.013 * sc, col: g1 });
        out.push(e(hand, face + fr.side * 0.4 * sg, Vec3::new(0.03, 0.006, 0.025), g2));
        let hip = body - tilt * 0.09 * sc + fr.side * 0.08 * sg * sc;
        let folded = fr.at(0.06, 0.07 + 0.06 * sq, 0.17 * sg);
        let knee = lerp(lerp(folded, fr.at(-0.10, 0.06, 0.10 * sg), hind), fr.at(0.04, 0.07, 0.15 * sg), tuck);
        let ankle = lerp(lerp(fr.at(-0.12, 0.03, 0.14 * sg), fr.at(-0.32, 0.02, 0.06 * sg), hind), fr.at(-0.02, 0.05, 0.12 * sg), tuck);
        out.push(Prim::Limb { a: hip, b: knee, r: 0.03 * sc, col: g1 });
        out.push(Prim::Limb { a: knee, b: ankle, r: 0.022 * sc, col: g1 });
        let foot = if airborne {
            let f = ankle - face * 0.07 * sc * (1.0 - tuck) - Vec3::Y * 0.02 * sc;
            out.push(Prim::Limb { a: ankle, b: f, r: 0.015 * sc, col: g1 });
            f
        } else {
            let on_floor = Vec3::new(ankle.x, gy, ankle.z);
            out.push(Prim::Limb { a: ankle, b: on_floor, r: 0.015 * sc, col: g1 });
            on_floor + face * 0.06 * sc
        };
        out.push(e(foot, face + fr.side * 0.3 * sg, Vec3::new(0.075, 0.006, 0.04), g2));
    }
    if let Some(tip) = p.tip {
        out.push(Prim::Limb { a: p.mouth, b: tip, r: 0.03, col: [0.9, 0.35, 0.4] });
        out.push(Prim::Bead { c: tip, r: p.spec.tip_r, col: [0.95, 0.45, 0.5] });
    }
    out
}

/// The mantis: IK walking legs on planted feet, a long raised prothorax, a triangular head, raptorial forelegs that cock back for the
/// strike and fire out to the tip.
pub fn layout_mantis(p: &Predator, rig: &mut LegRig, dt: f32) -> Vec<Prim> {
    let sc = p.spec.body_r / ART_MANTIS;
    let face = unit(Vec3::new(p.face.x, 0.0, p.face.z)).try_normalize().unwrap_or(Vec3::X);
    let fr = Frame { base: p.p, face, side: Vec3::new(-face.z, 0.0, face.x), sc };
    let (col, dark) = ([0.46, 0.64, 0.22], [0.30, 0.46, 0.14]);
    let ps = p.mantis_pose();
    let (rear, cock, sway) = (ps.rear, ps.cock, ps.sway);
    let mut out = Vec::new();
    let mid = fr.at(-0.01, 0.075 - 0.02 * rear, 0.0) + fr.side * 0.004 * sway * sc;
    for l in rig.update(p.p, face, dt, ps.walking, Some(p.p.y), &[], 0.0) {
        out.push(Prim::Limb { a: l.hip, b: l.knee, r: 0.006 * sc, col: dark });
        out.push(Prim::Limb { a: l.knee, b: l.foot, r: 0.005 * sc, col: dark });
    }
    let e = |c: Vec3, f: Vec3, size: Vec3, col: [f32; 3]| Prim::Ell { c, fwd: f, size: size * sc, col };
    out.push(e(fr.at(-0.11, 0.085, 0.0), unit(face * 0.8 + Vec3::Y * 0.12), Vec3::new(0.10, 0.026, 0.03), col));
    out.push(Prim::Wing { c: fr.at(-0.09, 0.105, 0.0), fwd: face + Vec3::Y * 0.08, size: Vec3::new(0.11, 0.008, 0.032) * sc });
    out.push(e(mid, face, Vec3::new(0.03, 0.022, 0.022), col));
    let neck = fr.at(0.06 - 0.05 * rear, 0.17 + 0.025 * rear, 0.0) + fr.side * 0.006 * sway * sc;
    out.push(Prim::Limb { a: mid + Vec3::Y * 0.01 * sc, b: neck, r: 0.011 * sc, col });
    let head = neck + Vec3::Y * 0.022 * sc + face * 0.012 * sc;
    out.push(e(head, face, Vec3::new(0.016, 0.02, 0.03), col));
    out.push(e(head - Vec3::Y * 0.02 * sc + face * 0.006 * sc, face, Vec3::new(0.01, 0.012, 0.014), col));
    let striking = p.tip.is_some() && matches!(p.state, State::Strike | State::Retract | State::Eat);
    for sg in [-1.0f32, 1.0] {
        out.push(Prim::Bead { c: head + fr.side * 0.027 * sg * sc + Vec3::Y * 0.008 * sc, r: 0.012 * sc, col: [0.70, 0.78, 0.40] });
        let ant = head + Vec3::Y * 0.012 * sc + fr.side * 0.008 * sg * sc;
        out.push(Prim::Limb { a: ant, b: ant + (face * 0.05 + Vec3::Y * 0.06 + fr.side * 0.03 * sg) * sc, r: 0.0018 * sc, col: dark });
        let sh = neck - Vec3::Y * 0.02 * sc + fr.side * 0.015 * sg * sc;
        if striking {
            let tip = p.tip.unwrap_or(sh);
            let elbow = (sh + tip) / 2.0 + Vec3::Y * 0.03 * sc + fr.side * 0.01 * sg * sc;
            out.push(Prim::Limb { a: sh, b: elbow, r: 0.011 * sc, col });
            out.push(Prim::Limb { a: elbow, b: tip, r: 0.008 * sc, col: dark });
            out.push(Prim::Bead { c: tip, r: p.spec.tip_r, col: dark });
        } else {
            let knee = sh + (face * (0.035 - 0.02 * cock) - Vec3::Y * (0.04 - 0.07 * cock) + fr.side * 0.006 * sg) * sc;
            out.push(Prim::Limb { a: sh, b: knee, r: 0.011 * sc, col });
            let claw = knee + (Vec3::Y * 0.05 * (1.0 - cock) + face * 0.045 * cock - face * 0.005) * sc;
            out.push(Prim::Limb { a: knee, b: claw, r: 0.007 * sc, col: dark });
            for k in 0..3 {
                let spn = sh + (knee - sh) * (0.3 + 0.25 * k as f32);
                out.push(Prim::Limb { a: spn, b: spn + (face * 0.01 + Vec3::Y * 0.008) * sc, r: 0.002 * sc, col: dark });
            }
        }
    }
    out
}

/// The dragonfly: banded abdomen, two huge eyes, four wings that beat, legs thrown forward into a basket for the catch.
pub fn layout_dragonfly(p: &Predator, now: f32) -> Vec<Prim> {
    let sc = p.spec.body_r / ART_DRAGONFLY;
    let face0 = unit(Vec3::new(p.face.x, 0.0, p.face.z)).try_normalize().unwrap_or(Vec3::X);
    let dirn = if p.vel.length() > 1e-3 { unit(p.vel) } else { face0 };
    let flat = unit(Vec3::new(dirn.x, 0.0, dirn.z)).try_normalize().unwrap_or(face0);
    let sd = Vec3::new(-flat.z, 0.0, flat.x);
    let ps = p.dragonfly_pose();
    let bank = if p.state == State::Pursue { ps.bank } else { 0.0 };
    let sd_b = unit(sd * bank.cos() + Vec3::Y * bank.sin());
    let body = p.p;
    let mut out = Vec::new();
    out.push(Prim::Ell { c: body, fwd: dirn, size: Vec3::new(0.03, 0.022, 0.02) * sc, col: [0.20, 0.42, 0.30] });
    let headc = body + dirn * 0.035 * sc;
    for sg in [-1.0f32, 1.0] {
        out.push(Prim::Bead { c: headc + sd * 0.012 * sg * sc + Vec3::Y * 0.004 * sc, r: 0.017 * sc, col: [0.15, 0.55, 0.45] });
    }
    let ab0 = body - dirn * 0.03 * sc;
    for i in 0..6 {
        let a0 = ab0 - dirn * 0.03 * i as f32 * sc;
        out.push(Prim::Limb {
            a: a0,
            b: a0 - dirn * 0.03 * sc,
            r: 0.007 * (1.0 - 0.06 * i as f32) * sc,
            col: if i % 2 == 0 { [0.12, 0.40, 0.72] } else { [0.08, 0.22, 0.40] },
        });
    }
    let flap = 0.18 * (now * 55.0).sin();
    for sg in [-1.0f32, 1.0] {
        for (off, ln, wd) in [(0.008f32, 0.10f32, 0.017f32), (-0.012, 0.095, 0.021)] {
            let span = unit(sd_b * sg + Vec3::Y * if off > 0.0 { flap } else { -flap * 0.6 });
            let c = body + dirn * off * sc + span * ln * sc;
            out.push(Prim::Wing { c, fwd: span, size: Vec3::new(ln * sc, 0.0015 * sc, wd * sc) });
            out.push(Prim::Limb {
                a: body + dirn * off * sc,
                b: body + dirn * off * sc + span * 2.0 * ln * sc,
                r: 0.0012 * sc,
                col: [0.25, 0.30, 0.35],
            });
        }
        for k in 0..3 {
            let hip = body - Vec3::Y * 0.012 * sc + dirn * (0.01 - 0.01 * k as f32) * sc;
            let o = ps.reach;
            out.push(Prim::Limb {
                a: hip,
                b: hip + (dirn * (0.02 + 0.05 * o) - Vec3::Y * (0.02 + 0.01 * o) + sd * (0.01 + 0.012 * o) * sg) * sc,
                r: 0.0025 * sc,
                col: [0.1, 0.1, 0.1],
            });
        }
    }
    out
}

// --- the spider --------------------------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiderMode {
    Drop,
    Hunt,
    Leave,
    Carry,
}

pub struct Spider {
    pub p: Vec3,
    pub mode: SpiderMode,
    pub cycle: SpiderCycle,
    pub rig: LegRig,
    pub face: Vec3,
    pub bites: u32,
    pub target: Option<Entity>,
    anchor: Vec3,
}

pub fn layout_spider(s: &mut Spider, dt: f32) -> Vec<Prim> {
    let pose = s.cycle.pose();
    let face = unit(s.face).try_normalize().unwrap_or(Vec3::Z);
    let side = Vec3::new(-face.z, 0.0, face.x);
    let (cr, ra, lu) = (pose.crouch, pose.raise, pose.lunge);
    let hang = s.mode == SpiderMode::Drop || (s.p.y > 0.13 && matches!(s.mode, SpiderMode::Leave | SpiderMode::Carry));
    let body = s.p + face * 0.12 * lu - Vec3::Y * 0.035 * cr;
    let mut over = Vec::new();
    if ra > 0.0 {
        for i in [0usize, 4] {
            let sg = if i == 0 { -1.0 } else { 1.0 };
            over.push((i, body + face * (0.20 + 0.08 * ra) + Vec3::Y * (0.05 + 0.20 * ra) + side * 0.10 * sg));
        }
    }
    let moving = s.mode == SpiderMode::Hunt && lu == 0.0 && s.cycle.walking();
    let black = [0.12, 0.11, 0.13];
    let mut out = Vec::new();
    for l in s.rig.update(body, face, dt, moving, if hang { None } else { Some(0.0) }, &over, 0.1) {
        out.push(Prim::Limb { a: l.hip, b: l.knee, r: 0.008, col: black });
        out.push(Prim::Limb { a: l.knee, b: l.foot, r: 0.006, col: black });
    }
    let tilt = unit(face * (0.5 * ra).cos() + Vec3::Y * (0.5 * ra).sin());
    out.push(Prim::Ell { c: body - face * 0.02, fwd: tilt, size: Vec3::new(0.09, 0.06, 0.075), col: black });
    let head = body + face * 0.09 + Vec3::Y * 0.01 * (1.0 + ra);
    out.push(Prim::Ell { c: head, fwd: tilt, size: Vec3::new(0.05, 0.04, 0.045), col: black });
    for dx in [-0.015f32, 0.015] {
        out.push(Prim::Bead { c: head + face * 0.04 + Vec3::Y * 0.02 + side * dx, r: 0.008, col: [1.0, 0.25, 0.25] });
    }
    for sg in [-1.0f32, 1.0] {
        let fang = head + face * 0.04 - Vec3::Y * (0.02 + 0.02 * lu) + side * 0.012 * sg * (1.0 + lu);
        out.push(Prim::Limb { a: fang, b: fang + face * 0.02 - Vec3::Y * 0.03, r: 0.004, col: [0.85, 0.8, 0.7] });
    }
    out
}

// --- the world side -----------------------------------------------------------------------------------------------------------------
pub struct Live {
    pub engine: Predator,
    pub rig: LegRig,
    pub pool: PrimPool,
    pub id: u32,
}

#[derive(Default)]
pub struct PrimPool {
    ells: Vec<Entity>,
    cyls: Vec<Entity>,
    beads: Vec<Entity>,
    wings: Vec<Entity>,
    joints: Vec<Entity>,
}

#[derive(Resource, Default)]
pub struct Predators {
    pub list: Vec<Live>,
    pub spider: Option<(Spider, PrimPool)>,
    held: HashMap<Entity, u32>,
    next_id: u32,
    rng: Option<Rng>,
}

#[derive(Resource, Default)]
pub struct PaintBox {
    mats: HashMap<[u8; 3], Handle<StandardMaterial>>,
    unlit: HashMap<[u8; 3], Handle<StandardMaterial>>,
    wing: Option<Handle<StandardMaterial>>,
    sphere: Option<Handle<Mesh>>,
    cyl: Option<Handle<Mesh>>,
}

impl PaintBox {
    fn material(&mut self, materials: &mut Assets<StandardMaterial>, col: [f32; 3], unlit: bool) -> Handle<StandardMaterial> {
        let key = [(col[0] * 63.0) as u8, (col[1] * 63.0) as u8, (col[2] * 63.0) as u8];
        let map = if unlit { &mut self.unlit } else { &mut self.mats };
        map.entry(key)
            .or_insert_with(|| {
                materials.add(StandardMaterial {
                    base_color: Color::srgb(col[0], col[1], col[2]),
                    perceptual_roughness: 0.55,
                    unlit,
                    ..default()
                })
            })
            .clone()
    }
}

/// Spawn the entities a body needs and place them (one entity per primitive, pooled and reused; unused ones are hidden).
#[allow(clippy::too_many_arguments)]
pub fn draw_prims(
    commands: &mut Commands,
    pool: &mut PrimPool,
    prims: &[Prim],
    paint: &mut PaintBox,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    tfs: &mut Query<(&mut Transform, &mut Visibility, &mut MeshMaterial3d<StandardMaterial>), With<PrimTag>>,
) {
    let sphere = paint.sphere.get_or_insert_with(|| meshes.add(Sphere::new(1.0).mesh().uv(14, 9))).clone();
    let cyl = paint.cyl.get_or_insert_with(|| meshes.add(Cylinder::new(1.0, 1.0))).clone();
    let wing_mat = paint
        .wing
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::srgba(0.85, 0.92, 0.98, 0.45),
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                cull_mode: None,
                perceptual_roughness: 0.2,
                ..default()
            })
        })
        .clone();
    let (mut ne, mut nc, mut nb, mut nw, mut nj) = (0, 0, 0, 0, 0);
    let place = |commands: &mut Commands,
                 list: &mut Vec<Entity>,
                 idx: &mut usize,
                 mesh: &Handle<Mesh>,
                 mat: Handle<StandardMaterial>,
                 t: Transform,
                 tfs: &mut Query<(&mut Transform, &mut Visibility, &mut MeshMaterial3d<StandardMaterial>), With<PrimTag>>| {
        if *idx >= list.len() {
            let e = commands.spawn((PrimTag, Mesh3d(mesh.clone()), MeshMaterial3d(mat.clone()), t, Visibility::Inherited)).id();
            list.push(e);
        } else if let Ok((mut tr, mut vis, mut m)) = tfs.get_mut(list[*idx]) {
            *tr = t;
            *vis = Visibility::Inherited;
            if m.0 != mat {
                m.0 = mat;
            }
        }
        *idx += 1;
    };
    for pr in prims {
        match *pr {
            Prim::Ell { c, fwd, size, col } => {
                let m = paint.material(materials, col, false);
                place(
                    commands,
                    &mut pool.ells,
                    &mut ne,
                    &sphere,
                    m,
                    Transform { translation: c, rotation: basis(fwd, Vec3::Y), scale: size.max(Vec3::splat(1e-4)) },
                    tfs,
                );
            }
            Prim::Limb { a, b, r, col } => {
                let d = b - a;
                let len = d.length().max(1e-4);
                let m = paint.material(materials, col, false);
                place(
                    commands,
                    &mut pool.cyls,
                    &mut nc,
                    &cyl,
                    m.clone(),
                    Transform {
                        translation: (a + b) / 2.0,
                        rotation: Quat::from_rotation_arc(Vec3::Y, d / len),
                        scale: Vec3::new(r, len, r),
                    },
                    tfs,
                );
                place(
                    commands,
                    &mut pool.joints,
                    &mut nj,
                    &sphere,
                    m,
                    Transform { translation: b, rotation: Quat::IDENTITY, scale: Vec3::splat(r) },
                    tfs,
                );
            }
            Prim::Wing { c, fwd, size } => place(
                commands,
                &mut pool.wings,
                &mut nw,
                &sphere,
                wing_mat.clone(),
                Transform { translation: c, rotation: basis(fwd, Vec3::Y), scale: size.max(Vec3::splat(1e-4)) },
                tfs,
            ),
            Prim::Bead { c, r, col } => {
                let m = paint.material(materials, col, true);
                place(
                    commands,
                    &mut pool.beads,
                    &mut nb,
                    &sphere,
                    m,
                    Transform { translation: c, rotation: Quat::IDENTITY, scale: Vec3::splat(r) },
                    tfs,
                );
            }
        }
    }
    for (list, used) in [(&pool.ells, ne), (&pool.cyls, nc), (&pool.beads, nb), (&pool.wings, nw), (&pool.joints, nj)] {
        for e in list.iter().skip(used) {
            if let Ok((_, mut v, _)) = tfs.get_mut(*e) {
                *v = Visibility::Hidden;
            }
        }
    }
}

#[derive(Component)]
pub struct PrimTag;

fn bounds_for(arena: &str) -> Bounds {
    let w = world::spec(arena);
    let ymax = if w.lost_radius.is_none() { 3.0 } else { 8.0 };
    Bounds { xmin: -w.rx, xmax: w.rx, ymin: 0.0, ymax, zmin: -w.rz, zmax: w.rz }
}

/// New predators and spiders asked for by the tools.
#[allow(clippy::too_many_arguments)]
fn spawn(
    mut reqs: MessageReader<SpawnPredator>,
    mut preds: ResMut<Predators>,
    arena: Res<Arena>,
    flies: Query<&Fly>,
    mut popups: MessageWriter<Popup>,
) {
    for r in reqs.read() {
        if !flies.iter().any(|f| !f.dead() && !f.gone()) {
            continue;
        }
        if r.kind == "spider" {
            if preds.spider.is_some() {
                continue;
            }
            let top = SPIDER_TOP.min(world::spec(arena.name()).ry);
            let at = Vec3::new(r.at.x, top - 0.05, r.at.z);
            preds.spider = Some((
                Spider {
                    p: at,
                    mode: SpiderMode::Drop,
                    cycle: SpiderCycle::default(),
                    rig: spider_rig(1.0),
                    face: Vec3::Z,
                    bites: 0,
                    target: None,
                    anchor: r.at,
                },
                PrimPool::default(),
            ));
            popups.write(Popup::new(at, "A SPIDER!", (200, 220, 160)));
            continue;
        }
        if preds.list.iter().any(|p| p.engine.kind == r.kind) || preds.list.len() >= MAX_AT_ONCE {
            continue;
        }
        let b = bounds_for(arena.name());
        let mut at = r.at;
        if r.kind == "dragonfly" {
            at.y = b.ymax - 0.35;
        }
        let seed = preds.rng.get_or_insert_with(|| Rng::new(7)).next_u64();
        let Ok(engine) = Predator::new(r.kind, b.clip(at, 0.25), Rng::new(seed), b) else { continue };
        let (rig, id) = (mantis_rig(&engine), preds.next_id);
        preds.next_id += 1;
        popups.write(Popup::new(engine.p + Vec3::Y * 0.3, label(r.kind), (200, 220, 160)));
        preds.list.push(Live { engine, rig, pool: PrimPool::default(), id });
    }
}

fn mantis_rig(p: &Predator) -> LegRig {
    let sc = p.spec.body_r / ART_MANTIS;
    let hips: Vec<Vec3> =
        [-1.0f32, 1.0].iter().flat_map(|sg| [0.0f32, -0.035].map(|f| Vec3::new(f * sc, 0.075 * sc, 0.018 * sg * sc))).collect();
    let rests: Vec<Vec3> = [-1.0f32, 1.0].iter().flat_map(|sg| [0.03f32, -0.06].map(|f| Vec3::new(f * sc, 0.0, 0.09 * sg * sc))).collect();
    LegRig::new(hips, rests, 0.07 * sc, 0.08 * sc, vec![0, 1, 1, 0], 0.035 * sc, 0.30, 0.02 * sc)
}

/// Step every predator against the live flies and act on what it reports.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn step(
    fixed: Res<Time<Fixed>>,
    virt: Res<Time<Virtual>>,
    settings: Res<Settings>,
    mut preds: ResMut<Predators>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<&Transform, With<FlyPart>>,
    link: Res<SimLink>,
    mut dmg: MessageWriter<FlyDamage>,
    mut popups: MessageWriter<Popup>,
    mut sfx: MessageWriter<Sfx>,
    mut notes: ResMut<Notes>,
    mut threats: ResMut<Threats>,
) {
    let now = virt.elapsed_secs();
    let dt = fixed.delta_secs();
    let preds = &mut *preds;
    let held_ids: Vec<Entity> = preds.held.keys().copied().collect();
    let targets: Vec<(Entity, Target)> = flies
        .iter()
        .map(|(e, f)| {
            let p = parts.get(f.parts[Part::Thorax.index()]).map(|t| t.translation).unwrap_or(Vec3::ZERO);
            (
                e,
                Target {
                    id: (e.to_bits() & 0xFFFF_FFFF) as u32,
                    p,
                    flying: f.flying_now,
                    alive: !f.dead() && !f.gone() && !held_ids.contains(&e),
                },
            )
        })
        .collect();
    let eng_targets: Vec<Target> = targets.iter().map(|t| t.1).collect();
    let mut finished: Vec<u32> = Vec::new();
    let mut captured: Vec<(u32, Entity, Vec3, &'static str)> = Vec::new();
    for live in &mut preds.list {
        for ev in live.engine.step(dt, &eng_targets) {
            handle(&ev, live, &targets, &mut captured, &mut finished, &mut sfx);
        }
        for (key, pos, r) in live.engine.threats() {
            threats.push(key, pos, r);
        }
    }
    for (id, e, at, kind) in captured {
        let Ok((_, mut f)) = flies.get_mut(e) else { continue };
        let Some(live) = preds.list.iter().find(|l| l.id == id) else { continue };
        for (region, side, strength) in predators::capture_touch(kind) {
            link.0.send(Command::PokeFly { slot: f.slot, region: (*region).into(), side: *side, strength: *strength });
        }
        f.last_hit = at;
        f.hurt = 1.0;
        dmg.write(FlyDamage { fly: e, amount: CAPTURE_DAMAGE, source: kind });
        sfx.write(Sfx::new("chomp"));
        let words: &[&str] = match kind {
            "frog" => &["SNAP!", "ZAP!", "GULP!"],
            "dragonfly" => &["GRABBED!", "SNATCH!"],
            _ => &["STRIKE!", "SNAP!"],
        };
        popups.write(Popup::new(live.engine.p + Vec3::Y * 0.3, words[f.rng.index(words.len())], (230, 120, 120)));
        f.grabbed = Some(Part::Thorax);
        preds.held.insert(e, id);
        notes.push(now, format!("CAUGHT   by a {kind}"), Tag::GameRule);
    }
    for id in finished {
        let victims: Vec<Entity> = preds.held.iter().filter(|(_, h)| **h == id).map(|(e, _)| *e).collect();
        for e in victims {
            let kind = preds.list.iter().find(|l| l.id == id).map_or("predator", |l| l.engine.kind);
            if settings.immortal {
                release(preds, e, &mut flies);
                popups.write(Popup::new(
                    parts
                        .get(flies.get(e).map(|(_, f)| f.parts[1]).unwrap_or(Entity::PLACEHOLDER))
                        .map(|t| t.translation)
                        .unwrap_or(Vec3::ZERO)
                        + Vec3::Y * 0.4,
                    "BROKE FREE!",
                    (255, 225, 120),
                ));
            } else {
                dmg.write(FlyDamage { fly: e, amount: 1000.0, source: kind_static(kind) });
                release(preds, e, &mut flies);
            }
        }
    }
    // a held fly stays at the mouth until let go; release anything whose holder is gone
    let alive_ids: Vec<u32> = preds.list.iter().map(|l| l.id).collect();
    let stale: Vec<Entity> = preds.held.iter().filter(|(_, id)| !alive_ids.contains(id)).map(|(e, _)| *e).collect();
    for e in stale {
        release(preds, e, &mut flies);
    }
    for (e, id) in preds.held.clone() {
        let Some(live) = preds.list.iter().find(|l| l.id == id) else { continue };
        if let Ok((_, mut f)) = flies.get_mut(e)
            && !f.wrapped
        {
            f.grabbed = Some(Part::Thorax);
            f.hold = live.engine.grab_point();
        }
    }
    preds.list.retain(|l| !l.engine.done);
}

fn kind_static(kind: &str) -> &'static str {
    predators::spec(kind).map_or("a predator", |s| s.label)
}

fn handle(
    ev: &Event,
    live: &Live,
    targets: &[(Entity, Target)],
    captured: &mut Vec<(u32, Entity, Vec3, &'static str)>,
    finished: &mut Vec<u32>,
    sfx: &mut MessageWriter<Sfx>,
) {
    match ev.kind {
        EventKind::Capture => {
            if let Some(id) = ev.target
                && let Some((e, _)) = targets.iter().find(|(_, t)| t.id == id)
            {
                captured.push((live.id, *e, ev.at.unwrap_or(live.engine.p), live.engine.kind));
            }
        }
        EventKind::Leave => finished.push(live.id),
        EventKind::Strike => {
            sfx.write(Sfx::vol("whack", 0.5));
        }
        _ => {}
    }
}

fn release(preds: &mut Predators, e: Entity, flies: &mut Query<(Entity, &mut Fly)>) {
    preds.held.remove(&e);
    if let Ok((_, mut f)) = flies.get_mut(e)
        && f.grabbed.is_some()
        && !f.wrapped
    {
        f.grabbed = None;
    }
}

/// The spider (`_spider3d`): drops on its thread, hunts the nearest fly through its walk/windup/strike/recover cycle, bites (venom, touch
/// neurons, 16 damage), wraps a twice-bitten fly in silk and carries it up.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn spider_step(
    fixed: Res<Time<Fixed>>,
    virt: Res<Time<Virtual>>,
    settings: Res<Settings>,
    arena: Res<Arena>,
    boxes: Res<WorldColliders>,
    mut preds: ResMut<Predators>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<&Transform, With<FlyPart>>,
    link: Res<SimLink>,
    mut dmg: MessageWriter<FlyDamage>,
    mut popups: MessageWriter<Popup>,
    mut sfx: MessageWriter<Sfx>,
    mut notes: ResMut<Notes>,
    mut threats: ResMut<Threats>,
) {
    let (now, dt) = (virt.elapsed_secs(), fixed.delta_secs());
    let _ = &boxes;
    let top = SPIDER_TOP.min(world::spec(arena.name()).ry);
    let Some((sp, _)) = preds.spider.as_mut() else { return };
    if matches!(sp.mode, SpiderMode::Drop | SpiderMode::Hunt) {
        threats.push("spider", sp.p, 0.1);
    }
    match sp.mode {
        SpiderMode::Drop => {
            sp.p.y -= 0.04 * 60.0 * dt;
            if sp.p.y <= 0.12 {
                sp.p.y = 0.12;
                sp.mode = SpiderMode::Hunt;
            }
        }
        SpiderMode::Hunt => {
            let cands: Vec<(Entity, Vec3)> = flies
                .iter()
                .filter(|(_, f)| !f.dead() && !f.gone())
                .filter_map(|(e, f)| parts.get(f.parts[Part::Thorax.index()]).ok().map(|t| (e, t.translation)))
                .collect();
            let Some((e, thx)) = cands.iter().min_by(|a, b| (a.1 - sp.p).length().total_cmp(&(b.1 - sp.p).length())).copied() else {
                sp.mode = SpiderMode::Leave;
                return;
            };
            let d = thx + Vec3::Y * 0.1 - sp.p;
            let dist = d.length();
            if dist > 1e-6 && matches!(sp.cycle.state, crate::anim::SpiderState::Walk | crate::anim::SpiderState::Windup) {
                sp.face = unit(Vec3::new(d.x, 0.0, d.z));
            }
            let ev = sp.cycle.step(dt, dist, 0.2);
            if sp.cycle.walking() && dist > 0.2 {
                sp.p += d / dist * (3.4 * world::S * 60.0 * dt).min(dist);
                sp.p.y = sp.p.y.max(0.12);
                sp.anchor = sp.p;
            }
            if ev == Some(SpiderEvent::Bite) {
                sp.bites += 1;
                sp.target = Some(e);
                if let Ok((_, mut f)) = flies.get_mut(e) {
                    f.venom = (f.venom + 0.3).min(1.0);
                    f.last_hit = sp.p;
                    f.hurt = 1.0;
                    link.0.send(Command::PokeFly { slot: f.slot, region: "body".into(), side: None, strength: 0.9 });
                    link.0.send(Command::PokeFly { slot: f.slot, region: "legs".into(), side: Some('L'), strength: 0.6 });
                    link.0.send(Command::PokeFly { slot: f.slot, region: "legs".into(), side: Some('R'), strength: 0.6 });
                    dmg.write(FlyDamage { fly: e, amount: 16.0, source: "a spider" });
                    sfx.write(Sfx::new("chomp"));
                    popups.write(Popup::new(sp.p + Vec3::Y * 0.3, ["CHOMP!", "BITE!", "SLURP!"][f.rng.index(3)], (230, 120, 120)));
                    if sp.bites >= 2 && !f.wrapped {
                        f.wrapped = true;
                        notes.push(now, "WRAPPED  in spider silk", Tag::GameRule);
                    }
                    if settings.immortal && sp.bites >= 5 {
                        f.wrapped = false;
                        f.grabbed = None;
                        sp.mode = SpiderMode::Leave;
                        popups.write(Popup::new(thx + Vec3::Y * 0.4, "BROKE FREE!", (255, 225, 120)));
                        notes.push(now, "BROKE FREE of the silk", Tag::GameRule);
                        return;
                    }
                }
            }
            if let Ok((_, mut f)) = flies.get_mut(e)
                && f.wrapped
            {
                f.grabbed = Some(Part::Thorax);
                f.hold = sp.p - Vec3::Y * 0.16;
            }
        }
        SpiderMode::Leave | SpiderMode::Carry => {
            let wrapped = sp.target.filter(|e| flies.get(*e).is_ok_and(|(_, f)| f.wrapped));
            if let Some(e) = wrapped
                && flies.get(e).is_ok_and(|(_, f)| f.dead() && f.frozen_at.is_none())
            {
                sp.mode = SpiderMode::Carry;
            }
            sp.p.y += 2.2 * world::S * 2.0 * 60.0 * dt;
            if sp.mode == SpiderMode::Carry
                && let Some(e) = wrapped
                && let Ok((_, mut f)) = flies.get_mut(e)
            {
                f.grabbed = Some(Part::Thorax);
                f.hold = sp.p - Vec3::Y * 0.16;
            }
            if sp.p.y > top + 0.4 {
                if let Some(e) = wrapped
                    && let Ok((_, mut f)) = flies.get_mut(e)
                {
                    f.grabbed = None;
                }
                preds.spider = None;
            }
        }
    }
}

/// Place every body's primitives for this frame.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn render(
    mut commands: Commands,
    time: Res<Time>,
    virt: Res<Time<Virtual>>,
    mut preds: ResMut<Predators>,
    mut paint: ResMut<PaintBox>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut tfs: Query<(&mut Transform, &mut Visibility, &mut MeshMaterial3d<StandardMaterial>), With<PrimTag>>,
) {
    let now = virt.elapsed_secs();
    let dt = time.delta_secs();
    let preds = &mut *preds;
    for live in &mut preds.list {
        let prims = match live.engine.kind {
            "frog" => layout_frog(&live.engine),
            "mantis" => layout_mantis(&live.engine, &mut live.rig, dt),
            _ => layout_dragonfly(&live.engine, now),
        };
        draw_prims(&mut commands, &mut live.pool, &prims, &mut paint, &mut meshes, &mut materials, &mut tfs);
    }
    if let Some((sp, pool)) = preds.spider.as_mut() {
        let prims = layout_spider(sp, dt);
        draw_prims(&mut commands, pool, &prims, &mut paint, &mut meshes, &mut materials, &mut tfs);
    }
}

/// Retire the pooled entities of predators that have gone.
fn cleanup(mut commands: Commands, preds: Res<Predators>, tagged: Query<Entity, With<PrimTag>>, mut known: Local<Vec<Entity>>) {
    if !preds.is_changed() {
        return;
    }
    let mut live: Vec<Entity> = Vec::new();
    for l in &preds.list {
        for v in [&l.pool.ells, &l.pool.cyls, &l.pool.beads, &l.pool.wings, &l.pool.joints] {
            live.extend(v.iter().copied());
        }
    }
    if let Some((_, p)) = &preds.spider {
        for v in [&p.ells, &p.cyls, &p.beads, &p.wings, &p.joints] {
            live.extend(v.iter().copied());
        }
    }
    for e in &tagged {
        if !live.contains(&e) && known.contains(&e) {
            commands.entity(e).despawn();
        }
    }
    *known = live;
}

pub struct PredatorPlugin;

impl Plugin for PredatorPlugin {
    fn build(&self, app: &mut App) {
        let _ = FRAC_PI_2;
        app.init_resource::<Predators>()
            .init_resource::<PaintBox>()
            .add_systems(FixedUpdate, (spawn, step, spider_step).chain().before(crate::fly::control))
            .add_systems(Update, (render, cleanup).chain());
    }
}
