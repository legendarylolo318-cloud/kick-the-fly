//! How the fly looks: Python's `Game3D._draw_fly` shape for shape. A fruit fly, not a house fly: a pale yellow-tan body, a short abdomen
//! with a dark band across the top of each segment and a dark tapered tip, the scutellum behind the wing bases, red eyes that fill most
//! of the head, short antennae, bristles, clear wings folded along the back and two-tone legs with knee and foot joints.
//! Six IK legs walk on a tripod gait with planted feet, the wings flap in flight, and burns, frost, melt, death and pain re-shade every
//! colour as `k2.shade` does. The legs and wings do not collide; they follow the physics bodies.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::anim::{LegRig, two_bone_ik};
use crate::fly::{Fly, FlyLook, FlyPart, Part};
use crate::world::{S, STAND3};

/// Thigh and shin length: the front leg's in Python's rest pose (`REST3`: hip to knee 0.21, knee to foot 0.27).
const LEG_L1: f32 = 0.21;
const LEG_L2: f32 = 0.27;
/// Thigh and shin radius, knee and foot bead radius (`_draw_fly`: 2.0, 1.3, 2.1, 1.8 times S).
const LEG_R: [f32; 2] = [2.0 * S, 1.3 * S];
const BEAD_R: [f32; 2] = [2.1 * S, 1.8 * S];

// the healthy colours of `_draw_fly`, in the order of `FlyLook.tinted`
const ABDOMEN: usize = 0;
const TIP: usize = 1;
const THORAX: usize = 2;
const SCUTELLUM: usize = 3;
const HEAD: usize = 4;
const EYE: usize = 5;
const LEG: usize = 6;
const FOOT: usize = 7;
const DARK: usize = 8;
const BRISTLE: usize = 9;
const PROBOSCIS: usize = 10;
const COLOURS: [[f32; 3]; 11] = [
    [206.0, 172.0, 104.0],
    [96.0, 70.0, 38.0],
    [176.0, 146.0, 92.0],
    [150.0, 120.0, 74.0],
    [186.0, 154.0, 96.0],
    [222.0, 44.0, 30.0],
    [132.0, 95.0, 50.0],
    [96.0, 66.0, 32.0],
    [70.0, 45.0, 20.0],
    [74.0, 52.0, 28.0],
    [128.0, 92.0, 56.0],
];

/// The body axis (abdomen to head) rises toward the head by this much in the rest pose (`REST`: head (42, 6), abdomen (-50, -6)), and
/// the abdomen's own axis (abdomen to thorax) by a little less. Python builds its frames from those axes; the rigid parts here keep them.
fn body_tilt() -> Quat {
    Quat::from_rotation_z((12.0f32).atan2(92.0))
}
fn abdomen_tilt() -> Quat {
    Quat::from_rotation_z((6.0f32).atan2(50.0))
}

/// `k2.shade`: one healthy colour (0-255) as the fly's state leaves it; returns rgb (0-255) and alpha.
pub fn shade(col: [f32; 3], hurt: f32, dead: bool, char_: f32, melt: f32, frost: f32) -> ([f32; 3], f32) {
    let mut c = col;
    if char_ != 0.0 {
        c = c.map(|v| v * (1.0 - 0.75 * char_) + 22.0 * char_);
    }
    if frost != 0.0 {
        let t = [175.0, 215.0, 245.0];
        c = std::array::from_fn(|i| c[i] * (1.0 - 0.7 * frost) + t[i] * 0.7 * frost);
    }
    if dead {
        let g = (c[0] + c[1] + c[2]) / 3.0;
        c = c.map(|v| 0.45 * v + 0.55 * g * 0.8);
    }
    if melt != 0.0 {
        let t = [150.0, 160.0, 115.0];
        c = std::array::from_fn(|i| c[i] * (1.0 - 0.6 * melt) + t[i] * 0.6 * melt);
    }
    (c.map(|v| (v + (255.0 - v) * hurt * 0.7).min(255.0).floor()), if melt != 0.0 { 1.0 - 0.6 * melt } else { 1.0 })
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The abdomen's pattern (`render3d.py`, pattern 4) at a point of the unit sphere, as a factor on each colour channel: a dark band at
/// the back of each tergite, on top only, stronger toward the tip, and a paler, yellower underside.
pub fn stripes(p: Vec3) -> [f32; 3] {
    let seg = (p.x * 2.4 + 0.5).rem_euclid(1.0);
    let band = smoothstep(0.42, 0.04, seg);
    let dorsal = smoothstep(-0.12, 0.40, p.y);
    let rear = (0.4 + 0.6 * smoothstep(0.75, -0.05, p.x)) * smoothstep(-0.92, -0.45, p.x);
    let dark = 1.0 - 0.7 * band * dorsal * rear;
    let under = (1.0 - dorsal) * 0.45;
    [1.10, 1.06, 0.82].map(|c: f32| dark * (1.0 + (c - 1.0) * under))
}

/// The eye's shading (pattern 5) without its facets: darker toward the back.
fn eye_shade(p: Vec3) -> [f32; 3] {
    [0.95 * (1.0 - 0.25 * smoothstep(0.2, 1.0, -p.x)); 3]
}

/// A unit sphere whose rings run around the x axis (so a pattern along the body stays sharp), each vertex coloured by `paint`.
/// Python's shader multiplies colours as they are shown, Bevy multiplies vertex colours in linear light: hence the power.
fn painted_sphere(rings: usize, around: usize, paint: impl Fn(Vec3) -> [f32; 3]) -> Mesh {
    let (mut pos, mut nor, mut col, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..=rings {
        let t = std::f32::consts::PI * i as f32 / rings as f32;
        for j in 0..=around {
            let a = std::f32::consts::TAU * j as f32 / around as f32;
            let p = Vec3::new(-t.cos(), t.sin() * a.cos(), t.sin() * a.sin());
            pos.push(p.to_array());
            nor.push(p.to_array());
            let c = paint(p).map(|v| v.max(0.0).powf(2.2));
            col.push([c[0], c[1], c[2], 1.0]);
        }
    }
    let w = (around + 1) as u32;
    for i in 0..rings as u32 {
        for j in 0..around as u32 {
            let (a, b, c, d) = (i * w + j, i * w + j + 1, (i + 1) * w + j + 1, (i + 1) * w + j);
            idx.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nor)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, col)
        .with_inserted_indices(Indices::U32(idx))
}

/// The meshes a fly's body is made of.
pub struct BodyMeshes {
    sphere: Handle<Mesh>,
    abdomen: Handle<Mesh>,
    eye: Handle<Mesh>,
    cylinder: Handle<Mesh>,
    cube: Handle<Mesh>,
}

impl BodyMeshes {
    pub fn new(meshes: &mut Assets<Mesh>) -> Self {
        Self {
            sphere: meshes.add(Sphere::new(1.0).mesh().uv(22, 14)),
            abdomen: meshes.add(painted_sphere(96, 28, stripes)),
            eye: meshes.add(painted_sphere(20, 16, eye_shade)),
            cylinder: meshes.add(Cylinder::new(1.0, 1.0).mesh().resolution(10)),
            cube: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        }
    }
}

/// The materials of one fly (its own, because each fly burns, freezes and bruises by itself).
pub fn build_look(materials: &mut Assets<StandardMaterial>) -> FlyLook {
    let tinted = COLOURS
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let m = materials.add(StandardMaterial {
                base_color: Color::srgb(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0),
                // the eye is glossy (the shader's `spec_k = 1.1, shin = 90`), the cuticle only a little
                perceptual_roughness: if i == EYE { 0.18 } else { 0.55 },
                reflectance: if i == EYE { 0.7 } else { 0.35 },
                ..default()
            });
            (m, *c)
        })
        .collect();
    let wing = materials.add(StandardMaterial {
        base_color: Color::srgba(0.90, 0.94, 0.99, 0.34),
        emissive: LinearRgba::rgb(0.90, 0.94, 0.99) * 0.12,
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.1,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let glint = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.85, 0.8),
        emissive: LinearRgba::rgb(1.0, 0.85, 0.8) * 0.6,
        ..default()
    });
    let cross = materials.add(StandardMaterial { base_color: Color::srgb(0.12, 0.08, 0.08), ..default() });
    FlyLook { tinted, wing, glint, cross }
}

/// On a living fly's eye.
#[derive(Component)]
struct EyeGlint;
/// On a dead fly's eye.
#[derive(Component)]
struct EyeCross;
/// Bristles go when the spider has wrapped the fly.
#[derive(Component)]
struct Bristle;
/// Which fly a marked shape belongs to.
#[derive(Component)]
struct OfFly(Entity);

/// The shapes that ride on one rigid part, in Python's numbers: positions and sizes in units of `S` along (forward, up, side).
pub fn dress(c: &mut ChildSpawnerCommands, part: Part, look: &FlyLook, m: &BodyMeshes) {
    let fly = c.target_entity();
    let mat = |i: usize| MeshMaterial3d(look.tinted[i].0.clone());
    let v = |f: f32, u: f32, s: f32| Vec3::new(f, u, s) * S;
    // an ellipsoid at `at` with half-axes `size`, both in the frame `frame`
    let ell = |frame: Quat, at: Vec3, size: Vec3| Transform { translation: frame * at, rotation: frame, scale: size };
    // a rod from `a` to `b`
    let rod = |frame: Quat, a: Vec3, b: Vec3, r: f32| {
        let d = b - a;
        Transform {
            translation: frame * ((a + b) / 2.0),
            rotation: frame * Quat::from_rotation_arc(Vec3::Y, d.normalize_or(Vec3::Y)),
            scale: Vec3::new(r * S, d.length(), r * S),
        }
    };
    match part {
        Part::Abdomen => {
            let f = abdomen_tilt();
            let (l, r) = (24.0, 17.5);
            c.spawn((Mesh3d(m.abdomen.clone()), mat(ABDOMEN), ell(f, Vec3::ZERO, v(l, r, r))));
            // the tapered, dark tip
            c.spawn((Mesh3d(m.sphere.clone()), mat(TIP), ell(f, v(-l * 0.52, 0.0, 0.0), v(l * 0.42, r * 0.78, r * 0.78))));
        }
        Part::Thorax => {
            let f = body_tilt();
            c.spawn((Mesh3d(m.sphere.clone()), mat(THORAX), ell(f, Vec3::ZERO, v(23.0, 19.5, 19.0))));
            // the scutellum behind the wing bases
            c.spawn((Mesh3d(m.sphere.clone()), mat(SCUTELLUM), ell(f, v(-6.0, 9.0, 0.0), v(13.0, 7.0, 15.0))));
            for (bx, by, bs, bl) in [
                (-2.0, 15.0, 0.0, 7.0),
                (4.0, 15.0, 5.0, 6.0),
                (4.0, 15.0, -5.0, 6.0),
                (-8.0, 13.0, 7.0, 7.0),
                (-8.0, 13.0, -7.0, 7.0),
                (-13.0, 9.0, 0.0, 8.0),
            ] {
                let root = v(bx, by, bs);
                c.spawn((Bristle, OfFly(fly), Mesh3d(m.cylinder.clone()), mat(BRISTLE), rod(f, root, root + v(-0.6, 0.8, 0.0) * bl, 0.7)));
            }
        }
        Part::Head => {
            let f = body_tilt();
            c.spawn((Mesh3d(m.sphere.clone()), mat(HEAD), ell(f, Vec3::ZERO, v(13.0, 14.0, 15.0))));
            for sgn in [1.0f32, -1.0] {
                let eye = v(1.5, 2.5, sgn * 8.5);
                c.spawn((Mesh3d(m.eye.clone()), mat(EYE), ell(f, eye, v(12.0, 14.0, 11.5))));
                c.spawn((
                    EyeGlint,
                    OfFly(fly),
                    Mesh3d(m.sphere.clone()),
                    MeshMaterial3d(look.glint.clone()),
                    ell(f, eye + v(4.0, 5.0, sgn * 6.0), Vec3::splat(2.4 * S)),
                ));
                for a in [0.8f32, -0.8] {
                    let mut t = ell(f, eye + v(0.0, 0.0, sgn * 8.0), v(18.0, 3.0, 3.0));
                    t.rotation = f * Quat::from_rotation_z(a);
                    c.spawn((EyeCross, OfFly(fly), Mesh3d(m.cube.clone()), MeshMaterial3d(look.cross.clone()), t, Visibility::Hidden));
                }
                let a0 = v(11.0, 8.0, sgn * 4.0);
                c.spawn((Mesh3d(m.cylinder.clone()), mat(DARK), rod(f, a0, a0 + v(10.0, 10.0, sgn * 5.0), 1.2)));
                let root = v(4.0, 9.0, sgn * 4.0);
                c.spawn((Bristle, OfFly(fly), Mesh3d(m.cylinder.clone()), mat(BRISTLE), rod(f, root, root + v(0.4, 0.9, 0.0) * 6.0, 0.6)));
            }
            // the mouthparts under the head
            c.spawn((Mesh3d(m.cylinder.clone()), mat(DARK), rod(f, v(8.0, -10.0, 0.0), v(14.0, -20.0, 0.0), 1.8)));
        }
    }
}

#[derive(Component)]
pub struct FlyVisuals {
    pub rig: LegRig,
    /// Per leg: thigh and shin.
    pub segments: Vec<[Entity; 2]>,
    /// Per leg: the knee and foot beads.
    pub beads: Vec<[Entity; 2]>,
    /// Per wing: the pivot at the wing base (its child is the blade).
    pub wings: [Entity; 2],
    pub proboscis: Entity,
}

impl FlyVisuals {
    /// Everything that is not a child of a body part (the respawn despawns these by hand).
    pub fn loose(&self) -> impl Iterator<Item = Entity> + '_ {
        self.segments.iter().chain(&self.beads).flatten().copied().chain([self.proboscis])
    }
}

#[derive(Component)]
struct LegSegment;

/// Where the wing leaves the thorax and where its tip rests (`_draw_fly`'s `base`, `REST3[WING]`), in the thorax's frame.
fn wing_base(sgn: f32) -> Vec3 {
    body_tilt() * Vec3::new(0.0, 12.0 * S, sgn * 5.0 * S)
}
fn wing_rest_tip(sgn: f32) -> Vec3 {
    Vec3::new(-62.0 * S, 22.0 * S, sgn * 0.07)
}

fn rig() -> LegRig {
    // the hips of `_draw_fly`: 14, 2 and -10 S along the body, 10 S under the thorax, 8 S to each side
    let fwd_hip = [14.0 * S, 2.0 * S, -10.0 * S];
    let fwd_rest = [0.348, 0.06, -0.264];
    let (mut hips, mut rests, mut groups) = (Vec::new(), Vec::new(), Vec::new());
    for far in [false, true] {
        let sgn = if far { -1.0 } else { 1.0 };
        for i in 0..3 {
            let k = if far { 0.06 } else { 0.0 };
            hips.push(Vec3::new(fwd_hip[i], -10.0 * S, sgn * 8.0 * S));
            rests.push(Vec3::new(fwd_rest[i] + k, -STAND3, sgn * 0.26));
            // near front/hind and far mid step together (the tripod)
            groups.push(((i + usize::from(far)) % 2) as u8);
        }
    }
    LegRig::new(hips, rests, LEG_L1, LEG_L2, groups, 0.10, 0.12, 0.05)
}

fn attach_visuals(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, new: Query<(Entity, &Fly, &FlyLook), Added<Fly>>) {
    if new.is_empty() {
        return;
    }
    let seg_mesh = meshes.add(Cylinder::new(1.0, 1.0).mesh().resolution(10));
    let bead_mesh = meshes.add(Sphere::new(1.0).mesh().uv(10, 8));
    let wing_mesh = meshes.add(Sphere::new(1.0).mesh().uv(20, 10));
    for (e, f, look) in &new {
        let (mut segments, mut beads) = (Vec::new(), Vec::new());
        for _ in 0..6 {
            let seg = |commands: &mut Commands, mesh: &Handle<Mesh>, mat: usize| {
                commands
                    .spawn((
                        LegSegment,
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(look.tinted[mat].0.clone()),
                        Transform::default(),
                        Visibility::default(),
                    ))
                    .id()
            };
            segments.push([seg(&mut commands, &seg_mesh, LEG), seg(&mut commands, &seg_mesh, LEG)]);
            beads.push([seg(&mut commands, &bead_mesh, LEG), seg(&mut commands, &bead_mesh, FOOT)]);
        }
        let thx = f.parts[Part::Thorax.index()];
        let mut wings = [Entity::PLACEHOLDER; 2];
        for (k, sgn) in [1.0f32, -1.0].into_iter().enumerate() {
            let pivot = commands.spawn((Transform::from_translation(wing_base(sgn)), Visibility::default())).id();
            let blade = commands.spawn((Mesh3d(wing_mesh.clone()), MeshMaterial3d(look.wing.clone()), Transform::default())).id();
            commands.entity(pivot).add_child(blade);
            commands.entity(thx).add_child(pivot);
            wings[k] = pivot;
        }
        let proboscis = commands
            .spawn((Mesh3d(seg_mesh.clone()), MeshMaterial3d(look.tinted[PROBOSCIS].0.clone()), Transform::default(), Visibility::Hidden))
            .id();
        commands.entity(e).insert(FlyVisuals { rig: rig(), segments, beads, wings, proboscis });
    }
}

fn place_segment(t: &mut Transform, a: Vec3, b: Vec3, radius: f32) {
    let d = b - a;
    let len = d.length().max(1e-4);
    t.translation = (a + b) / 2.0;
    t.rotation = Quat::from_rotation_arc(Vec3::Y, d / len);
    t.scale = Vec3::new(radius, len, radius);
}

#[allow(clippy::type_complexity)]
fn animate(
    time: Res<Time>,
    virt: Res<Time<Virtual>>,
    mut flies: Query<(Entity, &Fly, &mut FlyVisuals)>,
    parts: Query<(&FlyPart, &Transform), Without<LegSegment>>,
    mut segs: Query<(&mut Transform, &mut Visibility), With<LegSegment>>,
    mut pivots: Query<(&mut Transform, &Children), (Without<LegSegment>, Without<FlyPart>)>,
    mut blades: Query<&mut Transform, (Without<LegSegment>, Without<FlyPart>, Without<Children>)>,
) {
    let now = virt.elapsed_secs();
    let dt = time.delta_secs();
    for (e, f, mut vis) in &mut flies {
        let Some((_, t)) = parts.iter().find(|(p, _)| p.fly == e && p.part == Part::Thorax) else { continue };
        let (pos, rot) = (t.translation, t.rotation);
        let fwd = rot * Vec3::X;
        let planted = f.strength > 0.5 && !f.flying_now && pos.y < STAND3 + 0.25 && f.grabbed.is_none() && !f.dead();
        let gone = f.gone();
        let poses = if planted {
            let moving = now < f.react.walk_until || now < f.react.back_until;
            vis.rig.update(pos, fwd, dt, moving, Some(0.012), &[], 0.0)
        } else {
            // dangling or flying: legs hang from the body's own frame (it may be tumbling)
            let mut out = Vec::new();
            for i in 0..6 {
                let (hip_l, rest_l) = (vis.rig.hips[i], vis.rig.rests[i]);
                let hip = pos + rot * hip_l;
                let mut foot_l = if f.flying_now {
                    Vec3::new(rest_l.x - 0.18, -0.20, rest_l.z * 0.55)
                } else {
                    Vec3::new(rest_l.x, -0.22, rest_l.z * 0.75)
                };
                if now < f.react.flail_until || f.grabbed.is_some() {
                    let k = now * 22.0 + i as f32 * 1.7;
                    foot_l += Vec3::new(k.sin() * 0.06, k.cos() * 0.05, (k * 1.3).sin() * 0.05);
                }
                if f.dead() {
                    foot_l = Vec3::new(rest_l.x * 0.5, -0.12, rest_l.z * 0.5);
                }
                let foot = pos + rot * foot_l;
                let bend = rot * (Vec3::Y + Vec3::Z * hip_l.z.signum() * 0.4);
                let (knee, foot) = two_bone_ik(hip, foot, LEG_L1, LEG_L2, bend);
                out.push(crate::anim::LegPose { hip, knee, foot });
            }
            out
        };
        let show = if gone { Visibility::Hidden } else { Visibility::Inherited };
        for (i, p) in poses.iter().enumerate() {
            for (k, (a, b)) in [(p.hip, p.knee), (p.knee, p.foot)].into_iter().enumerate() {
                if let Ok((mut ts, mut vs)) = segs.get_mut(vis.segments[i][k]) {
                    place_segment(&mut ts, a, b, LEG_R[k]);
                    *vs = show;
                }
                if let Ok((mut tb, mut vb)) = segs.get_mut(vis.beads[i][k]) {
                    *tb = Transform::from_translation(b).with_scale(Vec3::splat(BEAD_R[k]));
                    *vb = show;
                }
            }
        }
        // wings (`_draw_fly`): a flat clear ellipsoid from the wing base to its tip, 9 S longer than half the span, 12 S wide. Folded
        // along the back at rest; in flight (or struggling in your hand) the tip beats up and down at 90 rad/s and swings out.
        let flap = !f.dead() && (f.flying_now || (f.grabbed.is_some() && (now * 31.0).sin() > 0.4));
        for (k, sgn) in [1.0f32, -1.0].into_iter().enumerate() {
            let Ok((mut pivot, children)) = pivots.get_mut(vis.wings[k]) else { continue };
            let base = wing_base(sgn);
            let mut tip = wing_rest_tip(sgn);
            if flap {
                tip = base + (tip - base) * 0.8 + Vec3::new(0.0, 38.0 * S * (now * 90.0 + k as f32).sin(), sgn * 10.0 * S);
            } else if f.dead() {
                tip = Vec3::new(-50.0 * S, 4.0 * S, sgn * 0.2); // limp, spread
            }
            let span = (tip - base).length().max(1e-4);
            pivot.rotation = Quat::from_rotation_arc(Vec3::NEG_X, (tip - base) / span);
            if let Some(mut blade) = children.first().and_then(|c| blades.get_mut(*c).ok()) {
                *blade = Transform::from_xyz(-span / 2.0, 0.0, 0.0).with_scale(Vec3::new(span / 2.0 + 9.0 * S, 0.9 * S, 12.0 * S));
            }
        }
    }
}

/// The marks on the body that come and go: the glint on a living eye, the cross on a dead one, no bristles on a wrapped fly.
#[allow(clippy::type_complexity)]
fn marks(
    flies: Query<&Fly>,
    mut shapes: Query<
        (&OfFly, &mut Visibility, Has<EyeGlint>, Has<EyeCross>, Has<Bristle>),
        Or<(With<EyeGlint>, With<EyeCross>, With<Bristle>)>,
    >,
) {
    for (of, mut v, glint, cross, bristle) in &mut shapes {
        let Ok(f) = flies.get(of.0) else { continue };
        let on = (glint && !f.dead()) || (cross && f.dead()) || (bristle && !f.wrapped);
        let want = if on { Visibility::Inherited } else { Visibility::Hidden };
        if *v != want {
            *v = want;
        }
    }
}

/// `_draw_fly`'s `col`: every body colour goes through `k2.shade` (burnt black, frosted blue-white, grey when dead, sickly and
/// see-through while it dissolves, flashing white with pain), and a zapped fly flickers pale blue. The glow of a burning or frozen
/// body is this port's own (Python showed those with particles and an ice block only).
fn tint(mut materials: ResMut<Assets<StandardMaterial>>, flies: Query<(&Fly, &FlyLook)>, virt: Res<Time<Virtual>>) {
    let now = virt.elapsed_secs();
    for (f, look) in &flies {
        let zapped = now < f.zap_until && (now * 40.0) as i64 % 2 == 0;
        let emissive = if now < f.burn_until {
            LinearRgba::rgb(1.4, 0.35, 0.05)
        } else if f.frozen_at.is_some() {
            LinearRgba::rgb(0.1, 0.25, 0.4)
        } else {
            LinearRgba::BLACK
        };
        for (handle, base) in &look.tinted {
            let (c, alpha) = if zapped {
                ([204.0, 235.0, 255.0], 1.0)
            } else {
                shade(*base, f.hurt.clamp(0.0, 1.0), f.dead(), f.char_.clamp(0.0, 1.0), f.melt.clamp(0.0, 1.0), f.frost.clamp(0.0, 1.0))
            };
            let color = Color::srgba(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, alpha);
            let mode = if alpha < 1.0 { AlphaMode::Blend } else { AlphaMode::Opaque };
            // only touch the asset when something changed (a touched material is prepared again for the GPU)
            if materials.get(handle).is_some_and(|m| m.base_color != color || m.emissive != emissive)
                && let Some(mut m) = materials.get_mut(handle)
            {
                m.base_color = color;
                m.emissive = emissive;
                m.alpha_mode = mode;
            }
        }
        let wing_alpha = if f.melt > 0.0 { 0.18 } else { 0.34 };
        if materials.get(&look.wing).is_some_and(|m| m.base_color.alpha() != wing_alpha)
            && let Some(mut m) = materials.get_mut(&look.wing)
        {
            m.base_color.set_alpha(wing_alpha);
        }
    }
}

/// A fly that has dissolved or shattered is gone: hide its body and take it out of the physics.
fn hide_gone(mut commands: Commands, flies: Query<&Fly>, mut parts: Query<(Entity, &FlyPart, &mut Visibility)>) {
    for (e, p, mut v) in &mut parts {
        if let Ok(f) = flies.get(p.fly)
            && f.gone()
            && *v != Visibility::Hidden
        {
            *v = Visibility::Hidden;
            commands.entity(e).insert(avian3d::prelude::RigidBodyDisabled);
        }
    }
}

pub struct FlyVisualsPlugin;

impl Plugin for FlyVisualsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, attach_visuals)
            .add_systems(PostUpdate, (animate, marks, tint, hide_gone).after(TransformSystems::Propagate));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from `k2.shade` (legacy-python/kickthefly/game/kick_the_fly.py) for the abdomen's colour.
    #[test]
    fn shade_matches_python() {
        let c = [206.0, 172.0, 104.0];
        assert_eq!(shade(c, 0.0, false, 0.0, 0.0, 0.0), (c, 1.0));
        assert_eq!(shade(c, 0.5, false, 0.0, 0.0, 0.0).0, [223.0, 201.0, 156.0]);
        assert_eq!(shade(c, 0.0, true, 0.0, 0.0, 0.0).0, [163.0, 148.0, 117.0]);
        assert_eq!(shade(c, 0.0, false, 1.0, 0.0, 0.0).0, [73.0, 65.0, 48.0]);
        assert_eq!(shade(c, 0.0, false, 0.0, 0.0, 1.0).0, [184.0, 202.0, 202.0]);
        let (m, a) = shade(c, 0.0, false, 0.0, 0.5, 0.0);
        assert_eq!(m, [189.0, 168.0, 107.0]);
        assert!((a - 0.7).abs() < 1e-6);
    }

    #[test]
    fn the_abdomen_is_banded_on_top_and_pale_below() {
        // on top, just behind a tergite's edge: dark; further along the same segment: the plain colour
        let top = |x: f32| stripes(Vec3::new(x, (1.0 - x * x).sqrt(), 0.0));
        assert!(top(-0.2)[0] < 0.55, "{:?}", top(-0.2));
        assert!(top(-0.02)[0] > 0.97);
        // underneath: no band, a little yellower (red up, blue down)
        let under = stripes(Vec3::new(-0.2, -(1.0f32 - 0.04).sqrt(), 0.0));
        assert!(under[0] > 1.0 && under[2] < 0.95);
        // no band at the very tip, where the dark tip takes over
        assert!(stripes(Vec3::new(-0.95, 0.31, 0.0))[0] > 0.95);
    }

    #[test]
    fn six_legs_reach_their_rest_spots() {
        let r = rig();
        assert_eq!(r.hips.len(), 6);
        // every foot's rest spot is within reach (the far front leg, set 10 S further forward as in `REST`, is at full stretch)
        for (h, f) in r.hips.iter().zip(&r.rests) {
            assert!(h.distance(*f) < (LEG_L1 + LEG_L2) * 1.06, "{h:?} {f:?}");
        }
        // a folded wing lies back along the abdomen and is about as long as the body
        let span = (wing_rest_tip(1.0) - wing_base(1.0)).length();
        assert!((0.34..0.42).contains(&span), "{span}");
    }
}
