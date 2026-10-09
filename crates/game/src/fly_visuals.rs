//! How the fly looks: six IK legs on a tripod gait with planted feet, wings that flap in flight, and the tint of burns, frost, melt and pain.
//! The legs and wings do not collide; they follow the physics bodies. Python: `Game3D._draw_fly` (which drew the 17 particles).

use bevy::prelude::*;

use crate::anim::{LegRig, two_bone_ik};
use crate::fly::{Fly, FlyLook, FlyPart, Part};
use crate::world::{S, STAND3};

const LEG_L1: f32 = 0.20;
const LEG_L2: f32 = 0.24;
const LEG_R: f32 = 0.011;

#[derive(Component)]
pub struct FlyVisuals {
    pub rig: LegRig,
    pub segments: Vec<[Entity; 2]>,
    pub wings: [Entity; 2],
    pub proboscis: Entity,
}

#[derive(Component)]
struct LegSegment;

fn rig() -> LegRig {
    let fwd_hip = [0.14, 0.02, -0.12];
    let fwd_rest = [0.348, 0.06, -0.264];
    let (mut hips, mut rests, mut groups) = (Vec::new(), Vec::new(), Vec::new());
    for far in [false, true] {
        let sgn = if far { -1.0 } else { 1.0 };
        for i in 0..3 {
            let k = if far { 0.06 } else { 0.0 };
            hips.push(Vec3::new(fwd_hip[i], -0.05, sgn * 0.07));
            rests.push(Vec3::new(fwd_rest[i] + k, -STAND3, sgn * 0.26));
            // near front/hind and far mid step together (the tripod)
            groups.push(((i + usize::from(far)) % 2) as u8);
        }
    }
    LegRig::new(hips, rests, LEG_L1, LEG_L2, groups, 0.10, 0.12, 0.05)
}

fn attach_visuals(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, new: Query<(Entity, &Fly, &FlyLook), Added<Fly>>) {
    let seg_mesh = meshes.add(Cylinder::new(LEG_R, 1.0));
    let wing_mesh = meshes.add(Cuboid::new(0.36, 0.004, 0.12));
    let prob_mesh = meshes.add(Cylinder::new(0.012, 1.0));
    for (e, f, look) in &new {
        let mut segments = Vec::new();
        for _ in 0..6 {
            let a = commands
                .spawn((
                    LegSegment,
                    Mesh3d(seg_mesh.clone()),
                    MeshMaterial3d(look.body.clone()),
                    Transform::default(),
                    Visibility::default(),
                ))
                .id();
            let b = commands
                .spawn((
                    LegSegment,
                    Mesh3d(seg_mesh.clone()),
                    MeshMaterial3d(look.body.clone()),
                    Transform::default(),
                    Visibility::default(),
                ))
                .id();
            segments.push([a, b]);
        }
        let thx = f.parts[Part::Thorax.index()];
        let mut wings = [Entity::PLACEHOLDER; 2];
        for (k, side) in [1.0f32, -1.0].into_iter().enumerate() {
            let pivot = commands.spawn((Transform::from_translation(Vec3::new(-0.03, 0.095, side * 0.05)), Visibility::default())).id();
            let blade = commands
                .spawn((
                    Mesh3d(wing_mesh.clone()),
                    MeshMaterial3d(look.wing.clone()),
                    Transform::from_translation(Vec3::new(-0.18, 0.0, side * 0.05)),
                ))
                .id();
            commands.entity(pivot).add_child(blade);
            commands.entity(thx).add_child(pivot);
            wings[k] = pivot;
        }
        let proboscis =
            commands.spawn((Mesh3d(prob_mesh.clone()), MeshMaterial3d(look.body.clone()), Transform::default(), Visibility::Hidden)).id();
        commands.entity(e).insert(FlyVisuals { rig: rig(), segments, wings, proboscis });
    }
}

fn place_segment(t: &mut Transform, a: Vec3, b: Vec3) {
    let d = b - a;
    let len = d.length().max(1e-4);
    t.translation = (a + b) / 2.0;
    t.rotation = Quat::from_rotation_arc(Vec3::Y, d / len);
    t.scale = Vec3::new(1.0, len, 1.0);
}

#[allow(clippy::type_complexity)]
fn animate(
    time: Res<Time>,
    virt: Res<Time<Virtual>>,
    mut flies: Query<(Entity, &Fly, &mut FlyVisuals)>,
    parts: Query<(&FlyPart, &Transform), Without<LegSegment>>,
    mut segs: Query<(&mut Transform, &mut Visibility), With<LegSegment>>,
    mut pivots: Query<&mut Transform, (Without<LegSegment>, Without<FlyPart>)>,
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
        for (i, p) in poses.iter().enumerate() {
            let [a, b] = vis.segments[i];
            if let Ok((mut ta, mut va)) = segs.get_mut(a) {
                place_segment(&mut ta, p.hip, p.knee);
                *va = if gone { Visibility::Hidden } else { Visibility::Inherited };
            }
            if let Ok((mut tb, mut vb)) = segs.get_mut(b) {
                place_segment(&mut tb, p.knee, p.foot);
                *vb = if gone { Visibility::Hidden } else { Visibility::Inherited };
            }
        }
        // wings: flap when flying, fold along the back otherwise, spread limp when dead
        let flap = if f.flying_now {
            (now * 75.0).sin() * 0.75 + 0.35
        } else if f.dead() {
            0.9
        } else {
            0.0
        };
        for (k, side) in [1.0f32, -1.0].into_iter().enumerate() {
            if let Ok(mut w) = pivots.get_mut(vis.wings[k]) {
                let fold = if f.flying_now || f.dead() { 0.0 } else { 0.35 };
                w.rotation =
                    Quat::from_rotation_x(side * (flap + fold)) * Quat::from_rotation_y(side * if f.flying_now { 0.4 } else { 0.1 });
            }
        }
    }
    let _ = S;
}

/// Burns blacken the fly, frost turns it blue-white, melting and pain redden it.
fn tint(mut materials: ResMut<Assets<StandardMaterial>>, flies: Query<(&Fly, &FlyLook)>, virt: Res<Time<Virtual>>) {
    let now = virt.elapsed_secs();
    for (f, look) in &flies {
        let Some(mut m) = materials.get_mut(&look.body) else { continue };
        let base = Vec3::new(0.16, 0.13, 0.11);
        let char_c = Vec3::new(0.02, 0.02, 0.02);
        let frost_c = Vec3::new(0.75, 0.88, 1.0);
        let melt_c = Vec3::new(0.55, 0.62, 0.45);
        let mut c = base.lerp(char_c, f.char_.clamp(0.0, 1.0));
        c = c.lerp(melt_c, f.melt.clamp(0.0, 1.0) * 0.8);
        c = c.lerp(frost_c, f.frost.clamp(0.0, 1.0));
        if f.hurt > 0.0 {
            c = c.lerp(Vec3::new(0.7, 0.1, 0.05), f.hurt * 0.5);
        }
        if now < f.zap_until {
            c = c.lerp(Vec3::new(0.7, 0.9, 1.0), 0.8);
        }
        m.base_color = Color::srgb(c.x, c.y, c.z);
        m.emissive = if now < f.burn_until {
            LinearRgba::rgb(1.4, 0.35, 0.05)
        } else if f.frozen_at.is_some() {
            LinearRgba::rgb(0.1, 0.25, 0.4)
        } else {
            LinearRgba::BLACK
        };
        m.perceptual_roughness = if f.frost > 0.3 { 0.1 } else { 0.45 };
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
        app.add_systems(Update, attach_visuals).add_systems(PostUpdate, (animate, tint, hide_gone).after(TransformSystems::Propagate));
    }
}
