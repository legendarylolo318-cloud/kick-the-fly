//! You: a first-person body and camera. Python: `game/kick3d.py:{Player, FreeCamera}` and the walking/looking input.
//!
//! Walking is 3 m/s, sprint 5.5, crouch 1.4 with the eye lowered, smoothed toward the wanted velocity; the body slides along furniture
//! and stays inside the arena. The body is a kinematic collider so walking into a fly shoves it (and walking into it fast is a kick, see
//! `tools.rs`). Tab frees the mouse for the menus and the brain panel.

use avian3d::prelude::*;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::arena::WorldColliders;
use crate::config::{Arena, Settings};
use crate::world::{self, Aabb};

pub const EYE: f32 = 1.6;
pub const CROUCH_EYE: f32 = 0.85;
pub const PLAYER_RADIUS: f32 = 0.3;
pub const WALK: f32 = 3.0;
pub const SPRINT: f32 = 5.5;
pub const CROUCH: f32 = 1.4;

#[derive(Component)]
pub struct Player {
    pub yaw: f32,
    pub pitch: f32,
    pub eye_h: f32,
    pub vel: Vec2,
    pub walk_phase: f32,
}

#[derive(Component)]
pub struct PlayerCamera;

/// Where you are looking: the ray tools fire along.
#[derive(Resource, Debug, Clone, Copy)]
pub struct Aim {
    pub eye: Vec3,
    pub dir: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

impl Default for Aim {
    fn default() -> Self {
        Self { eye: Vec3::new(0.0, EYE, 2.6), dir: Vec3::NEG_Z, right: Vec3::X, up: Vec3::Y }
    }
}

/// A fixed camera for screenshots and tests: `(eye, target)`.
#[derive(Resource, Debug, Clone, Copy)]
pub struct CameraOverride(pub Vec3, pub Vec3);

/// Whether the mouse is captured by the game (looking) or free (menus, the brain panel).
#[derive(Resource, Debug, Clone, Copy)]
pub struct MouseLook {
    pub captured: bool,
    /// The tool wheel is open: the mouse points at a tool instead of turning the view.
    pub wheel: bool,
}

impl Default for MouseLook {
    fn default() -> Self {
        Self { captured: true, wheel: false }
    }
}

/// Push a horizontal point out of the furniture boxes along the shallowest axis (the walking body slides along them).
pub fn push_out_xz(p: &mut Vec2, r: f32, boxes: &[Aabb]) {
    for b in boxes {
        if b.lo.y > 0.9 {
            continue; // above head height: not an obstacle for the walking body
        }
        let (lo, hi) = (Vec2::new(b.lo.x - r, b.lo.z - r), Vec2::new(b.hi.x + r, b.hi.z + r));
        if p.x > lo.x && p.x < hi.x && p.y > lo.y && p.y < hi.y {
            let pen = [p.x - lo.x, hi.x - p.x, p.y - lo.y, hi.y - p.y];
            let k = pen.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |e| e.0);
            match k {
                0 => p.x = lo.x,
                1 => p.x = hi.x,
                2 => p.y = lo.y,
                _ => p.y = hi.y,
            }
        }
    }
}

fn spawn_player(mut commands: Commands, settings: Res<Settings>) {
    commands.spawn((
        Player { yaw: -std::f32::consts::FRAC_PI_2, pitch: -0.32, eye_h: EYE, vel: Vec2::ZERO, walk_phase: 0.0 },
        RigidBody::Kinematic,
        Collider::cylinder(PLAYER_RADIUS, 1.4),
        Transform::from_xyz(0.0, 0.7, 2.6),
    ));
    commands.spawn((
        PlayerCamera,
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection { fov: settings.fov_deg.to_radians(), near: 0.03, far: 200.0, ..default() }),
        Transform::from_xyz(0.0, EYE, 2.6),
    ));
}

/// Look and walk. Runs every frame in real time.
#[allow(clippy::too_many_arguments)]
fn walk(
    time: Res<Time<Real>>,
    act: Res<crate::actions::Actions>,
    motion: Res<AccumulatedMouseMotion>,
    look: Res<MouseLook>,
    settings: Res<Settings>,
    arena: Res<Arena>,
    boxes: Res<WorldColliders>,
    photo: Res<crate::photo_play::Photo>,
    mut player: Single<(&mut Player, &mut Transform, &mut LinearVelocity)>,
) {
    // photo mode: the free camera takes the mouse and the keys; the body stands still
    let look = &MouseLook { captured: look.captured && !photo.active, ..*look };
    let dt = time.delta_secs().min(0.05);
    let (p, tf, lv) = &mut *player;
    let p = &mut **p;
    if look.captured && !look.wheel {
        p.yaw += motion.delta.x * settings.mouse_sensitivity;
        p.pitch = (p.pitch - motion.delta.y * settings.mouse_sensitivity * if settings.invert_y { -1.0 } else { 1.0 }).clamp(-1.45, 1.45);
    }
    let axis = |a: &str, b: &str| f32::from(u8::from(act.held(a))) - f32::from(u8::from(act.held(b)));
    let (fwd_in, side_in) = if look.captured { (axis("forward", "back"), axis("right", "left")) } else { (0.0, 0.0) };
    let fh = Vec2::new(p.yaw.cos(), p.yaw.sin());
    let rh = Vec2::new(-p.yaw.sin(), p.yaw.cos());
    let want = fh * fwd_in + rh * side_in;
    let crouch = act.held("crouch") && look.captured;
    let speed = if crouch {
        CROUCH
    } else if act.held("sprint") {
        SPRINT
    } else {
        WALK
    };
    let target = if want.length() > 0.0 { want.normalize() * speed } else { Vec2::ZERO };
    let dv = (target - p.vel) * (dt * 12.0).min(1.0);
    p.vel += dv;
    let mut pos = Vec2::new(tf.translation.x, tf.translation.z) + p.vel * dt;
    let w = world::spec(arena.name());
    pos.x = pos.x.clamp(-w.rx + 0.3, w.rx - 0.3);
    pos.y = pos.y.clamp(-w.rz + 0.3, w.rz - 0.3);
    push_out_xz(&mut pos, 0.3, &boxes.0);
    tf.translation = Vec3::new(pos.x, 0.7, pos.y);
    lv.0 = Vec3::new(p.vel.x, 0.0, p.vel.y); // the collider's own velocity shoves flies it walks into
    p.walk_phase += p.vel.length() * dt * 2.0;
    let eye_target = if crouch { CROUCH_EYE } else { EYE };
    p.eye_h += (eye_target - p.eye_h) * (dt * 10.0).min(1.0);
}

pub fn update_camera(
    player: Single<(&Player, &Transform), Without<PlayerCamera>>,
    mut cam: Single<&mut Transform, With<PlayerCamera>>,
    mut aim: ResMut<Aim>,
    settings: Res<Settings>,
    mut proj: Single<&mut Projection, With<PlayerCamera>>,
    over: Option<Res<CameraOverride>>,
) {
    let (p, tf) = *player;
    let eye = Vec3::new(tf.translation.x, p.eye_h + 0.012 * (p.walk_phase * 2.0).sin(), tf.translation.z);
    let cp = p.pitch.cos();
    let dir = Vec3::new(cp * p.yaw.cos(), p.pitch.sin(), cp * p.yaw.sin());
    cam.translation = eye;
    cam.look_to(dir, Vec3::Y);
    if let Projection::Perspective(pp) = &mut **proj {
        pp.fov = settings.fov_deg.to_radians();
    }
    let right = Vec3::new(-p.yaw.sin(), 0.0, p.yaw.cos());
    *aim = Aim { eye, dir, right, up: right.cross(dir) };
    if let Some(o) = over {
        cam.translation = o.0;
        cam.look_at(o.1, Vec3::Y);
        let d = (o.1 - o.0).normalize_or_zero();
        let r = Vec3::new(-d.z, 0.0, d.x).normalize_or_zero();
        *aim = Aim { eye: o.0, dir: d, right: r, up: r.cross(d) };
    }
}

fn cursor(act: Res<crate::actions::Actions>, mut look: ResMut<MouseLook>, mut opts: Single<&mut CursorOptions, With<PrimaryWindow>>) {
    if act.just("free_mouse") {
        look.captured = !look.captured;
    }
    let (grab, visible) = if look.captured { (CursorGrabMode::Locked, false) } else { (CursorGrabMode::None, true) };
    if opts.grab_mode != grab || opts.visible != visible {
        opts.grab_mode = grab;
        opts.visible = visible;
    }
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Aim>()
            .init_resource::<MouseLook>()
            .add_systems(Startup, spawn_player)
            .add_systems(Update, (cursor, walk, update_camera).chain());
    }
}
