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

/// Whether the mouse is captured by the game (looking) or free (menus, the brain panel). Free at launch: the first click in the room
/// captures it (see `cursor`).
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct MouseLook {
    pub captured: bool,
    /// The tool wheel is open: the mouse points at a tool instead of turning the view.
    pub wheel: bool,
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
    duel: Res<crate::duel_play::Duel>,
    mut player: Single<(&mut Player, &mut Transform, &mut LinearVelocity)>,
) {
    // photo mode: the free camera takes the mouse and the keys; the body stands still
    // dead in a duel: no looking and no walking, the body slumps (`update_player` is skipped; `duel_play` lowers the eye)
    if duel.player_dead() {
        let (_, _, lv) = &mut *player;
        lv.0 = Vec3::ZERO;
        return;
    }
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
    shake: Res<crate::feedback::Shake>,
    virt: Res<Time<Virtual>>,
    mut tick: Local<u32>,
) {
    let (p, tf) = *player;
    let eye = Vec3::new(tf.translation.x, p.eye_h + 0.012 * (p.walk_phase * 2.0).sin(), tf.translation.z);
    let cp = p.pitch.cos();
    let dir = Vec3::new(cp * p.yaw.cos(), p.pitch.sin(), cp * p.yaw.sin());
    // the screen shake after a hit or an explosion (`shake = uniform(-0.01, 0.01, 3)` while `now < shake_until`, not with reduced flashing)
    *tick = tick.wrapping_add(1);
    let jolt = if virt.elapsed_secs() < shake.until && !settings.reduced_flashing {
        let mut r = crate::rng::Rng::new(u64::from(*tick));
        Vec3::new(r.uniform(-0.01, 0.01), r.uniform(-0.01, 0.01), r.uniform(-0.01, 0.01))
    } else {
        Vec3::ZERO
    };
    cam.translation = eye + jolt;
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

/// The ONE system that owns the OS cursor (grab mode and visibility). Everything else only says what the game wants, through
/// [`MouseLook::captured`] (menus, the wheel, challenges clear it; a click in the room or closing a menu sets it) and `MenuOpen`.
///
/// The cursor is held only while the game wants the mouse AND no menu is open AND the window is focused and not minimized; in every
/// other case it is released and visible. The game starts with the mouse free (`MouseLook::default`), so nothing is grabbed until the
/// player clicks into the room.
#[allow(clippy::too_many_arguments)]
fn cursor(
    act: Res<crate::actions::Actions>,
    menu: Res<crate::menu_ui::MenuOpen>,
    time: Res<Time<Real>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut look: ResMut<MouseLook>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut opts: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut mem: Local<CursorMemory>,
) {
    if act.just("free_mouse") {
        look.captured = !look.captured;
    }
    let (focused, now) = (window.focused, time.elapsed_secs_f64());
    // Focus lost (alt-tab, clicked another window, minimized): the game gives the mouse back at once, and remembers whether you were
    // looking around.
    if mem.was_focused == Some(true) && !focused {
        mem.resume = look.captured;
        mem.resume_at = None;
        look.captured = false;
    }
    // Focus back: if you were looking around, the game takes the mouse again a moment later, by itself (alt-tab back needs no click).
    // Not at once: when a click brought the window back, that click must reach 'click the room' first, which captures without using
    // the tool in your hand; taking the mouse before it would make the same click fire the tool.
    if mem.was_focused == Some(false) && focused && std::mem::take(&mut mem.resume) {
        mem.resume_at = Some(now + REFOCUS_GRACE_S);
    }
    mem.was_focused = Some(focused);
    if let Some(at) = mem.resume_at
        && let Some(take) = refocus_step(now, at, focused, look.captured, menu.0, buttons.get_pressed().next().is_some())
    {
        mem.resume_at = None;
        if take {
            look.captured = true;
        }
    }
    let shown = window.physical_width() > 0 && window.physical_height() > 0; // a minimized window is 0 x 0 on Windows
    let hold = holds_cursor(look.captured, menu.0, focused, shown);
    // A confined pointer (Windows, X11) kept moving, unseen, while you looked around, and usually ended in a corner of the window.
    // When it is given back with the window still in front (a menu, Tab), put it in the middle, where a locked pointer would be.
    if mem.held && !hold && focused && shown && hold_mode() == CursorGrabMode::Confined {
        let middle = Vec2::new(window.width(), window.height()) / 2.0;
        window.set_cursor_position(Some(middle));
    }
    mem.held = hold;
    let (grab, visible) = if hold { (hold_mode(), false) } else { (CursorGrabMode::None, true) };
    if opts.grab_mode != grab || opts.visible != visible {
        opts.grab_mode = grab;
        opts.visible = visible;
    }
}

/// What [`cursor`] remembers between frames.
#[derive(Default)]
struct CursorMemory {
    was_focused: Option<bool>,
    /// The pointer was held last frame.
    held: bool,
    /// You were looking around when the window lost focus.
    resume: bool,
    /// When to take the mouse back after the focus returned (real seconds).
    resume_at: Option<f64>,
}

/// How long after the window is focused again the game takes the mouse back.
const REFOCUS_GRACE_S: f64 = 0.15;

/// The mouse after the focus came back, with the mouse to be taken again at `at`: `None` to keep waiting, `Some(true)` to take it now,
/// `Some(false)` to drop the plan (the focus went again, the player already clicked into the room, or a menu opened). It is never
/// taken while a mouse button is down, so no press is ever turned into a tool use.
pub fn refocus_step(now: f64, at: f64, focused: bool, captured: bool, menu_open: bool, button_down: bool) -> Option<bool> {
    if !focused || captured || menu_open {
        Some(false)
    } else if now >= at && !button_down {
        Some(true)
    } else {
        None
    }
}

/// Whether the game holds (grabs and hides) the pointer: only while it wants the mouse for looking, no menu is open, and its window is
/// focused and on screen.
pub fn holds_cursor(captured: bool, menu_open: bool, focused: bool, shown: bool) -> bool {
    captured && !menu_open && focused && shown
}

/// How the pointer is held while looking around. Windows (winit has no `Locked` there) and X11/XWayland can only keep the pointer inside
/// the window (`Confined`); asking for `Locked` on them was an error that left the pointer free. The look reads raw mouse motion, so it
/// does not matter that a confined pointer can sit on the window's edge. Wayland, macOS: a real lock.
fn hold_mode() -> CursorGrabMode {
    if cfg!(target_os = "windows") || on_x11() { CursorGrabMode::Confined } else { CursorGrabMode::Locked }
}

/// Linux without a Wayland session variable: winit opens an X11 window.
fn on_x11() -> bool {
    cfg!(target_os = "linux") && std::env::var_os("WAYLAND_DISPLAY").is_none_or(|v| v.is_empty())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mouse_is_free_at_launch() {
        assert!(!MouseLook::default().captured, "nothing is grabbed until the player clicks into the room");
    }

    #[test]
    fn the_pointer_is_held_only_in_focused_gameplay() {
        assert!(holds_cursor(true, false, true, true), "looking around in the focused window");
        assert!(!holds_cursor(false, false, true, true), "the mouse was freed (Tab, a panel, a challenge)");
        assert!(!holds_cursor(true, true, true, true), "a menu is open");
        assert!(!holds_cursor(true, false, false, true), "the window lost focus (alt-tab)");
        assert!(!holds_cursor(true, false, true, false), "the window is minimized");
    }

    #[test]
    fn coming_back_to_the_window_takes_the_mouse_again_but_never_under_a_click() {
        // alt-tab back: nothing pressed, the mouse is taken once the grace is over
        assert_eq!(refocus_step(10.0, 10.15, true, false, false, false), None, "not yet");
        assert_eq!(refocus_step(10.2, 10.15, true, false, false, false), Some(true));
        // a click brought the window back and is still down: wait, so that press is never a tool use
        assert_eq!(refocus_step(10.2, 10.15, true, false, false, true), None);
        // that click already captured the mouse ('click the room'), a menu opened, or the focus went again: nothing left to do
        assert_eq!(refocus_step(10.2, 10.15, true, true, false, false), Some(false));
        assert_eq!(refocus_step(10.2, 10.15, true, false, true, false), Some(false));
        assert_eq!(refocus_step(10.2, 10.15, false, false, false, false), Some(false));
    }

    #[test]
    fn windows_confines_the_pointer_because_it_cannot_lock_it() {
        if cfg!(target_os = "windows") {
            assert_eq!(hold_mode(), CursorGrabMode::Confined, "winit has no Locked on Windows");
        } else if !on_x11() {
            assert_eq!(hold_mode(), CursorGrabMode::Locked, "Wayland and macOS lock the pointer in place");
        }
    }
}
