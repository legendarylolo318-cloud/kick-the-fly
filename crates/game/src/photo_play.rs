//! Photo mode and picture saving (`kick3d.py:{FreeCamera, toggle_photo_mode, take_photo, _draw_photo_viewfinder}`, `Game.save_png`).
//!
//! F10 detaches a free camera from you (WASD, Space/C up and down, Shift sprints, the mouse aims); `[` `]` change the field of view, `,` `.`
//! the depth-of-field blur, `K` `L` the focus distance, `F` focuses on the nearest fly in view, `H` hides the HUD, and F12 / Enter / Space /
//! a click takes the picture into the pictures folder. The blur is Bevy's `DepthOfField` post effect on the camera (Python blitted its own
//! DoF shader). Pictures are the window's size: Python's `graphics.photo_scale` supersampling is not done.

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::post_process::dof::{DepthOfField, DepthOfFieldMode};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

use crate::actions::Actions;
use crate::config::{Arena, Settings};
use crate::feedback::{Notes, Sfx, Tag};
use crate::fly::{Fly, FlyPart, Part};
use crate::menu_ui::{MenuOpen, UserConfig};
use crate::player::{Aim, MouseLook, PlayerCamera};

/// What the free camera keeps from frame to frame, and the picture waiting to be taken.
#[derive(Resource)]
pub struct Photo {
    pub active: bool,
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    vel: Vec3,
    pub fov_deg: f32,
    pub dof: f32,
    pub focus: f32,
    pub hide_ui: bool,
    /// A picture was asked for (F12, the shutter keys, a click).
    pub snap: bool,
    /// The toast under the top edge: text and the real time it was set.
    pub toast: Option<(String, f32)>,
    /// Frames until the HUD is back after a clean capture.
    clean_frames: u32,
    path: Option<std::path::PathBuf>,
}

impl Default for Photo {
    fn default() -> Self {
        Self {
            active: false,
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            vel: Vec3::ZERO,
            fov_deg: 70.0,
            dof: 0.0,
            focus: 1.8,
            hide_ui: false,
            snap: false,
            toast: None,
            clean_frames: 0,
            path: None,
        }
    }
}

impl Photo {
    /// Photo mode already on, looking from `eye` at `target` (the `--photo` screenshot harness).
    pub fn at(eye: Vec3, target: Vec3, fov_deg: f32, dof: f32, focus: f32) -> Self {
        let d = (target - eye).normalize_or_zero();
        Self { active: true, pos: eye, yaw: d.z.atan2(d.x), pitch: d.y.clamp(-1.0, 1.0).asin(), fov_deg, dof, focus, ..Self::default() }
    }
}

/// `Game.media_path`: a new file in the pictures folder; two saves in the same second do not overwrite each other.
pub fn media_path(ext: &str) -> std::path::PathBuf {
    let dir = ktf_data::paths::get().pictures_dir.clone();
    let _ = std::fs::create_dir_all(&dir);
    let stem = format!("kick-the-fly-{}", crate::savestate_play::utc_stamp_compact().replace('-', "").replacen("T", "-", 1));
    let mut n = 2;
    let mut path = dir.join(format!("{stem}.{ext}"));
    while path.exists() {
        path = dir.join(format!("{stem}-{n}.{ext}"));
        n += 1;
    }
    path
}

fn nearest_fly_dist(eye: Vec3, fwd: Vec3, flies: &Query<&Fly>, parts: &Query<&Transform, With<FlyPart>>) -> f32 {
    let mut best = f32::INFINITY;
    let mut any = None;
    for f in flies.iter() {
        let Ok(t) = parts.get(f.parts[Part::Thorax.index()]) else { continue };
        let v = t.translation - eye;
        let d = v.length();
        any = any.or(Some(d));
        if d < 0.05 {
            continue;
        }
        if v.dot(fwd) / d > 0.2 {
            best = best.min(d);
        }
    }
    if best < 100.0 {
        (best * 100.0).round() / 100.0
    } else {
        any.map_or(1.8, |d| (d * 100.0).round() / 100.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn keys(
    act: Res<Actions>,
    kb: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    menu: Res<MenuOpen>,
    mut photo: ResMut<Photo>,
    mut look: ResMut<MouseLook>,
    aim: Res<Aim>,
    cfg: Res<UserConfig>,
    settings: Res<Settings>,
    flies: Query<&Fly>,
    parts: Query<&Transform, With<FlyPart>>,
    time: Res<Time<Real>>,
    mut notes: ResMut<Notes>,
) {
    let now = time.elapsed_secs();
    if menu.0 {
        return;
    }
    if act.just("photo_mode") || kb.just_pressed(KeyCode::F10) {
        photo.active = !photo.active;
        if photo.active {
            photo.pos = aim.eye;
            photo.yaw = aim.dir.z.atan2(aim.dir.x);
            photo.pitch = aim.dir.y.clamp(-1.0, 1.0).asin();
            photo.vel = Vec3::ZERO;
            photo.fov_deg = settings.fov_deg;
            photo.dof = cfg.0.get_f32("graphics.photo_dof");
            photo.focus = nearest_fly_dist(aim.eye, aim.dir, &flies, &parts);
            look.captured = true;
            notes.push(now, "PHOTO MODE: WASD/Space/C fly, Mouse aim, [ ] FOV, , . DOF, F12 snap", Tag::GameRule);
        } else {
            photo.hide_ui = false;
            notes.push(now, "PHOTO MODE off", Tag::GameRule);
        }
        return;
    }
    if !photo.active {
        return;
    }
    let toast = |p: &mut Photo, s: String| p.toast = Some((s, now));
    if kb.just_pressed(KeyCode::BracketLeft) {
        photo.fov_deg = (photo.fov_deg - 5.0).max(20.0);
        let s = format!("FOV: {:.0}°", photo.fov_deg);
        toast(&mut photo, s);
    }
    if kb.just_pressed(KeyCode::BracketRight) {
        photo.fov_deg = (photo.fov_deg + 5.0).min(120.0);
        let s = format!("FOV: {:.0}°", photo.fov_deg);
        toast(&mut photo, s);
    }
    if kb.just_pressed(KeyCode::Comma) {
        photo.dof = (((photo.dof - 0.05) * 100.0).round() / 100.0).max(0.0);
        let s = format!("DOF blur: {:.2}", photo.dof);
        toast(&mut photo, s);
    }
    if kb.just_pressed(KeyCode::Period) {
        photo.dof = (((photo.dof + 0.05) * 100.0).round() / 100.0).min(1.0);
        let s = format!("DOF blur: {:.2}", photo.dof);
        toast(&mut photo, s);
    }
    if kb.just_pressed(KeyCode::KeyK) {
        photo.focus = (((photo.focus - 0.2) * 100.0).round() / 100.0).max(0.2);
        let s = format!("Focus: {:.1}m", photo.focus);
        toast(&mut photo, s);
    }
    if kb.just_pressed(KeyCode::KeyL) {
        photo.focus = (((photo.focus + 0.2) * 100.0).round() / 100.0).min(15.0);
        let s = format!("Focus: {:.1}m", photo.focus);
        toast(&mut photo, s);
    }
    if kb.just_pressed(KeyCode::KeyF) {
        let (eye, fwd) = free_view(&photo);
        photo.focus = nearest_fly_dist(eye, fwd, &flies, &parts);
        let s = format!("Auto-focus: {:.1}m", photo.focus);
        toast(&mut photo, s);
    }
    if kb.just_pressed(KeyCode::KeyH) {
        photo.hide_ui = !photo.hide_ui;
    }
    if kb.just_pressed(KeyCode::Enter) || kb.just_pressed(KeyCode::Space) || (mouse.just_pressed(MouseButton::Left) && look.captured) {
        photo.snap = true;
    }
}

fn free_view(p: &Photo) -> (Vec3, Vec3) {
    let cp = p.pitch.cos();
    (p.pos, Vec3::new(cp * p.yaw.cos(), p.pitch.sin(), cp * p.yaw.sin()))
}

/// `FreeCamera.update`: velocity eases toward the wanted one (2.5 m/s, 5.5 sprinting), kept inside the room.
#[allow(clippy::too_many_arguments)]
fn fly_camera(
    time: Res<Time<Real>>,
    act: Res<Actions>,
    kb: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    look: Res<MouseLook>,
    menu: Res<MenuOpen>,
    settings: Res<Settings>,
    arena: Res<Arena>,
    mut photo: ResMut<Photo>,
) {
    if !photo.active {
        return;
    }
    let dt = time.delta_secs().min(0.05);
    if look.captured && !look.wheel && !menu.0 {
        photo.yaw += motion.delta.x * settings.mouse_sensitivity;
        photo.pitch = (photo.pitch - motion.delta.y * settings.mouse_sensitivity * if settings.invert_y { -1.0 } else { 1.0 }).clamp(-1.55, 1.55);
    }
    let (_, f) = free_view(&photo);
    let r = Vec3::new(-photo.yaw.sin(), 0.0, photo.yaw.cos()).normalize_or_zero();
    let mut want = Vec3::ZERO;
    if look.captured && !menu.0 {
        if act.held("forward") {
            want += f;
        }
        if act.held("back") {
            want -= f;
        }
        if act.held("right") {
            want += r;
        }
        if act.held("left") {
            want -= r;
        }
        if kb.pressed(KeyCode::Space) || kb.pressed(KeyCode::KeyE) {
            want += Vec3::Y;
        }
        if act.held("crouch") || kb.pressed(KeyCode::KeyC) || kb.pressed(KeyCode::KeyQ) || kb.pressed(KeyCode::ControlLeft) {
            want -= Vec3::Y;
        }
    }
    let speed = if act.held("sprint") { 5.5 } else { 2.5 };
    let target = if want.length() > 0.0 { want.normalize() * speed } else { Vec3::ZERO };
    let vel = photo.vel;
    photo.vel += (target - vel) * (dt * 10.0).min(1.0);
    let step = photo.vel * dt;
    photo.pos += step;
    let w = crate::world::spec(arena.name());
    photo.pos.x = photo.pos.x.clamp(-w.rx + 0.15, w.rx - 0.15);
    photo.pos.y = photo.pos.y.clamp(0.08, w.ry - 0.08);
    photo.pos.z = photo.pos.z.clamp(-w.rz + 0.15, w.rz - 0.15);
}

/// After the walking camera is placed: the free camera takes over, with its own field of view and the blur.
fn apply_camera(
    photo: Res<Photo>,
    mut cam: Single<(Entity, &mut Transform, &mut Projection), With<PlayerCamera>>,
    mut commands: Commands,
    mut had: Local<bool>,
) {
    let (e, tf, proj) = &mut *cam;
    if photo.active {
        let (eye, fwd) = free_view(&photo);
        tf.translation = eye;
        tf.look_to(fwd, Vec3::Y);
        if let Projection::Perspective(p) = &mut **proj {
            p.fov = photo.fov_deg.to_radians();
        }
        if photo.dof > 0.0 {
            commands.entity(*e).insert(DepthOfField {
                mode: DepthOfFieldMode::Gaussian,
                focal_distance: photo.focus,
                aperture_f_stops: 1.0 / photo.dof.max(0.05),
                max_depth: 60.0,
                ..default()
            });
            *had = true;
        } else if *had {
            commands.entity(*e).remove::<DepthOfField>();
            *had = false;
        }
    } else if *had {
        commands.entity(*e).remove::<DepthOfField>();
        *had = false;
    }
}

/// The picture: with `graphics.clean_capture` the HUD is hidden for a few frames first, then the window is saved to the pictures folder.
#[allow(clippy::too_many_arguments)]
fn capture(
    mut photo: ResMut<Photo>,
    cfg: Res<UserConfig>,
    mut hud: Query<&mut Visibility, With<crate::hud_canvas::HudNode>>,
    mut commands: Commands,
    mut sfx: MessageWriter<Sfx>,
    time: Res<Time<Real>>,
) {
    let now = time.elapsed_secs();
    let clean = cfg.0.get_bool("graphics.clean_capture");
    if photo.snap {
        photo.snap = false;
        photo.path = Some(media_path("png"));
        photo.clean_frames = if clean { 4 } else { 1 };
        sfx.write(Sfx::new("shutter"));
    }
    if photo.clean_frames > 0 {
        if clean {
            for mut v in &mut hud {
                *v = Visibility::Hidden;
            }
        }
        photo.clean_frames -= 1;
        if photo.clean_frames == 0
            && let Some(path) = photo.path.take()
        {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            let folder = path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("").to_string();
            commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
            photo.toast = Some((format!("saved {name} in {folder}"), now));
            photo.clean_frames = u32::MAX; // restore the HUD next frame
        }
        return;
    }
    if photo.clean_frames == u32::MAX {
        photo.clean_frames = 0;
    }
    for mut v in &mut hud {
        *v = if photo.active && photo.hide_ui { Visibility::Hidden } else { Visibility::Inherited };
    }
}

/// Photo mode takes its keys: the other actions bound to them do not fire (Python's handler returns first).
fn swallow_keys(photo: Res<Photo>, cfg: Res<UserConfig>, mut act: ResMut<Actions>) {
    if !photo.active {
        return;
    }
    for k in ["[", "]", ",", ".", "k", "l", "f", "h", "return", "space"] {
        for a in cfg.0.actions_for(k) {
            if a != "photo_mode" && a != "screenshot" {
                act.swallow(&a);
            }
        }
    }
}

pub struct PhotoPlugin;

impl Plugin for PhotoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Photo>().add_systems(PreUpdate, swallow_keys.after(crate::actions::collect)).add_systems(
            Update,
            (
                keys,
                fly_camera,
                apply_camera.after(crate::player::update_camera),
                capture.after(keys),
            ),
        );
    }
}
