//! Global keys: pause, slow motion, single step, kill, reset, arena, immortal, mute, fullscreen, screenshot, quit.
//! Python: `Game3D.handle3d` and `Game.time_action`. Time controls scale Bevy's virtual clock (so physics, predators and weather slow
//! together) and the brain's own speed (so what you watch is what it computes).

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PrimaryWindow, WindowMode};

use crate::app::{LatestSnapshot, SimLink};
use crate::config::{Arena, Settings, SwitchArena};
use crate::fly::ResetFly;
use crate::sim_thread::Command;
use crate::snapshot::Phase;
use crate::world::ARENAS;

#[derive(Resource)]
pub struct TimeControl {
    pub speed: f32,
    pub paused: bool,
    /// The pause menu is open (Python: `clock.menu_paused`): game time and the brain stand still.
    pub menu_paused: bool,
}

impl Default for TimeControl {
    fn default() -> Self {
        Self { speed: 1.0, paused: false, menu_paused: false }
    }
}

#[allow(clippy::too_many_arguments)]
fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    link: Res<SimLink>,
    snap: Res<LatestSnapshot>,
    mut tc: ResMut<TimeControl>,
    mut virt: ResMut<Time<Virtual>>,
    mut settings: ResMut<Settings>,
    arena: Res<Arena>,
    mut switch: MessageWriter<SwitchArena>,
    mut reset: MessageWriter<ResetFly>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut commands: Commands,
) {
    let brain_paused = snap.0.as_deref().is_some_and(|s| matches!(s.phase, Phase::Paused));
    let speed_now = snap.0.as_deref().map_or(1.0, |s| s.speed);
    if keys.just_pressed(KeyCode::KeyP) {
        link.0.send(Command::Pause(!brain_paused));
    }
    if keys.just_pressed(KeyCode::KeyZ) {
        tc.paused = !tc.paused;
        link.0.send(Command::Pause(tc.paused));
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        tc.speed = (tc.speed * 0.5).max(0.1);
        link.0.send(Command::SetSpeed(tc.speed));
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        tc.speed = (tc.speed * 2.0).min(1.0);
        link.0.send(Command::SetSpeed(tc.speed));
    }
    if keys.just_pressed(KeyCode::Period) && tc.paused {
        link.0.send(Command::StepOnce(4));
        virt.advance_by(std::time::Duration::from_millis(20));
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        link.0.send(Command::SetSpeed((speed_now * 2.0).min(8.0)));
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        link.0.send(Command::SetSpeed((speed_now / 2.0).max(0.1)));
    }
    virt.set_relative_speed(if tc.paused || tc.menu_paused { 0.0 } else { tc.speed });
    if keys.just_pressed(KeyCode::KeyK) {
        link.0.send(Command::Kill);
    }
    if keys.just_pressed(KeyCode::KeyR) && !keys.pressed(KeyCode::ShiftLeft) {
        reset.write(ResetFly::default());
    }
    if keys.just_pressed(KeyCode::KeyE) {
        switch.write(SwitchArena(arena.0 + 1));
        reset.write(ResetFly::default());
    }
    if keys.just_pressed(KeyCode::KeyI) {
        settings.immortal = !settings.immortal;
    }
    if keys.just_pressed(KeyCode::KeyM) {
        settings.mute = !settings.mute;
    }
    if keys.just_pressed(KeyCode::F11) {
        window.mode = if matches!(window.mode, WindowMode::Windowed) {
            WindowMode::BorderlessFullscreen(MonitorSelection::Current)
        } else {
            WindowMode::Windowed
        };
    }
    if keys.just_pressed(KeyCode::F12) {
        let path =
            format!("kick-the-fly-{}.png", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()));
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
    }
    let _ = ARENAS;
}

pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TimeControl>().add_systems(Update, keys);
    }
}
