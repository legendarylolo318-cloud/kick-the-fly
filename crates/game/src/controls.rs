//! Global keys: pause, slow motion, single step, kill, reset, arena, immortal, mute, fullscreen, screenshot, quit.
//! Python: `Game3D.handle3d` and `Game.time_action`. Time controls scale Bevy's virtual clock (so physics, predators and weather slow
//! together) and the brain's own speed (so what you watch is what it computes).

use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowMode};

use crate::actions::Actions;
use crate::app::{LatestSnapshot, SimLink};
use crate::config::{Arena, Settings, SwitchArena};
use crate::feedback::{Notes, Popup, Tag};
use crate::fly::ResetFly;
use crate::menu_ui::UserConfig;
use crate::sim_thread::Command;

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

/// `SimClock.SPEEDS`
const SPEEDS: [f32; 4] = [0.1, 0.25, 0.5, 1.0];

/// The time, arena, immortal, mute, pain, reset and fullscreen hotkeys (`Game.do_action`, `Game.time_action`). Every key is a rebindable action.
#[allow(clippy::too_many_arguments)]
fn keys(
    act: Res<Actions>,
    keys: Res<ButtonInput<KeyCode>>,
    link: Res<SimLink>,
    snap: Res<LatestSnapshot>,
    mut tc: ResMut<TimeControl>,
    mut virt: ResMut<Time<Virtual>>,
    mut settings: ResMut<Settings>,
    mut cfg: ResMut<UserConfig>,
    arena: Res<Arena>,
    mut switch: MessageWriter<SwitchArena>,
    mut reset: MessageWriter<ResetFly>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut notes: ResMut<Notes>,
    mut popups: MessageWriter<Popup>,
    time: Res<Time<Real>>,
    mut photo: ResMut<crate::photo_play::Photo>,
) {
    let now = time.elapsed_secs();
    if act.just("time_pause") {
        tc.paused = !tc.paused;
        link.0.send(Command::Pause(tc.paused));
        notes.push(now, if tc.paused { "TIME     paused" } else { "TIME     running" }, Tag::GameRule);
    }
    let speed_index = SPEEDS.iter().position(|s| (*s - tc.speed).abs() < 1e-6).unwrap_or(SPEEDS.len() - 1);
    if act.just("time_slower") {
        tc.speed = SPEEDS[speed_index.saturating_sub(1)];
        link.0.send(Command::SetSpeed(tc.speed));
        cfg.set_and_save("brain.sim_speed", serde_json::json!(tc.speed));
    }
    if act.just("time_faster") {
        tc.speed = SPEEDS[(speed_index + 1).min(SPEEDS.len() - 1)];
        link.0.send(Command::SetSpeed(tc.speed));
        cfg.set_and_save("brain.sim_speed", serde_json::json!(tc.speed));
    }
    if act.just("time_step") {
        if !tc.paused {
            tc.paused = true;
            link.0.send(Command::Pause(true));
        }
        link.0.send(Command::StepOnce(4));
        virt.advance_by(std::time::Duration::from_millis(20));
    }
    virt.set_relative_speed(if tc.paused || tc.menu_paused { 0.0 } else { tc.speed });
    // Shift+R is the video recorder (Python), plain R is "reset / respawn"
    if act.just("reset") && !keys.pressed(KeyCode::ShiftLeft) {
        reset.write(ResetFly::default());
    }
    if act.just("arena") {
        switch.write(SwitchArena(arena.0 + 1));
        reset.write(ResetFly::default());
        cfg.set_and_save("brain.arena", serde_json::json!(crate::world::ARENAS[(arena.0 + 1) % crate::world::ARENAS.len()]));
    }
    if act.just("immortal") {
        settings.immortal = !settings.immortal;
        cfg.set_and_save("brain.immortal", serde_json::json!(settings.immortal));
        notes.push(
            now,
            format!("IMMORTAL {}", if settings.immortal { "on: it can feel pain but never die" } else { "off" }),
            Tag::GameRule,
        );
    }
    if act.just("mute") {
        settings.mute = !settings.mute;
        cfg.set_and_save("audio.mute", serde_json::json!(settings.mute));
    }
    if act.just("pain") {
        let level = (snap.0.as_deref().map_or(0, |s| s.pain_level as usize) + 1) % fly_sim::groups::PAIN_LEVELS.len();
        link.0.send(Command::PainLevel(level));
        cfg.set_and_save("brain.pain_level", serde_json::json!(level));
        let n = snap.0.as_deref().map_or(0, |s| s.pain_neurons_all[level]);
        notes.push(
            now,
            format!("PAIN     {}: {} neurons", fly_sim::groups::PAIN_LEVELS[level].0, ktf_data::sharecode::group_thousands(n as usize)),
            Tag::GameRule,
        );
    }
    if act.just("fullscreen") {
        window.mode = if matches!(window.mode, WindowMode::Windowed) {
            WindowMode::BorderlessFullscreen(MonitorSelection::Current)
        } else {
            WindowMode::Windowed
        };
        cfg.set_and_save("graphics.fullscreen", serde_json::json!(!matches!(window.mode, WindowMode::Windowed)));
    }
    if act.just("screenshot") {
        photo.snap = true;
    }
    let _ = &mut popups;
}

pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TimeControl>().add_systems(Update, keys);
    }
}
