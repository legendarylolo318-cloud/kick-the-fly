//! The Bevy side: connect the sim thread to the world, and the first HUD.

use bevy::app::AppExit;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::sim_thread::{Command, STIMULUS_AMP, SimConfig, SimHandle, spawn};
use crate::snapshot::{Phase, Snapshot};

/// Starts the sim thread and keeps the newest [`Snapshot`] in [`LatestSnapshot`]. Needs no window or GPU, so tests and headless
/// tools use it with `MinimalPlugins`.
pub struct SimPlugin {
    pub config: SimConfig,
}

/// The render side of the sim thread.
#[derive(Resource)]
pub struct SimLink(pub SimHandle);

/// The newest snapshot seen so far (None until the thread has published one).
#[derive(Resource, Default)]
pub struct LatestSnapshot(pub Option<Box<Snapshot>>);

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SimLink(spawn(self.config.clone())))
            .init_resource::<LatestSnapshot>()
            .add_systems(PreUpdate, poll_snapshot);
    }
}

/// Swap in the newest snapshot, if one arrived, and return the old buffer to the sim thread. Never waits.
fn poll_snapshot(link: Res<SimLink>, mut latest: ResMut<LatestSnapshot>) {
    if let Some(new) = link.0.latest()
        && let Some(old) = latest.0.replace(new)
    {
        link.0.mailbox.recycle(old);
    }
}

/// A camera, a text HUD and a few keys. The foundation the real screens grow from.
pub struct HudPlugin;

#[derive(Component)]
struct HudText;

#[derive(Component)]
struct ActivityBar;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, setup_hud)
            .add_systems(Update, (hud_text, hud_bar, keys));
    }
}

fn setup_hud(mut commands: Commands) {
    commands.spawn(Camera2d);
    commands.spawn((
        Text::new("starting..."),
        TextFont { font_size: FontSize::Px(18.0), ..default() },
        Node { position_type: PositionType::Absolute, top: px(12), left: px(12), ..default() },
        HudText,
    ));
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: px(16),
            left: px(12),
            width: px(400),
            height: px(10),
            ..default()
        })
        .insert(BackgroundColor(Color::srgb(0.15, 0.15, 0.2)))
        .with_children(|p| {
            p.spawn((Node { width: percent(0), height: percent(100), ..default() }, BackgroundColor(Color::srgb(0.3, 0.8, 0.5)), ActivityBar));
        });
}

fn describe(s: &Snapshot) -> String {
    match &s.phase {
        Phase::Loading => "Loading the brain pack...".to_string(),
        Phase::WarmingUp => format!("{}: {} neurons. Warming the brain up...", s.source, s.neurons),
        Phase::Failed(msg) => format!("The brain could not start:\n{msg}\n\nPut data/kick_brain.npz next to the game, or run with --synthetic."),
        Phase::Running | Phase::Paused => {
            let state = if matches!(s.phase, Phase::Paused) { "PAUSED" } else { "running" };
            format!(
                "{} | {} neurons | {state}{}{}\nbrain: step {} ({:.2} s)  {:.0} steps/s  x{:.2}  dropped {}\ngain {:.3}   spiking now {} ({:.2}%)",
                s.source,
                s.neurons,
                if s.driving { " | looming detectors driven" } else { "" },
                if s.dead { " | DEAD" } else { "" },
                s.step,
                s.sim_time_s,
                s.steps_per_s,
                s.speed,
                s.steps_dropped,
                s.gain,
                s.active_count,
                100.0 * s.active_count as f64 / s.neurons.max(1) as f64,
            )
        }
    }
}

fn hud_text(latest: Res<LatestSnapshot>, diagnostics: Res<DiagnosticsStore>, mut text: Single<&mut Text, With<HudText>>) {
    let fps = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(|d| d.smoothed()).unwrap_or(0.0);
    let body = latest.0.as_deref().map_or_else(|| "waiting for the brain...".to_string(), describe);
    text.0 = format!("{body}\nrender: {fps:.0} fps\n\n[Space] drive the looming detectors  [K] kill  [P] pause  [Up/Down] speed  [Esc] quit");
}

fn hud_bar(latest: Res<LatestSnapshot>, mut bar: Single<&mut Node, With<ActivityBar>>) {
    if let Some(s) = latest.0.as_deref() {
        // 5 Hz target = 2.5 % of the neurons per step; the bar is full at 5 %
        let frac = (s.active_count as f32 / s.neurons.max(1) as f32 / 0.05).clamp(0.0, 1.0);
        bar.width = percent(frac * 100.0);
    }
}

fn keys(keys: Res<ButtonInput<KeyCode>>, link: Res<SimLink>, latest: Res<LatestSnapshot>, mut exit: MessageWriter<AppExit>) {
    let snap = latest.0.as_deref();
    if keys.just_pressed(KeyCode::Space) {
        let on = !snap.is_some_and(|s| s.driving);
        link.0.send(Command::DriveSpec { spec: "type:LPLC2,LC4".into(), amp: if on { STIMULUS_AMP } else { 0.0 } });
    }
    if keys.just_pressed(KeyCode::KeyK) {
        link.0.send(Command::Kill);
    }
    if keys.just_pressed(KeyCode::KeyP) {
        link.0.send(Command::Pause(!snap.is_some_and(|s| matches!(s.phase, Phase::Paused))));
    }
    let speed = snap.map_or(1.0, |s| s.speed);
    if keys.just_pressed(KeyCode::ArrowUp) {
        link.0.send(Command::SetSpeed((speed * 2.0).min(8.0)));
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        link.0.send(Command::SetSpeed((speed / 2.0).max(0.1)));
    }
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}
