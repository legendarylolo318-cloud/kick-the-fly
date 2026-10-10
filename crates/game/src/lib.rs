//! Kick the Fly, the Bevy game.
//!
//! * [`sim_thread`]: the connectome simulation on a thread of its own at a fixed 200 steps/s.
//! * [`snapshot`]: what it publishes (a copy of its state) and the lock-light mailbox the render thread reads it from.
//! * [`app`]: the Bevy plugin that connects the two. Everything the player sees and touches is in the feature plugins listed in [`GamePlugin`].
//!
//! The simulation never waits for a frame and a frame never waits for the simulation.

// Bevy systems take many parameters and queries have long types; that is how ECS code reads, so these two lints are off crate-wide.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod actions;
pub mod anim;
pub mod app;
pub mod arcade;
pub mod arena;
pub mod audio;
pub mod brainview;
pub mod capture;
pub mod config;
pub mod contraption;
pub mod controls;
pub mod crash;
pub mod duel;
pub mod environment;
pub mod feedback;
pub mod first_run;
pub mod fly;
pub mod fly_visuals;
pub mod flyeye;
pub mod fx;
pub mod gpu;
pub mod hud_canvas;
pub mod killcam;
pub mod killcam_play;
pub mod kitchen;
pub mod lesion;
pub mod memory_play;
pub mod menu_ui;
pub mod mic;
pub mod mic_analysis;
pub mod neurodex_play;
pub mod paths;
pub mod player;
pub mod points;
pub mod predator_play;
pub mod predators;
pub mod profiler;
pub mod props;
pub mod puppeteer;
pub mod race;
pub mod reactions;
pub mod rng;
pub mod savestate_play;
pub mod selftest;
pub mod senses;
pub mod sim_thread;
pub mod snapshot;
pub mod synth;
pub mod tools;
pub mod tournament;
pub mod tutorial_play;
pub mod weather;
pub mod weather_play;
pub mod world;

use std::time::{Duration, Instant};

use crate::sim_thread::{SimConfig, spawn};
use crate::snapshot::Phase;

/// Result of [`run_smoke`].
#[derive(Debug, Clone)]
pub struct SmokeReport {
    pub neurons: usize,
    pub steps: u64,
    pub seconds: f64,
    pub steps_per_s: f64,
    pub snapshots: u64,
    pub steps_dropped: u64,
    pub active_fraction: f64,
}

/// Run the sim thread with no window for `seconds`, reading snapshots at 60 Hz like a render loop, and report how it went.
pub fn run_smoke(cfg: SimConfig, seconds: f64) -> Result<SmokeReport, String> {
    let handle = spawn(cfg);
    let deadline = Instant::now() + Duration::from_secs(120);
    // wait for loading + warm-up
    let first = loop {
        if let Some(s) = handle.latest() {
            match &s.phase {
                Phase::Failed(msg) => return Err(msg.clone()),
                Phase::Running | Phase::Paused => break s,
                _ => handle.mailbox.recycle(s),
            }
        }
        if Instant::now() > deadline {
            return Err("the brain did not start within 120 s".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let (start, start_step) = (Instant::now(), first.step);
    let mut snapshots = 0u64;
    let mut newest = first;
    while start.elapsed().as_secs_f64() < seconds {
        if let Some(s) = handle.latest() {
            snapshots += 1;
            let old = std::mem::replace(&mut newest, s);
            handle.mailbox.recycle(old);
        }
        std::thread::sleep(Duration::from_micros(16_667));
    }
    let elapsed = start.elapsed().as_secs_f64();
    let steps = newest.step - start_step;
    Ok(SmokeReport {
        neurons: newest.neurons,
        steps,
        seconds: elapsed,
        steps_per_s: steps as f64 / elapsed,
        snapshots,
        steps_dropped: newest.steps_dropped,
        active_fraction: newest.active_count as f64 / newest.neurons.max(1) as f64,
    })
}

use avian3d::prelude::*;
use bevy::prelude::*;

/// Everything the game adds to Bevy's defaults, one plugin per feature area.
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::default())
            // the 2D game's gravity (0.9 px/frame^2 at 0.006 m/px, 60 Hz) is 19.44 m/s^2: the feel everything was tuned at
            .insert_resource(Gravity(Vec3::NEG_Y * predators::GRAVITY))
            .insert_resource(Time::<Fixed>::from_hz(60.0))
            .init_resource::<config::Settings>()
            .add_plugins((
                feedback::FeedbackPlugin,
                gpu::GpuPlugin,
                arena::ArenaPlugin,
                player::PlayerPlugin,
                brainview::BrainViewPlugin,
                profiler::ProfilerPlugin,
                fx::FxPlugin,
                senses::SensesPlugin,
            ))
            .add_plugins(neurodex_play::DexPlugin)
            .add_plugins(menu_ui::MenuPlugin)
            .add_plugins((
                crash::CrashPlugin,
                memory_play::MemoryPlayPlugin,
                actions::ActionsPlugin,
                tutorial_play::TutorialPlugin,
                first_run::FirstRunPlugin,
                savestate_play::SaveStatePlugin,
            ))
            .add_plugins((
                hud_canvas::HudCanvasPlugin,
                weather_play::WeatherPlugin,
                controls::ControlsPlugin,
                environment::EnvironmentPlugin,
                props::PropsPlugin,
                mic::MicPlugin,
                audio::AudioPlugin,
                contraption::ContraptionPlugin,
                flyeye::FlyEyePlugin,
                killcam_play::KillCamPlugin,
                arcade::ArcadePlugin,
                tools::ToolsPlugin,
                predator_play::PredatorPlugin,
                fly::FlyPlugin,
                fly_visuals::FlyVisualsPlugin,
            ));
    }
}
