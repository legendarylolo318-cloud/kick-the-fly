//! Kick the Fly, the Bevy game.
//!
//! * [`sim_thread`]: the connectome simulation on a thread of its own at a fixed 200 steps/s.
//! * [`snapshot`]: what it publishes (a copy of its state) and the lock-light mailbox the render thread reads it from.
//! * [`app`]: the Bevy plugins that connect the two and draw the first (text) HUD.
//!
//! The simulation never waits for a frame and a frame never waits for the simulation.

pub mod app;
pub mod sim_thread;
pub mod snapshot;

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
