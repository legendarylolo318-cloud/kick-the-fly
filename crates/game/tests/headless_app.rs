//! The Bevy plugins without a window or GPU: the sim thread runs, snapshots reach the world, and the world can steer the brain.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use kick_the_fly::app::{LatestSnapshot, SimLink, SimPlugin};
use kick_the_fly::sim_thread::{Command, STIMULUS_AMP, SimConfig, Source};
use kick_the_fly::snapshot::Phase;

fn app() -> App {
    let mut config = SimConfig::new(Source::Synthetic { n: 4000 });
    config.warmup = 30;
    config.publish_every = 2;
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, SimPlugin { config }));
    app.finish();
    app.cleanup();
    app
}

fn update_until(app: &mut App, what: &str, mut done: impl FnMut(&LatestSnapshot) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        app.update();
        if done(app.world().resource::<LatestSnapshot>()) {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("timed out waiting for {what}");
}

#[test]
fn snapshots_reach_the_world_and_commands_reach_the_brain() {
    let mut app = app();
    update_until(&mut app, "the brain to run", |l| l.0.as_deref().is_some_and(|s| s.phase == Phase::Running && s.step > 20));
    app.world().resource::<SimLink>().0.send(Command::DriveSpec { spec: "type:LPLC2".into(), amp: STIMULUS_AMP });
    update_until(&mut app, "the drive flag", |l| l.0.as_deref().is_some_and(|s| s.driving));
    let snap = app.world().resource::<LatestSnapshot>().0.as_deref().unwrap().clone();
    assert_eq!(snap.neurons, 4000);
    assert_eq!(snap.rates.len(), 4000);
}

#[test]
fn the_world_is_not_blocked_by_the_brain() {
    let mut app = app();
    update_until(&mut app, "the brain to run", |l| l.0.as_deref().is_some_and(|s| s.phase == Phase::Running));
    // 200 world updates must take far less than the 1 s the brain needs for 200 steps
    let t = Instant::now();
    for _ in 0..200 {
        app.update();
    }
    assert!(t.elapsed() < Duration::from_millis(500), "updates took {:?}", t.elapsed());
}
