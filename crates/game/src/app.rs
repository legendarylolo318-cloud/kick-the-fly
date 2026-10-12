//! The Bevy side: connect the sim thread to the world.

use bevy::prelude::*;

use crate::sim_thread::{SimConfig, SimHandle, spawn};
use crate::snapshot::Snapshot;

/// Starts the sim thread and keeps the newest [`Snapshot`] in [`LatestSnapshot`]. Needs no window or GPU, so tests and headless
/// tools use it with `MinimalPlugins`.
pub struct SimPlugin {
    pub config: SimConfig,
}

/// The render side of the sim thread.
#[derive(Resource)]
pub struct SimLink(pub SimHandle);

/// What the sim thread reported this frame (discoveries, saved states): drained once in `PreUpdate`, read by whoever cares.
#[derive(Resource, Default)]
pub struct SimEvents(pub Vec<crate::sim_thread::SimEvent>);

/// The newest snapshot seen so far (None until the thread has published one).
#[derive(Resource, Default)]
pub struct LatestSnapshot(pub Option<Box<Snapshot>>);

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SimLink(spawn(self.config.clone())))
            .init_resource::<LatestSnapshot>()
            .init_resource::<SimEvents>()
            .add_systems(PreUpdate, (poll_snapshot, drain_events));
    }
}

fn drain_events(link: Res<SimLink>, mut ev: ResMut<SimEvents>) {
    ev.0 = link.0.drain_events();
}

/// Swap in the newest snapshot, if one arrived, and return the old buffer to the sim thread. Never waits.
fn poll_snapshot(link: Res<SimLink>, mut latest: ResMut<LatestSnapshot>) {
    if let Some(new) = link.0.latest()
        && let Some(old) = latest.0.replace(new)
    {
        link.0.mailbox.recycle(old);
    }
}
