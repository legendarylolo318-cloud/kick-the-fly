//! The brain stethoscope (`K`, Python `update_stethoscope_target`, `play_spike_click`): the spikes of the chosen neurons as extracellular
//! clicks. The brain thread counts them every step (a snapshot every 4th step would lose some); this module sends the target and plays a
//! click per frame whose length says how many spikes there were.

use std::sync::atomic::Ordering;

use bevy::prelude::*;

use crate::actions::Actions;
use crate::app::SimLink;
use crate::feedback::{Notes, Sfx, Tag};
use crate::menu_ui::UserConfig;
use crate::sim_thread::Command;

#[derive(Default)]
struct Sent {
    enabled: bool,
    target: String,
}

/// K switches it on and off and says what it listens to.
fn toggle(act: Res<Actions>, mut cfg: ResMut<UserConfig>, mut notes: ResMut<Notes>, time: Res<Time<Real>>) {
    if act.just("stethoscope") {
        let on = !cfg.0.get_bool("audio.stethoscope_enabled");
        cfg.set_and_save("audio.stethoscope_enabled", serde_json::json!(on));
        let target = cfg.0.get_str("audio.stethoscope_target").replace('_', " ");
        let mut chars = target.chars();
        let label = chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default();
        notes.push(time.elapsed_secs(), format!("STETHOSCOPE {} ({label})", if on { "ON" } else { "OFF" }), Tag::GameRule);
    }
}

fn sync(cfg: Res<UserConfig>, link: Res<SimLink>, mut sent: Local<Sent>) {
    let enabled = cfg.0.get_bool("audio.stethoscope_enabled");
    let target = cfg.0.get_str("audio.stethoscope_target").to_string();
    if enabled != sent.enabled || (enabled && target != sent.target) {
        link.0.send(Command::Stethoscope { target: enabled.then(|| target.clone()), region: None });
        sent.enabled = enabled;
        sent.target = target;
    }
}

/// `Sound.play_spike_click`: 1 spike, up to 4, up to 12, or a burst.
fn clicks(cfg: Res<UserConfig>, link: Res<SimLink>, mut sfx: MessageWriter<Sfx>) {
    let n = link.0.steth.swap(0, Ordering::Relaxed);
    if n == 0 || !cfg.0.get_bool("audio.stethoscope_enabled") {
        return;
    }
    let vol = cfg.0.get_f32("audio.stethoscope_vol").clamp(0.0, 1.0);
    if vol <= 0.001 {
        return;
    }
    let name = match n {
        1 => "spike_click1",
        2..=4 => "spike_click2",
        5..=12 => "spike_click3",
        _ => "spike_click_many",
    };
    sfx.write(Sfx { name, volume: vol });
}

pub struct StethoscopePlugin;

impl Plugin for StethoscopePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (toggle, sync, clicks).chain());
    }
}
