//! What a launch shows first (`Game.show_first_run_notices`): one of the loadout notice (a config migrated from before 2.13), the tutorial, What's
//! New (once after an upgrade), the question about downloading real neuron shapes, or else the Neuron of the Day card. Decided once, from the run
//! loop (never from setup, so tests and headless runs never see it), and written to `config.toml`'s `[first_run]` flags when shown.

use bevy::prelude::*;

use crate::menu_ui::UserConfig;

/// The one thing this launch opens with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Launch {
    LoadoutNotice,
    Tutorial,
    WhatsNew,
    NeuronShapes,
    NeuronOfDay,
}

/// `None` until decided (a second after the first frame).
#[derive(Resource, Default)]
pub struct FirstRun(pub Option<Launch>);

fn decide(real: Res<Time<Real>>, mut cfg: ResMut<UserConfig>, mut fr: ResMut<FirstRun>) {
    if fr.0.is_some() || real.elapsed_secs() < 1.0 {
        return;
    }
    let flag = |c: &UserConfig, k: &str| c.0.first_run.get(k).copied().unwrap_or(false);
    let set = |c: &mut UserConfig, k: &str| {
        c.0.first_run.insert(k.into(), true);
        c.0.dirty = true;
        c.0.save(env!("CARGO_PKG_VERSION"));
    };
    let choice = if flag(&cfg, "loadout_notice") {
        // shown once, then cleared (`Game.show_loadout_notice`)
        cfg.0.first_run.insert("loadout_notice".into(), false);
        cfg.0.dirty = true;
        cfg.0.save(env!("CARGO_PKG_VERSION"));
        Launch::LoadoutNotice
    } else if !flag(&cfg, "tutorial_done") {
        Launch::Tutorial
    } else if !flag(&cfg, "whatsnew_3_0_seen") {
        set(&mut cfg, "whatsnew_3_0_seen");
        Launch::WhatsNew
    } else if !flag(&cfg, "neuron_shapes_asked") && !cfg.0.get_bool("brain.neuron_shapes") {
        set(&mut cfg, "neuron_shapes_asked");
        Launch::NeuronShapes
    } else {
        Launch::NeuronOfDay
    };
    fr.0 = Some(choice);
}

pub struct FirstRunPlugin;

impl Plugin for FirstRunPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FirstRun>().add_systems(Update, decide);
    }
}
