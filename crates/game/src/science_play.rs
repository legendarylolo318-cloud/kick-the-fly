//! "Real flies do this too" (Python: `Game.on_reaction`, `_draw_science_card`, `validation.passing_events`).
//!
//! In Play and Pet mode, with Settings > Brain > Real-science popups on (it is off by default), the first time the fly shows a behavior
//! that a validation test reproduces on this brain pack (DODGE, PROBOSCIS, GROOM, AVOID, SONG), a card says so in the test's own
//! words for 12 seconds, or until it is clicked. Each behavior gets its card once: the list is kept in `science_seen.json` in the data
//! folder, as Python keeps it. Nothing shows during a challenge, in Lab mode, or when the validation results were made with another
//! brain pack or with changed Lab parameters.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use bevy::prelude::*;
use serde_json::Value;

use crate::app::LatestSnapshot;
use crate::fly::ReactionFired;
use crate::menu_ui::UserConfig;

/// How long the card stays (`now - t0 > 12.0`).
pub const CARD_S: f32 = 12.0;

#[derive(Resource, Default)]
pub struct Science {
    seen: Option<BTreeSet<String>>,
    /// The test's `play` sentence and when the card came up (game clock).
    pub card: Option<(String, f32)>,
}

fn seen_file() -> PathBuf {
    ktf_data::paths::get().data_dir.join("science_seen.json")
}

/// `Game._load_seen`: a missing or damaged file is an empty list.
pub fn load_seen(path: &std::path::Path) -> BTreeSet<String> {
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str::<BTreeSet<String>>(&t).ok()).unwrap_or_default()
}

/// `validation.passing_events`: popup event -> the `play` sentence of the test, for tests that passed on this brain pack with the Lab's
/// default parameters.
pub fn passing_events(res: &Value, n_neurons: u64, synapses: u64) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if res["n_neurons"].as_u64() != Some(n_neurons) || res["synapses"].as_u64() != Some(synapses) {
        return out;
    }
    let changed = ktf_data::labparams::params().iter().any(|p| {
        let v = res["lab_params"].get(&p.name).and_then(Value::as_f64).unwrap_or(p.default);
        (v - p.default).abs() > 1e-9
    });
    if changed {
        return out;
    }
    for t in res["tests"].as_array().into_iter().flatten() {
        if let (Some(true), Some(ev), Some(play)) = (t["passed"].as_bool(), t["popup_event"].as_str(), t["play"].as_str())
            && !ev.is_empty()
        {
            out.insert(ev.to_string(), play.to_string());
        }
    }
    out
}

/// Looks at the reactions that fired. It holds no `LabRuns` borrow of its own (that resource is busy in the menu systems every frame, and a
/// mutable borrow here kept the scheduler from overlapping them): the rare card lookup runs as a queued command instead.
fn watch(
    mut fired: MessageReader<ReactionFired>,
    mut sci: ResMut<Science>,
    cfg: Res<UserConfig>,
    snap: Res<LatestSnapshot>,
    mut commands: Commands,
    virt: Res<Time<Virtual>>,
) {
    let now = virt.elapsed_secs();
    let lab = cfg.0.lab();
    // the card goes after 12 s, and at once in Lab mode
    if sci.card.as_ref().is_some_and(|c| now - c.1 > CARD_S || lab) {
        sci.card = None;
    }
    let kinds: Vec<String> = fired.read().map(|f| f.name.clone()).collect();
    if kinds.is_empty() || lab || !cfg.0.get_bool("brain.science_popups") {
        return;
    }
    // challenges hear the reaction instead; popups are for normal play only
    if snap.0.as_deref().is_some_and(|s| s.challenge.is_some()) {
        return;
    }
    let seen = sci.seen.get_or_insert_with(|| load_seen(&seen_file())).clone();
    let Some(kind) = kinds.into_iter().find(|k| !seen.contains(k)) else { return };
    commands.queue(move |world: &mut World| show_card(world, kind, now));
}

fn show_card(world: &mut World, kind: String, now: f32) {
    let Some(sh) = world.resource::<crate::app::SimLink>().0.shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    else {
        return;
    };
    let play = {
        let mut runs = world.resource_mut::<crate::lab_play::LabRuns>();
        runs.refresh();
        passing_events(runs.results(), sh.weights.n() as u64, sh.weights.csr.nnz() as u64).remove(&kind)
    };
    let Some(play) = play else { return };
    let mut sci = world.resource_mut::<Science>();
    sci.card = Some((play, now));
    if let Some(s) = sci.seen.as_mut() {
        s.insert(kind);
        let path = seen_file();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, serde_json::to_string(s).unwrap_or_default());
    }
}

pub struct SciencePlugin;

impl Plugin for SciencePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Science>().add_systems(Update, watch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_tests_that_passed_on_this_pack_with_default_parameters_speak() {
        let res = json!({"n_neurons": 10, "synapses": 20, "lab_params": {"noise_std": ktf_data::labparams::by_name("noise_std").unwrap().default},
            "tests": [
                {"id": "a", "passed": true, "popup_event": "DODGE", "play": "Loom at it and it dodges."},
                {"id": "b", "passed": false, "popup_event": "SONG", "play": "x"},
                {"id": "c", "passed": true, "popup_event": "", "play": "y"},
                {"id": "d", "passed": true, "play": "z"}]});
        let ev = passing_events(&res, 10, 20);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev["DODGE"], "Loom at it and it dodges.");
        // another pack, or a changed model parameter: nothing
        assert!(passing_events(&res, 11, 20).is_empty());
        assert!(passing_events(&res, 10, 21).is_empty());
        let mut changed = res.clone();
        changed["lab_params"]["noise_std"] = json!(0.9);
        assert!(passing_events(&changed, 10, 20).is_empty());
        assert!(passing_events(&Value::Null, 10, 20).is_empty());
    }

    #[test]
    fn the_seen_list_is_pythons_file() {
        let d = std::env::temp_dir().join(format!("ktf-science-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("science_seen.json");
        assert!(load_seen(&p).is_empty());
        std::fs::write(&p, r#"["DODGE", "SONG"]"#).unwrap();
        assert_eq!(load_seen(&p).into_iter().collect::<Vec<_>>(), ["DODGE", "SONG"]);
        std::fs::write(&p, "not json").unwrap();
        assert!(load_seen(&p).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }
}
