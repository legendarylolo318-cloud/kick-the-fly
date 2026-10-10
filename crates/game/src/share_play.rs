//! Applying a share code in the game (`core/sharecode.py:apply`): the share page validates and previews, the game does the work.
//!
//! surgery -> the brain surgery switches (`SurgeryClear`, the group entries by label, the cell types); loadout is applied by the page through
//! the config; challenge -> seed, arena, surgery, then the challenge starts; contraption -> the first free of the 8 slots of `contraptions.json`;
//! protocol -> `<data>/protocols/<name>.yaml` (JSON text, which YAML reads; the Lab's protocol runner is not in this build yet); lab parameters
//! are refused until the Lab's parameters page exists.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use bevy::prelude::*;
use serde_json::{Value, json};

use crate::app::SimLink;
use crate::config::SwitchArena;
use crate::menu_ui::{MenuFlash, UserConfig};
use crate::sim_thread::Command;

/// `(kind, payload)` of codes applied from the share page, waiting for the game.
#[derive(Resource, Default, Clone)]
pub struct ShareQueue(pub Arc<Mutex<Vec<(String, Value)>>>);

/// The Lab's current parameter values by name (only those changed from their defaults are kept).
#[derive(Resource, Default, Clone)]
pub struct LabState {
    pub values: std::collections::BTreeMap<String, f64>,
}

impl LabState {
    pub fn json(&self) -> Value {
        Value::Object(self.values.iter().map(|(k, v)| (k.clone(), json!(v))).collect())
    }
}

/// `lab.apply_to_sim` for flies spawned or restored after a parameter was changed.
pub fn send_model_params(link: &SimLink, lab: &LabState) {
    for (name, value) in &lab.values {
        if ktf_data::labparams::by_name(name).is_some_and(|p| p.kind == "model") {
            link.0.send(Command::SetModelParam { name: name.clone(), value: *value });
        }
    }
}

fn surgery_switches(p: &Value) -> usize {
    p["groups"].as_object().map_or(0, |g| g.values().filter(|m| m.as_i64().unwrap_or(0) != 0).count())
        + p["types"].as_object().map_or(0, |g| g.values().filter(|m| m.as_i64().unwrap_or(0) != 0).count())
}

fn contraptions_path() -> PathBuf {
    ktf_data::paths::get().data_dir.join("contraptions.json")
}

fn load_slots() -> Vec<Value> {
    let mut slots: Vec<Value> = std::fs::read_to_string(contraptions_path())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    slots.truncate(8);
    slots.resize(8, Value::Null);
    slots
}

/// The result lines `sharecode.apply` returns, or why it cannot be applied here.
pub fn lines_for(kind: &Value, p: &Value) -> Result<Vec<String>, String> {
    match kind.as_str().unwrap_or("") {
        "surgery" => Ok(vec![format!("surgery set ({} switches)", surgery_switches(p))]),
        "challenge" => {
            let mut lines = Vec::new();
            if let Some(s) = p.get("seed") {
                lines.push(format!("seed {s}"));
            }
            if let Some(a) = p.get("arena").and_then(Value::as_str) {
                lines.push(format!("arena {a}"));
            }
            if let Some(s) = p.get("surgery") {
                lines.push(format!("surgery set ({} switches)", surgery_switches(s)));
            }
            if p.get("params").is_some() {
                return Err("this challenge code sets Lab parameters, which are not part of this build yet".into());
            }
            lines.push(format!("started {}", p["challenge"].as_str().unwrap_or("?")));
            Ok(lines)
        }
        "contraption" => match load_slots().iter().position(Value::is_null) {
            Some(i) => Ok(vec![format!("contraption saved in slot {}", i + 1)]),
            None => Ok(vec!["all 8 contraption slots are full: not saved".into()]),
        },
        "protocol" => {
            let name = p["protocol"]["name"].as_str().unwrap_or("shared-protocol");
            Ok(vec![format!("protocol saved as {}.yaml", stem(name))])
        }
        "lab" => Err("Lab parameters are not part of this build yet".into()),
        other => Err(format!("this build can't apply a {other} code yet")),
    }
}

fn stem(name: &str) -> String {
    // re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    let mut s = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || "._-".contains(c) {
            s.push(c);
        } else if !s.ends_with('\u{1}') {
            s.push('\u{1}');
        }
    }
    let s = s.replace('\u{1}', "-");
    let s = s.trim_matches(|c| c == '-' || c == '.');
    let s: String = s.chars().take(40).collect();
    if s.is_empty() { "shared-protocol".into() } else { s }
}

fn apply_surgery(link: &SimLink, p: &Value) {
    link.0.send(Command::SurgeryClear);
    if let Some(groups) = p["groups"].as_object() {
        for (label, m) in groups {
            let mode = m.as_i64().unwrap_or(0) as i8;
            if mode == 0 {
                continue;
            }
            if let Some(index) = fly_sim::surgery::ENTRIES.iter().position(|e| e.0 == label) {
                link.0.send(Command::SurgeryEntry { index, mode });
            }
        }
    }
    if let Some(types) = p["types"].as_object() {
        for (ty, m) in types {
            let mode = m.as_i64().unwrap_or(0) as i8;
            if mode != 0 {
                link.0.send(Command::SurgeryType { ty: ty.clone(), mode });
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_queue(
    queue: Res<ShareQueue>,
    link: Res<SimLink>,
    mut cfg: ResMut<UserConfig>,
    mut switch: MessageWriter<SwitchArena>,
    mut chal: ResMut<crate::challenges::Challenges>,
    mut flash: MessageWriter<MenuFlash>,
    mut lab: ResMut<LabState>,
) {
    let todo: Vec<(String, Value)> = std::mem::take(&mut *queue.0.lock().unwrap_or_else(PoisonError::into_inner));
    for (kind, p) in todo {
        match kind.as_str() {
            "lab_param" => {
                let name = p["name"].as_str().unwrap_or("");
                if let Some(v) = ktf_data::labparams::clamp(name, p["value"].as_f64().unwrap_or(0.0)) {
                    let spec = ktf_data::labparams::by_name(name).expect("clamp found it");
                    if (v - spec.default).abs() > 1e-9 {
                        lab.values.insert(name.to_string(), v);
                    } else {
                        lab.values.remove(name);
                    }
                    if spec.kind == "model" {
                        link.0.send(Command::SetModelParam { name: name.to_string(), value: v });
                    } else {
                        crate::reactions::set_rule(name, v as f32);
                    }
                }
                continue; // no flash for a slider drag
            }
            "surgery" => apply_surgery(&link, &p),
            "challenge" => {
                if let Some(seed) = p.get("seed").and_then(Value::as_i64) {
                    let _ = cfg.0.set("brain.seed", &json!(seed));
                    cfg.0.save(env!("CARGO_PKG_VERSION"));
                }
                if let Some(i) = p.get("arena").and_then(Value::as_str).and_then(|a| crate::world::ARENAS.iter().position(|x| *x == a)) {
                    switch.write(SwitchArena(i));
                }
                if let Some(s) = p.get("surgery") {
                    apply_surgery(&link, s);
                }
                if let Some(key) = p["challenge"].as_str() {
                    chal.request = Some(key.to_string());
                }
            }
            "contraption" => {
                let mut slots = load_slots();
                if let Some(i) = slots.iter().position(Value::is_null) {
                    let name: String = p["name"].as_str().unwrap_or("shared").chars().take(24).collect();
                    slots[i] = json!({"name": name, "parts": p["parts"]});
                    let path = contraptions_path();
                    if let Some(d) = path.parent() {
                        let _ = std::fs::create_dir_all(d);
                    }
                    let _ = std::fs::write(&path, serde_json::to_string_pretty(&slots).unwrap_or_default());
                }
            }
            "protocol" => {
                let pr = &p["protocol"];
                let folder = ktf_data::paths::get().data_dir.join("protocols");
                let _ = std::fs::create_dir_all(&folder);
                let base = stem(pr["name"].as_str().unwrap_or("shared-protocol"));
                let (mut path, mut k) = (folder.join(format!("{base}.yaml")), 2);
                while path.exists() {
                    path = folder.join(format!("{base}-{k}.yaml"));
                    k += 1;
                }
                // JSON is YAML: Python's `yaml.safe_load` reads this file
                let _ = std::fs::write(&path, serde_json::to_string_pretty(pr).unwrap_or_default());
            }
            _ => {}
        }
        flash.write(MenuFlash(format!("SHARE    imported {kind} code"), true));
    }
}

pub struct SharePlugin;

impl Plugin for SharePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShareQueue>().init_resource::<LabState>().add_systems(Update, apply_queue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_lines_say_what_was_done() {
        let p = json!({"groups": {"Optic lobes": -1, "Antennae": 0}, "types": {"MDN": 1}});
        assert_eq!(lines_for(&json!("surgery"), &p).unwrap(), vec!["surgery set (2 switches)"]);
        let c = json!({"challenge": "tmaze", "seed": 7, "arena": "pool"});
        assert_eq!(lines_for(&json!("challenge"), &c).unwrap(), vec!["seed 7", "arena pool", "started tmaze"]);
        assert!(lines_for(&json!("lab"), &json!({})).is_err());
        assert_eq!(stem("My protocol: v2!"), "My-protocol-v2");
        assert_eq!(stem("!!"), "shared-protocol");
    }
}
