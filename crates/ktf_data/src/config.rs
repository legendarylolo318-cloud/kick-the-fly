//! `config.toml` (schema v3): the same keys, defaults, clamping, migrations and tolerance as Python's `core/config.py`, so a file the
//! Python game wrote loads here and a file written here loads in Python. The schema (the 75 settings with their tabs, tooltips,
//! options and bounds, the rebindable actions, the pad actions) is `data/schema.json`, dumped from the Python source by
//! `legacy-python/make_persistence_fixtures.py`, so the Settings menu and the file never disagree with the original.
//!
//! A missing, unreadable or corrupt file falls back to defaults with a warning (the bad file is kept as `config.toml.bad`), never
//! a crash. Unknown keys are ignored and out-of-range values are clamped, so an older or newer config still loads.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::{Number, Value};

pub const SCHEMA_VERSION: i64 = 3;
pub const SCHEMA_JSON: &str = include_str!("../data/schema.json");

#[derive(Debug, Clone, Deserialize)]
pub struct Setting {
    pub key: String,
    pub tab: String,
    pub label: String,
    pub kind: String,
    pub default: Value,
    pub tip: String,
    #[serde(default)]
    pub options: Vec<Value>,
    #[serde(default)]
    pub labels: Vec<String>,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
    pub tag: String,
    pub restart: bool,
    pub only: String,
    pub fmt: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Schema {
    pub schema_version: i64,
    pub settings: Vec<Setting>,
    /// (action, label, default key name as pygame spells it)
    pub actions: Vec<(String, String, String)>,
    pub pad_actions: Vec<(String, String, String)>,
    pub pad_axes: Vec<String>,
    pub pad_buttons: Vec<String>,
    pub reserved_keys: Vec<String>,
    pub overlap_ok: Vec<Vec<String>>,
    pub first_run: BTreeMap<String, bool>,
    pub tools: Vec<String>,
    pub max_saved_loadouts: usize,
    pub loadout: LoadoutSchema,
    /// `kick_the_fly.py:TOOLS`: (name, hotbar label, one-line hint)
    pub toolbar: Vec<(String, String, String)>,
}

/// `core/loadout.py` data: the catalog (labels, descriptions, tags), the presets and their texts.
#[derive(Debug, Clone, Deserialize)]
pub struct LoadoutSchema {
    pub catalog: Vec<ToolInfo>,
    pub tool_names: Vec<String>,
    pub presets: std::collections::BTreeMap<String, Vec<String>>,
    pub preset_labels: std::collections::BTreeMap<String, String>,
    pub preset_tips: std::collections::BTreeMap<String, String>,
    pub choices: Vec<String>,
    pub mode_default: std::collections::BTreeMap<String, String>,
    pub categories: Vec<String>,
    pub page_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub label: String,
    pub category: String,
    pub desc: String,
    pub neurons: String,
    /// "CONNECTOME" or "GAME RULE"
    pub tag: String,
    pub lab_only: bool,
    pub larva: bool,
    pub larva_note: String,
}

pub fn schema() -> &'static Schema {
    static S: OnceLock<Schema> = OnceLock::new();
    S.get_or_init(|| serde_json::from_str(SCHEMA_JSON).expect("data/schema.json is generated and valid"))
}

pub fn setting(key: &str) -> Option<&'static Setting> {
    schema().settings.iter().find(|s| s.key == key)
}

/// The order Python writes the `[first_run]` flags in (its dict order).
const FIRST_RUN_ORDER: [&str; 4] = ["tutorial_done", "loadout_notice", "neuron_shapes_asked", "whatsnew_3_0_seen"];

pub const TABS: [&str; 6] = ["Graphics", "Audio", "Brain", "Controls", "Accessibility", "Help"];

fn num(v: f64) -> Value {
    Value::Number(Number::from_f64(v).expect("finite"))
}
fn as_number(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite() && !v.is_boolean())
}

/// A valid value for this setting, or the reason it isn't (Python: `_coerce`).
pub fn coerce(s: &Setting, v: &Value) -> Result<Value, String> {
    match s.kind.as_str() {
        "bool" => v.as_bool().map(Value::Bool).ok_or_else(|| format!("{}: expected true/false", s.key)),
        "choice" => {
            for o in &s.options {
                let same = match (o.as_f64(), v.as_f64()) {
                    (Some(a), Some(b)) if !v.is_boolean() => a == b,
                    _ => o == v,
                };
                if same {
                    return Ok(o.clone());
                }
            }
            let opts: Vec<String> =
                s.options.iter().map(|o| if let Some(t) = o.as_str() { format!("'{t}'") } else { o.to_string() }).collect();
            Err(format!("{}: {} is not one of ({})", s.key, py_repr(v), opts.join(", ")))
        }
        "float" | "int" => {
            let Some(x) = as_number(v) else {
                return Err(format!("{}: expected a number", s.key));
            };
            let x = x.clamp(s.lo, s.hi);
            Ok(if s.kind == "int" { Value::Number(Number::from(x.round_ties_even() as i64)) } else { num(x) })
        }
        k => Err(k.to_string()),
    }
}

fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{s}'"),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        o => o.to_string(),
    }
}

/// A gamepad binding: SDL's standard axis/button names, a raw `axisN`/`buttonN` (N < 64), or "" (unbound). None = not a binding.
pub fn pad_binding(v: &Value) -> Option<String> {
    let v = v.as_str()?.trim().to_lowercase();
    let sc = schema();
    if sc.pad_axes.contains(&v) || sc.pad_buttons.contains(&v) {
        return Some(v);
    }
    for kind in ["axis", "button"] {
        if let Some(n) = v.strip_prefix(kind)
            && !n.is_empty()
            && n.bytes().all(|b| b.is_ascii_digit())
            && n.parse::<u64>().is_ok_and(|n| n < 64)
        {
            return Some(v);
        }
    }
    (v.is_empty()).then(String::new)
}

fn allowed_overlap(actions: &[&str]) -> bool {
    schema().overlap_ok.iter().any(|pair| actions.iter().all(|a| pair.iter().any(|p| p == a)))
}

#[derive(Debug, Clone, PartialEq)]
pub struct SavedLoadout {
    pub name: String,
    pub tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub path: Option<PathBuf>,
    pub values: BTreeMap<String, Value>,
    /// action → key name, in the schema's order
    pub keys: Vec<(String, String)>,
    pub pad: Vec<(String, String)>,
    pub warnings: Vec<String>,
    pub dirty: bool,
    pub custom_loadout: Vec<String>,
    pub saved_loadouts: Vec<SavedLoadout>,
    pub first_run: BTreeMap<String, bool>,
    pub migrated_from: Option<i64>,
    /// key → the saved value a command-line flag is overriding
    session: BTreeMap<String, Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Config {
    pub fn new(path: Option<PathBuf>) -> Self {
        let sc = schema();
        Config {
            path,
            values: sc.settings.iter().map(|s| (s.key.clone(), s.default.clone())).collect(),
            keys: sc.actions.iter().map(|(a, _, k)| (a.clone(), k.clone())).collect(),
            pad: sc.pad_actions.iter().map(|(a, _, b)| (a.clone(), b.clone())).collect(),
            warnings: vec![],
            dirty: false,
            custom_loadout: vec![],
            saved_loadouts: vec![],
            first_run: sc.first_run.clone(),
            migrated_from: None,
            session: BTreeMap::new(),
        }
    }

    // --- values -------------------------------------------------------------------------------------------------------------
    pub fn get(&self, key: &str) -> &Value {
        self.values.get(key).unwrap_or_else(|| panic!("no setting {key}"))
    }
    pub fn get_bool(&self, key: &str) -> bool {
        self.get(key).as_bool().unwrap_or(false)
    }
    pub fn get_f32(&self, key: &str) -> f32 {
        self.get(key).as_f64().unwrap_or(0.0) as f32
    }
    pub fn get_i64(&self, key: &str) -> i64 {
        self.get(key).as_i64().unwrap_or_else(|| self.get(key).as_f64().unwrap_or(0.0) as i64)
    }
    pub fn get_str(&self, key: &str) -> &str {
        self.get(key).as_str().unwrap_or("")
    }

    /// Validate and store. Returns whether the stored value changed.
    pub fn set(&mut self, key: &str, value: &Value) -> Result<bool, String> {
        let s = setting(key).ok_or_else(|| format!("no setting {key}"))?;
        let v = coerce(s, value)?;
        if self.values.get(key) == Some(&v) {
            return Ok(false);
        }
        self.values.insert(key.into(), v);
        self.session.remove(key);
        self.dirty = true;
        Ok(true)
    }

    /// A command-line flag (--autopilot, --arena, ...): in effect now, never written to config.toml.
    pub fn set_for_session(&mut self, key: &str, value: &Value) -> Result<(), String> {
        let s = setting(key).ok_or_else(|| format!("no setting {key}"))?;
        let v = coerce(s, value)?;
        self.session.entry(key.into()).or_insert_with(|| self.values.get(key).cloned().unwrap_or(Value::Null));
        self.values.insert(key.into(), v);
        Ok(())
    }

    pub fn reset_tab(&mut self, tab: &str) -> Vec<String> {
        let mut changed = vec![];
        for s in schema().settings.iter().filter(|s| s.tab == tab) {
            if self.values[&s.key] != s.default {
                self.values.insert(s.key.clone(), s.default.clone());
                changed.push(s.key.clone());
            }
        }
        if tab == "Controls" {
            let d: Vec<_> = schema().actions.iter().map(|(a, _, k)| (a.clone(), k.clone())).collect();
            if self.keys != d {
                self.keys = d;
                changed.push("keys".into());
            }
            let d: Vec<_> = schema().pad_actions.iter().map(|(a, _, b)| (a.clone(), b.clone())).collect();
            if self.pad != d {
                self.pad = d;
                changed.push("keys".into());
            }
        }
        self.dirty |= !changed.is_empty();
        changed
    }

    pub fn lab(&self) -> bool {
        self.get_str("brain.mode") == "lab"
    }
    pub fn pet(&self) -> bool {
        self.get_str("brain.mode") == "pet"
    }
    pub fn larva(&self) -> bool {
        self.get_str("brain.brain") == "larva"
    }
    pub fn tags_on(&self) -> bool {
        match self.get_str("brain.real_vs_rule") {
            "on" => true,
            "auto" => self.lab(),
            _ => false,
        }
    }

    // --- keys ---------------------------------------------------------------------------------------------------------------
    pub fn key_of(&self, action: &str) -> &str {
        self.keys.iter().find(|(a, _)| a == action).map_or("", |(_, k)| k.as_str())
    }
    fn set_key(&mut self, action: &str, key: &str) {
        if let Some(e) = self.keys.iter_mut().find(|(a, _)| a == action) {
            e.1 = key.into();
        }
    }
    pub fn action_for(&self, key_name: &str) -> Option<&str> {
        if key_name.is_empty() {
            return None;
        }
        self.keys.iter().find(|(_, k)| k == key_name).map(|(a, _)| a.as_str())
    }
    /// Every action bound to a key (D is walk-right and the Neurodex key at once by design).
    pub fn actions_for(&self, key_name: &str) -> Vec<String> {
        if key_name.is_empty() {
            return vec![];
        }
        self.keys.iter().filter(|(_, k)| k == key_name).map(|(a, _)| a.clone()).collect()
    }
    pub fn action_label(action: &str) -> &'static str {
        schema().actions.iter().find(|(a, _, _)| a == action).map_or("", |(_, l, _)| l.as_str())
    }

    /// key → the actions sharing it, for keys that clash (an allowed overlap and "unbound" don't count).
    pub fn conflicts(&self) -> Vec<(String, Vec<String>)> {
        let mut seen: Vec<(String, Vec<String>)> = vec![];
        for (a, k) in &self.keys {
            if k.is_empty() {
                continue;
            }
            match seen.iter_mut().find(|(x, _)| x == k) {
                Some(e) => e.1.push(a.clone()),
                None => seen.push((k.clone(), vec![a.clone()])),
            }
        }
        seen.into_iter().filter(|(_, v)| v.len() > 1 && !allowed_overlap(&v.iter().map(String::as_str).collect::<Vec<_>>())).collect()
    }

    /// Rebind an action. A key another action already uses swaps the two bindings. Returns (ok, message).
    pub fn bind(&mut self, action: &str, key_name: &str) -> (bool, String) {
        let key_name = key_name.to_lowercase();
        if schema().reserved_keys.contains(&key_name) {
            return (false, format!("'{key_name}' is reserved (Esc opens the menu)"));
        }
        let old = self.key_of(action).to_string();
        let displaced: Vec<String> =
            self.actions_for(&key_name).into_iter().filter(|a| a != action && !allowed_overlap(&[a, action])).collect();
        self.set_key(action, &key_name);
        self.dirty = true;
        let mut notes = vec![];
        for a in displaced {
            let holders: Vec<String> =
                if old.is_empty() { vec![] } else { self.actions_for(&old).into_iter().filter(|b| *b != a).collect() };
            if !old.is_empty() && holders.iter().all(|b| allowed_overlap(&[&a, b])) {
                self.set_key(&a, &old);
                notes.push(format!("'{key_name}' was used by {}; swapped, that is now '{old}'", Self::action_label(&a)));
            } else {
                self.set_key(&a, "");
                notes.push(format!("'{key_name}' was used by {}; that is now unbound (Settings > Controls)", Self::action_label(&a)));
            }
        }
        (true, notes.join("; "))
    }

    /// Rebind a gamepad action. A binding another action uses swaps the two. Returns (ok, message).
    pub fn bind_pad(&mut self, action: &str, binding: &str) -> (bool, String) {
        let Some(b) = pad_binding(&Value::String(binding.into())) else {
            return (false, format!("'{binding}' isn't a gamepad button or axis"));
        };
        let other = self.pad.iter().find(|(a, x)| *x == b && a != action && !b.is_empty()).map(|(a, _)| a.clone());
        let old = self.pad.iter().find(|(a, _)| a == action).map(|(_, x)| x.clone()).unwrap_or_default();
        if let Some(e) = self.pad.iter_mut().find(|(a, _)| a == action) {
            e.1 = b.clone();
        }
        self.dirty = true;
        if let Some(o) = other {
            if let Some(e) = self.pad.iter_mut().find(|(a, _)| *a == o) {
                e.1 = old.clone();
            }
            let shown = if old.is_empty() { "unbound" } else { old.as_str() };
            return (true, format!("{b} was used by {o}; swapped, that is now {shown}"));
        }
        (true, String::new())
    }

    // --- loadouts -----------------------------------------------------------------------------------------------------------
    pub fn set_custom_loadout(&mut self, tools: &[String]) {
        self.custom_loadout = tools.to_vec();
        self.dirty = true;
    }

    /// Store a named loadout (replacing one of the same name). False if all slots are taken.
    pub fn save_loadout(&mut self, name: &str, tools: &[String]) -> bool {
        let name: String = name.trim().chars().take(24).collect();
        let name = if name.is_empty() { "Loadout".to_string() } else { name };
        if let Some(i) = self.saved_loadouts.iter_mut().find(|i| i.name == name) {
            i.tools = tools.to_vec();
            self.dirty = true;
            return true;
        }
        if self.saved_loadouts.len() >= schema().max_saved_loadouts {
            return false;
        }
        self.saved_loadouts.push(SavedLoadout { name, tools: tools.to_vec() });
        self.dirty = true;
        true
    }

    pub fn delete_loadout(&mut self, name: &str) {
        self.saved_loadouts.retain(|i| i.name != name);
        self.dirty = true;
    }

    // --- files --------------------------------------------------------------------------------------------------------------
    /// Load `path`. Never fails: problems become `warnings` and defaults.
    pub fn load(path: &Path) -> Config {
        let mut cfg = Config::new(Some(path.to_path_buf()));
        if !path.exists() {
            // a fresh install has no "what's new": the tutorial is its welcome
            cfg.first_run.insert("whatsnew_3_0_seen".into(), true);
            return cfg;
        }
        let parsed = std::fs::read(path)
            .map_err(|e| e.to_string())
            .and_then(|b| String::from_utf8(b).map_err(|e| e.to_string()))
            .and_then(|t| toml::from_str::<Value>(&t).map_err(|e| e.to_string()));
        match parsed {
            Ok(data) => cfg.apply(&data),
            Err(e) => {
                cfg.warnings.push(format!("config.toml could not be read ({e}); using default settings"));
                let bad = path.with_file_name(format!("{}.bad", path.file_name().unwrap_or_default().to_string_lossy()));
                if std::fs::rename(path, &bad).is_ok() {
                    cfg.warnings.push(format!("the unreadable file was kept as {}", bad.file_name().unwrap_or_default().to_string_lossy()));
                }
            }
        }
        cfg
    }

    /// Load from TOML text (tests, share previews).
    pub fn from_toml(text: &str) -> Result<Config, String> {
        let data: Value = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut cfg = Config::new(None);
        cfg.apply(&data);
        Ok(cfg)
    }

    fn apply(&mut self, data: &Value) {
        for s in &schema().settings {
            let (section, name) = s.key.split_once('.').expect("section.name");
            if let Some(v) = data.get(section).and_then(|d| d.as_object()).and_then(|d| d.get(name)) {
                match coerce(s, v) {
                    Ok(v) => {
                        self.values.insert(s.key.clone(), v);
                    }
                    Err(e) => self.warnings.push(format!("{e}; using the default")),
                }
            }
        }
        if let Some(pad) = data.get("gamepad").and_then(|p| p.as_object()) {
            for (a, _, _) in &schema().pad_actions {
                if let Some(b) = pad.get(a).and_then(pad_binding)
                    && let Some(e) = self.pad.iter_mut().find(|(x, _)| x == a)
                {
                    e.1 = b;
                }
            }
            // an action newer than this file whose default is a button the player gave to another action: theirs wins
            let mine: Vec<String> = self.pad.iter().filter(|(a, b)| pad.contains_key(a) && !b.is_empty()).map(|(_, b)| b.clone()).collect();
            for i in 0..self.pad.len() {
                let (a, b) = self.pad[i].clone();
                if !pad.contains_key(&a) && mine.contains(&b) {
                    self.warnings.push(format!("gamepad '{b}' is yours for another action; {a} is unbound"));
                    self.pad[i].1.clear();
                }
            }
        }
        if let Some(keys) = data.get("keys").and_then(|k| k.as_object()) {
            for (a, _, _) in &schema().actions {
                if let Some(k) = keys.get(a).and_then(|k| k.as_str()) {
                    let k = k.to_lowercase();
                    if !schema().reserved_keys.contains(&k) {
                        self.set_key(a, &k);
                    }
                }
            }
            for (k, acts) in self.conflicts() {
                let saved: Vec<&String> = acts.iter().filter(|a| keys.contains_key(*a)).collect();
                if !saved.is_empty() && saved.len() < acts.len() {
                    // a default of an action newer than this file on a key the player chose for something else: unbound
                    let names: Vec<&str> = saved.iter().map(|s| s.as_str()).collect();
                    let lost: Vec<String> = acts.iter().filter(|a| !keys.contains_key(*a)).cloned().collect();
                    for a in lost {
                        self.set_key(&a, "");
                        self.warnings.push(format!(
                            "key '{k}' is yours for {}; {} is unbound (Settings > Controls)",
                            names.join(", "),
                            Self::action_label(&a)
                        ));
                    }
                }
            }
            for (k, acts) in self.conflicts() {
                // a hand-edited file bound one key twice: keep the first
                for a in &acts[1..] {
                    let default = schema().actions.iter().find(|(x, _, _)| x == a).map(|(_, _, d)| d.clone()).unwrap_or_default();
                    let taken = self.actions_for(&default).into_iter().any(|b| b != *a && !allowed_overlap(&[a, &b]));
                    let new = if default == k || taken { String::new() } else { default };
                    self.set_key(a, &new);
                    self.warnings.push(format!(
                        "key '{k}' was bound twice; {a} {}",
                        if new.is_empty() { "is unbound (Settings > Controls)" } else { "reset to its default" }
                    ));
                }
            }
        }
        let schema_v = data.get("schema_version").and_then(|v| v.as_i64()).unwrap_or(1);
        self.load_loadout(data, schema_v);
        if schema_v < 2 && self.values.get("brain.science_popups") == Some(&Value::Bool(true)) {
            self.values.insert("brain.science_popups".into(), Value::Bool(false));
            self.dirty = true;
            self.warnings.push("migrated brain.science_popups to off for schema 2".into());
        }
    }

    fn load_loadout(&mut self, data: &Value, schema_v: i64) {
        let tools = &schema().tools;
        let names = |v: Option<&Value>| -> Vec<String> {
            v.and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|n| n.as_str())
                .filter(|n| tools.iter().any(|t| t == n))
                .map(String::from)
                .collect()
        };
        if let Some(sec) = data.get("loadout").and_then(|s| s.as_object()) {
            self.custom_loadout = names(sec.get("custom"));
            for item in sec.get("saved").and_then(|s| s.as_array()).into_iter().flatten().take(schema().max_saved_loadouts) {
                let name =
                    item.get("name").and_then(|n| n.as_str()).map(|n| crate::sharecode::clean_name(n, usize::MAX)).unwrap_or_default();
                if !name.is_empty() {
                    self.saved_loadouts.push(SavedLoadout { name: name.chars().take(24).collect(), tools: names(item.get("tools")) });
                }
            }
        }
        if let Some(fr) = data.get("first_run").and_then(|f| f.as_object()) {
            for k in schema().first_run.keys() {
                if let Some(b) = fr.get(k).and_then(|b| b.as_bool()) {
                    self.first_run.insert(k.clone(), b);
                }
            }
        }
        if schema_v < 3 {
            self.migrated_from = Some(schema_v);
            let has_preset = data.get("controls").and_then(|c| c.get("loadout_preset")).is_some();
            if self.get_str("controls.loadout_preset") == "auto" && !has_preset {
                self.values.insert("controls.loadout_preset".into(), Value::String("all".into()));
            }
            self.first_run.insert("loadout_notice".into(), true);
            self.first_run.insert("tutorial_done".into(), true);
            self.dirty = true;
            self.warnings.push(
                "migrated to schema 3: hotbar loadout set to All so keys 1-9, 0, - and = still reach every tool (Settings > Controls > Tool loadout, or the editor on Q)"
                    .into(),
            );
        }
    }

    /// The file's text, in the same layout Python writes. `version` goes in the first comment line.
    pub fn to_toml(&self, version: &str) -> String {
        let mut out = vec![
            format!("# Kick the Fly {version} settings. Edit in the game (Esc > Settings) or by hand."),
            format!("schema_version = {SCHEMA_VERSION}"),
            String::new(),
        ];
        let mut sections: Vec<(&str, Vec<String>)> = vec![];
        for s in &schema().settings {
            let (section, name) = s.key.split_once('.').expect("section.name");
            let v = self.session.get(&s.key).unwrap_or(&self.values[&s.key]);
            let line = format!("{name} = {}", toml_value(v));
            match sections.iter_mut().find(|(n, _)| *n == section) {
                Some(e) => e.1.push(line),
                None => sections.push((section, vec![line])),
            }
        }
        for (section, lines) in sections {
            out.push(format!("[{section}]"));
            out.extend(lines);
            out.push(String::new());
        }
        out.push("[keys]".into());
        out.extend(self.keys.iter().map(|(a, k)| format!("{a} = {}", toml_str(k))));
        out.push(String::new());
        out.push("[gamepad]".into());
        out.extend(self.pad.iter().map(|(a, b)| format!("{a} = {}", toml_str(b))));
        out.extend(["".into(), "[loadout]".into(), format!("custom = {}", toml_list(&self.custom_loadout))]);
        for item in &self.saved_loadouts {
            out.extend([
                "".into(),
                "[[loadout.saved]]".into(),
                format!("name = {}", toml_str(&item.name)),
                format!("tools = {}", toml_list(&item.tools)),
            ]);
        }
        out.extend(["".into(), "[first_run]".into()]);
        out.extend(FIRST_RUN_ORDER.iter().map(|k| format!("{k} = {}", self.first_run[*k])));
        out.join("\n") + "\n"
    }

    /// Write the file atomically (temp file, then rename). Returns whether it worked.
    pub fn save(&mut self, version: &str) -> bool {
        let Some(path) = self.path.clone() else {
            return false;
        };
        let ok = (|| -> std::io::Result<()> {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            let tmp = path.with_file_name(format!("{}.tmp", path.file_name().unwrap_or_default().to_string_lossy()));
            std::fs::write(&tmp, self.to_toml(version))?;
            std::fs::rename(&tmp, &path)
        })();
        match ok {
            Ok(()) => {
                self.dirty = false;
                true
            }
            Err(e) => {
                eprintln!("could not save settings to {}: {e}", path.display());
                false
            }
        }
    }
}

fn toml_value(v: &Value) -> String {
    match v {
        Value::Bool(b) => b.to_string(),
        Value::Number(n) if n.is_f64() => format!("{:?}", n.as_f64().unwrap()),
        Value::Number(n) => n.to_string(),
        Value::String(s) => toml_str(s),
        other => toml_str(&other.to_string()),
    }
}

fn toml_str(s: &str) -> String {
    let s = s.replace('\\', "\\\\").replace('"', "\\\"");
    let mut out = String::from("\"");
    for ch in s.chars() {
        if (ch as u32) < 0x20 || ch as u32 == 0x7F {
            out.push_str(&format!("\\u{:04x}", ch as u32));
        } else {
            out.push(ch);
        }
    }
    out.push('"');
    out
}

fn toml_list(items: &[String]) -> String {
    format!("[{}]", items.iter().map(|s| toml_str(s)).collect::<Vec<_>>().join(", "))
}
