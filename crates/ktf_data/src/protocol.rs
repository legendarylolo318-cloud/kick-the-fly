//! Protocol files (`lab/protocol.py`): an experiment described in YAML, checked before anything runs.
//!
//! Kinds: a stimulus protocol (drive neurons on a schedule and record spikes), an assay protocol, and the stand-alone `patch`, `rig` and
//! `screen` blocks; a stimulus protocol may add `thermogenetics`, `drug`, `imaging`, `weather`, `audio` and `predator`. A lecture
//! (`classroom: true`) is loaded and checked but is not runnable. [`check`] returns the normalised protocol (seeds expanded, numbers made
//! numbers) or the reason; the messages are Python's.

use std::path::Path;

use serde_json::{Map, Value, json};

use fly_sim::sensitivity::fmt_gp;

#[derive(Debug, Clone, PartialEq)]
pub struct ProtocolError(pub String);

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProtocolError {}

fn err<T>(s: String) -> Result<T, ProtocolError> {
    Err(ProtocolError(s))
}

pub const MAX_FLIES: i64 = 10_000;
const TOP_KEYS: [&str; 28] = [
    "name",
    "description",
    "title",
    "classroom",
    "steps",
    "seed",
    "seeds",
    "flies",
    "warmup_s",
    "duration_s",
    "params",
    "surgery",
    "control",
    "stimuli",
    "recordings",
    "assay",
    "assay_options",
    "workers",
    "nwb",
    "thermogenetics",
    "drug",
    "imaging",
    "patch",
    "weather",
    "audio",
    "predator",
    "rig",
    "screen",
];
const STIM_KEYS: [&str; 8] = ["at_s", "for_s", "target", "strength", "recruit", "mode", "amp", "side"];
const ASSAYS: [&str; 8] = ["tmaze", "looming", "sugar", "orchard", "thermo_escape", "predator_escape", "hum_demo", "sleep_deprivation"];
const DAY2_KEYS: [&str; 4] = ["thermogenetics", "drug", "imaging", "patch"];
const DAY3_KEYS: [&str; 3] = ["weather", "audio", "predator"];
const RIG_KEYS: [&str; 8] = ["name", "mode", "seconds", "omega", "gain", "scene", "bar_deg", "individuality"];
const SCREEN_KEYS: [&str; 10] =
    ["kind", "controls", "min_neurons", "max_types", "types", "behaviors", "top", "candidate_batch", "batch", "backend"];
const PREDATORS: [&str; 3] = ["frog", "dragonfly", "mantis"];
const HUM_CONDITIONS: [&str; 6] = ["silence", "steady_200", "pulses_200_ipi35", "pulses_200_ipi70", "steady_50", "steady_600"];

/// `{x:g}`.
fn g(x: f64) -> String {
    fmt_gp(x, 6)
}

fn py_list(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", "))
}

fn is_num(v: &Value) -> bool {
    v.is_number()
}

fn num(v: &Value, lo: f64, hi: f64, what: &str, where_: &str, allow_none: bool) -> Result<Option<f64>, ProtocolError> {
    if v.is_null() && allow_none {
        return Ok(None);
    }
    match v.as_f64() {
        Some(x) if x.is_finite() && lo <= x && x <= hi => Ok(Some(x)),
        _ => err(format!("{where_}: {what} must be a number from {} to {}", g(lo), g(hi))),
    }
}

fn f(v: f64) -> Value {
    json!(v)
}

fn keys_of(v: &Value) -> Vec<String> {
    let mut k: Vec<String> = v.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    k.sort();
    k
}

fn unknown(v: &Value, allowed: &[&str]) -> Vec<String> {
    keys_of(v).into_iter().filter(|k| !allowed.contains(&k.as_str())).collect()
}

/// `folder_name`: a protocol's name as one safe path component.
pub fn folder_name(name: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || "._+-".contains(c) {
            out.push(c);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    let t: String = out.trim_matches(|c| c == '-' || c == '.').chars().take(60).collect();
    if t.is_empty() { "protocol".into() } else { t }
}

/// `load`: read a YAML file and check it.
pub fn load(path: &Path) -> Result<Value, ProtocolError> {
    let text = std::fs::read_to_string(path).map_err(|e| ProtocolError(format!("{}: {e}", path.display())))?;
    let data: Value = serde_yaml_ng::from_str(&text).map_err(|e| ProtocolError(format!("{}: not valid YAML ({e})", path.display())))?;
    check(&data, &path.display().to_string())
}

fn check_screen(p: &mut Value, where_: &str) -> Result<(), ProtocolError> {
    let sc = p["screen"].clone();
    let bad_shape =
        !sc.is_object() || !unknown(&sc, &SCREEN_KEYS).is_empty() || !matches!(sc["kind"].as_str(), Some("activation" | "knockout"));
    if bad_shape {
        return err(format!(
            "{where_}: screen needs {{kind: activation | knockout, ...}} with only {}",
            py_list(&SCREEN_KEYS.iter().map(|s| (*s).to_string()).collect::<Vec<_>>().tap_sort())
        ));
    }
    let mut out = sc.as_object().cloned().unwrap_or_default();
    for (key, lo, hi) in [
        ("controls", 1.0, 20.0),
        ("min_neurons", 1.0, 100000.0),
        ("max_types", 1.0, 100000.0),
        ("top", 1.0, 500.0),
        ("candidate_batch", 0.0, 100.0),
        ("batch", 0.0, 64.0),
    ] {
        if let Some(v) = sc.get(key) {
            let x = num(v, lo, hi, &format!("screen.{key}"), where_, false)?.unwrap_or(0.0);
            out.insert(key.into(), json!(x as i64));
        }
    }
    for key in ["types", "behaviors"] {
        if let Some(v) = sc.get(key) {
            let ok = v.as_array().is_some_and(|a| !a.is_empty() && a.len() <= 20000 && a.iter().all(Value::is_string));
            if !ok {
                return err(format!("{where_}: screen.{key} must be a non-empty list of names"));
            }
        }
    }
    let kind = sc["kind"].as_str().unwrap_or("");
    let has = |ks: &[&str]| ks.iter().any(|k| sc.get(*k).is_some());
    if kind == "activation" && has(&["behaviors", "top", "candidate_batch"]) {
        return err(format!("{where_}: behaviors, top and candidate_batch belong to a knockout screen"));
    }
    if kind == "knockout" && has(&["min_neurons", "max_types"]) {
        return err(format!("{where_}: min_neurons and max_types belong to an activation screen"));
    }
    for b in sc["behaviors"].as_array().into_iter().flatten() {
        let b = b.as_str().unwrap_or("");
        if !fly_sim::screens::VALIDATED_PATHWAYS.contains(&b) {
            return err(format!(
                "{where_}: screen.behaviors: '{b}' is not a validated pathway behavior; use {}",
                fly_sim::screens::VALIDATED_PATHWAYS.join(", ")
            ));
        }
    }
    p["screen"] = Value::Object(out);
    Ok(())
}

trait TapSort {
    fn tap_sort(self) -> Self;
}

impl TapSort for Vec<String> {
    fn tap_sort(mut self) -> Self {
        self.sort();
        self
    }
}

fn check_rig(p: &mut Value, where_: &str) -> Result<(), ProtocolError> {
    let r = p["rig"].clone();
    let name = r["name"].as_str().unwrap_or("");
    if !r.is_object() || !unknown(&r, &RIG_KEYS).is_empty() || !fly_sim::rigs::RIGS.contains(&name) {
        return err(format!(
            "{where_}: rig needs {{name: {}, ...}} with only {}",
            fly_sim::rigs::RIGS.join(" | "),
            py_list(&RIG_KEYS.iter().map(|s| (*s).to_string()).collect::<Vec<_>>().tap_sort())
        ));
    }
    let modes = fly_sim::rigs::scene_modes(name);
    let dflt = fly_sim::rigs::scene_defaults(name);
    let mode = r.get("mode").and_then(Value::as_str).unwrap_or_else(|| dflt["mode"].as_str().unwrap_or(""));
    if r.get("mode").is_some_and(|m| !m.is_string()) || !modes.contains(&mode) {
        return err(format!("{where_}: rig {name} mode must be one of {}", modes.join(", ")));
    }
    let scene = r.get("scene").map_or("bar", |s| s.as_str().unwrap_or(""));
    if scene != "panorama" && scene != "bar" {
        return err(format!("{where_}: rig.scene must be panorama or bar"));
    }
    let ind = r.get("individuality").map_or("subtle", |s| s.as_str().unwrap_or(""));
    if !["off", "subtle", "strong"].contains(&ind) {
        return err(format!("{where_}: rig.individuality must be off, subtle or strong"));
    }
    let mut out = r.as_object().cloned().unwrap_or_default();
    for (key, lo, hi) in [("seconds", 5.0, 600.0), ("omega", -10.0, 10.0), ("gain", 0.0, 3.0), ("bar_deg", -180.0, 180.0)] {
        if let Some(v) = r.get(key) {
            out.insert(key.into(), f(num(v, lo, hi, &format!("rig.{key}"), where_, false)?.unwrap_or(0.0)));
        }
    }
    p["rig"] = Value::Object(out);
    Ok(())
}

fn check_day3(p: &mut Value, where_: &str) -> Result<(), ProtocolError> {
    if let Some(w) = p.get("weather").cloned() {
        let allowed = ["rain", "gust_hz", "storm", "wind_speed", "wind_dir"];
        if !w.is_object() || !unknown(&w, &allowed).is_empty() {
            return err(format!(
                "{where_}: weather has unknown keys; allowed: {}",
                py_list(&allowed.iter().map(|s| (*s).to_string()).collect::<Vec<_>>().tap_sort())
            ));
        }
        if w.get("storm").is_some_and(|s| !s.is_boolean()) {
            return err(format!("{where_}: weather.storm must be true or false"));
        }
        let n =
            |k: &str, d: f64, lo: f64, hi: f64| num(&w.get(k).cloned().unwrap_or(json!(d)), lo, hi, &format!("weather.{k}"), where_, false);
        p["weather"] = json!({"rain": n("rain", 0.0, 0.0, 1.0)?, "gust_hz": n("gust_hz", 0.0, 0.0, 0.5)?,
                              "storm": w.get("storm").and_then(Value::as_bool).unwrap_or(false),
                              "wind_speed": n("wind_speed", 0.0, 0.0, 20.0)?, "wind_dir": n("wind_dir", 180.0, 0.0, 360.0)?});
    }
    if let Some(a) = p.get("audio").cloned() {
        let allowed = ["hz", "ipi_ms", "amp", "at_s", "seconds", "sensitivity"];
        if !a.is_object() || !unknown(&a, &allowed).is_empty() {
            return err(format!(
                "{where_}: audio has unknown keys (it is a synthetic hum, never the microphone); allowed: {}",
                py_list(&allowed.iter().map(|s| (*s).to_string()).collect::<Vec<_>>().tap_sort())
            ));
        }
        let n = |k: &str, d: Value, lo: f64, hi: f64, none: bool| {
            num(&a.get(k).cloned().unwrap_or(d), lo, hi, &format!("audio.{k}"), where_, none)
        };
        p["audio"] = json!({"hz": n("hz", json!(200.0), 10.0, 2000.0, false)?, "ipi_ms": n("ipi_ms", Value::Null, 10.0, 500.0, true)?,
                            "amp": n("amp", json!(0.1), 0.0, 1.0, false)?, "at_s": n("at_s", json!(0.0), 0.0, 3600.0, false)?,
                            "seconds": n("seconds", json!(1.0), 0.05, 600.0, false)?,
                            "sensitivity": n("sensitivity", json!(1.0), 0.1, 10.0, false)?});
    }
    if let Some(d) = p.get("predator").cloned() {
        let kind = d["kind"].as_str().unwrap_or("");
        if !d.is_object() || !unknown(&d, &["kind", "at_s"]).is_empty() || !PREDATORS.contains(&kind) {
            return err(format!("{where_}: predator needs kind: one of {} (and optionally at_s)", PREDATORS.join(", ")));
        }
        let at = num(&d.get("at_s").cloned().unwrap_or(json!(0.0)), 0.0, 3600.0, "predator.at_s", where_, false)?;
        p["predator"] = json!({"kind": kind, "at_s": at});
    }
    Ok(())
}

fn check_thermo_escape(opts: &Value, where_: &str) -> Result<(), ProtocolError> {
    let bad = unknown(opts, &["temps", "target", "effector_name"]);
    if !bad.is_empty() {
        return err(format!("{where_}: thermo_escape options are temps, target and effector_name, not {}", py_list(&bad)));
    }
    let (lo, hi) = fly_sim::thermogenetics::TEMP_RANGE_C;
    if let Some(t) = opts.get("temps") {
        let ok = t.as_array().is_some_and(|a| {
            (1..=50).contains(&a.len()) && a.iter().all(|x| x.is_number() && x.as_f64().is_some_and(|v| lo <= v && v <= hi))
        });
        if !ok {
            return err(format!("{where_}: thermo_escape temps must be a list of 1-50 temperatures within {}-{} C", g(lo), g(hi)));
        }
    }
    if opts.get("target").is_some_and(|t| !t.is_string()) {
        return err(format!("{where_}: thermo_escape target must be a neuron spec"));
    }
    if let Some(e) = opts.get("effector_name").map_or(Some("trpa1"), Value::as_str) {
        if let Err(m) = fly_sim::thermogenetics::effector(e) {
            return err(format!("{where_}: thermo_escape: {m}"));
        }
    }
    Ok(())
}

fn check_sleep_deprivation(p: &Value, where_: &str) -> Result<(), ProtocolError> {
    let opts = p.get("assay_options").cloned().unwrap_or(json!({}));
    if !opts.is_object() {
        return err(format!("{where_}: sleep_deprivation assay_options must be a mapping (deprive_s, recover_s, mode)"));
    }
    let bad = unknown(&opts, &["deprive_s", "recover_s", "mode"]);
    if !bad.is_empty() {
        return err(format!("{where_}: sleep_deprivation options are deprive_s, recover_s and mode, not {}", py_list(&bad)));
    }
    num(&opts.get("deprive_s").cloned().unwrap_or(json!(45.0)), 5.0, 120.0, "sleep_deprivation deprive_s", where_, false)?;
    num(&opts.get("recover_s").cloned().unwrap_or(json!(60.0)), 5.0, 180.0, "sleep_deprivation recover_s", where_, false)?;
    let mode = opts.get("mode").map_or("off", |m| m.as_str().unwrap_or(""));
    if !["off", "subtle", "strong"].contains(&mode) {
        return err(format!("{where_}: sleep_deprivation mode must be off, subtle or strong"));
    }
    if p.get("surgery").is_some_and(|s| s.as_object().is_some_and(|o| !o.is_empty())) {
        return err(format!("{where_}: sleep_deprivation is a paired design (each seed is its own control) and takes no surgery"));
    }
    Ok(())
}

fn check_hum_demo(opts: &Value, where_: &str) -> Result<(), ProtocolError> {
    if !opts.is_object() {
        return err(format!("{where_}: hum_demo assay_options must be a mapping (conditions, seconds, amp)"));
    }
    let bad = unknown(opts, &["conditions", "seconds", "amp"]);
    if !bad.is_empty() {
        return err(format!("{where_}: hum_demo options are conditions, seconds and amp, not {}", py_list(&bad)));
    }
    if let Some(c) = opts.get("conditions") {
        let ok = c.as_array().is_some_and(|a| {
            (1..=HUM_CONDITIONS.len()).contains(&a.len()) && a.iter().all(|x| x.as_str().is_some_and(|s| HUM_CONDITIONS.contains(&s)))
        });
        if !ok {
            return err(format!("{where_}: hum_demo conditions must be a list of {}", HUM_CONDITIONS.join(", ")));
        }
    }
    num(&opts.get("seconds").cloned().unwrap_or(json!(3.0)), 0.2, 30.0, "hum_demo seconds", where_, false)?;
    num(&opts.get("amp").cloned().unwrap_or(json!(0.1)), 0.0, 1.0, "hum_demo amp", where_, false)?;
    Ok(())
}

fn check_predator_escape(opts: &Value, where_: &str) -> Result<(), ProtocolError> {
    if !opts.is_object() {
        return err(format!("{where_}: predator_escape assay_options must be a mapping (kinds, trials)"));
    }
    let bad = unknown(opts, &["kinds", "trials"]);
    if !bad.is_empty() {
        return err(format!("{where_}: predator_escape options are kinds and trials, not {}", py_list(&bad)));
    }
    if let Some(k) = opts.get("kinds") {
        let ok = k.as_array().is_some_and(|a| {
            (1..=PREDATORS.len()).contains(&a.len()) && a.iter().all(|x| x.as_str().is_some_and(|s| PREDATORS.contains(&s)))
        });
        if !ok {
            return err(format!("{where_}: predator_escape kinds must be a list of {}", PREDATORS.join(", ")));
        }
    }
    if let Some(t) = opts.get("trials") {
        if !t.as_i64().is_some_and(|n| (1..=20).contains(&n)) || t.is_boolean() {
            return err(format!("{where_}: predator_escape trials must be a whole number from 1 to 20"));
        }
    }
    Ok(())
}

fn check_thermo_spec(spec: &Value, where_: &str) -> Result<Value, ProtocolError> {
    use fly_sim::thermogenetics as tg;
    const SPEC_KEYS: [&str; 4] = ["expression", "temperature_c", "kinetics", "time_scale"];
    if !spec.is_object() {
        return err(format!("{where_}: thermogenetics must be a mapping"));
    }
    let bad = unknown(spec, &SPEC_KEYS);
    if !bad.is_empty() {
        return err(format!(
            "{where_}: thermogenetics has unknown keys {}; allowed: {}",
            py_list(&bad),
            py_list(&SPEC_KEYS.iter().map(|s| (*s).to_string()).collect::<Vec<_>>().tap_sort())
        ));
    }
    let ex = spec.get("expression").and_then(Value::as_array).filter(|a| !a.is_empty());
    let Some(ex) = ex else {
        return err(format!("{where_}: thermogenetics.expression must be a list of {{effector, target}}"));
    };
    if ex.len() > 64 {
        return err(format!("{where_}: at most 64 expressions"));
    }
    let mut out = spec.as_object().cloned().unwrap_or_default();
    for (i, e) in ex.iter().enumerate() {
        let shape_ok = e.is_object()
            && unknown(e, &["effector", "target", "strength"]).is_empty()
            && e.get("effector").is_some()
            && e.get("target").is_some();
        if !shape_ok {
            return err(format!("{where_}: expression {} needs effector and target (and optionally strength)", i + 1));
        }
        if !e["target"].as_str().is_some_and(|t| !t.trim().is_empty()) {
            return err(format!("{where_}: expression {}: target must be a neuron spec such as type:DNp01 or line:SS00727", i + 1));
        }
        let strength = e.get("strength").map_or(Some(1.0), Value::as_f64);
        let eff = e["effector"].as_str().unwrap_or("");
        let r = match strength {
            Some(s) => tg::Expression::new(eff, e["target"].as_str().unwrap_or(""), s).map(|_| ()),
            None => Err("strength must be a number from 0 to 3".to_string()),
        };
        if let Err(m) = r {
            return err(format!("{where_}: expression {}: {m}", i + 1));
        }
    }
    out.insert("expression".into(), Value::Array(ex.clone()));
    let t = spec.get("temperature_c").cloned().unwrap_or(json!(22.0));
    let pts: Vec<(f64, f64)> = if t.is_number() {
        vec![(0.0, t.as_f64().unwrap_or(0.0))]
    } else if let Some(a) = t.as_array().filter(|a| !a.is_empty() && a.iter().all(|p| p.is_object() && keys_of(p) == ["at_s", "c"])) {
        if a.len() > 1000 {
            return err(format!("{where_}: a temperature schedule has at most 1000 points"));
        }
        let mut v = Vec::new();
        for p in a {
            match (p["at_s"].as_f64(), p["c"].as_f64()) {
                (Some(at), Some(c)) => v.push((at, c)),
                _ => return err(format!("{where_}: temperature_c points need numbers: {{at_s: seconds, c: degrees}}")),
            }
        }
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        v
    } else {
        return err(format!("{where_}: temperature_c must be a number or a list of {{at_s, c}}"));
    };
    if pts.iter().any(|(at, _)| !(0.0..=3600.0).contains(at)) {
        return err(format!("{where_}: at_s must be within 0-3600 s"));
    }
    let (lo, hi) = tg::TEMP_RANGE_C;
    if pts.iter().any(|(_, c)| !(lo..=hi).contains(c)) {
        return err(format!("{where_}: temperatures must be within {}-{} C", g(lo), g(hi)));
    }
    out.insert("temperature_c".into(), t);
    if let Some(k) = spec.get("kinetics") {
        if !matches!(k.as_str(), Some("real" | "steady")) {
            return err(format!("{where_}: kinetics must be real or steady"));
        }
    }
    if let Some(ts) = spec.get("time_scale") {
        match ts.as_f64() {
            None => return err(format!("{where_}: time_scale must be a number")),
            Some(x) if !(1e-3..=1e3).contains(&x) => return err(format!("{where_}: time_scale must be between 0.001 and 1000")),
            _ => {}
        }
    }
    Ok(Value::Object(out))
}

fn check_day2(p: &mut Value, where_: &str) -> Result<(), ProtocolError> {
    use fly_sim::patchclamp as pc;
    let has = |p: &Value, k: &str| p.get(k).is_some();
    if has(p, "screen") {
        let ignored: Vec<String> = [
            "assay",
            "classroom",
            "thermogenetics",
            "drug",
            "imaging",
            "stimuli",
            "recordings",
            "surgery",
            "control",
            "warmup_s",
            "duration_s",
            "patch",
            "params",
            "nwb",
            "assay_options",
            "rig",
        ]
        .iter()
        .chain(DAY3_KEYS.iter())
        .filter(|k| has(p, k))
        .map(|k| (*k).to_string())
        .collect();
        if !ignored.is_empty() {
            return err(format!(
                "{where_}: a screen protocol stands alone; it can't use {} (a screen sets its own timeline and builds its flies with the default parameters)",
                ignored.join(", ")
            ));
        }
        return check_screen(p, where_);
    }
    if has(p, "rig") {
        let ignored: Vec<String> = [
            "assay",
            "classroom",
            "thermogenetics",
            "drug",
            "imaging",
            "stimuli",
            "recordings",
            "surgery",
            "control",
            "warmup_s",
            "duration_s",
            "patch",
            "params",
            "nwb",
            "assay_options",
        ]
        .iter()
        .chain(DAY3_KEYS.iter())
        .filter(|k| has(p, k))
        .map(|k| (*k).to_string())
        .collect();
        if !ignored.is_empty() {
            return err(format!(
                "{where_}: a rig protocol stands alone; it can't use {} (the rig sets its own timeline and builds its flies with the default parameters)",
                ignored.join(", ")
            ));
        }
        return check_rig(p, where_);
    }
    if has(p, "patch") {
        let ignored: Vec<String> = [
            "assay",
            "classroom",
            "thermogenetics",
            "drug",
            "imaging",
            "stimuli",
            "recordings",
            "surgery",
            "control",
            "warmup_s",
            "duration_s",
        ]
        .iter()
        .chain(DAY3_KEYS.iter())
        .filter(|k| has(p, k))
        .map(|k| (*k).to_string())
        .collect();
        if !ignored.is_empty() {
            return err(format!(
                "{where_}: a patch protocol stands alone; it can't use {} (its own warm-up is patch.warmup_s)",
                ignored.join(", ")
            ));
        }
        let pa = p["patch"].clone();
        let keys = ["neuron", "index", "mode", "amplitudes", "duration_ms", "repeats", "warmup_s"];
        if !pa.is_object() || !unknown(&pa, &keys).is_empty() || pa.get("neuron").is_none() {
            return err(format!(
                "{where_}: patch needs {{neuron, ...}} with only {}",
                py_list(&keys.iter().map(|s| (*s).to_string()).collect::<Vec<_>>().tap_sort())
            ));
        }
        if !pa["neuron"].as_str().is_some_and(|s| !s.trim().is_empty()) {
            return err(format!("{where_}: patch.neuron must be a neuron spec such as type:DNp01"));
        }
        let amps: Vec<f64> = match pa.get("amplitudes") {
            None => vec![0.0, 0.05, 0.1],
            Some(a) => match a.as_array().map(|a| a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>()) {
                Some(Some(v)) => v,
                _ => return err(format!("{where_}: patch: amplitudes must be a list of numbers")),
            },
        };
        let dur = pa.get("duration_ms").and_then(Value::as_f64).unwrap_or(500.0);
        let reps = pa.get("repeats").and_then(Value::as_i64).unwrap_or(3);
        let proto = pc::ClampProtocol { amplitudes: amps, duration_ms: dur, repeats: reps.max(0) as usize, ..Default::default() };
        if let Err(m) = proto.check() {
            return err(format!("{where_}: patch: {m}"));
        }
        let mode = pa.get("mode").map_or("embedded", |m| m.as_str().unwrap_or(""));
        if mode != "embedded" && mode != "isolated" {
            return err(format!("{where_}: patch: mode must be one of ('embedded', 'isolated')"));
        }
        let index = pa.get("index").and_then(Value::as_i64).unwrap_or(0);
        let warmup = pa.get("warmup_s").and_then(Value::as_f64).unwrap_or(1.0);
        if index < 0 || !(0.0..=3600.0).contains(&warmup) {
            return err(format!("{where_}: patch: index must be >= 0 and warmup_s within 0-3600"));
        }
        let mut out = pa.as_object().cloned().unwrap_or_default();
        out.insert("amplitudes".into(), json!(proto.amplitudes));
        out.insert("duration_ms".into(), json!(proto.duration_ms));
        out.insert("repeats".into(), json!(proto.repeats));
        out.insert("index".into(), json!(index));
        out.insert("warmup_s".into(), json!(warmup));
        p["patch"] = Value::Object(out);
        return Ok(());
    }
    let classroom = p.get("classroom").is_some_and(|c| c.as_bool().unwrap_or(!c.is_null()));
    if has(p, "assay") || classroom {
        let used: Vec<String> = DAY2_KEYS.iter().chain(DAY3_KEYS.iter()).filter(|k| has(p, k)).map(|k| (*k).to_string()).collect();
        if !used.is_empty() {
            let kind = if has(p, "assay") { "an assay" } else { "a classroom" };
            return err(format!("{where_}: {} can't be used in {kind} protocol yet (stimulus protocols only)", used.join(", ")));
        }
        let opts = p.get("assay_options").cloned().unwrap_or(json!({}));
        match p.get("assay").and_then(Value::as_str) {
            Some("thermo_escape") => check_thermo_escape(&opts, where_)?,
            Some("predator_escape") => check_predator_escape(&opts, where_)?,
            Some("hum_demo") => check_hum_demo(&opts, where_)?,
            Some("sleep_deprivation") => check_sleep_deprivation(p, where_)?,
            _ => {}
        }
        return Ok(());
    }
    check_day3(p, where_)?;
    if let Some(t) = p.get("thermogenetics").cloned() {
        p["thermogenetics"] = check_thermo_spec(&t, where_)?;
    }
    if let Some(d) = p.get("drug").cloned() {
        let shape_ok = d.is_object()
            && unknown(&d, &["doses", "include_low_confidence", "cut"]).is_empty()
            && d["doses"].as_object().is_some_and(|o| !o.is_empty());
        if !shape_ok {
            return err(format!("{where_}: drug needs doses: {{name: 0-1}} (and optionally include_low_confidence, cut)"));
        }
        if d.get("include_low_confidence").is_some_and(|v| !v.is_boolean()) {
            return err(format!("{where_}: drug.include_low_confidence must be true or false"));
        }
        let cut = d.get("cut").and_then(Value::as_f64).unwrap_or(fly_sim::pharmacology::DEFAULT_CUT);
        let doses: Vec<(String, f64)> =
            d["doses"].as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(f64::NAN))).collect();
        let refs: Vec<(&str, f64)> = doses.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        if let Err(m) = fly_sim::pharmacology::wiring_for(
            &refs,
            d.get("include_low_confidence").and_then(Value::as_bool).unwrap_or(true),
            cut,
            &Default::default(),
        ) {
            return err(format!("{where_}: drug: {m}"));
        }
    }
    if let Some(im) = p.get("imaging").cloned() {
        let keys = ["indicator", "fps", "rois", "tiff", "shot_noise", "f0_photons", "dff_per_spike", "f0_tau_s"];
        if !im.is_object() || !unknown(&im, &keys).is_empty() {
            return err(format!(
                "{where_}: imaging has unknown keys; allowed: {}",
                py_list(&keys.iter().map(|s| (*s).to_string()).collect::<Vec<_>>().tap_sort())
            ));
        }
        let wrap = |m: String| ProtocolError(format!("{where_}: imaging: {m}"));
        let ind = im.get("indicator").map_or(Some("gcamp6s"), Value::as_str).unwrap_or("None");
        if let Err(m) = fly_sim::imaging::indicator(ind) {
            return Err(wrap(m));
        }
        if !im.get("fps").map_or(20.0, |v| v.as_f64().unwrap_or(f64::NAN)).clamp(f64::MIN, f64::MAX).is_finite()
            || !(1.0..=200.0).contains(&im.get("fps").map_or(20.0, |v| v.as_f64().unwrap_or(f64::NAN)))
        {
            return Err(wrap("fps must be between 1 and 200".into()));
        }
        for (key, default) in [("f0_photons", 100.0), ("dff_per_spike", 0.2), ("f0_tau_s", 30.0)] {
            let v = im.get(key).cloned().unwrap_or(json!(default));
            if v.is_boolean() || !v.as_f64().is_some_and(|x| x.is_finite() && x > 0.0) {
                return Err(wrap(format!("{key} must be a positive number")));
            }
        }
        for key in ["shot_noise", "tiff"] {
            if im.get(key).is_some_and(|v| !v.is_boolean()) {
                return Err(wrap(format!("{key} must be true or false")));
            }
        }
        let rois = im.get("rois").cloned().unwrap_or(json!("regions"));
        let ok = rois.as_str() == Some("regions") || rois.as_array().is_some_and(|a| !a.is_empty() && a.iter().all(Value::is_string));
        if !ok {
            return err(format!("{where_}: imaging.rois must be 'regions' or a list of neuron specs"));
        }
        if rois.as_array().is_some_and(|a| a.len() > fly_sim::imaging::MAX_ROIS) {
            return err(format!("{where_}: imaging: at most {} ROIs", fly_sim::imaging::MAX_ROIS));
        }
    }
    Ok(())
}

/// `check`: validate a parsed protocol and return it normalised.
pub fn check(data: &Value, where_: &str) -> Result<Value, ProtocolError> {
    if !data.is_object() {
        return err(format!("{where_}: the top level must be a mapping of keys"));
    }
    let unk = unknown(data, &TOP_KEYS);
    if !unk.is_empty() {
        let mut allowed: Vec<String> = TOP_KEYS.iter().map(|s| (*s).to_string()).collect();
        allowed.sort();
        return err(format!("{where_}: unknown keys {}; allowed: {}", py_list(&unk), py_list(&allowed)));
    }
    let mut p = data.clone();
    if p.get("name").is_none() {
        let stem = if where_ == "protocol" {
            "protocol".to_string()
        } else {
            Path::new(where_).file_stem().map_or(String::new(), |s| s.to_string_lossy().to_string())
        };
        p["name"] = json!(stem);
    }
    let has = |p: &Value, k: &str| p.get(k).is_some();
    if has(&p, "screen") && !["seeds", "seed", "flies"].iter().any(|k| has(&p, k)) {
        p["seeds"] = json!(fly_sim::screens::SCREEN_SEEDS);
    }
    if has(&p, "seeds") {
        let ok = p["seeds"].as_array().is_some_and(|a| a.iter().all(|s| s.is_i64() || s.is_u64()));
        if !ok {
            return err(format!("{where_}: seeds must be a list of integers"));
        }
        if p["seeds"].as_array().map_or(0, Vec::len) as i64 > MAX_FLIES {
            return err(format!("{where_}: at most 10,000 seeds"));
        }
    } else {
        let seed = p.get("seed").cloned().unwrap_or(json!(0));
        let flies = p.get("flies").cloned().unwrap_or(json!(1));
        let (si, fi) = (seed.as_i64().filter(|_| !seed.is_boolean()), flies.as_i64().filter(|_| !flies.is_boolean()));
        match (si, fi) {
            (Some(s), Some(n)) if n >= 1 => {
                if n > MAX_FLIES {
                    return err(format!("{where_}: flies must be at most 10,000"));
                }
                p["seeds"] = json!((s..s + n).collect::<Vec<i64>>());
            }
            _ => return err(format!("{where_}: seed must be an integer and flies a positive integer")),
        }
    }
    for key in ["params", "surgery", "assay_options"] {
        if p.get(key).is_some_and(|v| !v.is_object()) {
            return err(format!("{where_}: {key} must be a mapping"));
        }
    }
    if p.get("nwb").is_some_and(|v| !v.is_boolean()) {
        return err(format!("{where_}: nwb must be true or false"));
    }
    let names: Vec<&str> = ktf_params();
    let bad: Vec<String> = {
        let mut b: Vec<String> = p
            .get("params")
            .and_then(Value::as_object)
            .into_iter()
            .flat_map(|o| o.keys())
            .filter(|k| !names.contains(&k.as_str()))
            .cloned()
            .collect();
        b.sort();
        b
    };
    if !bad.is_empty() {
        let mut all: Vec<String> = names.iter().map(|s| (*s).to_string()).collect();
        all.sort();
        return err(format!("{where_}: unknown params {}; allowed: {}", py_list(&bad), py_list(&all)));
    }
    for (spec, mode) in p.get("surgery").and_then(Value::as_object).into_iter().flatten() {
        if !matches!(mode.as_i64(), Some(-1 | 1)) || mode.is_boolean() {
            return err(format!("{where_}: surgery '{spec}' must be -1 (silence) or 1 (stimulate)"));
        }
    }
    if has(&p, "assay") {
        let a = p["assay"].as_str().unwrap_or("");
        if !ASSAYS.contains(&a) {
            return err(format!(
                "{where_}: assay must be one of {}",
                py_list(&ASSAYS.iter().map(|s| (*s).to_string()).collect::<Vec<_>>())
            ));
        }
        check_day2(&mut p, where_)?;
        return Ok(p);
    }
    if p.get("classroom").is_some_and(|c| c.as_bool().unwrap_or(!c.is_null())) {
        check_day2(&mut p, where_)?;
        return Ok(p);
    }
    if has(&p, "patch") || has(&p, "rig") || has(&p, "screen") {
        check_day2(&mut p, where_)?;
        return Ok(p);
    }
    for (key, default) in [("warmup_s", 1.0), ("duration_s", 2.0)] {
        let v = p.get(key).cloned().unwrap_or_else(|| json!(default));
        p[key] = v.clone();
        if v.is_boolean() || !v.as_f64().is_some_and(|x| (0.0..=3600.0).contains(&x)) {
            return err(format!("{where_}: {key} must be a number of seconds (0-3600)"));
        }
    }
    if !has(&p, "stimuli") {
        p["stimuli"] = json!([]);
    }
    let stimuli = p["stimuli"].as_array().cloned().unwrap_or_default();
    for (i, s) in stimuli.iter().enumerate() {
        if !s.is_object() || s.get("target").is_none() || s.get("at_s").is_none() {
            return err(format!("{where_}: stimulus {} needs at least 'at_s' and 'target'", i + 1));
        }
        let unk = unknown(s, &STIM_KEYS);
        if !unk.is_empty() {
            return err(format!("{where_}: stimulus {} has unknown keys {}", i + 1, py_list(&unk)));
        }
        if !matches!(s.get("mode").map_or(Some("poke"), Value::as_str), Some("poke" | "drive")) {
            return err(format!("{where_}: stimulus {} mode must be poke or drive", i + 1));
        }
    }
    if !has(&p, "recordings") {
        p["recordings"] = json!([]);
    }
    let ok = p["recordings"].as_array().is_some_and(|a| a.iter().all(|r| r.is_object() && r.get("neurons").is_some()));
    if !ok {
        return err(format!("{where_}: recordings must be a list of {{name, neurons}}"));
    }
    check_day2(&mut p, where_)?;
    let _: &Map<String, Value> = p.as_object().expect("object");
    let _ = is_num(&Value::Null);
    Ok(p)
}

fn ktf_params() -> Vec<&'static str> {
    crate::labparams::params().iter().map(|p| p.name.as_str()).collect()
}
