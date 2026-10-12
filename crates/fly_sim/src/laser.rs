//! The targeted optogenetics laser (`lab/laser.py`): its settings and the cell-type lookup. Current is injected straight into connectome
//! rows; no Gal4/UAS specificity or opsin kinetics is claimed.

use crate::pack::Labels;

pub const BASE_STIM_CURRENT: f64 = 0.48;
pub const BASE_SILENCE_CURRENT: f64 = -2.40;

pub const DEFAULT_TARGETS: [&str; 11] = ["LC10", "LPLC2", "LC4", "dnp01", "DNg02", "MDN", "aDN1", "MN9", "PPL1", "PAM", "DNa02"];

/// Configuration and live state of the laser (`LaserState`).
#[derive(Debug, Clone, PartialEq)]
pub struct LaserState {
    pub target_type: String,
    /// "activate" or "silence".
    pub mode: String,
    pub intensity: f64,
    /// "hold" or "pulse".
    pub trigger_mode: String,
    pub pulse_duration: f64,
    pub firing: bool,
    pub pulse_until: f64,
}

impl Default for LaserState {
    fn default() -> Self {
        Self {
            target_type: "dnp01".into(),
            mode: "activate".into(),
            intensity: 1.0,
            trigger_mode: "hold".into(),
            pulse_duration: 0.20,
            firing: false,
            pulse_until: 0.0,
        }
    }
}

impl LaserState {
    pub fn set_target(&mut self, t: &str) {
        self.target_type = t.trim().to_string();
    }

    pub fn set_mode(&mut self, m: &str) {
        if m == "activate" || m == "silence" {
            self.mode = m.into();
        }
    }

    pub fn set_intensity(&mut self, v: f64) {
        self.intensity = v.clamp(0.1, 3.0);
    }

    pub fn set_trigger_mode(&mut self, m: &str) {
        if m == "hold" || m == "pulse" {
            self.trigger_mode = m.into();
        }
    }

    pub fn set_pulse_duration(&mut self, s: f64) {
        self.pulse_duration = s.clamp(0.05, 2.0);
    }

    pub fn current_value(&self) -> f64 {
        (if self.mode == "activate" { BASE_STIM_CURRENT } else { BASE_SILENCE_CURRENT }) * self.intensity
    }

    pub fn is_active(&self, now: f64) -> bool {
        match self.trigger_mode.as_str() {
            "hold" => self.firing,
            "pulse" => now < self.pulse_until,
            _ => false,
        }
    }

    pub fn trigger_press(&mut self, now: f64) {
        self.firing = true;
        if self.trigger_mode == "pulse" {
            self.pulse_until = now + self.pulse_duration;
        }
    }

    pub fn trigger_release(&mut self) {
        self.firing = false;
    }
}

/// `LaserState._scan_rows`: `type:A,B` / `prefix:A,B` / `side:R:A,B` specs, else exact > prefix > contains on the lower-cased type.
/// (`line:` specs need the genetic toolkit, which is not ported: they select nothing.)
pub fn resolve_rows(labels: &Labels, target: &str) -> Vec<u32> {
    let tt = target.to_lowercase();
    let types: Vec<String> = labels.types.iter().map(|t| t.to_lowercase()).collect();
    let rows = |pred: &dyn Fn(&str) -> bool| -> Vec<u32> { (0..types.len() as u32).filter(|&i| pred(&types[i as usize])).collect() };
    if tt.starts_with("line:") {
        return crate::genetics::rows_of_lines(labels, target).unwrap_or_default();
    }
    if tt.starts_with("type:") || tt.starts_with("prefix:") || tt.starts_with("side:") {
        let (kind, rest) = tt.split_once(':').unwrap_or((&tt, ""));
        let (mut kind, mut rest, mut side) = (kind.to_string(), rest.to_string(), None::<String>);
        if kind == "side" {
            let (s, r) = rest.split_once(':').map_or((rest.clone(), String::new()), |(a, b)| (a.to_string(), b.to_string()));
            side = Some(s);
            rest = r;
            kind = "type".into();
        }
        let names: Vec<&str> = rest.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        let mut out: Vec<u32> = rows(&|t| names.iter().any(|n| if kind == "prefix" { t.starts_with(n) } else { t == *n }));
        if let Some(s) = side {
            let suffix = format!("_{s}");
            out.retain(|&i| labels.instance.get(i as usize).is_some_and(|x| x.to_lowercase().ends_with(&suffix)));
        }
        return out;
    }
    let exact = rows(&|t| t == tt);
    if !exact.is_empty() {
        return exact;
    }
    let prefix = rows(&|t| t.starts_with(&tt));
    if !prefix.is_empty() {
        return prefix;
    }
    rows(&|t| t.contains(&tt))
}
