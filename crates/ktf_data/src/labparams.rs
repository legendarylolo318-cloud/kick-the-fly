//! The Lab's parameters (`lab/lab.py:PARAMS`, 43 of them): the LIF model's own, and the game rules' thresholds and constants. The table is
//! generated from the Python module (`crates/ktf_data/data/lab_params.json`), so labels, ranges, steps, formats and tips are its own.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub label: String,
    /// "model" or "rule"
    pub kind: String,
    pub default: f64,
    pub lo: f64,
    pub hi: f64,
    pub step: f64,
    pub fmt: String,
    pub tip: String,
}

pub fn params() -> &'static [Param] {
    static P: OnceLock<Vec<Param>> = OnceLock::new();
    P.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("../data/lab_params.json")).expect("lab_params.json is valid");
        v.as_array()
            .expect("an array")
            .iter()
            .map(|r| {
                let s = |k: &str| r[k].as_str().unwrap_or("").to_string();
                let f = |k: &str| r[k].as_f64().unwrap_or(0.0);
                Param {
                    name: s("name"),
                    label: s("label"),
                    kind: s("kind"),
                    default: f("default"),
                    lo: f("lo"),
                    hi: f("hi"),
                    step: f("step"),
                    fmt: s("fmt"),
                    tip: s("tip"),
                }
            })
            .collect()
    })
}

pub fn by_name(name: &str) -> Option<&'static Param> {
    params().iter().find(|p| p.name == name)
}

/// `Game.set_lab_param`'s clamp.
pub fn clamp(name: &str, v: f64) -> Option<f64> {
    by_name(name).map(|p| v.max(p.lo).min(p.hi))
}

/// `lab.modified`: the parameters that differ from their defaults.
pub fn modified(values: &BTreeMap<String, f64>) -> Vec<&'static Param> {
    params().iter().filter(|p| values.get(&p.name).is_some_and(|v| (v - p.default).abs() > 1e-9)).collect()
}

/// The Lab hub's pages `(label, menu page, tip)` in `Game.lab_pages` order; the hub shows the ones that exist in this build.
pub fn pages() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| serde_json::from_str(include_str!("../data/lab_pages.json")).expect("lab_pages.json is valid"))
}

/// `(title, category, what the sim does, what biology does, where)`.
pub type Assumption = (String, String, String, String, String);

/// The model-assumption cards of `lab.ASSUMPTIONS`.
pub fn assumptions() -> &'static [Assumption] {
    static A: OnceLock<Vec<Assumption>> = OnceLock::new();
    A.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("../data/lab_assumptions.json")).expect("lab_assumptions.json is valid");
        v.as_array()
            .expect("an array")
            .iter()
            .map(|r| {
                let s = |k: &str| r[k].as_str().unwrap_or("").to_string();
                (s("title"), s("tag"), s("what"), s("why"), s("where"))
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_the_pythons() {
        let p = params();
        assert_eq!(p.len(), 43);
        assert_eq!(p[0].name, "noise_std");
        assert_eq!((p[0].default, p[0].lo, p[0].hi), (0.05, 0.0, 0.2));
        assert_eq!(by_name("thresh.escape").unwrap().kind, "rule");
        assert_eq!(clamp("bias", 9.0), Some(0.3));
        assert!(clamp("nope", 1.0).is_none());
        let mut v = BTreeMap::new();
        v.insert("bias".to_string(), 0.25);
        v.insert("noise_std".to_string(), 0.05);
        assert_eq!(modified(&v).len(), 1);
        assert!(assumptions().len() >= 12);
    }
}
