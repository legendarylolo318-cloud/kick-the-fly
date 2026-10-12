//! The five curated lecture protocols of Classroom mode (`lab/classroom.py:CURATED_LECTURES`), dumped from Python as data.

use std::sync::OnceLock;

use serde_json::Value;

/// `[{id, title, description, steps: [{title, explanation, key_takeaway, citation, grounding, action, neurons: [{label, spec, role, count}]}]}]`.
pub fn lectures() -> &'static Value {
    static L: OnceLock<Value> = OnceLock::new();
    L.get_or_init(|| serde_json::from_str(include_str!("../data/classroom_lectures.json")).expect("classroom_lectures.json is valid"))
}

pub fn lecture(id: &str) -> Option<&'static Value> {
    lectures().as_array()?.iter().find(|l| l["id"] == id)
}
