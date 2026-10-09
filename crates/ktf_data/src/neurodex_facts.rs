//! The curated Neurodex facts (`data/neurodex_facts.yaml`, converted to JSON by `tests/golden/make_neurodex.py`): one literature
//! statement with its citation per type or type prefix. LITERATURE, hand-written and checked against the cited paper.

use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Fact {
    pub id: String,
    pub text: String,
    pub cite: String,
    pub doi: String,
    pub checked: String,
    #[serde(default)]
    pub types: Vec<String>,
    #[serde(default)]
    pub prefix: Vec<String>,
    #[serde(default)]
    pub game_note: String,
}

impl Fact {
    pub fn matches(&self, type_name: &str) -> bool {
        self.types.iter().any(|t| t == type_name) || self.prefix.iter().any(|p| type_name.starts_with(p.as_str()))
    }
}

pub fn facts() -> &'static [Fact] {
    static F: OnceLock<Vec<Fact>> = OnceLock::new();
    F.get_or_init(|| serde_json::from_str(include_str!("../data/neurodex_facts.json")).expect("neurodex_facts.json is generated and valid"))
}

/// The curated fact for a type, or None. The table is about the adult fly, so the larva dex has none.
pub fn fact_for(type_name: &str, brain: &str) -> Option<&'static Fact> {
    if brain != "adult" {
        return None;
    }
    facts().iter().find(|f| f.matches(type_name))
}
