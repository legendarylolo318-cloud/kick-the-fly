//! Lesion battles: two players each perform surgery on their own fly, within a budget, then the flies fight (the tournament's duel).
//! Python: `game/lesion_battle.py`. Tags: the surgery's effect on the neurons is CONNECTOME; the budget, the size cap, the arena, the
//! blaster and the rounds are GAME RULES; who wins is a MODEL PREDICTION about this model, not about real flies.
//!
//! The budget is a GAME RULE: at most `BUDGET` picks per fly, each a cell type (silenced or stimulated) of at most `MAX_TYPE_NEURONS`
//! neurons. Not ported: the `KTF1-SRG-` share code and the saved-loadout file (they need `core/sharecode.py`).

use std::collections::HashMap;

pub const BUDGET: usize = 3;
pub const MAX_TYPE_NEURONS: usize = 500;
pub const MAX_SAVED: usize = 20;
pub const ROUNDS: [u32; 2] = [1, 3];
pub const SILENCE: i8 = -1;
pub const STIMULATE: i8 = 1;
pub const TAG_TEXT: &str = "CONNECTOME for what the surgery does to the neurons, GAME RULE for the budget, the size cap and the arena, MODEL PREDICTION for who wins";
/// The types the duel itself reads: a hint for the picker, never an answer.
pub const SUGGESTED: [&str; 15] = ["DNp35", "DNpe052", "DNa01", "DNa02", "DNp01", "DNp09", "LC10a", "LC10b", "LC11", "LPLC2", "LC4", "MDN", "DNg02_a", "PPL1", "PAM"];

pub fn mode_word(m: i8) -> &'static str {
    if m < 0 { "silenced" } else { "stimulated" }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadoutError(pub String);

impl std::fmt::Display for LoadoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Loadout {
    pub name: String,
    /// Cell type -> -1 silenced / 1 stimulated, in the order picked.
    pub picks: Vec<(String, i8)>,
}

impl Loadout {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), picks: Vec::new() }
    }

    /// Add (or change) a pick, refusing what the rules do not allow. `sizes` maps each cell type of the brain to its neuron count.
    pub fn add(&mut self, cell_type: &str, mode: i8, sizes: Option<&HashMap<String, usize>>) -> Result<(), LoadoutError> {
        let t = cell_type.trim();
        if mode != SILENCE && mode != STIMULATE {
            return Err(LoadoutError("a pick is silenced (-1) or stimulated (1)".into()));
        }
        if let Some(sizes) = sizes {
            let Some(&n) = sizes.get(t) else { return Err(LoadoutError(format!("{t:?} is not a cell type of this brain"))) };
            if n > MAX_TYPE_NEURONS {
                return Err(LoadoutError(format!("{t} has {n} neurons; a pick may have at most {MAX_TYPE_NEURONS}")));
            }
        }
        if let Some(i) = self.picks.iter().position(|p| p.0 == t) {
            self.picks[i].1 = mode;
        } else if self.picks.len() >= BUDGET {
            return Err(LoadoutError(format!("the budget is {BUDGET} picks: take one out first")));
        } else {
            self.picks.push((t.to_string(), mode));
        }
        Ok(())
    }

    pub fn remove(&mut self, cell_type: &str) {
        self.picks.retain(|p| p.0 != cell_type);
    }

    pub fn neurons(&self, sizes: &HashMap<String, usize>) -> usize {
        self.picks.iter().map(|p| sizes.get(&p.0).copied().unwrap_or(0)).sum()
    }

    pub fn describe(&self) -> String {
        if self.picks.is_empty() {
            return "no surgery".into();
        }
        self.picks.iter().map(|(t, m)| format!("{t} {}", mode_word(*m))).collect::<Vec<_>>().join(", ")
    }

    /// The `(spec, mode)` commands that perform this surgery on a brain.
    pub fn surgery(&self) -> Vec<(String, i8)> {
        self.picks.iter().map(|(t, m)| (format!("type:{t}"), *m)).collect()
    }
}

/// Cell-type sizes of a brain from its labels.
pub fn type_sizes(types: &[String]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for t in types.iter().filter(|t| !t.is_empty()) {
        *m.entry(t.clone()).or_insert(0) += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes() -> HashMap<String, usize> {
        type_sizes(&["DNp09", "DNp09", "KCg", "KCg"].map(String::from).iter().cloned().chain((0..600).map(|_| "huge".to_string())).collect::<Vec<_>>())
    }

    #[test]
    fn the_budget_the_size_cap_and_unknown_types_are_enforced() {
        let s = sizes();
        let mut l = Loadout::new("mine");
        l.add("DNp09", SILENCE, Some(&s)).unwrap();
        l.add("KCg", STIMULATE, Some(&s)).unwrap();
        l.add("DNp09", STIMULATE, Some(&s)).unwrap();
        assert_eq!(l.picks.len(), 2, "changing a pick does not use budget");
        assert!(l.add("huge", SILENCE, Some(&s)).unwrap_err().0.contains("at most 500"));
        assert!(l.add("nope", SILENCE, Some(&s)).unwrap_err().0.contains("not a cell type"));
        assert!(l.add("DNp09", 0, Some(&s)).is_err());
        l.add("x", SILENCE, None).unwrap();
        assert!(l.add("y", SILENCE, None).unwrap_err().0.contains("budget is 3"));
        l.remove("x");
        l.add("y", SILENCE, None).unwrap();
    }

    #[test]
    fn describe_and_surgery_commands() {
        let mut l = Loadout::new("a");
        assert_eq!(l.describe(), "no surgery");
        l.add("DNp01", SILENCE, None).unwrap();
        l.add("PAM", STIMULATE, None).unwrap();
        assert_eq!(l.describe(), "DNp01 silenced, PAM stimulated");
        assert_eq!(l.surgery(), vec![("type:DNp01".to_string(), -1), ("type:PAM".to_string(), 1)]);
        assert_eq!(l.neurons(&sizes()), 0);
    }
}
