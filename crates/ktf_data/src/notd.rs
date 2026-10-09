//! Neuron of the Day (`core/neuron_of_day.py`): a small card at launch with one curated Neurodex type, its literature fact and a "Try it"
//! button that sets up a one-click experiment. GAME RULE: the choice of type, the card, what Try it sets up; the fact on it is LITERATURE.
//!
//! The pick is by date and nothing else: a fixed shuffle seeded by the cycle number, so every curated entry comes up once per cycle of N
//! days and the same fact never repeats inside a cycle. Only curated types are ever picked.

use fly_sim::neurodex::TypeTable;
use sha2::{Digest, Sha256};

use crate::neurodex_facts::{Fact, facts};
use crate::pyrandom::Mt;

/// A fact that matches more types than this is too broad to try in one click.
pub const MAX_SURGERY_TYPES: usize = 80;

/// The curated facts that match at least one type in this brain, sorted by id.
pub fn candidates(tab: &TypeTable) -> Vec<&'static Fact> {
    if tab.brain != "adult" {
        return vec![];
    }
    let mut out: Vec<&Fact> = facts().iter().filter(|f| !matching_types(f, tab).is_empty()).collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn matching_types(fact: &Fact, tab: &TypeTable) -> Vec<String> {
    tab.names.iter().filter(|n| fact.matches(n)).cloned().collect()
}

/// `date.toordinal()`: 1 for 0001-01-01 (proleptic Gregorian).
pub fn toordinal(y: i32, m: u32, d: u32) -> i64 {
    let (m, d) = (i64::from(m), i64::from(d));
    let y = i64::from(y) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_1970 = era * 146_097 + doe - 719_468;
    days_since_1970 + 719_163
}

/// The fact of the day. Deterministic in (date, candidates).
pub fn pick(ordinal: i64, tab: &TypeTable) -> Option<&'static Fact> {
    let cands = candidates(tab);
    if cands.is_empty() {
        return None;
    }
    let n = cands.len() as i64;
    let (cycle, pos) = (ordinal.div_euclid(n), ordinal.rem_euclid(n));
    let digest = Sha256::digest(format!("kick-the-fly-notd-{n}-{cycle}").as_bytes());
    let seed = u64::from_str_radix(&digest.iter().map(|b| format!("{b:02x}")).collect::<String>()[..12], 16).expect("hex");
    let mut order: Vec<usize> = (0..n as usize).collect();
    Mt::seeded(seed).shuffle(&mut order);
    Some(cands[order[pos as usize]])
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// "laser", "surgery" or "none"
    pub action: &'static str,
    pub target: Option<String>,
    pub types: Vec<String>,
    pub text: String,
}

/// What Try it will do, as data.
pub fn plan(fact: &Fact, tab: &TypeTable, lab: bool) -> Plan {
    let types = matching_types(fact, tab);
    if lab && fact.types.len() == 1 && fact.prefix.is_empty() {
        let target = types[0].clone();
        return Plan {
            action: "laser",
            text: format!("Aims the laser at {target} (activate) and puts it in your hand. Replaces your laser target."),
            target: Some(target.clone()),
            types: vec![target],
        };
    }
    if lab && !fact.prefix.is_empty() && fact.types.is_empty() && fact.prefix.len() == 1 {
        let target = fact.prefix[0].clone();
        return Plan {
            action: "laser",
            text: format!("Aims the laser at every {target}* type (activate) and puts it in your hand. Replaces your laser target."),
            target: Some(target),
            types,
        };
    }
    if types.len() > MAX_SURGERY_TYPES {
        return Plan { action: "none", target: None, text: format!("Matches {} types: too broad for a one-click experiment.", types.len()), types };
    }
    let shown = types.iter().take(4).cloned().collect::<Vec<_>>().join(", ") + &if types.len() > 4 { format!(" and {} more", types.len() - 4) } else { String::new() };
    Plan { action: "surgery", target: None, text: format!("Stimulates {shown} in brain surgery. Replaces the surgery you have set."), types }
}

/// Everything the launch card shows.
#[derive(Debug, Clone)]
pub struct Card {
    pub fact: &'static Fact,
    pub types: Vec<String>,
    pub plan: Plan,
}

impl Card {
    pub fn first_type(&self) -> &str {
        &self.types[0]
    }
    pub fn also(&self) -> usize {
        self.types.len() - 1
    }
}

pub fn card(ordinal: i64, tab: &TypeTable, lab: bool) -> Option<Card> {
    let fact = pick(ordinal, tab)?;
    let types = matching_types(fact, tab);
    let plan = plan(fact, tab, lab);
    Some(Card { fact, types, plan })
}
