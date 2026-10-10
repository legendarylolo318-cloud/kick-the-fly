//! The neuron groups of the brain surgery screen (`kick_the_fly.SURGERY`, `Game._surgery_rows`, `lab/lesions.py`): 23 entries, each OFF
//! (-2.4 per step in the drive), normal or ON (+0.48). Row sets are checked against Python's (`tests/golden/surgery.json`).

use crate::groups::{Groups, population_names, population_of};
use crate::pack::Labels;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spec {
    /// A detail group by name(s) (`brain.det_id`).
    Group(&'static [&'static str]),
    /// A population (`POPS`) by name.
    Pop(&'static [&'static str]),
    Type(&'static [&'static str]),
    Prefix(&'static [&'static str]),
    HemifieldVisual(char),
    Hemisphere(char),
    All,
}

pub const P1_TYPES: [&str; 25] = [
    "pC1_1a",
    "pC1_1b",
    "pC1_2a",
    "pC1_2a/2b",
    "pC1_2b",
    "pC1_2c",
    "pC1_3a",
    "pC1_3b",
    "pC1_3c",
    "pC1_5a",
    "pC1_5b",
    "pC1_6a",
    "pC1_7a",
    "pC1_7b",
    "pC1_12a",
    "pC1_13a",
    "pC1_14a",
    "pC1_14b",
    "pC1_15a",
    "pC1_15b",
    "pC1_15c",
    "pC1_16a",
    "pC1_16b",
    "pC1_17a",
    "pC1_17b",
];

/// (label, rows). The order is the screen's (and the old saves' surgery settings point at it).
pub const ENTRIES: [(&str, Spec); 23] = [
    ("Giant fiber escape neuron (DNp01)", Spec::Type(&["DNp01"])),
    ("Looming detectors (LPLC2, LC4)", Spec::Group(&["loom"])),
    ("Touch neurons", Spec::Group(&["head", "body", "legs", "wing"])),
    ("Pain relay: ascending neurons", Spec::Pop(&["ascending"])),
    ("Heat and cold sensors", Spec::Group(&["heat", "cold"])),
    ("Smell neurons (ORNs)", Spec::Group(&["smell"])),
    ("Mushroom bodies: Kenyon cells", Spec::Prefix(&["KC"])),
    ("Reward dopamine neurons (PAM)", Spec::Group(&["reward"])),
    ("Punishment dopamine neurons (PPL1)", Spec::Group(&["punish"])),
    ("Moonwalker neurons (MDN)", Spec::Group(&["back"])),
    ("Walking neurons (DNp09)", Spec::Group(&["walk"])),
    ("Flight power neurons (DNg02)", Spec::Group(&["fly"])),
    (
        "Central complex (compass, steering)",
        Spec::Prefix(&["EPG", "PEN", "PEG", "PFL", "PFN", "PFG", "PFR", "ER", "FB", "hDelta", "vDelta", "FC", "FS", "EL", "ExR"]),
    ),
    ("Optic lobes and photoreceptors", Spec::Pop(&["optic lobe", "photoreceptors"])),
    ("Left visual hemifield (LC10, LPLC2, LC4, LPTC)", Spec::HemifieldVisual('L')),
    ("Right visual hemifield (LC10, LPLC2, LC4, LPTC)", Spec::HemifieldVisual('R')),
    ("Left hemisphere (all left neurons)", Spec::Hemisphere('L')),
    ("Right hemisphere (all right neurons)", Spec::Hemisphere('R')),
    ("Every neuron", Spec::All),
    ("Song command neuron (pIP10)", Spec::Type(&["pIP10"])),
    ("Courtship neurons P1 (pC1, pMP-e/pMP4)", Spec::Type(&P1_TYPES)),
    ("Aggression neurons TK-FruM (AVLP727m)", Spec::Type(&["AVLP727m"])),
    ("Sleep neurons: dorsal fan-shaped body (FB6, FB7)", Spec::Prefix(&["FB6", "FB7"])),
];

fn nanmedian_x(soma: &[[f32; 3]]) -> Option<f32> {
    let mut v: Vec<f32> = soma.iter().map(|s| s[0]).filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f32::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

/// A hemisphere (or just its visual cell types): the instance name ends `_L`/`_R`; neurons without a side go by the cell body's side of the
/// midline (`lesions.hemisphere_rows`, `hemifield_visual_rows`).
fn side_rows(labels: &Labels, side: char, visual_only: bool) -> Vec<u32> {
    let n = labels.types.len();
    let suffix = format!("_{side}");
    let inst = |i: usize| labels.instance.get(i).map_or("", String::as_str);
    let mid = nanmedian_x(&labels.soma);
    (0..n)
        .filter(|&i| {
            if visual_only {
                let t = labels.types[i].as_str();
                if !["LC", "LPLC", "LPTC", "VS", "HS"].iter().any(|p| t.starts_with(p)) {
                    return false;
                }
            }
            if inst(i).ends_with(&suffix) {
                return true;
            }
            let has_side = inst(i).ends_with("_L") || inst(i).ends_with("_R");
            match (mid, labels.soma.get(i)) {
                (Some(m), Some(s)) if !has_side && !s[0].is_nan() => {
                    if side == 'L' {
                        s[0] > m
                    } else {
                        s[0] <= m
                    }
                }
                _ => false,
            }
        })
        .map(|i| i as u32)
        .collect()
}

/// The rows of one entry.
pub fn rows(labels: &Labels, groups: &Groups, spec: Spec) -> Vec<u32> {
    let n = labels.types.len();
    let by = |pred: &dyn Fn(usize) -> bool| -> Vec<u32> { (0..n).filter(|&i| pred(i)).map(|i| i as u32).collect() };
    match spec {
        Spec::Group(names) => {
            let mut r: Vec<u32> = names.iter().flat_map(|g| groups.det_members(g)).collect();
            r.sort_unstable();
            r.dedup();
            r
        }
        Spec::Pop(names) => {
            let pops = population_names();
            let ids: Vec<usize> = names.iter().filter_map(|n| pops.iter().position(|p| p == n)).collect();
            by(&|i| population_of(&labels.superclass[i]).is_some_and(|p| ids.contains(&p)))
        }
        Spec::Type(names) => by(&|i| names.contains(&labels.types[i].as_str())),
        Spec::Prefix(names) => by(&|i| names.iter().any(|p| labels.types[i].starts_with(p))),
        Spec::HemifieldVisual(s) => side_rows(labels, s, true),
        Spec::Hemisphere(s) => side_rows(labels, s, false),
        Spec::All => (0..n as u32).collect(),
    }
}

/// All 23 row sets.
pub fn all_rows(labels: &Labels, groups: &Groups) -> Vec<Vec<u32>> {
    ENTRIES.iter().map(|(_, s)| rows(labels, groups, *s)).collect()
}
