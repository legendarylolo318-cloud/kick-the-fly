//! Pharmacology (`lab/pharmacology.py`): drugs as synaptic scaling by predicted neurotransmitter, with the confidence of each prediction
//! on show. MODEL PREDICTION: a drug is one rule, a factor on every synapse whose presynaptic neuron releases the transmitter.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::csr::Csr;
use crate::pack::Labels;
use crate::wiring::Wiring;

pub const AGONIST_MAX: f64 = 2.0;
pub const DEFAULT_CUT: f64 = 0.7;
pub const TAG_TEXT: &str =
    "MODEL PREDICTION: scales synapses by the dataset's predicted transmitter. No receptor, kinetic or neuromodulatory model.";
pub const LEVELS: [&str; 6] =
    ["measured", "predicted >= 0.9", "predicted 0.7-0.9", "predicted 0.5-0.7", "predicted < 0.5", "no confidence"];

pub struct Drug {
    pub key: &'static str,
    pub label: &'static str,
    pub targets: &'static [&'static str],
    pub block: bool,
}

pub const DRUGS: [Drug; 4] = [
    Drug { key: "picrotoxin", label: "Picrotoxin", targets: &["gaba", "glutamate"], block: true },
    Drug { key: "cholinergic", label: "Cholinergic block", targets: &["acetylcholine"], block: true },
    Drug { key: "glucl", label: "Glutamate-Cl block", targets: &["glutamate"], block: true },
    Drug { key: "gabaa_agonist", label: "GABA-A agonist", targets: &["gaba"], block: false },
];

pub fn drug(name: &str) -> Result<&'static Drug, String> {
    let key = match name.trim().to_lowercase().replace(' ', "_").as_str() {
        "picrotoxin" | "ptx" => "picrotoxin",
        "cholinergic" | "cholinergic_block" | "ach_block" => "cholinergic",
        "glucl" | "glutamate_cl" | "glutamate-cl" | "glucl_block" => "glucl",
        "gabaa_agonist" | "gaba_agonist" | "gabaa" => "gabaa_agonist",
        _ => return Err(format!("unknown drug {name:?}; use one of [\"cholinergic\", \"gabaa_agonist\", \"glucl\", \"picrotoxin\"]")),
    };
    Ok(DRUGS.iter().find(|d| d.key == key).expect("the keys exist"))
}

/// The synaptic scale a dose (0-1) gives.
pub fn scale_for(name: &str, dose: f64) -> Result<f64, String> {
    let d = drug(name)?;
    if !(0.0..=1.0).contains(&dose) {
        return Err("dose must be between 0 and 1".into());
    }
    Ok(if d.block { 1.0 - dose } else { 1.0 + dose * (AGONIST_MAX - 1.0) })
}

/// `wiring_for`: the wiring that applies these drugs; two on one transmitter multiply.
pub fn wiring_for(doses: &[(&str, f64)], include_low: bool, cut: f64, base: &Wiring) -> Result<Wiring, String> {
    if !(cut > 0.0 && cut <= 1.0) {
        return Err("the confidence cut must be in (0, 1]".into());
    }
    let mut scales: BTreeMap<&str, f64> = BTreeMap::new();
    for (name, dose) in doses {
        let d = drug(name)?;
        let f = scale_for(name, *dose)?;
        for nt in d.targets {
            let e = scales.entry(nt).or_insert(1.0);
            *e *= f;
        }
    }
    let mut nts: Vec<(String, f64)> = scales.into_iter().filter(|(_, f)| *f != 1.0).map(|(k, f)| (k.to_string(), f)).collect();
    nts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    Ok(Wiring {
        min_synapses: base.min_synapses,
        flip_rows: base.flip_rows.clone(),
        inhibition_scale: base.inhibition_scale,
        nt_scales: nts,
        nt_min_conf: if include_low { 0.0 } else { cut },
    })
}

fn levels(labels: &Labels) -> Vec<i8> {
    (0..labels.nt.len())
        .map(|i| {
            let c = if labels.nt_conf[i].is_nan() { -1.0f32 } else { labels.nt_conf[i] };
            let mut l = 5i8;
            if (0.0..0.5f32).contains(&c) {
                l = 4;
            }
            if (0.5f32..0.7f32).contains(&c) {
                l = 3;
            }
            if (0.7f32..0.9f32).contains(&c) {
                l = 2;
            }
            if c >= 0.9f32 {
                l = 1;
            }
            if labels.nt_source.get(i).is_some_and(|s| s == "ground_truth") {
                l = 0;
            }
            l
        })
        .collect()
}

fn ok_mask(labels: &Labels, cut: f64) -> Vec<bool> {
    if cut <= 0.0 {
        return vec![true; labels.nt.len()];
    }
    let c = cut as f32;
    (0..labels.nt.len())
        .map(|i| {
            labels.nt_source.get(i).is_some_and(|s| s == "ground_truth")
                || (if labels.nt_conf[i].is_nan() { -1.0f32 } else { labels.nt_conf[i] }) >= c
        })
        .collect()
}

/// `_counts`: `{transmitter: [per level: (neurons, connections, synapses)]}` counting only neurons the cut lets a drug scale.
fn counts(labels: &Labels, csr: &Csr, syn: &[i32], cut: f64) -> BTreeMap<String, [(u64, u64, i64); 6]> {
    let lvl = levels(labels);
    let ok = ok_mask(labels, cut);
    let mut out: BTreeMap<String, [(u64, u64, i64); 6]> = BTreeMap::new();
    for i in 0..labels.nt.len() {
        let t = &labels.nt[i];
        if t.is_empty() || !ok[i] {
            continue;
        }
        out.entry(t.clone()).or_insert([(0, 0, 0); 6])[lvl[i] as usize].0 += 1;
    }
    for (e, &pre) in csr.indices.iter().enumerate() {
        let p = pre as usize;
        let t = &labels.nt[p];
        if t.is_empty() || !ok[p] {
            continue;
        }
        let slot = &mut out.entry(t.clone()).or_insert([(0, 0, 0); 6])[lvl[p] as usize];
        slot.1 += 1;
        slot.2 += i64::from(syn[e]);
    }
    out
}

/// `affected`: what a drug touches at each confidence level.
pub fn affected(labels: &Labels, csr: &Csr, syn: &[i32], name: &str, dose: f64, include_low: bool, cut: f64) -> Result<Value, String> {
    let d = drug(name)?;
    let full = counts(labels, csr, syn, 0.0);
    let kept = counts(labels, csr, syn, if include_low { 0.0 } else { cut });
    let mut rows = Vec::new();
    let mut tot = [0u64; 6];
    let mut tot_syn = [0i64; 2];
    for (i, label) in LEVELS.iter().enumerate() {
        let mut r = serde_json::Map::new();
        r.insert("level".into(), json!(label));
        let mut vals = [(0u64, 0u64, 0i64); 2];
        for (k, table) in [&full, &kept].into_iter().enumerate() {
            for t in d.targets {
                if let Some(per) = table.get(*t) {
                    vals[k].0 += per[i].0;
                    vals[k].1 += per[i].1;
                    vals[k].2 += per[i].2;
                }
            }
        }
        r.insert("neurons".into(), json!(vals[0].0));
        r.insert("connections".into(), json!(vals[0].1));
        r.insert("synapses".into(), json!(vals[0].2));
        r.insert("neurons_applied".into(), json!(vals[1].0));
        r.insert("connections_applied".into(), json!(vals[1].1));
        r.insert("synapses_applied".into(), json!(vals[1].2));
        r.insert("applied".into(), json!(vals[1].2 > 0 || vals[0].2 == 0));
        tot[0] += vals[0].0;
        tot[1] += vals[0].1;
        tot_syn[0] += vals[0].2;
        tot[3] += vals[1].0;
        tot[4] += vals[1].1;
        tot_syn[1] += vals[1].2;
        rows.push(Value::Object(r));
    }
    Ok(json!({
        "drug": d.key, "label": d.label, "dose": dose, "scale": scale_for(name, dose)?, "targets": d.targets, "rows": rows,
        "neurons": tot[0], "connections": tot[1], "synapses": tot_syn[0],
        "neurons_applied": tot[3], "connections_applied": tot[4], "synapses_applied": tot_syn[1],
        "include_low_confidence": include_low, "cut": cut, "tag": TAG_TEXT,
    }))
}

/// `dose_response`: mean whole-brain firing at each dose of one drug.
pub fn dose_response(
    c: &crate::robustness::Ctx,
    name: &str,
    doses: &[f64],
    include_low: bool,
    cut: f64,
    seeds: &[u64],
    steps: usize,
) -> Result<Value, String> {
    let mut rows = Vec::new();
    for &dose in doses {
        let w = if dose != 0.0 { wiring_for(&[(name, dose)], include_low, cut, &Wiring::default())? } else { Wiring::default() };
        let dist = crate::robustness::measure_firing_distribution(c, &w, seeds, steps);
        rows.push(json!({
            "dose": dose, "scale": scale_for(name, dose)?, "mean_hz": dist["mean"], "median_hz": dist["median"], "p95_hz": dist["p95"],
        }));
    }
    Ok(json!({
        "drug": drug(name)?.key, "rows": rows, "include_low_confidence": include_low, "cut": cut, "seeds": seeds, "steps": steps,
        "tag": TAG_TEXT,
    }))
}
