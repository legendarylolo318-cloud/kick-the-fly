//! Wiring manipulations (`sim/wiring.py`): change the connectome the simulation runs on and say exactly what changed. They edit the
//! synapse matrix itself: drop weak connections, flip signs, scale inhibition or one transmitter's synapses.

use std::path::Path;

use crate::csr::{Csr, Weights};
use crate::npy::Npz;
use crate::pack::Labels;

pub const SIGN_OF: [(&str, i32); 4] = [("acetylcholine", 1), ("gaba", -1), ("glutamate", -1), ("histamine", -1)];
pub const INHIBITORY: [&str; 2] = ["gaba", "glutamate"];

/// One reversible change to the connectome; the default changes nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct Wiring {
    pub min_synapses: i64,
    pub flip_rows: Vec<u32>,
    pub inhibition_scale: f64,
    pub nt_scales: Vec<(String, f64)>,
    pub nt_min_conf: f64,
}

impl Default for Wiring {
    fn default() -> Self {
        Self { min_synapses: 1, flip_rows: vec![], inhibition_scale: 1.0, nt_scales: vec![], nt_min_conf: 0.0 }
    }
}

impl Wiring {
    pub fn is_identity(&self) -> bool {
        self.min_synapses <= 1 && self.flip_rows.is_empty() && self.inhibition_scale == 1.0 && self.nt_scales.iter().all(|(_, f)| *f == 1.0)
    }

    /// `Wiring.label()`.
    pub fn label(&self) -> String {
        let mut bits = Vec::new();
        if self.min_synapses > 1 {
            bits.push(format!("drop <{} synapses", self.min_synapses));
        }
        if !self.flip_rows.is_empty() {
            bits.push(format!("{} neurons sign-flipped", crate::netsci::group(self.flip_rows.len() as f64)));
        }
        if self.inhibition_scale != 1.0 {
            bits.push(format!("inhibition x{}", crate::sensitivity::fmt_g(self.inhibition_scale)));
        }
        for (nt, f) in &self.nt_scales {
            if *f != 1.0 {
                let mut s = format!("{nt} synapses x{}", crate::sensitivity::fmt_g(*f));
                if self.nt_min_conf > 0.0 {
                    s.push_str(&format!(" (confidence >= {} only)", crate::sensitivity::fmt_g(self.nt_min_conf)));
                }
                bits.push(s);
            }
        }
        if bits.is_empty() { "unmodified connectome".into() } else { bits.join(", ") }
    }
}

/// `|synapses|` for every entry of W in the pack's CSR order (`synapse_counts`): the dataset's integers.
pub fn synapse_counts(pack: &Path) -> Result<Vec<i32>, String> {
    let f = std::fs::File::open(pack).map_err(|e| e.to_string())?;
    let mut npz = Npz::read(std::io::BufReader::new(f)).map_err(|e| e.to_string())?;
    let data = npz.take("data").and_then(|a| a.into_i16()).map_err(|e| e.to_string())?;
    Ok(data.into_iter().map(|c| i32::from(c).abs()).collect())
}

/// `(entry indices this wiring changes, multiplier for each)` (`modified_entries`).
pub fn modified_entries(csr: &Csr, labels: &Labels, counts: &[i32], w: &Wiring) -> (Vec<usize>, Vec<f32>) {
    let mut mult = vec![1.0f32; counts.len()];
    if w.min_synapses > 1 {
        for (m, &c) in mult.iter_mut().zip(counts) {
            if i64::from(c) < w.min_synapses {
                *m = 0.0;
            }
        }
    }
    let pre = &csr.indices;
    if !w.flip_rows.is_empty() {
        let mut flip = vec![false; csr.n_cols];
        for &r in &w.flip_rows {
            flip[r as usize] = true;
        }
        for (m, &p) in mult.iter_mut().zip(pre) {
            if flip[p as usize] {
                *m *= -1.0;
            }
        }
    }
    if w.inhibition_scale != 1.0 {
        let inh: Vec<bool> = labels.nt.iter().map(|t| INHIBITORY.contains(&t.as_str())).collect();
        for (m, &p) in mult.iter_mut().zip(pre) {
            if inh[p as usize] {
                *m *= w.inhibition_scale as f32;
            }
        }
    }
    for (name, factor) in &w.nt_scales {
        if *factor != 1.0 {
            let ok: Vec<bool> = (0..labels.nt.len())
                .map(|i| {
                    labels.nt[i] == *name
                        && (w.nt_min_conf <= 0.0
                            || labels.nt_source.get(i).is_some_and(|s| s == "ground_truth")
                            || f64::from(if labels.nt_conf[i].is_nan() { -1.0 } else { labels.nt_conf[i] }) >= w.nt_min_conf)
                })
                .collect();
            for (m, &p) in mult.iter_mut().zip(pre) {
                if ok[p as usize] {
                    *m *= *factor as f32;
                }
            }
        }
    }
    let idx: Vec<usize> = (0..mult.len()).filter(|&i| mult[i] != 1.0).collect();
    let m = idx.iter().map(|&i| mult[i]).collect();
    (idx, m)
}

/// The weights with this wiring applied (a copy; the original stays the unmodified connectome).
pub fn apply(weights: &Weights, labels: &Labels, counts: &[i32], w: &Wiring) -> Weights {
    if w.is_identity() {
        return weights.clone();
    }
    let (idx, mult) = modified_entries(&weights.csr, labels, counts, w);
    let mut csr = weights.csr.clone();
    for (&i, &m) in idx.iter().zip(&mult) {
        csr.data[i] *= m;
    }
    Weights::new(csr, Weights::default_blocks())
}
