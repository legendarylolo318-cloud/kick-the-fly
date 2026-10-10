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

// --- what a wiring would do, and the dataset's confidence in its transmitters --------------------------------------------------
use serde_json::{Value, json};

/// `threshold_stats(n)`: what a minimum-synapse threshold of n would remove, over the whole connectome.
pub fn threshold_stats(csr: &Csr, counts: &[i32], n: i64) -> Value {
    let nn = csr.n_rows;
    let thr = n.max(1);
    let (mut in_b, mut in_a, mut out_b, mut out_a, mut touched) =
        (vec![0u32; nn], vec![0u32; nn], vec![0u32; nn], vec![0u32; nn], vec![0u32; nn]);
    let (mut n_drop, mut syn_total, mut syn_drop) = (0u64, 0i64, 0i64);
    for post in 0..nn {
        let (a, b) = (csr.indptr[post] as usize, csr.indptr[post + 1] as usize);
        for (&pre, &cnt) in csr.indices[a..b].iter().zip(&counts[a..b]) {
            let pre = pre as usize;
            let c = i64::from(cnt);
            syn_total += c;
            in_b[post] += 1;
            out_b[pre] += 1;
            if c < thr {
                n_drop += 1;
                syn_drop += c;
                touched[post] += 1;
                touched[pre] += 1;
            } else {
                in_a[post] += 1;
                out_a[pre] += 1;
            }
        }
    }
    let total = counts.len() as u64;
    json!({
        "threshold": n, "connections": total, "connections_dropped": n_drop,
        "connections_dropped_share": n_drop as f64 / total.max(1) as f64,
        "synapses": syn_total, "synapses_dropped": syn_drop,
        "neurons": nn, "neurons_touched": touched.iter().filter(|&&t| t > 0).count(),
        "neurons_cut_off": (0..nn).filter(|&i| in_b[i] > 0 && in_a[i] == 0).count(),
        "neurons_silenced_output": (0..nn).filter(|&i| out_b[i] > 0 && out_a[i] == 0).count(),
    })
}

fn signed(labels: &Labels) -> Vec<bool> {
    labels.nt.iter().map(|t| SIGN_OF.iter().any(|(k, _)| k == t)).collect()
}

/// Signed neurons the dataset is less than `max_conf` sure about, plus those it gives no confidence for (`flip_candidates`).
pub fn flip_candidates(labels: &Labels, max_conf: f64) -> Vec<u32> {
    let s = signed(labels);
    let cut = max_conf as f32; // float32 array against a Python float: compared in float32
    (0..labels.nt.len()).filter(|&i| s[i] && (labels.nt_conf[i].is_nan() || labels.nt_conf[i] < cut)).map(|i| i as u32).collect()
}

/// `confidence_stats`.
pub fn confidence_stats(labels: &Labels, max_conf: f64) -> Value {
    let s = signed(labels);
    let cand = flip_candidates(labels, max_conf);
    let unknown = (0..s.len()).filter(|&i| s[i] && labels.nt_conf[i].is_nan()).count();
    let mut conf: Vec<f32> = (0..s.len()).filter(|&i| s[i] && !labels.nt_conf[i].is_nan()).map(|i| labels.nt_conf[i]).collect();
    conf.sort_by(|a, b| a.total_cmp(b));
    let median = if conf.is_empty() {
        f64::NAN
    } else if conf.len() % 2 == 1 {
        f64::from(conf[conf.len() / 2])
    } else {
        f64::from((conf[conf.len() / 2 - 1] + conf[conf.len() / 2]) / 2.0)
    };
    json!({
        "cutoff": max_conf, "neurons": labels.nt.len(), "signed": s.iter().filter(|&&b| b).count(),
        "candidates": cand.len(), "unknown_confidence": unknown,
        "measured": labels.nt_source.iter().filter(|s| *s == "ground_truth").count(),
        "excitatory": labels.nt.iter().filter(|t| *t == "acetylcholine").count(),
        "inhibitory": labels.nt.iter().filter(|t| INHIBITORY.contains(&t.as_str())).count(),
        "median_confidence": median,
    })
}

/// `random_flip`: each candidate flipped independently with probability `share`, from `default_rng(seed)`.
pub fn random_flip(labels: &Labels, max_conf: f64, share: f64, seed: u64) -> Vec<u32> {
    let cand = flip_candidates(labels, max_conf);
    if share >= 1.0 {
        return cand;
    }
    let mut rng = crate::rng::Generator::new(seed);
    cand.into_iter().filter(|_| rng.random() < share).collect()
}

/// `inhibition_stats(g, scale)`.
pub fn inhibition_stats(csr: &Csr, labels: &Labels, scale: f64) -> Value {
    let inh: Vec<bool> = labels.nt.iter().map(|t| INHIBITORY.contains(&t.as_str())).collect();
    let n_edges = csr.indices.len();
    let n_inh = csr.indices.iter().filter(|&&p| inh[p as usize]).count();
    json!({
        "scale": scale, "severity": (1.0 - scale).clamp(0.0, 1.0), "total_connections": n_edges,
        "inhibitory_connections": n_inh, "inhibitory_share": n_inh as f64 / n_edges.max(1) as f64,
        "inhibitory_neurons": inh.iter().filter(|&&b| b).count(),
        "gaba_neurons": labels.nt.iter().filter(|t| *t == "gaba").count(),
        "glutamate_neurons": labels.nt.iter().filter(|t| *t == "glutamate").count(),
        "total_neurons": labels.nt.len(),
    })
}
