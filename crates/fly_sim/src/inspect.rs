//! The neuron inspector's data (`Game._neuron_info`, `transmitter_of`): the type, instance and population of one neuron, its transmitter,
//! the six strongest inputs and outputs by weight, and how many partners it has.

use crate::csr::Weights;
use crate::groups::{population_names, population_of};
use crate::pack::Labels;

#[derive(Debug, Clone)]
pub struct NeuronInfo {
    pub i: u32,
    pub ty: String,
    pub instance: String,
    pub pop: String,
    pub nt: String,
    pub nt_conf: f32,
    pub nt_source: String,
    pub body_id: i64,
    /// (partner row, weight): `W[i, pre]` for the inputs, `W[post, i]` for the outputs; strongest first.
    pub ins: Vec<(u32, f32)>,
    pub outs: Vec<(u32, f32)>,
    pub n_in: usize,
    pub n_out: usize,
}

fn top(mut items: Vec<(u32, f32)>) -> Vec<(u32, f32)> {
    items.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    items.truncate(6);
    items
}

pub fn neuron_info(labels: &Labels, w: &Weights, i: usize) -> Option<NeuronInfo> {
    if i >= labels.types.len() || i >= w.n() {
        return None;
    }
    let (cols, vals) = w.csr.row(i);
    let ins: Vec<(u32, f32)> = cols.iter().copied().zip(vals.iter().copied()).collect();
    let n_in = ins.len();
    let mut outs: Vec<(u32, f32)> = vec![];
    for b in &w.blocks {
        let (a, z) = (b.colptr[i] as usize, b.colptr[i + 1] as usize);
        for k in a..z {
            outs.push((b.row0 as u32 + b.rows[k], b.data[k]));
        }
    }
    let n_out = outs.len();
    let sc = labels.superclass.get(i).map_or("", String::as_str);
    let names = population_names();
    let pop =
        population_of(sc).map_or_else(|| if sc.is_empty() { "unknown".to_string() } else { sc.to_string() }, |p| names[p].to_string());
    Some(NeuronInfo {
        i: i as u32,
        ty: labels.types.get(i).filter(|t| !t.is_empty()).cloned().unwrap_or_else(|| "untyped".into()),
        instance: labels.instance.get(i).cloned().unwrap_or_default(),
        pop,
        nt: labels.nt.get(i).cloned().unwrap_or_default(),
        nt_conf: labels.nt_conf.get(i).copied().unwrap_or(f32::NAN),
        nt_source: labels.nt_source.get(i).cloned().unwrap_or_default(),
        body_id: labels.body_id.get(i).copied().unwrap_or(0),
        ins: top(ins),
        outs: top(outs),
        n_in,
        n_out,
    })
}
