//! Named neuron sets of the assays (`lab/assays.py:groups`) that the game needs: the sugar- and bitter-pathway taste neurons.
//!
//! `head_gust` (labellar, pharyngeal and tarsal gustatory receptor neurons) are ranked by how much one- and two-synapse input each sends
//! onto the sugar projection neurons (GNG540/GNG550) minus onto the bitter command neuron DNg28; the top `SWEET_N` are `sweet`, the bottom
//! `SWEET_N` `bitter`. Checked row for row against the Python result (`tests/golden/assay_groups.json`).

use crate::csr::Csr;
use crate::pack::Labels;

pub const SWEET_N: usize = 30;
/// Proboscis extension counts when MN9 fires at least this many times its rate before the offer (`assays.PER_RATIO`).
pub const PER_RATIO: f32 = 1.5;

/// (sweet, bitter) rows; empty when the pack has none of the types (a synthetic network).
pub fn taste_groups(labels: &Labels, w: &Csr) -> (Vec<u32>, Vec<u32>) {
    let n = labels.types.len();
    if w.n_rows != n {
        return (vec![], vec![]);
    }
    let head_gust: Vec<u32> = (0..n as u32)
        .filter(|&i| {
            let t = labels.types[i as usize].as_str();
            (t.starts_with("LB") || t.starts_with("PhG") || t.starts_with("claw_tpGRN") || t.starts_with("dorsal_tpGRN"))
                && labels.superclass[i as usize] == "cb_sensory"
        })
        .collect();
    if head_gust.is_empty() {
        return (vec![], vec![]);
    }
    let rows_of = |names: &[&str]| -> Vec<u32> { (0..n as u32).filter(|&i| names.contains(&labels.types[i as usize].as_str())).collect() };
    let sugar_pn = rows_of(&["GNG540", "GNG550"]);
    let bitter_sel = rows_of(&["DNg28"]);
    let mut slot = vec![u32::MAX; n];
    for (k, &g) in head_gust.iter().enumerate() {
        slot[g as usize] = k as u32;
    }
    let reach = |targets: &[u32]| -> Vec<f32> {
        let mut into = vec![0f32; n];
        for &t in targets {
            let (cols, vals) = w.row(t as usize);
            for (&c, &v) in cols.iter().zip(vals) {
                into[c as usize] += v.abs();
            }
        }
        let mut out: Vec<f32> = head_gust.iter().map(|&g| into[g as usize]).collect();
        for (r, &inr) in into.iter().enumerate() {
            if inr == 0.0 {
                continue;
            }
            let (cols, vals) = w.row(r);
            for (&c, &v) in cols.iter().zip(vals) {
                let k = slot[c as usize];
                if k != u32::MAX {
                    out[k as usize] += v.abs() * inr;
                }
            }
        }
        out
    };
    let (a, b) = (reach(&sugar_pn), reach(&bitter_sel));
    let pref: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    // np.argsort(-pref, kind="stable")
    let mut order: Vec<usize> = (0..pref.len()).collect();
    order.sort_by(|&i, &j| (-pref[i]).total_cmp(&(-pref[j])));
    let k = SWEET_N.min(order.len());
    let sweet = order[..k].iter().map(|&i| head_gust[i]).collect();
    let bitter = order[order.len() - k..].iter().map(|&i| head_gust[i]).collect();
    (sweet, bitter)
}
