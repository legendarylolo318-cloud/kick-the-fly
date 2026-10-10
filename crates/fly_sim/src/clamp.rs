//! Neural clamp (`lab/clamp.py`): record a run's exact spike train and replay it into a modified connectome under dynamic clamp.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::cards::{Build, LIF_DEFAULTS};
use crate::csr::Weights;
use crate::pack::Labels;
use crate::wiring::{self, Wiring};

pub const NOTICE: &str = "DYNAMIC CLAMP: Forced spikes override the network's own state and break feedback loops such as proprioception. This is dynamic clamping, not an autonomous run.";

/// A recorded free-running spike train: `spikes[step][neuron]`.
pub struct Reference {
    pub stimulus: String,
    pub steps: usize,
    pub spikes: Vec<Vec<bool>>,
    pub region_rates: serde_json::Map<String, Value>,
}

fn rate(spikes: &[Vec<bool>], mask: &[bool]) -> Option<f64> {
    let cols: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
    if cols.is_empty() {
        return None;
    }
    let count: usize = spikes.iter().map(|row| cols.iter().filter(|&&c| row[c]).count()).sum();
    Some(count as f64 / (spikes.len() * cols.len()) as f64 / 0.005)
}

fn region_rates(labels: &Labels, spikes: &[Vec<bool>]) -> Vec<(String, f64)> {
    let n = labels.types.len();
    let region = |i: usize| labels.region.get(i).filter(|s| !s.is_empty()).map_or("unassigned", String::as_str);
    let mut names: Vec<&str> = (0..n).map(region).collect();
    names.sort_unstable();
    names.dedup();
    names
        .into_iter()
        .filter_map(|reg| {
            let mask: Vec<bool> = (0..n).map(|i| region(i) == reg).collect();
            rate(spikes, &mask).map(|r| (reg.to_string(), r))
        })
        .collect()
}

/// `record_reference_run`: free-running spikes with the stimulus held for the whole run.
pub fn record_reference_run(weights: &Arc<Weights>, labels: &Arc<Labels>, stimulus: &str, steps: usize, seed: u64) -> Reference {
    let b = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = b.brain_with(seed, None, 150, true);
    let group = match stimulus {
        "looming" => Some("loom"),
        "sugar" => Some("sweet"),
        "antenna" => Some("head"),
        _ => None,
    };
    let rows: Vec<u32> = group.map(|g| br.groups_ref().sense(g, None).to_vec()).unwrap_or_default();
    if group.is_some() {
        br.drive(&rows, 0.5);
    }
    let n = br.n();
    let mut spikes = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut row = vec![false; n];
        for &i in br.step() {
            row[i as usize] = true;
        }
        spikes.push(row);
    }
    if group.is_some() {
        br.undrive(&rows);
    }
    let rr: serde_json::Map<String, Value> = region_rates(labels, &spikes).into_iter().map(|(k, v)| (k, json!(v))).collect();
    Reference { stimulus: stimulus.into(), steps, spikes, region_rates: rr }
}

/// `run_neural_clamp`: replay `reference` into a brain with this wiring and lesion (the clamp set defaults to the stimulated input group).
#[allow(clippy::too_many_arguments)]
pub fn run_neural_clamp(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    counts: &[i32],
    reference: &Reference,
    wiring: &Wiring,
    lesion_rows: Option<&[u32]>,
    seed: u64,
    created: &str,
) -> Value {
    let t0 = std::time::Instant::now();
    let wired = (!wiring.is_identity()).then(|| Arc::new(wiring::apply(weights, labels, counts, wiring)));
    let b = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: wired.as_ref() };
    let mut br = b.brain_with(seed, None, 150, true);
    if let Some(rows) = lesion_rows.filter(|r| !r.is_empty()) {
        br.set_surgery(rows, -1);
    }
    let n = br.n();
    let clamp_rows: Vec<u32> = match reference.stimulus.as_str() {
        "looming" => br.groups_ref().sense("loom", None).to_vec(),
        "sugar" => br.groups_ref().sense("sweet", None).to_vec(),
        "antenna" => br.groups_ref().sense("head", None).to_vec(),
        _ => (0..n as u32).collect(),
    };
    let mut clamp_mask = vec![false; n];
    for &r in &clamp_rows {
        clamp_mask[r as usize] = true;
    }
    let mut clamped: Vec<Vec<bool>> = Vec::with_capacity(reference.steps);
    for t in 0..reference.steps {
        let forced: Vec<u32> = clamp_rows.iter().copied().filter(|&r| reference.spikes[t][r as usize]).collect();
        br.sim.force_spikes(&clamp_mask, &forced);
        let mut row = vec![false; n];
        for &i in br.sim_step_override() {
            row[i as usize] = true;
        }
        for &r in &clamp_rows {
            row[r as usize] = false;
        }
        for &r in &forced {
            row[r as usize] = true;
        }
        clamped.push(row);
    }
    let mut c_rates = serde_json::Map::new();
    let mut d_rates = serde_json::Map::new();
    for (reg, c) in region_rates(labels, &clamped) {
        let f = reference.region_rates.get(&reg).and_then(Value::as_f64).unwrap_or(0.0);
        d_rates.insert(reg.clone(), json!(c - f));
        c_rates.insert(reg, json!(c));
    }
    let mut readouts = serde_json::Map::new();
    let mut order = Vec::new();
    for (name, types) in
        [("giant_fiber_DNp01", &["DNp01"][..]), ("sugar_MN9", &["MN9"][..]), ("steering_DNa01_02", &["DNa01", "DNa02"][..])]
    {
        let mask: Vec<bool> = labels.types.iter().map(|t| types.contains(&t.as_str())).collect();
        if let (Some(f), Some(c)) = (rate(&reference.spikes, &mask), rate(&clamped, &mask)) {
            readouts.insert(name.into(), json!({"free_hz": f, "clamped_hz": c, "diff_hz": c - f}));
            order.push(name);
        }
    }
    json!({
        "kind": "neural_clamp", "created": created, "seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        "stimulus": reference.stimulus, "steps": reference.steps, "clamped_neurons": clamp_rows.len(),
        "lesioned_neurons": lesion_rows.map_or(0, <[u32]>::len),
        "wiring": {"min_synapses": wiring.min_synapses, "flipped_neurons": wiring.flip_rows.len(),
                   "inhibition_scale": wiring.inhibition_scale, "label": wiring.label()},
        "notice": NOTICE, "free_region_rates": reference.region_rates, "clamped_region_rates": c_rates, "diff_region_rates": d_rates,
        "readouts": readouts, "readout_order": order,
    })
}

/// `save`: `neural_clamp.json` and `neural_clamp_diff.csv`.
pub fn save(res: &Value, folder: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(folder)?;
    std::fs::write(folder.join("neural_clamp.json"), serde_json::to_string_pretty(res).unwrap_or_default())?;
    let f = |v: &Value| v.as_f64().unwrap_or(0.0);
    let mut text = format!("# {NOTICE}\r\nregion,free_running_hz,clamped_hz,diff_hz\r\n");
    for (reg, free) in res["free_region_rates"].as_object().into_iter().flatten() {
        let c = f(&res["clamped_region_rates"][reg]);
        let d = f(&res["diff_region_rates"][reg]);
        text.push_str(&format!("{reg},{:.2},{c:.2},{d:+.2}\r\n", f(free)));
    }
    text.push_str("\r\nreadout,free_running_hz,clamped_hz,diff_hz\r\n");
    for name in res["readout_order"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        let r = &res["readouts"][name];
        text.push_str(&format!("{name},{:.2},{:.2},{:+.2}\r\n", f(&r["free_hz"]), f(&r["clamped_hz"]), f(&r["diff_hz"])));
    }
    std::fs::write(folder.join("neural_clamp_diff.csv"), text)
}
