//! Connectome diff mode (`lab/diffmode.py`): two flies from the same seed and inputs with different configurations, run in lockstep and
//! diffed region by region.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::cards::{Build, LIF_DEFAULTS};
use crate::csr::Weights;
use crate::pack::Labels;
use crate::wiring::{self, Wiring};

/// One fly's configuration.
pub struct Config {
    pub wiring: Wiring,
    pub lesion_rows: Option<Vec<u32>>,
    pub label: String,
}

/// `np.add.reduce` of a float32 array (pairwise, eight running totals per block of 128).
fn np_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f32;
        for x in a {
            r += x;
        }
        return r;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    np_sum_f32(&a[..n2]) + np_sum_f32(&a[n2..])
}

fn build<'a>(w: &'a Arc<Weights>, l: &'a Arc<Labels>, wired: Option<&'a Arc<Weights>>) -> Build<'a> {
    Build { weights: w, labels: l, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired }
}

#[allow(clippy::too_many_arguments)]
pub fn run_connectome_diff(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    counts: &[i32],
    a: &Config,
    b: &Config,
    stimulus: &str,
    steps: usize,
    seed: u64,
    created: &str,
) -> Value {
    let t0 = std::time::Instant::now();
    let make = |c: &Config| {
        let wired = (!c.wiring.is_identity()).then(|| Arc::new(wiring::apply(weights, labels, counts, &c.wiring)));
        let mut br = build(weights, labels, wired.as_ref()).brain_with(seed, None, 150, true);
        if let Some(rows) = c.lesion_rows.as_deref().filter(|r| !r.is_empty()) {
            br.set_surgery(rows, -1);
        }
        br
    };
    let mut br_a = make(a);
    let mut br_b = make(b);
    let n = br_a.n();
    let region = |i: usize| labels.region.get(i).filter(|s| !s.is_empty()).map_or("unassigned", String::as_str);
    let mut names: Vec<&str> = (0..n).map(region).collect();
    names.sort_unstable();
    names.dedup();
    let masks: Vec<(String, Vec<usize>)> =
        names.iter().map(|r| (r.to_string(), (0..n).filter(|&i| region(i) == *r).collect::<Vec<_>>())).collect();
    let group = match stimulus {
        "looming" => Some("loom"),
        "sugar" => Some("sweet"),
        "antenna" => Some("head"),
        _ => None,
    };
    let stim_rows: Vec<u32> = group.map(|g| br_a.groups_ref().sense(g, None).to_vec()).unwrap_or_default();
    if group.is_some() {
        br_a.drive(&stim_rows, 0.5);
        br_b.drive(&stim_rows, 0.5);
    }
    let bin_size = 5usize;
    let n_bins = steps / bin_size;
    let mut tl_a: Vec<Vec<f64>> = vec![Vec::new(); masks.len()];
    let mut tl_b: Vec<Vec<f64>> = vec![Vec::new(); masks.len()];
    let mut divergence: Vec<f64> = Vec::new();
    let mut diverge_step: Option<usize> = None;
    for bin in 0..n_bins {
        let (mut ca, mut cb) = (vec![0i32; n], vec![0i32; n]);
        for _ in 0..bin_size {
            br_a.step();
            br_b.step();
            for &i in br_a.sim.spikes() {
                ca[i as usize] += 1;
            }
            for &i in br_b.sim.spikes() {
                cb[i as usize] += 1;
            }
        }
        let dt_bin = bin_size as f64 * 0.005;
        let mut div_sum = 0.0;
        for (k, (_, idx)) in masks.iter().enumerate() {
            let ra = idx.iter().map(|&i| f64::from(ca[i])).sum::<f64>() / idx.len() as f64 / dt_bin;
            let rb = idx.iter().map(|&i| f64::from(cb[i])).sum::<f64>() / idx.len() as f64 / dt_bin;
            tl_a[k].push(ra);
            tl_b[k].push(rb);
            div_sum += (rb - ra).abs();
        }
        let mean_div = div_sum / masks.len().max(1) as f64;
        divergence.push(mean_div);
        if diverge_step.is_none() && mean_div > 1.5 {
            diverge_step = Some(bin * bin_size);
        }
    }
    if group.is_some() {
        br_a.undrive(&stim_rows);
        br_b.undrive(&stim_rows);
    }
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { crate::validation::np_sum(v) / v.len() as f64 };
    let mut rows: Vec<Value> = masks
        .iter()
        .enumerate()
        .map(|(k, (name, _))| {
            let (ma, mb) = (mean(&tl_a[k]), mean(&tl_b[k]));
            let diff = mb - ma;
            let ratio = (mb + 0.05) / (ma + 0.05);
            json!({"name": name, "rate_a": ma, "rate_b": mb, "diff": diff, "ratio": ratio, "log2_ratio": ratio.max(1e-4).log2(),
                   "divergence": diff.abs()})
        })
        .collect();
    rows.sort_by(|x, y| y["divergence"].as_f64().unwrap_or(0.0).total_cmp(&x["divergence"].as_f64().unwrap_or(0.0)));
    let top = rows.first().cloned().unwrap_or(Value::Null);
    let rates_a = br_a.sim.activity.rates();
    let rates_b = br_b.sim.activity.rates();
    let mut readouts = serde_json::Map::new();
    let mut order = Vec::new();
    for (name, types) in
        [("giant_fiber_DNp01", &["DNp01"][..]), ("sugar_MN9", &["MN9"][..]), ("steering_DNa01_02", &["DNa01", "DNa02"][..])]
    {
        let sel: Vec<usize> = (0..n).filter(|&i| types.contains(&labels.types[i].as_str())).collect();
        if sel.is_empty() {
            continue;
        }
        let hz = |r: &[f32]| {
            let v: Vec<f32> = sel.iter().map(|&i| r[i]).collect();
            // float32 mean, then float32 / 0.005, as NumPy 2 evaluates `float32_array.mean() / 0.005`
            f64::from(np_sum_f32(&v) / v.len() as f32 / 0.005f32)
        };
        let (ha, hb) = (hz(rates_a), hz(rates_b));
        readouts.insert(name.into(), json!({"rate_a": ha, "rate_b": hb, "diff": hb - ha}));
        order.push(name);
    }
    json!({
        "kind": "connectome_diff", "created": created, "seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0,
        "stimulus": stimulus, "steps": steps, "seed": seed, "label_a": a.label, "label_b": b.label,
        "diverge_step": diverge_step, "diverge_time_s": diverge_step.map(|s| s as f64 * 0.005),
        "divergence_timeline": divergence, "timeline_bin_ms": bin_size as f64 * 5.0,
        "rows": rows, "top_divergence": top, "readouts": readouts, "readout_order": order,
    })
}

/// `save`: `connectome_diff.json` and `connectome_diff.csv`.
pub fn save(res: &Value, folder: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(folder)?;
    std::fs::write(folder.join("connectome_diff.json"), serde_json::to_string_pretty(res).unwrap_or_default())?;
    let f = |v: &Value| v.as_f64().unwrap_or(0.0);
    let (la, lb) = (res["label_a"].as_str().unwrap_or(""), res["label_b"].as_str().unwrap_or(""));
    let mut text = format!("region,rate_{la}_hz,rate_{lb}_hz,diff_hz,ratio\r\n");
    for r in res["rows"].as_array().into_iter().flatten() {
        let name = r["name"].as_str().unwrap_or("");
        text.push_str(&format!("{name},{:.2},{:.2},{:+.2},{:.2}\r\n", f(&r["rate_a"]), f(&r["rate_b"]), f(&r["diff"]), f(&r["ratio"])));
    }
    text.push_str("\r\ndivergence_timeline_time_ms,mean_regional_divergence_hz\r\n");
    let bin_ms = res["timeline_bin_ms"].as_f64().unwrap_or(25.0);
    for (i, d) in res["divergence_timeline"].as_array().into_iter().flatten().enumerate() {
        text.push_str(&format!("{:.1},{:.3}\r\n", i as f64 * bin_ms, f(d)));
    }
    std::fs::write(folder.join("connectome_diff.csv"), text)
}
