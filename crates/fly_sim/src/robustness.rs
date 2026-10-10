//! How much do this simulation's results depend on the reconstruction being right (`lab/robustness.py`): the validation suite's behaviors
//! re-run on a changed connectome (a synapse threshold, sign flips of uncertain neurons, an inhibition block), with validation's own pass
//! criteria.

use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::cards::{Build, LIF_DEFAULTS};
use crate::csr::Weights;
use crate::pack::Labels;
use crate::stats;
use crate::validation::{self, np_sum};
use crate::wiring::{self, Wiring};

pub const DEFAULT_TESTS: [&str; 4] = ["looming_escape", "sugar_feeding", "antenna_grooming_circuit", "mb_conditioning"];
pub const DEFAULT_THRESHOLDS: [i64; 6] = [1, 4, 5, 6, 8, 10];
pub const MIN_PACK_SYNAPSES: i64 = 3;
pub const FLIP_TESTS: [&str; 3] = ["looming_escape", "sugar_feeding", "antenna_grooming_circuit"];
pub const DEFAULT_TRIALS: usize = 5;
pub const DEFAULT_CUTOFF: f64 = 0.7;
pub const DEFAULT_SHARE: f64 = 0.5;

/// Everything a run needs: the connectome, its labels, the raw counts.
pub struct Ctx<'a> {
    pub weights: &'a Arc<Weights>,
    pub labels: &'a Arc<Labels>,
    pub counts: &'a [i32],
    pub workers: usize,
}

fn results_by_id(res: &Value) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    for t in res["tests"].as_array().into_iter().flatten() {
        let m = &t["measured"];
        let f = |k: &str, d: f64| m[k].as_f64().unwrap_or(d);
        let is_ratio = m.get("drive_ratio_mean").is_some();
        out.insert(
            t["id"].as_str().unwrap_or("").to_string(),
            json!({
                "passed": t["passed"].as_bool().unwrap_or(false), "name": t["name"],
                "effect": if is_ratio { f("drive_ratio_mean", 0.0) } else { f("pi_mean", 0.0) },
                "control": if m.get("control_ratio_mean").is_some() { f("control_ratio_mean", 0.0) } else { f("control_pi_mean", 0.0) },
                "p_value": f("p_value", 1.0),
                "base_hz": f("readout_base_hz", f64::NAN), "driven_hz": f("readout_driven_hz", f64::NAN),
                "metric": if is_ratio { "drive/baseline ratio" } else { "performance index" },
            }),
        );
    }
    out
}

/// `run_behaviors`: the validation suite's behaviors on a changed connectome.
pub fn run_behaviors(
    c: &Ctx,
    w: &Wiring,
    seeds: &[u64],
    include: &[&str],
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<serde_json::Map<String, Value>, String> {
    let wired = (!w.is_identity()).then(|| Arc::new(wiring::apply(c.weights, c.labels, c.counts, w)));
    let res = validation::run_with(c.weights, c.labels, seeds, Some(include), c.workers, LIF_DEFAULTS, wired.as_ref(), |d, n, l| {
        progress(d, n, l);
        true
    })?;
    Ok(results_by_id(&res))
}

/// `threshold_sweep`.
pub fn threshold_sweep(
    c: &Ctx,
    thresholds: &[i64],
    seeds: Option<&[u64]>,
    include: &[&str],
    created: &str,
    progress: &mut dyn FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let seeds: Vec<u64> = seeds.map_or_else(validation::seeds, <[u64]>::to_vec);
    let mut th: Vec<i64> = thresholds.iter().map(|&t| t.max(1)).collect();
    th.sort_unstable();
    th.dedup();
    let t0 = std::time::Instant::now();
    let mut steps: Vec<Value> = Vec::new();
    for (i, &n) in th.iter().enumerate() {
        let stats = wiring::threshold_stats(&c.weights.csr, c.counts, n);
        let mut cancelled = false;
        let res = run_behaviors(c, &Wiring { min_synapses: n, ..Wiring::default() }, &seeds, include, &mut |d, tot, _| {
            if !progress(i, th.len(), &format!("threshold {n}: {d}/{tot}")) {
                cancelled = true;
            }
        })?;
        if cancelled {
            return Err("cancelled".into());
        }
        steps.push(json!({"threshold": n, "stats": stats, "behaviors": res}));
    }
    let mut breaks = serde_json::Map::new();
    for id in include {
        let passed_at: Vec<i64> =
            steps.iter().filter(|s| s["behaviors"][*id]["passed"] == true).map(|s| s["threshold"].as_i64().unwrap_or(0)).collect();
        let failed_at: Vec<i64> =
            steps.iter().filter(|s| s["behaviors"][*id]["passed"] == false).map(|s| s["threshold"].as_i64().unwrap_or(0)).collect();
        let mut first_fail = None;
        for s in &steps {
            if s["behaviors"][*id]["passed"] == false {
                let t = s["threshold"].as_i64().unwrap_or(0);
                let later: Vec<&Value> =
                    steps.iter().filter(|x| x["threshold"].as_i64().unwrap_or(0) >= t && x["behaviors"].get(*id).is_some()).collect();
                if later.iter().all(|x| x["behaviors"][*id]["passed"] == false) {
                    first_fail = Some(t);
                    break;
                }
            }
        }
        let name = steps.first().and_then(|s| s["behaviors"][*id]["name"].as_str()).unwrap_or(id).to_string();
        breaks.insert((*id).into(), json!({"breaks_at": first_fail, "survived": passed_at, "failed": failed_at, "name": name}));
    }
    Ok(json!({
        "kind": "threshold_sweep", "created": created, "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "seeds": seeds, "thresholds": th, "pack_min_synapses": MIN_PACK_SYNAPSES, "steps": steps, "breaks": breaks,
        "order": include,
    }))
}

/// 95 % Wilson interval for k of n trials.
fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let z = 1.96f64;
    let (p, nf) = (k as f64 / n as f64, n as f64);
    let d = 1.0 + z * z / nf;
    let centre = (p + z * z / (2.0 * nf)) / d;
    let half = z * ((p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt()) / d;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// `signflip_trials`.
#[allow(clippy::too_many_arguments)]
pub fn signflip_trials(
    c: &Ctx,
    cutoff: f64,
    share: f64,
    trials: usize,
    seeds: Option<&[u64]>,
    include: &[&str],
    created: &str,
    progress: &mut dyn FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let seeds: Vec<u64> = seeds.map_or_else(validation::seeds, <[u64]>::to_vec);
    let cand_stats = wiring::confidence_stats(c.labels, cutoff);
    let t0 = std::time::Instant::now();
    let total = trials + 1;
    let cancelled = std::cell::Cell::new(false);
    let progress = std::cell::RefCell::new(progress);
    let sub = |i: usize, label: String| {
        let (progress, cancelled) = (&progress, &cancelled);
        move |d: usize, n: usize, _l: &str| {
            if !(progress.borrow_mut())(i, total, &format!("{label}: {d}/{n}")) {
                cancelled.set(true);
            }
        }
    };
    let control = run_behaviors(c, &Wiring::default(), &seeds, include, &mut sub(0, "unperturbed control".into()))?;
    let mut runs: Vec<Value> = Vec::new();
    for t in 0..trials {
        let rows = wiring::random_flip(c.labels, cutoff, share, t as u64);
        let n_rows = rows.len();
        let res = run_behaviors(
            c,
            &Wiring { flip_rows: rows, ..Wiring::default() },
            &seeds,
            include,
            &mut sub(t + 1, format!("trial {}", t + 1)),
        )?;
        runs.push(json!({"trial": t + 1, "flipped": n_rows, "behaviors": res}));
    }
    if cancelled.get() {
        return Err("cancelled".into());
    }
    let mut per = serde_json::Map::new();
    for id in include {
        let passed = runs.iter().filter(|r| r["behaviors"][*id]["passed"] == true).count();
        let effects: Vec<f64> = runs
            .iter()
            .filter(|r| r["behaviors"].get(*id).is_some())
            .map(|r| r["behaviors"][*id]["effect"].as_f64().unwrap_or(f64::NAN))
            .collect();
        let (lo, hi) = wilson(passed, runs.len());
        let ci = stats::mean_ci(&effects);
        let base = control.get(*id).cloned().unwrap_or(Value::Null);
        per.insert(
            (*id).into(),
            json!({
                "name": base.get("name").cloned().unwrap_or_else(|| json!(id)),
                "metric": base.get("metric").cloned().unwrap_or_else(|| json!("")),
                "control_effect": base.get("effect"), "control_passed": base.get("passed"),
                "survived": passed, "trials": runs.len(), "survival": passed as f64 / runs.len().max(1) as f64,
                "survival_ci": [lo, hi], "effect_mean": ci.mean, "effect_ci": [ci.lo, ci.hi], "effect_sd": ci.sd,
                "effects": effects,
                "verdict": if passed == runs.len() { "survives" } else if passed == 0 { "breaks" } else { "sometimes breaks" },
            }),
        );
    }
    Ok(json!({
        "kind": "signflip_trials", "created": created, "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "seeds": seeds, "cutoff": cutoff, "share": share, "trials": trials, "trial_seed": 0,
        "candidates": cand_stats, "control": control, "runs": runs, "behaviors": per, "order": include,
    }))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let virt = (sorted.len() - 1) as f64 * (p / 100.0);
    let lo = virt.floor();
    let (a, b) = (sorted[lo as usize], sorted[(lo as usize + 1).min(sorted.len() - 1)]);
    let g = virt - lo;
    let d = b - a;
    if g >= 0.5 { b - d * (1.0 - g) } else { a + d * g }
}

/// `measure_firing_distribution`: the firing-rate distribution across the whole connectome under a wiring.
pub fn measure_firing_distribution(c: &Ctx, w: &Wiring, seeds: &[u64], steps: usize) -> Value {
    let wired = (!w.is_identity()).then(|| Arc::new(wiring::apply(c.weights, c.labels, c.counts, w)));
    let n = c.weights.n();
    let mut all: Vec<Vec<f64>> = Vec::new();
    for &s in seeds {
        let b = Build { weights: c.weights, labels: c.labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: wired.as_ref() };
        let mut br = b.brain_with(s, None, 150, true);
        let mut counts = vec![0u32; n];
        for _ in 0..steps {
            for &i in br.step() {
                counts[i as usize] += 1;
            }
        }
        all.push(counts.iter().map(|&k| f64::from(k) / (steps as f64 * 0.005)).collect());
    }
    let mean_rates: Vec<f64> = if all.len() > 1 {
        let mut acc = vec![0.0f64; n];
        for r in &all {
            for (a, x) in acc.iter_mut().zip(r) {
                *a += x;
            }
        }
        acc.iter().map(|a| a / all.len() as f64).collect()
    } else {
        all.remove(0)
    };
    let bins = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0];
    let mut hist = [0u64; 6];
    for &r in &mean_rates {
        if let Some(k) = (0..6).find(|&k| r >= bins[k] && (r < bins[k + 1] || (k == 5 && r <= bins[6]))) {
            hist[k] += 1;
        }
    }
    let mut sorted = mean_rates.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let len = mean_rates.len();
    let median = if len % 2 == 1 { sorted[len / 2] } else { (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0 };
    json!({
        "mean": np_sum(&mean_rates) / len as f64, "median": median, "p95": percentile(&sorted, 95.0), "max": sorted[len - 1],
        "active_share": mean_rates.iter().filter(|&&r| r > 0.1).count() as f64 / len as f64,
        "histogram": hist, "bin_labels": ["<2 Hz", "2-5 Hz", "5-10 Hz", "10-20 Hz", "20-40 Hz", ">40 Hz"],
        "bin_edges": [0, 2, 5, 10, 20, 40, 100], "bin_shares": hist.iter().map(|&h| h as f64 / len as f64).collect::<Vec<_>>(),
        "n_neurons": len,
    })
}

/// `inhibition_report`.
pub fn inhibition_report(c: &Ctx, scale: f64, seeds: &[u64], steps: usize, created: &str) -> Value {
    let stats = wiring::inhibition_stats(&c.weights.csr, c.labels, scale);
    let t0 = std::time::Instant::now();
    let before = measure_firing_distribution(c, &Wiring::default(), seeds, steps);
    let after = measure_firing_distribution(c, &Wiring { inhibition_scale: scale, ..Wiring::default() }, seeds, steps);
    json!({
        "kind": "inhibition_block", "created": created, "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "scale": scale, "severity": stats["severity"], "stats": stats, "before": before, "after": after, "seeds": seeds,
    })
}

// --- files -------------------------------------------------------------------------------------------------------------------
fn csv(path: &Path, header: &[&str], rows: Vec<Vec<String>>) -> std::io::Result<()> {
    let mut t = header.join(",") + "\r\n";
    for r in rows {
        t.push_str(&r.join(","));
        t.push_str("\r\n");
    }
    std::fs::write(path, t)
}

fn n4(v: &Value) -> String {
    format!("{:.4}", v.as_f64().unwrap_or(f64::NAN)).replace("NaN", "nan")
}

fn n5(v: &Value) -> String {
    format!("{:.5}", v.as_f64().unwrap_or(f64::NAN)).replace("NaN", "nan")
}

fn pass(v: &Value) -> String {
    if v.as_bool().unwrap_or(false) { "pass" } else { "fail" }.to_string()
}

/// `save`: `<kind>.json` and the CSV of that kind.
pub fn save(res: &Value, folder: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(folder)?;
    let kind = res["kind"].as_str().unwrap_or("result");
    std::fs::write(folder.join(format!("{kind}.json")), serde_json::to_string_pretty(res).unwrap_or_default())?;
    match kind {
        "threshold_sweep" => {
            let mut rows = Vec::new();
            for step in res["steps"].as_array().into_iter().flatten() {
                let st = &step["stats"];
                for (id, b) in step["behaviors"].as_object().into_iter().flatten() {
                    rows.push(vec![
                        id.clone(),
                        step["threshold"].to_string(),
                        st["connections_dropped"].to_string(),
                        n4(&st["connections_dropped_share"]),
                        st["neurons_cut_off"].to_string(),
                        b["metric"].as_str().unwrap_or("").into(),
                        n4(&b["effect"]),
                        n4(&b["control"]),
                        n4(&b["base_hz"]),
                        n4(&b["driven_hz"]),
                        n5(&b["p_value"]),
                        pass(&b["passed"]),
                    ]);
                }
            }
            csv(
                &folder.join("threshold_sweep.csv"),
                &[
                    "behavior",
                    "threshold",
                    "connections_dropped",
                    "connections_dropped_share",
                    "neurons_cut_off",
                    "metric",
                    "effect",
                    "control",
                    "readout_base_hz",
                    "readout_driven_hz",
                    "p_value",
                    "passed",
                ],
                rows,
            )?;
        }
        "signflip_trials" => {
            let mut rows = Vec::new();
            let mut all: Vec<(i64, i64, &Value)> = res["runs"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| (r["trial"].as_i64().unwrap_or(0), r["flipped"].as_i64().unwrap_or(0), &r["behaviors"]))
                .collect();
            all.push((0, 0, &res["control"]));
            for (trial, flipped, behaviors) in all {
                for (id, b) in behaviors.as_object().into_iter().flatten() {
                    rows.push(vec![
                        id.clone(),
                        trial.to_string(),
                        flipped.to_string(),
                        b["metric"].as_str().unwrap_or("").into(),
                        n4(&b["effect"]),
                        n4(&res["control"][id]["effect"]),
                        n4(&b["base_hz"]),
                        n4(&b["driven_hz"]),
                        n5(&b["p_value"]),
                        pass(&b["passed"]),
                    ]);
                }
            }
            csv(
                &folder.join("signflip_trials.csv"),
                &[
                    "behavior",
                    "trial",
                    "neurons_flipped",
                    "metric",
                    "effect",
                    "control_effect",
                    "readout_base_hz",
                    "readout_driven_hz",
                    "p_value",
                    "passed",
                ],
                rows,
            )?;
        }
        "inhibition_block" => {
            let f2 = |v: &Value| format!("{:.2}", v.as_f64().unwrap_or(f64::NAN));
            let (b, a) = (&res["before"], &res["after"]);
            let scale = format!("{:.2}", res["scale"].as_f64().unwrap_or(0.0));
            let mut text = format!("metric,before_100pct_inhibition,after_scale_{scale}\r\n");
            for (name, k) in [("mean_hz", "mean"), ("median_hz", "median"), ("p95_hz", "p95"), ("max_hz", "max")] {
                text.push_str(&format!("{name},{},{}\r\n", f2(&b[k]), f2(&a[k])));
            }
            text.push_str(&format!("active_neurons_share,{},{}\r\n\r\n", n4(&b["active_share"]), n4(&a["active_share"])));
            text.push_str("firing_rate_bin,before_count,before_share,after_count,after_share\r\n");
            for i in 0..6 {
                text.push_str(&format!(
                    "{},{},{},{},{}\r\n",
                    b["bin_labels"][i].as_str().unwrap_or(""),
                    b["histogram"][i],
                    n4(&b["bin_shares"][i]),
                    a["histogram"][i],
                    n4(&a["bin_shares"][i])
                ));
            }
            std::fs::write(folder.join("inhibition_block.csv"), text)?;
        }
        _ => {}
    }
    Ok(())
}
