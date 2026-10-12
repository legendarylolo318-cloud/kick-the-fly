//! Psychometric curve generator (`lab/psychometrics.py`): sweep an experimental parameter across headless trials, mean with SEM and a 95 %
//! interval, and export CSV, SVG and PDF figures.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::assay_jobs;
use crate::cards::{Build, py_float};
use crate::csr::Weights;
use crate::pack::Labels;
use crate::validation::{self, np_sum};

pub struct Preset {
    pub key: &'static str,
    pub label: &'static str,
    pub param_name: &'static str,
    pub param_label: &'static str,
    pub x_values: &'static [f64],
    pub metric_primary: &'static str,
    pub metric_primary_label: &'static str,
    pub metric_secondary: Option<&'static str>,
    pub metric_secondary_label: Option<&'static str>,
    pub y_max: f64,
}

pub const PRESETS: [Preset; 3] = [
    Preset {
        key: "looming",
        label: "Looming escape vs approach speed",
        param_name: "approach_speed",
        param_label: "Approach speed (m/s)",
        x_values: &[0.5, 1.0, 2.0, 3.5, 5.5, 8.0, 11.0],
        metric_primary: "escape_prob",
        metric_primary_label: "Escape probability",
        metric_secondary: Some("latency_s"),
        metric_secondary_label: Some("Latency (s)"),
        y_max: 1.0,
    },
    Preset {
        key: "sugar",
        label: "Proboscis extension vs sugar concentration",
        param_name: "sugar_concentration",
        param_label: "Sugar concentration (fraction of sweet GRNs)",
        x_values: &[0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
        metric_primary: "response_rate",
        metric_primary_label: "Proboscis extension rate",
        metric_secondary: Some("mn9_ratio"),
        metric_secondary_label: Some("MN9 rate ratio (during/before)"),
        y_max: 1.0,
    },
    Preset {
        key: "tmaze",
        label: "Memory performance index vs conditioning pairings",
        param_name: "pairings",
        param_label: "Conditioning cycles (CS+ / shock pairings)",
        x_values: &[1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
        metric_primary: "pi",
        metric_primary_label: "Performance index (PI)",
        metric_secondary: None,
        metric_secondary_label: None,
        y_max: 1.0,
    },
];

pub fn preset(key: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.key == key)
}

fn spec_json(p: &Preset) -> Value {
    json!({
        "label": p.label, "param_name": p.param_name, "param_label": p.param_label, "x_values": p.x_values,
        "metric_primary": p.metric_primary, "metric_primary_label": p.metric_primary_label,
        "metric_secondary": p.metric_secondary, "metric_secondary_label": p.metric_secondary_label, "y_max": p.y_max,
    })
}

/// `_calc_stats`.
fn calc_stats(vals: &[f64]) -> Value {
    let v: Vec<f64> = vals.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return json!({"mean": Value::Null, "sem": 0.0, "ci95": 0.0, "n": 0});
    }
    let n = v.len();
    let m = np_sum(&v) / n as f64;
    let sd = if n > 1 {
        let sq: Vec<f64> = v.iter().map(|x| (x - m) * (x - m)).collect();
        (np_sum(&sq) / (n as f64 - 1.0)).sqrt()
    } else {
        0.0
    };
    let sem = if n > 1 { sd / (n as f64).sqrt() } else { 0.0 };
    json!({"mean": m, "sem": sem, "ci95": 1.96 * sem, "n": n})
}

fn mean(v: &[f64]) -> f64 {
    np_sum(v) / v.len() as f64
}

/// One fly at one x: `(primary value, secondary value)` of `run_sweep`.
fn one(build: &Build, g: &std::collections::HashMap<&'static str, Vec<u32>>, target: &str, x: f64, seed: u64) -> (f64, Option<f64>) {
    match target {
        "looming" => {
            let res = assay_jobs::looming_fly(build, g, seed, &[x], 2, &[]);
            let trials = res["trials"].as_object().and_then(|o| o.values().next()).and_then(Value::as_array).cloned().unwrap_or_default();
            let esc: Vec<f64> = trials.iter().map(|t| if t["escaped"].as_bool().unwrap_or(false) { 1.0 } else { 0.0 }).collect();
            let lats: Vec<f64> =
                trials.iter().filter(|t| t["escaped"].as_bool().unwrap_or(false)).filter_map(|t| t["latency_s"].as_f64()).collect();
            (mean(&esc), (!lats.is_empty()).then(|| mean(&lats)))
        }
        "sugar" => {
            let res = assay_jobs::sugar_fly(build, g, seed, &[x], 2, &[]);
            let offers = res["offers"].as_object().and_then(|o| o.values().next()).and_then(Value::as_array).cloned().unwrap_or_default();
            let ext: Vec<f64> = offers.iter().map(|o| if o["extended"].as_bool().unwrap_or(false) { 1.0 } else { 0.0 }).collect();
            let ratio: Vec<f64> = offers.iter().map(|o| o["ratio"].as_f64().unwrap_or(f64::NAN)).collect();
            (mean(&ext), Some(mean(&ratio)))
        }
        _ => {
            let cycles = x as usize;
            let a = validation::tmaze_fly_with(build, g, seed, "odor_a", cycles, 20, true, 0.08, &[]);
            let b = validation::tmaze_fly_with(build, g, seed + 50_000, "odor_b", cycles, 20, true, 0.08, &[]);
            ((a.pi + b.pi) / 2.0, None)
        }
    }
}

/// `run_sweep`: `lif` are the Lab's model parameters; `progress(done, total)` returns false to cancel.
#[allow(clippy::too_many_arguments)]
pub fn run_sweep(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    target: &str,
    x_values: Option<&[f64]>,
    n_flies: usize,
    base_seed: u64,
    lif: [f64; 5],
    workers: usize,
    version: &str,
    created: &str,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<Value, String> {
    let p = preset(target).ok_or_else(|| format!("Unknown target '{target}', must be one of [looming, sugar, tmaze]"))?;
    let xs: Vec<f64> = x_values.map_or_else(|| p.x_values.to_vec(), <[f64]>::to_vec);
    let seeds: Vec<u64> = (0..n_flies as u64).map(|i| base_seed + i).collect();
    let tasks: Vec<(usize, usize)> = (0..xs.len()).flat_map(|xi| (0..n_flies).map(move |si| (xi, si))).collect();
    let total = tasks.len();
    let next = AtomicUsize::new(0);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, (f64, Option<f64>))>();
    let t0 = std::time::Instant::now();
    let mut out: Vec<Option<(f64, Option<f64>)>> = vec![None; total];
    let mut cancelled = false;
    let (tasks_ref, xs_ref, seeds_ref) = (&tasks, &xs, &seeds);
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1).min(total.max(1)) {
            let tx = tx.clone();
            let (next, stop) = (&next, &stop);
            scope.spawn(move || {
                let build = Build { weights, labels, mode: "off", sigma: None, lif, wired: None };
                let g = validation::groups(labels, weights);
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(xi, si)) = tasks_ref.get(i) else { return };
                    let r = one(&build, &g, target, xs_ref[xi], seeds_ref[si]);
                    if tx.send((i, r)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0;
        while let Ok((i, r)) = rx.recv() {
            out[i] = Some(r);
            done += 1;
            if !progress(done, total) {
                cancelled = true;
                stop.store(true, Ordering::Relaxed);
            }
        }
    });
    if cancelled {
        return Err("cancelled".into());
    }
    let mut points = Vec::new();
    for (xi, &x) in xs.iter().enumerate() {
        let vals: Vec<(f64, Option<f64>)> = (0..n_flies).filter_map(|si| out[xi * n_flies + si]).collect();
        let prim: Vec<f64> = vals.iter().map(|v| v.0).collect();
        let sec: Vec<f64> = vals.iter().filter_map(|v| v.1).collect();
        // Python appends a secondary value only for flies that have one (latency) and for every fly otherwise
        let secondary = if p.metric_secondary.is_some() { calc_stats(&sec) } else { Value::Null };
        points.push(json!({"x": x, "primary": calc_stats(&prim), "secondary": secondary}));
    }
    let caption = format!(
        "Kick the Fly v{version} | Target: {} | Seeds: {base_seed}..{} (n={n_flies}) | Default LIF sim (dt=5ms, tau=20ms) | Generated: {created}",
        p.label,
        base_seed + n_flies as u64 - 1
    );
    Ok(json!({
        "target": target, "spec": spec_json(p), "xs": xs, "n_flies": n_flies, "base_seed": base_seed, "seeds": seeds, "points": points,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "caption": caption, "app_version": version, "created": created,
    }))
}

// --- files -------------------------------------------------------------------------------------------------------------------
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// `export_csv`.
pub fn export_csv(res: &Value, path: &Path) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let spec = &res["spec"];
    let pm = spec["metric_primary"].as_str().unwrap_or("");
    let sec = spec["metric_secondary"].as_str();
    let mut text = format!("{}\r\n", csv_field(&format!("# {}", res["caption"].as_str().unwrap_or(""))));
    text.push_str(&format!("{},n,{pm}_mean,{pm}_sem,{pm}_ci95", spec["param_name"].as_str().unwrap_or("")));
    if let Some(s) = sec {
        text.push_str(&format!(",{s}_mean,{s}_sem"));
    }
    text.push_str("\r\n");
    for pt in res["points"].as_array().into_iter().flatten() {
        let p = &pt["primary"];
        let mut row = format!("{},{},{:.4},{:.4},{:.4}", py_float(f(&pt["x"])), p["n"], f(&p["mean"]), f(&p["sem"]), f(&p["ci95"]))
            .replace("NaN", "nan");
        if sec.is_some() && pt["secondary"].is_object() {
            let s = &pt["secondary"];
            row.push_str(&format!(",{:.4},{:.4}", f(&s["mean"]), f(&s["sem"])).replace("NaN", "nan"));
        }
        text.push_str(&row);
        text.push_str("\r\n");
    }
    std::fs::write(path, text)
}

/// Python's `f"{x:g}"`.
fn g(x: f64) -> String {
    crate::sensitivity::fmt_g(x)
}

fn xs_of(res: &Value) -> (Vec<f64>, f64, f64) {
    let xs: Vec<f64> = res["points"].as_array().into_iter().flatten().map(|p| f(&p["x"])).collect();
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let mut max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == min {
        max += 1.0;
    }
    (xs, min, max)
}

/// A value Python holds as an `int` (the layout constants).
fn pi(x: f64) -> String {
    format!("{}", x as i64)
}

/// `export_svg`.
pub fn export_svg(res: &Value, path: &Path) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let spec = &res["spec"];
    let pts: Vec<&Value> = res["points"].as_array().into_iter().flatten().collect();
    let (w, h) = (800.0f64, 560.0f64);
    let (pad_l, pad_r, pad_t, pad_b) = (90.0f64, 40.0f64, 70.0f64, 100.0f64);
    let (plot_w, plot_h) = (w - pad_l - pad_r, h - pad_t - pad_b);
    let (_, x_min, x_max) = xs_of(res);
    let y_max = f(&spec["y_max"]);
    let y_min = 0.0;
    let tx = |x: f64| pad_l + (x - x_min) / (x_max - x_min) * plot_w;
    let ty = |y: f64| pad_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h;
    let (wi, hi) = (800, 560);
    let n_flies = res["n_flies"].as_u64().unwrap_or(0);
    let pf = py_float;
    let mut lines = vec![
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {wi} {hi}\" width=\"{wi}\" height=\"{hi}\" style=\"background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif;\">"
        ),
        format!("  <rect width=\"{wi}\" height=\"{hi}\" fill=\"#ffffff\"/>"),
        format!(
            "  <text x=\"{}\" y=\"36\" font-size=\"18\" font-weight=\"bold\" fill=\"#111827\">{}</text>",
            pi(pad_l),
            spec["label"].as_str().unwrap_or("")
        ),
        format!(
            "  <text x=\"{}\" y=\"56\" font-size=\"12\" fill=\"#4b5563\">n = {n_flies} flies per point (mean \u{b1} 95% CI)</text>",
            pi(pad_l)
        ),
    ];
    for i in 0..6 {
        let y_val = y_min + f64::from(i) * (y_max - y_min) / 5.0;
        let py = ty(y_val);
        lines.push(format!(
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>",
            pi(pad_l),
            pf(py),
            pi(pad_l + plot_w),
            pf(py)
        ));
        lines.push(format!(
            "  <text x=\"{}\" y=\"{}\" font-size=\"11\" text-anchor=\"end\" fill=\"#6b7280\">{y_val:.2}</text>",
            pi(pad_l - 12.0),
            pf(py + 4.0)
        ));
    }
    for pt in &pts {
        let px = tx(f(&pt["x"]));
        lines.push(format!(
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#9ca3af\" stroke-width=\"1.5\"/>",
            pf(px),
            pi(pad_t + plot_h),
            pf(px),
            pi(pad_t + plot_h + 6.0)
        ));
        lines.push(format!(
            "  <text x=\"{}\" y=\"{}\" font-size=\"11\" text-anchor=\"middle\" fill=\"#6b7280\">{}</text>",
            pf(px),
            pi(pad_t + plot_h + 20.0),
            g(f(&pt["x"]))
        ));
    }
    lines.push(format!(
        "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#374151\" stroke-width=\"1.5\"/>",
        pi(pad_l),
        pi(pad_t),
        pi(pad_l),
        pi(pad_t + plot_h)
    ));
    lines.push(format!(
        "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#374151\" stroke-width=\"1.5\"/>",
        pi(pad_l),
        pi(pad_t + plot_h),
        pi(pad_l + plot_w),
        pi(pad_t + plot_h)
    ));
    lines.push(format!(
        "  <text x=\"{}\" y=\"{}\" font-size=\"13\" font-weight=\"600\" text-anchor=\"middle\" fill=\"#1f2937\">{}</text>",
        pf(pad_l + plot_w / 2.0),
        pi(pad_t + plot_h + 46.0),
        spec["param_label"].as_str().unwrap_or("")
    ));
    lines.push(format!(
        "  <text x=\"24\" y=\"{0}\" font-size=\"13\" font-weight=\"600\" text-anchor=\"middle\" fill=\"#1f2937\" transform=\"rotate(-90 24 {0})\">{1}</text>",
        pf(pad_t + plot_h / 2.0),
        spec["metric_primary_label"].as_str().unwrap_or("")
    ));
    let poly: Vec<String> = pts.iter().map(|pt| format!("{:.1},{:.1}", tx(f(&pt["x"])), ty(f(&pt["primary"]["mean"])))).collect();
    lines.push(format!(
        "  <polyline points=\"{}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2.5\" stroke-linejoin=\"round\"/>",
        poly.join(" ")
    ));
    for pt in &pts {
        let px = tx(f(&pt["x"]));
        let pm = f(&pt["primary"]["mean"]);
        let py = ty(pm);
        let ci = f(&pt["primary"]["ci95"]);
        let y_hi = ty(y_max.min(pm + ci));
        let y_lo = ty(y_min.max(pm - ci));
        lines.push(format!(
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#1d4ed8\" stroke-width=\"1.5\"/>",
            pf(px),
            pf(y_lo),
            pf(px),
            pf(y_hi)
        ));
        lines.push(format!(
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#1d4ed8\" stroke-width=\"1.5\"/>",
            pf(px - 4.0),
            pf(y_hi),
            pf(px + 4.0),
            pf(y_hi)
        ));
        lines.push(format!(
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#1d4ed8\" stroke-width=\"1.5\"/>",
            pf(px - 4.0),
            pf(y_lo),
            pf(px + 4.0),
            pf(y_lo)
        ));
        lines.push(format!(
            "  <circle cx=\"{}\" cy=\"{}\" r=\"5\" fill=\"#2563eb\" stroke=\"#ffffff\" stroke-width=\"1.5\"/>",
            pf(px),
            pf(py)
        ));
    }
    lines.push(format!(
        "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>",
        pi(pad_l),
        hi - 34,
        pi(pad_l + plot_w),
        hi - 34
    ));
    lines.push(format!(
        "  <text x=\"{}\" y=\"{}\" font-size=\"10\" fill=\"#9ca3af\">{}</text>",
        pi(pad_l),
        hi - 18,
        res["caption"].as_str().unwrap_or("")
    ));
    lines.push("</svg>".into());
    std::fs::write(path, lines.join("\n"))
}

/// `export_pdf`: a standalone PDF-1.4 page with the figure.
pub fn export_pdf(res: &Value, path: &Path) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let spec = &res["spec"];
    let pts: Vec<&Value> = res["points"].as_array().into_iter().flatten().collect();
    let (pw, ph) = (792.0f64, 612.0f64);
    let (pad_l, pad_r, pad_t, pad_b) = (80.0f64, 50.0f64, 80.0f64, 90.0f64);
    let (plot_w, plot_h) = (pw - pad_l - pad_r, ph - pad_t - pad_b);
    let (_, x_min, x_max) = xs_of(res);
    let y_max = f(&spec["y_max"]);
    let y_min = 0.0;
    let tx = |x: f64| pad_l + (x - x_min) / (x_max - x_min) * plot_w;
    let ty = |y: f64| pad_b + (y - y_min) / (y_max - y_min) * plot_h;
    let pf = py_float;
    let mut cmds: Vec<String> = vec![format!("1 1 1 rg 0 0 {} {} re f", pw as i64, ph as i64), "0.85 0.85 0.85 RG 0.75 w".into()];
    for i in 0..6 {
        let py = ty(y_min + f64::from(i) * (y_max - y_min) / 5.0);
        cmds.push(format!("{} {py:.1} m {} {py:.1} l S", pad_l as i64, pi(pad_l + plot_w)));
    }
    cmds.push("0.3 0.3 0.3 RG 1.5 w".into());
    for pt in &pts {
        let px = tx(f(&pt["x"]));
        cmds.push(format!("{px:.1} {} m {px:.1} {} l S", pad_b as i64, pi(pad_b - 6.0)));
    }
    cmds.push(format!(
        "{} {} m {} {} l {} {} l S",
        pad_l as i64,
        pi(pad_b + plot_h),
        pad_l as i64,
        pad_b as i64,
        pi(pad_l + plot_w),
        pad_b as i64
    ));
    cmds.push("0.145 0.388 0.921 RG 2.5 w".into());
    let poly: Vec<(f64, f64)> = pts.iter().map(|pt| (tx(f(&pt["x"])), ty(f(&pt["primary"]["mean"])))).collect();
    cmds.push(format!("{:.1} {:.1} m", poly[0].0, poly[0].1));
    for (px, py) in &poly[1..] {
        cmds.push(format!("{px:.1} {py:.1} l"));
    }
    cmds.push("S".into());
    cmds.push("0.114 0.306 0.847 RG 1.5 w".into());
    for pt in &pts {
        let px = tx(f(&pt["x"]));
        let pm = f(&pt["primary"]["mean"]);
        let ci = f(&pt["primary"]["ci95"]);
        let (y_hi, y_lo) = (ty(y_max.min(pm + ci)), ty(y_min.max(pm - ci)));
        cmds.push(format!("{px:.1} {y_lo:.1} m {px:.1} {y_hi:.1} l S"));
        cmds.push(format!("{:.1} {y_hi:.1} m {:.1} {y_hi:.1} l S", px - 4.0, px + 4.0));
        cmds.push(format!("{:.1} {y_lo:.1} m {:.1} {y_lo:.1} l S", px - 4.0, px + 4.0));
    }
    cmds.push("0.145 0.388 0.921 rg 1 1 1 RG 1.5 w".into());
    for (px, py) in &poly {
        let r = 4.0f64;
        let k = 0.552284749831 * r;
        cmds.push(format!(
            "{:.1} {py:.1} m {:.1} {:.1} {:.1} {:.1} {px:.1} {:.1} c {:.1} {:.1} {:.1} {:.1} {:.1} {py:.1} c {:.1} {:.1} {:.1} {:.1} {px:.1} {:.1} c {:.1} {:.1} {:.1} {:.1} {:.1} {py:.1} c B",
            px + r,
            px + r,
            py + k,
            px + k,
            py + r,
            py + r,
            px - k,
            py + r,
            px - r,
            py + k,
            px - r,
            px - r,
            py - k,
            px - k,
            py - r,
            py - r,
            px + k,
            py - r,
            px + r,
            py - k,
            px + r
        ));
    }
    let clean = |s: &str| s.replace('\\', "\\\\").replace('(', "\\(").replace(')', "\\)");
    let mut t: Vec<String> = vec!["BT".into(), "/F1 16 Tf 0.1 0.1 0.1 rg".into()];
    t.push(format!("{} {} Td ({}) Tj", pad_l as i64, (ph - 42.0) as i64, clean(spec["label"].as_str().unwrap_or(""))));
    t.push("/F1 10 Tf 0.3 0.3 0.3 rg".into());
    t.push(format!("0 -18 Td (n = {} flies per point, mean +/- 95% CI) Tj", res["n_flies"]));
    t.push("/F1 12 Tf 0.15 0.15 0.15 rg".into());
    t.push(format!(
        "{} {} Td ({}) Tj",
        pf(pad_l + plot_w / 2.0 - 60.0),
        pi(pad_b - 36.0),
        clean(spec["param_label"].as_str().unwrap_or(""))
    ));
    t.push("/F1 9 Tf 0.4 0.4 0.4 rg".into());
    for i in 0..6 {
        let y_val = y_min + f64::from(i) * (y_max - y_min) / 5.0;
        let py = ty(y_val);
        t.push(format!("{} {:.1} Td ({y_val:.2}) Tj 0 0 Td", pi(pad_l - 30.0), py - 3.0));
    }
    for pt in &pts {
        let px = tx(f(&pt["x"]));
        t.push(format!("{:.1} {} Td ({}) Tj 0 0 Td", px - 8.0, pi(pad_b - 18.0), g(f(&pt["x"]))));
    }
    t.push("/F1 8 Tf 0.5 0.5 0.5 rg".into());
    t.push(format!("{} 28 Td ({}) Tj", pad_l as i64, clean(res["caption"].as_str().unwrap_or(""))));
    t.push("ET".into());
    let content: Vec<u8> = cmds
        .iter()
        .chain(&t)
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .map(|c| if (c as u32) < 256 { c as u8 } else { b'?' })
        .collect();
    let mut objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            pw as i64, ph as i64
        )
        .into_bytes(),
    ];
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(&content);
    stream.extend_from_slice(b"\nendstream");
    objects.push(stream);
    objects.push(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec());
    let mut body = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objects.iter().enumerate() {
        offsets.push(body.len());
        body.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        body.extend_from_slice(o);
        body.extend_from_slice(b"\nendobj\n");
    }
    let startxref = body.len();
    body.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
    for off in offsets {
        body.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    body.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n", objects.len() + 1).as_bytes());
    std::fs::write(path, body)
}

/// `save_all_formats`: `<base>.csv`, `.svg` and `.pdf` in `folder`.
pub fn save_all_formats(res: &Value, base_name: &str, folder: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(folder)?;
    export_csv(res, &folder.join(format!("{base_name}.csv")))?;
    export_svg(res, &folder.join(format!("{base_name}.svg")))?;
    export_pdf(res, &folder.join(format!("{base_name}.pdf")))
}
