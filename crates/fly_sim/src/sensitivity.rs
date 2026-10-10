//! Sensitivity analysis (`lab/sensitivity.py`): vary each simulation parameter one at a time and re-run the validated behaviors with the
//! validation suite's own code and criteria. Analysis only: the defaults never change.

use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::csr::Weights;
use crate::pack::Labels;
use crate::validation::{self, np_sum};
use crate::wiring::{self, Wiring};

pub struct Param {
    pub id: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub default: f64,
    pub values: &'static [f64],
    pub note: &'static str,
}

pub const PARAMETERS: [Param; 6] = [
    Param {
        id: "noise_std",
        label: "Membrane noise",
        kind: "lif",
        default: 0.05,
        values: &[0.025, 0.0375, 0.075, 0.10],
        note: "Random current on every neuron each 5 ms step. The code notes that 0.10 drowned vision and 0.03 was too quiet.",
    },
    Param {
        id: "bias",
        label: "Tonic drive",
        kind: "lif",
        default: 0.20,
        values: &[0.15, 0.175, 0.225, 0.25],
        note: "Constant input to every neuron. At 0.20 a neuron rests at 80% of threshold; 0.25 puts it at threshold.",
    },
    Param {
        id: "target_rate_hz",
        label: "Target rate (Hz)",
        kind: "lif",
        default: 5.0,
        values: &[2.5, 3.5, 7.5, 10.0],
        note: "The whole-brain firing rate the slow gain controller steers toward.",
    },
    Param {
        id: "ext_gain",
        label: "Sensory input gain",
        kind: "lif",
        default: 4.0,
        values: &[2.0, 3.0, 5.0, 6.0],
        note: "How strongly driven neurons are pushed: the stimulus strength of every pathway test.",
    },
    Param {
        id: "gain_adapt",
        label: "Gain adaptation rate",
        kind: "lif",
        default: 0.002,
        values: &[0.0, 0.001, 0.004, 0.008],
        note: "How fast the gain controller reacts; 0 freezes the synaptic gain at its starting value.",
    },
    Param {
        id: "min_synapses",
        label: "Synapse threshold",
        kind: "wiring",
        default: 3.0,
        values: &[4.0, 5.0, 6.0, 8.0, 10.0],
        note: "Keep only connections with at least this many synapses (the pack already keeps those with 3 or more).",
    },
];
pub const BASELINE: &str = "baseline";
pub const UNDERPOWERED_BELOW: usize = 7;
const NO_PARAMS: [&str; 4] = ["mb_extinction", "mb_second_order", "epg_compass", "epg_compass_wind"];
const EXPECTED: [&str; 12] = [
    "looming_escape",
    "sugar_feeding",
    "antenna_grooming_circuit",
    "mb_conditioning",
    "courtship_song",
    "bitter_grn_to_dng28",
    "co2_orn_to_pn",
    "hot_trn_to_vp2pn",
    "cold_trn_to_vp3pn",
    "optomotor_turning",
    "or67d_to_da1pn",
    "da1pn_to_lh_asp",
];

/// Python's `format(x, "g")`.
pub fn fmt_g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let e = format!("{x:.5e}"); // 6 significant digits
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&exp) {
        let m = mant.trim_end_matches('0').trim_end_matches('.');
        return format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let dec = (5 - exp).max(0) as usize;
    let s = format!("{x:.dec$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// The behaviors validation reproduces (expected to pass), in the suite's order.
pub fn validated_behaviors() -> Vec<String> {
    validation::tests().iter().filter_map(|t| t["id"].as_str()).filter(|id| EXPECTED.contains(id)).map(String::from).collect()
}

pub fn parameter_ids() -> Vec<&'static str> {
    PARAMETERS.iter().map(|p| p.id).collect()
}

/// The `(parameter, value)` cells, the baseline first.
pub fn cells(params: Option<&[&str]>, values: Option<&[(&str, Vec<f64>)]>) -> Result<Vec<(String, Option<f64>)>, String> {
    let ids: Vec<&str> = match params {
        Some(p) => p.to_vec(),
        None => parameter_ids(),
    };
    let mut out = vec![(BASELINE.to_string(), None)];
    for p in ids {
        let spec = PARAMETERS
            .iter()
            .find(|x| x.id == p)
            .ok_or_else(|| format!("unknown parameter '{p}'; use one of {}", parameter_ids().join(", ")))?;
        let vals: Vec<f64> = values
            .and_then(|v| v.iter().find(|(k, _)| *k == p).map(|(_, v)| v.clone()))
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| spec.values.to_vec());
        out.extend(vals.into_iter().map(|v| (p.to_string(), Some(v))));
    }
    Ok(out)
}

pub fn cell_key(param: &str, value: Option<f64>) -> String {
    match (param, value) {
        (BASELINE, _) | (_, None) => BASELINE.to_string(),
        (p, Some(v)) => format!("{p}={}", fmt_g(v)),
    }
}

fn std1(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = np_sum(x) / n;
    let sq: Vec<f64> = x.iter().map(|v| (v - mean) * (v - mean)).collect();
    (np_sum(&sq) / (n - 1.0)).sqrt()
}

/// `_effect_size`: the headline number, its control and Cohen's d_z of drive minus control over the seeds.
fn effect_size(t: &Value) -> Value {
    let m = &t["measured"];
    let per = t["per_seed"].as_array().cloned().unwrap_or_default();
    let (a, b, effect, control, metric): (Vec<f64>, Vec<f64>, f64, f64, &str) = if m.get("drive_ratio_mean").is_some() {
        (
            per.iter().map(|p| p["drive"]["ratio"].as_f64().unwrap_or(f64::NAN)).collect(),
            per.iter().map(|p| p["control"]["ratio"].as_f64().unwrap_or(f64::NAN)).collect(),
            m["drive_ratio_mean"].as_f64().unwrap_or(f64::NAN),
            m["control_ratio_mean"].as_f64().unwrap_or(f64::NAN),
            "drive/baseline ratio",
        )
    } else {
        (
            per.iter().map(|p| p["pi"].as_f64().unwrap_or(f64::NAN)).collect(),
            per.iter().map(|p| p["control_pi"].as_f64().unwrap_or(f64::NAN)).collect(),
            m["pi_mean"].as_f64().unwrap_or(f64::NAN),
            m["control_pi_mean"].as_f64().unwrap_or(f64::NAN),
            "performance index",
        )
    };
    let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let sd = if d.len() > 1 { std1(&d) } else { 0.0 };
    let mean = if d.is_empty() { f64::NAN } else { np_sum(&d) / d.len() as f64 };
    let dz = if sd > 1e-12 {
        mean / sd
    } else if d.iter().all(|v| v.abs() <= 1e-8) {
        0.0
    } else {
        f64::INFINITY
    };
    json!({
        "passed": t["passed"].as_bool().unwrap_or(false), "effect": effect, "control": control, "metric": metric,
        "effect_size_dz": if dz.is_finite() { json!(dz) } else { Value::Null },
        "p_value": m["p_value"].as_f64().unwrap_or(1.0),
        "n": m["n"].as_u64().unwrap_or(per.len() as u64),
    })
}

/// One `(parameter, value)` cell: the validation suite's code on the chosen behaviors, with that one change.
#[allow(clippy::too_many_arguments)]
pub fn run_cell(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    counts: Option<&[i32]>,
    param: &str,
    value: Option<f64>,
    seeds: &[u64],
    tests: &[String],
    workers: usize,
    progress: impl FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let mut lif = crate::cards::LIF_DEFAULTS;
    let mut wired: Option<Arc<Weights>> = None;
    if param != BASELINE {
        let v = value.ok_or("a cell needs a value")?;
        match PARAMETERS.iter().find(|p| p.id == param).map(|p| p.kind) {
            Some("wiring") => {
                let counts = counts.ok_or("the synapse counts of the pack are needed for the synapse threshold")?;
                let wr = Wiring { min_synapses: v.round_ties_even() as i64, ..Wiring::default() };
                wired = Some(Arc::new(wiring::apply(weights, labels, counts, &wr)));
            }
            Some(_) => {
                let k = crate::cards::LIF_KEYS.iter().position(|k| *k == param).ok_or("unknown parameter")?;
                lif[k] = v;
            }
            None => return Err(format!("unknown parameter '{param}'")),
        }
    }
    let inc: Vec<&str> = tests.iter().map(String::as_str).collect();
    let res = validation::run_with(weights, labels, seeds, Some(&inc), workers, lif, wired.as_ref(), progress)?;
    let mut by = serde_json::Map::new();
    for t in res["tests"].as_array().into_iter().flatten() {
        by.insert(t["id"].as_str().unwrap_or("").to_string(), effect_size(t));
    }
    Ok(json!({
        "key": cell_key(param, value), "parameter": param, "value": value, "behaviors": by,
        "backend": res.get("backend").cloned().unwrap_or(Value::Null), "seconds": res.get("seconds").cloned().unwrap_or(Value::Null),
    }))
}

/// `sha256(json.dumps(config, sort_keys=True))[:16]`.
fn config_hash(seeds: &[u64], tests: &[String], cell_list: &[(String, Option<f64>)]) -> String {
    let q = |v: &[String]| v.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", ");
    let keys: Vec<String> = cell_list.iter().map(|c| cell_key(&c.0, c.1)).collect();
    let s = format!(
        "{{\"cells\": [{}], \"seeds\": [{}], \"tests\": [{}]}}",
        q(&keys),
        seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(", "),
        q(tests)
    );
    crate::util::sha256_hex(s.as_bytes())[..16].to_string()
}

pub const PROGRESS_NAME: &str = "sensitivity_progress.json";

/// `run`: the analysis; `progress(done_cells, total_cells, label)` returns false to cancel. With `folder` the finished cells are kept in
/// `sensitivity_progress.json`; `resume` skips those already there (a file for another configuration is refused).
#[allow(clippy::too_many_arguments)]
pub fn run(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    counts: Option<&[i32]>,
    params: Option<&[&str]>,
    tests: Option<&[&str]>,
    seeds: Option<&[u64]>,
    values: Option<&[(&str, Vec<f64>)]>,
    workers: usize,
    folder: Option<&Path>,
    resume: bool,
    created: &str,
    mut progress: impl FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let seeds: Vec<u64> = seeds.map_or_else(validation::seeds, <[u64]>::to_vec);
    let tests: Vec<String> = tests.map_or_else(validated_behaviors, |t| t.iter().map(|s| s.to_string()).collect());
    let adult: Vec<&str> = validation::tests().iter().filter_map(|t| t["id"].as_str()).collect();
    for t in &tests {
        if !adult.contains(&t.as_str()) {
            return Err(format!("unknown adult behavior '{t}'"));
        }
        if NO_PARAMS.contains(&t.as_str()) {
            return Err(format!("{t} builds its flies with the default parameters, so varying them cannot change it; leave it out"));
        }
    }
    let cell_list = cells(params, values)?;
    let chash = config_hash(&seeds, &tests, &cell_list);
    let mut done: serde_json::Map<String, Value> = serde_json::Map::new();
    let path = folder.map(|f| f.join(PROGRESS_NAME));
    if let (Some(p), true, Ok(text)) = (&path, resume, path.as_ref().map_or(Err(()), |p| std::fs::read_to_string(p).map_err(|_| ()))) {
        let saved: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if saved["config_hash"].as_str() != Some(&chash) {
            return Err(format!(
                "{} was made for a different run (other seeds, behaviors or cells); use another --out, or run without --resume to start over",
                p.display()
            ));
        }
        if let Some(c) = saved["cells"].as_object() {
            done = c.clone();
        }
    }
    let t0 = std::time::Instant::now();
    let total = cell_list.len();
    let cfg = json!({"seeds": seeds, "tests": tests, "cells": cell_list.iter().map(|c| cell_key(&c.0, c.1)).collect::<Vec<_>>()});
    for (p, v) in &cell_list {
        let key = cell_key(p, *v);
        if done.contains_key(&key) {
            if !progress(done.len(), total, &format!("{key} (already done)")) {
                return Err("cancelled".into());
            }
            continue;
        }
        let nd = done.len();
        let cell = run_cell(weights, labels, counts, p, *v, &seeds, &tests, workers, |d, n, label| {
            progress(nd, total, &format!("{key}: {d}/{n} {label}"))
        })?;
        done.insert(key.clone(), cell);
        if let Some(path) = &path {
            if let Some(d) = path.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let body = json!({"config": cfg, "config_hash": chash, "cells": done});
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, serde_json::to_string_pretty(&body).unwrap_or_default()).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
        if !progress(done.len(), total, &format!("{key} done")) {
            return Err("cancelled".into());
        }
    }
    let ordered: Vec<Value> = cell_list.iter().map(|c| done[&cell_key(&c.0, c.1)].clone()).collect();
    let used: Vec<Value> = PARAMETERS
        .iter()
        .filter(|p| params.is_none_or(|ps| ps.contains(&p.id)))
        .map(|p| {
            let vals: Vec<f64> = values
                .and_then(|v| v.iter().find(|(k, _)| *k == p.id).map(|(_, v)| v.clone()))
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| p.values.to_vec());
            json!({"id": p.id, "label": p.label, "kind": p.kind, "default": p.default, "note": p.note, "values": vals})
        })
        .collect();
    Ok(json!({
        "kind": "sensitivity", "created": created, "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "seeds": seeds, "tests": tests, "underpowered": seeds.len() < UNDERPOWERED_BELOW,
        "criteria": {"ratio_min": 1.5, "p_max": 0.01, "pi_min": 0.5, "control_pi_max": 0.25},
        "parameters": used, "cells": ordered, "config_hash": chash,
        "tags": {"pathways": "CONNECTOME", "ranges_and_criteria": "GAME RULE", "every_cell": "MODEL PREDICTION"},
    }))
}

// --- reading a result --------------------------------------------------------------------------------------------------
/// `heatmap_cells`: `(behavior columns, [(row label, {behavior: result})])`.
pub fn heatmap_cells(res: &Value) -> (Vec<String>, Vec<(String, Value)>) {
    let cols: Vec<String> = res["tests"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(String::from)).collect();
    let grid = res["cells"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| {
            let label = if c["parameter"] == BASELINE {
                "baseline (defaults)".to_string()
            } else {
                let p = c["parameter"].as_str().unwrap_or("");
                let l = PARAMETERS.iter().find(|x| x.id == p).map_or(p, |x| x.label);
                format!("{l} = {}", fmt_g(c["value"].as_f64().unwrap_or(f64::NAN)))
            };
            (label, c["behaviors"].clone())
        })
        .collect();
    (cols, grid)
}

// --- export ------------------------------------------------------------------------------------------------------------------
pub fn baseline_of(res: &Value) -> Value {
    res["cells"].as_array().and_then(|c| c.iter().find(|c| c["parameter"] == BASELINE)).map_or(Value::Null, |c| c["behaviors"].clone())
}

fn csv_cell(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::String(s) => {
            if s.contains([',', '"', '\n']) {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.clone()
            }
        }
        Value::Number(n) if n.is_f64() => crate::cards::py_float(n.as_f64().unwrap_or(f64::NAN)),
        o => o.to_string(),
    }
}

fn write_csv(path: &Path, header: &[&str], rows: &[Vec<Value>]) -> std::io::Result<()> {
    let mut text = header.join(",") + "\r\n";
    for r in rows {
        text.push_str(&r.iter().map(csv_cell).collect::<Vec<_>>().join(","));
        text.push_str("\r\n");
    }
    std::fs::write(path, text)
}

/// The long table: one row per (parameter, value, behavior).
pub fn rows(res: &Value) -> Vec<Vec<Value>> {
    let base = baseline_of(res);
    let mut out = Vec::new();
    for c in res["cells"].as_array().into_iter().flatten() {
        for (b, r) in c["behaviors"].as_object().into_iter().flatten() {
            out.push(vec![
                c["parameter"].clone(),
                if c["value"].is_null() { json!("") } else { c["value"].clone() },
                json!(c["parameter"] == BASELINE),
                json!(b),
                r["passed"].clone(),
                r["metric"].clone(),
                r["effect"].clone(),
                r["control"].clone(),
                r["effect_size_dz"].clone(),
                r["p_value"].clone(),
                r["n"].clone(),
                match base.get(b) {
                    Some(bb) => json!(r["effect"].as_f64().unwrap_or(f64::NAN) - bb["effect"].as_f64().unwrap_or(f64::NAN)),
                    None => json!(""),
                },
            ]);
        }
    }
    out
}

/// Per parameter and behavior: how many tested values still pass and the first value away from the default at which it stops.
pub fn robustness_table(res: &Value) -> Vec<Vec<Value>> {
    let mut out = Vec::new();
    for p in res["parameters"].as_array().into_iter().flatten() {
        let id = p["id"].as_str().unwrap_or("");
        let default = p["default"].as_f64().unwrap_or(0.0);
        let cs: Vec<&Value> = res["cells"].as_array().into_iter().flatten().filter(|c| c["parameter"] == id).collect();
        for b in res["tests"].as_array().into_iter().flatten().filter_map(|t| t.as_str()) {
            let val = |c: &&Value| c["value"].as_f64().unwrap_or(f64::NAN);
            let mut below: Vec<&Value> = cs.iter().copied().filter(|c| val(c) < default).collect();
            below.sort_by(|a, b| val(b).total_cmp(&val(a)));
            let mut above: Vec<&Value> = cs.iter().copied().filter(|c| val(c) > default).collect();
            above.sort_by(|a, b| val(a).total_cmp(&val(b)));
            let first_fail = |seq: &[&Value]| {
                seq.iter().find(|c| c["behaviors"].get(b).is_some_and(|r| r["passed"] == false)).map_or(Value::Null, |c| c["value"].clone())
            };
            let n_pass = cs.iter().filter(|c| c["behaviors"][b]["passed"] == true).count();
            out.push(vec![json!(id), json!(b), json!(cs.len()), json!(n_pass), first_fail(&below), first_fail(&above)]);
        }
    }
    out
}

pub const PALETTES: [(&str, [[u8; 3]; 5]); 3] = [
    ("default", [[46, 125, 50], [183, 28, 28], [255, 255, 255], [255, 255, 255], [60, 60, 60]]),
    ("blue-yellow", [[30, 100, 200], [240, 190, 20], [255, 255, 255], [20, 20, 20], [90, 90, 90]]),
    ("high-contrast", [[255, 255, 255], [0, 0, 0], [0, 0, 0], [255, 255, 255], [128, 128, 128]]),
];

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;")
}

/// `heatmap_svg` (palette "default").
pub fn heatmap_svg(res: &Value) -> String {
    let [ok, bad, ok_t, bad_t, _] = PALETTES[0].1;
    let (cols, grid) = heatmap_cells(res);
    let (cw, rh, lw, top) = (74, 22, 220, 150);
    let (w, h) = (lw + cw * cols.len() as i32 + 20, top + rh * grid.len() as i32 + 60);
    let seeds = res["seeds"].as_array().cloned().unwrap_or_default();
    let (s0, s1) = (seeds.first().cloned().unwrap_or(Value::Null), seeds.last().cloned().unwrap_or(Value::Null));
    let mut out = vec![
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" font-family=\"sans-serif\" font-size=\"11\" role=\"img\" aria-label=\"Sensitivity heatmap: parameter value by validated behavior, pass or fail with effect size\">"
        ),
        format!("<rect width=\"{w}\" height=\"{h}\" fill=\"#ffffff\"/>"),
        "<text x=\"8\" y=\"18\" font-size=\"14\" font-weight=\"bold\">Sensitivity: parameter value x validated behavior</text>".into(),
        format!(
            "<text x=\"8\" y=\"36\">each cell: PASS/FAIL (validation's own criteria) and the effect (drive ratio, or the T-maze PI); seeds {s0}-{s1}, n = {}</text>",
            seeds.len()
        ),
        "<text x=\"8\" y=\"52\">MODEL PREDICTION. Analysis only: the defaults are not changed by this.</text>".into(),
    ];
    if res["underpowered"].as_bool().unwrap_or(false) {
        out.push(format!(
            "<text x=\"8\" y=\"68\" fill=\"#b71c1c\">UNDERPOWERED: fewer than {UNDERPOWERED_BELOW} seeds cannot reach p &lt; 0.01</text>"
        ));
    }
    for (j, b) in cols.iter().enumerate() {
        let x = f64::from(lw) + j as f64 * f64::from(cw) + f64::from(cw) / 2.0;
        out.push(format!(
            "<text transform=\"translate({},{}) rotate(-50)\" text-anchor=\"start\">{b}</text>",
            crate::cards::py_float(x),
            top - 6
        ));
    }
    for (i, (label, by)) in grid.iter().enumerate() {
        let y = top + i as i32 * rh;
        out.push(format!("<text x=\"8\" y=\"{}\">{}</text>", y + 15, esc(label)));
        for (j, b) in cols.iter().enumerate() {
            let x = lw + j as i32 * cw;
            let Some(r) = by.get(b) else {
                out.push(format!("<rect x=\"{x}\" y=\"{y}\" width=\"{}\" height=\"{}\" fill=\"#dddddd\"/>", cw - 2, rh - 2));
                continue;
            };
            let passed = r["passed"].as_bool().unwrap_or(false);
            let (fill, txt) = if passed { (ok, ok_t) } else { (bad, bad_t) };
            out.push(format!(
                "<rect x=\"{x}\" y=\"{y}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"#222\" stroke-width=\"0.5\"/>",
                cw - 2,
                rh - 2,
                hex(fill)
            ));
            out.push(format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"{}\">{} {:.2}</text>",
                crate::cards::py_float(f64::from(x) + f64::from(cw) / 2.0 - 1.0),
                y + 15,
                hex(txt),
                if passed { "PASS" } else { "FAIL" },
                r["effect"].as_f64().unwrap_or(f64::NAN)
            ));
        }
    }
    out.push("</svg>".into());
    out.join("\n")
}

/// `save`: `sensitivity.json`, `sensitivity.csv`, `sensitivity_matrix.csv`, `sensitivity_robustness.csv`, `sensitivity_heatmap.svg`.
pub fn save(res: &Value, folder: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(folder)?;
    let mut paths = Vec::new();
    let mut put = |name: &str| -> std::path::PathBuf {
        let p = folder.join(name);
        paths.push(p.clone());
        p
    };
    std::fs::write(put("sensitivity.json"), serde_json::to_string_pretty(res).unwrap_or_default())?;
    write_csv(
        &put("sensitivity.csv"),
        &[
            "parameter",
            "value",
            "is_default",
            "behavior",
            "passed",
            "metric",
            "effect",
            "control",
            "effect_size_dz",
            "p_value",
            "n_seeds",
            "effect_vs_baseline",
        ],
        &rows(res),
    )?;
    let (cols, grid) = heatmap_cells(res);
    let mut header: Vec<String> = vec!["parameter_value".into()];
    header.extend(cols.iter().cloned());
    let matrix: Vec<Vec<Value>> = grid
        .iter()
        .map(|(label, by)| {
            let mut r = vec![json!(label)];
            for b in &cols {
                r.push(match by.get(b) {
                    Some(x) => json!(format!(
                        "{} {:.3} (dz {:.2})",
                        if x["passed"].as_bool().unwrap_or(false) { "PASS" } else { "FAIL" },
                        x["effect"].as_f64().unwrap_or(f64::NAN),
                        x["effect_size_dz"].as_f64().unwrap_or(f64::NAN)
                    )),
                    None => json!(""),
                });
            }
            r
        })
        .collect();
    let h: Vec<&str> = header.iter().map(String::as_str).collect();
    write_csv(&put("sensitivity_matrix.csv"), &h, &matrix)?;
    write_csv(
        &put("sensitivity_robustness.csv"),
        &["parameter", "behavior", "values_tested", "values_passing", "first_fail_below", "first_fail_above"],
        &robustness_table(res),
    )?;
    std::fs::write(put("sensitivity_heatmap.svg"), heatmap_svg(res))?;
    Ok(paths)
}
