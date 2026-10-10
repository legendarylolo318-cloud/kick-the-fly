//! Critical path finder (`lab/criticalpath.py`): silence one cell type at a time and rank the types by how much a behavior changes,
//! against an unperturbed fly of the same seed.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::assay_jobs;
use crate::cards::{Build, LIF_DEFAULTS};
use crate::csr::Weights;
use crate::pack::Labels;
use crate::stats;
use crate::validation::{self, PRE, STIM, STIM_AMP, np_sum};

pub const DEFAULT_TOP: usize = 25;
pub const ASSAYS: [&str; 3] = ["tmaze", "looming", "sugar"];

fn headline_label(kind: &str) -> &'static str {
    match kind {
        "tmaze" => "performance index",
        "looming" => "escape probability (all speeds)",
        _ => "MN9 ratio (all doses)",
    }
}

fn headline(kind: &str, fly: &Value) -> f64 {
    let mean = |v: &[f64]| np_sum(v) / v.len().max(1) as f64;
    match kind {
        "tmaze" => fly["pi"].as_f64().unwrap_or(f64::NAN),
        "looming" => {
            let per: Vec<f64> = fly["trials"]
                .as_object()
                .into_iter()
                .flat_map(|o| o.values())
                .map(|tr| {
                    mean(
                        &tr.as_array()
                            .into_iter()
                            .flatten()
                            .map(|t| if t["escaped"].as_bool().unwrap_or(false) { 1.0 } else { t["escaped"].as_f64().unwrap_or(0.0) })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            mean(&per)
        }
        _ => {
            let per: Vec<f64> = fly["offers"]
                .as_object()
                .into_iter()
                .flat_map(|o| o.values())
                .map(|offs| {
                    mean(&offs.as_array().into_iter().flatten().map(|o| o["ratio"].as_f64().unwrap_or(f64::NAN)).collect::<Vec<_>>())
                })
                .collect();
            mean(&per)
        }
    }
}

/// Cell types ranked by how much of the readout's input they supply, directly or one hop further back (`shortlist`).
pub fn shortlist(weights: &Weights, labels: &Labels, test_id: &str, top: usize) -> Vec<Value> {
    let g = validation::groups(labels, weights);
    let t = validation::tests().iter().find(|t| t["id"] == test_id).cloned().unwrap_or(Value::Null);
    let readout = g.get(t["readout"].as_str().unwrap_or("")).cloned().unwrap_or_default();
    let drive = g.get(t["drive"].as_str().unwrap_or("")).cloned().unwrap_or_default();
    let n = weights.n();
    let csr = &weights.csr;
    // |W[readout, :]|.sum(axis=0) in float32, rows in readout order; then |W|.T @ direct (columns accumulate over rows ascending)
    let mut direct = vec![0.0f32; n];
    for &r in &readout {
        let (cols, vals) = csr.row(r as usize);
        for (&c, &v) in cols.iter().zip(vals) {
            direct[c as usize] += v.abs() * 1.0;
        }
    }
    let mut indirect = vec![0.0f32; n];
    if !readout.is_empty() {
        for (j, &x) in direct.iter().enumerate() {
            let (cols, vals) = csr.row(j);
            for (&c, &v) in cols.iter().zip(vals) {
                indirect[c as usize] += v.abs() * x;
            }
        }
    }
    let mut order: Vec<String> = Vec::new();
    let mut acc: HashMap<&str, (f64, usize)> = HashMap::new();
    for (i, ty) in labels.types.iter().enumerate() {
        if ty.is_empty() {
            continue;
        }
        let s = f64::from(direct[i] + indirect[i]);
        let e = acc.entry(ty.as_str()).or_insert_with(|| {
            order.push(ty.clone());
            (0.0, 0)
        });
        e.0 += s;
        e.1 += 1;
    }
    let drive_types: std::collections::BTreeSet<&str> =
        drive.iter().map(|&r| labels.types[r as usize].as_str()).filter(|t| !t.is_empty()).collect();
    let mut ranked: Vec<&String> = order.iter().collect();
    ranked.sort_by(|a, b| acc[b.as_str()].0.total_cmp(&acc[a.as_str()].0)); // stable: ties keep first-seen order
    let mut picked: Vec<Value> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = Default::default();
    for name in ranked {
        let (s, cnt) = acc[name.as_str()];
        if cnt < 1 || s <= 0.0 {
            continue;
        }
        picked.push(json!({"type": name, "neurons": cnt, "influence": s, "drives_the_behavior": drive_types.contains(name.as_str())}));
        seen.insert(name);
        if picked.len() >= top {
            break;
        }
    }
    for name in drive_types.iter().filter(|t| !seen.contains(*t)) {
        let n = labels.types.iter().filter(|t| t.as_str() == *name).count();
        picked.push(json!({"type": name, "neurons": n, "influence": acc.get(name).map_or(0.0, |e| e.0), "drives_the_behavior": true}));
    }
    picked
}

fn behavior_seed(weights: &Arc<Weights>, labels: &Arc<Labels>, seed: u64, test_id: &str, lesion: Option<&str>) -> Value {
    let b = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = b.brain_with(seed, None, 600, true);
    let g = validation::groups(labels, weights);
    if let Some(l) = lesion {
        assay_jobs::apply_surgery(&mut br, &g, &[(l.to_string(), -1)]);
    }
    let t = validation::tests().iter().find(|t| t["id"] == test_id).cloned().unwrap_or(Value::Null);
    let drive = g.get(t["drive"].as_str().unwrap_or("")).cloned().unwrap_or_default();
    let readout = g.get(t["readout"].as_str().unwrap_or("")).cloned().unwrap_or_default();
    let r = validation::pathway_response(&mut br, &[(drive, STIM_AMP)], &[&readout], PRE, STIM);
    let (base, driven) = (r[0].base_hz, r[0].driven_hz);
    json!({"score": driven, "base_hz": base, "driven_hz": driven, "ratio": driven / base.max(0.5)})
}

fn assay_seed(weights: &Arc<Weights>, labels: &Arc<Labels>, seed: u64, kind: &str, lesion: Option<&str>) -> Result<Value, String> {
    let surgery: Vec<(String, i8)> = lesion.map(|l| vec![(l.to_string(), -1)]).unwrap_or_default();
    let res = assay_jobs::assay_task(weights, labels, kind, seed, &Value::Null, &surgery)?;
    Ok(json!({"score": headline(kind, &res)}))
}

fn resume_path(p: Option<&Path>) -> Option<std::path::PathBuf> {
    p.map(|p| if p.extension().is_some_and(|e| e == "json") { p.to_path_buf() } else { p.join("critical_path_progress.json") })
}

fn load_resume(p: Option<&Path>, target: &str, seeds: &[u64]) -> Value {
    let Some(p) = resume_path(p) else { return json!({}) };
    let Ok(text) = std::fs::read_to_string(p) else { return json!({}) };
    let Ok(v) = serde_json::from_str::<Value>(&text) else { return json!({}) };
    let same_seeds = v["seeds"].as_array().is_some_and(|a| a.iter().filter_map(Value::as_u64).eq(seeds.iter().copied()));
    if v["target"] == target && same_seeds { v } else { json!({}) }
}

fn save_resume(p: Option<&Path>, target: &str, seeds: &[u64], cands: &[Value], state: &Value) {
    let Some(p) = resume_path(p) else { return };
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let body = json!({"target": target, "seeds": seeds, "candidates": cands, "control": state["control"], "results": state["results"]});
    let tmp = p.with_extension("tmp");
    if std::fs::write(&tmp, serde_json::to_string_pretty(&body).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(tmp, p);
    }
}

fn score(cand: &Value, vals: &[Value], base: &[Value]) -> Value {
    let col = |xs: &[Value], k: &str| -> Vec<f64> { xs.iter().map(|x| x[k].as_f64().unwrap_or(f64::NAN)).collect() };
    let v = col(vals, "score");
    let b_ = col(base, "score");
    let ci = stats::mean_ci(&v);
    let diff = assay_jobs::paired(&v, &b_);
    let b = np_sum(&b_) / b_.len() as f64;
    let mut out = json!({
        "type": cand["type"], "neurons": cand["neurons"].as_u64().unwrap_or(0),
        "influence": cand["influence"].as_f64().map_or(Value::Null, |f| json!(f)),
        "drives_the_behavior": cand["drives_the_behavior"].as_bool().unwrap_or(false),
        "lesioned_mean": ci.mean, "lesioned_ci": [ci.lo, ci.hi], "control_mean": b,
        "effect": ci.mean - b, "effect_share": if b != 0.0 { (ci.mean - b) / b } else { f64::NAN },
        "p_value": diff["p_value"].as_f64().map_or(Value::Null, |f| json!(f)), "n": ci.n, "per_seed": v,
    });
    for key in ["base_hz", "driven_hz", "ratio"] {
        if vals[0].get(key).is_some() {
            let m = |xs: &[Value]| {
                let c = col(xs, key);
                np_sum(&c) / c.len() as f64
            };
            out[format!("lesioned_{key}")] = json!(m(vals));
            out[format!("control_{key}")] = json!(m(base));
        }
    }
    out
}

/// Run one batch of `(seed, lesion)` jobs on `workers` threads; results in job order. `tick` returns false to stop.
fn run_many(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    target: &str,
    is_assay: bool,
    jobs: &[(u64, Option<String>)],
    workers: usize,
    tick: &mut dyn FnMut() -> bool,
) -> Result<Vec<Value>, String> {
    let next = AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<Value, String>)>();
    let mut out: Vec<Option<Value>> = vec![None; jobs.len()];
    let mut failure = None;
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1).min(jobs.len().max(1)) {
            let tx = tx.clone();
            let (next, stop) = (&next, &stop);
            scope.spawn(move || {
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some((seed, lesion)) = jobs.get(i) else { return };
                    let r = if is_assay {
                        assay_seed(weights, labels, *seed, target, lesion.as_deref())
                    } else {
                        Ok(behavior_seed(weights, labels, *seed, target, lesion.as_deref()))
                    };
                    if tx.send((i, r)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        while let Ok((i, r)) = rx.recv() {
            match r {
                Ok(v) => out[i] = Some(v),
                Err(e) => {
                    failure = Some(e);
                    stop.store(true, Ordering::Relaxed);
                }
            }
            if !tick() {
                failure.get_or_insert_with(|| "cancelled".to_string());
                stop.store(true, Ordering::Relaxed);
            }
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    Ok(out.into_iter().map(Option::unwrap_or_default_value).collect())
}

trait OrNull {
    fn unwrap_or_default_value(self) -> Value;
}
impl OrNull for Option<Value> {
    fn unwrap_or_default_value(self) -> Value {
        self.unwrap_or(Value::Null)
    }
}

/// `run`: silence each candidate type in turn and rank them. `progress(done, total, label)` returns false to cancel.
#[allow(clippy::too_many_arguments)]
pub fn run(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    target: &str,
    seeds: Option<&[u64]>,
    top: usize,
    workers: usize,
    resume: Option<&Path>,
    types: Option<&[String]>,
    created: &str,
    progress: &mut dyn FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let is_assay = ASSAYS.contains(&target);
    let seeds: Vec<u64> = seeds.map_or_else(validation::seeds, <[u64]>::to_vec);
    let t0 = std::time::Instant::now();
    let mut state = load_resume(resume, target, &seeds);
    let cands: Vec<Value> = if let Some(ts) = types {
        ts.iter().map(|t| json!({"type": t, "neurons": 0, "influence": null, "drives_the_behavior": false})).collect()
    } else if let Some(c) = state["candidates"].as_array().filter(|c| !c.is_empty()) {
        c.clone()
    } else {
        shortlist(weights, labels, if is_assay { "looming_escape" } else { target }, top)
    };
    let done_types: Vec<String> = state["results"].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    let todo: Vec<&Value> = cands.iter().filter(|c| !done_types.contains(&c["type"].as_str().unwrap_or("").to_string())).collect();
    let has_control = state["control"].is_object();
    let total = seeds.len() * (todo.len() + usize::from(!has_control));
    let mut done = 0usize;
    if !has_control {
        let jobs: Vec<(u64, Option<String>)> = seeds.iter().map(|&s| (s, None)).collect();
        let got = run_many(weights, labels, target, is_assay, &jobs, workers, &mut || {
            done += 1;
            progress(done, total, "unperturbed control")
        })?;
        let mut m = serde_json::Map::new();
        for (s, v) in seeds.iter().zip(got) {
            m.insert(s.to_string(), v);
        }
        state["control"] = Value::Object(m);
        save_resume(resume, target, &seeds, &cands, &state);
    }
    let mut results: serde_json::Map<String, Value> = state["results"].as_object().cloned().unwrap_or_default();
    let base: Vec<Value> = seeds.iter().map(|s| state["control"][s.to_string()].clone()).collect();
    for c in todo {
        let name = c["type"].as_str().unwrap_or("").to_string();
        let spec = format!("type:{name}");
        let jobs: Vec<(u64, Option<String>)> = seeds.iter().map(|&s| (s, Some(spec.clone()))).collect();
        let got = run_many(weights, labels, target, is_assay, &jobs, workers, &mut || {
            done += 1;
            progress(done, total, &format!("silencing {name}"))
        })?;
        results.insert(name, score(c, &got, &base));
        state["results"] = Value::Object(results.clone());
        save_resume(resume, target, &seeds, &cands, &state);
    }
    let mut ranked: Vec<Value> = results.values().cloned().collect();
    ranked.sort_by(|a, b| {
        let k = |v: &Value| -v["effect_share"].as_f64().unwrap_or(f64::NAN).abs();
        k(a).total_cmp(&k(b))
    });
    for (i, r) in ranked.iter_mut().enumerate() {
        r["rank"] = json!(i + 1);
    }
    let bs: Vec<f64> = base.iter().map(|x| x["score"].as_f64().unwrap_or(f64::NAN)).collect();
    Ok(json!({
        "kind": "critical_path", "target": target, "is_assay": is_assay, "created": created,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "seeds": seeds, "top": top,
        "metric": if is_assay { headline_label(target) } else { "readout firing while driven (spikes/s)" },
        "control": {"mean": np_sum(&bs) / bs.len() as f64, "per_seed": bs}, "candidates": cands, "ranked": ranked,
    }))
}

/// `save`: `critical_path.json` and `critical_path.csv`.
pub fn save(res: &Value, folder: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(folder)?;
    std::fs::write(folder.join("critical_path.json"), serde_json::to_string_pretty(res).unwrap_or_default())?;
    let ranked = res["ranked"].as_array().cloned().unwrap_or_default();
    let extra: Vec<&str> = ["lesioned_base_hz", "control_base_hz", "lesioned_ratio", "control_ratio"]
        .into_iter()
        .filter(|k| ranked.first().is_some_and(|r| r.get(*k).is_some()))
        .collect();
    let mut text =
        "rank,cell_type,neurons,drives_the_behavior,lesioned_mean,ci_lo,ci_hi,control_mean,effect,effect_share,p_value,n_seeds".to_string();
    for k in &extra {
        text.push(',');
        text.push_str(k);
    }
    text.push_str("\r\n");
    let f = |v: &Value, d: usize| format!("{:.d$}", v.as_f64().unwrap_or(f64::NAN)).replace("NaN", "nan");
    for r in &ranked {
        let mut cols = vec![
            r["rank"].to_string(),
            r["type"].as_str().unwrap_or("").to_string(),
            r["neurons"].to_string(),
            i32::from(r["drives_the_behavior"].as_bool().unwrap_or(false)).to_string(),
            f(&r["lesioned_mean"], 4),
            f(&r["lesioned_ci"][0], 4),
            f(&r["lesioned_ci"][1], 4),
            f(&r["control_mean"], 4),
            f(&r["effect"], 4),
            f(&r["effect_share"], 4),
            f(&r["p_value"], 5),
            r["n"].to_string(),
        ];
        cols.extend(extra.iter().map(|k| f(&r[*k], 4)));
        text.push_str(&cols.join(","));
        text.push_str("\r\n");
    }
    std::fs::write(folder.join("critical_path.csv"), text)
}
