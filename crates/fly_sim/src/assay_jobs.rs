//! The Lab's assays page engine (`lab/labjobs.py`, `lab/assays.py`): looming escape and sugar dose-response over several flies, the T-maze
//! (in `validation`), their summaries with 95 % intervals and the paired comparison with a same-seed control under a surgery.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::Brain;
use crate::cards::{Build, LIF_DEFAULTS};
use crate::csr::Weights;
use crate::pack::Labels;
use crate::rng::Generator;
use crate::stats::{self, MeanCi};
use crate::validation::{self, TmazeResult, mean, rest};

pub const LOOM_SPEEDS: [f64; 5] = [0.7, 1.4, 3.0, 5.5, 9.0];
pub const LOOM_MIN: f64 = 1.5;
pub const LOOM_FULL: f64 = 8.0;
pub const T_ESCAPE: f64 = 4.0;
pub const PER_RATIO: f64 = 1.5;
pub const DOSES: [f64; 7] = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0];
const STEPS_PER_TICK: [usize; 3] = [4, 3, 3];

/// `assays.apply_surgery`: `{spec: -1 | +1}`, specs as in `rows_of` or an assay group name.
pub fn apply_surgery(br: &mut Brain, g: &std::collections::HashMap<&'static str, Vec<u32>>, surgery: &[(String, i8)]) {
    for (spec, mode) in surgery {
        let rows = match g.get(spec.as_str()) {
            Some(r) => r.clone(),
            None if spec.starts_with("type:")
                || spec.starts_with("prefix:")
                || spec.starts_with("superclass:")
                || spec.starts_with("rows:") =>
            {
                br.rows_of(spec).unwrap_or_default()
            }
            None => {
                // a detail group or a population of the brain surgery screen, by name
                let det = br.groups_ref().det_members(spec);
                if det.is_empty() {
                    let pop = crate::surgery::ENTRIES.iter().find_map(|(_, sp)| match sp {
                        crate::surgery::Spec::Pop(n) if n.contains(&spec.as_str()) => Some(*sp),
                        _ => None,
                    });
                    match pop {
                        Some(sp) => crate::surgery::rows(br.labels(), br.groups_ref(), sp),
                        None => Vec::new(),
                    }
                } else {
                    det
                }
            }
        };
        br.set_surgery(&rows, *mode);
    }
}

/// `lab.surgery_options()`: `(label, {spec: -1})` for every entry of the brain surgery screen that names cell types, groups or populations.
pub fn surgery_options() -> Vec<(String, Vec<String>)> {
    use crate::surgery::Spec;
    let mut out = vec![("none".to_string(), Vec::new())];
    for (label, spec) in crate::surgery::ENTRIES.iter() {
        let keys: Vec<String> = match spec {
            Spec::Type(n) => vec![format!("type:{}", n.join(","))],
            Spec::Prefix(n) => vec![format!("prefix:{}", n.join(","))],
            Spec::Group(n) | Spec::Pop(n) => n.iter().map(|x| (*x).to_string()).collect(),
            _ => continue,
        };
        out.push(((*label).to_string(), keys));
    }
    out
}

fn new_brain(build: &Build, seed: u64, memory: bool) -> Brain {
    build.brain_with(seed, None, 600, memory)
}

/// `assays.looming_fly`
pub fn looming_fly(
    build: &Build,
    g: &std::collections::HashMap<&'static str, Vec<u32>>,
    seed: u64,
    speeds: &[f64],
    approaches: usize,
    surgery: &[(String, i8)],
) -> Value {
    let (radius, start) = (0.28f64, 3.0f64);
    let contact = radius + 0.02;
    let mut br = new_brain(build, seed, true);
    apply_surgery(&mut br, g, surgery);
    rest(&mut br, 400);
    let mut out = serde_json::Map::new();
    for &v in speeds {
        let mut trials = Vec::new();
        for _ in 0..approaches {
            let (mut d, mut prev, mut tick, mut t) = (start, None::<f64>, 0usize, 0.0f64);
            let (mut escaped_at, mut peak) = (None::<f64>, 0.0f64);
            while d > contact {
                let theta = 2.0 * (radius / d).atan();
                if let Some(p) = prev {
                    let rate = (theta - p) * 60.0;
                    let strength = ((rate - LOOM_MIN) / LOOM_FULL).clamp(0.0, 1.0);
                    if strength > 0.0 {
                        br.poke_recruit64("loom", None, strength, Some(0.6 * strength));
                    }
                }
                prev = Some(theta);
                for _ in 0..STEPS_PER_TICK[tick % 3] {
                    br.step();
                }
                let lvl = f64::from(br.level("escape"));
                peak = peak.max(lvl);
                if escaped_at.is_none() && lvl > T_ESCAPE {
                    escaped_at = Some(t);
                    break;
                }
                tick += 1;
                t += 1.0 / 60.0;
                d -= v / 60.0;
            }
            trials.push(json!({
                "escaped": escaped_at.is_some(), "latency_s": escaped_at,
                "distance_m": escaped_at.map(|_| d), "time_to_contact_s": escaped_at.map(|_| (d - contact) / v), "peak_level": peak,
            }));
            rest(&mut br, 400);
        }
        out.insert(format!("{v}"), Value::Array(trials));
    }
    json!({"seed": seed, "radius": radius, "start": start, "speeds": speeds, "trials": out})
}

fn hz(spikes: usize, n: usize, steps: usize) -> f64 {
    spikes as f64 / n.max(1) as f64 / steps.max(1) as f64 / 0.005
}

/// `assays.offer_sugar`
fn offer_sugar(br: &mut Brain, mn9: &[u32], dose: f64) -> Value {
    let (steps, baseline) = (200usize, 200usize);
    let mut mask = vec![false; br.n()];
    for &r in mn9 {
        mask[r as usize] = true;
    }
    let count = |br: &mut Brain| br.step().iter().filter(|&&s| mask[s as usize]).count();
    let mut before = 0;
    for _ in 0..baseline {
        before += count(br);
    }
    let mut during = 0;
    for i in 0..steps {
        if dose > 0.0 && i % 3 == 0 {
            br.poke_recruit64("sweet", None, 0.5, Some(dose));
        }
        during += count(br);
    }
    let (b, d) = (hz(before, mn9.len(), baseline), hz(during, mn9.len(), steps));
    json!({"dose": dose, "mn9_before_hz": b, "mn9_hz": d, "ratio": d / b.max(1.0), "extended": d >= PER_RATIO * b.max(1.0)})
}

/// `assays.sugar_fly`
pub fn sugar_fly(
    build: &Build,
    g: &std::collections::HashMap<&'static str, Vec<u32>>,
    seed: u64,
    doses: &[f64],
    repeats: usize,
    surgery: &[(String, i8)],
) -> Value {
    let mut br = new_brain(build, seed, true);
    apply_surgery(&mut br, g, surgery);
    rest(&mut br, 200);
    let mut offers: std::collections::BTreeMap<String, Vec<Value>> = doses.iter().map(|d| (format!("{d}"), Vec::new())).collect();
    let mut rng = Generator::new(seed + 3);
    let mut order: Vec<f64> = doses.iter().flat_map(|&d| std::iter::repeat_n(d, repeats)).collect();
    rng.shuffle(&mut order);
    for d in order {
        let one = offer_sugar(&mut br, &g["mn9"], d);
        if let Some(v) = offers.get_mut(&format!("{d}")) {
            v.push(one);
        }
        rest(&mut br, 200);
    }
    json!({"seed": seed, "doses": doses, "offers": offers})
}

fn ci_json(c: MeanCi) -> Value {
    let f = |x: f64| if x.is_nan() { Value::Null } else { json!(x) };
    json!({"mean": f(c.mean), "sd": f(c.sd), "n": c.n, "lo": f(c.lo), "hi": f(c.hi)})
}

fn fvals(v: &[Value], key: &str) -> Vec<f64> {
    v.iter().map(|x| x[key].as_f64().unwrap_or(f64::NAN)).collect()
}

/// `labjobs.summarize` for `tmaze`, `looming` and `sugar`.
pub fn summarize(kind: &str, flies: &[Value]) -> Value {
    match kind {
        "tmaze" => {
            let col = |k: &str| -> Value { ci_json(stats::mean_ci(&fvals(flies, k))) };
            json!({
                "metric": "performance index (Tully & Quinn)", "pi": col("pi"), "fear_cs_plus": col("fear_plus"), "fear_cs_minus": col("fear_minus"),
                "mbon_cs_plus_hz": col("mbon_plus_hz"), "mbon_cs_minus_hz": col("mbon_minus_hz"), "per_fly": fvals(flies, "pi"),
            })
        }
        "looming" => {
            let speeds: Vec<f64> = flies[0]["speeds"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            let mut rows = Vec::new();
            for v in &speeds {
                let key = format!("{v}");
                let trials = |f: &Value| f["trials"][&key].as_array().cloned().unwrap_or_default();
                let probs: Vec<f64> = flies
                    .iter()
                    .map(|f| {
                        mean(&trials(f).iter().map(|t| f64::from(u8::from(t["escaped"].as_bool().unwrap_or(false)))).collect::<Vec<_>>())
                    })
                    .collect();
                let esc: Vec<Value> =
                    flies.iter().flat_map(|f| trials(f).into_iter().filter(|t| t["escaped"].as_bool().unwrap_or(false))).collect();
                let lat: Vec<f64> = esc.iter().map(|t| t["latency_s"].as_f64().unwrap_or(f64::NAN)).collect();
                let dist: Vec<f64> = esc.iter().map(|t| t["distance_m"].as_f64().unwrap_or(f64::NAN)).collect();
                let n: usize = flies.iter().map(|f| trials(f).len()).sum();
                rows.push(json!({
                    "speed": v, "escape_probability": ci_json(stats::mean_ci(&probs)), "escapes": esc.len(), "approaches": n,
                    "latency_s": ci_json(stats::mean_ci(&lat)), "distance_m": ci_json(stats::mean_ci(&dist)),
                }));
            }
            let per_fly: Vec<f64> = flies
                .iter()
                .map(|f| {
                    let per: Vec<f64> = f["trials"]
                        .as_object()
                        .map(|m| {
                            m.values()
                                .map(|tr| {
                                    let tr = tr.as_array().cloned().unwrap_or_default();
                                    mean(
                                        &tr.iter()
                                            .map(|t| f64::from(u8::from(t["escaped"].as_bool().unwrap_or(false))))
                                            .collect::<Vec<_>>(),
                                    )
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    mean(&per)
                })
                .collect();
            json!({"metric": "escape probability and latency vs approach speed", "rows": rows, "per_fly": per_fly})
        }
        "sugar" => {
            let doses: Vec<f64> = flies[0]["doses"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            let mut rows = Vec::new();
            for d in &doses {
                let key = format!("{d}");
                let offs = |f: &Value| f["offers"][&key].as_array().cloned().unwrap_or_default();
                let col = |field: &str| -> Vec<f64> { flies.iter().map(|f| mean(&fvals(&offs(f), field))).collect() };
                let ext: Vec<f64> = flies
                    .iter()
                    .map(|f| {
                        mean(&offs(f).iter().map(|o| f64::from(u8::from(o["extended"].as_bool().unwrap_or(false)))).collect::<Vec<_>>())
                    })
                    .collect();
                let yes: usize = flies.iter().map(|f| offs(f).iter().filter(|o| o["extended"].as_bool().unwrap_or(false)).count()).sum();
                let n: usize = flies.iter().map(|f| offs(f).len()).sum();
                rows.push(json!({
                    "dose": d, "mn9_ratio": ci_json(stats::mean_ci(&col("ratio"))), "mn9_hz": ci_json(stats::mean_ci(&col("mn9_hz"))),
                    "extension_probability": ci_json(stats::mean_ci(&ext)), "extensions": yes, "offers": n,
                }));
            }
            let per_fly: Vec<f64> = flies
                .iter()
                .map(|f| {
                    let per: Vec<f64> = f["offers"]
                        .as_object()
                        .map(|m| m.values().map(|offs| mean(&fvals(offs.as_array().map_or(&[][..], Vec::as_slice), "ratio"))).collect())
                        .unwrap_or_default();
                    mean(&per)
                })
                .collect();
            json!({"metric": "MN9 dose-response", "rows": rows, "per_fly": per_fly})
        }
        _ => Value::Null,
    }
}

/// `labstats.paired`
pub fn paired(treated: &[f64], control: &[f64]) -> Value {
    let pairs: Vec<(f64, f64)> = treated.iter().zip(control).filter(|(a, b)| !a.is_nan() && !b.is_nan()).map(|(a, b)| (*a, *b)).collect();
    let (a, b): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let n = d.len();
    let mut out = json!({
        "n": n, "mean_difference": if n > 0 { json!(mean(&d)) } else { Value::Null }, "test": "Wilcoxon signed-rank (paired, two-sided)",
    });
    if n < 2 || d.iter().all(|x| x.abs() <= 1e-8) {
        out["p_value"] = if n > 0 { json!(1.0) } else { Value::Null };
        out["t_p_value"] = Value::Null;
        return out;
    }
    out["p_value"] = json!(stats::wilcoxon_two_sided(&d));
    let t = stats::ttest_rel_p(&a, &b);
    out["t_p_value"] = if t.is_finite() { json!(t) } else { Value::Null };
    out["difference_ci"] = ci_json(stats::mean_ci(&d));
    out
}

/// `labjobs.compare`
pub fn compare(kind: &str, treated: &Value, control: &Value) -> Value {
    let f = |v: &Value| -> Vec<f64> { v["per_fly"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default() };
    let mut out = json!({"overall": paired(&f(treated), &f(control))});
    if kind == "looming" || kind == "sugar" {
        let (tk, ck, lk) = if kind == "looming" { ("escapes", "approaches", "speed") } else { ("extensions", "offers", "dose") };
        let rows: Vec<Value> = treated["rows"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .zip(control["rows"].as_array().cloned().unwrap_or_default().iter())
            .map(|(rt, rc)| {
                let g = |r: &Value, k: &str| r[k].as_u64().unwrap_or(0);
                json!({"level": rt[lk], "fisher_p": stats::fisher_exact(g(rt, tk), g(rt, ck), g(rc, tk), g(rc, ck))})
            })
            .collect();
        out["per_level"] = Value::Array(rows);
    }
    out
}

/// One fly of an assay (`labjobs.assay_task`).
pub fn assay_task(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    kind: &str,
    seed: u64,
    options: &Value,
    surgery: &[(String, i8)],
) -> Result<Value, String> {
    let build = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let g = validation::groups(labels, weights);
    match kind {
        "tmaze" => {
            let cycles = options["cycles"].as_u64().unwrap_or(6) as usize;
            let trials = options["test_trials"].as_u64().unwrap_or(20) as usize;
            let run =
                |s: u64, cs: &str| -> TmazeResult { validation::tmaze_fly_with(&build, &g, s, cs, cycles, trials, true, 0.08, surgery) };
            let (a, b) = (run(seed, "odor_a"), run(seed + 50_000, "odor_b"));
            let avg = |x: f64, y: f64| (x + y) / 2.0;
            Ok(json!({
                "seed": seed, "pi": avg(a.pi, b.pi), "fear_plus": avg(a.fear_plus, b.fear_plus), "fear_minus": avg(a.fear_minus, b.fear_minus),
                "mbon_plus_hz": avg(a.mbon_plus_hz, b.mbon_plus_hz), "mbon_minus_hz": avg(a.mbon_minus_hz, b.mbon_minus_hz),
            }))
        }
        "looming" => {
            let speeds: Vec<f64> =
                options["speeds"].as_array().map_or_else(|| LOOM_SPEEDS.to_vec(), |a| a.iter().filter_map(Value::as_f64).collect());
            let approaches = options["approaches"].as_u64().unwrap_or(3) as usize;
            Ok(looming_fly(&build, &g, seed, &speeds, approaches, surgery))
        }
        "sugar" => {
            let doses: Vec<f64> =
                options["doses"].as_array().map_or_else(|| DOSES.to_vec(), |a| a.iter().filter_map(Value::as_f64).collect());
            let repeats = options["repeats"].as_u64().unwrap_or(2) as usize;
            Ok(sugar_fly(&build, &g, seed, &doses, repeats, surgery))
        }
        other => Err(format!("the {other} assay is not part of this build yet")),
    }
}

/// `labjobs.run_sync`: every seed with the surgery, and without it as the matched control; `progress(done, total)` returns false to cancel.
#[allow(clippy::too_many_arguments)]
pub fn run_sync(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    kind: &str,
    seeds: &[u64],
    options: &Value,
    surgery: &[(String, i8)],
    workers: usize,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<Value, String> {
    let mut tasks: Vec<(u64, bool)> = seeds.iter().map(|&s| (s, true)).collect();
    if !surgery.is_empty() {
        tasks.extend(seeds.iter().map(|&s| (s, false)));
    }
    let total = tasks.len();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let (tx, rx) = std::sync::mpsc::channel::<(u64, bool, Result<Value, String>)>();
    let mut out: std::collections::HashMap<(u64, bool), Value> = Default::default();
    let mut failure: Option<String> = None;
    let tasks_ref = &tasks;
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1).min(total.max(1)) {
            let tx = tx.clone();
            let (next, stop) = (&next, &stop);
            scope.spawn(move || {
                loop {
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(seed, treated)) = tasks_ref.get(i) else { return };
                    let s: &[(String, i8)] = if treated { surgery } else { &[] };
                    let r = assay_task(weights, labels, kind, seed, options, s);
                    if tx.send((seed, treated, r)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0;
        while let Ok((seed, treated, r)) = rx.recv() {
            done += 1;
            match r {
                Ok(v) => {
                    out.insert((seed, treated), v);
                }
                Err(e) => {
                    failure = Some(e);
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            if !progress(done, total) {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                failure = Some("cancelled".into());
            }
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    let get = |treated: bool| -> Vec<Value> { seeds.iter().map(|s| out[&(*s, treated)].clone()).collect() };
    let treated = summarize(kind, &get(true));
    let mut res = json!({"kind": kind, "seeds": seeds, "options": options, "treated": treated, "notes": []});
    if !surgery.is_empty() {
        let control = summarize(kind, &get(false));
        res["comparison"] = compare(kind, &res["treated"], &control);
        res["control"] = control;
        res["surgery"] = Value::Object(surgery.iter().map(|(k, m)| (k.clone(), json!(m))).collect());
    }
    Ok(res)
}
