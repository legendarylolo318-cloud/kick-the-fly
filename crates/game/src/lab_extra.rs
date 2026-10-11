//! The assay kinds that need the game's own engines (`lab/assays.py:orchard_fly`, `lab/predators.py`, `lab/audio.py`): orchard feeding,
//! predator escape and the hum demo. They plug into `fly_sim::assay_jobs` through [`register`].
//!
//! The predator and the orchard use this port's own generators (the Python ones are NumPy's), so those two give the same kind of result,
//! not the same numbers; the hum demo is deterministic and is compared with Python.

#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

use std::sync::Arc;

use fly_sim::cards::{Build, LIF_DEFAULTS};
use fly_sim::csr::Weights;
use fly_sim::pack::Labels;
use fly_sim::validation;
use serde_json::{Value, json};

pub const KINDS: [&str; 3] = ["orchard", "predator_escape", "hum_demo"];
const STEPS_PER_TICK: [usize; 3] = [4, 3, 3];

// --- the hum demo -------------------------------------------------------------------------------------------------------------------
pub const HUM_CONDITIONS: [(&str, Option<(f64, Option<f64>)>); 6] = [
    ("silence", None),
    ("steady_200", Some((200.0, None))),
    ("pulses_200_ipi35", Some((200.0, Some(35.0)))),
    ("pulses_200_ipi70", Some((200.0, Some(70.0)))),
    ("steady_50", Some((50.0, None))),
    ("steady_600", Some((600.0, None))),
];
const HUM_AMP: f64 = 0.1;
const HUM_SECONDS: f64 = 3.0;
const PRE_STEPS: usize = 600;
const RATE: f32 = 22050.0;

/// `mic.hum`: a sine at `hz`, steady or as a pulse train of 12 ms pulses every `ipi_ms`.
pub fn hum(hz: f64, seconds: f64, amp: f64, ipi_ms: Option<f64>) -> Vec<f32> {
    let n = (seconds * f64::from(RATE)).round_ties_even() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(RATE);
            let mut x = (2.0 * std::f64::consts::PI * hz * t).sin();
            if let Some(ipi) = ipi_ms.filter(|v| *v > 0.0) {
                let phase = (t * 1000.0).rem_euclid(ipi);
                let env = if phase < 12.0 { (std::f64::consts::PI * phase / 12.0).sin().powi(2) } else { 0.0 };
                x *= env;
            }
            (amp * x) as f32
        })
        .collect()
}

fn hz_of(count: usize, n: usize, steps: usize) -> f64 {
    count as f64 / n.max(1) as f64 / steps.max(1) as f64 / 0.005
}

/// `audio.demo_fly`.
pub fn hum_demo_fly(weights: &Arc<Weights>, labels: &Arc<Labels>, seed: u64, options: &Value) -> Result<Value, String> {
    let conditions: Vec<String> = options["conditions"]
        .as_array()
        .map(|a| a.iter().filter_map(|c| c.as_str().map(String::from)).collect())
        .unwrap_or_else(|| HUM_CONDITIONS.iter().map(|c| c.0.to_string()).collect());
    let seconds = options["seconds"].as_f64().unwrap_or(HUM_SECONDS);
    let amp = options["amp"].as_f64().unwrap_or(HUM_AMP);
    if let Some(bad) = conditions.iter().find(|c| !HUM_CONDITIONS.iter().any(|h| h.0 == c.as_str())) {
        return Err(format!("unknown condition '{bad}'; use one of {}", HUM_CONDITIONS.iter().map(|c| c.0).collect::<Vec<_>>().join(", ")));
    }
    let build = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = build.brain_with(seed, None, 600, true);
    let g = validation::groups(labels, weights);
    let t = &labels.types;
    let a_rows: Vec<u32> = (0..t.len() as u32).filter(|&i| t[i as usize].starts_with("JO-A")).collect();
    let b_rows: Vec<u32> = (0..t.len() as u32).filter(|&i| t[i as usize].starts_with("JO-B")).collect();
    let mut watch: Vec<(&str, Vec<u32>)> = ["p1", "pip10", "ps1"].iter().map(|k| (*k, g[*k].clone())).collect();
    watch.push(("jo_a", a_rows.clone()));
    watch.push(("jo_b", b_rows.clone()));
    let mask = |rows: &[u32]| -> Vec<bool> {
        let mut m = vec![false; br.n()];
        for &r in rows {
            m[r as usize] = true;
        }
        m
    };
    let masks: Vec<Vec<bool>> = watch.iter().map(|w| mask(&w.1)).collect();
    validation::rest(&mut br, 400);
    let mut out = serde_json::Map::new();
    for cond in &conditions {
        let mut base = vec![0usize; watch.len()];
        for _ in 0..PRE_STEPS {
            let sp = br.step();
            for (k, m) in masks.iter().enumerate() {
                base[k] += sp.iter().filter(|&&s| m[s as usize]).count();
            }
        }
        let spec = HUM_CONDITIONS.iter().find(|h| h.0 == cond.as_str()).and_then(|h| h.1);
        let wave = match spec {
            None => vec![0.0f32; (seconds * f64::from(RATE)) as usize],
            Some((hz, ipi)) => hum(hz, seconds, amp, ipi),
        };
        // hear(): analyse each 256-sample chunk, put its drive on JO-A/B and step the brain for that chunk's time
        let mut an = crate::mic_analysis::Analyzer::new(RATE, 1.0);
        let (mut counts, mut steps_done, mut carry) = (vec![0usize; watch.len()], 0usize, 0.0f64);
        let (mut peaks, mut da, mut db): (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
        let rows: Vec<u32> = a_rows.iter().chain(&b_rows).copied().collect();
        for ch in wave.as_chunks::<256>().0 {
            let r = an.push(ch);
            if !rows.is_empty() {
                let vals: Vec<f32> = a_rows.iter().map(|_| r.drive_a).chain(b_rows.iter().map(|_| r.drive_b)).collect();
                br.set_current_values("mic", &rows, &vals);
            }
            carry += 256.0 / f64::from(RATE) / 0.005;
            let n = carry as usize;
            carry -= n as f64;
            for _ in 0..n {
                let sp = br.step();
                for (k, m) in masks.iter().enumerate() {
                    counts[k] += sp.iter().filter(|&&s| m[s as usize]).count();
                }
            }
            steps_done += n;
            if r.peak_hz != 0.0 {
                peaks.push(f64::from(r.peak_hz));
            }
            da.push(f64::from(r.drive_a));
            db.push(f64::from(r.drive_b));
        }
        br.clear_current("mic");
        let mut row = serde_json::Map::new();
        for (k, (name, rr)) in watch.iter().enumerate() {
            let b_hz = hz_of(base[k], rr.len(), PRE_STEPS);
            let d_hz = hz_of(counts[k], rr.len(), steps_done.max(1));
            row.insert((*name).into(), json!({"calm_hz": b_hz, "hz": d_hz, "ratio": d_hz / b_hz.max(0.05)}));
        }
        let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { validation::np_sum(v) / v.len() as f64 };
        peaks.sort_by(f64::total_cmp);
        let med = if peaks.is_empty() {
            0.0
        } else if peaks.len() % 2 == 1 {
            peaks[peaks.len() / 2]
        } else {
            (peaks[peaks.len() / 2 - 1] + peaks[peaks.len() / 2]) / 2.0
        };
        row.insert("analysis".into(), json!({"peak_hz": med, "drive_a": mean(&da), "drive_b": mean(&db)}));
        out.insert(cond.clone(), Value::Object(row));
        validation::rest(&mut br, PRE_STEPS);
    }
    Ok(json!({"seed": seed, "seconds": seconds, "amp": amp, "conditions": out, "n_jo_a": a_rows.len(), "n_jo_b": b_rows.len()}))
}

fn ci(c: fly_sim::stats::MeanCi) -> Value {
    let f = |x: f64| if x.is_finite() { json!(x) } else { Value::Null };
    json!({"mean": f(c.mean), "sd": f(c.sd), "n": c.n, "lo": f(c.lo), "hi": f(c.hi)})
}

const HUM_CRITERIA: [&str; 3] = [
    "H1  the 200 Hz pulse train at a 35 ms interpulse interval drives the P1 cluster to >= 1.2x its calm rate in >= 8 of 10 flies",
    "H2  silence does not (P1 < 1.2x calm) in >= 9 of 10 flies",
    "H3  the ps1 song motor neurons stay below THRESH['song'] (1.8x calm) during that pulse train in >= 8 of 10 flies",
];

/// `audio.headline`: P1's firing ratio during the 200 Hz pulse train.
fn hum_headline(f: &Value) -> f64 {
    f["conditions"]["pulses_200_ipi35"]["p1"]["ratio"].as_f64().unwrap_or(f64::NAN)
}

/// `audio.summarize`.
pub fn hum_summarize(flies: &[Value]) -> Value {
    let conds: Vec<String> = flies[0]["conditions"].as_object().into_iter().flatten().map(|(k, _)| k.clone()).collect();
    // Python keeps the condition order the demo ran them in; serde sorts keys, so the table follows CONDITIONS
    let mut ordered: Vec<String> = HUM_CONDITIONS.iter().map(|c| c.0.to_string()).filter(|c| conds.contains(c)).collect();
    for c in conds {
        if !ordered.contains(&c) {
            ordered.push(c);
        }
    }
    let mut rows = Vec::new();
    for cond in &ordered {
        let mut r = serde_json::Map::new();
        r.insert("condition".into(), json!(cond));
        for k in ["p1", "pip10", "ps1", "jo_a", "jo_b"] {
            let ratios: Vec<f64> = flies.iter().map(|f| f["conditions"][cond][k]["ratio"].as_f64().unwrap_or(f64::NAN)).collect();
            r.insert(
                k.into(),
                json!({"ratio": ci(fly_sim::stats::mean_ci(&ratios)), "flies_over_1_2": ratios.iter().filter(|x| **x >= 1.2).count()}),
            );
        }
        let mut peaks: Vec<f64> = flies.iter().map(|f| f["conditions"][cond]["analysis"]["peak_hz"].as_f64().unwrap_or(0.0)).collect();
        peaks.sort_by(f64::total_cmp);
        let n = peaks.len();
        r.insert("peak_hz".into(), json!(if n % 2 == 1 { peaks[n / 2] } else { (peaks[n / 2 - 1] + peaks[n / 2]) / 2.0 }));
        rows.push(Value::Object(r));
    }
    let scored = ordered.iter().any(|c| c == "pulses_200_ipi35") && ordered.iter().any(|c| c == "silence");
    let verdict = if scored {
        let n = flies.len();
        let need = |k: usize| ((k * n) as f64 / 10.0).ceil() as usize;
        let song = f64::from(crate::reactions::thresholds().song);
        let h1n = flies.iter().filter(|f| f["conditions"]["pulses_200_ipi35"]["p1"]["ratio"].as_f64().unwrap_or(0.0) >= 1.2).count();
        let h2n = flies.iter().filter(|f| f["conditions"]["silence"]["p1"]["ratio"].as_f64().unwrap_or(0.0) < 1.2).count();
        let h3n = flies.iter().filter(|f| f["conditions"]["pulses_200_ipi35"]["ps1"]["ratio"].as_f64().unwrap_or(0.0) < song).count();
        json!({"H1": h1n >= need(8), "H2": h2n >= need(9), "H3": h3n >= need(8), "p1_flies_over": h1n, "silence_flies_under": h2n,
               "ps1_flies_under_song_threshold": h3n, "flies": n, "passed": h1n >= need(8) && h2n >= need(9) && h3n >= need(8)})
    } else {
        Value::Null
    };
    json!({"metric": "hum -> JO-A/B -> courtship pathway: firing ratio vs calm (MODEL PREDICTION)", "rows": rows, "n": flies.len(),
           "criteria": HUM_CRITERIA, "verdict": verdict, "per_fly": flies.iter().map(hum_headline).collect::<Vec<_>>()})
}

// --- predator escape ------------------------------------------------------------------------------------------------------------------
const WARM_FRAMES: usize = 120;
const SETTLE_STEPS: usize = 400;

/// `predators.loom_rates`: the game's loom measure for each frame of a trace.
pub fn loom_rates(tr: &crate::predators::Trace, flying: bool) -> Vec<f64> {
    let head =
        if flying { bevy::math::Vec3::new(0.0, 1.51, 0.0) } else { bevy::math::Vec3::new(0.0, crate::predators::THORAX_Y + 0.01, 0.0) };
    let mut prev: std::collections::HashMap<(&str, &str), f32> = Default::default();
    let mut out = Vec::new();
    for frame in &tr.frames {
        let (mut seen, mut best) = (std::collections::HashMap::new(), 0.0f32);
        for (key, pos, r) in frame {
            let d = pos.distance(head).max(r + 0.02);
            let theta = 2.0 * (r / d).atan();
            seen.insert(*key, theta);
            if let Some(pv) = prev.get(key) {
                best = best.max((theta - pv) * 60.0);
            }
        }
        prev = seen;
        out.push(f64::from(best));
    }
    out
}

/// `predators.escape_trial`: one attack against one brain.
fn escape_trial(br: &mut fly_sim::Brain, kind: &str, tr: &crate::predators::Trace) -> Value {
    let th = crate::reactions::thresholds();
    let rates = loom_rates(tr, kind == "dragonfly");
    let n = rates.len();
    let end = tr.capture_frame.unwrap_or(n.saturating_sub(1));
    let strike = tr.strike_frame.unwrap_or(end);
    let first = rates.iter().position(|r| *r > f64::from(th.loom_min)).unwrap_or(end);
    let start = first.saturating_sub(WARM_FRAMES);
    let stop = if tr.capture_frame.is_some() { end } else { end + 1 };
    let (mut escaped_at, mut before, mut peak, mut tick) = (None::<usize>, false, 0.0f64, 0usize);
    for i in start..stop.min(n) {
        let strength = ((rates[i] - f64::from(th.loom_min)) / f64::from(th.loom_full)).clamp(0.0, 1.0);
        if strength > 0.0 {
            br.poke_recruit64("loom", None, strength, Some(0.6 * strength));
        }
        for _ in 0..STEPS_PER_TICK[tick % 3] {
            br.step();
        }
        tick += 1;
        let lvl = f64::from(br.level("escape"));
        peak = peak.max(lvl);
        if escaped_at.is_none() && lvl > f64::from(th.escape) {
            escaped_at = Some(i);
            before = i < strike;
            break;
        }
    }
    let max_to = |a: usize, b: usize| rates.get(a..=b.min(n.saturating_sub(1))).map_or(0.0, |s| s.iter().copied().fold(0.0, f64::max));
    json!({"kind": kind, "escaped": escaped_at.is_some(), "captured": tr.capture_frame.is_some(),
           "latency_s": escaped_at.map(|e| (e as f64 - first as f64) / 60.0), "lead_s": escaped_at.map(|e| (end as f64 - e as f64) / 60.0),
           "noticed_before_strike": before, "peak_level": peak, "max_loom_before_strike": max_to(0, strike),
           "strike_loom_peak": max_to(strike, end), "frames": end + 1})
}

/// `predators.escape_fly`.
pub fn predator_escape_fly(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    seed: u64,
    options: &Value,
    surgery: &[(String, i8)],
) -> Result<Value, String> {
    let kinds: Vec<String> = options["kinds"]
        .as_array()
        .map(|a| a.iter().filter_map(|k| k.as_str().map(String::from)).collect())
        .unwrap_or_else(|| ["frog", "dragonfly", "mantis"].iter().map(|s| s.to_string()).collect());
    let trials = options["trials"].as_u64().unwrap_or(3) as usize;
    let build = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = build.brain_with(seed, None, 600, true);
    let g = validation::groups(labels, weights);
    fly_sim::assay_jobs::apply_surgery(&mut br, &g, surgery);
    validation::rest(&mut br, SETTLE_STEPS);
    let mut out = serde_json::Map::new();
    for kind in &kinds {
        let mut rows = Vec::new();
        for j in 0..trials {
            let tr = crate::predators::trace(kind, seed * 1000 + j as u64, 40.0)?;
            rows.push(escape_trial(&mut br, kind, &tr));
            validation::rest(&mut br, SETTLE_STEPS);
        }
        out.insert(kind.clone(), Value::Array(rows));
    }
    Ok(json!({"seed": seed, "kinds": kinds, "trials": out}))
}

/// `predators.wilson`.
fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let z = 1.96f64;
    let (nf, p) = (n as f64, k as f64 / n as f64);
    let d = 1.0 + z * z / nf;
    let c = (p + z * z / (2.0 * nf)) / d;
    let h = z * (p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt() / d;
    ((c - h).max(0.0), (c + h).min(1.0))
}

const PRED_CRITERIA: [&str; 2] = [
    "P1  the mantis's approach, before it strikes, never expands faster than LOOM_MIN (geometry, per trace)",
    "P2  in at least 9 of 10 flies (seeds 1000-1009) DNp01 stays below the escape threshold during the mantis's creep, until the strike begins",
];

/// `predators.summarize`.
pub fn predator_summarize(flies: &[Value]) -> Value {
    let kinds: Vec<String> = flies[0]["kinds"].as_array().into_iter().flatten().filter_map(|k| k.as_str().map(String::from)).collect();
    let mut rows = Vec::new();
    for kind in &kinds {
        let trials: Vec<&Value> = flies.iter().flat_map(|f| f["trials"][kind].as_array().into_iter().flatten()).collect();
        let k = trials.iter().filter(|t| t["escaped"].as_bool().unwrap_or(false)).count();
        let n = trials.len();
        let (lo, hi) = wilson(k, n);
        let per_fly: Vec<f64> = flies
            .iter()
            .map(|f| {
                let t = f["trials"][kind].as_array().cloned().unwrap_or_default();
                t.iter().filter(|x| x["escaped"].as_bool().unwrap_or(false)).count() as f64 / t.len().max(1) as f64
            })
            .collect();
        let lead: Vec<f64> =
            trials.iter().filter(|t| t["escaped"].as_bool().unwrap_or(false)).filter_map(|t| t["lead_s"].as_f64()).collect();
        rows.push(json!({"predator": kind, "escapes": k, "trials": n, "escape_probability": if n > 0 { json!(k as f64 / n as f64) } else { Value::Null },
                         "ci95": [lo, hi], "per_fly": ci(fly_sim::stats::mean_ci(&per_fly)),
                         "lead_s": if lead.is_empty() { Value::Null } else { ci(fly_sim::stats::mean_ci(&lead)) },
                         "noticed_before_strike": trials.iter().filter(|t| t["noticed_before_strike"].as_bool().unwrap_or(false)).count(),
                         "max_loom_before_strike": trials.iter().filter_map(|t| t["max_loom_before_strike"].as_f64()).fold(0.0, f64::max)}));
    }
    let mantis: Vec<&Value> = flies.iter().filter(|f| f["trials"].get("mantis").is_some()).collect();
    let verdict = if mantis.is_empty() {
        json!({"P1": null, "P2": null, "passed": null, "flies": flies.len()})
    } else {
        let loom_min = f64::from(crate::reactions::thresholds().loom_min);
        let p1 = mantis.iter().all(|f| {
            f["trials"]["mantis"].as_array().into_iter().flatten().all(|t| t["max_loom_before_strike"].as_f64().unwrap_or(0.0) <= loom_min)
        });
        let calm = mantis
            .iter()
            .filter(|f| {
                !f["trials"]["mantis"].as_array().into_iter().flatten().any(|t| t["noticed_before_strike"].as_bool().unwrap_or(false))
            })
            .count();
        let p2 = calm >= (0.9 * mantis.len() as f64).ceil() as usize;
        json!({"P1": p1, "P2": p2, "flies_that_did_not_notice_the_creep": calm, "flies": mantis.len(), "passed": p1 && p2})
    };
    let per_fly: Vec<f64> = flies
        .iter()
        .map(|f| {
            let ks: Vec<String> = f["kinds"].as_array().into_iter().flatten().filter_map(|k| k.as_str().map(String::from)).collect();
            let m: Vec<f64> = ks
                .iter()
                .map(|k| {
                    let t = f["trials"][k].as_array().cloned().unwrap_or_default();
                    t.iter().filter(|x| x["escaped"].as_bool().unwrap_or(false)).count() as f64 / t.len().max(1) as f64
                })
                .collect();
            validation::np_sum(&m) / m.len().max(1) as f64
        })
        .collect();
    json!({"metric": "escape probability by predator (Wilson 95% CI over trials) - MODEL PREDICTION", "rows": rows, "criteria": PRED_CRITERIA,
           "verdict": verdict, "per_fly": per_fly})
}

// --- orchard feeding ----------------------------------------------------------------------------------------------------------------
const ORCHARD_TRAVEL_S: f64 = 2.0;

/// `assays.orchard_fly`: one fly feeding in the Orchard for `duration_s`, in lockstep.
pub fn orchard_fly(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    seed: u64,
    options: &Value,
    surgery: &[(String, i8)],
) -> Result<Value, String> {
    use crate::world;
    let feeds = options["feeds"].as_u64().map_or(world::DEFAULT_FEEDS, |v| v as u32);
    let regrow_s = options["regrow_s"].as_f64().map_or(world::DEFAULT_REGROW_S, |v| v as f32);
    let cap = options["cap"].as_u64().map_or(world::DEFAULT_CAP, |v| v as usize);
    let duration_s = options["duration_s"].as_f64().unwrap_or(120.0);
    let build = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
    let mut br = build.brain_with(seed, None, 600, true);
    let g = validation::groups(labels, weights);
    fly_sim::assay_jobs::apply_surgery(&mut br, &g, surgery);
    validation::rest(&mut br, 200);
    let pam = br.groups.sense("reward", None).to_vec();
    let mn9 = g["mn9"].clone();
    let trees = world::scenery("orchard", 7).trees;
    let mut orch = world::Orchard::new(&trees, feeds, regrow_s, cap, seed);
    let n = br.n();
    let (mut pam_mask, mut mn9_mask) = (vec![false; n], vec![false; n]);
    for &r in &pam {
        pam_mask[r as usize] = true;
    }
    for &r in &mn9 {
        mn9_mask[r as usize] = true;
    }
    let mut pos = bevy::math::Vec3::new(0.0, 1.0, 0.0);
    let (mut t, dt) = (0.0f64, 0.005f64);
    let mut counts = [[0usize; 3]; 2]; // [feed, travel] x [PAM spikes, MN9 spikes, steps]
    let (mut n_feeds, mut emptied, mut fermented_feeds) = (0usize, 0usize, 0usize);
    let (mut timeline, mut next_sample) = (Vec::new(), 0.0f64);
    let run =
        |br: &mut fly_sim::Brain, orch_fermented: Option<bool>, steps: usize, phase: usize, counts: &mut [[usize; 3]; 2], t: &mut f64| {
            for i in 0..steps {
                if let Some(ferm) = orch_fermented {
                    if i % 3 == 0 {
                        br.poke("taste", None, 0.5);
                        br.poke_recruit("sweet", None, if ferm { 0.6 } else { 0.5 }, Some(0.6));
                        br.poke("reward", None, if ferm { 0.6 } else { 0.4 });
                        if ferm && i % 10 == 0 {
                            br.poke("scent:alcohol", None, 0.35);
                        }
                    }
                }
                let sp = br.step();
                counts[phase][0] += sp.iter().filter(|&&s| pam_mask[s as usize]).count();
                counts[phase][1] += sp.iter().filter(|&&s| mn9_mask[s as usize]).count();
                counts[phase][2] += 1;
                *t += dt;
            }
        };
    while t < duration_s {
        orch.step(t as f32);
        if t >= next_sample {
            let ripe: Vec<&world::Fruit> = orch.fruit.iter().filter(|f| f.ripe()).collect();
            timeline.push(json!({"t": (t * 100.0).round_ties_even() / 100.0, "ripe": ripe.len(), "fermented": ripe.iter().filter(|f| f.fermented).count(),
                                 "slots": orch.fruit.len(), "regrowing": orch.fruit.iter().filter(|f| !f.ripe()).count()}));
            next_sample += 5.0;
        }
        let nearest = orch.nearest_ripe(pos, true);
        run(&mut br, None, (ORCHARD_TRAVEL_S / dt) as usize, 1, &mut counts, &mut t);
        let Some(fi) = nearest.filter(|&i| orch.fruit[i].ripe()) else { continue };
        pos = orch.fruit[fi].pos;
        let ferm = orch.fruit[fi].fermented;
        run(&mut br, Some(ferm), (f64::from(world::FEED_BOUT_S) / dt) as usize, 0, &mut counts, &mut t);
        n_feeds += 1;
        fermented_feeds += usize::from(ferm);
        emptied += usize::from(orch.feed(fi, t as f32));
    }
    let (fe, tr) = (counts[0], counts[1]);
    let (pam_feed, pam_travel) = (hz_of(fe[0], pam.len(), fe[2]), hz_of(tr[0], pam.len(), tr[2]));
    let (mn9_feed, mn9_travel) = (hz_of(fe[1], mn9.len(), fe[2]), hz_of(tr[1], mn9.len(), tr[2]));
    Ok(json!({"seed": seed, "feeds_per_fruit": feeds, "regrow_s": regrow_s, "cap": cap, "duration_s": duration_s, "feeds": n_feeds,
              "fermented_feeds": fermented_feeds, "fruit_emptied": emptied, "pam_feed_hz": pam_feed, "pam_travel_hz": pam_travel,
              "pam_ratio": pam_feed / pam_travel.max(0.5), "mn9_feed_hz": mn9_feed, "mn9_travel_hz": mn9_travel,
              "mn9_ratio": mn9_feed / mn9_travel.max(0.5), "timeline": timeline}))
}

fn orchard_summarize(flies: &[Value]) -> Value {
    let col =
        |k: &str| -> Value { ci(fly_sim::stats::mean_ci(&flies.iter().map(|f| f[k].as_f64().unwrap_or(f64::NAN)).collect::<Vec<_>>())) };
    json!({"metric": "orchard feeding: MN9 and PAM feeding vs travelling, and the orchard's depletion", "mn9_ratio": col("mn9_ratio"),
           "pam_ratio": col("pam_ratio"), "feeds": col("feeds"), "fruit_emptied": col("fruit_emptied"), "fermented_feeds": col("fermented_feeds"),
           "settings": {"feeds_per_fruit": flies[0]["feeds_per_fruit"], "regrow_s": flies[0]["regrow_s"], "cap": flies[0]["cap"], "duration_s": flies[0]["duration_s"]},
           "per_fly": flies.iter().map(|f| f["mn9_ratio"].clone()).collect::<Vec<_>>()})
}

/// Register the three kinds with `fly_sim::assay_jobs` (once, at startup).
pub fn register() {
    fly_sim::assay_jobs::register_ext(fly_sim::assay_jobs::ExtAssays {
        task: |w, l, kind, seed, options, surgery| match kind {
            "orchard" => Some(orchard_fly(w, l, seed, options, surgery)),
            "predator_escape" => Some(predator_escape_fly(w, l, seed, options, surgery)),
            "hum_demo" => Some(hum_demo_fly(w, l, seed, options)),
            _ => None,
        },
        summarize: |kind, flies| match kind {
            "orchard" => Some(orchard_summarize(flies)),
            "predator_escape" => Some(predator_summarize(flies)),
            "hum_demo" => Some(hum_summarize(flies)),
            _ => None,
        },
    });
}
