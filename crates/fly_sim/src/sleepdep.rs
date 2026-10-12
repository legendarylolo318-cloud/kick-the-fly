//! The sleep-deprivation assay (`lab/sleepdep.py`): keep a fly awake through a stretch of the night, then measure its rebound sleep
//! against an undisturbed control of the same seed. Sleep pressure (a game rule) drives the real dFB (FB6/FB7) neurons; the dFB level
//! says whether the fly sleeps.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::brain::Brain;
use crate::cards::{Build, LIF_DEFAULTS};
use crate::csr::Weights;
use crate::duel::round_to;
use crate::outdoors;
use crate::pack::Labels;
use crate::validation::{np_sum, wilcoxon_greater};

pub const DAY_S: f64 = 300.0;
pub const START_PHASE: f64 = 0.80;
pub const SETTLE_S: f64 = 10.0;
pub const DEPRIVE_S: f64 = 45.0;
pub const RECOVER_S: f64 = 60.0;
pub const START_PRESSURE: f64 = 0.10;
pub const RISE_PER_S: f64 = 1.0 / 60.0;
pub const FALL_PER_S: f64 = 1.0 / 30.0;
pub const DFB_CURRENT_AT_FULL: f64 = 0.08;
pub const SLEEP_AFTER_S: f64 = 1.0;
pub const DISTURB_EVERY_S: f64 = 3.0;
pub const WAKE_HOLD_S: f64 = 2.0;
pub const DISTURB_POKE: f64 = 0.6;
pub const TICK_STEPS: usize = 4;
pub const P_MAX: f64 = 0.01;
pub const DEPRIVED_SHARE_MAX: f64 = 0.25;

pub const COMPONENTS: [(&str, &str); 7] = [
    ("sleep pressure (rises awake, falls asleep)", "GAME RULE"),
    ("the disturbances and the forced wake", "GAME RULE"),
    ("what counts as sleep: dFB above THRESH['sleep'] for 1 s", "GAME RULE"),
    ("pressure -> current on every FB6/FB7 neuron", "GAME RULE"),
    ("dFB (FB6/FB7) firing in response to that current", "CONNECTOME readout"),
    ("daylight on the photoreceptors and the l-LNv/s-LNv clock neurons", "CONNECTOME (the drive is a game rule)"),
    ("the rebound itself", "MODEL PREDICTION"),
];

/// One fly through settle -> deprivation window -> recovery window (`run_fly`). `disturb = false` is its undisturbed control.
/// `thresh_sleep` is `THRESH["sleep"]` (the Lab can change it). Returns `None` when cancelled.
#[allow(clippy::too_many_arguments)]
pub fn run_fly(
    brain: &mut Brain,
    seed: u64,
    disturb: bool,
    deprive_s: f64,
    recover_s: f64,
    settle_s: f64,
    thresh_sleep: f64,
    cancel: &AtomicBool,
) -> Option<Value> {
    let rows: Vec<u32> = brain.groups_ref().sense("sleep", None).to_vec();
    let dt = TICK_STEPS as f64 * (brain.sim.p.dt_ms / 1000.0);
    let t0 = START_PHASE * DAY_S;
    let total = settle_s + deprive_s + recover_s;
    let ticks = (total / dt).round_ties_even() as usize;
    let (mut pressure, mut above_since, mut asleep, mut wake_until, mut next_disturb) =
        (START_PRESSURE, None::<f64>, false, 0.0f64, settle_s);
    let mut sleep_s = [0.0f64; 3]; // settle, deprivation, recovery
    let mut dfb_last: Vec<f64> = Vec::new();
    let mut first_sleep_after: Option<f64> = None;
    let mut peak = pressure;
    let mut trace: Vec<[f64; 4]> = Vec::new();
    for tick in 0..ticks {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let t = tick as f64 * dt;
        let phase = if settle_s <= t && t < settle_s + deprive_s {
            1
        } else if t < settle_s {
            0
        } else {
            2
        };
        let (az, el) = outdoors::diurnal_cycle(t0 + t, DAY_S);
        let (left, right) = outdoors::sun_light(0.0, az, el);
        if left > 0.02 {
            brain.poke_recruit64("light", Some('L'), left, Some(0.25 * left));
        }
        if right > 0.02 {
            brain.poke_recruit64("light", Some('R'), right, Some(0.25 * right));
        }
        let clock = outdoors::morning_cells(el);
        if clock > 0.02 {
            brain.poke64("clock", None, clock);
        }
        if disturb && phase == 1 && t >= next_disturb {
            brain.poke64("legs", None, DISTURB_POKE);
            brain.poke64("body", None, DISTURB_POKE);
            wake_until = t + WAKE_HOLD_S;
            next_disturb += DISTURB_EVERY_S;
            asleep = false;
            above_since = None;
        }
        brain.set_current("sleep_pressure", &rows, (DFB_CURRENT_AT_FULL * pressure) as f32);
        for _ in 0..TICK_STEPS {
            brain.step();
        }
        let lvl = f64::from(brain.level("sleep"));
        if lvl > thresh_sleep && t >= wake_until {
            if above_since.is_none() {
                above_since = Some(t);
            }
            if !asleep && t - above_since.unwrap() >= SLEEP_AFTER_S {
                asleep = true;
            }
        } else {
            asleep = false;
            above_since = None;
        }
        if asleep {
            pressure = (pressure - FALL_PER_S * dt).max(0.0);
            sleep_s[phase] += dt;
            if phase == 2 && first_sleep_after.is_none() {
                first_sleep_after = Some(t - (settle_s + deprive_s));
            }
        } else {
            pressure = (pressure + RISE_PER_S * dt).min(1.0);
        }
        peak = peak.max(pressure);
        if phase == 1 && t >= settle_s + deprive_s - 5.0 {
            dfb_last.push(lvl);
        }
        if tick % 25 == 0 {
            trace.push([round_to(t, 2), round_to(pressure, 3), round_to(lvl, 2), if asleep { 1.0 } else { 0.0 }]);
        }
    }
    brain.clear_current("sleep_pressure");
    let want = settle_s + deprive_s;
    let at = trace.iter().fold(None::<&[f64; 4]>, |best, r| match best {
        Some(b) if (b[0] - want).abs() <= (r[0] - want).abs() => Some(b),
        _ => Some(r),
    });
    Some(json!({
        "seed": seed, "disturbed": disturb,
        "sleep_s": {"settle": round_to(sleep_s[0], 2), "deprivation": round_to(sleep_s[1], 2), "recovery": round_to(sleep_s[2], 2)},
        "pressure_end_of_deprivation": at.map_or(Value::Null, |r| json!(r[1])),
        "peak_pressure": round_to(peak, 3),
        "dfb_level_end_of_deprivation": if dfb_last.is_empty() { Value::Null } else { json!(np_sum(&dfb_last) / dfb_last.len() as f64) },
        "first_sleep_after_deprivation_s": first_sleep_after,
        "trace": trace,
    }))
}

fn field(r: &Value, path: &[&str]) -> f64 {
    let mut v = r;
    for p in path {
        v = &v[*p];
    }
    v.as_f64().unwrap_or(f64::NAN)
}

fn mean(x: &[f64]) -> f64 {
    np_sum(x) / x.len() as f64
}

fn nanmean(x: &[f64]) -> f64 {
    let v: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    if v.is_empty() { f64::NAN } else { mean(&v) }
}

fn wilcoxon_g(a: &[f64], b: &[f64]) -> f64 {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (p, q) in a.iter().zip(b) {
        if (p - q).is_finite() {
            x.push(*p);
            y.push(*q);
        }
    }
    if x.is_empty() { 1.0 } else { wilcoxon_greater(&x, &y) }
}

/// `aggregate`: the three pre-registered criteria over the paired flies.
#[allow(clippy::too_many_arguments)]
pub fn aggregate(
    seeds: &[u64],
    ctrl: &[Value],
    dep: &[Value],
    mode: &str,
    deprive_s: f64,
    recover_s: f64,
    seconds: f64,
    created: &str,
) -> Value {
    let col = |rs: &[Value], path: &[&str]| -> Vec<f64> { rs.iter().map(|r| field(r, path)).collect() };
    let rebound_c = col(ctrl, &["sleep_s", "recovery"]);
    let rebound_d = col(dep, &["sleep_s", "recovery"]);
    let sd_c = col(ctrl, &["sleep_s", "deprivation"]);
    let sd_d = col(dep, &["sleep_s", "deprivation"]);
    let dfb_c = col(ctrl, &["dfb_level_end_of_deprivation"]);
    let dfb_d = col(dep, &["dfb_level_end_of_deprivation"]);
    let p1 = wilcoxon_g(&rebound_d, &rebound_c);
    let p3 = wilcoxon_g(&dfb_d, &dfb_c);
    let share = if mean(&sd_c) > 0.0 { mean(&sd_d) / mean(&sd_c) } else { f64::NAN };
    let diff: Vec<f64> = rebound_d.iter().zip(&rebound_c).map(|(a, b)| a - b).collect();
    let dmean = mean(&diff);
    let sd = if diff.len() > 1 {
        let sq: Vec<f64> = diff.iter().map(|v| (v - dmean) * (v - dmean)).collect();
        (np_sum(&sq) / (diff.len() as f64 - 1.0)).sqrt()
    } else {
        0.0
    };
    let dfb_diff: Vec<f64> = dfb_d.iter().zip(&dfb_c).map(|(a, b)| a - b).collect();
    let dfb_mean = nanmean(&dfb_diff);
    let crit = json!({
        "S1": {"label": "rebound sleep, deprived - control (s)", "mean_difference": dmean,
               "dz": if sd > 1e-12 { json!(dmean / sd) } else { Value::Null }, "p": p1, "passed": p1 < P_MAX && dmean > 0.0},
        "S2": {"label": "deprived sleep as a share of control sleep during the window", "share": share,
               "passed": !share.is_nan() && share <= DEPRIVED_SHARE_MAX},
        "S3": {"label": "dFB level at the end of the window, deprived - control", "mean_difference": dfb_mean, "p": p3,
               "passed": p3 < P_MAX && dfb_mean > 0.0},
    });
    let all = ["S1", "S2", "S3"].iter().all(|k| crit[k]["passed"].as_bool().unwrap_or(false));
    json!({
        "kind": "sleep_deprivation", "created": created, "seconds": (seconds * 10.0).round() / 10.0, "seeds": seeds,
        "individuality": mode,
        "timeline": {"day_s": DAY_S, "start_phase": START_PHASE, "settle_s": SETTLE_S, "deprive_s": deprive_s, "recover_s": recover_s},
        "rules": {"rise_per_s": RISE_PER_S, "fall_per_s": FALL_PER_S, "dfb_current_at_full": DFB_CURRENT_AT_FULL,
                  "sleep_after_s": SLEEP_AFTER_S, "disturb_every_s": DISTURB_EVERY_S, "wake_hold_s": WAKE_HOLD_S,
                  "start_pressure": START_PRESSURE},
        "control": ctrl, "deprived": dep,
        "mean": {
            "recovery_sleep_control": mean(&rebound_c), "recovery_sleep_deprived": mean(&rebound_d),
            "deprivation_sleep_control": mean(&sd_c), "deprivation_sleep_deprived": mean(&sd_d),
            "dfb_control": nanmean(&dfb_c), "dfb_deprived": nanmean(&dfb_d),
            "pressure_control": mean(&col(ctrl, &["pressure_end_of_deprivation"])),
            "pressure_deprived": mean(&col(dep, &["pressure_end_of_deprivation"])),
        },
        "criteria": crit, "all_passed": all,
        "tags": {"sleep_pressure_and_disturbance": "GAME RULE", "dfb_firing_and_daylight": "CONNECTOME", "rebound": "MODEL PREDICTION"},
        "components": COMPONENTS.iter().map(|c| json!([c.0, c.1])).collect::<Vec<_>>(),
        "p_text": p_text(p1),
    })
}

fn p_text(p: f64) -> String {
    if p.is_nan() {
        "p = n/a".into()
    } else if p < 0.001 {
        "p < 0.001".into()
    } else {
        format!("p = {p:.3}")
    }
}

/// `run`: the assay over the seeds in `workers` threads; `progress(done, total, label)` returns false to cancel.
#[allow(clippy::too_many_arguments)]
pub fn run(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    seeds: &[u64],
    mode: &str,
    workers: usize,
    thresh_sleep: f64,
    created: &str,
    mut progress: impl FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let tasks: Vec<(u64, bool)> = seeds.iter().flat_map(|&s| [(s, false), (s, true)]).collect();
    let total = tasks.len();
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (tx, rx) = std::sync::mpsc::channel::<(u64, bool, Option<Value>)>();
    let t0 = std::time::Instant::now();
    let mut results: std::collections::HashMap<(u64, bool), Value> = Default::default();
    let mut cancelled = false;
    let tasks_ref = &tasks;
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1).min(total.max(1)) {
            let tx = tx.clone();
            let (next, stop) = (&next, &stop);
            scope.spawn(move || {
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(seed, disturb)) = tasks_ref.get(i) else { return };
                    let b = Build { weights, labels, mode, sigma: None, lif: LIF_DEFAULTS, wired: None };
                    let mut brain = b.brain_with(seed, None, 600, true);
                    let r = run_fly(&mut brain, seed, disturb, DEPRIVE_S, RECOVER_S, SETTLE_S, thresh_sleep, stop);
                    if tx.send((seed, disturb, r)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        while let Ok((seed, disturb, r)) = rx.recv() {
            match r {
                Some(v) => {
                    results.insert((seed, disturb), v);
                    let label = format!("fly {seed} {}", if disturb { "deprived" } else { "control" });
                    if !progress(results.len(), total, &label) {
                        cancelled = true;
                        stop.store(true, Ordering::Relaxed);
                    }
                }
                None => cancelled = true,
            }
        }
    });
    if cancelled {
        return Err("cancelled".into());
    }
    let ctrl: Vec<Value> = seeds.iter().map(|s| results[&(*s, false)].clone()).collect();
    let dep: Vec<Value> = seeds.iter().map(|s| results[&(*s, true)].clone()).collect();
    Ok(aggregate(seeds, &ctrl, &dep, mode, DEPRIVE_S, RECOVER_S, t0.elapsed().as_secs_f64(), created))
}

/// `fly_pair`: both conditions for one seed (the Lab assay path); the control and deprived flies share the seed.
pub fn fly_pair(weights: &Arc<Weights>, labels: &Arc<Labels>, seed: u64, mode: &str, thresh_sleep: f64) -> Result<Value, String> {
    let never = AtomicBool::new(false);
    let run = |disturb: bool| -> Result<Value, String> {
        let b = Build { weights, labels, mode, sigma: None, lif: LIF_DEFAULTS, wired: None };
        let mut brain = b.brain_with(seed, None, 600, true);
        run_fly(&mut brain, seed, disturb, DEPRIVE_S, RECOVER_S, SETTLE_S, thresh_sleep, &never).ok_or_else(|| "cancelled".to_string())
    };
    let control = run(false)?;
    let deprived = run(true)?;
    Ok(
        json!({"seed": seed, "deprive_s": DEPRIVE_S, "recover_s": RECOVER_S, "individuality": mode, "control": control, "deprived": deprived}),
    )
}

/// `summarize`: the assay's statistics over `fly_pair` results (the labjobs path).
pub fn summarize(flies: &[Value]) -> Value {
    let seeds: Vec<u64> = flies.iter().filter_map(|f| f["seed"].as_u64()).collect();
    let ctrl: Vec<Value> = flies.iter().map(|f| f["control"].clone()).collect();
    let dep: Vec<Value> = flies.iter().map(|f| f["deprived"].clone()).collect();
    let res = aggregate(
        &seeds,
        &ctrl,
        &dep,
        flies[0]["individuality"].as_str().unwrap_or("off"),
        flies[0]["deprive_s"].as_f64().unwrap_or(DEPRIVE_S),
        flies[0]["recover_s"].as_f64().unwrap_or(RECOVER_S),
        0.0,
        "",
    );
    let per_fly: Vec<f64> =
        ctrl.iter().zip(&dep).map(|(c, d)| field(d, &["sleep_s", "recovery"]) - field(c, &["sleep_s", "recovery"])).collect();
    json!({
        "metric": "rebound sleep, deprived minus control (s) - GAME RULE pressure through the CONNECTOME dFB; MODEL PREDICTION",
        "criteria": res["criteria"], "mean": res["mean"], "all_passed": res["all_passed"], "components": res["components"], "per_fly": per_fly,
    })
}
