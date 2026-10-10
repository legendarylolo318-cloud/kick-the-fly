//! Thermogenetics (`lab/thermogenetics.py`): TrpA1 activates the neurons that express it above a temperature, shibire-ts silences them.
//! The expression is a neuron spec, the temperature a number, and the effector turns into a current on those neurons (a named source of
//! the brain, `"thermo"`). Also the escape-vs-temperature assay.

use serde_json::{Value, json};

use crate::brain::Brain;
use crate::rng::Generator;
use crate::stats;

pub const ARENA_COLD_C: f64 = 15.0;
pub const ARENA_HOT_C: f64 = 35.0;
pub const TEMP_RANGE_C: (f64, f64) = (10.0, 45.0);
pub const ACTIVATE_CURRENT: f64 = 0.5;
pub const SILENCE_CURRENT: f64 = -0.6;
pub const MAX_STRENGTH: f64 = 3.0;
pub const SOURCE: &str = "thermo";

pub struct Effector {
    pub key: &'static str,
    pub label: &'static str,
    pub activates: bool,
    pub onset_c: f64,
    pub full_c: f64,
    pub tau_on_s: f64,
    pub tau_off_s: f64,
    pub current: f64,
    pub cited: &'static str,
    pub rule: &'static str,
}

impl Effector {
    pub fn target_fraction(&self, t: f64) -> f64 {
        if t <= self.onset_c {
            0.0
        } else if t >= self.full_c {
            1.0
        } else {
            (t - self.onset_c) / (self.full_c - self.onset_c)
        }
    }
}

pub const EFFECTORS: [Effector; 2] = [
    Effector {
        key: "trpa1",
        label: "TrpA1 (activates above a threshold temperature)",
        activates: true,
        onset_c: 25.0,
        full_c: 29.0,
        tau_on_s: 1.0,
        tau_off_s: 1.0,
        current: ACTIVATE_CURRENT,
        cited: "onset near 25 C (Pulver 2009: tonic spiking at about 25 C on a 21-27 C ramp; little adaptation)",
        rule: "full activation at 29 C, the 1 s kinetics, and the current size are game rules",
    },
    Effector {
        key: "shibire",
        label: "shibire-ts (silences above a threshold temperature)",
        activates: false,
        onset_c: 28.0,
        full_c: 30.0,
        tau_on_s: 40.0,
        tau_off_s: 20.0,
        current: SILENCE_CURRENT,
        cited: "restrictive temperature 30 C: motionless within 2 min, walking again within about 1 min of cooling (Kitamoto 2001)",
        rule: "the 28 C start of the ramp, the 40 s / 20 s time constants and the silencing current are game rules",
    },
];

/// `thermogenetics.effector`: by name or alias.
pub fn effector(name: &str) -> Result<&'static Effector, String> {
    let key = match name.trim().to_lowercase().as_str() {
        "trpa1" | "dtrpa1" => "trpa1",
        "shibire" | "shibire-ts" | "shi" | "shits" | "shi-ts" | "shibire_ts" => "shibire",
        _ => return Err(format!("unknown effector {name:?}; use one of [\"shibire\", \"trpa1\"]")),
    };
    Ok(EFFECTORS.iter().find(|e| e.key == key).expect("both keys exist"))
}

/// The thermo arena's temperature at x position `t` in [-1, +1] (a game rule).
pub fn arena_temperature(t: f64) -> f64 {
    let (mid, half) = ((ARENA_HOT_C + ARENA_COLD_C) / 2.0, (ARENA_HOT_C - ARENA_COLD_C) / 2.0);
    mid + half * t.clamp(-1.0, 1.0)
}

#[derive(Debug, Clone)]
pub struct Expression {
    pub effector: &'static str,
    pub target: String,
    pub strength: f64,
    pub rows: Option<Vec<u32>>,
}

impl Expression {
    pub fn new(effector_name: &str, target: &str, strength: f64) -> Result<Self, String> {
        let e = effector(effector_name)?;
        if !(0.0..=MAX_STRENGTH).contains(&strength) {
            return Err(format!("strength must be a number from 0 to {MAX_STRENGTH}"));
        }
        Ok(Self { effector: e.key, target: target.to_string(), strength, rows: None })
    }
}

/// A set of expressions and their live state on one brain.
pub struct Thermogenetics {
    pub expressions: Vec<Expression>,
    pub steady: bool,
    pub time_scale: f64,
    pub activation: Vec<f64>,
    pub temperature_c: f64,
    written: Option<Vec<f32>>,
}

impl Thermogenetics {
    pub fn new(expressions: Vec<Expression>, steady: bool, time_scale: f64) -> Self {
        let n = expressions.len();
        Self { expressions, steady, time_scale: time_scale.max(1e-3), activation: vec![0.0; n], temperature_c: 22.0, written: None }
    }

    /// Resolve every target on this brain; an error names the first that selects nothing.
    pub fn resolve(&mut self, br: &Brain) -> Result<(), String> {
        for e in &mut self.expressions {
            if e.rows.is_none() {
                let rows = if e.target.to_lowercase().starts_with("line:") {
                    crate::genetics::rows_of_lines(br.labels(), &e.target)?
                } else {
                    br.rows_of(&e.target).map_err(|x| x.to_string())?
                };
                if rows.is_empty() {
                    return Err(format!("{:?} selects no neurons in this brain", e.target));
                }
                e.rows = Some(rows);
            }
        }
        Ok(())
    }

    pub fn neurons(&self) -> usize {
        let mut all: Vec<u32> = self.expressions.iter().filter_map(|e| e.rows.as_ref()).flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        all.len()
    }

    /// Advance the activations by `dt_s` at this temperature and write the currents; the largest activation.
    pub fn update(&mut self, br: &mut Brain, temperature_c: f64, dt_s: f64) -> Result<f64, String> {
        self.resolve(br)?;
        self.temperature_c = temperature_c;
        let n = br.n();
        let mut cur: Option<Vec<f32>> = (!self.expressions.is_empty()).then(|| vec![0.0f32; n]);
        for (i, e) in self.expressions.iter().enumerate() {
            let ef = effector(e.effector)?;
            let goal = ef.target_fraction(temperature_c);
            if self.steady {
                self.activation[i] = goal;
            } else {
                let tau = (if goal > self.activation[i] { ef.tau_on_s } else { ef.tau_off_s }) / self.time_scale;
                self.activation[i] += (goal - self.activation[i]) * (1.0 - (-dt_s / tau).exp());
                if (goal - self.activation[i]).abs() < 1e-4 {
                    self.activation[i] = goal;
                }
            }
            if self.activation[i] > 0.0 {
                let v = (ef.current * e.strength * self.activation[i]) as f32;
                if let (Some(c), Some(rows)) = (cur.as_mut(), e.rows.as_ref()) {
                    for &r in rows {
                        c[r as usize] += v;
                    }
                }
            }
        }
        self.apply(br, cur);
        Ok(self.activation.iter().copied().fold(0.0, f64::max))
    }

    fn apply(&mut self, br: &mut Brain, cur: Option<Vec<f32>>) {
        let Some(cur) = cur.filter(|c| c.iter().any(|&v| v != 0.0)) else {
            if self.written.take().is_some() {
                br.clear_current(SOURCE);
            }
            return;
        };
        if self.written.as_ref() == Some(&cur) {
            return;
        }
        let rows: Vec<u32> = (0..cur.len() as u32).filter(|&i| cur[i as usize] != 0.0).collect();
        let vals: Vec<f32> = rows.iter().map(|&r| cur[r as usize]).collect();
        br.set_current_values(SOURCE, &rows, &vals);
        self.written = Some(cur);
    }

    pub fn clear(&mut self, br: &mut Brain) {
        br.clear_current(SOURCE);
        self.written = None;
        self.activation = vec![0.0; self.expressions.len()];
    }

    /// One row per expression (the page's status).
    pub fn describe(&self) -> Vec<Value> {
        self.expressions
            .iter()
            .zip(&self.activation)
            .map(|(e, a)| {
                let ef = effector(e.effector).expect("an expression holds a known effector");
                json!({
                    "effector": ef.label, "target": e.target, "neurons": e.rows.as_ref().map_or(0, Vec::len), "activation": a,
                    "onset_c": ef.onset_c, "full_c": ef.full_c, "cited": ef.cited, "rule": ef.rule,
                })
            })
            .collect()
    }
}

// --- the assay: thermogenetic activation of DNp01, escape vs temperature ----------------------------------------------------------------
pub const ASSAY_TEMPS: [f64; 8] = [18.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0];
const HOLD_STEPS: usize = 200;

/// One fly (`escape_fly`): at each temperature, shuffled by the seed, once with the effector in `target` and once as its own control.
pub fn escape_fly(
    br: &mut Brain,
    dn: &[u32],
    seed: u64,
    temps: &[f64],
    target: &str,
    effector_name: &str,
    thresh_escape: f64,
) -> Result<Value, String> {
    let mut order: Vec<(f64, bool)> = temps.iter().flat_map(|&t| [(t, true), (t, false)]).collect();
    Generator::new(seed + 11).shuffle(&mut order);
    let mut out: std::collections::BTreeMap<String, serde_json::Map<String, Value>> =
        temps.iter().map(|t| (crate::cards::py_float(*t), Default::default())).collect();
    for (t, expr) in order {
        let exprs = if expr { vec![Expression::new(effector_name, target, 1.0)?] } else { vec![] };
        let mut th = Thermogenetics::new(exprs, true, 1.0);
        th.update(br, t, 0.05)?;
        let (mut peak, mut spikes) = (0.0f64, 0usize);
        for _ in 0..HOLD_STEPS {
            br.step();
            peak = peak.max(f64::from(br.level("escape")));
            let sp = br.sim.spikes();
            spikes += dn.iter().filter(|r| sp.binary_search(r).is_ok()).count();
        }
        th.clear(br);
        out.get_mut(&crate::cards::py_float(t)).expect("every temperature has a slot").insert(
            if expr { "expressing" } else { "control" }.into(),
            json!({
                "escaped": peak > thresh_escape, "peak_level": peak,
                "dnp01_hz": spikes as f64 / dn.len().max(1) as f64 / (HOLD_STEPS as f64 * 0.005),
            }),
        );
        for _ in 0..400 {
            br.step();
        }
    }
    Ok(json!({"seed": seed, "temps": temps, "target": target, "effector": effector_name, "trials": out}))
}

fn ci(values: &[f64]) -> Value {
    let c = stats::mean_ci(values);
    let f = |x: f64| if x.is_nan() { Value::Null } else { json!(x) };
    json!({"mean": f(c.mean), "sd": f(c.sd), "n": c.n, "lo": f(c.lo), "hi": f(c.hi)})
}

/// `summarize`: escape rate vs temperature, expressing vs control, with Fisher exact p per temperature.
pub fn summarize(flies: &[Value]) -> Value {
    let temps: Vec<f64> = flies[0]["temps"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
    let mut rows = Vec::new();
    let mut per_fly = vec![0.0f64; flies.len()];
    for &t in &temps {
        let key = crate::cards::py_float(t);
        let e: Vec<bool> = flies.iter().map(|f| f["trials"][&key]["expressing"]["escaped"].as_bool().unwrap_or(false)).collect();
        let c: Vec<bool> = flies.iter().map(|f| f["trials"][&key]["control"]["escaped"].as_bool().unwrap_or(false)).collect();
        let (ne, nc) = (e.iter().filter(|&&x| x).count(), c.iter().filter(|&&x| x).count());
        let hz =
            |who: &str| -> Vec<f64> { flies.iter().map(|f| f["trials"][&key][who]["dnp01_hz"].as_f64().unwrap_or(f64::NAN)).collect() };
        rows.push(json!({
            "temperature_c": t, "escapes": ne, "control_escapes": nc, "flies": flies.len(),
            "escape_rate": ne as f64 / flies.len() as f64, "control_rate": nc as f64 / flies.len() as f64,
            "dnp01_hz": ci(&hz("expressing")), "control_dnp01_hz": ci(&hz("control")),
            "fisher_p": stats::fisher_exact(ne as u64, e.len() as u64, nc as u64, c.len() as u64),
        }));
        for (i, x) in e.iter().enumerate() {
            per_fly[i] += f64::from(u8::from(*x));
        }
    }
    let per_fly: Vec<f64> = per_fly.iter().map(|s| s / temps.len() as f64).collect();
    json!({"metric": "thermogenetic activation: escape rate vs temperature (expressing vs control)", "rows": rows, "per_fly": per_fly})
}
