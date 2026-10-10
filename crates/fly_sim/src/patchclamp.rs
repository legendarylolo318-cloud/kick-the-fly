//! Virtual patch clamp (`lab/patchclamp.py`): current-clamp one simulated neuron and read its potential, spikes and I-F curve. A point-neuron
//! model, not electrophysiology: the potential is in model units (threshold 1.0, reset 0.0).

use serde_json::{Value, json};

use crate::brain::Brain;
use crate::lif::LifParams;
use crate::rng::Generator;

pub const DT_MS: f64 = 5.0;
pub const SOURCE: &str = "patch";
pub const MAX_CURRENT: f64 = 5.0;
pub const UNITS_NOTE: &str =
    "model units: the firing threshold is 1.0 and the reset is 0.0; the tonic drive alone rests at bias/leak = 0.8. Not millivolts.";
pub const TAG_TEXT: &str = "MODEL: a point-neuron leaky integrate-and-fire unit. Not real electrophysiology.";

/// `check_current`: a finite number within +-MAX_CURRENT.
pub fn check_current(a: f64) -> Result<f64, String> {
    if a.abs() <= MAX_CURRENT { Ok(a) } else { Err(format!("the current must be a number within +-{MAX_CURRENT}")) }
}

#[derive(Debug, Clone)]
pub struct ClampProtocol {
    pub amplitudes: Vec<f64>,
    pub duration_ms: f64,
    pub pre_ms: f64,
    pub post_ms: f64,
    pub repeats: usize,
    pub gap_ms: f64,
    pub holding: f64,
}

impl Default for ClampProtocol {
    fn default() -> Self {
        Self {
            amplitudes: vec![0.0, 0.02, 0.05, 0.1, 0.2, 0.5],
            duration_ms: 500.0,
            pre_ms: 100.0,
            post_ms: 100.0,
            repeats: 1,
            gap_ms: 100.0,
            holding: 0.0,
        }
    }
}

impl ClampProtocol {
    /// The checks of `__post_init__`.
    pub fn check(&self) -> Result<(), String> {
        if self.amplitudes.is_empty() || self.amplitudes.len() > 200 {
            return Err("give between 1 and 200 current amplitudes".into());
        }
        if !self.amplitudes.iter().chain(std::iter::once(&self.holding)).all(|a| a.abs() <= MAX_CURRENT) {
            return Err(format!(
                "currents are limited to finite numbers within +-{MAX_CURRENT} (the validation suite's activation current is 0.5)"
            ));
        }
        for (name, v, lo, hi) in [
            ("duration_ms", self.duration_ms, 10.0, 20000.0),
            ("pre_ms", self.pre_ms, 0.0, 20000.0),
            ("post_ms", self.post_ms, 0.0, 20000.0),
            ("gap_ms", self.gap_ms, 0.0, 20000.0),
        ] {
            if !(lo..=hi).contains(&v) {
                return Err(format!("{name} must be between {lo} and {hi} ms"));
            }
        }
        if !(1..=100).contains(&self.repeats) {
            return Err("repeats must be between 1 and 100".into());
        }
        Ok(())
    }

    pub fn sweeps(&self) -> Vec<f64> {
        self.amplitudes.iter().flat_map(|&a| std::iter::repeat_n(a, self.repeats)).collect()
    }

    fn steps(ms: f64) -> usize {
        (ms / DT_MS).round_ties_even() as usize
    }

    pub fn total_steps(&self) -> usize {
        let per = Self::steps(self.pre_ms) + Self::steps(self.duration_ms) + Self::steps(self.post_ms) + Self::steps(self.gap_ms);
        per * self.sweeps().len()
    }

    /// Injected current at every step of the whole protocol.
    pub fn schedule(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for a in self.sweeps() {
            out.extend(std::iter::repeat_n(self.holding as f32, Self::steps(self.pre_ms)));
            out.extend(std::iter::repeat_n(a as f32, Self::steps(self.duration_ms)));
            out.extend(std::iter::repeat_n(self.holding as f32, Self::steps(self.post_ms) + Self::steps(self.gap_ms)));
        }
        out
    }

    /// `(amplitude, first step of the current step, last + 1)` for each sweep.
    pub fn sweep_windows(&self) -> Vec<(f64, usize, usize)> {
        let (mut out, mut t) = (Vec::new(), 0usize);
        for a in self.sweeps() {
            t += Self::steps(self.pre_ms);
            out.push((a, t, t + Self::steps(self.duration_ms)));
            t += Self::steps(self.duration_ms) + Self::steps(self.post_ms) + Self::steps(self.gap_ms);
        }
        out
    }
}

/// What the electrode recorded.
#[derive(Debug, Clone)]
pub struct Recording {
    pub v: Vec<f32>,
    pub spikes: Vec<bool>,
    pub current: Vec<f32>,
    pub mode: String,
    pub row: Option<u32>,
    pub meta: Value,
}

impl Recording {
    pub fn t_ms(&self) -> Vec<f64> {
        (0..self.v.len()).map(|i| i as f64 * DT_MS).collect()
    }

    pub fn spike_times_ms(&self) -> Vec<f64> {
        self.spikes.iter().enumerate().filter(|(_, s)| **s).map(|(i, _)| i as f64 * DT_MS).collect()
    }
}

/// `isolated`: one LIF unit run standalone with no synaptic input (the simulator's own update, float32).
pub fn isolated(amps: &[f32], p: &LifParams, seed: u64, settle_steps: usize) -> Recording {
    let leak = (p.dt_ms / p.tau_ms) as f32;
    let mut rng = Generator::new(seed);
    let n = settle_steps + amps.len();
    let noise: Vec<f32> = (0..n).map(|_| rng.standard_normal_f64() as f32 * (p.noise_std as f32)).collect();
    let mut cur = vec![0.0f32; settle_steps];
    cur.extend_from_slice(amps);
    let (mut v, mut refr) = (0.0f32, 0i32);
    let (mut vs, mut sp) = (vec![0.0f32; n], vec![false; n]);
    let keep = 1.0f32 - leak;
    for i in 0..n {
        let drive = (p.bias as f32 + noise[i]) + p.ext_gain as f32 * cur[i];
        v *= keep;
        if p.v_reset != 0.0 {
            v += leak * p.v_reset as f32;
        }
        v += drive;
        if refr > 0 {
            v = p.v_reset as f32;
            refr -= 1;
        }
        if v >= p.v_thresh as f32 {
            sp[i] = true;
            v = p.v_reset as f32;
            refr = i32::from(p.refractory_steps);
        }
        vs[i] = v;
    }
    Recording {
        v: vs[settle_steps..].to_vec(),
        spikes: sp[settle_steps..].to_vec(),
        current: cur[settle_steps..].to_vec(),
        mode: "isolated".into(),
        row: None,
        meta: json!({"seed": seed, "parameters": {
            "tau_ms": p.tau_ms, "dt_ms": p.dt_ms, "v_thresh": p.v_thresh, "v_reset": p.v_reset, "bias": p.bias, "noise_std": p.noise_std,
            "ext_gain": p.ext_gain, "refractory_steps": p.refractory_steps,
        }}),
    }
}

/// `describe_neuron`.
pub fn describe_neuron(br: &Brain, row: usize) -> Value {
    let l = br.labels();
    json!({
        "row": row, "type": l.types[row], "superclass": l.superclass[row], "instance": l.instance.get(row).cloned().unwrap_or_default(),
        "body_id": l.body_id.get(row), "region": l.region.get(row).cloned().unwrap_or_default(),
    })
}

/// `embedded`: the electrode on neuron `row` of a running brain; the current is a named source that is cleared at the end.
pub fn embedded(br: &mut Brain, row: usize, amps: &[f32], settle_steps: usize) -> Result<Recording, String> {
    if row >= br.n() {
        return Err(format!("neuron {row} is outside this brain (0-{})", br.n() - 1));
    }
    for _ in 0..settle_steps {
        br.step();
    }
    let n = amps.len();
    let (mut vs, mut sp) = (vec![0.0f32; n], vec![false; n]);
    let mut last: Option<f32> = None;
    for i in 0..n {
        let a = amps[i];
        if last != Some(a) {
            if a == 0.0 {
                br.clear_current(SOURCE);
            } else {
                br.set_current(SOURCE, &[row as u32], a);
            }
            last = Some(a);
        }
        br.step();
        vs[i] = br.sim.v[row];
        sp[i] = br.sim.spikes().binary_search(&(row as u32)).is_ok();
    }
    br.clear_current(SOURCE);
    Ok(Recording {
        v: vs,
        spikes: sp,
        current: amps.to_vec(),
        mode: "embedded".into(),
        row: Some(row as u32),
        meta: json!({"neuron": describe_neuron(br, row)}),
    })
}

/// `pick_neuron`: the index-th neuron (row order) of a spec: `type:`/`prefix:`/`line:`, a group name, or a bare cell type.
pub fn pick_neuron(br: &Brain, spec: &str, index: usize) -> Result<u32, String> {
    let spec = spec.trim();
    let mut rows: Vec<u32> = if spec.to_lowercase().starts_with("line:") {
        crate::genetics::rows_of_lines(br.labels(), spec)?
    } else if spec.contains(':') {
        br.rows_of(spec).map_err(|e| e.to_string())?
    } else {
        let det = br.groups_ref().det_members(spec);
        if !det.is_empty() {
            det
        } else {
            // a bare cell type (the page's own default, "DNp01"): the Python code refuses it as an unknown spec
            br.rows_of(&format!("type:{spec}")).map_err(|e| e.to_string())?
        }
    };
    rows.sort_unstable();
    if rows.is_empty() {
        return Err(format!("{spec:?} selects no neurons"));
    }
    rows.get(index).copied().ok_or_else(|| format!("{spec:?} has {} neurons; index {index} is out of range", rows.len()))
}

/// `run_current_clamp`.
pub fn run_current_clamp(
    br: &mut Brain,
    row: Option<u32>,
    proto: &ClampProtocol,
    mode: &str,
    seed: u64,
    p: &LifParams,
) -> Result<Recording, String> {
    proto.check()?;
    let sched = proto.schedule();
    let rec = match mode {
        "embedded" => embedded(br, row.ok_or("an embedded recording needs a neuron")? as usize, &sched, 0)?,
        "isolated" => return run_isolated(row, proto, seed, p),
        _ => return Err("mode must be one of (\"embedded\", \"isolated\")".into()),
    };
    Ok(tag_recording(rec, proto))
}

/// `run_current_clamp(..., "isolated")`: no brain is needed.
pub fn run_isolated(row: Option<u32>, proto: &ClampProtocol, seed: u64, p: &LifParams) -> Result<Recording, String> {
    proto.check()?;
    let mut r = isolated(&proto.schedule(), p, seed, 400);
    r.row = row;
    Ok(tag_recording(r, proto))
}

fn tag_recording(mut rec: Recording, proto: &ClampProtocol) -> Recording {
    rec.meta["protocol"] = json!({
        "amplitudes": proto.amplitudes, "duration_ms": proto.duration_ms, "pre_ms": proto.pre_ms, "post_ms": proto.post_ms,
        "repeats": proto.repeats, "gap_ms": proto.gap_ms, "holding": proto.holding,
    });
    rec.meta["tag"] = json!(TAG_TEXT);
    rec.meta["units"] = json!(UNITS_NOTE);
    rec
}

#[allow(clippy::too_many_arguments)]
/// `if_curve`: firing rate (spikes/s over the step) against injected current.
pub fn if_curve(
    br: &mut Brain,
    row: Option<u32>,
    amplitudes: &[f64],
    duration_ms: f64,
    repeats: usize,
    mode: &str,
    seed: u64,
    p: &LifParams,
) -> Result<Value, String> {
    let proto = ClampProtocol { amplitudes: amplitudes.to_vec(), duration_ms, repeats, ..Default::default() };
    let rec = run_current_clamp(br, row, &proto, mode, seed, p)?;
    Ok(curve_of(&proto, rec, row, amplitudes, duration_ms, repeats, mode))
}

/// `if_curve` for the isolated mode: needs no brain.
pub fn if_curve_isolated(
    row: Option<u32>,
    amplitudes: &[f64],
    duration_ms: f64,
    repeats: usize,
    seed: u64,
    p: &LifParams,
) -> Result<Value, String> {
    let proto = ClampProtocol { amplitudes: amplitudes.to_vec(), duration_ms, repeats, ..Default::default() };
    let rec = run_isolated(row, &proto, seed, p)?;
    Ok(curve_of(&proto, rec, row, amplitudes, duration_ms, repeats, "isolated"))
}

fn curve_of(
    proto: &ClampProtocol,
    rec: Recording,
    row: Option<u32>,
    _amplitudes: &[f64],
    duration_ms: f64,
    repeats: usize,
    mode: &str,
) -> Value {
    let mut per: std::collections::BTreeMap<String, (f64, Vec<f64>)> = Default::default();
    for (a, lo, hi) in proto.sweep_windows() {
        let n = rec.spikes[lo..hi].iter().filter(|s| **s).count() as f64;
        per.entry(crate::cards::py_float(a)).or_insert((a, Vec::new())).1.push(n / ((hi - lo) as f64 * DT_MS / 1000.0).max(1e-9));
    }
    let mut amps: Vec<(f64, Vec<f64>)> = per.into_values().collect();
    amps.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mean = |v: &[f64]| crate::validation::np_sum(v) / v.len() as f64;
    let rates: Vec<f64> = amps.iter().map(|(_, v)| mean(v)).collect();
    let sds: Vec<Value> = amps
        .iter()
        .map(|(_, v)| {
            if v.len() > 1 {
                let m = mean(v);
                let sq: Vec<f64> = v.iter().map(|x| (x - m) * (x - m)).collect();
                json!((crate::validation::np_sum(&sq) / (v.len() as f64 - 1.0)).sqrt())
            } else {
                Value::Null
            }
        })
        .collect();
    let rheobase = amps.iter().zip(&rates).find(|((a, _), r)| **r > 0.0 && *a > 0.0).map(|((a, _), _)| *a);
    json!({
        "amplitudes": amps.iter().map(|a| a.0).collect::<Vec<_>>(), "rate_hz": rates, "rate_sd": sds,
        "sweeps": amps.iter().map(|(a, v)| (crate::cards::py_float(*a), json!(v))).collect::<serde_json::Map<_, _>>(),
        "mode": mode, "row": row, "duration_ms": duration_ms, "repeats": repeats, "rheobase": rheobase, "tag": TAG_TEXT, "units": UNITS_NOTE,
        "recording": {"v": rec.v, "spikes": rec.spikes, "current": rec.current},
    })
}

fn g6(x: f64) -> String {
    crate::sensitivity::fmt_gp(x, 6)
}

/// `export_csv`.
pub fn export_csv(rec: &Recording, path: &std::path::Path) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    // the dict's own key order (serde_json sorts keys)
    let neuron = rec.meta.get("neuron").map_or("None".to_string(), |n| {
        let parts: Vec<String> =
            ["row", "type", "superclass", "instance", "body_id", "region"].iter().map(|k| format!("'{k}': {}", py_repr(&n[*k]))).collect();
        format!("{{{}}}", parts.join(", "))
    });
    let mut text = format!(
        "# {TAG_TEXT}\n# membrane_potential is in {UNITS_NOTE}\n# mode={} row={} neuron={neuron}\n",
        rec.mode,
        rec.row.map_or("None".into(), |r| r.to_string())
    );
    text.push_str("time_ms,membrane_potential_model_units,injected_current_model_units,spike\r\n");
    for (i, t) in rec.t_ms().iter().enumerate() {
        text.push_str(&format!("{t:.1},{},{},{}\r\n", g6(f64::from(rec.v[i])), g6(f64::from(rec.current[i])), i32::from(rec.spikes[i])));
    }
    std::fs::write(path, text)
}

/// `export_if_csv`.
pub fn export_if_csv(curve: &Value, path: &std::path::Path) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let row = if curve["row"].is_null() { "None".to_string() } else { curve["row"].to_string() };
    let mut text = format!(
        "# {TAG_TEXT}\n# mode={} row={row} step={} ms, {} sweeps per amplitude; current in the simulator's units\n",
        curve["mode"].as_str().unwrap_or(""),
        crate::cards::py_float(curve["duration_ms"].as_f64().unwrap_or(0.0)),
        curve["repeats"]
    );
    text.push_str("current,rate_hz_mean,rate_hz_sd,sweeps_hz\r\n");
    for (i, a) in curve["amplitudes"].as_array().into_iter().flatten().enumerate() {
        let a = a.as_f64().unwrap_or(0.0);
        let g4 = |x: f64| crate::sensitivity::fmt_gp(x, 4);
        let sd = curve["rate_sd"][i].as_f64().map_or(String::new(), g4);
        let sweeps: Vec<String> =
            curve["sweeps"][crate::cards::py_float(a)].as_array().into_iter().flatten().filter_map(Value::as_f64).map(g4).collect();
        text.push_str(&format!("{},{},{sd},{}\r\n", g6(a), g4(curve["rate_hz"][i].as_f64().unwrap_or(0.0)), sweeps.join(" ")));
    }
    std::fs::write(path, text)
}

/// A Python `repr` of a small JSON dict (the CSV header prints the neuron's dict).
fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::String(s) => format!("'{}'", s.replace('\'', "\\'")),
        Value::Number(n) => n.as_f64().filter(|_| n.is_f64()).map_or_else(|| n.to_string(), crate::cards::py_float),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("'{k}': {}", py_repr(v))).collect::<Vec<_>>().join(", ")),
    }
}
