//! Running protocol files (`lab/protocol.py:run`, `run_seed`, `run_patch`, `run_rig`, `run_screen`): a new folder per run with, per fly, the
//! recorder's CSV/npz/metadata files, and a `summary.json`. Checking is `ktf_data::protocol`.
//!
//! Not done: the NWB file and the TIFF stack (no HDF5/TIFF writers), `--record-replay`. The weather and the predator's loom use this
//! port's own generators, so those two blocks are statistically, not spike for spike, the Python run.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fly_sim::brain::Brain;
use fly_sim::cards::Build;
use fly_sim::csr::Weights;
use fly_sim::pack::Labels;
use fly_sim::validation;
use ktf_data::protocol::folder_name;
use serde_json::{Value, json};

type Groups = HashMap<&'static str, Vec<u32>>;

/// `_rows`: an assay group, a detail group or population by name, `line:` specs, or `simcore.rows_of`'s specs.
pub fn rows_of_spec(br: &Brain, g: &Groups, spec: &str) -> Result<Vec<u32>, String> {
    if let Some(r) = g.get(spec) {
        return Ok(r.clone());
    }
    if spec.starts_with("line:") {
        return fly_sim::genetics::rows_of_lines(br.labels(), spec);
    }
    if !spec.contains(':') {
        let s = br.groups_ref().sense(spec, None);
        if !s.is_empty() {
            return Ok(s.to_vec());
        }
        if let Some(r) = br.groups_ref().group_rows(spec) {
            return Ok(r);
        }
        return Err(format!("unknown neuron spec '{spec}'"));
    }
    br.rows_of(spec).map_err(|e| e.to_string())
}

fn lif_of(p: &Value) -> [f64; 5] {
    let pairs: Vec<(String, f64)> =
        p["params"].as_object().into_iter().flatten().filter_map(|(k, v)| v.as_f64().map(|x| (k.clone(), x))).collect();
    fly_sim::cards::lif_values(&pairs)
}

fn surgery_of(v: &Value) -> Vec<(String, i8)> {
    v.as_object().into_iter().flatten().map(|(k, m)| (k.clone(), m.as_i64().unwrap_or(0) as i8)).collect()
}

/// `bundle.spike_sha256`: a fingerprint of a recording's spikes.
pub fn spike_sha256(steps: &[i32], index: &[i32], n_steps: i64) -> String {
    let mut order: Vec<usize> = (0..steps.len()).collect();
    order.sort_by_key(|&k| (steps[k], index[k]));
    let mut buf: Vec<u8> = Vec::with_capacity(8 + 16 * steps.len());
    buf.extend(n_steps.to_le_bytes());
    for &k in &order {
        buf.extend(i64::from(steps[k]).to_le_bytes());
    }
    for &k in &order {
        buf.extend(i64::from(index[k]).to_le_bytes());
    }
    fly_sim::util::sha256_hex(&buf)
}

/// The Lab recording of a rig run (`rigassay.rec_groups`): spikes of the steering, walking, T4/T5 and LC10 neurons and the body.
#[derive(Default)]
struct RigRecorder {
    rec: Option<ktf_data::recorder::Recorder>,
}

impl fly_sim::rigs::RigObserver for RigRecorder {
    fn begin(&mut self, br: &mut Brain, g: &Groups) {
        br.log_stimuli(true);
        let cat = |a: &str, b: &str| -> Vec<u32> { g[a].iter().chain(&g[b]).copied().collect() };
        let t = &br.labels().types;
        let lc10: Vec<u32> = (0..t.len() as u32).filter(|&i| t[i as usize].starts_with("LC10")).collect();
        let groups = vec![
            ("steer_R".to_string(), g["dna_steer_r"].clone()),
            ("steer_L".to_string(), g["dna_steer_l"].clone()),
            ("walk_DNp09".to_string(), g["dnp09"].clone()),
            ("T4T5_progressive".to_string(), cat("t45_prog_r", "t45_prog_l")),
            ("T4T5_regressive".to_string(), cat("t45_regr_r", "t45_regr_l")),
            ("LC10".to_string(), lc10),
        ];
        self.rec = Some(ktf_data::recorder::Recorder::new(br.n(), groups, br.steps));
    }

    fn step(&mut self, step: u64, spikes: &[u32]) {
        if let Some(r) = self.rec.as_mut() {
            r.push(step, spikes);
        }
    }

    fn stimuli(&mut self, events: Vec<(u64, String, String, f64)>) {
        if let Some(r) = self.rec.as_mut() {
            for (st, label, detail, v) in events {
                r.log_event(st, "stimulus", &label, &detail, v);
            }
        }
    }

    fn kinematics(&mut self, step: u64, x: f64, y: f64, speed: f64, action: &str, arena: &str) {
        if let Some(r) = self.rec.as_mut() {
            r.log_kinematics(step, x, y, 0.0, speed, 100.0, action, arena);
        }
    }
}

pub struct Env<'a> {
    pub weights: &'a Arc<Weights>,
    pub labels: &'a Arc<Labels>,
    pub counts: Option<Arc<Vec<i32>>>,
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

fn mean_ci_json(c: fly_sim::stats::MeanCi) -> Value {
    let f = |x: f64| if x.is_finite() { json!(x) } else { Value::Null };
    json!({"mean": f(c.mean), "sd": f(c.sd), "n": c.n, "lo": f(c.lo), "hi": f(c.hi)})
}

fn cancelled(env: &Env) -> bool {
    env.cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
}

/// One fly of a stimulus protocol: the mean firing per recording group.
pub fn run_seed(
    env: &Env,
    p: &Value,
    seed: u64,
    surgery: &[(String, i8)],
    folder: &Path,
    tag: &str,
) -> Result<BTreeMap<String, f64>, String> {
    let lif = lif_of(p);
    let wiring = match p.get("drug") {
        Some(d) => {
            let doses: Vec<(String, f64)> =
                d["doses"].as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(f64::NAN))).collect();
            let refs: Vec<(&str, f64)> = doses.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            Some(fly_sim::pharmacology::wiring_for(
                &refs,
                d["include_low_confidence"].as_bool().unwrap_or(true),
                d["cut"].as_f64().unwrap_or(fly_sim::pharmacology::DEFAULT_CUT),
                &Default::default(),
            )?)
        }
        None => None,
    };
    let wired: Option<Arc<Weights>> = match (&wiring, &env.counts) {
        (Some(w), Some(c)) => Some(Arc::new(fly_sim::wiring::apply(env.weights, env.labels, c, w))),
        (Some(_), None) => return Err("the drug block needs the brain pack".into()),
        _ => None,
    };
    let build = Build { weights: env.weights, labels: env.labels, mode: "off", sigma: None, lif, wired: wired.as_ref() };
    let mut br = build.brain_with(seed, None, 600, true);
    let g = validation::groups(env.labels, env.weights);
    fly_sim::assay_jobs::apply_surgery(&mut br, &g, surgery);
    let warm = (p["warmup_s"].as_f64().unwrap_or(1.0) / 0.005).round_ties_even() as usize;
    validation::rest(&mut br, warm);
    // the groups to record and the stimuli
    let mut named: Vec<(String, Vec<u32>)> = Vec::new();
    for (i, r) in p["recordings"].as_array().into_iter().flatten().enumerate() {
        let name = r["name"].as_str().map_or_else(|| format!("rec{i}"), String::from);
        let rows = rows_of_spec(&br, &g, r["neurons"].as_str().unwrap_or(""))?;
        match named.iter_mut().find(|n| n.0 == name) {
            Some(e) => e.1 = rows,
            None => named.push((name, rows)),
        }
    }
    struct Stim {
        rows: Vec<u32>,
        start: i64,
        stop: i64,
        drive: bool,
        amp: f32,
        strength: f64,
        recruit: Option<f64>,
        region: String,
    }
    let mut stims = Vec::new();
    for s in p["stimuli"].as_array().into_iter().flatten() {
        let target = s["target"].as_str().map_or_else(|| s["target"].to_string(), String::from);
        let rows = rows_of_spec(&br, &g, &target)?;
        let at = s["at_s"].as_f64().unwrap_or(0.0);
        let start = (at / 0.005).round_ties_even() as i64;
        let len = ((s["for_s"].as_f64().unwrap_or(0.5) / 0.005).round_ties_even() as i64).max(1);
        let at_txt = match &s["at_s"] {
            Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
            _ => fly_sim::cards::py_float(at),
        };
        let region = format!("protocol:{target}{at_txt}");
        let drive = s["mode"].as_str() == Some("drive");
        if !drive {
            br.groups.add_sense(&region, rows.clone());
        }
        stims.push(Stim {
            rows,
            start,
            stop: start + len,
            drive,
            amp: s["amp"].as_f64().unwrap_or(0.5) as f32,
            strength: s["strength"].as_f64().unwrap_or(0.8).clamp(0.0, 1.0),
            recruit: s["recruit"].as_f64(),
            region,
        });
    }
    let mut rec = ktf_data::recorder::Recorder::new(br.n(), named.clone(), br.steps);
    br.log_stimuli(true);
    let n = (p["duration_s"].as_f64().unwrap_or(2.0) / 0.005).round_ties_even() as i64;
    let mut driving: std::collections::HashSet<usize> = Default::default();
    // thermogenetics
    let th_spec = p.get("thermogenetics").cloned();
    let mut th = match &th_spec {
        Some(s) => {
            let ex: Result<Vec<_>, String> = s["expression"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|e| {
                    fly_sim::thermogenetics::Expression::new(
                        e["effector"].as_str().unwrap_or(""),
                        e["target"].as_str().unwrap_or(""),
                        e["strength"].as_f64().unwrap_or(1.0),
                    )
                })
                .collect();
            Some(fly_sim::thermogenetics::Thermogenetics::new(
                ex?,
                s["kinetics"].as_str() == Some("steady"),
                s["time_scale"].as_f64().unwrap_or(1.0),
            ))
        }
        None => None,
    };
    let temperature_at = |t_s: f64| -> f64 {
        let t = &th_spec.as_ref().map_or(json!(22.0), |s| s.get("temperature_c").cloned().unwrap_or(json!(22.0)));
        if let Some(x) = t.as_f64() {
            return x;
        }
        let mut pts: Vec<(f64, f64)> =
            t.as_array().into_iter().flatten().map(|p| (p["at_s"].as_f64().unwrap_or(0.0), p["c"].as_f64().unwrap_or(22.0))).collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut cur = pts.first().map_or(22.0, |p| p.1);
        for (at, c) in pts {
            if t_s >= at {
                cur = c;
            }
        }
        cur
    };
    // weather
    let mut wx = p.get("weather").map(|_| crate::weather::Weather::new(seed));
    // audio: a synthetic hum, analysed once
    let mut hum: Option<(Vec<crate::mic_analysis::Reading>, Vec<u32>, Vec<u32>)> = None;
    if let Some(a) = p.get("audio") {
        let rate = 22050.0f32;
        let (hz, secs, amp) = (a["hz"].as_f64().unwrap_or(200.0), a["seconds"].as_f64().unwrap_or(1.0), a["amp"].as_f64().unwrap_or(0.1));
        let ipi = a["ipi_ms"].as_f64();
        let count = (secs * f64::from(rate)).round_ties_even() as usize;
        let wave: Vec<f32> = (0..count)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let mut x = (2.0 * std::f64::consts::PI * hz * t).sin();
                if let Some(ipi) = ipi.filter(|v| *v > 0.0) {
                    let phase = (t * 1000.0).rem_euclid(ipi);
                    let env = if phase < 12.0 { (std::f64::consts::PI * phase / 12.0).sin().powi(2) } else { 0.0 };
                    x *= env;
                }
                (amp * x) as f32
            })
            .collect();
        let mut an = crate::mic_analysis::Analyzer::new(rate, a["sensitivity"].as_f64().unwrap_or(1.0) as f32);
        let readings: Vec<_> = wave.chunks_exact(256).map(|c| an.push(c)).collect();
        let t = &br.labels().types;
        let a_rows: Vec<u32> = (0..t.len() as u32).filter(|&i| t[i as usize].starts_with("JO-A")).collect();
        let b_rows: Vec<u32> = (0..t.len() as u32).filter(|&i| t[i as usize].starts_with("JO-B")).collect();
        hum = Some((readings, a_rows, b_rows));
    }
    // predator: the loom measure per 1/60 s frame
    let loom_rates: Option<Vec<f64>> = match p.get("predator") {
        Some(d) => {
            let kind = d["kind"].as_str().unwrap_or("frog");
            let tr = crate::predators::trace(kind, seed, 40.0)?;
            let head = if kind == "dragonfly" {
                bevy::math::Vec3::new(0.0, 1.51, 0.0)
            } else {
                bevy::math::Vec3::new(0.0, crate::predators::THORAX_Y + 0.01, 0.0)
            };
            let mut prev: HashMap<(&str, &str), f32> = HashMap::new();
            let mut out = Vec::new();
            for frame in &tr.frames {
                let (mut seen, mut best) = (HashMap::new(), 0.0f32);
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
            Some(out)
        }
        None => None,
    };
    // imaging
    let mut session = match p.get("imaging") {
        Some(im) => {
            let rois = if im["rois"].as_str().unwrap_or("regions") == "regions" {
                fly_sim::imaging::rois_by_region(&br.labels().region)?
            } else {
                let specs: Vec<String> = im["rois"].as_array().into_iter().flatten().filter_map(|s| s.as_str().map(String::from)).collect();
                fly_sim::imaging::rois_from_specs(&specs, false, &|s| rows_of_spec(&br, &g, s), &|_| false)?
            };
            let ip = fly_sim::imaging::Params {
                indicator: im["indicator"].as_str().unwrap_or("gcamp6s").into(),
                fps: im["fps"].as_f64().unwrap_or(20.0),
                f0_photons: im["f0_photons"].as_f64().unwrap_or(100.0),
                dff_per_spike: im["dff_per_spike"].as_f64().unwrap_or(0.2),
                shot_noise: im["shot_noise"].as_bool().unwrap_or(true),
                seed,
                baseline_tau_s: im["f0_tau_s"].as_f64().unwrap_or(30.0),
                ..Default::default()
            };
            if im["tiff"].as_bool().unwrap_or(false) {
                return Err("imaging.tiff: the TIFF stack is not available in this build".into());
            }
            Some(fly_sim::imaging::Session::new(br.n(), &rois, &ip)?)
        }
        None => None,
    };
    if p["nwb"].as_bool().unwrap_or(false) {
        return Err("NWB export is not built into this version (it needs an HDF5 writer); the CSV and npz hold the same data".into());
    }
    for t in 0..n {
        if cancelled(env) {
            return Err("cancelled".into());
        }
        if let Some(th) = th.as_mut() {
            if t % 10 == 0 {
                th.resolve(&br)?;
                th.update(&mut br, temperature_at(t as f64 * 0.005), 0.05)?;
            }
        }
        if let (Some(wx), Some(a)) = (wx.as_mut(), p.get("weather")) {
            if t % 10 == 0 {
                let (rain, gust, storm) = (
                    a["rain"].as_f64().unwrap_or(0.0) as f32,
                    a["gust_hz"].as_f64().unwrap_or(0.0) as f32,
                    a["storm"].as_bool().unwrap_or(false),
                );
                wx.update(0.05, rain, gust, storm);
                for (region, side, s) in wx.hits(0.05, true) {
                    br.poke(region, side, s);
                }
                let h = wx.humid(0.05);
                if h > 0.0 {
                    br.poke("humid", None, h);
                }
                let (ws, wd) = wx.wind(a["wind_speed"].as_f64().unwrap_or(0.0) as f32, a["wind_dir"].as_f64().unwrap_or(180.0) as f32);
                if ws > 0.0 {
                    let (l, r) = crate::world::wind_drive(0.0, wd, ws);
                    if l > 0.02 {
                        br.poke("wind", Some('L'), l);
                    }
                    if r > 0.02 {
                        br.poke("wind", Some('R'), r);
                    }
                }
                let light = wx.lightning();
                if light > 0.0 {
                    br.poke_recruit("light", Some('L'), light, Some(0.6 * light));
                    br.poke_recruit("light", Some('R'), light, Some(0.6 * light));
                }
            }
        }
        if let (Some((readings, a_rows, b_rows)), Some(a)) = (hum.as_ref(), p.get("audio")) {
            let sec = t as f64 * 0.005 - a["at_s"].as_f64().unwrap_or(0.0);
            if sec >= 0.0 {
                let i = (sec * 22050.0 / 256.0) as usize;
                if i < readings.len() {
                    let r = readings[i];
                    let rows: Vec<u32> = a_rows.iter().chain(b_rows).copied().collect();
                    let vals: Vec<f32> = a_rows.iter().map(|_| r.drive_a).chain(b_rows.iter().map(|_| r.drive_b)).collect();
                    if !rows.is_empty() {
                        br.set_current_values("mic", &rows, &vals);
                    }
                } else if i == readings.len() {
                    br.clear_current("mic");
                }
            }
        }
        if let (Some(rates), Some(a)) = (loom_rates.as_ref(), p.get("predator")) {
            let at = a["at_s"].as_f64().unwrap_or(0.0);
            let f = ((t as f64 * 0.005 - at) * 60.0) as i64;
            let same = t > 0 && f == (((t - 1) as f64 * 0.005 - at) * 60.0) as i64;
            if f >= 0 && (f as usize) < rates.len() && !same {
                let th = crate::reactions::thresholds();
                let strength = ((rates[f as usize] - f64::from(th.loom_min)) / f64::from(th.loom_full)).clamp(0.0, 1.0);
                if strength > 0.0 {
                    br.poke_recruit64("loom", None, strength, Some(0.6 * strength));
                }
            }
        }
        for (k, s) in stims.iter().enumerate() {
            let active = s.start <= t && t < s.stop;
            if s.drive {
                if active && !driving.contains(&k) {
                    br.drive(&s.rows, s.amp);
                    driving.insert(k);
                } else if !active && driving.contains(&k) {
                    br.undrive(&s.rows);
                    driving.remove(&k);
                }
            } else if active && (t - s.start) % 10 == 0 {
                br.poke_recruit64(&s.region, None, s.strength, s.recruit);
            }
        }
        let step = br.steps;
        let spikes = br.step().to_vec();
        rec.push(step, &spikes);
        for (st, label, detail, v) in br.take_stimuli() {
            rec.log_event(st, "stimulus", &label, &detail, v);
        }
        if let Some(sess) = session.as_mut() {
            sess.push(&spikes);
        }
    }
    if let Some(th) = th.as_mut() {
        th.clear(&mut br);
    }
    if hum.is_some() {
        br.clear_current("mic");
    }
    rec.active = false;
    let stem = folder.join(format!("{tag}-seed{seed}"));
    if let Some(sess) = session.as_ref() {
        let res = sess.result()?;
        fly_sim::imaging::export_csv(&res, &folder.join(format!("{tag}-seed{seed}-imaging.csv"))).map_err(|e| e.to_string())?;
    }
    let (steps_arr, idx_arr) = rec.arrays();
    let extra = json!({"protocol": p["name"], "protocol_spec": p, "condition": tag,
                       "surgery": surgery.iter().map(|(k, m)| (k.clone(), json!(m))).collect::<serde_json::Map<_, _>>(),
                       "sim_backend": "rust-cpu", "sim_dtype": "float32", "seed": seed, "n_neurons": br.n(),
                       "spike_sha256": spike_sha256(&steps_arr, &idx_arr, (rec.end_step - rec.start_step).max(1) as i64)});
    let meta = ktf_data::recorder::base_metadata(
        env!("CARGO_PKG_VERSION"),
        &crate::savestate_play::utc_stamp_pub().replace(' ', "T"),
        json!({}),
        extra,
    );
    rec.save(&stem, env.labels, meta).map_err(|e| e.to_string())?;
    let n_steps = (rec.end_step - rec.start_step) as f64;
    let mut out = BTreeMap::new();
    for (name, rows) in &named {
        let members: std::collections::HashSet<i32> =
            rows.iter().filter_map(|r| rec.rows.binary_search(r).ok().map(|i| i as i32)).collect();
        let spikes = idx_arr.iter().filter(|i| members.contains(i)).count();
        out.insert(name.clone(), spikes as f64 / rows.len().max(1) as f64 / (n_steps * 0.005).max(1e-9));
    }
    Ok(out)
}

/// `run`: one run of any kind into `out/<stamp>-<name>/`. `progress(done, total)` returns false to cancel.
pub fn run(
    env: &Env,
    p: &Value,
    out: &Path,
    workers: usize,
    mut progress: impl FnMut(usize, usize, &str) -> bool,
    created: &str,
) -> Result<PathBuf, String> {
    let stamp = crate::savestate_play::utc_stamp_compact();
    let folder = out.join(format!("{stamp}-{}", folder_name(p["name"].as_str().unwrap_or("protocol"))));
    std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    std::fs::write(folder.join("protocol.json"), serde_json::to_string_pretty(p).unwrap_or_default()).map_err(|e| e.to_string())?;
    let t0 = std::time::Instant::now();
    let seeds: Vec<u64> = p["seeds"].as_array().into_iter().flatten().filter_map(Value::as_u64).collect();
    let secs = || (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0;
    let summary: Value = if let Some(sc) = p.get("screen") {
        let kind = sc["kind"].as_str().unwrap_or("activation");
        let dir = folder.join("screen");
        let u = |k: &str| sc[k].as_u64().map(|x| x as usize);
        let w = p["workers"].as_u64().map_or(workers, |x| x as usize).max(1);
        let res = if kind == "activation" {
            let o = fly_sim::screens::ActivityOptions {
                seeds: seeds.clone(),
                types: sc["types"].as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()),
                min_neurons: u("min_neurons").unwrap_or(1),
                max_types: u("max_types"),
                controls: u("controls").unwrap_or(2),
                workers: w,
                ..Default::default()
            };
            fly_sim::screens::run_activation(env.weights, env.labels, &dir, &o, created, |d, n, l| progress(d, n, l))?
        } else {
            let o = fly_sim::screens::KnockoutOptions {
                behaviors: sc["behaviors"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                    .unwrap_or_else(|| fly_sim::screens::VALIDATED_PATHWAYS.iter().map(|s| s.to_string()).collect()),
                seeds: seeds.clone(),
                top: u("top").unwrap_or(25),
                batch_size: u("candidate_batch").unwrap_or(0),
                controls: u("controls").unwrap_or(2),
                workers: w,
                types: sc["types"].as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()),
                ..Default::default()
            };
            fly_sim::screens::run_knockout(env.weights, env.labels, &dir, &o, created, |d, n, l| progress(d, n, l))?
        };
        let mut files: Vec<String> =
            std::fs::read_dir(&dir).into_iter().flatten().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
        files.sort();
        let top_rows: Vec<Value> = res["rows"]
            .as_array()
            .into_iter()
            .flatten()
            .take(10)
            .map(|r| {
                Value::Object(
                    r.as_object()
                        .into_iter()
                        .flatten()
                        .filter(|(k, _)| !["lo_", "hi_", "z_", "sign_", "p_", "q_", "eff_", "_"].iter().any(|p| k.starts_with(p)))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                )
            })
            .collect();
        json!({"protocol": p["name"], "seeds": seeds, "screen": sc, "tag": fly_sim::screens::TAG, "criteria": res["criteria"],
               "text": fly_sim::screens::summary(&res, 12), "files": files, "top_rows": top_rows, "seconds": secs()})
    } else if let Some(r) = p.get("rig") {
        let name = r["name"].as_str().unwrap_or("");
        let ind = r["individuality"].as_str().unwrap_or("subtle");
        let mut opt = r.clone();
        if let Some(o) = opt.as_object_mut() {
            o.remove("name");
            o.remove("individuality");
        }
        let c = fly_sim::rigs::Ctx { weights: env.weights, labels: env.labels, cancel: env.cancel };
        let mut per_seed = serde_json::Map::new();
        for (i, &seed) in seeds.iter().enumerate() {
            let mut obs = RigRecorder::default();
            let res = fly_sim::rigs::scene_run_observed(&c, name, seed, ind, &opt, &mut obs)?;
            let stem = folder.join(format!("{name}-seed{seed}"));
            if let Some(mut rec) = obs.rec.take() {
                rec.active = false;
                let meta = ktf_data::recorder::base_metadata(
                    env!("CARGO_PKG_VERSION"),
                    &crate::savestate_play::utc_stamp_pub().replace(' ', "T"),
                    json!({}),
                    json!({"rig": res["rig"], "params": res["params"], "seed": seed, "individuality": ind, "tags": res["tags"],
                           "rule_version": res["rule_version"], "summary": res["summary"]}),
                );
                rec.save(&stem, env.labels, meta).map_err(|e| e.to_string())?;
            }
            let mut csv = String::from(fly_sim::rigs::COLS.join(",").as_str());
            csv.push_str("\r\n");
            for row in res["trace"].as_array().into_iter().flatten() {
                csv.push_str(
                    &row.as_array()
                        .into_iter()
                        .flatten()
                        .map(|v| fly_sim::cards::py_float(v.as_f64().unwrap_or(0.0)))
                        .collect::<Vec<_>>()
                        .join(","),
                );
                csv.push_str("\r\n");
            }
            std::fs::write(stem.with_file_name(format!("{name}-seed{seed}-rig_trace.csv")), csv).map_err(|e| e.to_string())?;
            let mut r2 = res.clone();
            if let Some(o) = r2.as_object_mut() {
                o.remove("trace");
            }
            std::fs::write(
                stem.with_file_name(format!("{name}-seed{seed}-rig_summary.json")),
                serde_json::to_string_pretty(&r2).unwrap_or_default(),
            )
            .map_err(|e| e.to_string())?;
            per_seed.insert(seed.to_string(), res["summary"].clone());
            if !progress(i + 1, seeds.len(), "") {
                return Err("cancelled".into());
            }
        }
        json!({"protocol": p["name"], "seeds": seeds, "rig": r, "title": fly_sim::rigs::rig_title(name),
               "tags": fly_sim::rigs::rig_tags(name).iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
               "rig_summary": per_seed, "seconds": secs()})
    } else if let Some(pa) = p.get("patch") {
        run_patch(env, p, pa, &folder, &seeds, &mut progress, secs())?
    } else if let Some(assay) = p["assay"].as_str() {
        let surgery = surgery_of(&p["surgery"]);
        let opts = p.get("assay_options").cloned().unwrap_or(json!({}));
        let res =
            fly_sim::assay_jobs::run_sync(env.weights, env.labels, assay, &seeds, &opts, &surgery, workers, |d, n| progress(d, n, ""))?;
        export_result(&res, &folder, p).map_err(|e| e.to_string())?;
        json!({"protocol": p["name"], "assay": assay, "seeds": seeds, "seconds": secs(), "treated": res["treated"],
               "control": res.get("control").cloned().unwrap_or(Value::Null), "comparison": res.get("comparison").cloned().unwrap_or(Value::Null)})
    } else {
        let surgery = surgery_of(&p["surgery"]);
        let mut conditions: Vec<(&str, Vec<(String, i8)>)> = vec![(if surgery.is_empty() { "run" } else { "treated" }, surgery.clone())];
        if !surgery.is_empty() && p["control"].as_bool().unwrap_or(true) {
            conditions.push(("control", Vec::new()));
        }
        let mut rates: BTreeMap<&str, BTreeMap<u64, BTreeMap<String, f64>>> = Default::default();
        let (total, mut done) = (conditions.len() * seeds.len(), 0);
        for &seed in &seeds {
            for (tag, sg) in &conditions {
                let r = run_seed(env, p, seed, sg, &folder, tag)?;
                rates.entry(tag).or_default().insert(seed, r);
                done += 1;
                if !progress(done, total, "") {
                    return Err("cancelled".into());
                }
            }
        }
        let groups: Vec<String> = p["recordings"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(i, r)| r["name"].as_str().map_or_else(|| format!("rec{i}"), String::from))
            .collect();
        let col = |tag: &str, g: &str| -> Vec<f64> { seeds.iter().map(|s| rates[tag][s][g]).collect() };
        let mut mean_rate = serde_json::Map::new();
        for (tag, _) in &conditions {
            mean_rate.insert(
                (*tag).into(),
                Value::Object(groups.iter().map(|g| (g.clone(), mean_ci_json(fly_sim::stats::mean_ci(&col(tag, g))))).collect()),
            );
        }
        let mut s = json!({"protocol": p["name"], "seeds": seeds, "seconds": secs(),
                           "surgery": if surgery.is_empty() { Value::Null } else { Value::Object(surgery.iter().map(|(k, m)| (k.clone(), json!(m))).collect()) },
                           "mean_rate_hz": mean_rate});
        if conditions.len() == 2 {
            s["comparison"] = Value::Object(
                groups.iter().map(|g| (g.clone(), fly_sim::assay_jobs::paired(&col("treated", g), &col("control", g)))).collect(),
            );
        }
        s
    };
    std::fs::write(folder.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap_or_default()).map_err(|e| e.to_string())?;
    Ok(folder)
}

fn run_patch(
    env: &Env,
    p: &Value,
    pa: &Value,
    folder: &Path,
    seeds: &[u64],
    progress: &mut dyn FnMut(usize, usize, &str) -> bool,
    secs: f64,
) -> Result<Value, String> {
    use fly_sim::patchclamp as pc;
    let amps: Vec<f64> =
        pa["amplitudes"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_else(|| vec![0.0, 0.05, 0.1]);
    let lif = lif_of(p);
    let build = Build { weights: env.weights, labels: env.labels, mode: "off", sigma: None, lif, wired: None };
    let mut per_seed: BTreeMap<u64, Vec<(f64, f64)>> = BTreeMap::new();
    let mut rheo = serde_json::Map::new();
    let mut neuron = Value::Null;
    for (i, &seed) in seeds.iter().enumerate() {
        let mut br = build.brain_with(seed, None, 600, true);
        validation::rest(&mut br, (pa["warmup_s"].as_f64().unwrap_or(1.0) / 0.005).round_ties_even() as usize);
        let row = pc::pick_neuron(&br, pa["neuron"].as_str().unwrap_or(""), pa["index"].as_u64().unwrap_or(0) as usize)?;
        let curve = pc::if_curve(
            &mut br,
            Some(row),
            &amps,
            pa["duration_ms"].as_f64().unwrap_or(500.0),
            pa["repeats"].as_u64().unwrap_or(3) as usize,
            pa["mode"].as_str().unwrap_or("embedded"),
            seed,
            &fly_sim::cards::lif_params(&lif),
        )?;
        neuron = pc::describe_neuron(&br, row as usize);
        let rec = &curve["recording"];
        let rec_obj = pc::Recording {
            v: rec["v"].as_array().into_iter().flatten().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect(),
            spikes: rec["spikes"].as_array().into_iter().flatten().map(|x| x.as_bool().unwrap_or(false)).collect(),
            current: rec["current"].as_array().into_iter().flatten().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect(),
            mode: curve["mode"].as_str().unwrap_or("").to_string(),
            row: Some(row),
            meta: json!({"neuron": neuron}),
        };
        pc::export_csv(&rec_obj, &folder.join(format!("patch-seed{seed}-trace.csv"))).map_err(|e| e.to_string())?;
        pc::export_if_csv(&curve, &folder.join(format!("patch-seed{seed}-if.csv"))).map_err(|e| e.to_string())?;
        let a: Vec<f64> = curve["amplitudes"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
        let r: Vec<f64> = curve["rate_hz"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
        per_seed.insert(seed, a.into_iter().zip(r).collect());
        rheo.insert(seed.to_string(), curve["rheobase"].clone());
        if !progress(i + 1, seeds.len(), "") {
            return Err("cancelled".into());
        }
    }
    let mut uniq: Vec<f64> = amps.clone();
    uniq.sort_by(f64::total_cmp);
    uniq.dedup();
    let mut by_amp = serde_json::Map::new();
    for a in uniq {
        let vals: Vec<f64> = seeds.iter().filter_map(|s| per_seed[s].iter().find(|x| x.0 == a).map(|x| x.1)).collect();
        by_amp.insert(fly_sim::sensitivity::fmt_gp(a, 6), mean_ci_json(fly_sim::stats::mean_ci(&vals)));
    }
    Ok(json!({"protocol": p["name"], "seeds": seeds, "patch": pa, "tag": pc::TAG_TEXT, "units": pc::UNITS_NOTE, "neuron": neuron,
              "rate_hz_by_current": by_amp, "rheobase_by_seed": rheo, "seconds": secs}))
}

/// `recorder.export_result`: result.json, per_fly.csv and metadata.json of an assay.
fn export_result(res: &Value, folder: &Path, p: &Value) -> std::io::Result<()> {
    std::fs::create_dir_all(folder)?;
    std::fs::write(folder.join("result.json"), serde_json::to_string_pretty(res).unwrap_or_default())?;
    let mut csv = String::from("group,seed,score\r\n");
    for (group, key) in [("treated", "treated"), ("control", "control")] {
        if res.get(key).is_some_and(Value::is_object) {
            for (s, v) in res["seeds"].as_array().into_iter().flatten().zip(res[key]["per_fly"].as_array().into_iter().flatten()) {
                csv.push_str(&format!("{group},{s},{}\r\n", v.as_f64().map_or_else(|| v.to_string(), fly_sim::cards::py_float)));
            }
        }
    }
    std::fs::write(folder.join("per_fly.csv"), csv)?;
    let meta = ktf_data::recorder::base_metadata(
        env!("CARGO_PKG_VERSION"),
        &crate::savestate_play::utc_stamp_pub().replace(' ', "T"),
        json!({}),
        json!({"assay": res["kind"], "seeds": res["seeds"], "surgery": res.get("surgery"), "assay_options": res.get("options"), "assay_params": p.get("params")}),
    );
    std::fs::write(folder.join("metadata.json"), serde_json::to_string_pretty(&meta).unwrap_or_default())
}
