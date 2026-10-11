//! The Lab tool pages that run a job of their own (`lab/labday4.py` and the toolkit): one resource keeps their jobs and results, pages read
//! them through `GameHost` (`query("lab.<name>")`, commands named `lab.<verb>`).
//!
//! Python runs these in a `BgJob` thread with a progress fraction, a note and a cancel flag; here the job is a thread too and the result a
//! `serde_json::Value` shaped like the Python dict, so the page code is a line-by-line port.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Instant;

use bevy::prelude::*;
use serde_json::{Value, json};

use crate::sim_thread::{PackPath, Shared};

struct Prog {
    frac: f64,
    note: String,
    result: Option<Result<Value, String>>,
}

pub struct BgJob {
    label: String,
    prog: Arc<Mutex<Prog>>,
    cancel: Arc<AtomicBool>,
    t0: Instant,
}

impl BgJob {
    fn start(label: &str, work: impl FnOnce(&dyn Fn(f64, &str), &AtomicBool) -> Result<Value, String> + Send + 'static) -> Self {
        let prog = Arc::new(Mutex::new(Prog { frac: 0.0, note: String::new(), result: None }));
        let cancel = Arc::new(AtomicBool::new(false));
        let (p, c) = (prog.clone(), cancel.clone());
        thread::Builder::new()
            .name(format!("bg-{label}"))
            .spawn(move || {
                let upd = |frac: f64, note: &str| {
                    let mut g = p.lock().unwrap_or_else(PoisonError::into_inner);
                    g.frac = frac.clamp(0.0, 1.0);
                    g.note = note.to_string();
                };
                let r = work(&upd, &c);
                p.lock().unwrap_or_else(PoisonError::into_inner).result = Some(r);
            })
            .ok();
        Self { label: label.into(), prog, cancel, t0: Instant::now() }
    }

    fn json(&self) -> Value {
        let g = self.prog.lock().unwrap_or_else(PoisonError::into_inner);
        json!({"label": self.label, "frac": g.frac, "note": g.note, "elapsed": self.t0.elapsed().as_secs_f64()})
    }
}

/// Lab > Patch clamp: what the page chose and the recordings it keeps (`st.p_*`).
struct PatchPage {
    spec: String,
    index: i64,
    mode: u64,
    amp: f64,
    dur: i64,
    repeats: i64,
    hold: f64,
    live: bool,
    note: String,
    rec: Option<fly_sim::patchclamp::Recording>,
    curve: Option<Value>,
    awaiting_run: bool,
}

impl Default for PatchPage {
    fn default() -> Self {
        Self {
            spec: "DNp01".into(),
            index: 0,
            mode: 0,
            amp: 0.1,
            dur: 500,
            repeats: 1,
            hold: 0.0,
            live: false,
            note: String::new(),
            rec: None,
            curve: None,
            awaiting_run: false,
        }
    }
}

impl PatchPage {
    fn view() -> crate::sim_thread::PatchView {
        crate::sim_thread::PATCH_VIEW.lock().unwrap_or_else(PoisonError::into_inner).clone().unwrap_or_default()
    }

    fn proto(&self) -> fly_sim::patchclamp::ClampProtocol {
        fly_sim::patchclamp::ClampProtocol {
            amplitudes: vec![self.amp],
            duration_ms: self.dur as f64,
            repeats: self.repeats.max(0) as usize,
            holding: self.hold,
            ..Default::default()
        }
    }
}

/// Lab > Imaging: what the page chose (`livelab.ImagingLive`'s settings; the session itself runs on the sim thread).
struct ImagingPage {
    init: bool,
    on: bool,
    indicator: String,
    fps: f64,
    f0: f64,
    noise: bool,
    roi_mode: String,
    keep: bool,
    tau: f64,
    reduced: bool,
    msg: String,
}

impl Default for ImagingPage {
    fn default() -> Self {
        Self {
            init: false,
            on: false,
            indicator: "gcamp6s".into(),
            fps: 20.0,
            f0: 100.0,
            noise: true,
            roi_mode: "regions".into(),
            keep: false,
            tau: 30.0,
            reduced: false,
            msg: String::new(),
        }
    }
}

impl ImagingPage {
    fn cfg(&self) -> Box<crate::sim_thread::ImagingCfg> {
        Box::new(crate::sim_thread::ImagingCfg {
            indicator: self.indicator.clone(),
            fps: self.fps,
            f0_photons: self.f0,
            shot_noise: self.noise,
            roi_mode: self.roi_mode.clone(),
            gain: 1.0,
            baseline_tau_s: self.tau,
            reduced: self.reduced,
        })
    }

    /// (Re)start the session with the page's settings, or stop it.
    fn send(&self, link: &crate::app::SimLink) {
        link.0.send(crate::sim_thread::Command::Imaging(self.on.then(|| self.cfg())));
    }
}

/// The queue the pages' commands land in, and the state they read.
#[derive(Resource)]
pub struct LabTools {
    pub queue: Arc<Mutex<Vec<(String, Value)>>>,
    job: Option<BgJob>,
    error: String,
    msg: String,
    net: BTreeMap<String, Value>,
    sleep: Option<Value>,
    sens: Option<Value>,
    /// The loaded pack's file (set each frame from `PackPath`), for the cache lookups the pages show.
    pack: PackPath,
    /// Which brains have a valid cached analysis: looked up once when the page opens (hashing the pack), not every frame.
    cached_map: Option<Value>,
    /// Lab > Connectome robustness: the live wiring, what the sliders ask about, the two analyses.
    counts: Option<Arc<Vec<i32>>>,
    live: fly_sim::wiring::Wiring,
    wire_stats: Value,
    sweep: Option<Value>,
    flips: Option<Value>,
    /// Critical path results by target; the types silenced on the live flies (from the snapshot).
    critical: BTreeMap<String, Value>,
    type_ops: Vec<String>,
    clamp: Option<Value>,
    diff: Option<Value>,
    laser: Value,
    psych: Option<Value>,
    autopsy_done: BTreeMap<String, Value>,
    autopsy_fails: Option<Value>,
    autopsy_error: String,
    gen_query: String,
    gen_results: Vec<Value>,
    gen_selected: Option<String>,
    gen_note: String,
    gen_coverage: Option<Value>,
    gen_describe: Value,
    thermo_cfg: (Vec<(String, String, f64)>, bool, bool, f64),
    thermo_note: String,
    pharm: (String, f64, bool, f64),
    pharm_aff: Option<((String, i64, bool, i64), Value)>,
    pharm_dr: Option<Value>,
    pharm_report: Option<Value>,
    pharm_error: String,
    thermo_t: String,
    thermo_eff: usize,
    all_type_ops: Vec<(String, i8)>,
    class_status: (u64, String),
    class_groups: Option<std::collections::HashMap<&'static str, Vec<u32>>>,
    psych_msg: String,
    patch: PatchPage,
    imaging: ImagingPage,
    /// Lab > Mini-papers: the results by paper.
    papers: BTreeMap<String, Value>,
    base_seed: i64,
}

impl Default for LabTools {
    fn default() -> Self {
        Self {
            queue: Default::default(),
            job: None,
            error: String::new(),
            msg: String::new(),
            net: BTreeMap::new(),
            sleep: None,
            sens: None,
            pack: PackPath(None),
            cached_map: None,
            counts: None,
            live: Default::default(),
            wire_stats: Value::Null,
            sweep: None,
            flips: None,
            critical: BTreeMap::new(),
            type_ops: Vec::new(),
            clamp: None,
            diff: None,
            laser: Value::Null,
            psych: None,
            autopsy_done: BTreeMap::new(),
            autopsy_fails: None,
            autopsy_error: String::new(),
            gen_query: "DNp01".into(),
            gen_results: Vec::new(),
            gen_selected: None,
            gen_note: String::new(),
            gen_coverage: None,
            gen_describe: Value::Null,
            thermo_cfg: (Vec::new(), false, false, 22.0),
            thermo_note: String::new(),
            pharm: ("picrotoxin".into(), 0.5, true, 0.7),
            pharm_aff: None,
            pharm_dr: None,
            pharm_report: None,
            pharm_error: String::new(),
            thermo_t: "type:DNp01".into(),
            thermo_eff: 0,
            all_type_ops: Vec::new(),
            class_status: (0, String::new()),
            class_groups: None,
            psych_msg: String::new(),
            patch: PatchPage::default(),
            imaging: ImagingPage::default(),
            papers: BTreeMap::new(),
            base_seed: 1000,
        }
    }
}

fn exports_dir() -> std::path::PathBuf {
    ktf_data::paths::get().data_dir.join("exports")
}

fn cache_dir() -> std::path::PathBuf {
    ktf_data::paths::get().data_dir.join("cache")
}

fn netsci_cache_path(brain: &str, sha: &str, settings: &Value) -> std::path::PathBuf {
    // Python: sha256(json.dumps(settings, sort_keys=True))[:10]
    let s = format!(
        "{{\"nulls\": {}, \"rich_percentiles\": [{}], \"seed\": {}, \"sweeps_per_level\": {}, \"version\": {}, \"wedges\": {}}}",
        settings["nulls"],
        settings["rich_percentiles"].as_array().map(|a| a.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")).unwrap_or_default(),
        settings["seed"],
        settings["sweeps_per_level"],
        settings["version"],
        settings["wedges"]
    );
    let h = fly_sim::util::sha256_hex(s.as_bytes());
    cache_dir().join(format!("netsci_{brain}_{}_{}.json", &sha[..12.min(sha.len())], &h[..10]))
}

fn netsci_load_cached(path: &std::path::Path, sha: &str) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    let w: Value = serde_json::from_str(&text).ok()?;
    let payload = w.get("result")?;
    let digest = fly_sim::util::sha256_hex(serde_json::to_string(payload).ok()?.as_bytes());
    (w["sha256"].as_str()? == digest && payload["pack_sha256"].as_str()? == sha).then(|| payload.clone())
}

fn pack_for(brain: &str, pack: &PackPath) -> Result<std::path::PathBuf, String> {
    if brain == "adult" {
        if let Some(p) = pack.0.clone().filter(|p| p.exists()) {
            return Ok(p);
        }
        if let Some(p) = fly_sim::pack::find_default() {
            return Ok(p);
        }
    }
    Err(format!("FileNotFoundError: brain pack for '{brain}' not found (run 'python -m kickthefly.sim.brainpack build')"))
}

impl LabTools {
    fn start_net(&mut self, brain: &str, force: bool, pack: &PackPath) {
        self.error.clear();
        let path = match pack_for(brain, pack) {
            Ok(p) => p,
            Err(e) => {
                self.error = e;
                return;
            }
        };
        let brain = brain.to_string();
        self.job = Some(BgJob::start("Network science", move |upd, cancel| {
            let (nulls, wedges, seed) = (fly_sim::netsci::DEFAULT_NULLS, fly_sim::netsci::DEFAULT_WEDGES, fly_sim::netsci::DEFAULT_SEED);
            let settings = fly_sim::netsci::settings(nulls, wedges, seed);
            let sha = fly_sim::cards::pack_sha256(&path).ok_or("could not read the pack")?;
            let cache = netsci_cache_path(&brain, &sha, &settings);
            if let (false, Some(mut hit)) = (force, netsci_load_cached(&cache, &sha)) {
                upd(1.0, "loaded from the cache");
                hit["from_cache"] = json!(true);
                return Ok(hit);
            }
            upd(0.0, "reading the pack");
            let g = fly_sim::netsci::load_edges(&path).map_err(|e| e.to_string())?;
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let created = crate::savestate_play::utc_stamp_pub();
            let res = fly_sim::netsci::analyse(&g, &brain, &name, &sha, &created, (nulls, wedges, seed), cancel, &mut |f, l| upd(f, l))
                .map_err(|e| e.to_string())?;
            let wrapper =
                json!({"sha256": fly_sim::util::sha256_hex(serde_json::to_string(&res).unwrap_or_default().as_bytes()), "result": res});
            let _ = std::fs::create_dir_all(cache_dir());
            let _ = std::fs::write(&cache, serde_json::to_string(&wrapper).unwrap_or_default());
            Ok(res)
        }));
    }

    fn cached(&self, pack: &PackPath) -> Value {
        let mut m = serde_json::Map::new();
        for brain in ["adult", "larva"] {
            let hit = pack_for(brain, pack).ok().is_some_and(|path| {
                let settings = fly_sim::netsci::settings(
                    fly_sim::netsci::DEFAULT_NULLS,
                    fly_sim::netsci::DEFAULT_WEDGES,
                    fly_sim::netsci::DEFAULT_SEED,
                );
                fly_sim::cards::pack_sha256(&path)
                    .is_some_and(|sha| netsci_load_cached(&netsci_cache_path(brain, &sha, &settings), &sha).is_some())
            });
            m.insert(brain.into(), json!(hit));
        }
        Value::Object(m)
    }

    fn start_sleep(&mut self, seeds: usize, shared: Shared, thresh_sleep: f64) {
        self.error.clear();
        let seeds: Vec<u64> = (0..seeds as u64).map(|i| 1000 + i).collect();
        self.job = Some(BgJob::start("Sleep deprivation", move |upd, cancel| {
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 2);
            let created = crate::savestate_play::utc_stamp_pub();
            fly_sim::sleepdep::run(&shared.weights, &shared.labels, &seeds, "off", workers, thresh_sleep, &created, |d, n, label| {
                upd(d as f64 / n.max(1) as f64, label);
                !cancel.load(Ordering::Relaxed)
            })
        }));
    }

    fn start_sens(&mut self, smoke: bool, shared: Shared, pack: &PackPath) {
        self.error.clear();
        let path = pack.0.clone().filter(|p| p.exists()).or_else(fly_sim::pack::find_default);
        let folder = exports_dir().join(if smoke { "sensitivity-smoke" } else { "sensitivity" });
        self.job = Some(BgJob::start("Sensitivity", move |upd, cancel| {
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 2);
            let counts = match &path {
                Some(p) => Some(fly_sim::wiring::synapse_counts(p)?),
                None => None,
            };
            let created = crate::savestate_play::utc_stamp_pub();
            let seeds = fly_sim::validation::seeds();
            let res = if smoke {
                fly_sim::sensitivity::run(
                    &shared.weights,
                    &shared.labels,
                    counts.as_deref(),
                    Some(&["noise_std"]),
                    Some(&["looming_escape", "sugar_feeding"]),
                    Some(&seeds[..3]),
                    Some(&[("noise_std", vec![0.0375, 0.075])]),
                    workers,
                    Some(&folder),
                    true,
                    &created,
                    |d, n, label| {
                        upd(d as f64 / n.max(1) as f64, label);
                        !cancel.load(Ordering::Relaxed)
                    },
                )?
            } else {
                fly_sim::sensitivity::run(
                    &shared.weights,
                    &shared.labels,
                    counts.as_deref(),
                    None,
                    None,
                    None,
                    None,
                    workers,
                    Some(&folder),
                    true,
                    &created,
                    |d, n, label| {
                        upd(d as f64 / n.max(1) as f64, label);
                        !cancel.load(Ordering::Relaxed)
                    },
                )?
            };
            let _ = fly_sim::sensitivity::save(&res, &folder);
            Ok(res)
        }));
    }

    /// Take a finished job's result to ITS page (the three pages share one job slot).
    fn collect(&mut self) {
        let Some(job) = &self.job else { return };
        let Some(res) = job.prog.lock().unwrap_or_else(PoisonError::into_inner).result.take() else { return };
        let label = job.label.clone();
        self.job = None;
        match res {
            Err(e) if e == "cancelled" => {}
            Err(e) => self.error = e,
            Ok(v) => match label.as_str() {
                "Network science" => {
                    if let Some(b) = v["brain"].as_str() {
                        self.net.insert(b.to_string(), v.clone());
                    }
                    self.cached_map = None;
                }
                "Sleep deprivation" => self.sleep = Some(v),
                "Sensitivity" => self.sens = Some(v),
                "Threshold sweep" => self.sweep = Some(v),
                "Sign flips" => self.flips = Some(v),
                "Neural clamp" => self.clamp = Some(v),
                "Connectome diff" => self.diff = Some(v),
                "Psych sweep" => self.psych = Some(v),
                "Patch I-F curve" => self.patch.curve = Some(v),
                l if l.starts_with("Mini-paper ") => {
                    if let Some(pid) = v["paper"].as_str() {
                        self.papers.insert(pid.to_string(), v.clone());
                    }
                }
                "Pharm dose-response" => self.pharm_dr = Some(v),
                "Inhibition report" => self.pharm_report = Some(v),
                "Failure autopsy" => {
                    for a in v.as_array().into_iter().flatten() {
                        if let Some(id) = a["test"]["id"].as_str() {
                            self.autopsy_done.insert(id.to_string(), a.clone());
                        }
                    }
                }
                "Critical path" => {
                    if let Some(t) = v["target"].as_str() {
                        self.critical.insert(t.to_string(), v.clone());
                    }
                }
                _ => {}
            },
        }
    }

    /// The state a screen's page asks for (`lab.day4_state`), or nothing when it is not one of ours.
    pub fn state_for(&self, screen: &str) -> Value {
        match screen {
            "lab_netsci" | "lab_sleepdep" | "lab_sensitivity" => json!({
                "lab.day4_state": {
                    "job": self.job.as_ref().map(BgJob::json),
                    "error": self.error,
                    "msg": self.msg,
                    "net": self.net,
                    "cached": self.cached_map.clone().unwrap_or(Value::Null),
                    "sleep": self.sleep,
                    "sens": self.sens,
                }
            }),
            "lab_laser" => json!({"lab.laser_state": self.laser}),
            "lab_minipapers" => {
                let job = self.job.as_ref().filter(|j| j.label.starts_with("Mini-paper ")).map(BgJob::json);
                json!({"lab.minipapers_state": {
                    "results": self.papers, "job": job, "error": self.error, "adult": self.pack.0.is_some(),
                }})
            }
            "lab_imaging" => {
                let i = &self.imaging;
                let v = crate::sim_thread::IMAGING_VIEW.lock().unwrap_or_else(PoisonError::into_inner).clone();
                let ind = fly_sim::imaging::indicator(&i.indicator).unwrap_or(&fly_sim::imaging::INDICATORS[0]);
                json!({"lab.imaging_state": {
                    "on": i.on && v.on, "indicator": i.indicator, "fps": i.fps, "f0": i.f0, "noise": i.noise, "roi_mode": i.roi_mode,
                    "keep": i.keep, "reduced": i.reduced, "error": v.error, "frames": false,
                    "indicators": fly_sim::imaging::INDICATORS.iter().map(|d| json!({"key": d.key, "name": d.name})).collect::<Vec<_>>(),
                    "source": ind.source, "verified": ind.verified,
                    "msg": if i.msg.is_empty() { v.exported.clone() } else { i.msg.clone() },
                    "nwb": "NWB export is not built into this version (it needs an HDF5 writer); the CSV holds the same data",
                    "view": {"names": v.names, "t": v.t, "dff": v.dff, "true_dff": v.true_dff, "frames": v.frames},
                }})
            }
            "lab_patch" => {
                let p = &self.patch;
                let v = PatchPage::view();
                let trio = |t: &(Vec<f32>, Vec<bool>, Vec<f32>)| json!({"v": t.0, "spikes": t.1, "current": t.2});
                let rec = p.rec.as_ref().map(|r| json!({"v": r.v, "spikes": r.spikes, "current": r.current}));
                let curve = p.curve.as_ref().map(|c| {
                    let mut c = c.clone();
                    if let Some(o) = c.as_object_mut() {
                        o.remove("recording");
                    }
                    c
                });
                let job = self.job.as_ref().filter(|j| j.label == "Patch I-F curve").map(BgJob::json);
                let note = if p.note.is_empty() && v.running { "running on the live brain".to_string() } else { p.note.clone() };
                json!({"lab.patch_state": {
                    "type": p.spec, "index": p.index, "mode": p.mode, "amp": p.amp, "dur": p.dur, "repeats": p.repeats, "hold": p.hold,
                    "live": v.live, "row": v.row, "desc": v.desc, "error": v.error, "note": note,
                    "window": trio(&v.window), "recording": rec, "curve": curve, "job": job,
                    "nwb": "NWB export is not built into this version (it needs an HDF5 writer); the CSV holds the same data",
                }})
            }
            "lab_pharm" => {
                let busy = |l: &str| self.job.as_ref().is_some_and(|j| j.label == l);
                let new = self.pharm_wiring();
                let note = self.job.as_ref().filter(|j| j.label == "Inhibition report").map(|j| j.json()["note"].clone());
                json!({"lab.pharm_state": {
                    "drug": self.pharm.0, "dose": self.pharm.1, "low": self.pharm.2, "cut": self.pharm.3,
                    "affected": self.pharm_aff.as_ref().map(|a| a.1.clone()),
                    "applied": new.as_ref().is_some_and(|n| *n == self.live),
                    "has_nt": !self.live.nt_scales.is_empty(), "live_label": self.live.label(),
                    "busy": if crate::sim_thread::WIRING_BUSY.load(Ordering::Relaxed) { self.live.label() } else { String::new() },
                    "dr": self.pharm_dr, "dr_running": busy("Pharm dose-response"),
                    "report": self.pharm_report, "report_running": busy("Inhibition report"),
                    "report_note": note.unwrap_or(Value::Null),
                    "error": self.pharm_error,
                }})
            }
            "lab_thermo" => {
                let st = crate::sim_thread::THERMO_STATUS.lock().unwrap_or_else(PoisonError::into_inner).clone();
                json!({"lab.thermo_state": {
                    "expressions": self.thermo_cfg.0.iter().map(|(e, t, s)| json!({"effector": e, "target": t, "strength": s})).collect::<Vec<_>>(),
                    "steady": self.thermo_cfg.1, "arena_source": self.thermo_cfg.2, "temperature_c": self.thermo_cfg.3,
                    "status": st.as_ref().map_or(Value::Null, |v| v["rows"].clone()),
                    "arena": st.as_ref().map_or(Value::Null, |v| v["arena"].clone()),
                    "note": if self.thermo_note.is_empty() { st.as_ref().and_then(|v| v["error"].as_str().map(String::from)).unwrap_or_default() } else { self.thermo_note.clone() },
                    "target": self.thermo_t, "effector": self.thermo_eff,
                }})
            }
            "lab_genetics" => {
                let ops: serde_json::Map<String, Value> = self.all_type_ops.iter().map(|(t, m)| (t.clone(), json!(m))).collect();
                json!({"lab.genetics_state": {
                    "error": self.error, "coverage": self.gen_coverage, "query": self.gen_query, "results": self.gen_results,
                    "selected": self.gen_selected, "describe": self.gen_describe, "type_ops": ops, "note": self.gen_note,
                }})
            }
            "lab_autopsy" => json!({"lab.autopsy_state": {
                "fails": self.autopsy_fails, "error": if self.autopsy_fails.as_ref().is_some_and(|f| f.as_array().is_some_and(|a| !a.is_empty())) { self.error.clone() } else { self.autopsy_error.clone() },
                "done": self.autopsy_done, "job": self.job.as_ref().map(BgJob::json),
            }}),
            "lab_classroom" => json!({"lab.class_state": {"seq": self.class_status.0, "status": self.class_status.1}}),
            "lab_psych" => json!({
                "lab.psych_state": {
                    "job": self.job.as_ref().map(BgJob::json),
                    "error": self.error,
                    "result": self.psych,
                    "msg": self.psych_msg,
                    "base_seed": self.base_seed,
                }
            }),
            "lab_diff" => json!({
                "lab.diff_state": {
                    "job": self.job.as_ref().map(BgJob::json),
                    "error": self.error,
                    "result": self.diff,
                }
            }),
            "lab_clamp" => json!({
                "lab.clamp_state": {
                    "job": self.job.as_ref().map(BgJob::json),
                    "error": self.error,
                    "result": self.clamp,
                }
            }),
            "lab_critical" => json!({
                "lab.critical_state": {
                    "job": self.job.as_ref().map(BgJob::json),
                    "error": self.error,
                    "results": self.critical,
                    "type_ops": self.type_ops,
                }
            }),
            "lab_wiring" => json!({
                "lab.wiring_state": {
                    "job": self.job.as_ref().map(BgJob::json),
                    "error": self.error,
                    "msg": self.msg,
                    "busy": if crate::sim_thread::WIRING_BUSY.load(Ordering::Relaxed) { self.live.label() } else { String::new() },
                    "live": {"identity": self.live.is_identity(), "label": self.live.label(), "min_synapses": self.live.min_synapses,
                             "flips": self.live.flip_rows.len(), "inhibition_scale": self.live.inhibition_scale},
                    "stats": self.wire_stats,
                    "sweep": self.sweep,
                    "flips": self.flips,
                }
            }),
            _ => Value::Null,
        }
    }

    fn start_autopsy(&mut self, only: Option<Vec<String>>, results: Value, shared: Shared, pack: &PackPath) {
        self.error.clear();
        let Some(path) = pack.0.clone().filter(|p| p.exists()).or_else(fly_sim::pack::find_default) else {
            self.error = "needs the brain pack".into();
            return;
        };
        self.job = Some(BgJob::start("Failure autopsy", move |upd, _cancel| {
            let f = std::fs::File::open(&path).map_err(|e| e.to_string())?;
            let mut npz = fly_sim::npy::Npz::read(std::io::BufReader::new(f)).map_err(|e| e.to_string())?;
            let inv = npz.take("inv").and_then(|a| a.into_f32()).map_err(|e| e.to_string())?;
            let created = crate::savestate_play::utc_stamp_pub();
            let folder = exports_dir().join("failure-autopsy");
            let out = fly_sim::failure_autopsy::run_all(
                &shared.weights,
                &shared.labels,
                &inv,
                &results,
                &folder,
                only.as_deref(),
                &created,
                &mut |d, n, label| upd(d as f64 / n.max(1) as f64, &format!("{label}  ·  {d}/{n}")),
            )?;
            Ok(Value::Array(out))
        }));
    }

    fn load_autopsy(&mut self, results: &Value) {
        if self.autopsy_fails.is_some() {
            return;
        }
        if results.is_null() {
            self.autopsy_fails = Some(json!([]));
            self.autopsy_error = "No validation result yet: run Lab > Validation (or --validate) first.".into();
            return;
        }
        let fails = fly_sim::failure_autopsy::failing_tests(results);
        let folder = exports_dir().join("failure-autopsy");
        for f in &fails {
            let Some(id) = f["id"].as_str() else { continue };
            if let Some(v) = std::fs::read_to_string(folder.join(format!("{id}.json")))
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t.replace("NaN", "null")).ok())
            {
                self.autopsy_done.insert(id.to_string(), v);
            }
        }
        self.autopsy_fails = Some(Value::Array(fails.iter().map(|f| json!({"id": f["id"], "name": f["name"]})).collect()));
    }

    fn start_psych(&mut self, target: &str, flies: usize, lif: [f64; 5], shared: Shared) {
        self.error.clear();
        self.psych_msg.clear();
        let (target, base) = (target.to_string(), self.base_seed.max(0) as u64);
        self.job = Some(BgJob::start("Psych sweep", move |upd, cancel| {
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 4);
            let created = crate::savestate_play::utc_stamp_pub();
            fly_sim::psychometrics::run_sweep(
                &shared.weights,
                &shared.labels,
                &target,
                None,
                flies,
                base,
                lif,
                workers,
                env!("CARGO_PKG_VERSION"),
                &created,
                |d, n| {
                    upd(d as f64 / n.max(1) as f64, &format!("{d} / {n}"));
                    !cancel.load(Ordering::Relaxed)
                },
            )
        }));
    }

    fn start_diff(&mut self, scenario: &str, modification: &str, shared: Shared, pack: &PackPath) {
        self.error.clear();
        let Some(counts) = self.ensure_counts(pack) else {
            self.error = "needs the brain pack".into();
            return;
        };
        let (scenario, modification) = (scenario.to_string(), modification.to_string());
        self.job = Some(BgJob::start("Connectome diff", move |upd, _cancel| {
            use fly_sim::diffmode::Config;
            use fly_sim::wiring::Wiring;
            upd(0.0, "running connectome diff");
            let (labels, types) = (&shared.labels, &shared.labels.types);
            let rows_of =
                |names: &[&str]| -> Vec<u32> { (0..types.len() as u32).filter(|&i| names.contains(&types[i as usize].as_str())).collect() };
            let a = Config { wiring: Wiring::default(), lesion_rows: None, label: "Fly A (Unperturbed)".into() };
            let mut b = Config { wiring: Wiring::default(), lesion_rows: None, label: format!("Fly B ({modification})") };
            match modification.as_str() {
                "lesion_loom" => b.lesion_rows = Some(rows_of(&["LPLC2", "LC4"])),
                "lesion_gf" => b.lesion_rows = Some(rows_of(&["DNp01"])),
                "hemi_vis" => b.lesion_rows = Some(fly_sim::lesions::hemifield_visual_rows(labels, 'L')),
                "hemi_all" => b.lesion_rows = Some(fly_sim::lesions::hemisphere_rows(labels, 'L')),
                "thresh_10" => b.wiring.min_synapses = 10,
                "flips" => b.wiring.flip_rows = fly_sim::wiring::random_flip(labels, 0.5, 0.5, 1),
                "inhibition_0" => b.wiring.inhibition_scale = 0.0,
                _ => {}
            }
            let created = crate::savestate_play::utc_stamp_pub();
            Ok(fly_sim::diffmode::run_connectome_diff(&shared.weights, &shared.labels, &counts, &a, &b, &scenario, 200, 1000, &created))
        }));
    }

    fn start_clamp(&mut self, scenario: &str, modification: &str, shared: Shared, pack: &PackPath) {
        self.error.clear();
        let Some(counts) = self.ensure_counts(pack) else {
            self.error = "needs the brain pack".into();
            return;
        };
        let (scenario, modification) = (scenario.to_string(), modification.to_string());
        self.job = Some(BgJob::start("Neural clamp", move |upd, _cancel| {
            use fly_sim::wiring::Wiring;
            upd(0.5, "recording reference free-running spike train");
            let reference = fly_sim::clamp::record_reference_run(&shared.weights, &shared.labels, &scenario, 200, 1000);
            let types = &shared.labels.types;
            let rows_of =
                |names: &[&str]| -> Vec<u32> { (0..types.len() as u32).filter(|&i| names.contains(&types[i as usize].as_str())).collect() };
            let (mut wiring, mut lesion) = (Wiring::default(), None);
            match modification.as_str() {
                "lesion_loom" => lesion = Some(rows_of(&["LPLC2", "LC4"])),
                "lesion_gf" => lesion = Some(rows_of(&["DNp01"])),
                "thresh_10" => wiring.min_synapses = 10,
                "flips" => wiring.flip_rows = fly_sim::wiring::random_flip(&shared.labels, 0.5, 0.5, 1),
                "inhibition_0" => wiring.inhibition_scale = 0.0,
                _ => {}
            }
            upd(1.0, "replaying into modified connectome under dynamic clamp");
            let created = crate::savestate_play::utc_stamp_pub();
            Ok(fly_sim::clamp::run_neural_clamp(
                &shared.weights,
                &shared.labels,
                &counts,
                &reference,
                &wiring,
                lesion.as_deref(),
                1000,
                &created,
            ))
        }));
    }

    fn start_critical(&mut self, target: &str, top: usize, shared: Shared) {
        self.error.clear();
        let target = target.to_string();
        let folder = exports_dir().join(format!("critical-path-{target}"));
        self.job = Some(BgJob::start("Critical path", move |upd, cancel| {
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 4);
            let created = crate::savestate_play::utc_stamp_pub();
            let res = fly_sim::criticalpath::run(
                &shared.weights,
                &shared.labels,
                &target,
                None,
                top,
                workers,
                Some(&folder),
                None,
                &created,
                &mut |d, n, label| {
                    upd(d as f64 / n.max(1) as f64, &format!("{label}  ·  {d}/{n} flies"));
                    !cancel.load(Ordering::Relaxed)
                },
            )?;
            let _ = fly_sim::criticalpath::save(&res, &folder);
            Ok(res)
        }));
    }

    /// `ph.wiring_for({drug: dose}, low, cut, base=live)` (the live wiring without drugs at dose 0).
    fn pharm_wiring(&self) -> Option<fly_sim::wiring::Wiring> {
        let (drug, dose, low, cut) = (&self.pharm.0, self.pharm.1, self.pharm.2, self.pharm.3);
        if dose > 0.0 {
            fly_sim::pharmacology::wiring_for(&[(drug.as_str(), dose)], low, cut, &self.live).ok()
        } else {
            Some(fly_sim::wiring::Wiring {
                min_synapses: self.live.min_synapses,
                flip_rows: self.live.flip_rows.clone(),
                inhibition_scale: self.live.inhibition_scale,
                ..Default::default()
            })
        }
    }

    fn send_thermo(&self, link: &crate::app::SimLink) {
        let (e, steady, arena, t) = self.thermo_cfg.clone();
        link.0.send(crate::sim_thread::Command::ThermoConfig { expressions: e, steady, arena_source: arena, temperature_c: t });
    }

    fn ensure_counts(&mut self, pack: &PackPath) -> Option<Arc<Vec<i32>>> {
        if self.counts.is_none() {
            let path = pack.0.clone().filter(|p| p.exists()).or_else(fly_sim::pack::find_default)?;
            self.counts = fly_sim::wiring::synapse_counts(&path).ok().map(Arc::new);
        }
        self.counts.clone()
    }

    fn start_wiring_job(&mut self, kind: &str, a: &Value, shared: Shared, pack: &PackPath) {
        self.error.clear();
        let Some(counts) = self.ensure_counts(pack) else {
            self.error = "needs the brain pack".into();
            return;
        };
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 4);
        let created = crate::savestate_play::utc_stamp_pub();
        match kind {
            "sweep" => {
                self.job = Some(BgJob::start("Threshold sweep", move |upd, cancel| {
                    let c = fly_sim::robustness::Ctx { weights: &shared.weights, labels: &shared.labels, counts: &counts, workers };
                    fly_sim::robustness::threshold_sweep(
                        &c,
                        &fly_sim::robustness::DEFAULT_THRESHOLDS,
                        None,
                        &fly_sim::robustness::DEFAULT_TESTS,
                        &created,
                        &mut |i, n, label| {
                            upd(i as f64 / n.max(1) as f64, label);
                            !cancel.load(Ordering::Relaxed)
                        },
                    )
                }));
            }
            _ => {
                let (cutoff, share, trials) = (
                    a["cutoff"].as_f64().unwrap_or(fly_sim::robustness::DEFAULT_CUTOFF),
                    a["share"].as_f64().unwrap_or(fly_sim::robustness::DEFAULT_SHARE),
                    a["trials"].as_u64().unwrap_or(10) as usize,
                );
                self.job = Some(BgJob::start("Sign flips", move |upd, cancel| {
                    let c = fly_sim::robustness::Ctx { weights: &shared.weights, labels: &shared.labels, counts: &counts, workers };
                    fly_sim::robustness::signflip_trials(
                        &c,
                        cutoff,
                        share,
                        trials,
                        None,
                        &fly_sim::robustness::FLIP_TESTS,
                        &created,
                        &mut |i, n, label| {
                            upd(i as f64 / n.max(1) as f64, label);
                            !cancel.load(Ordering::Relaxed)
                        },
                    )
                }));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn tick(
    mut tools: ResMut<LabTools>,
    link: Res<crate::app::SimLink>,
    pack: Res<PackPath>,
    mut flash: MessageWriter<crate::menu_ui::MenuFlash>,
    screen: Res<crate::menu_ui::MenuScreen>,
    snap: Res<crate::app::LatestSnapshot>,
    mut tool_state: ResMut<crate::tools::ToolState>,
    lab: Res<crate::share_play::LabState>,
    cfg: Res<crate::menu_ui::UserConfig>,
    runs: Res<crate::lab_play::LabRuns>,
) {
    tools.base_seed = cfg.0.get_i64("brain.seed");
    tools.pack = pack.clone();
    if screen.0 == "lab_autopsy" && tools.autopsy_fails.is_none() {
        let results = runs.query()["results"].clone();
        tools.load_autopsy(&results);
    }
    if screen.0 == "lab_laser" {
        let l = &tool_state.laser;
        let laser_idx = crate::tools::TOOLS.iter().position(|t| t.0 == "laser");
        tools.laser = json!({
            "target": l.target_type, "mode": l.mode, "intensity": l.intensity, "trigger": l.trigger_mode,
            "pulse_ms": l.pulse_duration * 1000.0, "current": l.current_value(),
            "has_tool": laser_idx.is_some(), "equipped": laser_idx == Some(tool_state.selected),
        });
    }
    if screen.0 != "lab_patch" && tools.patch.live {
        tools.patch.live = false;
        link.0.send(crate::sim_thread::Command::PatchLive(false));
    }
    if screen.0 == "lab_patch" {
        let v = PatchPage::view();
        tools.patch.live = v.live;
        if tools.patch.awaiting_run && !v.running {
            #[allow(clippy::collapsible_if)]
            if let Some((vv, sp, cur)) = v.recording.clone() {
                let proto = tools.patch.proto();
                let mut meta =
                    json!({"neuron": v.desc, "tag": fly_sim::patchclamp::TAG_TEXT, "units": fly_sim::patchclamp::UNITS_NOTE, "live": true});
                meta["protocol"] = json!({
                    "amplitudes": proto.amplitudes, "duration_ms": proto.duration_ms, "pre_ms": proto.pre_ms, "post_ms": proto.post_ms,
                    "repeats": proto.repeats, "gap_ms": proto.gap_ms, "holding": proto.holding,
                });
                tools.patch.rec =
                    Some(fly_sim::patchclamp::Recording { v: vv, spikes: sp, current: cur, mode: "embedded".into(), row: v.row, meta });
                tools.patch.awaiting_run = false;
                tools.patch.note.clear();
            }
        }
    }
    if !tools.imaging.init {
        tools.imaging.init = true;
        tools.imaging.indicator = cfg.0.get_str("brain.imaging_indicator").to_string();
        tools.imaging.fps = cfg.0.get_i64("brain.imaging_fps") as f64;
        tools.imaging.tau = cfg.0.get_i64("brain.imaging_f0_tau_s") as f64;
    }
    let reduced = cfg.0.get_bool("access.reduced_flashing");
    if reduced != tools.imaging.reduced {
        tools.imaging.reduced = reduced;
        if tools.imaging.on {
            tools.imaging.send(&link);
        }
    }
    if screen.0 == "lab_pharm" {
        let key = (tools.pharm.0.clone(), (tools.pharm.1 * 1000.0).round() as i64, tools.pharm.2, (tools.pharm.3 * 1000.0).round() as i64);
        if tools.pharm_aff.as_ref().map(|a| &a.0) != Some(&key) {
            let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
            if let (Some(counts), Some(sh)) = (tools.ensure_counts(&pack), shared) {
                let (d, dose, low, cut) = tools.pharm.clone();
                match fly_sim::pharmacology::affected(&sh.labels, &sh.weights.csr, &counts, &d, dose, low, cut) {
                    Ok(v) => {
                        tools.pharm_aff = Some((key, v));
                        tools.pharm_error.clear();
                    }
                    Err(e) => tools.pharm_error = e,
                }
            } else {
                tools.pharm_error = "needs the brain pack".into();
            }
        }
    }
    if screen.0 == "lab_genetics" {
        tools.all_type_ops = snap.0.as_deref().map_or_else(Vec::new, |s| s.surgery_types.clone());
        if tools.gen_coverage.is_none() {
            let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
            if let Some(sh) = shared {
                tools.gen_coverage = Some(fly_sim::genetics::coverage(&sh.labels));
                let q = tools.gen_query.clone();
                tools.gen_results = fly_sim::genetics::search(&q, 50)
                    .iter()
                    .map(|l| json!({"name": l.name, "quality": l.quality, "cell_types": l.cell_types}))
                    .collect();
                tools.gen_selected = tools.gen_results.first().and_then(|r| r["name"].as_str().map(String::from));
            }
        }
        let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let stale = tools.gen_describe["line"].as_str() != tools.gen_selected.as_deref();
        if let (Some(sel), Some(sh), true) = (tools.gen_selected.clone(), shared, stale) {
            tools.gen_describe = fly_sim::genetics::describe(&sel, &sh.labels).unwrap_or(Value::Null);
        }
    }
    if screen.0 == "lab_critical" {
        tools.type_ops =
            snap.0.as_deref().map_or_else(Vec::new, |s| s.surgery_types.iter().filter(|t| t.1 == -1).map(|t| t.0.clone()).collect());
    }
    tools.collect();
    if screen.0 == "lab_wiring" && tools.wire_stats.is_null() {
        let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
        if let (Some(counts), Some(shared)) = (tools.ensure_counts(&pack), shared) {
            tools.wire_stats = json!({
                "threshold": fly_sim::wiring::threshold_stats(&shared.weights.csr, &counts, 1),
                "confidence": fly_sim::wiring::confidence_stats(&shared.labels, 0.5),
            });
        }
    }
    if tools.cached_map.is_none() && screen.0 == "lab_netsci" {
        tools.cached_map = Some(tools.cached(&pack));
    }
    let todo: Vec<(String, Value)> = std::mem::take(&mut *tools.queue.lock().unwrap_or_else(PoisonError::into_inner));
    for (cmd, a) in todo {
        let busy = tools.job.is_some();
        match cmd.as_str() {
            "lab.day4_cancel" => {
                if let Some(j) = &tools.job {
                    j.cancel.store(true, Ordering::Relaxed);
                }
            }
            "lab.net_start" if !busy => {
                tools.msg.clear();
                tools.start_net(a["brain"].as_str().unwrap_or("adult"), a["force"].as_bool().unwrap_or(false), &pack);
            }
            "lab.net_export" => {
                let brain = a["brain"].as_str().unwrap_or("adult").to_string();
                if let Some(res) = tools.net.get(&brain).cloned() {
                    let folder = exports_dir().join(format!("netsci-{brain}"));
                    match fly_sim::netsci::export_csv(&res, &folder) {
                        Ok(files) => tools.msg = format!("Wrote {} CSV files to {}", files.len(), folder.display()),
                        Err(e) => tools.error = e.to_string(),
                    }
                }
            }
            "lab.sleep_start" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let n = a["seeds"].as_u64().unwrap_or(5) as usize;
                    tools.start_sleep(n, shared, f64::from(crate::reactions::thresholds().sleep));
                } else {
                    flash.write(crate::menu_ui::MenuFlash("The brain is still loading".into(), false));
                }
            }
            "lab.wiring_want" => {
                if let (Some(counts), Some(shared)) =
                    (tools.ensure_counts(&pack), link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone())
                {
                    let n = a["min_syn"].as_i64().unwrap_or(1);
                    let conf = a["conf"].as_f64().unwrap_or(0.5);
                    tools.wire_stats = json!({
                        "threshold": fly_sim::wiring::threshold_stats(&shared.weights.csr, &counts, n),
                        "confidence": fly_sim::wiring::confidence_stats(&shared.labels, conf),
                    });
                }
            }
            "lab.wiring_set" => {
                if let (Some(counts), Some(shared)) =
                    (tools.ensure_counts(&pack), link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone())
                {
                    let mut w = tools.live.clone();
                    if a["restore"].as_bool().unwrap_or(false) {
                        w = fly_sim::wiring::Wiring::default();
                    }
                    if let Some(n) = a["min_synapses"].as_i64() {
                        w.min_synapses = n;
                    }
                    if let (Some(conf), Some(share)) = (a["flip_conf"].as_f64(), a["flip_share"].as_f64()) {
                        w.flip_rows = fly_sim::wiring::random_flip(&shared.labels, conf, share, 0);
                    }
                    if let Some(s) = a["inhibition_scale"].as_f64() {
                        w.inhibition_scale = s;
                    }
                    tools.live = w.clone();
                    link.0.send(crate::sim_thread::Command::SetWiring { wiring: Box::new(w), counts });
                }
            }
            "lab.wiring_job" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let kind = a["kind"].as_str().unwrap_or("sweep").to_string();
                    tools.start_wiring_job(&kind, &a, shared, &pack);
                }
            }
            "lab.wiring_export" => {
                let which = a["which"].as_str().unwrap_or("sweep");
                let res = if which == "sweep" { tools.sweep.clone() } else { tools.flips.clone() };
                if let Some(res) = res {
                    let kind = res["kind"].as_str().unwrap_or("result").to_string();
                    let folder = exports_dir().join(format!("{}-{kind}", crate::savestate_play::utc_stamp_compact()));
                    match fly_sim::robustness::save(&res, &folder) {
                        Ok(()) => {
                            let name = folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            flash.write(crate::menu_ui::MenuFlash(format!("saved to {name} in exports"), true));
                        }
                        Err(e) => {
                            flash.write(crate::menu_ui::MenuFlash(e.to_string(), false));
                        }
                    }
                }
            }
            "lab.diff_run" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let (sc, md) =
                        (a["scenario"].as_str().unwrap_or("looming").to_string(), a["mod"].as_str().unwrap_or("lesion_loom").to_string());
                    tools.start_diff(&sc, &md, shared, &pack);
                }
            }
            "lab.diff_export" => {
                if let Some(res) = tools.diff.clone() {
                    let folder = exports_dir().join(format!("{}-connectome-diff", crate::savestate_play::utc_stamp_compact()));
                    match fly_sim::diffmode::save(&res, &folder) {
                        Ok(()) => {
                            let name = folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            flash.write(crate::menu_ui::MenuFlash(format!("saved to {name} in exports"), true));
                        }
                        Err(e) => {
                            flash.write(crate::menu_ui::MenuFlash(e.to_string(), false));
                        }
                    }
                }
            }
            "lab.clamp_run" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let (sc, md) =
                        (a["scenario"].as_str().unwrap_or("looming").to_string(), a["mod"].as_str().unwrap_or("lesion_gf").to_string());
                    tools.start_clamp(&sc, &md, shared, &pack);
                }
            }
            "lab.clamp_export" => {
                if let Some(res) = tools.clamp.clone() {
                    let folder = exports_dir().join(format!("{}-neural-clamp", crate::savestate_play::utc_stamp_compact()));
                    match fly_sim::clamp::save(&res, &folder) {
                        Ok(()) => {
                            let name = folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            flash.write(crate::menu_ui::MenuFlash(format!("saved to {name} in exports"), true));
                        }
                        Err(e) => {
                            flash.write(crate::menu_ui::MenuFlash(e.to_string(), false));
                        }
                    }
                }
            }
            "lab.class_action" => {
                let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                if let Some(sh) = shared {
                    use crate::sim_thread::Command;
                    let kind = a["kind"].as_str().unwrap_or("none");
                    let target = a["target"].as_str().unwrap_or("");
                    let status: Option<String> = if kind == "reset_session" {
                        link.0.send(Command::SurgeryClear);
                        None
                    } else if kind == "reset" {
                        link.0.send(Command::SurgeryClear);
                        Some("Reset all overrides to baseline".into())
                    } else if kind == "assay_step" {
                        Some(format!("Assay step executed: {target}"))
                    } else {
                        let groups = tools.class_groups.get_or_insert_with(|| fly_sim::validation::groups(&sh.labels, &sh.weights));
                        let rows: Vec<u32> = if target.is_empty() {
                            Vec::new()
                        } else if let Some(r) = groups.get(target) {
                            r.clone()
                        } else {
                            fly_sim::brain::rows_of(&sh.labels, sh.weights.n(), target).unwrap_or_default()
                        };
                        match kind {
                            "stimulus" if !rows.is_empty() => {
                                let strength = a["strength"].as_f64().unwrap_or(0.8);
                                let n = rows.len();
                                link.0.send(Command::PokeRows { rows, strength: strength as f32 });
                                Some(format!("Stimulus delivered to {n} neurons ({target})"))
                            }
                            "drive" if !rows.is_empty() => {
                                let amp = a["amp"].as_f64().unwrap_or(0.5);
                                let n = rows.len();
                                link.0.send(Command::DriveRows { rows, amp: amp as f32 });
                                Some(format!("Current drive ({amp:+.2}) injected into {n} neurons ({target})"))
                            }
                            // Python writes override 0.0 (silence) or 1.0 without raising the surgery flag, so nothing happens
                            "surgery" if !rows.is_empty() => Some(format!("Surgery: silenced {} neurons ({target})", rows.len())),
                            _ => Some("Action completed".into()),
                        }
                    };
                    if let Some(st) = status {
                        tools.class_status = (tools.class_status.0 + 1, st);
                    }
                }
            }
            "lab.autopsy_run" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let only: Option<Vec<String>> =
                        a["only"].as_array().map(|v| v.iter().filter_map(|x| x.as_str().map(String::from)).collect());
                    let results = runs.query()["results"].clone();
                    tools.start_autopsy(only, results, shared, &pack);
                }
            }
            "lab.autopsy_export" => {
                let src = exports_dir().join("failure-autopsy");
                let dst = exports_dir().join(format!("{}-failure-autopsy", crate::savestate_play::utc_stamp_compact()));
                let ok = std::fs::create_dir_all(&dst).is_ok()
                    && std::fs::read_dir(&src)
                        .is_ok_and(|rd| rd.flatten().all(|e| std::fs::copy(e.path(), dst.join(e.file_name())).is_ok()));
                let name = dst.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                flash.write(crate::menu_ui::MenuFlash(if ok { format!("saved to {name} in exports") } else { "export failed".into() }, ok));
            }
            "lab.gen_search" => {
                let q = a["q"].as_str().unwrap_or("").trim().to_string();
                tools.gen_results = fly_sim::genetics::search(&q, 50)
                    .iter()
                    .map(|l| json!({"name": l.name, "quality": l.quality, "cell_types": l.cell_types}))
                    .collect();
                tools.gen_query = q;
                if let Some(first) = tools.gen_results.first().and_then(|r| r["name"].as_str().map(String::from)) {
                    tools.gen_selected = Some(first);
                }
                tools.gen_note.clear();
            }
            "lab.gen_pick" => tools.gen_selected = a["line"].as_str().map(String::from),
            "lab.gen_surgery" => {
                let mode = a["mode"].as_i64().unwrap_or(0) as i8;
                let types: Vec<String> = tools.gen_describe["matched"].as_object().into_iter().flat_map(|o| o.keys().cloned()).collect();
                let line = a["line"].as_str().unwrap_or("").to_string();
                for t in &types {
                    link.0.send(crate::sim_thread::Command::SurgeryType { ty: t.clone(), mode });
                }
                let what = if mode < 0 {
                    "silenced"
                } else if mode > 0 {
                    "stimulated"
                } else {
                    "cleared"
                };
                tools.gen_note = format!("line {line}: {what}");
                flash.write(crate::menu_ui::MenuFlash(
                    format!(
                        "SURGERY  line {line} ({}): {}",
                        types.join(", "),
                        if mode < 0 {
                            "off"
                        } else if mode > 0 {
                            "on"
                        } else {
                            "normal"
                        }
                    ),
                    true,
                ));
            }
            "lab.gen_laser" => {
                let line = a["line"].as_str().unwrap_or("");
                tool_state.laser.set_target(&format!("line:{line}"));
                tool_state.laser.set_mode("activate");
                if let Some(i) = crate::tools::TOOLS.iter().position(|t| t.0 == "laser") {
                    tool_state.selected = i;
                }
                tools.gen_note = format!("laser aimed at line {line} (activate)");
            }
            "lab.gen_thermo" => {
                let (eff, line) = (a["effector"].as_str().unwrap_or("trpa1").to_string(), a["line"].as_str().unwrap_or("").to_string());
                let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                let target = format!("line:{line}");
                let ok = shared.as_ref().map_or(Err("the brain is still loading".to_string()), |sh| {
                    fly_sim::genetics::rows_of_lines(&sh.labels, &target).map(|_| ())
                });
                match ok {
                    Ok(()) => {
                        tools.thermo_cfg.0.push((eff.clone(), target, 1.0));
                        tools.send_thermo(&link);
                        tools.gen_note = format!("{eff} expressed in line {line}: set the temperature in Lab > Thermogenetics");
                    }
                    Err(e) => tools.gen_note = e,
                }
            }
            "lab.thermo_set" => {
                if let Some(v) = a["steady"].as_bool() {
                    tools.thermo_cfg.1 = v;
                }
                if let Some(v) = a["arena_source"].as_bool() {
                    tools.thermo_cfg.2 = v;
                }
                if let Some(v) = a["temperature_c"].as_f64() {
                    tools.thermo_cfg.3 = v;
                }
                if let Some(v) = a["effector"].as_u64() {
                    tools.thermo_eff = v as usize;
                }
                if let Some(v) = a["target"].as_str() {
                    tools.thermo_t = v.to_string();
                }
                if let Some(i) = a["remove"].as_u64().filter(|&i| (i as usize) < tools.thermo_cfg.0.len()) {
                    tools.thermo_cfg.0.remove(i as usize);
                }
                if a["clear"].as_bool().unwrap_or(false) {
                    tools.thermo_cfg.0.clear();
                }
                if a["add"].as_bool().unwrap_or(false) {
                    let eff = if tools.thermo_eff == 0 { "trpa1" } else { "shibire" };
                    let target = tools.thermo_t.trim().to_string();
                    let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                    let check = match &shared {
                        None => Err("the brain is still loading".to_string()),
                        Some(sh) => {
                            if target.is_empty() {
                                Err("give a neuron target: a cell type, prefix:KC or line:SS00727".to_string())
                            } else if target.to_lowercase().starts_with("line:") {
                                fly_sim::genetics::rows_of_lines(&sh.labels, &target).map(|_| ())
                            } else {
                                fly_sim::brain::rows_of(&sh.labels, sh.weights.n(), &target).map_err(|e| e.to_string()).and_then(|r| {
                                    if r.is_empty() { Err(format!("{target:?} selects no neurons in this brain")) } else { Ok(()) }
                                })
                            }
                        }
                    };
                    match check {
                        Ok(()) => {
                            tools.thermo_cfg.0.push((eff.to_string(), target, 1.0));
                            tools.thermo_note = "added; it applies to every fly on the next frames".into();
                        }
                        Err(e) => tools.thermo_note = e,
                    }
                } else if a.get("note").is_some() {
                    tools.thermo_note.clear();
                }
                tools.send_thermo(&link);
            }
            "lab.thermo_assay" => {
                tools.thermo_note.clear();
            }
            "lab.mp_run" if !busy => {
                let pid = a["paper"].as_str().unwrap_or("").to_string();
                let seeds: Vec<u64> = if a["full"].as_bool().unwrap_or(false) {
                    fly_sim::minipapers::FULL_SEEDS.to_vec()
                } else {
                    fly_sim::minipapers::QUICK_SEEDS.to_vec()
                };
                let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                tools.error.clear();
                match fly_sim::minipapers::paper(&pid) {
                    None => tools.error = format!("unknown mini-paper {pid:?}"),
                    Some(p) => {
                        let (ok, why) = fly_sim::minipapers::available(p, shared.is_some());
                        let created = crate::savestate_play::utc_stamp_pub();
                        if ok {
                            let sh = shared.expect("available means the brain is loaded");
                            let workers = thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 3);
                            tools.job = Some(BgJob::start(&format!("Mini-paper {pid}"), move |upd, cancel| {
                                fly_sim::minipapers::run_paper(
                                    &pid,
                                    &seeds,
                                    Some((&sh.weights, &sh.labels)),
                                    workers,
                                    |d, n, label| {
                                        upd(d as f64 / n.max(1) as f64, if label.is_empty() { "running" } else { label });
                                        !cancel.load(Ordering::Relaxed)
                                    },
                                    &created,
                                )
                            }));
                        } else {
                            // recorded numbers: instant, no brain
                            match fly_sim::minipapers::run_paper(&pid, &fly_sim::minipapers::FULL_SEEDS, None, 1, |_, _, _| true, &created)
                            {
                                Ok(v) => {
                                    tools.papers.insert(pid, v);
                                }
                                Err(e) => tools.error = if e.is_empty() { why } else { e },
                            }
                        }
                    }
                }
            }
            "lab.imaging_set" => {
                let i = &mut tools.imaging;
                let mut restart = false;
                i.msg.clear();
                if let Some(v) = a["indicator"].as_str().filter(|v| fly_sim::imaging::indicator(v).is_ok()) {
                    i.indicator = v.to_string();
                    restart = i.on;
                }
                if let Some(v) = a["fps"].as_f64() {
                    i.fps = v;
                }
                if let Some(v) = a["f0"].as_f64() {
                    i.f0 = v;
                }
                if let Some(v) = a["noise"].as_bool() {
                    i.noise = v;
                }
                if let Some(v) = a["roi_mode"].as_str() {
                    i.roi_mode = v.to_string();
                }
                if let Some(v) = a["keep"].as_bool() {
                    i.keep = v;
                }
                if let Some(v) = a["on"].as_bool() {
                    i.on = v;
                    restart = true;
                }
                if restart {
                    tools.imaging.send(&link);
                }
            }
            "lab.imaging_export" => {
                let folder = exports_dir().join(format!("{}-imaging", crate::savestate_play::utc_stamp_compact()));
                match a["kind"].as_str() {
                    Some("csv") => drop(link.0.send(crate::sim_thread::Command::ImagingExport(folder))),
                    Some("nwb") => tools.imaging.msg = "export failed: NWB export is not available in this version".into(),
                    _ => tools.imaging.msg = "export failed: TIFF export is not available in this version".into(),
                }
            }
            "lab.patch_set" => {
                let p = &mut tools.patch;
                if let Some(v) = a["type"].as_str() {
                    p.spec = v.to_string();
                }
                if let Some(v) = a["index"].as_i64() {
                    p.index = v.max(0);
                }
                if let Some(v) = a["mode"].as_u64() {
                    p.mode = v.min(1);
                }
                if let Some(v) = a["amp"].as_f64() {
                    p.amp = v;
                }
                if let Some(v) = a["dur"].as_i64() {
                    p.dur = v.clamp(10, 20000);
                }
                if let Some(v) = a["repeats"].as_i64() {
                    p.repeats = v.clamp(1, 100);
                }
                if let Some(v) = a["hold"].as_f64() {
                    match fly_sim::patchclamp::check_current(v) {
                        Ok(h) => {
                            p.hold = h;
                            link.0.send(crate::sim_thread::Command::PatchHold(h as f32));
                        }
                        Err(e) => p.note = e,
                    }
                }
                if let Some(on) = a["live"].as_bool() {
                    if on && PatchPage::view().row.is_none() {
                        tools.patch.note = "choose a neuron first".into();
                    } else {
                        link.0.send(crate::sim_thread::Command::PatchHold(tools.patch.hold as f32));
                        link.0.send(crate::sim_thread::Command::PatchLive(on));
                        tools.patch.live = on;
                    }
                }
            }
            "lab.patch_pick" => {
                tools.patch.note.clear();
                tools.patch.awaiting_run = false;
                let (spec, index) = (tools.patch.spec.clone(), tools.patch.index.max(0) as usize);
                link.0.send(crate::sim_thread::Command::PatchPick { spec, index });
            }
            "lab.patch_steps" => {
                if tools.patch.live {
                    link.0.send(crate::sim_thread::Command::StepOnce(a["k"].as_u64().unwrap_or(0).min(40) as u32));
                }
            }
            "lab.patch_run" => {
                let v = PatchPage::view();
                let proto = tools.patch.proto();
                if let Err(e) = proto.check() {
                    tools.patch.note = e;
                } else if let Some(row) = v.row {
                    if tools.patch.mode == 1 {
                        let mut lif = fly_sim::cards::LIF_DEFAULTS;
                        for (k, key) in fly_sim::cards::LIF_KEYS.iter().enumerate() {
                            if let Some(x) = lab.values.get(*key) {
                                lif[k] = *x;
                            }
                        }
                        let seed = tools.base_seed.max(0) as u64;
                        match fly_sim::patchclamp::run_isolated(Some(row), &proto, seed, &fly_sim::cards::lif_params(&lif)) {
                            Ok(mut r) => {
                                r.meta["neuron"] = v.desc.clone();
                                tools.patch.rec = Some(r);
                                tools.patch.note = "isolated unit: no synaptic input".into();
                            }
                            Err(e) => tools.patch.note = e,
                        }
                    } else {
                        link.0.send(crate::sim_thread::Command::PatchHold(tools.patch.hold as f32));
                        link.0.send(crate::sim_thread::Command::PatchRun(proto.schedule()));
                        tools.patch.live = true;
                        tools.patch.awaiting_run = true;
                        tools.patch.note = format!("running {:.1} s of steps on the live brain", proto.total_steps() as f64 * 5.0 / 1000.0);
                    }
                }
            }
            "lab.patch_if" if !busy => {
                let v = PatchPage::view();
                let mode = tools.patch.mode;
                let (spec, idx, row) = (tools.patch.spec.clone(), tools.patch.index.max(0) as usize, v.row);
                let seed = tools.base_seed.max(0) as u64;
                let mut lif = fly_sim::cards::LIF_DEFAULTS;
                for (k, key) in fly_sim::cards::LIF_KEYS.iter().enumerate() {
                    if let Some(x) = lab.values.get(*key) {
                        lif[k] = *x;
                    }
                }
                let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                let amps = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
                tools.error.clear();
                tools.job = Some(BgJob::start("Patch I-F curve", move |upd, _c| {
                    if mode == 1 {
                        upd(0.5, "sweeping");
                        fly_sim::patchclamp::if_curve_isolated(row, &amps, 500.0, 3, seed, &fly_sim::cards::lif_params(&lif))
                    } else {
                        let sh = shared.ok_or("the brain is still loading")?;
                        upd(0.1, "building a brain");
                        let build =
                            fly_sim::cards::Build { weights: &sh.weights, labels: &sh.labels, mode: "off", sigma: None, lif, wired: None };
                        let mut br = build.brain(seed, None, 600);
                        let r2 = match row {
                            Some(r) => r,
                            None => fly_sim::patchclamp::pick_neuron(&br, &spec, idx)?,
                        };
                        upd(0.4, "sweeping");
                        fly_sim::patchclamp::if_curve(
                            &mut br,
                            Some(r2),
                            &amps,
                            500.0,
                            3,
                            "embedded",
                            seed,
                            &fly_sim::cards::lif_params(&lif),
                        )
                    }
                }));
            }
            "lab.patch_export" => {
                let folder = exports_dir().join(format!("{}-patch", crate::savestate_play::utc_stamp_compact()));
                let r = if a["kind"].as_str() == Some("csv") {
                    let mut r = Ok(());
                    if let Some(rec) = &tools.patch.rec {
                        r = fly_sim::patchclamp::export_csv(rec, &folder.join("patch_trace.csv"));
                    }
                    if let (Ok(()), Some(c)) = (&r, &tools.patch.curve) {
                        r = fly_sim::patchclamp::export_if_csv(c, &folder.join("patch_if_curve.csv"));
                    }
                    r
                } else {
                    Err(std::io::Error::other("NWB export is not available in this version"))
                };
                match r {
                    Ok(()) => flash.write(crate::menu_ui::MenuFlash(
                        format!("saved to {} in exports", folder.file_name().and_then(|f| f.to_str()).unwrap_or("")),
                        true,
                    )),
                    Err(e) => flash.write(crate::menu_ui::MenuFlash(format!("export failed: {e}"), false)),
                };
            }
            "lab.pharm_set" => {
                if let Some(v) = a["drug"].as_str() {
                    tools.pharm.0 = v.to_string();
                }
                if let Some(v) = a["dose"].as_f64() {
                    tools.pharm.1 = v;
                }
                if let Some(v) = a["low"].as_bool() {
                    tools.pharm.2 = v;
                }
                if let Some(v) = a["cut"].as_f64() {
                    tools.pharm.3 = v;
                }
            }
            "lab.pharm_apply" | "lab.pharm_wash" => {
                let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                if let (Some(counts), Some(_)) = (tools.ensure_counts(&pack), shared) {
                    let new = if cmd == "lab.pharm_wash" {
                        let l = &tools.live;
                        Some(fly_sim::wiring::Wiring {
                            min_synapses: l.min_synapses,
                            flip_rows: l.flip_rows.clone(),
                            inhibition_scale: l.inhibition_scale,
                            ..Default::default()
                        })
                    } else {
                        tools.pharm_wiring()
                    };
                    if let Some(w) = new {
                        tools.live = w.clone();
                        link.0.send(crate::sim_thread::Command::SetWiring { wiring: Box::new(w), counts });
                    }
                }
            }
            "lab.pharm_dr" if !busy => {
                let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                if let (Some(counts), Some(sh)) = (tools.ensure_counts(&pack), shared) {
                    let (d, _, low, cut) = tools.pharm.clone();
                    tools.error.clear();
                    tools.job = Some(BgJob::start("Pharm dose-response", move |upd, _c| {
                        upd(0.5, "measuring");
                        let ctx = fly_sim::robustness::Ctx { weights: &sh.weights, labels: &sh.labels, counts: &counts, workers: 1 };
                        fly_sim::pharmacology::dose_response(&ctx, &d, &[0.0, 0.25, 0.5, 0.75, 1.0], low, cut, &[1000], 200)
                    }));
                }
            }
            "lab.pharm_report" if !busy => {
                let shared = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone();
                if let (Some(counts), Some(sh)) = (tools.ensure_counts(&pack), shared) {
                    let scale = 1.0 - tools.pharm.1;
                    tools.error.clear();
                    tools.job = Some(BgJob::start("Inhibition report", move |upd, _c| {
                        upd(0.5, "measuring before (control x1.00) …");
                        let ctx = fly_sim::robustness::Ctx { weights: &sh.weights, labels: &sh.labels, counts: &counts, workers: 1 };
                        let created = crate::savestate_play::utc_stamp_pub();
                        Ok(fly_sim::robustness::inhibition_report(&ctx, scale, &[1000], 200, &created))
                    }));
                }
            }
            "lab.psych_run" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let mut lif = fly_sim::cards::LIF_DEFAULTS;
                    for (k, key) in fly_sim::cards::LIF_KEYS.iter().enumerate() {
                        if let Some(v) = lab.values.get(*key) {
                            lif[k] = *v;
                        }
                    }
                    let target = a["target"].as_str().unwrap_or("looming").to_string();
                    tools.start_psych(&target, a["flies"].as_u64().unwrap_or(3) as usize, lif, shared);
                }
            }
            "lab.psych_export" => {
                if let (Some(res), Some(t)) = (tools.psych.clone(), a["target"].as_str()) {
                    let folder = exports_dir();
                    let stamp = crate::savestate_play::utc_stamp_compact().replace('-', "_");
                    match fly_sim::psychometrics::save_all_formats(&res, &format!("psych_{t}_{stamp}"), &folder) {
                        Ok(()) => tools.psych_msg = "Exported CSV, SVG & PDF to exports/".into(),
                        Err(e) => tools.error = e.to_string(),
                    }
                }
            }
            "lab.laser_set" => {
                let l = &mut tool_state.laser;
                if let Some(t) = a["target"].as_str() {
                    l.set_target(t);
                }
                if let Some(m) = a["mode"].as_str() {
                    l.set_mode(m);
                }
                if let Some(v) = a["intensity"].as_f64() {
                    l.set_intensity(v);
                }
                if let Some(t) = a["trigger"].as_str() {
                    l.set_trigger_mode(t);
                }
                if let Some(ms) = a["pulse_ms"].as_f64() {
                    l.set_pulse_duration(ms / 1000.0);
                }
            }
            "lab.laser_equip" => {
                if let Some(i) = crate::tools::TOOLS.iter().position(|t| t.0 == "laser") {
                    tool_state.selected = i;
                }
            }
            "lab.cp_run" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let target = a["target"].as_str().unwrap_or("looming_escape").to_string();
                    tools.start_critical(&target, a["top"].as_u64().unwrap_or(15) as usize, shared);
                }
            }
            "lab.cp_apply" => {
                let ty = a["type"].as_str().unwrap_or("").to_string();
                let on = tools.type_ops.contains(&ty);
                link.0.send(crate::sim_thread::Command::SurgeryType { ty: ty.clone(), mode: if on { 0 } else { -1 } });
                flash.write(crate::menu_ui::MenuFlash(format!("SURGERY  {ty}: {}", if on { "normal" } else { "off" }), true));
            }
            "lab.cp_export" => {
                if let Some(res) = a["target"].as_str().and_then(|t| tools.critical.get(t)).cloned() {
                    let folder = exports_dir().join(format!(
                        "{}-critical-path-{}",
                        crate::savestate_play::utc_stamp_compact(),
                        res["target"].as_str().unwrap_or("")
                    ));
                    match fly_sim::criticalpath::save(&res, &folder) {
                        Ok(()) => {
                            let name = folder.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            flash.write(crate::menu_ui::MenuFlash(format!("saved to {name} in exports"), true));
                        }
                        Err(e) => {
                            flash.write(crate::menu_ui::MenuFlash(e.to_string(), false));
                        }
                    }
                }
            }
            "lab.sens_start" if !busy => {
                if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    tools.start_sens(a["smoke"].as_bool().unwrap_or(true), shared, &pack);
                } else {
                    flash.write(crate::menu_ui::MenuFlash("The brain is still loading".into(), false));
                }
            }
            _ => {}
        }
    }
}

pub struct LabToolsPlugin;

impl Plugin for LabToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LabTools>().add_systems(Update, tick);
    }
}
