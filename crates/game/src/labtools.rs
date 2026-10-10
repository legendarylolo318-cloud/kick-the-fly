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
) {
    tools.pack = pack.clone();
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
