//! Lab jobs that run on their own threads with the loaded connectome: the validation suite (`lab/validation.py:run`, page `lab_validation`).
//!
//! Results live in `<data>/validation_results.json` (a run on this PC) or `data/validation_results.json` next to the brain pack (the build's
//! bundled run); a run on this PC wins, as in Python (`validation.load_results`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::SystemTime;

use bevy::prelude::*;
use serde_json::{Value, json};

use crate::sim_thread::Shared;

pub const RESULTS_NAME: &str = "validation_results.json";

#[derive(Default)]
struct JobState {
    done: usize,
    total: usize,
    error: Option<String>,
    finished: Option<Value>,
}

struct Job {
    state: Arc<Mutex<JobState>>,
    cancel: Arc<AtomicBool>,
}

struct AssayJob {
    state: Arc<Mutex<JobState>>,
    cancel: Arc<AtomicBool>,
    started: std::time::Instant,
    workers: usize,
    kind: String,
    label: String,
}

/// `labjobs.ASSAYS` and `ASSAY_LABEL`, all eight as the page lists them (the engines are `fly_sim::assay_jobs` and `lab_extra`).
pub const ASSAY_KINDS: [(&str, &str); 8] = [
    ("tmaze", "T-maze conditioning"),
    ("looming", "Looming escape"),
    ("sugar", "Sugar response"),
    ("orchard", "Orchard feeding"),
    ("thermo_escape", "Thermogenetic escape"),
    ("predator_escape", "Predator escape"),
    ("hum_demo", "Hum demo (JO-A/B)"),
    ("sleep_deprivation", "Sleep deprivation"),
];

struct BenchJob {
    status: Arc<Mutex<(String, Option<Value>)>>,
}

#[derive(Resource, Default)]
pub struct LabRuns {
    bench: Option<BenchJob>,
    bench_results: Option<Value>,
    job: Option<Job>,
    assay: Option<AssayJob>,
    assay_result: Option<Value>,
    results: Arc<Value>,
    source: String,
    loaded: Option<(PathBuf, SystemTime)>,
}

fn local_path() -> PathBuf {
    ktf_data::paths::get().data_dir.join(RESULTS_NAME)
}

/// The build's own run: next to the brain pack (the AppImage's `share/kick-the-fly`), else the repository's `docs/results/4.0/` found upward
/// from the working directory (a source checkout).
fn bundled_path() -> Option<PathBuf> {
    if let Some(dir) = fly_sim::pack::find_default().and_then(|p| p.parent().map(PathBuf::from)) {
        let p = dir.join(RESULTS_NAME);
        if p.exists() {
            return Some(p);
        }
    }
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let p = dir.join("docs").join("results").join("4.0").join(RESULTS_NAME);
        if p.exists() {
            return Some(p);
        }
        if !dir.pop() {
            return None;
        }
    }
}

impl LabRuns {
    /// `validation.load_results`: re-read when the file or its choice changed.
    pub fn refresh(&mut self) {
        for (path, source) in [(Some(local_path()), "run on this PC"), (bundled_path(), "from this build")] {
            let Some(path) = path else { continue };
            let Ok(meta) = std::fs::metadata(&path) else { continue };
            let stamp = (path.clone(), meta.modified().unwrap_or(SystemTime::UNIX_EPOCH));
            if self.loaded.as_ref() == Some(&stamp) {
                return;
            }
            if let Ok(v) = std::fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| e.to_string()))
            {
                self.results = Arc::new(v);
                self.source = source.into();
                self.loaded = Some(stamp);
                return;
            }
        }
    }

    /// The validation results in use (`Value::Null` before any were found).
    pub fn results(&self) -> &Value {
        &self.results
    }

    pub fn query(&self) -> Value {
        let job = self.job.as_ref().map(|j| {
            let s = j.state.lock().unwrap_or_else(PoisonError::into_inner);
            json!({"done": s.done, "total": s.total.max(1), "error": s.error})
        });
        json!({"results": &*self.results, "source": self.source, "job": job})
    }

    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    /// The page's `benchmark_state` (see `ktf_ui::pages::lab::LabBenchmark`).
    pub fn bench_load(&mut self) {
        if self.bench_results.is_none() {
            let path = ktf_data::paths::get().state_dir.join(fly_sim::benchmark::FILE);
            self.bench_results = std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok());
        }
    }

    pub fn bench_query(&self) -> Value {
        let status = self.bench.as_ref().map(|b| b.status.lock().unwrap_or_else(PoisonError::into_inner).0.clone());
        json!({"results": self.bench_results, "system": fly_sim::benchmark::system_info(), "backend": "Rust CPU (rayon)", "device": "CPU", "status": status})
    }

    pub fn start_benchmark(&mut self, shared: Shared) {
        if self.bench.is_some() {
            return;
        }
        let status = Arc::new(Mutex::new(("starting".to_string(), None)));
        let st = status.clone();
        thread::Builder::new()
            .name("benchmark".into())
            .spawn(move || {
                let mut res = fly_sim::benchmark::run(&shared.weights, &shared.labels, &[1, 8, 16], 2.5, |d, n, label| {
                    st.lock().unwrap_or_else(PoisonError::into_inner).0 = format!("{label} ({d}/{n})");
                });
                res["timestamp"] = json!(crate::savestate_play::utc_stamp_pub());
                st.lock().unwrap_or_else(PoisonError::into_inner).1 = Some(res);
            })
            .ok();
        self.bench = Some(BenchJob { status });
    }

    /// The page's `assay_state` (see `ktf_ui::pages::lab::LabAssays`).
    pub fn assay_query(&self, base_seed: i64) -> Value {
        let job = self.assay.as_ref().map(|j| {
            let s = j.state.lock().unwrap_or_else(PoisonError::into_inner);
            json!({"done": s.done, "total": s.total.max(1), "elapsed": j.started.elapsed().as_secs_f64(), "workers": j.workers, "error": s.error})
        });
        let options: Vec<Value> = fly_sim::assay_jobs::surgery_options().into_iter().map(|(l, specs)| json!([l, specs])).collect();
        json!({
            "job": job, "result": self.assay_result, "options": options, "base_seed": base_seed,
            "kinds": ASSAY_KINDS.iter().map(|(k, l)| json!([k, l])).collect::<Vec<_>>(),
        })
    }

    pub fn busy_assay(&self) -> bool {
        self.assay.is_some()
    }

    pub fn assay_result(&self) -> Option<&Value> {
        self.assay_result.as_ref()
    }

    pub fn start_assay(&mut self, shared: Shared, args: &Value) {
        if self.assay.as_ref().is_some_and(|j| {
            let s = j.state.lock().unwrap_or_else(PoisonError::into_inner);
            s.finished.is_none() && s.error.is_none()
        }) {
            return;
        }
        let kind = args["kind"].as_str().unwrap_or("tmaze").to_string();
        let flies = args["flies"].as_i64().unwrap_or(6).clamp(2, 30) as u64;
        let base = args["base_seed"].as_i64().unwrap_or(1000);
        let seeds: Vec<u64> = (0..flies).map(|i| (base + 2000) as u64 + i).collect();
        let surgery: Vec<(String, i8)> = args["surgery"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(-1) as i8)).collect())
            .unwrap_or_default();
        let label = args["label"].as_str().unwrap_or("none").to_string();
        let state = Arc::new(Mutex::new(JobState { total: seeds.len() * if surgery.is_empty() { 1 } else { 2 }, ..Default::default() }));
        let cancel = Arc::new(AtomicBool::new(false));
        let workers = std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(1, 4);
        let (st, c, k, sd, sg) = (state.clone(), cancel.clone(), kind.clone(), seeds, surgery);
        thread::Builder::new()
            .name("assay".into())
            .spawn(move || {
                let res = fly_sim::assay_jobs::run_sync(&shared.weights, &shared.labels, &k, &sd, &json!({}), &sg, workers, |d, n| {
                    let mut s = st.lock().unwrap_or_else(PoisonError::into_inner);
                    s.done = d;
                    s.total = n;
                    !c.load(Ordering::Relaxed)
                });
                let mut s = st.lock().unwrap_or_else(PoisonError::into_inner);
                match res {
                    Ok(v) => s.finished = Some(v),
                    Err(e) if e == "cancelled" => s.finished = Some(Value::Null),
                    Err(e) => s.error = Some(e),
                }
            })
            .ok();
        self.assay_result = None;
        self.assay = Some(AssayJob { state, cancel, started: std::time::Instant::now(), workers, kind, label });
    }

    pub fn cancel_assay(&self) {
        if let Some(j) = &self.assay {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// `recorder.export_result`: `result.json` and `per_fly.csv` in a new folder of the exports folder. Returns the folder.
    pub fn export_assay(&self) -> Option<PathBuf> {
        let res = self.assay_result.as_ref()?;
        let kind = res["kind"].as_str().unwrap_or("result");
        let folder = ktf_data::paths::get().data_dir.join("exports").join(format!("{}-{kind}", crate::savestate_play::utc_stamp_compact()));
        std::fs::create_dir_all(&folder).ok()?;
        std::fs::write(folder.join("result.json"), serde_json::to_string_pretty(res).ok()?).ok()?;
        let mut csv = String::from("group,seed,score\n");
        for group in ["treated", "control"] {
            let (Some(per), Some(seeds)) = (res[group]["per_fly"].as_array(), res["seeds"].as_array()) else { continue };
            for (s, v) in seeds.iter().zip(per) {
                csv.push_str(&format!("{group},{s},{}\n", v.as_f64().map_or(String::new(), |x| format!("{x}"))));
            }
        }
        std::fs::write(folder.join("per_fly.csv"), csv).ok()?;
        Some(folder)
    }

    pub fn start_validation(&mut self, shared: Shared) {
        if self.job.is_some() {
            return;
        }
        let state = Arc::new(Mutex::new(JobState { total: 1, ..Default::default() }));
        let cancel = Arc::new(AtomicBool::new(false));
        let (st, c) = (state.clone(), cancel.clone());
        thread::Builder::new()
            .name("validation".into())
            .spawn(move || {
                let t0 = std::time::Instant::now();
                let seeds = fly_sim::validation::seeds();
                let workers = std::thread::available_parallelism().map_or(2, |n| n.get()).clamp(1, 4);
                let res = fly_sim::validation::run(&shared.weights, &shared.labels, &seeds, None, workers, |d, n, _| {
                    let mut s = st.lock().unwrap_or_else(PoisonError::into_inner);
                    s.done = d;
                    s.total = n;
                    !c.load(Ordering::Relaxed)
                });
                let mut s = st.lock().unwrap_or_else(PoisonError::into_inner);
                match res {
                    Ok(mut v) => {
                        v["app_version"] = json!(env!("CARGO_PKG_VERSION"));
                        v["created"] = json!(crate::savestate_play::utc_stamp_pub());
                        v["seconds"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
                        v["backend"] = json!("Rust CPU (rayon)");
                        v["device"] = json!("CPU");
                        v["synapses"] = json!(shared.weights.csr.nnz());
                        v["wiring"] = Value::Null;
                        v["lab_params"] = Value::Object(
                            ktf_data::labparams::params().iter().map(|p| (p.name.clone(), json!(p.default))).collect(),
                        );
                        let th = crate::reactions::Thresholds::DEFAULT;
                        v["thresholds"] = json!({
                            "groom": th.groom, "jump": th.jump, "run": th.run, "kick": th.kick, "walk": th.walk, "back": th.back, "turn": th.turn,
                            "fly": th.fly, "escape": th.escape, "fire": th.fire, "song": th.song, "aggression": th.aggression, "sleep": th.sleep,
                            "co2": th.co2, "hot_pn": th.hot_pn, "cold_pn": th.cold_pn
                        });
                        v["loom"] = json!([th.loom_min, th.loom_full]);
                        s.finished = Some(v);
                    }
                    Err(e) if e == "cancelled" => s.finished = Some(Value::Null),
                    Err(e) => s.error = Some(e),
                }
            })
            .ok();
        self.job = Some(Job { state, cancel });
    }

    /// Take a finished job's result: save it where Python does and show it.
    pub fn poll(&mut self) {
        if let Some(b) = &self.bench {
            let done = b.status.lock().unwrap_or_else(PoisonError::into_inner).1.take();
            if let Some(res) = done {
                let path = ktf_data::paths::get().state_dir.join(fly_sim::benchmark::FILE);
                if let Some(d) = path.parent() {
                    let _ = std::fs::create_dir_all(d);
                }
                let _ = std::fs::write(path, serde_json::to_string_pretty(&res).unwrap_or_default());
                self.bench_results = Some(res);
                self.bench = None;
            }
        }
        if let Some(j) = &self.assay {
            let (finished, errored) = {
                let mut s = j.state.lock().unwrap_or_else(PoisonError::into_inner);
                (s.finished.take(), s.error.is_some())
            };
            if let Some(mut v) = finished {
                if !v.is_null() {
                    v["label"] = json!(ASSAY_KINDS.iter().find(|k| k.0 == j.kind).map_or("", |k| k.1));
                    v["surgery_label"] = json!(j.label);
                    self.assay_result = Some(v);
                }
                self.assay = None;
            } else if errored {
                // the error stays on the page until the next run
            }
        }
        let Some(job) = &self.job else { return };
        let finished = job.state.lock().unwrap_or_else(PoisonError::into_inner).finished.take();
        let Some(res) = finished else { return };
        let errored = job.state.lock().unwrap_or_else(PoisonError::into_inner).error.is_some();
        if errored {
            return; // the page keeps showing the error until the next run
        }
        self.job = None;
        if res.is_null() {
            return;
        }
        let path = local_path();
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(&path, serde_json::to_string_pretty(&res).unwrap_or_default());
        self.loaded = None;
        self.refresh();
    }

    pub fn cancel(&self) {
        if let Some(j) = &self.job {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }
}

fn tick(mut runs: ResMut<LabRuns>, screen: Res<crate::menu_ui::MenuScreen>) {
    if screen.0 == "lab_validation" || screen.0 == "lab" {
        runs.refresh();
    }
    if screen.0 == "lab_benchmark" {
        runs.bench_load();
    }
    runs.poll();
}

pub struct LabPlugin;

impl Plugin for LabPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LabRuns>().add_systems(Update, (tick, sync_mirror));
    }
}

/// `brain.mirror_weights` from the config (the Lab asymmetry page's toggle, `--mirror-weights`) to the sim thread, when it changes.
fn sync_mirror(link: Res<crate::app::SimLink>, cfg: Res<crate::menu_ui::UserConfig>, mut last: Local<Option<bool>>) {
    let on = cfg.0.get_bool("brain.mirror_weights");
    if *last != Some(on) {
        // nothing to undo on the first look when it is off
        if last.is_some() || on {
            link.0.send(crate::sim_thread::Command::SetMirror(on));
        }
        *last = Some(on);
    }
}
