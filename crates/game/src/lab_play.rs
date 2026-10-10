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

#[derive(Resource, Default)]
pub struct LabRuns {
    job: Option<Job>,
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
    runs.poll();
}

pub struct LabPlugin;

impl Plugin for LabPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LabRuns>().add_systems(Update, tick);
    }
}
