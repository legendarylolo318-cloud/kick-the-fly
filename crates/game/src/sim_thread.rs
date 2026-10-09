//! The brain on a thread of its own, at a fixed 5 ms timestep (200 steps/s), like `Brain._loop` in the Python game.
//!
//! The render thread talks to it with [`Command`]s over an unbounded channel (sending never blocks) and reads [`Snapshot`]s from a
//! [`Mailbox`] (reading never waits for a step). The sim thread paces itself against the wall clock: if it falls more than 40 steps
//! behind it drops the backlog instead of racing, exactly as the Python loop does, and counts what it dropped.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use fly_sim::brain::STIM_AMP;
use fly_sim::csr::Weights;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{Labels, Pack};
use fly_sim::{Brain, synthetic};

use crate::snapshot::{BrainInfo, Mailbox, Phase, Snapshot};

/// Brain steps behind real time that are dropped rather than caught up (Python: `due - done > 40`).
const MAX_BACKLOG: u64 = 40;
/// Calm steps a new brain runs before it is shown (`simcore.new_brain(warmup=600)`).
pub const WARMUP_STEPS: usize = 600;

/// What to simulate.
#[derive(Debug, Clone)]
pub enum Source {
    /// The pack compiled into the program (feature `bundle-pack`).
    Embedded,
    /// The brain pack file (`data/kick_brain.npz`).
    Pack(std::path::PathBuf),
    /// A random network of `n` neurons: for tests and for running without the dataset.
    Synthetic { n: usize },
}

/// Where the loaded brain came from, for things that read the same file again (the Neurodex partner table).
#[derive(bevy::prelude::Resource, Clone, Debug)]
pub struct PackPath(pub Option<std::path::PathBuf>);

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub source: Source,
    pub seed: u64,
    pub warmup: usize,
    /// Publish a snapshot every this many steps (4 = 50 per second, like the Python brain view's history).
    pub publish_every: u64,
}

impl SimConfig {
    pub fn new(source: Source) -> Self {
        Self { source, seed: 1000, warmup: WARMUP_STEPS, publish_every: 4 }
    }
}

/// Messages from the game to the brain.
#[derive(Debug)]
pub enum Command {
    /// Hold the neurons named by a spec (`type:LPLC2,LC4`, `prefix:KC`, ...) driven at `amp` (0 releases them).
    DriveSpec {
        spec: String,
        amp: f32,
    },
    /// Hold these rows driven at `amp`.
    DriveRows {
        rows: Vec<u32>,
        amp: f32,
    },
    /// Real-time multiplier: 1 normal, below 1 slow motion, above 1 faster than real time.
    SetSpeed(f32),
    Pause(bool),
    /// While paused, run this many steps.
    StepOnce(u32),
    Kill,
    /// Bring a dead brain back (R).
    Revive,
    /// `Brain.poke`: a touch/scent/heat... pulse on a region (`head`, `body`, `legs`, `wing`, `scent:sugar`, `heat`, `reward`...).
    Poke {
        region: String,
        side: Option<char>,
        strength: f32,
    },
    /// Brain surgery on the neurons of a spec: -1 silence, 0 release, +1 stimulate.
    Surgery {
        spec: String,
        mode: i8,
    },
    /// 0..1 share of the death inhibition (solvent, cold, venom).
    Sedation(f32),
    /// Pain level 0..2 (normal, more, max).
    PainLevel(usize),
    /// The Neurodex's type table and the types already discovered: from now on the sim thread checks for discoveries every 10 steps.
    DexTable {
        table: Arc<fly_sim::neurodex::TypeTable>,
        known: Vec<String>,
    },
    /// The discovered set changed outside the tracker (a wipe): re-read it.
    DexKnown(Vec<String>),
    /// `brain.neurodex` on/off.
    DexEnabled(bool),
    /// Report the whole brain state (a save): the answer comes as [`SimEvent::State`].
    ExportState,
    /// Put a saved brain state back (a loaded save).
    ImportState(Box<fly_sim::brain::BrainState>),
    Shutdown,
}

/// The render side's handle to the sim thread. Dropping it shuts the thread down and waits for it.
/// The loaded connectome, shared with the arcade's background brains (tournaments, races, lesion battles) so they do not load it again.
#[derive(Clone)]
pub struct Shared {
    pub weights: Arc<Weights>,
    pub labels: Arc<Labels>,
}

/// What the sim thread tells the game that must not be lost (the snapshot mailbox only keeps the newest).
#[derive(Debug, Clone)]
pub enum SimEvent {
    /// A type was discovered (`fly_sim::neurodex::Tracker`).
    Discovered(fly_sim::neurodex::Found),
    /// The answer to [`Command::ExportState`].
    State(Box<fly_sim::brain::BrainState>),
}

pub struct SimHandle {
    events: Mutex<Receiver<SimEvent>>,
    pub shared: Arc<Mutex<Option<Shared>>>,
    tx: Sender<Command>,
    pub mailbox: Arc<Mailbox<Snapshot>>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl SimHandle {
    /// A sender for background threads (the Neurodex table builder hands its table over with it).
    pub fn sender(&self) -> Sender<Command> {
        self.tx.clone()
    }

    /// Send a command. Never blocks; returns false when the thread has already ended.
    pub fn send(&self, cmd: Command) -> bool {
        self.tx.send(cmd).is_ok()
    }

    /// The newest unread snapshot, if one arrived since the last call.
    pub fn latest(&self) -> Option<Box<Snapshot>> {
        self.mailbox.take()
    }

    /// Everything the thread reported since the last call.
    pub fn drain_events(&self) -> Vec<SimEvent> {
        self.events.lock().unwrap_or_else(PoisonError::into_inner).try_iter().collect()
    }

    /// Whether the thread is still alive.
    pub fn is_running(&self) -> bool {
        self.join.as_ref().is_some_and(|j| !j.is_finished())
    }
}

impl Drop for SimHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Command::Shutdown);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Start the sim thread. Returns at once; loading and warming up happen on the thread, reported through the snapshots' phase.
pub fn spawn(cfg: SimConfig) -> SimHandle {
    let (tx, rx) = mpsc::channel();
    let mailbox = Arc::new(Mailbox::<Snapshot>::default());
    let stop = Arc::new(AtomicBool::new(false));
    let shared = Arc::new(Mutex::new(None));
    let (ev_tx, ev_rx) = mpsc::channel();
    let join = {
        let (mailbox, stop, shared) = (mailbox.clone(), stop.clone(), shared.clone());
        thread::Builder::new()
            .name("fly-sim".into())
            .spawn(move || run(cfg, rx, mailbox, stop, shared, ev_tx))
            .expect("could not start the sim thread")
    };
    SimHandle { events: Mutex::new(ev_rx), shared, tx, mailbox, stop, join: Some(join) }
}

fn publish_phase(mailbox: &Mailbox<Snapshot>, phase: Phase, source: &str, neurons: usize) {
    mailbox.publish(Snapshot::default, |s| {
        s.phase = phase;
        s.source = source.to_string();
        s.neurons = neurons;
    });
}

/// The brain pack inside the program, when it was built with `--features bundle-pack`.
pub fn embedded_pack() -> Option<&'static [u8]> {
    #[cfg(feature = "bundle-pack")]
    {
        static PACK: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/kick_brain.npz"));
        Some(PACK)
    }
    #[cfg(not(feature = "bundle-pack"))]
    {
        None
    }
}

fn load(cfg: &SimConfig) -> Result<(Arc<Weights>, Arc<Labels>, String), String> {
    match &cfg.source {
        Source::Embedded => {
            let pack = Pack::load_bytes(embedded_pack().ok_or("this build has no embedded brain pack")?).map_err(|e| e.to_string())?;
            Ok((Arc::new(pack.weights), Arc::new(pack.labels), "embedded pack".to_string()))
        }
        Source::Pack(path) => {
            let pack = Pack::load(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let name = path.file_name().map_or_else(|| "pack".into(), |n| n.to_string_lossy().into_owned());
            Ok((Arc::new(pack.weights), Arc::new(pack.labels), name))
        }
        Source::Synthetic { n } => {
            let w = synthetic::random_weights(*n, 62, 1, Weights::default_blocks());
            let labels = Labels {
                types: (0..*n).map(|i| if i < n / 50 { "LPLC2".to_string() } else { format!("syn{}", i % 50) }).collect(),
                superclass: vec!["synthetic".to_string(); *n],
                ..Labels::default()
            };
            Ok((Arc::new(w), Arc::new(labels), "synthetic".to_string()))
        }
    }
}

fn run(
    cfg: SimConfig,
    rx: Receiver<Command>,
    mailbox: Arc<Mailbox<Snapshot>>,
    stop: Arc<AtomicBool>,
    shared: Arc<Mutex<Option<Shared>>>,
    events: Sender<SimEvent>,
) {
    publish_phase(&mailbox, Phase::Loading, "", 0);
    let (weights, labels, source) = match load(&cfg) {
        Ok(v) => v,
        Err(msg) => {
            publish_phase(&mailbox, Phase::Failed(msg), "", 0);
            // keep the thread alive until told to stop so the game can show the message and still send commands
            while !stop.load(Ordering::Relaxed) {
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(Command::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    _ => {}
                }
            }
            return;
        }
    };
    let n = weights.n();
    *shared.lock().unwrap_or_else(PoisonError::into_inner) = Some(Shared { weights: weights.clone(), labels: labels.clone() });
    publish_phase(&mailbox, Phase::WarmingUp, &source, n);
    let params = LifParams::default();
    let dt = Duration::from_secs_f64(params.dt_ms / 1000.0);
    let info = Arc::new(brain_info(&labels, &weights));
    let mut brain = Brain::new(LifSim::new(weights, params, cfg.seed), labels);
    brain.warmup(cfg.warmup);

    let mut state = LoopState { speed: 1.0, paused: false, pending_steps: 0 };
    let mut t0 = Instant::now();
    let mut done: u64 = 0;
    let (mut rate_t, mut rate_steps, mut steps_per_s) = (Instant::now(), brain.steps, 0.0f32);
    let mut dropped: u64 = 0;
    let mut driving = false;
    let mut dex = DexState { tracker: None, table: None, enabled: true };
    publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving, &info);

    while !stop.load(Ordering::Relaxed) {
        let before = (state.speed, state.paused);
        loop {
            match rx.try_recv() {
                Ok(Command::Shutdown) | Err(TryRecvError::Disconnected) => return,
                Ok(cmd) => driving |= apply(&mut brain, &mut state, &mut dex, cfg.seed, &events, cmd),
                Err(TryRecvError::Empty) => break,
            }
        }
        if (state.speed, state.paused) != before {
            t0 = Instant::now(); // re-anchor the clock when the speed changes, as Brain._loop does
            done = 0;
            publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving, &info);
        }
        if state.pending_steps > 0 {
            for _ in 0..state.pending_steps {
                brain.step();
            }
            state.pending_steps = 0;
            publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving, &info);
        }
        if state.paused {
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        let elapsed = t0.elapsed();
        let due = (elapsed.as_secs_f64() / dt.as_secs_f64() * f64::from(state.speed)) as u64;
        if due > done + MAX_BACKLOG {
            dropped += due - MAX_BACKLOG - done;
            done = due - MAX_BACKLOG;
        }
        if done >= due {
            // sleep until the next step is due (shorter than a step so commands and shutdown stay responsive)
            let next = Duration::from_secs_f64((done + 1) as f64 * dt.as_secs_f64() / f64::from(state.speed.max(1e-3)));
            let wait = next.saturating_sub(elapsed).min(Duration::from_millis(2));
            thread::sleep(wait.saturating_sub(Duration::from_micros(100)));
            continue;
        }
        brain.step();
        done += 1;
        dex_check(&brain, &mut dex, &events);
        if rate_t.elapsed() >= Duration::from_secs(1) {
            steps_per_s = (brain.steps - rate_steps) as f32 / rate_t.elapsed().as_secs_f32();
            rate_t = Instant::now();
            rate_steps = brain.steps;
        }
        if brain.steps.is_multiple_of(cfg.publish_every) {
            publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving, &info);
        }
    }
}

struct LoopState {
    speed: f32,
    paused: bool,
    pending_steps: u32,
}

/// Returns true when the command turned a stimulus on.
fn apply(brain: &mut Brain, state: &mut LoopState, dex: &mut DexState, seed: u64, events: &Sender<SimEvent>, cmd: Command) -> bool {
    match cmd {
        Command::DriveSpec { spec, amp } => match brain.rows_of(&spec) {
            Ok(rows) => {
                brain.drive(&rows, amp);
                amp != 0.0
            }
            Err(_) => false,
        },
        Command::DriveRows { rows, amp } => {
            brain.drive(&rows, amp);
            amp != 0.0
        }
        Command::SetSpeed(s) => {
            state.speed = s.clamp(0.05, 20.0);
            false
        }
        Command::Pause(p) => {
            state.paused = p;
            false
        }
        Command::StepOnce(k) => {
            state.pending_steps += k;
            false
        }
        Command::Kill => {
            brain.kill();
            false
        }
        Command::Revive => {
            brain.revive();
            false
        }
        Command::Poke { region, side, strength } => {
            brain.poke(&region, side, strength);
            false
        }
        Command::Surgery { spec, mode } => {
            if let Ok(rows) = brain.rows_of(&spec) {
                brain.set_surgery(&rows, mode);
            }
            false
        }
        Command::Sedation(s) => {
            brain.sedation = s.clamp(0.0, 1.0);
            false
        }
        Command::PainLevel(l) => {
            brain.groups.set_pain_level(l);
            false
        }
        Command::DexTable { table, known } => {
            let set: std::collections::HashSet<String> = known.into_iter().collect();
            dex.tracker = Some(fly_sim::neurodex::Tracker::new(&table, &|n| set.contains(n)));
            dex.table = Some(table);
            false
        }
        Command::DexKnown(known) => {
            if let (Some(t), Some(tab)) = (dex.tracker.as_mut(), dex.table.as_ref()) {
                let set: std::collections::HashSet<String> = known.into_iter().collect();
                t.sync(tab, &|n| set.contains(n));
            }
            false
        }
        Command::DexEnabled(on) => {
            dex.enabled = on;
            false
        }
        Command::ExportState => {
            let _ = events.send(SimEvent::State(Box::new(brain.export_state(seed))));
            false
        }
        Command::ImportState(st) => {
            brain.import_state(&st);
            false
        }
        Command::Shutdown => false,
    }
}

struct DexState {
    tracker: Option<fly_sim::neurodex::Tracker>,
    table: Option<Arc<fly_sim::neurodex::TypeTable>>,
    enabled: bool,
}

/// `Extras.dex_tick`: every `CHECK_STEPS` steps, feed the recent spikes to the discovery rule and report what it found.
fn dex_check(brain: &Brain, dex: &mut DexState, events: &Sender<SimEvent>) {
    use fly_sim::neurodex::{CHECK_STEPS, WINDOW_STEPS};
    if !dex.enabled || brain.dead() || !brain.steps.is_multiple_of(u64::from(CHECK_STEPS)) {
        return;
    }
    let (Some(tracker), Some(table)) = (dex.tracker.as_mut(), dex.table.as_ref()) else { return };
    let (spikes, steps) = brain.window_spikes(WINDOW_STEPS);
    let dt = brain.sim.p.dt_ms / 1000.0;
    for f in tracker.observe(table, 0, &spikes, dt, brain.dex_calm(), brain.driven(), steps) {
        let _ = events.send(SimEvent::Discovered(f));
    }
}

#[allow(clippy::too_many_arguments)]
fn publish(
    mailbox: &Mailbox<Snapshot>,
    brain: &Brain,
    source: &str,
    state: &LoopState,
    steps_per_s: f32,
    dropped: u64,
    driving: bool,
    info: &Arc<BrainInfo>,
) {
    let sim = &brain.sim;
    mailbox.publish(Snapshot::default, |s| {
        s.phase = if state.paused { Phase::Paused } else { Phase::Running };
        s.source.clear();
        s.source.push_str(source);
        s.neurons = sim.n;
        s.step = brain.steps;
        s.sim_time_s = brain.steps as f64 * sim.p.dt_ms / 1000.0;
        s.steps_per_s = steps_per_s;
        s.step_ms = sim.last_step_ms as f32;
        s.steps_dropped = dropped;
        s.gain = sim.gain;
        s.active_count = sim.activity.active_count;
        s.spikes.clear();
        s.spikes.extend_from_slice(sim.spikes());
        s.rates.clear();
        s.rates.extend_from_slice(sim.activity.rates());
        s.driving = driving;
        s.dead = brain.dead();
        s.speed = state.speed;
        s.hz.clear();
        s.hz.extend_from_slice(&brain.groups.fast);
        s.levels.clear();
        s.levels.extend(brain.groups.names.iter().map(|n| brain.groups.level(n)));
        let level = brain.groups.pain_level;
        s.pain_parts = brain.groups.pain_parts(level);
        s.pain_neurons = brain.groups.pain_neurons(level) as u32;
        s.pain_level = level as u8;
        s.pain_neurons_all = [0, 1, 2].map(|l| brain.groups.pain_neurons(l) as u32);
        s.whole_trace.clear();
        if let Some(c) = brain.groups.col("whole brain") {
            let n = brain.groups.hist_n;
            s.whole_trace.extend(brain.groups.history_col(c, n.saturating_sub(600), n));
        }
        if s.info.is_none() {
            s.info = Some(info.clone());
        }
        s.published_at = Some(Instant::now());
    });
}

fn brain_info(labels: &Labels, weights: &Weights) -> BrainInfo {
    let n = labels.types.len();
    let pop_names: Vec<String> = fly_sim::groups::population_names().iter().map(|s| (*s).to_string()).collect();
    let population = (0..n)
        .map(|i| {
            let sc = labels.superclass.get(i).map_or("", String::as_str);
            fly_sim::groups::population_of(sc).map_or(255, |p| p as u8)
        })
        .collect();
    // group names come from a throwaway Groups so the snapshot's `levels` order is documented in one place
    let g = fly_sim::groups::Groups::new(labels, n, 5.0, 0);
    let mut region_names: Vec<String> = labels.region.iter().cloned().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    region_names.truncate(250);
    let region_id = labels.region.iter().map(|r| region_names.iter().position(|n| n == r).map_or(255, |i| i as u8)).collect();
    BrainInfo {
        types: labels.types.clone(),
        region_id,
        region_names,
        layout: layout(labels, weights),
        soma: labels.soma.clone(),
        population,
        group_names: g.names.clone(),
        population_names: pop_names,
        synapses: weights.csr.nnz(),
    }
}

/// Drawing positions: the real cell body where the dataset has one, else the synapse-weighted centre of the placed partners (Python
/// `BrainView.__init__`: `A = (|W| + |W|^T) @ (S * has) / total`). Normalised so the brain spans about [-1, 1] across; y is up.
pub fn layout(labels: &Labels, weights: &Weights) -> Vec<[f32; 3]> {
    let n = weights.n();
    if labels.soma.len() != n {
        return vec![[f32::NAN; 3]; n];
    }
    let has: Vec<bool> = labels.soma.iter().map(|s| s[0].is_finite()).collect();
    let mut sum = vec![[0.0f64; 3]; n];
    let mut tot = vec![0.0f64; n];
    for i in 0..n {
        let (cols, data) = weights.csr.row(i);
        for (&j, &w) in cols.iter().zip(data) {
            let (j, w) = (j as usize, f64::from(w.abs()));
            if has[j] {
                tot[i] += w;
                for (acc, v) in sum[i].iter_mut().zip(labels.soma[j]) {
                    *acc += w * f64::from(v);
                }
            }
            if has[i] {
                tot[j] += w;
                for (acc, v) in sum[j].iter_mut().zip(labels.soma[i]) {
                    *acc += w * f64::from(v);
                }
            }
        }
    }
    // MaleCNS coordinates: x 0..96,000, y 0..70,000 (down), z 0..135,000 (depth); the Python front view is centred on (48,000, 28,000, 30,000)
    let (cx, cy, cz, sc) = (48_000.0f32, 35_000.0f32, 65_000.0f32, 1.0f32 / 48_000.0);
    (0..n)
        .map(|i| {
            let p = if has[i] {
                labels.soma[i]
            } else if tot[i] > 0.0 {
                [(sum[i][0] / tot[i]) as f32, (sum[i][1] / tot[i]) as f32, (sum[i][2] / tot[i]) as f32]
            } else {
                return [f32::NAN; 3];
            };
            [(p[0] - cx) * sc, (cy - p[1]) * sc, (p[2] - cz) * sc]
        })
        .collect()
}

/// The current of a held stimulus, re-exported so the game does not need to reach into `fly_sim`.
pub const STIMULUS_AMP: f32 = STIM_AMP;

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_for(h: &SimHandle, what: &str, mut ok: impl FnMut(&Snapshot) -> bool) -> Box<Snapshot> {
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if let Some(s) = h.latest() {
                if ok(&s) {
                    return s;
                }
                h.mailbox.recycle(s);
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("timed out waiting for {what}");
    }

    fn cfg(n: usize) -> SimConfig {
        SimConfig { source: Source::Synthetic { n }, seed: 1, warmup: 20, publish_every: 2 }
    }

    #[test]
    fn runs_at_the_fixed_rate_and_publishes() {
        let h = spawn(cfg(3000));
        let first = wait_for(&h, "the first running snapshot", |s| s.phase == Phase::Running && s.step > 40);
        let t = Instant::now();
        let s0 = first.step;
        thread::sleep(Duration::from_millis(500));
        let last = wait_for(&h, "a later snapshot", |s| s.step > s0 + 30);
        let steps = (last.step - s0) as f64;
        let secs = t.elapsed().as_secs_f64();
        let rate = steps / secs;
        assert!((120.0..=240.0).contains(&rate), "expected ~200 steps/s, got {rate:.0}");
        assert_eq!(last.rates.len(), 3000);
        assert!(last.sim_time_s > 0.0);
    }

    #[test]
    fn the_sim_does_not_wait_for_a_reader() {
        let h = spawn(cfg(3000));
        wait_for(&h, "start", |s| s.phase == Phase::Running);
        thread::sleep(Duration::from_millis(600)); // never reading: publishing must not block the stepping
        let s = wait_for(&h, "a snapshot after the idle reader", |_| true);
        assert!(s.step >= 80, "the sim kept stepping while nobody read ({} steps)", s.step);
    }

    #[test]
    fn commands_drive_pause_and_stop() {
        let h = spawn(cfg(3000));
        wait_for(&h, "start", |s| s.phase == Phase::Running);
        assert!(h.send(Command::DriveSpec { spec: "type:LPLC2".into(), amp: STIMULUS_AMP }));
        let s = wait_for(&h, "driving flag", |s| s.driving);
        assert!(s.driving);
        h.send(Command::Pause(true));
        wait_for(&h, "paused", |s| s.phase == Phase::Paused);
        h.send(Command::StepOnce(5));
        h.send(Command::Kill);
        wait_for(&h, "dead", |s| s.dead);
        drop(h); // joins the thread
    }

    #[test]
    fn a_missing_pack_is_reported_not_fatal() {
        let h = spawn(SimConfig::new(Source::Pack("/nonexistent/kick_brain.npz".into())));
        let s = wait_for(&h, "failure message", |s| matches!(s.phase, Phase::Failed(_)));
        assert!(matches!(&s.phase, Phase::Failed(m) if m.contains("kick_brain.npz")));
        assert!(h.is_running());
    }
}
