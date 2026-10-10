//! The brain on a thread of its own, at a fixed 5 ms timestep (200 steps/s), like `Brain._loop` in the Python game.
//!
//! The render thread talks to it with [`Command`]s over an unbounded channel (sending never blocks) and reads [`Snapshot`]s from a
//! [`Mailbox`] (reading never waits for a step). The sim thread paces itself against the wall clock: if it falls more than 40 steps
//! behind it drops the backlog instead of racing, exactly as the Python loop does, and counts what it dropped.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use fly_sim::brain::STIM_AMP;
use fly_sim::csr::Weights;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{Labels, Pack};
use fly_sim::training::{Kind, Outcome, Training};
use fly_sim::{Brain, synthetic};

use crate::snapshot::{BrainInfo, ChallengeView, Mailbox, MemoryView, Phase, Snapshot, TrainView};

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
    /// Where the fly's memory lives (`fly-memory.npz`): learning is on when this is set and the pack has the dopamine tables.
    pub memory_dir: Option<std::path::PathBuf>,
}

impl SimConfig {
    pub fn new(source: Source) -> Self {
        Self { source, seed: 1000, warmup: WARMUP_STEPS, publish_every: 4, memory_dir: None }
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
    /// An entry of the brain surgery screen (`fly_sim::surgery::ENTRIES`): -1 off, 0 normal, +1 on.
    SurgeryEntry {
        index: usize,
        mode: i8,
    },
    /// A cell type switched from the neuron inspector.
    SurgeryType {
        ty: String,
        mode: i8,
    },
    /// CLEAR ALL.
    SurgeryClear,
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
    /// `Memory.observe`: remember the Kenyon-cell pattern of a smell that is present now (the game sends it every 3rd frame).
    Observe(String),
    /// Start conditioning (`fear`, `like` or `test`) on a smell; `trials` 0 means 10.
    Train {
        kind: String,
        scent: String,
        trials: u32,
    },
    StopTrain,
    /// Forget everything the fly learned (and save the empty memory).
    WipeMemory,
    /// A challenge borrows the memory: nothing is autosaved while it does (`Challenge.borrows_memory`).
    MemoryBorrowed(bool),
    /// Write `fly-memory.npz` now.
    SaveMemory,
    /// `TMaze.pick`: start the door-learning challenge (`odor_a` or `odor_b` zaps).
    StartTMaze {
        zap: String,
        seed: u64,
    },
    StartSweet,
    /// Offer sugar at this strength (0..100 %).
    SweetOffer(u32),
    /// Close the running challenge (the memory it borrowed is put back).
    EndChallenge,
    /// Look one neuron up for the inspector: the answer is [`SimEvent::Neuron`].
    Inspect(u32),
    /// The brain stethoscope's target (`audio.stethoscope_target`, or a selected brain region by its pack label): None switches it off.
    Stethoscope {
        target: Option<String>,
        region: Option<String>,
    },
    /// Report the whole brain state (a save): the answer comes as [`SimEvent::State`].
    ExportState,
    /// Put a saved brain state back (a loaded save).
    ImportState(Box<fly_sim::brain::BrainState>),
    /// Every fly's brain state, the focused fly first: the answer comes as [`SimEvent::StateAll`] (a save of several flies).
    ExportAll,
    /// Put a saved brain state into this fly's brain (it must exist: `SpawnFly` first).
    ImportFly {
        slot: u32,
        state: Box<fly_sim::brain::BrainState>,
    },
    /// Add a fly: a second (third...) brain with its own individuality (`sigma`) and noise, stepped with the first. `slot` is the id the
    /// render side gave its body; slot 0 is the first fly.
    SpawnFly {
        slot: u32,
        seed: u64,
        sigma: f64,
    },
    /// Take a fly's brain away (its body is gone).
    RemoveFly(u32),
    /// Make this fly's brain the one the HUD, the brain views and every single-brain command (surgery, challenges, memory) address.
    FocusFly(u32),
    /// `Command::Poke` on one fly's brain.
    PokeFly {
        slot: u32,
        region: String,
        side: Option<char>,
        strength: f32,
    },
    KillFly(u32),
    ReviveFly(u32),
    /// A Lab model parameter (`noise_std`, `bias`, `target_rate_hz`, `ext_gain`, `gain_adapt`) on every fly's brain.
    SetModelParam {
        name: String,
        value: f64,
    },
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
    /// The answer to [`Command::ExportAll`]: `(slot, seed, state)`, the focused fly first.
    StateAll(Vec<(u32, u64, Box<fly_sim::brain::BrainState>)>),
    /// The answer to [`Command::ExportState`].
    State(Box<fly_sim::brain::BrainState>),
    /// The T-maze made a choice (`right` = the door that does not zap).
    TMazeChose {
        right: bool,
    },
    TMazeFinished {
        right: usize,
    },
    /// The fly answered a sugar offer: (dose, extended, ratio).
    SweetAnswer(u32, bool, f32),
    SweetFinished(Option<u32>),
    /// The brain autopsy, 3 s of brain time after death.
    Autopsy(Box<fly_sim::autopsy::Autopsy>),
    Neuron(Box<fly_sim::inspect::NeuronInfo>),
}

pub struct SimHandle {
    events: Mutex<Receiver<SimEvent>>,
    pub shared: Arc<Mutex<Option<Shared>>>,
    tx: Sender<Command>,
    pub mailbox: Arc<Mailbox<Snapshot>>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    /// Spikes of the stethoscope's neurons since the game last took them (`Brain.take_stethoscope_spikes`).
    pub steth: Arc<AtomicU32>,
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
    let steth = Arc::new(AtomicU32::new(0));
    let join = {
        let (mailbox, stop, shared, steth) = (mailbox.clone(), stop.clone(), shared.clone(), steth.clone());
        thread::Builder::new()
            .name("fly-sim".into())
            .spawn(move || run(cfg, rx, mailbox, stop, shared, ev_tx, steth))
            .expect("could not start the sim thread")
    };
    SimHandle { events: Mutex::new(ev_rx), shared, tx, mailbox, stop, join: Some(join), steth }
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
    steth: Arc<AtomicU32>,
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
    let (spawn_weights, spawn_labels) = (weights.clone(), labels.clone());
    let mut multi = Multi { extras: Vec::new(), focus: 0, seeds: vec![(0, cfg.seed)] };
    let mut brain = Brain::new(LifSim::new(weights, params, cfg.seed), labels);
    let mut mem = MemState::new(cfg.memory_dir.clone());
    if mem.dir.is_some() && brain.enable_memory() {
        mem.load(&mut brain);
    }
    brain.warmup(cfg.warmup);

    let mut state = LoopState { speed: 1.0, paused: false, pending_steps: 0 };
    let mut t0 = Instant::now();
    let mut done: u64 = 0;
    let (mut rate_t, mut rate_steps, mut steps_per_s) = (Instant::now(), brain.steps, 0.0f32);
    let mut dropped: u64 = 0;
    let mut driving = false;
    let mut dex = DexState { tracker: None, table: None, enabled: true };
    publish(&mailbox, &brain, &multi, &source, &state, steps_per_s, dropped, driving, &info, &mem);

    while !stop.load(Ordering::Relaxed) {
        let before = (state.speed, state.paused);
        loop {
            match rx.try_recv() {
                Ok(Command::Shutdown) | Err(TryRecvError::Disconnected) => {
                    mem.save_if_dirty(&mut brain);
                    return;
                }
                Ok(Command::SpawnFly { slot, seed, sigma }) => {
                    if slot != multi.focus && !multi.extras.iter().any(|e| e.0 == slot) {
                        let mut sim = LifSim::new(spawn_weights.clone(), LifParams::default(), seed);
                        sim.set_individuality(fly_sim::lif::Individuality::compute(seed, spawn_weights.n(), sigma));
                        let mut b = Brain::new(sim, spawn_labels.clone());
                        b.warmup(cfg.warmup);
                        b.steps = brain.steps; // one clock: a focus change must not make the step counter jump
                        multi.extras.push((slot, b));
                        multi.seeds.push((slot, seed));
                    }
                }
                Ok(Command::RemoveFly(slot)) => {
                    multi.extras.retain(|e| e.0 != slot);
                    multi.seeds.retain(|s| s.0 != slot || s.0 == multi.focus);
                }
                Ok(Command::ExportState) => {
                    let _ = events.send(SimEvent::State(Box::new(brain.export_state(multi.seed_of(multi.focus, cfg.seed)))));
                }
                Ok(Command::ExportAll) => {
                    let mut all = vec![(
                        multi.focus,
                        multi.seed_of(multi.focus, cfg.seed),
                        Box::new(brain.export_state(multi.seed_of(multi.focus, cfg.seed))),
                    )];
                    for (slot, b) in &multi.extras {
                        let seed = multi.seed_of(*slot, cfg.seed);
                        all.push((*slot, seed, Box::new(b.export_state(seed))));
                    }
                    let _ = events.send(SimEvent::StateAll(all));
                }
                Ok(Command::ImportFly { slot, state }) => {
                    if let Some(b) = multi.brain_of(&mut brain, slot) {
                        b.import_state(&state);
                    }
                }
                Ok(Command::FocusFly(slot)) => {
                    if slot != multi.focus
                        && let Some(i) = multi.extras.iter().position(|e| e.0 == slot)
                    {
                        std::mem::swap(&mut brain, &mut multi.extras[i].1);
                        // the learning state belongs to whoever is in front
                        std::mem::swap(&mut brain.memory, &mut multi.extras[i].1.memory);
                        multi.extras[i].0 = multi.focus;
                        multi.focus = slot;
                        driving = brain.driven();
                    }
                }
                Ok(Command::PokeFly { slot, region, side, strength }) => {
                    if let Some(b) = multi.brain_of(&mut brain, slot) {
                        b.poke(&region, side, strength);
                    }
                }
                Ok(Command::SetModelParam { name, value }) => {
                    brain.set_model_param(&name, value);
                    for (_, b) in &mut multi.extras {
                        b.set_model_param(&name, value);
                    }
                }
                Ok(Command::KillFly(slot)) => {
                    if let Some(b) = multi.brain_of(&mut brain, slot) {
                        b.kill();
                    }
                }
                Ok(Command::ReviveFly(slot)) => {
                    if let Some(b) = multi.brain_of(&mut brain, slot) {
                        b.revive();
                    }
                }
                Ok(cmd) => driving |= apply(&mut brain, &mut state, &mut dex, &mut mem, &events, cmd),
                Err(TryRecvError::Empty) => break,
            }
        }
        if (state.speed, state.paused) != before {
            t0 = Instant::now(); // re-anchor the clock when the speed changes, as Brain._loop does
            done = 0;
            publish(&mailbox, &brain, &multi, &source, &state, steps_per_s, dropped, driving, &info, &mem);
        }
        if state.pending_steps > 0 {
            for _ in 0..state.pending_steps {
                brain.step();
                for (_, b) in &mut multi.extras {
                    b.step();
                }
            }
            state.pending_steps = 0;
            publish(&mailbox, &brain, &multi, &source, &state, steps_per_s, dropped, driving, &info, &mem);
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
        for (_, b) in &mut multi.extras {
            b.step();
        }
        done += 1;
        dex_check(&brain, &mut dex, &events);
        mem.after_step(&mut brain, &events);
        if let Some(mask) = mem.steth_mask.as_ref() {
            let n = brain.sim.spikes().iter().filter(|&&i| mask[i as usize]).count() as u32;
            if n > 0 {
                steth.fetch_add(n, Ordering::Relaxed);
            }
        }
        if rate_t.elapsed() >= Duration::from_secs(1) {
            steps_per_s = brain.steps.saturating_sub(rate_steps) as f32 / rate_t.elapsed().as_secs_f32();
            rate_t = Instant::now();
            rate_steps = brain.steps;
        }
        if brain.steps.is_multiple_of(cfg.publish_every) {
            publish(&mailbox, &brain, &multi, &source, &state, steps_per_s, dropped, driving, &info, &mem);
        }
    }
    mem.save_if_dirty(&mut brain);
}

/// The other flies' brains. The focused fly's brain is the loop's `brain` (everything single-brain keeps working on it); a focus change
/// swaps it with the new one's, and the learning state travels with the focus so training and memory stay with the fly in front.
struct Multi {
    extras: Vec<(u32, Brain)>,
    focus: u32,
    /// The seed each fly's simulator was made with (a save needs it; the sim does not remember it).
    seeds: Vec<(u32, u64)>,
}

impl Multi {
    fn seed_of(&self, slot: u32, default: u64) -> u64 {
        self.seeds.iter().find(|s| s.0 == slot).map_or(default, |s| s.1)
    }

    fn brain_of<'a>(&'a mut self, primary: &'a mut Brain, slot: u32) -> Option<&'a mut Brain> {
        if slot == self.focus {
            return Some(primary);
        }
        self.extras.iter_mut().find(|e| e.0 == slot).map(|e| &mut e.1)
    }
}

struct LoopState {
    speed: f32,
    paused: bool,
    pending_steps: u32,
}

/// Returns true when the command turned a stimulus on.
fn apply(
    brain: &mut Brain,
    state: &mut LoopState,
    dex: &mut DexState,
    mem: &mut MemState,
    events: &Sender<SimEvent>,
    cmd: Command,
) -> bool {
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
        // the multi-fly commands are handled in the loop, where the other brains are
        Command::SpawnFly { .. }
        | Command::RemoveFly(_)
        | Command::FocusFly(_)
        | Command::PokeFly { .. }
        | Command::KillFly(_)
        | Command::SetModelParam { .. }
        | Command::ReviveFly(_) => false,
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
        Command::Inspect(i) => {
            if let Some(info) = fly_sim::inspect::neuron_info(brain.labels(), brain.sim.weights(), i as usize) {
                let _ = events.send(SimEvent::Neuron(Box::new(info)));
            }
            false
        }
        Command::Stethoscope { target, region } => {
            mem.steth_mask = target.and_then(|t| steth_rows(brain, &t, region.as_deref())).map(|rows| {
                let mut m = vec![false; brain.n()];
                for r in rows {
                    m[r as usize] = true;
                }
                m
            });
            false
        }
        Command::SurgeryEntry { index, mode } => {
            if index < mem.surgery_modes.len() {
                mem.surgery_modes[index] = mode;
                mem.apply_surgery(brain);
            }
            false
        }
        Command::SurgeryType { ty, mode } => {
            mem.surgery_types.retain(|t| t.0 != ty);
            if mode != 0 {
                mem.surgery_types.push((ty, mode));
            }
            mem.apply_surgery(brain);
            false
        }
        Command::SurgeryClear => {
            mem.surgery_modes.fill(0);
            mem.surgery_types.clear();
            mem.apply_surgery(brain);
            false
        }
        Command::StartTMaze { zap, seed } => {
            mem.end_challenge(brain);
            if let (Some(t), Some(m)) = (fly_sim::challenge::TMaze::start(brain, &zap, seed), brain.memory.as_ref()) {
                mem.saved = Some(m.snapshot());
                mem.borrowed = true;
                mem.tmaze = Some(t);
            }
            false
        }
        Command::StartSweet => {
            mem.end_challenge(brain);
            mem.sweet = Some(fly_sim::challenge::Sweet::new());
            false
        }
        Command::SweetOffer(dose) => {
            if let Some(s) = mem.sweet.as_mut() {
                s.offer(brain, dose);
            }
            false
        }
        Command::EndChallenge => {
            mem.end_challenge(brain);
            false
        }
        Command::Observe(scent) => {
            if let Some(m) = brain.memory.as_mut() {
                let rates = brain.sim.activity.rates();
                m.observe(&scent, rates);
            }
            false
        }
        Command::Train { kind, scent, trials } => {
            if mem.training.is_none()
                && let Some(k) = Kind::parse(&kind)
            {
                mem.training = Training::start(brain, &scent, k, trials);
            }
            false
        }
        Command::StopTrain => {
            if mem.training.take().is_some() {
                mem.save(brain);
            }
            false
        }
        Command::WipeMemory => {
            if mem.training.is_none()
                && let Some(mut m) = brain.memory.take()
            {
                m.wipe(&mut brain.sim);
                brain.memory = Some(m);
                mem.save(brain);
            }
            false
        }
        Command::MemoryBorrowed(b) => {
            mem.borrowed = b;
            false
        }
        Command::SaveMemory => {
            mem.save(brain);
            false
        }
        Command::ExportState | Command::ExportAll | Command::ImportFly { .. } => false,
        Command::ImportState(st) => {
            brain.import_state(&st);
            false
        }
        Command::Shutdown => false,
    }
}

/// `Game.update_stethoscope_target`: a selected region wins, then the setting. `central_complex` and `optic_lobes` find nothing in the Python
/// game (its region table is keyed by display names such as "Central Complex"); here they name those regions.
fn steth_rows(brain: &Brain, target: &str, region: Option<&str>) -> Option<Vec<u32>> {
    let labels = brain.labels();
    let n = brain.n() as u32;
    let by_region = |name: &str| -> Vec<u32> { (0..n).filter(|&i| labels.region.get(i as usize).is_some_and(|r| r == name)).collect() };
    let rows = if let Some(r) = region {
        by_region(r)
    } else {
        match target {
            "antennal_lobe" => brain.groups_ref().det_members("smell"),
            "mushroom_body" => (0..n).filter(|&i| labels.types[i as usize].starts_with("KC")).collect(),
            "central_complex" => by_region("Central Complex"),
            "optic_lobes" => by_region("Optic Lobe"),
            "motor" => (0..n).filter(|&i| matches!(labels.superclass[i as usize].as_str(), "descending_neuron" | "motor")).collect(),
            "whole_brain" => (0..n).collect(),
            _ => vec![],
        }
    };
    (!rows.is_empty()).then_some(rows)
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
    multi: &Multi,
    source: &str,
    state: &LoopState,
    steps_per_s: f32,
    dropped: u64,
    driving: bool,
    info: &Arc<BrainInfo>,
    mem: &MemState,
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
        s.focus = multi.focus;
        s.others.clear();
        for (slot, b) in &multi.extras {
            s.others.push(crate::snapshot::OtherFly {
                slot: *slot,
                levels: b.groups.names.iter().map(|n| b.groups.level(n)).collect(),
                hz: b.groups.fast.clone(),
                dead: b.dead(),
            });
        }
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
        s.surgery_modes.clone_from(&mem.surgery_modes);
        s.surgery_types.clone_from(&mem.surgery_types);
        s.memory.clone_from(&mem.view);
        s.challenge.clone_from(&mem.chal_view);
        s.challenge_speed = mem.chal_speed;
        s.published_at = Some(Instant::now());
    });
}

/// The fly's memory on the sim thread: its file, the training in progress, the autosave clock and the view the game reads.
struct MemState {
    dir: Option<std::path::PathBuf>,
    last_save: Instant,
    training: Option<Training>,
    borrowed: bool,
    view: Option<Arc<MemoryView>>,
    tmaze: Option<fly_sim::challenge::TMaze>,
    sweet: Option<fly_sim::challenge::Sweet>,
    saved: Option<fly_sim::memory::Saved>,
    /// Brain surgery screen: the rows of each entry, the modes set, and the types switched from the inspector.
    surgery_rows: Vec<Vec<u32>>,
    surgery_modes: Vec<i8>,
    surgery_types: Vec<(String, i8)>,
    chal_view: Option<Arc<ChallengeView>>,
    chal_speed: f32,
    autopsy_done: bool,
    /// Which neurons the stethoscope listens to.
    steth_mask: Option<Vec<bool>>,
}

fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

impl MemState {
    fn new(dir: Option<std::path::PathBuf>) -> Self {
        MemState {
            dir,
            last_save: Instant::now(),
            training: None,
            borrowed: false,
            view: None,
            tmaze: None,
            sweet: None,
            saved: None,
            surgery_rows: vec![],
            surgery_modes: vec![0; fly_sim::surgery::ENTRIES.len()],
            surgery_types: vec![],
            chal_view: None,
            chal_speed: 1.0,
            autopsy_done: false,
            steth_mask: None,
        }
    }

    fn file(&self) -> Option<std::path::PathBuf> {
        self.dir.as_ref().map(|d| d.join("fly-memory.npz"))
    }

    /// `Memory.load`: the saved weights, smell patterns and calm baselines of an earlier session, if they were made for this pack.
    fn load(&mut self, brain: &mut Brain) {
        let (Some(file), Some(mut m)) = (self.file(), brain.memory.take()) else { return };
        if let Some(saved) = ktf_data::stores::FlyMemory::load(&file, &m.signature) {
            let templates = saved.templates.clone();
            if m.restore(&mut brain.sim, saved.w, saved.kc_calm, saved.dan_calm, templates) {
                if let Some(o) = saved.naive_mbon.as_object() {
                    m.naive_mbon = o.iter().filter_map(|(k, v)| v.as_f64().map(|x| (k.clone(), x))).collect();
                }
                if let Some(o) = saved.trials.as_object() {
                    for (scent, rows) in o {
                        let rows = rows.as_array().map(Vec::as_slice).unwrap_or_default();
                        let parsed = rows
                            .iter()
                            .filter_map(|r| {
                                let r = r.as_array()?;
                                Some(fly_sim::memory::Trial {
                                    time: r.first()?.as_f64()?,
                                    kind: r.get(1)?.as_str()?.to_string(),
                                    fear: r.get(2)?.as_f64()?,
                                    like: r.get(3)?.as_f64()?,
                                    mbon_hz: r.get(4).and_then(serde_json::Value::as_f64),
                                })
                            })
                            .collect();
                        m.log.insert(scent.clone(), parsed);
                    }
                }
            }
        }
        brain.memory = Some(m);
    }

    fn save_if_dirty(&mut self, brain: &mut Brain) {
        if brain.memory.as_ref().is_some_and(|m| m.dirty) {
            self.save(brain);
        }
    }

    /// `Memory.save`: `fly-memory.npz` and `training-log.json`. A full or read-only folder is a warning, never a hang.
    fn save(&mut self, brain: &mut Brain) {
        let (Some(file), Some(m)) = (self.file(), brain.memory.as_mut()) else { return };
        if self.borrowed {
            return;
        }
        let trials: serde_json::Map<String, serde_json::Value> = m
            .log
            .iter()
            .map(|(k, rows)| {
                (
                    k.clone(),
                    serde_json::Value::Array(rows.iter().map(|t| serde_json::json!([t.time, t.kind, t.fear, t.like, t.mbon_hz])).collect()),
                )
            })
            .collect();
        let saved = ktf_data::stores::FlyMemory {
            signature: m.signature.to_vec(),
            w: m.w.clone(),
            templates: m.templates.clone(),
            kc_calm: m.kc_calm.clone(),
            dan_calm: m.dan_calm.clone(),
            naive_mbon: serde_json::to_value(&m.naive_mbon).unwrap_or_default(),
            trials: serde_json::Value::Object(trials),
        };
        match saved.save(&file) {
            Ok(()) => m.dirty = false,
            Err(e) => eprintln!("could not save the fly's memory to {}: {e}", file.display()),
        }
        self.last_save = Instant::now();
    }

    /// After every brain step: the conditioning protocol, the 30 s autosave, and the view the game reads (every 12 steps).
    fn after_step(&mut self, brain: &mut Brain, events: &Sender<SimEvent>) {
        self.tick_challenges(brain, events);
        self.check_autopsy(brain, events);
        if brain.memory.is_none() {
            return;
        }
        if let Some(t) = self.training.as_mut()
            && t.tick(brain, unix_now()) == Outcome::Done
        {
            self.training = None;
            self.save(brain);
        }
        if brain.steps.is_multiple_of(600)
            && self.last_save.elapsed() > Duration::from_secs(30)
            && brain.memory.as_ref().is_some_and(|m| m.dirty)
        {
            self.save(brain);
        }
        if brain.steps.is_multiple_of(12) || self.view.is_none() {
            self.rebuild_view(brain);
        }
    }

    /// `Game._apply_surgery`: clear, then every entry in order, then the types switched from the inspector.
    fn apply_surgery(&mut self, brain: &mut Brain) {
        if self.surgery_rows.is_empty() {
            self.surgery_rows = fly_sim::surgery::all_rows(brain.labels(), brain.groups_ref());
        }
        brain.clear_surgery();
        for (rows, &mode) in self.surgery_rows.iter().zip(&self.surgery_modes) {
            if mode != 0 {
                brain.set_surgery(rows, mode);
            }
        }
        for (ty, mode) in &self.surgery_types {
            if let Ok(rows) = brain.rows_of(&format!("type:{ty}")) {
                brain.set_surgery(&rows, *mode);
            }
        }
    }

    /// Put back the memory a challenge borrowed and let go of it.
    fn end_challenge(&mut self, brain: &mut Brain) {
        self.tmaze = None;
        self.sweet = None;
        if let (Some(saved), Some(mut m)) = (self.saved.take(), brain.memory.take()) {
            m.restore_snapshot(&mut brain.sim, saved);
            brain.memory = Some(m);
        }
        self.borrowed = false;
        self.chal_view = None;
        self.chal_speed = 1.0;
    }

    fn check_autopsy(&mut self, brain: &Brain, events: &Sender<SimEvent>) {
        if !brain.dead() {
            self.autopsy_done = false;
            return;
        }
        let g = brain.groups_ref();
        if !self.autopsy_done && g.hist_n >= brain.death_sample + 150 {
            self.autopsy_done = true;
            if let Some(a) = fly_sim::autopsy::compute(g, brain.death_sample, &brain.death_base, g.pain_level) {
                let _ = events.send(SimEvent::Autopsy(Box::new(a)));
            }
        }
    }

    fn tick_challenges(&mut self, brain: &mut Brain, events: &Sender<SimEvent>) {
        use fly_sim::challenge::{SEvent, SPhase, TEvent, TPhase};
        let now = unix_now();
        if let Some(t) = self.tmaze.as_mut() {
            match t.tick(brain, now) {
                TEvent::Chose { right } => {
                    let _ = events.send(SimEvent::TMazeChose { right });
                }
                TEvent::Finished { right } => {
                    let _ = events.send(SimEvent::TMazeFinished { right });
                    // the borrowed memory goes back at once; the panel keeps showing the result
                    if let (Some(saved), Some(mut m)) = (self.saved.take(), brain.memory.take()) {
                        m.restore_snapshot(&mut brain.sim, saved);
                        brain.memory = Some(m);
                    }
                    self.borrowed = false;
                }
                TEvent::None => {}
            }
        }
        if let Some(s) = self.sweet.as_mut() {
            match s.tick(brain, now) {
                SEvent::Answer(d, e, r) => {
                    let _ = events.send(SimEvent::SweetAnswer(d, e, r));
                }
                SEvent::Finished(score) => {
                    let _ = events.send(SimEvent::SweetFinished(score));
                }
                SEvent::None => {}
            }
        }
        if brain.steps.is_multiple_of(4) {
            self.chal_view = if let Some(t) = &self.tmaze {
                self.chal_speed = t.speed();
                Some(Arc::new(ChallengeView {
                    kind: "tmaze".into(),
                    phase: match t.phase {
                        TPhase::Naive => "naive",
                        TPhase::Train => "train",
                        TPhase::Test => "test",
                        TPhase::Done => "done",
                    }
                    .into(),
                    stage: t.stage,
                    cycle: t.cycle,
                    zap: t.zap.clone(),
                    choices: t.choices.clone(),
                    anim_at: t.anim_at,
                    ..Default::default()
                }))
            } else {
                self.sweet.as_ref().map(|s| {
                    Arc::new(ChallengeView {
                        kind: "sweet".into(),
                        phase: match s.phase {
                            SPhase::Ready => "ready",
                            SPhase::Before => "before",
                            SPhase::During => "during",
                            SPhase::Shown => "shown",
                            SPhase::Done => "done",
                        }
                        .into(),
                        dose: s.dose,
                        offers: s.offers.clone(),
                        score: s.score,
                        ..Default::default()
                    })
                })
            };
        }
    }

    fn rebuild_view(&mut self, brain: &Brain) {
        let Some(m) = brain.memory.as_ref() else { return };
        let mut scores: Vec<(String, f32, f32)> = m
            .templates
            .keys()
            .map(|k| {
                let (f, l) = m.memory_of(k);
                (k.clone(), f, l)
            })
            .collect();
        scores.sort_by(|a, b| a.0.cmp(&b.0));
        let train = self.training.as_ref().map(|t| TrainView {
            scent: t.scent.clone(),
            kind: t.kind.name().into(),
            trials: t.trials,
            done: t.done,
            phase: t.phase.describe(t.kind),
        });
        self.view = Some(Arc::new(MemoryView {
            scores,
            weakened: m.weakened_share(),
            synapses: m.w0.len(),
            log: m.log.clone(),
            naive_mbon: m.naive_mbon.clone(),
            path: self.file().map(|p| p.parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default()).unwrap_or_default(),
            train,
        }));
    }
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
        surgery_counts: fly_sim::surgery::all_rows(labels, &g).iter().map(|r| r.len() as u32).collect(),
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
        SimConfig { source: Source::Synthetic { n }, seed: 1, warmup: 20, publish_every: 2, memory_dir: None }
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
    fn the_stethoscope_counts_spikes_of_its_neurons() {
        let h = spawn(cfg(3000));
        wait_for(&h, "start", |s| s.phase == Phase::Running);
        assert_eq!(h.steth.load(Ordering::Relaxed), 0, "off until a target is set");
        h.send(Command::Stethoscope { target: Some("whole_brain".into()), region: None });
        h.send(Command::DriveSpec { spec: "type:LPLC2".into(), amp: STIMULUS_AMP });
        thread::sleep(Duration::from_millis(400));
        assert!(h.steth.swap(0, Ordering::Relaxed) > 0, "the whole brain spikes");
        h.send(Command::Stethoscope { target: None, region: None });
        thread::sleep(Duration::from_millis(100));
        h.steth.store(0, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(300));
        assert_eq!(h.steth.load(Ordering::Relaxed), 0, "off again");
    }

    #[test]
    fn a_missing_pack_is_reported_not_fatal() {
        let h = spawn(SimConfig::new(Source::Pack("/nonexistent/kick_brain.npz".into())));
        let s = wait_for(&h, "failure message", |s| matches!(s.phase, Phase::Failed(_)));
        assert!(matches!(&s.phase, Phase::Failed(m) if m.contains("kick_brain.npz")));
        assert!(h.is_running());
    }
}
