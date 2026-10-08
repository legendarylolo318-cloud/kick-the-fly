//! The brain on a thread of its own, at a fixed 5 ms timestep (200 steps/s), like `Brain._loop` in the Python game.
//!
//! The render thread talks to it with [`Command`]s over an unbounded channel (sending never blocks) and reads [`Snapshot`]s from a
//! [`Mailbox`] (reading never waits for a step). The sim thread paces itself against the wall clock: if it falls more than 40 steps
//! behind it drops the backlog instead of racing, exactly as the Python loop does, and counts what it dropped.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use fly_sim::brain::STIM_AMP;
use fly_sim::csr::Weights;
use fly_sim::lif::{LifParams, LifSim};
use fly_sim::pack::{Labels, Pack};
use fly_sim::{synthetic, Brain};

use crate::snapshot::{Mailbox, Phase, Snapshot};

/// Brain steps behind real time that are dropped rather than caught up (Python: `due - done > 40`).
const MAX_BACKLOG: u64 = 40;
/// Calm steps a new brain runs before it is shown (`simcore.new_brain(warmup=600)`).
pub const WARMUP_STEPS: usize = 600;

/// What to simulate.
#[derive(Debug, Clone)]
pub enum Source {
    /// The brain pack file (`data/kick_brain.npz`).
    Pack(std::path::PathBuf),
    /// A random network of `n` neurons: for tests and for running without the dataset.
    Synthetic { n: usize },
}

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
    DriveSpec { spec: String, amp: f32 },
    /// Hold these rows driven at `amp`.
    DriveRows { rows: Vec<u32>, amp: f32 },
    /// Real-time multiplier: 1 normal, below 1 slow motion, above 1 faster than real time.
    SetSpeed(f32),
    Pause(bool),
    /// While paused, run this many steps.
    StepOnce(u32),
    Kill,
    Shutdown,
}

/// The render side's handle to the sim thread. Dropping it shuts the thread down and waits for it.
pub struct SimHandle {
    tx: Sender<Command>,
    pub mailbox: Arc<Mailbox<Snapshot>>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl SimHandle {
    /// Send a command. Never blocks; returns false when the thread has already ended.
    pub fn send(&self, cmd: Command) -> bool {
        self.tx.send(cmd).is_ok()
    }

    /// The newest unread snapshot, if one arrived since the last call.
    pub fn latest(&self) -> Option<Box<Snapshot>> {
        self.mailbox.take()
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
    let join = {
        let (mailbox, stop) = (mailbox.clone(), stop.clone());
        thread::Builder::new()
            .name("fly-sim".into())
            .spawn(move || run(cfg, rx, mailbox, stop))
            .expect("could not start the sim thread")
    };
    SimHandle { tx, mailbox, stop, join: Some(join) }
}

fn publish_phase(mailbox: &Mailbox<Snapshot>, phase: Phase, source: &str, neurons: usize) {
    mailbox.publish(Snapshot::default, |s| {
        s.phase = phase;
        s.source = source.to_string();
        s.neurons = neurons;
    });
}

fn load(cfg: &SimConfig) -> Result<(Arc<Weights>, Arc<Labels>, String), String> {
    match &cfg.source {
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

fn run(cfg: SimConfig, rx: Receiver<Command>, mailbox: Arc<Mailbox<Snapshot>>, stop: Arc<AtomicBool>) {
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
    publish_phase(&mailbox, Phase::WarmingUp, &source, n);
    let params = LifParams::default();
    let dt = Duration::from_secs_f64(params.dt_ms / 1000.0);
    let mut brain = Brain::new(LifSim::new(weights, params, cfg.seed), labels);
    brain.warmup(cfg.warmup);

    let mut state = LoopState { speed: 1.0, paused: false, pending_steps: 0 };
    let mut t0 = Instant::now();
    let mut done: u64 = 0;
    let (mut rate_t, mut rate_steps, mut steps_per_s) = (Instant::now(), brain.steps, 0.0f32);
    let mut dropped: u64 = 0;
    let mut driving = false;
    publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving);

    while !stop.load(Ordering::Relaxed) {
        let before = (state.speed, state.paused);
        loop {
            match rx.try_recv() {
                Ok(Command::Shutdown) | Err(TryRecvError::Disconnected) => return,
                Ok(cmd) => driving |= apply(&mut brain, &mut state, cmd),
                Err(TryRecvError::Empty) => break,
            }
        }
        if (state.speed, state.paused) != before {
            t0 = Instant::now(); // re-anchor the clock when the speed changes, as Brain._loop does
            done = 0;
            publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving);
        }
        if state.pending_steps > 0 {
            for _ in 0..state.pending_steps {
                brain.step();
            }
            state.pending_steps = 0;
            publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving);
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
        if rate_t.elapsed() >= Duration::from_secs(1) {
            steps_per_s = (brain.steps - rate_steps) as f32 / rate_t.elapsed().as_secs_f32();
            rate_t = Instant::now();
            rate_steps = brain.steps;
        }
        if brain.steps.is_multiple_of(cfg.publish_every) {
            publish(&mailbox, &brain, &source, &state, steps_per_s, dropped, driving);
        }
    }
}

struct LoopState {
    speed: f32,
    paused: bool,
    pending_steps: u32,
}

/// Returns true when the command turned a stimulus on.
fn apply(brain: &mut Brain, state: &mut LoopState, cmd: Command) -> bool {
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
        Command::Shutdown => false,
    }
}

fn publish(mailbox: &Mailbox<Snapshot>, brain: &Brain, source: &str, state: &LoopState, steps_per_s: f32, dropped: u64, driving: bool) {
    let sim = &brain.sim;
    mailbox.publish(Snapshot::default, |s| {
        s.phase = if state.paused { Phase::Paused } else { Phase::Running };
        s.source.clear();
        s.source.push_str(source);
        s.neurons = sim.n;
        s.step = brain.steps;
        s.sim_time_s = brain.steps as f64 * sim.p.dt_ms / 1000.0;
        s.steps_per_s = steps_per_s;
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
    });
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
