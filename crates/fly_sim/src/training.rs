//! Lab-style conditioning on the mushroom body (`Game.start_training` / `_training_tick` in `kick_the_fly.py`): a smell is paired with a
//! shock (fear) or with sugar (liking) while the dopamine neurons teach the fly's real Kenyon cell -> output neuron synapses.
//!
//! Timed in brain steps (5 ms each) so it runs the same at any speed: smell 1.2 s -> smell + shock or sugar 1.0 s -> rest 1.3 s, repeated;
//! a test is 2 s of the smell alone. The first fear/liking run on a smell first measures the untrained response (2 s smell, 1.3 s rest).
//! It runs on the brain's own thread, so the timing does not depend on the frame rate (Python ticked it per frame).
//!
//! One deliberate difference from Python: pressing TEST there started in the `odor` phase, so it ran the *pairing* branch of the liking
//! route (`kind != "fear"` -> reward + taste) and trained liking; here a test is the smell alone, as its docstring says.

use crate::brain::Brain;

/// Steps of each phase (Python's `until` offsets).
pub const NAIVE_STEPS: u64 = 400;
pub const ODOR_STEPS: u64 = 240;
pub const PAIR_STEPS: u64 = 200;
pub const REST_STEPS: u64 = 260;
pub const TEST_STEPS: u64 = 400;
/// The smell is poked every this many steps (Python: every frame, about every 3.3 steps at 60 fps).
pub const POKE_EVERY: u64 = 3;
/// Pattern sampling every this many steps (Python: every 3rd frame).
pub const OBSERVE_EVERY: u64 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Fear,
    Like,
    Test,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Fear => "fear",
            Kind::Like => "like",
            Kind::Test => "test",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "fear" => Kind::Fear,
            "like" => Kind::Like,
            "test" => Kind::Test,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Naive,
    Odor,
    Pair,
    Rest,
    Test,
}

impl Phase {
    /// What the progress line says.
    pub fn describe(self, kind: Kind) -> String {
        match self {
            Phase::Naive => "measuring the untrained response".into(),
            Phase::Odor => "smell".into(),
            Phase::Pair => format!("smell + {}", if kind == Kind::Fear { "shock" } else { "sugar" }),
            Phase::Rest => "rest".into(),
            Phase::Test => "testing: smell alone".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Training {
    pub scent: String,
    pub kind: Kind,
    pub trials: u32,
    pub done: u32,
    pub phase: Phase,
    until: u64,
    hz: Vec<f32>,
    odor_hz: Option<f32>,
    /// The untrained response is measured before the first trial.
    first: bool,
}

/// What a tick tells the game.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Running,
    /// Finished or stopped: the game saves the memory.
    Done,
}

impl Training {
    /// `start_training`: None when the brain has no memory. A trial count of 0 means 10.
    pub fn start(brain: &Brain, scent: &str, kind: Kind, trials: u32) -> Option<Self> {
        let mem = brain.memory.as_ref()?;
        let first = kind != Kind::Test && !mem.naive_mbon.contains_key(scent) && mem.log.get(scent).is_none_or(Vec::is_empty);
        let (phase, until) = match (first, kind) {
            (true, _) => (Phase::Naive, NAIVE_STEPS),
            (false, Kind::Test) => (Phase::Test, TEST_STEPS),
            (false, _) => (Phase::Odor, ODOR_STEPS),
        };
        Some(Training {
            scent: scent.to_string(),
            kind,
            trials: if kind == Kind::Test {
                1
            } else if trials == 0 {
                10
            } else {
                trials
            },
            done: 0,
            phase,
            until: brain.steps + until,
            hz: vec![],
            odor_hz: None,
            first,
        })
    }

    /// One brain step of the protocol (call after `brain.step()`).
    pub fn tick(&mut self, brain: &mut Brain, now_unix: f64) -> Outcome {
        let step = brain.steps;
        if matches!(self.phase, Phase::Naive | Phase::Odor | Phase::Pair | Phase::Test) {
            if step.is_multiple_of(POKE_EVERY) {
                brain.poke(&format!("scent:{}", self.scent), None, 0.5);
            }
            if step.is_multiple_of(OBSERVE_EVERY) {
                let rates = brain.sim.activity.rates().to_vec();
                if let Some(m) = brain.memory.as_mut() {
                    m.observe(&self.scent, &rates);
                    let tail = if self.phase == Phase::Odor { 180 } else { 300 };
                    if self.phase != Phase::Pair && step + tail > self.until {
                        self.hz.push(m.mbon_response(&rates));
                    }
                }
            }
        }
        if self.phase == Phase::Pair && step.is_multiple_of(POKE_EVERY) {
            if self.kind == Kind::Fear {
                brain.poke("punish", None, 1.0); // the punishment dopamine neurons
                brain.poke("legs", Some('L'), 0.6); // and an electric shock through the legs
                brain.poke("legs", Some('R'), 0.6);
            } else {
                brain.poke("reward", None, 0.8); // the reward dopamine neurons
                brain.poke("taste", None, 0.6); // and the taste of sugar
            }
        }
        if step < self.until {
            return Outcome::Running;
        }
        let hz = if self.hz.is_empty() { None } else { Some(self.hz.iter().sum::<f32>() / self.hz.len() as f32) };
        let scent = self.scent.clone();
        match self.phase {
            Phase::Naive => {
                if let (Some(h), Some(m)) = (hz, brain.memory.as_mut()) {
                    m.naive_mbon.insert(scent.clone(), f64::from(h));
                }
                if let Some(m) = brain.memory.as_mut() {
                    m.record(&scent, "naive", hz.map(f64::from), now_unix);
                }
                self.enter(Phase::Rest, step + REST_STEPS);
            }
            Phase::Odor => {
                self.odor_hz = hz;
                self.enter(Phase::Pair, step + PAIR_STEPS);
            }
            Phase::Pair => {
                self.done += 1;
                if let Some(m) = brain.memory.as_mut() {
                    m.record(&scent, self.kind.name(), self.odor_hz.map(f64::from), now_unix);
                }
                self.enter(Phase::Rest, step + REST_STEPS);
            }
            Phase::Test => {
                if let Some(m) = brain.memory.as_mut() {
                    m.record(&scent, "test", hz.map(f64::from), now_unix);
                }
                self.done = 1;
                self.enter(Phase::Rest, step + 1);
            }
            Phase::Rest => {
                if self.first {
                    self.first = false;
                    if self.kind == Kind::Test {
                        return Outcome::Done;
                    }
                    self.next_trial(step);
                } else if self.done >= self.trials {
                    return Outcome::Done;
                } else {
                    self.next_trial(step);
                }
            }
        }
        Outcome::Running
    }

    fn enter(&mut self, phase: Phase, until: u64) {
        self.phase = phase;
        self.until = until;
        self.hz.clear();
    }

    fn next_trial(&mut self, step: u64) {
        if self.kind == Kind::Test {
            self.enter(Phase::Test, step + TEST_STEPS);
        } else {
            self.enter(Phase::Odor, step + ODOR_STEPS);
        }
    }
}
