//! The two Play-mode challenges that run on the brain's own clock (`lab/challenges.py`): the T-maze ("Teach it to pick the right door") and the
//! sweet-tooth dose offers. Timed in brain steps like Python's (which counted `br.steps`); the short pauses between choices and offers use the
//! wall clock the game passes in, as Python's did.
//!
//! The T-maze trains the fly's real mushroom body synapses and puts them back as they were when it ends (the caller takes a
//! [`crate::memory::Saved`] before and restores it after), so a practice round never changes how the fly treats the player's tools.

use std::collections::BTreeMap;

use crate::assays::PER_RATIO;
use crate::brain::Brain;
use crate::rng::Generator;

pub const CYCLES: u32 = 6;
pub const TRIALS: usize = 10;
/// Brain speed while the fly is being trained and tested.
pub const TMAZE_SPEED: f32 = 3.0;
pub const SWEET_TRIES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TPhase {
    Naive,
    Train,
    Test,
    Done,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TEvent {
    None,
    /// A door was chosen (the game plays the sound and starts the walk animation).
    Chose {
        right: bool,
    },
    /// Ten choices made.
    Finished {
        right: usize,
    },
}

pub struct TMaze {
    pub zap: String,
    pub phase: TPhase,
    pub stage: u32,
    pub cycle: u32,
    until: u64,
    drive: BTreeMap<String, f32>,
    pub choices: Vec<(String, bool)>,
    rng: Generator,
    /// Wall time (seconds) of the latest choice, for the walk animation.
    pub anim_at: f64,
}

/// `schedule` of the training phase: (smell, shock, steps).
const SCHEDULE: [(Option<bool>, bool, u64); 5] =
    [(Some(true), false, 240), (Some(true), true, 200), (None, false, 260), (Some(false), false, 440), (None, false, 260)];

fn other(zap: &str) -> &'static str {
    if zap == "odor_a" { "odor_b" } else { "odor_a" }
}

impl TMaze {
    /// `TMaze.pick`: `zap` is `odor_a` or `odor_b`. None when the brain has no memory.
    pub fn start(brain: &Brain, zap: &str, seed: u64) -> Option<Self> {
        brain.memory.as_ref()?;
        Some(TMaze {
            zap: zap.into(),
            phase: TPhase::Naive,
            stage: 0,
            cycle: 0,
            until: brain.steps + 240,
            drive: BTreeMap::new(),
            choices: vec![],
            rng: Generator::new(seed.wrapping_add(424_242)),
            anim_at: 0.0,
        })
    }

    pub fn speed(&self) -> f32 {
        if self.phase == TPhase::Done { 1.0 } else { TMAZE_SPEED }
    }

    fn smell(&self, brain: &mut Brain, odor: &str) {
        let step = brain.steps;
        if step.is_multiple_of(3) {
            brain.poke(&format!("scent:{odor}"), None, 0.5);
        }
        if step.is_multiple_of(10) {
            let rates = brain.sim.activity.rates().to_vec();
            if let Some(m) = brain.memory.as_mut() {
                m.observe(odor, &rates);
            }
        }
    }

    /// One brain step (call after `brain.step()`); `now` is wall-clock seconds.
    pub fn tick(&mut self, brain: &mut Brain, now: f64) -> TEvent {
        if brain.memory.is_none() || self.phase == TPhase::Done {
            return TEvent::None;
        }
        let plus = self.zap.clone();
        let minus = other(&self.zap).to_string();
        let step = brain.steps;
        match self.phase {
            TPhase::Naive => {
                let odor = if self.stage == 0 { &plus } else { &minus };
                self.smell(brain, odor);
                if step >= self.until {
                    self.stage += 1;
                    self.until = step + 240;
                    if self.stage == 2 {
                        self.phase = TPhase::Train;
                        self.stage = 0;
                        self.cycle = 0;
                        self.until = step + 240;
                    }
                }
            }
            TPhase::Train => {
                let (which, shock, _) = SCHEDULE[self.stage as usize];
                if let Some(is_plus) = which {
                    let odor = if is_plus { &plus } else { &minus };
                    self.smell(brain, odor);
                }
                if shock && step.is_multiple_of(3) {
                    brain.poke("punish", None, 1.0);
                    brain.poke("legs", Some('L'), 0.6);
                    brain.poke("legs", Some('R'), 0.6);
                }
                if step >= self.until {
                    self.stage = (self.stage + 1) % SCHEDULE.len() as u32;
                    if self.stage == 0 {
                        self.cycle += 1;
                        if self.cycle >= CYCLES {
                            self.phase = TPhase::Test;
                            self.stage = 0;
                            self.choices.clear();
                        }
                    }
                    self.until = step + SCHEDULE[self.stage as usize].2;
                }
            }
            TPhase::Test => {
                let (first, second) = if self.choices.len().is_multiple_of(2) { (&plus, &minus) } else { (&minus, &plus) };
                if self.stage < 2 {
                    let odor = if self.stage == 0 { first } else { second }.clone();
                    self.smell(brain, &odor);
                    if step >= self.until {
                        let (fear, like) = brain.memory.as_ref().map_or((0.0, 0.0), |m| m.memory_of(&odor));
                        self.drive.insert(odor, like - fear);
                        self.stage += 1;
                        self.until = step + if self.stage < 2 { 60 } else { 1 };
                    }
                } else if self.stage == 2 {
                    let noise = 0.08;
                    let d = |k: &str, s: &Self| f64::from(*s.drive.get(k).unwrap_or(&0.0));
                    let a = d(&plus, self) + self.rng.normal(0.0, noise);
                    let b = d(&minus, self) + self.rng.normal(0.0, noise);
                    let pick_plus = a > b;
                    let door = if pick_plus { plus } else { minus };
                    self.choices.push((door, !pick_plus));
                    self.anim_at = now;
                    self.stage = 3;
                    return TEvent::Chose { right: !pick_plus };
                } else if now - self.anim_at > 0.9 {
                    if self.choices.len() >= TRIALS {
                        self.phase = TPhase::Done;
                        let right = self.choices.iter().filter(|c| c.1).count();
                        return TEvent::Finished { right };
                    }
                    self.stage = 0;
                    self.until = step + 60;
                }
            }
            TPhase::Done => {}
        }
        TEvent::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SPhase {
    Ready,
    Before,
    During,
    Shown,
    Done,
}

/// "Find its sweet tooth": offer a dose of sugar-pathway neurons 8 times; the weakest one MN9 answers wins.
pub struct Sweet {
    pub phase: SPhase,
    pub dose: u32,
    /// (dose, extended, ratio) per offer.
    pub offers: Vec<(u32, bool, f32)>,
    before: Vec<f32>,
    during: Vec<f32>,
    until: u64,
    shown_at: f64,
    /// The weakest dose it reached for, once the 8 tries are done.
    pub score: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SEvent {
    None,
    /// An offer was answered: (dose, extended, ratio).
    Answer(u32, bool, f32),
    Finished(Option<u32>),
}

impl Sweet {
    pub fn new() -> Self {
        Sweet { phase: SPhase::Ready, dose: 50, offers: vec![], before: vec![], during: vec![], until: 0, shown_at: 0.0, score: None }
    }

    pub fn offer(&mut self, brain: &Brain, dose: u32) {
        if self.phase != SPhase::Ready {
            return;
        }
        self.dose = dose.min(100);
        self.phase = SPhase::Before;
        self.before.clear();
        self.during.clear();
        self.until = brain.steps + 200;
    }

    pub fn tick(&mut self, brain: &mut Brain, now: f64) -> SEvent {
        let step = brain.steps;
        match self.phase {
            SPhase::Before => {
                if step.is_multiple_of(3) {
                    self.before.push(brain.hz("proboscis"));
                }
                if step >= self.until {
                    self.phase = SPhase::During;
                    self.until = step + 200;
                }
            }
            SPhase::During => {
                if step.is_multiple_of(3) {
                    if self.dose > 0 {
                        brain.poke_recruit("sweet", None, 0.5, Some(self.dose as f32 / 100.0));
                    }
                    self.during.push(brain.hz("proboscis"));
                }
                if step >= self.until {
                    let mean = |v: &[f32]| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
                    let ratio = mean(&self.during) / mean(&self.before).max(1.0);
                    let ext = ratio >= PER_RATIO;
                    self.offers.push((self.dose, ext, ratio));
                    self.phase = SPhase::Shown;
                    self.shown_at = now;
                    return SEvent::Answer(self.dose, ext, ratio);
                }
            }
            SPhase::Shown => {
                if now - self.shown_at > 1.4 {
                    if self.offers.len() >= SWEET_TRIES {
                        self.phase = SPhase::Done;
                        self.score = self.offers.iter().filter(|o| o.1).map(|o| o.0).min();
                        return SEvent::Finished(self.score);
                    }
                    self.phase = SPhase::Ready;
                }
            }
            SPhase::Ready | SPhase::Done => {}
        }
        SEvent::None
    }
}

impl Default for Sweet {
    fn default() -> Self {
        Self::new()
    }
}
