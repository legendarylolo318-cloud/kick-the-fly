//! What the sim thread tells the render thread, and the mailbox it travels through.
//!
//! The render thread must never wait for the simulation and the simulation must never wait for rendering. The [`Mailbox`] is a
//! one-slot "latest value" box with a pool of recycled buffers: the sim thread fills a free buffer (outside any lock), then swaps
//! it into the slot while holding a lock for the length of a pointer move; the render thread takes whatever is newest, and hands the
//! buffer it is done with back to the pool. A frame that arrives while the renderer is busy simply replaces the older one.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// Where the sim thread is in its life.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Phase {
    /// Reading the brain pack from disk.
    #[default]
    Loading,
    /// Warming the brain up (600 calm steps), then running.
    WarmingUp,
    Running,
    Paused,
    /// The pack could not be loaded; the message says why. The thread keeps answering commands and exits on shutdown.
    Failed(String),
}

/// What never changes while a brain runs: positions and labels for the brain views, and the names of the group readouts.
#[derive(Debug, Clone, Default)]
pub struct BrainInfo {
    /// Cell-body position per neuron (NaN where the dataset has none).
    pub soma: Vec<[f32; 3]>,
    /// Where each neuron is drawn, normalised to about [-1, 1] (x right, y up, z depth). A neuron without a cell body in the data sits at
    /// the synapse-weighted centre of its partners (`BrainView.__init__`'s `A`); NaN when it has no placed partner either.
    pub layout: Vec<[f32; 3]>,
    /// Population index (0..9, `fly_sim::groups` POPS order) per neuron, or 255 for none.
    pub population: Vec<u8>,
    /// Names of the entries of [`Snapshot::levels`] and [`Snapshot::hz`].
    pub group_names: Vec<String>,
    pub population_names: Vec<String>,
    /// The neuropil region of each neuron (index into `region_names`), for the sonification's voices.
    pub region_id: Vec<u8>,
    pub region_names: Vec<String>,
    /// Cell type of each neuron (for the kill cam's "which neurons rose" list and the lesion picker).
    pub types: Vec<String>,
}

/// A copy of the sim's state for drawing. Plain data, cheap to hold on to.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub phase: Phase,
    /// Which network is running: the pack's name or "synthetic".
    pub source: String,
    pub neurons: usize,
    /// Brain steps since the thread started warming up.
    pub step: u64,
    /// Simulated seconds (steps x dt).
    pub sim_time_s: f64,
    /// Measured brain steps per wall-clock second, over the last second.
    pub steps_per_s: f32,
    /// Wall time of the latest brain step, milliseconds.
    pub step_ms: f32,
    /// Steps the thread dropped to catch up after falling behind (it never races to recover more than 40 at once).
    pub steps_dropped: u64,
    pub gain: f64,
    /// Neurons that spiked in the latest step.
    pub active_count: usize,
    /// The neurons that spiked in the latest step, ascending.
    pub spikes: Vec<u32>,
    /// EMA firing rate per neuron, spikes per step.
    pub rates: Vec<f32>,
    /// Whether a held stimulus is on, and whether the fly is dead.
    pub driving: bool,
    pub dead: bool,
    pub speed: f32,
    /// Per group (`BrainInfo::group_names`): firing in Hz per neuron, and as a multiple of the calm baseline.
    pub hz: Vec<f32>,
    pub levels: Vec<f32>,
    /// `Brain.pain_parts` now: touch overload, thermal, chemical, DN alarm, body relay (0..1).
    pub pain_parts: [f32; 5],
    /// How many real neurons the pain index listens to at this pain level.
    pub pain_neurons: u32,
    /// Pain level 0..2.
    pub pain_level: u8,
    /// Whole-brain rate, spikes/s per neuron, the last 600 samples of 20 ms (12 s), oldest first (the WHOLE BRAIN trace).
    pub whole_trace: Vec<f32>,
    pub info: Option<Arc<BrainInfo>>,
    /// When the sim thread made this snapshot, to measure how stale the render side's view is.
    pub published_at: Option<Instant>,
}

/// One-slot latest-value box with buffer recycling. See the module docs.
pub struct Mailbox<T> {
    latest: Mutex<Option<Box<T>>>,
    free: Mutex<Vec<Box<T>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<T> Default for Mailbox<T> {
    fn default() -> Self {
        Self { latest: Mutex::new(None), free: Mutex::new(Vec::new()) }
    }
}

impl<T> Mailbox<T> {
    /// Publish a new value: `fill` writes into a recycled (or `make`d) buffer, which then replaces whatever was waiting.
    pub fn publish(&self, make: impl FnOnce() -> T, fill: impl FnOnce(&mut T)) {
        let recycled = lock(&self.free).pop();
        let mut buf = recycled.unwrap_or_else(|| Box::new(make()));
        fill(&mut buf);
        let displaced = lock(&self.latest).replace(buf);
        if let Some(old) = displaced {
            lock(&self.free).push(old);
        }
    }

    /// The newest unread value, if any.
    pub fn take(&self) -> Option<Box<T>> {
        lock(&self.latest).take()
    }

    /// Give a finished buffer back for reuse.
    pub fn recycle(&self, buf: Box<T>) {
        let mut free = lock(&self.free);
        if free.len() < 4 {
            free.push(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn newest_value_wins_and_buffers_are_reused() {
        let m = Mailbox::<Vec<u32>>::default();
        m.publish(Vec::new, |v| v.push(1));
        m.publish(Vec::new, |v| {
            v.clear();
            v.push(2);
        });
        let newest = m.take().unwrap();
        assert_eq!(*newest, vec![2], "an unread older value is replaced");
        assert!(m.take().is_none(), "a value is delivered once");
        m.recycle(newest);
        m.publish(|| unreachable!("a recycled buffer is reused"), |v| v.push(3));
        assert_eq!(*m.take().unwrap(), vec![2, 3]);
    }

    #[test]
    fn publisher_is_never_blocked_by_a_slow_reader() {
        let m = Arc::new(Mailbox::<u64>::default());
        let writer = {
            let m = m.clone();
            std::thread::spawn(move || {
                for i in 0..10_000u64 {
                    m.publish(|| 0, |v| *v = i);
                }
            })
        };
        writer.join().unwrap();
        assert_eq!(*m.take().unwrap(), 9_999);
    }
}
