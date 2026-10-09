//! Kill cam: on death, a slow-motion replay of the last ~6 s of the fly's brain, with the neurons whose firing rose most before death
//! highlighted. Python: `core/killcam.py`. No engine here: the game draws it.
//!
//! CONNECTOME: what is replayed and highlighted is every neuron's own firing rate (the simulator's per-neuron rate estimate, sampled
//! every `SAMPLE_STEPS` steps). "Rose most" is each neuron's mean rate over the last `TAIL_S` before death minus its mean over the first
//! `BASE_S` of the window. GAME RULE: the offer, the window length, the slow-motion speed, how many neurons are highlighted, the minimum
//! rise that counts. The replay is frames in memory and never touches the live brain.

pub const WINDOW_S: f32 = 6.0;
/// Sim steps between samples (40 ms, 25 per second).
pub const SAMPLE_STEPS: u64 = 8;
pub const DT: f32 = 0.005;
pub const BASE_S: f32 = 1.5;
pub const TAIL_S: f32 = 1.5;
pub const TOP_N: usize = 12;
pub const MIN_RISE_HZ: f32 = 5.0;
pub const SPEED: f32 = 0.25;
pub const SPEED_REDUCED: f32 = 0.2;
pub const MAX_HZ: f32 = 255.0;

/// Per-neuron rates (spikes per step) as `u8`: sqrt-companded Hz, fine near calm rates.
pub fn encode_rates(rates_per_step: &[f32], out: &mut [u8]) {
    for (o, r) in out.iter_mut().zip(rates_per_step) {
        let hz = (r / DT).clamp(0.0, MAX_HZ);
        *o = ((hz / MAX_HZ).sqrt() * 255.0).round() as u8;
    }
}

pub fn decode_hz(q: u8) -> f32 {
    MAX_HZ * (f32::from(q) / 255.0).powi(2)
}

/// A ring of the last `WINDOW_S` seconds of per-neuron rates for one brain. Feed it `push(step, rates)` every frame; it keeps one sample
/// per `SAMPLE_STEPS` brain steps, so slow motion and pauses add nothing.
pub struct Buffer {
    n: usize,
    cap: usize,
    q: Vec<u8>,
    steps: Vec<u64>,
    count: usize,
    last_step: Option<u64>,
}

impl Buffer {
    pub fn new(n_neurons: usize) -> Self {
        let cap = (WINDOW_S / (SAMPLE_STEPS as f32 * DT)).round() as usize;
        Self { n: n_neurons, cap, q: vec![0; cap * n_neurons], steps: vec![0; cap], count: 0, last_step: None }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.last_step = None;
    }

    pub fn len(&self) -> usize {
        self.count.min(self.cap)
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Would `push` store a sample at this step? (Lets the caller skip work on the frames in between.)
    pub fn due(&self, step: u64) -> bool {
        self.last_step.is_none_or(|l| step < l || step - l >= SAMPLE_STEPS)
    }

    /// Store a sample if the brain has advanced `SAMPLE_STEPS`. A step count that goes backwards clears the buffer first.
    pub fn push(&mut self, step: u64, rates_per_step: &[f32]) -> bool {
        if let Some(l) = self.last_step {
            if step < l {
                self.reset();
            } else if step - l < SAMPLE_STEPS {
                return false;
            }
        }
        let i = self.count % self.cap;
        encode_rates(rates_per_step, &mut self.q[i * self.n..(i + 1) * self.n]);
        self.steps[i] = step;
        self.count += 1;
        self.last_step = Some(step);
        true
    }

    /// The buffered samples, oldest first, as an independent [`Replay`]. `None` if there are too few to show anything.
    pub fn freeze(&self) -> Option<Replay> {
        let n = self.len();
        if n < 8 {
            return None;
        }
        let order: Vec<usize> = (self.count - n..self.count).map(|k| k % self.cap).collect();
        let last = self.steps[*order.last()?];
        let keep: Vec<usize> = order.into_iter().filter(|&i| (last - self.steps[i]) as f32 * DT <= WINDOW_S + 1e-6).collect();
        if keep.len() < 8 {
            return None;
        }
        let mut q = Vec::with_capacity(keep.len() * self.n);
        let mut steps = Vec::with_capacity(keep.len());
        for &i in &keep {
            q.extend_from_slice(&self.q[i * self.n..(i + 1) * self.n]);
            steps.push(self.steps[i]);
        }
        Some(Replay { n: self.n, q, steps })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Riser {
    pub index: usize,
    pub rise_hz: f32,
    pub before_hz: f32,
    pub after_hz: f32,
}

#[derive(Debug, Clone)]
pub struct Replay {
    pub n: usize,
    q: Vec<u8>,
    pub steps: Vec<u64>,
}

impl Replay {
    pub fn n_frames(&self) -> usize {
        self.steps.len()
    }

    /// Brain time from the first frame to the last, from the recorded step numbers.
    pub fn seconds(&self) -> f32 {
        if self.n_frames() > 1 { (self.steps[self.n_frames() - 1] - self.steps[0]) as f32 * DT } else { 0.0 }
    }

    fn rel(&self, i: usize) -> f32 {
        (self.steps[i] - self.steps[0]) as f32 * DT
    }

    /// The frame showing at `t` seconds into the replay.
    pub fn frame_at(&self, t: f32) -> usize {
        (0..self.n_frames()).take_while(|&i| self.rel(i) <= t + 1e-6).count().saturating_sub(1)
    }

    pub fn hz(&self, frame: usize, neuron: usize) -> f32 {
        decode_hz(self.q[frame.min(self.n_frames() - 1) * self.n + neuron])
    }

    /// The frame as spikes per step (what the brain view takes).
    pub fn rates_per_step(&self, frame: usize, out: &mut Vec<f32>) {
        let f = frame.min(self.n_frames() - 1);
        out.clear();
        out.extend(self.q[f * self.n..(f + 1) * self.n].iter().map(|&q| decode_hz(q) * DT));
    }

    /// The neurons whose firing rose most before death, biggest first. The baseline is the first `BASE_S` of the window and the tail the
    /// last `TAIL_S`, both clipped to a third of the window for a short replay. Ties break by neuron index.
    pub fn top_risers(&self, k: usize, min_rise_hz: f32) -> Vec<Riser> {
        let n = self.n_frames();
        let last = self.rel(n - 1);
        let nb = (0..n).filter(|&i| self.rel(i) < BASE_S).count().clamp(1, (n / 3).max(1));
        let nt = (0..n).filter(|&i| last - self.rel(i) <= TAIL_S).count().clamp(1, (n / 3).max(1));
        let mean = |frames: std::ops::Range<usize>, j: usize| frames.clone().map(|f| decode_hz(self.q[f * self.n + j])).sum::<f32>() / frames.len() as f32;
        let mut all: Vec<Riser> = (0..self.n)
            .map(|j| {
                let (before, after) = (mean(0..nb, j), mean(n - nt..n, j));
                Riser { index: j, rise_hz: after - before, before_hz: before, after_hz: after }
            })
            .collect();
        all.sort_by(|a, b| b.rise_hz.total_cmp(&a.rise_hz).then(a.index.cmp(&b.index)));
        all.into_iter().take(k).take_while(|r| r.rise_hz >= min_rise_hz).collect()
    }

    /// Firing rates in Hz for one neuron over the replay, for the timeline.
    pub fn trace(&self, neuron: usize) -> Vec<f32> {
        (0..self.n_frames()).map(|f| self.hz(f, neuron)).collect()
    }
}

/// The playback clock. `advance(wall_dt)` moves the replay by `wall_dt * speed` seconds of brain time; `done` becomes true at the last
/// frame (the moment of death). Skippable at any time.
pub struct Player {
    pub replay: Replay,
    pub speed: f32,
    pub t: f32,
    pub done: bool,
    pub skipped: bool,
}

impl Player {
    pub fn new(replay: Replay, speed: f32) -> Self {
        Self { replay, speed, t: 0.0, done: false, skipped: false }
    }

    pub fn frame(&self) -> usize {
        self.replay.frame_at(self.t)
    }

    pub fn progress(&self) -> f32 {
        if self.replay.seconds() <= 0.0 { 0.0 } else { (self.t / self.replay.seconds()).min(1.0) }
    }

    pub fn wall_seconds(&self) -> f32 {
        self.replay.seconds() / self.speed.max(1e-6)
    }

    pub fn advance(&mut self, wall_dt: f32) {
        if self.done {
            return;
        }
        self.t += wall_dt.max(0.0) * self.speed;
        if self.t >= self.replay.seconds() {
            self.t = self.replay.seconds();
            self.done = true;
        }
    }

    pub fn skip(&mut self) {
        self.t = self.replay.seconds();
        self.done = true;
        self.skipped = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 neurons: #0 calm, #1 rises from 2 Hz to 60 Hz, #2 falls.
    fn buffer_with_a_riser(steps: u64) -> Buffer {
        let mut b = Buffer::new(3);
        for step in 0..steps {
            let f = step as f32 / steps as f32;
            let hz = [10.0, 2.0 + 58.0 * f.powi(8), 40.0 * (1.0 - f)];
            let r: Vec<f32> = hz.iter().map(|h| h * DT).collect();
            b.push(step, &r);
        }
        b
    }

    #[test]
    fn rates_round_trip_through_the_companded_byte_within_a_few_percent() {
        for hz in [0.5f32, 2.0, 10.0, 60.0, 200.0] {
            let mut q = [0u8];
            encode_rates(&[hz * DT], &mut q);
            let back = decode_hz(q[0]);
            assert!((back - hz).abs() / hz < 0.06 || (back - hz).abs() < 0.3, "{hz} -> {back}");
        }
        let mut q = [0u8];
        encode_rates(&[10.0], &mut q);
        assert_eq!(decode_hz(q[0]), MAX_HZ, "clamped at the top");
    }

    #[test]
    fn it_keeps_a_sample_every_8_steps_and_only_the_last_6_seconds() {
        let b = buffer_with_a_riser(3000); // 15 s of brain time
        assert_eq!(b.len(), 150);
        let r = b.freeze().unwrap();
        assert!(r.seconds() <= WINDOW_S + 0.05 && r.seconds() > 5.8, "{}", r.seconds());
        assert_eq!(r.steps[1] - r.steps[0], SAMPLE_STEPS);
        let mut small = Buffer::new(3);
        small.push(0, &[0.0; 3]);
        assert!(small.freeze().is_none(), "too short to show anything");
        assert!(!small.push(3, &[0.0; 3]) && small.due(8));
    }

    #[test]
    fn the_biggest_riser_is_found_and_calm_or_falling_neurons_are_not_listed() {
        let r = buffer_with_a_riser(3000).freeze().unwrap();
        let top = r.top_risers(TOP_N, MIN_RISE_HZ);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].index, 1);
        assert!(top[0].rise_hz > 30.0 && top[0].before_hz < 5.0);
        assert_eq!(r.trace(1).len(), r.n_frames());
    }

    #[test]
    fn a_step_counter_that_goes_backwards_clears_the_buffer() {
        let mut b = buffer_with_a_riser(400);
        assert!(b.len() > 10);
        assert!(b.push(5, &[0.0; 3]));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn playback_runs_at_a_quarter_speed_stops_at_the_death_and_can_be_skipped() {
        let r = buffer_with_a_riser(3000).freeze().unwrap();
        let secs = r.seconds();
        let mut p = Player::new(r, SPEED);
        assert!((p.wall_seconds() - secs * 4.0).abs() < 1e-3);
        for _ in 0..60 {
            p.advance(1.0 / 60.0);
        }
        assert!((p.t - 0.25).abs() < 0.01 && !p.done);
        assert!(p.frame() < p.replay.n_frames() / 4);
        p.skip();
        assert!(p.done && p.skipped && p.frame() == p.replay.n_frames() - 1 && p.progress() == 1.0);
    }
}
