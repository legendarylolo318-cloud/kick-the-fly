//! Rolling activity for the UI: per-neuron EMA firing rate, the recent spike raster, and the step of each neuron's latest spike.
//! Python: `sim.py:ActivityBuffer`.

/// Single-threaded by design: the sim thread owns it and publishes copies (snapshots) to readers.
#[derive(Debug, Clone)]
pub struct ActivityBuffer {
    pub n: usize,
    /// EMA of spikes per step (time constant 40 steps = 200 ms).
    pub(crate) rate: Vec<f32>,
    pub(crate) decay: f32,
    pub(crate) bump: f32,
    /// Step index of each neuron's latest spike (`i32::MIN / 2` = never).
    pub(crate) last_spike: Vec<i32>,
    raster: Vec<Vec<u32>>,
    raster_head: usize,
    raster_len: usize,
    pub active_count: usize,
    /// Steps pushed so far.
    pub steps: u64,
}

const NEVER: i32 = -(1 << 30);

impl ActivityBuffer {
    pub fn new(n: usize) -> Self {
        Self::with_shape(n, 400, 40.0)
    }

    /// `raster_steps` recent steps are kept; `rate_tau_steps` is the EMA time constant.
    pub fn with_shape(n: usize, raster_steps: usize, rate_tau_steps: f64) -> Self {
        let decay = (-1.0f64 / rate_tau_steps).exp() as f32; // NumPy: np.float32(np.exp(-1.0 / tau))
        Self {
            n,
            rate: vec![0.0; n],
            decay,
            bump: 1.0 - decay,
            last_spike: vec![NEVER; n],
            raster: vec![Vec::new(); raster_steps.max(1)],
            raster_head: 0,
            raster_len: 0,
            active_count: 0,
            steps: 0,
        }
    }

    /// Record one step's spike list (ascending neuron indices). The per-neuron arrays were already updated by the step itself.
    pub(crate) fn finish_step(&mut self, spikes: &[u32]) {
        let slot = &mut self.raster[self.raster_head];
        slot.clear();
        slot.extend_from_slice(spikes);
        self.raster_head = (self.raster_head + 1) % self.raster.len();
        self.raster_len = (self.raster_len + 1).min(self.raster.len());
        self.active_count = spikes.len();
        self.steps += 1;
    }

    /// Step index a spike pushed *now* is stamped with (the number of steps already pushed).
    pub(crate) fn stamp(&self) -> i32 {
        self.steps as i32
    }

    /// EMA firing rate per neuron, spikes per step.
    pub fn rates(&self) -> &[f32] {
        &self.rate
    }

    /// The recent spike lists, oldest first.
    pub fn raster(&self) -> impl Iterator<Item = &[u32]> {
        let cap = self.raster.len();
        let start = (self.raster_head + cap - self.raster_len) % cap;
        (0..self.raster_len).map(move |k| self.raster[(start + k) % cap].as_slice())
    }

    /// Neurons that spiked within the last `steps` steps.
    pub fn active_within(&self, steps: u64) -> usize {
        let limit = self.steps as i64 - 1 - steps as i64;
        self.last_spike.iter().filter(|&&s| i64::from(s) > limit).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_keeps_the_last_steps_in_order() {
        let mut a = ActivityBuffer::with_shape(4, 3, 40.0);
        for s in 0..5u32 {
            a.finish_step(&[s]);
        }
        let got: Vec<Vec<u32>> = a.raster().map(<[u32]>::to_vec).collect();
        assert_eq!(got, vec![vec![2], vec![3], vec![4]]);
        assert_eq!(a.steps, 5);
        assert_eq!(a.active_count, 1);
    }
}
