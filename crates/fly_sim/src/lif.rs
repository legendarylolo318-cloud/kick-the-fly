//! Leaky integrate-and-fire over the signed connectome: the Rust twin of `sim.py:LIFSim` with `backends.py:CPUBackend`.
//!
//! One step, in the order and `f32` arithmetic NumPy uses:
//!
//! ```text
//! i_syn  = W · spikes_prev            (rows summed in ascending presynaptic order)
//! drive  = i_syn * gain; drive += bias; drive += noise[off + i]; drive += ext_gain * sensory   (if any)
//! v      = v * (1 - leak) [+ leak * v_reset]; v += drive
//! refractory neurons: v = v_reset, counter -= 1
//! spike  = v >= v_thresh; spiking neurons: v = v_reset, counter = refractory_steps
//! gain  *= exp(gain_adapt * clip((target_p - fired / n) / target_p, -1, 1)), clipped to gain_bounds
//! ```
//!
//! A slow global controller (`gain`) holds the population rate near `target_rate_hz`. All of it runs in one rayon pass over row
//! blocks (see [`crate::csr`]): each task zeroes its slice of `i_syn`, scatters the active columns into it, and updates its
//! neurons. The result is bit-identical to the Python reference on any thread count.

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::activity::ActivityBuffer;
use crate::csr::Weights;
use crate::rng::Generator;

/// Model constants. Defaults are `LIFParams` in `sim.py`; they are the simulation, so they are not tuned here.
#[derive(Debug, Clone, PartialEq)]
pub struct LifParams {
    pub dt_ms: f64,
    pub tau_ms: f64,
    pub v_thresh: f64,
    pub v_reset: f64,
    pub refractory_steps: i16,
    /// Initial synaptic gain; adapted toward `target_rate_hz`.
    pub syn_gain: f64,
    pub ext_gain: f64,
    pub bias: f64,
    pub noise_std: f64,
    pub target_rate_hz: f64,
    pub gain_adapt: f64,
    pub gain_bounds: (f64, f64),
    /// Active fraction below which the column-gather path is used (the result is identical either way).
    pub sparse_path_max_active: f64,
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            dt_ms: 5.0,
            tau_ms: 20.0,
            v_thresh: 1.0,
            v_reset: 0.0,
            refractory_steps: 2,
            syn_gain: 6.0,
            ext_gain: 4.0,
            bias: 0.20,
            noise_std: 0.05,
            target_rate_hz: 5.0,
            gain_adapt: 0.002,
            gain_bounds: (1.5, 40.0),
            sparse_path_max_active: 0.10,
        }
    }
}

/// Per-neuron lognormal gains `D_pre`, `D_post` (`core/individuality.py`): `W_fly = D_post · W · D_pre`.
#[derive(Debug, Clone)]
pub struct Individuality {
    pub d_pre: Vec<f32>,
    pub d_post: Vec<f32>,
}

impl Individuality {
    /// `compute_fly_gains(seed, n, mode, sigma)` for a given sigma (0.05 subtle, 0.15 strong). `None` for sigma ≤ 1e-6.
    ///
    /// The normals match NumPy bit for bit; the final `exp` is `libm`'s, while NumPy's float32 `exp` is a SIMD routine that may
    /// differ in the last bit on some CPUs, so the gains agree to a few ulp (the golden test allows 1e-6 relative).
    pub fn compute(seed: u64, n: usize, sigma: f64) -> Option<Self> {
        if sigma <= 1e-6 {
            return None;
        }
        let gain_seed = seed.wrapping_mul(1_000_003).wrapping_add(7919) & 0xFFFF_FFFF;
        let mut rng = Generator::new(gain_seed);
        let lim = (3.0 * sigma) as f32;
        let draw = |rng: &mut Generator| -> Vec<f32> {
            (0..n).map(|_| (rng.normal(0.0, sigma) as f32).clamp(-lim, lim).exp()).collect()
        };
        let d_pre = draw(&mut rng);
        let d_post = draw(&mut rng);
        Some(Self { d_pre, d_post })
    }
}

/// One fly's simulator.
pub struct LifSim {
    pub p: LifParams,
    pub n: usize,
    weights: Arc<Weights>,
    pub v: Vec<f32>,
    pub refr: Vec<i16>,
    /// The synaptic gain controller's current value.
    pub gain: f64,
    /// Spikes of the latest step, ascending neuron indices.
    active: Vec<u32>,
    mask: Vec<bool>,
    i_syn: Vec<f32>,
    noise: Vec<f32>,
    rng: Generator,
    leak: f32,
    target_p: f64,
    pub activity: ActivityBuffer,
    pub spike_total: u64,
    pub steps: u64,
    /// Wall time of the latest step, milliseconds.
    pub last_step_ms: f64,
    pub path_counts: PathCounts,
    indiv: Option<Individuality>,
    block_spikes: Vec<Vec<u32>>,
    block_size: usize,
}

/// How often each propagation path ran (the Python `path_counts`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PathCounts {
    pub columns: u64,
    pub full: u64,
}

impl LifSim {
    /// A fresh sim: `v = 0`, the noise bank drawn from `default_rng(seed)` like `LIFSim.__init__`.
    pub fn new(weights: Arc<Weights>, params: LifParams, seed: u64) -> Self {
        let n = weights.n();
        let mut rng = Generator::new(seed);
        let mut noise = vec![0f32; n * 16];
        rng.fill_standard_normal_f32(&mut noise);
        let std = params.noise_std as f32;
        for x in &mut noise {
            *x *= std;
        }
        let leak = (params.dt_ms / params.tau_ms) as f32;
        let target_p = params.target_rate_hz * params.dt_ms / 1000.0;
        let n_blocks = weights.blocks.len();
        let block_size = n.div_ceil(n_blocks.max(1)).max(1);
        Self {
            gain: params.syn_gain,
            p: params,
            n,
            v: vec![0.0; n],
            refr: vec![0; n],
            active: Vec::new(),
            mask: vec![false; n],
            i_syn: vec![0.0; n],
            noise,
            rng,
            leak,
            target_p,
            activity: ActivityBuffer::new(n),
            spike_total: 0,
            steps: 0,
            last_step_ms: 0.0,
            path_counts: PathCounts::default(),
            indiv: None,
            block_spikes: vec![Vec::new(); n_blocks],
            block_size,
            weights,
        }
    }

    /// Per-fly gains, as `LIFParams.individuality` does (None = identical connectome).
    pub fn set_individuality(&mut self, gains: Option<Individuality>) {
        self.indiv = gains;
    }

    pub fn individuality(&self) -> Option<&Individuality> {
        self.indiv.as_ref()
    }

    pub fn weights(&self) -> &Arc<Weights> {
        &self.weights
    }

    /// Spikes of the latest step (ascending indices).
    pub fn spikes(&self) -> &[u32] {
        &self.active
    }

    /// Replace the noise generator, as `Brain.reseed` does: fresh bank and offsets from `seed`.
    pub fn reseed(&mut self, seed: u64) {
        let mut rng = Generator::new(seed);
        rng.fill_standard_normal_f32(&mut self.noise);
        let std = self.p.noise_std as f32;
        for x in &mut self.noise {
            *x *= std;
        }
        self.rng = rng;
    }

    /// Advance one `dt`. `sensory`: current per neuron (added as `ext_gain * sensory`), or `None`. Returns the spikes (ascending).
    pub fn step(&mut self, sensory: Option<&[f32]>) -> &[u32] {
        let t0 = Instant::now();
        let n = self.n;
        if let Some(s) = sensory {
            assert_eq!(s.len(), n, "sensory input must have one value per neuron");
        }
        let off = self.rng.integers(0, (self.noise.len() - n) as u64) as usize;

        let p = &self.p;
        let g32 = self.gain as f32;
        let bias = p.bias as f32;
        let ext = p.ext_gain as f32;
        let ext_is_one = p.ext_gain == 1.0;
        let keep = 1.0f32 - self.leak;
        let leak_reset = if p.v_reset != 0.0 { Some(self.leak * p.v_reset as f32) } else { None };
        let v_reset = p.v_reset as f32;
        let v_thresh = p.v_thresh as f32;
        let refr_steps = p.refractory_steps;
        let stamp = self.activity.stamp();
        let (decay, bump) = (self.activity.decay, self.activity.bump);

        let k = self.active.len();
        let dense = k as f64 > p.sparse_path_max_active * n as f64;
        if k > 0 {
            if dense {
                self.path_counts.full += 1;
                for &i in &self.active {
                    self.mask[i as usize] = true;
                }
            } else {
                self.path_counts.columns += 1;
            }
        }

        let weights = &*self.weights;
        let active = &self.active[..];
        let mask = &self.mask[..];
        let noise = &self.noise[off..off + n];
        let indiv = self.indiv.as_ref();
        let bs = self.block_size;

        self.v
            .par_chunks_mut(bs)
            .zip(self.refr.par_chunks_mut(bs))
            .zip(self.i_syn.par_chunks_mut(bs))
            .zip(self.activity.rate.par_chunks_mut(bs))
            .zip(self.activity.last_spike.par_chunks_mut(bs))
            .zip(self.block_spikes.par_iter_mut())
            .zip(weights.blocks.par_iter())
            .for_each(|((((((v, refr), isyn), rate), last), out), block)| {
                let row0 = block.row0;
                isyn.fill(0.0);
                if k > 0 {
                    if !dense {
                        match indiv {
                            None => {
                                for &c in active {
                                    let (a, b) = (block.colptr[c as usize] as usize, block.colptr[c as usize + 1] as usize);
                                    for (&r, &w) in block.rows[a..b].iter().zip(&block.data[a..b]) {
                                        isyn[r as usize] += w;
                                    }
                                }
                            }
                            Some(g) => {
                                for &c in active {
                                    let scale = g.d_pre[c as usize];
                                    let (a, b) = (block.colptr[c as usize] as usize, block.colptr[c as usize + 1] as usize);
                                    for (&r, &w) in block.rows[a..b].iter().zip(&block.data[a..b]) {
                                        isyn[r as usize] += w * scale;
                                    }
                                }
                            }
                        }
                    } else {
                        for (local, o) in isyn.iter_mut().enumerate() {
                            let (cols, vals) = weights.csr.row(row0 + local);
                            let mut sum = 0.0f32;
                            match indiv {
                                None => {
                                    for (&c, &w) in cols.iter().zip(vals) {
                                        if mask[c as usize] {
                                            sum += w;
                                        }
                                    }
                                }
                                Some(g) => {
                                    for (&c, &w) in cols.iter().zip(vals) {
                                        if mask[c as usize] {
                                            sum += w * g.d_pre[c as usize];
                                        }
                                    }
                                }
                            }
                            *o = sum;
                        }
                    }
                    if let Some(g) = indiv {
                        for (o, &d) in isyn.iter_mut().zip(&g.d_post[row0..row0 + v.len()]) {
                            *o *= d;
                        }
                    }
                }

                out.clear();
                for local in 0..v.len() {
                    let i = row0 + local;
                    let mut drive = isyn[local] * g32;
                    drive += bias;
                    drive += noise[i];
                    if let Some(s) = sensory {
                        drive += if ext_is_one { s[i] } else { ext * s[i] };
                    }
                    let mut vv = v[local] * keep;
                    if let Some(lr) = leak_reset {
                        vv += lr;
                    }
                    vv += drive;
                    if refr[local] > 0 {
                        vv = v_reset;
                        refr[local] -= 1;
                    }
                    rate[local] *= decay;
                    if vv >= v_thresh {
                        vv = v_reset;
                        refr[local] = refr_steps;
                        out.push(i as u32);
                        rate[local] += bump;
                        last[local] = stamp;
                    }
                    v[local] = vv;
                }
            });

        if dense {
            for &i in &self.active {
                self.mask[i as usize] = false;
            }
        }

        self.active.clear();
        for b in &self.block_spikes {
            self.active.extend_from_slice(b);
        }
        let fired = self.active.len();
        self.spike_total += fired as u64;
        self.steps += 1;

        // the slow gain controller (f64, like Python)
        let frac = fired as f64 / n as f64;
        let err = (self.target_p - frac) / self.target_p.max(1e-9);
        let (lo, hi) = self.p.gain_bounds;
        self.gain = (self.gain * (self.p.gain_adapt * err.clamp(-1.0, 1.0)).exp()).clamp(lo, hi);

        self.activity.finish_step(&self.active);
        self.last_step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        &self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    fn sim(seed: u64) -> LifSim {
        LifSim::new(Arc::new(synthetic::random_weights(2000, 30, 5, 4)), LifParams::default(), seed)
    }

    #[test]
    fn is_deterministic_and_seed_dependent() {
        let run = |seed| {
            let mut s = sim(seed);
            (0..60).flat_map(|_| s.step(None).to_vec()).collect::<Vec<u32>>()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    #[test]
    fn gain_stays_in_bounds_and_activity_is_tracked() {
        let mut s = sim(1);
        for _ in 0..200 {
            s.step(None);
        }
        assert!((1.5..=40.0).contains(&s.gain));
        assert_eq!(s.activity.steps, 200);
        assert_eq!(s.spike_total, s.activity.raster().map(|r| r.len() as u64).sum::<u64>());
        assert!(s.activity.rates().iter().any(|&r| r > 0.0));
    }

    #[test]
    fn driven_neurons_fire_once_per_refractory_cycle() {
        let mut s = sim(2);
        for _ in 0..50 {
            s.step(None);
        }
        let mut cur = vec![0f32; s.n];
        cur[..10].fill(0.5);
        let mut fired = vec![Vec::new(); 10];
        for t in 0..12 {
            let spikes = s.step(Some(&cur)).to_vec();
            for (i, f) in fired.iter_mut().enumerate() {
                if spikes.contains(&(i as u32)) {
                    f.push(t);
                }
            }
        }
        for f in &fired {
            assert!(f.len() >= 3, "a driven neuron fires every third step: {f:?}");
            assert!(f.windows(2).all(|w| w[1] - w[0] >= 3), "refractory for two steps: {f:?}");
        }
    }

    #[test]
    fn sparse_and_dense_paths_agree() {
        let w = Arc::new(synthetic::random_weights(2000, 30, 5, 4));
        let mut a = LifSim::new(w.clone(), LifParams::default(), 3);
        let mut b = LifSim::new(w, LifParams { sparse_path_max_active: 0.0, ..LifParams::default() }, 3);
        for _ in 0..80 {
            assert_eq!(a.step(None).to_vec(), b.step(None).to_vec());
        }
        assert!(a.path_counts.columns > 0 && b.path_counts.full > 0);
    }

    #[test]
    fn individuality_changes_the_run_only_when_on() {
        let w = Arc::new(synthetic::random_weights(2000, 30, 5, 4));
        let run = |ind| {
            let mut s = LifSim::new(w.clone(), LifParams::default(), 9);
            s.set_individuality(ind);
            (0..120).map(|_| s.step(None).len()).collect::<Vec<_>>()
        };
        assert_eq!(run(None), run(Individuality::compute(9, 2000, 0.0)));
        assert_ne!(run(None), run(Individuality::compute(9, 2000, 0.15)));
    }
}
