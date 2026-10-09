//! Real training: dopamine-gated plasticity on the connectome's own mushroom body synapses (`core/memory.py`).
//!
//! A smell switches on a sparse set of Kenyon cells (KCs). Dopamine neurons (DANs) that fire while it is present weaken the synapses from the
//! active KCs onto the mushroom body output neurons (MBONs) of their compartment: PPL1 dopamine (punishment) weakens the approach MBONs, so
//! the fly avoids the smell; PAM dopamine (reward) weakens the avoidance MBONs, so it approaches. The plastic synapses are every KC -> MBON
//! connection in the pack; which DANs gate each MBON comes from the pack's DAN -> MBON synapse counts. Every 50 ms of brain time (10 steps):
//! eligibility = how far each KC fires above its calm rate (only the top 280 count), dopamine = how far each MBON's DANs fire above calm
//! (weighted by synapse counts); each synapse is weakened by `rate * dopamine * eligibility * weight` down to 10 % of its connectome value,
//! recovers slowly (half-life 30 min of brain time), and dopamine in one compartment restores the same smell's synapses in the opposite one
//! (reversal learning). The changed weights are the ones the simulation runs on.
//!
//! The arithmetic follows the NumPy code operation by operation (f32 where NumPy is, f64 where a Python float array makes it so); the
//! results agree with Python's to rounding (`tests/golden/memory.json`, `tests/memory_golden.rs`).

use std::collections::BTreeMap;

use crate::csr::Weights;
use crate::lif::LifSim;
use crate::pack::Labels;

/// Kenyon cells counted per pattern (of 4,064).
pub const KC_SPARSE: usize = 280;
/// Per 50 ms update at full dopamine and eligibility: about 6-10 pairings to full memory.
pub const LEARN_RATE: f32 = 0.012;
/// A synapse can be weakened to 10 % of its connectome weight.
pub const FLOOR: f32 = 0.1;
/// Opposite dopamine restores a smell's weakened synapses.
pub const REVERSAL_RATE: f64 = 0.03;
/// Brain-time seconds for a weakened synapse to recover half-way.
pub const FORGET_HALF_LIFE_S: f64 = 1800.0;
pub const DA_MIN_HZ: f32 = 12.0;
pub const DA_FULL_HZ: f32 = 28.0;
pub const KC_MIN_HZ: f32 = 8.0;
pub const KC_FULL_HZ: f32 = 18.0;
/// A smell whose KC input is weakened this much reads as memory 1.0.
pub const FULL_DEPRESSION: f32 = 0.55;
/// Simulation steps (5 ms each) between plasticity updates.
pub const UPDATE_STEPS: u64 = 10;
/// 5 s of brain time to learn resting activity before any plasticity.
pub const SETTLE_UPDATES: u64 = 100;
pub const VERSION: i64 = 1;
const DT: f32 = 0.005;

/// One row of `training-log.json`: `[time, kind, fear, like, mbon_hz]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub time: f64,
    /// `naive`, `fear`, `like` or `test`.
    pub kind: String,
    pub fear: f64,
    pub like: f64,
    pub mbon_hz: Option<f64>,
}

pub type TrialLog = BTreeMap<String, Vec<Trial>>;

pub struct Memory {
    pub kc: Vec<u32>,
    pub mbon: Vec<u32>,
    pub dan: Vec<u32>,
    /// True: PPL1 compartment (approach MBON).
    pub punish_comp: Vec<bool>,
    /// `[DAN][plastic MBON]`: each DAN's share of that MBON's dopamine.
    gate: Vec<f32>,
    csr_pos: Vec<u32>,
    blk: Vec<(u16, u32)>,
    syn_mbon: Vec<u32>,
    syn_kc: Vec<u32>,
    pub w0: Vec<f32>,
    pub w: Vec<f32>,
    /// What the simulator currently holds for each plastic synapse.
    written: Vec<f32>,
    punish_syn: Vec<bool>,
    pub kc_calm: Vec<f32>,
    pub dan_calm: Vec<f32>,
    pub calm_ready: bool,
    pub updates: u64,
    pub templates: BTreeMap<String, Vec<f32>>,
    pub log: TrialLog,
    /// Approach-MBON response to a smell before any training.
    pub naive_mbon: BTreeMap<String, f64>,
    pub last_elig: Vec<f32>,
    pub last_da: Vec<f32>,
    pub enabled: bool,
    pub dirty: bool,
    pub signature: [f64; 2],
}

impl Memory {
    /// `None` when the pack has no dopamine/MBON tables (an older pack, a synthetic network).
    pub fn new(labels: &Labels, weights: &Weights) -> Option<Self> {
        if labels.dan.is_empty() || labels.mbon.is_empty() || labels.dan_mbon.len() != labels.dan.len() * labels.mbon.len() {
            return None;
        }
        let n_mb_all = labels.mbon.len();
        let kc: Vec<u32> = (0..labels.types.len() as u32).filter(|&i| labels.types[i as usize].starts_with("KC")).collect();
        let (mut ppl, mut pam) = (vec![0f32; n_mb_all], vec![0f32; n_mb_all]);
        for (d, &row) in labels.dan.iter().enumerate() {
            let is_ppl1 = labels.types[row as usize].starts_with("PPL1");
            for m in 0..n_mb_all {
                let c = f32::from(labels.dan_mbon[d * n_mb_all + m]);
                if is_ppl1 { ppl[m] += c } else { pam[m] += c }
            }
        }
        let plastic: Vec<usize> = (0..n_mb_all).filter(|&m| ppl[m] + pam[m] >= 20.0).collect();
        let mbon: Vec<u32> = plastic.iter().map(|&m| labels.mbon[m]).collect();
        let punish_comp: Vec<bool> = plastic.iter().map(|&m| ppl[m] >= pam[m]).collect();
        let n_p = plastic.len();
        let mut gate = vec![0f32; labels.dan.len() * n_p];
        for d in 0..labels.dan.len() {
            for (j, &m) in plastic.iter().enumerate() {
                gate[d * n_p + j] = f32::from(labels.dan_mbon[d * n_mb_all + m]);
            }
        }
        for j in 0..n_p {
            let s: f32 = (0..labels.dan.len()).map(|d| gate[d * n_p + j]).sum::<f32>().max(1.0);
            for d in 0..labels.dan.len() {
                gate[d * n_p + j] /= s;
            }
        }
        // the KC -> MBON entries of the matrix, in CSR order
        let mut kc_local = vec![u32::MAX; weights.csr.n_cols];
        for (i, &k) in kc.iter().enumerate() {
            kc_local[k as usize] = i as u32;
        }
        let mut mb_local = vec![u32::MAX; weights.csr.n_rows];
        for (i, &m) in mbon.iter().enumerate() {
            mb_local[m as usize] = i as u32;
        }
        let (mut csr_pos, mut syn_mbon, mut syn_kc, mut w0): (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>) = (vec![], vec![], vec![], vec![]);
        #[allow(clippy::needless_range_loop)]
        for row in 0..weights.csr.n_rows {
            if mb_local[row] == u32::MAX {
                continue;
            }
            let (a, b) = (weights.csr.indptr[row] as usize, weights.csr.indptr[row + 1] as usize);
            for p in a..b {
                let c = weights.csr.indices[p] as usize;
                if kc_local[c] != u32::MAX && weights.csr.data[p] > 0.0 {
                    csr_pos.push(p as u32);
                    syn_mbon.push(mb_local[row]);
                    syn_kc.push(kc_local[c]);
                    w0.push(weights.csr.data[p]);
                }
            }
        }
        // the block copy: which block holds the row, and where in it
        let block_size = weights.n().div_ceil(weights.blocks.len().max(1)).max(1);
        let mut blk = Vec::with_capacity(csr_pos.len());
        for i in 0..csr_pos.len() {
            let row = mbon[syn_mbon[i] as usize] as usize;
            let col = kc[syn_kc[i] as usize] as usize;
            let b = row / block_size;
            let blockref = &weights.blocks[b];
            let (lo, hi) = (blockref.colptr[col] as usize, blockref.colptr[col + 1] as usize);
            let rel = (row - blockref.row0) as u32;
            let at = blockref.rows[lo..hi].binary_search(&rel).map(|k| lo + k).ok()?;
            blk.push((b as u16, at as u32));
        }
        let punish_syn: Vec<bool> = syn_mbon.iter().map(|&m| punish_comp[m as usize]).collect();
        let sum: f64 = w0.iter().map(|&x| f64::from(x)).sum();
        let signature = [w0.len() as f64, sum];
        let written = w0.clone();
        let n_kc = kc.len();
        let n_mb = mbon.len();
        Some(Memory {
            kc,
            mbon,
            dan: labels.dan.clone(),
            punish_comp,
            gate,
            csr_pos,
            blk,
            syn_mbon,
            syn_kc,
            w: w0.clone(),
            w0,
            written,
            punish_syn,
            kc_calm: vec![0.0; n_kc],
            dan_calm: vec![0.0; labels.dan.len()],
            calm_ready: false,
            updates: 0,
            templates: BTreeMap::new(),
            log: TrialLog::new(),
            naive_mbon: BTreeMap::new(),
            last_elig: vec![0.0; n_kc],
            last_da: vec![0.0; n_mb],
            enabled: true,
            dirty: false,
            signature,
        })
    }

    /// `Memory.step`: called every [`UPDATE_STEPS`] steps with the simulator (its rate EMA, its weights) and whether the brain is calm.
    pub fn step(&mut self, sim: &mut LifSim, calm: bool) {
        let rates = sim.activity.rates();
        let kr: Vec<f32> = self.kc.iter().map(|&i| rates[i as usize]).collect();
        let dr: Vec<f32> = self.dan.iter().map(|&i| rates[i as usize]).collect();
        self.updates += 1;
        if self.updates == 1 && !self.calm_ready {
            self.kc_calm.copy_from_slice(&kr);
            self.dan_calm.copy_from_slice(&dr);
        }
        let k = if calm || self.updates < SETTLE_UPDATES { 0.02f32 } else { 0.0005f32 };
        for (c, &r) in self.kc_calm.iter_mut().zip(&kr) {
            *c += (r - *c) * k;
        }
        for (c, &r) in self.dan_calm.iter_mut().zip(&dr) {
            *c += (r - *c) * k;
        }
        if self.updates < SETTLE_UPDATES {
            return;
        }
        let elig = self.pattern(&kr);
        let n_p = self.mbon.len();
        let da_hz: Vec<f32> = dr.iter().zip(&self.dan_calm).map(|(&r, &c)| (r - c).max(0.0) / DT).collect();
        let mut gated = vec![0f32; n_p];
        for (d, &h) in da_hz.iter().enumerate() {
            if h != 0.0 {
                for (g, &gt) in gated.iter_mut().zip(&self.gate[d * n_p..(d + 1) * n_p]) {
                    *g += h * gt;
                }
            }
        }
        let da: Vec<f32> = gated.iter().map(|&g| ((g - DA_MIN_HZ) / (DA_FULL_HZ - DA_MIN_HZ)).clamp(0.0, 1.0)).collect();
        let da_max = da.iter().copied().fold(0.0f32, f32::max);
        let mut changed = false;
        if self.enabled && elig.iter().any(|&e| e != 0.0) && da_max > 0.02 {
            let mut dw_max = 0.0f32;
            let mut dw = vec![0f32; self.w.len()];
            for i in 0..self.w.len() {
                let e = elig[self.syn_kc[i] as usize];
                let d = LEARN_RATE * da[self.syn_mbon[i] as usize] * e * self.w[i];
                let d = d.min(self.w[i] - FLOOR * self.w0[i]);
                dw[i] = d;
                dw_max = dw_max.max(d);
            }
            if dw_max > 1e-7 {
                for (w, d) in self.w.iter_mut().zip(&dw) {
                    *w -= d;
                }
                changed = true;
            }
            let mean_of = |mask_val: bool| -> f64 {
                let (mut s, mut c) = (0f64, 0usize);
                for (j, &d) in da.iter().enumerate() {
                    if self.punish_comp[j] == mask_val {
                        s += f64::from(d);
                        c += 1;
                    }
                }
                if c == 0 { 0.0 } else { s / c as f64 }
            };
            let pun_da = mean_of(true);
            let rew_da = mean_of(false);
            if pun_da.max(rew_da) > 0.02 {
                for i in 0..self.w.len() {
                    let opp = if self.punish_syn[i] { rew_da } else { pun_da };
                    let e = f64::from(elig[self.syn_kc[i] as usize]);
                    let x = REVERSAL_RATE * opp * e * f64::from(self.w0[i] - self.w[i]);
                    self.w[i] = (f64::from(self.w[i]) + x) as f32;
                }
            }
        }
        self.last_elig = elig;
        self.last_da = da;
        let kf = (1.0 - 0.5f64.powf(UPDATE_STEPS as f64 * 0.005 / FORGET_HALF_LIFE_S)) as f32;
        for (w, &w0) in self.w.iter_mut().zip(&self.w0) {
            *w += (w0 - *w) * kf;
        }
        self.write_back(sim, false);
        self.dirty = self.dirty || changed;
    }

    /// Copy the plastic weights into the simulator (only the ones that changed, and only then is the weight matrix made private to this
    /// sim: it is shared with the arcade's brains until the first change).
    pub fn write_back(&mut self, sim: &mut LifSim, full: bool) {
        let dirty: Vec<usize> =
            if full { (0..self.w.len()).collect() } else { (0..self.w.len()).filter(|&i| self.w[i] != self.written[i]).collect() };
        if dirty.is_empty() {
            return;
        }
        let weights = std::sync::Arc::make_mut(sim.weights_mut());
        for i in dirty {
            weights.csr.data[self.csr_pos[i] as usize] = self.w[i];
            let (b, at) = self.blk[i];
            weights.blocks[b as usize].data[at as usize] = self.w[i];
            self.written[i] = self.w[i];
        }
    }

    /// KC eligibility in 0..1: firing above calm past the resting noise, sparsified to the most active KCs.
    pub fn pattern(&self, kc_rates: &[f32]) -> Vec<f32> {
        let mut act: Vec<f32> = kc_rates
            .iter()
            .zip(&self.kc_calm)
            .map(|(&r, &c)| (((r - c) / DT - KC_MIN_HZ) / (KC_FULL_HZ - KC_MIN_HZ)).clamp(0.0, 1.0))
            .collect();
        sparsify(&mut act);
        act
    }

    /// `(fear, liking)` in 0..1 for a KC pattern: how much its input to each compartment's MBONs is weakened.
    pub fn score(&self, pattern: Option<&[f32]>) -> (f32, f32) {
        let Some(p) = pattern else { return (0.0, 0.0) };
        if p.iter().all(|&x| x == 0.0) {
            return (0.0, 0.0);
        }
        let mut pat = p.to_vec();
        if pat.iter().filter(|&&x| x != 0.0).count() > KC_SPARSE {
            let thr = kth_largest(&pat, KC_SPARSE);
            for x in &mut pat {
                if *x < thr {
                    *x = 0.0;
                }
            }
        }
        let mut out = [0f32; 2];
        for (slot, want) in out.iter_mut().zip([true, false]) {
            let (mut base, mut now) = (0f64, 0f64);
            for i in 0..self.w.len() {
                if self.punish_syn[i] == want {
                    let e = f64::from(pat[self.syn_kc[i] as usize]);
                    base += f64::from(self.w0[i]) * e;
                    now += f64::from(self.w[i]) * e;
                }
            }
            *slot = if base > 0.0 { (((1.0 - now / base) as f32) / FULL_DEPRESSION).clamp(0.0, 1.0) } else { 0.0 };
        }
        (out[0], out[1])
    }

    /// Keep a running KC pattern for a smell while it is present (what memory is read against).
    pub fn observe(&mut self, scent: &str, rates: &[f32]) {
        let kr: Vec<f32> = self.kc.iter().map(|&i| rates[i as usize]).collect();
        let mut p = self.pattern(&kr);
        let s: f32 = p.iter().sum();
        if s == 0.0 {
            return;
        }
        for x in &mut p {
            *x /= s;
        }
        match self.templates.get_mut(scent) {
            None => {
                self.templates.insert(scent.to_string(), p);
            }
            Some(t) => {
                for (t, &x) in t.iter_mut().zip(&p) {
                    *t += (x - *t) * 0.05;
                }
            }
        }
        self.dirty = true;
    }

    pub fn memory_of(&self, scent: &str) -> (f32, f32) {
        self.score(self.templates.get(scent).map(Vec::as_slice))
    }

    /// Mean firing (Hz) of the approach-promoting (PPL1-compartment) MBONs: what fear training lowers.
    pub fn mbon_response(&self, rates: &[f32]) -> f32 {
        let (mut s, mut n) = (0f32, 0usize);
        for (j, &m) in self.mbon.iter().enumerate() {
            if self.punish_comp[j] {
                s += rates[m as usize];
                n += 1;
            }
        }
        if n == 0 { 0.0 } else { s / n as f32 / DT }
    }

    /// `Memory.record`: one row of the training log.
    pub fn record(&mut self, scent: &str, kind: &str, mbon_hz: Option<f64>, now_unix: f64) {
        let (fear, like) = self.memory_of(scent);
        let r4 = |x: f32| (f64::from(x) * 1e4).round() / 1e4;
        self.log.entry(scent.to_string()).or_default().push(Trial {
            time: now_unix,
            kind: kind.into(),
            fear: r4(fear),
            like: r4(like),
            mbon_hz: mbon_hz.map(|h| (h * 1e3).round() / 1e3),
        });
        self.dirty = true;
    }

    /// Share of the plastic synapses weakened by more than 10 %.
    pub fn weakened_share(&self) -> f32 {
        self.w.iter().zip(&self.w0).filter(|(w, w0)| **w < **w0 * 0.9).count() as f32 / self.w.len().max(1) as f32
    }

    /// Forget everything (`Memory.wipe`; the caller saves).
    pub fn wipe(&mut self, sim: &mut LifSim) {
        self.w.copy_from_slice(&self.w0);
        self.templates.clear();
        self.log.clear();
        self.naive_mbon.clear();
        self.write_back(sim, true);
    }

    /// Put a saved memory back (`Memory.load` after the file was read and its signature checked).
    pub fn restore(
        &mut self,
        sim: &mut LifSim,
        w: Vec<f32>,
        kc_calm: Vec<f32>,
        dan_calm: Vec<f32>,
        templates: BTreeMap<String, Vec<f32>>,
    ) -> bool {
        if w.len() != self.w.len() || kc_calm.len() != self.kc.len() || dan_calm.len() != self.dan.len() {
            return false;
        }
        self.w = w;
        self.kc_calm = kc_calm;
        self.dan_calm = dan_calm;
        self.templates = templates;
        self.calm_ready = true;
        self.write_back(sim, true);
        true
    }
}

/// `np.partition(x, -k)[-k]`: the k-th largest value.
fn kth_largest(x: &[f32], k: usize) -> f32 {
    let mut v = x.to_vec();
    let idx = v.len() - k;
    *v.select_nth_unstable_by(idx, |a, b| a.total_cmp(b)).1
}

/// Zero everything below the [`KC_SPARSE`]-th largest value when more than that many are non-zero (ties are all kept).
fn sparsify(act: &mut [f32]) {
    if act.iter().filter(|&&x| x != 0.0).count() > KC_SPARSE {
        let thr = kth_largest(act, KC_SPARSE);
        for x in act.iter_mut() {
            if *x < thr {
                *x = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparsify_keeps_the_top_and_all_ties() {
        let mut a: Vec<f32> = (0..400).map(|i| (i % 100) as f32 / 100.0 + 0.001).collect();
        sparsify(&mut a);
        let kept = a.iter().filter(|&&x| x != 0.0).count();
        assert!((KC_SPARSE..400).contains(&kept), "{kept}");
        assert_eq!(kth_largest(&[1.0, 5.0, 3.0, 4.0], 2), 4.0);
        let mut few = vec![0.5; 10];
        sparsify(&mut few);
        assert_eq!(few, vec![0.5; 10]);
    }
}
