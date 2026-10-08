//! Synthetic connectivity for unit tests and for the benchmark when no brain pack is installed.
//!
//! A random sparse signed network with the pack's conventions: weights are small integer synapse counts times `1 / total input
//! count` of the postsynaptic neuron (rate-normalised), about 80 % excitatory. It is not a connectome and means nothing biologically.

use crate::csr::{Csr, Weights};

/// A tiny deterministic generator (SplitMix64), so tests need no `rand` dependency.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// `n` neurons, about `avg_in` inputs each, as a [`Csr`] (rows sorted, no duplicate columns).
pub fn random_csr(n: usize, avg_in: usize, seed: u64) -> Csr {
    let mut rng = SplitMix(seed);
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0u32);
    for _ in 0..n {
        let k = (avg_in / 2 + rng.below(avg_in as u64 + 1) as usize).min(n - 1).max(1);
        let mut cols: Vec<u32> = (0..k).map(|_| rng.below(n as u64) as u32).collect();
        cols.sort_unstable();
        cols.dedup();
        let counts: Vec<i32> = cols
            .iter()
            .map(|_| {
                let c = 1 + rng.below(5) as i32;
                if rng.below(5) == 0 { -c } else { c }
            })
            .collect();
        let total: i32 = counts.iter().map(|c| c.abs()).sum();
        let inv = 1.0f32 / total as f32;
        indices.extend(&cols);
        data.extend(counts.iter().map(|&c| c as f32 * inv));
        indptr.push(indices.len() as u32);
    }
    Csr { n_rows: n, n_cols: n, indptr, indices, data }
}

/// [`random_csr`] wrapped into [`Weights`] with `blocks` row blocks.
pub fn random_weights(n: usize, avg_in: usize, seed: u64, blocks: usize) -> Weights {
    Weights::new(random_csr(n, avg_in, seed), blocks)
}
