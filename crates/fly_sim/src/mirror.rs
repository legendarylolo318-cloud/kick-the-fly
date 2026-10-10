//! `simcore.symmetrize_weights`: the "Mirror-average weights" game rule. Every neuron with a bilateral partner (instance names
//! `X_L` / `X_R` that point at each other) swaps places with it in a permuted copy of the matrix, and the matrix becomes the mean of
//! the original and the permuted copy, with exact zeros dropped.

use crate::csr::{Csr, Weights};
use std::collections::HashMap;

/// The left/right partner permutation of `instance` (identity for neurons without an unambiguous partner).
pub fn partner_permutation(instance: &[String]) -> Vec<u32> {
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (i, name) in instance.iter().enumerate() {
        if !name.is_empty() {
            by_name.insert(name.as_str(), i); // the last one wins, as in the dict comprehension
        }
    }
    let mut perm: Vec<u32> = (0..instance.len() as u32).collect();
    for (idx, name) in instance.iter().enumerate() {
        let Some(stem) = name.strip_suffix("_L") else { continue };
        let Some(&other) = by_name.get(format!("{stem}_R").as_str()) else { continue };
        let Some(r_stem) = instance[other].strip_suffix("_R") else { continue };
        if by_name.get(format!("{r_stem}_L").as_str()) == Some(&idx) {
            perm[idx] = other as u32;
            perm[other] = idx as u32;
        }
    }
    perm
}

/// `(W + W[perm, :][:, perm]) * 0.5` with zeros eliminated.
pub fn symmetrize(csr: &Csr, instance: &[String]) -> Csr {
    let perm = partner_permutation(instance);
    let n = csr.n_rows;
    let mut indptr = Vec::with_capacity(n + 1);
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    indptr.push(0u32);
    let mut row: Vec<(u32, f32, f32)> = Vec::new();
    for i in 0..n {
        row.clear();
        let (cols, vals) = csr.row(i);
        row.extend(cols.iter().zip(vals).map(|(&c, &v)| (c, v, 0.0)));
        let (pcols, pvals) = csr.row(perm[i] as usize);
        row.extend(pcols.iter().zip(pvals).map(|(&c, &v)| (perm[c as usize], 0.0, v)));
        row.sort_by_key(|e| e.0);
        let mut k = 0;
        while k < row.len() {
            let c = row[k].0;
            let (mut a, mut b) = (0.0f32, 0.0f32);
            while k < row.len() && row[k].0 == c {
                a += row[k].1;
                b += row[k].2;
                k += 1;
            }
            let v = (a + b) * 0.5;
            if v != 0.0 {
                indices.push(c);
                data.push(v);
            }
        }
        indptr.push(indices.len() as u32);
    }
    Csr { n_rows: n, n_cols: csr.n_cols, indptr, indices, data }
}

/// The symmetrised weights as the simulator wants them.
pub fn symmetrized_weights(weights: &Weights, instance: &[String]) -> Weights {
    Weights::new(symmetrize(&weights.csr, instance), Weights::default_blocks())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrored_pair_averages() {
        let inst: Vec<String> = ["A_L", "A_R", "B"].iter().map(|s| s.to_string()).collect();
        // 0 -> 2 weight 4 (left to midline); the mirror image is 1 -> 2 weight 0
        let csr = Csr { n_rows: 3, n_cols: 3, indptr: vec![0, 0, 0, 1], indices: vec![0], data: vec![4.0] };
        // W[post=2, pre=0] = 4: post B (self partner), pre A_L; mirrored: W[2, 1] = 4
        let s = symmetrize(&csr, &inst);
        let (c, v) = s.row(2);
        assert_eq!(c, &[0, 1]);
        assert_eq!(v, &[2.0, 2.0]);
    }
}
