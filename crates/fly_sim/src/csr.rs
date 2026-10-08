//! Sparse connectivity: a CSR matrix `W[post, pre]` plus a row-blocked CSC copy for the sparse-activity path.
//!
//! The simulator needs `i_syn[post] = sum over pre that spiked of W[post, pre]` every step, with about 2.5 % of the presynaptic
//! neurons active. Two ways to compute it, both adding each row's contributions in ascending presynaptic order in `f32`, which is
//! the order SciPy's `csc_matvec` (column gather) and `csr_matvec` use, so the sums are bit-identical to the Python reference:
//!
//! * **sparse**: for each active column, walk its entries. A plain CSC scatter would race between threads, so the rows are split
//!   into blocks; each block owns a CSC of just its rows and one rayon task handles it end to end.
//! * **dense** (more than a tenth of the neurons active): walk each CSR row and add the entries whose column spiked.

use rayon::prelude::*;

/// Compressed sparse row matrix of `f32`, `n_rows x n_cols`.
#[derive(Debug, Clone)]
pub struct Csr {
    pub n_rows: usize,
    pub n_cols: usize,
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub data: Vec<f32>,
}

impl Csr {
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Entries of row `i` as `(columns, values)`.
    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let (a, b) = (self.indptr[i] as usize, self.indptr[i + 1] as usize);
        (&self.indices[a..b], &self.data[a..b])
    }

    /// Sort every row by column (a no-op for the shipped pack, whose rows are already sorted).
    pub fn sort_rows(&mut self) {
        for i in 0..self.n_rows {
            let (a, b) = (self.indptr[i] as usize, self.indptr[i + 1] as usize);
            if self.indices[a..b].is_sorted() {
                continue;
            }
            let mut pairs: Vec<(u32, f32)> = self.indices[a..b].iter().copied().zip(self.data[a..b].iter().copied()).collect();
            pairs.sort_by_key(|p| p.0);
            for (k, (c, v)) in pairs.into_iter().enumerate() {
                self.indices[a + k] = c;
                self.data[a + k] = v;
            }
        }
    }

    /// Reference `y = A x` with a 0/1 spike vector, columns ascending per row (used by tests and the dense path).
    pub fn spmv_spikes(&self, spikes: &[bool], out: &mut [f32]) {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let (cols, vals) = self.row(i);
            let mut sum = 0.0f32;
            for (&c, &w) in cols.iter().zip(vals) {
                if spikes[c as usize] {
                    sum += w;
                }
            }
            *o = sum;
        });
    }
}

/// One block of rows in CSC form: for each column, the entries that fall in `[row0, row1)`.
#[derive(Debug, Clone)]
pub struct CscBlock {
    pub row0: usize,
    pub row1: usize,
    pub colptr: Vec<u32>,
    /// Row index relative to `row0`.
    pub rows: Vec<u32>,
    pub data: Vec<f32>,
}

/// The connectivity as the simulator uses it.
#[derive(Debug, Clone)]
pub struct Weights {
    pub csr: Csr,
    pub blocks: Vec<CscBlock>,
}

impl Weights {
    /// Build the blocked CSC from a CSR. `n_blocks` is clamped to `[1, n_rows]`; more blocks than threads balances the work.
    pub fn new(mut csr: Csr, n_blocks: usize) -> Self {
        csr.sort_rows();
        let n = csr.n_rows;
        let n_blocks = n_blocks.clamp(1, n.max(1));
        let size = n.div_ceil(n_blocks);
        let blocks: Vec<CscBlock> = (0..n_blocks)
            .into_par_iter()
            .map(|b| {
                let (row0, row1) = (b * size, ((b + 1) * size).min(n));
                if row0 >= row1 {
                    return CscBlock { row0: n, row1: n, colptr: vec![0; csr.n_cols + 1], rows: Vec::new(), data: Vec::new() };
                }
                let mut colptr = vec![0u32; csr.n_cols + 1];
                for i in row0..row1 {
                    for &c in csr.row(i).0 {
                        colptr[c as usize + 1] += 1;
                    }
                }
                for c in 0..csr.n_cols {
                    colptr[c + 1] += colptr[c];
                }
                let nnz = colptr[csr.n_cols] as usize;
                let mut rows = vec![0u32; nnz];
                let mut data = vec![0f32; nnz];
                let mut next = colptr.clone();
                // rows ascending → within a column the entries come out in ascending row order
                for i in row0..row1 {
                    let (cols, vals) = csr.row(i);
                    for (&c, &w) in cols.iter().zip(vals) {
                        let slot = next[c as usize] as usize;
                        rows[slot] = (i - row0) as u32;
                        data[slot] = w;
                        next[c as usize] += 1;
                    }
                }
                CscBlock { row0, row1, colptr, rows, data }
            })
            .collect();
        Self { csr, blocks }
    }

    pub fn n(&self) -> usize {
        self.csr.n_rows
    }

    /// Default block count: twice the rayon thread count (at least 8, at most 64).
    pub fn default_blocks() -> usize {
        (rayon::current_num_threads() * 2).clamp(8, 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Csr {
        // 3x3: row0: [0]=1 [2]=2 ; row1: [1]=3 ; row2: [0]=4 [1]=5 [2]=6
        Csr {
            n_rows: 3,
            n_cols: 3,
            indptr: vec![0, 2, 3, 6],
            indices: vec![0, 2, 1, 0, 1, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    }

    #[test]
    fn blocked_csc_matches_csr() {
        let w = Weights::new(tiny(), 2);
        assert_eq!(w.blocks.len(), 2);
        let mut total = 0usize;
        for b in &w.blocks {
            total += b.rows.len();
        }
        assert_eq!(total, 6);
        // column 0 has entries in rows 0 and 2
        let b0 = &w.blocks[0];
        assert_eq!(b0.row1 - b0.row0, 2);
        assert_eq!(&b0.rows[b0.colptr[0] as usize..b0.colptr[1] as usize], &[0]);
    }

    #[test]
    fn spmv_sums_active_columns() {
        let csr = tiny();
        let mut out = [0.0f32; 3];
        csr.spmv_spikes(&[true, false, true], &mut out);
        assert_eq!(out, [3.0, 0.0, 10.0]);
    }

    #[test]
    fn unsorted_rows_are_sorted() {
        let mut csr = tiny();
        csr.indices.swap(0, 1);
        csr.data.swap(0, 1);
        csr.sort_rows();
        assert_eq!(csr.row(0).0, &[0, 2]);
        assert_eq!(csr.row(0).1, &[1.0, 2.0]);
    }
}
