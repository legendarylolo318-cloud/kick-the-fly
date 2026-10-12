//! Network science of the brain pack (`lab/netsci.py`): degree distributions, reciprocity, 3-node motifs against a degree-preserving
//! null, the rich club, Louvain-style communities and per-region summaries. Everything is computed from the pack's raw synapse counts;
//! nothing is simulated. The random streams are NumPy's (`default_rng(seed)`: `permutation`, `random`, `choice(p=)`), the sums are
//! NumPy's pairwise sums and the sparse matrix products follow SciPy's order of accumulation, so the numbers equal Python's.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use serde_json::{Value, json};

use crate::npy::Npz;
use crate::rng::Generator;
use crate::validation::np_sum;

pub const ANALYSIS_VERSION: i64 = 2;
pub const DEFAULT_NULLS: usize = 3;
pub const DEFAULT_WEDGES: usize = 300_000;
pub const DEFAULT_SEED: u64 = 0;
pub const RICH_PERCENTILES: [f64; 6] = [50.0, 75.0, 90.0, 95.0, 98.0, 99.0];
pub const SWEEPS_PER_LEVEL: usize = 12;

#[derive(Debug)]
pub enum Error {
    Cancelled,
    Pack(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cancelled => f.write_str("cancelled"),
            Error::Pack(s) => f.write_str(s),
        }
    }
}

fn check(cancel: &AtomicBool) -> Result<(), Error> {
    if cancel.load(Ordering::Relaxed) { Err(Error::Cancelled) } else { Ok(()) }
}

/// The pack's connections as arrays (`load_edges`).
pub struct Graph {
    pub n: usize,
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
    pub syn: Vec<f64>,
    pub sign: Vec<i8>,
    pub region: Vec<String>,
    pub superclass: Vec<String>,
    pub types: Vec<String>,
}

pub fn load_edges(path: &std::path::Path) -> Result<Graph, Error> {
    let file = std::fs::File::open(path).map_err(|e| Error::Pack(e.to_string()))?;
    let mut npz = Npz::read(std::io::BufReader::new(file)).map_err(|e| Error::Pack(e.to_string()))?;
    let pe = |e: crate::Error| Error::Pack(e.to_string());
    let n = npz.take("inv").map_err(pe)?.into_f32().map_err(pe)?.len();
    let indptr = npz.take("indptr").map_err(pe)?.into_i32().map_err(pe)?;
    let indices = npz.take("indices").map_err(pe)?.into_i32().map_err(pe)?;
    let data = npz.take("data").map_err(pe)?.into_i16().map_err(pe)?;
    let (mut src, mut dst, mut syn, mut sign) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for post in 0..n {
        for e in indptr[post] as usize..indptr[post + 1] as usize {
            let (pre, raw) = (indices[e] as usize, data[e]);
            if raw != 0 && pre != post {
                src.push(pre as u32);
                dst.push(post as u32);
                syn.push(f64::from(raw).abs());
                sign.push(raw.signum() as i8);
            }
        }
    }
    let region = match npz.take_opt("region") {
        Some(a) => a.into_strings().map_err(pe)?,
        None => vec!["unassigned".to_string(); n],
    };
    let superclass = npz.take("superclass").map_err(pe)?.into_strings().map_err(pe)?;
    let types = npz.take("type").map_err(pe)?.into_strings().map_err(pe)?;
    Ok(Graph { n, src, dst, syn, sign, region, superclass, types })
}

// --- NumPy helpers -------------------------------------------------------------------------------------------------------
/// `np.percentile(x, p)` (linear) of an ascending-sorted array.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let q = p / 100.0;
    let virt = (sorted.len() - 1) as f64 * q;
    let lo = virt.floor();
    let (prev, next) = (sorted[lo as usize], sorted[(lo as usize + 1).min(sorted.len() - 1)]);
    let gamma = virt - lo;
    let diff = next - prev;
    if gamma >= 0.5 { next - diff * (1.0 - gamma) } else { prev + diff * gamma }
}

fn summary(x: &[f64]) -> Value {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let median = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
    json!({
        "mean": np_sum(x) / n as f64,
        "median": median,
        "p90": percentile_sorted(&s, 90.0),
        "p99": percentile_sorted(&s, 99.0),
        "max": s[n - 1],
        "zero": x.iter().filter(|&&v| v == 0.0).count(),
    })
}

/// Counts in log2 bins: degree 0, 1, 2-3, 4-7, ... (`_log_hist`).
fn log_hist(deg: &[u32]) -> Value {
    let max = deg.iter().copied().max().unwrap_or(0) as u64;
    let mut edges = vec![0u64, 1];
    while *edges.last().unwrap() <= max {
        edges.push(edges.last().unwrap() * 2);
    }
    // bins: [-0.5, 0.5], then [e - 0.5, next e - 0.5) for edges[2..], the last open at the top
    let mut bounds: Vec<f64> = vec![-0.5, 0.5];
    bounds.extend(edges[2..].iter().map(|&e| e as f64 - 0.5));
    let mut h = vec![0u64; bounds.len()];
    for &d in deg {
        let d = f64::from(d);
        let mut b = bounds.partition_point(|&e| e <= d); // number of bounds at or below d
        b = b.saturating_sub(1);
        h[b.min(bounds.len() - 1)] += 1;
    }
    // np.histogram's last real bin is closed on the right; the final bound is +inf here, so the loop above matches.
    let mut out = vec![json!({"low": 0, "high": 0, "count": h[0]})];
    for (i, w) in edges[1..].windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        out.push(json!({"low": a, "high": b - 1, "count": h[i + 1]}));
    }
    Value::Array(out.into_iter().filter(|r| r["count"].as_u64().unwrap_or(0) > 0 || r["low"] == 0).collect())
}

fn bincount(idx: &[u32], n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    for &i in idx {
        out[i as usize] += 1;
    }
    out
}

fn bincount_w(idx: &[u32], w: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (&i, &x) in idx.iter().zip(w) {
        out[i as usize] += x;
    }
    out
}

fn as_f64(v: &[u32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// `np.corrcoef(a, b)[0, 1]`.
fn corrcoef(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (np_sum(a) / n, np_sum(b) / n);
    let xa: Vec<f64> = a.iter().map(|v| v - ma).collect();
    let xb: Vec<f64> = b.iter().map(|v| v - mb).collect();
    let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(x, y)| x * y).sum::<f64>();
    let (caa, cbb, cab) = (dot(&xa, &xa) / (n - 1.0), dot(&xb, &xb) / (n - 1.0), dot(&xa, &xb) / (n - 1.0));
    (cab / caa.sqrt() / cbb.sqrt()).clamp(-1.0, 1.0)
}

pub fn degrees(g: &Graph) -> Value {
    let n = g.n;
    let indeg = bincount(&g.dst, n);
    let outdeg = bincount(&g.src, n);
    let insyn = bincount_w(&g.dst, &g.syn, n);
    let outsyn = bincount_w(&g.src, &g.syn, n);
    let mut hubs = Vec::new();
    for (name, d) in [("in", &indeg), ("out", &outdeg)] {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(d[i])); // stable: ties in row order
        for &i in order.iter().take(8) {
            hubs.push(json!({"direction": name, "row": i, "type": g.types[i], "degree": d[i]}));
        }
    }
    let with_hist = |d: &[u32]| {
        let mut s = summary(&as_f64(d));
        s["histogram"] = log_hist(d);
        s
    };
    json!({
        "n": n,
        "connections": g.src.len(),
        "synapses": np_sum(&g.syn),
        "density": g.src.len() as f64 / (n as f64 * (n as f64 - 1.0)),
        "in_degree": with_hist(&indeg),
        "out_degree": with_hist(&outdeg),
        "in_synapses": summary(&insyn),
        "out_synapses": summary(&outsyn),
        "hubs": hubs,
        "in_out_degree_correlation": corrcoef(&as_f64(&indeg), &as_f64(&outdeg)),
    })
}

fn has_key(sorted: &[u64], k: u64) -> bool {
    sorted.binary_search(&k).is_ok()
}

pub fn reciprocity(src: &[u32], dst: &[u32], syn: &[f64], n: usize) -> Value {
    let nn = n as u64;
    let mut pairs: Vec<(u64, f64)> = src.iter().zip(dst).zip(syn).map(|((&s, &d), &w)| (u64::from(s) * nn + u64::from(d), w)).collect();
    pairs.par_sort_unstable_by_key(|p| p.0);
    let keys: Vec<u64> = pairs.iter().map(|p| p.0).collect();
    let mut has_n = 0usize;
    let mut minw = Vec::with_capacity(src.len());
    for ((&s, &d), &w) in src.iter().zip(dst).zip(syn) {
        let rk = u64::from(d) * nn + u64::from(s);
        match keys.binary_search(&rk) {
            Ok(p) => {
                has_n += 1;
                minw.push(w.min(pairs[p].1));
            }
            Err(_) => minw.push(0.0f64.min(w)),
        }
    }
    json!({
        "binary": has_n as f64 / src.len() as f64,
        "weighted": np_sum(&minw) / np_sum(syn),
        "reciprocal_pairs": has_n / 2,
    })
}

fn binary_reciprocity(src: &[u32], dst: &[u32], n: usize) -> f64 {
    let nn = n as u64;
    let mut keys: Vec<u64> = src.iter().zip(dst).map(|(&s, &d)| u64::from(s) * nn + u64::from(d)).collect();
    keys.par_sort_unstable();
    let has = src.iter().zip(dst).filter(|(s, d)| has_key(&keys, u64::from(**d) * nn + u64::from(**s))).count();
    has as f64 / src.len() as f64
}

/// `ok &= ~dup[:half] & ~dup[half:]` helper: for each key of `new1 ++ new2`, whether it occurs more than once among them.
fn duplicated(new1: &[u64], new2: &[u64]) -> (Vec<bool>, Vec<bool>) {
    let half = new1.len();
    let mut both: Vec<(u64, u32)> = new1.iter().chain(new2).enumerate().map(|(i, &k)| (k, i as u32)).collect();
    both.par_sort_unstable();
    let mut dup = vec![false; 2 * half];
    let mut s = 0;
    while s < both.len() {
        let mut e = s + 1;
        while e < both.len() && both[e].0 == both[s].0 {
            e += 1;
        }
        if e - s > 1 {
            for item in &both[s..e] {
                dup[item.1 as usize] = true;
            }
        }
        s = e;
    }
    let second = dup.split_off(half);
    (dup, second)
}

/// `rewire`: degree-preserving directed edge swaps.
pub fn rewire(src: &[u32], dst: &[u32], n: usize, rng: &mut Generator, rounds: usize, cancel: &AtomicBool) -> Result<Vec<u32>, Error> {
    let nn = n as u64;
    let mut dst = dst.to_vec();
    let m = src.len();
    for _ in 0..rounds {
        check(cancel)?;
        let mut perm: Vec<u32> = (0..m as u32).collect();
        rng.shuffle(&mut perm);
        let half = m / 2;
        let (pi, pj) = (&perm[..half], &perm[half..2 * half]);
        let mut existing: Vec<u64> = src.iter().zip(&dst).map(|(&s, &d)| u64::from(s) * nn + u64::from(d)).collect();
        existing.par_sort_unstable();
        let (mut new1, mut new2) = (vec![0u64; half], vec![0u64; half]);
        let mut ok = vec![true; half];
        for t in 0..half {
            let (i, j) = (pi[t] as usize, pj[t] as usize);
            let (a, b, c, d) = (src[i], dst[i], src[j], dst[j]);
            new1[t] = u64::from(a) * nn + u64::from(d);
            new2[t] = u64::from(c) * nn + u64::from(b);
            ok[t] = a != d && c != b && !has_key(&existing, new1[t]) && !has_key(&existing, new2[t]);
        }
        let (d1, d2) = duplicated(&new1, &new2);
        let swaps: Vec<(usize, usize, u32, u32)> = (0..half)
            .filter(|&t| ok[t] && !d1[t] && !d2[t])
            .map(|t| {
                let (i, j) = (pi[t] as usize, pj[t] as usize);
                (i, j, dst[j], dst[i])
            })
            .collect();
        for (i, j, d, b) in swaps {
            dst[i] = d;
            dst[j] = b;
        }
    }
    Ok(dst)
}

/// `rewire_undirected`: degree-preserving swaps of an undirected edge list; returns (min, max) per edge.
pub fn rewire_undirected(
    lo: &[u32],
    hi: &[u32],
    n: usize,
    rng: &mut Generator,
    rounds: usize,
    cancel: &AtomicBool,
) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let nn = n as u64;
    let (mut a_, mut b_) = (lo.to_vec(), hi.to_vec());
    let m = a_.len();
    for _ in 0..rounds {
        check(cancel)?;
        for e in 0..m {
            if rng.random() < 0.5 {
                std::mem::swap(&mut a_[e], &mut b_[e]);
            }
        }
        let mut perm: Vec<u32> = (0..m as u32).collect();
        rng.shuffle(&mut perm);
        let half = m / 2;
        let (pi, pj) = (&perm[..half], &perm[half..2 * half]);
        let key = |x: u32, y: u32| u64::from(x.min(y)) * nn + u64::from(x.max(y));
        let mut existing: Vec<u64> = a_.iter().zip(&b_).map(|(&x, &y)| key(x, y)).collect();
        existing.par_sort_unstable();
        let (mut k1, mut k2) = (vec![0u64; half], vec![0u64; half]);
        let mut ok = vec![true; half];
        for t in 0..half {
            let (i, j) = (pi[t] as usize, pj[t] as usize);
            let (a, b, c, d) = (a_[i], b_[i], a_[j], b_[j]);
            k1[t] = key(a, d);
            k2[t] = key(c, b);
            ok[t] = a != d && c != b && !has_key(&existing, k1[t]) && !has_key(&existing, k2[t]);
        }
        let (d1, d2) = duplicated(&k1, &k2);
        let swaps: Vec<(usize, usize, u32, u32)> = (0..half)
            .filter(|&t| ok[t] && !d1[t] && !d2[t])
            .map(|t| {
                let (i, j) = (pi[t] as usize, pj[t] as usize);
                (i, j, b_[j], b_[i])
            })
            .collect();
        for (i, j, d, b) in swaps {
            b_[i] = d;
            b_[j] = b;
        }
    }
    let lo: Vec<u32> = a_.iter().zip(&b_).map(|(&x, &y)| x.min(y)).collect();
    let hi: Vec<u32> = a_.iter().zip(&b_).map(|(&x, &y)| x.max(y)).collect();
    Ok((lo, hi))
}

/// `(indptr, indices, degree)` of the undirected skeleton; neighbours above a node come first (ascending), then those below.
pub struct Skeleton {
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub deg: Vec<u32>,
}

/// The skeleton's unique edges `(lo, hi)` in ascending key order.
pub fn skel_edges(src: &[u32], dst: &[u32], n: usize) -> (Vec<u32>, Vec<u32>) {
    let nn = n as u64;
    let mut keys: Vec<u64> = src.iter().zip(dst).map(|(&s, &d)| u64::from(s.min(d)) * nn + u64::from(s.max(d))).collect();
    keys.par_sort_unstable();
    keys.dedup();
    (keys.iter().map(|&k| (k / nn) as u32).collect(), keys.iter().map(|&k| (k % nn) as u32).collect())
}

pub fn skeleton(src: &[u32], dst: &[u32], n: usize) -> Skeleton {
    let (a, b) = skel_edges(src, dst, n);
    let mut deg = vec![0u32; n];
    for (&x, &y) in a.iter().zip(&b) {
        deg[x as usize] += 1;
        deg[y as usize] += 1;
    }
    let mut indptr = vec![0u64; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + u64::from(deg[i]);
    }
    let mut fill = indptr.clone();
    let mut indices = vec![0u32; indptr[n] as usize];
    // stable argsort of u = [a, b] with v = [b, a]
    for (&x, &y) in a.iter().zip(&b) {
        indices[fill[x as usize] as usize] = y;
        fill[x as usize] += 1;
    }
    for (&x, &y) in a.iter().zip(&b) {
        indices[fill[y as usize] as usize] = x;
        fill[y as usize] += 1;
    }
    Skeleton { indptr, indices, deg }
}

// --- motifs ----------------------------------------------------------------------------------------------------------------
type Motif = (&'static str, &'static str, &'static [(usize, usize)]);
const PAIRS: [(usize, usize); 6] = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)];
pub const MOTIFS: [Motif; 13] = [
    ("021D", "out-star: one neuron sends to two others", &[(1, 0), (1, 2)]),
    ("021U", "in-star: two neurons send to one", &[(0, 1), (2, 1)]),
    ("021C", "chain: A -> B -> C", &[(0, 1), (1, 2)]),
    ("111a", "mutual pair, one of them also sends to a third", &[(0, 1), (1, 0), (1, 2)]),
    ("111b", "mutual pair, a third sends to one of them", &[(0, 1), (1, 0), (2, 1)]),
    ("030T", "feed-forward loop", &[(0, 1), (1, 2), (0, 2)]),
    ("030C", "3-cycle", &[(0, 1), (1, 2), (2, 0)]),
    ("201", "two mutual pairs sharing a neuron", &[(0, 1), (1, 0), (1, 2), (2, 1)]),
    ("120D", "a neuron sends to both of a mutual pair", &[(1, 0), (1, 2), (0, 2), (2, 0)]),
    ("120U", "both of a mutual pair send to a third", &[(0, 1), (2, 1), (0, 2), (2, 0)]),
    ("120C", "a chain closed by a mutual pair", &[(0, 1), (1, 2), (0, 2), (2, 0)]),
    ("210", "two mutual pairs and a one-way link", &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2)]),
    ("300", "all three pairs mutual", &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]),
];

fn motif_table_codes() -> [i8; 64] {
    let mut t = [-1i8; 64];
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for (mi, (_, _, edges)) in MOTIFS.iter().enumerate() {
        for p in perms {
            let mut code = 0usize;
            for &(a, b) in *edges {
                let e = (p[a], p[b]);
                let bit = PAIRS.iter().position(|&x| x == e).unwrap();
                code |= 1 << bit;
            }
            t[code] = mi as i8;
        }
    }
    t
}

pub struct MotifEstimate {
    pub counts: [f64; 13],
    pub se: [f64; 13],
    pub total_wedges: f64,
    pub connected_triples: f64,
}

/// `np.std(x, ddof=1)` of a float array.
fn std1(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = np_sum(x) / n;
    let sq: Vec<f64> = x.iter().map(|v| (v - mean) * (v - mean)).collect();
    (np_sum(&sq) / (n - 1.0)).sqrt()
}

pub fn motif_estimate(src: &[u32], dst: &[u32], n: usize, rng: &mut Generator, wedges: usize, sk: &Skeleton) -> MotifEstimate {
    let nn = n as u64;
    let wc: Vec<f64> = sk.deg.iter().map(|&d| f64::from(d) * (f64::from(d) - 1.0) / 2.0).collect();
    let total_wedges = np_sum(&wc);
    // rng.choice(n, size=wedges, p=wc / total): cdf of the probabilities, a uniform draw per sample, searchsorted right
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &w in &wc {
        acc += w / total_wedges;
        cdf.push(acc);
    }
    let last = *cdf.last().unwrap();
    for c in &mut cdf {
        *c /= last;
    }
    let centre: Vec<u32> = (0..wedges)
        .map(|_| {
            let u = rng.random();
            cdf.partition_point(|&c| c <= u) as u32
        })
        .collect();
    let d: Vec<f64> = centre.iter().map(|&c| f64::from(sk.deg[c as usize])).collect();
    let oa: Vec<i64> = (0..wedges).map(|i| (rng.random() * d[i]) as i64).collect();
    let ob: Vec<i64> = (0..wedges)
        .map(|i| {
            let v = (rng.random() * (d[i] - 1.0)) as i64;
            if v >= oa[i] { v + 1 } else { v }
        })
        .collect();
    let mut keys: Vec<u64> = src.iter().zip(dst).map(|(&s, &t)| u64::from(s) * nn + u64::from(t)).collect();
    keys.par_sort_unstable();
    let table = motif_table_codes();
    let mut cls = vec![-1i8; wedges];
    let mut closed = vec![false; wedges];
    for i in 0..wedges {
        let c = centre[i];
        let base = sk.indptr[c as usize] as usize;
        let a = sk.indices[base + oa[i] as usize];
        let b = sk.indices[base + ob[i] as usize];
        let mut bits = 0usize;
        for (bit, (x, y)) in [(c, a), (a, c), (c, b), (b, c), (a, b), (b, a)].into_iter().enumerate() {
            if has_key(&keys, u64::from(x) * nn + u64::from(y)) {
                bits |= 1 << bit;
            }
        }
        cls[i] = table[bits];
        closed[i] = (bits >> 4) & 3 != 0;
    }
    let w: Vec<f64> = closed.iter().map(|&c| if c { 1.0 / 3.0 } else { 1.0 }).collect();
    let mut counts = [0.0; 13];
    let mut se = [0.0; 13];
    for k in 0..13 {
        let x: Vec<f64> = (0..wedges).map(|i| if cls[i] == k as i8 { w[i] } else { 0.0 }).collect();
        counts[k] = total_wedges * (np_sum(&x) / wedges as f64);
        se[k] = total_wedges * (std1(&x) / (wedges as f64).sqrt());
    }
    MotifEstimate { counts, se, total_wedges, connected_triples: total_wedges * (np_sum(&w) / wedges as f64) }
}

pub fn motif_table(real: &MotifEstimate, null_counts: &[[f64; 13]], wedges: usize) -> Value {
    let mut rows = Vec::new();
    for (k, (name, desc, _)) in MOTIFS.iter().enumerate() {
        let col: Vec<f64> = null_counts.iter().map(|r| r[k]).collect();
        let mean = (!col.is_empty()).then(|| np_sum(&col) / col.len() as f64);
        let sd = (col.len() > 1).then(|| std1(&col));
        let realc = real.counts[k];
        let sd_eff = sd.map(|s| s.hypot(real.se[k]));
        let (mut enr, mut z, mut note) = (None, None, String::new());
        match mean {
            None => note = "undefined: no null graphs were made".into(),
            Some(m) if m <= 0.0 => {
                note = format!(
                    "undefined: none of the {} null graphs had this motif in {} sampled wedges each (the real graph has about {}); real / 0 has no value",
                    col.len(),
                    group(wedges as f64),
                    group(realc)
                );
            }
            Some(m) => {
                enr = Some(realc / m);
                z = sd_eff.filter(|&s| s > 0.0).map(|s| (realc - m) / s);
                if z.is_none() {
                    note = "z undefined: fewer than two null graphs, or no spread among them".into();
                }
            }
        }
        rows.push(json!({
            "motif": name, "description": desc, "count_estimate": realc, "sampling_se": real.se[k],
            "null_mean": mean, "null_sd": sd, "enrichment": enr, "z": z, "enrichment_note": note,
        }));
    }
    Value::Array(rows)
}

/// Python's `f"{x:,.0f}"`.
pub fn group(x: f64) -> String {
    let s = format!("{:.0}", x.abs());
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if x < 0.0 { format!("-{out}") } else { out }
}

pub fn fmt_enrichment(m: &Value) -> String {
    m["enrichment"].as_f64().map_or("undefined".into(), |e| format!("x{e:.2}"))
}

pub fn fmt_z(m: &Value) -> String {
    m["z"].as_f64().map_or("undefined".into(), |z| format!("{z:.1}"))
}

// --- rich club -----------------------------------------------------------------------------------------------------------------
fn rich_club(a: &[u32], b: &[u32], deg: &[u32], ks: &[f64]) -> Vec<(f64, usize, usize, f64)> {
    ks.iter()
        .map(|&k| {
            let nk = deg.iter().filter(|&&d| f64::from(d) > k).count();
            let ek = a.iter().zip(b).filter(|(x, y)| f64::from(deg[**x as usize].min(deg[**y as usize])) > k).count();
            let phi = if nk > 1 { 2.0 * ek as f64 / (nk as f64 * (nk as f64 - 1.0)) } else { f64::NAN };
            (k, nk, ek, phi)
        })
        .collect()
}

// --- communities ---------------------------------------------------------------------------------------------------------------
struct Sym {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    data: Vec<f64>,
}

fn symmetrised(src: &[u32], dst: &[u32], syn: &[f64], n: usize) -> Sym {
    let mut cnt = vec![0usize; n + 1];
    for (&s, &d) in src.iter().zip(dst) {
        cnt[s as usize + 1] += 1;
        cnt[d as usize + 1] += 1;
    }
    for i in 0..n {
        cnt[i + 1] += cnt[i];
    }
    let mut fill = cnt.clone();
    let mut ent: Vec<(u32, f64)> = vec![(0, 0.0); cnt[n]];
    for ((&s, &d), &w) in src.iter().zip(dst).zip(syn) {
        ent[fill[s as usize]] = (d, w);
        fill[s as usize] += 1;
        ent[fill[d as usize]] = (s, w);
        fill[d as usize] += 1;
    }
    let (mut indptr, mut indices, mut data) = (vec![0usize; n + 1], Vec::new(), Vec::new());
    let rows: Vec<Vec<(u32, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = ent[cnt[i]..cnt[i + 1]].to_vec();
            row.sort_by_key(|e| e.0);
            let mut out: Vec<(u32, f64)> = Vec::with_capacity(row.len());
            for (c, w) in row {
                match out.last_mut() {
                    Some(l) if l.0 == c => l.1 += w,
                    _ => out.push((c, w)),
                }
            }
            out
        })
        .collect();
    for (i, r) in rows.iter().enumerate() {
        for &(c, w) in r {
            indices.push(c);
            data.push(w);
        }
        indptr[i + 1] = indices.len();
    }
    Sym { n, indptr, indices, data }
}

/// `H.T @ A @ H` with SciPy's order of accumulation (see the module notes): `C[c, c'] = sum over k in cluster c', ascending, of
/// (sum over i in cluster c, ascending, of A[i, k])`.
fn aggregate(a: &Sym, comm: &[u32], m: usize) -> Sym {
    let n = a.n;
    // CSC of A: rows ascending within each column
    let mut cnt = vec![0usize; n + 1];
    for &c in &a.indices {
        cnt[c as usize + 1] += 1;
    }
    for i in 0..n {
        cnt[i + 1] += cnt[i];
    }
    let mut fill = cnt.clone();
    let mut rows = vec![0u32; a.indices.len()];
    let mut vals = vec![0.0f64; a.indices.len()];
    for i in 0..n {
        for e in a.indptr[i]..a.indptr[i + 1] {
            let c = a.indices[e] as usize;
            rows[fill[c]] = i as u32;
            vals[fill[c]] = a.data[e];
            fill[c] += 1;
        }
    }
    let mut targets: Vec<HashMap<u32, f64>> = vec![HashMap::new(); m];
    let (mut acc, mut touched) = (vec![0.0f64; m], Vec::<u32>::new());
    let mut seen = vec![false; m];
    for j in 0..n {
        for e in cnt[j]..cnt[j + 1] {
            let c = comm[rows[e] as usize] as usize;
            if !seen[c] {
                seen[c] = true;
                touched.push(c as u32);
            }
            acc[c] += vals[e];
        }
        let tj = comm[j] as usize;
        for &c in &touched {
            let v = acc[c as usize];
            if v != 0.0 {
                *targets[tj].entry(c).or_insert(0.0) += v;
            }
            acc[c as usize] = 0.0;
            seen[c as usize] = false;
        }
        touched.clear();
    }
    // rows c, columns c' ascending
    let mut per_row: Vec<Vec<(u32, f64)>> = vec![Vec::new(); m];
    for (cp, map) in targets.into_iter().enumerate() {
        for (c, v) in map {
            if v != 0.0 {
                per_row[c as usize].push((cp as u32, v));
            }
        }
    }
    let (mut indptr, mut indices, mut data) = (vec![0usize; m + 1], Vec::new(), Vec::new());
    for (i, r) in per_row.iter_mut().enumerate() {
        r.sort_by_key(|e| e.0);
        for &(c, v) in r.iter() {
            indices.push(c);
            data.push(v);
        }
        indptr[i + 1] = indices.len();
    }
    Sym { n: m, indptr, indices, data }
}

/// Louvain-style communities (`louvain`): `(label per neuron, modularity)`.
#[allow(clippy::too_many_arguments)]
pub fn louvain(
    src: &[u32],
    dst: &[u32],
    syn: &[f64],
    n: usize,
    rng: &mut Generator,
    sweeps: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str),
) -> Result<(Vec<u32>, f64), Error> {
    let mut a = symmetrised(src, dst, syn, n);
    // A.sum(): the row sums by matvec (sequential), then NumPy's pairwise sum over them
    let seq_rows: Vec<f64> = (0..a.n)
        .map(|i| {
            let mut s = 0.0;
            for e in a.indptr[i]..a.indptr[i + 1] {
                s += a.data[e];
            }
            s
        })
        .collect();
    let m2 = np_sum(&seq_rows);
    let mut labels: Vec<u32> = (0..n as u32).collect();
    let mut level = 0;
    loop {
        check(cancel)?;
        let nn = a.n;
        let k: Vec<f64> = (0..nn).map(|i| np_sum(&a.data[a.indptr[i]..a.indptr[i + 1]])).collect();
        let selfw: Vec<f64> = (0..nn)
            .map(|i| {
                let (s, e) = (a.indptr[i], a.indptr[i + 1]);
                a.indices[s..e].binary_search(&(i as u32)).map_or(0.0, |p| a.data[s + p])
            })
            .collect();
        let mut comm: Vec<u32> = (0..nn as u32).collect();
        let mut tot = k.clone();
        let mut moved_any = false;
        let (mut acc, mut touched) = (vec![0.0f64; nn], Vec::<u32>::new());
        let mut seen = vec![false; nn];
        for sweep in 0..sweeps {
            check(cancel)?;
            let mut best = vec![f64::NEG_INFINITY; nn];
            let mut stay = vec![0.0f64; nn];
            let mut pick = vec![-1i64; nn];
            for i in 0..nn {
                for e in a.indptr[i]..a.indptr[i + 1] {
                    let c = comm[a.indices[e] as usize] as usize;
                    if !seen[c] {
                        seen[c] = true;
                        touched.push(c as u32);
                    }
                    acc[c] += a.data[e];
                }
                touched.sort_unstable();
                let ki = k[i];
                let mut has_own = false;
                let mut row_best = f64::NEG_INFINITY;
                let mut gains: Vec<(u32, f64)> = Vec::with_capacity(touched.len());
                for &c in &touched {
                    let own = comm[i] == c;
                    let w = acc[c as usize];
                    let g = w - if own { selfw[i] } else { 0.0 } - ki * (tot[c as usize] - if own { ki } else { 0.0 }) / m2;
                    if own {
                        has_own = true;
                        stay[i] = g;
                    }
                    if g > row_best {
                        row_best = g;
                    }
                    gains.push((c, g));
                    acc[c as usize] = 0.0;
                    seen[c as usize] = false;
                }
                touched.clear();
                if !has_own {
                    stay[i] = -ki * (tot[comm[i] as usize] - ki) / m2;
                }
                if !gains.is_empty() {
                    best[i] = row_best;
                    pick[i] = gains.iter().find(|(_, g)| *g >= row_best - 1e-12).map_or(-1, |(c, _)| i64::from(*c));
                }
            }
            let mut moved: Vec<usize> = Vec::new();
            for i in 0..nn {
                let r = rng.random() < 0.5;
                if best[i] > stay[i] + 1e-12 && pick[i] >= 0 && pick[i] != i64::from(comm[i]) && r {
                    moved.push(i);
                }
            }
            if moved.is_empty() {
                if sweep == 0 {
                    break;
                }
                continue;
            }
            for &i in &moved {
                tot[comm[i] as usize] += -k[i];
            }
            for &i in &moved {
                tot[pick[i] as usize] += k[i];
            }
            for &i in &moved {
                comm[i] = pick[i] as u32;
            }
            moved_any = true;
            let mut uniq: Vec<u32> = comm.clone();
            uniq.sort_unstable();
            uniq.dedup();
            progress(&format!("communities: level {level}, sweep {}, {} communities", sweep + 1, uniq.len()));
            if moved.len() < (nn / 2000).max(1) {
                break;
            }
        }
        if !moved_any {
            break;
        }
        let mut uniq: Vec<u32> = comm.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let idx: HashMap<u32, u32> = uniq.iter().enumerate().map(|(i, &c)| (c, i as u32)).collect();
        let comm: Vec<u32> = comm.iter().map(|c| idx[c]).collect();
        for l in &mut labels {
            *l = comm[*l as usize];
        }
        a = aggregate(&a, &comm, uniq.len());
        level += 1;
        if a.n == 1 {
            break;
        }
    }
    let q = modularity(src, dst, syn, n, &labels);
    Ok((labels, q))
}

pub fn modularity(src: &[u32], dst: &[u32], syn: &[f64], n: usize, labels: &[u32]) -> f64 {
    let m2 = 2.0 * np_sum(syn);
    let mut k = bincount_w(src, syn, n);
    let kd = bincount_w(dst, syn, n);
    for (a, b) in k.iter_mut().zip(&kd) {
        *a += b;
    }
    let same: Vec<f64> =
        src.iter().zip(dst).zip(syn).filter(|((s, d), _)| labels[**s as usize] == labels[**d as usize]).map(|(_, &w)| w).collect();
    let inside = 2.0 * np_sum(&same);
    let nl = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut tot = vec![0.0f64; nl];
    for (i, &l) in labels.iter().enumerate() {
        tot[l as usize] += k[i];
    }
    let sq: Vec<f64> = tot.iter().map(|t| (t / m2) * (t / m2)).collect();
    inside / m2 - np_sum(&sq)
}

fn unique_inverse(a: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut u: Vec<String> = a.to_vec();
    u.sort();
    u.dedup();
    let idx: HashMap<&str, usize> = u.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let inv = a.iter().map(|s| idx[s.as_str()]).collect();
    (u, inv)
}

pub fn nmi(a: &[u32], b: &[String]) -> f64 {
    let mut ua: Vec<u32> = a.to_vec();
    ua.sort_unstable();
    ua.dedup();
    let ia: HashMap<u32, usize> = ua.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let (ub, ib) = unique_inverse(b);
    let (na, nb) = (ua.len(), ub.len());
    let mut cont = vec![0.0f64; na * nb];
    for (x, &y) in a.iter().zip(&ib) {
        cont[ia[x] * nb + y] += 1.0;
    }
    let total = np_sum(&cont);
    let p: Vec<f64> = cont.iter().map(|c| c / total).collect();
    let pa: Vec<f64> = (0..na).map(|i| np_sum(&p[i * nb..(i + 1) * nb])).collect();
    let mut pb = vec![0.0f64; nb];
    for i in 0..na {
        for j in 0..nb {
            pb[j] += p[i * nb + j];
        }
    }
    let terms: Vec<f64> = (0..na)
        .flat_map(|i| (0..nb).map(move |j| (i, j)))
        .filter(|&(i, j)| p[i * nb + j] > 0.0)
        .map(|(i, j)| p[i * nb + j] * (p[i * nb + j] / (pa[i] * pb[j])).ln())
        .collect();
    let mi = np_sum(&terms);
    let ha = -np_sum(&pa.iter().filter(|&&v| v > 0.0).map(|v| v * v.ln()).collect::<Vec<_>>());
    let hb = -np_sum(&pb.iter().filter(|&&v| v > 0.0).map(|v| v * v.ln()).collect::<Vec<_>>());
    if ha > 0.0 && hb > 0.0 { mi / (ha * hb).sqrt().max(1e-12) } else { 0.0 }
}

fn community_summary(labels: &[u32], g: &Graph, top: usize) -> Value {
    let nl = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut cnt = vec![0usize; nl];
    for &l in labels {
        cnt[l as usize] += 1;
    }
    let mut order: Vec<usize> = (0..nl).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(cnt[c]));
    let mut out = Vec::new();
    for &ci in order.iter().take(top) {
        let members: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] as usize == ci).collect();
        let top_of = |names: &[String]| {
            let mut m: Vec<&str> = members.iter().map(|&i| names[i].as_str()).collect();
            m.sort_unstable();
            let (mut best, mut best_n, mut i) = ("", 0usize, 0usize);
            while i < m.len() {
                let mut j = i;
                while j < m.len() && m[j] == m[i] {
                    j += 1;
                }
                if j - i > best_n {
                    best_n = j - i;
                    best = m[i];
                }
                i = j;
            }
            (best.to_string(), best_n as f64 / members.len() as f64)
        };
        let (reg, rs) = top_of(&g.region);
        let (sup, ss) = top_of(&g.superclass);
        out.push(json!({
            "community": ci, "size": cnt[ci], "top_region": reg, "top_region_share": rs,
            "top_superclass": sup, "top_superclass_share": ss,
        }));
    }
    Value::Array(out)
}

fn region_summary(g: &Graph) -> Value {
    let n = g.n;
    let (names, ridx) = unique_inverse(&g.region);
    let indeg = bincount(&g.dst, n);
    let outdeg = bincount(&g.src, n);
    let mut rows: Vec<(usize, Value)> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let members: Vec<usize> = (0..n).filter(|&j| ridx[j] == i).collect();
        let mean = |d: &[u32]| members.iter().map(|&j| f64::from(d[j])).sum::<f64>() / members.len() as f64;
        let (mut out_s, mut in_s, mut out_same, mut in_same, mut out_inh) = (vec![], vec![], vec![], vec![], vec![]);
        for e in 0..g.src.len() {
            let (rs, rd) = (ridx[g.src[e] as usize], ridx[g.dst[e] as usize]);
            let same = rs == rd;
            if rs == i {
                out_s.push(g.syn[e]);
                if same {
                    out_same.push(g.syn[e]);
                }
                if g.sign[e] < 0 {
                    out_inh.push(g.syn[e]);
                }
            }
            if rd == i {
                in_s.push(g.syn[e]);
                if same {
                    in_same.push(g.syn[e]);
                }
            }
        }
        let (osyn, isyn) = (np_sum(&out_s), np_sum(&in_s));
        let ratio = |x: f64, tot: f64| if tot != 0.0 { x / tot } else { f64::NAN };
        rows.push((
            members.len(),
            json!({
                "region": name, "neurons": members.len(), "share_of_neurons": members.len() as f64 / n as f64,
                "mean_in_degree": mean(&indeg), "mean_out_degree": mean(&outdeg),
                "out_synapses": osyn, "in_synapses": isyn,
                "share_out_staying_inside": ratio(np_sum(&out_same), osyn),
                "share_in_from_inside": ratio(np_sum(&in_same), isyn),
                "inhibitory_share_of_out_synapses": ratio(np_sum(&out_inh), osyn),
            }),
        ));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    Value::Array(rows.into_iter().map(|r| r.1).collect())
}

pub fn settings(nulls: usize, wedges: usize, seed: u64) -> Value {
    json!({
        "version": ANALYSIS_VERSION, "nulls": nulls, "wedges": wedges, "seed": seed,
        "rich_percentiles": RICH_PERCENTILES.iter().map(|&p| p as i64).collect::<Vec<_>>(),
        "sweeps_per_level": SWEEPS_PER_LEVEL,
    })
}

/// The whole analysis (`compute` without the cache). `progress(fraction, label)`.
#[allow(clippy::too_many_arguments)]
pub fn analyse(
    g: &Graph,
    brain: &str,
    pack_name: &str,
    pack_sha256: &str,
    created: &str,
    (nulls, wedges, seed): (usize, usize, u64),
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(f64, &str),
) -> Result<Value, Error> {
    let t0 = std::time::Instant::now();
    let progress = std::cell::RefCell::new(progress);
    let step = |frac: f64, label: &str, cancel: &AtomicBool| -> Result<(), Error> {
        check(cancel)?;
        (progress.borrow_mut())(frac, label);
        Ok(())
    };
    let (n, src, dst, syn) = (g.n, &g.src[..], &g.dst[..], &g.syn[..]);
    let mut rng = Generator::new(seed);
    step(0.04, "degrees and reciprocity", cancel)?;
    let deg = degrees(g);
    let mut rec = reciprocity(src, dst, syn, n);
    step(0.08, "the undirected skeleton", cancel)?;
    let sk = skeleton(src, dst, n);
    let se = skel_edges(src, dst, n);
    let mut sorted: Vec<f64> = sk.deg.iter().map(|&d| f64::from(d)).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut ks: Vec<f64> = RICH_PERCENTILES.iter().map(|&p| percentile_sorted(&sorted, p)).collect();
    ks.sort_by(|a, b| a.total_cmp(b));
    ks.dedup();
    let rich_real = rich_club(&se.0, &se.1, &sk.deg, &ks);
    step(0.12, "sampling motifs in the real graph", cancel)?;
    let mot_real = motif_estimate(src, dst, n, &mut rng, wedges, &sk);
    let (mut null_counts, mut null_rich, mut null_rec) = (Vec::<[f64; 13]>::new(), Vec::<Vec<f64>>::new(), Vec::<f64>::new());
    let mut rich_rng = Generator::new(seed + 2);
    for i in 0..nulls {
        let base = 0.15 + 0.4 * i as f64 / nulls.max(1) as f64;
        step(base, &format!("null graph {}/{nulls}: swapping edges", i + 1), cancel)?;
        let nd = rewire(src, dst, n, &mut rng, 10, cancel)?;
        step(base + 0.2 / nulls.max(1) as f64, &format!("null graph {}/{nulls}: motifs and rich club", i + 1), cancel)?;
        let nsk = skeleton(src, &nd, n);
        null_counts.push(motif_estimate(src, &nd, n, &mut rng, wedges, &nsk).counts);
        null_rec.push(binary_reciprocity(src, &nd, n));
        let (ulo, uhi) = rewire_undirected(&se.0, &se.1, n, &mut rich_rng, 10, cancel)?;
        null_rich.push(rich_club(&ulo, &uhi, &sk.deg, &ks).iter().map(|r| r.3).collect());
    }
    let motifs = motif_table(&mot_real, &null_counts, wedges);
    let mut rich = Vec::new();
    for (i, r) in rich_real.iter().enumerate() {
        let pn: Vec<f64> = null_rich.iter().map(|row| row[i]).filter(|v| !v.is_nan()).collect();
        let phi_null = if pn.is_empty() { f64::NAN } else { np_sum(&pn) / pn.len() as f64 };
        let rho = if phi_null.is_finite() && phi_null > 0.0 { r.3 / phi_null } else { f64::NAN };
        rich.push(json!({"k": r.0, "neurons": r.1, "edges": r.2, "phi": r.3, "phi_null": phi_null, "rho": rho}));
    }
    let rec_null = if null_rec.is_empty() { f64::NAN } else { np_sum(&null_rec) / null_rec.len() as f64 };
    rec["null_binary_mean"] = json!(rec_null);
    rec["enrichment"] = json!(if rec_null > 0.0 { rec["binary"].as_f64().unwrap() / rec_null } else { f64::NAN });
    step(0.62, "communities", cancel)?;
    let mut lrng = Generator::new(seed + 1);
    let (labels, q) = louvain(src, dst, syn, n, &mut lrng, SWEEPS_PER_LEVEL, cancel, &mut |s| (progress.borrow_mut())(0.77, s))?;
    let mut uniq = labels.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let comm = json!({
        "modularity": q, "communities": uniq.len(), "nmi_with_regions": nmi(&labels, &g.region),
        "nmi_with_superclass": nmi(&labels, &g.superclass), "largest": community_summary(&labels, g, 12),
    });
    step(0.95, "regions", cancel)?;
    let regions = region_summary(g);
    let result = json!({
        "kind": "netsci", "brain": brain, "analysis_version": ANALYSIS_VERSION, "settings": settings(nulls, wedges, seed),
        "pack_sha256": pack_sha256, "pack_name": pack_name, "created": created,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        "degrees": deg, "reciprocity": rec, "motifs": motifs,
        "motif_meta": {"total_wedges": mot_real.total_wedges, "connected_triples": mot_real.connected_triples,
                        "wedges_sampled": wedges, "null_graphs": nulls},
        "rich_club": rich, "communities": comm, "regions": regions,
        "tags": {"everything": "CONNECTOME (computed from the pack; no simulation)",
                 "analysis_choices": "GAME RULE (nulls, samples, seed, cut-offs, community method)"},
    });
    step(1.0, "done", cancel)?;
    Ok(result)
}

// --- reading and exporting -------------------------------------------------------------------------------------------------------
fn cell(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => csv_quote(s),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => crate::cards::py_float(f),
            _ => n.to_string(),
        },
        other => other.to_string(),
    }
}

fn csv_quote(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

/// One CSV per table into `folder`; the names and columns are Python's.
pub fn export_csv(res: &Value, folder: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(folder)?;
    let brain = res["brain"].as_str().unwrap_or("adult");
    let mut written = Vec::new();
    let mut w = |name: &str, header: &[&str], rows: Vec<Vec<Value>>| -> std::io::Result<()> {
        let p = folder.join(format!("netsci_{brain}_{name}.csv"));
        let mut text = header.join(",") + "\r\n";
        for r in rows {
            text.push_str(&r.iter().map(cell).collect::<Vec<_>>().join(","));
            text.push_str("\r\n");
        }
        std::fs::write(&p, text)?;
        written.push(p);
        Ok(())
    };
    let d = &res["degrees"];
    let mut hist = Vec::new();
    for k in ["in_degree", "out_degree"] {
        for h in d[k]["histogram"].as_array().into_iter().flatten() {
            hist.push(vec![json!(k.split('_').next().unwrap()), h["low"].clone(), h["high"].clone(), h["count"].clone()]);
        }
    }
    w("degree_histogram", &["direction", "degree_low", "degree_high", "neurons"], hist)?;
    let mut summ = Vec::new();
    for k in ["in_degree", "out_degree", "in_synapses", "out_synapses"] {
        let mut r = vec![json!(k)];
        for x in ["mean", "median", "p90", "p99", "max", "zero"] {
            r.push(d[k][x].clone());
        }
        summ.push(r);
    }
    w("degree_summary", &["quantity", "mean", "median", "p90", "p99", "max", "neurons_with_zero"], summ)?;
    w(
        "hubs",
        &["direction", "row", "type", "degree"],
        d["hubs"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|h| vec![h["direction"].clone(), h["row"].clone(), h["type"].clone(), h["degree"].clone()])
            .collect(),
    )?;
    let r = &res["reciprocity"];
    w(
        "reciprocity",
        &["binary", "weighted", "reciprocal_pairs", "null_binary_mean", "enrichment"],
        vec![["binary", "weighted", "reciprocal_pairs", "null_binary_mean", "enrichment"].iter().map(|k| r[*k].clone()).collect()],
    )?;
    let keys = ["motif", "description", "count_estimate", "sampling_se", "null_mean", "null_sd", "enrichment", "z", "enrichment_note"];
    w(
        "motifs",
        &["motif", "description", "count_estimate", "sampling_se", "null_mean", "null_sd", "enrichment", "z", "note"],
        res["motifs"].as_array().into_iter().flatten().map(|m| keys.iter().map(|k| m[*k].clone()).collect()).collect(),
    )?;
    let rk = ["k", "neurons", "edges", "phi", "phi_null", "rho"];
    w(
        "rich_club",
        &["k", "neurons_above_k", "edges_among_them", "phi", "phi_null", "rho"],
        res["rich_club"].as_array().into_iter().flatten().map(|x| rk.iter().map(|k| x[*k].clone()).collect()).collect(),
    )?;
    let c = &res["communities"];
    let ck = ["size", "top_region", "top_region_share", "top_superclass", "top_superclass_share"];
    w(
        "communities",
        &["rank", "size", "top_region", "top_region_share", "top_superclass", "top_superclass_share"],
        c["largest"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(i, x)| std::iter::once(json!(i + 1)).chain(ck.iter().map(|k| x[*k].clone())).collect())
            .collect(),
    )?;
    let gk = [
        "region",
        "neurons",
        "share_of_neurons",
        "mean_in_degree",
        "mean_out_degree",
        "out_synapses",
        "in_synapses",
        "share_out_staying_inside",
        "share_in_from_inside",
        "inhibitory_share_of_out_synapses",
    ];
    w("regions", &gk, res["regions"].as_array().into_iter().flatten().map(|x| gk.iter().map(|k| x[*k].clone()).collect()).collect())?;
    let mut about = vec![
        vec![json!("pack"), res["pack_name"].clone()],
        vec![json!("pack_sha256"), res["pack_sha256"].clone()],
        vec![json!("modularity"), c["modularity"].clone()],
        vec![json!("communities"), c["communities"].clone()],
        vec![json!("nmi_with_regions"), c["nmi_with_regions"].clone()],
    ];
    if let Some(s) = res["settings"].as_object() {
        // Python's dict order (version, nulls, wedges, seed, rich_percentiles, sweeps_per_level); python writes the list as its repr
        for k in ["version", "nulls", "wedges", "seed", "rich_percentiles", "sweeps_per_level"] {
            let v = &s[k];
            let shown = if let Some(a) = v.as_array() {
                json!(format!("[{}]", a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")))
            } else {
                v.clone()
            };
            about.push(vec![json!(k), shown]);
        }
    }
    about.push(vec![json!("tag"), json!("CONNECTOME for every number; analysis choices GAME RULE; no MODEL PREDICTION")]);
    about.push(vec![
        json!("note"),
        json!("motif counts are sampled estimates; the community partition is a heuristic, not a unique optimum"),
    ]);
    w("about", &["key", "value"], about)?;
    Ok(written)
}
