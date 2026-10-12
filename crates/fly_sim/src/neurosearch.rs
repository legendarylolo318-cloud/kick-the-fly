//! Find neurons and the strongest paths between them (`lab/neurosearch.py`), for the big brain view's search box and PATH FROM/TO buttons.
//!
//! `search`: neurons by body ID, row number, type or instance. `top_paths`: the k strongest paths of at most 3 synapses; a path's strength
//! is the product of |weight| along it (the simulator's own weights: roughly the share of the target's input that flows along it), its sign
//! the product of the synapses' signs. All connectome, nothing tuned.

use std::collections::HashMap;

use crate::csr::Weights;
use crate::pack::Labels;

/// Rows matching `query`, best first: an exact body ID (else the row number), then an exact type, types starting with it, instances starting
/// with it, then any type or instance containing it (case-insensitive).
pub fn search(labels: &Labels, query: &str, limit: usize) -> Vec<u32> {
    let q = query.trim();
    if q.is_empty() {
        return vec![];
    }
    let n = labels.types.len();
    let mut out: Vec<u32> = vec![];
    if q.chars().all(|c| c.is_ascii_digit())
        && let Ok(id) = q.parse::<i64>()
    {
        out.extend(labels.body_id.iter().enumerate().filter(|&(_, &b)| b == id).map(|(i, _)| i as u32));
        if (id as u64) < n as u64 && out.is_empty() {
            out.push(id as u32); // a row number, as the inspector shows (#123)
        }
    }
    let lq = q.to_lowercase();
    let types: Vec<String> = labels.types.iter().map(|t| t.to_lowercase()).collect();
    let inst: Vec<String> = (0..n).map(|i| labels.instance.get(i).map_or_else(String::new, |s| s.to_lowercase())).collect();
    type Pred<'a> = Box<dyn Fn(usize) -> bool + 'a>;
    let masks: [Pred; 4] = [
        Box::new(|i| types[i] == lq),
        Box::new(|i| types[i].starts_with(&lq)),
        Box::new(|i| inst[i].starts_with(&lq)),
        Box::new(|i| types[i].contains(&lq) || inst[i].contains(&lq)),
    ];
    for mask in &masks {
        for i in 0..n {
            if mask(i) {
                if out.len() >= limit {
                    return out;
                }
                if !out.contains(&(i as u32)) {
                    out.push(i as u32);
                }
            }
        }
    }
    out.truncate(limit);
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub nodes: Vec<u32>,
    pub weights: Vec<f64>,
    pub strength: f64,
    pub sign: i32,
}

/// `heapq` as Python runs it, over `(strength, tie)` keys, so tied strengths come out in Python's order.
struct Heap {
    a: Vec<(f64, u64, Path)>,
}

impl Heap {
    fn lt(x: &(f64, u64, Path), y: &(f64, u64, Path)) -> bool {
        x.0 < y.0 || (x.0 == y.0 && x.1 < y.1)
    }

    fn sift_down(&mut self, startpos: usize, mut pos: usize) {
        while pos > startpos {
            let parent = (pos - 1) >> 1;
            if Self::lt(&self.a[pos], &self.a[parent]) {
                self.a.swap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    fn sift_up(&mut self, mut pos: usize) {
        let end = self.a.len();
        let start = pos;
        let mut child = 2 * pos + 1;
        while child < end {
            let right = child + 1;
            if right < end && !Self::lt(&self.a[child], &self.a[right]) {
                child = right;
            }
            self.a.swap(pos, child);
            pos = child;
            child = 2 * pos + 1;
        }
        self.sift_down(start, pos);
    }

    fn push(&mut self, item: (f64, u64, Path)) {
        self.a.push(item);
        let last = self.a.len() - 1;
        self.sift_down(0, last);
    }

    fn replace(&mut self, item: (f64, u64, Path)) {
        self.a[0] = item;
        self.sift_up(0);
    }
}

/// The k strongest paths `src -> ... -> dst` with at most `max_hops` synapses, strongest first. Exhaustive over 1, 2 and 3 hops.
pub fn top_paths(w: &Weights, src: u32, dst: u32, k: usize, max_hops: usize) -> Vec<Path> {
    if src == dst {
        return vec![];
    }
    let n = w.n();
    if src as usize >= n || dst as usize >= n {
        return vec![];
    }
    let mut heap = Heap { a: Vec::new() };
    let mut tie = 0u64;
    let mut offer = |heap: &mut Heap, nodes: Vec<u32>, weights: Vec<f64>| {
        let mut s = 1.0f64;
        for x in &weights {
            s *= x.abs();
        }
        if s <= 0.0 || s.is_nan() {
            return;
        }
        tie += 1;
        let sign = if s == 0.0 { 1 } else { weights.iter().fold(1.0f64, |a, x| a * x).signum() as i32 };
        let sign = if sign == 0 { 1 } else { sign };
        let item = (s, tie, Path { nodes, weights, strength: s, sign });
        if heap.a.len() < k {
            heap.push(item);
        } else if s > heap.a[0].0 {
            heap.replace(item);
        }
    };
    // outputs of src: (post rows, weights) ascending; inputs of dst
    let mut outs: Vec<(u32, f32)> = vec![];
    for b in &w.blocks {
        let (a, z) = (b.colptr[src as usize] as usize, b.colptr[src as usize + 1] as usize);
        for i in a..z {
            outs.push((b.row0 as u32 + b.rows[i], b.data[i]));
        }
    }
    let (in_idx, in_w) = w.csr.row(dst as usize);
    if let Some(p) = in_idx.iter().position(|&c| c == src) {
        offer(&mut heap, vec![src, dst], vec![f64::from(in_w[p])]);
    }
    let w_to: HashMap<u32, f32> = outs.iter().copied().collect();
    if max_hops >= 2 {
        for (&m, &w2) in in_idx.iter().zip(in_w) {
            if let Some(&w1) = w_to.get(&m)
                && m != src
                && m != dst
            {
                offer(&mut heap, vec![src, m, dst], vec![f64::from(w1), f64::from(w2)]);
            }
        }
    }
    if max_hops >= 3 && !outs.is_empty() && !in_idx.is_empty() {
        let a_ok: Vec<(u32, f32)> = outs.iter().copied().filter(|(a, _)| *a != src && *a != dst).collect();
        let b_ok: Vec<(u32, f32)> = in_idx.iter().copied().zip(in_w.iter().copied()).filter(|(b, _)| *b != src && *b != dst).collect();
        if !a_ok.is_empty() && !b_ok.is_empty() {
            let wa: HashMap<u32, f32> = a_ok.iter().copied().collect();
            // every 3-hop path scored in full; only the strongest few go through the heap
            let mut cand: Vec<(f64, u32, u32, f32, f32, f32)> = vec![];
            for &(b, wbv) in &b_ok {
                let (cols, vals) = w.csr.row(b as usize);
                for (&a, &wab) in cols.iter().zip(vals) {
                    if let Some(&wav) = wa.get(&a) {
                        let score = f64::from(wav).abs() * f64::from(wab).abs() * f64::from(wbv).abs();
                        cand.push((score, a, b, wav, wab, wbv));
                    }
                }
            }
            let mut idx: Vec<usize> = (0..cand.len()).collect();
            idx.sort_by(|&x, &y| cand[y].0.total_cmp(&cand[x].0));
            idx.truncate((k * 4).max(20));
            for t in idx {
                let (_, a, b, wav, wab, wbv) = cand[t];
                if a != b {
                    offer(&mut heap, vec![src, a, b, dst], vec![f64::from(wav), f64::from(wab), f64::from(wbv)]);
                }
            }
        }
    }
    // `sorted(best, key=-strength)`: stable over the heap's array order
    let mut items = heap.a;
    items.sort_by(|a, b| b.0.total_cmp(&a.0));
    items.into_iter().map(|(_, _, p)| p).collect()
}
