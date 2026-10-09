//! Neurodex (3.0): the type table behind the collectible encyclopedia and the rule that decides when a type counts as discovered.
//! Python: `core/neurodex.py`. No Bevy and no UI here: the game, the Lab and the tests all read this module.
//!
//! CONNECTOME: every number of an entry (a type's neuron count, superclass and regions as the pack annotates them, the transmitter the
//! dataset predicts with its confidence, the strongest partner types by synapse count). GAME RULE: "discovered" (see [`Tracker`]).

use std::collections::HashMap;

use crate::pack::Labels;

pub const DISCOVER_MIN_HZ: f64 = 6.0;
pub const DISCOVER_FACTOR: f64 = 3.0;
pub const DISCOVER_CALM_FLOOR_HZ: f64 = 2.0;
pub const DISCOVER_SUSTAIN: i16 = 3;
/// Spikes are counted over the last 150 ms (30 steps).
pub const WINDOW_STEPS: usize = 30;
/// One false discovery per this many hours of calm play, if every type fired as a Poisson process at its calm rate.
pub const FALSE_ALARM_HOURS: f64 = 100.0;
pub const SETTLE_CHECKS: u32 = 100;
pub const CHECK_STEPS: u32 = 10;
pub const CALM_K_QUIET: f64 = 0.02;
pub const CALM_K_BUSY: f64 = 0.0005;
pub const TOP_PARTNERS: usize = 6;
pub const UNNAMED: &str = "(no type)";
/// Transmitter sources that are the dataset's own (the larva pack's "inferred" is the game's sign rule, shown as no transmitter).
const DATASET_NT_SOURCES: [&str; 3] = ["ground_truth", "consensus_nt", "predicted_nt"];

/// `DISCOVER_ALPHA = 1 / (11_751 * (3600 / 0.05) * FALSE_ALARM_HOURS)`
pub fn discover_alpha() -> f64 {
    1.0 / (11_751.0 * (3600.0 / 0.05) * FALSE_ALARM_HOURS)
}

/// The signed synapse counts as CSR `[post, pre]` (what the partner matrix is summed from).
pub struct RawCsr<'a> {
    pub indptr: &'a [i32],
    pub indices: &'a [i32],
    pub data: &'a [i16],
}

#[derive(Debug, Clone)]
pub struct TypeTable {
    pub brain: String,
    pub names: Vec<String>,
    pub count: Vec<u32>,
    pub superclass: Vec<String>,
    /// Primary region (most common; "unassigned" only if nothing else).
    pub region: Vec<String>,
    /// Per type: (region, neurons), most common first.
    pub regions: Vec<Vec<(String, u32)>>,
    /// Most common predicted transmitter ("" = none in the dataset).
    pub nt: Vec<String>,
    /// Mean confidence of that call (NaN = none).
    pub nt_conf: Vec<f32>,
    /// Share of the type's neurons whose transmitter was measured (0..1).
    pub nt_truth: Vec<f32>,
    /// Type index per neuron, -1 for a neuron without a type.
    pub type_id: Vec<i32>,
    /// `in_rows[post type]`: (pre type, synapses) ascending by pre type; the last index is "no type".
    in_rows: Vec<Vec<(u32, i64)>>,
    out_cols: Vec<Vec<(u32, i64)>>,
    index: HashMap<String, usize>,
    order: Vec<u32>,
    start: Vec<usize>,
    pub body_id: Vec<i64>,
}

/// `_mode`: the most common value, ties to the one seen first.
fn mode<'a>(values: impl Iterator<Item = &'a str>) -> String {
    let mut counts: Vec<(&str, u32)> = vec![];
    for v in values {
        match counts.iter_mut().find(|c| c.0 == v) {
            Some(c) => c.1 += 1,
            None => counts.push((v, 1)),
        }
    }
    let mut best: Option<(&str, u32)> = None;
    for c in counts {
        if best.is_none_or(|b| c.1 > b.1) {
            best = Some(c);
        }
    }
    best.map(|b| b.0.to_string()).unwrap_or_default()
}

impl TypeTable {
    /// `build_table`: from a pack's labels (and its signed synapse CSR, for the partners).
    pub fn build(labels: &Labels, brain: &str, raw: Option<RawCsr>) -> TypeTable {
        let n = labels.types.len();
        let not_a_type: &[&str] = if brain == "larva" { &["unassigned"] } else { &[] };
        let named = |t: &str| !t.is_empty() && !not_a_type.contains(&t);
        let mut names: Vec<String> = labels.types.iter().filter(|t| named(t)).cloned().collect();
        names.sort();
        names.dedup();
        let index: HashMap<String, usize> = names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let t_n = names.len();
        let type_id: Vec<i32> = labels.types.iter().map(|t| if named(t) { index[t] as i32 } else { -1 }).collect();
        let mut count = vec![0u32; t_n];
        for &t in &type_id {
            if t >= 0 {
                count[t as usize] += 1;
            }
        }
        // rows of each type, ascending neuron order (stable argsort of the type ids)
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&r| type_id[r as usize]);
        let tid_sorted: Vec<i32> = order.iter().map(|&r| type_id[r as usize]).collect();
        let start: Vec<usize> = (0..=t_n as i32).map(|t| tid_sorted.partition_point(|&x| x < t)).collect();
        let has_nt = !labels.nt.is_empty() && labels.nt.iter().any(|s| !s.is_empty());
        let nt_of = |r: usize| -> &str {
            let src = labels.nt_source.get(r).map_or("", String::as_str);
            if DATASET_NT_SOURCES.contains(&src) { labels.nt[r].as_str() } else { "" }
        };
        let (mut superclass, mut region, mut regions, mut nts) = (vec![], vec![], vec![], vec![]);
        let (mut nt_conf, mut nt_truth) = (vec![f32::NAN; t_n], vec![0f32; t_n]);
        for i in 0..t_n {
            let rows = &order[start[i]..start[i + 1]];
            superclass.push(mode(rows.iter().map(|&r| labels.superclass[r as usize].as_str())));
            // Counter(reg[rows]).most_common(): by count, ties in first-seen order
            let mut rc: Vec<(String, u32)> = vec![];
            for &r in rows {
                let g = labels.region.get(r as usize).map_or("unassigned", String::as_str);
                match rc.iter_mut().find(|c| c.0 == g) {
                    Some(c) => c.1 += 1,
                    None => rc.push((g.to_string(), 1)),
                }
            }
            rc.sort_by_key(|c| std::cmp::Reverse(c.1));
            region.push(rc.iter().find(|(r, _)| r != "unassigned").map_or("unassigned".to_string(), |(r, _)| r.clone()));
            regions.push(rc);
            if !has_nt {
                nts.push(String::new());
                continue;
            }
            let label = mode(rows.iter().map(|&r| nt_of(r as usize)).filter(|v| !v.is_empty()));
            if !label.is_empty() {
                let (mut sum, mut k) = (0f64, 0u32);
                for &r in rows.iter().filter(|&&r| nt_of(r as usize) == label) {
                    let c = labels.nt_conf.get(r as usize).copied().unwrap_or(f32::NAN);
                    if !c.is_nan() {
                        sum += c as f64;
                        k += 1;
                    }
                }
                if k > 0 {
                    nt_conf[i] = (sum / k as f64) as f32;
                }
                let truth = rows.iter().filter(|&&r| labels.nt_source.get(r as usize).is_some_and(|s| s == "ground_truth")).count();
                nt_truth[i] = (truth as f64 / rows.len().max(1) as f64) as f32;
            }
            nts.push(label);
        }
        let (mut in_rows, mut out_cols) = (vec![Vec::new(); t_n + 1], vec![Vec::new(); t_n + 1]);
        if let Some(csr) = raw {
            let tid = |r: usize| -> u32 { if type_id[r] >= 0 { type_id[r] as u32 } else { t_n as u32 } };
            let mut maps: Vec<HashMap<u32, i64>> = vec![HashMap::new(); t_n + 1];
            for post in 0..n {
                let tp = tid(post) as usize;
                for e in csr.indptr[post] as usize..csr.indptr[post + 1] as usize {
                    *maps[tp].entry(tid(csr.indices[e] as usize)).or_insert(0) += (csr.data[e] as i64).abs();
                }
            }
            for (tp, m) in maps.into_iter().enumerate() {
                let mut row: Vec<(u32, i64)> = m.into_iter().collect();
                row.sort_unstable_by_key(|e| e.0);
                for &(tq, v) in &row {
                    out_cols[tq as usize].push((tp as u32, v));
                }
                in_rows[tp] = row;
            }
            for c in &mut out_cols {
                c.sort_unstable_by_key(|e| e.0);
            }
        }
        TypeTable {
            brain: brain.into(),
            names,
            count,
            superclass,
            region,
            regions,
            nt: nts,
            nt_conf,
            nt_truth,
            type_id,
            in_rows,
            out_cols,
            index,
            order,
            start,
            body_id: labels.body_id.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// The neuron rows of a type, ascending.
    pub fn rows(&self, name: &str) -> &[u32] {
        match self.index(name) {
            Some(i) => &self.order[self.start[i]..self.start[i + 1]],
            None => &[],
        }
    }

    pub fn region_names(&self) -> Vec<String> {
        let mut r: Vec<String> = self.region.clone();
        r.sort();
        r.dedup();
        r.sort_by_key(|x| x == "unassigned");
        r
    }

    /// `top_partners`: (partner type, synapses, share), strongest first. "in": partners that synapse onto this type; "out": partners it
    /// synapses onto. The share is of this type's total input (or output) synapses, counting those on neurons without a type.
    pub fn top_partners(&self, name: &str, direction: &str, k: usize) -> Vec<(String, i64, f64)> {
        let Some(i) = self.index(name) else { return vec![] };
        let entries = if direction == "in" { &self.in_rows[i] } else { &self.out_cols[i] };
        if entries.is_empty() {
            return vec![];
        }
        let total: f64 = entries.iter().map(|e| e.1 as f64).sum();
        let mut idx: Vec<usize> = (0..entries.len()).collect();
        idx.sort_by_key(|&j| std::cmp::Reverse(entries[j].1)); // stable: ties keep ascending type order
        idx.into_iter()
            .take(k)
            .map(|j| {
                let (t, v) = entries[j];
                let nm = if (t as usize) < self.names.len() { self.names[t as usize].clone() } else { UNNAMED.into() };
                (nm, v, v as f64 / total)
            })
            .collect()
    }

    /// `region_progress`: (region, discovered, total) by primary region, regions with types only, largest first, "(all)" last.
    pub fn region_progress(&self, known: &dyn Fn(&str) -> bool) -> Vec<(String, u32, u32)> {
        let mut tot: Vec<(String, u32, u32)> = vec![];
        for (name, reg) in self.names.iter().zip(&self.region) {
            let e = match tot.iter_mut().find(|r| &r.0 == reg) {
                Some(e) => e,
                None => {
                    tot.push((reg.clone(), 0, 0));
                    tot.last_mut().unwrap()
                }
            };
            e.2 += 1;
            e.1 += u32::from(known(name));
        }
        tot.sort_by(|a, b| (a.0 == "unassigned").cmp(&(b.0 == "unassigned")).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
        let (got, all) = tot.iter().fold((0, 0), |a, r| (a.0 + r.1, a.1 + r.2));
        tot.push(("(all)".into(), got, all));
        tot
    }

    /// `list_entries`: type indices for the panel list, filtered by primary region, by a substring of a discovered type's name,
    /// sorted by name. An undiscovered type's name is not revealed by search.
    pub fn list_entries(
        &self,
        known: &dyn Fn(&str) -> bool,
        region: Option<&str>,
        query: &str,
        only_found: bool,
        only_curated: Option<&dyn Fn(&str) -> bool>,
    ) -> Vec<usize> {
        let q = query.trim().to_lowercase();
        (0..self.names.len())
            .filter(|&i| {
                let n = &self.names[i];
                if region.is_some_and(|r| r != "(all)" && self.region[i] != r) {
                    return false;
                }
                let found = known(n);
                if only_found && !found {
                    return false;
                }
                if only_curated.is_some_and(|c| !c(n)) {
                    return false;
                }
                !(!q.is_empty() && !(found && n.to_lowercase().contains(&q)))
            })
            .collect()
    }
}

// --- the discovery rule (GAME RULE) ------------------------------------------------------------------------------------------------

/// `scipy.stats.poisson.sf(k - 1, lam)` = P(X >= k) for k >= 1: the regularised lower incomplete gamma P(k, lam).
pub fn poisson_tail(k: f64, lam: f64) -> f64 {
    if k <= 0.0 {
        return 1.0;
    }
    if lam <= 0.0 {
        return 0.0;
    }
    let ln_gamma = ln_gamma(k + 1.0);
    if lam < k + 1.0 {
        // series: P(k, x) = x^k e^-x / Gamma(k+1) * sum_n x^n / ((k+1)...(k+n))
        let (mut term, mut sum, mut n) = (1.0f64, 1.0f64, 0.0f64);
        loop {
            n += 1.0;
            term *= lam / (k + n);
            sum += term;
            if term < sum * 1e-16 || n > 5000.0 {
                break;
            }
        }
        (k * lam.ln() - lam - ln_gamma).exp() * sum
    } else {
        // continued fraction for Q(k, x), then P = 1 - Q (a large value either way: never below the tiny alpha)
        let tiny = 1e-300;
        let mut b = lam + 1.0 - k;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..5000 {
            let an = -(i as f64) * (i as f64 - k);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        1.0 - (k * lam.ln() - lam - ln_gamma).exp() * h
    }
}

/// Lanczos approximation of ln Gamma(x), x > 0.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Found {
    pub type_index: usize,
    pub name: String,
    pub ratio: f64,
    /// "play", "stimulated" or "rest"
    pub how: &'static str,
}

struct BrainState {
    calm: Vec<f64>,
    n: u32,
    run: Vec<i16>,
}

/// Watches running brains and reports discoveries. One per game; feed it each brain's recent spikes. A separate calm baseline is kept per
/// brain (`key`), because each fly's brain settles on its own.
pub struct Tracker {
    count: Vec<f64>,
    known: Vec<bool>,
    state: HashMap<u64, BrainState>,
    alpha: f64,
}

impl Tracker {
    pub fn new(tab: &TypeTable, known: &dyn Fn(&str) -> bool) -> Self {
        let mut t = Tracker {
            count: tab.count.iter().map(|&c| f64::from(c.max(1))).collect(),
            known: vec![],
            state: HashMap::new(),
            alpha: discover_alpha(),
        };
        t.sync(tab, known);
        t
    }

    /// Re-read which types are known (after the progress file changed).
    pub fn sync(&mut self, tab: &TypeTable, known: &dyn Fn(&str) -> bool) {
        self.known = tab.names.iter().map(|n| known(n)).collect();
    }

    /// `observe`: `spikes` are the neuron indices of every spike in the last `window_steps` steps (repeated once per spike), `dt` the
    /// step length in seconds. Returns the types discovered just now.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        tab: &TypeTable,
        key: u64,
        spikes: &[u32],
        dt: f64,
        calm: bool,
        driven: bool,
        window_steps: usize,
    ) -> Vec<Found> {
        let t_n = tab.len();
        let mut counts = vec![0f64; t_n];
        for &s in spikes {
            let t = tab.type_id[s as usize];
            if t >= 0 {
                counts[t as usize] += 1.0;
            }
        }
        let window_s = window_steps.max(1) as f64 * dt;
        let hz: Vec<f64> = counts.iter().zip(&self.count).map(|(c, n)| c / n / window_s).collect();
        let st = self.state.entry(key).or_insert_with(|| BrainState { calm: hz.clone(), n: 0, run: vec![0; t_n] });
        st.n += 1;
        let k = if calm || st.n < SETTLE_CHECKS { CALM_K_QUIET } else { CALM_K_BUSY };
        if st.n > 1 {
            for (c, h) in st.calm.iter_mut().zip(&hz) {
                *c += (h - *c) * k;
            }
        }
        if st.n < SETTLE_CHECKS {
            return vec![];
        }
        let base: Vec<f64> = st.calm.iter().map(|c| c.max(DISCOVER_CALM_FLOOR_HZ)).collect();
        let mut hot: Vec<bool> =
            (0..t_n).map(|i| hz[i] >= DISCOVER_MIN_HZ && hz[i] / base[i] >= DISCOVER_FACTOR && !self.known[i]).collect();
        if calm && !driven {
            hot.iter_mut().for_each(|h| *h = false);
        }
        for i in 0..t_n {
            if hot[i] {
                let lam = base[i] * self.count[i] * window_s;
                if poisson_tail(counts[i], lam) >= self.alpha {
                    hot[i] = false;
                }
            }
        }
        let mut out = vec![];
        for i in 0..t_n {
            if hot[i] {
                st.run[i] += 1;
            } else {
                st.run[i] = 0;
            }
            if st.run[i] >= DISCOVER_SUSTAIN {
                out.push(Found {
                    type_index: i,
                    name: tab.names[i].clone(),
                    ratio: hz[i] / base[i],
                    how: if driven {
                        "stimulated"
                    } else if calm {
                        "rest"
                    } else {
                        "play"
                    },
                });
                self.known[i] = true;
                st.run[i] = 0;
            }
        }
        out
    }
}
