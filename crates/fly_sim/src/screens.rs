//! Whole-brain screens (`lab/screens.py`, 3.1.0 tasks 6 and 7): the activation screen (hold one cell type driven at a time and record the
//! descending neurons' response and the behavior readouts) and the knockout screen (silence cell types for each validated behavior and
//! rank them by how much the response drops). MODEL PREDICTION.
//!
//! Every condition of a seed starts from the same snapshot of one warmed brain (a condition differs from the unperturbed run only by the
//! perturbation, noise included); an effect is its change minus its matched controls' change (random sets of the same size and the same
//! superclasses), tested with a paired t-test over seeds and corrected over the whole screen with Benjamini-Hochberg. Resumable: every
//! finished job is appended to `records.jsonl`; a different run in the same folder is refused.
//!
//! Not ported: the Parquet copies (Python writes them when pyarrow is installed; the CSVs hold everything), the GPU batch mode.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::brain::{Brain, BrainState};
use crate::cards::Build;
use crate::csr::Weights;
use crate::pack::Labels;
use crate::rng::Generator;
use crate::stats;
use crate::validation::{self, np_sum};

pub const SCREEN_SEEDS: [u64; 8] = [4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007];
pub const PRE: usize = 400;
pub const STIM: usize = 400;
pub const AMP: f32 = 0.5;
pub const DT: f64 = 0.005;
pub const MIN_SIGN_SHARE: f64 = 0.75;
pub const MIN_EFFECT_HZ: f64 = 1.0;
pub const DN_Z: f64 = 4.0;
pub const DN_MIN_HZ: f64 = 0.5;
pub const Q_MAX: f64 = 0.05;
pub const BATCH_DROP: f64 = 0.10;
pub const BATCH_Q: f64 = 0.10;
pub const VERSION: i64 = 1;
pub const TAG: &str = "MODEL PREDICTION";

/// `(readout group, what a rise in it is called)`.
pub const READOUTS: [(&str, &str); 10] = [
    ("dnp01", "escape (giant fiber)"),
    ("dnp09", "forward walking command"),
    ("mdn", "backward walking command (moonwalker)"),
    ("adn", "antennal grooming command"),
    ("mn9", "proboscis extension"),
    ("mn_legs", "leg motor output"),
    ("pip10", "courtship song command"),
    ("ps1", "courtship song motor output"),
    ("dna_steer_r", "steer right"),
    ("dna_steer_l", "steer left"),
];

/// The pathway behaviors that pass in docs/validation.md: the knockout screen's default behaviors.
pub const VALIDATED_PATHWAYS: [&str; 11] = [
    "looming_escape",
    "sugar_feeding",
    "antenna_grooming_circuit",
    "courtship_song",
    "bitter_grn_to_dng28",
    "co2_orn_to_pn",
    "hot_trn_to_vp2pn",
    "cold_trn_to_vp3pn",
    "optomotor_turning",
    "or67d_to_da1pn",
    "da1pn_to_lh_asp",
];

fn readout_label(id: &str) -> &'static str {
    READOUTS.iter().find(|r| r.0 == id).map_or("", |r| r.1)
}

type Row = Vec<(String, Value)>;

fn row_get<'a>(r: &'a Row, k: &str) -> &'a Value {
    r.iter().find(|(n, _)| n == k).map_or(&Value::Null, |(_, v)| v)
}

fn row_set(r: &mut Row, k: &str, v: Value) {
    if let Some(e) = r.iter_mut().find(|(n, _)| n == k) {
        e.1 = v;
    } else {
        r.push((k.to_string(), v));
    }
}

fn fnum(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn to_object(r: &Row) -> Value {
    Value::Object(r.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

// --- the catalog and matched controls ---------------------------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct CatEntry {
    pub name: String,
    pub neurons: usize,
    pub superclass: String,
}

/// `catalog`: the cell types of the pack with their sizes and (majority) superclass, in name order.
pub fn catalog(
    labels: &Labels,
    min_neurons: usize,
    max_neurons: Option<usize>,
    superclasses: Option<&[String]>,
    types: Option<&[String]>,
    max_types: Option<usize>,
) -> Result<Vec<CatEntry>, String> {
    let mut by: std::collections::BTreeMap<&str, Vec<usize>> = Default::default();
    for (i, t) in labels.types.iter().enumerate() {
        if !t.is_empty() {
            by.entry(t.as_str()).or_default().push(i);
        }
    }
    let mut out = Vec::new();
    for (name, idx) in &by {
        let n = idx.len();
        if n < min_neurons || max_neurons.is_some_and(|m| m > 0 && n > m) {
            continue;
        }
        let mut cnt: std::collections::BTreeMap<&str, usize> = Default::default();
        for &i in idx {
            *cnt.entry(labels.superclass[i].as_str()).or_default() += 1;
        }
        // np.unique order (sorted), argmax = the first maximum
        let mut best = ("", 0usize);
        for (s, c) in &cnt {
            if *c > best.1 {
                best = (s, *c);
            }
        }
        if superclasses.is_some_and(|sc| !sc.iter().any(|s| s == best.0)) {
            continue;
        }
        out.push(CatEntry { name: (*name).to_string(), neurons: n, superclass: best.0.to_string() });
    }
    if let Some(types) = types.filter(|t| !t.is_empty()) {
        let mut want: Vec<&String> = Vec::new();
        for t in types {
            if !want.contains(&t) {
                want.push(t);
            }
        }
        let missing: Vec<&str> = want.iter().filter(|w| !out.iter().any(|c| &c.name == **w)).map(|w| w.as_str()).collect();
        if !missing.is_empty() {
            return Err(format!("unknown or filtered-out cell types: {}", missing.iter().take(5).copied().collect::<Vec<_>>().join(", ")));
        }
        out.retain(|c| want.iter().any(|w| **w == c.name));
    }
    if let Some(m) = max_types.filter(|m| *m > 0) {
        out.truncate(m);
    }
    Ok(out)
}

/// `_seed_of`: the first four bytes of sha256 of the `|`-joined parts, little endian.
pub fn seed_of(parts: &[String]) -> u64 {
    let h = Sha256::digest(parts.join("|").as_bytes());
    u64::from(u32::from_le_bytes([h[0], h[1], h[2], h[3]]))
}

/// `matched_random`: a random set the size of `rows` with the same count from each superclass, never overlapping `rows` or `exclude`.
pub fn matched_random(superclass: &[String], n: usize, rows: &[u32], exclude: &[u32], seed: u64) -> Vec<u32> {
    let mut taken = vec![false; n];
    for &r in rows.iter().chain(exclude) {
        taken[r as usize] = true;
    }
    let mut rng = Generator::new(seed);
    let mut classes: Vec<&str> = rows.iter().map(|&r| superclass[r as usize].as_str()).collect();
    classes.sort_unstable();
    classes.dedup();
    let mut out: Vec<u32> = Vec::new();
    for s in classes {
        let need = rows.iter().filter(|&&r| superclass[r as usize] == s).count();
        let pool: Vec<u32> = (0..n as u32).filter(|&i| superclass[i as usize] == s && !taken[i as usize]).collect();
        let k = need.min(pool.len());
        let idx = rng.choice_indices(pool.len(), k);
        out.extend(idx.into_iter().map(|i| pool[i]));
    }
    out.sort_unstable();
    out
}

// --- one brain: snapshot, then conditions -----------------------------------------------------------------------------------------
/// A warmed brain for one seed, snapshotted after `PRE` calm steps. `restore` + a perturbation + a window is one condition.
pub struct Arena {
    pub br: Brain,
    pub g: HashMap<&'static str, Vec<u32>>,
    snap: BrainState,
    pub dn_rows: Vec<u32>,
    dn_inv: Vec<usize>,
    pub dn_names: Vec<String>,
    dn_sizes: Vec<f64>,
}

impl Arena {
    pub fn new(weights: &Arc<Weights>, labels: &Arc<Labels>, seed: u64, pre: usize) -> Self {
        let build = Build { weights, labels, mode: "off", sigma: None, lif: crate::cards::LIF_DEFAULTS, wired: None };
        let mut br = build.brain_with(seed, None, 600, false);
        let g = validation::groups(labels, weights);
        validation::rest(&mut br, pre);
        let snap = br.export_state(seed);
        let dn = g["dn"].clone();
        let mut names: Vec<String> = dn.iter().map(|&r| labels.types[r as usize].clone()).collect();
        names.sort();
        names.dedup();
        let dn_inv: Vec<usize> = dn.iter().map(|&r| names.binary_search(&labels.types[r as usize]).unwrap_or(0)).collect();
        let mut dn_sizes = vec![0.0f64; names.len()];
        for &i in &dn_inv {
            dn_sizes[i] += 1.0;
        }
        Self { br, g, snap, dn_rows: dn, dn_inv, dn_names: names, dn_sizes }
    }

    pub fn restore(&mut self) {
        self.br.import_state(&self.snap);
    }

    /// Step `steps`; `(readout Hz, DN type rates in Hz)`.
    pub fn window(&mut self, steps: usize, readouts: &[&[u32]], dn: bool) -> (Vec<f64>, Option<Vec<f64>>) {
        let n = self.br.n();
        let masks: Vec<Vec<bool>> = readouts
            .iter()
            .map(|rows| {
                let mut m = vec![false; n];
                for &r in rows.iter() {
                    m[r as usize] = true;
                }
                m
            })
            .collect();
        let mut counts = vec![0usize; readouts.len()];
        let mut acc = vec![0i64; if dn { self.dn_rows.len() } else { 0 }];
        let mut dn_pos = vec![-1i64; if dn { n } else { 0 }];
        if dn {
            for (i, &r) in self.dn_rows.iter().enumerate() {
                dn_pos[r as usize] = i as i64;
            }
        }
        for _ in 0..steps {
            let s = self.br.step();
            for (k, m) in masks.iter().enumerate() {
                counts[k] += s.iter().filter(|&&x| m[x as usize]).count();
            }
            if dn {
                for &x in s {
                    let p = dn_pos[x as usize];
                    if p >= 0 {
                        acc[p as usize] += 1;
                    }
                }
            }
        }
        let hz: Vec<f64> = counts.iter().zip(readouts).map(|(&c, rows)| c as f64 / rows.len().max(1) as f64 / steps as f64 / DT).collect();
        let rates = dn.then(|| {
            let mut sum = vec![0.0f64; self.dn_names.len()];
            for (i, &a) in acc.iter().enumerate() {
                sum[self.dn_inv[i]] += a as f64;
            }
            sum.iter().zip(&self.dn_sizes).map(|(s, n)| s / n.max(1.0) / steps as f64 / DT).collect()
        });
        (hz, rates)
    }

    pub fn drive_window(&mut self, rows: &[u32], steps: usize, readouts: &[&[u32]], dn: bool, amp: f32) -> (Vec<f64>, Option<Vec<f64>>) {
        self.restore();
        if !rows.is_empty() {
            self.br.drive(rows, amp);
        }
        let r = self.window(steps, readouts, dn);
        if !rows.is_empty() {
            self.br.undrive(rows);
        }
        r
    }

    /// `silence_response`: validation's pathway test under a lesion; `(baseline Hz, driven Hz)`.
    pub fn silence_response(&mut self, lesion: &[u32], drive: &[u32], readout: &[u32], pre: usize, stim: usize) -> (f64, f64) {
        self.restore();
        if !lesion.is_empty() {
            self.br.set_surgery(lesion, -1);
        }
        let r = validation::pathway_response(&mut self.br, &[(drive.to_vec(), AMP)], &[readout], pre, stim)[0];
        if !lesion.is_empty() {
            self.br.set_surgery(lesion, 0);
        }
        (r.base_hz, r.driven_hz)
    }
}

fn round3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap_or(x)
}

fn sparse(v: &[f64], floor: f64) -> Value {
    Value::Array(v.iter().enumerate().filter(|(_, x)| x.abs() >= floor).map(|(i, x)| json!([i, round3(*x)])).collect())
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn engine() -> Value {
    json!(["rust", "cpu"])
}

// --- the activation screen --------------------------------------------------------------------------------------------------------
struct ActJob {
    seed: u64,
    types: Vec<String>,
    controls: usize,
    steps: usize,
}

fn activation_job(weights: &Arc<Weights>, labels: &Arc<Labels>, job: &ActJob) -> Vec<Value> {
    let mut ar = Arena::new(weights, labels, job.seed, PRE);
    let g: Vec<Vec<u32>> = READOUTS.iter().map(|r| ar.g[r.0].clone()).collect();
    let readouts: Vec<&[u32]> = g.iter().map(Vec::as_slice).collect();
    let mut exclude: Vec<u32> = g.iter().flatten().copied().collect();
    exclude.sort_unstable();
    exclude.dedup();
    let n = labels.types.len();
    let (base_hz, base_dn) = ar.drive_window(&[], job.steps, &readouts, true, AMP);
    let base_dn = base_dn.unwrap_or_default();
    let mut out = Vec::new();
    for name in &job.types {
        let rows: Vec<u32> = (0..n as u32).filter(|&i| labels.types[i as usize] == *name).collect();
        let (hz, dn) = ar.drive_window(&rows, job.steps, &readouts, true, AMP);
        let dn = dn.unwrap_or_default();
        let mut c = Vec::new();
        for j in 0..job.controls {
            let crow = matched_random(
                &labels.superclass,
                n,
                &rows,
                &exclude,
                seed_of(&["act".into(), name.clone(), job.seed.to_string(), j.to_string()]),
            );
            let (chz, cdn) = ar.drive_window(&crow, job.steps, &readouts, true, AMP);
            c.push(json!({"r": sub(&chz, &base_hz).iter().map(|x| round3(*x)).collect::<Vec<_>>(),
                          "dn": sparse(&sub(&cdn.unwrap_or_default(), &base_dn), 0.02)}));
        }
        let ov: Vec<usize> =
            g.iter().enumerate().filter(|(_, r)| r.iter().any(|x| rows.binary_search(x).is_ok())).map(|(i, _)| i).collect();
        out.push(json!({"k": "act", "t": name, "s": job.seed, "n": rows.len(),
                        "r": sub(&hz, &base_hz).iter().map(|x| round3(*x)).collect::<Vec<_>>(),
                        "dn": sparse(&sub(&dn, &base_dn), 0.02), "c": c, "ov": ov}));
    }
    out.push(json!({"k": "base", "s": job.seed, "r": base_hz.iter().map(|x| round3(*x)).collect::<Vec<_>>(), "engine": engine(), "n_dn": ar.dn_names.len()}));
    out
}

/// `paired_p`: `(two-sided paired t-test p, share of seeds whose change has the sign of the mean change)`.
pub fn paired_p(treated: &[f64], control: &[f64]) -> (f64, f64) {
    let d: Vec<f64> = sub(treated, control).into_iter().filter(|x| !x.is_nan()).collect();
    if d.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let (m, sd) = (validation::mean(&d), validation::sd(&d));
    let share = if m != 0.0 {
        let sgn = |x: f64| {
            if x > 0.0 {
                1
            } else if x < 0.0 {
                -1
            } else {
                0
            }
        };
        d.iter().filter(|&&x| sgn(x) == sgn(m)).count() as f64 / d.len() as f64
    } else {
        0.0
    };
    if sd == 0.0 {
        return (if m != 0.0 { 0.0 } else { 1.0 }, share);
    }
    (stats::t_two_sided(m.abs() / (sd / (d.len() as f64).sqrt()), (d.len() - 1) as f64), share)
}

/// `bh`: Benjamini-Hochberg q-values (NaN stays NaN).
pub fn bh(p: &[f64]) -> Vec<f64> {
    let mut q = vec![f64::NAN; p.len()];
    let mut ok: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    if ok.is_empty() {
        return q;
    }
    ok.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let m = ok.len() as f64;
    let raw: Vec<f64> = ok.iter().enumerate().map(|(r, &i)| p[i] * m / (r as f64 + 1.0)).collect();
    let mut run = f64::INFINITY;
    for r in (0..ok.len()).rev() {
        run = run.min(raw[r]);
        q[ok[r]] = run.min(1.0);
    }
    q
}

fn dense(sparse: &Value, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for e in sparse.as_array().into_iter().flatten() {
        if let (Some(i), Some(x)) = (e[0].as_u64().filter(|&i| (i as usize) < n), e[1].as_f64()) {
            v[i as usize] = x;
        }
    }
    v
}

/// Sequential column means of rows (`np.mean(rows, axis=0)`: a reduction over the first axis adds row by row).
fn col_mean(rows: &[Vec<f64>]) -> Vec<f64> {
    let k = rows.first().map_or(0, Vec::len);
    let mut s = vec![0.0; k];
    for r in rows {
        for (a, b) in s.iter_mut().zip(r) {
            *a += b;
        }
    }
    s.iter().map(|x| x / rows.len() as f64).collect()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().into_iter().flatten().map(num).collect()
}

/// `aggregate_activation`: per type, the effect over matched controls on each readout (paired over seeds), the call and the DN responders.
pub fn aggregate_activation(records: &[Value], cat: &[CatEntry], seeds: &[u64], dn_names: &[String]) -> (Vec<Row>, Value) {
    let acts: Vec<&Value> = records.iter().filter(|r| r["k"] == "act" && seeds.contains(&r["s"].as_u64().unwrap_or(u64::MAX))).collect();
    let n_dn = records.iter().filter(|r| r["k"] == "base").map(|b| b["n_dn"].as_u64().unwrap_or(0)).max().unwrap_or(0) as usize;
    let nr = READOUTS.len();
    let mut by_type: std::collections::BTreeMap<&str, Vec<&Value>> = Default::default();
    for r in &acts {
        by_type.entry(r["t"].as_str().unwrap_or("")).or_default().push(r);
    }
    // the noise of each readout and each DN type: the spread of the matched controls' changes over the whole screen
    let ctl: Vec<Vec<f64>> = acts.iter().flat_map(|r| r["c"].as_array().into_iter().flatten().map(|c| f64s(&c["r"]))).collect();
    let noise_r: Vec<f64> = if ctl.len() > 1 {
        let mean = col_mean(&ctl);
        let mut ss = vec![0.0; nr];
        for r in &ctl {
            for j in 0..nr {
                let x = r[j] - mean[j];
                ss[j] += x * x;
            }
        }
        ss.iter().map(|s| (s / (ctl.len() - 1) as f64).sqrt().max(0.05)).collect()
    } else {
        vec![1.0; nr]
    };
    let noise_dn: Vec<f64> = if n_dn > 0 {
        let mut sq = vec![0.0; n_dn];
        let mut cnt = 0usize;
        for r in &acts {
            for c in r["c"].as_array().into_iter().flatten() {
                for (s, x) in sq.iter_mut().zip(dense(&c["dn"], n_dn)) {
                    *s += x * x;
                }
                cnt += 1;
            }
        }
        sq.iter().map(|s| (s / cnt.max(1) as f64).sqrt().max(0.05)).collect()
    } else {
        Vec::new()
    };
    let meta: HashMap<&str, &CatEntry> = cat.iter().map(|c| (c.name.as_str(), c)).collect();
    let mut rows: Vec<Row> = Vec::new();
    let mut dn_lists: Vec<Vec<(String, f64, f64)>> = Vec::new();
    let mut p_all: Vec<f64> = Vec::new();
    let mut owners: Vec<(usize, &str)> = Vec::new();
    for (name, recs) in &by_type {
        let mut recs = recs.clone();
        recs.sort_by_key(|r| r["s"].as_u64().unwrap_or(0));
        let a: Vec<Vec<f64>> = recs.iter().map(|r| f64s(&r["r"])).collect();
        let c: Vec<Vec<f64>> =
            recs.iter().map(|r| col_mean(&r["c"].as_array().into_iter().flatten().map(|x| f64s(&x["r"])).collect::<Vec<_>>())).collect();
        let d: Vec<Vec<f64>> = a.iter().zip(&c).map(|(x, y)| sub(x, y)).collect();
        let ns = recs.len();
        let mut row: Row = vec![
            ("type".into(), json!(name)),
            ("superclass".into(), json!(meta.get(name).map_or("", |c| c.superclass.as_str()))),
            ("neurons".into(), recs[0]["n"].clone()),
            ("n_seeds".into(), json!(ns)),
        ];
        let mut overlap: Vec<usize> =
            recs[0]["ov"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
        overlap.sort_unstable();
        row.push(("self_readouts".into(), json!(overlap.iter().map(|&i| READOUTS[i].0).collect::<Vec<_>>().join(","))));
        for (j, (k, _)) in READOUTS.iter().enumerate() {
            let col: Vec<f64> = d.iter().map(|r| r[j]).collect();
            let ci = stats::mean_ci(&col);
            let (pv, share) = if overlap.contains(&j) {
                (f64::NAN, f64::NAN)
            } else {
                paired_p(&a.iter().map(|r| r[j]).collect::<Vec<_>>(), &c.iter().map(|r| r[j]).collect::<Vec<_>>())
            };
            row.push((format!("eff_{k}"), fnum(ci.mean)));
            row.push((format!("lo_{k}"), fnum(ci.lo)));
            row.push((format!("hi_{k}"), fnum(ci.hi)));
            row.push((format!("p_{k}"), fnum(pv)));
            row.push((format!("sign_{k}"), fnum(share)));
            row.push((format!("z_{k}"), fnum(ci.mean / (noise_r[j] / (ns as f64).sqrt()))));
            p_all.push(pv);
            owners.push((rows.len(), k));
        }
        let mut dn_list = Vec::new();
        if n_dn > 0 {
            let dn_a: Vec<Vec<f64>> = recs.iter().map(|r| dense(&r["dn"], n_dn)).collect();
            let dn_c: Vec<Vec<f64>> = recs
                .iter()
                .map(|r| col_mean(&r["c"].as_array().into_iter().flatten().map(|x| dense(&x["dn"], n_dn)).collect::<Vec<_>>()))
                .collect();
            let diffs: Vec<Vec<f64>> = dn_a.iter().zip(&dn_c).map(|(x, y)| sub(x, y)).collect();
            let mean_d = col_mean(&diffs);
            let z: Vec<f64> = mean_d.iter().zip(&noise_dn).map(|(m, n)| m / (n / (ns as f64).sqrt())).collect();
            let own = dn_names.iter().position(|n| n == name);
            let mut resp: Vec<usize> =
                (0..n_dn).filter(|&i| z[i].abs() >= DN_Z && mean_d[i].abs() >= DN_MIN_HZ && Some(i) != own).collect();
            resp.sort_by(|&x, &y| z[y].abs().total_cmp(&z[x].abs()));
            row.push(("dn_responders".into(), json!(resp.len())));
            dn_list =
                resp.iter().take(25).map(|&i| (dn_names.get(i).cloned().unwrap_or_else(|| format!("DN{i}")), mean_d[i], z[i])).collect();
            let top: Vec<String> = dn_list.iter().take(5).map(|(n, x, _)| format!("{n} {}Hz", fmt_signed1(*x))).collect();
            row.push(("top_dn".into(), json!(top.join("; "))));
        }
        dn_lists.push(dn_list);
        rows.push(row);
    }
    let q = bh(&p_all);
    for ((i, k), qv) in owners.iter().zip(&q) {
        row_set(&mut rows[*i], &format!("q_{k}"), fnum(*qv));
    }
    for row in &mut rows {
        let (mut best, mut best_z): (Option<&str>, f64) = (None, 0.0);
        for (k, _) in &READOUTS {
            let (qv, eff, sg, z) = (
                num(row_get(row, &format!("q_{k}"))),
                num(row_get(row, &format!("eff_{k}"))),
                num(row_get(row, &format!("sign_{k}"))),
                num(row_get(row, &format!("z_{k}"))),
            );
            if qv < Q_MAX && eff.abs() >= MIN_EFFECT_HZ && sg >= MIN_SIGN_SHARE && z.abs() > best_z {
                best = Some(k);
                best_z = z.abs();
            }
        }
        row_set(row, "behavior_readout", json!(best.unwrap_or("")));
        match best {
            Some(b) => {
                let e = num(row_get(row, &format!("eff_{b}")));
                row_set(row, "behavior_call", json!(format!("{}{}", if e > 0.0 { "raises " } else { "lowers " }, readout_label(b))));
                row_set(row, "behavior_effect_hz", fnum(e));
                let qb = row_get(row, &format!("q_{b}")).clone();
                row_set(row, "behavior_q", qb);
            }
            None => {
                row_set(row, "behavior_call", json!("none"));
                row_set(row, "behavior_effect_hz", json!(0.0));
                row_set(row, "behavior_q", Value::Null);
            }
        }
    }
    // sort: (-|behavior_effect_hz|, type), stable
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&x, &y| {
        let (a, b) = (num(row_get(&rows[x], "behavior_effect_hz")).abs(), num(row_get(&rows[y], "behavior_effect_hz")).abs());
        b.total_cmp(&a).then_with(|| row_get(&rows[x], "type").as_str().cmp(&row_get(&rows[y], "type").as_str()))
    });
    let mut sorted: Vec<Row> = Vec::new();
    for (rank, &i) in order.iter().enumerate() {
        let mut r = rows[i].clone();
        row_set(&mut r, "rank", json!(rank + 1));
        // the descending neurons moved, for the detail panel (`_dn`): kept in the JSON, not in the CSV
        row_set(&mut r, "_dn", Value::Array(dn_lists[i].iter().map(|(n, x, z)| json!([n, x, z])).collect()));
        sorted.push(r);
    }
    let n_called = sorted.iter().filter(|r| row_get(r, "behavior_readout").as_str().is_some_and(|s| !s.is_empty())).count();
    let summary = json!({
        "kind": "activation", "tag": TAG, "seeds": seeds,
        "readouts": READOUTS.iter().map(|(k, l)| json!({"id": k, "label": l})).collect::<Vec<_>>(),
        "noise_readout_hz": READOUTS.iter().zip(&noise_r).map(|((k, _), n)| (k.to_string(), json!(n))).collect::<serde_json::Map<_, _>>(),
        "n_types": sorted.len(), "n_called": n_called, "n_dn_types": n_dn,
        "criteria": format!(
            "called when q < {} and |change| >= {} Hz over matched controls; DN responder when |z| >= {} and >= {} Hz",
            crate::cards::py_float(Q_MAX), crate::cards::py_float(MIN_EFFECT_HZ), crate::cards::py_float(DN_Z), crate::cards::py_float(DN_MIN_HZ)
        ),
    });
    (sorted, summary)
}

/// Python's `f"{x:+.1f}"`.
fn fmt_signed1(x: f64) -> String {
    format!("{x:+.1}")
}

// --- the knockout screen ----------------------------------------------------------------------------------------------------------
struct KoJob {
    seed: u64,
    behavior: String,
    specs: Vec<Vec<String>>,
    controls: usize,
}

fn knockout_job(weights: &Arc<Weights>, labels: &Arc<Labels>, job: &KoJob) -> Vec<Value> {
    let t = validation::tests().iter().find(|t| t["id"].as_str() == Some(job.behavior.as_str())).cloned().unwrap_or(Value::Null);
    let mut ar = Arena::new(weights, labels, job.seed, PRE);
    let drive = ar.g[t["drive"].as_str().unwrap_or("")].clone();
    let readout = ar.g[t["readout"].as_str().unwrap_or("")].clone();
    let mut exclude: Vec<u32> = drive.iter().chain(&readout).copied().collect();
    for x in t["extra_readouts"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        exclude.extend(&ar.g[x]);
    }
    exclude.sort_unstable();
    exclude.dedup();
    let n = labels.types.len();
    let mut out = Vec::new();
    let base = ar.silence_response(&[], &drive, &readout, PRE, STIM);
    out.push(json!({"k": "ko_base", "b": job.behavior, "s": job.seed, "base": base.0, "driven": base.1, "engine": engine()}));
    for spec in &job.specs {
        let rows: Vec<u32> = (0..n as u32).filter(|&i| spec.contains(&labels.types[i as usize])).collect();
        let (bh_, dh) = ar.silence_response(&rows, &drive, &readout, PRE, STIM);
        let mut c = Vec::new();
        for j in 0..job.controls {
            let crow = matched_random(
                &labels.superclass,
                n,
                &rows,
                &exclude,
                seed_of(&["ko".into(), job.behavior.clone(), spec.join("+"), job.seed.to_string(), j.to_string()]),
            );
            let (cb, cd) = ar.silence_response(&crow, &drive, &readout, PRE, STIM);
            c.push(json!([cb, cd]));
        }
        out.push(json!({"k": "ko", "b": job.behavior, "s": job.seed, "spec": spec, "n": rows.len(), "base": bh_, "driven": dh, "c": c}));
    }
    out
}

type KoTable = std::collections::BTreeMap<Vec<String>, std::collections::BTreeMap<u64, (f64, f64)>>;

/// `_knock_table`: `{spec: {seed: (lesion evoked, mean matched-control evoked)}}` and the unperturbed evoked response per seed.
fn knock_table(records: &[Value], bid: &str, seeds: &[u64]) -> (std::collections::BTreeMap<u64, f64>, KoTable) {
    let base = records
        .iter()
        .filter(|r| r["k"] == "ko_base" && r["b"] == bid)
        .map(|r| (r["s"].as_u64().unwrap_or(0), num(&r["driven"]) - num(&r["base"])))
        .collect();
    let mut out: KoTable = Default::default();
    for r in records.iter().filter(|r| r["k"] == "ko" && r["b"] == bid && seeds.contains(&r["s"].as_u64().unwrap_or(u64::MAX))) {
        let diffs: Vec<f64> = r["c"].as_array().into_iter().flatten().map(|c| num(&c[1]) - num(&c[0])).collect();
        let ctl = np_sum(&diffs) / diffs.len().max(1) as f64;
        let spec: Vec<String> = r["spec"].as_array().into_iter().flatten().map(|s| s.as_str().unwrap_or("").to_string()).collect();
        out.entry(spec).or_default().insert(r["s"].as_u64().unwrap_or(0), (num(&r["driven"]) - num(&r["base"]), ctl));
    }
    (base, out)
}

/// `_open_batches`: the members of the batches that cut the response, per behavior.
fn open_batches(records: &[Value], behaviors: &[String], seeds: &[u64]) -> Vec<(String, Vec<String>)> {
    let mut opened = Vec::new();
    for b in behaviors {
        let (base, table) = knock_table(records, b, seeds);
        let (mut names, mut ps, mut drops): (Vec<&Vec<String>>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());
        for (spec, per) in &table {
            if spec.len() < 2 || per.len() < 2 {
                continue;
            }
            let les: Vec<f64> = per.values().map(|v| v.0).collect();
            let ctl: Vec<f64> = per.values().map(|v| v.1).collect();
            let bs: Vec<f64> = per.keys().filter_map(|s| base.get(s).copied()).collect();
            let b0 = if bs.is_empty() { f64::NAN } else { np_sum(&bs) / bs.len() as f64 };
            names.push(spec);
            let (cm, lm) = (np_sum(&ctl) / ctl.len() as f64, np_sum(&les) / les.len() as f64);
            drops.push(if b0 != 0.0 && !b0.is_nan() { (cm - lm) / b0 } else { f64::NAN });
            ps.push(paired_p(&ctl, &les).0);
        }
        let q = bh(&ps);
        let mut members = Vec::new();
        for ((spec, d), qv) in names.iter().zip(&drops).zip(&q) {
            if *d >= BATCH_DROP && *qv < BATCH_Q {
                members.extend(spec.iter().cloned());
            }
        }
        opened.push((b.clone(), members));
    }
    opened
}

fn neurons_of(records: &[Value], bid: &str, spec: &[String]) -> u64 {
    records
        .iter()
        .find(|r| {
            r["k"] == "ko"
                && r["b"] == bid
                && r["spec"]
                    .as_array()
                    .is_some_and(|a| a.len() == spec.len() && a.iter().zip(spec).all(|(x, y)| x.as_str() == Some(y.as_str())))
        })
        .and_then(|r| r["n"].as_u64())
        .unwrap_or(0)
}

/// `aggregate_knockout`.
pub fn aggregate_knockout(
    records: &[Value],
    behaviors: &[String],
    seeds: &[u64],
    cands: &HashMap<String, Vec<Value>>,
    batch_size: usize,
) -> Value {
    let mut rows: Vec<Row> = Vec::new();
    let mut p_all = Vec::new();
    let mut owners = Vec::new();
    let mut summary = serde_json::Map::new();
    for b in behaviors {
        let (base, table) = knock_table(records, b, seeds);
        let b0s: Vec<f64> = base.values().copied().collect();
        let b0 = if b0s.is_empty() { f64::NAN } else { np_sum(&b0s) / b0s.len() as f64 };
        summary.insert(b.clone(), json!({"unperturbed_evoked_hz": fnum(b0), "seeds": b0s.len()}));
        let cmeta: HashMap<&str, &Value> = cands.get(b).into_iter().flatten().filter_map(|c| c["type"].as_str().map(|t| (t, c))).collect();
        for (spec, per) in &table {
            let les: Vec<f64> = per.values().map(|v| v.0).collect();
            let ctl: Vec<f64> = per.values().map(|v| v.1).collect();
            let ci = stats::mean_ci(&sub(&ctl, &les));
            let (pv, share) = paired_p(&ctl, &les);
            let single = spec.len() == 1;
            let c = if single { cmeta.get(spec[0].as_str()) } else { None };
            let row: Row = vec![
                ("behavior".into(), json!(b)),
                ("candidate".into(), json!(spec.join("+"))),
                ("is_batch".into(), json!(!single)),
                ("members".into(), json!(spec.len())),
                ("neurons".into(), json!(neurons_of(records, b, spec))),
                ("n_seeds".into(), json!(per.len())),
                ("unperturbed_evoked_hz".into(), fnum(b0)),
                ("lesion_evoked_hz".into(), fnum(np_sum(&les) / les.len() as f64)),
                ("control_evoked_hz".into(), fnum(np_sum(&ctl) / ctl.len() as f64)),
                ("drop_hz".into(), fnum(ci.mean)),
                ("drop_lo".into(), fnum(ci.lo)),
                ("drop_hi".into(), fnum(ci.hi)),
                ("drop_share".into(), if b0 != 0.0 { fnum(ci.mean / b0) } else { Value::Null }),
                ("p".into(), fnum(pv)),
                ("sign_share".into(), fnum(share)),
                ("drives_the_behavior".into(), json!(single && c.is_some_and(|c| c["drives_the_behavior"].as_bool().unwrap_or(false)))),
                ("influence".into(), if single { c.map_or(Value::Null, |c| c["influence"].clone()) } else { Value::Null }),
            ];
            rows.push(row);
            p_all.push(pv);
            owners.push(rows.len() - 1);
        }
    }
    let q = bh(&p_all);
    for (i, qv) in owners.iter().zip(&q) {
        row_set(&mut rows[*i], "q", fnum(*qv));
    }
    for b in behaviors {
        let mut idx: Vec<usize> = (0..rows.len())
            .filter(|&i| row_get(&rows[i], "behavior").as_str() == Some(b) && row_get(&rows[i], "is_batch") == &json!(false))
            .collect();
        let key = |i: usize| {
            let d = num(row_get(&rows[i], "drop_share"));
            if d.is_nan() { 9.0 } else { -d }
        };
        idx.sort_by(|&x, &y| key(x).total_cmp(&key(y)));
        for (r, &i) in idx.iter().enumerate() {
            row_set(&mut rows[i], "rank", json!(r + 1));
        }
        let sig = idx
            .iter()
            .filter(|&&i| {
                num(row_get(&rows[i], "q")) < Q_MAX
                    && num(row_get(&rows[i], "drop_share")) >= 0.1
                    && num(row_get(&rows[i], "sign_share")) >= MIN_SIGN_SHARE
            })
            .count();
        if let Some(s) = summary.get_mut(b).and_then(Value::as_object_mut) {
            s.insert("tested".into(), json!(idx.len()));
            s.insert("significant".into(), json!(sig));
        }
    }
    for r in &mut rows {
        if !r.iter().any(|(k, _)| k == "rank") {
            r.push(("rank".into(), Value::Null));
        }
    }
    // sort: (behavior, is_batch, -drop_share with NaN last)
    rows.sort_by(|x, y| {
        let bx = (row_get(x, "behavior").as_str().unwrap_or(""), row_get(x, "is_batch") == &json!(true));
        let by = (row_get(y, "behavior").as_str().unwrap_or(""), row_get(y, "is_batch") == &json!(true));
        let k = |r: &Row| {
            let d = num(row_get(r, "drop_share"));
            if d.is_nan() { 9.0 } else { -d }
        };
        bx.cmp(&by).then_with(|| k(x).total_cmp(&k(y)))
    });
    let criteria = if batch_size > 1 {
        format!(
            "a drop is reported when the response is below its matched controls' (paired t-test over seeds, q < {} across the screen, and {:.0}% of seeds the same way); batches of {batch_size} are opened when they cut the response by >= {:.0}% (q < {})",
            crate::cards::py_float(Q_MAX),
            MIN_SIGN_SHARE * 100.0,
            BATCH_DROP * 100.0,
            crate::cards::py_float(BATCH_Q)
        )
    } else {
        format!(
            "a drop is reported when the response is below its matched controls' (paired t-test over seeds, q < {} across the screen, and {:.0}% of seeds the same way)",
            crate::cards::py_float(Q_MAX),
            MIN_SIGN_SHARE * 100.0
        )
    };
    json!({"kind": "knockout", "tag": TAG, "seeds": seeds, "behaviors": behaviors, "rows": rows.iter().map(to_object).collect::<Vec<_>>(),
           "summary": summary, "batch_size": batch_size, "criteria": criteria})
}

// --- the resumable store and the run ----------------------------------------------------------------------------------------------
pub struct Store {
    pub folder: PathBuf,
    meta_path: PathBuf,
    pub records: Vec<Value>,
    fh: std::fs::File,
}

impl Store {
    /// `records.jsonl` (appended as each job finishes) and `meta.json`; a different run in the same folder is refused.
    pub fn open(folder: &Path, signature: &Value, restart: bool, created: &str) -> Result<Self, String> {
        std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
        let (meta_path, rec_path) = (folder.join("meta.json"), folder.join("records.jsonl"));
        if restart {
            let _ = std::fs::remove_file(&meta_path);
            let _ = std::fs::remove_file(&rec_path);
        }
        if meta_path.exists() {
            let old: Value = std::fs::read_to_string(&meta_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
            let old_sig = &old["signature"];
            if old_sig != signature {
                let mut keys: Vec<String> =
                    old_sig.as_object().into_iter().chain(signature.as_object()).flat_map(|o| o.keys().cloned()).collect();
                keys.sort();
                keys.dedup();
                let diff: Vec<String> = keys.into_iter().filter(|k| old_sig.get(k) != signature.get(k)).collect();
                return Err(format!(
                    "{} holds a different run (differs in {}); use another --out, or restart it",
                    folder.display(),
                    diff.join(", ")
                ));
            }
        } else {
            let meta = json!({"signature": signature, "created": created, "tag": TAG});
            std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default()).map_err(|e| e.to_string())?;
        }
        let mut records = Vec::new();
        if let Ok(t) = std::fs::read_to_string(&rec_path) {
            for line in t.lines() {
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    records.push(v);
                }
            }
        }
        let mut fh = std::fs::OpenOptions::new().create(true).append(true).open(&rec_path).map_err(|e| e.to_string())?;
        if std::fs::read(&rec_path).is_ok_and(|b| !b.is_empty() && !b.ends_with(b"\n")) {
            let _ = fh.write_all(b"\n"); // end the line an interruption cut short
        }
        Ok(Self { folder: folder.to_path_buf(), meta_path, records, fh })
    }

    pub fn add(&mut self, recs: Vec<Value>) {
        for r in &recs {
            let _ = writeln!(self.fh, "{}", serde_json::to_string(r).unwrap_or_default());
        }
        let _ = self.fh.flush();
        self.records.extend(recs);
    }

    pub fn update_meta(&self, kv: &[(&str, Value)]) {
        let mut m: Value =
            std::fs::read_to_string(&self.meta_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}));
        for (k, v) in kv {
            m[*k] = v.clone();
        }
        let _ = std::fs::write(&self.meta_path, serde_json::to_string_pretty(&m).unwrap_or_default());
    }
}

/// Run `jobs` on `workers` threads, adding each one's records to the store as it finishes. `progress(done, total, label)` returns false to stop.
#[allow(clippy::too_many_arguments)]
fn execute<J: Sync>(
    jobs: &[J],
    f: &(dyn Fn(&J) -> Vec<Value> + Sync),
    store: &mut Store,
    workers: usize,
    done: &mut usize,
    total: usize,
    label: &str,
    progress: &mut dyn FnMut(usize, usize, &str) -> bool,
) -> Result<(), String> {
    if jobs.is_empty() {
        return Ok(());
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<Vec<Value>>();
    let tx = Mutex::new(tx);
    let mut stopped = false;
    std::thread::scope(|sc| {
        for _ in 0..workers.max(1).min(jobs.len()) {
            let (next, stop, tx) = (&next, &stop, &tx);
            sc.spawn(move || {
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        return;
                    }
                    let recs = f(&jobs[i]);
                    if tx.lock().unwrap_or_else(std::sync::PoisonError::into_inner).send(recs).is_err() {
                        return;
                    }
                }
            });
        }
        let mut got = 0;
        while got < jobs.len() {
            match rx.recv() {
                Ok(recs) => {
                    store.add(recs);
                    got += 1;
                    *done += 1;
                    if !progress(*done, total, label) {
                        stop.store(true, Ordering::Relaxed);
                        stopped = true;
                        // drain what the running jobs still finish so nothing is lost
                        while let Ok(r) = rx.recv_timeout(std::time::Duration::from_millis(200)) {
                            store.add(r);
                        }
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    if stopped { Err("cancelled".into()) } else { Ok(()) }
}

#[derive(Debug, Clone)]
pub struct ActivityOptions {
    pub seeds: Vec<u64>,
    pub types: Option<Vec<String>>,
    pub min_neurons: usize,
    pub max_types: Option<usize>,
    pub superclasses: Option<Vec<String>>,
    pub controls: usize,
    pub workers: usize,
    pub steps: usize,
    pub chunk: usize,
    pub restart: bool,
}

impl Default for ActivityOptions {
    fn default() -> Self {
        Self {
            seeds: SCREEN_SEEDS.to_vec(),
            types: None,
            min_neurons: 1,
            max_types: None,
            superclasses: None,
            controls: 2,
            workers: 1,
            steps: STIM,
            chunk: 25,
            restart: false,
        }
    }
}

fn chunks<T: Clone>(v: &[T], n: usize) -> Vec<Vec<T>> {
    v.chunks(n.max(1)).map(<[T]>::to_vec).collect()
}

/// `run_activation`: the activation screen; the result is also written to `folder`.
pub fn run_activation(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    folder: &Path,
    o: &ActivityOptions,
    created: &str,
    mut progress: impl FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    if o.seeds.is_empty() || o.controls < 1 || o.steps < 50 {
        return Err("an activation screen needs at least one seed, one matched control and 50 steps".into());
    }
    let cat = catalog(labels, o.min_neurons, None, o.superclasses.as_deref(), o.types.as_deref(), o.max_types)?;
    let sig = json!({"kind": "activation", "version": VERSION, "seeds": o.seeds, "controls": o.controls, "amp": f64::from(AMP), "steps": o.steps,
                     "pre": PRE, "readouts": READOUTS.iter().map(|r| r.0).collect::<Vec<_>>(), "backend": "auto"});
    let mut store = Store::open(folder, &sig, o.restart, created)?;
    let done_pairs: std::collections::HashSet<(String, u64)> = store
        .records
        .iter()
        .filter(|r| r["k"] == "act")
        .map(|r| (r["t"].as_str().unwrap_or("").to_string(), r["s"].as_u64().unwrap_or(0)))
        .collect();
    let mut jobs: Vec<ActJob> = Vec::new();
    for &s in &o.seeds {
        let todo: Vec<String> = cat.iter().filter(|c| !done_pairs.contains(&(c.name.clone(), s))).map(|c| c.name.clone()).collect();
        for part in chunks(&todo, o.chunk) {
            jobs.push(ActJob { seed: s, types: part, controls: o.controls, steps: o.steps });
        }
    }
    let total = jobs.len();
    let label = format!("activation screen: {} types x {} seeds, {} matched controls each", cat.len(), o.seeds.len(), o.controls);
    let mut done = 0usize;
    if jobs.is_empty() {
        progress(1, 1, "nothing left to run");
    } else {
        let f = |j: &ActJob| activation_job(weights, labels, j);
        execute(&jobs, &f, &mut store, o.workers, &mut done, total, &label, &mut progress)?;
    }
    let dn_names = dn_type_names(labels);
    let (rows, mut res) = aggregate_activation(&store.records, &cat, &o.seeds, &dn_names);
    res["rows"] = Value::Array(rows.iter().map(to_object).collect());
    res["meta"] = json!({"signature": sig, "resumed_jobs": jobs.is_empty(), "catalog_types": cat.len()});
    res["engines"] = json!([engine()]);
    store.update_meta(&[("engines", json!([engine()])), ("types", json!(cat.len()))]);
    save_activation(&rows, &res, folder)?;
    Ok(res)
}

/// `_dn_names`: the descending-neuron type names in `Arena`'s order (sorted).
pub fn dn_type_names(labels: &Labels) -> Vec<String> {
    let mut v: Vec<String> =
        labels.types.iter().zip(&labels.superclass).filter(|(_, s)| s.as_str() == "descending_neuron").map(|(t, _)| t.clone()).collect();
    v.sort();
    v.dedup();
    v
}

#[derive(Debug, Clone)]
pub struct KnockoutOptions {
    pub behaviors: Vec<String>,
    pub seeds: Vec<u64>,
    pub top: usize,
    pub batch_size: usize,
    pub controls: usize,
    pub workers: usize,
    pub types: Option<Vec<String>>,
    pub restart: bool,
}

impl Default for KnockoutOptions {
    fn default() -> Self {
        Self {
            behaviors: VALIDATED_PATHWAYS.iter().map(|s| s.to_string()).collect(),
            seeds: SCREEN_SEEDS.to_vec(),
            top: 25,
            batch_size: 0,
            controls: 2,
            workers: 1,
            types: None,
            restart: false,
        }
    }
}

/// `run_knockout`: the knockout screen; the result is also written to `folder`.
pub fn run_knockout(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    folder: &Path,
    o: &KnockoutOptions,
    created: &str,
    mut progress: impl FnMut(usize, usize, &str) -> bool,
) -> Result<Value, String> {
    let bad: Vec<&String> = o
        .behaviors
        .iter()
        .filter(|b| !VALIDATED_PATHWAYS.contains(&b.as_str()) || !validation::tests().iter().any(|t| t["id"].as_str() == Some(b.as_str())))
        .collect();
    if !bad.is_empty() {
        return Err(format!(
            "not a validated pathway behavior: {} (the screen runs {}; a behavior the model does not reproduce, docs/validation.md, would only show what silencing does to noise)",
            bad.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
            VALIDATED_PATHWAYS.join(", ")
        ));
    }
    if o.seeds.is_empty() || o.controls < 1 || o.top < 1 {
        return Err("a knockout screen needs a seed, a matched control, at least one candidate and a non-negative batch size".into());
    }
    let mut tys: Option<Vec<String>> = o.types.clone();
    if let Some(t) = tys.as_mut() {
        t.sort();
    }
    let sig = json!({"kind": "knockout", "version": VERSION, "seeds": o.seeds, "controls": o.controls, "amp": f64::from(AMP), "pre": PRE, "stim": STIM,
                     "top": o.top, "behaviors": o.behaviors, "batch_size": o.batch_size, "types": tys, "backend": "auto"});
    let mut store = Store::open(folder, &sig, o.restart, created)?;
    let saved = folder.join("candidates.json");
    let mut cands: HashMap<String, Vec<Value>> = std::fs::read_to_string(&saved)
        .ok()
        .and_then(|t| serde_json::from_str::<HashMap<String, Vec<Value>>>(&t.replace("NaN", "null")).ok())
        .unwrap_or_default();
    for b in &o.behaviors {
        if !cands.contains_key(b) {
            let list = match &o.types {
                Some(types) if !types.is_empty() => {
                    types.iter().map(|x| json!({"type": x, "neurons": 0, "influence": Value::Null, "drives_the_behavior": false})).collect()
                }
                _ => crate::criticalpath::shortlist(weights, labels, b, o.top),
            };
            cands.insert(b.clone(), list);
        }
    }
    let _ = std::fs::write(&saved, serde_json::to_string_pretty(&cands).unwrap_or_default());
    let stage1: Vec<(String, Vec<Vec<String>>)> = o
        .behaviors
        .iter()
        .map(|b| {
            let names: Vec<String> = cands[b].iter().filter_map(|c| c["type"].as_str().map(String::from)).collect();
            let specs = if o.batch_size > 1 { chunks(&names, o.batch_size) } else { names.into_iter().map(|n| vec![n]).collect() };
            (b.clone(), specs)
        })
        .collect();
    let plan = |store: &Store, stage: &[(String, Vec<Vec<String>>)]| -> Vec<KoJob> {
        let done: std::collections::HashSet<(String, u64, Vec<String>)> = store
            .records
            .iter()
            .filter(|r| r["k"] == "ko")
            .map(|r| {
                (
                    r["b"].as_str().unwrap_or("").to_string(),
                    r["s"].as_u64().unwrap_or(0),
                    r["spec"].as_array().into_iter().flatten().map(|s| s.as_str().unwrap_or("").to_string()).collect(),
                )
            })
            .collect();
        let mut jobs = Vec::new();
        for (b, specs) in stage {
            for &s in &o.seeds {
                let todo: Vec<Vec<String>> = specs.iter().filter(|sp| !done.contains(&(b.clone(), s, (*sp).clone()))).cloned().collect();
                for part in chunks(&todo, 10) {
                    jobs.push(KoJob { seed: s, behavior: b.clone(), specs: part, controls: o.controls });
                }
            }
        }
        jobs
    };
    let jobs = plan(&store, &stage1);
    let total_est = o.behaviors.len() * o.seeds.len();
    let mut total = jobs.len() + total_est;
    let label = format!("knockout screen: {} behaviors x {} seeds", o.behaviors.len(), o.seeds.len());
    let mut done = 0usize;
    let f = |j: &KoJob| knockout_job(weights, labels, j);
    execute(&jobs, &f, &mut store, o.workers, &mut done, total, &label, &mut progress)?;
    if o.batch_size > 1 {
        let opened = open_batches(&store.records, &o.behaviors, &o.seeds);
        let stage2: Vec<(String, Vec<Vec<String>>)> =
            opened.into_iter().filter(|(_, m)| !m.is_empty()).map(|(b, m)| (b, m.into_iter().map(|t| vec![t]).collect())).collect();
        let jobs2 = plan(&store, &stage2);
        total += jobs2.len();
        execute(&jobs2, &f, &mut store, o.workers, &mut done, total, &label, &mut progress)?;
    }
    let mut res = aggregate_knockout(&store.records, &o.behaviors, &o.seeds, &cands, o.batch_size);
    res["meta"] = json!({"signature": sig});
    res["engines"] = json!([engine()]);
    store.update_meta(&[("engines", json!([engine()]))]);
    save_knockout(&res, folder)?;
    Ok(res)
}

// --- output -----------------------------------------------------------------------------------------------------------------------
fn csv_cell(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap_or(0.0);
            if x.is_nan() { String::new() } else { crate::sensitivity::fmt_gp(x, 5) }
        }
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn csv_quote(c: &str) -> String {
    if c.contains([',', '"', '\n', '\r']) { format!("\"{}\"", c.replace('"', "\"\"")) } else { c.to_string() }
}

/// `_write_rows`: a CSV of the rows (the first row's keys, in order; `_dn` is not a column).
fn write_rows(rows: &[Row], path: &Path) -> std::io::Result<PathBuf> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let cols: Vec<&String> = rows.first().map(|r| r.iter().map(|(k, _)| k).filter(|k| *k != "_dn").collect()).unwrap_or_default();
    let mut text = String::new();
    if !cols.is_empty() {
        text.push_str(&cols.iter().map(|c| csv_quote(c)).collect::<Vec<_>>().join(","));
        text.push_str("\r\n");
        for r in rows {
            text.push_str(&cols.iter().map(|c| csv_quote(&csv_cell(row_get(r, c)))).collect::<Vec<_>>().join(","));
            text.push_str("\r\n");
        }
    } else {
        text.push_str("\r\n");
    }
    let p = path.with_extension("csv");
    std::fs::write(&p, text)?;
    Ok(p)
}

fn finish_meta(folder: &Path, written: &[PathBuf]) {
    let mp = folder.join("meta.json");
    if let Some(mut m) = std::fs::read_to_string(&mp).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
        m["outputs"] = json!(written.iter().filter_map(|p| p.file_name().and_then(|f| f.to_str())).collect::<Vec<_>>());
        m["parquet_note"] = json!("parquet skipped: this build writes CSV and JSON only; the CSVs hold everything");
        m["tag"] = json!(TAG);
        let _ = std::fs::write(&mp, serde_json::to_string_pretty(&m).unwrap_or_default());
    }
}

pub fn save_activation(rows: &[Row], res: &Value, folder: &Path) -> Result<Vec<PathBuf>, String> {
    let io = |e: std::io::Error| e.to_string();
    let mut written = vec![write_rows(rows, &folder.join("activation_screen")).map_err(io)?];
    let resp: Vec<Row> = rows
        .iter()
        .flat_map(|r| {
            row_get(r, "_dn")
                .as_array()
                .into_iter()
                .flatten()
                .map(|d| -> Row {
                    vec![
                        ("type".into(), row_get(r, "type").clone()),
                        ("dn_type".into(), d[0].clone()),
                        ("delta_hz".into(), d[1].clone()),
                        ("z".into(), d[2].clone()),
                        ("n_seeds".into(), row_get(r, "n_seeds").clone()),
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect();
    written.push(write_rows(&resp, &folder.join("activation_screen_responses")).map_err(io)?);
    std::fs::write(folder.join("activation_screen.json"), serde_json::to_string_pretty(res).unwrap_or_default()).map_err(io)?;
    finish_meta(folder, &written);
    Ok(written)
}

pub fn save_knockout(res: &Value, folder: &Path) -> Result<Vec<PathBuf>, String> {
    let io = |e: std::io::Error| e.to_string();
    let rows: Vec<Row> = res["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| {
            // the CSV's column order is the row's own (Python's insertion order)
            const ORDER: [&str; 19] = [
                "behavior",
                "candidate",
                "is_batch",
                "members",
                "neurons",
                "n_seeds",
                "unperturbed_evoked_hz",
                "lesion_evoked_hz",
                "control_evoked_hz",
                "drop_hz",
                "drop_lo",
                "drop_hi",
                "drop_share",
                "p",
                "sign_share",
                "drives_the_behavior",
                "influence",
                "q",
                "rank",
            ];
            ORDER.iter().map(|k| (k.to_string(), r[*k].clone())).collect()
        })
        .collect();
    let written = vec![write_rows(&rows, &folder.join("knockout_screen")).map_err(io)?];
    std::fs::write(folder.join("knockout_screen.json"), serde_json::to_string_pretty(res).unwrap_or_default()).map_err(io)?;
    finish_meta(folder, &written);
    Ok(written)
}

/// `load`: a finished screen from its folder (what the Lab page shows).
pub fn load(folder: &Path) -> Option<Value> {
    for name in ["activation_screen.json", "knockout_screen.json"] {
        if let Ok(t) = std::fs::read_to_string(folder.join(name)) {
            return serde_json::from_str(&t.replace("NaN", "null")).ok();
        }
    }
    None
}

fn group_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `summary`: a few lines of text.
pub fn summary(res: &Value, top: usize) -> String {
    let seeds: Vec<i64> = res["seeds"].as_array().into_iter().flatten().filter_map(Value::as_i64).collect();
    let (s0, s1) = (seeds.first().copied().unwrap_or(0), seeds.last().copied().unwrap_or(0));
    if res["kind"].as_str() == Some("activation") {
        let mut lines = vec![format!(
            "Activation screen ({TAG}): {} cell types, seeds {s0}-{s1}, {} with a behavior call. {}",
            group_thousands(res["n_types"].as_u64().unwrap_or(0)),
            res["n_called"],
            res["criteria"].as_str().unwrap_or("")
        )];
        for r in res["rows"].as_array().into_iter().flatten().take(top) {
            if r["behavior_readout"].as_str().is_some_and(|s| !s.is_empty()) {
                lines.push(format!(
                    "  {:3}. {:<18}{:5} neurons  {:<44} {:>+7.1} Hz  q={}  DN responders {}",
                    r["rank"].as_u64().unwrap_or(0),
                    r["type"].as_str().unwrap_or(""),
                    r["neurons"].as_u64().unwrap_or(0),
                    r["behavior_call"].as_str().unwrap_or(""),
                    num(&r["behavior_effect_hz"]),
                    crate::sensitivity::fmt_gp(num(&r["behavior_q"]), 3),
                    r["dn_responders"].as_u64().unwrap_or(0)
                ));
            }
        }
        return lines.join("\n");
    }
    let behaviors: Vec<String> = res["behaviors"].as_array().into_iter().flatten().filter_map(|b| b.as_str().map(String::from)).collect();
    let mut lines = vec![format!(
        "Knockout screen ({TAG}): {} behaviors, seeds {s0}-{s1}. {}",
        behaviors.len(),
        res["criteria"].as_str().unwrap_or("")
    )];
    for b in &behaviors {
        let s = &res["summary"][b];
        lines.push(format!(
            "  {b}: unperturbed response {:.2} Hz, {} types tested, {} cut it by >= 10% (q < 0.05)",
            num(&s["unperturbed_evoked_hz"]),
            s["tested"].as_u64().unwrap_or(0),
            s["significant"].as_u64().unwrap_or(0)
        ));
        for r in res["rows"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|x| x["behavior"].as_str() == Some(b) && x["is_batch"] == json!(false))
            .take(3)
        {
            lines.push(format!(
                "      {}. {:<16} drop {:+.0}%  q={}",
                r["rank"].as_u64().unwrap_or(0),
                r["candidate"].as_str().unwrap_or(""),
                num(&r["drop_share"]) * 100.0,
                crate::sensitivity::fmt_gp(num(&r["q"]), 3)
            ));
        }
    }
    lines.join("\n")
}
