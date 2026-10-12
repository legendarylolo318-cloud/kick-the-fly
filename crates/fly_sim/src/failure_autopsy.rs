//! Failure autopsy (`lab/failure_autopsy.py`): for each validation test that fails, where does the driven signal go and where does it fade.
//! Read-only: it reads the connectome, the simulator's own weights and a run of the brain, and changes nothing.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::cards::{Build, LIF_DEFAULTS};
use crate::csr::{Csr, Weights};
use crate::diffmode::np_sum_f32;
use crate::pack::Labels;
use crate::validation::{self, PRE, STIM, np_sum};

pub const TAG: &str = "MODEL PREDICTION";
pub const RATIO_MIN: f64 = 1.5;
pub const AUTOPSY_SEEDS: [u64; 3] = [1000, 1001, 1002];
const AMP: f32 = 0.5;
const HOPS: usize = 3;
const TOP_ROUTES: usize = 6;
const VERSION: i64 = 1;

/// One input -> target pathway of a test.
pub struct Pathway {
    pub label: String,
    pub inp: Vec<u32>,
    pub tgt: Vec<u32>,
    pub inp_label: String,
    pub tgt_label: String,
}

fn prefix_rows(labels: &Labels, prefixes: &[&str]) -> Vec<u32> {
    (0..labels.types.len() as u32).filter(|&i| prefixes.iter().any(|p| labels.types[i as usize].starts_with(p))).collect()
}

fn type_rows(labels: &Labels, ty: &str) -> Vec<u32> {
    (0..labels.types.len() as u32).filter(|&i| labels.types[i as usize] == ty).collect()
}

fn special_pathways(test_id: &str, labels: &Labels, g: &HashMap<&'static str, Vec<u32>>) -> Result<Vec<Pathway>, String> {
    let grp = |k: &str| g.get(k).cloned().unwrap_or_default();
    let p = |label: &str, inp: Vec<u32>, tgt: Vec<u32>, il: &str, tl: String| Pathway {
        label: label.into(),
        inp,
        tgt,
        inp_label: il.into(),
        tgt_label: tl,
    };
    match test_id {
        "grooming_hierarchy" => Ok(vec![
            p(
                "head bristles to the front-leg motor neurons (the head grooming program)",
                grp("groom_anterior"),
                grp("mn_front"),
                "head bristle neurons (BM_*)",
                "front-leg motor neurons".into(),
            ),
            p(
                "abdominal sensory neurons to the hind-leg motor neurons (the abdomen program)",
                grp("groom_posterior"),
                grp("mn_hind"),
                "abdominal mechanosensory neurons",
                "hind-leg motor neurons".into(),
            ),
            p(
                "head bristles to the hind-leg motor neurons (the suppression the hierarchy needs)",
                grp("groom_anterior"),
                grp("mn_hind"),
                "head bristle neurons (BM_*)",
                "hind-leg motor neurons".into(),
            ),
        ]),
        "epg_compass" => {
            let epg = type_rows(labels, "EPG");
            let split = 4.min(epg.len());
            let rest = epg.len().saturating_sub(4);
            Ok(vec![p(
                "the driven wedge to the rest of the ring (what a bump needs to spread and persist)",
                epg[..split].to_vec(),
                epg[split..].to_vec(),
                "4 EPG neurons (the wedge the test drives)",
                format!("the other {rest} EPG neurons"),
            )])
        }
        "epg_compass_wind" => Ok(vec![p(
            "wind neurons to the E-PG compass neurons",
            grp("jo_ce"),
            type_rows(labels, "EPG"),
            "JO-C/E mechanosensory neurons (the wind's route in the open field)",
            "EPG neurons".into(),
        )]),
        "mb_extinction" => Ok(vec![p(
            "mushroom body output neurons to the reward dopamine neurons (the route an opposing, extinguishing memory needs)",
            prefix_rows(labels, &["MBON"]),
            prefix_rows(labels, &["PAM"]),
            "MBON types",
            "PAM reward dopamine neurons".into(),
        )]),
        "mb_second_order" => Ok(vec![p(
            "mushroom body output neurons to the punishment dopamine neurons (odor A's memory must drive PPL1 for odor B to gain fear)",
            prefix_rows(labels, &["MBON"]),
            prefix_rows(labels, &["PPL1"]),
            "MBON types",
            "PPL1 punishment dopamine neurons".into(),
        )]),
        other => Err(format!("{other} has no autopsy defined")),
    }
}

fn pathways(test_id: &str, labels: &Labels, g: &HashMap<&'static str, Vec<u32>>) -> Result<Vec<Pathway>, String> {
    let t = validation::tests().iter().find(|t| t["id"] == test_id).ok_or_else(|| format!("unknown test {test_id:?}"))?;
    let s = |k: &str| t[k].as_str().unwrap_or("");
    if !s("drive").is_empty()
        && !s("readout").is_empty()
        && t.get("kind").is_none()
        && !["epg_compass", "epg_compass_wind"].contains(&test_id)
    {
        return Ok(vec![Pathway {
            label: s("claim").into(),
            inp: g.get(s("drive")).cloned().unwrap_or_default(),
            tgt: g.get(s("readout")).cloned().unwrap_or_default(),
            inp_label: s("drive_label").into(),
            tgt_label: s("readout_label").into(),
        }]);
    }
    special_pathways(test_id, labels, g)
}

/// The adult tests that failed in a validation result.
pub fn failing_tests(results: &Value) -> Vec<Value> {
    let ids: Vec<&str> = validation::tests().iter().filter_map(|t| t["id"].as_str()).collect();
    let adult = results.get("brain").and_then(Value::as_str).unwrap_or("adult") == "adult";
    results["tests"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| r["passed"] != true && adult && r["id"].as_str().is_some_and(|i| ids.contains(&i)))
        .cloned()
        .collect()
}

// --- the connectome side ------------------------------------------------------------------------------------------------------------------
/// `W.data * (1 / inv[post])` in float32: the signed synapse counts behind the normalised weights.
fn raw_counts(w: &Csr, inv: &[f32]) -> Csr {
    let mut data = w.data.clone();
    for i in 0..w.n_rows {
        let v = inv[i];
        let scale: f32 = if v > 0.0 { 1.0f32 / v.max(1e-12f32) } else { 0.0 };
        for d in &mut data[w.indptr[i] as usize..w.indptr[i + 1] as usize] {
            *d *= scale;
        }
    }
    Csr { n_rows: w.n_rows, n_cols: w.n_cols, indptr: w.indptr.clone(), indices: w.indices.clone(), data }
}

fn split(w: &Csr) -> (Csr, Csr) {
    let build = |keep: &dyn Fn(f32) -> Option<f32>| {
        let (mut indptr, mut indices, mut data) = (vec![0u32], Vec::new(), Vec::new());
        for i in 0..w.n_rows {
            let (cols, vals) = w.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                if let Some(x) = keep(v).filter(|x| *x != 0.0) {
                    indices.push(c);
                    data.push(x);
                }
            }
            indptr.push(indices.len() as u32);
        }
        Csr { n_rows: w.n_rows, n_cols: w.n_cols, indptr, indices, data }
    };
    (build(&|v| (v > 0.0).then_some(v)), build(&|v| (v < 0.0).then_some(-v)))
}

fn matvec(a: &Csr, x: &[f64]) -> Vec<f64> {
    (0..a.n_rows)
        .map(|i| {
            let (cols, vals) = a.row(i);
            let mut s = 0.0f64;
            for (&c, &v) in cols.iter().zip(vals) {
                s += f64::from(v) * x[c as usize];
            }
            s
        })
        .collect()
}

/// `A.T @ x`: columns accumulate over rows ascending.
#[allow(clippy::needless_range_loop)]
fn matvec_t(a: &Csr, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; a.n_cols];
    for j in 0..a.n_rows {
        let (cols, vals) = a.row(j);
        for (&c, &v) in cols.iter().zip(vals) {
            y[c as usize] += f64::from(v) * x[j];
        }
    }
    y
}

fn mean_at(v: &[f64], idx: &[u32]) -> f64 {
    if idx.is_empty() {
        return 0.0;
    }
    let vals: Vec<f64> = idx.iter().map(|&i| v[i as usize]).collect();
    np_sum(&vals) / vals.len() as f64
}

fn layer_budget(w: &Csr, inp: &[u32], tgt: &[u32]) -> Vec<Value> {
    let (pm, nm) = split(w);
    let n = w.n_rows;
    let mut p = vec![0.0f64; n];
    for &i in inp {
        p[i as usize] = 1.0;
    }
    let mut q = vec![0.0f64; n];
    let mut out = Vec::new();
    for h in 1..=HOPS {
        let (pp, nq, pq, np_) = (matvec(&pm, &p), matvec(&nm, &q), matvec(&pm, &q), matvec(&nm, &p));
        p = pp.iter().zip(&nq).map(|(a, b)| a + b).collect();
        q = pq.iter().zip(&np_).map(|(a, b)| a + b).collect();
        let (e, i) = (mean_at(&p, tgt), mean_at(&q, tgt));
        out.push(json!({
            "hop": h, "excitatory": e, "inhibitory": i, "net": e - i, "total": e + i,
            "inhibitory_share": if e + i > 0.0 { json!(i / (e + i)) } else { Value::Null },
        }));
    }
    out
}

/// `P[tgt][:, inp].sum()` in float32 as SciPy computes it (row sums by matvec, then a pairwise sum).
fn sub_sum_f32(m: &Csr, tgt: &[u32], inp_mask: &[bool]) -> f32 {
    let rows: Vec<f32> = tgt
        .iter()
        .map(|&r| {
            let (cols, vals) = m.row(r as usize);
            let mut s = 0.0f32;
            for (&c, &v) in cols.iter().zip(vals) {
                if inp_mask[c as usize] {
                    s += v;
                }
            }
            s
        })
        .collect();
    np_sum_f32(&rows)
}

fn by_type(values: &[f64], types: &[String]) -> Vec<(String, f64)> {
    let mut sums: std::collections::BTreeMap<&str, f64> = Default::default();
    for (v, t) in values.iter().zip(types) {
        *sums.entry(t.as_str()).or_insert(0.0) += v;
    }
    sums.into_iter().filter(|(_, s)| *s > 0.0).map(|(k, s)| (k.to_string(), s)).collect()
}

/// A candidate route: `(share, excitatory, inhibitory, intermediate types)`.
type Cand = (f64, f64, f64, Vec<String>);

fn routes(w: &Csr, counts: &Csr, types: &[String], inp: &[u32], tgt: &[u32]) -> Vec<Value> {
    let k = TOP_ROUTES;
    let n = w.n_rows;
    let (pm, nm) = split(w);
    let mut ind_i = vec![0.0f64; n];
    for &i in inp {
        ind_i[i as usize] = 1.0;
    }
    let mut ind_t = vec![0.0f64; n];
    for &i in tgt {
        ind_t[i as usize] = 1.0 / tgt.len().max(1) as f64;
    }
    let (vp, vn) = (matvec(&pm, &ind_i), matvec(&nm, &ind_i));
    let (bp, bn) = (matvec_t(&pm, &ind_t), matvec_t(&nm, &ind_t));
    let mut cand: Vec<Cand> = Vec::new();
    let mut inp_mask = vec![false; n];
    for &i in inp {
        inp_mask[i as usize] = true;
    }
    let tl = tgt.len().max(1) as f32;
    let direct = (sub_sum_f32(&pm, tgt, &inp_mask) / tl, sub_sum_f32(&nm, tgt, &inp_mask) / tl);
    if direct.0 + direct.1 > 0.0 {
        cand.push((f64::from(direct.0 + direct.1), f64::from(direct.0), f64::from(direct.1), Vec::new()));
    }
    // one intermediate type
    let ex: Vec<f64> = (0..n).map(|x| vp[x] * bp[x] + vn[x] * bn[x]).collect();
    let inh: Vec<f64> = (0..n).map(|x| vp[x] * bn[x] + vn[x] * bp[x]).collect();
    let (e_by, i_by) = (by_type(&ex, types), by_type(&inh, types));
    let mut names: Vec<String> = e_by.iter().chain(&i_by).map(|(k, _)| k.clone()).collect();
    names.sort();
    names.dedup();
    let get = |v: &[(String, f64)], k: &str| v.iter().find(|(n, _)| n == k).map_or(0.0, |(_, x)| *x);
    let mut merged: Vec<(String, f64, f64)> =
        names.into_iter().map(|nm_| (get(&e_by, &nm_), get(&i_by, &nm_), nm_)).map(|(e, i, n)| (n, e, i)).collect();
    merged.sort_by(|a, b| (b.1 + b.2).total_cmp(&(a.1 + a.2)));
    merged.truncate(TOP_ROUTES * 4);
    for (name, e, i) in merged {
        cand.push((e + i, e, i, vec![name]));
    }
    // two intermediate types
    let xs: Vec<usize> = (0..n).filter(|&i| vp[i] + vn[i] > 0.0).collect();
    let ys: Vec<usize> = (0..n).filter(|&i| bp[i] + bn[i] > 0.0).collect();
    if !xs.is_empty() && !ys.is_empty() {
        let mut in_xs = vec![false; n];
        for &x in &xs {
            in_xs[x] = true;
        }
        let mut order: Vec<(String, String)> = Vec::new();
        let mut pair: HashMap<(String, String), (f64, f64)> = HashMap::new();
        for &y in &ys {
            let (cols, vals) = w.row(y);
            for (&c, &wv) in cols.iter().zip(vals) {
                let x = c as usize;
                if !in_xs[x] {
                    continue;
                }
                let aw = f64::from(wv.abs());
                let posw = wv > 0.0;
                let (a, b) = (vp[x], vn[x]);
                let e_same = if posw { a * bp[y] + b * bn[y] } else { a * bn[y] + b * bp[y] };
                let i_same = if posw { a * bn[y] + b * bp[y] } else { a * bp[y] + b * bn[y] };
                let (e, i) = (aw * e_same, aw * i_same);
                if e + i > 0.0 {
                    let key = (types[x].clone(), types[y].clone());
                    let slot = pair.entry(key.clone()).or_insert_with(|| {
                        order.push(key);
                        (0.0, 0.0)
                    });
                    slot.0 += e;
                    slot.1 += i;
                }
            }
        }
        let mut items: Vec<((String, String), (f64, f64))> = order.into_iter().map(|k| (k.clone(), pair[&k])).collect();
        items.sort_by(|a, b| (b.1.0 + b.1.1).total_cmp(&(a.1.0 + a.1.1)));
        items.truncate(k * 4);
        for ((t1, t2), (e, i)) in items {
            cand.push((e + i, e, i, vec![t1, t2]));
        }
    }
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut per: HashMap<usize, usize> = HashMap::new();
    let mut picked: Vec<Cand> = Vec::new();
    for c in cand {
        let h = c.3.len() + 1;
        let cnt = per.entry(h).or_insert(0);
        if *cnt < 2.max(k / 3) {
            *cnt += 1;
            picked.push(c);
        }
    }
    picked.sort_by(|a, b| b.0.total_cmp(&a.0));
    picked.truncate(k + 3);
    let mut out = Vec::new();
    let mut seen: Vec<Vec<String>> = Vec::new();
    for (share, e, i, chain) in picked {
        if seen.contains(&chain) {
            continue;
        }
        seen.push(chain.clone());
        let mut stages: Vec<Vec<u32>> = vec![inp.to_vec()];
        stages.extend(chain.iter().map(|c| type_rows_in(types, c)));
        stages.push(tgt.to_vec());
        let mut edges = Vec::new();
        for pair in stages.windows(2) {
            let (pre, post) = (&pair[0], &pair[1]);
            let mut pre_mask = vec![false; n];
            for &p in pre {
                pre_mask[p as usize] = true;
            }
            let mut vals: Vec<f32> = Vec::new();
            for &r in post {
                let (cols, vs) = counts.row(r as usize);
                for (&c, &v) in cols.iter().zip(vs) {
                    if pre_mask[c as usize] {
                        vals.push(v);
                    }
                }
            }
            let abs: Vec<f32> = vals.iter().map(|v| v.abs()).collect();
            let pos: Vec<f32> = vals.iter().copied().filter(|&v| v > 0.0).collect();
            let neg: Vec<f32> = vals.iter().copied().filter(|&v| v < 0.0).collect();
            edges.push(json!({
                "synapses": f64::from(np_sum_f32(&abs)).round_ties_even() as i64,
                "excitatory": f64::from(np_sum_f32(&pos)).round_ties_even() as i64,
                "inhibitory": f64::from(-np_sum_f32(&neg)).round_ties_even() as i64,
            }));
        }
        out.push(json!({
            "hops": chain.len() + 1, "types": chain, "share": share, "excitatory": e, "inhibitory": i,
            "net_sign": if e >= i { 1 } else { -1 }, "inhibitory_share": if e + i > 0.0 { json!(i / (e + i)) } else { Value::Null }, "edges": edges,
        }));
        if out.len() >= k + 3 {
            break;
        }
    }
    out
}

fn type_rows_in(types: &[String], t: &str) -> Vec<u32> {
    (0..types.len() as u32).filter(|&i| types[i as usize] == t).collect()
}

// --- the run side --------------------------------------------------------------------------------------------------------------------------
/// Firing of the input, each route's types and the target, before and while the input is driven, averaged over seeds.
fn stage_rates(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    pw: &Pathway,
    route_types: &[Vec<String>],
    seeds: &[u64],
) -> serde_json::Map<String, Value> {
    let mut names: Vec<String> = vec!["__input__".into(), "__target__".into()];
    let mut rows: Vec<Vec<u32>> = vec![pw.inp.clone(), pw.tgt.clone()];
    for chain in route_types {
        for t in chain {
            if !names.contains(t) {
                names.push(t.clone());
                rows.push(type_rows_in(&labels.types, t));
            }
        }
    }
    let mut acc: Vec<Vec<(f64, f64)>> = vec![Vec::new(); names.len()];
    for &s in seeds {
        let b = Build { weights, labels, mode: "off", sigma: None, lif: LIF_DEFAULTS, wired: None };
        let mut br = b.brain_with(s, None, 600, false);
        let readouts: Vec<&[u32]> = rows.iter().map(Vec::as_slice).collect();
        let r = validation::pathway_response(&mut br, &[(pw.inp.clone(), AMP)], &readouts, PRE, STIM);
        for (k, x) in r.iter().enumerate() {
            acc[k].push((x.base_hz, x.driven_hz));
        }
    }
    let mut out = serde_json::Map::new();
    for (k, name) in names.iter().enumerate() {
        let b = np_sum(&acc[k].iter().map(|v| v.0).collect::<Vec<_>>()) / acc[k].len() as f64;
        let d = np_sum(&acc[k].iter().map(|v| v.1).collect::<Vec<_>>()) / acc[k].len() as f64;
        out.insert(name.clone(), json!({"base_hz": b, "driven_hz": d, "ratio": d / b.max(0.5), "neurons": rows[k].len()}));
    }
    out
}

fn verdict(route: Option<&Value>, rates: &serde_json::Map<String, Value>) -> Value {
    let Some(route) = route else {
        return json!({"stage": null, "text": "No route of up to three synapses connects the input to the target: nothing can carry the signal."});
    };
    let types: Vec<String> = route["types"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(String::from)).collect();
    let mut names = vec!["input".to_string()];
    names.extend(types.iter().cloned());
    names.push("target".into());
    let mut keys = vec!["__input__".to_string()];
    keys.extend(types.iter().cloned());
    keys.push("__target__".into());
    let seq: Vec<(String, f64)> =
        names.iter().zip(&keys).map(|(n, k)| (n.clone(), rates[k]["ratio"].as_f64().unwrap_or(f64::NAN))).collect();
    let mut lost: Option<(String, String)> = None;
    for w in seq.windows(2) {
        if w[0].1 >= RATIO_MIN && w[1].1 < RATIO_MIN {
            lost = Some((w[0].0.clone(), w[1].0.clone()));
            break;
        }
    }
    let ratio_of = |n: &str| seq.iter().rev().find(|s| s.0 == n).map_or(f64::NAN, |s| s.1);
    let (stage, text): (Value, String) = match lost {
        None if seq.first().is_some_and(|s| s.1 < RATIO_MIN) => (
            json!("input"),
            format!("the input itself ran at x{:.2}, below {RATIO_MIN}x: the drive never lifted the first stage", seq[0].1),
        ),
        None => {
            let end = seq.last().map_or(f64::NAN, |s| s.1);
            (
                Value::Null,
                if end >= RATIO_MIN {
                    format!("every stage along the strongest route met {RATIO_MIN}x (the target ended at x{end:.2})")
                } else {
                    format!("the ratios never crossed from above to below {RATIO_MIN}x along the strongest route (target x{end:.2})")
                },
            )
        }
        Some((a, b)) => (
            json!(format!("{a} -> {b}")),
            format!("the signal is lost between {a} (x{:.2}) and {b} (x{:.2}), below the test's {RATIO_MIN}x", ratio_of(&a), ratio_of(&b)),
        ),
    };
    json!({"stage": stage, "text": text, "ratios": seq.iter().map(|(n, r)| json!({"stage": n, "ratio": r})).collect::<Vec<_>>()})
}

fn digest(w: &Csr) -> String {
    let mut bytes: Vec<u8> = Vec::with_capacity(w.data.len() * 8 + w.indptr.len() * 4);
    for d in &w.data {
        bytes.extend_from_slice(&d.to_le_bytes());
    }
    for i in &w.indices {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    for p in &w.indptr {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    crate::util::sha256_hex(&bytes)[..16].to_string()
}

#[allow(clippy::too_many_arguments)]
/// `autopsy`: one failing test. `inv` is the pack's `1 / input synapses` per neuron.
pub fn autopsy(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    inv: &[f32],
    test_id: &str,
    seeds: &[u64],
    result: Option<&Value>,
    created: &str,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<Value, String> {
    let t = validation::tests().iter().find(|t| t["id"] == test_id).ok_or_else(|| format!("unknown test {test_id:?}"))?;
    let w = &weights.csr;
    let before = digest(w);
    let g = validation::groups(labels, weights);
    let pws = pathways(test_id, labels, &g)?;
    let counts = raw_counts(w, inv);
    let mut sections = Vec::new();
    for (idx, pw) in pws.iter().enumerate() {
        if pw.inp.is_empty() || pw.tgt.is_empty() {
            sections.push(json!({"label": pw.label, "error": "the input or the target is empty in this pack"}));
            continue;
        }
        let budget = layer_budget(w, &pw.inp, &pw.tgt);
        let mut rt = routes(w, &counts, &labels.types, &pw.inp, &pw.tgt);
        let chains: Vec<Vec<String>> =
            rt.iter().map(|r| r["types"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(String::from)).collect()).collect();
        let rates = stage_rates(weights, labels, pw, &chains, seeds);
        for (r, chain) in rt.iter_mut().zip(&chains) {
            let mut names = vec!["input".to_string()];
            names.extend(chain.iter().cloned());
            names.push("target".into());
            let mut keys = vec!["__input__".to_string()];
            keys.extend(chain.iter().cloned());
            keys.push("__target__".into());
            let mut stages = Vec::new();
            for (n, k) in names.iter().zip(&keys) {
                let mut s = rates[k].clone();
                s["stage"] = json!(n);
                stages.push(s);
            }
            r["stage_ratios"] = Value::Array(stages);
        }
        let mut inp_mask = vec![false; w.n_cols];
        for &i in &pw.inp {
            inp_mask[i as usize] = true;
        }
        let mut dvals: Vec<f32> = Vec::new();
        for &r in &pw.tgt {
            let (cols, vs) = counts.row(r as usize);
            for (&c, &v) in cols.iter().zip(vs) {
                if inp_mask[c as usize] {
                    dvals.push(v.abs());
                }
            }
        }
        let verdict_v = verdict(rt.first(), &rates);
        sections.push(json!({
            "label": pw.label, "input": pw.inp_label, "target": pw.tgt_label, "n_input": pw.inp.len(), "n_target": pw.tgt.len(),
            "direct_synapses": f64::from(np_sum_f32(&dvals)).round_ties_even() as i64, "budget": budget, "routes": rt,
            "input_rate": rates["__input__"], "target_rate": rates["__target__"], "verdict": verdict_v,
        }));
        progress(idx + 1, pws.len(), &pw.label);
    }
    let after = digest(w);
    if after != before {
        return Err("an autopsy changed the weights (it must not)".into());
    }
    let result_line = result.map(|r| json!({"passed": r["passed"] == true, "criteria": r["criteria"], "measured": r["measured"]}));
    Ok(json!({
        "version": VERSION, "tag": TAG,
        "test": {"id": test_id, "name": t["name"], "claim": t["claim"], "citation": t["citation"], "note": t.get("note").cloned().unwrap_or_else(|| json!(""))},
        "result": result_line, "seeds": seeds, "created": created, "read_only": true, "weights_digest": before, "sections": sections,
        "ratio_min": RATIO_MIN,
        "method": format!(
            "walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from {} seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm)",
            seeds.len()
        ),
    }))
}

// --- markdown ----------------------------------------------------------------------------------------------------------------------------
/// A share as a percentage with enough digits for the very small ones (`pct`).
pub fn pct(x: f64) -> String {
    if x.is_nan() {
        "n/a".into()
    } else if x == 0.0 {
        "0%".into()
    } else {
        format!("{}%", crate::sensitivity::fmt_gp(100.0 * x, 3))
    }
}

fn comma(x: i64) -> String {
    crate::netsci::group(x as f64)
}

pub fn to_markdown(a: &Value) -> String {
    let t = &a["test"];
    let s = |v: &Value| v.as_str().unwrap_or("").to_string();
    let mut l: Vec<String> = vec![
        format!("# Autopsy: {}", s(&t["name"])),
        String::new(),
        format!(
            "**{}** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.",
            s(&a["tag"])
        ),
        String::new(),
    ];
    if let Some(e) = a["error"].as_str() {
        l.push(format!("Not autopsied: {e}"));
        l.push(String::new());
        return l.join("\n");
    }
    l.push(format!("Claim: {}", s(&t["claim"])));
    l.push(format!("Source: {}", s(&t["citation"])));
    l.push(String::new());
    if let Some(r) = a["result"].as_object().filter(|_| !a["result"].is_null()) {
        l.push(format!("Result: **FAIL**. {}", r.get("criteria").and_then(Value::as_str).unwrap_or("")));
        l.push(String::new());
        if let Some(m) = r.get("measured").and_then(Value::as_object) {
            let parts: Vec<String> =
                m.iter().filter_map(|(k, v)| v.as_f64().map(|x| format!("{k} {}", crate::sensitivity::fmt_gp(x, 3)))).collect();
            if !parts.is_empty() {
                l.push(format!("Measured: {}.", parts.join(", ")));
                l.push(String::new());
            }
        }
    }
    if let Some(n) = t["note"].as_str().filter(|n| !n.is_empty()) {
        l.push(format!("The test's own note: {n}"));
        l.push(String::new());
    }
    for sec in a["sections"].as_array().into_iter().flatten() {
        l.push(format!("## {}", s(&sec["label"])));
        l.push(String::new());
        if let Some(e) = sec["error"].as_str() {
            l.push(e.to_string());
            l.push(String::new());
            continue;
        }
        l.push(format!(
            "Input: {} ({} neurons). Target: {} ({} neurons). Direct synapses from input to target: {}.",
            s(&sec["input"]),
            sec["n_input"],
            s(&sec["target"]),
            sec["n_target"],
            sec["direct_synapses"]
        ));
        l.push(String::new());
        l.push("### How much of the target's input the input can supply".into());
        l.push(String::new());
        l.push("| synapses | excitatory walks | inhibitory walks | net | inhibitory share |".into());
        l.push("|---|---|---|---|---|".into());
        let f = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
        for b in sec["budget"].as_array().into_iter().flatten() {
            let ish = f(&b["inhibitory_share"]);
            let ish = if ish.is_nan() { "n/a".to_string() } else { format!("{:.0}%", ish * 100.0) };
            l.push(format!(
                "| {} | {} | {} | {} | {ish} |",
                b["hop"],
                pct(f(&b["excitatory"])),
                pct(f(&b["inhibitory"])),
                pct(f(&b["net"]))
            ));
        }
        l.push(String::new());
        l.push("Shares of the average target neuron's input synapses, along walks of exactly that many synapses.".into());
        l.push(String::new());
        l.push("### The strongest routes, through cell types".into());
        l.push(String::new());
        for (i, rt) in sec["routes"].as_array().into_iter().flatten().enumerate() {
            let mut chain = vec!["input".to_string()];
            chain.extend(rt["types"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(String::from)));
            chain.push("target".into());
            l.push(format!(
                "{}. **{}**: {} of the target's input ({} excitatory, {} inhibitory; net {}).",
                i + 1,
                chain.join(" -> "),
                pct(f(&rt["share"])),
                pct(f(&rt["excitatory"])),
                pct(f(&rt["inhibitory"])),
                if rt["net_sign"].as_i64().unwrap_or(1) > 0 { "excites" } else { "inhibits" }
            ));
            let stages = rt["stage_ratios"].as_array().cloned().unwrap_or_default();
            for (st, e) in stages.iter().skip(1).zip(rt["edges"].as_array().into_iter().flatten()) {
                l.push(format!(
                    "   - into **{}**: {} synapses ({} excitatory, {} inhibitory); in the run {:.1} Hz at rest, {:.1} Hz driven (x{:.2}, {} neurons)",
                    s(&st["stage"]),
                    comma(e["synapses"].as_i64().unwrap_or(0)),
                    comma(e["excitatory"].as_i64().unwrap_or(0)),
                    comma(e["inhibitory"].as_i64().unwrap_or(0)),
                    f(&st["base_hz"]),
                    f(&st["driven_hz"]),
                    f(&st["ratio"]),
                    st["neurons"]
                ));
            }
        }
        l.push(String::new());
        l.push("### Where it fades".into());
        l.push(String::new());
        l.push(format!("{}.", s(&sec["verdict"]["text"])));
        l.push(String::new());
        if let Some(r) = sec["verdict"]["ratios"].as_array().filter(|r| !r.is_empty()) {
            let parts: Vec<String> = r.iter().map(|x| format!("{} x{:.2}", s(&x["stage"]), f(&x["ratio"]))).collect();
            l.push(format!("Driven / baseline along the strongest route: {}.", parts.join(", ")));
            l.push(String::new());
        }
    }
    l.push(format!("Method: {}.", s(&a["method"])));
    l.push(format!("Weights digest (unchanged by the autopsy): {}.", s(&a["weights_digest"])));
    l.push(String::new());
    l.join("\n")
}

pub fn index_markdown(autopsies: &[Value]) -> String {
    let mut lines = vec![
        format!("# Failure autopsy ({TAG}, read-only)"),
        String::new(),
        "One page per failing validation test: where the driven signal goes and where it fades. Nothing here changes the model or suggests a change.".into(),
        String::new(),
    ];
    for a in autopsies {
        let v: Vec<String> = a["sections"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|s| s["verdict"]["text"].as_str().or_else(|| s["error"].as_str()).unwrap_or("").to_string())
            .collect();
        let tail =
            a["error"].as_str().map_or_else(|| v.into_iter().filter(|x| !x.is_empty()).collect::<Vec<_>>().join(" / "), String::from);
        lines.push(format!("- [{}]({}.md): {tail}", a["test"]["name"].as_str().unwrap_or(""), a["test"]["id"].as_str().unwrap_or("")));
    }
    lines.join("\n") + "\n"
}

/// `run_all`: autopsy every failing adult test of a validation result into `folder` (`<id>.json`, `<id>.md`, `index.md`).
#[allow(clippy::too_many_arguments)]
pub fn run_all(
    weights: &Arc<Weights>,
    labels: &Arc<Labels>,
    inv: &[f32],
    results: &Value,
    folder: &Path,
    only: Option<&[String]>,
    created: &str,
    progress: &mut dyn FnMut(usize, usize, &str),
) -> Result<Vec<Value>, String> {
    std::fs::create_dir_all(folder).map_err(|e| e.to_string())?;
    let mut fails = failing_tests(results);
    if let Some(o) = only {
        fails.retain(|f| o.iter().any(|x| f["id"] == x.as_str()));
    }
    let mut out = Vec::new();
    let total = fails.len();
    for (i, f) in fails.iter().enumerate() {
        let id = f["id"].as_str().unwrap_or("");
        let a = match autopsy(weights, labels, inv, id, &AUTOPSY_SEEDS, Some(f), created, &mut |_, _, _| {}) {
            Ok(a) => a,
            Err(e) => {
                json!({"test": {"id": id, "name": f["name"].as_str().unwrap_or(id)}, "error": e, "tag": TAG, "sections": [], "read_only": true})
            }
        };
        let _ = std::fs::write(folder.join(format!("{id}.json")), serde_json::to_string_pretty(&a).unwrap_or_default());
        let _ = std::fs::write(folder.join(format!("{id}.md")), to_markdown(&a));
        out.push(a);
        progress(i + 1, total, id);
    }
    let _ = std::fs::write(folder.join("index.md"), index_markdown(&out));
    Ok(out)
}
