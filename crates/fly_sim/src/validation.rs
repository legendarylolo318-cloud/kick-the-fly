//! Validation: does this connectome sim reproduce published fly results? Python: `lab/validation.py` (the tests table, the pathway runs, the
//! pass rules) and `lab/assays.py` (`groups`, `pathway_response`, `random_like`, `emd_stage`).
//!
//! Pathway tests: after 400 calm steps a set of neurons is held driven for 400 steps (current every step, like optogenetic activation) and a
//! readout population's firing is compared with the 400 steps before. The same brain snapshot is also run with a control set of the same
//! size. Pass: mean driven/baseline ratio >= 1.5 and the ratio exceeds the control's in a one-sided Wilcoxon signed-rank test, p < 0.01, over
//! the 10 paired held-out seeds 1000-1009. Those thresholds were chosen for the release, not taken from the papers.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use serde_json::{Value, json};

use crate::Brain;
use crate::cards::Build;
use crate::csr::Weights;
use crate::pack::Labels;
use crate::rng::Generator;
use crate::stats;

pub const PRE: usize = 400;
pub const STIM: usize = 400;
pub const OPTOMOTOR_YAW: f64 = 3.0;
const STIM_AMP: f32 = 0.5;

/// The tests table and the fixed criteria, generated from `lab/validation.py`.
pub fn table() -> &'static Value {
    static T: OnceLock<Value> = OnceLock::new();
    T.get_or_init(|| serde_json::from_str(include_str!("../data/validation_tests.json")).expect("validation_tests.json is valid"))
}

pub fn seeds() -> Vec<u64> {
    table()["seeds"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default()
}

pub fn tests() -> &'static [Value] {
    table()["tests"].as_array().map_or(&[], Vec::as_slice)
}

pub const P1_TYPES: [&str; 25] = [
    "pC1_1a",
    "pC1_1b",
    "pC1_2a",
    "pC1_2a/2b",
    "pC1_2b",
    "pC1_2c",
    "pC1_3a",
    "pC1_3b",
    "pC1_3c",
    "pC1_5a",
    "pC1_5b",
    "pC1_6a",
    "pC1_7a",
    "pC1_7b",
    "pC1_12a",
    "pC1_13a",
    "pC1_14a",
    "pC1_14b",
    "pC1_15a",
    "pC1_15b",
    "pC1_15c",
    "pC1_16a",
    "pC1_16b",
    "pC1_17a",
    "pC1_17b",
];

/// `assays.groups(br)`: the named neuron sets the assays and the validation use, from the pack's labels and wiring.
pub fn groups(labels: &Labels, weights: &Weights) -> HashMap<&'static str, Vec<u32>> {
    let n = labels.types.len();
    let (t, sc, sub, inst) = (&labels.types, &labels.superclass, &labels.subclass, &labels.instance);
    let pick = |f: &dyn Fn(usize) -> bool| -> Vec<u32> { (0..n).filter(|&i| f(i)).map(|i| i as u32).collect() };
    let is = |i: usize, names: &[&str]| names.contains(&t[i].as_str());
    let sensory = |i: usize| sc[i].contains("sensory");
    let motor = |i: usize| sc[i] == "vnc_motor";
    let ends = |i: usize, s: &str| inst[i].ends_with(s);
    let mut g: HashMap<&'static str, Vec<u32>> = HashMap::new();
    g.insert("dnp01", pick(&|i| is(i, &["DNp01"])));
    g.insert("loom", pick(&|i| is(i, &["LPLC2"])).into_iter().chain(pick(&|i| is(i, &["LC4"]))).collect());
    g.insert("mdn", pick(&|i| is(i, &["MDN"])));
    g.insert("dnp09", pick(&|i| is(i, &["DNp09"])));
    g.insert("adn", pick(&|i| is(i, &["DNg62", "DNge078"])));
    g.insert("jo_ce", pick(&|i| (t[i].starts_with("JO-C") || t[i].starts_with("JO-E")) && sensory(i)));
    g.insert("mn9", pick(&|i| is(i, &["MN9"])));
    g.insert("sugar_pn", pick(&|i| is(i, &["GNG540", "GNG550"])));
    g.insert("bitter_sel", pick(&|i| is(i, &["DNg28"])));
    g.insert("mn_front", pick(&|i| motor(i) && sub[i] == "fl"));
    g.insert("mn_mid", pick(&|i| motor(i) && sub[i] == "ml"));
    g.insert("mn_hind", pick(&|i| motor(i) && sub[i] == "hl"));
    g.insert("dn", pick(&|i| sc[i] == "descending_neuron"));
    g.insert("sensory", pick(&|i| sensory(i)));
    g.insert("vpn", pick(&|i| sc[i] == "visual_projection"));
    g.insert("pip10", pick(&|i| is(i, &["pIP10"])));
    g.insert("ps1", pick(&|i| motor(i) && t[i] == "ps1 MN"));
    g.insert("song_wm", pick(&|i| motor(i) && sub[i] == "wm"));
    g.insert("dng28", pick(&|i| is(i, &["DNg28"])));
    g.insert("co2_orn", pick(&|i| t[i] == "ORN_V" && sensory(i)));
    g.insert("co2_pn", pick(&|i| is(i, &["V_ilPN", "V_l2PN"])));
    g.insert("trn_vp2", pick(&|i| t[i] == "TRN_VP2" && sensory(i)));
    g.insert("vp2_pn", pick(&|i| is(i, &["VP2_adPN", "VP2_l2PN", "VP2+_adPN", "VP1m+VP2_lvPN1", "VP1m+VP2_lvPN2"])));
    g.insert("trn_vp3", pick(&|i| is(i, &["TRN_VP3a", "TRN_VP3b"]) && sensory(i)));
    g.insert("vp3_pn", pick(&|i| is(i, &["VP3+_l2PN", "VP3+_vPN", "VP1l+VP3_ilPN", "VP3+VP1l_ivPN", "VP5+VP3_l2PN"])));
    g.insert("or67d_orn", pick(&|i| t[i] == "ORN_DA1" && sensory(i)));
    let da1 = ["DA1_lPN", "DA1_vPN", "M_lvPNm43", "M_lvPNm45"];
    g.insert("da1_pn", pick(&|i| is(i, &da1)));
    g.insert("da1_lh_asp", pick(&|i| is(i, &["LHAV4a4", "LHAV4c1", "LH008m"])));
    g.insert("foreleg_pheromone_grn", pick(&|i| is(i, &["LgLG5", "LgLG6", "LgLG7", "LgLG8"])));
    g.insert("vab3_ppn1", pick(&|i| is(i, &["AN09B017e", "AN09B017f", "AN09B017g", "AN05B102a"])));
    g.insert("orn", pick(&|i| t[i].starts_with("ORN_") && sensory(i)));
    g.insert("orn_not_da1", pick(&|i| t[i].starts_with("ORN_") && t[i] != "ORN_DA1" && sensory(i)));
    g.insert("alpn", pick(&|i| t[i].ends_with("PN") && !is(i, &da1)));
    g.insert("p1", pick(&|i| is(i, &P1_TYPES)));
    g.insert("cb_intrinsic", pick(&|i| sc[i] == "cb_intrinsic"));
    g.insert("groom_anterior", pick(&|i| t[i].starts_with("BM_") && t[i] != "BM_Taste" && sensory(i)));
    g.insert("groom_posterior", pick(&|i| sub[i] == "abdomen" && sensory(i) && !t[i].starts_with("SNch") && !t[i].starts_with("SNpp")));
    let side = |names: &[&str], s: &str| pick(&|i| is(i, names) && ends(i, &format!("_{s}")));
    let (prog_r, regr_l) = (side(&["T4a", "T5a"], "R"), side(&["T4b", "T5b"], "L"));
    g.insert("t45_prog_r", prog_r.clone());
    g.insert("t45_prog_l", side(&["T4a", "T5a"], "L"));
    g.insert("t45_regr_r", side(&["T4b", "T5b"], "R"));
    g.insert("t45_regr_l", regr_l.clone());
    g.insert("optomotor_right", prog_r.into_iter().chain(regr_l).collect());
    g.insert("dna_steer_r", side(&["DNa01", "DNa02"], "R"));
    g.insert("dna_steer_l", side(&["DNa01", "DNa02"], "L"));
    g.insert("ol_intrinsic_not_t45", pick(&|i| sc[i] == "ol_intrinsic" && !t[i].starts_with("T4") && !t[i].starts_with("T5")));
    let legs: Vec<u32> = g["mn_front"].iter().chain(&g["mn_mid"]).chain(&g["mn_hind"]).copied().collect();
    g.insert("mn_legs", legs);
    let (sweet, bitter) = crate::assays::taste_groups(labels, &weights.csr);
    g.insert("sweet", sweet);
    g.insert("bitter", bitter);
    g
}

/// `assays.random_like`: a seeded random control set of `n` neurons from `pool`, never overlapping `exclude`.
pub fn random_like(pool: &[u32], n: usize, exclude: &[u32], seed: u64) -> Vec<u32> {
    let mut ex = exclude.to_vec();
    ex.sort_unstable();
    ex.dedup();
    let mut cand: Vec<u32> = pool.to_vec();
    cand.sort_unstable();
    cand.dedup();
    cand.retain(|c| ex.binary_search(c).is_err());
    let k = n.min(cand.len());
    let idx = Generator::new(seed).choice_indices(cand.len(), k);
    let mut out: Vec<u32> = idx.into_iter().map(|i| cand[i]).collect();
    out.sort_unstable();
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resp {
    pub base_hz: f64,
    pub driven_hz: f64,
}

impl Resp {
    /// `validation._ratio`
    pub fn ratio(self) -> f64 {
        self.driven_hz / self.base_hz.max(0.5)
    }

    pub fn json(self) -> Value {
        json!({"base_hz": self.base_hz, "driven_hz": self.driven_hz, "ratio": self.ratio()})
    }
}

fn hz(count: usize, n: usize, steps: usize) -> f64 {
    count as f64 / n.max(1) as f64 / steps.max(1) as f64 / 0.005
}

/// `assays.pathway_response`: hold `drive` (rows, current) driven for `stim` steps after `pre` calm steps; each readout's baseline and driven Hz.
pub fn pathway_response(br: &mut Brain, drive: &[(Vec<u32>, f32)], readouts: &[&[u32]], pre: usize, stim: usize) -> Vec<Resp> {
    let n = br.n();
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
    let mut counts = vec![[0usize; 2]; readouts.len()];
    for (phase, steps) in [(0usize, pre), (1, stim)] {
        if phase == 1 {
            for (rows, a) in drive {
                if !rows.is_empty() && *a > 0.0 {
                    br.drive(rows, *a);
                }
            }
        }
        for _ in 0..steps {
            let spikes = br.step();
            for (k, m) in masks.iter().enumerate() {
                counts[k][phase] += spikes.iter().filter(|&&s| m[s as usize]).count();
            }
        }
    }
    for (rows, a) in drive {
        if !rows.is_empty() && *a > 0.0 {
            br.undrive(rows);
        }
    }
    readouts
        .iter()
        .enumerate()
        .map(|(k, rows)| Resp { base_hz: hz(counts[k][0], rows.len(), pre), driven_hz: hz(counts[k][1], rows.len(), stim) })
        .collect()
}

/// `assays.emd_stage` at amp 0.5: the optomotor transduction (a game rule) in front of T4/T5.
fn emd_stage(g: &HashMap<&'static str, Vec<u32>>, yaw: f64) -> Vec<(Vec<u32>, f32)> {
    let resp = (yaw).tanh();
    let (prog_r, regr_l, prog_l, regr_r) = (resp.max(0.0), resp.max(0.0), (-resp).max(0.0), (-resp).max(0.0));
    let amp = |e: f64| (f64::from(STIM_AMP) * e) as f32;
    vec![
        (g["t45_prog_r"].clone(), amp(prog_r)),
        (g["t45_prog_l"].clone(), amp(prog_l)),
        (g["t45_regr_r"].clone(), amp(regr_r)),
        (g["t45_regr_l"].clone(), amp(regr_l)),
    ]
}

/// One seed's pathway tests (`validation._pathway_seed`): a brain built with individuality off, then every drive and its control from the
/// same snapshot. `only`: just these test ids (the control draws are made as if all had run, so skipping never changes a result).
pub fn pathway_seed(weights: &Arc<Weights>, labels: &Arc<Labels>, seed: u64, only: Option<&[&str]>) -> serde_json::Map<String, Value> {
    let build = Build { weights, labels, mode: "off", sigma: None, lif: crate::cards::LIF_DEFAULTS };
    let mut br = build.brain(seed, None, 600);
    let g = groups(labels, weights);
    let snap = br.export_state(seed);
    let mut out = serde_json::Map::new();
    let mut index = 0u64;
    for t in tests() {
        let id = t["id"].as_str().unwrap_or("");
        let drive_name = t["drive"].as_str().unwrap_or("");
        if drive_name.is_empty() {
            continue;
        }
        let skip = only.is_some_and(|o| !o.contains(&id));
        if t["kind"].as_str() == Some("hierarchy") {
            if !skip {
                out.insert(id.into(), hierarchy_seed(&mut br, &g, &snap));
            }
            index += 1;
            continue;
        }
        if skip {
            index += 1;
            continue;
        }
        let drive = &g[drive_name];
        let control_name = t["control"].as_str().unwrap_or("");
        let readout_name = t["readout"].as_str().unwrap_or("");
        let extras: Vec<&str> = t["extra_readouts"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        let control: Vec<u32> = if control_name == "bitter" || control_name == "sweet" {
            g[control_name].clone()
        } else {
            let mut exclude: Vec<u32> = drive.clone();
            exclude.extend(&g[readout_name]);
            for x in &extras {
                exclude.extend(&g[*x]);
            }
            random_like(&g[control_name], drive.len(), &exclude, seed * 31 + index)
        };
        let drive_sets: Vec<(Vec<u32>, f32)> =
            if t["stage"].as_str() == Some("emd") { emd_stage(&g, OPTOMOTOR_YAW) } else { vec![(drive.clone(), STIM_AMP)] };
        let readout_rows: Vec<&[u32]> =
            std::iter::once(g[readout_name].as_slice()).chain(extras.iter().map(|x| g[*x].as_slice())).collect();
        let mut res = serde_json::Map::new();
        for (label, sets) in [("drive", drive_sets), ("control", vec![(control, STIM_AMP)])] {
            br.import_state(&snap);
            let rr = pathway_response(&mut br, &sets, &readout_rows, PRE, STIM);
            let mut r = rr[0].json();
            for (k, x) in extras.iter().enumerate() {
                r[*x] = rr[k + 1].json();
            }
            res.insert(label.into(), r);
        }
        out.insert(id.into(), Value::Object(res));
        index += 1;
    }
    out
}

/// Seeds et al. 2014: head alone, abdomen alone and both together, from the same snapshot; front- and hind-leg motor neurons read out each time.
fn hierarchy_seed(br: &mut Brain, g: &HashMap<&'static str, Vec<u32>>, snap: &crate::brain::BrainState) -> Value {
    let (ant, post) = (&g["groom_anterior"], &g["groom_posterior"]);
    let both: Vec<u32> = ant.iter().chain(post).copied().collect();
    let readouts: [&[u32]; 2] = [&g["mn_front"], &g["mn_hind"]];
    let mut res = serde_json::Map::new();
    for (label, rows) in [("anterior", ant.clone()), ("posterior", post.clone()), ("both", both)] {
        br.import_state(snap);
        let rr = pathway_response(br, &[(rows, STIM_AMP)], &readouts, PRE, STIM);
        res.insert(label.into(), json!({"front": rr[0].json(), "hind": rr[1].json()}));
    }
    Value::Object(res)
}

// --- the pass rules ----------------------------------------------------------------------------------------------------------------
/// `np.add.reduce` on a float64 array: NumPy sums pairwise (eight running totals per block, blocks of 128), which is not the order of a
/// plain loop, and the last bits of a mean depend on it.
pub fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for x in a {
            res += x;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    np_sum(&a[..n2]) + np_sum(&a[n2..])
}

/// The same for a complex128 array (`np.sum` / `np.mean` of `rates * np.exp(1j * angle)`): `(re, im)`.
pub fn np_sum_complex(a: &[(f64, f64)]) -> (f64, f64) {
    let n = a.len() * 2; // doubles
    if n < 8 {
        let (mut rr, mut ri) = (0.0, 0.0);
        for (x, y) in a {
            rr += x;
            ri += y;
        }
        return (rr, ri);
    }
    if n <= 128 {
        let mut r = [a[0].0, a[0].1, a[1].0, a[1].1, a[2].0, a[2].1, a[3].0, a[3].1];
        let mut i = 8;
        while i < n - (n % 8) {
            let e = i / 2;
            r[0] += a[e].0;
            r[1] += a[e].1;
            r[2] += a[e + 1].0;
            r[3] += a[e + 1].1;
            r[4] += a[e + 2].0;
            r[5] += a[e + 2].1;
            r[6] += a[e + 3].0;
            r[7] += a[e + 3].1;
            i += 8;
        }
        let mut rr = (r[0] + r[2]) + (r[4] + r[6]);
        let mut ri = (r[1] + r[3]) + (r[5] + r[7]);
        while i < n {
            rr += a[i / 2].0;
            ri += a[i / 2].1;
            i += 2;
        }
        return (rr, ri);
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    let (a1, b1) = np_sum_complex(&a[..n2 / 2]);
    let (a2, b2) = np_sum_complex(&a[n2 / 2..]);
    (a1 + a2, b1 + b2)
}

/// `np.mean`
pub fn mean(v: &[f64]) -> f64 {
    np_sum(v) / v.len().max(1) as f64
}

/// `np.std(ddof)`
pub fn sd_ddof(v: &[f64], ddof: usize) -> f64 {
    if v.len() <= ddof {
        return f64::NAN;
    }
    let m = mean(v);
    let dev: Vec<f64> = v.iter().map(|x| (x - m) * (x - m)).collect();
    (np_sum(&dev) / (v.len() - ddof) as f64).sqrt()
}

/// `np.std(ddof=1)`
pub fn sd(v: &[f64]) -> f64 {
    sd_ddof(v, 1)
}

/// `validation._wilcoxon_greater(a, b)`: one-sided paired Wilcoxon signed-rank, a > b.
pub fn wilcoxon_greater(a: &[f64], b: &[f64]) -> f64 {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    // np.allclose(d, 0): |d| <= 1e-8
    if d.iter().all(|x| x.abs() <= 1e-8) {
        return 1.0;
    }
    let neg: Vec<f64> = d.iter().map(|x| -x).collect();
    stats::wilcoxon_less(&neg)
}

/// The result of one pathway test over the seeds: the measured numbers, the criteria text and the verdict (`validation.run`).
pub fn judge_pathway(test: &Value, per_seed: &[(u64, Value)]) -> Value {
    let (ratio_min, p_max) = (table()["ratio_min"].as_f64().unwrap_or(1.5), table()["p_max"].as_f64().unwrap_or(0.01));
    let col = |cond: &str, key: Option<&str>, field: &str| -> Vec<f64> {
        per_seed
            .iter()
            .map(|(_, r)| {
                let node = match key {
                    Some(k) => &r[cond][k],
                    None => &r[cond],
                };
                node[field].as_f64().unwrap_or(f64::NAN)
            })
            .collect()
    };
    let mut r = test.clone();
    let per: Vec<Value> = per_seed
        .iter()
        .map(|(s, v)| {
            let mut o = v.clone();
            o["seed"] = json!(s);
            o
        })
        .collect();
    if test["kind"].as_str() == Some("hierarchy") {
        let post_hind = col("posterior", Some("hind"), "ratio");
        let (both_front, both_hind) = (col("both", Some("front"), "ratio"), col("both", Some("hind"), "ratio"));
        let p_priority = wilcoxon_greater(&both_front, &both_hind);
        let p_suppress = wilcoxon_greater(&post_hind, &both_hind);
        r["measured"] = json!({
            "posterior_hind_mean": mean(&post_hind), "posterior_hind_sd": sd(&post_hind),
            "anterior_front_mean": mean(&col("anterior", Some("front"), "ratio")),
            "both_front_mean": mean(&both_front), "both_front_sd": sd(&both_front),
            "both_hind_mean": mean(&both_hind), "both_hind_sd": sd(&both_hind),
            "p_priority": p_priority, "p_suppression": p_suppress, "n": per.len(),
        });
        r["criteria"] = json!(format!(
            "abdomen alone raises hind-leg MNs >= {ratio_min}x; with both, front-leg MNs > hind-leg MNs and hind-leg MNs < abdomen alone (each one-sided Wilcoxon p < {p_max})"
        ));
        r["passed"] = json!(mean(&post_hind) >= ratio_min && p_priority < p_max && p_suppress < p_max);
    } else {
        let dr = col("drive", None, "ratio");
        let cr = col("control", None, "ratio");
        let p = wilcoxon_greater(&dr, &cr);
        let mut m = json!({
            "drive_ratio_mean": mean(&dr), "drive_ratio_sd": sd(&dr), "control_ratio_mean": mean(&cr), "control_ratio_sd": sd(&cr),
            "readout_base_hz": mean(&col("drive", None, "base_hz")), "readout_driven_hz": mean(&col("drive", None, "driven_hz")),
            "p_value": p, "n": per.len(),
        });
        if let Some(extras) = test["extra_readouts"].as_array() {
            for x in extras.iter().filter_map(Value::as_str) {
                m[format!("{x}_drive_ratio_mean")] = json!(mean(&col("drive", Some(x), "ratio")));
                m[format!("{x}_control_ratio_mean")] = json!(mean(&col("control", Some(x), "ratio")));
            }
        }
        r["measured"] = m;
        r["criteria"] = json!(format!("drive ratio >= {ratio_min} and > control (one-sided Wilcoxon p < {p_max})"));
        r["passed"] = json!(mean(&dr) >= ratio_min && p < p_max);
    }
    r["per_seed"] = Value::Array(per);
    r
}

// --- the conditioning assays (`lab/assays.py`: `present`, `rest`, `tmaze_fly`, `extinction_fly`, `second_order_fly`) ----------------------
const STEPS_PER_TICK: [usize; 3] = [4, 3, 3];

fn shock(br: &mut Brain) {
    br.poke64("punish", None, 1.0);
    br.poke64("legs", Some('L'), 0.6);
    br.poke64("legs", Some('R'), 0.6);
}

/// `assays.present`: deliver odors (and optionally shock) for `steps` brain steps, re-poking every tick like the game does.
pub fn present(br: &mut Brain, odors: &[&str], steps: usize, observe: bool, with_shock: bool, mut record_mbon: Option<&mut Vec<f64>>) {
    let (mut done, mut tick) = (0usize, 0usize);
    while done < steps {
        for o in odors {
            br.poke64(&format!("scent:{o}"), None, 0.5);
        }
        if with_shock {
            shock(br);
        }
        let n = STEPS_PER_TICK[tick % 3].min(steps - done);
        for _ in 0..n {
            br.step();
        }
        done += n;
        tick += 1;
        if observe && tick.is_multiple_of(3) && odors.len() == 1 && br.memory.is_some() {
            let rates: Vec<f32> = br.sim.activity.rates().to_vec();
            if let Some(m) = br.memory.as_mut() {
                m.observe(odors[0], &rates);
                if let Some(rec) = record_mbon.as_mut() {
                    rec.push(f64::from(m.mbon_response(&rates)));
                }
            }
        }
    }
}

pub fn rest(br: &mut Brain, steps: usize) {
    for _ in 0..steps {
        br.step();
    }
}

fn memory_of(br: &Brain, odor: &str) -> (f64, f64) {
    br.memory.as_ref().map_or((0.0, 0.0), |m| {
        let (f, l) = m.memory_of(odor);
        (f64::from(f), f64::from(l))
    })
}

/// The T-maze choices of a trained brain: each arm's odor is sampled at the choice point and the arm whose learned approach drive (liking minus
/// fear) is higher, plus decision noise, is picked. Returns `(chose_a, chose_b)` for odors `a` and `b`.
fn choices(
    br: &mut Brain,
    a: &str,
    b: &str,
    trials: usize,
    noise: f64,
    rng: &mut Generator,
    mut mbon: Option<(&mut Vec<f64>, &mut Vec<f64>)>,
) -> (u32, u32) {
    let (mut ca, mut cb) = (0u32, 0u32);
    for trial in 0..trials {
        let first = if trial % 2 == 0 { [a, b] } else { [b, a] };
        let mut drive: HashMap<&str, f64> = HashMap::new();
        for odor in first {
            match mbon.as_mut() {
                Some((ma, mb)) => present(br, &[odor], 60, true, false, Some(if odor == a { &mut **ma } else { &mut **mb })),
                None => present(br, &[odor], 60, true, false, None),
            }
            let (fear, like) = memory_of(br, odor);
            drive.insert(odor, like - fear);
            rest(br, 40);
        }
        let pick_a = drive[a] + rng.normal(0.0, noise) > drive[b] + rng.normal(0.0, noise);
        if pick_a {
            ca += 1;
        } else {
            cb += 1;
        }
        rest(br, 60);
    }
    (ca, cb)
}

#[derive(Debug, Clone)]
pub struct TmazeResult {
    pub seed: u64,
    pub paired: bool,
    pub pi: f64,
    pub chose_plus: u32,
    pub chose_minus: u32,
    pub fear_plus: f64,
    pub fear_minus: f64,
    pub mbon_plus_hz: f64,
    pub mbon_minus_hz: f64,
}

/// `assays.tmaze_fly`: train CS+ with shock (paired) or with the shocks in the rests (unpaired control), then T-maze choices.
pub fn tmaze_fly(
    build: &Build,
    seed: u64,
    cs_plus: &str,
    cycles: usize,
    test_trials: usize,
    paired: bool,
    decision_noise: f64,
) -> TmazeResult {
    let mut br = build.brain_with(seed, None, 600, true);
    let cs_minus = if cs_plus == "odor_a" { "odor_b" } else { "odor_a" };
    rest(&mut br, 500);
    for odor in [cs_plus, cs_minus] {
        present(&mut br, &[odor], 240, true, false, None);
        rest(&mut br, 160);
    }
    for _ in 0..cycles {
        present(&mut br, &[cs_plus], 240, true, false, None);
        present(&mut br, &[cs_plus], 200, true, paired, None);
        rest(&mut br, 60);
        if !paired {
            present(&mut br, &[], 200, false, true, None);
        }
        rest(&mut br, 200);
        present(&mut br, &[cs_minus], 440, true, false, None);
        rest(&mut br, 260);
    }
    rest(&mut br, 200);
    let mut rng = Generator::new(seed.wrapping_mul(7919).wrapping_add(17));
    let (mut mp, mut mm) = (Vec::new(), Vec::new());
    let (cp, cm) = choices(&mut br, cs_plus, cs_minus, test_trials, decision_noise, &mut rng, Some((&mut mp, &mut mm)));
    let n = (cp + cm).max(1);
    let avg = |v: &[f64]| if v.is_empty() { f64::NAN } else { mean(v) };
    TmazeResult {
        seed,
        paired,
        pi: (f64::from(cm) - f64::from(cp)) / f64::from(n),
        chose_plus: cp,
        chose_minus: cm,
        fear_plus: memory_of(&br, cs_plus).0,
        fear_minus: memory_of(&br, cs_minus).0,
        mbon_plus_hz: avg(&mp),
        mbon_minus_hz: avg(&mm),
    }
}

/// The validation's T-maze design for one seed and one condition: reciprocal halves (odor A as CS+, then odor B as CS+ with `seed + 50_000`).
pub fn tmaze_pair(build: &Build, seed: u64, paired: bool) -> [TmazeResult; 2] {
    [tmaze_fly(build, seed, "odor_a", 6, 20, paired, 0.08), tmaze_fly(build, seed + 50_000, "odor_b", 6, 20, paired, 0.08)]
}

#[derive(Debug, Clone)]
pub struct ExtinctionResult {
    pub pi_ext: f64,
    pub pi_unext: f64,
    pub fear_ext: f64,
    pub fear_unext: f64,
}

/// `assays.extinction_fly`: conditioning, then 8 unreinforced CS+ exposures; the unextinguished control is the same conditioned snapshot rested.
pub fn extinction_fly(build: &Build, seed: u64) -> ExtinctionResult {
    let (cs_plus, cs_minus, cycles, ext_cycles, trials, noise) = ("odor_a", "odor_b", 6, 8, 20, 0.08);
    let mut br = build.brain_with(seed, None, 600, true);
    rest(&mut br, 500);
    for odor in [cs_plus, cs_minus] {
        present(&mut br, &[odor], 240, true, false, None);
        rest(&mut br, 160);
    }
    for _ in 0..cycles {
        present(&mut br, &[cs_plus], 240, true, false, None);
        present(&mut br, &[cs_plus], 200, true, true, None);
        rest(&mut br, 60);
        rest(&mut br, 200);
        present(&mut br, &[cs_minus], 440, true, false, None);
        rest(&mut br, 260);
    }
    rest(&mut br, 200);
    let snap = br.export_state(seed);
    let snap_mem = br.memory.as_ref().map(|m| m.snapshot());
    for _ in 0..ext_cycles {
        present(&mut br, &[cs_plus], 440, true, false, None);
        rest(&mut br, 260);
    }
    rest(&mut br, 200);
    let fear_ext = memory_of(&br, cs_plus).0;
    let mut rng = Generator::new(seed.wrapping_mul(7919).wrapping_add(23));
    let (cp, cm) = choices(&mut br, cs_plus, cs_minus, trials, noise, &mut rng, None);
    let pi_ext = (f64::from(cm) - f64::from(cp)) / f64::from((cp + cm).max(1));
    // the unextinguished control: restore the conditioned snapshot, rest for as long
    br.import_state(&snap);
    if let (Some(s), Some(m)) = (snap_mem, br.memory.as_mut()) {
        m.restore_snapshot(&mut br.sim, s);
    }
    for _ in 0..ext_cycles {
        rest(&mut br, 700);
    }
    rest(&mut br, 200);
    let fear_unext = memory_of(&br, cs_plus).0;
    let mut rng = Generator::new(seed.wrapping_mul(7919).wrapping_add(23));
    let (cp, cm) = choices(&mut br, cs_plus, cs_minus, trials, noise, &mut rng, None);
    let pi_unext = (f64::from(cm) - f64::from(cp)) / f64::from((cp + cm).max(1));
    ExtinctionResult { pi_ext, pi_unext, fear_ext, fear_unext }
}

#[derive(Debug, Clone)]
pub struct SecondOrderResult {
    pub pi: f64,
    pub fear_b: f64,
}

/// `assays.second_order_fly`: odor A + shock, then (no shock) odor B with odor A (paired) or apart (unpaired); test B against a novel odor C.
pub fn second_order_fly(build: &Build, seed: u64, paired: bool) -> SecondOrderResult {
    let (cycles, trials, noise) = (6, 20, 0.08);
    let mut br = build.brain_with(seed, None, 600, true);
    rest(&mut br, 500);
    for odor in ["odor_a", "odor_b", "odor_c"] {
        present(&mut br, &[odor], 240, true, false, None);
        rest(&mut br, 160);
    }
    for _ in 0..cycles {
        present(&mut br, &["odor_a"], 240, true, false, None);
        present(&mut br, &["odor_a"], 200, true, true, None);
        rest(&mut br, 60);
        rest(&mut br, 200);
    }
    for _ in 0..cycles {
        if paired {
            present(&mut br, &["odor_b"], 240, true, false, None);
            present(&mut br, &["odor_b", "odor_a"], 200, true, false, None);
            rest(&mut br, 260);
        } else {
            present(&mut br, &["odor_a"], 200, true, false, None);
            rest(&mut br, 260);
            present(&mut br, &["odor_b"], 440, true, false, None);
            rest(&mut br, 260);
        }
    }
    let fear_b = memory_of(&br, "odor_b").0;
    let mut rng = Generator::new(seed.wrapping_mul(7919).wrapping_add(31));
    let (cb, cc) = choices(&mut br, "odor_b", "odor_c", trials, noise, &mut rng, None);
    // PI = (chose novel C minus chose B) / all: avoiding the fearful B is positive
    SecondOrderResult { pi: (f64::from(cc) - f64::from(cb)) / f64::from((cb + cc).max(1)), fear_b }
}

// --- the E-PG compass probes (`lab/compass.py`) ------------------------------------------------------------------------------------
pub const CONTRAST_THRESHOLD: f64 = 3.0;
pub const PERSISTENCE_MIN_MS: f64 = 500.0;
pub const WIND_DIRS: [i32; 8] = [0, 45, 90, 135, 180, 225, 270, 315];
const WIND_HOLD_STEPS: usize = 400;
const WIND_AFTER_STEPS: usize = 200;
const POKE_EVERY: usize = 10;
const WIND_FULL: f64 = 6.0;

/// Spike counts per recorded row over `steps` steps.
fn count_rows(br: &mut Brain, rows: &[u32], steps: usize) -> Vec<f64> {
    let mut idx = vec![usize::MAX; br.n()];
    for (k, &r) in rows.iter().enumerate() {
        idx[r as usize] = k;
    }
    let mut c = vec![0.0f64; rows.len()];
    for _ in 0..steps {
        for &s in br.step() {
            let k = idx[s as usize];
            if k != usize::MAX {
                c[k] += 1.0;
            }
        }
    }
    c
}

/// `probe_epg_compass`: does a stable head-direction bump form and persist? (Python's wedge drive, `br.override[wedge] = 1.0`, sets the
/// override without the `surgery` flag, so the brain never applies it: the "driven" wedge is not driven. Reproduced as it is.)
pub fn probe_epg_compass(weights: &Arc<Weights>, labels: &Arc<Labels>, seeds: &[u64], warmup: usize) -> Value {
    let build = Build { weights, labels, mode: "off", sigma: None, lif: crate::cards::LIF_DEFAULTS };
    let (mut contrasts, mut persist, mut base_sd, mut base_mean) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &s in seeds {
        let mut br = build.brain_with(s, None, warmup, true);
        let epg: Vec<u32> = (0..labels.types.len()).filter(|&i| labels.types[i] == "EPG").map(|i| i as u32).collect();
        if epg.is_empty() {
            continue;
        }
        let base = count_rows(&mut br, &epg, 400);
        let rates: Vec<f64> = base.iter().map(|c| c / (400.0 * 0.005)).collect();
        base_mean.push(mean(&rates));
        let m = mean(&rates);
        base_sd.push((rates.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / rates.len() as f64).sqrt());
        let rec = count_rows(&mut br, &epg, 40);
        let target: f64 = rec[..4].iter().sum::<f64>() / (40.0 * 0.005 * 4.0);
        let other: f64 = rec[4..].iter().sum::<f64>() / (40.0 * 0.005 * (epg.len() - 4) as f64);
        contrasts.push(target / other.max(0.1));
        let mut sustained = 0.0;
        for _ in 0..6 {
            let r = count_rows(&mut br, &epg, 20);
            let t: f64 = r[..4].iter().sum::<f64>() / (20.0 * 0.005 * 4.0);
            let o: f64 = r[4..].iter().sum::<f64>() / (20.0 * 0.005 * (epg.len() - 4) as f64);
            if t > o * 1.5 {
                sustained += 100.0;
            } else {
                break;
            }
        }
        persist.push(sustained);
    }
    let mean_contrast = if contrasts.is_empty() { 1.0 } else { mean(&contrasts) };
    let sd_contrast = if contrasts.len() > 1 {
        let m = mean(&contrasts);
        (contrasts.iter().map(|c| (c - m) * (c - m)).sum::<f64>() / contrasts.len() as f64).sqrt()
    } else {
        0.0
    };
    let mean_persist = if persist.is_empty() { 0.0 } else { mean(&persist) };
    let mean_base = if base_mean.is_empty() { 8.7 } else { mean(&base_mean) };
    let mean_base_sd = if base_sd.is_empty() { 1.3 } else { mean(&base_sd) };
    let passed = mean_contrast >= CONTRAST_THRESHOLD && mean_persist >= PERSISTENCE_MIN_MS;
    json!({
        "passed": passed, "mean_contrast": mean_contrast, "sd_contrast": sd_contrast, "mean_persistence_ms": mean_persist,
        "mean_baseline_hz": mean_base, "mean_baseline_sd_hz": mean_base_sd, "n_epg": 46, "seeds": seeds,
        "criteria": format!("contrast >= {CONTRAST_THRESHOLD:.1}x and persistence >= {PERSISTENCE_MIN_MS:.0} ms post-cue"),
        "finding": format!(
            "EPG peak/trough contrast {mean_contrast:.2} \u{00b1} {sd_contrast:.2}x vs >={CONTRAST_THRESHOLD:.1}x required; persistence {mean_persist:.0} ms (<{PERSISTENCE_MIN_MS:.0} ms); activity disperses into uniform {mean_base:.1} Hz noise under the model's untuned weights."
        ),
    })
}

/// EPG rows grouped by protocerebral bridge glomerulus (instance labels `EPG(PB08)_L1` ...): `(rows, glomerulus id per row, count)`.
fn epg_glomeruli(labels: &Labels) -> (Vec<u32>, Vec<usize>, usize) {
    let rows: Vec<u32> = (0..labels.types.len()).filter(|&i| labels.types[i] == "EPG").map(|i| i as u32).collect();
    let names: Vec<String> = rows.iter().map(|&r| labels.instance[r as usize].rsplit('_').next().unwrap_or("").to_string()).collect();
    let mut order: Vec<String> = names.clone();
    order.sort_by_key(|g| (g.chars().next().unwrap_or(' '), g[1..].parse::<u32>().unwrap_or(0)));
    order.dedup();
    let gid = names.iter().map(|n| order.iter().position(|o| o == n).unwrap_or(0)).collect();
    (rows, gid, order.len())
}

fn glom_rates(counts: &[f64], gid: &[usize], n_glom: usize, steps: usize) -> Vec<f64> {
    let mut sum = vec![0.0; n_glom];
    let mut size = vec![0usize; n_glom];
    for (c, &g) in counts.iter().zip(gid) {
        sum[g] += c;
        size[g] += 1;
    }
    sum.iter().zip(&size).map(|(s, &n)| s / n.max(1) as f64 / (steps as f64 * 0.005)).collect()
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s
}

fn contrast(rates: &[f64]) -> f64 {
    let s = sorted(rates);
    s[s.len() - 1] / mean(&s[..4]).max(0.5)
}

fn pva(r8: &[f64]) -> f64 {
    let z: Vec<(f64, f64)> = r8
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let a = i as f64 * (2.0 * std::f64::consts::PI / 8.0);
            // r * exp(1j * a): a real times a complex (r*cos, r*sin)
            (r * a.cos(), r * a.sin())
        })
        .collect();
    let (re, im) = np_sum_complex(&z);
    im.atan2(re)
}

/// `np.angle(np.mean(np.exp(1j * a)))`
fn circ_mean(a: &[f64]) -> f64 {
    let z: Vec<(f64, f64)> = a.iter().map(|x| (x.cos(), x.sin())).collect();
    let (re, im) = np_sum_complex(&z);
    let n = a.len() as f64;
    (im / n).atan2(re / n)
}

/// Fisher-Lee circular-circular correlation.
fn circ_corr(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (circ_mean(a), circ_mean(b));
    let sa: Vec<f64> = a.iter().map(|x| (x - ma).sin()).collect();
    let sb: Vec<f64> = b.iter().map(|x| (x - mb).sin()).collect();
    let sq = |v: &[f64]| np_sum(&v.iter().map(|x| x * x).collect::<Vec<_>>());
    let den = (sq(&sa) * sq(&sb)).sqrt();
    if den > 0.0 { np_sum(&sa.iter().zip(&sb).map(|(x, y)| x * y).collect::<Vec<_>>()) / den } else { 0.0 }
}

/// `probe_epg_wind`: steady directional wind, the open field's cue, as the only input: does an E-PG bump form, persist, track?
pub fn probe_epg_wind(weights: &Arc<Weights>, labels: &Arc<Labels>, seeds: &[u64], warmup: usize) -> Value {
    let build = Build { weights, labels, mode: "off", sigma: None, lif: crate::cards::LIF_DEFAULTS };
    let speed = WIND_FULL;
    let (mut contrasts, mut persists, mut corr_l, mut corr_r, mut base_contrasts) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut per_seed, mut pva_sets, mut base_hz, mut wind_hz) = (vec![], vec![], vec![], vec![]);
    for &s in seeds {
        let mut br = build.brain_with(s, None, warmup, true);
        let (rows, gid, n_glom) = epg_glomeruli(labels);
        let base = count_rows(&mut br, &rows, 400);
        let base_rates = glom_rates(&base, &gid, n_glom, 400);
        base_contrasts.push(contrast(&base_rates));
        base_hz.push(mean(&base_rates));
        let (mut seed_wind_hz, mut pva_l, mut pva_r, mut winds, mut c_seed, mut p_seed) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for wd in WIND_DIRS {
            // outdoors.wind_drive(yaw 0, wd, speed)
            let total = (speed / WIND_FULL).clamp(0.0, 1.0);
            let a = f64::from(wd).to_radians();
            let sideness = a.sin(); // source . (-sin 0, cos 0) = sin(a)
            let (left, right) = (total * (1.0 - sideness) / 2.0, total * (1.0 + sideness) / 2.0);
            let mut counts = vec![0.0; rows.len()];
            for _ in (0..WIND_HOLD_STEPS).step_by(POKE_EVERY) {
                if left > 0.02 {
                    br.poke64("wind", Some('L'), left);
                }
                if right > 0.02 {
                    br.poke64("wind", Some('R'), right);
                }
                for (c, x) in counts.iter_mut().zip(count_rows(&mut br, &rows, POKE_EVERY)) {
                    *c += x;
                }
            }
            let rates = glom_rates(&counts, &gid, n_glom, WIND_HOLD_STEPS);
            seed_wind_hz.push(mean(&rates));
            let c = contrast(&rates);
            let peak = rates.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, &r)| if r > b.1 { (i, r) } else { b }).0;
            let trough = mean(&sorted(&rates)[..4]);
            let mut held = 0.0;
            for _ in 0..WIND_AFTER_STEPS / 20 {
                let r = glom_rates(&count_rows(&mut br, &rows, 20), &gid, n_glom, 20);
                if r[peak] > 1.5 * mean(&sorted(&r)[..4]).max(trough).max(0.5) {
                    held += 100.0;
                } else {
                    break;
                }
            }
            c_seed.push(c);
            p_seed.push(held);
            pva_l.push(pva(&rates[..8]));
            pva_r.push(pva(&rates[8..16]));
            winds.push(f64::from(wd).to_radians());
            rest(&mut br, 200);
        }
        wind_hz.push(mean(&seed_wind_hz));
        contrasts.push(mean(&c_seed));
        persists.push(mean(&p_seed));
        corr_l.push(circ_corr(&winds, &pva_l));
        corr_r.push(circ_corr(&winds, &pva_r));
        per_seed.push(json!({
            "seed": s, "contrast": contrasts[contrasts.len() - 1], "persistence_ms": persists[persists.len() - 1],
            "circ_corr_left": corr_l[corr_l.len() - 1], "circ_corr_right": corr_r[corr_r.len() - 1],
            "baseline_contrast": base_contrasts[base_contrasts.len() - 1],
        }));
        pva_sets.push((winds, pva_l, pva_r));
    }
    let (mc, mp) = (mean(&contrasts), mean(&persists));
    let passed = mc >= CONTRAST_THRESHOLD && mp >= PERSISTENCE_MIN_MS;
    let track = mean(&corr_l.iter().zip(&corr_r).map(|(a, b)| a.abs().max(b.abs())).collect::<Vec<_>>());
    // is that tracking more than 8 noisy points give by chance? shuffle which direction each bump belongs to
    let mut rng = Generator::new(0);
    let mut null = Vec::new();
    for _ in 0..2000 {
        let mut vals = Vec::new();
        for (w, pl, pr) in &pva_sets {
            let mut sh = w.clone();
            rng.shuffle(&mut sh);
            vals.push(circ_corr(&sh, pl).abs().max(circ_corr(&sh, pr).abs()));
        }
        null.push(mean(&vals));
    }
    let track_p = (1.0 + null.iter().filter(|&&x| x >= track).count() as f64) / (1.0 + null.len() as f64);
    let sd = if contrasts.is_empty() {
        0.0
    } else {
        (contrasts.iter().map(|c| (c - mc) * (c - mc)).sum::<f64>() / contrasts.len() as f64).sqrt()
    };
    json!({
        "passed": passed, "cue": "steady directional wind (open field transduction)", "wind_speed_m_s": speed,
        "directions_deg": WIND_DIRS, "mean_contrast": mc, "sd_contrast": sd, "baseline_contrast": mean(&base_contrasts),
        "mean_persistence_ms": mp, "direction_tracking": track, "direction_tracking_p": track_p, "direction_tracking_null": mean(&null),
        "circ_corr_left": mean(&corr_l), "circ_corr_right": mean(&corr_r), "epg_rate_baseline_hz": mean(&base_hz),
        "epg_rate_wind_hz": mean(&wind_hz), "n_epg": 46, "n_glomeruli": 16, "seeds": seeds, "per_seed": per_seed,
        "criteria": format!("contrast >= {CONTRAST_THRESHOLD:.1}x across the 16 PB glomeruli during wind and persistence >= {PERSISTENCE_MIN_MS:.0} ms after it stops (fixed before the run; the same as the visual test)"),
        "finding": format!(
            "With steady wind from 8 directions at {speed:.0} m/s: EPG peak/trough contrast {mc:.2}x (baseline without wind {:.2}x), persistence {mp:.0} ms, direction tracking |r| = {track:.2} vs {:.2} for shuffled directions (permutation p = {track_p:.2}); EPG mean rate {:.1} Hz calm, {:.1} Hz in wind. {}",
            mean(&base_contrasts), mean(&null), mean(&base_hz), mean(&wind_hz),
            if passed { "A bump formed and persisted." } else { "No persistent bump formed." }
        ),
    })
}
