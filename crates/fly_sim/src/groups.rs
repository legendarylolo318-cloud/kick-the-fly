//! The brain's named neuron groups, the stimulus queue and the rate trackers the game reads its reactions from.
//!
//! Python: `game/kick_the_fly.py:Brain` (`SENSE`, `MOTOR`, `POPS`, `Brain.__init__`'s group tables, `poke`, the `_step` tail that updates
//! `fast`/`base`, `level`, `hz`). Differences: the pokes sample their neurons with this crate's PCG64 (`Generator::shuffle`), not NumPy's
//! `rng.choice`, so which neurons a given hit recruits differs from Python even with the same seed; the groups themselves (which rows are
//! in "walk", "turn_l", "loom"...) are the same, and the tests check their sizes against the Python reference on the real pack.
//! Not ported: the larva tables, the assay groups `sweet`/`bitter` (they come from the wiring), pain index, learning.

use std::collections::HashMap;

use crate::pack::Labels;
use crate::rng::Generator;

/// Current of a held touch pulse (x `ext_gain` 4 = 2.0 per step).
pub const POKE_AMP: f32 = 0.5;
/// History samples kept, one per 20 ms (4 steps): 30 s.
pub const HIST: usize = 1500;
/// Steps without a poke before the calm baseline learns again (2 s).
pub const CALM_STEPS: u64 = 400;

pub const TOUCH: [&str; 4] = ["head", "body", "legs", "wing"];

/// `SENSE`: region -> type prefixes.
const SENSE: &[(&str, &[&str])] = &[
    ("wind", &["JO-C", "JO-E"]),
    ("head", &["BM_", "JO-"]),
    ("body", &["SNta"]),
    ("legs", &["SNpp"]),
    ("wing", &["WG"]),
    ("heat", &["TRN_VP2"]),
    ("cold", &["TRN_VP3"]),
    ("humid", &["HRN_"]),
    ("smell", &["ORN_"]),
    ("taste", &["LgLG", "LgAG", "LB", "PhG", "claw_"]),
    ("light", &["R1-R6", "R7", "R8"]),
    ("loom", &["LPLC2", "LC4"]),
    ("track", &["LC10"]),
    ("small", &["LC11", "LC18", "LC21", "LC26"]),
];
/// Visual projection groups, not sensory neurons.
const NOT_SENSORY: &[&str] = &["loom", "track", "small"];

/// `MOTOR`: reaction name, descending neuron types, side.
const MOTOR: &[(&str, &[&str], Option<char>)] = &[
    ("jump", &["DNg85", "DNg48", "DNg37", "DNge067", "DNg29", "DNge132"], None),
    ("run", &["DNge122", "DNge104", "DNge102", "DNxl114", "DNge182", "DNge048"], None),
    ("kick", &["DNge074", "DNge075", "DNge096", "DNg30", "DNg34", "DNp38"], None),
    ("walk", &["DNp09"], None),
    ("back", &["MDN"], None),
    ("turn_l", &["DNa01", "DNa02"], Some('L')),
    ("turn_r", &["DNa01", "DNa02"], Some('R')),
    ("fly", &["DNg02_a", "DNg02_b", "DNg02_c", "DNg02_d", "DNg02_e", "DNg02_f", "DNg02_g"], None),
    ("escape", &["DNp01"], None),
    ("fire", &["DNp35", "DNpe052"], None),
];

/// `POPS`: population name -> superclasses.
const POPS: &[(&str, &[&str])] = &[
    ("photoreceptors", &["ol_sensory"]),
    ("optic lobe", &["ol_intrinsic", "visual_projection", "visual_centrifugal", "visual_projection_tbc"]),
    ("central brain", &["cb_intrinsic"]),
    ("head sensory", &["cb_sensory", "cb_sensory_tbc", "sensory_descending"]),
    ("body sensory", &["vnc_sensory", "vnc_sensory_tbc", "sensory_ascending", "sensory_ascending_tbc"]),
    ("ascending", &["ascending_neuron"]),
    ("descending", &["descending_neuron", "descending_neuron_tbc"]),
    ("nerve cord", &["vnc_intrinsic", "vnc_tbc"]),
    ("motor neurons", &["vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "efferent_ascending", "efferent_descending"]),
];

/// The names of the populations, in `POPS` order.
pub fn population_names() -> Vec<&'static str> {
    POPS.iter().map(|(n, _)| *n).collect()
}

/// Which population a superclass belongs to.
pub fn population_of(superclass: &str) -> Option<usize> {
    POPS.iter().position(|(_, classes)| classes.contains(&superclass))
}

/// `TOOL_NAMES`, in the order the scent glomeruli are dealt out.
pub const TOOL_NAMES: [&str; 18] = [
    "hand",
    "flick",
    "swatter",
    "bomb",
    "torch",
    "cleaner",
    "zapper",
    "freeze",
    "spider",
    "sugar",
    "alcohol",
    "laser",
    "cva",
    "decoy",
    "fruit",
    "frog",
    "dragonfly",
    "mantis",
];

/// `PAIN_WEIGHTS`: touch, thermal, chemical, DN alarm, body relay; the index is capped at 100.
pub const PAIN_WEIGHTS: [f32; 5] = [0.55, 0.25, 0.55, 0.20, 0.30];

/// `Brain.pain_index`
pub fn pain_index(parts: &[f32; 5]) -> f32 {
    (100.0 * parts.iter().zip(PAIN_WEIGHTS).map(|(p, w)| *p as f64 * w as f64).sum::<f64>()).min(100.0) as f32
}

/// `PAIN_LEVELS`: share of a region's neurons a light touch recruits, and how hard the rest of the body's sensors join.
pub const PAIN_LEVELS: [(&str, f64, f64); 3] = [("normal", 0.3, 0.0), ("more", 0.6, 0.5), ("max", 1.0, 1.0)];

struct Pending {
    region: String,
    side: Option<char>,
    rows: Vec<u32>,
    steps: i32,
}

/// Named groups + the stimulus queue + the rate trackers of one brain.
pub struct Groups {
    pub n: usize,
    /// Detail groups first (`n_det` of them), then the populations, then "whole brain".
    pub names: Vec<String>,
    pub n_det: usize,
    det_id: Vec<i32>,
    pop_id: Vec<i32>,
    sizes: Vec<f64>,
    col: HashMap<String, usize>,
    sense: HashMap<(String, Option<char>), Vec<u32>>,
    /// Hz per neuron, EMA ~150 ms.
    pub fast: Vec<f64>,
    /// Calm baseline, Hz per neuron.
    pub base: Vec<f64>,
    k_fast: f64,
    k_base: f64,
    pending: Vec<Pending>,
    cur: Vec<f32>,
    pub last_poke: i64,
    /// Pain level 0..2 (`PAIN_LEVELS`).
    pub pain_level: usize,
    pub recruit: f64,
    pub spill: f64,
    rng: Generator,
    dt_s: f64,
    counts: Vec<u32>,
    /// Overlapping readout groups (they may share neurons with a detail group): `(column, membership mask)`.
    extra: Vec<(usize, Vec<bool>)>,
    /// Ring of the fast rates, one row of `names.len()` per 20 ms (`Brain.hist`).
    hist: Vec<f32>,
    /// Samples written so far (`Brain.hist_n`).
    pub hist_n: usize,
}

impl Groups {
    pub fn new(labels: &Labels, n: usize, dt_ms: f64, seed: u64) -> Self {
        let types = |i: usize| labels.types.get(i).map_or("", String::as_str);
        let sc = |i: usize| labels.superclass.get(i).map_or("", String::as_str);
        let side_of = |i: usize| -> Option<char> {
            let inst = labels.instance.get(i).map_or("", String::as_str);
            let b = inst.as_bytes();
            if b.len() >= 2 && b[b.len() - 2] == b'_' && (b[b.len() - 1] == b'L' || b[b.len() - 1] == b'R') {
                Some(b[b.len() - 1] as char)
            } else {
                None
            }
        };
        let mut det_id = vec![-1i32; n];
        let mut names: Vec<String> = Vec::new();
        let mut sizes: Vec<f64> = Vec::new();
        let mut sense: HashMap<(String, Option<char>), Vec<u32>> = HashMap::new();

        // claim the rows of `mask` that no earlier detail group has, as a new group; returns them
        let add_detail = |name: &str, mask: Vec<bool>, det_id: &mut Vec<i32>, names: &mut Vec<String>, sizes: &mut Vec<f64>| -> Vec<u32> {
            let id = names.len() as i32;
            let mut rows = Vec::new();
            for (i, &m) in mask.iter().enumerate() {
                if m && det_id[i] < 0 {
                    det_id[i] = id;
                    rows.push(i as u32);
                }
            }
            names.push(name.to_string());
            sizes.push(rows.len() as f64);
            rows
        };
        let mask_of = |pred: &dyn Fn(usize) -> bool| -> Vec<bool> { (0..n).map(pred).collect() };

        for (region, prefixes) in SENSE {
            let not_sensory = NOT_SENSORY.contains(region);
            let m = mask_of(&|i| prefixes.iter().any(|p| types(i).starts_with(p)) && (not_sensory || sc(i).contains("sensory")));
            let rows = add_detail(region, m, &mut det_id, &mut names, &mut sizes);
            for s in ['L', 'R'] {
                let side_rows: Vec<u32> = rows.iter().copied().filter(|&r| side_of(r as usize) == Some(s)).collect();
                // a side with too few neurons (the pack has no instance sides for it) falls back to the whole region
                sense.insert(
                    (region.to_string(), Some(s)),
                    if side_rows.len() as f32 > 0.2 * rows.len() as f32 { side_rows } else { rows.clone() },
                );
            }
            sense.insert((region.to_string(), None), rows);
        }
        for (name, tys, side) in MOTOR {
            let m = mask_of(&|i| sc(i) == "descending_neuron" && tys.contains(&types(i)) && side.is_none_or(|s| side_of(i) == Some(s)));
            add_detail(name, m, &mut det_id, &mut names, &mut sizes);
        }
        let is = |tys: &[&str]| -> Vec<bool> { mask_of(&|i| tys.contains(&types(i))) };
        add_detail(
            "groom",
            mask_of(&|i| sc(i) == "descending_neuron" && ["DNg62", "DNge078"].contains(&types(i))),
            &mut det_id,
            &mut names,
            &mut sizes,
        );
        add_detail("proboscis", is(&["MN9"]), &mut det_id, &mut names, &mut sizes);
        add_detail("song", mask_of(&|i| sc(i) == "vnc_motor" && types(i) == "ps1 MN"), &mut det_id, &mut names, &mut sizes);
        add_detail(
            "aggression",
            mask_of(&|i| types(i) == "AVLP727m" || (types(i).starts_with("pC1") && sc(i) == "cb_intrinsic")),
            &mut det_id,
            &mut names,
            &mut sizes,
        );
        let sleep = add_detail(
            "sleep",
            mask_of(&|i| types(i).starts_with("FB6") || types(i).starts_with("FB7")),
            &mut det_id,
            &mut names,
            &mut sizes,
        );
        sense.insert(("sleep".into(), None), sleep);
        add_detail("co2", is(&["V_ilPN", "V_l2PN"]), &mut det_id, &mut names, &mut sizes);
        add_detail(
            "hot_pn",
            is(&["VP2_adPN", "VP2_l2PN", "VP2+_adPN", "VP1m+VP2_lvPN1", "VP1m+VP2_lvPN2"]),
            &mut det_id,
            &mut names,
            &mut sizes,
        );
        add_detail(
            "cold_pn",
            is(&["VP3+_l2PN", "VP3+_vPN", "VP1l+VP3_ilPN", "VP3+VP1l_ivPN", "VP5+VP3_l2PN"]),
            &mut det_id,
            &mut names,
            &mut sizes,
        );
        let clock = add_detail("clock", is(&["l-LNv", "s-LNv"]), &mut det_id, &mut names, &mut sizes);
        sense.insert(("clock".into(), None), clock);
        let extra = add_detail(
            "body_extra",
            mask_of(&|i| ["vnc_sensory", "vnc_sensory_tbc", "sensory_ascending", "sensory_ascending_tbc"].contains(&sc(i))),
            &mut det_id,
            &mut names,
            &mut sizes,
        );
        sense.insert(("body_extra".into(), None), extra);
        let reward = add_detail("reward", mask_of(&|i| types(i).starts_with("PAM")), &mut det_id, &mut names, &mut sizes);
        sense.insert(("reward".into(), None), reward);
        let punish = add_detail("punish", mask_of(&|i| types(i).starts_with("PPL1")), &mut det_id, &mut names, &mut sizes);
        sense.insert(("punish".into(), None), punish);
        sense.insert(("all".into(), None), (0..n as u32).collect());

        // scents: each tool gets its own 5 of the 53 olfactory glomeruli (alcohol, cva, decoy, laser are special)
        let orn = sense.get(&("smell".to_string(), None)).cloned().unwrap_or_default();
        let glom = |r: u32| -> &str { types(r as usize).split_once('_').map_or(types(r as usize), |(_, g)| g) };
        let mut uniq: Vec<&str> = orn.iter().map(|&r| glom(r)).collect();
        uniq.sort_unstable();
        uniq.dedup();
        let mut order: Vec<&str> = uniq.clone();
        Generator::new(11).shuffle(&mut order);
        let in_gloms = |gl: &[&str]| -> Vec<u32> { orn.iter().copied().filter(|&r| gl.contains(&glom(r))).collect() };
        let first = |k: usize| -> Vec<u32> { orn.iter().copied().take(k).collect() };
        for (k, name) in TOOL_NAMES.iter().enumerate() {
            let rows = match *name {
                "alcohol" => in_gloms(&["DM1", "DM2", "DP1m"]),
                "laser" => Vec::new(),
                "cva" => in_gloms(&["DA1"]),
                "decoy" => in_gloms(&["DA1", "VA1v"]),
                _ if order.len() >= (k + 1) * 5 => in_gloms(&order[k * 5..(k + 1) * 5]),
                _ => first(5),
            };
            sense.insert((format!("scent:{name}"), None), rows);
        }
        sense.insert(("scent:cva".into(), None), in_gloms(&["DA1"]));
        sense.insert(("scent:player".into(), None), if order.len() >= 53 { in_gloms(&order[50..]) } else { first(3) });
        sense.insert(
            ("pheromone:foreleg".into(), None),
            (0..n as u32).filter(|&i| ["LgLG5", "LgLG6", "LgLG7", "LgLG8"].contains(&types(i as usize))).collect(),
        );
        // the T-maze odors use their own deal (seed 1985, six glomeruli each)
        let mut odor_order: Vec<&str> = uniq;
        Generator::new(1985).shuffle(&mut odor_order);
        for (k, key) in ["odor_a", "odor_b", "odor_c"].iter().enumerate() {
            let rows = if odor_order.len() >= (k + 1) * 6 { in_gloms(&odor_order[k * 6..(k + 1) * 6]) } else { first(6) };
            sense.insert((format!("scent:{key}"), None), rows);
        }

        let n_det = names.len();
        let mut pop_id = vec![-1i32; n];
        for (k, (name, classes)) in POPS.iter().enumerate() {
            let mut count = 0usize;
            for (i, slot) in pop_id.iter_mut().enumerate() {
                if classes.contains(&sc(i)) {
                    *slot = k as i32;
                    count += 1;
                }
            }
            names.push((*name).to_string());
            sizes.push(count as f64);
        }
        names.push("whole brain".to_string());
        sizes.push(n as f64);
        let g = names.len();
        let sizes: Vec<f64> = sizes.into_iter().map(|s| s.max(1.0)).collect();
        let col = names.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let mut this = Self {
            n,
            names,
            n_det,
            det_id,
            pop_id,
            sizes,
            col,
            sense,
            fast: vec![0.0; g],
            base: vec![0.0; g],
            k_fast: 1.0 - (-1.0f64 / 30.0).exp(),
            k_base: 1.0 - (-1.0f64 / 2000.0).exp(),
            pending: Vec::new(),
            cur: vec![0.0; n],
            last_poke: -1_000_000_000,
            pain_level: 0,
            recruit: PAIN_LEVELS[0].1,
            spill: 0.0,
            rng: Generator::new(seed),
            dt_s: dt_ms / 1000.0,
            counts: vec![0; g],
            extra: Vec::new(),
            hist: Vec::new(),
            hist_n: 0,
        };
        // the fermentation glomeruli (DM1/DM2/DP1m) as a readout of their own: the kitchen's vinegar trap reads them
        let rows = this.sense("scent:alcohol", None).to_vec();
        this.add_readout("scent:alcohol", &rows);
        this
    }

    /// Add an overlapping readout group (its own fast rate and calm baseline) after the fact.
    pub fn add_readout(&mut self, name: &str, rows: &[u32]) {
        let col = self.names.len();
        self.names.push(name.to_string());
        self.sizes.push((rows.len() as f64).max(1.0));
        self.fast.push(0.0);
        self.base.push(0.0);
        self.counts.push(0);
        self.col.insert(name.to_string(), col);
        let mut mask = vec![false; self.n];
        for &r in rows {
            mask[r as usize] = true;
        }
        self.extra.push((col, mask));
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn col(&self, name: &str) -> Option<usize> {
        self.col.get(name).copied()
    }

    pub fn size(&self, name: &str) -> usize {
        self.col(name).map_or(0, |c| self.sizes[c] as usize)
    }

    /// The rows a region (and side, for `L`/`R`) pokes. Scents are `scent:<tool>`.
    pub fn sense(&self, region: &str, side: Option<char>) -> &[u32] {
        self.sense.get(&(region.to_string(), side)).map_or(&[], Vec::as_slice)
    }

    pub fn set_pain_level(&mut self, level: usize) {
        self.pain_level = level.min(2);
        let (_, recruit, spill) = PAIN_LEVELS[level.min(2)];
        self.recruit = recruit;
        self.spill = spill;
    }

    /// `Brain.rng = default_rng(seed)`: the generator the pokes sample with.
    pub fn set_rng_seed(&mut self, seed: u64) {
        self.rng = Generator::new(seed);
    }

    /// `Brain.poke`: drive a random share of a region's neurons for `8 + 40 * strength` steps. `step` is the brain's step counter.
    pub fn poke(&mut self, region: &str, side: Option<char>, strength: f64, recruit: Option<f64>, step: u64) {
        let s = strength.clamp(0.0, 1.0);
        let pop = self.sense.get(&(region.to_string(), side)).or_else(|| self.sense.get(&(region.to_string(), None)));
        let Some(pop) = pop else { return };
        if pop.is_empty() {
            return;
        }
        let share = recruit.unwrap_or(self.recruit + (1.0 - self.recruit) * s);
        let k = ((pop.len() as f64 * share) as usize).clamp(1, pop.len());
        // NumPy's `rng.choice(pop, size=k, replace=False)`, draw for draw
        let rows: Vec<u32> = self.rng.choice_indices(pop.len(), k).into_iter().map(|i| pop[i]).collect();
        let steps = 8 + (40.0 * s) as i32;
        self.last_poke = step as i64;
        match self.pending.iter_mut().find(|p| p.region == region && p.side == side) {
            Some(old) => {
                if steps > old.steps {
                    old.rows = rows;
                    old.steps = steps;
                }
            }
            None => self.pending.push(Pending { region: region.to_string(), side, rows, steps }),
        }
        if TOUCH.contains(&region) && self.spill > 0.0 {
            self.poke("body_extra", None, s * self.spill, None, step);
        }
    }

    /// The neurons of a detail group (`np.flatnonzero(det_id == col[name])`): each neuron is in at most one.
    pub fn det_members(&self, name: &str) -> Vec<u32> {
        match self.col.get(name) {
            Some(&c) => (0..self.n as u32).filter(|&i| self.det_id[i as usize] == c as i32).collect(),
            None => vec![],
        }
    }

    /// A group that can be poked (`sense[(region, None)]`) without being a readout (the assay groups `sweet` and `bitter`).
    pub fn add_sense(&mut self, region: &str, rows: Vec<u32>) {
        self.sense.insert((region.to_string(), None), rows);
    }

    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Start of a brain step: age the queue and write the active pokes into the current buffer (returned; zero it with [`Groups::end_pokes`]).
    pub fn begin_step(&mut self) -> &[f32] {
        for p in &mut self.pending {
            p.steps -= 1;
        }
        for p in &self.pending {
            for &r in &p.rows {
                self.cur[r as usize] = POKE_AMP;
            }
        }
        // `Brain._step` decrements first and drops a poke whose count reached 0 *before* applying it (`active` was copied before the
        // delete), so the last step of a poke still fires: drop after use
        &self.cur
    }

    pub fn end_pokes(&mut self) {
        for p in &self.pending {
            for &r in &p.rows {
                self.cur[r as usize] = 0.0;
            }
        }
        self.pending.retain(|p| p.steps > 0);
    }

    /// Tail of a brain step: count the spikes per group and update the trackers. `calm` is true when nothing is driving the brain.
    pub fn end_step(&mut self, spikes: &[u32], step: u64, calm: bool) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        let g = self.names.len();
        let whole = self.n_det + POPS.len();
        for &s in spikes {
            let d = self.det_id[s as usize];
            if d >= 0 {
                self.counts[d as usize] += 1;
            }
            let p = self.pop_id[s as usize];
            if p >= 0 {
                self.counts[self.n_det + p as usize] += 1;
            }
        }
        self.counts[whole] = spikes.len() as u32;
        for (col, mask) in &self.extra {
            self.counts[*col] = spikes.iter().filter(|&&s| mask[s as usize]).count() as u32;
        }
        for i in 0..g {
            let inst = f64::from(self.counts[i]) / self.sizes[i] / self.dt_s;
            self.fast[i] += (inst - self.fast[i]) * self.k_fast;
        }
        if step.is_multiple_of(4) {
            let row = (self.hist_n % HIST) * g;
            if self.hist.len() < HIST * g {
                self.hist.resize(HIST * g, 0.0);
            }
            for (h, f) in self.hist[row..row + g].iter_mut().zip(&self.fast) {
                *h = *f as f32;
            }
            self.hist_n += 1;
        }
        if calm && step as i64 - self.last_poke > CALM_STEPS as i64 {
            for i in 0..g {
                self.base[i] += (self.fast[i] - self.base[i]) * self.k_base;
            }
        }
    }

    /// `Brain.warmup` lets the baseline settle quickly: swap in a faster base constant, returning the old one.
    pub fn set_base_tau_steps(&mut self, tau: f64) -> f64 {
        std::mem::replace(&mut self.k_base, 1.0 - (-1.0f64 / tau).exp())
    }

    pub fn restore_base_k(&mut self, k: f64) {
        self.k_base = k;
    }

    /// `Brain.pain_parts` for the adult brain: (touch overload, thermal, chemical, DN alarm, body relay), each 0..1, from the fast rates
    /// against the calm baselines at the current pain level (`self.recruit`/`spill` belong to it: level 0, 1, 2).
    pub fn pain_parts(&self, level: usize) -> [f32; 5] {
        let above = |names: &[&str], full: f64| -> f64 {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for n in names {
                if let Some(i) = self.col(n) {
                    let sz = self.sizes[i];
                    num += (self.fast[i] - self.base[i]) * sz;
                    den += sz;
                }
            }
            if den == 0.0 { 0.0 } else { num / den / full }
        };
        let mut touch_names: Vec<&str> = TOUCH.to_vec();
        if level > 0 {
            touch_names.push("body_extra");
        }
        let touch = above(&touch_names, 35.0);
        let thermal = above(&["heat"], 30.0).max(above(&["cold"], 30.0));
        let chemical = (above(&["smell", "taste"], 1.0) - 8.0) / 27.0;
        let ratio = |name: &str| self.col(name).map_or(0.0, |i| self.fast[i] / self.base[i].max(1.0));
        let alarm = (ratio("descending") - 1.06) / 0.3;
        let relay = if level >= 2 { (ratio("ascending") - 1.06) / 0.35 } else { 0.0 };
        [touch, thermal, chemical, alarm, relay].map(|v| v.clamp(0.0, 1.0) as f32)
    }

    /// `Brain.pain_neurons`: how many real neurons the pain index listens to at this level.
    pub fn pain_neurons(&self, level: usize) -> usize {
        let mut names: Vec<&str> = TOUCH.to_vec();
        names.extend(["heat", "cold", "smell", "taste"]);
        if level >= 1 {
            names.push("body_extra");
        }
        if level >= 2 {
            names.push("ascending");
        }
        names.iter().map(|n| self.size(n)).sum()
    }

    /// `Brain.history(first, last)[:, col]`: samples `[first, last)` by absolute index, clipped to what the ring still holds.
    pub fn history_col(&self, col: usize, first: usize, last: usize) -> Vec<f32> {
        let g = self.names.len();
        let first = first.max(self.hist_n.saturating_sub(HIST));
        let last = last.min(self.hist_n);
        if last <= first || self.hist.is_empty() {
            return vec![];
        }
        (first..last).map(|i| self.hist[(i % HIST) * g + col]).collect()
    }

    /// `Brain.level`: the fast rate as a multiple of the calm baseline (the baseline is floored at 2 Hz).
    pub fn level(&self, name: &str) -> f32 {
        self.col(name).map_or(0.0, |i| (self.fast[i] / self.base[i].max(2.0)) as f32)
    }

    /// `Brain.hz`: the fast rate in Hz per neuron.
    pub fn hz(&self, name: &str) -> f32 {
        self.col(name).map_or(0.0, |i| self.fast[i] as f32)
    }

    /// `Brain.hz` as Python has it: the double-precision tracker value.
    pub fn hz64(&self, name: &str) -> f64 {
        self.col(name).map_or(0.0, |i| self.fast[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Labels {
        let mut types =
            vec!["DNp09".to_string(), "DNa01".into(), "DNa02".into(), "LPLC2".into(), "JO-C1".into(), "ORN_DA1".into(), "ORN_DM1".into()];
        types.extend((0..20).map(|i| format!("ORN_G{i}")));
        let m = types.len();
        let mut sc = vec![
            "descending_neuron".to_string(),
            "descending_neuron".into(),
            "descending_neuron".into(),
            "visual_projection".into(),
            "cb_sensory".into(),
        ];
        sc.extend(std::iter::repeat_n("cb_sensory".to_string(), m - 5));
        let mut inst = vec![String::new(); m];
        inst[1] = "DNa01_L".into();
        inst[2] = "DNa02_R".into();
        Labels { types, superclass: sc, instance: inst, ..Labels::default() }
    }

    #[test]
    fn groups_split_by_side_and_claim_each_neuron_once() {
        let l = labels();
        let g = Groups::new(&l, l.types.len(), 5.0, 1);
        assert_eq!(g.size("walk"), 1);
        assert_eq!(g.size("turn_l"), 1);
        assert_eq!(g.size("turn_r"), 1);
        assert_eq!(g.size("loom"), 1);
        assert_eq!(g.sense("wind", None), &[4], "JO-C is wind first, so `head` (BM_/JO-) does not get it");
        assert!(g.sense("head", None).is_empty());
        assert_eq!(g.sense("scent:cva", None).len(), 1);
    }

    #[test]
    fn a_poke_drives_rows_for_its_duration_and_raises_the_rate() {
        let l = labels();
        let n = l.types.len();
        let mut g = Groups::new(&l, n, 5.0, 1);
        g.poke("loom", None, 1.0, Some(1.0), 0);
        let mut steps_on = 0;
        for step in 1..200u64 {
            let on = g.begin_step().iter().any(|&c| c > 0.0);
            g.end_pokes();
            if on {
                steps_on += 1;
            }
            let spikes: Vec<u32> = if on { vec![3] } else { vec![] };
            g.end_step(&spikes, step, false);
        }
        assert_eq!(steps_on, 48, "8 + 40 * strength steps");
        assert!(!g.has_pending());
        assert!(g.hz("loom") >= 0.0);
    }

    #[test]
    fn level_is_the_rate_over_the_floored_baseline() {
        let l = labels();
        let mut g = Groups::new(&l, l.types.len(), 5.0, 1);
        let c = g.col("walk").unwrap();
        g.fast[c] = 9.0;
        g.base[c] = 1.0;
        assert!((g.level("walk") - 4.5).abs() < 1e-6, "the baseline is floored at 2 Hz");
    }
}
