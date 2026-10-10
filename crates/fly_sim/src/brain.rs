//! A fly's brain as the game and the assays drive it: a [`LifSim`] plus the held currents `Brain._step` adds to the sensory input.
//!
//! Python: `game/kick_the_fly.py:Brain` (the parts that touch the simulator: `drive_cur`, `override`, sedation/death inhibition,
//! `warmup`) and `core/simcore.py:rows_of`. The group tables, `poke` queue, rate trackers, pain index and plasticity are not
//! ported yet; `PORTING.md` section 2.3 lists them.

use std::sync::Arc;

use crate::Error;
use crate::groups::Groups;
use crate::lif::{LifParams, LifSim};
use crate::pack::Labels;

/// `Brain._inhib`: the per-step current of full sedation or death, before the `ext_gain` of 4 (so -1.6 per step).
pub const INHIB: f32 = -0.4;
/// `STIM_AMP`: current of a driven neuron (x `ext_gain` 4 = 2.0 per step, enough to fire every refractory cycle).
pub const STIM_AMP: f32 = 0.5;
/// `SURGERY_CURRENT`: silence (-1) and stimulate (+1).
pub const SURGERY_SILENCE: f32 = -0.6;
pub const SURGERY_STIMULATE: f32 = 0.12;
/// Steps over which death (and the `kill` ramp) deepens to full inhibition.
pub const DEATH_RAMP_STEPS: u64 = 300;

/// Neuron rows from a spec: `type:DNp01,MDN`, `prefix:KC`, `superclass:descending_neuron`, `rows:1,2,3` (`simcore.rows_of`).
pub fn rows_of(labels: &Labels, n: usize, spec: &str) -> Result<Vec<u32>, Error> {
    let (kind, arg) = spec.split_once(':').ok_or_else(|| Error::Spec(format!("'{spec}' has no kind: prefix")))?;
    let pick = |names: &[String], pred: &dyn Fn(&str) -> bool| -> Vec<u32> {
        names.iter().enumerate().filter(|(_, t)| pred(t)).map(|(i, _)| i as u32).collect()
    };
    if kind == "side" {
        // `side:R:DNa01,DNa02`: those types, on one side of the brain (the instance name ends `_R` or `_L`)
        let (side, types) = arg.split_once(':').ok_or_else(|| Error::Spec(format!("'{spec}': side:<L|R>:<types>")))?;
        let suffix = format!("_{side}");
        let items: Vec<&str> = types.split(',').collect();
        return Ok((0..n as u32)
            .filter(|&i| {
                items.contains(&labels.types[i as usize].as_str()) && labels.instance.get(i as usize).is_some_and(|x| x.ends_with(&suffix))
            })
            .collect());
    }
    let items: Vec<&str> = arg.split(',').collect();
    match kind {
        "type" => Ok(pick(&labels.types, &|t| items.contains(&t))),
        "prefix" => Ok(pick(&labels.types, &|t| items.iter().any(|p| t.starts_with(p)))),
        "superclass" => Ok(pick(&labels.superclass, &|t| items.contains(&t))),
        "rows" => {
            let mut rows = Vec::new();
            for part in arg.split(',').filter(|p| !p.is_empty()) {
                let r: u32 = part.trim().parse().map_err(|_| Error::Spec(format!("bad row '{part}'")))?;
                if r as usize >= n {
                    return Err(Error::Spec(format!("row {r} is outside 0..{n}")));
                }
                rows.push(r);
            }
            Ok(rows)
        }
        _ => Err(Error::Spec(format!("unknown neuron spec '{spec}'"))),
    }
}

/// One fly: the simulator, its held currents and its life/death state.
pub struct Brain {
    pub sim: LifSim,
    labels: Arc<Labels>,
    /// Experimental activation (assays, protocols): its own current, added to surgery.
    drive_cur: Vec<f32>,
    driving: bool,
    /// Brain surgery: per-neuron silencing (-0.6) or stimulating (+0.12) current.
    override_cur: Vec<f32>,
    surgery: bool,
    /// 0..1 share of the death inhibition (solvent, cold, venom).
    pub sedation: f32,
    death_step: Option<u64>,
    /// Brain steps taken.
    pub steps: u64,
    scratch: Vec<f32>,
    /// Named groups, the poke queue and the rate trackers (`Brain.poke`, `level`, `hz`).
    pub groups: Groups,
    gain_adapt: f64,
    /// History sample count and calm baselines at the moment of death (the autopsy reads them).
    pub death_sample: usize,
    pub death_base: Vec<f32>,
    /// Learning on the mushroom body synapses (`Brain.memory`), when the pack has the dopamine tables and it was switched on.
    pub memory: Option<Box<crate::memory::Memory>>,
}

/// What a save keeps of a brain: the simulator, the group rate trackers and the step counter (`core/savestate.py:brain_state`).
#[derive(Debug, Clone)]
pub struct BrainState {
    pub sim: crate::lif::SimState,
    pub fast: Vec<f32>,
    pub base: Vec<f32>,
    pub steps: u64,
}

impl Brain {
    /// `seed` is the one the simulator was made with.
    pub fn export_state(&self, seed: u64) -> BrainState {
        BrainState { sim: self.sim.export_state(seed), fast: self.groups.fast.clone(), base: self.groups.base.clone(), steps: self.steps }
    }

    /// Put a saved state back (the caller has checked that the save is for this brain pack).
    pub fn import_state(&mut self, st: &BrainState) {
        self.sim.import_state(&st.sim);
        if st.fast.len() == self.groups.fast.len() && st.base.len() == self.groups.base.len() {
            self.groups.fast.copy_from_slice(&st.fast);
            self.groups.base.copy_from_slice(&st.base);
        }
        self.steps = st.steps;
        self.groups.last_poke = -1_000_000_000;
        self.groups.clear_pending();
    }

    pub fn new(sim: LifSim, labels: Arc<Labels>) -> Self {
        let n = sim.n;
        let mut groups = Groups::new(&labels, n, sim.p.dt_ms, 0x6b69_636b);
        let (sweet, bitter) = crate::assays::taste_groups(&labels, &sim.weights().csr);
        groups.add_sense("sweet", sweet);
        groups.add_sense("bitter", bitter);
        let gain_adapt = sim.p.gain_adapt;
        Self {
            sim,
            labels,
            drive_cur: vec![0.0; n],
            driving: false,
            override_cur: vec![0.0; n],
            surgery: false,
            sedation: 0.0,
            death_step: None,
            steps: 0,
            scratch: vec![0.0; n],
            groups,
            gain_adapt,
            memory: None,
            death_sample: 0,
            death_base: Vec::new(),
        }
    }

    /// Switch learning on (`simcore.new_brain(memory=True)`); false when the pack has no dopamine/MBON tables.
    pub fn enable_memory(&mut self) -> bool {
        self.memory = crate::memory::Memory::new(&self.labels, self.sim.weights()).map(Box::new);
        self.memory.is_some()
    }

    /// A warmed-up brain like `simcore.new_brain(seed, warmup=600)`.
    pub fn warmed_up(weights: Arc<crate::csr::Weights>, labels: Arc<Labels>, params: LifParams, seed: u64, warmup: usize) -> Self {
        let mut brain = Self::new(LifSim::new(weights, params, seed), labels);
        brain.warmup(warmup);
        brain
    }

    /// `Brain.seed`/`Brain.rng` as `simcore.new_brain(seed)` sets them (the pokes' sampler is `default_rng(seed)`).
    pub fn set_poke_seed(&mut self, seed: u64) {
        self.groups.set_rng_seed(seed);
    }

    /// `Game.set_lab_param` for a model parameter on this brain (a dead brain's gain stays frozen until it revives).
    pub fn set_model_param(&mut self, name: &str, v: f64) -> bool {
        if !self.sim.set_model_param(name, v) {
            return false;
        }
        if name == "gain_adapt" {
            self.gain_adapt = v;
            if self.dead() {
                self.sim.p.gain_adapt = 0.0;
            }
        }
        true
    }

    pub fn n(&self) -> usize {
        self.sim.n
    }

    pub fn labels(&self) -> &Labels {
        &self.labels
    }

    /// Rows of a spec (`type:..`, `prefix:..`, ...) or of a pokeable group by name (`sweet`, `bitter`, `reward`, ...).
    pub fn rows_of(&self, spec: &str) -> Result<Vec<u32>, Error> {
        if !spec.contains(':') {
            let g = self.groups.sense(spec, None);
            if !g.is_empty() {
                return Ok(g.to_vec());
            }
        }
        rows_of(&self.labels, self.sim.n, spec)
    }

    /// `simcore.drive`: hold these neurons driven at `amp` until [`Brain::undrive`].
    pub fn drive(&mut self, rows: &[u32], amp: f32) {
        for &r in rows {
            self.drive_cur[r as usize] = amp;
        }
        self.driving = self.drive_cur.iter().any(|&c| c != 0.0);
    }

    pub fn undrive(&mut self, rows: &[u32]) {
        self.drive(rows, 0.0);
    }

    /// `set_override`: silence (`-1`), leave alone (`0`) or stimulate (`1`) these neurons.
    pub fn set_surgery(&mut self, rows: &[u32], mode: i8) {
        let c = match mode {
            m if m < 0 => SURGERY_SILENCE,
            0 => 0.0,
            _ => SURGERY_STIMULATE,
        };
        for &r in rows {
            self.override_cur[r as usize] = c;
        }
        self.surgery = self.override_cur.iter().any(|&c| c != 0.0);
    }

    /// `clear_overrides`: no brain surgery at all.
    pub fn clear_surgery(&mut self) {
        self.override_cur.fill(0.0);
        self.surgery = false;
    }

    /// Groups, for the surgery screen's row sets.
    pub fn groups_ref(&self) -> &Groups {
        &self.groups
    }

    pub fn dead(&self) -> bool {
        self.death_step.is_some()
    }

    /// `Brain.kill`: start the death ramp and stop the gain controller adapting.
    pub fn kill(&mut self) {
        self.death_sample = self.groups.hist_n;
        self.death_base = self.groups.base.clone();
        self.death_step = Some(self.steps);
        self.sim.p.gain_adapt = 0.0;
    }

    /// `Brain.revive`: bring a dead brain back (the gain controller adapts again).
    pub fn revive(&mut self) {
        if self.death_step.take().is_some() {
            self.sim.p.gain_adapt = self.gain_adapt;
        }
    }

    /// True while nothing but the sensory pokes drives the brain: the baseline only learns then.
    fn calm(&self) -> bool {
        self.death_step.is_none() && self.sedation == 0.0 && !self.surgery && !self.driving
    }

    /// The Neurodex's "calm" (`Extras.dex_tick`): nothing touched it for 2 s, no sedation, no surgery, no drive.
    pub fn dex_calm(&self) -> bool {
        self.sedation == 0.0 && !self.surgery && !self.driving && self.steps as i64 - self.groups.last_poke > 400
    }

    /// Brain surgery or a held drive is on.
    pub fn driven(&self) -> bool {
        self.surgery || self.driving
    }

    /// `nd.window_spikes`: the neuron indices of every spike in the last `steps` steps, and how many steps that really covers.
    pub fn window_spikes(&self, steps: usize) -> (Vec<u32>, usize) {
        let raster: Vec<&[u32]> = self.sim.activity.raster().collect();
        let tail = &raster[raster.len().saturating_sub(steps)..];
        if tail.is_empty() {
            return (vec![], 1);
        }
        (tail.iter().flat_map(|r| r.iter().copied()).collect(), tail.len())
    }

    /// `Brain.poke` on this brain.
    pub fn poke(&mut self, region: &str, side: Option<char>, strength: f32) {
        if self.dead() {
            return;
        }
        let step = self.steps;
        self.groups.poke(region, side, strength, None, step);
    }

    /// `Brain.poke` with an explicit share of the region's neurons recruited.
    pub fn poke_recruit(&mut self, region: &str, side: Option<char>, strength: f32, recruit: Option<f32>) {
        if self.dead() {
            return;
        }
        let step = self.steps;
        self.groups.poke(region, side, strength, recruit, step);
    }

    /// `Brain.level`.
    pub fn level(&self, name: &str) -> f32 {
        self.groups.level(name)
    }

    /// `Brain.hz`.
    pub fn hz(&self, name: &str) -> f32 {
        self.groups.hz(name)
    }

    /// One brain step (`Brain._step`): queued pokes, held currents, the step itself, then the group rate trackers.
    pub fn step(&mut self) -> &[u32] {
        self.steps += 1;
        if let Some(d) = self.death_step {
            let ramp = (((self.steps - 1 - d) as f64 / DEATH_RAMP_STEPS as f64).clamp(0.0, 1.0)).max(f64::from(self.sedation)).min(1.0);
            let level = INHIB * ramp as f32;
            self.scratch.fill(level);
            self.groups.clear_pending();
            self.sim.step(Some(&self.scratch));
            let calm = false;
            self.groups.end_step(self.sim.spikes(), self.steps, calm);
            return self.sim.spikes();
        }
        let poked = self.groups.has_pending();
        let any = self.sedation > 0.0 || self.surgery || self.driving || poked;
        if !any {
            self.sim.step(None);
        } else {
            self.scratch.fill(0.0);
            if poked {
                let cur = self.groups.begin_step();
                for (x, c) in self.scratch.iter_mut().zip(cur) {
                    *x += c;
                }
                self.groups.end_pokes();
            }
            if self.sedation > 0.0 {
                let s = INHIB * self.sedation;
                for x in &mut self.scratch {
                    *x += s;
                }
            }
            if self.surgery {
                for (x, o) in self.scratch.iter_mut().zip(&self.override_cur) {
                    *x += o;
                }
            }
            if self.driving {
                for (x, d) in self.scratch.iter_mut().zip(&self.drive_cur) {
                    *x += d;
                }
            }
            self.sim.step(Some(&self.scratch));
        }
        let calm = self.calm();
        self.groups.end_step(self.sim.spikes(), self.steps, calm);
        if self.steps.is_multiple_of(crate::memory::UPDATE_STEPS) && self.memory.is_some() {
            let calm_for_memory = calm && self.steps as i64 - self.groups.last_poke > crate::groups::CALM_STEPS as i64;
            if let Some(m) = self.memory.as_mut() {
                m.step(&mut self.sim, calm_for_memory);
            }
        }
        self.sim.spikes()
    }

    /// Run `steps` calm steps (the game warms a new brain up for 600).
    pub fn warmup(&mut self, steps: usize) {
        let k = self.groups.set_base_tau_steps(60.0); // let the baseline settle quickly
        for _ in 0..steps {
            self.step();
        }
        self.groups.restore_base_k(k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Labels {
        Labels {
            types: ["DNp01", "DNp01", "KCg-m", "KCa", "MDN"].map(String::from).to_vec(),
            superclass: ["descending_neuron", "descending_neuron", "cb_intrinsic", "cb_intrinsic", "descending_neuron"]
                .map(String::from)
                .to_vec(),
            ..Labels::default()
        }
    }

    #[test]
    fn specs_resolve_like_python() {
        let l = labels();
        assert_eq!(rows_of(&l, 5, "type:DNp01,MDN").unwrap(), vec![0, 1, 4]);
        assert_eq!(rows_of(&l, 5, "prefix:KC").unwrap(), vec![2, 3]);
        assert_eq!(rows_of(&l, 5, "superclass:cb_intrinsic").unwrap(), vec![2, 3]);
        assert_eq!(rows_of(&l, 5, "rows:4,1").unwrap(), vec![4, 1]);
        assert!(rows_of(&l, 5, "side:R:DNp01").unwrap().is_empty(), "no instance names in this fixture");
        assert!(rows_of(&l, 5, "rows:9").is_err());
        assert!(rows_of(&l, 5, "bogus").is_err());
        assert!(rows_of(&l, 5, "nope:x").is_err());
    }
}
