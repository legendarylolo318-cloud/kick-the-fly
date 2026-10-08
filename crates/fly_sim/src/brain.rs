//! A fly's brain as the game and the assays drive it: a [`LifSim`] plus the held currents `Brain._step` adds to the sensory input.
//!
//! Python: `game/kick_the_fly.py:Brain` (the parts that touch the simulator: `drive_cur`, `override`, sedation/death inhibition,
//! `warmup`) and `core/simcore.py:rows_of`. The group tables, `poke` queue, rate trackers, pain index and plasticity are not
//! ported yet; `PORTING.md` section 2.3 lists them.

use std::sync::Arc;

use crate::lif::{LifParams, LifSim};
use crate::pack::Labels;
use crate::Error;

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
}

impl Brain {
    pub fn new(sim: LifSim, labels: Arc<Labels>) -> Self {
        let n = sim.n;
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
        }
    }

    /// A warmed-up brain like `simcore.new_brain(seed, warmup=600)`.
    pub fn warmed_up(weights: Arc<crate::csr::Weights>, labels: Arc<Labels>, params: LifParams, seed: u64, warmup: usize) -> Self {
        let mut brain = Self::new(LifSim::new(weights, params, seed), labels);
        brain.warmup(warmup);
        brain
    }

    pub fn n(&self) -> usize {
        self.sim.n
    }

    pub fn labels(&self) -> &Labels {
        &self.labels
    }

    pub fn rows_of(&self, spec: &str) -> Result<Vec<u32>, Error> {
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

    pub fn dead(&self) -> bool {
        self.death_step.is_some()
    }

    /// `Brain.kill`: start the death ramp and stop the gain controller adapting.
    pub fn kill(&mut self) {
        self.death_step = Some(self.steps);
        self.sim.p.gain_adapt = 0.0;
    }

    /// One brain step (`Brain._step`, without the group trackers): returns the spikes.
    pub fn step(&mut self) -> &[u32] {
        self.steps += 1;
        if let Some(d) = self.death_step {
            let ramp = (((self.steps - 1 - d) as f64 / DEATH_RAMP_STEPS as f64).clamp(0.0, 1.0)).max(f64::from(self.sedation)).min(1.0);
            let level = INHIB * ramp as f32;
            self.scratch.fill(level);
            return self.sim.step(Some(&self.scratch));
        }
        let any = self.sedation > 0.0 || self.surgery || self.driving;
        if !any {
            return self.sim.step(None);
        }
        self.scratch.fill(0.0);
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
        self.sim.step(Some(&self.scratch))
    }

    /// Run `steps` calm steps (the game warms a new brain up for 600).
    pub fn warmup(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
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
        assert!(rows_of(&l, 5, "rows:9").is_err());
        assert!(rows_of(&l, 5, "bogus").is_err());
        assert!(rows_of(&l, 5, "nope:x").is_err());
    }
}
