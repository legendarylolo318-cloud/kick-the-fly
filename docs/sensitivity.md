# Sensitivity analysis (Lab > Sensitivity analysis, `--sensitivity`) — 3.0 day 4

How much does each validated behavior depend on the simulation's own parameters? Each LIF parameter is varied across a documented range,
**one at a time**, and every validated behavior is re-run with **validation's own code and pass criteria** (drive ratio >= 1.5 and above the
control in a one-sided Wilcoxon at p < 0.01; T-maze PI >= 0.5, |unpaired PI| <= 0.25 and paired > unpaired). The output is a heatmap:
parameter value x behavior -> PASS/FAIL and an effect size.

**This is analysis only. The defaults are not changed, retuned or re-centred on anything it reports.** A behavior that is fragile to a
parameter is a finding about the model; one that is robust is not evidence that the default is right.

| | what it is |
|---|---|
| **CONNECTOME** | the pathways being tested and the synapse counts the threshold prunes |
| **GAME RULE** | the ranges below (how far to look) and the pass criteria (chosen for this release, from validation.py, not from the papers) |
| **MODEL PREDICTION** | every cell: "this behavior still passes when the noise is doubled" is a statement about the model |

## The grid

| parameter | default | tested values | note |
|---|---|---|---|
| noise (`noise_std`) | 0.05 | 0.025, 0.0375, 0.075, 0.10 | random current on every neuron each step; the code notes 0.10 drowned vision and 0.03 was too quiet |
| tonic drive (`bias`) | 0.20 | 0.15, 0.175, 0.225, 0.25 | at 0.20 a neuron rests at 80% of threshold; at 0.25 it rests at threshold |
| target rate (`target_rate_hz`) | 5 Hz | 2.5, 3.5, 7.5, 10 | the whole-brain rate the slow gain controller steers toward |
| sensory gain (`ext_gain`) | 4 | 2, 3, 5, 6 | how hard driven neurons are pushed: the stimulus strength of every pathway test |
| gain adaptation (`gain_adapt`) | 0.002 | 0, 0.001, 0.004, 0.008 | 0 freezes the synaptic gain where it starts |
| synapse threshold (`min_synapses`) | 3 (the pack's own floor) | 4, 5, 6, 8, 10 | keep only connections of at least this many synapses (the robustness sweep's range) |

Every LIF value lies inside that parameter's Lab > Parameters range (tested). The behaviors are the **12 validation reproduces** (the ones
that fail on the unmodified connectome are left out: "stops passing" means nothing for a behavior that never passed). The baseline column
(every parameter at its default) **is validation's own result** for the same seeds (tested).

## Using it

- **Lab > Sensitivity analysis:** a quick check (one parameter at two values, two behaviors, three seeds: **underpowered**, a FAIL there only
  means "cannot pass at n = 3") or the full grid. The heatmap is drawn in the accessibility palette you have chosen; hover a cell for its
  control, effect size (Cohen's d_z of drive minus control over the seeds), p and n.
- **Headless:** `python kick_the_fly.py --headless --sensitivity [--sens-params noise_std bias ...] [--sens-tests looming_escape ...]
  [--sens-values noise_std=0.03,0.07] [--seeds 1000-1009] --workers 6 --out DIR`. **Resumable:** every finished (parameter, value) cell is
  written to `sensitivity_progress.json` as it finishes; `--resume` continues and refuses a progress file made for different seeds, behaviors
  or cells. Cells run through the same worker-process pool validation uses.
- **Exports** (written to `--out`): `sensitivity.json` (everything), `sensitivity.csv` (one row per parameter, value, behavior),
  `sensitivity_matrix.csv` (the heatmap as a table), `sensitivity_robustness.csv` (per parameter and behavior: how many values pass and the
  first value, going away from the default each way, at which it stops) and `sensitivity_heatmap.svg` (self-contained, no scripts).
- **Python:** `from kickthefly.lab.api import sensitivity`.
- **Cost:** a full grid is 26 cells of one validation run each (hours on a desktop); that is why it is resumable.

## How a parameter reaches the brain

`validation.run(..., params=...)` (new, optional; `None` is the validation suite as published, and `--validate` is byte-identical, diffed) builds
each fly with `lab.apply_to_sim` before its warm-up, so the gain controller settles into the changed value before the stimulus starts. The
synapse threshold is a `Wiring(min_synapses=N)`, the path the threshold sweep already takes. A subset run skips the pathway tests nobody asked
for; each test restores the same snapshot, and the control neurons are drawn as if every test had run, so skipping never changes a result
(tested: the subset's numbers equal the full run's).
