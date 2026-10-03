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

## Result (this release; the 11 pathway behaviors over the full grid, seeds 1000-1009, NumPy CPU, 5,140 s with 4 workers)

The T-maze (mb_conditioning) is a second, much slower run (`--sens-tests mb_conditioning`); it is complete (all 26 cells; the first 5 by
Sonnet, the other 21 resumed from that progress file in the 3.0 day 4 review, 10,916 s with 6 workers; files in
`docs/results/day4/sensitivity_tmaze/`), and its table is below the pathway one. The baseline column equals validation's own result for the same seeds exactly (checked for all 11). **PASS/FAIL by parameter value
(effect = the drive ratio):**

| parameter | values at which a behavior FAILS (default passes) |
|---|---|
| noise 0.025 / 0.0375 / **0.05** / 0.075 / 0.10 | cold pathway at 0.025; sugar -> MN9 and bitter -> DNg28 at 0.075 and 0.10; antennal -> aDN, courtship song and optomotor at 0.10 |
| tonic drive 0.15 / 0.175 / **0.20** / 0.225 / 0.25 | cold at 0.175; sugar, song, bitter, optomotor at 0.225 and 0.25; antennal -> aDN and DA1 PN -> LH at 0.25 |
| target rate 2.5 / 3.5 / **5** / 7.5 / 10 Hz | sugar at 2.5; bitter at 2.5, 3.5 and 7.5; hot and cold at 7.5 and 10; ORN DA1 -> DA1 PN and DA1 PN -> LH at 10 |
| sensory gain 2 / 3 / **4** / 5 / 6 | only bitter -> DNg28, at 2 and 3 |
| gain adaptation 0 / 0.001 / **0.002** / 0.004 / 0.008 | hot and cold only, at 0 (frozen gain) |
| synapse threshold **3** / 4 / 5 / 6 / 8 / 10 | only bitter -> DNg28, at 8 and 10 |

23 of the 66 parameter x behavior pairs fail at one or more tested values. Reading it: the looming -> giant fiber pathway never fails (its drive ratio
stays between 1.75 and 30x); the **pathway results are most fragile to the two parameters that set how close every neuron sits to threshold** (tonic drive
above the default and noise at 2x or more both drown the weaker pathways) and to a lower or higher whole-brain target rate; they are least sensitive to the
sensory gain (the drive ratio saturates above gain 4: identical looming ratios at 5 and 6) and to the synapse threshold (the existing threshold sweep's
finding, here on 11 behaviors). Bitter -> DNg28 is the most fragile (it fails at 9 of the 25 tested values): its FAILs at the highest thresholds and lowest
gains come from the sugar-pathway control rising along with it (the criterion compares drive with control), not from the drive ratio itself falling below
1.5. At the lowest tonic drive (0.15) every ratio is 4-25x: a quieter brain has a lower baseline, and the ratio divides by it (with a floor of 0.5 Hz), so a big
ratio is not the same as a stronger pathway (a hypothesis, not tested here: the rates themselves are in the JSON's per-seed values of a validation run, not in this
table). **None of this changes a default and none of it says the
defaults are right or wrong.** The heatmap, matrix and tables are in the exports; the numbers above are from `sensitivity_matrix.csv` of this run.

## T-maze conditioning (mb_conditioning), full grid (3.0 day 4 review; seeds 1000-1009, n = 10, NumPy CPU)

Effect = the paired T-maze PI (pass: PI >= 0.5, |unpaired PI| <= 0.25, paired > unpaired at p < 0.01). Baseline (all defaults): **PASS, PI 1.00**.

| parameter (default) | tested values: PASS/FAIL and PI |
|---|---|
| Membrane noise (0.05) | 0.025 PASS 0.99 · 0.0375 PASS 1.00 · 0.075 PASS 0.53 · **0.1 FAIL -0.04** |
| Tonic drive (0.20) | 0.15 PASS 0.66 · 0.175 PASS 1.00 · **0.225 FAIL 0.29** · **0.25 FAIL 0.19** |
| Target rate, Hz (5) | 2.5 PASS 0.97 · 3.5 PASS 1.00 · 7.5 PASS 1.00 · **10 FAIL -0.04** |
| Sensory input gain (4) | 2, 3, 5, 6: PASS 1.00 at every value |
| Gain adaptation rate (0.002) | 0 PASS 0.99 · 0.001, 0.004, 0.008 PASS 1.00 |
| Synapse threshold (3) | 4, 5, 6, 8, 10: PASS 1.00 at every value |

Conditioning survives every tested sensory gain, gain-adaptation rate and synapse threshold, and fails at the high end of noise (0.1), of the
tonic drive (0.225 and above) and of the target rate (10 Hz): the same three parameters the pathway behaviors are most fragile to. Every cell is
a MODEL PREDICTION; nothing here changes a default.
