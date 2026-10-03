# Lab mode

**Lab** (Esc > Mode, or Settings > Brain) replaces Play's challenges with research tools. Everything here also runs
headless, and the same machinery is a Python library ([api.md](api.md)). What comes from the connectome and what is
a game rule is the same as in the game (README: What is the connectome and what is a game rule).

## Lab tools

- **Validation:** every test with PASS or FAIL, the measured numbers, the pass criteria and the citation. "Run validation now" reruns the suite on your PC. Failures stay in the table. The full table and what each result means: [validation.md](validation.md).
- **Assays and repeated trials:** T-maze conditioning (Tully & Quinn performance index), looming escape (escape probability, latency and distance vs approach speed) and sugar response (MN9 dose-response) over any number of flies (seeds), with mean and 95% confidence interval. Pick a surgery and every fly also runs unperturbed with the same seed as its control, compared with a paired Wilcoxon signed-rank test (paired t-test and, for yes/no outcomes, Fisher's exact test alongside). Each fly is a fresh, untrained brain in a worker process; your saved training memory isn't touched. Results export to JSON and CSV.
- **Psychometrics generator:** sweep any stimulus parameter across a continuous range, run N trials per level with mean and 95% confidence interval error bars, and export publication-ready vector figures (pure vector PDF-1.4, SVG) and raw CSV data.
- **Optogenetics laser:** in-world aimable beam activating or silencing selected cell types directly in real time for rapid perturbation experiments.
- **Classroom mode & lecture protocols:** self-contained teaching modules with guided steps, hypothesis prompts, and bundled interactive YAML lecture protocols (`protocols/lecture_*.yaml`).
- **Parameters:** the LIF model's parameters (noise, tonic drive, target rate, sensory gain, gain adaptation; tagged MODEL) and the game-rule thresholds that turn neuron firing into moves, live, including 2.9's song, lunge, sleep and CO2/heat/cold log thresholds and the outdoor day length. Validation results, exports and save states record when anything is changed from the defaults.
- **Record and export:** pick neuron groups and a duration and record the fly you're looking at while you play: spike times and firing rates as CSV and npz, with a metadata JSON (app version, seed, parameters, thresholds, connectome version, brain pack checksum, surgery, arena and its weather/fruit settings). Tick **NWB** to also get one Neurodata Without Borders file (units with spike times and connectome labels, per-group and per-region rates, stimuli, tool events, the fly's movement, surgery, arena, KC->MBON weights before and after, full metadata and the MaleCNS v1.0 / CC BY 4.0 citation). NWB needs `pip install pynwb`; it isn't bundled in the exe or AppImage, and the checkbox says so when it's missing.
- **Critical path finder:** pick a validated behavior or an assay and it silences each candidate cell type in turn (a shortlist ranked by how much of the readout's input they supply within two synapses), re-runs it over the validation seeds against same-seed unperturbed controls, and ranks the types by effect with 95% CI and a paired Wilcoxon test. Resumable, CSV/JSON export, and a one-click "apply this lesion" in the game. Headless: `--critical-path TARGET`.
- **Predator escape and the hum demo (3.0 day 3):** two more assays (`predator_escape`, `hum_demo`). Predator escape: how often a fly escapes a frog, a dragonfly and a mantis, with 95% Wilson intervals ([predators.md](predators.md)). Hum demo: a synthetic hum onto JO-A/B and the courtship pathway's response ([microphone.md](microphone.md)). Both are MODEL PREDICTIONS with pre-registered criteria. **Weather** (`weather.rain`, `weather.gust_hz`, `weather.storm`) is under Parameters ([weather.md](weather.md)), and Esc > Mic and streamer holds the opt-in microphone and Streamer mode ([streamer.md](streamer.md)).
- **Network science, Sleep deprivation and Sensitivity analysis (3.0 day 4):** three more Lab pages, each a background job with a progress bar and a Cancel button. **Network science** computes degree distributions, reciprocity, 3-node motifs against a degree-preserving null, rich club, communities and per-region summaries of the adult or larva pack, cached with a checksum, exported to CSV ([network-science.md](network-science.md)). **Sleep deprivation** keeps flies awake with timed disturbances and measures rebound sleep against their own controls; sleep pressure is a game rule, the dFB readout the connectome's ([sleep-deprivation.md](sleep-deprivation.md)); it is also an assay kind (`sleep_deprivation`, paired, no surgery) for protocols. **Sensitivity analysis** varies each LIF parameter and the synapse threshold and re-runs the validated behaviors with validation's own criteria, drawing a parameter x behavior heatmap; analysis only, resumable, `--sensitivity` headless ([sensitivity.md](sensitivity.md)). The **Fly arcade** (tournament and racing, [tournament.md](tournament.md), [racing.md](racing.md)) is in the pause menu, not the Lab; its headless forms are `--tournament N` and `--race`.
- **Outdoor arena parameters:** open field wind direction and speed, sun azimuth and elevation, the day/night cycle's day length (0 = off), and the orchard's feeds per fruit, regrow time and fruit cap (all GAME RULE). They're also valid protocol `params`, and `assay: orchard` runs the orchard's feeding schedule headless and reproducibly (`protocols/orchard-feeding.yaml`).
- **Simulation benchmark** ([performance.md](performance.md)): runs 1, 8 and 16 flies on the compute backend you picked and reports the one that actually ran and its device, paced and uncapped steps/s and the sim/real ratio, neuron updates/s, synaptic events/s (measured spikes x the mean out-degree of 61.6) and the process's memory. Headless: `--benchmark --backend NAME --flies 1 8 16 32 --seconds 5`.
- **Connectome robustness & research findings:**
  - **Synapse threshold sweeps:** drops connections below any synapse count and re-runs the validated behaviors with validation's own criteria (seeds 1000-1009). The brain pack is already filtered at 3 synapses, so 1-3 change nothing. Pruning below 10 synapses removes 73.6% of all connections (7,562,973) and every input of 10,151 neurons, yet all 4 validated behaviors survive. Watch the rates, not only the ratios: a pruned brain is quieter at rest, so looming's ratio rises (11.8 -> 20.3) while DNp01's driven rate stays at 66.5 spikes/s.
  - **Transmitter sign flips:** flips a random half of the neurons whose transmitter the dataset is less than 70% sure of (11,013, of which 4,366 have no confidence at all), over randomized trials. In 3 trials on seeds 1000-1006: looming -> giant fiber survives 3/3 (x13.2 vs x11.7 unperturbed) and antennal -> aDN survives 3/3 (x3.6 vs x4.9), while sugar -> MN9 fails 3/3 (x1.06 vs x2.22).
  - **Looming critical path:** single-group silencing shows LC4 (-50%) and LPLC2 (-43%) carry nearly all looming drive; other visual groups have near-zero effect.
  - **Global inhibition block (Picrotoxin), now in Lab > Pharmacology (3.0):** 0-100% severity slider scales inhibitory synapses down (`inhibition_scale = 1 - severity`). Runaway firing emerges from disinhibition without scripted seizures: at 100% the brain-wide mean goes from 6.7 to 33.8 spikes/s (one seed, 1 s; re-checked for this release). Reports before/after firing distributions.
- **Neural clamp:** records spike trains from a reference run and replays forced spikes into an altered connectome (lesion, threshold, sign-flip) to isolate wiring changes from sensory feedback. Dynamic clamping overrides intrinsic membrane state and breaks closed-loop feedback loops (e.g. proprioception and visual flow). Shows side-by-side activity diffs and exports.
- **Connectome diff mode:** runs two flies (reference vs perturbed) side-by-side with identical seeds and inputs in lockstep. Tracks region-by-region activity divergence live with an autopsy-style diverging bar chart and a timeline showing when the two brains diverge.
- **Hemifield & hemisphere lesions:** one-click surgery silencing unilateral visual pathways (LC10, LPLC2, LC4, LPTC, VS, HS) or an entire hemisphere. Demonstrates blind-side dodge failure, asymmetric steering bias, and broken 1v1 duel tracking. Reported strictly as a connectome wiring outcome, not physical injury.
- **Protocols:** YAML experiment files, from the bundled examples or your protocols folder.
- **Real vs rule tags** are on by default in Lab: each reaction in the brain panel and its popup is tagged REAL (live descending-neuron firing crossed a threshold; the movement itself is always game physics) or RULE (a game rule). In morphology, key cell types use REAL EM skeletons from neuPrint with fallback to synthetic fibers.
- **Neuron search and path tracer (big brain view, B):** type a cell type, instance or body ID into the search box to
  find and inspect a neuron. In the inspector, PATH FROM HERE and PATH TO HERE on two neurons draw the 5 strongest
  paths of up to 3 synapses between them (strength = the product of the synapses' weights, each the signed synapse
  count as a share of the target's input; sign = the product of signs), with live spikes running along them.
  `kickthefly.lab.neurosearch` does the same from a script.
- **Brain surgery groups added in 2.9:** pIP10 (song), P1, AVLP727m (TK-FruM) and the dorsal fan-shaped body
  (FB6/FB7). Stimulating pIP10 makes ps1 fire and the fly buzz; P1 makes flies lunge at each other; FB6/FB7 put it to
  sleep. Nothing in play drives these on its own.

## Protocols and headless runs

A protocol is either a stimulus schedule with recordings or a standard assay over many flies:

```yaml
name: looming-giant-fiber
seed: 100
flies: 5
warmup_s: 2
duration_s: 3
surgery: {"type:LPLC2,LC4": -1}     # silence (-1) or stimulate (1); each seed also runs unperturbed as its control
stimuli:
  - {at_s: 1.0, for_s: 1.0, target: loom, strength: 0.9, recruit: 0.6}      # game-style stimulus
  - {at_s: 2.5, for_s: 0.5, target: "type:MDN", mode: drive, amp: 0.5}      # constant activation
recordings:
  - {name: giant_fiber, neurons: dnp01}
  - {name: descending, neurons: "superclass:descending_neuron"}
```

```yaml
name: kc-silencing-tmaze
assay: tmaze          # tmaze | looming | sugar
seed: 3000
flies: 6
surgery: {"prefix:KC": -1}
```

Neurons are named by a group (`loom`, `escape`, `head`, `reward`, `sweet`, `dnp01`, `mn9`, `adn`, `jo_ce`, `mn_front`...), `type:A,B`, `prefix:KC`, `superclass:descending_neuron` or `rows:1,2,3`. The full format is in `kickthefly/lab/protocol.py`, and examples are in `protocols/` (bundled in the exe and AppImage: `--protocol smoke.yaml`, `looming-giant-fiber.yaml`, `kc-silencing-tmaze.yaml`, `sugar-dose-response.yaml`, `orchard-feeding.yaml`, and 2.9's `courtship_song.yaml`, `male_aggression.yaml`, `optomotor.yaml`, `grooming_hierarchy.yaml`, `co2_orn_to_pn.yaml`, `bitter_grn_to_dng28.yaml`, `thermosensory_pathways.yaml`, `light_clock_dfb.yaml`). Groups added in 2.9 include `pip10`, `p1`, `ps1`, `co2_orn`, `co2_pn`, `trn_vp2`, `vp2_pn`, `trn_vp3`, `vp3_pn`, `optomotor_right`, `dna_steer_r`/`dna_steer_l`, `groom_anterior` and `groom_posterior`.

```yaml
name: orchard-feeding
assay: orchard        # the Orchard's feeding schedule, headless and reproducible
seed: 4000
flies: 4
assay_options: {feeds: 4, regrow_s: 75, cap: 4, duration_s: 120}   # or params: {orchard.feeds: 4, ...}
```

Run one without a window, from the exe, the AppImage or source:

```bash
./KickTheFly-x86_64.AppImage --headless --protocol protocols/looming-giant-fiber.yaml --out results
python kick_the_fly.py --headless --validate --out validation.json --strict
python kick_the_fly.py --headless --benchmark --backend numba --flies 1 8 16 --seconds 5
```

`--backend NAME` and `--dtype` apply to headless runs too, including the worker processes of `--validate` and the assays, and the backend that ran is recorded in the results. Headless runs need no display (SSH, CI), never open a window or audio device, write their files to `--out` (or the exports folder), and exit 0 on success, 2 for a bad or missing file, and with `--validate --strict` 1 if a validation result differs from the expected one. On Windows use `start /wait KickTheFly.exe ...` from cmd. Runs are seeded and stepped in lockstep, so the same protocol and seed give the same spikes on the same machine.


## The 3.0 day-2 toolkit

Five more Lab screens, each with its own page: [Genetic toolkit](genetics.md) (neurons by driver line), [Thermogenetics](thermogenetics.md)
(TrpA1 / shibire-ts, with the DNp01 escape-vs-temperature assay in Assays), [Patch clamp](patchclamp.md) (a point-neuron model, not
electrophysiology), [Calcium imaging](imaging.md) (the brain view's Imaging mode, MODEL) and [Pharmacology](pharmacology.md) (picrotoxin and
other drugs as synaptic scaling, MODEL PREDICTION). Protocol examples: `genetics_line_silencing`, `thermogenetic_dnp01`,
`thermogenetic_escape_assay`, `patch_dnp01_if_curve`, `imaging_looming`, `drug_picrotoxin`, `drug_cholinergic_block`.
