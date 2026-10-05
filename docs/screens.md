# Whole-brain screens (3.1.0): the activation screen and the knockout screen

Tag: **MODEL PREDICTION**, on the Lab pages, in `meta.json` and in every output file's folder. These screens ask what this connectome *simulation* does, with
the model's known failures (`validation.md`); they do not say what a fly does. Lab > **Activation screen** and Lab > **Knockout screen**; headless
`--activation-screen` and `--knockout-screen`; or a `screen:` block in a protocol. All three run the same code (`kickthefly/lab/screens.py`).

## The activation screen

*What does each cell type make the fly do?* For every cell type (11,751 in the adult pack; a subset with `--types`, `--min-neurons`, `--max-types`) the
type is held driven (the optogenetic-style current every pathway test uses, 0.5, for 2 s) on each held-out seed, and the screen records

- **every descending neuron type's** change in firing (one number per DN type, a few hundred to a thousand a seed), and
- ten **behavior readouts**, the validated behaviors' own neuron sets: the giant fiber DNp01 (escape), DNp09 (forward walking command), MDN (backward
  walking), aDN1/aDN2 (antennal grooming), the proboscis motor neuron MN9, the leg motor neurons, pIP10 and the ps1 wing motor neurons (courtship song),
  and the right- and left-steering DNa01/DNa02.

### How it is controlled
- **Held-out seeds 4000-4007.** They were never used while building or tuning anything (validation uses 1000-1009).
- **Same start for every condition.** A seed gives one warmed-up brain; its state after 400 calm steps is snapshotted and every condition for that seed
  starts from that snapshot (what validation does), so a condition differs from the unperturbed run only by the perturbation, noise included. The
  unperturbed run per seed is the baseline.
- **Matched controls.** For each type and seed, `--screen-controls` (default 2) random neuron sets of the *same size and the same count from each
  superclass*, never overlapping the type or any readout, are driven the same way. A type's effect on a readout is its change **minus its matched controls'
  change**, seed by seed. A type that does no more than a random set of its size is not an effect.
- **Statistics.** A paired t-test over seeds, then Benjamini-Hochberg over every type x readout in the screen (q). The Wilcoxon signed-rank test the Lab
  uses elsewhere cannot get below p = 0.0078 with 8 seeds, so no correction over a whole-brain screen (more than 100,000 tests) could ever pass it; the
  t-test has no such floor, and a share of at least 75% of the seeds must move the same way so one wild seed cannot carry a call.
- **A behavior call** is made only when q < 0.05, the change is at least 1 Hz, and the seeds agree. The call is the readout with the largest standardized
  effect ("raises escape (giant fiber)", "lowers antennal grooming command"). Otherwise "none".
- **Trivial effects are not counted.** Driving a readout's own neurons (DNp01 for the escape readout, MN9 for proboscis, ...) is excluded for that
  readout (the column `self_readouts` says which); a descending type driven directly is not listed among its own responders.
- **DN responders:** a descending type responds when its change over its controls is at least 4 noise standard errors (noise = the spread of all matched
  controls' changes in that DN type, over the screen) and at least 0.5 Hz.

### Output (a folder)
`meta.json` (settings, engines that ran, what was written), `records.jsonl` (raw per-seed records), `activation_screen.csv` and `.parquet` (one row per
type: neuron count, superclass, per-readout effect / CI / p / q / sign share, the call, DN responder count and the top five), 
`activation_screen_responses.csv` and `.parquet` (each type's descending-neuron responders: DN type, change, z), `activation_screen.json` (what the Lab
page reads). Parquet needs `pyarrow` (optional; without it the CSVs are the output and `meta.json` says so).

### Reading one: an example run
`--types LPLC2 LC4 LC6 LC10a Mi1 pIP10 ...` (16 types, 8 seeds, 2 controls; 52 s on 8 CPU processes) called LC4 and LPLC2 as raising the **escape** readout
(+33.5 and +27.2 Hz, q < 1e-8), which is the validated looming pathway recovered from a blind start, LC6 (+5.7 Hz, the other known giant-fiber input) and
pIP10 as raising the **courtship song motor** readout (+5.2 Hz, q 4e-6, the validated pIP10 pathway). It also called Mi1 (an optic-lobe cell type with 1,773
neurons) as raising escape (+9.7 Hz) and KCg-m as lowering backward walking: these are what the model does, not claims about flies, and they are the kind of
unexpected row the screen exists to surface and a human to check.

## The knockout screen

*What is each validated behavior built on?* For each of the **eleven pathway behaviors that pass** validation (looming escape, sugar to MN9, antennal
touch to the grooming neurons, pIP10 to song, bitter taste to DNg28, CO2, hot and cold pathways, optomotor turning, cVA to DA1, DA1 to the lateral horn;
`screens.VALIDATED_PATHWAYS`; a behavior the model does not reproduce is refused, because silencing something for a response that is not there only
measures noise) the screen silences candidate cell types and ranks them by how much the **response to the drive** (driven minus baseline firing of the
readout, validation's own test) drops.

- **Candidates:** the types that can reach the readout in one or two synaptic hops (the critical path finder's shortlist, `--top N`, default 25), or the
  types you name with `--types`. A type outside the shortlist was never tried: say so when you report a result.
- **Ranked batches:** `--candidate-batch B` silences the ranked candidates B at a time; a batch that cuts the response by at least 10% (q < 0.1) is
  opened and its members are tested one by one, the rest are reported as batch-neutral. Batches can miss redundant types (two types that each cover for
  the other look neutral alone and strong together, and the reverse); the table marks which rows are batches.
- **Same design as the activation screen:** the held-out seeds, the shared snapshot, the unperturbed run, matched random lesions (same size and superclass
  mix, never the drive or the readout), a paired t-test over seeds, Benjamini-Hochberg q over the screen and the 75% agreement rule.
- **Output:** `knockout_screen.csv` / `.parquet` (behavior, candidate, whether it is a batch, neurons, the unperturbed, lesioned and matched-control response
  in Hz, the drop in Hz with its CI, as a share, p, q, rank), `knockout_screen.json`, `candidates.json`, `records.jsonl`, `meta.json`.
- On the first check, looming escape ranks DNp01 (-100%, the readout's own neurons), LC4 (-54%) and LPLC2 (-39%), with the batch of all three at -99% and
  a batch of LC6, LC10a and LC10b at 0%: the same picture as the critical path finder's single-target numbers in `lab.md` (LC4 -50%, LPLC2 -43%).

## Running them

```bash
python kick_the_fly.py --headless --activation-screen --out screen-act                       # all 11,751 types x 8 seeds x (1 + 2 controls): hours
python kick_the_fly.py --headless --activation-screen --types LC4 LPLC2 LC6 --out screen-act # a few types: a minute
python kick_the_fly.py --headless --activation-screen --max-types 200 --min-neurons 5 --workers 16
python kick_the_fly.py --headless --knockout-screen                                          # the eleven validated behaviors, top 25 candidates each
python kick_the_fly.py --headless --knockout-screen looming_escape sugar_feeding --top 40 --candidate-batch 5
python kick_the_fly.py --headless --activation-screen --backend gl --screen-batch 8          # eight brains on one GPU, stepped together
```

- **Resumable.** Each finished (type, seed) is appended to `records.jsonl` as it completes. Run the same command into the same `--out` again and it
  continues; a run in a folder that holds different settings is refused (`--restart` discards it). A line cut short by an interruption is ignored.
- **Progress** is a bar on stderr (`[#####.....] 52/260  20%  4m elapsed, ~16m left`), and the Lab pages show the same bar with a Stop button (everything
  finished is kept).
- **Parallel:** CPU processes by default (`--workers`); `--screen-batch K` runs K brains on threads of one process and, on the `gl` backend, the GPU steps
  them together. A GPU engine agrees with NumPy statistically, not spike for spike; the engine is recorded in `meta.json` and every comparison is within one
  engine.
- **Cost:** one activation window is 400 brain steps (about 0.5 s on one CPU core); a type costs (1 + controls) windows per seed. A protocol runs the same
  thing: `screen: {kind: activation, types: [LC4, LPLC2], controls: 2}` (or `kind: knockout, behaviors: [...], top: 25, candidate_batch: 5`), with the
  protocol's `seeds` (default 4000-4007).

## The Lab pages
A progress bar and Run / Stop; a search box (cell type, what it does, DN names); sortable columns (click a header); rows colored by direction; click a
row for its **detail panel**: the numbers with their confidence intervals and q, the descending neurons it moved, and two buttons: **Activate (or
Silence) on the live flies** (the brain surgery switch, so you can watch the row in the game) and **Inspect a neuron of this type** (closes the menu and opens
the big brain view's neuron inspector on one of its neurons). Pages load the last finished run from `exports/activation-screen` or
`exports/knockout-screen` and Export copies the files to a new folder.

## What was and was not verified
Verified: the machinery, on the synthetic pack (`tests/test_screens.py`: controls matched in size and superclass, the call rules, resuming, interrupted and
extended runs, determinism on the CPU, the statistics, the batch logic, parquet, the protocol, the command line, the Lab pages), and the two example runs
above on the real pack. **Not** run: the whole-connectome screens (11,751 types; hours to days of CPU), so no whole-brain result is claimed here. Parquet
was written and read back with the installed pyarrow; the no-pyarrow path is tested by hiding the import.
