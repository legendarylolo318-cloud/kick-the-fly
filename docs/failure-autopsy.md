# Failure autopsy (3.1.0)

Tag: **MODEL PREDICTION** for what the run measured; the path numbers are **CONNECTOME**. **Read-only**: it reads the connectome, the simulator's own weights
and a run of the brain. It never changes a weight, a time constant, a threshold or a parameter, and it does not suggest changing one (it reports a diagnosis,
not a fix; `tests/test_failure_autopsy.py` checks the weight matrix is bit-identical afterward and that the code has no write path to the model).

For every validation test that FAILS (`docs/validation.md`), one page: where does the driven signal go, and where does it fade?
Lab > **Failure autopsy** (a list of the failures in the latest validation result, one page each, with a progress bar), `--failure-autopsy [TEST ...]`
(headless; a Markdown page and a JSON file per failure and an `index.md` into `--out`), and `--validate --autopsy` (autopsy every failure right after a validation).

## What a page contains

The test's **input** (the neurons it drives) and **target** (its readout) are fixed by the test. Then:

1. **How much of the target's input the input can supply, hop by hop.** The share of the average target neuron's input synapses that arrives along walks of exactly
   1, 2 and 3 synapses from the input, split into walks that **excite** (an even number of inhibitory synapses on the way) and walks that **inhibit** (odd), with
   the inhibitory share. A weight is the simulator's own (the signed synapse count divided by all of the post neuron's input synapses), so a share is a fraction of
   what the target receives. These are walks, not simple paths: a walk may pass through the input or the target again, as current does in a recurrent network.
2. **The strongest routes, through cell types** (the best of each length of 1, 2 and 3 synapses): the share each carries, how much of it excites and inhibits, and the
   **raw synapse counts** between its stages (total synapses from the types before onto the types in the stage, split into excitatory and inhibitory).
3. **What each stage did in the actual run.** The input, every type on the strongest routes and the target, at rest and while the input is driven (the pathway tests'
   own drive: current 0.5 for 2 s after 2 s of calm), averaged over seeds 1000-1002: Hz before, Hz driven, and the ratio, with the number of neurons. Drawn as bars
   against the test's 1.5x line.
4. **Where it fades.** The first stage along the strongest route whose driven/baseline ratio falls below 1.5x after a stage that met it (`input -> target` when the
   input is driven but nothing downstream follows it), or "the drive never lifted the first stage".

The drive is the pathway tests' drive, not each failing test's own special stimulus (the E-PG wedge, the wind arena): the question is whether the wiring carries a driven
signal to the target. For the tests that are not a plain drive-and-readout pathway the input and target are fixed in `failure_autopsy.special_pathways`, with the reason:
the grooming hierarchy (three pathways: head to front legs, abdomen to hind legs, head to hind legs = the suppression it needs), the E-PG ring (the four-neuron wedge
the test drives to the other 42 E-PG neurons), the E-PG wind test (JO-C/E to E-PG), mushroom body extinction (MBON to the reward PAM neurons: the route an opposing,
extinguishing memory would need) and second-order conditioning (MBON to PPL1 punishment neurons). The two memory pages trace the *wiring those tests need*; they do not
measure the learning rule.

## What it found (run 2026-10-04 on the 3.0 model, the nine adult failures; `docs/results/3.1.0/failure-autopsy/`)

| failing test | where the driven signal fades (strongest route) |
|---|---|
| MDN drives backward walking | MDN is lifted x40, the strongest 2-synapse partner IN03A010 x2.6, the leg motor neurons x0.96. Of the 3-synapse walks reaching the motor neurons, 62% inhibit |
| aDN drives front-leg motor neurons | the input x21 but the motor neurons only x1.12 |
| E-PG compass bump | the driven wedge x10.9 but the rest of the ring **falls to x0.41**: of the wedge's 2-synapse walks onto the ring 71% inhibit (ER4m, ExR6: 19,148 and 5,705 inhibitory synapses onto the ring), so lateral inhibition wins over spread |
| E-PG wind bump | lifted at CB2585 (x4.7), gone at ER1_b (x1.1) |
| grooming hierarchy | head bristles lift DNg15 x4.9 but the front-leg motor neurons stay x1.06; the abdomen route reaches x1.46 (1.5x needed); head stimulation does not lower the hind-leg motor neurons (x0.94: no suppression) |
| P1 drives ps1 | 0 direct synapses; the best route aSP22 lifts x6.0 and reaches ps1 at x1.44 (1.5x needed) through 13 synapses; the 3-synapse walks are 58% inhibitory |
| foreleg GRNs excite P1 | AN09B017f x1.71 falls to P1 x1.26 |
| mushroom body extinction | MBON drive x2.2 but PAM x0.93 |
| second-order conditioning | MBON drive x2.2 but PPL1 x0.96 |

These restate, with the numbers, what the tests' own notes already said (the near balance of excitation and inhibition in the VNC interneuron layer for MDN; the ring's
inhibition). They are what the model does. They are not evidence about the fly, and nothing here says which weight would "fix" a test: a model that was tuned until it passed would
no longer be a test of the connectome.

## Not covered
- **The two larva failures** (`larva_noci_to_goro_rolling`, `larva_chordotonal_to_basin`) are listed with a page saying why: the larva brain has its own groups and an optional pack
  that this machine does not have, so that path could not be tested. Their page is the single line "not autopsied".
- Walks, not simple paths, and cell-type routes, not neuron routes: a route is a sum over every neuron of those types.
- Three seeds for the run; the path numbers do not depend on seeds.
