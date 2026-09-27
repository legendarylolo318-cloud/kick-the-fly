# Validation

`kickthefly/lab/validation.py` asks whether this simulation reproduces published fly results, and reports pass or
fail with numbers. Lab > Validation shows the same table in the game, and "Run validation now" reruns it on your PC.
Headless: `python kick_the_fly.py --headless --validate --out validation.json` (`--strict` exits 1 if any result
differs from the expected one). `pytest` runs it too and fails if any result changes in either direction.

Every cell type was checked against the MaleCNS v1.0 annotations, whose synonyms record the published names (DNg62
and DNge078 are "Hampel 2015: aDN1/aDN2", GNG540/GNG550 are "Yao & Scott 2022: Sugar SEL PN", DNg28 is "Yao & Scott
2022: Bitter-SEL", AVLP727m is "Asahina 2014: TK-FruM"). The dataset has no "P1" label: P1 is the clone Cachero 2010
calls pMP-e and Yu 2010 pMP4, and exactly 25 pC1 types (86 neurons) carry that synonym, with no pC1 type mixing
neurons with and without it (`assays.P1_TYPES`).

## Method

Seeds 1000-1009, never used while developing (exploratory probing used seeds 0-299). A set of neurons is driven for
2 s after 2 s of calm, from the same brain snapshot as a matched control set of the same size. **Pass: the
readout's driven/baseline ratio averages at least 1.5x and beats the control's in a one-sided Wilcoxon signed-rank
test, p < 0.01.** For conditioning: PI at least 0.5, the unpaired control's |PI| at most 0.25, and paired above
unpaired (p < 0.01). These thresholds were chosen for this release after exploratory probing, not taken from the
papers: a pass means the sim shows the effect in the stated direction and strength, not that its numbers match the
papers'. With n = 10 the smallest p a one-sided Wilcoxon test can give is 1/1024, shown as p < 0.001.

The grooming hierarchy uses its own criteria, fixed before its run (below). The optomotor test puts a game-rule
transduction (the EMD stage) in front of T4/T5; everything after T4/T5 is the connectome.

## Results

This release, on the NumPy reference backend (n = 10 flies, mean ± SD; Numba and torch-cpu give identical numbers):

| test | readout: drive vs control | result |
|---|---|---|
| Looming detectors LPLC2 + LC4 excite the giant fiber DNp01 ([von Reyn et al. 2014](https://www.nature.com/articles/nn.3741); [Ache et al. 2019](https://www.cell.com/current-biology/fulltext/S0960-9822(19)30138-1)) | DNp01 x11.81 ± 1.77 vs x0.80 ± 0.18 for 311 random visual projection neurons, p < 0.001 | **PASS** |
| MDN activation drives backward walking ([Bidaye et al. 2014](https://pubmed.ncbi.nlm.nih.gov/24700860/)) | leg motor neurons x0.96 ± 0.03 vs x0.96 ± 0.05 for 4 random descending neurons, p = 0.38 | **FAIL: does not reproduce** |
| Sugar-sensing taste neurons activate the proboscis motor neuron MN9; bitter ones don't ([Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9)) | MN9 x2.11 ± 0.40 vs x1.25 ± 0.29 for bitter-pathway neurons, p < 0.001 | **PASS** |
| Antennal mechanosensory neurons JO-C/E excite the antennal grooming neurons aDN1/aDN2 ([Hampel et al. 2015](https://elifesciences.org/articles/08758); Shiu et al. 2024) | aDN1/aDN2 x4.87 ± 1.70 vs x0.85 ± 0.30 for random sensory neurons, p < 0.001 | **PASS** |
| aDN1/aDN2 activation drives antennal grooming, a front-leg movement (Hampel et al. 2015) | front-leg motor neurons x1.13 ± 0.07 vs x0.95 ± 0.06, p < 0.001 | **FAIL: too weak** (consistent, but far below 1.5x) |
| Odor + shock conditioning gives a positive T-maze performance index; unpaired doesn't ([Tully & Quinn 1985](https://pubmed.ncbi.nlm.nih.gov/3939242/)) | PI 1.00 ± 0.00 vs unpaired -0.03 ± 0.15, p < 0.001 | **PASS** (with caveats below) |
| The E-PG ring forms a persistent head-direction bump from a driven wedge ([Seelig & Jayaraman 2015](https://www.nature.com/articles/nature14446)) | EPG peak/trough contrast x1.04 ± 0.14 (3.0x needed), persistence 0 ms (500 ms needed) | **FAIL: no bump** |
| Steady directional wind anchors an E-PG bump that follows the wind ([Okubo et al. 2020](https://www.cell.com/neuron/fulltext/S0896-6273(20)30473-3)); the open field's wind, 8 directions | contrast x1.81 in wind vs x1.71 without (3.0x needed), persistence 101 ms (500 ms needed), direction tracking \|r\| 0.46 vs 0.40 for shuffled directions, p = 0.17 | **FAIL: no bump** |
| The male-specific command neuron pIP10 excites the ps1 wing motor neurons of pulse song ([von Philipsborn et al. 2011](https://doi.org/10.1016/j.neuron.2011.01.011)) | ps1 x1.93 ± 0.41 vs x0.94 ± 0.18 for 2 random descending neurons, p < 0.001 | **PASS** |
| P1 activation excites the ps1 wing motor neurons (von Philipsborn et al. 2011; Kimura et al. 2008) | ps1 x1.43 ± 0.24 vs x0.98 ± 0.17 for 86 random central-brain intrinsic neurons, p < 0.001 | **FAIL: too weak** (consistent, below 1.5x) |
| `bitter_grn_to_dng28`: bitter-pathway taste neurons excite the bitter SEL neuron DNg28 (Shiu et al. 2024; Yao & Scott 2022) | DNg28 x1.58 ± 0.26 vs x1.28 ± 0.21 for sugar-pathway neurons, p < 0.001 | **PASS** (activation, not avoidance) |
| `co2_orn_to_pn`: CO2-sensing ORNs (ORN_V) excite the V glomerulus projection neurons ([Suh et al. 2004](https://doi.org/10.1038/nature02980)) | V PNs x2.21 ± 0.08 vs x0.94 ± 0.03 for 55 random sensory neurons, p < 0.001 | **PASS** (activation, not avoidance) |
| `hot_trn_to_vp2pn`: hot-sensing antennal neurons (TRN_VP2) excite the VP2 projection neurons (Gallio et al. 2011, Cell 144:614; Frank et al. 2015, Nature 519:358) | VP2 PNs x2.00 ± 0.10 vs x0.94 ± 0.04 for 7 random sensory neurons, p < 0.001 | **PASS** (activation, not thermotaxis) |
| `cold_trn_to_vp3pn`: cold-sensing antennal neurons (TRN_VP3a/b) excite the VP3 projection neurons (Gallio et al. 2011; Frank et al. 2015) | VP3 PNs x1.86 ± 0.08 vs x0.96 ± 0.05 for 7 random sensory neurons, p < 0.001 | **PASS** (activation, not thermotaxis) |
| Grooming hierarchy: with head and abdomen stimulated together, the head (front-leg) program wins and the abdomen (hind-leg) program is suppressed ([Seeds et al. 2014](https://elifesciences.org/articles/02951)) | abdomen alone: hind-leg motor neurons x1.46 ± 0.03 (1.5x needed); both: front-leg x1.09 ± 0.06 vs hind-leg x1.44 ± 0.02 (priority p = 1.00), hind-leg not lower than abdomen alone (p = 0.07) | **FAIL: no hierarchy** |
| Optomotor: rightward wide-field rotation (T4a/T5a_R front-to-back, T4b/T5b_L back-to-front) excites the right steering neurons ([Maisak et al. 2013](https://www.nature.com/articles/nature12320); Rayshubskiy et al. 2020) | DNa01_R + DNa02_R x2.72 ± 0.88 vs x0.80 ± 0.32 for as many random optic-lobe intrinsic neurons, p < 0.001; the left pair x1.09 | **PASS** (through a game-rule EMD stage) |

## What the failures and passes mean

- **MDN:** in this sim MDN activity doesn't reach the leg motor neurons. MDN makes 12 synapses onto them directly;
  the 369 neurons it sends at least 5 synapses to (mostly VNC interneurons) send them 27,560 excitatory and 24,186
  inhibitory synapses, and that near balance of excitation and inhibition in the VNC interneuron layer is where the
  drive cancels out. The game's "backs up" reaction reads MDN directly, which is a game rule, and gets no card.
- **aDN to front legs:** the upstream half of the grooming circuit (antennal touch to aDN) reproduces strongly; the
  motor half doesn't. The fly shows no grooming movement, and the GROOM reaction only logs the command neurons.
- **Grooming hierarchy (redesigned in 2.9):** the first version drove aDN and read the hind-leg motor neurons, which
  isn't what Seeds et al. showed. It now stimulates the head's bristle neurons (BM_*, not the taste ones) and the
  abdomen's mechanosensory neurons (subclass "abdomen", minus the chemosensory SNch and proprioceptive SNpp types)
  alone and together, from one snapshot. Criteria, fixed before the run: the abdomen alone raises the hind-leg motor
  neurons at least 1.5x, and with both, front-leg beats hind-leg and hind-leg falls below the abdomen-alone level
  (each one-sided Wilcoxon p < 0.01). Neither half shows: head bristles barely move the front-leg motor neurons, and
  adding them doesn't suppress the abdomen's drive to the hind legs. Front- and hind-leg motor neurons stand in for
  head and abdomen grooming; the sim has no legs to groom with.
- **Courtship song:** pIP10 has no direct synapses onto ps1; its drive goes through VNC interneurons (for example
  pIP10 > DNg74_b > ps1). P1 sends pIP10 1,503 synapses and ps1 none, and its effect on ps1 is consistent but too weak.
  The SONG buzz in the game is synthesized (a game rule).
- **Renamed in 2.9: bitter, CO2, hot and cold.** These were called `bitter_avoidance`, `co2_avoidance`,
  `thermosensory_hot` and `thermosensory_cold`. They test one- or two-synapse activation (sensory neuron to its
  second-order neuron), not avoidance or thermotaxis: nothing in the sim turns DNg28, the V PNs or the VP2/VP3 PNs into
  a behavior. They don't get "Real flies do this too" cards; the game only logs HEAT, COLD and CO2 when the projection
  neurons fire. The bitter set is chosen by its wiring to DNg28 (as the sugar set is by its wiring to the sugar SEL
  PNs), so that test partly restates how the set was picked.
- **Optomotor (fixed in 2.9):** the first version drove T4a/T5a_L + T4b/T5b_R (a *leftward* rotation) while reading
  the right steering neurons, and compared it with visual projection neurons, which reach the DNs directly (x3.37);
  it failed for those reasons, not because of the connectome. It now drives the rightward set through the EMD stage
  (`assays.emd_stage`: a saturating Reichardt-style output on each channel, full motion = the other tests' current)
  and compares with optic-lobe intrinsic neurons like T4/T5 itself, with the same pass criteria. The response is
  lateralized: the left steering neurons barely move (x1.09). The EMD stage replaces the photoreceptor-to-medulla
  motion computation, which the sim doesn't produce from pixels; it is used by validation and protocols, not fed the
  fly's view in the game.
- **Sugar:** the dataset doesn't label taste neurons by taste, so the sugar and bitter sets are chosen from their
  wiring to the Yao & Scott 2022 sugar and bitter neurons (MN9 is never used to choose them). Driving 30 random head
  taste neurons also raises MN9 somewhat.
- **T-maze:** the learning rule, shock driving dopamine neurons and the choice at the T-maze are game rules running on
  the connectome's real synapses; the test shows they give odor-specific memory. The PI of 1.00 is above real flies'
  typical ~0.8-0.9 and not tuned to match. The approach output neurons' overall firing barely differs between the two
  odors (30.9 vs 31.0 spikes/s), so the choice is read from the learned synapses, not from output-neuron firing.
- **E-PG compass, twice:** the first test drives a wedge of EPG neurons directly; the second (2.7) uses the open
  field's steady wind as the cue, through exactly the transduction the arena uses, with the first test's pass
  criteria fixed before the run and no weights or time constants tuned. It reaches the ring only weakly (EPG firing
  actually drops, 6.8 to 5.5 spikes/s) and forms no bump. No compass HUD ships.

## Checked in the 2.9 review

The science added in 2.9 was reviewed before release: each number above is from a rerun on the held-out seeds, not
from the branch's notes (most of which didn't match a rerun of its own code). The review also found no weights,
thresholds or time constants changed to make a test pass; the pass criteria are unchanged. New thresholds that turn
firing into game reactions (song, lunge, sleep, CO2/heat/cold logs) are game rules in Lab > Parameters, set above
each group's calm maximum on exploration seeds.
