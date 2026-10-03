# Fly racing (Esc > Fly arcade > Racing, `--race`) — 3.0 day 4

Flies race down a track with **sugar and fruit lures** along it, each through its **own brain and individuality**. You can bet **in-game
points** on a fly (never money: there is nothing to buy and no path to a payment of any kind); the odds come from the flies' measured
personality stats. Races can be replayed, and a headless **race assay** reports whether individuality predicts the finishing order.

| | what it is |
|---|---|
| **CONNECTOME** | the walking command neurons' firing (DNp09, "P9") sets the speed; a lure ahead drives the real olfactory neurons of its scent (the game's own scent pokes) and touching one drives the sugar-pathway taste neurons and the PAM reward neurons exactly as sugar and ripe fruit do in the game |
| **GAME RULE** | the 8 m track, the three lures (sugar at 2 m, fruit at 4 m, sugar at 6 m), the smell range (1.5 m) and contact distance (0.15 m), the pulse strengths (the game's: scent 0.3, taste 0.5, reward 0.5), the speed rule, the 90 s cap, the lanes (flies never touch or block each other, so each lane is its own simulation), the odds and the points |
| **MODEL PREDICTION** | the finishing order, and whether individuality predicts it |

**Speed rule (GAME RULE, version 2 since the 3.0 day 4 review):** speed = 0.6 m/s x clip(DNp09 level / the game's walking threshold, 0, 1),
where the level is DNp09's firing divided by **one fixed reference rate, 4.32 Hz, the same for every fly**, and the threshold is the game's
`walk` threshold (3.0x). Version 1 divided by each fly's own calm baseline, which the brain sets during its 3 s warm-up from its seed's noise
(1.75-6.43 Hz between seeds): that made a fly's speed mostly an accident of its warm-up (below). 4.32 Hz is the mean of that baseline over
exploration seeds 0-15 (individuality off), fixed before any held-out run of version 2. Nothing is tuned to make races close or orders
predictable. The card's "calm walking drive" (half of the odds' form score) is read against the same reference (card version 2).

## The odds, and the points

Before a race each fly's card is measured (the same measured card as the tournament, plus its calm walking drive). **Form** is the mean of
two z-scores inside the field: the sugar -> MN9 ratio (how hard the lures' taste drives it) and the calm walking drive (which sets the
speed). P(win) = softmax(form), and a bet pays stake x (0.9 / P(win)), rounded to a tenth and at least 1.1. All of this is a **GAME RULE**: a
modelling choice, not a finding. The race assay measures whether it works.

Points live in `arcade_points.json` in your data folder: you start with 100, stakes are 5, 10 or 25, and below 5 points you are topped up
to 25 for free. They cannot be bought, sold or cashed out. There is no telemetry and no network use anywhere in the arcade.

## Using it

- **Esc > Fly arcade > Racing:** choose 3 to 8 lanes, **Look at the field** (measures the cards, a few seconds each, in the background),
  read the odds, click a fly and a stake, **Start race**. The replay animates the lanes; the result and your points are shown. Palettes, larger
  text and the gamepad apply (3D: bumpers pick the fly to bet on, kill cam / big-view change the lanes, the trigger looks at the field and then starts
  the race, the Neurodex button switches to Tournament, B or Start closes the page); nothing flashes.
- **Headless race assay:** `python kick_the_fly.py --headless --race --seeds 1000-1047 --lanes 6 --races 8 --workers 4 [--individuality off] --out DIR`.
  Each fly runs the same race twice (new noise each time), so a fly's finishing time can be compared with itself.
- **Python:** `from kickthefly.lab.api import race; r = race(range(3000, 3006), repeats=2)`.

## Does individuality predict the finish? (pre-registered; written in `lab/racing.py` before any run)

Primary family, Holm-corrected, alpha 0.05: **R1 repeatability** (Spearman correlation of a fly's finishing time between its first and
second run, across all flies, one-sided), **R2 form** (the form score against the mean finishing time, one-sided), **R3 the odds** (the Brier
score of the win probabilities beats a uniform field's, per race, one-sided Wilcoxon over the races; a floor of p < 0.001 is reported as
that). Secondary, uncorrected: each card trait against the mean finishing time. Run `--individuality off` as the control: identical
brains should show no repeatability.

## Result with speed rule version 1 (Sonnet's run; 8 races of 6 flies, seeds 1000-1047, each fly run twice, NumPy CPU)

| | individuality `subtle` | `off` (control: identical brains) |
|---|---|---|
| **R1** repeatability (finishing time, run 1 vs run 2) | rho = 0.96, Holm p < 0.001 | **rho = 0.92, Holm p < 0.001** |
| **R2** form score vs finishing time | rho = -0.54, Holm p < 0.001 | rho = -0.53, Holm p < 0.001 |
| **R3** odds beat a uniform field (Brier, 8 races) | mean difference -0.252, Holm p = 0.008 | -0.274, Holm p = 0.039 |
| (uncorrected) sugar -> MN9 ratio | rho = -0.09, p = 0.53 | -0.14, p = 0.35 |
| (uncorrected) calm walking drive | rho = -0.79, p < 0.001 | -0.77, p < 0.001 |
| (uncorrected) looming latency / steering ratio | 0.32 (p = 0.027) / 0.17 (p = 0.26) | 0.11 / 0.11 (n.s.) |

**All three pre-registered tests pass, but the control says they are not about individuality.** With identical brains (`off`) a fly's finishing time is
just as repeatable (rho 0.92 against 0.96). What persists from run to run is **the state each seed's warm-up leaves the brain in** (its settled gain and
calm firing rates), not the per-neuron individuality gains: the `subtle` gains add little that this assay can see. The honest reading is "a fly's
speed is a stable property of its seed's brain state", not "individuality predicts the finish". The form score works (R2, R3) because half of it, the
calm DNp09 level, is the quantity the speed rule uses: that is the model doing what its rule says (rho = -0.79 for that term alone), while the sugar ratio
carries no information about speed (rho = -0.09). No parameter was changed after seeing this.

## Why R1 does not test individuality, and R4 (3.0 day 4 review)

**The cause (exploration seeds 0-15, individuality off, each fly run twice; `tools/race_diagnosis.py`, `docs/results/day4/race_diagnosis/`):**

| condition | run 1 vs run 2 Spearman rho |
|---|---|
| A: as the assay does it (both runs start from the seed's post-warm-up state) | **+0.98** (p = 1e-10) |
| B: a fresh warm-up per run, under the lane's own noise | +0.55 (p = 0.027) |
| C: as A, the synaptic gain (the slow controller's one number) set to the mean over seeds | +0.94 |
| D: as A, the calm baselines (`Brain.base`, the rates every level is read against) set to the mean over seeds | **-0.08** (p = 0.77) |

The post-warm-up calm baseline of the walking group predicts a fly's finishing time almost perfectly (rho +0.98); its gain only weakly
(-0.51). The speed rule (a GAME RULE) reads DNp09 firing **as a multiple of the fly's own calm rate**, and that calm rate is estimated
during the 3 s warm-up from the seed's noise and then carried through the race. A fly whose warm-up happened to set a high baseline reads a
low walking level all race long, so it is slow in both runs, individuality or not. Equalising the baselines removes the repeatability (D);
equalising the gain does not (C). B is lower but not zero; with n = 16 that may be chance, and it is **not resolved** here. No rule was
changed: this explains R1, it does not alter it.

**R4, pre-registered in `lab/racing.py` before it was run** (committed 5c0abd2 before any held-out lane): every lane of a race shares one
brain-state seed, so the baseline artefact is the same for all six lanes; only the individuality seed differs. Statistic: within-race rank
repeatability r_w (run 1 vs run 2), against the `off` control (six identical brains in one state). PASS needs a within-race permutation
p < 0.05 for `subtle` AND a 95% bootstrap CI of r_w(subtle) - r_w(off) above 0.
`python kick_the_fly.py --headless --race-r4 --workers 3 --out DIR`

## Result with speed rule version 2 (3.0 day 4 review, approved follow-up; seeds 1000-1047, 8 races of 6, each fly run twice, NumPy CPU)

Same pre-registered R1-R3 criteria as above, unchanged; the rule was changed and committed (15d45fa) before this run.

| | individuality `subtle` | `off` (control: identical brains) |
|---|---|---|
| **R1** repeatability (finishing time, run 1 vs run 2) | **rho = 0.77, Holm p < 0.001** | **rho = 0.09, Holm p = 0.85** |
| **R2** form score vs finishing time | rho = -0.09, Holm p = 0.53 (FAIL) | rho = 0.20, Holm p = 1.0 |
| **R3** odds beat a uniform field (Brier, 8 races) | mean difference -0.006, Holm p = 0.58 (FAIL) | +0.038, Holm p = 1.0 |
| (uncorrected) calm walking drive / sugar ratio | rho = -0.19 (p = 0.18) / 0.00 | 0.00 / 0.26 |

**With the warm-up artefact gone, R1 does what it was meant to:** individual flies keep their speed from run to run (0.77) and identical
brains do not (0.09). This agrees with R4. **R2 and R3 now FAIL:** the odds no longer predict the finish. Under version 1 they "worked"
only because half of the form score was the same warm-up baseline the speed rule divided by. Neither of the card's measured traits
predicts a fly's speed under version 2. So the odds are a game, not a forecast. Files: `docs/results/day4/race_v2/`, `race_v2_off/`.
