# Fly Individuality (Idiosyncratic Connectome Gains)

Kick the Fly 2.11 models biological individuality in *Drosophila* behavior ([Kain et al. 2012](https://doi.org/10.1016/j.cub.2012.08.027); [Linneweber et al. 2020](https://doi.org/10.1126/science.aaw7182)). No two flies have identical brains, even when sharing identical nominal wiring diagrams.

---

## 1. Mathematical Formulation & Architecture

To support large-scale multi-fly simulation and streaming GPU acceleration, per-fly variation is factored into diagonal scaling operators:

$$W_{\text{fly}} = D_{\text{post}} \cdot W \cdot D_{\text{pre}}$$

where:
- $W$ is the nominal, unperturbed connectome synaptic weight matrix.
- $D_{\text{pre}} = \text{diag}(d_{\text{pre}, 1}, \dots, d_{\text{pre}, N})$ and $D_{\text{post}} = \text{diag}(d_{\text{post}, 1}, \dots, d_{\text{post}, N})$ are diagonal matrices of per-neuron presynaptic excitability and postsynaptic receptor sensitivity gains.
- The gains are drawn deterministically from a log-normal distribution:
  $$\ln(d) \sim \mathcal{N}(0, \sigma^2)$$
  clamped to $[-3\sigma, +3\sigma]$ to prevent non-biological runaway excitation or silencing.
- **Signs are strictly preserved**: because $d > 0$ for all neurons, excitatory connections remain strictly excitatory and inhibitory connections remain strictly inhibitory.
- **Plasticity Interaction**: Mushroom body Kenyon cell $\to$ MBON plastic weights still update dynamically on top of individuality gains.

### Streaming GPU Multi-Fly Invariant
Because $W$ remains invariant across all individuals, batched PyTorch execution streams **one shared weight matrix** per step:
1. Multiply the per-fly spike vector by $D_{\text{pre}}$.
2. Execute batched SpMM against the shared matrix $W$.
3. Multiply the resulting post-synaptic current vector by $D_{\text{post}}$.

Backend support, measured in the 2.11 review (`tests/test_individuality.py`):
- **NumPy and Numba**: bit-exact (every spike and voltage) with individuality off, subtle and strong.
- **torch-cpu**: bit-exact with individuality off. With it on, the scaled products are summed in a different order, the
  float32 rounding differs and the chaotic network drifts apart (after 150 steps at `subtle`, spikes differ and
  voltages by up to 6.5). Not bit-exact.
- **gl** (OpenGL compute): **not implemented**. The shaders use the shared $W$ only; a gl brain logs a warning and runs
  with individuality off (its gains are cleared so nothing reports gains it does not use).
- torch-cuda / torch-rocm: implemented as above, untested in 2.11 (no such device in the review).

---

## 2. Configuration & Modes

- **Settings**: `Settings > Brain > Individuality`:
  - `off`: $\sigma = 0.0$ (identical clone brains).
  - `subtle`: $\sigma = 0.05$ (the setting's default).
  - `strong`: $\sigma = 0.15$.
  - Both are GAME RULE and adjustable in `Lab > Parameters` (`Individuality: subtle/strong sigma`).
  - Configs from before 2.11 get the default (`subtle`) on upgrade, so existing players' flies change slightly unless
    they set it to `off`.
- **Validation Invariant**: Individuality is **strictly forced OFF** during `--validate` and within the automated validation suite to preserve the reproducibility of published benchmark results.
- **Metadata**: recorded in save files (`.ktfsave`). Not recorded in NWB exports or replays in 2.11.

---

## 3. Repeated-Assay Consistency & Intraclass Correlation (ICC)

To evaluate whether flies show individual consistency across sessions ([Kain et al. 2012](https://doi.org/10.1016/j.cub.2012.08.027); [Linneweber et al. 2020](https://doi.org/10.1126/science.aaw7182)), the Lab assay runs repeated sessions across multiple assays (looming escape latency, sugar feeding response, steering bias via DNa01/02, and T-maze memory).

### Gemini's results ($n = 8$ flies, $k = 4$ sessions; not re-run in the 2.11 review)

These were run with $\sigma = 0.15$: a bug in `simcore.new_brain` then forced $\sigma = 0.15$ for every setting other than
`off`, so they correspond to today's `strong`, not `subtle`.

1. **Self-Consistency vs Population Variance Test**:
   - Within-individual Euclidean distance across sessions: **2.675**.
   - Between-individual Euclidean distance: **2.649**.
   - One-sided Wilcoxon signed-rank test ($H_1: \text{within} < \text{between}$): **$p = 0.4375$**.
   - Result: **FAIL**. While individual flies show unique drift trajectories, baseline membrane noise across repeated sessions produces variance comparable to the subtle individuality gain parameter. Reported honestly without post-hoc tuning.

2. **Intraclass Correlation Coefficient (ICC(1,1))**:
   - **Steering Bias (DNa01/02 R/L ratio)**: Experimental ICC = **+0.097** (Control ICC = -0.293).
   - **Looming Latency**: Experimental ICC = **+0.068** (Control ICC = +0.174).
   - **Sugar Feeding (MN9 activation)**: Experimental ICC = **-0.011** (Control ICC = -0.254).
   - **T-maze Conditioning PI**: Experimental ICC = **-0.098** (Control ICC = -0.038).

---

## 4. Adult Validated Behaviors Pass Rates

Pass rates evaluated across held-out seeds (1000–1009) at each individuality setting:

| Behavior Test | Mode: `off` | Mode: `subtle` ($\sigma=0.15$) | Mode: `strong` ($\sigma=0.30$) |
|---|---|---|---|
| **Looming Escape $\to$ GF DNp01** | **100.0%** | **100.0%** | **100.0%** |
| **Sugar Feeding $\to$ MN9** | **70.0%** | **80.0%** | **80.0%** |
| **Antennal Touch JO $\to$ aDN** | **70.0%** | **30.0%** | **30.0%** |
| **T-maze Odor Conditioning** | **100.0%** | **100.0%** | **100.0%** |

*Findings*: Robust pathways like looming escape and associative olfactory memory retain 100% pass rates under all individuality settings. Narrow threshold pathways like JO $\to$ aDN show sensitivity to synaptic gain variation, dropping from 70% to 30%.

---

## 5. Measured Personality Cards

**Known issue (2.11 review): the cards are not measured.** In the game (`FlySlot`) and in Pet mode the card is built by
`compute_personality_card(seed)` without assay results, and then its metrics are seeded random draws, independent of the
fly's actual gains and of the individuality setting (flies get cards even with individuality off). Only a caller that
passes measured metrics gets a measured card. The thresholds the card applies (`core/individuality.py`, shown on the card):
- **Temperament**:
  - *Bold*: Looming escape latency $\ge 0.30$ s.
  - *Skittish*: Looming escape latency $\le 0.20$ s.
  - *Alert*: Typical response latency.
- **Steering Bias**:
  - *Right-turner*: DNa01/02 R/L firing ratio $\ge 1.12$.
  - *Left-turner*: DNa01/02 R/L firing ratio $\le 0.89$.
  - *Straight-walker*: Balanced steering.
- **Feeding Drive**:
  - *Sugar lover*: MN9 proboscis response $\ge 2.20$x baseline.
  - *Finicky eater*: MN9 proboscis response $\le 1.50$x baseline.
  - *Steady feeder*: Typical response.
- **Learning Ability**:
  - *Quick learner*: T-maze Performance Index $\ge 0.60$.
  - *Stubborn*: T-maze Performance Index $\le 0.30$.
  - *Moderate learner*: Typical memory retention.

The personality profile is displayed in the fly focus cycling notification (**F**), in the neuron inspector card header, and in the Pet Mode HUD widget.
