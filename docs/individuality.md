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
Because $W$ remains invariant across all individuals, batched GPU execution (PyTorch batched SpMM and OpenGL compute shaders) streams **one shared weight matrix** per step:
1. Multiply the per-fly spike vector by $D_{\text{pre}}$.
2. Execute batched SpMM against the shared matrix $W$.
3. Multiply the resulting post-synaptic current vector by $D_{\text{post}}$.

NumPy, Numba, and torch-cpu remain **100% bit-exact** with individuality enabled.

---

## 2. Configuration & Modes

- **Settings**: `Settings > Brain > Individuality`:
  - `off`: $\sigma = 0.0$ (identical clone brains).
  - `subtle`: $\sigma = 0.15$ (default in Play & Pet modes).
  - `strong`: $\sigma = 0.30$ (amplified individual behavioral divergence).
- **Validation Invariant**: Individuality is **strictly forced OFF** during `--validate` and within the automated validation suite to preserve the reproducibility of published benchmark results.
- **Metadata**: Recorded in save files (`.ktfsave`), NWB exports, and replays.

---

## 3. Repeated-Assay Consistency & Intraclass Correlation (ICC)

To evaluate whether flies show individual consistency across sessions ([Kain et al. 2012](https://doi.org/10.1016/j.cub.2012.08.027); [Linneweber et al. 2020](https://doi.org/10.1126/science.aaw7182)), the Lab assay runs repeated sessions across multiple assays (looming escape latency, sugar feeding response, steering bias via DNa01/02, and T-maze memory).

### Empirical Results (Mode: `subtle`, $n = 8$ flies, $k = 4$ sessions):

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

Each fly receives a non-scripted personality profile card calculated deterministically from its unique individual metrics:
- **Temperament**:
  - *Bold*: Looming escape latency $\ge 0.35$ s.
  - *Skittish*: Looming escape latency $\le 0.18$ s.
  - *Alert*: Typical response latency.
- **Steering Bias**:
  - *Right-turner*: DNa01/02 R/L firing ratio $\ge 1.35$.
  - *Left-turner*: DNa01/02 R/L firing ratio $\le 0.74$.
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
