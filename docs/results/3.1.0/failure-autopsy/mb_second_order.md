# Autopsy: Mushroom body second-order conditioning: odor A + shock, then odor B with odor A

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: After odor A is paired with shock, pairing odor B with odor A, without shock, makes flies avoid odor B; explicitly unpaired presentations don't.
Source: Tabone & de Belle 2011, Learn Mem 18:250

Result: **FAIL**. PI >= 0.5, |unpaired PI| <= 0.25, paired > unpaired (one-sided Wilcoxon p < 0.01)

Measured: pi_mean 5.55e-18, pi_sd 0.194, control_pi_mean -0.03, control_pi_sd 0.211, fear_b_paired 0.0359, fear_b_unpaired 0.0344, p_value 0.125, n 10.

The test's own note: No rule is added: odor B can only gain fear if odor A's learned MBON output drives the dopamine neurons through the connectome while odor B's Kenyon cells are active.

## mushroom body output neurons to the punishment dopamine neurons (odor A's memory must drive PPL1 for odor B to gain fear)

Input: MBON types (97 neurons). Target: PPL1 punishment dopamine neurons (16 neurons). Direct synapses from input to target: 4205.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 2.17% | 1.95% | 0.22% | 47% |
| 2 | 1.36% | 1.69% | -0.323% | 55% |
| 3 | 0.85% | 0.841% | 0.00915% | 50% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> target**: 4.13% of the target's input (2.17% excitatory, 1.95% inhibitory; net excites).
   - into **target**: 4,205 synapses (1,776 excitatory, 2,429 inhibitory); in the run 30.0 Hz at rest, 28.7 Hz driven (x0.96, 16 neurons)
2. **input -> CB1357 -> target**: 0.15% of the target's input (0.0835% excitatory, 0.0668% inhibitory; net excites).
   - into **CB1357**: 975 synapses (522 excitatory, 453 inhibitory); in the run 14.3 Hz at rest, 13.0 Hz driven (x0.91, 11 neurons)
   - into **target**: 533 synapses (533 excitatory, 0 inhibitory); in the run 30.0 Hz at rest, 28.7 Hz driven (x0.96, 16 neurons)
3. **input -> CRE055 -> target**: 0.126% of the target's input (0.0484% excitatory, 0.078% inhibitory; net inhibits).
   - into **CRE055**: 2,061 synapses (1,276 excitatory, 785 inhibitory); in the run 27.0 Hz at rest, 37.2 Hz driven (x1.38, 17 neurons)
   - into **target**: 301 synapses (0 excitatory, 301 inhibitory); in the run 30.0 Hz at rest, 28.7 Hz driven (x0.96, 16 neurons)
4. **input -> APL -> KCg-m -> target**: 0.0173% of the target's input (0.0167% excitatory, 0.000682% inhibitory; net excites).
   - into **APL**: 2,347 synapses (91 excitatory, 2,256 inhibitory); in the run 39.5 Hz at rest, 37.1 Hz driven (x0.94, 2 neurons)
   - into **KCg-m**: 79,151 synapses (0 excitatory, 79,151 inhibitory); in the run 30.4 Hz at rest, 28.7 Hz driven (x0.94, 1342 neurons)
   - into **target**: 23,903 synapses (23,903 excitatory, 0 inhibitory); in the run 30.0 Hz at rest, 28.7 Hz driven (x0.96, 16 neurons)
5. **input -> SMP247 -> SMP115 -> target**: 0.0132% of the target's input (0.00994% excitatory, 0.00323% inhibitory; net excites).
   - into **SMP247**: 451 synapses (84 excitatory, 367 inhibitory); in the run 1.3 Hz at rest, 0.3 Hz driven (x0.25, 9 neurons)
   - into **SMP115**: 982 synapses (982 excitatory, 0 inhibitory); in the run 8.9 Hz at rest, 11.3 Hz driven (x1.27, 2 neurons)
   - into **target**: 727 synapses (0 excitatory, 727 inhibitory); in the run 30.0 Hz at rest, 28.7 Hz driven (x0.96, 16 neurons)

### Where it fades

the signal is lost between input (x2.19) and target (x0.96), below the test's 1.5x.

Driven / baseline along the strongest route: input x2.19, target x0.96.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
