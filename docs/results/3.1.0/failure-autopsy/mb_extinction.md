# Autopsy: Mushroom body extinction: unreinforced odor exposure reduces learned avoidance

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: After aversive conditioning, repeated exposure to the trained odor without shock lowers the T-maze avoidance of it.
Source: Felsenberg et al. 2018, Cell 175:709

Result: **FAIL**. extinguished PI < 0.5 and unextinguished > extinguished (one-sided Wilcoxon p < 0.01)

Measured: extinguished_pi_mean 1, extinguished_pi_sd 0, unextinguished_pi_mean 1, unextinguished_pi_sd 0, fear_extinguished 0.62, fear_unextinguished 0.667, p_value 1, n 10.

The test's own note: No rule is added for extinction: the existing learning rule (and its reversal term) is all there is, so the brain would have to produce it through its own dopamine neurons. In real flies extinction is a parallel opposing memory formed through reward dopamine neurons, with the original memory kept (Felsenberg et al. 2018).

## mushroom body output neurons to the reward dopamine neurons (the route an opposing, extinguishing memory needs)

Input: MBON types (97 neurons). Target: PAM reward dopamine neurons (316 neurons). Direct synapses from input to target: 4879.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 1.54% | 2.94% | -1.4% | 66% |
| 2 | 1.17% | 1.54% | -0.375% | 57% |
| 3 | 0.726% | 0.675% | 0.0509% | 48% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> target**: 4.48% of the target's input (1.54% excitatory, 2.94% inhibitory; net inhibits).
   - into **target**: 4,879 synapses (1,449 excitatory, 3,430 inhibitory); in the run 31.7 Hz at rest, 29.5 Hz driven (x0.93, 316 neurons)
2. **input -> LHAV9a1_b -> target**: 0.115% of the target's input (0.0516% excitatory, 0.0636% inhibitory; net inhibits).
   - into **LHAV9a1_b**: 483 synapses (222 excitatory, 261 inhibitory); in the run 5.9 Hz at rest, 4.6 Hz driven (x0.78, 6 neurons)
   - into **target**: 276 synapses (276 excitatory, 0 inhibitory); in the run 31.7 Hz at rest, 29.5 Hz driven (x0.93, 316 neurons)
3. **input -> CB1357 -> target**: 0.106% of the target's input (0.0567% excitatory, 0.0498% inhibitory; net excites).
   - into **CB1357**: 975 synapses (522 excitatory, 453 inhibitory); in the run 14.3 Hz at rest, 13.0 Hz driven (x0.91, 11 neurons)
   - into **target**: 303 synapses (303 excitatory, 0 inhibitory); in the run 31.7 Hz at rest, 29.5 Hz driven (x0.93, 316 neurons)
4. **input -> APL -> KCg-m -> target**: 0.0262% of the target's input (0.0252% excitatory, 0.00103% inhibitory; net excites).
   - into **APL**: 2,347 synapses (91 excitatory, 2,256 inhibitory); in the run 39.5 Hz at rest, 37.1 Hz driven (x0.94, 2 neurons)
   - into **KCg-m**: 79,151 synapses (0 excitatory, 79,151 inhibitory); in the run 30.4 Hz at rest, 28.7 Hz driven (x0.94, 1342 neurons)
   - into **target**: 19,914 synapses (19,914 excitatory, 0 inhibitory); in the run 31.7 Hz at rest, 29.5 Hz driven (x0.93, 316 neurons)
5. **input -> KCg-m -> KCg-m -> target**: 0.0167% of the target's input (0.000791% excitatory, 0.0159% inhibitory; net inhibits).
   - into **KCg-m**: 1,793 synapses (86 excitatory, 1,707 inhibitory); in the run 30.4 Hz at rest, 28.7 Hz driven (x0.94, 1342 neurons)
   - into **KCg-m**: 127,343 synapses (127,343 excitatory, 0 inhibitory); in the run 30.4 Hz at rest, 28.7 Hz driven (x0.94, 1342 neurons)
   - into **target**: 19,914 synapses (19,914 excitatory, 0 inhibitory); in the run 31.7 Hz at rest, 29.5 Hz driven (x0.93, 316 neurons)

### Where it fades

the signal is lost between input (x2.19) and target (x0.93), below the test's 1.5x.

Driven / baseline along the strongest route: input x2.19, target x0.93.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
