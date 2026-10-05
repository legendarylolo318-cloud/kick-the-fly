# Autopsy: aDN activation drives front-leg (grooming) motor neurons

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: Activating aDN1/aDN2 drives antennal grooming, a front-leg movement.
Source: Hampel et al. 2015, eLife 4:e08758

Result: **FAIL**. drive ratio >= 1.5 and > control (one-sided Wilcoxon p < 0.01)

Measured: drive_ratio_mean 1.13, drive_ratio_sd 0.0674, control_ratio_mean 0.948, control_ratio_sd 0.0554, readout_base_hz 3.37, readout_driven_hz 3.8, p_value 0.000977, n 10.

The test's own note: The sim has no leg movement; this asks whether aDN activity reaches the front-leg motor neurons.

## Activating aDN1/aDN2 drives antennal grooming, a front-leg movement.

Input: aDN1/aDN2 (4) (4 neurons). Target: front-leg motor neurons (135) (135 neurons). Direct synapses from input to target: 907.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0.26% | 0% | 0.26% | 0% |
| 2 | 0.111% | 0.085% | 0.0258% | 43% |
| 3 | 0.127% | 0.178% | -0.0506% | 58% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> target**: 0.26% of the target's input (0.26% excitatory, 0% inhibitory; net excites).
   - into **target**: 907 synapses (907 excitatory, 0 inhibitory); in the run 3.4 Hz at rest, 3.8 Hz driven (x1.12, 135 neurons)
2. **input -> DNge012 -> target**: 0.0237% of the target's input (0.0237% excitatory, 0% inhibitory; net excites).
   - into **DNge012**: 773 synapses (773 excitatory, 0 inhibitory); in the run 6.0 Hz at rest, 30.5 Hz driven (x5.08, 2 neurons)
   - into **target**: 503 synapses (503 excitatory, 0 inhibitory); in the run 3.4 Hz at rest, 3.8 Hz driven (x1.12, 135 neurons)
3. **input -> IN13A035 -> target**: 0.0229% of the target's input (0% excitatory, 0.0229% inhibitory; net inhibits).
   - into **IN13A035**: 352 synapses (352 excitatory, 0 inhibitory); in the run 7.7 Hz at rest, 22.3 Hz driven (x2.89, 10 neurons)
   - into **target**: 1,817 synapses (0 excitatory, 1,817 inhibitory); in the run 3.4 Hz at rest, 3.8 Hz driven (x1.12, 135 neurons)
4. **input -> DNge022 -> IN16B070 -> target**: 0.0155% of the target's input (0% excitatory, 0.0155% inhibitory; net inhibits).
   - into **DNge022**: 953 synapses (953 excitatory, 0 inhibitory); in the run 5.5 Hz at rest, 41.6 Hz driven (x7.56, 2 neurons)
   - into **IN16B070**: 219 synapses (219 excitatory, 0 inhibitory); in the run 8.2 Hz at rest, 28.7 Hz driven (x3.49, 6 neurons)
   - into **target**: 649 synapses (0 excitatory, 649 inhibitory); in the run 3.4 Hz at rest, 3.8 Hz driven (x1.12, 135 neurons)
5. **input -> DNge011 -> IN09A002 -> target**: 0.0054% of the target's input (0% excitatory, 0.0054% inhibitory; net inhibits).
   - into **DNge011**: 575 synapses (575 excitatory, 0 inhibitory); in the run 3.7 Hz at rest, 31.5 Hz driven (x8.59, 2 neurons)
   - into **IN09A002**: 175 synapses (175 excitatory, 0 inhibitory); in the run 9.1 Hz at rest, 8.1 Hz driven (x0.89, 6 neurons)
   - into **target**: 825 synapses (0 excitatory, 825 inhibitory); in the run 3.4 Hz at rest, 3.8 Hz driven (x1.12, 135 neurons)

### Where it fades

the signal is lost between input (x21.44) and target (x1.12), below the test's 1.5x.

Driven / baseline along the strongest route: input x21.44, target x1.12.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
