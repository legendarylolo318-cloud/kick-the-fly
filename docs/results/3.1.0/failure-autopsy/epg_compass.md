# Autopsy: E-PG central complex compass forms an orientation bump

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: E-PG / PEN / Delta7 recurrent ring attractor forms a persistent head-direction bump.
Source: Seelig & Jayaraman 2015, Nature 521:186; Green et al. 2017, Nature 546:101

Result: **FAIL**. contrast >= 3.0x and persistence >= 500 ms post-cue

Measured: mean_contrast 1.04, sd_contrast 0.141, mean_persistence_ms 0, mean_baseline_hz 9.37, mean_baseline_sd_hz 1.56, n 10.

The test's own note: With the connectome's signed synapse counts and no ring weights tuned, bump contrast and persistence fail; reported as a negative result.

## the driven wedge to the rest of the ring (what a bump needs to spread and persist)

Input: 4 EPG neurons (the wedge the test drives) (4 neurons). Target: the other 42 EPG neurons (42 neurons). Direct synapses from input to target: 1010.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0.556% | 0% | 0.556% | 0% |
| 2 | 0.242% | 0.604% | -0.362% | 71% |
| 3 | 0.368% | 0.296% | 0.0719% | 45% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> target**: 0.556% of the target's input (0.556% excitatory, 0% inhibitory; net excites).
   - into **target**: 1,010 synapses (1,010 excitatory, 0 inhibitory); in the run 6.8 Hz at rest, 2.8 Hz driven (x0.41, 42 neurons)
2. **input -> ER4m -> target**: 0.16% of the target's input (0% excitatory, 0.16% inhibitory; net inhibits).
   - into **ER4m**: 399 synapses (399 excitatory, 0 inhibitory); in the run 8.5 Hz at rest, 5.4 Hz driven (x0.63, 11 neurons)
   - into **target**: 19,148 synapses (0 excitatory, 19,148 inhibitory); in the run 6.8 Hz at rest, 2.8 Hz driven (x0.41, 42 neurons)
3. **input -> ExR6 -> target**: 0.145% of the target's input (0% excitatory, 0.145% inhibitory; net inhibits).
   - into **ExR6**: 884 synapses (884 excitatory, 0 inhibitory); in the run 11.9 Hz at rest, 17.0 Hz driven (x1.43, 2 neurons)
   - into **target**: 5,705 synapses (0 excitatory, 5,705 inhibitory); in the run 6.8 Hz at rest, 2.8 Hz driven (x0.41, 42 neurons)
4. **input -> ER4m -> ER4m -> target**: 0.088% of the target's input (0.088% excitatory, 0% inhibitory; net excites).
   - into **ER4m**: 399 synapses (399 excitatory, 0 inhibitory); in the run 8.5 Hz at rest, 5.4 Hz driven (x0.63, 11 neurons)
   - into **ER4m**: 14,580 synapses (0 excitatory, 14,580 inhibitory); in the run 8.5 Hz at rest, 5.4 Hz driven (x0.63, 11 neurons)
   - into **target**: 19,148 synapses (0 excitatory, 19,148 inhibitory); in the run 6.8 Hz at rest, 2.8 Hz driven (x0.41, 42 neurons)
5. **input -> ExR6 -> PEN_a(PEN1) -> target**: 0.0414% of the target's input (0% excitatory, 0.0414% inhibitory; net inhibits).
   - into **ExR6**: 884 synapses (884 excitatory, 0 inhibitory); in the run 11.9 Hz at rest, 17.0 Hz driven (x1.43, 2 neurons)
   - into **PEN_a(PEN1)**: 5,089 synapses (0 excitatory, 5,089 inhibitory); in the run 12.0 Hz at rest, 8.0 Hz driven (x0.66, 20 neurons)
   - into **target**: 13,146 synapses (13,146 excitatory, 0 inhibitory); in the run 6.8 Hz at rest, 2.8 Hz driven (x0.41, 42 neurons)

### Where it fades

the signal is lost between input (x10.86) and target (x0.41), below the test's 1.5x.

Driven / baseline along the strongest route: input x10.86, target x0.41.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
