# Autopsy: P1 activation drives the ps1 wing motor neurons (courtship song)

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: Activating the male P1 cluster excites the ps1 wing motor neurons used in pulse song.
Source: von Philipsborn et al. 2011, Neuron 69:509; Kimura et al. 2008, Neuron 59:759

Result: **FAIL**. drive ratio >= 1.5 and > control (one-sided Wilcoxon p < 0.01)

Measured: drive_ratio_mean 1.43, drive_ratio_sd 0.236, control_ratio_mean 0.976, control_ratio_sd 0.172, readout_base_hz 5.22, readout_driven_hz 7.25, p_value 0.000977, n 10.

The test's own note: MaleCNS v1.0 has no "P1" label; P1 is the pC1 types carrying the synonym "Cachero 2010: pMP-e; Yu 2010: pMP4" (assays.P1_TYPES). Added in 2.9 alongside pIP10, with the same criteria. Consistent but too weak. P1 sends pIP10 1,503 synapses and ps1 none, so it is one step further from the motor neurons than pIP10.

## Activating the male P1 cluster excites the ps1 wing motor neurons used in pulse song.

Input: P1 (pMP-e/pMP4 pC1 types, 86) (86 neurons). Target: ps1 wing motor neurons (2) (2 neurons). Direct synapses from input to target: 0.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0% | 0% | 0% | n/a |
| 2 | 0.00214% | 0% | 0.00214% | 0% |
| 3 | 0.00956% | 0.0135% | -0.0039% | 58% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> aSP22 -> target**: 0.00212% of the target's input (0.00212% excitatory, 0% inhibitory; net excites).
   - into **aSP22**: 1,018 synapses (1,018 excitatory, 0 inhibitory); in the run 2.6 Hz at rest, 15.4 Hz driven (x5.97, 2 neurons)
   - into **target**: 13 synapses (13 excitatory, 0 inhibitory); in the run 4.8 Hz at rest, 6.8 Hz driven (x1.44, 2 neurons)
2. **input -> DNa08 -> IN08A011 -> target**: 0.00121% of the target's input (0% excitatory, 0.00121% inhibitory; net inhibits).
   - into **DNa08**: 99 synapses (99 excitatory, 0 inhibitory); in the run 0.3 Hz at rest, 14.3 Hz driven (x28.67, 2 neurons)
   - into **IN08A011**: 313 synapses (313 excitatory, 0 inhibitory); in the run 3.4 Hz at rest, 12.8 Hz driven (x3.75, 14 neurons)
   - into **target**: 271 synapses (0 excitatory, 271 inhibitory); in the run 4.8 Hz at rest, 6.8 Hz driven (x1.44, 2 neurons)
3. **input -> SIP136m -> DNg74_b -> target**: 0.0011% of the target's input (0% excitatory, 0.0011% inhibitory; net inhibits).
   - into **SIP136m**: 1,287 synapses (1,287 excitatory, 0 inhibitory); in the run 0.0 Hz at rest, 3.7 Hz driven (x7.33, 2 neurons)
   - into **DNg74_b**: 632 synapses (632 excitatory, 0 inhibitory); in the run 7.2 Hz at rest, 6.2 Hz driven (x0.86, 2 neurons)
   - into **target**: 159 synapses (0 excitatory, 159 inhibitory); in the run 4.8 Hz at rest, 6.8 Hz driven (x1.44, 2 neurons)
4. **input -> DNp34 -> target**: 2.36e-05% of the target's input (2.36e-05% excitatory, 0% inhibitory; net excites).
   - into **DNp34**: 3 synapses (3 excitatory, 0 inhibitory); in the run 6.5 Hz at rest, 5.6 Hz driven (x0.86, 2 neurons)
   - into **target**: 13 synapses (13 excitatory, 0 inhibitory); in the run 4.8 Hz at rest, 6.8 Hz driven (x1.44, 2 neurons)

### Where it fades

the signal is lost between aSP22 (x5.97) and target (x1.44), below the test's 1.5x.

Driven / baseline along the strongest route: input x19.99, aSP22 x5.97, target x1.44.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
