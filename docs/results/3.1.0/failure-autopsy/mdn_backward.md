# Autopsy: Moonwalker neurons drive backward walking

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: Activating MDN drives the leg motor program for walking backward.
Source: Bidaye et al. 2014, Science 344:97

Result: **FAIL**. drive ratio >= 1.5 and > control (one-sided Wilcoxon p < 0.01)

Measured: drive_ratio_mean 0.96, drive_ratio_sd 0.0296, control_ratio_mean 0.963, control_ratio_sd 0.0513, readout_base_hz 4.02, readout_driven_hz 3.86, p_value 0.385, n 10.

The test's own note: The sim has no walking rhythm; this asks whether MDN activity reaches the leg motor neurons at all. MDN makes 12 synapses onto them directly; the 369 neurons it sends at least 5 synapses to (mostly VNC interneurons) send them 27,560 excitatory and 24,186 inhibitory synapses. That near balance of excitation and inhibition in the VNC interneuron layer is where the drive cancels out in this model.

## Activating MDN drives the leg motor program for walking backward.

Input: MDN (4) (4 neurons). Target: leg motor neurons (T1-T3, 381) (381 neurons). Direct synapses from input to target: 12.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0.000256% | 0% | 0.000256% | 0% |
| 2 | 0.0302% | 0.0107% | 0.0195% | 26% |
| 3 | 0.0212% | 0.0344% | -0.0132% | 62% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> IN03A010 -> target**: 0.006% of the target's input (0.006% excitatory, 0% inhibitory; net excites).
   - into **IN03A010**: 324 synapses (324 excitatory, 0 inhibitory); in the run 2.2 Hz at rest, 5.6 Hz driven (x2.60, 6 neurons)
   - into **target**: 6,735 synapses (6,735 excitatory, 0 inhibitory); in the run 4.1 Hz at rest, 3.9 Hz driven (x0.96, 381 neurons)
2. **input -> LBL40 -> target**: 0.00449% of the target's input (0.00449% excitatory, 0% inhibitory; net excites).
   - into **LBL40**: 463 synapses (463 excitatory, 0 inhibitory); in the run 4.6 Hz at rest, 18.4 Hz driven (x4.02, 2 neurons)
   - into **target**: 444 synapses (444 excitatory, 0 inhibitory); in the run 4.1 Hz at rest, 3.9 Hz driven (x0.96, 381 neurons)
3. **input -> IN03B015 -> IN01A015 -> target**: 0.000708% of the target's input (0% excitatory, 0.000708% inhibitory; net inhibits).
   - into **IN03B015**: 462 synapses (462 excitatory, 0 inhibitory); in the run 8.7 Hz at rest, 22.0 Hz driven (x2.54, 4 neurons)
   - into **IN01A015**: 579 synapses (0 excitatory, 579 inhibitory); in the run 1.7 Hz at rest, 0.7 Hz driven (x0.42, 6 neurons)
   - into **target**: 6,804 synapses (6,804 excitatory, 0 inhibitory); in the run 4.1 Hz at rest, 3.9 Hz driven (x0.96, 381 neurons)
4. **input -> IN20A.22A073 -> IN16B016 -> target**: 0.000638% of the target's input (0% excitatory, 0.000638% inhibitory; net inhibits).
   - into **IN20A.22A073**: 345 synapses (345 excitatory, 0 inhibitory); in the run 1.6 Hz at rest, 9.4 Hz driven (x6.04, 18 neurons)
   - into **IN16B016**: 247 synapses (247 excitatory, 0 inhibitory); in the run 9.3 Hz at rest, 8.4 Hz driven (x0.90, 6 neurons)
   - into **target**: 6,865 synapses (0 excitatory, 6,865 inhibitory); in the run 4.1 Hz at rest, 3.9 Hz driven (x0.96, 381 neurons)
5. **input -> target**: 0.000256% of the target's input (0.000256% excitatory, 0% inhibitory; net excites).
   - into **target**: 12 synapses (12 excitatory, 0 inhibitory); in the run 4.1 Hz at rest, 3.9 Hz driven (x0.96, 381 neurons)

### Where it fades

the signal is lost between IN03A010 (x2.60) and target (x0.96), below the test's 1.5x.

Driven / baseline along the strongest route: input x40.20, IN03A010 x2.60, target x0.96.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
