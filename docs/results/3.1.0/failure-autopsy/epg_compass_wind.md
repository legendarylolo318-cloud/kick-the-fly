# Autopsy: E-PG compass forms a bump from steady directional wind (open field)

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: Wind direction is a head-direction cue: steady directional wind anchors a persistent E-PG bump whose position follows the wind.
Source: Okubo, Patella, D'Alessandro & Wilson 2020, Neuron 107:924; Seelig & Jayaraman 2015, Nature 521:186

Result: **FAIL**. contrast >= 3.0x across the 16 PB glomeruli during wind and persistence >= 500 ms after it stops (fixed before the run; the same as the visual test)

Measured: mean_contrast 1.81, sd_contrast 0.107, baseline_contrast 1.71, mean_persistence_ms 101, direction_tracking 0.46, direction_tracking_p 0.17, direction_tracking_null 0.399, epg_rate_baseline_hz 6.8, epg_rate_wind_hz 5.47, wind_speed_m_s 6, n 10.

The test's own note: A second, different test from the visual one: wind, not a visual landmark. The pass criteria are the visual test's, fixed before the run. Direction tracking is reported (circular correlation with a permutation null) but not needed to pass. No weights or time constants were tuned.

## wind neurons to the E-PG compass neurons

Input: JO-C/E mechanosensory neurons (the wind's route in the open field) (335 neurons). Target: EPG neurons (46 neurons). Direct synapses from input to target: 0.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0% | 0% | 0% | n/a |
| 2 | 0% | 0% | 0% | n/a |
| 3 | 0.00148% | 0.00737% | -0.00589% | 83% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> CB2585 -> ER1_b -> target**: 0.00209% of the target's input (0% excitatory, 0.00209% inhibitory; net inhibits).
   - into **CB2585**: 262 synapses (262 excitatory, 0 inhibitory); in the run 0.2 Hz at rest, 2.3 Hz driven (x4.67, 6 neurons)
   - into **ER1_b**: 120 synapses (120 excitatory, 0 inhibitory); in the run 10.2 Hz at rest, 11.5 Hz driven (x1.13, 13 neurons)
   - into **target**: 5,624 synapses (0 excitatory, 5,624 inhibitory); in the run 6.7 Hz at rest, 6.2 Hz driven (x0.92, 46 neurons)
2. **input ->  -> ER1_b -> target**: 0.00173% of the target's input (0% excitatory, 0.00173% inhibitory; net inhibits).
   - into ****: 2,278 synapses (2,278 excitatory, 0 inhibitory); in the run 7.5 Hz at rest, 7.3 Hz driven (x0.97, 2194 neurons)
   - into **ER1_b**: 165 synapses (165 excitatory, 0 inhibitory); in the run 10.2 Hz at rest, 11.5 Hz driven (x1.13, 13 neurons)
   - into **target**: 5,624 synapses (0 excitatory, 5,624 inhibitory); in the run 6.7 Hz at rest, 6.2 Hz driven (x0.92, 46 neurons)

### Where it fades

the signal is lost between CB2585 (x4.67) and ER1_b (x1.13), below the test's 1.5x.

Driven / baseline along the strongest route: input x116.37, CB2585 x4.67, ER1_b x1.13, target x0.92.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
