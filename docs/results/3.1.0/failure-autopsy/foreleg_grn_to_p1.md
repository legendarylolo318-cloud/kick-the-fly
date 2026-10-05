# Autopsy: Foreleg pheromone GRNs excite the male P1 cluster

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: Activating the foreleg gustatory neurons annotated putative ppk23/ppk25 (LgLG5-8) excites the male P1 courtship cluster.
Source: Clowney et al. 2015, Neuron 87:1036; Kallman et al. 2015, eLife 4:e11188

Result: **FAIL**. drive ratio >= 1.5 and > control (one-sided Wilcoxon p < 0.01)

Measured: drive_ratio_mean 1.2, drive_ratio_sd 0.108, control_ratio_mean 0.925, control_ratio_sd 0.0612, readout_base_hz 3.45, readout_driven_hz 4.13, p_value 0.000977, n 10.

The test's own note: MaleCNS v1.0 annotates LgLG5-8 as gustatory leg-bristle neurons of the prothoracic leg nerve (ProLN), receptorType putative_ppk23 or putative_ppk25. In real flies this reaches P1 through the ascending vAB3 and PPN1 neurons (AN09B017e/f/g and AN05B102a in MaleCNS). Consistent but too weak.

## Activating the foreleg gustatory neurons annotated putative ppk23/ppk25 (LgLG5-8) excites the male P1 courtship cluster.

Input: LgLG5-8, putative ppk23/ppk25 (64) (64 neurons). Target: P1 courtship cluster (86) (86 neurons). Direct synapses from input to target: 0.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0% | 0% | 0% | n/a |
| 2 | 0.166% | 0.241% | -0.075% | 59% |
| 3 | 0.52% | 0.555% | -0.0352% | 52% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> AN09B017f -> target**: 0.205% of the target's input (0.118% excitatory, 0.0877% inhibitory; net excites).
   - into **AN09B017f**: 1,232 synapses (533 excitatory, 699 inhibitory); in the run 0.6 Hz at rest, 1.0 Hz driven (x1.71, 2 neurons)
   - into **target**: 612 synapses (0 excitatory, 612 inhibitory); in the run 3.4 Hz at rest, 4.2 Hz driven (x1.26, 86 neurons)
2. **input -> AN09B017a -> mAL_m1 -> target**: 0.0861% of the target's input (0.0602% excitatory, 0.0259% inhibitory; net excites).
   - into **AN09B017a**: 947 synapses (663 excitatory, 284 inhibitory); in the run 1.0 Hz at rest, 55.7 Hz driven (x55.67, 2 neurons)
   - into **mAL_m1**: 1,492 synapses (0 excitatory, 1,492 inhibitory); in the run 0.2 Hz at rest, 0.0 Hz driven (x0.00, 12 neurons)
   - into **target**: 2,062 synapses (0 excitatory, 2,062 inhibitory); in the run 3.4 Hz at rest, 4.2 Hz driven (x1.26, 86 neurons)
3. **input -> LgLG5 -> AN09B017f -> target**: 0.0807% of the target's input (0% excitatory, 0.0807% inhibitory; net inhibits).
   - into **LgLG5**: 2,675 synapses (0 excitatory, 2,675 inhibitory); in the run 0.1 Hz at rest, 65.1 Hz driven (x130.15, 13 neurons)
   - into **AN09B017f**: 699 synapses (0 excitatory, 699 inhibitory); in the run 0.6 Hz at rest, 1.0 Hz driven (x1.71, 2 neurons)
   - into **target**: 612 synapses (0 excitatory, 612 inhibitory); in the run 3.4 Hz at rest, 4.2 Hz driven (x1.26, 86 neurons)
4. **input -> AN09B017b -> target**: 0.0792% of the target's input (0.00063% excitatory, 0.0786% inhibitory; net inhibits).
   - into **AN09B017b**: 976 synapses (968 excitatory, 8 inhibitory); in the run 0.7 Hz at rest, 66.4 Hz driven (x99.63, 2 neurons)
   - into **target**: 230 synapses (0 excitatory, 230 inhibitory); in the run 3.4 Hz at rest, 4.2 Hz driven (x1.26, 86 neurons)

### Where it fades

the signal is lost between AN09B017f (x1.71) and target (x1.26), below the test's 1.5x.

Driven / baseline along the strongest route: input x132.98, AN09B017f x1.71, target x1.26.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
