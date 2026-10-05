# Autopsy: Grooming hierarchy: head stimulation wins over, and suppresses, abdomen grooming

**MODEL PREDICTION** for the run numbers; **CONNECTOME** for the path numbers. Read-only: no weight, constant or threshold is changed or suggested.

Claim: With head and abdomen stimulated together, flies groom anterior first: the head (front-leg) program runs and the posterior (hind-leg) program is suppressed.
Source: Seeds et al. 2014, eLife 3:e02951

Result: **FAIL**. abdomen alone raises hind-leg MNs >= 1.5x; with both, front-leg MNs > hind-leg MNs and hind-leg MNs < abdomen alone (each one-sided Wilcoxon p < 0.01)

Measured: posterior_hind_mean 1.46, posterior_hind_sd 0.0253, anterior_front_mean 1.07, both_front_mean 1.09, both_front_sd 0.0558, both_hind_mean 1.44, both_hind_sd 0.0214, p_priority 1, p_suppression 0.0654, n 10.

The test's own note: Redesigned in 2.9 to match Seeds et al. (simultaneous stimulation), criteria fixed before the run. Front-leg motor neurons stand in for head grooming and hind-leg ones for abdomen grooming. Neither half of the hierarchy shows: head bristles barely move the front-leg motor neurons, and adding them doesn't suppress the abdomen's drive to the hind legs.

## head bristles to the front-leg motor neurons (the head grooming program)

Input: head bristle neurons (BM_*) (823 neurons). Target: front-leg motor neurons (135 neurons). Direct synapses from input to target: 0.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0% | 0% | 0% | n/a |
| 2 | 0.0448% | 0.00739% | 0.0374% | 14% |
| 3 | 0.128% | 0.113% | 0.0151% | 47% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> DNg15 -> target**: 0.0117% of the target's input (0.0117% excitatory, 0% inhibitory; net excites).
   - into **DNg15**: 3,129 synapses (3,129 excitatory, 0 inhibitory); in the run 5.6 Hz at rest, 27.5 Hz driven (x4.93, 2 neurons)
   - into **target**: 491 synapses (491 excitatory, 0 inhibitory); in the run 3.4 Hz at rest, 3.6 Hz driven (x1.06, 135 neurons)
2. **input -> DNg48 -> IN03A024 -> target**: 0.00937% of the target's input (0.00937% excitatory, 0% inhibitory; net excites).
   - into **DNg48**: 2,624 synapses (2,624 excitatory, 0 inhibitory); in the run 3.2 Hz at rest, 44.2 Hz driven (x13.59, 2 neurons)
   - into **IN03A024**: 165 synapses (165 excitatory, 0 inhibitory); in the run 3.6 Hz at rest, 8.2 Hz driven (x2.28, 4 neurons)
   - into **target**: 941 synapses (941 excitatory, 0 inhibitory); in the run 3.4 Hz at rest, 3.6 Hz driven (x1.06, 135 neurons)
3. **input -> DNg48 -> INXXX194 -> target**: 0.00775% of the target's input (0% excitatory, 0.00775% inhibitory; net inhibits).
   - into **DNg48**: 2,624 synapses (2,624 excitatory, 0 inhibitory); in the run 3.2 Hz at rest, 44.2 Hz driven (x13.59, 2 neurons)
   - into **INXXX194**: 125 synapses (125 excitatory, 0 inhibitory); in the run 3.9 Hz at rest, 33.4 Hz driven (x8.53, 2 neurons)
   - into **target**: 628 synapses (0 excitatory, 628 inhibitory); in the run 3.4 Hz at rest, 3.6 Hz driven (x1.06, 135 neurons)
4. **input -> DNge078 -> target**: 0.00696% of the target's input (0.00696% excitatory, 0% inhibitory; net excites).
   - into **DNge078**: 131 synapses (131 excitatory, 0 inhibitory); in the run 3.5 Hz at rest, 18.3 Hz driven (x5.24, 2 neurons)
   - into **target**: 600 synapses (600 excitatory, 0 inhibitory); in the run 3.4 Hz at rest, 3.6 Hz driven (x1.06, 135 neurons)

### Where it fades

the signal is lost between DNg15 (x4.93) and target (x1.06), below the test's 1.5x.

Driven / baseline along the strongest route: input x19.27, DNg15 x4.93, target x1.06.

## abdominal sensory neurons to the hind-leg motor neurons (the abdomen program)

Input: abdominal mechanosensory neurons (1043 neurons). Target: hind-leg motor neurons (130 neurons). Direct synapses from input to target: 4.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0.000397% | 0% | 0.000397% | 0% |
| 2 | 0.175% | 0.0599% | 0.115% | 25% |
| 3 | 0.329% | 0.271% | 0.0572% | 45% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> INXXX025 -> target**: 0.0433% of the target's input (0.0433% excitatory, 0% inhibitory; net excites).
   - into **INXXX025**: 3,115 synapses (3,115 excitatory, 0 inhibitory); in the run 2.2 Hz at rest, 64.8 Hz driven (x28.81, 2 neurons)
   - into **target**: 259 synapses (259 excitatory, 0 inhibitory); in the run 5.7 Hz at rest, 8.4 Hz driven (x1.46, 130 neurons)
2. **input -> INXXX058 -> target**: 0.0227% of the target's input (0% excitatory, 0.0227% inhibitory; net inhibits).
   - into **INXXX058**: 7,647 synapses (7,647 excitatory, 0 inhibitory); in the run 2.4 Hz at rest, 59.5 Hz driven (x24.62, 6 neurons)
   - into **target**: 141 synapses (0 excitatory, 141 inhibitory); in the run 5.7 Hz at rest, 8.4 Hz driven (x1.46, 130 neurons)
3. **input -> INXXX025 -> IN21A010 -> target**: 0.00676% of the target's input (0.00676% excitatory, 0% inhibitory; net excites).
   - into **INXXX025**: 3,115 synapses (3,115 excitatory, 0 inhibitory); in the run 2.2 Hz at rest, 64.8 Hz driven (x28.81, 2 neurons)
   - into **IN21A010**: 605 synapses (605 excitatory, 0 inhibitory); in the run 4.8 Hz at rest, 6.1 Hz driven (x1.27, 6 neurons)
   - into **target**: 1,233 synapses (1,233 excitatory, 0 inhibitory); in the run 5.7 Hz at rest, 8.4 Hz driven (x1.46, 130 neurons)
4. **input -> INXXX114 -> IN20A.22A008 -> target**: 0.0061% of the target's input (0.0061% excitatory, 0% inhibitory; net excites).
   - into **INXXX114**: 1,489 synapses (1,489 excitatory, 0 inhibitory); in the run 11.1 Hz at rest, 60.3 Hz driven (x5.44, 2 neurons)
   - into **IN20A.22A008**: 321 synapses (321 excitatory, 0 inhibitory); in the run 3.0 Hz at rest, 8.3 Hz driven (x2.76, 12 neurons)
   - into **target**: 2,071 synapses (2,071 excitatory, 0 inhibitory); in the run 5.7 Hz at rest, 8.4 Hz driven (x1.46, 130 neurons)
5. **input -> target**: 0.000397% of the target's input (0.000397% excitatory, 0% inhibitory; net excites).
   - into **target**: 4 synapses (4 excitatory, 0 inhibitory); in the run 5.7 Hz at rest, 8.4 Hz driven (x1.46, 130 neurons)

### Where it fades

the signal is lost between INXXX025 (x28.81) and target (x1.46), below the test's 1.5x.

Driven / baseline along the strongest route: input x5.59, INXXX025 x28.81, target x1.46.

## head bristles to the hind-leg motor neurons (the suppression the hierarchy needs)

Input: head bristle neurons (BM_*) (823 neurons). Target: hind-leg motor neurons (130 neurons). Direct synapses from input to target: 0.

### How much of the target's input the input can supply

| synapses | excitatory walks | inhibitory walks | net | inhibitory share |
|---|---|---|---|---|
| 1 | 0% | 0% | 0% | n/a |
| 2 | 0.0207% | 0.00467% | 0.016% | 18% |
| 3 | 0.0272% | 0.0236% | 0.00368% | 46% |

Shares of the average target neuron's input synapses, along walks of exactly that many synapses.

### The strongest routes, through cell types

1. **input -> DNg35 -> target**: 0.0178% of the target's input (0.0178% excitatory, 0% inhibitory; net excites).
   - into **DNg35**: 2,207 synapses (2,207 excitatory, 0 inhibitory); in the run 6.0 Hz at rest, 17.8 Hz driven (x2.96, 2 neurons)
   - into **target**: 372 synapses (372 excitatory, 0 inhibitory); in the run 5.7 Hz at rest, 5.3 Hz driven (x0.94, 130 neurons)
2. **input -> DNge067 -> target**: 0.00406% of the target's input (0% excitatory, 0.00406% inhibitory; net inhibits).
   - into **DNge067**: 613 synapses (613 excitatory, 0 inhibitory); in the run 1.2 Hz at rest, 30.2 Hz driven (x24.13, 2 neurons)
   - into **target**: 82 synapses (0 excitatory, 82 inhibitory); in the run 5.7 Hz at rest, 5.3 Hz driven (x0.94, 130 neurons)
3. **input -> BM_InOm -> DNg35 -> target**: 0.00353% of the target's input (0.00353% excitatory, 0% inhibitory; net excites).
   - into **BM_InOm**: 2,986 synapses (2,986 excitatory, 0 inhibitory); in the run 3.5 Hz at rest, 65.6 Hz driven (x18.85, 745 neurons)
   - into **DNg35**: 1,376 synapses (1,376 excitatory, 0 inhibitory); in the run 6.0 Hz at rest, 17.8 Hz driven (x2.96, 2 neurons)
   - into **target**: 372 synapses (372 excitatory, 0 inhibitory); in the run 5.7 Hz at rest, 5.3 Hz driven (x0.94, 130 neurons)
4. **input -> BM_Vt_PoOc -> DNg35 -> target**: 0.00167% of the target's input (0.00167% excitatory, 0% inhibitory; net excites).
   - into **BM_Vt_PoOc**: 1,129 synapses (1,129 excitatory, 0 inhibitory); in the run 5.9 Hz at rest, 66.6 Hz driven (x11.25, 8 neurons)
   - into **DNg35**: 723 synapses (723 excitatory, 0 inhibitory); in the run 6.0 Hz at rest, 17.8 Hz driven (x2.96, 2 neurons)
   - into **target**: 372 synapses (372 excitatory, 0 inhibitory); in the run 5.7 Hz at rest, 5.3 Hz driven (x0.94, 130 neurons)

### Where it fades

the signal is lost between DNg35 (x2.96) and target (x0.94), below the test's 1.5x.

Driven / baseline along the strongest route: input x19.27, DNg35 x2.96, target x0.94.

Method: walks of 1-3 synapses, simulator weights (signed count / post's input count); stage rates from 3 seeds, the pathway tests' drive (0.5 for 2 s after 2 s calm).
Weights digest (unchanged by the autopsy): 31952e3631ba801d.
