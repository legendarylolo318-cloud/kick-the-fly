# Pharmacology (3.0)

Lab > **Pharmacology**. A drug is one rule: multiply the weight of every synapse whose presynaptic neuron the dataset predicts
releases transmitter X by a factor that follows the **dose** (a slider, 0-100%). **MODEL PREDICTION.**

| drug | scales | effect at full dose |
|---|---|---|
| Picrotoxin | GABA and glutamate synapses | x0 (the game's original inhibition block, moved here from Lab > Connectome robustness; identical arithmetic) |
| Cholinergic block | acetylcholine synapses | x0 |
| Glutamate-Cl block | glutamate synapses | x0 |
| GABA-A agonist | GABA synapses | up to x2 (the x2 is a game rule) |

Blockers scale by `1 - dose`; the agonist by `1 + dose x (2 - 1)`. Drugs that act on the same transmitter multiply.

## Confidence

MaleCNS v1.0 gives each neuron's transmitter as measured (`ground_truth`, 85,484 neurons) or predicted with a confidence. The panel
shows, for the drug and dose you chose, how many presynaptic neurons, connections and **synapses** are affected at each level:
*measured*, predicted >= 0.9, 0.7-0.9, 0.5-0.7, < 0.5, and *no confidence*. A toggle **includes or excludes low-confidence predictions**
(below an adjustable cut, default 0.7; measured transmitters always count). Excluding them leaves the drug off those neurons. The counts
are what the wiring will actually change (tested equal), not an estimate.

## What it does not model

No receptor subtypes, subunit affinities, location on the cell, kinetics, washout curves or side effects. The simulator holds the brain's
average firing near a target with a slow global gain, which works against any drug that changes overall synaptic strength, so read effects
soon after a drug goes on (the dose-response button measures 1 s windows on fresh brains).

**Octopamine and dopamine modulation are left out on purpose.** In this simulation the synapses of octopamine, dopamine, serotonin and
"unclear" neurons carry no sign and are not in the synaptic matrix at all, and there is no existing gain to scale; any knob would be
invented. (The plastic dopamine -> mushroom-body learning rule is a separate game rule and is not touched.)

## Apply, wash out, compatibility

"Apply to every fly" uses the same reversible wiring machinery as the synapse threshold and sign flips (`sim/wiring.py`, new optional
`nt_scales` / `nt_min_conf` fields; a default wiring serializes exactly as before, so older saves and validation output are unchanged).
"Wash out" restores every weight exactly, learned synapses included. A drug also works together with a synapse threshold or sign flips.

## Protocols and Python

```yaml
drug: {doses: {picrotoxin: 0.5}, include_low_confidence: false, cut: 0.7}
```

`protocols/drug_picrotoxin.yaml`, `protocols/drug_cholinergic_block.yaml`. `fly.drug("cholinergic", 0.5)`, `fly.washout()`. A drug block
is for stimulus protocols; assay protocols don't take one yet.
