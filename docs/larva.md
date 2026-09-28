# Drosophila Larva Brain

Kick the Fly 2.11 can load the synaptic-resolution connectome of the first-instar *Drosophila* larva brain
(Winding et al. 2023, *Science* 379:eadd9330). In this release it is **headless only**: validation, benchmarks and
Python use. The windowed game (2D and 3D) always runs the adult brain and logs a warning if `brain.brain = "larva"`,
because the larva body (`kickthefly/game/larva.py`) does not yet implement what the game loop expects of a fly.

## Source, license and how it is shipped

- **Reference**: Winding, M., Pedigo, B.D., Barnes, C.L., et al. (2023). "The connectome of an insect brain."
  *Science* 379(6636), eadd9330. DOI: [10.1126/science.add9330](https://doi.org/10.1126/science.add9330).
- **Data**: the paper's Supplementary Data S1 (`Supplementary-Data-S1.zip`). `larva_loader.py` downloads it from the
  [brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome) mirror
  (Betzel lab; not the publisher) and only accepts the exact archive (SHA-256
  `8c1f4380…72a4c`, checked before use).
- **License**: neither the Science supplement nor the mirror states a license (the mirror repository has none, and
  Crossref lists none for the article). Until one is confirmed, **the larva data is not redistributed**: it is not in
  git, not in the exe or AppImage, and not in release artifacts. The pack `data/kick_larva_brain.npz` is built on the
  user's machine on first use (`python -m kickthefly.sim.connectome.larva_loader build`). Please cite the paper above
  in any work that uses it.
- **Counts** (read from the matrix at build time, not hardcoded): 2,952 neurons, 110,677 connected pairs,
  352,611 synapses.

## What the pack keeps, and what it adds

Kept from Data S1: the all-to-all synapse-count matrix, the broad cell class (`type`, e.g. `sensory`, `DN-VNC`, `KC`),
the additional annotations (`subtype`, e.g. `noci`, `mechano-Ch`, `_telegoro-1`), hemisphere and cluster.

Added by the pack builder, **not from the data** (GAME RULE):
- **Signs.** Data S1 has no transmitter identities. The builder makes local neurons (`LN`) and MBONs inhibitory and
  every other neuron excitatory. This is a guess and it matters: the network idles at about 60 Hz.
- **Cell-body positions** for the brain view are laid out from hemisphere and cluster; they are not anatomical.

## Annotations used, and what they really are

`larva_loader.larva_groups()` selects by exact annotation tag (`subtype`):

| Group | Annotation | Count | What it is |
|---|---|---|---|
| `noci` | `noci`, `A00c_a4/a5/a6; noci` | 12 | nociceptive **ascending** neurons from the nerve cord. The class IV md sensory neurons are not in this dataset. |
| `chordo` | `mechano-Ch` | 12 | chordotonal-pathway **ascending** neurons |
| `noci_pn`, `chordo_pn` | `noci 2nd_order PN`, `mechano-Ch 2nd_order PN` | 84, 64 (8 in both) | brain neurons downstream of those. **Not Basins**: the Basin interneurons are in the nerve cord. |
| `goro` | `_telegoro-1` | 2 | a DN-VNC pair annotated `_telegoro-1`. Goro itself is in the nerve cord; this pair is not verified as Goro. |

## Validation (seeds 1000-1009, `--headless --validate --brain larva`)

Written to `validation_results_larva.json`, separately from the adult results. Individuality is forced off.

| Test | Measured | Result |
|---|---|---|
| `larva_noci_to_goro_rolling` | `_telegoro-1` x0.66 ± 0.01 vs x0.65 ± 0.01 for 12 random other sensory neurons, p = 0.001 | **FAIL: no excitation** |
| `larva_chordotonal_to_basin` | 2nd-order PNs x0.68 ± 0.01 vs x0.64 ± 0.01, p = 0.001 | **FAIL: no excitation** |

Any sensory drive lowers these readouts, whichever neurons are driven (see the signs caveat above). Nothing was tuned.
Because neither pathway is validated, larva reactions (ROLL etc.) would be GAME RULE, not real.

## Save states

Save states record the brain. Loading one made for the other connectome is refused with
`can't load this save: made for the adult brain (currently using larva)` (or the reverse).
