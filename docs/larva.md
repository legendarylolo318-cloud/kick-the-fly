# Drosophila Larva Brain Mode

Kick the Fly 2.11 integrates the complete synaptic-resolution connectome of the *Drosophila melanogaster* larva (*Science* 379:eadd9330, 2023).

---

## 1. Connectome Source & Dataset Specifications

- **Reference**: Winding, M., Pedigo, B.D., Barnes, C.L., et al. (2023). "The connectome of an insect brain." *Science*, 379(6636), eadd9330. DOI: [10.1126/science.add9330](https://doi.org/10.1126/science.add9330).
- **Official Repository**: [brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome) (`Supplementary-Data-S1.zip`).
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
  Because the license permits non-commercial redistribution with attribution and share-alike terms, the compact brain pack `kick_larva_brain.npz` is built from official data files and managed via `kickthefly.sim.connectome.larva_loader`.
- **Neuron & Synapse Counts (as read dynamically from dataset)**:
  - **Neurons**: 2,952 annotated cells.
  - **Synapse Pairs (connected graph edges)**: 110,677.
  - **Total Synapses**: 352,611.
  - **Annotated Cell Types**: 96 nociceptors and 2nd-order nociceptive projection neurons (`A00c_a4-a6; noci`, `FFN-27/29`, `Tel5/10`), 94 chordotonal/mechanosensory neurons (`mechano-Ch`, `mechano-II/III`), 358 gustatory sensory neurons (`gustatory-external`, `gustatory-pharyngeal`), 28 thermo-cold/warm receptors, and 2 Goro escape command descending neurons (`5754346`, `3764792` annotated `DN-VNC`, `_telegoro-1`).

---

## 2. Selection, Brain Pack Separation & Save Compatibility

- **CLI Flag**: `--brain adult|larva` (default `adult`).
- **Settings**: `Settings > Brain > Brain: Adult | Larva`.
- **Save State Separation**:
  - Save states (`.ktfsave`), NWB exports, and replays record the active brain type in metadata.
  - Attempting to load a save file generated for the other connectome is strictly rejected with a clear message:
    `"can't load this save: made for the adult brain (currently using larva)"` (or vice versa).

---

## 3. Crawler Physics & Mechanics (GAME RULE)

Unlike adult flies, larvae cannot fly. Physics are modeled by `LarvaBody` in `kickthefly/game/larva.py`:
- **Morphology**: 10 articulated segments (Head, T1–T3 thoracic, A1–A6 abdominal segments) with tapering radii (6–13 px) and viscoelastic spring-damper constraints.
- **Locomotion**:
  - **Forward Crawling**: Posterior-to-anterior peristaltic wave of segment contractions at 1.4 Hz.
  - **Head-Casting / Turning**: Lateral bending and orientation search driven by anterior segments (Head–T3).
  - **Rolling Escape**: High-frequency (up to 4.5 Hz) lateral corkscrew C-curling escape maneuver triggered by noxious stimulation through Goro command descending neurons.
- **Arena Gating**:
  - **Allowed Arenas**: Room, Flypaper, Lamp, Thermo, Pool.
  - **Restricted Arenas**: Open Field, Orchard, Fan, Escape Room. Selecting these arenas prompts an explanatory dialog:
    *e.g. "The open field requires adult flight and wind navigation."*

---

## 4. Sensory & Pain Mapping (CONNECTOME vs GAME RULE)

- **Touch**: Maps to chordotonal organ sensory neurons (`mechano-Ch`) and sub-threshold mechanosensors (`mechano-II/III`).
- **Blowtorch / Heat**: Directly activates annotated Class IV multi-dendritic (md-IV) nociceptors (`A00c` / `noci`).
- **Sugar**: Maps to external and pharyngeal gustatory receptor neurons.
- **Cold**: Maps to cold thermosensory neurons (`thermo-cold`).
- **Direct Nociceptor Pain Meter (CONNECTOME)**:
  In the adult connectome, nociceptors are unannotated, requiring a composite estimate (GAME RULE). In larval mode, annotated Class IV md nociceptors are monitored directly, and the HUD card is tagged `CONNECTOME (md-IV)`.

---

## 5. Validation Results (Held-Out Seeds 1000–1009)

Validation follows pre-fixed empirical criteria: 2 s calm baseline followed by 2 s stimulation (amplitude 0.5), tested against matched random sensory controls using one-sided Wilcoxon signed-rank tests ($n = 10$). Pass requires ratio $\ge 1.50$ and $p < 0.01$.

| Test ID | Pathway & Citation | Empirical Readout | Result |
|---|---|---|---|
| `larva_noci_to_goro_rolling` | Class IV md nociceptors $\to$ Basin interneurons $\to$ Goro command neurons rolling escape ([Ohyama et al. 2015](https://doi.org/10.1038/nature14424); [Winding et al. 2023](https://doi.org/10.1126/science.add9330)) | Goro DNs x1.00 ± 0.04 vs x0.97 ± 0.05 control ($p = 0.0322$) | **FAIL: below 1.5x threshold** (activation reaches Goro consistently above control, but drive ratio is below criterion) |
| `larva_chordotonal_to_basin` | Chordotonal sensory neurons (`mechano-Ch`) $\to$ Basin interneurons ([Ohyama et al. 2015](https://doi.org/10.1038/nature14424); [Jovanic et al. 2016](https://doi.org/10.1016/j.cell.2016.10.025)) | Basin PNs x1.06 ± 0.05 vs x0.98 ± 0.02 control ($p < 0.001$) | **FAIL: below 1.5x threshold** (highly significant $p < 0.001$, but weak synaptic gain in linear LIF model) |

Both failures are reported honestly without artificial weight-tuning.

---

## 6. Performance Benchmarks & Fly Cap

Due to the larva's smaller matrix size (2,952 neurons vs 139,255 adult neurons), throughput is substantially higher:
- **1 Larva (CPU)**: 9,698 steps/s (**161.6x realtime**).
- **1 Larva (Numba JIT)**: 12,815 steps/s (**213.6x realtime**).
- **Multi-Larva Scaling**:
  - 64 larvae run smoothly at 2.40x realtime on CPU, and 3.39x realtime on Numba.
  - **Measured Max Fly Cap**:
    - **CPU (NumPy / torch-cpu)**: **64 larvae** (vs 16 adult flies).
    - **Numba / GPU (CUDA / ROCm)**: **128 larvae** (vs 32/64 adult flies).
