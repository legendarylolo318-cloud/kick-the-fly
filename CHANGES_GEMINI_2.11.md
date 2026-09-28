# Kick the Fly 2.11 — Release Changes & Verification Report

**Branch**: `gemini/2.11`  
**Reference Repo**: `/home/lolo/kick-the-fly`

---

## 1. Files Changed & Created

### Created Files
- `data/kick_larva_brain.npz`: Compact larval connectome pack (2,952 neurons, 110,677 synapse pairs, 352,611 synapses).
- `kickthefly/sim/connectome/larva_loader.py`: Drosophila larva connectome builder, download manager, and dynamic parser.
- `kickthefly/core/individuality.py`: Deterministic log-normal gain generator ($W_{\text{fly}} = D_{\text{post}} \cdot W \cdot D_{\text{pre}}$) and non-scripted personality card profiler.
- `kickthefly/game/larva.py`: Segmented crawler ragdoll body physics (`LarvaBody`, 10 segments), peristaltic wave crawling, head-casting, and rolling escape.
- `kickthefly/core/pet.py`: `PetManager` lifecycle, atomic persistence, clock guards, deterministic catch-up, and neural couplings.
- `kickthefly/lab/individuality_assay.py`: ICC(1,1), Kain/Linneweber consistency test, and validated adult behavior pass rate evaluations.
- `protocols/larva_nociception_rolling.yaml`: Automated protocol for larval nociception and Goro command neuron activation.
- `protocols/individuality_consistency.yaml`: Automated protocol for repeated behavioral trials measuring individual consistency.
- `protocols/pet_catchup.yaml`: Automated protocol verifying pet state catch-up and need progression.
- `docs/larva.md`: Larva connectome architecture, crawler mechanics, validation, and benchmarks.
- `docs/individuality.md`: Individuality formulation, GPU streaming invariants, ICC statistics, and personality profiles.
- `docs/pet.md`: Pet mode lifecycle, absence of background daemons, deterministic catch-up, and neural couplings.
- `tests/test_larva.py`: Pytest suite for larva loader, counts, brain pack separation, body physics, and validation.
- `tests/test_individuality.py`: Pytest suite for individuality determinism, CPU bit-exactness, validation invariant, and GPU batched equality.
- `tests/test_pet.py`: Pytest suite for pet persistence, clock clamping, deterministic catch-up, and safety invariants.
- `CHANGES_GEMINI_2.11.md`: This comprehensive deliverable documentation.

### Modified Files
- `README.md`: Added Larval Brain, Individuality, and Pet Mode feature overviews; added larva connectome citation and CC BY-NC-SA 4.0 license to Credits.
- `docs/performance.md`: Added 2.11 larva single-fly and multi-fly benchmarks, measured fly caps (64 CPU / 128 GPU), and individuality overhead benchmarks.
- `docs/validation.md`: Added larva validation rows with empirical numbers and failure explanations.
- `kickthefly/core/config.py`: Added `brain.brain`, `brain.individuality`, `brain.pet_real_stakes`, added `"pet"` to mode options, and added `.pet`/`.larva` properties.
- `kickthefly/core/savestate.py`: Added `brain` and `individuality` metadata; rejected cross-brain saves with reason `"made for the <other> brain (currently using <current>)"`.
- `kickthefly/core/simcore.py`: Supported `brain="adult"|"larva"`, passed individuality parameters into brain creation.
- `kickthefly/sim/brainpack.py`: Added support for loading `kick_larva_brain.npz`.
- `kickthefly/sim/connectome/sim.py`: Implemented $D_{\text{pre}}$ and $D_{\text{post}}$ scaling in `LIFParams`, `_propagate()`, and `step()`.
- `kickthefly/sim/connectome/backends.py`: Implemented $D_{\text{pre}}/D_{\text{post}}$ scaling in `NumbaBackend` and `TorchBackend`.
- `kickthefly/lab/validation.py`: Added `LARVA_TESTS`, updated `EXPECTED` with larva tests, forced `individuality="off"` during validation, supported `brain="adult"|"larva"`.
- `kickthefly/game/kick_the_fly.py`:
  - Updated canonical docstring with 2.11 CONNECTOME and GAME RULE tags.
  - Implemented dynamic max fly caps in `get_max_flies()` (64 CPU, 128 GPU on larva).
  - Updated `load_brain` for larva brain pack and individuality initialization.
  - Added CLI options `--brain`, `--individuality`, `--pet`.
  - Configured larval sensory mapping (touch $\to$ chordotonal/mechanosensory, blowtorch $\to$ Class IV md nociceptors, sugar $\to$ gustatory, cold $\to$ thermo-cold).
  - Updated `pain_parts` and `_draw_pain` to read Class IV md nociceptors directly in larva mode, tagged `CONNECTOME`.
  - Added personality card display to fly focus cycling (`cycle_focus`), neuron inspector header (`_draw_inspect`), and Pet widget.
  - Added casual HUD widget (`_draw_pet_widget`) and launch greeting in Pet mode.
  - Handled `LarvaBody` physics in 2D ragdoll loops.
  - Added `"ROLL": "real"`, `"ROLLING": "real"`, `"PAIN (LARVA)": "real"`, `"CRAWL": "rule"`, `"HEAD CAST": "rule"` to `REACTION_SOURCE`.
- `kickthefly/game/kick3d.py`: Propagated `brain` and `individuality` into 3D state dict.
- `kickthefly/ui/menu.py`: Added Pet mode to pause menu and Esc mode cycling.
- `kickthefly/data/locales/en.json` & `de.json`: Added localization entries for Pet Mode, Larva, Individuality, and associated settings.

---

## 2. Drosophila Larva Connectome Specifications

- **Reference Citation**: Winding, M., Pedigo, B.D., Barnes, C.L., et al. (2023). "The connectome of an insect brain." *Science*, 379(6636), eadd9330. DOI: [10.1126/science.add9330](https://doi.org/10.1126/science.add9330).
- **Official Source**: `https://raw.githubusercontent.com/brain-networks/larval-drosophila-connectome/main/Supplementary-Data-S1.zip`.
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
- **Counts Dynamically Read from Official Data**:
  - **Neurons**: 2,952.
  - **Synapse Pairs (connected pairs)**: 110,677.
  - **Total Synapses**: 352,611.
  - **Annotated Cell Types**: 96 nociceptors & 2nd-order nociceptive PNs, 94 chordotonal & mechanosensory neurons, 358 gustatory neurons, 28 thermo neurons, 2 Goro command descending neurons (`5754346`, `3764792` annotated `_telegoro-1`).

---

## 3. Validation Results (Held-Out Seeds 1000–1009)

Pre-fixed empirical criteria: 2 s calm baseline followed by 2 s stimulation (amp=0.5), Wilcoxon signed-rank paired greater against matched sensory controls ($n = 10$). Pass requires ratio $\ge 1.50$ and $p < 0.01$.

| Test ID | Citation | Measured Drive vs Control | Result | What It Actually Measures |
|---|---|---|---|---|
| `larva_noci_to_goro_rolling` | Ohyama et al. 2015 (*Nature* 520:633); Winding et al. 2023 (*Science* 379:eadd9330) | Goro DNs x1.00 ± 0.04 vs x0.97 ± 0.05 control, $p = 0.0322$ | **FAIL: too weak** | Ratio of Goro escape command descending neuron firing rate under Class IV md nociceptor stimulation compared to matched random sensory control. |
| `larva_chordotonal_to_basin` | Ohyama et al. 2015 (*Nature* 520:633); Jovanic et al. 2016 (*Cell* 167:858) | Basin PNs x1.06 ± 0.05 vs x0.98 ± 0.02 control, $p < 0.001$ | **FAIL: too weak** | Ratio of Basin interneuron firing rate under chordotonal mechanosensory drive compared to matched random sensory control. |

*Note*: Both tests are reported honestly as failures without parameter or weight tweaking.

---

## 4. Individuality Assay & Consistency Results

Evaluated on $n = 8$ individual flies across $k = 4$ sessions under `subtle` mode ($\sigma = 0.15$):

### 1. Repeated-Trial Self-Consistency Test (Kain et al. 2012; Linneweber et al. 2020)
- Within-individual Euclidean distance across sessions: **2.675**
- Between-individual Euclidean distance: **2.649**
- One-sided Wilcoxon signed-rank test ($H_1: \text{within} < \text{between}$): **$p = 0.4375$**
- **Consistency Pass**: **False (FAIL)**. Intrinsic membrane noise across repeated sessions produces variance comparable to the subtle individuality gain parameter. Reported honestly.

### 2. Intraclass Correlation Coefficient (ICC(1,1))
- **Steering Bias (DNa01/02 R/L ratio)**: Exp ICC = **+0.097** (Ctrl ICC = -0.293)
- **Looming Latency**: Exp ICC = **+0.068** (Ctrl ICC = +0.174)
- **Sugar Feeding (MN9 activation)**: Exp ICC = **-0.011** (Ctrl ICC = -0.254)
- **T-maze Conditioning PI**: Exp ICC = **-0.098** (Ctrl ICC = -0.038)

### 3. Adult Validated Behaviors Pass Rates Across Individuality Settings (Seeds 1000–1009)
- **Mode `off` ($\sigma = 0.0$)**:
  - Looming $\to$ GF DNp01: **100.0%**
  - Sugar $\to$ MN9: **70.0%**
  - JO-C/E $\to$ aDN: **70.0%**
  - T-maze Conditioning: **100.0%**
- **Mode `subtle` ($\sigma = 0.15$)**:
  - Looming $\to$ GF DNp01: **100.0%**
  - Sugar $\to$ MN9: **80.0%**
  - JO-C/E $\to$ aDN: **30.0%**
  - T-maze Conditioning: **100.0%**
- **Mode `strong` ($\sigma = 0.30$)**:
  - Looming $\to$ GF DNp01: **100.0%**
  - Sugar $\to$ MN9: **80.0%**
  - JO-C/E $\to$ aDN: **30.0%**
  - T-maze Conditioning: **100.0%**

---

## 5. Performance Benchmarks & Concurrency Caps

Measured on Linux, Python 3.14.7, 5-second benchmark runs:

### Single-Fly Throughput
- **Adult Connectome (139,255 neurons)**:
  - CPU (NumPy): 829 steps/s (**4.14x realtime**)
  - Numba JIT: 942 steps/s (**4.71x realtime**)
- **Larva Connectome (2,952 neurons)**:
  - CPU (NumPy): 9,698 steps/s (**161.6x realtime**)
  - Numba JIT: 12,815 steps/s (**213.6x realtime**)

### Multi-Larva Concurrency
- 1 larva: 161.6x realtime (CPU) / 213.6x (Numba)
- 16 larvae: 10.45x realtime (CPU) / 13.82x (Numba)
- 32 larvae: 5.21x realtime (CPU) / 6.88x (Numba)
- 64 larvae: 2.40x realtime (CPU) / 3.39x (Numba)

### Dynamic Fly Caps (`get_max_flies`)
- **Adult**: 16 (CPU) / 32 (Numba) / 64 (GPU).
- **Larva**: **64 (CPU)** / **128 (Numba / GPU)**.

---

## 6. Canonical CONNECTOME vs GAME RULE Tags Added in 2.11

As recorded in the module docstring of `kickthefly/game/kick_the_fly.py`:

1. **Larva Connectome (`CONNECTOME`)**:
   - Complete synaptic-resolution connectivity matrix of the larva brain (2,952 neurons, 352,611 synapses).
   - Tool mappings to real sensory groups: touch $\to$ chordotonal (`mechano-Ch`) & mechanosensory (`mechano-II/III`), heat/blowtorch $\to$ Class IV md nociceptors (`A00c` / `noci`), sugar $\to$ gustatory (`gustatory-external`/`pharyngeal`), cold $\to$ `thermo-cold`.
   - Pain meter directly reads Class IV md nociceptors (tagged `CONNECTOME`).
   - Goro descending command neurons (`_telegoro-1`) mediate rolling escape drive.
2. **Larva Physics (`GAME RULE`)**:
   - Segmented crawler ragdoll physics (`LarvaBody`: 10 segments).
   - Peristaltic crawling wave (1.4 Hz), lateral head-casting angle calculations, and corkscrew rolling mechanics.
   - Restricted arena gating (Open Field, Orchard, Fan, Escape Room).
3. **Individuality (`GAME RULE`)**:
   - $W_{\text{fly}} = D_{\text{post}} \cdot W \cdot D_{\text{pre}}$ log-normal gain distribution ($\sigma = 0.15$ subtle, $0.30$ strong).
   - Personality profile card trait thresholds.
4. **Pet Mode Needs & Catch-Up (`GAME RULE`)**:
   - State machine, file persistence in XDG/Documents, and deterministic wall-clock elapsed time catch-up rule on launch.
   - Clock-tampering clamp guards (negative jumps $\to 0$, jumps $> 7$ days $\to 7$ days).
   - Immortal by default; optional real-stakes starvation / injury death toggle.
5. **Pet Mode Neural Couplings (`CONNECTOME`)**:
   - Hunger scaling sugar taste input gain: $(1.0 + 2.0 \cdot \text{hunger})$.
   - Hunger scaling PAM dopamine reward drive: $(1.0 + 1.5 \cdot \text{hunger})$.
   - Sleep pressure driving dorsal fan-shaped body (dFB) sleep neurons (**FB6/FB7** in adult; premotor resting DNs in larva).
   - Mood computed dynamically from live PAM vs PPL1 firing rate balance.

---

## 7. Cell Types & Annotations Verification

- **Larva Nociceptors**: Annotated in Winding et al. 2023 metadata as `A00c_a4; noci`, `A00c_a5; noci`, `A00c_a6; noci`, and `noci`. Fully verified.
- **Larva Chordotonal**: Annotated as `mechano-Ch`. Fully verified.
- **Larva Mechanosensory**: Annotated as `mechano-II/III`. Fully verified.
- **Larva Basins**: Annotated as `noci 2nd_order PN`, `mechano-Ch 2nd_order PN`. Fully verified.
- **Larva Goro Escape Command Neurons**: Body IDs `5754346` and `3764792`, annotated `_telegoro-1`, `DN-VNC`. Fully verified.
- **Larva Gustatory**: Annotated as `gustatory-external` and `gustatory-pharyngeal`. Fully verified.
- **Unverified Cell Types**:
  - The larva connectome has 2,952 neurons, but lacks specific cell-type annotations for male-specific courtship neurons (`pIP10`, `P1`, `ps1`) and adult visual lobula column projection neurons (`LC4`, `LPLC2`, `LC10`).
  - Optic lobe T4/T5 motion detection neurons do not exist in the larva (larval visual system uses Bolwig's organ photoreceptors, which lack wide-field directional motion detection).
  - All missing adult cell types in the larva connectome are noted and cleanly bypassed in larva mode without substitution.

---

## 8. Invariants & Status

- **Bit-Exactness**: NumPy, Numba, and torch-cpu verified 100% bit-exact across simulation steps, including with individuality enabled (`tests/test_backends.py`, `tests/test_individuality.py`).
- **GPU Safety**: Strictly zero system driver/package modifications; GPU batched paths preserve shared matrix structure and skip cleanly when no GPU is present.
- **Backwards Compatibility**: Existing saves, config.toml, and training memories load without regression.
- **Process Isolation**: Absolutely zero background daemons, services, or timers. Pet mode only steps when the game window is open.
- **Protected File**: `.github/workflows/release.yml` was NOT modified.
