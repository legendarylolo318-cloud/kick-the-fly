# CHANGES_GEMINI.md — Branch `gemini/science-2.9`

This document details all biological modeling extensions, validation results, CONNECTOME vs GAME RULE tags, and architectural additions implemented for Kick the Fly on branch `gemini/science-2.9`.

---

## 1. Summary of Changes & Deliverable

- **Branch**: `gemini/science-2.9`
- **Connectome Dataset**: Janelia MaleCNS v1.0 (166,700 neurons, 10,272,125 synapses)
- **Held-Out Validation Seeds**: 1000–1009 (pass criteria fixed prior to validation execution; exploration on seeds 0–299; no weight tuning).
- **Backend Compatibility**: CPU (NumPy reference), Numba, and PyTorch (torch-cpu) confirmed 100% bit-exact (`tests/test_backends.py` passes).

---

## 2. Modified & Created Files

### Modified Files:
1. `kickthefly/lab/assays.py`:
   - Added assay neuron groups: `pip10` (pIP10 descending courtship neurons), `ps1` (ps1 wing motor neurons), `song_wm` (wing motor neurons), `dng28` (DNg28 Bitter-SEL PN), `co2_orn` (ORN_V olfactory receptor neurons), `co2_pn` (V_ilPN, V_l2PN projection neurons), `trn_vp2` (TRN_VP2 hot cells), `vp2_pn` (VP2 warming projection neurons), `trn_vp3` (TRN_VP3a/b cold cells), `vp3_pn` (VP3 cooling projection neurons), `optomotor_right` (T4a/T5a right + T4b/T5b left wide-field right yaw detectors), and `dna_steer_r` (DNa01_R and DNa02_R steering descending neurons).
2. `kickthefly/lab/validation.py`:
   - Updated `_pathway_seed()` to support matched taste controls (`control in ("bitter", "sweet")`).
   - Added critical path diagnosis note to `mdn_backward` documenting the VNC feedforward inhibition failure mechanism without parameter tuning.
   - Appended 7 new validation tests to `TESTS`: `courtship_song`, `bitter_avoidance`, `co2_avoidance`, `thermosensory_hot`, `thermosensory_cold`, `grooming_hierarchy`, and `optomotor_turning`.
   - Updated `EXPECTED` mapping with held-out results for all 15 validation tests.
3. `kickthefly/lab/lab.py`:
   - Added threshold controls `thresh.song`, `thresh.aggression`, `thresh.sleep` to `PARAMS`.
   - Registered 5 new entries in `ASSUMPTIONS` (Model Assumptions page) documenting courtship song audio, multi-fly aggression, EMD motion detection, thermotaxis arena gradient, and circadian day/night & sleep.
4. `kickthefly/game/kick_the_fly.py`:
   - Updated top canonical docstring documenting all CONNECTOME vs GAME RULE additions and new validation results.
   - Dynamically registered `"thermo"` arena in `config.BY_KEY["brain.arena"]` options/labels and added `"thermo"` to `ARENAS`.
   - Added thresholds `"song": 1.5`, `"aggression": 2.0`, `"sleep": 2.0` to `THRESH`.
   - Added `Brain` detail groups for `"song"`, `"aggression"`, `"sleep"`, and `"co2"`.
   - Implemented procedural pulse-song buzz synthesis in `Sound._make()` (`"pulse_song"`, 220 Hz oscillation, 35 ms IPI).
   - Added courtship song audio playback and sensory event logging in `Game.readouts()`.
   - Implemented multi-fly aggression lunge reaction for $N > 1$ flies in `Game._movement()` and collision routines.
   - Implemented `"thermo"` arena floor temperature gradient in `Game._environment_one()` (driving `TRN_VP3` on left and `TRN_VP2` on right) and visual rendering in `Game._draw_arena_back()`.
   - Updated `REACTION_SOURCE` and `POPUP_SOURCE` tags.
5. `kickthefly/game/outdoors.py`:
   - Documented motion and circadian CONNECTOME vs GAME RULE tags in module docstring.
   - Implemented `emd_motion_drive()` (Reichardt elementary motion detector filter).
   - Implemented `diurnal_cycle()`, `circadian_clock_drive()`, and `dfb_sleep_state()`.
6. `tests/test_validation.py`:
   - Expanded parametrized test list in `test_matches_expected` to cover all 15 validation tests.

### Created Files:
1. `tests/test_science_2_9.py`:
   - Test suite verifying courtship song groups & audio, aggression circuit & tags, optomotor EMD & groups, thermotaxis arena & groups, circadian cycle & dFB sleep readout, protocol file loading, and model assumptions registry.
2. Protocol YAML Examples:
   - `protocols/courtship_song.yaml`: pIP10 activation driving ps1 wing motor neurons.
   - `protocols/male_aggression.yaml`: FruM/TK neuron activation and pC1 cluster recordings.
   - `protocols/optomotor.yaml`: Wide-field progressive/regressive motion driving DNa01/DNa02.
   - `protocols/thermotaxis.yaml`: Antennal hot (TRN_VP2) and cold (TRN_VP3) activation and projection neuron readouts.
   - `protocols/bitter_avoidance.yaml`: Bitter GRNs driving DNg28 (Bitter-SEL).
   - `protocols/co2_avoidance.yaml`: Antennal ORN_V driving V glomerulus projection neurons.
   - `protocols/grooming_hierarchy.yaml`: Anterior command aDN1/aDN2 driving front vs hind leg motor neurons.
   - `protocols/day_night_cycle.yaml`: Photic drive to circadian clock neurons and dFB sleep circuits.

---

## 3. Validation Results (Seeds 1000–1009)

All tests run on the held-out seed set 1000–1009 with criteria fixed in advance (ratio $\ge 1.5$, one-sided Wilcoxon $p < 0.01$).

| Test ID | Name | Drive | Readout | Measured (Drive vs Control) | p-value | Result | Citation |
|---|---|---|---|---|---|---|---|
| `courtship_song` | Courtship song wing motor excitation | pIP10 (2) | ps1 MN (2) | $1.68\times \pm 0.17$ vs $0.81\times \pm 0.15$ (base: 6.7 Hz, drv: 11.1 Hz) | $0.00098$ | **PASS** | von Philipsborn et al. 2011, Nat Neurosci 14:1413 |
| `bitter_avoidance` | Bitter taste second-order excitation | Bitter GRNs (30) | DNg28 Bitter-SEL (2) | $1.58\times \pm 0.26$ vs $1.28\times \pm 0.21$ (base: 7.4 Hz, drv: 11.5 Hz) | $0.00098$ | **PASS** | Yao & Scott 2022, Neuron 110:4098; Shiu et al. 2024 |
| `co2_avoidance` | CO2 olfactory receptor to V glomerulus | ORN_V (55) | V glomerulus PNs (4) | $1.92\times \pm 0.11$ vs $0.87\times \pm 0.06$ (base: 27.5 Hz, drv: 52.6 Hz) | $0.00098$ | **PASS** | Suh et al. 2004, Cell 119:173; Lin et al. 2013 |
| `thermosensory_hot` | Antennal warming to VP2 PNs | TRN_VP2 (7) | VP2 PNs (15) | $1.62\times \pm 0.11$ vs $0.85\times \pm 0.04$ (base: 30.0 Hz, drv: 48.5 Hz) | $0.00098$ | **PASS** | Gallio et al. 2011, Cell 144:614; Frank et al. 2015 |
| `thermosensory_cold` | Antennal cooling to VP3 PNs | TRN_VP3a/b (7) | VP3 PNs (14) | $1.56\times \pm 0.06$ vs $0.86\times \pm 0.05$ (base: 38.1 Hz, drv: 59.3 Hz) | $0.00098$ | **PASS** | Gallio et al. 2011, Cell 144:614; Alpert et al. 2020 |
| `grooming_hierarchy` | Anterior grooming command to hind motor | aDN1/aDN2 (4) | mn_hind T3 (126) | $1.01\times \pm 0.06$ vs $0.88\times \pm 0.06$ (base: 4.1 Hz, drv: 4.2 Hz) | $0.00098$ | **FAIL** | Seeds et al. 2014, eLife 3:e02951; Hampel et al. 2015 |
| `optomotor_turning` | Wide-field motion to steering DNs | T4/T5 R-yaw (532) | DNa01_R/DNa02_R (2) | $0.87\times \pm 0.14$ vs $2.41\times \pm 0.50$ (base: 5.1 Hz, drv: 4.4 Hz) | $1.00000$ | **FAIL** | Borst et al. 2020, J Neurogenet 34:331; Rayshubskiy 2020 |
| `mdn_backward` | Moonwalker backward walking motor drive | MDN (4) | mn_legs T1-T3 (381) | $0.89\times \pm 0.09$ vs $1.15\times \pm 0.14$ | $0.99902$ | **FAIL** | Bidaye et al. 2014, Science 344:97 |
| `looming_escape` | Looming detectors drive giant fiber | LPLC2+LC4 (311) | DNp01 (1) | $9.82\times \pm 1.24$ vs $0.83\times \pm 0.06$ | $0.00098$ | **PASS** | von Reyn 2014; Ache 2019 |
| `sugar_feeding` | Sugar taste neurons drive MN9 | Sweet GRNs (30) | MN9 (1) | $2.14\times \pm 0.31$ vs $1.25\times \pm 0.18$ | $0.00098$ | **PASS** | Shiu et al. 2024; Yao & Scott 2022 |
| `antenna_grooming_circuit` | Antennal mechanosensory to aDN | JO-C/E (335) | aDN1/aDN2 (4) | $4.88\times \pm 0.42$ vs $0.85\times \pm 0.05$ | $0.00098$ | **PASS** | Hampel et al. 2015 |
| `adn_grooming_motor` | aDN to front-leg motor neurons | aDN1/aDN2 (4) | mn_front T1 (135) | $1.13\times \pm 0.08$ vs $1.10\times \pm 0.12$ | $0.37500$ | **FAIL** | Hampel et al. 2015 |
| `mb_conditioning` | Mushroom body T-maze conditioning | Odor + shock | T-maze choice | PI $0.62 \pm 0.08$ vs unpaired $-0.02 \pm 0.06$ | $0.00098$ | **PASS** | Tully & Quinn 1985 |
| `epg_compass` | Central complex ring attractor bump | Wedge drive | EPG (46) | Contrast $1.72\times$, persist 84 ms | — | **FAIL** | Seelig & Jayaraman 2015; Green 2017 |
| `epg_compass_wind` | E-PG compass from directional wind | Steady wind | EPG (46) | Contrast $1.81\times$, persist 101 ms | $0.17$ | **FAIL** | Okubo et al. 2020 |

---

## 4. Task 3: MDN Failure Diagnosis (Critical Path Findings)

- **Interneuron Layer Activation**: MDN activation (4 neurons, descending) successfully reaches 330 VNC local interneurons. Overall firing in this interneuron layer increases by $1.40\times$ (from $7.11$ to $9.93$ Hz). High-ranking bridge interneurons (such as `LBL40`, `IN03A010`, `IN21A022`) increase firing by $2.75\times$.
- **Direct Connectivity**: MDN makes only **1 single direct synapse** across all 381 leg motor neurons (`mn_legs`).
- **Feedforward Inhibition in VNC**: From the VNC interneuron layer to `mn_legs`, MDN recruits balanced feedforward inhibition: 832 inhibitory synapses (weight sum $-8.52$) vs 1,198 excitatory synapses (weight sum $+9.82$) via GABAergic/glutamatergic interneurons (e.g. `IN12B003`).
- **Motor Neuron Outcome**: Across the 381 leg motor neurons, 197 are suppressed below baseline while 132 are excited, yielding a net whole-pool ratio of $0.89\times$.
- **Conclusion**: The signal dies due to dispersed feedforward inhibition in the unweighted VNC interneuron layer. The connectome wiring alone lacks the tuned synaptic weighting or rhythmogenic central pattern generators needed to coordinate backward locomotion. Per the hard rules, no parameters were tuned to force a pass; this is documented as an honest negative validation finding.

---

## 5. Canonical CONNECTOME vs GAME RULE Tags

Every added feature is tagged across the top docstring in `kick_the_fly.py`, `Settings`, and the Lab `Model Assumptions` page:

1. **Courtship Song**:
   - `CONNECTOME`: Male-specific descending command neuron `pIP10` innervates pleurosternal motor neuron `ps1 MN` (verified in MaleCNS v1.0, 2 neurons each).
   - `GAME RULE`: Procedural synthesis of the courtship pulse-song buzz audio waveform (220 Hz carrier, 35 ms inter-pulse interval, 12 ms pulse width) played when the `ps1` readout crosses threshold.
2. **Male–Male Aggression**:
   - `CONNECTOME`: Fruitless/Tachykinin neurons (`AVLP727m`, 5 neurons, annotated `Asahina 2014: TK-FruM`) and Fruitless male cluster `pC1` (156 neurons). Inter-fly contact is mediated strictly via real visual looming (LPLC2/LC4) and tactile (SNta, SNpp) pathways.
   - `GAME RULE`: For $N > 1$ flies, threshold crossing triggers a scripted lunge/fight impulse directed toward the nearest fly upon looming or physical collision. Tagged `RULE` in `REACTION_SOURCE` and `POPUP_SOURCE`.
3. **Optomotor Response**:
   - `CONNECTOME`: Direction-selective visual neurons `T4a`/`T5a` (progressive front-to-back) and `T4b`/`T5b` (regressive back-to-front); steering descending command neurons `DNa01`/`DNa02`.
   - `GAME RULE`: Reichardt elementary motion detector (EMD) filter stage converting rotational visual angular velocity into progressive and regressive input drive to T4/T5.
4. **Thermotaxis Arena ("thermo")**:
   - `CONNECTOME`: Real hot-sensing antennal neurons `TRN_VP2` (7 neurons) and cold-sensing antennal neurons `TRN_VP3a/b` (7 neurons).
   - `GAME RULE`: Horizontal spatial floor temperature gradient mapping floor coordinates ($x < 380$ cold, $x > 510$ hot) to current injection into antennal thermosensors.
5. **Day/Night Diurnal Cycle & Sleep**:
   - `CONNECTOME`: Photoreceptors R1-R8, circadian clock neurons (`l-LNv`, `s-LNv`, `LNd`, `DN1`), and dorsal fan-shaped body (dFB) neurons (`FB6`, `FB7`).
   - `GAME RULE`: Orbital 24-hr sun elevation cycle ($+60^\circ$ noon to $-30^\circ$ midnight) and thresholded dFB activity state readout mapping elevated dFB firing to quiet sleep.

---

## 6. Cell Type Verifications in MaleCNS v1.0

- **Courtship Song**:
  - `pIP10`: Verified (2 neurons, descending).
  - `ps1 MN`: Verified (`vnc_motor`, subclass `wm`, 2 neurons).
- **Male–Male Aggression**:
  - `AVLP727m`: Verified (5 neurons, `cb_intrinsic`). Synonym column explicitly records `Asahina 2014: TK-FruM`.
  - `P1a`: **Unannotated / Absent in MaleCNS v1.0**. The dataset does not contain `P1a` as a type name or synonym; the Fruitless male cluster is annotated as `pC1` (`pC1_1a` through `pC1_19`, `pC1x_*`, 156 neurons). Both `AVLP727m` and `pC1` were utilized.
- **CO2 Pathway**:
  - `ORN_V`: Verified (55 neurons, `cb_sensory`).
  - `V_ilPN` & `V_l2PN`: Verified (4 neurons total, projection neurons from V glomerulus).
- **Thermosensory Antennal Neurons**:
  - `TRN_VP2`: Verified (7 neurons, `cb_sensory`, warming cells).
  - `VP2_adPN`, `VP2_l2PN`, `VP2+_adPN`, `VP1m+VP2_lvPN1/2`: Verified (15 neurons, warming projection neurons).
  - `TRN_VP3a`, `TRN_VP3b`: Verified (7 neurons, `cb_sensory`, cooling cells).
  - `VP3+_l2PN`, `VP3+_vPN`, `VP1l+VP3_ilPN`, `VP3+VP1l_ivPN`, `VP5+VP3_l2PN`: Verified (14 neurons, cooling projection neurons).
- **Grooming Hierarchy**:
  - `aDN1` (`DNg62`) & `aDN2` (`DNge078`): Verified (4 neurons, descending).
  - `mn_front`: 135 motor neurons (`vnc_motor`, subclass `fl`).
  - `mn_hind`: 126 motor neurons (`vnc_motor`, subclass `hl`).
- **Optomotor Pathways**:
  - Directional cells `T4a`, `T5a`, `T4b`, `T5b`: Verified (subclasses `ol_intrinsic`, instances with `_L` and `_R` suffixes).
  - Steering command DNs `DNa01`, `DNa02`: Verified (instances `_L` and `_R`).
- **Circadian Clock & Sleep**:
  - Clock neurons: `l-LNv` (4), `s-LNv` (4), `5thsLNv_LNd6` (2), `LNd_b` (2), `LNd_c` (2), `DN1a` (4), `DN1pA` (16), `DN1pB` (8).
  - Sleep neurons: `FB6A`–`FB6D`, `FB7A`–`FB7B` (16 neurons, `cb_intrinsic`, dorsal fan-shaped body).

---

## 7. Verification & Test Suite Status

- `pytest tests/test_backends.py`: **15 passed, 6 skipped** (GPU backends skipped; CPU, Numba, torch-cpu bit-exactness fully verified).
- `pytest tests/test_validation.py`: **16 passed in 431s** (All 15 validation tests matched `EXPECTED`, popup event filtering verified).
- `pytest tests/test_science_2_9.py`: **7 passed in 1.91s** (Comprehensive test suite for all new features).
- `pytest tests/test_outdoors.py`: **14 passed in 20.82s** (2D/3D compatibility verified).
- All 8 new/updated protocol YAML files validated via `protocol.load()`.
