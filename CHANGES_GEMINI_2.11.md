# Changes in Kick the Fly 2.11 (Gemini)

This document details all changes implemented on branch `gemini/2.11` starting from tag `v2.10.0` (commit `3a2fc9d`). Every claim is tied to an automated test or command execution output.

---

## 1. Overview of Changes

### Task 1: Decoy Female in 3D & 2D
- **3D Fly Model Rendering (`kickthefly/game/kick3d.py`)**:
  - Implemented `_draw_decoy3d()` using the game's fly model components from `render3d.py`.
  - Accurately renders female morphology: larger, rounder abdomen with the female banded pattern (`P_STRIPES`), pointed light-colored ovipositor, legs without male sex combs, and a distinctive decoy tint `(200, 180, 220)`.
  - Adds ground shadow projection matching female proportions.
  - Decoys render in all arenas (`room`, `fan`, `flypaper`, `pool`, `lamp`, `escaperoom`, `thermo`, `field`, `orchard`) and in photo mode via `_draw_extras()`.
- **2D Fly Model Rendering (`kickthefly/game/kick_the_fly.py`)**:
  - Upgraded 2D decoy rendering from simple placeholder circles to detailed female morphology: 32x21 banded abdomen, light ovipositor tip, segmented thorax, red compound eye, antenna, wings, legs without sex combs, decoy tint `(200, 180, 220)`, and ground shadow.
- **Decoy Lifecycle, Controls, Limits & State**:
  - **Reset (R)**: Clears all dropped decoys (`self.decoys = []`, `self.decoys3 = []`).
  - **Arena Change**: Clears all decoys on arena switch (`on_arena_changed()`).
  - **Save State Roundtrip**: Saves and restores decoys backwards-compatibly in `save_extra()` and `load_extra()` in both 2D and 3D.
  - **Hand Tool Pick-up, Throw, and Removal**: Left click with the hand tool grabs a decoy within reach, tracks the hold position in real time, releases/throws on mouse button release, and right-click removes the decoy with a `DECOY REMOVED` popup and `pop` sound.
  - **Cap**: Enforces maximum of 3 decoys; attempting to drop a 4th decoy displays an on-screen notice (`MAX 3 DECOYS` popup with `force=True`, log note `DECOY limit reached (max 3)`).

### Task 2: Live Foreleg GRN and P1 Contact HUD Readout
- **Connectome Wiring & Activation (`kickthefly/game/kick_the_fly.py`, `kickthefly/game/kick3d.py`)**:
  - In `Brain`: Added tracking of foreleg gustatory receptor neurons `LgLG5-8` (putative ppk23/ppk25, 64 neurons) and male courtship cluster `P1`.
  - Added public level query methods: `br.lglg_level()` and `br.p1_level()`, tracking instantaneous exponential moving average firing relative to calm baseline.
  - Foreleg contact within 0.4 m in 3D and 45 px in 2D stimulates `("pheromone", "foreleg")` and activates `fly.decoy_contact_until`.
- **HUD Readout Display**:
  - While contact is active, displays a dedicated HUD banner at `y=88` (positioned cleanly below the top arena header):
    `DECOY CONTACT: LgLG5-8 x... P1 x... calm [REAL]`
  - Explicitly labeled with the blue `[REAL]` chip because the neural firing readout reflects live simulated connectome activity.
  - The `COURTSHIP` tag triggered on contact remains labeled `[RULE]`. No artificial motor rule was added to force courtship.

### Task 3: Visible cVA Reach Cloud (2D and 3D)
- **Visible Pheromone Cloud**:
  - 2D: Expands to full reach of 250 px (`self.cva_puffs` with `reach: 250.0`, rendering translucent expanding concentric waves).
  - 3D: Expands to full reach of 2.5 m (`self.cva_puffs3` with `reach: 2.5`, rendering particle clouds and expanding reach sphere).
- **Assumptions & Documentation**:
  - Documented as a `GAME RULE` in `kickthefly/lab/lab.py:ASSUMPTIONS`.
  - Documented in `README.md` under the Game Rules list and the cVA pheromone tool paragraph.
  - Documented in the canonical docstring of `kickthefly/game/kick_the_fly.py`.

### Task 4: Screenshot Scene & Documentation Asset
- **Scene "decoy" (`tools/make_screenshots.py`)**:
  - Added scene `"decoy"`: immortal fly walking into a decoy female in the 3D room, framed from a three-quarters side angle to show both the male fly and female decoy morphology, with active contact and the live HUD readout on screen.
  - Rendered offscreen with GPU context to `docs/decoy.png` (162 KB).
  - Referenced `docs/decoy.png` in `README.md` in the decoy paragraph.

---

## 2. Modified and Created Files

| File | Status | Description |
|---|---|---|
| `kickthefly/game/kick_the_fly.py` | Modified | Added 2D female decoy drawing, live contact HUD with `REAL` chip, `LgLG5-8` and `P1` levels in `Brain`, decoy save/load/reset/removal, max 3 limit, 2D cVA reach cloud, and docstring updates. |
| `kickthefly/game/kick3d.py` | Modified | Added 3D female decoy model rendering with `render3d.py` components, contact detection, hand tool grab/throw/removal, decoy save/load/reset, max 3 limit, 3D cVA reach cloud, and `handle()` wrapper. |
| `kickthefly/lab/lab.py` | Modified | Updated `ASSUMPTIONS` to document the cVA visible reach cloud as `GAME RULE` and decoy contact readout as `REAL` (with `COURTSHIP` remaining `RULE`). |
| `README.md` | Modified | Updated cVA and decoy paragraphs, linked `docs/decoy.png`, updated Game Rules list. |
| `tools/make_screenshots.py` | Modified | Added `"decoy"` screenshot scene. |
| `docs/decoy.png` | Created | Offscreen-rendered screenshot showing male fly and female decoy in contact in the 3D room with the live HUD line. |
| `tests/test_decoy.py` | Created | Comprehensive test suite for 3D render items, LgLG5-8 contact activation in 2D & 3D, reset clearance, arena change clearance, save states, hand tool manipulation, drop limits, and cVA reach. |
| `CHANGES_GEMINI_2.11.md` | Created | This document. |

---

## 3. Verification & Test Execution Results

All commands were executed inside the local virtual environment `.venv` on this machine.

### Decoy Screenshot Generation
Command:
```bash
.venv/bin/python tools/make_screenshots.py decoy
```
Output:
```
pygame-ce 2.5.8 (SDL 2.32.10, Python 3.14.7)
decoy: /home/lolo/kick-the-fly/docs/decoy.png (162 KB, 12 s)
```

### Decoy & cVA Unit Tests
Command:
```bash
.venv/bin/python -m pytest tests/test_decoy.py
```
Output:
```
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/lolo/kick-the-fly
configfile: pytest.ini
collecting ... collecting 10 items                                                            collected 10 items                                                             

tests/test_decoy.py ..........                                           [100%]

============================= 10 passed in 39.79s ==============================
```

### Full Non-Validation Test Suite
Command:
```bash
.venv/bin/python -m pytest -m "not validation"
```
Output:
```
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/lolo/kick-the-fly
configfile: pytest.ini
testpaths: tests
collecting ... collecting 42 items                                                            collected 299 items / 24 deselected / 1 skipped / 275 selected                 

tests/test_api.py ..                                                     [  0%]
tests/test_asymmetry.py ...                                              [  1%]
tests/test_backends.py .............................ss...s               [ 14%]
tests/test_benchmark.py ..                                               [ 15%]
tests/test_clamp.py ..                                                   [ 16%]
tests/test_classroom.py ........                                         [ 18%]
tests/test_compass.py ...                                                [ 20%]
tests/test_compat.py ..                                                  [ 20%]
tests/test_config.py ..........                                          [ 24%]
tests/test_criticalpath.py .....                                         [ 26%]
tests/test_decoy.py ..........                                           [ 29%]
tests/test_determinism.py ....                                           [ 31%]
tests/test_diffmode.py ..                                                [ 32%]
tests/test_drop_item.py ....                                             [ 33%]
tests/test_escaperoom.py ........                                        [ 36%]
tests/test_gamepad.py ......                                             [ 38%]
tests/test_hemifield.py ...                                              [ 39%]
tests/test_i18n.py .......                                               [ 42%]
tests/test_lab.py ................                                       [ 48%]
tests/test_labpages.py ..................                                [ 54%]
tests/test_laser.py .....                                                [ 56%]
tests/test_layout.py .....                                               [ 58%]
tests/test_modes.py ....                                                 [ 59%]
tests/test_morphology.py .....                                           [ 61%]
tests/test_neurosearch.py ....                                           [ 62%]
tests/test_outdoors.py ..............                                    [ 68%]
tests/test_paths.py ...........ss                                        [ 72%]
tests/test_photo_mode.py ...                                             [ 73%]
tests/test_predict_neuron.py ...                                         [ 74%]
tests/test_psychometrics.py ...                                          [ 76%]
tests/test_replay.py ......                                              [ 78%]
tests/test_reverse_surgery.py ..                                         [ 78%]
tests/test_science_2_9.py ..........                                     [ 82%]
tests/test_stethoscope.py ...                                            [ 83%]
tests/test_swarm.py .....                                                [ 85%]
tests/test_timelapse.py ...                                              [ 86%]
tests/test_tooltip.py ...............                                    [ 92%]
tests/test_video_record.py .........                                     [ 95%]
tests/test_wiring.py .............                                       [100%]

==== 270 passed, 6 skipped, 24 deselected, 2 warnings in 518.48s (0:08:38) =====
```

### Full Validation Test Suite
Command:
```bash
.venv/bin/python -m pytest -m validation
```
Output:
```
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/lolo/kick-the-fly
configfile: pytest.ini
testpaths: tests
collecting ... collecting 42 items                                                            collected 299 items / 275 deselected / 1 skipped / 24 selected                 

tests/test_backends.py ..                                                [  8%]
tests/test_validation.py ......................                          [100%]

========== 24 passed, 1 skipped, 275 deselected in 638.89s (0:10:38) ===========
```

---

## 4. Unfinished Items
None. All 4 requested tasks and test suites are complete, fully tested, documented, and committed.
