# continue.md: where the Bevy port stands (read this, PORTING.md, then work)

Branch `bevy-port`. Task = the user's 7-point brief (exact parity with the pygame game incl. GUI; UI; Lab tools; serde persistence; real neuron shapes;
packaging + CI; README). Tick PORTING.md only for verified things; unmatched things get a written reason there. Commit with
`git -c user.name=Logan -c user.email=legendarylolo318@gmail.com commit` and end messages with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
The user is on Wayland tiling: **run the game with `--fullscreen`** (memory notes: kick-the-fly-bevy-port, feedback-fullscreen-wayland-tiling).

## How to verify (keep using these)
- Venv for the Python references (recreate if /tmp was cleaned): `python3 -m venv V; V/bin/pip install numpy scipy pygame-ce pytest pillow pyyaml pyarrow moderngl fonttools`
  and `ln -s ../../data/kick_brain.npz legacy-python/data/kick_brain.npz` (untracked; do not commit `legacy-python/data/`). Run scripts with `SDL_VIDEODRIVER=dummy`.
- Menu pages: `legacy-python/make_ui_reference.py` -> `tests/ui_reference/*.png`, compared in `crates/ktf_ui/tests/reference.rs` (< ~1 % clearly different pixels).
- HUD-style panels (challenges, surgery, autopsy, wheel): `legacy-python/make_challenge_reference.py` -> `tests/ui_reference/challenge_*.png`; Rust tests in
  `crates/game/src/{challenges,autopsy_play,wheel_play}.rs` draw the recorded `ktf_ui::hud::Op` lists and compare.
- Goldens (`tests/golden/make_*.py`, written with the real Python code on the real pack): simulator, groups, pain, neurodex, **memory, assay_groups, surgery, autopsy**
  (tests in `crates/fly_sim/tests/{golden,memory_golden,training_real,challenge_real,autopsy_golden}.rs`).
- In-game screenshots: `kick_the_fly --fullscreen --shot out.png --shot-frames 400` plus `--menu pause|settings:Audio|selftest|bugreport`, `--train-screen`,
  `--surgery-screen`, `--challenge tmaze|sneak|sweet|reverse_surgery|predict`, `--inspect ROW` (with `KTF_BIG_BRAIN=1`), `--kill-at S` (autopsy), `--arena NAME`.
  Set `KICK_THE_FLY_HOME=<tmp>` for a clean config (a config with `neuron_shapes = true` makes the shape download fire).
- Before committing: `cargo fmt --all` and `cargo clippy -p ktf_data -p ktf_ui -p fly_sim -p kick_the_fly --all-targets -- -D warnings`, then `cargo test -p ktf_data -p ktf_ui -p fly_sim -p kick_the_fly`.
  Bevy systems may take at most 16 params (use `SystemParam` bundles, see `ClickCtx`, `MenuCtx`, `HudIn`).

## Architecture notes added this session
- `ktf_data::diag` (self-test report, bug report, crash text, log files), `ktf_data::shapes` (SWC, shape cache).
- `fly_sim`: `memory` (plasticity), `training`, `challenge` (T-maze, sweet), `assays` (sweet/bitter groups), `surgery` (23 entries), `autopsy`, `inspect`.
- `game`: `selftest`, `crash` (panic hook, `--crash-screen` window), `memory_play`, `challenges`, `surgery_play`, `autopsy_play`, `wheel_play`,
  `stethoscope_play`, `inspector_play`, `shapes_play`. Overlay panels are lists of `ktf_ui::hud::Op` (Card, Text, Button, Hit, Icon, ...) that the HUD replays;
  `HudState.challenge` carries whichever overlay is up (wheel > autopsy > surgery/inspector > challenge).
- The arcade (Bevy-text overlay in `arcade.rs`) now opens from Esc > Fly arcade; **O is brain surgery**, as in Python.
- The sim thread owns learning, training, challenges, surgery, stethoscope counting, autopsy and the inspector lookups (commands/events in `sim_thread.rs`).

## Done (all committed unless noted)
Packaging verified (AppImage, Windows type-check); self-test/bug report/crash; mushroom-body learning end to end; 5 challenges + page; brain surgery; autopsy; tool wheel;
stethoscope; arena status text; neuron inspector; real shapes (last piece: Neurodex skeleton drawing, inspector thumbnail, gizmo overlay, reset on opt-in; tested incl. one real
download). Everything above has its PORTING.md entry with the verification and the Python defects found (TEST trains liking, Mystery defect silences nothing,
Predict KeyError, x-goog-hash first-header bug, stethoscope central_complex/optic_lobes silent).

## Session 2026-10-10 (day 2) status
Everything below is committed; `cargo build/test/clippy` were clean at the last full run (the long goldens are `--release` tests, see below).

**Done since the arcade** (engine in `fly_sim`, page in `ktf_ui::pages`, game side in `game::labtools`, each equal to Python on the real pack):
arcade + battle, multi-fly, share codes, validation, assays, asymmetry (+ mirror weights), benchmark, **netsci, sleepdep, sensitivity, wiring (robustness), critical path, neural clamp**; the diff-mode *engine*.
Goldens (Python, `tests/golden/make_*.py` -> json; Rust tests are `crates/fly_sim/tests/*_real.rs`, run with `cargo test -p fly_sim --release --test NAME`):
netsci (3 min), sleepdep, sensitivity, robustness, critical, clamp, diff. Page references: `legacy-python/make_lab_reference.py`
(needs the venv: `python3 -m venv V; V/bin/pip install numpy scipy pygame-ce pytest pillow pyyaml pyarrow moderngl fonttools`, run with `SDL_VIDEODRIVER=dummy`).

**How a Lab tool page is wired** (copy `pages/critical.rs` + `game/src/labtools.rs`): page queries `lab.<name>_state`, sends commands `lab.<verb>`;
`LabTools` owns one `BgJob` slot, results, exports; `GameHost::with_tools` carries both. Gotchas learned: `serde_json` sorts map keys
(so carry an explicit `order` array where Python's insertion order is drawn; do NOT enable `preserve_order`, the stores rely on sorted keys);
the Python `labclamp.py` and `labdiff.py` forget `import numpy as np` (their lesion options crash there; ours run); commit with
`git -c user.name=Logan -c user.email=legendarylolo318@gmail.com commit`.

## Next, in order
1. (done) `lab_diff` page.
2. Remaining Lab pages: (`lab_laser`, `lab_psych`, `lab_classroom` done)
   (`lab_minipapers` done) (`lab_activation` / `lab_knockout` done).
3. Toolkit (`lab_genetics`, `lab_thermo`, `lab_pharm`, `lab_patch`, `lab_imaging` done; patch lacks NWB export + inspector shortcut, imaging lacks TIFF/NWB + keep frames) (`imaging.py` 533, `livelab.ImagingLive`); `lab_rigs` done.
4. (DONE: `lab_export`, `lab_protocols`, recorder, runner, bundles; the CLI entry points are with `--headless`) recorder (CSV/npz + metadata; NWB unavailable like Python's `nwbexport.available()`), YAML protocol runner (`protocol.py`, 883 lines;
   needs a YAML reader for the 42 shipped files), bundles (`bundle.py`).
5. Five assay kinds missing from `assay_jobs` (orchard, thermo_escape, predator_escape, hum_demo, sleep_deprivation as a labjobs kind), larva validation tests,
   `--headless`/`--validate`, pet mode, photo mode, `live_inputs`, neuron search.
6. Multi-fly leftovers (contraption wind / memory scent pokes go to the focused brain; personality card not on bodies).
7. Finish: README, tick PORTING.md only for verified boxes, clean `cargo build/test/clippy`, commit.

## Known deviations (written in PORTING.md)
FreeSans font (check redistribution), linear-light alpha LUT, glyph AA, 3D brain cloud and whole-window big view instead of the flat image, Neuron of the Day uses UTC,
times in reports are UTC, Python can't restore a Rust save's fly body, Rust kill cam auto-plays on death (Python offers it on the autopsy card).
