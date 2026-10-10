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
Done and committed since the arcade: arcade page + battle + game hookup, multiple flies (N spawn, F focus, J recall; saves hold every fly), share codes apply
(surgery, loadout, challenge, contraption slot, protocol file), `--flies`, 2D game decision (not ported, written), validation engine (21 tests, 12 PASS / 9 FAIL as
Python; full run equals the published results except one control response), Lab hub / parameters (live) / assumptions / validation / assays / asymmetry / benchmark pages
matched with pygame, `fly_sim::mirror` (Mirror-weights rule applied to every fly via `Command::SetMirror`; learning restarts on a flip).
Engines in `fly_sim`: `duel, cards, race, tournament, stats, validation, assay_jobs, benchmark, mirror`. Test goldens: `duel_golden, arcade_real, stats_golden,
bracket_golden, validation_real (+ #[ignore] full_suite_equals_the_published_python_run, 22 min), assay_jobs_golden, mirror_real`.
Waiting: never use `pgrep` loops; read the task output file (see memory note).
Git identity is not set in this checkout: commit with `git -c user.name=Logan -c user.email=legendarylolo318@gmail.com commit`.

## Next, in order (what is still `[ ]` in PORTING.md section 16 and the Lab list)
Each of these is a read-the-Python, port-the-engine, test-against-a-golden, draw-the-page, compare-with-pygame cycle:
1. Lab pages that need small engines: `lab_minipapers` (`lab/minipapers.py`, 487 lines), `lab_classroom`, `lab_netsci` (`lab/netsci.py`, 692), `lab_sleepdep`
   (`sleepdep.py`), `lab_sensitivity` (`sensitivity.py`, `robustness.py`), `lab_wiring` (`labwiring.py`), `lab_critical`, `lab_clamp`, `lab_diff`, `lab_laser` (`laser.py`),
   `lab_psych` (`psychometrics.py`), `lab_autopsy`.
2. Toolkit pages in `lab/labtoolkit.py`: genetics, thermo (`thermogenetics.py`), patch (`patchclamp.py`), imaging, pharm (`pharmacology.py`); rigs (`labrigs.py`,
   `rigassay.py`); screens (`screens.py`, `labscreen.py`: activation/knockout).
3. `lab_export` + `lab_protocols`: need the recorder (`recorder.py`: CSV/npz + metadata JSON; NWB needs HDF5, report "not available" like Python's `nwbexport.available()`),
   the YAML protocol loader/runner (`protocol.py`, 883 lines; protocols are JSON-as-YAML here, so a YAML reader is needed to run the 42 shipped files), bundles (`bundle.py`).
4. The five assay kinds not in `assay_jobs`: orchard, thermo_escape, predator_escape, hum_demo, sleep_deprivation; larva validation tests; `--validate` / `--headless`.
5. UI leftovers: `live_inputs`, remaining arcade pad navigation, photo mode, neuron search + path tracer, UI scale, pet mode (+ `pet.ktfsave`).
6. Multi-fly leftovers: contraption wind / memory scent pokes go to the focused brain; arcade/tool wheel/puppeteer assume one fly; personality card not on bodies.
7. Finish: README, tick PORTING.md boxes only when verified, clean `cargo build/test/clippy`, commit.

## Known deviations (written in PORTING.md)
FreeSans font (check redistribution), linear-light alpha LUT, glyph AA, 3D brain cloud and whole-window big view instead of the flat image, Neuron of the Day uses UTC,
times in reports are UTC, Python can't restore a Rust save's fly body, Rust kill cam auto-plays on death (Python offers it on the autopsy card).
