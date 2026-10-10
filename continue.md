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

## Next, in order
1. Commit anything uncommitted (`git status`); the real-shapes commit may still be pending.
2. Remaining UI pages (port line by line from `legacy-python/kickthefly/ui/*.py`, `lab/*.py`): `arcade` + battle (`arcade_ui.py`, `battle_ui.py`; needs personality cards
   `core/cards.py` and replay frames from the Rust tournament/race engines), `live_inputs` (mic + streamer), `lab_minipapers`, Lab hub + ~20 `lab_*` pages.
   Also: duel card, photo mode (F10), neuron search + path tracer (`lab/neurosearch.py`), pad navigation, UI scale/menu size (U), language switch live, transitions,
   big-view region selection feeding the stethoscope, kill cam button on the autopsy card, GIF capture (SAVE DEATH GIF).
3. The **2D game** (`--2d`): not ported at all. Decide and write it into PORTING.md (likely "not ported, reason").
4. Item 3, Lab tools (`lab/*.py`): driver lines (`data/driver_lines.yaml`, needs a YAML crate), thermogenetics, patch clamp, calcium imaging, pharmacology, rigs, netsci,
   bundles, sleep deprivation, sensitivity, knockout/activation screens, failure autopsy, assays-as-challenges, stats export (CSV/JSON identical to Python; compare with
   `docs/results/`), validation suite (expected 12 PASS / 9 FAIL), headless flags, protocols (42 YAML), replays/bundles formats, pet mode + `pet.ktfsave`.
5. Item 6: apply a loaded save fully (fly body from `flies[i].fly/slot`, surgery modes), share-code apply for surgery/lab/challenge/protocol/contraption (host returns
   "can't apply" now), scores sharing.
6. Finish: every PORTING.md box ticked or reasoned (incl. section 16), README, clean `cargo build/test/clippy`, commit.

## Known deviations (written in PORTING.md)
FreeSans font (check redistribution), linear-light alpha LUT, glyph AA, 3D brain cloud and whole-window big view instead of the flat image, Neuron of the Day uses UTC,
times in reports are UTC, Python can't restore a Rust save's fly body, Rust kill cam auto-plays on death (Python offers it on the autopsy card).
