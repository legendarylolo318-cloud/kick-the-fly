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

## Session 2026-10-11 status: items 1-4 of the last brief are DONE and committed
Headless CLI (`crates/game/src/headless.rs`: `--headless --validate|--protocol|--rerun-bundle|--rig|--rig-assay|--minipaper|--activation-screen|--knockout-screen`),
pet mode (`ktf_data::pet` + `game::pet_play`, golden `make_pet.py`), photo mode (`game::photo_play`: free camera, Bevy `DepthOfField`, viewfinder, clean capture into the
pictures folder, shutter sound), Mic and streamer page (`ktf_data::streamer` golden `make_streamer.py`, `game::twitch` anonymous IRC over rustls, `game::live_play`, page
`ktf_ui::pages::live`, matched 0.6-0.7 %), neuron search + path tracer (`fly_sim::neurosearch` golden `make_neurosearch.py`, `inspector_play.rs`).
The pygame references need `/tmp/pgvenv` recreated: `python3 -m venv V; V/bin/pip install numpy scipy pygame-ce pyyaml pillow` (pygame-ce, NOT pygame: no font module on 3.14).

## Next, in order (the user ran out of tokens; resume here)
5. **Multi-fly leftovers** (PORTING.md "multi-fly" lines, around the `flies_play.rs` notes): contraption wind and the memory scent pokes go to the focused brain only;
   the personality card is not shown on the bodies (fly-vs-fly cards); the fly cap is not adapted to the machine; arcade/tool wheel/puppeteer assume one fly.
   Check each against `legacy-python/kickthefly/game/kick_the_fly.py` (`FlySlot`, `_scents`, `pet_card`) and fix what is cheap; write the rest down.
6. **README** (`README.md` is still the Python one or a stub: check): what the Rust build is, how to build (`cargo build --release`, `--features bundle-pack`), run flags
   (`--fullscreen`, `--headless ...`, `--selftest`, `--bugreport`, `--shot`), the data folders, the pack (`data/kick_brain.npz`), what is not ported (PORTING.md summary),
   licenses/credits (FreeSans, neuPrint shapes credit).
7. **Final full verification**: `cargo build --release`, `cargo test --workspace --release` (long goldens: 10-20 min; wait with an output-file loop, never pgrep),
   `cargo clippy --workspace --all-targets --release` clean, `cargo fmt --all`; tick PORTING.md only for verified boxes; commit.

## Extras worth doing if there is time (not in the 1-7 brief)
- `--headless --validate --seeds 1000-1009 --strict` end to end once (about 25 min): the 2-seed run only proves the plumbing (p cannot reach 0.01 at n = 2).
- `--replay` / `--record-replay` (`.ktfreplay`: `legacy-python/kickthefly/core/replay.py`), `--benchmark`, `--sensitivity`, `--threshold-sweep`, `--signflip-test`,
  `--critical-path`, `--failure-autopsy`, `--tournament`, `--race`, `--netsci`, `--sleep-deprivation`, `--playthrough`, `--share-decode` on the command line
  (the engines for sensitivity/robustness/critical/netsci/sleepdep/tournament/race exist in `fly_sim`; they only need a `headless.rs` arm each, see `lab/headless.py`).
- Pet: `Esc > Mode` already cycles Play/Lab/Pet; the pet's HUD card is not pixel-compared; `pet.*` lab rates are live. Streamer: pause menu does not show the red pill;
  the hum test's "N of 50 neurons firing" counts are not shown. Photo: no `graphics.photo_scale` supersampling (the picture is window sized).
- Neuron search box overlaps the pain card at the top left (Python's big view covers the cards); `lab_diff`/`lab_clamp` Python files forget `import numpy` (ours run).
- Big-view path tracing is drawn as HUD ops over the 3D cloud (not pixel-compared with pygame's flat picture).

## Known deviations (written in PORTING.md)
FreeSans font (check redistribution), linear-light alpha LUT, glyph AA, 3D brain cloud and whole-window big view instead of the flat image, Neuron of the Day uses UTC,
times in reports are UTC, Python can't restore a Rust save's fly body, Rust kill cam auto-plays on death (Python offers it on the autopsy card).
