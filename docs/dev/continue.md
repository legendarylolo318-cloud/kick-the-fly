# continue.md: where the Bevy port stands (read this, PORTING.md, then work)

Branch `rust-3.1.2` (same commits as `bevy-port` up to 825329e; work continues on `rust-3.1.2` only). Task = the user's 7-point brief (exact parity with the pygame game incl. GUI; UI; Lab tools; serde persistence; real neuron shapes;
packaging + CI; README). Tick PORTING.md only for verified things; unmatched things get a written reason there. Commit with
`git -c user.name=Logan -c user.email=legendarylolo318@gmail.com commit` and end messages with `Co-Authored-By: Claude Opus <noreply@anthropic.com>`.
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

## Next, in order (1-7 of the brief are done; only the extras remain)
5. (DONE, 49740a8) Multi-fly leftovers: contraption wind goes to every live fly; the FOCUS note shows the measured card label. Still open (written in PORTING.md):
   the memory scent is observed by the focused brain only, the card is not drawn on the bodies, the cap is a fixed 16.
6. (DONE, 49740a8) README: the Rust build section (what is here / not ported, pack, build and test commands, flags table, layout).
7. (DONE) Final verification at commit after 2fa2c80: `cargo fmt --all`, `cargo clippy --workspace --all-targets --release` clean, `cargo test --workspace --release`
   60 test binaries, 286 tests, 0 failed (goldens with the real pack included).

## Extras worth doing if there is time (not in the 1-7 brief)
- `--headless --validate --seeds 1000-1009 --strict` end to end once (about 25 min): the 2-seed run only proves the plumbing (p cannot reach 0.01 at n = 2).
- `--replay` / `--record-replay` (`.ktfreplay`: `legacy-python/kickthefly/core/replay.py`), `--benchmark`, `--sensitivity`, `--threshold-sweep`, `--signflip-test`,
  `--critical-path`, `--failure-autopsy`, `--tournament`, `--race`, `--netsci`, `--sleep-deprivation`, `--playthrough`, `--share-decode` on the command line
  (the engines for sensitivity/robustness/critical/netsci/sleepdep/tournament/race exist in `fly_sim`; they only need a `headless.rs` arm each, see `lab/headless.py`).
- Pet: `Esc > Mode` already cycles Play/Lab/Pet; the pet's HUD card is not pixel-compared; `pet.*` lab rates are live. Streamer: pause menu does not show the red pill;
  the hum test's "N of 50 neurons firing" counts are not shown. Photo: no `graphics.photo_scale` supersampling (the picture is window sized).
- Neuron search box overlaps the pain card at the top left (Python's big view covers the cards); `lab_diff`/`lab_clamp` Python files forget `import numpy` (ours run).
- Big-view path tracing is drawn as HUD ops over the 3D cloud (not pixel-compared with pygame's flat picture).

## Session 2026-10-11/12 (laptop: i7-13620H, Intel UHD + RTX 4060 Laptop, Wayland): the review of REVIEW_PROMPT.md

Baseline before any change: `cargo test --workspace --release` = 60 test binaries, 286 passed, 0 failed, 1 ignored (the 25-minute validation suite).
`data/kick_brain.npz` was present (the goldens ran, none skipped).

**Fixed and committed (one commit each, details in BUGS.md 'Review of 2026-10-11'):** big-view zoom / tilt / pan / presets; the held tool
(`viewmodel.rs`); the fruit-fly model (`fly_visuals.rs`); FPS counter, FPS cap, VSync (`framerate.rs`); sugar (treats land, `sweet` poke) and
per-fly pokes (`Fly.slot`); ten settings that did nothing (audio, sonification, panel mode, day/night); the profiler overlay; the Twitch reader's
blocking join; the glyph cache in `ktf_ui::font`; `ci.yml` (branch, no LTO in the test job). New tests: `brainview::tests`, `viewmodel::tests`,
`fly_visuals::tests`, `framerate::tests`, `audio::policy_tests`, `tools::tests`, `ktf_data/tests/fuzz_parsers.rs`.

After the changes: `cargo fmt --all --check` clean, `cargo clippy --workspace --all-targets --release -- -D warnings` clean, and the whole suite
= 61 test binaries, 302 passed, 0 failed, 1 ignored. **That last run was made with `CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`
(what `ci.yml` now uses)**: on this 15 GB laptop the nine fat-LTO links of the game crate's test binaries run at once and swap for hours
(use `-j 3` if you want the fat-LTO run here). The shipped fat-LTO binary was built and smoke-tested through the AppImage: `--smoke 3` in an empty
environment (198.6 steps/s), `--selftest` (19 passed), a window on native Wayland and one on XWayland (`WAYLAND_DISPLAY=`; there the log shows
one 'Unable to grab cursor' error at start).

**Not a bug:** UI scale (Python's `compute_layout` snaps it, see BUGS.md 6); the focus does follow the nearest fly (BUGS.md 2).

**How to look at things without a screenshot** (new): `KTF_DEBUG=1` prints `focus: game N sim Some(N) blocked ...` with each fly's distance every
second, and `profiler: frame X ms (N fps)  cpu: hud state / hud draw / hud copy / brain view` every two seconds. A clean test home needs
`[first_run] tutorial_done = true, loadout_notice = false, neuron_shapes_asked = true, whatsnew_3_0_seen = true` in `config/config.toml`,
otherwise a first-run page covers the game and pauses it (a config without `schema_version` raises the loadout notice).

**Performance** (1920x1080 fullscreen, room, no cap, no vsync, nothing else running; see the report for the table): the frame is CPU-bound in
the software HUD. The GPU's main 3D pass takes 0.26 ms. Before: 11.0 ms a frame (91 fps), of which 6.3 ms `hud draw`; after the glyph cache:
9.3 ms (107 fps), 4.8 ms `hud draw`, 0.7 ms copying the canvas into the image, 0.2 ms brain view. What is left in `hud draw` is clearing and
alpha-blending a full-window canvas every frame; the next step would be to redraw only when the HUD's content changed, or to draw the cards on
the GPU (neither done: both change how the HUD is produced and want the pixel references re-run).

**Still open / not done in this review**
- GPU symptom (BUGS.md 7): needs the user's description.
- Missing features found (BUGS.md 13): PROBOSCIS reaction, aggression lunges, science cards, Autopilot's spectator behaviour, 3-item cap on treats.
- Settings with no effect: the table at the end of PORTING.md (rows marked **none**).
- Never run: `ci.yml` / `release.yml` on GitHub, anything on Windows, the 25-minute `--include-ignored` validation suite, the real Twitch service,
  a microphone, AMD / Intel-only GPUs, X11 / XWayland.
- No pixel reference: the in-game HUD (only by eye against `py_3d_hud_room.png`), pet / photo / live-input HUD cards, the big view's inspector,
  search and path drawing, `load_state`, `loadout_notice`, `neuron_shapes_ask`, kill cam, fly's eye, profiler.

## Known deviations (written in PORTING.md)
FreeSans font (check redistribution), linear-light alpha LUT, glyph AA, 3D brain cloud and whole-window big view instead of the flat image, Neuron of the Day uses UTC,
times in reports are UTC, Python can't restore a Rust save's fly body, Rust kill cam auto-plays on death (Python offers it on the autopsy card).
