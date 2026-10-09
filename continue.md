# continue.md: where the Bevy port stands (say "continue" and work from here)

Branch `bevy-port`. Task text (the user's 7 points) is summarized in the checklist below. Read `PORTING.md` first (esp. section 16 "GUI parity").
Memory file `kick-the-fly-bevy-port.md` has the build/verify tricks (`--shot`, Bevy 0.19 gotchas).

## Done
- **Item 4, part 1 (persistence with serde)**: `crates/ktf_data` (no Bevy): `config` (config.toml v3, all 75 settings from `data/schema.json`, keys,
  pad, loadouts, first_run, migrations), `sharecode` (all 6 kinds, validate, preview, `.ktfshare`), `paths` (`directories` crate, Python's layout).
  Tests use real Python-written files in `tests/python_files/` (regenerate: `python3 legacy-python/make_persistence_fixtures.py`). Python also reads Rust codes
  (checked by hand with `cargo run -p ktf_data --example share_encode`). The game loads `config.toml` at startup (`Settings::from_config`).
- **Item 7**: README has Rust build instructions (Arch `pacman -S rustup`, Vulkan: vulkan-icd-loader + vulkan-radeon / nvidia-utils).
- PORTING.md: persistence status block, section 10 rows, new section 16 "GUI parity" (checkbox list, all unchecked, with "Cannot be matched exactly" subsection).

## Done (second session)
- `.ktfsave` read/write (`ktf_data::savestate`), `LifSim::{export,import}_state` (+ `PcgState`, `ActivityBuffer::restore`): a real Python save continues bit-exactly in Rust.
- `ktf_data::stores`: neurodex.json, contraptions.json, lesion_loadouts.json, fly-memory.npz + training-log.json. Python reads everything Rust writes (checked with `examples/rewrite_files.rs`).
- Python test env for regenerating fixtures: a venv with numpy scipy pygame-ce pytest pillow pyyaml pyarrow (pygame 2.6.1 does not work on Python 3.14; use pygame-ce).
- `rustfmt.toml` (width 140) and the whole workspace is `cargo fmt` clean now.

## Done (third session): the UI foundation
- `crates/ktf_ui`: software canvas + the pygame `Menu` (see PORTING.md "GUI parity"): widgets, pages (`pages/`: whatsnew, share), Settings with all tabs incl. Help, page framework
  (`Page` trait + `PageHost` JSON bridge: pages live in ktf_ui and are tested against pygame screenshots; the game answers queries/commands in `crates/game/src/menu_ui.rs:GameHost`).
- Reference tooling: `legacy-python/make_ui_reference.py` (pygame screenshots of each screen), `make_ui_metrics.py` (font metrics/advances), `tests/ui_reference/*.png`, `crates/ktf_ui/tests/{reference,behavior,font_metrics}.rs`.
  Python env: `python3 -m venv V; V/bin/pip install numpy scipy pygame-ce pytest pillow pyyaml pyarrow moderngl fonttools`; the Python 3D game runs for screenshots with
  `ln -s ../data legacy-python/data` (remove it after) and `KICK_THE_FLY_HOME=<tmp> V/bin/python kick_the_fly.py --smoke 8 out.png` (opens a real window for ~10 s).
- `tests/ui_reference/py_3d_hud_room.png` is the pygame 3D game's HUD: **this is the next big GUI target**: the Rust HUD (hud.rs, Bevy text) looks nothing like it.
  Plan: draw the HUD like Python does, on a canvas (`ktf_ui::hud`): status card, health bar, hint box, PAIN / REWARD / MEMORY cards (kick_the_fly.py `_draw_hud`, `_draw_pain`,
  `_draw_reward`, `_draw_memory`), hotbar with `draw_icon` (l.2077; 18 icons), crosshair, popups, science card, the right-hand brain panel (`_draw_brain`: title, LIVE dot, the brain view
  image (existing `brainview` render, placed in the panel rect), TOUCH / DESCENDING NEURONS bars, WHOLE BRAIN trace, REACTIONS log). Window is W x H logical, PLAY_W = W - 280.
  Needs sim-side data that is not ported yet: pain index (`Brain.pain_parts`), reward, memory (`core/memory.py`), reaction log. Window default there is 1280x760.
- 3D scene look also differs from the pygame renderer (room, materials, lighting): decide how far to take it.

## Done (fourth session)
- HUD ported (see PORTING "HUD"): `ktf_ui::hud` + `crates/game/src/hud_canvas.rs`; pain index in `fly_sim::groups` (golden `tests/golden/pain.json`); history ring; layout `ktf_ui::layout`.
- Rebindable hotkeys: `crates/game/src/actions.rs` (`Actions::just("big_view")`); controls/player/tools/brainview/flyeye/profiler/audio/help use it. Python semantics: P = pain level, Z pause,
  [ ] slower/faster (0.1/0.25/0.5/1), . step. **Still on raw KeyCode**: arcade.rs (Rust-only menu), killcam, mic (F6), contraption (C), laser keys ; '.
- Loadouts: `ktf_data::loadout` + editor page; `ActiveLoadout` resource drives the hotbar and selection.
- Fonts: HUD uses FreeMono 13 + FreeSans (tables in `crates/ktf_ui/data/advances.json`, regenerate with `legacy-python/make_ui_metrics.py OUT.json advances.json`).
- Run the game fullscreen for screenshots: `kick_the_fly --fullscreen --menu loadout --shot out.png --shot-frames 200`.

## Next, in this order
1. **Leftovers of item 4**: `.ktfreplay` + experiment bundles (do them with the Lab headless runner, item 3), `pet.ktfsave` + `.bak` (`core/pet.py`), applying a loaded save to the Rust game's fly/body/timers
   (`object_state`/`restore_object` JSON in `flies[i].fly/slot`, `extra`) with Save/Load UI (`ui` load_state page), wire `scores.json` etc. Share-code import UI + deeper payload validation (protocol, contraption).
2. **Item 1 + 2: the UI** (biggest; foundation + pause/settings/help/whatsnew/share done, see above). Remaining pages: neurodex_ui, loadout_ui (+ core/loadout.py, hotbar), tutorial, arcade_ui + battle_ui, minipaper_ui, challenges, live_inputs, load_state/save_state, selftest, bugreport, crash, lab pages; then the HUD. Read each pygame screen in `legacy-python/kickthefly/ui/*.py` before building it; bundle the same fonts/assets; run the Python game
   (`cd legacy-python && python3 kick_the_fly.py`) to screenshot and compare. Order: menu framework (bevy_ui, custom theme) -> pause menu + Settings tabs (built from `schema.json`,
   writes `config.toml` through `ktf_data::config`) -> HUD/popups/science cards (default OFF) -> Play/Lab split -> Neurodex, neuron of the day, mini-papers, streamer. Tick section 16 as each is verified.
3. **Item 3: Lab tools** (`legacy-python/kickthefly/lab/`): driver lines, thermogenetics, patch clamp, calcium imaging, pharmacology, rigs, netsci, bundles, sleep dep, sensitivity,
   knockout screen, failure autopsy, assays-as-challenges, stats export (same CSV/JSON as Python; compare with `docs/results/`). Per-feature tests against Python outputs.
4. **Item 5**: "Real neuron shapes" download: same source/checksum/cache as `sim/realshapes.py`, async (`bevy_tasks::IoTaskPool`/`AsyncComputeTaskPool`, no blocking in a frame); SWC parser.
5. **Item 6**: packaging + CI. Workflows in `.github/workflows/` still build Python. Needed: `rustfmt.toml` + (done), `clippy -D warnings` (fix what shows), `windows_subsystem = "windows"` in release (`#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]`
   in `crates/game/src/main.rs`), AppImage script (bundle `assets/` + `data/kick_brain.npz`, test in a clean container with `--appimage-extract-and-run`), Windows MSVC exe + assets, CI matrix
   ubuntu+windows: fmt, clippy, test, golden tests, upload both artifacts.
6. Finish: tick PORTING.md (every item checked or a written reason), clean `cargo build/test/clippy`, commit (end message with the Co-Authored-By line from the session reminder).

## Notes
- Nothing from this session is committed yet unless the log says so (`git log -1`).
- Windows build is still unverified here; real microphone, gamepads untested.
