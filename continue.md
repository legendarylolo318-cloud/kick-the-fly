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

## Next, in this order
1. **Rest of item 4**: `.ktfsave` (zip: npz arrays + JSON header, format `kick-the-fly-save` v1, includes numpy PCG64 state: `fly_sim::rng` has the pieces; `core/savestate.py`),
   `.ktfreplay`, `pet.ktfsave`+`.bak`, `fly-memory.npz`+`training-log.json` (fixtures in `legacy-python/tests/fixtures/fly-memory-v2.5.0*`), `neurodex.json`, `contraptions.json`,
   `lesion_loadouts.json`, experiment bundles. Add fixtures to `make_persistence_fixtures.py` (real Python output) and tests in `crates/ktf_data/tests/`. Also: wire the game's
   save/load, share-code import UI (`ui/share_ui.py`) and `scores.json` shared with Python; deepen share validation for protocol and contraption payloads (container check only today).
2. **Item 1 + 2: the UI** (biggest). Read each pygame screen in `legacy-python/kickthefly/ui/*.py` before building it; bundle the same fonts/assets; run the Python game
   (`cd legacy-python && python3 kick_the_fly.py`) to screenshot and compare. Order: menu framework (bevy_ui, custom theme) -> pause menu + Settings tabs (built from `schema.json`,
   writes `config.toml` through `ktf_data::config`) -> HUD/popups/science cards (default OFF) -> Play/Lab split -> Neurodex, neuron of the day, mini-papers, streamer. Tick section 16 as each is verified.
3. **Item 3: Lab tools** (`legacy-python/kickthefly/lab/`): driver lines, thermogenetics, patch clamp, calcium imaging, pharmacology, rigs, netsci, bundles, sleep dep, sensitivity,
   knockout screen, failure autopsy, assays-as-challenges, stats export (same CSV/JSON as Python; compare with `docs/results/`). Per-feature tests against Python outputs.
4. **Item 5**: "Real neuron shapes" download: same source/checksum/cache as `sim/realshapes.py`, async (`bevy_tasks::IoTaskPool`/`AsyncComputeTaskPool`, no blocking in a frame); SWC parser.
5. **Item 6**: packaging + CI. Workflows in `.github/workflows/` still build Python. Needed: `rustfmt.toml` + one `cargo fmt` commit (the existing code is NOT fmt-clean under any config:
   260+ diffs at width 140), `clippy -D warnings` (fix what shows), `windows_subsystem = "windows"` in release (`#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]`
   in `crates/game/src/main.rs`), AppImage script (bundle `assets/` + `data/kick_brain.npz`, test in a clean container with `--appimage-extract-and-run`), Windows MSVC exe + assets, CI matrix
   ubuntu+windows: fmt, clippy, test, golden tests, upload both artifacts.
6. Finish: tick PORTING.md (every item checked or a written reason), clean `cargo build/test/clippy`, commit (end message with the Co-Authored-By line from the session reminder).

## Notes
- Nothing from this session is committed yet unless the log says so (`git log -1`).
- Windows build is still unverified here; real microphone, gamepads untested.
