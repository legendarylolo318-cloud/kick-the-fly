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

## Next, in this order
1. **Leftovers of item 4**: `.ktfreplay` + experiment bundles (do them with the Lab headless runner, item 3), `pet.ktfsave` + `.bak` (`core/pet.py`), applying a loaded save to the Rust game's fly/body/timers
   (`object_state`/`restore_object` JSON in `flies[i].fly/slot`, `extra`) with Save/Load UI (`ui` load_state page), wire `scores.json` etc. Share-code import UI + deeper payload validation (protocol, contraption).
2. **Item 1 + 2: the UI** (biggest). Read each pygame screen in `legacy-python/kickthefly/ui/*.py` before building it; bundle the same fonts/assets; run the Python game
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
