# continue.md: where the Bevy port stands (read this, PORTING.md, then work)

Branch `bevy-port`. Task = the user's 7-point brief (exact parity with the pygame game incl. GUI; UI; Lab tools; serde persistence; real neuron shapes;
packaging + CI; README). Tick PORTING.md only for verified things; unmatched things get a written reason there. Commit with
`git -c user.name=Logan -c user.email=legendarylolo318@gmail.com commit` and end messages with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
Memory notes exist (kick-the-fly-bevy-port, feedback-fullscreen-wayland-tiling): the user is on Wayland tiling, **run the game with `--fullscreen`**.

## How to verify (keep using these)
- Screens: `cargo run -q -p ktf_ui --example render_menu`; pygame references via `legacy-python/make_ui_reference.py` (needs a venv:
  `python3 -m venv V; V/bin/pip install numpy scipy pygame-ce pytest pillow pyyaml pyarrow moderngl fonttools`; run with `SDL_VIDEODRIVER=dummy`); compare in
  `crates/ktf_ui/tests/reference.rs` (< ~0.9 % clearly different pixels). In-game: `kick_the_fly --fullscreen --menu <screen[:Tab]> --shot out.png --shot-frames 300`
  (set `KICK_THE_FLY_HOME=<tmp>` for a clean config). Python 3D game screenshot: `ln -s ../data legacy-python/data` then `KICK_THE_FLY_HOME=<tmp> V/bin/python kick_the_fly.py --smoke 8 out.png`.
- Goldens: `tests/golden/make_*.py` (simulator, groups, pain, neurodex), `legacy-python/make_persistence_fixtures.py` (config, share codes, stores, loadouts, savestate recipe in `make_savestate_fixture.py`).
- Always `cargo fmt --all` + `cargo clippy -p ktf_data -p ktf_ui -p fly_sim -p kick_the_fly --all-targets -- -D warnings` before committing. Systems may take at most 16 params (use `SystemParam` bundles).
- Do not edit sources while a release build runs in the background.

## Done so far (all committed; last commit "Save/Load State ... CI workflows ...")
- Crates: `fly_sim` (sim, groups, pain index, neurodex table+discovery, state export/import), `ktf_data` (config.toml v3, share codes, paths, savestate, stores, loadout,
  facts, Neuron of the Day, CPython-compatible shuffle), `ktf_ui` (software-canvas port of pygame menus + HUD, fonts FreeSans/FreeMono/DejaVuMono bundled, `Page` framework), `game` (Bevy).
- Menus verified against pygame screenshots: pause, quit, Settings (all 6 tabs), What's New, Share, Loadout editor, Neurodex, Load State (listing). HUD ported (`ktf_ui::hud`, `hud_canvas.rs`)
  incl. pain/reward cards, hotbar = loadout, brain panel text, toasts, Neuron of the Day, tutorial card, H overlay. First-run chain, rebindable actions (`actions.rs`).
- Persistence: config, share codes, save states (sim state continues Python's raster bit-exactly), neurodex/contraptions/lesion/memory stores.
- Packaging written but NOT yet verified: `.github/workflows/ci.yml` + `release.yml` (Python workflows moved to `legacy-python/workflows-python/`), `build_appimage.sh`, `bundle-pack` feature,
  `windows_subsystem` + console attach (unchecked on Windows). **A release build was started in the background** (`cargo build --release -p kick_the_fly`, log in the scratchpad `release.log`);
  check it finished, then run `./build_appimage.sh` (needs curl for appimagetool) and test: `env -i HOME=/nonexistent dist/KickTheFly-x86_64.AppImage --appimage-extract-and-run --smoke 3`.
  Try `rustup target add x86_64-pc-windows-msvc && cargo check --target x86_64-pc-windows-msvc -p kick_the_fly` for the Windows code.

## Next, in order
1. Verify packaging (above), README mentions of AppImage/CI, tick PORTING section 1 items.
2. Remaining UI pages (port line by line from `legacy-python/kickthefly/ui/*.py`, `lab/*.py`, `game/kick_the_fly.py`): selftest + bugreport + crash (Help tab buttons), `challenges`
   (`lab/challenges.py`: T-maze, Sneak, Sweet, ReverseSurgery, PredictNeuron, Puppeteer, Contraption, Escape room; scores.json), `arcade` + battle (`arcade_ui.py`, `battle_ui.py`; the Rust arcade is a Bevy-text overlay in `arcade.rs`
   that must be replaced by the page), `live_inputs` (mic + streamer), `lab_minipapers`, Lab hub + ~20 `lab_*` pages. Also: tool wheel, duel card, autopsy card, big brain view, brain surgery (O) and
   training (T) screens, MEMORY card (needs `core/memory.py` plasticity), arena status text, photo mode, stethoscope (K), pad navigation, UI scale/menu-size U key, language switch live.
3. The **2D game** (`--2d`, `Game` class): not ported at all. Decide: a 2D mode on the same sim, or list it in PORTING.md as not ported with the reason.
4. Item 3, Lab tools (`lab/*.py`): driver lines (`data/driver_lines.yaml`; needs a YAML crate), thermogenetics, patch clamp, calcium imaging, pharmacology, rigs, netsci, bundles, sleep deprivation,
   sensitivity, knockout/activation screens, failure autopsy, assays-as-challenges, stats export (CSV/JSON identical to Python; compare with `docs/results/`), validation suite (expected 12 PASS / 9 FAIL),
   headless flags, protocols (42 YAML), replays/bundles formats, pet mode + `pet.ktfsave`.
5. Item 5: real neuron shapes: same source/checksum/cache as `sim/realshapes.py`, async (IoTaskPool), SWC parser, draw in the Neurodex entry and inspector.
6. Apply a loaded save fully (fly body from `flies[i].fly/slot`), scoring `scores.json` sharing, share-code apply for surgery/lab/challenge/protocol/contraption (host returns "can't apply" now).
7. Finish: every PORTING.md box ticked or reasoned (incl. section 16 GUI parity), clean `cargo build/test/clippy`, README (Rust build text already added), commit.

## Known deviations to keep in PORTING.md (already written there)
FreeSans font (GPL+font exception: check redistribution; swap = `assets/fonts` + `make_ui_metrics.py`), linear-light alpha LUT, glyph AA, 3D brain cloud instead of the flat 2D brain image, Neuron of the Day uses UTC date,
Python can't restore a Rust save's fly body.
