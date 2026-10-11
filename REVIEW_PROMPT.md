# Prompt for the verification, bug-fix and release pass (paste into a fresh Claude Opus session in the repo)

````text
Project: Kick the Fly. A full port from Python/pygame to Rust + Bevy 0.19. You are in a clone of
https://github.com/legendarylolo318-cloud/kick-the-fly on branch `rust-3.1.2` (the same work as `bevy-port`, plus the instruction files).
`PORTING.md` is the parity checklist (including section 16, "GUI parity"). `legacy-python/` is the old version and the reference
(it is also frozen on the branch `python-legacy` and the tag `python-3.1.1`: never delete or rewrite those). Your job is thorough verification,
bug fixing and, only when I say so, release. Do not add new features.

Read first: `RUST_BUILD_INSTRUCTIONS.md`, `PORTING.md`, `continue.md`, `BUGS.md`, README.md. Do not trust my notes or commit messages: check the
code and re-run things. The brain pack `data/kick_brain.npz` is git-ignored; it must be present (copy it into `data/`), otherwise many
goldens silently skip. Say clearly if it is missing.

WHERE THINGS STAND (so you do not redo it, but verify it): all items of the last brief are done and `cargo fmt`, `cargo clippy --workspace
--all-targets --release` and `cargo test --workspace --release` passed (60 test binaries, 286 tests, 0 failed). Done and not yet independently
reviewed since commit 1d69ce2: the headless CLI (`crates/game/src/headless.rs`), pet mode (`ktf_data/src/pet.rs`, `game/src/pet_play.rs`),
photo mode (`game/src/photo_play.rs`), Mic and streamer (`ktf_data/src/streamer.rs`, `game/src/{twitch,live_play,mic,mic_analysis}.rs`,
`ktf_ui/src/pages/live.rs`), neuron search and the path tracer (`fly_sim/src/neurosearch.rs`, `game/src/inspector_play.rs`), a small multi-fly fix
and the README. Known and documented gaps are in PORTING.md and `RUST_BUILD_INSTRUCTIONS.md` section 8.

STEP 0: BUGS I FOUND BY HAND (reproduce first, fix, add a test where one is possible, one commit per fix; details and leads are in BUGS.md):
  a) In the big brain view (B) I can no longer zoom in and out. Python: mouse wheel over the big view with the mouse free calls
     `view.zoom_by(1.15 if ev.y > 0 else 0.87)` (`legacy-python/kickthefly/game/kick3d.py`); drag orbits, Shift+drag pans. In Rust
     `BrainView.zoom` (`crates/game/src/brainview.rs`) is only initialised and read; nothing reads the wheel for it, and `tools.rs` uses the wheel
     to cycle tools. Wire zoom (and verify orbit and pan), keep `inspector_play::project()` and `shapes_play` projections consistent, and make
     sure the wheel does not also cycle tools while the big view has the mouse.
  b) With several flies (N spawns more) the brain panel/HUD/big view does not switch by itself to the fly nearest to me (Python `_update_focus`).
     `crates/game/src/flies_play.rs:follow_nearest` exists but has no visible effect. Find out why (early returns: `arcade.open`,
     `manual_until`, `snap.challenge`, `memory.train`; whether `Command::FocusFly` changes the brain in the snapshot; whether `Flies.focus` and the
     sim thread's focus can disagree) and fix it.
  c) The held tool and its swing animation (swatter etc.) are not visible: Python `Game3D.draw_viewmodel` (kick3d.py:3211) was probably never ported.
  d) The fly looks like a generic fly, not a fruit fly: match the Python 3D fly model (proportions, red eyes, wings, abdomen stripes, legs).
  e) The FPS counter setting (`graphics.show_fps`) does nothing: it is never read in Rust. Same for `graphics.fps_cap`, `graphics.vsync`.
  f) `graphics.ui_scale` does nothing visible (only reaches `compute_layout`); check HUD and menu scale like Python.
  g) "Not running properly on the GPU" (symptom not specified: ask me; wrong adapter on hybrid laptops is the cheap first suspect) and "the fly does
     not react when I put sugar down" (`tools.rs:treats`; Python also pokes the `sweet` group): details in BUGS.md items 7 and 8.
  h) Every other item in BUGS.md and any I add later: same way.

BUG-HUNT MODE AND TOKEN BUDGET (read this, it matters): hunt actively, but be economical. Work bottom-up from cheap evidence to expensive:
  - Prefer `grep`, reading the two implementations side by side and small unit tests over launching the game. Do NOT take screenshots unless
    a question truly cannot be settled from code (a layout/colour question with no pixel reference); then take one, at 1280x720, of that one
    screen, and look at it once. Never screenshot to "see if it works" in general.
  - Do not run the full 40-minute `cargo test --workspace --release` more than twice (once at the start for a baseline, once at the end). While
    iterating run only the affected crate/test: `cargo test --release -p <crate> <test_name>`.
  - Use `cargo check`/`cargo clippy -p <crate>` for fast feedback; a release build of the game crate takes 10+ minutes, so batch several fixes per build.
  - Do not read whole large files (labtools.rs, sim_thread.rs, PORTING.md are huge): grep for the symbol and read the 40 lines around it.
  - For every simple task use the simplest tool; do not spawn sub-agents for something one grep answers. Write findings into BUGS.md as you go.

SETTINGS AUDIT (mechanical first, then targeted): there are 75 settings in `crates/ktf_data/data/schema.json`. BUGS.md lists the ones the Rust code never
references (34). For each setting decide, from code only: is it (1) read by the game and acts live, (2) read but only at startup, (3) stored but has no
effect. For every Python setting with an effect, find where Python reads it (`grep -rn "<key>" legacy-python/kickthefly`) and wire the same effect in
Rust, or document in PORTING.md why not (e.g. no GL backend selection in Rust). Regenerate the unreferenced list after your fixes with:
  python3 - <<'PY'
  import json,glob
  s=json.load(open('crates/ktf_data/data/schema.json')); ks=[x['key'] for x in s["settings"]]
  src=''.join(open(f).read() for f in glob.glob('crates/*/src/**/*.rs',recursive=True) if not f.endswith('ktf_data/src/config.rs'))
  print([k for k in ks if k not in src])
  PY
Then write one table into PORTING.md ("Settings audit": key | where read | effect | verified how). A settings page screenshot is NOT needed for this.

1. Feature + GUI parity (highest priority):
   - Go through PORTING.md item by item and verify each feature against legacy-python by reading both implementations and, where you can, running
     both versions side by side (Python: a venv with `numpy scipy pygame-ce pyyaml pillow`; pygame-ce, not pygame).
   - The Bevy version must look and function the same as the pygame version: same screens, menus, layout, positions, labels, colours, fonts,
     assets, sizes, scaling, animations, navigation order, keybinds, mouse interactions (click/drag/hover/scroll), default settings, popups,
     tooltips, timing and feedback sounds.
   - Verify every item of section 16 "GUI parity". Fix any screen or interaction that looks or behaves differently. Do not redesign or "improve"
     anything; match the pygame version. The reference screenshots are `tests/ui_reference/*.png` (made by `legacy-python/make_*_reference.py`,
     compared in `crates/ktf_ui/tests/reference.rs`); say which screens have NO pixel reference (HUD cards for pet/photo/live inputs, the big
     view's path drawing) and compare those by eye against the Python code.
   - Anything that truly cannot match exactly must still work and act similarly, with the difference and reason documented in PORTING.md. Any
     feature still missing gets flagged in your report, not silently dropped.

2. Correctness: run the golden sim tests and inspect tolerances (suspiciously loose?). Verify fixed-timestep determinism, that the sim thread
   never blocks rendering, and no deadlocks or channel leaks on pause/slow-mo/save/load/mode switch/quit (including TwitchChat and MicHandle
   thread shutdown).

3. Compatibility: old Python save files and share codes load correctly; stats export matches the Python CSV/JSON formats (note: Rust JSON has
   sorted keys, Python kept insertion order: check nothing reads key order); config/save paths on Linux (XDG) and Windows; mic missing/denied fails
   gracefully; the neuron-shapes download handles offline and partial downloads; "Real flies do this too" cards default to OFF.

4. Platform support: build and smoke-test the Linux AppImage (`./build_appimage.sh`) and the Windows .exe via CI (`.github/workflows/ci.yml`,
   `release.yml` have never run on GitHub). Check wgpu behaviour: `PowerPreference::HighPerformance` on hybrid laptops, Vulkan on Linux and
   DX12/Vulkan on Windows, and a clear error instead of a crash when no GPU adapter is found. Verify native Wayland and X11/XWayland.
   You cannot test AMD vs NVIDIA hardware from a sandbox: say exactly what you could and could not test; do not claim it.

5. Performance: compare with the Python version (FPS, frame time, sim steps/s, GPU use) using the profiler overlay (F3, `KTF_PROFILER=1`) and
   `cargo bench`. Look for per-frame allocations (e.g. `neurosearch::search` on a keystroke, `mic::availability()` opening devices every
   frame), excessive draw calls, systems running when they should not. Report the numbers.

6. Quality: `cargo fmt --check`, `cargo clippy --workspace --all-targets --release -- -D warnings`, tests on both OSes, no `unwrap()` on fallible
   I/O or user-data paths, no panics on malformed saves/share codes (fuzz the parsers briefly: `.ktfsave`, `.ktfshare`, `KTF1-` codes, protocol
   YAML, `pet.ktfsave`, bundle zips).

7. Safety and privacy of the new features: the Twitch reader must be anonymous, read-only, never send chat, never store a token, refuse CR/LF/NUL,
   be OFF at every launch, off in Lab mode, off under KTF_NO_NETWORK and in headless runs; the microphone is never recorded or stored; nothing
   writes outside the data/pictures folders.

8. Release: ONLY after I confirm in chat. Do NOT merge to `main`, push to `main`, tag a release or publish anything on your own: `main` and
   `python-legacy` must keep the Python 3.1.1 code until I say otherwise. When every check passes on Linux and Windows AND every PORTING.md item
   (features and GUI parity) is checked or has a documented reason, tell me and wait. If I then say go: open a PR from `rust-3.1.2` to `main`
   (do not delete `python-legacy`), run CI, bump the version in `Cargo.toml` (it says `4.0.0-dev`), and publish a release with the Windows .exe
   and the Linux AppImage plus release notes written for casual players (summarize the engine switch and performance gains). If CI fails, fix and
   re-run; never merge on failure.

8b. Store page (README, pictures, GitHub topics): the GitHub page must describe the Rust game, not the Python one. Do this together with step 8, after the
   bugs are fixed (the pictures must show the fixed game: viewmodel, fruit-fly model, zoom), and only publish with my go:
   - README.md: rewrite the top part (title block, "Download", "Build from source") for the Rust build: no Python 3.11+ / pygame / moderngl /
     Numba / PyTorch instructions as the way to run it; the download rows point at the Rust `KickTheFly.exe` and AppImage; keep the Python game only
     as "the original Python version lives on the `python-legacy` branch / tag `python-3.1.1`". Remove claims that are not true of the Rust build
     (backends, `--2d`, Python API, NWB export) or mark them as not available. Keep it readable for casual players. `RUST_BUILD_INSTRUCTIONS.md`,
     `BUGS.md`, `REVIEW_PROMPT.md` and `continue.md` stay in the repo root only if I say so; otherwise move them under `docs/dev/`.
   - Pictures: regenerate every image README.md references from the RUST game: docs/arcade.png autopsy.png brain.png duel.png field.png killcam.png
     lab.png minipaper.png neurodex.png orchard.png rig.png room3d.png spider.png swat3d.png validation.png whatsnew.png, and docs/demo.gif.
     Use the screenshot harness (one launch per picture, 1280x720 or 1920x1080, same size for all): `kick_the_fly --fullscreen --shot FILE
     --shot-frames 200 [--arena NAME] [--cam x,y,z,tx,ty,tz] [--tool NAME --fire S --hold S] [--menu SCREEN] [--challenge KEY] [--kill-at S]
     [KTF_BIG_BRAIN=1]`, with `KICK_THE_FLY_HOME=<empty temp dir>` for a clean config. Use the real brain pack. Look at each result ONCE and
     keep or retake it; do not iterate more than twice per picture. The GIF: there is no GIF recorder in the Rust build, so take 40-60 `--shot`
     frames of one scripted scene (swat + torch in the open field, as in the old demo) and assemble them with ffmpeg or Pillow
     (`ffmpeg -framerate 15 -i f%03d.png -vf "scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" demo.gif`); keep it under
     ~8 MB. Replace the files in `docs/` (same names, so README links keep working) and keep the old ones in git history only.
   - GitHub topics and description (at release time, with my go; needs `gh auth`): the repo currently has python, pygame, moderngl and opengl topics.
       gh repo edit legendarylolo318-cloud/kick-the-fly --remove-topic python --remove-topic pygame --remove-topic moderngl --remove-topic opengl \
         --add-topic rust --add-topic bevy --add-topic wgpu --add-topic vulkan --add-topic rust-lang --add-topic gamedev
     Keep the neuroscience topics (connectome, drosophila, fruit-fly, ...). Update the repo description if it mentions Python. Do not touch topics
     before the Rust game is on `main`.

Rules: wait on output files for long builds and tests (never poll with pgrep; a full `cargo test --workspace --release` takes about 40
minutes); run the game only with `--fullscreen`; commit on `rust-3.1.2`, one commit per fix, messages ending
`Co-Authored-By: Claude Opus <noreply@anthropic.com>`; tick PORTING.md only for what you verified; update `continue.md` as you go.

End with a report: what was fixed (STEP 0 first), which pictures were regenerated, GUI parity differences found and fixed, the perf numbers before/after, what you could not
test, a numbered findings list (severity, file:line, concrete failure scenario, fix) and any unchecked PORTING.md items.
````
