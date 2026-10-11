# Prompt for the verification, bug-fix and release pass (paste into a fresh Claude Opus session in the repo)

````text
Project: Kick the Fly. A full port from Python/pygame to Rust + Bevy 0.19. You are in a clone of
https://github.com/legendarylolo318-cloud/kick-the-fly on branch `rust-3.1.1` (the same work as `bevy-port`, plus the instruction files).
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
  c) Any further items I add to BUGS.md: same way.

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
   (features and GUI parity) is checked or has a documented reason, tell me and wait. If I then say go: open a PR from `rust-3.1.1` to `main`
   (do not delete `python-legacy`), run CI, bump the version in `Cargo.toml` (it says `4.0.0-dev`), and publish a release with the Windows .exe
   and the Linux AppImage plus release notes written for casual players (summarize the engine switch and performance gains). If CI fails, fix and
   re-run; never merge on failure.

Rules: wait on output files for long builds and tests (never poll with pgrep; a full `cargo test --workspace --release` takes about 40
minutes); run the game only with `--fullscreen`; commit on `rust-3.1.1`, one commit per fix, messages ending
`Co-Authored-By: Claude Opus <noreply@anthropic.com>`; tick PORTING.md only for what you verified; update `continue.md` as you go.

End with a report: what was fixed (STEP 0 first), GUI parity differences found and fixed, the perf numbers before/after, what you could not
test, a numbered findings list (severity, file:line, concrete failure scenario, fix) and any unchecked PORTING.md items.
````
