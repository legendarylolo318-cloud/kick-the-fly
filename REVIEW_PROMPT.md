# Prompt for an independent review (paste into a fresh Claude Opus session in the repo)

````text
You are reviewing a port, not writing one. Repo root: the clone of this repository, branch `rust-3.1.1` (or `bevy-port`). It is a Rust + Bevy 0.19
port of a Python/pygame game (the Python original is untouched in `legacy-python/` and is the reference). Read `PORTING.md` (the parity
checklist), `continue.md`, `RUST_BUILD_INSTRUCTIONS.md`, `README.md` and `BUGS.md` first. Do NOT trust my notes in them or in commit messages:
check the code and re-run things. `BUGS.md` holds problems found by hand while playing: reproduce and fix those first.

Scope: everything after commit 1d69ce2 (`git log --oneline 1d69ce2..HEAD`, `git diff 1d69ce2..HEAD --stat`). In order, that stretch added:
1. the headless CLI: `crates/game/src/headless.rs` (`--headless --validate/--protocol/--rerun-bundle/--rig/--rig-assay/--minipaper/--activation-screen/--knockout-screen`) vs `legacy-python/kickthefly/lab/headless.py`
2. pet mode: `crates/ktf_data/src/pet.rs`, `crates/game/src/pet_play.rs` vs `core/pet.py` and the pet code in `game/kick_the_fly.py`
3. photo mode and picture saving: `crates/game/src/photo_play.rs` (+ `player.rs`, `tools.rs`, `controls.rs`, `synth.rs`) vs `game/kick3d.py` (FreeCamera, photo_mode, viewfinder)
4. Mic and streamer: `crates/ktf_data/src/streamer.rs`, `crates/game/src/{twitch.rs,live_play.rs,mic.rs,mic_analysis.rs}`, `crates/ktf_ui/src/pages/live.rs`, HUD pills/tally in `ktf_ui/src/hud.rs` vs `core/streamer.py`, `core/netguard.py`, `game/live_inputs.py`, `core/mic.py`
5. neuron search and path tracer: `crates/fly_sim/src/neurosearch.rs`, `crates/game/src/inspector_play.rs` vs `lab/neurosearch.py`
6. small multi-fly fixes (contraption wind to every fly, FOCUS note card label) and the README

Do these, and report findings ranked by severity with file:line and a concrete failure scenario for each (no vague smells):

A. Re-run everything: `cargo fmt --check`, `cargo clippy --workspace --all-targets --release`, `cargo test --workspace --release`
   (long goldens take 10-20 min; wait on output files, do not poll with pgrep). Report any failure or any test that is skipped
   because the brain pack `data/kick_brain.npz` was not found (tests like `neurosearch_real` silently `return` without it: say so).
B. Parity: for each item above, read the Python and the Rust side by side and list behaviours that differ and are NOT already written
   down in PORTING.md. Pay attention to: numeric formatting (Python `f"{x:.1e}"`, `{:.1f}`), rounding, ordering, off-by-one in
   thresholds, state machines (pet catch-up, vote rounds, cooldown), what is saved/persisted and when, and file formats other tools read
   (`pet.ktfsave`, `rerun_report.json`, bundle zips, validation results JSON).
C. Golden validity: for each new golden (`tests/golden/make_pet.py`, `make_streamer.py`, `make_neurosearch.py`, hum/protocol ones) check
   the Rust test really compares against values produced by the Python code, that tolerances are not so loose they hide bugs, and that
   the golden covers the branches that matter (e.g. neurosearch 3-hop ties, pet real-stakes death, vote cooldown/min-votes).
D. Safety/privacy: the Twitch reader must be anonymous, read-only, never send chat, never store a token, refuse CR/LF/NUL, be OFF at
   every launch, off in Lab mode, off under KTF_NO_NETWORK and in headless runs. Confirm each in code. Confirm nothing writes outside
   the data/pictures folders, and that the microphone is never recorded or stored.
E. Bevy correctness: systems with more than 16 params, ordering bugs between `update_camera`/`photo_play::apply_camera`/`walk`,
   input leaking (hotkeys firing while typing in the search box or in photo mode), a `DepthOfField` left on the camera after leaving photo
   mode, per-frame allocations or locks that hurt the 60 Hz frame (e.g. `neurosearch::search` on a keystroke, `mic::availability()`
   opening devices every frame), thread shutdown/join hangs (TwitchChat, MicHandle).
F. Honesty of docs: every claim in PORTING.md, continue.md and README.md added in this stretch must be true. List any box ticked `[x]`
   that is not fully verified, and any "not done" item that actually is done.
G. Anything that would make the headless exit codes lie (0/1/2) or make `--headless` open a window.

Rules: do not change code unless a finding is a clear, small bug; if you fix something, make it one commit per fix with a test, and say
so. Commit messages end with `Co-Authored-By: Claude Opus <noreply@anthropic.com>`. Do not run the game windowed except with
`--fullscreen`. Output: a numbered findings list (severity, file:line, scenario, suggested fix), then a short verdict
("safe to merge" / "fix these first").
````
