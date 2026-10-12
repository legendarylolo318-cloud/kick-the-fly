# Contributing

Two rules matter more than anything else here:

1. **Every behavior is either traceable to the connectome or labeled a game rule.** If the fly does something, either
   you can point at the neurons and synapses that made it happen, or it says GAME RULE on screen, in the README's
   "What is the connectome and what is a game rule" tables, and in Lab > Model assumptions.
2. **Nothing ships as a biological claim you can't ground.** A published result that this simulation reproduces goes
   in the validation table (`crates/fly_sim/data/validation_tests.json`, `crates/fly_sim/src/validation.rs`) with a pass criterion fixed before the run and a citation. A result it doesn't
   reproduce stays in the table as a FAIL rather than quietly disappearing.

## Repo layout

```
crates/fly_sim/    the simulator and every analysis engine (pure Rust, no Bevy)
crates/ktf_data/   config, saves, share codes, protocols, bundles, per-OS folders
crates/ktf_ui/     the menu pages and the HUD, drawn in software
crates/game/       the Bevy game: one plugin per feature
tests/             golden data, real files and page pictures from the Python game 3.1.1, and the scripts that made them
docs/              the manual; docs/dev/BUGS.md is the list of open issues
```

[PORTING.md](PORTING.md) says, feature by feature, what matches the Python game (the branch `python-legacy`), how that was checked, and what
is missing.

## Before you send a change

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p <the crate you touched> <the test you touched>
```

The whole suite (`cargo test --workspace --release`) links about sixty binaries with fat LTO; use
`CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`, as CI does, unless you have a lot of memory and time. CI runs it on
Linux and Windows for every push. The goldens need `data/kick_brain.npz` (see the README) and skip themselves without it.

- **The simulation must stay the Python one's.** The goldens in `tests/golden/` are bit-exact on Linux; do not loosen one to make a change
  pass. If a number can only agree on one platform, say why in the test (see `duel_golden.rs`).
- **Menus and the HUD must look like the reference pictures** in `tests/ui_reference/` (`crates/ktf_ui/tests/reference.rs`).
- **New player-visible behavior** needs its CONNECTOME / GAME RULE / MODEL PREDICTION tag and a line in PORTING.md or the manual.
- Run the game with `--fullscreen` on a tiling Wayland compositor. `KTF_DEBUG=1` prints the focus and a per-frame profile to the terminal;
  `--shot FILE --shot-frames N` (with `--menu`, `--arena`, `--demo`) takes pictures without a person at the mouse.
