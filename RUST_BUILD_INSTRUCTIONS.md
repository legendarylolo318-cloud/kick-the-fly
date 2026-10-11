# Running the Rust build on your main PC (branch `rust-3.1.1`)

This branch is the Rust + Bevy port of Kick the Fly. It is **not** on `main`. `main` still has the Python 3.1.1 game, and the same Python code is
kept for good on the branch `python-legacy` and the tag `python-3.1.1` (both point at the 3.1.1 release commit `2fd9e5b`). Nothing here changes either.

| What | Where |
|---|---|
| Rust port (this) | branch `rust-3.1.1` |
| Python 3.1.1, frozen | branch `python-legacy`, tag `python-3.1.1` (also `legacy-python/` inside this branch, used as the reference) |
| Parity checklist | `PORTING.md` (what is ported, how it was verified, what is not) |
| Where the work stands | `continue.md` |

## 1. Get the code

```bash
git clone -b rust-3.1.1 https://github.com/legendarylolo318-cloud/kick-the-fly.git kick-the-fly-rust
cd kick-the-fly-rust
```

## 2. Get the brain pack (needed; it is not in git)

The connectome pack `data/kick_brain.npz` (37 MB) is git-ignored. Without it the game runs a **synthetic random network**, which is not the fly.
Put the file at `data/kick_brain.npz` inside the clone (`mkdir -p data`), or point `KICK_THE_FLY_PACK` / `--pack FILE` at it.

- Easiest: copy it from the machine this was built on: `~/kick-the-fly/data/kick_brain.npz` (USB stick, `scp`, cloud drive).
- Or: the Python release's AppImage/exe carries the same pack inside (`KickTheFly-x86_64.AppImage --appimage-extract`, then look for
  `kick_brain.npz`; unverified that it is byte-identical, so check the game's `--selftest` says "Adult brain pack ... PASS").
- Or build it from the MaleCNS download with the Python dev tool (README, "Build from source").

## 3. Toolchain

- **Arch**: `sudo pacman -S rustup && rustup default stable`, plus a Vulkan driver: `sudo pacman -S vulkan-icd-loader nvidia-utils`
  (or `vulkan-radeon` / `vulkan-intel`). `vulkaninfo --summary` should list your GPU.
- **Debian/Ubuntu**: `sudo apt install build-essential pkg-config libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev libx11-dev`, then rustup.rs.
- **Windows**: rustup.rs with the MSVC build tools.

## 4. Build and run

```bash
cargo run --release -p kick_the_fly -- --fullscreen     # first build ~10-15 min (Bevy + fat LTO), later builds are faster
```

`--fullscreen` matters on a tiling Wayland compositor (otherwise the window opens half-size). `H` in the game lists the keys, Esc is the menu.

A ready-made AppImage can be built with `./build_appimage.sh` (needs `data/kick_brain.npz` and internet once for `appimagetool`); it writes
`dist/KickTheFly-x86_64.AppImage`. Run it with `--fullscreen` (and `--appimage-extract-and-run` if FUSE is missing).

## 5. Quick health checks (2 minutes)

```bash
cargo run --release -p kick_the_fly -- --selftest        # install check: exit 0 pass / 1 fail / 3 warnings
cargo run --release -p kick_the_fly -- --smoke 5         # headless; should say about 198-200 steps/s
cargo run --release -p kick_the_fly -- --headless --minipaper list
cargo run --release -p kick_the_fly -- --headless --rig tethered --individuality off --out /tmp/rig
```

The whole test suite: `cargo test --workspace --release` (about 40 min including compiling; last result: 286 passed, 0 failed).

## 6. What to try in the game (things that are new in this branch)

1. **Pet mode**: Esc > Mode > Pet. A card under the memory card shows hunger/sleep/mood. Quit, wait a minute, reopen: the "welcome back" lines.
2. **Photo mode**: F10. WASD/Space/C fly, mouse aims, `[` `]` field of view, `,` `.` blur, `K` `L` focus, `F` autofocus, `H` hides the HUD, F12/Enter/click snaps
   (saved in your pictures folder, with a shutter sound).
3. **Mic and streamer**: Esc > Mic and streamer. Turn the microphone on and hum (a red pill shows). Twitch: type a channel name and switch it on; viewers
   vote with `!tool NAME` (off in Lab mode, off at every launch).
4. **Neuron search**: B (big view), click the search box at the top left, type `DNp01`, Enter; click a neuron, then PATH FROM HERE / PATH TO HERE on another.
5. **Lab** (Esc > Lab tools): validation, assays, rigs, mini-papers, activation/knockout screens, export, protocols.
6. Everything else from the Python game: arenas (E), tools (number keys), more flies (N), surgery (O), training (T), save states.

## 7. Reporting what you find

Write each problem in `BUGS.md` (template inside) as you go, then hand `BUGS.md` and `REVIEW_PROMPT.md` to Claude Opus. For each bug: what you did,
what you expected, what happened, and anything the terminal printed. A screenshot (F12, or the AppImage's `--shot FILE`) helps.
If the game crashes it writes a crash report (the crash window says where); `--bugreport` prints what a report would contain (nothing is sent).

## 8. Known gaps (so you do not report them as bugs)

Written in full in `PORTING.md`. Short list: the 2D game is not ported (`--2d` is ignored); NWB/TIFF export, GIF/video/time-lapse capture; Numba/PyTorch/GL
backends (one CPU backend here); individuality "subtle" cannot be spike-for-spike equal to NumPy (last-bit gains); photo mode's blur is Bevy's, not
Python's shader, and pictures are window-sized; `--replay`, `--benchmark`, `--sensitivity`, `--threshold-sweep` and friends are not on the Rust command line yet;
the Twitch connection has never been exercised against the real service; the pause menu does not show the red mic/Twitch pill.

## 9. If you want to publish this later

Nothing is merged or released. When you are happy: open a PR from `rust-3.1.1` to `main` (the Python code stays on `python-legacy` and in `legacy-python/`),
bump the version in `Cargo.toml` (it says `4.0.0-dev`; saves and share codes stamp it), and let `release.yml` build the AppImage and the `.exe`
(it has never run on GitHub).
