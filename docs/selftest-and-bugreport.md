# Self-test, bug report and the tutorial (2.13)

All three are in **Settings > Help**.

## Self-test

```bash
python kick_the_fly.py --selftest [--sim-backend cpu] [--out selftest.json]
```

Also in the exe and the AppImage (`KickTheFly.exe --selftest`, `KickTheFly-x86_64.AppImage --selftest`), and in the game
(Settings > Help > Run). Each check ends **PASS**, **WARN** or **FAIL** with a plain-English fix for anything that isn't a PASS.
It only looks: it never installs, downloads, removes or configures anything, and never touches a driver.

| check | PASS means | otherwise |
|---|---|---|
| App and Python | version, Python, source / exe / AppImage, OS | (info) |
| Adult and larva brain packs | the file is a valid zip, its CRCs hold, it has the expected neurons, and its SHA-256 matches a `.sha256` or `SHA256SUMS` beside it if there is one | FAIL for a missing or damaged adult pack; the larva pack is optional (PASS when absent) |
| Backends | each backend that can run here does a 200-step run; cpu, numba and torch-cpu must match NumPy **spike for spike**; gl, torch-cuda and torch-rocm are compared statistically (firing within 5%, population rates r > 0.9) | FAIL for an error or a bit-exact mismatch, WARN for a GPU outside the tolerance. GPU backends run in a child process with a time limit, so a hung driver can't take the self-test down |
| OpenGL 3.3 (3D game) and GPU | a context of at least 3.3; reports the vendor, renderer and GL version the app sees | WARN: the game starts in 2D instead |
| Audio, ffmpeg | a device opened; `ffmpeg` on PATH | WARN: silent play; videos fall back to a GIF |
| Folders | config, data, log and screenshot folders are writable | FAIL for config and data, WARN for the others |
| Disk and memory | free space (FAIL under 200 MB, WARN under 2 GB) and free RAM against the per-fly estimate (350 MB per fly on top of about 1.5 GB) | says how many flies fit |
| Display backend | Wayland or X11 (Linux), and what the game would pick | WARN when there is no display (fine headless) |
| Smoke protocol | 10 s of simulated time (`protocols/selftest_smoke.yaml`): the driven looming detectors fire | FAIL |

Exit codes: **0** all pass, **1** any FAIL, **3** warnings only (2 is bad usage, as elsewhere). `--out FILE.json` writes the same
content as JSON (`format`, `counts`, `exit_code`, `checks[]`). A build that can't do something says so instead of failing.

## Report a bug

Settings > Help > Report a bug, also offered on the crash screen (a small window after a crash of the windowed game;
`KICK_THE_FLY_NO_CRASH_SCREEN=1` skips it). It collects:

- the summary: app version, OS, backend, Python, display and GPU strings;
- the self-test JSON (the last one from this session, or a quick one when there is none);
- the last 500 lines of `kickthefly.log`;
- the crash report, if a crash left one.

Your home folder and user name are replaced by `~` and `<user>` first. The screen shows **exactly** what would be included, item by item,
and every item has a switch. Then you choose: **Copy to clipboard**, **Save to file**, or **Open GitHub issue**, which opens your browser on a
prefilled new-issue page. GitHub issue links are limited to about 8,000 characters, so the link carries the summary and the self-test
verdicts, and the long items (log, crash trace, full JSON) are saved to a text file next to your logs whose path is shown, for you to
attach by hand.

**Nothing is ever uploaded and there is no telemetry.** The module has no network code (a test fails if it imports any); the only thing
that leaves the program is a URL handed to your own browser when you press the button. `python kick_the_fly.py --bugreport [--out DIR]`
prints and saves the same text without opening anything.

## First-launch tutorial

About a minute, on the first launch of a fresh install, skippable at any step (Backspace or the Skip button; on a gamepad Back), and replayable from
Settings > Help. Five steps: moving, using a tool, watching the brain panel light up, sugar as a reward (it puts sugar in your hand), and the loadout
editor. Enter (or the Next button, or A on a pad) skips one step. It runs on keyboard and mouse or a gamepad, its highlight never flashes (with
Reduced flashing on it doesn't even swell), and it uses the Larger text and colorblind palette settings. Whether it has been shown is a flag in
`config.toml` (`[first_run] tutorial_done`). A config migrated from before 2.13 counts as already shown.
