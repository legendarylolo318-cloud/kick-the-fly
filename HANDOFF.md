# Handoff: Kick the Fly 3.1.0 (branch `release/3.1.0`, 14 commits on `main`, nothing pushed)

Every task below is one commit, in order (`git log main..release/3.1.0`). The simulator was not touched: **`--headless --validate` gives 21 tests, 12 PASS / 9 FAIL, and all 2,026 numbers in `validation_results.json` are identical to the 3.0 baseline** (only the app version and the timestamp differ). NumPy / Numba / torch-cpu were not changed; the one gl-backend change is listed under task 2.

## Checks run
| check | result |
|---|---|
| `--validate` (final, CPU, 4 workers, 702 s) | 12 PASS / 9 FAIL, identical to baseline (also diffed after tasks 2, 4/5, 6/7) |
| `--selftest` | 22 passed, 0 failed, warnings only (after every task) |
| 3D and `--2d` smoke launches (20 s, offscreen, autopilot) | no traceback (after every task) |
| fast suite (`tools/run_tests.py --fast --chunks 6 --parallel 3`) | 1,584 passed; the only 3 failures (a sonify test assuming an idle mixer channel, the i18n catalog check) were fixed and rerun green; the full suite was not rerun after those two fixes |
| not run | the full `--playthrough all`, the AppImage/exe builds, Windows |

Known flake: `test_puppeteer.py::test_every_level_is_solvable_at_par_on_the_real_brain[3d-clean_antennae]` failed once in a combined run and passed alone and in the full fast suite; not understood.

## Benchmarks (`--benchmark`, CPU engine, this machine, main 5ff8a0c vs this branch)
| flies | before: paced / uncapped per fly | after: paced / uncapped per fly |
|---|---|---|
| 1 | 199.8 / 641 steps/s | 199.8 / 584 |
| 8 | 199.8 / 396 | 199.8 / 382 |
| 16 | 163.3 (0.82x real time) / 165 | 154.8 (0.77x) / 155 |

The CPU simulation reads 4-9% slower after, consistent with noise on a shared machine, but not proven to be. The headless benchmark uses the exact CPU chain; the GL engine is what the game now picks. Frame profile (new, no "before" exists), offscreen, GL engine: 3D 61 fps, 3D uncapped 117 fps (8.5 ms), 3D with the CPU brain view 51 fps, 2D 62 fps. Brain thread: 200 of 200 steps/s under a busy main thread (27 before the switch-interval fix). Raw logs: `~/ktf_final/bench/`.

## Tasks
| # | task | status | main files | known issues / not verified |
|---|---|---|---|---|
| 0 | Frog jump | done | `game/predators.py`, `predator_play.py` | looming assay re-run: 0/30 escapes, as before |
| 1 | Predator animation | done | `game/predator_anim.py` (new), `predator_play.py`, `kick3d.py` | 2D dragonfly keeps its flat drawing |
| 2 | GPU by default | done | `sim/connectome/backends.py`, `sim.py`, `ui/menu.py`, `core/config.py` | fixed two gl bugs found on the way (stale slot removal fell back to CPU; shaders ignored individuality). Exe/AppImage bundle ModernGL per the spec but were **not built** |
| 3 | Sim off the render thread | done (it already was; fixed GIL starvation) | `game/kick_the_fly.py` (Brain.start) | no separate process, no render interpolation (reasons in `docs/performance.md`) |
| 4 | GPU brain panel | done | `game/gpu_brainview.py` (new) | matches the CPU image to a few boundary pixels; falls back on any GL error |
| 5 | Profiler (F3) + `--benchmark` frame profile | done | `core/profiler.py`, `lab/framebench.py`, `lab/headless.py` | |
| 6 | Activation screen (Lab) | done | `lab/screens.py`, `labscreen.py`, `lab/headless.py`, `protocol.py` | **whole-connectome screens not run** (hours to days); example runs only |
| 7 | Knockout screen (Lab) | done | same engine | same |
| 8 | Failure autopsy (Lab, read-only) | done | `lab/failure_autopsy.py`, `labautopsy.py` | covers the nine adult failures; the two larva failures only say why they were not autopsied. Never suggests changing a weight |
| 9 | Real neuron shapes | done, **live download untested** | `sim/realshapes.py`, `game/shape_draw.py`, `sim/morphology.py` | needs your say-so to download; tests use a local fake of the bucket. Meshes are not offered per neuron, so skeletons are drawn |
| 10 | Puppeteer (Play) | done | `game/puppeteer.py`, `lab/puppet_challenge.py` | all ten levels solved at par on the real brain, 2D and 3D; MDN's backward walking works only through a game rule (its pathway fails validation) |
| 11 | Fly's-eye view (F4) | done | `game/fly_eye.py`, `extras3.py`, `kick3d.py` | visual only, not fed to the brain (tested); spectral model approximate |
| 12 | Lesion battles | done | `game/lesion_battle.py`, `ui/battle_ui.py`, `ui/arcade_ui.py` | hot-seat only; budget 3 picks, 500-neuron cap are rules, not tuned |
| 13 | Contraption builder | done | `game/contraption.py`, `lab/contraption_challenge.py`, `core/sharecode.py` (new kind CTN), `ui/share_ui.py` | fires the game's own tools (3D via a temporary aim/tool-tip override); fly is tethered by default; physics is deliberately simple |
| 14 | Brain sonification | done | `core/sonify.py`, `game/sonify_play.py`, `extras3.py`, `core/config.py` | "respect Streamer mode" is my reading: silent in Streamer mode unless allowed in Settings. **Audio was only checked on SDL's dummy driver and by analysing the waveform; I have not heard it** |

Every feature is tagged CONNECTOME / GAME RULE / MODEL PREDICTION on screen where it appears and in `docs/connectome-and-game-rules.md`; each has its own page in `docs/` and an entry at the top of `docs/changelog.md`. New settings default off or to sensible values; old saves, settings, pet files, replays, protocols and share codes still load (the new share kind is appended, so earlier kinds keep their byte; tested).

## Things to know
* Hotkeys added: F3 profiler, F4 fly's-eye, F5 sonification (all rebindable, off at launch).
* The game ignores SIGTERM, so `timeout 20 python -m kickthefly...` never ends it; use `timeout -k`.
* My manual script runs once wrote `contraption: 9.0` into your real `scores.json`; I removed it. Smoke launches of the real game also touch your real `neurodex.json` (as before this work).
* Untracked leftovers: `~/ktf_wt_base` (a git worktree of main for the baseline benchmark; remove with `git worktree remove ~/ktf_wt_base`).
* `watch_tests.sh` in the repo root runs the tests with a live percentage (added at your request).
* Docs sweep not done: the long docstring at the top of `game/kick_the_fly.py` still describes the old backend chain and shape-download text.
