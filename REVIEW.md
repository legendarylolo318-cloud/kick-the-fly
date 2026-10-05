# Kick the Fly 3.1.0: release review

Reviewer: Claude (Opus 5.5), 2026-10-04, on branch `release/3.1.0` (14 feature commits by the previous model, listed in HANDOFF.md, plus the
review commits below). Machine: Arch Linux, Python 3.14.7, AMD Radeon RX 9070 XT (Mesa 26.2.4), 24 cores, 30 GB. Nothing in HANDOFF.md was taken
on trust; every row below was run or read during the review.

## Verdict: **GO** (subject to CI)

No blocker was found. The simulation is unchanged, every compatibility check passes, and every new feature works as specified. The bugs found
were fixed in small commits (below). Merge and release only if the PR's CI (tests, validation, checks, and the new release gate with the
exe and AppImage builds) is fully green.

Not verified here (not blockers, stated so nobody assumes otherwise): the **live** neuron-shape download (needs the owner's OK to touch
the network; it is tested against a local fake bucket); the Windows exe on real hardware (built and self-tested in CI only); the
whole-connectome screens (hours to days; small screens were run); sonification heard through speakers (analysed offline only).

## 1. Baseline integrity (blocking)

| check | result |
|---|---|
| `--validate` (default engine, 4 workers) on v3.0.0-rc.2 and on this branch | **PASS**: both 21 tests, 12 PASS / 9 FAIL; all 1,699 numbers identical (only `created`, `seconds` differ) |
| diff of `kickthefly/sim` against rc.2 | **FLAGGED, accepted**: (a) `sim.py`: a `busy_s` counter for the profiler, no math change; (b) `backends.py` (gl only): individuality gains are now applied in the shaders (before, a gl fly silently ran without them); stale-slot and closed-group fixes; `auto` policy; (c) `morphology.py`: tries the public MaleCNS skeleton first (drawing only). No weight, time constant, threshold or LIF update changed |
| NumPy vs Numba vs torch-cpu, seeds 1000-1009, 600 warm-up + 1,000 steps with stimuli | **PASS**: spike-for-spike identical (SHA-256 per step) on all 10 seeds |
| GPU (gl) vs NumPy | **PASS**: paired over 10 seeds, TOST equivalence (±5% of the mean rate) p < 0.0001, Wilcoxon p = 1.0, per-neuron rate r ≥ 0.9999. On this driver gl matched NumPy exactly. With individuality=strong gl also matches NumPy, and differs from gl without individuality (r = 0.981), so the new shader path really applies the gains |
| brain thread vs single-threaded loop | **PASS**: `tests/test_brain_thread.py` (same seed, same spikes) |
| extra: torch-cpu with individuality=strong | differs on seed 1000 (NumPy = Numba). Pre-existing, documented in docs/individuality.md |

## 2. Full test pass

| check | result |
|---|---|
| full suite (`tools/run_tests.py`, 8 chunks, incl. the validation suite) | 1,631 passed, 23 skipped, **2 failed**: `test_puppeteer::test_a_wrong_neuron_does_not_win[2d/3d-lift_off]`. Not reproducible: passes alone, in the same file order serially (235 passed), and in 32 seeded re-runs; it failed only with two heavy chunks in parallel plus another heavy job. Same family as the flake HANDOFF.md reports. **Left as a known load-dependent flake**, not a regression (fails/passes identically before and after the review commits) |
| `--selftest` | **PASS**: 22 passed, 2 warnings (dummy audio driver, pynwb), 0 failed |
| every protocol in `protocols/` with `--headless` | **37/42 PASS**; the 5 `lecture_*.yaml` crashed with `KeyError: 'warmup_s'` (also on rc.2). **Fixed**: they are classroom lectures and are now refused with a message (exit 2) |
| Python API notebook (`docs/api_example.ipynb`) | **PASS**: every cell runs; every printed number matches the saved outputs |
| Python 3.11 (CI's version) | every source file compiles under 3.11.17 |

## 3. Compatibility (artifacts made with the rc.2 code itself, loaded with 3.1.0)

| artifact | result |
|---|---|
| config.toml | **PASS**: no warnings, every setting kept, new keys at defaults |
| pet file | **PASS** |
| save states, 2D and 3D | **PASS**: load and keep stepping |
| 42 protocols | **PASS**: all accepted by the checker |
| share codes (surgery, loadout, protocol, challenge, lab) | **PASS**: decode and validate |
| replay | **PASS**: "spikes identical to the recording" |

## 4. Features (tasks 0-14)

| # | feature | result | notes |
|---|---|---|---|
| 0 | Frog hop | **PASS** | 40 simulated frogs: 0 ground moves outside `airborne`, 243 landings exactly on the aim point, strikes fire; looming only through `preds.threats()`, no escape special case (grep) |
| 1 | Predator animation | PASS | tests; drawing only |
| 2 | GPU auto-select | **PASS** | fallback at init (auto picks the next engine) and mid-run (a GL error hands that brain to the CPU, it keeps stepping) checked by injection; the override persists in config.toml |
| 3 | Sim off the render thread | PASS, **improved** | see section 6 |
| 4 | GPU brain panel | PASS | picking uses the CPU `spark_pix` table (unchanged); CPU fallback on any GL error; tests |
| 5 | Profiler / frame profile | PASS | |
| 6-7 | Activation / knockout screens | **PASS** | interrupted at 60/120 records and resumed: CSVs byte-identical to an uninterrupted run; different settings in the same folder refused; thresholds are module constants (fixed before running); knockout on looming escape ranks LC4 (54% drop) and LPLC2 (39%) first against matched random lesions |
| 8 | Failure autopsy | **PASS** | read-only (weights digest asserted); **13/13** synapse counts for the MDN and P1 autopsies re-counted by hand from the raw MaleCNS feather files match (with the pack's documented weight ≥ 3), signs too |
| 9 | Real neuron shapes | PASS (offline) | md5 required, sha256 cache manifest, CC BY 4.0 credit drawn with the shape; nothing bundled (spec unchanged). **Fixed**: the consent dialog only mentioned ten neuPrint skeletons |
| 10 | Puppeteer | PASS | MDN level honestly labelled (validation FAIL, game rule stands in); flake above |
| 11 | Fly's-eye view | **PASS** | output only reaches a texture; not fed to the brain |
| 12-13 | Lesion battles, contraptions | **PASS** | round trips exact; 4,000 fuzzed/corrupted codes: no exception other than the module's own errors |
| 14 | Sonification | **PASS** | gated on mute, master and own volume, no device, mic, Streamer mode (unless allowed), pause; reserved channel 0; peak 0.797 under a 0.8 limiter, no clipped int16 samples, no clicks at chunk boundaries |

Both 3D and `--2d` were exercised for the frog, tool spam, death mid-leap and saves (section 7).

## 5. Honesty audit

Every new feature carries its tags on screen and in docs/connectome-and-game-rules.md. Fixed overstatements: the engine-choice entry implied GPU runs
match NumPy ("only within float32 summation order"); the GPU brain view "changes no look" (it differs by a few edge pixels); performance.md said a
brain process "would change nothing" (with many flies the game is GIL-bound); the shape-download consent, the self-test and streamer.md did not
mention the 3.1.0 per-neuron download. Not changed: the "What's new" screen still describes 3.0 (a 3.1 page would be a new feature).

## 6. Performance

Headless simulation (`--benchmark --no-render-bench --flies 1 4 16 --seconds 5`), steps/s per fly, uncapped (paced real-time ratio for 16):

| engine | flies | rc.2 | 3.1.0 |
|---|---|---|---|
| NumPy | 1 / 4 / 16 | 808 / 639 / 173 (0.86x) | 819 / 650 / 177 (0.90x) |
| Numba | 1 / 4 / 16 | 961 / 816 / 377 (1.00x) | 962 / 832 / 356 (1.00x) |
| gl | 1 / 4 / 16 | 1020 / 754 / 288 (1.00x) | 960 / 739 / 292 (1.00x) |

No regression (differences are run-to-run noise; HANDOFF.md's "4-9% slower" does not reproduce).

**In the game the brains were starved by Python's GIL, not by the GPU or CPU.** With 8 flies the GPU sat at ~20% and the process used under two
cores while the brains ran at 0.39x real time. py-spy showed the render thread holding the lock for long C calls. Fixed (pixel- and bit-identical):
the HUD upload (`tobytes` conversion every frame, ~1 ms at 1280x760, ~4 ms at 1440p) now sends the surface's own memory with a texture swizzle;
`segment()`/`frame_from_x()` on plain floats (11x faster, bit-identical on 20,000 cases); the memory card rescored 5x/s instead of every frame.
3D game, 60 fps cap unless noted, quiet machine, steps/s per fly:

| scene | rc.2 (NumPy) | 3.1.0 before review | 3.1.0 |
|---|---|---|---|
| 1 fly | 200, GPU 9% | 200 | 200, GPU 31%, main thread 28% (was 43%) |
| 4 flies | 200 | 200 | 200 |
| 8 flies | 70 (0.35x) | 78 (0.39x), 52 fps | **132 (0.66x), 58 fps** |
| 1 fly uncapped | 59 (0.29x) | 133 (0.67x), 170 fps | **187 (0.94x), 302 fps** |

GPU utilization rose (9% -> 24-31%) and frame time dropped (uncapped 5.9 -> 3.3 ms). Beyond ~5 flies the game still falls behind real time; the
remaining cost is fly physics on the main thread (left alone: a rewrite could change trajectories bit for bit). A brain process would remove it.

10-minute session (4 flies, gl, autopilot with the orbiting big brain view): **no leak**: RSS 2,321 MB at 1 min, 2,326 MB at 10 min; VRAM flat (1,860-1,900 MB, system-wide); brains 200 steps/s throughout; frames mean 21.4 ms, p95 23.8, max 25.0 (47 fps: autopilot draws the big brain view every frame), no spikes.

## 7. Robustness

| case | result |
|---|---|
| no GPU (llvmpipe) | PASS: 3D runs, auto does not pick the software "GPU" |
| OpenGL < 3.3 (`MESA_GL_VERSION_OVERRIDE=3.2`) | PASS: falls back to 2D |
| no audio device | PASS |
| read-only data folder | **FAIL -> fixed**: PermissionError at quit and the process hung (also rc.2); now a warning and a clean exit |
| offline (`KICK_THE_FLY_OFFLINE=1`) | PASS |
| 16 and 20 flies | PASS (no errors; slow, see section 6) |
| tool spam (400 uses, 18 tools), 3D and 2D | PASS |
| dying mid-leap, 3D and 2D | PASS |
| save/load mid-jump, 3D and 2D; with a frog mid-leap | PASS |

## 8. Packaging

Built and self-tested in CI by the new `release gate` workflow (AppImage on Ubuntu 22.04, exe on Windows), and again by release.yml on the tag. The
spec and build scripts are unchanged from rc.2: the bundled pack is `data/kick_brain.npz` (SHA-256 a65baf8602b4...), nothing shape-related is
bundled. Sizes: see the CI logs and the release.

## 9. Docs

Checked the changelog, README, docs/lab.md, docs/playing.md, docs/performance.md against the code. Added F5 (sonification) to the controls table and
the in-game help, Lesion battles and the Contraption builder to playing.md, the in-game performance table, and fixed README's claim that the GPU
backend needs a source install.

## Bugs found and fixed (commits)

| commit | fix |
|---|---|
| perf(3d) HUD upload | GIL held ~1-4 ms per frame |
| perf(3d) segment/frame_from_x | numpy on 3-vectors per limb |
| perf(hud) memory card | 41k-synapse rescoring every frame |
| fix: read-only data folder | crash and hang at quit |
| fix: --protocol on a classroom lecture | KeyError |
| honesty: shape-download consent, selftest, streamer.md | understated network use |
| docs: F5, new modes | missing controls |
| honesty: engine choice, GPU view | overstated exactness |
| ci: release gate | numbers vs rc.2, exe + AppImage on PRs |
| Release 3.1.0 | version, CITATION, changelogs, notes, appdata, performance docs |
| fix: no-GPU brain view (found by PR CI) | without a GPU the view failed over inside its first frame, holding a closed game and its brains for seconds |
| fix: autopsy empty state (found by PR CI) | the 'no validation result yet' line ran past the panel |
| tests: fly's-eye throttle, sonification (found by PR CI) | depended on the runner's wall-clock speed |
| fix: GL context restore (found by the owner, **release blocker**) | on Wayland the 3D window stayed on the loading screen while the game ran (the GPU probe restored the window's context without its surface; X11 lost its drawable the same way). No automated test has a real window, so none caught it |
| fix: Neuron of the Day card (found by the owner) | in a wide 3D window the card shrank to ~90 px with its text spilling out (also in 3.0) |

Left: the load-dependent Puppeteer flake; torch-cpu with individuality (documented); the 3.0 What's-new screen.
