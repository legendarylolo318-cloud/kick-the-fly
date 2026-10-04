# Changelog, 3.1.0 (in progress)

- **Fly's-eye view** (GAME RULE, visual only; 3.1.0 task 11): **F4** (rebindable, off at every launch) shows the scene as a fly's eyes sample it. 3D: three renders from the fly's head (about 280 degrees), sampled through two hexagonal ommatidia lattices (5 degree spacing, about 770 an eye) with an approximate spectral model (R1-R6 480 nm, R7 345 nm which a screen never lights, R8 437 and 520 nm; red goes dark, blue and green stay bright; light adaptation; false color). 2D: a hexagon-sampled, spectrally filtered crop of what is ahead of the fly. Beside it, the brain's own visual neurons (R1-R6, R7/R8, L1/L2, T4/T5, LC4, LPLC2, LC10, LC11). The screen says it is a GAME RULE and why it is not fed to the brain (pixel streaming through the photoreceptors failed); a test checks the brain ends in the same state with it on or off. [fly-eye.md](fly-eye.md). Tests: `tests/test_fly_eye.py` (19).
- **Puppeteer mode** (GAME RULE, with CONNECTOME notes per level; Play, 3.1.0 task 10): Esc > Challenges > Puppeteer, in 3D and `--2d`. You cannot hit the fly (every tool is locked except the optogenetics laser; no touch, no kick, no smell reaches it), and steer it only by activating or silencing its real cell types (laser, latch, the inspector, brain surgery). Ten levels (walk, turn right, walk 0.9 m, back up, take off, sing, groom, sleep, let three looming shadows pass, turn then walk), each with a par, a hint, stars, and a "why it worked" card that says which circuit is the CONNECTOME and which part is a rule (including that MDN's backward walking works through a game rule because the model's MDN pathway FAILS validation). The laser accepts `type:`, `prefix:` and `side:R:` target specs. All ten were solved at par on the real brain in both games, and wrong neurons did not win. [puppeteer.md](puppeteer.md). Tests: `tests/test_puppeteer.py` (69, 36 on the real brain).
- **Real neuron shapes** (GAME RULE for the drawing; the data is the CONNECTOME release's, CC BY 4.0, credited; 3.1.0 task 9): checked what MaleCNS v1.0 offers publicly: a per-neuron **skeleton** SWC for every neuron in Janelia's public bucket (no token), with an md5 on every download; **meshes are not offered per neuron** (they are inside the Neuroglancer volume), so skeletons are what is drawn. Behind the existing opt-in (Settings > Brain > Download real neuron shapes), inspecting a neuron fetches its skeleton (10-100 kB, one file, a pause between requests, size and node limits, one attempt a session), verifies the server's md5 (a file without one is refused), caches it with a sha256 manifest (a cache file that no longer matches is discarded, not drawn) in your data folder, never bundled. The full skeleton is drawn over the big view and a fitted inset with the credit sits in its corner, on the GPU (one line buffer, one draw call, in the brain view's own camera; pygame lines when there is no GPU); the ten key neurons now come from this public verified release first. Anything that fails falls back to the estimated fiber, and the card says why. **The live download was not run** (a download needs your say-so; the tests use a local fake of the bucket's `x-goog-hash` header): [real-shapes.md](real-shapes.md). Tests: `tests/test_real_shapes.py` (23).
- **Failure autopsy** (MODEL PREDICTION, read-only, Lab, 3.1.0 task 8): a page per failing validation test: the strongest routes (through cell types, 1-3 synapses) from the test's input to its target, how much of the target's input they supply hop by hop, how much excites and inhibits, raw synapse counts, each stage's firing in the actual run (three seeds, the pathway tests' drive), and where the driven signal falls below the test's 1.5x. Covers the nine adult failures (MDN, aDN, both E-PG tests, the grooming hierarchy, P1, foreleg GRNs to P1, extinction, second-order); the two larva failures get a page saying why they were not autopsied. Never changes or suggests changing a weight, constant or threshold (checked: the weight matrix digest is unchanged). Lab page with ratio bars; `--failure-autopsy [TEST ...]`, `--validate --autopsy`. [failure-autopsy.md](failure-autopsy.md). Tests: `tests/test_failure_autopsy.py` (14).
- **Activation screen and Knockout screen** (MODEL PREDICTION, Lab, 3.1.0 tasks 6 and 7): two whole-brain screens on one engine. The activation screen holds each cell type driven in turn on held-out seeds (4000-4007) and records every descending neuron type's response and ten behavior readouts; the knockout screen silences ranked candidate cell types (singly, or in batches that are opened when they cut the response) for each of the eleven validated pathway behaviors and ranks them by how much the response drops. Both: a shared brain snapshot per seed, an unperturbed run, matched random controls (same size and superclass mix), a paired t-test over seeds (the Wilcoxon cannot get under p = 0.0078 at 8 seeds, so no whole-screen correction could pass it), Benjamini-Hochberg q and a 75% agreement rule; trivial effects (driving a readout's own neurons) excluded; resumable (`records.jsonl`, a different run in the same folder is refused), a progress bar, CPU processes or GPU-batched brains (`--screen-batch`, gl); CSV and Parquet (pyarrow optional). Lab pages with search, sort, a detail panel, "activate/silence on live flies" and "inspect a neuron"; `--activation-screen`, `--knockout-screen`, and a `screen:` protocol block. Not run: the whole-connectome screens (hours to days); the example runs and the real-pack sanity check (LC4/LPLC2 recovered as the escape pathway, pIP10 as song) are in [screens.md](screens.md). Tests: `tests/test_screens.py` (21).
- **Profiler overlay and frame profile** (GAME RULE: timings only; 3.1.0 task 5): **F3** (rebindable, off at every launch) shows FPS, frame time (mean, p95, worst), a strip of the last frames, and the milliseconds of `sim` (the brains' step time, off-thread), `physics`, `render`, `ui` and `present`, with the compute engine and the brain view renderer, in the 3D game and in `--2d`. `--benchmark` now follows the simulation table with a frame profile of fixed rendered scenes (seed 1, the room, offscreen; 3D at the 60 fps cap, 3D uncapped, 3D with the CPU brain view, 2D); `--no-render-bench` skips it. Sections nest, and the instrumentation is a shared no-op when off. [performance.md](performance.md). Tests: `tests/test_profiler.py`.
- **GPU brain panel** (GAME RULE, 3.1.0 task 4): the brain panel and the big view are drawn by the GPU when there is one. Every neuron is an instance whose 21 fiber and arbor points sit in one static buffer; each frame uploads one number per neuron (its activity, and a sparkle flag) and draws all samples as additive one-pixel points, which is the image the CPU's sparse product made, so the tone map, bloom, HUD, see-through alpha and the Imaging recolor are the same code. Turning, panning and zooming the big view is a uniform now: the CPU rebuilt a sparse matrix per camera (99 ms here, 13 ms on the GPU). The neuron inspector, the path overlay and the labels read the same `spark_pix` cell-body table, computed for the camera without the matrix. Colors, depth shading, region mode, palettes, sparkles and the see-through panel are unchanged; the GPU picture matches the CPU one to a few boundary pixels (`tests/test_gpu_brainview.py`). On any GL error, with no real GPU, or with Settings > Graphics > Brain view drawn by = CPU, it draws on the CPU as before; Settings > Brain says which. Per frame, fixed camera, on a 9070 XT: panel 2.9 ms against 6.1 ms, big 7.6 against 10.6 (the frame is mostly the read-back and the CPU tone map, so the saving is modest; the camera is the win).
- **Simulation off the render thread** (3.1.0 task 3): the brain was already on its own thread at a fixed 200 steps/s (times the game speed; pause and slow motion work), so this task checked it and fixed the one way it failed under load: a main thread that holds the GIL left it 27 of 200 steps/s at CPython's default 5 ms switch interval; the brain thread now sets a 0.2 ms interval and gets all 200. Output unchanged. No separate process and no render interpolation, and why: [performance.md](performance.md). Tests: `tests/test_brain_thread.py`.
- **GPU by default** (3.1.0 task 2): in the game, `auto` now picks the fastest engine that works: OpenGL compute on a real GPU, then a PyTorch GPU, then Numba, then NumPy, with a fallback at any error. Settings > Brain shows the engine in use, what auto would try and a note when a chosen engine failed, and keeps the manual override. Software rasterizers do not count as GPUs. Headless paths keep the exact chain, so `--validate` is unchanged (12 PASS / 9 FAIL, same numbers). The exe and AppImage already bundle ModernGL and the EGL context (`KickTheFly.spec`), so they get gl; building them was not verified here. Found and fixed while testing it, in the gl backend: (1) a garbage-collected brain's late departure could empty the slot a new brain had taken, closing the group under it, so a gl fly fell back to the CPU within its first seconds (`tests/test_auto_engine.py`); (2) the gl shaders ignored individuality (default `subtle`), running every gl fly as a clone: they now apply D_pre and D_post, with the off path unchanged and bit-exact (`tests/test_individuality.py`).
- **Predator animation pass** (GAME RULE, 3.1.0 task 1): the **spider** has eight IK legs in a tetrapod gait with planted feet and a bite that is a cycle (walk, windup 0.28 s with the body rearing and the front legs raised, a lunge, recover): it stands still through the wind-up, and a fly that moves out of reach before the lunge ends is missed (before, the bite was instant at 0.2 m). The **mantis** rocks while it stalks, rears and cocks its forelegs in its aim pause, and walks on IK legs that plant (3D and 2D). The **dragonfly** banks into turns and throws its legs forward as it closes on prey. Shared code in `game/predator_anim.py` (`two_bone_ik`, `LegRig`, `SpiderCycle`) and a `pose()` on every predator in the engine. The 2D dragonfly keeps its flat drawing. Looming and capture touch are unchanged. Tests: `tests/test_predator_anim.py`.
- **Frog hop** (GAME RULE, 3.1.0 task 0): the frog now hops instead of sliding. A state machine (idle, track, crouch, leap, airborne, land, recover, plus the tongue strike) with a ballistic arc from the distance to the fly and a fixed apex, using the game's gravity; procedural squash, stretch, leg and pitch animation, a tongue with overshoot and a height-scaled shadow, in the 3D game and `--2d`. The fly senses it only through the existing looming pipeline. The simulation is untouched. The predator-escape assay was re-run for the frog (seeds 1000-1009): still 0 / 30 escapes, max looming before the strike 1.7 rad/s. Tests: `tests/test_frog_hop.py`.

# What changed in 3.0 (consolidated)

3.0 was built in five days, each by one author and checked by another, then given a release review. This page is the whole release in one
place; `CHANGELOG.md` at the repository root keeps it day by day, with the review fixes, and [3.0-review.md](3.0-review.md) has the decisions,
the known limits and what was never verified. Released as **3.0.0** on 2026-10-03.

**What did not change.** The simulation. `--headless --validate` on the 3.0 branch has been diffed against the 2.13 baseline after every
day, and gives the same results (21 tests, 12 PASS / 9 FAIL, `docs/validation.md`); no weight, time constant or threshold was tuned to make any
test pass. Old saves, training memory, `config.toml`, pet files, replays, protocols, loadouts and share codes keep loading (the compatibility
tests are `tests/test_compat*.py`). Everything new that reads a number off the connectome is tagged **CONNECTOME**, every rule **GAME RULE**, every
outcome **MODEL PREDICTION**, on screen, in the README and in Lab > Model assumptions.

## New: things to play with

| | what | where | docs |
|---|---|---|---|
| **Neurodex** (day 1) | a collection of the cell types your fly fires; entries show the dataset's numbers and, for the curated types, a cited one-line fact | D, Esc > Neurodex | [neurodex.md](neurodex.md) |
| **Kill cam** (day 1) | slow-motion replay of the brain's last seconds on death | `;` | [killcam.md](killcam.md) |
| **Neuron of the Day** (day 1) | one curated type at launch, with a Try it button | launch card | [neurodex.md](neurodex.md) |
| **Predators** (day 3) | frog, dragonfly and mantis, seen only through the real looming neurons; a `predator_escape` assay | Creatures tools; Lab assays | [predators.md](predators.md) |
| **Rain, gusts, storms** (day 3) | outdoors weather through touch, humidity, wind and light neurons | Lab parameters | [weather.md](weather.md) |
| **Kitchen arena** (day 3) | bowl, vinegar trap, sink, burner and a cook with a swatter | E, `--arena kitchen` | [kitchen.md](kitchen.md) |
| **Microphone** (day 3, opt-in) | sound drives the Johnston's organ; off at every launch, red pill while on | Esc > Mic and streamer | [microphone.md](microphone.md) |
| **Streamer mode** (day 3, opt-in) | Twitch viewers vote on tools; read-only anonymous chat; off at every launch, off in Lab mode | Esc > Mic and streamer | [streamer.md](streamer.md) |
| **Fly arcade** (day 4) | a tournament of flies and fly racing, in-game points only | Esc > Fly arcade | [tournament.md](tournament.md), [racing.md](racing.md) |
| **Behavior rigs** (day 5) | tethered flight simulator, fly on a ball, Buridan's paradigm, a four-field olfactory arena | Lab > Behavior rigs | [rigs.md](rigs.md) |
| **Mini-papers** (day 5) | guided experiments that reproduce a classic paper and compare your result with what it found | Esc > Mini-papers, Lab | [minipapers.md](minipapers.md) |
| **What's New** (day 5) | one skippable screen on the first launch after an upgrade | Settings > Help brings it back | below |

## New: research tools (Lab)

- **Genetic toolkit** (split-GAL4 lines, `line:SS00727`), **thermogenetics** (TrpA1, shibire-ts), **virtual patch clamp**, **simulated calcium imaging**
  (GCaMP6s/6f/8m kernels from the sources' numbers) and **pharmacology** (day 2); five protocol blocks and the matching Python API.
- **Experiment bundles** (RO-Crate, bit-exact rerun on the CPU backends) and **share codes** (`KTF1-...`) (day 1).
- **Network science** (motifs against a degree-preserving null, rich club, communities; adult and larva), **sleep deprivation assay**, **sensitivity
  analysis** of every LIF parameter against every validated behavior (day 4).
- **Behavior rigs and their pre-registered assays** with `--individuality off` controls (day 5): three reproduce (the optomotor response, bar and
  stripe fixation), **one fails** (an odor gives the model's fly no preference).
- **Mini-papers** (day 5): six; the model reproduces four (looming to the giant fiber, T-maze conditioning, sugar to MN9, Buridan's stripe fixation), half of
  one (antennal touch to aDN yes, aDN to the leg motor neurons no) and none of the larva pair, and says why.
- `--selftest` and Settings > Help > Self-test, a playthrough bot over every arena and tool, a nightly CI, `tools/run_tests.py` (the test suite in
  balanced chunks) (2.13 and 3.0).

## Changed

- The game shows only **measured** personality cards (day 4 review); a fly in play says "card not measured" until Esc > Fly arcade > Measure the flies in
  play. The seed-drawn card the game used to show described no brain.
- A calm, untouched fly discovers nothing in the Neurodex; entries tagged "at rest" by earlier builds are kept (day 2 decision).
- The neuPrint skeleton download for the brain view is **opt-in** (Settings > Brain; one question after the tutorial); cached shapes still load.
- Race speed rule version 2 (day 4 review): speed reads DNp09 against one fixed reference rate (4.32 Hz) instead of each fly's own warm-up baseline.
- The pause menu has 12 entries (Mini-papers is new) and its buttons are sized from the font.
- `config.toml`: one new key, `[first_run] whatsnew_3_0_seen` (a missing key means not seen). No schema bump.

## What 3.0 found out about itself (reviews)

The reviews are the part of 3.0 that is worth reading: each one found real bugs and several found honest limits.
- **Day 1:** a calm fly "discovered" ~190 types a minute; the rule was rebuilt with a Poisson test; a share code could write outside the exports
  folder (path traversal) and could make the protocol checker build a billion seeds; double key bindings around D.
- **Day 2:** Imaging mode never turned on on the real pack (a race the synthetic pack was too fast to lose); the GCaMP6f half-decay was 140 ms and is
  142 ms (Chen et al. 2013, Supplementary Table 3); the tests opened real windows under a Wayland shell.
- **Day 3:** a hidden second Twitch connection after toggling Streamer mode; a click pulled a fly out of a frog's mouth and the frog ate it anyway;
  the predator assay counted an escape in the capture frame (dragonfly 2/30 became 0/30); the model disagrees with a measurement (pC1 neurons are
  tuned to pulse intervals; the model's P1 is not); the test suite leaked a game per test (20 GB).
- **Day 4:** the in-game personality card was numbers drawn from a random generator; the race's repeatability was a warm-up artefact; the rich club's
  null was built on the wrong skeleton; Cancel did not cancel; Lab pages took each other's job results; a hand-edited points file crashed the arcade.
- **Day 5:** a citation in `docs/validation.md` pointed to an unrelated paper (see below); pygame had been built without fonts or sound on a new
  machine (fixed in `.venv` only, see the handoff).

## Day 5 in detail

- **Behavior rigs** (`game/rigs.py`, `lab/rigassay.py`, `lab/labrigs.py`; `--rig`, `--rig-assay`, `protocols/rig_*.yaml`, `rigs.md`). Yaw is read from
  DNa01/02 right minus left, walking from DNp09; wide-field motion goes through the existing optomotor stage; a stripe or bar is tracked by the duel's LC10
  rule. Criteria were committed before any held-out run. Results on seeds 1000-1009: tethered open and closed loop PASS, ball PASS, Buridan PASS, olfactory
  arena **FAIL** (PI +0.005, p = 0.69), identical in outcome with individuality off.
- **Mini-papers** (`lab/minipapers.py`, `ui/minipaper_ui.py`; `--minipaper`, `minipapers.md`): von Reyn 2014, Tully & Quinn 1985, Shiu 2024, Hampel
  2015, Ohyama 2015, Colomb 2012 (Buridan). What each paper states, and how much of it was read, is on the page.
- **What's New in 3.0** (`ui/whatsnew_ui.py`): once, skippable, a migrated config key; a fresh install does not get it.
- **Neurodex:** two new curated facts (the sugar and bitter SEL neurons, Yao & Scott 2022); each type the new features read (DNa01/02, DNp09, T4/T5, LC10,
  the giant fiber, aDN, MN9) already had one.
- **Fixed:** `docs/validation.md` cited Ohyama et al. 2015 as doi:10.1038/nature14424, which is a different paper; the right DOI is
  10.1038/nature14297.
- Python API: `rig`, `rig_assay`, `minipaper`. Selftest `day5`. Three playthrough checks (`extra:rigs`, `extra:minipapers`, `extra:day5-pages`). 81 new
  localization strings. Lab > Model assumptions: two new cards.

## Known limits of 3.0 (not fixed, stated)

- The MDN, aDN-to-leg, P1, grooming-hierarchy, E-PG compass, foreleg-to-P1, extinction and second-order conditioning and both larva tests FAIL in
  validation; the docs say why for each. The larva brain is headless only.
- Every live stimulus in the windowed game is delivered per screen frame (a slow brain gets more per simulated second); Lab protocols, assays and the
  API are lockstep and exact.
- The 2D game has a source of nondeterminism nobody has found (one playthrough leg).
- Weather costs the brain 8-17% of its real-time speed.
- The paywalled papers behind the mini-papers were not read in full; only their abstracts were.
- The German catalog does not have the 3.0 strings yet; they show in English.
