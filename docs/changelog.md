# Changelog, 3.1.0 (in progress)

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
