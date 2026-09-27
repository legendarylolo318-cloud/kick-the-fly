# Contributing

Two rules matter more than anything else here:

1. **Every behavior is either traceable to the connectome or labeled a game rule.** If the fly does something, either
   you can point at the neurons and synapses that made it happen, or it says GAME RULE on screen, in the README's
   "What is the connectome and what is a game rule" tables, and in Lab > Model assumptions.
2. **Nothing ships as a biological claim you can't ground.** A published result that this simulation reproduces goes
   in `kickthefly/lab/validation.py` with a pass criterion fixed before the run and a citation. A result it doesn't
   reproduce stays in the table as a FAIL rather than quietly disappearing.

## Repo layout

```
kick_the_fly.py            launcher shim: keeps `python kick_the_fly.py ...` working, points at the canonical docstring
kickthefly/
  __main__.py              entry point (`python -m kickthefly`)
  core/                    simclock, simcore, memory, savestate, paths, platform_env, version, config, crash
  sim/                     brainpack, connectome/ (loader + LIF simulator), neuron and synapse state
  game/                    kick_the_fly (2D game + Brain + brain panel), kick3d, render3d, outdoors (open field,
                           orchard, day/night), gamepad: physics, tools, arenas, input
  ui/                      menu framework and the settings screens
  lab/                     lab, labjobs, labstats, validation, assays, challenges, protocol, recorder, nwbexport,
                           headless, benchmark, api (`from kickthefly import Fly`), neurosearch (brain view search and
                           path tracer), and the Lab-only manipulations (threshold, signflip, criticalpath, clamp,
                           diffmode, lesions)
  data/                    non-code assets bundled inside the package
tests/                     pytest suite; `-m "not validation"` skips the slow validation suite
protocols/                 bundled YAML example protocols
docs/                      the Lab, validation, performance and Python API docs, the example notebook, and the
                           screenshots and demo GIF used by the README (tools/make_screenshots.py makes them)
packaging/                 Linux desktop/appdata files, the AUR package and the Flatpak manifest (local builds)
tools/                     dev and one-off scripts (benchmarks, profiler, screenshots, icon) — not imported by the game
data/                      **not in git**: the connectome download, graph.pkl and kick_brain.npz
```

### Where does my new code go?

- A new **tool, arena or body behavior** → `kickthefly/game/`. The 2D game and the 3D room share the same `Brain`,
  so a hit should drive the same sensory neurons in both.
- A new **Lab manipulation or analysis** (a lesion, a stress test, an export format) → `kickthefly/lab/`, as its own
  module, with a headless entry point so it can run in a protocol or in CI. Register its screen in
  `Game.lab_pages()` (`kickthefly/game/kick_the_fly.py`) and its page function in `kickthefly/lab/lab.py:install`.
- Anything that **reads or writes a user file** → go through `kickthefly/core/paths.py`. Never hardcode a folder.
- Anything that **touches the connectome data itself** → `kickthefly/sim/`. The brain pack is versioned by content,
  not by a format number: add new arrays optionally and keep `brainpack.load` working with packs that lack them.
- A **dev script you'd run once** → `tools/`. The repo root stays at the shim, the build scripts and the docs.

Don't add new modules to the repo root.

## House rules

- Python 3.11, no new runtime dependencies without a reason. Optional dependencies (like `pynwb`) must degrade to a
  clear message, never an import error at startup.
- User-facing file names, config keys, save-state formats and data paths are a compatibility surface. Existing saves
  and training memory must keep loading; `tests/test_compat.py` guards that.
- Tests: `python -m pytest -m "not validation"` for the fast suite, `python -m pytest` for everything. New Lab
  features need at least a headless round-trip test.
- Real vs rule tags: if you add a reaction, tag it in `REACTION_SOURCE` and say which it is in the README table and
  in the `kickthefly/game/kick_the_fly.py` docstring. Both are part of the change, not follow-up work.
- Determinism: anything in `kickthefly/lab/` runs in lockstep (`kickthefly/core/simcore.py`), so the same seed gives
  the same spikes. Don't reach for wall-clock time or thread state there.
- Compute backends (`kickthefly/sim/connectome/backends.py`): `CPUBackend` is the reference. A CPU-side backend
  (`numba`, `torch-cpu`) must be **bit-exact** with it: the same float operations in the same order and precision, no
  `fastmath`, scalars cast to the state dtype. `tests/test_backends.py` runs 1000 steps in float32 and float64 and
  compares every spike, membrane potential and the gain; a single rounding difference shows up within a few hundred
  steps because the network is chaotic. GPU backends are held to the statistical tolerance in that file instead.
  Anything that writes `sim.v`, `sim.refr`, `sim.spikes` or the weight arrays directly must work on every backend
  (host arrays are the source of truth; call `sim.backend.on_weights_changed()` after editing weights).

## Screenshots and the demo GIF

Every image in the README is made by `tools/make_screenshots.py`, from the current build, so they can be remade the
same way after a UI change:

```bash
python kick_the_fly.py --headless --validate --out data/validation_results.json   # the Validation page shows these
python tools/make_screenshots.py              # every screenshot into docs/ (about 5 minutes)
python tools/make_screenshots.py lamp duel    # just some of them; --list shows them all
python tools/make_screenshots.py demo         # docs/demo.gif, recorded with the in-game video recorder
python tools/make_screenshots.py portrait --out /tmp    # a close-up of the fly model, for checking it after a change
```

Each scene is a short script (seed 7, a fixed arena, camera and inputs) that runs inside the real game loop through
`kick3d.run(script=...)` and captures the whole window at 1280x760, with the brain panel set to solid (the see-through
scene excepted). It renders offscreen on your GPU (SDL's offscreen driver: no window opens) with a throwaway
`KICK_THE_FLY_HOME`, so your settings, memory and saves are untouched, and it uses cached neuPrint skeletons without
fetching. Game states that are awkward to reach by hand (trained memory, a wrapped fly, an autopsy, five flies feeding
in the orchard) are reached by the script driving the game's own actions, never by drawing anything special. Add a
scene with `scene(...)` in that file and reference its PNG from the README. Keep new images at the same size and
quantized (the script does it); `docs/` isn't bundled into the exe or AppImage.

## Before you open a PR

```bash
python -m pytest -m "not validation"     # fast suite
python kick_the_fly.py --headless --protocol smoke.yaml --out /tmp/smoke
python kick_the_fly.py --smoke 3         # the 3D room actually opens
```

The release workflow (`.github/workflows/release.yml`) builds the brain pack from the public connectome, runs the
full suite including validation on Linux and Windows, and builds the AppImage and the exe. It can be run by hand
(workflow_dispatch) without publishing anything.
