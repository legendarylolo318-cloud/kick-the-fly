# Python API

Kick the Fly's brain is usable as a library: the same headless, lockstep machinery the Lab, the protocols and the
validation suite use, from a script or a notebook. It needs a source checkout with the brain pack built
(README: Run from source); the exe and the AppImage don't expose it.

```python
from kickthefly import Fly

fly = Fly(seed=1000)                       # 166,700 LIF neurons, warmed up for 3 s
rec = fly.record({"giant fiber": "dnp01", "looming": "loom"})
fly.step(2.0)                              # 2 s of calm
fly.drive("loom", amp=0.5)                 # hold LPLC2 + LC4 driven
fly.step(2.0)
fly.undrive()
print(rec.rates(0, 2), rec.rates(2, 4))    # {'giant fiber': 7.2, 'looming': 5.1} {'giant fiber': 66.5, ...}
fly.export("results/looming")              # CSV, npz and metadata JSON; nwb=True adds an NWB file
```

A worked example with its outputs is in [api_example.ipynb](api_example.ipynb).

## `Fly(seed=0, *, backend=None, params=None, warmup_s=3.0, learn=True, surgery=None)`

A fresh, untrained brain. It never reads or writes the training memory the game saves.

| argument | meaning |
|---|---|
| `seed` | the brain's noise and every random choice; the same seed and the same calls give the same spikes |
| `backend` | `cpu`, `numba`, `torch-cpu`, `torch-cuda`, `torch-rocm`, `gl` or `None`/`auto` (see README: Performance). `fly.backend` says what actually ran |
| `params` | Lab parameters by name, e.g. `{"noise_std": 0.06, "thresh.escape": 5.0}` (Lab > Parameters lists them) |
| `warmup_s` | seconds stepped before you get the fly, so it starts at rest |
| `learn` | give it a mushroom body that learns (dopamine-gated KC -> MBON plasticity) |
| `surgery` | `{spec: -1 or +1}`, as in protocols: silence or stimulate from the start |

## Naming neurons

Anywhere a method takes `spec`, use what protocols use:

- a group: `"loom"`, `"dnp01"`, `"mdn"`, `"sweet"`, `"bitter"`, `"jo_ce"`, `"adn"`, `"mn9"`, `"pip10"`, `"p1"`,
  `"ps1"`, `"co2_orn"`, `"trn_vp2"`, `"optomotor_right"`, `"mn_front"`, ... (`kickthefly/lab/assays.py:groups`), or a
  game group (`"head"`, `"body"`, `"legs"`, `"heat"`, `"cold"`, `"reward"`, `"escape"`, ...)
- `"type:DNp01,MDN"`, `"prefix:KC"`, `"superclass:descending_neuron"`, `"rows:1,2,3"`
- a list or array of neuron rows

`fly.neurons(spec)` returns the rows; `fly.describe(row)` returns type, instance, superclass and body ID.
For free-text search and paths between neurons, `kickthefly.lab.neurosearch.search(fly.brain, "DNa02")` and
`neurosearch.top_paths(fly.brain.sim.W_csr, src, dst, k=5, Wc=fly.brain.sim.W_csc)` are what the big brain view uses.

## Doing things to it

| method | what it does |
|---|---|
| `drive(spec, amp=0.5)` | holds a current on these neurons every step until `undrive()`, like optogenetic activation. 0.5 is the validation suite's drive |
| `undrive(spec=None)` | stops driving these neurons, or all of them |
| `silence(spec)` / `stimulate(spec)` | brain surgery OFF / ON, the game's surgery currents |
| `restore(spec=None)` | undoes surgery on these neurons, or everywhere |
| `poke(region, strength=0.8, side=None)` | a game-style sensory hit: a share of the region's sensory neurons fires briefly (`"head"`, `"body"`, `"legs"`, `"wing"`, `"heat"`, `"cold"`, `"wind"`, `"light"`, `"smell"`, `"taste"`, `"loom"`, ...) |
| `step(seconds)` or `step(steps=n)` | advances 5 ms per step on the calling thread; returns the last step's spike vector |

All of these are connectome manipulations or the game's own stimuli. Which parts of the game are game rules is
unchanged (the docstring of `kickthefly/game/kick_the_fly.py`).

## Recording and export

`rec = fly.record(groups)` starts recording `{name: spec}` (or a list of specs, or one spec) and returns a
`Recording`:

- `rec.rates(start_s=0, stop_s=None)`: mean spikes/s per neuron of each group, in a time window since the recording
  started
- `rec.spikes()`: `(times in s, neuron rows)` for every recorded spike, in time order
- `rec.groups`, `rec.seconds`

`fly.export(path, nwb=False, note="")` writes the recording the way Lab > Record does: `<path>-spikes.csv`,
`-rates.csv`, `-group-rates.csv`, `.npz` and `-metadata.json` (app version, seed, backend, parameters, brain pack
checksum...). With `nwb=True` (`pip install pynwb`) it also writes `<path>.nwb`.

## Reproducibility

Everything runs in lockstep on your thread: the same seed, backend family and calls give the same spikes. `cpu`,
`numba` and `torch-cpu` are bit-exact with each other; GPU backends agree statistically (README: Deterministic runs).
For many flies or seeds at once, protocols (`python kick_the_fly.py --headless --protocol FILE`) already run seeds in
worker processes.


## Tool loadouts (2.13)

```python
fly = Fly(seed=1, mode="lab")          # mode: "play" (default), "lab" (the laser is allowed) or "pet"
fly.loadout.tools                      # ['hand', 'swatter', 'torch', 'freeze', 'sugar'] in Play (Base)
fly.loadout.page_tools()               # one page of ten
fly.set_loadout("chaos")               # a preset: base, chaos, chemist, lab, all, pet, auto
fly.set_loadout(["swatter", "cva"])    # or your own list (the hand is added first)
fly.use_tool("sugar")                  # drives the tool's documented sensory neurons once (Brain.poke); fly.tool == "sugar"
```

`fly.use_tool` is the sensory drive only: no body, no room. A tool the mode doesn't allow raises `ValueError`. The presets and the
tool catalog (`ToolInfo`: category, description, the neurons it drives, CONNECTOME or GAME RULE) are in `kickthefly.core.loadout`.


## Neurodex, kill cam (3.0)

```python
from kickthefly import Fly

fly = Fly(seed=7)
prog = fly.collect()               # collect Neurodex discoveries while stepping; a throwaway collection, never your own file
fly.step(6.0)                      # the discovery rule needs the first 5 s to settle
fly.drive("type:DNp01", amp=0.5)
fly.step(3.0)
print(fly.discoveries)             # [(brain time s, "DNp01", "stimulated")]
print(fly.neurodex("DNp01"))       # dataset facts, the curated fact and citation (if any), discovered or not

fly.killcam()                      # keep the last 6 s of per-neuron firing
fly.step(3.0)
fly.kill()
rep = fly.killcam_replay()         # kickthefly.core.killcam.Replay
print(rep.summary(fly.brain.types)["risers"][:3])
```

`collect(progress=path)` writes to that file; pass `kickthefly.core.neurodex.progress_path()` to add to your own Neurodex. The rule
and its numbers are in `kickthefly/core/neurodex.py` and [docs/neurodex.md](neurodex.md). Share codes are `kickthefly.core.sharecode`
(`encode`, `decode`, `validate`, `preview`) and bundles are `kickthefly.lab.bundle` (`create`, `inspect`, `rerun`).


## 3.0 day 2: toolkit methods

| call | what | tag |
|---|---|---|
| `fly.line("SS00727")` | a driver line's cell types, matched types, neuron count and the source's off-target note; `"line:SS00727"` works as a neuron spec everywhere | LITERATURE + CONNECTOME |
| `fly.express("trpa1" \| "shibire", spec)`, `fly.temperature(c, kinetics=)`, `fly.unexpress()` | thermogenetics; effectors follow the temperature during `step()` | GAME RULE + MODEL |
| `fly.patch(neuron, amplitudes, duration_ms, repeats, mode="embedded" \| "isolated")` | virtual current clamp, I-F curve and recording | MODEL |
| `fly.image(seconds, rois=None, indicator="gcamp6s", fps=20)` | simulated GCaMP imaging of the next seconds; returns an `ImagingResult` | MODEL |
| `fly.drug(name, dose, include_low_confidence=True, cut=0.7)`, `fly.washout()` | synaptic scaling by predicted transmitter | MODEL PREDICTION |
| `fly.attack(kind, seed=None)` | a frog, dragonfly or mantis attack through the looming pathway; whether DNp01 crossed the escape threshold before the capture (3.0 day 3) | GAME RULE attack, MODEL PREDICTION result |
| `fly.weather(rain, gust_hz, storm, wind_speed, wind_dir)` | rain on touch neurons, humid air, gusts on JO-C/E, lightning on the eyes (all off clears) | GAME RULE |
| `fly.hear(hz, seconds, ipi_ms=None, amp=0.1)` | a synthetic hum through the microphone's analysis onto JO-A/B; never a microphone | GAME RULE |
