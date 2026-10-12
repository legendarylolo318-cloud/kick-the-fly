# Kick the Fly

A kick-the-buddy game whose buddy is a real fruit fly brain: all **166,700 neurons of the MaleCNS v1.0 connectome**, simulated live while you
throw, swat, burn, freeze and feed it to a spider, or reward it with sugar. Every hit lands on the fly's real sensory neurons, and what it does
next is read from its real descending neurons. Switch to **Lab mode** and the same brain becomes a research bench: validation against published
experiments (including the ones it fails), assays, classic behavior rigs, a virtual patch clamp, calcium imaging, and protocols that run headless.

![Kick the Fly in the open field: swatting and torching the fly while its live brain lights up on the right](docs/demo.gif)

*The open field in first person: a swat and the blowtorch fire the fly's real touch and heat neurons, and the panel on the right shows its whole
brain responding.*

**Version 4.0 is a new engine.** The game was rewritten from Python/pygame to Rust and [Bevy](https://bevyengine.org/): same game, same
brain model (the Rust simulator reproduces the Python one spike for spike), a native program with no Python inside. The original Python game
(3.1.1) is kept on the [`python-legacy`](https://github.com/legendarylolo318-cloud/kick-the-fly/tree/python-legacy) branch and the tag
`python-3.1.1`. What 4.0 does not have yet is listed under [What is different in 4.0](#what-is-different-in-40).

## Download

| | |
|---|---|
| **Windows** | **[KickTheFly.exe](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/latest/download/KickTheFly.exe)**. Double-click it: nothing to install, the whole brain is inside. Windows 10 or 11, 64-bit, a GPU with DirectX 12 or Vulkan. |
| **Linux** | **[KickTheFly-x86_64.AppImage](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/latest/download/KickTheFly-x86_64.AppImage)**. `chmod +x` it and run it. Needs a Vulkan driver (`vulkan-icd-loader` plus your GPU's driver). On a tiling Wayland compositor start it with `--fullscreen`. |
| **From source** | Any OS with Rust: see [Build from source](#build-from-source) below. |

Each release lists `SHA256SUMS`. Run the program with `--selftest` (or Settings > Help) to check an install; it explains how to fix whatever
it finds and changes nothing. Settings, saves, share codes and the pet from the Python game are read as they are.

## What is different in 4.0

- **Not in 4.0 (they are in the Python version):** the 2D game (`--2d` is accepted and ignored), the larva brain, the Numba / PyTorch / OpenGL
  compute backends (4.0 has one multi-threaded CPU backend), the Python API, NWB and TIFF export, GIF / video / time-lapse recording,
  replays, and a few command-line analyses. The full list, with what was checked and how, is [PORTING.md](PORTING.md).
- **Works a little differently:** the brain panel and the big brain view are a 3D cloud of every neuron instead of a flat picture; photo mode's
  blur is Bevy's; pictures are the window's size; two Graphics settings (resolution scale, screenshot resolution) have no effect yet.
- **New:** a native Wayland and X11 window, a profiler overlay with GPU pass times (F3).
- **Speed, measured:** one brain step takes about 0.6 ms (the Python NumPy backend: roughly 2 ms), and the brain runs on its own thread, so
  several flies no longer slow the picture down as much. The picture itself is *not* faster than the Python game's at the moment: the HUD
  is drawn in software and is the bottleneck (see [Performance](#performance)).

## Build from source

Rust 1.95 or newer through `rustup`, and a Vulkan driver.

| OS | Install |
|---|---|
| **Arch Linux** | `sudo pacman -S rustup vulkan-icd-loader && rustup default stable`, plus `nvidia-utils`, `vulkan-radeon` or `vulkan-intel` |
| Debian / Ubuntu | `sudo apt install build-essential pkg-config libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev libx11-dev`, then [rustup.rs](https://rustup.rs) |
| Windows | [rustup.rs](https://rustup.rs) with the MSVC build tools (`x86_64-pc-windows-msvc`) |

**The brain pack.** The connectome is `data/kick_brain.npz` (37 MB, not in git: it is built from the public MaleCNS v1.0 download with the dev
tool in `legacy-python/`, or taken out of a release's AppImage). It is found from the working directory upward, next to the program, or at
`$KICK_THE_FLY_PACK`; `--pack FILE` overrides. Without it a synthetic random network runs, which is not the fly.
`cargo build --release --features bundle-pack` compiles the pack into the program.

```bash
cargo run --release -p kick_the_fly -- --fullscreen   # the game (--fullscreen: tiling compositors open half-size windows otherwise)
cargo run -p kick_the_fly --features dev              # fast rebuilds: Bevy linked dynamically (never for release)
cargo test --workspace --release                      # unit tests and the goldens against the Python reference
cargo fmt --check && cargo clippy --workspace --all-targets --release
./build_appimage.sh                                   # dist/KickTheFly-x86_64.AppImage
```

A release build links everything with fat LTO: about ten minutes, and the test suite links some sixty such binaries (on a machine with less
than 32 GB of memory add `-j 3`, or set `CARGO_PROFILE_RELEASE_LTO=off` as CI does). `H` in the game lists the keys; Esc opens the menu.
`KICK_THE_FLY_HOME=<dir>` puts settings, saves and pictures under one folder.

**Command line.**

| Flag | What it does |
|---|---|
| `--pack FILE`, `--synthetic N`, `--seed S` | which brain, and the noise seed |
| `--arena NAME`, `--flies N`, `--rain F --storm --day S` | start in an arena (room fan flypaper pool lamp escaperoom field orchard thermo kitchen), with N flies, with weather |
| `--fullscreen`, `--autopilot` | borderless fullscreen; spectator mode (hands off, the big brain view up and turning) |
| `--smoke SECONDS` | no window: run the brain thread and report steps per second |
| `--selftest [--out FILE.json]`, `--bugreport [--out DIR]` | check this install; print and write what a bug report would contain (nothing is sent) |
| `--shot FILE [--shot-frames N]`, `--cam x,y,z,tx,ty,tz`, `--menu SCREEN`, `--tool NAME`, `--photo DOF,FOCUS` | screenshot harness |
| `--headless --validate [--seeds 1000-1009 --workers N --out FILE --strict]` | the 21 validation tests, no window |
| `--headless --protocol FILE [--out DIR --bundle ZIP]`, `--headless --rerun-bundle ZIP --out DIR` | run a protocol and bundle it; rerun a bundle and judge whether it matches |
| `--headless --rig NAME`, `--rig-assay NAME`, `--minipaper ID` (or `list`) | a behavior rig, its pre-registered assay, a mini-paper |
| `--headless --activation-screen`, `--knockout-screen [BEHAVIORS]` | the resumable whole-brain screens |

**Layout.** `crates/fly_sim` (the simulator and every engine: pure Rust, no Bevy), `crates/ktf_data` (config, stores, saves, recorder, bundles,
protocols, pet, streamer rules), `crates/ktf_ui` (the software-drawn menu pages and HUD), `crates/game` (the Bevy app: render thread, simulation
thread, one plugin per feature). `legacy-python/` is the Python game, unchanged, as the reference every part is checked against: golden data
from it is in `tests/golden/`, page pictures in `tests/ui_reference/`. Notes for developers: `docs/dev/`.

## New in 3.0

<table>
<tr>
<td width="50%"><img src="docs/neurodex.png" alt="The Neurodex page listing discovered cell types by brain region"><br><b>Neurodex.</b> Collect the cell types your fly fires, each with the dataset's numbers and, for curated ones, a cited fact.</td>
<td width="50%"><img src="docs/killcam.png" alt="The kill cam replaying the brain's last seconds in slow motion"><br><b>Kill cam.</b> The last seconds of its brain, replayed in slow motion, with the neurons that rose most.</td>
</tr>
<tr>
<td><img src="docs/rig.png" alt="Lab > Behavior rigs: a fly's walking path between two stripes in Buridan's paradigm"><br><b>Behavior rigs.</b> A tethered flight simulator, a fly on a ball, Buridan's paradigm and an olfactory arena, each with a pre-registered assay. Three reproduce the classic result; the olfactory arena does not, and says so.</td>
<td><img src="docs/minipaper.png" alt="A mini-paper comparing your result on the model with what Colomb et al. 2012 found"><br><b>Mini-papers.</b> Six classic papers as guided experiments: state a hypothesis, run it, and read your result next to what the paper states.</td>
</tr>
<tr>
<td><img src="docs/arcade.png" alt="The fly arcade: a tournament bracket of flies"><br><b>Fly arcade.</b> Tournaments and races where every fly is its own brain. In-game points only: no money, nothing to buy.</td>
<td><img src="docs/whatsnew.png" alt="The What's New in 3.0 screen"><br><b>And more.</b> Predators, rain and storms, a kitchen arena, an opt-in microphone and Streamer mode, a genetic toolkit, thermogenetics, a patch clamp, calcium imaging, pharmacology, network science, sleep deprivation and a sensitivity analysis.</td>
</tr>
</table>

**What did not change: the simulation.** `--validate` gives the same results as 2.13 (21 tests, **12 PASS / 9 FAIL**), and no weight, time
constant or threshold was tuned to make anything pass. Old saves, settings, pet files, replays and protocols keep working. Everything new is
tagged **CONNECTOME**, **GAME RULE** or **MODEL PREDICTION**. The whole release: [docs/changelog.md](docs/changelog.md); how it was checked, the
decisions and the known limits: [docs/3.0-review.md](docs/3.0-review.md).

## Play

First person in a 3D living room. Walk up and use a tool, or walk into the fly to kick it.

- **Hits fire real sensory neurons**: head bristles and Johnston's organ, body touch, leg proprioceptors, wing sensors; the blowtorch fires heat
  sensors, the freeze spray cold sensors, the brake cleaner smell and taste neurons, the zapper every touch neuron.
- **Its reactions are its descending neurons**: running, kicking, walking, backing up (the moonwalker neurons), turning (DNa01/DNa02), taking
  off (DNg02). Move a tool at it fast and its looming detectors LPLC2 and LC4 fire the giant fiber, and it dodges. Sneak up slowly and it won't.
- **It learns** with its real mushroom body: pair a smell with pain or sugar and dopamine reshapes the actual Kenyon-cell-to-output synapses.
- **Ten arenas**: the room, fan, flypaper, pool, lamp, a thermal gradient, an escape room, the kitchen, and outdoors an open field with wind and
  sun and an orchard where flies fly to fruit and feed.
- **Up to 16 flies**, each a complete independent brain; they notice each other only through
  their real looming and touch neurons.
- **Brain surgery, a neuron inspector, a 1v1 duel** in which the fly aims with LC10 and shoots with DNp35, **challenges**, a **Pet mode** with
  one persistent fly, slow motion, save states and screenshots.

Controls (all rebindable, gamepad included), settings, loadouts, challenges and file locations: **[docs/playing.md](docs/playing.md)**.

| | |
|---|---|
| ![first person in the 3D room, the fly on the rug and its brain panel on the right](docs/room3d.png) | ![swatting the fly in first person; touch neurons and descending neurons light up in the brain panel](docs/swat3d.png) |
| **The room**, first person, with the live brain on the right. | **Swatting**: the touch neurons fire, then the leg-touch descending neurons kick. |
| ![the fly flying over the open field](docs/field.png) | ![a fly feeding on a fruit inside an orchard tree's crown](docs/orchard.png) |
| **The open field**: wind drives its real antennal wind neurons. | **The orchard**: flies fly to fruit, land and feed. |
| ![big brain view lit up while the fly is torched](docs/brain.png) | ![the fly shooting at you in the 1v1 duel](docs/duel.png) |
| **The big brain view** under the blowtorch: the heat-sensing neurons and everything downstream. | **The 1v1 duel**: it aims through LC10 and its steering neurons, and shoots through DNp35. |
| ![a spider wrapping the fly in silk](docs/spider.png) | ![the brain autopsy after the fly died under the blowtorch](docs/autopsy.png) |
| **A spider** bites (body and leg touch neurons) and wraps it. | **Autopsy**: every brain region's last 2 s alive against its calm baseline. |

More pictures (flypaper, the lamp, the thermal arena, the escape room, surgery, training, the inspector, the path tracer, the decoy, the laser,
the see-through panel, settings): [docs/gallery.md](docs/gallery.md).

## Lab

Esc > Mode > Lab replaces the challenges with research tools: the validation dashboard, assays over many flies with same-seed controls and
statistics, psychometric sweeps, the optogenetics laser, recording with NWB export, a critical path finder, robustness tests (synapse thresholds,
transmitter sign flips), neural clamp, diff mode, classroom lectures, the 3.0 toolkit and rigs, and YAML protocols that also run headless
(`--headless --protocol FILE`). All of it, with examples: **[docs/lab.md](docs/lab.md)**.

| | |
|---|---|
| ![the Lab tools menu](docs/lab.png) | ![the validation dashboard with PASS and FAIL results](docs/validation.png) |
| **Lab mode**, with the simulation engine shown top right. | **Validation**: which published fly results it reproduces, with numbers, and which it doesn't. |

From a source checkout the brain is also a Python library ([docs/api.md](docs/api.md), with a [notebook](docs/api_example.ipynb)):

```python
from kickthefly import Fly

fly = Fly(seed=1000)
rec = fly.record({"giant fiber": "dnp01"})
fly.step(2.0)
fly.drive("loom")           # LPLC2 + LC4, like optogenetic activation
fly.step(2.0)
print(rec.rates(0, 2), rec.rates(2, 4))
```

## Validation

Does it reproduce published fly experiments? Each test drives a set of neurons for 2 s on held-out seeds (1000-1009) next to a matched control,
with pass criteria fixed before the run: at least 1.5x the calm rate and above the control, one-sided Wilcoxon p < 0.01, n = 10.

| passes | fails |
|---|---|
| looming detectors -> giant fiber (x11.81 vs x0.80) | MDN -> leg motor neurons, i.e. backward walking (x0.96 vs x0.96) |
| sugar taste neurons -> proboscis motor neuron MN9 (x2.11 vs x1.25) | aDN -> front-leg motor neurons (x1.13, too weak) |
| antennal touch -> grooming command neurons aDN (x4.87 vs x0.85) | E-PG head-direction bump, from a wedge or from wind |
| T-maze odor + shock conditioning (PI 1.00 vs -0.03) | P1 -> ps1 wing motor neurons (x1.43, too weak) |
| song neuron pIP10 -> ps1 wing motor neurons (x1.93 vs x0.94) | grooming hierarchy, head over abdomen (Seeds et al. 2014) |
| optomotor: rightward motion -> right steering DNs (x2.72 vs x0.80), through a game-rule motion stage | foreleg pheromone GRNs -> P1 cluster (x1.20, too weak) |
| cVA: DA1 ORNs (Or67d) -> DA1 projection neurons (x2.58 vs x0.80) | mushroom body extinction (PI 1.00 after vs 1.00 without) |
| cVA: DA1 PNs -> lateral horn / aSP targets (x2.22 vs x0.96) | second-order conditioning (odor B PI 0.00 vs -0.03 unpaired) |
| one-synapse activation (not avoidance): bitter -> DNg28, CO2 -> V PNs, hot -> VP2 PNs, cold -> VP3 PNs | |

The full table with SDs, controls, citations and what each failure means: **[docs/validation.md](docs/validation.md)**. The 3.0 rigs add
pre-registered assays on top ([docs/rigs.md](docs/rigs.md)): tethered flight, the ball and Buridan's paradigm pass, the olfactory arena fails.

## Connectome or game rule

A game needs rules the wiring can't supply. Everything is tagged, on screen and in the docs:

| comes from the connectome | is a game rule |
|---|---|
| The spiking model (leaky integrate-and-fire over the brain pack's 10.3 million signed connections) and all firing | Which move each descending-neuron group triggers, and every threshold (Lab > Parameters) |
| Which sensory neurons each hit, tool, arena, scent or taste drives | The body: ragdoll physics, flight paths, damage, death, the pain index (no neurons are annotated as nociceptors) |
| The descending neurons read for reactions (jump, run, kick, walk, back up, turn, take off) | How looming reaches LPLC2/LC4 (an object's growth in view), the optomotor motion stage in front of T4/T5 |
| Looming detectors exciting the giant fiber, sugar neurons reaching MN9, antennal touch reaching the grooming neurons (all validated) | Sugar driving the PAM reward neurons directly, pain driving PPL1 punishment, each tool having a smell |
| The learning: dopamine reshaping the real Kenyon cell to MBON synapses, with the connectome's own dopamine-to-MBON map | The learning rate and forgetting, the T-maze's choice rule, alcohol's inebriation (no neuron is intoxicated) |
| In the duel and the rigs: aiming and turning from DNa01/DNa02, shooting from DNp35, walking from DNp09 | The duel's blaster and arena, the rigs' platforms, balls and arenas, LC10 tracking of a target |
| The Neurodex entries' numbers, the kill cam's replayed firing | What counts as a discovery, the kill cam window, every UI choice |

The complete list: **[docs/connectome-and-game-rules.md](docs/connectome-and-game-rules.md)**. The canonical map is the docstring at the top of
`kickthefly/game/kick_the_fly.py`.

## Performance

The brain runs on its own thread at a fixed 200 steps a second whatever the picture does. Measured on a laptop (i7-13620H, RTX 4060 Laptop,
Linux, 1920x1080, the room, no frame cap):

| | 4.0 (Rust) | 3.1.1 (Python, NumPy, 1280x760) |
|---|---|---|
| one brain step, 166,700 neurons | about 0.6 ms | roughly 2 ms |
| frame, 1 fly | about 13 ms (77 fps) | about 4.5 ms (225 fps) |
| brain steps per second, 4 flies | 200 (real time) | 200 (real time) |

The two frame times are not like for like (the Python game was measured in a smaller, off-screen window), but they say what is true today:
4.0's picture is limited by its software-drawn HUD (about 3 ms of every frame at 1080p), not by the GPU, whose 3D pass takes a quarter of a
millisecond. The default cap is 60 frames a second (Settings > Graphics), which both versions hold. F3 shows where a frame's time goes.

## Repository

```
crates/fly_sim/      the LIF connectome simulator and every analysis engine (pure Rust, golden-tested against the Python reference)
crates/ktf_data/     config.toml, saves, share codes, protocols, bundles and per-OS folders, compatible with the Python game's files
crates/ktf_ui/       the menu pages and the HUD (drawn in software, compared with pictures of the Python game)
crates/game/         the Bevy game
legacy-python/       the Python game 3.1.1, unchanged (the reference; also on the python-legacy branch)
tests/golden/  tests/python_files/  tests/ui_reference/    reference outputs, real files and page pictures from the Python game
docs/                the manual (written for 3.x; where 4.0 differs, PORTING.md says so) and docs/dev/ (working notes)
data/                not in git: the brain pack (bundled in the exe and AppImage)
```

Contributing (layout, tests, screenshots, translations): [CONTRIBUTING.md](CONTRIBUTING.md). History: [CHANGELOG.md](CHANGELOG.md). If you use Kick the
Fly in research or teaching, cite it with [CITATION.cff](CITATION.cff) and cite the MaleCNS v1.0 connectome.

## Credits

The adult connectome data is Janelia FlyEM MaleCNS v1.0, a collaboration between HHMI Janelia, the University of Cambridge, the MRC Laboratory of Molecular Biology and Google Research. It is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and available at [male-cns.janelia.org](https://male-cns.janelia.org/download/).

The Drosophila larva connectome data is from Winding, M., Pedigo, B.D., Barnes, C.L., et al. (2023). "The connectome of an insect brain." *Science*, 379(6636), eadd9330. DOI: [10.1126/science.add9330](https://doi.org/10.1126/science.add9330) (Supplementary Data S1). No license is stated for it, so Kick the Fly does not redistribute it: the larva pack is built on your machine from the downloaded Data S1 (checksum-verified; mirror: [github.com/brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome)).

Real neuron shapes (opt-in, Settings > Brain; [docs/real-shapes.md](docs/real-shapes.md)) are the skeletons of the same MaleCNS v1.0 release (FlyEM at HHMI Janelia with the University of Cambridge, the MRC LMB and Google Research; CC BY 4.0, [male-cns.janelia.org](https://male-cns.janelia.org/)), downloaded one neuron at a time from Janelia's public bucket when you inspect it, checksummed and cached on your machine, and never bundled.

The exe and AppImage bundle a compact pack derived from the adult data only. The pack keeps the signed synapse counts, the neuron labels (type, superclass, subclass, instance), body IDs and the cell-body positions, and is otherwise unmodified.
