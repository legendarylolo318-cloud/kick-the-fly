# Changelog

## 2.10.0 (2026-09-27)

### Added
- **cVA pheromone tool** (mouse wheel or toolbar): drives the DA1 olfactory receptor neurons (the Or67d cVA sensors).
  Validated as activation: DA1 ORNs -> DA1 PNs (x2.58) and DA1 PNs -> lateral horn / aSP targets (x2.22).
- **Decoy female** (mouse wheel or toolbar): foreleg contact drives LgLG5-8, the foreleg taste neurons annotated
  putative ppk23/ppk25. Their drive reaches P1 only weakly (validation FAIL); the COURTSHIP tag is a game rule.
- **Validation:** or67d_to_da1pn and da1pn_to_lh_asp (PASS); foreleg_grn_to_p1, mb_extinction and mb_second_order
  (FAIL: extinction and second-order conditioning don't emerge from the existing learning rule; none was added).
- **Replay files (.ktfreplay), headless:** `--headless --protocol FILE --record-replay OUT` and
  `--headless --replay FILE --out DIR`, with the brain pack's SHA-256 and a spike checksum; exact on NumPy, Numba and
  torch-cpu.
- **Language setting** (Settings > Accessibility): the pause menu and Settings labels; German is machine-translated
  and incomplete.
- **Plume tracking assay** (Python API, a game rule throughout; the open field has no plume).
- **CI:** gl backend on llvmpipe, Flatpak build, docs link check and replay determinism (checks.yml); releases wait
  for them.

### Changed
- **gl backend: batched multi-fly.** A process's brains step together on one GL context, the connectome read once per
  step for up to 32 flies, bit-exact with unbatched gl. 1 fly 0.96 ms/step (2.37 in 2.9), 16 flies in real time
  (0.11x in 2.9), 32 at 0.86x. Its fly cap stays 32 and no longer goes to 64 with KICK_THE_FLY_EXPANDED_SWARM.
- **Real-science cards are off by default**, in Play and Lab; older configs switch them off once.

### Fixed
- gl: Lab parameter changes after a brain's first step never reached the GPU; a brain lost its membrane state when it
  moved from its warm-up thread to its own.
- The benchmark kept each fly count's brains alive into the next count.

## 2.9.0 (2026-09-26)

### Added
- **Courtship song, aggression, thermo arena, day/night** (science 2.9). pIP10 -> ps1 wing motor neurons (validated)
  with a synthesized pulse-song buzz (game rule); an aggression LUNGE read from AVLP727m ("TK-FruM") and the pC1
  cluster, P1 included, in 2D and 3D; the thermo arena (a cold-to-hot floor driving the real hot and cold antennal
  neurons) in 2D and 3D; an outdoor day/night cycle that drives the photoreceptors and morning clock neurons, with a
  dorsal fan-shaped body SLEEP readout (off by default). Lunges and sleep need P1 or FB6/FB7 stimulated in surgery:
  nothing in play drives them far enough on its own.
- **Validation:** pIP10 -> ps1 (PASS), optomotor T4/T5 -> DNa through a game-rule EMD stage (PASS), bitter -> DNg28,
  CO2 -> V PNs, hot -> VP2 PNs, cold -> VP3 PNs (PASS, as one-synapse activation), P1 -> ps1 (FAIL, too weak), the
  Seeds et al. 2014 grooming hierarchy (FAIL). docs/validation.md.
- **Brain view search and path tracer:** find a neuron by type, instance or body ID; trace the 5 strongest paths of
  up to 3 synapses between two neurons, with live spikes along them.
- **Python API:** `from kickthefly import Fly` (docs/api.md, docs/api_example.ipynb).
- **Gamepad** in the 3D game, rebindable in Settings > Controls.
- **Flatpak manifest** for local builds (packaging/flatpak/).

### Changed
- **gl backend:** learning uploads only the KC -> MBON synapses it changed (68 KB and 0.03 ms per update instead of
  41 MB and 2.1 ms; #2). Still not in `auto`: it is slower than NumPy on the machines tested.
- README restructured: Download and contents at the top; Lab, validation and performance detail moved to docs/.

### Fixed
- The 3D brain panel's whole-brain trace crashed with NumPy 2 and pygame-ce 2.5 (float32 points).
- Settings > Brain > Compute backend said `auto` picks OpenGL; it doesn't. The fly cap is documented as it is
  (GPU backends 32, or 64 with KICK_THE_FLY_EXPANDED_SWARM=1).

## 2.8.3 (2026-09-20)

### Added
- **ModernGL compute shader backend (`gl`, experimental)**: Vendor-neutral OpenGL 4.3+ compute shaders (`cs_spmv`, `cs_lif`) over SSBOs, bit-exact with the CPU path and needing no PyTorch. Not chosen by `auto` and not yet a speedup: reading the spike buffer back each step costs more than the compute saves (2.35 ms/step against NumPy's 1.41 on a Radeon RX 9070 XT), and plasticity re-uploads the whole 41 MB weight buffer every 10 steps. Select it with `--backend gl`.
- **Zero-copy device-resident state**: Membrane potentials, refractory counters, spike buffers, and pre-scaled noise buffers remain resident in VRAM across simulation steps in `TorchBackend` and `GLBackend`, dropping single-fly latency by ~1.9x.
- **Batched multi-fly SpMM (`torch-rocm`, `torch-cuda`)**: Streams the connectome sparse matrix once per step across N flies, achieving ~0.32 ms/fly step latency with 32 concurrent flies.
- **Fused LIF kernel execution**: Added optional `fuse_lif` torch.compile elementwise kernel fusion.
- **Simulation telemetry & latency metrics**: Added per-fly uncapped step latency (ms) across `--benchmark` reports, JSON exports, and the Lab simulation benchmark dashboard.
- **Dynamic Fly Cap**: PyTorch GPU backends scale to 32–64 concurrent flies; Numba scales with core count.

## 2.8.2 (2026-09-20)

### Fixed
- **The game crashed when a second sugar pile, alcohol drop or bomb was used up** ("The truth value of an array with
  more than one element is ambiguous"), in both the 3D and the 2D game. These are removed from a list of dicts holding
  numpy arrays, and `list.remove()` compares with `==`: for anything but the first item in the list, that compares
  numpy arrays and raises. They are now removed by identity (`drop_item`). The same bug was waiting in the screen
  flashes and popups. Present since the tools were added.

## 2.8.1 (2026-09-19)

### Changed
- **The fly looks like a fruit fly, not a bee.** It was a honey-gold body with wasp bands wrapped all the way around a
  round abdomen, and small eyes. Now: eyes about three times the area and bright red, filling most of the head as a real
  *Drosophila*'s do; a pale yellow-tan body instead of honey gold; a shorter abdomen tapering to a dark tip; dark bands
  only across the top of each segment, fading out underneath and at the tip, instead of wrapping right around; thinner,
  paler legs; clearer, longer wings; bristles on the thorax and head; and a scutellum behind the wing bases. Only the
  drawing changed; the simulation, physics and hit detection are untouched.
- Every README screenshot and the demo GIF regenerated with the new model.
- The orchard feeding test seeds the random generators itself: earlier tests in its module run for a wall-clock
  duration, so the fly's choice of fruit shifted with machine load and the test failed about half the time.

## 2.8.0 (2026-09-19)

### Added
- **Simulation backends** (`--backend`, Settings > Brain > Compute backend): the LIF step runs on NumPy (`cpu`, the
  reference), Numba (`numba`) or PyTorch (`torch-cpu`, `torch-cuda`, `torch-rocm`). `auto` picks a GPU, then Numba,
  then NumPy, and a backend that can't start falls back to NumPy with the reason in the log. Numba and PyTorch are
  optional and only used from source; the exe and AppImage include neither.
- **Determinism across backends:** `numba` and `torch-cpu` are bit-exact with NumPy (same float32 operations in the same
  order): `tests/test_backends.py` compares every spike, membrane potential and the gain over 1000 steps in float32 and
  float64, and the full validation suite gives identical results on all three. GPU backends are held to a statistical
  tolerance (brain-wide rate within 2%, per-population rates r > 0.95); they were not run on GPU hardware for this
  release (see the release notes).
- **State precision** float32 (default) or float64 (`--dtype`, Settings > Brain > State precision).
- **More flies:** the N cap follows the backend that actually runs: 16 on NumPy and `torch-cpu` (unchanged), one per CPU core between 16 and 32 on Numba, 32 on a GPU
  (unmeasured); and a spawn is refused, with the reason, when there isn't about 700 MB of memory free. **F** picks which fly the brain panel,
  surgery and training follow for 5 s.
- **Video recording of any length** (Shift+R, `--record-video [PATH]`): MP4 through ffmpeg (WebM by name), otherwise a
  GIF (downscaled, up to 60 s). Frames are paced by the wall clock, so the video plays at real speed; saved with the
  screenshots.
- **Real neuron shapes for ten neurons:** two each of DNp01, DNa02, MBON01, MBON14 and KCg are drawn from 21 points
  sampled along their EM skeletons from neuPrint (MaleCNS v1.0), downloaded once and cached; offline they fall back to
  estimated fibers, and the big brain view says which. Drawing only: the simulation is point neurons either way.
- **Benchmark reporting:** `--benchmark` and Lab > Simulation benchmark take a backend and report the one that ran, its
  device, neuron updates/s, synaptic events/s (measured spikes x mean out-degree; before this it was an estimate from
  an assumed 2.5% activity), and the process's current memory. Validation results, save states, NWB exports and crash
  reports record the backend and device too.
- **Profiler** `tools/profile_sim.py`: where a step's time goes, for 1, 8 and 16 flies.
- **Citation:** `CITATION.cff` (GitHub's "Cite this repository"), citing the connectome too. `.zenodo.json` holds the
  metadata for a Zenodo DOI if Zenodo archiving is switched on later; it isn't yet.
- **Screenshots:** every README image regenerated from this build by `tools/make_screenshots.py`, plus new ones of the
  open field, the orchard, the escape room, the settings menu, Lab mode, the validation dashboard and the laser, and an
  animated demo at the top.

### Fixed
- **Spawning another fly (N) did nothing** in the first cut of this release (a lost import killed the spawn thread);
  a failed spawn now logs why and N keeps working.
- **The fly's state title** (FLYING, STUCK ON FLYPAPER, WRAPPED IN SILK...) was missing from the HUD.
- **Neuron shapes never loaded** in the first cut (a keyword mismatch was hidden by the fallback), and the skeleton
  cache pointed inside the exe/AppImage; it's in your data folder there now.
- **The big brain view's header** no longer draws its buttons, counts and status lines over each other.
- **The Numba kernels** now reproduce NumPy bit for bit (they used fastmath and float64 intermediates, and a run
  drifted apart after ~300 steps) and release the interpreter lock, so flies' brains run in parallel.
- **The PyTorch backend** kept its own copy of the brain state, so loading a save, the neural clamp or a float64
  switch didn't reach it; `torch-cpu` could not be selected at all, so the old backend test compared NumPy with itself.
- **The video recorder** stopped on plain R (reset fly / respawn in the duel), wrote to the current directory,
  sped the video up when the game drew slower than 30 fps, broke on a window resize and kept every GIF frame at full
  size in memory.
- Screenshots and GIFs save on systems whose pygame lacks PNG support (Pillow fallback).
- **The escape room's speedrun timer** showed the machine's uptime (thousands of seconds) when the game started in
  the escape room (from the saved arena or `--arena escaperoom`): the 3D game set its arena up at time 0 on a clock
  that doesn't start at 0, which also gave an orchard chosen at startup the wrong time. Present since 2.7.0.

## 2.7.2 (2026-09-18)

### Fixed
- **The blowtorch flame and the brake cleaner and freeze sprays now come out of the tool in your hand** (3D). They
  were started 0.7 m behind your head and flew forward through the camera. Thrown bombs, sugar and alcohol, the
  zapper and the laser also start at your hand now. The fire point follows the tool's nozzle on screen at any field
  of view.
- **The laser's key shows as "=" on the toolbar** instead of "12". It always worked on =; `=` is now also reserved so
  it can't be bound to another action.

## 2.7.1 (2026-09-18)

### Fixed
- **The Linux AppImage started in 2D on current distros** (2.6.0 and 2.7.0, seen on Arch with Mesa radeonsi): it
  bundled the C++ runtime and X11 client libraries from its Ubuntu 22.04 build machine, which are too old for a new
  graphics driver, so OpenGL failed to load on Wayland and X11 alike. Those libraries now always come from the host,
  as the AppImage project recommends, and the build fails if any of them slip back in. Native Wayland is still used
  first, with X11 as the fallback.

## 2.7.0 (2026-09-17)

### Added
- **Two outdoor arenas (3D), Open field and Orchard,** selectable with E, in Settings > Brain > Arena, and with `--arena`; the room stays the default.
  - *Open field:* 30 m x 30 m of ground, rocks, grass and sky, with much larger bounds than the room and no ceiling for the fly. Steady wind drives the real JO-C/E wind neurons of each antenna, and sunlight drives the photoreceptors; wind direction and speed and the sun's azimuth and elevation are Lab parameters. Escapes carry the fly away outdoors; a fly more than 26 m from you or 15 m up is lost, and **J** calls it back.
  - *Orchard:* 24 fruit trees. The fly flies to a ripe fruit, lands and feeds, which drives the real sugar-pathway taste and PAM reward neurons exactly as the sugar tool does and heals it; fermented fruit act like the alcohol tool. Each fruit holds a limited number of feeds (default 4), shrinks and browns as it's eaten, then drops, and grows back after about 75 s, staggered, with a per-tree cap (default 4). The fruit, trees and flying to them are game rules and are tagged RULE. Several flies compete only through the looming and touch neurons they already had.
  - Feeds per fruit, regrow time and the cap are Lab parameters, valid protocol `params`, and the new headless `assay: orchard` (`protocols/orchard-feeding.yaml`) runs the feeding schedule reproducibly.
  - The arena is saved in `config.toml`, in save states (by name) with the orchard's fruit, in export metadata with its weather and fruit settings, and in the `--smoke` status line.
- **E-PG compass, second test: steady directional wind** (`validation: epg_compass_wind`). The open field's wind, through the arena's own transduction, from 8 directions, with the visual test's pass criteria fixed beforehand and nothing tuned. It fails: contrast 1.81x (1.71x without wind; 3.0x needed), persistence 101 ms (500 ms needed), direction tracking |r| 0.46 vs 0.40 for shuffled directions (p = 0.17). No compass HUD.
- **Repo restructure:** the ~25 root modules moved into a `kickthefly/` package (`core`, `sim`, `game`, `ui`, `lab`, `data`) with `git mv`. `python kick_the_fly.py` still works through a shim; `python -m kickthefly` does the same. `CONTRIBUTING.md` says where code goes.
- **NWB export:** recordings (and protocols with `nwb: true` or `--nwb`) as one Neurodata Without Borders file with units, rates, stimuli, events, kinematics, surgery, arena, KC->MBON weights before and after, and full metadata with the MaleCNS v1.0 / CC BY 4.0 citation. Optional `pynwb`, not bundled.
- **Synapse threshold slider and report** (Lab > Connectome robustness; `--threshold-sweep`).
- **Sign-flip stress test** on the dataset's transmitter predictions and confidence, with per-neuron flips from the inspector (`--signflip-test`). The brain pack now carries each neuron's transmitter, its confidence and where it came from.
- **Critical path finder** (Lab; `--critical-path TARGET`), resumable, with one-click "apply this lesion".
- **Optogenetics Laser (Lab > Laser):** In-world aimable beam that activates or silences selected cell types directly in real time, replacing menu-only surgery for rapid targeted stimulation/silencing experiments.
- **Psychometrics Curve Generator (Lab > Psychometrics):** Sweeps stimulus parameters across continuous ranges, executes N trials with 95% confidence intervals, and exports publication-ready vector figures (PDF-1.4, SVG) and raw CSV data.
- **E-PG Compass Validation Test (Lab > Validation):** Negative validation assay testing whether a driven E-PG wedge forms a persistent head-direction bump (it doesn't: no self-sustaining bump, so no compass HUD is displayed).
- **Classroom Mode & Lecture Protocols (Lab > Classroom):** Self-contained educational modules with guided steps, hypothesis prompts, and bundled interactive YAML lecture protocols (`protocols/lecture_*.yaml`).
- **Mystery Defect / Reverse Brain Surgery Challenge (Play mode):** Curated circuit silencing challenges without jargon where players test the fly with tools, request hints, and deduce missing neural circuits.
- **Predict-the-Neuron Minigame (Play mode):** Reaction minigame restricted strictly to descending motor readouts (jump, run, kick, back up, take off) where players predict upcoming movements from motor surge cues.
- **Escape-Room Arena (Play mode):** Multi-hazard emergent navigation gauntlet combining fan wind, flypaper strip, and hot lamp overhead; reach the sugar dish to stop the speedrun timer and generate a tamper-evident verification code (`KTF-<SEED>-<TIME>-<SIG>`).
- **Neural Clamp (Lab > Neural clamp):** Record reference spike trains from calibrated runs (looming, antennal grooming, sweet taste, calm baseline) and replay forced spikes into altered connectomes (lesion, threshold pruning, transmitter sign flips) to isolate structural changes from sensory feedback. Side-by-side activity diff table and export. Explicit dynamic clamp notice in UI and exports that forced spikes override membrane state and break closed-loop feedback.
- **Connectome Diff Mode (Lab > Connectome diff):** Run two flies (reference vs perturbed) side-by-side with identical seeds, initial states, and sensory inputs in lockstep. Real-time region-by-region activity divergence with autopsy-style diverging bar charts, timeline tracking the moment trajectories split, and CSV/JSON export.
- **Hemifield & Hemisphere Silencing:** Brain surgery presets for unilateral visual pathways (`LC10`, `LPLC2`, `LC4`, `LPTC`, `VS`, `HS`) and whole hemibrains. Validated behavioral consequences: failure to dodge blind-side looming, asymmetric steering bias (-1.8 Hz vs +0.10 Hz baseline), and broken 1v1 duel aiming on the blind side. Reported strictly as connectome wiring outcomes.
- **Global Inhibition Block / Picrotoxin (Lab > Robustness > Inhibition block):** 0-100% severity slider scaling inhibitory synaptic weights (`inhibition_scale = 1 - severity`). Emergent runaway excitation (>30 Hz brain-wide mean) without scripted seizures; before/after firing rate distributions; convulsion twitching and severity labels tagged as game rules.
- **Research findings in the docs:** the synapse threshold sweep (dropping every connection below 10 synapses removes 73.6% of connections and every input of 10,151 neurons; all 4 validated behaviors survive), sign flips (looming and antennal grooming survive 3/3 trials, sugar -> MN9 fails 3/3), and the looming critical path (LC4 -50%, LPLC2 -43%).

### Changed
- Experimental activation (`simcore.drive`) is its own current, added to surgery instead of replacing it, so a silenced cell type stays silenced when an assay drives it.
- Outdoor-scale rendering: static scenery matrices are built once, instance data is packed in bulk, and collision boxes far from the fly are skipped. With one fly every arena runs at real time and 62 fps; with 8 flies the outdoor arenas run within ~20% of the room (see README > Performance).
- Lab hub scrolls; Model assumptions gains entries for weak connections, transmitter signs, outdoor transduction, the orchard, the alcohol/zapper scent overlap and the compass.

### Fixed
- Dragging across the big brain view no longer freezes the game (the view is re-projected on the brain-view thread, not on every mouse move), and clicking a neuron no longer crashes the inspector.
- Fullscreen scales correctly in 2D and 3D: the 3D window opens at the display's size instead of toggling after creation, F11 works in both, and the 3D HUD is never shorter than 760 units (the bottom toolbar used to be cut off at 1440p).
- The laser works in the 3D game (beam, hit feedback, viewmodel, HUD badge), pulses finish instead of stopping when the mouse is released, and several flies no longer overwrite each other's laser state.
- `--autopilot` and `--mirror-weights` crashed at startup (`Config` doesn't support item assignment).
- Holding the laser near the fly crashed the 3D game (its scent group is empty).
- A packaged build with a pack older than 2.6 crashed on startup instead of leaving regions unassigned.
- README numbers from the previous session corrected against re-measurement (threshold 10 cuts off 10,151 neurons' input, not 12,234; the sign-flip looming effect is x13.2, not x10.9).

## 2.6.0 (2026-09-17)

### Added
- **Pause menu and Settings.** Esc opens Resume, Challenges or Lab tools, Settings, Save State, Load State, Mode and Quit (with confirmation). Settings has Graphics, Audio, Brain, Controls and Accessibility tabs with hover tooltips, sliders that take typed values, per-tab reset, rebindable keys with conflict swapping, and Connectome / Game rule tags on brain settings. Saved to `config.toml`.
- **Play and Lab modes.** Play (default) has three challenges with best scores: Teach it to pick the right door (T-maze), How close can you sneak? (looming escape) and Find its sweet tooth (sugar response). Lab adds the validation dashboard, assays and repeated trials with 95% CIs and same-seed controls, live parameters, recording and export, and protocols.
- **Real vs rule tags** on reactions and popups (on by default in Lab).
- **Validation suite** (`--validate`, `pytest`) with held-out seeds and fixed criteria. Passes: looming -> giant fiber, sugar-pathway taste neurons -> MN9, antennal JO-C/E -> aDN1/aDN2, T-maze conditioning. Fails, reported as such: MDN -> backward walking, aDN -> front-leg motor neurons.
- **Real-science cards** in Play, the first time a validated behavior happens (only for passing tests).
- **Time controls:** pause (Z), slow motion 0.1x-1x ([ ]), single step (.), with an on-screen badge; the room and all brains slow together.
- **Save states:** the whole simulation in a versioned, platform-independent `.ktfsave` file.
- **Protocols:** YAML stimulus or assay protocols, from Lab > Protocols or `--headless --protocol FILE`, with spike/rate CSV and npz exports and metadata.
- **GROOM and PROBOSCIS reactions** read from aDN1/aDN2 and MN9; the proboscis extends while eating sugar.
- **Accessibility:** blue/yellow and high-contrast brain palettes, reduced flashing, larger text.
- **Linux:** XDG Base Directory locations (with a one-time copy of older data), native Wayland with automatic X11 fallback and `--backend`, AppStream metadata and desktop file in the AppImage, an AUR `kickthefly-bin` PKGBUILD.
- **Windows:** per-monitor DPI awareness, Known Folders for Documents and Pictures, version info in the exe's properties.
- **Crash reports** with version, seed, OS, session type, video driver, GPU, OpenGL and driver versions, also written to the per-user state folder.
- **Release workflow** that builds the brain pack, runs all tests on Ubuntu 22.04 and Windows, builds the AppImage and exe, and publishes both with SHA256SUMS.

### Changed
- Esc no longer quits; it closes a panel or opens the pause menu.
- A missing OpenGL 3.3 is logged clearly before falling back to 2D.
- The brain pack also stores subclass labels and FlyEM body IDs (all existing arrays unchanged, so saved training memory still loads).
- The multiple-flies notes in the README match the game (up to 16 flies; the panel follows the nearest fly).

### Fixed
- Two screenshots or GIFs saved in the same second no longer overwrite each other.
- Crash reports from the AppImage no longer try to write inside its read-only mount.

## 2.5.0 (2026-09-15)
- The brain panel follows the nearest fly; up to 16 flies.
