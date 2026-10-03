# Changelog

## 3.0.0 (unreleased)

The whole release in one page, by feature: [docs/changelog.md](docs/changelog.md). This file keeps it day by day, with the reviews.

### Day 5 (behavior rigs, mini-papers, release polish)
Added. Nothing the simulation does changed: `--validate` was diffed against the setup baseline on this machine (identical, see HANDOFF_3.0_DAY5.md).
- **Behavior rigs** (Lab > Behavior rigs, `--rig NAME`, `--rig-assay NAME`, `protocols/rig_*.yaml`, `docs/rigs.md`): a tethered flight simulator, a fly on a
  ball, Buridan's paradigm and a four-field olfactory arena. Yaw is read from DNa01 + DNa02 right minus left, walking from DNp09, wide-field motion goes
  through the existing optomotor stage, a stripe or bar through the duel's LC10 tracking rule. Pre-registered criteria (committed before any held-out
  run) and an `--individuality off` control for each. Results on seeds 1000-1009: tethered PASS, ball PASS, Buridan PASS, olfactory arena **FAIL** (PI
  +0.005, p = 0.69); the control gives the same outcomes. Exports in the Lab's recorder format plus a trace CSV.
- **Guided mini-papers** (Esc > Mini-papers, Lab > Mini-papers, `--minipaper ID`, `docs/minipapers.md`): von Reyn 2014, Tully & Quinn 1985, Shiu 2024, Hampel
  2015, Ohyama 2015 and Colomb 2012 (Buridan). Hypothesis, run, plot, then your result next to what the paper states (abstract or full text, said which),
  with why where the model misses. Built on the lecture protocol system (`classroom.EXTRA_LECTURES`); the experiments are the validation's own tests.
- **What's New in 3.0**: one skippable screen on the first launch after an upgrade; `[first_run] whatsnew_3_0_seen` is a migrated config key (a missing
  key means not seen; a fresh install starts with it set); Settings > Help brings it back.
- Neurodex: curated facts for the sugar and bitter SEL neurons (Yao & Scott 2022). Python API `rig`, `rig_assay`, `minipaper`; selftest `day5`; three
  playthrough checks; 81 localization strings; two Lab > Model assumptions cards; the main docstring; `docs/changelog.md` (new, consolidated).
- Fixed: `docs/validation.md` cited Ohyama et al. 2015 with the DOI of an unrelated paper (10.1038/nature14424); the right one is 10.1038/nature14297.

### Day 4 (network science, sensitivity analysis, sleep deprivation, fly tournament, fly racing)
Added. Nothing the simulation does changed: `--validate` was diffed against `release/3.0` (2,027 values, identical). The only edits near
validation are optional arguments (`params`, `only`) that default to the old behavior.
- **Fly tournament** (Esc > Fly arcade, `--tournament N`): brackets of 4, 8 or 16, brain-vs-brain duels (`game/flyduel.py`), measured
  personality cards, a favorite, match replays, the champion's drivers, and a pre-registered personality-predicts-winning analysis.
- **Fly racing** (Esc > Fly arcade, `--race`): lures, brain-driven speed, odds from the cards, **points only** (`core/points.py`), replays and a
  race assay (repeatability, form, odds calibration).
- **Network science** (Lab, `--netsci`): degrees, reciprocity, motifs vs a degree-preserving null, rich club, communities, regions, cached with
  a checksum, CSV export, background job with a progress bar. Adult and larva.
- **Sleep deprivation assay** (Lab, `--sleep-deprivation`, protocol assay kind `sleep_deprivation`): paired deprivation vs control; GAME RULE
  pressure, CONNECTOME dFB readout, MODEL PREDICTION rebound.
- **Sensitivity analysis** (Lab, `--sensitivity`): LIF parameters and the synapse threshold across a documented range against every validated
  behavior with validation's own criteria; heatmap, CSV/JSON/SVG, resumable, worker processes. Analysis only.
- Python API (`network_science`, `tournament`, `race`, `sleep_deprivation`, `sensitivity`, `fly.card`, `fly.duel`), `--selftest` day4 check,
  seven playthrough checks, five Lab > Model assumptions cards, the main docstring, 104 localization strings, five docs pages.
- `tools/run_tests.py`: the test suite in balanced chunks, a few at a time, with a memory guard and a per-file memory report
  (`KTF_MEM_REPORT`); `tests/.durations.json` holds the timings it balances by.

### Day 4 review (Opus)
- The game shows only **measured** personality cards (`core/cards.py`; "card not measured" until Esc > Fly arcade > Measure the flies in play);
  the pet shows its own fly's card; old pet cards are kept as `legacy_personality_card`.
- Race test **R4** (pre-registered): individuality with the brain state held fixed, `--headless --race-r4`; PASS. Why R1's control was
  repeatable: the per-seed calm baseline (docs/racing.md). `individuality_seed` (optional) in `new_brain`/`LIFParams`.
- Network science analysis version 2: the rich club's own undirected null; undefined enrichments shown as such; networkx cross-check tool.
- Fixes: Lab pages took each other's job results; Cancel waited for queued work; validation recorded default params and silently ignored
  them for four tests; wallet crashes and negative balances; the test suite's leaked games (`Game.shutdown`, conftest); arcade layout at
  larger text. T-maze sensitivity grid completed. `--validate` identical to release/3.0.

### Day 3 (predators, weather, kitchen, microphone, Streamer mode) and its review
Added, with the review's fixes folded in. Validation unchanged (diffed after each step; only three off-by-default Lab-parameter keys were added to the
recorded metadata).
- **Predators:** frog, dragonfly and mantis as tools in the room and the 2D game, seen only through the real looming neurons (LPLC2/LC4 -> DNp01);
  `predator_escape` assay (frog 0/30, mantis 0/30, dragonfly 0/30 after the review fixed a capture-frame off-by-one that had counted 2/30).
- **Rain, gusts and storms** (`weather.rain`, `weather.gust_hz`, `weather.storm`, off by default): touch, humidity, wind and light neurons.
- **Kitchen arena** (E, `--arena kitchen`): bowl, vinegar trap, sink, burner and a cook whose swatter is a looming object.
- **Microphone -> JO-A/B** (opt-in, off at every launch, red MIC ON pill, nothing recorded or sent) and `hum_demo` assay; **Streamer mode** (opt-in, off
  at every launch and in Lab mode, anonymous read-only Twitch chat, `!tool` on, `!arena` and `!surgery` off by default, every connection shown).
- Review: a hidden second Twitch connection after toggling; a click pulled a fly out of a frog's mouth; the neuPrint skeleton fetch ignored the network
  switch (now opt-in: Settings > Brain, one question after the tutorial); the test suite leaked a game per `LiveInputs` (20 GB); the model disagrees with
  Zhou et al. 2015 on pC1 tuning (stated); frog, mantis and dragonfly redrawn.

### Day 2 decisions and review
- A calm, untouched fly discovers nothing in the Neurodex (the rule is unchanged; discoveries now need a touch or a drive); the imaging F0 time constant
  (`brain.imaging_f0_tau_s`) and 5 Hz are settings; GCaMP6 kernels are from Chen et al. 2013 Supplementary Table 3 (GCaMP6f half-decay 142 ms, not 140).
- Review: Imaging mode never turned on on the real pack (a race); the inspector's PATCH button was off its card; the thermo page showed the idle slider;
  conftest forces the dummy SDL drivers (a Wayland shell opened real windows).

### Day 1 review
- A calm fly "discovered" ~190 types a minute: the rule gained a Poisson test whose alpha comes from a stated budget; curated facts matched against the
  real pack (EPG, JO-C/E prefixes); a share code could write outside the exports folder and could make the protocol checker build a billion seeds;
  double key bindings around D.

### Day 2 (genetic toolkit, thermogenetics, patch clamp, imaging, pharmacology)
Added, all in the Lab and all tagged on screen. Existing validation results are unchanged (diffed); the only edits near the simulation
are inert hooks on `Brain` (named extra currents, a per-step probe) and optional fields on `sim.wiring.Wiring`.
- **Genetic toolkit:** choose neurons by split-GAL4 line (`line:SS00727`); `kickthefly/data/driver_lines.yaml` (2,667 lines, Meissner
  et al. 2025, CC BY 4.0) built by `tools/build_driver_lines.py`; off-target note from the source's quality score. No GAL4 (non-split) table.
- **Thermogenetics:** TrpA1 and shibire-ts by cell type or line; temperature from the thermo arena, a Lab slider or a protocol; assay
  `thermo_escape` (DNp01 escape rate vs temperature).
- **Virtual patch clamp** (Lab > Patch clamp, inspector PATCH): potential, spikes, current steps, I-F curve, CSV/NWB. MODEL.
- **Simulated calcium imaging** (Lab > Calcium imaging): Imaging mode in the brain view, GCaMP6s/6f/8m, ROI traces, CSV/NWB/TIFF. MODEL.
- **Pharmacology** (Lab > Pharmacology): picrotoxin (moved from Robustness), cholinergic block, glutamate-Cl block, GABA-A agonist,
  dose slider, synapses affected per confidence level, include/exclude low-confidence predictions. MODEL PREDICTION.
- Protocols: `thermogenetics:`, `drug:`, `imaging:` blocks and a `patch:` protocol kind; seven example protocols. Python API: `fly.line`,
  `express`, `temperature`, `patch`, `image`, `drug`, `washout`. Settings > Brain: imaging indicator and frame rate. `--selftest` checks the
  toolkit. Playthrough: six new checks. Lab > Model assumptions: five new cards. docs/genetics.md, thermogenetics.md, patchclamp.md,
  imaging.md, pharmacology.md.

Day 1 of the 3.0 build. Nothing the simulation does changed: `kickthefly/lab/validation.py`, `assays.py` and `kickthefly/sim/` are
untouched, and no rule or threshold was tuned.

### Added
- **Neurodex** (D, Esc > Neurodex, d-pad up): a collectible encyclopedia of cell types. A type is discovered the first time its
  neurons fire well above their calm rate while you play (GAME RULE). Entries show the dataset's numbers (CONNECTOME) and, for about
  30 types, a one-line fact with its checked citation (LITERATURE). Progress per region, saved next to the training memory; the larva
  has its own list. docs/neurodex.md.
- **Neuron of the Day:** a launch card with one curated type, its fact and a Try it button (Lab laser or brain surgery). Own setting,
  default on, own off switch; separate from the real-science cards.
- **Kill cam** (;, d-pad down): slow-motion replay of the last ~6 s of the brain on death, with the neurons whose firing rose most
  highlighted; skippable, saved with the existing recorder; Reduced flashing slows it and steadies the highlight. docs/killcam.md.
- **Experiment bundles:** Lab > Record and export > Bundle, `--bundle ZIP` with `--protocol`, and `--headless --rerun-bundle ZIP --out DIR`
  (bit-exact on CPU backends, statistical on GPU; an RO-Crate 1.1 description with a SHA-256 for every file). docs/bundles.md.
- **Share codes:** Esc > Share makes and imports `KTF1-...` codes for surgeries, loadouts, protocols, challenge setups and Lab parameters,
  with a preview before applying and a file fallback for codes over 1,200 characters. `--share-decode CODE`. docs/share-codes.md.
- Settings > Brain: Neurodex discoveries, Kill cam offer, Neuron of the day (all default on). Keys: Neurodex (D), Kill cam (;).
  Gamepad: Neurodex (d-pad up), Kill cam / skip (d-pad down). Ctrl+V pastes into the menu's text boxes.
- Python API: `fly.collect()`, `fly.neurodex(type)`, `fly.killcam()`, `fly.kill()`, `fly.killcam_replay()`.
- The self-test checks the Neurodex facts; the playthrough bot checks discovery, the kill cam, share codes and a bundle rerun.
- Lab > Model assumptions: five new cards tagging each new behavior.

### Changed
- Per-fly protocol metadata now also records the backend and precision that ran and a SHA-256 of the recorded spikes (extra keys only).
- The pause menu has Neurodex and Share entries (slightly smaller buttons).
- D is both walk-right and the Neurodex key in 3D: the Neurodex opens on D only while the mouse is free. Configs that already use D
  for something else keep it and start with the Neurodex unbound.

## 2.13.1 (2026-09-30)

2.13.0 was tagged, but its release run stopped before publishing any files, so 2.13.0 ships as 2.13.1. Nothing in the
game or the simulation changed.

### Fixed
- The self-test's unwritable-folder test failed on Windows: it made its folder read-only with chmod, which does nothing
  to a folder there. It now puts a file where the folders go, which no OS can create a folder under, so it also runs
  as root where it used to be skipped.

## 2.13.0 (2026-09-29)

### Added
- **Tool loadouts.** The hotbar is a loadout of up to 10 tools on keys 1-9 and 0 (the mouse wheel and gamepad bumpers step through it;
  - and = turn the page of a longer one). Presets: Base (Play's default), Chaos, Chemist, Lab (Lab's default), All, Pet (Pet mode's
  default) and your own, with up to five saved. The hand is always in and can't be removed; the laser only in Lab mode; larva mode hides
  tools with no larval sensory mapping. The **loadout editor** (Q) lists every tool by category with its icon, a description and which
  real neurons it drives (CONNECTOME or GAME RULE), with click-and-drag equipping and reordering. The **tool wheel** (hold `) reaches
  every tool. New keys are rebindable (hotbar slots, page keys, editor, wheel); only Esc is fixed. Saved in `config.toml`, recorded as
  `tool` events in replays, and in the Python API (`fly.loadout`, `fly.set_loadout`, `fly.use_tool`). docs/loadouts.md.
- **Fruit tool:** drop ripe fruit; it is eaten exactly as sugar is (same taste and PAM neurons, same rules). In the Pet preset.
- **`--selftest`** and Settings > Help > Self-test: build info, brain pack checksums, every compute backend (CPU backends compared spike for
  spike, GPU backends statistically, GPU vendor, renderer and GL version reported), OpenGL 3.3, audio, ffmpeg, writable folders, disk and
  memory against the per-fly estimate, Wayland or X11, and a 10-second smoke protocol. PASS / WARN / FAIL with a plain-English fix; exit
  code 0, 1 or 3; JSON with `--out`. Never installs or configures anything.
- **Report a bug** (Settings > Help, the crash screen, `--bugreport`): shows exactly what it would include (version, OS, backend, self-test,
  the last 500 log lines, the crash report), lets you remove items, then copies it or opens a prefilled GitHub issue (long parts go to a file
  to attach). Nothing is uploaded; there is no telemetry.
- **First-launch tutorial** (about a minute, skippable at any step, replayable from Settings > Help, keyboard, mouse or gamepad, no flashing).
- **Playthrough bot:** `--headless --playthrough [adult|larva|all] --out DIR` uses every tool in every arena on each brain and checks: no
  exception, a sim/real floor, the documented neurons fire above baseline, ranges, death and autopsy, save-then-load, replay determinism on
  the CPU backends; plus multi-fly, surgery, training, duel, pet catch-up, individuality and every loadout preset. JSON and Markdown report.
  docs/playthrough.md.
- **CI:** a Claude (Sonnet) review of pull requests against `.github/claude-review.md` (skipped cleanly without the `ANTHROPIC_API_KEY`
  secret), a nightly workflow (full validation and playthrough, history table and trend charts on GitHub Pages, an issue when a validation
  result flips or the playthrough fails), and releases now need the self-test and the full playthrough. docs/ci.md.

### Changed
- `config.toml` schema 3: a config from before 2.13 migrates to the **All** preset (the order the number keys always had) with a one-time
  popup, and counts as already onboarded. A fresh install gets Base and the tutorial.
- The number keys, - and = are rebindable actions now (they were fixed); - and = turn hotbar pages instead of picking alcohol and the laser.
- The mouse wheel steps through the loadout in the 2D game as well; the gamepad's wheel lists every tool the mode allows.
- Settings gains a Help tab.

### Fixed (review of the 2.13 work, before release)
- An old config's own binding on Q, ` or pad X collided with the new actions' defaults and stayed bound twice; the old
  binding now wins and the new action takes a free key (4d06313).
- A control character in a saved loadout's name made `config.toml` unreadable on the next launch (b0b52c6).
- Each launch overwrote `kickthefly.log`, so Report a bug lost the crashed session's log (c8d86f4).
- Esc on the tool wheel also opened the pause menu (5fa9660).
- The Windows release self-test step failed on exit code 3 (warnings only), which every GPU-less runner gives
  (b19bcef).
- A digit rebound to the big brain view (a swap can put it there) opened the view but couldn't close it (9c673a8).
- The 2D spider now drops on its thread before it hunts, as the 3D spider does (0.04 m a frame down to 0.12 m above
  the floor; a game rule). It used to appear at the ceiling already crawling, so a fly that couldn't move toward it (stuck
  to flypaper, floating in the pool) barely saw it loom: the playthrough's one failure. Measured with the bot, 3 runs
  per indoor arena: 18/21 passed before, 21/21 after, LPLC2/LC4 now ~48 Hz as in 3D (21dc8f4).

### Known issues
- The default tool wheel key ` sits on a different physical key on non-US layouts (rebindable).
- Accented letters typed through dead keys or an IME don't reach loadout names.
- The German catalog doesn't have the 2.13 strings yet; they show in English.

## 2.12.0 (2026-09-28)

2.11.0 was already tagged, so the 2.11 work (Gemini's larval brain, fly individuality and pet mode, plus the review
fixes, PR #11) ships as 2.12.0.

### Added
- **Larval brain (headless only).** The first-instar *Drosophila* larva connectome (Winding et al. 2023: 2,952
  neurons, 110,677 connected pairs, 352,611 synapses) for `--headless --validate --brain larva` and Python use. It is
  built on your machine from the paper's Data S1 on first use (SHA-256 checked) and is not redistributed: no license
  is stated for it. Its transmitter signs are a GAME RULE guess. Two larva validation tests; neither passes. The
  windowed game (2D and 3D) always runs the adult brain and logs a warning when `brain.brain = "larva"`.
  See docs/larva.md.
- **Fly individuality.** Per-fly variation as per-neuron gains, W_fly = D_post·W·D_pre with the shared matrix
  unchanged and signs kept (Settings > Brain > Individuality: off / subtle σ 0.05 / strong σ 0.15, GAME RULE in
  Lab > Parameters). NumPy and Numba stay bit-exact; torch-cpu is bit-exact only with it off; gl does not implement
  it (logs that and runs with it off); torch-cuda/rocm untested. Always off in validation. Personality cards; measured
  pass rates in docs/individuality.md.
- **Pet mode** (`Esc > Mode`): one persistent adult fly across real days. No background process, service, autostart
  or timer: time is caught up deterministically at launch, clamped to 0-7 days. Hunger and sleep pressure (GAME RULE)
  scale real taste, PAM reward and dFB sleep neurons. Death is off by default. Saves are written atomically with a
  `.bak`, never deleted; unreadable ones are kept as `*.unreadable-<date>`. See docs/pet.md.

### Fixed (review of the 2.11 work, before release)
- Startup and menu crashes: the splash loop exited before the brain loaded, kick3d lost its splash font, the pause
  menu lost 'lab', Individuality did `float(None)`; pet mode poked a missing site, saved every eating frame and never
  saved on quit (d9c4a5a).
- Larva validation drove 0 neurons (selection used the broad type, not the annotations); `--validate --brain larva`
  now runs the larva and writes `validation_results_larva.json`; 3 invalid protocol files removed (5afbfde).
- Individuality: `new_brain` forced σ 0.15 for every setting, so subtle and strong were the same brain; gl no longer
  reports gains it does not use (924c0d2).
- Pet saves are never overwritten or deleted (9e318ea).
- Docs match what was measured (94f6a18, 50d0115).
- CI: torch imported only in the GPU test (31ecd1a); 2.11 strings in the translation template and the pause menu reads
  `cfg.pet` (ed40fe9); the arena setting treats a host without `is_larva` as adult (a1cdc65); the individuality
  validation test builds the larva pack it uses (6bd28f5).

### Known issues
- The windowed larva game is not playable (the larva body lacks the fly interface); it falls back to the adult brain.
- Larva transmitter signs are guessed (LN/MBON inhibitory, the rest excitatory). The "class IV md nociceptors" and
  "chordotonal" groups are ascending neurons, the "Basin" readout is 2nd-order PNs, and "_telegoro-1" is not verified
  Goro.
- Personality cards show their thresholds, but in-game metrics are seeded random draws, not measurements.
- Pet live sleep entry uses `random.random()`, so it is not deterministic (the catch-up is).
- Individuality defaults to "subtle", so existing players' flies change on upgrade unless they set it to off.
- The ICC/consistency results in docs/individuality.md were run at σ 0.15 and not re-run.
- Larva fly caps in `get_max_flies()` (64 NumPy / 128 Numba/GPU) rest on a larva benchmark that did not reproduce;
  they are unreachable while the larva game falls back to the adult.
- `test_batched_gpu_plastic_weights_with_individuality` uses backend `"torch-gpu"`, which is not a backend name, so
  on a CUDA/ROCm machine it runs NumPy and never tests the GPU.
- `CHANGES_GEMINI_2.11.md` is kept as Gemini wrote it and contains claims that did not hold (larva validation x1.00 /
  x1.06, a CC BY-NC-SA license for the larva data, σ 0.30 for strong, torch-cpu bit-exact with individuality, the
  larva benchmark and a <0.4% individuality cost, protocol files the parser rejects).

## 2.11.0 (2026-09-28)

### Fixed
- **The decoy female was invisible in the 3D game** (the default): she was dropped and touched but never drawn. She
  is now drawn as a female fly (rounder banded abdomen, no sex combs, slightly tinted) in every arena and in photo
  mode; the 2D decoy is redrawn as a female fly too.

### Added
- Decoys: the hand picks one up and throws it, right-click with the hand removes it; reset (R) and changing arena
  clear them; save states keep them; a 4th drop says "MAX 3 DECOYS".
- While a fly touches a decoy, a HUD line shows the live firing of the foreleg taste neurons she drives (LgLG5-8,
  putative ppk23/ppk25) and of P1 (REAL). COURTSHIP stays a game rule.
- The cVA puff shows a short-lived cloud as far as it reaches (250 px, 2.5 m in 3D; game rule).
- README screenshot of the decoy (docs/decoy.png).

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
