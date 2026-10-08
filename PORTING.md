# PORTING.md: Python → Rust + Bevy parity checklist

Source of truth for the `bevy-port` branch. Every feature of the Python game (3.1.1, `kickthefly/`) is listed once, with the Python file
and function/class it lives in. Later steps tick items off. **Do not tick an item until the Rust version reproduces the Python behavior**
(for the simulator: the golden tests in `crates/fly_sim/tests/golden.rs`; for everything else: a test or a documented manual check).

Legend: `[x]` done and verified · `[~]` partly done (the note says what is missing) · `[ ]` not started.
Paths are relative to `legacy-python/kickthefly/` after the move (today: `kickthefly/`). `Tag` is the game's own honesty tag
(CONNECTOME / GAME RULE / MODEL PREDICTION, see `docs/connectome-and-game-rules.md`); a port must keep it on screen.

Where the Rust code lives:

| Crate | Role |
|---|---|
| `crates/fly_sim` | pure Rust, no Bevy: pack loader, LIF model, RNG, headless-capable, rayon-parallel step |
| `crates/game` | Bevy app: render thread + dedicated sim thread, snapshot exchange |
| `tests/golden/` | reference outputs written by `legacy-python/make_golden.py` |

## Status at the end of the "port foundation" step

Done: section 2 (simulator core), the golden harness, the benchmark, the thread skeleton. Everything else is **not started**; sections 3-14
are the work list for the following steps. See "What could not be reproduced exactly" at the end of the report for the simulator.

---

## 1. Command line, entry points and packaging

Entry: `kick_the_fly.py` shim → `kickthefly/__main__.py:main` → `game/kick_the_fly.py:main` (`parse_args` at line ~7639, `main` ~7776).

- [ ] Windowed game: 3D by default (`game/kick3d.py:run`), `--2d` the original (`game/kick_the_fly.py` `Game`). Falls back to 2D with an explanation if OpenGL 3.3 is missing.
- [ ] Flags: `--2d --arena --autopilot --brain {adult,larva} --backend --sim-backend --dtype --individuality --mirror-weights --flies --seed --fullscreen --pet --spectator --smoke --record-video --resume --verbose --strict`
- [ ] Headless flags (`lab/headless.py:main`): `--headless --validate --protocol --bundle --rerun-bundle --replay --record-replay --out --workers --seeds --trials --nwb --benchmark --no-render-bench --selftest --selftest-child --bugreport --share-decode`
- [ ] Headless analyses: `--threshold-sweep --thresholds --signflip-test --flip-share --flip-confidence --critical-path --failure-autopsy --autopsy --activation-screen --knockout-screen --types --min-neurons --max-types --top --candidate-batch --screen-controls --screen-batch --restart --sensitivity --sens-params --sens-tests --sens-values --audit-asymmetry --netsci --tournament --race --races --race-r4 --lanes --match-seconds --sleep-deprivation --rig --rig-assay --minipaper --playthrough --playthrough-quick`
- [ ] Windows: console attach for headless (`core/platform_env.py:attach_console`), DPI awareness, exe icon (`tools/make_icon.py`), `build_exe.ps1`.
- [ ] Linux: Wayland preferred with X11 fallback (`core/platform_env.py:choose_backend`), AppImage (`build_appimage.sh`), Flatpak manifest (`packaging/flatpak/`), `.desktop`/appdata (`packaging/linux/`).
- [ ] CI: `.github/workflows/{tests,checks,nightly,release,release-gate,claude-review}.yml` → cargo equivalents.
- [ ] Release artifacts: Windows `.exe`, Linux AppImage, `SHA256SUMS`.

## 2. Simulator core (`sim/`): `crates/fly_sim`

### 2.1 Data formats read
- [x] Brain pack `data/kick_brain.npz` (zip of `.npy`): `indptr,indices,data` (CSR `[post,pre]`, int32/int32/int16 signed counts), `inv` (1/total input synapses, f32), `type,superclass,instance,subclass,region,nt,nt_source` (unicode `<U`), `nt_conf,soma` (f32), `body_id` (i64), `dan,mbon` (i32), `dan_mbon` (i16 `[DAN,MBON]`), `brain_type`. Python: `sim/brainpack.py:load`. Rust: `fly_sim::pack`.
- [x] Rate-normalisation on load: `W[post,pre] = count * inv[post]` (f32). Python: `brainpack.load` (row `data *= repeat(inv, diff(indptr))`).
- [ ] Larva pack `data/kick_larva_brain.npz` (same format, 3,016 neurons): `sim/connectome/larva_loader.py:larva_groups`. (Loader works; the larva sensory/motor groups do not exist yet.)
- [ ] Building a pack from the MaleCNS download (`sim/connectome/loader.py:build_graph`, `sim/brainpack.py:build`): pyarrow/feather → Rust `arrow` crate or keep as a Python dev tool. Decision pending; the pack is the shipped format.
- [ ] `soma` positions, `region` labels (`brainpack.regions`), `neurotransmitters()` confidence logic (used by Lab sign-flip and the big brain view).
- [ ] Skeletons (`data/skeletons/*.swc`, `sim/morphology.py`, `sim/realshapes.py`): opt-in download, checksummed cache, SWC parser.

### 2.2 Neuron model (`sim/connectome/sim.py:LIFParams, LIFSim`, `backends.py:CPUBackend.step`)
- [x] Parameters: `dt_ms=5, tau_ms=20, v_thresh=1, v_reset=0, refractory_steps=2, syn_gain=6 (initial), ext_gain=4, bias=0.2, noise_std=0.05, target_rate_hz=5, gain_adapt=0.002, gain_bounds=(1.5,40)`.
- [x] Per-step order: `I = W·spikes_prev` → `drive = I*gain + bias + noise_bank[off..off+n] (+ ext_gain*sensory)` → `v = v*(1-leak) + drive` → refractory neurons clamped to `v_reset`, counter−1 → `spike = v >= v_thresh` → reset `v`, `refr = 2`.
- [x] Propagation sums in ascending presynaptic order in f32 (identical to scipy's CSC-gather and CSR-matvec paths), so the spikes are bit-identical.
- [x] Slow global gain controller: `gain *= exp(gain_adapt*clip((target_p - frac)/target_p, -1, 1))`, clipped to bounds, `target_p = rate*dt/1000`.
- [x] Noise bank: `standard_normal(n*16, f32) * noise_std` drawn once from `default_rng(seed)`, read at `rng.integers(0, size-n)` per step. Rust reimplements `SeedSequence` + PCG64 + the NumPy float32 ziggurat (`fly_sim::rng`).
- [x] Sparse (≤10 % active, column gather) and dense (CSR) propagation paths: Rust uses a row-blocked CSC for sparse and CSR for dense, rayon-parallel.
- [x] `dtype=float64` option (`KICK_THE_FLY_SIM_DTYPE`). *(f32 only in Rust; see report.)*
- [ ] Individuality `D_post·W·D_pre` (`core/individuality.py:compute_fly_gains`: lognormal, σ 0.05/0.15, uses NumPy `normal` f64 ziggurat).
- [x] `ActivityBuffer` (EMA rate τ=40 steps, 400-step raster, last-spike step): `sim.py:ActivityBuffer`.
- [ ] Backends other than the reference: Numba/torch/GL are replaced by rayon + (later) a wgpu/compute path. `backends.py:{detect_available_backends,auto_chain,create_backend,GLBackend}`.
- [ ] Max flies per backend (`kick_the_fly.py:get_max_flies`), multi-fly batching, brain in its own process (`core/brainproc.py`; the Rust design uses a thread, see section 14).

### 2.3 Brain wrapper (`game/kick_the_fly.py:Brain`, `core/simcore.py`)
- [~] `rows_of` neuron specs: `type:`, `prefix:`, `superclass:`, `rows:` done in Rust; group names (`loom`, `escape`, …) and `line:` driver lines not yet.
- [x] Drive currents added to sensory input (`drive_cur`, `STIM_AMP=0.5`, sedation/inhibition `-0.4*ramp`), `new_brain` warm-up (600 steps), lockstep `step`.
- [ ] Detail groups `SENSE`/`MOTOR`/`POPS` tables (touch sites, 10 reaction DN groups, 9 populations), `poke()` with recruit share, pending-stimulus queue (8+40s steps).
- [ ] Brain surgery overrides (`SURGERY_CURRENT -0.6/0/+0.12`), named currents (`set_current`), probe hook, stethoscope counters.
- [ ] Rate trackers: `fast` EMA (150 ms), calm `base` EMA (10 s, only 2 s after the last poke), 1500-sample history ring, `level()/hz()`.
- [ ] Pain index (`pain_parts`, `hold_alarm`, `pain_index`, `PAIN_WEIGHTS`), pain levels normal/more/max, `kill()`/`revive()` (300-step death ramp).
- [ ] Brain thread loop with speed multiplier/pause/single-step (`Brain._loop`) → Rust sim thread (skeleton done in `crates/game`).
- [ ] Mirror-weights (`simcore.symmetrize_weights`), Lab `apply_to_sim`, wiring edits (`sim/wiring.py`: threshold drop, sign flip, drugs).
- [ ] Mushroom-body learning (`core/memory.py:Memory`): dopamine-gated KC→MBON plasticity, `fly-memory.npz` + `training-log.json`, v2.5.0 fixtures `tests/fixtures/fly-memory-v2.5.0*`.

## 3. Game modes and flow

- [ ] Play / Lab split (`Esc > Mode`; `core/config.py` `brain.mode`), Pet mode (`core/pet.py:PetManager`, `pet.ktfsave`+`.bak`, sleep pressure, real-stakes option), Spectator/autopilot (`Y`, `--autopilot`), tutorial (`ui/tutorial.py`), What's New screen (`ui/whatsnew_ui.py`), first-run notices (`Game.show_first_run_notices`).
- [ ] Time: pause / slow-mo 0.1×–1× / single step (`core/simclock.py:SimClock`, `Game.time_action`).
- [ ] Larva brain + crawler body (`game/larva.py`, `sim/connectome/larva_loader.py`; arenas restricted by `is_arena_allowed_for_larva`).
- [ ] Multiple flies (up to N), focus cycling (`F`), spawn (`N`), reset (`R`), recall lost flies (`J`).

## 4. The 2D and 3D games

### 4.1 Shared (`game/kick_the_fly.py`)
- [ ] Fly body: ragdoll particles, flight, `Fly` class (l.1503), `FlySlot` (l.1783), poses, health, damage, immortal option, death/shatter/melt/freeze visuals, `_die`, `_autopsy` (brain autopsy: every region's last 2 s vs calm).
- [ ] Reactions read from DNs: jump, run, kick, walk, back up, turn, fly, dodge (giant fiber), shoot (`MOTOR`); looming input from object growth (`Game._threats/_vision`); scents (`_scents`); learning hooks (`_learn`, `_memory_behavior`).
- [ ] HUD: pain meter, reward meter, hit counter, toolbar/hotbar, hotkey help (`H`), science cards with source chips (`draw_science_card`, `reaction_source`), tag chips.
- [ ] Brain panel + big brain view (`BrainView` l.1189, `B`): soma-position front view, fiber estimates, region colors, hover/inspect, neuron search and top-path tracing (`neurosearch.py`), panel modes solid/see-through/faint/hidden (`V`), GPU variant (`game/gpu_brainview.py`).
- [ ] Neuron inspector (`_inspect_at`, `_neuron_info`, `flip_neuron`, `transmitter_of`), real neuron shapes (`_real_shape`, `game/shape_draw.py`).
- [ ] Brain surgery UI (`O`; `_apply_surgery`, `_draw_surgery`, `silence_hemifield`), training screen (`T`; `start_training`, `_training_tick`, `_draw_training`).
- [ ] Captures: screenshot (F12), GIF (G), time-lapse (L; `capture_timelapse_frame`), video (Shift+R; `game/video_recorder.py`, ffmpeg or GIF), photo mode (F10; DoF), clean capture.
- [ ] Save states (`core/savestate.py`, `.ktfsave`, `Game.save_state/load_state`), replays (`core/replay.py`, `.ktfreplay`), speedrun codes (`game/speedrun.py`).
- [ ] Sounds (`Sound` l.2275: buzz following DNg02, hits, tools, splashes, menu), stethoscope (K; `update_stethoscope_target`, regions mushroom_body…whole_brain), sonification (F5; `core/sonify.py`, `game/sonify_play.py`).
- [ ] Profiler overlay (F3; `core/profiler.py:Profiler`, `--benchmark` frame table `lab/framebench.py`), optional FPS counter (`graphics.show_fps`).

### 4.2 3D (`game/kick3d.py`, `game/render3d.py`)
- [ ] First-person room, `Player`, `FreeCamera`, collisions (`push_out_boxes`), viewmodel (`draw_viewmodel`), mouse look, FOV, sprint/crouch.
- [ ] Instanced renderer (`render3d.py:Renderer`: sphere/cylinder/cube/disk/torus meshes, procedural patterns, glow, shadows, `perspective/look_at`), render-scale, vsync, fps cap, UI scale, menu size.
- [ ] 3D fly (`Fly3D`), tools in 3D (`use_tool3d`, `_swat3d`, `_explode3d`, `_zap3d`, `_jet`, `_spider3d`, `_sugar3d`, `_alcohol3d`, `_decoy3d`, `_kick`).
- [ ] 1v1 duel (X; `toggle_duel`, `_duel_senses`, `_duel_motor`, `_fly_shoot`, `_pellets3d`, `_player_hit`).
- [ ] Gamepad (`game/gamepad.py`: SDL controller, stick/trigger/bumper bindings, `capture_binding`) → `gilrs`/Bevy gamepad.
- [ ] Hotkeys: all `config.ACTIONS` are rebindable (walk WASD, sprint, crouch, free mouse Tab, B, O, T, X, E, P, I, M, F12, G, V, U, F11, N, R, H, Z, `[`, `]`, `.`, Y, F10, K, F3, F4, F5, L, J, F, Q, D, `;`, tool wheel `` ` ``, hotbar 1-0, `-`/`=`).

## 5. Tools (18; `kick_the_fly.py:TOOLS, TOOL_NAMES`, `core/loadout.py`)
- [ ] hand (drag & throw), flick, swatter, bomb (1.5 s fuse), torch (heat + max pain), cleaner (dissolve), zapper (all touch neurons + random 30 % for 40 ms), freeze (cold cells, frozen solid, shatter), spider, sugar, alcohol (PAM reward + inebriation), laser (`lab/laser.py`), cva, decoy (`_decoy`, decoy female, courtship), fruit, frog, dragonfly, mantis.
- [ ] Loadouts: hotbar slots/pages, presets, editor (Q), tool wheel, saved loadouts (`core/loadout.py`, `ui/loadout_ui.py`), migrated-config notice.
- [ ] Per-tool scent glomeruli (`Brain.__init__` TOOL_NAMES → 6 of 53, alcohol DM1/DM2/DP1m, cva DA1, decoy DA1+VA1v, laser none) using `default_rng(11)` permutation of glomeruli.

## 6. Arenas (10; `kick_the_fly.py:ARENAS`)
- [ ] room, fan (JO-C/E wind), flypaper (glue), pool (HRN humidity, drowning), lamp (photoreceptors, phototaxis, hot bulb), thermo (15–35 °C gradient; `thermo_gradient`), escape room (`Game.reset_escaperoom`, speedrun codes), kitchen (3D only; `game/kitchen.py`: bowl, vinegar trap, sink, stove, cook), field and orchard (3D only; `game/outdoors.py`: `WorldSpec`, wind, sun, `Orchard`, `Fruit`, trees, LOST flies).
- [ ] Weather (`game/weather.py:Weather`): rain, gusts, storms; day/night cycle (`outdoors.diurnal_cycle`, circadian clock neurons l-LNv/s-LNv, dFB sleep state).
- [ ] Predators (`game/predators.py` engine: frog tongue, dragonfly, mantis creep + strike; `game/predator_play.py`, `game/predator_anim.py` IK spider legs; Lab assay `lab/predators.py`).
- [ ] Streamer mode (`core/streamer.py`: Twitch IRC chat voting, `netguard`, `game/live_inputs.py`), microphone → Johnston's organ (`core/mic.py`: FFT bands → JO-A/JO-B drive; `lab/audio.py` hum demo).

## 7. Play-mode extras

- [ ] Challenges (`lab/challenges.py:Challenge` + `TMaze`, `Sneak`, `Sweet`, `ReverseSurgery`, `PredictNeuron`; `scores.json`; stars), Puppeteer (10 levels, no hands; `game/puppeteer.py`, `lab/puppet_challenge.py`), Contraption builder (`game/contraption.py`, `lab/contraption_challenge.py`, `contraptions.json`).
- [ ] Neurodex (`core/neurodex.py`, `ui/neurodex_ui.py`, `data/neurodex_facts.yaml`, `neurodex.json`), Neuron of the Day (`core/neuron_of_day.py`), kill cam (`core/killcam.py`, `game/extras3.py:Extras`), personality cards (`core/cards.py`, `core/individuality.py:compute_personality_card`), individuality modes off/subtle/strong.
- [ ] Fly arcade (`ui/arcade_ui.py`): tournaments (`lab/tournament.py`, `game/flyduel.py`), racing + odds + bets (`lab/racing.py`, `game/flyrace.py`, `core/points.py:Wallet`, points only), lesion battles (`game/lesion_battle.py`, `ui/battle_ui.py`, `lesion_loadouts.json`).
- [ ] Fly's-eye view (F4; `game/fly_eye.py`, visual only), mini-papers (`lab/minipapers.py`, `ui/minipaper_ui.py`, six papers), classroom/lectures (`lab/classroom.py`, `lab/labclassroom.py`, `protocols/lecture_*.yaml`).
- [ ] Share codes (`core/sharecode.py`, `ui/share_ui.py`): `KTF1-<KIND>-…` Crockford base32, kinds SRG, LDT, PRT, CHL, LAB, CTN; `.ktfshare` files; clipboard (`core/clipboard.py`).

## 8. Lab mode (`lab/`)

### 8.1 Hub pages (`lab/lab.py:install`, `menu.pages[...]`)
- [ ] `lab` hub, `lab_params` (MODEL vs GAME RULE parameters, `apply_to_sim`), `lab_assays`, `lab_validation`, `lab_export`, `lab_protocols`, `lab_assumptions`, `lab_asymmetry`, `lab_benchmark`.
- [ ] `lab_wiring` (synapse threshold drop, sign flip; `lab/labwiring.py`, `sim/wiring.py`, `lab/robustness.py`), `lab_critical` (`criticalpath.py`), `lab_clamp` (`clamp.py`, neural clamp), `lab_diff` (`diffmode.py`), `lab_laser` (`laser.py`), `lab_psych` (`psychometrics.py`, SVG/PDF/CSV), `lab_classroom`.
- [ ] 3.0 toolkit (`lab/labtoolkit.py`): genetic toolkit (`genetics.py`, `data/driver_lines.yaml`, `line:` specs), thermogenetics (`thermogenetics.py`, TrpA1/shibire-ts), patch clamp (`patchclamp.py`, I-F curve), calcium imaging (`imaging.py`, GCaMP kernels, TIFF/CSV/NWB), pharmacology (`pharmacology.py`, drugs as scaling by predicted NT).
- [ ] 3.0 day 4: network science (`netsci.py`), sleep deprivation (`sleepdep.py`), sensitivity analysis (`sensitivity.py`) via `labday4.py`; behavior rigs (`rigs.py`, `rigassay.py`, `labrigs.py`: tethered, ball, Buridan, four-field).
- [ ] 3.1: activation screen + knockout screen (`screens.py`, `labscreen.py`), failure autopsy (`failure_autopsy.py`, `labautopsy.py`), mini-papers page.
- [ ] Recording/export: `recorder.py` (CSV/npz + metadata JSON), `nwbexport.py` (needs an NWB/HDF5 writer), bundles (`bundle.py`: zip with protocol, results, SHA-256s, rerun), `neurosearch.py`.

### 8.2 Validation (`lab/validation.py:TESTS`; 21 tests, expected **12 PASS / 9 FAIL**, held-out seeds 1000-1009, one-sided Wilcoxon p<0.01, ≥1.5× calm)
- [ ] looming_escape, mdn_backward, sugar_feeding, antenna_grooming_circuit, adn_grooming_motor, mb_conditioning, epg_compass, epg_compass_wind, courtship_song, bitter_grn_to_dng28, co2_orn_to_pn, hot_trn_to_vp2pn, cold_trn_to_vp3pn, grooming_hierarchy, optomotor_turning, p1_courtship_song, or67d_to_da1pn, da1pn_to_lh_asp, foreleg_grn_to_p1, mb_extinction, mb_second_order (+ 2 larva tests: larva_noci_to_goro_rolling, larva_chordotonal_to_basin).
- [ ] Assays (`lab/assays.py`): T-maze, looming, sugar, extinction, second order, plume tracking, orchard; statistics (`labstats.py`), Lab jobs in worker processes (`labjobs.py`).
- [ ] Headless runs, protocol YAML (`lab/protocol.py`; 43 files in `protocols/`), playthrough bot (`lab/playthrough.py`), benchmark (`lab/benchmark.py`), Python API (`lab/api.py:Fly`) → Rust crate API + optional PyO3 later.

## 9. Menus, settings and screens (`ui/`, `core/config.py`)

- [ ] Immediate-mode menu framework (`ui/menu.py:Menu`: pages, buttons, sliders, charts, tooltips, gamepad nav) → Bevy UI or egui (decision pending; pin the version if a plugin is used).
- [ ] Pause menu + Settings with six tabs (Graphics 17, Audio 11, Brain 35, Controls 8, Accessibility 4 settings = 75 in `config.SETTINGS`); `config.toml`, schema version 3, `first_run` flags, clamping, unknown-key tolerance, `.bad` backup.
  - Graphics: fullscreen, resolution_scale, fps_cap, show_fps, vsync, backend, brain_view, panel_mode, menu_size, ui_scale, photo_scale, clean_capture, photo_dof, photo_focus, timelapse_speedup/format/target.
  - Audio: master, buzz, sfx, mute, stethoscope_{enabled,vol,target}, sonify, sonify_vol, sonify_song, stream.allow_sonify.
  - Brain: mode, brain, individuality, pet_real_stakes, pain_level, immortal, sim_speed, seed, real_vs_rule, science_popups, neurodex, killcam, neuron_of_day, neuron_shapes, mic_sensitivity, stream.{allow_tool,allow_arena,allow_surgery,window_s,cooldown_s,min_votes}, imaging_{indicator,fps,f0_tau_s}, autopilot{,_orbit,_hide_hud}, arena, song_buzz, lunge, day_night, mirror_weights, dtype, backend, process.
  - Controls: mouse_sensitivity, invert_y, fov, loadout_preset, gamepad, pad_look_speed, pad_deadzone, pad_invert_y, plus the rebinding table (`ACTIONS`, `PAD_*`, `RESERVED_KEYS`).
  - Accessibility: palette, reduced_flashing, larger_text, language (en, de; `core/i18n.py`, `data/locales/*.json`, `tools/i18n_sync.py`).
- [ ] Menu pages: `challenges`, `arcade`, `neurodex`, `loadout`, `loadout_notice`, `share`, `load_state`, `live_inputs` (Mic and streamer), `neuron_shapes_ask`, `whatsnew`, `selftest`, `bugreport`, `crash`, `lab_*`, `lab_minipapers`, `lab_rigs`.
- [ ] Self-test (`core/selftest.py`, 22 checks), bug report with scrubbing (`core/bugreport.py`), crash reports (`core/crash.py`, `ui/crashscreen.py`), paths per OS (`core/paths.py`: XDG on Linux, Known Folders on Windows, `KICK_THE_FLY_HOME`), netguard (no telemetry), version (`core/version.py`).

## 10. Save / share / data formats (every one must keep loading)

| Format | Python | Notes |
|---|---|---|
| `kick_brain.npz`, `kick_larva_brain.npz` | `sim/brainpack.py` | **loaded by Rust** |
| `config.toml` (schema v3) | `core/config.py` | |
| `.ktfsave` (zip: npz arrays + JSON header, format `kick-the-fly-save` v1) | `core/savestate.py` | includes numpy RNG state: needs the Rust PCG64 state import/export (`fly_sim::rng` has the pieces) |
| `.ktfreplay` (zip, per-step spike SHA-256) | `core/replay.py` | |
| pet `pet.ktfsave` + `.bak` | `core/pet.py` | |
| memory `fly-memory.npz` + `training-log.json` | `core/memory.py` | fixtures in `tests/fixtures/` |
| `neurodex.json`, `scores.json`, `contraptions.json`, `lesion_loadouts.json` | `core/neurodex.py`, `lab/challenges.py`, `game/contraption.py`, `game/lesion_battle.py` | |
| share codes `KTF1-…`, `.ktfshare` | `core/sharecode.py` | |
| experiment bundles (zip) | `lab/bundle.py` | |
| recordings CSV/npz + metadata JSON; NWB; TIFF | `lab/recorder.py`, `nwbexport.py`, `imaging.py` | |
| protocols YAML (43), lecture YAML | `lab/protocol.py`, `protocols/` | |
| `data/driver_lines.yaml`, `data/neurodex_facts.yaml`, `data/locales/*.json` | `data/__init__.py` | |
| result CSV/JSON/SVG (validation, race, netsci, sensitivity, screens, autopsy) | `lab/*` | compare outputs with `docs/results/` |

## 11. Audio
- [ ] Sound effects and wing buzz (pygame mixer synthesised in `Sound`), stethoscope clicks, sonification synth (`core/sonify.py:Synth`), hum demo; Bevy audio or `rodio` via `bevy_audio`.

## 12. Tests to port (110 files in `tests/`)
- [x] Simulator golden tests (this step): `crates/fly_sim/tests/golden.rs` against `tests/golden/`.
- [ ] Everything else: `test_backends`, `test_determinism`, `test_validation`, `test_config`, `test_savestate`/replay/share/bundle, assay and Lab page tests, playthrough bot, etc. Port by area together with the feature.

## 13. Docs and release
- [ ] `README.md`, `docs/*.md` (47 files) updated for the Rust build; `CHANGELOG.md`; `RELEASE_NOTES.md`; screenshots tool (`tools/make_screenshots.py`).
- [ ] Move the Python tree to `legacy-python/` **unchanged** (done at the end of the port task, see git history).

## 14. Architecture decisions for the Rust build
- [x] Simulation on its own OS thread at a fixed 200 steps/s timestep (`crates/game/src/sim_thread.rs`); the render thread only reads the latest published snapshot (triple-buffered, never blocks the sim, never blocked by it) and sends commands over a bounded channel.
- [x] Dev builds use Bevy's `dynamic_linking` feature; release builds do not (`crates/game/Cargo.toml` feature `dev`).
- [x] Exact version pins in `Cargo.toml` (`bevy = "=0.19.1"`, no third-party Bevy plugins yet).
- [ ] UI toolkit, audio, gamepad and video-recording crates: choose and pin when their feature is ported.
