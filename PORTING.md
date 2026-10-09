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
| `crates/game` | Bevy app: render thread + dedicated sim thread, snapshot exchange; one plugin per feature (see `GamePlugin` in `lib.rs`): `arena`, `environment`, `fly`, `fly_visuals`, `tools`, `predator_play`, `weather_play`, `gpu`, `brainview`, `flyeye`, `killcam_play`, `arcade`, `contraption`, `audio`, `mic`, `profiler`, `hud`, `controls`; pure logic with tests: `anim`, `predators`, `weather`, `world`, `kitchen`, `reactions`, `duel`, `race`, `points`, `tournament`, `puppeteer`, `lesion`, `killcam`, `synth`, `mic_analysis`, `paths` |
| `tests/golden/` | reference outputs written by `legacy-python/make_golden.py` |

## Status at the end of the "persistence" step

Added `crates/ktf_data` (no Bevy, fast tests): `config` (config.toml v3), `sharecode` (KTF1 codes), `paths` (per-OS folders through the `directories` crate, same layout as
Python). The game reads `config.toml` at startup (`Settings::from_config`: master volume, mute, reduced flashing, mouse sensitivity, invert-Y, FOV, immortal, kill cam) and
keeps `scores.json`/`arcade_points.json` in the Python data folder. Real files written by `legacy-python/make_persistence_fixtures.py` are in `tests/python_files/`
(regenerate with that script; it also writes `crates/ktf_data/data/schema.json`, the 75-setting schema the Settings menu will be built from).
Second persistence step: `.ktfsave` container + simulator-state import/export, `fly-memory.npz`, `neurodex.json`, `contraptions.json`, `lesion_loadouts.json` (fixtures in `tests/python_files/{savestate,stores}`; recipes in `legacy-python/make_persistence_fixtures.py` and `make_savestate_fixture.py`). **Still to do**: `.ktfreplay` and experiment bundles (with the Lab headless runner), `pet.ktfsave`, applying a loaded save to the game's bodies (see `continue.md`).
README has the Rust build instructions (section 7 of the task).

## Status at the end of the "game port" step (GPU rendering, physics, predators, weather, audio, profiler)

Added: the whole `crates/game` feature set (modules below), the sim-side group readouts (`fly_sim::groups`, golden-tested against NumPy), `Generator::shuffle`
(NumPy's permutation stream), `Brain::{poke, poke_recruit, level, hz, revive}`, `side:` neuron specs, and the sim thread's `Poke`/`Surgery`/`Sedation`/`Revive`
commands and richer snapshots (group Hz and levels, layout, labels, publish time for the lag readout).

How to look at it: `cargo run --release -p kick_the_fly` (needs `data/kick_brain.npz`, else a synthetic net runs). `H` is the in-game key list. For a
screenshot without a person: `kick_the_fly --arena kitchen --cam 0.5,1.7,3.3,0,0.3,-1.5 --shot out.png --shot-frames 150` (see `--help`).

**Not verified here**: the Windows build, a real microphone with sound in it, audio output by ear, gamepads. **Not started**: everything in sections 1 (most flags), 8
(the Lab), 9 (menus and settings), 10 (formats other than `scores.json`/`arcade_points.json`), 13.


## Status at the end of the "port foundation" step

Done: the simulator core (sections 2.1-2.2), the golden harness, the benchmark, the thread skeleton. Not verified: `cargo run --features dev` (dynamic linking) was not built here; the Windows build was not attempted. Everything else is **not started**; sections 3-14
are the work list for the following steps. See "What could not be reproduced exactly" at the end of the report for the simulator.

---

## 1. Command line, entry points and packaging

Entry: `kick_the_fly.py` shim → `kickthefly/__main__.py:main` → `game/kick_the_fly.py:main` (`parse_args` at line ~7639, `main` ~7776).

- [~] Windowed game: 3D by default (`game/kick3d.py:run`), `--2d` the original (`game/kick_the_fly.py` `Game`). Falls back to 2D with an explanation if OpenGL 3.3 is missing. **Rust:** the 3D window runs (`kick_the_fly`, Bevy 0.19.1: Wayland+X11 features on; Windows untried). NOT done: `--2d`, the OpenGL-3.3 fallback.
- [~] Flags: `--2d --arena --autopilot --brain {adult,larva} --backend --sim-backend --dtype --individuality --mirror-weights --flies --seed --fullscreen --pet --spectator --smoke --record-video --resume --verbose --strict` **Rust:** `--pack --synthetic --seed --smoke` (foundation) plus `--arena NAME`, `--rain F --storm --day S`, and the test/CI harness `--cam --tool --fire --hold --kill-at --shot --shot-frames`. The rest of the Python flags are not ported.
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
- [~] `soma` positions, `region` labels (`brainpack.regions`), `neurotransmitters()` confidence logic (used by Lab sign-flip and the big brain view). **Rust:** `Labels.soma`, `region` and `type`/`superclass` are loaded and drive the GPU brain view layout (`sim_thread::layout`: real soma, else the synapse-weighted centre of placed partners, like `BrainView`) and the sonification's regions. `neurotransmitters()` confidence logic: not ported.
- [ ] Skeletons (`data/skeletons/*.swc`, `sim/morphology.py`, `sim/realshapes.py`): opt-in download, checksummed cache, SWC parser.

### 2.2 Neuron model (`sim/connectome/sim.py:LIFParams, LIFSim`, `backends.py:CPUBackend.step`)
- [x] Parameters: `dt_ms=5, tau_ms=20, v_thresh=1, v_reset=0, refractory_steps=2, syn_gain=6 (initial), ext_gain=4, bias=0.2, noise_std=0.05, target_rate_hz=5, gain_adapt=0.002, gain_bounds=(1.5,40)`.
- [x] Per-step order: `I = W·spikes_prev` → `drive = I*gain + bias + noise_bank[off..off+n] (+ ext_gain*sensory)` → `v = v*(1-leak) + drive` → refractory neurons clamped to `v_reset`, counter−1 → `spike = v >= v_thresh` → reset `v`, `refr = 2`.
- [x] Propagation sums in ascending presynaptic order in f32 (identical to scipy's CSC-gather and CSR-matvec paths), so the spikes are bit-identical.
- [x] Slow global gain controller: `gain *= exp(gain_adapt*clip((target_p - frac)/target_p, -1, 1))`, clipped to bounds, `target_p = rate*dt/1000`.
- [x] Noise bank: `standard_normal(n*16, f32) * noise_std` drawn once from `default_rng(seed)`, read at `rng.integers(0, size-n)` per step. Rust reimplements `SeedSequence` + PCG64 + the NumPy float32 ziggurat (`fly_sim::rng`).
- [x] Sparse (≤10 % active, column gather) and dense (CSR) propagation paths: Rust uses a row-blocked CSC for sparse and CSR for dense, rayon-parallel.
- [ ] `dtype=float64` option (`KICK_THE_FLY_SIM_DTYPE`, `Settings > Brain > dtype`): the Rust sim is f32 only.
- [~] Individuality `D_post·W·D_pre` (`core/individuality.py:compute_fly_gains`): `fly_sim::lif::Individuality`; the normals are bit-exact, the final f32 `exp` agrees to ~1e-6 (NumPy's SIMD exp varies by CPU), so runs with individuality on are statistically, not spike-for-spike, equal. No whole-run golden for it yet; personality cards (`compute_personality_card`) are not ported.
- [x] `ActivityBuffer` (EMA rate τ=40 steps, 400-step raster, last-spike step): `sim.py:ActivityBuffer`.
- [ ] Backends other than the reference: Numba/torch/GL are replaced by rayon + (later) a wgpu/compute path. `backends.py:{detect_available_backends,auto_chain,create_backend,GLBackend}`.
- [ ] Max flies per backend (`kick_the_fly.py:get_max_flies`), multi-fly batching, brain in its own process (`core/brainproc.py`; the Rust design uses a thread, see section 14).

### 2.3 Brain wrapper (`game/kick_the_fly.py:Brain`, `core/simcore.py`)
- [~] `rows_of` neuron specs: `type:`, `prefix:`, `superclass:`, `rows:` done in Rust; group names (`loom`, `escape`, …) and `line:` driver lines not yet. **Rust:** `type:`, `prefix:`, `superclass:`, `rows:` and the puppeteer's `side:L|R:types`. Group names are Rust-side (`fly_sim::groups`); `line:` driver lines and the larva groups are not ported.
- [x] Drive currents added to sensory input (`drive_cur`, `STIM_AMP=0.5`, sedation/inhibition `-0.4*ramp`), `new_brain` warm-up (600 steps), lockstep `step`.
- [~] Detail groups `SENSE`/`MOTOR`/`POPS` tables (touch sites, 10 reaction DN groups, 9 populations), `poke()` with recruit share, pending-stimulus queue (8+40s steps). **Rust:** `fly_sim::groups`: SENSE, MOTOR, POPS, the extra readouts, the per-tool scent deals (`default_rng(11)` permutation via `Generator::shuffle`) and the T-maze odors. **Verified against the Python NumPy code** (`tests/golden/make_groups.py` -> `groups.json`, test `groups_match_python`: sizes and row SHA-256 of every group, every scent). `poke` + the pending queue (8+40s steps) are unit-tested; which neurons a poke recruits differs from Python (different RNG), documented in the module. Not ported: assay groups `sweet`/`bitter`, larva groups.
- [~] Brain surgery overrides (`SURGERY_CURRENT -0.6/0/+0.12`), named currents (`set_current`), probe hook, stethoscope counters. **Rust:** `Command::Surgery{spec,mode}` / `Brain::set_surgery` (-0.6/0/+0.12) are used by lesion battles and tested on the real pack (`arcade_real.rs`: a silenced DNp09 does not fire). Named currents, the probe hook and stethoscope counters are not ported.
- [~] Rate trackers: `fast` EMA (150 ms), calm `base` EMA (10 s, only 2 s after the last poke), 1500-sample history ring, `level()/hz()`. **Rust:** `fast` EMA (150 ms) and calm `base` EMA with the 2 s-after-poke rule, `level()` (baseline floored at 2 Hz) and `hz()` are in `groups.rs` (unit-tested) and published in every snapshot. The 1500-sample history ring is not.
- [ ] Pain index (`pain_parts`, `hold_alarm`, `pain_index`, `PAIN_WEIGHTS`), pain levels normal/more/max, `kill()`/`revive()` (300-step death ramp).
- [x] Brain thread loop with speed multiplier/pause/single-step (`Brain._loop`) → Rust sim thread (skeleton done in `crates/game`). **Rust:** `sim_thread.rs`: fixed 200 steps/s, speed multiplier, pause, single step, kill/revive, drops a backlog over 40 steps like Python; tested in `sim_thread::tests` and `tests/headless_app.rs`.
- [ ] Mirror-weights (`simcore.symmetrize_weights`), Lab `apply_to_sim`, wiring edits (`sim/wiring.py`: threshold drop, sign flip, drugs).
- [ ] Mushroom-body learning (`core/memory.py:Memory`): dopamine-gated KC→MBON plasticity, `fly-memory.npz` + `training-log.json`, v2.5.0 fixtures `tests/fixtures/fly-memory-v2.5.0*`.

## 3. Game modes and flow

- [ ] Play / Lab split (`Esc > Mode`; `core/config.py` `brain.mode`), Pet mode (`core/pet.py:PetManager`, `pet.ktfsave`+`.bak`, sleep pressure, real-stakes option), Spectator/autopilot (`Y`, `--autopilot`), tutorial (`ui/tutorial.py`), What's New screen (`ui/whatsnew_ui.py`), first-run notices (`Game.show_first_run_notices`).
- [~] Time: pause / slow-mo 0.1×–1× / single step (`core/simclock.py:SimClock`, `Game.time_action`). **Rust:** Z (pause time), P (pause the brain), `[` `]` (0.1x-1x), `.` (single step) in `controls.rs`: they scale Bevy's virtual clock (physics, predators, weather) and the brain's speed together. Implemented, not covered by an automatic test.
- [ ] Larva brain + crawler body (`game/larva.py`, `sim/connectome/larva_loader.py`; arenas restricted by `is_arena_allowed_for_larva`).
- [~] Multiple flies (up to N), focus cycling (`F`), spawn (`N`), reset (`R`), recall lost flies (`J`). **Rust:** Reset (R) and arena switch re-create the fly. ONE fly and one brain only: no spawning more (N), no focus cycling (F); lost-fly detection outdoors is implemented but R resets rather than recalls (J).

## 4. The 2D and 3D games

### 4.1 Shared (`game/kick_the_fly.py`)
- [~] Fly body: ragdoll particles, flight, `Fly` class (l.1503), `FlySlot` (l.1783), poses, health, damage, immortal option, death/shatter/melt/freeze visuals, `_die`, `_autopsy` (brain autopsy: every region's last 2 s vs calm). **Rust:** `fly.rs`/`fly_visuals.rs`: a three-body avian ragdoll (head, thorax, abdomen, spherical joints) with a PD controller using the same `strength` as the Python pose pull; flight with hover target and wander; health, damage, immortal, char/soak/melt/frost/venom/dissolve/shatter, death. Legs are IK on a tripod gait, wings flap. **Verified headlessly** (`tests/fly_physics.rs`: stands at STAND3, upright, walks 2.3 px/frame, limp when stunned and recovers, held-by-thorax hangs the abdomen, released falls) and by screenshots. Not ported: the 17-particle ragdoll itself (legs/wings are visual only), `FlySlot` multi-fly, `_autopsy`.
- [~] Reactions read from DNs: jump, run, kick, walk, back up, turn, fly, dodge (giant fiber), shoot (`MOTOR`); looming input from object growth (`Game._threats/_vision`); scents (`_scents`); learning hooks (`_learn`, `_memory_behavior`). **Rust:** `reactions.rs` (THRESH table, hold times, cooldowns; 6 unit tests) runs on the snapshot's group levels; `senses.rs` turns every threat's angular expansion into LPLC2/LC4 pokes (`LOOM_MIN 1.5`, `LOOM_FULL 8`) and smells into ORN pokes. Seen live in the game log (WALK, TAKE OFF, GROOM, DODGE ...). Not ported: learning hooks (`_learn`, `_memory_behavior`), the duel's SHOOT in play.
- [~] HUD: pain meter, reward meter, hit counter, toolbar/hotbar, hotkey help (`H`), science cards with source chips (`draw_science_card`, `reaction_source`), tag chips. **Rust:** `hud.rs`: hotbar (18 tools, 10 per page), tool hint, the reaction log with CONNECTOME / GAME RULE tag chips, floating popups, health bar, crosshair, `H` help. Not ported: pain meter, reward meter, hit counter, science cards (pain index is not ported).
- [~] Brain panel + big brain view (`BrainView` l.1189, `B`): soma-position front view, fiber estimates, region colors, hover/inspect, neuron search and top-path tracing (`neurosearch.py`), panel modes solid/see-through/faint/hidden (`V`), GPU variant (`game/gpu_brainview.py`). **Rust:** `brainview.rs` + `shaders/brain.wgsl`: all ~140,000 neurons drawn in ONE draw call by vertex pulling from a storage buffer (static layout + a per-frame activity buffer written from the snapshot's EMA rates and spike list), a second camera renders into an image shown as a UI node; `B` big, `V` solid/see-through/faint/hidden, drag to turn. Verified by screenshots on an RTX 4060 at ~60 fps. Not ported: fibers, region colouring toggle, hover/inspect, neuron search and path tracing.
- [ ] Neuron inspector (`_inspect_at`, `_neuron_info`, `flip_neuron`, `transmitter_of`), real neuron shapes (`_real_shape`, `game/shape_draw.py`).
- [ ] Brain surgery UI (`O`; `_apply_surgery`, `_draw_surgery`, `silence_hemifield`), training screen (`T`; `start_training`, `_training_tick`, `_draw_training`).
- [~] Captures: screenshot (F12), GIF (G), time-lapse (L; `capture_timelapse_frame`), video (Shift+R; `game/video_recorder.py`, ffmpeg or GIF), photo mode (F10; DoF), clean capture. **Rust:** F12 saves a PNG (`controls.rs`) and the `--shot` harness uses the same path. GIF, time-lapse, video, photo mode: not ported.
- [ ] Save states (`core/savestate.py`, `.ktfsave`, `Game.save_state/load_state`), replays (`core/replay.py`, `.ktfreplay`), speedrun codes (`game/speedrun.py`).
- [~] Sounds (`Sound` l.2275: buzz following DNg02, hits, tools, splashes, menu), stethoscope (K; `update_stethoscope_target`, regions mushroom_body…whole_brain), sonification (F5; `core/sonify.py`, `game/sonify_play.py`). **Rust:** `synth.rs` ports every `Sound._make` effect (tested: present, bounded, not silent, valid WAV), `audio.rs` plays them as generated WAV assets, the wing buzz follows DNg02 while the fly flies, F5 starts the live sonification (voices per region on the D minor pentatonic, the song gate) as a streaming `Decodable`. Not ported: stethoscope clicks (K), the hum demo, `stream.allow_sonify`. Not verified by ear (no audio device in CI).
- [x] Profiler overlay (F3; `core/profiler.py:Profiler`, `--benchmark` frame table `lab/framebench.py`), optional FPS counter (`graphics.show_fps`). **Rust:** F3: FPS and frame time (smoothed, worst, 120-frame graph), sim steps/s against 200, steps dropped, sim-thread lag (age of the snapshot read), draw **batches** (distinct mesh+material among visible meshes; Bevy does not expose the driver's draw-call counter, so this is derived and says so), points drawn by instanced shaders, per-pass GPU time from `RenderDiagnosticsPlugin`. Verified by screenshot. `--benchmark` frame table: not ported.

### 4.2 3D (`game/kick3d.py`, `game/render3d.py`)
- [~] First-person room, `Player`, `FreeCamera`, collisions (`push_out_boxes`), viewmodel (`draw_viewmodel`), mouse look, FOV, sprint/crouch. **Rust:** `player.rs`: WASD walk/sprint/crouch, mouse look, FOV from settings, slide along furniture, kinematic body that shoves flies, Tab frees the mouse, `--cam` override for tests. Not ported: `FreeCamera`, the viewmodel (tool in hand).
- [~] Instanced renderer (`render3d.py:Renderer`: sphere/cylinder/cube/disk/torus meshes, procedural patterns, glow, shadows, `perspective/look_at`), render-scale, vsync, fps cap, UI scale, menu size. **Rust:** Bevy's PBR renderer plus three custom GPU materials in `gpu.rs` (vertex pulling, one draw per cloud): `PointsMaterial` (particles: fire, flame, spray, dust, sparks, ice, wind streaks), `BrainMaterial`, `RainMaterial` (6,000 streaks simulated on the GPU, no CPU state). Grass is one merged mesh, the books one merged mesh. Procedural textures for wood, wallpaper, rug, tiles, glue. Render scale, vsync, fps cap, UI scale: not ported.
- [~] 3D fly (`Fly3D`), tools in 3D (`use_tool3d`, `_swat3d`, `_explode3d`, `_zap3d`, `_jet`, `_spider3d`, `_sugar3d`, `_alcohol3d`, `_decoy3d`, `_kick`). **Rust:** All 18 tools are implemented in `tools.rs` (+ `predator_play.rs`): the hand pins a part of the ragdoll and throws on release, flick, swatter (lands 0.12 s after the swing), bomb (rigid body, 1.8 s fuse, falloff 48*(1-d/R)^1.3), torch/cleaner/freeze jets (cone, char/melt/frost), zapper bolt, spider (drop, hunt, bite, wrap, carry), sugar/fruit/alcohol treats (rigid bodies the fly walks to and eats, inebriation), laser (cell-type picker), cVA puff, decoy female, predators. Walking into a fly is a kick. Screenshot-verified: torch, bomb (KABOOM), frog. The rest are compile- and logic-checked only.
- [~] 1v1 duel (X; `toggle_duel`, `_duel_senses`, `_duel_motor`, `_fly_shoot`, `_pellets3d`, `_player_hit`). **Rust:** The fly-vs-fly duel engine is ported (`duel.rs`, 3 tests, and run on real brains in `arcade_real.rs`); the player-vs-fly duel (X, the fly shoots at YOU) is not.
- [ ] Gamepad (`game/gamepad.py`: SDL controller, stick/trigger/bumper bindings, `capture_binding`) → `gilrs`/Bevy gamepad.
- [ ] Hotkeys: all `config.ACTIONS` are rebindable (walk WASD, sprint, crouch, free mouse Tab, B, O, T, X, E, P, I, M, F12, G, V, U, F11, N, R, H, Z, `[`, `]`, `.`, Y, F10, K, F3, F4, F5, L, J, F, Q, D, `;`, tool wheel `` ` ``, hotbar 1-0, `-`/`=`).

## 5. Tools (18; `kick_the_fly.py:TOOLS, TOOL_NAMES`, `core/loadout.py`)
- [~] hand (drag & throw), flick, swatter, bomb (1.5 s fuse), torch (heat + max pain), cleaner (dissolve), zapper (all touch neurons + random 30 % for 40 ms), freeze (cold cells, frozen solid, shatter), spider, sugar, alcohol (PAM reward + inebriation), laser (`lab/laser.py`), cva, decoy (`_decoy`, decoy female, courtship), fruit, frog, dragonfly, mantis. **Rust:** implemented (see 4.2). Torch pain maxing needs the pain index, which is not ported.
- [ ] Loadouts: hotbar slots/pages, presets, editor (Q), tool wheel, saved loadouts (`core/loadout.py`, `ui/loadout_ui.py`), migrated-config notice.
- [x] Per-tool scent glomeruli (`Brain.__init__` TOOL_NAMES → 6 of 53, alcohol DM1/DM2/DP1m, cva DA1, decoy DA1+VA1v, laser none) using `default_rng(11)` permutation of glomeruli. **Rust:** `groups_match_python`: all 18 tools' scent rows and the player/odor deals match the NumPy code exactly.

## 6. Arenas (10; `kick_the_fly.py:ARENAS`)
- [~] room, fan (JO-C/E wind), flypaper (glue), pool (HRN humidity, drowning), lamp (photoreceptors, phototaxis, hot bulb), thermo (15–35 °C gradient; `thermo_gradient`), escape room (`Game.reset_escaperoom`, speedrun codes), kitchen (3D only; `game/kitchen.py`: bowl, vinegar trap, sink, stove, cook), field and orchard (3D only; `game/outdoors.py`: `WorldSpec`, wind, sun, `Orchard`, `Fruit`, trees, LOST flies). **Rust:** all ten arenas build and run (`arena.rs`, `environment.rs`, `kitchen.rs`, `world.rs`): room, fan (JO-C/E wind from the Python gust formula), flypaper (glue holds feet, release odds as Python), pool (buoyancy, HRN, drowning), lamp (photoreceptors, heat, phototaxis), thermo gradient, escape room (fan+paper+lamp+sugar goal; the speedrun CODE and `scores.json` entry are not ported), kitchen (counter, fridge, bowl, vinegar trap with the DM1/DM2/DP1m scent tracker, sink, burner, the cook who swings every 10-20 s, 7 unit tests in `kitchen.rs`), open field and orchard (trees, rocks, grass, fruit that regrows, wind, sun, lost flies). Every arena was screenshotted.
- [~] Weather (`game/weather.py:Weather`): rain, gusts, storms; day/night cycle (`outdoors.diurnal_cycle`, circadian clock neurons l-LNv/s-LNv, dFB sleep state). **Rust:** `weather.rs` (6 unit tests: storm preset, 6 hits/s on wings/body, flashes, thunder delay, reduced-flashing swell, wing wetting) + `weather_play.rs` (GPU rain, lightning drives the light, thunder) + `environment.rs` (rain pokes by body part, humid air, wet wings, gusts on JO, sun on the photoreceptors, clock neurons). Day/night: sun path and scene light done; the dFB sleep readout is the `sleep` reaction. Not verified: a full day cycle by eye.
- [~] Predators (`game/predators.py` engine: frog tongue, dragonfly, mantis creep + strike; `game/predator_play.py`, `game/predator_anim.py` IK spider legs; Lab assay `lab/predators.py`). **Rust:** `predators.rs` (7 tests): the frog crouches, leaps on a ballistic arc (apex 0.45 m, 0.43 s in the air, never slides), lands and recovers, tongue in 0.07 s with overshoot; mantis creep/freeze/strike; dragonfly pursuit with bank. `predator_play.rs`: captures poke the touch neurons, damage 55, carry and eat (immortal: let go); procedural bodies from the engine's pose (squash/stretch/hind/tuck/reach for the frog, rear/cock/sway for the mantis, bank/reach for the dragonfly, IK legs with planted feet for the mantis and spider). Frog screenshotted mid-approach. Lab assay (`lab/predators.py`): not ported.
- [~] Streamer mode (`core/streamer.py`: Twitch IRC chat voting, `netguard`, `game/live_inputs.py`), microphone → Johnston's organ (`core/mic.py`: FFT bands → JO-A/JO-B drive; `lab/audio.py` hum demo). **Rust:** microphone -> JO-A/JO-B only (`mic_analysis.rs`: two 8th-order band-passes 10-100 Hz and 100-1000 Hz, noise gate, full drive at RMS 0.05, 5 tests; `mic.rs`: cpal capture on its own thread, PulseAudio host first (what PipeWire serves), then ALSA/WASAPI/default; with no device it reports why and carries on, tested). F6 toggles. The band-passes are Butterworth biquads (HP x LP), same order as scipy's but not the identical transfer function. Streamer mode (Twitch) and the hum demo are not ported.

## 7. Play-mode extras

- [~] Challenges (`lab/challenges.py:Challenge` + `TMaze`, `Sneak`, `Sweet`, `ReverseSurgery`, `PredictNeuron`; `scores.json`; stars), Puppeteer (10 levels, no hands; `game/puppeteer.py`, `lab/puppet_challenge.py`), Contraption builder (`game/contraption.py`, `lab/contraption_challenge.py`, `contraptions.json`). **Rust:** Puppeteer (`puppeteer.rs`: 10 levels, goal tracker, par/stars, 7 tests; `arcade.rs`: the level flow with the laser palette, latching, hints, loom shadows, score file) and a physical Contraption (`contraption.rs`: ramp, marble, 7 dominoes, button, 4 payloads). Not ported: the T-maze/Sneak/Sweet/ReverseSurgery/Predict challenges, the 2D contraption bench, its build format/validator/slots/share codes, `contraptions.json`.
- [~] Neurodex (`core/neurodex.py`, `ui/neurodex_ui.py`, `data/neurodex_facts.yaml`, `neurodex.json`), Neuron of the Day (`core/neuron_of_day.py`), kill cam (`core/killcam.py`, `game/extras3.py:Extras`), personality cards (`core/cards.py`, `core/individuality.py:compute_personality_card`), individuality modes off/subtle/strong. **Rust:** kill cam only: `killcam.rs` (5 tests: encode/decode, ring of 6 s at 25 Hz, top risers, playback clock) + `killcam_play.rs` (slow-motion replay in the big brain view, risers pulse, Enter skips; screenshotted). Neurodex, Neuron of the Day, personality cards, individuality modes: not ported.
- [~] Fly arcade (`ui/arcade_ui.py`): tournaments (`lab/tournament.py`, `game/flyduel.py`), racing + odds + bets (`lab/racing.py`, `game/flyrace.py`, `core/points.py:Wallet`, points only), lesion battles (`game/lesion_battle.py`, `ui/battle_ui.py`, `lesion_loadouts.json`). **Rust:** `arcade.rs` (O): tournaments (`tournament.rs`, 3 tests: bracket order, 2 rematches then a seeded coin toss), racing (`race.rs`, 3 tests: speed rule v2, lures, ordering) with bets (`points.rs`, 3 tests: odds, house take, top-up, file), lesion battles (`lesion.rs`: budget 3, <=500 neurons, 2 tests). The matches run on background threads with brains built from the loaded connectome; a short real-brain duel/lane/lesion check is in `tests/arcade_real.rs` (a full 20 s duel takes minutes). Not ported: personality cards, race odds from sampled runs (odds are flat and labelled), share codes, the bracket replay viewer, saved loadouts.
- [x] Fly's-eye view (F4; `game/fly_eye.py`, visual only), mini-papers (`lab/minipapers.py`, `ui/minipaper_ui.py`, six papers), classroom/lectures (`lab/classroom.py`, `lab/labclassroom.py`, `protocols/lecture_*.yaml`). **Rust:** `flyeye.rs` + `shaders/eye.wgsl` (F4): a wide-angle camera at the fly's head renders to a texture shown as two discs of hexagonal ommatidia in a UV-blue-green spectrum, all on the GPU; visual only. Screenshotted. (Mini-papers and the classroom are not ported.)
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
- [ ] Headless runs, protocol YAML (`lab/protocol.py`; 42 files in `protocols/`), playthrough bot (`lab/playthrough.py`), benchmark (`lab/benchmark.py`), Python API (`lab/api.py:Fly`) → Rust crate API + optional PyO3 later.

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
| `config.toml` (schema v3) | `core/config.py` | **loaded and written by Rust** (`ktf_data::config`): all 75 settings, keys, pad, loadouts, first_run, migrations, clamping, `.bad` copy; tests load 6 real/hand files and compare to Python's values, and rewriting a Python file gives the same text |
| `.ktfsave` (zip: npz arrays + JSON header, format `kick-the-fly-save` v1) | `core/savestate.py` | **read and written by Rust** (`ktf_data::savestate`): header, compatibility texts, lossless rewrite (Python reads it back), 128-bit PCG64 state. `LifSim::import_state` continues a real Python save **spike-for-spike** (60 steps hashed against Python's raster). Not yet: restoring bodies/timers (`flies[i].fly/slot`, `extra`) into the Rust fly, the Load/Save menu |
| `.ktfreplay` (zip, per-step spike SHA-256) | `core/replay.py` | |
| pet `pet.ktfsave` + `.bak` | `core/pet.py` | |
| memory `fly-memory.npz` + `training-log.json` | `core/memory.py` | **read and written** (`ktf_data::stores::FlyMemory`), v2.5.0 fixture loads; the plasticity itself (Memory.step) is not ported |
| `neurodex.json`, `scores.json`, `contraptions.json`, `lesion_loadouts.json` | `core/neurodex.py`, `lab/challenges.py`, `game/contraption.py`, `game/lesion_battle.py` | **all four read/written** (`ktf_data::stores`, `game::paths::Scores`) incl. Python's tolerance (`.bad`, newer-version read-only, bad entries skipped) |
| share codes `KTF1-…`, `.ktfshare` | `core/sharecode.py` | **read and written by Rust** (`ktf_data::sharecode`): all 6 kinds, validate/preview, refusal texts equal to Python's; tests decode real Python codes + a file, and Python decodes Rust's. Protocol/contraption payloads get a container check only (per-part ranges and protocol schema: with their features) |
| experiment bundles (zip) | `lab/bundle.py` | |
| recordings CSV/npz + metadata JSON; NWB; TIFF | `lab/recorder.py`, `nwbexport.py`, `imaging.py` | |
| protocols YAML (42), lecture YAML | `lab/protocol.py`, `protocols/` | |
| `data/driver_lines.yaml`, `data/neurodex_facts.yaml`, `data/locales/*.json` | `data/__init__.py` | |
| result CSV/JSON/SVG (validation, race, netsci, sensitivity, screens, autopsy) | `lab/*` | compare outputs with `docs/results/` |

## 11. Audio
- [~] Sound effects and wing buzz (pygame mixer synthesised in `Sound`), stethoscope clicks, sonification synth (`core/sonify.py:Synth`), hum demo; Bevy audio or `rodio` via `bevy_audio`. **Rust:** see 4.1 Sounds.

## 12. Tests to port (110 files in `tests/`)
- [x] Simulator golden tests (this step): `crates/fly_sim/tests/golden.rs` against `tests/golden/`.
- [~] Everything else: `test_backends`, `test_determinism`, `test_validation`, `test_config`, `test_savestate`/replay/share/bundle, assay and Lab page tests, playthrough bot, etc. Port by area together with the feature. **Rust:** game logic and physics now have tests: `crates/game` 84 lib tests (anim, predators, weather, world, kitchen, reactions, duel, race, points, tournament, puppeteer, lesion, killcam, synth, mic analysis, paths, contraption), `tests/fly_physics.rs`, `tests/arcade_real.rs`, `tests/headless_app.rs`; `fly_sim` 26 incl. the groups golden test. Still to port by area: determinism/validation/config/savestate/replay/share/bundle tests and the Lab suites.

## 13. Docs and release
- [ ] `README.md`, `docs/*.md` (47 files) updated for the Rust build; `CHANGELOG.md`; `RELEASE_NOTES.md`; screenshots tool (`tools/make_screenshots.py`).
- [ ] Move the Python tree to `legacy-python/` **unchanged** (done at the end of the port task, see git history).

## 14. Architecture decisions for the Rust build
- [x] Simulation on its own OS thread at a fixed 200 steps/s timestep (`crates/game/src/sim_thread.rs`); the render thread only reads the latest published snapshot (a one-slot mailbox with recycled buffers, `snapshot.rs`; locks are held only for a pointer move) and sends commands over a bounded channel.
- [x] Dev builds use Bevy's `dynamic_linking` feature; release builds do not (`crates/game/Cargo.toml` feature `dev`).
- [x] Exact version pins in `Cargo.toml` (`bevy = "=0.19.1"`, no third-party Bevy plugins yet).
- [x] UI toolkit, audio, gamepad and video-recording crates: choose and pin when their feature is ported. **Rust:** UI: Bevy UI (text and nodes) + custom `UiMaterial` for the fly's eye; audio: `bevy_audio` (WAV assets + a streaming `Decodable`) and `cpal =0.18.2` for the microphone; physics: `avian3d =0.7.0` (built for Bevy 0.19; bevy_rapier3d 0.36 is too, avian was chosen for its ECS-native joints and `Forces` API); `bytemuck =1.25.2` for GPU buffers. No gamepad (gilrs) or video-recording crate yet.

## 15. Layout after the Python move
- `legacy-python/` holds the Python game unchanged (`kickthefly/`, `kick_the_fly.py`, `tools/`, `protocols/`, `tests/`, `requirements.txt`, `pytest.ini`, `watch_tests.sh`) plus `make_golden.py`. Python resolves `data/` and `protocols/` relative to its own folder, so to run it put (or symlink) the dataset at `legacy-python/data/`.
- Left at the root and **not yet updated for Rust**: `build_appimage.sh`, `build_exe.ps1`, `.github/workflows/*`, `packaging/`, `README.md`, `docs/`. They still describe and build the Python game.
- `tests/golden/` stays at the root (shared by `legacy-python/make_golden.py` and `crates/fly_sim/tests/golden.rs`).

## 16. GUI parity (Rust must look and act like pygame)

Rule: before building a screen, run or read its pygame version (`legacy-python/kickthefly/ui/*.py`, `lab/*.py`) and replicate layout, labels, colors, fonts,
hotkeys and timing; do not redesign. If `bevy_egui` can't match, use `bevy_ui` with custom styling. Tick a box only when the note says how it was checked
against the pygame version (screenshot pair, or the code read side by side). Anything that can't match exactly gets a written reason below it.

- [~] Menu framework (`ui/menu.py:Menu`): **ported** as `crates/ktf_ui` (software canvas + immediate-mode `Menu`, same widgets: button, toggle, segmented, slider with typed value, number/text field, chip, tooltip after 0.35 s, scrolling, clipping, key/pad capture, flash messages). Verified: 9 pygame screenshots (`tests/ui_reference/`, `legacy-python/make_ui_reference.py`) vs Rust renders, < 0.9 % of pixels differ clearly (glyph anti-aliasing); 6 interaction tests. Not yet: charts, `flow_buttons`/`button_row`/`subtitle` helpers (added with the pages that use them), gamepad navigation, UI scale / menu size U
- [~] Fonts and sprites: `assets/fonts/FreeSans(.Bold).otf` = the font pygame's `SysFont("segoeui,dejavusans,...")` resolved to on the reference machine (Segoe UI exists only on Windows and cannot be bundled); glyph advances are pygame's hinted ones (`crates/ktf_ui/data/advances.json`), so every text width equals pygame's (330 strings checked). Icons/sprites: none used by the menus. Colors: `ktf_ui::theme`
- [x] Pause menu (Esc): 12 buttons in the original order, labels, tips, Play/Lab/Pet title; screenshot pair identical (pause, pause_lab); Esc opens, frees the mouse and stops time and brain (`menu_ui.rs`); quit asks first (`confirm_quit`, screenshot pair)
- [~] Settings: Graphics, Audio, Brain (+ENGINE panel lines supplied by the game), Controls (keys table, pad table in 3D), Accessibility: all 75 settings drawn from the same schema, each tab's screenshot compared with pygame's; changes applied live (volume, mute, FOV, mouse, immortal, fullscreen, sim speed, language) and saved to config.toml. Not yet: Help tab (`ui/help_ui.py`), live effect of graphics options (fps cap, vsync, resolution scale, ui scale...), ENGINE lines
- [ ] HUD: pain / reward meters, hit counter, hotbar + pages, hotkey help (H), science cards + source chips, tag chips, toasts, validation popups
- [ ] Brain panel + big brain view (B), neuron inspector, search, stethoscope readout
- [ ] Tutorial / first-run flow, What's New, loadout notice
- [ ] Menu page `arcade`
- [ ] Menu page `bugreport`
- [ ] Menu page `challenges`
- [ ] Menu page `crash`
- [ ] Menu page `lab`
- [ ] Menu page `lab_assays`
- [ ] Menu page `lab_assumptions`
- [ ] Menu page `lab_asymmetry`
- [ ] Menu page `lab_autopsy`
- [ ] Menu page `lab_benchmark`
- [ ] Menu page `lab_clamp`
- [ ] Menu page `lab_classroom`
- [ ] Menu page `lab_critical`
- [ ] Menu page `lab_diff`
- [ ] Menu page `lab_export`
- [ ] Menu page `lab_laser`
- [ ] Menu page `lab_minipapers`
- [ ] Menu page `lab_netsci`
- [ ] Menu page `lab_params`
- [ ] Menu page `lab_protocols`
- [ ] Menu page `lab_psych`
- [ ] Menu page `lab_rigs`
- [ ] Menu page `lab_sensitivity`
- [ ] Menu page `lab_sleepdep`
- [ ] Menu page `lab_validation`
- [ ] Menu page `lab_wiring`
- [ ] Menu page `live_inputs`
- [ ] Menu page `load_state`
- [ ] Menu page `loadout`
- [ ] Menu page `loadout_notice`
- [ ] Menu page `neurodex`
- [ ] Menu page `neuron_shapes_ask`
- [ ] Menu page `selftest`
- [ ] Menu page `share`
- [ ] Menu page `whatsnew`
- [ ] Time controls UI (pause, slow-mo, single step), popups and overlays (death/autopsy, kill cam, photo mode, profiler F3, fly's-eye F4)
- [ ] Mode split: Play stays simple; Lab-only stats only in Lab; "Real flies do this too" cards default OFF (`brain.science_popups` false)
- [ ] Streamer mode panel, Mic panel (`live_inputs`)
- [ ] Neurodex, Neuron of the Day, guided mini-papers
- [ ] Transitions and animations (menu open/close, fades, flashes honoring `reduced_flashing`)

### Cannot be matched exactly
- **Text rasterization**: pygame renders hinted glyphs with FreeType; `ab_glyph` renders unhinted outlines. Widths and positions are identical (pygame's advances are a lookup table), edges differ by anti-aliasing (a few percent of text pixels).
- **Font**: Python used the first installed of Segoe UI, DejaVu Sans, Liberation Sans, Arial: the look depended on the OS. The Rust game bundles GNU FreeSans (what that list resolved to on the reference Linux machine) so all OSes look like it. FreeSans is GPL-3.0+ with the GNU font exception: **check this is acceptable for redistribution before a release**; swapping in DejaVu (permissive) changes only `assets/fonts/` and `crates/ktf_ui/data/advances.json` (regenerate with `legacy-python/make_ui_metrics.py`).
- **Veil**: pygame blends the dark veil in sRGB space, Bevy in linear light; the in-game veil alpha is 240 instead of 196 so the room is as dark as in pygame (measured within 3 levels).
- **Window**: the menu canvas is the window's logical size (Python: the fixed HUD surface scaled to the window), so layout is identical at 1280x720 and reflows like Python's at other sizes only approximately.
