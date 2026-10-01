# Handoff: 3.0 day 2 (written by Sonnet on branch `sonnet/3.0-day2`, for Opus to test and merge into `release/3.0`)

Nothing is pushed. Branch made from `release/3.0` (323f8f2). Version is not bumped. CHANGELOG has a "Day 2" block under 3.0.0.

## Read this first

1. **Validation is identical.** `--headless --validate --sim-backend cpu` on a pristine `release/3.0` worktree and on this branch's final
   code: **0 differences** (timestamps, workers and version ignored), 21 tests, 12 PASS / 9 FAIL, the documented results. I ran the baseline
   twice (once from this checkout early on, once from a clean worktree) and they agree with each other too. The only code near the
   simulation that changed is inert unless a new feature is used: `Brain.currents` / `set_current` (named extra currents, added to the drive
   only while one is set), `Brain.probe` (an optional per-step callback, `None` by default), and two optional fields on `sim.wiring.Wiring`
   (`nt_scales`, `nt_min_conf`; `Wiring().as_dict()` is byte-identical to before, so validation metadata and old saves don't change).
2. **Everything ran on the real adult pack** (166,700 neurons), built here with the README's documented steps (Janelia's public MaleCNS
   v1.0 bucket, CC BY 4.0, into the git-ignored `data/`; you said go ahead). Plumbing tests also run on the synthetic pack.
3. **Two environment facts you'll hit.** (a) `.venv` and `data` were committed by the day-1 review as **absolute symlinks pointing at
   themselves** (`.venv -> /home/lolo/kick-the-fly/.venv`), so the checkout had no venv and no data; I made a real `.venv` and a real `data/`
   and **untracked the two symlinks in my commit** (`git rm --cached`). (b) Your shell exports `SDL_VIDEODRIVER=wayland`, which overrides
   `conftest.py`'s `setdefault("dummy")`: my first full pytest run opened a real, empty pygame window on your screen. I killed it; every later
   run sets `SDL_VIDEODRIVER=dummy` (playthrough: `offscreen`). You may want `conftest.py` to force the dummy driver rather than default to it.
4. **Venv changes (all inside `.venv`, `requirements.txt` untouched):** `pygame` replaced by `pygame-ce` (pygame 2.6.1 has no font module on
   Python 3.14, as day 1 found), `pynwb` installed (optional dependency; it is what showed the NWB bug below). No system package, driver,
   sudo, kernel or env-var change. GPU: only `--selftest`'s own probe ran (OpenGL 4.6 Mesa, AMD RX 9070 XT, GL backend and GPU checks PASS); the
   playthrough ran on `--sim-backend cpu`. No GPU error occurred.
5. **Network use, all at development time, none at runtime.** The builder `tools/build_driver_lines.py` downloads one public table; I
   also read PubMed / Europe PMC / the eLife API to check citations. The game itself makes no request for any of this, collects no
   telemetry, and uses no microphone.

## What was built

### 1. Genetic toolkit (`lab/genetics.py`, `data/driver_lines.yaml`, `tools/build_driver_lines.py`, Lab > Genetic toolkit, docs/genetics.md)
- **Source: a real, citable, redistributable table.** Meissner et al. 2025, *A split-GAL4 driver line resource for Drosophila neuron types*,
  eLife 13:RP98405, doi:10.7554/eLife.98405, Figure 1-source data 1 (**CC BY 4.0**, checked in the eLife API record). It lists each line, the
  cell types it labels and an expression-quality score. `driver_lines.yaml` holds its 2,667 adult lines that name a cell type, with the
  table's SHA-256 and the license in the header; the builder is the only way it should change. **No line name was written by hand.** (My
  own memory of SS00731 as the giant fiber was wrong: the table says DNa01; SS00727 and SS02299 are the DNp01 lines.)
- It lives in `kickthefly/data/`, not `data/` (the repo-root `data/` is git-ignored), so it ships in the exe/AppImage (`--collect-data
  kickthefly` from day 1).
- Matching to MaleCNS types is by **exact spelling only** (GAME RULE). On the real pack 1,338 of 2,667 lines match at least one type; 962
  of 2,044 names match. The rest are shown as unmatched, never guessed; a line with no match selects nothing and says why.
- **Off-target** is the paper's own quality score in its own words (1 one cell type, 2 two, 3 three or more, 4 weak/variable, 5 not
  stabilized; any other value, e.g. the single "Control", says "nothing is claimed").
- `line:SS00727` is a neuron spec everywhere `simcore.rows_of` is used (surgery in protocols, recordings, stimuli, `Fly.neurons`) and in the
  laser's target. The screen lets you silence / stimulate the line's types (writes the same `type_ops` switches as the inspector, so
  save states and share codes just work), aim the laser, or express TrpA1 / shibire-ts in it.
- **Gap, documented:** no GAL4 (non-split) mapping was found that could be redistributed, so none is offered.

### 2. Thermogenetics (`lab/thermogenetics.py`, `lab/livelab.py`, Lab > Thermogenetics, assay `thermo_escape`, docs/thermogenetics.md)
- **Cited (approximate, read in the abstracts):** TrpA1 tonic spiking near 25 C (Pulver 2009, J Neurophysiol, doi:10.1152/jn.00071.2009);
  shibire-ts motionless within 2 min at 30 C, recovery in about 1 min (Kitamoto 2001, J Neurobiol, doi:10.1002/neu.1018). **Game rules:** the
  full-on ends (29 C; shibire-ts ramp 28 -> 30 C), kinetics (TrpA1 1 s; shibire-ts 40 s on / 20 s off), current sizes (0.5 = the validation
  drive; -0.6 = surgery's silencing), the thermo arena's 15-35 C map. **Model:** only expressing neurons respond; shibire-ts silences the
  neuron instead of blocking its terminal. Each is tagged on screen, in the docstring and in Lab > Model assumptions.
- Temperature sources: thermo arena (per fly, set in `Game._thermo_tick`), a Lab slider, protocol `thermogenetics:` (constant or schedule).
- **Assay (seeds 1000-1009, criteria written in `thermogenetics.ASSAY_CRITERIA` before the run):** C1 PASS, C2 PASS, C3 PASS. 18-24 C:
  0/10 escape in both arms; 26-36 C: 10/10 expressing vs 0/10 control (Fisher p < 0.001 at every temperature from 26 C up); DNp01 5.2-5.5 Hz
  below 26 C, 50 Hz at 26 C, 67 Hz from 28 C. Spearman rho = 0.845 (C3's bar is 0.8, cleared narrowly because the curve is a step, which
  ties). **Read it as a step, not a dose-response:** DNp01 is two neurons that hit their refractory limit, so a quarter of the TrpA1 current
  (26 C under the game-rule ramp) already crosses the game's 4x escape rule. The shape near 25 C is the game-rule onset passed through the
  model, not a finding about TrpA1. Full table in docs/thermogenetics.md.

### 3. Virtual patch clamp (`lab/patchclamp.py`, Lab > Patch clamp, inspector **PATCH (MODEL)** button in Lab mode, docs/patchclamp.md)
- **MODEL: a point-neuron LIF unit, not electrophysiology**, on screen, in the docstring, CSV headers, the NWB description and the
  assumptions page. The potential is in model units (threshold 1.0, reset 0.0), never converted to millivolts. Live trace with the
  threshold line and spike markers (the model has no spike waveform), current steps (amplitude, duration, repeats), holding current, I-F
  curve, CSV and NWB. The NWB is a generic `TimeSeries` (unit `a.u.`) plus a `Units` table, **deliberately not a volts `CurrentClampSeries`**.
- Two modes: **embedded** (the live wired brain; real synaptic input) and **isolated** (the same equation with no synaptic input). Isolated
  equals `LIFSim`'s own update to 1e-6 with noise off (test), its rheobase is analytic (I >= 0.0125), and **every neuron gives the same
  isolated curve** (all units are identical), which the screen says.
- The live trace asks the brain for real-time steps (`request_steps`), because the menu pauses the game; I did not observe it in a
  windowed game, only in tests and a headless-rendered page.

### 4. Simulated calcium imaging (`lab/imaging.py`, `lab/livelab.py`, Lab > Calcium imaging, docs/imaging.md)
- **MODEL.** spikes -> two-exponential GCaMP kernel -> per-neuron dF/F -> ROI mean -> Poisson shot noise -> dF/F. ROIs: per brain region, or
  any neuron specs (types, prefixes, `line:`), one ROI each. Brain view **Imaging mode** (badge "IMAGING (MODEL)"; the view's own renderer is
  fed dF/F-derived rates and recolored by an imaging palette). Exports: CSV, NWB (`DfOverF` + `Fluorescence` `RoiResponseSeries` over a
  `PlaneSegmentation`, plus an `ImageSeries` of the rendered frames), TIFF stack (multi-page RGB, JSON description with the MODEL tag).
- **Kernel speeds: what is and is not verified.** jGCaMP8m: half-rise 58 ms and half-decay 137 ms are stated in Zhang et al. 2023's text
  (Drosophila in-vivo visual responses); used as the kernel's time to peak because a two-exponential kernel cannot have both as written.
  **GCaMP6s / 6f: Chen et al. 2013's text only gives the rise ranges (100-150 ms, 50-75 ms; I use the middles). The half-decay numbers
  (550 and 140 ms) are the commonly quoted values that I could NOT find in the paper's text; they are labeled unverified on screen, in the
  docs and in the NWB. Someone with the paper open should check Figure 1.** I did not want to present them as published.
- **A flaw caught by looking at a real render:** with F0 = "no calcium" baseline, spontaneous firing alone read as dF/F of 4-5. dF/F is
  now against a 30 s running mean of each neuron's own fluorescence (the indicator state is primed from the brain's last 2 s of spikes); the
  zero baseline remains as an option for analytic tests. dF/F per spike (0.2) and the photon budget (100/cell/frame) are game parameters.
- Accessibility: palette follows Settings > Accessibility > Brain view colors (green / blue-yellow / high contrast); Reduced flashing smooths
  the view and plots the noise-free trace. Settings > Brain has the default indicator and frame rate (tagged MODEL).
- Cost on the real brain (median over 20 frames): feed 1.4 ms, view rates 0.5 ms, view render 10.6 ms, recolor 16.2 ms per frame, inside
  the view thread's 50 ms budget. **Not measured in a live window.**

### 5. Pharmacology (`lab/pharmacology.py`, extended `sim/wiring.py`, Lab > Pharmacology, docs/pharmacology.md)
- **MODEL PREDICTION.** Picrotoxin (GABA + glutamate, moved here from Lab > Connectome robustness; the tab is unregistered, the report
  helpers are reused by the new page), cholinergic block, glutamate-Cl block, GABA-A agonist (x2 at full dose, a game rule). Dose slider;
  a table of neurons / connections / **synapses affected at each confidence level** (measured, >= 0.9, 0.7-0.9, 0.5-0.7, < 0.5, no
  confidence); include/exclude low-confidence predictions with an adjustable cut (default 0.7; measured always counts). The counts are
  what the wiring will change (tested equal, and tested equal to the original `inhibition_scale` for picrotoxin on the real pack).
- **Octopamine and dopamine are left out on purpose:** those neurons' synapses carry no sign and are not in the simulated matrix, and there
  is no existing gain to scale honestly. Said in the docs, the docstring, the page's hover text and Model assumptions.
- Limit stated on screen: the simulator's slow global gain pushes back on anything that changes overall synaptic strength.

### Integration
- **Protocols** (all load; `protocols/`): `genetics_line_silencing`, `thermogenetic_dnp01`, `thermogenetic_escape_assay`,
  `patch_dnp01_if_curve`, `imaging_looming`, `drug_picrotoxin`, `drug_cholinergic_block`. New blocks `thermogenetics:`, `drug:`, `imaging:`
  on stimulus protocols and a standalone `patch:` kind; assays and `--record-replay` refuse them with a reason; bundles carry and rerun them
  bit-exactly (test). I ran the five non-assay examples through the CLI on the real pack; results below.
- **Python API:** `fly.line`, `express`, `temperature`, `unexpress`, `patch`, `image`, `drug`, `washout`; `line:` works as a neuron spec.
- **Playthrough:** six new checks (`extra:genetics`, `thermogenetics`, `patch`, `imaging`, `pharmacology`, `toolkit-pages`), criteria
  written first. **`--selftest`:** new "Lab toolkit" check. **i18n:** catalog synced (Lab screens, like the other Lab screens, are English
  only). **Docs:** five new pages, README section + tables, docs/lab.md, docs/api.md, CONTRIBUTING layout, docstring map, five assumptions cards.
- **Not added, on purpose:** nothing in the loadout editor (none of these is a tool; the laser already accepts `line:`), and no share-code
  kinds. **Gamepad:** the Lab menus remain mouse/keyboard (the general menu has never been pad-navigable, as day 1 noted); the game's own pad
  controls are unchanged.

## Test results (all on the real pack, CPU backend)
| run | result |
|---|---|
| `--validate` before (pristine `release/3.0`) vs after (this branch) | **identical**, 0 differences, 21 tests, 12 PASS / 9 FAIL |
| new tests | 146 (genetics 12, thermogenetics 22, patch clamp 19, imaging 30, pharmacology 19, Lab screens in the running game 23, API 7, protocols 14), all pass |
| full `pytest -m "not validation"` (earlier code, 46 min) | 749 passed, 2 failed, 15 skipped: `test_nwb_round_trip` (the NWB bug below, fixed mid-run, passes now) and the playthrough slice (see below) |
| full `pytest` again on the final code | **763 passed, 2 failed, 15 skipped** (45 min). The failures: the playthrough slice (the pre-existing `extra:neurodex`) and `test_playthrough_extras.py::test_bundle_extra_passes`, which called `extra_bundle` with the same wrong argument order as the bot bug; I fixed the test to the bot's real call order and it passes |
| `--selftest` | 18 PASS, 1 WARN (audio, only because I forced the dummy audio driver), 0 FAIL; GL 4.6 / GPU probe / GL backend PASS |
| `--headless --playthrough all --sim-backend cpu`, **final code** (SDL offscreen) | **315 passed, 1 failed, 69 gated, 11 skipped.** All six new checks PASS, `extra:bundle-rerun` PASS (bit-exact). The one FAIL is the pre-existing `extra:neurodex`. (An earlier run on pre-fix code: 313 / 4 failed, see below.) |
| assay `thermo_escape`, seeds 1000-1009 | C1 / C2 / C3 PASS (details above) |
| protocols through the CLI (real pack) | `genetics_line_silencing`: treated giant fiber 0.00 Hz vs control 19.23 Hz (looming detectors unchanged, 21.3 vs 21.3); `thermogenetic_dnp01` ran (giant fiber 40.0 Hz over a run that warms from 22 to 32 C at 2 s); `patch_dnp01_if_curve`: 4.9 / 13.3 / 19.6 / 34.7 / 47.1 / 57.1 / 67.8 Hz at 0 / .01 / .02 / .05 / .1 / .2 / .5; `drug_picrotoxin` whole-brain 5.16 Hz; `drug_cholinergic_block` whole-brain 2.68 Hz (baseline about 6.4 Hz, see the playthrough check) |
| Lab pages rendered on the real adult brain and looked at | genetics, thermogenetics, patch (isolated and live), imaging (traces and the big view), pharmacology: overflow / overlap problems found that way are fixed (tag row wrapping, toggle label overlap, legend over axis label) |

### The four playthrough failures
- `extra:neurodex` ("calm discoveries not tagged 'rest'", e.g. AN01B002 tagged 'play'): **pre-existing**. I ran the same playthrough test on a
  pristine `release/3.0` worktree and it fails identically. Day 1's review wrote that the playthrough was not run on the real pack. I did
  **not** touch it: it is a question about the day-1 discovery/tagging rule, which is yours to decide.
- `extra:bundle-rerun`: **pre-existing bug** (`add(r, fn, *a)` calls `fn(*a, r)` so the function's parameters are `(backend, tmp, r)`, but it was
  declared `(backend, r, tmp)`; `TypeError`). I fixed the signature; the check now PASSes bit-exact on the real pack.
- `extra:imaging` and `extra:toolkit-pages` were **my** checks with the same argument-order bug and a missing import; fixed, both pass.
  The full `--playthrough all` was re-run on the final code: only the neurodex check fails.
- The 10-11 skips are the pool / flypaper / lamp-alcohol / escape-room combos (the count moved by one between runs: a 2D flypaper:hand
  combo skipped in the second run); they are skips by the bot's own rules, not new.

## Bugs found on the way (beyond the brief)
1. **`nwbexport._provenance` referenced an undefined `br`**, so every NWB export of a recording raised `NameError` once `pynwb` was installed
   (it was never caught because pynwb wasn't). One-line fix (`rec.brain`), `tests/test_nwb.py::test_nwb_round_trip` passes. This also means NWB
   export has probably never worked since commit 6615742.
2. My own: `fly.temperature(..., kinetics="steady")` before `fly.express(...)` was silently dropped. Found by the playthrough bot; fixed with a
   regression test.

## Known issues / what I did NOT verify
- **GCaMP6s / 6f half-decay (550 / 140 ms) are not checked against the paper** (above). Everything else cited was read from abstracts or full
  text through PubMed / Europe PMC / eLife; the TrpA1 and shibire-ts onsets come from abstracts only.
- **3D game:** the new screens, the inspector PATCH button and Imaging mode were exercised through the shared 2D/3D code paths and draw
  tests (`Game3D` inherits the big-view and inspector code), and I looked at headless-rendered 2D pages; **nobody looked at a real 3D frame**
  of the imaging view or the patch button. The thermo arena hook (`arena_temp_c`) is tested by calling `_thermo_tick`, not in a windowed 3D run.
- The live patch trace and imaging view thread were tested piece by piece, not through a real window's frame loop; the live trace uses
  `request_steps` to advance the menu-paused brain.
- NWB / TIFF files were read back with `pynwb` and Pillow only; `nwbinspector` was not run, and no other viewer opened them.
- No GPU compute backend (torch / numba) was available or tried; no real gamepad or clipboard; exe / AppImage not built.
- Imaging: a real ROI pipeline estimates F0 and segments cells; here F0 is a running mean and ROIs are neuron sets. The first seconds of a
  session are primed, not exact. Shot-noise photon budget and dF/F per spike are my choices.
- **Design choices for you:** (a) the assay reports a step, not a curve (above); do you want a slower TrpA1 ramp or accept that? (b) the
  running-mean F0 time constant (30 s) and the imaging default frame rate; (c) whether `conftest.py` should force the dummy SDL driver; (d)
  the day-1 neurodex-at-rest bot check.

## Files
Added: `kickthefly/lab/{genetics,thermogenetics,patchclamp,imaging,pharmacology,livelab,labtoolkit}.py`, `kickthefly/data/driver_lines.yaml`,
`tools/build_driver_lines.py`, `docs/{genetics,thermogenetics,patchclamp,imaging,pharmacology}.md`, seven `protocols/*.yaml`, eight
test files in `tests/` (`test_genetics`, `test_thermogenetics`, `test_patchclamp`, `test_imaging`, `test_pharmacology`, `test_toolkit_day2`,
`test_api_day2`, `test_protocols_day2`).
Changed: `game/kick_the_fly.py` (docstring; `Brain.currents/set_current/probe`; `Game._lab_tick`, `patch_neuron`, inspector PATCH button,
imaging view path, status badge, lab hub entries, thermo-arena temperature), `game/kick3d.py` (calls `_lab_tick`), `core/simcore.py`
(`line:` spec), `sim/wiring.py` (nt scaling), `lab/{api,protocol,labjobs,lab,labwiring,laser,playthrough,nwbexport}.py`, `core/{config,selftest}.py`,
locale catalogs, `tools/i18n_sync.py`, README, CHANGELOG, CONTRIBUTING, docs/lab.md, docs/api.md. `.venv` and `data` symlinks untracked.
