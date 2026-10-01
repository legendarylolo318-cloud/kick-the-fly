# Handoff: 3.0 day 2 review (Opus, branch `opus/3.0-day2-review`, merged into `release/3.0`)

Reviewed Sonnet's `sonnet/3.0-day2` (4745422; see HANDOFF_3.0_DAY2.md). Review commit c9a50f8, merge commit b1ab364. Everything
below was run on this machine, on the real adult pack (166,700 neurons) unless it says "synthetic pack". Numbers are measured.

## Read this first

1. **Validation is identical.** `--headless --validate --sim-backend cpu --workers 5`, baseline regenerated today from a clean
   `release/3.0` worktree (323f8f2), against the merge commit b1ab364: **2,020 leaf values compared, 0 differences** (only `created`,
   `seconds`, `workers` and `app_version` at the top level ignored), 21 tests, **12 PASS / 9 FAIL**, the documented results.
2. **The pre-existing `extra:neurodex` playthrough FAIL was a bug in the bot's check, not in the game.** Details below. The
   criterion is unchanged; the check now measures what it says.
3. **GCaMP6s / 6f are now verified, and one number was wrong.** Chen et al. 2013's main text has no half-decay numbers; they are in
   its **Supplementary Table 3** (mouse V1 in vivo, cell-attached, single action potentials): GCaMP6s rise time to peak 179 +- 23 ms,
   half-decay 550 +- 52 ms; GCaMP6f 45 +- 4 ms and **142** +- 11 ms (day 2 had 140). Both numbers of each kernel now come from that
   one table: 179/550 and 45/142 ms (day 2 used the middles of the main text's rise ranges, 125 and 62 ms, which are the intervals at
   which single spikes became resolvable, a different quantity). Read in the paper's supplementary PDF, fetched through Europe PMC's
   API (PMC3777791). Screen, docs, Settings hover, Lab > Model assumptions and the docstring say so. jGCaMP8m unchanged.
4. **conftest.py now forces the dummy SDL drivers** (`KTF_TEST_SDL_VIDEODRIVER` / `KTF_TEST_SDL_AUDIODRIVER` to override on
   purpose). With your shell's `SDL_VIDEODRIVER=wayland` the old `setdefault` opened real windows. Regression test included.
5. **Imaging mode did not work on the real pack, in 2D or 3D** (a race, below). Found only by looking at a real 3D frame: every
   day 2 test passed because the synthetic pack builds the session too fast to lose the race.
6. **A GPU-related failure happened once; I did not work around it.** The first 3D scene rendered fine (SDL offscreen, OpenGL);
   a second `kick3d.run` in the **same process** failed with
   `kickthefly.game.kick3d.GLUnavailable: Exception: (detect) glXGetCurrentContext: cannot detect OpenGL context`
   (from `moderngl.create_context()` in `kick3d.App.__init__`). I changed nothing (no env var, no driver, no package) and ran
   every later 3D scene in its own process, which is exactly the path that had just worked; all 14 such runs succeeded. Note for
   `tools/make_screenshots.py`: it renders several scenes in one process, so it may hit the same error here.

## Bugs found and fixed (each with a regression test in `tests/test_day2_review.py`; 64 tests, 60 fail on 4745422)

The four that pass on Sonnet's code are guards, said so in their docstrings: two hostile inputs the old code already refused
(`amplitudes: "0.1"`, a `-inf` current), the imaging step-count rewind (my hypothesis was wrong: it already recovered, so I
reverted my change), and the conftest guard, which fails on the old conftest only when the shell sets another driver (checked
with `SDL_VIDEODRIVER=wayland`, running only that test, so no window could open).

### Seen in real frames
1. **Imaging mode never turned on from Lab > Calcium imaging.** `ImagingLive.set_on` set `on = True` before building the session;
   on the real brain that takes a moment (kernel fit, region ROIs over 166,700 neurons), the view thread saw imaging on with no
   session, raised "no imaging session" and switched it off, then `start()` cleared the error. The session is now built first and
   published last. `test_imaging_is_not_on_until_its_session_exists`.
2. **The inspector's PATCH button was off the card** (`card.x + 312` on a 308 px card): clipped by the big view, only "PAT"
   showed, in 2D and 3D. Now its own row ("PATCH CLAMP THIS NEURON (MODEL)"), inside the view; a real mouse down/up opens Lab >
   Patch. `test_the_inspector_patch_button_is_on_screen_and_clickable`.
3. **Imaging mode showed "50,776 firing / 3,127 pain"** and the firing legend: the view is fed dF/F-derived rates, so those counted
   imaging pixels. Hidden in Imaging mode; the big view says "dF/F (MODEL)" and explains the brightness.
   `test_imaging_mode_does_not_show_firing_counts`.
4. **The thermo page showed the idle slider's 22.0 C** with "Thermo arena" as the source. It now shows what each fly senses (or
   that the slider applies outside the thermo arena). `test_thermo_page_shows_what_the_flies_sense_in_the_arena`.

### Threads and state
5. **`Brain.set_current` zeroed the live `inject` array in place, then re-added the sources,** from the game thread (the Lab tick)
   while the brain thread steps and reads it: a step could see a half-built sum. The sum is now built in a new array and swapped in
   (under the brain's lock). `test_setting_a_current_swaps_in_a_new_array_...`.
6. **A probe that raises ended the brain thread for good** (the game freezes). It is now logged and removed.
   `test_a_probe_that_raises_is_removed_and_does_not_kill_the_brain`.
7. **`LiveElectrode.run` set the schedule before the sample buffers existed:** a brain step in between raised AttributeError on the
   brain thread (and so, with 6, froze the brain). Buffers first, schedule last; `_push` reads it once.
   `test_starting_a_live_protocol_never_exposes_half_set_state_to_the_brain_thread` (steps exactly when the schedule appears).
8. **Live thermogenetics keyed its per-brain state by `id(brain)`.** A respawned fly or a loaded save gets a new Brain, which
   CPython can place at a dead brain's address: it inherited the old state, including "this current is already written", and never
   got its current. State is now tied to the brain object; flies that leave are forgotten (they used to accumulate).
   `test_live_thermo_state_is_tied_to_the_brain_object_not_its_id`, `test_live_thermo_forgets_brains_that_left_the_game`. Imaging
   used `id()` the same way; it now holds a weak reference.
9. **The patch electrode stayed on the brain after leaving Lab > Patch,** holding its current on that neuron while the game ran,
   with nothing on screen. It now comes off once the page is not showing. `test_leaving_the_patch_page_takes_the_electrode_off_the_brain`.
10. **A patch row from another brain** (chosen, then the brain changed) gave the page's error screen (IndexError); now a note asks
    to choose again. `test_patch_page_survives_a_row_from_another_brain`.
11. **One bad thermogenetics target dropped every expression.** Add now resolves the target on the live brain and refuses a typo
    there; at tick time only the expression that can't resolve is dropped (and named).
    `test_a_bad_thermo_target_is_refused_and_the_good_expressions_stay`.

### Hostile input
12. **NaN currents reached the live brain:** `abs(nan) > 5` is False, so `ClampProtocol` and `set_hold` accepted NaN and that
    neuron's potential stayed NaN. Refused now. `test_clamp_protocol_refuses_non_finite_currents`, `test_a_nan_current_never_reaches_the_brain`.
13. **Protocol `check()` let ValueError / TypeError escape** (a schedule point `at_s: "x"` or `c: null`), which bundle reruns and
    other callers catching ProtocolError don't catch. Only ProtocolError comes out now. Also: at most 64 expressions, 1,000 schedule
    points (100,000 used to be accepted), `at_s` within 0-3600, a target must be a text spec, imaging `f0_photons` / `dff_per_spike`
    positive and finite and `shot_noise` / `tiff` real booleans (`shot_noise: "no"` was `True`; a bad number failed mid-run in a
    worker), at most 600 ROIs. `test_hostile_day2_protocol_blocks_raise_protocol_error` (13 cases).
14. **A patch protocol silently ignored** `surgery`, `recordings`, `control` and a top-level `warmup_s` / `duration_s`; now refused
    with a reason. Its amplitudes written as text (`["0", "0.2"]`) passed the check and crashed the summary after the whole run;
    they are stored as numbers now. `test_a_patch_protocol_refuses_keys_it_would_silently_ignore`,
    `test_patch_protocol_amplitudes_written_as_strings_are_numbers_after_check` (runs it).
15. **`thermo_escape` assay options went straight into `escape_fly(**opts)`:** `brain: 1` crashed every worker, `temps: []` the
    summary. Allowed keys and ranges are checked. `test_thermo_escape_assay_options_are_checked` (7 cases).
16. **Classroom protocols accepted day 2 blocks and ignored them** (the refusal branch was unreachable). `test_a_classroom_protocol_refuses_day2_blocks_too`.
17. **Thermogenetic `strength` NaN became 3 silently** (`min(3, nan)`), -1 became 0. Must be finite, 0-3.
18. **`ImagingSession` accepted inf / NaN** photon budget, dF/F per spike, cap and baseline time constant (NaN everywhere after).
    Imaging shorter than one frame raised a bare numpy reshape error; it now says why. `rois_from_specs` resolved 100,000 specs
    before refusing them.
19. **`Fly.drug` with a refused dose poisoned every later call** (the bad dose was stored before the check).
20. **`Wiring.from_dict` crashed on a damaged save** (`nt_scales` as a list, a text factor) or put NaN into every synapse of a
    transmitter. Bad entries are left out.
21. **Kept imaging frames stopped matching imaging frames** once the 300-frame buffer was full (every render was kept), so the TIFF
    / NWB ImageSeries timing was wrong; frames of two sizes (panel and big view) broke the NWB export. Kept only when a new imaging
    frame exists, with its time (the ImageSeries now uses timestamps); a size change restarts them.

### The playthrough's `extra:neurodex` (pre-existing, also on clean release/3.0)
Two mechanisms, both measured. (a) `rig.frames` puts the bot's hand on the fly's head by default, so the fly smelled the tool
every frame (`scent` poke from `Game._scents`, 120 pokes in 2 s): never calm by the game's own rule, so a discovery in the "calm"
6 s was correctly tagged "play". (b) The check judged every discovery since the rig started, including dozens of earlier tool legs,
whose play discoveries are correctly tagged "play" (Sonnet's AN01B002). Run alone on a fresh rig it PASSed (0 calm discoveries);
after 3 s of pokes it FAILed (`SNch10`, tagged play, `steps - last_poke` = 0 at discovery). Fix: during the window the hand rests
over the brain panel (2D) or the player steps back (3D), and only discoveries made in that window are judged. The criterion text
is unchanged. `test_neurodex_check_judges_only_its_own_calm_window`, `test_neurodex_check_keeps_the_hand_off_the_fly_while_calm`
(360 scent pokes in the window before). This is a bot fix, not a game change: the Day 1 question (should calm discoveries count at
all?) stays open.

### Small
- `.gitignore` also matches `.venv` and `data` as symlinks (the pattern `.venv/` does not match a symlink, which is how day 1
  committed them). Note: checking out an old `release/3.0` commit still replaces them with the committed self-pointing links,
  and the merge's deletion of those links removes a worktree's own link; recreate it if that happens (it happened to me once).
- i18n catalogs synced (one Settings string changed; the stale entry removed).

## What I ran (on the merge commit b1ab364 unless noted)
| run | result |
|---|---|
| `--validate` before (clean release/3.0, 323f8f2) vs after (b1ab364), cpu, 5 workers | **identical**: 2,020 values, 0 differences; 21 tests, 12 PASS / 9 FAIL; 590 s and 597 s |
| `--headless --playthrough all --sim-backend cpu` (SDL offscreen) | 315 passed, 2 failed, 69 gated, 10 skipped (b1ab364). FAILs: `extra:neurodex` (second half of the same check bug; fixed, and the rerun passed: HANDOFF_3.0_DAY2_DECISIONS.md) and `game2d:adult:flypaper:bomb` (pre-existing and history-dependent: the same leg FAILs on run 0 on Sonnet's code too, 4.574 Hz vs 8.725 needed; diagnosis unfinished; it PASSED on the decisions branch's playthrough, cause still not understood) |
| full `pytest` (two processes on disjoint halves of the files) | NOT FINISHED here; run to completion afterwards on the decisions branch, see HANDOFF_3.0_DAY2_DECISIONS.md (all passed except the playthrough slice, whose cause was a game bug, fixed there) |
| targeted suites while fixing (day 2 files, wiring, nwb, config, playthrough extras, neurodex, i18n, selftest, savestate, sharecode, bundle, labpages, extras3, compat) | 235 + 61 + 143 passed, 1 skipped |
| `tests/test_day2_review.py` on Sonnet's 4745422 (clean worktree) | 60 failed, 4 passed (the guards above) |
| protocol fuzz: 40,000 random protocols with hostile day 2 blocks (seeded) | non-ProtocolError exceptions: 0 before and after (the at_s case was found by a targeted test); accepted 10,084 before vs 54 after |
| real 3D frames (offscreen GL, one process per scene) | imaging (panel, big view, two flies with focus switched), inspector PATCH, patch page live, thermo arena with TrpA1 in DNp01 (fly on the hot side: DNp01 x9.6, DODGE), thermo page, genetics, pharmacology, imaging page: looked at; bugs 1-4 found this way; re-rendered after the fix |
| real 2D frames (software surfaces) | all five Lab pages, live patch trace, steps, isolated unit, imaging view panel and big (blue-yellow), inspector; genetics page draw 5.4 ms median; live imaging export TIFF + NWB written |
| Chen et al. 2013 | main text (Europe PMC full text) and Supplementary Tables 1-4 (supplementary PDF) read |

## Not verified
- The live patch trace, imaging view and thermo arena in a **visible window** with real input (frames were offscreen, driven by
  script). No gamepad, no clipboard.
- GPU compute backends (no torch / numba here); the exe / AppImage builds; `nwbinspector` on the NWB files (read back with pynwb
  only, as day 2).
- `tools/make_screenshots.py` as a whole (see the one-context-per-process note above).
- The thermo_escape assay was not rerun (none of its code changed except the options check).
- Day 2's design questions are still yours: the TrpA1 step (not a dose-response), the 30 s F0 time constant, the imaging frame
  rate, and day 1's calm-discovery question. New one: the GCaMP6 kernels are now the paper's in-vivo single-spike numbers (179/550,
  45/142 ms), which makes GCaMP6s slower to rise than before; say if you prefer the main text's resolvability ranges instead.
