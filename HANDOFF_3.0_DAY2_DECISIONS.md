# Handoff: 3.0 day 2 decisions (branch `day2-decisions`, from `release/3.0` at 837514c)

You asked for the open Day 2 design questions to be decided for researchers and players alike, and for the unverified citations to
be checked. Nothing here changes a threshold, a weight or a time constant of the model. Results are in the last section.

## Decisions
| question | decision | why |
|---|---|---|
| Should "calm" Neurodex discoveries count? | **No.** A calm, untouched fly (the game's own test) discovers nothing. A driven fly still does, even if calm. Old `at rest` entries load, keep their tag and stay in the collection. | The remaining calm discoveries on the real pack were correlated spontaneous bursts of sensory types: the model's background activity, not something a player or a researcher did. For players an untouched fly was filling the collection; for researchers it would read as a finding. The review's criterion "a calm fly discovers nothing" FAILED before and now holds **by rule, not by tuning** (no threshold changed; the bot's check now expects zero). |
| TrpA1 step, not a dose-response | **Kept as is**, documented. | The effector is graded (linear 25 -> 29 C, a game rule). The step is DNp01's: two neurons at their refractory limit. Smoothing the ramp or the escape threshold to get a nicer curve would be tuning to a result. Docs point to a graded readout (express TrpA1 in a larger type, read its rate). |
| 30 s F0 time constant | **Default kept, now a setting** (`brain.imaging_f0_tau_s`: 10, 30, 60, 120 s) **and a protocol key** (`imaging.f0_tau_s`, any positive number); the API already had `baseline_tau_s`. | It is a game rule with no source, so it should be visible and changeable, not hidden. Tagged GAME RULE. |
| Imaging frame rate | **20 Hz default kept; 5 Hz added to Settings.** API / protocols still 1-200 Hz. | No published rate is claimed. 20 Hz (50 ms) puts about one frame on the fastest kernel's rise and about eleven on GCaMP6s's decay. |
| GCaMP6 kernels | **Kept** at 179 / 550 ms (6s) and 45 / 142 ms (6f), Chen 2013 Supplementary Table 3. | They are measured single-spike numbers; the main text's ranges are a different quantity (how fast a response must rise to resolve a spike). |

## Citations re-checked (3.0 Day 2 decisions; what I read and what I could not)
- **Pulver et al. 2009** (PMC2694103, open-access author manuscript): spiking onset about 25-26 C; junction-potential threshold 26 +- 0.2 C;
  no significant decay over 20 min of constant heating; the deactivation threshold is higher than the activation threshold
  (**hysteresis, not modelled**). Onset kept at 25 C.
- **Hamada et al. 2008** (PMC2730888): anterior cell neurons first respond at 24.9 +- 0.6 C (n = 10).
- **Kitamoto 2001**: the **abstract** confirms every number quoted (motionless within 2 min at 30 C; walking again in 1 min). The full
  text is not open access and **was not read**.
- **Zhang et al. 2023** (PMC10060165): jGCaMP8m half-rise 58 +- 6 ms and half-decay 137 +- 21 ms are for Drosophila L2 dendrites under
  0.5 Hz light-dark flashes (not single spikes); single-spike half-rise under 5 ms in mouse. Kernel unchanged.
- **Caveat on the method:** the Pulver, Hamada and Zhang numbers were extracted from the full texts by a small summarizing model through
  Europe PMC / PMC, not read by a person. The abstracts (Kitamoto, Zhang, Hamada) were read directly. Someone should confirm the
  four numbers above in the papers before they are quoted anywhere citable.

## Files changed
`core/neurodex.py` (the rule, docstring, tests), `core/config.py` (new setting, 5 Hz), `lab/livelab.py`, `game/kick_the_fly.py` (setting wiring,
docstring), `lab/protocol.py` (`f0_tau_s`), `lab/playthrough.py` (neurodex check expects zero calm discoveries), `lab/lab.py` (Model Assumptions),
`lab/imaging.py` and `lab/thermogenetics.py` (docstrings), docs (neurodex, imaging, thermogenetics), README, locale catalogs, `tools/i18n_sync.py`.
Tests: `tests/test_neurodex.py` (+2), `tests/test_protocols_day2.py` (+2), `tests/test_extras3.py` (2 assertions now say a calm fly discovers nothing).

## Known issues / not done
- Old English tips replaced in `config.py` leave their old entries in the locale catalogs (harmless; the sync tool never removes entries).
- The F0 and frame-rate Settings are read when the game starts (both now carry the `restart` flag so the UI says "applies on the next launch"; the frame rate was already read only at launch). Only the indicator can be changed live, in Lab > Calcium imaging. Not tried in a visible window. A live control for them would be a small follow-up if you want it.
- Thermogenetic hysteresis (Pulver) is a documented gap, not built.

## Two bugs found while verifying (both fixed, with regression tests)
1. **Neurodex discovery froze after a new fly or a loaded save (a game bug).** `Extras3.dex_tick` skips a check unless the brain has run
   `CHECK_STEPS` since the last one, but the brain's step counter goes back to 0 on a new fly or a loaded save, so no check ran until it
   caught up with the old count (reproduced in the playthrough: `_last_check` 5307, brain at 3282; driving DNp01 for 3 s discovered nothing).
   Fix: a counter that went back resets the check. `test_discovery_resumes_after_the_step_counter_goes_back` fails without the fix.
2. **The bot's neurodex calm window opened too early (a bot bug).** After the training check the last tool poke was under 2 s old, so
   types the fly legitimately smelled were discovered "in play" inside the "calm" window. The window now opens once the fly has been
   untouched for 2.5 s. The criterion is unchanged.

## What I ran (branch `day2-decisions`, 81811d4 and the handoff commit; full-speed, one machine, real adult pack)
| run | result |
|---|---|
| full `pytest`, two halves (before the two fixes above) | half 1: 383 passed, 19 skipped, **1 failed** (the playthrough slice: bug 1); half 2: 464 passed, 0 failed (run in three pieces: a memory guard stopped the first runs inside `test_validation`, which spawns six workers; nothing failed before that, and the last four files, `test_validation`, `test_video_record`, `test_wiring`, `test_workflows`, passed 51/51 on their own) |
| after the fixes: `tests/test_extras3.py`, `tests/playthrough/` | 38 passed; 13 passed (the slice that failed now passes) |
| `--validate` baseline (837514c) vs this branch, cpu, 6 workers | **identical: 2,020 values, 0 differences**; 21 tests, 12 PASS / 9 FAIL in both (the documented results) |
| `--headless --playthrough adult --sim-backend cpu` | **261 passed, 0 failed**, 3 gated, 12 skipped; `extra:neurodex` PASS, `game2d` and `game3d:adult:flypaper:bomb` both PASS |
| `--headless --playthrough larva --sim-backend cpu` | 55 passed, 0 failed, 67 gated |
| `tools/i18n_sync.py --check`, links in changed docs | clean |

## Not done / not verified
- The **full pytest was not rerun end to end after the two fixes**: only the files they touch (above). The fixes change `Extras3.dex_tick` and the bot's neurodex check only.
- `flypaper:bomb` passed this time in both 2D and 3D; the review saw it fail history-dependently (4.574 Hz vs 8.725 needed on one leg). I did not diagnose why it passes now, so treat it as not understood rather than fixed.
- Everything the day 2 review listed as unverified still is: live windows with real input and a gamepad, GPU compute backends, exe / AppImage builds, `nwbinspector`.
- The four paper numbers extracted by a summarizing model (see Citations) still need a human check.
