# Handoff: 3.0 day 2 decisions (branch `day2-decisions`, from `release/3.0` at 837514c)

You asked for the open Day 2 design questions to be decided for researchers and players alike, and for the unverified citations to
be checked. Nothing here changes a threshold, a weight or a time constant of the model. Tests of the results are in the table at the end.

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
- The F0 and frame-rate Settings apply on the next imaging session, like the other imaging settings; not tried in a visible window.
- Thermogenetic hysteresis (Pulver) is a documented gap, not built.
