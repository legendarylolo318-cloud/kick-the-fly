# Handoff: 3.0 day 4 (written by Sonnet on branch `sonnet/3.0-day4`, from `release/3.0` at 37bf60e, for Opus to test and merge)

Local commits only, nothing pushed. 47 files, +6,400 lines. Real adult pack (166,700 neurons), Python 3.14.7, this machine, CPU backend. Nothing installed,
no system or driver change, no sudo. The only GPU use was the existing `--selftest`'s OpenGL probe (no error). No network, no microphone.

## Read this first

1. **Validation is unchanged.** `--headless --validate --sim-backend cpu --workers 6`, `release/3.0` (worktree) vs this branch: **2,027 values compared, only
   `created`/`seconds` differ**; 21 tests, 12 PASS / 9 FAIL both. `validation.py` gained optional `params` and `only` arguments (default `None` = old behavior);
   `assays.py` and `kickthefly/sim/` are untouched.
2. **Tests:** the day-4 tests (`tests/test_day4_*.py`, `test_compat_day4.py`) = 125 passed. The rest of the fast suite via the new `tools/run_tests.py --fast`,
   6 chunks one at a time: **1,135 passed, 23 skipped, 0 failed** (58 min with 4-7 other jobs running; on a quiet machine about a third of that). **I did not
   rerun the 22 validation tests** (the `--validate` diff above covers the same simulations; `tools/run_tests.py --only-validation` runs them).
3. **`--selftest`: 22 passed, 0 failed.** **`--headless --playthrough all --sim-backend cpu`: 395 passed, 0 failed, 103 gated, 12 skipped**; the seven new checks
   (`extra:tournament`, `race`, `netsci`, `sleepdep`, `sensitivity`, `day4-pages`) all PASS. The `--sensitivity` smoke run (1 parameter, 2 values, 2 behaviors, 3 seeds) ran; its
   playthrough twin passes.
4. **Two findings that change how the results read** (details below): (a) the race assay's individuality control is as repeatable as the individuals
   (rho 0.92 vs 0.96): what persists is each seed's warm-up brain state, not the individuality gains; (b) the game's ordinary personality card is derived from
   the seed, not measured; the tournament and race measure it.
5. **Sensitivity: the 11 pathway behaviors are complete over the full grid; the T-maze (mb_conditioning) run is only 5 of 26 cells** (I stopped it). See below.

## What was built

| # | feature | where | tags |
|---|---|---|---|
| 1 | **Fly tournament** (4/8/16, favorite, bracket view, match replay, champion's drivers, `--tournament N --seeds`) | `game/flyduel.py` (engine), `lab/tournament.py`, `ui/arcade_ui.py` (Esc > Fly arcade), headless, API `tournament`, `fly.card()`, `fly.duel()` | CONNECTOME steer/shoot/dodge/run + measured card; GAME RULE arena/blaster/pairing/tie-break; MODEL PREDICTION winner, drivers |
| 2 | **Fly racing** (lures, odds from cards, points-only bets, replay, `--race`) | `game/flyrace.py`, `lab/racing.py`, `core/points.py` | CONNECTOME DNp09 speed + lure pokes; GAME RULE track/odds/points; MODEL PREDICTION order |
| 3 | **Network science** (Lab, `--netsci`, adult + larva, CSV, cached with checksum, background job) | `lab/netsci.py`, `lab/labday4.py` | CONNECTOME (analysis choices GAME RULE) |
| 4 | **Sleep deprivation assay** (Lab, `--sleep-deprivation`, protocol assay kind, paired) | `lab/sleepdep.py`, labjobs, protocol | GAME RULE pressure, CONNECTOME dFB readout, MODEL PREDICTION rebound |
| 5 | **Sensitivity analysis** (Lab, `--sensitivity`, resumable, workers, CSV/JSON/SVG) | `lab/sensitivity.py`, `validation.py` (params/only) | MODEL PREDICTION cells; analysis only |

Also: pause-menu "Fly arcade" (buttons now 40 px to fit 11), Lab hub entries, five Lab > Model assumptions cards, main docstring section, 104 localization strings,
five docs pages + README/CHANGELOG/api/lab/playthrough/CONTRIBUTING, example protocol `protocols/sleep_deprivation_assay.yaml`, `--selftest` `day4`, gamepad
for the arcade (`arcade_ui.pad_nav`), `ui/bgjob.py` (thread + progress + cancel). **Not a loadout tool:** tournament and racing are modes, so the loadout editor is unchanged
(guarded by a compat test). No new Settings keys, so no config migration; `arcade_points.json` is a new file nothing else reads.

## Results (all seeds 1000+; designs chosen on exploration seeds 3-4, 11-13; criteria written in the modules before each run)

- **Sleep deprivation** (10 seeds): S1 rebound **PASS** (+13.3 s, p < 0.001), S2 **PASS** (0.0 s slept in the window; by construction, reworded as a manipulation check),
  S3 dFB readout **PASS** (4.51x vs 2.11x calm, p < 0.001). The rebound is expected from the game-rule pressure; the docs and output say so.
- **Tournament** (64 flies, 4 brackets of 16, 60 matches): **personality does not predict winning** (3 traits, Holm p = 1.0 / 1.0 / 1.0; control `off`: 0.84 / 0.84 / 0.56).
  0 knockouts in 60 matches (every match went to the time limit): the duel rules make hits rare.
- **Race** (48 flies, 8 races, each run twice): R1 repeatability rho 0.96, R2 form rho -0.54, R3 odds beat uniform p = 0.008, all Holm-significant. **Control `off`: R1 0.92, R2 -0.53, R3 p = 0.039.**
  So these do not show that individuality predicts the finish; see Decisions. The calm-walking-drive term of the form score is the speed rule's own variable (rho -0.79); the sugar ratio carries nothing (-0.09).
- **Network science, adult** (3 nulls, 300k wedges, 300 s, 2.4 GB): reciprocity 0.183 vs null 0.002; modularity Q 0.700, 653 communities, NMI with regions 0.53; one-way motifs depleted, reciprocal ones strongly
  enriched; **no rich club** (rho 1.04, 1.15, 1.08, 1.02, 0.92, 0.73). Larva: Q 0.533, 12 communities. Full tables in `docs/network-science.md`.
- **Sensitivity, 11 pathway behaviors, 6 parameters, n = 10** (5,140 s, 4 workers): baseline equals validation exactly; 23 of 66 parameter x behavior pairs fail at some tested value; most fragile to tonic drive above
  default, noise at 2x+, and target rate; the sensory gain and the synapse threshold barely matter (bitter -> DNg28 is the fragile one). T-maze: baseline PASS (PI 1.00), noise 0.025 and 0.0375 PASS, 0.075 PASS (PI 0.53, borderline), 0.1 FAIL (PI -0.04);
  the other 21 cells did not run. Resume: `python kick_the_fly.py --headless --sensitivity --sens-tests mb_conditioning --workers 6 --resume --out DIR` after copying `docs/results/day4/sensitivity_tmaze_partial/sensitivity_progress.json` into DIR (about 8 min per cell).
- CSV/SVG outputs are in `docs/results/day4/` (tournament JSON with replay frames not committed: 634 s per 16-fly bracket to regenerate).

## Decisions for you

1. **Race R1 and "individuality".** Pre-registered, passes, but the `off` control passes equally. I kept the criteria and reported it. Do you want a stricter pre-registered test (e.g. repeatability beyond the `off` control, with its CI)? I did not add one after seeing the data.
2. **The game's personality card is seed-derived, not measured** (`core/individuality.compute_personality_card` with no metrics draws numbers from the seed). Pet mode and in-game cards show these. I left it alone and used measured cards in the new modes. Should the in-game card be measured?
3. **Sleep S2 design change** (before any held-out run): the first disturbance (every 4 s, hold 2 s) let a fly sleep ~1 s per cycle, so S2 could not pass; changed to 3 s / 2 s and reworded S2 as a manipulation check. No rate/current/threshold was tuned for S1/S3. The dFB current scale (0.08) was chosen from exploration seeds 11 and 12 (I first probed on 1001/1002 by mistake, then redid it on 11/12 before using it).
4. Duel has 0 knockouts: blaster/arena numbers are the 3D duel's own; I did not change them to make fights decisive.

## Not verified / known issues

- GPU backends: nothing run (tournament refuses the gl backend, which lacks individuality; untested live). Motif counts, Louvain and the null are my own implementations, not compared with networkx/igraph (not installed).
- The arcade page was only drawn (screenshots + tests); I never played a bracket in a live window or on a gamepad. Brain-driven 3D visuals of the duel do not exist (the replay is a top-down 2D drawing).
- Memory finding (test suite): `--mem-report` shows files that build whole games keep 1-1.8 GB resident after the file (`test_swarm`, `tests/playthrough`, `test_tutorial`, `test_drop_item`, `test_decoy`; chunk peaks 2.4-5.4 GB). Not fixed; worth a `gc`/teardown pass. `tools/run_tests.py` + `tests/.durations.json` are the day-3 follow-up for test time.
- Motif enrichment for the all-mutual triad is `nan` (the null never produced one); the page shows nan.
- The in-game netsci job uses a thread (numpy releases the GIL mostly); adult takes 5 minutes and 2.4 GB on top of the game.
- A tournament match costs ~40 s (20 s duel, 2 brains); a 16-fly bracket ~10 min with 3 workers plus card measurement.
