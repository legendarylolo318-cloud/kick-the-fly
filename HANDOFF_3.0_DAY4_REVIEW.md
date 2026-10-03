# Handoff: 3.0 day 4 review (Opus, branch `opus/3.0-day4-review` from `sonnet/3.0-day4` at 898e20a, merged into `release/3.0`)

Adversarial review of the five Day 4 features. Real adult pack (166,700 neurons), Python 3.14.7, this machine, CPU backend. Nothing
installed system-wide; **networkx 3.7 was pip-installed into `.venv` only** for the one-off cross-check (the game does not import it). No GPU
or driver change, no sudo. No network or microphone use.

Housekeeping: four orphaned worker processes from Sonnet's stopped T-maze run (parent PID 1, folder deleted, 3.8 GB) were still running; I
killed them. My first T-maze launch ran from the working tree (a spawned worker would have imported my later edits), so I restarted it from a
frozen worktree of 898e20a.

## Results

| run | result |
|---|---|
| `--validate` release/3.0 (worktree, data symlinked) vs this branch's final code, cpu, 4 workers | **identical**: 2,027 values compared, only `created`/`seconds` differ; 21 tests, **12 PASS / 9 FAIL in both** |
| **Race R4** (new, pre-registered in `lab/racing.py`, committed 5c0abd2 before any held-out lane; seeds 1000-1047, 8 races x 6) | **PASS**: within-race rank repeatability r_w **subtle +0.936** (95% CI +0.901 to +0.960), permutation **p < 0.001**; **control off -0.061** (95% CI -0.311 to +0.181), p = 0.658; subtle - off **+0.997, 95% CI +0.752 to +1.249**. Files: `docs/results/day4/race_r4/` |
| Race diagnosis (exploration seeds 0-15, individuality off) | run 1 vs run 2 rho: as the assay **+0.98**; fresh warm-up per run +0.55 (p = 0.027); common gain +0.94; **common calm baselines -0.08**. The post-warm-up calm baseline of the walking group vs finishing time: rho +0.98 |
| **T-maze sensitivity, all 26 cells** (5 Sonnet's + 21 resumed; 10,916 s, 6 workers) | baseline PASS (PI 1.00). FAIL at noise 0.1 (PI -0.04), tonic drive 0.225 (0.29) and 0.25 (0.19), target rate 10 Hz (-0.04); PASS at every other value (sensory gain, gain adaptation and synapse threshold: all pass). `docs/sensitivity.md`, `docs/results/day4/sensitivity_tmaze/` |
| Network science, adult and larva, analysis version 2 | adult rich club rho **1.02, 1.08, 1.06, 0.97, 0.86, 0.70** (v1: 1.04, 1.15, 1.08, 1.02, 0.92, 0.73): still no rich club; larva 1.07-1.36 (mild, as before); the all-mutual motif's enrichment is undefined (null 0 in the sampled wedges); motifs, reciprocity and communities unchanged. `docs/results/day4/netsci_v2/` |
| networkx cross-check (`tools/netsci_crosscheck.py`) | larva whole: motif estimates vs exact triad census, largest error 2.8 SE (13 classes); estimator on a null graph matches its exact census; phi(k) identical; modularity of our partition identical (0.533384); NMI identical to a from-the-definition implementation; networkx's Louvain Q 0.548 vs ours 0.533. Adult induced subgraph of 4,000 neurons: largest error 2.24 SE, phi/modularity/NMI identical, networkx Louvain 0.8587 vs ours 0.8576. The directed null changed 2,632 of 2,952 larva skeleton degrees (bug 2) |
| Real window (Wayland), 3D game, scripted gamepad through the game's own `pad_tick -> arcade_ui.pad_nav` | a real 4-fly bracket (pad: size, favorite, run; champion 2002, coin toss none) and a real 3-lane race (pad: switch tab, lanes, look at the field, bet, start; order 3002 > 3000 > 3001, the 10-point bet on 3000 at x2.1 lost: 100 -> 90); the three Lab pages; every palette x both text sizes; reduced flashing; B closes the page. The window manager tiled the window to 448x906, so the layout pass was redone offscreen at 1280x760 with the same script (bug 10) |
| tests | __TESTS__ |
| `--selftest` | __SELFTEST__ |
| `--headless --playthrough all --sim-backend cpu` | __PT__ |
| regression tests on the original code | `tests/test_day4_review.py` on 898e20a: 41 failed, 4 passed (the 4: wallet cases the old code handled: NaN, "12", [3], NaN odds) |

## Bugs found and fixed (regression tests in `tests/test_day4_review.py`; 41 of its 45 original tests fail on 898e20a, the other 4 are
## wallet edge cases the old code already handled, kept as coverage)

**Science / honesty**
1. **Personality card (decision A).** The card shown for every fly in play and for the pet was numbers *drawn from a random generator*
   seeded by the fly's seed; the function's docstring said they were "measured from the fly's own neuron gain distributions" and that the
   cut-offs were "published". Neither was true. **The pet's card was drawn from `pet.seed`, which is not the seed its brain is built from**,
   so it described a different fly. Fixed by decision A (below).
2. **Rich club null.** The rich-club coefficient is defined on the undirected skeleton's degrees, but its null was the skeleton of the
   *directed* degree-preserving null, which splits reciprocal pairs: on the larva it changed the skeleton degree of **2,632 of 2,952
   neurons**. The rich club now has its own undirected degree-preserving null (`netsci.rewire_undirected`); analysis version 2 (old caches
   are recomputed). The adult numbers changed: see Results.
3. **Undefined motif enrichment shown as `nan`.** An enrichment over a null mean of 0 is now `undefined` with its reason (on screen, in the
   summary and the CSV `note` column); the JSON has no NaN for it.
4. **`validation.run(params=...)` recorded the default parameters** in `lab_params` whatever it ran with; now it records what ran (a plain
   run records exactly what it always did).
5. **`params` silently ignored** by `mb_extinction`, `mb_second_order`, `epg_compass`, `epg_compass_wind` (their flies are built without
   them), so a sensitivity cell on one of them would have reported the baseline as the changed result. Now refused (validation and
   sensitivity); none of them is in the validated set, so no reported result changes.

**Lab / UI**
6. **The three Day 4 Lab pages share one job slot and the open page took any finished job**: a sleep result collected on the network-science
   page raised `KeyError: 'brain'` (and on the sensitivity page `KeyError: 'cells'`). Jobs are now collected by their own label.
7. **Cancel did not cancel**: the sensitivity page's Cancel did nothing (the run took no cancel flag); sleep deprivation and the race field
   measurement raised inside a `with ProcessPoolExecutor` block, whose exit waited for every queued fly; the tournament checked once per
   round (8 duels). Now every pool is shut down with `cancel_futures=True` and checked as results arrive; `validation.run` and
   `sensitivity.run` take `cancel` (default `None`, no change).

**Points**
8. A hand-edited `arcade_points.json` loaded a **negative balance** (`-500`), **crashed the arcade page** on `Infinity` (OverflowError) or a
   JSON list (AttributeError). Clamped and topped up on load. `settle` now refuses odds that are not a finite payout >= 1.1 and stakes
   that are not whole positive numbers.

**Test suite memory**
9. **The leak**: every `Game` starts a brain-view thread that holds the game, and through it every brain, until `view_stop`. `test_decoy`
   and `test_tutorial` stopped their own games; `test_swarm`, `test_drop_item` and the playthrough tests did not. `Game.LIVE` (a weak set)
   + `Game.shutdown()`, and a module-scoped conftest fixture that shuts down every live game after each test file. Measured (`KTF_MEM_REPORT`, the same three files in one process, real pack): resident after
   `test_swarm` **2,813 MB on 898e20a -> 1,357 MB**; after `test_swarm` + `test_drop_item` + `test_decoy` 2,050 -> 1,505 MB (the rest is
   the per-process pack cache, `simcore.pack`, which is meant to stay); the three files took 417 s -> 209 s (partly machine load). The memory
   report itself now frees the file's games before it measures (it measured before the cleanup ran at first).
10. **Arcade page at larger text**: the intro, tag, odds and events lines ran off the panel and the favorite buttons clipped their labels.
    They wrap now and the buttons are sized from the font. Found in the real-window session; checked by eye (no unit test: layout).

## Decision A: the in-game personality card

**The game shows only measured cards.** A fly in play and the pet show "Title (measured)" or **"card not measured"** (no trait words,
no numbers). Why: a researcher reading a card must be able to trust it describes that brain; a seed draw describes nothing, so labelling it
"derived" would still invite reading it as a trait. Measured over derived, always.
- **How measured:** `lab/tournament.measure_card` (the tournament's method: looming latency to the dodge threshold, sugar -> MN9 ratio,
  DNa01/02 R/L), now with the fly's own individuality sigma and LIF parameters. Esc > Fly arcade > **Measure the flies in play** (worker
  processes, ~10 s per fly); `Fly.card()`; every tournament and race adds its cards.
- **Cost / caching:** `<data>/cache/personality_cards.json`, keyed by seed, individuality mode and sigma, the five LIF parameters, the pack
  SHA-256 and `CARD_VERSION`. measure_card is deterministic for that key, so a card never goes stale and never needs migrating; a changed
  key is just "not measured". A gl-backend fly (no individuality gains) looks up the `off` card, which is what it runs.
- **Saves:** nothing new is stored (the key is already in them), so old saves need no migration.
- **Pet files:** the pet shows the card of the pet fly (built from Settings > Brain > Random seed), never `pet.seed`. A measured card is
  written into the pet file with its key and restored to the cache from it; a **pre-review pet file's drawn card is kept as
  `legacy_personality_card`** (never deleted, never shown).
- **Individuality off:** a measured card says its differences are noise and warm-up state, not individuality (B explains why).
- Docs: main docstring, Lab > Model assumptions (new card), `docs/individuality.md`, `docs/pet.md`, `docs/tournament.md`.

## Decision B: why the race's `off` control is repeatable, and R4

**Why.** Both runs of a fly start from the same post-warm-up state of its seed (the brain warms up 600 steps with the seed's noise, then
the lane noise is reseeded). The speed rule (GAME RULE) reads DNp09 firing **as a multiple of the fly's calm baseline**, and that baseline is
estimated during the 3 s warm-up and carried through the race. A fly whose warm-up set a high baseline reads a low walking level all race,
in both runs, whatever its individuality. Measured on exploration seeds (table above): equalising the baselines removes the repeatability
(-0.08), equalising the slow synaptic gain does not (+0.94), and the baseline alone predicts finishing time (rho +0.98). A fresh warm-up per
run leaves +0.55 (p = 0.027, n = 16): chance or a remaining seed effect, **not resolved**. R1-R3 and their reported results are untouched.

**R4** (decision: hold the brain state fixed, vary only the individual; within-race statistics so the shared state cannot contribute;
both a permutation test and a CI against the off control, written and committed before the run): **PASS**. With the state artefact held
fixed, a fly's within-race rank is highly repeatable with individuality on (+0.94) and not at all with it off (-0.06). So individuality *does*
persist in race speed in this model; R1 just could not show it. Reported as a MODEL PREDICTION about this model's per-neuron-gain rule.
New: `simcore.new_brain(individuality_seed=...)` / `LIFParams.individuality_seed` (default None: the gains come from the seed, as before),
`--headless --race-r4`, `docs/racing.md`.

**Recommendation (not done, needs your call):** the speed rule's per-seed baseline makes race speed mostly a warm-up artefact. A fixed
baseline window, or measuring the calm rate over a longer standardised calm period, would make R1 meaningful. That changes a game rule, so I
did not do it.

## Not verified / known limits

- **No physical gamepad**: the pad was a scripted stand-in for the device (the game's own Gamepad class with `poll()` scripted); the code path
  from the pad's button set onward is the real one.
- GPU backends: nothing run. The gl refusal (`tournament.check_individual`) is covered by its unit test with a fake sim, not on a live gl context.
- The duel's side/heading signs were checked by reading (they match the 3D duel's convention), not by a probe of the real brain's LC10 ->
  DNa02 lateralisation. 0 knockouts: no geometry, hit-radius, cooldown or damage bug found (hits are rare because DNp35 rarely crosses its
  threshold: 3-9 shots per fly per 20 s, 38% landing); I changed no blaster or arena number.
- The 3.0 s / 2.0 s sleep redesign: Sonnet's scratchpad logs show one held-out run (16:52, already 3 s / 2 s) and no earlier one; the dFB
  current probe on held-out seed 1001 (disclosed by Sonnet) measured the response, not a criterion. S1/S3 constants are the pet's rule and the
  exploration-seed probe; I found nothing tuned. S3's direction is also expected from the rule (now said in the docs).
- Race diagnosis condition B (+0.55) is not explained.
- The card cache is per machine; a card measured on one backend is shown on another (cpu/numba/torch-cpu are bit-exact; GPU would differ slightly).
- networkx 3.7 remains installed in `.venv` (only `tools/netsci_crosscheck.py` imports it).
