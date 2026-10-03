# Handoff: 3.0 day 4 review (Opus, branch `opus/3.0-day4-review` from `sonnet/3.0-day4` at 898e20a, merged into `release/3.0`)

Adversarial review of the five Day 4 features. Real adult pack (166,700 neurons), Python 3.14.7, this machine, CPU backend. Nothing
installed system-wide; **networkx 3.7 was pip-installed into `.venv` only** for the one-off cross-check (the game does not import it). No GPU
or driver change, no sudo. No network or microphone use.

Housekeeping: four orphaned worker processes from Sonnet's stopped T-maze run (parent PID 1, folder deleted, 3.8 GB) were still running; I
killed them. My first T-maze launch ran from the working tree (a spawned worker would have imported my later edits), so I restarted it from a
frozen worktree of 898e20a.

__RESULTS__

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
   + `Game.shutdown()`, and a module-scoped conftest fixture that shuts down every live game after each test file. __MEM__

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

__B__

## Not verified / known limits

__NOTVERIFIED__
