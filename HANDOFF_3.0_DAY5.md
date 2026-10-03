# Handoff: 3.0 day 5 (written by Sonnet on branch `sonnet/3.0-day5`, from `release/3.0` at fd0d20c, for Opus to test and merge)

Local commits only, nothing pushed. Real adult pack (166,700 neurons), Python 3.14.7, this machine (new), CPU backend. No system package, driver or kernel
setting was touched, no sudo; pip only inside `.venv`. The only GPU use was `--selftest`'s existing OpenGL probe (passed). The only network use was the
documented connectome download for setup, plus reading papers' abstracts and open-access pages through Europe PMC / PMC (see "Papers").

## Read this first

1. **Validation is unchanged.** `--headless --validate --sim-backend cpu --workers 4`, `release/3.0` baseline (worktree, this machine) against this branch's final
   code: **2,025 values compared, 0 differences** (`created`/`seconds` ignored), 21 tests, **12 PASS / 9 FAIL in both**, same pass/fail per test
   (`docs/results/day5/validate_before.json`, `validate_after.json`). No file validation reads was edited (`validation.py`, `assays.py`, `sim/` untouched).
2. **Rig results (seeds 1000-1009, criteria committed in f4b5f5c before any held-out run): three PASS, one FAIL, with the `off` control agreeing.**
   Tethered (T1 +8.58 Hz, T2 0.43) PASS; ball (BL1 0.50, BL2 +88.8 deg) PASS; Buridan (B1 +39.2 deg, B2 +11.1 transits) PASS;
   **four-field olfactory arena F1 FAIL** (PI +0.005, p = 0.69; no preference, as predicted from validation.md). Full table in `docs/rigs.md`.
3. **Tests:** fast suite (`tools/run_tests.py --fast --chunks 6 --parallel 3`) on the final code: all failures are baseline ones, see below. `--selftest`:
   **23 passed, 1 warning (pynwb missing), 0 failed.** `--headless --playthrough all --sim-backend cpu`: **344 passed, 0 failed, 4 gated, 13 skipped** (adult only:
   there is no larva pack here); the three new checks (`extra:rigs`, `extra:minipapers`, `extra:day5-pages`) PASS.
4. **An environment change you should know about:** pygame 2.6.1 was built from source here with no SDL2_ttf / SDL2_mixer (no Python 3.14 wheel), so fonts and
   sound did not work and about 47 baseline tests failed. You chose `pygame-ce` in `.venv` only; that is what everything below ran on. `requirements.txt` is unchanged.
5. **A citation bug found and fixed:** `docs/validation.md` cited Ohyama et al. 2015 as doi:10.1038/nature14424, which is an unrelated paper (Carmi et al. 2015). Right DOI:
   10.1038/nature14297.

## Environment setup report (each step)

| step | result |
|---|---|
| Python | system python3 3.14.7 (>= 3.11); `.venv` made; `pip install -r requirements.txt pytest` into it |
| `data/` | not present, not copied: built with the documented `python -m kickthefly.sim.connectome.loader build` (network, 1.1 GB, 13 s) then `python -m kickthefly.sim.brainpack build` |
| pack hash | `data/kick_brain.npz` SHA-256 **a65baf8602b48c62806b63c728d95dc2fc0146ced643e41044078fb6deb87fc0**: starts a65baf8602b4, as days 1-4 |
| baseline validation | `release/3.0` in a worktree: 21 tests, **12 PASS / 9 FAIL**, 629 s (`docs/results/day5/validate_before.json`) |
| baseline selftest | first with pygame 2.6.1: 21 passed, 2 warnings (no audio mixer, pynwb). After pygame-ce: **22 passed, 1 warning (pynwb)**. GPU/OpenGL: the `gl` backend PASS on an AMD Radeon RX 9070 XT (OpenGL 4.6, Mesa 26.2.4), the OpenGL 3.3 check PASS; nothing to fix, nothing touched |
| baseline fast tests | with the broken pygame: 47+ failures. After pygame-ce, 6 chunks: **1,183 passed, 5 failed**, 10.8 min: `test_larva_connectome_loader_and_counts` and `test_larva_validation_execution` and `test_individuality_forced_off_in_validation` (all need the larva pack, which needs a separate download; not built), `test_playthrough_extras::test_neurodex_extra_passes` (passes alone, fails in its chunk: `IndexError: index 2932 ... size 2346`, an order-dependent leak that predates this branch; not investigated) and `test_morphology::test_brainview_with_skeletons` (a stale assertion that only passed on a machine with a cached skeleton; fixed, below) |
| git | no identity was set on this machine: set **repo-locally** to `Claude <noreply@anthropic.com>` (the author of earlier commits) |

After my changes, the fast suite shows exactly the baseline's remaining 4 failures (3 larva-pack ones and `test_neurodex_extra_passes`); `test_morphology` now passes. (Two
of my own changes made two existing first-run tests fail in the first final run; I fixed them and reran them on their own: 7 passed. I did not rerun the whole suite a
second time after that fix.) I did not run the 22 validation tests (`--only-validation`, ~18 min): the `--validate` diff above runs the same simulations.

## What was built

| # | feature | where | tags |
|---|---|---|---|
| 1 | **Behavior rigs**: tethered flight simulator (open/closed loop), fly on a ball (bar and panorama VR), Buridan's paradigm, four-field olfactory arena; each its own Lab scene; headless `--rig`, `--rig-assay`, `protocols/rig_*.yaml` (a new stand-alone `rig:` protocol kind); exports in the recorder's format + trace CSV; pre-registered assays with an `off` control | `game/rigs.py`, `lab/rigassay.py`, `lab/labrigs.py`, `lab/protocol.py`, `docs/rigs.md` | CONNECTOME yaw (DNa01/02 R - L) and walking (DNp09); GAME RULE yaw gain, EMD stage, LC10 tracking rule, platform, edge, ball, VR, arena, measures; MODEL PREDICTION the response |
| 2 | **Guided mini-papers**: von Reyn 2014, Tully & Quinn 1985, Shiu 2024, Hampel 2015, Ohyama 2015, Colomb 2012 (Buridan); hypothesis, run (quick 4 flies on exploration seeds, or full on the validation seeds), plot, comparison with what the paper states; built on `classroom.py`'s lecture protocols (`EXTRA_LECTURES`; the five curated lectures still five); headless `--minipaper` | `lab/minipapers.py`, `ui/minipaper_ui.py`, pause menu entry, Lab hub, `docs/minipapers.md` | LITERATURE next to MODEL PREDICTION |
| 3 | **Release polish**: What's New in 3.0 (once, skippable, migrated config key `[first_run] whatsnew_3_0_seen`; fresh installs don't get it; Settings > Help reopens it); README top section; `docs/changelog.md` (new, consolidated) and `CHANGELOG.md` (day 3 and the day 1-3 reviews were missing: added); two curated Neurodex facts (sugar and bitter SEL neurons, Yao & Scott 2022) | `ui/whatsnew_ui.py`, `core/config.py`, `kick_the_fly.py` (`show_first_run_notices`), docs | interface; LITERATURE |

Also: Lab > Model assumptions (two cards), main docstring (three paragraphs), 83 localization strings, `--selftest` `day5`, three playthrough checks, Python API
`rig`/`rig_assay`/`minipaper`, `docs/lab.md`, `api.md`, `playthrough.md`, `validation.md`, `neurodex.md`, `CONTRIBUTING.md`. **Not a loadout tool:** rigs and mini-papers are
scenes and pages, so the loadout editor is unchanged (guarded by a compat test). Gamepad: `labrigs.pad_nav` and `minipaper_ui.pad_nav` (hooked next to the arcade's in `kick3d`).
Larger text and a 448 px window: one scrolling column, `Menu.wrapped`, buttons sized from the font; looked at by eye in screenshots (1280x760 and larger text) and drawn in
every palette by the tests.

## Rig design, exploration and results (details in docs/rigs.md)

- **Exploration (seeds 0-9 only), disclosed:** the optomotor probe (seed 0) and an LC10 lateralization probe (seeds 0, 1: right-side drive moves R - L by +8.6 to +9.7 Hz, left by
  -8.1 to -10.2: the duel's rule steers the right way, which no validation test checks), a first ball run where the bar was a near landmark and the fly walked past it (made a distant
  landmark before any assay), and ten-fly runs of all four rigs from which the minimum effects were set well below what was seen. The criteria were then written into
  `rigassay.py` and committed (f4b5f5c) **before** the held-out run, which ran from a frozen worktree of that commit. Only one line of `rigassay.py` changed afterwards
  (`scene_run` returns `dt`), outside the assay path.
- **Predicted from validation.md before the runs:** T1, T2, BL1 pass (optomotor passes); BL2, B1, B2 pass given the unvalidated LC10 rule; F1 fails (nothing links an odor to steering).
  The E-PG compass failing means no rig asks for a heading memory.
- **Honest reading:** bar and stripe fixation are far tighter (about 3 degrees) than real flies' (Colomb et al. give a chance level of 45 deg and fixation/antifixation), because the
  LC10 rule is strong and deterministic; the transits (B2) are largely produced by the edge reflection and the "stripe nearer the heading" rule. The olfactory arena misses and
  nothing was tuned. The tethered response saturates early by the EMD stage's design, so the rig cannot show a speed tuning curve. The `off` control gives the same pass/fail everywhere.

## Papers: what I read, and how

| paper | read |
|---|---|
| Colomb et al. 2012 (PLoS ONE, open access) | full text page, through WebFetch's summarizing model (not line by line) |
| Hampel et al. 2015 (eLife, PMC4599031) | full text page, same way, plus the abstract |
| von Reyn 2014, Shiu 2024, Ohyama 2015, Tully & Quinn 1985 | **abstracts only** (Europe PMC / PubMed records); Nature and J Comp Physiol full texts are paywalled |
| Maisak 2013 | abstract only (used only to check the T4/T5 fact's citation; not a mini-paper) |
| Yao & Scott 2022 (Neuron) | abstract plus the Results passage that defines "sugar-SELs" / "bitter-SELs" (PMC8930643), through the summarizer |

DOIs checked through Europe PMC: von Reyn, Shiu, Hampel, Ohyama (and its correction), Tully & Quinn (10.1007/BF01350033). No number is quoted that the part I read does not state
(von Reyn, Shiu, Hampel, Ohyama have none in what I read; Tully & Quinn: 95% of trained flies avoided the shock-associated odor; Colomb: 45 degrees chance, about 33% of the time active).
The conversion of the sim's PI to a share of avoiding choices, (1 + PI) / 2, is derived here and labelled so.

## Things I found while building (fixed, with tests)

1. The quick Buridan mini-paper said "does not reproduce" because a 4-fly run cannot reach p < 0.01: now judged on the minimum effects with the reason stated (regression test).
2. A rig protocol with a list as `name` raised TypeError (unhashable); refused now. API `rig(seconds=1e9)` would have run forever: ranges are checked.
3. The pause menu's button sizing crashed when a Menu was built before its fonts; guarded.
4. A rig/mini-paper job's result is collected by its own rig or paper, not by whichever page is open; Cancel is checked each tick and as results arrive, pools use `cancel_futures=True`.
5. Two existing first-run tests (`test_extras3`, `test_day3_review`) now set the new What's New flag (it comes before the neuPrint question). `test_morphology`'s stale assertion accepts the opt-in message.

## Decisions for you

1. **pygame-ce** is a venv-only workaround. If you would rather have the stock package, install `sdl2_ttf` and `sdl2_mixer` (pacman) and run `pip install --force-reinstall --no-binary pygame pygame` in `.venv`; I did not change `requirements.txt`.
2. **Larva pack:** I did not build it (it needs a separate download, outside the documented adult setup). Three baseline tests and the Ohyama mini-paper's live path need it; the mini-paper shows the recorded validation numbers, marked as recorded. Say if I should build it.
3. **Version:** nothing bumped or tagged (the app still says 2.13.1); the docs say 3.0 is unreleased.
4. The rigs' LC10 tracking rule is very strong; making it noisier would be a game-rule change I did not make.

## What I did NOT verify

- No physical gamepad (the pad paths are unit-tested through `pad_nav` with scripted button sets) and no real window watched by a person: the pages were drawn onto surfaces (tests, the playthrough's real `Game`, screenshots I looked at), and the What's New flow was driven through `Game.show_first_run_notices`, not a launched window.
- The Ohyama mini-paper live path (no larva pack), and the **Full** (10-fly) mini-paper runs: I ran only the quick runs of five of them (`docs/results/day5/minipapers/`); the T-maze quick run took 111 s with 4 workers.
- GPU backends: nothing run; every number is the CPU reference.
- The 22 validation tests (`--only-validation`) and a second full fast-suite run after the last two test fixes.
- `extra:rigs` takes about 53 s and the playthrough about 10-15 min on this machine; the playthrough ran alone and passed, the fast suite ran next to the validation run (load), nothing flaked.
- That the real LC10 -> DNa lateralization holds beyond seeds 0 and 1 (the assay results are the evidence, on 10 held-out flies).
- Papers: see the table; the paywalled four are abstract-level.

Files (63): see `git diff --stat fd0d20c..HEAD`. Results: `docs/results/day5/` (rig assays JSON/CSV/logs, validation before/after, mini-paper outputs, playthrough report).
