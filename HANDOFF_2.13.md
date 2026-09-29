# Handoff: 2.13 (written by Sonnet on branch `sonnet/2.12`; merged into main and renumbered by Opus on `opus/2.13`)

Written by Sonnet for Opus to test thoroughly and push. Everything below is what I did and measured; the last section is what I did **not** verify.

## Read this first

- **Base.** I branched from `claude/2.11` (94f6a18), not from `main`. It is a descendant of `main` and carries the five 2.11 review commits (startup and menu crash fixes, pet-save safety, docs) that `main` doesn't have. If you want it on bare `main`, rebase and expect the 2.11 review fixes to be missing.
- **Version not bumped** (`kickthefly/core/version.py` still says 2.12.0; CHANGELOG has a "2.13.0 (unreleased)" entry). Bump when releasing; the release workflow checks `RELEASE_NOTES.md` against the tag.
- **Validation is identical before and after.** `--headless --validate` on 2.11 vs this branch: all 21 tests, every measured value and per-seed number equal (ignoring only timestamps, worker count, version). Nothing in the simulation changed; the new fruit tool is appended to the tool list and eaten through sugar's own code.
- **GPU/driver safety.** Nothing installed, removed or configured; no sudo, no system env vars, pip untouched. The self-test and `test_backends` ran the app's own `gl` backend and an EGL context on the RX 9070 XT (normal app usage, child process with a time limit); no GPU failure occurred.
- **Pre-existing failures I fixed in tests** (they failed on 2.11, not caused by this work): `tests/test_i18n.py` (a stand-in config without `get`; `template.json` missing 11 keys of `en.json`) and `tests/test_outdoors.py::test_the_2d_game_stays_indoors` (a stand-in host without `is_larva`, read since 2.11).

## Test results (final state)

| run | result |
|---|---|
| `--headless --validate` | identical to the pre-change run (see above) |
| `pytest -m validation` | 24 passed |
| `tests/test_backends.py` (incl. gl on the GPU) | 32 passed, 3 skipped |
| rest of the suite (one serial run, before the last small fixes) | 396 passed, 2 failed, 4 skipped; both failures fixed afterwards (the outdoors stand-in above, and a test-order problem in the clipboard test) |
| re-run after those fixes (3 parallel shards) | 211 passed, 0 failed: loadout, gamepad, config, compat (107); tutorial, outdoors, modes, i18n (34); selftest, bugreport, workflows, nightly report, replay, api, labpages, pet, larva (70). `tests/playthrough` (9) passed separately. The other ~200 tests of the serial run were not re-run: nothing they cover changed after it. |
| `--selftest` (all backends incl. gl) | 16 checks, all PASS, exit 0 (earlier run; audio device present) |
| `--headless --playthrough all` (full, NumPy CPU) | 391 rows in about 10 min: 306 pass, **1 fail**, 69 gated by design, 10 inconclusive (skip). 135/135 save-then-load restored, 69/69 replays identical, 106 combos ended in death + autopsy, sim/real 2.8x-46x |

**The one playthrough failure is a real finding, left as is:** 2D game, flypaper arena, spider: the spider kills the stuck fly (6 bites, ~6 s) but LPLC2/LC4 barely rise (4.4 Hz vs 3.3 Hz calm). Looming is computed from how fast an object grows within one frame; it is 40-49 Hz in every other arena and marginal (5.6 vs 5.4 needed) in the 2D pool. Worth a look; nothing was changed to make it pass.

**Inconclusive skips (10), by design and visible in the counts:** item tools (sugar, fruit, alcohol) in the pool and alcohol under the lamp, where the fly never ate; a light hand grab in flypaper, pool and the escape room, where the arena already drives the touch neurons 10-15x above rest. A silent group is a FAIL only when the fly was engaged and the calm baseline was calm.

**Criterion, and one change I made after seeing results (please review):** "fired above baseline" = the busiest 100 ms after use is > 4 SD above the mean of the calm 100 ms windows before it, and >= 15% and 0.5 Hz above it. The first full run had 33 failures. I did **not** loosen that criterion; I fixed the harness: (1) each combo now starts from a saved calm state (the PAM calm rate had drifted from ~30 to ~55 Hz across sugar combos), (2) the two "inconclusive" preconditions above. The sigma count was moved 3 -> 4 earlier, before any full run, to keep false positives low on a null test (`tests/playthrough`), not to make tools pass.

## Config migration (schema 2 -> 3)

`config.toml` gains `[loadout]` (custom list, up to five `[[loadout.saved]]` with names), `[first_run]` (`tutorial_done`, `loadout_notice`), and `[controls] loadout_preset` (default `auto`). A file with schema < 3 (or none) migrates once: preset `all` (the number keys' old order), `loadout_notice = true` (a one-time popup pointing at the editor, shown from the run loops), `tutorial_done = true` (existing players aren't onboarded again; Settings > Help replays it), a warning logged, the rest untouched, and an explicit older `loadout_preset` is kept. A second load doesn't migrate again. Fresh install: `auto` (Base in Play, Lab in Lab, Pet in Pet) and the tutorial. Tests: `tests/test_loadout.py` (migration, 1.x file, bad sections, round trip, five-save limit). Save states restore the tool by index (the list only grows; a tool the mode disallows is put down for the hand). Replays get an informational `tool` event; older versions skip it.

## New keybindings

Only Esc is reserved now (0-9, - and = were reserved before, and bound to tools 1-12).

| action | default |
|---|---|
| hotbar slots 1-10 | 1 2 3 4 5 6 7 8 9 0 |
| previous / next hotbar page (loadouts > 10 tools) | - and = |
| loadout editor | Q (not in photo mode, where Q flies the camera down) |
| tool wheel (hold) | ` (backquote). I chose it because C is taken by 3D crouch/fly-down, Tab keeps freeing the mouse, and Tab-long-press would have collided with that |
| gamepad loadout editor | X (new pad action); LB/RB now cycle the loadout; hold Y still opens the wheel, now with every tool |

## What changed, by task

**Task 1, loadouts.** `core/loadout.py` (catalog: category, one-line description, which neurons, CONNECTOME/GAME RULE, Lab-only/larva flags, `probes` for the bot; presets base/chaos/chemist/lab/all/pet/custom/auto; `Loadout` with pages, hand pinned in slot 1, equip/move; `use()` = the tool's documented stimuli through `Brain.poke`), `core/config.py`, `ui/loadout_ui.py` (editor as a Menu page: categories grid, click/drag equip and reorder, drag off to remove, Reset to preset, five named saves via a new text field; the radial wheel; the migration notice; all colors from the accessibility palette), `ui/menu.py` (generic drag-and-drop and text entry, the Help tab), `game/kick_the_fly.py` and `game/kick3d.py` (hotbar drawn from the loadout with page arrows, key handling through rebindable actions, wheel for keyboard/mouse (3D: the mouse points instead of looking) and pad, wheel/hotbar clicks, `refresh_loadout` on mode/preset change), `lab/api.py` (`fly.loadout`, `set_loadout`, `use_tool`, `Fly(mode=...)`), `core/replay.py` (tool events). Per-mode defaults and larva hiding are as specified. **Pet preset = Base plus fruit**: Base has no bomb/cleaner/zapper to remove, and "orchard fruit" did not exist as a tool, so I added the **fruit** tool (Reward): appended to `TOOLS` (indices of older saves keep meaning), a sugar-list item flagged `fruit`, same eating code, smells like sugar, drawn as a red fruit. Larva-hidden tools (no larval mapping in this model): bomb, spider, alcohol, cVA, decoy, laser, each with its reason in the editor.

**Task 2, self-test.** `core/selftest.py`, `protocols/selftest_smoke.yaml`; `--selftest [--out]`, Settings > Help > Self-test (background thread, rows with verdict and fix, palette-safe colors). CPU backends compared spike for spike with NumPy, GPU statistically (firing within 5%, population rates r > 0.9 over 200 steps: my numbers, chosen because the repo's 1000-step tolerance is 2%/0.95), each GPU backend and the GL query in a child process with a time limit, GPU vendor/renderer/GL version from the app's own ModernGL context, pack zip CRC + SHA-256 (against a `.sha256`/`SHA256SUMS` beside it when one exists; there is no published pack checksum in the repo), OpenGL 3.3, audio, ffmpeg, four folders, disk/RAM against the 350 MB per-fly estimate, Wayland/X11, a 10 s protocol. Exit 0/1/3.

**Task 3, bug report.** `core/bugreport.py`, `ui/help_ui.py` (review page), `ui/crashscreen.py`, `core/crash.py:handle_crash` (used by all three crash handlers; windowed runs only; `KICK_THE_FLY_NO_CRASH_SCREEN=1` skips). Items are scrubbed of the home path and user name; each has a switch; the page shows each item's exact text plus two view-only rows: the report exactly as copied/saved and the exact text placed in the issue link. Copy (pygame scrap, else wl-copy/xclip/xsel/clip/pbcopy if present), save to file, or open a prefilled GitHub issue URL (limit 6000 chars; log/crash/JSON go to a file under `<state>/bugreports/` for the player to attach). `tests/test_bugreport.py` fails if the module imports anything network-capable; `--bugreport` prints/saves without opening anything.

**Task 4, tutorial.** `ui/tutorial.py`: move, use a tool, watch the brain panel (7 s or Next), sugar (puts sugar in hand), loadout editor. Enter/Next/A skip a step, Backspace/Skip/Back end it. Started only from the run loops, never from `Game.__init__`. The highlight never flashes and doesn't even swell with Reduced flashing. Flag in `[first_run]`. **"About 60 seconds" is not measured with a person.**

**Task 5, playthrough bot.** `lab/playthrough.py`, `tests/playthrough/`, docs/playthrough.md. Real `Game3D` (no window) in all 9 arenas and the 2D `Game` in the indoor ones, lockstep; larva at brain level (no windowed larva game exists); gates asserted (larva x unusable arena, larva x unmapped tool, laser outside Lab, 2D x outdoor arena); extras: 8-fly spawn/despawn, surgery, 5-pairing training, duel, pet 3-day catch-up (with a fake clock; real-stakes death), individuality (sigma, positivity, determinism), every loadout preset x mode x brain with every slot used, an offscreen render (skipped without GL). `--playthrough-quick` for PRs. The larva pack isn't built on CI: those combos record a skip.

**Task 6, CI.** `.github/claude-review.md` and `claude-review.yml` (Sonnet, skips cleanly without `ANTHROPIC_API_KEY` or on forks), `nightly.yml` + `tools/nightly_report.py` (static HTML with inline-SVG charts, no scripts/requests, pushed to `gh-pages`; issue with label `nightly` on a flip, an off-expected result or a playthrough failure), `tests.yml` (new "selftest and playthrough" job; add it to branch protection), `release.yml` (selftest on Linux, Windows, the AppImage and the exe; full playthrough job; `release` needs it). Setup is in CONTRIBUTING.md and docs/ci.md.

**Also:** `tools/i18n_sync.py` (adds new strings to `en.json`/`template.json`; `--check` is a test), README, CONTRIBUTING, CHANGELOG, docs/loadouts.md, docs/selftest-and-bugreport.md, docs/playthrough.md, docs/ci.md, docs/api.md, and the canonical docstring in `game/kick_the_fly.py`.

## Files changed (48)

Added: `.github/claude-review.md`, `.github/workflows/{claude-review,nightly}.yml`, `docs/{ci,loadouts,playthrough,selftest-and-bugreport}.md`, `kickthefly/core/{bugreport,loadout,selftest}.py`, `kickthefly/lab/playthrough.py`, `kickthefly/ui/{crashscreen,help_ui,loadout_ui,tutorial}.py`, `protocols/selftest_smoke.yaml`, `tests/playthrough/{README.md,test_playthrough.py}`, `tests/test_{bugreport,loadout,nightly_report,selftest,tutorial,workflows}.py`, `tools/{i18n_sync,nightly_report}.py`, `HANDOFF_2.13.md`.
Modified: `.github/workflows/{release,tests}.yml`, `CHANGELOG.md`, `CONTRIBUTING.md`, `README.md`, `docs/api.md`, `kick_the_fly.py`, `kickthefly/__main__.py`, `kickthefly/core/{config,crash,replay,savestate}.py`, `kickthefly/data/locales/{en,template}.json`, `kickthefly/game/{kick3d,kick_the_fly}.py`, `kickthefly/lab/{api,headless}.py`, `kickthefly/ui/menu.py`, `tests/{test_gamepad,test_i18n,test_outdoors}.py`.

## Known issues and limits

- The 2D flypaper spider finding above; 10 inconclusive skips.
- **German** was not extended: new strings are in `en.json`/`template.json` and fall back to English in `de`.
- The pet mode's `tr()`/catalog covers the new UI; older screens are as translated as before.
- 2D game: no multi-fly, duel or render checks (3D only); the wheel and hotbar work in both.
- `k2.TOOL_KEYS` is kept but unused; keypad digits don't select slots (they never did).
- The tutorial's "brain" step is time-based (7 s), not tied to a neuron firing.
- The fast suite is slower: roughly +6 min serial (tutorial 66 s, loadouts 52 s, playthrough 80 s, selftest/CLI ~15 s). Serial full pytest took 33 min here; it parallelizes by file.
- Playthrough game legs are not bit-deterministic (the 2D game uses Python's `random`); only the brain-level leg is recorded and replayed.
- `Fly.use_tool` is the sensory drive only (no body, no items).

## Things I did NOT manually verify

- Frozen builds: the exe and AppImage `--selftest`, the child-process re-exec of `sys.executable`, the bundled `protocols/selftest_smoke.yaml`, and the crash screen inside a frozen app. No build was made.
- Windows anything (clipboard `clip`, paths, the new pwsh steps in `release.yml`).
- Every GitHub-side behavior: the Claude Code Action's input names and version (`anthropics/claude-code-action@v1`, `claude-sonnet-5-5`), the fork/no-secret skip on a real PR, the nightly's gh-pages push, Pages setup and issue creation, the `git --work-tree` history restore. Workflows parse and are unit-tested only.
- A real window: everything ran under SDL dummy/offscreen (real GL context, screenshots inspected), never a visible window on Wayland; the real mouse-drag feel of the editor and wheel (events are simulated); the crash screen's `show()` loop interactively.
- A real gamepad (only fake pads), pygame's clipboard on a real display, wl-copy/xclip, and opening a real browser at the GitHub URL.
- The tutorial's length and wording with a real first-time player.
- Larger text at a real 125%/150% display scale for the editor and the wheel.
- The `--selftest` result on a machine with no GPU, no audio, no ffmpeg (the WARN paths are unit-tested with monkeypatching).
- GPU backends other than `gl` (no CUDA/ROCm torch here), so their statistical comparison path is untested on hardware.

The last edits after the serial run were: a `hasattr` guard in `apply_setting`, three test fixes (clipboard order, outdoors and i18n stand-ins), and the playthrough harness (snapshot reset, inconclusive skips, `rest` baselines). The full serial pytest was **not** repeated end to end after them; the shard re-run above covers every file they touch. Run it once (`python -m pytest`, about 33 min serial, or shard by file) before pushing.
