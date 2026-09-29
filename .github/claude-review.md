# Pull request review checklist

You are reviewing a pull request to Kick the Fly, a game whose fly is the MaleCNS v1.0 connectome running as a live
simulation. Review the diff (`git diff origin/main...HEAD`) against every item below. Be specific: name the file and line,
say what is wrong and what would fix it. Say plainly when an item does not apply or holds. Do not praise, do not
summarize the PR, do not nitpick style the surrounding code doesn't already enforce. If everything holds, say so in two
lines. Read CONTRIBUTING.md and the docstring at the top of `kickthefly/game/kick_the_fly.py` first: they are the
rules this checklist enforces.

## 1. GPU and system safety (blocking)

The change must never install, remove, upgrade or downgrade system packages, GPU drivers, Mesa, Vulkan, ROCm, CUDA,
amdgpu, linux-firmware or kernel modules. Flag any of these in code, scripts, docs, workflows or instructions to users:

- `pacman`, `yay`, `apt`, `dnf`, `pip install` outside a virtual environment, `sudo`, `modprobe`, `udev`, edits under
  `/etc`, bootloader or kernel parameters.
- System-wide `LIBGL_*`, `MESA_*`, `HSA_*`, `RADV_*`, `AMD_*` environment variables (a per-command variable for one test
  run is fine; llvmpipe/software GL is for CI runners only, never something the game or a doc tells a player to do).
- The self-test, the bug report or any new code that tries to install, configure or repair something instead of only
  saying what the player could do.

## 2. CONNECTOME or GAME RULE (blocking)

Every addition that touches the simulation is tagged, in all three places:

- the docstring of `kickthefly/game/kick_the_fly.py` (the canonical map) and the README's "What is the connectome and
  what is a game rule" tables;
- `REACTION_SOURCE` / `POPUP_SOURCE` for a new reaction or popup;
- the on-screen tag (Lab > Model assumptions, the tool catalog in `kickthefly/core/loadout.py`).

A behavior is either traceable to named neurons and synapses (CONNECTOME) or says GAME RULE. Flag an untagged rule, a
rule presented as biology, and any claim about the fly's biology that isn't grounded in the dataset or a citation.

## 3. No tuning to pass (blocking)

- A test, validation criterion, threshold, tolerance, seed or timeout must not be loosened, skipped, `xfail`ed or
  re-seeded to make a result pass. A criterion changes only with a written reason that is independent of the result.
- Simulation parameters (`THRESH`, LIF parameters, drive strengths, sensory mappings) must not change in a PR that isn't
  about them. A change there needs a validation run before and after, in the PR description.
- `--validate` results and every existing replay must come out identical unless the PR says why they change.

## 4. Validation numbers match a real run (blocking)

Any number in the PR, README, docs or CHANGELOG about validation (pass/fail counts, ratios, p-values), performance
(steps/s, sim/real) or behavior (rates, latencies) must be one the PR shows a run for. Flag numbers that are round,
copied from an older version, or contradict `data/validation_results.json` / the test output in the PR.

## 5. Migrations are tested

A change to `config.toml`, save states, replays, training memory, protocols or command lines keeps old files working.
There must be a test that loads an old file and checks what it becomes, and one that a second load doesn't migrate
again. Bump `SCHEMA_VERSION` (config) or the format version (saves, replays) when the format changes.

## 6. Strings are localized

Every new user-facing string goes through `tr()` and is in `kickthefly/data/locales/en.json` and `template.json`
(`python tools/i18n_sync.py --check`). Every new setting has a hover tip. Cell types, gene names and citations are never
translated. New UI honors Accessibility: reduced flashing, larger text, the colorblind palettes.

## 7. Everything else that breaks the build for a player

- New files a player writes go through `kickthefly/core/paths.py`, never a hardcoded folder.
- No new runtime dependency without a reason; optional ones degrade to a message, never an import error at startup.
- No network access, upload or telemetry beyond what the PR states (the bug report only hands a URL to the player's own
  browser).
- CPU backends stay bit-exact with the NumPy reference; anything that writes `sim.v`, `sim.refr`, `sim.spikes` or the
  weights works on every backend.
- A dev script belongs in `tools/`, not the repo root.
