# Continuous integration

| workflow | when | what |
|---|---|---|
| `tests.yml` | every pull request into `main`, every push to `main` | the fast suite, the validation suite, and (2.12) `--selftest` plus the quick playthrough on CPU |
| `checks.yml` | pull requests, pushes to `main`, and from `release.yml` | gl on llvmpipe, the Flatpak build, docs links, replay determinism |
| `claude-review.yml` | every non-draft pull request | a Claude (Sonnet) review against `.github/claude-review.md`; skipped cleanly without the secret |
| `nightly.yml` | every night at 03:17 UTC, and by hand | full validation and the full playthrough (CPU, ubuntu-22.04), a history table and trend charts on GitHub Pages, an issue if something flipped or failed |
| `release.yml` | a `v*` tag, or by hand (builds without publishing) | everything above, on Linux and Windows, plus the self-test and the full playthrough, then the AppImage and the exe (each self-tested) |

## Claude review

`.github/workflows/claude-review.yml` runs `anthropics/claude-code-action` with `claude-sonnet-5-5` on each pull request and posts one
comment. The checklist it reviews against is `.github/claude-review.md`: GPU and system safety, CONNECTOME / GAME RULE tagging, no tuning to
pass, validation numbers that match a real run, migrations that are tested, strings that are localized. Edit that file to change what it looks for.

Setup: add a repository secret named `ANTHROPIC_API_KEY` (Settings > Secrets and variables > Actions > New repository secret). The workflow only
comments; it has no write access to the code. **If the secret is missing, or the pull request comes from a fork (GitHub doesn't give forks
secrets), the job prints a notice and finishes green.** Nothing else waits on it and it isn't a required check. It is a reviewer, not a gate.

## Nightly

`nightly.yml` runs the full validation (`--headless --validate`), the full adult playthrough and the self-test, then `tools/nightly_report.py`
appends the night to `history.json` and writes one static `index.html` (a table and inline-SVG trend charts: validation tests passed, playthrough
failures, the median sim/real ratio, each validated pathway's drive ratio; no scripts, no external requests, no trackers) that the workflow pushes
to the `gh-pages` branch. One-time setup: Settings > Pages > Source: Deploy from a branch > `gh-pages` / root. It needs no secret.

It opens an issue (label `nightly`) when a validation result **flips** relative to the previous night, when a result differs from the expected
pass/fail in `kickthefly/lab/validation.py:EXPECTED`, or when the playthrough fails; while one is open, later failures add a comment instead of
a new issue. Nothing is tuned or retried to make the night pass.

## Making the required checks

`main`'s branch protection should require `fast tests`, `validation` and `selftest and playthrough`. `claude-review` should stay optional.
