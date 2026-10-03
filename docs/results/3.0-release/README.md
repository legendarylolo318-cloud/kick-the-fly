# 3.0 release review: results

- `validate_before.json`: `--headless --validate --sim-backend cpu --workers 4` on `release/3.0` before day 5 (fd0d20c).
- `validate_after.json`: the same on the 3.0.0 code (fdec1ec). Identical except `created`, `seconds` and `app_version`.
- `rigs/`: the four rig assays re-run on seeds 1000-1009 from a frozen worktree of day 5's head (bf5c5b0); every value identical to `../day5/`.
- `playthrough.md`: `--headless --playthrough all --sim-backend cpu` on fdec1ec: 345 passed, 0 failed, 4 gated, 12 skipped.
- `selftest.json`: `--selftest --sim-backend cpu`: 22 passed, 1 warning (pynwb), 0 failed.
- `compat/`: the scripts that made real 2.13.1 artifacts with the v2.13.1 checkout (`gen_2131.py`) and loaded them with 3.0 (`load_30.py`).
- `sky_night_storm_day.png`: the open field at night, in a storm and by day after the sky-dome fix (offscreen, 1280x760).
