#!/bin/bash
# Run the tests with a live per-test list and percentage. Usage: ./watch_tests.sh [pytest args or test files]
cd "$(dirname "$0")"
export SDL_VIDEODRIVER=offscreen SDL_AUDIODRIVER=dummy KICK_THE_FLY_HOME="${KICK_THE_FLY_HOME:-/tmp/ktf-home}"
exec .venv/bin/python -m pytest "${@:-tests}" -v -p no:cacheprovider
