# The playthrough bot

The bot itself is `kickthefly/lab/playthrough.py` (so the exe, the AppImage and the nightly workflow run it with
`--headless --playthrough`). This folder is its pytest side:

- `test_playthrough.py` unit-tests the bot's own machinery (the probe statistics, the gates, the report) and runs a
  small slice of the real thing: a few tools in the room, on both brains. It is quick enough for every pull request.
- The full matrix (every arena x every tool x each brain, 2D and 3D games, the extras) is
  `python kick_the_fly.py --headless --playthrough all --out playthrough-report`, which the nightly workflow runs on CPU.
  `--playthrough-quick` is the pull-request-sized version (each tool once in the room and the orchard).

What a run checks, and what is gated by design, is the docstring of `kickthefly/lab/playthrough.py`. A failure there is
a finding: the bot never loosens a criterion to pass, and a tool whose documented neurons don't fire is reported with the
numbers.
