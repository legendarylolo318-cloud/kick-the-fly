"""Kick the Fly: launcher shim.

The game lives in the `kickthefly` package next to this file; this shim keeps `python kick_the_fly.py ...` working
exactly as before, with every command line flag unchanged.

    python kick_the_fly.py                                  the game (3D, --2d for the 2D one)
    python kick_the_fly.py --headless --validate            no window
    python kick_the_fly.py --headless --protocol smoke.yaml
    python -m kickthefly                                    the same thing, as a module

The canonical map of what comes from the connectome and what is a game rule is the module docstring at the top of
kickthefly/game/kick_the_fly.py. README.md carries the same split in prose, and kickthefly/__init__.py says what
lives in each subpackage.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))    # run from anywhere, not only the repo root

from kickthefly.__main__ import main                        # noqa: E402

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()                        # Lab worker processes in the exe and AppImage start here
    from kickthefly.core import crash                       # noqa: E402

    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        crash.handle_crash()
        raise
