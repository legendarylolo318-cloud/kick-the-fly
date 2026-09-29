"""Entry point: `python -m kickthefly` (and what the `kick_the_fly.py` shim in the repo root calls)."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from kickthefly.game.kick_the_fly import main as run

    return run(argv)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()             # Lab worker processes in the exe and AppImage start here
    from kickthefly.core import crash

    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        crash.handle_crash()
        raise
