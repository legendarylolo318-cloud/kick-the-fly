"""Non-code assets bundled inside the package (and inside the exe and AppImage).

Small files that ship with the code live here and are read through `path()`. The big, generated things do not:
the connectome download, data/graph.pkl and data/kick_brain.npz stay in the repo-root `data/` folder
(kickthefly.DATA_DIR), and user protocols, saves and exports stay in the user folders kickthefly.core.paths
resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def path(name: str) -> Path:
    """A bundled asset, from the PyInstaller bundle when frozen and from this folder otherwise."""
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        bundled = Path(meipass) / "kickthefly" / "data" / name
        if bundled.exists():
            return bundled
    return HERE / name
