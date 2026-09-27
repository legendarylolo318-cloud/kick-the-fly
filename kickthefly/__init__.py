"""Kick the Fly: the MaleCNS v1.0 connectome, simulated live, as a game and as a lab bench.

Subpackages:
  core      clock, save states, settings, user folders, crash reporting, version, headless brain helpers
  sim       the connectome itself: loading, the brain pack, the LIF simulator, neuron and synapse state
  game      the 3D room, the 2D game, physics, tools, arenas
  ui        the menu, settings screens, HUD, brain panel and neuron inspector
  lab       Lab mode: validation, assays, challenges, statistics, protocols, recording, export, headless runs
  data      non-code assets bundled inside the package

The canonical map of what comes from the connectome and what is a game rule is the module docstring of
kickthefly/game/kick_the_fly.py; README.md carries the same split in prose.

Run it with `python -m kickthefly`, or with the `kick_the_fly.py` shim in the repo root. As a library:
`from kickthefly import Fly` (kickthefly/lab/api.py, docs/api.md).
"""
from pathlib import Path

__all__ = ["SOURCE_ROOT", "DATA_DIR", "PROTOCOLS_DIR", "Fly"]

# The checkout this package was imported from. Frozen builds (PyInstaller) look in sys._MEIPASS first and only fall
# back to these; they are what keeps data/ and protocols/ where every earlier version put them.
SOURCE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SOURCE_ROOT / "data"
PROTOCOLS_DIR = SOURCE_ROOT / "protocols"


def __getattr__(name):                       # `from kickthefly import Fly`, without importing the game up front
    if name == "Fly":
        from kickthefly.lab.api import Fly

        return Fly
    raise AttributeError(f"module 'kickthefly' has no attribute {name!r}")
