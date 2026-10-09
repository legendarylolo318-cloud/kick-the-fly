"""Recipe for tests/python_files/savestate/ (a real .ktfsave and the raster Python produces after it). Not collected as a test:
copy it into legacy-python/tests/ as _tmp_save_fixture.py, then run (needs numpy, scipy, pygame-ce, pytest):

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy DATA=$PWD/../data OUT=$PWD/../tests/python_files/savestate \
        python -m pytest tests/_tmp_save_fixture.py -q -s

and delete the copy. (DATA is the folder with kick_brain.npz; the pack moved out of legacy-python/.)
"""
import os, pathlib, json, hashlib
import numpy as np
from kickthefly.sim import brainpack
brainpack.DATA_DIR = pathlib.Path(os.environ["DATA"])
from test_determinism import game_run, game2d
def test_make(game2d):
    from kickthefly.core import savestate
    g = game2d(0)
    game_run(g, 60)
    g.arena_i = 3
    out = pathlib.Path(os.environ["OUT"])
    savestate.save_game(g, out / "real.ktfsave")
    sim = g.flies[0].brain.sim
    rows = []
    for _ in range(60):
        idx = np.flatnonzero(sim.step(None))
        rows.append(dict(n=int(len(idx)), sha=hashlib.sha256(idx.astype("<u4").tobytes()).hexdigest()[:16], gain=float(sim.gain)))
    (out / "continue.json").write_text(json.dumps(rows, indent=0))
