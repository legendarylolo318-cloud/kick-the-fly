"""Writes tests/golden/choice.json: np.random.default_rng(seed).choice(pop, size, replace=False) call sequences (what Brain.poke uses)."""
import json
from pathlib import Path

import numpy as np

out = []
for seed, calls in ((0, [(5, 1), (5, 5), (100, 30), (100, 1), (1000, 300), (3000, 10), (3000, 2999)]),
                    (7, [(12000, 100), (12000, 300), (12000, 6000), (50000, 100), (50000, 1500), (20000, 20000), (166700, 30), (166700, 9000)]),
                    (3000, [(40, 12), (7, 3), (60, 60), (2, 1), (1, 1), (333, 100)])):
    rng = np.random.default_rng(seed)
    res = [[int(v) for v in rng.choice(np.arange(n), size=k, replace=False)] for n, k in calls]
    out.append(dict(seed=seed, calls=calls, results=res, after=int(rng.integers(0, 1 << 30))))
Path(__file__).with_name("choice.json").write_text(json.dumps(out))
