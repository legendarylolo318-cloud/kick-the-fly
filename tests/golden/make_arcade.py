"""Writes tests/golden/arcade.json: measured personality cards (lab/tournament.measure_card), one duel with drivers
(game/flyduel.run_duel) and one race lane (game/flyrace.run_lane), all on the real pack. Run from the repo root."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy-python"))
from kickthefly.core import cards  # noqa: E402
from kickthefly.game import flyduel, flyrace  # noqa: E402
from kickthefly.lab import tournament  # noqa: E402

out = {}
out["cards"] = {}
for seed, mode in ((2000, "subtle"), (2001, "subtle"), (5, "off"), (7, "strong")):
    c = tournament.measure_card(seed, mode)
    c.pop("backend", None)
    out["cards"][f"{seed}:{mode}"] = c
out["pack_sha256"] = cards.pack_sha()
a, b = 2000, 2001
mseed, na, nb = tournament.match_noise(0, a, b, 0)
ba, bb = tournament.build_fighter(a, "subtle", na), tournament.build_fighter(b, "subtle", nb)
r = flyduel.run_duel(ba, bb, seed=mseed, seconds=6.0, names=(str(a), str(b)), drivers=True)
d = r.pop("_drivers")
sums = {k: flyduel.combine_drivers([v]) for k, v in d.items()}
out["duel"] = dict(a=a, b=b, match_seed=mseed, noise=[na, nb], result=r,
                   top={k: flyduel.top_drivers(v, min_base=1, min_hit=1, min_windows=1) for k, v in sums.items()},
                   named={k: flyduel.named_drivers(v) for k, v in sums.items()},
                   hit_windows={k: v["hit_windows"] for k, v in d.items()})
mm = tournament.match_noise(7, 1, 2, 1)
out["match_noise"] = dict(args=[7, 1, 2, 1], value=list(mm))
from kickthefly.core import simcore  # noqa: E402

br = simcore.new_brain(seed=3000, individuality="subtle", warmup=600)
out["lane"] = flyrace.run_lane(br, cap_s=14.0)
Path(__file__).with_name("arcade.json").write_text(json.dumps(out, default=float))
print({k: (v if k != "cards" else list(v)) for k, v in out.items() if k in ("cards", "match_noise")})
print(r["winner"], r["hp"], r["shots"], r["hits"], len(r["frames"]), r["steps"])
print({k: out["lane"][k] for k in out["lane"] if k != "trace"})
