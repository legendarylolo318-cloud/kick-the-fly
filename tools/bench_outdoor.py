"""Outdoor performance, in the real 3D game (3.0 day 3): frames a second and the brain's simulation speed in the open field and the
orchard, with and without weather. One process per measurement (one GL context per process), rendered offscreen with SDL's offscreen
driver, so no window opens.

    python tools/bench_outdoor.py field clear            # 25 s in the open field, no weather
    python tools/bench_outdoor.py field storm            # the same with a storm (rain 1, gusts, lightning); needs 3.0 day 3 or later
    python tools/bench_outdoor.py orchard rain --seconds 25 --out result.json

Prints one JSON line: arena, weather, fps over the last half, sim/real (the brain's steps a second over 200, mean over flies).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
WEATHER = {"clear": {}, "rain": {"weather.rain": 1.0}, "storm": {"weather.storm": 1.0, "weather.rain": 1.0, "weather.gust_hz": 0.3}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arena", choices=("field", "orchard"))
    ap.add_argument("weather", choices=tuple(WEATHER))
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--out")
    ap.add_argument("--no-rain-draw", action="store_true", help="skip drawing the streaks, to separate drawing cost from the neurons' cost")
    a = ap.parse_args()
    os.environ["KICK_THE_FLY_HOME"] = tempfile.mkdtemp()                 # a throwaway home: settings and saves are untouched
    import pygame

    pygame.init()
    from kickthefly.core import config, paths
    from kickthefly.game import kick3d

    if a.no_rain_draw and hasattr(kick3d.Game3D, "_draw_rain"):
        kick3d.Game3D._draw_rain = lambda self, rd, eye, now: None
    paths.reset_cache()
    cfg = config.Config(None)
    cfg.set("brain.arena", a.arena)
    state = {"t0": None, "frames": 0, "marks": []}

    def script(game, app, t, lay):
        if state["t0"] is None:
            state["t0"] = t
            for k, v in WEATHER[a.weather].items():
                if k not in game.lab_params:
                    raise SystemExit(f"{k} does not exist in this build: weather needs 3.0 day 3")
                game.lab_params[k] = v
        state["frames"] += 1
        state["marks"].append((t, state["frames"]))
        if t - state["t0"] >= a.seconds:
            half = [m for m in state["marks"] if m[0] - state["t0"] >= a.seconds / 2]
            fps = (half[-1][1] - half[0][1]) / max(1e-9, half[-1][0] - half[0][0])
            rates = [s.brain.steps_per_s for s in game.flies]
            state["result"] = dict(arena=a.arena, weather=a.weather, seconds=a.seconds, fps=round(fps, 1),
                                   sim_real=round(sum(rates) / len(rates) / 200.0, 3), steps_per_s=round(sum(rates) / len(rates), 1))
            return False
        return True

    rc = kick3d.run(0.0, None, False, seed=3, cfg=cfg, flies=1, script=script)
    res = state.get("result")
    if res is None:
        print("no result", file=sys.stderr)
        return 1
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res))
    return 0 if rc in (0, None) else rc


if __name__ == "__main__":
    sys.exit(main())
