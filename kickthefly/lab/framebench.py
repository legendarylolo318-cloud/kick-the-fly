"""`--benchmark`'s second table (3.1.0 task 5): where a frame's milliseconds go, in a fixed rendered scenario.

Each scene is the game's own smoke run in a child process: seed 1, the room, a few flies (3D), the default 60 fps cap (one row uncapped), an offscreen window
(EGL, so it needs no display and the monitor's refresh rate and vsync cannot cap it), ten seconds of which the first five are warm-up. The
child switches the profiler on (core/profiler.py) and writes its numbers as JSON. The rows:
  3D            the 3D game as shipped (the brain view on the GPU where there is one), at the 60 fps cap
  3D uncapped   the same with no cap: how fast the frame can go
  3D, CPU view  the same as `3D` with the brain view drawn by the CPU (KICK_THE_FLY_BRAINVIEW=cpu), which isolates what the GPU view saves
  2D            the 2D game, one fly
The simulation table (lab/benchmark.py) is separate and runs first.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# (name, extra arguments, extra environment, frame-rate cap). The capped scenes are what a player gets at the default 60 fps cap, and show whether the
# brains keep real time while the game draws; the uncapped one shows how fast the frame itself can go (and, drawing flat out, it competes with a GPU
# brain for the GPU, so its steps/s are lower: that is the point of capping).
SCENES = (
    ("3D", ["--flies", "4"], {}, 60),
    ("3D uncapped", ["--flies", "4"], {}, 0),
    ("3D, CPU view", ["--flies", "4"], {"KICK_THE_FLY_BRAINVIEW": "cpu"}, 60),
    ("2D", ["--2d"], {}, 60),
)


def _command() -> list[str]:
    base = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "kickthefly"]
    return base


def run_scene(name: str, extra: list[str], env_extra: dict, seconds: float, backend: str | None, seed: int = 1, cap: int = 0) -> dict:
    with tempfile.TemporaryDirectory(prefix="ktf-bench-") as tmp:
        home = Path(tmp) / "home"
        (home / "config").mkdir(parents=True)
        (home / "config" / "config.toml").write_text(f"[graphics]\nfps_cap = {int(cap)}\nvsync = false\n", encoding="utf-8")
        out = Path(tmp) / "scene.json"
        env = dict(os.environ, KICK_THE_FLY_HOME=str(home), KICK_THE_FLY_PROFILE_JSON=str(out), SDL_VIDEODRIVER="offscreen",
                   SDL_AUDIODRIVER="dummy", **env_extra)
        cmd = _command() + ["--smoke", f"{seconds:g}", "--seed", str(seed), "--arena", "room", *extra]
        if backend and backend != "auto":
            cmd += ["--backend", backend]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=max(120.0, seconds * 6))
        except subprocess.TimeoutExpired:
            return {"scene": name, "error": "timed out"}
        if not out.exists():
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["no output"]
            return {"scene": name, "error": f"exit {r.returncode}: {tail[0][:100]}"}
        row = json.loads(out.read_text(encoding="utf-8"))
        row["scene"] = name
        return row


def run_scenes(seconds: float = 10.0, backend: str | None = None, scenes=SCENES) -> list[dict]:
    return [run_scene(n, e, env, seconds, backend, cap=cap) for n, e, env, cap in scenes]


def format_table(rows: list[dict]) -> str:
    head = f"{'scene':<14}{'fps':>6}{'frame ms':>9}{'sim*':>7}{'physics':>8}{'render':>8}{'ui':>7}{'present':>8}  {'engine':<7}{'view':<5}{'steps/s':>8}"
    out = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            out.append(f"{r['scene']:<14}  not run: {r['error']}")
            continue
        m = lambda k: r[k]["mean"]                                                        # noqa: E731
        out.append(f"{r['scene']:<14}{r['fps']:>6.0f}{m('frame'):>9.2f}{m('sim'):>7.2f}{m('physics'):>8.2f}{m('render'):>8.2f}{m('ui'):>7.2f}"
                   f"{m('present'):>8.2f}  {r.get('engine', '?'):<7}{('GPU' if r.get('view', '').startswith('GPU') else 'CPU'):<5}"
                   f"{r.get('steps_per_s', 0):>8.0f}")
    out.append("Milliseconds per frame (mean over the second half of each run). *sim is the brains' own step time summed over the flies, on "
               "their own threads: compute used while the frame ran, not time the frame waited (flies batched on one GPU overlap, so it "
               "can exceed the frame). render and ui are CPU time spent building and submitting the frame; present is the buffer swap, "
               "where a busy GPU shows. steps/s is the mean per fly (200 is real time).")
    return "\n".join(out)
