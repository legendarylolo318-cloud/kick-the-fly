"""The frame profiler (3.1.0 task 5): where a frame's milliseconds go. Off by default; a rebindable key (F3) shows it over the game.

Sections, per displayed frame, in milliseconds of the game's own thread (CPU time spent submitting work, not GPU time):
  sim      the brains' own step time summed over the steps they took this frame. Brains run on their own threads, so this is *not* time the
           frame waited; it is how much compute the simulation used while the frame ran
  physics  fly physics, tools, predators and the rest of the game update, and the player's movement
  render   drawing the world (3D: building and submitting the scene; 2D: drawing the arena)
  ui       the HUD, brain panel, menus and the science card
  present  swapping buffers (this is where vsync, the frame cap and a busy GPU show up)
Sections nest: time spent in a section inside another is taken out of the outer one, so the rows add up to the frame.

`PROF` is the one profiler. With it off, `PROF.section(name)` returns a shared do-nothing context, so the instrumentation costs a
method call per section. GAME RULE-free: it reads clocks and counters, nothing the simulation does.
"""
from __future__ import annotations

import time
from collections import deque

ORDER = ("sim", "physics", "render", "ui", "present")
WINDOW = 180                                  # frames kept: three seconds at 60 fps


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_NULL = _Null()


class _Section:
    __slots__ = ("p", "name", "t0", "inner")

    def __init__(self, p: "Profiler", name: str):
        self.p, self.name, self.inner = p, name, 0.0

    def __enter__(self):
        self.p._stack.append(self)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        dt = (time.perf_counter() - self.t0) * 1000.0
        st = self.p._stack
        st.pop()
        self.p._cur[self.name] = self.p._cur.get(self.name, 0.0) + dt - self.inner
        if st:
            st[-1].inner += dt
        return False


class Profiler:
    def __init__(self) -> None:
        self.on = False
        self._cur: dict[str, float] = {}
        self._stack: list[_Section] = []
        self.hist: dict[str, deque] = {k: deque(maxlen=WINDOW) for k in ORDER}
        self.frames: deque = deque(maxlen=WINDOW)      # whole-frame milliseconds (wall clock between end_frame calls)
        self._last: float | None = None
        self._sim_busy: float | None = None

    # --- switching --------------------------------------------------------------------------------------------------------------
    def set(self, on: bool) -> None:
        if on and not self.on:
            self.reset()
        self.on = bool(on)

    def toggle(self) -> bool:
        self.set(not self.on)
        return self.on

    def reset(self) -> None:
        self._cur.clear()
        self._stack.clear()
        for h in self.hist.values():
            h.clear()
        self.frames.clear()
        self._last = None
        self._sim_busy = None

    # --- measuring --------------------------------------------------------------------------------------------------------------
    def section(self, name: str):
        return _Section(self, name) if self.on else _NULL

    def add(self, name: str, ms: float) -> None:
        if self.on:
            self._cur[name] = self._cur.get(name, 0.0) + ms

    def sim_busy(self, total_busy_s: float) -> None:
        """The brains' cumulative step time (all flies, seconds): this frame's `sim` is how much it grew since the last frame."""
        if not self.on:
            return
        if self._sim_busy is not None and total_busy_s >= self._sim_busy:
            self.add("sim", (total_busy_s - self._sim_busy) * 1000.0)
        self._sim_busy = total_busy_s

    def end_frame(self) -> None:
        if not self.on:
            return
        now = time.perf_counter()
        if self._last is not None:
            self.frames.append((now - self._last) * 1000.0)
            for k in ORDER:
                self.hist[k].append(self._cur.get(k, 0.0))
        self._last = now
        self._cur.clear()
        self._stack.clear()                    # an exception inside a section must not leave it open forever

    # --- reading ----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _stats(values) -> tuple[float, float, float]:
        v = sorted(values)
        if not v:
            return 0.0, 0.0, 0.0
        return sum(v) / len(v), v[min(len(v) - 1, int(0.95 * len(v)))], v[-1]

    def stats(self) -> dict:
        """{section: (mean, p95, max) ms, ..., "frame": (...), "fps": float}."""
        out = {k: self._stats(self.hist[k]) for k in ORDER}
        out["frame"] = self._stats(self.frames)
        mean = out["frame"][0]
        out["fps"] = 1000.0 / mean if mean > 0 else 0.0
        return out

    def lines(self, backend: str = "", view: str = "") -> list[str]:
        s = self.stats()
        rows = [f"FPS {s['fps']:5.1f}   frame {s['frame'][0]:5.2f} ms  (p95 {s['frame'][1]:5.2f}, max {s['frame'][2]:5.1f})"]
        for k in ORDER:
            m, p95, mx = s[k]
            tag = "  off-thread" if k == "sim" else ""
            rows.append(f"{k:<8}{m:6.2f} ms   p95 {p95:6.2f}   max {mx:6.1f}{tag}")
        if backend:
            rows.append(f"engine  {backend}")
        if view:
            rows.append(f"view    {view}")
        return rows


PROF = Profiler()


# --- the --benchmark scenes -----------------------------------------------------------------------------------------------------
# `--benchmark` runs the game's own smoke run (a fixed seed, the room, a few flies) in a child process with KICK_THE_FLY_PROFILE_JSON set;
# the game loops call these four functions. The first half of the run is warm-up and is thrown away.
import json as _json
import os as _os


def bench_path() -> str | None:
    return _os.environ.get("KICK_THE_FLY_PROFILE_JSON") or None


def bench_start() -> None:
    if bench_path():
        PROF.set(True)


def bench_warm_done() -> None:
    if bench_path():
        PROF.reset()
        PROF.on = True


def bench_finish(game, status: str) -> None:
    path = bench_path()
    if not path:
        return
    st = PROF.stats()
    row = {k: {"mean": st[k][0], "p95": st[k][1], "max": st[k][2]} for k in (*ORDER, "frame")}
    row["fps"] = st["fps"]
    row["frames"] = len(PROF.frames)
    try:
        be = game.flies[0].brain.sim.backend
        row["engine"], row["device"] = be.name, be.device
        rates = [sl.brain.steps_per_s for sl in game.flies if sl.brain.steps_per_s > 0]      # a fly spawned this second has no rate yet
        row["steps_per_s"] = sum(rates) / len(rates) if rates else 0.0
        row["neurons"] = int(game.flies[0].brain.n)
        row["flies"] = len(game.flies)
    except Exception:
        pass
    row["view"] = (getattr(getattr(game, "view", None), "engine_note", "") or "CPU (sparse matrices)").split(" (")[0]
    row["status"] = status
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(row, f, indent=1)
