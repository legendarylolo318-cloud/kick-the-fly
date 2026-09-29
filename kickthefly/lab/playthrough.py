"""The playthrough bot (2.12): scripted runs that use every tool in every arena on every brain and check what should hold.

    python kick_the_fly.py --headless --playthrough [adult|larva|all] --out DIR [--playthrough-quick] [--sim-backend cpu]
    python -m pytest tests/playthrough          # a quick subset, as pytest tests

For every arena x every tool x each brain the bot uses the tool on the fly and asserts:
  - no exception
  - the sim/real ratio (simulated seconds per wall second, brain steps included) stays above a floor. The floor only
    catches hangs and collapses, it is not a performance target: KTF_PLAYTHROUGH_MIN_RATIO or --min-ratio to change it
  - the tool's documented sensory neurons (loadout.ToolInfo.probes, the machine-readable form of what the editor
    says the tool drives) fire above their own calm baseline
  - reactions and meters stay in valid ranges (health, pain, group levels, membrane potentials: finite, in range)
  - death and the autopsy work where the tool kills it within the run
  - save then load in the middle of the run restores the state (positions, membrane potentials, health, tool, arena)
  - a replay of the run reproduces its spikes exactly on the CPU backends (numpy, numba, torch-cpu), and is only
    reported on GPU backends
Combos that are gated by design are not run: the bot asserts the gate's message is what a player would be told, and
records them as "gated" (passing):
  - larva x an arena the larva can't use (game/larva.py:is_arena_allowed_for_larva)
  - larva x a tool with no larval sensory mapping (the editor's "No larval sensory mapping" reason)
  - the laser outside Lab mode ("Lab mode only")
  - the 2D game x the outdoor arenas ("needs the 3D game")

Where the run is: adult flies run in the real games (Game3D everywhere, without a window, and the 2D Game in the
indoor arenas), stepped in lockstep from the calling thread, frame by frame. The larva has no windowed game yet
(docs/larva.md), so its runs are brain-level: the tool's documented stimuli through Brain.poke, the same call the
game's tools make. Every tool also gets a brain-level run on both brains, which is the one that is recorded and
replayed.

Also covered: multi-fly spawn and despawn up to 8, brain surgery on and off, training with 5 pairings, a duel start and
end, pet mode catch-up over a simulated 3-day gap, individuality off / subtle / strong, and every loadout preset in
every mode. The 3D renderer runs offscreen where OpenGL is available; with no GL its checks are SKIPPED, never failed.

Nothing here tunes anything: a tool whose documented neurons don't fire is reported as a failure with the numbers. The
run uses a temporary KICK_THE_FLY_HOME, so it never reads or writes your settings, saves or training memory.

Output (--out DIR): playthrough.json and playthrough.md (a table: combo, verdict, seconds, failures with tracebacks).
Exit code 0 if nothing failed (skips and gates are fine), 1 otherwise.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DT = 0.005
PASS, FAIL, SKIP, GATED = "pass", "fail", "skip", "gated"
DEFAULT_MIN_RATIO = 0.25
FRAME = 1 / 60
STEPS_PER_FRAME = 3                     # 15 ms of brain per 16.7 ms frame: the game's own real-time pace, in lockstep
# "Fired above baseline": the busiest 100 ms after the tool was used is more than 4 standard deviations above the mean of
# the calm 100 ms windows before it, and at least 15% and 0.5 Hz above that mean. The calm rate and its spread are
# measured in each run (a group like the PAM neurons sits at ~30 Hz calm, a touch group at ~2 Hz), so this is not a
# fixed ratio someone picked to make tools pass.
Z_SIGMAS, MIN_RISE, MIN_DELTA_HZ, WINDOW = 4.0, 1.15, 0.5, 20
# How each tool is used, and for how long at most (game seconds): "click" tools are clicked again every 0.8 s until it
# dies or the time is up, "hold" tools are held, "item" tools are dropped once and the player steps back, "once" is one use.
TOOL_PLAN = {"hand": ("hold", 3.0), "flick": ("click", 15.0), "swatter": ("click", 15.0), "bomb": ("click", 15.0),
             "zapper": ("click", 15.0), "torch": ("hold", 10.0), "freeze": ("hold", 8.0), "cleaner": ("hold", 12.0),
             "alcohol": ("item", 8.0), "cva": ("once", 3.0), "sugar": ("item", 8.0), "fruit": ("item", 8.0),
             "spider": ("once", 25.0), "decoy": ("item", 4.0), "laser": ("hold", 3.0)}
SAVE_LOAD_AT_S = 1.0        # seconds after the tool is first used: before anything has died
QUICK_ARENAS = ("room", "orchard")


# --- results -----------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    id: str
    group: str
    brain: str = ""
    arena: str = ""
    tool: str = ""
    status: str = PASS
    seconds: float = 0.0
    notes: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    traceback: str = ""
    metrics: dict = field(default_factory=dict)

    def expect(self, cond: bool, message: str) -> bool:
        if not cond:
            self.failures.append(message)
            self.status = FAIL
        return bool(cond)

    def note(self, text: str) -> None:
        self.notes.append(text)


class Report:
    def __init__(self, meta: dict):
        self.meta = meta
        self.results: list[Result] = []

    def add(self, r: Result) -> Result:
        self.results.append(r)
        return r

    def counts(self) -> dict:
        return {s: sum(1 for r in self.results if r.status == s) for s in (PASS, FAIL, SKIP, GATED)}

    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)

    def to_json(self) -> dict:
        return dict(format="kick-the-fly-playthrough", meta=self.meta, counts=self.counts(),
                    results=[asdict(r) for r in self.results])

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [f"# Kick the Fly {self.meta.get('app_version', '')} playthrough", "",
                 f"{self.meta.get('created', '')}, backend `{self.meta.get('backend', '')}`, "
                 f"brains: {self.meta.get('brains', '')}, {'quick' if self.meta.get('quick') else 'full'} run.", "",
                 f"**{c[PASS]} passed, {c[FAIL]} failed, {c[GATED]} gated by design, {c[SKIP]} skipped.**", "",
                 "| combo | group | verdict | seconds | notes / failures |", "|---|---|---|---|---|"]
        for r in self.results:
            what = "; ".join(r.failures) if r.failures else "; ".join(r.notes[:2])
            lines.append(f"| {r.id.replace('|', '/')} | {r.group} | {r.status.upper()} | {r.seconds:.1f} | {what.replace('|', '/')[:300]} |")
        bad = [r for r in self.results if r.status == FAIL and r.traceback]
        if bad:
            lines += ["", "## Tracebacks", ""]
            for r in bad:
                lines += [f"### {r.id}", "```", r.traceback.rstrip(), "```", ""]
        return "\n".join(lines) + "\n"

    def write(self, out: Path) -> tuple[Path, Path]:
        out.mkdir(parents=True, exist_ok=True)
        j, m = out / "playthrough.json", out / "playthrough.md"
        j.write_text(json.dumps(self.to_json(), indent=1, default=str), encoding="utf-8")
        m.write_text(self.to_markdown(), encoding="utf-8")
        return j, m


def guarded(r: Result, fn, *a, **k):
    """Run fn; an exception is this result's failure, with its traceback."""
    t0 = time.perf_counter()
    try:
        return fn(*a, **k)
    except Exception as e:
        r.status = FAIL
        r.failures.append(f"exception: {type(e).__name__}: {e}")
        r.traceback = traceback.format_exc()
        return None
    finally:
        r.seconds = round(time.perf_counter() - t0, 2)


# --- probes: did the documented neurons fire? --------------------------------------------------------------------------
class Probe:
    """Counts the spikes of the neurons a tool documents, every brain step, then compares the run after the tool was
    used with the calm before it (peak over 100 ms windows against the mean before)."""

    def __init__(self, br, probes, extra_rows: dict | None = None):
        self.br = br
        self.parts: list[tuple[str, np.ndarray]] = []
        self.empty: list[str] = []
        for probe in probes:
            name = "+".join(k[0] if k[1] is None else f"{k[0]}:{k[1]}" for k in probe)
            rows = np.unique(np.concatenate([np.asarray(br.sense.get(k, []), np.int64) for k in probe])) \
                if probe else np.array([], np.int64)
            (self.parts.append((name, rows)) if len(rows) else self.empty.append(name))
        for name, rows in (extra_rows or {}).items():
            if len(rows):
                self.parts.append((name, np.asarray(rows, np.int64)))
        self.counts: list[list[int]] = [[] for _ in self.parts]

    @property
    def n(self) -> int:
        return len(self.counts[0]) if self.counts else 0

    def sample(self) -> None:
        sp = self.br.sim.spikes
        for i, (_, rows) in enumerate(self.parts):
            self.counts[i].append(int(np.count_nonzero(sp[rows])))

    def summarize(self, use_step: int, window: int = WINDOW) -> list[dict]:
        out = []
        for (name, rows), c in zip(self.parts, self.counts):
            c = np.asarray(c, float)
            scale = 1.0 / (window * len(rows) * DT)                    # spikes in a window -> Hz per neuron
            calm = c[:use_step]
            k = len(calm) // window
            wins = calm[:k * window].reshape(k, window).sum(1) * scale if k else np.array([0.0])
            mu, sd = float(wins.mean()), float(wins.std(ddof=1)) if len(wins) > 1 else 0.0
            after = c[use_step:]
            peak = float(np.convolve(after, np.ones(window), "valid").max() * scale) if len(after) >= window else \
                float(after.mean() * len(after) / window * scale) if len(after) else 0.0
            need = max(mu + Z_SIGMAS * sd, mu * MIN_RISE, mu + MIN_DELTA_HZ)
            out.append(dict(probe=name, neurons=int(len(rows)), baseline_hz=round(mu, 3), baseline_sd_hz=round(sd, 3),
                            peak_hz=round(peak, 3), needed_hz=round(need, 3), ratio=round(peak / max(mu, 0.05), 2),
                            fired=bool(peak >= need)))
        return out


def judge_probes(r: Result, probe: Probe, use_step: int, tool: str) -> None:
    rows = probe.summarize(use_step)
    r.metrics["probes"] = rows
    for name in probe.empty:
        r.note(f"documented neurons '{name}' are not mapped in this brain: not checked")
    for p in rows:
        r.expect(p["fired"], f"documented neurons '{p['probe']}' did not fire above baseline "
                             f"({p['peak_hz']} Hz peak, needed {p['needed_hz']}; calm {p['baseline_hz']} +- {p['baseline_sd_hz']} Hz)")
    if not rows and not probe.empty:
        r.note("no documented neurons to check for this tool")


def check_ranges(r: Result, br, tag: str = "") -> None:
    """Meters and levels stay in their valid ranges; nothing went NaN."""
    v = np.asarray(br.sim.v)
    r.expect(bool(np.all(np.isfinite(v))), f"{tag}membrane potentials are not all finite")
    fast = np.asarray(br.fast, float)
    r.expect(bool(np.all(np.isfinite(fast))) and bool(np.all(fast >= 0)), f"{tag}group firing rates out of range")
    try:
        parts = br.pain_parts(fast[None, :], br.base)
        r.expect(bool(np.all((parts >= 0) & (parts <= 1))), f"{tag}pain components outside 0..1")
    except Exception as e:                                            # only for brains that have the groups
        r.note(f"pain components not checked ({type(e).__name__})")
    for name in ("reward", "head", "body"):
        if name in br.col:
            lv = br.level(name)
            r.expect(math.isfinite(lv) and lv >= 0, f"{tag}level of '{name}' is {lv}")


# --- brain-level leg (both brains): the recorded and replayed one --------------------------------------------------------
def _new_brain(kind: str, seed: int, backend: str | None):
    from kickthefly.core import simcore

    return simcore.new_brain(seed=seed, backend=backend, brain=kind, individuality="off")


def brain_leg(kind: str, tool: str, backend: str, seed: int, min_ratio: float, tmp: Path) -> Result:
    """Baseline, the tool's documented stimuli for half a second, the aftermath; recorded and replayed."""
    from kickthefly.core import loadout as lo
    from kickthefly.core import replay

    info = lo.BY_NAME[tool]
    r = Result(id=f"brain:{kind}:{tool}", group="brain", brain=kind, tool=tool)

    def run():
        br = _new_brain(kind, seed, backend)
        real_backend = br.sim.backend.name
        rec = replay.ReplayRecorder(seed=seed, backend=real_backend, dtype="float32", signature={}, arena="room")
        rec.attach(br)
        rec.record(0, "tool", name=tool)                              # the tool choice, replayed as a no-op
        probe = Probe(br, info.probes)
        if tool == "laser":
            probe = Probe(br, (), {"laser target DNp01": _rows(br, "type:DNp01")})
        keys = lo.stimulus_keys(br, tool)
        t0 = time.perf_counter()
        for _ in range(200):
            br.step()
            probe.sample()
        use_step = probe.n
        for i in range(100):
            if i % 10 == 0:
                for k in keys:
                    br.poke(k[0], k[1], 0.8)
                if tool == "laser":
                    from kickthefly.core import simcore
                    simcore.drive(br, _rows(br, "type:DNp01"), 0.5)
            br.step()
            probe.sample()
        if tool == "laser":
            from kickthefly.core import simcore
            simcore.undrive(br, _rows(br, "type:DNp01"))
        for _ in range(100):
            br.step()
            probe.sample()
        wall = time.perf_counter() - t0
        r.metrics["sim_real_ratio"] = round(400 * DT / max(wall, 1e-9), 2)
        r.expect(r.metrics["sim_real_ratio"] >= min_ratio,
                 f"sim/real ratio {r.metrics['sim_real_ratio']} is below the floor {min_ratio}")
        judge_probes(r, probe, use_step, tool)
        if kind == "larva" and tool != "laser":
            r.expect(bool(probe.parts), f"{tool} is offered in larva mode but none of its documented neurons are mapped in the larval brain")
        check_ranges(r, br)
        rec.detach()
        path = tmp / f"{kind}-{tool}.ktfreplay"
        rec.save(path)
        player = replay.ReplayPlayer.load(path)
        br2 = _new_brain(kind, seed, backend)
        sha = player.play(br2)
        same = sha == rec.spike_sha256
        r.metrics["replay_identical"] = bool(same)
        if real_backend in replay.BIT_EXACT_BACKENDS:
            r.expect(same, f"replay did not reproduce the spikes on {real_backend}")
        else:
            r.note(f"replay on {real_backend} is reported, not judged: identical={same}")
        r.metrics["backend"] = real_backend
        del br, br2
        gc.collect()

    guarded(r, run)
    return r


def _rows(br, spec) -> np.ndarray:
    from kickthefly.core import simcore

    return simcore.rows_of(br, spec)


# --- adult game rigs (real games, no window, lockstep) ---------------------------------------------------------------------
class Rig:
    """A real game (2D or 3D) driven frame by frame from this thread: game.update, then a few brain steps."""

    def __init__(self, three_d: bool, backend: str, seed: int = 5, lab: bool = True):
        import pygame

        from kickthefly.core import config
        from kickthefly.game import kick_the_fly as k2

        pygame.init()
        self.k2, self.three_d = k2, three_d
        state = {"seed": seed, "backend": backend, "individuality": "off"}
        k2.load_brain(state)
        if "error" in state:
            raise RuntimeError(state["error"])
        self.cfg = config.Config(None)
        self.cfg.set("brain.mode", "lab" if lab else "play")
        self.cfg.set("brain.individuality", "off")
        if three_d:
            from kickthefly.game import kick3d

            self.k3 = kick3d
            hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
            self.game = kick3d.Game3D(hud, state["brain"], state["view"], state["graph"], state["weights"], cfg=self.cfg)
        else:
            screen = pygame.Surface((k2.W, k2.H))
            self.game = k2.Game(screen, state["brain"], state["view"], state["graph"], state["weights"], cfg=self.cfg)
        self.state = state
        self.probe: Probe | None = None
        self.brain_steps = 0

    # --- lifecycle -----------------------------------------------------------------------------------------------
    def close(self) -> None:
        g = self.game
        g.view_stop = True
        for slot in getattr(g, "flies", []):
            slot.brain.stop()
        for th in threading.enumerate():
            if th.name == "brain-view":
                th.join(timeout=2.0)
        gc.collect()

    @property
    def slot(self):
        return self.game.flies[self.game.focus]

    @property
    def brain(self):
        return self.slot.brain

    def set_arena(self, name: str) -> None:
        self.game.set_setting("brain.arena", name, save=False, force=True)

    def _service(self, br) -> None:
        """What the brain's own thread does on revive(): nothing runs that thread here, so do it."""
        if getattr(br, "_revive", False):
            br._revive = False
            br.death_step = None
            br.sim.p.gain_adapt = br._gain_adapt
            br.sim.gain = br._gain
            for _ in range(300):
                br._step()

    def reset(self) -> None:
        g = self.game
        g.new_fly()
        g.immortal = False
        self._service(g.flies[0].brain)
        self.game.report = None
        self.game.torching = False

    # --- frames -----------------------------------------------------------------------------------------------------
    def frames(self, n: int, mouse=None) -> None:
        g = self.game
        for _ in range(n):
            g.clock.now += FRAME
            if self.three_d:
                g.update3d(g.clock.now, FRAME, {}, (0.0, 0.0))
            else:
                g.update(g.clock.now, mouse if mouse is not None else tuple(self.slot.fly.p[self.k2.HEAD]))
            br = self.brain
            for _ in range(STEPS_PER_FRAME):
                br.step()
                self.brain_steps += 1
                if self.probe is not None:
                    self.probe.sample()

    def seconds(self, s: float, **k) -> None:
        self.frames(max(1, int(round(s / FRAME))), **k)

    # --- using tools ---------------------------------------------------------------------------------------------------
    def face_fly(self) -> None:
        """3D: stand about 1.2 m from the fly, looking at its thorax."""
        g, k2 = self.game, self.k2
        fly = self.slot.fly.p[k2.THX]
        for d in ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)):
            pos = np.array([fly[0] + d[0] * 1.2, fly[2] + d[1] * 1.2])
            if abs(pos[0]) < self.k3.RX - 0.3 and abs(pos[1]) < self.k3.RZ - 0.3:
                break
        g.player.pos = pos.astype(float)
        eye = g.player.eye
        v = fly - eye
        g.player.yaw = math.atan2(v[2], v[0])
        g.player.pitch = math.atan2(v[1], math.hypot(v[0], v[2]))

    def use(self, tool: str) -> None:
        g, k2 = self.game, self.k2
        g.select_tool(tool)
        assert g.tool_name() == tool, f"could not put {tool} in hand"
        if tool == "laser":
            from kickthefly.lab.laser import LaserState

            g.laser_state = LaserState(target_type="dnp01", mode="activate")
        if self.three_d:
            self.face_fly()
            g.use_tool3d(g.clock.now)
            self._after_use(tool)
        else:
            pos = tuple(self.slot.fly.p[k2.HEAD])
            g.use_tool(pos, g.clock.now)
            self._after_use(tool)

    def _after_use(self, tool: str) -> None:
        g, k2 = self.game, self.k2
        if tool in ("decoy",):                                        # drop it in the fly's reach, like a player would
            lst = g.decoys3 if self.three_d else g.decoys
            if lst:
                lst[-1]["p"] = self.slot.fly.p[k2.HEAD].copy()
                if self.three_d:
                    lst[-1]["p"][1] = 0.05
        if tool in ("sugar", "fruit"):
            lst = g.sugars3 if self.three_d else g.sugars
            if lst and self.three_d:
                lst[-1]["p"] = self.slot.fly.p[k2.HEAD].copy() + np.array([0.15, 0, 0])
                lst[-1]["p"][1] = 0.035
        if tool == "alcohol":
            lst = g.alcohols3 if self.three_d else g.alcohols
            if lst and self.three_d:
                lst[-1]["p"] = self.slot.fly.p[k2.HEAD].copy() + np.array([0.15, 0, 0])
                lst[-1]["p"][1] = 0.035

    def step_back(self) -> None:
        """A player drops an item and steps away, so the fly is not spooked by them standing over it."""
        if not self.three_d:
            return
        g = self.game
        eye = g.player.eye
        fly = self.slot.fly.p[self.k2.THX]
        v = np.array([eye[0] - fly[0], eye[2] - fly[2]])
        v = v / (np.linalg.norm(v) or 1.0)
        pos = np.array([fly[0], fly[2]]) + v * 4.0
        pos = np.clip(pos, (-self.k3.RX + 0.3, -self.k3.RZ + 0.3), (self.k3.RX - 0.3, self.k3.RZ - 0.3))
        g.player.pos = pos.astype(float)

    def release(self) -> None:
        g = self.game
        g.torching = False
        if not getattr(self.slot.fly, "wrapped", False):
            self.slot.fly.grabbed = None
        if hasattr(g, "laser_state"):
            g.laser_state.trigger_release()

    def fingerprint(self) -> dict:
        g, br = self.game, self.brain
        return dict(p=np.array(self.slot.fly.p, copy=True), v=np.array(br.sim.v, copy=True), health=float(self.slot.fly.health),
                    tool=int(g.tool), arena=int(g.arena_i), kills=int(g.kills))

    def dead(self) -> bool:
        return any(s.fly.dead for s in self.game.flies)


def _same(a: dict, b: dict) -> list[str]:
    bad = []
    if not np.allclose(a["p"], b["p"], atol=1e-5):
        bad.append("body positions")
    if not np.array_equal(a["v"], b["v"]):
        bad.append("membrane potentials")
    for k in ("health", "tool", "arena", "kills"):
        if a[k] != b[k]:
            bad.append(k)
    return bad


def game_leg(rig: Rig, arena: str, tool: str, min_ratio: float, tmp: Path, save_load: bool = True) -> Result:
    from kickthefly.core import loadout as lo
    from kickthefly.core import savestate
    from kickthefly.game import kick_the_fly as k2

    info = lo.BY_NAME[tool]
    dim = "3d" if rig.three_d else "2d"
    r = Result(id=f"game{dim}:adult:{arena}:{tool}", group=f"game{dim}", brain="adult", arena=arena, tool=tool)

    def run():
        g = rig.game
        rig.set_arena(arena)
        rig.reset()
        r.expect(k2.ARENAS[g.arena_i] == arena, f"arena did not change to {arena}")
        br = rig.brain
        extra = {"laser target DNp01": _rows(br, "type:DNp01")} if tool == "laser" else None
        rig.probe = Probe(br, info.probes, extra)
        rig.seconds(1.5)                                              # calm baseline, sampled
        use_step = rig.probe.n
        rig.brain_steps = 0
        t0 = time.perf_counter()
        rig.use(tool)
        kind, window = TOOL_PLAN[tool]
        if kind == "item":
            rig.step_back()
        spent = 0.0                                                    # wall seconds spent saving and loading
        elapsed, next_click, saved_at = 0.0, 0.8, None
        while elapsed < window and not rig.dead():
            rig.seconds(0.25)
            elapsed += 0.25
            if kind in ("hold", "click") and rig.three_d and tool != "hand":
                rig.face_fly()                                        # a player keeps the tool on the fly as it moves
            if tool == "decoy" and elapsed <= 0.75:
                rig._after_use(tool)                                   # held against its forelegs: contact is what drives LgLG5-8
            if kind == "click" and elapsed >= next_click:
                next_click += 0.8
                rig.use(tool)
            if save_load and saved_at is None and elapsed >= SAVE_LOAD_AT_S:
                saved_at = elapsed
                t1 = time.perf_counter()
                path = tmp / f"{r.id.replace(':', '_')}.ktfsave"
                savestate.save_game(g, path)
                saved = rig.fingerprint()
                rig.seconds(0.5)
                savestate.load_game(g, path)
                bad = _same(saved, rig.fingerprint())
                r.expect(not bad, f"save then load did not restore: {', '.join(bad)}")
                r.metrics["save_load_restored"] = not bad
                try:
                    path.unlink()
                except OSError:
                    pass
                spent += time.perf_counter() - t1
                if kind == "hold" or (tool == "spider" and g.spider is None):
                    rig.use(tool)                                      # a load lets go of the tool: pick it up again
                rig.seconds(0.25)                                      # and it runs on after the load
        rig.release()
        wall = time.perf_counter() - t0 - spent
        ratio = rig.brain_steps * DT / max(wall, 1e-9)
        r.metrics["sim_real_ratio"] = round(ratio, 2)
        r.expect(ratio >= min_ratio, f"sim/real ratio {ratio:.2f} is below the floor {min_ratio}")
        judge_probes(r, rig.probe, use_step, tool)
        rig.probe = None
        fly = rig.slot.fly
        r.expect(0 <= fly.health <= k2.MAX_HEALTH, f"health {fly.health} outside 0..{k2.MAX_HEALTH}")
        r.expect(0 <= rig.slot.pain <= 100 + 1e-6, f"pain {rig.slot.pain} outside 0..100")
        r.expect(bool(np.all(np.isfinite(fly.p))), "body positions are not finite")
        check_ranges(r, rig.brain)
        if rig.dead():
            r.metrics["died"] = True
            rig.probe = None
            rig.seconds(k2.AUTOPSY_DELAY + 1.0)
            r.expect(g.report is not None, "it died but no autopsy report was made")
            if g.report is not None:
                r.expect(isinstance(g.report, dict) and len(g.report) > 0, "the autopsy report is empty")
        else:
            r.metrics["died"] = False
            r.note("not lethal within the run: the autopsy path was not exercised")

    guarded(r, run)
    return r


# --- gates ---------------------------------------------------------------------------------------------------------------
def gate_results(brains: list[str], arenas: list[str], tools: list[str]) -> list[Result]:
    """Every combo that is gated by design, with the message a player is given asserted (not run)."""
    from types import SimpleNamespace

    from kickthefly.core import loadout as lo
    from kickthefly.game import larva
    from kickthefly.ui import loadout_ui

    out = []
    if "larva" in brains:
        for arena in arenas:
            allowed, why = larva.is_arena_allowed_for_larva(arena)
            if not allowed:
                for tool in tools:
                    r = Result(id=f"gate:larva:{arena}:{tool}", group="gate", brain="larva", arena=arena, tool=tool, status=GATED)
                    r.expect(bool(why.strip()), f"larva x {arena} is gated but gives no message")
                    if r.status == GATED:
                        r.note(f'gate message: "{why[:120]}"')
                    out.append(r)
        host = SimpleNamespace(cfg=SimpleNamespace(lab=True), is_larva=True)
        for tool in tools:
            t = lo.BY_NAME[tool]
            if not t.larva:
                r = Result(id=f"gate:larva:*:{tool}", group="gate", brain="larva", tool=tool, status=GATED)
                msg = loadout_ui._why_unavailable(host, t)
                r.expect("larval" in msg.lower(), f"larva x {tool} is hidden but the editor's reason is {msg!r}")
                r.expect(not lo.available(tool, lab=True, larva=True), f"{tool} is still available in larva mode")
                if r.status == GATED:
                    r.note(f'gate message: "{msg[:120]}"')
                out.append(r)
    for tool in tools:
        t = lo.BY_NAME[tool]
        if t.lab_only:
            r = Result(id=f"gate:play:*:{tool}", group="gate", brain="adult", tool=tool, status=GATED)
            msg = loadout_ui._why_unavailable(SimpleNamespace(cfg=SimpleNamespace(lab=False), is_larva=False), t)
            r.expect("Lab mode" in msg, f"{tool} outside Lab: the editor's reason is {msg!r}")
            r.expect(not lo.available(tool, lab=False, larva=False), f"{tool} is available outside Lab mode")
            if r.status == GATED:
                r.note(f'gate message: "{msg}"')
            out.append(r)
    return out


def gate_2d_outdoor(rig2d: Rig) -> list[Result]:
    """The 2D game keeps the outdoor arenas out, and says why."""
    from kickthefly.game import kick_the_fly as k2

    out = []
    for arena in sorted(k2.OUTDOOR_ARENAS):
        r = Result(id=f"gate:2d:{arena}:*", group="gate", brain="adult", arena=arena, status=GATED)

        def run():
            g = rig2d.game
            g.log.clear()
            g.set_setting("brain.arena", arena, save=False, force=True)
            r.expect(k2.ARENAS[g.arena_i] == "room", f"the 2D game switched to {arena}")
            said = [t for _, t, _ in g.log if "needs the 3D game" in t]
            r.expect(bool(said), "the 2D game gave no 'needs the 3D game' message")
            if said:
                r.note(f'gate message: "{said[-1].strip()[:100]}"')

        guarded(r, run)
        out.append(r)
    return out


# --- extras ------------------------------------------------------------------------------------------------------------------
def extra_multi_fly(rig: Rig, r: Result) -> None:
    """Spawn up to 8 flies, run them, then despawn (R): the extra brains stop."""
    g = rig.game
    rig.reset()
    max_f = min(8, g.max_flies)
    t_end = time.time() + 240
    while len(g.flies) < max_f and time.time() < t_end:
        g.spawn_fly()
        n = len(g.flies)
        for _ in range(4000):
            rig.frames(1)
            time.sleep(0.002)
            if len(g.flies) > n or not g._spawning:
                break
    r.metrics["flies"] = len(g.flies)
    r.expect(len(g.flies) == max_f, f"only {len(g.flies)} of {max_f} flies spawned")
    rig.seconds(1.0)
    for slot in g.flies:
        r.expect(bool(np.all(np.isfinite(slot.fly.p))), "a spawned fly has non-finite positions")
    extra = [s.brain for s in g.flies[1:]]
    g.new_fly()
    r.expect(len(g.flies) == 1, f"despawn left {len(g.flies)} flies")
    r.expect(all(getattr(b, "_stop", False) for b in extra), "a despawned fly's brain thread was not stopped")


def extra_surgery(backend: str, r: Result) -> None:
    from kickthefly.lab.api import Fly

    fly = Fly(seed=31, backend=backend, warmup_s=1.0)
    rec = fly.record({"gf": "dnp01", "loom": "loom"})
    fly.step(1.0)
    base = rec.rates(0.0, 1.0)
    fly.silence("type:LPLC2,LC4")
    fly.step(1.0)
    off = rec.rates(1.2, 2.0)["loom"]
    fly.restore()
    fly.step(1.0)
    back = rec.rates(2.2, 3.0)["loom"]
    fly.stimulate("type:LPLC2,LC4")
    fly.step(1.0)
    on = rec.rates(3.2, 4.0)["loom"]
    r.metrics.update(baseline=base["loom"], silenced=off, restored=back, stimulated=on)
    r.expect(off < 0.2 * max(base["loom"], 0.5), f"silencing left {off:.2f} Hz (calm {base['loom']:.2f})")
    r.expect(back > 0.5 * base["loom"], f"restoring gave {back:.2f} Hz (calm {base['loom']:.2f})")
    r.expect(on > 1.5 * max(base["loom"], 0.5), f"stimulating gave {on:.2f} Hz (calm {base['loom']:.2f})")


def extra_training(rig: Rig, r: Result) -> None:
    """Five fear pairings: the mushroom body's memory of the scent must move, and the log holds the pairings."""
    g = rig.game
    rig.reset()
    br = rig.brain
    mem = br.memory
    r.expect(mem is not None, "this brain has no mushroom body memory")
    if mem is None:
        return
    mem.save = lambda: None
    scent = g.train_scent
    g.start_training("fear", trials=5)
    r.expect(g.train is not None, "training did not start")
    g.train_speed = 1.0
    for _ in range(60 * 90):
        rig.frames(1)
        if g.train is None:
            break
    r.expect(g.train is None, "training did not finish in 90 s of game time")
    entries = mem.log.get(scent, [])
    pairings = [e for e in entries if e[1] == "fear"]
    naive = mem.naive_mbon.get(scent)
    hz = [e[4] for e in pairings if e[4] is not None]
    r.metrics.update(pairings=len(pairings), naive_mbon_hz=naive, mbon_hz_per_pairing=hz)
    r.expect(len(pairings) == 5, f"{len(pairings)} pairings were logged, not 5")
    r.expect(naive is not None and len(hz) == 5, "the untrained response and each pairing's response were not measured")
    if naive is not None and len(hz) == 5:
        # what fear training lowers: the firing of the approach-promoting MBONs to the smell alone (memory.mbon_response)
        r.expect(hz[-1] < naive, f"the MBON response to the scent did not fall ({naive:.2f} Hz naive -> {hz[-1]:.2f} Hz after 5 pairings)")
        r.expect(mem.weakened_share() > 0, "no KC -> MBON synapse was weakened")


def extra_duel(rig: Rig, r: Result) -> None:
    g = rig.game
    rig.reset()
    g.toggle_duel()
    r.expect(g.duel, "the duel did not start")
    rig.seconds(4.0)
    r.expect(0 <= g.player_hp <= g.k3.PLAYER_HP if hasattr(g, "k3") else 0 <= g.player_hp, f"player hp {g.player_hp} out of range")
    g.toggle_duel()
    r.expect(not g.duel, "the duel did not end")
    r.expect(len(g.pellets3) == 0, "pellets were left after the duel ended")
    rig.seconds(1.0)


def extra_pet(tmp: Path, r: Result) -> None:
    """Pet mode catch-up over a simulated 3-day gap: deterministic, clamped to 0..1, logs the sleep, idempotent."""
    from kickthefly.core import pet

    class Clock:
        t = 1_800_000_000.0

        def time(self):
            return self.t

    real_time = pet.time
    clock = Clock()
    pet.time = clock
    try:
        results = []
        for run in range(2):                                          # twice: the same gap must give the same pet
            d = tmp / f"pet{run}"
            pm = pet.PetManager(d)
            pm.create_new_pet(seed=77)
            pm.save()
            clock.t += 3 * 86400
            pm2 = pet.PetManager(d)
            pm2.load_or_create()
            results.append((pm2.hunger, pm2.sleep_pressure, pm2.is_dead, len(pm2.timeline), pm2.welcome_message))
            clock.t = 1_800_000_000.0
            pm2.real_stakes = False
        h, sp, dead, n_events, msg = results[0]
        r.metrics.update(hunger=h, sleep_pressure=sp, events=n_events, message=msg)
        r.expect(results[0] == results[1], "the same 3-day gap gave two different pets")
        r.expect(0.0 <= h <= 1.0 and 0.0 <= sp <= 1.0, f"hunger {h} / sleep pressure {sp} out of range")
        r.expect(h > 0.15, f"3 days away did not make it hungrier ({h})")
        r.expect(not dead, "it died without real stakes")
        r.expect(n_events >= 2, "the catch-up left no sleep event in the timeline")
        r.expect("3.0 days" in msg, f"the welcome message doesn't say 3 days: {msg!r}")
        # real stakes: starving for 48 hours or more, hunger full: dead
        d = tmp / "pet-stakes"
        clock.t = 1_800_000_000.0
        pm = pet.PetManager(d)
        pm.create_new_pet(seed=78)
        pm.real_stakes = True
        pm.hunger = 0.99
        pm.save()
        clock.t += 3 * 86400
        pm3 = pet.PetManager(d)
        pm3.load_or_create()
        r.expect(pm3.is_dead, "with real stakes on, three days of neglect should have starved it")
    finally:
        pet.time = real_time


def extra_individuality(backend: str, r: Result) -> None:
    from kickthefly.core import individuality, simcore

    out = {}
    for mode, sigma in (("off", None), ("subtle", 0.05), ("strong", 0.15)):
        br = simcore.new_brain(seed=41, backend=backend, warmup=100, individuality=mode)
        n = br.n
        d_pre, d_post = getattr(br.sim, "d_pre", None), getattr(br.sim, "d_post", None)
        if mode == "off":
            r.expect(d_pre is None or bool(np.all(np.asarray(d_pre) == 1.0)), "off: gains are not all 1")
        else:
            pre, post = np.asarray(d_pre, float), np.asarray(d_post, float)
            r.expect(bool(np.all(pre > 0) and np.all(post > 0)), f"{mode}: a gain is not positive (a sign could flip)")
            s = float(np.std(np.log(pre)))
            r.metrics[f"{mode}_log_sigma"] = round(s, 4)
            r.expect(abs(s - sigma) < 0.2 * sigma, f"{mode}: sigma {s:.3f} is not the documented {sigma}")
        h = hashlib.sha256()
        for _ in range(100):
            br.step()
            h.update(np.packbits(br.sim.spikes).tobytes())
        out[mode] = h.hexdigest()
        del br
    r.expect(len(set(out.values())) == 3, "the three individuality settings did not give three different runs")
    a, b = individuality.compute_fly_gains(41, 2000, mode="subtle", sigma=0.05)
    c, d = individuality.compute_fly_gains(41, 2000, mode="subtle", sigma=0.05)
    r.expect(bool(np.array_equal(a, c) and np.array_equal(b, d)), "gains are not deterministic per seed")
    e, _ = individuality.compute_fly_gains(42, 2000, mode="subtle", sigma=0.05)
    r.expect(not np.array_equal(a, e), "two seeds gave the same gains")


def extra_loadouts(rig: Rig | None, r: Result) -> None:
    from kickthefly.core import config
    from kickthefly.core import loadout as lo

    for preset in lo.CHOICES:
        for mode in ("play", "lab", "pet"):
            for larva in (False, True):
                cfg = config.Config(None)
                cfg.set("brain.mode", mode)
                cfg.set("controls.loadout_preset", preset)
                if preset == "custom":
                    cfg.custom_loadout(["swatter", "sugar", "laser", "bomb"])
                cfg.set("brain.brain", "larva" if larva else "adult")
                ld = lo.resolve(cfg)
                tag = f"{preset}/{mode}/{'larva' if larva else 'adult'}"
                r.expect(ld.tools[0] == lo.HAND, f"{tag}: the hand is not first")
                r.expect(len(set(ld.tools)) == len(ld.tools), f"{tag}: duplicate tools")
                r.expect(all(lo.available(t, lab=cfg.lab, larva=larva) for t in ld.tools), f"{tag}: a gated tool is on the hotbar")
                r.expect("laser" not in ld.tools or mode == "lab", f"{tag}: the laser outside Lab mode")
                r.expect(ld.n_pages == max(1, -(-len(ld.tools) // 10)), f"{tag}: wrong page count")
    if rig is None:
        return
    g = rig.game
    for preset in ("base", "chaos", "chemist", "lab", "all", "pet", "custom"):
        rig.reset()
        g.cfg.set("brain.mode", "lab" if preset in ("lab", "all") else "play")
        if preset == "custom":
            g.cfg.custom_loadout(["swatter", "sugar"])
        g.set_setting("controls.loadout_preset", preset, save=False, force=True)
        for page in range(g.loadout.n_pages):
            g.loadout.page = page
            for slot in range(10):
                name = g.loadout.slot_tool(slot)
                if name is None:
                    r.expect(not g.select_slot(slot), f"{preset}: an empty slot {slot + 1} took a tool")
                    continue
                r.expect(g.select_slot(slot) and g.tool_name() == name, f"{preset}: slot {slot + 1} did not give {name}")
                rig.use(name)
                rig.seconds(0.3)
                rig.release()
                rig.reset()
            hud = rig.game.screen
            g._draw_toolbar(hud)                                     # the hotbar draws with this many tools on this page
    g.cfg.set("brain.mode", "lab")
    g.set_setting("controls.loadout_preset", "auto", save=False, force=True)


def extra_render(rig3d: Rig | None, r: Result) -> None:
    """The 3D scene through the real Renderer into an offscreen framebuffer. No OpenGL here: skipped, not failed."""
    try:
        import moderngl
    except Exception as e:
        r.status = SKIP
        r.note(f"no ModernGL: {e}")
        return
    ctx = None
    for kw in ({"standalone": True, "backend": "egl"}, {"standalone": True}):
        try:
            ctx = moderngl.create_context(**kw)
            break
        except Exception as e:
            err = e
    if ctx is None:
        r.status = SKIP
        r.note(f"no OpenGL context here ({err}): render checks skipped")
        return
    if ctx.version_code < 330:
        r.status = SKIP
        r.note(f"OpenGL {ctx.version_code / 100:.1f} < 3.3: render checks skipped")
        ctx.release()
        return
    from kickthefly.game import kick3d, render3d

    try:
        rd = render3d.Renderer(ctx)
        g = rig3d.game
        rig3d.reset()
        fb = ctx.simple_framebuffer((320, 240))
        seen = []
        for arena in ("room", "orchard"):
            rig3d.set_arena(arena)
            rig3d.reset()
            rig3d.seconds(0.5)
            fb.use()
            ctx.enable(moderngl.DEPTH_TEST)
            lights, clear_col, far = kick3d.scene_setup(g)
            ctx.clear(*clear_col, depth=1.0)
            rig3d.face_fly()
            eye, (f, _, _) = g.player.eye, g.player.basis()
            view = kick3d.look_at(eye, eye + f)
            proj = kick3d.perspective(math.radians(70), 320 / 240, 0.03, far)
            rd.clear()
            g.draw_world(rd, g.clock.now)
            rd.set_scene(view, proj, eye, lights, g.clock.now)
            rd.draw_layer("opaque")
            rd.draw_layer("blend")
            rd.draw_particles()
            img = np.frombuffer(fb.read(components=3), np.uint8).reshape(240, 320, 3)
            seen.append(float(img.std()))
            r.expect(img.std() > 4.0, f"{arena}: the rendered frame is blank (std {img.std():.1f})")
        r.metrics["frame_std"] = seen
        r.note(f"rendered offscreen on {ctx.info.get('GL_RENDERER', '?')}")
    finally:
        ctx.release()


# --- the run -----------------------------------------------------------------------------------------------------------------
def run(brains=("adult", "larva"), out: Path | None = None, quick: bool = False, backend: str = "cpu",
        min_ratio: float | None = None, progress=lambda s: print(s, flush=True), tools=None, arenas=None) -> Report:
    """Run the playthrough. Returns the Report (and writes it when out is given)."""
    from kickthefly.core import loadout as lo
    from kickthefly.core.version import __version__
    from kickthefly.game import kick_the_fly as k2

    min_ratio = float(os.environ.get("KTF_PLAYTHROUGH_MIN_RATIO", DEFAULT_MIN_RATIO)) if min_ratio is None else min_ratio
    all_tools = [t for t in lo.TOOL_NAMES if tools is None or t in tools]
    all_arenas = [a for a in k2.ARENAS if arenas is None or a in arenas]
    rep = Report(dict(app_version=__version__, created=time.strftime("%Y-%m-%d %H:%M:%S"), backend=backend,
                      brains=",".join(brains), quick=quick, min_ratio=min_ratio, arenas=all_arenas, tools=all_tools))

    def add(r: Result, fn=None, *a) -> Result:
        if fn is not None:
            guarded(r, fn, *a, r)
        rep.add(r)
        progress(f"[{r.status.upper():5s}] {r.id} ({r.seconds:.1f}s)" + (f"  {r.failures[0]}" if r.failures else ""))
        return r

    with tempfile.TemporaryDirectory(prefix="ktf-playthrough-") as tmpdir:
        tmp = Path(tmpdir)
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        old_home = os.environ.get("KICK_THE_FLY_HOME")
        os.environ["KICK_THE_FLY_HOME"] = str(tmp / "home")           # never the player's settings, saves or memory
        from kickthefly.core import paths
        paths.reset_cache()
        try:
            from kickthefly.sim import brainpack
            have = []
            for kind in brains:                                       # the larva pack is built locally (docs/larva.md)
                if brainpack.find(brain=kind) is None:
                    add(Result(id=f"skip:{kind}", group="setup", brain=kind, status=SKIP,
                               notes=[f"the {kind} brain pack is not built here: every {kind} combo is skipped"]))
                else:
                    have.append(kind)
            brains = tuple(have)
            for r in gate_results(list(brains), all_arenas, all_tools):
                add(r)
            for kind in brains:
                shared: dict[str, Result] = {}
                for i, tool in enumerate(all_tools):
                    from kickthefly.core import loadout as _lo
                    if kind == "larva" and not _lo.BY_NAME[tool].larva:
                        continue
                    shared[tool] = add(brain_leg(kind, tool, backend, 1000 + i, min_ratio, tmp))
                if kind == "larva":                                   # one row per arena x tool that runs (see the docstring)
                    from kickthefly.game import larva
                    for arena in all_arenas:
                        if not larva.is_arena_allowed_for_larva(arena)[0]:
                            continue
                        for tool, base in shared.items():
                            if quick and arena not in QUICK_ARENAS:
                                continue
                            r = Result(id=f"larva:{arena}:{tool}", group="larva", brain="larva", arena=arena, tool=tool,
                                       status=base.status, seconds=0.0, failures=list(base.failures),
                                       metrics=dict(base.metrics))
                            r.note("brain-level run shared across arenas: the larva has no windowed game yet")
                            rep.add(r)
            if "adult" in brains:
                rig3 = None
                try:
                    rig3 = Rig(True, backend)
                    arenas3 = [a for a in all_arenas if not quick or a in QUICK_ARENAS]
                    for arena in arenas3:
                        for tool in all_tools:
                            if not lo.available(tool, lab=True, larva=False):
                                continue
                            add(game_leg(rig3, arena, tool, min_ratio, tmp, save_load=not quick or arena == "room"))
                    add(Result(id="extra:loadouts", group="extra", brain="adult"), extra_loadouts, rig3)
                    add(Result(id="extra:duel", group="extra", brain="adult"), extra_duel, rig3)
                    add(Result(id="extra:render", group="extra", brain="adult"), extra_render, rig3)
                    add(Result(id="extra:multi-fly-3d", group="extra", brain="adult"), extra_multi_fly, rig3)
                finally:
                    if rig3 is not None:
                        rig3.close()
                rig2 = None
                try:
                    rig2 = Rig(False, backend)
                    for r in gate_2d_outdoor(rig2):
                        add(r)
                    if not quick:
                        for arena in (a for a in all_arenas if a not in k2.OUTDOOR_ARENAS):
                            for tool in all_tools:
                                if lo.available(tool, lab=True, larva=False):
                                    add(game_leg(rig2, arena, tool, min_ratio, tmp, save_load=False))
                    add(Result(id="extra:training", group="extra", brain="adult"), extra_training, rig2)
                finally:
                    if rig2 is not None:
                        rig2.close()
                add(Result(id="extra:surgery", group="extra", brain="adult"), extra_surgery, backend)
                add(Result(id="extra:individuality", group="extra", brain="adult"), extra_individuality, backend)
            add(Result(id="extra:pet-catch-up", group="extra"), extra_pet, tmp)
        finally:
            if old_home is None:
                os.environ.pop("KICK_THE_FLY_HOME", None)
            else:
                os.environ["KICK_THE_FLY_HOME"] = old_home
            paths.reset_cache()
    if out is not None:
        j, m = rep.write(Path(out))
        progress(f"report written to {j} and {m}")
    return rep


def main(args) -> int:
    """--headless --playthrough [adult|larva|all] --out DIR [--playthrough-quick] [--sim-backend NAME]"""
    which = getattr(args, "playthrough", None) or "all"
    brains = ("adult", "larva") if which == "all" else (which,)
    out = Path(args.out) if getattr(args, "out", None) else Path("playthrough-report")
    backend = getattr(args, "sim_backend", None) or "cpu"
    rep = run(brains, out, quick=bool(getattr(args, "playthrough_quick", False)), backend=backend)
    c = rep.counts()
    print(f"playthrough: {c[PASS]} passed, {c[FAIL]} failed, {c[GATED]} gated by design, {c[SKIP]} skipped")
    return 1 if rep.failed() else 0
