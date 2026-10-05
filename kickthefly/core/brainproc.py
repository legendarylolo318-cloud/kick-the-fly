"""The flies' brains in a process of their own (3.1.0 review: the hardware was idle while the game slowed down).

Why. Every fly's brain steps on a thread at 200 steps/s, and in one Python process those threads and the render thread share one
interpreter lock. With many flies the lock was saturated (about one core's worth of Python) while the GPU and the other cores idled:
16 flies ran at 0.4x real time in the game although the same brains, with no render thread competing, run at 1.46x (`--benchmark`).
Here the brains run in a second process with its own lock, the game process only draws, and the two share what the game reads every
frame through shared memory.

What does not change. The brains are the same `Brain` and `LIFSim` objects, built the same way and stepped by the same `Brain._loop`;
only the process they live in differs. Same seed and same inputs give the same spikes (tests/test_brainproc.py). Headless runs
(--validate, protocols, screens, replays, the selftest's checks) never use this.

How.
  BrainProcess  (game side) starts the server with `spawn`, owns one shared-memory block per fly, and sends commands down a pipe. A
                command that changes the brain (a hit, surgery, a current, kill, revive, speed) is sent without waiting; a question waits
                for its answer, in order after every command sent before it.
  RemoteBrain   (game side) is a `Brain` whose state lives in the server. What the game reads every frame (group rates and baselines,
                step counts, history, death, the focused fly's activity and recent spikes) comes from shared memory; read-only methods
                (level, hz, pain_parts, history...) are Brain's own, run on that data; everything else is forwarded.
  _serve        (server side) builds brains, starts their threads, answers commands, and publishes each brain's state ~400 times a
                second (the activity and spikes only for the fly on screen).
"""
from __future__ import annotations

import itertools
import logging
import math
import multiprocessing
import os
import queue
import threading
import time
import traceback
from concurrent.futures import Future
from multiprocessing import shared_memory
from typing import Any

import numpy as np

log = logging.getLogger("kickthefly")

R_STEPS = 32                     # recent steps of spikes published for the fly on screen (the Neurodex reads 30)
N_SCALARS = 32
# scalar slots
S_SEQ, S_STEPS, S_HIST_N, S_DEATH_STEP, S_SPS, S_LAST_POKE, S_LGLG_F, S_LGLG_B, S_P1_F, S_P1_B, S_DEATH_SAMPLE, S_GAIN, \
    S_STEP_MS, S_BUSY, S_SPIKES, S_STETH, S_VIEW_SEQ, S_RASTER_STEP, S_HEARTBEAT, S_VIEWED, S_SURGERY, S_DRIVING, S_INJECTING, \
    S_SEDATION = range(24)
PUBLISH_S = 0.0025
VIEW_PUBLISH_S = 0.02


LOCAL_ONLY = ("Recording spikes and the patch electrode need the brains in the game process: Settings > Brain > Run the brains in "
              "their own process: Off, then restart (Auto does that when the game starts in Lab mode).")


class Layout:
    """Byte offsets of one fly's block: the same arithmetic on both sides."""

    def __init__(self, n: int, groups: int, hist: int):
        self.n, self.G, self.H = n, groups, hist
        self.nb = (n + 7) // 8
        parts = (("scalars", np.float64, (N_SCALARS,)), ("fast", np.float64, (groups,)), ("base", np.float64, (groups,)),
                 ("death_base", np.float64, (groups,)), ("hist", np.float32, (hist, groups)), ("rates", np.float32, (n,)),
                 ("raster_step", np.int64, (R_STEPS,)), ("raster", np.uint8, (R_STEPS, self.nb)))
        self.fields, off = {}, 0
        for name, dt, shape in parts:
            off = (off + 15) // 16 * 16
            size = int(np.prod(shape)) * np.dtype(dt).itemsize
            self.fields[name] = (off, dt, shape)
            off += size
        self.size = off

    def views(self, buf) -> dict[str, np.ndarray]:
        return {name: np.ndarray(shape, dtype=dt, buffer=buf, offset=off) for name, (off, dt, shape) in self.fields.items()}


def _plain(v, depth: int = 0) -> bool:
    """Whether a value can be sent as itself (a copy) rather than as a reference to an object that stays in the server."""
    if v is None or isinstance(v, (bool, int, float, complex, str, bytes, np.generic, np.ndarray)):
        return True
    if depth > 3:
        return False
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(_plain(k, depth + 1) and _plain(x, depth + 1) for k, x in v.items())
    return False


class _Ref:
    """An answer that is an object living in the server: the game side wraps it in a RemoteObject."""
    __slots__ = ()


class _Method:
    __slots__ = ()


# ============================================================ server ===============================================================
def _serve(conn, init: dict) -> None:
    """The brain process. Runs until the game closes the pipe (or dies: then the pipe breaks and this ends too)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import faulthandler
    import sys
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)       # if this process ever dies in native code, say where
    except Exception:
        pass
    from kickthefly.sim.connectome import backends
    backends.set_auto_policy(init.get("auto_policy", "fastest"))
    from kickthefly.core import simcore
    from kickthefly.game import kick_the_fly as k2
    from kickthefly.sim.connectome.sim import LIFParams, LIFSim

    g, W, _ = simcore.pack(brain=init.get("brain_type", "adult"))
    if init.get("mirror_weights"):
        W = simcore.symmetrize_weights(g, W)
    brains: dict[int, Any] = {}
    shms: dict[int, tuple[Any, dict, Any]] = {}       # fid -> (SharedMemory, views, Layout)
    send_lock = threading.Lock()
    stop = threading.Event()

    def reply(rid, kind, value=None):
        if rid is None:
            return
        with send_lock:
            try:
                conn.send((rid, kind, value))
            except (OSError, EOFError, BrokenPipeError):
                stop.set()

    def resolve(fid, path):
        obj = brains[fid]
        for name in path:
            obj = getattr(obj, name)
        return obj

    def answer(v):
        if _plain(v):
            return "v", v
        if callable(v):
            return "m", None
        return "r", None

    def build(fid, opts):
        p = LIFParams()
        for k in ("backend", "dtype", "individuality"):
            if opts.get(k) is not None:
                setattr(p, k, opts[k])
        seed = int(opts.get("seed", 0))
        sim = LIFSim(None, p, W_in=W, seed=seed)
        if opts.get("lab_params"):
            from kickthefly.lab import lab
            lab.apply_to_sim(sim, opts["lab_params"])
        br = k2.Brain(g, sim, seed=seed)
        br.brain_type = init.get("brain_type", "adult")
        br.graph = g
        if opts.get("pain_level") is not None:
            br.set_pain_level(int(opts["pain_level"]))
        if getattr(g, "dan_mbon", None) is not None and opts.get("memory", True):
            from kickthefly.core import memory
            br.memory = memory.Memory(g, sim)
            if not opts.get("persist_memory", False):
                br.memory.save = lambda: None
        if opts.get("warmup", 600):
            br.warmup(int(opts.get("warmup", 600)))
        return br

    def publish_loop():
        last_save = time.time()
        last_view: dict[int, float] = {}
        hist_done: dict[int, int] = {}
        while not stop.is_set():
            t = time.perf_counter()
            for fid, br in list(brains.items()):
                ent = shms.get(fid)
                if ent is None:
                    continue
                _, v, lay = ent
                sc = v["scalars"]
                try:
                    v["fast"][:] = br.fast
                    v["base"][:] = br.base
                    hn = br.hist_n
                    done = hist_done.get(fid, 0)
                    if hn < done or hn - done > lay.H:
                        done = max(0, hn - lay.H)
                    if hn > done:
                        idx = np.arange(done, hn) % lay.H
                        v["hist"][idx] = br.hist[idx]
                    hist_done[fid] = hn
                    sc[S_STEPS] = br.steps
                    sc[S_HIST_N] = hn
                    sc[S_DEATH_STEP] = -1 if br.death_step is None else br.death_step
                    sc[S_DEATH_SAMPLE] = br.death_sample
                    if br.death_base is not None:
                        v["death_base"][:] = br.death_base
                    sc[S_SPS] = br.steps_per_s
                    sc[S_LAST_POKE] = br.last_poke
                    sc[S_LGLG_F], sc[S_LGLG_B], sc[S_P1_F], sc[S_P1_B] = br.lglg_fast, br.lglg_base, br.p1_fast, br.p1_base
                    sc[S_GAIN] = br.sim.gain
                    sc[S_STEP_MS] = br.sim.last_step_ms
                    sc[S_BUSY] = br.sim.busy_s
                    sc[S_SPIKES] = br.sim.spike_total
                    sc[S_SURGERY], sc[S_DRIVING], sc[S_INJECTING] = br.surgery, br.driving, br.injecting
                    sc[S_SEDATION] = br.sedation
                    if br.stethoscope_indices is not None:
                        sc[S_STETH] += br.take_stethoscope_spikes()
                    if time.time() - sc[S_VIEWED] < 1.0 and t - last_view.get(fid, 0.0) >= VIEW_PUBLISH_S:
                        last_view[fid] = t
                        act = br.sim.activity
                        v["rates"][:] = act.rates()
                        ras = act.raster()[-R_STEPS:]
                        steps = br.steps
                        v["raster_step"][:] = -1
                        k0 = R_STEPS - len(ras)
                        for k, idx in enumerate(ras):
                            bits = np.zeros(lay.n, bool)
                            bits[idx] = True
                            v["raster"][k0 + k] = np.packbits(bits)
                            v["raster_step"][k0 + k] = steps - len(ras) + k
                        sc[S_RASTER_STEP] = steps
                        sc[S_VIEW_SEQ] += 1
                    sc[S_HEARTBEAT] = time.time()
                    sc[S_SEQ] += 1
                except Exception:
                    log.exception("publishing fly %d failed", fid)
            if time.time() - last_save > 30:                     # the game used to autosave the player's fly's memory; it lives here now
                last_save = time.time()
                for br in list(brains.values()):
                    mem = br.memory
                    if mem is not None and getattr(mem, "dirty", False):
                        threading.Thread(target=mem.save, daemon=True).start()
            time.sleep(max(0.0, PUBLISH_S - (time.perf_counter() - t)))

    def handle(msg):
        rid, op, fid, args = msg
        try:
            if op == "new":
                opts, shm_name, dims = args
                lay = Layout(*dims)
                shm = shared_memory.SharedMemory(name=shm_name)
                br = build(fid, opts)
                shms[fid] = (shm, lay.views(shm.buf), lay)
                brains[fid] = br
                reply(rid, "v", dict(backend=br.sim.backend.name, device=getattr(br.sim.backend, "device_name", ""),
                                     seed=br.seed, steps=br.steps))
            elif op == "del":
                br = brains.pop(fid, None)
                if br is not None:
                    br.stop()
                    if br.memory is not None and getattr(br.memory, "dirty", False):
                        br.memory.save()                          # a no-op for every fly but the player's own
                ent = shms.pop(fid, None)
                if ent is not None:
                    try:
                        ent[0].close()
                    except Exception:
                        pass
                reply(rid, "v", None)
            elif op == "get":
                path, name = args
                reply(rid, *answer(getattr(resolve(fid, path), name)))
            elif op == "set":
                path, name, value = args
                setattr(resolve(fid, path), name, value)
                reply(rid, "v", None)
            elif op == "setitem":
                path, name, key, value = args
                getattr(resolve(fid, path), name)[key] = value
                reply(rid, "v", None)
            elif op == "call":
                path, name, a, kw = args
                reply(rid, *answer(getattr(resolve(fid, path), name)(*a, **kw)))
            elif op == "run":
                modname, fn, a, kw = args
                import importlib
                f = getattr(importlib.import_module(modname), fn)
                reply(rid, "v", f(brains[fid], *a, **kw))          # a run's answer is data: sent as it is
            else:
                raise ValueError(f"unknown brain-process command {op!r}")
        except Exception as e:
            if rid is None:
                log.warning("brain process: %s on fly %s failed: %s", op, fid, "".join(traceback.format_exception_only(type(e), e)).strip())
            reply(rid, "e", f"{type(e).__name__}: {e}")

    threading.Thread(target=publish_loop, name="publish", daemon=True).start()
    builders = queue.Queue()

    def build_worker():                   # builds run here so a new fly's warm-up never holds up another fly's commands
        while True:
            msg = builders.get()
            if msg is None:
                return
            handle(msg)

    threading.Thread(target=build_worker, name="builder", daemon=True).start()
    with send_lock:
        conn.send((None, "ready", os.getpid()))
    while not stop.is_set():
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        if msg[1] == "new":
            builders.put(msg)
        else:
            handle(msg)
    stop.set()
    for br in brains.values():
        try:
            br.stop()
            if br.memory is not None and getattr(br.memory, "dirty", False):
                br.memory.save()
        except Exception:
            pass


# ============================================================ game side =============================================================
class BrainProcess:
    """The game's handle on the brain server: one per game, started on first use."""

    def __init__(self, brain_type: str = "adult", mirror_weights: bool = False, auto_policy: str = "fastest"):
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_serve, args=(child, dict(brain_type=brain_type, mirror_weights=mirror_weights,
                                                                  auto_policy=auto_policy)), name="kickthefly-brains", daemon=True)
        self.proc.start()
        child.close()
        self._send_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._fids = itertools.count(1)
        self._waiting: dict[int, Future] = {}
        self._wait_lock = threading.Lock()
        self.ready = Future()
        self.sync_calls = 0
        self.closed = False
        self.trace = None
        if os.environ.get("KTF_RPC_TRACE"):                  # debugging: which questions wait for the brain process, and how often
            import atexit
            import collections
            self.trace = collections.Counter()
            atexit.register(lambda: log.warning("brain-process waits: %s", self.trace.most_common(25)))
        threading.Thread(target=self._recv_loop, name="brainproc-recv", daemon=True).start()
        self.pid = self.ready.result(timeout=120)

    def _recv_loop(self) -> None:
        while True:
            try:
                rid, kind, value = self.conn.recv()
            except (EOFError, OSError):
                break
            if rid is None:
                if kind == "ready":
                    self.ready.set_result(value)
                continue
            with self._wait_lock:
                fut = self._waiting.pop(rid, None)
            if callable(fut):
                try:
                    fut(kind, value)
                except Exception:
                    log.exception("a brain-process reply callback failed")
            elif fut is not None:
                fut.set_result((kind, value))
        self.closed = True
        with self._wait_lock:
            for fut in self._waiting.values():
                if not callable(fut):
                    fut.set_result(("e", "the brain process ended"))
            self._waiting.clear()
        if not self.ready.done():
            self.ready.set_exception(RuntimeError("the brain process did not start"))

    def send(self, op: str, fid: int, *args, wait: bool = True, timeout: float | None = 60.0, callback=None):
        """Send a command. wait: block for the answer (in order after everything sent before); callback(kind, value): get the answer
        later on the receiving thread instead; neither: fire and forget."""
        if self.closed:
            if not wait:                              # a command to a brain that is gone: dropped (said once), never a crash mid-game
                if not getattr(self, "_said_gone", False):
                    self._said_gone = True
                    log.error("the brain process has ended (exit code %s); the flies' brains have stopped", self.proc.exitcode)
                return None
            raise RuntimeError("the brain process has ended")
        rid = next(self._ids) if (wait or callback) else None
        fut = None
        if callback is not None:
            with self._wait_lock:
                self._waiting[rid] = callback
            wait = False
        elif wait:
            if self.trace is not None:
                self.trace[(op, args[1] if op in ("get", "call") and len(args) > 1 else args[:1] if op == "run" else "")] += 1
            fut = Future()
            with self._wait_lock:
                self._waiting[rid] = fut
            self.sync_calls += 1
        with self._send_lock:
            self.conn.send((rid, op, fid, args))
        if not wait:
            return None
        kind, value = fut.result(timeout=timeout)
        if kind == "e":
            if str(value).startswith("AttributeError"):
                raise AttributeError(value)
            raise RuntimeError(f"brain process: {value}")
        return kind, value

    def new_brain(self, template, opts: dict, view=None) -> "RemoteBrain":
        fid = next(self._fids)
        lay = Layout(template.n, len(template.names), template.hist.shape[0])
        shm = shared_memory.SharedMemory(create=True, size=lay.size)
        np.frombuffer(shm.buf, np.uint8)[:] = 0
        kind, info = self.send("new", fid, opts, shm.name, (template.n, len(template.names), template.hist.shape[0]), timeout=600)
        return RemoteBrain(self, fid, shm, lay, template, info)

    def close(self) -> None:
        if self.closed:
            return
        try:
            with self._send_lock:
                self.conn.send(None)
        except Exception:
            pass
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()
        self.closed = True


class _StubSim:
    """Just enough of a LIFSim for Brain.__init__ to compute a brain's fixed tables (groups, sense rows) on the game side."""

    def __init__(self, n: int, weights):
        from kickthefly.sim.connectome.sim import LIFParams

        self.p = LIFParams()
        self.gain = float(self.p.syn_gain)
        self.n = n
        self.W_csr = weights


def make_template(g, weights):
    """The fixed tables every brain of this pack shares (Brain.__init__ on a stub sim, ~1 s): RemoteBrains copy them."""
    from kickthefly.game import kick_the_fly as k2

    t = k2.Brain(g, _StubSim(g.n, weights), seed=0)
    t._weights = weights
    return t


# --- run in the brain process by RemoteBrain.run / RemoteMemory -------------------------------------------------------------------------
def _observe_now(br, scent):
    if br.memory is not None:
        br.memory.observe(scent, br.sim.activity.rates())


def _memory_no_save(br):
    if br.memory is not None:
        br.memory.save = lambda: None


def _memory_of(br, scent):
    return (0.0, 0.0) if br.memory is None else tuple(float(x) for x in br.memory.memory_of(scent))


# attributes computed in Brain.__init__ from the graph alone: the same for every fly, so RemoteBrain copies the template's
_STATIC = ("det_id", "pop_id", "names", "sense", "n_det", "col", "g_size", "kc", "brain_type", "is_larva", "subclass", "noci_idx",
           "chord_idx", "goro_idx", "lglg_indices", "p1_indices", "types", "superclass", "instance", "body_id", "region", "n", "dt",
           "k_fast", "k_base", "_assay_groups")
# set on the game side, kept there too, and forwarded
_FORWARD = {"sedation", "speed", "meter_reset", "driving", "surgery", "injecting", "_laser_rows", "_gain_adapt", "_gain", "recruit", "spill",
            "pain_level", "seed", "graph", "_revive", "last_poke", "death_base", "death_sample", "death_step"}


class _SyncedArray(np.ndarray):
    """A RemoteBrain's surgery and drive currents (override, drive_cur): kept here, and every write is sent to the brain process too (the
    laser and the Lab write into them in place)."""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        sync = getattr(self, "_sync", None)
        if sync is not None:
            sync(key, value)

    def __getitem__(self, key):
        return np.asarray(super().__getitem__(key))

    def __array_finalize__(self, obj):
        self._sync = None


def _synced(n: int, rb: "RemoteBrain", name: str) -> _SyncedArray:
    a = np.zeros(n, np.float32).view(_SyncedArray)

    def sync(key, value):
        if isinstance(key, np.ndarray) or isinstance(value, np.ndarray):
            key, value = (np.asarray(key) if isinstance(key, np.ndarray) else key), np.asarray(value)
        rb._proc.send("setitem", rb._fid, (), name, key, value, wait=False)
    a._sync = sync
    return a


class RemoteActivity:
    """sim.activity of a RemoteBrain: rates() and raster() from shared memory (published while this fly is on screen)."""

    def __init__(self, rb: "RemoteBrain"):
        self._rb = rb

    def _want(self) -> None:
        sc = self._rb._v["scalars"]
        first = time.time() - sc[S_VIEWED] > 1.0
        sc[S_VIEWED] = time.time()                     # published while asked for within the last second (the fly on screen)
        if first:
            t_end = time.monotonic() + 0.5             # give the server a moment to publish
            seq0 = sc[S_VIEW_SEQ]
            while sc[S_VIEW_SEQ] == seq0 and time.monotonic() < t_end:
                time.sleep(0.005)

    def rates(self) -> np.ndarray:
        self._want()
        return self._rb._v["rates"].copy()

    def raster(self) -> "_LazyRaster":
        self._want()
        return _LazyRaster(self._rb)

    def __getattr__(self, name):
        return RemoteObject(self._rb, ("sim", "activity"))._get(name)


class _LazyRaster:
    """The recent steps' spiking neurons (a list of index arrays, oldest first), decoded only when indexed."""

    def __init__(self, rb: "RemoteBrain"):
        v = rb._v
        ok = np.flatnonzero(v["raster_step"] >= 0)
        self._bits = v["raster"][ok].copy()
        self._n = rb._lay.n

    def __len__(self) -> int:
        return len(self._bits)

    def _one(self, i: int) -> np.ndarray:
        return np.flatnonzero(np.unpackbits(self._bits[i], count=self._n)).astype(np.int64)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self._one(i) for i in range(len(self._bits))[k]]
        return self._one(range(len(self._bits))[k])

    def __iter__(self):
        return (self._one(i) for i in range(len(self._bits)))

    def __bool__(self) -> bool:
        return len(self._bits) > 0


class RemoteObject:
    """An object that lives in the brain process (sim.p, sim.backend, memory...): attribute reads, writes and calls are forwarded."""

    def __init__(self, rb: "RemoteBrain", path: tuple):
        object.__setattr__(self, "_rb", rb)
        object.__setattr__(self, "_path", path)

    def _get(self, name):
        kind, value = self._rb._proc.send("get", self._rb._fid, self._path, name)
        if kind == "v":
            return value
        if kind == "m":
            return lambda *a, **kw: self._rb._call(self._path, name, *a, **kw)
        return RemoteObject(self._rb, self._path + (name,))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._get(name)

    def __setattr__(self, name, value):
        self._rb._proc.send("set", self._rb._fid, self._path, name, value, wait=False)     # in order with everything else; no wait


class RemoteSim:
    """brain.sim of a RemoteBrain. The weights are the game's own copy of the pack's matrix (what every brain starts from; a fly's
    learned KC -> MBON synapses live in its own process)."""

    def __init__(self, rb: "RemoteBrain", weights, info: dict):
        object.__setattr__(self, "_rb", rb)
        object.__setattr__(self, "activity", RemoteActivity(rb))
        object.__setattr__(self, "W_csr", weights)
        object.__setattr__(self, "_csc", None)
        object.__setattr__(self, "n", rb.n)
        object.__setattr__(self, "backend", _BackendInfo(rb, info))

    @property
    def W_csc(self):
        if self._csc is None:
            object.__setattr__(self, "_csc", self.W_csr.tocsc())
        return self._csc

    gain = property(lambda self: float(self._rb._v["scalars"][S_GAIN]))
    last_step_ms = property(lambda self: float(self._rb._v["scalars"][S_STEP_MS]))
    busy_s = property(lambda self: float(self._rb._v["scalars"][S_BUSY]))
    spike_total = property(lambda self: int(self._rb._v["scalars"][S_SPIKES]))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return RemoteObject(self._rb, ("sim",))._get(name)

    def __setattr__(self, name, value):
        if name in ("W_csr", "W_csc"):
            raise RuntimeError("the matrix of a brain in the brain process is changed with RemoteBrain.run()")
        RemoteObject(self._rb, ("sim",)).__setattr__(name, value)


class _BackendInfo:
    def __init__(self, rb, info):
        self._rb, self.name, self.device_name = rb, info.get("backend", "?"), info.get("device", "")
        self.device = self.device_name

    def refresh(self) -> None:
        obj = RemoteObject(self._rb, ("sim", "backend"))
        self.name, self.device_name = obj._get("name"), obj._get("device_name")
        self.device = self.device_name

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return RemoteObject(self._rb, ("sim", "backend"))._get(name)


def _mk_view(name):
    return property(lambda self: self._v[name])


def _mk_scalar(slot, kind=float):
    return property(lambda self: kind(self._v["scalars"][slot]))


class RemoteBrain:
    """Stands in for a `Brain` whose state lives in the brain process. Constructed by BrainProcess.new_brain."""
    remote = True

    def __init__(self, proc: BrainProcess, fid: int, shm, lay: Layout, template, info: dict):
        from kickthefly.game import kick_the_fly as k2
        d = self.__dict__
        d["_proc"], d["_fid"], d["_shm"], d["_lay"] = proc, fid, shm, lay
        d["_v"] = lay.views(shm.buf)
        for name in _STATIC:
            if name in template.__dict__:
                d[name] = template.__dict__[name]
        d["seed"] = int(info.get("seed", 0))
        d["override"] = _synced(template.n, self, "override")
        d["drive_cur"] = _synced(template.n, self, "drive_cur")
        d["_missing"] = set()
        d["_steth_taken"] = 0.0
        d["_mem"] = None
        d["sedation"] = 0.0
        d["speed"] = 1.0
        d["surgery"] = d["driving"] = d["injecting"] = False
        d["pain_level"], d["recruit"], d["spill"] = 0, 0.3, 0.0
        d["sim"] = RemoteSim(self, getattr(template, "_weights", None), info)
        d["_k2"] = k2

    # --- what the game reads every frame, from shared memory ---------------------------------------------------------------------
    fast, base, hist, death_base = _mk_view("fast"), _mk_view("base"), _mk_view("hist"), _mk_view("death_base")
    steps, hist_n = _mk_scalar(S_STEPS, int), _mk_scalar(S_HIST_N, int)
    steps_per_s, last_poke = _mk_scalar(S_SPS), _mk_scalar(S_LAST_POKE, int)
    lglg_fast, lglg_base, p1_fast, p1_base = (_mk_scalar(s) for s in (S_LGLG_F, S_LGLG_B, S_P1_F, S_P1_B))
    death_sample = _mk_scalar(S_DEATH_SAMPLE, int)

    @property
    def death_step(self):
        v = self._v["scalars"][S_DEATH_STEP]
        return None if v < 0 else int(v)

    @property
    def dead(self) -> bool:
        return self.death_step is not None

    @property
    def stethoscope_spikes(self) -> int:
        return int(self._v["scalars"][S_STETH] - self._steth_taken)

    # Brain's read-only methods, run here on the shared data
    def level(self, name):
        return self._k2.Brain.level(self, name)

    def hz(self, name):
        return self._k2.Brain.hz(self, name)

    def lglg_level(self):
        return self._k2.Brain.lglg_level(self)

    def p1_level(self):
        return self._k2.Brain.p1_level(self)

    def pain_parts(self, rates, base):
        return self._k2.Brain.pain_parts(self, rates, base)

    def pain_neurons(self, level=None):
        return self._k2.Brain.pain_neurons(self, level)

    def history(self, first, last):
        return self._k2.Brain.history(self, first, last)

    @staticmethod
    def hold_alarm(parts, decay=0.95):
        from kickthefly.game import kick_the_fly as k2
        return k2.Brain.hold_alarm(parts, decay)

    @staticmethod
    def pain_index(parts):
        from kickthefly.game import kick_the_fly as k2
        return k2.Brain.pain_index(parts)

    # --- forwarded -------------------------------------------------------------------------------------------------------------------
    def _call(self, path, name, *a, **kw):
        kind, value = self._proc.send("call", self._fid, path, name, a, kw)
        return RemoteObject(self, path + (name,)) if kind == "r" else value

    def _cast(self, name, *a, **kw):
        """A call whose answer the game does not need: sent without waiting."""
        self._proc.send("call", self._fid, (), name, a, kw, wait=False)

    def poke(self, region, side, strength, recruit=None):
        if not self.dead:
            self._cast("poke", region, side, strength, recruit)

    def set_override(self, rows, mode):
        np.ndarray.__setitem__(self.override, rows, self._k2.SURGERY_CURRENT[mode])    # the brain process does its own
        self.__dict__["surgery"] = bool(np.any(self.override))
        self._cast("set_override", np.asarray(rows), mode)

    def clear_overrides(self):
        np.ndarray.__setitem__(self.override, slice(None), 0)
        self.__dict__["surgery"] = False
        self._cast("clear_overrides")

    def set_current(self, name, rows, values):
        self._cast("set_current", name, np.asarray(rows), values)

    def clear_current(self, name):
        self._cast("clear_current", name)

    def set_pain_level(self, level):
        _, recruit, spill = self._k2.PAIN_LEVELS[level]
        self.__dict__.update(pain_level=level, recruit=recruit, spill=spill)
        self._cast("set_pain_level", level)

    def kill(self):
        self._cast("kill")

    def revive(self):
        self._cast("revive")

    def reseed(self, seed):
        self.__dict__["seed"] = seed
        self._cast("reseed", seed)

    def request_steps(self, n):
        self._cast("request_steps", n)

    def set_stethoscope_indices(self, indices):
        self.__dict__["_steth_taken"] = float(self._v["scalars"][S_STETH])
        self._cast("set_stethoscope_indices", indices)

    def take_stethoscope_spikes(self) -> int:
        total = float(self._v["scalars"][S_STETH])
        n = int(total - self._steth_taken)
        self.__dict__["_steth_taken"] = total
        return n

    def start(self):
        self._cast("start")

    def stop(self):
        """A stopped brain is never started again in the game, so this also frees it in the brain process and its shared memory."""
        if self.__dict__.get("_closed"):
            return
        self.__dict__["_closed"] = True
        try:
            self._proc.send("del", self._fid, wait=False)
        except Exception:
            pass
        try:
            self._shm.unlink()                    # the name goes now; the memory when the last view of it does
        except Exception:
            pass

    def warmup(self, steps=600):
        self._call((), "warmup", steps)

    def step(self):
        self._call((), "step")

    def _step(self):
        self._call((), "_step")

    def run(self, module: str, fn: str, *a, **kw):
        """Run `module.fn(brain, *a, **kw)` in the brain process on the real brain (save states, wiring, replays...)."""
        kind, value = self._proc.send("run", self._fid, module, fn, a, kw, timeout=600)
        self.sim.backend.refresh()                     # a run may have changed the engine (Settings > Brain)
        return value if kind == "v" else None

    def refresh_mirrors(self) -> None:
        """Read back what this side keeps a copy of, after the brain process changed it wholesale (a loaded save state)."""
        d = self.__dict__
        for name in ("sedation", "speed", "surgery", "driving", "injecting", "seed", "pain_level", "recruit", "spill"):
            kind, value = self._proc.send("get", self._fid, (), name)
            d[name] = value
        for name in ("override", "drive_cur"):
            kind, value = self._proc.send("get", self._fid, (), name)
            np.ndarray.__setitem__(d[name], slice(None), value)
        d["_steth_taken"] = float(self._v["scalars"][S_STETH])

    @property
    def memory(self):
        if self.__dict__.get("_closed"):
            return None
        if self._mem is None:
            kind, _ = self._proc.send("get", self._fid, (), "memory")
            if kind == "v":                                       # None: this brain has no mushroom body
                return None
            self.__dict__["_mem"] = RemoteMemory(self)
        return self._mem

    def close(self):
        self.stop()

    @property
    def step_lock(self):
        import contextlib
        return contextlib.nullcontext()             # the brain process takes the real lock around what it does

    # --- everything else ---------------------------------------------------------------------------------------------------------------
    @property
    def _stop(self) -> bool:
        return bool(self.__dict__.get("_closed"))

    def __getattr__(self, name):
        if name.startswith("__") or name in self.__dict__.get("_missing", ()) or self.__dict__.get("_closed"):
            raise AttributeError(name)            # (a stopped brain is gone from the brain process: nothing left to ask)
        try:
            return RemoteObject(self, ())._get(name)
        except AttributeError:
            self.__dict__["_missing"].add(name)        # asked for every frame with a default (getattr(brain, "_laser_rows", [])): ask once
            raise

    def __setattr__(self, name, value):
        if name in ("recorder", "probe") and value is not None:
            raise RuntimeError(LOCAL_ONLY)
        self.__dict__.get("_missing", set()).discard(name)
        cls_attr = getattr(type(self), name, None)
        if isinstance(cls_attr, property) and cls_attr.fset is None and name not in _FORWARD:
            raise AttributeError(f"{name} of a brain in the brain process is set there (RemoteBrain.run)")
        if name in _FORWARD:
            if not isinstance(cls_attr, property):
                self.__dict__[name] = value
            self._proc.send("set", self._fid, (), name, value, wait=False)
        else:
            self.__dict__[name] = value


class _SyncedDict(dict):
    """A copy of a dict that lives in the brain process; item writes go there too."""

    def __init__(self, data, rb, path, name):
        super().__init__(data)
        self._rb, self._path, self._name = rb, path, name

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._rb._proc.send("setitem", self._rb._fid, self._path, self._name, key, value, wait=False)


class RemoteMemory:
    """brain.memory of a RemoteBrain: the mushroom body learns in the brain process. memory_of() answers from the last reply and asks
    again without waiting, so a frame never waits for the other process; everything else is forwarded."""

    def __init__(self, rb: RemoteBrain):
        self.__dict__.update(_rb=rb, _obj=RemoteObject(rb, ("memory",)), _last={}, _pending={})

    def observe(self, scent, rates=None):
        self._rb._proc.send("run", self._rb._fid, "kickthefly.core.brainproc", "_observe_now", (scent,), {}, wait=False)

    def memory_of(self, scent):
        if not self._pending.get(scent) and not self._rb._proc.closed:
            self._pending[scent] = True

            def got(kind, value, scent=scent):
                self._pending[scent] = False
                if kind == "v" and value is not None:
                    self._last[scent] = value
            self._rb._proc.send("run", self._rb._fid, "kickthefly.core.brainproc", "_memory_of", (scent,), {}, callback=got)
        return self._last.get(scent, (0.0, 0.0))

    dirty = False                                # saved by the brain process itself (every 30 s, and when the fly goes)

    def __getattr__(self, name):
        v = self._obj._get(name)
        if isinstance(v, dict):                    # mem.naive_mbon[scent] = hz (training) must reach the real memory
            return _SyncedDict(v, self._rb, ("memory",), name)
        return v

    def __setattr__(self, name, value):
        if name == "save" and callable(value):
            # `memory.save = lambda: None` is how the game and its tests stop a memory writing the player's file: done there instead
            self._rb.run("kickthefly.core.brainproc", "_memory_no_save")
            return
        if callable(value):
            raise TypeError(f"memory.{name}: a function cannot be handed to a brain in the brain process")
        self._obj.__setattr__(name, value)


def _spike_digest(br, steps: int = 600, schedule=((50, "head", "L", 0.8), (200, "smell", None, 0.6), (350, "legs", "R", 1.0),
                                                     (500, "taste", None, 0.7))) -> dict:
    """Step a brain `steps` times with fixed stimuli and digest every step's spikes (tests/test_brainproc.py: the same in either process)."""
    import hashlib

    h = hashlib.sha256()
    total = 0
    for t in range(steps):
        for at, region, side, strength in schedule:
            if t == at:
                br.poke(region, side, strength)
        br._step()
        sp = np.asarray(br.sim.spikes, bool)
        h.update(np.packbits(sp).tobytes())
        total += int(sp.sum())
    return {"sha256": h.hexdigest(), "spikes": total, "steps": br.steps, "fast": [float(x) for x in br.fast[:8]]}


def _override_at(br, rows) -> list:
    return [float(x) for x in br.override[rows]]


def _get(br, name):
    return getattr(br, name)
