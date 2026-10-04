"""Save states: the whole simulation at one moment, in a versioned, platform-independent file.

A .ktfsave file is a zip (numpy .npz) of little-endian arrays plus a JSON header, so a save made on Linux loads on
Windows and the other way round. It holds, for every fly:
  - the LIF sim: membrane potentials, refractory counters, last spikes, synaptic gain, model parameters and the exact
    random-generator state (the noise bank is rebuilt from the seed, so it isn't stored)
  - the brain: rate estimates, calm baselines, history, surgery currents, stimuli still being delivered, death state,
    pain setting and its own random-generator state
  - the mushroom body: the learned Kenyon cell -> MBON weights, calm rates and smell templates
  - the body: every particle position and velocity, health and all timers (stored relative to game time)
plus the arena, surgery settings, Lab parameters, seed, sim speed, sugar piles and (3D) the player.

Not saved: things in flight (bombs, sprays, pellets, particles) and the spider; they are cleared on load.

Loading checks the format name and version, the brain pack (neuron count, synapse count, mushroom body signature)
and 2D vs 3D, and refuses a mismatch with a message instead of loading something wrong.
"""
from __future__ import annotations

import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np

from kickthefly.core.version import __version__

FORMAT = "kick-the-fly-save"
FORMAT_VERSION = 1
SUFFIX = ".ktfsave"
TIME_WORDS = ("until", "ready", "_at", "last_damage", "born")


class SaveError(Exception):
    pass


def _le(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and np.little_endian is False):
        a = a.astype(a.dtype.newbyteorder("<"))
    return a


def _is_time(name: str) -> bool:
    return any(w in name for w in TIME_WORDS)


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    raise TypeError


# --- brain ---------------------------------------------------------------------------------------------------------------
class _Copying(dict):
    """Snapshots copy arrays, so a brain that keeps running can't change what was saved."""

    def __setitem__(self, key, value):
        super().__setitem__(key, np.array(value, copy=True))


def brain_state(brain, prefix: str, arrays: dict) -> dict:
    sim = brain.sim
    if getattr(sim, "backend", None) is not None:
        sim.backend.sync_to_host()
    a = _Copying()
    a[prefix + "v"], a[prefix + "refr"], a[prefix + "spikes"] = sim.v, sim.refr, sim.spikes
    act = sim.activity
    with act._lock:
        a[prefix + "act_rate"], a[prefix + "act_last"] = act._rate.copy(), act._last_spike.copy()
        act_steps, act_count = act.steps, act.active_count
    a[prefix + "fast"], a[prefix + "base"], a[prefix + "hist"] = brain.fast, brain.base, brain.hist
    a[prefix + "override"] = brain.override
    pending = []
    with brain._lock:
        for i, ((region, side), (rows, steps)) in enumerate(brain._pending.items()):
            a[f"{prefix}pend{i}"] = np.asarray(rows, np.int64)
            pending.append([region, side, int(steps)])
    meta = dict(
        seed=int(getattr(brain, "seed", 0)), steps=int(brain.steps), hist_n=int(brain.hist_n),
        last_poke=int(brain.last_poke), death_step=brain.death_step, death_sample=int(brain.death_sample),
        sedation=float(brain.sedation), surgery=bool(brain.surgery), pain_level=int(brain.pain_level),
        recruit=float(brain.recruit), spill=float(brain.spill), gain_adapt=float(brain._gain_adapt),
        saved_gain=float(brain._gain), rng=brain.rng.bit_generator.state, pending=pending,
        sim=dict(gain=float(sim.gain), rng=sim.rng.bit_generator.state, target_p=float(sim.target_p),
                 params={k: _jsonable(v) if not isinstance(v, tuple) else list(v) for k, v in vars(sim.p).items()},
                 act_steps=int(act_steps), act_count=int(act_count)),
    )
    if brain.death_base is not None:
        a[prefix + "death_base"] = brain.death_base
    mem = brain.memory
    if mem is not None:
        with mem.lock:
            a[prefix + "mem_w"], a[prefix + "mem_kc_calm"], a[prefix + "mem_dan_calm"] = mem.w, mem.kc_calm, mem.dan_calm
            names = sorted(mem.templates)
            a[prefix + "mem_templates"] = (np.array([mem.templates[n] for n in names], np.float32) if names
                                           else np.zeros((0, len(mem.kc)), np.float32))
            meta["memory"] = dict(names=names, updates=int(mem.updates), calm_ready=bool(mem.calm_ready),
                                  signature=[float(x) for x in mem.signature], naive_mbon=dict(mem.naive_mbon),
                                  enabled=bool(mem.enabled))
    arrays.update(a)
    return meta


def restore_brain(brain, meta: dict, z, prefix: str) -> None:
    from kickthefly.sim.connectome.sim import LIFParams

    sim = brain.sim
    params = dict(meta["sim"]["params"])
    if "gain_bounds" in params:
        params["gain_bounds"] = tuple(params["gain_bounds"])
    with brain._lock:
        sim.p = LIFParams(**{k: v for k, v in params.items() if k in LIFParams.__dataclass_fields__})
        sim.leak = np.float32(sim.p.dt_ms / sim.p.tau_ms)
        seed = int(meta["seed"])
        brain.seed = seed
        sim.rng = np.random.default_rng(seed)       # rebuild the noise bank exactly as construction did...
        sim._noise = sim.rng.standard_normal(sim.n * 16, dtype=np.float32) * np.float32(sim.p.noise_std)
        sim.rng.bit_generator.state = meta["sim"]["rng"]     # ...then continue the stream from the saved point
        sim.v[:] = z[prefix + "v"]
        sim.refr[:] = z[prefix + "refr"]
        sim.spikes = z[prefix + "spikes"].astype(bool).copy()
        sim.gain = float(meta["sim"]["gain"])
        sim.target_p = float(meta["sim"]["target_p"])
        act = sim.activity
        with act._lock:
            act._rate[:] = z[prefix + "act_rate"]
            act._last_spike[:] = z[prefix + "act_last"]
            act.steps, act.active_count = int(meta["sim"]["act_steps"]), int(meta["sim"]["act_count"])
            act._raster.clear()
        brain.fast[:] = z[prefix + "fast"]
        brain.base[:] = z[prefix + "base"]
        if getattr(sim, "backend", None) is not None:
            sim.backend.sync_from_host()
        brain.hist[:] = z[prefix + "hist"]
        brain.override[:] = z[prefix + "override"]
        brain.surgery = bool(meta["surgery"])
        brain.steps, brain.hist_n, brain.last_poke = int(meta["steps"]), int(meta["hist_n"]), int(meta["last_poke"])
        brain.death_step = meta["death_step"]
        brain.death_sample = int(meta["death_sample"])
        brain.death_base = z[prefix + "death_base"].copy() if (prefix + "death_base") in z else None
        brain.sedation = float(meta["sedation"])
        brain.pain_level, brain.recruit, brain.spill = int(meta["pain_level"]), float(meta["recruit"]), float(meta["spill"])
        brain._gain_adapt, brain._gain = float(meta["gain_adapt"]), float(meta["saved_gain"])
        brain.rng = np.random.default_rng()
        brain.rng.bit_generator.state = meta["rng"]
        brain._pending = {}
        for i, (region, side, steps) in enumerate(meta["pending"]):
            brain._pending[(region, side)] = [z[f"{prefix}pend{i}"].copy(), int(steps)]
        brain._revive = False
        brain.meter_reset = True
    mem, mm = brain.memory, meta.get("memory")
    if mem is not None and mm is not None:
        with mem.lock:
            mem.w = z[prefix + "mem_w"].astype(np.float32).copy()
            mem.kc_calm = z[prefix + "mem_kc_calm"].astype(np.float32).copy()
            mem.dan_calm = z[prefix + "mem_dan_calm"].astype(np.float32).copy()
            mem.templates = {n: t.copy() for n, t in zip(mm["names"], z[prefix + "mem_templates"])}
            mem.updates, mem.calm_ready = int(mm["updates"]), bool(mm["calm_ready"])
            mem.naive_mbon, mem.enabled = dict(mm["naive_mbon"]), bool(mm["enabled"])
            mem._write_back(full=True)
            mem.dirty = True


# --- bodies --------------------------------------------------------------------------------------------------------------
def object_state(obj, now: float, prefix: str, arrays: dict, skip=()) -> dict:
    out = {}
    for name, v in vars(obj).items():
        if name in skip or name.startswith("_"):
            continue
        if isinstance(v, np.ndarray):
            arrays[prefix + name] = v.copy()
            out[name] = {"array": prefix + name}
        elif isinstance(v, dict) and name == "stuck":
            out[name] = {"stuck": [[int(i), [float(x) for x in np.ravel(p)]] for i, p in v.items()]}
        elif isinstance(v, (bool, int, float, str, np.integer, np.floating, np.bool_)) or v is None:
            v = _jsonable(v)
            if _is_time(name) and isinstance(v, float):
                out[name] = {"rel": v - now}
            else:
                out[name] = v
    return out


def restore_object(obj, state: dict, z, now: float) -> None:
    for name, v in state.items():
        if isinstance(v, dict):
            if "array" in v:
                setattr(obj, name, z[v["array"]].copy())
            elif "rel" in v:
                setattr(obj, name, now + v["rel"])
            elif "stuck" in v:
                setattr(obj, name, {int(i): np.array(p) for i, p in v["stuck"]})
        else:
            setattr(obj, name, v)


# --- files ---------------------------------------------------------------------------------------------------------------
def pack_signature(game) -> dict:
    br = game.flies[0].brain
    W = br.sim.W_csr
    mem = br.memory
    return dict(n_neurons=int(br.n), synapses=int(W.nnz),
                memory_signature=[float(x) for x in mem.signature] if mem is not None else None)


def _arena_name(game) -> str:
    from kickthefly.game import kick_the_fly as k2

    return k2.ARENAS[int(game.arena_i)]


def _wiring_meta(game, arrays: dict) -> dict | None:
    """Lab changes to the connectome itself. The flipped rows go in the arrays; the rest is small enough for JSON."""
    w = getattr(game, "wiring", None)
    if w is None or w.is_identity:
        return None
    arrays["wiring_flip_rows"] = np.asarray(w.flip_rows, np.int32)
    return w.as_dict()


def save_game(game, path: Path) -> Path:
    now = game.clock.now
    arrays: dict[str, np.ndarray] = {}
    flies = []
    for i, slot in enumerate(game.flies):
        br = slot.brain
        with br.step_lock:                               # never snapshot a brain mid-step
            bmeta = brain_state(br, f"f{i}_b_", arrays)
        fly_meta = object_state(slot.fly, now, f"f{i}_fly_", arrays, skip=("p_tick",))
        slot_meta = object_state(slot, now, f"f{i}_slot_", arrays, skip=("fly", "brain", "pending_hits", "loom_prev"))
        flies.append(dict(seed=int(slot.seed), primary=bool(slot.primary), brain=bmeta, fly=fly_meta, slot=slot_meta,
                          fly_class=type(slot.fly).__name__))
    b_obj = getattr(getattr(getattr(game, "brain", None), "sim", None), "backend", None)
    backend_name = getattr(b_obj, "name", "unknown")
    device_name = getattr(b_obj, "device", "unknown")
    meta = dict(
        format=FORMAT, format_version=FORMAT_VERSION, app_version=__version__, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        platform=platform.system(), mode="3d" if game.three_d else "2d", seed=int(game.cfg["brain.seed"]),
        brain=getattr(game, "brain_type", getattr(game, "cfg", {}).get("brain.brain", "adult")),
        individuality=getattr(game, "cfg", {}).get("brain.individuality", "subtle"),
        backend=backend_name, device=device_name,
        signature=pack_signature(game), arena_i=int(game.arena_i), arena=_arena_name(game), tool=int(game.tool),
        focus=int(game.focus),
        kills=int(game.kills), immortal=bool(game.immortal), pain_level=int(game.pain_level),
        sim_speed=float(game.clock.scale), lab_params=dict(game.lab_params), surgery_modes=list(game.surgery_modes),
        wiring=_wiring_meta(game, arrays),
        type_ops=dict(game.type_ops), flies=flies, extra=game.save_extra(arrays, now),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: _le(v) for k, v in arrays.items()})
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta))
        zf.writestr("arrays.npz", buf.getvalue())
    tmp.replace(path)
    return path


def read_meta(path: Path) -> dict:
    try:
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("meta.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SaveError(f"not a readable save file ({type(e).__name__})") from None
    if meta.get("format") != FORMAT:
        raise SaveError("not a Kick the Fly save file")
    return meta


def compatible(meta: dict, game) -> str | None:
    """None if this save can load into this game, else a plain-language reason."""
    v = meta.get("format_version")
    if not isinstance(v, int) or v > FORMAT_VERSION:
        return f"made by a newer version ({meta.get('app_version', '?')}); update the game to load it"
    if v < 1:
        return "unsupported save format"
    if meta.get("mode") != ("3d" if game.three_d else "2d"):
        return f"made in the {meta.get('mode', '?').upper()} game"
    meta_brain = meta.get("brain", "adult")
    curr_brain = getattr(game, "brain_type", getattr(game, "cfg", {}).get("brain.brain", "adult"))
    if meta_brain != curr_brain:
        return f"made for the {meta_brain} brain (currently using {curr_brain})"
    sig, mine = meta.get("signature", {}), pack_signature(game)
    if sig.get("n_neurons") != mine["n_neurons"] or sig.get("synapses") != mine["synapses"]:
        return "made with a different brain pack"
    if mine["memory_signature"] and sig.get("memory_signature") and not np.allclose(sig["memory_signature"],
                                                                                    mine["memory_signature"]):
        return "made with a different mushroom body wiring"
    return None


def load_game(game, path: Path) -> dict:
    meta = read_meta(path)
    why = compatible(meta, game)
    if why:
        raise SaveError(f"can't load this save: {why}")
    with zipfile.ZipFile(path) as zf:
        z = dict(np.load(io.BytesIO(zf.read("arrays.npz")), allow_pickle=False))
    now = game.clock.now
    game.prepare_load(len(meta["flies"]), [f["seed"] for f in meta["flies"]])
    # the arena first, so the world's bounds are right before any body is put back. New arenas are only ever appended
    # to ARENAS, so an old save's index still names the same one; saves since 2.7 also carry the name.
    from kickthefly.game import kick_the_fly as k2

    arena = meta.get("arena")
    arena = arena if arena in k2.ARENAS else k2.ARENAS[min(int(meta["arena_i"]), len(k2.ARENAS) - 1)]
    game.cfg.set("brain.arena", arena)
    game.arena_i = k2.ARENAS.index(arena)
    game.on_arena_changed(now)
    for i, (slot, fm) in enumerate(zip(game.flies, meta["flies"])):
        br = slot.brain
        with br.step_lock:
            restore_brain(br, fm["brain"], z, f"f{i}_b_")
        restore_object(slot.fly, fm["fly"], z, now)
        restore_object(slot, fm["slot"], z, now)
        slot.pending_hits, slot.loom_prev = {}, {}
        # 3.0 day 3 review: what holds a fly (your hand, a predator, the vinegar trap) is not saved, but the fly's own `grabbed`
        # part was, so a loaded fly stayed pinned to the hand point by nothing; the trap's perch and trip went the same way
        if not getattr(slot.fly, "wrapped", False):
            slot.fly.grabbed = None
        if getattr(slot.fly, "perch", None) == "vinegar":
            slot.fly.perch = None
        slot.trap_trip = None
        slot.seed = int(fm["seed"])
    game.tool = min(max(int(meta["tool"]), 0), len(k2.TOOLS) - 1)      # by index: the tool list only ever grows
    if hasattr(game, "refresh_loadout"):
        game.refresh_loadout()                                # a tool this mode doesn't allow is put down for the hand
    game.focus = min(int(meta["focus"]), len(game.flies) - 1)
    game.kills = int(meta["kills"])
    game.set_setting("brain.immortal", bool(meta["immortal"]), save=False)
    game.cfg.set("brain.pain_level", int(meta["pain_level"]))
    game.pain_level = int(meta["pain_level"])
    game.set_setting("brain.sim_speed", float(meta["sim_speed"]), save=False)
    for name, v in meta.get("lab_params", {}).items():
        if name in game.lab_params:
            game.set_lab_param(name, float(v))
    saved_modes = list(meta.get("surgery_modes", []))
    if len(saved_modes) < len(game.surgery_modes):
        saved_modes = saved_modes + [0] * (len(game.surgery_modes) - len(saved_modes))
    game.surgery_modes = saved_modes[:len(game.surgery_modes)]
    game.type_ops = dict(meta["type_ops"])
    from kickthefly.sim.wiring import Wiring                  # the connectome the save was made on (Lab)

    want = Wiring.from_dict(meta.get("wiring"), z["wiring_flip_rows"] if "wiring_flip_rows" in z else ())
    if want != getattr(game, "wiring", Wiring()):
        game.set_wiring(want, note=False)
    game.load_extra(meta.get("extra", {}), z, now)
    return meta
