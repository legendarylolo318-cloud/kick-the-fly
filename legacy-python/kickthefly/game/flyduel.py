"""Fly vs fly duel (3.0 day 4): the existing 1v1 duel with a second brain in place of you. Nothing about who wins is scripted.

The 3D game's duel (kick3d.py: _duel_senses, _duel_motor, _fly_shoot) gives one fly a blaster and a view of YOU. Here both sides
are brains, stepped together in lockstep, and each sees the other the way it sees you:

  CONNECTOME   what the neurons do with what they are given. A fly's turning is DNa01/02 right minus left firing; its shooting is
               DNp35/DNpe052 firing above THRESH["fire"]; its dodge is the giant fiber DNp01 above THRESH["escape"]; its run is
               the body-touch group above THRESH["run"]. Which of those fire, and when, is decided by the wiring, the fly's own
               individuality gains (core/individuality.py) and the noise: that is the part nobody scripted.
  GAME RULE    everything around them: the arena (a flat square), where the flies start, how a turning command becomes a
               heading change, the blaster (cooldown, speed, spread, damage), a pellet's size for looming, what a hit does
               (a touch poke on the body and a punishment pulse on the one hit, a reward pulse for the shooter, exactly as in the
               duel against you), dodge and run distances, the match length and the tie-break. How an opponent becomes sensory
               input (LC10 tracking on the side it is on, small-object detectors when it is dead ahead, looming from its pellets)
               is the duel's own rule (kick3d._duel_senses) applied to the other fly.
  MODEL PREDICTION  who wins. It is a prediction of this model, with its rules and its noise, not a measurement of real flies.

The engine takes two brain-like objects (poke, level, hz, _step, sim.spikes, types) so it can be tested without the real pack.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

# --- GAME RULE constants (the blaster's are the 3D duel's own: kick3d.py) ----------------------------------------------------
HP = 100.0
TICK_STEPS = 4                       # brain steps (5 ms) per engine tick: 20 ms
ARENA_HALF = 6.0                     # m: a flat square, 12 m a side
START_APART = 6.0                    # m between the flies at the start
START_YAW_JITTER = 1.2               # rad: each fly starts facing the other plus or minus this (match noise)
FLY_RADIUS = 0.32
PELLET_SPEED, PELLET_SPREAD, PELLET_DAMAGE = 9.0, 0.035, 9.0
PELLET_LIFE_S, PELLET_RADIUS = 1.6, 0.10
FIRE_COOLDOWN, REWARD_PULSE_S, PUNISH_PULSE_S = 0.45, 0.45, 0.6
STEER_DEADZONE_HZ, STEER_GAIN, STEER_MAX, TRACK_SCALE = 1.5, 0.004, 0.035, 0.45   # per 1/60 s frame, as in the 3D duel
WALK_SPEED = 0.5                     # m/s while DNp09 is above its threshold
DODGE_HOP, DODGE_COOLDOWN = 0.9, 1.2
RUN_HOP, RUN_COOLDOWN = 0.7, 1.5
SEE_RANGE = 7.5                      # m: the duel's own range
DEFAULT_SECONDS = 20.0
WINDOW_STEPS = 40                    # 200 ms of spikes before a shot are kept for "what drove the shots that landed"
REPLAY_EVERY = 5                     # a replay frame every this many ticks (10 Hz)
TIEBREAK_REMATCHES = 2


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Duelist:
    """One side: a brain and a body (position, heading, health). Brains are read and poked the way the game does."""

    def __init__(self, brain, name: str, pos, yaw: float):
        self.br, self.name = brain, name
        self.pos = np.array(pos, float)
        self.yaw = float(yaw)
        self.hp = HP
        self.steer = 0.0
        self.fire_ready = self.dodge_ready = self.run_ready = 0.0
        self.reward_until = self.punish_until = 0.0
        self.hop = np.zeros(2)                 # a hop in progress: remaining displacement
        self.shots = self.hits = 0
        self.max_levels = dict(fire=0.0, escape=0.0, run=0.0, walk=0.0)
        self.ring: deque = deque(maxlen=WINDOW_STEPS)
        self.hit_windows = 0
        self.miss_windows = 0
        self._events: list = []
        self._hit_idx: list = []

    def fwd(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw)])


def _senses(me: Duelist, other: Duelist, pellets: list, k, t: float) -> None:
    """What `me` sees of `other`: the duel's own _duel_senses (LC10 target tracking on the side the other is on, small-object
    detectors when it is dead ahead), plus looming from the other's pellets coming at it (game rule: a 0.1 m pellet)."""
    to = other.pos - me.pos
    dist = float(np.hypot(*to))
    if dist <= SEE_RANGE and me.hp > 0:
        u = to / max(dist, 1e-6)
        f = me.fwd()
        side_v = np.array([-f[1], f[0]])
        ahead, sideness = float(u @ f), float(u @ side_v)
        near = float(np.clip(1.3 - dist / 7.0, 0.3, 1.0))
        s_side = float(np.clip(abs(sideness) / TRACK_SCALE, 0, 1)) * near
        key = "L" if sideness > 0 else "R"             # the 3D duel's +side is its right: this plane's +side is the left
        if ahead < 0.995 and s_side > 0.03:
            me.br.poke("track", key, 0.05, recruit=0.15 + 0.45 * s_side)
        if ahead > 0.985 and dist < 7.0:
            s = near * float(np.clip((ahead - 0.985) / 0.012, 0, 1))
            if s > 0.02:
                for sd in ("L", "R") if abs(sideness) < 0.12 else (key,):
                    me.br.poke("small", sd, 0.05, recruit=0.2 + 0.4 * s)
    best = 0.0
    for p in pellets:                                   # looming: how fast an incoming pellet grows in the view
        if p["owner"] is me:
            continue
        rel = p["pos"] - me.pos
        d = float(np.hypot(*rel))
        if d < 0.3 or d > 4.0:
            continue
        closing = -float(rel @ p["vel"]) / d
        if closing <= 0:
            continue
        best = max(best, 2 * PELLET_RADIUS * closing / (d * d))
    if best > 0 and me.hp > 0:
        strength = float(np.clip((best - k.LOOM_MIN) / k.LOOM_FULL, 0, 1))
        if strength > 0:
            me.br.poke("loom", None, strength, recruit=0.6 * strength)


def _motor(me: Duelist, other: Duelist, pellets: list, k, t: float, dt: float, rng, pid: list) -> None:
    """Steering from DNa01/02 right minus left, shooting when DNp35/DNpe052 fire above threshold, dodging on the giant fiber,
    running on the body-touch group: the 3D duel's _duel_motor and the game's reactions, with this plane's geometry."""
    br = me.br
    if me.hp <= 0:
        return
    if t < me.reward_until:
        br.poke("reward", None, 0.8)
    if t < me.punish_until:
        br.poke("punish", None, 1.0)
    turn = br.hz("turn_r") - br.hz("turn_l")           # DNa01/02 right minus left: positive turns toward the fly's right
    me.steer += (turn - me.steer) * 0.35
    mag = max(0.0, abs(me.steer) - STEER_DEADZONE_HZ)
    if mag > 0:
        dyaw = math.copysign(min(STEER_MAX, mag * STEER_GAIN), me.steer) * (dt * 60.0)
        me.yaw = _wrap(me.yaw - dyaw)                  # heading is counter-clockwise in this plane, so "right" is a negative step
    lv = {n: br.level(n) for n in ("fire", "escape", "run", "walk")}
    for n_, v in lv.items():
        me.max_levels[n_] = max(me.max_levels[n_], v)
    if lv["fire"] > k.THRESH["fire"] and t >= me.fire_ready:
        me.fire_ready = t + FIRE_COOLDOWN
        ang = math.atan2(*me.fwd()[::-1]) + rng.normal(0, PELLET_SPREAD)
        v = np.array([math.cos(ang), math.sin(ang)]) * PELLET_SPEED
        pid[0] += 1
        pellets.append(dict(pos=me.pos + me.fwd() * 0.25, vel=v, born=t, owner=me, id=pid[0],
                            window=np.concatenate(list(me.ring)) if me.ring else np.zeros(0, np.int64)))
        me.shots += 1
        me._events.append(("shot", t, me.name))
    away = me.pos - other.pos
    away = away / max(float(np.hypot(*away)), 1e-6)
    if lv["escape"] > k.THRESH["escape"] and t >= me.dodge_ready and pellets:
        threats = [p for p in pellets if p["owner"] is not me and np.hypot(*(p["pos"] - me.pos)) < 4.0]
        if threats:                                    # a hop sideways to the nearest incoming pellet's line
            p = min(threats, key=lambda q: np.hypot(*(q["pos"] - me.pos)))
            dirp = p["vel"] / PELLET_SPEED
            perp = np.array([-dirp[1], dirp[0]])
            if perp @ (me.pos - p["pos"]) < 0:
                perp = -perp
            me.hop += perp * DODGE_HOP
            me.dodge_ready = t + DODGE_COOLDOWN
            me._events.append(("dodge", t, me.name))
    if lv["run"] > k.THRESH["run"] and t >= me.run_ready and t < me.punish_until + 0.5:
        me.hop += away * RUN_HOP                       # hurt, and the body-touch group says run
        me.run_ready = t + RUN_COOLDOWN
        me._events.append(("run", t, me.name))
    step = np.zeros(2)
    if lv["walk"] > k.THRESH["walk"]:
        step += me.fwd() * WALK_SPEED * dt
    if np.hypot(*me.hop) > 1e-6:                       # hops play out over ~0.2 s
        part = me.hop * min(1.0, dt / 0.2)
        step += part
        me.hop = me.hop - part
    me.pos = np.clip(me.pos + step, -ARENA_HALF, ARENA_HALF)


def _pellets(pellets: list, duelists: tuple, t: float, dt: float, k) -> None:
    keep = []
    for p in pellets:
        p["pos"] = p["pos"] + p["vel"] * dt
        hit = None
        for d in duelists:
            if d is p["owner"] or d.hp <= 0:
                continue
            if float(np.hypot(*(p["pos"] - d.pos))) < FLY_RADIUS:
                hit = d
                break
        shooter = p["owner"]
        if hit is not None:
            hit.hp = max(0.0, hit.hp - PELLET_DAMAGE)
            shooter.hits += 1
            shooter.reward_until = t + REWARD_PULSE_S
            hit.punish_until = t + PUNISH_PULSE_S
            hit.br.poke("body", None, 0.5)             # the same touch as a hit from you: it feels the pellet
            shooter._events.append(("hit", t, shooter.name))
            shooter.hit_windows += 1
            shooter._hit_idx.append(p["window"])
            continue
        outside = np.any(np.abs(p["pos"]) > ARENA_HALF)
        if outside or t - p["born"] > PELLET_LIFE_S:
            shooter.miss_windows += 1
            continue
        keep.append(p)
    pellets[:] = keep


def run_duel(brain_a, brain_b, seed: int = 0, seconds: float = DEFAULT_SECONDS, names=("A", "B"), record: bool = True,
             drivers: bool = False, progress=None, cancel=None) -> dict:
    """One match between two brains. Returns the winner ('A', 'B' or None for a draw), health, shots, hits, the replay frames
    and events, and (drivers=True) the neurons' spike counts before the shots that landed.

    The match seed sets the starting headings and the pellets' scatter; the brains' own noise is their seeds' (a caller that
    wants a different match between the same two flies reseeds them: Brain.reseed). The match ends when one health reaches 0 or
    the time is up (more health wins; equal health is a draw: lib code decides what a draw means).
    """
    from kickthefly.game import kick_the_fly as k

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    a = Duelist(brain_a, names[0], (-START_APART / 2, 0.0), rng.uniform(-START_YAW_JITTER, START_YAW_JITTER))
    b = Duelist(brain_b, names[1], (START_APART / 2, 0.0), math.pi + rng.uniform(-START_YAW_JITTER, START_YAW_JITTER))
    pellets: list = []
    pid = [0]
    events: list = []
    for d in (a, b):
        d._events = events
        d._hit_idx = []
    dt = TICK_STEPS * float(getattr(brain_a, "dt", 0.005))
    ticks = int(round(seconds / dt))
    frames = []
    base = {d.name: None for d in (a, b)}
    if drivers:
        for d in (a, b):
            base[d.name] = np.zeros(d.br.n, np.int32)
    steps = 0
    for tick in range(ticks):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        t = tick * dt
        for me, other in ((a, b), (b, a)):
            _senses(me, other, pellets, k, t)
        for _ in range(TICK_STEPS):
            for d in (a, b):
                d.br._step()
                if drivers:
                    sp = d.br.sim.spikes
                    d.ring.append(np.flatnonzero(sp))
                    base[d.name] += sp
            steps += 1
        for me, other in ((a, b), (b, a)):
            _motor(me, other, pellets, k, t, dt, rng, pid)
        _pellets(pellets, (a, b), t, dt, k)
        if record and tick % REPLAY_EVERY == 0:
            frames.append([round(t, 3), *map(float, (*a.pos, a.yaw, a.hp, *b.pos, b.yaw, b.hp)),
                           [[round(float(p["pos"][0]), 2), round(float(p["pos"][1]), 2), 0 if p["owner"] is a else 1]
                            for p in pellets]])
        if progress and tick % 50 == 0:
            progress(tick / ticks)
        if a.hp <= 0 or b.hp <= 0:
            break
    if a.hp > b.hp:
        winner = names[0]
    elif b.hp > a.hp:
        winner = names[1]
    else:
        winner = None
    res = dict(winner=winner, hp={names[0]: a.hp, names[1]: b.hp}, shots={names[0]: a.shots, names[1]: b.shots},
               hits={names[0]: a.hits, names[1]: b.hits}, seconds=round(min(ticks, tick + 1) * dt, 3), knockout=bool(a.hp <= 0 or b.hp <= 0),
               max_levels={names[0]: {n: round(v, 3) for n, v in a.max_levels.items()},
                           names[1]: {n: round(v, 3) for n, v in b.max_levels.items()}},
               events=[[e[0], round(e[1], 3), e[2]] for e in events], frames=frames if record else [], steps=steps)
    if drivers:
        res["_drivers"] = {d.name: _type_sums(d, base[d.name], steps) for d in (a, b)}
    return res


def _type_sums(d: Duelist, base: np.ndarray, steps: int) -> dict:
    """Per cell type: spikes in the 200 ms windows before the shots that landed, spikes over the whole match, neurons."""
    types = np.asarray(d.br.types).astype(str)
    names, inv = np.unique(types, return_inverse=True)
    hit = np.zeros(d.br.n, np.int64)
    for w in d._hit_idx:
        np.add.at(hit, w, 1)
    return dict(types=names, hit=np.bincount(inv, weights=hit, minlength=len(names)),
                base=np.bincount(inv, weights=base, minlength=len(names)), n=np.bincount(inv, minlength=len(names)),
                hit_windows=int(d.hit_windows), steps=int(steps), dt=float(d.br.dt))


def combine_drivers(parts: list[dict]) -> dict | None:
    """Sum the type sums of several matches (they share one pack, so one type list) into one."""
    if not parts:
        return None
    out = dict(types=parts[0]["types"], hit=np.zeros_like(parts[0]["hit"]), base=np.zeros_like(parts[0]["base"]),
               n=parts[0]["n"], hit_windows=0, steps=0, dt=parts[0]["dt"])
    for p in parts:
        out["hit"] = out["hit"] + p["hit"]
        out["base"] = out["base"] + p["base"]
        out["hit_windows"] += p["hit_windows"]
        out["steps"] += p["steps"]
    return out


def top_drivers(sums: dict | None, top: int = 10, min_base: int = 50, min_hit: int = 5, min_windows: int = 3) -> list[dict]:
    """The cell types whose firing in the 200 ms before landed shots most exceeds their firing over the whole match. A correlation
    of the brain's own activity with its shots, not a causal test (nothing was silenced)."""
    if not sums or sums["hit_windows"] < min_windows:
        return []
    win_s = WINDOW_STEPS * sums["dt"]
    rows = []
    for i, name in enumerate(sums["types"]):
        n = sums["n"][i]
        if sums["base"][i] < min_base or sums["hit"][i] < min_hit or n < 1:
            continue
        r_hit = sums["hit"][i] / (sums["hit_windows"] * win_s * n)
        r_base = sums["base"][i] / (sums["steps"] * sums["dt"] * n)
        rows.append(dict(type=str(name), neurons=int(n), hz_before_hits=float(r_hit), hz_overall=float(r_base),
                         ratio=float(r_hit / r_base) if r_base > 0 else float("nan")))
    rows = [r for r in rows if r["ratio"] == r["ratio"]]
    rows.sort(key=lambda r: -r["ratio"])
    return rows[:top]


# the cell types the duel's own rules read or poke: how they fired before landed shots, whatever their rank in the full list
READOUT_TYPES = ("DNp35", "DNpe052", "DNa01", "DNa02", "LC10", "LC11", "LC18", "LC21", "LC26", "DNp01", "SNta13")


def named_drivers(sums: dict | None, names=READOUT_TYPES) -> list[dict]:
    if not sums or sums["hit_windows"] < 1:
        return []
    win_s = WINDOW_STEPS * sums["dt"]
    idx = {str(t): i for i, t in enumerate(sums["types"])}
    rows = []
    for name in names:
        i = idx.get(name)
        if i is None or sums["n"][i] < 1:
            continue
        n = sums["n"][i]
        r_hit = sums["hit"][i] / (sums["hit_windows"] * win_s * n)
        r_base = sums["base"][i] / (sums["steps"] * sums["dt"] * n)
        rows.append(dict(type=name, neurons=int(n), hz_before_hits=float(r_hit), hz_overall=float(r_base),
                         ratio=float(r_hit / r_base) if r_base > 0 else float("nan")))
    return rows
