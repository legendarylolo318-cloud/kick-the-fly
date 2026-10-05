"""Contraption builder, the engine (3.1.0 task 13, Play): a Rube Goldberg sandbox that runs against the fly.

You place parts on a side-view bench: a marble, ramps, springs, dominoes, fans, lamps, sugar, tool triggers, buttons and timers. The marble rolls, the dominoes tip, a button is pressed,
and whatever is wired to it goes off. This module is only the machine: a small deterministic 2D simulation (no randomness, no pygame, no brain), a build format, a validator, a share
codec and the example builds. What a part does to the *fly* is not here: the engine hands out events ("fire the swatter at x, y"), and lab/contraption_challenge.py fires them through
the game's own tool code (2D and 3D), so a swat, a bomb, a zap, a cVA puff, sugar and the rest drive exactly the neurons they drive when you use the tool yourself. A fan drives the
Johnston's organ wind neurons and a lamp the photoreceptors, through the same `Brain.poke` regions the fan and lamp arenas use.

Tags: the wiring of the fly's response (which neurons a tool or a gust drives, and what they then do) is CONNECTOME; the bench, the parts, their physics (a marble that rolls, a domino that tips
in 0.2-0.4 s), the channels, the limits and what a part does to the world are GAME RULES; what the fly does when the contraption runs is a MODEL PREDICTION.

Units in a build are whole numbers so a build is small, exact and shareable: x and y in centimetres (x from the bench's middle, y up from the floor), angles in degrees, times in tenths of a second.
Channels: a part on channel 0 goes off when the run starts; channels 1-5 go off when something sends them a signal (a button that is pressed, a timer that rings).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

XMAX = 2.40                      # m: the bench is 4.8 m wide (x from -2.4 to 2.4) and 3.0 m high
YMAX = 3.00
GRID_CM = 5
GRAVITY = 9.8
BALL_R = 0.045
DT = 1.0 / 240.0
MAX_PARTS = 24
MAX_MARBLES = 3
MAX_RUN_S = 60.0
CHANNELS = 6                     # 0 = at the start, 1-5 = signals
FAN_ON_S = 4.0                   # a fan blows this long each time its channel goes off
LAMP_DEFAULT_TENTHS = 30
FAN_ACCEL = 7.0                  # m/s^2 on a marble in front of a fan's mouth
SPRING_SPEED = 4.0               # m/s: what a spring launches a marble at
TIP_BASE_S, TIP_PER_M = 0.12, 0.8
SHOT_GAP_S = 0.30                # a tool part's burst: one shot this often

KINDS = ("marble", "ramp", "spring", "domino", "fan", "lamp", "sugar", "tool", "button", "timer", "fly")
LABEL = {"marble": "Marble", "ramp": "Ramp", "spring": "Spring", "domino": "Domino", "fan": "Fan", "lamp": "Lamp", "sugar": "Sugar",
         "tool": "Tool", "button": "Button", "timer": "Timer", "fly": "Fly"}
HELP = {
    "marble": "Rolls from here when the run starts (or after its delay).",
    "ramp": "A slope the marble rolls on. Angle and length can be changed.",
    "spring": "A pad that throws a marble up (and sideways, with an angle).",
    "domino": "Stands up; a marble or a falling domino tips it over into the next.",
    "fan": "Blows when its channel goes off: pushes marbles and drives the fly's wind neurons.",
    "lamp": "Lights when its channel goes off: drives the fly's photoreceptors.",
    "sugar": "Drops sugar when its channel goes off (the game's sugar, the fly smells and eats it).",
    "tool": "Fires one of the game's tools here when its channel goes off.",
    "button": "Pressed by a marble or a falling domino; sends its channel's signal.",
    "timer": "Sends its channel's signal at a time, and again every so often.",
    "fly": "Where the fly is put when the run starts, and which way it faces. Tethered (the default), it is held at this spot, as a fly is on the bench; its brain still reacts to everything.",
}
# the game's own tools a Tool part may fire: the ones that are one click at a place (the held ones, the laser and the predators that are spawned over time are not parts)
TOOLS = ("flick", "swatter", "bomb", "zapper", "cva", "alcohol", "fruit", "decoy", "spider")
TOOL_NOTE = {"flick": "a flick of the marble's weight", "swatter": "the swatter", "bomb": "a bomb (1.5 s fuse)", "zapper": "the electric zap", "cva": "a cVA puff",
             "alcohol": "alcohol", "fruit": "ripe fruit", "decoy": "a decoy female", "spider": "a spider that hunts it"}
TAGS = {"CONNECTOME": "what a tool, a gust or a light drives in the fly (the game's own mappings)",
        "GAME RULE": "the bench, the parts, the physics, the channels, the limits",
        "MODEL PREDICTION": "what the fly does when the contraption runs"}


class BuildError(ValueError):
    """A build that cannot be used; str(e) is the reason."""


@dataclass
class Part:
    k: str
    x: int = 0                      # cm
    y: int = 0                      # cm (the floor is 0)
    a: int = 0                      # degrees
    n: int = 0                      # cm: length (ramp), height (domino), reach (fan)
    ch: int = 0                     # channel
    tool: str = ""
    t: int = 0                      # tenths of a second
    ev: int = 0                     # tenths: a timer rings again every this often (0 = once)
    c: int = 1                      # a count: a timer's rings, a tool's shots in a burst

    def to_json(self) -> dict:
        d = {"k": self.k, "x": self.x, "y": self.y}
        default = defaults(self.k)
        for f in ("a", "n", "ch", "tool", "t", "ev", "c"):
            if getattr(self, f) != getattr(default, f):
                d[f] = getattr(self, f)
        return d


def defaults(kind: str) -> Part:
    return {"marble": Part("marble", 0, 150),
            "ramp": Part("ramp", 0, 100, a=-15, n=90),
            "spring": Part("spring", 0, 0, a=0),
            "domino": Part("domino", 0, 0, n=25),
            "fan": Part("fan", 0, 20, a=0, n=120, ch=1),
            "lamp": Part("lamp", 0, 120, ch=1, t=LAMP_DEFAULT_TENTHS),
            "sugar": Part("sugar", 0, 30, ch=1),
            "tool": Part("tool", 0, 30, ch=1, tool="swatter"),
            "button": Part("button", 0, 0, ch=1),
            "timer": Part("timer", 0, 150, ch=1, t=20),
            "fly": Part("fly", 0, 0, a=0)}[kind]


# the editor's ranges: (low, high, step) per field per kind; the validator enforces the same
RANGES = {
    "marble": dict(t=(0, 100, 5)),
    "ramp": dict(a=(-60, 60, 5), n=(20, 200, 10)),
    "spring": dict(a=(-60, 60, 15)),
    "domino": dict(n=(10, 40, 5)),
    "fan": dict(a=(0, 345, 15), n=(30, 200, 10), ch=(0, CHANNELS - 1, 1)),
    "lamp": dict(ch=(0, CHANNELS - 1, 1), t=(10, 100, 10)),
    "sugar": dict(ch=(0, CHANNELS - 1, 1)),
    "tool": dict(ch=(0, CHANNELS - 1, 1), c=(1, 5, 1)),
    "button": dict(ch=(1, CHANNELS - 1, 1)),
    "timer": dict(t=(5, 600, 5), ev=(0, 100, 5), c=(1, 10, 1), ch=(1, CHANNELS - 1, 1)),
    "fly": dict(a=(0, 180, 180), c=(0, 1, 1)),
}
EDITABLE = {k: tuple(v) for k, v in RANGES.items()}


def snap(v: float) -> int:
    return int(round(v / GRID_CM)) * GRID_CM


def clamp_position(kind: str, x: float, y: float) -> tuple[int, int]:
    x = max(-int(XMAX * 100) + 10, min(int(XMAX * 100) - 10, snap(x)))
    y = max(0, min(int(YMAX * 100), snap(y)))
    if kind in ("domino", "button", "spring", "fly"):            # these stand on a surface; the editor lets the level be chosen, the floor is the default
        y = max(0, y)
    return x, y


def new_part(kind: str, x: float, y: float) -> Part:
    if kind not in KINDS:
        raise BuildError(f"{kind!r} is not a part")
    p = defaults(kind)
    p.x, p.y = clamp_position(kind, x, y)
    if kind in ("domino", "button", "spring", "fly"):
        p.y = 0 if y < 20 else p.y
    return p


def step_field(part: Part, field_name: str, direction: int) -> bool:
    """Nudge one field of a part up or down by its step, within its range. Returns whether it changed."""
    if field_name == "tool":
        i = TOOLS.index(part.tool) if part.tool in TOOLS else 0
        part.tool = TOOLS[(i + direction) % len(TOOLS)]
        return True
    rng = RANGES.get(part.k, {}).get(field_name)
    if rng is None:
        return False
    lo, hi, st = rng
    old = getattr(part, field_name)
    if field_name == "a" and part.k == "fan":                      # a direction goes round
        new = (old + direction * st) % 360
    else:
        new = max(lo, min(hi, old + direction * st))
    setattr(part, field_name, new)
    return new != old


# --- builds ------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Build:
    name: str = "my contraption"
    parts: list[Part] = field(default_factory=list)

    def count(self, kind: str) -> int:
        return sum(1 for p in self.parts if p.k == kind)

    def fly_mark(self) -> Part | None:
        return next((p for p in self.parts if p.k == "fly"), None)

    def add(self, part: Part) -> None:
        if part.k == "fly":                                        # one fly mark: placing it again moves it
            self.parts = [p for p in self.parts if p.k != "fly"]
        elif len(self.parts) >= MAX_PARTS:
            raise BuildError(f"a contraption has at most {MAX_PARTS} parts")
        elif part.k == "marble" and self.count("marble") >= MAX_MARBLES:
            raise BuildError(f"a contraption has at most {MAX_MARBLES} marbles")
        self.parts.append(part)

    def problems(self, available=None) -> list[str]:
        return check(self.to_json(), available)

    def to_json(self) -> dict:
        return {"name": self.name, "parts": [p.to_json() for p in self.parts]}

    @classmethod
    def from_json(cls, d: dict, available=None) -> "Build":
        bad = check(d, available)
        if bad:
            raise BuildError(bad[0])
        parts = []
        for r in d["parts"]:
            p = defaults(r["k"])
            for f, v in r.items():
                setattr(p, f, v)
            parts.append(p)
        return cls(str(d.get("name", "shared"))[:24], parts)

    def code(self) -> str:
        from kickthefly.core import sharecode

        if not self.parts:
            raise BuildError("nothing to share: place a part first")
        return sharecode.encode("contraption", self.to_json())

    @classmethod
    def from_code(cls, text: str, available=None) -> "Build":
        from kickthefly.core import sharecode

        try:
            c = sharecode.decode(text)
        except sharecode.ShareError as e:
            raise BuildError(str(e)) from None
        if c.kind != "contraption":
            raise BuildError(f"this is a {sharecode.KIND_LABEL[c.kind].lower()} code; a contraption build is a contraption code")
        return cls.from_json(c.payload, available)


_INT_FIELDS = ("x", "y", "a", "n", "ch", "t", "ev", "c")


def check(d, available=None) -> list[str]:
    """Reasons a build cannot be used (empty when it can). `available(tool)` says whether this game has a tool (Play or Lab, larva or adult); None = all."""
    if not isinstance(d, dict):
        return ["a contraption is a mapping with a name and parts"]
    extra = set(d) - {"name", "parts"}
    if extra:
        return [f"unknown fields in a contraption: {sorted(extra)}"]
    parts = d.get("parts")
    if not isinstance(parts, list) or not parts:
        return ["a contraption needs at least one part"]
    if not isinstance(d.get("name", ""), str) or len(d.get("name", "")) > 24:
        return ["a contraption's name is at most 24 characters"]
    if len(parts) > MAX_PARTS + 1:
        return [f"a contraption has at most {MAX_PARTS} parts"]
    out = []
    marbles = flies = 0
    for i, r in enumerate(parts):
        where = f"part {i + 1}"
        if not isinstance(r, dict) or r.get("k") not in KINDS:
            return [f"{where}: not a part this version has"]
        k = r["k"]
        bad = set(r) - {"k", "x", "y", "a", "n", "ch", "tool", "t", "ev", "c"}
        if bad:
            return [f"{where}: unknown fields {sorted(bad)}"]
        for f in _INT_FIELDS:
            if f in r and (isinstance(r[f], bool) or not isinstance(r[f], int)):
                return [f"{where} ({k}): {f} must be a whole number"]
        if not (-int(XMAX * 100) <= r.get("x", 0) <= int(XMAX * 100) and 0 <= r.get("y", 0) <= int(YMAX * 100)):
            out.append(f"{where} ({k}) is off the bench")
        for f, (lo, hi, _) in RANGES[k].items():
            v = r.get(f, getattr(defaults(k), f))
            if not lo <= v <= hi:
                out.append(f"{where} ({k}): {f} must be from {lo} to {hi}")
        if k == "tool":
            tool = r.get("tool", defaults(k).tool)
            if tool not in TOOLS:
                out.append(f"{where}: {tool!r} is not a tool a part can fire")
            elif available is not None and not available(tool):
                out.append(f"{where}: the {tool} is not available in this game mode")
        marbles += k == "marble"
        flies += k == "fly"
    if marbles > MAX_MARBLES:
        out.append(f"at most {MAX_MARBLES} marbles")
    if flies > 1:
        out.append("at most one fly mark")
    if len([r for r in parts if r["k"] != "fly"]) > MAX_PARTS:
        out.append(f"a contraption has at most {MAX_PARTS} parts")
    return out


# --- saved builds (eight slots) -----------------------------------------------------------------------------------------------------------
SLOTS = 8


def saved_path() -> Path:
    from kickthefly.core import paths

    return paths.get().data_dir / "contraptions.json"


def load_slots() -> list[dict | None]:
    """The eight save slots, each a build dict or None. A damaged file or a damaged slot is ignored."""
    try:
        raw = json.loads(saved_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [None] * SLOTS
    out: list[dict | None] = [None] * SLOTS
    for i, d in enumerate((raw if isinstance(raw, list) else [])[:SLOTS]):
        if isinstance(d, dict) and not check(d):
            out[i] = d
    return out


def save_slot(i: int, build: Build) -> bool:
    if not 0 <= i < SLOTS or not build.parts:
        return False
    slots = load_slots()
    slots[i] = build.to_json()
    p = saved_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(slots, indent=1), encoding="utf-8")
    tmp.replace(p)
    return True


# --- the machine ------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Event:
    t: float
    kind: str                       # "tool" (fire a tool at x, y), "fire" (a channel went off), "on" (a fan or lamp came on), "hit" (a marble or domino met the fly)
    x: float = 0.0                  # m
    y: float = 0.0
    tool: str = ""
    part: int = -1
    ch: int = -1
    by: str = ""                    # what set it off: start, marble, domino, timer


@dataclass
class _Ball:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    still_s: float = 0.0
    last_fly: float = -9.0
    part: int = -1


@dataclass
class _Domino:
    x: float
    y: float
    h: float
    state: str = "standing"         # standing | tipping | fallen
    t0: float = 0.0
    dir: int = 1
    part: int = -1

    @property
    def fall_s(self) -> float:
        return TIP_BASE_S + TIP_PER_M * self.h

    def angle(self, t: float) -> float:
        """Radians from upright, signed by the way it falls."""
        if self.state == "standing":
            return 0.0
        f = 1.0 if self.state == "fallen" else min(1.0, (t - self.t0) / self.fall_s)
        return self.dir * (math.pi / 2) * f * f


class Machine:
    """The contraption running. `step(dt, fly)` advances it and returns the events of that step; `fly` is (x, y, z, radius) in metres or None (the engine
    does not know the fly except to let a marble or a domino meet it)."""

    def __init__(self, build: Build):
        self.build = build
        self.t = 0.0
        self.parts = list(build.parts)
        self.balls: list[_Ball] = []
        self.ramps = [(i, p, self._ramp_ends(p)) for i, p in enumerate(self.parts) if p.k == "ramp"]
        self.dominoes = [_Domino(p.x / 100, p.y / 100, p.n / 100, part=i) for i, p in enumerate(self.parts) if p.k == "domino"]
        self.springs = [(i, p) for i, p in enumerate(self.parts) if p.k == "spring"]
        self.buttons = [(i, p) for i, p in enumerate(self.parts) if p.k == "button"]
        self.pending_marbles = sorted([(p.t / 10, i) for i, p in enumerate(self.parts) if p.k == "marble"])
        self.rings: list[tuple[float, int, int]] = []           # (time, channel, part) the timers will ring
        for i, p in enumerate(self.parts):
            if p.k == "timer":
                for j in range(max(1, p.c) if p.ev else 1):
                    self.rings.append((p.t / 10 + j * p.ev / 10, p.ch, i))
        self.rings.sort()
        self.fan_until: dict[int, float] = {}
        self.lamp_until: dict[int, float] = {}
        self.shots: list[tuple[float, int]] = []                # (time, part) queued tool shots
        self.button_cool: dict[int, float] = {}
        self.started = False
        self.finished = False
        self.quiet_s = 0.0
        self.log: list[Event] = []
        self.fired: dict[int, float] = {}                       # part -> the first time it went off

    @staticmethod
    def _ramp_ends(p: Part):
        a = math.radians(p.a)
        L = p.n / 100
        ax, ay = p.x / 100, p.y / 100
        return ax, ay, ax + L * math.cos(a), ay + L * math.sin(a)

    # --- channels ------------------------------------------------------------------------------------------------------------------------------
    def _fire(self, ch: int, part: int, by: str, out: list) -> None:
        ev = Event(self.t, "fire", part=part, ch=ch, by=by)
        out.append(ev)
        for i, p in enumerate(self.parts):
            if p.ch != ch or p.k not in ("fan", "lamp", "sugar", "tool"):
                continue
            self.fired.setdefault(i, self.t)
            if p.k == "fan":
                self.fan_until[i] = self.t + FAN_ON_S
                out.append(Event(self.t, "on", p.x / 100, p.y / 100, "fan", i, ch, by))
            elif p.k == "lamp":
                self.lamp_until[i] = self.t + p.t / 10
                out.append(Event(self.t, "on", p.x / 100, p.y / 100, "lamp", i, ch, by))
            elif p.k == "sugar":
                self.shots.append((self.t, i))
            else:
                for j in range(max(1, p.c)):
                    self.shots.append((self.t + j * SHOT_GAP_S, i))
        self.shots.sort()

    # --- what the fly is exposed to ------------------------------------------------------------------------------------------------------------------
    def wind_on(self, x: float, y: float, z: float = 0.0) -> float:
        """0..1: how hard the running fans blow on a point (the strength the game's wind neurons are driven at)."""
        best = 0.0
        for i, until in self.fan_until.items():
            if self.t >= until or abs(z) > 0.9:
                continue
            p = self.parts[i]
            ux, uy = math.cos(math.radians(p.a)), math.sin(math.radians(p.a))
            dx, dy = x - p.x / 100, y - p.y / 100
            along = dx * ux + dy * uy
            across = abs(-dx * uy + dy * ux)
            reach = p.n / 100
            if 0 <= along <= reach and across <= 0.45:
                best = max(best, 0.9 * (1 - along / reach) + 0.1)
        return min(1.0, best)

    def light_on(self, x: float, y: float, z: float = 0.0) -> float:
        """0..1: how brightly the lit lamps shine on a point."""
        best = 0.0
        for i, until in self.lamp_until.items():
            if self.t >= until:
                continue
            p = self.parts[i]
            d = math.sqrt((x - p.x / 100) ** 2 + (y - p.y / 100) ** 2 + z * z)
            best = max(best, max(0.0, 1.2 - d / 2.0))
        return min(1.0, best)

    # --- stepping ------------------------------------------------------------------------------------------------------------------------------
    def step(self, dt: float = DT, fly=None) -> list[Event]:
        out: list[Event] = []
        if self.finished:
            return out
        if not self.started:
            self.started = True
            self._fire(0, -1, "start", out)
        self.t += dt
        t = self.t
        while self.pending_marbles and self.pending_marbles[0][0] <= t:
            _, i = self.pending_marbles.pop(0)
            p = self.parts[i]
            self.balls.append(_Ball(p.x / 100, p.y / 100, part=i))
            self.fired.setdefault(i, t)
        while self.rings and self.rings[0][0] <= t:
            _, ch, i = self.rings.pop(0)
            self.fired.setdefault(i, t)
            self._fire(ch, i, "timer", out)
        while self.shots and self.shots[0][0] <= t:
            _, i = self.shots.pop(0)
            p = self.parts[i]
            tool = "sugar" if p.k == "sugar" else p.tool
            out.append(Event(t, "tool", p.x / 100, p.y / 100, tool, i, p.ch, by=f"channel {p.ch}"))
        for b in self.balls:
            self._move_ball(b, dt, fly, out)
        for d in self.dominoes:
            if d.state == "tipping" and t - d.t0 >= d.fall_s:
                d.state = "fallen"
                self._strike(d, fly, out)
        self._settle(dt)
        self.log.extend(out)
        return out

    def _move_ball(self, b: _Ball, dt: float, fly, out: list) -> None:
        t = self.t
        b.vy -= GRAVITY * dt
        for i, until in self.fan_until.items():                 # a fan pushes a marble in front of its mouth
            if t >= until:
                continue
            p = self.parts[i]
            ux, uy = math.cos(math.radians(p.a)), math.sin(math.radians(p.a))
            dx, dy = b.x - p.x / 100, b.y - p.y / 100
            along, across = dx * ux + dy * uy, abs(-dx * uy + dy * ux)
            reach = p.n / 100
            if 0 <= along <= reach and across <= 0.35:
                k = FAN_ACCEL * (1 - along / reach) * dt
                b.vx += ux * k
                b.vy += uy * k
        speed = math.hypot(b.vx, b.vy)
        if speed > 8.0:
            b.vx, b.vy = b.vx * 8.0 / speed, b.vy * 8.0 / speed
        b.x += b.vx * dt
        b.y += b.vy * dt
        on_surface = False
        if b.y < BALL_R:                                        # the floor
            b.y = BALL_R
            if b.vy < -0.5:
                b.vy = -b.vy * 0.25
            else:
                b.vy = 0.0
            b.vx *= 1 - 0.8 * dt
            on_surface = True
        if abs(b.x) > XMAX - BALL_R:                            # the bench's ends
            b.x = math.copysign(XMAX - BALL_R, b.x)
            b.vx = -b.vx * 0.4
        for _, p, (ax, ay, bx, by) in self.ramps:               # ramps are one-sided: a marble passes up through one from below
            sx, sy = bx - ax, by - ay
            L2 = sx * sx + sy * sy
            u = max(0.0, min(1.0, ((b.x - ax) * sx + (b.y - ay) * sy) / L2))
            cx, cy = ax + u * sx, ay + u * sy
            dx, dy = b.x - cx, b.y - cy
            dist = math.hypot(dx, dy)
            if dist >= BALL_R:
                continue
            nx, ny = -sy / math.sqrt(L2), sx / math.sqrt(L2)
            if ny < 0:
                nx, ny = -nx, -ny
            if (b.x - ax) * nx + (b.y - ay) * ny < -BALL_R:
                continue
            if dist > 1e-9 and 0 < u < 1:
                nx, ny = dx / dist, dy / dist
                if ny < 0:
                    continue
            elif dist > 1e-9:
                nx, ny = dx / dist, dy / dist
            b.x += nx * (BALL_R - dist)
            b.y += ny * (BALL_R - dist)
            vn = b.vx * nx + b.vy * ny
            if vn < 0:
                b.vx -= vn * nx * 1.1
                b.vy -= vn * ny * 1.1
            tx, ty = ny, -nx
            vt = b.vx * tx + b.vy * ty
            damp = 0.15 * dt
            b.vx -= vt * tx * damp
            b.vy -= vt * ty * damp
            on_surface = True
        for i, p in self.springs:
            if abs(b.x - p.x / 100) < 0.11 and b.vy < 0 and p.y / 100 - 0.04 <= b.y - BALL_R <= p.y / 100 + 0.03:
                a = math.radians(p.a)
                b.vx, b.vy = SPRING_SPEED * math.sin(a), SPRING_SPEED * math.cos(a)
                b.y = p.y / 100 + BALL_R + 0.03
                self.fired.setdefault(i, t)
        for d in self.dominoes:
            if d.state == "standing" and abs(b.x - d.x) < BALL_R + 0.02 and d.y - BALL_R <= b.y <= d.y + d.h and abs(b.vx) > 0.08:
                self._tip(d, 1 if b.vx > 0 else -1, "marble")
                b.vx *= 0.5
        for i, p in self.buttons:
            if abs(b.x - p.x / 100) < 0.09 and abs(b.y - BALL_R - p.y / 100) < 0.05 and i not in self.button_cool:
                self._press(i, "marble", out)
        if fly is not None:
            fx, fy, fz, fr = fly
            if abs(fz) < 0.4 and math.hypot(b.x - fx, b.y - fy) < BALL_R + fr and t - b.last_fly > 0.8:
                b.last_fly = t
                out.append(Event(t, "hit", b.x, b.y, "flick", b.part, by="marble"))
                out.append(Event(t, "tool", b.x, b.y, "flick", b.part, by="marble"))
                b.vx = -math.copysign(max(abs(b.vx), 0.5) * 0.3, b.vx if b.vx else 1.0)
        if on_surface and math.hypot(b.vx, b.vy) < 0.04:
            b.still_s += dt
        else:
            b.still_s = 0.0

    def _tip(self, d: _Domino, direction: int, by: str) -> None:
        if d.state != "standing":
            return
        d.state, d.t0, d.dir = "tipping", self.t, direction
        self.fired.setdefault(d.part, self.t)

    def _press(self, i: int, by: str, out: list) -> None:
        self.button_cool[i] = self.t
        self.fired.setdefault(i, self.t)
        self._fire(self.parts[i].ch, i, by, out)

    def _strike(self, d: _Domino, fly, out: list) -> None:
        """A domino that has finished falling hits whatever stands within its reach in the way it fell."""
        lo, hi = sorted((d.x, d.x + d.dir * d.h * 1.05))
        for o in self.dominoes:
            if o is not d and o.state == "standing" and lo <= o.x <= hi and abs(o.y - d.y) < 0.08:
                self._tip(o, d.dir, "domino")
        for i, p in self.buttons:
            if lo - 0.05 <= p.x / 100 <= hi + 0.05 and abs(p.y / 100 - d.y) < 0.08 and i not in self.button_cool:
                self._press(i, "domino", out)
        if fly is not None:
            fx, fy, fz, fr = fly
            if abs(fz) < 0.4 and lo - fr <= fx <= hi + fr and d.y - fr <= fy <= d.y + d.h + fr:
                tipx = d.x + d.dir * d.h
                out.append(Event(self.t, "hit", tipx, d.y + 0.03, "flick", d.part, by="domino"))
                out.append(Event(self.t, "tool", tipx, d.y + 0.03, "flick", d.part, by="domino"))

    def _settle(self, dt: float) -> None:
        moving = any(b.still_s < 1.0 for b in self.balls) or bool(self.pending_marbles)
        busy = (moving or any(d.state == "tipping" for d in self.dominoes) or bool(self.rings) or bool(self.shots)
                or any(self.t < u for u in self.fan_until.values()) or any(self.t < u for u in self.lamp_until.values()))
        self.quiet_s = 0.0 if busy else self.quiet_s + dt
        if self.t >= MAX_RUN_S or self.quiet_s >= 1.0:
            self.finished = True

    # --- for drawing and reports ---------------------------------------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        t = self.t
        return {"t": t,
                "balls": [(b.x, b.y) for b in self.balls],
                "dominoes": [(d.x, d.y, d.h, d.angle(t)) for d in self.dominoes],
                "fans": [i for i, u in self.fan_until.items() if t < u],
                "lamps": [i for i, u in self.lamp_until.items() if t < u],
                "fired": dict(self.fired)}


def simulate(build: Build, seconds: float = MAX_RUN_S, fly=None) -> tuple[list[Event], Machine]:
    """Run a build to its end (or `seconds`) and return every event. `fly` is a fixed (x, y, z, radius) or None. Deterministic: the same build gives the same events."""
    m = Machine(build)
    out: list[Event] = []
    n = int(seconds / DT)
    for _ in range(n):
        out += m.step(DT, fly)
        if m.finished:
            break
    return out, m


# --- the example builds ---------------------------------------------------------------------------------------------------------------------------
def _P(k, x, y, **kw) -> Part:
    p = defaults(k)
    p.x, p.y = x, y
    for f, v in kw.items():
        setattr(p, f, v)
    return p


EXAMPLES = {
    "Domino swat": Build("Domino swat", [
        _P("marble", -230, 100), _P("ramp", -235, 60, a=-10, n=120),
        _P("domino", -50, 0, n=30), _P("domino", -25, 0, n=30), _P("domino", 0, 0, n=30), _P("domino", 25, 0, n=30),
        _P("button", 60, 0, ch=1), _P("tool", 100, 40, ch=1, tool="swatter"), _P("fly", 100, 0)]),
    "Fan and lamp": Build("Fan and lamp", [
        _P("timer", 0, 150, t=20, ch=1), _P("lamp", -40, 130, ch=1, t=40), _P("fan", -60, 20, a=0, n=150, ch=2),
        _P("timer", 0, 150, t=40, ch=2), _P("fly", 0, 0)]),
    "Bait and bomb": Build("Bait and bomb", [
        _P("sugar", 100, 30, ch=0), _P("marble", -230, 100), _P("ramp", -235, 60, a=-10, n=120),
        _P("spring", -70, 0, a=30), _P("button", 70, 0, ch=2),
        _P("tool", 100, 40, ch=2, tool="bomb"), _P("fly", 100, 0, a=180)]),
}
for _b in EXAMPLES.values():
    assert not check(_b.to_json()), (_b.name, check(_b.to_json()))
