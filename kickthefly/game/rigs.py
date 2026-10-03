"""Classic behavior rigs (3.0 day 5): a tethered flight simulator, a fly on a ball, Buridan's paradigm and a four-field olfactory arena.

Each rig is a small piece of physics around a real brain. The brain's steering and walking descending neurons are read exactly as the
duel and the race read them; the rig decides what the fly sees and where its body goes.

  CONNECTOME   what the neurons do with what they are given:
               - yaw is read from the steering descending neurons DNa01 + DNa02, right minus left (`br.hz("turn_r") - br.hz("turn_l")`);
               - forward speed is read from DNp09 against the race's fixed reference rate (flyrace.walk_level, speed rule version 2);
               - wide-field rotation reaches the steering neurons through T4/T5 and everything after them (validated:
                 `optomotor_turning`, DNa01_R + DNa02_R x2.72 vs x0.80);
               - odor reaches the olfactory receptor neurons of the game's own scent pokes (the race's, 0.3 per tick).
  GAME RULE    everything around the neurons:
               - how steering Hz becomes a yaw rate (the duel's gain 0.24 rad/s per Hz after a 1.5 Hz dead zone, at most 2.1 rad/s, smoothed
                 by 0.35 per 20 ms tick);
               - how a rotation of the world becomes T4/T5 current (the optomotor EMD stage, assays.emd_stage; used here for self-motion in
                 the tethered rig and the ball, not in the free-walking arenas);
               - how a stripe or a bar becomes LC10 target tracking (the duel's rule: LC10 on the side the object is on, scaled by how far
                 to the side it is, nothing within 5.7 degrees of straight ahead);
               - one target at a time in Buridan's arena (the stripe nearer the heading is the one tracked);
               - the platform, its edge (a fly reaching it is reflected specularly) and its quadrants, walking speed in metres, the panorama,
                 the ball's and the arena's sizes, the odor's quadrants and the preference index's definition.
  MODEL PREDICTION  whatever the fly then does: whether it follows the panorama, fixates a bar, walks between two stripes or prefers an odor.
                    Each is a prediction of this model; lab/rigassay.py has the pre-registered criteria and reports a miss as a miss.

Conventions: angles in radians, clockwise from above positive (a right turn is positive), heading 0 points along +y, bearing of an object
relative to the heading is positive when it is on the fly's right. Time advances in engine ticks of 4 brain steps (20 ms).

The engines take a brain-like object (poke, hz, level, _step, dt, plus the optomotor `Transducer`) so they run without the real pack.
"""
from __future__ import annotations

import math

import numpy as np

# --- GAME RULE constants --------------------------------------------------------------------------------------------------------
TICK_STEPS = 4                       # brain steps (5 ms) per engine tick: 20 ms
YAW_DEADZONE_HZ = 1.5                # the duel's own: STEER_DEADZONE_HZ
YAW_GAIN = 0.004 * 60                # rad/s per Hz of DNa01/02 R-L above the dead zone (the duel's STEER_GAIN per 1/60 s frame)
YAW_MAX = 0.035 * 60                 # rad/s (the duel's STEER_MAX)
STEER_SMOOTH = 0.35                  # the duel's per-tick smoothing of R-L
TRACK_SCALE = 0.45                   # sideness that drives LC10 fully (the duel's)
TRACK_AHEAD = 0.995                  # cos of the angle inside which an object counts as straight ahead (5.7 degrees)
TRACK_RECRUIT = (0.15, 0.45)         # LC10 recruit share = 0.15 + 0.45 * sideness (the duel's)
TRACK_POKE = 0.05                    # the duel's poke strength for LC10
PLATFORM_RADIUS = 0.5                # m: Buridan's round platform (virtual metres)
STRIPE_DISTANCE = 1.5                # m from the centre: the two stripes are beyond the edge and cannot be reached
FIELD_HALF = 0.5                     # m: the four-field arena is a 1 m square
SCENT_POKE = 0.3                     # the race's scent pulse strength, delivered every tick while the fly is in odor
BALL_BAR_DISTANCE = 3.0              # m: the VR bar's distance from where the fly starts
FOV_HALF = math.radians(150.0)       # a fly sees 300 degrees; a stripe or bar further round the back is not seen (GAME RULE)
TRACE_EVERY = 5                      # one trace sample per 5 ticks (100 ms)
RULE_VERSION = 1

RIGS = ("tethered", "ball", "buridan", "fourfield")
RIG_TITLE = {"tethered": "Tethered flight simulator", "ball": "Fly on a ball", "buridan": "Buridan's paradigm",
             "fourfield": "Four-field olfactory arena"}
RIG_TAGS = {"tethered": dict(yaw_from_DNa01_02="CONNECTOME", optomotor_input="CONNECTOME (through a GAME RULE EMD stage)",
                             gain_loop_panorama="GAME RULE", response="MODEL PREDICTION"),
            "ball": dict(turning_and_walking_from_DNa01_02_and_DNp09="CONNECTOME", ball_and_VR="GAME RULE", response="MODEL PREDICTION"),
            "buridan": dict(steering_and_walking_from_DNs="CONNECTOME", stripes_edge_attention="GAME RULE", trajectory="MODEL PREDICTION"),
            "fourfield": dict(odor_to_ORNs="CONNECTOME", arena_odor_quadrants_PI_definition="GAME RULE", preference="MODEL PREDICTION")}


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Transducer:
    """The optomotor transduction (GAME RULE): a world rotation relative to the fly (rad/s, positive = rightward) becomes current on
    the T4/T5 subtypes it matches (assays.emd_stage through outdoors.emd_motion_drive). Tests replace this class."""

    def drive(self, br, slip: float) -> None:
        from kickthefly.core import simcore
        from kickthefly.lab import assays

        g = assays.groups(br)
        for rows, amp in assays.emd_stage(g, float(slip)):
            simcore.drive(br, rows, amp)

    def release(self, br) -> None:
        from kickthefly.core import simcore
        from kickthefly.lab import assays

        g = assays.groups(br)
        for key in ("t45_prog_r", "t45_prog_l", "t45_regr_r", "t45_regr_l"):
            simcore.undrive(br, g[key])


class Steerer:
    """Yaw rate from the steering neurons: the duel's rule (smoothed R - L, dead zone, gain, cap)."""

    def __init__(self):
        self.s = 0.0

    def update(self, br) -> float:
        turn = float(br.hz("turn_r")) - float(br.hz("turn_l"))      # DNa01/02 right minus left: positive turns right
        self.s += (turn - self.s) * STEER_SMOOTH
        mag = max(0.0, abs(self.s) - YAW_DEADZONE_HZ)
        return math.copysign(min(YAW_MAX, mag * YAW_GAIN), self.s) if mag > 0 else 0.0


def walk_speed(br) -> float:
    """Forward speed from DNp09 (CONNECTOME) through the race's speed rule (GAME RULE, version 2): m/s."""
    from kickthefly.game import flyrace
    from kickthefly.game import kick_the_fly as k

    return flyrace.speed_of(flyrace.walk_level(br), k.THRESH["walk"])


def _advance(br, ticks: int = 1) -> None:
    for _ in range(ticks * TICK_STEPS):
        br._step()


def track_object(br, beta: float, dist: float, see_range: float = 7.5) -> float:
    """LC10 tracking of one object at relative bearing `beta` (positive = on the right) and distance `dist` (GAME RULE: the duel's
    rule). Returns the sideness that drove it (0 when nothing was poked)."""
    if dist > see_range or abs(beta) > FOV_HALF:
        return 0.0
    ahead, side = math.cos(beta), math.sin(beta)
    near = float(np.clip(1.3 - dist / 7.0, 0.3, 1.0))
    s_side = float(np.clip(abs(side) / TRACK_SCALE, 0, 1)) * near
    if ahead < TRACK_AHEAD and s_side > 0.03:
        br.poke("track", "R" if side > 0 else "L", TRACK_POKE, recruit=TRACK_RECRUIT[0] + TRACK_RECRUIT[1] * s_side)
        return s_side
    return 0.0


def _kin(br, k: int, x: float, y: float, speed: float, action: str, arena: str) -> None:
    """A Lab recording (lab/recorder.py) keeps the rig's body as its kinematics file: x, y, speed, a rig action label."""
    rec = getattr(br, "recorder", None)
    if rec is not None and k % TRACE_EVERY == 0:
        rec.log_kinematics(x, y, 0.0, speed, 100.0, action, arena)


def _check(cancel, progress, tick: int, ticks: int) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("cancelled")
    if progress and tick % 50 == 0:
        progress(tick / max(1, ticks))


def _finish(rows: list, extra: dict, dt: float) -> dict:
    a = np.asarray(rows, float).reshape(-1, len(COLS))
    out = dict(cols=list(COLS), dt=dt, ticks=int(len(a)), seconds=round(len(a) * dt, 3),
               trace=np.round(a[::TRACE_EVERY], 4).tolist(), rule_version=RULE_VERSION)
    out["_full"] = a                                # full-resolution array for the analysis; dropped before anything is written
    out.update(extra)
    return out


COLS = ("t", "omega_ext", "slip", "steer_hz", "yaw_rate", "psi", "x", "y", "speed", "turn_r", "turn_l", "aux")


# --- 1. tethered flight simulator ---------------------------------------------------------------------------------------------
def run_tethered(br, schedule, mode: str = "open", gain: float = 1.0, cancel=None, progress=None, transducer=None) -> dict:
    """The fly is fixed; the panorama around it rotates. `schedule` = [(seconds, omega_ext)]: the panorama's own rotation, rad/s,
    positive = rightward. open: the panorama follows the schedule whatever the fly does. closed: the panorama is also rotated by the
    fly's own yaw (slip = omega_ext - gain * yaw rate), so the fly can null the rotation. Yaw rate (CONNECTOME DNs, GAME RULE gain) is
    integrated into a heading that is only read out: the fly does not move. aux = the phase index."""
    if mode not in ("open", "closed"):
        raise ValueError("mode is open or closed")
    tr = transducer or Transducer()
    st = Steerer()
    dt = TICK_STEPS * float(br.dt)
    plan = [(i, max(1, int(round(sec / dt))), float(om)) for i, (sec, om) in enumerate(schedule)]
    ticks = sum(n for _, n, _ in plan)
    rows, psi, yaw, t, k = [], 0.0, 0.0, 0.0, 0
    try:
        for i, n, om in plan:
            for _ in range(n):
                _check(cancel, progress, k, ticks)
                slip = om - (gain * yaw if mode == "closed" else 0.0)
                tr.drive(br, slip)
                _advance(br)
                yaw = st.update(br)
                psi = wrap(psi + yaw * dt)
                t += dt
                k += 1
                rows.append((t, om, slip, st.s, yaw, psi, 0.0, 0.0, 0.0, br.hz("turn_r"), br.hz("turn_l"), i))
                _kin(br, k, 0.0, 0.0, yaw, f"{mode} loop, panorama {om:+.2f} rad/s", "tethered")
    finally:
        tr.release(br)
    return _finish(rows, dict(rig="tethered", mode=mode, gain=gain, schedule=[list(p) for p in schedule]), dt)


# --- 2. fly on a ball ---------------------------------------------------------------------------------------------------------
def run_ball(br, scene: str = "panorama", mode: str = "closed", seconds: float = 30.0, omega_ext=0.0, bar_offset: float = math.radians(90),
             bar_visible: bool = True, cancel=None, progress=None, transducer=None) -> dict:
    """A spherical treadmill with a virtual-reality world driven by the ball. The fly is held in place; what it does to the ball is read
    from its descending neurons: ball forward velocity from DNp09 (walk_speed), ball turning velocity from DNa01/02 R - L (Steerer).
    Those integrate into a position and heading in the VR.
    scene 'panorama': a distant stripe pattern; its own drift omega_ext (rad/s, rightward positive, a constant or a function of t) plus
      the fly's turning gives the slip the optomotor stage sees.
    scene 'bar': one vertical bar at `bar_offset` to the right of the starting heading (use a negative offset for the left), over a
      stationary textured background. The bar is a distant landmark, as in the classic closed-loop bar-fixation rigs: its bearing changes
      only when the fly turns, never when it walks (no parallax), and it is tracked (LC10, at BALL_BAR_DISTANCE for the duel's
      nearness factor). The background's slip is the fly's own turning.
    mode 'closed': the VR responds to the ball. mode 'open': the VR ignores the ball (the panorama keeps its drift; the bar stays at its
      starting bearing relative to the fly).
    bar_visible=False keeps the bar's bearing in the trace (aux) but never shows it to the fly: the control for bar fixation.
    aux = the bar's relative bearing (rad) in the bar scene."""
    if scene not in ("panorama", "bar") or mode not in ("open", "closed"):
        raise ValueError("scene is panorama or bar; mode is open or closed")
    tr = transducer or Transducer()
    st = Steerer()
    dt = TICK_STEPS * float(br.dt)
    ticks = max(1, int(round(seconds / dt)))
    om_f = omega_ext if callable(omega_ext) else (lambda _t, v=float(omega_ext): v)
    x = y = psi = yaw = 0.0
    t = 0.0
    rows = []
    try:
        for k in range(ticks):
            _check(cancel, progress, k, ticks)
            om = float(om_f(t)) if scene == "panorama" else 0.0
            closed = mode == "closed"
            slip = om - (yaw if closed else 0.0)
            tr.drive(br, slip)
            beta = 0.0
            if scene == "bar":
                beta = wrap(bar_offset - psi) if closed else wrap(bar_offset)
                if bar_visible:                                  # the control hides the bar: same VR, nothing to track
                    track_object(br, beta, BALL_BAR_DISTANCE)
            _advance(br)
            yaw = st.update(br)
            v = walk_speed(br)
            if closed:
                psi = wrap(psi + yaw * dt)
                x += v * math.sin(psi) * dt
                y += v * math.cos(psi) * dt
            else:
                psi = wrap(psi + yaw * dt)                  # the heading is read out, the VR does not move
            t += dt
            rows.append((t, om, slip, st.s, yaw, psi, x, y, v, br.hz("turn_r"), br.hz("turn_l"), beta))
            _kin(br, k, x, y, v, f"{scene} {mode} loop", "ball")
    finally:
        tr.release(br)
    return _finish(rows, dict(rig="ball", scene=scene, mode=mode, bar_offset=bar_offset, bar_visible=bool(bar_visible)), dt)


# --- 3. Buridan's paradigm ----------------------------------------------------------------------------------------------------
STRIPE_BEARINGS = (math.radians(90.0), math.radians(270.0))     # world bearings of the two stripes: east and west


def stripe_points(distance: float = STRIPE_DISTANCE) -> list[np.ndarray]:
    return [np.array([distance * math.sin(a), distance * math.cos(a)]) for a in STRIPE_BEARINGS]


def _reflect_circle(pos: np.ndarray, psi: float, radius: float) -> tuple[np.ndarray, float, bool]:
    r = float(np.hypot(*pos))
    if r < radius:
        return pos, psi, False
    n = pos / r
    f = np.array([math.sin(psi), math.cos(psi)])
    if float(f @ n) <= 0:                                      # already heading back in
        return pos / r * (radius - 1e-3), psi, False
    f2 = f - 2 * float(f @ n) * n                              # specular reflection (GAME RULE)
    return n * (radius - 1e-3), math.atan2(f2[0], f2[1]), True


def run_buridan(br, seconds: float = 120.0, stripes: bool = True, start_heading: float = 0.0, cancel=None, progress=None) -> dict:
    """Buridan's paradigm: a round platform (PLATFORM_RADIUS) with two opposite stripes beyond its edge (east and west, STRIPE_DISTANCE).
    The fly starts at the centre facing start_heading, walks at DNp09's speed and turns on DNa01/02 R - L. With stripes, the one nearer
    its heading is tracked (LC10, GAME RULE); without, no object is seen. At the edge it is reflected (GAME RULE). aux = number of edge
    contacts so far. Records the trajectory."""
    st = Steerer()
    dt = TICK_STEPS * float(br.dt)
    ticks = max(1, int(round(seconds / dt)))
    pts = stripe_points()
    pos, psi, yaw, t = np.zeros(2), float(start_heading), 0.0, 0.0
    rows, edges = [], 0
    for k in range(ticks):
        _check(cancel, progress, k, ticks)
        if stripes:
            cands = []
            for p in pts:
                d = p - pos
                cands.append((abs(wrap(math.atan2(d[0], d[1]) - psi)), d))
            ab, d = min(cands, key=lambda c: c[0])             # one target at a time: the stripe nearer the heading (GAME RULE)
            track_object(br, wrap(math.atan2(d[0], d[1]) - psi), float(np.hypot(*d)))
        _advance(br)
        yaw = st.update(br)
        v = walk_speed(br)
        psi = wrap(psi + yaw * dt)
        pos = pos + v * dt * np.array([math.sin(psi), math.cos(psi)])
        pos, psi, hit = _reflect_circle(pos, psi, PLATFORM_RADIUS)
        edges += int(hit)
        t += dt
        rows.append((t, 0.0, 0.0, st.s, yaw, psi, float(pos[0]), float(pos[1]), v, br.hz("turn_r"), br.hz("turn_l"), edges))
        _kin(br, k, float(pos[0]), float(pos[1]), v, "stripes" if stripes else "no stripes", "buridan")
    return _finish(rows, dict(rig="buridan", stripes=bool(stripes), platform_radius=PLATFORM_RADIUS, stripe_distance=STRIPE_DISTANCE,
                              stripe_bearings_deg=[90.0, 270.0]), dt)


# --- 4. olfactory four-field arena --------------------------------------------------------------------------------------------
ODOR_QUADRANTS = (0, 2)                  # quadrants 0 (x>0,y>0) and 2 (x<0,y<0) carry the odor; 1 and 3 carry air (GAME RULE)


def quadrant(x: float, y: float) -> int:
    """0: x >= 0, y >= 0; 1: x >= 0, y < 0; 2: x < 0, y < 0; 3: x < 0, y >= 0 (clockwise from the top right)."""
    return 0 if (x >= 0 and y >= 0) else 1 if (x >= 0 and y < 0) else 2 if (x < 0 and y < 0) else 3


def run_fourfield(br, seconds: float = 120.0, odor: str = "fruit", deliver: bool = True, start_heading: float = 0.0, cancel=None, progress=None) -> dict:
    """The four-field arena: a 1 m square in four quadrants, the odor in two opposite ones (ODOR_QUADRANTS), air in the other two, with
    sharp boundaries (GAME RULE; a real arena mixes at the seams). The fly starts at the centre and walks and turns as in Buridan's
    arena (walls reflect it), with no object to track. In an odor quadrant the odor's olfactory neurons are driven every tick
    (the game's scent poke, `br.poke("scent", odor, 0.3)`); deliver=False is the sham (same arena, no odor ever). aux = the quadrant.
    Time in odor and in air is counted after the first 10 s."""
    st = Steerer()
    dt = TICK_STEPS * float(br.dt)
    ticks = max(1, int(round(seconds / dt)))
    pos, psi, t = np.zeros(2) + 1e-3, float(start_heading), 0.0
    rows = []
    for k in range(ticks):
        _check(cancel, progress, k, ticks)
        q = quadrant(float(pos[0]), float(pos[1]))
        if deliver and q in ODOR_QUADRANTS:
            br.poke("scent", odor, SCENT_POKE)
        _advance(br)
        yaw = st.update(br)
        v = walk_speed(br)
        psi = wrap(psi + yaw * dt)
        pos = pos + v * dt * np.array([math.sin(psi), math.cos(psi)])
        for ax in (0, 1):                                      # walls reflect the heading component (GAME RULE)
            if abs(pos[ax]) > FIELD_HALF:
                pos[ax] = math.copysign(FIELD_HALF - 1e-3, pos[ax])
                f = np.array([math.sin(psi), math.cos(psi)])
                f[ax] = -f[ax]
                psi = math.atan2(f[0], f[1])
        t += dt
        rows.append((t, 0.0, 0.0, st.s, yaw, psi, float(pos[0]), float(pos[1]), v, br.hz("turn_r"), br.hz("turn_l"), q))
        _kin(br, k, float(pos[0]), float(pos[1]), v, f"quadrant {q}" + (" odor" if deliver and q in ODOR_QUADRANTS else " air"), "fourfield")
    return _finish(rows, dict(rig="fourfield", odor=odor, delivered=bool(deliver), field_half=FIELD_HALF,
                              odor_quadrants=list(ODOR_QUADRANTS)), dt)


def preference_index(full: np.ndarray, dt: float, settle_s: float = 10.0) -> dict:
    """PI = (time in odor quadrants - time in air quadrants) / (their sum), after settle_s (GAME RULE definition), plus the mean speed in
    each. `full` is a run's _full array."""
    a = np.asarray(full, float)
    a = a[a[:, 0] > settle_s]
    q = a[:, COLS.index("aux")].astype(int)
    odor = np.isin(q, ODOR_QUADRANTS)
    n_o, n_a = int(odor.sum()), int((~odor).sum())
    pi = (n_o - n_a) / max(1, n_o + n_a)
    sp = a[:, COLS.index("speed")]
    return dict(pi=float(pi), seconds_odor=round(n_o * dt, 2), seconds_air=round(n_a * dt, 2),
                speed_odor=float(sp[odor].mean()) if n_o else float("nan"), speed_air=float(sp[~odor].mean()) if n_a else float("nan"))


def stripe_deviation(full: np.ndarray, settle_s: float = 10.0, min_step: float = 1e-4) -> float:
    """Median stripe deviation in degrees (GAME RULE definition, named after Colomb et al. 2012's measure but not the same): the angle
    between the fly's direction of travel and the line joining the stripes (the east-west axis), folded to 0-90 degrees (0 = along the
    axis, 45 = chance for a random walk). Colomb et al. measure the angle to the centre of the stripe in front of the fly (0-120 degrees,
    chance 45 from simulated walks): both have chance at 45 and fall as walking lines up with the stripes, but the values differ.
    Steps of the trajectory shorter than min_step are ignored."""
    a = np.asarray(full, float)
    a = a[a[:, 0] > settle_s]
    dx, dy = np.diff(a[:, COLS.index("x")]), np.diff(a[:, COLS.index("y")])
    m = np.hypot(dx, dy) > min_step
    if not m.any():
        return float("nan")
    ang = np.degrees(np.arctan2(np.abs(dy[m]), np.abs(dx[m])))      # 0 = along east-west, 90 = across it
    return float(np.median(ang))


def transits(full: np.ndarray, settle_s: float = 10.0, zone: float = 0.8) -> int:
    """Transits between the stripes (GAME RULE definition): the fly's east-west position must reach the east edge zone (x > zone * radius)
    and then the west one (x < -zone * radius), or the reverse; each such crossing counts once."""
    a = np.asarray(full, float)
    a = a[a[:, 0] > settle_s]
    x = a[:, COLS.index("x")] / PLATFORM_RADIUS
    n, side = 0, 0
    for v in x:
        s = 1 if v > zone else -1 if v < -zone else 0
        if s and side and s != side:
            n += 1
        if s:
            side = s
    return n
