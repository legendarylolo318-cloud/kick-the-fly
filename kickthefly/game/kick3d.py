"""Kick the Fly in first person: the same live connectome fly, in a 3D room.

You walk around a living room (WASD, mouse to look) holding the selected tool. Everything the brain does is
unchanged from the 2D game (kick_the_fly.py): the same touch, heat, cold, smell, taste, wind, humidity, light and
looming inputs, the same descending-neuron reactions, pain, reward, learning, surgery and autopsy. Only the world
is new: the ragdoll, the fly's walking, turning and flight now happen in 3D, and tools are aimed with the camera.

Scale: the fly is a cartoon about 60 cm long. The 2D game's physics in pixels map to meters by S = 0.006, so every
tuned speed, impulse and threshold carries over.

Looming in 3D: the threats the fly sees are your body, the tool in your hand, a swatter mid-swing, thrown bombs
and the spider. Charge at it or swing fast and its giant fiber makes it dodge; walk up slowly and it won't notice.

    .venv\\Scripts\\python.exe kick_the_fly.py            (3D, the default)
    .venv\\Scripts\\python.exe kick_the_fly.py --2d       (the original 2D game)
"""
from __future__ import annotations

import math
import random

import threading
import time
from pathlib import Path

import moderngl
import numpy as np
import pygame

from kickthefly.core import crash
from kickthefly.game import kick_the_fly as k2
from kickthefly.game import gamepad, outdoors
from kickthefly.game.kick_the_fly import (ABD, FOOT, HEAD, KNEE, LINKS, MAX_HEALTH, N_P, PULL, RADIUS, REST, THRESH, THX, TOOLS,
                          TORCH_KEYS, TRIPOD, WING, drop_item)
from kickthefly.game.render3d import (P_BOOKS, P_CEIL, P_EYE, P_GRASS, P_ICE, P_NONE, P_PAPER, P_RUG, P_SKYDOME, P_STRIPES,
                      P_WALLPAPER, P_WATER, P_WOOD, Renderer, frame_from_x, look_at, perspective, rot_x, rot_y, rot_z, segment, trs)

S = 0.006                                   # meters per 2D pixel
RX, RY, RZ = 4.2, 3.0, 3.6                  # room half-width, height, half-depth
GRAV = 0.9 * S
STAND3 = k2.STAND * S
WATER3 = 0.6
EYE, CROUCH_EYE = 1.6, 0.85
LAMP3 = np.array([0.0, 2.2, -0.8])
FAN3 = np.array([-RX + 0.45, 0.0, 0.6])
PAPER3 = (-1.6, 1.6, -1.9, 1.1)             # flypaper x0, x1, z0, z1
THERMO_HALF3 = RX - 0.3                     # thermo arena: full cold at x = -3.9 m, full hot at +3.9 m (game rule)
REACH, GRAB_REACH = 3.0, 3.2
PANEL_W = k2.W - k2.PLAY_W                 # the brain panel's width in HUD units (390)
PANEL_MODES = (("solid", 255), ("see-through", 150), ("faint", 70), ("hidden", 0))
UI_MODES = ("crisp", "large")


def compute_layout(Wn: int, Hn: int, ui_mode: str, panel_mode: int, ui_scale: float = 1.0):
    """HUD units -> window pixels. The HUD always fills the window (no black bars). In "crisp" mode the scale is a
    whole number whenever the window is at least 760 px tall per step (1520p -> 2x), so text maps to whole pixels;
    "large" keeps the original 760-unit-tall HUD, smoothly scaled. Returns (scale, hud_w, hud_h, play_w, view_w):
    play_w is the HUD area left of the brain panel, view_w how wide the 3D view is."""
    s = max(1, Hn // 760) if ui_mode == "crisp" and Hn >= 760 else Hn / 760
    if ui_scale != 1.0:                             # Settings > UI scale (crisp mode snaps back to whole pixels)
        s = s * ui_scale
        if ui_mode == "crisp" and s >= 1:
            s = max(1, round(s))
    need = 1280 if PANEL_MODES[panel_mode][0] != "hidden" else 900
    if Wn / s < need:                               # too narrow for the HUD: scale down to fit the width
        s = Wn / need
    if Hn / s < 760:                                # ensure minimum vertical HUD units to prevent clipping
        s = Hn / 760
    hud_w, hud_h = max(1, int(round(Wn / s))), max(1, int(round(Hn / s)))
    hidden = PANEL_MODES[panel_mode][0] == "hidden"
    play_w = hud_w if hidden else hud_w - PANEL_W
    view_w = play_w if PANEL_MODES[panel_mode][0] == "solid" else hud_w
    return s, hud_w, hud_h, play_w, view_w
TOOL_SIZE = {"flick": 0.06, "swatter": 0.2, "bomb": 0.08, "torch": 0.09, "cleaner": 0.09, "zapper": 0.12,
             "freeze": 0.09, "spider": 0.08, "alcohol": 0.07, "laser": 0.08}
SKY_CLEAR = (0.08, 0.09, 0.11)
PLAYER_HP, PLAYER_RADIUS = 100.0, 0.32
PELLET_SPEED, PELLET_SPREAD, PELLET_DAMAGE = 9.0, 0.035, 9.0     # m/s, radians of scatter, health per hit
FIRE_COOLDOWN, REWARD_PULSE_S = 0.45, 0.45
# steering: tested on a still player 2.5-4 m away from 60-140 degrees off. Faster gains overshoot, because the brain
# loop (poke -> LC10 -> DNa02 -> 150 ms rate average) lags ~0.2 s; these hit 82% of shots, ~9 hits per 15 s.
STEER_DEADZONE_HZ = 1.5                                        # DNa02 R-L Hz below this is noise
STEER_GAIN = 0.004                                             # rad/frame of turning per Hz above the deadzone
STEER_MAX = 0.035                                              # rad/frame (2 rad/s)
TRACK_SCALE = 0.45                                             # sideness that drives LC10 fully
PUNISH_PULSE_S = 0.6                                           # getting hurt while it smells you: PPL1 fires this long
FLY_TOUCH_RADIUS3 = k2.FLY_TOUCH_RADIUS * S                    # how close two flies get before they bump, in meters

# furniture you and the fly bump into: (min corner, max corner)
COLLIDERS = [
    (np.array([-3.5, 0.0, -RZ]), np.array([-1.1, 0.62, -RZ + 1.0])),        # couch
    (np.array([RX - 0.45, 0.0, -1.9]), np.array([RX, 2.05, -0.1])),         # bookshelf
    (np.array([RX - 1.0, 0.0, RZ - 1.0]), np.array([RX - 0.2, 0.55, RZ - 0.2])),   # plant
]
ROOM_COLLIDERS = tuple(COLLIDERS)
# The fly's own bounds. Indoors they are the room's walls and ceiling; outdoors they are far beyond the ground you can
# walk on, so a fly that flies off really leaves (see set_world and Game3D._check_lost).
FLY_RX, FLY_RY, FLY_RZ = RX, RY, RZ
OUTDOOR_FAR = 90.0                          # far plane outdoors, metres; the room keeps its 40
DRAW_DIST = 38.0                            # scenery further than this from the camera isn't drawn
TUFT_DIST = 16.0                            # grass tufts are only drawn this close (they are small and many)
SPIDER_TOP = 3.0                            # outdoors the spider drops from a branch height, not from the sky
# The tool in your hand (draw_viewmodel): where it's held, in camera space (x right, y up, -z forward), the lens it's
# drawn with, and each tool's nozzle relative to that. tool_tip() fires effects from these same points.
VIEW_BASE = np.array([0.27, -0.26, -0.55])
VIEWMODEL_FOV = 60.0
NOZZLE = {"torch": (0.0, 0.07, -0.26), "cleaner": (0.0, 0.15, -0.06), "freeze": (0.0, 0.15, -0.06),
          "laser": (0.0, 0.062, -0.285), "zapper": (0.0, 0.12, -0.12), "swatter": (0.0, 0.3, -0.2)}


def scene_setup(game) -> tuple[dict, tuple, float]:
    """(lights, clear colour, far plane) for the arena the game is in. Outdoors the sun sits where the Lab puts it,
    the room's ceiling light is off, and distant scenery fades into haze instead of stopping at the far plane."""
    arena = k2.ARENAS[game.arena_i] if game is not None else "room"
    if arena in outdoors.OUTDOOR:
        az, el = outdoors.sun_now(game.lab_params, game.clock.now, game.cfg["brain.day_night"])
        sun = outdoors.sun_direction(az, el)
        up = max(0.0, float(sun[1]))
        day = 0.12 + 0.88 * outdoors.dusk(el)          # 1 unless the sun is low: night falls with the day/night cycle
        w = outdoors.spec(arena)
        haze = tuple(np.array((0.72, 0.78, 0.84)) * day)
        lights = dict(u_sun_dir=-sun, u_sun_col=tuple(np.array((1.05, 0.98, 0.86)) * (0.25 + 0.9 * up ** 0.5) * day),
                      u_sky=tuple(np.array(w.sky) * (0.45 + 0.55 * up) * day), u_ground=w.ground, u_lp0=(0.0, 100.0, 0.0),
                      u_lc0=(0, 0, 0), u_lp1=(0.0, 100.0, 0.0), u_lc1=(0, 0, 0), u_fog=(*haze, DRAW_DIST * 2.0))
        return lights, haze, OUTDOOR_FAR
    lamp_on = arena == "lamp"
    lights = dict(u_sun_dir=np.array([0.3, -0.55, 0.78]) / np.linalg.norm([0.3, -0.55, 0.78]), u_sun_col=(0.95, 0.88, 0.75),
                  u_sky=(0.42, 0.44, 0.5), u_ground=(0.24, 0.2, 0.17), u_lp0=(0.0, RY - 0.3, 0.0), u_lc0=(2.4, 2.2, 1.9),
                  u_lp1=tuple(LAMP3), u_lc1=(3.5, 2.8, 1.8) if lamp_on else (0, 0, 0), u_fog=(0.0, 0.0, 0.0, 0.0))
    return lights, SKY_CLEAR, 40.0


def set_world(arena: str, trees=()) -> None:
    """Rebind the world's bounds and obstacles for an arena. Every function here reads them at call time."""
    global RX, RY, RZ, FLY_RX, FLY_RY, FLY_RZ, COLLIDERS
    w = outdoors.spec(arena)
    RX, RY, RZ = w.rx, w.ry, w.rz
    FLY_RX, FLY_RY, FLY_RZ = w.fly_rx, w.fly_ry, w.fly_rz
    COLLIDERS = list(outdoors.tree_colliders(trees)) if arena in outdoors.OUTDOOR else list(ROOM_COLLIDERS)

HELP3D = (
    ("WASD", "walk (Shift sprint, Ctrl crouch)"),
    ("Mouse", "look; left click uses the tool in your hand"),
    ("1-9, 0, -, = / wheel", "pick a tool (= is the laser)"),
    ("Gamepad", "sticks walk and look, RT/ZR uses, LB/RB or hold Y (wheel) pick tools, Start/+ menu"),
    ("Tab", "free the mouse to click the brain panel and menus"),
    ("B", "big live brain view; click a neuron to inspect it"),
    ("O", "brain surgery"),
    ("T", "training: teach it to fear or like a smell (saved)"),
    ("X", "1v1 duel: the fly gets a blaster and can kill you"),
    ("E", "arena: room, fan, flypaper, pool, lamp, escape room, open field, orchard, thermo"),
    ("J", "outdoors: call back a fly that flew out of sight"),
    ("P / I", "pain neurons / immortal mode"),
    ("M", "mute"),
    ("F12 / G", "save a screenshot / a GIF of the last 6 s"),
    ("Shift+R", "record a video (MP4 with ffmpeg, else GIF; again to stop)"),
    ("V", "brain panel: solid, see-through, faint, hidden"),
    ("U", "menu size: crisp (whole-pixel scaling) or large"),
    ("F11", "fullscreen"),
    ("N", "spawn another fly, each with its own brain (16; one per CPU core, up to 32, on Numba)"),
    ("F", "cycle focused fly (in photo mode: autofocus)"),
    ("R", "reset to a single fresh fly"),
    ("Z  [  ]  .", "pause time, slower, faster, single step"),
    ("Esc", "close a panel, or open the menu (settings, save, quit)"),
)


def _rest3() -> np.ndarray:
    """Rest pose in the fly's body frame (forward, up, side), meters; legs splay to both sides."""
    R = np.zeros((N_P, 3))
    R[:, 0] = REST[:, 0] * S
    R[:, 1] = -REST[:, 1] * S
    for i in range(3):
        R[KNEE[i + 3], :2] = R[KNEE[i], :2]
        R[FOOT[i + 3], :2] = R[FOOT[i], :2]
        R[KNEE[i], 2], R[KNEE[i + 3], 2] = 0.13, -0.13
        R[FOOT[i], 2], R[FOOT[i + 3], 2] = 0.25, -0.25
    R[WING[1], :2] = R[WING[0], :2]
    R[WING[0], 2], R[WING[1], 2] = 0.07, -0.07
    return R


REST3 = _rest3()
RAD3 = RADIUS * S
LINK_LEN3 = [float(np.linalg.norm(REST3[a] - REST3[b])) for a, b, _, _ in LINKS]


def body_axes(yaw: float):
    fwd = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    side = np.array([-math.sin(yaw), 0.0, math.cos(yaw)])
    return fwd, np.array([0.0, 1.0, 0.0]), side


def to_world(off: np.ndarray, yaw: float) -> np.ndarray:
    f, u, s = body_axes(yaw)
    return np.outer(off[:, 0], f) + np.outer(off[:, 1], u) + np.outer(off[:, 2], s)


def angle_to(v) -> float:
    return math.atan2(float(v[2]), float(v[0]))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def push_out_boxes(p: np.ndarray, r) -> None:
    """Push points (n, 3) out of the furniture boxes along the shallowest axis."""
    rmax = float(np.max(r)) if np.ndim(r) else float(r)
    pmin, pmax = p.min(axis=0) - rmax, p.max(axis=0) + rmax
    for lo, hi in COLLIDERS:
        if (pmax < lo).any() or (pmin > hi).any():     # nowhere near this box (outdoors: most tree trunks)
            continue
        inside = np.all((p > lo - r[:, None]) & (p < hi + r[:, None]), axis=1) if np.ndim(r) else \
            np.all((p > lo - r) & (p < hi + r), axis=1)
        for i in np.flatnonzero(inside):
            ri = r[i] if np.ndim(r) else r
            pen = np.concatenate([p[i] - (lo - ri), (hi + ri) - p[i]])        # distance to each face
            k = int(np.argmin(pen))
            if k < 3:
                p[i, k] = lo[k] - ri
            else:
                p[i, k - 3] = hi[k - 3] + ri


class Fly3D:
    """The 2D ragdoll lifted into 3D: the same particles, links, pose pull and flight, facing any direction."""

    def __init__(self, pos_xz=(0.0, -0.8), yaw: float = math.pi / 2):
        self.yaw = self.yaw_target = yaw
        self.p = to_world(REST3, yaw) + (pos_xz[0], STAND3, pos_xz[1])
        self.prev = self.p.copy()
        self.anchor = np.array(pos_xz, float)
        self.phase = 0.0
        self.grabbed: int | None = None
        self.stun_until = self.escape_until = self.walk_until = self.back_until = self.flail_until = 0.0
        self.run = False
        self.turn_ready = self.escape_ready = 0.0
        self.recover, self.hurt = 1.0, 0.0
        self.last_hit = self.p[THX].copy()
        self.action = "idle"
        self.health = MAX_HEALTH
        self.dead_at: float | None = None
        self.char = self.soak = self.melt = self.frost = self.venom = 0.0
        self.burn_until = self.zap_until = self.eating_until = 0.0
        self.dissolved_at = self.frozen_at = self.shattered_at = None
        self.hover = self.p[THX].copy()
        self.fly_target = self.hover.copy()
        self.wander = False
        self.power = 1.0
        self.wrapped = False
        self.arena = "room"
        self.wind = np.zeros(3)
        self.wet = 0.0
        self.inebriation = 0.0
        self.stuck: dict[int, np.ndarray] = {}
        self.perch = None                         # the orchard fruit it is sitting on, if any (game rule)

    @property
    def dead(self) -> bool:
        return self.dead_at is not None

    @property
    def flying(self) -> bool:
        return (self.escape_until > 0 and self.grabbed is None and not self.dead and not self.wrapped and self.melt < 0.3
                and self.frost < 0.5 and self.venom < 0.5 and self.wet <= 0 and len(self.stuck) < 2)

    def nearest_to_ray(self, eye, d, reach: float, max_perp: float):
        rel = self.p - eye
        t = rel @ d
        perp = np.linalg.norm(rel - np.outer(t, d), axis=1) - RAD3
        ok = (t > 0) & (t < reach) & (perp < max_perp)
        if not ok.any():
            return None, None, None
        i = int(np.flatnonzero(ok)[np.argmin(perp[ok])])
        return i, float(t[i]), float(perp[i])

    def impulse(self, i: int, v) -> None:
        self.prev[i] -= v

    def stun(self, now: float, s: float) -> None:
        self.stun_until = max(self.stun_until, now + s)
        self.hurt = 1.0
        self.escape_until = min(self.escape_until, now)

    def away_from(self, point) -> np.ndarray:
        v = self.p[THX] - np.asarray(point, float)
        v[1] = 0
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else body_axes(self.yaw)[0]

    def escape(self, now: float, seconds: float = 2.4, wander: bool = False) -> None:
        if self.arena in outdoors.OUTDOOR:              # open sky: the same escape keeps going (game rule)
            seconds *= 2.5
        away = self.away_from(self.last_hit)
        launch = away * 4.0 * S + np.array([0, 9.0 * S, 0])
        if getattr(self, "inebriation", 0.0) > 0:
            launch += np.random.normal(0, 2.5 * S * self.inebriation, 3)
        self.prev[:] = self.p - launch
        self.hover = self.p[THX].copy()
        self.wander = wander
        self._new_target(None if wander else away)
        self.escape_until = now + seconds
        delay = 2.0 * getattr(self, "inebriation", 0.0)
        self.escape_ready = now + seconds + 1.0 + delay

    def _new_target(self, away=None) -> None:
        if self.arena in outdoors.OUTDOOR:              # open sky: escapes carry it metres away, not across a room
            if away is None:
                t = np.array([random.uniform(-RX + 1, RX - 1), random.uniform(1.0, 4.0), random.uniform(-RZ + 1, RZ - 1)])
            else:
                t = self.p[THX] + away * random.uniform(4.0, 12.0) + np.array([0, random.uniform(1.5, 5.0), 0])
            self.fly_target = np.clip(t, (-FLY_RX + 1, 0.5, -FLY_RZ + 1), (FLY_RX - 1, FLY_RY - 1, FLY_RZ - 1))
            return
        for _ in range(12):
            if away is None:
                t = np.array([random.uniform(-RX + 0.8, RX - 0.8), random.uniform(0.8, 2.4), random.uniform(-RZ + 0.8, RZ - 0.8)])
            else:
                t = self.p[THX] + away * random.uniform(1.3, 2.3) + np.array([0, random.uniform(0.5, 1.4), 0])
                t = np.clip(t, (-RX + 0.7, 0.8, -RZ + 0.7), (RX - 0.7, 2.4, RZ - 0.7))
            if not any(np.all((t > lo - 0.4) & (t < hi + 0.4)) for lo, hi in COLLIDERS):
                break
            away = None
        self.fly_target = t

    def step(self, now: float, pin) -> list[tuple[int, float]]:
        """One 60 Hz frame. Returns (particle, impact speed in 2D px/frame units) for hard contacts."""
        if self.frozen_at is not None:
            self.prev = self.p.copy()
            return []
        flying = now < self.escape_until and self.flying
        if flying:
            self.recover = 1.0
        elif self.grabbed is not None or now < self.stun_until or self.dead or self.wrapped:
            self.recover = 0.0
        else:
            self.recover = min(1.0, self.recover + 1 / 30)
        height = self.p[THX, 1] - STAND3
        stiff = (1 - self.melt) ** 0.5 * (1 - self.frost) * (1 - self.venom)
        strength = self.recover * (1.0 if flying else float(np.clip(1 - height / (160 * S), 0, 1)) * stiff)
        if self.grabbed is not None or (self.arena == "pool" and not flying and self.p[THX, 1] < WATER3 + 30 * S):
            strength = 0.0
        shrink = 1 - 0.35 * self.melt

        v = (self.p - self.prev) * (0.992 - 0.12 * strength)
        speed = np.linalg.norm(v, axis=1)
        v *= np.minimum(1.0, 60 * S / np.maximum(speed, 1e-9))[:, None]
        self.prev = self.p.copy()
        self.p += v
        self.p[:, 1] -= GRAV * (1 - strength)
        if self.arena == "pool":
            sub = self.p[:, 1] < WATER3
            if sub.any():
                self.p[sub, 1] += 1.35 * S
                self.p[sub] -= (self.p[sub] - self.prev[sub]) * 0.12
        if self.wind.any():
            w = np.full(N_P, 0.35)
            w[list(WING)] *= 2.5
            self.p += np.outer(w, self.wind)
            if flying:
                self.hover += self.wind * 3.0

        if flying:
            self._fly(now)
        elif strength > 0:
            self._pose(now, strength)
        else:
            self.anchor = self.p[THX, [0, 2]].copy()
        if self.dead:
            idx = list(KNEE + FOOT)
            self.p[idx] += (self.p[THX] - self.p[idx]) * 0.02
        elif now < self.flail_until or (self.grabbed is not None and random.random() < 0.5):
            idx = list(KNEE + FOOT)
            self.p[idx] += np.random.normal(0, 2.5 * S, (12, 3))

        if getattr(self, "inebriation", 0.0) > 0 and not self.dead:
            # Uncoordinated motor tremors (GAME RULE: alcohol inebriation)
            self.p += np.random.normal(0, 1.8 * S * self.inebriation, (N_P, 3))

        posed = strength > 0.5
        for _ in range(6):
            if self.grabbed is not None:
                self.p[self.grabbed] = pin
            for i, at in self.stuck.items():
                self.p[i] = at
            for (a, b, stiff_k, shape), rest in zip(LINKS, LINK_LEN3):
                if shape and posed:
                    continue
                d = self.p[b] - self.p[a]
                dist = math.sqrt(float(d @ d)) or 1e-9
                corr = d * (0.5 * stiff_k * (dist - rest * shrink) / dist)
                self.p[a] += corr
                self.p[b] -= corr
            self._clamp()
        if self.grabbed is not None:
            self.p[self.grabbed] = pin
        for i, at in self.stuck.items():
            self.p[i] = at
            self.prev[i] = at
        if self.arena in ("flypaper", "escaperoom") and self.grabbed is None:
            x0, x1, z0, z1 = PAPER3
            for i in range(N_P):
                if i not in self.stuck and x0 < self.p[i, 0] < x1 and z0 < self.p[i, 2] < z1 and self.p[i, 1] <= RAD3[i] + 1.5 * S:
                    self.stuck[i] = self.p[i].copy()
        self.hurt = max(0.0, self.hurt - 1 / 20)
        return self._contacts()

    def _fly(self, now: float) -> None:
        d = self.fly_target - self.hover
        dist = float(np.linalg.norm(d))
        speed = (2.5 + 2.5 * min(self.power, 2.0)) * S
        if getattr(self, "perch", None) is not None and dist < 25 * S:
            self.hover += d * 0.3                       # landed on a fruit: stay on it (game rule)
        elif dist < 25 * S:
            self._new_target()
        else:
            self.hover += d / dist * min(speed, dist)
        if math.hypot(d[0], d[2]) > 8 * S:
            self.yaw_target = angle_to(d)
        self.yaw += wrap(self.yaw_target - self.yaw) * 0.15
        self.anchor = self.hover[[0, 2]].copy()
        self.action = "flying"
        bob = 4 * S * math.sin(now * 9)
        if getattr(self, "inebriation", 0.0) > 0:
            # Erratic 3D flight drift (GAME RULE: alcohol inebriation)
            self.hover += np.array([math.sin(now * 3.7) * 4.5 * S * self.inebriation,
                                    math.sin(now * 2.5) * 2.0 * S * self.inebriation,
                                    math.cos(now * 2.9) * 4.5 * S * self.inebriation])
        tgt = to_world(REST3, self.yaw) + self.hover + (0, bob, 0)
        f, u, _ = body_axes(self.yaw)
        for i in range(6):
            tgt[FOOT[i]] = tgt[KNEE[i]] - f * 6 * S - u * 14 * S
        dd = tgt - self.p
        self.p += dd * 0.25
        self.prev += dd * 0.2

    def _blocked(self, xz) -> bool:
        if not (-RX + 0.5 < xz[0] < RX - 0.5 and -RZ + 0.5 < xz[1] < RZ - 0.5):
            return True
        p = np.array([xz[0], 0.2, xz[1]])
        return any(np.all((p > lo - 0.35) & (p < hi + 0.35)) for lo, hi in COLLIDERS)

    def _pose(self, now: float, strength: float) -> None:
        walking, backing = now < self.walk_until, now < self.back_until
        speed = (5.0 if self.run else 2.3) * S
        v = -1.7 * S if backing else speed if walking else 0.0
        f, u, _ = body_axes(self.yaw)
        if v and self._blocked(self.anchor + f[[0, 2]] * v * 12):
            self.yaw_target = self.yaw + math.pi + random.uniform(-0.7, 0.7)    # walked into a wall: turn around
            v = 0.0
        self.yaw += float(np.clip(wrap(self.yaw_target - self.yaw), -0.09, 0.09))
        self.anchor += f[[0, 2]] * v
        self.anchor += 0.05 * (self.p[THX, [0, 2]] - self.anchor)
        self.action = "back up" if backing else ("run" if self.run else "walk") if walking else "idle"
        if v:
            self.phase += 0.12 * abs(v) / S
        s = 1 - 0.35 * self.melt
        bob = 1.5 * S * math.sin(now * 2.2)
        tgt = to_world(REST3 * s, self.yaw) + (self.anchor[0], STAND3 * s + bob, self.anchor[1])
        if getattr(self, "inebriation", 0.0) > 0:
            # Stumbling gait and wobbling roll/pitch drift (GAME RULE: alcohol inebriation)
            wobble = math.sin(now * 3.8) * 6.0 * S * self.inebriation
            self.yaw += 0.04 * self.inebriation * math.sin(now * 2.7)
            tgt[:, 0] += wobble
            tgt[:, 1] += math.cos(now * 4.5) * 3.0 * S * self.inebriation
            if v:
                self.phase += 0.06 * self.inebriation * math.sin(now * 7.0)
        for leg in range(6):
            ph = self.phase + math.pi * TRIPOD[leg]
            if v:
                tgt[FOOT[leg]] += f * 14 * S * math.cos(ph) * math.copysign(1, v)
                tgt[FOOT[leg], 1] += 12 * S * max(0.0, math.sin(ph))
        d = tgt - self.p
        self.p += d * (PULL * strength)[:, None]
        self.prev += d * (PULL * strength * 0.85)[:, None]

    def _clamp(self) -> None:
        r = RAD3
        np.clip(self.p[:, 0], -FLY_RX + r, FLY_RX - r, out=self.p[:, 0])
        np.clip(self.p[:, 2], -FLY_RZ + r, FLY_RZ - r, out=self.p[:, 2])
        np.clip(self.p[:, 1], r, FLY_RY - r, out=self.p[:, 1])
        push_out_boxes(self.p, r)

    def _contacts(self) -> list[tuple[int, float]]:
        hits = []
        r = RAD3
        v = self.p - self.prev
        eps = 0.5 * S
        for i in range(N_P):
            x, y, z = self.p[i]
            if y <= r[i] + eps and v[i, 1] < 0:
                if -v[i, 1] > 9 * S:
                    hits.append((i, -v[i, 1] / S))
                self.prev[i, 1] = y + v[i, 1] * 0.35
                self.prev[i, 0] = x - v[i, 0] * 0.75
                self.prev[i, 2] = z - v[i, 2] * 0.75
            elif y >= FLY_RY - r[i] - eps and v[i, 1] > 0:
                if v[i, 1] > 9 * S:
                    hits.append((i, v[i, 1] / S))
                self.prev[i, 1] = y + v[i, 1] * 0.35
            for ax, lim in ((0, FLY_RX), (2, FLY_RZ)):
                c = self.p[i, ax]
                if (c <= -lim + r[i] + eps and v[i, ax] < 0) or (c >= lim - r[i] - eps and v[i, ax] > 0):
                    if abs(v[i, ax]) > 9 * S:
                        hits.append((i, abs(v[i, ax]) / S))
                    self.prev[i, ax] = c + v[i, ax] * 0.4
        return hits


class Player:
    def __init__(self):
        self.pos = np.array([0.0, 2.6])
        self.yaw, self.pitch = -math.pi / 2, -0.32
        self.eye_h = EYE
        self.vel = np.zeros(2)
        self.walk_phase = 0.0

    @property
    def eye(self) -> np.ndarray:
        return np.array([self.pos[0], self.eye_h + 0.012 * math.sin(self.walk_phase * 2), self.pos[1]])

    def forward(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array([cp * math.cos(self.yaw), math.sin(self.pitch), cp * math.sin(self.yaw)])

    def basis(self):
        f = self.forward()
        r = np.array([-math.sin(self.yaw), 0.0, math.cos(self.yaw)])
        u = np.cross(r, f)
        return f, r, u

    def to_world(self, local) -> np.ndarray:
        """Camera space (x right, y up, -z forward) to world."""
        f, r, u = self.basis()
        return self.eye + r * local[0] + u * local[1] - f * local[2]

    def update(self, dt: float, keys, rel) -> None:
        self.yaw += rel[0] * 0.0024
        self.pitch = float(np.clip(self.pitch - rel[1] * 0.0024, -1.45, 1.45))
        fh = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        rh = np.array([-math.sin(self.yaw), math.cos(self.yaw)])
        want = fh * (keys["w"] - keys["s"]) + rh * (keys["d"] - keys["a"])
        n = float(np.linalg.norm(want))
        crouch = keys["crouch"]
        speed = 1.4 if crouch else 5.5 if keys["sprint"] else 3.0
        target = want / n * speed if n > 0 else np.zeros(2)
        self.vel += (target - self.vel) * min(1.0, dt * 12)
        self.pos += self.vel * dt
        self.pos[0] = float(np.clip(self.pos[0], -RX + 0.3, RX - 0.3))
        self.pos[1] = float(np.clip(self.pos[1], -RZ + 0.3, RZ - 0.3))
        for lo, hi in COLLIDERS:                         # slide along furniture
            q = np.array([self.pos[0], 0.5, self.pos[1]])
            if np.all((q > lo - (0.3, 0, 0.3)) & (q < hi + (0.3, 0, 0.3))):
                pen = [q[0] - (lo[0] - 0.3), (hi[0] + 0.3) - q[0], q[2] - (lo[2] - 0.3), (hi[2] + 0.3) - q[2]]
                k = int(np.argmin(pen))
                if k == 0:
                    self.pos[0] = lo[0] - 0.3
                elif k == 1:
                    self.pos[0] = hi[0] + 0.3
                elif k == 2:
                    self.pos[1] = lo[2] - 0.3
                else:
                    self.pos[1] = hi[2] + 0.3
        self.eye_h += ((CROUCH_EYE if crouch else EYE) - self.eye_h) * min(1.0, dt * 10)
        if float(np.linalg.norm(self.vel)) > 0.3:
            self.walk_phase += dt * float(np.linalg.norm(self.vel)) * 2.2


class FreeCamera:
    """Detached 6-DOF flying camera for photo mode."""

    def __init__(self, pos, yaw: float, pitch: float):
        self.pos = np.array(pos, dtype=float)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.vel = np.zeros(3, dtype=float)

    @property
    def eye(self) -> np.ndarray:
        return self.pos

    def forward(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array([cp * math.cos(self.yaw), math.sin(self.pitch), cp * math.sin(self.yaw)])

    def basis(self):
        f = self.forward()
        r = np.array([-math.sin(self.yaw), 0.0, math.cos(self.yaw)])
        rn = np.linalg.norm(r)
        r = r / (rn if rn > 1e-6 else 1.0)
        u = np.cross(r, f)
        un = np.linalg.norm(u)
        u = u / (un if un > 1e-6 else 1.0)
        return f, r, u

    def update(self, dt: float, keys: dict, rel: tuple[float, float]) -> None:
        self.yaw += rel[0] * 0.0024
        self.pitch = float(np.clip(self.pitch - rel[1] * 0.0024, -1.55, 1.55))
        f, r, u = self.basis()
        want = np.zeros(3, dtype=float)
        if keys.get("w"): want += f
        if keys.get("s"): want -= f
        if keys.get("d"): want += r
        if keys.get("a"): want -= r
        if keys.get("up"): want += np.array([0.0, 1.0, 0.0])
        if keys.get("down_fly"): want -= np.array([0.0, 1.0, 0.0])
        n = float(np.linalg.norm(want))
        speed = 5.5 if keys.get("sprint") else 2.5
        target = want / n * speed if n > 0 else np.zeros(3)
        self.vel += (target - self.vel) * min(1.0, dt * 10)
        self.pos += self.vel * dt
        self.pos[0] = float(np.clip(self.pos[0], -RX + 0.15, RX - 0.15))
        self.pos[1] = float(np.clip(self.pos[1], 0.08, RY - 0.08))
        self.pos[2] = float(np.clip(self.pos[2], -RZ + 0.15, RZ - 0.15))


def _c(col, alpha: float | None = None):
    rgb = tuple(c / 255.0 for c in col[:3])
    a = col[3] / 255.0 if len(col) > 3 else 1.0
    return rgb + ((a if alpha is None else alpha),)


class Game3D(k2.Game):
    three_d = True

    def __init__(self, hud: pygame.Surface, brain, view, graph=None, weights=None, cfg=None):
        self.player = Player()
        self.photo_mode = False
        self.free_cam: FreeCamera | None = None
        self.photo_fov = 70.0
        self.photo_dof = 0.0
        self.photo_focus = 1.8
        self.photo_hide_ui = False
        self.panel_mode, self.ui_mode = 0, "crisp"
        self.panel_alpha = 255
        self.look = False
        self.key_codes: dict[str, int] = {}
        self.mouse_logical = (0, 0)
        super().__init__(hud, brain, view, graph, weights, cfg=cfg)
        self.swing_t = -9.0
        self.flick_t = -9.0
        self.throw_t = -9.0
        self.hold_dist = 1.2
        self.popups3: list = []
        set_world("room")                              # the room's furniture is built at the room's size
        self._room = self._build_room()
        self.world, self.scenery, self.orchard = "room", outdoors.scenery("room"), None
        self.on_arena_changed(self.clock.now)          # config.toml may have chosen an outdoor arena (or the escape room)
        self.quit_armed = False
        self.view_w, self.hud_h = k2.PLAY_W, k2.H
        self.hint_extra = "V panel   X 1v1   F11 fullscreen   H help"
        self.duel = False
        self.player_hp, self.player_dead_at = PLAYER_HP, None
        self.duel_stats = dict(shots=0, hits=0, deaths=0)
        self.pad = gamepad.Gamepad(self.cfg)
        self._pad_using = False

    # --- gamepad (game/gamepad.py): merged into the keyboard and mouse input, never replacing it ----------------------
    def pad_tick(self, dt: float, now: float) -> tuple[dict, tuple[float, float]]:
        """Read the pad once per frame. Returns movement keys to OR into the keyboard's and a look delta in mouse
        units (it goes through update_player like the mouse, so it is pre-scaled to ignore the mouse settings)."""
        pad = self.pad
        keys, (lx, ly), down = pad.poll()
        if not pad.connected:
            return {}, (0.0, 0.0)
        if "menu" in down:
            esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="\x1b", scancode=0)
            self.handle3d(esc, now, lambda q: q)
        if self.menu.open:
            return {}, (0.0, 0.0)
        if "big_view" in down:
            self.do_action("big_view", now)
        if "tool_next" in down:
            self.tool = (self.tool + 1) % len(TOOLS)
        if "tool_prev" in down:
            self.tool = (self.tool - 1) % len(TOOLS)
        rel = (0.0, 0.0)
        if pad.was.get("tool_wheel"):                   # held: the right stick points at a tool instead of looking
            pad.wheel_open = True
            pad.wheel(len(TOOLS))
        else:
            if pad.wheel_open and pad.wheel_pick is not None:
                self.tool = pad.wheel_pick
            pad.wheel_open, pad.wheel_pick = False, None
            c = self.cfg
            speed = float(c["controls.pad_look_speed"]) * dt / 0.0024 / max(0.1, float(c["controls.mouse_sensitivity"]))
            flip = (-1 if c["controls.pad_invert_y"] else 1) * (-1 if c["controls.invert_y"] else 1)
            rel = (lx * speed, ly * speed * flip)
        if "use" in down:
            if self.look:
                if not getattr(self, "autopilot", False) and not self.photo_mode:
                    self.use_tool3d(now)
                    self._pad_using = True
            elif self.player_dead_at is None and not self._overlay_open():
                self.set_look(True)                      # like clicking the room
        if self._pad_using and not pad.was.get("use"):  # trigger let go: what a mouse-up does
            self._pad_using = False
            if not self.fly.wrapped:
                self.fly.grabbed = None
            self.torching = False
            if hasattr(self, "laser_state"):
                self.laser_state.trigger_release()
        return keys, rel

    def _draw_tool_wheel(self, hud) -> None:
        cx, cy, r = k2.PLAY_W // 2, self.hud_h // 2, 150
        card = pygame.Surface((2 * r + 120, 2 * r + 80), pygame.SRCALPHA)
        pygame.draw.circle(card, (8, 10, 16, 190), (r + 60, r + 40), r + 50)
        hud.blit(card, (cx - r - 60, cy - r - 40))
        n = len(TOOLS)
        for i, t in enumerate(TOOLS):
            a = i / n * 2 * math.pi
            x, y = cx + r * math.sin(a), cy - r * math.cos(a)
            on = i == self.pad.wheel_pick or (self.pad.wheel_pick is None and i == self.tool)
            pygame.draw.circle(hud, (70, 120, 200) if on else (30, 36, 48), (int(x), int(y)), 30)
            self._text(hud, t[0], (int(x), int(y)), k2.INK if on else k2.TEXT, self.f_small, "center")
        self._text(hud, "release to pick", (cx, cy), k2.LABEL, self.f_small, "center")

    # --- settings -------------------------------------------------------------------------------------------------
    def save_extra(self, arrays: dict, now: float) -> dict:
        pl = self.player
        return dict(
            sugars3=[dict(p=[float(x) for x in sg["p"]], v=[float(x) for x in sg["v"]], left=float(sg["left"]),
                          landed=bool(sg["landed"])) for sg in self.sugars3],
            alcohols3=[dict(p=[float(x) for x in al["p"]], v=[float(x) for x in al["v"]], left=float(al["left"]),
                            landed=bool(al["landed"])) for al in getattr(self, "alcohols3", [])],
            decoys3=[dict(p=[float(x) for x in dec["p"]], v=[float(x) for x in dec["v"]],
                          landed=bool(dec.get("landed", False)), yaw=float(dec.get("yaw", 0.0)))
                     for dec in getattr(self, "decoys3", [])],
            player=dict(pos=[float(x) for x in pl.pos], yaw=pl.yaw, pitch=pl.pitch, eye_h=pl.eye_h,
                        vel=[float(x) for x in pl.vel]),
            duel=bool(self.duel), player_hp=float(self.player_hp), duel_stats=dict(self.duel_stats),
            valence=float(self.valence),
            orchard=self.orchard.state(now) if getattr(self, "orchard", None) is not None else None)

    def load_extra(self, extra: dict, z, now: float) -> None:
        self.sugars3 = [dict(p=np.array(sg["p"]), v=np.array(sg["v"]), left=sg["left"], landed=sg["landed"])
                        for sg in extra.get("sugars3", [])]
        self.alcohols3 = [dict(p=np.array(al["p"]), v=np.array(al["v"]), left=al["left"], landed=al["landed"])
                          for al in extra.get("alcohols3", [])]
        self.decoys3 = [dict(p=np.array(dec["p"], float), v=np.array(dec["v"], float),
                             landed=bool(dec.get("landed", False)), yaw=float(dec.get("yaw", 0.0)))
                        for dec in extra.get("decoys3", [])]
        self.grabbed_decoy3 = None
        pl, sp = self.player, extra.get("player")
        if sp:
            pl.pos, pl.yaw, pl.pitch, pl.eye_h = np.array(sp["pos"]), sp["yaw"], sp["pitch"], sp["eye_h"]
            pl.vel = np.array(sp["vel"])
        self.duel = bool(extra.get("duel", False))
        self.player_hp, self.player_dead_at = float(extra.get("player_hp", PLAYER_HP)), None
        self.duel_stats = dict(extra.get("duel_stats", self.duel_stats))
        self.valence = float(extra.get("valence", 0.0))
        if getattr(self, "orchard", None) is not None and extra.get("orchard"):
            self.orchard.load_state(extra["orchard"], now)

    def apply_setting(self, key: str) -> None:
        super().apply_setting(key)
        c = self.cfg
        if key == "graphics.panel_mode":
            self.panel_mode = [m for m, _ in PANEL_MODES].index(c[key])
            self.panel_alpha = PANEL_MODES[self.panel_mode][1]
        elif key == "graphics.menu_size":
            self.ui_mode = c[key]
        elif key == "access.palette" or key == "brain.seed":
            pass
        self.key_codes = {a: pygame.key.key_code(k) for a, k in c.keys.items() if k}

    def set_setting(self, key, value, save: bool = True, force: bool = False) -> None:
        super().set_setting(key, value, save, force)
        if key == "keys":
            self.key_codes = {a: pygame.key.key_code(k) for a, k in self.cfg.keys.items() if k}

    def open_menu(self, screen: str = "pause") -> None:
        self.set_look(False)
        super().open_menu(screen)

    def held(self, kp) -> dict:
        """Movement keys held this frame, from the rebindable bindings (plus C as a second crouch key)."""
        kc = self.key_codes

        def down(action):
            code = kc.get(action)
            try:
                return bool(code is not None and kp[code])
            except IndexError:
                return False

        return dict(w=down("forward"), s=down("back"), a=down("left"), d=down("right"),
                    sprint=down("sprint") or kp[pygame.K_RSHIFT],
                    crouch=down("crouch") or kp[pygame.K_c],
                    up=kp[pygame.K_SPACE] or kp[pygame.K_e],
                    down_fly=down("crouch") or kp[pygame.K_c] or kp[pygame.K_q] or kp[pygame.K_LCTRL])

    def update_player(self, dt: float, keys, rel) -> None:
        """You move and look in real time, even in slow motion or while time is paused."""
        if self.player_dead_at is None and self.look and not self.menu.open:
            c = self.cfg
            sens = c["controls.mouse_sensitivity"]
            rel = (rel[0] * sens, rel[1] * sens * (-1 if c["controls.invert_y"] else 1))
            if self.photo_mode and self.free_cam is not None:
                self.free_cam.update(dt, keys, rel)
            else:
                self.player.update(dt, keys, rel)

    def toggle_photo_mode(self) -> None:
        self.photo_mode = not self.photo_mode
        if self.photo_mode:
            self.free_cam = FreeCamera(self.player.eye.copy(), self.player.yaw, self.player.pitch)
            self.photo_fov = float(self.cfg["controls.fov"])
            self.photo_dof = float(self.cfg.get("graphics.photo_dof", 0.0))
            self.photo_focus = self.nearest_fly_dist()
            self.set_look(True)
            self.note("PHOTO MODE: WASD/Space/C fly, Mouse aim, [ ] FOV, , . DOF, F12 snap", source="rule")
        else:
            self.free_cam = None
            self.note("PHOTO MODE off", source="rule")

    def nearest_fly_dist(self) -> float:
        eye = self.free_cam.eye if (self.photo_mode and self.free_cam) else self.player.eye
        fwd = self.free_cam.forward() if (self.photo_mode and self.free_cam) else self.player.forward()
        best_dist = float("inf")
        flies = getattr(self, "flies3", []) or ([self.fly] if getattr(self, "fly", None) else [])
        for fly in flies:
            pos = fly.p[THX]
            vec = pos - eye
            d = float(np.linalg.norm(vec))
            if d < 0.05:
                continue
            cos = float(np.dot(vec / d, fwd))
            if cos > 0.2:
                if d < best_dist:
                    best_dist = d
        if best_dist < 100.0:
            return round(best_dist, 2)
        if getattr(self, "fly", None):
            return round(float(np.linalg.norm(self.fly.p[THX] - eye)), 2)
        return 1.8

    def take_photo(self) -> None:
        self.want_png = True
        self.sound.play("shutter")

    def _draw_photo_viewfinder(self, hud: pygame.Surface) -> None:
        w, h = hud.get_size()
        margin = 32
        corner_len = 28
        col = (220, 225, 235, 180)
        thick = 2
        # Four viewfinder corners
        pygame.draw.line(hud, col, (margin, margin), (margin + corner_len, margin), thick)
        pygame.draw.line(hud, col, (margin, margin), (margin, margin + corner_len), thick)
        pygame.draw.line(hud, col, (w - margin, margin), (w - margin - corner_len, margin), thick)
        pygame.draw.line(hud, col, (w - margin, margin), (w - margin, margin + corner_len), thick)
        pygame.draw.line(hud, col, (margin, h - margin), (margin + corner_len, h - margin), thick)
        pygame.draw.line(hud, col, (margin, h - margin), (margin, h - margin - corner_len), thick)
        pygame.draw.line(hud, col, (w - margin, h - margin), (w - margin - corner_len, h - margin), thick)
        pygame.draw.line(hud, col, (w - margin, h - margin), (w - margin, h - margin - corner_len), thick)

        # Center subtle crosshair / focus dot
        cx, cy = w // 2, h // 2
        pygame.draw.circle(hud, (255, 255, 255, 120), (cx, cy), 3, 1)

        # Bottom info bar
        dof = float(getattr(self, "photo_dof", 0.0))
        focus = float(getattr(self, "photo_focus", 1.8))
        fov = float(getattr(self, "photo_fov", 70.0))
        scale = int(self.cfg.get("graphics.photo_scale", 2))

        text = f"PHOTO MODE  |  FOV: {fov:.0f}° [ ]  |  Focus: {focus:.1f}m (K/L, F auto)  |  DOF: {dof:.2f} (, .)  |  Scale: {scale}x  |  F12/Click: Snap"
        txt_surf = self.f_small.render(text, True, (240, 243, 248))
        chip_w = txt_surf.get_width() + 24
        bg_rect = pygame.Rect((w - chip_w) // 2, h - margin - 26, chip_w, 24)
        pygame.draw.rect(hud, (12, 15, 22, 200), bg_rect, border_radius=6)
        pygame.draw.rect(hud, (70, 80, 100, 180), bg_rect, 1, border_radius=6)
        hud.blit(txt_surf, txt_surf.get_rect(center=bg_rect.center))

        # Top-left mode badge with RULE tag
        tag_bg = pygame.Rect(margin + 8, margin + 8, 140, 24)
        pygame.draw.rect(hud, (15, 18, 26, 190), tag_bg, border_radius=4)
        title_surf = self.f_small.render("FREE CAMERA", True, (240, 240, 240))
        hud.blit(title_surf, (margin + 14, margin + 12))
        if self.cfg.tags_on():
            k2.draw_source_chip(hud, (margin + 14 + title_surf.get_width() + 8, margin + 12), "rule", self.f_small)

        # Saved toast
        if self.saved_msg:
            msg, t0 = self.saved_msg
            age = time.perf_counter() - t0
            if age < 3.0:
                alpha = int(255 * min(1.0, (3.0 - age) * 2))
                s_surf = self.f_bold.render(msg, True, (120, 255, 160))
                s_surf.set_alpha(alpha)
                s_box = s_surf.get_rect(center=(cx, margin + 20)).inflate(20, 8)
                s_bg = pygame.Surface(s_box.size, pygame.SRCALPHA)
                s_bg.fill((10, 14, 20, int(200 * (alpha / 255))))
                hud.blit(s_bg, s_box.topleft)
                hud.blit(s_surf, s_box.center)


    # --- lifecycle -------------------------------------------------------------------------------------------------
    def _spawn_point(self) -> tuple[np.ndarray, float]:
        """A point in front of the player (or a random spot if the player isn't set up yet), and a facing yaw."""
        pl = getattr(self, "player", None)
        if pl is None:
            return np.zeros(2), math.pi / 2
        start = pl.pos + np.array([math.cos(pl.yaw), math.sin(pl.yaw)]) * random.uniform(2.0, 3.5)
        start += np.array([random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)])
        start = np.clip(start, (-RX + 0.8, -RZ + 0.8), (RX - 0.8, RZ - 0.8))
        return start, pl.yaw + math.pi / 2

    def _new_primary_fly(self) -> "Fly3D":
        start, yaw = self._spawn_point()
        return Fly3D(tuple(start), yaw=yaw)

    def _new_spawn_fly(self) -> "Fly3D":
        start, yaw = self._spawn_point()
        return Fly3D(tuple(start), yaw=yaw)

    def new_fly(self) -> None:
        super().new_fly()
        self.sugars3: list = []
        self.alcohols3: list = []
        self.decoys3: list = []
        self.grabbed_decoy3 = None

    def clear_transients(self) -> None:
        super().clear_transients()
        self.bombs3: list = []
        self.parts: list = []                     # particles: dict(p, v, t, life, kind, size, color)
        self.bolts3: list = []
        self.shards3: list = []
        self.popups3 = []
        self.spider3: dict | None = None
        self.pellets3: list = []
        self.fly_reward_until = self.fly_punish_until = self.fire_ready = 0.0
        self.steer = self.valence = self.trigger = self.hurt_flash = 0.0
        self.spider = None

    # --- helpers ---------------------------------------------------------------------------------------------------
    def popup(self, pos, text: str, color=(255, 245, 235), force=False) -> None:
        now = self.clock.now
        if self.popups3 and now - self.popups3[-1][2] < 0.3 and not force:
            return
        p = np.asarray(pos, float)
        if p.shape != (3,):
            return
        self.popups3.append([p.copy(), text, now, color, k2.POPUP_SOURCE.get(text)])

    def puff(self, pos, n: int, spread: float = 3.0) -> None:
        pos = np.asarray(pos, float)
        if pos.shape != (3,):
            return
        for _ in range(n):
            v = np.array([random.uniform(-1, 1), random.uniform(0, 0.6), random.uniform(-1, 1)]) * spread * S
            self.parts.append(dict(p=pos.copy(), v=v, t=self.clock.now, life=random.uniform(0.35, 0.8),
                                   kind="dust", size=random.uniform(0.03, 0.06)))

    def _you_pos(self) -> np.ndarray:
        return self.player.eye

    def aim(self):
        f, _, _ = self.player.basis()
        return self.player.eye, f

    def tool_tip(self) -> np.ndarray:
        """Where the tool in your hand fires from, in the world: the nozzle of the model draw_viewmodel draws.

        The held tool is drawn in camera space (x right, y up, -z forward) through its own 60 degree lens, while the
        world uses your field-of-view setting, so the point is scaled sideways by the ratio of the two lenses to land
        on the same spot on screen. (Before 2.7.2 these points had +z, i.e. behind your head, so flames and sprays
        started behind you and flew through the camera instead of out of the can.)"""
        name = TOOLS[self.tool][0]
        bob = 0.012 * math.sin(self.player.walk_phase * 2)
        x, y, z = VIEW_BASE + (0.0, bob, 0.0) + NOZZLE.get(name, (0.0, 0.05, -0.1))
        k = math.tan(math.radians(float(self.cfg["controls.fov"])) / 2) / math.tan(math.radians(VIEWMODEL_FOV) / 2)
        return self.player.to_world((x * k, y * k, z))

    def _overlay_open(self) -> bool:
        ch = getattr(self, "challenge", None)
        return (self.report is not None or self.big_view or self.surgery_open or self.help_open or self.training_open
                or (ch is not None and ch.overlay))

    def sneak_distance(self, slot) -> float:
        """Your body or the tool in your hand, whichever is closer to the fly's head, in fly lengths (0.55 m)."""
        head = slot.fly.p[HEAD]
        d = float(np.linalg.norm((self.player.eye - head)[[0, 2]]))          # your body: distance along the floor
        if TOOLS[self.tool][0] not in ("hand", "sugar", "alcohol"):
            d = min(d, float(np.linalg.norm(self.tool_tip() - head)))
        return d / 0.55

    def _above_head(self):
        return self.fly.p[HEAD] + (0, 0.45, 0)

    def _nearest_fly3d(self, eye, d, reach: float, max_perp: float):
        """The living fly (and its nearest particle) a ray hits first, across every spawned fly."""
        best_slot, best_i, best_t = None, None, None
        for slot in self.flies:
            i, t, _ = slot.fly.nearest_to_ray(eye, d, reach, max_perp)
            if i is not None and (best_t is None or t < best_t):
                best_slot, best_i, best_t = slot, i, t
        return best_slot, best_i, best_t

    def _nearest_decoy3d(self, eye, d, reach: float, max_perp: float):
        if not hasattr(self, "decoys3") or not self.decoys3:
            return None, None
        best_dec, best_t = None, None
        for dec in self.decoys3:
            rel = dec["p"] - eye
            t = float(rel @ d)
            if 0 < t < reach:
                perp = float(np.linalg.norm(rel - t * d))
                if perp < max_perp and (best_t is None or t < best_t):
                    best_dec, best_t = dec, t
        return best_dec, best_t

    def remove_decoy(self, dec) -> None:
        if hasattr(self, "decoys3") and dec in self.decoys3:
            drop_item(self.decoys3, dec)
            if getattr(self, "grabbed_decoy3", None) is dec:
                self.grabbed_decoy3 = None
            self.popup(dec["p"], "DECOY REMOVED", (200, 180, 220))
            self.sound.play("pop")

    def _flies_within3d(self, pos, radius: float) -> list:
        """Every fly with at least one particle within radius of a point (for area-effect tools)."""
        return [slot for slot in self.flies if np.any(np.linalg.norm(slot.fly.p - pos, axis=1) < radius)]

    # --- tools --------------------------------------------------------------------------------------------------------
    def use_tool3d(self, now: float) -> None:
        if self.cfg["brain.autopilot"]:
            return
        name = TOOLS[self.tool][0]
        eye, d = self.aim()
        self.record_event("tool", name, f"eye=({eye[0]:.2f},{eye[1]:.2f},{eye[2]:.2f})")
        if name in ("hand", "flick", "swatter", "zapper"):
            slot, i, _ = self._nearest_fly3d(eye, d, REACH, 0.35)
            if slot is not None and slot.fly.frozen_at is not None and slot.fly.shattered_at is None:
                self._shatter(slot, now)
                return
        if name == "hand":
            slot, i, t = self._nearest_fly3d(eye, d, GRAB_REACH, 0.18)
            dec, dec_t = self._nearest_decoy3d(eye, d, GRAB_REACH, 0.35)
            if dec is not None and (t is None or dec_t < t):
                self.grabbed_decoy3 = dec
                self.hold_dist = float(np.clip(dec_t, 0.6, 2.2))
                return
            if slot is not None and i is not None and slot.fly.frozen_at is None:
                slot.fly.grabbed, self.hold_dist = i, float(np.clip(t, 0.6, 2.2))
                slot.fly.last_hit = eye.copy()
                self.hit(slot, i, 0.25)
                self.focus = self.flies.index(slot)
            elif dec is not None:
                self.grabbed_decoy3 = dec
                self.hold_dist = float(np.clip(dec_t, 0.6, 2.2))
                return
        elif name == "flick":
            self.flick_t = now
            slot, i, t = self._nearest_fly3d(eye, d, 2.6, 0.4)
            if slot is not None:
                fly = slot.fly
                center = eye + d * t
                fly.last_hit = eye.copy()
                for j in range(N_P):
                    dist = float(np.linalg.norm(fly.p[j] - center))
                    if dist < 0.45:
                        s = 1 - dist / 0.45
                        push = d * 0.7 + np.array([0, 0.6, 0])
                        fly.impulse(j, push * (10 + 16 * s) * S)
                        self.hit(slot, j, 0.35 + 0.4 * s)
                fly.stun(now, 0.35)
                self.damage(slot, 4, "a flick")
                self.popup(center + (0, 0.25, 0), "FLICK!")
                self.sound.play("flick")
                self.focus = self.flies.index(slot)
        elif name == "swatter":
            if now - self.swing_t > 0.35:
                self.swing_t = now
                self.swats.append([None, now, False])
        elif name == "bomb" and len(self.bombs3) < 3 and now - self.throw_t > 0.4:
            self.throw_t = now
            self.bombs3.append(dict(p=self.tool_tip(), v=d * 0.1 + np.array([0, 0.025, 0]), t=now))
        elif name in ("torch", "cleaner", "freeze"):
            self.torching = True
        elif name == "laser":
            if hasattr(self, "laser_state"):
                self.laser_state.trigger_press(now)
            self.torching = True
        elif name == "zapper" and now >= self.zap_ready:
            self._zap3d(now)
        elif name == "spider" and self.spider3 is None and any(not s.fly.dead for s in self.flies):
            if d[1] < -0.05:
                pt = eye + d * (-eye[1] / d[1])
            else:
                pt = eye + np.array([d[0], 0, d[2]]) * 2.0
            pt = np.clip(pt, (-RX + 0.4, 0, -RZ + 0.4), (RX - 0.4, 0, RZ - 0.4))
            self.spider3 = dict(p=np.array([pt[0], min(RY, SPIDER_TOP) - 0.05, pt[2]]), state="drop", bite_at=0.0,
                                bites=0, anchor=pt.copy())
            self.spider = self.spider3
            self.sound.play("drop")
        elif name == "sugar" and len(self.sugars3) < 3 and now - self.throw_t > 0.3:
            self.throw_t = now
            self.sugars3.append(dict(p=self.tool_tip(), v=d * 0.07 + np.array([0, 0.02, 0]), left=1.0, landed=False))
            self.sound.play("pop")
        elif name == "alcohol" and len(self.alcohols3) < 3 and now - self.throw_t > 0.3:
            self.throw_t = now
            self.alcohols3.append(dict(p=self.tool_tip(), v=d * 0.07 + np.array([0, 0.02, 0]), left=1.0, landed=False))
            self.sound.play("drop")
        elif name == "cva":
            self.sound.play("pop")
            self.popup(self.tool_tip(), "cVA PUFF", (255, 180, 100))
            if not hasattr(self, "cva_puffs3"):
                self.cva_puffs3 = []
            self.cva_puffs3.append(dict(p=self.tool_tip().copy(), t=now, duration=0.65, reach=2.5))
            for slot in self.flies:
                dist = float(np.linalg.norm(slot.fly.p[HEAD] - self.tool_tip()))
                if dist < 2.5:
                    strength = 0.8 * (1.0 - dist / 2.5)
                    slot.brain.poke("scent", "cva", strength)
        elif name == "decoy" and now - self.throw_t > 0.3:
            self.throw_t = now
            if not hasattr(self, "decoys3"):
                self.decoys3 = []
            if len(self.decoys3) < 3:
                self.sound.play("drop")
                self.popup(self.tool_tip(), "DECOY FEMALE", (200, 180, 220))
                self.decoys3.append(dict(p=self.tool_tip().copy(), v=d * 0.07 + np.array([0, 0.02, 0]),
                                         landed=False, yaw=self.player.yaw + math.pi))
            else:
                self.popup(self.tool_tip(), "MAX 3 DECOYS", (200, 180, 220), force=True)
                self.note("DECOY    limit reached (max 3)")

    def _swat3d(self, now: float) -> None:
        eye, d = self.aim()
        center = eye + d * 1.05
        self.shake_until = now + 0.15
        self.sound.play("whack")
        dh = np.array([d[0], 0, d[2]])
        dh /= max(float(np.linalg.norm(dh)), 1e-6)
        for slot in self._flies_within3d(center, 0.8):
            fly = slot.fly
            near = [i for i in range(N_P) if np.linalg.norm(fly.p[i] - center) < 0.8]
            if not near:
                continue
            if fly.frozen_at is not None and fly.shattered_at is None:
                self._shatter(slot, now)
                continue
            fly.last_hit = eye.copy()
            for i in near:
                push = (fly.p[i] - center) * 0.15 + dh * 8 * S + np.array([0, -32 * S, 0])
                fly.impulse(i, push)
                self.hit(slot, i, 1.0)
            fly.stun(now, 1.8)
            self.damage(slot, 14, "the swatter")
            self.popup(fly.p[THX] + (0, 0.45, 0), "SWAT!", (255, 230, 120))
            self.puff(np.array([fly.p[THX, 0], 0.02, fly.p[THX, 2]]), 10, 4)

    def _explode3d(self, b: dict, now: float) -> None:
        pos = b["p"]
        self.shake_until = now + 0.4
        self.sound.play("boom")
        self.puff(pos, 24, 7)
        for _ in range(40):
            v = np.random.normal(0, 1, 3)
            v = v / np.linalg.norm(v) * random.uniform(0.02, 0.07)
            self.parts.append(dict(p=pos.copy(), v=v, t=now, life=random.uniform(0.25, 0.5), kind="fire",
                                   size=random.uniform(0.12, 0.3)))
        R = 330 * S
        for slot in self._flies_within3d(pos, R):
            fly = slot.fly
            d = np.linalg.norm(fly.p - pos, axis=1)
            worst = 0.0
            for i in np.flatnonzero(d < R):
                f = 48 * (1 - d[i] / R) ** 1.3
                dirv = (fly.p[i] - pos) / max(d[i], 1e-6) + np.array([0, 0.8, 0])
                fly.impulse(i, dirv * f * S)
                if f > 4:
                    self.hit(slot, i, f / 40)
                    worst = max(worst, f)
            if fly.frozen_at is not None and fly.shattered_at is None and worst:
                self._shatter(slot, now)
            elif worst:
                fly.last_hit = pos.copy()
                fly.stun(now, 2.4)
                self.damage(slot, 32 * worst / 48, "a bomb")
        self.popup(pos + (0, 0.5, 0), "KABOOM!", (255, 160, 60), force=True)

    def _zap3d(self, now: float) -> None:
        self.zap_ready = now + 0.3
        self.sound.play("zap")
        eye, d = self.aim()
        tip = self.tool_tip()
        best_slot, best_dist = None, None
        for slot in self.flies:
            fly = slot.fly
            to = fly.p[THX] - eye
            dist = float(np.linalg.norm(to))
            aligned = float(to @ d) / max(dist, 1e-6) >= 0.93
            i, _, _ = fly.nearest_to_ray(eye, d, REACH, 0.5)
            hit_ok = i is not None or (dist <= REACH and aligned)
            if hit_ok and fly.dissolved_at is None and fly.shattered_at is None and (best_dist is None or dist < best_dist):
                best_slot, best_dist = slot, dist
        if best_slot is None:
            self.bolts3.append([tip, eye + d * 2.0, now])
            return
        slot, fly = best_slot, best_slot.fly
        self.bolts3.append([tip, fly.p[THX].copy(), now])
        fly.zap_until = now + 0.3
        fly.char = min(1.0, fly.char + 0.06)
        fly.last_hit = eye.copy()
        for j in range(N_P):
            fly.impulse(j, np.random.normal(0, 3 * S, 3))
        self.shake_until = now + 0.12
        if fly.dead:
            return
        for region, side in TORCH_KEYS[:-1]:
            slot.brain.poke(region, side, 1.0)
        slot.brain.poke("all", None, 0.0)
        fly.stun(now, 1.0)
        self.damage(slot, 16, "the zapper")
        self.popup(fly.p[THX] + (0, 0.45, 0), random.choice(("BZZZT!", "ZAP!", "KRZZT!")), (200, 235, 255))

    def _jet(self, now: float, kind: str) -> None:
        """Blowtorch flame or a spray can: a cone from the tool tip along the view, hitting every fly it crosses."""
        eye, d = self.aim()
        tip = self.tool_tip()
        n, spread, speed = (7, 0.12, (0.05, 0.08)) if kind == "torch" else (5, 0.18, (0.035, 0.055))
        for _ in range(n):
            jitter = np.random.normal(0, spread, 3)
            v = (d + jitter) / np.linalg.norm(d + jitter) * random.uniform(*speed)
            self.parts.append(dict(p=tip.copy(), v=v, t=now, life=random.uniform(0.22, 0.4) if kind == "torch" else random.uniform(0.4, 0.65),
                                   kind="flame" if kind == "torch" else kind, size=0.05))
        rng, cone = (1.9, 0.85) if kind == "torch" else (2.0, 0.8)
        for slot in self.flies:
            fly = slot.fly
            rel = fly.p - tip
            dist = np.linalg.norm(rel, axis=1)
            cosang = (rel @ d) / np.maximum(dist, 1e-6)
            inside = np.flatnonzero((dist < rng) & (cosang > cone))
            if kind == "torch":
                if len(inside):
                    fly.burn_until = now + 0.9
                    for i in inside:
                        fly.impulse(i, d * 0.5 * S + np.array([0, 0.2 * S, 0]))
                if now >= fly.burn_until:
                    continue
                fly.char = min(1.0, fly.char + 0.004)
                fly.hurt = max(fly.hurt, 0.5)
                fly.last_hit = eye.copy()
                if fly.dead:
                    continue
                for region, side in TORCH_KEYS:
                    slot.brain.poke(region, side, 1.0)
                self.damage(slot, 0.35, "the blowtorch")
                words, col = ("SIZZLE!", "TSSSS!", "HOT HOT!"), (255, 150, 60)
            else:
                if fly.dissolved_at is not None or fly.frozen_at is not None or not len(inside):
                    continue
                fly.last_hit = eye.copy()
                if kind == "cleaner":
                    fly.soak = min(1.0, fly.soak + 0.04)
                else:
                    fly.frost = min(0.9 if self.immortal else 1.0, fly.frost + 0.006)
                if fly.dead:
                    continue
                if kind == "cleaner":
                    slot.brain.poke("smell", None, 1.0)
                    slot.brain.poke("taste", None, 0.8)
                    self.damage(slot, 0.12, "brake cleaner")
                    words, col = ("FSSSSH!", "MELTING!", "IT BURNS!"), (170, 230, 255)
                else:
                    slot.brain.poke("cold", None, 1.0)
                    for region, side in TORCH_KEYS[:-1]:
                        slot.brain.poke(region, side, 0.35)
                    self.damage(slot, 0.1, "freezing")
                    words, col = ("SO COLD!", "BRRRR!", "ICING!"), (190, 230, 255)
            if int(now * 2) != int((now - 1 / 60) * 2):
                slot.hits += 1
            if random.random() < 0.02:
                self.popup(fly.p[HEAD] + (0, 0.4, 0), random.choice(words), col)

    def _shatter(self, slot: "k2.FlySlot", now: float) -> None:
        fly = slot.fly
        fly.shattered_at = now
        self.shake_until = now + 0.25
        self.sound.play("shatter")
        for i in range(N_P):
            for _ in range(3):
                self.shards3.append(dict(p=fly.p[i].copy(), v=np.random.normal(0, 0.02, 3) + (0, 0.03, 0),
                                         rot=np.random.uniform(0, 6.28, 3), size=random.uniform(0.02, 0.05),
                                         fly=random.random() < 0.35))
        self.popup(fly.p[THX] + (0, 0.5, 0), "SHATTERED!", (200, 235, 255), force=True)

    # --- the fly's senses ------------------------------------------------------------------------------------------------
    def _threats(self, slot: "k2.FlySlot", now: float, mouse=None) -> list:
        out = []
        if not self.cfg["brain.autopilot"] and not self._overlay_open():
            eye = self.player.eye
            out.append(("player", eye - (0, 0.35, 0), 0.28))
            name = TOOLS[self.tool][0]
            if name not in ("hand", "sugar", "alcohol"):
                out.append(("tool", self.tool_tip(), TOOL_SIZE.get(name, 0.08)))
            ph = now - self.swing_t
            if ph < 0.14:
                _, d = self.aim()
                a = self.player.to_world((0.28, 0.35, 0.35))
                b = self.player.eye + d * 1.05
                out.append(("swing", a + (b - a) * (ph / 0.14), 0.2))
        if self.spider3 is not None and self.spider3["state"] in ("drop", "hunt"):
            out.append(("spider", self.spider3["p"].copy(), 0.1))
        for b in self.bombs3:
            out.append((("bomb", id(b)), b["p"].copy(), 0.08))
        for other in self.flies:                      # other flies loom too: a real, symmetric dodge reaction
            if other is slot or other.fly.dead:
                continue
            out.append((("fly", id(other.fly)), other.fly.p[THX].copy(), RAD3[THX] * 2))
        return out

    def _vision(self, slot: "k2.FlySlot", now: float, mouse=None) -> None:
        fly = slot.fly
        if fly.dead or fly.frozen_at is not None:
            slot.loom = 0.0
            return
        head = fly.p[HEAD]
        best, best_pos, seen = 0.0, None, {}
        for key, pos, r in self._threats(slot, now):
            dist = max(float(np.linalg.norm(pos - head)), r + 0.02)
            theta = 2 * math.atan(r / dist)
            prev = slot.loom_prev.get(key)
            seen[key] = theta
            if prev is not None and (theta - prev) * 60.0 > best:
                best, best_pos = (theta - prev) * 60.0, pos
        slot.loom_prev = seen
        slot.loom += (best - slot.loom) * 0.5
        strength = float(np.clip((best - k2.LOOM_MIN) / k2.LOOM_FULL, 0, 1))
        if strength > 0 and best_pos is not None:
            slot.brain.poke("loom", None, strength, recruit=0.6 * strength)
            slot.threat_x = np.asarray(best_pos, float).copy()

    def _scents(self, slot: "k2.FlySlot", now: float, mouse=None) -> None:
        fly = slot.fly
        slot.scent_now, slot.sugar_scent = None, False
        if fly.dead:
            return
        head = fly.p[HEAD]
        if not self._overlay_open() and np.linalg.norm(self.tool_tip() - head) < 2.5:
            slot.scent_now = TOOLS[self.tool][0]
            slot.brain.poke("scent", slot.scent_now, 0.3)
        if any(np.linalg.norm((s["p"] - head)[[0, 2]]) < 2.4 for s in self.sugars3):
            slot.sugar_scent = True
            slot.brain.poke("scent", "sugar", 0.3)
        if any(np.linalg.norm((a["p"] - head)[[0, 2]]) < 2.4 for a in self.alcohols3):
            slot.brain.poke("scent", "alcohol", 0.35)     # fermented fruit odor on DM1/DM2/DP1m (real ORNs)
        slot.player_scent = bool(self.duel and slot is self.flies[self.focus] and self.player_dead_at is None
                                 and np.linalg.norm((self.player.eye - head)[[0, 2]]) < 3.5)
        if slot.player_scent:
            slot.brain.poke("scent", "player", 0.35)               # you smell like you

    def _memory_behavior(self, slot: "k2.FlySlot", now: float, free: bool, can_fly: bool) -> None:
        fly = slot.fly
        if self.duel and getattr(slot, "player_scent", False) and free and now >= slot.avoid_ready and now >= fly.stun_until:
            fear, like = self._valence()
            if fear - like > k2.FEAR_ACT:                           # its mushroom body learned to fear you
                slot.avoid_ready = now + 2.5
                fly.last_hit = self.player.eye.copy()
                if can_fly and fear - like > 0.6:
                    fly.escape(now)
                else:
                    fly.yaw_target = angle_to(fly.away_from(self.player.eye))
                    fly.walk_until, fly.back_until, fly.run = now + 1.2, 0.0, True
                self.note(f"FLEE     fears you ({fear:.2f})")
                self.popup(fly.p[HEAD] + (0, 0.4, 0), "RUN AWAY!", (140, 200, 255))
                return
        if not slot.scent_now or not free or now < slot.avoid_ready or fly.frozen_at is not None or now < fly.stun_until:
            return
        eye = self.player.eye
        if slot.scent_now != "sugar" and slot.fear_now > k2.FEAR_ACT:
            slot.avoid_ready = now + 2.5
            fly.last_hit = eye.copy()
            if can_fly and slot.fear_now > 0.6:
                fly.escape(now)
            else:
                fly.yaw_target = angle_to(fly.away_from(eye))
                fly.walk_until, fly.back_until, fly.run = now + 1.2, 0.0, True
            self.note(f"AVOID    remembers the {slot.scent_now} ({slot.fear_now:.2f})")
            self.on_reaction("AVOID", slot)
            self.popup(fly.p[HEAD] + (0, 0.4, 0), "NOPE!", (255, 220, 120))
        elif slot.scent_now == "sugar" and slot.like_now > k2.LIKE_ACT and now >= fly.walk_until:
            slot.avoid_ready = now + 1.5
            fly.yaw_target = angle_to(-fly.away_from(eye))
            fly.walk_until, fly.run = now + 1.0, False
            self.note(f"APPROACH remembers sugar ({slot.like_now:.2f})")

    # --- outdoor worlds: Open field and Orchard (kickthefly/game/outdoors.py) --------------------------------------------------
    def on_arena_changed(self, now: float) -> None:
        if getattr(self, "_room", None) is None:     # still inside __init__; runs again once the room exists
            return
        super().on_arena_changed(now)
        self.decoys3 = []
        self.grabbed_decoy3 = None
        arena = k2.ARENAS[self.arena_i]
        was = getattr(self, "world", "room")
        self.scenery = outdoors.scenery(arena)
        set_world(arena, self.scenery["trees"])
        self.world = arena
        self.orchard = None
        if arena == "orchard":
            p = self.lab_params
            self.orchard = outdoors.Orchard(self.scenery["trees"], feeds=int(p.get("orchard.feeds", outdoors.DEFAULT_FEEDS)),
                                            regrow_s=float(p.get("orchard.regrow_s", outdoors.DEFAULT_REGROW_S)),
                                            cap=int(p.get("orchard.cap", outdoors.DEFAULT_CAP)),
                                            seed=int(self.cfg["brain.seed"]))
            for f in self.orchard.fruit:             # the game clock doesn't start at 0: staggered regrowth from now
                if f.regrow_at is not None:
                    f.regrow_at += now
        for slot in self.flies:
            slot.lost, slot.fruit = False, None
            slot.fly.perch = None
        if (was in outdoors.OUTDOOR) != (arena in outdoors.OUTDOOR) or arena in outdoors.OUTDOOR:
            self.player.pos = np.array([0.0, 2.6]) if arena not in outdoors.OUTDOOR else np.array([0.0, 3.0])
            self.player.vel = np.zeros(2)
            self.player.yaw = -math.pi / 2
            for slot in self.flies:
                self._move_fly(slot.fly, self._spawn_point()[0])
            self.sugars3 = [sg for sg in self.sugars3 if abs(sg["p"][0]) < RX and abs(sg["p"][2]) < RZ]
            self.alcohols3 = [al for al in getattr(self, "alcohols3", []) if abs(al["p"][0]) < RX and abs(al["p"][2]) < RZ]
            self.spider3 = self.spider = None

    def arena_status(self) -> list[str]:
        arena = k2.ARENAS[self.arena_i]
        if arena not in outdoors.OUTDOOR:
            return []
        p = self.lab_params
        out = []
        if arena == "field":
            out.append(f"wind {p.get('field.wind_speed', 3.0):.0f} m/s from {p.get('field.wind_dir', 180.0):.0f}°")
        if self.orchard is not None:
            c = self.orchard.counts()
            out.append(f"fruit {c['ripe']} ({c['fermented']} fermented)")
        lost = sum(1 for slot in self.flies if getattr(slot, "lost", False))
        if lost:
            out.append(f"{lost} LOST ({self.cfg.keys.get('recall', 'j').upper()} recall)")
        return out

    def _move_fly(self, fly: "Fly3D", xz) -> None:
        """Put a fly's whole body down at a new spot on the ground, standing, not flying. A GAME RULE move."""
        off = np.array([xz[0] - fly.p[THX, 0], STAND3 - fly.p[THX, 1], xz[1] - fly.p[THX, 2]])
        fly.p += off
        fly.prev = fly.p.copy()
        fly.anchor = np.array(xz, float)
        fly.hover = fly.p[THX].copy()
        fly.fly_target = fly.hover.copy()
        fly.escape_until = 0.0
        fly.stuck.clear()
        fly.perch = None

    def recall_flies(self, now: float) -> None:
        """J: bring every fly that flew out of sight back in front of you. GAME RULE, like respawning the player."""
        lost = [slot for slot in self.flies if getattr(slot, "lost", False)]
        if not lost:
            self.note("RECALL   no fly is lost", source="rule")
            return
        for slot in lost:
            self._move_fly(slot.fly, self._spawn_point()[0])
            slot.lost = False
            if getattr(slot, "fruit", None) is not None:
                self._leave_fruit(slot, now)
        self.note(f"RECALL   {len(lost)} lost fl{'y' if len(lost) == 1 else 'ies'} called back", source="rule")

    def _check_lost(self, now: float) -> None:
        w = outdoors.spec(self.world)
        if w.lost_radius is None:
            return
        me = self.player.pos
        for slot in self.flies:
            th = slot.fly.p[THX]
            d = math.hypot(th[0] - me[0], th[2] - me[1])
            far = d > w.lost_radius or th[1] > 15.0
            if far and not getattr(slot, "lost", False):
                slot.lost = True
                self.note(f"LOST     flew out of sight, {d:.0f} m away ({self.cfg.keys.get('recall', 'j').upper()} "
                          f"calls it back)")
            elif not far and getattr(slot, "lost", False):
                slot.lost = False

    def _weather(self, slot, arena: str, now: float) -> None:
        """Outdoor wind and sunlight onto the real wind and light neurons. The transduction is a GAME RULE."""
        fly, br = slot.fly, slot.brain
        p = self.lab_params
        if arena == "field":
            wd, ws = float(p.get("field.wind_dir", 180.0)), float(p.get("field.wind_speed", 3.0))
            fly.wind = outdoors.wind_vector(wd, ws) * 0.0006           # how hard the air pushes the body (game rule)
            if not fly.dead and self.frame % 3 == 0 and ws > 0:
                left, right = outdoors.wind_drive(fly.yaw, wd, ws)
                if left > 0.02:
                    br.poke("wind", "L", left)
                if right > 0.02:
                    br.poke("wind", "R", right)
            if random.random() < 0.25 * min(1.0, ws / 4):
                me = self.player.eye
                src = me + np.array([random.uniform(-6, 6), random.uniform(0.2, 2.5), random.uniform(-6, 6)])
                self.parts.append(dict(p=src, v=outdoors.wind_vector(wd, ws) / 60, t=now, life=1.2, kind="streak",
                                       size=0.02))
        if not fly.dead and self.frame % 2 == 0:
            day_s = self.cfg["brain.day_night"]
            az, el = outdoors.sun_now(p, now, day_s)
            left, right = outdoors.sun_light(fly.yaw, az, el)
            if left > 0.02:
                br.poke("light", "L", left, recruit=0.25 * left)
            if right > 0.02:
                br.poke("light", "R", right, recruit=0.25 * right)
            if float(p.get("outdoor.day_s", 0.0)) > 0 or day_s:   # day/night on: daylight reaches the LNvs (rule)
                clock = outdoors.circadian_clock_drive(el)["morning_cells"]
                if clock > 0.02:
                    br.poke("clock", None, clock)

    # the orchard: fruit, flying to it, feeding. All of it GAME RULE except the neurons feeding drives.
    def _perch_of(self, f) -> np.ndarray:
        return f.pos + np.array([0.0, 0.07, 0.0])

    def _fly_to(self, fly: "Fly3D", now: float, target: np.ndarray, seconds: float) -> None:
        """Take off toward a point without touching escape_ready, so the neurons can still trigger a real escape."""
        if now >= fly.escape_until:
            fly.hover = fly.p[THX].copy()
            fly.prev[:] = fly.p - np.array([0, 6.0 * S, 0])
        fly.escape_until = now + seconds
        fly.wander = False
        fly.fly_target = np.asarray(target, float).copy()

    def _leave_fruit(self, slot, now: float) -> None:
        f = getattr(slot, "fruit", None)
        if f is not None and f.eater is slot:
            f.eater = None
        slot.fruit, slot.landed = None, False
        slot.fly.perch = None
        slot.fruit_ready = now + random.uniform(8.0, 16.0)

    def _orchard_tick(self, now: float) -> None:
        o = self.orchard
        p = self.lab_params
        o.set_params(p.get("orchard.feeds"), p.get("orchard.regrow_s"), p.get("orchard.cap"))
        o.step(now)
        for slot in self.flies:
            fly, br = slot.fly, slot.brain
            f = getattr(slot, "fruit", None)
            busy = (fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None
                    or getattr(slot, "lost", False))
            if f is not None:
                perch = self._perch_of(f)
                diverted = not np.allclose(fly.fly_target, perch)            # the neurons made it escape: they win
                if busy or diverted or not f.ripe or now >= fly.escape_until \
                        or (getattr(slot, "landed", False) and f.eater is not slot):
                    self._leave_fruit(slot, now)
                    continue
                if not getattr(slot, "landed", False):
                    if float(np.linalg.norm(fly.hover - perch)) < 0.1:        # landed
                        if f.eater is not None:                            # another fly got there first
                            self._leave_fruit(slot, now)
                            continue
                        f.eater, slot.landed, slot.feed_start = slot, True, now
                        fly.escape_until = now + outdoors.FEED_BOUT_S + 1.0
                        self.note("EATING   fermented fruit: sweet + PAM reward [GAME RULE: inebriation]" if f.fermented
                                  else "EATING   fruit: taste + PAM reward", source="rule")
                        self.popup(fly.p[HEAD] + (0, 0.3, 0), random.choice(("YUM!", "NOM NOM")), (255, 160, 190))
                    elif now - slot.fruit_since > 14.0:
                        self._leave_fruit(slot, now)                         # took too long: give up
                    continue
                # feeding: the same real neurons the sugar tool (or, fermented, the alcohol tool) drives
                fly.eating_until = now + 0.4
                br.poke("taste", None, 0.5)
                br.poke("sweet", None, 0.6 if f.fermented else 0.5, recruit=0.6)
                br.poke("reward", None, 0.6 if f.fermented else 0.4)
                if f.fermented:
                    fly.inebriation = min(1.0, getattr(fly, "inebriation", 0.0) + 0.008)     # GAME RULE
                    if self.frame % 10 == 0:
                        br.poke("scent", "alcohol", 0.35)                    # DM1/DM2/DP1m fermentation glomeruli
                fly.health = min(MAX_HEALTH, fly.health + 0.15)
                if now - slot.feed_start >= outdoors.FEED_BOUT_S:
                    emptied = o.feed(f, now)
                    if emptied:
                        self.popup(f.pos + (0, 0.3, 0), "ALL GONE", (230, 200, 150))
                    self._leave_fruit(slot, now)
                    self._fly_to(fly, now, fly.p[THX] + np.array([random.uniform(-2, 2), 1.2, random.uniform(-2, 2)]), 1.5)
                continue
            if busy or now < getattr(slot, "fruit_ready", now + random.uniform(2.0, 6.0)) or now < fly.escape_until:
                slot.fruit_ready = getattr(slot, "fruit_ready", now + random.uniform(2.0, 6.0))
                continue
            can_fly = fly.frost < 0.5 and fly.melt < 0.3 and fly.venom < 0.5 and fly.wet <= 0 and len(fly.stuck) < 2
            target = o.nearest_ripe(fly.p[THX])
            if not can_fly or target is None or float(np.linalg.norm(target.pos - fly.p[THX])) > 14.0:
                slot.fruit_ready = now + 3.0
                continue
            slot.fruit, slot.fruit_since, slot.landed = target, now, False
            self._fly_to(fly, now, self._perch_of(target), 16.0)
            fly.perch = target                                               # hold there on arrival, don't wander
            self.note("TO FRUIT flies to a fruit", source="rule")

    # --- arenas and ongoing effects -------------------------------------------------------------------------------------------
    def _environment(self, now: float, mouse=None) -> None:
        arena = k2.ARENAS[self.arena_i]
        for slot in self.flies:
            self._environment_one(slot, arena, now)
        if arena in outdoors.OUTDOOR:
            self._check_lost(now)
            if self.orchard is not None:
                self._orchard_tick(now)

    def _environment_one(self, slot: "k2.FlySlot", arena: str, now: float) -> None:
        fly, br = slot.fly, slot.brain
        fly.arena, fly.wind = arena, np.zeros(3)
        if arena not in ("flypaper", "escaperoom"):
            fly.stuck.clear()
        if arena in outdoors.OUTDOOR:
            self._weather(slot, arena, now)
        elif arena == "fan":
            gust = 0.75 + 0.25 * math.sin(now * 1.3) + 0.15 * math.sin(now * 4.1)
            dx = fly.p[THX, 0] - FAN3[0]
            lateral = math.exp(-((fly.p[THX, 2] - FAN3[2]) ** 2) / (2 * 1.9 ** 2))
            w = 0.55 * gust * float(np.clip(1.15 - dx / 7.0, 0.25, 1.0)) * lateral
            fly.wind = np.array([w * S, 0, 0])
            if not fly.dead and self.frame % 3 == 0:
                br.poke("wind", None, min(1.0, w * 1.4))
            if random.random() < 0.8:
                self.parts.append(dict(p=np.array([FAN3[0] + 0.5, random.uniform(0.4, 1.6), FAN3[2] + random.uniform(-1.2, 1.2)]),
                                       v=np.array([random.uniform(0.09, 0.15), 0, 0]), t=now, life=1.0, kind="streak", size=0.025))
        elif arena == "flypaper" and fly.stuck:
            kicking = now < fly.flail_until
            for i in list(fly.stuck):
                pull = fly.grabbed is not None and np.linalg.norm(self._hold_point() - fly.stuck[i]) > 0.55
                if random.random() < 0.0012 + (0.012 if kicking else 0) + (0.06 if pull else 0):
                    del fly.stuck[i]
            if not fly.dead:
                if self.frame % 6 == 0:
                    br.poke("legs", "L", 0.4)
                    br.poke("legs", "R", 0.4)
                    br.poke("body", None, 0.2)
                if len(fly.stuck) >= 3:
                    self.damage(slot, 0.012, "the flypaper")
        elif arena == "pool":
            sub = fly.p[:, 1] < WATER3
            if sub.any():
                if fly.wet <= 0 and float(np.max((fly.prev - fly.p)[sub, 1])) > 4 * S:
                    self.sound.play("splash")
                    self.puff(np.array([fly.p[THX, 0], WATER3, fly.p[THX, 2]]), 10, 3)
                fly.wet = 3.0
                if not fly.dead:
                    if self.frame % 4 == 0:
                        br.poke("humid", None, 0.9)
                        br.poke("body", None, 0.25)
                    if fly.p[HEAD, 1] < WATER3 - 6 * S:
                        self.damage(slot, 0.05, "drowning")
        elif arena == "lamp":
            dist = float(np.linalg.norm(fly.p[HEAD] - LAMP3))
            if not fly.dead:
                light = float(np.clip(1.2 - dist / 3.0, 0.15, 1.0))
                if self.frame % 2 == 0:
                    br.poke("light", None, light, recruit=0.25 * light)
                if dist < 0.36:
                    br.poke("heat", None, 0.8)
                    self.damage(slot, 0.08, "the hot lamp")
                    away = (fly.p[HEAD] - LAMP3) / max(dist, 1e-6)
                    for i in (HEAD, THX, ABD):
                        fly.impulse(i, away * 1.5 * S)
            if now < fly.escape_until and np.linalg.norm(fly.fly_target - LAMP3) > 0.8:
                fly.fly_target = LAMP3 + np.array([random.uniform(-0.6, 0.6), -random.uniform(0.3, 0.8), random.uniform(-0.6, 0.6)])
        elif arena == "thermo":
            self._thermo_tick(slot, float(fly.p[THX, 0]) / THERMO_HALF3, fly.p[THX, 1] < STAND3 + 20 * S)
        elif arena == "escaperoom":
            # 1. Fan wind from left
            gust = 0.75 + 0.25 * math.sin(now * 1.3) + 0.15 * math.sin(now * 4.1)
            dx = fly.p[THX, 0] - FAN3[0]
            lateral = math.exp(-((fly.p[THX, 2] - FAN3[2]) ** 2) / (2 * 1.9 ** 2))
            w = 0.55 * gust * float(np.clip(1.15 - dx / 7.0, 0.25, 1.0)) * lateral
            fly.wind = np.array([w * S, 0, 0])
            if not fly.dead and self.frame % 3 == 0:
                br.poke("wind", None, min(1.0, w * 1.4))
            if random.random() < 0.8:
                self.parts.append(dict(p=np.array([FAN3[0] + 0.5, random.uniform(0.4, 1.6), FAN3[2] + random.uniform(-1.2, 1.2)]),
                                       v=np.array([random.uniform(0.09, 0.15), 0, 0]), t=now, life=1.0, kind="streak", size=0.025))

            # 2. Flypaper in middle
            if fly.stuck:
                kicking = now < fly.flail_until
                for i in list(fly.stuck):
                    pull = fly.grabbed is not None and np.linalg.norm(self._hold_point() - fly.stuck[i]) > 0.55
                    if random.random() < 0.0012 + (0.012 if kicking else 0) + (0.06 if pull else 0):
                        del fly.stuck[i]
                if not fly.dead:
                    if self.frame % 6 == 0:
                        br.poke("legs", "L", 0.4)
                        br.poke("legs", "R", 0.4)
                        br.poke("body", None, 0.2)
                    if len(fly.stuck) >= 3:
                        self.damage(slot, 0.012, "the flypaper")

            # 3. Hot Lamp overhead
            dist = float(np.linalg.norm(fly.p[HEAD] - LAMP3))
            if not fly.dead:
                light = float(np.clip(1.2 - dist / 3.0, 0.15, 1.0))
                if self.frame % 2 == 0:
                    br.poke("light", None, light, recruit=0.25 * light)
                if dist < 0.36:
                    br.poke("heat", None, 0.8)
                    self.damage(slot, 0.08, "the hot lamp")
                    away = (fly.p[HEAD] - LAMP3) / max(dist, 1e-6)
                    for i in (HEAD, THX, ABD):
                        fly.impulse(i, away * 1.5 * S)
            if now < fly.escape_until and np.linalg.norm(fly.fly_target - LAMP3) > 0.8:
                fly.fly_target = LAMP3 + np.array([random.uniform(-0.6, 0.6), -random.uniform(0.3, 0.8), random.uniform(-0.6, 0.6)])

            # 4. Sugar Goal on the right
            sugar_pos = np.array([RX - 0.7, 0.04, 0.0])
            d_sugar = float(np.linalg.norm(fly.p[HEAD] - sugar_pos))
            if d_sugar < 0.4 and not fly.dead:
                if self.frame % 3 == 0:
                    br.poke("sweet", None, 1.0)
                if not self.escaperoom_completed:
                    self.escaperoom_completed = True
                    self.escaperoom_finish_t = now
                    run_time = round(max(0.1, now - self.escaperoom_start_t), 2)
                    from kickthefly.game.speedrun import make_speedrun_code
                    self.escaperoom_code = make_speedrun_code(self.escaperoom_seed, run_time)
                    from kickthefly.lab.challenges import record_score
                    record_score("escaperoom_speedrun", run_time, "low")
                    self.sound.play("yum")
                    self.note(f"ESCAPEROOM CLEAR {run_time:.2f}s ({self.escaperoom_code})")
        if arena != "pool" or not (fly.p[:, 1] < WATER3).any():
            fly.wet = max(0.0, fly.wet - 1 / 60)

    def _kick(self, now: float) -> None:
        """Your legs are a 0.28 m cylinder: walking into a fly shoves it, and walking into it fast is a kick."""
        if self.cfg["brain.autopilot"]:
            return
        pl = self.player
        for slot in self.flies:
            fly = slot.fly
            if fly.frozen_at is not None or fly.dissolved_at is not None or fly.shattered_at is not None:
                continue
            rel = fly.p[:, [0, 2]] - pl.pos
            dist = np.linalg.norm(rel, axis=1)
            inside = np.flatnonzero((dist < 0.28 + RAD3) & (fly.p[:, 1] < 0.9))
            if not len(inside):
                continue
            speed = float(np.linalg.norm(pl.vel))
            for i in inside:
                n = rel[i] / max(dist[i], 1e-6)
                fly.p[i, [0, 2]] = pl.pos + n * (0.28 + RAD3[i])
            if speed > 1.2 and now - getattr(self, "kick_t", 0) > 0.4:
                self.kick_t = now
                push = np.append(pl.vel, 0)[[0, 2, 1]] / 60 * 1.6 + np.array([0, 0.012 + 0.006 * speed, 0])
                for i in range(N_P):
                    fly.impulse(i, push)
                for i in inside:
                    self.hit(slot, i, min(1.0, speed / 5))
                fly.last_hit = pl.eye.copy()
                if speed > 2.5:
                    fly.stun(now, 0.6)
                    self.damage(slot, 3 + 2 * speed, "a kick")
                    self.popup(fly.p[THX] + (0, 0.4, 0), random.choice(("KICK!", "BOOT!", "PUNT!")), (255, 235, 150))
                    self.sound.play("whack", 0.6)

    def _hold_point(self) -> np.ndarray:
        eye, d = self.aim()
        return eye + d * self.hold_dist

    def _effects(self, now: float) -> None:
        for slot in self.flies:
            self._effects_one(slot, now)
        self._spider3d(now)
        self._sugar3d(now)
        self._alcohol3d(now)
        self._decoy3d(now)

    def _effects_one(self, slot: "k2.FlySlot", now: float) -> None:
        fly = slot.fly
        if getattr(fly, "inebriation", 0.0) > 0:
            fly.inebriation = max(0.0, fly.inebriation - 0.00035)   # ethanol metabolism (~45s)
        if self.immortal:
            fly.melt = min(fly.melt, 0.85)
            if fly.soak < 0.05:
                fly.melt = max(0.0, fly.melt - 0.002)
            fly.venom = max(0.0, fly.venom - 0.002)
        if fly.dissolved_at is None and fly.soak > 0.02:
            fly.melt = min(0.85 if self.immortal else 1.0, fly.melt + 0.0045 * fly.soak)
            fly.soak *= 0.985 if self.immortal else 0.997
            if not fly.dead:
                self.damage(slot, 0.25 * fly.soak, "brake cleaner")
                for region, side in TORCH_KEYS[:-1]:
                    slot.brain.poke(region, side, 0.5 * fly.soak)
        if fly.melt >= 1.0 and fly.dissolved_at is None:
            fly.dissolved_at = now
            self.puff(np.array([fly.p[THX, 0], 0.05, fly.p[THX, 2]]), 14, 3)
            self.popup(fly.p[THX] + (0, 0.4, 0), "DISSOLVED", (170, 230, 255), force=True)
            self.sound.play("squish")
            if not fly.dead:
                self.damage(slot, MAX_HEALTH, "brake cleaner")
        if fly.frozen_at is None and fly.frost > 0:
            if not (self.torching and TOOLS[self.tool][0] == "freeze"):
                fly.frost = max(0.0, fly.frost - 0.0015)
            if fly.frost >= 1.0:
                fly.frozen_at = now
                self.popup(fly.p[THX] + (0, 0.45, 0), "FROZEN SOLID", (190, 230, 255), force=True)
                if not fly.dead:
                    self.damage(slot, MAX_HEALTH, "freezing")
        if not fly.dead:
            slot.brain.sedation = max(0.25 * fly.melt ** 2.5, 0.2 * fly.frost ** 2, 0.3 * fly.venom ** 1.5)

    def _spider3d(self, now: float) -> None:
        sp = self.spider3
        if sp is None:
            return
        if sp["state"] == "drop":
            sp["p"][1] -= 0.04
            if sp["p"][1] <= 0.12:
                sp["p"][1], sp["state"] = 0.12, "hunt"
            return
        candidates = [s for s in self.flies if not (s.fly.dead or s.fly.dissolved_at is not None or s.fly.shattered_at is not None)]
        if sp["state"] == "hunt":
            if not candidates:
                sp["state"] = "leave"
                return
            slot = min(candidates, key=lambda s: float(np.linalg.norm(s.fly.p[THX] - sp["p"])))
            fly = slot.fly
            d = fly.p[THX] + (0, 0.1, 0) - sp["p"]
            dist = float(np.linalg.norm(d))
            if dist > 0.2:
                sp["p"] += d / dist * min(3.4 * S, dist)
                sp["p"][1] = max(0.12, sp["p"][1])
                sp["anchor"] = sp["p"].copy()
            elif now - sp["bite_at"] > 0.7:
                sp["bite_at"] = now
                sp["bites"] += 1
                fly.venom = min(1.0, fly.venom + 0.3)
                fly.last_hit = sp["p"].copy()
                fly.hurt = 1.0
                slot.brain.poke("body", None, 0.9)
                slot.brain.poke("legs", "L", 0.6)
                slot.brain.poke("legs", "R", 0.6)
                self.damage(slot, 16, "a spider")
                self.sound.play("chomp")
                self.popup(sp["p"] + (0, 0.3, 0), random.choice(("CHOMP!", "BITE!", "SLURP!")), (230, 120, 120))
                if sp["bites"] >= 2 and not fly.wrapped:
                    fly.wrapped = True
                    sp["target"] = slot
                    self.note("WRAPPED  in spider silk")
                if self.immortal and sp["bites"] >= 5:
                    fly.wrapped, fly.grabbed = False, None
                    sp["state"] = "leave"
                    self.popup(fly.p[HEAD] + (0, 0.4, 0), "BROKE FREE!", (255, 225, 120), force=True)
                    self.note("BROKE FREE of the silk")
                    return
            if fly.wrapped:
                fly.grabbed = THX
        elif sp["state"] in ("leave", "carry"):
            wrapped = sp.get("target")
            if wrapped is not None and wrapped.fly.wrapped and wrapped.fly.dead and wrapped.fly.frozen_at is None:
                sp["state"] = "carry"
            sp["p"][1] += 2.2 * S * 2
            if sp["state"] == "carry" and wrapped is not None:
                wrapped.fly.grabbed = THX
            if sp["p"][1] > min(RY, SPIDER_TOP) + 0.4:
                self.spider3 = self.spider = None
                if wrapped is not None and wrapped.fly.wrapped:
                    wrapped.fly.grabbed = None

    def _sugar3d(self, now: float) -> None:
        for s in self.sugars3:
            if not s["landed"]:
                s["v"][1] -= GRAV
                s["p"] += s["v"]
                for ax, lim in ((0, RX - 0.05), (2, RZ - 0.05)):
                    if abs(s["p"][ax]) > lim:
                        s["p"][ax] = math.copysign(lim, s["p"][ax])
                        s["v"][ax] *= -0.4
                if s["p"][1] <= 0.035:
                    s["p"][1], s["landed"] = 0.035, True
        if not self.sugars3:
            return
        for slot in self.flies:
            fly = slot.fly
            if fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
                continue
            s = min(self.sugars3, key=lambda s: float(np.linalg.norm((s["p"] - fly.p[THX])[[0, 2]])))
            dxz = (s["p"] - fly.p[HEAD])[[0, 2]]
            on_floor = fly.p[THX, 1] < STAND3 + 0.2 and now >= fly.escape_until
            if float(np.linalg.norm(dxz)) > 0.2:
                if on_floor and now >= fly.stun_until and s["landed"]:
                    fly.yaw_target = math.atan2(dxz[1], dxz[0])
                    fly.walk_until, fly.run = now + 0.2, False
                continue
            if not on_floor or not s["landed"]:
                continue
            if now >= fly.eating_until:
                self.popup(fly.p[HEAD] + (0, 0.4, 0), random.choice(("YUM!", "SWEET!", "NOM NOM")), (255, 160, 190))
                self.sound.play("yum")
                self.note("EATING   sugar: taste + PAM reward")
            fly.eating_until = now + 0.4
            s["left"] -= 1 / 240
            slot.brain.poke("taste", None, 0.5)
            slot.brain.poke("sweet", None, 0.5, recruit=0.6)          # the sugar-pathway taste neurons (assays.py)
            slot.brain.poke("reward", None, 0.4)
            fly.health = min(MAX_HEALTH, fly.health + 0.15)
            if s["left"] <= 0:
                drop_item(self.sugars3, s)
                break

    def _alcohol3d(self, now: float) -> None:
        """Alcohol drop: each fly walks over and sips the nearest droplet (fermented fruit / ethanol).
        Connectome: the sweet taste pathway and the PAM dopaminergic reward neurons, as for sugar.
        Game rule: the escalating inebriation (tremors, wobbly drift, stumbling gait, slower escape reflexes)."""
        for a in self.alcohols3:
            if not a["landed"]:
                a["v"][1] -= GRAV
                a["p"] += a["v"]
                for ax, lim in ((0, RX - 0.05), (2, RZ - 0.05)):
                    if abs(a["p"][ax]) > lim:
                        a["p"][ax] = math.copysign(lim, a["p"][ax])
                        a["v"][ax] *= -0.4
                if a["p"][1] <= 0.03:
                    a["p"][1], a["landed"] = 0.03, True
        if not self.alcohols3:
            return
        for slot in self.flies:
            fly = slot.fly
            if fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
                continue
            a = min(self.alcohols3, key=lambda a: float(np.linalg.norm((a["p"] - fly.p[THX])[[0, 2]])))
            dxz = (a["p"] - fly.p[HEAD])[[0, 2]]
            on_floor = fly.p[THX, 1] < STAND3 + 0.2 and now >= fly.escape_until
            if float(np.linalg.norm(dxz)) > 0.2:
                if on_floor and now >= fly.stun_until and a["landed"]:
                    fly.yaw_target = math.atan2(dxz[1], dxz[0])
                    fly.walk_until, fly.run = now + 0.2, False
                continue
            if not on_floor or not a["landed"]:
                continue
            if now >= fly.eating_until:
                self.popup(fly.p[HEAD] + (0, 0.4, 0), random.choice(("SIP...", "GLUG!", "*HIC*")), (255, 140, 210))
                self.sound.play("yum")
                self.note("ALCOHOL  drinking: sweet + PAM reward [GAME RULE: inebriation]", source="rule")
            fly.eating_until = now + 0.4
            a["left"] -= 1 / 240
            slot.brain.poke("taste", None, 0.5)
            slot.brain.poke("sweet", None, 0.6, recruit=0.6)          # the sugar-pathway taste neurons (assays.py)
            slot.brain.poke("reward", None, 0.6)                      # PAM dopaminergic reward (real neurons)
            fly.inebriation = min(1.0, getattr(fly, "inebriation", 0.0) + 0.008)   # GAME RULE
            if a["left"] <= 0:
                drop_item(self.alcohols3, a)
                break

    def _decoy3d(self, now: float) -> None:
        """Decoy female in 3D (GAME RULE body, contact distance and COURTSHIP tag; contact drives the LgLG5-8 GRNs)."""
        if not hasattr(self, "decoys3") or not self.decoys3:
            return
        for dec in self.decoys3:
            if getattr(self, "grabbed_decoy3", None) is dec:
                continue
            if not dec.get("landed", False):
                dec["v"][1] -= GRAV
                dec["p"] += dec["v"]
                if dec["p"][1] <= 0.05:
                    dec["p"][1], dec["landed"] = 0.05, True
                    dec["v"][:] = 0.0
        for slot in self.flies:
            fly = slot.fly
            if fly.dead or fly.wrapped or fly.frozen_at is not None or fly.grabbed is not None:
                continue
            dec = min(self.decoys3, key=lambda d: float(np.linalg.norm(d["p"] - fly.p[HEAD])))
            d_p = dec["p"].copy()
            if d_p[1] < STAND3 * 0.5:
                d_p[1] = STAND3
            dist = float(np.linalg.norm(d_p - fly.p[HEAD]))
            if dist < 0.4 or float(np.linalg.norm(dec["p"] - fly.p[HEAD])) < 0.4:
                slot.brain.poke("pheromone", "foreleg", 0.6)
                fly.decoy_contact_until = now + 0.35
                if now >= getattr(fly, "court_until", 0.0):
                    fly.court_until = now + 0.5
                    self.note("COURTSHIP   foreleg contact (rule): drives the putative ppk23/ppk25 GRNs")
                    self.popup(fly.p[HEAD] + (0, 0.4, 0), "COURTSHIP", (255, 180, 220))

    def _die(self, slot: "k2.FlySlot", now: float) -> None:
        fly = slot.fly
        fly.dead_at = now
        fly.grabbed = None
        self.kills += 1
        slot.brain.kill()
        self.sound.play("death")
        self.note("DIED     brain drive cut, activity fading")
        self.popup(fly.p[HEAD] + (0, 0.5, 0), "K.O.!", (255, 90, 80), force=True)
        self.shake_until = now + 0.3
        if all(s.fly.dead for s in self.flies):
            self.killed_by = slot.damage_src or "being kicked"
            self.focus = self.flies.index(slot)

    def _fly_state(self, now: float) -> str:
        f = self.fly
        for cond, word in ((f.shattered_at is not None, "shattered"), (f.dissolved_at is not None, "dissolved"),
                           (f.frozen_at is not None, "frozen solid"), (f.dead, "dead"), (bool(f.stuck), "stuck on flypaper")):
            if cond:
                return word
        if f.arena == "pool" and (f.p[:, 1] < WATER3).any() and now >= f.escape_until:
            return "swimming"
        if f.wrapped:
            return "wrapped in silk"
        if f.grabbed is not None:
            return "grabbed"
        if getattr(f, "perch", None) is not None and now < f.eating_until:
            return "eating fruit"                    # the orchard holds a feeding fly aloft with escape_until
        if now < f.escape_until:
            return "flying"
        if now < f.eating_until:
            return "eating sugar"
        if now < f.stun_until:
            return "stunned"
        return f.action

    # --- per frame ----------------------------------------------------------------------------------------------------------
    def _fly_collisions3d(self, now: float) -> None:
        """Two flies that bump: a soft push-apart always, and if the impact is hard enough, a real touch-neuron poke
        on both brains (the same pattern as _kick, but fly-on-fly and symmetric)."""
        flies = self.flies
        for a in range(len(flies)):
            sa = flies[a]
            if sa.fly.dead or sa.fly.dissolved_at is not None or sa.fly.shattered_at is not None:
                continue
            for b in range(a + 1, len(flies)):
                sb = flies[b]
                if sb.fly.dead or sb.fly.dissolved_at is not None or sb.fly.shattered_at is not None:
                    continue
                fa, fb = sa.fly, sb.fly
                d = fb.p[THX] - fa.p[THX]
                dist = float(np.linalg.norm(d))
                overlap = FLY_TOUCH_RADIUS3 * 2 - dist
                if overlap <= 0:
                    continue
                n = d / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
                closing = float(np.dot((fb.p[THX] - fb.prev[THX]) - (fa.p[THX] - fa.prev[THX]), n))
                for i in range(N_P):
                    fa.p[i] -= n * overlap * 0.5
                    fb.p[i] += n * overlap * 0.5
                if closing < -0.02:                       # a real bump, not just jostling
                    s = float(np.clip(-closing / 0.12, 0.15, 1.0))
                    for slot, fly, side in ((sa, fa, -n), (sb, fb, n)):
                        fly.impulse(THX, side * -0.03)
                        if not fly.dead:
                            slot.brain.poke("body", None, 0.5 * s)
                            slot.brain.poke("legs", None, 0.3 * s)
                            fly.hurt = max(fly.hurt, 0.3)

    def _poll_spawn(self) -> None:
        if self._new_slot is None:
            return
        slot, self._new_slot, self._spawning = self._new_slot, None, False
        self.flies.append(slot)
        self.focus = len(self.flies) - 1
        self.popup(slot.fly.p[HEAD] + (0, 0.45, 0), "NEW FLY!", (170, 255, 200), force=True)
        self.note(f"SPAWNED  fly #{len(self.flies)} (seed {slot.seed})")

    def update3d(self, now: float, dt: float, keys, rel) -> None:
        self.frame += 1
        self._poll_spawn()
        if self.player_dead_at is not None:
            self.player.eye_h += (0.3 - self.player.eye_h) * 0.05       # you slump to the floor
        for slot in self.flies:                                         # for drawing between ticks in slow motion
            slot.fly.p_tick = slot.fly.p.copy()
        if self.challenge is not None:
            self.challenge.update(now)
        if getattr(self, "grabbed_decoy3", None) is not None:
            self.grabbed_decoy3["p"] = self._hold_point()
            self.grabbed_decoy3["landed"] = False
        self._environment(now)
        self._kick(now)
        for slot in list(self.flies):
            fly = slot.fly
            pin = self._hold_point()
            if fly.wrapped and self.spider3 is not None and self.spider3.get("target") is slot:
                pin = self.spider3["p"] - (0, 0.16, 0)
            for i, sp in fly.step(now, pin):
                s = float(np.clip((sp - 9) / 35, 0.05, 1))
                self.hit(slot, i, s)
                if sp > 18:
                    self.damage(slot, min(8.0, (sp - 18) * 0.35), "the floor" if fly.p[i, 1] < 0.2 else "the wall")
                if sp > 14 and fly.p[i, 1] < 0.2:
                    self.puff(np.array([fly.p[i, 0], 0.02, fly.p[i, 2]]), 3)
                if sp > 24 and not fly.dead:
                    self.sound.play("bonk", 0.4 + 0.6 * s)
                    fly.stun(now, 0.8 * s + 0.3)
                    if random.random() < 0.5:
                        self.popup(fly.p[i] + (0, 0.3, 0), random.choice(k2.OUCH))
        for sw in self.swats:
            if not sw[2] and now - sw[1] > 0.12:
                sw[2] = True
                self._swat3d(now)
        self.swats = [s for s in self.swats if now - s[1] < 0.55]
        for b in list(self.bombs3):
            b["v"][1] -= GRAV
            b["p"] += b["v"]
            if b["p"][1] < 0.08:
                b["p"][1] = 0.08
                b["v"][1] *= -0.35
                b["v"][[0, 2]] *= 0.7
            for ax, lim in ((0, RX - 0.08), (2, RZ - 0.08)):
                if abs(b["p"][ax]) > lim:
                    b["p"][ax] = math.copysign(lim, b["p"][ax])
                    b["v"][ax] *= -0.5
            push_out_boxes(b["p"][None, :], 0.08)
            if now - b["t"] > 1.8:
                drop_item(self.bombs3, b)
                self._explode3d(b, now)
        if self.torching and self.report is None and TOOLS[self.tool][0] in ("torch", "cleaner", "freeze"):
            self._jet(now, TOOLS[self.tool][0])
        if hasattr(self, "laser_state") and (self.torching or self.laser_state.is_active(now)) and TOOLS[self.tool][0] == "laser" and self.report is None:
            eye, d = self.aim()
            hit_slot, hit_part, hit_t = self._nearest_fly3d(eye, d, 15.0, 0.25)
            hit_any = False
            for slot in self.flies:
                if slot is hit_slot and not slot.fly.dead:
                    hit_any = True
                    cur = self.laser_state.apply(slot.brain, now, is_hitting=True)
                    if random.random() < 0.04:
                        act_txt = "STIM" if self.laser_state.mode == "activate" else "SILENCE"
                        self.popup(slot.fly.p[HEAD] + (0, 0.3, 0), f"LASER {act_txt} {self.laser_state.target_type}",
                                   (255, 180, 80) if self.laser_state.mode == "activate" else (80, 200, 255))
                else:
                    self.laser_state.apply(slot.brain, now, is_hitting=False)
            self.laser_state.hit_fly = hit_any
            self.laser_state.hit_pos = tuple(eye + d * hit_t) if (hit_any and hit_t is not None) else tuple(eye + d * 15.0)
        elif hasattr(self, "laser_state"):
            for slot in self.flies:
                if len(getattr(slot.brain, "_laser_rows", [])):
                    self.laser_state.apply(slot.brain, now, is_hitting=False)
        self._effects(now)
        for p in self.parts:
            p["p"] = p["p"] + p["v"]
            if p["kind"] == "flame":
                p["v"] = p["v"] * 0.95 + (0, 0.0015, 0)
            elif p["kind"] in ("cleaner", "freeze"):
                p["v"] = p["v"] * 0.93 + (0, -0.0003, 0)
            elif p["kind"] == "dust":
                p["v"] = p["v"] * 0.96 + (0, 0.0004, 0)
            elif p["kind"] == "fire":
                p["v"] = p["v"] * 0.9 + (0, 0.001, 0)
        self.parts = [p for p in self.parts if now - p["t"] < p["life"]]
        if len(self.parts) > 1500:
            self.parts = self.parts[-1500:]
        for sh in self.shards3:
            sh["v"][1] -= GRAV
            sh["p"] += sh["v"]
            if sh["p"][1] < sh["size"]:
                sh["p"][1] = sh["size"]
                sh["v"] *= (0.6, -0.3, 0.6)
            sh["rot"] += sh["v"] * 8
        self.bolts3 = [b for b in self.bolts3 if now - b[2] < 0.25]
        self.popups3 = [pu for pu in self.popups3 if now - pu[2] < 0.9]

        self._fly_collisions3d(now)

        all_dead = all(s.fly.dead for s in self.flies)
        duel_free = duel_can_fly = None
        for slot in list(self.flies):
            fly, br = slot.fly, slot.brain
            parts = br.pain_parts(br.fast, br.base)[0]
            parts[3] = max(parts[3], slot.pain_parts[3] * 0.96)
            slot.pain_parts += (parts - slot.pain_parts) * 0.15
            slot.pain += (float(br.pain_index(parts)) - slot.pain) * 0.15
            slot.reward += (float(np.clip((br.level("reward") - 1.0) / 0.6, 0, 1)) * 100 - slot.reward) * 0.1
            if not fly.dead and slot.pain > 25 and self.frame % 3 == 0:
                br.poke("punish", None, slot.pain / 100)
            self._vision(slot, now)
            self._scents(slot, now)
            self._learn(slot, now)
            if not fly.dead:
                slot.pain_peak = max(slot.pain_peak, slot.pain)
                if slot.pain >= 99:
                    slot.pain_max_s += 1 / 60
            slot.pain_trace.append(slot.pain)
            slot.pain_trace = slot.pain_trace[-720:]

            if not fly.dead:
                for (region, side), s in slot.pending_hits.items():
                    br.poke(region, side, s)
                    slot.hits += 1
                if slot.pending_damage:
                    if self.duel and getattr(slot, "player_scent", False):   # hurt it up close: it learns to fear you
                        self.fly_punish_until = max(self.fly_punish_until, now + PUNISH_PULSE_S)
                    floor = 1.0 if self.immortal else 0.0
                    fly.health = max(floor, fly.health - slot.pending_damage)
                    slot.last_damage = now
                    if fly.health <= 0:
                        self._die(slot, now)
                elif self.immortal and now - slot.last_damage > 1.5:
                    fly.health = min(MAX_HEALTH, fly.health + 0.1)
            slot.pending_hits.clear()
            slot.pending_damage = 0.0
            if fly.dead:
                if (all_dead and self.report is None and slot is self.flies[self.focus]
                        and now - fly.dead_at > k2.AUTOPSY_DELAY):
                    self.report = self._autopsy(slot, now)
                    self.death_frames = list(self.frames)
                continue

            # reactions read from the descending neurons
            can_fly = fly.grabbed is None and not fly.wrapped and fly.frost < 0.5 and fly.melt < 0.3 and fly.venom < 0.5
            can_fly = can_fly and fly.wet <= 0 and len(fly.stuck) < 2
            free = fly.grabbed is None and now >= fly.escape_until
            lv = {n: br.level(n) for n in ("jump", "run", "kick", "walk", "back", "turn_l", "turn_r", "fly", "escape")}
            fly.power = lv["fly"]
            if lv["escape"] > THRESH["escape"] and now >= fly.escape_ready and fly.grabbed is None and can_fly:
                fly.stun_until = 0.0
                fly.last_hit = np.asarray(slot.threat_x, float).copy()
                fly.escape(now)
                self.note(f"DODGE    giant fiber DNp01 x{lv['escape']:.1f}")
                self.on_reaction("DODGE", slot)
                self.popup(fly.p[HEAD] + (0, 0.4, 0), "DODGE!", (170, 255, 200))
                self.sound.play("dodge")
            elif lv["jump"] > THRESH["jump"] and now >= fly.escape_ready and free and can_fly:
                fly.stun_until = 0.0
                fly.escape(now)
                self.note(f"FLY AWAY head-touch DNs x{lv['jump']:.1f}")
                self.popup(fly.p[HEAD] + (0, 0.4, 0), "YIKES!", (160, 230, 255))
            elif (lv["fly"] > THRESH["fly"] and now >= fly.escape_ready and free and can_fly and now >= fly.stun_until
                  and now >= fly.eating_until and now >= getattr(slot, "asleep_until", 0.0)):
                fly.escape(now, seconds=random.uniform(2.5, 4.0), wander=True)
                self.note(f"TAKE OFF DNg02 x{lv['fly']:.2f}")
            if lv["run"] > THRESH["run"] and now >= fly.walk_until and free:
                fly.stun_until = min(fly.stun_until, now + 0.2)
                fly.yaw_target = angle_to(fly.away_from(fly.last_hit))
                fly.walk_until, fly.back_until, fly.run = now + 1.1, 0.0, True
                self.note(f"RUN      body-touch DNs x{lv['run']:.1f}")
            if lv["kick"] > THRESH["kick"] and now >= fly.flail_until:
                fly.flail_until = now + 0.6
                self.note(f"KICK     leg-touch DNs x{lv['kick']:.1f}")
            if lv["back"] > THRESH["back"] and now >= fly.back_until and now >= fly.walk_until:
                fly.back_until = now + 0.8
                self.note(f"BACK UP  MDN x{lv['back']:.1f}")
            elif (lv["walk"] > THRESH["walk"] and now >= fly.walk_until and now >= fly.back_until
                  and now >= getattr(slot, "asleep_until", 0.0)):
                fly.walk_until, fly.run = now + 1.2, False
                self.note(f"WALK     DNp09 x{lv['walk']:.1f}")
            self._memory_behavior(slot, now, free, can_fly)
            self.readouts(slot, now)
            if self.duel and slot is self.flies[self.focus]:
                duel_free, duel_can_fly = free, can_fly
            lamp_idle = free and now >= fly.escape_until and now >= fly.stun_until and now >= fly.walk_until
            if k2.ARENAS[self.arena_i] == "lamp" and lamp_idle and now >= slot.photo_ready:
                slot.photo_ready = now + random.uniform(3.0, 6.0)
                fly.yaw_target = angle_to(LAMP3 - fly.p[THX])
                if can_fly and random.random() < 0.6:
                    fly.escape(now, seconds=random.uniform(3.0, 5.0), wander=True)
                    fly.fly_target = LAMP3 - (0, 0.5, 0)
                    self.note("TO LIGHT flies to the lamp")
                else:
                    fly.walk_until, fly.run = now + 1.5, False
            turn = lv["turn_r"] - lv["turn_l"]
            if abs(turn) > THRESH["turn"] and now >= fly.turn_ready and free and now >= fly.walk_until \
                    and not (self.duel and slot is self.flies[self.focus]):
                fly.turn_ready = now + 1.5
                fly.yaw_target = fly.yaw + math.copysign(random.uniform(0.9, 1.6), turn)
                self.note(f"TURN {'R' if turn > 0 else 'L'}   DNa01/02 R-L {turn:+.1f}")
            self._aggression(slot, now, free)
            self._sleep(slot, now, free)

        if self.duel:
            self._duel_senses(now)
        self._pellets3d(now)
        if self.duel and duel_free is not None:
            self._duel_motor(now, duel_free, duel_can_fly)
        self._update_focus()
        self._training_tick(now)
        self._sound_update(now)

    # --- world drawing --------------------------------------------------------------------------------------------------------
    def _build_room(self) -> list:
        R = []

        def box(center, size, color, pattern=P_NONE, glow=0.0):
            R.append(("cube", trs(center, None, size), color, pattern, glow))

        box((0, -0.05, 0), (2 * RX, 0.1, 2 * RZ), (0.66, 0.46, 0.28), P_WOOD)
        box((0, RY + 0.05, 0), (2 * RX, 0.1, 2 * RZ), (0.93, 0.92, 0.9), P_CEIL)
        wall = (0.8, 0.73, 0.6)
        box((0, RY / 2, -RZ - 0.05), (2 * RX, RY, 0.1), wall, P_WALLPAPER)
        box((0, RY / 2, RZ + 0.05), (2 * RX, RY, 0.1), wall, P_WALLPAPER)
        box((-RX - 0.05, RY / 2, 0), (0.1, RY, 2 * RZ), wall, P_WALLPAPER)
        box((RX + 0.05, RY / 2, 0), (0.1, RY, 2 * RZ), wall, P_WALLPAPER)
        trim = (0.94, 0.92, 0.88)
        for y, h in ((0.06, 0.12), (0.95, 0.04), (RY - 0.04, 0.08)):
            box((0, y, -RZ + 0.015), (2 * RX, h, 0.03), trim)
            box((0, y, RZ - 0.015), (2 * RX, h, 0.03), trim)
            box((-RX + 0.015, y, 0), (0.03, h, 2 * RZ), trim)
            box((RX - 0.015, y, 0), (0.03, h, 2 * RZ), trim)
        # window on the far wall
        box((1.4, 1.75, -RZ + 0.012), (1.9, 1.25, 0.02), (1, 1, 1), 8)
        for dx, dy, sx, sy in ((0, 0.66, 2.02, 0.08), (0, -0.66, 2.02, 0.08), (-0.99, 0, 0.08, 1.4), (0.99, 0, 0.08, 1.4),
                               (0, 0, 0.05, 1.3), (0, 0, 1.95, 0.05)):
            box((1.4 + dx, 1.75 + dy, -RZ + 0.04), (sx, sy, 0.06), trim)
        box((1.4, 1.08, -RZ + 0.1), (2.1, 0.05, 0.2), trim)
        # couch
        blue, cushion = (0.33, 0.4, 0.55), (0.42, 0.5, 0.66)
        box((-2.3, 0.22, -RZ + 0.55), (2.3, 0.36, 0.9), blue)
        box((-2.3, 0.72, -RZ + 0.16), (2.3, 0.75, 0.26), blue)
        box((-3.39, 0.45, -RZ + 0.55), (0.22, 0.52, 0.95), blue)
        box((-1.21, 0.45, -RZ + 0.55), (0.22, 0.52, 0.95), blue)
        box((-2.83, 0.48, -RZ + 0.6), (1.0, 0.16, 0.7), cushion)
        box((-1.77, 0.48, -RZ + 0.6), (1.0, 0.16, 0.7), cushion)
        # bookshelf
        dark = (0.33, 0.22, 0.14)
        box((RX - 0.22, 1.02, -1.0), (0.42, 2.04, 1.8), dark)
        for k in range(4):
            box((RX - 0.24, 0.3 + k * 0.48, -1.0), (0.36, 0.34, 1.62), (1, 1, 1), P_BOOKS)
            box((RX - 0.22, 0.12 + k * 0.48, -1.0), (0.4, 0.03, 1.76), dark)
        # plant
        R.append(("cylinder", trs((RX - 0.6, 0, RZ - 0.6), None, (0.25, 0.45, 0.25)), (0.7, 0.35, 0.22), P_NONE, 0.0))
        for k in range(7):
            a = k * 0.9
            R.append(("sphere", trs((RX - 0.6 + 0.2 * math.cos(a), 0.75 + 0.12 * (k % 3), RZ - 0.6 + 0.2 * math.sin(a)), None,
                                    (0.24, 0.3, 0.24)), (0.24, 0.52, 0.26), P_NONE, 0.0))
        # door behind you, a picture, the rug and the ceiling light
        box((-2.2, 1.05, RZ - 0.03), (0.95, 2.1, 0.05), (0.52, 0.34, 0.2))
        R.append(("sphere", trs((-1.85, 1.0, RZ - 0.08), None, (0.04, 0.04, 0.04)), (0.85, 0.72, 0.35), P_NONE, 0.3))
        box((-RX + 0.03, 1.75, 1.2), (0.04, 0.8, 1.1), (0.25, 0.18, 0.12))
        box((-RX + 0.055, 1.75, 1.2), (0.02, 0.66, 0.96), (0.9, 0.62, 0.3), P_RUG, 0.1)
        R.append(("cylinder", trs((-0.4, 0.0, -0.9), None, (1.35, 0.012, 1.0)), (0.9, 0.52, 0.36), P_RUG, 0.0))
        R.append(("cylinder", trs((0, RY - 0.06, 0), None, (0.3, 0.06, 0.3)), (1.0, 0.97, 0.9), P_NONE, 1.3))
        return R

    def draw_world(self, rd: Renderer, now: float) -> None:
        arena = k2.ARENAS[self.arena_i]
        if arena in outdoors.OUTDOOR:
            self._draw_outdoors(rd, now)
        else:
            for mesh, model, color, pattern, glow in self._room:
                rd.add(mesh, model, color, pattern, glow)
        if arena in outdoors.OUTDOOR:
            pass
        elif arena == "fan":
            self._draw_fan(rd, now)
        elif arena == "flypaper":
            x0, x1, z0, z1 = PAPER3
            rd.add("cube", trs(((x0 + x1) / 2, 0.018, (z0 + z1) / 2), None, (x1 - x0, 0.01, z1 - z0)), (0.88, 0.72, 0.24), P_PAPER)
        elif arena == "pool":
            rd.add("cube", trs((0, WATER3, 0), None, (2 * RX, 0.01, 2 * RZ)), (0.18, 0.42, 0.62, 0.55), P_WATER)
            rd.add("cube", trs((0, WATER3 / 2, 0), None, (2 * RX - 0.02, WATER3, 2 * RZ - 0.02)), (0.1, 0.3, 0.45, 0.25), P_NONE)
        elif arena == "lamp":
            top = np.array([LAMP3[0], RY, LAMP3[2]])
            rd.add("cylinder", segment(LAMP3 + (0, 0.2, 0), top, 0.008), (0.15, 0.15, 0.15))
            rd.add("cylinder", trs(LAMP3 + (0, 0.02, 0), None, (0.26, 0.2, 0.26)), (0.25, 0.32, 0.28))
            rd.add("sphere", trs(LAMP3, None, (0.11, 0.13, 0.11)), (1.0, 0.95, 0.8), P_NONE, 3.0)
            for k in range(3):
                rd.particle(LAMP3, 0.35 + 0.25 * k + 0.03 * math.sin(now * 3 + k), (1.0, 0.85, 0.5, 0.12), additive=True)
        elif arena == "thermo":
            n = 24                                   # the floor strip: blue (15 C) through green to red (35 C)
            w = 2 * THERMO_HALF3 / n
            for k in range(n):
                t = (k + 0.5) / n * 2 - 1
                cold, hot = k2.thermo_gradient(t)
                col = (0.25 + 0.65 * hot, 0.7 - 0.45 * (cold + hot), 0.4 + 0.5 * cold)
                rd.add("cube", trs((t * THERMO_HALF3, 0.012, 0), None, (w, 0.01, 2 * RZ - 0.4)), col, P_NONE)
        elif arena == "escaperoom":
            self._draw_fan(rd, now)
            x0, x1, z0, z1 = PAPER3
            rd.add("cube", trs(((x0 + x1) / 2, 0.018, (z0 + z1) / 2), None, (x1 - x0, 0.01, z1 - z0)), (0.88, 0.72, 0.24), P_PAPER)
            top = np.array([LAMP3[0], RY, LAMP3[2]])
            rd.add("cylinder", segment(LAMP3 + (0, 0.2, 0), top, 0.008), (0.15, 0.15, 0.15))
            rd.add("cylinder", trs(LAMP3 + (0, 0.02, 0), None, (0.26, 0.2, 0.26)), (0.25, 0.32, 0.28))
            rd.add("sphere", trs(LAMP3, None, (0.11, 0.13, 0.11)), (1.0, 0.95, 0.8), P_NONE, 3.0)
            for k in range(3):
                rd.particle(LAMP3, 0.35 + 0.25 * k + 0.03 * math.sin(now * 3 + k), (1.0, 0.85, 0.5, 0.12), additive=True)
            sugar_pos = np.array([RX - 0.7, 0.04, 0.0])
            rd.add("cylinder", trs(sugar_pos, None, (0.22, 0.02, 0.22)), (0.85, 0.85, 0.9))
            rd.add("sphere", trs(sugar_pos + (0, 0.02, 0), None, (0.16, 0.05, 0.16)), (0.95, 0.95, 0.95), P_NONE, 1.5)
        alpha = self.clock.alpha
        for slot in self.flies:
            fly = slot.fly
            prev = getattr(fly, "p_tick", None)
            if alpha < 1.0 and prev is not None and prev.shape == fly.p.shape:
                real_p = fly.p
                fly.p = prev + (real_p - prev) * alpha
                self._draw_fly(rd, now, fly)
                fly.p = real_p
            else:
                self._draw_fly(rd, now, fly)
        self._draw_extras(rd, now)

    def _camera(self) -> tuple[np.ndarray, np.ndarray]:
        if getattr(self, "photo_mode", False) and getattr(self, "free_cam", None) is not None:
            return np.asarray(self.free_cam.eye, float), np.asarray(self.free_cam.forward(), float)
        return self.player.eye, self.player.forward()

    @staticmethod
    def _visible(pos: np.ndarray, eye: np.ndarray, fwd: np.ndarray, max_dist: float, radius: np.ndarray | float = 0.5):
        """Which of these points to draw: within max_dist, and not well behind the camera. Keeps the outdoor worlds
        to a few hundred instances a frame so rendering never starves the brain threads of the interpreter."""
        v = pos - eye
        d = np.linalg.norm(v, axis=1)
        return (d < max_dist + radius) & ((v @ fwd) > -0.35 * d - radius)

    def _draw_outdoors(self, rd: Renderer, now: float) -> None:
        eye, fwd = self._camera()
        w = outdoors.spec(self.world)
        sky = w.sky
        rd.add("sphere", trs(eye, None, (80.0, 80.0, 80.0)), sky, P_SKYDOME)
        gx, gz = round(float(eye[0]) / 10) * 10, round(float(eye[2]) / 10) * 10   # the ground follows you in 10 m steps
        rd.add("cube", trs((gx, -0.01, gz), None, (240.0, 0.02, 240.0)), w.ground, P_GRASS)
        sc = self.scenery
        if sc.get("_models") is None:                 # static scenery: build every model matrix once per arena
            sc["_models"] = self._scenery_models(sc)
        for kind, max_d, rad in (("rocks", DRAW_DIST, 1.0), ("tufts", TUFT_DIST, 0.3), ("trees", DRAW_DIST, 2.5)):
            pos, rows = sc["_models"][kind]
            if not len(pos):
                continue
            for i in np.flatnonzero(self._visible(pos, eye, fwd, max_d, rad)):
                for mesh, model, col in rows[i]:
                    rd.add(mesh, model, col)
        if self.orchard is not None:
            self._draw_fruit(rd, now, eye, fwd)

    @staticmethod
    def _scenery_models(sc: dict) -> dict:
        """(positions for culling, per-object list of (mesh, model matrix, colour)) for rocks, tufts and trees."""
        out = {}
        rocks = [[("sphere", trs(c, rot_y(yaw), scale), (tone, tone * 0.97, tone * 0.92))]
                 for c, scale, yaw, tone in sc["rocks"]]
        out["rocks"] = (np.array([r[0] for r in sc["rocks"]]).reshape(-1, 3), rocks)
        tufts = []
        for c, h, tone in sc["tufts"]:
            base = np.array([c[0], 0.0, c[2]])
            col = (0.28 * tone, 0.5 * tone, 0.18 * tone)
            tufts.append([("cylinder", segment(base, base + (0.03, h, 0.01), 0.008), col),
                          ("cylinder", segment(base, base + (-0.02, h * 0.8, 0.03), 0.008), col)])
        out["tufts"] = (np.array([t[0] for t in sc["tufts"]]).reshape(-1, 3), tufts)
        trees = []
        for t in sc["trees"]:
            base, h, cr = t["pos"], t["height"], t["crown"]
            top = base + (0, h - 0.35, 0)
            trees.append([("cylinder", segment(base, base + (0, h * 0.62, 0), 0.13), (0.36, 0.25, 0.16)),
                          ("sphere", trs(top, None, (cr, cr * 0.72, cr)), (0.2, 0.42, 0.18)),
                          ("sphere", trs(top + (0.35, 0.25, -0.2), None, (cr * 0.7, cr * 0.55, cr * 0.7)),
                           (0.24, 0.48, 0.2))])
        out["trees"] = (np.array([t["pos"] for t in sc["trees"]]).reshape(-1, 3), trees)
        return out

    def _draw_fruit(self, rd: Renderer, now: float, eye, fwd) -> None:
        """Ripe fruit shrink and brown as they are eaten down; an emptied one drops to the ground and fades."""
        o = self.orchard
        if getattr(o, "_pos", None) is None:
            o._pos = np.array([f.pos for f in o.fruit])
        near = self._visible(o._pos, eye, fwd, DRAW_DIST, 0.2)
        for i in np.flatnonzero(near):
            f = o.fruit[i]
            if f.ripe:
                key = (f.feeds_left, f.feeds_max, f.fermented)
                cached = getattr(f, "_models", None)
                if cached is None or cached[0] != key:        # rebuilt only when it's eaten, not every frame
                    full = f.fullness
                    base = np.array((0.45, 0.22, 0.4)) if f.fermented else np.array((0.9, 0.22, 0.14))
                    col = tuple(base * (0.55 + 0.45 * full) + np.array((0.35, 0.25, 0.12)) * (1 - full))
                    r = 0.035 + 0.045 * full ** 0.5
                    f._models = cached = (key, [("sphere", trs(f.pos, None, (r, r * 1.05, r)), col),
                                                ("cylinder", segment(f.pos + (0, r * 0.8, 0), f.pos + (0, r + 0.05, 0),
                                                                     0.006), (0.3, 0.22, 0.1))])
                for mesh, model, col in cached[1]:
                    rd.add(mesh, model, col)
            elif f.fell_at is not None and now - f.fell_at < 6.0:
                t = now - f.fell_at
                y = max(0.03, float(f.pos[1]) - 4.9 * t * t)
                fade = float(np.clip(1.0 - (t - 3.0) / 3.0, 0.0, 1.0))
                rd.add("sphere", trs((f.pos[0], y, f.pos[2]), None, (0.04, 0.03, 0.04)), (0.35, 0.22, 0.1, fade))

    def _draw_fan(self, rd: Renderer, now: float) -> None:
        base = FAN3
        hub = base + (0, 1.0, 0)
        grey = (0.72, 0.74, 0.78)
        rd.add("cylinder", trs(base, None, (0.3, 0.05, 0.3)), grey)
        rd.add("cylinder", segment(base, hub - (0.1, 0, 0), 0.03), grey)
        rd.add("sphere", trs(hub - (0.12, 0, 0), None, (0.16, 0.14, 0.14)), grey)
        rim = rot_z(math.pi / 2)
        rd.add("torus", trs(hub + (0.1, 0, 0), rim, (0.48, 0.48, 0.48)), (0.85, 0.87, 0.9))
        for k in range(3):
            a = now * 22 + k * 2.094
            offset = np.array([0.06, math.cos(a), math.sin(a)]) * (1, 0.22, 0.22)
            blade = rot_x(a) @ np.eye(3)
            rd.add("sphere", trs(hub + offset, blade, (0.03, 0.22, 0.1)), (0.6, 0.8, 0.95, 0.85))
        rd.add("sphere", trs(hub + (0.08, 0, 0), None, (0.05, 0.05, 0.05)), (0.3, 0.3, 0.35))

    def _draw_fly(self, rd: Renderer, now: float, fly: "Fly3D") -> None:
        p = fly.p
        hurt, dead = fly.hurt, fly.dead
        if fly.dissolved_at is not None:
            e = min(1.0, (now - fly.dissolved_at) / 1.5)
            x, z = p[THX, 0], p[THX, 2]
            rd.add("cylinder", trs((x, 0.006, z), None, ((70 + 60 * e) * S, 0.012, (60 + 50 * e) * S)), (0.43, 0.41, 0.24, 0.85))
            rd.add("sphere", trs((x + 0.15, 0.03, z), rot_y(0.4), (0.16, 0.01, 0.08)), (0.8, 0.85, 0.93, 0.4))
            for dx in (-0.05, 0.04):
                rd.add("sphere", trs((x + dx, 0.025, z + 0.02), None, (0.018, 0.018, 0.018)), (0.7, 0.1, 0.1))
            for k in range(3):
                ph = (now * 0.8 + k / 3) % 1.0
                rd.particle((x + (k - 1) * 0.25, 0.03 + 0.08 * ph, z), 0.02 + 0.02 * ph, (0.85, 0.85, 0.7, 0.6 * (1 - ph)))
            return
        if fly.shattered_at is not None:
            return
        zapped = now < fly.zap_until and int(now * 40) % 2 == 0
        k_ = 1 - 0.4 * fly.melt

        def col(c):
            if zapped:
                return (0.8, 0.92, 1.0, 1.0)
            return _c(k2.shade(c, hurt, dead, fly.char, fly.melt, fly.frost))

        axis = p[HEAD] - p[ABD]
        fwd = axis / (np.linalg.norm(axis) or 1)
        side = body_axes(fly.yaw)[2]
        side = side - fwd * (side @ fwd)
        side = side / (np.linalg.norm(side) or 1)
        up = np.cross(side, fwd)
        Rb = np.stack([fwd, up, side], 1)
        if now < getattr(fly, "proboscis_until", 0.0):          # proboscis out while MN9 responds to sugar
            tip = p[HEAD] + fwd * 0.1 - up * 0.16
            rd.add("cylinder", segment(p[HEAD] - up * 0.06, tip, 0.012), (0.5, 0.36, 0.22))
        fa = p[THX] - p[ABD]
        fa = fa / (np.linalg.norm(fa) or 1)
        Ra = np.stack([fa, np.cross(side, fa), side], 1)
        leg_col = col((132, 95, 50))
        for k in range(6):
            sgn = 1 if k < 3 else -1
            hip = p[THX] + fwd * (14 - 12 * (k % 3)) * S - up * 10 * S + side * sgn * 8 * S
            rd.add("cylinder", segment(hip, p[KNEE[k]], 2.0 * S), leg_col)
            rd.add("sphere", trs(p[KNEE[k]], None, (2.1 * S,) * 3), leg_col)
            rd.add("cylinder", segment(p[KNEE[k]], p[FOOT[k]], 1.3 * S), leg_col)
            rd.add("sphere", trs(p[FOOT[k]], None, (1.8 * S,) * 3), col((96, 66, 32)))
        # Drosophila melanogaster, not a bee: a pale yellow-tan body, a short abdomen tapering to a dark tip with
        # bands only across the top of each segment (the shader's P_STRIPES), and eyes that fill most of the head.
        abd_l, abd_r = 24 * S * k_, 17.5 * S * k_
        rd.add("sphere", trs(p[ABD], Ra, (abd_l, abd_r, abd_r)), col((206, 172, 104)), P_STRIPES)
        rd.add("sphere", trs(p[ABD] - fa * abd_l * 0.52, Ra, (abd_l * 0.42, abd_r * 0.78, abd_r * 0.78)),
               col((96, 70, 38)))                                    # the tapered, dark tip
        rd.add("sphere", trs(p[THX], Rb, (23 * S * k_, 19.5 * S * k_, 19 * S * k_)), col((176, 146, 92)))
        rd.add("sphere", trs(p[THX] - fwd * 6 * S + up * 9 * S, Rb, (13 * S * k_, 7 * S * k_, 15 * S * k_)),
               col((150, 120, 74)))                                  # the scutellum behind the wing bases
        rd.add("sphere", trs(p[HEAD], Rb, (13 * S * k_, 14 * S * k_, 15 * S * k_)), col((186, 154, 96)))
        for sgn in (1, -1):
            eye = p[HEAD] + fwd * 1.5 * S + up * 2.5 * S + side * sgn * 8.5 * S * k_
            rd.add("sphere", trs(eye, Rb, (12 * S * k_, 14 * S * k_, 11.5 * S * k_)), col((222, 44, 30)), P_EYE)
            if dead:
                for a in (0.8, -0.8):
                    rd.add("cube", trs(eye + side * sgn * 8 * S, Rb @ rot_z(a), (18 * S, 3 * S, 3 * S)), (0.12, 0.08, 0.08))
            else:
                rd.add("sphere", trs(eye + up * 5 * S + fwd * 4 * S + side * sgn * 6 * S, None, (2.4 * S,) * 3), (1, 0.85, 0.8), P_NONE, 0.6)
            a0 = p[HEAD] + fwd * 11 * S + up * 8 * S + side * sgn * 4 * S
            rd.add("cylinder", segment(a0, a0 + fwd * 10 * S + up * 10 * S + side * sgn * 5 * S, 1.2 * S), col((70, 45, 20)))
        rd.add("cylinder", segment(p[HEAD] + fwd * 8 * S - up * 10 * S, p[HEAD] + fwd * 14 * S - up * 20 * S, 1.8 * S), col((70, 45, 20)))
        if not fly.wrapped:                     # bristles: a real fly is covered in them, a bee is furry instead
            bristle = col((74, 52, 28))
            for bx, by, bs, bl in ((-2, 15, 0, 7), (4, 15, 5, 6), (4, 15, -5, 6), (-8, 13, 7, 7), (-8, 13, -7, 7),
                                   (-13, 9, 0, 8)):
                root = p[THX] + fwd * bx * S + up * by * S + side * bs * S * k_
                rd.add("cylinder", segment(root, root + (up * 0.8 - fwd * 0.6) * bl * S, 0.7 * S), bristle)
            for sgn in (1, -1):
                root = p[HEAD] + fwd * 4 * S + up * 9 * S + side * sgn * 4 * S * k_
                rd.add("cylinder", segment(root, root + (up * 0.9 + fwd * 0.4) * 6 * S, 0.6 * S), bristle)
        flap = not dead and (now < fly.escape_until or (fly.grabbed is not None and random.random() < 0.3))
        for i, sgn in ((0, 1), (1, -1)):
            base = p[THX] + up * 12 * S + side * sgn * 5 * S
            tip = p[WING[i]]
            if flap:
                tip = base + (tip - base) * 0.8 + up * 38 * S * math.sin(now * 90 + i) + side * sgn * 10 * S
            span = float(np.linalg.norm(tip - base))
            rd.add("sphere", trs((base + tip) / 2, frame_from_x(tip - base), (span / 2 + 9 * S, 0.9 * S, 12 * S)),
                   (0.90, 0.94, 0.99, 0.34 if not fly.melt else 0.18), P_NONE, 0.12)
        if self.duel and not dead and fly is self.fly:             # the blaster, strapped under its head (duelist only)
            muzzle, _ = self._muzzle()
            base = p[THX] + fwd * 10 * S - up * 4 * S
            rd.add("cube", trs((base + muzzle) / 2 - up * 0.02, Rb, (0.2, 0.06, 0.07)), (0.25, 0.26, 0.3))
            rd.add("cylinder", segment(muzzle - fwd * 0.08, muzzle + fwd * 0.02, 0.018), (0.45, 0.47, 0.5))
            rd.add("sphere", trs(muzzle + fwd * 0.02, None, (0.022,) * 3), (0.45, 1.0, 0.4), P_NONE, 2.5)
        # overlays on the body
        if fly.soak > 0.05:
            rng = random.Random(int(now * 8))
            for _ in range(int(4 + 10 * fly.soak)):
                q = p[rng.choice((HEAD, THX, ABD, ABD))] + np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.12), rng.uniform(-0.12, 0.12)])
                rd.particle(q, rng.uniform(0.012, 0.03), (0.95, 0.97, 0.92, 0.7))
        if fly.frozen_at is not None:
            lo, hi = p.min(0) - 0.08, p.max(0) + 0.08
            rd.add("cube", trs((lo + hi) / 2, None, hi - lo), (0.72, 0.87, 0.97, 0.35), P_ICE, 0.1)
        elif fly.frost > 0.1:
            rng = random.Random(7)
            for _ in range(int(14 * fly.frost)):
                q = p[rng.choice((HEAD, THX, ABD, ABD))] + np.array([rng.uniform(-0.13, 0.13), rng.uniform(-0.1, 0.12), rng.uniform(-0.13, 0.13)])
                rd.add("cube", trs(q, rot_y(rng.uniform(0, 3)) @ rot_x(0.6), (0.02, 0.02, 0.02)), (0.9, 0.97, 1.0), P_ICE, 0.4)
        if fly.wrapped:
            mid = (p[HEAD] + p[ABD]) / 2
            L = float(np.linalg.norm(axis))
            rd.add("sphere", trs(mid, Rb, (L / 2 + 0.14, 0.15, 0.15)), (0.94, 0.94, 0.9, 0.78))
            for kk in range(-3, 4):
                rd.add("torus", trs(mid + fwd * kk * 0.06, Rb @ rot_z(math.pi / 2), (0.155, 0.155, 0.155)), (0.98, 0.98, 0.95), P_NONE, 0.2)
        if now < fly.burn_until:
            for i in (HEAD, THX, ABD):
                rd.particle(p[i] + np.random.uniform(-0.08, 0.08, 3) + (0, 0.06, 0), random.uniform(0.04, 0.09),
                            (1.0, random.uniform(0.45, 0.8), 0.15, 0.8), additive=True)
        if dead:
            e = min(1.0, (now - fly.dead_at) / 2.0)
            rd.add("torus", trs(p[HEAD] + (0, 0.2 + 0.06 * e, 0), None, (0.09, 0.09, 0.09)), (1.0, 0.85, 0.35), P_NONE, 1.5)
        elif now < fly.stun_until:
            for kk in range(4):
                a = now * 6 + kk * math.pi / 2
                rd.add("sphere", trs(p[HEAD] + (0.16 * math.cos(a), 0.18 + 0.03 * math.sin(a * 2), 0.16 * math.sin(a)), None, (0.022,) * 3),
                       (1.0, 0.84, 0.25), P_NONE, 1.2)
        if now < fly.eating_until:
            for kk in range(3):
                ph = (now * 0.9 + kk / 3) % 1.0
                rd.particle(p[HEAD] + ((kk - 1) * 0.1, 0.15 + 0.3 * ph, 0), 0.04, (1.0, 0.45, 0.6, 1 - ph), additive=True)
        self._shadow(rd, p[THX], 0.34)

    def _shadow(self, rd: Renderer, pos, radius: float) -> None:
        h = float(pos[1])
        k = float(np.clip(1 - h / 3.0, 0.15, 1))
        y = WATER3 + 0.004 if k2.ARENAS[self.arena_i] == "pool" and h > WATER3 else 0.008
        rd.add("cylinder", trs((pos[0], y, pos[2]), None, (radius * (1.2 - 0.5 * k), 0.002, radius * (1.2 - 0.5 * k) * 0.8)),
               (0.0, 0.0, 0.0, 0.35 * k), layer="blend")

    def _draw_cva_puffs3d(self, rd: Renderer, now: float) -> None:
        active = []
        for pf in getattr(self, "cva_puffs3", []):
            dt = now - pf["t"]
            dur = pf.get("duration", 0.65)
            if dt < dur:
                active.append(pf)
                progress = dt / dur
                reach_r = 2.5 * (progress ** 0.5)
                alpha = 0.28 * (1.0 - progress)
                rd.add("sphere", trs(pf["p"], None, (reach_r, reach_r, reach_r)),
                       (1.0, 0.72, 0.40, alpha), P_NONE, 0.1, layer="blend")
                rd.add("sphere", trs(pf["p"], None, (reach_r * 0.55, reach_r * 0.55, reach_r * 0.55)),
                       (1.0, 0.82, 0.50, alpha * 0.7), P_NONE, 0.15, layer="blend")
                if int(now * 20) % 2 == 0:
                    for _ in range(2):
                        off = np.random.uniform(-reach_r * 0.35, reach_r * 0.35, 3)
                        rd.particle(pf["p"] + off, 0.08 + 0.1 * (1.0 - progress),
                                    (1.0, 0.75, 0.45, alpha * 0.9), additive=True)
        self.cva_puffs3 = active

    def _draw_decoy3d(self, rd: Renderer, now: float, dec: dict) -> None:
        """Draw 3D decoy with the game's fly model as a female: larger, rounder abdomen with the
        female band pattern across all tergites, no sex combs, clearly a fly, tinted slightly so it reads as a decoy."""
        def col(c):
            if any(x > 1.0 for x in c):
                r, g, b = [x / 255.0 for x in c]
            else:
                r, g, b = c[:3]
            k = 0.22
            return (r * (1 - k) + 0.78 * k, g * (1 - k) + 0.71 * k, b * (1 - k) + 0.86 * k)

        yaw = float(dec.get("yaw", 0.0))
        dec_p = dec["p"].copy()
        if dec_p[1] < STAND3 * 0.5:
            dec_p[1] = STAND3
        p = to_world(REST3, yaw) + dec_p

        axis = p[HEAD] - p[ABD]
        fwd = axis / (np.linalg.norm(axis) or 1)
        side = body_axes(yaw)[2]
        side = side - fwd * (side @ fwd)
        side = side / (np.linalg.norm(side) or 1)
        up = np.cross(side, fwd)
        Rb = np.stack([fwd, up, side], 1)

        fa = p[THX] - p[ABD]
        fa = fa / (np.linalg.norm(fa) or 1)
        Ra = np.stack([fa, np.cross(side, fa), side], 1)

        leg_col = col((132, 95, 50))
        foot_col = col((96, 66, 32))
        for k in range(6):
            sgn = 1 if k < 3 else -1
            hip = p[THX] + fwd * (14 - 12 * (k % 3)) * S - up * 10 * S + side * sgn * 8 * S
            rd.add("cylinder", segment(hip, p[KNEE[k]], 2.0 * S), leg_col)
            rd.add("sphere", trs(p[KNEE[k]], None, (2.1 * S,) * 3), leg_col)
            rd.add("cylinder", segment(p[KNEE[k]], p[FOOT[k]], 1.3 * S), leg_col)
            rd.add("sphere", trs(p[FOOT[k]], None, (1.8 * S,) * 3), foot_col)

        # Female Drosophila melanogaster: larger, rounder abdomen with the female band pattern across all tergites
        # without the solid black/dark male tip cap, tapering instead to a light pointed ovipositor tip.
        abd_l, abd_r = 29.0 * S, 22.0 * S
        rd.add("sphere", trs(p[ABD], Ra, (abd_l, abd_r, abd_r)), col((206, 172, 104)), P_STRIPES)
        # Tapered lighter terminal segment (ovipositor tip)
        rd.add("sphere", trs(p[ABD] - fa * abd_l * 0.52, Ra, (abd_l * 0.40, abd_r * 0.65, abd_r * 0.65)),
               col((212, 185, 130)), P_STRIPES)
        rd.add("sphere", trs(p[ABD] - fa * abd_l * 0.74, Ra, (abd_l * 0.22, abd_r * 0.35, abd_r * 0.35)),
               col((220, 195, 145)))

        # Thorax and scutellum
        rd.add("sphere", trs(p[THX], Rb, (23 * S, 19.5 * S, 19 * S)), col((176, 146, 92)))
        rd.add("sphere", trs(p[THX] - fwd * 6 * S + up * 9 * S, Rb, (13 * S, 7 * S, 15 * S)),
               col((150, 120, 74)))

        # Head and eyes
        rd.add("sphere", trs(p[HEAD], Rb, (13 * S, 14 * S, 15 * S)), col((186, 154, 96)))
        for sgn in (1, -1):
            eye = p[HEAD] + fwd * 1.5 * S + up * 2.5 * S + side * sgn * 8.5 * S
            rd.add("sphere", trs(eye, Rb, (12 * S, 14 * S, 11.5 * S)), col((222, 44, 30)), P_EYE)
            rd.add("sphere", trs(eye + up * 5 * S + fwd * 4 * S + side * sgn * 6 * S, None, (2.4 * S,) * 3), (1, 0.85, 0.8), P_NONE, 0.6)
            a0 = p[HEAD] + fwd * 11 * S + up * 8 * S + side * sgn * 4 * S
            rd.add("cylinder", segment(a0, a0 + fwd * 10 * S + up * 10 * S + side * sgn * 5 * S, 1.2 * S), col((70, 45, 20)))
        rd.add("cylinder", segment(p[HEAD] + fwd * 8 * S - up * 10 * S, p[HEAD] + fwd * 14 * S - up * 20 * S, 1.8 * S), col((70, 45, 20)))

        # Bristles on thorax and head
        bristle = col((74, 52, 28))
        for bx, by, bs, bl in ((-2, 15, 0, 7), (4, 15, 5, 6), (4, 15, -5, 6), (-8, 13, 7, 7), (-8, 13, -7, 7), (-13, 9, 0, 8)):
            root = p[THX] + fwd * bx * S + up * by * S + side * bs * S
            rd.add("cylinder", segment(root, root + (up * 0.8 - fwd * 0.6) * bl * S, 0.7 * S), bristle)
        for sgn in (1, -1):
            root = p[HEAD] + fwd * 4 * S + up * 9 * S + side * sgn * 4 * S
            rd.add("cylinder", segment(root, root + (up * 0.9 + fwd * 0.4) * 6 * S, 0.6 * S), bristle)

        # Wings resting along the back
        for i, sgn in ((0, 1), (1, -1)):
            base = p[THX] + up * 12 * S + side * sgn * 5 * S
            tip = p[THX] - fwd * 48 * S + up * 8 * S + side * sgn * 6 * S
            span = float(np.linalg.norm(tip - base))
            rd.add("sphere", trs((base + tip) / 2, frame_from_x(tip - base), (span / 2 + 9 * S, 0.9 * S, 12 * S)),
                   (0.90, 0.94, 0.99, 0.35), P_NONE, 0.12)

        # Shadow on the ground
        self._shadow(rd, dec["p"], 0.22)

    def _draw_extras(self, rd: Renderer, now: float) -> None:
        for b in self.pellets3:
            rd.add("sphere", trs(b["p"], None, (0.035,) * 3), (0.55, 1.0, 0.45), P_NONE, 3.0)
            rd.particle(b["p"], 0.12, (0.4, 1.0, 0.35, 0.45), additive=True)
        for b in self.bombs3:
            rd.add("sphere", trs(b["p"], None, (0.08, 0.08, 0.08)), (0.12, 0.12, 0.14))
            rd.add("cylinder", segment(b["p"] + (0, 0.06, 0), b["p"] + (0.03, 0.13, 0), 0.01), (0.7, 0.6, 0.4))
            if int(now * 12) % 2:
                rd.particle(b["p"] + (0.03, 0.14, 0), 0.04, (1.0, 0.85, 0.3, 1.0), additive=True)
            self._shadow(rd, b["p"], 0.09)
        for s in self.sugars3:
            e = max(0.35, s["left"]) * 0.07
            rd.add("cube", trs(s["p"], rot_y(0.5), (e, e, e)), (0.98, 0.98, 1.0), P_NONE, 0.1)
        for a in self.alcohols3:                             # a shallow pink puddle of fermented fruit juice
            e = max(0.35, a["left"]) * 0.09
            flat = 0.35 if a["landed"] else 0.9
            rd.add("sphere", trs(a["p"], None, (e, e * flat, e)), (0.86, 0.32, 0.55, 0.85), P_NONE, 0.25)
            rd.add("sphere", trs(a["p"] + (e * 0.3, e * flat * 0.5, -e * 0.2), None, (e * 0.3,) * 3),
                   (1.0, 0.72, 0.85, 0.7), P_NONE, 0.4)
            if a["landed"]:
                self._shadow(rd, a["p"], e * 1.3)
        for dec in getattr(self, "decoys3", []):
            self._draw_decoy3d(rd, now, dec)
        self._draw_cva_puffs3d(rd, now)
        if self.spider3 is not None:
            sp = self.spider3
            x, y, z = sp["p"]
            rd.add("cylinder", segment((x, y + 0.05, z), (x, min(RY, SPIDER_TOP), z), 0.002), (0.9, 0.9, 0.92, 0.7))
            body = np.array([x, y, z])
            black = (0.12, 0.11, 0.13)
            rd.add("sphere", trs(body, None, (0.075, 0.06, 0.09)), black)
            rd.add("sphere", trs(body + (0, 0.01, 0.09), None, (0.045, 0.04, 0.045)), black)
            for sgn in (-1, 1):
                for kk in range(4):
                    a = (kk - 1.5) * 0.45
                    wig = 0.015 * math.sin(now * 14 + kk * 1.7 + sgn)
                    knee = body + (sgn * 0.12 * math.cos(a), 0.06 + wig, 0.12 * math.sin(a))
                    foot = body + (sgn * 0.2 * math.cos(a), -0.1 - wig, 0.2 * math.sin(a))
                    rd.add("cylinder", segment(body, knee, 0.008), black)
                    rd.add("cylinder", segment(knee, foot, 0.006), black)
            for dx in (-0.015, 0.015):
                rd.add("sphere", trs(body + (dx, 0.03, 0.13), None, (0.008,) * 3), (1.0, 0.25, 0.25), P_NONE, 1.0)
            self._shadow(rd, body, 0.15)
        for sh in self.shards3:
            R = rot_x(sh["rot"][0]) @ rot_y(sh["rot"][1]) @ rot_z(sh["rot"][2])
            rd.add("cube", trs(sh["p"], R, (sh["size"], sh["size"] * 0.6, sh["size"] * 0.3)),
                   (0.55, 0.36, 0.18) if sh["fly"] else (0.78, 0.9, 1.0, 0.6), P_ICE)
        for a, b, t0 in self.bolts3:
            pts = [a + (b - a) * j / 7 + (np.random.normal(0, 0.05, 3) if 0 < j < 7 else 0) for j in range(8)]
            for j in range(7):
                rd.add("cylinder", segment(pts[j], pts[j + 1], 0.012), (0.85, 0.93, 1.0), P_NONE, 3.0)
                rd.particle(pts[j], 0.1, (0.5, 0.7, 1.0, 0.25), additive=True)
        for pt in self.parts:
            e = (now - pt["t"]) / pt["life"]
            kind = pt["kind"]
            if kind == "flame":
                c = (1.0, 0.95, 0.65) if e < 0.25 else (1.0, 0.6, 0.15) if e < 0.55 else (0.85, 0.25, 0.1)
                rd.particle(pt["p"], 0.03 + 0.12 * e, c + (0.9 * (1 - e ** 2),), additive=True)
            elif kind == "fire":
                rd.particle(pt["p"], pt["size"] * (0.5 + e), (1.0, 0.55 - 0.3 * e, 0.12, 0.8 * (1 - e)), additive=True)
            elif kind == "cleaner":
                rd.particle(pt["p"], 0.04 + 0.16 * e, (0.8, 0.9, 0.98, 0.35 * (1 - e)))
            elif kind == "freeze":
                rd.particle(pt["p"], 0.04 + 0.16 * e, (0.9, 0.97, 1.0, 0.5 * (1 - e)))
            elif kind == "dust":
                rd.particle(pt["p"], pt["size"] * (0.6 + e), (0.75, 0.7, 0.65, 0.45 * (1 - e)))
            elif kind == "muzzle":
                rd.particle(pt["p"], 0.06 + 0.1 * e, (0.7, 1.0, 0.5, 0.9 * (1 - e)), additive=True)
            elif kind == "trail":
                rd.particle(pt["p"], 0.04, (0.4, 1.0, 0.35, 0.4 * (1 - e)), additive=True)
            elif kind == "streak":
                rd.particle(pt["p"], 0.025, (0.9, 0.95, 1.0, 0.35 * (1 - e)))
        if hasattr(self, "laser_state") and (self.torching or self.laser_state.is_active(now)) and TOOLS[self.tool][0] == "laser" and self.report is None:
            tip = self.tool_tip()
            hit_pos = getattr(self.laser_state, "hit_pos", None)
            if hit_pos is None:
                eye, d = self.aim()
                hit_pos = eye + d * 15.0
            else:
                hit_pos = np.array(hit_pos)
            ls = self.laser_state
            is_stim = (ls.mode == "activate")
            beam_col = (1.0, 0.45, 0.15) if is_stim else (0.2, 0.65, 1.0)
            core_col = (1.0, 1.0, 0.85) if is_stim else (0.85, 0.95, 1.0)
            rd.add("cylinder", segment(tip, hit_pos, 0.008), beam_col, P_NONE, 2.5)
            rd.add("cylinder", segment(tip, hit_pos, 0.003), core_col, P_NONE, 3.5)
            rd.particle(hit_pos, 0.08, beam_col + (0.7,), additive=True)
            rd.particle(hit_pos, 0.03, core_col + (0.9,), additive=True)

    def draw_viewmodel(self, rd: Renderer, now: float) -> None:
        """The tool in your hand, in camera space (x right, y up, -z forward)."""
        name = TOOLS[self.tool][0]
        bob = 0.012 * math.sin(self.player.walk_phase * 2)
        base = VIEW_BASE + (0.0, bob, 0.0)
        skin = (0.93, 0.74, 0.6)

        def hand(pos, curl: float, point: bool = False, flick: float = 0.0):
            rd.add("sphere", trs(pos, None, (0.045, 0.028, 0.06)), skin, layer="view")
            for j, dx in enumerate((-0.03, -0.01, 0.01, 0.03)):
                if point and j == 1:
                    ang = -0.2 - 1.2 * flick
                    tip = pos + (dx, 0.01 + 0.08 * math.sin(-ang) * 0.4, -0.05 - 0.07 * math.cos(ang))
                else:
                    tip = pos + (dx, -0.02 * curl, -0.05 - 0.05 * (1 - curl))
                rd.add("cylinder", segment(pos + (dx, 0, -0.04), tip, 0.009), skin, layer="view")
            rd.add("cylinder", segment(pos + (-0.04, 0, -0.01), pos + (-0.07, 0.01, -0.05), 0.01), skin, layer="view")

        if name == "hand":
            hand(base + (0, 0.02, 0), 0.9 if self.fly.grabbed is not None else 0.1)
        elif name == "flick":
            hand(base + (0, 0.02, 0), 0.9, point=True, flick=max(0.0, 1 - (now - self.flick_t) / 0.15))
        elif name == "swatter":
            ph = now - self.swing_t
            ang = -0.9 * (1 - (ph / 0.12)) + 0.2 if ph < 0.12 else 0.2 - 0.2 * min(1.0, (ph - 0.12) / 0.3) if ph < 0.42 else 0.0
            R = rot_x(-ang - 0.3)
            pivot = base + (0, -0.05, 0.05)
            handle_end = pivot + R @ np.array([0.0, 0.32, -0.12])
            rd.add("cylinder", segment(pivot, handle_end, 0.012), (0.5, 0.33, 0.18), layer="view")
            head = handle_end + R @ np.array([0.0, 0.1, -0.03])
            rd.add("cube", trs(head, R, (0.2, 0.22, 0.012)), (0.82, 0.16, 0.18, 0.9), layer="view_blend")
            hand(pivot + (0, 0.02, 0.02), 1.0)
        elif name == "bomb":
            if now - self.throw_t > 0.4:
                rd.add("sphere", trs(base + (0, 0.05, -0.05), None, (0.07, 0.07, 0.07)), (0.12, 0.12, 0.14), layer="view")
            hand(base, 0.7)
        elif name == "torch":
            rd.add("cylinder", segment(base + (0, 0, 0.05), base + (0, 0.06, -0.2), 0.035), (0.75, 0.2, 0.18), layer="view")
            rd.add("cylinder", segment(base + (0, 0.06, -0.2), base + (0, 0.07, -0.26), 0.012), (0.6, 0.6, 0.65), layer="view")
            hand(base + (0, -0.02, 0.04), 1.0)
        elif name in ("cleaner", "freeze"):
            body = (0.82, 0.14, 0.14) if name == "cleaner" else (0.25, 0.5, 0.85)
            rd.add("cylinder", segment(base + (0, -0.05, 0), base + (0, 0.12, -0.02), 0.04), body, layer="view")
            rd.add("cylinder", segment(base + (0, 0.02, -0.01), base + (0, 0.06, -0.012), 0.041), (0.95, 0.95, 0.96), layer="view")
            rd.add("cylinder", segment(base + (0, 0.12, -0.02), base + (0, 0.15, -0.06), 0.012), (0.2, 0.2, 0.22), layer="view")
            hand(base + (0, -0.03, 0.03), 1.0)
        elif name == "zapper":
            rd.add("cylinder", segment(base, base + (0, 0.12, -0.12), 0.014), (0.2, 0.2, 0.25), layer="view")
            ring = base + (0, 0.2, -0.2)
            rd.add("torus", trs(ring, rot_x(-0.7), (0.09, 0.09, 0.09)), (0.95, 0.85, 0.2), P_NONE,
                   2.0 if now - self.zap_ready > -0.3 and now < self.zap_ready else 0.3, layer="view")
            hand(base + (0, 0.0, 0.02), 1.0)
        elif name == "spider":
            rd.add("sphere", trs(base + (0, 0.06, -0.05), None, (0.04, 0.035, 0.05)), (0.12, 0.11, 0.13), layer="view")
            hand(base, 0.6)
        elif name == "sugar":
            if now - self.throw_t > 0.3:
                rd.add("cube", trs(base + (0, 0.05, -0.05), rot_y(0.5), (0.055, 0.055, 0.055)), (0.98, 0.98, 1.0), layer="view")
            hand(base, 0.6)
        elif name == "laser":
            rd.add("cylinder", segment(base + (0, 0, 0.05), base + (0, 0.05, -0.22), 0.025), (0.2, 0.22, 0.26), layer="view")
            rd.add("cylinder", segment(base + (0, 0.05, -0.22), base + (0, 0.06, -0.28), 0.015), (0.7, 0.75, 0.8), layer="view")
            col_led = (1.0, 0.5, 0.1) if (getattr(self, "laser_state", None) and self.laser_state.mode == "activate") else (0.2, 0.7, 1.0)
            rd.add("sphere", trs(base + (0, 0.062, -0.285), None, (0.008, 0.008, 0.008)), col_led, layer="view")
            hand(base + (0, -0.02, 0.04), 1.0)

    # --- 1v1 duel ------------------------------------------------------------------------------------------------------------------
    def toggle_duel(self) -> None:
        self.duel = not self.duel
        self.player_hp, self.player_dead_at = PLAYER_HP, None
        self.pellets3.clear()
        self.note("1V1      on: it has a blaster" if self.duel else "1V1      off")
        self.saved_msg = ("1v1 duel: " + ("ON, the fly can shoot you" if self.duel else "off"), time.perf_counter())
        self.sound.play("click")

    def respawn_player(self) -> None:
        fly = self.fly.p[THX]
        span_x, span_z = min(RX - 0.8, 6.0), min(RZ - 0.8, 6.0)   # outdoors: close enough for it to see you
        corners = [fly[[0, 2]] * (RX > 10) + np.array([x, z]) for x in (-span_x, span_x) for z in (-span_z, span_z)]
        corners = [np.clip(c, (-RX + 0.8, -RZ + 0.8), (RX - 0.8, RZ - 0.8)) for c in corners]
        self.player.pos = max(corners, key=lambda c: float(np.hypot(*(c - fly[[0, 2]])))).astype(float)
        v = fly[[0, 2]] - self.player.pos
        self.player.yaw, self.player.pitch = math.atan2(v[1], v[0]), -0.25
        self.player.eye_h = EYE
        self.player_hp, self.player_dead_at = PLAYER_HP, None
        self.note("RESPAWN  back in the fight")

    def _valence(self) -> tuple[float, float]:
        """How the fly's mushroom body sees you: (fear, liking) read from its real KC -> MBON synapses."""
        mem = self.brain.memory
        return (0.0, 0.0) if mem is None else mem.memory_of("player")

    def _duel_senses(self, now: float) -> None:
        """What the fly sees of you, fed into its real visual neurons: LC10 target tracking on the side you're on
        (which drives the same-side DNa02 steering neurons), and small-object detectors when you're in front (which
        drive DNp35). How strongly it attends to you is set by what it has learned about your smell."""
        fly, br = self.fly, self.brain
        if (fly.dead or self.player_dead_at is not None or fly.frozen_at is not None or fly.wrapped
                or fly.grabbed is not None):
            return
        fear, like = self._valence()
        self.valence += ((like - fear) - self.valence) * 0.05
        if self.frame % 3 == 0 and br.memory is not None and getattr(self.flies[self.focus], "player_scent", False):
            br.memory.observe("player", br.sim.activity.rates())
        head = fly.p[HEAD]
        to = self.player.eye - head
        dist = float(np.linalg.norm(to[[0, 2]]))
        if dist > 7.5:
            return
        to_h = np.array([to[0], 0, to[2]]) / max(dist, 1e-6)
        fwd, _, side = body_axes(fly.yaw)
        ahead, sideness = float(to_h @ fwd), float(to_h @ side)
        attend = float(np.clip(0.7 + 1.5 * self.valence, 0, 1))      # learned fear switches the pursuit off
        if attend <= 0.02:
            return
        near = float(np.clip(1.3 - dist / 7.0, 0.3, 1.0))
        s_side = float(np.clip(abs(sideness) / TRACK_SCALE, 0, 1)) * attend * near
        key = "R" if sideness > 0 else "L"
        if ahead < 0.995 and s_side > 0.03:
            br.poke("track", key, 0.05, recruit=0.15 + 0.45 * s_side)    # short pulses renewed each frame: little lag
        if ahead > 0.985 and dist < 7.0:                                # you are dead ahead: small-object detectors
            s = attend * near * float(np.clip((ahead - 0.985) / 0.012, 0, 1))
            if s > 0.02:
                for sd in ("L", "R") if abs(sideness) < 0.12 else (key,):
                    br.poke("small", sd, 0.05, recruit=0.2 + 0.4 * s)

    def _duel_motor(self, now: float, free: bool, can_fly: bool) -> None:
        """Steering from DNa02/DNa01 right minus left, shooting when DNp35/DNpe052 fire above threshold."""
        fly, br = self.fly, self.brain
        turn = br.hz("turn_r") - br.hz("turn_l")
        self.steer += (turn - self.steer) * 0.35
        grounded = free and now >= fly.stun_until and fly.p[THX, 1] < STAND3 + 0.25
        if grounded and self.player_dead_at is None:
            mag = max(0.0, abs(self.steer) - STEER_DEADZONE_HZ)
            if mag > 0:
                fly.yaw_target = fly.yaw + math.copysign(min(STEER_MAX, mag * STEER_GAIN), self.steer)
        if now < self.fly_reward_until:
            br.poke("reward", None, 0.8)                                # hitting you is rewarding (game rule)
        if now < self.fly_punish_until:
            br.poke("punish", None, 1.0)                                # getting hurt by you is punishing (game rule)
        self.trigger = br.level("fire")
        armed = (fly.grabbed is None and not fly.wrapped and now >= fly.stun_until and fly.frost < 0.5 and fly.melt < 0.5
                 and self.player_dead_at is None)
        if armed and self.trigger > THRESH["fire"] and now >= self.fire_ready:
            self.fire_ready = now + FIRE_COOLDOWN
            self._fly_shoot(now)
            self.note(f"SHOOT    DNp35/DNpe052 x{self.trigger:.1f}")

    def _muzzle(self):
        fly = self.fly
        fwd, up, _ = body_axes(fly.yaw)
        return fly.p[HEAD] + fwd * 0.22 + up * 0.03, fwd

    def _fly_shoot(self, now: float) -> None:
        """The blaster fires along the fly's heading (its brain aims it by turning). Elevation toward your chest is
        automatic, a game rule, since a fly on the floor has no neurons for aiming up at a person."""
        muzzle, fwd = self._muzzle()
        target = self.player.eye - (0, 0.35, 0)
        horiz = max(0.4, float(np.linalg.norm((target - muzzle)[[0, 2]])))
        pitch = math.atan2(target[1] - muzzle[1], horiz)
        v = fwd * math.cos(pitch) + np.array([0, math.sin(pitch), 0])
        v = v + np.random.normal(0, PELLET_SPREAD, 3)
        v = v / np.linalg.norm(v) * PELLET_SPEED
        self.pellets3.append(dict(p=muzzle.copy(), v=v / 60.0, t=now))
        self.duel_stats["shots"] += 1
        self.sound.play("pew", 0.8)
        for _ in range(6):
            self.parts.append(dict(p=muzzle.copy(), v=v / 60 * 0.2 + np.random.normal(0, 0.006, 3), t=now, life=0.15,
                                   kind="muzzle", size=0.05))

    def _pellets3d(self, now: float) -> None:
        pl = self.player
        keep = []
        for b in self.pellets3:
            b["v"][1] -= 0.9 / 3600.0
            b["p"] = b["p"] + b["v"]
            p = b["p"]
            if random.random() < 0.6:
                self.parts.append(dict(p=p.copy(), v=np.zeros(3), t=now, life=0.2, kind="trail", size=0.035))
            hit_player = (self.player_dead_at is None and float(np.hypot(p[0] - pl.pos[0], p[2] - pl.pos[1])) < PLAYER_RADIUS
                          and 0.05 < p[1] < pl.eye_h + 0.15)
            if hit_player:
                self._player_hit(now)
                continue
            outside = abs(p[0]) > RX or abs(p[2]) > RZ or p[1] < 0 or p[1] > RY
            inside_box = any(np.all((p > lo) & (p < hi)) for lo, hi in COLLIDERS)
            if outside or inside_box or now - b["t"] > 2.0:
                self.puff(np.clip(p, (-RX, 0.02, -RZ), (RX, RY, RZ)), 5, 1.5)
                continue
            keep.append(b)
        self.pellets3 = keep
        self.hurt_flash = max(0.0, self.hurt_flash - 1 / 30)

    def _player_hit(self, now: float) -> None:
        self.player_hp = max(0.0, self.player_hp - PELLET_DAMAGE)
        self.hurt_flash = 1.0
        self.shake_until = now + 0.15
        self.sound.play("hurt")
        self.duel_stats["hits"] += 1
        self.fly_reward_until = now + REWARD_PULSE_S                   # its reward dopamine neurons fire
        self.note("HIT YOU  PAM reward dopamine fires")
        if random.random() < 0.5:
            self.popup(self.fly.p[HEAD] + (0, 0.45, 0), random.choice(("GOTCHA!", "PEW PEW!", "TAKE THAT!")), (180, 255, 150))
        if self.player_hp <= 0:
            self.player_dead_at = now
            self.duel_stats["deaths"] += 1
            self.sound.play("death")
            self.note("YOU DIED the fly wins this round")
            self.popup(self.fly.p[HEAD] + (0, 0.55, 0), "FLY WINS!", (255, 120, 90), force=True)
            self.set_look(False)

    def _draw_duel(self, hud, now: float) -> None:
        w, h = 262, 150
        x, y = k2.PLAY_W - w - 12, 72
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (14, 8, 10, 205), card.get_rect(), border_radius=10)
        pygame.draw.rect(card, (170, 60, 60, 220), card.get_rect(), 1, border_radius=10)
        hud.blit(card, (x, y))
        self._text(hud, "1V1", (x + 12, y + 6), (255, 110, 90), self.f_head)
        self._text(hud, "X to end", (x + w - 12, y + 10), k2.LABEL, self.f_small, "topright")
        frac = self.player_hp / PLAYER_HP
        col = k2.S_GOOD if frac > 0.5 else k2.S_WARN if frac > 0.25 else k2.S_CRIT
        pygame.draw.rect(hud, (40, 30, 34), (x + 12, y + 34, w - 24, 14), border_radius=7)
        if frac > 0:
            pygame.draw.rect(hud, col, (x + 12, y + 34, max(10, int((w - 24) * frac)), 14), border_radius=7)
        self._text(hud, f"YOU {self.player_hp:.0f}", ((x + w // 2), y + 41), k2.INK, self.f_small, "center")
        fear, like = self._valence()
        mood, mcol = (("HUNTING YOU", (255, 120, 90)) if self.valence > 0.15 else ("FEARS YOU", (120, 190, 255))
                      if self.valence < -0.2 else ("SIZING YOU UP", k2.TEXT))
        self._text(hud, mood, (x + 12, y + 54), mcol, self.f_bold)
        self._text(hud, f"likes you {like:.2f}   fears you {fear:.2f}", (x + 12, y + 76), k2.LABEL, self.f_small)
        self._text(hud, f"steer DNa02 R-L {self.steer:+5.1f} Hz", (x + 12, y + 94), k2.TEXT, self.f_small)
        tc = k2.AMBER if self.trigger > THRESH["fire"] else k2.TEXT
        self._text(hud, f"trigger DNp35 x{self.trigger:.1f}", (x + 12, y + 110), tc, self.f_small)
        st = self.duel_stats
        self._text(hud, f"shots {st['shots']}  hits {st['hits']}  you died {st['deaths']}x", (x + 12, y + 128), k2.LABEL, self.f_small)
        if self.hurt_flash > 0 and not self.calm_fx:                     # red edges when you get hit, blended over the HUD
            size = (self.view_w, self.hud_h)
            if getattr(self, "_vignette_size", None) != size:
                self._vignette_size = size
                vw, vh = size
                xs = np.minimum(np.arange(vw), np.arange(vw)[::-1])[:, None]
                ys = np.minimum(np.arange(vh), np.arange(vh)[::-1])[None, :]
                edge = np.clip(1 - np.minimum(xs, ys) / 90.0, 0, 1) ** 2
                vig = pygame.Surface(size, pygame.SRCALPHA)
                vig.fill((210, 20, 20, 0))
                pygame.surfarray.pixels_alpha(vig)[:] = (edge * 190).astype(np.uint8)
                self._vignette = vig
            self._vignette.set_alpha(int(255 * self.hurt_flash))
            hud.blit(self._vignette, (0, 0))
        if self.player_dead_at is not None:
            veil = pygame.Surface((self.view_w, self.hud_h), pygame.SRCALPHA)
            veil.fill((60, 0, 0, int(min(140, (now - self.player_dead_at) * 200))))
            hud.blit(veil, (0, 0))
            cx, cy = k2.PLAY_W // 2, self.hud_h // 2 - 40
            t1 = self.f_title.render("YOU DIED", True, (255, 90, 80))
            hud.blit(t1, t1.get_rect(center=(cx, cy)))
            self._text(hud, f"The fly shot you {st['hits']} times with {st['shots']} shots.", (cx, cy + 36), k2.INK, self.f_text, "midtop")
            self._text(hud, f"Its mushroom body now likes you {like:.2f} and fears you {fear:.2f}.", (cx, cy + 60), k2.TEXT, self.f_text, "midtop")
            self._text(hud, "press R to respawn", (cx, cy + 92), k2.AMBER, self.f_bold, "midtop")

    # --- HUD --------------------------------------------------------------------------------------------------------------------------
    def draw_hud3d(self, now: float, project) -> None:
        hud = self.screen
        hud.fill((0, 0, 0, 0))
        if getattr(self, "photo_mode", False):
            if not getattr(self, "photo_hide_ui", False):
                self._draw_photo_viewfinder(hud)
            return
        for pos, text, t0, color, source in self.popups3:
            sp = project(pos)
            if sp is None:
                continue
            e = (now - t0) / 0.9
            txt = self.f_big.render(text, True, color)
            shd = self.f_big.render(text, True, (120, 16, 22))
            a = int(255 * (1 - e ** 3))
            txt.set_alpha(a)
            shd.set_alpha(a)
            x, y = sp[0] - txt.get_width() / 2, sp[1] - 50 * e - 20
            hud.blit(shd, (x + 3, y + 3))
            hud.blit(txt, (x, y))
            if source and self.cfg.tags_on():
                k2.draw_source_chip(hud, (sp[0], y + txt.get_height()), source, self.f_small, alpha=a)
        if not self._overlay_open() and not (self.cfg["brain.autopilot"] and self.cfg["brain.autopilot_hide_hud"]):
            cx, cy = k2.PLAY_W // 2, self.hud_h // 2
            eye, d = self.aim()
            _, i, _ = self._nearest_fly3d(eye, d, REACH, 0.25)
            col = (255, 190, 90) if i is not None else (240, 240, 240)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pygame.draw.line(hud, (0, 0, 0, 120), (cx + dx * 5 + 1, cy + dy * 5 + 1), (cx + dx * 13 + 1, cy + dy * 13 + 1), 3)
                pygame.draw.line(hud, col, (cx + dx * 5, cy + dy * 5), (cx + dx * 13, cy + dy * 13), 2)
            if hasattr(self, "laser_state") and (self.torching or self.laser_state.is_active(now)) and TOOLS[self.tool][0] == "laser":
                ls = self.laser_state
                col_b = (255, 140, 50) if ls.mode == "activate" else (60, 190, 255)
                badge = f"LASER: {ls.target_type} ({'STIM' if ls.mode == 'activate' else 'SILENCE'})"
                self._text(hud, badge, (cx + 18, cy - 18), col_b, self.f_small)
        self._draw_toolbar(hud)
        if self.pad.wheel_open and not self._overlay_open():
            self._draw_tool_wheel(hud)
        self._draw_hud(hud, now)
        if self.duel:
            self._draw_duel(hud, now)
        if not self.look and not self._overlay_open() and self.player_dead_at is None and not self.menu.open and not (self.cfg["brain.autopilot"] and self.cfg["brain.autopilot_hide_hud"]):
            pad = " or pull the trigger" if self.pad.connected else ""
            msg = self.f_bold.render(f"click the room (or press {self.cfg.keys['free_mouse'].title()}{pad}) to look around"
                                     "   ·   Esc: menu", True, INK_ON)
            box = msg.get_rect(center=(k2.PLAY_W // 2, self.hud_h // 2 + 60)).inflate(24, 12)
            pygame.draw.rect(hud, (8, 10, 16, 200), box, border_radius=8)
            hud.blit(msg, msg.get_rect(center=box.center))
        if self.report is not None:
            self._draw_autopsy(hud, now)
        elif self.big_view:
            self._draw_big_view(hud)
        if self.surgery_open:
            self._draw_surgery(hud)
        if self.training_open:
            self._draw_training(hud)
        if self.help_open:
            self._draw_help(hud)
        if PANEL_MODES[self.panel_mode][0] != "hidden":
            self._draw_brain(now)
        else:
            self.view_rect = pygame.Rect(0, 0, 0, 0)
        if self.challenge is not None:
            self.challenge.draw(hud, now, self.mouse_logical)
        self.draw_science_card(hud, now)
        self.draw_time_indicator(hud, k2.PLAY_W // 2, 92)
        self.draw_recording(hud, k2.PLAY_W // 2, 130)
        if self.menu.open:
            self.menu.draw(hud, self.mouse_logical, now)

    def _draw_help(self, surf) -> None:
        panel = pygame.Rect(135, 80, 620, 64 + 30 * len(HELP3D))
        pygame.draw.rect(surf, (18, 21, 28), panel, border_radius=16)
        pygame.draw.rect(surf, k2.BORDER, panel, 1, border_radius=16)
        self._text(surf, "CONTROLS", (panel.x + 24, panel.y + 16), k2.INK, self.f_head)
        for kk, (key, what) in enumerate(HELP3D):
            y = panel.y + 54 + kk * 30
            self._text(surf, key, (panel.x + 30, y), k2.AMBER, self.f_bold)
            self._text(surf, what, (panel.x + 190, y), k2.TEXT, self.f_text)

    # --- input ------------------------------------------------------------------------------------------------------------------------
    def set_look(self, on: bool) -> None:
        self.look = on
        self.quit_armed = False
        pygame.event.set_grab(on)
        pygame.mouse.set_visible(not on)
        pygame.mouse.get_rel()

    def handle(self, ev, now: float) -> bool:
        return self.handle3d(ev, now, lambda p: p)

    def handle3d(self, ev, now: float, to_logical) -> bool:
        """3D input. Returns False to quit. Keys go through the rebindable actions in config.py."""
        if hasattr(ev, "pos"):
            ev = pygame.event.Event(ev.type, {**ev.dict, "pos": to_logical(ev.pos)})
            self.mouse_logical = ev.pos
        if self.search_key(ev):
            return True
        if self.menu_first(ev, self.mouse_logical):
            return not self.want_quit
        if ev.type == pygame.KEYDOWN:
            if self.big_view and ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_0, pygame.K_KP1, pygame.K_KP2, pygame.K_KP3, pygame.K_KP0):
                return k2.Game.handle(self, ev, now)
            if ev.key in k2.TOOL_KEYS:
                self.tool = k2.TOOL_KEYS.index(ev.key)
                return True
            if (ev.key == pygame.K_r and (ev.mod & pygame.KMOD_SHIFT)) or getattr(ev, "unicode", "") == "R":
                self.toggle_video_recording()
                return True
            action = self.cfg.action_for(pygame.key.name(ev.key))
            if action == "photo_mode" or ev.key == pygame.K_F10:
                self.toggle_photo_mode()
                return True
            if self.photo_mode:
                if ev.key == pygame.K_LEFTBRACKET:
                    self.photo_fov = max(20.0, self.photo_fov - 5.0)
                    self.saved_msg = (f"FOV: {self.photo_fov:.0f}°", time.perf_counter())
                    return True
                elif ev.key == pygame.K_RIGHTBRACKET:
                    self.photo_fov = min(120.0, self.photo_fov + 5.0)
                    self.saved_msg = (f"FOV: {self.photo_fov:.0f}°", time.perf_counter())
                    return True
                elif ev.key == pygame.K_COMMA:
                    self.photo_dof = max(0.0, round(self.photo_dof - 0.05, 2))
                    self.saved_msg = (f"DOF blur: {self.photo_dof:.2f}", time.perf_counter())
                    return True
                elif ev.key == pygame.K_PERIOD:
                    self.photo_dof = min(1.0, round(self.photo_dof + 0.05, 2))
                    self.saved_msg = (f"DOF blur: {self.photo_dof:.2f}", time.perf_counter())
                    return True
                elif ev.key == pygame.K_k:
                    self.photo_focus = max(0.2, round(self.photo_focus - 0.2, 2))
                    self.saved_msg = (f"Focus: {self.photo_focus:.1f}m", time.perf_counter())
                    return True
                elif ev.key == pygame.K_l:
                    self.photo_focus = min(15.0, round(self.photo_focus + 0.2, 2))
                    self.saved_msg = (f"Focus: {self.photo_focus:.1f}m", time.perf_counter())
                    return True
                elif ev.key == pygame.K_f:
                    self.photo_focus = self.nearest_fly_dist()
                    self.saved_msg = (f"Auto-focus: {self.photo_focus:.1f}m", time.perf_counter())
                    return True
                elif ev.key == pygame.K_h:
                    self.photo_hide_ui = not self.photo_hide_ui
                    return True
                elif ev.key in (pygame.K_F12, pygame.K_RETURN, pygame.K_SPACE):
                    self.take_photo()
                    return True
            if action == "free_mouse":
                self.set_look(not self.look)
            elif action in ("forward", "back", "left", "right", "sprint", "crouch") or ev.key in (pygame.K_c, pygame.K_SPACE):
                pass
            elif action == "panel":
                names = [m for m, _ in PANEL_MODES]
                self.set_setting("graphics.panel_mode", names[(self.panel_mode + 1) % len(names)])
                self.saved_msg = (f"brain panel: {PANEL_MODES[self.panel_mode][0]}", time.perf_counter())
            elif action == "menu_size":
                self.set_setting("graphics.menu_size", UI_MODES[(UI_MODES.index(self.ui_mode) + 1) % len(UI_MODES)])
                self.saved_msg = (f"menu size: {self.ui_mode}", time.perf_counter())
            elif action == "reset" and self.player_dead_at is not None:
                self.respawn_player()
            elif action == "duel":
                self.toggle_duel()
            elif action is not None:
                self.do_action(action, now)
            return True
        if ev.type == pygame.MOUSEWHEEL:
            if not self.look and self.big_view:
                mpos = ev.pos if hasattr(ev, "pos") else self.mouse_logical
                if getattr(self, "big_rect", None) and self.big_rect.collidepoint(mpos):
                    self.view.zoom_by(1.15 if ev.y > 0 else 0.87)
                    return True
            if self.look:
                self.tool = (self.tool - ev.y) % len(TOOLS)
                return True
        if ev.type == pygame.MOUSEMOTION and not self.look and self.big_view and getattr(self, "big_drag", None):
            return k2.Game.handle(self, ev, now)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 2, 3):
            if self.photo_mode and ev.button == 1:
                self.take_photo()
                return True
            if self.look:
                if ev.button == 3 and TOOLS[self.tool][0] == "hand":
                    dec = getattr(self, "grabbed_decoy3", None)
                    if dec is None:
                        eye, d = self.aim()
                        dec, _ = self._nearest_decoy3d(eye, d, GRAB_REACH, 0.4)
                    if dec is not None:
                        self.remove_decoy(dec)
                        return True
                if not getattr(self, "autopilot", False) and not self.photo_mode:
                    self.use_tool3d(now)
                return True
            pos = ev.pos
            if self.science_card is not None and self.science_rect().collidepoint(pos):
                self.science_card = None
                return True
            if self.challenge is not None and any(b.rect.collidepoint(pos) for b in self.challenge.buttons):
                return self.challenge.click(pos)
            if self.challenge is not None and self.challenge.overlay:
                return self.challenge.click(pos)
            if self.player_dead_at is None and not self._overlay_open() and pos[0] < k2.PLAY_W and not any(r.collidepoint(pos) for r in getattr(self, "tool_rects", [])):
                self.set_look(True)
                return True
            if self.surgery_open or self.help_open or self.report is not None or self.big_view or self.training_open:
                return k2.Game.handle(self, ev, now)
            for kk, r in enumerate(getattr(self, "tool_rects", [])):
                if r.collidepoint(pos):
                    self.tool = kk
                    return True
            if self.view_rect.collidepoint(pos):
                self.big_view = True
            return True
        if ev.type == pygame.MOUSEBUTTONUP:
            if not self.look and self.big_view and getattr(self, "big_drag", None):
                return k2.Game.handle(self, ev, now)
            if getattr(self, "grabbed_decoy3", None) is not None:
                eye, d = self.aim()
                self.grabbed_decoy3["v"] = d * 0.12 + np.array([0.0, 0.02, 0.0])
                self.grabbed_decoy3["landed"] = False
                self.grabbed_decoy3 = None
            if not self.fly.wrapped:
                self.fly.grabbed = None
            self.torching = False
            if hasattr(self, "laser_state"):
                self.laser_state.trigger_release()
            return True
        return True

    def capture(self) -> None:                     # frames are captured by the 3D app (see App.capture)
        pass

    def save_png(self) -> None:
        self.want_png = True


INK_ON = (240, 243, 248)


class GLUnavailable(RuntimeError):
    """No OpenGL 3.3 core context (old GPU/driver, remote session, software GL too old): the caller falls back to 2D."""


class App:
    """Window, GL context, frame composition, fullscreen and scaling."""

    def __init__(self, fullscreen: bool, vsync: bool = False):
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        desk = pygame.display.get_desktop_sizes()[0] if pygame.display.get_desktop_sizes() else (k2.W, k2.H)
        fs = bool(fullscreen or desk[0] < k2.W or desk[1] < k2.H + 60)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if fs else pygame.RESIZABLE)
        size = (0, 0) if fs else (k2.W, k2.H)
        try:
            try:
                pygame.display.set_mode(size, flags, vsync=1 if vsync else 0)
            except pygame.error:                         # the driver refused vsync: run without it
                pygame.display.set_mode(size, flags)
            self.ctx = moderngl.create_context()
        except Exception as e:
            raise GLUnavailable(f"{type(e).__name__}: {e}") from e
        crash.record_gl(self.ctx)
        if self.ctx.version_code < 330:
            raise GLUnavailable(f"OpenGL {self.ctx.version_code / 100:.1f} ({crash.info.get('gl_renderer', '?')})")
        k2.log.info("OpenGL %s on %s", crash.info.get("gl_version"), crash.info.get("gl_renderer"))
        pygame.display.set_caption("Kick the Fly")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.rd = Renderer(self.ctx)
        self.hud_tex = None
        self.hud_size = None
        self.scene_size = None
        self.small = self.ctx.simple_framebuffer(k2.GIF_SIZE)

    def layout(self, game=None, size=None):
        Wn, Hn = size or pygame.display.get_window_size()
        ui, panel = (game.ui_mode, game.panel_mode) if game is not None else ("crisp", 0)
        ui_scale = game.cfg["graphics.ui_scale"] if game is not None else 1.0
        s, hud_w, hud_h, play_w, view_w = compute_layout(Wn, Hn, ui, panel, ui_scale)
        return Wn, Hn, s, hud_w, hud_h, play_w, view_w

    def hud_texture(self, hud_w: int, hud_h: int, s: float) -> moderngl.Texture:
        if self.hud_size != (hud_w, hud_h):              # only when the window or menu size changes
            self.hud_size = (hud_w, hud_h)
            if self.hud_tex is not None:
                self.hud_tex.release()                  # GL objects are not garbage collected: free them explicitly
            self.hud_tex = self.ctx.texture((hud_w, hud_h), 4)
        self.hud_filter = (moderngl.NEAREST, moderngl.NEAREST) if abs(s - round(s)) < 1e-6 else (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_tex.filter = self.hud_filter
        return self.hud_tex

    def _ensure_scene(self, w: int, h: int):
        if self.scene_size == (w, h):
            return
        self.scene_size = (w, h)
        for old in (getattr(self, "ms", None), getattr(self, "scene", None)):   # free the previous size's buffers
            if old is not None:
                for att in list(old.color_attachments) + ([old.depth_attachment] if old.depth_attachment else []):
                    att.release()
                old.release()
        samples = min(4, self.ctx.max_samples)
        try:
            self.ms = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((w, h), 4, samples=samples)],
                                           depth_attachment=self.ctx.depth_renderbuffer((w, h), samples=samples))
        except Exception:
            self.ms = None
        self.scene_tex = self.ctx.texture((w, h), 4)
        self.scene_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.scene = self.ctx.framebuffer(color_attachments=[self.scene_tex], depth_attachment=self.ctx.depth_texture((w, h)))

    def render(self, game: Game3D, now: float, target=None, size=None):
        ctx, rd = self.ctx, self.rd
        Wn, Hn, s, hud_w, hud_h, play_w, view_w = self.layout(game, size)
        k2.W, k2.H, k2.PLAY_W, k2.FLOOR = hud_w, hud_h, play_w, hud_h - 120   # the shared HUD code reads these
        game.view_w, game.hud_h = view_w, hud_h
        if game.screen.get_size() != (hud_w, hud_h):
            game.screen = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        rs = game.cfg["graphics.resolution_scale"]              # Settings > resolution scale: 3D drawn smaller, stretched
        vw, vh = max(1, int(round(view_w * s * rs))), max(1, int(round(Hn * rs)))
        self._ensure_scene(vw, vh)
        fb = self.ms or self.scene
        fb.use()
        ctx.viewport = (0, 0, vw, vh)
        lights, clear_col, far = scene_setup(game)
        ctx.clear(*clear_col, depth=1.0)
        cam = game.free_cam if (getattr(game, "photo_mode", False) and getattr(game, "free_cam", None) is not None) else game.player
        eye = cam.eye
        shake = np.random.uniform(-0.01, 0.01, 3) if now < game.shake_until and not game.calm_fx else 0
        f, r, u = cam.basis()
        view = look_at(eye + shake, eye + shake + f)
        cam_fov = getattr(game, "photo_fov", float(game.cfg["controls.fov"])) if getattr(game, "photo_mode", False) else float(game.cfg["controls.fov"])
        proj = perspective(math.radians(cam_fov), vw / vh, 0.03, far)
        # when the 3D view runs under a see-through panel, shift the lens so the crosshair and your hand stay centered
        # on the open part of the screen
        lens = np.eye(4)
        if not getattr(game, "photo_mode", False):
            lens[0, 3] = play_w / view_w - 1                  # shift in clip x by w: moves the image center left
        proj = lens @ proj
        rd.clear()
        game.draw_world(rd, now)
        rd.set_scene(view, proj, eye, lights, now)
        rd.draw_layer("opaque")
        rd.draw_layer("blend")
        rd.draw_particles()
        if not getattr(game, "photo_mode", False):
            # the tool in your hand: squeezed into the front 10% of the depth range so it never sinks into walls,
            # lit by the same lights moved into camera space
            game.draw_viewmodel(rd, now)
            Rv = view[:3, :3]
            cam_lights = dict(lights)
            cam_lights["u_sun_dir"] = Rv @ lights["u_sun_dir"]
            cam_lights["u_lp0"] = (view @ np.append(lights["u_lp0"], 1))[:3]
            cam_lights["u_lp1"] = (view @ np.append(lights["u_lp1"], 1))[:3]
            squeeze = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.1, -0.9], [0, 0, 0, 1.0]])
            rd.set_scene(np.eye(4), squeeze @ lens @ perspective(math.radians(VIEWMODEL_FOV), vw / vh, 0.01, 5.0), (0, 0, 0),
                         cam_lights, now)
            rd.draw_layer("view")
            rd.draw_layer("view_blend")
        if self.ms is not None:
            ctx.copy_framebuffer(self.scene, self.ms)

        def project(p):
            c = proj @ view @ np.append(p, 1.0)
            if c[3] <= 0.05:
                return None
            ndc = c[:3] / c[3]
            if abs(ndc[0]) > 1.2 or abs(ndc[1]) > 1.2:
                return None
            return ((ndc[0] + 1) / 2 * view_w, (1 - ndc[1]) / 2 * hud_h)

        game.draw_hud3d(now, project)
        tex = self.hud_texture(hud_w, hud_h, s)
        tex.write(pygame.image.tobytes(game.screen, "RGBA", False))
        out = target or ctx.screen
        out.use()
        ctx.viewport = (0, 0, Wn, Hn)
        out.clear(0, 0, 0, 1)
        dof = float(getattr(game, "photo_dof", 0.0)) if getattr(game, "photo_mode", False) else 0.0
        focus = float(getattr(game, "photo_focus", 1.8))
        if dof > 0:
            rd.blit_dof(self.scene_tex, self.scene.depth_attachment, (0, 0, round(view_w * s), Hn), (Wn, Hn),
                        focus=focus, dof=dof, flip=False, blend=False)
        else:
            rd.blit_texture(self.scene_tex, (0, 0, round(view_w * s), Hn), (Wn, Hn), flip=False, blend=False)
        if not (getattr(game, "photo_mode", False) and getattr(game, "photo_hide_ui", False)):
            rd.blit_texture(tex, (0, 0, hud_w * s, hud_h * s), (Wn, Hn), flip=True, blend=True)
        self.view_frac = view_w / hud_w
        return (Wn, Hn, s, hud_w, hud_h, play_w, view_w)

    def capture(self, game: Game3D) -> None:
        """Downscale the finished frame into the GIF buffer (15 fps)."""
        w, h = k2.GIF_SIZE
        self.small.use()
        self.ctx.viewport = (0, 0, w, h)
        self.small.clear(0, 0, 0, 1)
        self.scene_tex.build_mipmaps()
        self.scene_tex.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
        self.hud_tex.build_mipmaps()
        self.hud_tex.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
        self.rd.blit_texture(self.scene_tex, (0, 0, w * getattr(self, "view_frac", 0.7), h), (w, h), flip=False, blend=False)
        self.rd.blit_texture(self.hud_tex, (0, 0, w, h), (w, h), flip=True, blend=True)
        self.scene_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.hud_tex.filter = getattr(self, "hud_filter", (moderngl.LINEAR, moderngl.LINEAR))
        data = self.small.read(components=3)
        rows = np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1]
        game.frames.append(rows.tobytes())

    def capture_timelapse(self, game: Game3D) -> None:
        """Capture frame for time-lapse export."""
        if not getattr(game, "timelapse_recording", False):
            return
        target = game.cfg.get("graphics.timelapse_target", "brain")
        w, h = getattr(game, "timelapse_size", (600, 340))
        if target == "brain" and hasattr(game, "_view_surface"):
            surf = game._view_surface("big")
            scaled = pygame.transform.smoothscale(surf, (w, h))
            game.timelapse_frames.append(pygame.image.tobytes(scaled, "RGB"))
        else:
            gw, gh = k2.GIF_SIZE
            self.small.use()
            self.ctx.viewport = (0, 0, gw, gh)
            self.small.clear(0, 0, 0, 1)
            self.scene_tex.build_mipmaps()
            self.scene_tex.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
            self.hud_tex.build_mipmaps()
            self.hud_tex.filter = moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR
            self.rd.blit_texture(self.scene_tex, (0, 0, gw * getattr(self, "view_frac", 0.7), gh), (gw, gh), flip=False, blend=False)
            self.rd.blit_texture(self.hud_tex, (0, 0, gw, gh), (gw, gh), flip=True, blend=True)
            self.scene_tex.filter = moderngl.LINEAR, moderngl.LINEAR
            self.hud_tex.filter = getattr(self, "hud_filter", (moderngl.LINEAR, moderngl.LINEAR))
            data = self.small.read(components=3)
            rows = np.frombuffer(data, np.uint8).reshape(gh, gw, 3)[::-1]
            surf = pygame.image.frombuffer(rows.tobytes(), (gw, gh), "RGB")
            scaled = pygame.transform.smoothscale(surf, (w, h))
            game.timelapse_frames.append(pygame.image.tobytes(scaled, "RGB"))

    def capture_video_frame(self, game: Game3D, lay) -> None:
        """Hand the recorder the window's pixels, read back from the GPU only when the video needs a frame."""
        rec = getattr(game, "video_recorder", None)
        if rec is None or not rec.is_recording:
            return

        def grab():
            w, h = lay[0] - lay[0] % 2, lay[1] - lay[1] % 2
            self.ctx.screen.use()
            data = self.ctx.screen.read(viewport=(0, 0, w, h), components=3)
            return np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1].tobytes(), (w, h)

        rec.capture(grab)

    def screenshot(self, path: Path, lay, game: Game3D | None = None, now: float = 0.0,
                   scale: int | None = None, clean: bool | None = None) -> None:
        Wn, Hn = lay[0], lay[1]
        if game is not None:
            if scale is None:
                scale = int(game.cfg.get("graphics.photo_scale", 2))
            if clean is None:
                clean = bool(game.cfg.get("graphics.clean_capture", True))
        else:
            scale = scale or 1
            clean = False if clean is None else clean

        if scale <= 1 and not clean:
            self.ctx.screen.use()
            data = self.ctx.screen.read(viewport=(0, 0, Wn, Hn), components=3)
            img = pygame.image.frombytes(data, (Wn, Hn), "RGB", True)
            k2.save_image(img, path)
            return

        tw, th = int(Wn * scale), int(Hn * scale)
        tex = self.ctx.texture((tw, th), 4)
        tex.filter = moderngl.LINEAR, moderngl.LINEAR
        depth_tex = self.ctx.depth_texture((tw, th))
        fb = self.ctx.framebuffer(color_attachments=[tex], depth_attachment=depth_tex)
        fb.use()
        self.ctx.viewport = (0, 0, tw, th)
        lights, clear_col, far = scene_setup(game)
        self.ctx.clear(*clear_col, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        cam = (game.free_cam if (getattr(game, "photo_mode", False) and getattr(game, "free_cam", None) is not None) else game.player) if game else None
        if cam is not None:
            eye = cam.eye
            f, r, u = cam.basis()
            view = look_at(eye, eye + f)
            cam_fov = getattr(game, "photo_fov", float(game.cfg["controls.fov"])) if getattr(game, "photo_mode", False) else float(game.cfg["controls.fov"])
        else:
            eye = (0, 0, 0)
            view = np.eye(4)
            cam_fov = 70.0
        proj = perspective(math.radians(cam_fov), tw / th, 0.03, far)
        self.rd.clear()
        if game:
            game.draw_world(self.rd, now)
        self.rd.set_scene(view, proj, eye, lights, now)
        self.rd.draw_layer("opaque")
        self.rd.draw_layer("blend")
        self.rd.draw_particles()

        dof = float(getattr(game, "photo_dof", 0.0)) if game else 0.0
        focus = float(getattr(game, "photo_focus", 1.8)) if game else 1.8
        if dof > 0:
            out_tex = self.ctx.texture((tw, th), 4)
            out_fb = self.ctx.framebuffer(color_attachments=[out_tex])
            out_fb.use()
            self.ctx.viewport = (0, 0, tw, th)
            self.ctx.clear(0, 0, 0, 1)
            self.rd.blit_dof(tex, depth_tex, (0, 0, tw, th), (tw, th), focus=focus, dof=dof, flip=False, blend=False)
            data = out_fb.read(viewport=(0, 0, tw, th), components=3)
            out_fb.release()
            out_tex.release()
        else:
            data = fb.read(viewport=(0, 0, tw, th), components=3)

        fb.release()
        tex.release()
        depth_tex.release()

        img = pygame.image.frombytes(data, (tw, th), "RGB", True)
        k2.save_image(img, path)


def run(smoke: float = 0.0, shot: str | None = None, fullscreen: bool = False, seed: int = 0, cfg=None,
        flies: int = 1, record_video: str | None = None, script=None) -> int:
    """The 3D game. script(game, app, t, lay), if given, is called once per frame after the world updates, with t the
    seconds since the game started; returning False ends the run. tools/make_screenshots.py drives its scenes with it."""
    from kickthefly.core import config
    cfg = cfg if cfg is not None else config.Config(None)
    app = App(fullscreen, vsync=cfg["graphics.vsync"])
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui,consolas", 22)
    state: dict = {"stage": "starting", "seed": seed, "backend": cfg["brain.backend"], "dtype": cfg["brain.dtype"]}
    threading.Thread(target=k2.load_brain, args=(state,), daemon=True).start()
    t0 = time.perf_counter()
    splash = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    while "brain" not in state:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return 0
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
        splash.fill((9, 11, 15, 255))
        msg = state.get("error") or f"{state['stage']}{'.' * (int((time.perf_counter() - t0) * 3) % 4)}"
        img = font.render(msg, True, k2.RED if "error" in state else k2.TEXT)
        splash.blit(img, img.get_rect(center=(k2.W // 2, k2.H // 2)))
        Wn, Hn = pygame.display.get_window_size()
        s = min(Wn / k2.W, Hn / k2.H)
        tex = app.hud_texture(k2.W, k2.H, 1.5)
        tex.write(pygame.image.tobytes(splash, "RGBA", False))
        app.ctx.screen.use()
        app.ctx.viewport = (0, 0, Wn, Hn)
        app.ctx.screen.clear(0, 0, 0, 1)
        app.rd.blit_texture(tex, ((Wn - k2.W * s) / 2, (Hn - k2.H * s) / 2, k2.W * s, k2.H * s), (Wn, Hn), flip=True)
        pygame.display.flip()
        clock.tick(30)

    brain = state["brain"]
    brain.start()
    hud = pygame.Surface((k2.W, k2.H), pygame.SRCALPHA)
    game = Game3D(hud, brain, state["view"], state.get("graph"), state.get("weights"), cfg=cfg)
    if cfg["brain.autopilot"]:
        game.big_view = True
    if record_video:
        game.toggle_video_recording(None if record_video == "default" else record_video)
    game.want_png = False
    running = True
    t_game = last = time.perf_counter()
    to_spawn = max(0, int(flies) - 1)              # --flies N: spawn the rest once the game runs (benchmarks)
    fps_log: list[float] = []
    lay = app.layout(game)

    def to_logical(pos):
        Wn, Hn = pygame.display.get_window_size()
        s = app.layout(game, (Wn, Hn))[2]
        return (int(pos[0] / s), int(pos[1] / s))

    while running:
        real = time.perf_counter()
        dt = min(0.05, real - last)
        last = real
        ticks = game.clock.frame(real)
        game.mouse_logical = to_logical(pygame.mouse.get_pos())
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and (ev.key == pygame.K_F11 or (ev.key == pygame.K_RETURN and ev.mod & pygame.KMOD_ALT)):
                game.toggle_fullscreen()
                lay = app.layout(game)
                continue
            game.pad.device_event(ev)
            running = game.handle3d(ev, game.clock.now, to_logical) and running
        pad_keys, pad_rel = game.pad_tick(dt, game.clock.now)
        if game.look and (game._overlay_open() or game.menu.open):
            game.set_look(False)
        rel = pygame.mouse.get_rel() if game.look else (0, 0)
        rel = (rel[0] + pad_rel[0], rel[1] + pad_rel[1])
        keys = game.held(pygame.key.get_pressed())
        for k_, v_ in pad_keys.items():
            keys[k_] = keys[k_] or v_
        if to_spawn and not game._spawning and len(game.flies) < getattr(game, "max_flies", k2.MAX_FLIES):
            game.spawn_fly()
            to_spawn -= 1
        game.sync_time()
        game.update_player(dt, keys, rel)
        for tick_dt in ticks:
            game.clock.now += tick_dt
            game.update3d(game.clock.now, tick_dt, keys, rel)
        now = game.clock.now
        if script is not None and script(game, app, real - t_game, lay) is False:
            running = False
        lay = app.render(game, now)
        if ticks and game.frame % 4 == 0:
            app.capture(game)
        if ticks and getattr(game, "timelapse_recording", False) and game.frame % 2 == 0:
            app.capture_timelapse(game)
        if getattr(game, "video_recorder", None) and game.video_recorder.is_recording:
            app.capture_video_frame(game, lay)
        if game.want_png:
            game.want_png = False
            path = game.media_path("png")
            app.ctx.screen.use()
            app.screenshot(path, lay, game=game, now=now)
            game.saved_note(path)
        pygame.display.flip()
        if smoke and real - t_game > smoke / 2 and not to_spawn:
            fps_log.append(clock.get_fps())                # the second half of the run: after spawning and warm-up
        if smoke and real - t_game > smoke:
            try:
                from PIL import Image  # noqa: F401
                gif = "gif ok"
            except ImportError:
                gif = "no gif"
            rates = [sl.brain.steps_per_s for sl in game.flies]
            fps = float(np.mean(fps_log)) if fps_log else clock.get_fps()
            status = (f"smoke ok 3d: {brain.n:,} neurons, {brain.steps_per_s:.0f} steps/s, {fps:.0f} fps, "
                      f"sound {game.sound.ok}, {gif}, arena {k2.ARENAS[game.arena_i]}, {len(game.flies)} flies, "
                      f"sim/real {min(rates) / 200:.2f}x (slowest fly) {np.mean(rates) / 200:.2f}x (mean)")
            print(status)
            if shot:
                app.ctx.screen.use()
                app.screenshot(Path(shot), lay)
                with open(shot + ".txt", "w") as fh:
                    fh.write(status)
            break
        clock.tick(cfg["graphics.fps_cap"])
    k2.shutdown(game)
    return 0
