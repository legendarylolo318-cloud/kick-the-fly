"""Larva body physics, segmented crawler mechanics, and rendering.

GAME RULE physics:
A segmented caterpillar-like crawler (10 segments: head, T1-T3, A1-A6).
Locomotion:
- Forward crawling: peristaltic wave passing from posterior to anterior segments,
  modulated by descending motor activity (DN-VNC / pre-DN-VNC).
- Turning / Head-casting: lateral bending of anterior segments (head through T3).
- Rolling escape: fast lateral corkscrew / C-curling roll triggered by Goro command
  neurons (_telegoro-1, DN-VNC) in response to noxious stimulation (Ohyama et al. 2015).
- No flying: the larva has no wings and stays grounded.

Arenas:
- Larva-appropriate arenas: Room, Flypaper, Lamp, Thermo, Pool.
- Restricted arenas: Open field, Orchard, Fan, Escape room (gated with clear messages).
"""
from __future__ import annotations

import math
import random
from typing import Any
import numpy as np
import pygame
from pygame import gfxdraw

LARVA_SEGMENTS = 10
LARVA_HEAD = 0
LARVA_TAIL = LARVA_SEGMENTS - 1
LARVA_SEG_RADIUS = np.array([7.0, 9.0, 11.0, 12.0, 13.0, 13.0, 12.0, 11.0, 9.0, 6.0], dtype=np.float32)
LARVA_SEG_SPACING = 8.0
LARVA_TOTAL_LEN = float((LARVA_SEGMENTS - 1) * LARVA_SEG_SPACING)
LARVA_MAX_HEALTH = 100.0

# Appropriate arenas for larva
LARVA_ALLOWED_ARENAS = ("room", "flypaper", "lamp", "thermo", "pool")
LARVA_RESTRICTED_REASONS = {
    "field": "The open field requires adult flight and wind navigation.",
    "orchard": "The orchard requires flying between tree canopies to reach fruit.",
    "fan": "Strong fan wind sweeps larvae away; larvae cannot fly through headwinds.",
    "escaperoom": "The escape room puzzle requires adult jumping and aerial navigation.",
}


def is_arena_allowed_for_larva(arena: str) -> tuple[bool, str]:
    """Check if an arena is appropriate for larva mode."""
    arena = arena.lower()
    if arena in LARVA_ALLOWED_ARENAS:
        return True, ""
    reason = LARVA_RESTRICTED_REASONS.get(
        arena, f"The '{arena}' arena is not adapted for larval crawling."
    )
    return False, reason


class LarvaBody:
    """Segmented crawler ragdoll body for Drosophila larva."""

    def __init__(self, x: float, floor_y: float = 620.0):
        self.floor_y = floor_y
        self.facing = 1  # 1 = right, -1 = left
        self.n_segments = LARVA_SEGMENTS

        # Segment positions: arranged horizontally at rest
        self.p = np.zeros((LARVA_SEGMENTS, 2), dtype=np.float32)
        for i in range(LARVA_SEGMENTS):
            self.p[i] = [x - i * LARVA_SEG_SPACING * self.facing, floor_y - LARVA_SEG_RADIUS[i]]
        self.prev = self.p.copy()

        # States & timers
        self.health = LARVA_MAX_HEALTH
        self.dead_at: float | None = None
        self.stun_until = 0.0
        self.escape_until = 0.0
        self.rolling_until = 0.0
        self.roll_dir = 1
        self.roll_phase = 0.0
        self.crawl_phase = 0.0
        self.crawl_speed = 0.0
        self.head_cast_angle = 0.0
        self.head_cast_target = 0.0

        self.hurt = 0.0
        self.recover = 1.0
        self.grabbed: int | None = None
        self.last_hit_x = x
        self.action = "idle"

        # Environmental & damage effects
        self.char = 0.0
        self.burn_until = 0.0
        self.soak = 0.0
        self.melt = 0.0
        self.dissolved_at: float | None = None
        self.frost = 0.0
        self.frozen_at: float | None = None
        self.shattered_at: float | None = None
        self.venom = 0.0
        self.wrapped = False
        self.zap_until = 0.0
        self.eating_until = 0.0
        self.arena = "room"
        self.wind = 0.0
        self.wet = 0.0
        self.inebriation = 0.0
        self.stuck: dict[int, np.ndarray] = {}

    @property
    def flying(self) -> bool:
        """Larvae never fly."""
        return False

    @property
    def dead(self) -> bool:
        return self.dead_at is not None

    @property
    def num_segments(self) -> int:
        return self.n_segments

    @property
    def is_rolling(self) -> bool:
        return self.rolling_until > 0.0

    def nearest(self, pos: tuple[float, float] | np.ndarray, max_d: float) -> int | None:
        """Find index of nearest segment to pos."""
        d = np.hypot(*(self.p - pos).T) - LARVA_SEG_RADIUS
        i = int(np.argmin(d))
        return i if d[i] < max_d else None

    def impulse(self, i: int, v: tuple[float, float] | np.ndarray) -> None:
        """Apply an impulse vector to segment i."""
        if 0 <= i < self.n_segments:
            self.prev[i] -= v

    def stun(self, now: float, s: float) -> None:
        self.stun_until = max(self.stun_until, now + s)
        self.hurt = 1.0
        self.rolling_until = min(self.rolling_until, now)

    def trigger_roll(self, now: float, duration: float = 1.5, direction: int | None = None) -> None:
        """Trigger Goro-mediated lateral rolling escape."""
        if self.dead or self.frozen_at is not None or self.wrapped:
            return
        self.rolling_until = max(self.rolling_until, now + duration)
        self.roll_dir = direction if direction is not None else random.choice([-1, 1])
        self.action = "rolling"

    def escape(self, now: float, seconds: float = 2.0, wander: bool = False) -> None:
        """Larval escape reaction: triggers rapid rolling escape."""
        away = 1 if self.p[LARVA_HEAD, 0] >= self.last_hit_x else -1
        self.trigger_roll(now, duration=seconds, direction=away)

    def step(self, now: float, mouse: Any = None) -> list[tuple[int, float]]:
        """Advance one 60 Hz physics step."""
        contacts = []
        if self.frozen_at is not None:
            self.prev = self.p.copy()
            return contacts

        is_rolling = now < self.rolling_until and not self.dead
        is_stunned = now < self.stun_until or self.dead or self.wrapped

        # Verlet velocity integration with friction
        v = (self.p - self.prev) * 0.92
        self.prev = self.p.copy()
        self.p += v

        # Gravity & floor constraint
        for i in range(self.n_segments):
            if i not in self.stuck:
                self.p[i, 1] += 0.5  # gravity
                r = LARVA_SEG_RADIUS[i]
                if self.p[i, 1] > self.floor_y - r:
                    self.p[i, 1] = self.floor_y - r
                    self.prev[i, 1] = self.floor_y - r
                    self.p[i, 0] += v[i, 0] * 0.1  # ground friction

        # Stuck in flypaper
        for i, pos in self.stuck.items():
            if i < self.n_segments:
                self.p[i] = pos.copy()
                self.prev[i] = pos.copy()

        # Locomotion behaviors
        if not is_stunned and len(self.stuck) == 0:
            if is_rolling:
                # Rolling corkscrew motion: fast lateral rolling
                self.roll_phase += 0.35 * self.roll_dir
                roll_speed = 4.5 * self.roll_dir
                self.p[:, 0] += roll_speed
                # C-curling deformation during rolling
                mid = (self.n_segments - 1) / 2.0
                for i in range(self.n_segments):
                    curve = math.sin((i - mid) / mid * math.pi * 0.5)
                    self.p[i, 1] += math.sin(self.roll_phase) * curve * 5.0
            else:
                # Forward crawling peristalsis
                self.crawl_phase += 0.08
                for i in range(self.n_segments):
                    phase_i = self.crawl_phase - i * 0.5
                    wave = max(0.0, math.sin(phase_i))
                    # Contraction-extension forward movement
                    self.p[i, 0] += self.facing * wave * 0.8
                # Head casting
                self.head_cast_angle += (self.head_cast_target - self.head_cast_angle) * 0.1
                if random.random() < 0.02:
                    self.head_cast_target = random.uniform(-0.6, 0.6)
                for i in range(min(3, self.n_segments)):
                    offset = (3 - i) * math.sin(self.head_cast_angle) * 2.5
                    self.p[i, 1] += offset * 0.3

        # Distance constraints between adjacent segments (Verlet relaxation)
        for _ in range(4):
            for i in range(self.n_segments - 1):
                p1, p2 = self.p[i], self.p[i + 1]
                diff = p1 - p2
                dist = math.hypot(diff[0], diff[1])
                target_dist = LARVA_SEG_SPACING * (1.0 - 0.2 * math.sin(self.crawl_phase - i * 0.5))
                if dist > 1e-4:
                    corr = diff * ((dist - target_dist) / dist) * 0.5
                    if i not in self.stuck:
                        self.p[i] -= corr
                    if (i + 1) not in self.stuck:
                        self.p[i + 1] += corr

        return contacts


def draw_larva(surf: pygame.Surface, larva: LarvaBody, now: float) -> None:
    """Render segmented larva body on the pygame surface."""
    p = larva.p
    n = larva.n_segments

    # Color calculation (creamy translucent white with shading)
    base_color = (235, 230, 210)
    if larva.char > 0.0:
        base_color = tuple(int(c * (1.0 - 0.7 * larva.char) + 20 * larva.char) for c in base_color)
    if larva.frost > 0.0:
        base_color = tuple(int(c * (1.0 - 0.6 * larva.frost) + 200 * larva.frost) for c in base_color)
    if larva.dead:
        base_color = tuple(int(c * 0.6 + 50) for c in base_color)

    # Draw segments back to front
    for i in range(n - 1, -1, -1):
        x, y = int(round(p[i, 0])), int(round(p[i, 1]))
        r = int(round(LARVA_SEG_RADIUS[i]))
        # Segment body circle
        gfxdraw.filled_circle(surf, x, y, r, base_color)
        gfxdraw.aacircle(surf, x, y, r, (max(0, base_color[0] - 40), max(0, base_color[1] - 40), max(0, base_color[2] - 40)))

        # Segment inter-annuli bands
        if i < n - 1:
            x_next, y_next = int(round(p[i + 1, 0])), int(round(p[i + 1, 1]))
            pygame.draw.line(surf, (200, 195, 175), (x, y), (x_next, y_next), max(1, r - 2))

    # Internal gut tube visible through translucent cuticle
    for i in range(1, n - 1):
        x, y = int(round(p[i, 0])), int(round(p[i, 1]))
        gfxdraw.filled_circle(surf, x, y, 3, (160, 140, 90, 180))

    # Cephalopharyngeal skeleton / mouth hooks at head
    hx, hy = int(round(p[LARVA_HEAD, 0])), int(round(p[LARVA_HEAD, 1]))
    hook_x = hx + larva.facing * 5
    pygame.draw.circle(surf, (30, 25, 20), (hook_x, hy + 2), 2)
