"""Targeted optogenetics laser for in-world real-time stimulation and silencing.

Provides real-time targeted current injection into selected cell types without
opening menus. Replaces static surgery for rapid interactive experiments.

Scientific note: this models targeted cellular stimulation (+current) and
silencing (-current) directly on connectome neuron rows. It does NOT claim
Gal4/UAS driver specificity or named opsin kinetics (ChR2/GtACR), as expression
data is not part of the MaleCNS v1.0 reconstruction.
"""
from __future__ import annotations

import math
import numpy as np

# Standard current intensities matching surgery constants
BASE_STIM_CURRENT = 0.48    # depolarizing current (stimulate)
BASE_SILENCE_CURRENT = -2.40  # hyperpolarizing current (silence)

DEFAULT_TARGETS = [
    "LC10",     # visual target-tracking -> steering
    "LPLC2",    # looming escape -> giant fiber
    "LC4",      # looming escape -> giant fiber
    "dnp01",    # giant fiber escape command neuron
    "DNg02",    # wing-power flight command neuron
    "MDN",      # moonwalker backward walking command neuron
    "aDN1",     # antennal grooming command neuron
    "MN9",      # proboscis extension motor neuron
    "PPL1",     # punishment dopaminergic neurons
    "PAM",      # reward dopaminergic neurons
    "DNa02",    # descending steering neuron
]


class LaserState:
    """Configuration and live state of the targeted optogenetics laser."""

    def __init__(
        self,
        target_type: str = "dnp01",
        mode: str = "activate",       # "activate" (+1) or "silence" (-1)
        intensity: float = 1.0,       # multiplier 0.1 .. 3.0
        trigger_mode: str = "hold",   # "hold" or "pulse"
        pulse_duration: float = 0.20, # seconds
    ):
        self.target_type = target_type
        self.mode = mode
        self.intensity = float(np.clip(intensity, 0.1, 3.0))
        self.trigger_mode = trigger_mode
        self.pulse_duration = float(np.clip(pulse_duration, 0.05, 2.0))

        # Runtime dynamic state
        self.firing: bool = False
        self.pulse_until: float = 0.0
        self.hit_fly: bool = False
        self.hit_pos: tuple[float, ...] | None = None
        self.last_applied_rows: np.ndarray = np.array([], dtype=int)
        self.last_applied_current: float = 0.0

    def set_target(self, target: str) -> None:
        self.target_type = str(target).strip()

    def set_mode(self, mode: str) -> None:
        if mode in ("activate", "silence"):
            self.mode = mode

    def toggle_mode(self) -> str:
        self.mode = "silence" if self.mode == "activate" else "activate"
        return self.mode

    def set_intensity(self, val: float) -> None:
        self.intensity = float(np.clip(val, 0.1, 3.0))

    def set_trigger_mode(self, mode: str) -> None:
        if mode in ("hold", "pulse"):
            self.trigger_mode = mode

    def current_value(self) -> float:
        base = BASE_STIM_CURRENT if self.mode == "activate" else BASE_SILENCE_CURRENT
        return float(base * self.intensity)

    def is_active(self, now: float) -> bool:
        """Returns True if the laser is currently emitting light."""
        if self.trigger_mode == "hold":
            return self.firing
        elif self.trigger_mode == "pulse":
            return now < self.pulse_until
        return False

    def trigger_press(self, now: float) -> None:
        self.firing = True
        if self.trigger_mode == "pulse":
            self.pulse_until = now + self.pulse_duration

    def trigger_release(self) -> None:
        self.firing = False

    def resolve_target_rows(self, brain) -> np.ndarray:
        """Find neuron indices in brain matching target_type.

        The answer depends only on ``brain.types`` (the cell-type labels),
        which never change for the life of a brain. The lookup itself is a
        full ~166k string scan, so we cache it on the brain and skip it on
        every later call -- laser.apply() runs once per fly, once per frame,
        and a 26 ms scan per call was the cause of the laser-lag report.
        """
        tt = self.target_type.lower()
        if not hasattr(brain, "types"):
            return np.array([], dtype=int)
        cache = getattr(brain, "_laser_row_cache", None)
        if cache is None:
            cache = {}
            brain._laser_row_cache = cache
        rows = cache.get(tt)
        if rows is None:
            rows = self._scan_rows(brain, tt)
            cache[tt] = rows
        return rows

    def _scan_rows(self, brain, tt: str) -> np.ndarray:
        """One-off full-table scan for matching rows (exact > prefix > contains). `line:SS00727` (3.0 day 2) selects the
        cell types a split-GAL4 line is reported to label (lab/genetics.py)."""
        if tt.startswith("line:"):
            from kickthefly.lab import genetics

            try:
                return genetics.rows_of_lines(brain.types, tt)
            except genetics.GeneticsError:
                return np.array([], dtype=int)
        if tt.startswith(("type:", "prefix:", "side:")):            # 3.1.0 (Puppeteer): the same specs simcore.rows_of reads, plus one side
            return self._spec_rows(brain, tt)
        types = np.char.lower(brain.types.astype(str))
        exact = np.flatnonzero(types == tt)
        if len(exact):
            return exact
        prefix = np.flatnonzero(np.char.startswith(types, tt))
        if len(prefix):
            return prefix
        contains = np.flatnonzero(np.char.find(types, tt) >= 0)
        return contains

    @staticmethod
    def _spec_rows(brain, tt: str) -> np.ndarray:
        """`type:A,B` (exact types), `prefix:A,B` (type names starting with), `side:R:A,B` (those exact types, the right or left cell of each pair
        only: the instance name ends _R or _L). Case-insensitive."""
        types = np.char.lower(brain.types.astype(str))
        kind, _, rest = tt.partition(":")
        side = None
        if kind == "side":
            side, _, rest = rest.partition(":")
            kind = "type"
        names = [x.strip() for x in rest.split(",") if x.strip()]
        m = np.zeros(len(types), bool)
        for nme in names:
            m |= (np.char.startswith(types, nme) if kind == "prefix" else types == nme)
        if side:
            inst = np.array([("" if x is None else str(x)).lower() for x in getattr(brain, "instance", np.full(len(types), ""))])
            m &= np.char.endswith(inst, "_" + side.lower())
        return np.flatnonzero(m)

    def apply(self, brain, now: float, is_hitting: bool) -> float:
        """Applies laser current to target neurons if active and striking fly.

        Returns the injected current per neuron (0.0 if not active/hitting).
        """
        active = self.is_active(now) and is_hitting
        target_current = self.current_value() if active else 0.0
        applied = getattr(brain, "_laser_rows", np.array([], dtype=int))

        if active:
            rows = self.resolve_target_rows(brain)
            if hasattr(brain, "override"):
                if len(applied):
                    brain.override[applied] = 0.0
                if len(rows):
                    brain.override[rows] = target_current
                brain.surgery = bool(np.any(brain.override))
            brain._laser_rows = rows
            self.last_applied_rows = rows
            self.last_applied_current = target_current
        else:
            if len(applied) and hasattr(brain, "override"):
                brain.override[applied] = 0.0
                brain.surgery = bool(np.any(brain.override))
                brain._laser_rows = np.array([], dtype=int)
            if not self.is_active(now):
                self.last_applied_rows = np.array([], dtype=int)
                self.last_applied_current = 0.0

        self.hit_fly = active
        return target_current

    def clear(self, brain) -> None:
        """Completely clear any laser overrides on the brain."""
        applied = getattr(brain, "_laser_rows", self.last_applied_rows)
        if len(applied) and hasattr(brain, "override"):
            brain.override[applied] = 0.0
            brain.surgery = bool(np.any(brain.override))
        brain._laser_rows = np.array([], dtype=int)
        self.last_applied_rows = np.array([], dtype=int)
        self.last_applied_current = 0.0
        self.hit_fly = False
        self.firing = False
        self.pulse_until = 0.0
