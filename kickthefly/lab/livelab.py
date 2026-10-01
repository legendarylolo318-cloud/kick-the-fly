"""Live (in-game) state for the 3.0 day 2 Lab tools, kept apart from the drawing so it can be tested without a window.

  ThermoLive   the thermogenetic expressions of the running game and the temperature they see (Lab > Thermogenetics)
  ImagingLive  the brain view's Imaging mode: a live ImagingSession fed from the brain's activity, the dF/F-colored view, and the
               frames kept for export (Lab > Imaging)

Everything here is MODEL / GAME RULE as documented in thermogenetics.py and imaging.py. Nothing runs unless you turn it on, and
nothing here uses the network or the microphone.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from kickthefly.lab import imaging, thermogenetics


class ThermoLive:
    def __init__(self):
        self.expressions: list[dict] = []          # {effector, target, strength}
        self.source = "slider"                     # "slider" or "arena" (the thermo arena's own temperature)
        self.temperature_c = 22.0
        self.kinetics = "real"
        self.per_brain: dict[int, tuple] = {}      # id(brain) -> (Thermogenetics, brain steps at its last update)

    @property
    def active(self) -> bool:
        return bool(self.expressions)

    def add(self, effector: str, target: str, strength: float = 1.0) -> dict:
        thermogenetics.effector(effector)
        e = dict(effector=thermogenetics.effector(effector).key, target=str(target).strip(), strength=float(strength))
        if not e["target"]:
            raise thermogenetics.ThermoError("give a neuron target: a cell type, prefix:KC or line:SS00727")
        self.expressions.append(e)
        self.per_brain.clear()
        return e

    def remove(self, i: int, brains=()) -> None:
        if 0 <= i < len(self.expressions):
            del self.expressions[i]
        self.clear_state(brains)

    def clear(self, brains=()) -> None:
        self.expressions.clear()
        self.clear_state(brains)

    def clear_state(self, brains=()) -> None:
        for b in brains:
            b.clear_current(thermogenetics.Thermogenetics.SOURCE)
        self.per_brain.clear()

    def tick(self, slots, arena_is_thermo: bool = False) -> None:
        """Update every fly's effectors. slots: objects with .brain (and optionally .arena_temp_c, set by the thermo arena)."""
        if not self.expressions:
            if self.per_brain:
                self.clear_state([s.brain for s in slots])
            return
        for slot in slots:
            br = slot.brain
            key = id(br)
            th, last = self.per_brain.get(key, (None, br.steps))
            if th is None:
                th = thermogenetics.Thermogenetics([thermogenetics.Expression(**e) for e in self.expressions], self.kinetics)
            temp = getattr(slot, "arena_temp_c", None) if (self.source == "arena" and arena_is_thermo) else None
            temp = self.temperature_c if temp is None else temp
            dt = max(0.0, (br.steps - last) * 0.005)
            try:
                th.kinetics = self.kinetics
                th.update(br, temp, dt)
            except thermogenetics.ThermoError:
                raise
            self.per_brain[key] = (th, br.steps)

    def status(self) -> list[dict]:
        """One row per expression for the first live fly (activation and neuron count)."""
        if not self.per_brain:
            return [dict(effector=e["effector"], target=e["target"], neurons=0, activation=0.0) for e in self.expressions]
        th = next(iter(self.per_brain.values()))[0]
        return th.describe()


class ImagingLive:
    def __init__(self):
        self.on = False
        self.indicator = imaging.DEFAULT_INDICATOR
        self.fps = 20.0
        self.shot_noise = True
        self.f0_photons = 100.0
        self.dff_per_spike = 0.2
        self.roi_mode = "regions"                   # "regions" or a neuron spec (a type, prefix:..., line:...)
        self.keep_frames = False
        self.gain = 1.0
        self.session: imaging.ImagingSession | None = None
        self.frames: deque = deque(maxlen=300)
        self.smooth: np.ndarray | None = None
        self.error = ""
        self._brain_id = None

    def start(self, br, graph=None) -> None:
        if self.roi_mode == "regions":
            rois = imaging.rois_by_region(br, graph)
        else:
            rois = imaging.rois_from_specs(br, [s.strip() for s in self.roi_mode.split(";") if s.strip()])
        self.session = imaging.ImagingSession(br.n, rois, self.indicator, self.fps, self.f0_photons, self.dff_per_spike,
                                              self.shot_noise, seed=int(getattr(br, "seed", 0)), max_frames=600)
        self.frames.clear()
        self.smooth = None
        self._brain_id = id(br)
        self.error = ""

    def set_on(self, on: bool, br=None, graph=None) -> None:
        self.on = bool(on)
        if not self.on:
            self.session = None
            return
        try:
            self.start(br, graph)
        except imaging.ImagingError as e:
            self.on, self.session, self.error = False, None, str(e)

    def feed(self, br) -> None:
        """Called from the brain-view thread about 20 times a second. A different brain (a new focus) restarts the session."""
        if not self.on:
            return
        if self.session is None or self._brain_id != id(br):
            self.start(br, getattr(br, "graph", None))
        self.session.feed_from_activity(br.sim.activity)

    def recolor(self, view, surf, palette: str, reduced_flashing: bool):
        """The view's surface after rendering dF/F rates, mapped through the imaging palette (a new surface)."""
        import pygame

        arr = pygame.surfarray.array3d(surf).swapaxes(0, 1)
        out = imaging.apply_lut(arr, palette)
        if self.keep_frames and self.session is not None and len(self.session.t) > len(self.frames):
            self.frames.append(out.copy())
        return pygame.image.frombuffer(out.tobytes(), (out.shape[1], out.shape[0]), "RGB").copy()

    def view_rates(self, view, reduced_flashing: bool):
        s = self.session
        if s is None:
            return None
        rates, full = s.view_rates(view.calm, view.hot_mask, self.gain, self.smooth if reduced_flashing else None)
        self.smooth = full
        return rates

    def result(self):
        if self.session is None or not self.session.t:
            return None
        res = self.session.result(dict(live=True))
        res.frames = list(self.frames)
        return res
