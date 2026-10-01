"""Live (in-game) state for the 3.0 day 2 Lab tools, kept apart from the drawing so it can be tested without a window.

  ThermoLive   the thermogenetic expressions of the running game and the temperature they see (Lab > Thermogenetics)
  ImagingLive  the brain view's Imaging mode: a live ImagingSession fed from the brain's activity, the dF/F-colored view, and the
               frames kept for export (Lab > Imaging)

Everything here is MODEL / GAME RULE as documented in thermogenetics.py and imaging.py. Nothing runs unless you turn it on, and
nothing here uses the network or the microphone.
"""
from __future__ import annotations

import weakref
from collections import deque

import numpy as np

from kickthefly.lab import imaging, thermogenetics


class ThermoLive:
    def __init__(self):
        self.expressions: list[dict] = []          # {effector, target, strength}
        self.source = "slider"                     # "slider" or "arena" (the thermo arena's own temperature)
        self.temperature_c = 22.0
        self.kinetics = "real"
        # id(brain) -> (brain, Thermogenetics, brain steps at its last update). The brain itself is kept and compared: a new
        # Brain (a respawn, a loaded save) can get a dead one's id(), and must not inherit its state (3.0 day 2 review).
        self.per_brain: dict[int, tuple] = {}

    @property
    def active(self) -> bool:
        return bool(self.expressions)

    def add(self, effector: str, target: str, strength: float = 1.0, brain=None) -> dict:
        """Add an expression. With `brain`, the target is resolved on it first, so a typo is refused here instead of later."""
        e = dict(effector=thermogenetics.effector(effector).key, target=str(target).strip(), strength=float(strength))
        if not e["target"]:
            raise thermogenetics.ThermoError("give a neuron target: a cell type, prefix:KC or line:SS00727")
        thermogenetics.Expression(**e)                 # checks the strength
        if brain is not None:
            self._resolve(brain, e)
        self.expressions.append(e)
        self.per_brain.clear()
        return e

    @staticmethod
    def _resolve(br, e: dict) -> None:
        th = thermogenetics.Thermogenetics([thermogenetics.Expression(**e)])
        th.resolve(br)

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
        """Update every fly's effectors. slots: objects with .brain (and optionally .arena_temp_c, set by the thermo arena).
        An expression whose target this brain can't resolve is dropped (the others stay) and ThermoError says which."""
        if not self.expressions:
            if self.per_brain:
                self.clear_state([s.brain for s in slots])
            return
        live = {id(s.brain) for s in slots}
        for key in [k for k in self.per_brain if k not in live]:   # flies that left the game
            del self.per_brain[key]
        for slot in slots:
            br = slot.brain
            key = id(br)
            ent = self.per_brain.get(key)
            if ent is None or ent[0] is not br:
                bad = []
                for e in self.expressions:
                    try:
                        self._resolve(br, e)
                    except thermogenetics.ThermoError as err:
                        bad.append((e, err))
                if bad:
                    for e, _ in bad:
                        self.expressions.remove(e)
                    self.clear_state([s.brain for s in slots])
                    raise thermogenetics.ThermoError("dropped " + "; ".join(f"{e['effector']} in {e['target']}: {err}"
                                                                           for e, err in bad))
                th = thermogenetics.Thermogenetics([thermogenetics.Expression(**e) for e in self.expressions], self.kinetics)
                br.clear_current(thermogenetics.Thermogenetics.SOURCE)
                ent = (br, th, br.steps)
            _, th, last = ent
            temp = getattr(slot, "arena_temp_c", None) if (self.source == "arena" and arena_is_thermo) else None
            temp = self.temperature_c if temp is None else temp
            dt = max(0.0, (br.steps - last) * 0.005)
            th.kinetics = self.kinetics
            th.update(br, temp, dt)
            self.per_brain[key] = (br, th, br.steps)

    def status(self) -> list[dict]:
        """One row per expression for the first live fly (activation and neuron count)."""
        if not self.per_brain:
            return [dict(effector=e["effector"], target=e["target"], neurons=0, activation=0.0) for e in self.expressions]
        th = next(iter(self.per_brain.values()))[1]
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
        self.frame_times: deque = deque(maxlen=300)  # the imaging time of each kept frame (3.0 day 2 review)
        self._kept_upto = 0                         # session.frames_taken when the last frame was kept
        self.smooth: np.ndarray | None = None
        self.error = ""
        self._brain = None                          # weak reference to the brain being imaged

    def start(self, br, graph=None) -> None:
        if self.roi_mode == "regions":
            rois = imaging.rois_by_region(br, graph)
        else:
            rois = imaging.rois_from_specs(br, [s.strip() for s in self.roi_mode.split(";") if s.strip()])
        session = imaging.ImagingSession(br.n, rois, self.indicator, self.fps, self.f0_photons, self.dff_per_spike,
                                         self.shot_noise, seed=int(getattr(br, "seed", 0)), max_frames=600)
        self.frames.clear()
        self.frame_times.clear()
        self._kept_upto = 0
        self.smooth = None
        self._brain = weakref.ref(br)
        self.error = ""
        self.session = session                        # last: a session is complete when another thread can see it

    def set_on(self, on: bool, br=None, graph=None) -> None:
        if not on:
            self.on, self.session = False, None
            return
        # 3.0 day 2 review: `on` used to be set before the session was built. Building it takes a moment on the real brain
        # (the kernel fit, the region ROIs over 166,700 neurons), and the view thread, seeing imaging on with no session,
        # raised "no imaging session" and switched it off again; start() then cleared the error. Imaging mode never came on.
        try:
            self.start(br, graph)
        except imaging.ImagingError as e:
            self.on, self.session, self.error = False, None, str(e)
            return
        self.on = True

    def feed(self, br) -> None:
        """Called from the brain-view thread about 20 times a second. A different brain (a new focus) restarts the session."""
        if not self.on:
            return
        if self.session is None or self._brain is None or self._brain() is not br:     # not id(): a new brain can reuse it
            self.start(br, getattr(br, "graph", None))
        self.session.feed_from_activity(br.sim.activity)

    def recolor(self, view, surf, palette: str, reduced_flashing: bool):
        """The view's surface after rendering dF/F rates, mapped through the imaging palette (a new surface)."""
        import pygame

        arr = pygame.surfarray.array3d(surf).swapaxes(0, 1)
        out = imaging.apply_lut(arr, palette)
        s = self.session
        # 3.0 day 2 review: keep a render only when an imaging frame was taken since the last one kept (comparing list lengths
        # kept every render once the frame buffer was full), with its imaging time; a new view size (panel <-> big view)
        # starts the kept frames over, since a stack of frames has one size.
        if self.keep_frames and s is not None and s.frames_taken > self._kept_upto and s.t:
            if self.frames and self.frames[-1].shape != out.shape:
                self.frames.clear()
                self.frame_times.clear()
            self.frames.append(out.copy())
            self.frame_times.append(float(s.t[-1]))
            self._kept_upto = s.frames_taken
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
        res.frame_times = list(self.frame_times)
        return res
