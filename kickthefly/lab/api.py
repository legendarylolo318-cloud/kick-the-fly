"""A fly brain as a Python object: the headless machinery the Lab and protocols use, for notebooks and scripts.

    from kickthefly import Fly

    fly = Fly(seed=1000)                         # a warmed-up, untrained brain (never your saved training memory)
    rec = fly.record({"giant fiber": "dnp01", "looming": "loom"})
    fly.step(1.0)                                # 1 s of calm
    fly.drive("loom", amp=0.5)                   # hold LPLC2 + LC4 driven, like optogenetic activation
    fly.step(1.0)
    fly.undrive()
    print(rec.rates(), rec.rates(start_s=1.0))   # spikes/s per group, over the whole run or from 1 s on
    fly.export("out/looming")                    # spikes, rates and metadata CSV/JSON (+ .nwb with nwb=True)

Neurons are named exactly as in protocols: a group ("loom", "dnp01", "sweet", "pip10", "p1", ...), "type:A,B",
"prefix:KC", "superclass:descending_neuron", "rows:1,2,3", or an array of rows. Everything steps in lockstep on the
calling thread, so the same seed and the same calls give the same spikes (on cpu, numba and torch-cpu alike).
What is connectome and what is a game rule is the same as everywhere else (kickthefly/game/kick_the_fly.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DT = 0.005                                      # seconds per brain step


class Recording:
    """Spikes of the recorded neuron groups since Fly.record() (a lab.recorder.Recorder underneath)."""

    def __init__(self, rec):
        self._rec = rec

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return dict(self._rec.groups)

    @property
    def seconds(self) -> float:
        return self._rec.seconds

    def spikes(self) -> tuple[np.ndarray, np.ndarray]:
        """(time in s since the recording started, neuron row) of every recorded spike, in time order."""
        a = self._rec.arrays()
        order = np.lexsort((a["spike_index"], a["spike_steps"]))
        return a["spike_steps"][order] * DT, self._rec.rows[a["spike_index"][order]]

    def rates(self, start_s: float = 0.0, stop_s: float | None = None) -> dict[str, float]:
        """Mean spikes/s per neuron of each group between start_s and stop_s (seconds since the recording began)."""
        a = self._rec.arrays()
        n = a["n_steps"]
        lo, hi = int(round(start_s / DT)), n if stop_s is None else min(n, int(round(stop_s / DT)))
        keep = (a["spike_steps"] >= lo) & (a["spike_steps"] < hi)
        idx = a["spike_index"][keep]
        out = {}
        for name, rows in self._rec.groups.items():
            members = np.searchsorted(self._rec.rows, rows)
            out[name] = float(np.isin(idx, members).sum()) / max(1, len(rows)) / max(DT, (hi - lo) * DT)
        return out


class Fly:
    """One simulated brain (MaleCNS v1.0, 166,700 LIF neurons), stepped on demand. See the module docstring."""

    def __init__(self, seed: int = 0, *, backend: str | None = None, params: dict | None = None,
                 warmup_s: float = 3.0, learn: bool = True, surgery: dict | None = None, mode: str = "play"):
        from kickthefly.core import loadout, simcore
        from kickthefly.lab import assays

        self.seed = int(seed)
        self.brain = simcore.new_brain(seed=self.seed, memory=learn, warmup=int(round(warmup_s / DT)),
                                       params=params, backend=backend)
        assays.apply_surgery(self.brain, surgery)
        self._recording: Recording | None = None
        self._exports: list[Path] = []
        self.mode = mode if mode in ("play", "lab", "pet") else "play"
        self._loadout = loadout.build("auto", None, lab=self.mode == "lab", larva=False, mode=self.mode)
        self._tool = loadout.HAND

    # --- tool loadouts (2.13) --------------------------------------------------------------------------------------------
    @property
    def loadout(self):
        """The hotbar: a core.loadout.Loadout ("base" in Play, "lab" in Lab, "pet" in Pet mode by default). `.tools` is
        the list of tool names in slot order (the hand always first), `.page_tools()` one page of ten."""
        return self._loadout

    def set_loadout(self, preset_or_tools) -> "Fly":
        """A preset name ("base", "chaos", "chemist", "lab", "all", "pet", "auto") or your own list of tool names."""
        from kickthefly.core import loadout

        lab = self.mode == "lab"
        if isinstance(preset_or_tools, str):
            if preset_or_tools not in loadout.CHOICES or preset_or_tools == "custom":
                raise ValueError(f"unknown preset {preset_or_tools!r}: use one of {loadout.CHOICES[:-1]}")
            self._loadout = loadout.build(preset_or_tools, None, lab=lab, larva=False, mode=self.mode)
        else:
            bad = [t for t in preset_or_tools if t not in loadout.BY_NAME]
            if bad:
                raise ValueError(f"unknown tools {bad}: use names from kickthefly.core.loadout.TOOL_NAMES")
            self._loadout = loadout.build("custom", list(preset_or_tools), lab=lab, larva=False, mode=self.mode)
        if self._tool not in self._loadout:
            self._tool = loadout.HAND
        return self

    @property
    def tool(self) -> str:
        """The tool in hand (set by use_tool)."""
        return self._tool

    def use_tool(self, name: str, strength: float = 0.8) -> "Fly":
        """Take a tool in hand and use it once on this brain: its documented sensory neurons are driven through
        Brain.poke, as the game's tools drive them (no body, no room). A tool the mode doesn't allow raises ValueError
        (the laser outside Lab, which is `Fly(mode="lab")`). Tools that need a body to do anything else (moving the
        fly, an item the fly eats) are only their sensory drive here."""
        from kickthefly.core import loadout

        if name not in loadout.BY_NAME or not loadout.available(name, lab=self.mode == "lab", larva=False):
            raise ValueError(f"tool {name!r} is not available in {self.mode} mode")
        self._tool = name
        if name == "laser":
            self.drive("type:DNp01", amp=0.5)
        else:
            loadout.use(self.brain, name, strength)
        return self

    # --- who is who ------------------------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.brain.n)

    @property
    def t(self) -> float:
        """Brain time in seconds (the warm-up included)."""
        return self.brain.steps * DT

    @property
    def backend(self) -> str:
        """The compute backend that actually runs this brain (after any fallback)."""
        return self.brain.sim.backend.name

    def neurons(self, spec) -> np.ndarray:
        """Rows for a neuron spec (see the module docstring)."""
        from kickthefly.lab import protocol

        if isinstance(spec, (list, tuple, np.ndarray)):
            return np.asarray(spec, np.int64)
        return np.asarray(protocol._rows(self.brain, spec), np.int64)

    def describe(self, row: int) -> dict:
        """Type, instance, superclass and body ID of one neuron."""
        br = self.brain
        body = getattr(br, "body_id", None)
        return dict(row=int(row), type=str(br.types[row]), instance=str(br.instance[row]),
                    superclass=str(br.superclass[row]), body_id=None if body is None else int(body[row]))

    # --- manipulations -----------------------------------------------------------------------------------------------
    def drive(self, spec, amp: float = 0.5) -> "Fly":
        """Hold these neurons driven with a current every step until undrive(), like optogenetic activation (the
        validation suite's drive is amp 0.5)."""
        from kickthefly.core import simcore

        simcore.drive(self.brain, self.neurons(spec), float(amp))
        return self

    def undrive(self, spec=None) -> "Fly":
        """Stop driving these neurons, or every driven neuron."""
        from kickthefly.core import simcore

        rows = np.flatnonzero(self.brain.drive_cur) if spec is None else self.neurons(spec)
        simcore.undrive(self.brain, rows)
        return self

    def silence(self, spec) -> "Fly":
        """Brain surgery OFF: a strong inhibitory current on these neurons (as the game's surgery panel does)."""
        self.brain.set_override(self.neurons(spec), -1)
        return self

    def stimulate(self, spec) -> "Fly":
        """Brain surgery ON: the surgery panel's gentle constant excitation."""
        self.brain.set_override(self.neurons(spec), 1)
        return self

    def restore(self, spec=None) -> "Fly":
        """Undo silence()/stimulate() on these neurons, or everywhere."""
        if spec is None:
            self.brain.clear_overrides()
        else:
            self.brain.set_override(self.neurons(spec), 0)
        return self

    def poke(self, region: str, strength: float = 0.8, side: str | None = None) -> "Fly":
        """A game-style sensory stimulus ("head", "body", "legs", "wing", "heat", "cold", "wind", "light", "loom",
        "smell", "taste", ...): a share of that region's neurons fires for a short pulse, as when a tool hits."""
        self.brain.poke(region, side, strength)
        return self

    # --- time --------------------------------------------------------------------------------------------------------
    def step(self, seconds: float | None = None, *, steps: int | None = None) -> np.ndarray:
        """Advance the brain (5 ms per step). Returns the last step's spikes (a bool array over all neurons)."""
        n = int(steps) if steps is not None else int(round((DT if seconds is None else seconds) / DT))
        for _ in range(max(0, n)):
            self.brain._step()
        return self.brain.sim.spikes.copy()

    # --- recording and export ------------------------------------------------------------------------------------
    def record(self, groups) -> Recording:
        """Start recording spikes of these groups: {name: spec}, a list of specs, or one spec. Replaces any recording
        already running."""
        from kickthefly.lab import recorder

        if isinstance(groups, dict):
            named = groups
        elif isinstance(groups, (list, tuple)):
            named = {str(s): s for s in groups}
        else:
            named = {str(groups): groups}
        if self._recording is not None:
            self._recording._rec.stop()
        rec = recorder.Recorder(self.brain, {k: self.neurons(v) for k, v in named.items()}).start()
        self._recording = Recording(rec)
        return self._recording

    @property
    def recording(self) -> Recording | None:
        return self._recording

    def export(self, path, nwb: bool = False, note: str = "") -> list[Path]:
        """Write the current recording: <path>-spikes.csv, -rates.csv, -group-rates.csv, -meta.json and, with nwb=True
        (needs pynwb), <path>.nwb. Returns the files written."""
        from kickthefly.lab import recorder

        if self._recording is None:
            raise RuntimeError("nothing recorded yet: call fly.record(...) first")
        rec = self._recording._rec
        stem = Path(path)
        extra = dict(source="kickthefly.Fly", seed=self.seed, note=note, backend=self.backend,
                     device=self.brain.sim.backend.device)
        files = list(rec.save(stem, extra))
        if nwb:
            from kickthefly.lab import nwbexport

            reason = nwbexport.available()
            if reason:
                raise RuntimeError(reason)
            p = stem.with_name(stem.name + ".nwb")
            nwbexport.write(rec, p, recorder.metadata(self.brain, None, extra))
            files.append(p)
        return files
