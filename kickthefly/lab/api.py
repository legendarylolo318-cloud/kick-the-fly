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

3.0 day 2 (each is MODEL or GAME RULE, see the module docstring of the lab module named):

    fly.neurons("line:SS00727")                  # a split-GAL4 driver line's cell types (lab/genetics.py); fly.line("SS00727")
    fly.express("trpa1", "type:DNp01"); fly.temperature(32); fly.step(1.0)    # thermogenetics (lab/thermogenetics.py)
    fly.patch("type:DNp01", [0.0, 0.05, 0.1])    # virtual current clamp, I-F curve (lab/patchclamp.py)
    res = fly.image(5.0, indicator="gcamp6s")    # simulated GCaMP imaging of the next 5 s (lab/imaging.py)
    fly.drug("picrotoxin", 0.5); fly.washout()   # synaptic scaling by predicted transmitter (lab/pharmacology.py)

3.0 day 3 (game rules throughout; what they drive is the connectome's own neurons):

    fly.attack("frog")                           # a frog, dragonfly or mantis attack through the real looming pathway (lab/predators.py)
    fly.weather(rain=0.6, gust_hz=0.2, storm=True); fly.step(10.0)    # rain on touch neurons, gusts on JO-C/E, lightning on the eyes
    fly.hear(hz=200.0, seconds=2.0, ipi_ms=35.0) # a synthetic hum through the microphone's analysis onto JO-A/B (core/mic.py; no device)

Neurons are named exactly as in protocols: a group ("loom", "dnp01", "sweet", "pip10", "p1", ...), "type:A,B",
"prefix:KC", "superclass:descending_neuron", "rows:1,2,3", "line:SS00727" (a driver line), or an array of rows. Everything steps in lockstep on the
calling thread, so the same seed and the same calls give the same spikes (on cpu, numba and torch-cpu alike).
What is connectome and what is a game rule is the same as everywhere else (kickthefly/game/kick_the_fly.py).
"""
from __future__ import annotations

import math
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
        self._dex = None                              # 3.0: Fly.collect() attaches a Neurodex tracker
        self._kc = None                               # 3.0: Fly.killcam() attaches a kill cam buffer
        self._thermo = None                           # 3.0 day 2: Fly.express() adds thermogenetic expression
        self._temp = 22.0
        self._kinetics = "real"
        self._drug = None                             # 3.0 day 2: the pharmacology.Wiring currently applied by Fly.drug()
        self._wx = None                               # 3.0 day 3: Fly.weather() adds rain, gusts and lightning
        self._wx_args = (0.0, 0.0, False, 0.0, 180.0)

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

    # --- genetic toolkit, thermogenetics, patch clamp, imaging, pharmacology (3.0 day 2) ----------------------------------
    def line(self, name: str) -> dict:
        """A split-GAL4 driver line: its cell types, how many neurons of them this connectome has, and what the source says
        about off-target expression (lab/genetics.py). LITERATURE for the mapping, CONNECTOME for the counts."""
        from kickthefly.lab import genetics

        return genetics.describe(name, self.brain.types)

    def express(self, effector: str, target, strength: float = 1.0) -> "Fly":
        """Express TrpA1 ("trpa1") or shibire-ts ("shibire") in these neurons (a neuron spec, including `line:SS00727`). They
        respond to fly.temperature(); the effectors follow it during step(). Thermogenetics: see lab/thermogenetics.py."""
        from kickthefly.lab import thermogenetics as tg

        spec = target if isinstance(target, str) else "rows:" + ",".join(str(int(r)) for r in self.neurons(target))
        ex = (self._thermo.expressions if self._thermo is not None else []) + [tg.Expression(effector, spec, strength)]
        self._thermo = tg.Thermogenetics(ex, self._kinetics)
        return self

    def temperature(self, celsius: float | None = None, kinetics: str | None = None) -> float:
        """Set (or read) the temperature the expressed effectors sense, 10-45 C. kinetics "steady" makes them follow it
        instantly instead of with their time constants."""
        from kickthefly.lab import thermogenetics as tg

        if kinetics is not None:
            if kinetics not in ("real", "steady"):
                raise ValueError("kinetics must be 'real' or 'steady'")
            self._kinetics = kinetics                     # remembered, so it also applies to expression added later
            if self._thermo is not None:
                self._thermo.kinetics = kinetics
        if celsius is not None:
            if not tg.TEMP_RANGE_C[0] <= float(celsius) <= tg.TEMP_RANGE_C[1]:
                raise ValueError(f"temperature must be within {tg.TEMP_RANGE_C[0]:g}-{tg.TEMP_RANGE_C[1]:g} C")
            self._temp = float(celsius)
        return self._temp

    def unexpress(self) -> "Fly":
        """Remove every thermogenetic expression (the currents go away)."""
        if self._thermo is not None:
            self._thermo.clear(self.brain)
            self._thermo = None
        return self

    def patch(self, neuron, amplitudes=(0.0, 0.05, 0.1, 0.2), duration_ms: float = 500.0, repeats: int = 3,
              mode: str = "embedded", index: int = 0) -> dict:
        """Virtual patch clamp (MODEL, a point-neuron LIF unit): firing rate against injected current for one neuron. `neuron`
        is a row number or a spec (then `index` picks which neuron of it). Returns lab/patchclamp.if_curve()'s dict, with the
        recording under "recording". It steps this brain in embedded mode (the current is removed afterwards)."""
        from kickthefly.lab import patchclamp as pc

        row = int(neuron) if isinstance(neuron, (int, np.integer)) else pc.pick_neuron(self.brain, neuron, index)
        return pc.if_curve(self.brain, row, list(amplitudes), duration_ms, repeats, mode, self.seed, self.brain.sim.p)

    def image(self, seconds: float, rois=None, indicator: str = "gcamp6s", fps: float = 20.0, **kw):
        """Simulated calcium imaging (MODEL) of the next `seconds`: spikes convolved with the indicator's kernel, averaged over
        ROIs (default one per brain region; or a list of neuron specs), with photon shot noise. Returns an
        imaging.ImagingResult (export_csv / export_nwb / export_tiff in lab/imaging.py)."""
        from kickthefly.lab import imaging

        seconds = imaging.check_seconds(seconds)
        if rois is None:
            rois = imaging.rois_by_region(self.brain)
        elif not isinstance(rois, dict):
            rois = imaging.rois_from_specs(self.brain, rois)
        if self._thermo is None:
            return imaging.record(self.brain, seconds, rois, indicator, fps, seed=self.seed, **kw)
        s = imaging.ImagingSession(self.brain.n, rois, indicator, fps, seed=self.seed, **kw)
        s.prime_from_activity(self.brain.sim.activity)
        for _ in range(int(round(seconds / DT))):
            self.step(steps=1)
            s.push(np.flatnonzero(self.brain.sim.spikes))
        return s.result()

    def drug(self, name: str, dose: float, include_low_confidence: bool = True, cut: float = 0.7) -> dict:
        """Apply a drug (MODEL PREDICTION): "picrotoxin", "cholinergic", "glucl" or "gabaa_agonist", dose 0-1, scaling the
        synapses of the predicted transmitter. Adds to a drug already on (other drugs stay). Returns what changed. See
        lab/pharmacology.py for what this does and does not model."""
        from kickthefly.lab import pharmacology as ph
        from kickthefly.sim import wiring

        doses = dict(getattr(self, "_doses", {}))
        doses[ph.drug(name).key] = dose
        w = ph.wiring_for(doses, include_low_confidence, cut)       # refuses a bad dose or cut before anything is kept
        out = wiring.apply(self.brain, w, getattr(self.brain, "graph", None))
        self._doses, self._drug_opts = {k: float(v) for k, v in doses.items()}, (include_low_confidence, cut)
        return out

    def washout(self) -> "Fly":
        """Remove every drug (the synapses return to exactly what they were, learned weights included)."""
        from kickthefly.sim import wiring

        wiring.clear(self.brain)
        self._doses = {}
        return self

    # --- 3.0 day 3: predators, weather, a hum ------------------------------------------------------------------------------
    def attack(self, kind: str, seed: int | None = None) -> dict:
        """One predator ('frog', 'dragonfly' or 'mantis') attacks a fly that stays where it is: what its eyes see goes through the
        game's looming transduction onto LPLC2/LC4, and the result says whether DNp01 crossed the escape threshold before the
        capture. GAME RULE attack, MODEL PREDICTION result (lab/predators.py)."""
        from kickthefly.game import predators as pr
        from kickthefly.lab import predators as lp

        if not isinstance(kind, str) or kind not in pr.SPECS:
            raise ValueError(f"unknown predator {kind!r}; use one of {', '.join(pr.KINDS)}")
        return lp.escape_trial(self.brain, pr.trace(kind, self.seed if seed is None else int(seed)))

    def weather(self, rain: float = 0.0, gust_hz: float = 0.0, storm: bool = False, wind_speed: float = 0.0,
                wind_dir: float = 180.0) -> "Fly":
        """Rain, gusts and lightning from now on (rain 0-1, gusts a second 0-0.5, storm on/off, a steady wind in m/s). Drops fire the
        touch neurons by body part, the air the humidity neurons, wind and gusts JO-C/E through the game's wind transduction,
        lightning the photoreceptors. All off (the defaults) removes it. GAME RULE (game/weather.py)."""
        from kickthefly.game import weather

        self._wx_args = (float(rain), float(gust_hz), bool(storm), float(wind_speed), float(wind_dir))
        active = rain > 0 or gust_hz > 0 or storm or wind_speed > 0
        self._wx = (self._wx or weather.Weather(self.seed)) if active else None
        return self

    def _weather_tick(self, br) -> None:
        from kickthefly.game import outdoors

        w = self._wx
        rain, gust, storm, speed, direction = self._wx_args
        w.update(0.05, rain, gust, storm)
        for region, side, s in w.hits(0.05):
            br.poke(region, side, s)
        h = w.humid(0.05)
        if h > 0:
            br.poke("humid", None, h)
        ws, wd = w.wind(speed, direction)
        if ws > 0:
            left, right = outdoors.wind_drive(0.0, wd, ws)
            if left > 0.02:
                br.poke("wind", "L", left)
            if right > 0.02:
                br.poke("wind", "R", right)
        light = w.lightning()
        if light > 0:
            br.poke("light", "L", light, recruit=0.6 * light)
            br.poke("light", "R", light, recruit=0.6 * light)

    def hear(self, hz: float = 200.0, seconds: float = 1.0, ipi_ms: float | None = None, amp: float = 0.1,
             sensitivity: float = 1.0) -> dict:
        """A synthetic hum (a sine at `hz`; with ipi_ms, a train of pulses at that interval) goes through the same analysis the
        microphone uses and drives the JO-A and JO-B neurons for `seconds`; the brain steps meanwhile. Never uses a microphone.
        Returns what the analysis saw (frequency, drive) and the JO neurons' firing. GAME RULE transduction (core/mic.py)."""
        from kickthefly.core import mic
        from kickthefly.lab import audio

        # the protocol audio: block's ranges (3.0 day 3 review: seconds=1e6 tried to build 22 billion samples)
        for name, v, lo, hi in (("hz", hz, 10, 2000), ("seconds", seconds, 0.05, 600), ("amp", amp, 0, 1),
                                ("sensitivity", sensitivity, *mic.SENSITIVITY)) + ((("ipi_ms", ipi_ms, 10, 500),) if ipi_ms is not None else ()):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                raise ValueError(f"hear: {name} must be a number from {lo:g} to {hi:g}")
        a, b = mic.jo_rows(self.brain)
        watch = {"jo_a": a, "jo_b": b}
        res = audio.hear(self.brain, mic.hum(hz, seconds, mic.RATE, amp, ipi_ms), sensitivity=sensitivity, watch=watch)
        n = max(1, res["steps"])
        return dict(peak_hz=res["peak_hz"], drive_a=res["mean_drive_a"], drive_b=res["mean_drive_b"], steps=res["steps"],
                    jo_a_hz=res["counts"]["jo_a"] / max(1, len(a)) / (n * DT), jo_b_hz=res["counts"]["jo_b"] / max(1, len(b)) / (n * DT))

    # --- 3.0 day 4: a measured personality card, and a duel against another fly --------------------------------------------
    def card(self, mode: str = "subtle") -> dict:
        """This fly's personality card MEASURED from its brain (looming latency to its dodge threshold, sugar -> MN9 ratio, steering
        R/L ratio): CONNECTOME readouts of the individual this seed builds (lab/tournament.measure_card). Builds its own brain."""
        from kickthefly.lab import tournament

        return tournament.measure_card(self.seed, mode)

    def duel(self, other: "Fly", seconds: float = 20.0, seed: int = 0) -> dict:
        """1v1 against another Fly: both brains steer, shoot, dodge and run (game/flyduel.py). The arena and blaster are GAME RULE,
        what the neurons do with what they see is the CONNECTOME, the winner is a MODEL PREDICTION. Steps both brains."""
        from kickthefly.game import flyduel

        return flyduel.run_duel(self.brain, other.brain, seed=seed, seconds=seconds, names=("self", "other"))

    # --- time --------------------------------------------------------------------------------------------------------
    def step(self, seconds: float | None = None, *, steps: int | None = None) -> np.ndarray:
        """Advance the brain (5 ms per step). Returns the last step's spikes (a bool array over all neurons)."""
        n = int(steps) if steps is not None else int(round((DT if seconds is None else seconds) / DT))
        br = self.brain
        for _ in range(max(0, n)):
            if self._thermo is not None and br.steps % 10 == 0:
                self._thermo.update(br, self._temp, 10 * DT)
            if self._wx is not None and br.steps % 10 == 0:
                self._weather_tick(br)
            br._step()
            if self._dex is not None or self._kc is not None:
                self._observe()
        return br.sim.spikes.copy()

    # --- Neurodex and kill cam (3.0) --------------------------------------------------------------------------------
    def _observe(self) -> None:
        from kickthefly.core import neurodex as nd

        br = self.brain
        if self._dex is not None and br.steps % nd.CHECK_STEPS == 0 and not br.dead:
            calm = br.sedation == 0 and not br.surgery and not br.driving and br.steps - br.last_poke > 400
            driven = bool(br.surgery or br.driving)
            tracker, prog = self._dex
            spikes, steps = nd.window_spikes(br.sim.activity)
            for name in tracker.observe("fly", spikes, br.dt, calm, driven, window_steps=steps):
                self.discoveries.append((self.t, name, prog.types(tracker.tab.brain)[name]["how"]))
        if self._kc is not None and not br.dead:
            self._kc.push(br.steps, br.sim.activity.rates())

    @property
    def discoveries(self) -> list:
        """[(brain time s, type, "play" | "stimulated")] for what collect() has discovered during step()."""
        if not hasattr(self, "_discoveries"):
            self._discoveries: list = []
        return self._discoveries

    def collect(self, progress=None):
        """Start collecting Neurodex discoveries while you step(). It uses the game's rule (kickthefly/core/neurodex.py) and
        by default an empty, throwaway collection that is never saved: your own Neurodex is only touched if you pass its
        path (progress=neurodex.progress_path()). Returns the neurodex.Progress. A discovery needs the first 5 s of
        brain time to settle, like in the game."""
        import tempfile
        from pathlib import Path as _P

        from kickthefly.core import neurodex as nd

        tab = nd.table("adult")
        prog = nd.Progress(progress) if progress is not None else nd.Progress(_P(tempfile.mkdtemp(prefix="ktf-dex-")) / "neurodex.json")
        self._dex = (nd.Tracker(tab, prog), prog)
        return prog

    def neurodex(self, type_name: str) -> dict | None:
        """The Neurodex entry for a type: dataset facts (CONNECTOME), curated fact and citation (LITERATURE) if there is
        one, and whether this collection has discovered it. None for a type that isn't in the brain."""
        from kickthefly.core import neurodex as nd

        prog = self._dex[1] if self._dex is not None else None
        return nd.entry(nd.table("adult"), type_name, prog)

    def killcam(self, on: bool = True):
        """Keep the last 6 s of per-neuron firing while you step(). After fly.kill(), killcam_replay() returns the
        killcam.Replay (risers, traces, summary)."""
        from kickthefly.core import killcam as kc

        self._kc = kc.Buffer(self.brain.n) if on else None
        return self

    def kill(self) -> "Fly":
        """Kill the fly the way the game does (the drive is cancelled over 1.5 s of further steps)."""
        if self._kc is not None:
            self._replay = self._kc.freeze()
            self._kc.reset()
        self.brain.kill()
        return self

    def killcam_replay(self):
        return getattr(self, "_replay", None)

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


# --- 3.0 day 4: network science, tournaments, racing, sleep deprivation, sensitivity -------------------------------------------
def network_science(brain: str = "adult", **kw) -> dict:
    """Degree distributions, reciprocity, 3-node motifs against a degree-preserving null, rich club, communities and per-region
    summaries of the brain pack (CONNECTOME: computed from the wiring; the analysis choices are GAME RULE). Cached on disk with a
    checksum; adult takes minutes the first time. kw: nulls, wedges, seed, progress, force, cache (lab/netsci.py)."""
    from kickthefly.lab import netsci

    return netsci.compute(brain, **kw)


def tournament(seeds, **kw) -> dict:
    """A bracket of 4, 8 or 16 flies (one individuality seed each) in 1v1 duels where both sides are brains. Returns the bracket,
    each fly's measured personality card, the champion's drivers, and (key "analysis") whether personality predicted winning.
    kw: seconds, mode, favorite, workers, bracket_seed (lab/tournament.py). MODEL PREDICTION."""
    from kickthefly.lab import tournament as t

    b = t.run_bracket(seeds, **kw)
    b["analysis"] = t.analyze([b])
    return b


def race(seeds, **kw) -> dict:
    """Flies race down a track with sugar and fruit lures, each through its own brain; the odds come from the measured personality
    card (points only, never money). kw: mode, repeats, workers, race_seed (lab/racing.py). MODEL PREDICTION."""
    from kickthefly.lab import racing

    return racing.run_race(seeds, **kw)


def sleep_deprivation(seeds=range(1000, 1010), **kw) -> dict:
    """Keep each fly awake through part of the night (GAME RULE pressure and disturbances) and measure rebound sleep against its own
    undisturbed control; the dFB readout is the CONNECTOME's (lab/sleepdep.py). kw: mode, workers, deprive_s, recover_s."""
    from kickthefly.lab import sleepdep

    return sleepdep.run(seeds, **kw)


def sensitivity(**kw) -> dict:
    """Vary each LIF parameter across its documented range and re-run the validated behaviors with validation's own criteria
    (lab/sensitivity.py). Analysis only. kw: params, tests, seeds, values, workers, folder, resume, progress."""
    from kickthefly.lab import sensitivity as s

    return s.run(**kw)
