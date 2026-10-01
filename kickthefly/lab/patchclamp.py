"""Virtual patch clamp: current-clamp one simulated neuron and read its membrane potential, spikes and I-F curve.

THIS IS A POINT-NEURON MODEL, NOT ELECTROPHYSIOLOGY. Every neuron in the simulation is the same leaky integrate-and-fire unit
(sim/connectome/sim.py): one membrane potential, one time constant, one threshold, reset after a spike. There is no dendrite, no
ion channel, no spike shape, no adaptation, no cell-specific intrinsic property. What differs between neurons is only what
they are wired to. So:

  MODEL        the membrane potential is in model units, where the firing threshold is 1.0 and the reset is 0.0 (resting
               level with the model's tonic drive 0.8). It is NOT in millivolts and is never converted to them. The injected
               current is in the simulator's current units (what `mode: drive` protocols and the laser use: 0.5 is the
               validation suite's activation current; the simulator multiplies external current by ext_gain = 4). Nothing
               here is calibrated against a recording of any real cell.
  CONNECTOME   in `embedded` mode the electrode sits on a neuron of the running, wired brain: its membrane potential includes
               the real synaptic input from the neurons that spike around it, plus the model's noise. Which neuron it is (type,
               body id) is the dataset's.
  GAME RULE    `isolated` mode: the same LIF equation with the synaptic input removed and only the model's own tonic drive and
               noise, run standalone for this one unit. Because every unit shares the same parameters, the isolated I-F curve
               is IDENTICAL FOR EVERY NEURON (up to noise). It shows the model, not the cell. The current-step protocol, the
               rest periods and the I-F amplitudes are choices of this tool.

The spike itself is not drawn as a waveform: the model resets the potential on the step it crosses threshold, so a spike is a
marker on that step. The simulation steps at 5 ms, so the trace has a 5 ms resolution.

Exports: CSV (time, membrane potential, current, spike) and NWB (a TimeSeries in model units, plus a Units table of spike
times; it is not written as a volts CurrentClampSeries on purpose, because it is not volts).

    from kickthefly.lab import patchclamp as pc
    rec = pc.run_current_clamp(br, row, pc.ClampProtocol(amplitudes=[0.0, 0.05], duration_ms=500))
    curve = pc.if_curve(br, row, [0.0, 0.05, 0.1, 0.2])
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DT_MS = 5.0
SOURCE = "patch"
MODES = ("embedded", "isolated")
UNITS_NOTE = ("model units: the firing threshold is 1.0 and the reset is 0.0; the tonic drive alone rests at "
              "bias/leak = 0.8. Not millivolts.")
TAG_TEXT = "MODEL: a point-neuron leaky integrate-and-fire unit. Not real electrophysiology."


class PatchError(ValueError):
    pass


@dataclass
class ClampProtocol:
    """A train of current steps. Each sweep is pre_ms at the holding current, a step of `amplitude` for duration_ms, then
    post_ms back at the holding current; sweeps repeat `repeats` times per amplitude, gap_ms apart (at the holding current).
    Amplitudes are in the simulator's current units."""
    amplitudes: list = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.2, 0.5])
    duration_ms: float = 500.0
    pre_ms: float = 100.0
    post_ms: float = 100.0
    repeats: int = 1
    gap_ms: float = 100.0
    holding: float = 0.0

    def __post_init__(self):
        self.amplitudes = [float(a) for a in self.amplitudes]
        if not self.amplitudes or len(self.amplitudes) > 200:
            raise PatchError("give between 1 and 200 current amplitudes")
        if any(abs(a) > 5.0 for a in self.amplitudes) or abs(self.holding) > 5.0:
            raise PatchError("currents are limited to +-5 (the validation suite's activation current is 0.5)")
        for name, lo, hi in (("duration_ms", 10, 20000), ("pre_ms", 0, 20000), ("post_ms", 0, 20000), ("gap_ms", 0, 20000)):
            v = float(getattr(self, name))
            if not lo <= v <= hi:
                raise PatchError(f"{name} must be between {lo} and {hi} ms")
            setattr(self, name, v)
        self.repeats = int(self.repeats)
        if not 1 <= self.repeats <= 100:
            raise PatchError("repeats must be between 1 and 100")

    def sweeps(self) -> list[float]:
        return [a for a in self.amplitudes for _ in range(self.repeats)]

    def steps(self, ms: float) -> int:
        return int(round(ms / DT_MS))

    def total_steps(self) -> int:
        per = self.steps(self.pre_ms) + self.steps(self.duration_ms) + self.steps(self.post_ms) + self.steps(self.gap_ms)
        return per * len(self.sweeps())

    def schedule(self) -> np.ndarray:
        """Injected current at every step of the whole protocol."""
        parts = []
        for a in self.sweeps():
            parts += [np.full(self.steps(self.pre_ms), self.holding), np.full(self.steps(self.duration_ms), a),
                      np.full(self.steps(self.post_ms), self.holding), np.full(self.steps(self.gap_ms), self.holding)]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)

    def sweep_windows(self) -> list[tuple[float, int, int]]:
        """(amplitude, first step of the current step, last+1) for each sweep."""
        out, t = [], 0
        for a in self.sweeps():
            t += self.steps(self.pre_ms)
            out.append((a, t, t + self.steps(self.duration_ms)))
            t += self.steps(self.duration_ms) + self.steps(self.post_ms) + self.steps(self.gap_ms)
        return out


@dataclass
class Recording:
    """What the electrode recorded. v is the potential after each step (spike steps read the reset value, 0)."""
    v: np.ndarray
    spikes: np.ndarray
    current: np.ndarray
    mode: str
    row: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def t_ms(self) -> np.ndarray:
        return np.arange(len(self.v)) * DT_MS

    def spike_times_ms(self) -> np.ndarray:
        return np.flatnonzero(self.spikes) * DT_MS


# --- the isolated unit: the simulator's own LIF update with no synaptic input ---------------------------------------------
def isolated(amps_per_step: np.ndarray, params=None, seed: int = 0, settle_steps: int = 400) -> Recording:
    """One LIF unit run standalone. The arithmetic mirrors the reference CPU backend step (backends.CPUBackend.step) with
    i_syn = 0: drive = bias + noise + ext_gain * current; v = v * (1 - leak) + drive; refractory units sit at reset; a unit at or
    over threshold spikes and resets. A `settle_steps` run first (no current) gets it past the start-up."""
    from kickthefly.sim.connectome.sim import LIFParams

    p = params or LIFParams()
    dtype = np.float64 if getattr(p, "dtype", "float32") == "float64" else np.float32
    leak = dtype(p.dt_ms / p.tau_ms)
    rng = np.random.default_rng(seed)
    n = int(settle_steps) + len(amps_per_step)
    noise = rng.standard_normal(n).astype(dtype) * dtype(p.noise_std)
    cur = np.concatenate([np.zeros(int(settle_steps), np.float32), np.asarray(amps_per_step, np.float32)])
    v, refr = dtype(0.0), 0
    vs, sp = np.zeros(n, dtype), np.zeros(n, bool)
    for i in range(n):
        drive = dtype(p.bias) + noise[i] + dtype(p.ext_gain) * dtype(cur[i])
        v = v * dtype(1.0 - leak)
        if p.v_reset:
            v = v + leak * dtype(p.v_reset)
        v = v + drive
        if refr > 0:
            v = dtype(p.v_reset)
            refr -= 1
        if v >= p.v_thresh:
            sp[i] = True
            v = dtype(p.v_reset)
            refr = int(p.refractory_steps)
        vs[i] = v
    k = int(settle_steps)
    return Recording(vs[k:].astype(np.float32), sp[k:], cur[k:], "isolated", None,
                     dict(seed=seed, parameters=dict(tau_ms=p.tau_ms, dt_ms=p.dt_ms, v_thresh=p.v_thresh, v_reset=p.v_reset,
                                                     bias=p.bias, noise_std=p.noise_std, ext_gain=p.ext_gain,
                                                     refractory_steps=p.refractory_steps)))


def embedded(br, row: int, amps_per_step: np.ndarray, settle_steps: int = 0) -> Recording:
    """The electrode on neuron `row` of a running (lockstep) brain: the brain keeps stepping with all its synaptic input while
    the current is injected into that one neuron (added to whatever else drives the brain, as a named source)."""
    row = int(row)
    if not 0 <= row < br.n:
        raise PatchError(f"neuron {row} is outside this brain (0-{br.n - 1})")
    for _ in range(int(settle_steps)):
        br._step()
    n = len(amps_per_step)
    vs, sp = np.zeros(n, np.float32), np.zeros(n, bool)
    last = None
    try:
        for i in range(n):
            a = float(amps_per_step[i])
            if a != last:
                if a == 0.0:
                    br.clear_current(SOURCE)
                else:
                    br.set_current(SOURCE, [row], a)
                last = a
            br._step()
            vs[i] = br.sim.v[row]
            sp[i] = bool(br.sim.spikes[row])
    finally:
        br.clear_current(SOURCE)
    return Recording(vs, sp, np.asarray(amps_per_step, np.float32), "embedded", row,
                     dict(neuron=describe_neuron(br, row)))


def describe_neuron(br, row: int) -> dict:
    g = getattr(br, "graph", None)
    body = None
    if g is not None and getattr(g, "body_id", None) is not None:
        body = int(g.body_id[row])
    return dict(row=int(row), type=str(br.types[row]), superclass=str(br.superclass[row]),
                instance=str(br.instance[row]) if len(br.instance) > row else "", body_id=body,
                region=str(getattr(g, "region", [""] * (row + 1))[row]) if g is not None else "")


def pick_neuron(br, spec: str, index: int = 0) -> int:
    """The index-th neuron (by row order) of a neuron spec: a type, `line:...`, a group name."""
    from kickthefly.core import simcore

    try:
        rows = np.sort(np.asarray(simcore.rows_of(br, spec), np.int64))
    except ValueError as e:
        raise PatchError(str(e)) from None
    if not len(rows):
        raise PatchError(f"{spec!r} selects no neurons")
    if not 0 <= int(index) < len(rows):
        raise PatchError(f"{spec!r} has {len(rows)} neurons; index {index} is out of range")
    return int(rows[int(index)])


def run_current_clamp(br, row: int | None, proto: ClampProtocol, mode: str = "embedded", seed: int = 0,
                      params=None) -> Recording:
    if mode not in MODES:
        raise PatchError(f"mode must be one of {MODES}")
    sched = proto.schedule()
    rec = embedded(br, row, sched) if mode == "embedded" else isolated(sched, params or getattr(getattr(br, "sim", None), "p", None), seed)
    if mode == "isolated" and row is not None:
        rec.row = int(row)
    rec.meta["protocol"] = dict(amplitudes=proto.amplitudes, duration_ms=proto.duration_ms, pre_ms=proto.pre_ms,
                                post_ms=proto.post_ms, repeats=proto.repeats, gap_ms=proto.gap_ms, holding=proto.holding)
    rec.meta["tag"] = TAG_TEXT
    rec.meta["units"] = UNITS_NOTE
    return rec


def if_curve(br, row: int | None, amplitudes, duration_ms: float = 500.0, repeats: int = 3, mode: str = "embedded",
             seed: int = 0, params=None) -> dict:
    """Firing rate (spikes/s over the step) against injected current. Returns rates per sweep and their mean per amplitude."""
    proto = ClampProtocol(list(amplitudes), duration_ms=duration_ms, repeats=repeats)
    rec = run_current_clamp(br, row, proto, mode, seed, params)
    wins = proto.sweep_windows()
    per = {}
    for a, lo, hi in wins:
        per.setdefault(a, []).append(float(rec.spikes[lo:hi].sum()) / max(1e-9, (hi - lo) * DT_MS / 1000.0))
    amps = sorted(per)
    rates = [float(np.mean(per[a])) for a in amps]
    sds = [float(np.std(per[a], ddof=1)) if len(per[a]) > 1 else float("nan") for a in amps]
    return dict(amplitudes=amps, rate_hz=rates, rate_sd=sds, sweeps=per, mode=mode, row=row, recording=rec,
                duration_ms=duration_ms, repeats=repeats,
                rheobase=next((a for a, r in zip(amps, rates) if r > 0 and a > 0), None),
                tag=TAG_TEXT, units=UNITS_NOTE)


# --- live electrode (the game's running brain) ------------------------------------------------------------------------------
class LiveElectrode:
    """Attached to a game brain: records `window_steps` of the neuron's potential and spikes from the brain's own thread
    (one sample per 5 ms step), and can play a current-clamp protocol into it. Detach when done."""

    def __init__(self, br, row: int, window_steps: int = 600):
        self.br, self.row = br, int(row)
        self.n = int(window_steps)
        self.v = np.zeros(self.n, np.float32)
        self.spikes = np.zeros(self.n, bool)
        self.i = np.zeros(self.n, np.float32)
        self.count = 0
        self.proto: ClampProtocol | None = None
        self._sched: np.ndarray | None = None
        self._pos = 0
        self.hold = 0.0
        self._last = None
        self.log: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.done = False

    def attach(self) -> "LiveElectrode":
        self.br.probe = self._push
        return self

    def detach(self) -> None:
        if getattr(self.br, "probe", None) == self._push:
            self.br.probe = None
        self.br.clear_current(SOURCE)

    def set_hold(self, amp: float) -> None:
        self.hold = float(amp)
        self._inject(self.hold if self._sched is None else self._last)

    def run(self, proto: ClampProtocol) -> None:
        self.proto, self._sched, self._pos, self.done = proto, proto.schedule(), 0, False
        self.log = []
        self._buf_v, self._buf_s, self._buf_i = [], [], []

    def _inject(self, a) -> None:
        if a != self._last:
            if a == 0.0:
                self.br.clear_current(SOURCE)
            else:
                self.br.set_current(SOURCE, [self.row], float(a))
            self._last = a

    def _push(self, br) -> None:
        """Runs on the brain's thread after every step."""
        k = self.count % self.n
        self.v[k], self.spikes[k] = br.sim.v[self.row], bool(br.sim.spikes[self.row])
        self.i[k] = 0.0 if self._last is None else self._last
        self.count += 1
        if self._sched is not None:
            self._buf_v.append(self.v[k]); self._buf_s.append(self.spikes[k]); self._buf_i.append(self.i[k])
            self._pos += 1
            if self._pos >= len(self._sched):
                self.recording = Recording(np.array(self._buf_v, np.float32), np.array(self._buf_s), np.array(self._buf_i, np.float32),
                                           "embedded", self.row, dict(neuron=describe_neuron(br, self.row), tag=TAG_TEXT,
                                                                      units=UNITS_NOTE, live=True))
                self._sched, self.done = None, True
                self._inject(self.hold)
            else:
                self._inject(float(self._sched[self._pos]))

    def window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(v, spikes, current) for the last `window_steps` steps in time order."""
        k = self.count % self.n
        order = np.r_[k:self.n, 0:k] if self.count >= self.n else np.arange(self.count)
        return self.v[order].copy(), self.spikes[order].copy(), self.i[order].copy()


# --- exports ------------------------------------------------------------------------------------------------------------
def export_csv(rec: Recording, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {TAG_TEXT}\n# membrane_potential is in {UNITS_NOTE}\n# mode={rec.mode} row={rec.row} "
                f"neuron={rec.meta.get('neuron')}\n")
        w = csv.writer(f)
        w.writerow(["time_ms", "membrane_potential_model_units", "injected_current_model_units", "spike"])
        for t, v, i, s in zip(rec.t_ms, rec.v, rec.current, rec.spikes):
            w.writerow([f"{t:.1f}", f"{v:.6g}", f"{i:.6g}", int(s)])
    return path


def export_if_csv(curve: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {TAG_TEXT}\n# mode={curve['mode']} row={curve['row']} step={curve['duration_ms']} ms, "
                f"{curve['repeats']} sweeps per amplitude; current in the simulator's units\n")
        w = csv.writer(f)
        w.writerow(["current", "rate_hz_mean", "rate_hz_sd", "sweeps_hz"])
        for a, r, sd in zip(curve["amplitudes"], curve["rate_hz"], curve["rate_sd"]):
            w.writerow([f"{a:g}", f"{r:.4g}", "" if sd != sd else f"{sd:.4g}", " ".join(f"{x:.4g}" for x in curve["sweeps"][a])])
    return path


def nwb_available() -> str | None:
    from kickthefly.lab import nwbexport

    return nwbexport.available()


def export_nwb(rec: Recording, path: Path) -> Path:
    """NWB with the potential and current as generic TimeSeries in model units (not a volts CurrentClampSeries) and the spikes
    as a Units table. Needs pynwb."""
    reason = nwb_available()
    if reason:
        raise PatchError(reason)
    import uuid
    from datetime import datetime, timezone

    from pynwb import NWBFile, NWBHDF5IO, TimeSeries

    from kickthefly.core.version import __version__

    nwb = NWBFile(session_description=f"{TAG_TEXT} Simulated current-clamp of one point neuron ({rec.mode}); the data are model "
                                      f"output, not a recording from an animal. {UNITS_NOTE}",
                  identifier=str(uuid.uuid4()), session_start_time=datetime.now(timezone.utc),
                  experiment_description="Kick the Fly virtual patch clamp", lab="Kick the Fly (simulation)",
                  source_script=f"kickthefly {__version__}", source_script_file_name="kickthefly/lab/patchclamp.py")
    t0 = 0.0
    rate = 1000.0 / DT_MS
    nwb.add_acquisition(TimeSeries(name="membrane_potential_model_units", data=rec.v.astype(np.float32), unit="a.u.",
                                   starting_time=t0, rate=rate,
                                   description="MODEL membrane potential of the LIF unit, threshold = 1.0, reset = 0.0. Not volts."))
    nwb.add_stimulus(TimeSeries(name="injected_current_model_units", data=rec.current.astype(np.float32), unit="a.u.",
                                starting_time=t0, rate=rate,
                                description="Injected current in the simulator's units (ext_gain 4 applies)."))
    nwb.add_unit_column("neuron", "what the electrode was on")
    nwb.add_unit(spike_times=rec.spike_times_ms() / 1000.0, neuron=str(rec.meta.get("neuron", rec.mode)))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return Path(path)
