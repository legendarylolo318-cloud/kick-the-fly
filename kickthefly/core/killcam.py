"""Kill cam (3.0): on death, a slow-motion replay of the last ~6 s of the fly's brain, with the neurons whose firing rose
most before death highlighted.

No pygame here: the game draws it, the tests and the Python API read it.

CONNECTOME   what is replayed and highlighted: every neuron's own firing rate (the simulator's per-neuron rate estimate,
             a 200 ms exponential average of its spikes, sampled every SAMPLE_STEPS steps) for the last WINDOW_S seconds
             of the brain's life, exactly as the live brain panel showed it. "Rose most" is each neuron's mean rate
             over the last TAIL_S seconds before death minus its mean over the first BASE_S seconds of the window.
GAME RULE    the offer itself (Settings > Brain > Kill cam offer), the window length, the slow-motion speed, how many
             neurons are highlighted (TOP_N), the minimum rise that counts (MIN_RISE_HZ), the pacing and the Skip key.
             Death itself is a game rule (docs: a sim can't die on its own; the drive is cancelled over 1.5 s), so the
             cut-off at the moment of death is the game's, and the replay stops at it.
Nothing is estimated beyond the rates the simulator already keeps, nothing is fed back into the simulation, and the replay
never touches the live brain: it is frames in memory. A hit is shown only as what was recorded; the body is not re-simulated.

Reduced flashing (Settings > Accessibility): no pulsing rings, no flashes, no sparkles, a steady highlight, and a slower
default speed; the game applies that, this module only exposes `reduced` to the player.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WINDOW_S = 6.0               # GAME RULE: seconds of life replayed
SAMPLE_STEPS = 8             # sim steps between samples (40 ms, 25 per second)
DT = 0.005                   # seconds per sim step
BASE_S, TAIL_S = 1.5, 1.5    # baseline window (start of the buffer) and tail window (end), seconds
TOP_N = 12                   # GAME RULE: neurons highlighted
MIN_RISE_HZ = 5.0            # GAME RULE: a neuron must have risen at least this much to be listed
SPEED = 0.25                 # GAME RULE: default slow motion (x real time)
SPEED_REDUCED = 0.2          # slower again when Reduced flashing is on
MAX_HZ = 255.0               # rates are stored as uint8 on a square-root scale up to this


def encode_rates(rates_per_step: np.ndarray, dt: float = DT) -> np.ndarray:
    """Per-neuron rates (spikes per step) as uint8: sqrt-companded Hz, fine near calm rates (~0.2 Hz at 10 Hz)."""
    hz = np.clip(np.asarray(rates_per_step, np.float32) / dt, 0.0, MAX_HZ)
    return np.rint(np.sqrt(hz / MAX_HZ) * 255.0).astype(np.uint8)


def decode_hz(q: np.ndarray) -> np.ndarray:
    return (MAX_HZ * (q.astype(np.float32) / 255.0) ** 2).astype(np.float32)


class Buffer:
    """A ring of the last WINDOW_S seconds of per-neuron rates for one brain, plus small per-sample extras (the game puts
    what it wants drawn alongside, such as the body's points). Feed it `push(step, rates)` every frame; it keeps one
    sample per SAMPLE_STEPS brain steps, so slow motion and pauses (the brain doesn't step) add nothing."""

    def __init__(self, n_neurons: int, window_s: float = WINDOW_S, sample_steps: int = SAMPLE_STEPS):
        self.n = int(n_neurons)
        self.sample_steps = sample_steps
        self.cap = int(round(window_s / (sample_steps * DT)))
        self.q = np.zeros((self.cap, self.n), np.uint8)
        self.steps = np.zeros(self.cap, np.int64)
        self.extras: list = [None] * self.cap
        self.count = 0                 # samples written in total
        self.last_step: int | None = None

    def reset(self) -> None:
        self.count = 0
        self.last_step = None

    def __len__(self) -> int:
        return min(self.count, self.cap)

    def push(self, step: int, rates_per_step: np.ndarray, extra=None) -> bool:
        """Store a sample if the brain has advanced SAMPLE_STEPS since the last one. Returns True if it stored one. A step
        count that goes backwards (a save state was loaded, the fly was replaced) clears the buffer first."""
        if self.last_step is not None:
            if step < self.last_step:
                self.reset()
            elif step - self.last_step < self.sample_steps:
                return False
        i = self.count % self.cap
        self.q[i] = encode_rates(rates_per_step)
        self.steps[i] = step
        self.extras[i] = extra
        self.count += 1
        self.last_step = step
        return True

    def freeze(self) -> "Replay | None":
        """The buffered samples, oldest first, as an independent Replay. None if there are too few to show anything."""
        n = len(self)
        if n < 8:
            return None
        order = (np.arange(self.count - n, self.count)) % self.cap
        return Replay(self.q[order].copy(), self.steps[order].copy(), [self.extras[i] for i in order])


@dataclass
class Replay:
    q: np.ndarray                      # (frames, neurons) uint8
    steps: np.ndarray                  # (frames,) the brain step of each frame
    extras: list = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.q)

    @property
    def seconds(self) -> float:
        return (self.n_frames - 1) * SAMPLE_STEPS * DT if self.n_frames > 1 else 0.0

    def hz(self, frame: int) -> np.ndarray:
        return decode_hz(self.q[int(np.clip(frame, 0, self.n_frames - 1))])

    def rates_per_step(self, frame: int) -> np.ndarray:
        """The frame as spikes per step, which is what BrainView.render takes."""
        return self.hz(frame) * np.float32(DT)

    def top_risers(self, k: int = TOP_N, min_rise_hz: float = MIN_RISE_HZ) -> list[dict]:
        """The neurons whose firing rose most before death, biggest first: [{index, rise_hz, before_hz, after_hz}]. The
        baseline is the first BASE_S of the window and the tail the last TAIL_S, both clipped to a third of the window
        for a short replay. Ties break by neuron index, so the answer is deterministic."""
        n = self.n_frames
        per_s = 1.0 / (SAMPLE_STEPS * DT)
        nb = int(np.clip(round(BASE_S * per_s), 1, max(1, n // 3)))
        nt = int(np.clip(round(TAIL_S * per_s), 1, max(1, n // 3)))
        before = decode_hz(self.q[:nb]).mean(axis=0)
        after = decode_hz(self.q[-nt:]).mean(axis=0)
        rise = after - before
        order = np.lexsort((np.arange(len(rise)), -rise))
        out = []
        for i in order[:k]:
            if rise[i] < min_rise_hz:
                break
            out.append(dict(index=int(i), rise_hz=float(rise[i]), before_hz=float(before[i]), after_hz=float(after[i])))
        return out

    def traces(self, indices) -> np.ndarray:
        """(len(indices), frames) firing rates in Hz for a few neurons, for the replay's timeline."""
        return decode_hz(self.q[:, list(indices)]).T

    def summary(self, brain_types, brain_super=None, k: int = TOP_N) -> dict:
        """The risers with their cell types (and superclass), and the types that appear most among them."""
        risers = self.top_risers(k)
        types = np.asarray(brain_types).astype(str)
        for r in risers:
            r["type"] = str(types[r["index"]]) or "(no type)"
            if brain_super is not None:
                r["superclass"] = str(brain_super[r["index"]])
        counts: dict[str, int] = {}
        for r in risers:
            counts[r["type"]] = counts.get(r["type"], 0) + 1
        top_types = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return {"risers": risers, "types": top_types, "seconds": self.seconds, "frames": self.n_frames}


class Player:
    """The playback clock. `advance(wall_dt)` moves the replay by wall_dt * speed seconds of brain time; `frame` is the
    frame to show; `done` becomes True at the last frame (the moment of death). Skippable at any time."""

    def __init__(self, replay: Replay, speed: float = SPEED):
        self.replay = replay
        self.speed = float(speed)
        self.t = 0.0
        self.done = False
        self.skipped = False

    @property
    def frame(self) -> int:
        return min(self.replay.n_frames - 1, int(self.t / (SAMPLE_STEPS * DT)))

    @property
    def progress(self) -> float:
        return 0.0 if self.replay.seconds <= 0 else min(1.0, self.t / self.replay.seconds)

    @property
    def wall_seconds(self) -> float:
        return self.replay.seconds / max(self.speed, 1e-6)

    def advance(self, wall_dt: float) -> None:
        if self.done:
            return
        self.t += max(0.0, wall_dt) * self.speed
        if self.t >= self.replay.seconds:
            self.t = self.replay.seconds
            self.done = True

    def skip(self) -> None:
        self.skipped = True
        self.done = True

    def restart(self, speed: float | None = None) -> None:
        self.t, self.done, self.skipped = 0.0, False, False
        if speed is not None:
            self.speed = float(speed)
