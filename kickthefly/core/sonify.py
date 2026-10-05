"""Brain sonification (3.1.0 task 14, opt-in audio): each brain region is a voice whose pitch and volume follow how active it is, mixed so that it is music-like rather than noise,
plus a courtship song synthesized from the song readout when it fires. Pure numpy: no pygame, no brain, no game (game/sonify_play.py plays it).

What is what:
  CONNECTOME   the activity it listens to: the simulation's own smoothed firing rate of every neuron in each region (`brain.sim.activity.rates()`, the numbers the brain panel shows),
               and the song readout, the ps1 wing motor neurons that pIP10 and P1 drive (the game's own SONG signal, `Brain.level("song")`).
  GAME RULE    everything that makes it sound: which region is which voice and its note, the minor pentatonic scale, how a rate relative to the region's own slow baseline becomes
               loudness and a step up the scale, the glides, the soft limiter, and the song (a 220 Hz carrier in 12 ms pulses 35 ms apart, the game's existing song buzz; real
               *Drosophila melanogaster* pulse song has about that carrier and interval, but this is a rendering of the model's output, not a recording and not a prediction).
  It reads the brain and never writes to it, so it cannot change a spike.

Why it is not noise: every voice sits on one scale (D minor pentatonic), so any mix of them is consonant; pitch moves only between scale degrees, with a glide; loudness is a compressed
function of activity with an envelope of tens of milliseconds, so there are no clicks; the sum is normalised by the voices that are loud and passed through a soft limiter. A silent region
is silent (a faint presence when it is calm and alive, none when it is not firing at all).
"""
from __future__ import annotations

import math

import numpy as np

RATE = 22050
CHUNK_S = 0.1
SCALE_ROOT = 38                                # D2: the scale's tonic (MIDI)
SCALE = (0, 3, 5, 7, 10)                       # semitones: a minor pentatonic scale (D F G A C from a D root)
MAX_STEPS = 6                                  # scale degrees a voice can climb above its root
PRESENCE = 0.16                                # a calm, firing region's loudness (0..1) before it is stirred up
LOUD_GAIN = 1.1                                # how fast loudness rises with the log of the rate ratio
GLIDE_S = 0.09                                 # a pitch change glides in about this long
ATTACK_S, RELEASE_S = 0.04, 0.25               # loudness follows its target this fast up, and this slowly down
BASE_TAU_S = 45.0                              # a region's own baseline is its rate averaged over about this long
MIN_BASE_HZ = 0.05                             # a region quieter than this is not "alive": its voice is silent
SONG_CARRIER_HZ, SONG_PULSE_S, SONG_IPI_S = 220.0, 0.012, 0.035         # the game's own song buzz (kick_the_fly.Sound)
SONG_THRESH = 1.8                              # the game's THRESH["song"]: where its SONG reaction starts
SONG_OFF_RATIO = 0.7                           # the song stops below this share of the threshold (so it does not stutter on the edge)
SONG_GAIN = 0.55
DUCK = 0.55                                    # the voices are turned down this much under the song so it can be heard
OUT_PEAK = 0.8

# name, the note a calm voice sits on (MIDI), its overtones (relative amplitude of harmonics 1, 2, 3)
VOICES = (
    ("Descending neurons", 38, (1.0, 0.35, 0.10)),     # D2: the low end, the motor commands on their way down
    ("Nerve cord (VNC)", 45, (1.0, 0.30, 0.08)),       # A2
    ("Gnathal ganglion", 50, (1.0, 0.25, 0.05)),       # D3
    ("Central brain", 53, (1.0, 0.20, 0.0)),           # F3
    ("Mushroom body", 57, (1.0, 0.15, 0.05)),          # A3
    ("Central complex", 62, (1.0, 0.10, 0.0)),         # D4
    ("Antennal lobe", 67, (1.0, 0.08, 0.0)),           # G4
    ("Optic lobe", 69, (1.0, 0.05, 0.0)),              # A4: the high end
)
VOICE_NAMES = tuple(v[0] for v in VOICES)
# which of the brain view's neuropil names (lower case, a part of the name) feeds which voice; the descending voice uses the descending neurons wherever they sit
NEUROPIL_FOR = {"Nerve cord (VNC)": ("vnc",), "Gnathal ganglion": ("gnathal", "gng"), "Central brain": ("central brain",), "Mushroom body": ("mushroom",),
                "Central complex": ("central complex",), "Antennal lobe": ("antennal",), "Optic lobe": ("optic",)}
DESCENDING = ("descending_neuron", "descending_neuron_tbc", "efferent_descending", "sensory_descending")


def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def degree_semitones(step: int) -> int:
    """Semitones above the root of scale degree `step` (0 = the root, 5 = an octave up) of the pentatonic scale."""
    o, d = divmod(int(step), len(SCALE))
    return 12 * o + SCALE[d]


def voice_midi(root: int, step: int) -> int:
    """The note a voice sounds `step` degrees above its root, climbing the one scale every voice shares (D minor pentatonic), so any mix is consonant."""
    d0 = next((i for i in range(len(SCALE) * 8) if SCALE_ROOT + degree_semitones(i) >= root), 0)       # the first scale degree at or above the root
    return SCALE_ROOT + degree_semitones(d0 + int(step))


def activity_octaves(rate_hz: float, base_hz: float) -> float:
    """How far a region's rate is from its own baseline, in octaves (doublings), clipped to -1..4. 0 = calm."""
    if base_hz < MIN_BASE_HZ:
        return -1.0
    return float(np.clip(math.log2(max(rate_hz, 1e-3) / base_hz), -1.0, 4.0))


def voice_target(rate_hz: float, base_hz: float) -> tuple[int, float]:
    """(scale degree above the voice's root, loudness 0..1) for a region at `rate_hz` against its `base_hz`. Louder and higher as it is stirred up; a calm one
    sits on its root at PRESENCE; one that is not firing at all is silent."""
    if base_hz < MIN_BASE_HZ and rate_hz < MIN_BASE_HZ:
        return 0, 0.0
    x = activity_octaves(rate_hz, max(base_hz, MIN_BASE_HZ))
    up = max(0.0, x)
    step = int(min(MAX_STEPS, round(2.0 * up)))
    loud = PRESENCE + (1.0 - PRESENCE) * (1.0 - math.exp(-LOUD_GAIN * up))
    if x < 0:
        loud = PRESENCE * (1.0 + x)                     # below its baseline it fades toward silence at half the baseline
    return step, float(min(1.0, max(0.0, loud)))


# --- which neurons feed which voice ---------------------------------------------------------------------------------------------------------
def voice_masks(region_names, region_id: np.ndarray, superclass: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """{voice name: neuron indices} for the voices this brain has (at least 20 neurons). `region_names` and `region_id` are the brain view's (the adult's neuropils); the
    descending voice is the descending neurons by superclass. A brain whose regions match none of the names (the larva's) gets a voice per region with 50 or more neurons, in the order
    of VOICES from the top."""
    names = [str(n).lower() for n in region_names]
    out: dict[str, np.ndarray] = {}
    for voice, keys in NEUROPIL_FOR.items():
        ids = [i for i, n in enumerate(names) if any(k in n for k in keys)]
        if ids:
            idx = np.flatnonzero(np.isin(region_id, ids))
            if len(idx) >= 20:
                out[voice] = idx.astype(np.int64)
    if superclass is not None:
        idx = np.flatnonzero(np.isin(np.asarray(superclass).astype(str), DESCENDING))
        if len(idx) >= 20:
            out["Descending neurons"] = idx.astype(np.int64)
    if len(out) < 3:                                  # not the adult's neuropils: use what there is
        out = {}
        counts = np.bincount(region_id, minlength=len(names))
        order = [i for i in np.argsort(-counts) if counts[i] >= 50][:len(VOICES)]
        for v, i in zip(VOICE_NAMES[::-1][:len(order)], order):
            out[v] = np.flatnonzero(region_id == i).astype(np.int64)
    return {v: out[v] for v in VOICE_NAMES if v in out}


class RegionMeter:
    """Turns the brain's per-neuron rates into each voice's (rate, own baseline). The baseline is the region's rate averaged slowly (BASE_TAU_S), started at the first reading."""

    def __init__(self, masks: dict[str, np.ndarray]):
        self.masks = masks
        self.names = list(masks)
        self.rate = np.zeros(len(self.names))
        self.base = np.zeros(len(self.names))
        self._seen = False

    def update(self, rates_hz: np.ndarray, dt: float) -> None:
        self.rate = np.array([float(rates_hz[idx].mean()) for idx in self.masks.values()])
        if not self._seen:
            self.base, self._seen = self.rate.copy(), True
        else:
            self.base += (self.rate - self.base) * (1.0 - math.exp(-max(dt, 0.0) / BASE_TAU_S))

    def targets(self) -> dict[str, tuple[int, float]]:
        return {n: voice_target(float(r), float(b)) for n, r, b in zip(self.names, self.rate, self.base)}


# --- the song readout -------------------------------------------------------------------------------------------------------------------------
class SongGate:
    """Whether the song is sounding, from the song readout's level (the game's `Brain.level("song")`, a ratio to its calm). On above the game's own SONG threshold, off below 70% of it."""

    def __init__(self, thresh: float = SONG_THRESH):
        self.thresh = thresh
        self.on = False
        self.level = 0.0

    def update(self, level: float) -> bool:
        self.level = float(level)
        if not self.on and level >= self.thresh:
            self.on = True
        elif self.on and level < self.thresh * SONG_OFF_RATIO:
            self.on = False
        return self.on

    def loudness(self) -> float:
        """0.4 at the threshold, up to 0.9 at twice it."""
        return float(np.clip(0.4 + 0.5 * (self.level / self.thresh - 1.0), 0.4, 0.9))


# --- synthesis ----------------------------------------------------------------------------------------------------------------------------------
class Synth:
    """Renders the mix in chunks with continuous phase. `set_targets` says where each voice is heading; `render(n)` returns float samples in -OUT_PEAK..OUT_PEAK."""

    def __init__(self, voices=VOICES, rate: int = RATE):
        self.rate = rate
        self.voices = [v for v in voices]
        k = len(self.voices)
        self.phase = np.zeros(k)
        self.freq = np.array([midi_hz(v[1]) for v in self.voices])
        self.amp = np.zeros(k)
        self.f_target = self.freq.copy()
        self.a_target = np.zeros(k)
        self.song_on = False
        self.song_amp = 0.0
        self._duck = 1.0
        self._pos = 0                                    # samples rendered, for the song's pulse clock
        self._next_pulse = 0
        self._tail = np.zeros(0)                         # the part of the last pulse that ran past the chunk
        n = int(SONG_PULSE_S * rate)
        tau = np.arange(n) / rate
        self._pulse = np.sin(2 * np.pi * SONG_CARRIER_HZ * tau) * np.sin(np.pi * tau / SONG_PULSE_S)

    def set_targets(self, targets: dict[str, tuple[int, float]]) -> None:
        for i, (name, root, _) in enumerate(self.voices):
            step, loud = targets.get(name, (0, 0.0))
            self.f_target[i] = midi_hz(voice_midi(root, step))
            self.a_target[i] = loud

    def set_song(self, on: bool, loudness: float = 0.0) -> None:
        self.song_on, self.song_amp = bool(on), float(loudness)

    def render(self, n: int) -> np.ndarray:
        t = (np.arange(n) + 1) / self.rate
        out = np.zeros(n)
        norm = 0.0
        for i, (_, _, harm) in enumerate(self.voices):
            f = self.f_target[i] + (self.freq[i] - self.f_target[i]) * np.exp(-t / GLIDE_S)
            rising = self.a_target[i] > self.amp[i]
            tau = ATTACK_S if rising else RELEASE_S
            a = self.a_target[i] + (self.amp[i] - self.a_target[i]) * np.exp(-t / tau)
            ph = self.phase[i] + np.cumsum(2 * np.pi * f / self.rate)
            wave = sum(h * np.sin((k + 1) * ph) for k, h in enumerate(harm) if h > 0) / sum(h for h in harm if h > 0)
            out += a * wave
            norm += float(np.mean(a)) ** 2
            self.phase[i] = float(ph[-1]) % (2 * np.pi)
            self.freq[i], self.amp[i] = float(f[-1]), float(a[-1])
        out /= math.sqrt(1.0 + norm)                      # more loud voices do not mean a louder mix
        duck_to = DUCK if self.song_on else 1.0
        d = duck_to + (self._duck - duck_to) * np.exp(-t / 0.08)
        self._duck = float(d[-1])
        out = out * d
        if self.song_on or len(self._tail):
            out += SONG_GAIN * self._song(n)
        out = np.tanh(1.3 * out) * OUT_PEAK                          # the soft limiter: never above OUT_PEAK
        return out

    def _song(self, n: int) -> np.ndarray:
        buf = np.zeros(n + len(self._pulse))
        buf[:len(self._tail)] = self._tail
        if self.song_on:
            ipi = int(SONG_IPI_S * self.rate)
            while self._next_pulse < self._pos + n:
                at = max(self._next_pulse - self._pos, 0)
                buf[at:at + len(self._pulse)] += self._pulse * self.song_amp
                self._next_pulse += ipi
        else:
            self._next_pulse = self._pos + n
        self._pos += n
        self._tail = buf[n:].copy()
        if not self.song_on and not np.any(np.abs(self._tail) > 1e-6):
            self._tail = np.zeros(0)
        return buf[:n]


def to_pcm(x: np.ndarray, channels: int = 1) -> np.ndarray:
    """int16 samples for pygame.sndarray.make_sound, repeated across channels."""
    a = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    return np.ascontiguousarray(np.repeat(a[:, None], 2, axis=1) if channels == 2 else a)


# --- the policy: when it may make a sound ---------------------------------------------------------------------------------------------------------
def silent_reason(*, enabled: bool, audio_ok: bool, muted: bool, master: float, volume: float, mic_on: bool = False, stream_on: bool = False,
                  allow_in_stream: bool = False, paused: bool = False) -> str | None:
    """Why the sonification is silent right now, or None if it may play. The order is the order of importance to the person."""
    if not enabled:
        return "off"
    if not audio_ok:
        return "no audio device"
    if muted:
        return "muted"
    if master * volume <= 0.001:
        return "volume is 0"
    if mic_on:
        return "the microphone is on (it would hear the speakers and drive the fly's hearing neurons)"
    if stream_on and not allow_in_stream:
        return "Streamer mode is on (Settings > Audio > Sonification in Streamer mode)"
    if paused:
        return "time is paused"
    return None
