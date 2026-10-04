"""Microphone -> Johnston's organ (3.0 day 3). OPT-IN, off by default, and it never keeps or sends what it hears.

Sound from the microphone is turned into current on the fly's real auditory Johnston's organ neurons, JO-A and JO-B.

What is what:
  CONNECTOME   which neurons: JO-A* (50 neurons) and JO-B* (88) are in the MaleCNS pack by those type names. Johnston's organ
               neurons are grouped A to E by where they project; A and B are the vibration (sound) sensitive groups and C and E
               the deflection (gravity, wind) ones: Kamikouchi et al. 2009, Nature 458:165 (doi:10.1038/nature07810, abstract
               read: "gravity- and sound-sensitive neurons differ in their response characteristics"). The game already
               drives JO-C/E with wind; this drives A/B and leaves C/E alone.
  LITERATURE   what A and B prefer, as stated in the introduction of Ishikawa et al. 2019 (Front. Physiol. 10:1552, doi
               10.3389/fphys.2019.01552; a research article, not a review), citing Matsuo et al. 2014 and Patella & Wilson 2018:
               "When measured separately, the JO-B neurons show a low frequency preference at about <100 Hz, while JO-A neurons
               preferentially respond to higher frequency", the two together "ranging from ~10 Hz up to ~1,000 Hz". Checked
               word for word in the full text (Europe PMC, PMC6960095) by the 3.0 day 3 review.
  GAME RULE    the whole transduction, every number: the two band-pass filters (B: 10-100 Hz, A: 100-1,000 Hz, from the
               ranges above), how loud a band must be to drive its neurons fully (FULL_RMS) and the noise gate (GATE_RMS), the
               current at full drive (MAX_CURRENT, the validation suite's activation current), the sensitivity setting, and
               that every neuron of a subgroup gets the same current. The real organ is a mechanical resonator on a vibrating
               antenna and each neuron has its own tuning; none of that is modelled. A microphone is not an antenna: air
               pressure is treated as if it were the antennal vibration.
  Not claimed  no fly sound response is predicted by this module.

Privacy, by construction: samples are analysed in memory in 256-sample chunks and discarded. Nothing here opens a file,
writes a file, or touches the network; the only things kept are the latest few numbers (the Reading below). tests/test_mic.py
checks both. The capture device is opened only between Mic.start() and Mic.stop(), only when the player turns it on in
Settings, only in the real game: never in --validate, assays, protocols or tests (the headless demo feeds a synthetic waveform
through Analyzer, not the device).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

RATE = 22050                    # capture sample rate asked for (the device may give another; Analyzer takes the real one)
CHUNK = 256                     # samples per analysis hop (11.6 ms at 22,050 Hz)
WINDOW = 1024                   # samples the spectrum is taken over (46 ms)
BAND_B = (10.0, 100.0)          # JO-B: low frequencies (GAME RULE, from the review's "<100 Hz")
BAND_A = (100.0, 1000.0)        # JO-A: higher frequencies, to the top of the stated range
FULL_RMS = 0.05                 # band RMS (full scale = 1.0) that drives a subgroup fully, at sensitivity 1.0 (GAME RULE)
GATE_RMS = 0.002                # below this band RMS the neurons get nothing (the room's hiss)
MAX_CURRENT = 0.5               # current at full drive: the activation current of every pathway test
PEAK_RANGE = (40.0, 2000.0)     # where the "frequency" readout looks for the loudest peak
JO_A_PREFIX, JO_B_PREFIX = "JO-A", "JO-B"
SENSITIVITY = (0.1, 10.0)
FILTER_ORDER = 4                # each band-pass is an 8th-order Butterworth: 200 Hz leaks into the JO-B drive by 7% (GAME RULE)


class MicError(RuntimeError):
    """The microphone can't be used; the message says why in plain words."""


@dataclass
class Reading:
    """The latest numbers. This is everything that is kept of what the microphone heard."""

    peak_hz: float = 0.0            # loudest frequency in PEAK_RANGE (0 if too quiet to tell)
    rms: float = 0.0                # whole-signal level, 0 to 1 of full scale
    rms_a: float = 0.0              # band levels
    rms_b: float = 0.0
    drive_a: float = 0.0            # current onto JO-A / JO-B (0 to MAX_CURRENT)
    drive_b: float = 0.0
    bins: tuple = ()                # 24 log-spaced spectrum levels, 0-1, for the hum test's graph
    t: float = 0.0                  # seconds of audio analysed so far


def bin_edges(n: int = 24) -> np.ndarray:
    return np.geomspace(PEAK_RANGE[0], PEAK_RANGE[1], n + 1)


def band_drive(rms: float, sensitivity: float = 1.0) -> float:
    """Current for a band with this RMS: 0 under the gate, rising linearly to MAX_CURRENT at FULL_RMS / sensitivity. GAME RULE."""
    r = float(rms) * float(sensitivity)
    if r < GATE_RMS:
        return 0.0
    return MAX_CURRENT * float(np.clip(r / FULL_RMS, 0.0, 1.0))


class Analyzer:
    """Turns a stream of samples into Readings: two band-pass filters for the JO-A and JO-B drive, an FFT for the frequency.
    Deterministic: the same samples give the same readings. The device and the synthetic demo both feed it."""

    def __init__(self, rate: float = RATE, sensitivity: float = 1.0):
        from scipy import signal

        self.rate = float(rate)
        self.sensitivity = float(np.clip(sensitivity, *SENSITIVITY))
        nyq = self.rate / 2.0
        self._sos_a = signal.butter(FILTER_ORDER, [BAND_A[0] / nyq, min(BAND_A[1], nyq * 0.95) / nyq], btype="band", output="sos")
        self._sos_b = signal.butter(FILTER_ORDER, [BAND_B[0] / nyq, BAND_B[1] / nyq], btype="band", output="sos")
        self._zi_a = np.zeros((self._sos_a.shape[0], 2))
        self._zi_b = np.zeros((self._sos_b.shape[0], 2))
        self._signal = signal
        self._buf = np.zeros(WINDOW, np.float32)
        self._win = np.hanning(WINDOW).astype(np.float32)
        self._freqs = np.fft.rfftfreq(WINDOW, 1.0 / self.rate)
        self._edges = bin_edges()
        self._t = 0.0
        self.reading = Reading()

    def push(self, samples: np.ndarray) -> Reading:
        """Analyse one chunk (float samples in -1..1). Returns the new Reading."""
        x = np.asarray(samples, np.float32).ravel()
        if not len(x):
            return self.reading
        a, self._zi_a = self._signal.sosfilt(self._sos_a, x, zi=self._zi_a)
        b, self._zi_b = self._signal.sosfilt(self._sos_b, x, zi=self._zi_b)
        rms_a, rms_b = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
        n = len(x)
        self._buf = np.concatenate([self._buf[n:], x[-WINDOW:]]) if n < WINDOW else x[-WINDOW:].copy()
        spec = np.abs(np.fft.rfft(self._buf * self._win)) / (WINDOW / 4.0)
        peak = 0.0
        sel = (self._freqs >= PEAK_RANGE[0]) & (self._freqs <= PEAK_RANGE[1])
        if sel.any() and float(spec[sel].max()) > GATE_RMS * 2:
            i = int(np.argmax(np.where(sel, spec, 0.0)))
            peak = self._parabolic(spec, i)
        bins = []
        for lo, hi in zip(self._edges[:-1], self._edges[1:]):
            m = (self._freqs >= lo) & (self._freqs < hi)
            bins.append(float(np.clip(spec[m].max() / 0.2, 0.0, 1.0)) if m.any() else 0.0)
        self._t += n / self.rate
        self.reading = Reading(peak_hz=peak, rms=float(np.sqrt(np.mean(x * x))), rms_a=rms_a, rms_b=rms_b,
                               drive_a=band_drive(rms_a, self.sensitivity), drive_b=band_drive(rms_b, self.sensitivity),
                               bins=tuple(bins), t=self._t)
        return self.reading

    def _parabolic(self, spec: np.ndarray, i: int) -> float:
        """The peak's frequency, refined between FFT bins."""
        if 0 < i < len(spec) - 1:
            a, b, c = float(spec[i - 1]), float(spec[i]), float(spec[i + 1])
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
            return float((i + float(np.clip(off, -0.5, 0.5))) * self.rate / WINDOW)
        return float(self._freqs[i])


# --- onto the brain (CONNECTOME: the neurons; GAME RULE: the equal current) ---------------------------------------------------------
def jo_rows(br) -> tuple[np.ndarray, np.ndarray]:
    """(JO-A rows, JO-B rows) in this brain, by type name. Empty arrays in a brain without them (the larva)."""
    t = np.asarray(br.types).astype(str)
    return (np.flatnonzero(np.char.startswith(t, JO_A_PREFIX)), np.flatnonzero(np.char.startswith(t, JO_B_PREFIX)))


SOURCE = "mic"                  # the name of the current source on the brain (Brain.set_current)


def apply(br, reading: Reading, rows: tuple | None = None) -> None:
    """Put a Reading's drive onto the brain's JO-A and JO-B neurons (replacing the last one). Safe from any thread."""
    a, b = rows if rows is not None else jo_rows(br)
    if not len(a) and not len(b):
        return
    idx = np.concatenate([a, b]).astype(np.int64)
    vals = np.concatenate([np.full(len(a), reading.drive_a), np.full(len(b), reading.drive_b)]).astype(np.float32)
    br.set_current(SOURCE, idx, vals)


def release(br) -> None:
    br.clear_current(SOURCE)


# --- the device ------------------------------------------------------------------------------------------------------------------------
def devices() -> list[str]:
    """Names of capture devices, without opening any. [] if there is no audio capture here."""
    try:
        from pygame._sdl2 import audio

        return list(audio.get_audio_device_names(True))
    except Exception:
        return []


def availability() -> tuple[bool, str]:
    """(usable, why not). Looks for a capture device; does not open it."""
    try:
        from pygame._sdl2 import audio  # noqa: F401
    except Exception as e:
        return False, f"this build of pygame has no audio capture ({e})"
    names = devices()
    if not names:
        return False, "no microphone was found (check that one is plugged in and the system lets this program use it)"
    return True, f"{len(names)} capture device{'s' if len(names) != 1 else ''}: {names[0]}"


class Mic:
    """The opt-in microphone. start() opens the capture device and runs Analyzer on every chunk in memory; stop() closes it.
    latest() is the only thing it gives out: a Reading. Nothing is recorded, saved or sent."""

    def __init__(self, sensitivity: float = 1.0, device: str | None = None):
        self.sensitivity = float(sensitivity)
        self.device = device
        self._dev = None
        self._an: Analyzer | None = None
        self._lock = threading.Lock()
        self._latest = Reading()
        self.error = ""
        self.chunks = 0

    @property
    def active(self) -> bool:
        return self._dev is not None

    def start(self) -> None:
        if self._dev is not None:
            return
        try:
            import pygame
            from pygame._sdl2 import audio

            if not pygame.get_init():
                pygame.init()
            self._an = Analyzer(RATE, self.sensitivity)

            def callback(_dev, buf):                       # runs on SDL's audio thread: analyse and forget
                try:
                    x = np.frombuffer(bytes(buf), dtype="<i2").astype(np.float32) / 32768.0
                    r = self._an.push(x)
                    with self._lock:
                        self._latest = r
                        self.chunks += 1
                except Exception as e:                      # never let an error cross into the audio thread's C code
                    self.error = str(e)

            dev = audio.AudioDevice(devicename=self.device, iscapture=True, frequency=RATE, audioformat=audio.AUDIO_S16,
                                    numchannels=1, chunksize=CHUNK, allowed_changes=0, callback=callback)
            self._dev = dev
            dev.pause(0)
        except Exception as e:
            self._dev = None
            raise MicError(f"couldn't open the microphone: {e}") from None

    def stop(self) -> None:
        dev, self._dev = self._dev, None
        if dev is not None:
            try:
                dev.pause(1)
                dev.close()
            except Exception:
                pass
        with self._lock:
            self._latest = Reading()

    def set_sensitivity(self, s: float) -> None:
        self.sensitivity = float(np.clip(s, *SENSITIVITY))
        if self._an is not None:
            self._an.sensitivity = self.sensitivity

    def latest(self) -> Reading:
        with self._lock:
            return self._latest


# --- synthetic sounds (for the demo, the protocols, the API and the tests: no device) -----------------------------------------------------
def hum(hz: float, seconds: float, rate: float = RATE, amp: float = 0.1, ipi_ms: float | None = None, pulse_ms: float = 12.0) -> np.ndarray:
    """A hum: a sine at `hz`. With ipi_ms it is a pulse train (a pulse of pulse_ms every ipi_ms), the shape of the courtship
    song's pulses; without, it is steady. Deterministic."""
    n = int(round(seconds * rate))
    t = np.arange(n) / rate
    x = np.sin(2 * math.pi * hz * t)
    if ipi_ms:
        phase = (t * 1000.0) % ipi_ms
        env = np.where(phase < pulse_ms, np.sin(math.pi * phase / pulse_ms) ** 2, 0.0)
        x = x * env
    return (amp * x).astype(np.float32)


def chunks(x: np.ndarray, size: int = CHUNK):
    for i in range(0, len(x) - size + 1, size):
        yield x[i:i + size]
