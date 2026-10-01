"""Simulated calcium imaging: what a GCaMP-expressing fly's brain would look like under the microscope, made from the
simulation's spikes.

MODEL, end to end. The simulation has spikes and nothing else. Imaging here is a forward model on top of them:

  spikes --(convolve with an indicator kernel)--> dF/F per neuron --(average over an ROI)--> expected fluorescence
        --(Poisson photon shot noise)--> counted photons --> dF/F of the ROI

  CONNECTOME   which neurons spike, and which neurons an ROI covers (a region or a cell type, from the brain pack).
  LITERATURE   the kernel's rise and decay (see INDICATORS: which numbers are verified in the paper's own text and which are not).
  MODEL        everything else: each spike adds the same kernel (linear, no saturation but a cap, no calcium buffering, no
               indicator dynamics beyond two exponentials); every neuron in an ROI is equally bright and equally expressing
               (a real ROI is weighted by each cell's brightness and baseline); F0 is a slow running mean of each neuron's own
               fluorescence (30 s time constant, a game parameter; real pipelines use a running mean or low percentile too), so
               a steady firing rate reads as dF/F 0 and only changes show; the first seconds of a session, before the mean has
               settled, are inflated or deflated accordingly (the indicator state is primed from the brain's last 2 s of spikes to
               limit that); the amplitude per spike is a game parameter (`dff_per_spike`, default
               0.2; real values depend on the indicator, the cell and the preparation); the photon budget per neuron per
               frame (`f0_photons`) is a game parameter; there is no motion, no bleaching, no neuropil contamination, no
               scattering, no light-sheet or two-photon optics, no frame-rate-dependent distortion beyond the sampling itself.
               Calcium entry here follows only spikes: the real thing also follows subthreshold input the model does not have.
  GAME RULE    the ROI sets offered (per region, per type, per neuron) and the colors of the rendered view.

The imaging view colors the brain view by dF/F instead of firing rate. Accessibility: the palette follows Settings >
Accessibility > Brain view colors; with Reduced flashing the display is smoothed over frames and shows no shot noise, and the
spike sparkles are off.

    from kickthefly.lab import imaging
    res = imaging.record(br, seconds=5, rois=imaging.rois_by_region(br), indicator="gcamp6s", fps=20)
    imaging.export_csv(res, "roi.csv")
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DT_S = 0.005
TAG_TEXT = "MODEL: spikes from the simulation convolved with an indicator kernel, plus shot noise. Not a measurement."


@dataclass(frozen=True)
class Indicator:
    key: str
    name: str
    rise_ms: float               # the single-spike kernel's time to peak
    half_decay_ms: float         # time from the peak to half of it, decaying
    source: str
    verified: str                # what was checked in the paper's own text, and what was NOT


INDICATORS = {
    # 3.0 day 2 review: both numbers of each GCaMP6 kernel are now the paper's own single-spike measurements, read in Chen et al.
    # 2013's Supplementary Table 3 (the main text gives only the rise ranges quoted below; day 2 had used their middles, 125 and
    # 62 ms, and a 140 ms GCaMP6f half-decay that was not in the paper).
    "gcamp6s": Indicator("gcamp6s", "GCaMP6s (slow, sensitive)", 179.0, 550.0,
                         "Chen et al. 2013, Nature 499:295 (doi:10.1038/nature12354), Supplementary Table 3",
                         "Verified in the paper's Supplementary Table 3 (mouse V1 in vivo, cell-attached, 1 action potential): "
                         "rise time to peak 179 +- 23 ms, half-decay 550 +- 52 ms; this game uses the means. (The main text "
                         "gives a 100-150 ms rise time for resolving single spikes.) Mouse cortex, not fly neurons."),
    "gcamp6f": Indicator("gcamp6f", "GCaMP6f (fast)", 45.0, 142.0,
                         "Chen et al. 2013, Nature 499:295 (doi:10.1038/nature12354), Supplementary Table 3",
                         "Verified in the paper's Supplementary Table 3 (mouse V1 in vivo, cell-attached, 1 action potential): "
                         "rise time to peak 45 +- 4 ms, half-decay 142 +- 11 ms; this game uses the means. (The main text "
                         "gives a 50-75 ms rise time for resolving single spikes.) Mouse cortex, not fly neurons."),
    "jgcamp8m": Indicator("jgcamp8m", "jGCaMP8m (fast, sensitive)", 58.0, 137.0,
                          "Zhang et al. 2023, Nature 615:884 (doi:10.1038/s41586-023-05828-9), Drosophila in-vivo visual "
                          "responses: half-rise 58 +- 6 ms, half-decay 137 +- 21 ms",
                          "Both numbers are stated in the paper's text (a figure legend). They describe a fly's response to a "
                          "visual stimulus, not a single-spike impulse response, and a two-exponential kernel cannot have a 58 ms "
                          "half-rise and a 137 ms half-decay at once, so this game uses 58 ms as the kernel's time to peak (a "
                          "simplification). The same paper reports much faster kinetics in cultured neurons (half-rise 2 ms)."),
}
DEFAULT_INDICATOR = "gcamp6s"
ALIASES = {"6s": "gcamp6s", "gcamp6s": "gcamp6s", "6f": "gcamp6f", "gcamp6f": "gcamp6f", "8m": "jgcamp8m",
           "gcamp8m": "jgcamp8m", "jgcamp8m": "jgcamp8m"}


class ImagingError(ValueError):
    pass


def indicator(name: str) -> Indicator:
    key = ALIASES.get(str(name).strip().lower().replace("-", "").replace("_", ""))
    if key is None:
        raise ImagingError(f"unknown indicator {name!r}; use one of GCaMP6s, GCaMP6f, GCaMP8m")
    return INDICATORS[key]


# --- the kernel ---------------------------------------------------------------------------------------------------------------
def _shape(tau_r: float, tau_d: float) -> tuple[float, float]:
    """(time to peak, half-decay from the peak), ms, of k(t) = exp(-t/tau_d) - exp(-t/tau_r)."""
    tp = tau_d * tau_r / (tau_d - tau_r) * math.log(tau_d / tau_r)
    peak = math.exp(-tp / tau_d) - math.exp(-tp / tau_r)
    lo, hi = tp, tp + 20 * tau_d
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.exp(-mid / tau_d) - math.exp(-mid / tau_r) > 0.5 * peak:
            lo = mid
        else:
            hi = mid
    return tp, (lo + hi) / 2 - tp


def time_constants(ind: Indicator) -> tuple[float, float]:
    """(tau_rise, tau_decay) in ms of the two-exponential kernel whose time to peak and half-decay are the indicator's."""
    from scipy.optimize import least_squares

    def resid(x):
        tr, td = math.exp(x[0]), math.exp(x[1])
        if tr >= td:
            return [1e3, 1e3]
        tp, hd = _shape(tr, td)
        return [(tp - ind.rise_ms) / ind.rise_ms, (hd - ind.half_decay_ms) / ind.half_decay_ms]

    best = None
    for f in (0.25, 0.5, 0.1):
        x0 = [math.log(ind.rise_ms * f * 2), math.log(ind.half_decay_ms / math.log(2) * 0.8)]
        r = least_squares(resid, x0, xtol=1e-12, ftol=1e-12)
        if best is None or r.cost < best.cost:
            best = r
    tr, td = math.exp(best.x[0]), math.exp(best.x[1])
    if best.cost > 1e-8:
        raise ImagingError(f"no two-exponential kernel has time to peak {ind.rise_ms} ms and half-decay {ind.half_decay_ms} ms")
    return tr, td


def kernel(ind: Indicator, dt_ms: float = 5.0, length_ms: float | None = None, amplitude: float = 1.0) -> np.ndarray:
    """The single-spike dF/F response sampled every dt_ms, normalized so its peak is `amplitude`."""
    tau_r, tau_d = time_constants(ind)
    length_ms = length_ms or 10 * tau_d
    t = np.arange(0.0, length_ms, dt_ms)
    k = np.exp(-t / tau_d) - np.exp(-t / tau_r)
    return (amplitude * k / k.max()).astype(np.float64)


# --- ROIs -------------------------------------------------------------------------------------------------------------------
MAX_ROIS = 600


def rois_by_region(br, graph=None) -> dict[str, np.ndarray]:
    """One ROI per region label the dataset gives (antennal lobe, mushroom body, ...). Neurons without one are left out."""
    g = graph if graph is not None else getattr(br, "graph", None)
    region = None if g is None else getattr(g, "region", None)
    if region is None:
        raise ImagingError("this brain has no region labels; use rois_by_type or an explicit ROI list")
    region = np.asarray(region).astype(str)
    out = {}
    for name in sorted(set(region.tolist())):
        if name and name != "unassigned":
            out[name] = np.flatnonzero(region == name)
    return out


def rois_from_specs(br, specs, per_neuron: bool = False) -> dict[str, np.ndarray]:
    """ROIs from neuron specs (a type, `prefix:KC`, `line:SS00727`, a group). per_neuron=True gives each neuron its own ROI."""
    from kickthefly.core import simcore

    specs = [specs] if isinstance(specs, str) else list(specs)
    if len(specs) > MAX_ROIS:                       # 3.0 day 2 review: checked before resolving 100,000 specs one by one
        raise ImagingError(f"at most {MAX_ROIS} ROIs")
    out = {}
    for spec in specs:
        if not isinstance(spec, str):
            raise ImagingError(f"an ROI is a neuron spec (text), not {spec!r}")
        try:
            rows = np.sort(np.asarray(simcore.rows_of(br, spec if ":" in spec or spec in getattr(br, "col", {}) else f"type:{spec}"),
                                      np.int64))
        except ValueError as e:
            raise ImagingError(str(e)) from None
        if not len(rows):
            raise ImagingError(f"{spec!r} selects no neurons")
        if per_neuron:
            for r in rows[:MAX_ROIS]:
                out[f"{spec}#{int(r)}"] = np.array([r], np.int64)
        else:
            out[str(spec)] = rows
    if len(out) > MAX_ROIS:
        raise ImagingError(f"at most {MAX_ROIS} ROIs")
    return out


# --- the imaging session ----------------------------------------------------------------------------------------------------------
class ImagingSession:
    """Keeps the indicator state of the neurons in the ROIs and turns it into frames.

    Call push(spiked_indices) once per 5 ms simulation step; every frame_steps pushes a frame is taken."""

    def __init__(self, n: int, rois: dict[str, np.ndarray], indicator_name: str = DEFAULT_INDICATOR, fps: float = 20.0,
                 f0_photons: float = 100.0, dff_per_spike: float = 0.2, shot_noise: bool = True, seed: int = 0,
                 dff_cap: float = 5.0, max_frames: int | None = None, baseline: str = "running", baseline_tau_s: float = 30.0):
        self.ind = indicator(indicator_name)
        if not rois:
            raise ImagingError("an imaging session needs at least one ROI")
        if not (1.0 <= fps <= 200.0):
            raise ImagingError("frame rate must be between 1 and 200 Hz")
        for name, v in (("f0_photons", f0_photons), ("dff_per_spike", dff_per_spike), ("dff_cap", dff_cap),
                        ("baseline_tau_s", baseline_tau_s)):
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            if not (math.isfinite(v) and v > 0):        # 3.0 day 2 review: inf / NaN got through
                raise ImagingError(f"{name} must be a positive finite number")
        self.n, self.fps = int(n), float(fps)
        self.frame_steps = max(1, int(round(1.0 / fps / DT_S)))
        self.fps = 1.0 / (self.frame_steps * DT_S)              # the rate the 5 ms steps can really give
        self.f0, self.amp, self.shot_noise, self.cap = float(f0_photons), float(dff_per_spike), bool(shot_noise), float(dff_cap)
        self.roi_names = list(rois)
        self.roi_rows = [np.asarray(rois[k], np.int64) for k in self.roi_names]
        if any(len(r) == 0 for r in self.roi_rows):
            raise ImagingError("an ROI has no neurons")
        self.tracked = np.unique(np.concatenate(self.roi_rows))
        self.pos = np.full(self.n, -1, np.int32)
        self.pos[self.tracked] = np.arange(len(self.tracked), dtype=np.int32)
        self.roi_pos = [self.pos[r] for r in self.roi_rows]
        tr, td = time_constants(self.ind)
        self.tau_r_ms, self.tau_d_ms = tr, td
        self.a_r, self.a_d = math.exp(-DT_S * 1000 / tr), math.exp(-DT_S * 1000 / td)
        # peak of exp(-t/td)-exp(-t/tr) so one spike peaks at dff_per_spike
        tp = td * tr / (td - tr) * math.log(td / tr)
        self.norm = self.amp / (math.exp(-tp / td) - math.exp(-tp / tr))
        self.ed = np.zeros(len(self.tracked), np.float32)
        self.er = np.zeros(len(self.tracked), np.float32)
        if baseline not in ("running", "zero"):
            raise ImagingError("baseline must be 'running' or 'zero'")
        self.baseline = baseline                          # "zero": F0 is the fluorescence with no calcium (analytic tests)
        self.baseline_tau_s = float(baseline_tau_s)
        self.k_base = 1.0 - math.exp(-(self.frame_steps * DT_S) / max(1e-3, self.baseline_tau_s))
        self.base = np.zeros(len(self.tracked), np.float32)   # running-mean dF/F (absolute) of each tracked neuron
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.frames_taken = 0                               # every frame ever taken (t keeps only the last max_frames)
        self._seen = None                                   # brain step count already pushed (feed_from_activity)
        self.max_frames = max_frames
        self.t, self.true_dff, self.dff, self.photons = [], [], [], []

    # -- spikes in ------------------------------------------------------------------------------------------------------------
    def push(self, spiked) -> bool:
        """One simulation step. spiked: the indices of neurons that fired. Returns True when it completed a frame."""
        self.ed *= np.float32(self.a_d)
        self.er *= np.float32(self.a_r)
        if len(spiked):
            p = self.pos[np.asarray(spiked)]
            p = p[p >= 0]
            if len(p):
                self.ed[p] += 1.0
                self.er[p] += 1.0
        self.steps += 1
        if self.steps % self.frame_steps == 0:
            self.take_frame()
            return True
        return False

    def neuron_dff(self) -> np.ndarray:
        """dF/F of every tracked neuron right now relative to the fluorescence with no calcium (absolute, noise-free)."""
        return np.minimum(self.norm * (self.ed - self.er), self.cap).astype(np.float32)

    def neuron_delta(self) -> np.ndarray:
        """dF/F relative to each neuron's own baseline F0 (the running mean, or the no-calcium level in 'zero' mode)."""
        d = self.neuron_dff()
        return d if self.baseline == "zero" else (1.0 + d) / (1.0 + self.base) - 1.0

    def prime_from_activity(self, activity) -> None:
        """Start from where the brain already is: push the raster's last steps (up to 2 s) without taking frames, then take
        that as the baseline. Without this a session starts at zero calcium and spends seconds climbing to the tonic level."""
        self._seen = int(activity.steps)
        for idx in activity.raster():
            self.ed *= np.float32(self.a_d)
            self.er *= np.float32(self.a_r)
            if len(idx):
                p = self.pos[np.asarray(idx)]
                p = p[p >= 0]
                self.ed[p] += 1.0
                self.er[p] += 1.0
        self.base = self.neuron_dff().copy()

    def roi_dff_true(self) -> np.ndarray:
        """Noise-free dF/F of each ROI: its mean fluorescence over its baseline."""
        d = self.neuron_dff()
        num = np.array([1.0 + d[p].mean() for p in self.roi_pos], np.float64)
        den = np.array([1.0 + (self.base[p].mean() if self.baseline == "running" else 0.0) for p in self.roi_pos], np.float64)
        return num / den - 1.0

    def take_frame(self) -> None:
        d = self.neuron_dff()
        n_in = np.array([len(p) for p in self.roi_pos], np.float64)
        mean_d = np.array([d[p].mean() for p in self.roi_pos], np.float64)
        mean_b = np.array([self.base[p].mean() if self.baseline == "running" else 0.0 for p in self.roi_pos], np.float64)
        expect = n_in * self.f0 * (1.0 + mean_d)
        photons = self.rng.poisson(np.maximum(expect, 0.0)).astype(np.float64) if self.shot_noise else expect
        den = n_in * self.f0 * (1.0 + mean_b)
        true = expect / den - 1.0
        dff = photons / den - 1.0
        if self.baseline == "running":
            self.base += (d - self.base) * np.float32(self.k_base)
        self.frames_taken += 1
        self.t.append(self.steps * DT_S)
        self.true_dff.append(true)
        self.dff.append(dff)
        self.photons.append(photons)
        if self.max_frames and len(self.t) > self.max_frames:          # a live session keeps the last frames only
            for lst in (self.t, self.true_dff, self.dff, self.photons):
                del lst[0]

    def feed_from_activity(self, activity) -> int:
        """Live use: push the steps a running brain has taken since the last call, read from its ActivityBuffer (the last 400
        steps are kept there, so call at least every 2 s). The first call primes the indicator state from the raster.
        Returns how many steps were pushed."""
        steps = int(activity.steps)
        if self._seen is None:
            self.prime_from_activity(activity)
            return 0
        k = min(steps - self._seen, 400)
        self._seen = steps
        if k <= 0:
            return 0
        for idx in activity.raster()[-k:]:
            self.push(idx)
        return k

    # -- the brain view ----------------------------------------------------------------------------------------------------------
    def view_rates(self, view_calm: np.ndarray, hot_mask: np.ndarray, gain: float = 1.0, smooth: np.ndarray | None = None):
        """An array for BrainView.render(rates=...) that makes it draw dF/F. The view lights a neuron by how far its rate is
        above its calm rate and a threshold (0.6 or 1.0 x 0.025); this sets the rate to exactly calm + threshold + gain x dF/F,
        so the brightness is proportional to dF/F. Neurons outside every ROI stay dark. Returns (rates, dff_full)."""
        full = np.zeros(self.n, np.float32)
        full[self.tracked] = np.maximum(self.neuron_delta(), 0.0) * np.float32(gain)
        if smooth is not None:
            full = smooth + 0.35 * (full - smooth)
        thr = np.where(hot_mask, 1.0, 0.6).astype(np.float32)
        return (view_calm + 0.025 * (thr + full / 2.2)).astype(np.float32), full

    # -- results ----------------------------------------------------------------------------------------------------------------
    def result(self, extra: dict | None = None) -> "ImagingResult":
        if not self.t:
            raise ImagingError(f"no imaging frame was taken yet: record at least one frame ({1.0 / self.fps:.3g} s at "
                               f"{self.fps:.3g} Hz)")
        meta = dict(tag=TAG_TEXT, indicator=self.ind.name, indicator_source=self.ind.source, indicator_verified=self.ind.verified,
                    time_to_peak_ms=self.ind.rise_ms, half_decay_ms=self.ind.half_decay_ms,
                    tau_rise_ms=self.tau_r_ms, tau_decay_ms=self.tau_d_ms, fps=self.fps, frame_steps=self.frame_steps,
                    f0_photons_per_neuron_per_frame=self.f0, dff_per_spike=self.amp, dff_cap=self.cap,
                    shot_noise=self.shot_noise,
                    baseline=("F0 is a running mean of each neuron's own fluorescence, time constant "
                              f"{self.baseline_tau_s:g} s" if self.baseline == "running"
                              else "F0 is the fluorescence with no calcium (n_neurons x f0_photons)"))
        meta.update(extra or {})
        return ImagingResult(np.array(self.t), np.array(self.true_dff).reshape(len(self.t), -1),
                             np.array(self.dff).reshape(len(self.t), -1), np.array(self.photons).reshape(len(self.t), -1),
                             list(self.roi_names), np.array([len(r) for r in self.roi_rows]), meta,
                             [r.copy() for r in self.roi_rows])


@dataclass
class ImagingResult:
    t_s: np.ndarray
    true_dff: np.ndarray            # (frames, rois) noise-free
    dff: np.ndarray                 # (frames, rois) with shot noise (if on)
    photons: np.ndarray             # (frames, rois) counted (or expected) photons
    roi_names: list
    roi_neurons: np.ndarray
    meta: dict = field(default_factory=dict)
    roi_rows: list = field(default_factory=list)
    frames: list = field(default_factory=list)      # rendered RGB frames (uint8 arrays), if the recording kept them
    frame_times: list = field(default_factory=list)  # the imaging time (s) of each kept frame, when it isn't one per frame


def check_seconds(seconds) -> float:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        raise ImagingError("seconds must be a number") from None
    if not 0 < seconds <= 3600:
        raise ImagingError("seconds must be between 0 and 3600")
    return seconds


def record(br, seconds: float, rois: dict, indicator: str = DEFAULT_INDICATOR, fps: float = 20.0, seed: int = 0,
           frame_cb=None, **kw) -> ImagingResult:
    """Step a lockstep brain for `seconds` and image it. frame_cb(session, frame_index) runs after each frame (for rendering)."""
    check_seconds(seconds)
    s = ImagingSession(br.n, rois, indicator, fps, seed=seed, **kw)
    s.prime_from_activity(br.sim.activity)
    for _ in range(int(round(seconds / DT_S))):
        br._step()
        if s.push(np.flatnonzero(br.sim.spikes)) and frame_cb is not None:
            frame_cb(s, len(s.t) - 1)
    return s.result()


# --- exports ----------------------------------------------------------------------------------------------------------------------
def export_csv(res: ImagingResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {TAG_TEXT}\n# indicator={res.meta.get('indicator')} fps={res.meta.get('fps'):.4g} "
                f"f0_photons={res.meta.get('f0_photons_per_neuron_per_frame')} dff_per_spike={res.meta.get('dff_per_spike')}\n")
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"dFF:{n}" for n in res.roi_names] + [f"dFF_noise_free:{n}" for n in res.roi_names])
        for i, t in enumerate(res.t_s):
            w.writerow([f"{t:.4f}"] + [f"{x:.6g}" for x in res.dff[i]] + [f"{x:.6g}" for x in res.true_dff[i]])
    return path


def export_tiff(frames: list, path: Path, meta: dict | None = None) -> Path:
    """A multi-page RGB TIFF of the rendered brain-view frames (what the screen showed). The first page's description is
    the JSON metadata, including the MODEL tag."""
    from PIL import Image

    if not frames:
        raise ImagingError("no frames to write: record with render=True")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    desc = json.dumps(dict(meta or {}, tag=TAG_TEXT, frames=len(frames)), default=str)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], compression="tiff_deflate", description=desc)
    return path


def export_nwb(res: ImagingResult, path: Path) -> Path:
    """NWB: a RoiResponseSeries (dF/F) and another (photon counts) over a PlaneSegmentation whose ROIs are the neuron sets,
    plus an ImageSeries of the rendered frames when the recording kept them. Needs pynwb."""
    from kickthefly.lab import nwbexport

    reason = nwbexport.available()
    if reason:
        raise ImagingError(reason)
    import uuid
    from datetime import datetime, timezone

    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.base import TimeSeries  # noqa: F401
    from pynwb.device import Device
    from pynwb.image import ImageSeries
    from pynwb.ophys import (DfOverF, Fluorescence, ImageSegmentation, ImagingPlane, OpticalChannel, RoiResponseSeries)

    from kickthefly.core.version import __version__

    m = res.meta
    nwb = NWBFile(session_description=f"{TAG_TEXT} Simulated {m.get('indicator')} imaging of a simulated fly brain; the data are "
                                      f"model output, not a recording from an animal.",
                  identifier=str(uuid.uuid4()), session_start_time=datetime.now(timezone.utc),
                  experiment_description="Kick the Fly simulated calcium imaging", lab="Kick the Fly (simulation)",
                  source_script=f"kickthefly {__version__}", source_script_file_name="kickthefly/lab/imaging.py",
                  notes=json.dumps({k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}))
    dev = nwb.create_device(name="simulated-microscope", description="none: a forward model on the simulation's spikes")
    ch = OpticalChannel(name="GCaMP", description=f"{m.get('indicator')} (MODEL kernel)", emission_lambda=520.0)
    plane = nwb.create_imaging_plane(name="brain-front-view", optical_channel=ch, imaging_rate=float(m["fps"]),
                                     description="There is no optics: ROIs are sets of simulated neurons.", device=dev,
                                     excitation_lambda=488.0, indicator=str(m.get("indicator")), location="whole brain",
                                     grid_spacing=[1.0, 1.0], grid_spacing_unit="n.a.", origin_coords=[0.0, 0.0],
                                     origin_coords_unit="n.a.")
    seg = ImageSegmentation()
    ps = seg.create_plane_segmentation(name="rois", description="Each ROI is a set of simulated neurons (pixel mask is a placeholder)",
                                       imaging_plane=plane)
    ps.add_column("neurons", "number of simulated neurons in the ROI")
    for k, name in enumerate(res.roi_names):
        ps.add_roi(id=k, pixel_mask=[(0, 0, 1.0)], neurons=int(res.roi_neurons[k]))
    mod = nwb.create_processing_module("ophys", "simulated imaging")
    mod.add(seg)
    region = ps.create_roi_table_region("all ROIs", region=list(range(len(res.roi_names))))
    f = Fluorescence()
    mod.add(f)                                                   # added first, so the ROI table region shares its ancestry
    f.create_roi_response_series(name="photons", data=res.photons.astype(np.float64), rois=region, unit="photons (simulated)",
                                 starting_time=float(res.t_s[0]) if len(res.t_s) else 0.0, rate=float(m["fps"]),
                                 description="counted (Poisson) or expected photons per frame, summed over the ROI's neurons")
    d = DfOverF()
    mod.add(d)
    d.create_roi_response_series(name="dFF", data=res.dff.astype(np.float64), rois=region, unit="n.a.",
                                 starting_time=float(res.t_s[0]) if len(res.t_s) else 0.0, rate=float(m["fps"]),
                                 description="dF/F as a fraction (not percent)")
    if res.frames:
        from hdmf.backends.hdf5.h5_utils import H5DataIO

        # 3.0 day 2 review: the live view keeps one rendered frame per screen refresh that saw a new imaging frame, which is not
        # one per imaging frame, so those carry their own times; a protocol's frames are exactly one per imaging frame.
        timing = (dict(timestamps=[float(x) for x in res.frame_times]) if len(res.frame_times) == len(res.frames)
                  else dict(rate=float(m["fps"]), starting_time=float(res.t_s[0]) if len(res.t_s) else 0.0))
        nwb.add_acquisition(ImageSeries(name="rendered_view", data=H5DataIO(np.stack(res.frames).astype(np.uint8), compression="gzip"),
                                        unit="n.a.", format="raw", description="the rendered brain view colored by dF/F (RGB)",
                                        **timing))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return Path(path)


# --- rendering ------------------------------------------------------------------------------------------------------------------
LUTS = {
    "default": ((0, 0, 0), (0, 70, 10), (40, 255, 60), (230, 255, 230)),          # fluorescence green
    "blue-yellow": ((0, 0, 0), (20, 40, 140), (255, 220, 40), (255, 255, 220)),
    "high-contrast": ((0, 0, 0), (90, 90, 90), (255, 31, 217), (255, 255, 255)),
}


def apply_lut(img: np.ndarray, palette: str = "default") -> np.ndarray:
    """Map an RGB image's brightness through the imaging palette (black -> color -> white). img: (h, w, 3) uint8."""
    stops = np.array(LUTS.get(palette, LUTS["default"]), np.float32)
    lum = img.astype(np.float32).max(axis=2) / 255.0
    x = lum * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def make_view(br):
    """A BrainView for this lockstep brain (what the game's brain panel draws), built from the pack's cell-body positions."""
    from kickthefly.core import simcore
    from kickthefly.game import kick_the_fly as k

    g, W, soma = simcore.pack(getattr(br, "brain_type", "adult"))
    names = [n for n, _ in k.POPS]
    pain_groups = [br.col[n] for n in (*k.TOUCH, "heat", "cold", "smell", "taste", "body_extra")]
    pain_mask = np.isin(br.det_id, pain_groups) | (br.pop_id == names.index("ascending"))
    return k.BrainView(soma, W, pain_mask, regions=getattr(g, "region", None), graph=g)


def render_frame(view, session: ImagingSession, palette: str = "default", key: str = "panel", gain: float = 1.0,
                 smooth=None):
    """One rendered frame (RGB uint8 (h, w, 3)) of the brain view colored by dF/F. Returns (frame, smoothed dF/F)."""
    import pygame

    rates, full = session.view_rates(view.calm, view.hot_mask, gain, smooth)
    sparkle, view.sparkle = getattr(view, "sparkle", True), False
    try:
        surf = view.render(key, rates, np.zeros(0, np.int64), 0.0, False)
    finally:
        view.sparkle = sparkle
    arr = pygame.surfarray.array3d(surf).swapaxes(0, 1)
    return apply_lut(arr, palette), full
