"""3.0 day 2: simulated calcium imaging (MODEL). Arithmetic of the kernel, the ROI average and the shot noise; exports; the live
view path. The brains are the synthetic pack; nothing is read off it as biology."""
from __future__ import annotations

import numpy as np
import pytest

from kickthefly.lab import imaging as im


# --- the kernel ------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(im.INDICATORS))
def test_kernel_has_the_indicators_time_to_peak_and_half_decay(key):
    ind = im.INDICATORS[key]
    k = im.kernel(ind, dt_ms=0.1, amplitude=1.0)
    pk = int(k.argmax())
    assert k.max() == pytest.approx(1.0)
    assert pk * 0.1 == pytest.approx(ind.rise_ms, rel=0.01)
    half = np.interp(-0.5, -k[pk:], np.arange(pk, len(k)) * 0.1) - pk * 0.1
    assert half == pytest.approx(ind.half_decay_ms, rel=0.01)


def test_indicator_speeds_are_ordered_like_the_papers_say():
    s, f, m = (im.INDICATORS[k] for k in ("gcamp6s", "gcamp6f", "jgcamp8m"))
    assert s.half_decay_ms > f.half_decay_ms > m.half_decay_ms * 0.9 and s.rise_ms > f.rise_ms > m.rise_ms * 0.9
    assert m.half_decay_ms == 137.0 and m.rise_ms == 58.0            # Zhang et al. 2023's Drosophila in-vivo numbers, stated in its text


def test_every_indicator_says_which_of_its_numbers_were_not_verified():
    assert "NOT found in the paper's text" in im.INDICATORS["gcamp6s"].verified
    assert "NOT found in the paper's text" in im.INDICATORS["gcamp6f"].verified
    assert "stated in the paper's text" in im.INDICATORS["jgcamp8m"].verified
    for ind in im.INDICATORS.values():
        assert "doi:10." in ind.source


def test_indicator_aliases_and_errors():
    assert im.indicator("GCaMP6s").key == "gcamp6s" and im.indicator("8m").key == "jgcamp8m" and im.indicator("jGCaMP8m").key == "jgcamp8m"
    with pytest.raises(im.ImagingError, match="unknown indicator"):
        im.indicator("rcamp")


# --- the session ------------------------------------------------------------------------------------------------------------------------
def _session(n=10, rois=None, **kw):
    rois = rois or {"a": np.arange(0, 5), "b": np.arange(5, 10)}
    kw.setdefault("shot_noise", False)
    kw.setdefault("baseline", "zero")                    # analytic tests: F0 is the no-calcium level
    return im.ImagingSession(n, rois, kw.pop("indicator_name", "gcamp6f"), kw.pop("fps", 20.0), **kw)


def test_one_spike_peaks_at_dff_per_spike_then_decays():
    s = _session(1, {"a": np.array([0])}, dff_per_spike=0.3)
    s.push(np.array([0]))
    peak = 0.0
    trace = []
    for _ in range(400):
        s.push(np.array([], int))
        trace.append(s.neuron_dff()[0])
    assert max(trace) == pytest.approx(0.3, rel=0.03)              # 5 ms steps sample the peak to within a fraction of a percent
    assert trace[-1] < 0.3 * 0.05                                   # well decayed after 2 s for GCaMP6f


def test_responses_add_linearly_and_roi_is_the_mean():
    a, b = _session(4, {"r": np.arange(4)}), _session(4, {"r": np.arange(4)})
    a.push(np.array([0]))
    b.push(np.array([0, 1]))
    for _ in range(40):
        a.push(np.array([], int))
        b.push(np.array([], int))
    da, db = a.neuron_dff(), b.neuron_dff()
    assert db[1] == pytest.approx(da[0]) and db[0] == pytest.approx(da[0])
    assert a.roi_dff_true()[0] == pytest.approx(da[0] / 4)          # one of four neurons spiked: the ROI is the mean


def test_frames_follow_the_frame_rate():
    s = _session(2, {"a": np.array([0, 1])}, fps=20.0)
    assert s.frame_steps == 10 and s.fps == pytest.approx(20.0)
    n = sum(s.push(np.array([], int)) for _ in range(100))
    assert n == 10 and len(s.t) == 10 and s.t[0] == pytest.approx(0.05)
    assert im.ImagingSession(2, {"a": np.array([0])}, "gcamp6s", 33.0).fps == pytest.approx(1 / 0.03)     # the 5 ms step quantizes the rate
    with pytest.raises(im.ImagingError, match="frame rate"):
        im.ImagingSession(2, {"a": np.array([0])}, "gcamp6s", 500.0)


def test_shot_noise_is_poisson_and_shrinks_with_more_photons_and_more_neurons():
    def sd(n_neurons, f0):
        s = im.ImagingSession(n_neurons, {"a": np.arange(n_neurons)}, "gcamp6s", 20.0, f0_photons=f0, shot_noise=True, seed=1)
        for _ in range(4000):
            s.push(np.array([], int))
        return np.std(np.array(s.dff)[:, 0])

    assert sd(1, 100) == pytest.approx(1 / np.sqrt(100), rel=0.15)                # Poisson: sd(dF/F at rest) = 1/sqrt(photons)
    assert sd(1, 400) == pytest.approx(0.5 * sd(1, 100), rel=0.2)
    assert sd(16, 100) == pytest.approx(sd(1, 100) / 4, rel=0.2)                   # an ROI of 16 cells averages the noise down


def test_noise_is_seeded_and_the_noise_free_trace_is_always_kept():
    def run(seed, noise=True):
        s = im.ImagingSession(3, {"a": np.arange(3)}, "gcamp6s", 20.0, shot_noise=noise, seed=seed)
        for i in range(200):
            s.push(np.array([0]) if i % 7 == 0 else np.array([], int))
        return s

    a, b, c = run(1), run(1), run(2)
    assert np.array_equal(a.dff, b.dff) and not np.array_equal(a.dff, c.dff)
    assert np.array_equal(a.true_dff, c.true_dff)                                   # the signal doesn't depend on the noise seed
    assert np.array_equal(run(1, False).dff, run(1, False).true_dff)


def test_the_cap_limits_dff():
    s = _session(1, {"a": np.array([0])}, dff_cap=0.5)
    for _ in range(200):
        s.push(np.array([0]))
    assert s.neuron_dff()[0] == pytest.approx(0.5)


def test_session_needs_rois_and_rejects_empty_ones():
    with pytest.raises(im.ImagingError, match="at least one ROI"):
        im.ImagingSession(3, {})
    with pytest.raises(im.ImagingError, match="no neurons"):
        im.ImagingSession(3, {"a": np.array([], int)})


def test_live_buffers_are_bounded_and_the_feed_matches_direct_pushes(synthetic_pack):
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=8, warmup=80)
    rois = im.rois_by_region(br)
    direct = im.ImagingSession(br.n, rois, "gcamp6s", 20.0, shot_noise=False)
    fed = im.ImagingSession(br.n, rois, "gcamp6s", 20.0, shot_noise=False, max_frames=5)
    direct.prime_from_activity(br.sim.activity)
    fed.feed_from_activity(br.sim.activity)                                         # the first call primes and starts counting
    for i in range(300):
        br._step()
        direct.push(np.flatnonzero(br.sim.spikes))
        if i % 10 == 9:
            fed.feed_from_activity(br.sim.activity)
    assert len(fed.t) == 5 and len(direct.t) == 30
    assert np.allclose(np.array(fed.true_dff), np.array(direct.true_dff)[-5:], atol=1e-5)


# --- ROIs ------------------------------------------------------------------------------------------------------------------------------------
def test_rois_by_region_and_from_specs(synthetic_pack):
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=1, warmup=10)
    r = im.rois_by_region(br)
    assert "Mushroom Body" in r and "unassigned" not in r and all(len(v) for v in r.values())
    custom = im.rois_from_specs(br, ["DNp01", "prefix:KC", "line:SS00727"])
    assert len(custom) == 3 and len(custom["DNp01"]) == int((br.types == "DNp01").sum())
    each = im.rois_from_specs(br, "type:DNp01", per_neuron=True)
    assert len(each) == 2 and all(len(v) == 1 for v in each.values())
    with pytest.raises(im.ImagingError, match="selects no neurons|unknown"):
        im.rois_from_specs(br, "type:Nope")


# --- recording and exports -------------------------------------------------------------------------------------------------------------
@pytest.fixture
def result(synthetic_pack):
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=2, warmup=80)
    return im.record(br, 2.0, im.rois_by_region(br), "gcamp6f", 20.0, seed=3)


def test_record_returns_frames_for_every_roi(result):
    assert result.dff.shape == result.true_dff.shape == result.photons.shape == (40, len(result.roi_names))
    assert result.meta["tag"].startswith("MODEL") and result.meta["indicator"].startswith("GCaMP6f")
    assert result.meta["time_to_peak_ms"] == 62.0 and "F0 is a running mean" in result.meta["baseline"]


def test_csv_has_the_tag_and_both_traces(result, tmp_path):
    f = im.export_csv(result, tmp_path / "a" / "roi.csv")
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# MODEL") and lines[2].startswith("time_s,dFF:")
    assert len(lines) == 3 + 40 and lines[2].count("dFF_noise_free:") == len(result.roi_names)


def test_tiff_stack_roundtrips(tmp_path):
    from PIL import Image

    frames = [np.full((20, 30, 3), v, np.uint8) for v in (10, 100, 200)]
    f = im.export_tiff(frames, tmp_path / "v.tif", {"indicator": "x"})
    with Image.open(f) as t:
        assert t.n_frames == 3 and t.size == (30, 20)
        assert "MODEL" in t.tag_v2[270]
        t.seek(2)
        assert np.asarray(t.convert("RGB"))[0, 0, 0] == 200
    with pytest.raises(im.ImagingError, match="no frames"):
        im.export_tiff([], tmp_path / "e.tif")


def test_nwb_has_roi_response_series_and_the_image_series(result, tmp_path):
    pynwb = pytest.importorskip("pynwb")
    result.frames = [np.full((8, 10, 3), i, np.uint8) for i in range(len(result.t_s))]
    f = im.export_nwb(result, tmp_path / "i.nwb")
    with pynwb.NWBHDF5IO(str(f), "r") as io:
        nwb = io.read()
        assert "NOT" in nwb.session_description.upper() or "not a" in nwb.session_description
        ophys = nwb.processing["ophys"]
        dff = ophys["DfOverF"]["dFF"]
        ph = ophys["Fluorescence"]["photons"]
        assert dff.data.shape == (40, len(result.roi_names)) and ph.unit.startswith("photons")
        assert dff.rate == pytest.approx(20.0) and len(dff.rois.table) == len(result.roi_names)
        img = nwb.acquisition["rendered_view"]
        assert img.data.shape == (40, 8, 10, 3)
        assert np.allclose(dff.data[:], result.dff)


# --- rendering and accessibility --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("palette", list(im.LUTS))
def test_palettes_map_black_to_black_and_brighter_to_brighter(palette):
    ramp = np.tile(np.arange(0, 256, 5, dtype=np.uint8)[None, :, None], (2, 1, 3))
    out = im.apply_lut(ramp, palette)
    assert out.shape == ramp.shape and out[0, 0].sum() == 0
    lum = out.astype(float).sum(axis=2)[0]
    assert np.all(np.diff(lum) >= -1)                                # never gets darker as the signal rises
    assert lum[-1] > lum[len(lum) // 2]


def test_every_accessibility_palette_has_a_imaging_lut():
    from kickthefly.core import config

    assert set(config.BY_KEY["access.palette"].options) <= set(im.LUTS)


def test_colorblind_palettes_carry_the_signal_on_blue_yellow_or_luminance_not_red_green():
    ramp = np.tile(np.arange(0, 256, 5, dtype=np.uint8)[None, :, None], (1, 1, 3))
    by = im.apply_lut(ramp, "blue-yellow")[0].astype(float)
    blue = by[(by[:, 2] > by[:, 0] * 2) & (by[:, 2] > by[:, 1])]
    yellow = by[(by[:, 0] > 200) & (by[:, 1] > 180) & (by[:, 2] < 120)]
    assert len(blue) and len(yellow)                                 # the ramp runs through a blue stretch and a yellow one
    hc = im.apply_lut(ramp, "high-contrast")[0].astype(float)
    assert np.all(hc[:, 1] <= np.maximum(hc[:, 0], hc[:, 2]) + 1)    # never green-dominant: magenta, grey and white only
    default = im.apply_lut(ramp, "default")[0].astype(float)
    assert np.any(default[:, 1] > default[:, 0] * 2)                 # the default is the familiar fluorescence green, the others are not


def test_view_rates_make_the_view_brightness_proportional_to_dff():
    s = _session(6, {"a": np.arange(3), "b": np.arange(3, 6)})
    for _ in range(20):
        s.push(np.array([0, 1, 2]))
    calm = np.full(6, 0.01, np.float32)
    hot = np.zeros(6, bool)
    rates, full = s.view_rates(calm, hot, gain=1.0)
    excess = np.maximum(rates / 0.025 - calm / 0.025 - np.where(hot, 1.0, 0.6), 0)          # what BrainView.render computes
    assert np.allclose(2.2 * excess, full, atol=1e-4)
    assert full[3:].max() == 0 and full[:3].min() > 0                                       # neurons that didn't spike stay dark


def test_the_view_renders_dff_for_real(synthetic_pack):
    pytest.importorskip("pygame")
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=1, warmup=50)
    view = im.make_view(br)
    s = im.ImagingSession(br.n, im.rois_by_region(br), "gcamp6f", 20.0, shot_noise=False)
    for _ in range(300):
        br._step()
        s.push(np.flatnonzero(br.sim.spikes))
    frame, full = im.render_frame(view, s, "default", "panel")
    assert frame.shape == (193, 356, 3) and frame.dtype == np.uint8 and frame.max() > 0
    assert frame[..., 0].max() < frame[..., 1].max()                                        # the default palette is green-dominant
    quiet = im.ImagingSession(br.n, im.rois_by_region(br), "gcamp6f", 20.0, shot_noise=False)
    dark, _ = im.render_frame(view, quiet, "default", "panel")
    assert frame.mean() > dark.mean()                                                       # activity lights the view; silence doesn't


def test_imaging_live_starts_feeds_and_recolors(synthetic_pack):
    import pygame

    from kickthefly.core import simcore
    from kickthefly.lab import livelab

    br = simcore.new_brain(seed=3, warmup=50)
    live = livelab.ImagingLive()
    graph, br.graph = br.graph, None
    live.set_on(True, br, None)
    assert live.on is False and "region" in live.error               # a brain with no region labels: a clear error, not a crash
    br.graph = graph
    live.set_on(True, br, None)
    assert live.on and live.session is not None
    live.feed(br)
    for _ in range(200):
        br._step()
    live.feed(br)
    assert len(live.session.t) > 10
    view = im.make_view(br)
    rates = live.view_rates(view, reduced_flashing=True)
    surf = view.render("panel", rates, np.zeros(0, np.int64), 0.0, False)
    out = live.recolor(view, surf, "blue-yellow", True)
    assert isinstance(out, pygame.Surface) and out.get_size() == surf.get_size()
    live.set_on(False)
    assert live.session is None


# --- the running baseline ----------------------------------------------------------------------------------------------------------------
def test_a_steady_firing_rate_reads_as_zero_dff_once_the_baseline_has_settled():
    rng = np.random.default_rng(0)
    s = im.ImagingSession(40, {"a": np.arange(40)}, "gcamp6s", 20.0, shot_noise=False, baseline_tau_s=3.0)
    for _ in range(6000):                                                           # 30 s of 10 Hz Poisson firing
        s.push(np.flatnonzero(rng.random(40) < 10 * 0.005))
    tail = np.array(s.true_dff)[-100:, 0]
    assert abs(tail.mean()) < 0.1 and tail.std() < 0.2                              # flat: only changes show
    rng = np.random.default_rng(1)
    for _ in range(200):                                                            # then the rate triples for 1 s
        s.push(np.flatnonzero(rng.random(40) < 30 * 0.005))
    assert np.array(s.true_dff)[-1, 0] > 0.15


def test_the_zero_baseline_reads_the_tonic_level_as_signal():
    rng = np.random.default_rng(0)
    z = im.ImagingSession(40, {"a": np.arange(40)}, "gcamp6s", 20.0, shot_noise=False, baseline="zero")
    for _ in range(3000):
        z.push(np.flatnonzero(rng.random(40) < 10 * 0.005))
    assert np.array(z.true_dff)[-1, 0] > 1.0                                        # why the running baseline is the default
    with pytest.raises(im.ImagingError, match="baseline"):
        im.ImagingSession(2, {"a": np.array([0])}, "gcamp6s", 20.0, baseline="nope")


def test_priming_from_the_brains_raster_starts_near_zero_dff(synthetic_pack):
    from kickthefly.core import simcore

    br = simcore.new_brain(seed=12, warmup=450)
    rois = im.rois_by_region(br)
    cold = im.ImagingSession(br.n, rois, "gcamp6s", 20.0, shot_noise=False)
    primed = im.ImagingSession(br.n, rois, "gcamp6s", 20.0, shot_noise=False)
    primed.prime_from_activity(br.sim.activity)
    for _ in range(100):
        br._step()
        cold.push(np.flatnonzero(br.sim.spikes))
        primed.push(np.flatnonzero(br.sim.spikes))
    assert np.abs(np.array(primed.true_dff)).mean() < 0.5 * np.abs(np.array(cold.true_dff)).mean()
