# Simulated calcium imaging (3.0)

Lab > **Calcium imaging** turns the brain view into an **Imaging mode**: each neuron is colored by the simulated GCaMP dF/F instead of
by its firing. ROI traces (per brain region, or any neuron specs) are plotted live and can be exported as CSV, NWB
(`RoiResponseSeries` and an `ImageSeries`) and a TIFF stack of the rendered view.

## MODEL, end to end

spikes -> convolve with an indicator kernel -> dF/F per neuron -> average over an ROI -> Poisson photon shot noise -> dF/F of the ROI

| | tag |
|---|---|
| which neurons spike, and which neurons an ROI covers | CONNECTOME |
| the kernel's rise and decay | LITERATURE (see below) |
| everything else: linear in spikes (with a cap), equal brightness for every neuron in an ROI, dF/F per spike (0.2), photons per neuron per frame (100), a running-mean F0 (30 s time constant by default; Settings > Brain > Imaging baseline, or `f0_tau_s` in a protocol's `imaging:` block; 10, 30, 60 or 120 s) | MODEL / GAME RULE |

Not modeled: subthreshold calcium, bleaching, motion, scattering, neuropil, cell segmentation, optics. The baseline F0 is a slow running
mean of each neuron's own fluorescence, so a steady rate reads as dF/F 0 and only changes show; the first seconds of a session are primed
from the brain's last 2 s of spikes to limit the start-up error. A zero-calcium baseline option exists for tests and makes tonic firing
look like a large signal (spontaneous firing alone gave dF/F of 4-5 in one real-brain check, which is why the running baseline is the default).

## Indicators, and where their numbers come from

The kernel is a two-exponential function fitted to two numbers: the single-spike **time to peak** and the **half-decay**.

| indicator | time to peak | half-decay | source | verified in the paper's text? |
|---|---|---|---|---|
| GCaMP6s | 179 ms | 550 ms | Chen et al. 2013, Nature 499:295, [doi:10.1038/nature12354](https://doi.org/10.1038/nature12354), Supplementary Table 3 | **yes**: mouse V1 in vivo, cell-attached, 1 action potential: rise time to peak 179 +- 23 ms, half-decay 550 +- 52 ms (means used) |
| GCaMP6f | 45 ms | 142 ms | same table | **yes**: rise time to peak 45 +- 4 ms, half-decay 142 +- 11 ms (means used) |
| jGCaMP8m | 58 ms | 137 ms | Zhang et al. 2023, Nature 615:884, [doi:10.1038/s41586-023-05828-9](https://doi.org/10.1038/s41586-023-05828-9) | both stated (Drosophila in-vivo visual responses: half-rise 58 +- 6, half-decay 137 +- 21 ms); used as the kernel's time to peak, a simplification |

**3.0 day 2 review:** the GCaMP6 numbers were checked in Chen et al. 2013's Supplementary Table 3 (read from the paper's supplementary
PDF through Europe PMC). Day 2 had used the middles of the main text's rise ranges (125 and 62 ms) and half-decays of 550 and 140 ms that it
could not find in the text; 550 ms is in the table, 140 ms was 142 ms. Both numbers of each kernel now come from that one table (the same
experiment: single action potentials in mouse visual cortex in vivo), not from fly neurons. The main text's 100-150 ms (6s) and 50-75 ms (6f)
are the intervals at which single spikes became resolvable, a different quantity.

## Accessibility

The view's colors follow Settings > Accessibility > Brain view colors (fluorescence green, blue/yellow, high contrast); the signal is
also carried by brightness. **Reduced flashing** smooths the view over frames and plots the noise-free trace instead of the shot-noise one;
there are no spike sparkles in this mode.

## Python and protocols

```python
res = fly.image(5.0, rois=["type:LPLC2", "type:DNp01"], indicator="gcamp6f", fps=20)
imaging.export_csv(res, "roi.csv"); imaging.export_nwb(res, "roi.nwb")
```

`protocols/imaging_looming.yaml`: an `imaging:` block on a stimulus protocol writes `<seed>-imaging.csv` (and `.nwb` with `nwb: true`, a
TIFF stack with `imaging: {tiff: true}`). Settings > Brain has the default indicator, frame rate and baseline time constant.

**Choices made in 3.0 Day 2 (design decisions, not measurements):**
- *Frame rate:* 20 Hz stays the default (Settings offers 5, 10, 20, 30, 40; the API and protocols accept 1-200 Hz). At 20 Hz a frame is 50 ms, so the fastest kernel here (jGCaMP8m, 58 ms to peak) still has about one frame on its rise and the slowest (GCaMP6s, 550 ms half-decay) has about eleven on its decay. No published frame rate is claimed for any real preparation; pick the rate your question needs. The 5 ms simulation step quantizes every rate.
- *F0 time constant:* 30 s stays the default and is now a setting, because it is a game rule and a researcher should be able to see and change what dF/F is measured against. A shorter constant hides slow changes in the rate; a longer one takes longer to settle (the first seconds read high or low). It is not taken from a paper.
- *GCaMP6 kernels:* kept at the paper's own single-spike numbers (Chen et al. 2013, Supplementary Table 3: GCaMP6s 179 / 550 ms, GCaMP6f 45 / 142 ms), not the main text's resolvability ranges: those describe how fast a response must rise to resolve one spike, a different quantity.
- *jGCaMP8m, checked 3.0 Day 2:* the open-access full text (PMC10060165, read through Europe PMC) gives half-rise 58 +- 6 ms and half-decay 137 +- 21 ms for Drosophila L2 dendrites under 0.5 Hz light-dark flashes, and single-spike half-rise under 5 ms in mouse. The fly numbers are slower than a single-spike response because they follow a visual stimulus; this game uses them as written and says so.
