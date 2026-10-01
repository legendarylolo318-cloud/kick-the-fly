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
| everything else: linear in spikes (with a cap), equal brightness for every neuron in an ROI, dF/F per spike (0.2), photons per neuron per frame (100), a 30 s running-mean F0 | MODEL / GAME RULE |

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
TIFF stack with `imaging: {tiff: true}`). Settings > Brain has the default indicator and frame rate.
