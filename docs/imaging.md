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

## Indicators, and what was and was not verified

The kernel is a two-exponential function fitted to two numbers: the single-spike **time to peak** and the **half-decay**.

| indicator | time to peak | half-decay | source | verified in the paper's text? |
|---|---|---|---|---|
| GCaMP6s | 125 ms | 550 ms | Chen et al. 2013, Nature 499:295, [doi:10.1038/nature12354](https://doi.org/10.1038/nature12354) | rise: the text gives "100-150 ms" (this game takes the middle); **half-decay: not found in the text** |
| GCaMP6f | 62 ms | 140 ms | same | rise: the text gives "50-75 ms" (middle); **half-decay: not found in the text** |
| jGCaMP8m | 58 ms | 137 ms | Zhang et al. 2023, Nature 615:884, [doi:10.1038/s41586-023-05828-9](https://doi.org/10.1038/s41586-023-05828-9) | both stated (Drosophila in-vivo visual responses: half-rise 58 +- 6, half-decay 137 +- 21 ms); used as the kernel's time to peak, a simplification |

**The two GCaMP6 half-decay numbers are the commonly quoted values, not something I could confirm in the paper's text. Please check them
against Chen 2013 Figure 1 before anyone cites them.** The same caveat is on the screen and in the NWB metadata.

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
