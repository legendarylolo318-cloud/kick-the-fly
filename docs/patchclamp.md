# Virtual patch clamp (3.0)

Lab > **Patch clamp**, or click a neuron in the big brain view (B) and press **PATCH (MODEL)** (Lab mode). Current-clamp one simulated
neuron: a live membrane-potential trace with the firing threshold drawn in, spike markers, a holding current, current steps
(amplitude, duration, repeats), and an **I-F curve**. Export CSV or NWB.

## It is a point-neuron model, not electrophysiology

Every neuron in the simulation is the same leaky integrate-and-fire unit (one time constant, one threshold, reset after a spike). There
is no dendrite, ion channel, spike shape, adaptation or cell-specific property; neurons differ only in what they are wired to.

| | tag |
|---|---|
| the membrane potential is in **model units**: threshold 1.0, reset 0.0, the tonic drive alone rests at 0.8. Never millivolts | MODEL |
| injected current is in the simulator's units (0.5 is the validation suite's activation current; the simulator multiplies external current by 4) | MODEL |
| **embedded** mode: the neuron keeps the real synaptic input of the wired brain while you inject current | CONNECTOME |
| **isolated** mode: the same LIF equation with no synaptic input, run standalone; **every neuron gives the same curve** | GAME RULE |
| the step protocol, holding current and the I-F amplitudes | GAME RULE |

A spike is a marker on the 5 ms step where the potential crosses threshold and is reset; there is no waveform. The isolated unit's
arithmetic is the reference CPU backend's own update with the synaptic input removed (tested against `LIFSim` to 1e-6 with noise off),
so its rheobase follows from the equation: `bias + ext_gain x I >= leak x threshold`, i.e. I >= 0.0125.

## Exports

- **CSV**: time, membrane potential (model units), injected current, spike; header lines carry the MODEL tag and the units.
- **NWB** (needs `pynwb`): a generic `TimeSeries` in model units plus a `Units` table of spike times. It is deliberately **not** a
  volts `CurrentClampSeries`, because it is not volts.

## Protocols and Python

```yaml
patch: {neuron: "type:DNp01", index: 0, mode: embedded, amplitudes: [0, 0.05, 0.1], duration_ms: 500, repeats: 3}
```

`protocols/patch_dnp01_if_curve.yaml` runs it over several seeds and writes per-fly CSVs and a mean I-F table. `fly.patch("type:DNp01",
[0, 0.05, 0.1], mode="isolated")` returns the curve and the recording. A patch protocol stands alone: stimuli, recordings, surgery, a control, thermogenetics, a drug, imaging or a top-level warmup_s / duration_s are refused rather than ignored (its warm-up is `patch.warmup_s`). Currents must be finite and within +-5.
