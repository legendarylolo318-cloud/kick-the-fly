# Microphone -> Johnston's organ (3.0 day 3)

**Opt-in. Off every time the game starts.** Esc > **Mic and streamer** > Microphone. While it is on, a red **MIC ON** pill is on screen (2D and 3D) and in that page. **Nothing is recorded, saved or sent**: sound is analysed in memory in 256-sample chunks and thrown away; the only thing kept is the latest handful of numbers. `tests/test_mic.py` checks it: the module contains no file, socket or network calls, and analysing a sound writes nothing.

The sound drives the fly's real **auditory Johnston's organ neurons, JO-A (50) and JO-B (88)** (the dataset's `JO-A*` / `JO-B*` types); JO-C/E, the wind neurons, are left to the wind.

| what | tag |
|---|---|
| which neurons: JO-A and JO-B. Johnston's organ neurons are grouped A to E; A and B are the vibration (sound) sensitive groups, C and E the deflection (gravity, wind) ones (Kamikouchi et al. 2009, Nature 458:165, doi 10.1038/nature07810; abstract read) | CONNECTOME + LITERATURE |
| JO-B prefers low frequencies (below about 100 Hz), JO-A higher ones; together about 10 Hz to about 1,000 Hz (a 2019 review in Frontiers in Physiology 10:1552, **read through a summarising tool, not by a person**) | LITERATURE |
| **the transduction**, every number: two 8th-order band-pass filters (B: 10-100 Hz, A: 100-1,000 Hz), the loudness that drives a group fully (RMS 0.05 of full scale at sensitivity 1), a noise gate (0.002), the current at full drive (0.5, the activation current of every pathway test), the same current on every neuron of a group | **GAME RULE** |
| the hum demo's responses | **MODEL PREDICTION** |

What it is not: a microphone is not an antenna (air pressure is treated as antennal vibration), the real organ is a mechanical resonator and each neuron has its own tuning, and the JO neurons here are very sensitive (any current reaches them: a 200 Hz hum leaks about 7% into the JO-B drive and still fires JO-B).

## The hum test (Esc > Mic and streamer)

Shows the loudest frequency, a 24-bin log spectrum (40 Hz - 2 kHz), how loud each band is as a share of full drive and how many JO-A / JO-B neurons are firing in the focused fly. **Run the hum demo (no microphone)** opens the assay below.

## The demo: does humming reach the courtship song pathway?

A **synthetic** hum (never the microphone) goes through the same analysis onto JO-A/B, and the courtship pathway is read out: the P1 cluster (the pC1 neurons the dataset calls P1), pIP10 and the ps1 wing motor neurons that the game's SONG reaction listens to. The song's rhythm it imitates (a train of pulses of about 220 Hz that recur every ~35 ms, and a continuous ~150 Hz hum) was read in search-result summaries of Current Biology 2024 (doi 10.1016/j.cub.2024.01.015), eLife 2015;4:e08477 and Bennet-Clark & Ewing 1969, **not in the papers**: treat the numbers as approximate.

Criteria (in `kickthefly/lab/audio.py`, fixed before the run): **H1** the 200 Hz pulse train at 35 ms drives P1 to at least 1.2x calm in at least 8 of 10 flies; **H2** silence does not (at least 9 of 10); **H3** the ps1 song motor neurons stay below the game's SONG threshold (1.8x) in at least 8 of 10 flies (the expectation from exploration seeds 0-2).

Result on seeds 1000-1009 (**H1 PASS** with exactly 8 of 10 flies, **H2 PASS** 10/10, **H3 PASS** 10/10), P1 firing as a multiple of calm, mean with 95% CI:

| sound | P1 | ps1 (song motor neurons) |
|---|---|---|
| silence | 0.94 [0.88, 1.00] | 0.87 |
| steady 200 Hz | 1.40 [1.32, 1.47] | 1.09 |
| **200 Hz pulses, 35 ms interval** | **1.39 [1.29, 1.48]** | 1.05 |
| 200 Hz pulses, 70 ms interval | 1.32 [1.21, 1.42] | 1.03 |
| steady 50 Hz | 1.30 [1.23, 1.38] | 1.09 |
| steady 600 Hz | 1.20 [1.16, 1.25] | 1.06 |

The honest answer to "does humming near the song's rhythm drive the song pathway?": **partly, and not selectively.** In this model, loud JO-A/B drive raises the P1 courtship cluster by about 1.2-1.4x, but it **does not reach the song motor neurons**, and it **is not tuned to the 35 ms rhythm**: a steady hum, a 35 ms pulse train and a 70 ms pulse train raise P1 about equally (the intervals overlap), and even a 50 Hz or 600 Hz hum does. (pIP10 is two neurons and its ratio is noise, reported but never judged.) So the wiring carries sound toward P1, but this model does not reproduce a song-rhythm filter.

```python
fly.hear(hz=200.0, seconds=2.0, ipi_ms=35.0)     # synthetic; {'peak_hz': ~200, 'jo_a_hz': ..., 'jo_b_hz': ...}
```

Protocols: `audio: {hz: 200, ipi_ms: 35, at_s: 1, seconds: 3}` (synthetic only; there is no way to name a device), `protocols/mic_hum_pulses.yaml`; the assay is `protocols/mic_hum_demo.yaml`. `--selftest` lists the microphone as **optional** (it looks for a capture device and never opens it). Settings > Brain > Microphone sensitivity is saved; the switch is not.
