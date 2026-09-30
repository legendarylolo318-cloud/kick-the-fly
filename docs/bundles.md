# Experiment bundles (3.0)

A bundle is one zip that holds an experiment and everything needed to check it and run it again.

```
protocol.yaml            the experiment
results/summary.json     (assays also result.json and per_fly.csv)
raw/                     the recorder's CSV / npz / NWB files and per-fly metadata, untouched
metadata.json            version, backend and precision as they actually ran, seeds, brain pack SHA-256, parameters, surgery,
                         individuality, arena, platform, the spike SHA-256 of every fly, and whether it can be rerun
ro-crate-metadata.json   an RO-Crate 1.1 description (https://w3id.org/ro/crate/1.1), every file with its SHA-256
README.txt
```

**Make one:** Lab > Record and export > **Bundle** (also on the Protocols page after a run); or
`--headless --protocol FILE --out DIR --bundle OUT.zip`. It bundles the last protocol run, or else the last live recording. A live
recording is a record, not an experiment (the stimuli were delivered by hand), so its bundle says `rerunnable: false` and why.

**Check and rerun one:** `--headless --rerun-bundle BUNDLE.zip --out DIR [--backend NAME] [--dtype float32|float64]` verifies every
file against its hash, refuses a bundle from another brain pack or a newer bundle format, runs the protocol again with the recorded
precision and (by default) the recorded backend, writes `DIR/rerun_report.json`, prints the verdict and exits 0 on a match, 1 on a
mismatch, 2 on a bundle it can't use.

## What "match" means

- **Bit-exact** when the recorded and the rerun backend are both CPU-side (`cpu`, `numba`, `torch-cpu`: tests/test_backends.py holds
  them identical) and the precision is the same: every fly's spike SHA-256 must be equal. One differing spike is a MISMATCH.
- **Statistical** when either side is a GPU backend (`torch-cuda`, `torch-rocm`, `gl`). Spikes are not compared. Each recording group's
  across-seed mean rate must lie within the original's 95% confidence interval widened by 10% of its mean (one seed has no interval, so
  within 25% of the mean plus 0.5 Hz); assays apply the same rule to every mean that has an interval. **These three numbers are the
  game's choice, fixed before any rerun; they are not a measured tolerance.**

A matching rerun shows the software is deterministic on that backend. It says nothing about whether the simulated result is
biologically true.

## What is in the crate, and what is not

The dataset it was computed from is named with its own license (CC BY 4.0, from the brain pack's documentation). **The bundle's own
license is left for its author to choose: none is stated.** A bundle is a container; it adds nothing to the simulation, uses no network
and no microphone, and is never part of `--validate`. NWB files in `raw/` need pynwb when the protocol asks for them.
