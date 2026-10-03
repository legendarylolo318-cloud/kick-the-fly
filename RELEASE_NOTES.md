# Kick the Fly 3.0.0: Neurodex, Lab toolkit, behavior rigs and mini-papers

The biggest release so far, and **the simulation is unchanged**: `--validate` gives exactly the same results as 2.13 (21 tests, 12 PASS /
9 FAIL, every value identical), and no weight, time constant or threshold was tuned to make anything pass. Everything new is tagged
**CONNECTOME** (read off the wiring), **GAME RULE** (written for the game) or **MODEL PREDICTION** (what the model then does).

**Download:** `KickTheFly.exe` (Windows, about 90 MB) or `KickTheFly-x86_64.AppImage` (Linux, about 110 MB) below; nothing to install, the
whole MaleCNS v1.0 brain (166,700 neurons) is inside. Check them against `SHA256SUMS`. The exe is not code-signed: SmartScreen may warn
(More info, then Run anyway).

## Highlights

- **Neurodex, kill cam, Neuron of the Day:** collect the cell types your fly fires, each with the dataset's numbers and, for curated ones, a
  cited fact; watch the last seconds of its brain in slow motion; one curated neuron at launch.
- **Behavior rigs** (Lab): a tethered flight simulator, a fly on a ball, Buridan's paradigm and a four-field olfactory arena, steering read
  from DNa01/DNa02 and walking from DNp09, each with a pre-registered assay and a control. On held-out flies the tethered rig, the ball and
  Buridan's paradigm reproduce the classic result; **the olfactory arena does not**, and that is reported.
- **Guided mini-papers:** six classic papers (von Reyn 2014, Tully & Quinn 1985, Shiu 2024, Hampel 2015, Ohyama 2015, Colomb 2012) as
  hypothesis, run, plot, and your result next to what the paper itself states, with how much of each paper was read.
- **Lab toolkit:** split-GAL4 driver lines, thermogenetics, a virtual patch clamp, simulated calcium imaging, pharmacology, experiment
  bundles that rerun and say whether results match, share codes, network science, a sleep-deprivation assay, a sensitivity analysis.
- **Play:** a frog, a dragonfly and a mantis seen only through the real looming neurons; rain and storms; a kitchen arena; a fly arcade
  (tournaments and races, every fly its own brain, in-game points only, nothing to buy); an opt-in microphone and an opt-in Streamer mode
  (both off at every launch).
- **Polish:** a What's New screen once after an upgrade, layouts that hold at larger text, in German and in a narrow window, and a self-test
  that checks your install and changes nothing.

Old saves, settings, training memory, pet files, replays and protocols keep working (checked on files made by 2.13.1 itself).

## Known limits

- Fixation in the rigs is far tighter than real flies', and the olfactory arena gives an odor no valence. Race R2 and R3, the larva tests,
  the motor half of the grooming circuit and the E-PG compass fail and stay in the tables.
- Three mini-papers rest on abstracts only (von Reyn 2014, Ohyama 2015, Tully & Quinn 1985).
- Live sensory inputs are delivered per screen frame, so a slow brain gets more stimulus per simulated second; Lab protocols and assays are exact.
- Never verified: a physical gamepad, a real microphone or Twitch connection, GPU compute backends beyond the self-test, and a person playing a
  full session in a visible window. Every number is the CPU reference backend's.
- German covers part of the interface.

Full list: [docs/changelog.md](https://github.com/legendarylolo318-cloud/kick-the-fly/blob/main/docs/changelog.md); how 3.0 was checked, its
decisions and its limits: [docs/3.0-review.md](https://github.com/legendarylolo318-cloud/kick-the-fly/blob/main/docs/3.0-review.md).
