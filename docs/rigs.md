# Behavior rigs (3.0 day 5)

Four classic ways of watching a fly, each its own scene in **Lab > Behavior rigs**: a tethered flight simulator, a fly on a ball,
Buridan's paradigm and a four-field olfactory arena. In every one the fly is a real brain, and what it steers and walks with is read from
real descending neurons; everything around the brain is a rule. Each rig has a headless protocol, a pre-registered assay with an
`--individuality off` control, and an export in the Lab's own format. **A result the model misses is reported as a miss**: one of the four
(the olfactory arena) does.

Code: `kickthefly/game/rigs.py` (the physics), `kickthefly/lab/rigassay.py` (protocols, assay, criteria, export),
`kickthefly/lab/labrigs.py` (the Lab scenes). Tests: `tests/test_day5_rigs.py`, `tests/test_day5_pages.py`.

## What is CONNECTOME, what is GAME RULE, what is MODEL PREDICTION

| part | tag | where it comes from |
|---|---|---|
| **Yaw** | CONNECTOME readout | DNa01 + DNa02, right minus left firing (`br.hz("turn_r") - br.hz("turn_l")`), the same neurons the duel and the game's TURN reaction read |
| how that becomes a yaw rate | GAME RULE | the duel's own rule: smoothed (0.35 per 20 ms tick), a 1.5 Hz dead zone, 0.24 rad/s per Hz, at most 2.1 rad/s |
| **Forward speed** | CONNECTOME readout | DNp09's firing against the race's one fixed reference rate (4.32 Hz), through the race's speed rule (version 2, `game/flyrace.py`): V_MAX 0.6 m/s at 3x the reference |
| **Wide-field rotation** (tethered rig, ball) | CONNECTOME after the T4/T5 layer, GAME RULE before it | `assays.emd_stage` / `outdoors.emd_motion_drive`: a rotation becomes current on the T4/T5 subtypes whose preferred direction it matches in each eye. The validation's `optomotor_turning` passes through exactly this stage (DNa01_R + DNa02_R x2.72 versus x0.80 for matched optic-lobe neurons, the left pair x1.09) |
| A stripe or a bar | GAME RULE on CONNECTOME | the duel's target-tracking rule: LC10 is poked on the side the object is on, harder the further to the side (nothing within 5.7 degrees of straight ahead). No validation test checks it (see "A probe" below) |
| One stripe at a time (Buridan) | GAME RULE | the stripe nearer the fly's heading is the one tracked |
| The platform, its edge, the ball, the VR, the arena, the odor's quadrants, sharp odor boundaries, the measures | GAME RULE | `game/rigs.py` |
| Whatever the fly then does | MODEL PREDICTION | each rig's criteria below |

Conventions: angles clockwise from above, a right turn positive; heading 0 is +y; engine ticks of 4 brain steps (20 ms).

## The four rigs

**Tethered flight simulator.** The fly does not move. A striped panorama rotates around it; the fly's yaw is read from DNa01/02 and
integrated into a heading that is only a readout. *Open loop:* the panorama follows its own schedule (a calm panorama, the rotation to the
right, a rest, the rotation to the left). *Closed loop:* the panorama is also turned by the fly's own yaw (`slip = omega - gain * yaw rate`), so
the fly can cancel the rotation. The visual input is the existing optomotor transduction.

**Fly on a ball.** A spherical treadmill with a virtual-reality world driven by the ball: ball turning from DNa01/02, ball forward
velocity from DNp09, integrated into a position and heading in the VR. Scenes: a *panorama* that drifts (the fly's turning feeds back as
slip) or a distant *bar* at +90 or -90 degrees from the start, tracked by LC10 (a distant landmark as in the classic closed-loop rigs: its bearing
changes only when the fly turns). *Open loop* freezes the VR. There is no sideways ball velocity: the model has no lateral descending readout.

**Buridan's paradigm.** A round platform (radius 0.5 m) with two opposite stripes beyond its edge (east and west, 1.5 m from the centre, not
reachable). The fly walks from the centre, turns on DNa01/02 and goes at DNp09's speed; the stripe nearer its heading is tracked; a fly reaching the
edge is reflected (specularly). The trajectory is recorded. The control has no stripes. Measures (GAME RULE definitions, after Colomb et al.
2012's): *stripe deviation* (the median over the run of the angle between the direction of travel and the line joining the stripes, 0-90 degrees,
45 = chance for a random walk) and *transits* (the east-west position reaches beyond 80% of the radius on one side and then the other).

**Four-field olfactory arena.** A 1 m square in four quadrants, the odor ("fruit", the game's own scent pulse of 0.3 per tick) in two opposite
ones, air in the others, sharp boundaries (a real arena mixes at the seams). The fly walks as in Buridan's arena (walls reflect it) with nothing to
track. The preference index is PI = (time in odor - time in air) / (their sum) after the first 10 s. The sham delivers no odor anywhere.

## Run them

```bash
python kick_the_fly.py --headless --rig buridan --seeds 3 --out DIR            # one fly (the first seed) in the rig, recorded
python kick_the_fly.py --headless --rig-assay buridan --workers 8 --out DIR    # the pre-registered assay, seeds 1000-1009, subtle and off
python kick_the_fly.py --headless --protocol protocols/rig_buridan.yaml        # the same as a protocol (rig: {name, mode, seconds, ...})
```

`--rig NAME` is one of `tethered`, `ball`, `buridan`, `fourfield`; the protocol block is documented at the top of `kickthefly/lab/protocol.py`
(`rig:` stands alone like `patch:`). Python: `kickthefly.lab.api.rig(name, seed, ...)` and `rig_assay(name, seeds)`.

**Export (the existing format).** A recorded run (a protocol, `--rig`, or "Record to the exports folder" in a scene) writes the Lab's own
recorder files for that fly: `NAME-seedN-spikes.csv`, `-rates.csv`, `-group-rates.csv` (the steering neurons right and left, DNp09, T4/T5
progressive and regressive, LC10), `-events.csv` (stimulus pokes, when there are any), `-kinematics.csv` (x, y, speed, an action label, the rig as
the arena; for the tethered rig the "speed" column is the yaw rate in rad/s), `.npz` and `-metadata.json` (with the rig, its parameters and its
summary); plus `-rig_trace.csv` (the decimated trace, every column of `rigs.COLS`) and `-rig_summary.json`.

## The pre-registered assay

The criteria are written in the docstring of `kickthefly/lab/rigassay.py` and were committed (f4b5f5c) **before any held-out run**; the design
(durations, definitions, effect thresholds) was fixed looking only at exploration seeds 0-9. Seeds 1000-1009 (n = 10), individuality `subtle` and
the control `off`; every criterion is a paired within-fly comparison and **passes when the mean effect reaches the stated minimum and the
one-sided Wilcoxon signed-rank p is below 0.01** (at n = 10 the smallest p is 1/1024, shown as p < 0.001). Only the default mode's result counts;
the control is shown beside it.

| criterion | what it asks | minimum |
|---|---|---|
| T1 | open loop: right-minus-left DNa01/02 after a rightward minus after a leftward panorama rotation of 3 rad/s | 3.0 Hz |
| T2 | closed loop: the share of an imposed 1 rad/s rotation the fly cancels | 0.10 |
| BL1 | the same on the ball, while walking, with the VR panorama drifting | 0.10 |
| BL2 | closed-loop bar: heading error with the bar hidden minus visible (last 5 s of 20 s), and the visible error under 30 degrees | 10 degrees |
| B1 | Buridan: stripe deviation without stripes minus with stripes | 10 degrees |
| B2 | Buridan: transits with stripes minus without | 3 transits |
| F1 | four-field: PI with the odor minus the sham | 0.15 |

### What validation predicted (written before the runs)

`optomotor_turning` passes, so T1, T2 and BL1 were expected to pass: the rotation reaches DNa01/02 through T4/T5 and the yaw is read from them, the
only rules in the loop being the EMD stage and the yaw gain. BL2, B1 and B2 depend on the duel's LC10 rule, which no validation test checks;
a probe on exploration seeds 0 and 1 found it steers the right way, so they were expected to pass, with a deviation far tighter than real flies'.
F1 has no mechanism: nothing in the validation suite connects an olfactory input to steering or walking, so it was expected to **fail**. The
E-PG compass tests FAIL (`docs/validation.md`), so no rig here asks for a heading memory.

### Results (seeds 1000-1009, CPU backend; `docs/results/day5/`)

| | default (`subtle`) | control (`off`) | |
|---|---|---|---|
| **T1** open-loop direction | **+8.58 ± 1.33 Hz**, p < 0.001 | +8.62 ± 1.54 Hz, p < 0.001 | **PASS** both |
| **T2** closed-loop compensation | **0.43 ± 0.07**, p < 0.001 | 0.39 ± 0.06, p < 0.001 | **PASS** both |
| **BL1** ball, drift compensation | **0.50 ± 0.06**, p < 0.001 | 0.46 ± 0.08, p < 0.001 | **PASS** both |
| **BL2** bar fixation | **+88.8 ± 10.7 degrees** (visible error 3.4, hidden 92.2), p < 0.001 | +86.7 ± 9.5, p < 0.001 | **PASS** both |
| **B1** stripe deviation | **+39.2 ± 14.3 degrees** (3.3 with stripes, 42.5 without), p < 0.001 | +41.0 ± 7.6, p < 0.001 | **PASS** both |
| **B2** transits | **+11.1 ± 4.0** (18.0 with, 6.9 without, in 110 s), p < 0.001 | +12.7 ± 1.8, p < 0.001 | **PASS** both |
| **F1** olfactory preference | **+0.005 ± 0.153 PI**, p = 0.69 (PI 0.027 with odor, 0.023 sham) | +0.024 ± 0.162, p = 0.36 | **FAIL** both |

What each result means:

- **The tethered rig and the ball reproduce the optomotor response** (a fly turns with a rotating panorama, and in closed loop cancels a
  share of it). That is the optomotor_turning validation seen through a body: the response is lateralized (the right pair rises about +3.9 Hz and the
  left falls about 0.7 Hz for a rightward rotation, the reverse for a leftward one) and it saturates early (a 1 rad/s rotation already gives +4.7 Hz,
  3 rad/s +4.6 Hz) because the EMD stage saturates by rule, so a *speed tuning curve* is not something this rig can show. The share cancelled
  (about 0.4-0.5) is the yaw gain's, a rule.
- **Bar fixation and stripe fixation are very strong and very regular** (about 3 degrees from the target, in every fly), much tighter than real
  flies', which Colomb et al. describe as fixation and antifixation with a chance level of 45 degrees. The reason is plain: the tracking rule pokes LC10
  hard on the target's side and the real DNa01/02 lateralization turns that into a steady turn toward it. What the connectome contributes is that
  lateralization and the walking drive; the *transits* are produced by the edge reflection and the rule that tracks the stripe nearer the heading, so
  B2 mostly measures those rules. The model has no pauses, no antifixation and no stripe-width dependence.
- **The olfactory arena does not reproduce a preference.** The fruit odor drives the receptor neurons, but nothing downstream steers or slows the fly in
  response: PI with the odor is indistinguishable from the sham (and from zero), and walking speed in odor versus air does not differ either (-0.2 mm/s,
  one-sided p = 0.58; reported, not a criterion). Real flies show odor attraction or avoidance in such arenas; this model's walking and steering neurons are not
  driven by an olfactory valence, and the mushroom-body learning that could supply one only enters through the T-maze. Nothing was tuned to change it.
- **Individuality does not matter for any of these.** The `off` control gives the same pass or fail and almost the same numbers in every rig:
  the results are the wiring's and the rules', not an accident of one fly's gains.

### A probe made while designing (disclosed)

The duel's LC10 rule had never been probed on the real brain (Day 4 review). On exploration seeds 0 and 1, poking `track` on the right moved
DNa01/02 right-minus-left by +8.6 to +9.7 Hz and on the left by -8.1 to -10.2 Hz (recruit 0.4, 3 s windows): the lateralization is as the rule assumes.
That check is not a validation test.

### Exploration (disclosed)

Everything was designed on seeds 0-9: the tethered probe (seed 0), the object-tracking probe (seeds 0 and 1), a first ball run where the bar was a near
landmark and the fly walked past it (fixed by making the bar a distant landmark before any assay), and ten-fly runs of all four rigs
(`subtle`, seeds 0-9) from which the thresholds were set conservatively below what was seen (T1 3.0 Hz against 8.4 seen, T2 0.10 against 0.45, B1
10 degrees against 39, F1 0.15 against -0.07). No held-out seed was run before the criteria were committed.

## Honest limits

- One run in a Lab scene is one fly. It shows what the model does, not a result; the assay does.
- No GPU backend was used; every number is the CPU reference backend's.
- The yaw gain, the dead zone and the speed rule are the duel's and the race's, not fitted to flies; a different gain would change T2 and BL1 (the share
  cancelled) but not their direction.
- A bar or stripe is a point-like target for LC10 in the duel's rule; real LC10 tuning is not modeled.
- The rigs read steering from DNa01/02 only. DNa02 and DNa01 are known to differ (DNa02 for high-gain, transient steering, DNa01 for low-gain, per
  Rayshubskiy et al. 2025, as the Neurodex says); the model adds their rates.
