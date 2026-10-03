# Sleep deprivation assay (Lab > Sleep deprivation, `--sleep-deprivation`) — 3.0 day 4

Keep a fly awake through part of the night with timed disturbances, then measure its **rebound sleep** against an undisturbed control of the
same seed (paired statistics). It uses the day/night cycle and the **dFB readout** (the dorsal fan-shaped body neurons FB6/FB7).

## Which part is which (the assay prints the same table)

| part | tag |
|---|---|
| **sleep pressure** (0 to 1; rises while awake, falls while asleep) | **GAME RULE** |
| the disturbances (a touch on the legs and body every 3 s; each holds the fly awake for 2 s) | **GAME RULE** |
| what counts as sleep: dFB above `THRESH["sleep"]` (2.0x calm) for 1 s without a break | **GAME RULE** |
| pressure -> current on every FB6/FB7 neuron (0.08 x pressure) | **GAME RULE** |
| the compressed day (300 s; from 80% of the day, dusk; 10 s settle, 45 s deprivation, 60 s recovery) | **GAME RULE** |
| the dFB firing that current produces, read as a multiple of the neurons' own calm rate | **CONNECTOME** readout |
| daylight on the photoreceptors and on the morning clock neurons l-LNv/s-LNv | **CONNECTOME** (the drive is a game rule) |
| the rebound itself | **MODEL PREDICTION** |

The pressure rule is the **pet's** (`core/pet.py`): it rises 1 per 16 h awake and falls 1 per 8 h asleep, here compressed so a deprivation
fits in a minute (rise 1/60 per second awake, fall 1/30 per second asleep). **A rebound is expected from that rule** (a fly kept awake
has more pressure to discharge), so **this assay does not show that the fly's brain has a sleep homeostat.** What the connectome
contributes is the dFB's response: how much current the neurons need before they cross the sleep threshold. That is a property of the
wiring, and it is what the dFB columns report. The game's existing coupling (a poke that drives a random third of the dFB at any strength)
is an on/off switch, so this assay uses a graded current instead; the response curve on two exploration seeds (11, 12) was 2.8-2.9x calm
at a current of 0.04 and 4.6-4.7x at 0.08, and 0.08 per unit of pressure was fixed from that before any held-out run.

## Pre-registered criteria (written in `lab/sleepdep.py` before the run; seeds 1000-1009)

- **S1 rebound:** sleep in the recovery window, deprived minus control, one-sided Wilcoxon signed-rank over the seeds, p < 0.01 and a
  positive mean difference. With n = 10 an exact one-sided Wilcoxon cannot go below p = 0.00098; that floor is reported as **p < 0.001**.
- **S2 a manipulation check** (expected to pass by construction of the rule; it exists to catch a broken device): the deprived flies slept at
  most 25% as long as their controls during the window.
- **S3 the dFB readout:** the mean dFB level at the end of the window, deprived above control, one-sided Wilcoxon p < 0.01.

An earlier draft of the disturbance (a touch every 4 s holding for 2 s) left a 2 s gap in which a fly could sleep for up to 1 s: the deprived
fly still slept about three quarters as long as its control in the window, so S2 could not pass. That was found with a scripted brain, before
any held-out seed was run; the device was changed (every 3 s, hold 2 s) and S2 was reworded as the manipulation check it is. No threshold,
current or rate was changed to make S1 or S3 pass.

## Using it

- **Lab > Sleep deprivation:** pick the number of flies (seeds 1000 to 1000 + n - 1; at least 7 are needed to reach p < 0.01) and **Run the
  assay**. It runs in worker processes with a progress bar and Cancel, and shows each fly's numbers, the three criteria and the table above.
- **Headless:** `python kick_the_fly.py --headless --sleep-deprivation --seeds 1000-1009 --workers 4 --out DIR` (exit code 0 if all three pass).
- **Protocol:** `assay: sleep_deprivation` with `assay_options: {deprive_s, recover_s, mode}`; it is a paired design, so no `surgery:`.
  Lab > Assays has it too. **Python:** `from kickthefly.lab.api import sleep_deprivation`.
- Individuality is `off` by default (the effect is a population property); `--individuality subtle|strong` gives each seed its own gains.

## Result (this release; seeds 1000-1009, individuality off, NumPy CPU, 640 s with 3 worker processes)

| | control (undisturbed) | deprived |
|---|---|---|
| sleep during the 45 s window | 13.5 s | 0.0 s |
| sleep pressure at the end of the window (**GAME RULE**) | 0.34 | 1.00 |
| dFB level at the end of the window (**CONNECTOME** readout, x calm) | 2.11 | 4.51 |
| sleep in the 60 s recovery window | 20.0 s | 33.2 s |

**S1 PASS** (rebound +13.3 s, one-sided Wilcoxon **p < 0.001**, every one of the ten seeds positive; d_z = 24.5, large only because the rule is
nearly deterministic across seeds, not because it is a big biological effect), **S2 PASS** (0.0 s of sleep in the window; by construction), **S3
PASS** (dFB +2.4x calm, **p < 0.001**). Read the table with the first section in mind: the deprived fly's pressure reached its cap because
the rule says awake time raises it, and the rebound is that pressure being discharged. The part that is the connectome's is the dFB column:
**a sustained current of 0.08 on every FB6/FB7 neuron takes their firing to about 4.5x calm, and about 0.025 to 0.03 (pressure 0.3 to 0.4) is enough to cross
the sleep threshold of 2.0x**; the wiring decides that, and it is not tuned. This is not evidence that the fly's own brain tracks sleep need.
