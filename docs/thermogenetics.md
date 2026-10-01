# Thermogenetics (3.0)

Lab > **Thermogenetics**. **TrpA1** activates, and **shibire-ts** silences, the neurons that express them once the temperature passes a
threshold. Expression is chosen by **cell type or driver line** (`type:DNp01`, `prefix:KC`, `line:SS00727`; Lab > Genetic toolkit finds
lines). The temperature comes from the **thermo arena** (each fly senses where it stands), a **Lab slider**, or a **protocol**.

## What is cited, what is a game rule, what is model

| | value | tag |
|---|---|---|
| TrpA1 onset | about **25 C**: heat ramps from 21 to 27 C evoked tonic spiking at approximately 25 C, with little adaptation, in larval motor neurons expressing dTrpA1 (Pulver et al. 2009, J Neurophysiol 101:3075, [doi:10.1152/jn.00071.2009](https://doi.org/10.1152/jn.00071.2009)); dTrpA1 as a warmth sensor: Hamada et al. 2008, Nature 454:217 (anterior cell neurons first show a fluorescence increase at 24.9 +- 0.6 C, n = 10). **Checked 3.0 Day 2 against the open-access full text (PMC2694103, PMC2730888):** Pulver's own numbers are spiking onset at about 25-26 C, an evoked-junction-potential threshold of 26 +- 0.2 C, no significant decay over 20 min of constant heating, and a deactivation threshold higher than the activation threshold (hysteresis, **not modelled**) | LITERATURE (approximate) |
| shibire-ts restrictive temperature | **30 C**: adults expressing shi-ts in cholinergic neurons were motionless within 2 min at 30 C and walked again about 1 min after returning to the permissive temperature; the shi product is needed for synaptic vesicle recycling (Kitamoto 2001, J Neurobiol 47:81, [doi:10.1002/neu.1018](https://doi.org/10.1002/neu.1018)). **Checked 3.0 Day 2 against the full abstract (PubMed):** "becoming motionless within 2 min at 30 degrees C ... started to walk in 1 min" after the shift back. The paper's full text is not open access and was not read, so the numbers are the abstract's only | LITERATURE (approximate) |
| full activation: TrpA1 at 29 C; shibire-ts ramp 28 -> 30 C (linear between) | | GAME RULE (the 29 C and 28 C ends are choices) |
| kinetics: TrpA1 1 s on and off; shibire-ts 40 s on, 20 s off (about three time constants inside "within 2 min" and "about 1 min") | | GAME RULE |
| size: TrpA1 is the validation suite's activation current (0.5); shibire-ts is brain surgery's silencing current (-0.6) | | GAME RULE |
| thermo arena: 15 C at the cold wall to 35 C at the hot wall, linear | | GAME RULE |
| only expressing neurons respond; temperature changes nothing else (no Q10 on any neuron or synapse) | | MODEL |
| shibire-ts really blocks vesicle recycling at the terminal; here the neuron is silenced, which also stops its spiking | | MODEL (coarser than the real effect) |

The Lab screen shows the same split next to each effector. "Instant" kinetics sets the steady-state level (what the assay uses); "Realistic"
uses the time constants (a 120 s shibire-ts experiment takes 120 s of brain time; protocols can compress it with `time_scale`).

## Protocols and Python

```yaml
thermogenetics:
  expression: [{effector: trpa1, target: "line:SS00727"}]
  temperature_c: [{at_s: 0, c: 22}, {at_s: 2, c: 32}]      # or a number
  kinetics: real                                            # real | steady ; time_scale: 1
```

`protocols/thermogenetic_dnp01.yaml`. Stimulus protocols only (assay protocols, and `--record-replay`, refuse it: its currents are not
replay events). `fly.express("trpa1", "type:DNp01"); fly.temperature(32); fly.step(1.0)`.

## Assay: thermogenetic activation of DNp01, escape rate vs temperature

`protocols/thermogenetic_escape_assay.yaml`, or Lab > Assays > **Thermogenetic escape**. Each fly is tested at 18, 22, 24, 26, 28, 30, 32 and
36 C, twice per temperature in an order shuffled by its seed: once with TrpA1 in DNp01 (steady-state kinetics, 1 s) and once as its own
no-expression control. **An escape is the game's own rule** (DNp01 above 4x its calm rate, `THRESH["escape"]`), so this measures the
model's response to a game-rule current, not a measured TrpA1 dose-response. **Decision (3.0 Day 2):** the step shape is kept and is not smoothed. The effector itself is graded (a linear ramp 25 -> 29 C, a game rule); the step is DNp01's, two neurons at their refractory limit, and tuning the ramp or the escape threshold to get a smoother curve would be tuning to a result. For a graded readout, express TrpA1 in a larger cell type and read its firing rate against temperature (`fly.express`, `fly.temperature`), or use the Lab slider.

**Pass criteria, written in `thermogenetics.py` before any run** (validation seeds 1000-1009):

- C1: at 18, 22 and 24 C the expressing flies' escape rate equals the control's (Fisher exact p > 0.05);
- C2: at 30, 32 and 36 C at least 9 of 10 expressing flies escape and at most 1 of 10 control flies does;
- C3: the escape rate never falls as temperature rises (Spearman rho >= 0.8 across the 8 temperatures).

### Result (validation seeds 1000-1009, CPU backend, run once after the criteria were fixed)

| temperature | expressing flies escaping | control flies escaping | DNp01 firing (expressing / control) | Fisher exact |
|---|---|---|---|---|
| 18 C | 0/10 | 0/10 | 5.2 / 5.5 Hz | p = 1 |
| 22 C | 0/10 | 0/10 | 5.3 / 5.5 Hz | p = 1 |
| 24 C | 0/10 | 0/10 | 5.5 / 5.7 Hz | p = 1 |
| 26 C | 10/10 | 0/10 | 50.0 / 5.5 Hz | p < 0.001 |
| 28 C | 10/10 | 0/10 | 66.9 / 5.5 Hz | p < 0.001 |
| 30 C | 10/10 | 0/10 | 67.0 / 5.5 Hz | p < 0.001 |
| 32 C | 10/10 | 0/10 | 67.0 / 5.5 Hz | p < 0.001 |
| 36 C | 10/10 | 0/10 | 67.0 / 5.0 Hz | p < 0.001 |

**C1 PASS, C2 PASS, C3 PASS** (Spearman rho = 0.845, so C3 clears its 0.8 bar narrowly: the curve is a step, which produces ties).

How to read it: this is not a graded dose-response. DNp01 is two neurons that fire at their refractory limit (about 67 Hz) once driven, so
a quarter of the TrpA1 current (what 26 C gives under the game-rule ramp) already crosses the 4x escape threshold in every fly. The step
at 25-26 C is therefore the game-rule onset passed through the model's own DNp01 response; the shape between 24 and 26 C is not a finding
about TrpA1. The control flies show that temperature alone does nothing in the model, which is true by construction (nothing else depends
on temperature).
