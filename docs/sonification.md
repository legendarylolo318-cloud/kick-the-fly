# Brain sonification (Settings > Audio, or F5) — 3.1.0 task 14

An opt-in audio mode: each brain region is a voice whose **pitch and volume follow how active it is**, mixed so it is music-like rather than noise, and the **courtship song is synthesized** when the song neurons fire. Works in the 3D game and in `--2d`. Off by default and off until you turn it on (Settings > Audio > Brain sonification, or **F5**, rebindable). While it is on, a small legend (top right of the play area) lists the voices with a loudness bar each, says it is a GAME RULE, and says why it is silent when it is.

| | what it is |
|---|---|
| **CONNECTOME** | what it listens to: the simulation's own smoothed firing rate of every neuron in each region (`brain.sim.activity.rates()`, the numbers the brain panel shows), and the song readout, the ps1 wing motor neurons that pIP10 and P1 drive (`Brain.level("song")`, the game's own SONG signal) |
| **GAME RULE** | everything that makes it a sound: the eight voices and their notes, the scale, how a rate against the region's own baseline becomes loudness and a step up the scale, the glides, the limiter, the song's pulses, and when it is silent. It is a rendering, not a recording and not a prediction |

It only reads the brain: with it on, every neuron's rate is identical to what it is with it off (tested), and no `--validate` number can move.

## The voices

| voice | region | calm note |
|---|---|---|
| Descending neurons | the descending neurons wherever they sit (superclass) | D2 |
| Nerve cord (VNC) | the VNC neuropils (T1-T3, abdomen, other) | A2 |
| Gnathal ganglion | GNG | D3 |
| Central brain | central brain | F3 |
| Mushroom body | MB | A3 |
| Central complex | CX | D4 |
| Antennal lobe | AL | G4 |
| Optic lobe | optic lobe | A4 |

The low voices are the motor end of the fly, the high ones the senses. A brain whose neuropils are not named like the adult's (the larva) gets one voice per region of 50 or more neurons.

## From activity to sound

A region's **activity** is its mean rate against its **own slow baseline** (its rate averaged over about 45 s, started at the first reading), in octaves: doubling is one. Loudness and pitch follow it:

* Calm and firing: a quiet presence on the voice's root note. Below half its baseline: faded out. Not firing at all: silent.
* Stirred up: louder (a compressed curve, nearly full at four times its baseline) and up the scale, two degrees per doubling, to six degrees.

**Why it is not noise.** Every voice climbs the *same* scale (D minor pentatonic: D F G A C), so any mix is consonant; pitch moves only between degrees and glides (90 ms); loudness follows with a 40 ms attack and a 250 ms release, so there are no clicks (tested at every chunk join); the sum is normalised by how many voices are loud and passed through a soft limiter that never exceeds 0.8 of full scale. The audio is made in 0.1 s chunks with continuous phase and plays about 0.1-0.2 s behind the brain.

## The courtship song

The song voice is the game's own song buzz, 220 Hz carrier in 12 ms pulses 35 ms apart (about the pulse song of *Drosophila melanogaster*, as a rendering), sounding while the song readout is above the game's SONG threshold (1.8 times its calm; it stops below 70 % of that so it does not stutter on the edge) and louder the further above it goes. The other voices turn down under it. It replaces the game's one-off buzz for the SONG reaction rather than doubling it (the buzz still plays with sonification off, or with the song voice unticked). On the real brain, driving pIP10 (2 neurons here) takes the song readout over the threshold and the song starts in about 6-8 s, which is when the game's own SONG appears: this is the model's pIP10 > ps1 pathway, not a stronger claim. (In the 3D run it was just under the threshold at 8 s: the same as the game's own reaction.)

## When it is silent

It follows the game's audio rules, and says which applies (in the legend and in the note when you turn it on):

* **Muted** (M, or Settings > Audio > Mute): silent.
* **Volume**: its loudness is the master volume times *Sonification volume* (default 40 %); either at 0 is silent.
* **No audio device**: silent, the rest of the game is unaffected.
* **Microphone on**: silent, because the speakers would feed the fly's own hearing (Johnston's organ) neurons through the microphone, a loop.
* **Streamer mode on**: silent unless *Sonification in Streamer mode* is ticked (default off): a continuous tone bed on a stream is the streamer's call. This is my reading of "respect Streamer mode"; it can be flipped in the settings.
* **Time paused**: silent.

## Limits

Eight voices from region means cannot say which neurons fired; for that, the brain stethoscope (K) clicks the spikes of one region. The mapping from rate to pitch is a design choice made once and not tuned to any result. No microphone, file or network is involved; nothing is recorded. Tests: `tests/test_sonify.py` (19: the mapping, the synthesis, the song's 35 ms pulses across chunk joins, the policy, and in both games that it follows a smell, a pIP10 drive, mute, microphone, Streamer mode and volume, and never changes the brain).
