# Kick the Fly 2.9.0: song, fights, heat and night, and a brain you can trace

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over. Everything in
[2.8.3](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/tag/v2.8.3) is in this build.

## For players

**A thermo arena.** Press E until you reach it (2D and 3D): the floor runs from cold on the left to hot on the right,
and the fly's real hot- and cold-sensing antennal neurons fire more the further it strays from the comfortable middle.
The extremes hurt it. It doesn't walk to the comfortable middle by itself: that would need circuitry this simulation
doesn't produce.

**Day and night outdoors.** Settings > Brain > Day/night cycle makes the sun circle over the open field and the
orchard, and night fall. Daylight drives the fly's photoreceptors and its morning clock neurons.

**Courtship song and fights.** Switch on the song neuron pIP10 in brain surgery (O) and the fly buzzes a pulse song
through its real wing motor neurons. With several flies, switch on P1 and it lunges at the others. The sound and the
lunge are game rules; the neurons are real. Neither happens on its own in normal play, and the game says so.

**Find any neuron, and trace how two connect.** In the big brain view (B), type a cell type or body ID into the search
box. Inspect one neuron, press PATH FROM HERE, inspect another, press PATH TO HERE, and the strongest routes between
them (up to 3 synapses) light up, with spikes running along them live.

**Gamepad.** Play the 3D game with a controller: sticks walk and look, the right trigger uses the tool, the bumpers
or a tool wheel (hold Y) pick tools. Xbox, PlayStation and Switch Pro pads work without setup, and everything can be
rebound in Settings > Controls. Keyboard and mouse work exactly as before.

**Fixed:** the 3D brain panel's firing graph could crash the game on newer systems (NumPy 2 with pygame-ce 2.5).

## For researchers

**Validation grew, and was checked.** New tests, all on the held-out seeds with the pass criteria unchanged (see
[docs/validation.md](https://github.com/legendarylolo318-cloud/kick-the-fly/blob/main/docs/validation.md)):

- **Pass:** pIP10 -> ps1 wing motor neurons (x1.93 vs x0.94); optomotor, rightward wide-field motion -> the right
  steering neurons DNa01/DNa02 (x2.72 vs x0.80, the left pair x1.09), through a game-rule motion-detector stage in
  front of T4/T5; and four one-synapse activation tests: bitter taste neurons -> DNg28, CO2 neurons -> V glomerulus
  PNs, hot -> VP2 PNs, cold -> VP3 PNs. These four are activation, not avoidance, and get no "real flies" card.
- **Fail:** P1 -> ps1 (consistent but too weak, x1.43) and the Seeds et al. 2014 grooming hierarchy (head stimulation
  neither wins over nor suppresses abdomen grooming in this sim).

P1 isn't labelled "P1" in MaleCNS v1.0: it is the 25 pC1 types carrying the synonym "Cachero 2010: pMP-e; Yu 2010:
pMP4" (86 neurons). Every number was rerun before release; the NumPy, Numba and torch-cpu backends give identical
results.

**A Python API.** `from kickthefly import Fly`, then `drive()`, `silence()`, `step()`, `record()` and `export()` on a
whole-brain simulation, from a script or a notebook. Documentation in
[docs/api.md](https://github.com/legendarylolo318-cloud/kick-the-fly/blob/main/docs/api.md), with an example notebook.

**The OpenGL backend no longer re-uploads the whole connectome when the fly learns** (#2). Only the synapses that
changed go to the GPU: 68 KB and 0.03 ms per learning update instead of 41 MB and 2.1 ms. The game now loads and
runs on `--backend gl`, and the learned weights stay bit-exact with NumPy's. gl is still slower than NumPy here, so
`auto` doesn't pick it.

**Also:** new protocols (courtship song, male aggression, optomotor, grooming hierarchy, the thermosensory and CO2
pathways, light onto the clock neurons); new Lab parameters for every new reaction threshold; a Flatpak manifest for
building locally (`packaging/flatpak/`); and the README restructured, with Lab, validation and performance detail in
`docs/`.

Nothing that existed before changed its results: the eight earlier validation tests give the same numbers as 2.8.3.
