# Kick the Fly 2.11.0: you can see the decoy now

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over. Everything in
[2.10.0](https://github.com/legendarylolo318-cloud/kick-the-fly/releases/tag/v2.10.0) is in this build.

## For players

**The decoy female is visible in the 3D game.** In 2.10 she was there (flies touching her fired their foreleg taste
neurons) but never drawn. Now she's a female fly on the floor, slightly paler than yours, in every arena and in photo
mode. Pick her up and throw her with the hand, or right-click with the hand to remove her; R and changing arena clear
decoys, and save states keep them. Up to three at a time.

**See what she does to the brain.** While your fly's forelegs touch a decoy, a line at the top shows its foreleg taste
neurons for female pheromones firing and its P1 courtship neurons, live. P1 barely moves: in this simulation the
signal reaches the courtship neurons only weakly, and the game doesn't pretend otherwise (COURTSHIP is tagged RULE).

**The cVA puff is visible**: a short-lived cloud as far as it reaches.

## For researchers

Nothing in the simulation changed: the validation results are the same as 2.10.0. The contact readout is the live
firing of LgLG5-8 (MaleCNS v1.0 putative ppk23/ppk25 foreleg GRNs) in spikes/s against their calm rate, and of P1 as a
multiple of its calm rate.
