# Kick the Fly 2.10.0: pheromones, a decoy, and many flies on any GPU

Downloads: **KickTheFly.exe** (Windows) and **KickTheFly-x86_64.AppImage** (Linux). Check them against `SHA256SUMS`.
Your saves, settings and your fly's training memory carry over.

## For players

**A cVA pheromone puff.** Pick it from the mouse wheel or the toolbar: it drives the fly's real cVA-sensing smell
neurons, and you can watch the signal travel on in the brain view.

**A decoy female.** Drop one and let the fly walk into it: its foreleg taste neurons for female pheromones fire. The
COURTSHIP tag it earns is a game rule: in this simulation the signal reaches the courtship neurons only weakly.

**Many flies on the OpenGL backend.** With `--backend gl` (or Settings > Brain > Compute backend) a process's flies
now share the GPU: 16 flies keep real time and 32 run at 0.86x, where 2.9 managed 0.11x with 16.

**"Real flies do this too" cards are now off by default**, in Play and in the Lab. Turn them on in Settings > Brain;
the setting sticks.

**A language setting** (Settings > Accessibility). The pause menu and Settings are translated so far; German is
machine-translated and incomplete.

## For researchers

**Validation**, on the held-out seeds with the pass criteria unchanged
([docs/validation.md](https://github.com/legendarylolo318-cloud/kick-the-fly/blob/main/docs/validation.md)):

- **Pass:** DA1 ORNs (Or67d) -> DA1 projection neurons (x2.58 vs x0.80); DA1 PNs -> lateral horn / aSP targets
  (x2.22 vs x0.96). Activation, not cVA behavior.
- **Fail:** foreleg pheromone GRNs (putative ppk23/ppk25) -> P1 (x1.20, too weak); mushroom body extinction (PI 1.00
  vs 1.00) and second-order conditioning (odor B PI 0.00 vs -0.03 unpaired). No plasticity rule was added for either,
  so the brain would have to produce them through its own dopamine neurons, and it doesn't.

The 16 earlier tests give the same numbers as 2.9.0.

**Replay files.** `--headless --protocol FILE --record-replay OUT.ktfreplay` records a protocol run with every input
at its step and a checksum of the spikes; `--headless --replay FILE --out DIR` runs it again, exactly on NumPy, Numba
and torch-cpu. A replay from another brain pack is refused. The windowed game doesn't record replays yet.

**gl, batched.** One GL context holds the connectome once and up to 32 flies; each step streams the weights once for
all of them, and batched results are bit-exact with unbatched. One fly steps in 0.96 ms (2.37 ms in 2.9). Numbers and
method in [docs/performance.md](https://github.com/legendarylolo318-cloud/kick-the-fly/blob/main/docs/performance.md).

**Also:** a plume tracking assay (Python API; the plume and the navigation are game rules), new example protocols,
and CI that runs the gl backend on software OpenGL, builds the Flatpak, checks the docs' links and checks replay
determinism before any release.
