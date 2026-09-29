# The playthrough bot (2.12)

```bash
python kick_the_fly.py --headless --playthrough all --out playthrough-report            # adult and larva, the full matrix
python kick_the_fly.py --headless --playthrough adult --playthrough-quick --out report  # each tool once in the room and the orchard
python -m pytest tests/playthrough                                                       # the bot's own tests and a small slice
```

It is the answer to "does everything still work": scripted runs that, for every **arena x tool x brain**, use the tool on the fly and
check what should hold. The code is `kickthefly/lab/playthrough.py`; it runs from source, the exe and the AppImage. It uses a temporary
`KICK_THE_FLY_HOME`, so it never touches your settings, saves or training memory. Exit code 0 if nothing failed (skips and gated combos
are fine), 1 otherwise. The report is `playthrough.json` and `playthrough.md` (a table of combo, verdict, seconds, and failures with
their tracebacks) in `--out`.

## What each combo asserts

- **no exception**, anywhere in the run;
- the **sim/real ratio** (simulated seconds per wall second, brain steps included) stays above a floor. The floor (default 0.25,
  `KTF_PLAYTHROUGH_MIN_RATIO`) only catches hangs and collapses; it is not a performance target (a typical CPU run is 3-5x);
- the tool's **documented sensory neurons fire above their own calm baseline**. "Documented" is `ToolInfo.probes` in
  `kickthefly/core/loadout.py`, the machine-readable form of what the loadout editor says the tool drives. "Above baseline" means the
  busiest 100 ms after the tool was used is more than 4 standard deviations above the mean of the calm 100 ms windows before it,
  and at least 15% and 0.5 Hz above it. The calm rate and its spread are measured in each run (the PAM neurons sit near 30 Hz calm, a
  touch group near 2 Hz), so it is not a ratio someone picked to make tools pass. A probe whose neurons the brain doesn't have (the larva's
  reward neurons) is noted, not checked;
- **reactions and meters stay in range**: health, pain, group levels and membrane potentials are finite and inside their ranges;
- **death and the autopsy** work where the tool kills it in the run (the run keeps using the tool, clicking again, holding, tracking the fly,
  until it dies or the time is up; then the autopsy report must exist and not be empty);
- **save then load in the middle of the run restores state**: body positions, every membrane potential, health, the tool and the arena;
- a **replay** of the run's brain-level leg reproduces its spikes exactly on the CPU backends (numpy, numba, torch-cpu) and is only
  reported on GPU backends. The recorded run includes the tool choice as a replay event.

## What is gated by design (not run; the message a player gets is asserted)

- the larva in an arena it can't use (`game/larva.py:is_arena_allowed_for_larva`);
- the larva with a tool that has no larval sensory mapping (the editor's "No larval sensory mapping" reason);
- the laser outside Lab mode ("Lab mode only");
- the 2D game in the open field or the orchard ("needs the 3D game").

## Where the tools are used

Adult flies run in the real games, stepped in lockstep from one thread, frame by frame (`game.update`, then 3 brain steps: the game's own
real-time pace): `Game3D` (no window, no renderer) in every arena, and the 2D `Game` in the indoor arenas in the full run. The larva has no
windowed game yet ([larva.md](larva.md)), so it runs at brain level: the tool's documented stimuli through `Brain.poke`, the same call the
game's tools make. Every tool also gets a brain-level run on both brains; that leg is the recorded and replayed one.

## Also covered

Multi-fly spawn and despawn up to 8 (R despawns; the extra brain threads stop); brain surgery on and off (silence, restore, stimulate the
looming detectors); training with 5 fear pairings (the approach MBONs' response to the scent falls and weakened synapses appear); a duel
start and end; pet mode catch-up over a simulated 3-day gap (deterministic, clamped to 0-1, the sleep event, and death only with real stakes);
individuality off / subtle / strong (the gains' spread is the documented sigma, all positive so no sign flips, deterministic per seed, three
different runs); and every loadout preset in every mode, on both brains, with each slot used. The 3D renderer runs offscreen where
OpenGL is available (the real `Renderer` into a framebuffer, checked for a non-blank frame); with no GL its checks are **skipped**, never failed.

## In CI

Every pull request runs `--selftest` and the quick playthrough (`tests.yml`, "selftest and playthrough"); a release also needs the full
playthrough and the self-test to pass (`release.yml`); the nightly workflow runs the full playthrough with the full validation and publishes
the history ([ci.md](ci.md)). The larva pack isn't built on CI (it is built locally from Data S1, see [larva.md](larva.md)), so CI runs the adult brain and
the bot records the larva combos as skipped when its pack is missing.

## What a full run looked like (2.12, NumPy CPU backend, 24-core desktop)

`--headless --playthrough all` took about 10 minutes and made 391 rows: 306 passed, 1 failed, 69 gated by design, 10 inconclusive (skipped). Every one
of the 135 save-then-load checks restored its state, all 69 replays reproduced their spikes exactly, and 106 combos ended in a death whose autopsy
report existed. The sim/real ratio was 2.8x to 46x (median 4.1x).

- **Inconclusive (a skip, never a pass):** a documented group that stays silent is only a failure when the run could tell. It is recorded as a skip, with the
  reason, when the tool's effect never reached the fly (the fly never ate or drank the item: sugar, fruit and alcohol in the pool, where it floats; alcohol
  under the lamp) or the arena is already driving the same neurons (a light hand grab on a fly stuck to flypaper, in the pool or in the escape room, where the touch
  neurons run 10-15x above rest). Each brain also starts every combo from the same saved calm state, because the PAM neurons' calm rate otherwise
  drifted from about 30 Hz to about 55 Hz over a run of sugar combos.
- **One real failure, kept as a finding:** in the 2D game's flypaper arena the spider kills the stuck fly, but the looming detectors LPLC2/LC4 barely rise
  (4.4 Hz peak against 3.3 Hz calm). Looming is computed from how fast the spider grows in the fly's view within one frame; it is strong in every other arena
  (40-49 Hz) and marginal in the 2D pool. Nothing was changed to make it pass.

## A failure is a finding

The bot never loosens a criterion to pass. A tool whose documented neurons don't fire is reported with the numbers (peak, needed, calm mean and
spread). When the harness itself was at fault (the fly not being where the tool was aimed) the fix is in the harness, and the git history says so.
