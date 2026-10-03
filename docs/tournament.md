# Fly tournament (Esc > Fly arcade, `--tournament N`) — 3.0 day 4

A bracket of 4, 8 or 16 flies, each a different **individuality seed** with its own **personality card**, fighting 1v1 duels in which
**both sides are brains**. You pick a favorite; the bracket shows every result, any match can be replayed, and the champion's brain
report says which neurons fired before its winning shots. Headless, the same bracket asks whether personality predicts winning.

| | what it is |
|---|---|
| **CONNECTOME** | what the neurons do with what they are given: turning is DNa01/02 right minus left, shooting is DNp35/DNpe052 above the game's `fire` threshold, the dodge is the giant fiber DNp01 above its `escape` threshold, the run is the body-touch group above its threshold. Each fly's own per-neuron gains (individuality) and noise are its own |
| **GAME RULE** | the arena (a flat 12 m square), the start (6 m apart, facing the other fly plus or minus 1.2 rad of match noise), how a turning command becomes a heading change, the blaster (the 3D duel's own cooldown, speed, spread and damage), the hit's touch poke and the reward / punishment pulses, the dodge and run hops, the match length (default 20 s), the pairing (seeds in the order given, 1 v 2, 3 v 4, ...), the tie-break |
| **MODEL PREDICTION** | who wins; which neurons fired before the champion's landed shots; whether personality predicts winning |

## How a duel works

It is the 3D game's existing duel (`kick3d._duel_senses`, `_duel_motor`, `_fly_shoot`) with a second brain where **you** were. Every
20 ms (4 brain steps) each fly is given what it can see of the other, exactly as in the duel against you: LC10 target tracking on the
side the other fly is on (which drives the same-side DNa02 steering neurons), the small-object detectors when it is dead ahead (which
drive DNp35), and looming from the other's pellets coming at it (the game's looming transduction onto LPLC2/LC4). Then each brain's
output decides: it turns on DNa01/02 right minus left, shoots when DNp35/DNpe052 are above threshold (with the blaster's cooldown),
hops sideways off an incoming pellet's line when DNp01 is above its dodge threshold, and runs away when it is hurt and the body-touch
group says run. A hit pokes the touch neurons of the fly that was hit, a punishment pulse on it and a reward pulse on the shooter, as in
the duel against you. First to 0 health wins; at the time limit more health wins. A draw is rematched twice with new noise, and only
then decided by a seeded **coin toss that is labelled as one** in the bracket and left out of every statistic.

The code is `kickthefly/game/flyduel.py` (the engine, pure: it takes two brain-like objects, so it is tested on scripted fakes) and
`kickthefly/lab/tournament.py` (the bracket, the cards, the analysis).

## The personality card is measured

The card the game shows for an ordinary fly is derived from its seed. The tournament's is **measured from the fly's own brain**
(`tournament.measure_card`): the same seed builds the same individual, and three readouts are taken at rest: the giant fiber's latency
to its dodge threshold when the looming detectors are driven (temperament), the sugar-pathway -> MN9 drive ratio (feeding), and the
right/left DNa01/02 firing ratio (steering). The trait words and their cut-offs are `core/individuality.compute_personality_card`'s. The
T-maze (learning) is not measured here: it costs about a minute per fly.

## The champion's brain report

For every shot of the champion that **landed**, the spikes of every neuron in the 200 ms before it are counted; per cell type the report
shows the firing in those windows against the type's firing over the whole of its matches. **It is a correlation, not a test:** nothing
was silenced, and a fly that was just hit also fires its touch neurons, so those lead the list. A second list shows the cell types the
duel's own rules read (DNp35, DNpe052, DNa01, DNa02, LC10, LC11, LC18, LC21, LC26, DNp01) whatever their rank.

## Backends

Matches are deterministic for a seed on the CPU backends (cpu, numba; torch-cpu is bit-exact only with individuality off). **The gl backend
does not implement individuality** (docs/individuality.md), so a tournament on it would be between clones; it is refused with that
reason instead. The arcade uses the backend `auto` picks; GPU backends are statistical, not bit-exact.

## Using it

- **Esc > Fly arcade > Tournament:** choose 4, 8 or 16, the duel length, **New flies** for a different set of seeds, click a fly to pick it as
  your favorite, then **Run tournament**. It runs in the background (worker processes, a progress bar, Cancel). Click any match in the
  bracket to replay it (play, pause, scrub); the champion's card and brain report are under the bracket. Settings > Brain > Individuality
  (`off`, `subtle`, `strong`) sets how different the flies are. Accessibility: the colorblind palettes colour the two flies and the heatmap-style
  marks, Larger text applies, and nothing flashes (replays are smooth). **Gamepad (3D):** the bumpers pick the favorite, the kill cam / big-view
  buttons change the bracket size, the trigger runs the tournament, the Neurodex button switches to Racing, B or Start closes the page. A
  tournament is not a tool, so it has no loadout slot.
- **Headless:** `python kick_the_fly.py --headless --tournament 16 --seeds 1000-1063 --workers 4 [--individuality off] [--match-seconds 20] --out DIR`
  runs one bracket per 16 consecutive seeds and prints the analysis (below).
- **Python:** `from kickthefly.lab.api import tournament; b = tournament(range(1000, 1008))` (the bracket plus `b["analysis"]`),
  `Fly(seed=3).card()`, and `Fly(seed=3).duel(Fly(seed=4))` for one match between two Fly objects.
- **Protocols, saves, replays, share codes:** unchanged; a tournament adds no new file format (its JSON/CSV results are plain exports).

## Does personality predict winning? (pre-registered; written in `lab/tournament.py` before any run)

Primary, one test per trait (looming latency, sugar -> MN9 ratio, log DNa01/02 right/left ratio): over the matches decided by the brains
(coin tosses excluded), how often does the fly with the higher trait value win? An exact two-sided binomial test against 1/2,
**Holm-corrected over the three traits**, alpha 0.05. Secondary, reported and not corrected: the Spearman correlation of each trait with
the number of rounds won. A bracket gives n - 1 matches, so one bracket has very little power; pool brackets. **Run it with
`--individuality off` as the control:** identical brains cannot differ in personality, so a "significant" result there would be a false
positive of the analysis. Seeds: the design was chosen on exploration seeds (3-4, 11-13); the report below uses 1000-1063.
