# Predators (3.0 day 3)

Three tools in the **Creatures** category, next to the spider: **FROG**, **DRAGONFLY** and **MANTIS**. Click to put one down (in the 3D game and in the 2D game; not in larva mode, which has no looming detectors in this model).

| | what it does | tag |
|---|---|---|
| detection | the fly "sees" a predator only as an object that **grows in its view**, through the game's existing looming transduction (angular expansion speed above `loom_min` drives the real looming detectors LPLC2 and LC4, which drive the giant fiber DNp01; DNp01 above the escape threshold makes it DODGE) | CONNECTOME + the existing GAME RULE transduction |
| capture | a capture fires the fly's real touch neurons by body part (frog: body, head, legs; dragonfly: legs, wings, body; mantis: body, legs) | CONNECTOME for what it drives, GAME RULE for the capture |
| the predators' behaviour | when each hops, creeps, chases, aims and strikes, how fast and how far, how many strikes, when it gives up | **GAME RULE**, all of it |
| the Predator escape assay | escape probability per predator with a 95% Wilson interval | **MODEL PREDICTION** |

No real frog, dragonfly or mantis is measured or implied: **no published speed, reach or timing is used** (`kickthefly/game/predators.py` has every number, with the reason it was chosen: play). The fly in this game is large (its thorax stands 0.38 m up), so the predators are scaled up to be bigger than it.

- **Frog** (`FROG`): sits still, hops closer every 1.5 s, then aims for 0.3 s and shoots a tongue that takes **0.07 s** to reach the place the fly was when it aimed (it does not track). Up to three strikes, then it hops away. The tongue's tip is a second, fast-growing object in the fly's view.
- **Dragonfly** (`DRAGONFLY`): patrols high up and chases a **flying** fly from above (4 m/s, a limited turn rate, aimed slightly ahead); it grabs within 0.3 m. A walking fly is not prey: it gives up after 30 s.
- **Mantis** (`MANTIS`): creeps toward the fly at **10 cm/s**, freezing now and then, until it is within 0.8 m, then aims for 0.2 s and strikes with a forelimb in **0.06 s**. The creep never expands faster than 0.1 rad/s, below the looming threshold (1.5 rad/s), so "sneak up slowly and it won't notice" is a consequence of that threshold. The strike is the opposite.
- A caught fly is held at the predator's mouth and eaten; an **immortal** fly (I) is let go instead, as it breaks out of the spider's silk.

## The assay (Lab > Assays > Predator escape; protocol `assay: predator_escape`)

Each fly (seed) faces each predator 3 times. The attack runs against a fly that stays where it is; what its eyes would see goes through the game's looming formula into the real brain, and the fly has **escaped** if DNp01 crosses the game's escape threshold (4x its calm rate) before the capture. Headless and deterministic.

Criteria, written in `kickthefly/lab/predators.py` before the first run: **P1** the mantis's creep never expands faster than the looming threshold (geometry); **P2** in at least 9 of 10 flies (seeds 1000-1009) DNp01 stays below the escape threshold during the creep. Nothing else is judged.

Result on seeds 1000-1009 (10 flies x 3 attacks each, 30 attacks per predator; criteria **P1 PASS, P2 PASS**):

| predator | escaped | escape probability | 95% CI (Wilson) | max looming before the strike |
|---|---|---|---|---|
| frog | 0 / 30 | 0.00 | 0.00 - 0.11 | 5.0 rad/s (its hops) |
| dragonfly | 2 / 30 | 0.07 | 0.02 - 0.21 | 14.8 rad/s |
| mantis | 0 / 30 | 0.00 | 0.00 - 0.11 | 0.09 rad/s |

Read it as: in this model, a frog's 0.07 s tongue and a mantis's 0.06 s strike leave DNp01 too little time to cross its 4x threshold (it did in 0 of 60 of those attacks), whereas the dragonfly's longer dive gave it 2 of 30. These are **MODEL PREDICTIONS**: they move if a game rule moves, and the fly does not move in the assay, so an "escape" is the brain's decision to dodge, not a flight that clears the tongue. In the game itself the fly does move, and a strike that arrives after it has already started to escape misses.

## Python, protocols, tests

```python
from kickthefly import Fly
r = Fly(seed=1000).attack("mantis")     # {'escaped': ..., 'captured': True, 'max_loom_before_strike': 0.09, ...}
```

Protocol block `predator: {kind: mantis, at_s: 0.5}` plays one attack as looming onto LPLC2/LC4 (`protocols/predator_mantis_creep.yaml`); `protocols/predator_escape_assay.yaml` is the assay. The playthrough bot uses each tool in both games (`extra:predators` checks that each attack ends in a capture, that the mantis's creep never looms and its strike does). Tests: `tests/test_predators.py`.
