# Predators (3.0 day 3)

Three tools in the **Creatures** category, next to the spider: **FROG**, **DRAGONFLY** and **MANTIS**. Click to put one down (in the 3D game and in the 2D game; not in larva mode, which has no looming detectors in this model).

| | what it does | tag |
|---|---|---|
| detection | the fly "sees" a predator only as an object that **grows in its view**, through the game's existing looming transduction (angular expansion speed above `loom_min` drives the real looming detectors LPLC2 and LC4, which drive the giant fiber DNp01; DNp01 above the escape threshold makes it DODGE) | CONNECTOME + the existing GAME RULE transduction |
| capture | a capture fires the fly's real touch neurons by body part (frog: body, head, legs; dragonfly: legs, wings, body; mantis: body, legs) | CONNECTOME for what it drives, GAME RULE for the capture |
| the predators' behaviour | when each hops, creeps, chases, aims and strikes, how fast and how far, how many strikes, when it gives up | **GAME RULE**, all of it |
| the Predator escape assay | escape probability per predator with a 95% Wilson interval | **MODEL PREDICTION** |

No real frog, dragonfly or mantis is measured or implied: **no published speed, reach or timing is used** (`kickthefly/game/predators.py` has every number, with the reason it was chosen: play). The fly in this game is large (its thorax stands 0.38 m up), so the predators are scaled up to be bigger than it.

- **Frog** (`FROG`, hops since 3.1.0): **idle -> track -> crouch -> leap -> airborne -> land -> recover**, and a tongue strike (aim, strike, retract, regroup) when a fly is within 1.5 m. It never slides: it moves only while airborne, along one ballistic arc worked out from the distance to the fly's position at takeoff, one fixed apex (0.45 m) and the game's own gravity (`kick3d.GRAV`), so every leap takes the same 0.43 s and a fly that moves during the flight is not followed. A hunting frog lands 1.2 m short of the fly (leaps of 0.35-1.8 m); leaving the arena is the same hops in one direction. The body squashes with the hind legs folded in the crouch, stretches along the velocity with the legs extended at takeoff, pitches with the velocity in the air while the legs trail and then tuck, lands with the front legs reaching first, squashes and settles; the tongue overshoots its aim point by 8% and snaps back, and a blob shadow shrinks as the frog rises. The tongue takes **0.07 s** to reach the place the fly was when it aimed (it does not track). Up to three strikes, then it hops away. The frog's body, rising and falling, and the tongue's tip are objects growing in the fly's view; there is no other channel to the fly (looming only, **GAME RULE**: when, how far and how high it hops).
- **Dragonfly** (`DRAGONFLY`): patrols high up and chases a **flying** fly from above (4 m/s, a limited turn rate, aimed slightly ahead); it grabs within 0.3 m. A walking fly is not prey: it gives up after 30 s.
- **Mantis** (`MANTIS`): creeps toward the fly at **10 cm/s**, freezing now and then, until it is within 0.8 m, then aims for 0.2 s and strikes with a forelimb in **0.06 s**. The creep never expands faster than 0.1 rad/s, below the looming threshold (1.5 rad/s), so "sneak up slowly and it won't notice" is a consequence of that threshold. The strike is the opposite.
- A caught fly is held at the predator's mouth and eaten; an **immortal** fly (I) is let go instead, as it breaks out of the spider's silk.

## The assay (Lab > Assays > Predator escape; protocol `assay: predator_escape`)

Each fly (seed) faces each predator 3 times. The attack runs against a fly that stays where it is; what its eyes would see goes through the game's looming formula into the real brain, and the fly has **escaped** if DNp01 crosses the game's escape threshold (4x its calm rate) before the capture. Headless and deterministic.

Criteria, written in `kickthefly/lab/predators.py` before the first run: **P1** the mantis's creep never expands faster than the looming threshold (geometry); **P2** in at least 9 of 10 flies (seeds 1000-1009) DNp01 stays below the escape threshold during the creep. Nothing else is judged.

Result on seeds 1000-1009 (10 flies x 3 attacks each, 30 attacks per predator; criteria **P1 PASS, P2 PASS**):

| predator | escaped | escape probability | 95% CI (Wilson) | max looming before the strike |
|---|---|---|---|---|
| frog | 0 / 30 | 0.00 | 0.00 - 0.11 | 1.7 rad/s (its hops, 3.1.0; 5.0 rad/s with the 3.0 sliding hop) |
| dragonfly | 0 / 30 | 0.00 | 0.00 - 0.11 | 14.8 rad/s |
| mantis | 0 / 30 | 0.00 | 0.00 - 0.11 | 0.09 rad/s |

Read it as: in this model, none of the three attacks gives DNp01 time to cross its 4x threshold before the capture (0 of 90). The dragonfly's dive comes closest: in 2 of its 30 attacks DNp01 crossed in the very frame the fly was caught. The first Day 3 run counted those 2 as escapes (2 / 30, 0.07, CI 0.02 - 0.21); the 3.0 day 3 review counted them as captures, because the rule is "before the capture" and in the game a fly that is already held cannot escape (`tests/test_day3_review.py`). Every run now writes the P1/P2 verdict into its own `summary.json`. These are **MODEL PREDICTIONS**: they move if a game rule moves, and the fly does not move in the assay, so an "escape" is the brain's decision to dodge, not a flight that clears the tongue. In the game itself the fly does move, and a strike that arrives after it has already started to escape misses.

## Python, protocols, tests

```python
from kickthefly import Fly
r = Fly(seed=1000).attack("mantis")     # {'escaped': ..., 'captured': True, 'max_loom_before_strike': 0.09, ...}
```

Protocol block `predator: {kind: mantis, at_s: 0.5}` plays one attack as looming onto LPLC2/LC4 (`protocols/predator_mantis_creep.yaml`); `protocols/predator_escape_assay.yaml` is the assay. The playthrough bot uses each tool in both games (`extra:predators` checks that each attack ends in a capture, that the mantis's creep never looms and its strike does). Tests: `tests/test_predators.py`.


## Animation (3.1.0)

The frog hops; the other predators move like animals too. Everything here is drawing and timing (**GAME RULE**): the fly still sees a predator only as something that grows in its view, and a bite still fires the touch neurons it fired before.

- **Spider** (a tool, not an engine predator): eight two-bone IK legs in an alternating-tetrapod gait. A foot stays where it was planted until the body has walked away from it, then it steps in an arc, so the feet never slide. In reach it winds up for 0.28 s (rearing, front legs raised, nothing else moves), lunges for 0.08 s and bites at the end of the lunge if the fly is still within 1.45 reaches of it, else it misses; then it recovers for 0.4 s. A bite is a cycle a fly can dodge.
- **Mantis**: rocks side to side while it stalks (still while it freezes), rears back and cocks its forelegs through the aim pause, strikes and relaxes; four IK walking legs with planted feet.
- **Dragonfly**: pitch along its velocity, bank into turns, legs tucked on patrol and thrown forward into a basket as it closes on prey, pulled in once it has caught one.
- `kickthefly/game/predator_anim.py` holds the shared pieces; every engine predator has a `pose()` the games draw from.
