# Lesion battles (Esc > Fly arcade > Lesion battle) — 3.1.0 task 12

Two players, one screen. Each performs **surgery on their own fly**, within a budget, then the two flies fight in the Fly arcade's duel (a flat arena, a blaster,
100 health each). Nothing about who wins is scripted: both flies are real brains stepped together, and whatever the surgery does to the circuit is what decides it.

| | what it is |
|---|---|
| **CONNECTOME** | what the surgery does to the neurons: a silenced cell type receives a constant strongly negative current, a stimulated one a constant positive current (the game's own brain surgery, `kick_the_fly.SURGERY_CURRENT`), and the wiring does the rest; the duel reads the same neurons as the tournament (DNp35 / DNpe052 fire the blaster, DNa01 / DNa02 turn, DNp09 walks, the looming neurons dodge) |
| **GAME RULE** | the budget (3 picks per fly), the size cap (a pick may name only a cell type of at most 500 neurons, so "switch off the optic lobe" and "silence the mushroom body" are not options), the hot-seat order (player 2 builds without seeing player 1's picks; both are shown when the fight starts), the arena, blaster, health and match length (the duel's), the number of rounds (1, or best of 3), what a draw is |
| **MODEL PREDICTION** | who wins: a prediction about this model and these two seeds, not about real flies. A different pair of seeds can reverse it |

## Playing

1. **Player 1 picks** up to three cell types (search by name, or the suggested ones the duel reads), each silenced or stimulated, then **Done**. The screen hides the picks.
2. **Player 2 is ready**, picks the same way. A saved loadout can be loaded, the current one saved by name, copied as a code, or a code pasted in.
3. **Ready**: choose 1 round or best of 3, the round length (10 to 40 s), and **FIGHT**. The rounds run in the background (two processes), each with its own match seed and
   noise, always the same two flies. **New flies** changes the two seeds.
4. **Result**: a headline ("Player 2 wins 2-1 with LC4 silenced"), a replay of each round with health bars and shots, and the strongest readout each fly reached
   (fire, escape, run, walk), so you can see whether the surgery reached the thing the duel reads. **Rematch** keeps the surgery; **New battle** starts over.

A fly with no picks fights as it is. Both flies use the individuality setting you play with (Settings > Brain), seeded, so the same battle is the same battle.

## Sharing: an ordinary surgery code

A loadout is exactly a brain surgery that names only cell types, so it is shared as the existing `KTF1-SRG-...` surgery code (`core/sharecode.py`): Copy code, send it, Paste code
on the other side. Nothing new is added to the share format, and Esc > Share > Import accepts the same code as a surgery. Codes that are not a surgery, that also switch whole groups, that
name a type this brain does not have, or that break the budget or the size cap are refused with the reason. Saved loadouts live in `lesion_loadouts.json` in your data folder (20 names at most;
a damaged file is ignored).

## What was checked on the real brain

Seeds 5000 v 5001, subtle individuality, 10 s, one round:

| A | B | result |
|---|---|---|
| DNp35 and DNpe052 silenced | no surgery | Player 2 wins 1-0; A fires 0 shots (its highest fire level 0.18 against B's 4.19), B lands 2 |
| none | none | a draw, both fire (2 and 3 shots), both end at 91 health |

The same seeds and surgery gave the same health and shots twice. Silencing the two shooting neurons does stop the shooting, which the duel reads directly; that this loses to an
intact fly is a prediction of the arena's rules (a fly that cannot shoot cannot hit), not a finding about flies.

## Limits

Hot-seat only (no network). The budget and the cap are rules chosen to keep a battle about circuits rather than about size; they were not tuned to any outcome. The picker offers
suggested types the duel reads, and a hint is never an answer: most picks do nothing visible. Tests: `tests/test_lesion_battle.py` (14, one on the real brain).
