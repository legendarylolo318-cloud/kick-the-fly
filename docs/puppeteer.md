# Puppeteer mode (3.1.0)

Play mode, **Esc > Challenges > Puppeteer**, in the 3D game and in `--2d`. A puzzle mode in which **you cannot touch the fly**: every tool is locked except the optogenetics laser, nothing of yours
(a tool, your body, a kick) pokes its touch neurons, and nothing you carry has a smell. You steer it only by switching its **real cell types** on or off, and the level is won when its brain makes
the fly do the thing asked. Tags: **GAME RULE** for the mode's rules; each level says which part of what makes it work is the **CONNECTOME** and which is a rule.

## How you play
1. Pick a level. The fly is placed where the level says, facing the way it says, from the brain's state when you opened Puppeteer (so what one level did to the brain never leaks into the next).
2. Choose a cell type from the palette (keys 1-9, 0 for the first ten, or click): 16 real types named as the inspector names them, with no labels that give the answer away. Choose **Activate** or **Silence**.
3. **Laser**: aim at the fly and hold the button (in 3D, free the mouse with Tab to click the palette). **Latch**: choosing a type holds it on (or off) until you choose it again. Anything else is allowed too: the
   neuron inspector (B, click a neuron, SILENCE TYPE or STIMULATE), brain surgery (O). The puzzle is working out *which neurons*.
4. **Score = actions**: one per laser pulse fired at a chosen type, one per latch or surgery change, one per hint. **Par** is the fewest the author's own solution needed. Three stars at par or better, two up to par + 2,
   one otherwise. Best scores are kept in `scores.json` with the other challenges. The hint button names the author's solution and costs an action. Retry restarts the level.

## The ten levels
| level | the fly has to | solved by (author's solution, par) | the circuit |
|---|---|---|---|
| First steps | WALK | laser DNp09 (1) | **CONNECTOME** DNp09 (P9) is the forward-walking descending neuron; **GAME RULE** WALK = DNp09 above 3x calm, 1.2 s |
| About face | TURN R (it starts facing left) | laser DNa01+DNa02, right cells (1) | steering pair, right minus left (the validated optomotor readout); a turn above 2.1x; the side matters |
| The long walk | walk 0.9 m | latch DNp09 (1) | a pulse buys one step; a latch keeps the command on |
| Reverse gear | back up 0.5 m | latch MDN (1) | MDN is the moonwalker neuron. **In this model MDN does not reach the leg motor neurons (validation FAIL)**; the game's BACK UP reads MDN directly, so a game rule stands in for a circuit the model does not reproduce |
| Lift off | TAKE OFF | laser DNg02 (1) | DNg02 wing-power descending neurons above 1.58x |
| Serenade | SONG | latch pIP10 (1) | pIP10 to the ps1 wing motor neurons (validated, x1.93); the readout is read over about a second, so it wants steady drive |
| Clean antennae | GROOM | laser JO-C+JO-E (1) | Johnston's organ neurons excite aDN1/aDN2 (validated, x4.87); GROOM also needs the antennal neurons active, which the same drive does |
| Sleep tight | SLEEP | laser FB6+FB7 (1) | dorsal fan-shaped body; SLEEP above 2x, 2 s of rest |
| Nerves of steel | three looming shadows pass with no DODGE | latch **silence** DNp01 (1) | a shadow drives LPLC2/LC4, which excite the giant fiber DNp01 (validated, x11.8); silence the one exit and the dodge never comes |
| The detour | turn it round, then walk it 0.5 m the way it now faces | laser DNa right, latch DNp09 (2) | two circuits in order: steering, then walking |

Walking distance is what the fly walked **along the way it faces**: backing up counts negative, a wall that turns it round does not turn a walk into a backward distance, and flight counts for nothing. The
shadows of "Nerves of steel" (0.6 s every 3.5 s) go through the game's own looming drive onto LPLC2/LC4, and a dodge resets the count.

## How the fly "obeys"
Nothing new: the game already turns descending-neuron levels into moves (`THRESH` in `kick_the_fly.py`). A level only asks for one of those reactions, so every solution is the connectome acting on the fly. Wrong answers do
not win: for example MDN does not make it walk forward, DNp09 does not make it sing, and the left steering pair does not turn it right (tested on the real brain).

## What was verified
On the real brain (`tests/test_puppeteer.py`, needs the pack): **all ten levels were solved at par, in the 3D game and in the 2D game**, each from the same starting brain; eight wrong-neuron attempts did not win
within 6 s in either game. The game plumbing (you cannot hit the fly, the laser is the only tool, hotbar keys choose types, latches apply and clear a current, scoring, drawing every state) is tested on the synthetic pack.
Not verified: that a human finds the levels fun or fair, how hard the aim is in 3D (the tests aim perfectly), and the look of the 3D goal post (its drawing is tested not to crash).
