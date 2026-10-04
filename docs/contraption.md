# Contraption builder (Esc > Challenges > Contraption) — 3.1.0 task 13

A Rube Goldberg sandbox that runs against the fly, in the 3D game and in `--2d`. You place parts on a side-view bench, run the machine, and the fly meets it.
The machine is a small deterministic simulation (`game/contraption.py`: no randomness, no brain, no pygame); everything it does *to the fly* goes through the game's own code.

| | what it is |
|---|---|
| **CONNECTOME** | what a part does to the fly's neurons. Tool parts fire the game's own tools at their position (the same `use_tool` / `use_tool3d` code you reach by clicking), so a swat, a bomb, a zap, a cVA puff, sugar, alcohol, fruit, a decoy, a spider and a flick drive exactly the neurons they drive in play. A fan drives the Johnston's organ wind neurons (`Brain.poke("wind")`) and a lamp the photoreceptors (`Brain.poke("light")`), the regions the fan and lamp arenas use |
| **GAME RULE** | the bench (4.8 m wide, 3 m high, the plane z = 0 in 3D), the parts and their physics (a marble that rolls, a domino that tips in 0.2-0.4 s and strikes whatever it reaches, a spring that throws at 4 m/s), the channels, the 24-part and 3-marble limits, the one-shot buttons, the fan blowing 4 s each time it is triggered, the tether, the 60 s cap |
| **MODEL PREDICTION** | what the fly does when the contraption runs (the report lists the game's reactions with their times) |

## Parts

| part | what it does |
|---|---|
| Marble | rolls from its place when the run starts, or after a delay (up to 3) |
| Ramp | a slope the marble rolls on (angle, length); one-sided, a marble passes up through it from below |
| Spring | throws a marble up, and sideways if given an angle |
| Domino | stands; a marble or a falling domino tips it into whatever is in reach (the next domino, a button, the fly) |
| Fan | when its channel goes off it blows for 4 s: pushes marbles in front of its mouth, and the fly in the stream feels it |
| Lamp | when its channel goes off it is lit for 1-10 s: drives the fly's photoreceptors by distance |
| Sugar | drops sugar when its channel goes off (the game's sugar item, which the fly smells and eats) |
| Tool | fires one of the game's tools here when its channel goes off (flick, swatter, bomb, zapper, cVA, alcohol, fruit, decoy, spider), in a burst of 1-5 shots 0.3 s apart |
| Button | pressed by a marble or a falling domino; sends its channel's signal, once |
| Timer | sends its channel's signal at a time, and again every so often, a number of times |
| Fly | where the fly is put at the start, which way it faces, and whether it is tethered |

**Channels** wire parts together without drawing wires: a part's coloured dot is its channel. Channel 0 goes off when the run starts; a button or a timer sends channels 1-5, and every fan, lamp, sugar and tool part on that channel goes off. Held tools (torch, cleaner, freeze), the laser and the predators that are spawned and then act on their own are not parts; the Tool part's list is what is one click at a place.

**The tether (default on).** Walking is the fly's own, and left alone a fly in the 2D or the 3D game wanders off a bench plane in seconds, so a contraption aimed at it would miss. Tethered, the fly's body is slid back to its mark each frame (it can still rear, jump, flail in place), and its brain reacts as it would; untick it on the Fly part to let it roam. A tool that throws the fly about only moves it when it is not tethered (tested).

**What a part does not do.** A fan does not push the fly's body, and a lamp does not make it walk toward the light; those are the arenas' rules and a contraption leaves them out so that each part does one thing. A marble or a falling domino that meets the fly fires a flick at it (the game's flick: touch neurons, a small push, 4 damage).

## Using it

Esc > Challenges > **Contraption**. Pick a part on the left (or "Select / move"), click the bench to place it; click a part to select it and use the `-` and `+` buttons for its fields; click an empty place to move the selected part. **RUN** puts the fly at its mark and starts the machine; **Stop** and **Edit** are on the bar. After the machine settles the run goes on 3 s so you can see the fly's reaction, then a report lists what went off and when, and what the fly did. Three examples are built in (Domino swat, Fan and lamp, Bait and bomb); eight save slots (Slots > Save or Load, a `*` marks a filled one); **Copy code** and **Paste code** share a build.

In 2D the machine is drawn on the arena, side-on; in 3D it stands along the room's x axis in the plane z = 0. In 3D the game's tool code reads where you aim and where the tool is, so for a fraction of a second (the swatter's swing) it is pointed at the part and at the fly; ending the challenge puts it back (tested).

## Sharing

A build is a `KTF1-CTN-...` share code (`core/sharecode.py`, kind appended so earlier kinds keep their bytes): whole numbers only (cm, degrees, tenths of a second), defaults left out, at most 24 parts. Esc > Share lists it too, and Import saves a code into the first free slot (it does not run it, and it overwrites nothing). A code naming a tool this install lacks (Lab-only tools in Play, or larva mode), an unknown part, an off-bench position or an out-of-range number is refused with the reason.

## What was checked on the real brain

All three examples were run in both games (seed 5, individuality off, the CPU engine): the Domino swat's four dominoes fall in order, press the button at 2.9 s, the swatter swings at 1.00 m, and the fly takes off (FLY AWAY) and kicks about 0.15 s later in 2D and in 3D; in Fan and lamp the fan comes on at 4.0 s and the fly flies away 0.3 s (2D) and 0.6 s (3D) later (the fan arena's own wiring: wind reaches the head-touch escape neurons through the JO neurons); in Bait and bomb the sugar is dropped at the start, a marble thrown by a spring presses the button at 2.2 s and the bomb goes off a 1.5 s fuse later. That the same reactions follow in every seed is not claimed. Tests: `tests/test_contraption.py` (37, 12 of them on the real brain, 2D and 3D).

## Limits

The physics is deliberately simple (a rolling point, rods that tip, no stacking, no friction between parts) so that a build behaves the same every time and a code reproduces it exactly. It is a toy machine, not a rigid-body engine. Not verified: the 3D swatter's swing against a fly that moves during the 0.12 s before contact (the tether makes this moot by default).
