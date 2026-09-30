# Neurodex and Neuron of the Day (3.0)

The **Neurodex** is a collectible encyclopedia of cell types. A type is *discovered* the first time its neurons fire well
above their calm rate while you play. Open it with **D** (rebindable), from Esc > Neurodex, or with d-pad up on a gamepad (3D).

## What is what

| part | tag | where it comes from |
|---|---|---|
| neuron count, superclass, regions, predicted transmitter and its confidence, strongest input and output partner types, a cached EM skeleton | **CONNECTOME** | the MaleCNS v1.0 dataset, through the brain pack. Nothing is measured or estimated by the game. |
| "discovered", the collection, the progress bars, "discovered by stimulation" | **GAME RULE** | the rule below |
| a one-line fact and its citation | **LITERATURE** | `kickthefly/data/neurodex_facts.yaml`, hand-written, about 30 types |

**The discovery rule (GAME RULE).** Every 50 ms the game reads each neuron's firing rate (the simulator's own 200 ms average) and
averages it over each type. A type is discovered when that mean is at least **6 spikes/s** and at least **3x the type's own calm
rate** (never below 2 spikes/s) for **3 checks in a row**, after the brain has had 5 s to settle. The calm rate is tracked per type
while nothing touches the fly. The four numbers are the game's choice: they were fixed before anything was played and are not tuned.
A discovery made while you had brain surgery, a driven current or the laser on is tagged *discovered by stimulation*.

Two limits, stated rather than hidden:

- The rule reads the *mean* of the type, so a large type that codes sparsely (the ~4,000 Kenyon cells, where an odor activates a few
  percent) barely moves it and is mostly found by stimulating it.
- Regions are the coarse ones the game derives from the dataset's class and soma-neuromere annotations (Antennal Lobe, Mushroom Body,
  Central Complex, Optic Lobe, Central Brain, Gnathal, VNC, unassigned), not neuropil ROIs.

An undiscovered type shows only `???`: no name, no data. The search box finds only names you have discovered.
Partner counts are summed connection counts in the game's brain pack (connections below the loader's minimum weight, and unsigned
dopamine synapses, are not in it), and the share is of that type's total input or output synapses.

## Saving, larva, Python

- Saved per player next to the training memory: `memory/neurodex.json` (Linux `~/.local/share/kickthefly/memory`, Windows
  `Documents\Kick the Fly\memory`). A corrupt file is kept as `.bad` and the dex starts empty; a file from a newer version is never
  overwritten. Settings > Brain > **Neurodex discoveries** turns collecting off (what you found stays); **Reset Neurodex** in the panel
  erases it after a second click.
- The larva has its own list inside the same file. The windowed game runs the adult brain (docs/larva.md), so the larva dex fills only
  from code that steps a larval brain (the playthrough bot and tests do; `tests/test_neurodex.py` discovers KC on the real larval brain).
  Its "types" are the 17 cell classes of Winding et al. 2023's annotation table (KC, MBON, MBIN, LN, PN, LHN, DN-SEZ, DN-VNC, sensory...);
  the 346 neurons the dataset leaves "unassigned" are not a type and are skipped. Larval entries carry **no transmitter**: the larva pack's
  transmitter is the game's own sign rule (acetylcholine, or GABA for LN and MBON, at a constant 0.8), not the dataset's, so it is not
  shown as data. The larval superclass and region labels are also the larva loader's mapping from the class, not neuropil annotations.
  No curated facts (they are about the adult).
- Discovery runs only in the windowed game. It never runs in `--validate`, assays, protocols or tests, and never touches the network.
- Python: `fly.collect()` starts collecting during `fly.step()` into a throwaway collection (your own file only if you pass its path),
  `fly.neurodex("DNp01")` returns an entry, `kickthefly.core.neurodex` has the table, tracker and facts loader.

## The curated facts

Every fact is a paraphrase of what the cited paper reports, with its DOI and a `checked` note saying which part of the paper was read
(abstract or full text, through PubMed, PubMed Central or Crossref) and when. There are no driver-line names and no numbers the source
doesn't state. `tests/test_neurodex.py` refuses a fact without a citation, a duplicate, an over-long text or a prefix that would attach
a fact to types the paper doesn't cover (pC1 types that are not P1, for one). To add one: edit the YAML, add the paper to it, and read
the paper first. A type name is matched exactly (`types:`) or by prefix (`prefix:`).

What was deliberately left out: anything that could not be checked against a source. Two examples: the leg motor neuron ps1 (no
citation was verified for the dataset's "ps1 MN" naming), and the DNg02 flight descending neurons. The E-PG entry matches the names
`EPG` and `E-PG`; which one the real brain pack uses has not been checked against the pack (see the handoff).

## Neuron of the Day

A small card at launch: one curated type, its fact and citation, and a **Try it** button. **Separate** from the "Real flies do this too"
science cards: its own setting (Settings > Brain > **Neuron of the day**, default ON), its own "Don't show again" on the card, and it
appears once per launch after the tutorial. The type is chosen by date only: every curated entry comes up once per cycle of N days in a
shuffled order that changes from cycle to cycle. No clock reading beyond the date, no network, no history of what you did.

**Try it** (GAME RULE) sets up a one-click experiment and never runs it for you: in Lab mode a single type, or a prefix such as PAM, is
aimed at with the Lab laser (activate) and the laser is put in your hand; otherwise the type(s) are stimulated in brain surgery. It
replaces the surgery or laser target you had, and the card says so first. In the 3D game with the mouse captured the card tells you to
free the mouse (Tab); the same button is in the Neurodex panel (Esc > Neurodex) and on the gamepad (right trigger while the panel is open).

## Accessibility and gamepad

Reduced flashing: the discovery toast and the launch card only fade, nothing flashes. Larger text applies to the panel. The colors of the
brain view follow the colorblind palettes. Gamepad (3D): the Neurodex button (d-pad up) opens it; the bumpers move through the list,
d-pad down and Back change region, the trigger tries the Neuron of the Day, B or the Neurodex button closes it.

## D and walking

D is also walk-right in the 3D game. By design both bindings exist: in 3D, D opens the Neurodex only while the mouse is free (Tab), so
walking is never taken away. A config from before 3.0 keeps whatever its owner had on D: if another action already uses D the Neurodex
starts unbound (Settings > Controls). Rebind either and the overlap is gone.
