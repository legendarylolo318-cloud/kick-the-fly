# Neurodex and Neuron of the Day (3.0)

The **Neurodex** is a collectible encyclopedia of cell types. A type is *discovered* the first time its neurons fire well
above their calm rate while you play. Open it with **D** (rebindable), from Esc > Neurodex, or with d-pad up on a gamepad (3D).

## What is what

| part | tag | where it comes from |
|---|---|---|
| neuron count, superclass, regions, predicted transmitter and its confidence, strongest input and output partner types, a cached EM skeleton | **CONNECTOME** | the MaleCNS v1.0 dataset, through the brain pack. Nothing is measured or estimated by the game. |
| "discovered", the collection, the progress bars, "discovered by stimulation" | **GAME RULE** | the rule below |
| a one-line fact and its citation | **LITERATURE** | `kickthefly/data/neurodex_facts.yaml`, hand-written, about 30 types |

**The discovery rule (GAME RULE).** Every 50 ms the game counts each type's spikes over the last 150 ms (from the simulator's
own spike record). A type is discovered when, for **3 checks in a row** after the brain has had 5 s to settle, its mean firing is at
least **6 spikes/s** and at least **3x the type's own calm rate** (never below 2 spikes/s), **and** that many spikes would be a
one-sided Poisson event of probability below **α ≈ 1.2×10⁻¹¹** if the type were firing at its calm rate. The calm rate is tracked
per type while nothing touches the fly. How it happened is recorded: *by stimulation* (brain surgery, a driven current or the laser
was on), *at rest* (nothing had touched the fly for 2 s) or *in play*.

**How the rule got here, and what still fails.** Day 1 had only the first two conditions, fixed before anything was played. On the
real adult pack (review, 2026-09-30) a calm, untouched fly then "discovered" about 190 types a minute, nearly all of 1-4 neurons
resting near 2 spikes/s, whose noise easily triples a small mean over 150 ms. The review fixed two pass criteria **before** changing
anything, on exploration seeds 0-4, individuality off and subtle:

| criterion | Day 1 rule | review rule |
|---|---|---|
| a calm fly discovers nothing in 60 s | **FAIL** (~190) | **FAIL** (25-35 per seed) |
| driving DNp01, MDN or DNp09 at amp 0.5 (the validation suite's activation current) discovers it within 3 s | (not measured) | **PASS** 30/30 (0.15-0.25 s) |

The Poisson condition was added with α derived from a stated budget (about one false discovery per 100 hours of calm play *if* spiking
were Poisson), not fitted to the measurement. It removed the small-type noise; what remains are sensory types (olfactory receptor
neurons such as ORN_VM5v and ORN_VA2, wing and body sensory neurons) whose spontaneous firing in this model comes in correlated
bursts, which break the Poisson assumption. Rather than tune the rule further against the same measurement, those discoveries are
labelled **at rest**, so an entry never claims you did something you didn't. Whether calm discoveries should count at all is an open
decision (see HANDOFF_3.0_DAY1_REVIEW.md).

Stimulating with brain surgery's gentle current (what Neuron of the Day's Try it uses in Play mode) discovers MDN, DNp09 and LPLC2 on
the real brain but not DNp01, MBON01 or PAM08 (seeds 0 and 1); the Lab laser and driven currents do.

Two limits, stated rather than hidden:

- The rule reads the type as a whole, so a large type that codes sparsely (the ~4,000 Kenyon cells, where an odor activates a few
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
citation was verified for the dataset's "ps1 MN" naming), and the DNg02 flight descending neurons. The E-PG entry matches the pack's
type `EPG` (checked against the real pack; `EPGt` is a different type and is not matched). Johnston's organ types are matched by the
prefixes `JO-C` and `JO-E` (the pack's names are JO-CA1, JO-CM, JO-ED1, JO-EV3 and so on).

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
