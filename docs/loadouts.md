# Tool loadouts (2.13)

The game has eighteen tools. A **loadout** is the short list of them that sits on the hotbar, on the number keys. Every other
tool stays one gesture away on the tool wheel. A loadout is a convenience: it decides which tools you can reach with one
key and nothing else, so it never changes what a tool does to the fly, and the validation results are the same with any
loadout (nothing in `kickthefly/lab/` reads it).

## The hotbar

- Up to 10 slots on keys **1-9 and 0**, plus the mouse wheel (steps through the loadout, wrapping) and the gamepad's
  bumpers. A slot's key shows in its corner.
- A loadout longer than 10 tools has pages. **-** and **=** are previous and next page; the number keys follow the page
  showing. With 10 tools or fewer those two keys do nothing. The little arrows beside the hotbar turn the page with the
  mouse.
- Every key is rebindable in Settings > Controls (Hotbar slot 1-10, previous/next page, loadout editor, tool wheel), except
  Esc. A key that is taken swaps with the action that had it, as before. Before 2.13 the number keys, - and = were fixed.
- The **hand** is in every loadout, in slot 1, and can't be removed or moved.
- **Lab-only tools** (the laser) only appear in Lab mode. Switching to Play puts the laser down for the hand.
- **Larva mode** hides the tools that have no larval sensory mapping (bomb, spider, frog, dragonfly, mantis, alcohol, cVA, decoy, laser): the larva has no
  looming detectors, fermentation glomeruli, DA1 glomerulus or foreleg taste neurons in this model. The editor says why.

## Presets

| preset | tools (the hand is always first) |
|---|---|
| Base (Play's default) | hand, swatter, blowtorch, freeze spray, sugar |
| Chaos | bomb, blowtorch, brake cleaner, zapper, spider, alcohol, frog, mantis (the last two since the 3.0 day 3 review; appended, so keys 1-7 are unchanged; the dragonfly is in All and Lab, as it only hunts a flying fly) |
| Chemist | brake cleaner, alcohol, cVA, sugar, freeze spray |
| Lab (Lab's default) | every tool, the laser included |
| All | every tool the mode allows, in the order the number keys always had |
| Pet (Pet mode's default) | Base plus fruit (Base has no bomb, brake cleaner or zapper to take away) |
| Custom | your own list |
| Auto | Base in Play, Lab in Lab, Pet in Pet mode (the default setting) |

The preset is Settings > Controls > Tool loadout, and it is saved in `config.toml` (`[controls] loadout_preset`, your custom list
in `[loadout] custom`, up to five named loadouts in `[[loadout.saved]]`).

## The editor (Q)

A grid of every tool grouped by category (Touch, Thermal, Chemical, Reward, Creatures, Lab) with its icon, name, a one-line
description, and which real neurons it drives, tagged **CONNECTOME** (the sensory neurons the connectome really has) or
**GAME RULE** (something the game adds around them). Click a card to equip or unequip it; drag a card onto a hotbar
slot to put it there; drag hotbar slots to reorder; drag one off the hotbar to remove it. **Reset to preset** goes back to the
preset you started from. Type a name and **Save as new** to keep up to five custom loadouts; click one to load it.
Esc closes the editor (the game is paused while it is open). The editor is unavailable as a hotkey in photo mode, where Q flies
the camera down.

## The tool wheel

Hold the wheel key (default **`**, the backquote key; rebindable) and point with the mouse (in 3D the mouse points instead of
looking while the wheel is open); let go to take the tool. It shows every tool the mode allows, not only the hotbar's, with the
hotbar keys marked. Esc cancels. On a gamepad it is the Y button, as before, but it now lists every tool. Tab keeps freeing
the mouse; the wheel needed a key of its own.

## Fruit

The **fruit** tool (Reward) drops a piece of ripe fruit. It is eaten exactly as sugar is: the same sugar-pathway taste neurons and
PAM reward neurons (CONNECTOME), the same walking over and eating and the same scent and memory (GAME RULE); only the item
and its look differ. It is in the Pet preset so a pet can be fed the orchard's fruit anywhere.

## Migration and first launch

- A `config.toml` from before 2.13 (schema 2 or older) migrates to the **All** preset, in the order the number keys always had,
  so nobody's muscle memory breaks. One popup, shown once, says so and points at the editor. It also counts as already
  onboarded: no first-launch tutorial (Settings > Help replays it). The rest of the file is kept. Test: `tests/test_loadout.py`.
- A fresh install gets Base and the tutorial.
- Save states restore the tool by its index in the game's tool list, which only ever grows, so older saves load.

## Replays and the Python API

Replays (`.ktfreplay`, headless) record a `tool` event with the tool in use; it is informational, playing it back does nothing and
older versions skip it, so tool choices replay correctly. `fly.loadout` (a `Loadout`: `.tools`, `.page_tools()`), `fly.set_loadout("chaos")` or a
list of tool names, and `fly.use_tool("sugar")` (the tool's documented sensory neurons, driven once through `Brain.poke`) are in the Python
API ([api.md](api.md)); `Fly(mode="lab")` allows the laser.
