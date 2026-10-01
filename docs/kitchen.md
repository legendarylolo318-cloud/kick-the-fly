# The kitchen arena (3.0 day 3)

Press **E** (or `--arena kitchen`, or Settings > Brain > Arena) in the **3D game**. The 2D game stays in the room and says so (the same rule as the outdoor arenas). It is saved in save states (the fruit bowl's state too) and exported with recordings (`arena`, and the `weather.*` / `orchard.*` parameters). The countertop is the floor; the walls are tiled; there are cabinets, a window, a fridge and a hood.

| part | what it does | tag |
|---|---|---|
| **fruit bowl** | the orchard's feeding code with one "tree": the fly flies to the nearest ripe fruit, lands and feeds (sugar-pathway taste neurons and PAM reward neurons, heals); fermented fruit acts as alcohol does. The Lab's `orchard.*` parameters apply | CONNECTOME (taste, PAM, DM1/DM2/DP1m) / GAME RULE |
| **vinegar trap** | a jar whose smell pokes the **fermentation glomeruli DM1, DM2 and DP1m** (the same neurons the alcohol tool drives). The fly flies to the jar **only if those neurons' own firing is at least 2.5x its calm rate** (and 6 Hz): nothing sends it on a timer. A fly that hovers over the mouth for 0.4 s falls in and is **stuck**, struggling (leg touch, humidity) and slowly drowning; it can't get out (an immortal fly breaks out after 8 s) | CONNECTOME (the scent neurons) / **GAME RULE** (the jar, the reach, the trip, the trap physics) |
| **sink** | the pool's water rules inside a basin (0.3 m deep): floating, wet wings, humidity neurons, drowning | CONNECTOME (humidity neurons) / GAME RULE (the pool's rules, limited to the basin) |
| **stove burner** | the lamp's rules: heat sensors fire as the fly nears it; touching it hurts and pushes it off | CONNECTOME (heat sensors) / GAME RULE |
| **cook** | every 10-20 s a giant cook swings a swatter at where a fly was when he started (0.7 s wind-up, then a **0.9 s steady swing**). The swatter is an object that grows in the fly's view: the existing looming transduction does the noticing; a fly that sees it coming in time can dodge, one that doesn't is hurt (38 health) | CONNECTOME (looming, touch) / **GAME RULE** (that there is a cook, when, where, how fast) |

The first version of the cook swung in 0.28 s with an ease-in: the playthrough bot found the fly's looming neurons barely fired before impact (a peak of 5 Hz where 8 was needed), so a swat could not be dodged. The swing is now slower and steady and the swatter's looming radius matches the swatter plus forearm. That was a design fix for "a swat you can dodge", not a threshold tuned to pass. Whether a given fly dodges is a **MODEL PREDICTION**; the playthrough holds the fly still so it checks "seen" and "hurt", and reports how close the giant fiber came to the threshold (3.5x calm against 4x).

Honest limits: no real vinegar chemistry or trap catch rate, no heat transfer, a countertop that is the room's floor. "Cook you can dodge" is read as the fly dodging (the player is not hit in this mode).

Tests: `tests/test_kitchen.py`; playthrough `extra:kitchen` checks each station's documented neurons fire (sink: humidity; burner: heat; vinegar: DM1/DM2/DP1m; bowl: taste and reward), that a fly flying to the jar's mouth is trapped, that the cook's swat is seen as looming and hurts a held fly, and that E reaches the arena.
