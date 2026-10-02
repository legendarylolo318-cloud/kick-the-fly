# Fly racing (Esc > Fly arcade > Racing, `--race`) — 3.0 day 4

Flies race down a track with **sugar and fruit lures** along it, each through its **own brain and individuality**. You can bet **in-game
points** on a fly (never money: there is nothing to buy and no path to a payment of any kind); the odds come from the flies' measured
personality stats. Races can be replayed, and a headless **race assay** reports whether individuality predicts the finishing order.

| | what it is |
|---|---|
| **CONNECTOME** | the walking command neurons' firing (DNp09, "P9") sets the speed; a lure ahead drives the real olfactory neurons of its scent (the game's own scent pokes) and touching one drives the sugar-pathway taste neurons and the PAM reward neurons exactly as sugar and ripe fruit do in the game |
| **GAME RULE** | the 8 m track, the three lures (sugar at 2 m, fruit at 4 m, sugar at 6 m), the smell range (1.5 m) and contact distance (0.15 m), the pulse strengths (the game's: scent 0.3, taste 0.5, reward 0.5), the speed rule, the 90 s cap, the lanes (flies never touch or block each other, so each lane is its own simulation), the odds and the points |
| **MODEL PREDICTION** | the finishing order, and whether individuality predicts it |

**Speed rule (GAME RULE):** speed = 0.6 m/s x clip(DNp09 level / the game's walking threshold, 0, 1), where the level is DNp09's firing as a
multiple of the fly's own calm rate and the threshold is the game's `walk` threshold (3.0x). A fly at its calm rate goes a third as fast as
one whose walking command is above the threshold. Nothing is tuned to make races close or orders predictable.

## The odds, and the points

Before a race each fly's card is measured (the same measured card as the tournament, plus its calm walking drive). **Form** is the mean of
two z-scores inside the field: the sugar -> MN9 ratio (how hard the lures' taste drives it) and the calm walking drive (which sets the
speed). P(win) = softmax(form), and a bet pays stake x (0.9 / P(win)), rounded to a tenth and at least 1.1. All of this is a **GAME RULE**: a
modelling choice, not a finding. The race assay measures whether it works.

Points live in `arcade_points.json` in your data folder: you start with 100, stakes are 5, 10 or 25, and below 5 points you are topped up
to 25 for free. They cannot be bought, sold or cashed out. There is no telemetry and no network use anywhere in the arcade.

## Using it

- **Esc > Fly arcade > Racing:** choose 3 to 8 lanes, **Look at the field** (measures the cards, a few seconds each, in the background),
  read the odds, click a fly and a stake, **Start race**. The replay animates the lanes; the result and your points are shown.
- **Headless race assay:** `python kick_the_fly.py --headless --race --seeds 1000-1047 --lanes 6 --races 8 --workers 4 [--individuality off] --out DIR`.
  Each fly runs the same race twice (new noise each time), so a fly's finishing time can be compared with itself.
- **Python:** `from kickthefly.lab.api import race; r = race(range(3000, 3006), repeats=2)`.

## Does individuality predict the finish? (pre-registered; written in `lab/racing.py` before any run)

Primary family, Holm-corrected, alpha 0.05: **R1 repeatability** (Spearman correlation of a fly's finishing time between its first and
second run, across all flies, one-sided), **R2 form** (the form score against the mean finishing time, one-sided), **R3 the odds** (the Brier
score of the win probabilities beats a uniform field's, per race, one-sided Wilcoxon over the races; a floor of p < 0.001 is reported as
that). Secondary, uncorrected: each card trait against the mean finishing time. Run `--individuality off` as the control: identical
brains should show no repeatability.
