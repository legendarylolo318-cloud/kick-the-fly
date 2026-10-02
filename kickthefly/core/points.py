"""Arcade points (3.0 day 4): the in-game score that fly racing bets use. GAME RULE, and nothing else.

Points are not money and are not a purchase: they cannot be bought, sold, transferred or cashed out, nothing in the game costs real
money, and there is no code path from here to a payment of any kind. A wallet is one small JSON file in the player's data folder
(arcade_points.json). It starts at 100 points; below 5 it is topped up to 25 for free, so a player is never locked out of the arcade.
No telemetry: the file is never sent anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

START_POINTS = 100
TOP_UP_BELOW, TOP_UP_TO = 5, 25
STAKES = (5, 10, 25)
HOUSE_TAKE = 0.10                    # the odds pay out (1 - this) of a fair price: GAME RULE, so a bet is not free money
MIN_ODDS = 1.1


def decimal_odds(p: float) -> float:
    """Payout multiple for a fly with probability p of winning: a fair 1/p less the house take, to one decimal, at least 1.1."""
    p = min(max(float(p), 1e-6), 1.0)
    return max(MIN_ODDS, round((1.0 - HOUSE_TAKE) / p, 1))


class Wallet:
    def __init__(self, path: Path | None = None):
        from kickthefly.core import paths

        self.path = Path(path) if path else paths.get().data_dir / "arcade_points.json"
        self.points = START_POINTS
        self.history: list[dict] = []
        self.load()

    def load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.points = int(d.get("points", START_POINTS))
            self.history = list(d.get("history", []))[-50:]
        except (OSError, ValueError, TypeError):
            self.points, self.history = START_POINTS, []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(points=self.points, history=self.history[-50:]), indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass                                      # a read-only folder: the points just do not persist

    def can_bet(self, stake: int) -> bool:
        return 0 < int(stake) <= self.points

    def settle(self, stake: int, odds: float, won: bool, label: str = "") -> int:
        """Take the stake; if won, pay stake * odds. Returns the change in points. A bet the wallet cannot cover is refused."""
        stake = int(stake)
        if not self.can_bet(stake):
            raise ValueError("not enough points for that stake")
        delta = int(round(stake * float(odds))) - stake if won else -stake
        self.points += delta
        self.history.append(dict(label=label, stake=stake, odds=float(odds), won=bool(won), delta=delta))
        if self.points < TOP_UP_BELOW:
            self.points = TOP_UP_TO
            self.history.append(dict(label="free top-up", stake=0, odds=0.0, won=False, delta=0))
        self.save()
        return delta
