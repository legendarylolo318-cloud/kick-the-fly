//! Arcade points: the in-game score that fly racing bets use. GAME RULE, and nothing else. Python: `core/points.py`.
//!
//! Points are not money and are not a purchase: they cannot be bought, sold, transferred or cashed out, and there is no code path from
//! here to a payment of any kind. A wallet starts at 100 points; below 5 it is topped up to 25 for free so a player is never locked out.
//! The wallet is a small JSON file in the player's data folder; nothing is ever sent anywhere.

use std::path::PathBuf;

pub const START_POINTS: i64 = 100;
pub const TOP_UP_BELOW: i64 = 5;
pub const TOP_UP_TO: i64 = 25;
pub const STAKES: [i64; 3] = [5, 10, 25];
/// The odds pay out (1 - this) of a fair price: GAME RULE, so a bet is not free money.
pub const HOUSE_TAKE: f64 = fly_sim::race::HOUSE_TAKE;
pub const MIN_ODDS: f64 = fly_sim::race::MIN_ODDS;

/// Payout multiple for a fly with probability `p` of winning: a fair 1/p less the house take, to one decimal, at least 1.1
/// (`core/points.py:decimal_odds`, in f64 with Python's rounding).
pub fn decimal_odds(p: f64) -> f64 {
    fly_sim::race::decimal_odds(p)
}

#[derive(Debug, Clone, PartialEq)]
pub struct BetRecord {
    pub label: String,
    pub stake: i64,
    pub odds: f64,
    pub won: bool,
    pub delta: i64,
}

#[derive(Debug, Clone)]
pub struct Wallet {
    pub path: Option<PathBuf>,
    pub points: i64,
    pub history: Vec<BetRecord>,
}

impl Wallet {
    /// A wallet kept in memory only.
    pub fn in_memory() -> Self {
        Self { path: None, points: START_POINTS, history: Vec::new() }
    }

    /// A wallet stored at `path`. A missing or damaged file starts over; a hand-edited one is clamped; below the top-up line it is topped up.
    pub fn open(path: PathBuf) -> Self {
        let points = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v.get("points").and_then(serde_json::Value::as_f64))
            .filter(|p| p.is_finite())
            .map_or(START_POINTS, |p| p as i64);
        let mut w = Self { path: Some(path), points, history: Vec::new() };
        if w.points < TOP_UP_BELOW {
            w.points = TOP_UP_TO;
        }
        w
    }

    pub fn can_bet(&self, stake: i64) -> bool {
        stake > 0 && stake <= self.points
    }

    /// Take the stake; if won, pay `stake * odds`. Returns the change in points. A bet the wallet cannot cover or odds under
    /// `MIN_ODDS` are refused.
    pub fn settle(&mut self, stake: i64, odds: f64, won: bool, label: &str) -> Result<i64, String> {
        if stake <= 0 {
            return Err("a stake is a whole number of points above 0".into());
        }
        if !self.can_bet(stake) {
            return Err("not enough points for that stake".into());
        }
        if !odds.is_finite() || odds < MIN_ODDS {
            return Err(format!("odds must be a finite payout of at least {MIN_ODDS}"));
        }
        let delta = if won { (stake as f64 * odds).round_ties_even() as i64 - stake } else { -stake };
        self.points += delta;
        self.history.push(BetRecord { label: label.to_string(), stake, odds, won, delta });
        if self.points < TOP_UP_BELOW {
            self.points = TOP_UP_TO;
            self.history.push(BetRecord { label: "free top-up".into(), stake: 0, odds: 0.0, won: false, delta: 0 });
        }
        self.history.drain(..self.history.len().saturating_sub(50));
        self.save();
        Ok(delta)
    }

    fn save(&self) {
        let Some(p) = &self.path else { return };
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let history: Vec<serde_json::Value> = self
            .history
            .iter()
            .rev()
            .take(50)
            .rev()
            .map(|h| serde_json::json!({"label": h.label, "stake": h.stake, "odds": h.odds, "won": h.won, "delta": h.delta}))
            .collect();
        let body = serde_json::json!({ "points": self.points, "history": history });
        let tmp = p.with_extension("tmp");
        if std::fs::write(&tmp, body.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, p); // a read-only folder: the points just do not persist
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odds_are_a_fair_price_less_the_house_take() {
        assert_eq!(decimal_odds(0.5), 1.8);
        assert_eq!(decimal_odds(0.25), 3.6);
        assert_eq!(decimal_odds(0.99), 1.1, "never below 1.1");
        assert!(decimal_odds(0.0) > 1000.0);
    }

    #[test]
    fn bets_win_lose_refuse_and_top_up() {
        let mut w = Wallet::in_memory();
        assert_eq!(w.settle(10, 2.0, true, "a").unwrap(), 10);
        assert_eq!(w.points, 110);
        assert_eq!(w.settle(10, 2.0, false, "b").unwrap(), -10);
        assert!(w.settle(0, 2.0, true, "c").is_err());
        assert!(w.settle(500, 2.0, true, "d").is_err());
        assert!(w.settle(5, f64::NAN, true, "e").is_err());
        assert!(w.settle(5, 1.0, true, "f").is_err());
        w.points = 25;
        w.settle(25, 2.0, false, "g").unwrap();
        assert_eq!(w.points, TOP_UP_TO, "never locked out");
    }

    #[test]
    fn the_wallet_file_survives_and_garbage_starts_over() {
        let dir = std::env::temp_dir().join(format!("ktf-wallet-{}", std::process::id()));
        let path = dir.join("arcade_points.json");
        let mut w = Wallet::open(path.clone());
        assert_eq!(w.points, START_POINTS);
        w.settle(10, 3.0, true, "x").unwrap();
        assert_eq!(Wallet::open(path.clone()).points, 120);
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(Wallet::open(path.clone()).points, START_POINTS);
        std::fs::write(&path, "{\"points\": -500}").unwrap();
        assert_eq!(Wallet::open(path).points, TOP_UP_TO);
        let _ = std::fs::remove_dir_all(dir);
    }
}
