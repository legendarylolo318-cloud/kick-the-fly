//! The data behind the Fly arcade page (`ktf_ui::pages::arcade`, Python `ui/arcade_ui.py`): background jobs for the tournament, the field
//! look and the race, run on the verified `fly_sim` engines (`duel`, `cards`, `race`, `tournament`), and the JSON the page draws.
//!
//! The page asks `query arcade` (see `GameHost`) and sends `arcade:<command>:<args>` actions; this module owns the jobs. Everything heavy
//! runs on a thread of its own with brains built from the loaded connectome (`SimHandle::shared`); Cancel is checked between steps.
//! Points are in-game points only (`points::Wallet`): no money, nothing to buy.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Instant;

use bevy::prelude::*;
use fly_sim::cards::{self, Build, Card};
use fly_sim::duel::{self, DuelResult};
use fly_sim::race::{self, CardNumbers};
use fly_sim::tournament::{self, BracketOut, MatchPlayer};
use serde_json::{Value, json};

use crate::sim_thread::Shared;

type Progress = Arc<Mutex<(f64, String)>>;

struct HubJob {
    label: &'static str,
    progress: Progress,
    cancel: Arc<AtomicBool>,
    rx: Mutex<Receiver<Result<Done, String>>>,
    started: Instant,
}

enum Done {
    Measured(Vec<Card>),
    Bracket(Value),
    Field(Vec<Card>),
    Race { race: Value, bet: Option<(u64, i64)> },
}

#[derive(Resource)]
pub struct ArcadeHub {
    bracket_serial: i64,
    race_serial: i64,
    bracket: Value,
    field: Value,
    odds: Value,
    race: Value,
    bet_result: Value,
    error: String,
    job: Option<HubJob>,
    pub wallet: crate::points::Wallet,
    /// Odds of the field last looked at, by seed (to settle a bet on the race that follows).
    field_odds: Vec<(u64, f64)>,
    /// The page asked for the lesion battle: the game opens its own page.
    pub want_lesion: bool,
    /// A line for the menu's flash message.
    pub flash: Option<String>,
    /// The seeds of the flies in play (set every frame by `flies_play`), the personality-card cache keys already measured for them, and
    /// the individuality mode they were asked under.
    pub play: Vec<u64>,
    measured: std::collections::HashSet<String>,
    measured_for: (Vec<u64>, String),
}

impl Default for ArcadeHub {
    fn default() -> Self {
        Self {
            bracket_serial: 0,
            race_serial: 0,
            bracket: Value::Null,
            field: Value::Null,
            odds: Value::Null,
            race: Value::Null,
            bet_result: Value::Null,
            error: String::new(),
            job: None,
            wallet: crate::points::Wallet::open(crate::paths::data_dir().join("arcade_points.json")),
            field_odds: Vec::new(),
            want_lesion: false,
            flash: None,
            play: Vec::new(),
            measured: Default::default(),
            measured_for: Default::default(),
        }
    }
}

fn cache_path() -> std::path::PathBuf {
    crate::paths::data_dir().join("cache").join("personality_cards.json")
}

fn pack_sha() -> String {
    static SHA: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SHA.get_or_init(|| fly_sim::pack::find_default().and_then(|p| cards::pack_sha256(&p)).unwrap_or_default()).clone()
}

fn card_numbers(c: &Card) -> CardNumbers {
    CardNumbers {
        sugar_ratio: c.sugar_ratio,
        walk_level_calm: c.walk_level_calm,
        loom_latency_s: c.loom_latency_s,
        turning_log_ratio: c.turning_log_ratio,
    }
}

fn odds_json(seeds: &[u64], cards: &[Card]) -> Value {
    let nums: Vec<CardNumbers> = cards.iter().map(card_numbers).collect();
    Value::Array(
        race::odds_table(seeds, &nums)
            .iter()
            .zip(cards)
            .map(|(o, c)| json!({"fly": o.fly, "title": c.title, "form": o.form, "p_win": o.p_win, "decimal_odds": o.decimal_odds}))
            .collect(),
    )
}

fn bracket_json(b: &BracketOut, cards: &[(u64, Card)], mode: &str) -> Value {
    let name = |s: u64| s.to_string();
    let rounds: Vec<Value> = b
        .rounds
        .iter()
        .map(|r| {
            Value::Array(
                r.iter()
                    .map(|m| {
                        let pair = [m.a, m.b];
                        let by_side = |f: &dyn Fn(usize) -> Value| {
                            let mut o = serde_json::Map::new();
                            for (i, s) in pair.iter().enumerate() {
                                o.insert(name(*s), f(i));
                            }
                            Value::Object(o)
                        };
                        json!({
                            "a": m.a, "b": m.b, "winner": m.winner, "coin_toss": m.coin_toss, "attempts": m.attempts,
                            "decided_by": m.decided_by,
                            "hp": by_side(&|i| json!(m.hp[i])), "shots": by_side(&|i| json!(m.shots[i])),
                            "hits": by_side(&|i| json!(m.hits[i])), "seconds": m.seconds, "knockout": m.knockout,
                            "max_levels": by_side(&|i| {
                                let l = m.max_levels[i];
                                json!({"fire": l[0], "escape": l[1], "run": l[2], "walk": l[3]})
                            }),
                            "events": m.events.iter().map(|e| json!([e.kind, e.t, name(pair[e.who])])).collect::<Vec<_>>(),
                            "frames": m.frames.iter().map(|f| {
                                json!([f.t, f.pos[0][0], f.pos[0][1], f.yaw[0], f.hp[0], f.pos[1][0], f.pos[1][1], f.yaw[1], f.hp[1],
                                       f.pellets.iter().map(|p| json!([p.0, p.1, p.2])).collect::<Vec<_>>()])
                            }).collect::<Vec<_>>(),
                            "match_seed": m.match_seed,
                        })
                    })
                    .collect(),
            )
        })
        .collect();
    let rows = |v: &[duel::DriverRow]| -> Value {
        Value::Array(
            v.iter()
                .map(|r| json!({"type": r.kind, "neurons": r.neurons, "hz_before_hits": r.hz_before_hits, "hz_overall": r.hz_overall, "ratio": r.ratio}))
                .collect(),
        )
    };
    let cards_json: serde_json::Map<String, Value> = cards.iter().map(|(s, c)| (name(*s), c.to_json())).collect();
    let won: serde_json::Map<String, Value> = b.rounds_won.iter().map(|(s, n)| (name(*s), json!(n))).collect();
    json!({
        "kind": "tournament", "size": b.seeds.len(), "seeds": b.seeds, "individuality": mode, "match_seconds": b.match_seconds,
        "bracket_seed": b.bracket_seed, "favorite": b.favorite, "favorite_won": b.favorite_won,
        "favorite_rounds_won": b.favorite_rounds_won, "cards": cards_json, "rounds": rounds, "champion": b.champion,
        "rounds_won": won, "champion_drivers": rows(&b.champion_drivers), "champion_readouts": rows(&b.champion_readouts),
    })
}

struct RealPlayer<'a> {
    build: &'a Build<'a>,
    bracket_seed: u64,
    cancel: &'a AtomicBool,
    progress: &'a Progress,
    played: usize,
    total: usize,
}

impl MatchPlayer for RealPlayer<'_> {
    fn play(&mut self, a: u64, b: u64, attempt: u64, seconds: f64, drivers: bool) -> Result<(u64, DuelResult), String> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let (mseed, na, nb) = tournament::match_noise(self.bracket_seed, a, b, attempt);
        set(
            self.progress,
            self.played as f64 / self.total as f64,
            &format!("match {a} v {b}{}", if attempt > 0 { " (rematch)" } else { "" }),
        );
        let (mut ba, mut bb) = (self.build.brain(a, Some(na), 600), self.build.brain(b, Some(nb), 600));
        let cancel = self.cancel;
        let r = duel::run_duel(&mut ba, &mut bb, mseed, seconds, true, drivers, |_| !cancel.load(Ordering::Relaxed));
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        if attempt == 0 || r.winner.is_some() {
            self.played += 1;
        }
        Ok((mseed, r))
    }
}

fn set(p: &Progress, frac: f64, note: &str) {
    *p.lock().unwrap_or_else(PoisonError::into_inner) = (frac.clamp(0.0, 1.0), note.to_string());
}

fn measure(build: &Build, seeds: &[u64], sha: &str, cancel: &AtomicBool, progress: &Progress) -> Result<Vec<Card>, String> {
    let mut out = Vec::new();
    for (i, &s) in seeds.iter().enumerate() {
        set(progress, i as f64 / seeds.len() as f64, &format!("measuring the personality card of fly {s} ({}/{})", i + 1, seeds.len()));
        let card = cards::measure_card(build, s, sha, 600, || cancel.load(Ordering::Relaxed))?;
        cards::store(&cache_path(), std::slice::from_ref(&card)); // each card is kept as it comes back, as Python's measure_cards does
        out.push(card);
    }
    Ok(out)
}

impl ArcadeHub {
    /// A hub with this wallet (tests keep the points in memory).
    pub fn with_wallet(wallet: crate::points::Wallet) -> Self {
        Self { wallet, ..Self::default() }
    }

    /// How many flies in play have no measured card yet under the current individuality mode.
    fn unmeasured(&self) -> usize {
        let mode = &self.measured_for.1;
        self.play.iter().filter(|&&s| !self.measured.contains(&cards::card_key(s, mode, None, &cards::LIF_DEFAULTS, &pack_sha()))).count()
    }

    /// Called by `flies_play` each frame with the flies' seeds and the individuality mode; re-reads the cache when they change.
    pub fn set_play(&mut self, seeds: Vec<u64>, mode: &str) {
        if self.measured_for.0 != seeds || self.measured_for.1 != mode {
            self.measured = cards::read_cache(&cache_path()).into_keys().collect();
            self.measured_for = (seeds.clone(), mode.to_string());
        }
        self.play = seeds;
    }

    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    /// The JSON the page draws (`ktf_ui::pages::arcade` module doc).
    pub fn snapshot(&self, palette: &str) -> Value {
        let job = self.job.as_ref().map(|j| {
            let (frac, note) = j.progress.lock().unwrap_or_else(PoisonError::into_inner).clone();
            json!({"label": j.label, "note": note, "frac": frac, "elapsed": j.started.elapsed().as_secs_f64()})
        });
        json!({
            "points": self.wallet.points, "slots": self.play.len(), "unmeasured": self.unmeasured(), "job": job, "error": self.error, "palette": palette,
            "bracket_serial": self.bracket_serial, "bracket": self.bracket, "race_serial": self.race_serial, "field": self.field,
            "odds": self.odds, "race": self.race, "bet_result": self.bet_result,
        })
    }

    /// An `arcade:<command>:<args>` action from the page.
    pub fn command(&mut self, action: &str, shared: Option<Shared>, mode: &str) {
        let parts: Vec<&str> = action.split(':').collect();
        match parts.get(1).copied() {
            Some("cancel") => {
                if let Some(j) = &self.job {
                    j.cancel.store(true, Ordering::Relaxed);
                }
            }
            Some("battle") => self.want_lesion = true,
            Some("measure_play") if self.job.is_none() => {
                let Some(shared) = shared else { return self.fail("the brain is not loaded yet") };
                self.spawn_measure_play(shared, mode);
            }
            Some("tournament") if parts.len() >= 6 && self.job.is_none() => {
                let size: usize = parts[2].parse().unwrap_or(8);
                let base: u64 = parts[3].parse().unwrap_or(2000);
                let secs: f64 = parts[4].parse().unwrap_or(20.0);
                let fav: i64 = parts[5].parse().unwrap_or(-1);
                let Some(shared) = shared else { return self.fail("the brain is not loaded yet") };
                self.spawn_tournament(shared, size, base, secs, (fav >= 0).then_some(fav as u64), mode);
            }
            Some("look") if parts.len() >= 4 && self.job.is_none() => {
                let lanes: usize = parts[2].parse().unwrap_or(5);
                let base: u64 = parts[3].parse().unwrap_or(3000);
                let Some(shared) = shared else { return self.fail("the brain is not loaded yet") };
                self.spawn_field(shared, lanes, base, mode);
            }
            Some("race") if parts.len() >= 6 && self.job.is_none() => {
                let lanes: usize = parts[2].parse().unwrap_or(5);
                let base: u64 = parts[3].parse().unwrap_or(3000);
                let bet: i64 = parts[4].parse().unwrap_or(-1);
                let stake: i64 = parts[5].parse().unwrap_or(10);
                let Some(shared) = shared else { return self.fail("the brain is not loaded yet") };
                if self.field.is_null() {
                    return self.fail("look at the field first");
                }
                self.spawn_race(shared, lanes, base, (bet >= 0).then_some((bet as u64, stake)), mode);
            }
            _ => {}
        }
    }

    fn fail(&mut self, why: &str) {
        self.error = why.to_string();
    }

    fn start(&mut self, label: &'static str, f: impl FnOnce(&Progress, &AtomicBool) -> Result<Done, String> + Send + 'static) {
        let progress: Progress = Arc::new(Mutex::new((0.0, "starting".into())));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let (p, c) = (progress.clone(), cancel.clone());
        thread::Builder::new()
            .name(format!("arcade-{label}"))
            .spawn(move || {
                let _ = tx.send(f(&p, &c));
            })
            .ok();
        self.error.clear();
        self.job = Some(HubJob { label, progress, cancel, rx: Mutex::new(rx), started: Instant::now() });
    }

    fn spawn_tournament(&mut self, shared: Shared, size: usize, base: u64, secs: f64, favorite: Option<u64>, mode: &str) {
        let mode = mode.to_string();
        self.bracket = Value::Null;
        self.start("Tournament", move |progress, cancel| {
            let seeds: Vec<u64> = (0..size as u64).map(|i| base + i).collect();
            let sha = pack_sha();
            let build = Build { weights: &shared.weights, labels: &shared.labels, mode: &mode, sigma: None, lif: cards::LIF_DEFAULTS };
            let measured = measure(&build, &seeds, &sha, cancel, progress)?;
            let mut player = RealPlayer { build: &build, bracket_seed: 0, cancel, progress, played: 0, total: size - 1 };
            let out = tournament::run_bracket(&seeds, secs, favorite, 0, true, &mut player)?;
            let cards: Vec<(u64, Card)> = seeds.iter().copied().zip(measured).collect();
            Ok(Done::Bracket(bracket_json(&out, &cards, &mode)))
        });
    }

    fn spawn_measure_play(&mut self, shared: Shared, mode: &str) {
        let mode = mode.to_string();
        let seeds = self.play.clone();
        self.start("Measuring the flies in play", move |progress, cancel| {
            let build = Build { weights: &shared.weights, labels: &shared.labels, mode: &mode, sigma: None, lif: cards::LIF_DEFAULTS };
            Ok(Done::Measured(measure(&build, &seeds, &pack_sha(), cancel, progress)?))
        });
    }

    fn spawn_field(&mut self, shared: Shared, lanes: usize, base: u64, mode: &str) {
        let mode = mode.to_string();
        self.field = Value::Null;
        self.odds = Value::Null;
        self.race = Value::Null;
        self.bet_result = Value::Null;
        self.start("Looking at the field", move |progress, cancel| {
            let seeds: Vec<u64> = (0..lanes as u64).map(|i| base + i).collect();
            let sha = pack_sha();
            let build = Build { weights: &shared.weights, labels: &shared.labels, mode: &mode, sigma: None, lif: cards::LIF_DEFAULTS };
            Ok(Done::Field(measure(&build, &seeds, &sha, cancel, progress)?))
        });
    }

    fn spawn_race(&mut self, shared: Shared, lanes: usize, base: u64, bet: Option<(u64, i64)>, mode: &str) {
        let mode = mode.to_string();
        let field: Vec<Card> = self.field.as_array().map(|a| a.iter().filter_map(Card::from_json).collect()).unwrap_or_default();
        self.race = Value::Null;
        self.bet_result = Value::Null;
        self.start("Race", move |progress, cancel| {
            let seeds: Vec<u64> = field.iter().map(|c| c.seed).collect();
            if seeds.len() != lanes || seeds.is_empty() {
                return Err("the field changed: look at it again".into());
            }
            let build = Build { weights: &shared.weights, labels: &shared.labels, mode: &mode, sigma: None, lif: cards::LIF_DEFAULTS };
            let nums: Vec<CardNumbers> = field.iter().map(card_numbers).collect();
            let mut done = 0usize;
            let mut lane = |seed: u64, repeat: usize| -> Result<race::LaneResult, String> {
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                set(progress, done as f64 / seeds.len() as f64, &format!("fly {seed} run {}", repeat + 1));
                let mut b = build.brain(seed, Some(race::lane_noise(base, seed, repeat as u64)), 600);
                let r = race::run_lane(&mut b, race::TIME_CAP_S, race::TRACK_LENGTH, |_| !cancel.load(Ordering::Relaxed));
                done += 1;
                if cancel.load(Ordering::Relaxed) { Err("cancelled".into()) } else { Ok(r) }
            };
            let out = race::run_race(&seeds, 1, base, &nums, &mut lane)?;
            let runs: serde_json::Map<String, Value> = seeds
                .iter()
                .zip(&out.runs[0])
                .map(|(s, l)| {
                    (
                        s.to_string(),
                        json!({"finish_s": l.finish_s, "distance": l.distance, "trace": l.trace, "lures_touched": l.lures_touched,
                               "max_walk_level": l.max_walk_level, "seconds": l.seconds, "rule_version": race::RULE_VERSION}),
                    )
                })
                .collect();
            let cards_json: serde_json::Map<String, Value> = field.iter().map(|c| (c.seed.to_string(), c.to_json())).collect();
            let odds: serde_json::Map<String, Value> = out
                .odds
                .iter()
                .map(|o| (o.fly.to_string(), json!({"p_win": o.p_win, "decimal_odds": o.decimal_odds, "form": o.form})))
                .collect();
            let race = json!({
                "kind": "race", "seeds": seeds, "individuality": mode, "race_seed": base, "repeats": 1, "track_m": race::TRACK_LENGTH,
                "lures": race::LURES.iter().map(|(x, k)| json!([x, k])).collect::<Vec<_>>(), "cards": cards_json,
                "runs": {"0": runs}, "orders": out.orders, "winner": out.winner, "odds": odds,
                "speed_rule_version": race::RULE_VERSION,
            });
            Ok(Done::Race { race, bet })
        });
    }

    /// Collect a finished job (once per frame).
    pub fn poll(&mut self) {
        let Some(job) = &self.job else { return };
        let got = match job.rx.lock().unwrap_or_else(PoisonError::into_inner).try_recv() {
            Ok(r) => r,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err("the job stopped unexpectedly".into()),
        };
        let label = job.label;
        self.job = None;
        match got {
            Err(e) if e == "cancelled" => {}
            Err(e) => self.error = e,
            Ok(Done::Bracket(b)) => {
                self.bracket_serial += 1;
                if let Some(fav) = b["favorite"].as_u64() {
                    let champ = b["champion"].as_u64().unwrap_or(0);
                    self.flash = Some(if fav == champ {
                        "Your favorite won the tournament!".to_string()
                    } else {
                        format!("Your favorite lost. Fly {champ} won.")
                    });
                }
                self.bracket = b;
            }
            Ok(Done::Measured(cards)) => {
                self.measured.extend(cards.iter().map(|c| c.card_key.clone()));
                self.flash = Some(format!("Measured {} personality card(s): the flies in play show them now.", cards.len()));
            }
            Ok(Done::Field(cards)) => {
                let seeds: Vec<u64> = cards.iter().map(|c| c.seed).collect();
                let odds = odds_json(&seeds, &cards);
                self.field_odds = odds
                    .as_array()
                    .map(|a| a.iter().filter_map(|o| Some((o["fly"].as_u64()?, o["decimal_odds"].as_f64()?))).collect())
                    .unwrap_or_default();
                self.field = Value::Array(cards.iter().map(Card::to_json).collect());
                self.odds = odds;
            }
            Ok(Done::Race { race, bet }) => {
                self.race_serial += 1;
                if let Some((fly, stake)) = bet {
                    let odds = self.field_odds.iter().find(|o| o.0 == fly).map_or(race::MIN_ODDS, |o| o.1);
                    let won = race["winner"].as_u64() == Some(fly);
                    match self.wallet.settle(stake, odds, won, &format!("bet on fly {fly}")) {
                        Ok(delta) => {
                            self.bet_result = json!([{"fly": fly, "stake": stake, "odds": odds, "won": won, "delta": delta, "points": self.wallet.points}]);
                        }
                        Err(e) => self.error = e,
                    }
                }
                self.race = race;
            }
        }
        let _ = label;
    }
}

fn poll(mut hub: ResMut<ArcadeHub>) {
    hub.poll();
}

pub struct ArcadeHubPlugin;

impl Plugin for ArcadeHubPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ArcadeHub>().add_systems(Update, poll);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hub_with_no_job_answers_the_page_contract() {
        let mut hub = ArcadeHub::with_wallet(crate::points::Wallet::in_memory());
        let q = hub.snapshot("default");
        for k in [
            "points",
            "slots",
            "unmeasured",
            "job",
            "error",
            "palette",
            "bracket_serial",
            "bracket",
            "race_serial",
            "field",
            "odds",
            "race",
            "bet_result",
        ] {
            assert!(q.get(k).is_some(), "missing {k}");
        }
        assert!(q["job"].is_null() && q["bracket"].is_null());
        hub.command("arcade:look:5:3000", None, "subtle");
        assert!(q["error"].as_str().unwrap().is_empty());
        assert!(hub.snapshot("default")["error"].as_str().unwrap().contains("not loaded"));
    }
}
