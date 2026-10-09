//! The fly arcade and the puzzle modes, behind `O`: Puppeteer, lesion battles, tournaments and races. Python: `ui/arcade_ui.py`,
//! `lab/puppet_challenge.py`, `game/lesion_battle.py` + `ui/battle_ui.py`, `lab/tournament.py`, `lab/racing.py`, `core/points.py`.
//!
//! The rules are in pure modules (`puppeteer`, `lesion`, `tournament`, `duel`, `race`, `points`); this is the glue: the keyboard-driven
//! pages, and background threads that run the extra brains. A tournament, a race and a lesion battle each need several brains stepping
//! together, so they run on a thread of their own with brains built from the already loaded connectome (`SimHandle::shared`): the game
//! keeps rendering while they work, and the page shows the progress.
//!
//! Tags: what the neurons do is CONNECTOME; the arena, budget, track and betting are GAME RULES; who wins is a MODEL PREDICTION about this
//! model and its noise, not about real flies. Betting is points only: they cannot be bought, sold or cashed out.
//!
//! NOT ported: personality cards (`core/cards.py`), the pre-registered tournament analysis, saved lesion loadouts and their share codes,
//! race odds from sampled runs (the odds here are flat, labelled as such), the bracket replay viewer.

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;

use bevy::prelude::*;
use fly_sim::lif::{Individuality, LifParams, LifSim};
use fly_sim::{Brain, brain::rows_of};

use crate::app::{LatestSnapshot, SimLink};
use crate::duel::{self, DuelResult, Side};
use crate::feedback::{Notes, Popup, Tag};
use crate::fly::{Fly, FlyPart, Part, ReactionFired, ResetFly};
use crate::lesion::{self, Loadout, SILENCE, STIMULATE};
use crate::paths::Scores;
use crate::player::MouseLook;
use crate::points::{self, Wallet};
use crate::puppeteer::{self, Goal, Level, PALETTE, Tracker};
use crate::race::{self, LaneResult};
use crate::sim_thread::{Command, Shared};
use crate::tools::ToolState;
use crate::tournament::{self, Bracket, MatchRunner};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Page {
    Main,
    Puppeteer,
    Lesion,
    Tournament,
    Race,
}

struct PuppetRun {
    level: Level,
    tracker: Tracker,
    actions: u32,
    started: f32,
    latched: Option<(usize, bool)>,
    loom_until: f32,
    prev_laser: bool,
    won_at: Option<f32>,
    stars: u32,
    shown_hint: bool,
}

enum JobDone {
    Text(String),
    Bracket(Bracket, Vec<(u32, u32)>),
    Race { lanes: Vec<LaneResult>, order: Vec<usize>, pick: usize, stake: i64 },
}

struct Job {
    rx: Mutex<Receiver<JobDone>>,
    progress: Arc<Mutex<String>>,
    label: String,
}

#[derive(Resource)]
pub struct Arcade {
    pub open: bool,
    page: Page,
    puppet: Option<PuppetRun>,
    job: Option<Job>,
    picks: [Loadout; 2],
    player: usize,
    rounds: u32,
    race_lanes: usize,
    race_pick: usize,
    stake_i: usize,
    wallet: Wallet,
    scores: Scores,
    result: Vec<String>,
}

impl Default for Arcade {
    fn default() -> Self {
        Self {
            open: false,
            page: Page::Main,
            puppet: None,
            job: None,
            picks: [Loadout::new("player 1"), Loadout::new("player 2")],
            player: 0,
            rounds: 1,
            race_lanes: 4,
            race_pick: 0,
            stake_i: 0,
            wallet: Wallet::open(crate::paths::data_dir().join("arcade_points.json")),
            scores: Scores::default_file(),
            result: Vec::new(),
        }
    }
}

#[derive(Component)]
struct ArcadeText;
#[derive(Component)]
struct PuppetText;

fn setup(mut commands: Commands) {
    commands.spawn((
        ArcadeText,
        Text::new(""),
        TextFont { font_size: FontSize::Px(16.0), ..default() },
        TextColor(Color::srgb(0.92, 0.95, 1.0)),
        Node { position_type: PositionType::Absolute, left: percent(50), top: px(70), margin: UiRect { left: px(-340), ..default() }, padding: UiRect::all(px(16)), width: px(680), display: Display::None, ..default() },
        BackgroundColor(Color::srgba(0.03, 0.04, 0.08, 0.94)),
        ZIndex(10),
    ));
    commands.spawn((
        PuppetText,
        Text::new(""),
        TextFont { font_size: FontSize::Px(16.0), ..default() },
        TextColor(Color::srgb(1.0, 0.92, 0.6)),
        Node { position_type: PositionType::Absolute, left: percent(50), top: px(34), margin: UiRect { left: px(-260), ..default() }, padding: UiRect::all(px(8)), width: px(520), display: Display::None, ..default() },
        BackgroundColor(Color::srgba(0.03, 0.03, 0.06, 0.8)),
        ZIndex(10),
    ));
}

// --- background brains -------------------------------------------------------------------------------------------------------------
/// A warmed-up brain of individuality `seed` with its own noise stream (`tournament.build_fighter`).
pub fn build_brain(shared: &Shared, seed: u64, noise_seed: u64, sigma: f64) -> Brain {
    let mut sim = LifSim::new(shared.weights.clone(), LifParams::default(), noise_seed);
    sim.set_individuality(Individuality::compute(seed, shared.weights.n(), sigma));
    let mut b = Brain::new(sim, shared.labels.clone());
    b.warmup(600);
    b
}

pub fn apply_loadout(brain: &mut Brain, l: &Loadout) {
    for (spec, mode) in l.surgery() {
        if let Ok(rows) = rows_of(brain.labels(), brain.n(), &spec) {
            brain.set_surgery(&rows, mode);
        }
    }
}

struct RealMatches<'a> {
    shared: &'a Shared,
    seconds: f32,
    sigma: f64,
    bracket_seed: u64,
    progress: &'a Arc<Mutex<String>>,
    played: usize,
    total: usize,
}

impl MatchRunner for RealMatches<'_> {
    fn run(&mut self, a: u32, b: u32, match_seed: u64, attempt: usize) -> DuelResult {
        let (_, na, nb) = tournament::match_noise(self.bracket_seed, a, b, attempt);
        *self.progress.lock().unwrap_or_else(PoisonError::into_inner) = format!("match {} of {}: fly {a} v fly {b}{}", self.played + 1, self.total, if attempt > 0 { " (rematch)" } else { "" });
        let (mut ba, mut bb) = (build_brain(self.shared, u64::from(a), na, self.sigma), build_brain(self.shared, u64::from(b), nb, self.sigma));
        let r = duel::run_duel(&mut ba, &mut bb, match_seed, self.seconds);
        if attempt == 0 || r.winner.is_some() {
            self.played += 1;
        }
        r
    }
}

fn spawn_tournament(shared: Shared, size: usize) -> Job {
    let (tx, rx) = mpsc::channel();
    let progress = Arc::new(Mutex::new(format!("measuring... {size} flies")));
    let p = progress.clone();
    thread::Builder::new()
        .name("tournament".into())
        .spawn(move || {
            let seeds: Vec<u32> = (1..=size as u32).collect();
            let mut runner = RealMatches { shared: &shared, seconds: duel::DEFAULT_SECONDS, sigma: 0.05, bracket_seed: 0, progress: &p, played: 0, total: size - 1 };
            let out = tournament::run_bracket(&seeds, 0, &mut runner);
            let _ = tx.send(match out {
                Ok(b) => {
                    let wins = b.wins.clone();
                    JobDone::Bracket(b, wins)
                }
                Err(e) => JobDone::Text(format!("could not run the bracket: {e}")),
            });
        })
        .ok();
    Job { rx: Mutex::new(rx), progress, label: format!("tournament of {size}") }
}

fn spawn_lesion(shared: Shared, picks: [Loadout; 2], rounds: u32) -> Job {
    let (tx, rx) = mpsc::channel();
    let progress = Arc::new(Mutex::new("round 1".to_string()));
    let p = progress.clone();
    thread::Builder::new()
        .name("lesion-battle".into())
        .spawn(move || {
            let (mut wins, mut lines) = ([0u32; 2], Vec::new());
            for round in 0..rounds {
                *p.lock().unwrap_or_else(PoisonError::into_inner) = format!("round {} of {rounds}", round + 1);
                let (mut a, mut b) = (build_brain(&shared, 1, 100 + u64::from(round), 0.05), build_brain(&shared, 2, 200 + u64::from(round), 0.05));
                apply_loadout(&mut a, &picks[0]);
                apply_loadout(&mut b, &picks[1]);
                let r = duel::run_duel(&mut a, &mut b, 7 + u64::from(round), duel::DEFAULT_SECONDS);
                match r.winner {
                    Some(Side::A) => wins[0] += 1,
                    Some(Side::B) => wins[1] += 1,
                    None => {}
                }
                lines.push(format!("round {}: {}  (health {:.0} v {:.0}, shots {} v {}, hits {} v {})", round + 1, match r.winner { Some(Side::A) => "player 1", Some(Side::B) => "player 2", None => "a draw" }, r.hp.0, r.hp.1, r.shots.0, r.shots.1, r.hits.0, r.hits.1));
            }
            let verdict = if wins[0] > wins[1] { "PLAYER 1 wins" } else if wins[1] > wins[0] { "PLAYER 2 wins" } else { "a DRAW" };
            lines.push(format!("{verdict}  ({} - {})   [MODEL PREDICTION: this model and its noise, not real flies]", wins[0], wins[1]));
            let _ = tx.send(JobDone::Text(lines.join("\n")));
        })
        .ok();
    Job { rx: Mutex::new(rx), progress, label: "lesion battle".into() }
}

fn spawn_race(shared: Shared, lanes: usize, pick: usize, stake: i64) -> Job {
    let (tx, rx) = mpsc::channel();
    let progress = Arc::new(Mutex::new("on your marks".to_string()));
    let p = progress.clone();
    thread::Builder::new()
        .name("race".into())
        .spawn(move || {
            let mut results = Vec::new();
            for lane in 0..lanes {
                *p.lock().unwrap_or_else(PoisonError::into_inner) = format!("lane {} of {lanes}", lane + 1);
                let mut b = build_brain(&shared, lane as u64 + 1, 300 + lane as u64, 0.05);
                results.push(race::run_lane(&mut b, race::TIME_CAP_S, race::TRACK_LENGTH));
            }
            let order = race::order(&results);
            let _ = tx.send(JobDone::Race { lanes: results, order, pick, stake });
        })
        .ok();
    Job { rx: Mutex::new(rx), progress, label: format!("race of {lanes}") }
}

// --- the pages ----------------------------------------------------------------------------------------------------------------------
const LETTERS: &str = "abcdefghijklmno";
const KEYS: [KeyCode; 15] = [KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC, KeyCode::KeyD, KeyCode::KeyE, KeyCode::KeyF, KeyCode::KeyG, KeyCode::KeyH, KeyCode::KeyI, KeyCode::KeyJ, KeyCode::KeyK, KeyCode::KeyL, KeyCode::KeyM, KeyCode::KeyN, KeyCode::KeyO];
const DIGITS: [KeyCode; 10] = [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4, KeyCode::Digit5, KeyCode::Digit6, KeyCode::Digit7, KeyCode::Digit8, KeyCode::Digit9, KeyCode::Digit0];

fn stars_text(n: f64) -> String {
    let n = n as usize;
    format!("{}{}", "*".repeat(n), "-".repeat(3usize.saturating_sub(n)))
}

fn page_text(a: &Arcade, sizes_ok: bool) -> String {
    if let Some(job) = &a.job {
        let progress = job.progress.lock().unwrap_or_else(PoisonError::into_inner).clone();
        return format!("FLY ARCADE: {}\n\nrunning on its own brains in the background...\n{progress}\n\n(Esc: leave it running and close this page)", job.label);
    }
    let mut s = String::new();
    match a.page {
        Page::Main => {
            s += "FLY ARCADE\n\n  1  Puppeteer: steer the fly with the laser alone to make it do ten things\n  2  Lesion battle: operate on your own fly within a budget, then they fight\n  3  Tournament: a bracket of 4, 8 or 16 flies, each its own individual\n  4  Race: flies race down a track through their own brains; bet points on one\n\n";
            s += &format!("points: {} (arcade points are a score only: they cannot be bought, sold or cashed out)\n\nO or Esc closes", a.wallet.points);
        }
        Page::Puppeteer => {
            s += "PUPPETEER: you cannot touch the fly; only the laser (hold the button on the fly) and latching neurons on or off.\n; next cell type   ' activate/silence   L latch the chosen type   H hint (costs an action)   R restart   Esc back\n\n";
            for (i, l) in puppeteer::levels().iter().enumerate() {
                let best = a.scores.map.get(&puppeteer::score_key(l.id)).copied().unwrap_or(0.0);
                s += &format!("  {}  {:<18} {}   {}\n", (i + 1) % 10, l.title, stars_text(best), l.brief);
            }
            s += "\npress the number of a level to start it";
            if !sizes_ok {
                s += "\n(waiting for the brain to load)";
            }
        }
        Page::Lesion => {
            s += &format!("LESION BATTLE ({} round{}; R toggles 1 or 3)   Tab: switch player   F: fight   Esc: back\n", a.rounds, if a.rounds == 1 { "" } else { "s" });
            s += &format!("budget: {} picks per fly, each a cell type of at most {} neurons. Press the letter to cycle a cell type: silenced, stimulated, off.\n\n", lesion::BUDGET, lesion::MAX_TYPE_NEURONS);
            for p in 0..2 {
                s += &format!("{} player {}: {}\n", if a.player == p { ">>" } else { "  " }, p + 1, a.picks[p].describe());
            }
            s += "\n";
            for (i, t) in lesion::SUGGESTED.iter().enumerate() {
                let state = |p: usize| a.picks[p].picks.iter().find(|x| x.0 == *t).map_or(' ', |x| if x.1 == SILENCE { '-' } else { '+' });
                s += &format!("  [{}] {:<9} P1 {}  P2 {}\n", LETTERS.as_bytes()[i] as char, t, state(0), state(1));
            }
            s += "\nTAGS: surgery effects CONNECTOME; budget and arena GAME RULE; the winner is a MODEL PREDICTION.";
            if !a.result.is_empty() {
                s += &format!("\n\n{}", a.result.join("\n"));
            }
        }
        Page::Tournament => {
            s += "TOURNAMENT: a single-elimination bracket of flies that differ by their individuality seed (subtle: 5% gain spread).\n  4  four flies    8  eight flies    6  sixteen flies    Esc back\n(each 20 s duel steps two whole brains together, so this takes a few minutes: it runs in the background)\n";
            if !a.result.is_empty() {
                s += &format!("\n{}", a.result.join("\n"));
            }
        }
        Page::Race => {
            s += &format!("RACE: {} flies run an 8 m track with three lures (sugar, fruit, sugar). Speed comes from the walking neurons (DNp09).\n  4 / 8  lanes    1-8  pick your fly    S  stake ({} points)    Enter  start    Esc back\n", a.race_lanes, points::STAKES[a.stake_i]);
            s += &format!("your pick: lane {}   points: {}\nTAGS: speed from DNp09 is CONNECTOME; the track, rules and betting are GAME RULES; the order is a MODEL PREDICTION. Odds are flat in this build ({:.1}x).\n", a.race_pick + 1, a.wallet.points, points::decimal_odds(1.0 / a.race_lanes as f32));
            if !a.result.is_empty() {
                s += &format!("\n{}", a.result.join("\n"));
            }
        }
    }
    s
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn input(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Virtual>>,
    mut a: ResMut<Arcade>,
    mut look: ResMut<MouseLook>,
    link: Res<SimLink>,
    snap: Res<LatestSnapshot>,
    mut tools: ResMut<ToolState>,
    mut reset: MessageWriter<ResetFly>,
    mut notes: ResMut<Notes>,
) {
    let now = time.elapsed_secs();
    if keys.just_pressed(KeyCode::KeyO) && a.puppet.is_none() {
        a.open = !a.open;
        look.captured = !a.open;
        a.page = Page::Main;
        return;
    }
    // O during a puppeteer level aborts it and returns to the menu
    if keys.just_pressed(KeyCode::KeyO)
        && let Some(run) = a.puppet.take()
    {
        if let Some((t, _)) = run.latched {
            link.0.send(Command::DriveSpec { spec: PALETTE[t].1.into(), amp: 0.0 });
        }
        tools.puppet = false;
        a.open = true;
        a.page = Page::Puppeteer;
        look.captured = false;
        return;
    }
    if !a.open {
        // in a level: L latches, H hints, R restarts
        if a.puppet.is_some() {
            let pal = tools.laser_target % PALETTE.len();
            let silence = tools.laser_silence;
            let sel_spec = PALETTE[pal].1;
            let mut restart = None;
            if let Some(run) = a.puppet.as_mut() {
                if keys.just_pressed(KeyCode::KeyL) {
                    if run.latched == Some((pal, silence)) {
                        link.0.send(Command::DriveSpec { spec: sel_spec.into(), amp: 0.0 });
                        run.latched = None;
                    } else {
                        if let Some((t, _)) = run.latched {
                            link.0.send(Command::DriveSpec { spec: PALETTE[t].1.into(), amp: 0.0 });
                        }
                        link.0.send(Command::DriveSpec { spec: sel_spec.into(), amp: if silence { -0.6 } else { 0.12 } });
                        run.latched = Some((pal, silence));
                    }
                    run.actions += 1;
                }
                if keys.just_pressed(KeyCode::KeyH) && !run.shown_hint {
                    run.shown_hint = true;
                    run.actions += 1;
                }
                if keys.just_pressed(KeyCode::KeyR) {
                    restart = Some(run.level.clone());
                }
            }
            if let Some(level) = restart {
                start_level(&mut a, &level, now, &link, &mut tools, &mut reset, &mut notes);
            }
        }
        return;
    }
    let shared_ready = snap.0.as_deref().is_some_and(|s| s.info.is_some());
    if keys.just_pressed(KeyCode::Escape) {
        if a.job.is_some() || a.page == Page::Main {
            a.open = false;
            look.captured = true;
        } else {
            a.page = Page::Main;
        }
        return;
    }
    if a.job.is_some() {
        return;
    }
    match a.page {
        Page::Main => {
            for (k, p) in [(KeyCode::Digit1, Page::Puppeteer), (KeyCode::Digit2, Page::Lesion), (KeyCode::Digit3, Page::Tournament), (KeyCode::Digit4, Page::Race)] {
                if keys.just_pressed(k) {
                    a.page = p;
                    a.result.clear();
                }
            }
        }
        Page::Puppeteer => {
            if shared_ready {
                for (i, k) in DIGITS.iter().enumerate() {
                    if keys.just_pressed(*k)
                        && let Some(level) = puppeteer::levels().get(i).cloned()
                    {
                        a.open = false;
                        look.captured = true;
                        start_level(&mut a, &level, now, &link, &mut tools, &mut reset, &mut notes);
                    }
                }
            }
        }
        Page::Lesion => {
            if keys.just_pressed(KeyCode::Tab) {
                a.player = 1 - a.player;
            }
            if keys.just_pressed(KeyCode::KeyR) {
                a.rounds = if a.rounds == 1 { 3 } else { 1 };
            }
            let sizes = snap.0.as_deref().and_then(|s| s.info.as_deref()).map(|i| lesion::type_sizes(&i.types));
            for (i, k) in KEYS.iter().enumerate() {
                if *k == KeyCode::KeyR || *k == KeyCode::KeyF {
                    continue;
                }
                if keys.just_pressed(*k) {
                    let t = lesion::SUGGESTED[i];
                    let p = a.player;
                    let cur = a.picks[p].picks.iter().find(|x| x.0 == t).map(|x| x.1);
                    let next = match cur {
                        None => Some(SILENCE),
                        Some(m) if m == SILENCE => Some(STIMULATE),
                        _ => None,
                    };
                    match next {
                        Some(m) => {
                            if let Err(e) = a.picks[p].add(t, m, sizes.as_ref()) {
                                a.result = vec![format!("not allowed: {e}")];
                            } else {
                                a.result.clear();
                            }
                        }
                        None => a.picks[p].remove(t),
                    }
                }
            }
            if keys.just_pressed(KeyCode::KeyF)
                && let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone()
            {
                let picks = a.picks.clone();
                let rounds = a.rounds;
                a.job = Some(spawn_lesion(shared, picks, rounds));
                a.result.clear();
            }
        }
        Page::Tournament => {
            for (k, n) in [(KeyCode::Digit4, 4usize), (KeyCode::Digit8, 8), (KeyCode::Digit6, 16)] {
                if keys.just_pressed(k)
                    && let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone()
                {
                    a.job = Some(spawn_tournament(shared, n));
                    a.result.clear();
                }
            }
        }
        Page::Race => {
            if keys.just_pressed(KeyCode::Digit4) {
                a.race_lanes = 4;
            }
            if keys.just_pressed(KeyCode::Digit8) {
                a.race_lanes = 8;
            }
            for (i, k) in DIGITS.iter().take(8).enumerate() {
                if keys.just_pressed(*k) && i < a.race_lanes && *k != KeyCode::Digit4 && *k != KeyCode::Digit8 {
                    a.race_pick = i;
                }
            }
            if keys.just_pressed(KeyCode::KeyS) {
                a.stake_i = (a.stake_i + 1) % points::STAKES.len();
            }
            if keys.just_pressed(KeyCode::Enter) {
                let stake = points::STAKES[a.stake_i];
                if !a.wallet.can_bet(stake) {
                    a.result = vec![format!("not enough points for a stake of {stake}")];
                } else if let Some(shared) = link.0.shared.lock().unwrap_or_else(PoisonError::into_inner).clone() {
                    let (lanes, pick) = (a.race_lanes, a.race_pick);
                    a.job = Some(spawn_race(shared, lanes, pick, stake));
                    a.result.clear();
                }
            }
        }
    }
}

fn start_level(a: &mut Arcade, level: &Level, now: f32, link: &SimLink, tools: &mut ToolState, reset: &mut MessageWriter<ResetFly>, notes: &mut Notes) {
    if let Some(old) = a.puppet.take()
        && let Some((t, _)) = old.latched
    {
        link.0.send(Command::DriveSpec { spec: PALETTE[t].1.into(), amp: 0.0 });
    }
    let yaw = if level.facing >= 0 { 0.0 } else { std::f32::consts::PI };
    reset.write(ResetFly { place: Some((level.start_x, yaw)) });
    tools.puppet = true;
    tools.laser_target = 0;
    let mut tracker = Tracker::new(&level.goal);
    tracker.start(now);
    a.puppet = Some(PuppetRun { level: level.clone(), tracker, actions: 0, started: now, latched: None, loom_until: 0.0, prev_laser: false, won_at: None, stars: 0, shown_hint: false });
    notes.push(now, format!("PUPPETEER {}: {}", level.title, level.brief), Tag::GameRule);
}

/// Feed the running level: reactions, the fly's position, the looming shadows, the laser's pulses, the win.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn puppeteer_tick(
    time: Res<Time<Virtual>>,
    mut a: ResMut<Arcade>,
    mut fired: MessageReader<ReactionFired>,
    flies: Query<&Fly>,
    parts: Query<(&FlyPart, &Transform)>,
    tools: Res<ToolState>,
    link: Res<SimLink>,
    mut popups: MessageWriter<Popup>,
    mut text: Single<(&mut Text, &mut Node), (With<PuppetText>, Without<ArcadeText>)>,
) {
    let now = time.elapsed_secs();
    let names: Vec<String> = fired.read().map(|r| r.name.clone()).collect();
    let a = &mut *a;
    let Some(run) = a.puppet.as_mut() else {
        text.1.display = Display::None;
        return;
    };
    if run.won_at.is_none() {
        for n in &names {
            run.tracker.event(n, now);
        }
        if let Some(f) = flies.iter().next()
            && let Ok((_, thx)) = parts.get(f.parts[Part::Thorax.index()])
        {
            let fwd = f.fwd();
            run.tracker.position(thx.translation.x, thx.translation.z, fwd.x, fwd.z, f.flying_now);
        }
        // a pulse of the laser that reached the fly is an action
        let laser_now = tools.laser_on.is_some();
        if laser_now && !run.prev_laser {
            run.actions += 1;
        }
        run.prev_laser = laser_now;
        if run.tracker.loom_due(now) {
            run.loom_until = now + puppeteer::LOOM_FOR_S;
            run.tracker.loom_fired(now);
        }
        if now < run.loom_until {
            link.0.send(Command::Poke { region: "loom".into(), side: None, strength: 1.0 });
        }
        run.tracker.tick(now);
        if run.tracker.done {
            run.won_at = Some(now);
            run.stars = puppeteer::stars_for(run.actions, run.level.par);
            let key = puppeteer::score_key(run.level.id);
            let stars = run.stars;
            a.scores.record(&key, f64::from(stars), true);
            if let Some(f) = flies.iter().next()
                && let Ok((_, t)) = parts.get(f.parts[Part::Head.index()])
            {
                popups.write(Popup::new(t.translation + Vec3::Y * 0.5, "LEVEL CLEAR!", (160, 255, 190)));
            }
        }
    }
    let Some(run) = a.puppet.as_ref() else { return };
    text.1.display = Display::Flex;
    let mut s = format!("PUPPETEER  {}: {}\n", run.level.title, run.level.brief);
    s += &format!("progress {}   actions {} (par {})   target: {}{}\n", run.tracker.describe(), run.actions, run.level.par, PALETTE[tools.laser_target % PALETTE.len()].0, if tools.laser_silence { " SILENCE" } else { " ACTIVATE" });
    if let Some(l) = run.latched {
        s += &format!("latched: {} {}\n", PALETTE[l.0].0, if l.1 { "OFF" } else { "ON" });
    }
    if !run.tracker.message.is_empty() {
        s += &format!("{}\n", run.tracker.message);
    }
    if run.shown_hint {
        s += &format!("hint: {}\n", run.level.hint);
    }
    if run.won_at.is_some() {
        s += &format!("CLEARED in {:.1}s: {} stars ({}). R again, O menu. Circuit: {}\n", now - run.started, run.stars, stars_text(f64::from(run.stars)), match run.level.goal {
            Goal::Event(e) => e.to_string(),
            Goal::Travel(m) => format!("travel {m} m"),
            Goal::SurviveLoom { .. } => "survive the shadows".to_string(),
            Goal::Sequence(_) => "a sequence".to_string(),
        });
    } else {
        s += "L latch   H hint   R restart   O menu   ; ' choose the cell type and effect";
    }
    text.0.0 = s;
}

/// Collect a finished background job.
fn collect(mut a: ResMut<Arcade>, mut notes: ResMut<Notes>, time: Res<Time<Real>>) {
    let a = &mut *a;
    let Some(job) = a.job.as_ref() else { return };
    let Ok(done) = job.rx.lock().unwrap_or_else(PoisonError::into_inner).try_recv() else { return };
    let now = time.elapsed_secs();
    a.job = None;
    match done {
        JobDone::Text(t) => a.result = t.lines().map(str::to_string).collect(),
        JobDone::Bracket(b, wins) => {
            let mut lines = vec![format!("CHAMPION: fly {}   [MODEL PREDICTION: this model and its noise]", b.champion)];
            for (r, round) in b.rounds.iter().enumerate() {
                lines.push(format!("round {}:", r + 1));
                for m in round {
                    lines.push(format!("   fly {} v fly {} -> fly {}{}", m.a, m.b, m.winner, if m.coin_toss { "  (coin toss after two drawn rematches)" } else { "" }));
                }
            }
            let mut w = wins;
            w.sort_by_key(|e| std::cmp::Reverse(e.1));
            lines.push(format!("rounds won: {}", w.iter().map(|(s, n)| format!("fly {s}: {n}")).collect::<Vec<_>>().join(", ")));
            a.result = lines;
        }
        JobDone::Race { lanes, order, pick, stake } => {
            let mut lines = vec!["FINISH ORDER  [MODEL PREDICTION]".to_string()];
            for (rank, &i) in order.iter().enumerate() {
                let l = &lanes[i];
                lines.push(format!("  {}. lane {}  {}  (reached {:.1} m, lures touched {})", rank + 1, i + 1, l.finish_s.map_or("did not finish".to_string(), |t| format!("{t:.1} s")), l.distance, l.lures_touched.len()));
            }
            let odds = points::decimal_odds(1.0 / lanes.len() as f32);
            let won = order.first() == Some(&pick);
            match a.wallet.settle(stake, odds, won, &format!("race lane {}", pick + 1)) {
                Ok(d) => lines.push(format!("your lane {}: {} {:+} points (now {})", pick + 1, if won { "WON" } else { "lost" }, d, a.wallet.points)),
                Err(e) => lines.push(format!("bet not settled: {e}")),
            }
            a.result = lines;
        }
    }
    notes.push(now, "ARCADE   finished", Tag::Model);
}

fn show(a: Res<Arcade>, snap: Res<LatestSnapshot>, mut text: Single<(&mut Text, &mut Node), (With<ArcadeText>, Without<PuppetText>)>) {
    text.1.display = if a.open { Display::Flex } else { Display::None };
    if a.open {
        text.0.0 = page_text(&a, snap.0.as_deref().is_some_and(|s| s.info.is_some()));
    }
}

pub struct ArcadePlugin;

impl Plugin for ArcadePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Arcade>().add_systems(Startup, setup).add_systems(Update, (input, puppeteer_tick, collect, show).chain());
    }
}
