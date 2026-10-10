//! Pause menu > Fly arcade (`ui/arcade_ui.py`): the tournament and racing tabs. The jobs run in the game; this page keeps the selections
//! and draws what the game reports.
//!
//! Host contract: query `arcade` answers
//! `{points, slots, unmeasured, job: {label, note, frac, elapsed} | null, error, palette, bracket_serial, bracket, race_serial, field, odds, race,
//! bet_result, battle}`; the page's buttons run `Menu::action("arcade:<command>:<args>")`: `tournament:size:base_seed:match_s:favorite|-1`,
//! `measure_play`, `look:lanes:race_base`, `race:lanes:race_base:bet|-1:stake`, `cancel`, `battle` (the lesion battle tab is not ported here:
//! the game's keyboard page still has it, see PORTING.md).

use serde_json::Value;

use crate::canvas::{Canvas, Rect};
use crate::i18n::{tr, tr_args};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const STAKES: [i64; 3] = [5, 10, 25];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Tournament,
    Race,
    Battle,
}

pub struct ArcadePage {
    tab: Tab,
    size: usize,
    base_seed: i64,
    match_s: f64,
    favorite: Option<i64>,
    sel_match: Option<(usize, usize)>,
    replay_t: f64,
    playing: bool,
    last: f64,
    lanes: i64,
    race_base: i64,
    bet_fly: Option<i64>,
    stake: i64,
    bracket_serial: i64,
    race_serial: i64,
}

impl Default for ArcadePage {
    fn default() -> Self {
        Self {
            tab: Tab::Tournament,
            size: 8,
            base_seed: 2000,
            match_s: 20.0,
            favorite: None,
            sel_match: None,
            replay_t: 0.0,
            playing: true,
            last: 0.0,
            lanes: 5,
            race_base: 3000,
            bet_fly: None,
            stake: 10,
            bracket_serial: -1,
            race_serial: -1,
        }
    }
}

fn series(palette: &str) -> [[u8; 3]; 2] {
    match palette {
        "blue-yellow" => [[30, 100, 200], [240, 190, 20]],
        "high-contrast" => [[255, 255, 255], [255, 200, 0]],
        _ => [[57, 135, 229], [217, 89, 38]],
    }
}

fn tag(m: &Menu, cv: &mut Canvas, x: i32, y: i32, name: &str) -> i32 {
    let col = match name {
        "CONNECTOME" | "GAME RULE" => tag_color(name),
        _ => [150, 120, 220],
    };
    let f = m.fonts().small;
    let r = Rect::new(x, y, f.size(name) + 12, f.linesize + 4);
    cv.fill_rect(r, col, 5);
    m.text(cv, name, (r.x + 6, r.y + 2), [10, 12, 16], &f, Anchor::TopLeft);
    r.right() + 6
}

fn f64_of(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn card_of(cards: &Value, seed: i64) -> Option<&Value> {
    cards.get(seed.to_string()).filter(|c| c.is_object())
}

impl ArcadePage {
    /// Put the page in a known state (tests and scripted captures): tab 0/1/2, the favorite, the selected match, the bet and a paused replay.
    pub fn preset(&mut self, tab: u8, favorite: Option<i64>, sel_match: Option<(usize, usize)>, bet_fly: Option<i64>) {
        self.tab = match tab {
            1 => Tab::Race,
            2 => Tab::Battle,
            _ => Tab::Tournament,
        };
        self.favorite = favorite;
        self.sel_match = sel_match;
        self.bet_fly = bet_fly;
        self.playing = false;
        self.replay_t = 0.0;
        // the serials the fixture uses, so drawing does not reset the selection
        self.bracket_serial = 1;
        self.race_serial = 1;
    }

    fn draw_duel(&self, m: &Menu, cv: &mut Canvas, rect: Rect, frames: &[Value], t: f64, names: (&str, &str), cols: &[[u8; 3]; 2]) {
        cv.fill_rect(rect, [12, 15, 20], 8);
        cv.stroke_rect(rect, BORDER, 8);
        if frames.is_empty() {
            return;
        }
        let i = (((t * 10.0) as i64).max(0) as usize).min(frames.len() - 1);
        let fr = frames[i].as_array().cloned().unwrap_or_default();
        let half = 6.0;
        let px = |x: f64, y: f64| -> (i32, i32) {
            (
                (f64::from(rect.center().0) + x / half * (f64::from(rect.w) / 2.0 - 14.0)) as i32,
                (f64::from(rect.center().1) - y / half * (f64::from(rect.h) / 2.0 - 14.0)) as i32,
            )
        };
        let small = m.fonts().small;
        for (k, idx) in [(0usize, 1usize), (1, 5)] {
            let (x, y, yaw, hp) = (f64_of(&fr[idx]), f64_of(&fr[idx + 1]), f64_of(&fr[idx + 2]), f64_of(&fr[idx + 3]));
            let c = px(x, y);
            let alive = hp > 0.0;
            cv.circle(c, 11, if alive { cols[k] } else { [70, 70, 70] });
            let tip = (c.0 + (yaw.cos() * 20.0) as i32, c.1 - (yaw.sin() * 20.0) as i32);
            cv.line(c, tip, 3, [240, 240, 240], 255);
            let bar = Rect::new(rect.x + 10 + k as i32 * (rect.w / 2), rect.y + 8, rect.w / 2 - 20, 8);
            cv.fill_rect(bar, [40, 44, 54], 4);
            cv.fill_rect(Rect::new(bar.x, bar.y, (f64::from(bar.w) * hp / 100.0) as i32, bar.h), cols[k], 4);
            let name = if k == 0 { names.0 } else { names.1 };
            m.text(cv, &format!("{name}  {hp:.0}"), (bar.x, bar.bottom() + 2), TEXT, &small, Anchor::TopLeft);
        }
        if let Some(pellets) = fr.get(9).and_then(Value::as_array) {
            for p in pellets {
                let owner = p[2].as_u64().unwrap_or(0).min(1) as usize;
                cv.circle(px(f64_of(&p[0]), f64_of(&p[1])), 3, cols[owner]);
            }
        }
    }

    fn advance(&mut self, now: f64, length: f64) {
        if self.playing {
            self.replay_t = (self.replay_t + (now - self.last)).rem_euclid(length.max(0.5));
        }
        self.last = now;
    }

    #[allow(clippy::too_many_lines)]
    fn tournament(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, mut y: i32, d: &Value, busy: bool, cols: &[[u8; 3]; 2]) -> i32 {
        let f = m.fonts();
        let mut x = body.x + 8;
        y = m.wrapped(
            cv,
            &tr("A bracket of flies, each its own individuality seed. Both sides of every duel are brains: nothing about who wins is scripted."),
            (x, y),
            body.w - 24,
            LABEL,
            &f.small,
            4,
        ) + 6;
        for n in [4usize, 8, 16] {
            let tip = tr("How many flies in the bracket.");
            m.button(
                cv,
                Rect::new(x, y, 70, 32),
                &n.to_string(),
                Act::Custom(format!("size:{n}")),
                &format!("arc_size:{n}"),
                Style::Normal,
                !busy,
                Some(&tip),
                self.size == n,
                None,
            );
            x += 78;
        }
        m.page_slider(
            cv,
            Rect::new(x + 10, y, 250, 32),
            self.match_s,
            (10.0, 40.0, 5.0),
            "{:.0f} s duels",
            "arc_len",
            !busy,
            Some(&tr("How long a duel lasts if nobody is knocked out. More health at the end wins.")),
        );
        m.button(
            cv,
            Rect::new(x + 280, y, 130, 32),
            &tr("New flies"),
            Act::Custom("new_flies".into()),
            "arc_new",
            Style::Normal,
            !busy,
            Some(&tr("Pick a different set of individuality seeds.")),
            false,
            None,
        );
        let rw = 190.max(f.bold.size(&tr("Run tournament")) + 32);
        if x + 420 + rw > body.right() - 8 {
            y += 40;
            x = body.x + 8 - 420;
        }
        m.button(
            cv,
            Rect::new(x + 420, y, rw, 32),
            &tr("Run tournament"),
            Act::Custom("run_tournament".into()),
            "arc_run",
            Style::Primary,
            !busy,
            Some(&tr("Measures each fly's personality card, then plays the bracket. Minutes: it runs in the background.")),
            false,
            None,
        );
        y += 44;
        x = body.x + 8;
        let mut xx = tag(m, cv, x, y, "CONNECTOME");
        xx = tag(m, cv, xx, y, "GAME RULE");
        xx = tag(m, cv, xx, y, "MODEL PREDICTION");
        y = (y + 30).max(
            m.wrapped(
                cv,
                &tr("brains steer and shoot (CONNECTOME); the arena and blaster (GAME RULE); who wins (MODEL PREDICTION)"),
                (xx + 6, y + 2),
                body.right() - xx - 16,
                LABEL,
                &f.small,
                4,
            ) + 6,
        );
        let seeds: Vec<i64> = (0..self.size as i64).map(|i| self.base_seed + i).collect();
        m.text(cv, &tr("Pick your favorite (click a fly):"), (x, y), INK, &f.bold, Anchor::TopLeft);
        y += 26;
        let bracket = &d["bracket"];
        let cards = &bracket["cards"];
        let labels: Vec<String> = seeds
            .iter()
            .map(|&s| {
                let c = card_of(cards, s);
                format!(
                    "{}{}",
                    if self.favorite == Some(s) { "FAV  " } else { "" },
                    match c {
                        Some(c) => format!("{s}  {}", c["title"].as_str().unwrap_or("")),
                        None => format!("{} {s}", tr("Fly")),
                    }
                )
            })
            .collect();
        let cw = 220.max(labels.iter().map(|t| f.bold.size(t)).max().unwrap_or(0) + 24).min(body.w - 16);
        let per_row = ((body.w - 16) / (cw + 8)).max(1);
        for (i, &s) in seeds.iter().enumerate() {
            let (cx, cy) = (x + (i as i32 % per_row) * (cw + 8), y + (i as i32 / per_row) * 40);
            let c = card_of(cards, s);
            let tip = match c {
                Some(c) => format!(
                    "{} · looming latency {:.2} s, sugar x{:.2}, steering R/L {:.2} (measured from this fly's brain)",
                    c["summary"].as_str().unwrap_or(""),
                    f64_of(&c["loom_latency_s"]),
                    f64_of(&c["sugar_ratio"]),
                    f64_of(&c["turning_ratio"])
                ),
                None => tr("Its personality card is measured when the tournament runs."),
            };
            m.button(
                cv,
                Rect::new(cx, cy, cw, 34),
                &labels[i],
                Act::Custom(format!("fav:{s}")),
                &format!("arc_fav:{s}"),
                Style::Normal,
                !busy,
                Some(&tip),
                self.favorite == Some(s),
                None,
            );
        }
        y += ((seeds.len() as i32 + per_row - 1) / per_row) * 40 + 8;
        let Some(rounds) = bracket["rounds"].as_array() else {
            m.text(cv, &tr("No tournament yet."), (x, y), LABEL, &f.text, Anchor::TopLeft);
            return y + 30;
        };
        m.text(cv, &tr("Bracket (click a match to replay it)"), (x, y), INK, &f.bold, Anchor::TopLeft);
        y += 26;
        let colw = 150.max((body.w - 24) / rounds.len().max(1) as i32);
        let top = y;
        let fav = bracket["favorite"].as_i64();
        for (ri, rnd) in rounds.iter().enumerate() {
            for (mi, mt) in rnd.as_array().cloned().unwrap_or_default().iter().enumerate() {
                let span = 1 << ri;
                let my = top + ((mi as f64 + 0.5) * f64::from(span) * 50.0 - 22.0) as i32;
                let bx = Rect::new(x + ri as i32 * colw, my, colw - 14, 44);
                let sel = self.sel_match == Some((ri, mi));
                cv.fill_rect(bx, if sel { [36, 44, 60] } else { [26, 30, 40] }, 6);
                cv.stroke_rect(bx, if sel { ACCENT } else { BORDER }, 6);
                for (k, key) in ["a", "b"].iter().enumerate() {
                    let fly = mt[*key].as_i64().unwrap_or(0);
                    let won = mt["winner"].as_i64() == Some(fly);
                    m.text(
                        cv,
                        &format!("{}{fly}  {}", if fav == Some(fly) { "FAV " } else { "" }, if won { tr("won") } else { String::new() }),
                        (bx.x + 8, bx.y + 4 + k as i32 * 18),
                        if won { GOOD } else { TEXT },
                        &f.small,
                        Anchor::TopLeft,
                    );
                }
                let (a, b) = (mt["a"].to_string(), mt["b"].to_string());
                let tip = format!(
                    "{}; hits {}-{}{}",
                    mt["decided_by"].as_str().unwrap_or(""),
                    mt["hits"][&a],
                    mt["hits"][&b],
                    if mt["coin_toss"].as_bool().unwrap_or(false) { "; coin toss" } else { "" }
                );
                m.click_area(bx, Act::Custom(format!("match:{ri}:{mi}")), &format!("arc_match:{ri}:{mi}"), true, Some(&tip));
            }
        }
        y = top + rounds.first().and_then(Value::as_array).map_or(0, Vec::len) as i32 * 50 + 12;
        if let Some((ri, mi)) = self.sel_match
            && let Some(mt) = rounds.get(ri).and_then(|r| r.get(mi))
        {
            let (a, b) = (mt["a"].to_string(), mt["b"].to_string());
            let coin = mt["coin_toss"].as_bool().unwrap_or(false);
            m.text(
                cv,
                &format!(
                    "{}: {a} v {b}  ->  {}  ({}{})",
                    tr("Replay"),
                    mt["winner"],
                    mt["decided_by"].as_str().unwrap_or(""),
                    if coin { format!(", {}", tr("coin toss")) } else { String::new() }
                ),
                (x, y),
                INK,
                &f.bold,
                Anchor::TopLeft,
            );
            y += 26;
            let side = 330.min(body.w / 2);
            let arena = Rect::new(x, y, side, side);
            let frames = mt["frames"].as_array().cloned().unwrap_or_default();
            let length = frames.len() as f64 / 10.0;
            self.advance(m.now(), length);
            self.draw_duel(m, cv, arena, &frames, self.replay_t, (&a, &b), cols);
            m.button(
                cv,
                Rect::new(arena.right() + 16, y, 120, 32),
                &if self.playing { tr("Pause") } else { tr("Play") },
                Act::Custom("play".into()),
                "arc_play",
                Style::Normal,
                true,
                None,
                false,
                None,
            );
            m.page_slider(
                cv,
                Rect::new(arena.right() + 16, y + 40, 300, 30),
                self.replay_t,
                (0.0, length.max(0.5), 0.1),
                "{:.1f} s",
                "arc_scrub",
                true,
                None,
            );
            let ev: Vec<String> = mt["events"]
                .as_array()
                .map(|e| {
                    e.iter()
                        .filter(|e| matches!(e[0].as_str(), Some("hit" | "dodge" | "run")))
                        .take(6)
                        .map(|e| format!("{} {}@{:.1}s", e[0].as_str().unwrap_or(""), e[2].as_str().unwrap_or(""), f64_of(&e[1])))
                        .collect()
                })
                .unwrap_or_default();
            let ey = m.wrapped(
                cv,
                &format!("{} {}", tr("Events:"), ev.join(", ")),
                (arena.right() + 16, y + 80),
                body.right() - arena.right() - 32,
                LABEL,
                &f.small,
                4,
            );
            m.text(
                cv,
                &format!(
                    "{} {}-{}  ·  max DNp35 level {:.1} / {:.1}",
                    tr("shots"),
                    mt["shots"][&a],
                    mt["shots"][&b],
                    f64_of(&mt["max_levels"][&a]["fire"]),
                    f64_of(&mt["max_levels"][&b]["fire"])
                ),
                (arena.right() + 16, ey + 2),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            y += side + 14;
        }
        let ch = bracket["champion"].as_i64().unwrap_or(0);
        let c = card_of(cards, ch);
        let won = bracket["rounds_won"][ch.to_string()].as_i64().unwrap_or(0);
        m.text(
            cv,
            &format!("{}: fly {ch}  {}  ({} {won})", tr("Champion"), c.and_then(|c| c["title"].as_str()).unwrap_or(""), tr("rounds won")),
            (x, y),
            AMBER,
            &f.head,
            Anchor::TopLeft,
        );
        y += 38;
        if let Some(c) = c {
            m.text(cv, c["summary"].as_str().unwrap_or(""), (x, y), TEXT, &f.small, Anchor::TopLeft);
            y += 22;
        }
        m.text(
            cv,
            &tr("Which neurons fired before the champion's landed shots (a correlation: nothing was silenced; the touch neurons lead because a hit fly also fires them):"),
            (x, y),
            LABEL,
            &f.small,
            Anchor::TopLeft,
        );
        y += 22;
        for r in bracket["champion_drivers"].as_array().map(|v| v.as_slice()).unwrap_or(&[]).iter().take(6) {
            m.text(
                cv,
                &format!(
                    "{:<14} x{:.1}  ({:.1} vs {:.1} Hz, {} neurons)",
                    r["type"].as_str().unwrap_or(""),
                    f64_of(&r["ratio"]),
                    f64_of(&r["hz_before_hits"]),
                    f64_of(&r["hz_overall"]),
                    r["neurons"]
                ),
                (x + 8, y),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            y += 18;
        }
        let ro: Vec<String> = bracket["champion_readouts"]
            .as_array()
            .map(|v| v.iter().take(8).map(|r| format!("{} x{:.1}", r["type"].as_str().unwrap_or(""), f64_of(&r["ratio"]))).collect())
            .unwrap_or_default();
        if !ro.is_empty() {
            y += 4;
            m.text(
                cv,
                &format!("{}  {}", tr("Neurons the duel's own rules read:"), ro.join("  ")),
                (x, y),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            y += 22;
        }
        y + 6
    }

    #[allow(clippy::too_many_lines)]
    fn racing(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, mut y: i32, d: &Value, busy: bool, cols: &[[u8; 3]; 2]) -> i32 {
        let f = m.fonts();
        let points = d["points"].as_i64().unwrap_or(0);
        let mut x = body.x + 8;
        y = m.wrapped(
            cv,
            &tr("Flies race down a track with sugar and fruit lures, each through its own brain. Bet points on a fly; the odds come from its personality card."),
            (x, y),
            body.w - 24,
            LABEL,
            &f.small,
            4,
        ) + 6;
        m.page_slider(cv, Rect::new(x, y, 260, 32), self.lanes as f64, (3.0, 8.0, 1.0), "{:.0f} lanes", "race_lanes", !busy, None);
        m.button(
            cv,
            Rect::new(x + 280, y, 130, 32),
            &tr("New field"),
            Act::Custom("new_field".into()),
            "race_new",
            Style::Normal,
            !busy,
            None,
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(x + 420, y, 200, 32),
            &tr("Look at the field"),
            Act::Custom("look".into()),
            "race_look",
            Style::Normal,
            !busy,
            Some(&tr("Measures each fly's personality card (a few seconds each) and sets the odds.")),
            false,
            None,
        );
        y += 44;
        let mut xx = tag(m, cv, x, y, "CONNECTOME");
        xx = tag(m, cv, xx, y, "GAME RULE");
        xx = tag(m, cv, xx, y, "MODEL PREDICTION");
        y = (y + 30).max(
            m.wrapped(
                cv,
                &tr("walking neurons set the speed (CONNECTOME); track, lures, odds, points (GAME RULE); the finish (MODEL PREDICTION)"),
                (xx + 6, y + 2),
                body.right() - xx - 16,
                LABEL,
                &f.small,
                4,
            ) + 6,
        );
        let Some(odds) = d["odds"].as_array() else {
            m.text(cv, &tr("Look at the field to see the odds."), (x, y), LABEL, &f.text, Anchor::TopLeft);
            return y + 30;
        };
        y = m.wrapped(
            cv,
            &tr("Odds (form = sugar response and calm walking drive, from the measured card). Click a fly to bet on it:"),
            (x, y),
            body.w - 24,
            INK,
            &f.bold,
            3,
        ) + 4;
        let field = d["field"].as_array().cloned().unwrap_or_default();
        for o in odds {
            let fly = o["fly"].as_i64().unwrap_or(0);
            let c = field.iter().find(|c| c["seed"].as_i64() == Some(fly));
            let tip = c.map(|c| {
                format!(
                    "{} · sugar x{:.2}, calm walking drive {:.2}",
                    c["summary"].as_str().unwrap_or(""),
                    f64_of(&c["sugar_ratio"]),
                    f64_of(&c["walk_level_calm"])
                )
            });
            m.button(
                cv,
                Rect::new(x, y, body.w - 24, 32),
                &format!(
                    "{}fly {fly}  {}   x{}   ({:.0}%)",
                    if self.bet_fly == Some(fly) { "BET  " } else { "" },
                    o["title"].as_str().unwrap_or(""),
                    crate::menu::py_g(f64_of(&o["decimal_odds"])),
                    f64_of(&o["p_win"]) * 100.0
                ),
                Act::Custom(format!("bet:{fly}")),
                &format!("race_bet:{fly}"),
                Style::Normal,
                !busy,
                tip.as_deref(),
                self.bet_fly == Some(fly),
                None,
            );
            y += 36;
        }
        for s in STAKES {
            m.button(
                cv,
                Rect::new(x, y, 80, 32),
                &s.to_string(),
                Act::Custom(format!("stake:{s}")),
                &format!("race_stake:{s}"),
                Style::Normal,
                !busy && s <= points,
                Some(&tr("Points to stake (game points only).")),
                self.stake == s,
                None,
            );
            x += 88;
        }
        m.button(
            cv,
            Rect::new(x + 10, y, 170, 32),
            &tr("Start race"),
            Act::Custom("start_race".into()),
            "race_go",
            Style::Primary,
            !busy,
            Some(&tr("Runs every lane in the background. Betting is optional; no money is involved, ever.")),
            false,
            None,
        );
        y += 44;
        x = body.x + 8;
        let r = &d["race"];
        if !r.is_object() {
            return y;
        }
        let winner = r["winner"].as_i64().unwrap_or(-1);
        let order: Vec<String> = r["orders"][0].as_array().map(|v| v.iter().map(|s| s.to_string()).collect()).unwrap_or_default();
        m.text(cv, &format!("{} {}", tr("Finishing order:"), order.join("  >  ")), (x, y), AMBER, &f.bold, Anchor::TopLeft);
        y += 26;
        if let Some(b) = d["bet_result"].as_array().and_then(|a| a.first()) {
            let won = b["won"].as_bool().unwrap_or(false);
            m.text(
                cv,
                &format!(
                    "{} {} {}",
                    if won { tr("You won") } else { tr("You lost") },
                    b["delta"].as_i64().unwrap_or(0).abs(),
                    tr_args("points on fly {f} at x{o}.", &[("f", &b["fly"].to_string()), ("o", &crate::menu::py_g(f64_of(&b["odds"])))])
                ),
                (x, y),
                if won { GOOD } else { BAD },
                &f.text,
                Anchor::TopLeft,
            );
            y += 26;
        }
        let seeds: Vec<i64> = r["seeds"].as_array().map(|v| v.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
        let runs = &r["runs"]["0"];
        let length = seeds
            .iter()
            .filter_map(|s| runs[s.to_string()]["trace"].as_array().and_then(|t| t.last()).map(|l| f64_of(&l[0])))
            .fold(1.0f64, f64::max);
        self.advance(m.now(), length + 1.0);
        let track = Rect::new(x, y, body.w - 40, 34 * seeds.len() as i32 + 12);
        cv.fill_rect(track, [12, 15, 20], 8);
        let track_m = f64_of(&r["track_m"]).max(1e-9);
        for (i, s) in seeds.iter().enumerate() {
            let lane = &runs[s.to_string()];
            let ly = track.y + 8 + i as i32 * 34;
            cv.line((track.x + 70, ly + 14), (track.right() - 20, ly + 14), 2, [50, 56, 68], 255);
            for l in r["lures"].as_array().map(|v| v.as_slice()).unwrap_or(&[]) {
                let cx = track.x + 70 + (f64_of(&l[0]) / track_m * f64::from(track.w - 90)) as i32;
                cv.circle((cx, ly + 14), 6, if l[1].as_str() == Some("sugar") { [240, 200, 80] } else { [220, 90, 90] });
            }
            let finished = lane["finish_s"].is_number();
            let mut pos = if finished { track_m } else { f64_of(&lane["distance"]) };
            let trace = lane["trace"].as_array().cloned().unwrap_or_default();
            let mut broke = false;
            for p in &trace {
                if f64_of(&p[0]) >= self.replay_t {
                    broke = true;
                    break;
                }
                pos = f64_of(&p[1]);
            }
            if !broke {
                pos = if finished { track_m } else { trace.last().map_or(0.0, |l| f64_of(&l[1])) };
            }
            let px = track.x + 70 + (pos / track_m * f64::from(track.w - 90)) as i32;
            cv.circle((px, ly + 14), 10, if *s != winner { cols[i % 2] } else { AMBER });
            m.text(cv, &s.to_string(), (track.x + 8, ly + 6), TEXT, &f.small, Anchor::TopLeft);
        }
        y += track.h + 10;
        y = m.wrapped(
            cv,
            &tr("MODEL PREDICTION. Points only. The headless race assay asks whether individuality predicts the finish: --headless --race"),
            (x, y),
            body.w - 24,
            LABEL,
            &f.small,
            3,
        );
        y + 8
    }
}

impl Page for ArcadePage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let d = ctx.host.query("arcade");
        if d["bracket_serial"].as_i64().unwrap_or(0) != self.bracket_serial {
            self.bracket_serial = d["bracket_serial"].as_i64().unwrap_or(0);
            self.replay_t = 0.0;
            self.playing = true;
            self.sel_match = d["bracket"]["rounds"].as_array().filter(|r| !r.is_empty()).map(|r| (r.len() - 1, 0));
        }
        if d["race_serial"].as_i64().unwrap_or(0) != self.race_serial {
            self.race_serial = d["race_serial"].as_i64().unwrap_or(0);
            self.replay_t = 0.0;
            self.playing = true;
        }
        let cols = series(d["palette"].as_str().unwrap_or("default"));
        let busy = d["job"].is_object();
        m.text(cv, &tr("FLY ARCADE"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        m.text(
            cv,
            &tr_args("Points: {n}  (game points only: no money, nothing to buy)", &[("n", &d["points"].as_i64().unwrap_or(0).to_string())]),
            (rect.right() - 24, rect.y + 22),
            AMBER,
            &f.small,
            Anchor::TopRight,
        );
        let mut x = rect.x + 24;
        for (tab, label) in [(Tab::Tournament, tr("Tournament")), (Tab::Race, tr("Racing")), (Tab::Battle, tr("Lesion battle"))] {
            m.button(
                cv,
                Rect::new(x, rect.y + 54, 150, 32),
                &label,
                Act::Custom(format!("tab:{}", tab as u8)),
                &format!("arcade_tab:{}", tab as u8),
                Style::Normal,
                true,
                None,
                self.tab == tab,
                None,
            );
            x += 158;
        }
        let unmeasured = d["unmeasured"].as_i64().unwrap_or(0);
        let label = tr_args("Measure the flies in play ({n})", &[("n", &unmeasured.to_string())]);
        m.button(
            cv,
            Rect::new(x + 10, rect.y + 54, m.bw(&label, 300), 32),
            &label,
            Act::Custom("measure_play".into()),
            "arcade_measure_play",
            Style::Normal,
            d["slots"].as_i64().unwrap_or(0) > 0 && unmeasured > 0 && !busy,
            Some(&tr("Reads each fly's personality card from its own brain (looming latency, sugar response, steering), about 10 s of compute per fly in the background. Until then a fly shows 'card not measured': the game never shows numbers it did not measure.")),
            false,
            None,
        );
        let body = Rect::new(rect.x + 16, rect.y + 96, rect.w - 32, rect.h - 96 - 76);
        let key = match self.tab {
            Tab::Tournament => "arcade_tournament",
            Tab::Race => "arcade_race",
            Tab::Battle => "arcade_battle",
        };
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y + 4 - off;
        y = match self.tab {
            Tab::Tournament => self.tournament(m, cv, body, y, &d, busy, &cols),
            Tab::Race => self.racing(m, cv, body, y, &d, busy, &cols),
            Tab::Battle => m.wrapped(
                cv,
                &tr("The lesion battle is in the game's own page (the pause menu's Fly arcade overlay): see PORTING.md."),
                (body.x + 8, y),
                body.w - 24,
                LABEL,
                &f.small,
                4,
            ),
        };
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.end_clip(cv);
        if let Some(e) = d["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, e, (rect.x + 24, rect.bottom() - 84), BAD, &f.small, Anchor::TopLeft);
        }
        if let Some(j) = d["job"].as_object() {
            let bar = Rect::new(rect.x + 24, rect.bottom() - 44, rect.w - 48, 12);
            cv.fill_rect(Rect::new(bar.x, bar.y, bar.w - 110, 12), [30, 36, 48], 6);
            let frac = j.get("frac").map_or(0.0, f64_of);
            cv.fill_rect(Rect::new(bar.x, bar.y, 8.max(((bar.w - 110) as f64 * frac) as i32), 12), AMBER, 6);
            m.text(
                cv,
                &format!(
                    "{}: {}  ({:.0} s)",
                    j.get("label").and_then(Value::as_str).unwrap_or(""),
                    j.get("note").and_then(Value::as_str).unwrap_or(""),
                    j.get("elapsed").map_or(0.0, f64_of)
                ),
                (bar.x, bar.y + 18),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            m.button(
                cv,
                Rect::new(bar.right() - 100, bar.y - 8, 100, 30),
                "Cancel",
                Act::Action("arcade:cancel".into()),
                &format!("bgjob_cancel:{}", j.get("label").and_then(Value::as_str).unwrap_or("")),
                Style::Danger,
                true,
                Some("Stops after the current step. Nothing already written is lost."),
                false,
                None,
            );
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            &tr("Back"),
            Act::Back,
            "arcade:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let (verb, arg) = name.split_once(':').unwrap_or((name, ""));
        let busy = ctx.host.query("arcade")["job"].is_object();
        match verb {
            "tab" => {
                self.tab = match arg {
                    "1" => Tab::Race,
                    "2" => {
                        // the lesion battle is the game's own page (`ui/battle_ui.py` is not ported here)
                        m.action("arcade:battle");
                        Tab::Tournament
                    }
                    _ => Tab::Tournament,
                }
            }
            "size" => {
                self.size = arg.parse().unwrap_or(8);
                self.favorite = None;
            }
            "new_flies" => {
                self.base_seed += self.size as i64;
                self.favorite = None;
            }
            "fav" => self.favorite = arg.parse().ok(),
            "match" => {
                let mut it = arg.split(':').filter_map(|v| v.parse::<usize>().ok());
                self.sel_match = it.next().zip(it.next());
                self.replay_t = 0.0;
            }
            "play" => self.playing = !self.playing,
            "slide" => {
                let v: f64 = text.and_then(|t| t.parse().ok()).unwrap_or(0.0);
                match arg {
                    "arc_len" => self.match_s = v,
                    "arc_scrub" => self.replay_t = v,
                    "race_lanes" => self.lanes = v.round() as i64,
                    _ => {}
                }
            }
            "new_field" => self.race_base += self.lanes,
            "bet" => self.bet_fly = arg.parse().ok(),
            "stake" => self.stake = arg.parse().unwrap_or(10),
            "run_tournament" if !busy => {
                m.action(&format!("arcade:tournament:{}:{}:{}:{}", self.size, self.base_seed, self.match_s, self.favorite.unwrap_or(-1)))
            }
            "measure_play" if !busy => m.action("arcade:measure_play"),
            "look" if !busy => m.action(&format!("arcade:look:{}:{}", self.lanes, self.race_base)),
            "start_race" if !busy => {
                m.action(&format!("arcade:race:{}:{}:{}:{}", self.lanes, self.race_base, self.bet_fly.unwrap_or(-1), self.stake))
            }
            _ => {}
        }
    }
}
