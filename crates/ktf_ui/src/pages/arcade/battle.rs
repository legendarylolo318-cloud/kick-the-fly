//! The Lesion battle tab of the Fly arcade (`ui/battle_ui.py`, rules in `game/lesion_battle.py`): two players, one screen, each does surgery on
//! their own fly within a budget, then the flies fight. Hot-seat setup, saved loadouts, share codes (an ordinary surgery code), the fight as a
//! background job and the replay.
//!
//! Host contract (besides the arcade page's): query `arcade_sizes` = `{cell type: neurons}`, `lesion_saved` = `[[name, [[type, mode], ...]], ...]`;
//! actions `arcade:fight:<json {seed_a, seed_b, a, b, rounds, seconds}>` and `arcade:lesion_save:<json {name, picks}>`; the arcade query's
//! `battle` = `{serial, winner, wins:{A,B}, rounds:[{winner, frames, shots, hits, hp, max_levels}], loadouts:{A:[[t,m]],B:[[t,m]]}}`.

use serde_json::{Value, json};

use super::{ArcadePage, f64_of, tag};
use crate::canvas::{Canvas, Rect};
use crate::i18n::{tr, tr_args};
use crate::menu::{Act, Anchor, Ctx, Menu};
use crate::theme::*;

pub const BUDGET: usize = 3;
pub const MAX_TYPE_NEURONS: i64 = 500;
pub const MAX_SAVED: usize = 20;
pub const ROUNDS: [u32; 2] = [1, 3];
pub const SILENCE: i8 = -1;
pub const STIMULATE: i8 = 1;
const SUGGESTED: [&str; 15] =
    ["DNp35", "DNpe052", "DNa01", "DNa02", "DNp01", "DNp09", "LC10a", "LC10b", "LC11", "LPLC2", "LC4", "MDN", "DNg02_a", "PPL1", "PAM"];

pub fn mode_word(m: i8) -> &'static str {
    if m < 0 { "silenced" } else { "stimulated" }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Loadout {
    pub name: String,
    pub picks: Vec<(String, i8)>,
}

impl Loadout {
    pub fn new(name: &str) -> Self {
        Self { name: name.into(), picks: Vec::new() }
    }

    pub fn add(&mut self, t: &str, mode: i8, sizes: &Value) -> Result<(), String> {
        let t = t.trim();
        match sizes.get(t).and_then(Value::as_i64) {
            None => return Err(format!("{t:?} is not a cell type of this brain")),
            Some(n) if n > MAX_TYPE_NEURONS => {
                return Err(format!("{t} has {} neurons; a pick may have at most {MAX_TYPE_NEURONS}", group(n)));
            }
            _ => {}
        }
        if let Some(p) = self.picks.iter_mut().find(|p| p.0 == t) {
            p.1 = mode;
        } else if self.picks.len() >= BUDGET {
            return Err(format!("the budget is {BUDGET} picks: take one out first"));
        } else {
            self.picks.push((t.to_string(), mode));
        }
        Ok(())
    }

    pub fn neurons(&self, sizes: &Value) -> i64 {
        self.picks.iter().map(|p| sizes.get(&p.0).and_then(Value::as_i64).unwrap_or(0)).sum()
    }

    pub fn problems(&self, sizes: &Value) -> Vec<String> {
        let mut out = Vec::new();
        if self.picks.len() > BUDGET {
            out.push(format!("{} picks is over the budget of {BUDGET}", self.picks.len()));
        }
        for (t, _) in &self.picks {
            match sizes.get(t).and_then(Value::as_i64) {
                None => out.push(format!("{t} is not a cell type of this brain")),
                Some(n) if n > MAX_TYPE_NEURONS => out.push(format!("{t} has {} neurons (over {MAX_TYPE_NEURONS})", group(n))),
                _ => {}
            }
        }
        out
    }

    /// `KTF1-SRG-...`: the same code Esc > Share makes for a surgery that names these cell types.
    pub fn code(&self) -> Result<String, String> {
        if self.picks.is_empty() {
            return Err("nothing to share: add a pick first".into());
        }
        let types: serde_json::Map<String, Value> = self.picks.iter().map(|(t, m)| (t.clone(), json!(m))).collect();
        ktf_data::sharecode::encode("surgery", &json!({ "types": types }), None).map_err(|e| e.to_string())
    }

    pub fn from_code(text: &str, sizes: &Value, name: &str) -> Result<Self, String> {
        let c = ktf_data::sharecode::decode(text).map_err(|e| e.to_string())?;
        if c.kind != "surgery" {
            return Err(format!(
                "this is a {} code; a lesion loadout is a surgery code",
                ktf_data::sharecode::kind_label(&c.kind).to_lowercase()
            ));
        }
        if c.payload.get("groups").and_then(Value::as_object).is_some_and(|g| !g.is_empty()) {
            return Err("a lesion loadout names cell types only; this surgery code also switches whole groups".into());
        }
        let types = c.payload.get("types").and_then(Value::as_object).filter(|t| !t.is_empty());
        let Some(types) = types else { return Err("this surgery code names no cell types".into()) };
        let lo = Self { name: name.into(), picks: types.iter().map(|(t, m)| (t.clone(), m.as_i64().unwrap_or(0) as i8)).collect() };
        let bad = lo.problems(sizes);
        if !bad.is_empty() {
            return Err(format!("this loadout cannot be used here: {}", bad.into_iter().take(3).collect::<Vec<_>>().join("; ")));
        }
        Ok(lo)
    }
}

/// `{:,}`
fn group(n: i64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Setup1,
    Pass,
    Setup2,
    Ready,
    Fighting,
    Result,
}

#[derive(Debug, Clone)]
pub struct Battle {
    pub phase: Phase,
    pub los: [Loadout; 2],
    pub effect: i8,
    pub search: String,
    pub rounds: u32,
    pub seconds: f64,
    pub base_seed: i64,
    pub serial: i64,
    pub sel_round: usize,
    pub replay_t: f64,
    pub playing: bool,
    pub last: f64,
    pub msg: String,
    pub msg_bad: bool,
    pub paste_note: String,
}

impl Default for Battle {
    fn default() -> Self {
        Self {
            phase: Phase::Setup1,
            los: [Loadout::new("player 1"), Loadout::new("player 2")],
            effect: SILENCE,
            search: String::new(),
            rounds: 1,
            seconds: 20.0,
            base_seed: 5000,
            serial: 0,
            sel_round: 0,
            replay_t: 0.0,
            playing: true,
            last: 0.0,
            msg: String::new(),
            msg_bad: false,
            paste_note: String::new(),
        }
    }
}

impl Battle {
    fn player(&self) -> usize {
        usize::from(self.phase == Phase::Setup2)
    }

    fn reset(&mut self) {
        self.phase = Phase::Setup1;
        self.msg.clear();
        self.los = [Loadout::new("player 1"), Loadout::new("player 2")];
        self.search.clear();
    }

    fn say(&mut self, text: &str, bad: bool) {
        self.msg = text.into();
        self.msg_bad = bad;
    }
}

/// The cell types a pick may name that match the search (exact first, then starts-with, then contains), within the size cap.
pub fn candidates(sizes: &Value, query: &str, limit: usize) -> Vec<(String, i64)> {
    let q = query.trim().to_lowercase();
    let all: Vec<(String, i64)> =
        sizes.as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0))).collect()).unwrap_or_default();
    if q.is_empty() {
        return SUGGESTED
            .iter()
            .filter_map(|t| sizes.get(*t).and_then(Value::as_i64).filter(|&n| n <= MAX_TYPE_NEURONS).map(|n| ((*t).to_string(), n)))
            .take(limit)
            .collect();
    }
    let mut v: Vec<(String, i64)> = all.into_iter().filter(|(t, n)| *n <= MAX_TYPE_NEURONS && t.to_lowercase().contains(&q)).collect();
    v.sort_by_key(|(t, _)| (t.to_lowercase() != q, !t.to_lowercase().starts_with(&q), t.to_lowercase()));
    v.truncate(limit);
    v
}

pub fn headline(r: &Value) -> String {
    let wins = |s: &str| r["wins"][s].as_i64().unwrap_or(0);
    match r["winner"].as_str() {
        None => format!("A draw ({}-{}).", wins("A"), wins("B")),
        Some(w) => {
            let side = if w == "A" { "Player 1" } else { "Player 2" };
            let how = pairs(&r["loadouts"][w]).iter().map(|(t, m)| format!("{t} {}", mode_word(*m))).collect::<Vec<_>>().join(", ");
            let other = if w == "A" { "B" } else { "A" };
            format!("{side} wins {}-{} with {}.", wins(w), wins(other), if how.is_empty() { "no surgery at all".to_string() } else { how })
        }
    }
}

fn pairs(v: &Value) -> Vec<(String, i8)> {
    v.as_array().map(|a| a.iter().filter_map(|p| Some((p[0].as_str()?.to_string(), p[1].as_i64()? as i8))).collect()).unwrap_or_default()
}

impl ArcadePage {
    pub(super) fn battle(
        &mut self,
        m: &mut Menu,
        cv: &mut Canvas,
        body: Rect,
        mut y: i32,
        d: &Value,
        busy: bool,
        cols: &[[u8; 3]; 2],
        ctx: &mut Ctx,
    ) -> i32 {
        let f = m.fonts();
        let x = body.x + 8;
        // a finished (or abandoned) job
        let serial = d["battle"]["serial"].as_i64().unwrap_or(0);
        if serial != self.bt.serial {
            self.bt.serial = serial;
            if d["battle"]["rounds"].is_array() {
                self.bt.phase = Phase::Result;
                self.bt.sel_round = 0;
                self.bt.replay_t = 0.0;
                self.bt.playing = true;
            }
        }
        if self.bt.phase == Phase::Fighting && !busy && serial == self.bt.serial {
            self.bt.phase = Phase::Ready;
        }
        let sizes = ctx.host.query("arcade_sizes");
        y = m.wrapped(
            cv,
            &tr_args(
                "Each player does surgery on their own fly, within a budget of {n} cell types (each silenced or stimulated, up to {c} neurons), then the two flies fight. Hot-seat: the second player chooses while the first player's picks are hidden.",
                &[("n", &BUDGET.to_string()), ("c", &MAX_TYPE_NEURONS.to_string())],
            ),
            (x, y),
            body.w - 24,
            LABEL,
            &f.small,
            4,
        ) + 6;
        let mut xx = tag(m, cv, x, y, "CONNECTOME");
        xx = tag(m, cv, xx, y, "GAME RULE");
        xx = tag(m, cv, xx, y, "MODEL PREDICTION");
        y = (y + 30).max(
            m.wrapped(
                cv,
                &tr("what the surgery does (CONNECTOME); the budget and the arena (GAME RULE); who wins (MODEL PREDICTION)"),
                (xx + 6, y + 2),
                body.right() - xx - 16,
                LABEL,
                &f.small,
                3,
            ) + 6,
        );
        match self.bt.phase {
            Phase::Setup1 | Phase::Setup2 => self.setup(m, cv, body, y, &sizes, ctx),
            Phase::Pass => {
                m.text(cv, &tr("PLAYER 1 IS DONE."), (x, y), AMBER, &f.head, Anchor::TopLeft);
                y += 44;
                y = m.wrapped(
                    cv,
                    &tr("Player 1: look away. Hand over to player 2, who now operates on their own fly. Player 1's picks are not shown again until the fight."),
                    (x, y),
                    body.w - 24,
                    TEXT,
                    &f.text,
                    4,
                ) + 10;
                m.button(
                    cv,
                    Rect::new(x, y, 260, 40),
                    &tr("Player 2 is ready"),
                    Act::Custom("bt:pass".into()),
                    "bt_pass",
                    crate::menu::Style::Primary,
                    true,
                    None,
                    false,
                    None,
                );
                y + 52
            }
            Phase::Ready | Phase::Fighting => {
                m.text(cv, &tr("READY TO FIGHT"), (x, y), AMBER, &f.head, Anchor::TopLeft);
                y += 44;
                for (i, lo) in self.bt.los.iter().enumerate() {
                    m.text(
                        cv,
                        &tr_args(
                            "Player {n}: {k} of {b} picks, {t} neurons (hidden until the fight)",
                            &[
                                ("n", &(i + 1).to_string()),
                                ("k", &lo.picks.len().to_string()),
                                ("b", &BUDGET.to_string()),
                                ("t", &lo.neurons(&sizes).to_string()),
                            ],
                        ),
                        (x, y),
                        TEXT,
                        &f.text,
                        Anchor::TopLeft,
                    );
                    y += 26;
                }
                y += 6;
                let mut bx = x;
                for r in ROUNDS {
                    let label = if r == 1 {
                        tr_args("{n} round(s)", &[("n", &r.to_string())])
                    } else {
                        tr_args("best of {n}", &[("n", &r.to_string())])
                    };
                    m.button(
                        cv,
                        Rect::new(bx, y, 130, 32),
                        &label,
                        Act::Custom(format!("bt:rounds:{r}")),
                        &format!("bt_rounds:{r}"),
                        crate::menu::Style::Normal,
                        !busy,
                        None,
                        self.bt.rounds == r,
                        None,
                    );
                    bx += 138;
                }
                m.page_slider(
                    cv,
                    Rect::new(bx + 6, y, 240, 32),
                    self.bt.seconds,
                    (10.0, 40.0, 5.0),
                    "{:.0f} s rounds",
                    "bt_len",
                    !busy,
                    None,
                );
                m.button(
                    cv,
                    Rect::new(bx + 260, y, 190, 32),
                    &tr("New flies"),
                    Act::Custom("bt:new".into()),
                    "bt_new",
                    crate::menu::Style::Normal,
                    !busy,
                    Some(&tr("Pick a different pair of individuality seeds. The same two flies are used in every round.")),
                    false,
                    None,
                );
                y += 44;
                m.button(
                    cv,
                    Rect::new(x, y, 220, 44),
                    &tr("FIGHT"),
                    Act::Custom("bt:fight".into()),
                    "bt_fight",
                    crate::menu::Style::Primary,
                    !busy,
                    Some(&tr("Runs the duel in the background (about a minute a round).")),
                    false,
                    None,
                );
                m.button(
                    cv,
                    Rect::new(x + 232, y, 190, 44),
                    &tr("Start again"),
                    Act::Custom("bt:reset".into()),
                    "bt_reset",
                    crate::menu::Style::Normal,
                    !busy,
                    None,
                    false,
                    None,
                );
                y + 56
            }
            Phase::Result => self.result(m, cv, body, y, d, cols),
        }
    }

    fn setup(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, mut y: i32, sizes: &Value, ctx: &mut Ctx) -> i32 {
        let f = m.fonts();
        let x = body.x + 8;
        let p = self.bt.player();
        m.text(cv, &tr_args("PLAYER {n}: operate on your fly", &[("n", &(p + 1).to_string())]), (x, y), AMBER, &f.head, Anchor::TopLeft);
        y += 42;
        m.text(
            cv,
            &tr_args("{k} of {b} picks used", &[("k", &self.bt.los[p].picks.len().to_string()), ("b", &BUDGET.to_string())]),
            (x, y),
            INK,
            &f.bold,
            Anchor::TopLeft,
        );
        m.segmented(
            cv,
            Rect::new(x + 200, y - 4, 260, 32),
            &[tr("Silence"), tr("Stimulate")],
            usize::from(self.bt.effect != SILENCE),
            &[Act::Custom("bt:effect:-1".into()), Act::Custom("bt:effect:1".into())],
            "bt_effect",
            true,
            &tr("Silencing holds every neuron of the type quiet; stimulating drives it, as the game's brain surgery does."),
        );
        y += 38;
        let picks = self.bt.los[p].picks.clone();
        for (t, mode) in &picks {
            let r = Rect::new(x, y, 330, 28);
            cv.fill_rect(r, [28, 32, 42], 6);
            m.text(
                cv,
                &format!("{t}  {}  ({} neurons)", mode_word(*mode), sizes.get(t).and_then(Value::as_i64).unwrap_or(0)),
                (r.x + 10, r.y + 5),
                if *mode > 0 { [255, 150, 90] } else { [110, 190, 255] },
                &f.small,
                Anchor::TopLeft,
            );
            m.button(
                cv,
                Rect::new(r.right() + 8, y, 80, 28),
                &tr("Remove"),
                Act::Custom(format!("bt:rm:{t}")),
                &format!("bt_rm:{t}"),
                crate::menu::Style::Normal,
                true,
                None,
                false,
                Some(f.small.clone()),
            );
            y += 32;
        }
        if picks.is_empty() {
            m.text(cv, &tr("No picks yet: with none, your fly goes in as it is."), (x, y), LABEL, &f.small, Anchor::TopLeft);
            y += 24;
        }
        y += 6;
        m.text(cv, &tr("Find a cell type:"), (x, y + 4), TEXT, &f.small, Anchor::TopLeft);
        let tip = tr_args(
            "Type part of a name (DNp, LC, MDN, PPL...) and press Enter. Types over {c} neurons are not offered.",
            &[("c", &MAX_TYPE_NEURONS.to_string())],
        );
        m.text_field(cv, Rect::new(x + 150, y, 260, 30), &self.bt.search.clone(), "bt_search", Some(&tip), 24, false);
        y += 40;
        let cands = candidates(sizes, &self.bt.search, 14);
        if self.bt.search.trim().is_empty() {
            m.text(cv, &tr("The types the duel itself reads (a hint, not an answer):"), (x, y), LABEL, &f.small, Anchor::TopLeft);
            y += 20;
        } else if cands.is_empty() {
            m.text(
                cv,
                &tr_args("No cell type of at most {c} neurons matches.", &[("c", &MAX_TYPE_NEURONS.to_string())]),
                (x, y),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            y += 22;
        }
        let items: Vec<_> = cands
            .iter()
            .map(|(t, n)| {
                (format!("{t} ({n})"), Act::Custom(format!("bt:add:{t}")), format!("bt_cand:{t}"), picks.iter().any(|p| &p.0 == t), None)
            })
            .collect();
        y = m.flow_buttons(cv, x, y, body.right() - 16, &items, None, None, 8, None) + 8;
        if !self.bt.msg.is_empty() {
            m.text(cv, &self.bt.msg.clone(), (x, y), if self.bt.msg_bad { BAD } else { GOOD }, &f.small, Anchor::TopLeft);
            y += 22;
        }
        y += 4;
        m.button(
            cv,
            Rect::new(x, y, 190, 40),
            &tr("Done"),
            Act::Custom("bt:done".into()),
            "bt_done",
            crate::menu::Style::Primary,
            true,
            Some(&tr("Hands the screen over (player 1) or readies the fight (player 2). With no picks, the fly fights as it is.")),
            false,
            None,
        );
        let bx = x + 202;
        let has = !picks.is_empty();
        m.button(
            cv,
            Rect::new(bx, y, 140, 40),
            &tr("Copy code"),
            Act::Custom("bt:copy".into()),
            "bt_copy",
            crate::menu::Style::Normal,
            has,
            Some(&tr("A surgery code (KTF1-SRG-...): the same one Esc > Share makes. Anyone can Paste it for their own fly.")),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(bx + 150, y, 140, 40),
            &tr("Paste code"),
            Act::Custom("bt:paste".into()),
            "bt_paste",
            crate::menu::Style::Normal,
            true,
            None,
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(bx + 300, y, 130, 40),
            &tr("Save"),
            Act::Custom("bt:save".into()),
            "bt_save",
            crate::menu::Style::Normal,
            has,
            Some(&tr("Saves this loadout under its name.")),
            false,
            None,
        );
        y += 50;
        let saved = ctx.host.query("lesion_saved");
        if let Some(list) = saved.as_array().filter(|l| !l.is_empty()) {
            m.text(cv, &tr("Saved loadouts:"), (x, y + 4), LABEL, &f.small, Anchor::TopLeft);
            let items: Vec<_> = list
                .iter()
                .filter_map(|e| Some((e[0].as_str()?.to_string(), pairs(&e[1]))))
                .map(|(n, p)| (format!("{n} ({})", p.len()), Act::Custom(format!("bt:load:{n}")), format!("bt_load:{n}"), false, None))
                .collect();
            y = m.flow_buttons(cv, x + 130, y, body.right() - 16, &items, None, None, 8, None) + 6;
        }
        if !self.bt.paste_note.is_empty() {
            y = m.wrapped(cv, &self.bt.paste_note.clone(), (x, y), body.w - 24, DIM, &f.small, 3) + 4;
        }
        y
    }

    fn result(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, mut y: i32, d: &Value, cols: &[[u8; 3]; 2]) -> i32 {
        let f = m.fonts();
        let r = &d["battle"];
        let x = body.x + 8;
        m.text(cv, &headline(r), (x, y), AMBER, &f.head, Anchor::TopLeft);
        y += 42;
        for (i, side) in ["A", "B"].iter().enumerate() {
            let picks = pairs(&r["loadouts"][*side]);
            let what = if picks.is_empty() {
                "no surgery".to_string()
            } else {
                picks.iter().map(|(t, mo)| format!("{t} {}", mode_word(*mo))).collect::<Vec<_>>().join(", ")
            };
            m.text(cv, &format!("Player {}: {what}", i + 1), (x, y), TEXT, &f.text, Anchor::TopLeft);
            y += 24;
        }
        y += 6;
        let rounds = r["rounds"].as_array().cloned().unwrap_or_default();
        for (k, rd) in rounds.iter().enumerate() {
            let w = match rd["winner"].as_str() {
                Some("A") => "Player 1".to_string(),
                Some("B") => "Player 2".to_string(),
                _ => tr("draw"),
            };
            m.button(
                cv,
                Rect::new(x + k as i32 * 190, y, 180, 30),
                &tr_args("Round {n}: {w}", &[("n", &(k + 1).to_string()), ("w", &w)]),
                Act::Custom(format!("bt:round:{k}")),
                &format!("bt_round:{k}"),
                crate::menu::Style::Normal,
                true,
                None,
                self.bt.sel_round == k,
                None,
            );
        }
        y += 42;
        let Some(rd) = rounds.get(self.bt.sel_round.min(rounds.len().saturating_sub(1))) else { return y };
        let side = 330.min(body.w / 2);
        let arena = Rect::new(x, y, side, side);
        let frames = rd["frames"].as_array().cloned().unwrap_or_default();
        let length = frames.len() as f64 / 10.0;
        let now = m.now();
        if self.bt.playing {
            self.bt.replay_t = (self.bt.replay_t + (now - self.bt.last)).rem_euclid(length.max(0.5));
        }
        self.bt.last = now;
        self.draw_duel(m, cv, arena, &frames, self.bt.replay_t, ("Player 1", "Player 2"), cols);
        m.button(
            cv,
            Rect::new(arena.right() + 16, y, 120, 32),
            &if self.bt.playing { tr("Pause") } else { tr("Play") },
            Act::Custom("bt:play".into()),
            "bt_play",
            crate::menu::Style::Normal,
            true,
            None,
            false,
            None,
        );
        m.page_slider(
            cv,
            Rect::new(arena.right() + 16, y + 40, 300, 30),
            self.bt.replay_t,
            (0.0, length.max(0.5), 0.1),
            "{:.1f} s",
            "bt_scrub",
            true,
            None,
        );
        let mut ty = y + 84;
        let n = |side: &str, k: &str| rd[k][side].as_i64().unwrap_or(0);
        m.text(
            cv,
            &tr_args(
                "shots {a}-{b}   hits {c}-{d}   health {e}-{f}",
                &[
                    ("a", &n("A", "shots").to_string()),
                    ("b", &n("B", "shots").to_string()),
                    ("c", &n("A", "hits").to_string()),
                    ("d", &n("B", "hits").to_string()),
                    ("e", &format!("{:.0}", f64_of(&rd["hp"]["A"]))),
                    ("f", &format!("{:.0}", f64_of(&rd["hp"]["B"]))),
                ],
            ),
            (arena.right() + 16, ty),
            TEXT,
            &f.small,
            Anchor::TopLeft,
        );
        ty += 22;
        m.text(
            cv,
            &tr("What the surgery did (the highest level each fly's own readout reached, x its calm rate):"),
            (arena.right() + 16, ty),
            LABEL,
            &f.small,
            Anchor::TopLeft,
        );
        ty += 20;
        for (key, label) in
            [("fire", "shooting DNp35/DNpe052"), ("escape", "dodge DNp01"), ("run", "run (body-touch DNs)"), ("walk", "walk DNp09")]
        {
            m.text(
                cv,
                &format!("{label:<26} P1 x{:.1}   P2 x{:.1}", f64_of(&rd["max_levels"]["A"][key]), f64_of(&rd["max_levels"]["B"][key])),
                (arena.right() + 16, ty),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            ty += 18;
        }
        y += side + 14;
        m.button(
            cv,
            Rect::new(x, y, 230, 40),
            &tr("Rematch (same surgery)"),
            Act::Custom("bt:rematch".into()),
            "bt_rematch",
            crate::menu::Style::Normal,
            true,
            Some(&tr(
                "Back to the fight screen with the same two flies and the same surgery; round noise differs only if you change the seeds.",
            )),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(x + 242, y, 200, 40),
            &tr("New battle"),
            Act::Custom("bt:reset".into()),
            "bt_newbattle",
            crate::menu::Style::Primary,
            true,
            None,
            false,
            None,
        );
        y += 52;
        m.wrapped(
            cv,
            &tr("MODEL PREDICTION: who wins is what this model does with these two individuals, these rules and this noise. Not a measurement of real flies."),
            (x, y),
            body.w - 24,
            LABEL,
            &f.small,
            3,
        ) + 8
    }

    /// A click or text commit of the battle tab (`bt:<verb>[:<arg>]`).
    pub(super) fn battle_custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let rest = name.strip_prefix("bt:").unwrap_or(name);
        let (verb, arg) = rest.split_once(':').unwrap_or((rest, ""));
        let sizes = ctx.host.query("arcade_sizes");
        let p = self.bt.player();
        match verb {
            "pass" => self.bt.phase = Phase::Setup2,
            "effect" => self.bt.effect = if arg == "1" { STIMULATE } else { SILENCE },
            "rm" => self.bt.los[p].picks.retain(|x| x.0 != arg),
            "add" => match self.bt.los[p].add(arg, self.bt.effect, &sizes) {
                Ok(()) => self.bt.say("", false),
                Err(e) => self.bt.say(&e, true),
            },
            "done" => {
                self.bt.phase = if self.bt.phase == Phase::Setup1 { Phase::Pass } else { Phase::Ready };
                self.bt.search.clear();
                self.bt.msg.clear();
            }
            "copy" => match self.bt.los[p].code() {
                Err(e) => self.bt.say(&e, true),
                Ok(code) => {
                    let ok = ctx.host.command("clipboard_put", json!(code)).as_bool().unwrap_or(false);
                    let line = if ok {
                        tr_args("Code copied: {c}", &[("c", &format!("{}...", code.chars().take(30).collect::<String>()))])
                    } else {
                        tr("Could not reach the clipboard; the code is shown below.")
                    };
                    self.bt.say(&line, false);
                    self.bt.paste_note = code;
                }
            },
            "paste" => {
                let t = ctx.host.query("clipboard_get");
                let t = t.as_str().unwrap_or("");
                if t.trim().is_empty() {
                    self.bt.say(&tr("The clipboard has no text."), true);
                } else {
                    match Loadout::from_code(t, &sizes, &self.bt.los[p].name) {
                        Ok(lo) => {
                            let n = lo.picks.len();
                            self.bt.los[p] = lo;
                            self.bt.say(&tr_args("Loaded {n} picks from the code.", &[("n", &n.to_string())]), false);
                        }
                        Err(e) => self.bt.say(&e, true),
                    }
                }
            }
            "save" => {
                let saved = ctx.host.query("lesion_saved");
                let full = saved
                    .as_array()
                    .is_some_and(|l| l.len() >= MAX_SAVED && !l.iter().any(|e| e[0].as_str() == Some(self.bt.los[p].name.as_str())));
                if full {
                    self.bt.say(&tr_args("The saved list is full ({n}); delete one first.", &[("n", &MAX_SAVED.to_string())]), false);
                    return;
                }
                let lo = &mut self.bt.los[p];
                if matches!(lo.name.as_str(), "" | "my fly" | "player 1" | "player 2") {
                    let n = ctx.host.query("lesion_saved").as_array().map_or(0, Vec::len) + 1;
                    lo.name = format!("loadout {n}");
                }
                let picks: Vec<Value> = lo.picks.iter().map(|(t, m)| json!([t, m])).collect();
                m.action(&format!("arcade:lesion_save:{}", json!({"name": lo.name, "picks": picks})));
                let nm = lo.name.clone();
                self.bt.say(&tr_args("Saved as {n}.", &[("n", &nm)]), false);
            }
            "load" => {
                let saved = ctx.host.query("lesion_saved");
                if let Some(e) = saved.as_array().and_then(|l| l.iter().find(|e| e[0].as_str() == Some(arg))) {
                    let lo = Loadout { name: arg.into(), picks: pairs(&e[1]) };
                    let bad = lo.problems(&sizes);
                    if bad.is_empty() {
                        self.bt.los[p] = lo;
                        self.bt.say(&tr_args("Loaded {n}.", &[("n", arg)]), false);
                    } else {
                        self.bt.say(&bad.into_iter().take(2).collect::<Vec<_>>().join("; "), true);
                    }
                }
            }
            "rounds" => self.bt.rounds = arg.parse().unwrap_or(1),
            "new" => self.bt.base_seed += 2,
            "fight" => {
                let pk = |lo: &Loadout| -> Vec<Value> { lo.picks.iter().map(|(t, m)| json!([t, m])).collect() };
                let job = json!({
                    "seed_a": self.bt.base_seed, "seed_b": self.bt.base_seed + 1, "a": pk(&self.bt.los[0]), "b": pk(&self.bt.los[1]),
                    "rounds": self.bt.rounds, "seconds": self.bt.seconds, "battle_seed": self.bt.base_seed,
                });
                m.action(&format!("arcade:fight:{job}"));
                self.bt.phase = Phase::Fighting;
                self.bt.msg.clear();
            }
            "reset" => self.bt.reset(),
            "rematch" => self.bt.phase = Phase::Ready,
            "round" => {
                self.bt.sel_round = arg.parse().unwrap_or(0);
                self.bt.replay_t = 0.0;
            }
            "play" => self.bt.playing = !self.bt.playing,
            "text" if arg == "bt_search" => self.bt.search = text.unwrap_or("").chars().take(24).collect(),
            _ => {}
        }
    }
}
