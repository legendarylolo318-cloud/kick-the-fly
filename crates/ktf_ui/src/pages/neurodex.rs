//! The Neurodex panel (Esc > Neurodex, or the Neurodex key, default D): progress by region, the list of discovered cell types and an entry
//! for each (`ui/neurodex_ui.py`).
//!
//! CONNECTOME: the numbers in an entry. GAME RULE: "discovered" and the collection. LITERATURE: the one-line fact and citation on a curated
//! type. An undiscovered type shows only "???": no name, no data. The search box only finds names you have discovered.

use std::sync::{Arc, Mutex, PoisonError};

use fly_sim::neurodex::TypeTable;
use ktf_data::neurodex_facts::fact_for;
use ktf_data::notd::{self, Card};
use ktf_data::stores::Progress;

use crate::canvas::{Canvas, Rect};
use crate::font::Font;
use crate::i18n::{tr, tr_args};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const ROW_H: i32 = 26;

/// What the game and the panel share: the type table (built on a worker thread), the player's progress, and the requests the panel makes.
pub struct NeurodexShared {
    pub table: Option<Arc<TypeTable>>,
    /// "idle", "building", "ready", "none" or "error"
    pub state: String,
    pub error: String,
    pub progress: Progress,
    pub brain: String,
    pub lab: bool,
    /// Days from `date.toordinal()` for the Neuron of the Day.
    pub today: i64,
    /// Set by the panel: the discovered set changed (a wipe), re-sync the tracker and save.
    pub wiped: bool,
    /// Set by "Try it": the card whose experiment to set up (the game does it and closes the panel).
    pub try_card: Option<Card>,
}

impl NeurodexShared {
    pub fn new(progress: Progress) -> Self {
        NeurodexShared { table: None, state: "idle".into(), error: String::new(), progress, brain: "adult".into(), lab: false, today: 739_000, wiped: false, try_card: None }
    }
}

pub type SharedDex = Arc<Mutex<NeurodexShared>>;

fn truncate(font: &Font, text: &str, width: i32) -> String {
    if font.size(text) <= width {
        return text.to_string();
    }
    let mut t = text.to_string();
    while !t.is_empty() && font.size(&format!("{t}…")) > width {
        t.pop();
    }
    format!("{t}…")
}

pub struct NeurodexPage {
    shared: SharedDex,
    region: String,
    query: String,
    only_found: bool,
    only_curated: bool,
    selected: Option<String>,
    cache_key: Option<(String, String, bool, bool, usize, String)>,
    cache_list: Vec<usize>,
    wipe_armed: f64,
    rows_key: Option<(usize, String)>,
    rows: Vec<(String, u32, u32)>,
    notd: Option<(i64, bool, Option<Card>)>,
}

impl NeurodexPage {
    pub fn new(shared: SharedDex) -> Self {
        NeurodexPage {
            shared,
            region: "(all)".into(),
            query: String::new(),
            only_found: true,
            only_curated: false,
            selected: None,
            cache_key: None,
            cache_list: vec![],
            wipe_armed: -100.0,
            rows_key: None,
            rows: vec![],
            notd: None,
        }
    }
}

impl Page for NeurodexPage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let now = m.now();
        let guard = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
        let brain = guard.brain.clone();
        m.text(cv, &tr("NEURODEX"), (rect.x + 24, rect.y + 14), INK, &f.head, Anchor::TopLeft);
        m.button(cv, Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42), &tr("Back"), Act::Back, "dex:back", Style::Primary, true, None, false, None);
        let Some(tab) = guard.table.clone() else {
            let msg = match guard.state.as_str() {
                "building" => tr("Building the Neurodex from the brain pack..."),
                "none" => tr("No brain pack is built here, so there is no Neurodex to show."),
                "error" => format!("{}: {}", tr("The Neurodex could not be built"), guard.error),
                _ => "...".into(),
            };
            m.text(cv, &msg, rect.center(), LABEL, &f.text, Anchor::Center);
            return;
        };
        let known = |n: &str| guard.progress.is_known(&brain, n);
        let (got, total) = (guard.progress.count(&brain), tab.len());
        let hdr = m.text(cv, &format!("{} / {} {}   ({brain})", group(got), group(total), tr("cell types discovered")), (rect.x + 24, rect.y + 46), LABEL, &f.small, Anchor::TopLeft);
        let chip = m.chip(cv, (hdr.right() + 14, rect.y + 46), "GAME RULE");
        m.text(
            cv,
            &truncate(&f.small, &tr("discovered = fires well above its calm rate while you play"), rect.right() - chip.right() - 24),
            (chip.right() + 8, rect.y + 46),
            DIM,
            &f.small,
            Anchor::TopLeft,
        );

        // --- left: regions with progress
        let (colx, top) = (rect.x + 24, rect.y + 76);
        let rkey = (got, brain.clone());
        if self.rows_key.as_ref() != Some(&rkey) {
            self.rows = tab.region_progress(&known);
            self.rows_key = Some(rkey);
        }
        let mut y = top;
        m.text(cv, &tr("REGIONS"), (colx, y), LABEL, &f.small, Anchor::TopLeft);
        y += 20;
        for (reg, a, b) in self.rows.clone() {
            let sel = self.region == reg;
            let r = Rect::new(colx, y, 236, 30);
            m.button(
                cv,
                r,
                "",
                Act::Custom(format!("region:{reg}")),
                &format!("dex-reg:{reg}"),
                Style::Normal,
                true,
                Some(&tr("Filter the list to this region. Regions are the coarse ones the game derives from the dataset's class and soma-neuromere annotations.")),
                sel,
                None,
            );
            let shown = if reg == "unassigned" { tr("unassigned") } else { reg.clone() };
            m.text(cv, &truncate(&f.small, &shown, 140), (r.x + 8, r.y + 3), if sel { INK } else { TEXT }, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{a}/{b}"), (r.right() - 8, r.y + 3), LABEL, &f.small, Anchor::TopRight);
            cv.fill_rect(Rect::new(r.x + 8, r.bottom() - 9, r.w - 16, 4), [10, 12, 18], 2);
            if b > 0 {
                cv.fill_rect(Rect::new(r.x + 8, r.bottom() - 9, ((r.w - 16) as f32 * a as f32 / b as f32) as i32, 4), ACCENT, 2);
            }
            y += 34;
            if y > rect.bottom() - 190 {
                break;
            }
        }
        let y = rect.bottom() - 170;
        m.toggle(cv, Rect::new(colx, y, 72, 26), self.only_found, Act::Custom("only_found".into()), "dex-found", true, &tr("Off lists every type in the region as ??? (nothing is revealed about an undiscovered type)."));
        m.text(cv, &tr("Only discovered"), (colx + 82, y + 13), TEXT, &f.small, Anchor::MidLeft);
        m.toggle(cv, Rect::new(colx, y + 32, 72, 26), self.only_curated, Act::Custom("only_curated".into()), "dex-cur", true, &tr("Only discovered types that have a literature fact and citation."));
        m.text(cv, &tr("Only with a fact"), (colx + 82, y + 45), TEXT, &f.small, Anchor::MidLeft);
        let armed = now - self.wipe_armed < 3.0;
        let wlab = if armed { tr("Click again to erase") } else { tr("Reset Neurodex") };
        let wh = if Menu::label_fits(&f.bold, &wlab, 236, 30) { 30 } else { 30 + f.bold.linesize };
        m.button(
            cv,
            Rect::new(colx, y + 66, 236, wh),
            &wlab,
            Act::Custom("wipe".into()),
            "dex-wipe",
            if armed { Style::Danger } else { Style::Normal },
            true,
            Some(&tr("Forget every discovery for this brain. Training memory is not touched.")),
            false,
            None,
        );

        // --- middle: search and list
        let (mx, mw) = (rect.x + 280, 290);
        let q = self.query.clone();
        m.text_field(cv, Rect::new(mx, top, mw, 28), &q, "dex-q", Some(&tr("Type part of a discovered type's name.")), 24, false);
        if self.query.is_empty() && m.edit_text().is_none() {
            m.text(cv, &tr("search discovered types"), (mx + 10, top + 14), DIM, &f.small, Anchor::MidLeft);
        }
        let key = (self.region.clone(), self.query.clone(), self.only_found, self.only_curated, got, brain.clone());
        if self.cache_key.as_ref() != Some(&key) {
            let cur = |n: &str| fact_for(n, &brain).is_some();
            self.cache_list = tab.list_entries(&known, Some(&self.region), &self.query, self.only_found, if self.only_curated { Some(&cur) } else { None });
            self.cache_key = Some(key);
        }
        let lst = &self.cache_list;
        let body = Rect::new(mx, top + 38, mw, rect.bottom() - top - 38 - 70);
        let skey = "neurodex";
        let off = m.scroll_offset(skey);
        m.set_content_h(skey, (lst.len() as i32 * ROW_H - body.h).max(0));
        cv.fill_rect(body, [12, 14, 20], 8);
        m.begin_clip(cv, body);
        let first = off / ROW_H;
        for k in first..(lst.len() as i32).min(first + body.h / ROW_H + 2) {
            let i = lst[k as usize];
            let name = &tab.names[i];
            let r = Rect::new(body.x + 2, body.y + k * ROW_H - off, body.w - 14, ROW_H - 2);
            let found = known(name);
            m.button(cv, r, "", Act::Custom(format!("select:{name}")), &format!("dex-row:{name}"), Style::Normal, true, None, self.selected.as_deref() == Some(name.as_str()), None);
            if found {
                m.text(cv, &truncate(&f.small, name, r.w - 50), (r.x + 8, r.y + 4), INK, &f.small, Anchor::TopLeft);
                if fact_for(name, &brain).is_some() {
                    m.text(cv, "*", (r.right() - 26, r.y + 3), AMBER, &f.small, Anchor::TopLeft);
                }
                m.text(cv, &tab.count[i].to_string(), (r.right() - 6, r.y + 4), DIM, &f.small, Anchor::TopRight);
            } else {
                m.text(cv, "???", (r.x + 8, r.y + 4), DIM, &f.small, Anchor::TopLeft);
                m.text(cv, &tab.region[i], (r.right() - 6, r.y + 4), DIM, &f.small, Anchor::TopRight);
            }
        }
        if lst.is_empty() {
            let msg = if self.only_found { tr("Nothing here yet. Play, and it fills in.") } else { tr("Nothing matches.") };
            m.text(cv, &msg, body.center(), LABEL, &f.small, Anchor::Center);
        }
        m.end_clip(cv);

        // --- right: the entry
        let (dx, dw) = (rect.x + 588, rect.w - 588 - 24);
        self.entry(m, cv, Rect::new(dx, top, dw, rect.bottom() - top - 70), &tab, &guard.progress, &brain, &f);

        // --- bottom: Neuron of the Day
        if self.notd.as_ref().is_none_or(|n| n.0 != guard.today || n.1 != guard.lab) {
            self.notd = Some((guard.today, guard.lab, notd::card(guard.today, &tab, guard.lab)));
        }
        if let Some(Some(c)) = self.notd.as_ref().map(|n| &n.2) {
            let y = rect.bottom() - 52;
            let also = if c.also() > 0 { format!(" (+{})", c.also()) } else { String::new() };
            m.text(cv, &format!("{}: {}{also}", tr("Neuron of the day"), c.first_type()), (rect.x + 24, y + 4), TEXT, &f.small, Anchor::TopLeft);
            let try_x = (rect.x + 600).min(rect.right() - 164 - 150);
            m.text(cv, &truncate(&f.small, &c.plan.text, try_x - rect.x - 40), (rect.x + 24, y + 4 + f.small.linesize), DIM, &f.small, Anchor::TopLeft);
            m.button(
                cv,
                Rect::new(try_x, y, 140, 38),
                &tr("Try it"),
                Act::Custom("try".into()),
                "dex-try",
                Style::Normal,
                c.plan.action != "none",
                Some(&format!("{} {}", c.plan.text, tr("It closes this screen so you can watch."))),
                false,
                None,
            );
        }
        let _ = ctx;
    }

    fn custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, _ctx: &mut Ctx) {
        if let Some(r) = name.strip_prefix("region:") {
            self.region = r.into();
        } else if let Some(n) = name.strip_prefix("select:") {
            self.selected = Some(n.into());
        } else if name == "only_found" {
            self.only_found = !self.only_found;
        } else if name == "only_curated" {
            self.only_curated = !self.only_curated;
        } else if name == "text:dex-q" {
            self.query = text.unwrap_or("").into();
        } else if name == "wipe" {
            let now = m.now();
            if now - self.wipe_armed < 3.0 {
                let mut g = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
                let brain = g.brain.clone();
                g.progress.wipe(Some(&brain));
                g.wiped = true;
                self.wipe_armed = -100.0;
                self.selected = None;
            } else {
                self.wipe_armed = now;
            }
        } else if name == "try" {
            let card = self.notd.as_ref().and_then(|n| n.2.clone());
            if let Some(c) = card {
                self.shared.lock().unwrap_or_else(PoisonError::into_inner).try_card = Some(c);
                m.close();
            }
        }
    }
}

impl NeurodexPage {
    #[allow(clippy::too_many_arguments)]
    fn entry(&self, m: &mut Menu, cv: &mut Canvas, bx: Rect, tab: &TypeTable, prog: &Progress, brain: &str, f: &crate::menu::Fonts) {
        let Some(name) = self.selected.as_deref().filter(|n| tab.index(n).is_some()) else {
            m.text(cv, &tr("Pick a type."), bx.center(), DIM, &f.text, Anchor::Center);
            return;
        };
        let i = tab.index(name).unwrap();
        let rec = prog.record(brain, name);
        let (x, mut y) = (bx.x, bx.y);
        if rec.is_none() {
            m.text(cv, "???", (x, y), DIM, &f.head, Anchor::TopLeft);
            m.text(cv, &tab.region[i], (x, y + 34), LABEL, &f.small, Anchor::TopLeft);
            m.wrapped(
                cv,
                &tr("Not discovered yet. Its neurons have to fire well above their calm rate while you play. Nothing else is shown until they do."),
                (x, y + 62),
                bx.w,
                DIM,
                &f.small,
                4,
            );
            return;
        }
        let rec = rec.unwrap();
        m.text(cv, name, (x, y), INK, &f.head, Anchor::TopLeft);
        y += 34;
        let how = match rec.how.as_str() {
            "stimulated" => tr("discovered by stimulating it"),
            "rest" => tr("discovered at rest (a spontaneous burst, nothing touching the fly)"),
            _ => tr("discovered in play"),
        };
        let c1 = m.chip(cv, (x, y), "GAME RULE");
        let when: String = rec.t.chars().take(10).collect();
        m.text(cv, &format!("{how}, {when}, x{:.1} its calm rate", rec.x), (c1.right() + 8, y), LABEL, &f.small, Anchor::TopLeft);
        y += 26;
        let c2 = m.chip(cv, (x, y), "CONNECTOME");
        m.text(cv, &tr("from the dataset"), (c2.right() + 8, y), LABEL, &f.small, Anchor::TopLeft);
        y += 24;
        let count = tab.count[i];
        let regs: Vec<String> = tab.regions[i].iter().take(4).map(|(r, c)| format!("{r} ({c})")).collect();
        let mut lines = vec![
            format!("{} {}   ·   {}", group(count as usize), if count == 1 { tr("neuron") } else { tr("neurons") }, if tab.superclass[i].is_empty() { "?" } else { &tab.superclass[i] }),
            format!("{}: {}", tr("regions"), if regs.is_empty() { "?".into() } else { regs.join(", ") }),
        ];
        if !tab.nt[i].is_empty() {
            let mut t = format!("{} ({}): {}", tr("transmitter"), tr("predicted"), tab.nt[i]);
            if !tab.nt_conf[i].is_nan() {
                t += &format!(", {} {:.2}", tr("confidence"), tab.nt_conf[i]);
            }
            if tab.nt_truth[i] > 0.0 {
                t += &format!(", {:.0}% {}", tab.nt_truth[i] * 100.0, tr("measured"));
            }
            lines.push(t);
        } else {
            lines.push(format!("{}: {}{}", tr("transmitter"), tr("not in this dataset"), if brain == "larva" { format!(" ({})", tr("the game assigns signs by a rule")) } else { String::new() }));
        }
        for ln in lines {
            y = m.wrapped(cv, &ln, (x, y), bx.w, TEXT, &f.small, 2) + 2;
        }
        let half = bx.w / 2 - 6;
        for (col, (label, rows_)) in [(tr("TOP INPUTS"), tab.top_partners(name, "in", 6)), (tr("TOP OUTPUTS"), tab.top_partners(name, "out", 6))].into_iter().enumerate() {
            let (cx, cy) = (x + col as i32 * (half + 12), y + 4);
            m.text(cv, &label, (cx, cy), LABEL, &f.small, Anchor::TopLeft);
            for (k, (pn, syn, share)) in rows_.iter().take(5).enumerate() {
                m.text(cv, &truncate(&f.small, pn, half - 84), (cx, cy + 18 + k as i32 * 17), TEXT, &f.small, Anchor::TopLeft);
                m.text(cv, &format!("{}  {:.0}%", group(*syn as usize), share * 100.0), (cx + half, cy + 18 + k as i32 * 17), DIM, &f.small, Anchor::TopRight);
            }
        }
        y += 4 + 18 + 5 * 17 + 6;
        m.register_label(
            Rect::new(x, y - 110, bx.w, 110),
            "dex-partner-tip",
            &tr("Strongest partner types by synapse count in the game's brain pack. The share is of this type's total input or output synapses. Connections below the loader's minimum weight, and unsigned dopamine synapses, are not in the pack."),
        );
        // skeleton: only a cached EM skeleton is ever drawn (the downloader of real shapes is a separate, opt-in feature)
        let sk = Rect::new(x, y, 150, bx.bottom() - y - 4);
        if sk.h > 60 {
            cv.fill_rect(sk, [8, 10, 14], 6);
            m.wrapped(cv, &tr("no skeleton cached for this type"), (sk.x + 8, sk.centery() - 10), sk.w - 16, DIM, &f.small, 3);
        }
        // the curated fact
        let (fx, fw) = (x + 162, bx.w - 162);
        match fact_for(name, brain) {
            Some(cur) => {
                m.chip(cv, (fx, y), "LITERATURE");
                let mut yy = m.wrapped(cv, &cur.text, (fx, y + 22), fw, TEXT, &f.small, 6);
                yy = m.wrapped(cv, &cur.cite, (fx, yy + 2), fw, LABEL, &f.small, 3);
                yy = m.wrapped(cv, &format!("doi:{}", cur.doi), (fx, yy), fw, DIM, &f.small, 1);
                if !cur.game_note.is_empty() {
                    m.wrapped(cv, &format!("{}: {}", tr("In the game"), cur.game_note), (fx, yy + 4), fw, AMBER, &f.small, 4);
                }
            }
            None => {
                m.wrapped(cv, &tr("No curated fact for this type: data only."), (fx, y), fw, DIM, &f.small, 2);
            }
        }
        let _ = tr_args;
    }
}

fn group(n: usize) -> String {
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
