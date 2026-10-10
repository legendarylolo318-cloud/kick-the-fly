//! The 3.0 day 2 toolkit pages of `lab/labtoolkit.py`: the shared heading and the genetic toolkit.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, RowButton, Style};
use crate::theme::*;

/// `_title`: heading, a description and the tag row (chips wrap rather than leave the panel). Returns the y below.
pub fn title(m: &mut Menu, cv: &mut Canvas, rect: Rect, title: &str, sub: &str, tags: &[(&str, &str)]) -> i32 {
    let f = m.fonts();
    m.text(cv, title, (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
    let mut y = m.wrapped(cv, sub, (rect.x + 24, m.under_heading(rect).max(rect.y + 48)), rect.w - 60, LABEL, &f.small, 2) + 4;
    let mut x = rect.x + 24;
    let limit = rect.right() - 24;
    let step = f.small.linesize + 6;
    for (chip, text) in tags {
        let w_chip = f.small.size(&chip.to_uppercase()) + 12;
        let w_text = f.small.size(text) + 6;
        if x + w_chip + w_text > limit && x > rect.x + 24 {
            x = rect.x + 24;
            y += step;
        }
        let r = m.chip(cv, (x, y), chip);
        if r.right() + 6 + w_text > limit {
            let yb = m.wrapped(cv, text, (r.right() + 6, y), limit - r.right() - 6, LABEL, &f.small, 3);
            x = rect.x + 24;
            y = (y + step).max(yb + 4);
            continue;
        }
        let t = m.text(cv, text, (r.right() + 6, y), LABEL, &f.small, Anchor::TopLeft);
        x = t.right() + 18;
    }
    y + step + 6
}

pub fn back(m: &mut Menu, cv: &mut Canvas, rect: Rect, key: &str) {
    m.button(
        cv,
        Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
        "Back",
        Act::Back,
        &format!("{key}:back"),
        Style::Primary,
        true,
        None,
        false,
        None,
    );
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

// ==== Genetic toolkit ============================================================================================================
/// Host: query `lab.genetics_state` = `{error, coverage, query, results: [{name, quality, cell_types}], selected, describe, type_ops:
/// {type: mode}, note}`; commands `lab.gen_search {q}`, `lab.gen_pick {line}`, `lab.gen_surgery {line, mode}`, `lab.gen_laser {line}`,
/// `lab.gen_thermo {effector, line}`.
#[derive(Default)]
pub struct LabGenetics;

impl Page for LabGenetics {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.genetics_state");
        let mut y = title(
            m,
            cv,
            rect,
            "GENETIC TOOLKIT",
            "Choose neurons by split-GAL4 driver line instead of by cell type. Surgery, the laser and thermogenetics accept a line.",
            &[
                ("LITERATURE", "line -> cell types: Meissner et al. 2025, eLife (CC BY 4.0)"),
                ("CONNECTOME", "neuron counts"),
                ("RULE", "a cell-type name matches only if spelled identically"),
            ],
        );
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.wrapped(
                cv,
                &format!("needs the adult brain pack and the driver-line table: {e}"),
                (rect.x + 24, y + 10),
                rect.w - 48,
                BAD,
                &f.small,
                4,
            );
            back(m, cv, rect, "genetics");
            return;
        }
        let cov = &q["coverage"];
        y = m.wrapped(
            cv,
            &format!(
                "{} adult split-GAL4 lines; {} name at least one cell type spelled like this connectome's. Not included: GAL4 (non-split) lines, for which no redistributable mapping was found.",
                super::day4::grp(num(&cov["lines"]), 0),
                super::day4::grp(num(&cov["lines_with_a_match"]), 0)
            ),
            (rect.x + 24, y),
            rect.w - 60,
            LABEL,
            &f.small,
            2,
        ) + 6;
        m.text(cv, "Search (line or cell type)", (rect.x + 24, y + 12), TEXT, &f.text, Anchor::MidLeft);
        let query = q["query"].as_str().unwrap_or("");
        m.text_field(
            cv,
            Rect::new(rect.x + 260, y, 240, 28),
            query,
            "gen_query",
            Some("A line name such as SS00727, or a cell type such as DNp01. Enter to search."),
            24,
            false,
        );
        m.button(
            cv,
            Rect::new(rect.x + 510, y, 90, 28),
            "Search",
            Act::Custom("search".into()),
            "gen_go",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
        y += 40;
        let left = Rect::new(rect.x + 24, y, 430, rect.bottom() - y - 70);
        cv.fill_rect(left, [20, 23, 31], 8);
        let selected = q["selected"].as_str();
        let mut yy = left.y + 6;
        let results = q["results"].as_array().cloned().unwrap_or_default();
        for ln in results.iter().take(16) {
            let r = Rect::new(left.x + 4, yy, left.w - 8, 24);
            let name = ln["name"].as_str().unwrap_or("");
            let types: Vec<&str> = ln["cell_types"].as_array().into_iter().flatten().filter_map(|t| t.as_str()).collect();
            let joined: String = types.join(", ").chars().take(40).collect();
            m.button(
                cv,
                r,
                &format!("{name}   q{}   {joined}", ln["quality"].as_str().unwrap_or("")),
                Act::Custom(format!("pick:{name}")),
                &format!("gen_pick:{name}"),
                if selected == Some(name) { Style::Primary } else { Style::Normal },
                true,
                None,
                false,
                Some(f.small.clone()),
            );
            yy += 26;
        }
        if results.is_empty() {
            m.text(
                cv,
                if query.is_empty() { "type something to search" } else { "no lines found" },
                (left.x + 10, left.y + 10),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
        }
        let x2 = left.right() + 20;
        let w2 = rect.right() - x2 - 24;
        let d = &q["describe"];
        if d.is_object() {
            let mut yy = y;
            m.text(cv, d["line"].as_str().unwrap_or(""), (x2, yy), INK, &f.head, Anchor::TopLeft);
            yy += 34;
            m.text(
                cv,
                &format!("line  ->  cell types  ->  neurons in this connectome: {}", super::day4::grp(num(&d["neurons"]), 0)),
                (x2, yy),
                TEXT,
                &f.bold,
                Anchor::TopLeft,
            );
            yy += 24;
            for t in d["cell_types"].as_array().into_iter().flatten().filter_map(|t| t.as_str()) {
                let n = d["matched"][t].as_u64();
                m.text(cv, t, (x2 + 8, yy), if n.is_some() { INK } else { LABEL }, &f.small, Anchor::TopLeft);
                m.text(
                    cv,
                    &n.map_or("no identically spelled type in this connectome (not guessed)".to_string(), |n| {
                        format!("{} neurons", super::day4::grp(n as f64, 0))
                    }),
                    (x2 + 200, yy),
                    if n.is_some() { GOOD } else { AMBER },
                    &f.small,
                    Anchor::TopLeft,
                );
                yy += 18;
            }
            yy += 6;
            let col = match d["off_target"].as_str() {
                Some("minimal") => GOOD,
                Some("some") | Some("weak") => AMBER,
                Some("yes") | Some("unstable") => BAD,
                _ => LABEL,
            };
            m.text(cv, "Off-target expression (the source's own score)", (x2, yy), TEXT, &f.small, Anchor::TopLeft);
            yy = m.wrapped(cv, d["off_target_text"].as_str().unwrap_or(""), (x2, yy + 16), w2, col, &f.small, 4) + 4;
            let sd = d["sex_difference"].as_str().filter(|s| !s.is_empty()).unwrap_or("n/a");
            m.text(
                cv,
                &format!(
                    "{}  ·  doi:{}  ·  sex difference: {sd}",
                    d["cite"].as_str().unwrap_or("").replace(';', ""),
                    d["doi"].as_str().unwrap_or("")
                ),
                (x2, yy),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            yy += 28;
            let ok = num(&d["neurons"]) > 0.0;
            let matched: Vec<&String> = d["matched"].as_object().into_iter().flat_map(|o| o.keys()).collect();
            let cur: Vec<i64> = matched.iter().map(|t| q["type_ops"][t.as_str()].as_i64().unwrap_or(0)).collect();
            let state = if !cur.is_empty() && cur.iter().all(|&v| v < 0) {
                "silenced"
            } else if !cur.is_empty() && cur.iter().all(|&v| v > 0) {
                "stimulated"
            } else {
                "normal"
            };
            m.text(cv, &format!("surgery on these types: {state}"), (x2, yy), LABEL, &f.small, Anchor::TopLeft);
            yy += 22;
            for (k, (label, mode)) in [("Silence", -1), ("Stimulate", 1), ("Clear", 0)].into_iter().enumerate() {
                m.button(
                    cv,
                    Rect::new(x2 + k as i32 * 110, yy, 102, 30),
                    label,
                    Act::Custom(format!("surgery:{mode}")),
                    &format!("gen_op:{mode}"),
                    Style::Normal,
                    ok,
                    Some("Brain surgery on the cell types this line labels (the same switches as the inspector)."),
                    false,
                    None,
                );
            }
            yy += 38;
            let items = [
                RowButton {
                    label: "Aim the laser at this line".into(),
                    click: Act::Custom("laser".into()),
                    id: "gen_laser".into(),
                    style: Style::Normal,
                    enabled: ok,
                    tip: Some("Sets the Lab laser's target to this line (activate mode) and puts the laser in your hand.".into()),
                    w: 210,
                },
                RowButton {
                    label: "Express TrpA1 here".into(),
                    click: Act::Custom("thermo:trpa1".into()),
                    id: "gen_trp".into(),
                    style: Style::Normal,
                    enabled: ok,
                    tip: Some("Adds a thermogenetic expression: activates these neurons above the TrpA1 temperature.".into()),
                    w: 210,
                },
                RowButton {
                    label: "Express shibire-ts here".into(),
                    click: Act::Custom("thermo:shibire".into()),
                    id: "gen_shi".into(),
                    style: Style::Normal,
                    enabled: ok,
                    tip: Some("Adds a thermogenetic expression: silences these neurons above the shibire-ts temperature.".into()),
                    w: 210,
                },
            ];
            let (_, y2) = m.button_row(cv, x2, yy, 30, &items, 10, Some(rect.right() - 24));
            yy = y2 - 30 + 40;
            if let Some(note) = q["note"].as_str().filter(|n| !n.is_empty()) {
                m.text(cv, note, (x2, yy), AMBER, &f.small, Anchor::TopLeft);
            }
        }
        m.wrapped(
            cv,
            "What this does not know: expression strength, what a line labels beyond its authors' score, developmental timing. Nothing here is a measurement on this connectome.",
            (rect.x + 24, rect.bottom() - 62),
            rect.w - 220,
            LABEL,
            &f.small,
            2,
        );
        back(m, cv, rect, "genetics");
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let sel = ctx.host.query("lab.genetics_state")["selected"].as_str().map(String::from);
        if name == "text:gen_query" || name == "search" {
            let q =
                text.map(String::from).unwrap_or_else(|| ctx.host.query("lab.genetics_state")["query"].as_str().unwrap_or("").to_string());
            ctx.host.command("lab.gen_search", json!({"q": q}));
        } else if let Some(l) = name.strip_prefix("pick:") {
            ctx.host.command("lab.gen_pick", json!({"line": l}));
        } else if let (Some(mode), Some(line)) = (name.strip_prefix("surgery:"), sel.as_ref()) {
            ctx.host.command("lab.gen_surgery", json!({"line": line, "mode": mode.parse::<i64>().unwrap_or(0)}));
        } else if let (true, Some(line)) = (name == "laser", sel.as_ref()) {
            ctx.host.command("lab.gen_laser", json!({"line": line}));
        } else if let (Some(eff), Some(line)) = (name.strip_prefix("thermo:"), sel.as_ref()) {
            ctx.host.command("lab.gen_thermo", json!({"effector": eff, "line": line}));
        }
    }
}
