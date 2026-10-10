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

// ==== Thermogenetics ===============================================================================================================
/// Host: query `lab.thermo_state` = `{expressions: [{effector, target, strength}], steady, arena_source, temperature_c, status: [{neurons,
/// activation}] | null, arena: [temps] | null, note, target, effector}`; command `lab.thermo_set {arena_source?, temperature_c?, steady?,
/// effector?, target?, add?, remove?, clear?}`.
#[derive(Default)]
pub struct LabThermo;

const EFFECTORS: [(&str, &str, bool, &str, &str); 2] = [
    (
        "trpa1",
        "TrpA1 (activates above a threshold temperature)",
        true,
        "onset near 25 C (Pulver 2009: tonic spiking at about 25 C on a 21-27 C ramp; little adaptation)",
        "full activation at 29 C, the 1 s kinetics, and the current size are game rules",
    ),
    (
        "shibire",
        "shibire-ts (silences above a threshold temperature)",
        false,
        "restrictive temperature 30 C: motionless within 2 min, walking again within about 1 min of cooling (Kitamoto 2001)",
        "the 28 C start of the ramp, the 40 s / 20 s time constants and the silencing current are game rules",
    ),
];

impl Page for LabThermo {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.thermo_state");
        let mut y = title(
            m,
            cv,
            rect,
            "THERMOGENETICS",
            "TrpA1 activates and shibire-ts silences the neurons that express them, above a threshold temperature.",
            &[
                ("LITERATURE", "onset ~25 C TrpA1 (Pulver 2009); shibire-ts 30 C (Kitamoto 2001)"),
                ("RULE", "the rest of the curve, kinetics and current size"),
                ("MODEL", "only expressing neurons respond"),
            ],
        );
        let x = rect.x + 24;
        let arena = q["arena_source"].as_bool().unwrap_or(false);
        m.text(cv, "Temperature source", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        m.segmented(
            cv,
            Rect::new(x + 200, y, 360, 28),
            &["Lab slider".into(), "Thermo arena".into()],
            usize::from(arena),
            &[Act::Custom("src:0".into()), Act::Custom("src:1".into())],
            "th_src",
            true,
            "Thermo arena: each fly senses the temperature where it stands, 15 C at the cold wall to 35 C at the hot wall (a game rule). Elsewhere the slider applies.",
        );
        y += 36;
        m.text(cv, "Temperature", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(x + 200, y, 360, 28),
            num(&q["temperature_c"]),
            (10.0, 45.0, 0.5),
            "{:.1f} C",
            "th_temp",
            !arena,
            Some("Lab slider temperature. Nothing in the model changes with temperature except the effectors you add here."),
        );
        if arena {
            let temps: Vec<f64> = q["arena"].as_array().into_iter().flatten().filter_map(Value::as_f64).collect();
            let text = if temps.is_empty() {
                "not in the thermo arena: the slider applies".to_string()
            } else {
                format!(
                    "flies sense {} (thermo arena, game rule)",
                    temps.iter().take(4).map(|t| format!("{t:.1} C")).collect::<Vec<_>>().join(", ")
                )
            };
            m.text(cv, &text, (x + 580, y + 14), AMBER, &f.small, Anchor::MidLeft);
        }
        y += 36;
        m.text(cv, "Kinetics", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        m.segmented(
            cv,
            Rect::new(x + 200, y, 360.max(2 * f.small.size("Realistic (game rule)") + 40), 28.max(f.small.linesize + 10)),
            &["Realistic (game rule)".into(), "Instant".into()],
            usize::from(q["steady"].as_bool().unwrap_or(false)),
            &[Act::Custom("kin:0".into()), Act::Custom("kin:1".into())],
            "th_kin",
            true,
            "Realistic: TrpA1 follows the temperature in about 1 s, shibire-ts takes about 40 s to block and 20 s to recover (read from Kitamoto 2001: paralysis within 2 min, recovery in about 1 min). Instant: the steady-state level.",
        );
        y += 44;
        m.text(cv, "Expression", (x, y), INK, &f.bold, Anchor::TopLeft);
        y += 22.max(f.bold.linesize + 2);
        let exprs = q["expressions"].as_array().cloned().unwrap_or_default();
        let status = q["status"].as_array().cloned().unwrap_or_default();
        for (i, e) in exprs.iter().enumerate() {
            let s = status.get(i).cloned().unwrap_or(Value::Null);
            let eff = e["effector"].as_str().unwrap_or("trpa1");
            let ef = EFFECTORS.iter().find(|x| x.0 == eff).unwrap_or(&EFFECTORS[0]);
            m.text(cv, &format!("{}  ->  {}", ef.0, e["target"].as_str().unwrap_or("")), (x, y + 12), TEXT, &f.small, Anchor::MidLeft);
            let n = s["neurons"].as_u64().unwrap_or(0);
            m.text(
                cv,
                &if n > 0 { format!("{} neurons", super::day4::grp(n as f64, 0)) } else { "(resolved when a fly is live)".to_string() },
                (x + 330, y + 12),
                LABEL,
                &f.small,
                Anchor::MidLeft,
            );
            let a = s["activation"].as_f64().unwrap_or(0.0);
            cv.fill_rect(Rect::new(x + 480, y + 4, 160, 14), [30, 36, 48], 7);
            cv.fill_rect(Rect::new(x + 480, y + 4, 3.max((160.0 * a) as i32), 14), if ef.2 { AMBER } else { [90, 150, 230] }, 7);
            m.text(cv, &format!("{:.0}%", a * 100.0), (x + 650, y + 12), TEXT, &f.small, Anchor::MidLeft);
            m.button(
                cv,
                Rect::new(x + 700, y, m.bw("Remove", 70), 24.max(f.small.linesize + 6)),
                "Remove",
                Act::Custom(format!("rm:{i}")),
                &format!("th_rm:{i}"),
                Style::Normal,
                true,
                None,
                false,
                Some(f.small.clone()),
            );
            y += 28;
        }
        if exprs.is_empty() {
            m.text(cv, "none yet", (x, y), LABEL, &f.small, Anchor::TopLeft);
            y += 22;
        }
        y += 6;
        let seg_w = 340.max(2 * (f.small.size("shibire-ts (silence)") + 28));
        m.segmented(
            cv,
            Rect::new(x, y, seg_w, 28),
            &["TrpA1 (activate)".into(), "shibire-ts (silence)".into()],
            q["effector"].as_u64().unwrap_or(0) as usize,
            &[Act::Custom("eff:0".into()), Act::Custom("eff:1".into())],
            "th_eff",
            true,
            "",
        );
        m.text_field(
            cv,
            Rect::new(x + seg_w + 16, y, 240, 28),
            q["target"].as_str().unwrap_or(""),
            "th_target",
            Some("A cell type (type:DNp01), prefix:KC, a group, or a driver line (line:SS00727). Lab > Genetic toolkit finds lines."),
            40,
            false,
        );
        let items = [
            RowButton {
                label: "Add".into(),
                click: Act::Custom("add".into()),
                id: "th_add".into(),
                style: Style::Normal,
                enabled: true,
                tip: None,
                w: 90,
            },
            RowButton {
                label: "Clear".into(),
                click: Act::Custom("clear".into()),
                id: "th_clear".into(),
                style: Style::Danger,
                enabled: true,
                tip: None,
                w: 70,
            },
        ];
        m.button_row(cv, x + seg_w + 266, y, 28, &items, 8, None);
        y += 38;
        if let Some(note) = q["note"].as_str().filter(|n| !n.is_empty()) {
            m.text(cv, note, (x, y), AMBER, &f.small, Anchor::TopLeft);
            y += 20;
        }
        if let Some(e) = exprs.first() {
            let eff = e["effector"].as_str().unwrap_or("trpa1");
            let ef = EFFECTORS.iter().find(|x| x.0 == eff).unwrap_or(&EFFECTORS[0]);
            m.wrapped(cv, &format!("{}: {}. Game rules: {}.", ef.1, ef.3, ef.4), (x, y), rect.w - 60, LABEL, &f.small, 3);
        }
        let note = "MODEL: temperature changes nothing else (no Q10). shibire-ts really blocks synaptic vesicle recycling at the terminal; here it silences the neuron, a coarser intervention. Expression is all-or-nothing in the chosen neurons.";
        let n_note = Menu::fit_lines(&f.small, note, rect.w - 220, 3).0.len() as i32;
        m.wrapped(cv, note, (x, rect.bottom() - 66 - n_note * f.small.linesize), rect.w - 220, LABEL, &f.small, 3);
        m.button(
            cv,
            Rect::new(x, rect.bottom() - 58, 300.max(f.bold.size("Open the escape-vs-temperature assay") + 32), 42),
            "Open the escape-vs-temperature assay",
            Act::Custom("assay".into()),
            "th_assay",
            Style::Normal,
            true,
            Some("Thermogenetic activation of DNp01: escape rate vs temperature, expressing flies against a no-expression control."),
            false,
            None,
        );
        back(m, cv, rect, "thermo");
    }

    fn custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let val = text.and_then(|t| t.parse::<f64>().ok());
        if let Some(i) = name.strip_prefix("src:") {
            ctx.host.command("lab.thermo_set", json!({"arena_source": i == "1"}));
        } else if let Some(i) = name.strip_prefix("kin:") {
            ctx.host.command("lab.thermo_set", json!({"steady": i == "1"}));
        } else if let Some(i) = name.strip_prefix("eff:") {
            ctx.host.command("lab.thermo_set", json!({"effector": i.parse::<u64>().unwrap_or(0)}));
        } else if let Some(i) = name.strip_prefix("rm:") {
            ctx.host.command("lab.thermo_set", json!({"remove": i.parse::<u64>().unwrap_or(0)}));
        } else if name == "slide:th_temp" {
            if let Some(v) = val {
                ctx.host.command("lab.thermo_set", json!({"temperature_c": v}));
            }
        } else if name == "text:th_target" {
            if let Some(t) = text {
                ctx.host.command("lab.thermo_set", json!({"target": t}));
            }
        } else if name == "add" {
            ctx.host.command("lab.thermo_set", json!({"add": true}));
        } else if name == "clear" {
            ctx.host.command("lab.thermo_set", json!({"clear": true}));
        } else if name == "assay" {
            if let Some(p) = m.page_mut::<super::lab::LabAssays>("lab_assays") {
                p.set_kind("thermo_escape");
            }
            m.show("lab_assays");
        }
    }
}

// ==== shared plot ===================================================================================================================
/// `_plot`: a small line plot. `series`: `(points, color, width)`; `markers`: `(x, y, color)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn plot(
    m: &Menu,
    cv: &mut Canvas,
    rect: Rect,
    series: &[(Vec<(f64, f64)>, Color, i32)],
    xlim: (f64, f64),
    ylim: (f64, f64),
    xlabel: &str,
    ylabel: &str,
    markers: &[(f64, f64, Color)],
) {
    let f = m.fonts();
    cv.fill_rect(rect, [12, 14, 20], 8);
    let pl = Rect::new(rect.x + 46, rect.y + 24, rect.w - 60, rect.h - 54);
    let (sx, sy) = (f64::from(pl.w) / (xlim.1 - xlim.0).max(1e-9), f64::from(pl.h) / (ylim.1 - ylim.0).max(1e-9));
    let px = |x: f64, y: f64| -> (f64, f64) { (f64::from(pl.x) + (x - xlim.0) * sx, f64::from(pl.bottom()) - (y - ylim.0) * sy) };
    for k in 0..5 {
        let yy = ylim.0 + (ylim.1 - ylim.0) * f64::from(k) / 4.0;
        let py = f64::from(pl.bottom()) - (yy - ylim.0) * sy;
        cv.line((pl.x, py as i32), (pl.right(), py as i32), 1, [36, 40, 50], 255);
        m.text(cv, &super::day4::fmt_gp(yy, 3), (pl.x - 6, py as i32), LABEL, &f.small, Anchor::MidRight);
    }
    let prev = cv.clip;
    cv.clip = Some(pl);
    for (pts, col, w) in series {
        if pts.len() > 1 {
            for win in pts.windows(2) {
                let (a, b) = (px(win[0].0, win[0].1), px(win[1].0, win[1].1));
                cv.line((a.0 as i32, a.1 as i32), (b.0 as i32, b.1 as i32), *w, *col, 255);
            }
        }
    }
    for (x, y, col) in markers {
        let q = px(*x, *y);
        cv.circle((q.0 as i32, q.1 as i32), 4, *col);
    }
    cv.clip = prev;
    if !xlabel.is_empty() {
        m.text(cv, xlabel, (pl.center().0, rect.bottom() - 4), LABEL, &f.small, Anchor::MidBottom);
    }
    if !ylabel.is_empty() {
        m.text(cv, ylabel, (rect.x + 6, rect.y + 2), LABEL, &f.small, Anchor::TopLeft);
    }
}

/// `labwiring._draw_inhibition`: the firing-rate distribution report.
pub fn draw_inhibition(m: &Menu, cv: &mut Canvas, area: Rect, res: &Value) {
    let f = m.fonts();
    let (b, a) = (&res["before"], &res["after"]);
    let scale = num(&res["scale"]);
    let seeds: Vec<String> = res["seeds"].as_array().into_iter().flatten().map(|s| s.to_string()).collect();
    m.text(
        cv,
        &format!(
            "Firing Rate Distribution Report  ·  Severity: {:.0}% (scale x{scale:.2})  ·  seeds [{}]  ·  {:.1}s",
            num(&res["severity"]) * 100.0,
            seeds.join(", "),
            num(&res["seconds"])
        ),
        (area.x, area.y),
        INK,
        &f.small,
        Anchor::TopLeft,
    );
    let mut y = area.y + 20;
    let (bm, am) = (num(&b["mean"]), num(&a["mean"]));
    m.text(
        cv,
        &format!(
            "Mean rate: {bm:.1} Hz -> {am:.1} Hz ({:.1}x)   Median: {:.1} Hz -> {:.1} Hz   95th percentile: {:.1} Hz -> {:.1} Hz   Max: {:.1} Hz -> {:.1} Hz",
            am / bm.max(0.01),
            num(&b["median"]),
            num(&a["median"]),
            num(&b["p95"]),
            num(&a["p95"]),
            num(&b["max"]),
            num(&a["max"])
        ),
        (area.x, y),
        if am > bm * 2.0 { BAD } else { GOOD },
        &f.small,
        Anchor::TopLeft,
    );
    y += 24;
    let xs = [area.x, area.x + 130, area.x + 310, area.x + 500];
    for (hx, label) in xs.iter().zip([
        "Rate bin".to_string(),
        "Control (100% inhib)".to_string(),
        format!("Treated (x{scale:.2} inhib)"),
        "Shift".to_string(),
    ]) {
        m.text(cv, &label, (*hx, y), LABEL, &f.small, Anchor::TopLeft);
    }
    y += 18;
    let sg = |s: f64| format!("{:.1}%", s * 100.0);
    for i in 0..6 {
        let lbl = b["bin_labels"][i].as_str().unwrap_or("");
        let (bc, bs, ac, as_) = (
            b["histogram"][i].as_i64().unwrap_or(0),
            num(&b["bin_shares"][i]),
            a["histogram"][i].as_i64().unwrap_or(0),
            num(&a["bin_shares"][i]),
        );
        m.text(cv, lbl, (xs[0], y), TEXT, &f.small, Anchor::TopLeft);
        m.text(cv, &format!("{} ({})", super::day4::grp(bc as f64, 0), sg(bs)), (xs[1], y), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, &format!("{} ({})", super::day4::grp(ac as f64, 0), sg(as_)), (xs[2], y), INK, &f.small, Anchor::TopLeft);
        let diff = as_ - bs;
        let col = if diff > 0.0 && (lbl.contains('>') || lbl.contains("40") || lbl.contains("20")) {
            BAD
        } else if diff < 0.0 {
            GOOD
        } else {
            TEXT
        };
        m.text(cv, &format!("{:+.1}%", diff * 100.0), (xs[3], y), col, &f.small, Anchor::TopLeft);
        y += 18;
    }
    y += 6;
    m.wrapped(
        cv,
        "Runaway activity is an observed emergent outcome of uninhibited recurrent excitation across 166.7k neurons, not a scripted seizure.",
        (area.x, y),
        area.w,
        LABEL,
        &f.small,
        3,
    );
}

// ==== Pharmacology ===================================================================================================================
/// Host: query `lab.pharm_state` = `{drug, dose, low, cut, affected, applied, has_nt, live_label, busy, dr, dr_running, report, report_note,
/// report_running, error}`; commands `lab.pharm_set {drug?, dose?, low?, cut?}`, `lab.pharm_apply`, `lab.pharm_wash`, `lab.pharm_dr`,
/// `lab.pharm_report`.
#[derive(Default)]
pub struct LabPharm;

const DRUG_LABELS: [(&str, &str); 4] = [
    ("picrotoxin", "Picrotoxin"),
    ("cholinergic", "Cholinergic block"),
    ("glucl", "Glutamate-Cl block"),
    ("gabaa_agonist", "GABA-A agonist"),
];

impl Page for LabPharm {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.pharm_state");
        let mut y = title(
            m,
            cv,
            rect,
            "PHARMACOLOGY",
            "Drugs scale synapses by the dataset's predicted transmitter. Picrotoxin moved here from Robustness.",
            &[
                ("MODEL", "PREDICTION: a synaptic scale only, no receptor or kinetic model"),
                ("CONNECTOME", "which synapses: transmitter predictions and their confidence"),
            ],
        );
        let x = rect.x + 24;
        let drug = q["drug"].as_str().unwrap_or("picrotoxin").to_string();
        let dose = num(&q["dose"]);
        let low = q["low"].as_bool().unwrap_or(true);
        let fh = 30.max(f.small.linesize + 10);
        let lw = ["Drug", "Dose", "Low-confidence predictions"].iter().map(|t| f.text.size(t)).max().unwrap_or(0) + 16;
        let avail = rect.right() - 24 - (x + lw);
        m.text(cv, "Drug", (x, y + fh / 2), TEXT, &f.text, Anchor::MidLeft);
        let items: Vec<_> = DRUG_LABELS
            .iter()
            .map(|(k, l)| {
                (
                    l.to_string(),
                    Act::Custom(format!("drug:{k}")),
                    format!("ph_drug:{k}"),
                    drug == *k,
                    Some("Octopamine and dopamine modulation are left out: those neurons' synapses are not in the simulated matrix and there is no existing gain to scale honestly.".to_string()),
                )
            })
            .collect();
        y = m.flow_buttons(cv, x + lw, y, rect.right() - 24, &items, Some(fh), None, 8, None) + 10;
        m.text(cv, "Dose", (x, y + fh / 2), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(x + lw, y, 320.min(avail), fh),
            dose,
            (0.0, 1.0, 0.05),
            "{:.0%}",
            "ph_dose",
            true,
            Some("Blockers scale by (1 - dose). The GABA-A agonist scales by 1 + dose (x2 at full dose, a game rule)."),
        );
        let scale = if drug == "gabaa_agonist" { 1.0 + dose } else { 1.0 - dose };
        let mut sx = x + lw + 320.min(avail) + 20;
        let stext = format!("synapse weight scale x{scale:.2}");
        if sx + f.text.size(&stext) > rect.right() - 24 {
            y += fh + 4;
            sx = x + lw;
        }
        m.text(cv, &stext, (sx, y + fh / 2), INK, &f.text, Anchor::MidLeft);
        y += fh + 8;
        m.text(cv, "Low-confidence predictions", (x, y + fh / 2), TEXT, &f.text, Anchor::MidLeft);
        let sw = 240.min(avail / 2);
        m.segmented(
            cv,
            Rect::new(x + lw, y, sw, fh),
            &["Include".into(), "Exclude".into()],
            usize::from(!low),
            &[Act::Custom("low:1".into()), Act::Custom("low:0".into())],
            "ph_low",
            true,
            "Exclude leaves the drug off neurons whose transmitter is a prediction below the cut (and those with no confidence); measured transmitters always count.",
        );
        let mut cx = x + lw + sw + 16;
        m.text(cv, "cut", (cx, y + fh / 2), TEXT, &f.text, Anchor::MidLeft);
        cx += f.text.size("cut") + 10;
        m.page_slider(
            cv,
            Rect::new(cx, y, 160.max(200.min(rect.right() - 24 - cx)), fh),
            num(&q["cut"]),
            (0.5, 0.95, 0.05),
            "{:.2f}",
            "ph_cut",
            !low,
            Some("A predicted transmitter below this confidence is 'low confidence' (a game rule)."),
        );
        y += fh + 12;
        let a = &q["affected"];
        if !a.is_object() {
            let e = q["error"].as_str().unwrap_or("the brain pack is not loaded");
            m.wrapped(cv, &format!("needs an adult brain pack with transmitter predictions: {e}"), (x, y), rect.w - 48, BAD, &f.small, 3);
            back(m, cv, rect, "pharm");
            return;
        }
        let rows = a["rows"].as_array().cloned().unwrap_or_default();
        let g = |v: &Value| super::day4::grp(num(v), 0);
        let heads = ["confidence level", "neurons", "connections", "synapses", "scaled?"];
        let yes_no: Vec<String> = ["yes", "no", "-"].iter().map(|s| s.to_string()).collect();
        let cols: Vec<Vec<String>> = vec![
            rows.iter().map(|r| r["level"].as_str().unwrap_or("").to_string()).collect(),
            rows.iter().map(|r| g(&r["neurons"])).collect(),
            rows.iter().map(|r| g(&r["connections"])).collect(),
            rows.iter().map(|r| g(&r["synapses"])).collect(),
            yes_no,
        ];
        let mut xs = Vec::new();
        let mut cxx = x;
        for (h, vals) in heads.iter().zip(&cols) {
            xs.push(cxx);
            cxx += std::iter::once(&h.to_string()).chain(vals).map(|v| f.small.size(v)).max().unwrap_or(0) + 28;
        }
        let rh = f.small.linesize;
        for (hx, label) in xs.iter().zip(heads) {
            m.text(cv, label, (*hx, y), LABEL, &f.small, Anchor::TopLeft);
        }
        y += rh + 2;
        for r in &rows {
            let applied = r["applied"].as_bool().unwrap_or(false);
            let col = if applied { TEXT } else { LABEL };
            m.text(cv, r["level"].as_str().unwrap_or(""), (xs[0], y), col, &f.small, Anchor::TopLeft);
            m.text(cv, &g(&r["neurons"]), (xs[1], y), col, &f.small, Anchor::TopLeft);
            m.text(cv, &g(&r["connections"]), (xs[2], y), col, &f.small, Anchor::TopLeft);
            m.text(cv, &g(&r["synapses"]), (xs[3], y), col, &f.small, Anchor::TopLeft);
            let sa = num(&r["synapses_applied"]) > 0.0;
            let word = if num(&r["synapses"]) == 0.0 {
                "-"
            } else if sa {
                "yes"
            } else {
                "no"
            };
            m.text(cv, word, (xs[4], y), if sa { GOOD } else { LABEL }, &f.small, Anchor::TopLeft);
            y += rh;
        }
        y = m.wrapped(
            cv,
            &format!(
                "affected: {} of {} synapses  ·  {} connections  ·  {} presynaptic neurons",
                g(&a["synapses_applied"]),
                g(&a["synapses"]),
                g(&a["connections_applied"]),
                g(&a["neurons_applied"])
            ),
            (x, y + 4),
            rect.w - 48,
            INK,
            &f.small,
            2,
        ) - 16;
        y += 28;
        let applied = q["applied"].as_bool().unwrap_or(false);
        let busy_dr = q["dr_running"].as_bool().unwrap_or(false);
        let wbusy = !q["busy"].as_str().unwrap_or("").is_empty();
        let apply_label = if applied { "Applied" } else { "Apply to every fly" };
        let x_dr = x + m.bw(apply_label, 250) + 10 + m.bw("Wash out", 200) + 10;
        let items = [
            RowButton {
                label: apply_label.into(),
                click: Act::Custom("apply".into()),
                id: "ph_apply".into(),
                style: Style::Primary,
                enabled: !applied && !wbusy,
                tip: Some("Reversibly scales those synapses on the live flies (the wiring machinery of Lab > Robustness).".into()),
                w: 250,
            },
            RowButton {
                label: "Wash out".into(),
                click: Act::Custom("wash".into()),
                id: "ph_wash".into(),
                style: Style::Normal,
                enabled: q["has_nt"].as_bool().unwrap_or(false) && !wbusy,
                tip: None,
                w: 200,
            },
            RowButton {
                label: if busy_dr { "measuring…".into() } else { "Measure dose-response".into() },
                click: Act::Custom("dr".into()),
                id: "ph_dr".into(),
                style: Style::Normal,
                enabled: !busy_dr,
                tip: Some(
                    "Whole-brain firing (Hz) at 0, 25, 50, 75 and 100% dose of this drug, seed 1000, 1 s each, on fresh brains.".into(),
                ),
                w: 250,
            },
        ];
        let (x_after, _) = m.button_row(cv, x, y, 36, &items, 10, None);
        if q["has_nt"].as_bool().unwrap_or(false) {
            m.wrapped(
                cv,
                &format!("live on the flies: {}", q["live_label"].as_str().unwrap_or("")),
                (x_after, y + 4),
                rect.right() - 24 - x_after,
                AMBER,
                &f.small,
                2,
            );
        }
        y += 46;
        if drug == "picrotoxin" && low {
            if !q["report_running"].as_bool().unwrap_or(false) {
                let label = "Firing-rate distribution report";
                m.button(
                    cv,
                    Rect::new(x_dr, y - 46 + 42, 250.max(f.small.size(label) + 28), 26.max(f.small.linesize + 6)),
                    label,
                    Act::Custom("report".into()),
                    "ph_report",
                    Style::Normal,
                    true,
                    Some("The original picrotoxin report: firing-rate histogram before and after, seed 1000."),
                    false,
                    Some(f.small.clone()),
                );
            } else {
                m.text(
                    cv,
                    &format!("report: {}", q["report_note"].as_str().unwrap_or("")),
                    (x_dr, y - 46 + 50),
                    LABEL,
                    &f.small,
                    Anchor::TopLeft,
                );
            }
            if q["report"].is_object() {
                draw_inhibition(m, cv, Rect::new(x, y + 6, rect.w - 48, rect.bottom() - y - 80), &q["report"]);
                back(m, cv, rect, "pharm");
                return;
            }
        }
        if let Some(dr) = q["dr"].as_object().filter(|_| q["dr"].is_object()) {
            let rows = dr["rows"].as_array().cloned().unwrap_or_default();
            let pts: Vec<(f64, f64)> = rows.iter().map(|r| (num(&r["dose"]), num(&r["mean_hz"]))).collect();
            let top = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            let rect_p = Rect::new(x, y, 420, 120.max(rect.bottom() - y - 80));
            let markers: Vec<(f64, f64, Color)> = pts.iter().map(|p| (p.0, p.1, [255, 160, 80])).collect();
            plot(
                m,
                cv,
                rect_p,
                &[(pts.clone(), [255, 160, 80], 2)],
                (0.0, 1.0),
                (0.0, 1.0f64.max(top * 1.1)),
                "dose",
                "mean firing (Hz)",
                &markers,
            );
            let label = DRUG_LABELS.iter().find(|d| Some(d.0) == dr["drug"].as_str()).map_or("", |d| d.1);
            m.wrapped(
                cv,
                &format!(
                    "{label}: mean whole-brain firing over {:.1} s, seed [{}]. The simulator's slow global gain pushes back on anything that changes overall synaptic strength, so this is the drug plus a little of that compensation.",
                    num(&dr["steps"]) * 0.005,
                    dr["seeds"].as_array().into_iter().flatten().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")
                ),
                (x + 440, y + 6),
                rect.w - 500,
                LABEL,
                &f.small,
                6,
            );
        }
        back(m, cv, rect, "pharm");
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let val = text.and_then(|t| t.parse::<f64>().ok());
        if let Some(k) = name.strip_prefix("drug:") {
            ctx.host.command("lab.pharm_set", json!({"drug": k}));
        } else if let Some(k) = name.strip_prefix("low:") {
            ctx.host.command("lab.pharm_set", json!({"low": k == "1"}));
        } else if name == "slide:ph_dose" {
            if let Some(v) = val {
                ctx.host.command("lab.pharm_set", json!({"dose": v}));
            }
        } else if name == "slide:ph_cut" {
            if let Some(v) = val {
                ctx.host.command("lab.pharm_set", json!({"cut": v}));
            }
        } else {
            match name {
                "apply" => drop(ctx.host.command("lab.pharm_apply", Value::Null)),
                "wash" => drop(ctx.host.command("lab.pharm_wash", Value::Null)),
                "dr" => drop(ctx.host.command("lab.pharm_dr", Value::Null)),
                "report" => drop(ctx.host.command("lab.pharm_report", Value::Null)),
                _ => {}
            }
        }
    }
}
