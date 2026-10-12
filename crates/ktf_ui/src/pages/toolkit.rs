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
    hlines: &[(f64, Color, &str)],
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
    for (y, col, label) in hlines {
        let py = f64::from(pl.bottom()) - (y - ylim.0) * sy;
        if f64::from(pl.y) <= py && py <= f64::from(pl.bottom()) {
            cv.line((pl.x, py as i32), (pl.right(), py as i32), 1, *col, 255);
            m.text(cv, label, (pl.right() - 4, py as i32 - 2), *col, &f.small, Anchor::BottomRight);
        }
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
                &[],
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

// ==== Patch clamp ====================================================================================================================
/// Host: query `lab.patch_state` = `{type, index, mode, amp, dur, repeats, hold, live, row, desc, error, window: {v, spikes, current},
/// recording: {v, spikes, current} | null, curve, job, note, nwb}`; commands `lab.patch_set {...}`, `lab.patch_pick`, `lab.patch_steps {k}`,
/// `lab.patch_run`, `lab.patch_if`, `lab.patch_export {kind}`.
#[derive(Default)]
pub struct LabPatch {
    last_t: f64,
}

fn series_f32(v: &Value) -> Vec<f64> {
    v.as_array().into_iter().flatten().map(num).collect()
}

fn spike_idx(v: &Value) -> Vec<usize> {
    v.as_array().into_iter().flatten().enumerate().filter(|(_, b)| b.as_bool().unwrap_or(false)).map(|(i, _)| i).collect()
}

impl Page for LabPatch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.patch_state");
        let mut y = title(
            m,
            cv,
            rect,
            "VIRTUAL PATCH CLAMP",
            "Current-clamp one simulated neuron: membrane potential, spikes, threshold, current steps and an I-F curve.",
            &[
                ("MODEL", "a point-neuron LIF unit, NOT real electrophysiology; potential in model units, threshold = 1.0"),
                ("CONNECTOME", "embedded mode: the real synaptic input around the neuron"),
            ],
        );
        let x = rect.x + 24;
        m.text(cv, "Neuron", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        m.text_field(
            cv,
            Rect::new(x + 70, y, 170, 28),
            q["type"].as_str().unwrap_or(""),
            "pc_type",
            Some("A cell type (DNp01), or any neuron spec: prefix:KC, line:SS00727."),
            40,
            false,
        );
        m.text(cv, "#", (x + 250, y + 14), LABEL, &f.text, Anchor::MidLeft);
        m.text_field(
            cv,
            Rect::new(x + 268, y, 54, 28),
            &q["index"].as_i64().unwrap_or(0).to_string(),
            "pc_idx",
            Some("Which neuron of that type, in connectome row order."),
            6,
            false,
        );
        m.button(
            cv,
            Rect::new(x + 332, y, 110, 28),
            "Patch it",
            Act::Custom("pick".into()),
            "pc_pick",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
        let sw = 330.max(2 * f.small.size("Embedded (live brain)") + 40);
        let mx = if x + 680 + sw > rect.right() - 24 {
            y += 36;
            x
        } else {
            x + 680
        };
        m.segmented(
            cv,
            Rect::new(mx, y, sw, 28.max(f.small.linesize + 10)),
            &["Embedded (live brain)".into(), "Isolated unit".into()],
            q["mode"].as_u64().unwrap_or(0) as usize,
            &[Act::Custom("mode:0".into()), Act::Custom("mode:1".into())],
            "pc_mode",
            true,
            "Embedded: the neuron keeps all its synaptic input while you inject current. Isolated: the same LIF equation with no synaptic input; every neuron gives the same curve because every neuron is the same unit.",
        );
        y += 38;
        if q["row"].is_null() {
            m.text(
                cv,
                "Choose a neuron above, or click a neuron in the big brain view (B) and press PATCH.",
                (x, y),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
                m.text(cv, e, (x + 400, y + 90), AMBER, &f.small, Anchor::TopLeft);
            }
            back(m, cv, rect, "patch");
            return;
        }
        let d = &q["desc"];
        m.text(
            cv,
            &format!(
                "row {}  ·  {}  ·  {}  ·  body id {}  ·  {}",
                d["row"],
                d["type"].as_str().unwrap_or(""),
                d["instance"].as_str().unwrap_or(""),
                if d["body_id"].is_null() { "None".to_string() } else { d["body_id"].to_string() },
                d["superclass"].as_str().unwrap_or("")
            ),
            (x, y),
            INK,
            &f.small,
            Anchor::TopLeft,
        );
        y += 22;
        let pl = Rect::new(x, y, rect.w - 48, 190);
        let live = q["live"].as_bool().unwrap_or(false);
        let dt_s = 5.0 / 1000.0;
        let hl: [(f64, Color, &str); 1] = [(1.0, AMBER, "threshold 1.0 (model)")];
        if live {
            let w = &q["window"];
            let v = series_f32(&w["v"]);
            if !v.is_empty() {
                let len = v.len() as f64;
                let tt: Vec<f64> = (0..v.len()).map(|i| (i as f64 - len) * dt_s).collect();
                let vmax = 1.3f64.max(v.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.05);
                let vmin = (-0.1f64).min(v.iter().copied().fold(f64::INFINITY, f64::min) * 1.1);
                let pts: Vec<(f64, f64)> = tt.iter().copied().zip(v.iter().copied()).collect();
                let marks: Vec<(f64, f64, Color)> = spike_idx(&w["spikes"]).iter().map(|&i| (tt[i], 1.15, [255, 210, 90])).collect();
                plot(
                    m,
                    cv,
                    pl,
                    &[(pts, [120, 210, 255], 2)],
                    (-600.0 * dt_s, 0.0),
                    (vmin, vmax),
                    "seconds",
                    "membrane potential (model units)",
                    &hl,
                    &marks,
                );
            } else {
                cv.fill_rect(pl, [12, 14, 20], 8);
            }
        } else if q["recording"].is_object() {
            let rec = &q["recording"];
            let v = series_f32(&rec["v"]);
            let cur = series_f32(&rec["current"]);
            let tt: Vec<f64> = (0..v.len()).map(|i| i as f64 * 5.0 / 1000.0).collect();
            let cmax = cur.iter().map(|c| c.abs()).fold(0.0, f64::max).max(1e-6);
            let vmin = (-0.1f64).min(v.iter().copied().fold(f64::INFINITY, f64::min) * 1.1);
            let pts: Vec<(f64, f64)> = tt.iter().copied().zip(v.iter().copied()).collect();
            let cpts: Vec<(f64, f64)> = tt.iter().copied().zip(cur.iter().map(|c| c / cmax * 0.4 + 1.6)).collect();
            let marks: Vec<(f64, f64, Color)> = spike_idx(&rec["spikes"]).iter().map(|&i| (tt[i], 1.15, [255, 210, 90])).collect();
            plot(
                m,
                cv,
                pl,
                &[(pts, [120, 210, 255], 2), (cpts, [255, 160, 80], 1)],
                (0.0, tt.last().copied().unwrap_or(1.0)),
                (vmin, 2.1),
                "seconds",
                "membrane potential (model units); orange: injected current (scaled)",
                &hl,
                &marks,
            );
        } else {
            cv.fill_rect(pl, [12, 14, 20], 8);
            m.text(cv, "press Live to watch the potential, or Run steps", pl.center(), LABEL, &f.small, Anchor::Center);
        }
        // the menu pauses the game: ask the brain for real-time steps while the electrode is on
        if live {
            let now = ctx.now;
            if self.last_t > 0.0 && now > self.last_t {
                let k = ((0.1f64).min(now - self.last_t) / 0.005) as u32;
                if k > 0 {
                    ctx.host.command("lab.patch_steps", json!({"k": k}));
                }
            }
            self.last_t = now;
        } else {
            self.last_t = 0.0;
        }
        y = pl.bottom() + 10;
        m.text(cv, "Live", (x, y + 13), TEXT, &f.text, Anchor::MidLeft);
        m.toggle(
            cv,
            Rect::new(x + 40, y, 50, 26),
            live,
            Act::Custom(format!("live:{}", u8::from(!live))),
            "pc_live",
            true,
            "Watch this neuron live while the brain runs in real time.",
        );
        m.text(cv, "hold", (x + 130, y + 13), TEXT, &f.small, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(x + 160, y, 150, 26),
            num(&q["hold"]),
            (-0.5, 0.5, 0.01),
            "{:.2f}",
            "pc_hold",
            true,
            Some("Constant holding current (the simulator's current units; 0.5 is the validation suite's activation current)."),
        );
        m.text(cv, "step", (x + 330, y + 13), TEXT, &f.small, Anchor::MidLeft);
        m.page_slider(cv, Rect::new(x + 370, y, 150, 26), num(&q["amp"]), (-0.5, 1.0, 0.01), "{:.2f}", "pc_amp", true, None);
        m.text(cv, "ms", (x + 530, y + 13), TEXT, &f.small, Anchor::MidLeft);
        m.text_field(cv, Rect::new(x + 550, y, 64, 26), &q["dur"].as_i64().unwrap_or(500).to_string(), "pc_dur", None, 6, false);
        m.text(cv, "x", (x + 624, y + 13), TEXT, &f.small, Anchor::MidLeft);
        m.text_field(cv, Rect::new(x + 640, y, 44, 26), &q["repeats"].as_i64().unwrap_or(1).to_string(), "pc_rep", None, 4, false);
        m.button(
            cv,
            Rect::new(x + 694, y, m.bw("Run steps", 110), 26),
            "Run steps",
            Act::Custom("run".into()),
            "pc_run",
            Style::Primary,
            true,
            Some("Current-clamp steps of this amplitude and duration, repeated; the sweeps are recorded for export."),
            false,
            None,
        );
        y += 34;
        let job = &q["job"];
        let busy = job.is_object();
        let nwb = q["nwb"].as_str().filter(|s| !s.is_empty());
        let items = [
            RowButton {
                label: if busy { format!("running… {}", job["note"].as_str().unwrap_or("")) } else { "Record I-F curve".into() },
                click: Act::Custom("if".into()),
                id: "pc_if".into(),
                style: Style::Normal,
                enabled: !busy,
                tip: Some("Firing rate against injected current, several sweeps per amplitude, in the chosen mode.".into()),
                w: 210,
            },
            RowButton {
                label: "Export CSV".into(),
                click: Act::Custom("csv".into()),
                id: "pc_csv".into(),
                style: Style::Normal,
                enabled: q["recording"].is_object() || q["curve"].is_object(),
                tip: None,
                w: 120,
            },
            RowButton {
                label: "Export NWB".into(),
                click: Act::Custom("nwb".into()),
                id: "pc_nwb".into(),
                style: Style::Normal,
                enabled: q["recording"].is_object() && nwb.is_none(),
                tip: Some(nwb.unwrap_or("Neurodata Without Borders; the potential stays in model units.").to_string()),
                w: 120,
            },
        ];
        m.button_row(cv, x, y, 30, &items, 10, None);
        if let Some(e) = job["error"].as_str().or_else(|| q["error"].as_str()).filter(|e| !e.is_empty()) {
            m.text(cv, e, (x + 490, y + 14), BAD, &f.small, Anchor::MidLeft);
        }
        y += 38;
        let c = &q["curve"];
        if c.is_object() {
            let amps = series_f32(&c["amplitudes"]);
            let rates = series_f32(&c["rate_hz"]);
            let pts: Vec<(f64, f64)> = amps.iter().copied().zip(rates.iter().copied()).collect();
            let markers: Vec<(f64, f64, Color)> = pts.iter().map(|p| (p.0, p.1, [255, 160, 80])).collect();
            plot(
                m,
                cv,
                Rect::new(x, y, 380, rect.bottom() - y - 70),
                &[(pts, [255, 160, 80], 2)],
                (amps.iter().copied().fold(f64::INFINITY, f64::min), amps.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                (0.0, 10.0f64.max(rates.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.1)),
                "injected current (model units)",
                "spikes/s",
                &[],
                &markers,
            );
            let mode = c["mode"].as_str().unwrap_or("");
            let rheo = c["rheobase"]
                .as_f64()
                .map_or("It never fired.".to_string(), |r| format!("Smallest current that made it fire: {}", super::day4::fmt_gp(r, 6)));
            let iso = if mode == "isolated" {
                " In isolation every neuron has this same curve: every unit in the model is identical."
            } else {
                ""
            };
            m.wrapped(
                cv,
                &format!(
                    "I-F curve, {mode} mode, {} sweeps x {} ms per amplitude. {rheo}{iso}",
                    c["repeats"],
                    super::day4::fmt_gp(num(&c["duration_ms"]), 6)
                ),
                (x + 400, y + 6),
                rect.w - 480,
                TEXT,
                &f.small,
                4,
            );
        }
        if let Some(note) = q["note"].as_str().filter(|n| !n.is_empty()) {
            m.text(cv, note, (x + 400, y + 90), AMBER, &f.small, Anchor::TopLeft);
        }
        back(m, cv, rect, "patch");
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let val = text.and_then(|t| t.parse::<f64>().ok());
        match name {
            "pick" => drop(ctx.host.command("lab.patch_pick", Value::Null)),
            "run" => drop(ctx.host.command("lab.patch_run", Value::Null)),
            "if" => drop(ctx.host.command("lab.patch_if", Value::Null)),
            "csv" => drop(ctx.host.command("lab.patch_export", json!({"kind": "csv"}))),
            "nwb" => drop(ctx.host.command("lab.patch_export", json!({"kind": "nwb"}))),
            "text:pc_type" => {
                if let Some(t) = text {
                    ctx.host.command("lab.patch_set", json!({"type": t}));
                }
            }
            "text:pc_idx" => drop(ctx.host.command("lab.patch_set", json!({"index": val.map_or(0, |v| v.max(0.0) as i64)}))),
            "text:pc_dur" => drop(ctx.host.command("lab.patch_set", json!({"dur": val.map_or(500, |v| v.clamp(10.0, 20000.0) as i64)}))),
            "text:pc_rep" => drop(ctx.host.command("lab.patch_set", json!({"repeats": val.map_or(1, |v| v.clamp(1.0, 100.0) as i64)}))),
            "slide:pc_hold" => {
                if let Some(v) = val {
                    ctx.host.command("lab.patch_set", json!({"hold": v}));
                }
            }
            "slide:pc_amp" => {
                if let Some(v) = val {
                    ctx.host.command("lab.patch_set", json!({"amp": v}));
                }
            }
            n if n.starts_with("mode:") => drop(ctx.host.command("lab.patch_set", json!({"mode": n[5..].parse::<u64>().unwrap_or(0)}))),
            n if n.starts_with("live:") => drop(ctx.host.command("lab.patch_set", json!({"live": &n[5..] == "1"}))),
            _ => {}
        }
    }
}

// ==== Calcium imaging ===============================================================================================================
/// Host: query `lab.imaging_state` = `{on, indicator, indicators: [{key, name}], fps, f0, noise, roi_mode, keep, error, reduced, source,
/// verified, msg, nwb, view: {names, t, dff, true_dff, frames}}`; commands `lab.imaging_set {on|indicator|fps|f0|noise|roi_mode|keep}`,
/// `lab.imaging_export {kind}`.
#[derive(Default)]
pub struct LabImaging;

/// `np.nanpercentile(x, q)` (linear interpolation).
fn percentile(mut v: Vec<f64>, q: f64) -> f64 {
    v.retain(|x| !x.is_nan());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

impl Page for LabImaging {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[allow(clippy::too_many_lines)]
    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.imaging_state");
        let y0 = title(
            m,
            cv,
            rect,
            "SIMULATED CALCIUM IMAGING",
            "The brain view colored by GCaMP dF/F: spikes convolved with an indicator kernel, plus photon shot noise.",
            &[
                ("MODEL", "a forward model on the simulation's spikes, not a measurement"),
                ("LITERATURE", "kernel speeds: see the indicator notes below"),
                ("CONNECTOME", "which neurons spike"),
            ],
        );
        let x = rect.x + 24;
        let rh = 28.max(f.small.linesize + 10);
        let lw = ["Imaging mode", "Indicator", "Frame rate", "ROIs"].iter().map(|t| f.text.size(t)).max().unwrap_or(0) + 20;
        let right = rect.right() - 24;
        let mut cur = (x + lw, y0);
        let place = |w: i32, cur: &mut (i32, i32)| -> Rect {
            if cur.0 + w > right && cur.0 > x + lw {
                cur.0 = x + lw;
                cur.1 += rh + 6;
            }
            let r = Rect::new(cur.0, cur.1, w, rh);
            cur.0 += w + 14;
            r
        };
        let row = |m: &mut Menu, cv: &mut Canvas, label: &str, cur: &mut (i32, i32)| {
            let y = if cur.0 == x + lw { cur.1 } else { cur.1 + rh + 8 };
            *cur = (x + lw, y);
            m.text(cv, label, (x, y + rh / 2), TEXT, &f.text, Anchor::MidLeft);
        };
        let on = q["on"].as_bool().unwrap_or(false);
        row(m, cv, "Imaging mode", &mut cur);
        let r = place(100, &mut cur);
        m.toggle(
            cv,
            r,
            on,
            Act::Custom(format!("on:{}", u8::from(!on))),
            "im_on",
            true,
            "The brain view shows dF/F instead of firing. Press B for the big view.",
        );
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            let r = place(100.max(right - cur.0), &mut cur);
            m.wrapped(cv, e, (r.x, r.y + 4), r.w, BAD, &f.small, 2);
        }
        row(m, cv, "Indicator", &mut cur);
        let inds = q["indicators"].as_array().cloned().unwrap_or_default();
        let labels: Vec<String> =
            inds.iter().map(|i| i["name"].as_str().unwrap_or("").split(" (").next().unwrap_or("").to_string()).collect();
        let cur_key = q["indicator"].as_str().unwrap_or("gcamp6s");
        let sel = inds.iter().position(|i| i["key"].as_str() == Some(cur_key)).unwrap_or(0);
        let wl = labels.iter().map(|t| f.small.size(t)).max().unwrap_or(0);
        let r = place((right - (x + lw)).min(420.max(labels.len() as i32 * (wl + 24))), &mut cur);
        let acts: Vec<Act> = inds.iter().map(|i| Act::Custom(format!("ind:{}", i["key"].as_str().unwrap_or("")))).collect();
        m.segmented(cv, r, &labels, sel, &acts, "im_ind", true, "");
        row(m, cv, "Frame rate", &mut cur);
        let r = place(200, &mut cur);
        m.page_slider(
            cv,
            r,
            num(&q["fps"]),
            (5.0, 40.0, 5.0),
            "{:.0f} Hz",
            "im_fps",
            true,
            Some("Applies when imaging mode is (re)started. The 5 ms simulation step limits it to 200 Hz."),
        );
        let r = place(f.small.size("photons/cell/frame") + 8 + 170, &mut cur);
        m.text(cv, "photons/cell/frame", (r.x, r.centery()), TEXT, &f.small, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(r.right() - 170, r.y, 170, rh),
            num(&q["f0"]),
            (10.0, 1000.0, 10.0),
            "{:.0f}",
            "im_f0",
            true,
            Some("MODEL parameter: fewer photons, more shot noise."),
        );
        let r = place(f.small.size("shot noise") + 8 + 100, &mut cur);
        m.text(cv, "shot noise", (r.x, r.centery()), TEXT, &f.small, Anchor::MidLeft);
        let noise = q["noise"].as_bool().unwrap_or(true);
        m.toggle(
            cv,
            Rect::new(r.right() - 100, r.y, 100, rh),
            noise,
            Act::Custom(format!("noise:{}", u8::from(!noise))),
            "im_noise",
            true,
            "",
        );
        row(m, cv, "ROIs", &mut cur);
        let spec = q["roi_mode"].as_str().unwrap_or("regions");
        let r = place(230.max(2 * f.small.size("Brain regions") + 48), &mut cur);
        m.segmented(
            cv,
            r,
            &["Brain regions".into(), "Custom".into()],
            usize::from(spec != "regions"),
            &[Act::Custom("roi:0".into()), Act::Custom("roi:1".into())],
            "im_roi",
            true,
            "",
        );
        if spec != "regions" {
            let r = place(360, &mut cur);
            m.text_field(
                cv,
                r,
                spec,
                "im_spec",
                Some("Neuron specs separated by ;  for example  type:DNp01;prefix:KC;line:SS00727 . One ROI each."),
                60,
                false,
            );
        }
        let r = place(f.small.size("keep frames") + 8 + 100, &mut cur);
        m.text(cv, "keep frames", (r.x, r.centery()), TEXT, &f.small, Anchor::MidLeft);
        let keep = q["keep"].as_bool().unwrap_or(false);
        m.toggle(
            cv,
            Rect::new(r.right() - 100, r.y, 100, rh),
            keep,
            Act::Custom(format!("keep:{}", u8::from(!keep))),
            "im_keep",
            true,
            "Keep the rendered frames (last ~15 s) so they can be exported as TIFF or NWB ImageSeries.",
        );
        let mut y = cur.1 + rh + 12;
        let ind_text =
            format!("{}: {}. {}", sel_name(&inds, sel), q["source"].as_str().unwrap_or(""), q["verified"].as_str().unwrap_or(""));
        let model_text = "MODEL: each spike adds the same kernel; every neuron in an ROI is equally bright; F0 is a running mean (30 s), so a steady \
rate reads as 0 and only changes show; no bleaching, motion, scattering or neuropil. dF/F per spike (0.2) and the \
photon budget are game parameters.";
        let lh = f.small.linesize;
        let notes_h = (Menu::fit_lines(&f.small, &ind_text, rect.w - 200, 6).0.len()
            + Menu::fit_lines(&f.small, model_text, rect.w - 200, 4).0.len()) as i32
            * lh
            + 8;
        let pl = Rect::new(x, y, rect.w - 48, 60.max(190.min(rect.bottom() - 70 - notes_h - 46 - y)));
        let view = &q["view"];
        let t: Vec<f64> = view["t"].as_array().into_iter().flatten().map(num).collect();
        let names: Vec<String> = view["names"].as_array().into_iter().flatten().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        if on && t.len() > 2 {
            let reduced = q["reduced"].as_bool().unwrap_or(false);
            let data = if reduced { &view["true_dff"] } else { &view["dff"] };
            let rows: Vec<Vec<f64>> =
                data.as_array().into_iter().flatten().map(|r| r.as_array().into_iter().flatten().map(num).collect()).collect();
            let last = *t.last().unwrap_or(&0.0);
            let tt: Vec<f64> = t.iter().map(|v| v - last).collect();
            let cols: [Color; 7] =
                [[255, 170, 60], [90, 200, 255], [120, 230, 120], [240, 110, 200], [250, 220, 90], [170, 140, 255], [255, 255, 255]];
            let k = rows.first().map_or(0, Vec::len).min(7);
            #[allow(clippy::type_complexity)]
            let series: Vec<(Vec<(f64, f64)>, Color, i32)> =
                (0..k).map(|j| (tt.iter().copied().zip(rows.iter().map(|r| r[j])).collect(), cols[j % 7], 2)).collect();
            let all: Vec<f64> = rows.iter().flat_map(|r| r[..k].iter().copied()).collect();
            let top = 0.5f64.max(percentile(all, 99.0) * 1.1);
            plot(m, cv, pl, &series, (tt[0], 0.0), (-0.1, top), "seconds", "dF/F (fraction)", &[], &[]);
            let mut lx = pl.x + 150;
            for (j, n) in names.iter().take(k).enumerate() {
                let short: String = n.chars().take(16).collect();
                m.text(cv, &short, (lx, pl.y + 4), cols[j % 7], &f.small, Anchor::TopLeft);
                lx += 120;
            }
        } else {
            cv.fill_rect(pl, [12, 14, 20], 8);
            m.text(cv, "turn Imaging mode on to see the ROI traces", pl.center(), LABEL, &f.small, Anchor::Center);
        }
        y = pl.bottom() + 8;
        let nwb = q["nwb"].as_str().filter(|s| !s.is_empty());
        let have = on && t.len() > 1;
        let items = [
            RowButton {
                label: "Export CSV".into(),
                click: Act::Custom("csv".into()),
                id: "im_csv".into(),
                style: Style::Normal,
                enabled: have,
                tip: None,
                w: 130,
            },
            RowButton {
                label: "Export NWB".into(),
                click: Act::Custom("nwb".into()),
                id: "im_nwb".into(),
                style: Style::Normal,
                enabled: have && nwb.is_none(),
                tip: Some(nwb.unwrap_or("RoiResponseSeries (dF/F and photons) and an ImageSeries.").to_string()),
                w: 130,
            },
            RowButton {
                label: "Export TIFF".into(),
                click: Act::Custom("tiff".into()),
                id: "im_tiff".into(),
                style: Style::Normal,
                enabled: q["frames"].as_bool().unwrap_or(false),
                tip: Some("A multi-page TIFF of the rendered view. Turn on 'keep frames' first.".into()),
                w: 130,
            },
        ];
        let (xe, _) = m.button_row(cv, x, y, 30, &items, 10, None);
        if let Some(msg) = q["msg"].as_str().filter(|s| !s.is_empty()) {
            m.text(cv, msg, (xe + 10, y + 15), AMBER, &f.small, Anchor::MidLeft);
        }
        y += 38;
        y = m.wrapped(cv, &ind_text, (x, y), rect.w - 200, LABEL, &f.small, 6) + 8;
        m.wrapped(cv, model_text, (x, y), rect.w - 200, LABEL, &f.small, 4);
        back(m, cv, rect, "imaging");
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let val = text.and_then(|t| t.parse::<f64>().ok());
        let set = |ctx: &mut Ctx, v: Value| drop(ctx.host.command("lab.imaging_set", v));
        match name {
            "csv" | "nwb" | "tiff" => drop(ctx.host.command("lab.imaging_export", json!({"kind": name}))),
            "slide:im_fps" => {
                if let Some(v) = val {
                    set(ctx, json!({"fps": v}));
                }
            }
            "slide:im_f0" => {
                if let Some(v) = val {
                    set(ctx, json!({"f0": v}));
                }
            }
            "text:im_spec" => {
                let t = text.unwrap_or("").trim();
                set(ctx, json!({"roi_mode": if t.is_empty() { "regions" } else { t }}));
            }
            n if n.starts_with("on:") => set(ctx, json!({"on": &n[3..] == "1"})),
            n if n.starts_with("ind:") => set(ctx, json!({"indicator": &n[4..]})),
            n if n.starts_with("noise:") => set(ctx, json!({"noise": &n[6..] == "1"})),
            n if n.starts_with("keep:") => set(ctx, json!({"keep": &n[5..] == "1"})),
            n if n.starts_with("roi:") => set(ctx, json!({"roi_mode": if &n[4..] == "0" { "regions" } else { "type:DNp01" }})),
            _ => {}
        }
    }
}

fn sel_name(inds: &[Value], sel: usize) -> String {
    inds.get(sel).and_then(|i| i["name"].as_str()).unwrap_or("").to_string()
}
