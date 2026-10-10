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
