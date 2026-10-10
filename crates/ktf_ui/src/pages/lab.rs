//! Lab mode screens (`lab/lab.py`): the hub, the parameters and the model assumptions.
//!
//! Host contract: queries `lab_pages` = `[[label, page, tip], ...]` (the Lab's pages, in order; those not registered on the menu are left
//! out), `lab_engine` = the engine line, `lab_notes` = extra amber notes for the hub (the modified connectome), `lab_params` = `{name: value}`;
//! command `set_lab_param {name, value}`.

use std::collections::BTreeMap;

use ktf_data::labparams as lp;
use serde_json::json;

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

fn values(ctx: &Ctx) -> BTreeMap<String, f64> {
    ctx.host
        .query("lab_params")
        .as_object()
        .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_f64()?))).collect())
        .unwrap_or_default()
}

pub struct LabHub;

impl Page for LabHub {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, "LAB", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let top = (rect.y + 84).max(m.subtitle(
            cv,
            rect,
            "Research tools. Everything here runs on the same connectome sim as the game.",
            Some(rect.y + 50),
            2,
        ));
        let engine = ctx.host.query("lab_engine");
        m.text(
            cv,
            &format!("Engine: {}", engine.as_str().unwrap_or("CPU (NumPy) · CPU")),
            (rect.right() - 24, rect.y + 20),
            [120, 220, 240],
            &f.small,
            Anchor::TopRight,
        );
        let items: Vec<(String, String, String)> = ctx
            .host
            .query("lab_pages")
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|e| Some((e[0].as_str()?.to_string(), e[1].as_str()?.to_string(), e[2].as_str()?.to_string())))
                    .filter(|(_, page, _)| m.has_page(page))
                    .collect()
            })
            .unwrap_or_default();
        let body = Rect::new(rect.x + 8, top, rect.w - 16, rect.bottom() - 96 - top);
        let off = m.scroll_offset("lab");
        m.begin_clip(cv, body);
        let (bw, bh) = ((rect.w - 72) / 2, 64);
        for (i, (label, page, tip)) in items.iter().enumerate() {
            let x = rect.x + 24 + (i as i32 % 2) * (bw + 24);
            let y = body.y + 6 + (i as i32 / 2) * (bh + 16) - off;
            m.button(
                cv,
                Rect::new(x, y, bw, bh),
                label,
                Act::Show(page.clone()),
                &format!("lab:{page}"),
                Style::Normal,
                true,
                Some(tip),
                false,
                None,
            );
        }
        let rows = (items.len() as i32 + 1) / 2;
        m.set_content_h("lab", (body.y + 6 + rows * (bh + 16) - body.bottom() + 8).max(0));
        m.end_clip(cv);
        let vals = values(ctx);
        let mut notes: Vec<String> = Vec::new();
        let modified = lp::modified(&vals);
        if !modified.is_empty() {
            notes.push(format!("Modified parameters: {}", modified.iter().map(|p| p.label.as_str()).collect::<Vec<_>>().join(", ")));
        }
        if let Some(extra) = ctx.host.query("lab_notes").as_array() {
            notes.extend(extra.iter().filter_map(|n| n.as_str().map(String::from)));
        }
        let mut ny = rect.bottom() - 100;
        for note in notes {
            ny = m.wrapped(cv, &note, (rect.x + 24, ny), rect.w - 220, AMBER, &f.small, 2);
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "lab:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, _name: &str, _text: Option<&str>, _ctx: &mut Ctx) {}
}

pub struct LabParams;

impl Page for LabParams {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, "PARAMETERS", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let top = (rect.y + 80).max(m.subtitle(
            cv,
            rect,
            "Applies live to every fly. Validation results were measured at the defaults.",
            Some(rect.y + 50),
            2,
        ));
        let body = Rect::new(rect.x + 16, top, rect.w - 32, rect.bottom() - 70 - top);
        let key = "lab_params";
        let off = m.scroll_offset(key);
        let vals = values(ctx);
        m.begin_clip(cv, body);
        let mut y = body.y + 4 - off;
        for p in lp::params() {
            let value = vals.get(&p.name).copied().unwrap_or(p.default);
            let row = Rect::new(body.x, y, body.w, 40);
            let changed = (value - p.default).abs() > 1e-9;
            m.text(cv, &p.label, (row.x + 12, row.center().1), if changed { AMBER } else { TEXT }, &f.text, Anchor::MidLeft);
            m.chip(cv, (row.x + 330, row.center().1 - 10), if p.kind == "model" { "MODEL" } else { "GAME RULE" });
            let tip = format!("{}  Default {}.", p.tip, crate::menu::py_fmt(&p.fmt, p.default));
            m.register_label(Rect::new(row.x, row.y, 440, row.h), &format!("plabel:{}", p.name), &tip);
            m.page_slider(
                cv,
                Rect::new(row.x + 450, row.y + 6, row.w - 470, 28),
                value,
                (p.lo, p.hi, p.step),
                &p.fmt,
                &format!("param:{}", p.name),
                true,
                Some(&p.tip),
            );
            y += 44;
        }
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.end_clip(cv);
        m.button(
            cv,
            Rect::new(rect.x + 24, rect.bottom() - 58, m.bw("Reset to defaults", 200), 42),
            "Reset to defaults",
            Act::Custom("reset".into()),
            "params:reset",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "params:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        if let Some(p) = name.strip_prefix("slide:param:") {
            if let Some(v) = text.and_then(|t| t.parse::<f64>().ok()) {
                ctx.host.command("set_lab_param", json!({"name": p, "value": v}));
            }
        } else if name == "reset" {
            for p in lp::params() {
                ctx.host.command("set_lab_param", json!({"name": p.name, "value": p.default}));
            }
        }
    }
}

pub struct LabAssumptions;

impl Page for LabAssumptions {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, _ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, "MODEL ASSUMPTIONS & SIMPLIFICATIONS", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let top = m.wrapped(
            cv,
            "Scientific caveats and approximations distinguishing the simulation from living biology.",
            (rect.x + 24, m.under_heading(rect).max(rect.y + 50)),
            rect.w - 48,
            LABEL,
            &f.small,
            2,
        ) + 8;
        let body = Rect::new(rect.x + 16, top, rect.w - 32, rect.bottom() - 70 - top);
        let key = "lab_assumptions";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y + 4 - off;
        let color = |c: &str| -> Color {
            match c {
                "SYNAPSE" => [100, 180, 240],
                "BIOPHYSICS" => [150, 120, 220],
                "DYNAMICS" => [240, 160, 60],
                "TUNING" => [80, 200, 140],
                "DATASET" => [240, 100, 100],
                "GAME RULE" => [220, 150, 50],
                _ => LABEL,
            }
        };
        let (lh, bh) = (f.small.linesize, f.bold.linesize);
        let lab_w = ["Sim:", "Bio:", "Doc:"].iter().map(|t| f.small.size(t)).max().unwrap_or(0) + 10;
        let n_lines = |text: &str, font: &crate::font::Font, width: i32| Menu::fit_lines(font, text, width, 99).0.len() as i32;
        for (title, cat, sim_rule, bio, docs) in lp::assumptions() {
            let cw = body.w - 12;
            let chip_w = f.small.size(cat) + 14;
            let title_x = 12 + chip_w + 10;
            let tw = cw - title_x - 12;
            let text_w = cw - 14 - lab_w - 12;
            let rows = [n_lines(sim_rule, &f.small, text_w), n_lines(bio, &f.small, text_w), n_lines(docs, &f.small, text_w)];
            let card_h = 10 + (lh + 4).max(n_lines(title, &f.bold, tw) * bh) + 6 + rows.iter().map(|r| r * lh + 6).sum::<i32>() + 6;
            let card = Rect::new(body.x, y, cw, card_h);
            cv.fill_rect(card, [24, 28, 38], 8);
            cv.stroke_rect(card, [45, 52, 68], 8);
            let col = color(cat);
            let chip = Rect::new(card.x + 12, card.y + 10, chip_w, lh + 4);
            cv.fill_rect(chip, [15, 18, 26], 4);
            cv.stroke_rect(chip, col, 4);
            m.text(cv, cat, chip.center(), col, &f.small, Anchor::Center);
            let mut yy = m.wrapped(cv, title, (card.x + title_x, card.y + 10), tw, INK, &f.bold, 99);
            yy = yy.max(chip.bottom()) + 6;
            for (label, text, color, lab_col) in [
                ("Sim:", sim_rule, TEXT, [140, 180, 220]),
                ("Bio:", bio, [185, 190, 200], [220, 150, 100]),
                ("Doc:", docs, [130, 160, 210], LABEL),
            ] {
                m.text(cv, label, (card.x + 14, yy), lab_col, &f.small, Anchor::TopLeft);
                yy = m.wrapped(cv, text, (card.x + 14 + lab_w, yy), text_w, color, &f.small, 99) + 6;
            }
            y += card_h + 12;
        }
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.end_clip(cv);
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "assumptions:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, _name: &str, _text: Option<&str>, _ctx: &mut Ctx) {}
}

/// `labstats.fmt_p`
pub fn fmt_p(p: f64) -> String {
    if p.is_nan() {
        "p = n/a".into()
    } else if p < 0.001 {
        "p < 0.001".into()
    } else {
        format!("p = {p:.3}")
    }
}

/// Lab > Validation (`lab/lab.py:page_validation`): the card of every test with its numbers and criteria, and the run button.
///
/// Host: query `validation_results` = `{results: <validation.run dict> | null, source: "run on this PC" | "from this build", job:
/// {done, total, error} | null}`; command `start_validation`.
pub struct LabValidation;

impl Page for LabValidation {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, "VALIDATION", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let q = ctx.host.query("validation_results");
        let res = &q["results"];
        if res.is_null() {
            m.text(cv, "No results yet. Run the suite (a few minutes).", (rect.x + 24, rect.y + 50), LABEL, &f.small, Anchor::TopLeft);
        } else {
            let seeds = res["seeds"].as_array().cloned().unwrap_or_default();
            m.text(
                cv,
                &format!(
                    "Kick the Fly {}, {}, {}  ·  seeds {}-{} (n={})  ·  thresholds chosen for this release, not from the papers",
                    res["app_version"].as_str().unwrap_or(""),
                    res["created"].as_str().unwrap_or(""),
                    q["source"].as_str().unwrap_or(""),
                    seeds.first().map_or(String::new(), |s| s.to_string()),
                    seeds.last().map_or(String::new(), |s| s.to_string()),
                    seeds.len()
                ),
                (rect.x + 24, rect.y + 50),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
        }
        if !lp::modified(&values(ctx)).is_empty() {
            m.text(
                cv,
                "Parameters are modified: these results were measured at the defaults.",
                (rect.right() - 24, rect.y + 20),
                AMBER,
                &f.small,
                Anchor::TopRight,
            );
        }
        let body = Rect::new(rect.x + 16, rect.y + 76, rect.w - 32, rect.h - 76 - 70);
        let key = "lab_validation";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y - off;
        let n = |v: &serde_json::Value, k: &str| v[k].as_f64().unwrap_or(f64::NAN);
        for t in res["tests"].as_array().cloned().unwrap_or_default() {
            let card = Rect::new(body.x, y, body.w - 12, 156);
            cv.fill_rect(card, [26, 30, 40], 10);
            let ok = t["passed"].as_bool().unwrap_or(false);
            let chip = Rect::new(card.x + 12, card.y + 12, 64, 24);
            cv.fill_rect(chip, if ok { GOOD } else { BAD }, 6);
            m.text(cv, if ok { "PASS" } else { "FAIL" }, chip.center(), [10, 12, 16], &f.bold, Anchor::Center);
            m.text(cv, t["name"].as_str().unwrap_or(""), (card.x + 90, card.y + 12), INK, &f.bold, Anchor::TopLeft);
            let claim: String = t["claim"].as_str().unwrap_or("").chars().take(140).collect();
            m.text(cv, &claim, (card.x + 90, card.y + 36), TEXT, &f.small, Anchor::TopLeft);
            let mm = &t["measured"];
            let meas = if mm.get("drive_ratio_mean").is_some() {
                format!(
                    "{}: x{:.2} ± {:.2} driving {}  vs  x{:.2} ± {:.2} for {}  ·  {}",
                    t["readout_label"].as_str().unwrap_or(""),
                    n(mm, "drive_ratio_mean"),
                    n(mm, "drive_ratio_sd"),
                    t["drive_label"].as_str().unwrap_or(""),
                    n(mm, "control_ratio_mean"),
                    n(mm, "control_ratio_sd"),
                    t["control_label"].as_str().unwrap_or(""),
                    fmt_p(n(mm, "p_value"))
                )
            } else if mm.get("extinguished_pi_mean").is_some() {
                format!(
                    "PI after extinction {:.2} ± {:.2} vs {:.2} ± {:.2} without  ·  fear of CS+ {:.2} vs {:.2}  ·  {}",
                    n(mm, "extinguished_pi_mean"),
                    n(mm, "extinguished_pi_sd"),
                    n(mm, "unextinguished_pi_mean"),
                    n(mm, "unextinguished_pi_sd"),
                    n(mm, "fear_extinguished"),
                    n(mm, "fear_unextinguished"),
                    fmt_p(n(mm, "p_value"))
                )
            } else if mm.get("fear_b_paired").is_some() {
                format!(
                    "PI for odor B {:.2} ± {:.2} paired vs {:.2} ± {:.2} unpaired  ·  fear of odor B {:.2} vs {:.2}  ·  {}",
                    n(mm, "pi_mean"),
                    n(mm, "pi_sd"),
                    n(mm, "control_pi_mean"),
                    n(mm, "control_pi_sd"),
                    n(mm, "fear_b_paired"),
                    n(mm, "fear_b_unpaired"),
                    fmt_p(n(mm, "p_value"))
                )
            } else if mm.get("pi_mean").is_some() {
                format!(
                    "PI {:.2} ± {:.2} vs unpaired {:.2} ± {:.2}  ·  fear CS+ {:.2} vs CS- {:.2}  ·  approach MBONs {:.1} vs {:.1} Hz  ·  {}",
                    n(mm, "pi_mean"),
                    n(mm, "pi_sd"),
                    n(mm, "control_pi_mean"),
                    n(mm, "control_pi_sd"),
                    n(mm, "fear_cs_plus"),
                    n(mm, "fear_cs_minus"),
                    n(mm, "approach_mbon_cs_plus_hz"),
                    n(mm, "approach_mbon_cs_minus_hz"),
                    fmt_p(n(mm, "p_value"))
                )
            } else if mm.get("direction_tracking").is_some() {
                format!(
                    "EPG contrast in wind {:.2}x (no wind {:.2}x, need >= 3.0x)  ·  persistence {:.0} ms (need >= 500)  ·  tracking |r| {:.2} vs {:.2} shuffled, p = {:.2}  ·  EPG {:.1} -> {:.1} Hz",
                    n(mm, "mean_contrast"),
                    n(mm, "baseline_contrast"),
                    n(mm, "mean_persistence_ms"),
                    n(mm, "direction_tracking"),
                    n(mm, "direction_tracking_null"),
                    n(mm, "direction_tracking_p"),
                    n(mm, "epg_rate_baseline_hz"),
                    n(mm, "epg_rate_wind_hz")
                )
            } else if mm.get("mean_contrast").is_some() {
                format!(
                    "EPG peak/trough contrast: {:.2} ± {:.2}x (required >= 3.0x)  ·  persistence: {:.0} ms (< 500 ms)  ·  baseline: {:.1} ± {:.1} Hz",
                    n(mm, "mean_contrast"),
                    mm["sd_contrast"].as_f64().unwrap_or(0.0),
                    mm["mean_persistence_ms"].as_f64().unwrap_or(0.0),
                    mm["mean_baseline_hz"].as_f64().unwrap_or(0.0),
                    mm["mean_baseline_sd_hz"].as_f64().unwrap_or(0.0)
                )
            } else {
                t["finding"].as_str().map_or_else(|| mm.to_string(), String::from)
            };
            let yy = m.wrapped(cv, &meas, (card.x + 90, card.y + 58), card.w - 110, INK, &f.small, 2);
            m.text(
                cv,
                &format!("Pass if: {}", t["criteria"].as_str().unwrap_or("")),
                (card.x + 90, yy + 4),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            m.text(cv, t["citation"].as_str().unwrap_or(""), (card.x + 90, yy + 24), [150, 180, 220], &f.small, Anchor::TopLeft);
            if let Some(note) = t["note"].as_str().filter(|s| !s.is_empty()) {
                m.register_label(card, &format!("vnote:{}", t["id"].as_str().unwrap_or("")), note);
                m.text(cv, "hover for notes", (card.right() - 12, card.y + 12), DIM, &f.small, Anchor::TopRight);
            }
            y += 164;
        }
        m.set_content_h(key, (y + off - body.bottom()).max(0));
        m.end_clip(cv);
        if let Some(job) = q["job"].as_object() {
            let (done, total) =
                (job.get("done").and_then(|v| v.as_f64()).unwrap_or(0.0), job.get("total").and_then(|v| v.as_f64()).unwrap_or(1.0));
            let frac = done / total.max(1.0);
            cv.fill_rect(Rect::new(rect.x + 24, rect.bottom() - 44, rect.w - 420, 12), [30, 36, 48], 6);
            cv.fill_rect(Rect::new(rect.x + 24, rect.bottom() - 44, 8.max(((rect.w - 420) as f64 * frac) as i32), 12), AMBER, 6);
            let err = job.get("error").and_then(|e| e.as_str()).filter(|e| !e.is_empty());
            m.text(
                cv,
                &format!("running: {done:.0}/{total:.0}{}", err.map_or(String::new(), |e| format!("  error: {e}"))),
                (rect.x + 24, rect.bottom() - 28),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
        } else {
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 58, 240, 42),
                "Run validation now",
                Act::Custom("run".into()),
                "val_run",
                Style::Normal,
                true,
                Some("Runs every test on this PC with the default parameters (a few minutes). The result replaces the build's bundled one on this PC."),
                false,
                None,
            );
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "val:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if name == "run" {
            ctx.host.command("start_validation", json!({}));
        }
    }
}
