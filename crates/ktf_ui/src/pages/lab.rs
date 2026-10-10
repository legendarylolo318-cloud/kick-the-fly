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

// --- assays and repeated trials (`lab/lab.py:page_assays`, `draw_assay_result`, `draw_chart`) -------------------------------------------
/// `lab.draw_chart`: a line chart with 95 % interval bars. `series`: `(label, color, [mean_ci dicts per x])`.
#[allow(clippy::too_many_arguments)]
fn draw_chart(
    m: &Menu,
    cv: &mut Canvas,
    rect: Rect,
    xs: &[String],
    series: &[(String, Color, Vec<serde_json::Value>)],
    x_label: &str,
    y_label: &str,
    y_max: Option<f64>,
    connect: bool,
) {
    let f = m.fonts();
    cv.fill_rect(rect, [12, 14, 20], 8);
    let plot = Rect::new(rect.x + 52, rect.y + 44, rect.w - 70, rect.h - 82);
    let g = |c: &serde_json::Value, k: &str| c[k].as_f64().unwrap_or(f64::NAN);
    let vals: Vec<f64> = series
        .iter()
        .flat_map(|(_, _, cs)| cs.iter())
        .filter(|c| !g(c, "mean").is_nan())
        .map(|c| if g(c, "hi").is_nan() { g(c, "mean") } else { g(c, "hi") })
        .collect();
    let top = y_max.unwrap_or_else(|| vals.iter().copied().fold(1e-6, f64::max) * 1.1);
    for k in 0..5 {
        let y = (f64::from(plot.bottom()) - f64::from(plot.h) * f64::from(k) / 4.0) as i32;
        cv.line((plot.x, y), (plot.right(), y), 1, [36, 40, 50], 255);
        m.text(cv, &py_g_2g(top * f64::from(k) / 4.0), (plot.x - 6, y), LABEL, &f.small, Anchor::MidRight);
    }
    let n = xs.len().max(1);
    let px = |i: usize| -> i32 { (f64::from(plot.x) + f64::from(plot.w) * (i as f64 + 0.5) / n as f64) as i32 };
    let py = |v: f64| -> i32 { (f64::from(plot.bottom()) - f64::from(plot.h) * (v / top).clamp(0.0, 1.0)) as i32 };
    for (i, x) in xs.iter().enumerate() {
        m.text(cv, x, (px(i), plot.bottom() + 6), LABEL, &f.small, Anchor::MidTop);
    }
    for (li, (label, col, cs)) in series.iter().enumerate() {
        let mut pts = Vec::new();
        for (i, c) in cs.iter().enumerate() {
            let mean = g(c, "mean");
            if mean.is_nan() {
                continue;
            }
            if !g(c, "lo").is_nan() {
                cv.line((px(i), py(g(c, "lo"))), (px(i), py(g(c, "hi"))), 2, *col, 255);
            }
            pts.push((px(i), py(mean)));
            cv.circle((px(i), py(mean)), 4, *col);
        }
        if pts.len() > 1 && connect {
            for w in pts.windows(2) {
                cv.line(w[0], w[1], 2, *col, 255);
            }
        }
        m.text(cv, label, (rect.right() - 12 - li as i32 * 190, rect.y + 6), *col, &f.small, Anchor::TopRight);
    }
    m.text(cv, x_label, (plot.center().0, rect.bottom() - 16), TEXT, &f.small, Anchor::MidTop);
    m.text(cv, y_label, (rect.x + 6, rect.y + 2), TEXT, &f.small, Anchor::TopLeft);
}

/// `lab._ci`
fn ci(c: &serde_json::Value, fmt: &str) -> String {
    let g = |k: &str| c[k].as_f64().unwrap_or(f64::NAN);
    if c["n"].as_u64().unwrap_or(0) == 0 || g("mean").is_nan() {
        return "n/a".into();
    }
    if g("lo").is_nan() {
        return crate::menu::py_fmt(fmt, g("mean"));
    }
    format!("{} ± {}", crate::menu::py_fmt(fmt, g("mean")), crate::menu::py_fmt(fmt, (g("hi") - g("lo")) / 2.0))
}

const WHITE_GREY: Color = [200, 200, 200];

fn draw_assay_result(m: &mut Menu, cv: &mut Canvas, area: Rect, res: &serde_json::Value) {
    let f = m.fonts();
    let kind = res["kind"].as_str().unwrap_or("");
    let (t, c) = (&res["treated"], &res["control"]);
    let has_c = c.is_object();
    let surg = res["surgery"].is_object();
    let tl = if surg { "with surgery" } else { "flies" };
    let chart = Rect::new(area.x, area.y, area.w / 2 - 10, area.h);
    let tx = area.x + area.w / 2 + 10;
    let mut y = area.y;
    let nseeds = res["seeds"].as_array().map_or(0, Vec::len);
    m.text(cv, &format!("{}  ·  n = {nseeds} flies", res["label"].as_str().unwrap_or("")), (tx, y), INK, &f.bold, Anchor::TopLeft);
    y += 26;
    let rows_of = |v: &serde_json::Value| -> Vec<serde_json::Value> { v["rows"].as_array().cloned().unwrap_or_default() };
    let mut rows: Vec<(String, String, String)> = Vec::new();
    let nanv = || serde_json::json!({"mean": null, "lo": null, "hi": null, "n": 0});
    match kind {
        "tmaze" => {
            let (xs, pts): (Vec<String>, Vec<serde_json::Value>) = if has_c {
                (vec!["with surgery".into(), "control".into()], vec![t["pi"].clone(), c["pi"].clone()])
            } else {
                (vec!["all flies".into()], vec![t["pi"].clone()])
            };
            draw_chart(m, cv, chart, &xs, &[("mean and 95% CI".into(), ACCENT, pts)], "", "performance index", Some(1.0), false);
            let both = |k: &str, fmt: &str| (ci(&t[k], fmt), if has_c { ci(&c[k], fmt) } else { String::new() });
            for (label, k, fmt) in [
                ("performance index", "pi", "{:.2f}"),
                ("fear of CS+", "fear_cs_plus", "{:.2f}"),
                ("fear of CS-", "fear_cs_minus", "{:.2f}"),
                ("approach MBONs to CS+ (Hz)", "mbon_cs_plus_hz", "{:.1f}"),
                ("approach MBONs to CS- (Hz)", "mbon_cs_minus_hz", "{:.1f}"),
            ] {
                let (a, b) = both(k, fmt);
                rows.push((label.into(), a, b));
            }
        }
        "looming" => {
            let tr = rows_of(t);
            let xs: Vec<String> = tr.iter().map(|r| crate::menu::py_g(r["speed"].as_f64().unwrap_or(0.0))).collect();
            let mut series = vec![(tl.to_string(), ACCENT, tr.iter().map(|r| r["escape_probability"].clone()).collect::<Vec<_>>())];
            if has_c {
                series.push(("control".into(), WHITE_GREY, rows_of(c).iter().map(|r| r["escape_probability"].clone()).collect()));
            }
            draw_chart(m, cv, chart, &xs, &series, "approach speed (m/s)", "escape probability", Some(1.0), true);
            let cr = if has_c { rows_of(c) } else { tr.clone() };
            for (r, q) in tr.iter().zip(&cr) {
                rows.push((
                    format!("{} m/s", crate::menu::py_g(r["speed"].as_f64().unwrap_or(0.0))),
                    format!("{}/{} escaped, latency {} s", r["escapes"], r["approaches"], ci(&r["latency_s"], "{:.2f}")),
                    if has_c { format!("{}/{}", q["escapes"], q["approaches"]) } else { String::new() },
                ));
            }
        }
        _ => {
            // sugar
            let tr = rows_of(t);
            let xs: Vec<String> = tr.iter().map(|r| format!("{:.0}%", r["dose"].as_f64().unwrap_or(0.0) * 100.0)).collect();
            let mut series = vec![(tl.to_string(), ACCENT, tr.iter().map(|r| r["mn9_ratio"].clone()).collect::<Vec<_>>())];
            if has_c {
                series.push(("control".into(), WHITE_GREY, rows_of(c).iter().map(|r| r["mn9_ratio"].clone()).collect()));
            }
            draw_chart(m, cv, chart, &xs, &series, "sugar dose (share of sugar-pathway GRNs)", "MN9 firing x before", None, true);
            let cr = if has_c { rows_of(c) } else { tr.clone() };
            for (r, q) in tr.iter().zip(&cr) {
                let mean = q["mn9_ratio"]["mean"].as_f64().unwrap_or(f64::NAN);
                rows.push((
                    format!("{:.0}%", r["dose"].as_f64().unwrap_or(0.0) * 100.0),
                    format!("MN9 x{}, extended {}/{}", ci(&r["mn9_ratio"], "{:.2f}"), r["extensions"], r["offers"]),
                    if has_c { format!("x{mean:.2}") } else { String::new() },
                ));
            }
            let _ = nanv;
        }
    }
    let col_a = tx + 190;
    let col_b = tx + (area.w / 2 - 10) - 120;
    if has_c {
        m.text(cv, "surgery", (col_a, y), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, "control", (col_b, y), LABEL, &f.small, Anchor::TopLeft);
    } else {
        m.text(cv, "mean ± 95% CI half-width", (col_a, y), LABEL, &f.small, Anchor::TopLeft);
    }
    y += 18;
    for (label, a, b) in rows {
        m.text(cv, &label, (tx, y), TEXT, &f.small, Anchor::TopLeft);
        if has_c {
            m.wrapped(cv, &a, (col_a, y), col_b - col_a - 10, INK, &f.small, 1);
            m.text(cv, &b, (col_b, y), INK, &f.small, Anchor::TopLeft);
        } else {
            m.wrapped(cv, &a, (col_a, y), area.right() - col_a, INK, &f.small, 1);
        }
        y += 20;
    }
    if has_c {
        let o = &res["comparison"]["overall"];
        y += 8;
        m.text(
            cv,
            &format!(
                "Surgery vs same-seed control: mean difference {:+.3} (n={} pairs)",
                o["mean_difference"].as_f64().unwrap_or(f64::NAN),
                o["n"]
            ),
            (tx, y),
            INK,
            &f.small,
            Anchor::TopLeft,
        );
        let p = o["p_value"].as_f64().unwrap_or(f64::NAN);
        m.text(
            cv,
            &format!(
                "{}: {}   (paired t: {})",
                o["test"].as_str().unwrap_or(""),
                fmt_p(p),
                fmt_p(o["t_p_value"].as_f64().unwrap_or(f64::NAN))
            ),
            (tx, y + 18),
            if p < 0.05 { AMBER } else { TEXT },
            &f.small,
            Anchor::TopLeft,
        );
        let list: Vec<String> = res["surgery"]
            .as_object()
            .map(|s| s.iter().map(|(k, v)| format!("{k} {:+}", v.as_i64().unwrap_or(0))).collect())
            .unwrap_or_default();
        m.text(cv, &format!("surgery: {}", list.join(", ")), (tx, y + 36), LABEL, &f.small, Anchor::TopLeft);
    }
}

/// Lab > Assays and repeated trials.
///
/// Host: query `assay_state` = `{job: {done, total, elapsed, workers, error} | null, result: <labjobs.run_sync dict + label> | null, options:
/// [[label, [spec...]], ...] (the surgery options), kinds: [[key, label], ...], base_seed}`; commands `start_assay {kind, flies, surgery:
/// {spec: mode}, label}`, `cancel_assay`.
pub struct LabAssays {
    kind: String,
    flies: i64,
    surgery_i: usize,
    stimulate: bool,
}

impl Default for LabAssays {
    fn default() -> Self {
        Self { kind: "tmaze".into(), flies: 6, surgery_i: 0, stimulate: false }
    }
}

impl LabAssays {
    pub fn set_view(&mut self, kind: &str, flies: i64, surgery_i: usize, stimulate: bool) {
        self.kind = kind.into();
        self.flies = flies;
        self.surgery_i = surgery_i;
        self.stimulate = stimulate;
    }
}

impl Page for LabAssays {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("assay_state");
        let busy = q["job"].is_object() && q["job"]["error"].is_null();
        m.text(cv, "ASSAYS AND REPEATED TRIALS", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let mut y = m.subtitle(
            cv,
            rect,
            "Each fly is a fresh, untrained brain with its own seed. Your saved training memory isn't used or changed.",
            None,
            2,
        );
        let x0 = rect.x + 24;
        m.text(cv, "Assay", (x0, y + 15), TEXT, &f.text, Anchor::MidLeft);
        let kinds: Vec<(String, String)> = q["kinds"]
            .as_array()
            .map(|a| a.iter().filter_map(|k| Some((k[0].as_str()?.to_string(), k[1].as_str()?.to_string()))).collect())
            .unwrap_or_default();
        if busy {
            let label = kinds.iter().find(|k| k.0 == self.kind).map_or("", |k| k.1.as_str());
            m.text(cv, &format!("{label} (running)"), (x0 + 110, y + 15), INK, &f.bold, Anchor::MidLeft);
            y += 40;
        } else {
            let items: Vec<_> = kinds
                .iter()
                .map(|(k, label)| (label.clone(), Act::Custom(format!("kind:{k}")), format!("assay_kind:{k}"), self.kind == *k, None))
                .collect();
            y = m.flow_buttons(cv, x0 + 110, y, rect.right() - 24, &items, Some(30), None, 8, None) + 10;
        }
        m.text(cv, "Flies", (x0, y + 15), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(x0 + 110, y, 300, 30),
            self.flies as f64,
            (2.0, 30.0, 1.0),
            "{:.0f}",
            "assay_flies",
            !busy,
            Some("How many flies (seeds) to run. More gives tighter confidence intervals and takes longer."),
        );
        let base = q["base_seed"].as_i64().unwrap_or(1000);
        m.text(
            cv,
            &format!("seeds {}-{}", base + 2000, base + 2000 + self.flies - 1),
            (x0 + 430, y + 15),
            LABEL,
            &f.small,
            Anchor::MidLeft,
        );
        y += 40;
        let opts: Vec<(String, Vec<String>)> = q["options"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|o| {
                        Some((o[0].as_str()?.to_string(), o[1].as_array()?.iter().filter_map(|s| s.as_str().map(String::from)).collect()))
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![("none".into(), vec![])]);
        let si = self.surgery_i.min(opts.len().saturating_sub(1));
        let (label, spec) = (&opts[si].0, &opts[si].1);
        m.text(cv, "Surgery", (x0, y + 15), TEXT, &f.text, Anchor::MidLeft);
        m.button(cv, Rect::new(x0 + 110, y, 44, 30), "<", Act::Custom("surg:-1".into()), "surg<", Style::Normal, !busy, None, false, None);
        m.text(cv, label, (x0 + 170, y + 15), if spec.is_empty() { LABEL } else { INK }, &f.bold, Anchor::MidLeft);
        m.button(cv, Rect::new(x0 + 520, y, 44, 30), ">", Act::Custom("surg:1".into()), "surg>", Style::Normal, !busy, None, false, None);
        if !spec.is_empty() {
            m.segmented(
                cv,
                Rect::new(x0 + 580, y, 220, 30),
                &["Silence".into(), "Stimulate".into()],
                usize::from(self.stimulate),
                &[Act::Custom("mode:0".into()), Act::Custom("mode:1".into())],
                "surg_mode",
                !busy,
                "",
            );
        }
        y += 40;
        if !spec.is_empty() {
            m.text(
                cv,
                "Each fly also runs unperturbed with the same seed as its control; the results show both and a paired test.",
                (x0, y),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
        }
        y += 24;
        if busy {
            let (done, total) = (q["job"]["done"].as_f64().unwrap_or(0.0), q["job"]["total"].as_f64().unwrap_or(1.0));
            let frac = done / total.max(1.0);
            cv.fill_rect(Rect::new(x0, y, rect.w - 220, 14), [30, 36, 48], 7);
            cv.fill_rect(Rect::new(x0, y, 8.max(((rect.w - 220) as f64 * frac) as i32), 14), AMBER, 7);
            m.text(
                cv,
                &format!(
                    "{done:.0}/{total:.0} flies  ·  {:.0}s  ·  {} worker threads",
                    q["job"]["elapsed"].as_f64().unwrap_or(0.0),
                    q["job"]["workers"]
                ),
                (x0, y + 20),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            m.button(
                cv,
                Rect::new(rect.right() - 170, y - 8, 146, 36),
                "Cancel",
                Act::Custom("cancel".into()),
                "assay_cancel",
                Style::Danger,
                true,
                None,
                false,
                None,
            );
        } else {
            m.button(
                cv,
                Rect::new(x0, y - 6, 180, 40),
                "Run",
                Act::Custom("run".into()),
                "assay_run",
                Style::Primary,
                true,
                Some("Runs in background worker processes while the game stays paused."),
                false,
                None,
            );
            if let Some(e) = q["job"]["error"].as_str() {
                m.text(cv, e, (x0 + 200, y + 14), BAD, &f.small, Anchor::MidLeft);
            }
            if q["result"].is_object() {
                m.button(
                    cv,
                    Rect::new(x0 + 200, y - 6, 200, 40),
                    "Export results",
                    Act::Custom("export".into()),
                    "assay_export",
                    Style::Normal,
                    true,
                    Some("Save these results as JSON and CSV in the exports folder."),
                    false,
                    None,
                );
            }
        }
        y += 40;
        if q["result"].is_object() && !busy {
            draw_assay_result(m, cv, Rect::new(rect.x + 24, y, rect.w - 48, rect.bottom() - 70 - y), &q["result"]);
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "assays:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let q = ctx.host.query("assay_state");
        let n_opts = q["options"].as_array().map_or(1, Vec::len).max(1);
        match name.split_once(':') {
            Some(("kind", k)) => self.kind = k.into(),
            Some(("surg", d)) => {
                let d: i64 = d.parse().unwrap_or(0);
                self.surgery_i = ((self.surgery_i as i64 + d).rem_euclid(n_opts as i64)) as usize;
            }
            Some(("mode", i)) => self.stimulate = i == "1",
            Some(("slide", "assay_flies")) => {
                self.flies = text.and_then(|t| t.parse::<f64>().ok()).map_or(self.flies, |v| v.round() as i64)
            }
            _ if name == "run" => {
                let opts = q["options"].as_array().cloned().unwrap_or_default();
                let si = self.surgery_i.min(opts.len().saturating_sub(1));
                let specs: Vec<String> = opts
                    .get(si)
                    .and_then(|o| o[1].as_array())
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let mode = if self.stimulate { 1 } else { -1 };
                let surgery: serde_json::Map<String, serde_json::Value> = specs.into_iter().map(|s| (s, json!(mode))).collect();
                let label = opts.get(si).and_then(|o| o[0].as_str()).unwrap_or("none").to_string();
                ctx.host.command("start_assay", json!({"kind": self.kind, "flies": self.flies, "surgery": surgery, "label": label}));
            }
            _ if name == "cancel" => {
                ctx.host.command("cancel_assay", json!({}));
            }
            _ if name == "export" => {
                ctx.host.command("export_assay", json!({}));
            }
            _ => {}
        }
    }
}

/// `f"{x:.2g}"`
fn py_g_2g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let e = x.abs().log10().floor() as i32;
    if !(-5..2).contains(&e) {
        let m = x / 10f64.powi(e);
        let m = format!("{m:.1}").trim_end_matches('0').trim_end_matches('.').to_string();
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let dec = (1 - e).max(0) as usize;
    let s = format!("{x:.dec$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

// --- left/right asymmetry audit and the benchmark ------------------------------------------------------------------------------------
const ASYM_ROWS: [[&str; 11]; 6] = [
    [
        "DNa01",
        "Steering descending command",
        "1 / 1",
        "393 / 402",
        "577 / 577",
        "316 / 311",
        "532 / 532",
        "2.4 / 3.0 Hz",
        "2.2 / 3.2 Hz",
        "+0.60 Hz",
        "+1.00 Hz",
    ],
    [
        "DNa02",
        "Sharp steering command",
        "1 / 1",
        "695 / 716",
        "1084 / 1084",
        "339 / 322",
        "534 / 534",
        "6.0 / 5.6 Hz",
        "5.6 / 6.6 Hz",
        "-0.40 Hz",
        "+1.00 Hz",
    ],
    [
        "LC10",
        "Courtship tracking / fixation",
        "479 / 481",
        "78 / 97",
        "86 / 106",
        "54 / 60",
        "72 / 80",
        "3.8 / 3.9 Hz",
        "4.0 / 4.1 Hz",
        "+0.18 Hz",
        "+0.07 Hz",
    ],
    [
        "LPLC2",
        "Rapid looming escape",
        "94 / 91",
        "232 / 283",
        "243 / 295",
        "93 / 97",
        "125 / 131",
        "10.2 / 6.5 Hz",
        "10.5 / 7.7 Hz",
        "-3.66 Hz",
        "-2.79 Hz",
    ],
    [
        "LC4",
        "Collision looming avoidance",
        "112 / 90",
        "191 / 224",
        "210 / 244",
        "79 / 85",
        "116 / 125",
        "0.7 / 1.1 Hz",
        "0.8 / 0.9 Hz",
        "+0.38 Hz",
        "+0.13 Hz",
    ],
    [
        "DNp01",
        "Giant fiber: escape takeoff",
        "1 / 1",
        "558 / 482",
        "836 / 836",
        "136 / 125",
        "200 / 200",
        "6.6 / 6.2 Hz",
        "7.6 / 8.6 Hz",
        "-0.40 Hz",
        "+1.00 Hz",
    ],
];

/// Lab > Asymmetry audit (`lab/lab.py:page_asymmetry`): the recorded table of the raw and the mirror-averaged connectome. The mirror switch is
/// the setting `brain.mirror_weights` (the game's symmetrised-weights rule; see PORTING.md for what this build does with it).
pub struct LabAsymmetry;

impl Page for LabAsymmetry {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, "LEFT / RIGHT ASYMMETRY AUDIT", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let top = m.wrapped(
            cv,
            "Audit bilateral differences in connectome structure and spontaneous turning bias.",
            (rect.x + 24, m.under_heading(rect).max(rect.y + 50)),
            rect.w - 48,
            LABEL,
            &f.small,
            2,
        ) + 8;
        let body = Rect::new(rect.x + 16, top, rect.w - 32, rect.bottom() - 70 - top);
        let key = "lab_asymmetry";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y + 4 - off;
        let (fs, fb) = (&f.small, &f.bold);
        let lh = fs.linesize;
        let mirror = ctx.cfg.get_bool("brain.mirror_weights");
        let chip_col: Color = if mirror { [240, 160, 60] } else { [100, 180, 240] };
        let chip_txt = if mirror { "GAME RULE: MIRROR-AVERAGED" } else { "CONNECTOME: RAW DATA" };
        let bias_txt = if mirror { "+1.00 Hz (symmetric weights)" } else { "+0.10 Hz (right turn bias)" };
        let expl = "In the raw MaleCNS v1.0 connectome, bilateral asymmetries arise from both true biology and uneven electron microscopy (EM) reconstruction and proofreading depth between hemispheres. Descending steering neurons DNa01 and DNa02 drive a mild spontaneous rightward turning bias in quiet walking. Mirror-averaging synaptic weights across the 77,507 bilateral left/right pairs (155,014 neurons) enforces exact structural symmetry. Because this alters real connectome data, it is tagged strictly as a Game Rule.";
        let cw = body.w - 12;
        let n_expl = Menu::fit_lines(fs, expl, cw - 28, 99).0.len() as i32;
        let chip_w = fs.size(chip_txt) + 16;
        let bias_line = format!("Baseline turning bias: {bias_txt}");
        let bias_below = chip_w + 24 + fb.size(&bias_line) > cw - 14;
        let card_h = 10 + lh + 6 + if bias_below { fb.linesize + 6 } else { 0 } + n_expl * lh + 12;
        let card = Rect::new(body.x, y, cw, card_h);
        cv.fill_rect(card, [24, 28, 38], 8);
        cv.stroke_rect(card, [45, 52, 68], 8);
        let chip = Rect::new(card.x + 12, card.y + 10, chip_w, lh + 4);
        cv.fill_rect(chip, [15, 18, 26], 4);
        cv.stroke_rect(chip, chip_col, 4);
        m.text(cv, chip_txt, chip.center(), chip_col, fs, Anchor::Center);
        let yy = if bias_below {
            m.text(cv, &bias_line, (card.x + 14, chip.bottom() + 6), INK, fb, Anchor::TopLeft);
            chip.bottom() + 6 + fb.linesize + 6
        } else {
            m.text(cv, &bias_line, (chip.right() + 14, chip.center().1), INK, fb, Anchor::MidLeft);
            chip.bottom() + 6
        };
        m.wrapped(cv, expl, (card.x + 14, yy), cw - 28, TEXT, fs, 99);
        y += card_h + 10;
        y = m.wrapped(
            cv,
            &format!(
                "Recorded from python kick_the_fly.py --headless --audit-asymmetry{} (seed 0, a 5 s calm run, CPU backend). Partners: the number of distinct pre- (in) or postsynaptic (out) partner neurons, averaged over each side's neurons of the type; not synapse counts.",
                if mirror { " --mirror-weights" } else { "" }
            ),
            (body.x + 4, y),
            cw - 8,
            LABEL,
            fs,
            6,
        ) + 8;
        let heads = ["Count L/R", "In-partners L/R", "Out-partners L/R", "Calm rate L/R", "Diff (R-L)"];
        let pick = |r: &[&'static str; 11], a: usize, b: usize| if mirror { r[b] } else { r[a] };
        let cells: Vec<[&str; 5]> = ASYM_ROWS.iter().map(|r| [r[2], pick(r, 3, 4), pick(r, 5, 6), pick(r, 7, 8), pick(r, 9, 10)]).collect();
        let name_w = ASYM_ROWS.iter().map(|r| fb.size(r[0])).max().unwrap_or(0).max(fb.size("Type")) + 16;
        let col_w: Vec<i32> = (0..5).map(|i| fs.size(heads[i]).max(cells.iter().map(|c| fs.size(c[i])).max().unwrap_or(0)) + 16).collect();
        let role_max = ASYM_ROWS.iter().map(|r| fs.size(r[1])).max().unwrap_or(0) + 16;
        let role_inline = name_w + role_max + col_w.iter().sum::<i32>() <= cw - 28;
        let role_w = if role_inline { role_max } else { 0 };
        let spare = ((cw - 28) - (name_w + role_w + col_w.iter().sum::<i32>())).max(0);
        let mut xs = Vec::new();
        let mut x = 14 + name_w + role_w;
        for w in &col_w {
            xs.push(x);
            x += w + spare / 5;
        }
        let hdr = Rect::new(body.x, y, cw, 32.max(lh + 12));
        cv.fill_rect(hdr, [32, 38, 52], 6);
        m.text(cv, "Type", (hdr.x + 14, hdr.center().1), INK, fb, Anchor::MidLeft);
        if role_inline {
            m.text(cv, "Functional role", (hdr.x + 14 + name_w, hdr.center().1), LABEL, fs, Anchor::MidLeft);
        }
        for (xx, h) in xs.iter().zip(heads) {
            m.text(cv, h, (hdr.x + xx, hdr.center().1), LABEL, fs, Anchor::MidLeft);
        }
        y += hdr.h + 6;
        let row_h = if role_inline { 36.max(fb.linesize + 12) } else { fb.linesize + lh + 14 };
        for (r, c) in ASYM_ROWS.iter().zip(&cells) {
            let rbox = Rect::new(body.x, y, cw, row_h);
            cv.fill_rect(rbox, [20, 24, 33], 6);
            let cy = if role_inline {
                m.text(cv, r[0], (rbox.x + 14, rbox.center().1), INK, fb, Anchor::MidLeft);
                m.text(cv, r[1], (rbox.x + 14 + name_w, rbox.center().1), [150, 180, 220], fs, Anchor::MidLeft);
                rbox.center().1
            } else {
                m.text(cv, r[0], (rbox.x + 14, rbox.y + 6), INK, fb, Anchor::TopLeft);
                m.text(cv, r[1], (rbox.x + 14, rbox.y + 6 + fb.linesize), [150, 180, 220], fs, Anchor::TopLeft);
                rbox.y + 6 + fb.linesize / 2
            };
            for (i, (xx, v)) in xs.iter().zip(c).enumerate() {
                let mut col = TEXT;
                if i == c.len() - 1 {
                    let val: f64 = v.replace(" Hz", "").parse().unwrap_or(0.0);
                    col = if val.abs() < 0.2 { GOOD } else { [240, 160, 60] };
                }
                m.text(cv, v, (rbox.x + xx, cy), col, fs, Anchor::MidLeft);
            }
            y += row_h + 6;
        }
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.end_clip(cv);
        let btn = if mirror { "Mirror weights: ON [Rule]" } else { "Mirror weights: OFF [Raw]" };
        let bw = 250.max(fb.size(btn) + 32);
        m.button(
            cv,
            Rect::new(rect.x + 24, rect.bottom() - 58, bw, 42),
            btn,
            Act::Custom("mirror".into()),
            "asymmetry:toggle_mirror",
            Style::Normal,
            true,
            Some("Toggle bilateral weight symmetrization [GAME RULE]"),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "asymmetry:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if name == "mirror" {
            let on = !ctx.cfg.get_bool("brain.mirror_weights");
            m.set_setting(ctx.cfg, "brain.mirror_weights", &json!(on), true);
        }
    }
}

/// Lab > Simulation benchmark (`lab/lab.py:page_benchmark`).
///
/// Host: query `benchmark_state` = `{results: <benchmark.run dict + timestamp> | null, system: {...}, backend, device, status: string | null}`;
/// command `start_benchmark`.
pub struct LabBenchmark;

impl Page for LabBenchmark {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("benchmark_state");
        m.text(cv, "SIMULATION BENCHMARK", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let top = (rect.y + 80).max(m.subtitle(
            cv,
            rect,
            "Throughput, real-time pace, neurons/sec and memory footprint across 1, 8, and 16 flies.",
            Some(rect.y + 50),
            2,
        ));
        let body = Rect::new(rect.x + 16, top, rect.w - 32, rect.bottom() - 70 - top);
        let key = "lab_benchmark";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y + 4 - off;
        let res = &q["results"];
        let sys = if res.is_object() { &res["system"] } else { &q["system"] };
        let s = |v: &serde_json::Value, k: &str| v[k].as_str().map_or_else(|| v[k].to_string(), String::from);
        let backend = res["backend"].as_str().or(q["backend"].as_str()).unwrap_or("CPU (NumPy)").to_string();
        let device = res["device"].as_str().or(q["device"].as_str()).unwrap_or("CPU").to_string();
        let lines: Vec<(String, Color, &crate::font::Font)> = vec![
            (format!("CPU: {} ({} threads)", s(sys, "cpu_model"), sys["cpu_count"]), INK, &f.bold),
            (format!("OS: {}  ·  Python: {}", s(sys, "os"), s(sys, "python")), TEXT, &f.small),
            (format!("Sim Backend: {backend}  ·  Device: {device}"), [120, 220, 240], &f.small),
            ("Connectome: Janelia MaleCNS v1.0 (166,700 neurons, 10,272,125 connections)".to_string(), [140, 180, 220], &f.small),
        ];
        let cw = body.w - 12;
        let card_h = 20 + lines.iter().map(|(t, _, fo)| Menu::fit_lines(fo, t, cw - 28, 3).0.len() as i32 * fo.linesize + 2).sum::<i32>();
        let card = Rect::new(body.x, y, cw, card_h);
        cv.fill_rect(card, [24, 28, 38], 8);
        cv.stroke_rect(card, [45, 52, 68], 8);
        let mut yy = card.y + 10;
        for (t, col, fo) in &lines {
            yy = m.wrapped(cv, t, (card.x + 14, yy), cw - 28, *col, fo, 3) + 2;
        }
        y += card_h + 16;
        if let Some(records) = res["records"].as_array() {
            let hdr = Rect::new(body.x, y, body.w - 12, 32);
            cv.fill_rect(hdr, [32, 38, 52], 6);
            let cy = hdr.center().1;
            m.text(cv, "Flies", (hdr.x + 14, cy), INK, &f.bold, Anchor::MidLeft);
            for (x, label) in [(80, "Paced Rate"), (230, "Real-Time"), (340, "Uncapped Rate"), (490, "Max Speedup"), (620, "Throughput")] {
                m.text(cv, label, (hdr.x + x, cy), LABEL, &f.small, Anchor::MidLeft);
            }
            m.text(cv, "Memory", (hdr.right() - 14, cy), LABEL, &f.small, Anchor::MidRight);
            y += 32 + 6;
            for r in records {
                let rbox = Rect::new(body.x, y, body.w - 12, 36);
                cv.fill_rect(rbox, [20, 24, 33], 6);
                let cy = rbox.center().1;
                let g = |k: &str| r[k].as_f64().unwrap_or(0.0);
                m.text(cv, &format!("{} flies", r["flies"]), (rbox.x + 14, cy), INK, &f.bold, Anchor::MidLeft);
                m.text(cv, &format!("{:.1} steps/s", g("paced_steps_per_s")), (rbox.x + 80, cy), TEXT, &f.small, Anchor::MidLeft);
                let rt = g("paced_realtime_ratio");
                let rt_col = if rt >= 0.99 {
                    GOOD
                } else if rt >= 0.8 {
                    [240, 160, 60]
                } else {
                    BAD
                };
                m.text(cv, &format!("{rt:.2}x"), (rbox.x + 230, cy), rt_col, &f.bold, Anchor::MidLeft);
                let lat = r["uncapped_latency_ms"]
                    .as_f64()
                    .unwrap_or_else(|| (1000.0 / g("uncapped_steps_per_s").max(1e-6) * 100.0).round() / 100.0);
                m.text(
                    cv,
                    &format!("{:.1} st/s ({lat:.2}ms)", g("uncapped_steps_per_s")),
                    (rbox.x + 340, cy),
                    TEXT,
                    &f.small,
                    Anchor::MidLeft,
                );
                m.text(
                    cv,
                    &format!("{:.2}x", g("uncapped_realtime_ratio")),
                    (rbox.x + 490, cy),
                    [140, 200, 240],
                    &f.small,
                    Anchor::MidLeft,
                );
                m.text(
                    cv,
                    &format!("{:.1} M neurons/s", g("neurons_per_sec") / 1e6),
                    (rbox.x + 620, cy),
                    [170, 230, 190],
                    &f.small,
                    Anchor::MidLeft,
                );
                m.text(cv, &format!("{:.0} MB", g("memory_mb")), (rbox.right() - 14, cy), LABEL, &f.small, Anchor::MidRight);
                y += 36 + 6;
            }
            m.text(
                cv,
                &format!(
                    "Measured at {}  ·  {:.0}s per condition  ·  Target: 200 steps/s",
                    res["timestamp"].as_str().unwrap_or(""),
                    res["seconds_per_run"].as_f64().unwrap_or(3.0)
                ),
                (body.x + 12, y + 10),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            y += 36;
        } else {
            m.wrapped(
                cv,
                "No benchmark results found. Click 'Run Benchmark' to measure multi-fly simulation throughput.",
                (body.x + 14, y + 10),
                body.w - 40,
                TEXT,
                &f.text,
                3,
            );
            y += 40;
        }
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.end_clip(cv);
        if let Some(status) = q["status"].as_str() {
            m.text(cv, &format!("Benchmarking... {status}"), (rect.x + 24, rect.bottom() - 48), AMBER, &f.bold, Anchor::TopLeft);
        } else {
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 58, 220, 42),
                "Run Benchmark",
                Act::Custom("run".into()),
                "benchmark:run",
                Style::Normal,
                true,
                Some("Measure simulation throughput across 1, 8, 16 flies"),
                false,
                None,
            );
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "benchmark:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if name == "run" {
            ctx.host.command("start_benchmark", json!({}));
        }
    }
}
