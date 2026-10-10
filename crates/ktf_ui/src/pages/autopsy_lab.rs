//! Lab > Failure autopsy (`lab/labautopsy.py`): one page per failing validation test: where the driven signal goes and where it fades.
//!
//! Host: query `lab.autopsy_state` = `{fails: [{id, name}] | null, error, done: {id: autopsy}, job: {frac, note} | null}`; commands
//! `lab.autopsy_run {only: [id] | null}`, `lab.autopsy_export`.

use serde_json::{Value, json};

use super::day4::fmt_gp;
use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const TAG: &str = "MODEL PREDICTION";

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// `failure_autopsy.pct`.
fn pct(x: f64) -> String {
    if x.is_nan() {
        "n/a".into()
    } else if x == 0.0 {
        "0%".into()
    } else {
        format!("{}%", fmt_gp(100.0 * x, 3))
    }
}

fn comma(x: i64) -> String {
    super::day4::grp(x as f64, 0)
}

#[derive(Default)]
pub struct LabAutopsy {
    sel: Option<String>,
}

impl Page for LabAutopsy {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.autopsy_state");
        m.text(cv, "FAILURE AUTOPSY", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        m.chip(cv, (rect.x + 40 + f.head.size("FAILURE AUTOPSY"), rect.y + 22), TAG);
        let y = m.subtitle(
            cv,
            rect,
            "For each validation test that fails: the paths from its input to its target, how much of the target's input they can supply, how much of it excites or inhibits, and what each stage did in the run. Read-only: it never changes or suggests changing the model.",
            None,
            2,
        );
        let fails: Vec<(String, String)> = q["fails"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| (t["id"].as_str().unwrap_or("").to_string(), t["name"].as_str().unwrap_or("").to_string()))
            .collect();
        if self.sel.is_none() {
            self.sel = fails.first().map(|f| f.0.clone());
        }
        let left = Rect::new(rect.x + 16, y, 330, rect.bottom() - y - 66);
        let mut right = Rect::new(left.right() + 10, y, rect.right() - left.right() - 26, left.h);
        let job = &q["job"];
        if job.is_object() {
            let frac = num(&job["frac"]).clamp(0.0, 1.0);
            cv.fill_rect(Rect::new(right.x, y, right.w - 8, 10), [30, 36, 48], 5);
            cv.fill_rect(Rect::new(right.x, y, 8.max(((right.w - 8) as f64 * frac) as i32), 10), AMBER, 5);
            m.text(
                cv,
                &format!("{}  ·  {:.0}s", job["note"].as_str().unwrap_or(""), num(&job["elapsed"])),
                (right.x, y + 14),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            right = Rect::new(right.x, y + 34, right.w, right.h - 34);
        }
        // the list of failing tests
        cv.fill_rect(left, [14, 17, 24], 10);
        let mut ly = left.y + 8;
        m.text(cv, &format!("{} failing test(s)", fails.len()), (left.x + 10, ly), LABEL, &f.small, Anchor::TopLeft);
        ly += 24;
        for (id, name) in &fails {
            let r = Rect::new(left.x + 4, ly - 2, left.w - 8, 44);
            let sel = self.sel.as_deref() == Some(id);
            let over = m.click_area(r, Act::Custom(format!("row:{id}")), &format!("fa:row:{id}"), true, Some(name));
            if sel || over {
                cv.fill_rect(r, if sel { [44, 52, 68] } else { ROW_HOVER }, 6);
            }
            let done = q["done"].get(id).is_some();
            let short: String = name.chars().take(40).collect();
            m.text(cv, &short, (left.x + 12, ly), if done { TEXT } else { LABEL }, &f.small, Anchor::TopLeft);
            m.text(
                cv,
                if done { "autopsied" } else { "not yet" },
                (left.x + 12, ly + 20),
                if done { GOOD } else { DIM },
                &f.small,
                Anchor::TopLeft,
            );
            ly += 48;
        }
        let err = q["error"].as_str().filter(|e| !e.is_empty());
        if let (Some(e), true) = (err, fails.is_empty()) {
            m.wrapped(cv, e, (right.x + 8, right.y + 8), right.w - 16, BAD, &f.small, 4);
        } else if let Some(sel) = &self.sel {
            match q["done"].get(sel).filter(|d| d.is_object()) {
                None => {
                    m.text(cv, "Not autopsied yet.", (right.x + 8, right.y + 8), LABEL, &f.small, Anchor::TopLeft);
                }
                Some(a) => self.document(m, cv, right, a),
            }
        }
        if !fails.is_empty() {
            let w = m.bw("Autopsy this failure", 220);
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 58, w, 42),
                "Autopsy this failure",
                Act::Custom("one".into()),
                "fa:one",
                Style::Primary,
                !job.is_object() && self.sel.is_some(),
                Some("Runs three seeds of the brain (about 15 s) and reads the wiring. Read-only."),
                false,
                None,
            );
            let w2 = m.bw("Autopsy all", 150);
            m.button(
                cv,
                Rect::new(rect.x + 36 + w, rect.bottom() - 58, w2, 42),
                "Autopsy all",
                Act::Custom("all".into()),
                "fa:all",
                Style::Normal,
                !job.is_object(),
                None,
                false,
                None,
            );
            m.button(
                cv,
                Rect::new(rect.x + 48 + w + w2, rect.bottom() - 58, m.bw("Export", 130), 42),
                "Export",
                Act::Custom("export".into()),
                "fa:export",
                Style::Normal,
                q["done"].as_object().is_some_and(|d| !d.is_empty()),
                Some("Copies each page (Markdown and JSON) into a new folder in exports."),
                false,
                None,
            );
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "fa:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if let Some(id) = name.strip_prefix("row:") {
            self.sel = Some(id.to_string());
            return;
        }
        match name {
            "one" => {
                if let Some(s) = &self.sel {
                    ctx.host.command("lab.autopsy_run", json!({"only": [s]}));
                }
            }
            "all" => drop(ctx.host.command("lab.autopsy_run", json!({"only": null}))),
            "export" => drop(ctx.host.command("lab.autopsy_export", Value::Null)),
            _ => {}
        }
    }
}

impl LabAutopsy {
    pub fn set_sel(&mut self, id: &str) {
        self.sel = Some(id.to_string());
    }

    fn document(&self, m: &mut Menu, cv: &mut Canvas, area: Rect, a: &Value) {
        let f = m.fonts();
        let key = "lab_autopsy";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, area);
        let (x, mut y) = (area.x + 8, area.y + 4 - off);
        let lh = f.small.linesize + 3;
        let t = &a["test"];
        let line = |m: &mut Menu, cv: &mut Canvas, text: &str, col: Color, font: &crate::font::Font, dx: i32, gap: i32, y: &mut i32| {
            *y = m.wrapped(cv, text, (x + dx, *y), area.w - 24 - dx, col, font, 6) + gap;
        };
        line(m, cv, t["name"].as_str().unwrap_or(""), INK, &f.bold, 0, 2, &mut y);
        if let Some(e) = a["error"].as_str() {
            line(m, cv, e, BAD, &f.small, 0, 0, &mut y);
        } else {
            line(m, cv, &format!("Claim: {}", t["claim"].as_str().unwrap_or("")), LABEL, &f.small, 0, 0, &mut y);
            if let Some(r) = a["result"].as_object().filter(|_| a["result"].is_object()) {
                let keys = ["drive_ratio_mean", "control_ratio_mean", "mean_contrast", "mean_persistence_ms", "p_value", "n"];
                let ms: Vec<String> = r
                    .get("measured")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flatten()
                    .filter(|(k, v)| v.is_number() && keys.contains(&k.as_str()))
                    .map(|(k, v)| format!("{k} {}", fmt_gp(num(v), 3)))
                    .collect();
                line(
                    m,
                    cv,
                    &format!("FAIL. {}   {}", r.get("criteria").and_then(Value::as_str).unwrap_or(""), ms.join(", ")),
                    BAD,
                    &f.small,
                    0,
                    2,
                    &mut y,
                );
            }
            for s in a["sections"].as_array().into_iter().flatten() {
                y += 6;
                line(m, cv, s["label"].as_str().unwrap_or(""), AMBER, &f.bold, 0, 0, &mut y);
                if let Some(e) = s["error"].as_str() {
                    line(m, cv, e, BAD, &f.small, 0, 0, &mut y);
                    continue;
                }
                line(
                    m,
                    cv,
                    &format!(
                        "input {} ({})   target {} ({})   direct synapses {}",
                        s["input"].as_str().unwrap_or(""),
                        s["n_input"],
                        s["target"].as_str().unwrap_or(""),
                        s["n_target"],
                        s["direct_synapses"]
                    ),
                    LABEL,
                    &f.small,
                    0,
                    0,
                    &mut y,
                );
                line(m, cv, "share of the target's input along walks of exactly h synapses", DIM, &f.small, 0, 0, &mut y);
                let budget = s["budget"].as_array().cloned().unwrap_or_default();
                let max_total = budget.iter().map(|b| num(&b["total"])).fold(1e-12, f64::max);
                for b in &budget {
                    let w = area.w - 470;
                    let x0 = x + 160;
                    let hop = b["hop"].as_i64().unwrap_or(0);
                    m.text(cv, &format!("{hop} synapse{}", if hop > 1 { "s" } else { "" }), (x, y), TEXT, &f.small, Anchor::TopLeft);
                    let ew = (f64::from(w) * num(&b["excitatory"]) / max_total) as i32;
                    let iw = (f64::from(w) * num(&b["inhibitory"]) / max_total) as i32;
                    cv.fill_rect(Rect::new(x0, y + 3, ew.max(1), 11), [60, 170, 110], 0);
                    cv.fill_rect(Rect::new(x0 + ew.max(1), y + 3, iw.max(1), 11), [200, 80, 70], 0);
                    m.text(
                        cv,
                        &format!("{} excites, {} inhibits", pct(num(&b["excitatory"])), pct(num(&b["inhibitory"]))),
                        (x0 + w + 8, y),
                        LABEL,
                        &f.small,
                        Anchor::TopLeft,
                    );
                    y += lh;
                }
                y += 4;
                line(m, cv, "strongest routes (green: stage ratio met the test's 1.5x, red: below it)", DIM, &f.small, 0, 0, &mut y);
                for (i, rt) in s["routes"].as_array().into_iter().flatten().enumerate() {
                    let mut chain = vec!["input".to_string()];
                    chain.extend(rt["types"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(String::from)));
                    chain.push("target".into());
                    line(
                        m,
                        cv,
                        &format!(
                            "{}. {}: {} of the target's input, net {}",
                            i + 1,
                            chain.join(" > "),
                            pct(num(&rt["share"])),
                            if rt["net_sign"].as_i64().unwrap_or(1) > 0 { "excites" } else { "inhibits" }
                        ),
                        TEXT,
                        &f.small,
                        0,
                        1,
                        &mut y,
                    );
                    let stages = rt["stage_ratios"].as_array().cloned().unwrap_or_default();
                    for (st, e) in stages.iter().skip(1).zip(rt["edges"].as_array().into_iter().flatten()) {
                        let ratio = num(&st["ratio"]);
                        let bw = ((ratio / 4.0).min(1.0) * 110.0) as i32;
                        let col = if ratio >= 1.5 { [60, 170, 110] } else { [200, 80, 70] };
                        let name: String = st["stage"].as_str().unwrap_or("").chars().take(22).collect();
                        m.text(cv, &name, (x + 18, y), LABEL, &f.small, Anchor::TopLeft);
                        cv.fill_rect(Rect::new(x + 180, y + 3, 110, 11), [30, 36, 48], 0);
                        cv.fill_rect(Rect::new(x + 180, y + 3, bw.max(2), 11), col, 0);
                        let xl = x + 180 + (110.0 * 1.5 / 4.0) as i32;
                        cv.line((xl, y + 1), (xl, y + 15), 1, AMBER, 255);
                        m.text(
                            cv,
                            &format!(
                                "x{ratio:.2}  {:.1} to {:.1} Hz   {} syn ({}+ {}-)",
                                num(&st["base_hz"]),
                                num(&st["driven_hz"]),
                                comma(e["synapses"].as_i64().unwrap_or(0)),
                                comma(e["excitatory"].as_i64().unwrap_or(0)),
                                comma(e["inhibitory"].as_i64().unwrap_or(0))
                            ),
                            (x + 300, y),
                            DIM,
                            &f.small,
                            Anchor::TopLeft,
                        );
                        y += lh;
                    }
                }
                let v = &s["verdict"];
                line(
                    m,
                    cv,
                    &format!("Where it fades: {}", v["text"].as_str().unwrap_or("")),
                    if v["stage"].is_null() { GOOD } else { BAD },
                    &f.bold,
                    0,
                    4,
                    &mut y,
                );
            }
            y += 6;
            line(m, cv, &format!("Read-only. {}", a["method"].as_str().unwrap_or("")), DIM, &f.small, 0, 0, &mut y);
        }
        m.set_content_h(key, (y + off - area.bottom() + 8).max(0));
        m.end_clip(cv);
    }
}
