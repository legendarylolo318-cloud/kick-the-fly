//! Lab > Psychometric curve generator (`lab/labpsych.py`): sweep a parameter across headless trials; chart, table, CSV/SVG/PDF export.
//!
//! Host: query `lab.psych_state` = `{job: {frac, note} | null, result, msg, base_seed}`; commands `lab.psych_run {target, flies}`,
//! `lab.psych_export`.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const TARGETS: [(&str, &str); 3] = [("looming", "Looming speed"), ("sugar", "Sugar dose"), ("tmaze", "T-maze pairings")];

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

pub struct LabPsych {
    target: &'static str,
    flies: i64,
}

impl Default for LabPsych {
    fn default() -> Self {
        Self { target: "looming", flies: 3 }
    }
}

impl LabPsych {
    pub fn set_view(&mut self, target: &str, flies: i64) {
        if let Some(t) = TARGETS.iter().find(|t| t.0 == target) {
            self.target = t.0;
        }
        self.flies = flies;
    }
}

impl Page for LabPsych {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.psych_state");
        m.text(cv, "PSYCHOMETRIC CURVE GENERATOR", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let mut y = (rect.y + 80).max(m.subtitle(
            cv,
            rect,
            "Sweep experimental parameters across headless trials with error bars. Exports publication-quality SVG, PDF, and CSV.",
            None,
            2,
        ));
        let job = &q["job"];
        let busy = job.is_object();
        let lw = ["Assay target:", "Flies / point:"].iter().map(|t| f.text.size(t)).max().unwrap_or(0) + 20;
        m.text(cv, "Assay target:", (rect.x + 24, y + 6), TEXT, &f.text, Anchor::TopLeft);
        let idx = TARGETS.iter().position(|t| t.0 == self.target).unwrap_or(0);
        m.segmented(
            cv,
            Rect::new(rect.x + 24 + lw, y, 620.min(rect.right() - 48 - lw), 32),
            &TARGETS.iter().map(|t| t.1.to_string()).collect::<Vec<_>>(),
            idx,
            &TARGETS.iter().map(|t| Act::Custom(format!("target:{}", t.0))).collect::<Vec<_>>(),
            "psych_target",
            !busy,
            "",
        );
        y += 44;
        m.text(cv, "Flies / point:", (rect.x + 24, y + 6), TEXT, &f.text, Anchor::TopLeft);
        m.page_slider(
            cv,
            Rect::new(rect.x + 24 + lw, y, 260, 32),
            self.flies as f64,
            (2.0, 8.0, 1.0),
            "{:.0f} flies",
            "psych_flies",
            !busy,
            Some("Number of seeds to run per point (more gives tighter error bars)"),
        );
        m.text(
            cv,
            &format!("Base seed: {}", q["base_seed"].as_i64().unwrap_or(1000)),
            (rect.x + 24 + lw + 280, y + 16),
            LABEL,
            &f.small,
            Anchor::MidLeft,
        );
        y += 44;
        if busy {
            let frac = num(&job["frac"]).clamp(0.0, 1.0);
            let bar_w = rect.w - 240;
            cv.fill_rect(Rect::new(rect.x + 24, y, bar_w, 14), [30, 36, 48], 7);
            cv.fill_rect(Rect::new(rect.x + 24, y, 8.max((f64::from(bar_w) * frac) as i32), 14), AMBER, 7);
            m.text(
                cv,
                &format!("Running trials: {} ({:.0}%)", job["note"].as_str().unwrap_or(""), frac * 100.0),
                (rect.x + 24, y + 22),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
        } else {
            m.button(
                cv,
                Rect::new(rect.x + 24, y, 190, 38),
                "Run Sweep",
                Act::Custom("run".into()),
                "psych_run",
                Style::Primary,
                true,
                Some("Runs the parameter sweep across headless brains in background."),
                false,
                None,
            );
            if q["result"].is_object() {
                m.button(
                    cv,
                    Rect::new(rect.x + 230, y, m.bw("Export CSV, SVG & PDF", 240), 38),
                    "Export CSV, SVG & PDF",
                    Act::Custom("export".into()),
                    "psych_export_all",
                    Style::Normal,
                    true,
                    Some("Export publication-ready vector figures and CSV."),
                    false,
                    None,
                );
            }
            if let Some(msg) = q["msg"].as_str().filter(|s| !s.is_empty()) {
                m.text(cv, msg, (rect.x + 490, y + 20), GOOD, &f.small, Anchor::MidLeft);
            }
        }
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, &format!("error: {e}"), (rect.x + 24, y + 48), BAD, &f.small, Anchor::TopLeft);
        }
        y += 54;
        let res = &q["result"];
        if res.is_object() && !busy {
            self.chart(m, cv, rect, y, res);
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "psych:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        if let Some(t) = name.strip_prefix("target:") {
            self.set_view(t, self.flies);
        } else if name == "slide:psych_flies" {
            if let Some(v) = text.and_then(|t| t.parse::<f64>().ok()) {
                self.flies = v.round() as i64;
            }
        } else if name == "run" {
            ctx.host.command("lab.psych_run", json!({"target": self.target, "flies": self.flies}));
        } else if name == "export" {
            ctx.host.command("lab.psych_export", json!({"target": self.target}));
        }
    }
}

impl LabPsych {
    fn chart(&self, m: &mut Menu, cv: &mut Canvas, rect: Rect, y: i32, res: &Value) {
        let f = m.fonts();
        let spec = &res["spec"];
        let pts: Vec<&Value> = res["points"].as_array().into_iter().flatten().collect();
        let chart_w = (rect.w - 60) / 2;
        let cr = Rect::new(rect.x + 24, y, chart_w, rect.bottom() - y - 70);
        cv.fill_rect(cr, [20, 24, 34], 8);
        cv.stroke_rect(cr, [40, 48, 64], 8);
        let (p_l, p_r, p_t, p_b) = (cr.x + 50, cr.right() - 20, cr.y + 30, cr.bottom() - 40);
        let (pw, ph) = (p_r - p_l, p_b - p_t);
        let xs: Vec<f64> = pts.iter().map(|p| num(&p["x"])).collect();
        let x_min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let x_max = if mx > x_min { mx } else { x_min + 1.0 };
        let y_max = spec["y_max"].as_f64().unwrap_or(1.0);
        for i in 0..5 {
            let y_val = f64::from(i) * y_max / 4.0;
            let py = p_b - ((y_val / y_max) * f64::from(ph)) as i32;
            cv.line((p_l, py), (p_r, py), 1, [35, 42, 56], 255);
            m.text(cv, &format!("{y_val:.2}"), (p_l - 6, py), LABEL, &f.small, Anchor::MidRight);
        }
        let mut poly: Vec<(i32, i32)> = Vec::new();
        for pt in &pts {
            let px = p_l + ((num(&pt["x"]) - x_min) / (x_max - x_min) * f64::from(pw)) as i32;
            let pm = num(&pt["primary"]["mean"]);
            let py = p_b - ((pm / y_max).clamp(0.0, 1.0) * f64::from(ph)) as i32;
            poly.push((px, py));
            let ci = num(&pt["primary"]["ci95"]);
            let y_hi = p_b - (((pm + ci) / y_max).clamp(0.0, 1.0) * f64::from(ph)) as i32;
            let y_lo = p_b - (((pm - ci) / y_max).clamp(0.0, 1.0) * f64::from(ph)) as i32;
            let c = [100, 160, 240];
            cv.line((px, y_lo), (px, y_hi), 2, c, 255);
            cv.line((px - 4, y_hi), (px + 4, y_hi), 2, c, 255);
            cv.line((px - 4, y_lo), (px + 4, y_lo), 2, c, 255);
        }
        if poly.len() > 1 {
            for w in poly.windows(2) {
                cv.line(w[0], w[1], 2, [60, 140, 240], 255);
            }
        }
        for &(px, py) in &poly {
            cv.circle((px, py), 4, [220, 240, 255]);
        }
        m.text(cv, spec["param_label"].as_str().unwrap_or(""), (p_l + pw / 2, p_b + 20), LABEL, &f.small, Anchor::Center);
        m.text(cv, spec["metric_primary_label"].as_str().unwrap_or(""), (p_l, p_t - 14), INK, &f.bold, Anchor::MidLeft);
        let tbl_x = cr.right() + 20;
        let mut ty = y;
        m.text(cv, "MEASURED VALUES", (tbl_x, ty), INK, &f.bold, Anchor::TopLeft);
        ty += 24;
        m.text(cv, spec["param_name"].as_str().unwrap_or(""), (tbl_x, ty), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, "Mean \u{b1} SEM", (tbl_x + 120, ty), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, "95% CI", (tbl_x + 240, ty), LABEL, &f.small, Anchor::TopLeft);
        ty += 18;
        for pt in &pts {
            let p = &pt["primary"];
            let (mean, sem, ci) = (num(&p["mean"]), num(&p["sem"]), num(&p["ci95"]));
            m.text(cv, &crate::pages::day4::fmt_g_pub(num(&pt["x"])), (tbl_x, ty), TEXT, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{mean:.3} \u{b1} {sem:.3}"), (tbl_x + 120, ty), INK, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("[{:.3}, {:.3}]", mean - ci, mean + ci), (tbl_x + 240, ty), LABEL, &f.small, Anchor::TopLeft);
            ty += 20;
        }
        m.text(cv, res["caption"].as_str().unwrap_or(""), (rect.x + 24, rect.bottom() - 56), [140, 150, 170], &f.small, Anchor::TopLeft);
    }
}
