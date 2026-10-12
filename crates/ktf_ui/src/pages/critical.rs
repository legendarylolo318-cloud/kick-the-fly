//! Lab > Critical path finder (`lab/labcritical.py`): pick a behavior, rank the cell types it depends on, apply a lesion.
//!
//! Host: query `lab.critical_state` = `{job: {label, frac, note, elapsed} | null, error, results: {target: result}, type_ops: [type]}`;
//! commands `lab.cp_run {target, top}`, `lab.cp_apply {type}`, `lab.cp_export {target}`, `lab.day4_cancel`.

use serde_json::{Value, json};

use super::day4::grp;
use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const TARGETS: [(&str, &str); 7] = [
    ("looming_escape", "Looming -> giant fiber"),
    ("sugar_feeding", "Sugar -> MN9"),
    ("antenna_grooming_circuit", "Antennal touch -> aDN"),
    ("mb_conditioning", "T-maze conditioning"),
    ("looming", "Assay: looming escape"),
    ("sugar", "Assay: sugar response"),
    ("tmaze", "Assay: T-maze"),
];

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn f2(x: f64) -> String {
    if x.is_nan() { "nan".into() } else { format!("{x:.2}") }
}

fn signed_pct(x: f64) -> String {
    if x.is_nan() { "nan%".into() } else { format!("{:+.0}%", x * 100.0) }
}

pub struct LabCritical {
    target: &'static str,
    top: i64,
}

impl Default for LabCritical {
    fn default() -> Self {
        Self { target: "looming_escape", top: 15 }
    }
}

impl Page for LabCritical {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.critical_state");
        m.text(cv, "CRITICAL PATH FINDER", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let mut y = m.subtitle(
            cv,
            rect,
            "Silence one cell type at a time, re-run the behavior, and rank the types by what it did. Every lesion is compared with an unperturbed fly of the same seed.",
            None,
            2,
        );
        let items: Vec<_> = TARGETS
            .iter()
            .map(|(k, label)| (label.to_string(), Act::Custom(format!("target:{k}")), format!("cp:{k}"), self.target == *k, None))
            .collect();
        y = m.flow_buttons(cv, rect.x + 24, y, rect.right() - 24, &items, None, None, 8, None) - 30;
        y += 42;
        let job = &q["job"];
        let running = job.is_object();
        if running {
            cv.fill_rect(Rect::new(rect.x + 24, y + 10, rect.w - 300, 12), [30, 36, 48], 6);
            let frac = num(&job["frac"]).clamp(0.0, 1.0);
            cv.fill_rect(Rect::new(rect.x + 24, y + 10, 8.max(((rect.w - 300) as f64 * frac) as i32), 12), AMBER, 6);
            m.text(
                cv,
                &format!("{}  ·  {:.0}s", job["note"].as_str().unwrap_or(""), num(&job["elapsed"])),
                (rect.x + 24, y + 26),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            m.text(
                cv,
                "interrupting is safe: finished types are written as they go and --resume picks them up",
                (rect.x + 24, y + 44),
                DIM,
                &f.small,
                Anchor::TopLeft,
            );
        } else {
            let label = format!("Run on the top {} types", self.top);
            m.button(
                cv,
                Rect::new(rect.x + 24, y, m.bw(&label, 250), 36),
                &label,
                Act::Custom("run".into()),
                "cp_run",
                Style::Primary,
                true,
                Some("Each type is silenced on its own over the validation seeds, against a same-seed unperturbed control. Minutes, not seconds. Results are written as they finish, so it can be resumed."),
                false,
                None,
            );
            m.page_slider(
                cv,
                Rect::new(rect.x + 40 + m.bw(&label, 250), y + 3, 220, 30),
                self.top as f64,
                (5.0, 60.0, 5.0),
                "{:.0f} types",
                "cp_top",
                true,
                Some("How many candidate types to try. They are ranked by how much of the readout's input they supply within two synaptic hops; anything outside the shortlist is not tested at all."),
            );
        }
        let res = &q["results"][self.target];
        let body = Rect::new(rect.x + 16, y + 48, rect.w - 32, rect.bottom() - y - 48 - 66);
        if res.is_object() {
            self.table(m, cv, body, res, &q["type_ops"]);
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 58, m.bw("Export CSV + JSON", 200), 42),
                "Export CSV + JSON",
                Act::Custom("export".into()),
                "cp_export",
                Style::Normal,
                true,
                None,
                false,
                None,
            );
        } else {
            m.text(
                cv,
                if running { "" } else { "No run for this target yet." },
                (body.x + 8, body.y + 8),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
        }
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, &format!("error: {e}"), (rect.x + 24, y + 48), BAD, &f.small, Anchor::TopLeft);
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "cp:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        if let Some(t) = name.strip_prefix("target:") {
            if let Some(k) = TARGETS.iter().find(|k| k.0 == t) {
                self.target = k.0;
            }
        } else if let Some(t) = name.strip_prefix("apply:") {
            ctx.host.command("lab.cp_apply", json!({"type": t}));
        } else {
            match name {
                "run" => drop(ctx.host.command("lab.cp_run", json!({"target": self.target, "top": self.top}))),
                "export" => drop(ctx.host.command("lab.cp_export", json!({"target": self.target}))),
                "cancel" => drop(ctx.host.command("lab.day4_cancel", Value::Null)),
                "slide:cp_top" => {
                    if let Some(v) = text.and_then(|t| t.parse::<f64>().ok()) {
                        self.top = v as i64;
                    }
                }
                _ => {}
            }
        }
    }
}

impl LabCritical {
    fn table(&self, m: &mut Menu, cv: &mut Canvas, area: Rect, res: &Value, type_ops: &Value) {
        let f = m.fonts();
        let seeds = res["seeds"].as_array().cloned().unwrap_or_default();
        m.text(
            cv,
            &format!(
                "{}  ·  unperturbed {}  ·  seeds {}-{}  ·  {:.0}s  ·  {} types tried",
                res["metric"].as_str().unwrap_or(""),
                f2(num(&res["control"]["mean"])),
                seeds.first().map_or(String::new(), |s| s.to_string()),
                seeds.last().map_or(String::new(), |s| s.to_string()),
                num(&res["seconds"]),
                res["ranked"].as_array().map_or(0, Vec::len)
            ),
            (area.x + 8, area.y),
            INK,
            &f.small,
            Anchor::TopLeft,
        );
        let mut y = area.y + 22;
        for (dx, label) in
            [(8, "#"), (44, "cell type"), (180, "neurons"), (250, "lesioned"), (360, "95% CI"), (470, "change"), (560, "p"), (650, "")]
        {
            m.text(cv, label, (area.x + dx, y), LABEL, &f.small, Anchor::TopLeft);
        }
        y += 18;
        let view = Rect::new(area.x, y, area.w, area.bottom() - y);
        let off = m.scroll_offset("lab_critical");
        m.begin_clip(cv, view);
        let mut yy = y - off;
        let on: Vec<&str> = type_ops.as_array().into_iter().flatten().filter_map(|t| t.as_str()).collect();
        for r in res["ranked"].as_array().into_iter().flatten() {
            let share = num(&r["effect_share"]);
            let strong = num(&r["p_value"]) < 0.01 && share.abs() >= 0.1;
            let col = if strong && share < 0.0 {
                BAD
            } else if strong {
                GOOD
            } else {
                TEXT
            };
            let ty = r["type"].as_str().unwrap_or("");
            m.text(cv, &r["rank"].to_string(), (area.x + 8, yy), DIM, &f.small, Anchor::TopLeft);
            let star = if r["drives_the_behavior"].as_bool().unwrap_or(false) { "  *" } else { "" };
            m.text(cv, &format!("{ty}{star}"), (area.x + 44, yy), col, &f.small, Anchor::TopLeft);
            m.text(cv, &grp(num(&r["neurons"]), 0), (area.x + 180, yy), LABEL, &f.small, Anchor::TopLeft);
            m.text(cv, &f2(num(&r["lesioned_mean"])), (area.x + 250, yy), TEXT, &f.small, Anchor::TopLeft);
            m.text(
                cv,
                &format!("[{}, {}]", f2(num(&r["lesioned_ci"][0])), f2(num(&r["lesioned_ci"][1]))),
                (area.x + 360, yy),
                DIM,
                &f.small,
                Anchor::TopLeft,
            );
            m.text(cv, &signed_pct(share), (area.x + 470, yy), col, &f.small, Anchor::TopLeft);
            let p = num(&r["p_value"]);
            m.text(cv, &if p.is_nan() { "nan".into() } else { format!("{p:.4}") }, (area.x + 560, yy), LABEL, &f.small, Anchor::TopLeft);
            let live = on.contains(&ty);
            let label = if live { "lesion applied" } else { "Apply this lesion" };
            m.button(
                cv,
                Rect::new(area.x + 650, yy - 3, m.bw("Apply this lesion", 150), 22.max(f.bold.linesize + 2)),
                label,
                Act::Custom(format!("apply:{ty}")),
                &format!("cp_apply:{ty}"),
                Style::Normal,
                true,
                Some(&format!("Silences every {ty} neuron on the live flies, the same way brain surgery does, so you can watch what this row means in the game.")),
                live,
                None,
            );
            yy += 24;
        }
        m.set_content_h("lab_critical", (yy + off - view.bottom() + 8).max(0));
        m.end_clip(cv);
    }
}
