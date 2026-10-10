//! Lab > Classroom mode (`lab/labclassroom.py`, `lab/classroom.py`): step-by-step lecture walkthroughs of connectome circuits.
//!
//! Host: query `lab.class_state` = `{seq, status}` (the status line of the latest demonstration); command `lab.class_action {kind,
//! target, strength | amp | mode}`. The session (lecture, step) lives in the page.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const TABS: [(&str, &str); 5] = [
    ("looming", "Looming escape"),
    ("tmaze", "T-maze conditioning"),
    ("moonwalker", "Moonwalker (MDN)"),
    ("sugar", "Sugar feeding"),
    ("gf_lesion", "GF lesion"),
];

pub struct LabClassroom {
    lecture: String,
    step: usize,
    seen_seq: u64,
    status: Option<String>,
}

impl Default for LabClassroom {
    fn default() -> Self {
        Self { lecture: "looming".into(), step: 0, seen_seq: 0, status: None }
    }
}

impl LabClassroom {
    pub fn set_view(&mut self, lecture: &str, step: usize) {
        if ktf_data::classroom::lecture(lecture).is_some() {
            self.lecture = lecture.into();
            self.step = step;
            self.status = None;
        }
    }

    fn total(&self) -> usize {
        ktf_data::classroom::lecture(&self.lecture).and_then(|l| l["steps"].as_array()).map_or(0, Vec::len)
    }
}

impl Page for LabClassroom {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let (fs, ft, fb) = (&f.small, &f.text, &f.bold);
        let lh = fs.linesize;
        let q = ctx.host.query("lab.class_state");
        if let Some(seq) = q["seq"].as_u64().filter(|s| *s != self.seen_seq) {
            self.seen_seq = seq;
            self.status = q["status"].as_str().map(String::from);
        }
        let Some(proto) = ktf_data::classroom::lecture(&self.lecture) else { return };
        let steps = proto["steps"].as_array().cloned().unwrap_or_default();
        let step = &steps[self.step.min(steps.len() - 1)];

        m.text(cv, "CLASSROOM MODE & LECTURE PROTOCOLS", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let mut y = m.wrapped(
            cv,
            "Curated sequential demonstrations for lectures and teaching. Every behavior is either grounded in connectome wiring or labelled a game rule.",
            (rect.x + 24, m.under_heading(rect).max(rect.y + 48)),
            rect.w - 48,
            LABEL,
            fs,
            2,
        ) + 8;
        let lab = m.text(cv, "Select Lecture:", (rect.x + 24, y + 4), TEXT, fs, Anchor::TopLeft);
        let th = 26.max(lh + 8);
        let x0 = lab.right() + 12;
        let mut tx = x0;
        for (id, title) in TABS {
            let tw = fs.size(title) + 24;
            if tx + tw > rect.right() - 24 && tx > x0 {
                tx = x0;
                y += th + 6;
            }
            m.button(
                cv,
                Rect::new(tx, y, tw, th),
                title,
                Act::Custom(format!("lecture:{id}")),
                &format!("class_tab:{id}"),
                Style::Normal,
                true,
                None,
                self.lecture == id,
                Some(fs.clone()),
            );
            tx += tw + 8;
        }
        y += th + 10;

        let step_str = format!("Step {} of {}", self.step + 1, steps.len());
        let sw = fb.size(&step_str);
        let title = format!("Lecture: {}", proto["title"].as_str().unwrap_or(""));
        let title_lines = Menu::fit_lines(fb, &title, rect.w - 48 - sw - 40, 2).0;
        let banner = Rect::new(rect.x + 24, y, rect.w - 48, 32.max(title_lines.len() as i32 * fb.linesize + 10));
        cv.fill_rect(banner, [20, 24, 34], 6);
        cv.stroke_rect(banner, [60, 70, 95], 6);
        for (i, ln) in title_lines.iter().enumerate() {
            m.text(cv, ln, (banner.x + 12, banner.y + 5 + i as i32 * fb.linesize), [140, 190, 255], fb, Anchor::TopLeft);
        }
        m.text(cv, &step_str, (banner.right() - 12, banner.center().1), AMBER, fb, Anchor::MidRight);
        y = banner.bottom() + 10;

        let card = Rect::new(rect.x + 24, y, rect.w - 48, rect.bottom() - 72 - y);
        cv.fill_rect(card, [18, 22, 32], 8);
        cv.stroke_rect(card, [40, 48, 66], 8);
        let key = "lab_classroom";
        let off = m.scroll_offset(key);
        let inner = Rect::new(card.x + 2, card.y + 2, card.w - 4, card.h - 4);
        m.begin_clip(cv, inner);
        let top = card.y + 16 - off;
        let grounding = step["grounding"].as_str().unwrap_or("");
        let gw = fs.size(grounding);
        let tr = m.wrapped(
            cv,
            step["title"].as_str().unwrap_or(""),
            (card.x + 18, top),
            card.w - 36 - if card.w > 900 { gw + 20 } else { 0 },
            INK,
            &f.head,
            3,
        );
        let mut content_y;
        if card.w > 900 {
            m.text(cv, grounding, (card.right() - 18, top + 4), [100, 200, 140], fs, Anchor::TopRight);
            content_y = tr + 10;
        } else {
            content_y = m.wrapped(cv, grounding, (card.x + 18, tr + 2), card.w - 36, [100, 200, 140], fs, 3) + 10;
        }
        let col_top = content_y;
        let left_w = (f64::from(card.w) * 0.58) as i32;
        m.text(cv, "CIRCUIT MECHANISM & EXPLANATION", (card.x + 18, content_y), LABEL, fs, Anchor::TopLeft);
        content_y += lh + 4;
        content_y = m.wrapped(cv, step["explanation"].as_str().unwrap_or(""), (card.x + 18, content_y), left_w - 24, TEXT, ft, 99) + 12;
        let kt = step["key_takeaway"].as_str().unwrap_or("");
        let kt_lines = Menu::fit_lines(fs, kt, left_w - 44, 99).0;
        let tk = Rect::new(card.x + 18, content_y, left_w - 24, 8 + lh + 2 + kt_lines.len() as i32 * lh + 8);
        cv.fill_rect(tk, [28, 38, 52], 6);
        cv.stroke_rect(tk, [70, 120, 180], 6);
        m.text(cv, "Key Takeaway:", (tk.x + 10, tk.y + 8), [140, 200, 255], fs, Anchor::TopLeft);
        m.wrapped(cv, kt, (tk.x + 10, tk.y + 8 + lh + 2), tk.w - 20, INK, fs, 99);
        content_y = tk.bottom() + 10;
        let action = &step["action"];
        if action.as_object().is_some_and(|o| !o.is_empty()) {
            let kind = action["kind"].as_str().unwrap_or("demo").to_lowercase();
            let target = action["target"].as_str().unwrap_or("");
            let label = format!("Demonstrate step: {kind} {target}").trim().to_string();
            let bh = 36.max(fb.linesize + 12);
            let bw = (left_w - 24).min(fb.size(&label) + 32);
            m.button(
                cv,
                Rect::new(card.x + 18, content_y, bw, bh),
                &label,
                Act::Custom("demo".into()),
                "class_run_demo",
                Style::Primary,
                true,
                None,
                false,
                None,
            );
            content_y += bh + 6;
            if let Some(st) = &self.status {
                content_y = m.wrapped(cv, &format!("Status: {st}"), (card.x + 18, content_y), left_w - 24, [120, 220, 150], fs, 3);
            }
        } else {
            content_y =
                m.wrapped(cv, "No active stimulus required for this step.", (card.x + 18, content_y + 4), left_w - 24, LABEL, fs, 2);
        }
        let right_x = card.x + left_w + 12;
        let right_w = card.w - left_w - 30;
        m.text(cv, "NEURONS INVOLVED", (right_x, col_top), LABEL, fs, Anchor::TopLeft);
        let mut ny = col_top + lh + 6;
        for nr in step["neurons"].as_array().into_iter().flatten().take(4) {
            let count = nr["count"].as_i64().unwrap_or(0);
            let cnt = if count > 0 { format!("n={count}") } else { String::new() };
            let name_lines = Menu::fit_lines(
                fb,
                nr["label"].as_str().unwrap_or(""),
                right_w - 16 - if cnt.is_empty() { 0 } else { fs.size(&cnt) + 10 },
                2,
            )
            .0;
            let role = nr["role"].as_str().unwrap_or("");
            let role_lines = Menu::fit_lines(fs, role, right_w - 16, 99).0;
            let nb = Rect::new(right_x, ny, right_w, 6 + name_lines.len() as i32 * fb.linesize + 2 + role_lines.len() as i32 * lh + 6);
            cv.fill_rect(nb, [24, 28, 40], 5);
            cv.stroke_rect(nb, [45, 54, 76], 5);
            let mut yy = nb.y + 6;
            for ln in &name_lines {
                m.text(cv, ln, (nb.x + 8, yy), [255, 200, 110], fb, Anchor::TopLeft);
                yy += fb.linesize;
            }
            if !cnt.is_empty() {
                m.text(cv, &cnt, (nb.right() - 8, nb.y + 6), LABEL, fs, Anchor::TopRight);
            }
            m.wrapped(cv, role, (nb.x + 8, yy + 2), nb.w - 16, TEXT, fs, 99);
            ny = nb.bottom() + 6;
        }
        ny += 8;
        m.text(cv, "SCIENTIFIC CITATION", (right_x, ny), LABEL, fs, Anchor::TopLeft);
        ny += lh + 4;
        let citation = step["citation"].as_str().unwrap_or("");
        let c_lines = Menu::fit_lines(fs, citation, right_w - 20, 99).0;
        let cb = Rect::new(right_x, ny, right_w, 16 + c_lines.len() as i32 * lh);
        cv.fill_rect(cb, [15, 20, 30], 6);
        cv.stroke_rect(cb, [55, 75, 105], 6);
        m.wrapped(cv, citation, (cb.x + 10, cb.y + 8), cb.w - 20, [180, 210, 255], fs, 99);
        let bottom = content_y.max(cb.bottom()) + 12;
        m.set_content_h(key, (bottom + off - card.bottom()).max(0));
        m.end_clip(cv);

        let ctrl_y = rect.bottom() - 58;
        let mut x = rect.x + 24;
        let (can_prev, can_next) = (self.step > 0, self.step + 1 < steps.len());
        for (label, act, id, style, en) in [
            ("Reset protocol", "reset", "class_reset", Style::Normal, true),
            ("< Previous step", "prev", "class_prev", Style::Normal, can_prev),
            ("Next step >", "next", "class_next", if can_next { Style::Primary } else { Style::Normal }, can_next),
        ] {
            let bw = 140.max(fb.size(label) + 28);
            m.button(cv, Rect::new(x, ctrl_y, bw, 42), label, Act::Custom(act.into()), id, style, en, None, false, None);
            x += bw + 12;
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, ctrl_y, 140, 42),
            "Back",
            Act::Back,
            "class_back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if let Some(id) = name.strip_prefix("lecture:") {
            self.set_view(id, 0);
            return;
        }
        match name {
            "next" => {
                if self.step + 1 < self.total() {
                    self.step += 1;
                    self.status = None;
                }
            }
            "prev" => {
                if self.step > 0 {
                    self.step -= 1;
                    self.status = None;
                }
            }
            "reset" => {
                self.step = 0;
                self.status = None;
                ctx.host.command("lab.class_action", json!({"kind": "reset_session"}));
            }
            "demo" => {
                let lec = ktf_data::classroom::lecture(&self.lecture);
                let action = lec.and_then(|l| l["steps"].get(self.step)).map_or(Value::Null, |s| s["action"].clone());
                ctx.host.command("lab.class_action", action);
            }
            _ => {}
        }
    }
}
