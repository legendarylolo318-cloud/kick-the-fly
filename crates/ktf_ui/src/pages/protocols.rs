//! Lab > Protocols (`lab/lab.py:page_protocols`): the YAML experiment files, each with a Run button; the finished run can be bundled.
//!
//! Host: query `lab.protocols_state` = `{files: [{name, desc, error, runnable}], job: {name, done, total} | null, folder, error,
//! user_dir, bundle: {msg, path}}`; commands `lab.proto_run {name}`, `lab.bundle`.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

#[derive(Default)]
pub struct LabProtocols;

fn short(path: &str, n: usize) -> String {
    let c: Vec<char> = path.chars().collect();
    if c.len() <= n {
        return path.to_string();
    }
    let head: String = c[..n / 2 - 2].iter().collect();
    let tail: String = c[c.len() - (n / 2 - 1)..].iter().collect();
    format!("{head}…{tail}")
}

impl Page for LabProtocols {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.protocols_state");
        m.text(cv, "PROTOCOLS", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let top = m.wrapped(
            cv,
            &format!(
                "YAML experiment files. Put your own in {}. Headless: KickTheFly --headless --protocol FILE",
                short(q["user_dir"].as_str().unwrap_or(""), 60)
            ),
            (rect.x + 24, m.under_heading(rect).max(rect.y + 48)),
            rect.w - 48,
            LABEL,
            &f.small,
            3,
        ) + 10;
        let job = &q["job"];
        let busy = job.is_object();
        let body = Rect::new(rect.x + 16, top, rect.w - 32, rect.bottom() - 110 - top);
        let key = "lab_protocols";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y - off;
        let rh = 42.max(f.bold.linesize + f.small.linesize + 10);
        let bh = 30.max(f.small.linesize + 12);
        for file in q["files"].as_array().into_iter().flatten() {
            let name = file["name"].as_str().unwrap_or("");
            let err = file["error"].as_bool().unwrap_or(false);
            let runnable = file["runnable"].as_bool().unwrap_or(true);
            let row = Rect::new(rect.x + 24, y, rect.w - 48, rh);
            cv.fill_rect(row, [26, 30, 40], 8);
            m.text(cv, name, (row.x + 12, row.y + 4), INK, &f.bold, Anchor::TopLeft);
            m.wrapped(
                cv,
                file["desc"].as_str().unwrap_or(""),
                (row.x + 12, row.y + 4 + f.bold.linesize),
                row.w - 140,
                if err { BAD } else { LABEL },
                &f.small,
                1,
            );
            m.button(
                cv,
                Rect::new(row.right() - 110, row.centery() - bh / 2, 96, bh),
                "Run",
                Act::Custom(format!("run:{name}")),
                &format!("proto:{name}"),
                Style::Normal,
                !err && !busy && runnable,
                if runnable || err { None } else { Some("A classroom lecture: step through it in Lab > Classroom.") },
                false,
                Some(f.small.clone()),
            );
            y += rh + 8;
        }
        m.set_content_h(key, (y + off - body.bottom()).max(0));
        m.end_clip(cv);
        if busy {
            let frac = job["done"].as_f64().unwrap_or(0.0) / job["total"].as_f64().unwrap_or(1.0).max(1.0);
            cv.fill_rect(Rect::new(rect.x + 24, rect.bottom() - 100, rect.w - 220, 12), [30, 36, 48], 6);
            cv.fill_rect(Rect::new(rect.x + 24, rect.bottom() - 100, 8.max(((rect.w - 220) as f64 * frac) as i32), 12), AMBER, 6);
            m.text(
                cv,
                &format!("running {}: {}/{}", job["name"].as_str().unwrap_or(""), job["done"], job["total"]),
                (rect.x + 24, rect.bottom() - 84),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
        } else if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, e, (rect.x + 24, rect.bottom() - 90), BAD, &f.small, Anchor::TopLeft);
        } else if let Some(folder) = q["folder"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, &format!("Done: {}", short(folder, 110)), (rect.x + 24, rect.bottom() - 90), GOOD, &f.small, Anchor::TopLeft);
            m.button(
                cv,
                Rect::new(rect.right() - 320, rect.bottom() - 58, 140, 42),
                "Bundle",
                Act::Custom("bundle".into()),
                "proto:bundle",
                Style::Normal,
                true,
                Some("Zip this run with its protocol, metadata and an RO-Crate description, so it can be rerun and checked."),
                false,
                None,
            );
            if let Some(msg) = q["bundle"]["msg"].as_str().filter(|s| !s.is_empty()) {
                m.text(
                    cv,
                    msg,
                    (rect.x + 24, rect.bottom() - 70),
                    if q["bundle"]["path"].is_string() { GOOD } else { AMBER },
                    &f.small,
                    Anchor::TopLeft,
                );
            }
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "proto:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if let Some(n) = name.strip_prefix("run:") {
            drop(ctx.host.command("lab.proto_run", json!({"name": n})));
        } else if name == "bundle" {
            drop(ctx.host.command("lab.bundle", Value::Null));
        }
    }
}
