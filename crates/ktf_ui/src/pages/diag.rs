//! Settings > Help pages (`ui/help_ui.py`): the self-test, the bug report review screen, and the crash screen's page (`ui/crashscreen.py`).
//!
//! The self-test runs in the game on a worker thread: host query `selftest` -> `{running, checks: [{name, status, detail, fix}], counts}`,
//! command `selftest_run`. The bug report is built here from `ktf_data::diag` (the text on screen is the text that is copied, saved or put
//! in the URL); the host only copies to the clipboard (`clipboard_put`) and opens the browser (`open_url`). Crash page: query `crash_paths`.
//! Nothing is ever sent.

use ktf_data::diag::{self, Item};
use serde_json::Value;

use crate::canvas::{Canvas, Rect};
use crate::i18n::{tr, tr_args};
use crate::menu::Style;
use crate::menu::{Act, Anchor, Ctx, Menu, Page, RowButton};
use crate::theme::*;

/// `help_ui.STATUS_COLORS` per accessibility palette.
pub fn status_color(palette: &str, status: &str) -> Color {
    match (palette, status) {
        ("blue-yellow", "PASS") => [70, 130, 240],
        ("blue-yellow", "WARN") => [255, 224, 70],
        ("blue-yellow", "FAIL") => [255, 255, 255],
        ("high-contrast", "PASS") => [255, 255, 255],
        ("high-contrast", "WARN") => [255, 60, 220],
        ("high-contrast", "FAIL") => [255, 60, 60],
        (_, "PASS") => [90, 200, 120],
        (_, "WARN") => [240, 180, 60],
        (_, "FAIL") => [230, 90, 80],
        _ => TEXT,
    }
}

// --- self-test --------------------------------------------------------------------------------------------------------------------
pub struct SelfTestPage;

impl Page for SelfTestPage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let st = ctx.host.query("selftest");
        let palette = ctx.cfg.get_str("access.palette").to_string();
        m.text(cv, &tr("SELF-TEST"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let checks = st["checks"].as_array().cloned().unwrap_or_default();
        let running = st["running"].as_bool().unwrap_or(false);
        if running {
            let n = checks.len().to_string();
            m.text(
                cv,
                &tr_args("Running: {n} checks done...", &[("n", &n)]),
                (rect.right() - 24, rect.y + 22),
                AMBER,
                &f.small,
                Anchor::TopRight,
            );
        } else if let Some(c) = st["counts"].as_object() {
            let g = |k: &str| c.get(k).and_then(Value::as_i64).unwrap_or(0);
            let worst = if g("FAIL") > 0 {
                "FAIL"
            } else if g("WARN") > 0 {
                "WARN"
            } else {
                "PASS"
            };
            let (p, w, fl) = (g("PASS").to_string(), g("WARN").to_string(), g("FAIL").to_string());
            m.text(
                cv,
                &tr_args("{p} passed, {w} warnings, {f} failed", &[("p", &p), ("w", &w), ("f", &fl)]),
                (rect.right() - 24, rect.y + 22),
                status_color(&palette, worst),
                &f.small,
                Anchor::TopRight,
            );
        }
        let body = Rect::new(rect.x + 16, rect.y + 56, rect.w - 32, rect.h - 56 - 70);
        let key = "selftest";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y - off;
        for c in &checks {
            let status = c["status"].as_str().unwrap_or("");
            let col = status_color(&palette, status);
            let chip = Rect::new(body.x + 8, y + 4, 62, 22);
            cv.fill_rect(chip, col, 5);
            m.text(cv, status, chip.center(), [12, 14, 18], &f.small, Anchor::Center);
            m.text(cv, c["name"].as_str().unwrap_or(""), (body.x + 82, y + 3), INK, &f.bold, Anchor::TopLeft);
            y = m.wrapped(cv, c["detail"].as_str().unwrap_or(""), (body.x + 82, y + 26), body.w - 100, TEXT, &f.small, 4);
            if let Some(fix) = c["fix"].as_str().filter(|s| !s.is_empty()) {
                y = m.wrapped(cv, &tr_args("Fix: {fix}", &[("fix", fix)]), (body.x + 82, y + 2), body.w - 100, col, &f.small, 4);
            }
            y += 12;
        }
        m.end_clip(cv);
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.scrollbar(cv, body, (y + off - body.bottom() + 8).max(0), off);
        let fy = rect.bottom() - 58;
        m.button(
            cv,
            Rect::new(rect.x + 24, fy, 170, 42),
            &tr("Run again"),
            Act::Custom("again".into()),
            "st:again",
            Style::Normal,
            !running,
            Some(&tr("Run every check again.")),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(rect.x + 206, fy, 190, 42),
            &tr("Report a bug"),
            Act::Show("bugreport".into()),
            "st:report",
            Style::Normal,
            true,
            Some(&tr("Review what a bug report would include, with this self-test in it.")),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(rect.right() - 164, fy, 140, 42),
            &tr("Back"),
            Act::Back,
            "st:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if name == "again" {
            ctx.host.command("selftest_run", Value::Null);
        }
    }
}

// --- bug report -------------------------------------------------------------------------------------------------------------------
#[derive(Default)]
pub struct BugReportPage {
    items: Option<Vec<Item>>,
    sel: usize,
    note: String,
    saved: Option<String>,
    msg: String,
}

impl BugReportPage {
    fn texts(&self) -> (String, String, Option<String>, String) {
        let items = self.items.as_deref().unwrap_or(&[]);
        let full = diag::clipboard_text(items, &self.note);
        let (title, body, attach) = diag::compose_issue(items, &self.note, "the file it saves");
        let url = diag::issue_url(&title, &body, diag::URL_LIMIT);
        (full, url, attach, body)
    }
}

impl Page for BugReportPage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        if self.items.is_none() {
            let version = ctx.host.query("app_version");
            let state = ktf_data::paths::get().state_dir;
            let report = diag::last_report_value();
            self.items = Some(diag::collect(version.as_str().unwrap_or(env!("CARGO_PKG_VERSION")), report.as_ref(), &state));
            self.sel = 0;
        }
        let (full, url, attach, _) = self.texts();
        let n_items = self.items.as_ref().map_or(0, Vec::len);
        let views = [
            (tr("Report as copied"), full.clone()),
            (tr("Text in the issue link"), diag::unquote(url.split_once("&body=").map_or("", |x| x.1))),
        ];
        m.text(cv, &tr("REPORT A BUG"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        m.text(
            cv,
            &tr("Nothing is sent. This is exactly what would be included: switch items off to leave them out."),
            (rect.x + 24, rect.y + 48),
            LABEL,
            &f.small,
            Anchor::TopLeft,
        );
        m.text(cv, &tr("What happened?"), (rect.x + 24, rect.y + 76), TEXT, &f.small, Anchor::TopLeft);
        m.text_field(
            cv,
            Rect::new(rect.x + 150, rect.y + 70, rect.w - 176, 30),
            &self.note.clone(),
            "br:note",
            Some(&tr("A sentence about what you were doing. It goes into the report text below the items.")),
            240,
            false,
        );
        // the item list
        let left = Rect::new(rect.x + 16, rect.y + 112, 300, rect.h - 112 - 132);
        let mut y = left.y;
        for i in 0..n_items {
            let it = &self.items.as_ref().unwrap()[i];
            let row = Rect::new(left.x, y, left.w, 58);
            let on_sel = i == self.sel;
            cv.fill_rect(row, if on_sel { [30, 36, 50] } else { [22, 26, 34] }, 8);
            cv.stroke_rect(row, if on_sel { ACCENT } else { BORDER }, 8);
            m.register_drag_src(
                row,
                &format!("br-row:{i}"),
                "",
                Some(Act::Custom(format!("sel:{i}"))),
                Some(&tr("Show this item's exact text.")),
                None,
            );
            m.text(cv, &tr(&it.title), (row.x + 10, row.y + 8), if it.include { INK } else { DIM }, &f.bold, Anchor::TopLeft);
            m.text(cv, &tr(&it.size()), (row.x + 10, row.y + 32), LABEL, &f.small, Anchor::TopLeft);
            m.toggle(
                cv,
                Rect::new(row.right() - 100, row.y + 16, 90, 26),
                it.include,
                Act::Custom(format!("inc:{i}")),
                &format!("br-inc:{i}"),
                !it.text.trim().is_empty(),
                &tr("Include this item in the report. It is empty when there is nothing to include."),
            );
            y += 64;
        }
        for (j, (vt, _)) in views.iter().enumerate() {
            let i = n_items + j;
            let row = Rect::new(left.x, y, left.w, 40);
            let on_sel = i == self.sel;
            cv.fill_rect(row, if on_sel { [30, 36, 50] } else { [22, 26, 34] }, 8);
            cv.stroke_rect(row, if on_sel { ACCENT } else { BORDER }, 8);
            m.register_drag_src(
                row,
                &format!("br-row:{i}"),
                "",
                Some(Act::Custom(format!("sel:{i}"))),
                Some(&tr("The final text, exactly as it is copied, saved or put in the link.")),
                None,
            );
            m.text(cv, vt, (row.x + 10, row.y + 10), INK, &f.bold, Anchor::TopLeft);
            y += 46;
        }
        // the exact text
        let view = Rect::new(left.right() + 12, left.y, rect.right() - 16 - left.right() - 12, left.h);
        cv.fill_rect(view, [10, 12, 18], 8);
        cv.stroke_rect(view, BORDER, 8);
        let (text, include, nonempty) = if self.sel >= n_items {
            (views[(self.sel - n_items).min(1)].1.clone(), true, true)
        } else {
            let it = &self.items.as_ref().unwrap()[self.sel];
            (it.text.clone(), it.include, !it.text.trim().is_empty())
        };
        let shown = if text.trim().is_empty() { tr("(nothing to show: no data for this item on this computer)") } else { text };
        let mono = &f.mono12;
        let lh = mono.linesize;
        let lines: Vec<&str> = if shown.is_empty() { vec![""] } else { shown.lines().collect() };
        let key = "bugreport";
        let off = m.scroll_offset(key);
        let inner = Rect::new(view.x + 6, view.y + 4, view.w - 12, view.h - 8);
        m.begin_clip(cv, inner);
        let first = (off / lh).max(0) as usize;
        let last = (first + (inner.h / lh) as usize + 2).min(lines.len());
        for (j, line) in lines.iter().enumerate().take(last).skip(first) {
            let l: String = line.chars().take(220).collect();
            mono.draw(cv, &l, inner.x, inner.y + j as i32 * lh - off, if include { TEXT } else { DIM });
        }
        m.end_clip(cv);
        let content = (lines.len() as i32 * lh - inner.h + 8).max(0);
        m.set_content_h(key, content);
        m.scrollbar(cv, view, content, off);
        if !include && nonempty {
            m.text(cv, &tr("left out"), (view.right() - 10, view.y + 6), AMBER, &f.small, Anchor::TopRight);
        }
        // actions
        let fy = rect.bottom() - 116;
        let extra = if f.small.size(&"x".repeat(150)) > rect.w - 48 { f.small.linesize } else { 0 };
        let (nc, nu) = (diag::group3(full.chars().count()), diag::group3(url.chars().count()));
        m.wrapped(
            cv,
            &tr_args(
                "Report text: {n} characters. Opening the issue puts about {u} characters in the link; long items go to a file you attach.",
                &[("n", &nc), ("u", &nu)],
            ),
            (rect.x + 24, fy - 4 - extra),
            rect.w - 48,
            LABEL,
            &f.small,
            2,
        );
        if let Some(p) = &self.saved {
            m.text(
                cv,
                &tr_args("Saved to {path}: attach this file to the issue.", &[("path", p)]),
                (rect.x + 24, fy + 16),
                GOOD,
                &f.small,
                Anchor::TopLeft,
            );
        } else if !self.msg.is_empty() {
            m.text(cv, &self.msg.clone(), (rect.x + 24, fy + 16), GOOD, &f.small, Anchor::TopLeft);
        }
        let by = rect.bottom() - 62;
        let _ = attach;
        m.button_row(
            cv,
            rect.x + 24,
            by,
            42,
            &[
                RowButton::new(&tr("Copy to clipboard"), Act::Custom("copy".into()), "br:copy", 210)
                    .tip(&tr("Copy everything that is switched on. You paste it wherever you like.")),
                RowButton::new(&tr("Open GitHub issue"), Act::Custom("open".into()), "br:open", 250).primary().tip(&tr(
                    "Opens your browser on a prefilled new-issue page. If items are long, they are saved to a file to attach by hand. Nothing is sent by this program.",
                )),
                RowButton::new(&tr("Save to file"), Act::Custom("save".into()), "br:save", 180).tip(&tr("Save everything that is switched on as a text file.")),
            ],
            12,
            None,
        );
        m.button(
            cv,
            Rect::new(rect.right() - 164, by, 140, 42),
            &tr("Back"),
            Act::Custom("back".into()),
            "br:back",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        if let Some(i) = name.strip_prefix("sel:") {
            self.sel = i.parse().unwrap_or(0);
        } else if let Some(i) = name.strip_prefix("inc:") {
            if let (Ok(i), Some(items)) = (i.parse::<usize>(), self.items.as_mut())
                && let Some(it) = items.get_mut(i)
            {
                it.include = !it.include && !it.text.trim().is_empty();
            }
        } else if name == "text:br:note" {
            self.note = text.unwrap_or("").chars().take(240).collect();
        } else if name == "copy" {
            let (full, ..) = self.texts();
            let ok = ctx.host.command("clipboard_put", Value::from(full.clone())).as_bool().unwrap_or(false);
            if ok {
                self.msg = tr("copied to the clipboard");
                self.saved = None;
            } else {
                self.save(&full);
            }
        } else if name == "save" {
            let (full, ..) = self.texts();
            self.save(&full);
        } else if name == "open" {
            let (_, url, attach, _) = self.texts();
            if let Some(a) = attach {
                self.saved = diag::save_attachment(&a, None).map(|p| p.to_string_lossy().into_owned());
            }
            let ok = ctx.host.command("open_url", Value::from(url)).as_bool().unwrap_or(false);
            self.msg = tr(if ok { "Opened your browser." } else { "Could not open a browser: copy the report instead." });
        } else if name == "back" {
            *self = BugReportPage::default();
            m.back();
        }
    }
}

impl BugReportPage {
    /// Show this content instead of collecting it from the machine (the screenshot tests).
    pub fn preset(&mut self, items: Vec<Item>, sel: usize, note: &str, saved: Option<&str>) {
        self.items = Some(items);
        self.sel = sel;
        self.note = note.into();
        self.saved = saved.map(String::from);
    }

    fn save(&mut self, text: &str) {
        self.saved = diag::save_attachment(text, None).map(|p| p.to_string_lossy().into_owned());
        self.msg = if self.saved.is_none() { tr("could not write the file") } else { String::new() };
    }
}

// --- crash screen -----------------------------------------------------------------------------------------------------------------
/// The page the crash window shows (`crashscreen._page_crash`): where the report went, Report a bug, Quit (`Action("crash_quit")`).
pub struct CrashPage;

impl Page for CrashPage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let cx = rect.centerx();
        m.text(cv, &tr("Kick the Fly stopped"), (cx, rect.y + 34), INK, &f.head, Anchor::MidTop);
        m.wrapped(
            cv,
            &tr("Something went wrong and the game had to close. Your settings, saves and the fly's training memory are not affected. A crash report was written:"),
            (rect.x + 40, rect.y + 90),
            rect.w - 80,
            TEXT,
            &f.text,
            3,
        );
        let mut y = rect.y + 170;
        let paths = ctx.host.query("crash_paths");
        let paths = paths.as_array().cloned().unwrap_or_default();
        for p in &paths {
            m.text(cv, p.as_str().unwrap_or(""), (rect.x + 40, y), ACCENT, &f.small, Anchor::TopLeft);
            y += 22;
        }
        if paths.is_empty() {
            m.text(cv, &tr("(no folder was writable, so no file could be written)"), (rect.x + 40, y), AMBER, &f.small, Anchor::TopLeft);
        }
        m.wrapped(
            cv,
            &tr("You can report it: you will see exactly what would be included before anything leaves this computer, and you choose whether to copy it or open a prefilled GitHub issue. Nothing is sent automatically."),
            (rect.x + 40, y + 24),
            rect.w - 80,
            LABEL,
            &f.small,
            4,
        );
        m.button(
            cv,
            Rect::new(cx - 270, rect.bottom() - 90, 260, 50),
            &tr("Report a bug"),
            Act::Show("bugreport".into()),
            "crash:report",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(cx + 10, rect.bottom() - 90, 260, 50),
            &tr("Quit"),
            Act::Action("crash_quit".into()),
            "crash:quit",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, _name: &str, _text: Option<&str>, _ctx: &mut Ctx) {}
}
