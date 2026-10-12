//! Pause menu > Load State (`page_load_state` in `kick_the_fly.py`): the saves in the saves folder, newest first; ones that can't load say
//! why. The game lists the folder (query `saves`: `[{"file", "name", "meta", "why"}]`, `why` null when it can load) and loads on
//! `Menu::action("load_state:<file>")`.

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

pub struct LoadState;

impl Page for LoadState {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, "LOAD STATE", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let folder = ctx.host.query("saves_dir");
        m.text(cv, folder.as_str().unwrap_or(""), (rect.x + 24, rect.y + 50), LABEL, &f.small, Anchor::TopLeft);
        let saves = ctx.host.query("saves");
        let entries = saves.as_array().cloned().unwrap_or_default();
        let body = Rect::new(rect.x + 16, rect.y + 80, rect.w - 32, rect.h - 150);
        let off = m.scroll_offset("load_state");
        m.begin_clip(cv, body);
        let mut y = body.y - off;
        if entries.is_empty() {
            m.text(cv, "No saves yet. Use Save State in the pause menu.", body.center(), LABEL, &f.text, Anchor::Center);
        }
        for e in &entries {
            let row = Rect::new(body.x, y, body.w - 12, 56);
            let why = e["why"].as_str();
            let name = e["name"].as_str().unwrap_or("");
            let (label, sub) = match e["meta"].as_object() {
                Some(meta) => {
                    let nfl = meta["flies"].as_array().map_or(0, |f| f.len());
                    let s = |k: &str| meta.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                    (
                        format!(
                            "{}   {}   {nfl} {}   {}   seed {}",
                            s("created"),
                            s("mode").to_uppercase(),
                            if nfl == 1 { "fly" } else { "flies" },
                            s("arena"),
                            meta.get("seed").map_or("0".to_string(), |v| v.to_string())
                        ),
                        format!("{name}   v{} on {}", s("app_version"), s("platform")),
                    )
                }
                None => (name.to_string(), String::new()),
            };
            let tip = why.map(|w| format!("Can't load: {w}"));
            m.button(
                cv,
                row,
                "",
                Act::Custom(format!("load:{}", e["file"].as_str().unwrap_or(""))),
                &format!("load:{name}"),
                Style::Normal,
                why.is_none(),
                tip.as_deref(),
                false,
                None,
            );
            m.text(cv, &label, (row.x + 14, row.y + 8), if why.is_none() { INK } else { DIM }, &f.bold, Anchor::TopLeft);
            let (sub_text, col) = match why {
                Some(w) => (format!("can't load: {w}"), BAD),
                None => (sub, LABEL),
            };
            m.text(cv, &sub_text, (row.x + 14, row.y + 32), col, &f.small, Anchor::TopLeft);
            y += 62;
        }
        m.set_content_h("load_state", (y + off - body.bottom()).max(0));
        m.end_clip(cv);
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "load:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, _text: Option<&str>, _ctx: &mut Ctx) {
        if let Some(file) = name.strip_prefix("load:") {
            m.action(&format!("load_state:{file}"));
        }
    }
}
