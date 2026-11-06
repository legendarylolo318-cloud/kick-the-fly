//! Name the fly: a text box for the focused fly's name (shown over it) and the switch for showing name tags at all. Opened with C in the room.
//! The names live in the game (`fly_names`); this page reads `fly_name` / `name_tags` and sends `rename_fly` / `name_tags`.

use serde_json::json;

use crate::canvas::{Canvas, Rect};
use crate::i18n::tr;
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

pub const MAX_NAME: usize = 16;

pub struct Rename;

impl Page for Rename {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, &tr("Name the fly"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let name = ctx.host.query("fly_name").as_str().unwrap_or("").to_string();
        m.wrapped(
            cv,
            &tr("Click the box, type a name and press Enter. An empty name removes it."),
            (rect.x + 24, rect.y + 60),
            rect.w - 48,
            LABEL,
            &f.small,
            3,
        );
        m.text_field(
            cv,
            Rect::new(rect.x + 24, rect.y + 110, 360.min(rect.w - 48), 38),
            &name,
            "flyname",
            Some(&tr("The name floats over this fly while Name tags is on.")),
            MAX_NAME,
            false,
        );
        let on = ctx.host.query("name_tags").as_bool().unwrap_or(true);
        m.button(
            cv,
            Rect::new(rect.x + 24, rect.y + 170, 260, 42),
            &format!("{}: {}", tr("Name tags"), tr(if on { "on" } else { "off" })),
            Act::Custom("tags".into()),
            "rename:tags",
            Style::Normal,
            true,
            Some(&tr("Show or hide every fly's name over it. Saved.")),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(rect.centerx() - 130, rect.bottom() - 62, 260, 46),
            &tr("Done"),
            Act::Back,
            "rename:ok",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        if name == "text:flyname" {
            let t: String = text.unwrap_or("").trim().chars().take(MAX_NAME).collect();
            ctx.host.command("rename_fly", json!({ "name": t }));
        } else if name == "tags" {
            let on = ctx.host.query("name_tags").as_bool().unwrap_or(true);
            ctx.host.command("name_tags", json!({ "on": !on }));
        }
    }
}
