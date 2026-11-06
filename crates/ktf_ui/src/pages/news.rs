//! What's new in this version: one short skippable card, shown once after each version change (`last_version` in config.toml).
//! The 3.0 tour stays at `whatsnew` (Settings > Help). Edit `ITEMS` when a release has something to say.

use crate::canvas::{Canvas, Rect};
use crate::i18n::tr;
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

/// (heading, text). Keep it to a handful of lines: the full list is CHANGELOG.md.
const ITEMS: [(&str, &str); 4] = [
    ("Fly names", "Free the mouse (Tab) and press C to name the fly in view; the same page shows or hides the name tags."),
    ("Window", "The window remembers its size and position, and never opens off-screen if a monitor is gone."),
    ("Colors", "Settings > Accessibility has viridis and cividis, two palettes safe for every common kind of color blindness."),
    ("Steadier brains", "A bad input can no longer silence a fly's neurons for the rest of the session."),
];

pub struct News;

impl Page for News {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, _ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(
            cv,
            &format!("{} {}", tr("What's new in Kick the Fly"), env!("CARGO_PKG_VERSION")),
            (rect.x + 24, rect.y + 16),
            INK,
            &f.head,
            Anchor::TopLeft,
        );
        let mut y = rect.y + 70;
        for (head, text) in ITEMS {
            m.text(cv, &tr(head), (rect.x + 28, y), AMBER, &f.bold, Anchor::TopLeft);
            y += f.bold.linesize + 2;
            y = m.wrapped(cv, &tr(text), (rect.x + 28, y), rect.w - 56, TEXT, &f.text, 4) + 14;
        }
        m.button(
            cv,
            Rect::new(rect.centerx() - 130, rect.bottom() - 62, 260, 46),
            &tr("Got it"),
            Act::Back,
            "news:ok",
            Style::Primary,
            true,
            Some(&tr("Close this. It shows once after each update; Settings > Help has the older tour.")),
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, _name: &str, _text: Option<&str>, _ctx: &mut Ctx) {}
}
