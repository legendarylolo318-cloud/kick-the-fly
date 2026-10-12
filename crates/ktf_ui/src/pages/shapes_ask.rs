//! The one-time question: may the game download real neuron shapes? (`page_neuron_shapes`, `kick_the_fly.py`; asked once, Settings > Brain
//! changes it later, and it applies on the next launch.)

use serde_json::Value;

use crate::canvas::{Canvas, Rect};
use crate::i18n::tr;
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

pub struct ShapesAsk;

impl Page for ShapesAsk {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, _ctx: &mut Ctx) {
        let f = m.fonts();
        let cx = rect.centerx();
        m.text(cv, &tr("Download real neuron shapes?"), (cx, rect.y + 40), INK, &f.head, Anchor::MidTop);
        m.wrapped(
            cv,
            &tr("The game can draw real neurons from their electron-microscopy skeletons (Janelia's MaleCNS v1.0, CC BY 4.0): ten in the brain view, and any neuron you inspect, fetched when you click it (10-100 kB each), checksummed and cached. Without them it draws estimated fibers; every neuron is still simulated as a point. This is the game's only network use besides Streamer mode, so it is off unless you say yes. Settings > Brain changes it later; it applies on the next launch."),
            (rect.x + 60, rect.y + 96),
            rect.w - 120,
            TEXT,
            &f.text,
            9,
        );
        let bw = m.bw(&tr("Yes, download them"), 240).max(m.bw(&tr("No thanks"), 240));
        m.button(
            cv,
            Rect::new(cx - 10 - bw, rect.bottom() - 90, bw, 50),
            &tr("Yes, download them"),
            Act::Custom("yes".into()),
            "ns:yes",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(cx + 10, rect.bottom() - 90, bw, 50),
            &tr("No thanks"),
            Act::Custom("no".into()),
            "ns:no",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if name == "yes" || name == "no" {
            m.set_setting(ctx.cfg, "brain.neuron_shapes", &Value::Bool(name == "yes"), true);
            m.back();
        }
    }
}
