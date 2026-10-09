//! What's New in 3.0 (`ui/whatsnew_ui.py`): one skippable screen, shown once after an upgrade (`first_run.whatsnew_3_0_seen`); Settings >
//! Help opens it again without touching the key.

use crate::canvas::{Canvas, Rect};
use crate::i18n::tr;
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

/// (heading, text, where). Short on purpose: the full list is docs/changelog.md.
const ITEMS: [(&str, &str, &str); 8] = [
    (
        "Neurodex, Kill cam, Neuron of the Day",
        "Collect the cell types your fly fires, watch the last seconds of its brain in slow motion, and meet one curated neuron a day.",
        "D, ;, a card at launch",
    ),
    (
        "Lab toolkit",
        "Split-GAL4 driver lines, thermogenetics, a virtual patch clamp, simulated calcium imaging and pharmacology, all tagged CONNECTOME, GAME RULE or MODEL PREDICTION.",
        "Esc > Lab tools",
    ),
    (
        "Live inputs, predators, weather, kitchen",
        "Opt-in microphone and streamer mode (off at every launch), frog, dragonfly and mantis, rain and storms, and a kitchen arena.",
        "Esc > Mic and streamer; tools; the arenas",
    ),
    (
        "Fly arcade",
        "A tournament of flies and fly racing, each brain steering and walking for itself, with in-game points only (no money, nothing to buy). Cards are measured, never drawn.",
        "Esc > Fly arcade",
    ),
    (
        "Network science, sleep deprivation, sensitivity analysis",
        "Motifs, rich club and communities of the connectome; a sleep-deprivation assay; and how fragile each validated behavior is to each model parameter.",
        "Lab",
    ),
    (
        "Behavior rigs",
        "A tethered flight simulator, a fly on a ball, Buridan's paradigm and a four-field olfactory arena. Yaw and walking are read from real descending neurons; each rig has a headless protocol and a pre-registered assay, and a result the model misses is reported as a miss.",
        "Lab > Behavior rigs",
    ),
    (
        "Mini-papers",
        "Short guided experiments that reproduce a classic paper step by step: you state a hypothesis, run it on the model, and see your result next to what the paper found, cited. Where the model fails the paper, it says so and why.",
        "Esc > Mini-papers; Lab",
    ),
    (
        "Honest by construction",
        "Every new behavior is tagged. Validation, assays and rigs keep their pass criteria written down before the run, and failed results stay in the table.",
        "README, Lab > Model assumptions",
    ),
];

pub struct WhatsNew;

impl Page for WhatsNew {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, _ctx: &mut Ctx) {
        let key = "whatsnew";
        let f = m.fonts();
        m.text(cv, &tr("What's new in Kick the Fly 3.0"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        m.wrapped(
            cv,
            &tr("Everything below is optional, and nothing changed what your fly does by itself. This screen shows once."),
            (rect.x + 24, rect.y + 52),
            rect.w - 48,
            LABEL,
            &f.small,
            3,
        );
        let body = Rect::new(rect.x + 16, rect.y + 96, rect.w - 32, rect.h - 96 - 80);
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut y = body.y + 4 - off;
        let width = body.w - 40;
        for (head, text, wher) in ITEMS {
            m.text(cv, &tr(head), (body.x + 8, y), AMBER, &f.bold, Anchor::TopLeft);
            y += f.bold.linesize + 2;
            y = m.wrapped(cv, &tr(text), (body.x + 8, y), width, TEXT, &f.text, 14);
            y = m.wrapped(cv, &format!("{} {}", tr("Where:"), tr(wher)), (body.x + 8, y + 2), width, LABEL, &f.small, 5) + 12;
        }
        m.end_clip(cv);
        let ch = (y + off - body.bottom() + 8).max(0);
        m.set_content_h(key, ch);
        m.scrollbar(cv, body, ch, off);
        m.button(
            cv,
            Rect::new(rect.centerx() - 130, rect.bottom() - 62, 260, 46),
            &tr("Got it"),
            Act::Back,
            "whatsnew:ok",
            Style::Primary,
            true,
            Some(&tr("Close this. It will not show again; Settings > Help brings it back.")),
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, _name: &str, _text: Option<&str>, _ctx: &mut Ctx) {}
}
