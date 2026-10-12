//! Pause menu > Challenges (`lab/challenges.py:page_challenges`): one card per game built on a real fly experiment, with the best score,
//! its stars and a Start button. The game answers query `challenge_scores` (`{key: best}`) and runs `Menu::action("start_challenge:<key>")`.

use crate::canvas::{Canvas, Rect};
use crate::hud::draw_stars;
use crate::i18n::{tr, tr_args};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

/// (key, title, description, unit, better).
pub const INFO: [(&str, &str, &str, &str, &str); 7] = [
    (
        "tmaze",
        "Teach it to pick the right door",
        "Choose which door gives a zap. Train the fly, then watch it choose a door 10 times.",
        "right choices",
        "high",
    ),
    (
        "sneak",
        "How close can you sneak?",
        "Creep up on the fly. Move fast and its escape neuron fires and it dodges. How close can you get?",
        "fly lengths away",
        "low",
    ),
    (
        "sweet",
        "Find its sweet tooth",
        "Offer sugar at different strengths. Find the weakest sugar it still reaches for, in 8 tries.",
        "% sugar",
        "low",
    ),
    (
        "reverse_surgery",
        "Mystery defect (Reverse surgery)",
        "One brain circuit is turned off! Test the fly with tools, ask for hints, and deduce what's missing.",
        "stars",
        "high",
    ),
    (
        "puppeteer",
        "Puppeteer",
        "You cannot touch the fly. Steer it only with the optogenetics laser and by switching its real neurons on and off: ten puzzles, each with a par and the circuit that solves it.",
        "levels",
        "high",
    ),
    (
        "contraption",
        "Contraption builder",
        "A sandbox: place ramps, dominoes, springs, fans, lamps, sugar, tool triggers, buttons and timers, then run the machine against the fly.",
        "runs",
        "high",
    ),
    (
        "predict",
        "Predict the move (Motor readouts)",
        "A descending motor spike surge flashes on the monitor. Can you predict the fly's move before it triggers?",
        "correct predictions",
        "high",
    ),
];

/// `challenges.stars`.
pub fn stars(key: &str, v: f64) -> usize {
    match key {
        "tmaze" => {
            if v >= 9.0 {
                3
            } else if v >= 7.0 {
                2
            } else {
                usize::from(v >= 5.0)
            }
        }
        "sneak" => {
            if v <= 1.0 {
                3
            } else if v <= 2.0 {
                2
            } else {
                usize::from(v <= 3.5)
            }
        }
        "puppeteer" => {
            if v >= 10.0 {
                3
            } else if v >= 7.0 {
                2
            } else {
                usize::from(v >= 3.0)
            }
        }
        "contraption" => 0,
        "reverse_surgery" | "mystery" => {
            if v >= 3.0 {
                3
            } else if v >= 2.0 {
                2
            } else {
                usize::from(v >= 1.0)
            }
        }
        "predict" => {
            if v >= 8.0 {
                3
            } else if v >= 6.0 {
                2
            } else {
                usize::from(v >= 4.0)
            }
        }
        _ => {
            if v <= 10.0 {
                3
            } else if v <= 25.0 {
                2
            } else {
                usize::from(v <= 50.0)
            }
        }
    }
}

pub struct ChallengesPage;

impl Page for ChallengesPage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        m.text(cv, &tr("CHALLENGES"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let top = (rect.y + 80).max(m.subtitle(
            cv,
            rect,
            &tr("Games built on real fly experiments. The fly's brain decides how it does."),
            Some(rect.y + 50),
            2,
        ));
        let body = Rect::new(rect.x + 16, top, rect.w - 32, rect.bottom() - 70 - top);
        let key = "challenges";
        let off = m.scroll_offset(key);
        let scores = ctx.host.query("challenge_scores");
        m.begin_clip(cv, body);
        let mut y = body.y + 4 - off;
        for (c_key, title, desc, _unit, _better) in INFO {
            let n_desc = Menu::fit_lines(&f.text, &tr(desc), body.w - 56, 3).0.len() as i32;
            let extra = (n_desc - 1) * f.text.linesize;
            let card = Rect::new(body.x + 8, y, body.w - 16, 150 + extra);
            cv.fill_rect(card, [28, 32, 42], 12);
            m.text(cv, &tr(title), (card.x + 20, card.y + 16), INK, &f.head, Anchor::TopLeft);
            m.wrapped(cv, &tr(desc), (card.x + 20, card.y + 54), card.w - 40, TEXT, &f.text, 3);
            let card = Rect::new(card.x, card.y + extra, card.w, card.h);
            match scores[c_key].as_f64() {
                Some(best) => {
                    let shown = match c_key {
                        "tmaze" => format!("{best:.0}/10"),
                        "sneak" => format!("{best:.1} fly lengths"),
                        "reverse_surgery" => format!("{best:.0} stars"),
                        "puppeteer" => format!("{best:.0}/10 levels"),
                        "contraption" => format!("{best:.0} runs"),
                        _ => format!("{best:.0}%"),
                    };
                    m.text(
                        cv,
                        &tr_args("Best: {shown}", &[("shown", &shown)]),
                        (card.x + 20, card.y + 96),
                        AMBER,
                        &f.bold,
                        Anchor::TopLeft,
                    );
                    if c_key != "contraption" {
                        draw_stars(cv, ((card.x + 250) as f32, (card.y + 106) as f32), stars(c_key, best), 11);
                    }
                }
                None => {
                    m.text(cv, &tr("Not played yet"), (card.x + 20, card.y + 96), LABEL, &f.text, Anchor::TopLeft);
                }
            }
            m.button(
                cv,
                Rect::new(card.right() - 170, card.y + 86, 150, 46),
                &tr("Start"),
                Act::Custom(format!("start:{c_key}")),
                &format!("challenge:{c_key}"),
                Style::Primary,
                true,
                None,
                false,
                None,
            );
            y += 166 + extra;
        }
        m.end_clip(cv);
        let ch = (y + off - body.bottom() + 8).max(0);
        m.set_content_h(key, ch);
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            &tr("Back"),
            Act::Back,
            "ch:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, _text: Option<&str>, _ctx: &mut Ctx) {
        if let Some(k) = name.strip_prefix("start:") {
            m.action(&format!("start_challenge:{k}"));
        }
    }
}
