//! Esc > Mic and streamer (`game/live_inputs.py:page`): the two opt-in inputs. Both are off every time the game starts.
//!
//! Host: query `lab.live_state` = `{mic_on, mic_error, mic_ok, mic_why, peak_hz, bins[24], drive_a, drive_b, max_current, sens, stream_on,
//! stream_error, channel, chat: {state, desc, commands} | null, lab_locked, net_ok, net_why, lab_lock_text}`; commands `lab.live_mic {on}`,
//! `lab.live_stream {on}`, `lab.live_channel {text}`. The tuning (sensitivity, which commands count, window, cooldown, minimum votes) is
//! stored in the config by the page itself.

use serde_json::{Value, json};

use super::toolkit::{back, title};
use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

#[derive(Default)]
pub struct LiveInputs;

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

impl Page for LiveInputs {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[allow(clippy::too_many_lines)]
    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.live_state");
        let mut y = title(
            m,
            cv,
            rect,
            "MICROPHONE AND STREAMER MODE",
            "Two opt-in inputs. Both are off every time the game starts, both show a red indicator on screen while they are on, and neither is ever on during validation, assays, protocols or tests.",
            &[("GAME RULE", "every rule here is a game rule"), ("CONNECTOME", "mic: the JO-A/B neurons it drives")],
        );
        let x = rect.x + 24;
        let lw = ["Microphone", "Sensitivity", "Streamer mode", "Viewers may vote", "Vote window"].iter().map(|t| f.text.size(t)).max().unwrap_or(120).max(134) + 16;
        let c = x + lw;
        // --- microphone
        m.text(cv, "MICROPHONE  ->  JOHNSTON'S ORGAN (JO-A and JO-B)", (x, y), INK, &f.text, Anchor::TopLeft);
        y += 28;
        let mic_on = q["mic_on"].as_bool().unwrap_or(false);
        let mic_ok = q["mic_ok"].as_bool().unwrap_or(true);
        m.text(cv, "Microphone", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        m.toggle(
            cv,
            Rect::new(c, y, 50, 28),
            mic_on,
            Act::Custom(format!("mic:{}", u8::from(!mic_on))),
            "live_mic",
            mic_ok || mic_on,
            "Off every time the game starts. Sound is analysed in memory and thrown away: nothing is recorded, saved or sent.",
        );
        let err = q["mic_error"].as_str().unwrap_or("");
        if mic_on {
            m.text(cv, "ON: listening. Nothing is saved or sent.", (c + 120, y + 14), [255, 140, 140], &f.small, Anchor::MidLeft);
        } else {
            let why = if err.is_empty() { q["mic_why"].as_str().unwrap_or("") } else { err };
            m.text(cv, why, (c + 120, y + 14), if !err.is_empty() || !mic_ok { BAD } else { LABEL }, &f.small, Anchor::MidLeft);
        }
        y += 36;
        m.text(cv, "Sensitivity", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        m.setting_slider(
            cv,
            Rect::new(c, y, 240, 28),
            ctx.cfg,
            "brain.mic_sensitivity",
            true,
            "How loud a band must be to drive its neurons fully (game rule). Saved; the switch is not.",
        );
        y += 40;
        let box_ = Rect::new(x, y, rect.w - 48, 104);
        cv.fill_rect(box_, [12, 14, 20], 8);
        let peak = num(&q["peak_hz"]);
        m.text(cv, "Hum test", (box_.x + 12, box_.y + 8), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, &if peak > 0.0 { format!("{peak:.0} Hz") } else { "- Hz".into() }, (box_.x + 12, box_.y + 30), INK, &f.head, Anchor::TopLeft);
        let bins: Vec<f64> = q["bins"].as_array().map_or_else(|| vec![0.0; 24], |a| a.iter().map(num).collect());
        let bw = (box_.w - 380) as f64 / 24.0;
        for (i, v) in bins.iter().enumerate() {
            let h = (8.0 + 70.0 * v) as i32;
            cv.fill_rect(Rect::new(box_.x + 190 + (i as f64 * bw) as i32, box_.bottom() - 10 - h, (bw - 2.0).max(2.0) as i32, h), ACCENT, 0);
        }
        let maxc = num(&q["max_current"]).max(1e-6);
        m.text(cv, &format!("JO-A drive {:4.0}%", num(&q["drive_a"]) / maxc * 100.0), (box_.right() - 180, box_.y + 12), TEXT, &f.small, Anchor::MidTop);
        m.text(cv, &format!("JO-B drive {:4.0}%", num(&q["drive_b"]) / maxc * 100.0), (box_.right() - 180, box_.y + 36), TEXT, &f.small, Anchor::MidTop);
        m.wrapped(
            cv,
            "B prefers below ~100 Hz, A higher (LITERATURE); the split is a game rule",
            (box_.right() - 340, box_.y + 60),
            320,
            LABEL,
            &f.small,
            2,
        );
        y = box_.bottom() + 8;
        let demo = "Run the hum demo (no microphone)";
        m.button(
            cv,
            Rect::new(x, y, 330.max(f.bold.size(demo) + 32), 34),
            demo,
            Act::Custom("demo".into()),
            "live_demo",
            Style::Normal,
            true,
            Some("A synthetic 200 Hz hum, steady and in 35 ms pulses, through the same analysis, onto real JO-A/B: does it reach the courtship pathway?"),
            false,
            None,
        );
        y += 48;
        // --- streamer mode
        m.text(cv, "STREAMER MODE (Twitch chat, read-only)", (x, y), INK, &f.text, Anchor::TopLeft);
        y += 28;
        m.text(cv, "Streamer mode", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        let on = q["stream_on"].as_bool().unwrap_or(false);
        let net_ok = q["net_ok"].as_bool().unwrap_or(true);
        let locked = q["lab_locked"].as_bool().unwrap_or(false);
        m.toggle(
            cv,
            Rect::new(c, y, 50, 28),
            on,
            Act::Custom(format!("stream:{}", u8::from(!on))),
            "live_stream",
            (net_ok && !locked) || on,
            "Off every time the game starts. Connects to irc.chat.twitch.tv (port 6697) anonymously and only reads: it never logs in, never sends, never stores a token. The connection is shown on screen while it is open.",
        );
        m.text(cv, "channel", (c + 130, y + 14), TEXT, &f.small, Anchor::MidLeft);
        m.text_field(
            cv,
            Rect::new(c + 195, y, 220, 28),
            q["channel"].as_str().unwrap_or(""),
            "live_chan",
            Some("A Twitch channel name, for example  mychannel . Letters, digits and underscores only."),
            26,
            false,
        );
        y += 34;
        let (status, col) = match q["chat"].as_object() {
            Some(ch) if on => (
                format!(
                    "{}: {}   commands read: {}",
                    ch["state"].as_str().unwrap_or(""),
                    ch["desc"].as_str().unwrap_or(""),
                    ch["commands"].as_u64().unwrap_or(0)
                ),
                [255, 140, 140],
            ),
            _ => {
                let e = q["stream_error"].as_str().unwrap_or("");
                let s = if !e.is_empty() {
                    e.to_string()
                } else if locked {
                    q["lab_lock_text"].as_str().unwrap_or("").to_string()
                } else if net_ok {
                    String::new()
                } else {
                    q["net_why"].as_str().unwrap_or("").to_string()
                };
                (s, if e.is_empty() { LABEL } else { BAD })
            }
        };
        if !status.is_empty() {
            m.wrapped(cv, &status, (c, y), rect.right() - 24 - c, col, &f.small, 2);
        }
        y += 2 * f.small.linesize + 4;
        m.text(cv, "Viewers may vote", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        let mut cx = c;
        for name in ["tool", "arena", "surgery"] {
            let key = format!("stream.allow_{name}");
            let cur = ctx.cfg.get_bool(&key);
            m.text(cv, &format!("!{name}"), (cx, y + 14), TEXT, &f.small, Anchor::MidLeft);
            m.toggle(
                cv,
                Rect::new(cx + 90, y, 50, 28),
                cur,
                Act::Set { key, value: json!(!cur), save: true },
                &format!("live_allow_{name}"),
                true,
                &format!("Whether viewers can change the {name} by voting. You choose; the default is tools only."),
            );
            cx += 210;
        }
        y += 36;
        m.text(cv, "Vote window", (x, y + 14), TEXT, &f.text, Anchor::MidLeft);
        m.setting_slider(cv, Rect::new(c, y, 160, 28), ctx.cfg, "stream.window_s", true, "");
        let mut sx = c + 180;
        m.text(cv, "cooldown", (sx, y + 14), TEXT, &f.small, Anchor::MidLeft);
        sx += 80.max(f.small.size("cooldown") + 14);
        m.setting_slider(cv, Rect::new(sx, y, 160, 28), ctx.cfg, "stream.cooldown_s", true, "");
        sx += 180;
        m.text(cv, "min votes", (sx, y + 14), TEXT, &f.small, Anchor::MidLeft);
        sx += 80.max(f.small.size("min votes") + 14);
        m.setting_slider(cv, Rect::new(sx, y, 140, 28), ctx.cfg, "stream.min_votes", true, "");
        y += 38;
        m.wrapped(
            cv,
            "Only chat lines that start with ! are read, for one frame, to be counted. A viewer's name is hashed with a random salt and held only for the round (and the 2 s flood limit); the screen shows counts, never names. YouTube is not supported (it can't be done without storing a key).",
            (x, y),
            rect.w - 60,
            LABEL,
            &f.small,
            3,
        );
        back(m, cv, rect, "live");
    }

    fn custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        match name {
            n if n.starts_with("mic:") => drop(ctx.host.command("lab.live_mic", json!({"on": &n[4..] == "1"}))),
            n if n.starts_with("stream:") => {
                // the channel box commits on Enter; a click on the switch uses whatever is in it
                drop(ctx.host.command("lab.live_stream", json!({"on": &n[7..] == "1"})));
            }
            "demo" => {
                if let Some(a) = m.page_mut::<super::lab::LabAssays>("lab_assays") {
                    a.set_kind("hum_demo");
                }
                m.show("lab_assays");
            }
            "text:live_chan" => drop(ctx.host.command("lab.live_channel", json!({"text": text.unwrap_or("")}))),
            _ => {}
        }
    }
}
