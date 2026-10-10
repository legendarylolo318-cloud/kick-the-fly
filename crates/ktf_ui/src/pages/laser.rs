//! Lab > Targeted optogenetics laser (`lab/lablaser.py`): the laser tool's target, effect, intensity and trigger.
//!
//! Host: query `lab.laser_state` = `{target, mode, intensity, trigger, pulse_ms, current, has_tool, equipped}`; command `lab.laser_set
//! {target?, mode?, intensity?, trigger?, pulse_ms?}`, `lab.laser_equip`.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const TARGETS: [&str; 11] = ["LC10", "LPLC2", "LC4", "dnp01", "DNg02", "MDN", "aDN1", "MN9", "PPL1", "PAM", "DNa02"];

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

#[derive(Default)]
pub struct LabLaser;

impl Page for LabLaser {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.laser_state");
        let target = q["target"].as_str().unwrap_or("dnp01").to_string();
        let mode = q["mode"].as_str().unwrap_or("activate");
        let trigger = q["trigger"].as_str().unwrap_or("hold");
        m.text(cv, "TARGETED OPTOGENETICS LASER", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let y = m.subtitle(cv, rect, "In-world aimable laser for real-time cellular stimulation and silencing without menus.", None, 2);
        let note = "TARGETED CELLULAR STIMULATION / SILENCING: Injects current directly into connectome rows. Does not model Gal4/UAS drivers or opsin kinetics.";
        let n = Menu::fit_lines(&f.small, note, rect.w - 68, 4).0.len() as i32;
        let banner = Rect::new(rect.x + 24, y, rect.w - 48, n * f.small.linesize + 10);
        cv.fill_rect(banner, [20, 32, 45], 6);
        cv.stroke_rect(banner, [60, 140, 220], 6);
        m.wrapped(cv, note, (banner.x + 10, banner.y + 5), banner.w - 20, [160, 210, 255], &f.small, 4);
        let mut y = banner.bottom() + 10;
        let lw = ["Target cell type:", "Laser effect:", "Intensity:", "Trigger mode:", "Pulse duration:"]
            .iter()
            .map(|t| f.text.size(t))
            .max()
            .unwrap_or(0)
            + 20;
        let cx = rect.x + 24 + lw;
        let ch = 32.max(f.small.linesize + 12);
        m.text(cv, "Target cell type:", (rect.x + 24, y + 4), TEXT, &f.text, Anchor::TopLeft);
        m.text(cv, &format!("Active: {target}"), (cx, y + 4), INK, &f.bold, Anchor::TopLeft);
        y += 36.max(f.bold.linesize + 12);
        let items: Vec<_> = TARGETS
            .iter()
            .map(|t| (t.to_string(), Act::Custom(format!("target:{t}")), format!("chip:{t}"), target.eq_ignore_ascii_case(t), None))
            .collect();
        y = m.flow_buttons(cv, rect.x + 24, y, rect.right() - 24, &items, Some(24.max(f.small.linesize + 6)), None, 8, None) + 14;
        let seg_w = 280.max(
            2 * ["Activate (+current)", "Silence (-current)", "Hold (continuous)", "Pulse (timed)"]
                .iter()
                .map(|t| f.small.size(t))
                .max()
                .unwrap_or(0)
                + 40,
        );
        m.text(cv, "Laser effect:", (rect.x + 24, y + ch / 2), TEXT, &f.text, Anchor::MidLeft);
        m.segmented(
            cv,
            Rect::new(cx, y, seg_w, ch),
            &["Activate (+current)".into(), "Silence (-current)".into()],
            usize::from(mode != "activate"),
            &[Act::Custom("mode:activate".into()), Act::Custom("mode:silence".into())],
            "laser_mode",
            true,
            "",
        );
        let col = if mode == "activate" { [255, 160, 60] } else { [80, 180, 255] };
        m.text(cv, &format!("{:+.2} pA/step", num(&q["current"])), (cx + seg_w + 20, y + ch / 2), col, &f.bold, Anchor::MidLeft);
        y += ch + 12;
        m.text(cv, "Intensity:", (rect.x + 24, y + ch / 2), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(cx, y, seg_w, ch),
            num(&q["intensity"]),
            (0.1, 3.0, 0.1),
            "{:.1f}x",
            "laser_intensity",
            true,
            Some("Current intensity scale multiplier (0.1x to 3.0x)"),
        );
        y += ch + 12;
        m.text(cv, "Trigger mode:", (rect.x + 24, y + ch / 2), TEXT, &f.text, Anchor::MidLeft);
        m.segmented(
            cv,
            Rect::new(cx, y, seg_w, ch),
            &["Hold (continuous)".into(), "Pulse (timed)".into()],
            usize::from(trigger != "hold"),
            &[Act::Custom("trigger:hold".into()), Act::Custom("trigger:pulse".into())],
            "laser_trigger",
            true,
            "",
        );
        y += ch + 12;
        if trigger == "pulse" {
            m.text(cv, "Pulse duration:", (rect.x + 24, y + ch / 2), TEXT, &f.text, Anchor::MidLeft);
            m.page_slider(
                cv,
                Rect::new(cx, y, seg_w, ch),
                num(&q["pulse_ms"]),
                (50.0, 1000.0, 50.0),
                "{:.0f} ms",
                "laser_pulse_dur",
                true,
                Some("Duration of single laser pulse upon trigger"),
            );
            y += ch + 12;
        }
        y += 10;
        if q["has_tool"].as_bool().unwrap_or(false) {
            let equipped = q["equipped"].as_bool().unwrap_or(false);
            let txt = if equipped { "Equipped in Hand" } else { "Equip Laser Tool in World" };
            m.button(
                cv,
                Rect::new(rect.x + 24, y, 240.max(f.bold.size(txt) + 32), 42),
                txt,
                Act::Custom("equip".into()),
                "laser_equip",
                Style::Primary,
                true,
                Some("Equips the laser tool in hand for immediate 2D or 3D in-world use."),
                equipped,
                None,
            );
        }
        y += 56;
        m.wrapped(
            cv,
            "In-World Usage: Aim crosshair/cursor at the fly's body and click/hold. Beam shines cyan for silencing and orange for activation.",
            (rect.x + 24, y),
            rect.w - 220,
            LABEL,
            &f.small,
            3,
        );
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "laser:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let val = text.and_then(|t| t.parse::<f64>().ok());
        if let Some(t) = name.strip_prefix("target:") {
            ctx.host.command("lab.laser_set", json!({"target": t}));
        } else if let Some(t) = name.strip_prefix("mode:") {
            ctx.host.command("lab.laser_set", json!({"mode": t}));
        } else if let Some(t) = name.strip_prefix("trigger:") {
            ctx.host.command("lab.laser_set", json!({"trigger": t}));
        } else if name == "slide:laser_intensity" {
            if let Some(v) = val {
                ctx.host.command("lab.laser_set", json!({"intensity": v}));
            }
        } else if name == "slide:laser_pulse_dur" {
            if let Some(v) = val {
                ctx.host.command("lab.laser_set", json!({"pulse_ms": v}));
            }
        } else if name == "equip" {
            ctx.host.command("lab.laser_equip", Value::Null);
            m.back();
        }
    }
}
