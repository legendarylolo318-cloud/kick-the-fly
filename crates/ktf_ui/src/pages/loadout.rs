//! The tool loadout editor (default key Q) and the one-time notice for migrated configs (`ui/loadout_ui.py`, 2.13).
//!
//! Shows every tool grouped by category with its icon, name, one-line description and which real neurons it drives (tagged CONNECTOME or
//! GAME RULE), the hotbar, the presets and up to five saved custom loadouts. Click a card to equip or unequip it, drag a card onto the
//! hotbar to put it in a slot, drag hotbar slots to reorder, drag one off the hotbar to remove it. The hand can't be removed. Every
//! edit turns the preset into "Custom" and is saved at once.
//!
//! The page needs from the game: query `selected_tool` (the tool in hand), command-less `Menu::action("select_tool:<name>")`.

use ktf_data::config::{self, Config};
use ktf_data::loadout::{self, HAND};
use serde_json::Value;

use crate::canvas::{Canvas, Rect};
use crate::hud::draw_icon;
use crate::i18n::{tr, tr_args};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

struct Palette {
    select: Color,
    real: Color,
    rule: Color,
    ok: Color,
}

fn palette(cfg: &Config) -> Palette {
    match cfg.get_str("access.palette") {
        "blue-yellow" => Palette { select: [255, 224, 70], real: [70, 130, 240], rule: [255, 224, 70], ok: [70, 130, 240] },
        "high-contrast" => Palette { select: [255, 255, 255], real: [255, 255, 255], rule: [255, 60, 220], ok: [255, 255, 255] },
        _ => Palette { select: [255, 176, 64], real: [60, 170, 220], rule: [220, 150, 50], ok: [90, 200, 120] },
    }
}

fn draw_tag(m: &Menu, cv: &mut Canvas, pos: (i32, i32), tag: &str, cfg: &Config) -> Rect {
    let f = m.fonts().small;
    let label = tr(tag);
    let r = Rect::new(pos.0, pos.1, f.size(&label) + 10, f.linesize + 2);
    let p = palette(cfg);
    cv.fill_rect(r, if tag == "CONNECTOME" { p.real } else { p.rule }, 5);
    f.draw(cv, &label, r.centerx() - f.size(&label) / 2, r.centery() - f.linesize / 2, [12, 14, 18]);
    r
}

/// The key an action is on, as the player sees it, or where to bind it when it has none.
pub fn key_label(cfg: &Config, action: &str) -> String {
    let k = cfg.key_of(action);
    if k.is_empty() { tr("(unbound: Settings > Controls)") } else { k.to_uppercase() }
}

fn tool_short_label(name: &str) -> String {
    config::schema().toolbar.iter().find(|t| t.0 == name).map(|t| t.1.clone()).unwrap_or_else(|| name.to_uppercase())
}

#[derive(Default)]
pub struct LoadoutPage {
    lo_from: Option<String>,
    lo_name: String,
}

fn current(cfg: &Config) -> Vec<String> {
    loadout::resolve(cfg).tools
}

/// Save this list as the custom loadout, switch to it, and apply it right away (`commit`).
fn commit(m: &mut Menu, cfg: &mut Config, tools: &[String]) {
    let (lab, larva) = (cfg.lab(), cfg.larva());
    let tools = loadout::clean(tools, lab, larva);
    let hidden: Vec<String> = cfg
        .custom_loadout
        .iter()
        .filter(|n| config::schema().loadout.tool_names.contains(n) && !tools.contains(n) && !loadout::available(n, lab, larva))
        .cloned()
        .collect();
    let all: Vec<String> = tools.into_iter().chain(hidden).collect();
    cfg.set_custom_loadout(&all);
    m.set_setting(cfg, "controls.loadout_preset", &Value::from("custom"), false);
    m.save_config();
}

fn toggle(m: &mut Menu, cfg: &mut Config, name: &str) {
    let mut tools = current(cfg);
    if name == HAND {
        m.flash(&tr("The hand is in every loadout."), AMBER, 4.0);
    } else if let Some(i) = tools.iter().position(|t| t == name) {
        tools.remove(i);
        commit(m, cfg, &tools);
    } else {
        tools.push(name.into());
        commit(m, cfg, &tools);
    }
}

/// A dragged card lands on hotbar position `index`: a card from the grid equips, a hotbar slot reorders.
fn drop_on_slot(m: &mut Menu, cfg: &mut Config, payload: &str, index: usize) {
    let name = payload.split('|').nth(1).unwrap_or("").to_string();
    let mut tools = current(cfg);
    if let Some(i) = tools.iter().position(|t| *t == name) {
        if name == HAND {
            return;
        }
        tools.remove(i);
    }
    let at = index.clamp(1, tools.len());
    tools.insert(at, name);
    commit(m, cfg, &tools);
}

fn drop_off(m: &mut Menu, cfg: &mut Config, payload: &str) {
    let mut parts = payload.split('|');
    let name = parts.nth(1).unwrap_or("").to_string();
    let from_slot = parts.next().unwrap_or("-") != "-";
    if from_slot && name != HAND {
        let mut tools = current(cfg);
        if let Some(i) = tools.iter().position(|t| *t == name) {
            tools.remove(i);
            commit(m, cfg, &tools);
        }
    }
}

fn why_unavailable(cfg: &Config, t: &config::ToolInfo) -> String {
    if t.lab_only && !cfg.lab() {
        return tr("Lab mode only");
    }
    if cfg.larva() && !t.larva {
        return tr("No larval sensory mapping") + &if t.larva_note.is_empty() { String::new() } else { format!(": {}", tr(&t.larva_note)) };
    }
    String::new()
}

impl Page for LoadoutPage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let cfg: &Config = ctx.cfg;
        let pal = palette(cfg);
        let ls = &config::schema().loadout;
        let cur_preset = cfg.get_str("controls.loadout_preset").to_string();
        if self.lo_from.is_none() {
            self.lo_from = Some(cur_preset.clone());
        }
        let lop = loadout::resolve(cfg);
        let selected = ctx.host.query("selected_tool").as_str().unwrap_or("hand").to_string();
        m.text(cv, &tr("TOOL LOADOUT"), (rect.x + 24, rect.y + 14), INK, &f.head, Anchor::TopLeft);
        let mode = cfg.get_str("brain.mode");
        let mut mode_cap = mode.to_string();
        mode_cap[..1].make_ascii_uppercase();
        let sub = tr_args("{mode} mode", &[("mode", &tr(&mode_cap))])
            + &if cfg.larva() { format!("  ·  {}", tr("larva: tools without a larval mapping are hidden")) } else { String::new() };
        m.text(cv, &sub, (rect.right() - 24, rect.y + 20), LABEL, &f.small, Anchor::TopRight);

        // presets
        let mut y = rect.y + 52;
        let n = ls.choices.len() as i32;
        let w = (rect.w - 48) / n;
        for (i, name) in ls.choices.iter().enumerate() {
            m.button(
                cv,
                Rect::new(rect.x + 24 + i as i32 * w, y, w - 6, 34),
                &tr(&ls.preset_labels[name]),
                Act::Custom(format!("preset:{name}")),
                &format!("lo-preset:{name}"),
                Style::Normal,
                true,
                Some(&tr(&ls.preset_tips[name])),
                *name == cur_preset,
                Some(f.small.clone()),
            );
        }

        // saved loadouts
        y += 44;
        let saved = &cfg.saved_loadouts;
        let max_saved = config::schema().max_saved_loadouts as i32;
        m.text(cv, &tr("SAVED"), (rect.x + 24, y + 8), LABEL, &f.small, Anchor::TopLeft);
        let sw = (rect.w - 48 - 70) / max_saved;
        for i in 0..max_saved {
            let x = rect.x + 24 + 70 + i * sw;
            if let Some(item) = saved.get(i as usize) {
                m.button(
                    cv,
                    Rect::new(x, y, sw - 36, 30),
                    &item.name,
                    Act::Custom(format!("load:{i}")),
                    &format!("lo-saved:{i}"),
                    Style::Normal,
                    true,
                    Some(&tr_args("Load this loadout: {tools}", &[("tools", &item.tools.join(", "))])),
                    false,
                    Some(f.small.clone()),
                );
                m.button(
                    cv,
                    Rect::new(x + sw - 34, y, 28, 30),
                    "x",
                    Act::Custom(format!("del:{i}")),
                    &format!("lo-del:{i}"),
                    Style::Danger,
                    true,
                    Some(&tr("Delete this saved loadout.")),
                    false,
                    Some(f.small.clone()),
                );
            } else {
                cv.stroke_rect(Rect::new(x, y, sw - 6, 30), BORDER, 8);
                m.text(cv, &tr("empty"), (x + (sw - 6) / 2, y + 15), DIM, &f.small, Anchor::Center);
            }
        }

        // the hotbar
        y += 42;
        let hb = m.text(cv, &tr("HOTBAR"), (rect.x + 24, y), LABEL, &f.small, Anchor::TopLeft);
        y = (y + 22).max(
            m.wrapped(
                cv,
                &tr("Drag to reorder, drag off to remove, drop a card here to add. The hand always stays in slot 1."),
                (hb.right() + 16, y),
                rect.right() - 24 - hb.right() - 16,
                DIM,
                &f.small,
                2,
            ) + 4,
        );
        let tools = &lop.tools;
        let rows = lop.n_pages().max(1) as i32;
        let page_size = ls.page_size as i32;
        let page_w = if rows > 1 {
            (0..rows).map(|r| f.small.size(&tr_args("page {n}", &[("n", &(r + 1).to_string())]))).max().unwrap_or(0) + 12
        } else {
            0
        };
        let (slot_w, slot_h) = (84.min((rect.w - 48 - page_w) / page_size), 58);
        let x0 = rect.x + 24;
        let strip = Rect::new(x0, y, 10 * slot_w, rows * (slot_h + 6));
        m.register_drop(strip, "lo-strip", &format!("slot:{}", tools.len()));
        for i in 0..rows * page_size {
            let (col, row) = (i % page_size, i / page_size);
            let r = Rect::new(x0 + col * slot_w, y + row * (slot_h + 6), slot_w - 6, slot_h);
            let key = cfg.key_of(&format!("slot{}", col + 1)).to_uppercase();
            if let Some(name) = tools.get(i as usize) {
                let on = *name == selected;
                cv.fill_rect(r, if on { [60, 46, 22] } else { [24, 28, 38] }, 8);
                cv.stroke_rect2(r, if on { pal.select } else { BORDER }, if on { 2 } else { 1 }, 8);
                draw_icon(cv, name, (r.centerx() as f32, (r.y + 24) as f32), if on { pal.select } else { TEXT });
                let short = Menu::fit_lines(&f.small, &tool_short_label(name), r.w - 6, 1).0.remove(0);
                m.text(cv, &short, (r.centerx(), r.y + 40), TEXT, &f.small, Anchor::MidTop);
                m.text(cv, &key, (r.x + 5, r.y + 3), LABEL, &f.small, Anchor::TopLeft);
                m.register_drop(r, &format!("lo-drop:{i}"), &format!("slot:{i}"));
                let desc = tr(&ls.catalog.iter().find(|t| &t.name == name).map(|t| t.desc.clone()).unwrap_or_default());
                m.register_drag_src(
                    r,
                    &format!("lo-slot:{i}"),
                    &format!("tool|{name}|{i}"),
                    Some(Act::Custom(format!("select:{name}"))),
                    Some(&desc),
                    Some("dropoff"),
                );
            } else {
                cv.stroke_rect(r, BORDER, 8);
                m.text(cv, &key, (r.x + 5, r.y + 3), DIM, &f.small, Anchor::TopLeft);
            }
        }
        if rows > 1 {
            for row in 0..rows {
                m.text(
                    cv,
                    &tr_args("page {n}", &[("n", &(row + 1).to_string())]),
                    (x0 + 10 * slot_w + 4, y + row * (slot_h + 6) + 4),
                    DIM,
                    &f.small,
                    Anchor::TopLeft,
                );
                m.text(
                    cv,
                    &format!("{} {}", cfg.key_of("page_prev"), cfg.key_of("page_next")),
                    (x0 + 10 * slot_w + 4, y + row * (slot_h + 6) + 24),
                    DIM,
                    &f.small,
                    Anchor::TopLeft,
                );
            }
        }
        y += rows * (slot_h + 6) + 6;

        // the grid of every tool, by category (scrolls)
        let foot = rect.bottom() - 62;
        let body = Rect::new(rect.x + 16, y, rect.w - 32, foot - y - 6);
        let key = "loadout";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut yy = body.y - off;
        let colw = (body.w - 24) / 2;
        let (fsm, fbo) = (&f.small, &f.bold);
        let lh_s = fsm.linesize;
        let card_h = |t: &config::ToolInfo| -> i32 {
            let tw = colw - 80;
            let n_desc = Menu::fit_lines(fsm, &tr(&t.desc), tw, 3).0.len() as i32;
            let n_drv = Menu::fit_lines(fsm, &tr_args("Drives: {neurons}", &[("neurons", &tr(&t.neurons))]), tw, 3).0.len() as i32;
            82.max(6 + fbo.linesize + 4 + (n_desc + n_drv) * lh_s + 10)
        };
        let mouse = m.mouse();
        for cat in &ls.categories {
            let members: Vec<&config::ToolInfo> = ls.catalog.iter().filter(|t| &t.category == cat).collect();
            if members.is_empty() {
                continue;
            }
            m.text(cv, &tr(cat).to_uppercase(), (body.x + 8, yy + 4), ACCENT, fsm, Anchor::TopLeft);
            cv.line((body.x + 8 + 90, yy + 12), (body.right() - 12, yy + 12), 1, BORDER, 255);
            yy += 24;
            let row_h: Vec<i32> = members.chunks(2).map(|c| c.iter().map(|t| card_h(t)).max().unwrap_or(82)).collect();
            for (i, t) in members.iter().enumerate() {
                let cx = body.x + 8 + (i as i32 % 2) * (colw + 8);
                let cy = yy + row_h[..i / 2].iter().map(|h| h + 6).sum::<i32>();
                let card = Rect::new(cx, cy, colw, row_h[i / 2]);
                let why = why_unavailable(cfg, t);
                let equipped = lop.contains(&t.name);
                let hover = card.contains(mouse) && body.contains(mouse);
                cv.fill_rect(card, if hover && why.is_empty() { [36, 42, 56] } else { [24, 28, 38] }, 10);
                cv.stroke_rect2(card, if equipped { pal.select } else { BORDER }, if equipped { 2 } else { 1 }, 10);
                draw_icon(
                    cv,
                    &t.name,
                    ((card.x + 34) as f32, card.centery() as f32),
                    if !why.is_empty() {
                        DIM
                    } else if equipped {
                        pal.select
                    } else {
                        TEXT
                    },
                );
                let tx = card.x + 70;
                let nm = m.text(cv, &tr(&t.label), (tx, card.y + 6), if why.is_empty() { INK } else { DIM }, fbo, Anchor::TopLeft);
                draw_tag(m, cv, (nm.right() + 8, card.y + 8), &t.tag, cfg);
                let slot_text = tr_args(
                    "slot {n}",
                    &[("n", &((lop.tools.iter().position(|x| *x == t.name).unwrap_or(0) % ls.page_size) + 1).to_string())],
                );
                let badge = if !why.is_empty() {
                    why.clone()
                } else if equipped {
                    slot_text
                } else {
                    tr("click to equip")
                };
                m.text(
                    cv,
                    &badge,
                    (card.right() - 10, card.y + 8),
                    if !why.is_empty() || !equipped { DIM } else { pal.ok },
                    fsm,
                    Anchor::TopRight,
                );
                let ty = m.wrapped(
                    cv,
                    &tr(&t.desc),
                    (tx, card.y + 6 + fbo.linesize + 4),
                    card.w - 80,
                    if why.is_empty() { TEXT } else { DIM },
                    fsm,
                    3,
                );
                m.wrapped(cv, &tr_args("Drives: {neurons}", &[("neurons", &tr(&t.neurons))]), (tx, ty), card.w - 80, LABEL, fsm, 3);
                let mut tip = format!(
                    "{}: {} {} [{}]",
                    tr(&t.label),
                    tr(&t.desc),
                    tr_args("Drives: {neurons}", &[("neurons", &tr(&t.neurons))]),
                    tr(&t.tag)
                );
                if !why.is_empty() {
                    tip = format!("{why}. {tip}");
                    m.register_label(card, &format!("lo-card:{}", t.name), &tip);
                } else {
                    m.register_drag_src(
                        card,
                        &format!("lo-card:{}", t.name),
                        &format!("tool|{}|-", t.name),
                        Some(Act::Custom(format!("toggle:{}", t.name))),
                        Some(&tip),
                        None,
                    );
                }
            }
            yy += row_h.iter().map(|h| h + 6).sum::<i32>() + 6;
        }
        let ch = (yy + off - body.bottom() + 8).max(0);
        m.set_content_h(key, ch);
        m.end_clip(cv);
        m.scrollbar(cv, body, ch, off);

        // the card being dragged follows the pointer
        if let Some(d) = m.dragging() {
            let name = d.payload.split('|').nth(1).unwrap_or("hand").to_string();
            let pos = d.pos;
            cv.circle(pos, 30, [40, 46, 60]);
            cv.aa_ring((pos.0 as f32, pos.1 as f32), 30.0, pal.select);
            draw_icon(cv, &name, (pos.0 as f32, pos.1 as f32), pal.select);
        }

        // footer
        let fy = rect.bottom() - 54;
        m.button(
            cv,
            Rect::new(rect.x + 24, fy, 190, 42),
            &tr("Reset to preset"),
            Act::Custom("reset".into()),
            "lo:reset",
            Style::Normal,
            true,
            Some(&tr("Go back to the preset you started from, dropping your changes.")),
            false,
            None,
        );
        let default_name = format!("Custom {}", saved.len() + 1);
        let shown = if self.lo_name.is_empty() { default_name.clone() } else { self.lo_name.clone() };
        m.text_field(
            cv,
            Rect::new(rect.x + 232, fy + 6, 200, 30),
            &shown,
            "lo-name",
            Some(&tr("Name for a new saved loadout.")),
            24,
            false,
        );
        m.button(
            cv,
            Rect::new(rect.x + 442, fy, 170, 42),
            &tr("Save as new"),
            Act::Custom("save".into()),
            "lo:save",
            Style::Normal,
            true,
            Some(&tr_args("Save the hotbar as one of your {n} custom loadouts.", &[("n", &max_saved.to_string())])),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(rect.right() - 164, fy, 140, 42),
            &tr("Back"),
            Act::Back,
            "lo:back",
            Style::Primary,
            true,
            Some(&tr("Close the editor (Esc).")),
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let cfg = &mut *ctx.cfg;
        if let Some(p) = name.strip_prefix("preset:") {
            self.lo_from = Some(p.into());
            m.set_setting(cfg, "controls.loadout_preset", &Value::from(p), true);
        } else if let Some(i) = name.strip_prefix("load:").and_then(|i| i.parse::<usize>().ok()) {
            if let Some(item) = cfg.saved_loadouts.get(i).cloned() {
                commit(m, cfg, &item.tools);
                m.flash(&tr_args("Loaded '{name}'.", &[("name", &item.name)]), GOOD, 4.0);
            }
        } else if let Some(i) = name.strip_prefix("del:").and_then(|i| i.parse::<usize>().ok()) {
            if let Some(item) = cfg.saved_loadouts.get(i).cloned() {
                cfg.delete_loadout(&item.name);
                m.save_config();
            }
        } else if let Some(t) = name.strip_prefix("select:") {
            m.action(&format!("select_tool:{t}"));
        } else if let Some(t) = name.strip_prefix("toggle:") {
            toggle(m, cfg, t);
        } else if let Some(slot) = name.strip_prefix("drop:slot:").and_then(|i| i.parse::<usize>().ok()) {
            drop_on_slot(m, cfg, text.unwrap_or(""), slot);
        } else if name == "dropoff" {
            drop_off(m, cfg, text.unwrap_or(""));
        } else if name == "reset" {
            let from = self.lo_from.clone().filter(|f| f != "custom").unwrap_or_else(|| "auto".into());
            self.lo_from = Some(from.clone());
            m.set_setting(cfg, "controls.loadout_preset", &Value::from(from.as_str()), true);
            let label = config::schema().loadout.preset_labels[&from].clone();
            m.flash(&tr_args("Back to the {name} preset.", &[("name", &tr(&label))]), GOOD, 4.0);
        } else if name == "text:lo-name" {
            self.lo_name = text.unwrap_or("").into();
        } else if name == "save" {
            let n = cfg.saved_loadouts.len() + 1;
            let nm = if self.lo_name.trim().is_empty() { format!("Custom {n}") } else { self.lo_name.clone() };
            let tools = current(cfg);
            if cfg.save_loadout(&nm, &tools) {
                m.save_config();
                m.flash(&tr_args("Saved loadout '{name}'.", &[("name", nm.trim())]), GOOD, 4.0);
            } else {
                m.flash(
                    &tr_args(
                        "You can save up to {n} loadouts. Delete one first.",
                        &[("n", &config::schema().max_saved_loadouts.to_string())],
                    ),
                    AMBER,
                    4.0,
                );
            }
        }
    }
}

/// The one-time popup for a config from before 2.13: keys 1-9, 0, - and = still reach every tool.
#[derive(Default)]
pub struct LoadoutNotice;

impl Page for LoadoutNotice {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let cx = rect.centerx();
        m.text(cv, &tr("Tool loadouts are new"), (cx, rect.y + 40), INK, &f.head, Anchor::MidTop);
        m.wrapped(
            cv,
            &tr_args(
                "The hotbar is now a loadout: a short list of tools on the number keys, with every other tool one gesture away on the tool wheel. So your keys keep working, this install was set to the All preset, which puts every tool on the hotbar (- and = turn the page). Press {key} any time to choose your own tools, or pick a smaller preset.",
                &[("key", &key_label(ctx.cfg, "loadout"))],
            ),
            (rect.x + 60, rect.y + 96),
            rect.w - 120,
            TEXT,
            &f.text,
            8,
        );
        m.button(
            cv,
            Rect::new(cx - 250, rect.bottom() - 90, 240, 50),
            &tr("Open the editor"),
            Act::Custom("open".into()),
            "ln:open",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(cx + 10, rect.bottom() - 90, 240, 50),
            &tr("Got it"),
            Act::Back,
            "ln:ok",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, _text: Option<&str>, _ctx: &mut Ctx) {
        if name == "open" {
            m.back();
            m.show("loadout");
        }
    }
}
