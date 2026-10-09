//! Esc > Share (`ui/share_ui.py`): make a share code for what you have set up, or paste one and see what it would change before it does.
//! GAME RULE, all of it: a container for settings the game already has. Import previews first and changes nothing until you press Apply.
//!
//! What the page needs from the game it asks through [`PageHost`]: queries `share_context`, `surgery_payload`, `loadout_tools`,
//! `lab_modified`, `protocol_files`, `challenge_keys`, `contraption_payload`, `clipboard_get`, `now_stamp`; commands `protocol_payload`,
//! `clipboard_put`, `share_apply` (returns `{"lines": [...]}` or `{"error": "..."}`).

use std::collections::BTreeMap;

use ktf_data::sharecode as sc;
use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::i18n::tr;
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const KINDS: [&str; 6] = ["surgery", "loadout", "lab", "protocol", "challenge", "contraption"];

fn kind_name(k: &str) -> &'static str {
    sc::kind_label(k)
}

pub struct SharePage {
    tab: String,
    kind: String,
    loadout_name: String,
    protocol_file: Option<String>,
    challenge: String,
    text: String,
    parsed: Option<Parsed>,
    last: String,
    code_cache: Option<(String, Option<String>, Option<String>, bool)>,
    /// The protocol picked in the list, read once when picked: its YAML as JSON, or why not.
    proto_result: Option<Result<Value, String>>,
}

struct Parsed {
    text: String,
    code: Option<sc::Code>,
    reason: Option<String>,
    preview: Option<sc::Preview>,
}

impl Default for SharePage {
    fn default() -> Self {
        SharePage {
            tab: "export".into(),
            kind: "surgery".into(),
            loadout_name: "Shared loadout".into(),
            protocol_file: None,
            challenge: "tmaze".into(),
            text: String::new(),
            parsed: None,
            last: String::new(),
            code_cache: None,
            proto_result: None,
        }
    }
}

fn strings(v: &Value) -> Option<Vec<String>> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
}

/// What this running game knows, for validating and previewing a code (`make_context`).
pub fn context_from(q: &Value) -> sc::Context {
    let bounds: Option<BTreeMap<String, (f64, f64)>> = q.get("lab_params").and_then(|p| p.as_object()).map(|m| {
        m.iter()
            .filter_map(|(k, v)| {
                let a = v.as_array()?;
                Some((k.clone(), (a.first()?.as_f64()?, a.get(1)?.as_f64()?)))
            })
            .collect()
    });
    sc::Context {
        tools: strings(&q["tools"]),
        lab_params: bounds,
        current_params: q["current_params"]
            .as_object()
            .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_f64()?))).collect())
            .unwrap_or_default(),
        surgery_labels: strings(&q["surgery_labels"]),
        type_names: strings(&q["type_names"]),
        challenges: strings(&q["challenges"]),
        arenas: strings(&q["arenas"]),
        current_surgery_switches: q["current_surgery_switches"].as_u64().unwrap_or(0) as usize,
        current_loadout_len: q["current_loadout_len"].as_u64().unwrap_or(0) as usize,
        saved_loadouts: strings(&q["saved_loadouts"]).unwrap_or_default(),
        contraption_slots_used: q["contraption_slots_used"].as_u64().unwrap_or(0) as usize,
        max_saved_loadouts: q["max_saved_loadouts"].as_u64().unwrap_or(5) as usize,
        protocol_names: strings(&q["protocol_names"]).unwrap_or_default(),
        larva: q["larva"].as_bool().unwrap_or(false),
        lab: q["lab"].as_bool().unwrap_or(false),
        brain: q["brain"].as_str().unwrap_or("adult").into(),
    }
}

impl SharePage {
    /// (payload, why not) for the selected kind from the game's current state (`_payload`).
    fn payload(&self, ctx: &Ctx) -> (Option<Value>, Option<String>) {
        let h = &*ctx.host;
        match self.kind.as_str() {
            "surgery" => {
                let p = h.query("surgery_payload");
                let any = |k: &str| p.get(k).and_then(|g| g.as_object()).is_some_and(|g| !g.is_empty());
                if any("groups") || any("types") { (Some(p), None) } else { (None, Some(tr("No surgery is switched on."))) }
            }
            "loadout" => {
                let tools = h.query("loadout_tools");
                (Some(json!({"name": self.loadout_name.chars().take(24).collect::<String>(), "tools": tools})), None)
            }
            "lab" => {
                let m = h.query("lab_modified");
                match m.as_object() {
                    Some(o) if !o.is_empty() => (Some(json!({"params": m})), None),
                    _ => (None, Some(tr("No Lab parameter is changed."))),
                }
            }
            "protocol" => match (&self.protocol_file, &self.proto_result) {
                (None, _) => (None, Some(tr("Pick a protocol."))),
                (Some(_), Some(Ok(p))) => (Some(json!({"protocol": p})), None),
                (Some(f), Some(Err(e))) => (None, Some(format!("{f}: {e}"))),
                (Some(f), None) => (None, Some(format!("{f}: not available"))),
            },
            "contraption" => {
                let p = h.query("contraption_payload");
                if p.is_null() {
                    (None, Some(tr("No contraption yet: build and save one in Esc > Challenges > Contraption.")))
                } else {
                    (Some(p), None)
                }
            }
            "challenge" => {
                let mut p =
                    json!({"challenge": self.challenge, "seed": ctx.cfg.get_i64("brain.seed"), "arena": ctx.cfg.get_str("brain.arena")});
                let sg = h.query("surgery_payload");
                let any = |k: &str| sg.get(k).and_then(|g| g.as_object()).is_some_and(|g| !g.is_empty());
                if any("groups") || any("types") {
                    p["surgery"] = sg;
                }
                if let Some(o) = h.query("lab_modified").as_object().filter(|o| !o.is_empty()) {
                    p["params"] = Value::Object(o.clone());
                }
                (Some(p), None)
            }
            _ => (None, Some("?".into())),
        }
    }

    fn export_tab(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let (x, mut y) = (body.x, body.y);
        m.text(cv, &tr("What to share"), (x, y), LABEL, &f.small, Anchor::TopLeft);
        let items: Vec<_> = KINDS
            .iter()
            .map(|k| (tr(kind_name(k)), Act::Custom(format!("kind:{k}")), format!("share-kind:{k}"), self.kind == *k, None))
            .collect();
        let yb = m.flow_buttons(cv, x, y + 20, body.right(), &items, Some(34), Some(f.bold.clone()), 8, None);
        y = (y + 70).max(yb + 16);
        match self.kind.as_str() {
            "loadout" => {
                m.text(cv, &tr("Name"), (x, y + 5), TEXT, &f.small, Anchor::TopLeft);
                m.text_field(cv, Rect::new(x + 60, y, 260, 28), &self.loadout_name.clone(), "share-lname", None, 24, false);
                y += 38;
            }
            "protocol" => {
                let files = ctx.host.query("protocol_files");
                let files: Vec<String> = strings(&files).unwrap_or_default();
                m.text(cv, &format!("{}:", tr("Protocol file")), (x, y + 5), TEXT, &f.small, Anchor::TopLeft);
                for (i, name) in files.iter().take(6).enumerate() {
                    let (col, row) = (i as i32 % 3, i as i32 / 3);
                    let stem: String = name.trim_end_matches(".yaml").chars().take(30).collect();
                    m.button(
                        cv,
                        Rect::new(x + 110 + col * 250, y + row * 34, 240, 30),
                        &stem,
                        Act::Custom(format!("proto:{name}")),
                        &format!("share-proto:{name}"),
                        Style::Normal,
                        true,
                        None,
                        self.protocol_file.as_deref() == Some(name.as_str()),
                        None,
                    );
                }
                y += 72;
            }
            "challenge" => {
                let keys = strings(&ctx.host.query("challenge_keys")).unwrap_or_default();
                m.text(cv, &format!("{}:", tr("Challenge")), (x, y + 5), TEXT, &f.small, Anchor::TopLeft);
                for (i, key) in keys.iter().enumerate() {
                    m.button(
                        cv,
                        Rect::new(x + 100 + i as i32 * 140, y, 132, 30),
                        &key.replace('_', " "),
                        Act::Custom(format!("chal:{key}")),
                        &format!("share-chal:{key}"),
                        Style::Normal,
                        true,
                        None,
                        self.challenge == *key,
                        None,
                    );
                }
                y += 38;
                m.text(
                    cv,
                    &tr("Includes the seed, arena, any surgery and any changed Lab parameters."),
                    (x, y),
                    DIM,
                    &f.small,
                    Anchor::TopLeft,
                );
                y += 22;
            }
            _ => {}
        }
        let (payload, why) = self.payload(ctx);
        let bx = Rect::new(x, y + 6, body.w, 150);
        cv.fill_rect(bx, [12, 14, 20], 8);
        cv.stroke_rect(bx, BORDER, 8);
        let (mut code, mut err, mut too_big) = (None, why, false);
        if let Some(p) = &payload {
            let ck = format!("{}|{}", self.kind, serde_json::to_string(p).unwrap_or_default());
            if self.code_cache.as_ref().is_none_or(|c| c.0 != ck) {
                self.code_cache = Some(match sc::encode(&self.kind, p, Some(sc::MAX_CODE_CHARS)) {
                    Ok(t) => (ck, Some(t), None, false),
                    Err(e @ sc::ShareError::TooLarge { .. }) => (ck, None, Some(e.to_string()), true),
                    Err(e) => (ck, None, Some(e.to_string()), false),
                });
            }
            let c = self.code_cache.as_ref().unwrap();
            (code, err, too_big) = (c.1.clone(), c.2.clone(), c.3);
        }
        if let Some(code) = &code {
            let mono = &f.mono13;
            let mut lines: Vec<String> = vec![];
            let mut line = String::new();
            for part in code.split('-') {
                if mono.size(&format!("{line}{part}-")) > bx.w - 24 && !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                }
                line.push_str(part);
                line.push('-');
            }
            lines.push(line.trim_end_matches('-').to_string());
            for (i, ln) in lines.iter().take(8).enumerate() {
                m.text(cv, ln, (bx.x + 12, bx.y + 10 + i as i32 * 17), INK, mono, Anchor::TopLeft);
            }
            m.text(
                cv,
                &format!("{} {}", group(code.len()), tr("characters")),
                (bx.right() - 10, bx.bottom() - 18),
                DIM,
                &f.small,
                Anchor::TopRight,
            );
        } else {
            m.wrapped(cv, err.as_deref().unwrap_or(""), (bx.x + 12, bx.y + 12), bx.w - 24, if too_big { AMBER } else { LABEL }, &f.text, 4);
        }
        let by = bx.bottom() + 14;
        m.button(
            cv,
            Rect::new(x, by, 180, 40),
            &tr("Copy code"),
            Act::Custom("copy".into()),
            "share-copy",
            Style::Primary,
            code.is_some(),
            Some(&tr("Puts the code on your clipboard.")),
            false,
            None,
        );
        let can_file = payload.is_some() && err.is_none() || too_big;
        m.button(
            cv,
            Rect::new(x + 190, by, 200, 40),
            &tr("Save as a file"),
            Act::Custom("save_file".into()),
            "share-file",
            Style::Normal,
            can_file,
            Some(&tr("Writes a .ktfshare file in your data folder's share folder (for a code too big to paste).")),
            false,
            None,
        );
        if !self.last.is_empty() {
            m.text(cv, &self.last, (x, by + 52), GOOD, &f.small, Anchor::TopLeft);
        }
    }

    fn import_tab(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let (x, mut y) = (body.x, body.y);
        m.text(cv, &tr("Paste a share code (or the path of a .ktfshare file)"), (x, y), LABEL, &f.small, Anchor::TopLeft);
        m.text_field(
            cv,
            Rect::new(x, y + 22, body.w - 130, 32),
            &self.text.clone(),
            "share-in",
            Some(&tr("Click, then Ctrl+V. Spaces and line breaks are ignored.")),
            6000,
            true,
        );
        m.button(
            cv,
            Rect::new(body.right() - 120, y + 22, 120, 32),
            &tr("Paste"),
            Act::Custom("paste".into()),
            "share-paste",
            Style::Normal,
            true,
            Some(&tr("Reads your clipboard into the box.")),
            false,
            None,
        );
        y += 70;
        let text = self.text.trim().to_string();
        if self.parsed.as_ref().is_none_or(|p| p.text != text) {
            let mut p = Parsed { text: text.clone(), code: None, reason: None, preview: None };
            if !text.is_empty() {
                match sc::read_any(&text) {
                    Ok(code) => {
                        let c = context_from(&ctx.host.query("share_context"));
                        p.reason = sc::validate(&code, &c);
                        if p.reason.is_none() {
                            p.preview = Some(sc::preview(&code, &c));
                        }
                        p.code = Some(code);
                    }
                    Err(e) => p.reason = Some(e.to_string()),
                }
            }
            self.parsed = Some(p);
        }
        let p = self.parsed.as_ref().unwrap();
        if text.is_empty() {
            m.text(cv, &tr("Nothing pasted yet."), (x, y), DIM, &f.text, Anchor::TopLeft);
            return;
        }
        if let Some(reason) = &p.reason {
            m.chip(cv, (x, y), "REFUSED");
            m.wrapped(cv, reason, (x, y + 28), body.w, BAD, &f.text, 4);
            return;
        }
        let pv = p.preview.as_ref().unwrap();
        m.text(cv, &pv.title, (x, y), INK, &f.bold, Anchor::TopLeft);
        m.text(cv, &tr("Applying this will:"), (x, y + 26), LABEL, &f.small, Anchor::TopLeft);
        let mut yy = y + 46;
        for ln in pv.lines.iter().take(10) {
            yy = m.wrapped(cv, &format!("- {ln}"), (x + 8, yy), body.w - 16, TEXT, &f.small, 2) + 1;
        }
        for w in &pv.warnings {
            yy = m.wrapped(cv, w, (x + 8, yy + 2), body.w - 16, AMBER, &f.small, 3);
        }
        m.button(
            cv,
            Rect::new(x, body.bottom() - 46, 200, 42),
            &tr("Apply"),
            Act::Custom("apply".into()),
            "share-apply",
            Style::Primary,
            true,
            Some(&tr("Does what the list above says, now.")),
            false,
            None,
        );
        if !self.last.is_empty() {
            m.text(cv, &self.last, (x + 214, body.bottom() - 34), GOOD, &f.small, Anchor::TopLeft);
        }
    }
}

/// `{:,}`
fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl Page for SharePage {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let head = m.text(cv, &tr("SHARE"), (rect.x + 24, rect.y + 14), INK, &f.head, Anchor::TopLeft);
        let chip_r = m.chip(cv, ((rect.x + 120).max(head.right() + 12), rect.y + 18), "GAME RULE");
        let sub_bottom = m.wrapped(
            cv,
            &tr("A code carries settings the game already has. Import shows what it would change first."),
            (chip_r.right() + 16, rect.y + 14),
            rect.right() - 24 - chip_r.right() - 16,
            DIM,
            &f.small,
            2,
        );
        let ty = (rect.y + 52).max(sub_bottom + 6);
        for (i, (key, label)) in [("export", tr("Make a code")), ("import", tr("Import a code"))].iter().enumerate() {
            m.button(
                cv,
                Rect::new(rect.x + 24 + i as i32 * 190, ty, 180, 36),
                label,
                Act::Custom(format!("tab:{key}")),
                &format!("share-tab:{key}"),
                Style::Normal,
                true,
                None,
                self.tab == *key,
                None,
            );
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            &tr("Back"),
            Act::Back,
            "share:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
        let body = Rect::new(rect.x + 24, ty + 48, rect.w - 48, rect.bottom() - 70 - (ty + 48));
        if self.tab == "export" { self.export_tab(m, cv, body, ctx) } else { self.import_tab(m, cv, body, ctx) }
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        if let Some(t) = name.strip_prefix("tab:") {
            self.tab = t.into();
        } else if let Some(k) = name.strip_prefix("kind:") {
            self.kind = k.into();
        } else if let Some(k) = name.strip_prefix("chal:") {
            self.challenge = k.into();
        } else if let Some(f) = name.strip_prefix("proto:") {
            self.protocol_file = Some(f.into());
            let r = ctx.host.command("protocol_payload", json!({"name": f}));
            self.proto_result = match r.get("error").and_then(|e| e.as_str()) {
                Some(e) => Some(Err(e.to_string())),
                None => r.get("payload").cloned().map(Ok),
            };
        } else if name == "text:share-lname" {
            let v: String = text.unwrap_or("").trim().chars().take(24).collect();
            self.loadout_name = if v.is_empty() { "Shared loadout".into() } else { v };
        } else if name == "text:share-in" {
            self.text = text.unwrap_or("").into();
        } else if name == "copy" {
            if let Some((_, Some(code), _, _)) = &self.code_cache {
                let ok = ctx.host.command("clipboard_put", json!(code)).as_bool().unwrap_or(false);
                self.last = if ok {
                    tr("Copied. Paste it anywhere; Import reads it back.")
                } else {
                    tr("Couldn't reach the clipboard here: select the code above, or use Save as a file.")
                };
            }
        } else if name == "paste" {
            let t = ctx.host.query("clipboard_get");
            self.text = t.as_str().unwrap_or("").split_whitespace().collect::<String>().chars().take(6000).collect();
        } else if name == "save_file" {
            let (payload, _) = self.payload(ctx);
            let Some(p) = payload else { return };
            self.last = match sc::encode(&self.kind, &p, None) {
                Ok(text) => {
                    let stamp = ctx.host.query("now_stamp").as_str().unwrap_or("").to_string();
                    let dir = ktf_data::paths::get().data_dir.join("share");
                    match sc::export_file(&text, &dir, &format!("{}-{stamp}", self.kind)) {
                        Ok(path) => format!("{}: {}", tr("Saved"), path.display()),
                        Err(e) => e.to_string(),
                    }
                }
                Err(e) => e.to_string(),
            };
        } else if name == "apply" {
            let Some(code) = self.parsed.as_ref().and_then(|p| p.code.clone()) else { return };
            let res = ctx.host.command("share_apply", json!({"kind": code.kind, "payload": code.payload}));
            if let Some(e) = res.get("error").and_then(|e| e.as_str()) {
                self.last = format!("{}: {e}", tr("Could not apply it"));
                return;
            }
            let lines = strings(&res["lines"]).unwrap_or_default();
            self.last = format!("{}: {}", tr("Done"), lines.join("; "));
            self.text.clear();
            self.parsed = None;
        }
    }
}
