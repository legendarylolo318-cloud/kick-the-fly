//! Lab > Record and export (`lab/lab.py:page_export`): record spike times from the focused fly live, to CSV and npz with a metadata JSON.
//!
//! Host: query `lab.export_state` = `{options: [{label, spec}], recording: {seconds, elapsed, neurons, saving} | null, message, last_export,
//! exports_dir, nwb, bundle: {msg, path}}`; commands `lab.rec_start {groups, seconds}`, `lab.rec_stop`, `lab.bundle`. The picked groups
//! and the duration live in the page.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

pub struct LabExport {
    pick: BTreeSet<String>,
    seconds: i64,
    nwb: bool,
}

impl Default for LabExport {
    fn default() -> Self {
        Self { pick: ["dnp01", "loom", "reaction_dns"].iter().map(|s| s.to_string()).collect(), seconds: 10, nwb: true }
    }
}

/// `short`: a long path shortened in the middle.
fn short(path: &str, n: usize) -> String {
    let c: Vec<char> = path.chars().collect();
    if c.len() <= n {
        return path.to_string();
    }
    let head: String = c[..n / 2 - 2].iter().collect();
    let tail: String = c[c.len() - (n / 2 - 1)..].iter().collect();
    format!("{head}…{tail}")
}

fn fit_path(m: &Menu, prefix: &str, path: &str, width: i32) -> String {
    let f = m.fonts();
    let mut n = path.chars().count();
    while n > 12 && f.small.size(&format!("{prefix}{}", short(path, n))) > width {
        n -= 4;
    }
    format!("{prefix}{}", short(path, n))
}

impl Page for LabExport {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[allow(clippy::too_many_lines)]
    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.export_state");
        m.text(cv, "RECORD AND EXPORT", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let mut y = (rect.y + 84).max(m.subtitle(
            cv,
            rect,
            "Records spike times from the fly your brain panel shows, live, while you play. Files: spikes and rates as CSV and npz, plus a metadata JSON.",
            None,
            3,
        ));
        let options = q["options"].as_array().cloned().unwrap_or_default();
        let col_w = (rect.w - 48) / 2;
        for (i, o) in options.iter().enumerate() {
            let x = rect.x + 24 + (i as i32 % 2) * col_w;
            let yy = y + (i as i32 / 2) * 38;
            let spec = o["spec"].as_str().unwrap_or("");
            let on = self.pick.contains(spec);
            m.toggle(
                cv,
                Rect::new(x, yy, 90, 30),
                on,
                Act::Custom(format!("pick:{}:{spec}", u8::from(!on))),
                &format!("rec:{spec}"),
                true,
                "",
            );
            m.text(cv, o["label"].as_str().unwrap_or(""), (x + 100, yy + 15), TEXT, &f.text, Anchor::MidLeft);
        }
        y += ((options.len() as i32 + 1) / 2) * 38 + 12;
        m.text(cv, "Duration", (rect.x + 24, y + 15), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(rect.x + 140, y, 360, 30),
            self.seconds as f64,
            (1.0, 60.0, 1.0),
            "{:.0f} s",
            "rec_seconds",
            true,
            None,
        );
        y += 46;
        let why = q["nwb"].as_str().filter(|s| !s.is_empty());
        m.toggle(
            cv,
            Rect::new(rect.x + 24, y, 90, 30),
            self.nwb && why.is_none(),
            Act::Custom(format!("nwb:{}", u8::from(!self.nwb))),
            "rec_nwb",
            why.is_none(),
            "",
        );
        m.text(
            cv,
            "Also write NWB (Neurodata Without Borders)",
            (rect.x + 130, y + 15),
            if why.is_some() { LABEL } else { TEXT },
            &f.text,
            Anchor::MidLeft,
        );
        m.register_label(
            Rect::new(rect.x + 24, y, rect.w - 48, 30),
            "nwbtip",
            why.unwrap_or("One .nwb file with spike times, per-neuron and per-region rates, stimuli, tool events, the fly's movement, surgery, arena and the learned KC->MBON weights before and after, plus the full metadata and the MaleCNS v1.0 citation. Opens in pynwb, the NWB inspector and NWB Explorer. CSV and npz are always written as the quick option."),
        );
        if let Some(w) = why {
            y = m.wrapped(cv, w, (rect.x + 24, y + 34), rect.w - 48, AMBER, &f.small, 2) - 30;
        }
        y += 44;
        let rec = &q["recording"];
        if rec.is_object() {
            let label = if rec["saving"].as_bool().unwrap_or(false) { "saving…" } else { "Stop and save" };
            m.button(
                cv,
                Rect::new(rect.x + 24, y, 200, 44),
                label,
                Act::Custom("stop".into()),
                "rec_stop",
                Style::Danger,
                true,
                None,
                false,
                None,
            );
        } else {
            let label = "Start recording and resume";
            m.button(
                cv,
                Rect::new(rect.x + 24, y, 260.max(f.bold.size(label) + 32), 44),
                label,
                Act::Custom("start".into()),
                "rec_start",
                Style::Primary,
                !self.pick.is_empty(),
                Some("Closes the menu; the recording stops by itself after the duration."),
                false,
                None,
            );
        }
        let (mut ly, lh) = (y + 56, f.small.linesize);
        let dir = q["exports_dir"].as_str().unwrap_or("");
        m.text(cv, &fit_path(m, "Saved to ", dir, rect.w - 220), (rect.x + 24, ly), LABEL, &f.small, Anchor::TopLeft);
        ly += lh;
        if let Some(last) = q["last_export"].as_str().filter(|s| !s.is_empty()) {
            m.text(cv, &fit_path(m, "Last: ", last, rect.w - 220), (rect.x + 24, ly), GOOD, &f.small, Anchor::TopLeft);
            ly += lh;
        }
        y = y.max(ly + 6 - 98);
        m.button(
            cv,
            Rect::new(rect.x + 24, y + 98, 200, 40),
            "Bundle",
            Act::Custom("bundle".into()),
            "export_bundle",
            Style::Normal,
            true,
            Some("One zip with the protocol YAML, results, raw exports, metadata (version, backend, precision, seeds, brain pack checksum, parameters, surgery, individuality, arena) and an RO-Crate description. Bundles the last protocol run, or else the last recording. A protocol run's bundle can be rerun with --headless --rerun-bundle ZIP --out DIR."),
            false,
            None,
        );
        let b = &q["bundle"];
        if let Some(msg) = b["msg"].as_str().filter(|s| !s.is_empty()) {
            let has = b["path"].is_string();
            m.text(cv, msg, (rect.x + 236, y + 103), if has { GOOD } else { AMBER }, &f.small, Anchor::TopLeft);
            if let Some(p) = b["path"].as_str() {
                m.text(cv, &short(p, 120), (rect.x + 236, y + 121), LABEL, &f.small, Anchor::TopLeft);
            }
        } else if let Some(msg) = q["message"].as_str().filter(|s| !s.is_empty()) {
            m.text(cv, msg, (rect.x + 236, y + 103), AMBER, &f.small, Anchor::TopLeft);
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "export:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        match name {
            "start" => {
                let q = ctx.host.query("lab.export_state");
                let groups: Vec<Value> = q["options"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|o| o["spec"].as_str().is_some_and(|s| self.pick.contains(s)))
                    .map(|o| json!({"label": o["label"], "spec": o["spec"]}))
                    .collect();
                drop(ctx.host.command("lab.rec_start", json!({"groups": groups, "seconds": self.seconds})));
            }
            "stop" => drop(ctx.host.command("lab.rec_stop", Value::Null)),
            "bundle" => drop(ctx.host.command("lab.bundle", Value::Null)),
            "slide:rec_seconds" => {
                if let Some(v) = text.and_then(|t| t.parse::<f64>().ok()) {
                    self.seconds = v as i64;
                }
            }
            n if n.starts_with("nwb:") => self.nwb = &n[4..] == "1",
            n if n.starts_with("pick:") => {
                if let Some((on, spec)) = n[5..].split_once(':') {
                    if on == "1" {
                        self.pick.insert(spec.to_string());
                    } else {
                        self.pick.remove(spec);
                    }
                }
            }
            _ => {}
        }
    }
}
