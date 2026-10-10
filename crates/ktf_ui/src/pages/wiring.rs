//! Lab > Connectome robustness (`lab/labwiring.py`): change the wiring itself and re-run the behaviors.
//!
//! Host: query `lab.wiring_state` = `{job: {label, frac, note, elapsed} | null, error, busy, live: {identity, label, min_synapses, flips,
//! inhibition_scale}, stats: {threshold, confidence}, sweep, flips}`; commands `lab.wiring_want {min_syn, conf}` (what the sliders ask
//! about), `lab.wiring_set {min_synapses | flip_conf+flip_share | inhibition_scale | restore}`, `lab.wiring_job {kind: sweep|flips,
//! cutoff, share, trials}`, `lab.wiring_export {which}`, `lab.day4_cancel`.

use serde_json::{Value, json};

use super::day4::grp;
use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn pct(x: f64, dec: usize) -> String {
    format!("{:.dec$}%", x * 100.0)
}

pub struct LabWiring {
    tab: &'static str,
    min_syn: i64,
    flip_conf: f64,
    flip_share: f64,
    flip_trials: i64,
    asked: Option<(i64, i64)>,
}

impl Default for LabWiring {
    fn default() -> Self {
        Self { tab: "threshold", min_syn: 1, flip_conf: 0.5, flip_share: 0.5, flip_trials: 10, asked: None }
    }
}

impl LabWiring {
    /// The game restores the page's own state with this (and the tests).
    pub fn set_view(&mut self, signflip: bool, min_syn: i64, flip_conf: f64) {
        self.tab = if signflip { "signflip" } else { "threshold" };
        self.min_syn = min_syn;
        self.flip_conf = flip_conf;
    }
}

/// `_intro`: two description paragraphs and a tag line; returns the y below.
fn intro(m: &Menu, cv: &mut Canvas, body: Rect, y: i32, lead: &str, more: &str, chip: &str, chip_text: &str) -> i32 {
    let f = m.fonts();
    let w = body.w - 16;
    let y = m.wrapped(cv, lead, (body.x + 8, y), w, TEXT, &f.small, 3);
    let y = m.wrapped(cv, more, (body.x + 8, y + 2), w, LABEL, &f.small, 3) + 6;
    let r = m.chip(cv, (body.x + 8, y), chip);
    let y2 = m.wrapped(cv, chip_text, (r.right() + 10, y + 1), body.right() - 8 - r.right() - 10, LABEL, &f.small, 2);
    y2.max(r.bottom()) + 10
}

/// A result map's entries in the order Python made them (serde_json sorts keys; the result carries `order`).
pub fn ordered<'a>(map: &'a Value, order: &Value) -> Vec<(String, &'a Value)> {
    let Some(obj) = map.as_object() else { return Vec::new() };
    let mut out: Vec<(String, &Value)> = Vec::new();
    if let Some(o) = order.as_array() {
        for k in o.iter().filter_map(|k| k.as_str()) {
            if let Some(v) = obj.get(k) {
                out.push((k.to_string(), v));
            }
        }
    }
    for (k, v) in obj {
        if !out.iter().any(|(x, _)| x == k) {
            out.push((k.clone(), v));
        }
    }
    out
}

fn progress(m: &Menu, cv: &mut Canvas, x: i32, y: i32, w: i32, job: &Value) {
    let f = m.fonts();
    cv.fill_rect(Rect::new(x, y + 14, w, 12), [30, 36, 48], 6);
    cv.fill_rect(Rect::new(x, y + 14, 8.max((f64::from(w) * num(&job["frac"]).clamp(0.0, 1.0)) as i32), 12), AMBER, 6);
    m.text(cv, job["note"].as_str().unwrap_or(""), (x, y + 30), LABEL, &f.small, Anchor::TopLeft);
}

impl LabWiring {
    fn threshold_tab(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, q: &Value) {
        let f = m.fonts();
        let mut y = intro(
            m,
            cv,
            body,
            body.y,
            "Drop every connection the reconstruction found fewer than N synapses for, then watch the fly and re-run the behaviors.",
            "Weak contacts are the ones most likely to be reconstruction noise. The brain pack already drops anything below 3 synapses, so 1-3 change nothing.",
            "CONNECTOME",
            "the synapse counts are the dataset's; which thresholds to try is a game choice",
        );
        m.text(cv, "Minimum synapses", (body.x + 8, y + 15), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(body.x + 200, y, body.w - 420, 30),
            self.min_syn as f64,
            (1.0, 15.0, 1.0),
            "{:.0f}",
            "min_syn",
            true,
            Some("Connections with fewer synapses than this are set to zero weight. The neurons stay; only the weak wiring between them goes."),
        );
        y += 44;
        let st = &q["stats"]["threshold"];
        if !st.is_object() {
            m.text(cv, "needs the brain pack", (body.x + 8, y), BAD, &f.small, Anchor::TopLeft);
            return;
        }
        m.text(
            cv,
            &format!(
                "At >= {} synapses: {} of {} connections dropped ({}), {} of {} synapses",
                st["threshold"],
                grp(num(&st["connections_dropped"]), 0),
                grp(num(&st["connections"]), 0),
                pct(num(&st["connections_dropped_share"]), 1),
                grp(num(&st["synapses_dropped"]), 0),
                grp(num(&st["synapses"]), 0)
            ),
            (body.x + 8, y),
            INK,
            &f.text,
            Anchor::TopLeft,
        );
        m.text(
            cv,
            &format!(
                "{} of {} neurons lose at least one connection; {} lose every input, {} lose every output",
                grp(num(&st["neurons_touched"]), 0),
                grp(num(&st["neurons"]), 0),
                grp(num(&st["neurons_cut_off"]), 0),
                grp(num(&st["neurons_silenced_output"]), 0)
            ),
            (body.x + 8, y + 24),
            TEXT,
            &f.small,
            Anchor::TopLeft,
        );
        y += 56;
        let live = &q["live"];
        let applied = live["min_synapses"].as_i64() == Some(self.min_syn);
        let busy = !q["busy"].as_str().unwrap_or("").is_empty();
        m.button(
            cv,
            Rect::new(body.x + 8, y, 260, 40),
            if applied { "Applied" } else { "Apply to every fly" },
            Act::Custom("apply_thresh".into()),
            "apply_thresh",
            Style::Primary,
            !applied && !busy,
            Some("Rebuilds every live fly's synapse matrix. Takes about a second per fly."),
            false,
            None,
        );
        let job = &q["job"];
        let sweep_running = job.is_object() && job["label"] == "Threshold sweep";
        if sweep_running {
            progress(m, cv, body.x + 290, y, body.w - 320, job);
        } else {
            m.button(
                cv,
                Rect::new(body.x + 290, y, 300, 40),
                "Run the threshold report",
                Act::Custom("run_sweep".into()),
                "run_sweep",
                Style::Normal,
                !job.is_object(),
                Some("Re-runs each validated behavior at thresholds 1, 4, 5, 6, 8 and 10 on the validation seeds, with the validation suite's own pass criteria, and reports the threshold at which each one breaks. Takes several minutes."),
                false,
                None,
            );
        }
        y += 52;
        if q["sweep"].is_object() {
            self.draw_sweep(m, cv, Rect::new(body.x + 8, y, body.w - 16, body.bottom() - y), &q["sweep"]);
        }
    }

    fn draw_sweep(&self, m: &mut Menu, cv: &mut Canvas, area: Rect, res: &Value) {
        let f = m.fonts();
        let seeds = res["seeds"].as_array().cloned().unwrap_or_default();
        m.text(
            cv,
            &format!(
                "Threshold report  ·  seeds {}-{}  ·  {:.0}s  ·  validation's own pass criteria",
                seeds.first().map_or(String::new(), |s| s.to_string()),
                seeds.last().map_or(String::new(), |s| s.to_string()),
                num(&res["seconds"])
            ),
            (area.x, area.y),
            INK,
            &f.small,
            Anchor::TopLeft,
        );
        let mut y = area.y + 22;
        let th: Vec<i64> = res["thresholds"].as_array().into_iter().flatten().filter_map(|t| t.as_i64()).collect();
        let mut xs = vec![area.x];
        xs.extend((0..th.len()).map(|i| area.x + 300 + i as i32 * 56));
        xs.push(area.right() - 120);
        let mut head = vec!["behavior".to_string()];
        head.extend(th.iter().map(|t| format!(">={t}")));
        head.push("breaks at".into());
        for (hx, label) in xs.iter().zip(&head) {
            m.text(cv, label, (*hx, y), LABEL, &f.small, Anchor::TopLeft);
        }
        y += 20;
        for (id, b) in ordered(&res["breaks"], &res["order"]) {
            let name: String = b["name"].as_str().unwrap_or("").chars().take(44).collect();
            m.text(cv, &name, (area.x, y), TEXT, &f.small, Anchor::TopLeft);
            for (i, t) in th.iter().enumerate() {
                let step = res["steps"].as_array().and_then(|s| s.iter().find(|s| s["threshold"].as_i64() == Some(*t)));
                let ok = step.is_some_and(|s| s["behaviors"][&id]["passed"] == true);
                m.text(cv, if ok { "pass" } else { "fail" }, (xs[i + 1], y), if ok { GOOD } else { BAD }, &f.small, Anchor::TopLeft);
            }
            let br = b["breaks_at"].as_i64();
            m.text(
                cv,
                &br.map_or("holds".to_string(), |t| format!(">= {t}")),
                (*xs.last().unwrap(), y),
                if br.is_some() { BAD } else { GOOD },
                &f.small,
                Anchor::TopLeft,
            );
            y += 20;
        }
        y += 6;
        y = m.wrapped(
            cv,
            "A behavior that breaks at a low threshold depends on connections the reconstruction saw only a few synapses of.",
            (area.x, y),
            area.w,
            LABEL,
            &f.small,
            3,
        );
        m.button(
            cv,
            Rect::new(area.x, (y + 4).min(area.bottom() - 34), 200, 30),
            "Export CSV + JSON",
            Act::Custom("export_sweep".into()),
            "sweep_export",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
    }

    fn signflip_tab(&mut self, m: &mut Menu, cv: &mut Canvas, body: Rect, q: &Value) {
        let f = m.fonts();
        let mut y = intro(
            m,
            cv,
            body,
            body.y,
            "MaleCNS v1.0 predicts every neuron's neurotransmitter and says how sure it is. This flips the least certain ones from excitatory to inhibitory, or back.",
            "A neuron with the wrong sign in the dataset has the wrong sign here too. The stress test asks which validated behaviors survive that being true.",
            "CONNECTOME",
            "the predictions and confidences are the dataset's; the cutoff, the share and the trial count are game choices",
        );
        let st = &q["stats"]["confidence"];
        if !st.is_object() {
            m.text(cv, "needs a brain pack with transmitter predictions", (body.x + 8, y), BAD, &f.small, Anchor::TopLeft);
            return;
        }
        m.text(cv, "Confidence below", (body.x + 8, y + 15), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(body.x + 200, y, 300, 30),
            self.flip_conf,
            (0.3, 1.0, 0.05),
            "{:.2f}",
            "flip_conf",
            true,
            Some("Neurons the dataset is less sure than this about are the ones a trial may flip. 1.00 includes everything except the neurons with a measured transmitter."),
        );
        let mut sx = body.x + 530;
        if sx + 120 + 220 > body.right() - 8 {
            y += 40;
            sx = body.x + 8;
        }
        m.text(cv, "Share flipped", (sx, y + 15), TEXT, &f.text, Anchor::MidLeft);
        m.page_slider(
            cv,
            Rect::new(sx + 120.max(f.text.size("Share flipped") + 12), y, 220, 30),
            self.flip_share,
            (0.1, 1.0, 0.1),
            "{:.0%}",
            "flip_share",
            true,
            Some("Each trial flips this share of the candidates, chosen at random. Which of them are actually wrong is unknowable, so a trial samples one possible world and the trials together give the answer."),
        );
        y += 40;
        m.text(
            cv,
            &format!(
                "{} of {} signed neurons are below the cutoff ({} of those have no confidence at all)",
                grp(num(&st["candidates"]), 0),
                grp(num(&st["signed"]), 0),
                grp(num(&st["unknown_confidence"]), 0)
            ),
            (body.x + 8, y),
            INK,
            &f.text,
            Anchor::TopLeft,
        );
        m.text(
            cv,
            &format!(
                "{} neurons have a measured transmitter and are never flipped  ·  {} excitatory, {} inhibitory in the model",
                grp(num(&st["measured"]), 0),
                grp(num(&st["excitatory"]), 0),
                grp(num(&st["inhibitory"]), 0)
            ),
            (body.x + 8, y + 22),
            TEXT,
            &f.small,
            Anchor::TopLeft,
        );
        let live = &q["live"];
        if live["flips"].as_u64().unwrap_or(0) > 0 {
            m.text(
                cv,
                &format!(
                    "live: {} neurons flipped (click a neuron in the big brain view to flip one by hand)",
                    grp(num(&live["flips"]), 0)
                ),
                (body.x + 8, y + 40),
                AMBER,
                &f.small,
                Anchor::TopLeft,
            );
        }
        y += 62;
        let busy = !q["busy"].as_str().unwrap_or("").is_empty();
        m.button(
            cv,
            Rect::new(body.x + 8, y, m.bw("Flip them on every fly", 260), 40),
            "Flip them on every fly",
            Act::Custom("apply_flip".into()),
            "apply_flip",
            Style::Primary,
            !busy,
            Some("Applies one sampled flip set to the live flies so you can watch what it does. The stress test below is the measured version."),
            false,
            None,
        );
        let job = &q["job"];
        if job.is_object() && job["label"] == "Sign flips" {
            progress(m, cv, body.x + 290, y, body.w - 320, job);
        } else {
            m.button(
                cv,
                Rect::new(body.x + 290, y, 300, 40),
                &format!("Run {} randomized trials", self.flip_trials),
                Act::Custom("run_flips".into()),
                "run_flips",
                Style::Normal,
                !job.is_object(),
                Some("Each trial flips a different random sample, then re-runs the validated behaviors over the validation seeds with their own pass criteria. The unperturbed run is the control."),
                false,
                None,
            );
            m.page_slider(
                cv,
                Rect::new(body.x + 610, y + 5, 200, 30),
                self.flip_trials as f64,
                (2.0, 20.0, 1.0),
                "{:.0f} trials",
                "flip_trials",
                true,
                None,
            );
        }
        y += 52;
        if q["flips"].is_object() {
            self.draw_flips(m, cv, Rect::new(body.x + 8, y, body.w - 16, body.bottom() - y), &q["flips"]);
        }
    }

    fn draw_flips(&self, m: &mut Menu, cv: &mut Canvas, area: Rect, res: &Value) {
        let f = m.fonts();
        let c = &res["candidates"];
        let seeds = res["seeds"].as_array().cloned().unwrap_or_default();
        m.text(
            cv,
            &format!(
                "{} trials x {} of {} uncertain neurons  ·  seeds {}-{}  ·  {:.0}s  ·  validation's own criteria",
                res["trials"],
                pct(num(&res["share"]), 0),
                grp(num(&c["candidates"]), 0),
                seeds.first().map_or(String::new(), |s| s.to_string()),
                seeds.last().map_or(String::new(), |s| s.to_string()),
                num(&res["seconds"])
            ),
            (area.x, area.y),
            INK,
            &f.small,
            Anchor::TopLeft,
        );
        let mut y = area.y + 22;
        for (label, x) in [("behavior", 0), ("survived", 320), ("95% CI", 420), ("effect (flipped)", 540), ("unperturbed", 690)] {
            m.text(cv, label, (area.x + x, y), LABEL, &f.small, Anchor::TopLeft);
        }
        y += 20;
        for (_, b) in ordered(&res["behaviors"], &res["order"]) {
            let col = match b["verdict"].as_str() {
                Some("survives") => GOOD,
                Some("breaks") => BAD,
                _ => AMBER,
            };
            let name: String = b["name"].as_str().unwrap_or("").chars().take(46).collect();
            m.text(cv, &name, (area.x, y), TEXT, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{}/{}", b["survived"], b["trials"]), (area.x + 320, y), col, &f.small, Anchor::TopLeft);
            m.text(
                cv,
                &format!("{}-{}", pct(num(&b["survival_ci"][0]), 0), pct(num(&b["survival_ci"][1]), 0)),
                (area.x + 420, y),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            m.text(
                cv,
                &format!("{:.2} [{:.2}, {:.2}]", num(&b["effect_mean"]), num(&b["effect_ci"][0]), num(&b["effect_ci"][1])),
                (area.x + 540, y),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            m.text(cv, &format!("{:.2}", num(&b["control_effect"])), (area.x + 690, y), LABEL, &f.small, Anchor::TopLeft);
            y += 20;
        }
        y += 4;
        y = m.wrapped(
            cv,
            "A behavior that survives every trial does not depend on the signs the dataset is unsure of.",
            (area.x, y),
            area.w,
            LABEL,
            &f.small,
            3,
        );
        m.button(
            cv,
            Rect::new(area.x, (y + 4).min(area.bottom() - 34), 200, 30),
            "Export CSV + JSON",
            Act::Custom("export_flips".into()),
            "flips_export",
            Style::Normal,
            true,
            None,
            false,
            None,
        );
    }
}

impl Page for LabWiring {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.wiring_state");
        // ask the game for the numbers the sliders are on
        let want = (self.min_syn, (self.flip_conf * 100.0).round() as i64);
        let have = (q["stats"]["threshold"]["threshold"].as_i64(), q["stats"]["confidence"]["cutoff"].as_f64());
        let stale = have.0 != Some(self.min_syn.max(1)) || have.1.is_none_or(|c| (c - self.flip_conf).abs() > 1e-9);
        if stale && self.asked != Some(want) && q.is_object() {
            self.asked = Some(want);
            ctx.host.command("lab.wiring_want", json!({"min_syn": self.min_syn, "conf": self.flip_conf}));
        }
        m.text(cv, "CONNECTOME ROBUSTNESS", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let ty = (rect.y + 72).max(
            m.subtitle(
                cv,
                rect,
                "Change the wiring itself and re-run the behaviors. Everything here acts on the dataset's own synapse counts and neurotransmitter predictions.",
                None,
                2,
            ) - 4,
        );
        let mut x = rect.x + 24;
        for (key, label) in [("threshold", "Synapse threshold"), ("signflip", "Sign flips")] {
            let w = m.bw(label, 200);
            m.button(
                cv,
                Rect::new(x, ty, w, 34),
                label,
                Act::Custom(format!("tab:{key}")),
                &format!("wtab:{key}"),
                Style::Normal,
                true,
                None,
                self.tab == key,
                None,
            );
            x += w + 10;
        }
        let body = Rect::new(rect.x + 16, ty + 44, rect.w - 32, rect.bottom() - 70 - (ty + 44));
        if let Some(b) = q["busy"].as_str().filter(|b| !b.is_empty()) {
            m.text(cv, &format!("applying to every fly: {b} …"), (rect.right() - 24, rect.y + 84), AMBER, &f.small, Anchor::TopRight);
        }
        if self.tab == "threshold" {
            self.threshold_tab(m, cv, body, &q);
        } else {
            self.signflip_tab(m, cv, body, &q);
        }
        if q["live"]["identity"] == false {
            m.text(
                cv,
                &format!("live: {}", q["live"]["label"].as_str().unwrap_or("")),
                (rect.x + 24, rect.bottom() - 48),
                AMBER,
                &f.small,
                Anchor::TopLeft,
            );
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 34, m.bw("Restore the connectome", 200), 28),
                "Restore the connectome",
                Act::Custom("restore".into()),
                "wire_reset",
                Style::Danger,
                true,
                None,
                false,
                None,
            );
        }
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, &format!("error: {e}"), (rect.x + 24, rect.bottom() - 74), BAD, &f.small, Anchor::TopLeft);
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "wire:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let val = text.and_then(|t| t.parse::<f64>().ok());
        match name {
            "tab:threshold" => self.tab = "threshold",
            "tab:signflip" => self.tab = "signflip",
            "slide:min_syn" => {
                if let Some(v) = val {
                    self.min_syn = v as i64;
                }
            }
            "slide:flip_conf" => {
                if let Some(v) = val {
                    self.flip_conf = v;
                }
            }
            "slide:flip_share" => {
                if let Some(v) = val {
                    self.flip_share = v;
                }
            }
            "slide:flip_trials" => {
                if let Some(v) = val {
                    self.flip_trials = v as i64;
                }
            }
            "apply_thresh" => drop(ctx.host.command("lab.wiring_set", json!({"min_synapses": self.min_syn}))),
            "apply_flip" => drop(ctx.host.command("lab.wiring_set", json!({"flip_conf": self.flip_conf, "flip_share": self.flip_share}))),
            "restore" => drop(ctx.host.command("lab.wiring_set", json!({"restore": true}))),
            "run_sweep" => drop(ctx.host.command("lab.wiring_job", json!({"kind": "sweep"}))),
            "run_flips" => drop(ctx.host.command(
                "lab.wiring_job",
                json!({"kind": "flips", "cutoff": self.flip_conf, "share": self.flip_share, "trials": self.flip_trials}),
            )),
            "export_sweep" => drop(ctx.host.command("lab.wiring_export", json!({"which": "sweep"}))),
            "export_flips" => drop(ctx.host.command("lab.wiring_export", json!({"which": "flips"}))),
            "cancel" => drop(ctx.host.command("lab.day4_cancel", Value::Null)),
            _ => {}
        }
    }
}
