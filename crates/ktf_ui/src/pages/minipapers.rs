//! Lab > Mini-papers (`ui/minipaper_ui.py`): short guided experiments that reproduce a classic paper in five steps: the paper, YOUR
//! HYPOTHESIS, the run, the plot, and your result next to the paper's. The experiments are validation tests (`fly_sim::minipapers`).
//!
//! Host: query `lab.minipapers_state` = `{results: {paper: result}, job: {label, frac, note, elapsed} | null, error, adult}`; commands
//! `lab.mp_run {paper, full}` and `lab.day4_cancel`. The paper being read, the step, the hypotheses and which of them the shown result
//! was run with live in the page.

use std::collections::BTreeMap;

use fly_sim::minipapers::{self as mp, Paper};
use serde_json::{Value, json};

use super::day4::tags;
use crate::canvas::{Canvas, Rect};
use crate::i18n::{tr, tr_args};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

fn series(palette: &str) -> (Color, Color) {
    match palette {
        "blue-yellow" => ([30, 100, 200], [240, 190, 20]),
        "high-contrast" => ([255, 255, 255], [255, 200, 0]),
        _ => ([57, 135, 229], [217, 89, 38]),
    }
}

fn refs(test: &str) -> Vec<(f64, &'static str)> {
    match test {
        "mb_conditioning" => vec![(0.5, "PI 0.5 (validation's bar)")],
        "rig:buridan" => vec![(45.0, "45 deg: chance")],
        _ => vec![(1.0, "calm (x1)"), (1.5, "x1.5: validation's bar")],
    }
}

fn axis(test: &str) -> &'static str {
    match test {
        "mb_conditioning" => "performance index",
        "rig:buridan" => "stripe deviation (deg)",
        _ => "firing ratio to calm",
    }
}

fn legend(test: &str) -> (&'static str, &'static str) {
    match test {
        "mb_conditioning" => ("unpaired", "paired"),
        "rig:buridan" => ("no stripes", "stripes"),
        _ => ("control", "drive"),
    }
}

#[derive(Default)]
pub struct LabMinipapers {
    paper: Option<String>,
    step: usize,
    answers: BTreeMap<String, BTreeMap<String, String>>,
    committed: BTreeMap<String, BTreeMap<String, String>>,
    full: bool,
}

impl LabMinipapers {
    /// Open a paper at a step with hypotheses chosen (what the reference renders and the gamepad shortcuts set).
    pub fn set_view(&mut self, paper: Option<&str>, step: usize, answers: BTreeMap<String, String>, full: bool) {
        self.paper = paper.map(String::from);
        self.step = step;
        if let Some(p) = paper {
            self.committed.insert(p.to_string(), answers.clone());
            self.answers.insert(p.to_string(), answers);
        }
        self.full = full;
    }

    fn answers_of(&self, pid: &str) -> BTreeMap<String, String> {
        self.answers.get(pid).cloned().unwrap_or_default()
    }

    fn committed_of(&self, pid: &str) -> BTreeMap<String, String> {
        self.committed.get(pid).cloned().unwrap_or_else(|| self.answers_of(pid))
    }
}

fn row_h(m: &Menu) -> i32 {
    34.max(m.fonts().text.linesize + 14)
}

fn para(m: &Menu, cv: &mut Canvas, x: i32, y: i32, w: i32, text: &str, color: Color, small: bool, lines: usize) -> i32 {
    let f = m.fonts();
    m.wrapped(cv, text, (x, y), w, color, if small { &f.small } else { &f.text }, lines) + 6
}

fn fmt_measured(m_: &Value) -> String {
    mp::ordered(m_)
        .iter()
        .filter(|(k, _)| k != "passed")
        .map(|(k, v)| {
            let name = k.replace('_', " ");
            match v {
                Value::Number(n) if n.is_f64() => format!("{name} {}", fly_sim::sensitivity::fmt_gp(n.as_f64().unwrap_or(0.0), 3)),
                Value::Null => format!("{name} None"),
                Value::Bool(b) => format!("{name} {}", if *b { "True" } else { "False" }),
                other => format!("{name} {}", other.to_string().trim_matches('"')),
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn pairs_plot(m: &Menu, cv: &mut Canvas, bx: Rect, q: &Value, colors: (Color, Color)) {
    let f = m.fonts();
    cv.fill_rect(bx, [14, 16, 22], 8);
    let pairs = q["pairs"].as_array().cloned().unwrap_or_default();
    let test = q["test"].as_str().unwrap_or("");
    let rf = refs(test);
    let leg = legend(test);
    if pairs.is_empty() {
        m.wrapped(
            cv,
            &tr("Recorded numbers only (no per-fly data): see the comparison."),
            (bx.x + 12, bx.y + 12),
            bx.w - 24,
            LABEL,
            &f.small,
            3,
        );
        return;
    }
    let g = |v: &Value| v.as_f64().unwrap_or(0.0);
    let vals: Vec<f64> = pairs.iter().flat_map(|p| [g(&p["drive"]), g(&p["control"])]).chain(rf.iter().map(|r| r.0)).collect();
    let lo = vals.iter().copied().fold(0.0f64, f64::min);
    let mut hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max) * 1.12;
    if hi - lo < 1e-9 {
        hi = lo + 1.0;
    }
    let (top, bottom) = (bx.y + f.small.linesize + 14, bx.bottom() - f.small.linesize - 12);
    let (left, right) = (bx.x + 70, bx.right() - 16);
    let yy = |v: f64| f64::from(bottom) - (v - lo) / (hi - lo) * f64::from(bottom - top);
    for (r, label) in &rf {
        let py = yy(*r) as i32;
        cv.line((left, py), (right, py), 1, [70, 76, 92], 255);
        m.text(cv, &tr(label), (right, py - 2), LABEL, &f.small, Anchor::BottomRight);
    }
    let xs = (left + (right - left) / 4, left + 3 * (right - left) / 4);
    for p in &pairs {
        cv.line((xs.0, yy(g(&p["control"])) as i32), (xs.1, yy(g(&p["drive"])) as i32), 1, [110, 116, 130], 255);
    }
    for p in &pairs {
        cv.circle((xs.0, yy(g(&p["control"])) as i32), 5, colors.1);
        cv.circle((xs.1, yy(g(&p["drive"])) as i32), 5, colors.0);
    }
    m.text(cv, leg.0, (xs.0, bx.bottom() - 6), TEXT, &f.small, Anchor::MidBottom);
    m.text(cv, leg.1, (xs.1, bx.bottom() - 6), TEXT, &f.small, Anchor::MidBottom);
    m.text(cv, &tr(axis(test)), (bx.x + 8, bx.y + 6), LABEL, &f.small, Anchor::TopLeft);
    m.text(cv, &fly_sim::sensitivity::fmt_gp(lo, 6), (left - 6, bottom), LABEL, &f.small, Anchor::MidRight);
    m.text(cv, &fly_sim::sensitivity::fmt_gp(hi, 3), (left - 6, top), LABEL, &f.small, Anchor::MidRight);
}

impl Page for LabMinipapers {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[allow(clippy::too_many_lines)]
    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.minipapers_state");
        let key = "lab_minipapers";
        let colors = series(ctx.cfg.get_str("access.palette"));
        m.text(cv, &tr("MINI-PAPERS"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let body = Rect::new(rect.x + 16, rect.y + 56, rect.w - 32, rect.h - 56 - 80);
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let (x, w) = (body.x + 8, body.w - 40);
        let mut y = body.y - off;
        let h = row_h(m);
        let busy = q["job"].is_object();
        let adult = q["adult"].as_bool().unwrap_or(false);
        match self.paper.clone().as_deref().and_then(mp::paper) {
            None => {
                y = para(
                    m,
                    cv,
                    x,
                    y,
                    w,
                    &tr(
                        "Each mini-paper reproduces a classic paper in five steps: read it, state a hypothesis, run it on the model, plot it, and see your result next to what the paper found. Where the model misses the paper, it says so and why.",
                    ),
                    LABEL,
                    true,
                    12,
                );
                for p in mp::papers() {
                    let note = tr(mp::read_label(p));
                    let nw = f.small.size(&note);
                    let r = Rect::new(x, y, w.min(900) - nw - 16, h.max(2 * f.bold.linesize + 12));
                    m.button(
                        cv,
                        r,
                        &tr(&p.title),
                        Act::Custom(format!("open:{}", p.id)),
                        &format!("mp_paper:{}", p.id),
                        Style::Normal,
                        true,
                        Some(&tr(&p.summary)),
                        false,
                        None,
                    );
                    m.text(cv, &note, (r.right() + 12, r.centery()), LABEL, &f.small, Anchor::MidLeft);
                    y += r.h + 8;
                }
                y = para(
                    m,
                    cv,
                    x,
                    y + 4,
                    w,
                    &tr("Headless: --minipaper list, then --minipaper ID [--seeds A-B]. docs/minipapers.md."),
                    LABEL,
                    true,
                    12,
                );
            }
            Some(p) => {
                let total = p.steps.len();
                let step = self.step.min(total.saturating_sub(1));
                let kind = p.steps[step].2.as_str();
                m.text(cv, &tr(&p.title), (x, y), INK, &f.bold, Anchor::TopLeft);
                y += f.bold.linesize + 2;
                m.text(
                    cv,
                    &tr_args(
                        "Step {i} of {n}: {t}",
                        &[("i", &(step + 1).to_string()), ("n", &total.to_string()), ("t", &tr(&p.steps[step].0))],
                    ),
                    (x, y),
                    AMBER,
                    &f.text,
                    Anchor::TopLeft,
                );
                y += f.text.linesize + 8;
                let res = q["results"].get(&p.id).filter(|r| r.is_object());
                y = match kind {
                    "read" => self.step_read(m, cv, x, y, w, p),
                    "hypothesis" => self.step_hypothesis(m, cv, x, y, w, p),
                    "run" => self.step_run(m, cv, x, y, w, p, res, busy, adult),
                    "plot" => self.step_plot(m, cv, x, y, w, res, colors),
                    _ => self.step_compare(m, cv, x, y, w, p, res),
                };
                let last = step + 1 >= total;
                let nb_bottom = y + 8 + h;
                m.button(
                    cv,
                    Rect::new(x, y + 8, 150, h),
                    &tr("Previous"),
                    Act::Custom("prev".into()),
                    &format!("mp_prev:{}", p.id),
                    Style::Normal,
                    step > 0,
                    None,
                    false,
                    None,
                );
                m.button(
                    cv,
                    Rect::new(x + 160, y + 8, 150, h),
                    &if last { tr("All papers") } else { tr("Next") },
                    Act::Custom(if last { "close".into() } else { "next".into() }),
                    &format!("mp_next:{}", p.id),
                    Style::Primary,
                    true,
                    None,
                    false,
                    None,
                );
                y = nb_bottom + 14;
            }
        }
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.end_clip(cv);
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, e, (rect.x + 24, rect.bottom() - 74), BAD, &f.small, Anchor::TopLeft);
        }
        if busy {
            let j = &q["job"];
            let r = Rect::new(rect.x + 24, rect.bottom() - 56, rect.w - 220, 12);
            cv.fill_rect(Rect::new(r.x, r.y, r.w - 110, 12), [30, 36, 48], 6);
            let frac = j["frac"].as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
            cv.fill_rect(Rect::new(r.x, r.y, 8.max(((r.w - 110) as f64 * frac) as i32), 12), AMBER, 6);
            m.text(
                cv,
                &format!(
                    "{}: {}  ({:.0} s)",
                    j["label"].as_str().unwrap_or(""),
                    j["note"].as_str().unwrap_or(""),
                    j["elapsed"].as_f64().unwrap_or(0.0)
                ),
                (r.x, r.y + 18),
                TEXT,
                &f.small,
                Anchor::TopLeft,
            );
            m.button(
                cv,
                Rect::new(r.right() - 100, r.y - 8, 100, 30),
                "Cancel",
                Act::Custom("cancel".into()),
                "bgjob_cancel",
                Style::Danger,
                true,
                Some("Stops after the current step. Nothing already written is lost."),
                false,
                None,
            );
        }
        let in_paper = self.paper.is_some();
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            &if in_paper { tr("Papers") } else { tr("Back") },
            if in_paper { Act::Custom("close".into()) } else { Act::Back },
            &format!("{key}:back"),
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        match name {
            "close" => self.paper = None,
            "prev" => self.step = self.step.saturating_sub(1),
            "next" => {
                let total = self.paper.as_deref().and_then(mp::paper).map_or(0, |p| p.steps.len());
                if self.step + 1 < total {
                    self.step += 1;
                }
            }
            "cancel" => drop(ctx.host.command("lab.day4_cancel", Value::Null)),
            "run" => {
                if let Some(pid) = self.paper.clone() {
                    self.committed.insert(pid.clone(), self.answers_of(&pid));
                    drop(ctx.host.command("lab.mp_run", json!({"paper": pid, "full": self.full})));
                }
            }
            "size:0" => self.full = false,
            "size:1" => self.full = true,
            n if n.starts_with("open:") => {
                self.paper = Some(n[5..].to_string());
                self.step = 0;
            }
            n if n.starts_with("ans:") => {
                if let (Some(pid), Some((test, key))) = (self.paper.clone(), n[4..].split_once('=')) {
                    self.answers.entry(pid).or_default().insert(test.to_string(), key.to_string());
                }
            }
            _ => {}
        }
    }
}

impl LabMinipapers {
    fn step_read(&self, m: &Menu, cv: &mut Canvas, x: i32, mut y: i32, w: i32, p: &Paper) -> i32 {
        let f = m.fonts();
        y = para(m, cv, x, y, w, &tr(&p.summary), TEXT, false, 12);
        m.text(cv, &tr("The paper"), (x, y), AMBER, &f.bold, Anchor::TopLeft);
        y += f.bold.linesize + 2;
        y = para(m, cv, x, y, w, &p.citation, TEXT, true, 12);
        y = para(m, cv, x, y, w, &format!("doi:{}", p.doi), LABEL, true, 12);
        y = para(m, cv, x, y, w, &tr_args("What was read: {r}", &[("r", &p.read)]), LABEL, true, 12);
        m.text(cv, &tr("What the paper reports"), (x, y), AMBER, &f.bold, Anchor::TopLeft);
        y += f.bold.linesize + 2;
        y = para(m, cv, x, y, w, &tr(&p.found), TEXT, false, 12);
        y = para(m, cv, x, y, w, &tr_args("Numbers the paper states: {n}", &[("n", &p.found_numbers)]), LABEL, true, 12);
        tags(m, cv, x, y, &["LITERATURE"]);
        y + f.small.linesize + 12
    }

    fn step_hypothesis(&self, m: &mut Menu, cv: &mut Canvas, x: i32, mut y: i32, w: i32, p: &Paper) -> i32 {
        let h = row_h(m);
        y = para(
            m,
            cv,
            x,
            y,
            w,
            &tr(
                "Commit to a hypothesis before anything runs. It is fixed when you press Run; a change after that applies to your next run.",
            ),
            LABEL,
            true,
            12,
        );
        let ans = self.answers_of(&p.id);
        for q in &p.questions {
            y = para(m, cv, x, y, w, &tr(&q.prompt), TEXT, false, 12);
            for (key, label) in &q.options {
                let r = Rect::new(x + 8, y, (w - 8).min(760), h);
                let active = ans.get(&q.test) == Some(key);
                m.button(
                    cv,
                    r,
                    &tr(label),
                    Act::Custom(format!("ans:{}={}", q.test, key)),
                    &format!("mp_opt:{}:{}:{}", p.id, q.test, key),
                    Style::Normal,
                    true,
                    None,
                    active,
                    None,
                );
                y += h + 6;
            }
            y += 8;
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn step_run(
        &self,
        m: &mut Menu,
        cv: &mut Canvas,
        x: i32,
        mut y: i32,
        w: i32,
        p: &Paper,
        res: Option<&Value>,
        busy: bool,
        adult: bool,
    ) -> i32 {
        let f = m.fonts();
        let (ok, why) = mp::available(p, adult);
        let ans = self.answers_of(&p.id);
        let ready = p.questions.iter().all(|q| ans.contains_key(&q.test));
        let h = row_h(m);
        if !ready {
            y = para(m, cv, x, y, w, &tr("Choose a hypothesis on the previous step first: the comparison needs it."), AMBER, true, 12);
        }
        if !ok {
            y = para(m, cv, x, y, w, &tr_args("Cannot run live here: {why}.", &[("why", &why)]), AMBER, true, 12);
        } else {
            m.segmented(
                cv,
                Rect::new(x, y, w.min(560), h),
                &[tr("Quick: 4 flies, exploration seeds"), tr("Full: 10 flies, validation seeds")],
                usize::from(self.full),
                &[Act::Custom("size:0".into()), Act::Custom("size:1".into())],
                &format!("mp_size:{}", p.id),
                !busy,
                &tr("Quick is a first look (seeds 0-3, too few flies for the validation's p < 0.01 bar). Full is the validation seeds 1000-1009 and takes several times longer."),
            );
            y += h + 8;
        }
        let run_label = if ok { tr("Run the experiment") } else { tr("Show recorded numbers") };
        m.button(
            cv,
            Rect::new(x, y, 220.max(f.bold.size(&run_label) + 32), h),
            &run_label,
            Act::Custom("run".into()),
            &format!("mp_run:{}", p.id),
            Style::Primary,
            ready && !busy,
            None,
            false,
            None,
        );
        y += h + 8;
        if let Some(r) = res {
            let live = r["source"].as_str() == Some("live");
            let n = if live { r["seeds"].as_array().map_or(0, Vec::len) } else { 10 };
            let src = if live { tr("run now") } else { tr("recorded validation numbers") };
            y = para(
                m,
                cv,
                x,
                y,
                w,
                &tr_args(
                    "Done: {n} flies in {sec:.0f} s ({src}). Go on to the plot.",
                    &[("n", &n.to_string()), ("sec:.0f", &format!("{:.0}", r["seconds"].as_f64().unwrap_or(0.0))), ("src", &src)],
                ),
                TEXT,
                true,
                12,
            );
        }
        para(
            m,
            cv,
            x,
            y,
            w,
            &tr("The experiment is a validation test, run from one brain snapshot per fly with a matched control, in worker processes."),
            LABEL,
            true,
            12,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn step_plot(&self, m: &Menu, cv: &mut Canvas, x: i32, mut y: i32, w: i32, res: Option<&Value>, colors: (Color, Color)) -> i32 {
        let f = m.fonts();
        let Some(res) = res else {
            return para(m, cv, x, y, w, &tr("Run the experiment first."), AMBER, true, 12);
        };
        for q in res["questions"].as_array().into_iter().flatten() {
            y = para(m, cv, x, y, w, &tr(q["prompt"].as_str().unwrap_or("")), TEXT, true, 6);
            let bx = Rect::new(x, y, w.min(640), 150.max(f.small.linesize * 9));
            pairs_plot(m, cv, bx, q, colors);
            y = bx.bottom() + 8;
        }
        tags(m, cv, x, y, &["MODEL PREDICTION"]);
        y + f.small.linesize + 12
    }

    #[allow(clippy::too_many_lines)]
    fn step_compare(&self, m: &Menu, cv: &mut Canvas, x: i32, mut y: i32, w: i32, p: &Paper, res: Option<&Value>) -> i32 {
        let f = m.fonts();
        let Some(res) = res else {
            return para(m, cv, x, y, w, &tr("Run the experiment first."), AMBER, true, 12);
        };
        let rows = mp::compare(res, &self.committed_of(&p.id));
        let wide = w > 880;
        let cw = if wide { (w - 20) / 2 } else { w };
        let y0 = y;
        m.text(cv, &tr("Your result (the model)"), (x, y), AMBER, &f.bold, Anchor::TopLeft);
        y += f.bold.linesize + 2;
        let live = res["source"].as_str() != Some("recorded");
        let src = if !live {
            tr("recorded validation numbers (not run now)")
        } else {
            let kind = if res["full"].as_bool().unwrap_or(false) { String::new() } else { tr(" (exploration seeds, a first look)") };
            tr_args("run now on {n} flies{kind}", &[("n", &res["seeds"].as_array().map_or(0, Vec::len).to_string()), ("kind", &kind)])
        };
        y = para(m, cv, x, y, cw, &src, LABEL, true, 12);
        for r in &rows {
            let ok = r["model_reproduces_paper"].as_bool().unwrap_or(false);
            y = para(
                m,
                cv,
                x,
                y,
                cw,
                &tr_args(
                    "The model {v} the direction the paper's finding points to.",
                    &[("v", &if ok { tr("REPRODUCES") } else { tr("DOES NOT REPRODUCE") })],
                ),
                if ok { GOOD } else { BAD },
                false,
                3,
            );
            y = para(m, cv, x, y, cw, &fmt_measured(&r["measured"]), TEXT, true, 4);
            y = para(m, cv, x, y, cw, &tr(r["basis"].as_str().unwrap_or("")), LABEL, true, 5);
            y = para(m, cv, x, y, cw, &tr(r["note"].as_str().unwrap_or("")), TEXT, true, 12);
            if let Some(mine) = r["your_hypothesis"].as_str() {
                let mt = tr_args(
                    "You expected: {a}. The paper's finding points to: {b}.",
                    &[("a", &tr(mine)), ("b", &tr(r["paper_direction"].as_str().unwrap_or("")))],
                );
                let tail = if r["you_matched_the_model"].as_bool().unwrap_or(false) {
                    tr("Your hypothesis matched the model.")
                } else {
                    tr("The model went the other way from your hypothesis.")
                };
                y = para(m, cv, x, y, cw, &format!("{mt} {tail}"), TEXT, true, 6);
            }
            y += 6;
        }
        let ya = y;
        let (x2, mut y2) = if wide { (x + cw + 20, y0) } else { (x, y + 8) };
        m.text(cv, &tr("The paper found"), (x2, y2), AMBER, &f.bold, Anchor::TopLeft);
        y2 += f.bold.linesize + 2;
        y2 = para(m, cv, x2, y2, cw, &tr(&p.found), TEXT, true, 14);
        y2 = para(m, cv, x2, y2, cw, &tr_args("Numbers the paper states: {n}", &[("n", &p.found_numbers)]), LABEL, true, 5);
        y2 = para(m, cv, x2, y2, cw, &format!("{}  doi:{}", p.citation, p.doi), LABEL, true, 8);
        y2 = para(m, cv, x2, y2, cw, &tr_args("What was read: {r}", &[("r", &p.read)]), LABEL, true, 4);
        tags(m, cv, x2, y2, &["LITERATURE"]);
        y2 += f.small.linesize + 10;
        y = y2.max(ya);
        m.text(cv, &tr("What this model cannot check"), (x, y), AMBER, &f.bold, Anchor::TopLeft);
        y += f.bold.linesize + 2;
        para(m, cv, x, y, w, &tr(&p.cannot_check), TEXT, true, 14) + 6
    }
}
