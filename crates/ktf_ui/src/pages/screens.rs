//! Lab > Activation screen and Knockout screen (`lab/labscreen.py`): run one (resumable, on held-out seeds), browse a finished one in a
//! searchable, sortable table with a detail panel per row. MODEL PREDICTION.
//!
//! Host: query `lab.screen_activation_state` / `lab.screen_knockout_state` = `{res | null, job: {done, total, label, elapsed} | null, error,
//! type_ops: {type: mode}}`; commands `lab.screen_run {kind, seeds, size}`, `lab.screen_stop`, `lab.screen_export {kind}`,
//! `lab.screen_toggle {types, mode}`, `lab.screen_inspect {type}`. The table's search, sort and selection live in the page.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const ACT_COLS: [(&str, &str, i32); 7] = [
    ("rank", "#", 8),
    ("type", "cell type", 44),
    ("neurons", "neurons", 170),
    ("behavior_call", "what it does to the readouts", 236),
    ("behavior_effect_hz", "change", 478),
    ("behavior_q", "q", 548),
    ("dn_responders", "DN types", 610),
];
const KO_COLS: [(&str, &str, i32); 6] = [
    ("behavior", "behavior", 8),
    ("candidate", "cell type(s)", 190),
    ("drop_share", "drop", 400),
    ("q", "q", 470),
    ("neurons", "neurons", 540),
    ("rank", "rank", 610),
];

pub struct LabScreen {
    kind: &'static str,
    search: String,
    sort: Option<String>,
    desc: bool,
    sel: Option<String>,
    seeds: i64,
    size: i64,
}

impl LabScreen {
    pub fn activation() -> Self {
        Self { kind: "activation", search: String::new(), sort: None, desc: true, sel: None, seeds: 8, size: 60 }
    }

    pub fn knockout() -> Self {
        Self { kind: "knockout", ..Self::activation() }
    }

    pub fn set_view(&mut self, sel: Option<&str>, sort: Option<&str>, desc: bool, search: &str) {
        self.sel = sel.map(String::from);
        self.sort = sort.map(String::from);
        self.desc = desc;
        self.search = search.into();
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn fmt_q(q: &Value) -> String {
    let x = num(q);
    if x.is_nan() {
        String::new()
    } else if x < 0.001 {
        // Python's `f"{q:.1e}"`: a two-digit exponent
        let s = format!("{x:.1e}");
        match s.split_once('e') {
            Some((m, e)) => {
                let (sign, digits) = e.strip_prefix('-').map_or(("+", e), |d| ("-", d));
                format!("{m}e{sign}{digits:0>2}")
            }
            None => s,
        }
    } else {
        format!("{x:.3}")
    }
}

/// `f"{x:+{width}.{prec}f}"` (NaN as `nan`).
fn sf(x: f64, width: usize, prec: usize) -> String {
    if x.is_nan() { format!("{:>width$}", "nan") } else { format!("{x:+width$.prec$}") }
}

fn row_id(kind: &str, r: &Value) -> String {
    if kind == "activation" {
        r["type"].as_str().unwrap_or("").to_string()
    } else {
        format!("{}|{}", r["behavior"].as_str().unwrap_or(""), r["candidate"].as_str().unwrap_or(""))
    }
}

fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

impl LabScreen {
    fn rows(&self, res: &Value) -> Vec<Value> {
        let mut rows: Vec<Value> = res["rows"].as_array().cloned().unwrap_or_default();
        let q = self.search.trim().to_lowercase();
        if !q.is_empty() {
            let keys: &[&str] =
                if self.kind == "activation" { &["type", "behavior_call", "top_dn", "superclass"] } else { &["candidate", "behavior"] };
            rows.retain(|r| {
                keys.iter()
                    .any(|k| py_str(&r[*k]).to_lowercase().contains(&q) && !r[*k].is_null() || r[*k].is_null() && "none".contains(&q))
            });
        }
        if let Some(k) = &self.sort {
            let key = |r: &Value| -> (bool, Option<f64>, String) {
                let v = &r[k.as_str()];
                match v {
                    Value::Number(n) => {
                        let x = n.as_f64().unwrap_or(f64::NAN);
                        (x.is_nan(), Some(x), String::new())
                    }
                    Value::Null => (true, None, String::new()),
                    other => (false, None, py_str(other).to_lowercase()),
                }
            };
            let desc = self.desc;
            rows.sort_by(|a, b| {
                let (ka, kb) = (key(a), key(b));
                let o = ka.0.cmp(&kb.0).then_with(|| match (ka.1, kb.1) {
                    (Some(x), Some(y)) => x.total_cmp(&y),
                    _ => ka.2.cmp(&kb.2),
                });
                if desc { o.reverse() } else { o }
            });
        }
        rows
    }
}

impl Page for LabScreen {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[allow(clippy::too_many_lines)]
    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let kind = self.kind;
        let act = kind == "activation";
        let q = ctx.host.query(&format!("lab.screen_{kind}_state"));
        let (title, blurb) = if act {
            (
                "ACTIVATION SCREEN",
                "Hold each cell type driven in turn on held-out seeds, with matched random controls, and see which descending neurons respond and which behavior readout moves. Search a type, sort by effect, click a row to see its numbers or inspect its neurons.",
            )
        } else {
            (
                "KNOCKOUT SCREEN",
                "For each validated behavior, silence cell types one at a time (or in ranked batches first) and rank them by how much the response drops against matched random lesions. Click a row to see its numbers or inspect its neurons.",
            )
        };
        m.text(cv, title, (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        m.chip(cv, (rect.x + 40 + f.head.size(title), rect.y + 22), "MODEL PREDICTION");
        let mut y = m.subtitle(cv, rect, blurb, None, 3);
        let job = &q["job"];
        let busy = job.is_object();
        if busy {
            let frac = num(&job["done"]) / num(&job["total"]).max(1.0);
            cv.fill_rect(Rect::new(rect.x + 24, y + 6, rect.w - 220, 12), [30, 36, 48], 6);
            cv.fill_rect(Rect::new(rect.x + 24, y + 6, 8.max(((rect.w - 220) as f64 * frac) as i32), 12), AMBER, 6);
            let label: String = job["label"].as_str().unwrap_or("").chars().take(96).collect();
            m.text(
                cv,
                &format!("{label}  ·  {}/{} jobs  ·  {:.0}s", job["done"], job["total"], num(&job["elapsed"])),
                (rect.x + 24, y + 24),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            m.button(
                cv,
                Rect::new(rect.right() - 180, y - 2, 156, 32),
                "Stop",
                Act::Custom("stop".into()),
                &format!("scr:{kind}:stop"),
                Style::Normal,
                true,
                Some("Stops after the running jobs finish. Everything finished is kept: run it again and it continues."),
                false,
                None,
            );
            m.text(
                cv,
                "closing this page is safe: finished conditions are written as they go",
                (rect.x + 24, y + 42),
                DIM,
                &f.small,
                Anchor::TopLeft,
            );
            y += 60;
        } else {
            let has = q["res"].is_object();
            let label = if has { "Run it again (resumes)" } else { "Run the screen" };
            let w = m.bw(label, 190);
            m.button(
                cv,
                Rect::new(rect.x + 24, y, w, 34),
                label,
                Act::Custom("run".into()),
                &format!("scr:{kind}:run"),
                Style::Primary,
                true,
                Some("Runs on this machine's CPU threads. Minutes for a small selection; the whole connectome is hours: use the command line (--activation-screen / --knockout-screen) for that."),
                false,
                None,
            );
            let x = rect.x + 40 + w;
            m.page_slider(
                cv,
                Rect::new(x, y + 2, 190, 30),
                self.seeds as f64,
                (3.0, 8.0, 1.0),
                "{:.0f} seeds",
                "sc_seeds",
                true,
                Some("Held-out seeds 4000 upward. Fewer seeds run faster and detect less (eight is the default)."),
            );
            m.page_slider(
                cv,
                Rect::new(x + 210, y + 2, 230, 30),
                self.size as f64,
                (10.0, if act { 400.0 } else { 60.0 }, if act { 10.0 } else { 5.0 }),
                if act { "{:.0f} types" } else { "{:.0f} cand." },
                "sc_size",
                true,
                Some(if act {
                    "How many cell types, in name order, a quick run covers."
                } else {
                    "How many ranked candidate cell types to silence for each behavior."
                }),
            );
            y += 44;
            if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
                m.text(cv, &format!("error: {e}"), (rect.x + 24, y), BAD, &f.small, Anchor::TopLeft);
                y += 20;
            }
        }
        let res = &q["res"];
        let area = Rect::new(rect.x + 16, y, rect.w - 32, rect.bottom() - y - 66);
        if res.is_object() {
            self.table(m, cv, area, res, &q);
        } else {
            m.text(
                cv,
                "No results yet. Run the screen, or point --out at a folder to browse an earlier run.",
                (area.x + 8, area.y + 6),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
        }
        if res.is_object() {
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 58, m.bw("Export", 150), 42),
                "Export",
                Act::Custom("export".into()),
                &format!("scr:{kind}:export"),
                Style::Normal,
                true,
                Some("Copies the CSV files into a new folder in exports."),
                false,
                None,
            );
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            &format!("scr:{kind}:back"),
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let kind = self.kind;
        match name {
            "run" => drop(ctx.host.command("lab.screen_run", json!({"kind": kind, "seeds": self.seeds, "size": self.size}))),
            "stop" => drop(ctx.host.command("lab.screen_stop", Value::Null)),
            "export" => drop(ctx.host.command("lab.screen_export", json!({"kind": kind}))),
            "slide:sc_seeds" => {
                if let Some(v) = text.and_then(|t| t.parse::<f64>().ok()) {
                    self.seeds = v as i64;
                }
            }
            "slide:sc_size" => {
                if let Some(v) = text.and_then(|t| t.parse::<f64>().ok()) {
                    self.size = v as i64;
                }
            }
            "text:sc_search" => self.search = text.unwrap_or("").to_string(),
            "close" => self.sel = None,
            n if n.starts_with("sort:") => {
                let k = &n[5..];
                if self.sort.as_deref() == Some(k) {
                    self.desc = !self.desc;
                } else {
                    self.desc = !["type", "candidate", "behavior", "rank", "behavior_call"].contains(&k);
                }
                self.sort = Some(k.to_string());
            }
            n if n.starts_with("row:") => self.sel = Some(n[4..].to_string()),
            n if n.starts_with("live:") => {
                if let Some((mode, types)) = n[5..].split_once(':') {
                    let types: Vec<&str> = types.split('+').collect();
                    drop(ctx.host.command("lab.screen_toggle", json!({"types": types, "mode": mode.parse::<i64>().unwrap_or(1)})));
                }
            }
            n if n.starts_with("inspect:") => drop(ctx.host.command("lab.screen_inspect", json!({"type": &n[8..]}))),
            _ => {}
        }
    }
}

impl LabScreen {
    fn table(&self, m: &mut Menu, cv: &mut Canvas, area: Rect, res: &Value, q: &Value) {
        let f = m.fonts();
        let kind = self.kind;
        let act = kind == "activation";
        let cols: &[(&str, &str, i32)] = if act { &ACT_COLS } else { &KO_COLS };
        let rows = self.rows(res);
        let detail_w = if self.sel.is_some() { 340 } else { 0 };
        let left = Rect::new(area.x, area.y, area.w - detail_w - if detail_w > 0 { 8 } else { 0 }, area.h);
        let seeds: Vec<i64> = res["seeds"].as_array().into_iter().flatten().filter_map(Value::as_i64).collect();
        let (s0, s1) = (seeds.first().copied().unwrap_or(0), seeds.last().copied().unwrap_or(0));
        let grp = |n: u64| super::day4::grp(n as f64, 0);
        let head = if act {
            format!(
                "{} types, seeds {s0}-{s1}, {} with a behavior call",
                grp(res["n_types"].as_u64().unwrap_or(rows.len() as u64)),
                res["n_called"].as_u64().unwrap_or(0)
            )
        } else {
            format!("{} behaviors, seeds {s0}-{s1}", res["behaviors"].as_array().map_or(0, Vec::len))
        };
        m.text(cv, &format!("{head}  ·  showing {}", grp(rows.len() as u64)), (left.x + 8, left.y), INK, &f.small, Anchor::TopLeft);
        let bx = Rect::new(left.right() - 230, left.y - 4, 224, 26);
        m.text_field(cv, bx, &self.search, "sc_search", Some("Filter by cell type or what it does. Type and press Enter."), 40, false);
        if self.search.is_empty() {
            m.text(cv, "search…", (bx.x + 10, bx.centery()), DIM, &f.small, Anchor::MidLeft);
        }
        let mut y = left.y + 26;
        for (key, label, dx) in cols {
            let r = Rect::new(left.x + dx - 4, y - 2, f.small.size(label) + 22, 20);
            let arrow = if self.sort.as_deref() == Some(*key) { if self.desc { " v" } else { " ^" } } else { "" };
            m.click_area(r, Act::Custom(format!("sort:{key}")), &format!("scr:{kind}:sort:{key}"), true, Some(&format!("Sort by {label}")));
            m.text(
                cv,
                &format!("{label}{arrow}"),
                (left.x + dx, y),
                if self.sort.as_deref() == Some(*key) { AMBER } else { LABEL },
                &f.small,
                Anchor::TopLeft,
            );
        }
        y += 20;
        let view = Rect::new(left.x, y, left.w, left.bottom() - y);
        let key = format!("lab_screen_{kind}");
        let off = m.scroll_offset(&key);
        m.begin_clip(cv, view);
        let mut yy = y - off;
        for r in &rows {
            let id = row_id(kind, r);
            let sel = self.sel.as_deref() == Some(id.as_str());
            let rr = Rect::new(left.x, yy - 2, left.w, 22);
            if yy + 22 >= view.y && yy <= view.bottom() {
                let over = m.click_area(rr, Act::Custom(format!("row:{id}")), &format!("scr:{kind}:row:{id}"), true, None);
                if sel || over {
                    cv.fill_rect(rr, if sel { [44, 52, 68] } else { ROW_HOVER }, 4);
                }
                self.row(m, cv, left, yy, r);
            }
            yy += 22;
        }
        m.set_content_h(&key, (yy + off - view.bottom() + 8).max(0));
        m.end_clip(cv);
        if detail_w > 0 {
            let sel = res["rows"].as_array().into_iter().flatten().find(|r| Some(row_id(kind, r)) == self.sel);
            if let Some(r) = sel {
                self.detail(m, cv, Rect::new(area.right() - detail_w, area.y, detail_w, area.h), res, r, q);
            }
        }
    }

    fn row(&self, m: &Menu, cv: &mut Canvas, left: Rect, yy: i32, r: &Value) {
        let f = m.fonts();
        let put = |cv: &mut Canvas, dx: i32, t: &str, c: Color| {
            m.text(cv, t, (left.x + dx, yy), c, &f.small, Anchor::TopLeft);
        };
        if self.kind == "activation" {
            let called = r["behavior_readout"].as_str().is_some_and(|s| !s.is_empty());
            let eff = num(&r["behavior_effect_hz"]);
            let col = if called { if eff > 0.0 { GOOD } else { BAD } } else { DIM };
            let call = r["behavior_call"].as_str().unwrap_or("");
            let call_s: String =
                if call.chars().count() > 33 { format!("{}…", call.chars().take(33).collect::<String>()) } else { call.into() };
            let vals: Vec<(String, Color)> = vec![
                (r["rank"].as_u64().unwrap_or(0).to_string(), DIM),
                (r["type"].as_str().unwrap_or("").into(), if called { TEXT } else { LABEL }),
                (super::day4::grp(num(&r["neurons"]), 0), LABEL),
                (call_s, col),
                (if called { format!("{eff:+.1} Hz") } else { String::new() }, col),
                (fmt_q(&r["behavior_q"]), LABEL),
                (r["dn_responders"].as_u64().map_or(String::new(), |n| n.to_string()), LABEL),
            ];
            for ((_, _, dx), (t, c)) in ACT_COLS.iter().zip(&vals) {
                put(cv, *dx, t, *c);
            }
        } else {
            let strong = num(&r["q"]) < 0.05 && num(&r["drop_share"]) >= 0.1;
            let col = if strong { BAD } else { TEXT };
            let ds = num(&r["drop_share"]);
            let vals: Vec<(String, Color)> = vec![
                (r["behavior"].as_str().unwrap_or("").into(), LABEL),
                (format!("{}{}", r["candidate"].as_str().unwrap_or(""), if r["is_batch"] == json!(true) { "  (batch)" } else { "" }), col),
                (if ds.is_nan() { String::new() } else { format!("{:+.0}%", ds * 100.0) }, col),
                (fmt_q(&r["q"]), LABEL),
                (super::day4::grp(num(&r["neurons"]), 0), LABEL),
                (r["rank"].as_u64().map_or(String::new(), |n| n.to_string()), DIM),
            ];
            for ((_, _, dx), (t, c)) in KO_COLS.iter().zip(&vals) {
                let s: String = t.chars().take(30).collect();
                put(cv, *dx, &s, *c);
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn detail(&self, m: &mut Menu, cv: &mut Canvas, area: Rect, res: &Value, r: &Value, q: &Value) {
        let f = m.fonts();
        let kind = self.kind;
        cv.fill_rect(area, [14, 17, 24], 10);
        cv.stroke_rect(area, BORDER, 10);
        let (x, mut y) = (area.x + 12, area.y + 10);
        let lh = f.small.linesize + 2;
        let (types, mode): (Vec<String>, i64);
        if kind == "activation" {
            m.text(cv, r["type"].as_str().unwrap_or(""), (x, y), INK, &f.bold, Anchor::TopLeft);
            y += lh + 2;
            let sc = r["superclass"].as_str().filter(|s| !s.is_empty()).unwrap_or("unclassified");
            m.text(
                cv,
                &format!("{} neurons · {sc} · {} seeds", super::day4::grp(num(&r["neurons"]), 0), r["n_seeds"]),
                (x, y),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            y += lh;
            let called = r["behavior_readout"].as_str().is_some_and(|s| !s.is_empty());
            m.text(
                cv,
                r["behavior_call"].as_str().unwrap_or(""),
                (x, y),
                if called && num(&r["behavior_effect_hz"]) > 0.0 { GOOD } else { TEXT },
                &f.small,
                Anchor::TopLeft,
            );
            y += lh + 4;
            m.text(cv, "READOUTS vs MATCHED CONTROLS (Hz, CI, q)", (x, y), DIM, &f.small, Anchor::TopLeft);
            y += lh;
            let own_list: Vec<&str> = r["self_readouts"].as_str().unwrap_or("").split(',').collect();
            for rd in res["readouts"].as_array().into_iter().flatten() {
                let rid = rd["id"].as_str().unwrap_or("");
                if r[format!("eff_{rid}").as_str()].is_null() && r.get(format!("eff_{rid}").as_str()).is_none() {
                    continue;
                }
                let e = num(&r[format!("eff_{rid}").as_str()]);
                let (lo, hi) = (num(&r[format!("lo_{rid}").as_str()]), num(&r[format!("hi_{rid}").as_str()]));
                let qv = &r[format!("q_{rid}").as_str()];
                let own = own_list.contains(&rid);
                let tail = if own { "   itself".to_string() } else { format!("  {}", fmt_q(qv)) };
                let txt = format!("{rid:<12}{}  [{}, {}]{tail}", sf(e, 7, 1), sf(lo, 0, 1), sf(hi, 0, 1));
                let qn = num(qv);
                let col = if own {
                    DIM
                } else if !qn.is_nan() && qn < 0.05 {
                    TEXT
                } else {
                    LABEL
                };
                m.text(cv, &txt, (x, y), col, &f.small, Anchor::TopLeft);
                y += lh - 3;
            }
            y += 4;
            m.text(
                cv,
                &format!("DESCENDING TYPES MOVED ({})", r["dn_responders"].as_u64().unwrap_or(0)),
                (x, y),
                DIM,
                &f.small,
                Anchor::TopLeft,
            );
            y += lh;
            for d in r["_dn"].as_array().into_iter().flatten().take(5) {
                m.text(
                    cv,
                    &format!("{:<14}{} Hz", d[0].as_str().unwrap_or(""), sf(num(&d[1]), 7, 1)),
                    (x, y),
                    TEXT,
                    &f.small,
                    Anchor::TopLeft,
                );
                y += lh - 1;
            }
            types = vec![r["type"].as_str().unwrap_or("").to_string()];
            mode = 1;
        } else {
            let cand = r["candidate"].as_str().unwrap_or("");
            m.text(cv, cand, (x, y), INK, &f.bold, Anchor::TopLeft);
            y += lh + 2;
            m.text(
                cv,
                &format!(
                    "{} · {} neurons · {} seeds",
                    r["behavior"].as_str().unwrap_or(""),
                    super::day4::grp(num(&r["neurons"]), 0),
                    r["n_seeds"]
                ),
                (x, y),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            y += lh + 4;
            let ds = num(&r["drop_share"]);
            let lines: Vec<(&str, String)> = vec![
                ("unperturbed response", format!("{:.2} Hz", num(&r["unperturbed_evoked_hz"]))),
                ("with this lesion", format!("{:.2} Hz", num(&r["lesion_evoked_hz"]))),
                ("matched random lesions", format!("{:.2} Hz", num(&r["control_evoked_hz"]))),
                ("drop vs matched", format!("{:+.2} Hz  [{:+.2}, {:+.2}]", num(&r["drop_hz"]), num(&r["drop_lo"]), num(&r["drop_hi"]))),
                ("as a share", format!("{:+.0}%", ds * 100.0)),
                ("q (screen-wide)", fmt_q(&r["q"])),
                ("seeds the same way", format!("{:.0}%", num(&r["sign_share"]) * 100.0)),
            ];
            for (a, b) in lines {
                m.text(cv, &format!("{a:<24}{b}"), (x, y), TEXT, &f.small, Anchor::TopLeft);
                y += lh - 1;
            }
            if r["is_batch"] == json!(true) {
                m.text(cv, "A batch: silenced together. Batches that cut the response", (x, y + 4), DIM, &f.small, Anchor::TopLeft);
                m.text(cv, "were opened and their members tested one by one.", (x, y + 4 + lh - 2), DIM, &f.small, Anchor::TopLeft);
            }
            types = cand.split('+').map(String::from).collect();
            mode = -1;
        }
        let by = area.bottom() - 78;
        let ops = &q["type_ops"];
        let live = !types.is_empty() && types.iter().all(|t| ops[t.as_str()].as_i64() == Some(mode));
        let verb = if mode > 0 { "Activate on live flies" } else { "Silence on live flies" };
        m.button(
            cv,
            Rect::new(x, by, area.w - 24, 32),
            if live { "undo on live flies" } else { verb },
            Act::Custom(format!("live:{mode}:{}", types.join("+"))),
            &format!("scr:{kind}:live"),
            Style::Normal,
            true,
            Some("The brain surgery switch on every live fly, so you can watch what this row means in the game."),
            live,
            None,
        );
        m.button(
            cv,
            Rect::new(x, by + 38, area.w - 24, 32),
            "Inspect a neuron of this type",
            Act::Custom(format!("inspect:{}", types[0])),
            &format!("scr:{kind}:inspect"),
            Style::Normal,
            true,
            Some("Closes the menu and opens the big brain view with the neuron inspector on one of its neurons."),
            false,
            None,
        );
        let closer = Rect::new(area.right() - 30, area.y + 6, 24, 24);
        m.click_area(closer, Act::Custom("close".into()), &format!("scr:{kind}:close"), true, Some("Close this panel"));
        m.text(cv, "x", closer.center(), LABEL, &f.bold, Anchor::Center);
    }
}
