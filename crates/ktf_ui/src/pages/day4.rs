//! The Lab pages added in 3.0 day 4 (`lab/labday4.py`): network science, sleep deprivation and sensitivity analysis. Each runs its
//! analysis in a background job that the game owns; the page only reads it.
//!
//! Host: query `lab.day4_state` = `{job: {label, frac, note, elapsed} | null, error, msg, net: {brain: result}, cached: {brain: bool},
//! sleep: result | null, sens: result | null, components: [[label, tag], ...] (sleepdep.COMPONENTS), parameters: [label, ...]
//! (sensitivity.PARAMETERS), palette: [ok, bad, ok_t, bad_t]}`; commands `net_start {brain, force}`, `net_export {brain}`,
//! `sleep_start {seeds}`, `sens_start {smoke}`, `day4_cancel`.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::i18n::tr;
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

/// Python's `f"{x:,.{dec}f}"`.
pub fn grp(x: f64, dec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    let s = format!("{:.*}", dec, x.abs());
    let (int, frac) = s.split_once('.').map_or((s.as_str(), None), |(a, b)| (a, Some(b)));
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if let Some(f) = frac {
        out.push('.');
        out.push_str(f);
    }
    if x < 0.0 && out.chars().any(|c| c != '0' && c != '.' && c != ',') { format!("-{out}") } else { out }
}

fn fx(x: f64, dec: usize) -> String {
    if x.is_nan() { "nan".into() } else { format!("{x:.dec$}") }
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn pad_left(s: &str, w: usize) -> String {
    format!("{s:>w$}")
}

/// `_tags`: coloured chips, returns the x after the last one.
pub fn tags(m: &Menu, cv: &mut Canvas, mut x: i32, y: i32, names: &[&str]) -> i32 {
    let f = m.fonts();
    for n in names {
        let col = match *n {
            "CONNECTOME" => [60, 170, 220],
            "GAME RULE" => [220, 150, 50],
            "LITERATURE" => [110, 190, 130],
            _ => [150, 120, 220],
        };
        let w = f.small.size(n);
        let r = Rect::new(x, y, w + 12, f.small.linesize + 4);
        cv.fill_rect(r, col, 5);
        f.small.draw(cv, n, r.x + 6, r.y + 2, [10, 12, 16]);
        x = r.right() + 6;
    }
    x
}

struct Chrome {
    body: Rect,
    off: i32,
}

fn chrome(m: &mut Menu, cv: &mut Canvas, rect: Rect, title: &str, sub: &str, key: &str) -> Chrome {
    let f = m.fonts();
    m.text(cv, title, (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
    let top = (rect.y + 80).max(m.subtitle(cv, rect, sub, Some(rect.y + 50), 3));
    let body = Rect::new(rect.x + 16, top, rect.w - 32, rect.bottom() - 80 - top);
    Chrome { body, off: m.scroll_offset(key) }
}

/// `_end`: the scroll extent (Python passes `y + off`, so the offset counts twice: kept), the error, the progress bar and Back.
fn end(m: &mut Menu, cv: &mut Canvas, rect: Rect, c: &Chrome, key: &str, y: i32, q: &Value) {
    let f = m.fonts();
    m.set_content_h(key, (y + c.off - c.body.bottom() + 8).max(0));
    if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
        m.text(cv, e, (rect.x + 24, rect.bottom() - 74), BAD, &f.small, Anchor::TopLeft);
    }
    if q["job"].is_object() {
        let j = &q["job"];
        let r = Rect::new(rect.x + 24, rect.bottom() - 56, rect.w - 220, 12);
        cv.fill_rect(Rect::new(r.x, r.y, r.w - 110, 12), [30, 36, 48], 6);
        let frac = num(&j["frac"]).clamp(0.0, 1.0);
        cv.fill_rect(Rect::new(r.x, r.y, 8.max(((r.w - 110) as f64 * frac) as i32), 12), AMBER, 6);
        m.text(
            cv,
            &format!("{}: {}  ({:.0} s)", j["label"].as_str().unwrap_or(""), j["note"].as_str().unwrap_or(""), num(&j["elapsed"])),
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
    m.button(
        cv,
        Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
        &tr("Back"),
        Act::Back,
        &format!("{key}:back"),
        Style::Primary,
        true,
        None,
        false,
        None,
    );
}

fn busy(q: &Value) -> bool {
    q["job"].is_object()
}

// --- network science ---------------------------------------------------------------------------------------------------------
pub struct LabNetsci {
    brain: String,
}

impl Default for LabNetsci {
    fn default() -> Self {
        Self { brain: "adult".into() }
    }
}

fn hist(m: &Menu, cv: &mut Canvas, x: i32, y: i32, w: i32, hin: &[Value], hout: &[Value]) -> i32 {
    let f = m.fonts();
    let bars = hin.len().max(hout.len()) as i32;
    let h = 70;
    let count = |b: &Value| b["count"].as_f64().unwrap_or(0.0);
    let top = hin.iter().chain(hout).map(count).fold(1.0, f64::max);
    let bw = 6.max(36.min((w - 20) / bars.max(1)));
    cv.fill_rect(Rect::new(x, y, bars * bw * 2 + 10, h + 8), [18, 22, 30], 6);
    for i in 0..bars as usize {
        for (k, (series, col)) in [(hin, [57, 135, 229]), (hout, [217, 89, 38])].into_iter().enumerate() {
            let c = series.get(i).map_or(0.0, count);
            let hh = (f64::from(h) * c / top) as i32;
            cv.fill_rect(Rect::new(x + 5 + i as i32 * bw * 2 + k as i32 * bw, y + 4 + h - hh, bw - 1, hh), col, 0);
        }
    }
    m.text(
        cv,
        &tr("blue: in-degree, orange: out-degree; bins 0, 1, 2-3, 4-7, ..."),
        (x + bars * bw * 2 + 18, y + 4),
        LABEL,
        &f.small,
        Anchor::TopLeft,
    );
    y + h + 20
}

impl Page for LabNetsci {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.day4_state");
        let key = "lab_netsci";
        let c = chrome(
            m,
            cv,
            rect,
            &tr("NETWORK SCIENCE"),
            &tr("Degrees, reciprocity, motifs against a null model, rich club, communities and regions of the brain pack."),
            key,
        );
        let busy = busy(&q);
        let (mut x, mut y) = (c.body.x + 8, c.body.y + 4);
        for b in ["adult", "larva"] {
            let label = format!("{}{}", b[..1].to_uppercase(), &b[1..]);
            m.button(
                cv,
                Rect::new(x, y, 110, 32),
                &tr(&label),
                Act::Custom(format!("brain:{b}")),
                &format!("ns_brain:{b}"),
                Style::Normal,
                !busy,
                None,
                self.brain == b,
                None,
            );
            x += 118;
        }
        let cached = q["cached"][&self.brain].as_bool().unwrap_or(false);
        m.button(
            cv,
            Rect::new(x + 10, y, 220, 32),
            &if cached { tr("Compute (cached)") } else { tr("Compute") },
            Act::Custom("run".into()),
            "ns_run",
            Style::Primary,
            !busy,
            Some(&tr("Adult takes minutes (10 million connections), larva seconds. Cached next to your data with a checksum; recomputed if the pack or settings change.")),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(x + 240, y, 150, 32),
            &tr("Recompute"),
            Act::Custom("force".into()),
            "ns_force",
            Style::Normal,
            !busy,
            None,
            false,
            None,
        );
        let res = &q["net"][&self.brain];
        m.button(
            cv,
            Rect::new(x + 400, y, 190, 32),
            &tr("Export CSV"),
            Act::Custom("export".into()),
            "ns_csv",
            Style::Normal,
            res.is_object() && !busy,
            Some(&tr("Writes one CSV per table into your exports folder.")),
            false,
            None,
        );
        m.begin_clip(cv, c.body);
        y += 44 - c.off;
        x = c.body.x + 8;
        let xx = tags(m, cv, x, y, &["CONNECTOME"]);
        y = (y + 28).max(
            m.wrapped(
                cv,
                &tr("computed from the wiring (no simulation); the analysis choices (null graphs, samples, seed) are GAME RULE; nothing here is a MODEL PREDICTION"),
                (xx + 4, y + 2),
                c.body.right() - 12 - xx,
                LABEL,
                &f.small,
                3,
            ) + 8,
        );
        if !res.is_object() {
            m.text(cv, &tr("Not computed yet."), (x, y), LABEL, &f.text, Anchor::TopLeft);
            y += 30;
        } else {
            let (d, r, cm) = (&res["degrees"], &res["reciprocity"], &res["communities"]);
            let io = |k: &str| {
                let v = &d[k];
                (num(&v["mean"]), num(&v["median"]), num(&v["p99"]), num(&v["max"]))
            };
            let (i, o) = (io("in_degree"), io("out_degree"));
            let lines = [
                format!(
                    "{} {}, {} {}, {} {}  ·  {}  ·  {} {}",
                    grp(num(&d["n"]), 0),
                    tr("neurons"),
                    grp(num(&d["connections"]), 0),
                    tr("connections"),
                    grp(num(&d["synapses"]), 0),
                    tr("synapses"),
                    res["pack_name"].as_str().unwrap_or(""),
                    if res["from_cache"].as_bool().unwrap_or(false) { "cache".to_string() } else { tr("computed") },
                    res["created"].as_str().unwrap_or("")
                ),
                format!(
                    "{}: {} {}, {} {}, p99 {}, {} {}   ·   {}: {} {}, {} {}, p99 {}, {} {}",
                    tr("In-degree"),
                    tr("mean"),
                    fx(i.0, 1),
                    tr("median"),
                    fx(i.1, 0),
                    fx(i.2, 0),
                    tr("max"),
                    fx(i.3, 0),
                    tr("Out-degree"),
                    tr("mean"),
                    fx(o.0, 1),
                    tr("median"),
                    fx(o.1, 0),
                    fx(o.2, 0),
                    tr("max"),
                    fx(o.3, 0)
                ),
                format!(
                    "{}: {} ({} {}, x{}); {} {}",
                    tr("Reciprocity"),
                    fx(num(&r["binary"]), 3),
                    tr("null"),
                    fx(num(&r["null_binary_mean"]), 3),
                    fx(num(&r["enrichment"]), 1),
                    tr("synapse-weighted"),
                    fx(num(&r["weighted"]), 3)
                ),
                format!(
                    "{}: {}, {} Q = {}, NMI {} {}",
                    tr("Communities"),
                    grp(num(&cm["communities"]), 0),
                    tr("modularity"),
                    fx(num(&cm["modularity"]), 3),
                    tr("with regions"),
                    fx(num(&cm["nmi_with_regions"]), 2)
                ),
            ];
            for ln in lines {
                m.text(cv, &ln, (x, y), TEXT, &f.small, Anchor::TopLeft);
                y += 22;
            }
            y += 6;
            m.text(cv, &tr("In-degree histogram (log2 bins)"), (x, y), INK, &f.bold, Anchor::TopLeft);
            y += 24;
            let h = |k: &str| d[k]["histogram"].as_array().cloned().unwrap_or_default();
            y = hist(m, cv, x, y, c.body.w - 40, &h("in_degree"), &h("out_degree"));
            m.text(
                cv,
                &tr("3-node motifs (sampled estimate; enrichment over the degree-preserving null)"),
                (x, y),
                INK,
                &f.bold,
                Anchor::TopLeft,
            );
            y += 24;
            for mo in res["motifs"].as_array().into_iter().flatten() {
                let e = mo["enrichment"].as_f64();
                let zc = match e {
                    Some(e) if e > 1.5 => GOOD,
                    Some(e) if e < 0.67 => BAD,
                    _ => TEXT,
                };
                let enr = e.map_or("undefined".to_string(), |e| format!("x{e:.2}"));
                let z = mo["z"].as_f64().map_or("undefined".to_string(), |z| format!("{z:.1}"));
                m.text(
                    cv,
                    &format!(
                        "{:<5} {}   {}   z {}   {}",
                        mo["motif"].as_str().unwrap_or(""),
                        pad_left(&grp(num(&mo["count_estimate"]), 0), 14),
                        pad_left(&enr, 9),
                        pad_left(&z, 9),
                        mo["description"].as_str().unwrap_or("")
                    ),
                    (x + 8, y),
                    zc,
                    &f.small,
                    Anchor::TopLeft,
                );
                if let (None, Some(note)) = (e, mo["enrichment_note"].as_str().filter(|n| !n.is_empty())) {
                    y += 17;
                    m.text(cv, &format!("      {note}"), (x + 8, y), LABEL, &f.small, Anchor::TopLeft);
                }
                y += 19;
            }
            y += 8;
            m.text(
                cv,
                &tr("Rich club (rho = phi / phi_null; above 1 means more than degrees alone give)"),
                (x, y),
                INK,
                &f.bold,
                Anchor::TopLeft,
            );
            y += 24;
            let rich = res["rich_club"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|q| format!("k>{}: {}", fx(num(&q["k"]), 0), fx(num(&q["rho"]), 2)))
                .collect::<Vec<_>>()
                .join("   ");
            m.text(cv, &rich, (x + 8, y), TEXT, &f.small, Anchor::TopLeft);
            y += 28;
            m.text(cv, &tr("Regions"), (x, y), INK, &f.bold, Anchor::TopLeft);
            y += 24;
            for q in res["regions"].as_array().into_iter().flatten() {
                m.text(
                    cv,
                    &format!(
                        "{:<16} {} {}   {} {} / {} {}   {}% {}   {}% {}",
                        q["region"].as_str().unwrap_or(""),
                        pad_left(&grp(num(&q["neurons"]), 0), 8),
                        tr("neurons"),
                        tr("in"),
                        fx(num(&q["mean_in_degree"]), 1),
                        tr("out"),
                        fx(num(&q["mean_out_degree"]), 1),
                        fx(num(&q["share_out_staying_inside"]) * 100.0, 0),
                        tr("of output stays inside"),
                        fx(num(&q["inhibitory_share_of_out_synapses"]) * 100.0, 0),
                        tr("inhibitory")
                    ),
                    (x + 8, y),
                    TEXT,
                    &f.small,
                    Anchor::TopLeft,
                );
                y += 19;
            }
            y += 6;
            if let Some(msg) = q["msg"].as_str().filter(|s| !s.is_empty()) {
                m.text(cv, msg, (x, y), GOOD, &f.small, Anchor::TopLeft);
                y += 22;
            }
        }
        m.end_clip(cv);
        end(m, cv, rect, &c, key, y + c.off, &q);
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        match name.split_once(':') {
            Some(("brain", b)) => self.brain = b.into(),
            _ => match name {
                "run" => drop(ctx.host.command("lab.net_start", json!({"brain": self.brain, "force": false}))),
                "force" => drop(ctx.host.command("lab.net_start", json!({"brain": self.brain, "force": true}))),
                "export" => drop(ctx.host.command("lab.net_export", json!({"brain": self.brain}))),
                "cancel" => drop(ctx.host.command("lab.day4_cancel", Value::Null)),
                _ => {}
            },
        }
    }
}

// --- sleep deprivation -----------------------------------------------------------------------------------------------------------
pub struct LabSleepdep {
    seeds: i64,
}

impl Default for LabSleepdep {
    fn default() -> Self {
        Self { seeds: 5 }
    }
}

fn fmt_p(p: f64) -> String {
    if p.is_nan() {
        "p = n/a".into()
    } else if p < 0.001 {
        "p < 0.001".into()
    } else {
        format!("p = {p:.3}")
    }
}

impl Page for LabSleepdep {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.day4_state");
        let key = "lab_sleepdep";
        let c = chrome(
            m,
            cv,
            rect,
            &tr("SLEEP DEPRIVATION"),
            &tr("Keep a fly awake through the night, then measure its rebound sleep against an undisturbed control of the same seed."),
            key,
        );
        let busy = busy(&q);
        let (x, mut y) = (c.body.x + 8, c.body.y + 4);
        m.page_slider(
            cv,
            Rect::new(x, y, 260, 32),
            self.seeds as f64,
            (3.0, 10.0, 1.0),
            "{:.0f} flies",
            "sd_n",
            !busy,
            Some(&tr("Seeds 1000 to 1000+n-1 (the validation seeds). A paired test needs at least 7 to reach p < 0.01.")),
        );
        m.button(
            cv,
            Rect::new(x + 290, y, 190, 32),
            &tr("Run the assay"),
            Act::Custom("run".into()),
            "sd_run",
            Style::Primary,
            !busy,
            Some(&tr("Each fly runs twice (deprived and control), about 2 minutes of simulated time each, in worker processes.")),
            false,
            None,
        );
        m.begin_clip(cv, c.body);
        y += 44 - c.off;
        m.text(cv, &tr("What is what in this assay:"), (x, y), INK, &f.bold, Anchor::TopLeft);
        y += 24;
        for (label, tag) in fly_sim_components() {
            let t = ["GAME RULE", "CONNECTOME", "MODEL PREDICTION"].into_iter().find(|t| tag.starts_with(t)).unwrap_or("GAME RULE");
            let xx = tags(m, cv, x + 8, y, &[t]);
            m.text(cv, &tr(label), (xx + 4, y + 2), TEXT, &f.small, Anchor::TopLeft);
            y += 24;
        }
        y += 8;
        let res = &q["sleep"];
        if !res.is_object() {
            m.text(cv, &tr("Not run yet."), (x, y), LABEL, &f.text, Anchor::TopLeft);
            y += 30;
        } else {
            let mm = &res["mean"];
            let lines = [
                format!(
                    "{}: {} {} s, {} {} s",
                    tr("Sleep in the deprivation window"),
                    tr("control"),
                    fx(num(&mm["deprivation_sleep_control"]), 1),
                    tr("deprived"),
                    fx(num(&mm["deprivation_sleep_deprived"]), 1)
                ),
                format!(
                    "{} (GAME RULE): {} vs {}   ·   {} (CONNECTOME): {}x vs {}x",
                    tr("Sleep pressure at its end"),
                    fx(num(&mm["pressure_control"]), 2),
                    fx(num(&mm["pressure_deprived"]), 2),
                    tr("dFB level"),
                    fx(num(&mm["dfb_control"]), 2),
                    fx(num(&mm["dfb_deprived"]), 2)
                ),
                format!(
                    "{}: {} {} s, {} {} s",
                    tr("Sleep in the recovery window"),
                    tr("control"),
                    fx(num(&mm["recovery_sleep_control"]), 1),
                    tr("deprived"),
                    fx(num(&mm["recovery_sleep_deprived"]), 1)
                ),
            ];
            for ln in lines {
                m.text(cv, &ln, (x, y), TEXT, &f.small, Anchor::TopLeft);
                y += 22;
            }
            for (cid, cr) in res["criteria"].as_object().into_iter().flatten() {
                let extra = if cr.get("p").is_some() { format!("  {}", fmt_p(num(&cr["p"]))) } else { String::new() };
                let ok = cr["passed"].as_bool().unwrap_or(false);
                m.text(
                    cv,
                    &format!("{cid} {}  {}{extra}", if ok { "PASS" } else { "FAIL" }, cr["label"].as_str().unwrap_or("")),
                    (x + 8, y),
                    if ok { GOOD } else { BAD },
                    &f.small,
                    Anchor::TopLeft,
                );
                y += 20;
            }
            m.text(
                cv,
                &tr("A rebound is expected from the pressure rule (a game rule): this shows it works through the real dFB neurons."),
                (x, y + 6),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
            y += 34;
        }
        m.end_clip(cv);
        end(m, cv, rect, &c, key, y + c.off, &q);
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        match name {
            "run" => drop(ctx.host.command("lab.sleep_start", json!({"seeds": self.seeds}))),
            "cancel" => drop(ctx.host.command("lab.day4_cancel", Value::Null)),
            "slide:sd_n" => {
                if let Some(v) = text.and_then(|t| t.parse::<f64>().ok()) {
                    self.seeds = v.round() as i64;
                }
            }
            _ => {}
        }
    }
}

/// `sleepdep.COMPONENTS`.
fn fly_sim_components() -> [(&'static str, &'static str); 7] {
    [
        ("sleep pressure (rises awake, falls asleep)", "GAME RULE"),
        ("the disturbances and the forced wake", "GAME RULE"),
        ("what counts as sleep: dFB above THRESH['sleep'] for 1 s", "GAME RULE"),
        ("pressure -> current on every FB6/FB7 neuron", "GAME RULE"),
        ("dFB (FB6/FB7) firing in response to that current", "CONNECTOME readout"),
        ("daylight on the photoreceptors and the l-LNv/s-LNv clock neurons", "CONNECTOME (the drive is a game rule)"),
        ("the rebound itself", "MODEL PREDICTION"),
    ]
}

// --- sensitivity analysis ----------------------------------------------------------------------------------------------------
/// `sensitivity.PARAMETERS` labels, for the "not run yet" line.
const PARAM_LABELS: [&str; 6] =
    ["Membrane noise", "Tonic drive", "Target rate (Hz)", "Sensory input gain", "Gain adaptation rate", "Synapse threshold"];

/// `sensitivity.PALETTES`: (pass, fail, pass text, fail text).
fn palette(name: &str) -> ([u8; 3], [u8; 3], [u8; 3], [u8; 3]) {
    match name {
        "blue-yellow" => ([30, 100, 200], [240, 190, 20], [255, 255, 255], [20, 20, 20]),
        "high-contrast" => ([255, 255, 255], [0, 0, 0], [0, 0, 0], [255, 255, 255]),
        _ => ([46, 125, 50], [183, 28, 28], [255, 255, 255], [255, 255, 255]),
    }
}

/// `{x:g}` for the other pages.
pub fn fmt_g_pub(x: f64) -> String {
    fmt_gp(x, 6)
}

/// Python's `format(x, ".{p}g")`.
pub fn fmt_gp(x: f64, p: usize) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let p = p.max(1);
    let e = format!("{:.*e}", p - 1, x);
    let (mant, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if exp < -4 || exp >= p as i32 {
        let m = if mant.contains('.') { mant.trim_end_matches('0').trim_end_matches('.') } else { mant };
        return format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let dec = (p as i32 - 1 - exp).max(0) as usize;
    let s = format!("{x:.dec$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

fn fmt_g(x: f64) -> String {
    fmt_gp(x, 6)
}

pub struct LabSensitivity {
    smoke: bool,
}

impl Default for LabSensitivity {
    fn default() -> Self {
        Self { smoke: true }
    }
}

impl Page for LabSensitivity {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.day4_state");
        let key = "lab_sensitivity";
        let c = chrome(
            m,
            cv,
            rect,
            &tr("SENSITIVITY ANALYSIS"),
            &tr(
                "Vary each LIF parameter and re-run the validated behaviors with validation's own criteria. Analysis only: the defaults never change.",
            ),
            key,
        );
        let busy = busy(&q);
        let (quick, full) = (tr("Quick check (3 seeds)"), tr("Full grid (10 seeds)"));
        let (x, mut y) = (c.body.x + 8, c.body.y + 4);
        let mode_w = m.bw(&quick, 250).max(m.bw(&full, 250));
        m.button(
            cv,
            Rect::new(x, y, mode_w, 32),
            if self.smoke { &quick } else { &full },
            Act::Custom("mode".into()),
            "se_mode",
            Style::Normal,
            !busy,
            Some(&tr("The quick check varies one parameter at two values on 2 behaviors and 3 seeds (underpowered: a FAIL means 'cannot pass at n = 3'). The full grid is hours; run it headless with --sensitivity.")),
            false,
            None,
        );
        m.button(
            cv,
            Rect::new(x + mode_w + 10, y, 190, 32),
            &tr("Run"),
            Act::Custom("run".into()),
            "se_run",
            Style::Primary,
            !busy,
            Some(&tr("Resumable: finished cells are kept in your exports folder.")),
            false,
            None,
        );
        m.begin_clip(cv, c.body);
        y += 44 - c.off;
        let xx = tags(m, cv, x, y, &["CONNECTOME", "GAME RULE", "MODEL PREDICTION"]);
        y = (y + 30).max(
            m.wrapped(
                cv,
                &tr("pathways (CONNECTOME); ranges and criteria (GAME RULE); every cell (MODEL PREDICTION)"),
                (xx + 4, y + 2),
                c.body.right() - 12 - xx,
                LABEL,
                &f.small,
                3,
            ) + 8,
        );
        let res = &q["sens"];
        if !res.is_object() {
            y = m.wrapped(
                cv,
                &format!("{} {}", tr("Not run yet. Parameters:"), PARAM_LABELS.join(", ")),
                (x, y),
                c.body.w - 24,
                LABEL,
                &f.small,
                4,
            ) + 10;
        } else {
            y = heatmap(m, cv, ctx, x, y, c.body.w - 30, res);
        }
        m.end_clip(cv);
        end(m, cv, rect, &c, key, y + c.off, &q);
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        match name {
            "mode" => self.smoke = !self.smoke,
            "run" => drop(ctx.host.command("lab.sens_start", json!({"smoke": self.smoke}))),
            "cancel" => drop(ctx.host.command("lab.day4_cancel", Value::Null)),
            _ => {}
        }
    }
}

fn heatmap(m: &mut Menu, cv: &mut Canvas, ctx: &mut Ctx, x: i32, mut y: i32, w: i32, res: &Value) -> i32 {
    let f = m.fonts();
    let (ok, bad, ok_t, bad_t) = palette(ctx.cfg.get_str("access.palette"));
    let cols: Vec<String> = res["tests"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(String::from)).collect();
    let label_of = |id: &str| {
        res["parameters"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|p| p["id"] == id)
            .and_then(|p| p["label"].as_str().map(String::from))
            .unwrap_or_else(|| id.to_string())
    };
    let lw = 200;
    let cw = 64.max(110.min((w - lw) / (cols.len() as i32).max(1)));
    if res["underpowered"].as_bool().unwrap_or(false) {
        m.text(
            cv,
            &tr("UNDERPOWERED: fewer than 7 seeds cannot reach p < 0.01, so a FAIL only means 'cannot pass at this n'."),
            (x, y),
            AMBER,
            &f.small,
            Anchor::TopLeft,
        );
        y += 22;
    }
    for (j, c) in cols.iter().enumerate() {
        let name: String = c.replace('_', " ").chars().take((cw / 7).max(0) as usize).collect();
        m.text(cv, &name, (x + lw + j as i32 * cw + 2, y), LABEL, &f.small, Anchor::TopLeft);
    }
    y += 20;
    for (ci, cell) in res["cells"].as_array().into_iter().flatten().enumerate() {
        let label = if cell["parameter"] == "baseline" {
            "baseline (defaults)".to_string()
        } else {
            format!("{} = {}", label_of(cell["parameter"].as_str().unwrap_or("")), fmt_g(num(&cell["value"])))
        };
        m.text(cv, &label, (x, y + 3), TEXT, &f.small, Anchor::TopLeft);
        for (j, c) in cols.iter().enumerate() {
            let r = &cell["behaviors"][c];
            let bx = Rect::new(x + lw + j as i32 * cw, y, cw - 2, 20);
            if !r.is_object() {
                cv.fill_rect(bx, [40, 44, 54], 0);
                continue;
            }
            let passed = r["passed"].as_bool().unwrap_or(false);
            cv.fill_rect(bx, if passed { ok } else { bad }, 0);
            m.text(
                cv,
                &format!("{} {}", if passed { "PASS" } else { "FAIL" }, fx(num(&r["effect"]), 2)),
                bx.center(),
                if passed { ok_t } else { bad_t },
                &f.small,
                Anchor::Center,
            );
            m.register_label(
                bx,
                &format!("se_cell:{ci}:{j}"),
                &format!(
                    "{c}: {}, {} {} (control {}), dz {}, p {}, n {}",
                    if passed { "PASS" } else { "FAIL" },
                    r["metric"].as_str().unwrap_or(""),
                    fx(num(&r["effect"]), 2),
                    fx(num(&r["control"]), 2),
                    fx(num(&r["effect_size_dz"]), 2),
                    fx(num(&r["p_value"]), 4),
                    r["n"]
                ),
            );
        }
        y += 22;
    }
    m.text(
        cv,
        &tr("Each cell: PASS or FAIL by the validation criteria, and the effect (drive ratio, or T-maze PI). Hover for the control, effect size and p."),
        (x, y + 6),
        LABEL,
        &f.small,
        Anchor::TopLeft,
    );
    y + 34
}
