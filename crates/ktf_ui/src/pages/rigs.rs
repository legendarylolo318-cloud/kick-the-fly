//! Lab > Behavior rigs (`lab/labrigs.py`): a hub and one scene per rig (tethered flight simulator, fly on a ball, Buridan's paradigm,
//! four-field olfactory arena). A scene's run is a background job on the game side; the result is a trace that the scene replays at real
//! time. The physics and the tags are `fly_sim::rigs`'.
//!
//! Host: query `lab.rigs_state` = `{results: {rig: result + "_id"}, job: {label, frac, note, elapsed} | null, error}`; commands
//! `lab.rig_run {rig, params}` and `lab.day4_cancel`. The conditions, the replay clock and the play state live in the page.

use std::collections::BTreeMap;

use fly_sim::rigs as fr;
use serde_json::{Value, json};

use super::day4::tags;
use crate::canvas::{Canvas, Rect};
use crate::i18n::{tr, tr_args};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const PAGES: [(&str, &str); 4] =
    [("tethered", "lab_rig_tethered"), ("ball", "lab_rig_ball"), ("buridan", "lab_rig_buridan"), ("fourfield", "lab_rig_fourfield")];

fn blurb(rig: &str) -> &'static str {
    match rig {
        "tethered" => {
            "The fly is held still. A striped panorama rotates around it; the fly's yaw is read from its steering descending neurons DNa01/DNa02 (right minus left). Open loop: the panorama turns whatever the fly does. Closed loop: the panorama is also turned by the fly's own yaw, so it can cancel the rotation. The line from the fly is the heading its yaw would give (read out only: the fly never turns)."
        }
        "ball" => {
            "A fly on a spherical treadmill in a virtual world driven by the ball: turning comes from DNa01/02, walking from DNp09. In the bar scene a distant bar is tracked by LC10; in the panorama scene the world drifts and the fly's turning feeds back. Open loop freezes the VR."
        }
        "buridan" => {
            "A round platform with two opposite stripes beyond its edge. The fly walks wherever its steering and walking neurons take it; the trajectory is recorded. The control has no stripes."
        }
        _ => {
            "A square arena in four quadrants, one odor in two opposite ones. The odor drives the olfactory neurons of the game's scent; the preference index is the time spent in odor against air. The control delivers no odor."
        }
    }
}

fn tagline(rig: &str) -> &'static str {
    match rig {
        "tethered" => {
            "CONNECTOME: yaw from DNa01/02; the rotation reaches them through T4/T5. GAME RULE: the EMD stage, the yaw gain, the panorama. MODEL PREDICTION: the response."
        }
        "ball" => {
            "CONNECTOME: turning and walking from DNa01/02 and DNp09. GAME RULE: the ball, the VR, the bar (LC10 tracking by the duel's rule). MODEL PREDICTION: the response."
        }
        "buridan" => {
            "CONNECTOME: steering and speed from DNa01/02 and DNp09. GAME RULE: platform, edge reflection, one stripe tracked at a time (LC10), the measures. MODEL PREDICTION: the trajectory."
        }
        _ => {
            "CONNECTOME: the odor's receptor neurons. GAME RULE: arena, sharp quadrants, the preference index. MODEL PREDICTION: whether the fly prefers the odor (nothing gives it a reason to)."
        }
    }
}

fn series(palette: &str) -> (Color, Color) {
    match palette {
        "blue-yellow" => ([30, 100, 200], [240, 190, 20]),
        "high-contrast" => ([255, 255, 255], [255, 200, 0]),
        _ => ([57, 135, 229], [217, 89, 38]),
    }
}

fn col(name: &str) -> usize {
    fr::COLS.iter().position(|c| *c == name).unwrap_or(0)
}

pub struct RigHub;

pub struct RigScene {
    rig: &'static str,
    page: &'static str,
    state: std::sync::Arc<std::sync::Mutex<Shared>>,
}

/// What the four scenes share: the conditions per rig, the replay clock and which run is playing.
struct Shared {
    p: BTreeMap<&'static str, Value>,
    t: f64,
    playing: bool,
    last: f64,
    record: bool,
    seen: BTreeMap<String, u64>,
}

impl Shared {
    fn new() -> Self {
        let p = fr::RIGS.iter().map(|r| (*r, fr::scene_defaults(r))).collect();
        Self { p, t: 0.0, playing: true, last: 0.0, record: false, seen: BTreeMap::new() }
    }
}

/// `(hub page, four scene pages)` sharing one state.
pub fn pages() -> (RigHub, Vec<(&'static str, RigScene)>) {
    let st = std::sync::Arc::new(std::sync::Mutex::new(Shared::new()));
    let scenes = PAGES.iter().map(|(rig, page)| (*page, RigScene { rig, page, state: st.clone() })).collect();
    (RigHub, scenes)
}

fn rows(res: &Value) -> Vec<[f64; 12]> {
    res["trace"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| {
            let mut a = [0.0; 12];
            for (i, v) in r.as_array().into_iter().flatten().take(12).enumerate() {
                a[i] = v.as_f64().unwrap_or(0.0);
            }
            a
        })
        .collect()
}

fn upto(a: &[[f64; 12]], t: f64) -> &[[f64; 12]] {
    let n = a.partition_point(|r| r[0] <= t).max(1);
    &a[..n.min(a.len())]
}

fn fly_shape(cv: &mut Canvas, pos: (f64, f64), heading: f64, size: f64, c: Color) {
    let (co, si) = (heading.cos(), heading.sin());
    let pts = [
        ((pos.0 + si * size) as f32, (pos.1 - co * size) as f32),
        ((pos.0 - co * size * 0.6 - si * size * 0.6) as f32, (pos.1 - si * size * 0.6 + co * size * 0.6) as f32),
        ((pos.0 + co * size * 0.6 - si * size * 0.6) as f32, (pos.1 + si * size * 0.6 + co * size * 0.6) as f32),
    ];
    cv.aa_poly(&pts, c);
}

fn poly_line(cv: &mut Canvas, pts: &[(f64, f64)], w: i32, c: Color) {
    for p in pts.windows(2) {
        cv.line((p[0].0.round() as i32, p[0].1.round() as i32), (p[1].0.round() as i32, p[1].1.round() as i32), w, c, 255);
    }
}

fn arena(bx: Rect, half: f64, top: i32) -> (impl Fn(f64, f64) -> (f64, f64), f64, i32, i32) {
    let s = f64::from(bx.w.min(bx.h - 2 * top)) / 2.0 - 8.0;
    let (cx, cy) = (bx.centerx(), bx.centery() + (top - 24) / 2);
    (move |x: f64, y: f64| (f64::from(cx) + x / half * s, f64::from(cy) - y / half * s), s, cx, cy)
}

fn empty(m: &Menu, cv: &mut Canvas, bx: Rect, text: &str) {
    let f = m.fonts();
    cv.fill_rect(bx, [14, 16, 22], 8);
    cv.stroke_rect(bx, [40, 46, 60], 8);
    m.wrapped(cv, text, (bx.x + 16, bx.centery() - 18), bx.w - 32, LABEL, &f.small, 4);
}

fn trace_chart(m: &Menu, cv: &mut Canvas, bx: Rect, res: &Value, t: f64, colname: &str, label: &str, colors: (Color, Color)) {
    let f = m.fonts();
    cv.fill_rect(bx, [14, 16, 22], 6);
    m.text(cv, label, (bx.x + 8, bx.y + 4), LABEL, &f.small, Anchor::TopLeft);
    let a = rows(res);
    if a.len() < 2 {
        return;
    }
    let top = f.small.linesize + 6;
    let plot = Rect::new(bx.x, bx.y + top, bx.w, bx.h - top);
    let i = col(colname);
    let y: Vec<f64> = a.iter().map(|r| r[i]).collect();
    let ymin = y.iter().copied().fold(f64::INFINITY, f64::min);
    let ymax = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo0 = ymin.min(0.0);
    let hi0 = ymax.max(0.0);
    let (lo, hi) = (lo0, if hi0 > lo0 { hi0 } else { lo0 + 1.0 });
    let pad = {
        let d = (hi - lo) * 0.1;
        if d == 0.0 { 1.0 } else { d }
    };
    let (lo, hi) = (lo - pad, hi + pad);
    let secs = res["seconds"].as_f64().unwrap_or(1.0).max(1e-9);
    let (pw, ph) = (f64::from(plot.w - 8), f64::from(plot.h - 8));
    let xs: Vec<f64> = a.iter().map(|r| f64::from(plot.x + 4) + r[0] / secs * pw).collect();
    let ys: Vec<f64> = y.iter().map(|v| f64::from(plot.bottom() - 4) - (v - lo) / (hi - lo) * ph).collect();
    let z = (f64::from(plot.bottom() - 4) - (0.0 - lo) / (hi - lo) * ph).round() as i32;
    cv.line((plot.x + 4, z), (plot.right() - 4, z), 1, [60, 66, 80], 255);
    let pts: Vec<(f64, f64)> = xs.into_iter().zip(ys).collect();
    poly_line(cv, &pts, 2, colors.0);
    let cx = (f64::from(plot.x + 4) + t / secs * pw).round() as i32;
    cv.line((cx, plot.y + 2), (cx, plot.bottom() - 2), 1, colors.1, 255);
}

fn draw_tethered(m: &Menu, cv: &mut Canvas, bx: Rect, res: &Value, t: f64, colors: (Color, Color)) {
    let f = m.fonts();
    cv.fill_rect(bx, [14, 16, 22], 8);
    let (cx, cy, r) = (bx.centerx(), bx.centery(), bx.w.min(bx.h) / 2 - 14);
    let all = rows(res);
    let a = upto(&all, t);
    let dt = res["dt"].as_f64().unwrap_or(0.02) * fr::TRACE_EVERY as f64;
    let slips: Vec<f64> = a.iter().map(|r| r[col("slip")]).collect();
    let pano = slips.iter().sum::<f64>() * dt;
    let n = 24;
    let (rr, ri) = (f64::from(r), f64::from(r - 16));
    for i in (0..n).step_by(2) {
        let a0 = pano + f64::from(i) * 2.0 * std::f64::consts::PI / f64::from(n);
        let a1 = a0 + std::f64::consts::PI / f64::from(n);
        let (cxf, cyf) = (f64::from(cx), f64::from(cy));
        let pts = [
            ((cxf + a0.sin() * ri) as f32, (cyf - a0.cos() * ri) as f32),
            ((cxf + a0.sin() * rr) as f32, (cyf - a0.cos() * rr) as f32),
            ((cxf + a1.sin() * rr) as f32, (cyf - a1.cos() * rr) as f32),
            ((cxf + a1.sin() * ri) as f32, (cyf - a1.cos() * ri) as f32),
        ];
        cv.aa_poly(&pts, [120, 128, 146]);
    }
    cv.aa_ring((cx as f32, cy as f32), (r - 16) as f32, [50, 56, 68]);
    let row = a.last().copied().unwrap_or([0.0; 12]);
    fly_shape(cv, (f64::from(cx), f64::from(cy)), 0.0, 26.0, colors.0);
    let psi = row[col("psi")];
    cv.line(
        (cx, cy),
        ((f64::from(cx) + psi.sin() * f64::from(r - 22)) as i32, (f64::from(cy) - psi.cos() * f64::from(r - 22)) as i32),
        2,
        colors.1,
        255,
    );
    m.text(
        cv,
        &tr_args("panorama {w:+.1f} rad/s", &[("w:+.1f", &format!("{:+.1}", row[col("omega_ext")]))]),
        (bx.x + 10, bx.y + 8),
        TEXT,
        &f.small,
        Anchor::TopLeft,
    );
    m.text(
        cv,
        &tr_args("slip {s:+.2f} rad/s", &[("s:+.2f", &format!("{:+.2}", row[col("slip")]))]),
        (bx.x + 10, bx.y + 8 + f.small.linesize),
        LABEL,
        &f.small,
        Anchor::TopLeft,
    );
    let (bxx, byy, bh) = (bx.right() - 70, bx.bottom() - 14, bx.h - 60);
    for (i, (name, c)) in [("R", colors.0), ("L", colors.1)].iter().enumerate() {
        let v = row[col(if *name == "R" { "turn_r" } else { "turn_l" })];
        let hh = ((v / 20.0).min(1.0) * f64::from(bh)) as i32;
        cv.fill_rect(Rect::new(bxx + i as i32 * 30, byy - hh, 22, hh), *c, 0);
        m.text(cv, name, (bxx + i as i32 * 30 + 11, byy + 2), TEXT, &f.small, Anchor::MidTop);
    }
    m.text(cv, &tr("DNa01/02 Hz"), (bxx - 6, bx.bottom() - 30 - bh - f.small.linesize), LABEL, &f.small, Anchor::TopLeft);
}

fn draw_ball(m: &Menu, cv: &mut Canvas, bx: Rect, res: &Value, t: f64, colors: (Color, Color)) {
    let f = m.fonts();
    cv.fill_rect(bx, [14, 16, 22], 8);
    let all = rows(res);
    let a = upto(&all, t);
    let p = &res["params"];
    let ext = 1.0f64.max(all.iter().map(|r| r[col("x")].abs().max(r[col("y")].abs())).fold(0.0, f64::max) * 1.15);
    let (to, s, cx, cy) = arena(bx, ext, f.small.linesize + 14);
    let sq = Rect::new(cx - s as i32, cy - s as i32, (2.0 * s) as i32, (2.0 * s) as i32);
    cv.fill_rect(sq, [22, 26, 34], 0);
    if a.len() > 1 {
        let pts: Vec<(f64, f64)> = a.iter().map(|r| to(r[col("x")], r[col("y")])).collect();
        poly_line(cv, &pts, 2, colors.0);
    }
    let row = a.last().copied().unwrap_or([0.0; 12]);
    fly_shape(cv, to(row[col("x")], row[col("y")]), row[col("psi")], 12.0, colors.1);
    if p["scene"].as_str() == Some("bar") {
        let b = p["bar_deg"].as_f64().unwrap_or(90.0).to_radians();
        let (bxx, byy) = (f64::from(cx) + b.sin() * (s + 8.0), f64::from(cy) - b.cos() * (s + 8.0));
        cv.circle((bxx as i32, byy as i32), 7, [230, 230, 240]);
        m.text(cv, &tr("bar"), (bxx as i32, byy as i32 - 16), LABEL, &f.small, Anchor::MidBottom);
    }
    m.text(cv, &tr("VR view from above: the fly walks, the ball turns under it"), (bx.x + 10, bx.y + 8), LABEL, &f.small, Anchor::TopLeft);
}

fn draw_buridan(m: &Menu, cv: &mut Canvas, bx: Rect, res: &Value, t: f64, colors: (Color, Color)) {
    let f = m.fonts();
    cv.fill_rect(bx, [14, 16, 22], 8);
    let all = rows(res);
    let a = upto(&all, t);
    let r = res["geometry"]["platform_radius"].as_f64().unwrap_or(0.5);
    let (to, s, cx, cy) = arena(bx, r / 0.62, f.small.linesize + 14);
    cv.aa_circle((cx as f32, cy as f32), (0.62 * s) as i32 as f32, [26, 30, 40]);
    cv.aa_ring((cx as f32, cy as f32), (0.62 * s) as i32 as f32, [70, 76, 92]);
    if res["params"]["mode"].as_str() == Some("stripes") {
        for sx in [-1.0f64, 1.0] {
            let x = f64::from(cx) + sx * (s + 8.0);
            cv.fill_rect(Rect::new((x - 5.0) as i32, (f64::from(cy) - s * 0.35) as i32, 10, (s * 0.7) as i32), [230, 230, 240], 0);
        }
    }
    if a.len() > 1 {
        let pts: Vec<(f64, f64)> = a.iter().map(|r| to(r[col("x")], r[col("y")])).collect();
        poly_line(cv, &pts, 2, colors.0);
    }
    let row = a.last().copied().unwrap_or([0.0; 12]);
    fly_shape(cv, to(row[col("x")], row[col("y")]), row[col("psi")], 10.0, colors.1);
    m.text(
        cv,
        &tr("walking trajectory; the stripes lie beyond the edge (not to scale)"),
        (bx.x + 10, bx.y + 8),
        LABEL,
        &f.small,
        Anchor::TopLeft,
    );
}

fn draw_fourfield(m: &Menu, cv: &mut Canvas, bx: Rect, res: &Value, t: f64, colors: (Color, Color)) {
    let f = m.fonts();
    cv.fill_rect(bx, [14, 16, 22], 8);
    let all = rows(res);
    let a = upto(&all, t);
    let half = res["geometry"]["field_half"].as_f64().unwrap_or(0.5);
    let (to, s, cx, cy) = arena(bx, half, f.small.linesize + 14);
    let sodor = res["params"]["mode"].as_str() == Some("odor");
    let si = s as i32;
    for (q, (qx, qy)) in [(1, 1), (1, -1), (-1, -1), (-1, 1)].iter().enumerate() {
        let odor = fr::ODOR_QUADRANTS.contains(&(q as i64)) && sodor;
        let c: Color = if odor { [colors.1[0] / 4 + 14, colors.1[1] / 4 + 14, colors.1[2] / 4 + 14] } else { [22, 26, 34] };
        cv.fill_rect(Rect::new(if *qx > 0 { cx } else { cx - si }, if *qy > 0 { cy - si } else { cy }, si, si), c, 0);
    }
    cv.stroke_rect(Rect::new(cx - si, cy - si, 2 * si, 2 * si), [70, 76, 92], 0);
    if a.len() > 1 {
        let pts: Vec<(f64, f64)> = a.iter().map(|r| to(r[col("x")], r[col("y")])).collect();
        poly_line(cv, &pts, 1, colors.0);
    }
    let row = a.last().copied().unwrap_or([0.0; 12]);
    fly_shape(cv, to(row[col("x")], row[col("y")]), row[col("psi")], 9.0, INK);
    let cap = if sodor { tr("odor quadrants tinted") } else { tr("sham: no odor anywhere (quadrants as in the odor run)") };
    m.text(cv, &cap, (bx.x + 10, bx.y + 8), LABEL, &f.small, Anchor::TopLeft);
}

fn chart_of(rig: &str) -> (&'static str, &'static str) {
    match rig {
        "tethered" => ("steer_hz", "DNa01/02 R - L (Hz)"),
        "ball" | "buridan" => ("yaw_rate", "turning (rad/s)"),
        _ => ("speed", "speed (m/s)"),
    }
}

impl Page for RigHub {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.rigs_state");
        m.text(cv, &tr("BEHAVIOR RIGS"), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let mut y = m.wrapped(
            cv,
            &tr("Four classic ways to watch a fly, each its own scene. The fly is a real brain; what it steers and walks with are its descending neurons. A result the model misses is shown as a miss."),
            (rect.x + 24, m.under_heading(rect).max(rect.y + 50)),
            rect.w - 48,
            LABEL,
            &f.small,
            4,
        ) + 8;
        let bh = 56.max(f.text.linesize * 2 + 24);
        let body = Rect::new(rect.x + 16, y, rect.w - 32, rect.h - (y - rect.y) - 80);
        let key = "lab_rigs";
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let mut yy = body.y - off;
        for (rig, page) in PAGES {
            let r = Rect::new(body.x, yy, body.w, bh);
            m.button(
                cv,
                r,
                &tr(fr::rig_title(rig)),
                Act::Custom(format!("show:{page}")),
                &format!("rigs:{rig}"),
                Style::Normal,
                true,
                Some(&tr(blurb(rig))),
                false,
                None,
            );
            if q["results"].get(rig).is_some_and(Value::is_object) {
                m.text(cv, &tr("a run is ready to replay"), (r.right() - 14, r.centery()), LABEL, &f.small, Anchor::MidRight);
            }
            yy += bh + 12;
        }
        y = m.wrapped(
            cv,
            &tr("Headless: --rig NAME (one fly, recorded in the Lab's format), --rig-assay NAME (the pre-registered assay and its off control), and the rig_*.yaml protocols. docs/rigs.md has the criteria and the results."),
            (body.x + 4, yy + 4),
            body.w - 8,
            LABEL,
            &f.small,
            5,
        );
        m.set_content_h(key, (y + off - body.bottom() + 8).max(0));
        m.end_clip(cv);
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            &tr("Back"),
            Act::Back,
            "rigs:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, m: &mut Menu, name: &str, _text: Option<&str>, _ctx: &mut Ctx) {
        if let Some(p) = name.strip_prefix("show:") {
            m.show(p);
        }
    }
}

impl RigScene {
    /// The conditions, the replay clock and which result (`_id`) is playing (what the reference renders set).
    pub fn set_view(&mut self, params: Value, t: f64, playing: bool, result_id: u64) {
        let mut st = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        st.p.insert(self.rig, params);
        st.t = t;
        st.playing = playing;
        st.seen.insert(self.rig.to_string(), result_id);
    }
}

impl Page for RigScene {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[allow(clippy::too_many_lines)]
    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let rig = self.rig;
        let q = ctx.host.query("lab.rigs_state");
        let res = q["results"].get(rig).filter(|r| r.is_object());
        let busy = q["job"].is_object();
        let colors = series(ctx.cfg.get_str("access.palette"));
        let mut st = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // a new result starts its replay; the clock otherwise follows the page's own time
        if let Some(r) = res {
            let id = r["_id"].as_u64().unwrap_or(0);
            if st.seen.get(rig) != Some(&id) {
                st.seen.insert(rig.to_string(), id);
                st.t = 0.0;
                st.playing = true;
                st.last = ctx.now;
            }
        }
        if let (Some(r), true) = (res, st.playing) {
            let secs = r["seconds"].as_f64().unwrap_or(0.0);
            st.t = secs.min(st.t + (ctx.now - st.last).max(0.0));
            if st.t >= secs {
                st.playing = false;
            }
        }
        st.last = ctx.now;
        let (t, playing, record) = (st.t, st.playing, st.record);
        let key = self.page;
        m.text(cv, &tr(fr::rig_title(rig)), (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let body = Rect::new(rect.x + 16, rect.y + 56, rect.w - 32, rect.h - 56 - 80);
        let off = m.scroll_offset(key);
        m.begin_clip(cv, body);
        let (x, w) = (body.x + 8, body.w - 40);
        let mut y = body.y - off;
        y = m.wrapped(cv, &tr(blurb(rig)), (x, y), w, LABEL, &f.small, 8) + 6;
        let wide = w >= 940;
        let cw = if wide { ((f64::from(w) * 0.56) as i32).min(640) } else { w.min(640) };
        let canvas = Rect::new(x, y, cw, (cw * 3 / 4).min(300.max(body.h - 60)) + if wide { 0 } else { 20 });
        match res {
            None => empty(m, cv, canvas, &tr("Nothing run yet. Choose a condition and press Run; the run is replayed here at real time.")),
            Some(r) => match rig {
                "tethered" => draw_tethered(m, cv, canvas, r, t, colors),
                "ball" => draw_ball(m, cv, canvas, r, t, colors),
                "buridan" => draw_buridan(m, cv, canvas, r, t, colors),
                _ => draw_fourfield(m, cv, canvas, r, t, colors),
            },
        }
        let mut yl = canvas.bottom() + 6;
        if let Some(r) = res {
            let (c, label) = chart_of(rig);
            let chart = Rect::new(x, yl, canvas.w, 80.max(f.small.linesize * 5));
            trace_chart(m, cv, chart, r, t, c, &tr(label), colors);
            yl = chart.bottom() + 8;
        }
        let yc = if wide {
            let yr = controls(m, cv, x + cw + 24, y, w - cw - 24, rig, &mut st, busy, res.is_some(), playing, record);
            yl.max(yr)
        } else {
            controls(m, cv, x, yl, w, rig, &mut st, busy, res.is_some(), playing, record)
        };
        y = yc;
        if let Some(r) = res {
            m.text(
                cv,
                &format!(
                    "{} (seed {}, individuality {})",
                    tr("What this run measured"),
                    r["seed"],
                    r["individuality"].as_str().unwrap_or("")
                ),
                (x, y),
                AMBER,
                &f.bold,
                Anchor::TopLeft,
            );
            y += f.bold.linesize + 2;
            for (k, v) in r["summary"].as_object().into_iter().flatten() {
                if v.is_null() {
                    continue;
                }
                let name = k.replace('_', " ");
                let txt = match v {
                    Value::Number(n) if n.is_f64() => format!("{name}: {}", fly_sim::sensitivity::fmt_gp(n.as_f64().unwrap_or(0.0), 3)),
                    other => format!("{name}: {}", other.to_string().trim_matches('"')),
                };
                y = m.wrapped(cv, &txt, (x + 8, y), w - 16, TEXT, &f.small, 2);
            }
            y += 6;
        }
        tags(m, cv, x, y, &["CONNECTOME", "GAME RULE", "MODEL PREDICTION"]);
        y += f.small.linesize + 10;
        y = m.wrapped(cv, &tr(tagline(rig)), (x, y), w, LABEL, &f.small, 6) + 4;
        y = m.wrapped(
            cv,
            &tr_args(
                "One run is one fly: it shows what the model does, not a result. The pre-registered assay over ten flies and its off control is --headless --rig-assay {name}; docs/rigs.md has its result.",
                &[("name", rig)],
            ),
            (x, y),
            w,
            LABEL,
            &f.small,
            5,
        ) + 8;
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

    fn custom(&mut self, _m: &mut Menu, name: &str, text: Option<&str>, ctx: &mut Ctx) {
        let rig = self.rig;
        let mut st = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let val = text.and_then(|t| t.parse::<f64>().ok());
        match name {
            "cancel" => drop(ctx.host.command("lab.day4_cancel", Value::Null)),
            "run" => {
                let params = st.p[rig].clone();
                drop(ctx.host.command("lab.rig_run", json!({"rig": rig, "params": params, "record": st.record})));
            }
            "replay" => {
                st.t = 0.0;
                st.playing = true;
            }
            "pause" => st.playing = !st.playing,
            "rec" => st.record = !st.record,
            n if n.starts_with("mode:") => {
                if let Some(i) = n[5..].parse::<usize>().ok().filter(|i| *i < 2) {
                    let modes = fr::scene_modes(rig);
                    st.p.get_mut(rig).expect("rig")["mode"] = json!(modes[i]);
                }
            }
            n if n.starts_with("scene:") => st.p.get_mut(rig).expect("rig")["scene"] = json!(&n[6..]),
            n if n.starts_with("bar:") => st.p.get_mut(rig).expect("rig")["bar_deg"] = json!(n[4..].parse::<f64>().unwrap_or(90.0)),
            n if n.starts_with("slide:rig_sl_") => {
                if let Some(v) = val {
                    let k = n["slide:rig_sl_".len()..].to_string();
                    st.p.get_mut(rig).expect("rig")[k.as_str()] = json!(v);
                }
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn controls(
    m: &mut Menu,
    cv: &mut Canvas,
    x: i32,
    mut y: i32,
    w: i32,
    rig: &str,
    st: &mut Shared,
    busy: bool,
    has_res: bool,
    playing: bool,
    record: bool,
) -> i32 {
    let f = m.fonts();
    let h = 34.max(f.text.linesize + 14);
    let p = st.p[rig].clone();
    let modes = fr::scene_modes(rig);
    let label = |k: &str| match k {
        "open" => tr("Open loop"),
        "closed" => tr("Closed loop"),
        "stripes" => tr("Stripes"),
        "none" => tr("No stripes"),
        "odor" => tr("Odor"),
        _ => tr("Sham (no odor)"),
    };
    let cur = modes.iter().position(|k| Some(*k) == p["mode"].as_str()).unwrap_or(0);
    m.segmented(
        cv,
        Rect::new(x, y, w.min(200 * 2), h),
        &[label(modes[0]), label(modes[1])],
        cur,
        &[Act::Custom("mode:0".into()), Act::Custom("mode:1".into())],
        &format!("rig_mode:{rig}"),
        !busy,
        &tr("The rig's condition; the other one is its control."),
    );
    y += h + 8;
    let scene = p["scene"].as_str().unwrap_or("bar").to_string();
    if rig == "ball" {
        m.segmented(
            cv,
            Rect::new(x, y, w.min(400), h),
            &[tr("Bar"), tr("Panorama")],
            usize::from(scene != "bar"),
            &[Act::Custom("scene:bar".into()), Act::Custom("scene:panorama".into())],
            &format!("rig_scene:{rig}"),
            !busy,
            &tr("A distant bar to fixate, or a striped panorama that drifts."),
        );
        y += h + 8;
        if scene == "bar" {
            m.segmented(
                cv,
                Rect::new(x, y, w.min(400), h),
                &[tr("Bar on the right"), tr("Bar on the left")],
                usize::from(p["bar_deg"].as_f64().unwrap_or(90.0) <= 0.0),
                &[Act::Custom("bar:90".into()), Act::Custom("bar:-90".into())],
                &format!("rig_bar:{rig}"),
                !busy,
                "",
            );
            y += h + 8;
        }
    }
    type Slider = (&'static str, String, f64, f64, f64, &'static str);
    let sliders: Vec<Slider> = match rig {
        "tethered" => vec![
            ("omega", tr("Rotation (rad/s)"), 0.25, 5.0, 0.25, "{:.2f}"),
            ("gain", tr("Loop gain"), 0.0, 2.0, 0.25, "{:.2f}"),
            ("seconds", tr("Seconds"), 9.0, 30.0, 1.0, "{:.0f}"),
        ],
        "ball" => {
            vec![("omega", tr("Panorama drift (rad/s)"), -3.0, 3.0, 0.25, "{:+.2f}"), ("seconds", tr("Seconds"), 10.0, 60.0, 5.0, "{:.0f}")]
        }
        _ => vec![("seconds", tr("Seconds"), 20.0, 180.0, 10.0, "{:.0f}")],
    };
    for (key, lbl, lo, hi, step, fmt) in sliders {
        if rig == "tethered" && key == "gain" && p["mode"].as_str() != Some("closed") {
            continue;
        }
        if rig == "ball" && key == "omega" && scene != "panorama" {
            continue;
        }
        m.text(cv, &lbl, (x, y + h / 2), TEXT, &f.small, Anchor::MidLeft);
        let lw = (w / 3).min(220);
        m.page_slider(
            cv,
            Rect::new(x + lw, y, 160.max(w - lw), h),
            p[key].as_f64().unwrap_or(0.0),
            (lo, hi, step),
            fmt,
            &format!("rig_sl_{key}"),
            !busy,
            None,
        );
        y += h + 6;
    }
    m.toggle(
        cv,
        Rect::new(x, y, 120, h),
        record,
        Act::Custom("rec".into()),
        &format!("rig_rec:{rig}"),
        false,
        &tr("Also write the run in the Lab's format (spikes, rates, group rates, kinematics, metadata, trace CSV) into your exports folder. Not in this build yet: the Lab export is not ported."),
    );
    m.text(cv, &tr("Record to the exports folder"), (x + 130, y + h / 2), TEXT, &f.small, Anchor::MidLeft);
    y += h + 8;
    m.button(
        cv,
        Rect::new(x, y, 150, h),
        &tr("Run"),
        Act::Custom("run".into()),
        &format!("rig_run:{rig}"),
        Style::Primary,
        !busy,
        Some(&tr("Builds one fly (the Settings random seed and individuality) and runs the rig. Takes about a third of the simulated time in real time, plus a few seconds to build.")),
        false,
        None,
    );
    m.button(
        cv,
        Rect::new(x + 160, y, 150, h),
        &tr("Replay"),
        Act::Custom("replay".into()),
        &format!("rig_replay:{rig}"),
        Style::Normal,
        has_res && !busy,
        None,
        false,
        None,
    );
    m.button(
        cv,
        Rect::new(x + 320, y, 150, h),
        &if playing { tr("Pause") } else { tr("Play") },
        Act::Custom("pause".into()),
        &format!("rig_pause:{rig}"),
        Style::Normal,
        has_res && !busy,
        None,
        false,
        None,
    );
    y + h + 10
}
