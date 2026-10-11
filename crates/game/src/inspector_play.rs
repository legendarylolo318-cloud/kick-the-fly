//! The neuron inspector of the big brain view (`Game._inspect_at`, `_neuron_info`, `_draw_inspect`): click a neuron to see its type,
//! population, transmitter, firing against its calm rate and its strongest partners (drawn as lines to where they sit), and switch its whole
//! cell type off or on. Data from `fly_sim::inspect`; the card is a recorded [`ktf_ui::hud::Op`] list.
//!
//! Python drew the big view as a flat picture and picked from per-pixel neuron ids; this build's big view is the 3D cloud, so picking
//! projects every neuron the way the brain shader does and takes the nearest (then the busiest) within 8 px.
//! The search box (type, body ID or row number) and the PATH FROM/TO HERE tracer are `fly_sim::neurosearch` (equal to `lab/neurosearch.py`).
//! Not ported: the Lab's FLIP SIGN and PATCH CLAMP buttons.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use fly_sim::inspect::NeuronInfo;
use ktf_ui::Rect;
use ktf_ui::hud::{FontSel, Op};

use crate::app::{LatestSnapshot, SimEvents, SimLink};
use crate::brainview::BrainView;
use crate::hud_canvas::HudLayout;
use crate::menu_ui::{MenuOpen, UserConfig};
use crate::player::MouseLook;
use crate::sim_thread::{Command, SimEvent};
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use fly_sim::neurosearch::Path;

const TEXT: [u8; 3] = [205, 212, 224];
const LABEL: [u8; 3] = [120, 132, 150];
const INK: [u8; 3] = [240, 243, 248];
const DIM: [u8; 3] = [70, 78, 92];

/// What a click on the big view's drawing does.
#[derive(Debug, Clone, PartialEq)]
enum Hit {
    /// SILENCE TYPE / STIMULATE / CLEAR on the card
    Mode(i8),
    SearchBox,
    Result(u32),
    PathEnd(bool),
    PathClear,
}

#[derive(Resource, Default)]
pub struct Inspector {
    pub info: Option<NeuronInfo>,
    pub ops: Vec<Op>,
    hits: Vec<(Rect, Hit)>,
    calm: Vec<f32>,
    press: Option<Vec2>,
    /// The search box has the keyboard (the hotkeys wait meanwhile).
    pub typing: bool,
    text: String,
    results: Vec<u32>,
    /// Path tracer: (from, to) and the strongest paths between them.
    ends: (Option<u32>, Option<u32>),
    paths: Vec<Path>,
    labels: Option<std::sync::Arc<fly_sim::pack::Labels>>,
}

/// The brain shader's projection (`brain.wgsl`, camera at z 3.4, 32 degrees): a layout point to window fractions (0..1, y down).
pub fn project(view: &BrainView, p: [f32; 3], aspect: f32) -> Option<(f32, f32)> {
    let (sn, c) = view.yaw.sin_cos();
    let q = Vec3::new(p[0] * c + p[2] * sn, p[1], -p[0] * sn + p[2] * c) * view.zoom;
    let d = 3.4 - q.z;
    if d <= 0.05 {
        return None;
    }
    let t = (16f32).to_radians().tan();
    let (nx, ny) = (q.x / (d * t * aspect), q.y / (d * t));
    Some(((nx + 1.0) / 2.0, (1.0 - ny) / 2.0))
}

#[allow(clippy::too_many_arguments)]
fn update(
    mut ins: ResMut<Inspector>,
    snap: Res<LatestSnapshot>,
    events: Res<SimEvents>,
    view: Res<BrainView>,
    window: Single<&Window, With<PrimaryWindow>>,
    lay: Res<HudLayout>,
    buttons: Res<ButtonInput<MouseButton>>,
    look: Res<MouseLook>,
    menu: Res<MenuOpen>,
    link: Res<SimLink>,
    cfg: Res<UserConfig>,
    time: Res<Time<Real>>,
    shapes: Res<crate::shapes_play::Shapes>,
    mut was_opted: Local<bool>,
    mut notes: ResMut<crate::feedback::Notes>,
) {
    let opted_now = cfg.0.get_bool("brain.neuron_shapes");
    if opted_now && !*was_opted {
        shapes.reset_offline(); // the player just turned the download on: ask again for what was refused
    }
    *was_opted = opted_now;
    let Some(s) = snap.0.as_deref() else { return };
    let ins = &mut *ins;
    if ins.labels.is_none() {
        ins.labels = link.0.shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref().map(|sh| sh.labels.clone());
    }
    // the calm rate per neuron (`BrainView.calm`: a slow EMA)
    if ins.calm.len() != s.rates.len() {
        ins.calm = vec![0.025; s.rates.len()];
    }
    for (c, &r) in ins.calm.iter_mut().zip(&s.rates) {
        *c += (r - *c) * 0.01;
    }
    for e in &events.0 {
        if let SimEvent::Neuron(n) = e {
            ins.info = Some((**n).clone());
        }
    }
    if !view.big || menu.0 {
        ins.ops.clear();
        ins.hits.clear();
        return;
    }
    let Some(info) = s.info.as_deref() else { return };
    let (lw, lh) = (window.width(), window.height());
    let sf = window.scale_factor();
    let k = sf / lay.0.scale;
    let aspect = lw / lh.max(1.0);
    // a click (press and release in the same place) on the cloud picks; on the card's buttons it acts
    if !look.captured {
        if buttons.just_pressed(MouseButton::Left) {
            ins.press = window.cursor_position();
        }
        if buttons.just_released(MouseButton::Left)
            && let (Some(p0), Some(p1)) = (ins.press.take(), window.cursor_position())
            && p0.distance(p1) < 5.0
        {
            let hud = ((p1.x * k) as i32, (p1.y * k) as i32);
            let hit = ins.hits.iter().find(|(r, _)| r.contains(hud)).map(|(_, h)| h.clone());
            let was_typing = ins.typing;
            ins.typing = false;
            match hit {
                Some(Hit::Mode(mode)) => {
                    if let Some(n) = ins.info.as_ref() {
                        link.0.send(Command::SurgeryType { ty: n.ty.clone(), mode });
                    }
                }
                Some(Hit::SearchBox) => ins.typing = true,
                Some(Hit::Result(i)) => {
                    link.0.send(Command::Inspect(i));
                }
                Some(Hit::PathEnd(from)) => {
                    if let Some(i) = ins.info.as_ref().map(|n| n.i) {
                        if from {
                            ins.ends.0 = Some(i);
                        } else {
                            ins.ends.1 = Some(i);
                        }
                        ins.paths.clear();
                        if let (Some(a), Some(b)) = (ins.ends.0, ins.ends.1)
                            && let Some(sh) = link.0.shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
                        {
                            ins.paths = fly_sim::neurosearch::top_paths(&sh.weights, a, b, 5, 3);
                            let name = |i: u32| {
                                sh.labels.types.get(i as usize).filter(|t| !t.is_empty()).cloned().unwrap_or_else(|| format!("#{i}"))
                            };
                            notes.push(
                                time.elapsed_secs(),
                                format!("PATHS    {} -> {}: {} found (<= 3 synapses)", name(a), name(b), ins.paths.len()),
                                crate::feedback::Tag::Connectome,
                            );
                        }
                    }
                }
                Some(Hit::PathClear) => {
                    ins.ends = (None, None);
                    ins.paths.clear();
                }
                None if was_typing => {} // a click outside only closes the result list
                None if ins.info.as_ref().is_some_and(|_| card_rect(lay.0.play_w).contains(hud)) => {
                    // the card swallows the click
                }
                None => {
                let mut best: Option<(f32, usize)> = None;
                for (i, p) in info.layout.iter().enumerate() {
                    if !p[0].is_finite() {
                        continue;
                    }
                    let Some((fx, fy)) = project(&view, *p, aspect) else { continue };
                    let d = Vec2::new(fx * lw - p1.x, fy * lh - p1.y).length();
                    if d <= 8.0 {
                        let score = s.rates.get(i).copied().unwrap_or(0.0) / 0.025 * 0.5 - d;
                        if best.is_none_or(|b| score > b.0) {
                            best = Some((score, i));
                        }
                    }
                }
                match best {
                    Some((_, i)) => {
                        link.0.send(Command::Inspect(i as u32));
                    }
                    None => ins.info = None,
                }
                }
            }
        }
    }
    // ---- the drawing ----
    let now = time.elapsed_secs();
    let to_hud = |fx: f32, fy: f32| ((fx * lw * k) as i32, (fy * lh * k) as i32);
    let at = |j: u32| -> Option<(i32, i32)> {
        let p = info.layout.get(j as usize)?;
        if !p[0].is_finite() {
            return None;
        }
        project(&view, *p, aspect).map(|(fx, fy)| to_hud(fx, fy))
    };
    let mut ops: Vec<Op> = vec![];
    let mut hits: Vec<(Rect, Hit)> = vec![];
    // the traced paths sit under everything else, the search box and its results on top
    draw_paths(&mut ops, &mut hits, ins, s, info, &at, now, (lw * k) as i32, (lh * k) as i32);
    draw_search(&mut ops, &mut hits, ins, s, info, now);
    let Some(n) = ins.info.clone() else {
        ins.ops = ops;
        ins.hits = hits;
        return;
    };
    let me = at(n.i);
    for (list, col) in [(&n.ins, [110u8, 200, 255]), (&n.outs, [255u8, 160, 80])] {
        for (j, _) in list {
            if let (Some(a), Some(b)) = (me, at(*j)) {
                ops.push(Op::Line { a, b, w: 1, color: col });
                ops.push(Op::Circle { c: b, r: 3, color: col });
            }
        }
    }
    if let Some(m) = me {
        let r = (8.0 + 3.0 * (now * 8.0).sin()).max(1.0);
        ops.push(Op::Ring { c: (m.0 as f32, m.1 as f32), r, color: [255, 255, 255] });
        ops.push(Op::Ring { c: (m.0 as f32, m.1 as f32), r: r + 4.0, color: [120, 120, 120] });
    }
    let card = card_rect(lay.0.play_w);
    ops.push(Op::Card { rect: card, fill: [8, 10, 16], alpha: 225, radius: 10 });
    let (x, y) = (card.x + 12, card.y + 10);
    let text = |ops: &mut Vec<Op>, s: &str, pos: (i32, i32), color: [u8; 3], font: FontSel| {
        ops.push(Op::Text { s: s.into(), pos, color, font, anchor: "topleft" });
    };
    let rate = s.rates.get(n.i as usize).copied().unwrap_or(0.0) / 0.005;
    let calm = ins.calm.get(n.i as usize).copied().unwrap_or(0.025) / 0.005;
    text(&mut ops, &n.ty, (x, y), INK, FontSel::Head);
    let inst = if n.instance.is_empty() { String::new() } else { format!("   {}", n.instance) };
    text(&mut ops, &format!("{}{inst}", n.pop), (x, y + 26), LABEL, FontSel::Small);
    text(
        &mut ops,
        &format!("firing {rate:5.1} spikes/s   (calm {calm:4.1})"),
        (x, y + 44),
        if rate > calm + 2.0 { [255, 170, 90] } else { TEXT },
        FontSel::Small,
    );
    text(&mut ops, &format!("{} input partners, {} output partners", n.n_in, n.n_out), (x, y + 60), LABEL, FontSel::Small);
    let mut yy = y + 76;
    if !n.nt.is_empty() {
        let sign = match n.nt.as_str() {
            "acetylcholine" => "excitatory (+)",
            "gaba" | "glutamate" | "histamine" => "inhibitory (-)",
            _ => "no sign in this model (0)",
        };
        let how = match n.nt_source.as_str() {
            "ground_truth" => "measured",
            "consensus_nt" => "predicted (consensus)",
            "predicted_nt" => "predicted (per body)",
            "" => "unknown",
            other => other,
        };
        let conf = if n.nt_conf.is_nan() { "no confidence given".to_string() } else { format!("{:.0}% confident", n.nt_conf * 100.0) };
        text(&mut ops, &format!("{}: {sign}", n.nt), (x, yy), TEXT, FontSel::Small);
        text(&mut ops, &format!("{how}, {conf}"), (x, yy + 14), LABEL, FontSel::Small);
        yy += 34;
    }
    yy += 2;
    let opted = cfg.0.get_bool("brain.neuron_shapes");
    let state = if n.body_id > 0 {
        shapes.request(n.body_id, opted);
        shapes.state(n.body_id)
    } else {
        None
    };
    let shape =
        if n.body_id <= 0 { "estimated fiber (no body id)".to_string() } else { crate::shapes_play::status_line(state.as_ref(), opted) };
    text(&mut ops, &shape, (x, yy), if shape.starts_with("real shape:") { [255, 236, 140] } else { DIM }, FontSel::Small);
    if let Some(skel) = crate::shapes_play::skeleton_of(state.as_ref()) {
        // the inset: the whole skeleton fitted to a 236 x 118 box with its credit (`shape_draw.thumbnail`)
        let boxr = Rect::new(10, lh_hud(lh, k) - 150, 244, 140);
        ops.push(Op::Card { rect: boxr, fill: [8, 10, 16], alpha: 215, radius: 8 });
        thumbnail(&mut ops, &skel, Rect::new(boxr.x + 4, boxr.y + 4, 236, 118));
        ops.push(Op::Text {
            s: ktf_data::shapes::CREDIT_SHORT.into(),
            pos: (boxr.x + 6, boxr.bottom() - 16),
            color: DIM,
            font: FontSel::Small,
            anchor: "topleft",
        });
    }
    yy += 14;
    for (title, list, col) in [
        ("strongest inputs (% of its input)", &n.ins, [110u8, 200, 255]),
        ("strongest outputs (% of target's input)", &n.outs, [255u8, 160, 80]),
    ] {
        text(&mut ops, title, (x, yy), col, FontSel::Small);
        yy += 15;
        for (j, v) in list.iter().take(4) {
            let name = info.types.get(*j as usize).filter(|t| !t.is_empty()).map_or("untyped", String::as_str);
            let inst = s.info.as_deref().map(|_| "").unwrap_or("");
            let _ = inst;
            let sign = if *v > 0.0 { "+" } else { "-" };
            text(&mut ops, &format!("{sign}{:4.1}%  {name}", v.abs() * 100.0), (x + 6, yy), TEXT, FontSel::Small);
            yy += 14;
        }
        yy += 4;
    }
    let current = s.surgery_types.iter().find(|t| t.0 == n.ty).map_or(0, |t| t.1);
    for (k, (label, mode)) in [("SILENCE TYPE", -1i8), ("STIMULATE", 1), ("CLEAR", 0)].into_iter().enumerate() {
        let r = Rect::new(card.x + 12 + k as i32 * 98, card.bottom() - 62, 90, 24);
        let active = current == mode && mode != 0;
        let fill = if active && mode < 0 {
            [60, 110, 200]
        } else if active {
            [230, 130, 50]
        } else {
            [40, 46, 58]
        };
        ops.push(Op::Card { rect: r, fill, alpha: 255, radius: 6 });
        ops.push(Op::Text { s: label.into(), pos: r.center(), color: INK, font: FontSel::Small, anchor: "center" });
        hits.push((r, Hit::Mode(mode)));
    }
    // path tracer: this neuron as the start or the end of a path
    for (k, (label, from)) in [("PATH FROM HERE", true), ("PATH TO HERE", false)].into_iter().enumerate() {
        let r = Rect::new(card.x + k as i32 * 156, card.bottom() + 6, 150, 24);
        let on = if from { ins.ends.0 } else { ins.ends.1 } == Some(n.i);
        ops.push(Op::Card { rect: r, fill: if on { [40, 120, 90] } else { [40, 46, 58] }, alpha: 255, radius: 6 });
        ops.push(Op::Text { s: label.into(), pos: r.center(), color: INK, font: FontSel::Small, anchor: "center" });
        hits.push((r, Hit::PathEnd(from)));
    }
    ins.ops = ops;
    ins.hits = hits;
}

const PATH_COLORS: [[u8; 3]; 5] = [[120, 255, 170], [110, 200, 255], [255, 210, 110], [230, 140, 255], [255, 140, 140]];

/// `Game._draw_search`: the box at the top left of the big view and, while it has the keyboard, the results with their firing rates.
fn draw_search(ops: &mut Vec<Op>, hits: &mut Vec<(Rect, Hit)>, ins: &Inspector, s: &crate::snapshot::Snapshot, info: &crate::snapshot::BrainInfo, _now: f32) {
    let box_ = Rect::new(8, 58 + 8, 250, 24);
    ops.push(Op::Card { rect: box_, fill: [18, 24, 36], alpha: 255, radius: 5 });
    ops.push(Op::Stroke { rect: box_, color: if ins.typing { [86, 214, 255] } else { [60, 68, 84] }, width: 1, radius: 5 });
    let caret = if ins.typing && (_now * 2.0) as i64 % 2 == 0 { "|" } else { "" };
    let (label, col) = if !ins.text.is_empty() || ins.typing {
        (format!("{}{caret}", ins.text), INK)
    } else {
        ("Search neurons (type, body ID)".to_string(), LABEL)
    };
    ops.push(Op::Text { s: label, pos: (box_.x + 8, box_.y + 4), color: col, font: FontSel::Small, anchor: "topleft" });
    hits.push((box_, Hit::SearchBox));
    if !ins.typing {
        return;
    }
    for (k, &i) in ins.results.iter().enumerate() {
        let r = Rect::new(box_.x, box_.bottom() + 2 + k as i32 * 20, 330, 20);
        ops.push(Op::Card { rect: r, fill: if k % 2 == 1 { [14, 18, 28] } else { [20, 26, 38] }, alpha: 255, radius: 0 });
        let ty = info.types.get(i as usize).filter(|t| !t.is_empty()).map_or("untyped", String::as_str);
        let inst = ins.labels.as_ref().and_then(|l| l.instance.get(i as usize)).map_or("", String::as_str);
        let bid = ins.labels.as_ref().and_then(|l| l.body_id.get(i as usize)).map_or(String::new(), |b| format!("  {b}"));
        let line: String = format!("{ty}  {inst}{bid}").chars().take(48).collect();
        ops.push(Op::Text { s: line, pos: (r.x + 6, r.y + 2), color: TEXT, font: FontSel::Small, anchor: "topleft" });
        let hz = s.rates.get(i as usize).copied().unwrap_or(0.0) / 0.005;
        ops.push(Op::Text { s: format!("{hz:4.0} Hz"), pos: (r.right() - 6, r.y + 2), color: [140, 220, 180], font: FontSel::Small, anchor: "topright" });
        hits.push((r, Hit::Result(i)));
    }
}

/// `Game._draw_paths`: the traced paths over the brain (brighter for stronger ones, a dot runs along a synapse while its source fires above
/// calm), the FROM/TO labels, and the list at the bottom left with a CLEAR button.
#[allow(clippy::too_many_arguments)]
fn draw_paths(
    ops: &mut Vec<Op>,
    hits: &mut Vec<(Rect, Hit)>,
    ins: &Inspector,
    s: &crate::snapshot::Snapshot,
    info: &crate::snapshot::BrainInfo,
    at: &dyn Fn(u32) -> Option<(i32, i32)>,
    now: f32,
    w: i32,
    h: i32,
) {
    let (from, to) = ins.ends;
    if from.is_none() && to.is_none() {
        return;
    }
    let top = ins.paths.iter().map(|p| p.strength).fold(0.0f64, f64::max).max(f64::MIN_POSITIVE);
    let act = |j: u32| s.rates.get(j as usize).copied().unwrap_or(0.0) / ins.calm.get(j as usize).copied().unwrap_or(0.025).max(1e-4);
    for (rank, p) in ins.paths.iter().enumerate().rev() {
        let col = PATH_COLORS[rank % 5];
        let fade = 0.35 + 0.65 * (p.strength / top).powf(0.25);
        let c = col.map(|v| (f64::from(v) * fade) as u8);
        let pts: Vec<Option<(i32, i32)>> = p.nodes.iter().map(|&j| at(j)).collect();
        for ((pair, &pre), &wgt) in pts.windows(2).zip(&p.nodes).zip(&p.weights) {
            let (Some(a), Some(b)) = (pair[0], pair[1]) else { continue };
            ops.push(Op::Line { a, b, w: if rank == 0 { 3 } else { 2 }, color: c });
            if act(pre) > 1.2 {
                for f in [(now * 1.6 + rank as f32 * 0.2) % 1.0, (now * 1.6 + rank as f32 * 0.2 + 0.5) % 1.0] {
                    let q = ((a.0 as f32 + (b.0 - a.0) as f32 * f) as i32, (a.1 as f32 + (b.1 - a.1) as f32 * f) as i32);
                    ops.push(Op::Circle { c: q, r: 3, color: if wgt > 0.0 { [255, 255, 255] } else { [255, 110, 110] } });
                }
            }
        }
        for (&j, q) in p.nodes.iter().zip(&pts) {
            if let Some(q) = q {
                let glow = (act(j) / 3.0).min(1.0);
                ops.push(Op::Circle { c: *q, r: 4 + (3.0 * glow) as i32, color: col.map(|v| (f32::from(v) * (0.5 + 0.5 * glow)) as u8) });
            }
        }
    }
    for (label, end) in [("FROM", from), ("TO", to)] {
        if let Some(q) = end.and_then(at) {
            ops.push(Op::Text { s: label.into(), pos: (q.0 + 8, q.1 - 16), color: INK, font: FontSel::Small, anchor: "topleft" });
        }
    }
    let name = |j: u32| info.types.get(j as usize).filter(|t| !t.is_empty()).cloned().unwrap_or_else(|| format!("#{j}"));
    let mut lines: Vec<(Option<usize>, String)> = vec![];
    if let (Some(a), Some(b)) = (from, to) {
        let what = if ins.paths.is_empty() { "no path within 3 synapses" } else { "the strongest paths, up to 3 synapses" };
        lines.push((None, format!("{} -> {}: {what}", name(a), name(b))));
        for (rank, p) in ins.paths.iter().enumerate() {
            let names: Vec<String> = p.nodes.iter().map(|&j| name(j)).collect();
            let t = format!("{} {}  {}", if p.sign > 0 { "+" } else { "-" }, sci1(p.strength), names.join(" > "));
            lines.push((Some(rank), t.chars().take(70).collect()));
        }
    } else {
        lines.push((None, "Path tracer: now pick the other end (inspect a neuron, PATH FROM / TO HERE)".into()));
    }
    let _ = w;
    let bx = Rect::new(8, h - 14 - 18 * lines.len() as i32 - 8, 520.min(w - 16), 18 * lines.len() as i32 + 12);
    ops.push(Op::Card { rect: bx, fill: [8, 10, 16], alpha: 215, radius: 8 });
    for (n, (rank, t)) in lines.iter().enumerate() {
        match rank {
            Some(r) => {
                ops.push(Op::Circle { c: (bx.x + 12, bx.y + 14 + n as i32 * 18), r: 4, color: PATH_COLORS[r % 5] });
                ops.push(Op::Text { s: t.clone(), pos: (bx.x + 22, bx.y + 6 + n as i32 * 18), color: TEXT, font: FontSel::Small, anchor: "topleft" });
            }
            None => ops.push(Op::Text { s: t.clone(), pos: (bx.x + 8, bx.y + 6 + n as i32 * 18), color: INK, font: FontSel::Small, anchor: "topleft" }),
        }
    }
    let clear = Rect::new(bx.right() - 60, bx.y + 4, 54, 18);
    ops.push(Op::Card { rect: clear, fill: [40, 46, 58], alpha: 255, radius: 4 });
    ops.push(Op::Text { s: "CLEAR".into(), pos: clear.center(), color: INK, font: FontSel::Small, anchor: "center" });
    hits.push((clear, Hit::PathClear));
}

/// Python's `f"{x:.1e}"`: one digit, one decimal, a signed two-digit exponent.
fn sci1(x: f64) -> String {
    if x == 0.0 {
        return "0.0e+00".into();
    }
    let e = x.abs().log10().floor() as i32;
    let m = x / 10f64.powi(e);
    let (m, e) = if (m * 10.0).round() / 10.0 >= 10.0 { (m / 10.0, e + 1) } else { (m, e) };
    format!("{m:.1}e{}{:02}", if e < 0 { "-" } else { "+" }, e.abs())
}

fn lh_hud(lh: f32, k: f32) -> i32 {
    (lh * k) as i32
}

/// `shape_draw.fit_camera` + `thumbnail`: the skeleton from a slightly turned front view (25 deg yaw, -15 deg pitch), fitted to the box.
fn thumbnail(ops: &mut Vec<Op>, skel: &ktf_data::shapes::Skeleton, r: Rect) {
    let (lo, hi) = skel.bounds();
    let centre = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let (yaw, pitch) = (25f32.to_radians(), (-15f32).to_radians());
    let (sy, cy, sp, cp) = (yaw.sin(), yaw.cos(), pitch.sin(), pitch.cos());
    let rot = |p: [f32; 3]| -> (f32, f32) {
        let (x, y, z) = (p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]);
        let (x1, z1) = (x * cy + z * sy, -x * sy + z * cy);
        (x1, y * cp - z1 * sp)
    };
    let pts: Vec<(f32, f32)> = skel.xyz.iter().map(|p| rot(*p)).collect();
    let (mut x0, mut x1, mut y0, mut y1) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
    for p in &pts {
        x0 = x0.min(p.0);
        x1 = x1.max(p.0);
        y0 = y0.min(p.1);
        y1 = y1.max(p.1);
    }
    let (ex, ey) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
    let zoom = 0.92 * (r.w as f32 / ex).min(r.h as f32 / ey);
    let (mx, my) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let to = |p: (f32, f32)| ((r.centerx() as f32 + (p.0 - mx) * zoom) as i32, (r.centery() as f32 + (p.1 - my) * zoom) as i32);
    // at most about 4,000 segments: a thumbnail needs no more
    let step = (skel.n() / 4000).max(1);
    for (i, &p) in skel.parent.iter().enumerate().step_by(step) {
        if p >= 0 {
            ops.push(Op::Line { a: to(pts[i]), b: to(pts[p as usize]), w: 1, color: [255, 237, 140] });
        }
    }
}

fn card_rect(play_w: i32) -> Rect {
    // `big_rect` is (0, 58, 860, 466) in the pygame HUD (play area 890 wide): the card sits at the top right of the play area
    Rect::new(play_w - 30 - 318, 58 + 10, 308, 348)
}

/// Typing into the search box (`Game.search_key`): characters append, Backspace deletes, Enter picks the first result, Esc closes; the
/// hotkeys wait meanwhile (the keys are cleared before the actions are collected).
fn search_input(
    mut ins: ResMut<Inspector>,
    mut events: MessageReader<KeyboardInput>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    link: Res<SimLink>,
    view: Res<BrainView>,
    menu: Res<MenuOpen>,
) {
    if !ins.typing || !view.big || menu.0 {
        events.clear();
        return;
    }
    let mut changed = false;
    let mut pick = None;
    for e in events.read() {
        if e.state != ButtonState::Pressed {
            continue;
        }
        match &e.logical_key {
            Key::Escape => ins.typing = false,
            Key::Backspace => {
                ins.text.pop();
                changed = true;
            }
            Key::Enter => {
                pick = ins.results.first().copied();
                ins.typing = false;
            }
            _ => {
                if let Some(t) = &e.text {
                    for ch in t.chars().filter(|c| !c.is_control()) {
                        if ins.text.chars().count() < 40 {
                            ins.text.push(ch);
                            changed = true;
                        }
                    }
                }
            }
        }
    }
    keys.reset_all();
    if changed {
        let text = ins.text.clone();
        ins.results = match link.0.shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone() {
            Some(sh) if !text.trim().is_empty() => fly_sim::neurosearch::search(&sh.labels, &text, 12),
            _ => vec![],
        };
    }
    if let Some(i) = pick {
        link.0.send(Command::Inspect(i));
    }
}

pub struct InspectorPlugin;

impl Plugin for InspectorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Inspector>()
            .add_systems(Update, update)
            .add_systems(PreUpdate, search_input.after(crate::menu_ui::MenuInputSet).before(crate::actions::collect));
    }
}
