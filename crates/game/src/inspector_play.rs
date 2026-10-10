//! The neuron inspector of the big brain view (`Game._inspect_at`, `_neuron_info`, `_draw_inspect`): click a neuron to see its type,
//! population, transmitter, firing against its calm rate and its strongest partners (drawn as lines to where they sit), and switch its whole
//! cell type off or on. Data from `fly_sim::inspect`; the card is a recorded [`ktf_ui::hud::Op`] list.
//!
//! Python drew the big view as a flat picture and picked from per-pixel neuron ids; this build's big view is the 3D cloud, so picking
//! projects every neuron the way the brain shader does and takes the nearest (then the busiest) within 8 px.
//! Not ported: PATH FROM/TO and the neuron search box (`lab/neurosearch.py`), the Lab's FLIP SIGN and PATCH CLAMP buttons, real shapes.

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

const TEXT: [u8; 3] = [205, 212, 224];
const LABEL: [u8; 3] = [120, 132, 150];
const INK: [u8; 3] = [240, 243, 248];
const DIM: [u8; 3] = [70, 78, 92];

#[derive(Resource, Default)]
pub struct Inspector {
    pub info: Option<NeuronInfo>,
    pub ops: Vec<Op>,
    hits: Vec<(Rect, i8)>,
    calm: Vec<f32>,
    press: Option<Vec2>,
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
) {
    let opted_now = cfg.0.get_bool("brain.neuron_shapes");
    if opted_now && !*was_opted {
        shapes.reset_offline(); // the player just turned the download on: ask again for what was refused
    }
    *was_opted = opted_now;
    let Some(s) = snap.0.as_deref() else { return };
    let ins = &mut *ins;
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
            if let Some((_, mode)) = ins.hits.iter().find(|(r, _)| r.contains(hud)) {
                if let Some(n) = ins.info.as_ref() {
                    link.0.send(Command::SurgeryType { ty: n.ty.clone(), mode: *mode });
                }
            } else if ins.info.as_ref().is_some_and(|_| card_rect(lay.0.play_w).contains(hud)) {
                // the card swallows the click
            } else {
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
    let Some(n) = ins.info.clone() else {
        ins.ops.clear();
        ins.hits.clear();
        return;
    };
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
    let mut hits = vec![];
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
        hits.push((r, mode));
    }
    ins.ops = ops;
    ins.hits = hits;
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

pub struct InspectorPlugin;

impl Plugin for InspectorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Inspector>().add_systems(Update, update);
    }
}
