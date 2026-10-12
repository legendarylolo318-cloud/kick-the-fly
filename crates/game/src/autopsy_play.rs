//! The brain autopsy card (`Game._draw_autopsy`): three seconds after the fly dies, a veil fades in and a card shows the last two seconds of
//! every group against its calm baseline, a timeline of three summaries and the pain index. The numbers come from `fly_sim::autopsy`
//! (checked against Python's `_autopsy`); this module is the card.

use bevy::prelude::*;
use fly_sim::autopsy::{Autopsy, Row};
use ktf_ui::Rect;
use ktf_ui::hud::{FontSel, HudFonts, Op};

use crate::app::{SimEvents, SimLink};
use crate::feedback::{Notes, Sfx, Tag};
use crate::fly::{Fly, ResetFly};
use crate::hud_canvas::HudStats;
use crate::sim_thread::SimEvent;

const UP: [u8; 3] = [230, 103, 103];
const DOWN: [u8; 3] = [57, 135, 229];
const MID: [u8; 3] = [56, 56, 53];
const SERIES: [[u8; 3]; 3] = [[57, 135, 229], [217, 89, 38], [25, 158, 112]];
const S_CRIT: [u8; 3] = [208, 59, 59];
const INK: [u8; 3] = [240, 243, 248];
const TEXT: [u8; 3] = [205, 212, 224];
const LABEL: [u8; 3] = [120, 132, 150];
const DIM: [u8; 3] = [70, 78, 92];
const AMBER: [u8; 3] = [255, 176, 64];
/// Seconds of flat line shown before the report (`AUTOPSY_DELAY`).
pub const DELAY_S: f32 = 3.0;

pub struct Report {
    pub a: Autopsy,
    pub alive_s: f32,
    pub hits: u32,
    pub by: String,
    pub pain_peak: f32,
    pub pain_max_s: f32,
    /// Virtual time of the death.
    pub died_at: f32,
}

#[derive(Resource, Default)]
pub struct AutopsyCard {
    pub report: Option<Report>,
    born: f32,
    died_at: f32,
    pub ops: Vec<Op>,
    hits: Vec<(Rect, String)>,
}

fn text(ops: &mut Vec<Op>, s: &str, pos: (i32, i32), color: [u8; 3], font: FontSel, anchor: &'static str) {
    ops.push(Op::Text { s: s.into(), pos, color, font, anchor });
}

fn line(ops: &mut Vec<Op>, a: (i32, i32), b: (i32, i32), w: i32, color: [u8; 3]) {
    ops.push(Op::Line { a, b, w, color });
}

/// The card as drawing steps. `e` is the veil's fade-in 0..1.
fn build(rep: &Report, e: f32, f: &HudFonts) -> (Vec<Op>, Vec<(Rect, String)>) {
    let mut ops = vec![];
    let mut hits = vec![];
    ops.push(Op::VeilC([6, 7, 10], (200.0 * e) as u32));
    let card = Rect::new(28, 26, crate::hud_canvas::live_play_w() - 56, crate::hud_canvas::live_hud_h() - 52);
    ops.push(Op::Card { rect: card, fill: [20, 22, 28], alpha: 255, radius: 16 });
    ops.push(Op::Stroke { rect: card, color: [40, 47, 60], width: 1, radius: 16 });
    let (x, y) = (card.x + 26, card.y + 18);
    text(&mut ops, "BRAIN AUTOPSY", (x, y), INK, FontSel::Title, "topleft");
    let (m, s) = ((rep.alive_s as i32) / 60, (rep.alive_s as i32) % 60);
    text(
        &mut ops,
        &format!("survived {m}:{s:02}   {} hits   killed by {}", rep.hits, rep.by),
        (x + 2, y + 44),
        LABEL,
        FontSel::Text,
        "topleft",
    );
    let top = &rep.a.top;
    text(
        &mut ops,
        &format!(
            "Biggest change before death: {} at {:.1}x its calm rate ({:.1} > {:.1} spikes/s)",
            top.name, top.ratio, top.base, top.final_
        ),
        (x + 2, y + 68),
        TEXT,
        FontSel::Text,
        "topleft",
    );
    text(&mut ops, "PAIN", (x + 2, y + 92), S_CRIT, FontSel::Bold, "topleft");
    text(
        &mut ops,
        &format!("peak {:.0}/100    maxed out for {:.1} s    last 2 s alive {:.0}/100", rep.pain_peak, rep.pain_max_s, rep.a.pain_last),
        (x + 2 + f.bold.size("PAIN") + 10, y + 92),
        TEXT,
        FontSel::Text,
        "topleft",
    );
    let r_btn = Rect::new(card.right() - 196, card.y + 22, 170, 42);
    ops.push(Op::Card { rect: r_btn, fill: AMBER, alpha: 255, radius: 10 });
    text(&mut ops, "NEW FLY  (R)", r_btn.center(), [30, 20, 8], FontSel::Bold, "center");
    hits.push((r_btn, "new_fly".to_string()));
    for (k, (label, what)) in [("SAVE IMAGE", "png"), ("SAVE DEATH GIF", "gif")].into_iter().enumerate() {
        let r = Rect::new(card.right() - 196 + k as i32 * 88 - if k == 0 { 0 } else { 4 }, card.y + 70, if k == 0 { 84 } else { 110 }, 26);
        ops.push(Op::Card { rect: r, fill: [44, 50, 64], alpha: 255, radius: 7 });
        text(&mut ops, label, r.center(), INK, FontSel::Small, "center");
        hits.push((r, what.to_string()));
    }
    // diverging bars: the last 2 s alive against the calm baseline, log2 scale
    let y0 = card.y + 142;
    let (lab_w, col_w) = (170, 64);
    let cx0 = x + lab_w;
    let chart_w = card.w - 52 - lab_w - col_w * 3 - 40;
    let mid = cx0 + chart_w / 2;
    text(&mut ops, "Last 2 s alive vs its calm baseline", (x, y0 - 8), INK, FontSel::Bold, "topleft");
    let hdr_y = y0 + 18;
    text(&mut ops, "calm", (cx0 + chart_w + col_w, hdr_y), LABEL, FontSel::Small, "topright");
    text(&mut ops, "final", (cx0 + chart_w + col_w * 2, hdr_y), LABEL, FontSel::Small, "topright");
    text(&mut ops, "dead", (cx0 + chart_w + col_w * 3, hdr_y), LABEL, FontSel::Small, "topright");
    for (tick, lab) in [(-4, "1/16x"), (-2, "1/4x"), (0, "same"), (2, "4x"), (4, "16x")] {
        let tx = mid as f32 + tick as f32 / 4.0 * (chart_w as f32 / 2.0 - 8.0);
        text(&mut ops, lab, (tx as i32, hdr_y), LABEL, FontSel::Small, "midtop");
    }
    let mut ry = hdr_y + 20;
    let row_h = 15;
    let chart_top = ry;
    let n_rows = rep.a.rows.len() as i32;
    for k in [-4, -2, 2, 4] {
        let tx = (mid as f32 + k as f32 / 4.0 * (chart_w as f32 / 2.0 - 8.0)) as i32;
        line(&mut ops, (tx, chart_top - 2), (tx, chart_top + n_rows * row_h + 8), 1, [32, 35, 42]);
    }
    line(&mut ops, (mid, chart_top - 2), (mid, chart_top + n_rows * row_h + 8), 2, [90, 90, 86]);
    let mut prev: Option<&str> = None;
    for row in &rep.a.rows {
        if prev.is_some_and(|p| p != row.section) {
            ry += 8;
        }
        prev = Some(row.section);
        bar_row(&mut ops, row, (x, ry), mid, chart_w, cx0, col_w, row_h);
        ry += row_h;
    }
    text(&mut ops, "spikes/s per neuron. Red = higher than calm, blue = lower.", (x, ry + 6), LABEL, FontSel::Small, "topleft");
    // the timeline on one shared axis
    let mut ty = ry + 30;
    let mut th = card.bottom() - ty - 48;
    let (tx0, tw) = (x + 40, card.w - 52 - 40);
    text(&mut ops, "Activity vs calm, 20 s before death to 3 s after", (x, ty - 4), INK, FontSel::Bold, "topleft");
    let mut lx = x + 380;
    for ((label, _), col) in rep.a.series.iter().zip(SERIES) {
        ops.push(Op::Card { rect: Rect::new(lx, ty + 3, 14, 4), fill: col, alpha: 255, radius: 2 });
        text(&mut ops, label, (lx + 20, ty - 2), TEXT, FontSel::Small, "topleft");
        lx += 20 + f.small.size(label) + 18;
    }
    ty += 22;
    th -= 22;
    let (lo, hi) = (-3.0f32, 5.0f32);
    let yof = |v: f32| ty as f32 + th as f32 - ((v.max(1e-6).log2()).clamp(lo, hi) - lo) / (hi - lo) * th as f32;
    for (v, lab) in [(0.125f32, "0"), (0.25, "1/4x"), (1.0, "1x"), (4.0, "4x"), (16.0, "16x")] {
        let gy = yof(v) as i32;
        line(&mut ops, (tx0, gy), (tx0 + tw, gy), 1, if v == 1.0 { [70, 70, 66] } else { [32, 35, 42] });
        text(&mut ops, lab, (tx0 - 6, gy - 7), LABEL, FontSel::Small, "topright");
    }
    let n = rep.a.n.max(2) as f32;
    for ((_, vals), col) in rep.a.series.iter().zip(SERIES) {
        if vals.len() < 2 {
            continue;
        }
        let pts: Vec<(i32, i32)> =
            vals.iter().enumerate().map(|(k, v)| ((tx0 as f32 + k as f32 * tw as f32 / (n - 1.0)) as i32, yof(*v) as i32)).collect();
        for w in pts.windows(2) {
            line(&mut ops, w[0], w[1], 2, col);
        }
    }
    // the pain strip: one hue, dark = none, bright red = maxed
    let sy = ty + th + 8;
    text(&mut ops, "pain", (tx0 - 6, sy), LABEL, FontSel::Small, "topright");
    let pain = &rep.a.pain;
    if !pain.is_empty() {
        let step = (pain.len() / 300).max(1);
        let cw = tw as f32 / pain.len() as f32 * step as f32;
        for k in (0..pain.len()).step_by(step) {
            let t = (pain[k] / 100.0).clamp(0.0, 1.0);
            let col = [0, 1, 2].map(|j| (f32::from(MID[j]) + (f32::from(UP[j]) - f32::from(MID[j])) * t) as u8);
            ops.push(Op::Card {
                rect: Rect::new((tx0 as f32 + k as f32 * tw as f32 / pain.len() as f32) as i32, sy, cw.ceil() as i32 + 1, 14),
                fill: col,
                alpha: 255,
                radius: 0,
            });
        }
    }
    let dx = (tx0 as f32 + rep.a.death_at as f32 * tw as f32 / (n - 1.0)) as i32;
    line(&mut ops, (dx, ty - 2), (dx, sy + 14), 2, S_CRIT);
    text(&mut ops, "died", (dx + 6, ty), S_CRIT, FontSel::Small, "topleft");
    (ops, hits)
}

#[allow(clippy::too_many_arguments)]
fn bar_row(ops: &mut Vec<Op>, row: &Row, (x, ry): (i32, i32), mid: i32, chart_w: i32, cx0: i32, col_w: i32, row_h: i32) {
    let lr = row.ratio.max(1e-3).log2();
    let frac = (lr / 4.0).clamp(-1.0, 1.0);
    let bl = (frac.abs() * (chart_w as f32 / 2.0 - 8.0)) as i32;
    let col = if frac > 0.0 { UP } else { DOWN };
    if bl >= 2 {
        let rx = if frac > 0.0 { mid + 2 } else { mid - 2 - bl };
        ops.push(Op::Card { rect: Rect::new(rx, ry + 2, bl, row_h - 5), fill: col, alpha: 255, radius: 3 });
    } else {
        ops.push(Op::Circle { c: (mid, ry + row_h / 2), r: 3, color: MID });
    }
    text(ops, &row.name, (x, ry), TEXT, FontSel::Small, "topleft");
    let pct = (row.ratio - 1.0) * 100.0;
    let lab = if row.ratio >= 2.0 { format!("{:.1}x", row.ratio) } else { format!("{pct:+.0}%") };
    let (lx, anchor) = if frac > 0.0 { (mid + 2 + bl + 6, "topleft") } else { (mid - 2 - bl - 6, "topright") };
    text(ops, &lab, (lx, ry), INK, FontSel::Small, anchor);
    text(ops, &format!("{:.1}", row.base), (cx0 + chart_w + col_w, ry), LABEL, FontSel::Small, "topright");
    text(ops, &format!("{:.1}", row.final_), (cx0 + chart_w + col_w * 2, ry), INK, FontSel::Small, "topright");
    text(ops, &format!("{:.1}", row.after), (cx0 + chart_w + col_w * 3, ry), DIM, FontSel::Small, "topright");
}

#[allow(clippy::too_many_arguments)]
fn update(
    mut card: ResMut<AutopsyCard>,
    events: Res<SimEvents>,
    virt: Res<Time<Virtual>>,
    mut resets: MessageReader<ResetFly>,
    hud: Res<HudStats>,
    flies: Query<&Fly>,
    cfg: Res<crate::menu_ui::UserConfig>,
    mut fonts: Local<Option<HudFonts>>,
) {
    let now = virt.elapsed_secs();
    if resets.read().count() > 0 {
        card.report = None;
        card.born = now;
        card.ops.clear();
        card.hits.clear();
    }
    let dead_at = flies.iter().next().and_then(|f| f.dead_at);
    for e in &events.0 {
        if let SimEvent::Autopsy(a) = e {
            let died = dead_at.unwrap_or(now);
            card.died_at = died;
            card.report = Some(Report {
                a: (**a).clone(),
                alive_s: (died - card.born).max(0.0),
                hits: hud.hits,
                by: flies.iter().next().map_or("", |f| f.damage_src).to_string(),
                pain_peak: hud.pain_peak,
                pain_max_s: hud.pain_max_s,
                died_at: died,
            });
        }
    }
    let AutopsyCard { report, ops, hits, .. } = &mut *card;
    match report {
        Some(rep) => {
            let e = ((now - rep.died_at - DELAY_S) / 0.3).clamp(0.0, 1.0);
            let f = fonts.get_or_insert_with(|| HudFonts::new(cfg.0.get_bool("access.larger_text")));
            let (o, h) = build(rep, e.max(0.0), f);
            *ops = o;
            *hits = h;
        }
        None => {
            ops.clear();
            hits.clear();
        }
    }
}

/// A click at HUD position `pos` on the card; true when taken.
pub fn click(
    card: &AutopsyCard,
    pos: (i32, i32),
    reset: &mut MessageWriter<ResetFly>,
    notes: &mut Notes,
    sfx: &mut MessageWriter<Sfx>,
    now: f32,
) -> bool {
    if card.report.is_none() {
        return false;
    }
    if let Some((_, id)) = card.hits.iter().find(|(r, _)| r.contains(pos)) {
        match id.as_str() {
            "new_fly" => {
                reset.write(ResetFly::default());
            }
            "png" => {
                notes.push(now, "AUTOPSY  press F12 for a screenshot of the card", Tag::GameRule);
            }
            _ => notes.push(now, "AUTOPSY  GIF capture is not in this build", Tag::GameRule),
        }
        sfx.write(Sfx::new("click"));
    }
    true
}

pub struct AutopsyPlugin;

impl Plugin for AutopsyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AutopsyCard>().add_systems(Update, update);
        let _ = std::marker::PhantomData::<SimLink>;
    }
}

#[cfg(test)]
mod tests {
    //! The card against `Game._draw_autopsy` drawing the same report (`tests/ui_reference/challenge_autopsy.png`).
    use std::path::PathBuf;

    use ktf_ui::Canvas;
    use ktf_ui::hud::{HudFonts, draw_ops};

    use super::*;

    #[test]
    fn card_matches_pygame() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let g: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("tests/golden/brain_autopsy.json")).unwrap()).unwrap();
        let pops = fly_sim::groups::population_names();
        let rows: Vec<Row> = g["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let raw = r["name"].as_str().unwrap();
                let name = fly_sim::autopsy::PRETTY.iter().find(|p| p.0 == raw).map_or(raw, |p| p.1).to_string();
                Row {
                    name,
                    base: r["base"].as_f64().unwrap() as f32,
                    final_: r["final"].as_f64().unwrap() as f32,
                    after: r["after"].as_f64().unwrap() as f32,
                    ratio: r["ratio"].as_f64().unwrap() as f32,
                    section: if pops.contains(&raw) || raw == "whole brain" { "pop" } else { "detail" },
                }
            })
            .collect();
        let top = rows[1..].iter().max_by(|a, b| a.ratio.total_cmp(&b.ratio)).unwrap().clone();
        let f32s = |v: &serde_json::Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
        let labels = ["whole brain", "touch neurons", "descending"];
        let series = g["series"].as_array().unwrap().iter().zip(labels).map(|(s, l)| (l.to_string(), f32s(s))).collect();
        let a = Autopsy {
            rows,
            series,
            death_at: g["death_at"].as_u64().unwrap() as usize,
            n: g["n"].as_u64().unwrap() as usize,
            top,
            pain: f32s(&g["pain"]),
            pain_last: g["pain_last"].as_f64().unwrap() as f32,
        };
        let rep = Report { a, alive_s: 95.0, hits: 7, by: "being kicked".into(), pain_peak: 42.0, pain_max_s: 1.5, died_at: 0.0 };
        let fonts = HudFonts::new(false);
        let (ops, _) = build(&rep, 1.0, &fonts);
        let mut cv = Canvas { linear_light: false, ..Canvas::new(1280, 760) };
        draw_ops(&mut cv, &fonts, &ops, (-1, -1));
        let ours = cv.to_rgb_over([60, 70, 80]);
        let p = root.join("tests/ui_reference/challenge_autopsy.png");
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(p).unwrap()));
        let mut r = dec.read_info().unwrap();
        let mut buf = vec![0; r.output_buffer_size().unwrap()];
        let info = r.next_frame(&mut buf).unwrap();
        let theirs = &buf[..info.buffer_size()];
        let bad = ours
            .as_chunks::<3>()
            .0
            .iter()
            .zip(theirs.as_chunks::<3>().0.iter())
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (i32::from(*x) - i32::from(*y)).abs() > 60))
            .count();
        let share = bad as f64 / (1280.0 * 760.0);
        eprintln!("autopsy card: {:.3}% differ", share * 100.0);
        assert!(share < 0.02, "{:.3}% differ", share * 100.0);
    }
}
