//! The brain surgery screen (`Game._draw_surgery`, O): 23 groups of real neurons, each OFF / normal / ON, applied on the brain thread
//! (`fly_sim::surgery`, `Command::SurgeryEntry`). Drawn as a recorded [`ktf_ui::hud::Op`] list like the challenge panels.

use bevy::prelude::*;
use fly_sim::surgery::ENTRIES;
use ktf_ui::Rect;
use ktf_ui::hud::{FontSel, Op};

use crate::actions::Actions;
use crate::app::{LatestSnapshot, SimLink};
use crate::feedback::{Notes, Sfx, Tag};
use crate::menu_ui::{MenuInputSet, MenuOpen};
use crate::player::MouseLook;
use crate::sim_thread::Command;

const H: i32 = 760;

#[derive(Resource, Default)]
pub struct Surgery {
    pub open: bool,
    /// The screen as drawing steps and its click areas (`(rect, id)`; `row:<k>:<mode>` or `clear`).
    pub ops: Vec<Op>,
    hits: Vec<(Rect, String)>,
}

fn build(s: &mut Surgery, counts: &[u32], modes: &[i8], types: &[(String, i8)]) {
    let mut ops = vec![];
    let panel = Rect::new(75, 12, 740, (H - 24).min(690));
    ops.push(Op::Veil(170));
    ops.push(Op::Card { rect: panel, fill: [18, 21, 28], alpha: 255, radius: 16 });
    ops.push(Op::Stroke { rect: panel, color: [40, 47, 60], width: 1, radius: 16 });
    let x = panel.x + 24;
    let text = |ops: &mut Vec<Op>, s: &str, pos: (i32, i32), color: [u8; 3], font: FontSel, anchor: &'static str| {
        ops.push(Op::Text { s: s.into(), pos, color, font, anchor });
    };
    text(&mut ops, "BRAIN SURGERY", (x, panel.y + 14), [240, 243, 248], FontSel::Title, "topleft");
    text(
        &mut ops,
        "Silence (OFF) or stimulate (ON) real neuron groups. Hemifield lesions test wiring outcomes, not injury.",
        (x + 2, panel.y + 50),
        [120, 132, 150],
        FontSel::Small,
        "topleft",
    );
    let mut y = panel.y + 76;
    let mut hits = vec![];
    for (k, (label, _)) in ENTRIES.iter().enumerate() {
        let n = counts.get(k).copied().unwrap_or(0);
        let mode = modes.get(k).copied().unwrap_or(0);
        let lower = label.to_lowercase();
        let is_hemi = lower.contains("hemifield") || lower.contains("hemisphere");
        let col = if mode != 0 {
            [255, 176, 64]
        } else if is_hemi {
            [180, 210, 240]
        } else {
            [205, 212, 224]
        };
        text(&mut ops, label, (x, y), col, if label.chars().count() > 34 { FontSel::Small } else { FontSel::Text }, "topleft");
        text(&mut ops, &ktf_data::diag::group3(n as usize), (panel.right() - 240, y + 2), [120, 132, 150], FontSel::Small, "topright");
        for (j, (m, label)) in [(-1i8, "OFF"), (0, "-"), (1, "ON")].into_iter().enumerate() {
            let r = Rect::new(panel.right() - 216 + j as i32 * 64, y - 1, 56, 22);
            let on = mode == m;
            let fill = if on {
                match m {
                    -1 => [60, 110, 200],
                    1 => [230, 130, 50],
                    _ => [70, 76, 90],
                }
            } else {
                [30, 34, 44]
            };
            ops.push(Op::Card { rect: r, fill, alpha: 255, radius: 6 });
            text(&mut ops, label, r.center(), if on { [240, 243, 248] } else { [120, 132, 150] }, FontSel::Small, "center");
            ops.push(Op::Hit { rect: r, id: format!("row:{k}:{m}") });
            hits.push((r, format!("row:{k}:{m}")));
        }
        y += 26;
    }
    let active: Vec<String> =
        types.iter().filter(|t| t.1 != 0).map(|(t, m)| format!("{t} {}", if *m < 0 { "off" } else { "on" })).collect();
    if !active.is_empty() {
        text(&mut ops, &format!("From the inspector: {}", active.join(", ")), (x, y + 2), [255, 176, 64], FontSel::Small, "topleft");
    }
    let clear = Rect::new(panel.right() - 150, panel.bottom() - 36, 126, 26);
    ops.push(Op::Card { rect: clear, fill: [60, 64, 76], alpha: 255, radius: 8 });
    text(&mut ops, "CLEAR ALL", clear.center(), [240, 243, 248], FontSel::Bold, "center");
    ops.push(Op::Hit { rect: clear, id: "clear".into() });
    hits.push((clear, "clear".into()));
    text(
        &mut ops,
        "OFF adds -2.4 per step (silences). ON adds +0.48. Reported as wiring outcomes, not physical trauma.",
        (x, panel.bottom() - 30),
        [70, 78, 92],
        FontSel::Small,
        "topleft",
    );
    s.ops = ops;
    s.hits = hits;
}

#[cfg(test)]
pub fn build_for_test(s: &mut Surgery, counts: &[u32], modes: &[i8], types: &[(String, i8)]) {
    build(s, counts, modes, types);
}

fn update(mut s: ResMut<Surgery>, snap: Res<LatestSnapshot>) {
    if !s.open {
        s.ops.clear();
        s.hits.clear();
        return;
    }
    let Some(sn) = snap.0.as_deref() else { return };
    let counts = sn.info.as_deref().map(|i| i.surgery_counts.clone()).unwrap_or_default();
    build(&mut s, &counts, &sn.surgery_modes, &sn.surgery_types);
}

/// O opens and closes it; Esc closes it before the pause menu would open.
fn keys(
    act: Res<Actions>,
    mut keycodes: ResMut<ButtonInput<KeyCode>>,
    mut s: ResMut<Surgery>,
    menu: Res<MenuOpen>,
    mut look: ResMut<MouseLook>,
) {
    if keycodes.just_pressed(KeyCode::Escape) && s.open {
        s.open = false;
        keycodes.clear_just_pressed(KeyCode::Escape);
    } else if act.just("surgery") && !menu.0 {
        s.open = !s.open;
        if s.open {
            look.captured = false;
        }
    }
}

/// A click at HUD position `pos`; true when the screen took it.
pub fn click(
    s: &Surgery,
    pos: (i32, i32),
    link: &SimLink,
    notes: &mut Notes,
    sfx: &mut MessageWriter<Sfx>,
    now: f32,
    modes: &[i8],
) -> bool {
    if !s.open {
        return false;
    }
    let Some((_, id)) = s.hits.iter().find(|(r, _)| r.contains(pos)) else { return true };
    if id == "clear" {
        link.0.send(Command::SurgeryClear);
        notes.push(now, "SURGERY  cleared", Tag::GameRule);
    } else if let Some(rest) = id.strip_prefix("row:") {
        let mut it = rest.split(':');
        let (Some(k), Some(m)) = (it.next().and_then(|x| x.parse::<usize>().ok()), it.next().and_then(|x| x.parse::<i8>().ok())) else {
            return true;
        };
        if modes.get(k).copied() != Some(m) {
            link.0.send(Command::SurgeryEntry { index: k, mode: m });
            let label = ENTRIES[k].0.split(" (").next().unwrap_or("");
            notes.push(
                now,
                format!(
                    "SURGERY  {label}: {}",
                    if m < 0 {
                        "off"
                    } else if m > 0 {
                        "on"
                    } else {
                        "normal"
                    }
                ),
                Tag::GameRule,
            );
        }
    }
    sfx.write(Sfx::new("click"));
    true
}

pub struct SurgeryPlugin;

impl Plugin for SurgeryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Surgery>()
            .add_systems(PreUpdate, keys.before(MenuInputSet).after(bevy::input::InputSystems))
            .add_systems(Update, update);
    }
}
