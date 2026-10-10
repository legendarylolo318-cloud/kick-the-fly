//! The tool wheel (`open_wheel`, `close_wheel`, `loadout_ui.draw_wheel`): hold the wheel key (`), move the mouse toward a tool, let go to
//! take it; Esc cancels. Every tool the mode allows sits in a ring; the mouse drives a pointer instead of turning the view while it is open.

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use ktf_ui::hud::{FontSel, Op};

use crate::actions::Actions;
use crate::arcade::Arcade;
use crate::config::Settings;
use crate::hud_canvas::{HudLayout, HudStats};
use crate::menu_ui::{MenuInputSet, MenuOpen, UserConfig};
use crate::player::MouseLook;
use crate::tools::{ActiveLoadout, TOOLS, ToolState};

const WHEEL_R: f32 = 150.0;
const WHEEL_DEAD: f32 = 38.0;
const PLAY_W: i32 = 890;

#[derive(Resource, Default)]
pub struct Wheel {
    pub open: bool,
    vec: Vec2,
    pub ops: Vec<Op>,
}

/// `loadout_ui.wheel_pick`: the tool a pointer offset points at, 0 at the top and clockwise, or None in the middle.
pub fn pick(v: Vec2, n: usize) -> Option<usize> {
    if n == 0 || v.length() < WHEEL_DEAD {
        return None;
    }
    let ang = v.x.atan2(-v.y).rem_euclid(std::f32::consts::TAU);
    Some(((ang / std::f32::consts::TAU * n as f32).round_ties_even() as usize) % n)
}

fn tools_for(cfg: &UserConfig) -> Vec<String> {
    ktf_data::loadout::available_tools(cfg.0.get_str("brain.mode") == "lab", false)
}

#[allow(clippy::too_many_arguments)]
fn control(
    act: Res<Actions>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut w: ResMut<Wheel>,
    menu: Res<MenuOpen>,
    arcade: Res<Arcade>,
    hud: Res<HudStats>,
    motion: Res<AccumulatedMouseMotion>,
    settings: Res<Settings>,
    cfg: Res<UserConfig>,
    mut tools: ResMut<ToolState>,
    mut loadout: ResMut<ActiveLoadout>,
    extra: Res<WheelBlockers>,
) {
    let blocked = menu.0 || arcade.open || hud.help || extra.0;
    if !w.open {
        if act.just("tool_wheel") && !blocked {
            w.open = true;
            w.vec = Vec2::ZERO;
        }
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        w.open = false;
        w.vec = Vec2::ZERO;
        keys.clear_just_pressed(KeyCode::Escape);
        return;
    }
    let sens = settings.mouse_sensitivity;
    w.vec += Vec2::new(motion.delta.x * sens, motion.delta.y * sens * if settings.invert_y { -1.0 } else { 1.0 });
    if w.vec.length() > WHEEL_R {
        w.vec = w.vec.normalize() * WHEEL_R;
    }
    if !act.held("tool_wheel") {
        let list = tools_for(&cfg);
        if let Some(i) = pick(w.vec, list.len())
            && let Some(t) = TOOLS.iter().position(|t| t.0 == list[i])
        {
            tools.selected = t;
            loadout.0.show(&list[i]);
            tools.page = loadout.0.page;
        }
        w.open = false;
        w.vec = Vec2::ZERO;
    }
}

/// Overlays the wheel must not open over (set by the systems that own them each frame).
#[derive(Resource, Default)]
pub struct WheelBlockers(pub bool);

fn blockers(
    mut b: ResMut<WheelBlockers>,
    learned: Res<crate::memory_play::Learned>,
    surgery: Res<crate::surgery_play::Surgery>,
    autopsy: Res<crate::autopsy_play::AutopsyCard>,
    chal: Res<crate::challenges::Challenges>,
    view: Res<crate::brainview::BrainView>,
) {
    b.0 = learned.train_open || surgery.open || autopsy.report.is_some() || chal.overlay || view.big;
}

fn draw(mut w: ResMut<Wheel>, cfg: Res<UserConfig>, lay: Res<HudLayout>, tools: Res<ToolState>, loadout: Res<ActiveLoadout>) {
    if !w.open {
        w.ops.clear();
        return;
    }
    let list = tools_for(&cfg);
    w.ops = build(&list, w.vec, TOOLS[tools.selected].0, &loadout.0, &cfg.0, lay.0.hud_h);
}

/// The wheel as drawing steps (`loadout_ui.draw_wheel`): the ring of tools, the one under the pointer lit, its name and what it does.
fn build(
    list: &[String],
    vec: Vec2,
    cur: &str,
    loadout: &ktf_data::loadout::Loadout,
    cfg: &ktf_data::config::Config,
    hud_h: i32,
) -> Vec<Op> {
    let n = list.len();
    let (cx, cy) = (PLAY_W / 2, hud_h / 2);
    let select = ktf_ui::pages::loadout::select_color(cfg);
    let pick = pick(vec, n);
    let mut ops = vec![Op::DiscA { c: (cx as f32, cy as f32), r: WHEEL_R + 58.0, color: [8, 10, 16], alpha: 200 }];
    for (i, name) in list.iter().enumerate() {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let (x, y) = (cx as f32 + WHEEL_R * a.sin(), cy as f32 - WHEEL_R * a.cos());
        let on = pick == Some(i);
        let equipped = loadout.tools.iter().any(|t| t == name);
        let ring = if on || name == cur {
            select
        } else if equipped {
            [52, 60, 76]
        } else {
            [44, 50, 62]
        };
        let width = if on || name == cur { 3.0 } else { 1.0 };
        ops.push(Op::DiscA { c: (x.trunc(), y.trunc()), r: 31.0, color: ring, alpha: 255 });
        ops.push(Op::DiscA { c: (x.trunc(), y.trunc()), r: 31.0 - width, color: if on { [60, 66, 84] } else { [26, 30, 40] }, alpha: 255 });
        ops.push(Op::Icon { name: name.clone(), c: (x.trunc(), y.trunc() - 4.0), color: if on { select } else { [205, 212, 224] } });
        if let Some(slot) = loadout.slot_of(name) {
            let key = cfg.key_of(&format!("slot{}", slot + 1)).to_uppercase();
            ops.push(Op::Text {
                s: key,
                pos: (x as i32 - 26, y as i32 - 26),
                color: [120, 132, 150],
                font: FontSel::Small,
                anchor: "topleft",
            });
        }
    }
    let shown = pick.map_or(cur, |i| list[i].as_str());
    if let Some(info) = ktf_data::config::schema().loadout.catalog.iter().find(|t| t.name == shown) {
        let t = |s: &str, pos: (i32, i32), color: [u8; 3], font: FontSel| Op::Text { s: s.into(), pos, color, font, anchor: "center" };
        ops.push(t(&ktf_ui::i18n::tr(&info.label), (cx, cy - 14), [240, 243, 248], FontSel::Bold));
        ops.push(t(&ktf_ui::i18n::tr(&info.category), (cx, cy + 8), [120, 132, 150], FontSel::Small));
        ops.push(t(&ktf_ui::i18n::tr("let go to pick, Esc to cancel"), (cx, cy + WHEEL_R as i32 + 62), [120, 132, 150], FontSel::Small));
        if pick.is_some() {
            ops.push(t(&ktf_ui::i18n::tr(&info.desc), (cx, cy - WHEEL_R as i32 - 62), [205, 212, 224], FontSel::Small));
        }
    }
    ops
}

/// While the wheel is open the mouse moves its pointer instead of the view.
fn freeze_look(w: Res<Wheel>, mut look: ResMut<MouseLook>) {
    look.wheel = w.open;
}

pub struct WheelPlugin;

impl Plugin for WheelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Wheel>()
            .init_resource::<WheelBlockers>()
            .add_systems(PreUpdate, (blockers, control, freeze_look).chain().before(MenuInputSet).after(bevy::input::InputSystems))
            .add_systems(Update, draw);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ktf_ui::Canvas;
    use ktf_ui::hud::{HudFonts, draw_ops};

    use super::*;

    #[test]
    fn pick_is_clockwise_from_the_top() {
        assert_eq!(pick(Vec2::new(0.0, -100.0), 18), Some(0));
        assert_eq!(pick(Vec2::new(100.0, 0.0), 4), Some(1));
        assert_eq!(pick(Vec2::new(0.0, 100.0), 4), Some(2));
        assert_eq!(pick(Vec2::new(10.0, 10.0), 18), None);
    }

    #[test]
    fn wheel_matches_pygame() {
        let cfg = ktf_data::config::Config::new(None);
        let loadout = ktf_data::loadout::build("auto", &[], false, false, "play");
        let list = ktf_data::loadout::available_tools(false, false);
        let ops = build(&list, Vec2::new(100.0, 0.0), "hand", &loadout, &cfg, 760);
        let fonts = HudFonts::new(false);
        let mut cv = Canvas { linear_light: false, ..Canvas::new(1280, 760) };
        draw_ops(&mut cv, &fonts, &ops, (-1, -1));
        let ours = cv.to_rgb_over([60, 70, 80]);
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/ui_reference/challenge_wheel.png");
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
        eprintln!("wheel: {:.3}% differ", share * 100.0);
        assert!(share < 0.012, "{:.3}% differ", share * 100.0);
    }
}
