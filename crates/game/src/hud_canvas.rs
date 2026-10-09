//! The HUD as the pygame game drew it: one canvas of `hud_w x hud_h` units (`ktf_ui::hud`), shown over the 3D view and scaled to the window.
//! This module gathers the numbers (pain, reward, hits, the tool bar, the brain panel, popups) from the game's resources each frame, keeps
//! the few things the HUD itself remembers (the smoothed pain index, its trace, peaks), and places the 3D view and the brain image to
//! match the layout (`kick3d.compute_layout`).

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;
use ktf_ui::Canvas;
use ktf_ui::hud::*;
use ktf_ui::layout::{Layout, compute_layout};

use crate::app::{LatestSnapshot, SimLink};
use crate::arcade::Arcade;
use crate::brainview::{BrainView, PanelMode};
use crate::config::Arena;
use crate::controls::TimeControl;
use crate::feedback::{Notes, Popup as PopupMsg, Tag};
use crate::fly::{Fly, FlyDied, FlyHit, MAX_HEALTH, ResetFly};
use crate::menu_ui::{MenuOpen, UserConfig};
use crate::player::{MouseLook, PlayerCamera};
use crate::sim_thread::Command;
use crate::tools::{ActiveLoadout, LASER_TARGETS, TOOLS, ToolState};

/// Where the HUD sits in the window: recomputed every frame from the window size and the settings.
#[derive(Resource, Clone, Copy)]
pub struct HudLayout(pub Layout);

impl Default for HudLayout {
    fn default() -> Self {
        Self(compute_layout(1280, 720, true, 0, 1.0))
    }
}

/// What the HUD remembers: per-fly counters and the smoothed pain (`FlySlot.pain*`, `reward`).
#[derive(Resource, Default)]
pub struct HudStats {
    pub hits: u32,
    pub kills: u32,
    pub pain: f32,
    pub pain_parts: [f32; 5],
    pub pain_trace: Vec<f32>,
    pub pain_peak: f32,
    pub pain_max_s: f32,
    pub reward: f32,
    frame: u64,
    pub help: bool,
}

#[derive(Resource)]
struct HudCanvas {
    canvas: Canvas,
    image: Handle<Image>,
    crisp: bool,
}

#[derive(Component)]
struct HudNode;

#[derive(Default)]
struct PopupLive {
    live: Vec<(f32, Vec3, String, Color)>,
}

fn new_image(w: u32, h: u32, nearest: bool) -> Image {
    let mut img = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        vec![0; (w * h * 4) as usize],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    img.sampler = if nearest { ImageSampler::nearest() } else { ImageSampler::linear() };
    img
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // the UI camera covers the whole window, while the 3D camera may only cover the part left of the brain panel
    commands.spawn((
        Camera2d,
        IsDefaultUiCamera,
        Camera { order: 100, clear_color: ClearColorConfig::None, ..default() },
        bevy::camera::visibility::RenderLayers::layer(31),
    ));
    let image = images.add(new_image(1280, 720, true));
    commands.spawn((
        HudNode,
        ImageNode::new(image.clone()),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        GlobalZIndex(500),
    ));
    commands.insert_resource(HudCanvas { canvas: Canvas { linear_light: true, ..Canvas::new(1280, 720) }, image, crisp: true });
}

/// The layout for this frame, and the 3D camera's viewport: with a solid brain panel the room is drawn only left of it.
#[allow(clippy::too_many_arguments)]
fn layout(
    window: Single<&Window, With<PrimaryWindow>>,
    cfg: Res<UserConfig>,
    view: Res<BrainView>,
    mut lay: ResMut<HudLayout>,
    mut cam: Single<&mut Camera, With<PlayerCamera>>,
) {
    let (wn, hn) = (window.physical_width() as i32, window.physical_height() as i32);
    if wn < 2 || hn < 2 {
        return;
    }
    let mode = match view.mode {
        PanelMode::Solid => 0,
        PanelMode::SeeThrough => 1,
        PanelMode::Faint => 2,
        PanelMode::Hidden => 3,
    };
    let l = compute_layout(wn, hn, cfg.0.get_str("graphics.menu_size") != "large", mode, cfg.0.get_f32("graphics.ui_scale"));
    lay.0 = l;
    let want = if view.mode == PanelMode::Solid && !view.big {
        let w = ((l.view_w as f32 * l.scale).round() as u32).clamp(1, wn as u32);
        Some(bevy::camera::Viewport { physical_position: UVec2::ZERO, physical_size: UVec2::new(w, hn as u32), ..default() })
    } else {
        None
    };
    let same = match (&cam.viewport, &want) {
        (None, None) => true,
        (Some(a), Some(b)) => a.physical_size == b.physical_size && a.physical_position == b.physical_position,
        _ => false,
    };
    if !same {
        cam.viewport = want;
    }
}

fn count_events(
    mut hits: MessageReader<FlyHit>,
    mut died: MessageReader<FlyDied>,
    mut resets: MessageReader<ResetFly>,
    mut st: ResMut<HudStats>,
) {
    st.hits += hits.read().count() as u32;
    st.kills += died.read().count() as u32;
    if resets.read().count() > 0 {
        let (kills, help) = (st.kills, st.help);
        *st = HudStats { kills, help, ..default() };
    }
}

/// `Game.update`'s pain block, once per frame at 60 Hz: the pain index follows the sim's pain parts, the DN alarm held a little, a fly in
/// pain pokes its punishment neurons, reward follows the dopamine level.
fn update_stats(snap: Res<LatestSnapshot>, flies: Query<&Fly>, link: Res<SimLink>, mut st: ResMut<HudStats>) {
    let Some(s) = snap.0.as_deref() else { return };
    let Some(info) = s.info.as_deref() else { return };
    st.frame += 1;
    let mut parts = s.pain_parts;
    parts[3] = parts[3].max(st.pain_parts[3] * 0.96);
    for (cur, new) in st.pain_parts.iter_mut().zip(parts) {
        *cur += (new - *cur) * 0.15;
    }
    let idx = fly_sim::groups::pain_index(&parts);
    st.pain += (idx - st.pain) * 0.15;
    let reward_level = info.group_names.iter().position(|n| n == "reward").map_or(1.0, |i| s.levels.get(i).copied().unwrap_or(1.0));
    st.reward += (((reward_level - 1.0) / 0.6).clamp(0.0, 1.0) * 100.0 - st.reward) * 0.1;
    let dead = flies.iter().next().is_none_or(|f| f.dead());
    if !dead && st.pain > 25.0 && st.frame.is_multiple_of(3) {
        link.0.send(Command::Poke { region: "punish".into(), side: None, strength: st.pain / 100.0 });
    }
    if !dead {
        st.pain_peak = st.pain_peak.max(st.pain);
        if st.pain >= 99.0 {
            st.pain_max_s += 1.0 / 60.0;
        }
    }
    let p = st.pain;
    st.pain_trace.push(p);
    let n = st.pain_trace.len();
    if n > 720 {
        st.pain_trace.drain(0..n - 720);
    }
}

fn pygame_state(f: &Fly, now: f32, arena: &str) -> String {
    if f.shattered_at.is_some() {
        "shattered".into()
    } else if f.dissolved_at.is_some() {
        "dissolved".into()
    } else if f.frozen_at.is_some() {
        "frozen solid".into()
    } else if f.dead() {
        "dead".into()
    } else if f.stuck.len() >= 2 {
        "stuck on flypaper".into()
    } else if arena == "pool" && f.wet > 0.5 && now >= f.react.escape_until {
        "swimming".into()
    } else if f.wrapped {
        "wrapped in silk".into()
    } else if f.grabbed.is_some() {
        "grabbed".into()
    } else if now < f.react.escape_until {
        "flying".into()
    } else if now < f.react.eating_until {
        "eating sugar".into()
    } else if now < f.react.stun_until {
        "stunned".into()
    } else {
        f.action.into()
    }
}

const MOTOR_ROWS: [(&str, &str, Option<f32>); 10] = [
    ("jump", "head-touch DNs  > JUMP", Some(3.0)),
    ("run", "body-touch DNs  > RUN", Some(2.4)),
    ("kick", "leg-touch DNs   > KICK", Some(2.0)),
    ("walk", "DNp09 (P9)      > WALK", Some(3.0)),
    ("back", "MDN moonwalker  > BACK UP", Some(3.8)),
    ("turn_l", "DNa01/02 left   > TURN", Some(2.1)),
    ("turn_r", "DNa01/02 right  > TURN", Some(2.1)),
    ("fly", "DNg02 wing power> FLY", Some(1.58)),
    ("escape", "DNp01 giant fiber> DODGE", Some(4.0)),
    ("fire", "DNp35 object DNs> SHOOT", Some(3.0)),
];

/// Everything the HUD reads (bundled: a system takes at most 16 parameters).
#[derive(bevy::ecs::system::SystemParam)]
struct HudIn<'w, 's> {
    time: Res<'w, Time<Virtual>>,
    real: Res<'w, Time<Real>>,
    snap: Res<'w, LatestSnapshot>,
    st: Res<'w, HudStats>,
    notes: Res<'w, Notes>,
    tools: Res<'w, ToolState>,
    loadout: Res<'w, ActiveLoadout>,
    arena: Res<'w, Arena>,
    cfg: Res<'w, UserConfig>,
    view: Res<'w, BrainView>,
    tc: Res<'w, TimeControl>,
    look: Res<'w, MouseLook>,
    menu: Res<'w, MenuOpen>,
    arcade: Res<'w, Arcade>,
    settings: Res<'w, crate::config::Settings>,
    flies: Query<'w, 's, &'static Fly>,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn draw(
    mut hc: ResMut<HudCanvas>,
    lay: Res<HudLayout>,
    i: HudIn,
    mut incoming: MessageReader<PopupMsg>,
    mut popups: Local<PopupLive>,
    camera: Single<(&Camera, &GlobalTransform), With<PlayerCamera>>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut node: Single<&mut ImageNode, With<HudNode>>,
) {
    let HudIn { time, real, snap, st, notes, tools, loadout, arena, cfg, view, tc, look, menu, arcade, settings, flies } = i;
    let l = lay.0;
    let crisp = (l.scale.fract() == 0.0) && l.scale >= 1.0;
    if (hc.canvas.w, hc.canvas.h) != (l.hud_w, l.hud_h) || hc.crisp != crisp {
        hc.canvas = Canvas { linear_light: true, ..Canvas::new(l.hud_w, l.hud_h) };
        hc.image = images.add(new_image(l.hud_w as u32, l.hud_h as u32, crisp));
        hc.crisp = crisp;
        node.image = hc.image.clone();
    }
    let now = time.elapsed_secs();
    let rnow = real.elapsed_secs_f64();
    // popups: world points projected to HUD units, with Python's rate limit
    for p in incoming.read() {
        if popups.live.last().is_some_and(|x| now - x.0 < 0.3 && x.2 != p.text) {
            continue;
        }
        popups.live.push((now, p.pos, p.text.clone(), p.color));
        if popups.live.len() > 24 {
            popups.live.remove(0);
        }
    }
    popups.live.retain(|x| now - x.0 < 0.9);
    let sf = window.scale_factor();
    let to_hud = |v: Vec2| (v.x * sf / l.scale, v.y * sf / l.scale);
    let tags_on = cfg.0.tags_on();
    let hud_popups: Vec<ktf_ui::hud::Popup> = popups
        .live
        .iter()
        .filter_map(|(born, pos, word, c)| {
            let age = (now - born) / 0.9;
            let v = camera.0.world_to_viewport(camera.1, *pos).ok()?;
            let (x, y) = to_hud(v);
            let c = c.to_srgba();
            let src = crate::reactions::popup_source(word).map(String::from);
            Some(ktf_ui::hud::Popup {
                x,
                y,
                text: word.clone(),
                color: [(c.red * 255.0) as u8, (c.green * 255.0) as u8, (c.blue * 255.0) as u8],
                age,
                source: src,
            })
        })
        .collect();
    let fly = flies.iter().next();
    let (health, dead) = fly.map_or((0.0, true), |f| (f.health, f.dead()));
    // the hotbar: the loadout's page of ten (`Game._draw_toolbar`), keys from the rebindable slot actions
    let slots: Vec<ToolSlot> = loadout
        .0
        .page_tools()
        .iter()
        .enumerate()
        .filter_map(|(k, name)| {
            TOOLS.iter().find(|t| t.0 == name).map(|t| ToolSlot {
                name: t.0.into(),
                label: t.1.into(),
                key: cfg.0.key_of(&format!("slot{}", k + 1)).into(),
            })
        })
        .collect();
    let sel = TOOLS[tools.selected];
    // the key shown in the line above the bar: its slot, or "page N" on another page of a long loadout, or "wheel" when it is on no page
    let key_label = match loadout.0.slot_of(sel.0) {
        Some(slot) => cfg.0.key_of(&format!("slot{}", slot + 1)).to_uppercase(),
        None => match loadout.0.tools.iter().position(|t| t == sel.0) {
            Some(i) => format!("PAGE {}", i / 10 + 1),
            None => "WHEEL".into(),
        },
    };
    let hint = if sel.0 == "laser" {
        format!("{}  [{}  {}]", sel.2, LASER_TARGETS[tools.laser_target].0, if tools.laser_silence { "SILENCE" } else { "ACTIVATE" })
    } else {
        sel.2.to_string()
    };
    let panel = (view.mode != PanelMode::Hidden).then(|| {
        let s = snap.0.as_deref();
        let idx = |name: &str| s.and_then(|s| s.info.as_deref()).and_then(|i| i.group_names.iter().position(|n| n == name));
        let hz = |name: &str| idx(name).and_then(|i| s.and_then(|s| s.hz.get(i).copied())).unwrap_or(0.0);
        let lvl = |name: &str| idx(name).and_then(|i| s.and_then(|s| s.levels.get(i).copied())).unwrap_or(0.0);
        PanelData {
            dead: s.is_some_and(|s| s.dead),
            calm_fx: false,
            neurons: s.map_or(0, |s| s.neurons),
            big_view: view.big,
            touch: [hz("head"), hz("body"), hz("legs"), hz("wing")],
            motor: MOTOR_ROWS.iter().map(|(n, label, th)| MotorRow { label: (*label).into(), level: lvl(n), thresh: *th }).collect(),
            whole_trace: s.map_or(vec![], |s| s.whole_trace.clone()),
            log: notes
                .lines
                .iter()
                .map(|(born, text, tag)| LogLine {
                    age: now - born,
                    text: text.clone(),
                    source: match tag {
                        Tag::Connectome => Some("real".into()),
                        Tag::GameRule => Some("rule".into()),
                        Tag::Model => None,
                    },
                })
                .collect(),
            steps_per_s: s.map_or(0.0, |s| s.steps_per_s),
            step_ms: s.map_or(0.0, |s| s.step_ms),
        }
    });
    let reward_level = {
        let s = snap.0.as_deref();
        s.and_then(|s| s.info.as_deref())
            .and_then(|i| i.group_names.iter().position(|n| n == "reward"))
            .and_then(|i| s.and_then(|s| s.levels.get(i).copied()))
            .unwrap_or(1.0)
    };
    let (pain_level, pain_neurons) = snap.0.as_deref().map_or((0, 0), |s| (s.pain_level as usize, s.pain_neurons));
    let state = HudState {
        fly_state: fly.map_or("waiting".into(), |f| pygame_state(f, now, arena.name())),
        hits: st.hits,
        kills: st.kills,
        flies: 1,
        focus: 0,
        max_flies: 16,
        health,
        max_health: MAX_HEALTH,
        immortal: settings.immortal,
        fly_dead: dead,
        arena: arena.name().into(),
        surgery_on: false,
        muted: settings.mute,
        pain: st.pain,
        pain_parts: st.pain_parts,
        pain_trace: st.pain_trace.clone(),
        pain_peak: st.pain_peak,
        pain_max_s: st.pain_max_s,
        pain_level_name: fly_sim::groups::PAIN_LEVELS[pain_level.min(2)].0.into(),
        pain_level,
        pain_neurons,
        pain_connectome: false,
        reward: st.reward,
        reward_level,
        memory: vec![],
        tools: slots,
        tool_name: sel.0.into(),
        tool_line: format!("{key_label}  {}: {hint}", sel.1),
        page: loadout.0.page,
        pages: loadout.0.n_pages(),
        look: look.captured,
        free_mouse_key: cfg.0.key_of("free_mouse").into(),
        overlay_open: st.help || view.big || arcade.open,
        menu_open: menu.0,
        dead_player: false,
        crosshair_hot: false,
        hide_hud: cfg.0.get_bool("brain.autopilot") && cfg.0.get_bool("brain.autopilot_hide_hud"),
        tags_on,
        panel,
        panel_alpha: match view.mode {
            PanelMode::Solid => 255,
            PanelMode::SeeThrough => 150,
            PanelMode::Faint => 70,
            PanelMode::Hidden => 0,
        },
        popups: hud_popups,
        time_label: if tc.menu_paused {
            None
        } else if tc.paused {
            Some(("PAUSED  (Z resume, . step)".into(), true))
        } else if tc.speed < 1.0 {
            Some((format!("SLOW-MO {}x  ([ ] speed)", ktf_data::sharecode::g(tc.speed as f64)), false))
        } else {
            None
        },
        help_open: st.help,
        ..Default::default()
    };
    let fonts = HudFonts::new(cfg.0.get_bool("access.larger_text"));
    let HudCanvas { canvas, image, .. } = &mut *hc;
    canvas.clear();
    Hud { cv: canvas, lay: l, f: &fonts }.draw(&state, rnow);
    if let Some(mut img) = images.get_mut(&*image) {
        img.data = Some(canvas.px.clone());
    }
}

fn keys(act: Res<crate::actions::Actions>, keys: Res<ButtonInput<KeyCode>>, mut st: ResMut<HudStats>) {
    if act.just("help") {
        st.help = !st.help;
    }
    if keys.just_pressed(KeyCode::Escape) && st.help {
        st.help = false;
    }
}

pub struct HudCanvasPlugin;

impl Plugin for HudCanvasPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HudLayout>()
            .init_resource::<HudStats>()
            .add_systems(Startup, setup)
            .add_systems(PreUpdate, layout.after(bevy::input::InputSystems))
            .add_systems(FixedUpdate, update_stats)
            .add_systems(Update, (keys, count_events, draw).chain());
    }
}
