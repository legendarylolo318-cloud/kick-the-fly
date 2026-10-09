//! The on-screen display: the hotbar, the reaction log with its honesty tags, floating popups, health, the crosshair and the help page.
//! Python: `Game3D.draw_hud3d` and `Game.note`. Everything is Bevy UI text and nodes; popups are projected from the world each frame.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::app::LatestSnapshot;
use crate::brainview::BrainView;
use crate::config::{Arena, Settings};
use crate::feedback::{Notes, Popup, Tag};
use crate::fly::{Fly, MAX_HEALTH};
use crate::player::PlayerCamera;
use crate::sim_thread::Command;
use crate::tools::{LASER_TARGETS, TOOLS, ToolState};

const POPUP_POOL: usize = 24;

#[derive(Component)]
struct StatusText;
#[derive(Component)]
struct HotbarSlot(usize);
#[derive(Component)]
struct HintText;
#[derive(Component)]
struct NotesText(usize);
#[derive(Component)]
struct PopupText(usize);
#[derive(Component)]
struct HealthBar;
#[derive(Component)]
struct HelpPanel;
#[derive(Component)]
struct Crosshair;

#[derive(Resource, Default)]
struct PopupState {
    live: Vec<(f32, Vec3, String, Color)>,
}

pub const HELP: &[(&str, &str)] = &[
    ("WASD / mouse", "walk (Shift sprint, Ctrl crouch) and look; Tab frees the mouse"),
    ("Left click", "use the tool in your hand (hold for torch, cleaner, freezer, laser, the hand)"),
    ("1-9, 0 / wheel", "pick a tool; - and = turn the page of the 18 tools"),
    ("; and '", "laser: next target cell type, activate or silence"),
    ("E", "next arena: room, fan, flypaper, pool, lamp, escape room, field, orchard, thermo, kitchen"),
    ("B / V", "big brain view / brain panel: solid, see-through, faint, hidden"),
    ("F4", "fly's-eye view (visual only, not fed to the brain)"),
    ("F3", "profiler: FPS, frame time, sim steps/s, sim lag, draws"),
    ("P / Z / [ ] / .", "pause the brain / pause time / slower, faster / single step"),
    ("I / R / M", "immortal mode / reset the fly / mute"),
    ("O", "puppeteer, lesion battle, tournament and race (the arcade) menu"),
    ("K", "kill the fly (the kill cam replays its last seconds)"),
    ("H", "this help"),
    ("F12", "save a screenshot"),
    ("Esc", "close a panel, then quit"),
];

fn setup(mut commands: Commands) {
    commands.spawn((StatusText, Text::new(""), TextFont { font_size: FontSize::Px(15.0), ..default() }, TextColor(Color::srgb(0.9, 0.93, 0.98)), Node { position_type: PositionType::Absolute, top: px(10), left: px(12), ..default() }));
    // crosshair
    commands.spawn((Crosshair, Node { position_type: PositionType::Absolute, left: percent(50), top: percent(50), width: px(6), height: px(6), margin: UiRect { left: px(-3), top: px(-3), ..default() }, border_radius: BorderRadius::all(px(3)), ..default() }, BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.7))));
    // hotbar
    commands.spawn(Node { position_type: PositionType::Absolute, bottom: px(8), left: percent(50), margin: UiRect { left: px(-310), ..default() }, flex_direction: FlexDirection::Column, align_items: AlignItems::Center, row_gap: px(4), ..default() }).with_children(|p| {
        p.spawn((HintText, Text::new(""), TextFont { font_size: FontSize::Px(14.0), ..default() }, TextColor(Color::srgb(0.85, 0.9, 0.95))));
        p.spawn(Node { flex_direction: FlexDirection::Row, column_gap: px(4), ..default() }).with_children(|r| {
            for i in 0..10 {
                r.spawn((HotbarSlot(i), Node { width: px(58), height: px(40), justify_content: JustifyContent::Center, align_items: AlignItems::Center, border: UiRect::all(px(2)), ..default() }, BackgroundColor(Color::srgba(0.05, 0.06, 0.09, 0.75)), BorderColor::all(Color::srgb(0.25, 0.28, 0.35)))).with_children(|s| {
                    s.spawn((Text::new(""), TextFont { font_size: FontSize::Px(11.0), ..default() }, TextColor(Color::WHITE)));
                });
            }
        });
    });
    // reaction log
    commands.spawn(Node { position_type: PositionType::Absolute, bottom: px(100), left: px(12), flex_direction: FlexDirection::Column, row_gap: px(2), max_width: px(430), ..default() }).with_children(|p| {
        for i in 0..8 {
            p.spawn((NotesText(i), Text::new(""), TextFont { font_size: FontSize::Px(13.0), ..default() }, TextColor(Color::WHITE)));
        }
    });
    // health
    commands
        .spawn((Node { position_type: PositionType::Absolute, top: px(10), left: percent(50), margin: UiRect { left: px(-100), ..default() }, width: px(200), height: px(10), ..default() }, BackgroundColor(Color::srgba(0.1, 0.1, 0.12, 0.8))))
        .with_children(|p| {
            p.spawn((HealthBar, Node { width: percent(100), height: percent(100), ..default() }, BackgroundColor(Color::srgb(0.3, 0.85, 0.45))));
        });
    // popups
    for i in 0..POPUP_POOL {
        commands.spawn((PopupText(i), Text::new(""), TextFont { font_size: FontSize::Px(20.0), ..default() }, TextColor(Color::NONE), Node { position_type: PositionType::Absolute, display: Display::None, ..default() }));
    }
    // help
    let mut body = String::from("KICK THE FLY (Rust + Bevy)\n\n");
    for (k, v) in HELP {
        body += &format!("{k:<18} {v}\n");
    }
    commands.spawn((HelpPanel, Text::new(body), TextFont { font_size: FontSize::Px(15.0), ..default() }, TextColor(Color::srgb(0.92, 0.95, 1.0)), Node { position_type: PositionType::Absolute, left: percent(50), top: px(80), margin: UiRect { left: px(-330), ..default() }, padding: UiRect::all(px(16)), display: Display::None, ..default() }, BackgroundColor(Color::srgba(0.03, 0.04, 0.07, 0.92))));
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update(
    time: Res<Time<Virtual>>,
    snap: Res<LatestSnapshot>,
    arena: Res<Arena>,
    tools: Res<ToolState>,
    settings: Res<Settings>,
    view: Res<BrainView>,
    keys: Res<ButtonInput<KeyCode>>,
    notes: Res<Notes>,
    flies: Query<&Fly>,
    mut status: Single<&mut Text, (With<StatusText>, Without<HintText>)>,
    mut hint: Single<&mut Text, (With<HintText>, Without<StatusText>)>,
    mut slots: Query<(&HotbarSlot, &mut BorderColor, &mut BackgroundColor, &Children)>,
    mut texts: Query<&mut Text, (Without<StatusText>, Without<HintText>, Without<NotesText>, Without<PopupText>)>,
    mut note_lines: Query<(&NotesText, &mut Text, &mut TextColor), (Without<StatusText>, Without<HintText>, Without<PopupText>)>,
    mut health: Single<&mut Node, With<HealthBar>>,
    mut help: Single<&mut Node, (With<HelpPanel>, Without<HealthBar>)>,
) {
    let now = time.elapsed_secs();
    if keys.just_pressed(KeyCode::KeyH) {
        help.display = if help.display == Display::None { Display::Flex } else { Display::None };
    }
    let brain = snap.0.as_deref().map_or("waiting for the brain...".to_string(), |s| match &s.phase {
        crate::snapshot::Phase::Failed(m) => format!("brain failed: {m}"),
        crate::snapshot::Phase::Running | crate::snapshot::Phase::Paused => format!(
            "{} | {} neurons | {}  {:.0} steps/s  x{:.2} | {} spiking{}",
            s.source,
            s.neurons,
            if matches!(s.phase, crate::snapshot::Phase::Paused) { "PAUSED" } else { "running" },
            s.steps_per_s,
            s.speed,
            s.active_count,
            if s.dead { " | DEAD" } else { "" }
        ),
        _ => "loading the brain...".to_string(),
    });
    status.0 = format!("{}  |  arena: {}{}  |  brain panel: {}\n{brain}\nH help", "Kick the Fly", arena.name(), if settings.immortal { "  |  IMMORTAL" } else { "" }, view.mode.label());
    let t = TOOLS[tools.selected];
    hint.0 = if t.0 == "laser" {
        format!("{} {}: {}  [{}  {}]", tools.selected + 1, t.1, t.2, LASER_TARGETS[tools.laser_target].0, if tools.laser_silence { "SILENCE" } else { "ACTIVATE" })
    } else {
        format!("{} {}: {}", tools.selected + 1, t.1, t.2)
    };
    for (slot, mut border, mut bg, children) in &mut slots {
        let idx = tools.page * 10 + slot.0;
        let sel = idx == tools.selected;
        *border = BorderColor::all(if sel { Color::srgb(1.0, 0.8, 0.3) } else { Color::srgb(0.25, 0.28, 0.35) });
        bg.0 = if sel { Color::srgba(0.25, 0.2, 0.08, 0.9) } else { Color::srgba(0.05, 0.06, 0.09, 0.75) };
        for c in children.iter() {
            if let Ok(mut tx) = texts.get_mut(c) {
                tx.0 = TOOLS.get(idx).map_or(String::new(), |t| format!("{}\n{}", (slot.0 + 1) % 10, t.1));
            }
        }
    }
    let lines: Vec<_> = notes.lines.iter().collect();
    for (n, mut text, mut col) in &mut note_lines {
        match lines.get(n.0) {
            Some((born, line, tag)) => {
                let age = now - born;
                let a = (1.0 - ((age - 6.0) / 3.0).clamp(0.0, 1.0)).max(0.0);
                text.0 = format!("{}  [{}]", line, tag.label());
                col.0 = match tag {
                    Tag::Connectome => Color::srgba(0.55, 0.95, 0.65, a),
                    Tag::GameRule => Color::srgba(1.0, 0.82, 0.45, a),
                    Tag::Model => Color::srgba(0.7, 0.8, 1.0, a),
                };
            }
            None => text.0.clear(),
        }
    }
    if let Some(f) = flies.iter().next() {
        health.width = percent((f.health / MAX_HEALTH * 100.0).clamp(0.0, 100.0));
    }
}

/// Collect new popups and keep their world positions on screen.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn popups(
    time: Res<Time<Virtual>>,
    mut incoming: MessageReader<Popup>,
    mut state: Local<PopupState>,
    camera: Single<(&Camera, &GlobalTransform), With<PlayerCamera>>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut pool: Query<(&PopupText, &mut Text, &mut TextColor, &mut Node)>,
) {
    let now = time.elapsed_secs();
    for p in incoming.read() {
        // at most one popup every 0.3 s unless it is the same word (Python's rate limit), so a torch does not spam the screen
        if state.live.last().is_some_and(|l| now - l.0 < 0.3 && l.2 != p.text) {
            continue;
        }
        state.live.push((now, p.pos, p.text.clone(), p.color));
        if state.live.len() > POPUP_POOL {
            state.live.remove(0);
        }
    }
    state.live.retain(|l| now - l.0 < 0.9);
    let _ = window;
    for (slot, mut text, mut color, mut node) in &mut pool {
        match state.live.get(slot.0) {
            Some((born, pos, word, c)) => {
                let age = now - born;
                let world = *pos + Vec3::Y * 0.25 * age;
                match camera.0.world_to_viewport(camera.1, world) {
                    Ok(v) => {
                        text.0 = word.clone();
                        node.left = px(v.x - 40.0);
                        node.top = px(v.y - 12.0);
                        node.display = Display::Flex;
                        color.0 = c.with_alpha((1.0 - age / 0.9).clamp(0.0, 1.0));
                    }
                    Err(_) => node.display = Display::None,
                }
            }
            None => node.display = Display::None,
        }
    }
}

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(Update, (update, popups));
        let _ = Command::Revive;
    }
}
