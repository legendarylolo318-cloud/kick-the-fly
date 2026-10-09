//! The pause menu and Settings in the game: `ktf_ui::Menu` (the pygame menu ported onto a software canvas) shown as a full-window image.
//!
//! Esc opens it (and frees the mouse, and stops game time and the brain, like Python's `clock.menu_paused`); while it is open it gets
//! every key and mouse event and the game's own input is cleared, so typing a setting never resets the fly. Settings changes are applied
//! live and saved to the player's `config.toml`, the same file the Python game used.

use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonState;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::{MouseButtonInput, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::{MonitorSelection, PrimaryWindow, WindowMode};
use ktf_data::config::Config;
use ktf_ui::menu::PageHost;
use ktf_ui::{Canvas, Effect, Menu, UiEvent};
use serde_json::Value;

use crate::app::SimLink;
use crate::config::Settings;
use crate::controls::TimeControl;
use crate::hud_canvas::HudLayout;
use crate::player::MouseLook;
use crate::sim_thread::Command;

/// Set of the menu's input system: hotkeys read after it, so a menu that is open has already cleared the keys.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct MenuInputSet;

/// The player's `config.toml` (all 75 settings, keys, loadouts): the one the Python game wrote.
#[derive(Resource)]
pub struct UserConfig(pub Config);

impl UserConfig {
    /// `Game.set_setting(key, value)` for a hotkey: store (validated) and write the config file.
    pub fn set_and_save(&mut self, key: &str, value: Value) {
        if self.0.set(key, &value).is_ok() {
            self.0.save(env!("CARGO_PKG_VERSION"));
        }
    }
}

/// Whether the menu is open (other systems may want to stand back).
#[derive(Resource, Default)]
pub struct MenuOpen(pub bool);

/// `--menu pause` / `--menu settings:Audio`: open this screen at start (screenshots).
#[derive(Resource)]
pub struct MenuStart(pub String);

#[derive(Resource)]
struct MenuUi {
    menu: Menu,
    canvas: Canvas,
    image: Handle<Image>,
    node: Entity,
}

#[derive(Component)]
struct MenuNode;

fn new_image(w: u32, h: u32, nearest: bool) -> Image {
    let mut img = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        vec![0; (w * h * 4) as usize],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    img.sampler = if nearest { bevy::image::ImageSampler::nearest() } else { bevy::image::ImageSampler::linear() };
    img
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    window: Single<&Window, With<PrimaryWindow>>,
    cfg: Res<UserConfig>,
    start: Option<Res<MenuStart>>,
    mut open: ResMut<MenuOpen>,
    mut tc: ResMut<TimeControl>,
    mut look: ResMut<MouseLook>,
) {
    ktf_ui::i18n::set_language(cfg.0.get_str("access.language"));
    let (w, h) = (window.width().max(1.0) as u32, window.height().max(1.0) as u32);
    let image = images.add(new_image(w, h, true));
    let node = commands
        .spawn((
            MenuNode,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            ImageNode::new(image.clone()),
            GlobalZIndex(1000),
            Visibility::Hidden,
        ))
        .id();
    let mut menu = Menu::new(env!("CARGO_PKG_VERSION"));
    ktf_ui::pages::install(&mut menu);
    if let Some(s) = start {
        let (screen, tab) = s.0.split_once(':').unwrap_or((s.0.as_str(), ""));
        menu.show(screen);
        if !tab.is_empty() {
            menu.tab = tab.into();
        }
        open.0 = true;
        tc.menu_paused = true;
        look.captured = false;
    }
    commands.insert_resource(MenuUi { menu, canvas: Canvas { linear_light: true, ..Canvas::new(w as i32, h as i32) }, image, node });
}

/// `pygame.key.name` of a key, for the key-binding screen and Esc/Enter/Backspace handling.
pub fn pygame_key_name(k: KeyCode) -> Option<&'static str> {
    use KeyCode::*;
    Some(match k {
        KeyA => "a",
        KeyB => "b",
        KeyC => "c",
        KeyD => "d",
        KeyE => "e",
        KeyF => "f",
        KeyG => "g",
        KeyH => "h",
        KeyI => "i",
        KeyJ => "j",
        KeyK => "k",
        KeyL => "l",
        KeyM => "m",
        KeyN => "n",
        KeyO => "o",
        KeyP => "p",
        KeyQ => "q",
        KeyR => "r",
        KeyS => "s",
        KeyT => "t",
        KeyU => "u",
        KeyV => "v",
        KeyW => "w",
        KeyX => "x",
        KeyY => "y",
        KeyZ => "z",
        Digit0 => "0",
        Digit1 => "1",
        Digit2 => "2",
        Digit3 => "3",
        Digit4 => "4",
        Digit5 => "5",
        Digit6 => "6",
        Digit7 => "7",
        Digit8 => "8",
        Digit9 => "9",
        F1 => "f1",
        F2 => "f2",
        F3 => "f3",
        F4 => "f4",
        F5 => "f5",
        F6 => "f6",
        F7 => "f7",
        F8 => "f8",
        F9 => "f9",
        F10 => "f10",
        F11 => "f11",
        F12 => "f12",
        Escape => "escape",
        Enter => "return",
        NumpadEnter => "enter",
        Backspace => "backspace",
        Delete => "delete",
        Tab => "tab",
        Space => "space",
        ShiftLeft => "left shift",
        ShiftRight => "right shift",
        ControlLeft => "left ctrl",
        ControlRight => "right ctrl",
        AltLeft => "left alt",
        AltRight => "right alt",
        ArrowUp => "up",
        ArrowDown => "down",
        ArrowLeft => "left",
        ArrowRight => "right",
        Minus => "-",
        Equal => "=",
        BracketLeft => "[",
        BracketRight => "]",
        Semicolon => ";",
        Quote => "'",
        Comma => ",",
        Period => ".",
        Slash => "/",
        Backquote => "`",
        Backslash => "\\",
        Home => "home",
        End => "end",
        PageUp => "page up",
        PageDown => "page down",
        Insert => "insert",
        _ => return None,
    })
}

#[allow(clippy::too_many_arguments)]
fn input(
    mut ui: ResMut<MenuUi>,
    mut cfg: ResMut<UserConfig>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut buttons: ResMut<ButtonInput<MouseButton>>,
    mut key_events: MessageReader<KeyboardInput>,
    mut button_events: MessageReader<MouseButtonInput>,
    mut wheel: MessageReader<MouseWheel>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut look: ResMut<MouseLook>,
    mut tc: ResMut<TimeControl>,
    mut open: ResMut<MenuOpen>,
    mut settings: ResMut<Settings>,
    arcade: Res<crate::arcade::Arcade>,
    link: Res<SimLink>,
    mut exit: MessageWriter<AppExit>,
    lay: Res<HudLayout>,
) {
    let was_open = ui.menu.is_open();
    let k = window.scale_factor() / lay.0.scale;
    let mouse = window.cursor_position().map_or((-1, -1), |p| ((p.x * k) as i32, (p.y * k) as i32));
    let mut events: Vec<UiEvent> = vec![];
    if was_open {
        events.push(UiEvent::Motion { pos: mouse });
        for e in key_events.read() {
            if e.state != ButtonState::Pressed {
                continue;
            }
            let name = pygame_key_name(e.key_code).map(String::from).unwrap_or_default();
            let ch = e.text.as_ref().and_then(|t| t.chars().next()).filter(|c| !c.is_control());
            let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
            events.push(UiEvent::KeyDown { name, ch, ctrl });
        }
        for e in button_events.read() {
            if e.button == MouseButton::Left {
                events.push(if e.state == ButtonState::Pressed {
                    UiEvent::MouseDown { pos: mouse }
                } else {
                    UiEvent::MouseUp { pos: mouse }
                });
            }
        }
        for w in wheel.read() {
            events.push(UiEvent::Wheel { y: if w.unit == MouseScrollUnit::Line { w.y } else { w.y / 20.0 } });
        }
    } else {
        key_events.clear();
        button_events.clear();
        wheel.clear();
        if keys.just_pressed(KeyCode::Escape) && !arcade.open {
            ui.menu.show("pause");
            look.captured = false;
            tc.menu_paused = true;
            link.0.send(Command::Pause(true));
            open.0 = true;
        }
    }
    let three_d = true;
    let mut effects = vec![];
    for ev in events {
        let (_, fx) = ui.menu.handle(&ev, &mut cfg.0, three_d, &mut GameHost);
        effects.extend(fx);
    }
    for fx in effects {
        match fx {
            Effect::Action(a) => match a.as_str() {
                "resume" => ui.menu.close(),
                "closed" => {
                    open.0 = false;
                    tc.menu_paused = false;
                    link.0.send(Command::Pause(tc.paused));
                    look.captured = true;
                }
                "settings" => ui.menu.show("settings"),
                "quit" => ui.menu.show("confirm_quit"),
                "quit_now" => {
                    cfg.0.save(env!("CARGO_PKG_VERSION"));
                    exit.write(AppExit::Success);
                }
                "toggle_mode" => {
                    let next = match cfg.0.get_str("brain.mode") {
                        "play" => "lab",
                        "lab" => "pet",
                        _ => "play",
                    };
                    let _ = cfg.0.set("brain.mode", &Value::from(next));
                    cfg.0.save(env!("CARGO_PKG_VERSION"));
                    let mut label = next.to_string();
                    label[..1].make_ascii_uppercase();
                    ui.menu.flash(&format!("{label} mode"), ktf_ui::theme::GOOD, 4.0);
                }
                "click_sound" => {}
                // the pause menu's buttons open the page of the same name (Python: `menu_action`); minipapers is the Lab's page
                "minipapers" if ui.menu.has_page("lab_minipapers") => ui.menu.show("lab_minipapers"),
                name if ui.menu.has_page(name) => ui.menu.show(name),
                _ => ui.menu.flash("coming soon", ktf_ui::theme::AMBER, 4.0),
            },
            Effect::SettingChanged { key, save } => {
                settings.apply_config(&cfg.0);
                if let Some(k) = key.as_deref() {
                    apply_live(k, &cfg.0, &mut window, &mut tc, &link);
                }
                if save {
                    cfg.0.save(env!("CARGO_PKG_VERSION"));
                }
            }
            Effect::KeysChanged => {
                cfg.0.save(env!("CARGO_PKG_VERSION"));
            }
        }
    }
    if was_open || ui.menu.is_open() {
        // the menu owns the input this frame: the game must not also react to the same keys and clicks
        keys.reset_all();
        buttons.reset_all();
    }
}

/// A setting changed in the menu: do what it means right away.
fn apply_live(key: &str, cfg: &Config, window: &mut Window, tc: &mut TimeControl, link: &SimLink) {
    match key {
        "graphics.fullscreen" => {
            window.mode =
                if cfg.get_bool(key) { WindowMode::BorderlessFullscreen(MonitorSelection::Current) } else { WindowMode::Windowed };
        }
        "brain.sim_speed" => {
            tc.speed = cfg.get_f32(key);
            link.0.send(Command::SetSpeed(tc.speed));
        }
        "brain.pain_level" => {
            link.0.send(Command::PainLevel(cfg.get_i64(key).clamp(0, 2) as usize));
        }
        "access.language" => {
            ktf_ui::i18n::set_language(cfg.get_str(key));
        }
        _ => {}
    }
}

fn draw(
    mut ui: ResMut<MenuUi>,
    mut cfg: ResMut<UserConfig>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut vis: Query<(&mut Visibility, &mut ImageNode), With<MenuNode>>,
    time: Res<Time<Real>>,
    lay: Res<HudLayout>,
) {
    let Ok((mut v, mut node)) = vis.single_mut() else { return };
    if !ui.menu.is_open() {
        if *v != Visibility::Hidden {
            *v = Visibility::Hidden;
        }
        return;
    }
    let (w, h) = (lay.0.hud_w, lay.0.hud_h);
    if (ui.canvas.w, ui.canvas.h) != (w, h) {
        ui.canvas = Canvas { linear_light: true, ..Canvas::new(w, h) };
        ui.image = images.add(new_image(w as u32, h as u32, lay.0.scale.fract() == 0.0));
        node.image = ui.image.clone();
    }
    let k = window.scale_factor() / lay.0.scale;
    let mouse = window.cursor_position().map_or((-1, -1), |p| ((p.x * k) as i32, (p.y * k) as i32));
    let MenuUi { menu, canvas, image, .. } = &mut *ui;
    canvas.clear();
    menu.draw(canvas, &mut cfg.0, true, mouse, time.elapsed_secs_f64(), &mut GameHost);
    if let Some(mut img) = images.get_mut(&*image) {
        img.data = Some(canvas.px.clone());
    }
    *v = Visibility::Visible;
}

/// What the player's config says about the brain at start: the pain level and the simulation speed.
fn apply_config_at_start(cfg: Res<UserConfig>, link: Res<SimLink>, mut tc: ResMut<TimeControl>) {
    link.0.send(Command::PainLevel(cfg.0.get_i64("brain.pain_level").clamp(0, 2) as usize));
    let speed = cfg.0.get_f32("brain.sim_speed");
    if (speed - 1.0).abs() > 1e-6 {
        tc.speed = speed;
        link.0.send(Command::SetSpeed(speed));
    }
}

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<UserConfig>() {
            app.insert_resource(UserConfig(Config::new(None)));
        }
        app.init_resource::<MenuOpen>()
            .add_systems(Startup, (setup, apply_config_at_start))
            .add_systems(PreUpdate, input.in_set(MenuInputSet).after(bevy::input::InputSystems))
            .add_systems(Update, draw);
        let _ = |ui: &MenuUi| ui.node;
    }
}

/// What the menu pages ask of the game (`ktf_ui::PageHost`): the facts the share page validates against, the clipboard, and applying a code.
/// Only what this build has is answered; the rest comes back as null or an error that says so.
pub struct GameHost;

fn stamp() -> String {
    // UTC date and time as YYYYMMDD-HHMMSS (the Python game used local time)
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let (y, doy) = (yoe + era * 400, doe - (365 * yoe + yoe / 4 - yoe / 100));
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

impl PageHost for GameHost {
    fn query(&self, key: &str) -> Value {
        use serde_json::json;
        let schema = ktf_data::config::schema();
        match key {
            "share_context" => json!({
                "tools": schema.tools, "max_saved_loadouts": schema.max_saved_loadouts, "brain": "adult",
                "arenas": crate::world::ARENAS, "contraption_slots_used": 0,
            }),
            "surgery_payload" => json!({"groups": {}, "types": {}}),
            "loadout_tools" => json!(schema.tools.iter().take(10).collect::<Vec<_>>()),
            "lab_modified" => json!({}),
            "protocol_files" => json!([]),
            "challenge_keys" => json!([]),
            "clipboard_get" => json!(arboard::Clipboard::new().and_then(|mut c| c.get_text()).unwrap_or_default()),
            "now_stamp" => json!(stamp()),
            _ => Value::Null,
        }
    }

    fn command(&mut self, cmd: &str, args: Value) -> Value {
        use serde_json::json;
        match cmd {
            "clipboard_put" => json!(arboard::Clipboard::new().and_then(|mut c| c.set_text(args.as_str().unwrap_or(""))).is_ok()),
            "share_apply" => json!({"error": format!("this build can't apply a {} code yet", args["kind"].as_str().unwrap_or("?"))}),
            "protocol_payload" => json!({"error": "protocols are not part of this build yet"}),
            _ => Value::Null,
        }
    }
}
