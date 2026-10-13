//! Crash handling (`core/crash.py`, `ui/crashscreen.py`) and the diagnostics host shared by the game's menu and the crash window.
//!
//! A panic on the main thread or the brain thread writes `KickTheFly-crash.txt` (beside the exe when that folder is writable, and into the
//! state folder), then starts the same program as `kick_the_fly --crash-screen FILE...`: a small window of its own (the game's may be
//! gone with its GPU context) that says where the report went and offers Report a bug, the same review screen as Settings > Help.
//! `KICK_THE_FLY_NO_CRASH_SCREEN=1` skips the window (CI, wrappers). `KICK_THE_FLY_DEBUG_CRASH=SECONDS` panics after that long, for
//! testing all of it. Nothing is ever sent anywhere.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Mutex, OnceLock};

use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::input::ButtonState;
use bevy::input::keyboard::KeyboardInput;
use bevy::input::mouse::{MouseButtonInput, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;
use ktf_data::config::Config;
use ktf_data::diag;
use ktf_ui::menu::PageHost;
use ktf_ui::{Canvas, Effect, Menu, UiEvent};
use serde_json::{Value, json};

use crate::selftest;

const SIZE: (u32, u32) = (900, 620);

fn crash_paths() -> &'static Mutex<Vec<PathBuf>> {
    static P: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    P.get_or_init(Default::default)
}

// --- what the diagnostics pages ask of the program --------------------------------------------------------------------------------
/// Answers for `selftest`, `app_version` and `crash_paths`; None for any other key.
pub fn diag_query(key: &str) -> Option<Value> {
    Some(match key {
        "selftest" => selftest::page_json(),
        "app_version" => json!(selftest::app_version()),
        "crash_paths" => json!(crash_paths().lock().unwrap().iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>()),
        _ => return None,
    })
}

/// Hand a URL to the player's browser. The browser does the talking; this program sends nothing.
pub fn open_url(url: &str) -> bool {
    use std::process::{Command, Stdio};
    let mut cmd;
    if cfg!(windows) {
        cmd = Command::new("rundll32");
        cmd.args(["url.dll,FileProtocolHandler", url]);
    } else if cfg!(target_os = "macos") {
        cmd = Command::new("open");
        cmd.arg(url);
    } else {
        cmd = Command::new("xdg-open");
        cmd.arg(url);
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().is_ok()
}

/// Commands `selftest_run`, `open_url` and `clipboard_put`; None for any other.
pub fn diag_command(cmd: &str, args: &Value) -> Option<Value> {
    Some(match cmd {
        "selftest_run" => {
            selftest::start_page_run();
            Value::Null
        }
        "open_url" => json!(open_url(args.as_str().unwrap_or(""))),
        "clipboard_put" => json!(arboard::Clipboard::new().and_then(|mut c| c.set_text(args.as_str().unwrap_or(""))).is_ok()),
        _ => return None,
    })
}

// --- the log file -----------------------------------------------------------------------------------------------------------------
static LOG: OnceLock<std::sync::Arc<std::fs::File>> = OnceLock::new();

/// Start this session's `kickthefly.log` (the last one is kept as `kickthefly.previous.log`), before the window opens.
pub fn open_log() {
    if let Some(f) = diag::open_log(&ktf_data::paths::get().state_dir) {
        let _ = LOG.set(std::sync::Arc::new(f));
    }
}

/// `LogPlugin::custom_layer`: tee every log line into the file (Report a bug reads the tail of it).
pub fn log_layer(_app: &mut App) -> Option<bevy::log::BoxedLayer> {
    let f = LOG.get()?.clone();
    Some(Box::new(bevy::log::tracing_subscriber::fmt::layer().with_ansi(false).with_writer(f)))
}

pub struct CrashPlugin;

impl Plugin for CrashPlugin {
    fn build(&self, app: &mut App) {
        diag::set_info("mode", "3d");
        app.add_systems(Update, (debug_crash_system, record_gpu));
    }
}

/// Remember the GPU the renderer chose, for crash reports and bug reports.
fn record_gpu(info: Option<Res<bevy::render::renderer::RenderAdapterInfo>>, mut done: Local<bool>) {
    if *done {
        return;
    }
    if let Some(i) = info {
        *done = true;
        diag::set_info("gpu_name", i.name.clone());
        diag::set_info("gpu_backend", format!("{:?}", i.backend));
        diag::set_info("gpu_driver", format!("{} {}", i.driver, i.driver_info).trim().to_string());
        let kind = format!("{:?}", i.device_type);
        bevy::log::info!("graphics: {} via {:?} ({kind}), driver {} {}", i.name, i.backend, i.driver, i.driver_info);
        if kind == "Cpu" || kind == "IntegratedGpu" {
            bevy::log::warn!(
                "graphics: rendering on a {kind} adapter ({}); set WGPU_ADAPTER_NAME to the name of the graphics card to use",
                i.name
            );
        }
    }
}

// --- the panic hook ---------------------------------------------------------------------------------------------------------------
/// Write a crash report for every fatal panic and (windowed runs) show the crash window. Fatal = the main thread, the brain thread or a thread of Bevy's compute pool (where systems run); a
/// panic a worker catches (the self-test's, an asset task's) is not a crash.
pub fn install_panic_hook(windowed: bool) {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("");
        if name != "main" && name != "fly-sim" && name != crate::hud_canvas::HUD_THREAD && !name.contains("Compute Task Pool") {
            return;
        }
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "(no message)".into());
        let loc = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
        let what = format!("panic in thread '{name}': {msg}\n  at {loc}\n\n{}", std::backtrace::Backtrace::force_capture());
        let written = diag::write_crash_report(selftest::app_version(), &what, None);
        eprintln!(
            "crashed; report written to {}",
            if written.is_empty() {
                "nowhere (no writable folder)".into()
            } else {
                written.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
            }
        );
        if windowed && std::env::var_os("KICK_THE_FLY_NO_CRASH_SCREEN").is_none() {
            show_crash_screen(&written);
        }
        std::process::exit(101);
    }));
}

/// Start this program again as the crash window; it outlives this process (the game's own window goes away at once, as Python's does).
fn show_crash_screen(written: &[PathBuf]) {
    let Ok(exe) = std::env::current_exe() else { return };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--crash-screen").args(written).env("KICK_THE_FLY_NO_CRASH_SCREEN", "1");
    let _ = cmd.spawn();
}

/// `KICK_THE_FLY_DEBUG_CRASH=SECONDS`: panic from a system after that long.
pub fn debug_crash_system(time: Res<Time<Real>>) {
    if let Some(after) = std::env::var("KICK_THE_FLY_DEBUG_CRASH").ok().and_then(|v| v.parse::<f64>().ok())
        && time.elapsed_secs_f64() >= after
    {
        panic!("test crash requested by KICK_THE_FLY_DEBUG_CRASH={after}");
    }
}

// --- --bugreport ------------------------------------------------------------------------------------------------------------------
/// `--bugreport [--out DIR]`: print exactly what the report would include and write it to DIR. Sends nothing.
pub fn bugreport_main(out: Option<&Path>) -> i32 {
    // a quick self-test when this session has none (no GPU probing, no smoke run)
    let rep = selftest::run(&selftest::Options { smoke: false, graphics: false }, &mut |_| {});
    let items = diag::collect(selftest::app_version(), serde_json::to_value(&rep).ok().as_ref(), &ktf_data::paths::get().state_dir);
    let text = diag::clipboard_text(&items, "");
    println!("{text}");
    match out {
        Some(dir) => {
            let _ = std::fs::create_dir_all(dir);
            let p = dir.join("bugreport.md");
            match std::fs::write(&p, &text) {
                Ok(()) => println!("written to {} (nothing was sent anywhere)", p.display()),
                Err(e) => eprintln!("could not write {}: {e}", p.display()),
            }
        }
        None => println!("(nothing was sent anywhere; use --out DIR to save this, and attach it to an issue by hand)"),
    }
    0
}

// --- the crash window -------------------------------------------------------------------------------------------------------------
struct CrashHost;

impl PageHost for CrashHost {
    fn query(&self, key: &str) -> Value {
        diag_query(key).unwrap_or(Value::Null)
    }
    fn command(&mut self, cmd: &str, args: Value) -> Value {
        diag_command(cmd, &args).unwrap_or(Value::Null)
    }
}

#[derive(Resource)]
struct CrashUi {
    menu: Menu,
    canvas: Canvas,
    image: Handle<Image>,
    cfg: Config,
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let (w, h) = SIZE;
    let mut img = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        vec![0; (w * h * 4) as usize],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    img.sampler = bevy::image::ImageSampler::nearest();
    let image = images.add(img);
    commands.spawn(Camera2d);
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        ImageNode::new(image.clone()),
    ));
    let cfg = Config::load(&ktf_data::paths::get().config_file());
    ktf_ui::i18n::set_language(cfg.get_str("access.language"));
    let mut menu = Menu::new(selftest::app_version());
    menu.register_page("crash", Box::new(ktf_ui::pages::diag::CrashPage));
    menu.register_page("selftest", Box::new(ktf_ui::pages::diag::SelfTestPage));
    menu.register_page("bugreport", Box::<ktf_ui::pages::diag::BugReportPage>::default());
    menu.show("crash");
    commands.insert_resource(CrashUi { menu, canvas: Canvas { linear_light: true, ..Canvas::new(w as i32, h as i32) }, image, cfg });
}

#[allow(clippy::too_many_arguments)]
fn frame(
    mut ui: ResMut<CrashUi>,
    window: Single<&Window, With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut key_events: MessageReader<KeyboardInput>,
    mut button_events: MessageReader<MouseButtonInput>,
    mut wheel: MessageReader<MouseWheel>,
    mut images: ResMut<Assets<Image>>,
    time: Res<Time<Real>>,
    mut exit: MessageWriter<AppExit>,
) {
    let mouse = window.cursor_position().map_or((-1, -1), |p| (p.x as i32, p.y as i32));
    let mut events = vec![UiEvent::Motion { pos: mouse }];
    for e in key_events.read() {
        if e.state != ButtonState::Pressed {
            continue;
        }
        if e.key_code == KeyCode::Escape && ui.menu.screen.as_deref() == Some("crash") {
            exit.write(AppExit::Success);
            return;
        }
        let name = crate::menu_ui::pygame_key_name(e.key_code).map(String::from).unwrap_or_default();
        let ch = e.text.as_ref().and_then(|t| t.chars().next()).filter(|c| !c.is_control());
        let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
        events.push(UiEvent::KeyDown { name, ch, ctrl });
    }
    for e in button_events.read() {
        if e.button == MouseButton::Left {
            events.push(if e.state == ButtonState::Pressed { UiEvent::MouseDown { pos: mouse } } else { UiEvent::MouseUp { pos: mouse } });
        }
    }
    for w in wheel.read() {
        events.push(UiEvent::Wheel { y: if w.unit == MouseScrollUnit::Line { w.y } else { w.y / 20.0 } });
    }
    let CrashUi { menu, canvas, image, cfg } = &mut *ui;
    for ev in &events {
        let (_, fx) = menu.handle(ev, cfg, false, &mut CrashHost);
        for f in fx {
            if matches!(&f, Effect::Action(a) if a == "crash_quit") {
                exit.write(AppExit::Success);
            }
        }
    }
    if !menu.is_open() {
        menu.show("crash");
    }
    canvas.clear();
    canvas.fill_rect(ktf_ui::canvas::Rect::new(0, 0, SIZE.0 as i32, SIZE.1 as i32), ktf_ui::theme::BG, 0);
    menu.draw(canvas, cfg, false, mouse, time.elapsed_secs_f64(), &mut CrashHost);
    if let Some(mut img) = images.get_mut(&*image) {
        img.data = Some(canvas.px.clone());
    }
}

/// `kick_the_fly --crash-screen FILE...`: the crash window. `files` are the crash reports that were written.
pub fn run_crash_screen(files: Vec<PathBuf>) -> ExitCode {
    *crash_paths().lock().unwrap() = files;
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window { title: "Kick the Fly".into(), resolution: SIZE.into(), resizable: false, ..default() }),
                    ..default()
                })
                .set(bevy::render::RenderPlugin { render_creation: crate::gpu::wgpu_settings().into(), ..default() }),
        )
        .add_systems(Startup, setup)
        .add_systems(Update, frame.run_if(resource_exists::<CrashUi>))
        .run();
    ExitCode::SUCCESS
}
