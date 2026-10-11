#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]
//! `kick_the_fly [--pack FILE | --synthetic N] [--seed S] [--smoke SECONDS]`
//!
//! Opens the game window with the brain running on its own thread. `--smoke` runs the sim thread headless for a number of seconds
//! and prints how it did (exit code 2 if it could not keep real time), which is what CI uses on machines without a display.

use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use fly_sim::pack;
use kick_the_fly::app::SimPlugin;
use kick_the_fly::run_smoke;
use kick_the_fly::sim_thread::{SimConfig, Source};

struct Args {
    source: Source,
    seed: u64,
    smoke: Option<f64>,
    arena: Option<String>,
    shot: Option<(PathBuf, u32)>,
    cam: Option<(bevy::math::Vec3, bevy::math::Vec3)>,
    script: Option<(usize, f32, f32)>,
    rain: f32,
    storm: bool,
    day: f32,
    kill_at: Option<f32>,
    menu: Option<String>,
    fullscreen: bool,
    selftest: bool,
    train_screen: bool,
    surgery_screen: bool,
    inspect: Option<u32>,
    challenge: Option<String>,
    bugreport: bool,
    out: Option<PathBuf>,
    crash_screen: Option<Vec<PathBuf>>,
}

const USAGE: &str = "usage: kick_the_fly [--pack FILE | --synthetic N] [--seed S] [--smoke SECONDS]\n\
  --pack FILE      brain pack (default: $KICK_THE_FLY_PACK, else data/kick_brain.npz found upward from here)\n\
  --synthetic N    a random N-neuron network instead of the connectome (also what runs when no pack is found)\n\
  --seed S         noise seed (default 1000)\n\
  --smoke SECONDS  no window: run the brain thread for SECONDS and report steps per second\n\
  --arena NAME     start in this arena (room fan flypaper pool lamp escaperoom field orchard thermo kitchen)\n\
  --cam x,y,z,tx,ty,tz  fixed camera (screenshots)\n\
  --rain F --storm --day S   weather (outdoor arenas) and day/night cycle length\n\
  --fullscreen     start borderless fullscreen (a tiling compositor otherwise opens half-size windows)\n\
  --flies N        start with N flies (N spawns another, F cycles the focus, J recalls lost ones)
  --2d             accepted and ignored: the 2D game is not part of this build
  --menu SCREEN[:TAB]  open a menu screen at start (pause, confirm_quit, settings:Audio), for screenshots\n\
  --tool NAME [--fire S] [--hold S]  select a tool and press the left button at S seconds for HOLD seconds\n\
  --shot FILE [N]  save a screenshot after N frames (default 90) and exit\n\
  --selftest [--out FILE.json]  check this install (brain pack, simulator, GPU, audio, folders, disk, memory, a 10 s run); exit 0 all pass, 1 any FAIL, 3 warnings\n\
  --challenge KEY  start a challenge at launch (tmaze sneak sweet reverse_surgery predict), for screenshots\n\
  --train-screen   open the training screen (T) at start, for screenshots\n\
  --headless --validate|--protocol F|--rerun-bundle Z|--rig N|--rig-assay N|--minipaper ID|--activation-screen|--knockout-screen  no window; see headless.rs\n\
  --bugreport [--out DIR]  print what a bug report would contain (nothing is sent) and write DIR/bugreport.md";

fn parse(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = args.skip(1);
    let (mut source, mut seed, mut smoke, mut arena, mut shot, mut cam) = (None, 1000u64, None, None, None, None);
    let (mut tool, mut fire, mut hold) = (None, 1.0f32, 0.2f32);
    let (mut rain, mut storm, mut day) = (0.0f32, false, 0.0f32);
    let mut kill_at = None;
    let mut menu = None;
    let mut fullscreen = false;
    let (mut selftest, mut bugreport, mut out, mut crash_screen) = (false, false, None, None);
    let mut train_screen = false;
    let mut surgery_screen = false;
    let mut inspect = None;
    let mut challenge = None;
    while let Some(a) = args.next() {
        if a == "--crash-screen" {
            crash_screen = Some(args.by_ref().map(PathBuf::from).collect());
            break;
        }
        let mut value = |what: &str| args.next().ok_or_else(|| format!("{a} needs {what}"));
        match a.as_str() {
            "--pack" => source = Some(Source::Pack(PathBuf::from(value("a file")?))),
            "--synthetic" => source = Some(Source::Synthetic { n: value("a neuron count")?.parse().map_err(|_| "bad neuron count")? }),
            "--seed" => seed = value("a number")?.parse().map_err(|_| "bad seed")?,
            "--smoke" => smoke = Some(value("seconds")?.parse::<f64>().map_err(|_| "bad number of seconds")?),
            "--arena" => arena = Some(value("an arena name")?),
            "--shot" => {
                let file = PathBuf::from(value("a file")?);
                shot = Some((file, 90));
            }
            "--rain" => rain = value("0..1")?.parse().map_err(|_| "bad rain")?,
            "--storm" => storm = true,
            "--fullscreen" => fullscreen = true,
            "--selftest" => selftest = true,
            "--train-screen" => train_screen = true,
            "--surgery-screen" => surgery_screen = true,
            "--inspect" => inspect = Some(value("a neuron row")?.parse().map_err(|_| "bad neuron row")?),
            "--challenge" => challenge = Some(value("a challenge: tmaze sneak sweet reverse_surgery predict")?),
            "--bugreport" => bugreport = true,
            "--2d" => {
                // the original top-down pygame game is not part of this build (PORTING.md, "The 2D game")
                eprintln!("note: --2d is not part of this build; starting the 3D game (it has every feature the 2D game had)");
            }
            "--flies" => {
                let n: u32 = value("a number of extra flies")?.parse().map_err(|_| "bad number of flies")?;
                // SAFETY: argument parsing runs first thing in `main`, before any thread exists
                unsafe { std::env::set_var("KTF_FLIES", n.saturating_sub(1).to_string()) };
            }
            "--out" => out = Some(PathBuf::from(value("a file or folder")?)),
            "--menu" => menu = Some(value("a screen, e.g. pause or settings:Audio")?),
            "--kill-at" => kill_at = Some(value("seconds")?.parse::<f32>().map_err(|_| "bad seconds")?),
            "--day" => day = value("seconds per day")?.parse().map_err(|_| "bad day length")?,
            "--tool" => {
                let name = value("a tool name")?;
                tool = Some(kick_the_fly::tools::TOOLS.iter().position(|t| t.0 == name).ok_or("unknown tool")?);
            }
            "--fire" => fire = value("seconds")?.parse().map_err(|_| "bad seconds")?,
            "--hold" => hold = value("seconds")?.parse().map_err(|_| "bad seconds")?,
            "--cam" => {
                let v: Vec<f32> = value("six numbers: eye x y z, target x y z")?.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() != 6 {
                    return Err("--cam wants x,y,z,tx,ty,tz".into());
                }
                cam = Some((bevy::math::Vec3::new(v[0], v[1], v[2]), bevy::math::Vec3::new(v[3], v[4], v[5])));
            }
            "--shot-frames" => {
                let n: u32 = value("a number")?.parse().map_err(|_| "bad frame count")?;
                if let Some(s) = shot.as_mut() {
                    s.1 = n;
                }
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let source = source.unwrap_or_else(|| match pack::find_default() {
        Some(p) => Source::Pack(p),
        None if kick_the_fly::sim_thread::embedded_pack().is_some() => Source::Embedded,
        None => {
            eprintln!("no brain pack found (data/kick_brain.npz); running a synthetic network. Use --pack FILE for the connectome.");
            Source::Synthetic { n: 20_000 }
        }
    });
    Ok(Args {
        source,
        seed,
        smoke,
        arena,
        shot,
        cam,
        script: tool.map(|t| (t, fire, hold)),
        rain,
        storm,
        day,
        kill_at,
        menu,
        fullscreen,
        selftest,
        train_screen,
        surgery_screen,
        inspect,
        challenge,
        bugreport,
        out,
        crash_screen,
    })
}

/// The release build on Windows has no console window (`windows_subsystem = "windows"`); when started from a terminal with arguments
/// (`--help`, `--smoke`), attach to that terminal so the text shows (`core/platform_env.py:attach_console`).
#[cfg(windows)]
fn attach_console_for_cli() {
    use std::os::windows::io::IntoRawHandle;

    use windows_sys::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle};
    if std::env::args().len() < 2 {
        return;
    }
    // SAFETY: plain Win32 calls with valid arguments; failure just leaves the process without a console
    unsafe {
        if AttachConsole(ATTACH_PARENT_PROCESS) != 0
            && let Ok(out) = std::fs::OpenOptions::new().write(true).open("CONOUT$")
        {
            let h = out.into_raw_handle();
            SetStdHandle(STD_OUTPUT_HANDLE, h as _);
            SetStdHandle(STD_ERROR_HANDLE, h as _);
        }
    }
}

fn main() -> ExitCode {
    #[cfg(windows)]
    attach_console_for_cli();
    {
        let argv: Vec<String> = std::env::args().collect();
        if kick_the_fly::headless::wanted(&argv) {
            return ExitCode::from(kick_the_fly::headless::main(&argv) as u8);
        }
    }
    let args = match parse(std::env::args()) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(1);
        }
    };
    if let Some(files) = args.crash_screen {
        return kick_the_fly::crash::run_crash_screen(files);
    }
    kick_the_fly::selftest::set_pack_path(if let Source::Pack(p) = &args.source { Some(p.clone()) } else { None });
    if args.selftest {
        return ExitCode::from(kick_the_fly::selftest::main(args.out.as_deref()) as u8);
    }
    if args.bugreport {
        return ExitCode::from(kick_the_fly::crash::bugreport_main(args.out.as_deref()) as u8);
    }
    let pack_path = if let Source::Pack(p) = &args.source { Some(p.clone()) } else { None };
    let mut config = SimConfig::new(args.source);
    config.seed = args.seed;
    config.memory_dir = Some(ktf_data::paths::get().memory_dir);

    if let Some(seconds) = args.smoke {
        return match run_smoke(config, seconds) {
            Ok(r) => {
                println!(
                    "sim thread: {} neurons, {} steps in {:.2} s = {:.1} steps/s (real time is 200), {} snapshots read, {} steps dropped, {:.2}% spiking",
                    r.neurons,
                    r.steps,
                    r.seconds,
                    r.steps_per_s,
                    r.snapshots,
                    r.steps_dropped,
                    100.0 * r.active_fraction
                );
                if r.steps_per_s >= 190.0 { ExitCode::SUCCESS } else { ExitCode::from(2) }
            }
            Err(msg) => {
                eprintln!("smoke run failed: {msg}");
                ExitCode::from(1)
            }
        };
    }

    kick_the_fly::crash::open_log();
    kick_the_fly::crash::install_panic_hook(true);
    ktf_data::diag::set_info("seed", args.seed.to_string());
    ktf_data::diag::set_info("video_driver", if std::env::var_os("WAYLAND_DISPLAY").is_some() { "wayland" } else { "x11 or native" });
    let mut app = App::new();
    app.insert_resource(kick_the_fly::sim_thread::PackPath(pack_path));
    let mut full = args.fullscreen;
    {
        // the player's settings: the same config.toml the Python game wrote (missing or damaged files fall back to defaults)
        let cfg = ktf_data::config::Config::load(&ktf_data::paths::get().config_file());
        for w in &cfg.warnings {
            eprintln!("config: {w}");
        }
        app.insert_resource(kick_the_fly::config::Settings::from_config(&cfg));
        full |= cfg.get_bool("graphics.fullscreen");
        app.insert_resource(kick_the_fly::menu_ui::UserConfig(cfg));
    }
    let mode = if full {
        bevy::window::WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Current)
    } else {
        bevy::window::WindowMode::Windowed
    };
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin { primary_window: Some(Window { title: "Kick the Fly".into(), mode, ..default() }), ..default() })
            .set(bevy::log::LogPlugin { custom_layer: kick_the_fly::crash::log_layer, ..default() }),
    )
    .add_plugins((SimPlugin { config }, kick_the_fly::GamePlugin));
    if let Some(name) = &args.arena {
        match kick_the_fly::world::ARENAS.iter().position(|a| a == name) {
            Some(i) => {
                app.insert_resource(kick_the_fly::config::Arena(i));
            }
            None => {
                eprintln!("unknown arena {name}");
                return ExitCode::from(1);
            }
        }
    }
    {
        let mut st = app.world_mut().resource_mut::<kick_the_fly::config::Settings>();
        st.rain = args.rain;
        st.storm = args.storm;
        st.day_night_s = args.day;
    }
    if let Some(c) = args.challenge.clone() {
        app.world_mut().resource_mut::<kick_the_fly::challenges::Challenges>().request = Some(c);
    }
    if let Some(row) = args.inspect {
        let link = app.world().resource::<kick_the_fly::app::SimLink>();
        link.0.send(kick_the_fly::sim_thread::Command::Inspect(row));
    }
    if args.surgery_screen {
        app.world_mut().resource_mut::<kick_the_fly::surgery_play::Surgery>().open = true;
    }
    if args.train_screen {
        app.world_mut().resource_mut::<kick_the_fly::memory_play::Learned>().train_open = true;
    }
    if let Some(m) = &args.menu {
        app.insert_resource(kick_the_fly::menu_ui::MenuStart(m.clone()));
    }
    if let Some(t) = args.kill_at {
        app.add_plugins(kick_the_fly::capture::KillAtPlugin(t));
    }
    if let Some((tool, fire_at, hold_s)) = args.script {
        app.add_plugins(kick_the_fly::capture::ScriptPlugin(kick_the_fly::capture::Script { tool, fire_at, hold_s }));
    }
    if let Some((eye, target)) = args.cam {
        app.insert_resource(kick_the_fly::player::CameraOverride(eye, target));
    }
    if let Some((path, frames)) = args.shot {
        app.add_plugins(kick_the_fly::capture::CapturePlugin(kick_the_fly::capture::CaptureRequest {
            path,
            after_frames: frames,
            wait_after: 20,
        }));
    }
    app.run();
    ExitCode::SUCCESS
}
