//! `kick_the_fly [--pack FILE | --synthetic N] [--seed S] [--smoke SECONDS]`
//!
//! Opens the game window with the brain running on its own thread. `--smoke` runs the sim thread headless for a number of seconds
//! and prints how it did (exit code 2 if it could not keep real time), which is what CI uses on machines without a display.

use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use fly_sim::pack;
use kick_the_fly::app::{HudPlugin, SimPlugin};
use kick_the_fly::run_smoke;
use kick_the_fly::sim_thread::{SimConfig, Source};

struct Args {
    source: Source,
    seed: u64,
    smoke: Option<f64>,
}

const USAGE: &str = "usage: kick_the_fly [--pack FILE | --synthetic N] [--seed S] [--smoke SECONDS]\n\
  --pack FILE      brain pack (default: $KICK_THE_FLY_PACK, else data/kick_brain.npz found upward from here)\n\
  --synthetic N    a random N-neuron network instead of the connectome (also what runs when no pack is found)\n\
  --seed S         noise seed (default 1000)\n\
  --smoke SECONDS  no window: run the brain thread for SECONDS and report steps per second";

fn parse(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = args.skip(1);
    let (mut source, mut seed, mut smoke) = (None, 1000u64, None);
    while let Some(a) = args.next() {
        let mut value = |what: &str| args.next().ok_or_else(|| format!("{a} needs {what}"));
        match a.as_str() {
            "--pack" => source = Some(Source::Pack(PathBuf::from(value("a file")?))),
            "--synthetic" => source = Some(Source::Synthetic { n: value("a neuron count")?.parse().map_err(|_| "bad neuron count")? }),
            "--seed" => seed = value("a number")?.parse().map_err(|_| "bad seed")?,
            "--smoke" => smoke = Some(value("seconds")?.parse::<f64>().map_err(|_| "bad number of seconds")?),
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let source = source.unwrap_or_else(|| match pack::find_default() {
        Some(p) => Source::Pack(p),
        None => {
            eprintln!("no brain pack found (data/kick_brain.npz); running a synthetic network. Use --pack FILE for the connectome.");
            Source::Synthetic { n: 20_000 }
        }
    });
    Ok(Args { source, seed, smoke })
}

fn main() -> ExitCode {
    let args = match parse(std::env::args()) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(1);
        }
    };
    let mut config = SimConfig::new(args.source);
    config.seed = args.seed;

    if let Some(seconds) = args.smoke {
        return match run_smoke(config, seconds) {
            Ok(r) => {
                println!(
                    "sim thread: {} neurons, {} steps in {:.2} s = {:.1} steps/s (real time is 200), {} snapshots read, {} steps dropped, {:.2}% spiking",
                    r.neurons, r.steps, r.seconds, r.steps_per_s, r.snapshots, r.steps_dropped, 100.0 * r.active_fraction
                );
                if r.steps_per_s >= 190.0 { ExitCode::SUCCESS } else { ExitCode::from(2) }
            }
            Err(msg) => {
                eprintln!("smoke run failed: {msg}");
                ExitCode::from(1)
            }
        };
    }

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "Kick the Fly".into(), ..default() }),
            ..default()
        }))
        .add_plugins((SimPlugin { config }, HudPlugin))
        .run();
    ExitCode::SUCCESS
}
