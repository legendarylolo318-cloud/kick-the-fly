//! The microphone as an input to the fly's hearing (Johnston's organ). OPT-IN: off until you press F6, and nothing it hears is kept.
//! Python: `core/mic.py` (the `Mic` class); the analysis is `mic_analysis.rs`.
//!
//! The capture runs on a thread of its own that owns the cpal stream (a stream is not `Send` on every platform). It tries the hosts in
//! the order PulseAudio, JACK-free ALSA, then whatever else cpal has, which on Linux covers PipeWire (it serves the PulseAudio protocol and
//! the ALSA `default` device), PulseAudio and plain ALSA; on Windows the default host is WASAPI. If there is no host, no device, no
//! permission or the stream will not open, [`MicStatus`] says why in plain words and the game carries on without it: that is the normal,
//! tested path (the analysis never needs a device). Only the latest [`Reading`] is shared.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use bevy::prelude::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::app::SimLink;
use crate::feedback::{Notes, Tag};
use crate::mic_analysis::{Analyzer, Reading};
use crate::sim_thread::Command;

/// What the capture thread reports back once it has tried to open the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicStatus {
    Off,
    Starting,
    On {
        host: String,
        device: String,
        rate: u32,
    },
    /// Why it could not be used, in plain words.
    Unavailable(String),
}

struct Shared {
    reading: Mutex<Reading>,
    status: Mutex<MicStatus>,
    stop: AtomicBool,
}

pub struct MicHandle {
    shared: Arc<Shared>,
    join: Option<JoinHandle<()>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The hosts to try, best first: PulseAudio (also what PipeWire serves), then the platform default, then the rest.
fn host_order() -> Vec<cpal::HostId> {
    let mut ids = cpal::available_hosts();
    let rank = |id: &cpal::HostId| -> u8 {
        let n = format!("{id:?}").to_lowercase();
        if n.contains("pulse") {
            0
        } else if n.contains("wasapi") || n.contains("alsa") || n.contains("coreaudio") {
            1
        } else {
            2
        }
    };
    ids.sort_by_key(rank);
    ids
}

/// Open the first input device that works. Returns the running stream and a description, or the reason none did.
fn open(shared: &Arc<Shared>, sensitivity: f32) -> Result<(cpal::Stream, String, String, u32), String> {
    let mut why = Vec::new();
    for id in host_order() {
        let host = match cpal::host_from_id(id) {
            Ok(h) => h,
            Err(e) => {
                why.push(format!("{id:?}: {e}"));
                continue;
            }
        };
        let Some(device) = host.default_input_device() else {
            why.push(format!("{id:?}: no input device"));
            continue;
        };
        let cfg = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                why.push(format!("{id:?}: {e}"));
                continue;
            }
        };
        let (rate, channels, fmt) = (cfg.sample_rate(), usize::from(cfg.channels()), cfg.sample_format());
        let analyzer = Arc::new(Mutex::new(Analyzer::new(rate as f32, sensitivity)));
        let sh = shared.clone();
        let mut feed = move |mono: &[f32]| {
            let r = lock(&analyzer).push(mono);
            *lock(&sh.reading) = r;
        };
        let err = |e: cpal::Error| eprintln!("microphone stream error: {e}");
        let config = cfg.config();
        let built = match fmt {
            cpal::SampleFormat::F32 => {
                device.build_input_stream(config, move |d: &[f32], _: &cpal::InputCallbackInfo| mono_f32(d, channels, &mut feed), err, None)
            }
            cpal::SampleFormat::I16 => device.build_input_stream(
                config,
                move |d: &[i16], _: &cpal::InputCallbackInfo| mono_conv(d, channels, |s| f32::from(s) / 32768.0, &mut feed),
                err,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                config,
                move |d: &[u16], _: &cpal::InputCallbackInfo| mono_conv(d, channels, |s| (f32::from(s) - 32768.0) / 32768.0, &mut feed),
                err,
                None,
            ),
            other => {
                why.push(format!("{id:?}: unsupported sample format {other:?}"));
                continue;
            }
        };
        match built {
            Ok(stream) => match stream.play() {
                Ok(()) => {
                    let name = device.description().map_or_else(|_| "default input".to_string(), |d| d.name().to_string());
                    return Ok((stream, format!("{id:?}"), name, rate));
                }
                Err(e) => why.push(format!("{id:?}: could not start the stream ({e})")),
            },
            Err(e) => why.push(format!("{id:?}: could not open the device ({e})")),
        }
    }
    Err(if why.is_empty() {
        "this system reports no audio capture host".into()
    } else {
        format!("no microphone could be opened: {}", why.join("; "))
    })
}

fn mono_f32(data: &[f32], channels: usize, feed: &mut impl FnMut(&[f32])) {
    if channels <= 1 {
        feed(data);
    } else {
        let m: Vec<f32> = data.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
        feed(&m);
    }
}

fn mono_conv<T: Copy>(data: &[T], channels: usize, conv: impl Fn(T) -> f32, feed: &mut impl FnMut(&[f32])) {
    let m: Vec<f32> = data.chunks(channels.max(1)).map(|f| f.iter().map(|&s| conv(s)).sum::<f32>() / channels.max(1) as f32).collect();
    feed(&m);
}

impl MicHandle {
    /// Start capturing. Returns at once; [`MicHandle::status`] says when (and whether) it opened.
    pub fn start(sensitivity: f32) -> Self {
        let shared = Arc::new(Shared {
            reading: Mutex::new(Reading::default()),
            status: Mutex::new(MicStatus::Starting),
            stop: AtomicBool::new(false),
        });
        let sh = shared.clone();
        let (tx, _rx): (Sender<()>, _) = mpsc::channel();
        let join = thread::Builder::new()
            .name("microphone".into())
            .spawn(move || {
                let _keep = tx;
                match open(&sh, sensitivity) {
                    Ok((stream, host, device, rate)) => {
                        *lock(&sh.status) = MicStatus::On { host, device, rate };
                        while !sh.stop.load(Ordering::Relaxed) {
                            thread::sleep(Duration::from_millis(50));
                        }
                        drop(stream);
                    }
                    Err(why) => *lock(&sh.status) = MicStatus::Unavailable(why),
                }
            })
            .ok();
        if join.is_none() {
            *lock(&shared.status) = MicStatus::Unavailable("could not start the capture thread".into());
        }
        Self { shared, join }
    }

    pub fn status(&self) -> MicStatus {
        lock(&self.shared.status).clone()
    }

    pub fn latest(&self) -> Reading {
        *lock(&self.shared.reading)
    }
}

impl Drop for MicHandle {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

#[derive(Resource, Default)]
pub struct Mic {
    pub handle: Option<MicHandle>,
    pub last: Reading,
    sent: (f32, f32),
    reported: bool,
}

impl Mic {
    pub fn status(&self) -> MicStatus {
        self.handle.as_ref().map_or(MicStatus::Off, MicHandle::status)
    }
}

/// F6 turns the microphone on and off.
fn toggle(keys: Res<ButtonInput<KeyCode>>, mut mic: ResMut<Mic>, link: Res<SimLink>, mut notes: ResMut<Notes>, time: Res<Time<Real>>) {
    if !keys.just_pressed(KeyCode::F6) {
        return;
    }
    let now = time.elapsed_secs();
    if mic.handle.is_some() {
        mic.handle = None;
        mic.sent = (0.0, 0.0);
        mic.reported = false;
        link.0.send(Command::DriveSpec { spec: "prefix:JO-A".into(), amp: 0.0 });
        link.0.send(Command::DriveSpec { spec: "prefix:JO-B".into(), amp: 0.0 });
        notes.push(now, "MIC      off", Tag::GameRule);
    } else {
        mic.handle = Some(MicHandle::start(1.0));
        notes.push(now, "MIC      opening the microphone (opt-in; nothing is recorded)", Tag::GameRule);
    }
}

/// Report how opening went, and put the latest reading on JO-A / JO-B (only when it changes enough to matter).
fn feed_brain(mut mic: ResMut<Mic>, link: Res<SimLink>, mut notes: ResMut<Notes>, time: Res<Time<Real>>) {
    let now = time.elapsed_secs();
    let Some(h) = mic.handle.as_ref() else { return };
    let status = h.status();
    let reading = h.latest();
    mic.last = reading;
    if !mic.reported {
        match &status {
            MicStatus::On { host, device, rate } => {
                mic.reported = true;
                notes.push(now, format!("MIC      on: {device} ({host}, {rate} Hz): hum to drive JO-A/JO-B"), Tag::GameRule);
            }
            MicStatus::Unavailable(why) => {
                mic.reported = true;
                notes.push(now, format!("MIC      {why}"), Tag::GameRule);
            }
            _ => {}
        }
    }
    if !matches!(status, MicStatus::On { .. }) {
        return;
    }
    let q = |x: f32| (x * 50.0).round() / 50.0;
    let (a, b) = (q(reading.drive_a), q(reading.drive_b));
    if (a, b) != mic.sent {
        // Python applies the drive as a current on JO-A and JO-B (`set_current("mic", ...)`); the sim thread holds it like any driven group
        link.0.send(Command::DriveSpec { spec: "prefix:JO-A".into(), amp: a });
        link.0.send(Command::DriveSpec { spec: "prefix:JO-B".into(), amp: b });
        mic.sent = (a, b);
    }
}

pub struct MicPlugin;

impl Plugin for MicPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Mic>().add_systems(Update, (toggle, feed_brain).chain());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no capture device (a CI machine, a headless box) the handle reports why and does not crash or hang; with one it opens. Either
    /// way `stop` returns.
    #[test]
    fn starting_the_mic_never_panics_and_always_ends_in_a_status() {
        let h = MicHandle::start(1.0);
        let mut status = h.status();
        for _ in 0..200 {
            if status != MicStatus::Starting {
                break;
            }
            thread::sleep(Duration::from_millis(25));
            status = h.status();
        }
        assert!(matches!(status, MicStatus::On { .. } | MicStatus::Unavailable(_)), "settled: {status:?}");
        if let MicStatus::Unavailable(why) = &status {
            assert!(!why.is_empty());
        }
        drop(h);
    }

    #[test]
    fn host_order_puts_pulse_first_when_there_is_one() {
        let order = host_order();
        assert!(!order.is_empty() || cpal::available_hosts().is_empty());
        if let Some(i) = order.iter().position(|h| format!("{h:?}").to_lowercase().contains("pulse")) {
            assert_eq!(i, 0);
        }
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let mut got = Vec::new();
        mono_f32(&[1.0, 0.0, 0.5, 0.5], 2, &mut |m| got.extend_from_slice(m));
        assert_eq!(got, vec![0.5, 0.5]);
    }
}
