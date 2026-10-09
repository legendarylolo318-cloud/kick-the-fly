//! Sound: the synthesised effects, the wing buzz that follows the flight neurons, and the brain sonification.
//! Python: `Sound` (effects, buzz) and `game/sonify_play.py` (the live synth). The DSP is `synth.rs`; this is the engine side.
//!
//! * Effects are generated at startup as WAV bytes (no audio files ship) and played through `bevy_audio` when a [`Sfx`] message arrives.
//! * The buzz is one looping source whose pitch and volume follow DNg02 (the wing-power descending neurons) while the fly flies.
//! * The sonification (F5) is a streaming source: each frame the region firing rates from the sim snapshot are turned into voice targets,
//!   the synth renders that frame's samples into a short queue, and the audio thread drains it. If the queue runs dry it plays silence.
//!
//! No audio device (CI, a headless machine): `bevy_audio` creates no output, the messages are consumed, nothing plays, nothing panics.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bevy::audio::{AddAudioSource, AudioPlayer, AudioSink, AudioSinkPlayback, AudioSource, Decodable, PlaybackSettings, Source, Volume};
use bevy::prelude::*;

use crate::app::LatestSnapshot;
use crate::config::Settings;
use crate::feedback::{Notes, Sfx, Tag};
use crate::fly::Fly;
use crate::synth::{self, RATE, RegionMeter, SongGate, Synth};

const MAX_QUEUE_S: f32 = 0.25;
const MAX_PLAYING: usize = 20;

#[derive(Resource)]
pub struct SoundBank {
    pub sounds: HashMap<&'static str, Handle<AudioSource>>,
}

/// A live-fed mono stream: the game pushes samples, the audio thread pulls them.
#[derive(Asset, TypePath, Clone)]
pub struct LiveSynth {
    shared: Arc<Mutex<VecDeque<f32>>>,
}

pub struct LiveSource {
    shared: Arc<Mutex<VecDeque<f32>>>,
}

impl Iterator for LiveSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        Some(self.shared.lock().unwrap_or_else(PoisonError::into_inner).pop_front().unwrap_or(0.0))
    }
}

impl Source for LiveSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> bevy::audio::ChannelCount {
        std::num::NonZero::new(1).unwrap_or(std::num::NonZero::<u16>::MIN)
    }

    fn sample_rate(&self) -> bevy::audio::SampleRate {
        std::num::NonZero::new(RATE).unwrap_or(std::num::NonZero::<u32>::MIN)
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

impl Decodable for LiveSynth {
    type Decoder = LiveSource;

    fn decoder(&self) -> LiveSource {
        LiveSource { shared: self.shared.clone() }
    }
}

#[derive(Component)]
struct Buzz;
#[derive(Component)]
struct Sonification;

#[derive(Resource)]
pub struct Sonifier {
    pub on: bool,
    queue: Arc<Mutex<VecDeque<f32>>>,
    synth: Synth,
    meter: RegionMeter,
    gate: SongGate,
    voices: Vec<Vec<u32>>,
    built: bool,
    song_slow: f32,
}

impl Default for Sonifier {
    fn default() -> Self {
        Self {
            on: false,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            synth: Synth::new(RATE),
            meter: RegionMeter::new(synth::VOICES.len()),
            gate: SongGate::default(),
            voices: Vec::new(),
            built: false,
            song_slow: 1.0,
        }
    }
}

fn setup(mut commands: Commands, mut sources: ResMut<Assets<AudioSource>>, mut live: ResMut<Assets<LiveSynth>>, son: Res<Sonifier>) {
    let mut sounds = HashMap::new();
    for (name, samples) in synth::sfx_bank() {
        sounds.insert(name, sources.add(AudioSource { bytes: synth::wav_bytes(&samples, RATE).into() }));
    }
    let buzz = sounds["buzz1"].clone();
    commands.spawn((Buzz, AudioPlayer::new(buzz), PlaybackSettings::LOOP.with_volume(Volume::Linear(0.0))));
    commands.insert_resource(SoundBank { sounds });
    let handle = live.add(LiveSynth { shared: son.queue.clone() });
    commands.spawn((Sonification, AudioPlayer(handle), PlaybackSettings::LOOP.with_volume(Volume::Linear(0.0))));
}

/// Play the effects that were asked for this frame.
fn play_sfx(
    mut commands: Commands,
    mut msgs: MessageReader<Sfx>,
    bank: Res<SoundBank>,
    settings: Res<Settings>,
    playing: Query<(), (With<AudioPlayer>, Without<Buzz>, Without<Sonification>)>,
) {
    let mut room = MAX_PLAYING.saturating_sub(playing.iter().count());
    for m in msgs.read() {
        if settings.mute || room == 0 {
            continue;
        }
        if let Some(h) = bank.sounds.get(m.name) {
            let v = (settings.master_volume * m.volume).clamp(0.0, 1.5);
            commands.spawn((AudioPlayer::new(h.clone()), PlaybackSettings::DESPAWN.with_volume(Volume::Linear(v))));
            room -= 1;
        }
    }
}

/// The wing buzz follows DNg02 (the flight neurons) while a fly flies: louder and higher with more wing power.
fn buzz(settings: Res<Settings>, flies: Query<&Fly>, mut sink: Single<&mut AudioSink, With<Buzz>>) {
    let flying =
        flies.iter().filter(|f| f.flying_now && !f.dead()).map(|f| f.power).fold(None::<f32>, |a, p| Some(a.map_or(p, |x| x.max(p))));
    match flying {
        Some(power) if !settings.mute => {
            let k = (power / 2.0).clamp(0.0, 1.0);
            sink.set_volume(Volume::Linear(settings.master_volume * (0.18 + 0.3 * k)));
            sink.set_speed(0.9 + 0.35 * k);
        }
        _ => sink.set_volume(Volume::Linear(0.0)),
    }
}

/// F5 turns the sonification on and off.
fn toggle(act: Res<crate::actions::Actions>, mut son: ResMut<Sonifier>, mut notes: ResMut<Notes>, time: Res<Time<Real>>) {
    if act.just("sonify") {
        son.on = !son.on;
        let msg = if son.on { "SONIFY   the brain's regions as voices on one scale (F5 to stop)" } else { "SONIFY   off" };
        notes.push(time.elapsed_secs(), msg, Tag::GameRule);
        if !son.on {
            son.queue.lock().unwrap_or_else(PoisonError::into_inner).clear();
        }
    }
}

/// Each frame: region rates -> voice targets -> samples. Which neurons feed which voice is fixed once, from the snapshot's labels.
fn render(
    time: Res<Time>,
    snap: Res<LatestSnapshot>,
    settings: Res<Settings>,
    mut son: ResMut<Sonifier>,
    mut sink: Single<&mut AudioSink, With<Sonification>>,
) {
    sink.set_volume(Volume::Linear(if son.on && !settings.mute { settings.master_volume * 0.8 } else { 0.0 }));
    if !son.on {
        return;
    }
    let Some(s) = snap.0.as_deref() else { return };
    let Some(info) = s.info.as_deref() else { return };
    if s.rates.is_empty() {
        return;
    }
    if !son.built {
        let region_is = |i: usize, f: &dyn Fn(&str) -> bool| {
            info.region_id.get(i).and_then(|&r| info.region_names.get(r as usize)).is_some_and(|n| f(n))
        };
        let n = s.rates.len();
        let pick = |pred: &dyn Fn(usize) -> bool| -> Vec<u32> { (0..n).filter(|&i| pred(i)).map(|i| i as u32).collect() };
        son.voices = vec![
            pick(&|i| info.population.get(i) == Some(&6)),
            pick(&|i| region_is(i, &|r| r.starts_with("VNC"))),
            pick(&|i| region_is(i, &|r| r.starts_with("Gnathal"))),
            pick(&|i| region_is(i, &|r| r == "Central Brain")),
            pick(&|i| region_is(i, &|r| r == "Mushroom Body")),
            pick(&|i| region_is(i, &|r| r == "Central Complex")),
            pick(&|i| region_is(i, &|r| r == "Antennal Lobe")),
            pick(&|i| region_is(i, &|r| r == "Optic Lobe")),
        ];
        son.built = true;
    }
    let per_step_to_hz = 1.0 / 0.005;
    let hz: Vec<f32> = son
        .voices
        .iter()
        .map(|rows| {
            if rows.is_empty() { 0.0 } else { rows.iter().map(|&r| s.rates[r as usize]).sum::<f32>() / rows.len() as f32 * per_step_to_hz }
        })
        .collect();
    let dt = time.delta_secs();
    son.meter.update(&hz, dt);
    let targets = son.meter.targets();
    son.synth.set_targets(&targets);
    // the song: ps1's level averaged over about a second
    let song_level = info.group_names.iter().position(|n| n == "song").and_then(|i| s.levels.get(i).copied()).unwrap_or(1.0);
    son.song_slow += (song_level - son.song_slow) * (dt * 1.0).min(1.0);
    let slow = son.song_slow;
    let on = son.gate.update(slow);
    let loud = son.gate.loudness();
    son.synth.set_song(on, loud);
    let want = ((dt.min(0.1)) * RATE as f32) as usize;
    let queued = son.queue.lock().unwrap_or_else(PoisonError::into_inner).len();
    if (queued as f32) < MAX_QUEUE_S * RATE as f32 {
        let samples = son.synth.render(want.max(64));
        son.queue.lock().unwrap_or_else(PoisonError::into_inner).extend(samples);
    }
}

pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Sonifier>()
            .add_audio_source::<LiveSynth>()
            .add_systems(Startup, setup)
            .add_systems(Update, (play_sfx, buzz, toggle, render));
    }
}
