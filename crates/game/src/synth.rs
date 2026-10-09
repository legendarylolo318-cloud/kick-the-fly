//! Every sound the game makes is synthesised here at startup, so it ships no audio files; and the brain sonification's synthesiser.
//! Python: `game/kick_the_fly.py:Sound._make` (the effects) and `core/sonify.py` (the sonification). Plain DSP, no engine and no device.
//!
//! Sonification: CONNECTOME is what it listens to (the simulation's own smoothed firing rate of each region, and the song readout, the
//! ps1 wing motor neurons). GAME RULE is everything that makes it sound: which region is which voice and its note, the D minor pentatonic
//! scale every voice shares (so any mix is consonant), how a rate relative to the region's own slow baseline becomes loudness and a step up
//! the scale, the glides, the soft limiter, and the song (a 220 Hz carrier in 12 ms pulses 35 ms apart). It reads the brain and never
//! writes to it.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

use crate::rng::Rng;

pub const RATE: u32 = 22050;

fn n_of(d: f32) -> usize {
    (RATE as f32 * d) as usize
}

fn t(d: f32) -> Vec<f32> {
    (0..n_of(d)).map(|i| i as f32 / RATE as f32).collect()
}

fn noise(rng: &mut Rng, d: f32) -> Vec<f32> {
    (0..n_of(d)).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// A centred moving-average low-pass (`np.convolve(x, ones(k)/k, mode="same")`).
fn low(x: &[f32], k: usize) -> Vec<f32> {
    let h = k / 2;
    (0..x.len())
        .map(|i| {
            let (lo, hi) = (i.saturating_sub(h), (i + k - h).min(x.len()));
            x[lo..hi].iter().sum::<f32>() / k as f32
        })
        .collect()
}

fn sweep(f0: f32, f1: f32, d: f32) -> Vec<f32> {
    let tt = t(d);
    let mut acc = 0.0f32;
    tt.iter()
        .map(|&x| {
            acc += f0 + (f1 - f0) * x / d;
            (TAU * acc / RATE as f32).sin()
        })
        .collect()
}

fn zip(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect()
}

fn env(tt: &[f32], k: f32) -> Vec<f32> {
    tt.iter().map(|x| (-x * k).exp()).collect()
}

fn snd(x: Vec<f32>, vol: f32) -> Vec<f32> {
    x.into_iter().map(|v| v.clamp(-1.0, 1.0) * vol).collect()
}

fn sine(tt: &[f32], hz: f32) -> Vec<f32> {
    tt.iter().map(|x| (TAU * hz * x).sin()).collect()
}

/// All the effects, by name, as samples in -1..1 at [`RATE`].
pub fn sfx_bank() -> HashMap<&'static str, Vec<f32>> {
    let mut rng = Rng::new(5);
    let mut fx: HashMap<&'static str, Vec<f32>> = HashMap::new();
    // wing buzz loops: 0.5 s holds whole cycles, so they loop cleanly
    for (k, f) in [150.0f32, 190.0, 240.0, 300.0].into_iter().enumerate() {
        let tt = t(0.5);
        let v: Vec<f32> = tt.iter().map(|x| 0.45 * (2.0 * ((x * f) % 1.0) - 1.0) * (0.8 + 0.2 * (TAU * 2.0 * x).sin())).collect();
        fx.insert(["buzz0", "buzz1", "buzz2", "buzz3"][k], snd(v, 0.35));
    }
    let tt = t(0.18);
    let n = noise(&mut rng, 0.18);
    let v = zip(&low(&n, 6).iter().zip(env(&tt, 28.0)).map(|(a, e)| a * e * 2.0).collect::<Vec<_>>(), &sine(&tt, 85.0).iter().zip(env(&tt, 20.0)).map(|(a, e)| a * e).collect::<Vec<_>>(), |a, b| a + b);
    fx.insert("whack", snd(v, 0.6));
    let tt = t(0.3);
    let n = noise(&mut rng, 0.3);
    let a: Vec<f32> = low(&n, 30).iter().zip(env(&tt, 10.0)).map(|(a, e)| a * e * 3.0).collect();
    let b: Vec<f32> = sweep(220.0, 50.0, 0.3).iter().zip(env(&tt, 9.0)).map(|(a, e)| a * e * 0.6).collect();
    fx.insert("squish", snd(zip(&a, &b, |a, b| a + b), 0.6));
    let tt = t(0.06);
    let n = noise(&mut rng, 0.06);
    fx.insert("flick", snd(n.iter().zip(env(&tt, 90.0)).map(|(a, e)| a * e).collect(), 0.6));
    let tt = t(1.0);
    let n = noise(&mut rng, 1.0);
    let a: Vec<f32> = low(&n, 60).iter().zip(env(&tt, 4.0)).map(|(a, e)| a * e * 5.0).collect();
    let b: Vec<f32> = sine(&tt, 45.0).iter().zip(env(&tt, 5.0)).map(|(a, e)| a * e).collect();
    fx.insert("boom", snd(zip(&a, &b, |a, b| a + b), 0.8));
    let tt = t(0.4);
    let n = noise(&mut rng, 0.4);
    let sq: Vec<f32> = sine(&tt, 70.0).iter().map(|v| v.signum() * 0.4).collect();
    let crackle: Vec<f32> = n.iter().map(|v| if rng.f32() < 0.25 { *v } else { 0.0 }).collect();
    fx.insert("zap", snd(zip(&sq, &crackle, |a, b| a + b).iter().zip(env(&tt, 6.0)).map(|(a, e)| a * e).collect(), 0.5));
    let n1 = noise(&mut rng, 1.0);
    let lo1 = low(&n1, 4);
    fx.insert("sizzle", snd(zip(&n1, &lo1, |a, b| a - b).into_iter().map(|v| v * (0.5 + 0.5 * if rng.f32() < 0.08 { 1.0 } else { 0.0 })).collect(), 0.35));
    let n2 = noise(&mut rng, 1.0);
    let lo2 = low(&n2, 3);
    fx.insert("hiss", snd(zip(&n2, &lo2, |a, b| a - b), 0.25));
    let tt = t(0.2);
    let n = noise(&mut rng, 0.2);
    let bite: Vec<f32> = low(&n, 8).iter().zip(&tt).map(|(a, x)| a * 2.5 * ((-x * 40.0).exp() + (-(x - 0.1).max(0.0) * 40.0).exp() * if *x > 0.1 { 1.0 } else { 0.0 })).collect();
    fx.insert("chomp", snd(bite, 0.6));
    let (t1, t2) = (t(0.09), t(0.16));
    let mut yum: Vec<f32> = sine(&t1, 520.0).iter().zip(env(&t1, 20.0)).map(|(a, e)| a * e).collect();
    yum.extend(sine(&t2, 780.0).iter().zip(env(&t2, 12.0)).map(|(a, e)| a * e));
    fx.insert("yum", snd(yum, 0.4));
    let tt = t(0.6);
    let n = noise(&mut rng, 0.6);
    let lo3 = low(&n, 3);
    let pings: Vec<f32> = tt.iter().map(|x| [2200.0f32, 2950.0, 3700.0, 4400.0].iter().map(|f| (TAU * f * x).sin() * (-x * 9.0).exp()).sum()).collect();
    let v: Vec<f32> = (0..tt.len()).map(|i| (n[i] - lo3[i]) * (-tt[i] * 7.0).exp() + 0.25 * pings[i]).collect();
    fx.insert("shatter", snd(v, 0.5));
    let n = noise(&mut rng, 0.6);
    let a: Vec<f32> = low(&n, 12).iter().zip(env(&tt, 6.0)).map(|(a, e)| a * e * 2.0).collect();
    let b: Vec<f32> = sweep(300.0, 900.0, 0.6).iter().zip(env(&tt, 10.0)).map(|(a, e)| 0.3 * a * e).collect();
    fx.insert("splash", snd(zip(&a, &b, |a, b| a + b), 0.6));
    let tt = t(1.0);
    let n = noise(&mut rng, 1.0);
    fx.insert("whoosh", snd(low(&n, 40).iter().zip(&tt).map(|(a, x)| a * 4.0 * (0.7 + 0.3 * (TAU * x).sin())).collect(), 0.35));
    let tt = t(0.9);
    fx.insert("death", snd(sweep(440.0, 90.0, 0.9).iter().zip(env(&tt, 2.5)).map(|(a, e)| a * e).collect(), 0.45));
    let tt = t(0.18);
    fx.insert("dodge", snd(sweep(180.0, 950.0, 0.18).iter().zip(env(&tt, 6.0)).map(|(a, e)| a * e).collect(), 0.45));
    let tt = t(0.07);
    fx.insert("pop", snd(sine(&tt, 900.0).iter().zip(env(&tt, 50.0)).map(|(a, e)| a * e).collect(), 0.4));
    let tt = t(0.14);
    let n = noise(&mut rng, 0.14);
    let a: Vec<f32> = sine(&tt, 150.0).iter().zip(env(&tt, 25.0)).map(|(a, e)| a * e).collect();
    fx.insert("bonk", snd(a.iter().enumerate().map(|(i, v)| v + n[i] * (-tt[i] * 120.0).exp() * 0.5).collect(), 0.6));
    let tt = t(0.3);
    fx.insert("drop", snd(sweep(1400.0, 500.0, 0.3).iter().zip(env(&tt, 5.0)).map(|(a, e)| a * e).collect(), 0.25));
    let tt = t(0.5);
    fx.insert("hum", snd(zip(&sine(&tt, 120.0), &sine(&tt, 240.0), |a, b| a * 0.5 + b * 0.2), 0.2));
    let tt = t(0.25);
    fx.insert("click", snd(sine(&tt, 1300.0).iter().zip(env(&tt, 40.0)).map(|(a, e)| a * e).collect(), 0.3));
    let tt = t(0.16);
    let n = noise(&mut rng, 0.16);
    let a: Vec<f32> = sweep(1500.0, 260.0, 0.16).iter().zip(env(&tt, 14.0)).map(|(a, e)| a.signum() * 0.5 * e).collect();
    fx.insert("pew", snd(a.iter().enumerate().map(|(i, v)| v + n[i] * (-tt[i] * 60.0).exp() * 0.3).collect(), 0.35));
    let tt = t(0.25);
    let n = noise(&mut rng, 0.25);
    let a: Vec<f32> = low(&n, 20).iter().zip(env(&tt, 14.0)).map(|(a, e)| a * 3.0 * e).collect();
    let b: Vec<f32> = sine(&tt, 70.0).iter().zip(env(&tt, 10.0)).map(|(a, e)| a * e).collect();
    fx.insert("hurt", snd(zip(&a, &b, |a, b| a + b), 0.7));
    // courtship pulse song: 220 Hz in 12 ms pulses 35 ms apart
    let tt = t(0.42);
    let mut song = vec![0.0f32; tt.len()];
    for pi in 0..(0.42f32 / 0.035) as usize {
        let start = pi as f32 * 0.035;
        for (i, x) in tt.iter().enumerate() {
            if *x >= start && *x < start + 0.012 {
                let tau = x - start;
                song[i] = (TAU * 220.0 * tau).sin() * (PI * tau / 0.012).sin();
            }
        }
    }
    fx.insert("pulse_song", snd(song.into_iter().map(|v| v * 0.6).collect(), 0.4));
    // extracellular biphasic spike clicks for the brain stethoscope
    let ts = t(0.002);
    let n = noise(&mut rng, 0.002);
    fx.insert("spike_click1", snd(ts.iter().enumerate().map(|(i, x)| -(TAU * 950.0 * x).sin() * (-x * 2500.0).exp() + 0.15 * n[i] * (-x * 2000.0).exp()).collect(), 0.45));
    let ts = t(0.005);
    let n = noise(&mut rng, 0.005);
    fx.insert(
        "spike_click2",
        snd(
            ts.iter()
                .enumerate()
                .map(|(i, x)| {
                    let sh = (x - 0.0018).max(0.0);
                    -(TAU * 950.0 * x).sin() * (-x * 2500.0).exp() + 0.85 * -(TAU * 880.0 * sh).sin() * (-sh * 2500.0).exp() * if *x >= 0.0018 { 1.0 } else { 0.0 } + 0.1 * n[i] * (-x * 1500.0).exp()
                })
                .collect(),
            0.5,
        ),
    );
    let ts = t(0.009);
    let n = noise(&mut rng, 0.009);
    fx.insert(
        "spike_click3",
        snd(
            ts.iter()
                .enumerate()
                .map(|(i, x)| {
                    let burst: f32 = [(0.0f32, 950.0f32, 1.0f32), (0.0021, 1020.0, 0.9), (0.0045, 890.0, 0.8)].iter().map(|(d, f, a)| { let sub = (x - d).max(0.0); -(TAU * f * sub).sin() * (-sub * 2400.0).exp() * if *x >= *d { 1.0 } else { 0.0 } * a }).sum();
                    burst + 0.1 * n[i] * (-x * 1000.0).exp()
                })
                .collect(),
            0.55,
        ),
    );
    let ts = t(0.015);
    let n = noise(&mut rng, 0.015);
    fx.insert(
        "spike_click_many",
        snd(
            ts.iter()
                .enumerate()
                .map(|(i, x)| {
                    let many: f32 = [0.0f32, 0.0015, 0.0032, 0.0051, 0.0078, 0.0102].iter().zip([950.0f32, 1100.0, 850.0, 1050.0, 920.0, 980.0]).map(|(d, f)| { let sub = (x - d).max(0.0); -(TAU * f * sub).sin() * (-sub * 2200.0).exp() * if *x >= *d { 1.0 } else { 0.0 } * 0.6 }).sum();
                    many + 0.12 * n[i] * (-x * 500.0).exp()
                })
                .collect(),
            0.6,
        ),
    );
    fx
}

/// 16-bit mono PCM WAV bytes of `samples` (what `bevy_audio` decodes with its `wav` feature).
pub fn wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    out
}

// --- the sonification ------------------------------------------------------------------------------------------------------------------
pub const SCALE_ROOT: i32 = 38;
pub const SCALE: [i32; 5] = [0, 3, 5, 7, 10];
pub const MAX_STEPS: i32 = 6;
pub const PRESENCE: f32 = 0.16;
pub const LOUD_GAIN: f32 = 1.1;
pub const GLIDE_S: f32 = 0.09;
pub const ATTACK_S: f32 = 0.04;
pub const RELEASE_S: f32 = 0.25;
pub const BASE_TAU_S: f32 = 45.0;
pub const MIN_BASE_HZ: f32 = 0.05;
pub const SONG_CARRIER_HZ: f32 = 220.0;
pub const SONG_PULSE_S: f32 = 0.012;
pub const SONG_IPI_S: f32 = 0.035;
pub const SONG_THRESH: f32 = 1.8;
pub const SONG_OFF_RATIO: f32 = 0.7;
pub const SONG_GAIN: f32 = 0.55;
pub const DUCK: f32 = 0.55;
pub const OUT_PEAK: f32 = 0.8;

/// Name, the note a calm voice sits on (MIDI), overtones (relative amplitude of harmonics 1, 2, 3).
pub const VOICES: [(&str, i32, [f32; 3]); 8] = [
    ("Descending neurons", 38, [1.0, 0.35, 0.10]),
    ("Nerve cord (VNC)", 45, [1.0, 0.30, 0.08]),
    ("Gnathal ganglion", 50, [1.0, 0.25, 0.05]),
    ("Central brain", 53, [1.0, 0.20, 0.0]),
    ("Mushroom body", 57, [1.0, 0.15, 0.05]),
    ("Central complex", 62, [1.0, 0.10, 0.0]),
    ("Antennal lobe", 67, [1.0, 0.08, 0.0]),
    ("Optic lobe", 69, [1.0, 0.05, 0.0]),
];

pub fn midi_hz(m: f32) -> f32 {
    440.0 * 2.0f32.powf((m - 69.0) / 12.0)
}

/// Semitones above the root of scale degree `step` (0 = the root, 5 = an octave up).
pub fn degree_semitones(step: i32) -> i32 {
    let n = SCALE.len() as i32;
    12 * step.div_euclid(n) + SCALE[step.rem_euclid(n) as usize]
}

/// The note a voice sounds `step` degrees above its root, climbing the one scale every voice shares.
pub fn voice_midi(root: i32, step: i32) -> i32 {
    let d0 = (0..SCALE.len() as i32 * 8).find(|&i| SCALE_ROOT + degree_semitones(i) >= root).unwrap_or(0);
    SCALE_ROOT + degree_semitones(d0 + step)
}

/// How far a region's rate is from its own baseline, in octaves, clipped to -1..4. 0 = calm.
pub fn activity_octaves(rate: f32, base: f32) -> f32 {
    if base < MIN_BASE_HZ { -1.0 } else { (rate.max(1e-3) / base).log2().clamp(-1.0, 4.0) }
}

/// `(scale degree above the voice's root, loudness 0..1)`: louder and higher as it is stirred up; a calm one sits on its root at `PRESENCE`;
/// one that is not firing at all is silent.
pub fn voice_target(rate: f32, base: f32) -> (i32, f32) {
    if base < MIN_BASE_HZ && rate < MIN_BASE_HZ {
        return (0, 0.0);
    }
    let x = activity_octaves(rate, base.max(MIN_BASE_HZ));
    let up = x.max(0.0);
    let step = ((2.0 * up).round() as i32).min(MAX_STEPS);
    let mut loud = PRESENCE + (1.0 - PRESENCE) * (1.0 - (-LOUD_GAIN * up).exp());
    if x < 0.0 {
        loud = PRESENCE * (1.0 + x);
    }
    (step, loud.clamp(0.0, 1.0))
}

/// Whether the song is sounding, from the song readout's level: on above the threshold, off below 70% of it.
#[derive(Debug, Default, Clone)]
pub struct SongGate {
    pub on: bool,
    pub level: f32,
}

impl SongGate {
    pub fn update(&mut self, level: f32) -> bool {
        self.level = level;
        if !self.on && level >= SONG_THRESH {
            self.on = true;
        } else if self.on && level < SONG_THRESH * SONG_OFF_RATIO {
            self.on = false;
        }
        self.on
    }

    /// 0.4 at the threshold, up to 0.9 at twice it.
    pub fn loudness(&self) -> f32 {
        (0.4 + 0.5 * (self.level / SONG_THRESH - 1.0)).clamp(0.4, 0.9)
    }
}

/// Each voice's rate and its own slow baseline, started at the first reading.
#[derive(Debug, Clone)]
pub struct RegionMeter {
    pub rate: Vec<f32>,
    pub base: Vec<f32>,
    seen: bool,
}

impl RegionMeter {
    pub fn new(n: usize) -> Self {
        Self { rate: vec![0.0; n], base: vec![0.0; n], seen: false }
    }

    pub fn update(&mut self, rates_hz: &[f32], dt: f32) {
        self.rate.copy_from_slice(rates_hz);
        if !self.seen {
            self.base.copy_from_slice(rates_hz);
            self.seen = true;
        } else {
            let k = 1.0 - (-dt.max(0.0) / BASE_TAU_S).exp();
            for (b, r) in self.base.iter_mut().zip(&self.rate) {
                *b += (r - *b) * k;
            }
        }
    }

    pub fn targets(&self) -> Vec<(i32, f32)> {
        self.rate.iter().zip(&self.base).map(|(r, b)| voice_target(*r, *b)).collect()
    }
}

/// Renders the mix in chunks with continuous phase.
pub struct Synth {
    rate: f32,
    phase: Vec<f32>,
    freq: Vec<f32>,
    amp: Vec<f32>,
    f_target: Vec<f32>,
    a_target: Vec<f32>,
    song_on: bool,
    song_amp: f32,
    duck: f32,
    pos: usize,
    next_pulse: usize,
    tail: Vec<f32>,
    pulse: Vec<f32>,
}

impl Synth {
    pub fn new(rate: u32) -> Self {
        let k = VOICES.len();
        let n = (SONG_PULSE_S * rate as f32) as usize;
        let pulse = (0..n)
            .map(|i| {
                let tau = i as f32 / rate as f32;
                (TAU * SONG_CARRIER_HZ * tau).sin() * (PI * tau / SONG_PULSE_S).sin()
            })
            .collect();
        let freq: Vec<f32> = VOICES.iter().map(|v| midi_hz(v.1 as f32)).collect();
        Self { rate: rate as f32, phase: vec![0.0; k], f_target: freq.clone(), freq, amp: vec![0.0; k], a_target: vec![0.0; k], song_on: false, song_amp: 0.0, duck: 1.0, pos: 0, next_pulse: 0, tail: Vec::new(), pulse }
    }

    pub fn set_targets(&mut self, targets: &[(i32, f32)]) {
        for (i, v) in VOICES.iter().enumerate() {
            let (step, loud) = targets.get(i).copied().unwrap_or((0, 0.0));
            self.f_target[i] = midi_hz(voice_midi(v.1, step) as f32);
            self.a_target[i] = loud;
        }
    }

    pub fn set_song(&mut self, on: bool, loudness: f32) {
        self.song_on = on;
        self.song_amp = loudness;
    }

    pub fn render(&mut self, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        let mut norm = 0.0f32;
        for (i, (_, _, harm)) in VOICES.iter().enumerate() {
            let rising = self.a_target[i] > self.amp[i];
            let tau = if rising { ATTACK_S } else { RELEASE_S };
            let hsum: f32 = harm.iter().filter(|h| **h > 0.0).sum();
            let (mut ph, mut mean_a) = (self.phase[i], 0.0f32);
            let (mut f_last, mut a_last) = (self.freq[i], self.amp[i]);
            for (j, o) in out.iter_mut().enumerate() {
                let t = (j + 1) as f32 / self.rate;
                let f = self.f_target[i] + (self.freq[i] - self.f_target[i]) * (-t / GLIDE_S).exp();
                let a = self.a_target[i] + (self.amp[i] - self.a_target[i]) * (-t / tau).exp();
                ph += TAU * f / self.rate;
                let wave: f32 = harm.iter().enumerate().filter(|(_, h)| **h > 0.0).map(|(k, h)| h * ((k + 1) as f32 * ph).sin()).sum::<f32>() / hsum;
                *o += a * wave;
                mean_a += a;
                f_last = f;
                a_last = a;
            }
            norm += (mean_a / n.max(1) as f32).powi(2);
            self.phase[i] = ph % TAU;
            self.freq[i] = f_last;
            self.amp[i] = a_last;
        }
        let k = 1.0 / (1.0 + norm).sqrt(); // more loud voices do not mean a louder mix
        let duck_to = if self.song_on { DUCK } else { 1.0 };
        let start_duck = self.duck;
        for (j, o) in out.iter_mut().enumerate() {
            let t = (j + 1) as f32 / self.rate;
            let d = duck_to + (start_duck - duck_to) * (-t / 0.08).exp();
            *o *= k * d;
            self.duck = d;
        }
        if self.song_on || !self.tail.is_empty() {
            let s = self.song(n);
            for (o, v) in out.iter_mut().zip(s) {
                *o += SONG_GAIN * v;
            }
        }
        for o in &mut out {
            *o = (1.3 * *o).tanh() * OUT_PEAK; // the soft limiter: never above OUT_PEAK
        }
        out
    }

    fn song(&mut self, n: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; n + self.pulse.len()];
        buf[..self.tail.len()].copy_from_slice(&self.tail);
        if self.song_on {
            let ipi = (SONG_IPI_S * self.rate) as usize;
            while self.next_pulse < self.pos + n {
                let at = self.next_pulse.saturating_sub(self.pos);
                for (i, p) in self.pulse.iter().enumerate() {
                    buf[at + i] += p * self.song_amp;
                }
                self.next_pulse += ipi;
            }
        } else {
            self.next_pulse = self.pos + n;
        }
        self.pos += n;
        self.tail = buf[n..].to_vec();
        if !self.song_on && self.tail.iter().all(|v| v.abs() <= 1e-6) {
            self.tail.clear();
        }
        buf.truncate(n);
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_effect_exists_is_bounded_and_not_silent() {
        let bank = sfx_bank();
        for name in ["whack", "squish", "flick", "boom", "zap", "chomp", "yum", "shatter", "splash", "death", "dodge", "pop", "bonk", "drop", "hurt", "pulse_song", "buzz0", "buzz3", "spike_click1", "spike_click_many", "sizzle", "hiss", "whoosh", "click", "pew", "hum"] {
            let s = bank.get(name).unwrap_or_else(|| panic!("{name} missing"));
            assert!(!s.is_empty());
            assert!(s.iter().all(|v| v.abs() <= 1.0 && v.is_finite()), "{name} clips or is not finite");
            assert!(s.iter().any(|v| v.abs() > 0.02), "{name} is silent");
        }
        assert_eq!(bank["boom"].len(), RATE as usize);
        assert!((bank["whack"].len() as f32 / RATE as f32 - 0.18).abs() < 0.01);
    }

    #[test]
    fn wav_is_a_valid_16_bit_mono_file() {
        let w = wav_bytes(&[0.0, 0.5, -0.5, 1.0], 22050);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(&w[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(w[24..28].try_into().unwrap()), 22050);
        assert_eq!(u16::from_le_bytes(w[34..36].try_into().unwrap()), 16);
        assert_eq!(w.len(), 44 + 8);
        assert_eq!(i16::from_le_bytes(w[50..52].try_into().unwrap()), 32767);
    }

    #[test]
    fn the_buzz_is_half_a_second_of_whole_cycles_so_it_loops() {
        let bank = sfx_bank();
        for (name, hz) in [("buzz0", 150.0f32), ("buzz1", 190.0), ("buzz2", 240.0), ("buzz3", 300.0)] {
            let b = &bank[name];
            assert_eq!(b.len(), 11025);
            assert!((0.5 * hz).fract().abs() < 1e-3, "{name}: {hz} Hz fits whole cycles in 0.5 s");
        }
    }

    #[test]
    fn every_voice_sits_on_the_one_scale_so_any_mix_is_consonant() {
        for (_, root, _) in VOICES {
            for step in 0..=MAX_STEPS {
                let m = voice_midi(root, step);
                let pc = (m - SCALE_ROOT).rem_euclid(12);
                assert!(SCALE.contains(&pc), "voice {root} step {step}: note {m} is off the scale");
                assert!(m >= root);
            }
        }
        assert_eq!(voice_midi(38, 0), 38);
        assert_eq!(voice_midi(38, 5), 50, "five degrees is an octave");
    }

    #[test]
    fn activity_maps_to_pitch_and_loudness() {
        assert_eq!(voice_target(0.0, 0.0), (0, 0.0), "a region that is not firing is silent");
        let calm = voice_target(5.0, 5.0);
        assert_eq!(calm.0, 0);
        assert!((calm.1 - PRESENCE).abs() < 1e-5);
        let stirred = voice_target(40.0, 5.0);
        assert!(stirred.0 >= 5 && stirred.1 > 0.8);
        let quiet = voice_target(3.5, 5.0);
        assert!(quiet.1 < PRESENCE && quiet.1 > 0.0);
    }

    #[test]
    fn the_song_gate_has_hysteresis() {
        let mut g = SongGate::default();
        assert!(!g.update(1.5));
        assert!(g.update(1.9));
        assert!(g.update(1.5), "stays on above 70% of the threshold");
        assert!(!g.update(1.0));
        g.level = 3.6;
        assert!((g.loudness() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn the_mix_never_passes_the_limiter_and_silence_is_silent() {
        let mut s = Synth::new(RATE);
        let silent = s.render(2205);
        assert!(silent.iter().all(|v| v.abs() < 1e-6));
        s.set_targets(&[(6, 1.0); 8]);
        s.set_song(true, 0.9);
        let mut peak = 0.0f32;
        for _ in 0..20 {
            peak = s.render(2205).iter().fold(peak, |p, v| p.max(v.abs()));
        }
        assert!(peak > 0.1 && peak <= OUT_PEAK + 1e-6, "peak {peak}");
    }

    #[test]
    fn a_voice_glides_to_its_note() {
        let mut s = Synth::new(RATE);
        s.set_targets(&[(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)]);
        for _ in 0..10 {
            s.render(2205);
        }
        let x = s.render(RATE as usize);
        let crossings = x.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let hz = midi_hz(38.0);
        assert!((crossings as f32 - hz).abs() < hz * 0.2, "{crossings} rising crossings in a second for a {hz:.1} Hz voice");
    }

    #[test]
    fn region_baselines_follow_slowly() {
        let mut m = RegionMeter::new(2);
        m.update(&[10.0, 4.0], 0.1);
        assert_eq!(m.base, vec![10.0, 4.0]);
        m.update(&[20.0, 4.0], 0.1);
        assert!(m.base[0] > 10.0 && m.base[0] < 10.1, "baseline barely moves in 0.1 s: {}", m.base[0]);
        assert_eq!(m.targets()[1], (0, PRESENCE));
    }
}
