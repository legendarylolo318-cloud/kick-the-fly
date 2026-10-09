//! Microphone -> Johnston's organ, the analysis half: band-pass the sound into the two organ groups and read the loudest frequency.
//! Python: `core/mic.py` (`Analyzer`, `band_drive`). No device here (`mic.rs` has that), so it is tested with synthetic waveforms.
//!
//! CONNECTOME: which neurons: JO-A* (50) and JO-B* (88), the vibration (sound) sensitive groups of Johnston's organ; the game already drives
//! JO-C/E with wind. LITERATURE: B prefers low frequencies (< ~100 Hz), A higher ones, together ~10 Hz to ~1 kHz (Ishikawa et al. 2019,
//! citing Matsuo et al. 2014 and Patella & Wilson 2018). GAME RULE: the whole transduction, every number: the two band-pass filters
//! (B 10-100 Hz, A 100-1,000 Hz), how loud a band must be to drive its neurons fully (`FULL_RMS`), the noise gate (`GATE_RMS`), the current
//! at full drive (`MAX_CURRENT`), the sensitivity, and that every neuron of a subgroup gets the same current. A microphone is not an
//! antenna: air pressure is treated as if it were the antennal vibration. No fly sound response is predicted.
//!
//! The filters here are an 8th-order band-pass built as a 4th-order Butterworth high-pass cascaded with a 4th-order Butterworth low-pass
//! (four biquads), the same order as the Python (`scipy.signal.butter(4, band)`) but not the identical transfer function.
//!
//! Privacy, by construction: samples are analysed in memory in small chunks and discarded; the only thing kept is [`Reading`].

use std::f32::consts::PI;

pub const WINDOW: usize = 1024;
pub const BAND_B: (f32, f32) = (10.0, 100.0);
pub const BAND_A: (f32, f32) = (100.0, 1000.0);
pub const FULL_RMS: f32 = 0.05;
pub const GATE_RMS: f32 = 0.002;
pub const MAX_CURRENT: f32 = 0.5;
pub const PEAK_RANGE: (f32, f32) = (40.0, 2000.0);
pub const SENSITIVITY: (f32, f32) = (0.1, 10.0);

/// Current for a band with this RMS: 0 under the gate, rising linearly to `MAX_CURRENT` at `FULL_RMS / sensitivity`. GAME RULE.
pub fn band_drive(rms: f32, sensitivity: f32) -> f32 {
    let r = rms * sensitivity;
    if r < GATE_RMS { 0.0 } else { MAX_CURRENT * (r / FULL_RMS).clamp(0.0, 1.0) }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn lowpass(fc: f32, fs: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * fc / fs;
        let (c, alpha) = (w0.cos(), w0.sin() / (2.0 * q));
        let a0 = 1.0 + alpha;
        Self { b0: (1.0 - c) / 2.0 / a0, b1: (1.0 - c) / a0, b2: (1.0 - c) / 2.0 / a0, a1: -2.0 * c / a0, a2: (1.0 - alpha) / a0, z1: 0.0, z2: 0.0 }
    }

    fn highpass(fc: f32, fs: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * fc / fs;
        let (c, alpha) = (w0.cos(), w0.sin() / (2.0 * q));
        let a0 = 1.0 + alpha;
        Self { b0: (1.0 + c) / 2.0 / a0, b1: -(1.0 + c) / a0, b2: (1.0 + c) / 2.0 / a0, a1: -2.0 * c / a0, a2: (1.0 - alpha) / a0, z1: 0.0, z2: 0.0 }
    }

    /// Transposed direct form II.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Q of the two sections of a 4th-order Butterworth filter.
const BUTTER4_Q: [f32; 2] = [0.541_196_1, 1.306_563];

#[derive(Debug, Clone)]
struct Band {
    sections: Vec<Biquad>,
}

impl Band {
    fn new(lo: f32, hi: f32, fs: f32) -> Self {
        let hi = hi.min(fs / 2.0 * 0.95);
        let mut sections = Vec::new();
        for q in BUTTER4_Q {
            sections.push(Biquad::highpass(lo, fs, q));
        }
        for q in BUTTER4_Q {
            sections.push(Biquad::lowpass(hi, fs, q));
        }
        Self { sections }
    }

    fn process(&mut self, x: f32) -> f32 {
        self.sections.iter_mut().fold(x, |v, s| s.process(v))
    }
}

/// The latest numbers: everything that is kept of what the microphone heard.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Reading {
    /// Loudest frequency in `PEAK_RANGE` (0 if too quiet to tell).
    pub peak_hz: f32,
    pub rms: f32,
    pub rms_a: f32,
    pub rms_b: f32,
    /// Current onto JO-A / JO-B (0 to `MAX_CURRENT`).
    pub drive_a: f32,
    pub drive_b: f32,
    pub t: f32,
}

/// In-place radix-2 complex FFT (`re.len()` a power of two).
pub fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
        len <<= 1;
    }
}

/// Turns a stream of samples into [`Reading`]s: two band-pass filters for the JO-A and JO-B drive, an FFT for the frequency. Deterministic.
pub struct Analyzer {
    rate: f32,
    pub sensitivity: f32,
    a: Band,
    b: Band,
    buf: Vec<f32>,
    win: Vec<f32>,
    t: f32,
    pub reading: Reading,
}

impl Analyzer {
    pub fn new(rate: f32, sensitivity: f32) -> Self {
        let win = (0..WINDOW).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / WINDOW as f32).cos()).collect();
        Self { rate, sensitivity: sensitivity.clamp(SENSITIVITY.0, SENSITIVITY.1), a: Band::new(BAND_A.0, BAND_A.1, rate), b: Band::new(BAND_B.0, BAND_B.1, rate), buf: vec![0.0; WINDOW], win, t: 0.0, reading: Reading::default() }
    }

    /// Analyse one chunk (samples in -1..1) and return the new reading.
    pub fn push(&mut self, samples: &[f32]) -> Reading {
        if samples.is_empty() {
            return self.reading;
        }
        let (mut sa, mut sb, mut sx) = (0.0f32, 0.0f32, 0.0f32);
        for &x in samples {
            let (ya, yb) = (self.a.process(x), self.b.process(x));
            sa += ya * ya;
            sb += yb * yb;
            sx += x * x;
        }
        let n = samples.len() as f32;
        let (rms_a, rms_b, rms) = ((sa / n).sqrt(), (sb / n).sqrt(), (sx / n).sqrt());
        // slide the spectrum window
        if samples.len() >= WINDOW {
            self.buf.copy_from_slice(&samples[samples.len() - WINDOW..]);
        } else {
            self.buf.rotate_left(samples.len());
            let at = WINDOW - samples.len();
            self.buf[at..].copy_from_slice(samples);
        }
        let mut re: Vec<f32> = self.buf.iter().zip(&self.win).map(|(x, w)| x * w).collect();
        let mut im = vec![0.0f32; WINDOW];
        fft(&mut re, &mut im);
        let spec: Vec<f32> = (0..WINDOW / 2).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt() / (WINDOW as f32 / 4.0)).collect();
        let bin_hz = self.rate / WINDOW as f32;
        let (lo, hi) = ((PEAK_RANGE.0 / bin_hz).ceil() as usize, ((PEAK_RANGE.1 / bin_hz) as usize).min(WINDOW / 2 - 2));
        let mut peak = 0.0;
        if lo < hi {
            let (i, m) = (lo..=hi).map(|i| (i, spec[i])).fold((lo, 0.0f32), |best, c| if c.1 > best.1 { c } else { best });
            if m > GATE_RMS * 2.0 {
                let (a, b, c) = (spec[i - 1], spec[i], spec[i + 1]);
                let den = a - 2.0 * b + c;
                let off = if den.abs() > 1e-12 { (0.5 * (a - c) / den).clamp(-0.5, 0.5) } else { 0.0 };
                peak = (i as f32 + off) * bin_hz;
            }
        }
        self.t += n / self.rate;
        self.reading = Reading { peak_hz: peak, rms, rms_a, rms_b, drive_a: band_drive(rms_a, self.sensitivity), drive_b: band_drive(rms_b, self.sensitivity), t: self.t };
        self.reading
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, amp: f32, rate: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| amp * (2.0 * PI * hz * i as f32 / rate).sin()).collect()
    }

    fn settle(a: &mut Analyzer, hz: f32, amp: f32) -> Reading {
        let x = tone(hz, amp, 22050.0, 22050);
        let mut r = Reading::default();
        for chunk in x.chunks(256) {
            r = a.push(chunk);
        }
        r
    }

    #[test]
    fn fft_finds_a_sine() {
        let mut re = tone(1000.0, 1.0, 8192.0, 1024);
        let mut im = vec![0.0; 1024];
        fft(&mut re, &mut im);
        let k = (0..512).max_by(|&a, &b| (re[a].hypot(im[a])).total_cmp(&re[b].hypot(im[b]))).unwrap();
        assert_eq!(k, 125, "1000 Hz at 8 Hz per bin");
    }

    #[test]
    fn a_low_hum_drives_jo_b_and_a_high_one_drives_jo_a() {
        let low = settle(&mut Analyzer::new(22050.0, 1.0), 50.0, 0.1);
        assert!(low.drive_b > 0.3 && low.drive_a < low.drive_b * 0.25, "50 Hz: B {} A {}", low.drive_b, low.drive_a);
        let high = settle(&mut Analyzer::new(22050.0, 1.0), 400.0, 0.1);
        assert!(high.drive_a > 0.3 && high.drive_b < high.drive_a * 0.25, "400 Hz: A {} B {}", high.drive_a, high.drive_b);
        assert!((high.peak_hz - 400.0).abs() < 12.0, "peak {}", high.peak_hz);
    }

    #[test]
    fn the_bands_overlap_only_a_little_at_200_hz() {
        let r = settle(&mut Analyzer::new(22050.0, 1.0), 200.0, 0.1);
        assert!(r.rms_a > 8.0 * r.rms_b, "200 Hz belongs to JO-A: A {} B {}", r.rms_a, r.rms_b);
    }

    #[test]
    fn silence_and_hiss_are_gated_and_sensitivity_scales_the_drive() {
        let quiet = settle(&mut Analyzer::new(22050.0, 1.0), 300.0, 0.001);
        assert_eq!(quiet.drive_a, 0.0, "under the gate");
        let soft = settle(&mut Analyzer::new(22050.0, 1.0), 300.0, 0.02);
        let sens = settle(&mut Analyzer::new(22050.0, 4.0), 300.0, 0.02);
        assert!(sens.drive_a > soft.drive_a * 1.5, "{} vs {}", sens.drive_a, soft.drive_a);
        assert!(band_drive(10.0, 1.0) <= MAX_CURRENT);
        assert_eq!(Analyzer::new(22050.0, 100.0).sensitivity, SENSITIVITY.1);
    }

    #[test]
    fn it_is_deterministic() {
        let a = settle(&mut Analyzer::new(44100.0, 1.0), 300.0, 0.05);
        let b = settle(&mut Analyzer::new(44100.0, 1.0), 300.0, 0.05);
        assert_eq!(a, b);
    }
}
