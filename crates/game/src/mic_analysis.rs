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

/// One second-order section in transposed direct form II (float64, as `scipy.signal.sosfilt` runs it).
#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    b: [f64; 3],
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z1;
        self.z1 = self.b[1] * x - self.a1 * y + self.z2;
        self.z2 = self.b[2] * x - self.a2 * y;
        y
    }
}

#[derive(Clone, Copy)]
struct C(f64, f64);

impl C {
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
    fn sub(self, o: C) -> C {
        C(self.0 - o.0, self.1 - o.1)
    }
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn scale(self, k: f64) -> C {
        C(self.0 * k, self.1 * k)
    }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    fn sqrt(self) -> C {
        let r = self.0.hypot(self.1);
        let re = ((r + self.0) / 2.0).sqrt();
        let im = ((r - self.0) / 2.0).sqrt();
        C(re, if self.1 < 0.0 { -im } else { im })
    }
}

/// `scipy.signal.butter(4, [lo, hi] / nyquist, btype="band", output="sos")`: analog prototype, band-pass transform and bilinear transform,
/// then the pole pairs as sections (the transfer function is the same whatever the pairing; only rounding differs).
fn butter_bandpass_sos(lo: f64, hi: f64, fs: f64) -> Vec<Biquad> {
    const N: usize = 4;
    let nyq = fs / 2.0;
    let wn = [lo / nyq, hi.min(nyq * 0.95) / nyq];
    let warped = [4.0 * (std::f64::consts::PI * wn[0] / 2.0).tan(), 4.0 * (std::f64::consts::PI * wn[1] / 2.0).tan()];
    let (bw, wo) = (warped[1] - warped[0], (warped[0] * warped[1]).sqrt());
    let mut poles: Vec<C> = Vec::new();
    for m in (-(N as i32) + 1..N as i32).step_by(2) {
        let ang = std::f64::consts::PI * f64::from(m) / (2 * N) as f64;
        let p = C(-ang.cos(), -ang.sin()).scale(bw / 2.0);
        let r = p.mul(p).sub(C(wo * wo, 0.0)).sqrt();
        poles.push(p.add(r));
        poles.push(p.sub(r));
    }
    let k_bp = bw.powi(N as i32);
    // bilinear with fs2 = 4: z = (4 + p) / (4 - p); the N zeros at s = 0 map to z = 1 and N more appear at z = -1
    let four = C(4.0, 0.0);
    let zp: Vec<C> = poles.iter().map(|&p| four.add(p).div(four.sub(p))).collect();
    let mut den = C(1.0, 0.0);
    for &p in &poles {
        den = den.mul(four.sub(p));
    }
    let k_z = k_bp * four.div(den).0 * 4.0f64.powi(N as i32 - 1);
    // one section per pole pair (positive imaginary part), the pairs nearest the unit circle last
    let mut pairs: Vec<C> = zp.iter().copied().filter(|p| p.1 > 0.0).collect();
    pairs.sort_by(|a, b| a.0.hypot(a.1).total_cmp(&b.0.hypot(b.1)));
    pairs
        .iter()
        .enumerate()
        .map(|(i, p)| Biquad {
            b: if i == 0 { [k_z, 0.0, -k_z] } else { [1.0, 0.0, -1.0] },
            a1: -2.0 * p.0,
            a2: p.0 * p.0 + p.1 * p.1,
            z1: 0.0,
            z2: 0.0,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Band {
    sections: Vec<Biquad>,
}

impl Band {
    fn new(lo: f32, hi: f32, fs: f32) -> Self {
        Self { sections: butter_bandpass_sos(f64::from(lo), f64::from(hi), f64::from(fs)) }
    }

    fn process(&mut self, x: f32) -> f64 {
        self.sections.iter_mut().fold(f64::from(x), |v, s| s.process(v))
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
    /// 24 log-spaced spectrum levels over `PEAK_RANGE`, 0..1, for the hum test's graph.
    pub bins: [f32; 24],
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
        Self {
            rate,
            sensitivity: sensitivity.clamp(SENSITIVITY.0, SENSITIVITY.1),
            a: Band::new(BAND_A.0, BAND_A.1, rate),
            b: Band::new(BAND_B.0, BAND_B.1, rate),
            buf: vec![0.0; WINDOW],
            win,
            t: 0.0,
            reading: Reading::default(),
        }
    }

    /// Analyse one chunk (samples in -1..1) and return the new reading.
    pub fn push(&mut self, samples: &[f32]) -> Reading {
        if samples.is_empty() {
            return self.reading;
        }
        let (mut qa, mut qb, mut qx) = (Vec::with_capacity(samples.len()), Vec::with_capacity(samples.len()), 0.0f32);
        for &x in samples {
            let (ya, yb) = (self.a.process(x), self.b.process(x));
            qa.push(ya * ya);
            qb.push(yb * yb);
            qx += x * x;
        }
        let n = samples.len() as f32;
        let nf = f64::from(n);
        // `np.sqrt(np.mean(a * a))` in float64 (pairwise sums)
        let rms_a = (fly_sim::validation::np_sum(&qa) / nf).sqrt() as f32;
        let rms_b = (fly_sim::validation::np_sum(&qb) / nf).sqrt() as f32;
        let rms = (qx / n).sqrt();
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
        // 24 log-spaced bins over PEAK_RANGE: the loudest bin in each, over 0.2, clipped (`np.geomspace` edges, [lo, hi))
        let mut bins = [0.0f32; 24];
        let (g0, g1) = (PEAK_RANGE.0.ln(), PEAK_RANGE.1.ln());
        let edge = |i: usize| (g0 + (g1 - g0) * i as f32 / 24.0).exp();
        for (i, b) in bins.iter_mut().enumerate() {
            let (lo, hi) = (edge(i), if i == 23 { PEAK_RANGE.1 } else { edge(i + 1) });
            let m = (0..WINDOW / 2).filter(|&k| {
                let f = k as f32 * bin_hz;
                f >= lo && (f < hi || (i == 23 && f <= hi))
            });
            let mx = m.map(|k| spec[k]).fold(None::<f32>, |a, v| Some(a.map_or(v, |x| x.max(v))));
            *b = mx.map_or(0.0, |v| (v / 0.2).clamp(0.0, 1.0));
        }
        self.t += n / self.rate;
        self.reading = Reading {
            bins,
            peak_hz: peak,
            rms,
            rms_a,
            rms_b,
            drive_a: band_drive(rms_a, self.sensitivity),
            drive_b: band_drive(rms_b, self.sensitivity),
            t: self.t,
        };
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
