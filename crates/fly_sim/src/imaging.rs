//! Simulated calcium imaging (`lab/imaging.py`): the simulation's spikes convolved with an indicator kernel, averaged over ROIs,
//! plus Poisson photon shot noise. MODEL end to end (see the Python module's header); nothing here is a measurement.
//!
//! The per-neuron state is `float32` and the per-ROI maths `float64`, as in NumPy, so a recording equals Python's number for number.

use std::fmt::Write as _;

use crate::rng::Generator;

pub const DT_S: f64 = 0.005;
pub const TAG_TEXT: &str = "MODEL: spikes from the simulation convolved with an indicator kernel, plus shot noise. Not a measurement.";
pub const MAX_ROIS: usize = 600;
pub const DEFAULT_INDICATOR: &str = "gcamp6s";

#[derive(Debug, Clone, Copy)]
pub struct Indicator {
    pub key: &'static str,
    pub name: &'static str,
    pub rise_ms: f64,
    pub half_decay_ms: f64,
    pub source: &'static str,
    pub verified: &'static str,
}

pub const INDICATORS: [Indicator; 3] = [
    Indicator {
        key: "gcamp6s",
        name: "GCaMP6s (slow, sensitive)",
        rise_ms: 179.0,
        half_decay_ms: 550.0,
        source: "Chen et al. 2013, Nature 499:295 (doi:10.1038/nature12354), Supplementary Table 3",
        verified: "Verified in the paper's Supplementary Table 3 (mouse V1 in vivo, cell-attached, 1 action potential): \
rise time to peak 179 +- 23 ms, half-decay 550 +- 52 ms; this game uses the means. (The main text \
gives a 100-150 ms rise time for resolving single spikes.) Mouse cortex, not fly neurons.",
    },
    Indicator {
        key: "gcamp6f",
        name: "GCaMP6f (fast)",
        rise_ms: 45.0,
        half_decay_ms: 142.0,
        source: "Chen et al. 2013, Nature 499:295 (doi:10.1038/nature12354), Supplementary Table 3",
        verified: "Verified in the paper's Supplementary Table 3 (mouse V1 in vivo, cell-attached, 1 action potential): \
rise time to peak 45 +- 4 ms, half-decay 142 +- 11 ms; this game uses the means. (The main text \
gives a 50-75 ms rise time for resolving single spikes.) Mouse cortex, not fly neurons.",
    },
    Indicator {
        key: "jgcamp8m",
        name: "jGCaMP8m (fast, sensitive)",
        rise_ms: 58.0,
        half_decay_ms: 137.0,
        source: "Zhang et al. 2023, Nature 615:884 (doi:10.1038/s41586-023-05828-9), Drosophila in-vivo visual \
responses: half-rise 58 +- 6 ms, half-decay 137 +- 21 ms",
        verified: "Both numbers are stated in the paper's text (a figure legend). They describe a fly's response to a \
visual stimulus, not a single-spike impulse response, and a two-exponential kernel cannot have a 58 ms \
half-rise and a 137 ms half-decay at once, so this game uses 58 ms as the kernel's time to peak (a \
simplification). The same paper reports much faster kinetics in cultured neurons (half-rise 2 ms).",
    },
];

/// `indicator(name)`: a key or an alias, case and `-`/`_` ignored.
pub fn indicator(name: &str) -> Result<&'static Indicator, String> {
    let k = name.trim().to_lowercase().replace(['-', '_'], "");
    let key = match k.as_str() {
        "6s" | "gcamp6s" => "gcamp6s",
        "6f" | "gcamp6f" => "gcamp6f",
        "8m" | "gcamp8m" | "jgcamp8m" => "jgcamp8m",
        _ => return Err(format!("unknown indicator '{name}'; use one of GCaMP6s, GCaMP6f, GCaMP8m")),
    };
    Ok(INDICATORS.iter().find(|i| i.key == key).expect("known key"))
}

/// `(time to peak, half-decay from the peak)` in ms of `k(t) = exp(-t/tau_d) - exp(-t/tau_r)`.
fn shape(tau_r: f64, tau_d: f64) -> (f64, f64) {
    let tp = tau_d * tau_r / (tau_d - tau_r) * (tau_d / tau_r).ln();
    let peak = (-tp / tau_d).exp() - (-tp / tau_r).exp();
    let (mut lo, mut hi) = (tp, tp + 20.0 * tau_d);
    for _ in 0..80 {
        let mid = (lo + hi) / 2.0;
        if (-mid / tau_d).exp() - (-mid / tau_r).exp() > 0.5 * peak {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (tp, (lo + hi) / 2.0 - tp)
}

/// `(tau_rise, tau_decay)` in ms of the two-exponential kernel with the indicator's time to peak and half-decay (Python solves it with
/// `scipy.optimize.least_squares` from three starting points; this is a Levenberg-Marquardt on the same residuals).
pub fn time_constants(ind: &Indicator) -> Result<(f64, f64), String> {
    let resid = |x: [f64; 2]| -> [f64; 2] {
        let (tr, td) = (x[0].exp(), x[1].exp());
        if tr >= td {
            return [1e3, 1e3];
        }
        let (tp, hd) = shape(tr, td);
        [(tp - ind.rise_ms) / ind.rise_ms, (hd - ind.half_decay_ms) / ind.half_decay_ms]
    };
    let cost = |r: [f64; 2]| 0.5 * (r[0] * r[0] + r[1] * r[1]);
    let mut best: Option<([f64; 2], f64)> = None;
    for f in [0.25, 0.5, 0.1] {
        let mut x = [(ind.rise_ms * f * 2.0).ln(), (ind.half_decay_ms / std::f64::consts::LN_2 * 0.8).ln()];
        let mut lambda = 1e-3;
        let mut r = resid(x);
        for _ in 0..400 {
            // numeric Jacobian
            let h = 1e-7;
            let (r0, r1) = (resid([x[0] + h, x[1]]), resid([x[0], x[1] + h]));
            let j = [[(r0[0] - r[0]) / h, (r1[0] - r[0]) / h], [(r0[1] - r[1]) / h, (r1[1] - r[1]) / h]];
            let jtj = [
                [j[0][0] * j[0][0] + j[1][0] * j[1][0], j[0][0] * j[0][1] + j[1][0] * j[1][1]],
                [j[0][0] * j[0][1] + j[1][0] * j[1][1], j[0][1] * j[0][1] + j[1][1] * j[1][1]],
            ];
            let g = [j[0][0] * r[0] + j[1][0] * r[1], j[0][1] * r[0] + j[1][1] * r[1]];
            let mut improved = false;
            for _ in 0..30 {
                let a = [[jtj[0][0] + lambda, jtj[0][1]], [jtj[1][0], jtj[1][1] + lambda]];
                let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
                if det.abs() < 1e-300 {
                    lambda *= 10.0;
                    continue;
                }
                let dx = [-(a[1][1] * g[0] - a[0][1] * g[1]) / det, -(-a[1][0] * g[0] + a[0][0] * g[1]) / det];
                let xn = [x[0] + dx[0], x[1] + dx[1]];
                let rn = resid(xn);
                if cost(rn) < cost(r) {
                    x = xn;
                    r = rn;
                    lambda = (lambda * 0.3).max(1e-12);
                    improved = true;
                    break;
                }
                lambda *= 10.0;
            }
            if !improved || cost(r) < 1e-28 {
                break;
            }
        }
        let c = cost(r);
        if best.is_none_or(|b| c < b.1) {
            best = Some((x, c));
        }
    }
    let (x, c) = best.expect("three starts");
    if c > 1e-8 {
        return Err(format!("no two-exponential kernel has time to peak {} ms and half-decay {} ms", ind.rise_ms, ind.half_decay_ms));
    }
    Ok((x[0].exp(), x[1].exp()))
}

/// `kernel`: the single-spike dF/F response sampled every `dt_ms`, normalised so its peak is `amplitude`.
pub fn kernel(ind: &Indicator, dt_ms: f64, length_ms: Option<f64>, amplitude: f64) -> Result<Vec<f64>, String> {
    let (tr, td) = time_constants(ind)?;
    let length = length_ms.unwrap_or(10.0 * td);
    let n = (length / dt_ms).ceil() as usize;
    let k: Vec<f64> = (0..n).map(|i| (-(i as f64 * dt_ms) / td).exp() - (-(i as f64 * dt_ms) / tr).exp()).collect();
    let mx = k.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(k.iter().map(|v| amplitude * v / mx).collect())
}

/// `rois_by_region`: one ROI per region label (sorted names; neurons without one, or `unassigned`, are left out).
pub fn rois_by_region(region: &[String]) -> Result<Vec<(String, Vec<u32>)>, String> {
    if region.is_empty() {
        return Err("this brain has no region labels; use rois_by_type or an explicit ROI list".into());
    }
    let mut names: Vec<&str> = region.iter().map(String::as_str).filter(|n| !n.is_empty() && *n != "unassigned").collect();
    names.sort_unstable();
    names.dedup();
    Ok(names
        .into_iter()
        .map(|name| (name.to_string(), region.iter().enumerate().filter(|(_, r)| r.as_str() == name).map(|(i, _)| i as u32).collect()))
        .collect())
}

/// `rois_from_specs`: ROIs from neuron specs (a type, `prefix:KC`, `line:SS00727`, a group). `rows(spec)` resolves one spec the way the
/// brain does; a bare name that is not a group is a type.
pub fn rois_from_specs(
    specs: &[String],
    per_neuron: bool,
    rows: &dyn Fn(&str) -> Result<Vec<u32>, String>,
    is_group: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, Vec<u32>)>, String> {
    if specs.len() > MAX_ROIS {
        return Err(format!("at most {MAX_ROIS} ROIs"));
    }
    let mut out: Vec<(String, Vec<u32>)> = Vec::new();
    for spec in specs {
        let full = if spec.contains(':') || is_group(spec) { spec.clone() } else { format!("type:{spec}") };
        let mut r = rows(&full)?;
        r.sort_unstable();
        if r.is_empty() {
            return Err(format!("{spec:?} selects no neurons"));
        }
        if per_neuron {
            for &row in r.iter().take(MAX_ROIS) {
                put(&mut out, format!("{spec}#{row}"), vec![row]);
            }
        } else {
            put(&mut out, spec.clone(), r);
        }
    }
    if out.len() > MAX_ROIS {
        return Err(format!("at most {MAX_ROIS} ROIs"));
    }
    Ok(out)
}

/// A dict assignment: a repeated name replaces the earlier ROI in its place.
fn put(out: &mut Vec<(String, Vec<u32>)>, name: String, rows: Vec<u32>) {
    if let Some(e) = out.iter_mut().find(|e| e.0 == name) {
        e.1 = rows;
    } else {
        out.push((name, rows));
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    pub indicator: String,
    pub fps: f64,
    pub f0_photons: f64,
    pub dff_per_spike: f64,
    pub shot_noise: bool,
    pub seed: u64,
    pub dff_cap: f64,
    pub max_frames: Option<usize>,
    /// `"running"` or `"zero"`.
    pub baseline: String,
    pub baseline_tau_s: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            indicator: DEFAULT_INDICATOR.into(),
            fps: 20.0,
            f0_photons: 100.0,
            dff_per_spike: 0.2,
            shot_noise: true,
            seed: 0,
            dff_cap: 5.0,
            max_frames: None,
            baseline: "running".into(),
            baseline_tau_s: 30.0,
        }
    }
}

/// `ImagingSession`: the indicator state of the neurons in the ROIs, turned into frames.
pub struct Session {
    pub ind: &'static Indicator,
    n: usize,
    pub fps: f64,
    pub frame_steps: usize,
    f0: f64,
    amp: f64,
    pub shot_noise: bool,
    cap: f64,
    pub roi_names: Vec<String>,
    pub roi_rows: Vec<Vec<u32>>,
    pub tracked: Vec<u32>,
    pos: Vec<i32>,
    roi_pos: Vec<Vec<usize>>,
    pub tau_r_ms: f64,
    pub tau_d_ms: f64,
    a_r: f32,
    a_d: f32,
    norm: f32,
    ed: Vec<f32>,
    er: Vec<f32>,
    pub baseline: String,
    pub baseline_tau_s: f64,
    k_base: f32,
    base: Vec<f32>,
    rng: Generator,
    pub steps: u64,
    pub frames_taken: u64,
    /// The brain step count already pushed (`feed_from_activity`).
    seen: Option<u64>,
    max_frames: Option<usize>,
    pub t: Vec<f64>,
    pub true_dff: Vec<Vec<f64>>,
    pub dff: Vec<Vec<f64>>,
    pub photons: Vec<Vec<f64>>,
}

fn mean_f32(v: &[f32]) -> f32 {
    crate::diffmode::np_sum_f32(v) / v.len() as f32
}

impl Session {
    pub fn new(n: usize, rois: &[(String, Vec<u32>)], p: &Params) -> Result<Self, String> {
        let ind = indicator(&p.indicator)?;
        if rois.is_empty() {
            return Err("an imaging session needs at least one ROI".into());
        }
        if !(1.0..=200.0).contains(&p.fps) {
            return Err("frame rate must be between 1 and 200 Hz".into());
        }
        for (name, v) in
            [("f0_photons", p.f0_photons), ("dff_per_spike", p.dff_per_spike), ("dff_cap", p.dff_cap), ("baseline_tau_s", p.baseline_tau_s)]
        {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!("{name} must be a positive finite number"));
            }
        }
        let frame_steps = ((1.0 / p.fps / DT_S).round_ties_even() as usize).max(1);
        let fps = 1.0 / (frame_steps as f64 * DT_S);
        let roi_names: Vec<String> = rois.iter().map(|r| r.0.clone()).collect();
        let roi_rows: Vec<Vec<u32>> = rois.iter().map(|r| r.1.clone()).collect();
        if roi_rows.iter().any(Vec::is_empty) {
            return Err("an ROI has no neurons".into());
        }
        let mut tracked: Vec<u32> = roi_rows.iter().flatten().copied().collect();
        tracked.sort_unstable();
        tracked.dedup();
        let mut pos = vec![-1i32; n];
        for (i, &r) in tracked.iter().enumerate() {
            pos[r as usize] = i as i32;
        }
        let roi_pos: Vec<Vec<usize>> = roi_rows.iter().map(|r| r.iter().map(|&x| pos[x as usize] as usize).collect()).collect();
        let (tr, td) = time_constants(ind)?;
        let a_r = (-DT_S * 1000.0 / tr).exp();
        let a_d = (-DT_S * 1000.0 / td).exp();
        let tp = td * tr / (td - tr) * (td / tr).ln();
        let norm = p.dff_per_spike / ((-tp / td).exp() - (-tp / tr).exp());
        if p.baseline != "running" && p.baseline != "zero" {
            return Err("baseline must be 'running' or 'zero'".into());
        }
        let k_base = 1.0 - (-(frame_steps as f64 * DT_S) / p.baseline_tau_s.max(1e-3)).exp();
        let m = tracked.len();
        Ok(Self {
            ind,
            n,
            fps,
            frame_steps,
            f0: p.f0_photons,
            amp: p.dff_per_spike,
            shot_noise: p.shot_noise,
            cap: p.dff_cap,
            roi_names,
            roi_rows,
            tracked,
            pos,
            roi_pos,
            tau_r_ms: tr,
            tau_d_ms: td,
            a_r: a_r as f32,
            a_d: a_d as f32,
            norm: norm as f32,
            ed: vec![0.0; m],
            er: vec![0.0; m],
            baseline: p.baseline.clone(),
            baseline_tau_s: p.baseline_tau_s,
            k_base: k_base as f32,
            base: vec![0.0; m],
            rng: Generator::new(p.seed),
            steps: 0,
            frames_taken: 0,
            seen: None,
            max_frames: p.max_frames,
            t: Vec::new(),
            true_dff: Vec::new(),
            dff: Vec::new(),
            photons: Vec::new(),
        })
    }

    fn decay_and_add(&mut self, spiked: &[u32]) {
        for v in &mut self.ed {
            *v *= self.a_d;
        }
        for v in &mut self.er {
            *v *= self.a_r;
        }
        for &s in spiked {
            let p = self.pos[s as usize];
            if p >= 0 {
                self.ed[p as usize] += 1.0;
                self.er[p as usize] += 1.0;
            }
        }
    }

    /// One simulation step; true when it completed a frame.
    pub fn push(&mut self, spiked: &[u32]) -> bool {
        self.decay_and_add(spiked);
        self.steps += 1;
        if self.steps.is_multiple_of(self.frame_steps as u64) {
            self.take_frame();
            return true;
        }
        false
    }

    /// `neuron_dff`: dF/F of every tracked neuron against the no-calcium fluorescence (absolute, noise-free).
    pub fn neuron_dff(&self) -> Vec<f32> {
        let cap = self.cap as f32;
        self.ed.iter().zip(&self.er).map(|(&d, &r)| (self.norm * (d - r)).min(cap)).collect()
    }

    /// `neuron_delta`: dF/F against each neuron's own baseline F0.
    pub fn neuron_delta(&self) -> Vec<f32> {
        let d = self.neuron_dff();
        if self.baseline == "zero" {
            return d;
        }
        d.iter().zip(&self.base).map(|(&d, &b)| (1.0 + d) / (1.0 + b) - 1.0).collect()
    }

    /// `prime_from_activity`: push the raster's steps without taking frames, then take that as the baseline.
    pub fn prime(&mut self, raster: &[&[u32]], steps: u64) {
        self.seen = Some(steps);
        for idx in raster {
            self.decay_and_add(idx);
        }
        self.base = self.neuron_dff();
    }

    /// `feed_from_activity`: the steps a running brain took since the last call (the first call primes). Returns how many were pushed.
    pub fn feed(&mut self, raster: &[&[u32]], steps: u64) -> usize {
        let Some(seen) = self.seen else {
            self.prime(raster, steps);
            return 0;
        };
        let k = (steps.saturating_sub(seen)).min(400) as usize;
        self.seen = Some(steps);
        if k == 0 {
            return 0;
        }
        let k = k.min(raster.len());
        for idx in &raster[raster.len() - k..] {
            self.push(idx);
        }
        k
    }

    fn roi_means(&self, d: &[f32], v: &[Vec<usize>]) -> Vec<f64> {
        v.iter()
            .map(|p| {
                let g: Vec<f32> = p.iter().map(|&i| d[i]).collect();
                f64::from(mean_f32(&g))
            })
            .collect()
    }

    /// `roi_dff_true`: (1 + mean) is formed in float32 there (a NumPy float32 scalar plus a Python float), the quotient in float64.
    pub fn roi_dff_true(&self) -> Vec<f64> {
        let d = self.neuron_dff();
        let ones = |v: &[f32], pos: &[Vec<usize>]| -> Vec<f64> {
            pos.iter()
                .map(|p| {
                    let g: Vec<f32> = p.iter().map(|&i| v[i]).collect();
                    f64::from(1.0f32 + mean_f32(&g))
                })
                .collect()
        };
        let num = ones(&d, &self.roi_pos);
        let den = if self.baseline == "running" { ones(&self.base, &self.roi_pos) } else { vec![1.0; self.roi_pos.len()] };
        num.iter().zip(&den).map(|(a, b)| a / b - 1.0).collect()
    }

    pub fn take_frame(&mut self) {
        let d = self.neuron_dff();
        let n_in: Vec<f64> = self.roi_pos.iter().map(|p| p.len() as f64).collect();
        let mean_d = self.roi_means(&d, &self.roi_pos);
        let mean_b = if self.baseline == "running" { self.roi_means(&self.base, &self.roi_pos) } else { vec![0.0; n_in.len()] };
        let expect: Vec<f64> = n_in.iter().zip(&mean_d).map(|(n, m)| n * self.f0 * (1.0 + m)).collect();
        let photons: Vec<f64> =
            if self.shot_noise { expect.iter().map(|&e| self.rng.poisson(e.max(0.0)) as f64).collect() } else { expect.clone() };
        let den: Vec<f64> = n_in.iter().zip(&mean_b).map(|(n, m)| n * self.f0 * (1.0 + m)).collect();
        let tru: Vec<f64> = expect.iter().zip(&den).map(|(e, d)| e / d - 1.0).collect();
        let dff: Vec<f64> = photons.iter().zip(&den).map(|(p, d)| p / d - 1.0).collect();
        if self.baseline == "running" {
            for (b, &x) in self.base.iter_mut().zip(&d) {
                *b += (x - *b) * self.k_base;
            }
        }
        self.frames_taken += 1;
        self.t.push(self.steps as f64 * DT_S);
        self.true_dff.push(tru);
        self.dff.push(dff);
        self.photons.push(photons);
        if self.max_frames.is_some_and(|m| m > 0 && self.t.len() > m) {
            self.t.remove(0);
            self.true_dff.remove(0);
            self.dff.remove(0);
            self.photons.remove(0);
        }
    }

    /// `view_rates` (the part that is independent of the renderer): the displayed dF/F of every neuron, `max(delta, 0) * gain`,
    /// smoothed against `smooth` with reduced flashing. Neurons outside every ROI stay at 0.
    pub fn view_full(&self, gain: f32, smooth: Option<&[f32]>) -> Vec<f32> {
        let mut full = vec![0.0f32; self.n];
        for (&r, d) in self.tracked.iter().zip(self.neuron_delta()) {
            full[r as usize] = d.max(0.0) * gain;
        }
        if let Some(s) = smooth {
            for (f, &sv) in full.iter_mut().zip(s) {
                *f = sv + 0.35 * (*f - sv);
            }
        }
        full
    }

    pub fn result(&self) -> Result<Recording, String> {
        if self.t.is_empty() {
            return Err(format!(
                "no imaging frame was taken yet: record at least one frame ({} s at {} Hz)",
                crate::sensitivity::fmt_gp(1.0 / self.fps, 3),
                crate::sensitivity::fmt_gp(self.fps, 3)
            ));
        }
        Ok(Recording {
            t_s: self.t.clone(),
            true_dff: self.true_dff.clone(),
            dff: self.dff.clone(),
            photons: self.photons.clone(),
            roi_names: self.roi_names.clone(),
            roi_neurons: self.roi_rows.iter().map(Vec::len).collect(),
            indicator: self.ind.name.to_string(),
            fps: self.fps,
            f0_photons: self.f0,
            dff_per_spike: self.amp,
        })
    }
}

/// `ImagingResult` (the parts the exports use).
#[derive(Debug, Clone)]
pub struct Recording {
    pub t_s: Vec<f64>,
    pub true_dff: Vec<Vec<f64>>,
    pub dff: Vec<Vec<f64>>,
    pub photons: Vec<Vec<f64>>,
    pub roi_names: Vec<String>,
    pub roi_neurons: Vec<usize>,
    pub indicator: String,
    pub fps: f64,
    pub f0_photons: f64,
    pub dff_per_spike: f64,
}

/// `check_seconds`.
pub fn check_seconds(s: f64) -> Result<f64, String> {
    if s > 0.0 && s <= 3600.0 { Ok(s) } else { Err("seconds must be between 0 and 3600".into()) }
}

/// `export_csv`'s text.
pub fn csv_text(res: &Recording) -> String {
    let g6 = |x: f64| crate::sensitivity::fmt_gp(x, 6);
    let mut s = format!(
        "# {TAG_TEXT}\n# indicator={} fps={} f0_photons={} dff_per_spike={}\n",
        res.indicator,
        crate::sensitivity::fmt_gp(res.fps, 4),
        crate::cards::py_float(res.f0_photons),
        crate::cards::py_float(res.dff_per_spike)
    );
    let mut head = vec!["time_s".to_string()];
    head.extend(res.roi_names.iter().map(|n| format!("dFF:{n}")));
    head.extend(res.roi_names.iter().map(|n| format!("dFF_noise_free:{n}")));
    let q = |c: &str| {
        if c.contains([',', '"', '\n', '\r']) { format!("\"{}\"", c.replace('"', "\"\"")) } else { c.to_string() }
    };
    let _ = write!(s, "{}\r\n", head.iter().map(|c| q(c)).collect::<Vec<_>>().join(","));
    for (i, t) in res.t_s.iter().enumerate() {
        let mut row = vec![format!("{t:.4}")];
        row.extend(res.dff[i].iter().map(|&x| g6(x)));
        row.extend(res.true_dff[i].iter().map(|&x| g6(x)));
        let _ = write!(s, "{}\r\n", row.join(","));
    }
    s
}

/// `export_csv`.
pub fn export_csv(res: &Recording, path: &std::path::Path) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(path, csv_text(res))
}

/// The imaging palettes (`LUTS`): black, a dim colour, the colour, near white. `access.palette` picks one.
pub const LUTS: [(&str, [[u8; 3]; 4]); 3] = [
    ("default", [[0, 0, 0], [0, 70, 10], [40, 255, 60], [230, 255, 230]]),
    ("blue-yellow", [[0, 0, 0], [20, 40, 140], [255, 220, 40], [255, 255, 220]]),
    ("high-contrast", [[0, 0, 0], [90, 90, 90], [255, 31, 217], [255, 255, 255]]),
];

/// `apply_lut` for one brightness in 0..1.
pub fn lut_color(palette: &str, lum: f32) -> [u8; 3] {
    let stops = LUTS.iter().find(|l| l.0 == palette).unwrap_or(&LUTS[0]).1;
    let x = lum.clamp(0.0, 1.0) * 3.0;
    let i = (x as usize).min(2);
    let f = x - i as f32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = (f32::from(stops[i][c]) * (1.0 - f) + f32::from(stops[i + 1][c]) * f) as u8;
    }
    out
}
