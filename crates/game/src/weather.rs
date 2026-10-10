//! Rain, gusts and storms for the open field and the orchard. Plain logic. Python: `game/weather.py`.
//!
//! CONNECTOME: raindrop hits poke the fly's touch neurons by body part, wet air the humidity receptors (HRN), gusts the Johnston's organ
//! wind neurons JO-C/E (through the wind transduction: a gust is only a change in the wind speed that goes into it), lightning the
//! photoreceptors. GAME RULE: everything that is not a neuron (how often a drop hits and how hard, which part, when wings count as wet,
//! where gusts and flashes come from, thunder delay, the storm preset, how the scene darkens and flashes). Defaults keep every existing
//! outdoor result: rain 0, gusts 0, storm off.

use crate::rng::Rng;

/// Raindrop hits a second at rain 1.0 on a fly in the open.
pub const HITS_PER_S: f32 = 6.0;
/// Which part a drop hits: the share of the fly's area seen from above.
pub const PART_WEIGHTS: [(&str, f32); 4] = [("wing", 0.35), ("body", 0.40), ("head", 0.15), ("legs", 0.10)];
pub const HIT_STRENGTH: (f32, f32) = (0.15, 0.35);
pub const HUMID_POKE: f32 = 0.2;
pub const HUMID_EVERY_S: f32 = 0.5;
/// Seconds of full rain until the wings are wet.
pub const WET_FILL_S: f32 = 20.0;
pub const WET_DRY_S: f32 = 30.0;
/// `fly.wet` while soaked: the pool's value.
pub const WET_FOR_S: f32 = 3.0;
pub const GUST_LEN_S: (f32, f32) = (1.2, 2.6);
pub const GUST_AMP: (f32, f32) = (2.0, 6.0);
pub const GUST_TURN_DEG: f32 = 25.0;
pub const STORM_RAIN: f32 = 0.7;
pub const STORM_GUST_HZ: f32 = 0.2;
pub const STORM_EXTRA_WIND: f32 = 3.0;
pub const FLASH_EVERY_S: (f32, f32) = (5.0, 14.0);
/// `(start s, brightness, length s)`.
pub const FLASH_PATTERN: [(f32, f32, f32); 3] = [(0.00, 1.00, 0.07), (0.12, 0.60, 0.05), (0.25, 0.85, 0.10)];
pub const THUNDER_DELAY_S: (f32, f32) = (1.0, 3.0);
pub const REDUCED_FLASH_RISE_S: f32 = 0.6;
pub const REDUCED_FLASH_PEAK: f32 = 0.25;
pub const DARKEN_STORM: f32 = 0.45;

/// `(rain, gust rate)` after the storm preset: a storm is at least `STORM_RAIN` of rain and `STORM_GUST_HZ` of gusts.
pub fn effective(rain: f32, gust_hz: f32, storm: bool) -> (f32, f32) {
    let (mut rain, mut gust_hz) = (rain.clamp(0.0, 1.0), gust_hz.clamp(0.0, 0.5));
    if storm {
        rain = rain.max(STORM_RAIN);
        gust_hz = gust_hz.max(STORM_GUST_HZ);
    }
    (rain, gust_hz)
}

#[derive(Debug, Clone, Copy)]
struct Gust {
    start: f32,
    length: f32,
    amp: f32,
    turn_deg: f32,
}

/// A raindrop that hit a fly: `(region, side, strength)` for `Brain.poke`.
pub type RainHit = (&'static str, Option<char>, f32);

#[derive(Debug, Clone)]
pub struct Weather {
    rng: Rng,
    pub t: f32,
    gusts: Vec<Gust>,
    next_flash: f32,
    /// When the current lightning started.
    pub flash_at: Option<f32>,
    thunder: Vec<f32>,
    pub rain: f32,
    pub storm: bool,
    humid: f32,
}

impl Weather {
    /// Whether any gust is blowing (`Weather.gusts` is non-empty).
    pub fn has_gusts(&self) -> bool {
        !self.gusts.is_empty()
    }

    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let next_flash = rng.range(FLASH_EVERY_S);
        Self { rng, t: 0.0, gusts: Vec::new(), next_flash, flash_at: None, thunder: Vec::new(), rain: 0.0, storm: false, humid: 0.0 }
    }

    pub fn update(&mut self, dt: f32, rain: f32, gust_hz: f32, storm: bool) {
        let (r, gust_hz) = effective(rain, gust_hz, storm);
        self.rain = r;
        self.storm = storm;
        self.t += dt;
        if gust_hz > 0.0 && self.rng.f32() < gust_hz * dt {
            let g = Gust {
                start: self.t,
                length: self.rng.range(GUST_LEN_S),
                amp: self.rng.range(GUST_AMP),
                turn_deg: self.rng.uniform(-GUST_TURN_DEG, GUST_TURN_DEG),
            };
            self.gusts.push(g);
        }
        let t = self.t;
        self.gusts.retain(|g| t < g.start + g.length);
        if storm && self.t >= self.next_flash {
            self.flash_at = Some(self.t);
            self.next_flash = self.t + self.rng.range(FLASH_EVERY_S);
            let delay = self.rng.range(THUNDER_DELAY_S);
            self.thunder.push(self.t + delay);
        } else if !storm {
            self.next_flash = self.next_flash.max(self.t + 1.0);
        }
        // the flash stays "on" until both the lightning and the reduced-flashing swell are over
        if let Some(f) = self.flash_at {
            let last = FLASH_PATTERN[FLASH_PATTERN.len() - 1];
            if self.t - f > (last.0 + last.2).max(2.0 * REDUCED_FLASH_RISE_S) {
                self.flash_at = None;
            }
        }
    }

    /// The wind with the gusts in it: `(speed m/s, comes-from direction in degrees)`. A gust is a raised-cosine bump.
    pub fn wind(&self, base_speed: f32, base_dir: f32) -> (f32, f32) {
        let (mut extra, mut turn) = (0.0, 0.0);
        for g in &self.gusts {
            let f = (std::f32::consts::PI * ((self.t - g.start) / g.length).clamp(0.0, 1.0)).sin().powi(2);
            extra += g.amp * f;
            turn += g.turn_deg * f;
        }
        let bonus = if self.storm { STORM_EXTRA_WIND } else { 0.0 };
        ((base_speed + extra + bonus).max(0.0), (base_dir + turn).rem_euclid(360.0))
    }

    /// The raindrops that hit a fly this frame. At most 3 a frame.
    pub fn hits(&mut self, dt: f32, exposed: bool) -> Vec<RainHit> {
        if !exposed || self.rain <= 0.0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for _ in 0..3 {
            if self.rng.f32() < HITS_PER_S * self.rain * dt / 3.0 {
                let x = self.rng.f32();
                let (mut acc, mut part) = (0.0, PART_WEIGHTS[PART_WEIGHTS.len() - 1].0);
                for (name, w) in PART_WEIGHTS {
                    acc += w;
                    if x < acc {
                        part = name;
                        break;
                    }
                }
                let side = if part == "body" || part == "head" {
                    None
                } else if self.rng.f32() < 0.5 {
                    Some('L')
                } else {
                    Some('R')
                };
                let s = self.rng.range(HIT_STRENGTH) * (0.5 + 0.5 * self.rain);
                out.push((part, side, s));
            }
        }
        out
    }

    /// Strength for a poke of the humidity neurons, or 0 when it is not due (every `HUMID_EVERY_S` while raining).
    pub fn humid(&mut self, dt: f32) -> f32 {
        self.humid += dt;
        if self.rain > 0.0 && self.humid >= HUMID_EVERY_S {
            self.humid = 0.0;
            return HUMID_POKE * self.rain;
        }
        0.0
    }

    /// The lightning's brightness now, 0 to 1 (what the photoreceptors are driven by).
    pub fn lightning(&self) -> f32 {
        let Some(f) = self.flash_at else { return 0.0 };
        let e = self.t - f;
        FLASH_PATTERN.iter().find(|(start, _, len)| *start <= e && e < start + len).map_or(0.0, |p| p.1)
    }

    /// How much the scene brightens on screen. With reduced flashing it is a single slow swell no brighter than `REDUCED_FLASH_PEAK`.
    pub fn screen_flash(&self, reduced: bool) -> f32 {
        if !reduced {
            return self.lightning();
        }
        let Some(f) = self.flash_at else { return 0.0 };
        let e = self.t - f;
        let span = REDUCED_FLASH_RISE_S;
        if e < span { REDUCED_FLASH_PEAK * (e / span) } else { (REDUCED_FLASH_PEAK * (1.0 - (e - span) / span)).max(0.0) }
    }

    /// How many thunderclaps have arrived since the last call.
    pub fn thunder_due(&mut self) -> usize {
        let t = self.t;
        let due = self.thunder.iter().filter(|&&x| x <= t).count();
        self.thunder.retain(|&x| x > t);
        due
    }

    /// 0 to `DARKEN_STORM`: how much the scene darkens.
    pub fn darkness(&self) -> f32 {
        DARKEN_STORM * (self.rain / STORM_RAIN.max(1e-6)).clamp(0.0, 1.0) * if self.storm { 1.0 } else { 0.6 }
    }
}

/// The wing-wetness of one fly, 0 to 1: it fills with rain and dries out of it. At 1 the wings count as wet (the game then sets
/// `fly.wet = WET_FOR_S` and the fly can't take off, as after the pool). GAME RULE.
pub fn soak(level: f32, dt: f32, rain: f32, wing_hits: u32) -> f32 {
    let l = if rain > 0.0 { level + dt * rain / WET_FILL_S * (1.0 + 0.5 * wing_hits as f32) } else { level - dt / WET_DRY_S };
    l.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storm_preset_sets_a_floor() {
        assert_eq!(effective(0.1, 0.0, false), (0.1, 0.0));
        assert_eq!(effective(0.1, 0.0, true), (STORM_RAIN, STORM_GUST_HZ));
        assert_eq!(effective(5.0, 5.0, false), (1.0, 0.5));
    }

    #[test]
    fn clear_weather_does_nothing() {
        let mut w = Weather::new(1);
        for _ in 0..3600 {
            w.update(1.0 / 60.0, 0.0, 0.0, false);
            assert!(w.hits(1.0 / 60.0, true).is_empty());
            assert_eq!(w.lightning(), 0.0);
        }
        assert_eq!(w.wind(3.0, 180.0), (3.0, 180.0));
        assert_eq!(w.humid(1.0), 0.0);
    }

    #[test]
    fn rain_hits_about_six_times_a_second_and_mostly_on_wings_and_body() {
        let mut w = Weather::new(2);
        w.update(1.0 / 60.0, 1.0, 0.0, false);
        let mut n = 0;
        let mut by_part = std::collections::HashMap::new();
        for _ in 0..60 * 100 {
            for (part, side, s) in w.hits(1.0 / 60.0, true) {
                n += 1;
                *by_part.entry(part).or_insert(0) += 1;
                assert!((0.075..=0.35).contains(&s));
                assert_eq!(side.is_some(), part == "wing" || part == "legs");
            }
        }
        let per_s = n as f32 / 100.0;
        assert!((5.0..7.0).contains(&per_s), "{per_s}");
        assert!(by_part["wing"] + by_part["body"] > by_part["head"] + by_part["legs"]);
        assert!(w.hits(1.0 / 60.0, false).is_empty(), "under cover nothing hits");
    }

    #[test]
    fn a_storm_flashes_thunders_darkens_and_gusts() {
        let mut w = Weather::new(3);
        let (mut flashes, mut claps, mut peak_wind, mut max_light) = (0, 0, 0.0f32, 0.0f32);
        let mut was_at: Option<f32> = None;
        for _ in 0..60 * 120 {
            w.update(1.0 / 60.0, 0.0, 0.0, true);
            let strike = w.flash_at;
            if strike.is_some() && strike != was_at {
                flashes += 1;
            }
            was_at = strike;
            max_light = max_light.max(w.lightning());
            claps += w.thunder_due();
            peak_wind = peak_wind.max(w.wind(0.0, 0.0).0);
        }
        assert!(flashes >= 5, "{flashes} flashes in two minutes");
        assert!(claps >= flashes - 1, "every flash is followed by thunder");
        assert!(peak_wind > STORM_EXTRA_WIND, "gusts ride on the storm wind");
        assert_eq!(max_light, 1.0);
        assert!((w.darkness() - DARKEN_STORM).abs() < 1e-6);
    }

    #[test]
    fn reduced_flashing_is_one_slow_swell() {
        let mut w = Weather::new(4);
        let mut peak = 0.0f32;
        let mut prev = 0.0f32;
        let mut max_step = 0.0f32;
        for _ in 0..60 * 60 {
            w.update(1.0 / 60.0, 0.0, 0.0, true);
            let f = w.screen_flash(true);
            peak = peak.max(f);
            max_step = max_step.max((f - prev).abs());
            prev = f;
        }
        assert!(peak <= REDUCED_FLASH_PEAK + 1e-6 && peak > 0.2);
        assert!(max_step < 0.01, "no frame-to-frame jump: {max_step}");
    }

    #[test]
    fn wings_get_wet_in_rain_and_dry_out() {
        let mut l = 0.0;
        for _ in 0..60 * 21 {
            l = soak(l, 1.0 / 60.0, 1.0, 0);
        }
        assert!(l >= 1.0 - 1e-3, "wet after about 20 s of full rain: {l}");
        for _ in 0..60 * 31 {
            l = soak(l, 1.0 / 60.0, 0.0, 0);
        }
        assert!(l <= 1e-3);
    }
}
