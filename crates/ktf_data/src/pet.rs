//! Pet mode (`core/pet.py:PetManager`): one long-lived fly whose needs advance with the wall clock.
//!
//! The simulation only runs while the game is open; on launch the time since the last save is caught up by a deterministic rule (clamped to
//! 0..7 days). Needs and couplings are GAME RULE; the neurons they drive are real connectome groups. Every method takes `now` (seconds since
//! the Unix epoch) so the rules are testable against Python, and `update_live` takes the random roll Python drew with `random.random()`.
//! The save is `pet.ktfsave` (+ `.bak`), JSON with sorted keys here (Python kept insertion order; both read each other's files).

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

pub const MAX_CATCHUP_SECONDS: f64 = 7.0 * 86400.0;
pub const HUNGER_RATE_PER_SEC: f64 = 1.0 / 86400.0;
pub const SLEEP_RATE_PER_SEC: f64 = 1.0 / (16.0 * 3600.0);
pub const SLEEP_RECOVERY_RATE: f64 = 1.0 / (8.0 * 3600.0);
pub const HUNGER_SUGAR_GAIN: f64 = 2.0;
pub const HUNGER_PAM_GAIN: f64 = 1.5;
pub const SLEEP_DFB_DRIVE: f64 = 0.5;

/// The pet's rates (`lab.apply_to_game`: Lab > Parameters `pet.*`); [`Rates::default`] is the shipped rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    pub hunger_per_s: f64,
    pub sleep_per_s: f64,
    pub recovery_per_s: f64,
    pub sugar_gain: f64,
    pub pam_gain: f64,
    pub dfb_drive: f64,
    pub max_catchup_s: f64,
}

impl Default for Rates {
    fn default() -> Self {
        Self {
            hunger_per_s: HUNGER_RATE_PER_SEC,
            sleep_per_s: SLEEP_RATE_PER_SEC,
            recovery_per_s: SLEEP_RECOVERY_RATE,
            sugar_gain: HUNGER_SUGAR_GAIN,
            pam_gain: HUNGER_PAM_GAIN,
            dfb_drive: SLEEP_DFB_DRIVE,
            max_catchup_s: MAX_CATCHUP_SECONDS,
        }
    }
}

impl Rates {
    /// From the Lab's parameter values (names as `lab.PARAMS`); anything missing keeps the default.
    pub fn from_params(get: impl Fn(&str) -> Option<f64>) -> Self {
        let d = Self::default();
        Self {
            hunger_per_s: get("pet.hunger_h").map_or(d.hunger_per_s, |h| 1.0 / (h * 3600.0)),
            sleep_per_s: get("pet.awake_h").map_or(d.sleep_per_s, |h| 1.0 / (h * 3600.0)),
            recovery_per_s: get("pet.sleep_h").map_or(d.recovery_per_s, |h| 1.0 / (h * 3600.0)),
            sugar_gain: get("pet.sugar_gain").unwrap_or(d.sugar_gain),
            pam_gain: get("pet.pam_gain").unwrap_or(d.pam_gain),
            dfb_drive: get("pet.dfb_drive").unwrap_or(d.dfb_drive),
            max_catchup_s: get("pet.max_catchup_d").map_or(d.max_catchup_s, |d| d * 86400.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pet {
    pub rates: Rates,
    pub dir: PathBuf,
    pub seed: i64,
    pub brain_type: String,
    pub born_time: f64,
    pub last_saved_time: f64,
    pub hunger: f64,
    pub sleep_pressure: f64,
    pub is_sleeping: bool,
    pub mood: f64,
    pub real_stakes: bool,
    pub is_dead: bool,
    pub timeline: Vec<Value>,
    pub personality_card: Value,
    pub legacy_personality_card: Option<Value>,
    pub welcome_message: String,
}

fn event(ts: f64, kind: &str, text: &str) -> Value {
    json!({"timestamp": ts, "event_type": kind, "description": text})
}

/// `cards.check_pet_card`: a measured, stamped card of the current version.
pub fn check_pet_card(card: &Value) -> bool {
    card["measured"] == json!(true)
        && card["card_version"].as_u64() == Some(u64::from(fly_sim::cards::CARD_VERSION))
        && card["card_key"].as_str().is_some_and(|k| !k.is_empty())
}

impl Pet {
    pub fn new(dir: PathBuf, now: f64) -> Self {
        Self {
            rates: Rates::default(),
            dir,
            seed: 12345,
            brain_type: "adult".into(),
            born_time: now,
            last_saved_time: now,
            hunger: 0.2,
            sleep_pressure: 0.1,
            is_sleeping: false,
            mood: 0.6,
            real_stakes: false,
            is_dead: false,
            timeline: Vec::new(),
            personality_card: json!({}),
            legacy_personality_card: None,
            welcome_message: String::new(),
        }
    }

    pub fn save_file(&self) -> PathBuf {
        self.dir.join("pet.ktfsave")
    }

    pub fn backup_file(&self) -> PathBuf {
        self.dir.join("pet.ktfsave.bak")
    }

    pub fn exists(&self) -> bool {
        self.save_file().exists() || self.backup_file().exists()
    }

    pub fn create_new(&mut self, seed: Option<i64>, brain_type: &str, now: f64) {
        self.seed = seed.unwrap_or((now as i64) % 100_000);
        self.brain_type = brain_type.into();
        self.born_time = now;
        self.last_saved_time = now;
        self.hunger = 0.15;
        self.sleep_pressure = 0.10;
        self.is_sleeping = false;
        self.mood = 0.70;
        self.real_stakes = false;
        self.is_dead = false;
        self.timeline = vec![event(now, "born", &format!("Adopted {} fly!", self.brain_type))];
        self.personality_card = json!({});
        self.legacy_personality_card = None;
        self.welcome_message = format!("Welcome to Pet Mode! Meet your new {}.", self.brain_type);
        self.save(now);
    }

    /// `load_or_create`: the save (or its backup) with the catch-up applied, or a new pet. An unreadable save is kept under a new name.
    pub fn load_or_create(&mut self, brain_type: &str, now: f64) {
        if !self.exists() {
            self.create_new(None, brain_type, now);
            return;
        }
        let primary = self.save_file().exists();
        let target = if primary { self.save_file() } else { self.backup_file() };
        if self.try_load(&target, now) {
            return;
        }
        if primary && self.backup_file().exists() && self.try_load(&self.backup_file(), now) {
            return;
        }
        self.quarantine(now);
        self.create_new(None, brain_type, now);
    }

    fn try_load(&mut self, file: &Path, now: f64) -> bool {
        let Ok(text) = std::fs::read_to_string(file) else { return false };
        let Ok(data) = serde_json::from_str::<Value>(&text) else { return false };
        if !data.is_object() {
            return false;
        }
        self.unpack(&data, now);
        self.apply_catchup(now);
        true
    }

    fn quarantine(&self, now: f64) {
        let stamp = format!("{}", now as i64);
        for f in [self.save_file(), self.backup_file()] {
            if f.exists() {
                let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("pet");
                let _ = std::fs::rename(&f, f.with_file_name(format!("{name}.unreadable-{stamp}")));
            }
        }
    }

    fn unpack(&mut self, d: &Value, now: f64) {
        let f = |k: &str, default: f64| d[k].as_f64().unwrap_or(default);
        self.seed = d["seed"].as_i64().unwrap_or(12345);
        self.brain_type = d["brain_type"].as_str().unwrap_or("adult").into();
        self.born_time = f("born_time", now);
        self.last_saved_time = f("last_saved_time", now);
        self.hunger = f("hunger", 0.2);
        self.sleep_pressure = f("sleep_pressure", 0.1);
        self.is_sleeping = d["is_sleeping"].as_bool().unwrap_or(false);
        self.mood = f("mood", 0.6);
        self.real_stakes = d["real_stakes"].as_bool().unwrap_or(false);
        self.is_dead = d["is_dead"].as_bool().unwrap_or(false);
        self.timeline = d["timeline"].as_array().cloned().unwrap_or_default();
        let card = d.get("personality_card").cloned().unwrap_or(Value::Null);
        self.legacy_personality_card = d.get("legacy_personality_card").filter(|v| v.is_object()).cloned();
        if check_pet_card(&card) {
            self.personality_card = card;
        } else {
            self.personality_card = json!({});
            if card.as_object().is_some_and(|o| !o.is_empty()) && self.legacy_personality_card.is_none() {
                self.legacy_personality_card = Some(card);
            }
        }
    }

    /// `_apply_catchup`: the wall-clock rule on launch.
    pub fn apply_catchup(&mut self, now: f64) {
        let mut elapsed = now - self.last_saved_time;
        if elapsed < 0.0 {
            elapsed = 0.0;
        } else if elapsed > self.rates.max_catchup_s {
            elapsed = self.rates.max_catchup_s;
        }
        let hours = elapsed / 3600.0;
        self.hunger = (self.hunger + elapsed * self.rates.hunger_per_s).min(1.0);
        if hours >= 4.0 {
            self.sleep_pressure = (self.sleep_pressure - 0.7).max(0.05);
            self.is_sleeping = false;
            self.timeline.push(event(now, "sleep", &format!("Slept while away ({hours:.1}h)")));
        } else {
            self.sleep_pressure = (self.sleep_pressure + elapsed * self.rates.sleep_per_s).min(1.0);
        }
        if self.real_stakes && self.hunger >= 1.0 && hours >= 48.0 {
            self.is_dead = true;
            self.timeline.push(event(now, "injury", "Starved from prolonged neglect."));
        }
        self.welcome_message = if hours < 0.1 {
            "Welcome back! Your fly is buzzing cheerfully.".into()
        } else if hours < 2.0 {
            format!("Welcome back! You were away for {} minutes.", (hours * 60.0) as i64)
        } else if hours < 24.0 {
            format!("Welcome back! You were away for {hours:.1} hours. Your fly is getting hungry.")
        } else {
            format!("Welcome back! You were away for {:.1} days. Your fly missed you!", hours / 24.0)
        };
        self.last_saved_time = now;
        self.save(now);
    }

    /// `feed`: `save = false` for the per-frame feeding in the game (saved on quit).
    pub fn feed(&mut self, amount: f64, source: &str, save: bool, now: f64) {
        self.hunger = (self.hunger - amount).max(0.0);
        self.mood = (self.mood + 0.15 * (amount / 0.35).min(1.0)).min(1.0);
        if save {
            self.timeline.push(event(now, "meal", &format!("Fed on {source}")));
            self.save(now);
        }
    }

    pub fn record_training(&mut self, task: &str, now: f64) {
        self.timeline.push(event(now, "training", &format!("Trained in {task}")));
        self.save(now);
    }

    pub fn record_injury(&mut self, source: &str, now: f64) {
        self.mood = (self.mood - 0.25).max(0.0);
        self.timeline.push(event(now, "injury", &format!("Injured by {source}")));
        self.save(now);
    }

    pub fn sugar_gain_multiplier(&self) -> f64 {
        1.0 + self.rates.sugar_gain * self.hunger
    }

    pub fn pam_reward_multiplier(&self) -> f64 {
        1.0 + self.rates.pam_gain * self.hunger
    }

    pub fn dfb_sleep_drive(&self) -> f64 {
        self.rates.dfb_drive * self.sleep_pressure
    }

    /// `update_live`: one step while the game runs. `roll` is Python's `random.random()`, drawn only when tired or at night and awake.
    pub fn update_live(&mut self, dt: f64, is_night: bool, pam_rate: f64, ppl1_rate: f64, roll: &mut dyn FnMut() -> f64, now: f64) {
        self.hunger = (self.hunger + dt * self.rates.hunger_per_s * 1.5).min(1.0);
        if self.is_sleeping {
            self.sleep_pressure = (self.sleep_pressure - dt * self.rates.recovery_per_s).max(0.0);
            if self.sleep_pressure < 0.1 && !is_night {
                self.is_sleeping = false;
                self.timeline.push(event(now, "wake", "Woke up refreshed."));
            }
        } else {
            self.sleep_pressure = (self.sleep_pressure + dt * self.rates.sleep_per_s).min(1.0);
            if (self.sleep_pressure > 0.85 || is_night) && roll() < 0.005 {
                self.is_sleeping = true;
                self.timeline.push(event(now, "sleep", "Fell asleep."));
            }
        }
        let tot = pam_rate + ppl1_rate;
        if tot > 0.01 {
            let target = 0.5 + 0.5 * (pam_rate - ppl1_rate) / tot;
            self.mood += (target - self.mood) * 0.05;
            self.mood = self.mood.clamp(0.0, 1.0);
        }
    }

    /// `save`: write through a temp file, with the previous save kept as `.bak`.
    pub fn save(&mut self, now: f64) {
        self.last_saved_time = now;
        let tail = self.timeline.len().saturating_sub(50);
        let mut o = Map::new();
        for (k, v) in [
            ("seed", json!(self.seed)),
            ("brain_type", json!(self.brain_type)),
            ("born_time", json!(self.born_time)),
            ("last_saved_time", json!(self.last_saved_time)),
            ("hunger", json!(self.hunger)),
            ("sleep_pressure", json!(self.sleep_pressure)),
            ("is_sleeping", json!(self.is_sleeping)),
            ("mood", json!(self.mood)),
            ("real_stakes", json!(self.real_stakes)),
            ("is_dead", json!(self.is_dead)),
            ("timeline", Value::Array(self.timeline[tail..].to_vec())),
            ("personality_card", self.personality_card.clone()),
        ] {
            o.insert(k.into(), v);
        }
        if let Some(l) = &self.legacy_personality_card {
            o.insert("legacy_personality_card".into(), l.clone());
        }
        let _ = std::fs::create_dir_all(&self.dir);
        if self.save_file().exists() {
            let _ = std::fs::copy(self.save_file(), self.backup_file());
        }
        let tmp = self.dir.join("pet.ktfsave.tmp");
        if std::fs::write(&tmp, serde_json::to_string_pretty(&Value::Object(o)).unwrap_or_default()).is_ok() {
            let _ = std::fs::rename(&tmp, self.save_file());
        }
    }

    /// The HUD widget's mood word (`_draw_pet_widget`).
    pub fn mood_word(&self) -> &'static str {
        if self.is_sleeping {
            "sleeping (ZZZ)"
        } else if self.mood > 0.6 {
            "content"
        } else if self.mood > 0.4 {
            "happy"
        } else {
            "grumpy"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ktf-pet-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn catchup_rules() {
        let d = tmp("catch");
        let mut p = Pet::new(d.clone(), 1000.0);
        p.create_new(Some(7), "adult", 1000.0);
        p.hunger = 0.2;
        p.sleep_pressure = 0.3;
        // 2 hours: hunger +2/24, sleep pressure +2/16, no sleep
        p.apply_catchup(1000.0 + 7200.0);
        assert!((p.hunger - (0.2 + 7200.0 / 86400.0)).abs() < 1e-12);
        assert!((p.sleep_pressure - (0.3 + 7200.0 / 57600.0)).abs() < 1e-12);
        assert!(p.welcome_message.contains("2.0 hours"));
        // a clock moved back changes nothing
        let h = p.hunger;
        p.apply_catchup(10.0);
        assert_eq!(p.hunger, h);
        // away 9 days is clamped to 7; 4+ hours means a sleep
        let t = p.last_saved_time;
        p.apply_catchup(t + 9.0 * 86400.0);
        assert_eq!(p.hunger, 1.0);
        assert!(p.welcome_message.contains("7.0 days"));
        assert_eq!(p.timeline.last().unwrap()["event_type"], "sleep");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn save_load_backup_and_quarantine() {
        let d = tmp("save");
        let mut p = Pet::new(d.clone(), 5.0);
        p.create_new(Some(3), "adult", 5.0);
        p.feed(0.1, "sugar", true, 6.0);
        p.save(7.0);
        let mut q = Pet::new(d.clone(), 8.0);
        q.load_or_create("adult", 8.0);
        assert_eq!(q.seed, 3);
        assert!(q.timeline.iter().any(|e| e["event_type"] == "meal"));
        // damage the primary: the backup is used
        std::fs::write(q.save_file(), "{not json").unwrap();
        let mut r = Pet::new(d.clone(), 9.0);
        r.load_or_create("adult", 9.0);
        assert_eq!(r.seed, 3);
        // damage both: they are kept under new names and a new pet starts
        std::fs::write(r.save_file(), "x").unwrap();
        std::fs::write(r.backup_file(), "y").unwrap();
        let mut s = Pet::new(d.clone(), 10.0);
        s.load_or_create("larva", 10.0);
        assert_eq!(s.brain_type, "larva");
        let kept = std::fs::read_dir(&d).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains("unreadable")).count();
        assert_eq!(kept, 2);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn live_update_and_couplings() {
        let mut p = Pet::new(tmp("live"), 0.0);
        p.hunger = 0.5;
        p.sleep_pressure = 0.9;
        p.update_live(1.0, false, 3.0, 1.0, &mut || 0.001, 1.0);
        assert!(p.is_sleeping, "tired and the roll came up");
        assert!((p.sugar_gain_multiplier() - (1.0 + 2.0 * p.hunger)).abs() < 1e-12);
        assert!((p.dfb_sleep_drive() - 0.5 * p.sleep_pressure).abs() < 1e-12);
        assert!(p.mood > 0.6, "PAM above PPL1 lifts the mood");
        p.sleep_pressure = 0.05;
        p.update_live(1.0, false, 0.0, 0.0, &mut || 0.5, 2.0);
        assert!(!p.is_sleeping);
    }
}
