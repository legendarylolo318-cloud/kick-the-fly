//! Measured personality cards. Python: `lab/tournament.py:measure_card`, `core/individuality.py:compute_personality_card`,
//! `core/cards.py` (the cache key and the cache file `<data>/cache/personality_cards.json`).
//!
//! GAME RULE: the trait words and their cut-offs. CONNECTOME: the three readouts, each taken from the fly's own brain at rest: the
//! giant fiber's latency to its dodge threshold while the looming detectors are driven, the sugar pathway -> MN9 drive ratio, and the
//! DNa01/02 right/left firing ratio. Nothing here is a MODEL PREDICTION. A card that has not been measured shows no numbers.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::Brain;
use crate::csr::Weights;
use crate::lif::{Individuality, LifParams, LifSim};
use crate::pack::Labels;
use crate::util::sha256_hex;

pub const CARD_VERSION: u32 = 2;
pub const LIF_KEYS: [&str; 5] = ["noise_std", "bias", "target_rate_hz", "ext_gain", "gain_adapt"];
pub const LIF_DEFAULTS: [f64; 5] = [0.05, 0.2, 5.0, 4.0, 0.002];
pub const NOT_MEASURED: &str = "card not measured";
pub const HOW_TO_MEASURE: &str =
    "Not measured yet: Esc > Fly arcade > Measure the flies in play (about 10 s of compute per fly, in the background).";
pub const OFF_NOTE: &str = "individuality off: every fly has the same brain, so what differs between these cards is each seed's noise and \
warm-up state, not individuality";

pub const THRESH_LOOM_BOLD_S: f64 = 0.30;
pub const THRESH_LOOM_SKITTISH_S: f64 = 0.20;
pub const THRESH_TURN_BIAS_RATIO: f64 = 1.12;
pub const THRESH_SUGAR_LOVER_RATIO: f64 = 2.20;
pub const LATENCY_CAP_S: f64 = 1.0;
pub const CALM_STEPS: usize = 400;
pub const DRIVE_PRE: usize = 200;
pub const DRIVE_STIM: usize = 200;
pub const WALK_REF_HZ: f64 = 4.32;
const T_ESCAPE: f32 = 4.0;

/// `SIGMAS`: the individuality modes.
pub fn mode_sigma(mode: &str) -> f64 {
    match mode.to_lowercase().as_str() {
        "subtle" => 0.05,
        "strong" => 0.15,
        _ => 0.0,
    }
}

/// `cards.sigma_of`.
pub fn sigma_of(mode: &str, sigma: Option<f64>) -> f64 {
    if mode.eq_ignore_ascii_case("off") { 0.0 } else { sigma.unwrap_or_else(|| mode_sigma(mode)) }
}

/// The five LIF parameters a card depends on: `params` entries over the Lab defaults.
pub fn lif_values(params: &[(String, f64)]) -> [f64; 5] {
    let mut out = LIF_DEFAULTS;
    for (i, k) in LIF_KEYS.iter().enumerate() {
        if let Some((_, v)) = params.iter().find(|(n, _)| n == k) {
            out[i] = *v;
        }
    }
    out
}

/// `LifParams` with those five values.
pub fn lif_params(values: &[f64; 5]) -> LifParams {
    LifParams {
        noise_std: values[0],
        bias: values[1],
        target_rate_hz: values[2],
        ext_gain: values[3],
        gain_adapt: values[4],
        ..LifParams::default()
    }
}

/// Python's `repr(float)` (and so `json.dumps`).
pub fn py_float(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "NaN".into()
        } else if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let e = format!("{x:e}"); // shortest round-trip digits: "1.2345e-5"
    let (mant, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (neg, mant) = mant.strip_prefix('-').map_or((false, mant), |m| (true, m));
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let body = if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{}{}.0", digits, "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        }
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    if neg { format!("-{body}") } else { body }
}

/// `cards.key`: everything a measured card depends on, hashed.
pub fn card_key(seed: u64, mode: &str, sigma: Option<f64>, lif: &[f64; 5], pack_sha: &str) -> String {
    let mode = mode.to_lowercase();
    let r9 = |v: f64| py_float(crate::duel::round_to(v, 9));
    let mut keys: Vec<(&str, f64)> = LIF_KEYS.iter().copied().zip(lif.iter().copied()).collect();
    keys.sort_by(|a, b| a.0.cmp(b.0));
    let lif_s: Vec<String> = keys.iter().map(|(k, v)| format!("\"{k}\": {}", r9(*v))).collect();
    let text = format!(
        "{{\"lif\": {{{}}}, \"mode\": \"{mode}\", \"pack\": \"{pack_sha}\", \"seed\": {seed}, \"sigma\": {}, \"v\": {CARD_VERSION}}}",
        lif_s.join(", "),
        r9(sigma_of(&mode, sigma))
    );
    sha256_hex(text.as_bytes())[..24].to_string()
}

/// SHA-256 of the pack file (`replay.pack_sha256`).
pub fn pack_sha256(path: &std::path::Path) -> Option<String> {
    std::fs::read(path).ok().map(|b| sha256_hex(&b))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    pub seed: u64,
    pub mode: String,
    pub loom_latency_s: f64,
    pub loom_crossed: bool,
    pub sugar_ratio: f64,
    pub turning_ratio: f64,
    pub turning_log_ratio: f64,
    pub walk_level_calm: f64,
    pub title: String,
    pub summary: String,
    pub traits: Vec<String>,
    pub sigma: f64,
    pub lif_params: [f64; 5],
    pub pack_sha256: String,
    pub card_key: String,
}

impl Card {
    /// The card as the Python cache and the tournament JSON hold it.
    pub fn to_json(&self) -> Value {
        let lif: serde_json::Map<String, Value> = LIF_KEYS.iter().zip(self.lif_params).map(|(k, v)| ((*k).to_string(), json!(v))).collect();
        let mut v = json!({
            "seed": self.seed, "mode": self.mode, "loom_latency_s": self.loom_latency_s, "loom_crossed": self.loom_crossed,
            "sugar_ratio": self.sugar_ratio, "turning_ratio": self.turning_ratio, "turning_log_ratio": self.turning_log_ratio,
            "walk_level_calm": self.walk_level_calm, "title": self.title, "summary": self.summary, "traits": self.traits,
            "measured": true, "tmaze_pi": "not measured (a T-maze costs about a minute per fly)", "backend": "rust",
            "card_version": CARD_VERSION, "sigma": self.sigma, "lif_params": lif, "pack_sha256": self.pack_sha256,
            "card_key": self.card_key,
        });
        if self.mode.eq_ignore_ascii_case("off") {
            v["note"] = json!(OFF_NOTE);
        }
        v
    }

    /// A measured, stamped card from the cache or a result file; `None` for anything else.
    pub fn from_json(v: &Value) -> Option<Self> {
        if !v.get("measured")?.as_bool()? {
            return None;
        }
        let f = |k: &str| v.get(k)?.as_f64();
        let mut lif = LIF_DEFAULTS;
        for (i, k) in LIF_KEYS.iter().enumerate() {
            lif[i] = v.get("lif_params")?.get(k)?.as_f64()?;
        }
        Some(Self {
            seed: v.get("seed")?.as_u64()?,
            mode: v.get("mode")?.as_str()?.to_string(),
            loom_latency_s: f("loom_latency_s")?,
            loom_crossed: v.get("loom_crossed").and_then(Value::as_bool).unwrap_or(false),
            sugar_ratio: f("sugar_ratio")?,
            turning_ratio: f("turning_ratio")?,
            turning_log_ratio: f("turning_log_ratio")?,
            walk_level_calm: f("walk_level_calm")?,
            title: v.get("title")?.as_str()?.to_string(),
            summary: v.get("summary")?.as_str()?.to_string(),
            traits: v.get("traits")?.as_array()?.iter().filter_map(|t| t.as_str().map(str::to_string)).collect(),
            sigma: f("sigma")?,
            lif_params: lif,
            pack_sha256: v.get("pack_sha256")?.as_str()?.to_string(),
            card_key: v.get("card_key")?.as_str()?.to_string(),
        })
    }
}

/// `compute_personality_card` for measured numbers: `(title, summary, traits)`.
pub fn trait_words(latency: f64, turning: f64, sugar: f64) -> (String, String, Vec<String>) {
    let temperament = if latency >= THRESH_LOOM_BOLD_S {
        "Bold"
    } else if latency <= THRESH_LOOM_SKITTISH_S {
        "Skittish"
    } else {
        "Alert"
    };
    let steering = if turning >= THRESH_TURN_BIAS_RATIO {
        "Right-turner"
    } else if turning <= 1.0 / THRESH_TURN_BIAS_RATIO {
        "Left-turner"
    } else {
        "Straight-walker"
    };
    let feeding = if sugar >= THRESH_SUGAR_LOVER_RATIO {
        "Sugar lover"
    } else if sugar <= 1.50 {
        "Finicky eater"
    } else {
        "Steady feeder"
    };
    let traits = vec![temperament.to_string(), steering.to_string(), feeding.to_string()];
    (format!("{temperament} {steering}"), traits.join(" · "), traits)
}

/// A brain with everything a cached card depends on.
pub struct Build<'a> {
    pub weights: &'a Arc<Weights>,
    pub labels: &'a Arc<Labels>,
    pub mode: &'a str,
    pub sigma: Option<f64>,
    pub lif: [f64; 5],
}

impl Build<'_> {
    /// `simcore.new_brain(seed, individuality=mode, warmup=600)`, then `Brain.reseed(noise)` when a match noise stream is given
    /// (`tournament.build_fighter`).
    pub fn brain(&self, seed: u64, noise: Option<u64>, warmup: usize) -> Brain {
        self.brain_with(seed, noise, warmup, false)
    }

    /// As [`Build::brain`]; `memory` switches the mushroom-body learning on before the warm-up, as `simcore.new_brain(memory=True)` does.
    pub fn brain_with(&self, seed: u64, noise: Option<u64>, warmup: usize, memory: bool) -> Brain {
        let mut sim = LifSim::new(self.weights.clone(), lif_params(&self.lif), seed);
        let s = sigma_of(self.mode, self.sigma);
        if !self.mode.eq_ignore_ascii_case("off") {
            sim.set_individuality(Individuality::compute(seed, self.weights.n(), s));
        }
        let mut b = Brain::new(sim, self.labels.clone());
        b.set_poke_seed(seed);
        if memory {
            b.enable_memory();
        }
        b.warmup(warmup);
        if let Some(n) = noise {
            b.sim.reseed(n & 0x7FFF_FFFF);
            b.set_poke_seed(n & 0x7FFF_FFFF);
        }
        b
    }
}

/// `measure_card` (the personality card of the individual this seed builds, read from its own brain).
pub fn measure_card(b: &Build, seed: u64, pack_sha: &str, warmup: usize, mut cancelled: impl FnMut() -> bool) -> Result<Card, String> {
    let mut br = b.brain(seed, None, warmup);
    let labels = b.labels.clone();
    let dt = br.sim.p.dt_ms / 1000.0;
    let mut mask = vec![0u8; br.n()];
    let (mut n_right, mut n_left) = (0usize, 0usize);
    for (i, m) in mask.iter_mut().enumerate() {
        let t = labels.types[i].as_str();
        if t == "DNa01" || t == "DNa02" {
            if labels.instance[i].ends_with("_R") {
                *m = 1;
                n_right += 1;
            } else if labels.instance[i].ends_with("_L") {
                *m = 2;
                n_left += 1;
            }
        }
    }
    let mn9 = br.rows_of("type:MN9").map_err(|e| e.to_string())?;
    let loom = br.rows_of("type:LPLC2,LC4").map_err(|e| e.to_string())?;
    let sweet = br.groups_ref().sense("sweet", None).to_vec();
    let snap = br.export_state(seed);
    let mut counts = [0usize; 2];
    let mut walk_levels = Vec::with_capacity(CALM_STEPS / 4);
    for i in 0..CALM_STEPS {
        if cancelled() {
            return Err("cancelled".into());
        }
        for &r in br.step() {
            match mask[r as usize] {
                1 => counts[0] += 1,
                2 => counts[1] += 1,
                _ => {}
            }
        }
        if i % 4 == 3 {
            walk_levels.push(f64::from(br.hz("walk")) / WALK_REF_HZ);
        }
    }
    let hz_r = counts[0] as f64 / (CALM_STEPS as f64 * dt) / n_right.max(1) as f64;
    let hz_l = counts[1] as f64 / (CALM_STEPS as f64 * dt) / n_left.max(1) as f64;
    let turning_ratio = (hz_r + 0.1) / (hz_l + 0.1);
    br.import_state(&snap);
    // pathway_response(sweet -> MN9): baseline Hz and driven Hz, then validation._ratio
    let mut c = [0usize; 2];
    let mut in_mn9 = vec![false; br.n()];
    for &r in &mn9 {
        in_mn9[r as usize] = true;
    }
    for (phase, n) in [(0usize, DRIVE_PRE), (1, DRIVE_STIM)] {
        if phase == 1 {
            br.drive(&sweet, 0.5);
        }
        for _ in 0..n {
            c[phase] += br.step().iter().filter(|&&r| in_mn9[r as usize]).count();
        }
    }
    br.undrive(&sweet);
    let hz = |count: usize, steps: usize| count as f64 / mn9.len().max(1) as f64 / steps.max(1) as f64 / 0.005;
    let (base, driven) = (hz(c[0], DRIVE_PRE), hz(c[1], DRIVE_STIM));
    let sugar_ratio = driven / base.max(0.5);
    br.import_state(&snap);
    for _ in 0..DRIVE_PRE {
        br.step();
    }
    br.drive(&loom, 0.5);
    let (mut latency, mut crossed) = (LATENCY_CAP_S, false);
    for i in 0..(LATENCY_CAP_S / dt) as usize {
        br.step();
        if br.level("escape") > T_ESCAPE {
            latency = (i + 1) as f64 * dt;
            crossed = true;
            break;
        }
    }
    let (title, summary, traits) = trait_words(latency, turning_ratio, sugar_ratio);
    let key = card_key(seed, b.mode, b.sigma, &b.lif, pack_sha);
    Ok(Card {
        seed,
        mode: b.mode.to_lowercase(),
        loom_latency_s: latency,
        loom_crossed: crossed,
        sugar_ratio,
        turning_ratio,
        turning_log_ratio: turning_ratio.ln(),
        walk_level_calm: walk_levels.iter().sum::<f64>() / walk_levels.len().max(1) as f64,
        title,
        summary,
        traits,
        sigma: sigma_of(b.mode, b.sigma),
        lif_params: b.lif,
        pack_sha256: pack_sha.to_string(),
        card_key: key,
    })
}

// --- the cache file (`core/cards.py`: `<data>/cache/personality_cards.json`) --------------------------------------------------------
/// The cards of a cache file by key; an unreadable file, another version or garbage is an empty cache.
pub fn read_cache(path: &std::path::Path) -> std::collections::HashMap<String, Value> {
    let Ok(text) = std::fs::read_to_string(path) else { return Default::default() };
    let Ok(v) = serde_json::from_str::<Value>(&text) else { return Default::default() };
    if v.get("version").and_then(Value::as_u64) != Some(u64::from(CARD_VERSION)) {
        return Default::default();
    }
    v.get("cards").and_then(Value::as_object).map(|m| m.iter().map(|(k, c)| (k.clone(), c.clone())).collect()).unwrap_or_default()
}

/// `cards.lookup`: the measured card for exactly these settings, or None.
pub fn lookup(path: &std::path::Path, key: &str) -> Option<Card> {
    let c = read_cache(path).remove(key)?;
    let card = Card::from_json(&c)?;
    (card.card_key == key).then_some(card)
}

/// `cards.store`: add measured cards to the cache. Never fails the caller: a read-only folder just means the card is missing next time.
pub fn store(path: &std::path::Path, cards: &[Card]) -> usize {
    if cards.is_empty() {
        return 0;
    }
    let mut cur = read_cache(path);
    for c in cards {
        cur.insert(c.card_key.clone(), c.to_json());
    }
    let body = json!({"version": CARD_VERSION, "cards": cur});
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, body.to_string()).is_ok() && std::fs::rename(&tmp, path).is_ok() { cards.len() } else { 0 }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn a_stored_card_is_found_by_its_key_and_only_by_it() {
        let dir = std::env::temp_dir().join(format!("ktf-cards-{}", std::process::id()));
        let path = dir.join("cache").join("personality_cards.json");
        let (title, summary, traits) = trait_words(0.25, 1.0, 2.0);
        let key = card_key(7, "subtle", None, &LIF_DEFAULTS, "abc");
        let card = Card {
            seed: 7,
            mode: "subtle".into(),
            loom_latency_s: 0.25,
            loom_crossed: true,
            sugar_ratio: 2.0,
            turning_ratio: 1.0,
            turning_log_ratio: 0.0,
            walk_level_calm: 1.1,
            title,
            summary,
            traits,
            sigma: 0.05,
            lif_params: LIF_DEFAULTS,
            pack_sha256: "abc".into(),
            card_key: key.clone(),
        };
        assert_eq!(store(&path, std::slice::from_ref(&card)), 1);
        assert_eq!(lookup(&path, &key), Some(card));
        assert!(lookup(&path, "other").is_none());
        std::fs::write(&path, "garbage").unwrap();
        assert!(lookup(&path, &key).is_none(), "a damaged cache is an empty one");
        let _ = std::fs::remove_dir_all(dir);
    }
}
