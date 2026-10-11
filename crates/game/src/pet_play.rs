//! Pet mode in the game (`core/pet.py` coupled to the brain, `kick_the_fly.py` lines 5869, 6160 and the pet widget).
//!
//! Hunger scales the sugar taste and PAM reward pokes of eating; sleep pressure drives the dFB sleep neurons; the mood follows the PAM and
//! PPL1 levels; with real stakes on, a pet that starved dies. The state is `pet.ktfsave` in the data folder's `pet/`, caught up on launch and
//! saved every minute and on quit. All of the rates are GAME RULE (Lab > Parameters, `pet.*`).

use bevy::app::AppExit;
use bevy::prelude::*;
use ktf_data::pet::Pet;

use crate::app::{LatestSnapshot, SimLink};
use crate::fly::{FlyDamage, Slot};
use crate::menu_ui::UserConfig;
use crate::sim_thread::Command;

#[derive(Resource, Default)]
pub struct PetRes {
    pub pet: Option<Pet>,
    /// `cards.label(pet_card(...))`, refreshed once a second.
    pub label: String,
    since_save: f32,
    since_label: f32,
    rng: u64,
    pub welcome_shown: bool,
}

impl PetRes {
    pub fn sugar_gain(&self) -> f32 {
        self.pet.as_ref().map_or(1.0, |p| p.sugar_gain_multiplier() as f32)
    }

    pub fn pam_gain(&self) -> f32 {
        self.pet.as_ref().map_or(1.0, |p| p.pam_reward_multiplier() as f32)
    }

    /// Per eating frame (`pet.feed(amount=0.003, save=False)`).
    pub fn feed_frame(&mut self) {
        if let Some(p) = self.pet.as_mut() {
            p.feed(0.003, "sugar", false, now_s());
        }
    }

    fn roll(&mut self) -> f64 {
        // xorshift64*: only the 0.5 % per-frame chance of falling asleep needs it
        let mut x = self.rng.max(1);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn now_s() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// The Lab's `pet.*` rule parameters (changed ones from the page, the others at their defaults).
fn rates(lab: &crate::share_play::LabState) -> ktf_data::pet::Rates {
    ktf_data::pet::Rates::from_params(|n| lab.values.get(n).copied().or_else(|| ktf_data::labparams::by_name(n).map(|p| p.default)))
}

fn pet_dir() -> std::path::PathBuf {
    ktf_data::paths::get().data_dir.join("pet")
}

fn update(
    mut res: ResMut<PetRes>,
    cfg: Res<UserConfig>,
    snap: Res<LatestSnapshot>,
    lab: Res<crate::share_play::LabState>,
    link: Res<SimLink>,
    time: Res<Time<Real>>,
    flies: Query<(Entity, &crate::fly::Fly, &Slot)>,
    mut dmg: MessageWriter<FlyDamage>,
) {
    let want = cfg.0.get_str("brain.mode") == "pet";
    if want && res.pet.is_none() {
        let now = now_s();
        let mut p = Pet::new(pet_dir(), now);
        p.rates = rates(&lab);
        let brain = if cfg.0.larva() { "larva" } else { "adult" };
        p.load_or_create(brain, now);
        res.rng = (now * 1000.0) as u64 ^ 0x9E37_79B9_7F4A_7C15;
        res.pet = Some(p);
    }
    if res.pet.is_none() {
        return;
    }
    let real = cfg.0.get_bool("brain.pet_real_stakes");
    let (reward, punish) = snap
        .0
        .as_deref()
        .and_then(|s| {
            let i = s.info.as_deref()?;
            let at = |n: &str| i.group_names.iter().position(|g| g == n).and_then(|k| s.levels.get(k).copied());
            Some((at("reward")?, at("punish").unwrap_or(0.0)))
        })
        .unwrap_or((1.0, 0.0));
    let mut roll = res.roll();
    let now = now_s();
    let dt = time.delta_secs();
    res.since_save += dt;
    res.since_label += dt;
    let (save_now, relabel) = (res.since_save >= 60.0, res.since_label >= 1.0);
    if save_now {
        res.since_save = 0.0;
    }
    if relabel {
        res.since_label = 0.0;
    }
    let primary = flies.iter().find(|(_, _, s)| s.0 == 0);
    let alive = primary.is_some_and(|(_, f, _)| !f.dead());
    let mut drive = 0.0;
    let mut kill = None;
    let mut label = None;
    if let Some(p) = res.pet.as_mut() {
        p.real_stakes = real;
        p.rates = rates(&lab);
        if alive {
            // Python advances 0.016 s per frame (its 60 Hz frame at 0.016), so the pet's clock runs with frames, not wall time
            let mut r = || std::mem::replace(&mut roll, 1.0);
            p.update_live(0.016, false, f64::from(reward), f64::from(punish), &mut r, now);
            drive = p.dfb_sleep_drive();
            if p.real_stakes && p.is_dead {
                kill = primary.map(|(e, _, _)| e);
            }
        }
        if save_now {
            p.save(now);
        }
        if relabel {
            label = Some(card_label(p));
        }
    }
    if drive > 0.05 && snap.0.as_deref().is_some_and(|s| s.info.is_some()) {
        link.0.send(Command::Poke { region: "sleep".into(), side: None, strength: drive as f32 });
    }
    if let Some(e) = kill {
        dmg.write(FlyDamage { fly: e, amount: 999.0, source: "starvation" });
    }
    if let Some(l) = label {
        res.label = l;
    }
}

/// `cards.label(pet_card(pet, slot))`: the measured card of the original fly (cache), or the one the pet file kept for the same key.
fn card_label(p: &mut Pet) -> String {
    let mode = "subtle";
    let key = fly_sim::cards::card_key(p.seed.max(0) as u64, mode, None, &fly_sim::cards::LIF_DEFAULTS, &crate::arcade_hub::pack_sha());
    if let Some(c) = fly_sim::cards::lookup(&crate::arcade_hub::cache_path(), &key) {
        return format!("{} (measured)", c.title);
    }
    let stored = &p.personality_card;
    if ktf_data::pet::check_pet_card(stored) && stored["card_key"].as_str() == Some(key.as_str()) {
        return format!("{} (measured)", stored["title"].as_str().unwrap_or(""));
    }
    "card not measured".into()
}

fn save_on_exit(mut exit: MessageReader<AppExit>, mut res: ResMut<PetRes>) {
    if exit.read().count() > 0
        && let Some(p) = res.pet.as_mut()
    {
        p.save(now_s());
    }
}

pub struct PetPlugin;

impl Plugin for PetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PetRes>().add_systems(Update, update).add_systems(Last, save_on_exit);
    }
}
