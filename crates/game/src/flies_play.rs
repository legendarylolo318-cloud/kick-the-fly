//! More than one fly (Python: `Game.spawn_fly`, `cycle_focus`, `_update_focus`, `recall_flies`, `_fly_collisions3d`).
//!
//! Every fly body has its own brain (`fly::Slot` names it; the sim thread steps them all, `Command::SpawnFly`). One brain is *focused*: the
//! HUD, the brain views and every single-brain feature (surgery, memory, challenges) follow it. N spawns a fly, F cycles the focus, J calls
//! lost flies back, and the focus follows the fly nearest to you unless something needs it to stay (a challenge, an open panel, 5 s after F).
//!
//! GAME RULE: the cap (64 flies; every brain is stepped every 5 ms, so past about ten flies the brains run slower than real time, the frame rate does not), the spawn spots, recall, and the
//! bump between two flies (a soft push-apart and, for a real hit, a touch poke on both brains).

use avian3d::prelude::LinearVelocity;
use bevy::prelude::*;

use crate::actions::Actions;
use crate::app::{LatestSnapshot, SimLink};
use crate::environment::Forage;
use crate::feedback::{Notes, Popup, Tag};
use crate::fly::{Fly, FlyPart, Part, Slot, spawn_fly};
use crate::menu_ui::UserConfig;
use crate::player::Player;
use crate::sim_thread::Command;
use crate::world::STAND3;

pub const MAX_FLIES: usize = 64;

#[derive(Resource, Default)]
pub struct Flies {
    /// The slot whose brain is in front.
    pub focus: u32,
    next_slot: u32,
    manual_until: f32,
}

impl Flies {
    /// A fresh slot id for a new fly.
    pub fn alloc(&mut self) -> u32 {
        self.next_slot = self.next_slot.max(self.focus + 1);
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }
}

/// Where the n-th extra fly appears (GAME RULE: near the first fly's spot, spread so they do not start on top of each other).
fn spawn_point(slot: u32) -> Vec3 {
    let x = ((slot * 37) % 11) as f32 * 0.25 - 1.25;
    let z = -0.8 + ((slot * 17) % 7) as f32 * 0.2 - 0.6;
    Vec3::new(x, STAND3 + 0.05, z)
}

#[allow(clippy::too_many_arguments)]
fn spawn_and_cycle(
    mut commands: Commands,
    actions: Res<Actions>,
    cfg: Res<UserConfig>,
    link: Res<SimLink>,
    virt: Res<Time<Virtual>>,
    mut flies: ResMut<Flies>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    bodies: Query<(&Fly, &Slot)>,
    mut notes: ResMut<Notes>,
    mut popups: MessageWriter<Popup>,
    mut pending: Local<Option<u32>>,
    mut tick: Local<u32>,
    lab: Res<crate::share_play::LabState>,
) {
    let now = virt.elapsed_secs();
    let count = bodies.iter().count();
    // KTF_FLIES=n: spawn n more flies at start (for screenshots and checks), one every half second
    let pending = pending.get_or_insert_with(|| std::env::var("KTF_FLIES").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    *tick += 1;
    let auto = *pending > 0 && now > 2.0 && tick.is_multiple_of(30);
    if auto {
        *pending -= 1;
    }
    if actions.just("spawn") || auto {
        if count >= MAX_FLIES {
            notes.push(now, format!("SPAWN    {MAX_FLIES} flies is the most"), Tag::GameRule);
        } else {
            flies.next_slot = flies.next_slot.max(bodies.iter().map(|(_, s)| s.0 + 1).max().unwrap_or(1));
            let slot = flies.next_slot;
            flies.next_slot += 1;
            let sigma = fly_sim::cards::sigma_of(cfg.0.get_str("brain.individuality"), None);
            let pos = spawn_point(slot);
            spawn_fly(&mut commands, &mut meshes, &mut materials, pos, std::f32::consts::FRAC_PI_2, 1000 + u64::from(slot), slot);
            link.0.send(Command::SpawnFly { slot, seed: 1000 + u64::from(slot), sigma });
            crate::share_play::send_model_params(&link, &lab);
            link.0.send(Command::FocusFly(slot));
            flies.focus = slot;
            popups.write(Popup::new(pos + Vec3::Y * 0.45, "NEW FLY!", (170, 255, 200)));
            notes.push(now, format!("SPAWNED  fly #{} (seed {})", count + 1, 1000 + slot), Tag::GameRule);
        }
    }
    if actions.just("cycle_fly") && count > 1 {
        // dead flies are skipped (their brain is silent); only when every fly is dead does the cycle visit the corpses
        let mut slots: Vec<u32> = bodies.iter().filter(|(f, _)| !f.dead() && !f.gone()).map(|(_, s)| s.0).collect();
        if slots.is_empty() {
            slots = bodies.iter().map(|(_, s)| s.0).collect();
        }
        slots.sort_unstable();
        let i = slots.iter().position(|&s| s == flies.focus).unwrap_or(0);
        let next = slots[(i + 1) % slots.len()];
        link.0.send(Command::FocusFly(next));
        flies.focus = next;
        flies.manual_until = now + 5.0;
        // `[card label]`: the focused fly's measured personality card, or "card not measured" (Settings > Fly arcade measures them)
        let seed = if next == 0 { cfg.0.get_i64("brain.seed").max(0) as u64 } else { 1000 + u64::from(next) };
        let label = crate::pet_play::measured_label(seed, cfg.0.get_str("brain.individuality"));
        notes.push(now, format!("FOCUS    fly #{}/{} [{label}]", (i + 1) % slots.len() + 1, slots.len()), Tag::GameRule);
    }
}

/// `_update_focus`: the brain in front follows the fly nearest to you, except while the player has chosen one, a challenge or a panel needs
/// the focus where it is, or every fly is dead.
#[allow(clippy::too_many_arguments)]
fn follow_nearest(
    virt: Res<Time<Virtual>>,
    snap: Res<LatestSnapshot>,
    arcade: Res<crate::arcade::Arcade>,
    mut flies: ResMut<Flies>,
    link: Res<SimLink>,
    bodies: Query<(&Fly, &Slot)>,
    parts: Query<&Transform, With<FlyPart>>,
    player: Query<&Transform, With<Player>>,
    mut last_debug: Local<f32>,
) {
    let now = virt.elapsed_secs();
    // the fly in front died while others live: the focus moves on at once, whatever else would hold it, so no panel, card or kill cam
    // stays bound to a corpse (the death screen waits for the last fly)
    let focus_lost = bodies.iter().find(|(_, s)| s.0 == flies.focus).is_none_or(|(f, _)| f.dead() || f.gone());
    let any_alive = bodies.iter().any(|(f, _)| !f.dead() && !f.gone());
    let blocked = if bodies.iter().count() < 2 {
        Some("one fly")
    } else if focus_lost && any_alive {
        None
    } else if arcade.open {
        Some("arcade open")
    } else if now < flies.manual_until {
        Some("chosen with F")
    } else if !any_alive {
        Some("all dead")
    } else if snap.0.as_deref().is_some_and(|s| s.challenge.is_some()) {
        Some("challenge")
    } else if snap.0.as_deref().is_some_and(|s| s.memory.as_ref().is_some_and(|m| m.train.is_some())) {
        Some("training")
    } else {
        None
    };
    // KTF_DEBUG=1: once a second, who is in front and why the focus may not move
    let debug = std::env::var_os("KTF_DEBUG").is_some() && now - *last_debug >= 1.0;
    if debug {
        *last_debug = now;
        eprintln!("focus: game {} sim {:?} blocked {blocked:?}", flies.focus, snap.0.as_deref().map(|s| s.focus));
    }
    if blocked.is_some() {
        return;
    }
    let Ok(me) = player.single().map(|t| t.translation) else { return };
    let mut best: Option<(f32, u32)> = None;
    for (f, slot) in &bodies {
        if f.dead() || f.gone() {
            continue;
        }
        let Ok(t) = parts.get(f.parts[Part::Thorax.index()]) else { continue };
        let d = t.translation.distance(me);
        if debug {
            eprintln!("focus:   slot {} at {:.2} m", slot.0, d);
        }
        if best.is_none_or(|b| d < b.0) {
            best = Some((d, slot.0));
        }
    }
    if let Some((_, s)) = best
        && s != flies.focus
    {
        link.0.send(Command::FocusFly(s));
        flies.focus = s;
    }
}

/// J: every lost fly is put back in front of you (`recall_flies`).
fn recall(
    actions: Res<Actions>,
    virt: Res<Time<Virtual>>,
    mut bodies: Query<(&Fly, &mut Forage)>,
    mut parts: Query<(&mut Transform, &mut LinearVelocity), With<FlyPart>>,
    mut notes: ResMut<Notes>,
) {
    if !actions.just("recall") {
        return;
    }
    let now = virt.elapsed_secs();
    let mut n = 0;
    for (f, mut fo) in &mut bodies {
        if !fo.lost {
            continue;
        }
        let Ok(thx) = parts.get(f.parts[Part::Thorax.index()]).map(|p| p.0.translation) else { continue };
        let target = spawn_point(fo.slot);
        for e in f.parts {
            if let Ok((mut t, mut v)) = parts.get_mut(e) {
                t.translation = target + (t.translation - thx);
                v.0 = Vec3::ZERO;
            }
        }
        fo.lost = false;
        n += 1;
    }
    if n == 0 {
        notes.push(now, "RECALL   no fly is lost", Tag::GameRule);
    } else {
        notes.push(now, format!("RECALL   {n} lost fl{} called back", if n == 1 { "y" } else { "ies" }), Tag::GameRule);
    }
}

/// Two flies that bump: the bodies push apart by themselves (they collide), and a real bump pokes the touch neurons of both brains
/// (`_fly_collisions3d`: closing faster than 0.02 per tick, strength `clip(closing / 0.12, 0.15, 1)`).
fn collisions(
    link: Res<SimLink>,
    bodies: Query<(Entity, &Fly, &Slot)>,
    parts: Query<(&Transform, &LinearVelocity), With<FlyPart>>,
    mut cool: Local<std::collections::HashMap<(u32, u32), f32>>,
    virt: Res<Time<Virtual>>,
) {
    let now = virt.elapsed_secs();
    let list: Vec<_> = bodies.iter().filter(|(_, f, _)| !f.dead() && f.dissolved_at.is_none() && f.shattered_at.is_none()).collect();
    for i in 0..list.len() {
        for j in i + 1..list.len() {
            let (_, fa, sa) = list[i];
            let (_, fb, sb) = list[j];
            let (Ok((ta, va)), Ok((tb, vb))) = (parts.get(fa.parts[Part::Thorax.index()]), parts.get(fb.parts[Part::Thorax.index()]))
            else {
                continue;
            };
            let d = tb.translation - ta.translation;
            let dist = d.length();
            if dist > FLY_TOUCH_RADIUS * 2.0 || dist < 1e-6 {
                continue;
            }
            let closing = -(vb.0 - va.0).dot(d / dist) / 60.0; // metres per 1/60 s, closing is positive
            let key = (sa.0.min(sb.0), sa.0.max(sb.0));
            if closing > 0.02 && now >= cool.get(&key).copied().unwrap_or(0.0) {
                cool.insert(key, now + 0.25);
                let s = (closing / 0.12).clamp(0.15, 1.0);
                for slot in [sa.0, sb.0] {
                    link.0.send(Command::PokeFly { slot, region: "body".into(), side: None, strength: 0.5 * s });
                    link.0.send(Command::PokeFly { slot, region: "legs".into(), side: None, strength: 0.3 * s });
                }
            }
        }
    }
}

/// `k2.FLY_TOUCH_RADIUS * S`: how close two flies get before they bump, in metres.
const FLY_TOUCH_RADIUS: f32 = 40.0 * 0.006;

/// The seeds of the flies in play, for the arcade page's "Measure the flies in play".
fn publish_play(bodies: Query<&Slot, With<Fly>>, cfg: Res<UserConfig>, mut hub: ResMut<crate::arcade_hub::ArcadeHub>) {
    let mut seeds: Vec<u64> = bodies.iter().map(|s| 1000 + u64::from(s.0)).collect();
    seeds.sort_unstable();
    hub.set_play(seeds, cfg.0.get_str("brain.individuality"));
}

pub struct FliesPlugin;

impl Plugin for FliesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Flies>().add_systems(Update, (spawn_and_cycle, follow_nearest, recall, collisions, publish_play));
    }
}
