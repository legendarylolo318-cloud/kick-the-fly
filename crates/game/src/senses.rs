//! What the fly can perceive of the world, turned into pokes on its sensory neurons. Python: `Game3D._threats`, `_vision`, `_scents`.
//!
//! LOOMING (CONNECTOME transduction): the fly does not "know" anything is coming. Every object that can hurt it reports itself as a threat
//! (position, radius); each tick the angular size seen from the fly's head, `2 atan(r/d)`, is differenced to get how fast it is
//! expanding in rad/s. Above `LOOM_MIN` that drives the real looming detectors LPLC2/LC4 (which excite the giant fiber DNp01), by
//! `(rate - LOOM_MIN) / LOOM_FULL`. Sneaking up slowly therefore works, as a consequence of that rule. GAME RULE: the threshold and gain.
//! SCENT: tools, treats and the vinegar poke the real olfactory neurons of their own glomeruli (`scent:<tool>`).

use std::collections::HashMap;

use bevy::prelude::*;

use crate::app::SimLink;
use crate::fly::{Fly, FlyPart, Part};
use crate::sim_thread::Command;

#[derive(Debug, Clone, Copy)]
pub struct Threat {
    pub key: u64,
    pub pos: Vec3,
    pub radius: f32,
}

/// Filled by the systems that own threatening things (player, tools, bombs, predators, the cook) each fixed tick, read by [`vision`].
#[derive(Resource, Default)]
pub struct Threats(pub Vec<Threat>);

impl Threats {
    pub fn push(&mut self, key: impl std::hash::Hash, pos: Vec3, radius: f32) {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        self.0.push(Threat { key: h.finish(), pos, radius });
    }
}

/// Smells in the world: a position and which scent group it drives, and how far it reaches.
#[derive(Debug, Clone)]
pub struct ScentSource {
    pub pos: Vec3,
    pub scent: &'static str,
    pub range: f32,
    pub strength: f32,
    /// Only the horizontal distance counts (treats on the floor).
    pub flat: bool,
}

#[derive(Resource, Default)]
pub struct Scents(pub Vec<ScentSource>);

/// Per-fly memory of last tick's angular sizes.
#[derive(Component, Default)]
pub struct LoomMemory {
    prev: HashMap<u64, f32>,
    /// The fastest expansion seen this tick (rad/s), for the HUD.
    pub rate: f32,
}

fn clear(mut threats: ResMut<Threats>, mut scents: ResMut<Scents>) {
    threats.0.clear();
    scents.0.clear();
}

fn attach_memory(mut commands: Commands, new: Query<Entity, (With<Fly>, Without<LoomMemory>)>) {
    for e in &new {
        commands.entity(e).insert(LoomMemory::default());
    }
}

/// Difference each threat's angular size and poke the looming detectors with the fastest-growing one.
fn vision(
    threats: Res<Threats>,
    link: Res<SimLink>,
    time: Res<Time<Fixed>>,
    mut flies: Query<(&mut Fly, &mut LoomMemory, &crate::fly::Slot)>,
    parts: Query<(&FlyPart, &Transform)>,
) {
    let dt = time.delta_secs().max(1e-4);
    for (mut f, mut mem, slot) in &mut flies {
        if f.dead() || f.frozen_at.is_some() {
            mem.rate = 0.0;
            continue;
        }
        let Ok((_, head)) = parts.get(f.parts[Part::Head.index()]) else { continue };
        seen(&threats, &mut mem, &mut f, head.translation, dt, &link, slot.0);
    }
}

fn seen(threats: &Threats, mem: &mut LoomMemory, f: &mut Fly, head: Vec3, dt: f32, link: &SimLink, slot: u32) {
    let (mut best, mut best_pos) = (0.0f32, None);
    let mut now_seen = HashMap::with_capacity(threats.0.len());
    for t in &threats.0 {
        let dist = (t.pos - head).length().max(t.radius + 0.02);
        let theta = 2.0 * (t.radius / dist).atan();
        now_seen.insert(t.key, theta);
        if let Some(prev) = mem.prev.get(&t.key) {
            let rate = (theta - prev) / dt;
            if rate > best {
                best = rate;
                best_pos = Some(t.pos);
            }
        }
    }
    mem.prev = now_seen;
    mem.rate += (best - mem.rate) * 0.5;
    let th = crate::reactions::thresholds();
    let strength = ((best - th.loom_min) / th.loom_full).clamp(0.0, 1.0);
    if strength > 0.0
        && let Some(p) = best_pos
    {
        // `poke("loom", None, strength, recruit=0.6 * strength)`: a harder looming recruits more of the detectors
        link.0.send(Command::PokeFlyRecruit { slot, region: "loom".into(), side: None, strength, recruit: 0.6 * strength });
        f.threat_x = p;
    }
}

/// Smells that reach the fly's head poke its olfactory neurons.
fn smell(scents: Res<Scents>, link: Res<SimLink>, flies: Query<(&Fly, &crate::fly::Slot)>, parts: Query<&Transform, With<FlyPart>>) {
    for (f, slot) in &flies {
        if f.dead() {
            continue;
        }
        let Ok(head) = parts.get(f.parts[Part::Head.index()]).map(|t| t.translation) else { continue };
        for s in &scents.0 {
            let d = if s.flat { Vec2::new(s.pos.x - head.x, s.pos.z - head.z).length() } else { (s.pos - head).length() };
            if d < s.range {
                link.0.send(Command::PokeFly { slot: slot.0, region: format!("scent:{}", s.scent), side: None, strength: s.strength });
            }
        }
    }
}

pub struct SensesPlugin;

impl Plugin for SensesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Threats>()
            .init_resource::<Scents>()
            .add_systems(FixedFirst, clear)
            .add_systems(Update, attach_memory)
            .add_systems(FixedLast, (vision, smell));
    }
}
