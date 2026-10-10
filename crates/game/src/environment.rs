//! What each arena does to a fly, every tick: fan wind, flypaper glue, the pool's water, the lamp's light and heat, the thermo gradient,
//! the kitchen (sink, burner, vinegar trap, the cook, the fruit bowl), the outdoors (wind, rain, sun, day and night, the orchard's fruit,
//! flies that fly out of sight). Python: `Game3D._environment_one`, `_kitchen_tick`, `_cook_tick`, `_weather`, `_orchard_tick`,
//! `_check_lost`, `Game._thermo_tick`.
//!
//! CONNECTOME vs GAME RULE, per effect, is written where the effect is: what is POKED (wind -> JO-C/E, light -> photoreceptors, heat and
//! cold -> their sensors, humidity -> HRN, touch by body part, smell -> the ORN glomeruli, taste and reward) is the connectome's input;
//! where things are, how far they reach, and what they damage are game rules.

use bevy::prelude::*;

use crate::app::{LatestSnapshot, SimLink};
use crate::arena::WorldColliders;
use crate::config::{Arena, Settings};
use crate::feedback::{Notes, Popup, Sfx, Tag};
use crate::fly::{Fly, FlyDamage, FlyHit, FlyPart, MAX_HEALTH, Part};
use crate::fx::{Fx, Kind, Particle};
use crate::kitchen::{self as kit, CookEvent, KitchenState};
use crate::player::Player;
use crate::rng::Rng;
use crate::senses::Threats;
use crate::sim_thread::Command;
use crate::weather;
use crate::weather_play::WeatherRes;
use crate::world::{self, FAN3, LAMP3, Orchard, PAPER3, RX, S, STAND3, THERMO_HALF3, Tree, WATER3};

/// The orchard's trees' fruit, or the kitchen's bowl (the same fruit code with one "tree").
#[derive(Resource, Default)]
pub struct Fruit(pub Option<Orchard>);

#[derive(Resource, Default)]
pub struct Kitchen(pub Option<KitchenState>);

/// Per-fly foraging state (flying to a fruit and feeding).
#[derive(Component, Default)]
pub struct Forage {
    /// The brain this fly's senses poke (`fly::Slot`).
    pub slot: u32,
    pub fruit: Option<usize>,
    pub landed: bool,
    pub since: f32,
    pub feed_start: f32,
    pub ready: f32,
    pub lost: bool,
    pub rain_wet: f32,
    pub photo_ready: f32,
    pub trap_ready: f32,
    pub trap_trip: Option<Vec3>,
}

/// Escape room: the goal (sugar on the right) and the clock.
#[derive(Resource, Default)]
pub struct EscapeRoom {
    pub start: Option<f32>,
    pub cleared: Option<f32>,
}

fn attach(mut commands: Commands, new: Query<(Entity, &crate::fly::Slot), (With<Fly>, Without<Forage>)>) {
    for (e, slot) in &new {
        commands.entity(e).insert(Forage { slot: slot.0, ..default() });
    }
}

/// Rebuild the arena state when the arena changes.
fn on_arena_change(
    arena: Res<Arena>,
    mut fruit: ResMut<Fruit>,
    mut kitchen: ResMut<Kitchen>,
    mut room: ResMut<EscapeRoom>,
    time: Res<Time<Virtual>>,
) {
    if !arena.is_changed() {
        return;
    }
    let name = arena.name();
    *fruit = Fruit(None);
    *kitchen = Kitchen(None);
    *room = EscapeRoom::default();
    match name {
        "orchard" => {
            let trees = world::scenery("orchard", 7).trees;
            fruit.0 = Some(Orchard::new(&trees, world::DEFAULT_FEEDS, world::DEFAULT_REGROW_S, world::DEFAULT_CAP, 5));
        }
        "kitchen" => {
            let tree = Tree { pos: kit::BOWL_POS, height: kit::BOWL_H + 0.3, crown: 0.3 };
            let mut o =
                Orchard::new(&[tree], world::DEFAULT_FEEDS, world::DEFAULT_REGROW_S, world::SLOTS_PER_TREE.min(world::DEFAULT_CAP), 5);
            let mut rng = Rng::new(16);
            let n = o.fruit.len();
            for (k, f) in o.fruit.iter_mut().enumerate() {
                let a = std::f32::consts::TAU * k as f32 / n as f32 + rng.uniform(-0.25, 0.25);
                let r = kit::BOWL_R * rng.uniform(0.25, 0.6);
                f.pos = kit::BOWL_POS + Vec3::new(r * a.cos(), 0.2 + 0.05 * (k % 2) as f32, r * a.sin());
            }
            fruit.0 = Some(o);
            kitchen.0 = Some(KitchenState::new(11));
        }
        "escaperoom" => room.start = Some(time.elapsed_secs()),
        _ => {}
    }
}

/// The groups' firing the snapshot carries, by name.
fn group_hz(snap: &LatestSnapshot, name: &str) -> f32 {
    snap.0
        .as_deref()
        .and_then(|s| s.info.as_deref().and_then(|i| i.group_names.iter().position(|n| n == name)).and_then(|i| s.hz.get(i).copied()))
        .unwrap_or(0.0)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn arena_effects(
    virt: Res<Time<Virtual>>,
    fixed: Res<Time<Fixed>>,
    arena: Res<Arena>,
    settings: Res<Settings>,
    link: Res<SimLink>,
    mut flies: Query<(Entity, &mut Fly, &mut Forage)>,
    parts: Query<(&FlyPart, &Transform)>,
    mut vels: Query<(&FlyPart, &mut avian3d::prelude::LinearVelocity)>,
    mut dmg: MessageWriter<FlyDamage>,
    mut popups: MessageWriter<Popup>,
    mut sfx: MessageWriter<Sfx>,
    mut notes: ResMut<Notes>,
    mut fx: ResMut<Fx>,
    mut counter: Local<u64>,
    mut room: ResMut<EscapeRoom>,
) {
    let now = virt.elapsed_secs();
    let dt = fixed.delta_secs();
    *counter += 1;
    let tick = *counter;
    let name = arena.name();
    let mut rng = Rng::new(tick ^ 0x5eed);
    for (e, mut f, mut fo) in &mut flies {
        let ppos: [Vec3; 3] = std::array::from_fn(|i| parts.get(f.parts[i]).map(|(_, t)| t.translation).unwrap_or(Vec3::ZERO));
        let pos = |p: Part| ppos[p.index()];
        let (head, thx) = (pos(Part::Head), pos(Part::Thorax));
        f.wind = Vec3::ZERO;
        let dead = f.dead();
        if !matches!(name, "flypaper" | "escaperoom") {
            f.stuck.clear();
        }
        let want_fan = matches!(name, "fan" | "escaperoom");
        let want_paper = matches!(name, "flypaper" | "escaperoom");
        let want_lamp = matches!(name, "lamp" | "escaperoom");
        if want_fan {
            let gust = 0.75 + 0.25 * (now * 1.3).sin() + 0.15 * (now * 4.1).sin();
            let dx = thx.x - FAN3.x;
            let lateral = (-((thx.z - FAN3.z).powi(2)) / (2.0 * 1.9 * 1.9)).exp();
            let w = 0.55 * gust * (1.15 - dx / 7.0).clamp(0.25, 1.0) * lateral;
            f.wind = Vec3::new(w * S * 60.0, 0.0, 0.0);
            if !dead && tick.is_multiple_of(3) {
                link.0.send(Command::PokeFly { slot: fo.slot, region: "wind".into(), side: None, strength: (w * 1.4).min(1.0) });
            }
            if rng.chance(0.8) {
                let at = Vec3::new(FAN3.x + 0.5, rng.uniform(0.4, 1.6), FAN3.z + rng.uniform(-1.2, 1.2));
                fx.add(Particle {
                    pos: at,
                    vel: Vec3::new(rng.uniform(0.09, 0.15) * 60.0, 0.0, 0.0),
                    born: now,
                    life: 1.0,
                    size: 0.025,
                    kind: Kind::Streak,
                });
            }
        }
        if want_paper {
            let (x0, x1, z0, z1) = PAPER3;
            if f.grabbed.is_none() && f.stuck.len() < 3 && thx.x > x0 && thx.x < x1 && thx.z > z0 && thx.z < z1 && thx.y < 0.3 {
                f.stuck.push(Vec3::new(thx.x, 0.0, thx.z));
            }
            if !f.stuck.is_empty() {
                let kicking = now < f.react.flail_until;
                let pull = f.grabbed.is_some() && (f.hold - thx).length() > 0.55;
                let p = 0.0012 + if kicking { 0.012 } else { 0.0 } + if pull { 0.06 } else { 0.0 };
                f.stuck.retain(|_| !rng.chance(p));
                if !dead {
                    if tick.is_multiple_of(6) {
                        link.0.send(Command::PokeFly { slot: fo.slot, region: "legs".into(), side: Some('L'), strength: 0.4 });
                        link.0.send(Command::PokeFly { slot: fo.slot, region: "legs".into(), side: Some('R'), strength: 0.4 });
                        link.0.send(Command::PokeFly { slot: fo.slot, region: "body".into(), side: None, strength: 0.2 });
                    }
                    if f.stuck.len() >= 3 {
                        dmg.write(FlyDamage { fly: e, amount: 0.012, source: "the flypaper" });
                    }
                }
            }
        }
        if want_lamp {
            let dist = (head - LAMP3).length();
            if !dead {
                let light = (1.2 - dist / 3.0).clamp(0.15, 1.0);
                if tick.is_multiple_of(2) {
                    link.0.send(Command::PokeFly { slot: fo.slot, region: "light".into(), side: None, strength: light });
                }
                if dist < 0.36 {
                    link.0.send(Command::PokeFly { slot: fo.slot, region: "heat".into(), side: None, strength: 0.8 });
                    dmg.write(FlyDamage { fly: e, amount: 0.08, source: "the hot lamp" });
                    let away = (head - LAMP3) / dist.max(1e-6);
                    for (p, mut v) in &mut vels {
                        if p.fly == e {
                            v.0 += away * 1.5 * S * 60.0 * 0.5;
                        }
                    }
                }
            }
            if now < f.react.escape_until && (f.fly_target - LAMP3).length() > 0.8 {
                f.fly_target = LAMP3 + Vec3::new(rng.uniform(-0.6, 0.6), -rng.uniform(0.3, 0.8), rng.uniform(-0.6, 0.6));
            }
            // phototaxis: an idle fly turns to the lamp now and then and flies to it (60%) or walks toward it
            let idle = f.grabbed.is_none() && now >= f.react.escape_until && now >= f.react.stun_until && now >= f.react.walk_until;
            if !dead && idle && now >= fo.photo_ready {
                fo.photo_ready = now + rng.uniform(3.0, 6.0);
                let to = LAMP3 - thx;
                f.yaw_target = to.z.atan2(to.x);
                if f.can_fly() && rng.chance(0.6) {
                    f.hover = thx;
                    f.react.escape_until = now + rng.uniform(3.0, 5.0);
                    f.fly_target = LAMP3 - Vec3::Y * 0.5;
                    notes.push(now, "TO LIGHT flies to the lamp", Tag::GameRule);
                } else {
                    f.react.walk_until = now + 1.5;
                    f.run = false;
                }
            }
        }
        match name {
            "pool" => {
                let sub: Vec<bool> = Part::ALL.iter().map(|p| pos(*p).y < WATER3).collect();
                if sub.iter().any(|b| *b) {
                    if f.wet <= 0.0 {
                        let down = vels.iter().find(|(p, _)| p.fly == e && p.part == Part::Thorax).map_or(0.0, |(_, v)| -v.0.y);
                        if down > 4.0 * S * 60.0 {
                            sfx.write(Sfx::new("splash"));
                            fx.puff(Vec3::new(thx.x, WATER3, thx.z), 10, 3.0, now);
                        }
                    }
                    f.wet = 3.0;
                    if !dead {
                        if tick.is_multiple_of(4) {
                            link.0.send(Command::PokeFly { slot: fo.slot, region: "humid".into(), side: None, strength: 0.9 });
                            link.0.send(Command::PokeFly { slot: fo.slot, region: "body".into(), side: None, strength: 0.25 });
                        }
                        if head.y < WATER3 - 6.0 * S {
                            dmg.write(FlyDamage { fly: e, amount: 0.05, source: "drowning" });
                        }
                    }
                }
            }
            "thermo" => {
                let t = thx.x / THERMO_HALF3;
                let (cold, hot) = (t.clamp(-1.0, 1.0).min(0.0).abs(), t.clamp(-1.0, 1.0).max(0.0));
                if tick.is_multiple_of(3) {
                    // what TrpA1 / shibire-ts expressed in this fly sense (`slot.arena_temp_c`)
                    link.0.send(Command::ThermoArena { slot: fo.slot, temp_c: fly_sim::thermogenetics::arena_temperature(f64::from(t)) });
                }
                if !dead && tick.is_multiple_of(3) {
                    if cold > 0.05 {
                        link.0.send(Command::PokeFly { slot: fo.slot, region: "cold".into(), side: None, strength: cold });
                    }
                    if hot > 0.05 {
                        link.0.send(Command::PokeFly { slot: fo.slot, region: "heat".into(), side: None, strength: hot });
                    }
                }
                if thx.y < STAND3 + 20.0 * S && !dead {
                    if hot > 0.85 {
                        dmg.write(FlyDamage { fly: e, amount: 0.02, source: "extreme heat" });
                    } else if cold > 0.85 {
                        dmg.write(FlyDamage { fly: e, amount: 0.02, source: "extreme cold" });
                    }
                }
            }
            "escaperoom" => {
                let sugar = Vec3::new(RX - 0.7, 0.04, 0.0);
                if (head - sugar).length() < 0.4 && !dead {
                    if tick.is_multiple_of(3) {
                        link.0.send(Command::PokeFly { slot: fo.slot, region: "taste".into(), side: None, strength: 1.0 });
                    }
                    if room.cleared.is_none() {
                        let t = now - room.start.unwrap_or(now);
                        room.cleared = Some(t.max(0.1));
                        sfx.write(Sfx::new("yum"));
                        notes.push(now, format!("ESCAPEROOM CLEAR {t:.2}s"), Tag::GameRule);
                        popups.write(Popup::new(head + Vec3::Y * 0.5, "ESCAPED!", (160, 255, 190)));
                    }
                }
            }
            _ => {}
        }
        // the water-line decay of `wet` (it only stays set while some part is in the water)
        let in_water = matches!(name, "pool" | "kitchen") && Part::ALL.iter().any(|p| crate::fly::submerged(name, pos(*p)));
        if !in_water {
            f.wet = (f.wet - dt).max(0.0);
        }
        let _ = (&settings, MAX_HEALTH);
    }
}

/// Outdoors: wind, rain, lightning and sunlight onto the real wind, touch, humidity and light neurons; flies that fly out of sight.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn outdoor_effects(
    virt: Res<Time<Virtual>>,
    fixed: Res<Time<Fixed>>,
    arena: Res<Arena>,
    settings: Res<Settings>,
    link: Res<SimLink>,
    mut wx: ResMut<WeatherRes>,
    mut flies: Query<(&mut Fly, &mut Forage)>,
    parts: Query<(&FlyPart, &Transform)>,
    player: Single<&Transform, With<Player>>,
    mut notes: ResMut<Notes>,
    mut fx: ResMut<Fx>,
    mut counter: Local<u64>,
) {
    if !world::is_outdoor(arena.name()) {
        return;
    }
    let (now, dt) = (virt.elapsed_secs(), fixed.delta_secs());
    *counter += 1;
    let tick = *counter;
    let field = arena.name() == "field";
    let (mut wd, mut ws) = (settings.wind_dir_deg, if field { settings.wind_speed.max(3.0) } else { 0.0 });
    if wx.0.storm || wx.0.rain > 0.0 || settings.gust_hz > 0.0 {
        (ws, wd) = wx.0.wind(ws, wd);
    }
    let humid = if wx.0.rain > 0.0 { wx.0.humid(dt) } else { 0.0 };
    let (az, el) = world::sun_now(now, settings.day_night_s);
    let w = world::spec(arena.name());
    for (mut f, mut fo) in &mut flies {
        let thx = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t)| t.translation).unwrap_or(Vec3::ZERO);
        let dead = f.dead();
        if field || ws > 0.0 {
            f.wind = world::wind_vector(wd, ws) * 0.036;
            if !dead && tick.is_multiple_of(3) && ws > 0.0 {
                let (l, r) = world::wind_drive(f.yaw, wd, ws);
                if l > 0.02 {
                    link.0.send(Command::PokeFly { slot: fo.slot, region: "wind".into(), side: Some('L'), strength: l });
                }
                if r > 0.02 {
                    link.0.send(Command::PokeFly { slot: fo.slot, region: "wind".into(), side: Some('R'), strength: r });
                }
            }
            if field && fx.rng.chance(0.25 * (ws / 4.0).min(1.0)) {
                let me = player.translation;
                let src = me + Vec3::new(fx.rng.uniform(-6.0, 6.0), fx.rng.uniform(0.2, 2.5), fx.rng.uniform(-6.0, 6.0));
                let v = world::wind_vector(wd, ws);
                fx.add(Particle { pos: src, vel: v, born: now, life: 1.2, size: 0.02, kind: Kind::Streak });
            }
        }
        if wx.0.rain > 0.0 && !dead {
            let mut wing_hits = 0;
            for (region, side, s) in wx.0.hits(dt, true) {
                link.0.send(Command::PokeFly { slot: fo.slot, region: region.into(), side, strength: s });
                wing_hits += u32::from(region == "wing");
            }
            if humid > 0.0 {
                link.0.send(Command::PokeFly { slot: fo.slot, region: "humid".into(), side: None, strength: humid });
            }
            fo.rain_wet = weather::soak(fo.rain_wet, dt, wx.0.rain, wing_hits);
            if fo.rain_wet >= 1.0 {
                f.wet = f.wet.max(weather::WET_FOR_S);
            }
        } else if fo.rain_wet > 0.0 {
            fo.rain_wet = weather::soak(fo.rain_wet, dt, 0.0, 0);
        }
        let light = wx.0.lightning();
        if light > 0.0 && !dead {
            link.0.send(Command::PokeFly { slot: fo.slot, region: "light".into(), side: Some('L'), strength: light });
            link.0.send(Command::PokeFly { slot: fo.slot, region: "light".into(), side: Some('R'), strength: light });
        }
        if !dead && tick.is_multiple_of(2) {
            let (l, r) = world::sun_light(f.yaw, az, el);
            if l > 0.02 {
                link.0.send(Command::PokeFly { slot: fo.slot, region: "light".into(), side: Some('L'), strength: l });
            }
            if r > 0.02 {
                link.0.send(Command::PokeFly { slot: fo.slot, region: "light".into(), side: Some('R'), strength: r });
            }
            if settings.day_night_s > 0.0 {
                let clock = world::circadian_clock_drive(el);
                if clock > 0.02 {
                    link.0.send(Command::PokeFly { slot: fo.slot, region: "clock".into(), side: None, strength: clock });
                }
            }
        }
        // lost: out of sight of you
        if let Some(r) = w.lost_radius {
            let me = player.translation;
            let d = Vec2::new(thx.x - me.x, thx.z - me.z).length();
            let far = d > r || thx.y > 15.0;
            if far && !fo.lost {
                fo.lost = true;
                notes.push(now, format!("LOST     flew out of sight, {d:.0} m away (J calls it back)"), Tag::GameRule);
            } else if !far {
                fo.lost = false;
            }
        }
    }
}

/// The orchard's and the bowl's fruit: flies fly to the nearest ripe fruit, land, and feed (taste + PAM reward; fermented fruit also
/// smells of the alcohol glomeruli and makes it drunk).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn foraging(
    virt: Res<Time<Virtual>>,
    arena: Res<Arena>,
    mut fruit: ResMut<Fruit>,
    link: Res<SimLink>,
    mut flies: Query<(Entity, &mut Fly, &mut Forage)>,
    parts: Query<(&FlyPart, &Transform)>,
    mut popups: MessageWriter<Popup>,
    mut notes: ResMut<Notes>,
    mut counter: Local<u64>,
) {
    let Some(o) = fruit.0.as_mut() else { return };
    let now = virt.elapsed_secs();
    *counter += 1;
    o.step(now);
    let _ = &arena;
    for (e, mut f, mut fo) in &mut flies {
        let thx = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t)| t.translation).unwrap_or(Vec3::ZERO);
        let head = parts.get(f.parts[Part::Head.index()]).map(|(_, t)| t.translation).unwrap_or(Vec3::ZERO);
        let busy = f.dead() || f.wrapped || f.frozen_at.is_some() || f.grabbed.is_some() || fo.lost;
        let leave = |fo: &mut Forage, f: &mut Fly, o: &mut Orchard, now: f32, rng: &mut Rng| {
            if let Some(i) = fo.fruit.take()
                && o.fruit[i].eater == Some(index_u32(e))
            {
                o.fruit[i].eater = None;
            }
            fo.landed = false;
            f.perch = None;
            fo.ready = now + rng.uniform(8.0, 16.0);
        };
        let mut rng = Rng::new(*counter ^ e.to_bits());
        if let Some(i) = fo.fruit {
            let perch = o.fruit[i].pos + Vec3::Y * 0.07;
            let diverted = (f.fly_target - perch).length() > 1e-3;
            if busy
                || diverted
                || !o.fruit[i].ripe()
                || now >= f.react.escape_until
                || (fo.landed && o.fruit[i].eater != Some(index_u32(e)))
            {
                leave(&mut fo, &mut f, o, now, &mut rng);
                continue;
            }
            if !fo.landed {
                if (f.hover - perch).length() < 0.1 {
                    if o.fruit[i].eater.is_some() {
                        leave(&mut fo, &mut f, o, now, &mut rng);
                        continue;
                    }
                    o.fruit[i].eater = Some(index_u32(e));
                    fo.landed = true;
                    fo.feed_start = now;
                    f.react.escape_until = now + world::FEED_BOUT_S + 1.0;
                    notes.push(
                        now,
                        if o.fruit[i].fermented {
                            "EATING   fermented fruit: sweet + PAM reward [GAME RULE: inebriation]"
                        } else {
                            "EATING   fruit: taste + PAM reward"
                        },
                        Tag::GameRule,
                    );
                    popups.write(Popup::new(head + Vec3::Y * 0.3, ["YUM!", "NOM NOM"][rng.index(2)], (255, 160, 190)));
                } else if now - fo.since > 14.0 {
                    leave(&mut fo, &mut f, o, now, &mut rng);
                }
                continue;
            }
            f.react.eating_until = now + 0.4;
            let ferm = o.fruit[i].fermented;
            link.0.send(Command::PokeFly { slot: fo.slot, region: "taste".into(), side: None, strength: 0.5 });
            link.0.send(Command::PokeFly { slot: fo.slot, region: "reward".into(), side: None, strength: if ferm { 0.6 } else { 0.4 } });
            if ferm {
                f.inebriation = (f.inebriation + 0.008).min(1.0);
                if (*counter).is_multiple_of(10) {
                    link.0.send(Command::PokeFly { slot: fo.slot, region: "scent:alcohol".into(), side: None, strength: 0.35 });
                }
            }
            f.health = (f.health + 0.15).min(MAX_HEALTH);
            if now - fo.feed_start >= world::FEED_BOUT_S {
                if o.feed(i, now) {
                    popups.write(Popup::new(o.fruit[i].pos + Vec3::Y * 0.3, "ALL GONE", (230, 200, 150)));
                }
                leave(&mut fo, &mut f, o, now, &mut rng);
                f.hover = thx;
                f.fly_target = thx + Vec3::new(rng.uniform(-2.0, 2.0), 1.2, rng.uniform(-2.0, 2.0));
                f.react.escape_until = now + 1.5;
            }
            continue;
        }
        if busy || now < fo.ready.max(1.0) || now < f.react.escape_until {
            if fo.ready == 0.0 {
                fo.ready = now + rng.uniform(2.0, 6.0);
            }
            continue;
        }
        let Some(target) = o.nearest_ripe(thx, true) else {
            fo.ready = now + 3.0;
            continue;
        };
        if !f.can_fly() || (o.fruit[target].pos - thx).length() > 14.0 {
            fo.ready = now + 3.0;
            continue;
        }
        fo.fruit = Some(target);
        fo.since = now;
        fo.landed = false;
        f.hover = thx;
        f.react.escape_until = now + 16.0;
        f.wander = false;
        f.fly_target = o.fruit[target].pos + Vec3::Y * 0.07;
        f.perch = Some(target);
        notes.push(now, "TO FRUIT flies to a fruit", Tag::GameRule);
    }
}

fn index_u32(e: Entity) -> u32 {
    (e.to_bits() & 0xFFFF_FFFF) as u32
}

/// The kitchen's sink, burner, vinegar trap and cook.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn kitchen_tick(
    virt: Res<Time<Virtual>>,
    fixed: Res<Time<Fixed>>,
    snap: Res<LatestSnapshot>,
    settings: Res<Settings>,
    link: Res<SimLink>,
    mut kitchen: ResMut<Kitchen>,
    mut flies: Query<(Entity, &mut Fly, &mut Forage)>,
    parts: Query<(&FlyPart, &Transform)>,
    mut vels: Query<(&FlyPart, &mut avian3d::prelude::LinearVelocity)>,
    mut fb: crate::tools::Fb,
    mut ff: crate::tools::FlyFx,
    mut fx: ResMut<Fx>,
    mut threats: ResMut<Threats>,
    mut counter: Local<u64>,
    boxes: Res<WorldColliders>,
) {
    let Some(k) = kitchen.0.as_mut() else { return };
    let (now, dt) = (virt.elapsed_secs(), fixed.delta_secs());
    *counter += 1;
    let tick = *counter;
    let _ = &boxes;
    let alcohol_hz = group_hz(&snap, "scent:alcohol");
    let mut live_thx: Vec<Vec3> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    for (e, mut f, mut fo) in &mut flies {
        let key = index_u32(e);
        ids.push(key);
        if f.dead() {
            k.release(key);
            continue;
        }
        let pos = |p: Part| parts.get(f.parts[p.index()]).map(|(_, t)| t.translation).unwrap_or(Vec3::ZERO);
        let (head, thx) = (pos(Part::Head), pos(Part::Thorax));
        if !f.gone() {
            live_thx.push(thx);
        }
        // the sink: the pool's water rules in the basin
        if Part::ALL.iter().any(|p| crate::fly::submerged("kitchen", pos(*p))) {
            if f.wet <= 0.0 {
                fb.sfx.write(Sfx::new("splash"));
                fx.puff(Vec3::new(thx.x, kit::SINK_WATER, thx.z), 10, 3.0, now);
            }
            f.wet = 3.0;
            if tick.is_multiple_of(4) {
                link.0.send(Command::PokeFly { slot: fo.slot, region: "humid".into(), side: None, strength: 0.9 });
                link.0.send(Command::PokeFly { slot: fo.slot, region: "body".into(), side: None, strength: 0.25 });
            }
            if head.y < kit::SINK_WATER - 6.0 * S {
                ff.dmg.write(FlyDamage { fly: e, amount: 0.05, source: "drowning in the sink" });
            }
        }
        // the burner: the lamp's heat rules
        let (heat, touching) = kit::burner_heat(head);
        if heat > 0.05 && tick.is_multiple_of(2) {
            link.0.send(Command::PokeFly { slot: fo.slot, region: "heat".into(), side: None, strength: 0.8 * heat });
        }
        if touching {
            link.0.send(Command::PokeFly { slot: fo.slot, region: "legs".into(), side: Some('L'), strength: 0.5 });
            link.0.send(Command::PokeFly { slot: fo.slot, region: "legs".into(), side: Some('R'), strength: 0.5 });
            ff.dmg.write(FlyDamage { fly: e, amount: 0.1, source: "the stove burner" });
            let away = Vec3::new(thx.x - kit::BURNER_POS.x, 0.0, thx.z - kit::BURNER_POS.z).normalize_or_zero();
            for (p, mut v) in &mut vels {
                if p.fly == e {
                    v.0 += (away * 1.5 + Vec3::Y) * S * 60.0 * 0.5;
                }
            }
        }
        // the vinegar's smell on DM1/DM2/DP1m, and whether those neurons say "go there"
        let s = kit::scent_strength(head);
        if s > 0.0 && tick.is_multiple_of(2) {
            link.0.send(Command::PokeFly { slot: fo.slot, region: "scent:alcohol".into(), side: None, strength: s });
        }
        if tick.is_multiple_of(6) {
            k.tracker(key).update(6.0 / 60.0, alcohol_hz, s > 0.0);
        }
        let busy = f.wrapped
            || f.frozen_at.is_some()
            || f.grabbed.is_some()
            || f.wet > 0.0
            || fo.fruit.is_some()
            || f.stuck.len() >= 2
            || k.trapped.contains_key(&key);
        if let Some(trip) = fo.trap_trip
            && (k.trapped.contains_key(&key) || now >= f.react.escape_until || (f.fly_target - trip).length() > 1e-3)
        {
            fo.trap_trip = None;
            if f.perch == Some(usize::MAX) {
                f.perch = None;
            }
        }
        if k.tracker(key).smelling() && !busy && now >= f.react.escape_until && now >= fo.trap_ready {
            fo.trap_ready = now + 6.0;
            let trip = kit::mouth() + Vec3::new(0.0, 0.15, 0.0);
            fo.trap_trip = Some(trip);
            f.hover = thx;
            f.react.escape_until = now + 8.0;
            f.wander = false;
            f.fly_target = trip;
            f.perch = Some(usize::MAX); // hold over the mouth on arrival, as at a fruit
            fb.notes.push(now, "TO VINEGAR its DM1/DM2/DP1m neurons fire for the smell: it flies to the trap", Tag::GameRule);
        }
        // the trap: hover over the mouth and fall in; once in, stuck and slowly drowning
        if let Some(t) = k.trapped.get_mut(&key) {
            *t += dt;
            let held_for = *t;
            f.grabbed = Some(Part::Thorax);
            f.hold = k.inside_point(key, held_for);
            f.wet = 3.0;
            if tick.is_multiple_of(6) {
                link.0.send(Command::PokeFly { slot: fo.slot, region: "legs".into(), side: Some('L'), strength: 0.4 });
                link.0.send(Command::PokeFly { slot: fo.slot, region: "legs".into(), side: Some('R'), strength: 0.4 });
                link.0.send(Command::PokeFly { slot: fo.slot, region: "humid".into(), side: None, strength: 0.5 });
            }
            ff.dmg.write(FlyDamage { fly: e, amount: kit::TRAP_DROWN, source: "the vinegar trap" });
            if settings.immortal && held_for > kit::TRAP_IMMORTAL_ESCAPE_S {
                k.release(key);
                f.grabbed = None;
                f.last_hit = kit::TRAP_POS;
                fb.popups.write(Popup::new(head + Vec3::Y * 0.5, "BROKE FREE!", (255, 225, 120)));
            }
        } else if kit::over_mouth(thx) && f.grabbed.is_none() {
            let h = k.hover.entry(key).or_insert(0.0);
            *h += dt;
            if *h >= kit::HOVER_TO_FALL_S {
                k.trapped.insert(key, 0.0);
                f.grabbed = Some(Part::Thorax);
                f.react.escape_until = 0.0;
                fb.sfx.write(Sfx::vol("splash", 0.6));
                fb.popups.write(Popup::new(head + Vec3::Y * 0.6, "STUCK IN THE TRAP!", (230, 200, 120)));
                fb.notes.push(now, "TRAPPED  fell into the vinegar trap", Tag::GameRule);
            }
        } else {
            k.hover.remove(&key);
        }
    }
    for map in [&mut k.trapped, &mut k.hover] {
        map.retain(|key, _| ids.contains(key));
    }
    k.trackers.retain(|key, _| ids.contains(key));
    // the cook
    for ev in k.cook.step(dt, &live_thx) {
        match ev {
            CookEvent::Windup(_) => {
                fb.popups.write(Popup::new(Vec3::new(k.cook.x, 2.2, kit::COOK_Z + 0.4), "THE COOK RAISES A SWATTER", (240, 220, 200)));
            }
            CookEvent::Impact(at) => {
                fb.shake.until = now + 0.15;
                fb.shake.strength = 0.02;
                fb.sfx.write(Sfx::new("whack"));
                fx.puff(Vec3::new(at.x, 0.02, at.z), 12, 4.0, now);
                for (e, mut f, _) in &mut flies {
                    let thx = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t)| t.translation).unwrap_or(Vec3::ZERO);
                    if f.dead() || f.gone() || !k.cook.hit(thx) {
                        continue;
                    }
                    f.last_hit = at + Vec3::Y;
                    for (p, mut v) in &mut vels {
                        if p.fly == e && Vec2::new(thx.x - at.x, thx.z - at.z).length() < kit::COOK_REACH + 0.3 {
                            v.0 += Vec3::new(0.0, -30.0 * S * 60.0, 0.0);
                        }
                    }
                    ff.hits.write(FlyHit { fly: e, part: Part::Thorax, strength: 1.0 });
                    f.stun(now, 1.8);
                    ff.dmg.write(FlyDamage { fly: e, amount: kit::COOK_DAMAGE, source: "the cook's swatter" });
                    fb.popups.write(Popup::new(thx + Vec3::Y * 0.45, "SWAT!", (255, 230, 120)));
                }
            }
            CookEvent::Swat(_) => {}
        }
    }
    for (key, pos, r) in k.cook.threats() {
        threats.push(key, pos, r);
    }
}

pub struct EnvironmentPlugin;

impl Plugin for EnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Fruit>()
            .init_resource::<Kitchen>()
            .init_resource::<EscapeRoom>()
            .add_systems(Update, (attach, on_arena_change).chain())
            .add_systems(FixedUpdate, (arena_effects, outdoor_effects, foraging, kitchen_tick).chain().before(crate::fly::control));
    }
}
