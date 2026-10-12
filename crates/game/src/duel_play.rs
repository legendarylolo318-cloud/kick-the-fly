//! The 1v1 duel in play (X): the fly gets a blaster and can kill you. Python: `Game3D.{toggle_duel, respawn_player, _duel_senses,
//! _duel_motor, _fly_shoot, _pellets3d, _player_hit}` and the duel branches of `_scents`, `_memory_behavior` and the damage loop.
//! The rules and numbers are in [`crate::duel_rules`] (tested); this file feeds them from the world and the brain's snapshot and
//! applies what they say.
//!
//! The duel is about the *focused* fly only (`self.flies[self.focus]`): its brain gets your position through LC10 / small-object
//! pokes, its DNa01/DNa02 right-minus-left rate turns its body, its DNp35/DNpe052 level fires the blaster.

use avian3d::prelude::LinearVelocity;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;

use crate::actions::Actions;
use crate::app::{LatestSnapshot, SimLink};
use crate::arena::WorldColliders;
use crate::config::{Arena, Settings};
use crate::duel_rules::{self as rules, Pellet, Stats};
use crate::feedback::{Notes, Popup, SavedMsg, Sfx, Shake, Tag};
use crate::flies_play::Flies;
use crate::fly::{Fly, FlyDamage, FlyPart, Part, ReactionFired, Slot};
use crate::fx::{Fx, Kind, Particle};
use crate::player::{EYE, MouseLook, Player};
use crate::rng::Rng;
use crate::sim_thread::Command;
use crate::world::{self, S, STAND3};

/// The duel's state (`Game3D.duel`, `player_hp`, `player_dead_at`, `duel_stats`, `valence`, `steer`, `trigger`, `hurt_flash`, ...).
#[derive(Resource)]
pub struct Duel {
    pub on: bool,
    pub hp: f32,
    pub dead_at: Option<f32>,
    pub stats: Stats,
    /// How its mushroom body sees you, smoothed: `like - fear`.
    pub valence: f32,
    pub steer: f32,
    pub trigger: f32,
    pub hurt_flash: f32,
    /// What it has learned about your smell (from the snapshot's memory view).
    pub fear: f32,
    pub like: f32,
    /// You are within 3.5 m of the focused fly: your smell reaches it (`slot.player_scent`).
    pub scent: bool,
    pub reward_until: f32,
    pub punish_until: f32,
    fire_ready: f32,
    pub pellets: Vec<Pellet>,
    tick: u64,
    rng: Rng,
}

impl Default for Duel {
    fn default() -> Self {
        Self {
            on: false,
            hp: rules::PLAYER_HP,
            dead_at: None,
            stats: Stats::default(),
            valence: 0.0,
            steer: 0.0,
            trigger: 0.0,
            hurt_flash: 0.0,
            fear: 0.0,
            like: 0.0,
            scent: false,
            reward_until: 0.0,
            punish_until: 0.0,
            fire_ready: 0.0,
            pellets: Vec::new(),
            tick: 0,
            rng: Rng::new(0xd0e1),
        }
    }
}

impl Duel {
    /// You are dead: the fly won this round and R respawns you.
    pub fn player_dead(&self) -> bool {
        self.dead_at.is_some()
    }

    /// `toggle_duel`: on or off, you back at full health, no pellets in the air.
    pub fn toggle(&mut self) {
        self.on = !self.on;
        self.hp = rules::PLAYER_HP;
        self.dead_at = None;
        self.pellets.clear();
    }

    /// `Game3D._valence`: what its mushroom body holds about your smell, `(fear, liking)`.
    fn learn(&mut self, snap: &LatestSnapshot) {
        let m = snap.0.as_deref().and_then(|s| s.memory.as_deref());
        let (fear, like) = m.and_then(|m| m.scores.iter().find(|r| r.0 == "player").map(|r| (r.1, r.2))).unwrap_or((0.0, 0.0));
        self.fear = fear;
        self.like = like;
    }
}

/// Your eye (`Player.eye`): over your feet at the eye height, bobbing a little with the steps.
pub fn eye_of(p: &Player, tf: &Transform) -> Vec3 {
    Vec3::new(tf.translation.x, p.eye_h + 0.012 * (p.walk_phase * 2.0).sin(), tf.translation.z)
}

/// X toggles the duel; R respawns you after the fly killed you (`handle3d`: `reset` is respawn_player while dead).
fn toggle_and_respawn(
    act: Res<Actions>,
    keys: Res<ButtonInput<KeyCode>>,
    mut duel: ResMut<Duel>,
    mut notes: ResMut<Notes>,
    mut saved: ResMut<SavedMsg>,
    mut sfx: MessageWriter<Sfx>,
    virt: Res<Time<Virtual>>,
    real: Res<Time<Real>>,
    arena: Res<Arena>,
    flies: Res<Flies>,
    bodies: Query<(&Fly, &Slot)>,
    parts: Query<&Transform, (With<FlyPart>, Without<Player>)>,
    mut player: Single<(&mut Player, &mut Transform)>,
) {
    let now = virt.elapsed_secs();
    if act.just("duel") {
        duel.toggle();
        let on = duel.on;
        notes.push(now, if on { "1V1      on: it has a blaster" } else { "1V1      off" }, Tag::Model);
        saved.set(format!("1v1 duel: {}", if on { "ON, the fly can shoot you" } else { "off" }), real.elapsed_secs());
        sfx.write(Sfx::new("click"));
    }
    if act.just("reset") && !keys.pressed(KeyCode::ShiftLeft) && duel.player_dead() {
        let fly_xz = bodies
            .iter()
            .find(|(_, s)| s.0 == flies.focus)
            .and_then(|(f, _)| parts.get(f.parts[Part::Thorax.index()]).ok())
            .map_or(Vec2::ZERO, |t| Vec2::new(t.translation.x, t.translation.z));
        let w = world::spec(arena.name());
        let (pos, yaw) = rules::respawn(fly_xz, w.rx, w.rz);
        let (p, tf) = &mut *player;
        tf.translation.x = pos.x;
        tf.translation.z = pos.y;
        p.yaw = yaw;
        p.pitch = -0.25;
        p.eye_h = EYE;
        duel.hp = rules::PLAYER_HP;
        duel.dead_at = None;
        notes.push(now, "RESPAWN  back in the fight", Tag::Model);
    }
}

/// The focused fly's body, thorax and head, if it is there and alive.
fn focus_fly<'a>(flies: &Flies, bodies: &'a mut Query<(Entity, &mut Fly, &Slot)>) -> Option<(Entity, Mut<'a, Fly>)> {
    bodies.iter_mut().find(|(_, f, s)| s.0 == flies.focus && !f.dead()).map(|(e, f, _)| (e, f))
}

/// Your smell reaches it, and what its mushroom body learned about it makes it run (`_scents`, `_memory_behavior`).
#[allow(clippy::too_many_arguments)]
fn scent_and_flee(
    mut duel: ResMut<Duel>,
    virt: Res<Time<Virtual>>,
    snap: Res<LatestSnapshot>,
    link: Res<SimLink>,
    flies: Res<Flies>,
    arena: Res<Arena>,
    tools: Res<crate::tools::ToolState>,
    mut learned: ResMut<crate::memory_play::Learned>,
    mut bodies: Query<(Entity, &mut Fly, &Slot)>,
    tf: Query<&Transform, (With<FlyPart>, Without<Player>)>,
    mut vel: Query<(&FlyPart, &mut LinearVelocity)>,
    player: Single<(&Player, &Transform)>,
    mut notes: ResMut<Notes>,
    mut popups: MessageWriter<Popup>,
    mut fired: MessageWriter<ReactionFired>,
) {
    duel.learn(&snap);
    let now = virt.elapsed_secs();
    duel.scent = false;
    if !duel.on || duel.player_dead() || tools.puppet {
        return;
    }
    let Some((e, mut f)) = focus_fly(&flies, &mut bodies) else { return };
    let Ok(head) = tf.get(f.parts[Part::Head.index()]).map(|t| t.translation) else { return };
    let Ok(thx) = tf.get(f.parts[Part::Thorax.index()]).map(|t| t.translation) else { return };
    let eye = eye_of(player.0, player.1);
    let d = Vec2::new(eye.x - head.x, eye.z - head.z).length();
    duel.scent = d < rules::SCENT_M;
    if !duel.scent {
        return;
    }
    link.0.send(Command::PokeFly { slot: f.slot, region: "scent:player".into(), side: None, strength: 0.35 }); // you smell like you
    let free = f.grabbed.is_none() && now >= f.react.escape_until;
    if !(free && now >= learned.avoid_ready && now >= f.react.stun_until) {
        return;
    }
    let (fear, like) = (duel.fear, duel.like);
    if fear - like > rules::FEAR_ACT {
        // its mushroom body learned to fear you
        learned.avoid_ready = now + 2.5;
        f.last_hit = eye;
        if f.can_fly() && fear - like > 0.6 {
            let outdoor = world::is_outdoor(arena.name());
            let launch = f.escape(now, thx, 2.4, false, outdoor);
            for (p, mut v) in &mut vel {
                if p.fly == e {
                    v.0 = launch;
                }
            }
        } else {
            let away = f.away_from(thx, eye);
            f.yaw_target = away.z.atan2(away.x);
            f.react.walk_until = now + 1.2;
            f.react.back_until = 0.0;
            f.run = true;
        }
        notes.push(now, format!("FLEE     fears you ({fear:.2})"), Tag::of_reaction("FLEE"));
        fired.write(ReactionFired { fly: e, name: "FLEE".into() });
        popups.write(Popup::new(head + Vec3::Y * 0.4, "RUN AWAY!", (140, 200, 255)));
    }
}

/// Getting hurt up close teaches it to fear you: PPL1 fires for 0.6 s after any damage while it smells you.
fn punish_on_damage(
    mut dmg: MessageReader<FlyDamage>,
    mut duel: ResMut<Duel>,
    virt: Res<Time<Virtual>>,
    flies: Res<Flies>,
    bodies: Query<(&Fly, &Slot)>,
) {
    let now = virt.elapsed_secs();
    for d in dmg.read() {
        if d.amount > 0.0 && duel.on && duel.scent && bodies.get(d.fly).is_ok_and(|(_, s)| s.0 == flies.focus) {
            duel.punish_until = duel.punish_until.max(now + rules::PUNISH_PULSE_S);
        }
    }
}

/// What the fly sees of you, fed into its LC10 and small-object neurons (`_duel_senses`).
#[allow(clippy::too_many_arguments)]
fn senses(
    mut duel: ResMut<Duel>,
    link: Res<SimLink>,
    flies: Res<Flies>,
    mut bodies: Query<(Entity, &mut Fly, &Slot)>,
    tf: Query<&Transform, (With<FlyPart>, Without<Player>)>,
    player: Single<(&Player, &Transform)>,
) {
    duel.tick += 1;
    if !duel.on || duel.player_dead() {
        return;
    }
    let Some((_, f)) = focus_fly(&flies, &mut bodies) else { return };
    if f.frozen_at.is_some() || f.wrapped || f.grabbed.is_some() {
        return;
    }
    let Ok(head) = tf.get(f.parts[Part::Head.index()]).map(|t| t.translation) else { return };
    let eye = eye_of(player.0, player.1);
    let s = rules::sense(head, f.yaw, eye, duel.valence, duel.fear, duel.like);
    duel.valence = s.valence;
    // the fly's own mushroom body remembers what you smell like, every third tick while you are close
    if duel.tick.is_multiple_of(3) && duel.scent {
        link.0.send(Command::Observe("player".into()));
    }
    let slot = f.slot;
    if let Some((side, recruit)) = s.track {
        link.0.send(Command::PokeFlyRecruit { slot, region: "track".into(), side: Some(side), strength: 0.05, recruit });
    }
    for (side, recruit) in s.small {
        link.0.send(Command::PokeFlyRecruit { slot, region: "small".into(), side: Some(side), strength: 0.05, recruit });
    }
}

/// Pellets fly, hit you or burst; you slump when dead; the red edge fades (`_pellets3d`, `_player_hit`).
#[allow(clippy::too_many_arguments)]
fn pellets(
    mut duel: ResMut<Duel>,
    virt: Res<Time<Virtual>>,
    arena: Res<Arena>,
    boxes: Res<WorldColliders>,
    settings: Res<Settings>,
    flies: Res<Flies>,
    mut fx: ResMut<Fx>,
    mut notes: ResMut<Notes>,
    mut popups: MessageWriter<Popup>,
    mut sfx: MessageWriter<Sfx>,
    mut shake: ResMut<Shake>,
    mut look: ResMut<MouseLook>,
    bodies: Query<(&Fly, &Slot)>,
    tf: Query<&Transform, (With<FlyPart>, Without<Player>)>,
    mut player: Single<(&mut Player, &Transform)>,
) {
    let now = virt.elapsed_secs();
    if duel.player_dead() {
        player.0.eye_h = rules::slump(player.0.eye_h); // you slump to the floor
    }
    let w = world::spec(arena.name());
    let half = Vec3::new(w.rx, w.ry, w.rz);
    let pos = Vec2::new(player.1.translation.x, player.1.translation.z);
    let eye_h = player.0.eye_h;
    let head = bodies
        .iter()
        .find(|(_, s)| s.0 == flies.focus)
        .and_then(|(f, _)| tf.get(f.parts[Part::Head.index()]).ok())
        .map_or(Vec3::new(0.0, 0.3, 0.0), |t| t.translation);
    let list = std::mem::take(&mut duel.pellets);
    let mut keep = Vec::with_capacity(list.len());
    for mut b in list {
        rules::step_pellet(&mut b);
        if duel.rng.f32() < 0.6 {
            fx.add(Particle { pos: b.p, vel: Vec3::ZERO, born: now, life: 0.2, size: 0.04, kind: Kind::Trail });
        }
        if rules::hits_player(b.p, pos, eye_h, !duel.player_dead()) {
            // `_player_hit`
            duel.hp = (duel.hp - rules::PELLET_DAMAGE).max(0.0);
            duel.hurt_flash = 1.0;
            shake.until = now + 0.15;
            shake.strength = 0.01;
            sfx.write(Sfx::new("hurt"));
            duel.stats.hits += 1;
            duel.reward_until = now + rules::REWARD_PULSE_S; // its reward dopamine neurons fire
            notes.push(now, "HIT YOU  PAM reward dopamine fires", Tag::of_reaction("HIT YOU"));
            if duel.rng.f32() < 0.5 {
                let words = ["GOTCHA!", "PEW PEW!", "TAKE THAT!"];
                let word = words[duel.rng.index(words.len())];
                popups.write(Popup::new(head + Vec3::Y * 0.45, word, (180, 255, 150)));
            }
            if duel.hp <= 0.0 {
                duel.dead_at = Some(now);
                duel.stats.deaths += 1;
                sfx.write(Sfx::new("death"));
                notes.push(now, "YOU DIED the fly wins this round", Tag::of_reaction("YOU DIED"));
                popups.write(Popup::new(head + Vec3::Y * 0.55, "FLY WINS!", (255, 120, 90)));
                look.captured = false;
            }
            continue;
        }
        let inside_box = boxes.0.iter().any(|bx| bx.contains(b.p, 0.0));
        if rules::expired(b.p, now - b.born, half) || inside_box {
            fx.puff(b.p.clamp(Vec3::new(-half.x, 0.02, -half.z), half), 5, 1.5, now);
            continue;
        }
        keep.push(b);
    }
    duel.pellets = keep;
    duel.hurt_flash = (duel.hurt_flash - 1.0 / 30.0).max(0.0);
    let _ = settings;
}

/// Steering from DNa02/DNa01 right minus left, and shooting when DNp35/DNpe052 fire above the threshold (`_duel_motor`).
#[allow(clippy::too_many_arguments)]
fn motor(
    mut duel: ResMut<Duel>,
    virt: Res<Time<Virtual>>,
    snap: Res<LatestSnapshot>,
    link: Res<SimLink>,
    flies: Res<Flies>,
    mut bodies: Query<(Entity, &mut Fly, &Slot)>,
    tf: Query<&Transform, (With<FlyPart>, Without<Player>)>,
    mut fx: ResMut<Fx>,
    mut notes: ResMut<Notes>,
    mut sfx: MessageWriter<Sfx>,
    player: Single<(&Player, &Transform)>,
) {
    if !duel.on {
        return;
    }
    let now = virt.elapsed_secs();
    let Some((_, mut f)) = focus_fly(&flies, &mut bodies) else { return };
    let Some(s) = snap.0.as_deref() else { return };
    let Some(info) = s.info.as_deref() else { return };
    if s.focus != flies.focus || s.hz.len() != info.group_names.len() || s.levels.len() != info.group_names.len() {
        return;
    }
    let col = |n: &str| info.group_names.iter().position(|g| g == n);
    let hz = |n: &str| col(n).map_or(0.0, |i| s.hz[i]);
    let level = |n: &str| col(n).map_or(0.0, |i| s.levels[i]);
    let Ok(thx) = tf.get(f.parts[Part::Thorax.index()]).map(|t| t.translation) else { return };
    let Ok(head) = tf.get(f.parts[Part::Head.index()]).map(|t| t.translation) else { return };
    let free = f.grabbed.is_none() && now >= f.react.escape_until;
    duel.steer = rules::steer_step(duel.steer, hz("turn_r") - hz("turn_l"));
    let grounded = free && now >= f.react.stun_until && thx.y < STAND3 + 0.25;
    if grounded
        && !duel.player_dead()
        && let Some(t) = rules::steer_yaw_target(f.yaw, duel.steer)
    {
        f.yaw_target = t;
    }
    if now < duel.reward_until {
        link.0.send(Command::PokeFly { slot: f.slot, region: "reward".into(), side: None, strength: 0.8 }); // hitting you is rewarding (game rule)
    }
    if now < duel.punish_until {
        link.0.send(Command::PokeFly { slot: f.slot, region: "punish".into(), side: None, strength: 1.0 }); // getting hurt is punishing (game rule)
    }
    duel.trigger = level("fire");
    let armed = f.grabbed.is_none() && !f.wrapped && now >= f.react.stun_until && f.frost < 0.5 && f.melt < 0.5 && !duel.player_dead();
    let fire = crate::reactions::thresholds().fire;
    if armed && duel.trigger > fire && now >= duel.fire_ready {
        duel.fire_ready = now + rules::FIRE_COOLDOWN;
        // `_fly_shoot`
        let (muzzle, fwd) = rules::muzzle(head, f.yaw);
        let eye = eye_of(player.0, player.1);
        let noise = Vec3::new(duel.rng.normal(), duel.rng.normal(), duel.rng.normal());
        let v = rules::shot_velocity(muzzle, fwd, eye, noise);
        duel.pellets.push(Pellet { p: muzzle, v, born: now });
        duel.stats.shots += 1;
        sfx.write(Sfx::vol("pew", 0.8));
        for _ in 0..6 {
            let jitter = Vec3::new(duel.rng.normal(), duel.rng.normal(), duel.rng.normal()) * 0.006;
            fx.add(Particle {
                pos: muzzle,
                vel: (v * 0.2 + jitter) * rules::TICKS_PER_S,
                born: now,
                life: 0.15,
                size: 0.05,
                kind: Kind::Muzzle,
            });
        }
        notes.push(now, format!("SHOOT    DNp35/DNpe052 x{:.1}", duel.trigger), Tag::of_reaction("SHOOT"));
    }
}

// --- drawing: the blaster under its head, and the pellets --------------------------------------------------------------------------
#[derive(Component)]
enum Gun {
    Body,
    Barrel,
    Muzzle,
}

#[derive(Component)]
struct PelletMesh(usize);

const POOL: usize = 8;

/// `KTF_DUEL=1`: start with the duel on (for screenshots and checks without a person at the keyboard).
fn start_on(mut duel: ResMut<Duel>) {
    if std::env::var_os("KTF_DUEL").is_some() {
        duel.on = true;
    }
}

/// `KTF_DEBUG=1`: the duel's numbers on the terminal every 2 s.
fn debug_line(duel: Res<Duel>, virt: Res<Time<Virtual>>, mut last: Local<f32>) {
    let now = virt.elapsed_secs();
    if duel.on && std::env::var_os("KTF_DEBUG").is_some() && now - *last >= 2.0 {
        *last = now;
        eprintln!(
            "duel: hp {:.0} steer {:+.1} Hz trigger x{:.1} valence {:+.2} fear {:.2} like {:.2} scent {} shots {} hits {} died {} pellets {}",
            duel.hp,
            duel.steer,
            duel.trigger,
            duel.valence,
            duel.fear,
            duel.like,
            duel.scent,
            duel.stats.shots,
            duel.stats.hits,
            duel.stats.deaths,
            duel.pellets.len()
        );
    }
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let sphere = meshes.add(Sphere::new(1.0).mesh().uv(16, 10));
    let cube = meshes.add(Cuboid::new(1.0, 1.0, 1.0));
    let cyl = meshes.add(Cylinder::new(1.0, 1.0).mesh().resolution(12));
    let mut mat = |c: [f32; 3], glow: f32| {
        materials.add(StandardMaterial {
            base_color: Color::srgb(c[0], c[1], c[2]),
            emissive: LinearRgba::from(Color::srgb(c[0], c[1], c[2])) * glow,
            perceptual_roughness: 0.6,
            ..default()
        })
    };
    let (body_m, barrel_m, muzzle_m, pellet_m) =
        (mat([0.25, 0.26, 0.3], 0.0), mat([0.45, 0.47, 0.5], 0.0), mat([0.45, 1.0, 0.4], 2.5), mat([0.55, 1.0, 0.45], 3.0));
    let hidden = (Visibility::Hidden, NotShadowCaster, NotShadowReceiver, Transform::default());
    commands.spawn((Gun::Body, Mesh3d(cube), MeshMaterial3d(body_m), hidden.clone()));
    commands.spawn((Gun::Barrel, Mesh3d(cyl), MeshMaterial3d(barrel_m), hidden.clone()));
    commands.spawn((Gun::Muzzle, Mesh3d(sphere.clone()), MeshMaterial3d(muzzle_m), hidden.clone()));
    for i in 0..POOL {
        commands.spawn((PelletMesh(i), Mesh3d(sphere.clone()), MeshMaterial3d(pellet_m.clone()), hidden.clone()));
    }
}

#[allow(clippy::too_many_arguments)]
fn draw(
    duel: Res<Duel>,
    flies: Res<Flies>,
    bodies: Query<(&Fly, &Slot)>,
    tf: Query<&Transform, (With<FlyPart>, Without<Gun>, Without<PelletMesh>)>,
    mut guns: Query<(&Gun, &mut Transform, &mut Visibility), Without<PelletMesh>>,
    mut shots: Query<(&PelletMesh, &mut Transform, &mut Visibility), Without<Gun>>,
    mut cloud: Single<&mut crate::gpu::ParticleCloud, With<crate::fx::FxCloud>>,
) {
    // the blaster, strapped under its head (the duelist only, while it lives)
    let fly = bodies.iter().find(|(_, s)| s.0 == flies.focus).map(|(f, _)| f).filter(|f| !f.dead() && !f.gone());
    let pose = fly.and_then(|f| {
        let t = |p: Part| tf.get(f.parts[p.index()]).ok().map(|t| t.translation);
        Some((f.yaw, t(Part::Head)?, t(Part::Thorax)?, t(Part::Abdomen)?))
    });
    let show = duel.on && pose.is_some();
    if let (true, Some((yaw, head, thx, abd))) = (show, pose) {
        let (muzzle, _) = rules::muzzle(head, yaw);
        let fwd = (head - abd).try_normalize().unwrap_or(Vec3::X);
        let side0 = rules::axes(yaw).1;
        let side = (side0 - fwd * side0.dot(fwd)).try_normalize().unwrap_or(Vec3::Z);
        let up = side.cross(fwd);
        let rot = Quat::from_mat3(&Mat3::from_cols(fwd, up, side));
        let base = thx + fwd * (10.0 * S) - up * (4.0 * S);
        for (g, mut t, mut v) in &mut guns {
            *v = Visibility::Inherited;
            *t = match g {
                Gun::Body => Transform { translation: (base + muzzle) / 2.0 - up * 0.02, rotation: rot, scale: Vec3::new(0.2, 0.06, 0.07) },
                Gun::Barrel => {
                    let (a, b) = (muzzle - fwd * 0.08, muzzle + fwd * 0.02);
                    Transform {
                        translation: (a + b) / 2.0,
                        rotation: Quat::from_rotation_arc(Vec3::Y, (b - a).normalize()),
                        scale: Vec3::new(0.018, (b - a).length(), 0.018),
                    }
                }
                Gun::Muzzle => Transform::from_translation(muzzle + fwd * 0.02).with_scale(Vec3::splat(0.022)),
            };
        }
    } else {
        for (_, _, mut v) in &mut guns {
            *v = Visibility::Hidden;
        }
    }
    for (m, mut t, mut v) in &mut shots {
        match duel.pellets.get(m.0) {
            Some(p) => {
                *v = Visibility::Inherited;
                *t = Transform::from_translation(p.p).with_scale(Vec3::splat(0.035));
            }
            None => *v = Visibility::Hidden,
        }
    }
    for p in &duel.pellets {
        cloud.push(p.p, 0.12, [0.4, 1.0, 0.35], 0.45);
    }
}

pub struct DuelPlugin;

impl Plugin for DuelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Duel>()
            .add_systems(Startup, (setup, start_on))
            .add_systems(Update, (toggle_and_respawn, punish_on_damage, debug_line))
            .add_systems(
                FixedUpdate,
                (scent_and_flee, senses, pellets, motor).chain().after(crate::fly::brain_reactions).before(crate::fly::control),
            )
            .add_systems(Update, draw.after(crate::fx::step).before(crate::gpu::flush_particles));
    }
}
