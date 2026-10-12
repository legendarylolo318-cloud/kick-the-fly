//! The fly: a physics ragdoll (head, thorax, abdomen as avian rigid bodies joined by spherical joints) with a controller that stands it,
//! walks it, turns it and flies it, plus everything that makes it react: the descending-neuron reactions, hits, damage, death.
//! Python: `game/kick3d.py:Fly3D` (a 17-particle verlet ragdoll) and the reaction block of `Game3D.update3d`.
//!
//! The Python body was particles pulled toward a pose by `strength` (1 = standing/flying, 0 = limp). Here the same `strength` scales a PD
//! controller on the rigid bodies: standing height, upright torque, walking velocity, hover thrust. Limp (stunned, grabbed, dead) means the
//! controller is off and the joints and gravity do the rest, which is the ragdoll. Legs and wings are visual (`fly_visuals.rs`): their
//! IK/flap is driven by the body, they do not collide.
//!
//! Brain link: reactions come from [`reactions::decide`] on the snapshot's group levels (CONNECTOME); hits poke the touch neurons through
//! [`Command::Poke`]. No simulation runs here.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::app::{LatestSnapshot, SimLink};
use crate::arena::WorldColliders;
use crate::config::{Arena, Settings};
use crate::feedback::{Notes, Popup, Sfx, Shake, Tag};
use crate::reactions::{self, Body, Levels, ReactState, Reaction};
use crate::rng::Rng;
use crate::sim_thread::Command;
use crate::world::{self, S, STAND3};

pub const HEAD_R: f32 = 16.0 * S;
pub const THX_R: f32 = 19.0 * S;
pub const ABD_R: f32 = 24.0 * S;
/// Rest offsets in the body frame (forward, up, side), m.
pub const HEAD_OFF: Vec3 = Vec3::new(42.0 * S, 6.0 * S, 0.0);
pub const ABD_OFF: Vec3 = Vec3::new(-50.0 * S, -6.0 * S, 0.0);
pub const MAX_HEALTH: f32 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Head,
    Thorax,
    Abdomen,
}

impl Part {
    pub const ALL: [Part; 3] = [Part::Head, Part::Thorax, Part::Abdomen];

    pub fn index(self) -> usize {
        self as usize
    }

    /// The touch neuron region a hit on this part pokes (`particle_region`).
    pub fn region(self) -> &'static str {
        match self {
            Part::Head => "head",
            _ => "body",
        }
    }
}

#[derive(Component)]
pub struct FlyPart {
    pub fly: Entity,
    pub part: Part,
}

/// Which brain a fly's body listens to (slot 0 is the first fly; see `flies_play`).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot(pub u32);

#[derive(Component)]
pub struct Fly {
    pub parts: [Entity; 3],
    /// The brain this body listens to (the same number as its [`Slot`]): what touches this fly pokes that brain.
    pub slot: u32,
    pub yaw: f32,
    pub yaw_target: f32,
    pub react: ReactState,
    pub run: bool,
    pub health: f32,
    pub dead_at: Option<f32>,
    pub char_: f32,
    pub soak: f32,
    pub melt: f32,
    pub frost: f32,
    pub venom: f32,
    pub burn_until: f32,
    pub zap_until: f32,
    pub dissolved_at: Option<f32>,
    pub frozen_at: Option<f32>,
    pub shattered_at: Option<f32>,
    pub hover: Vec3,
    pub fly_target: Vec3,
    pub wander: bool,
    pub power: f32,
    pub wrapped: bool,
    pub wet: f32,
    pub inebriation: f32,
    /// Feet stuck on flypaper: `(world point)`; two or more and it cannot fly.
    pub stuck: Vec<Vec3>,
    pub grabbed: Option<Part>,
    pub hold: Vec3,
    pub recover: f32,
    pub hurt: f32,
    pub last_hit: Vec3,
    pub action: &'static str,
    pub wind: Vec3,
    pub phase: f32,
    pub rng: Rng,
    pub prev_speed: f32,
    pub threat_x: Vec3,
    pub damage_src: &'static str,
    pub perch: Option<usize>,
    pub strength: f32,
    pub flying_now: bool,
    /// The joint entities (neck, waist), so a reset can remove them with the body.
    pub joints: Vec<Entity>,
}

impl Fly {
    pub fn dead(&self) -> bool {
        self.dead_at.is_some()
    }

    pub fn gone(&self) -> bool {
        self.dissolved_at.is_some() || self.shattered_at.is_some()
    }

    /// Can it take off right now? (`can_fly` in the reaction block.)
    pub fn can_fly(&self) -> bool {
        self.grabbed.is_none()
            && !self.wrapped
            && self.frost < 0.5
            && self.melt < 0.3
            && self.venom < 0.5
            && self.wet <= 0.0
            && self.stuck.len() < 2
            && !self.dead()
    }

    pub fn flying(&self, now: f32) -> bool {
        now < self.react.escape_until && self.can_fly() && self.frozen_at.is_none()
    }

    pub fn fwd(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    pub fn away_from(&self, thorax: Vec3, point: Vec3) -> Vec3 {
        let mut v = thorax - point;
        v.y = 0.0;
        v.try_normalize().unwrap_or(self.fwd())
    }

    pub fn stun(&mut self, now: f32, s: f32) {
        self.react.stun_until = self.react.stun_until.max(now + s);
        self.hurt = 1.0;
        self.react.escape_until = self.react.escape_until.min(now);
    }

    /// Take off and flee (or wander): a launch away from the last hit, a new target, and a cooldown.
    pub fn escape(&mut self, now: f32, thorax: Vec3, mut seconds: f32, wander: bool, outdoor: bool) -> Vec3 {
        if outdoor {
            seconds *= 2.5; // open sky: the same escape keeps going (game rule)
        }
        let away = self.away_from(thorax, self.last_hit);
        let mut launch = away * 4.0 * S * 60.0 + Vec3::Y * 9.0 * S * 60.0;
        if self.inebriation > 0.0 {
            let k = 2.5 * S * 60.0 * self.inebriation;
            launch += Vec3::new(self.rng.normal(), self.rng.normal(), self.rng.normal()) * k;
        }
        self.hover = thorax;
        self.wander = wander;
        self.new_target(thorax, if wander { None } else { Some(away) }, outdoor, &[]);
        self.react.escape_until = now + seconds;
        self.react.escape_ready = now + seconds + 1.0 + 2.0 * self.inebriation;
        launch
    }

    /// Pick where a flight goes: anywhere (wander) or away from a point; indoors it avoids the furniture.
    pub fn new_target(&mut self, thorax: Vec3, away: Option<Vec3>, outdoor: bool, boxes: &[world::Aabb]) {
        let w = world::spec(if outdoor { "field" } else { "room" });
        if outdoor {
            let t = match away {
                None => Vec3::new(
                    self.rng.uniform(-w.rx + 1.0, w.rx - 1.0),
                    self.rng.uniform(1.0, 4.0),
                    self.rng.uniform(-w.rz + 1.0, w.rz - 1.0),
                ),
                Some(a) => thorax + a * self.rng.uniform(4.0, 12.0) + Vec3::Y * self.rng.uniform(1.5, 5.0),
            };
            self.fly_target =
                t.clamp(Vec3::new(-w.fly_rx + 1.0, 0.5, -w.fly_rz + 1.0), Vec3::new(w.fly_rx - 1.0, w.fly_ry - 1.0, w.fly_rz - 1.0));
            return;
        }
        let (rx, rz) = (world::RX, world::RZ);
        let mut away = away;
        let mut t = thorax;
        for _ in 0..12 {
            t = match away {
                None => Vec3::new(self.rng.uniform(-rx + 0.8, rx - 0.8), self.rng.uniform(0.8, 2.4), self.rng.uniform(-rz + 0.8, rz - 0.8)),
                Some(a) => (thorax + a * self.rng.uniform(1.3, 2.3) + Vec3::Y * self.rng.uniform(0.5, 1.4))
                    .clamp(Vec3::new(-rx + 0.7, 0.8, -rz + 0.7), Vec3::new(rx - 0.7, 2.4, rz - 0.7)),
            };
            if !boxes.iter().any(|b| b.contains(t, 0.4)) {
                break;
            }
            away = None;
        }
        self.fly_target = t;
    }
}

// --- messages -----------------------------------------------------------------------------------------------------------------------
/// A touch on the fly: pokes the touch neurons of the part's region (and the rest of the body a little, since the legs and wings are
/// not separate bodies here).
#[derive(Message, Debug, Clone, Copy)]
pub struct FlyHit {
    pub fly: Entity,
    pub part: Part,
    pub strength: f32,
}

#[derive(Message, Debug, Clone, Copy)]
pub struct FlyDamage {
    pub fly: Entity,
    pub amount: f32,
    pub source: &'static str,
}

/// A reaction the brain made (WALK, TURN R, DODGE...), for the puppeteer's goals and the arcade.
#[derive(Message, Debug, Clone)]
pub struct ReactionFired {
    pub fly: Entity,
    pub name: String,
}

/// A fly died (for the kill cam, the autopsy and the score).
#[derive(Message, Debug, Clone, Copy)]
pub struct FlyDied {
    pub fly: Entity,
    pub source: &'static str,
}

/// Apply an instantaneous velocity change to every part of a fly (what `fly.impulse(i, v)` did for every particle).
pub fn kick_all(parts: &mut Query<(&FlyPart, &mut LinearVelocity)>, fly: Entity, dv: Vec3) {
    for (p, mut v) in parts.iter_mut() {
        if p.fly == fly {
            v.0 += dv;
        }
    }
}

// --- spawning -----------------------------------------------------------------------------------------------------------------------
/// The materials a fly's visuals share (tinted every frame for burns, frost and damage).
#[derive(Component)]
pub struct FlyLook {
    /// The body's materials with their healthy colours (0-255, `_draw_fly`'s `col(...)`), re-shaded every frame (`fly_visuals::tint`).
    pub tinted: Vec<(Handle<StandardMaterial>, [f32; 3])>,
    pub wing: Handle<StandardMaterial>,
    /// Never re-shaded: the glint on a living eye and the cross on a dead one.
    pub glint: Handle<StandardMaterial>,
    pub cross: Handle<StandardMaterial>,
}

pub fn spawn_fly(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    pos: Vec3,
    yaw: f32,
    seed: u64,
    slot: u32,
) -> Entity {
    let look = crate::fly_visuals::build_look(materials);
    let shapes = crate::fly_visuals::BodyMeshes::new(meshes);
    let rot = Quat::from_rotation_y(-yaw);
    let root = commands.spawn((Transform::from_translation(pos), Visibility::default())).id();
    let specs = [(Part::Head, HEAD_R, HEAD_OFF, 0.35), (Part::Thorax, THX_R, Vec3::ZERO, 0.9), (Part::Abdomen, ABD_R, ABD_OFF, 0.7)];
    let mut parts = [Entity::PLACEHOLDER; 3];
    for (part, r, off, mass) in specs {
        let at = pos + rot * off;
        let e = commands
            .spawn((
                FlyPart { fly: root, part },
                RigidBody::Dynamic,
                Collider::sphere(r),
                Mass(mass),
                Restitution::new(0.15),
                Friction::new(0.9),
                LinearDamping(0.05),
                AngularDamping(0.8),
                SleepingDisabled,
                GravityScale(1.0),
                TransformInterpolation,
                Transform::from_translation(at).with_rotation(rot),
                Visibility::default(),
            ))
            .with_children(|c| crate::fly_visuals::dress(c, part, &look, &shapes))
            .id();
        parts[part.index()] = e;
    }
    // the neck and the waist: spherical joints with a limited swing so the head and abdomen flop but do not fold through the thorax
    let thx = parts[Part::Thorax.index()];
    let mut joints_spawned = Vec::new();
    for (e, off) in [(parts[Part::Head.index()], HEAD_OFF), (parts[Part::Abdomen.index()], ABD_OFF)] {
        let anchor = off.normalize() * (THX_R * 0.9);
        let local = Vec3::new(-(off.length() - THX_R * 0.9), 0.0, 0.0);
        let _ = local;
        joints_spawned.push(
            commands
                .spawn(
                    SphericalJoint::new(thx, e).with_local_anchor1(anchor).with_local_anchor2(anchor - off).with_point_compliance(0.000_01),
                )
                .id(),
        );
    }
    commands.entity(root).insert((
        Fly {
            parts,
            slot,
            yaw,
            yaw_target: yaw,
            react: ReactState::default(),
            run: false,
            health: MAX_HEALTH,
            dead_at: None,
            char_: 0.0,
            soak: 0.0,
            melt: 0.0,
            frost: 0.0,
            venom: 0.0,
            burn_until: 0.0,
            zap_until: 0.0,
            dissolved_at: None,
            frozen_at: None,
            shattered_at: None,
            hover: pos,
            fly_target: pos,
            wander: false,
            power: 1.0,
            wrapped: false,
            wet: 0.0,
            inebriation: 0.0,
            stuck: Vec::new(),
            grabbed: None,
            hold: pos,
            recover: 1.0,
            hurt: 0.0,
            last_hit: pos,
            action: "idle",
            wind: Vec3::ZERO,
            phase: 0.0,
            rng: Rng::new(seed),
            prev_speed: 0.0,
            threat_x: pos,
            damage_src: "",
            perch: None,
            strength: 1.0,
            flying_now: false,
            joints: joints_spawned,
        },
        look,
        Slot(slot),
    ));
    root
}

fn spawn_primary_fly(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    spawn_fly(&mut commands, &mut meshes, &mut materials, Vec3::new(0.0, STAND3 + 0.05, -0.8), std::f32::consts::FRAC_PI_2, 1000, 0);
}

// --- reactions: the brain decides, the body does ---------------------------------------------------------------------------------
fn wrap(a: f32) -> f32 {
    (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

#[allow(clippy::too_many_arguments)]
pub fn brain_reactions(
    virt: Res<Time<Virtual>>,
    snap: Res<LatestSnapshot>,
    arena: Res<Arena>,
    mut flies: Query<(Entity, &mut Fly, &Slot)>,
    mut parts: Query<(&FlyPart, &mut LinearVelocity)>,
    transforms: Query<&Transform, With<FlyPart>>,
    mut notes: ResMut<Notes>,
    mut popups: MessageWriter<Popup>,
    mut sfx: MessageWriter<Sfx>,
    mut fired: MessageWriter<ReactionFired>,
    settings: Res<Settings>,
    sonifier: Option<Res<crate::audio::Sonifier>>,
    duel: Res<crate::duel_play::Duel>,
    focus: Res<crate::flies_play::Flies>,
) {
    // the buzz of a singing fly, unless the sonification's song voice is already sounding it (`Sonify.voicing_song`)
    let song_buzz = settings.song_buzz && !sonifier.is_some_and(|s| s.on && !s.silent && s.song_voice);
    let Some(s) = snap.0.as_deref() else { return };
    let Some(info) = s.info.as_deref() else { return };
    if s.levels.len() != info.group_names.len() || s.levels.is_empty() {
        return;
    }
    let now = virt.elapsed_secs();
    let outdoor = world::is_outdoor(arena.name());
    let mn9_col = info.group_names.iter().position(|n| n == "proboscis");
    // the living flies, for the lunge (`Game._aggression`)
    let lunges = settings.lunge;
    let rivals: Vec<(Entity, Vec3)> = flies
        .iter()
        .filter(|(_, f, _)| !f.dead() && !f.gone())
        .filter_map(|(e, f, _)| transforms.get(f.parts[Part::Thorax.index()]).ok().map(|t| (e, t.translation)))
        .collect();
    for (e, mut f, slot) in &mut flies {
        if f.dead() || f.frozen_at.is_some() || f.gone() {
            continue;
        }
        // the focused fly's levels are the snapshot's own; the others' travel beside them
        let (values, hz): (&[f32], &[f32]) = if slot.0 == s.focus {
            (&s.levels, &s.hz)
        } else {
            match s.others.iter().find(|o| o.slot == slot.0) {
                Some(o) if o.levels.len() == info.group_names.len() => (&o.levels, &o.hz),
                _ => continue,
            }
        };
        let thx_e = f.parts[Part::Thorax.index()];
        let Ok(thx) = transforms.get(thx_e).map(|t| t.translation) else { continue };
        let lv = Levels { names: &info.group_names, values };
        f.power = lv.get("fly");
        let body = Body {
            can_fly: f.can_fly(),
            free: f.grabbed.is_none() && now >= f.react.escape_until,
            grabbed: f.grabbed.is_some(),
            asleep_ok: true,
            eating: now < f.react.eating_until,
            // in a duel the focused fly's turning is the duel's steering, not the 1.5 s TURN reaction
            duel_focus: duel.on && slot.0 == focus.focus,
            mn9_hz: mn9_col.and_then(|c| hz.get(c)).copied().unwrap_or(0.0),
            rival: if lunges { rivals.iter().filter(|r| r.0 != e).map(|r| r.1.distance(thx)).min_by(f32::total_cmp) } else { None },
        };
        let mut rs = f.react.clone();
        let mut rng = Rng::new((now * 1000.0) as u64 ^ e.to_bits());
        let out = reactions::decide(&lv, &mut rs, body, now, &mut rng);
        f.react = rs;
        for r in out {
            let word = r.word();
            fired.write(ReactionFired { fly: e, name: crate::puppeteer::event_name(&r) });
            let note = match &r {
                Reaction::Dodge(l) => format!("DODGE    giant fiber DNp01 x{l:.1}"),
                Reaction::Yikes(l) => format!("FLY AWAY head-touch DNs x{l:.1}"),
                Reaction::TakeOff { level, .. } => format!("TAKE OFF DNg02 x{level:.2}"),
                Reaction::Run(l) => format!("RUN      body-touch DNs x{l:.1}"),
                Reaction::Kick(l) => format!("KICK     leg-touch DNs x{l:.1}"),
                Reaction::BackUp(l) => format!("BACK UP  MDN x{l:.1}"),
                Reaction::Walk(l) => format!("WALK     DNp09 x{l:.1}"),
                Reaction::Turn { right, diff, .. } => format!("TURN {}   DNa01/02 R-L {diff:+.1}", if *right { 'R' } else { 'L' }),
                Reaction::Groom(l) => format!("GROOM    aDN1/aDN2 x{l:.1}"),
                Reaction::Song(l) => format!("SONG     ps1 wing MNs x{l:.1}"),
                Reaction::Sleep(l) => format!("SLEEP    dFB FB6/FB7 x{l:.1}"),
                Reaction::Proboscis(l) => format!("PROBOSCIS MN9 x{l:.1}"),
                Reaction::Lunge(l) => format!("LUNGE    AVLP727m/pC1 x{l:.1}"),
            };
            notes.push(now, note, Tag::of_reaction(word));
            let above = thx + Vec3::Y * 0.4;
            match r {
                Reaction::Dodge(_) => {
                    f.react.stun_until = 0.0;
                    f.last_hit = f.threat_x;
                    let launch = f.escape(now, thx, 2.4, false, outdoor);
                    for (p, mut v) in &mut parts {
                        if p.fly == e {
                            v.0 = launch;
                        }
                    }
                    popups.write(Popup::new(above, "DODGE!", (170, 255, 200)));
                    sfx.write(Sfx::new("dodge"));
                }
                Reaction::Yikes(_) => {
                    f.react.stun_until = 0.0;
                    let launch = f.escape(now, thx, 2.4, false, outdoor);
                    for (p, mut v) in &mut parts {
                        if p.fly == e {
                            v.0 = launch;
                        }
                    }
                    popups.write(Popup::new(above, "YIKES!", (160, 230, 255)));
                }
                Reaction::TakeOff { seconds, .. } => {
                    let launch = f.escape(now, thx, seconds, true, outdoor);
                    for (p, mut v) in &mut parts {
                        if p.fly == e {
                            v.0 = launch;
                        }
                    }
                }
                Reaction::Run(_) => {
                    let away = f.away_from(thx, f.last_hit);
                    f.yaw_target = away.z.atan2(away.x);
                    f.run = true;
                }
                Reaction::Walk(_) => f.run = false,
                Reaction::Turn { radians, .. } => f.yaw_target = f.yaw + radians,
                Reaction::Sleep(_) => {
                    popups.write(Popup::new(above, "ZZZ", (170, 190, 255)));
                }
                Reaction::Groom(_) => {}
                Reaction::Lunge(_) => {
                    // thrown at the nearest other fly: 6 fly-pixels a tick toward it and 2.5 up (`fly.impulse`)
                    if let Some((_, at)) = rivals.iter().filter(|r| r.0 != e).min_by(|a, b| a.1.distance(thx).total_cmp(&b.1.distance(thx)))
                    {
                        let to = (*at - thx).normalize_or_zero();
                        f.yaw_target = to.z.atan2(to.x);
                        if let Ok((_, mut v)) = parts.get_mut(thx_e) {
                            v.0 += (to * 6.0 + Vec3::Y * 2.5) * world::S * 60.0;
                        }
                    }
                    popups.write(Popup::new(above, "LUNGE!", (255, 120, 80)));
                    sfx.write(Sfx::new("whack"));
                }
                Reaction::Song(_) if song_buzz => {
                    popups.write(Popup::new(above, "~ BUZZ ~", (255, 180, 220)));
                    sfx.write(Sfx::new("song"));
                }
                _ => {}
            }
        }
    }
}

// --- control: the same `strength` as the Python pose pull -------------------------------------------------------------------------
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn control(
    virt: Res<Time<Virtual>>,
    fixed: Res<Time<Fixed>>,
    arena: Res<Arena>,
    boxes: Res<WorldColliders>,
    mut flies: Query<(Entity, &mut Fly)>,
    mut bodies: Query<(&FlyPart, &Transform, &mut LinearVelocity, &mut AngularVelocity, &mut GravityScale)>,
) {
    let now = virt.elapsed_secs();
    let dt = fixed.delta_secs();
    let outdoor = world::is_outdoor(arena.name());
    for (e, mut f) in &mut flies {
        let mut tf = [(Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO); 3];
        for (p, t, v, _, _) in bodies.iter() {
            if p.fly == e {
                tf[p.part.index()] = (t.translation, t.rotation, v.0);
            }
        }
        let (thx, thx_rot, thx_vel) = tf[Part::Thorax.index()];
        if f.frozen_at.is_some() {
            for (p, _, mut v, mut w, _) in &mut bodies {
                if p.fly == e {
                    v.0 = Vec3::ZERO;
                    w.0 = Vec3::ZERO;
                }
            }
            continue;
        }
        let flying = f.flying(now);
        f.flying_now = flying;
        f.recover = if flying {
            1.0
        } else if f.grabbed.is_some() || now < f.react.stun_until || f.dead() || f.wrapped {
            0.0
        } else {
            (f.recover + dt * 2.0).min(1.0)
        };
        let height = thx.y - STAND3;
        let stiff = (1.0 - f.melt).max(0.0).sqrt() * (1.0 - f.frost) * (1.0 - f.venom);
        let mut strength = f.recover * if flying { 1.0 } else { (1.0 - height / (160.0 * S)).clamp(0.0, 1.0) * stiff };
        let in_water = matches!(arena.name(), "pool" | "kitchen") && water_at(arena.name(), thx);
        if f.grabbed.is_some() || (in_water && !flying) || f.dissolved_at.is_some() {
            strength = 0.0;
        }
        f.strength = strength;
        let g = crate::predators::GRAVITY;

        // walking intent
        let walking = now < f.react.walk_until;
        let backing = now < f.react.back_until;
        let speed = if backing {
            -1.7 * S * 60.0
        } else if walking {
            (if f.run { 5.0 } else { 2.3 }) * S * 60.0
        } else {
            0.0
        };
        let mut v_cmd = speed;
        if v_cmd != 0.0 && !flying {
            let ahead = thx + f.fwd() * v_cmd * 0.2;
            if blocked(ahead, &boxes.0, outdoor) {
                f.yaw_target = f.yaw + std::f32::consts::PI + f.rng.uniform(-0.7, 0.7); // walked into a wall: turn around
                v_cmd = 0.0;
            }
        }
        f.action = if backing {
            "back up"
        } else if walking {
            if f.run { "run" } else { "walk" }
        } else {
            "idle"
        };
        if v_cmd != 0.0 {
            f.phase += 0.12 * v_cmd.abs() / (S * 60.0) * dt * 60.0;
        }
        let max_turn = if flying { 0.15 } else { 0.09 } * dt * 60.0;
        f.yaw =
            if flying { f.yaw + wrap(f.yaw_target - f.yaw) * 0.15 } else { f.yaw + wrap(f.yaw_target - f.yaw).clamp(-max_turn, max_turn) };
        if f.inebriation > 0.0 && !f.dead() {
            let drift = 0.04 * f.inebriation * (now * 2.7).sin();
            f.yaw += drift;
        }

        // flight: the hover point moves toward the target; the thorax is pulled to it
        if flying {
            let d = f.fly_target - f.hover;
            let dist = d.length();
            let spd = (2.5 + 2.5 * f.power.min(2.0)) * S * 60.0;
            if f.perch.is_some() && dist < 25.0 * S {
                f.hover += d * 0.3;
            } else if dist < 25.0 * S {
                let t = thx;
                let ob = if outdoor { Vec::new() } else { boxes.0.clone() };
                f.new_target(t, None, outdoor, &ob);
            } else {
                f.hover += d / dist * (spd * dt).min(dist);
            }
            if Vec2::new(d.x, d.z).length() > 8.0 * S {
                f.yaw_target = d.z.atan2(d.x);
            }
            let wind = f.wind;
            f.hover += wind * 3.0 * dt;
            if f.inebriation > 0.0 {
                let k = f.inebriation;
                f.hover += Vec3::new((now * 3.7).sin() * 4.5, (now * 2.5).sin() * 2.0, (now * 2.9).cos() * 4.5) * S * k * 60.0 * dt;
            }
            f.action = "flying";
        }
        let rot_target = Quat::from_rotation_y(-f.yaw);
        let bob = if flying { 4.0 * S * (now * 9.0).sin() } else { 1.5 * S * (now * 2.2).sin() };
        let stand_h = STAND3 * (1.0 - 0.35 * f.melt) + bob;
        let wind = f.wind;
        let jit: [Vec3; 3] = std::array::from_fn(|_| Vec3::new(f.rng.normal(), f.rng.normal(), f.rng.normal()));
        let tremor = if f.inebriation > 0.0 && !f.dead() { 1.8 * S * f.inebriation * 60.0 } else { 0.0 };
        let flail = (now < f.react.flail_until || (f.grabbed.is_some() && f.rng.f32() < 0.5)) && !f.dead();
        let grabbed = f.grabbed;
        let hold = f.hold;
        let hover = f.hover;
        let wobble = f.inebriation * 0.3;
        let pitch_tilt = if flying { Quat::from_rotation_z(-0.18) } else { Quat::IDENTITY };
        for (p, t, mut v, mut w, mut gs) in &mut bodies {
            if p.fly != e {
                continue;
            }
            let i = p.part.index();
            // gravity cancels as `strength` rises: standing and flying hold the body up (Python: `p.y -= GRAV * (1 - strength)`)
            gs.0 = 1.0 - strength;
            // wind pushes the light parts harder (the Python wing/leg weights); the whole body drifts
            if wind != Vec3::ZERO {
                v.0 += wind * 3.0 * dt * (1.0 - 0.5 * strength);
            }
            if tremor > 0.0 {
                v.0 += jit[p.part.index()] * tremor * dt;
                let _ = wobble;
            }
            // the held part is pinned to the hold point, like the Python `p[grabbed] = pin`
            if grabbed == Some(p.part) {
                let to = hold - t.translation;
                v.0 = to * 28.0;
                continue;
            }
            if strength > 0.0 {
                let desired_rot = rot_target * pitch_tilt;
                let rest = rest_point(thx, rot_target, p.part);
                if p.part == Part::Thorax {
                    // vertical: hold the standing height, or hover on the target
                    let (ty, k, c) = if flying { (hover.y + bob, 60.0, 14.0) } else { (stand_h, 110.0, 20.0) };
                    let ay = (k * (ty - t.translation.y) - c * v.0.y) * strength;
                    v.0.y += ay * dt;
                    // horizontal: walking velocity, or the pull to the hover point
                    if flying {
                        let a = (hover - t.translation) * 60.0 - v.0 * 12.0;
                        v.0.x += a.x * dt * strength;
                        v.0.z += a.z * dt * strength;
                    } else {
                        let target = f.fwd_dir() * v_cmd;
                        let dv = (target - Vec3::new(v.0.x, 0.0, v.0.z)) * (14.0 * strength);
                        v.0.x += dv.x * dt;
                        v.0.z += dv.z * dt;
                    }
                    // upright and facing: a PD torque on the orientation error, expressed as an angular velocity change
                    let err = (desired_rot * t.rotation.inverse()).normalize();
                    let (axis, angle) = err.to_axis_angle();
                    let angle = wrap(angle);
                    let alpha = axis * angle * 90.0 - w.0 * 14.0;
                    let dw = alpha * dt * strength;
                    w.0 += dw;
                } else {
                    // head and abdomen follow the thorax's pose (the Python pose pull on every particle)
                    let pull = 130.0 * strength;
                    let a = (rest - t.translation) * pull - (v.0 - thx_vel) * 16.0 * strength;
                    v.0 += a * dt;
                    let err = (desired_rot * t.rotation.inverse()).normalize();
                    let (axis, angle) = err.to_axis_angle();
                    let dw = (axis * wrap(angle) * 40.0 - w.0 * 8.0) * dt * strength;
                    w.0 += dw;
                }
            }
            if !f.stuck.is_empty() {
                // feet in the glue: it cannot walk off, and the glue holds the body down
                let k = if f.stuck.len() >= 2 { 0.5 } else { 0.85 };
                v.0.x *= k;
                v.0.z *= k;
            }
            if submerged(arena.name(), t.translation) {
                // the water buoys the body up and drags on it (Python: `p.y += 1.35 px` and a 12% drag per frame)
                v.0.y += 26.0 * dt;
                v.0 *= 0.88f32.powf(dt * 60.0);
            }
            if flail && p.part != Part::Thorax {
                v.0 += jit[i] * (2.5 * S * 60.0) * dt * 6.0;
            }
            if f.dead() && p.part == Part::Thorax {
                // a dead fly settles: drain the leftover motion
                v.0 *= 1.0 - 0.6 * dt;
            }
            let _ = (g, thx_rot, i);
        }
        f.hurt = (f.hurt - dt * 3.0).max(0.0);
    }
}

impl Fly {
    fn fwd_dir(&self) -> Vec3 {
        self.fwd()
    }
}

fn rest_point(thx: Vec3, rot: Quat, part: Part) -> Vec3 {
    match part {
        Part::Head => thx + rot * HEAD_OFF,
        Part::Abdomen => thx + rot * ABD_OFF,
        Part::Thorax => thx,
    }
}

/// Is this point under water (the pool, or the kitchen sink's basin)?
pub fn submerged(arena: &str, p: Vec3) -> bool {
    match arena {
        "pool" => p.y < world::WATER3,
        "kitchen" => crate::kitchen::in_sink(p) && p.y < crate::kitchen::SINK_WATER,
        _ => false,
    }
}

fn water_at(arena: &str, p: Vec3) -> bool {
    match arena {
        "pool" => p.y < world::WATER3 + 30.0 * S,
        "kitchen" => crate::kitchen::in_sink(p) && p.y < crate::kitchen::SINK_WATER + 30.0 * S,
        _ => false,
    }
}

/// `Fly3D._blocked`: a point the walking fly will not step to (outside the walls, or inside furniture).
fn blocked(p: Vec3, boxes: &[world::Aabb], outdoor: bool) -> bool {
    if outdoor {
        return boxes.iter().any(|b| b.contains(Vec3::new(p.x, 0.2, p.z), 0.35));
    }
    if !(-world::RX + 0.5 < p.x && p.x < world::RX - 0.5 && -world::RZ + 0.5 < p.z && p.z < world::RZ - 0.5) {
        return true;
    }
    boxes.iter().any(|b| b.contains(Vec3::new(p.x, 0.2, p.z), 0.35))
}

// --- impacts, hits, damage ------------------------------------------------------------------------------------------------------
/// A hard stop is a hit: the thorax's speed loss between physics steps, in the units the Python thresholds used (px/frame * 60 * 0.006 m).
#[allow(clippy::too_many_arguments)]
fn impacts(
    virt: Res<Time<Virtual>>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<(&FlyPart, &LinearVelocity, &Transform)>,
    mut hits: MessageWriter<FlyHit>,
    mut dmg: MessageWriter<FlyDamage>,
    mut popups: MessageWriter<Popup>,
    mut sfx: MessageWriter<Sfx>,
) {
    let now = virt.elapsed_secs();
    for (e, mut f) in &mut flies {
        let Some((_, v, t)) = parts.iter().find(|(p, _, _)| p.fly == e && p.part == Part::Thorax) else { continue };
        let speed = v.0.length();
        let lost = f.prev_speed - speed;
        f.prev_speed = speed;
        // Python: sp in px/frame; hits above 9, damage above 18, stun above 24
        let sp = lost / (S * 60.0);
        if sp > 9.0 && !f.dead() {
            let s = ((sp - 9.0) / 35.0).clamp(0.05, 1.0);
            hits.write(FlyHit { fly: e, part: Part::Thorax, strength: s });
            if sp > 18.0 {
                dmg.write(FlyDamage {
                    fly: e,
                    amount: ((sp - 18.0) * 0.35).min(8.0),
                    source: if t.translation.y < 0.3 { "the floor" } else { "the wall" },
                });
            }
            if sp > 24.0 {
                sfx.write(Sfx::vol("bonk", 0.4 + 0.6 * s));
                f.stun(now, 0.8 * s + 0.3);
                if f.rng.chance(0.5) {
                    let words = ["BONK!", "OOF!", "SPLAT!", "THWACK!", "BZZT!", "OW!"];
                    popups.write(Popup::new(t.translation + Vec3::Y * 0.3, words[f.rng.index(words.len())], (255, 245, 235)));
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_hits_and_damage(
    virt: Res<Time<Virtual>>,
    link: Res<SimLink>,
    settings: Res<Settings>,
    mut flies: Query<&mut Fly>,
    slots: Query<&Slot>,
    tools: Res<crate::tools::ToolState>,
    mut hits: MessageReader<FlyHit>,
    mut dmg: MessageReader<FlyDamage>,
    mut died: MessageWriter<FlyDied>,
    mut notes: ResMut<Notes>,
    mut sfx: MessageWriter<Sfx>,
    mut popups: MessageWriter<Popup>,
    mut shake: ResMut<Shake>,
    transforms: Query<&Transform, With<FlyPart>>,
) {
    let now = virt.elapsed_secs();
    for h in hits.read() {
        let Ok(f) = flies.get(h.fly) else { continue };
        if f.dead() || tools.puppet {
            continue;
        }
        let s = h.strength.clamp(0.0, 1.0);
        let region = h.part.region();
        let slot = slots.get(h.fly).map_or(0, |s| s.0);
        link.0.send(Command::PokeFly { slot, region: region.into(), side: None, strength: s });
        if h.part != Part::Head {
            // the legs and wings are not separate bodies here: a body blow is felt there too
            link.0.send(Command::PokeFly { slot, region: "legs".into(), side: Some('L'), strength: 0.6 * s });
            link.0.send(Command::PokeFly { slot, region: "legs".into(), side: Some('R'), strength: 0.6 * s });
            link.0.send(Command::PokeFly { slot, region: "wing".into(), side: None, strength: 0.4 * s });
        }
    }
    for d in dmg.read() {
        let Ok(mut f) = flies.get_mut(d.fly) else { continue };
        if f.dead() || d.amount <= 0.0 {
            continue;
        }
        if d.amount > 0.0 {
            f.damage_src = d.source;
        }
        let floor = if settings.immortal { 1.0 } else { 0.0 };
        f.health = (f.health - d.amount).max(floor);
        if f.health <= 0.0 {
            f.dead_at = Some(now);
            f.grabbed = None;
            link.0.send(Command::KillFly(slots.get(d.fly).map_or(0, |s| s.0)));
            sfx.write(Sfx::new("death"));
            notes.push(now, "DIED     brain drive cut, activity fading", Tag::GameRule);
            if let Ok(t) = transforms.get(f.parts[Part::Head.index()]) {
                popups.write(Popup::new(t.translation + Vec3::Y * 0.5, "K.O.!", (255, 90, 80)));
            }
            shake.until = now + 0.3;
            shake.strength = 0.03;
            died.write(FlyDied { fly: d.fly, source: d.source });
        }
    }
}

/// Pain: a fly in pain pokes its punishment neurons (the Python `pain_index > 25` -> PPL1 pulse, simplified to damage taken recently).
fn immortal_regen(virt: Res<Time<Virtual>>, settings: Res<Settings>, mut flies: Query<&mut Fly>) {
    let now = virt.elapsed_secs();
    for mut f in &mut flies {
        if settings.immortal && !f.dead() && f.health < MAX_HEALTH && now - f.react.eating_until > 1.5 {
            f.health = (f.health + 0.1).min(MAX_HEALTH);
        }
    }
}

/// `KTF_DEBUG=1`: print each fly's part positions once a second.
fn debug_dump(time: Res<Time<Virtual>>, flies: Query<&Fly>, parts: Query<(&FlyPart, &Transform, &LinearVelocity)>, mut last: Local<f32>) {
    if std::env::var_os("KTF_DEBUG").is_none() || time.elapsed_secs() - *last < 1.0 {
        return;
    }
    *last = time.elapsed_secs();
    for (p, t, v) in &parts {
        if let Ok(f) = flies.get(p.fly) {
            eprintln!(
                "{:?} pos {:.3?} vel {:.2?} strength {:.2} flying {} yaw {:.2}",
                p.part, t.translation, v.0, f.strength, f.flying_now, f.yaw
            );
        }
    }
}

/// Put a fresh fly in place of the current one (R, a new arena).
#[derive(Message, Debug, Clone, Default)]
pub struct ResetFly {
    /// `(x metres, yaw)`: put the new fly here (the puppeteer's levels); None: the usual spot.
    pub place: Option<(f32, f32)>,
    /// A loaded save's flies, the first one taking the focused brain: bodies where they were, with their health, and the other brains put back.
    pub restore: Vec<FlyPose>,
}

/// One fly of a loaded save.
#[derive(Debug, Clone)]
pub struct FlyPose {
    pub thorax: Vec3,
    pub yaw: f32,
    pub health: f32,
    pub dead: bool,
    pub seed: u64,
    /// The brain state of a fly that is not the focused one (the focused fly's is imported by `ImportState`).
    pub state: Option<Box<fly_sim::brain::BrainState>>,
}

fn reset_fly(
    mut commands: Commands,
    mut msgs: MessageReader<ResetFly>,
    flies: Query<(Entity, &Fly, &Slot, Option<&crate::fly_visuals::FlyVisuals>)>,
    link: Res<SimLink>,
    mut flies_res: ResMut<crate::flies_play::Flies>,
    cfg: Option<Res<crate::menu_ui::UserConfig>>,
    virt: Res<Time<Virtual>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut seed: Local<u64>,
) {
    let Some(req) = msgs.read().last().cloned() else {
        return;
    };
    for (e, f, slot, vis) in &flies {
        // every brain but the focused fly's goes with its body; the new fly takes the focused fly's brain (revived)
        if slot.0 != flies_res.focus {
            link.0.send(Command::RemoveFly(slot.0));
        }
        for p in f.parts {
            commands.entity(p).despawn();
        }
        for j in &f.joints {
            commands.entity(*j).despawn();
        }
        if let Some(v) = vis {
            for x in v.loose() {
                commands.entity(x).despawn();
            }
        }
        commands.entity(e).despawn();
    }
    link.0.send(Command::Revive);
    *seed += 1;
    if !req.restore.is_empty() {
        let now = virt.elapsed_secs();
        let sigma = fly_sim::cards::sigma_of(cfg.as_ref().map_or("off", |c| c.0.get_str("brain.individuality")), None);
        for (i, pose) in req.restore.iter().enumerate() {
            let slot = if i == 0 { flies_res.focus } else { flies_res.alloc() };
            let e = spawn_fly(&mut commands, &mut meshes, &mut materials, pose.thorax, pose.yaw, pose.seed, slot);
            if i > 0 {
                link.0.send(Command::SpawnFly { slot, seed: pose.seed, sigma });
                if let Some(st) = &pose.state {
                    link.0.send(Command::ImportFly { slot, state: st.clone() });
                }
            }
            let (health, dead) = (pose.health, pose.dead);
            commands.queue(move |world: &mut World| {
                if let Some(mut f) = world.get_mut::<Fly>(e) {
                    f.health = health;
                    if dead {
                        f.dead_at = Some(now);
                    }
                }
            });
        }
        return;
    }
    let (x, yaw) = req.place.unwrap_or((0.0, std::f32::consts::FRAC_PI_2));
    spawn_fly(&mut commands, &mut meshes, &mut materials, Vec3::new(x, STAND3 + 0.05, -0.8), yaw, 1000 + *seed, flies_res.focus);
}

pub struct FlyPlugin;

impl Plugin for FlyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<crate::flies_play::Flies>()
            .add_message::<ReactionFired>()
            .add_message::<ResetFly>()
            .add_message::<FlyHit>()
            .add_message::<FlyDamage>()
            .add_message::<FlyDied>()
            .add_systems(Startup, spawn_primary_fly)
            .add_systems(FixedUpdate, (brain_reactions, control).chain())
            .add_systems(FixedPostUpdate, impacts.after(PhysicsSystems::Writeback))
            .add_systems(Update, (apply_hits_and_damage, immortal_regen, debug_dump, reset_fly));
    }
}
