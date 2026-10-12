//! The eighteen tools and what they do to the fly. Python: `Game3D.use_tool3d` and its helpers (`_swat3d`, `_explode3d`, `_zap3d`, `_jet`,
//! `_sugar3d`, `_alcohol3d`, `_decoy3d`, `_kick`, `_effects_one`).
//!
//! The physics tools act through avian: the hand pins a part of the ragdoll to a point in front of you (release to throw: it keeps the
//! velocity it had), the flick, the swatter, the bomb and the kick apply velocity changes to the rigid bodies, the bomb itself and the
//! treats are rigid bodies that bounce and land. Everything a tool does to the brain goes through [`FlyHit`] (touch neurons) or
//! `Command::Poke` (heat, cold, smell, taste, reward): the connectome decides what the fly then does about it.
//!
//! The spider and the three predators are in `predator_play.rs`; the tools only ask for them with a [`SpawnPredator`] message.

use avian3d::prelude::*;
use bevy::ecs::system::SystemParam;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;

use crate::actions::Actions;
use crate::app::{LatestSnapshot, SimLink};
use crate::config::{Arena, Settings};
use crate::feedback::{Notes, Popup, Sfx, Shake, Tag};
use crate::fly::{Fly, FlyDamage, FlyHit, FlyPart, MAX_HEALTH, Part};
use crate::fx::{Fx, Kind};
use crate::menu_ui::UserConfig;
use crate::player::{Aim, MouseLook, Player};
use crate::rng::Rng;
use crate::senses::{ScentSource, Scents, Threats};
use crate::sim_thread::Command;
use crate::world::{S, STAND3};

/// What nearly every tool system writes: the popups, sounds, shake and log (one parameter instead of four).
#[derive(SystemParam)]
pub struct Fb<'w> {
    pub popups: MessageWriter<'w, Popup>,
    pub sfx: MessageWriter<'w, Sfx>,
    pub shake: ResMut<'w, Shake>,
    pub notes: ResMut<'w, Notes>,
}

/// Touch and damage messages to the fly.
#[derive(SystemParam)]
pub struct FlyFx<'w> {
    pub hits: MessageWriter<'w, FlyHit>,
    pub dmg: MessageWriter<'w, FlyDamage>,
}

/// Mouse, look state and aim ray.
#[derive(SystemParam)]
pub struct Pointer<'w> {
    pub mouse: Res<'w, ButtonInput<MouseButton>>,
    pub look: Res<'w, MouseLook>,
    pub aim: Res<'w, Aim>,
    pub photo: Res<'w, crate::photo_play::Photo>,
}

/// Mesh and material stores for spawning things.
#[derive(SystemParam)]
pub struct Gfx<'w> {
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub materials: ResMut<'w, Assets<StandardMaterial>>,
}

pub const REACH: f32 = 3.0;
pub const GRAB_REACH: f32 = 3.2;

/// `TOOLS`: id, label, hint.
pub const TOOLS: [(&str, &str, &str); 18] = [
    ("hand", "HAND", "drag the fly and throw it"),
    ("flick", "FLICK", "click near the fly"),
    ("swatter", "SWAT", "click on the fly"),
    ("bomb", "BOMB", "click to throw a bomb, 1.5 s fuse"),
    ("torch", "TORCH", "hold: burns it, maxes out pain"),
    ("cleaner", "CLEANER", "hold: brake cleaner dissolves it"),
    ("zapper", "ZAP", "click: electric shock through its body"),
    ("freeze", "FREEZE", "hold: freezes it solid, then smash the ice"),
    ("spider", "SPIDER", "click: drop a spider that hunts it"),
    ("sugar", "SUGAR", "click: drop sugar to reward it"),
    ("alcohol", "ALCOHOL", "click: drop alcohol, sweet PAM reward but escalating drunkenness"),
    ("laser", "LASER", "targeted laser: hold to stimulate or silence cell types in real time"),
    ("cva", "CVA", "puffs cVA pheromone: activates Or67d/DA1 glomerulus"),
    ("decoy", "DECOY", "spawns a decoy female target to evoke courtship"),
    ("fruit", "FRUIT", "click: drop ripe fruit, eaten like sugar"),
    ("frog", "FROG", "click: put down a frog; its tongue is very fast"),
    ("dragonfly", "DRAGONFLY", "click: a dragonfly chases flying flies from above"),
    ("mantis", "MANTIS", "click: a mantis creeps up slowly, then strikes"),
];

/// The size a tool presents to the fly's eyes when it is held near it (`TOOL_SIZE`).
fn tool_size(name: &str) -> f32 {
    match name {
        "flick" => 0.06,
        "swatter" => 0.2,
        "bomb" => 0.08,
        "torch" | "cleaner" | "freeze" => 0.09,
        "zapper" => 0.12,
        "spider" | "laser" => 0.08,
        "alcohol" => 0.07,
        _ => 0.08,
    }
}

/// The targets the laser cycles through (`DEFAULT_TARGETS`) as neuron specs.
pub const LASER_TARGETS: [(&str, &str); 11] = [
    ("LC10", "prefix:LC10"),
    ("LPLC2 + LC4", "type:LPLC2,LC4"),
    ("DNp01 (giant fiber)", "type:DNp01"),
    ("DNg02 (wing power)", "prefix:DNg02"),
    ("MDN (moonwalker)", "type:MDN"),
    ("aDN1/aDN2 (grooming)", "type:DNg62,DNge078"),
    ("MN9 (proboscis)", "type:MN9"),
    ("PPL1 (punishment)", "prefix:PPL1"),
    ("PAM (reward)", "prefix:PAM"),
    ("DNa02 (steering)", "type:DNa02"),
    ("DNp09 (walking)", "type:DNp09"),
];
/// `BASE_STIM_CURRENT` / `BASE_SILENCE_CURRENT` (0.48, -2.40) before the sim's x4 external gain.
const LASER_STIM: f32 = 0.12;
const LASER_SILENCE: f32 = -0.6;

#[derive(Resource)]
pub struct ToolState {
    pub selected: usize,
    pub page: usize,
    pub hold_dist: f32,
    pub holding: bool,
    pub throw_t: f32,
    pub swing_t: f32,
    /// When the finger last flicked (`Game3D.flick_t`: the held hand's finger snaps for 0.15 s).
    pub flick_t: f32,
    /// How many treats lie about: sugar and fruit together, and alcohol (Python allows 3 of each at a time).
    pub treats_out: [usize; 2],
    pub zap_ready: f32,
    pub kick_t: f32,
    pub laser_target: usize,
    pub laser_silence: bool,
    pub laser_on: Option<(usize, bool)>,
    /// The Lab's laser (`lab/laser.py:LaserState`): target text, effect, intensity, hold or pulse; and the rows it currently drives.
    pub laser: fly_sim::laser::LaserState,
    laser_rows: (String, Vec<u32>),
    laser_applied: (Vec<u32>, f32),
    laser_was_held: bool,
    /// Puppeteer: every tool but the laser is locked and nothing of yours touches the fly.
    pub puppet: bool,
    /// How many times a tool was used (`Game.tool_uses`) and which one last (`last_tool_used`): the tutorial reads them.
    pub uses: u32,
    pub last_used: &'static str,
    swings: Vec<f32>,
    grabbed_decoy: Option<Entity>,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            selected: 0,
            page: 0,
            hold_dist: 1.2,
            holding: false,
            throw_t: -9.0,
            swing_t: -9.0,
            flick_t: -9.0,
            treats_out: [0; 2],
            zap_ready: 0.0,
            kick_t: -9.0,
            laser_target: 2,
            laser_silence: false,
            laser_on: None,
            laser: fly_sim::laser::LaserState::default(),
            laser_rows: (String::new(), Vec::new()),
            laser_applied: (Vec::new(), 0.0),
            laser_was_held: false,
            puppet: false,
            uses: 0,
            last_used: "",
            swings: Vec::new(),
            grabbed_decoy: None,
        }
    }
}

impl ToolState {
    pub fn name(&self) -> &'static str {
        TOOLS[self.selected].0
    }
}

/// Ask `predator_play` for a spider/frog/dragonfly/mantis at a point on the floor.
#[derive(Message, Debug, Clone)]
pub struct SpawnPredator {
    pub kind: &'static str,
    pub at: Vec3,
}

#[derive(Component)]
pub struct Bomb {
    pub born: f32,
}

#[derive(Component)]
pub struct Treat {
    pub kind: &'static str,
    pub left: f32,
    /// It has come to rest on something and stays there (`sugars3[..]["landed"]`); flies only walk to a landed treat.
    pub landed: bool,
    radius: f32,
    born: f32,
}

/// Has a thrown treat landed? On the floor at once (`_sugar3d`: `p.y <= 0.035`), or once it lies still on something higher (a table).
pub fn treat_landed(y: f32, radius: f32, age: f32, speed: f32) -> bool {
    y <= radius + 0.01 || (age > 0.25 && speed < 0.2)
}

impl Treat {
    /// A full treat (a ball of this radius) that was let go at `now`.
    pub fn new(kind: &'static str, radius: f32, now: f32) -> Self {
        Self { kind, left: 1.0, landed: false, radius, born: now }
    }
}

#[derive(Component)]
pub struct Decoy;

/// A cVA puff in the air, for its visual.
#[derive(Component)]
struct Puff {
    until: f32,
}

// --- aiming helpers ---------------------------------------------------------------------------------------------------------------
/// Which part of which fly the aim ray passes nearest to: `(fly, part, distance along the ray, miss distance)`.
pub fn nearest_to_ray(
    parts: &Query<(&FlyPart, &Transform, Entity)>,
    flies: &Query<(Entity, &mut Fly)>,
    eye: Vec3,
    dir: Vec3,
    reach: f32,
    max_perp: f32,
) -> Option<(Entity, Part, f32, f32)> {
    let mut best: Option<(Entity, Part, f32, f32)> = None;
    for (p, t, _) in parts.iter() {
        if flies.get(p.fly).is_ok_and(|(_, f)| f.gone()) {
            continue;
        }
        let r = match p.part {
            Part::Head => crate::fly::HEAD_R,
            Part::Thorax => crate::fly::THX_R,
            Part::Abdomen => crate::fly::ABD_R,
        };
        let rel = t.translation - eye;
        let along = rel.dot(dir);
        let perp = (rel - dir * along).length() - r;
        if along > 0.0 && along < reach && perp < max_perp && best.is_none_or(|b| perp < b.3) {
            best = Some((p.fly, p.part, along, perp));
        }
    }
    best
}

/// The point on the floor the aim ray hits (or two metres ahead when looking up).
pub fn floor_point(aim: &Aim, bounds: (f32, f32)) -> Vec3 {
    let pt =
        if aim.dir.y < -0.05 { aim.eye + aim.dir * (-aim.eye.y / aim.dir.y) } else { aim.eye + Vec3::new(aim.dir.x, 0.0, aim.dir.z) * 2.0 };
    Vec3::new(pt.x.clamp(-bounds.0 + 0.4, bounds.0 - 0.4), 0.0, pt.z.clamp(-bounds.1 + 0.4, bounds.1 - 0.4))
}

/// Where a tool's effect comes from: the nozzle in your hand, right-down-forward of the eye.
pub fn tool_tip(aim: &Aim) -> Vec3 {
    aim.eye + aim.right * 0.27 - aim.up * 0.2 + aim.dir * 0.55
}

// --- selection ----------------------------------------------------------------------------------------------------------------------
/// The hotbar: the player's loadout (`core/loadout.py`) resolved for the current mode and preset (`Game.loadout`).
#[derive(Resource)]
pub struct ActiveLoadout(pub ktf_data::loadout::Loadout);

impl Default for ActiveLoadout {
    fn default() -> Self {
        Self(ktf_data::loadout::build("auto", &[], false, false, "play"))
    }
}

/// Keep the loadout in step with Settings (mode, Settings > Controls > Tool loadout, the custom slots): rebuilt when what it depends on
/// changes, keeping the page the player was on.
fn sync_loadout(cfg: Res<UserConfig>, mut lo: ResMut<ActiveLoadout>, mut sig: Local<Option<(String, String, bool, Vec<String>)>>) {
    let now = (
        cfg.0.get_str("controls.loadout_preset").to_string(),
        cfg.0.get_str("brain.mode").to_string(),
        cfg.0.larva(),
        cfg.0.custom_loadout.clone(),
    );
    if sig.as_ref() == Some(&now) {
        return;
    }
    let page = lo.0.page;
    lo.0 = ktf_data::loadout::resolve(&cfg.0);
    lo.0.page = page.min(lo.0.n_pages() - 1);
    *sig = Some(now);
}

fn tool_index(name: &str) -> usize {
    TOOLS.iter().position(|t| t.0 == name).unwrap_or(0)
}

fn select(
    act: Res<Actions>,
    mut wheel: MessageReader<MouseWheel>,
    mut st: ResMut<ToolState>,
    mut lo: ResMut<ActiveLoadout>,
    look: Res<MouseLook>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if !look.captured {
        return;
    }
    if st.puppet {
        st.selected = TOOLS.iter().position(|t| t.0 == "laser").unwrap_or(0);
        if keys.just_pressed(KeyCode::Semicolon) {
            st.laser_target = (st.laser_target + 1) % crate::puppeteer::PALETTE.len();
        }
        if keys.just_pressed(KeyCode::Quote) {
            st.laser_silence = !st.laser_silence;
        }
        return;
    }
    // slot1..slot10 are the rebindable keys 1-9, 0: the tool in that slot on the page showing (`Game.select_slot`)
    for i in 0..10 {
        if act.just(&format!("slot{}", i + 1))
            && let Some(name) = lo.0.slot_tool(i)
        {
            st.selected = tool_index(name);
        }
    }
    if act.just("page_prev") && lo.0.turn_page(-1) {
        st.page = lo.0.page;
    }
    if act.just("page_next") && lo.0.turn_page(1) {
        st.page = lo.0.page;
    }
    for w in wheel.read() {
        // the wheel steps through the loadout, wrapping (`Game.cycle_tool`)
        let cur = TOOLS[st.selected].0;
        let name = lo.0.neighbor(cur, if w.y > 0.0 { -1 } else { 1 }).to_string();
        st.selected = tool_index(&name);
        lo.0.show(&name);
    }
    if keys.just_pressed(KeyCode::Semicolon) && TOOLS[st.selected].0 == "laser" {
        st.laser_target = (st.laser_target + 1) % LASER_TARGETS.len();
        // the Lab laser steps through its own quick targets (`DEFAULT_TARGETS`)
        let cur = fly_sim::laser::DEFAULT_TARGETS.iter().position(|t| t.eq_ignore_ascii_case(&st.laser.target_type));
        let next = fly_sim::laser::DEFAULT_TARGETS[cur.map_or(0, |i| (i + 1) % fly_sim::laser::DEFAULT_TARGETS.len())];
        st.laser.set_target(next);
    }
    if keys.just_pressed(KeyCode::Quote) && TOOLS[st.selected].0 == "laser" {
        st.laser_silence = !st.laser_silence;
        let mode = if st.laser.mode == "activate" { "silence" } else { "activate" };
        st.laser.set_mode(mode);
    }
}

// --- using them -------------------------------------------------------------------------------------------------------------------------
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn use_tools(
    mut fb: Fb,
    mut ff: FlyFx,
    mut commands: Commands,
    virt: Res<Time<Virtual>>,
    ptr: Pointer,
    arena: Res<Arena>,
    mut st: ResMut<ToolState>,
    mut gfx: Gfx,
    link: Res<SimLink>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<(&FlyPart, &Transform, Entity)>,
    mut vels: Query<(&FlyPart, &mut LinearVelocity), Without<Bomb>>,
    mut spawn_pred: MessageWriter<SpawnPredator>,
    mut fx: ResMut<Fx>,
    mut decoys: Query<(Entity, &mut Transform), (With<Decoy>, Without<FlyPart>)>,
) {
    let now = virt.elapsed_secs();
    let name = st.name();
    let (mouse, look, aim) = (&*ptr.mouse, &*ptr.look, &*ptr.aim);
    let active = look.captured && !ptr.photo.active;
    let (eye, dir) = (aim.eye, aim.dir);
    let tip = tool_tip(aim);
    let bounds = {
        let w = crate::world::spec(arena.name());
        (w.rx, w.rz)
    };
    let pressed = active && mouse.just_pressed(MouseButton::Left);
    let held = active && mouse.pressed(MouseButton::Left);
    if pressed {
        st.uses += 1;
        st.last_used = TOOLS[st.selected].0;
    }

    // the hand: grab on press, pin to the hold point while held, throw on release
    if name == "hand" {
        if pressed {
            // decoys can be grabbed too
            let mut grabbed_decoy = None;
            for (e, t) in &decoys {
                let rel = t.translation - eye;
                let along = rel.dot(dir);
                if along > 0.0 && along < GRAB_REACH && (rel - dir * along).length() < 0.35 {
                    grabbed_decoy = Some((e, along));
                }
            }
            let hit = nearest_to_ray(&parts, &flies, eye, dir, GRAB_REACH, 0.18);
            if let Some((e, along)) = grabbed_decoy.filter(|d| hit.is_none_or(|h| d.1 < h.2)) {
                st.grabbed_decoy = Some(e);
                st.hold_dist = along.clamp(0.6, 2.2);
            } else if let Some((fly_e, part, along, _)) = hit
                && let Ok((_, mut f)) = flies.get_mut(fly_e)
                && f.frozen_at.is_none()
            {
                f.grabbed = Some(part);
                f.last_hit = eye;
                st.hold_dist = along.clamp(0.6, 2.2);
                ff.hits.write(FlyHit { fly: fly_e, part, strength: 0.25 });
                st.holding = true;
            }
        }
        let hold = eye + dir * st.hold_dist;
        if held {
            for (_, mut f) in &mut flies {
                if f.grabbed.is_some() {
                    f.hold = hold;
                }
            }
            if let Some(d) = st.grabbed_decoy
                && let Ok((_, mut t)) = decoys.get_mut(d)
            {
                t.translation = hold;
            }
        } else {
            for (_, mut f) in &mut flies {
                if f.grabbed.is_some() && !f.wrapped {
                    f.grabbed = None; // thrown: the parts keep the velocity the pin gave them
                }
            }
            st.holding = false;
            st.grabbed_decoy = None;
        }
    } else if st.holding {
        for (_, mut f) in &mut flies {
            if !f.wrapped {
                f.grabbed = None;
            }
        }
        st.holding = false;
    }

    if pressed && name != "hand" {
        match name {
            "flick" => {
                st.flick_t = now;
                if let Some((fly_e, _, along, _)) = nearest_to_ray(&parts, &flies, eye, dir, 2.6, 0.4) {
                    let center = eye + dir * along;
                    if let Ok((_, mut f)) = flies.get_mut(fly_e) {
                        f.last_hit = eye;
                        for (p, t, _) in parts.iter().filter(|(p, _, _)| p.fly == fly_e) {
                            let dist = (t.translation - center).length();
                            if dist < 0.45 {
                                let s = 1.0 - dist / 0.45;
                                let push = (dir * 0.7 + Vec3::Y * 0.6) * (10.0 + 16.0 * s) * S * 60.0;
                                if let Ok((_, mut v)) = vels.get_mut(f.parts[p.part.index()]) {
                                    v.0 += push;
                                }
                                ff.hits.write(FlyHit { fly: fly_e, part: p.part, strength: 0.35 + 0.4 * s });
                            }
                        }
                        f.stun(now, 0.35);
                        ff.dmg.write(FlyDamage { fly: fly_e, amount: 4.0, source: "a flick" });
                        fb.popups.write(Popup::new(center + Vec3::Y * 0.25, "FLICK!", (255, 245, 235)));
                        fb.sfx.write(Sfx::new("flick"));
                    }
                }
            }
            "swatter" => {
                if now - st.swing_t > 0.35 {
                    st.swing_t = now;
                    st.swings.push(now);
                }
            }
            "bomb" => {
                let n_bombs = 0; // the cap of three is enforced by the query in `bombs` (spawned bombs count there)
                let _ = n_bombs;
                if now - st.throw_t > 0.4 {
                    st.throw_t = now;
                    commands.spawn((
                        Bomb { born: now },
                        RigidBody::Dynamic,
                        Collider::sphere(0.08),
                        Restitution::new(0.35),
                        Friction::new(0.6),
                        LinearVelocity(dir * 6.0 + Vec3::Y * 1.5),
                        Mesh3d(gfx.meshes.add(Sphere::new(0.08).mesh().uv(16, 10))),
                        MeshMaterial3d(gfx.materials.add(StandardMaterial {
                            base_color: Color::srgb(0.08, 0.08, 0.1),
                            emissive: LinearRgba::rgb(0.6, 0.1, 0.0),
                            perceptual_roughness: 0.4,
                            ..default()
                        })),
                        Transform::from_translation(tip),
                    ));
                    fb.sfx.write(Sfx::new("drop"));
                }
            }
            "zapper" if now >= st.zap_ready => {
                st.zap_ready = now + 0.3;
                fb.sfx.write(Sfx::new("zap"));
                let mut best: Option<(Entity, f32)> = None;
                for (e, f) in flies.iter() {
                    if f.gone() {
                        continue;
                    }
                    let Ok(thx) = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t, _)| t.translation) else { continue };
                    let to = thx - eye;
                    let dist = to.length();
                    let aligned = to.dot(dir) / dist.max(1e-6) >= 0.93;
                    let near = nearest_to_ray(&parts, &flies, eye, dir, REACH, 0.5).is_some_and(|h| h.0 == e);
                    if (near || (dist <= REACH && aligned)) && best.is_none_or(|b| dist < b.1) {
                        best = Some((e, dist));
                    }
                }
                match best {
                    None => fx.bolt(tip, eye + dir * 2.0, now),
                    Some((e, _)) => {
                        if let Ok((_, mut f)) = flies.get_mut(e) {
                            let thx = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t, _)| t.translation).unwrap_or(eye);
                            fx.bolt(tip, thx, now);
                            f.zap_until = now + 0.3;
                            f.char_ = (f.char_ + 0.06).min(1.0);
                            f.last_hit = eye;
                            let mut rng = Rng::new((now * 1000.0) as u64);
                            for (p, mut v) in &mut vels {
                                if p.fly == e {
                                    v.0 += Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 3.0 * S * 60.0;
                                }
                            }
                            fb.shake.until = now + 0.12;
                            fb.shake.strength = 0.015;
                            if !f.dead() {
                                // every touch neuron fires at once, plus a random 30% of the whole brain for 40 ms (`poke("all", 0)`)
                                for (region, side) in [
                                    ("head", None),
                                    ("body", None),
                                    ("legs", Some('L')),
                                    ("legs", Some('R')),
                                    ("wing", Some('L')),
                                    ("wing", Some('R')),
                                ] {
                                    link.0.send(Command::PokeFly { slot: f.slot, region: region.into(), side, strength: 1.0 });
                                }
                                link.0.send(Command::PokeFly { slot: f.slot, region: "all".into(), side: None, strength: 0.0 });
                                f.stun(now, 1.0);
                                ff.dmg.write(FlyDamage { fly: e, amount: 16.0, source: "the zapper" });
                                fb.popups.write(Popup::new(
                                    thx + Vec3::Y * 0.45,
                                    ["BZZZT!", "ZAP!", "KRZZT!"][rng.index(3)],
                                    (200, 235, 255),
                                ));
                            }
                        }
                    }
                }
            }
            "spider" | "frog" | "dragonfly" | "mantis" => {
                spawn_pred.write(SpawnPredator { kind: name_static(name), at: floor_point(aim, bounds) });
                fb.sfx.write(Sfx::new("drop"));
            }
            "sugar" | "fruit" | "alcohol" => {
                // `len(self.sugars3) < 3` / `len(self.alcohols3) < 3`
                let kind = usize::from(name == "alcohol");
                if st.treats_out[kind] < 3 && now - st.throw_t > 0.3 {
                    st.throw_t = now;
                    st.treats_out[kind] += 1;
                    let (r, col) = match name {
                        "sugar" => (0.035, Color::srgb(1.0, 0.96, 0.96)),
                        "fruit" => (0.055, Color::srgb(0.85, 0.2, 0.12)),
                        _ => (0.04, Color::srgb(0.9, 0.7, 0.2)),
                    };
                    commands.spawn((
                        Treat::new(name_static(name), r, now),
                        RigidBody::Dynamic,
                        Collider::sphere(r),
                        Restitution::new(0.1),
                        Friction::new(1.0),
                        LinearVelocity(dir * 4.2 + Vec3::Y * 1.2),
                        Mesh3d(gfx.meshes.add(Sphere::new(r).mesh().uv(12, 8))),
                        MeshMaterial3d(gfx.materials.add(StandardMaterial {
                            base_color: col,
                            perceptual_roughness: 0.5,
                            emissive: if name == "alcohol" { LinearRgba::rgb(0.5, 0.35, 0.05) } else { LinearRgba::BLACK },
                            ..default()
                        })),
                        Transform::from_translation(tip),
                    ));
                    fb.sfx.write(Sfx::new(if name == "alcohol" { "drop" } else { "pop" }));
                }
            }
            "cva" => {
                fb.sfx.write(Sfx::new("pop"));
                fb.popups.write(Popup::new(tip, "cVA PUFF", (255, 180, 100)));
                commands.spawn((Puff { until: now + 0.65 }, Transform::from_translation(tip)));
                for (_, f) in flies.iter() {
                    if let Ok((_, t, _)) = parts.get(f.parts[Part::Head.index()]) {
                        let dist = (t.translation - tip).length();
                        if dist < 2.5 {
                            link.0.send(Command::PokeFly {
                                slot: f.slot,
                                region: "scent:cva".into(),
                                side: None,
                                strength: 0.8 * (1.0 - dist / 2.5),
                            });
                        }
                    }
                }
            }
            "decoy" if now - st.throw_t > 0.3 => {
                st.throw_t = now;
                if decoys.iter().count() < 3 {
                    fb.sfx.write(Sfx::new("drop"));
                    fb.popups.write(Popup::new(tip, "DECOY FEMALE", (200, 180, 220)));
                    let body = gfx.materials.add(StandardMaterial {
                        base_color: Color::srgb(0.55, 0.42, 0.3),
                        perceptual_roughness: 0.5,
                        ..default()
                    });
                    commands
                        .spawn((
                            Decoy,
                            RigidBody::Dynamic,
                            Collider::cuboid(0.34, 0.18, 0.16),
                            LinearVelocity(dir * 4.2 + Vec3::Y * 1.2),
                            Transform::from_translation(tip)
                                .with_rotation(Quat::from_rotation_y(-(aim.dir.z.atan2(aim.dir.x)) + std::f32::consts::PI)),
                            Visibility::default(),
                        ))
                        .with_children(|c| {
                            c.spawn((
                                Mesh3d(gfx.meshes.add(Sphere::new(1.0).mesh().uv(14, 10))),
                                MeshMaterial3d(body.clone()),
                                Transform::from_scale(Vec3::new(0.16, 0.07, 0.07)),
                            ));
                            c.spawn((
                                Mesh3d(gfx.meshes.add(Sphere::new(1.0).mesh().uv(14, 10))),
                                MeshMaterial3d(body),
                                Transform::from_translation(Vec3::new(-0.2, 0.0, 0.0)).with_scale(Vec3::new(0.18, 0.08, 0.08)),
                            ));
                        });
                } else {
                    fb.popups.write(Popup::new(tip, "MAX 3 DECOYS", (200, 180, 220)));
                    fb.notes.push(now, "DECOY    limit reached (max 3)", Tag::GameRule);
                }
            }
            _ => {}
        }
    }
}

/// `&'static str` for a tool name (the name lives in the `TOOLS` table).
/// The neuron spec of laser target `i` (the puppeteer's palette in puppeteer mode).
pub fn laser_spec(i: usize, puppet: bool) -> &'static str {
    if puppet { crate::puppeteer::PALETTE[i % crate::puppeteer::PALETTE.len()].1 } else { LASER_TARGETS[i % LASER_TARGETS.len()].1 }
}

pub fn name_static(name: &str) -> &'static str {
    TOOLS.iter().find(|t| t.0 == name).map_or("hand", |t| t.0)
}

// --- the swatter, held jets, the laser ----------------------------------------------------------------------------------------------
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn continuous(
    mut fb: Fb,
    mut ff: FlyFx,
    virt: Res<Time<Virtual>>,
    mouse: Res<ButtonInput<MouseButton>>,
    look: Res<MouseLook>,
    aim: Res<Aim>,
    settings: Res<Settings>,
    mut st: ResMut<ToolState>,
    link: Res<SimLink>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<(&FlyPart, &Transform, Entity)>,
    mut vels: Query<(&FlyPart, &mut LinearVelocity), Without<Bomb>>,
    mut fx: ResMut<Fx>,
    mut gizmos: Gizmos,
) {
    let now = virt.elapsed_secs();
    let name = st.name();
    let held = look.captured && mouse.pressed(MouseButton::Left);
    let (eye, dir) = (aim.eye, aim.dir);
    let tip = tool_tip(&aim);

    // the swatter lands 0.12 s after the swing starts
    let mut landed = Vec::new();
    st.swings.retain(|&t| {
        if now - t > 0.12 {
            landed.push(t);
        }
        now - t < 0.12
    });
    for _ in landed {
        let center = eye + dir * 1.05;
        fb.shake.until = now + 0.15;
        fb.shake.strength = 0.02;
        fb.sfx.write(Sfx::new("whack"));
        let dh = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
        let snapshot: Vec<(Entity, bool, bool)> =
            flies.iter().map(|(e, f)| (e, f.gone(), f.frozen_at.is_some() && f.shattered_at.is_none())).collect();
        for (fly_e, gone, frozen) in snapshot {
            let near: Vec<Part> = parts
                .iter()
                .filter(|(p, t, _)| p.fly == fly_e && (t.translation - center).length() < 0.8)
                .map(|(p, _, _)| p.part)
                .collect();
            if near.is_empty() || gone {
                continue;
            }
            if frozen {
                shatter_fly(&mut flies, &parts, fly_e, now, &mut fx, &mut fb.popups, &mut fb.sfx);
                continue;
            }
            if let Ok((_, mut f)) = flies.get_mut(fly_e) {
                f.last_hit = eye;
                for part in &near {
                    let Ok((_, t, _)) = parts.get(f.parts[part.index()]) else { continue };
                    let push = (t.translation - center) * 0.15 * 60.0 * 0.25 + dh * 8.0 * S * 60.0 + Vec3::new(0.0, -32.0 * S * 60.0, 0.0);
                    if let Ok((_, mut v)) = vels.get_mut(f.parts[part.index()]) {
                        v.0 += push;
                    }
                    ff.hits.write(FlyHit { fly: fly_e, part: *part, strength: 1.0 });
                }
                f.stun(now, 1.8);
                ff.dmg.write(FlyDamage { fly: fly_e, amount: 14.0, source: "the swatter" });
                if let Ok((_, t, _)) = parts.get(f.parts[Part::Thorax.index()]) {
                    fb.popups.write(Popup::new(t.translation + Vec3::Y * 0.45, "SWAT!", (255, 230, 120)));
                    fx.puff(Vec3::new(t.translation.x, 0.02, t.translation.z), 10, 4.0, now);
                }
            }
        }
    }

    // jets: the torch burns, the cleaner dissolves, the freezer freezes (cone from the nozzle)
    if held && matches!(name, "torch" | "cleaner" | "freeze") {
        let kind = match name {
            "torch" => Kind::Flame,
            "cleaner" => Kind::Cleaner,
            _ => Kind::Freeze,
        };
        fx.jet(tip, dir, kind, now);
        let (range, cone) = if name == "torch" { (1.9, 0.85) } else { (2.0, 0.8) };
        for (e, mut f) in &mut flies {
            if f.gone() {
                continue;
            }
            let inside: Vec<Part> = parts
                .iter()
                .filter(|(p, _, _)| p.fly == e)
                .filter(|(_, t, _)| {
                    let rel = t.translation - tip;
                    let d = rel.length();
                    d < range && rel.dot(dir) / d.max(1e-6) > cone
                })
                .map(|(p, _, _)| p.part)
                .collect();
            let dead = f.dead();
            match name {
                "torch" => {
                    if !inside.is_empty() {
                        f.burn_until = now + 0.9;
                        for part in &inside {
                            if let Ok((_, mut v)) = vels.get_mut(f.parts[part.index()]) {
                                v.0 += (dir * 0.5 + Vec3::Y * 0.2) * S * 60.0 * 0.5;
                            }
                        }
                    }
                    if now >= f.burn_until {
                        continue;
                    }
                    f.char_ = (f.char_ + 0.004).min(1.0);
                    f.hurt = f.hurt.max(0.5);
                    f.last_hit = eye;
                    if dead {
                        continue;
                    }
                    for (region, side) in [
                        ("head", None),
                        ("body", None),
                        ("legs", Some('L')),
                        ("legs", Some('R')),
                        ("wing", Some('L')),
                        ("wing", Some('R')),
                        ("heat", None),
                    ] {
                        link.0.send(Command::PokeFly { slot: f.slot, region: region.into(), side, strength: 1.0 });
                    }
                    ff.dmg.write(FlyDamage { fly: e, amount: 0.35, source: "the blowtorch" });
                    if f.rng.chance(0.02) {
                        let words = ["SIZZLE!", "TSSSS!", "HOT HOT!"];
                        if let Ok((_, t, _)) = parts.get(f.parts[Part::Head.index()]) {
                            fb.popups.write(Popup::new(t.translation + Vec3::Y * 0.4, words[f.rng.index(3)], (255, 150, 60)));
                        }
                    }
                }
                _ => {
                    if f.dissolved_at.is_some() || f.frozen_at.is_some() || inside.is_empty() {
                        continue;
                    }
                    f.last_hit = eye;
                    if name == "cleaner" {
                        f.soak = (f.soak + 0.04).min(1.0);
                    } else {
                        f.frost = (f.frost + 0.006).min(if settings.immortal { 0.9 } else { 1.0 });
                    }
                    if dead {
                        continue;
                    }
                    if name == "cleaner" {
                        link.0.send(Command::PokeFly { slot: f.slot, region: "smell".into(), side: None, strength: 1.0 });
                        link.0.send(Command::PokeFly { slot: f.slot, region: "taste".into(), side: None, strength: 0.8 });
                        ff.dmg.write(FlyDamage { fly: e, amount: 0.12, source: "brake cleaner" });
                    } else {
                        link.0.send(Command::PokeFly { slot: f.slot, region: "cold".into(), side: None, strength: 1.0 });
                        for (region, side) in [
                            ("head", None),
                            ("body", None),
                            ("legs", Some('L')),
                            ("legs", Some('R')),
                            ("wing", Some('L')),
                            ("wing", Some('R')),
                        ] {
                            link.0.send(Command::PokeFly { slot: f.slot, region: region.into(), side, strength: 0.35 });
                        }
                        ff.dmg.write(FlyDamage { fly: e, amount: 0.1, source: "freezing" });
                    }
                    if f.rng.chance(0.02)
                        && let Ok((_, t, _)) = parts.get(f.parts[Part::Head.index()])
                    {
                        let (words, col) = if name == "cleaner" {
                            (["FSSSSH!", "MELTING!", "IT BURNS!"], (170, 230, 255))
                        } else {
                            (["SO COLD!", "BRRRR!", "ICING!"], (190, 230, 255))
                        };
                        fb.popups.write(Popup::new(t.translation + Vec3::Y * 0.4, words[f.rng.index(3)], col));
                    }
                }
            }
        }
    }

    // the laser: while it is on (held, or a timed pulse) and the beam is on a fly, the chosen cell type is driven (or silenced)
    if name == "laser" {
        let hit = nearest_to_ray(&parts, &flies, eye, dir, 15.0, 0.25);
        if st.puppet {
            let want = if held && hit.is_some() { Some((st.laser_target, st.laser_silence)) } else { None };
            if want != st.laser_on {
                if let Some((t, _)) = st.laser_on {
                    link.0.send(Command::DriveSpec { spec: laser_spec(t, true).into(), amp: 0.0 });
                }
                if let Some((t, sil)) = want {
                    link.0.send(Command::DriveSpec { spec: laser_spec(t, true).into(), amp: if sil { LASER_SILENCE } else { LASER_STIM } });
                }
                st.laser_on = want;
            }
            if held {
                let end = hit.map_or(eye + dir * 15.0, |h| eye + dir * h.2);
                let c = if st.laser_silence { Color::srgb(0.3, 0.8, 1.0) } else { Color::srgb(1.0, 0.3, 0.2) };
                gizmos.line(tip, end, c);
                gizmos.sphere(Isometry3d::from_translation(end), 0.025, c);
            }
        } else {
            let t = f64::from(now);
            if held && !st.laser_was_held {
                st.laser.trigger_press(t);
            } else if !held && st.laser_was_held {
                st.laser.trigger_release();
            }
            st.laser_was_held = held;
            let on = st.laser.is_active(t) && hit.is_some();
            // the quick targets and typed ones resolve the way Python's `_scan_rows` does, once per target text
            if st.laser_rows.0 != st.laser.target_type {
                let shared = link.0.shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
                if let Some(sh) = shared {
                    st.laser_rows = (st.laser.target_type.clone(), fly_sim::laser::resolve_rows(&sh.labels, &st.laser.target_type));
                }
            }
            // the drive current is the surgery scale: Python's 0.48 / -2.40 is after the simulator's external gain of 4
            let amp = if on { (st.laser.current_value() / 4.0) as f32 } else { 0.0 };
            let want = (if on { st.laser_rows.1.clone() } else { Vec::new() }, amp);
            if want != st.laser_applied {
                if !st.laser_applied.0.is_empty() {
                    link.0.send(Command::DriveRows { rows: st.laser_applied.0.clone(), amp: 0.0 });
                }
                if !want.0.is_empty() {
                    link.0.send(Command::DriveRows { rows: want.0.clone(), amp });
                }
                st.laser_applied = want;
            }
            if st.laser.is_active(t) {
                let end = hit.map_or(eye + dir * 15.0, |h| eye + dir * h.2);
                let c = if st.laser.mode == "silence" { Color::srgb(0.3, 0.8, 1.0) } else { Color::srgb(1.0, 0.3, 0.2) };
                gizmos.line(tip, end, c);
                gizmos.sphere(Isometry3d::from_translation(end), 0.025, c);
            }
        }
    } else {
        if let Some((t, _)) = st.laser_on.take() {
            link.0.send(Command::DriveSpec { spec: laser_spec(t, st.puppet).into(), amp: 0.0 });
        }
        if !st.laser_applied.0.is_empty() {
            link.0.send(Command::DriveRows { rows: std::mem::take(&mut st.laser_applied.0), amp: 0.0 });
            st.laser_applied.1 = 0.0;
        }
        st.laser_was_held = false;
        st.laser.trigger_release();
    }
}

pub fn shatter_fly(
    flies: &mut Query<(Entity, &mut Fly)>,
    parts: &Query<(&FlyPart, &Transform, Entity)>,
    e: Entity,
    now: f32,
    fx: &mut Fx,
    popups: &mut MessageWriter<Popup>,
    sfx: &mut MessageWriter<Sfx>,
) {
    let Ok((_, mut f)) = flies.get_mut(e) else { return };
    f.shattered_at = Some(now);
    let at = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t, _)| t.translation).unwrap_or(Vec3::ZERO);
    fx.shatter(at, now);
    popups.write(Popup::new(at + Vec3::Y * 0.45, "SHATTERED!", (190, 230, 255)));
    sfx.write(Sfx::new("shatter"));
}

// --- bombs ---------------------------------------------------------------------------------------------------------------------------
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn bombs(
    mut fb: Fb,
    mut ff: FlyFx,
    mut commands: Commands,
    virt: Res<Time<Virtual>>,
    bombs: Query<(Entity, &Bomb, &Transform)>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<(&FlyPart, &Transform, Entity)>,
    mut vels: Query<(&FlyPart, &mut LinearVelocity), Without<Bomb>>,
    mut fx: ResMut<Fx>,
    mut threats: ResMut<Threats>,
) {
    let now = virt.elapsed_secs();
    for (e, b, t) in &bombs {
        threats.push(("bomb", e), t.translation, 0.08);
        if now - b.born < 1.8 {
            continue;
        }
        let pos = t.translation;
        commands.entity(e).despawn();
        fb.shake.until = now + 0.4;
        fb.shake.strength = 0.05;
        fb.sfx.write(Sfx::new("boom"));
        fx.explosion(pos, now);
        let radius = 330.0 * S;
        let snapshot: Vec<(Entity, bool, bool, [Entity; 3])> =
            flies.iter().map(|(e, f)| (e, f.gone(), f.frozen_at.is_some() && f.shattered_at.is_none(), f.parts)).collect();
        for (fly_e, gone, frozen, fparts) in snapshot {
            if gone {
                continue;
            }
            let mut worst = 0.0f32;
            let mut touched = Vec::new();
            for (p, pt, _) in parts.iter().filter(|(p, _, _)| p.fly == fly_e) {
                let d = (pt.translation - pos).length();
                if d < radius {
                    let force = 48.0 * (1.0 - d / radius).powf(1.3);
                    let dirv = (pt.translation - pos) / d.max(1e-6) + Vec3::new(0.0, 0.8, 0.0);
                    if let Ok((_, mut v)) = vels.get_mut(fparts[p.part.index()]) {
                        v.0 += dirv * force * S * 60.0;
                    }
                    if force > 4.0 {
                        ff.hits.write(FlyHit { fly: fly_e, part: p.part, strength: force / 40.0 });
                        worst = worst.max(force);
                    }
                }
                touched.push(p.part);
            }
            if frozen && worst > 0.0 {
                shatter_fly(&mut flies, &parts, fly_e, now, &mut fx, &mut fb.popups, &mut fb.sfx);
            } else if worst > 0.0
                && let Ok((_, mut f)) = flies.get_mut(fly_e)
            {
                f.last_hit = pos;
                f.stun(now, 2.4);
                ff.dmg.write(FlyDamage { fly: fly_e, amount: 32.0 * worst / 48.0, source: "a bomb" });
            }
        }
        fb.popups.write(Popup::new(pos + Vec3::Y * 0.5, "KABOOM!", (255, 160, 60)));
    }
}

// --- treats: sugar, fruit, alcohol ---------------------------------------------------------------------------------------------------
#[allow(clippy::type_complexity)]
fn treats(
    mut fb: Fb,
    mut commands: Commands,
    virt: Res<Time<Virtual>>,
    mut treats: Query<(Entity, &mut Treat, &Transform, &LinearVelocity)>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<(&FlyPart, &Transform), Without<Treat>>,
    link: Res<SimLink>,
    mut scents: ResMut<Scents>,
    mut pet: ResMut<crate::pet_play::PetRes>,
    mut st: ResMut<ToolState>,
) {
    let now = virt.elapsed_secs();
    let count = |alcohol: bool| treats.iter().filter(|t| (t.1.kind == "alcohol") == alcohol).count();
    let out = [count(false), count(true)];
    if st.treats_out != out {
        st.treats_out = out;
    }
    for (e, mut t, tf, v) in &mut treats {
        // `_sugar3d`: a treat stays where it first touches the floor. Here it is a physics ball, which would roll on for seconds and
        // keep the fly waiting; so it is pinned as soon as it reaches the floor (or has come to rest on something higher up).
        if !t.landed && treat_landed(tf.translation.y, t.radius, now - t.born, v.0.length()) {
            t.landed = true;
            commands.entity(e).insert((RigidBody::Static, LinearVelocity::ZERO, AngularVelocity::ZERO));
        }
        scents.0.push(ScentSource {
            pos: tf.translation,
            scent: if t.kind == "alcohol" { "alcohol" } else { "sugar" },
            range: 2.4,
            strength: if t.kind == "alcohol" { 0.35 } else { 0.3 },
            flat: true,
        });
    }
    for (_, mut f) in &mut flies {
        if f.dead() || f.wrapped || f.frozen_at.is_some() || f.grabbed.is_some() {
            continue;
        }
        let Ok(head) = parts.get(f.parts[Part::Head.index()]).map(|(_, t)| t.translation) else { continue };
        let Ok(thx) = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t)| t.translation) else { continue };
        let nearest = treats.iter().min_by(|a, b| {
            Vec2::new(a.2.translation.x - thx.x, a.2.translation.z - thx.z)
                .length()
                .total_cmp(&Vec2::new(b.2.translation.x - thx.x, b.2.translation.z - thx.z).length())
        });
        let Some((te, tr, tt, _)) = nearest else { continue };
        let landed = tr.landed;
        let dxz = Vec2::new(tt.translation.x - head.x, tt.translation.z - head.z);
        let on_floor = thx.y < STAND3 + 0.2 && now >= f.react.escape_until;
        if dxz.length() > 0.2 {
            if on_floor && now >= f.react.stun_until && landed {
                f.yaw_target = dxz.y.atan2(dxz.x);
                f.react.walk_until = now + 0.2;
                f.run = false;
            }
            continue;
        }
        if !on_floor || !landed {
            continue;
        }
        let Ok((_, mut treat, _, _)) = treats.get_mut(te) else { continue };
        if now >= f.react.eating_until {
            let words = if treat.kind == "alcohol" { ["SIP...", "GLUG!", "*HIC*"] } else { ["YUM!", "SWEET!", "NOM NOM"] };
            fb.popups.write(Popup::new(
                head + Vec3::Y * 0.4,
                words[f.rng.index(3)],
                if treat.kind == "alcohol" { (255, 140, 210) } else { (255, 160, 190) },
            ));
            fb.sfx.write(Sfx::new("yum"));
            fb.notes.push(
                now,
                if treat.kind == "alcohol" {
                    "ALCOHOL  drinking: sweet + PAM reward [GAME RULE: inebriation]"
                } else {
                    "EATING   sugar: taste + PAM reward"
                },
                if treat.kind == "alcohol" { Tag::GameRule } else { Tag::Connectome },
            );
        }
        f.react.eating_until = now + 0.4;
        treat.left -= 1.0 / 240.0;
        // pet mode: hunger scales the sugar taste and the PAM reward drive (`core/pet.py` couplings)
        let sugar = treat.kind != "alcohol";
        let (tg, rg) = if sugar { (pet.sugar_gain(), pet.pam_gain()) } else { (1.0, 1.0) };
        link.0.send(Command::PokeFly { slot: f.slot, region: "taste".into(), side: None, strength: 0.5 * tg });
        // the sugar-pathway taste neurons (`assays.py`), 60 % of them at a time: sugar 0.5, alcohol 0.6
        link.0.send(Command::PokeFlyRecruit {
            slot: f.slot,
            region: "sweet".into(),
            strength: if sugar { 0.5 * tg } else { 0.6 },
            recruit: 0.6,
        });
        link.0.send(Command::PokeFly { slot: f.slot, region: "reward".into(), side: None, strength: if sugar { 0.4 } else { 0.6 } * rg });
        if sugar {
            pet.feed_frame();
        }
        if treat.kind == "alcohol" {
            f.inebriation = (f.inebriation + 0.008).min(1.0);
        } else {
            f.health = (f.health + 0.15).min(MAX_HEALTH);
        }
        if treat.left <= 0.0 {
            commands.entity(te).despawn();
        }
    }
}

// --- decoys, cVA puffs, the player's body --------------------------------------------------------------------------------------------
#[allow(clippy::type_complexity)]
fn decoys_and_puffs(
    mut fb: Fb,
    mut commands: Commands,
    virt: Res<Time<Virtual>>,
    puffs: Query<(Entity, &Puff)>,
    decoys: Query<&Transform, (With<Decoy>, Without<FlyPart>)>,
    mut flies: Query<&mut Fly>,
    parts: Query<&Transform, With<FlyPart>>,
    link: Res<SimLink>,
) {
    let now = virt.elapsed_secs();
    for (e, p) in &puffs {
        if now > p.until {
            commands.entity(e).despawn();
        }
    }
    for mut f in &mut flies {
        if f.dead() || f.wrapped || f.frozen_at.is_some() || f.grabbed.is_some() {
            continue;
        }
        let Ok(head) = parts.get(f.parts[Part::Head.index()]).map(|t| t.translation) else { continue };
        let near = decoys.iter().min_by(|a, b| (a.translation - head).length().total_cmp(&(b.translation - head).length()));
        if let Some(d) = near
            && (d.translation - head).length() < 0.4
        {
            link.0.send(Command::PokeFly { slot: f.slot, region: "pheromone:foreleg".into(), side: None, strength: 0.6 });
            if now >= f.react.song_ready - 100.0 + 0.0 && f.rng.chance(0.03) {
                fb.notes.push(now, "COURTSHIP   foreleg contact (rule): drives the putative ppk23/ppk25 GRNs", Tag::GameRule);
                fb.popups.write(Popup::new(head + Vec3::Y * 0.4, "COURTSHIP", (255, 180, 220)));
            }
        }
    }
}

/// Your legs are a 0.3 m cylinder: walking into a fly shoves it (the collider does that), and walking into it fast is a kick.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn kick(
    mut fb: Fb,
    mut ff: FlyFx,
    virt: Res<Time<Virtual>>,
    player: Single<(&Player, &Transform)>,
    mut st: ResMut<ToolState>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<(&FlyPart, &Transform)>,
    mut vels: Query<(&FlyPart, &mut LinearVelocity), Without<Bomb>>,
    mut threats: ResMut<Threats>,
    aim: Res<Aim>,
    mut rng: Local<Option<Rng>>,
) {
    let now = virt.elapsed_secs();
    let (p, tf) = *player;
    threats.push("player", Vec3::new(tf.translation.x, aim.eye.y - 0.35, tf.translation.z), 0.28);
    let tool = st.name();
    if !matches!(tool, "hand" | "sugar" | "fruit" | "alcohol") {
        threats.push("tool", tool_tip(&aim), tool_size(tool));
    }
    if now - st.swing_t < 0.14 {
        let a = aim.eye + aim.right * 0.28 + aim.up * 0.35 - aim.dir * 0.35 * 0.0;
        let b = aim.eye + aim.dir * 1.05;
        threats.push("swing", a + (b - a) * ((now - st.swing_t) / 0.14), 0.2);
    }
    let rng = rng.get_or_insert_with(|| Rng::new(5));
    let speed = p.vel.length();
    for (e, mut f) in &mut flies {
        if f.frozen_at.is_some() || f.gone() {
            continue;
        }
        let mut inside = Vec::new();
        for (fp, t) in parts.iter().filter(|(fp, _)| fp.fly == e) {
            let rel = Vec2::new(t.translation.x - tf.translation.x, t.translation.z - tf.translation.z);
            if rel.length() < 0.3 + 0.12 && t.translation.y < 0.9 {
                inside.push(fp.part);
            }
        }
        if inside.is_empty() {
            continue;
        }
        if speed > 1.2 && now - st.kick_t > 0.4 {
            st.kick_t = now;
            let push = Vec3::new(p.vel.x, 0.0, p.vel.y) * 1.6 + Vec3::new(0.0, 0.72 + 0.36 * speed, 0.0);
            for (fp, mut v) in &mut vels {
                if fp.fly == e {
                    v.0 += push;
                }
            }
            for part in inside {
                ff.hits.write(FlyHit { fly: e, part, strength: (speed / 5.0).min(1.0) });
            }
            f.last_hit = aim.eye;
            if speed > 2.5 {
                f.stun(now, 0.6);
                ff.dmg.write(FlyDamage { fly: e, amount: 3.0 + 2.0 * speed, source: "a kick" });
                if let Ok((_, t)) = parts.get(f.parts[Part::Thorax.index()]) {
                    fb.popups.write(Popup::new(t.translation + Vec3::Y * 0.4, ["KICK!", "BOOT!", "PUNT!"][rng.index(3)], (255, 235, 150)));
                }
                fb.sfx.write(Sfx::vol("whack", 0.6));
            }
        }
    }
}

/// Burns, melting, freezing: the slow effects (`_effects_one`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn slow_effects(
    mut fb: Fb,
    mut ff: FlyFx,
    virt: Res<Time<Virtual>>,
    mouse: Res<ButtonInput<MouseButton>>,
    st: Res<ToolState>,
    settings: Res<Settings>,
    link: Res<SimLink>,
    mut flies: Query<(Entity, &mut Fly)>,
    parts: Query<(&FlyPart, &Transform)>,
    mut fx: ResMut<Fx>,
    snap: Res<LatestSnapshot>,
) {
    let now = virt.elapsed_secs();
    let _ = snap;
    for (e, mut f) in &mut flies {
        let thx = parts.get(f.parts[Part::Thorax.index()]).map(|(_, t)| t.translation).unwrap_or(Vec3::ZERO);
        if f.inebriation > 0.0 {
            f.inebriation = (f.inebriation - 0.00035).max(0.0); // ethanol metabolism (~45 s)
        }
        if settings.immortal {
            f.melt = f.melt.min(0.85);
            if f.soak < 0.05 {
                f.melt = (f.melt - 0.002).max(0.0);
            }
            f.venom = (f.venom - 0.002).max(0.0);
        }
        if f.dissolved_at.is_none() && f.soak > 0.02 {
            f.melt = (f.melt + 0.0045 * f.soak).min(if settings.immortal { 0.85 } else { 1.0 });
            f.soak *= if settings.immortal { 0.985 } else { 0.997 };
            if !f.dead() {
                ff.dmg.write(FlyDamage { fly: e, amount: 0.25 * f.soak, source: "brake cleaner" });
                let s = 0.5 * f.soak;
                for (region, side) in
                    [("head", None), ("body", None), ("legs", Some('L')), ("legs", Some('R')), ("wing", Some('L')), ("wing", Some('R'))]
                {
                    link.0.send(Command::PokeFly { slot: f.slot, region: region.into(), side, strength: s });
                }
            }
        }
        if f.melt >= 1.0 && f.dissolved_at.is_none() {
            f.dissolved_at = Some(now);
            fx.puff(Vec3::new(thx.x, 0.05, thx.z), 14, 3.0, now);
            fb.popups.write(Popup::new(thx + Vec3::Y * 0.4, "DISSOLVED", (170, 230, 255)));
            fb.sfx.write(Sfx::new("squish"));
            if !f.dead() {
                ff.dmg.write(FlyDamage { fly: e, amount: MAX_HEALTH, source: "brake cleaner" });
            }
        }
        if f.frozen_at.is_none() && f.frost > 0.0 {
            if !(mouse.pressed(MouseButton::Left) && st.name() == "freeze") {
                f.frost = (f.frost - 0.0015).max(0.0);
            }
            if f.frost >= 1.0 {
                f.frozen_at = Some(now);
                fb.popups.write(Popup::new(thx + Vec3::Y * 0.45, "FROZEN SOLID", (190, 230, 255)));
                if !f.dead() {
                    ff.dmg.write(FlyDamage { fly: e, amount: MAX_HEALTH, source: "freezing" });
                }
            }
        }
        if !f.dead() {
            let sedation = (0.25 * f.melt.powf(2.5)).max(0.2 * f.frost.powi(2)).max(0.3 * f.venom.powf(1.5));
            link.0.send(Command::Sedation(sedation));
        }
    }
}

pub struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ToolState>()
            .init_resource::<ActiveLoadout>()
            .add_message::<SpawnPredator>()
            .add_systems(Update, (sync_loadout, select).chain())
            .add_systems(FixedUpdate, (use_tools, continuous, bombs, treats, decoys_and_puffs, kick, slow_effects).chain());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_treat_lands_where_it_touches_the_floor() {
        // thrown at 4.2 m/s: still fast when it reaches the floor, and it must count as landed there (it used to roll on and the fly waited)
        assert!(treat_landed(0.04, 0.035, 0.4, 4.0));
        assert!(!treat_landed(0.9, 0.035, 0.1, 4.0));
        // at rest on a table
        assert!(treat_landed(0.8, 0.035, 1.0, 0.05));
        // the top of its arc is not a landing
        assert!(!treat_landed(1.6, 0.035, 0.2, 0.1));
    }
}
