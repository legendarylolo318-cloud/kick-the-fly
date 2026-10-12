//! Contraptions: a Rube Goldberg machine of real rigid bodies that runs against the fly. Python: `game/contraption.py` (a deterministic 2D
//! bench with a build format, validator and share codes) and `lab/contraption_challenge.py` (firing the game's own tools from its events).
//!
//! Here the machine is built in the 3D room out of avian bodies and the physics engine does the work: a marble rolls down a ramp, tips a
//! row of dominoes, and the last domino lands on a button plate; the button then fires a payload through the game's own tool code (a bomb
//! dropped over the fly, a swat, a gust of wind, a sugar cube), so what reaches the fly's neurons is exactly what the tool reaches. C
//! builds and runs one in front of you, Shift+C picks the next payload.
//!
//! Tags: what a payload drives in the fly is CONNECTOME; the machine, its parts and its physics are GAME RULES; what the fly does when it
//! runs is a MODEL PREDICTION. NOT ported: the 2D bench editor, the 24-part build format, the validator, saved slots and share codes.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::app::SimLink;
use crate::feedback::{Notes, Popup, Sfx, Tag};
use crate::fly::{Fly, FlyDamage, FlyHit, FlyPart, Part};
use crate::player::{Aim, MouseLook, Player};
use crate::sim_thread::Command;
use crate::tools::Bomb;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Payload {
    Bomb,
    Swat,
    Wind,
    Sugar,
}

impl Payload {
    pub fn next(self) -> Self {
        match self {
            Payload::Bomb => Payload::Swat,
            Payload::Swat => Payload::Wind,
            Payload::Wind => Payload::Sugar,
            Payload::Sugar => Payload::Bomb,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Payload::Bomb => "a bomb dropped over the fly",
            Payload::Swat => "a swat on the fly",
            Payload::Wind => "a gust of wind (Johnston's organ)",
            Payload::Sugar => "a sugar cube",
        }
    }
}

#[derive(Component)]
struct MachinePart;
#[derive(Component)]
struct Marble;
#[derive(Component)]
struct ButtonPlate;

#[derive(Resource)]
pub struct Contraption {
    pub payload: Payload,
    pub running: bool,
    pub fired: bool,
    started: f32,
    wind_until: f32,
}

impl Default for Contraption {
    fn default() -> Self {
        Self { payload: Payload::Bomb, running: false, fired: false, started: 0.0, wind_until: 0.0 }
    }
}

/// How the machine is laid out, in a frame whose x axis runs along the ramp and the dominoes (`dir`), starting at `origin` on the floor.
pub struct Layout {
    pub ramp_top: Vec3,
    pub ramp_len: f32,
    pub ramp_angle: f32,
    pub dominoes: Vec<Vec3>,
    pub button: Vec3,
}

pub fn layout(origin: Vec3, dir: Vec3) -> Layout {
    let ramp_len = 1.2;
    let angle = 0.35f32;
    let drop = ramp_len * angle.sin();
    let ramp_top = origin + Vec3::Y * (drop + 0.05);
    let foot = origin + dir * (ramp_len * angle.cos()) + Vec3::Y * 0.05;
    let dominoes: Vec<Vec3> = (0..7).map(|i| foot + dir * (0.35 + 0.16 * i as f32) + Vec3::Y * 0.12).collect();
    let button = foot + dir * (0.35 + 0.16 * 7.0 + 0.25);
    Layout { ramp_top, ramp_len, ramp_angle: angle, dominoes, button }
}

fn build(commands: &mut Commands, meshes: &mut Assets<Mesh>, materials: &mut Assets<StandardMaterial>, origin: Vec3, dir: Vec3) {
    let l = layout(origin, dir);
    let yaw = Quat::from_rotation_y(-dir.z.atan2(dir.x));
    let wood = materials.add(StandardMaterial { base_color: Color::srgb(0.7, 0.5, 0.3), perceptual_roughness: 0.8, ..default() });
    let ivory = materials.add(StandardMaterial { base_color: Color::srgb(0.95, 0.93, 0.85), perceptual_roughness: 0.5, ..default() });
    let steel = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.82, 0.88),
        metallic: 0.8,
        perceptual_roughness: 0.25,
        ..default()
    });
    let red = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.15, 0.1),
        emissive: LinearRgba::rgb(0.6, 0.05, 0.02),
        ..default()
    });
    // the ramp: a plank tilted down along `dir`
    let mid = l.ramp_top + dir * (l.ramp_len * l.ramp_angle.cos() / 2.0) - Vec3::Y * (l.ramp_len * l.ramp_angle.sin() / 2.0);
    commands.spawn((
        MachinePart,
        RigidBody::Static,
        Collider::cuboid(l.ramp_len, 0.03, 0.3),
        Friction::new(0.2),
        Mesh3d(meshes.add(Cuboid::new(l.ramp_len, 0.03, 0.3))),
        MeshMaterial3d(wood.clone()),
        Transform::from_translation(mid).with_rotation(yaw * Quat::from_rotation_z(-l.ramp_angle)),
    ));
    // the marble waits at the top of the ramp and is released a moment after the run starts
    commands.spawn((
        MachinePart,
        Marble,
        RigidBody::Dynamic,
        Collider::sphere(0.045),
        Restitution::new(0.2),
        Friction::new(0.3),
        Mass(0.4),
        Mesh3d(meshes.add(Sphere::new(0.045).mesh().uv(16, 10))),
        MeshMaterial3d(steel),
        Transform::from_translation(l.ramp_top + Vec3::Y * 0.05 + dir * 0.02),
    ));
    // the dominoes
    for p in &l.dominoes {
        commands.spawn((
            MachinePart,
            RigidBody::Dynamic,
            Collider::cuboid(0.04, 0.24, 0.12),
            Mass(0.12),
            Friction::new(0.6),
            Mesh3d(meshes.add(Cuboid::new(0.04, 0.24, 0.12))),
            MeshMaterial3d(ivory.clone()),
            Transform::from_translation(*p).with_rotation(yaw),
        ));
    }
    // the button plate
    commands.spawn((
        MachinePart,
        ButtonPlate,
        RigidBody::Static,
        Collider::cuboid(0.3, 0.03, 0.3),
        Mesh3d(meshes.add(Cuboid::new(0.3, 0.03, 0.3))),
        MeshMaterial3d(red),
        Transform::from_translation(l.button + Vec3::Y * 0.015),
    ));
}

#[allow(clippy::too_many_arguments)]
fn control(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    look: Res<MouseLook>,
    virt: Res<Time<Virtual>>,
    aim: Res<Aim>,
    player: Single<&Transform, With<Player>>,
    mut c: ResMut<Contraption>,
    old: Query<Entity, With<MachinePart>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut notes: ResMut<Notes>,
) {
    if !look.captured || !keys.just_pressed(KeyCode::KeyC) {
        return;
    }
    let now = virt.elapsed_secs();
    if keys.pressed(KeyCode::ShiftLeft) {
        c.payload = c.payload.next();
        notes.push(now, format!("CONTRAPTION payload: {}", c.payload.label()), Tag::GameRule);
        return;
    }
    for e in &old {
        commands.entity(e).despawn();
    }
    let dir = Vec3::new(aim.dir.x, 0.0, aim.dir.z).normalize_or_zero();
    let dir = if dir == Vec3::ZERO { Vec3::NEG_Z } else { dir };
    let origin = Vec3::new(player.translation.x, 0.0, player.translation.z) + dir * 1.0 - Vec3::new(-dir.z, 0.0, dir.x) * 0.0;
    build(&mut commands, &mut meshes, &mut materials, origin, dir);
    c.running = true;
    c.fired = false;
    c.started = now;
    notes.push(now, format!("CONTRAPTION marble, ramp, 7 dominoes, a button -> {}", c.payload.label()), Tag::GameRule);
}

/// Let the marble go after half a second (it is held in place until then), and fire the payload when something lands on the button.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run(
    mut commands: Commands,
    virt: Res<Time<Virtual>>,
    mut c: ResMut<Contraption>,
    mut marbles: Query<(&mut Transform, &mut LinearVelocity), (With<Marble>, Without<ButtonPlate>, Without<FlyPart>)>,
    dominoes: Query<&Transform, (With<MachinePart>, Without<Marble>, Without<ButtonPlate>)>,
    button: Query<&Transform, With<ButtonPlate>>,
    mut flies: Query<(Entity, &mut Fly)>,
    slots: Query<&crate::fly::Slot>,
    parts: Query<&Transform, (With<FlyPart>, Without<MachinePart>)>,
    mut fly_vel: Query<(&FlyPart, &mut LinearVelocity), Without<MachinePart>>,
    link: Res<SimLink>,
    mut fb: crate::tools::Fb,
    mut ff: crate::tools::FlyFx,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !c.running {
        return;
    }
    let now = virt.elapsed_secs();
    if now - c.started < 0.5 {
        for (_, mut v) in &mut marbles {
            v.0 = Vec3::ZERO; // held on the ramp until the run starts
        }
        return;
    }
    let Some(b) = button.iter().next() else { return };
    let pressed = marbles.iter().any(|(t, _)| (t.translation - b.translation).length() < 0.35)
        || dominoes.iter().any(|t| (t.translation - b.translation).length() < 0.3 && t.rotation.mul_vec3(Vec3::Y).y < 0.5);
    let fly_at =
        flies.iter().find(|(_, f)| !f.dead()).and_then(|(_, f)| parts.get(f.parts[Part::Thorax.index()]).ok().map(|t| t.translation));
    if pressed && !c.fired {
        c.fired = true;
        fb.sfx.write(Sfx::new("click"));
        fb.notes.push(now, format!("CONTRAPTION button pressed -> {}", c.payload.label()), Tag::GameRule);
        if let Some(at) = fly_at {
            match c.payload {
                Payload::Bomb => {
                    commands.spawn((
                        Bomb { born: now - 0.8 },
                        RigidBody::Dynamic,
                        Collider::sphere(0.08),
                        Restitution::new(0.35),
                        LinearVelocity(Vec3::ZERO),
                        Mesh3d(meshes.add(Sphere::new(0.08).mesh().uv(16, 10))),
                        MeshMaterial3d(materials.add(StandardMaterial {
                            base_color: Color::srgb(0.08, 0.08, 0.1),
                            emissive: LinearRgba::rgb(0.6, 0.1, 0.0),
                            ..default()
                        })),
                        Transform::from_translation(at + Vec3::Y * 1.0),
                    ));
                }
                Payload::Swat => {
                    for (e, mut f) in &mut flies {
                        if f.dead() {
                            continue;
                        }
                        f.last_hit = at + Vec3::Y;
                        f.stun(now, 1.8);
                        for part in Part::ALL {
                            ff.hits.write(FlyHit { fly: e, part, strength: 1.0 });
                        }
                        for (p, mut v) in &mut fly_vel {
                            if p.fly == e {
                                v.0 += Vec3::new(0.0, -32.0 * 0.006 * 60.0, 0.0);
                            }
                        }
                        ff.dmg.write(FlyDamage { fly: e, amount: 14.0, source: "a contraption's swatter" });
                    }
                    fb.popups.write(Popup::new(at + Vec3::Y * 0.5, "SWAT!", (255, 230, 120)));
                    fb.sfx.write(Sfx::new("whack"));
                }
                Payload::Wind => c.wind_until = now + 4.0,
                Payload::Sugar => {
                    commands.spawn((
                        crate::tools::Treat::new("sugar", 0.035, now),
                        RigidBody::Dynamic,
                        Collider::sphere(0.035),
                        Friction::new(1.0),
                        LinearVelocity(Vec3::ZERO),
                        Mesh3d(meshes.add(Sphere::new(0.035).mesh().uv(12, 8))),
                        MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::WHITE, ..default() })),
                        Transform::from_translation(at + Vec3::new(0.3, 0.8, 0.0)),
                    ));
                }
            }
        }
    }
    if now < c.wind_until {
        // the gust reaches every fly's Johnston's organ, each on its own brain
        for (e, f) in &flies {
            if f.dead() {
                continue;
            }
            if let Ok(slot) = slots.get(e) {
                link.0.send(Command::PokeFly { slot: slot.0, region: "wind".into(), side: None, strength: 0.8 });
            }
        }
    }
}

pub struct ContraptionPlugin;

impl Plugin for ContraptionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Contraption>().add_systems(Update, control).add_systems(FixedUpdate, run);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_machine_is_laid_out_downhill_along_its_direction() {
        let l = layout(Vec3::ZERO, Vec3::X);
        assert!(l.ramp_top.y > 0.3, "the marble starts high");
        assert!(l.dominoes.windows(2).all(|w| w[1].x > w[0].x), "the dominoes run away from the ramp");
        assert!(l.button.x > l.dominoes.last().map_or(0.0, |d| d.x), "the button is past the last domino");
        let l2 = layout(Vec3::new(1.0, 0.0, 1.0), Vec3::Z);
        assert!(l2.button.z > l2.dominoes[0].z, "the same machine turned to face +z");
    }

    #[test]
    fn payloads_cycle_through_all_four() {
        let mut p = Payload::Bomb;
        let mut seen = vec![p];
        for _ in 0..3 {
            p = p.next();
            seen.push(p);
        }
        assert_eq!(p.next(), Payload::Bomb);
        seen.dedup();
        assert_eq!(seen.len(), 4);
    }
}
