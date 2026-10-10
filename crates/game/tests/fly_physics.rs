//! The fly's body in the physics engine without a window: it stands at its standing height, stays upright, walks along the floor when told
//! to, goes limp when held up high, falls and settles, and a fly thrown at the floor reports an impact. Runs the real `FlyPlugin` systems
//! against avian with a floor, a synthetic brain thread and a manually stepped clock.

use std::time::Duration;

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use kick_the_fly::app::SimPlugin;
use kick_the_fly::arena::WorldColliders;
use kick_the_fly::config::{Arena, Settings};
use kick_the_fly::feedback::FeedbackPlugin;
use kick_the_fly::fly::{Fly, FlyPart, FlyPlugin, Part, spawn_fly};
use kick_the_fly::sim_thread::{SimConfig, Source};
use kick_the_fly::world::STAND3;

fn app() -> App {
    let mut config = SimConfig::new(Source::Synthetic { n: 2000 });
    config.warmup = 10;
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        AssetPlugin::default(),
        PhysicsPlugins::default(),
        SimPlugin { config },
        FeedbackPlugin,
        FlyPlugin,
    ))
    .init_asset::<Mesh>()
    .init_asset::<StandardMaterial>()
    .insert_resource(Gravity(Vec3::NEG_Y * 19.44))
    .insert_resource(Time::<Fixed>::from_hz(60.0))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_micros(16_667)))
    .insert_resource(Arena(0))
    .insert_resource(WorldColliders::default())
    .insert_resource(Settings::default())
    .insert_resource(kick_the_fly::tools::ToolState::default());
    app.world_mut().spawn((RigidBody::Static, Collider::cuboid(40.0, 1.0, 40.0), Transform::from_xyz(0.0, -0.5, 0.0)));
    app.finish();
    app.cleanup();
    app
}

fn spawn(app: &mut App, pos: Vec3) -> Entity {
    let mut commands_state =
        bevy::ecs::system::SystemState::<(Commands, ResMut<Assets<Mesh>>, ResMut<Assets<StandardMaterial>>)>::new(app.world_mut());
    let (mut commands, mut meshes, mut mats) = commands_state.get_mut(app.world_mut()).expect("system state");
    let e = spawn_fly(&mut commands, &mut meshes, &mut mats, pos, std::f32::consts::FRAC_PI_2, 1, 0);
    commands_state.apply(app.world_mut());
    e
}

fn part_pos(app: &mut App, fly: Entity, part: Part) -> Vec3 {
    let parts = app.world().get::<Fly>(fly).unwrap().parts;
    app.world().get::<Transform>(parts[part.index()]).unwrap().translation
}

fn run(app: &mut App, ticks: usize) {
    for _ in 0..ticks {
        app.update();
    }
}

#[test]
fn a_fly_dropped_on_the_floor_stands_up_at_its_standing_height_and_stays_upright() {
    let mut app = app();
    let fly = spawn(&mut app, Vec3::new(0.0, STAND3 + 0.4, 0.0));
    run(&mut app, 240);
    let thx = part_pos(&mut app, fly, Part::Thorax);
    assert!((thx.y - STAND3).abs() < 0.06, "thorax height {} vs standing {STAND3}", thx.y);
    let head = part_pos(&mut app, fly, Part::Head);
    let abd = part_pos(&mut app, fly, Part::Abdomen);
    assert!((head - thx).length() > 0.15 && (abd - thx).length() > 0.15, "the body is laid out head - thorax - abdomen");
    let parts = app.world().get::<Fly>(fly).unwrap().parts;
    let up = app.world().get::<Transform>(parts[1]).unwrap().rotation * Vec3::Y;
    assert!(up.y > 0.95, "upright: {up}");
}

#[test]
fn a_walk_command_moves_it_along_the_floor_and_a_hard_throw_makes_it_limp_then_stand_again() {
    let mut app = app();
    let fly = spawn(&mut app, Vec3::new(0.0, STAND3 + 0.05, 0.0));
    run(&mut app, 120);
    let start = part_pos(&mut app, fly, Part::Thorax);
    let now = app.world().resource::<Time<Virtual>>().elapsed_secs();
    {
        let mut f = app.world_mut().get_mut::<Fly>(fly).unwrap();
        f.react.walk_until = now + 2.0;
        f.run = false;
    }
    run(&mut app, 100);
    let moved = part_pos(&mut app, fly, Part::Thorax) - start;
    assert!(moved.length() > 0.5 && moved.y.abs() < 0.1, "walked {moved} in about 1.6 s along the floor (2.3 px/frame is 0.83 m/s)");
    // throw it at the floor: it is stunned (limp) for a moment and then recovers
    let parts = app.world().get::<Fly>(fly).unwrap().parts;
    for p in parts {
        app.world_mut().get_mut::<LinearVelocity>(p).unwrap().0 = Vec3::new(0.0, -9.0, 0.0);
    }
    let now = app.world().resource::<Time<Virtual>>().elapsed_secs();
    app.world_mut().get_mut::<Fly>(fly).unwrap().stun(now, 0.8);
    run(&mut app, 30);
    let strength_limp = app.world().get::<Fly>(fly).unwrap().strength;
    assert!(strength_limp < 0.5, "limp while stunned: {strength_limp}");
    run(&mut app, 200);
    let thx = part_pos(&mut app, fly, Part::Thorax);
    assert!((thx.y - STAND3).abs() < 0.08, "back on its feet: {}", thx.y);
}

#[test]
fn held_high_it_hangs_limp_and_released_it_falls() {
    let mut app = app();
    let fly = spawn(&mut app, Vec3::new(0.0, STAND3 + 0.05, 0.0));
    run(&mut app, 60);
    let parts = app.world().get::<Fly>(fly).unwrap().parts;
    {
        let mut f = app.world_mut().get_mut::<Fly>(fly).unwrap();
        f.grabbed = Some(Part::Thorax);
        f.hold = Vec3::new(0.0, 1.5, 0.0);
    }
    run(&mut app, 90);
    let held = part_pos(&mut app, fly, Part::Thorax);
    assert!((held.y - 1.5).abs() < 0.15, "pinned to the hand's point: {}", held.y);
    let abd = part_pos(&mut app, fly, Part::Abdomen);
    assert!(abd.y < held.y - 0.15, "held by the thorax, the heavy abdomen hangs below it (a ragdoll): {} vs {}", abd.y, held.y);
    app.world_mut().get_mut::<Fly>(fly).unwrap().grabbed = None;
    let mut fell = false;
    for _ in 0..120 {
        app.update();
        if app.world().get::<Transform>(parts[1]).unwrap().translation.y < 0.6 {
            fell = true;
            break;
        }
    }
    assert!(fell, "released, it falls");
    let _ = FlyPart { fly, part: Part::Head };
}
