//! The arenas as scenery and static colliders: the room (and its fan, flypaper, pool, lamp, thermo and escape-room variants), the kitchen,
//! the open field and the orchard, plus the lighting that follows the day, the weather and the lightning.
//!
//! What lives here is geometry and light. What the arenas DO to a fly (wind, glue, drowning, heat, the cook) is `environment.rs`. The numbers
//! are `world.rs` and `kitchen.rs`. Anything numerous (grass) is one merged mesh, one draw call; the rest are a handful of solids.

use avian3d::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::config::{Arena, SwitchArena};
use crate::kitchen as kit;
use crate::rng::Rng;
use crate::world::{self, FAN3, LAMP3, PAPER3, RX, RY, RZ, THERMO_HALF3, WATER3};

/// Everything spawned for the current arena (despawned together on a switch).
#[derive(Component)]
pub struct ArenaScene;

/// A thing that turns (the fan's blades): radians per second about its local axis.
#[derive(Component)]
pub struct Spin(pub Vec3);

/// The sun, the ceiling light and the lamp, so the lighting system can find them.
#[derive(Component)]
pub struct SunLight;
#[derive(Component)]
pub struct CeilingLight;
#[derive(Component)]
pub struct LampLight;
/// The kitchen burner's glowing ring (brighter when a fly is near, it is a hot plate).
#[derive(Component)]
pub struct BurnerGlow;

/// The boxes you and the walking fly go around in the current arena (furniture, the jar, tree trunks).
#[derive(Resource, Debug, Clone, Default)]
pub struct WorldColliders(pub Vec<world::Aabb>);

/// How bright the world is right now, set by the weather and the day/night cycle and read by [`update_lighting`].
#[derive(Resource, Debug, Clone, Copy)]
pub struct Atmosphere {
    /// 1 in daylight, 0.12 at night, darkened by a storm.
    pub day: f32,
    /// Lightning brightness 0..1 (or the slow swell under reduced flashing).
    pub flash: f32,
    pub sun_dir: Vec3,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self { day: 1.0, flash: 0.0, sun_dir: Vec3::new(0.3, 0.55, -0.78).normalize() }
    }
}

/// A procedural texture: `f(u, v)` -> sRGB colour, with u, v in 0..1.
pub fn procedural_image(w: u32, h: u32, f: impl Fn(f32, f32) -> [f32; 3]) -> Image {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let c = f(x as f32 / w as f32, y as f32 / h as f32);
            data.extend_from_slice(&[
                (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ]);
        }
    }
    Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn hash2(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(374_761_393).wrapping_add(y.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) & 0xffff) as f32 / 65535.0
}

struct Kit<'a> {
    commands: Commands<'a, 'a>,
    meshes: &'a mut Assets<Mesh>,
    materials: &'a mut Assets<StandardMaterial>,
}

impl Kit<'_> {
    fn mat(&mut self, color: Color, rough: f32) -> Handle<StandardMaterial> {
        self.materials.add(StandardMaterial { base_color: color, perceptual_roughness: rough, ..default() })
    }

    fn tex_mat(&mut self, image: Handle<Image>, tint: Color, rough: f32, tile: Vec2) -> Handle<StandardMaterial> {
        self.materials.add(StandardMaterial {
            base_color: tint,
            base_color_texture: Some(image),
            perceptual_roughness: rough,
            uv_transform: bevy::math::Affine2::from_scale(tile),
            ..default()
        })
    }

    /// A solid box with a collider. `half` is half-extents.
    fn solid(&mut self, center: Vec3, half: Vec3, mat: Handle<StandardMaterial>) -> Entity {
        self.commands
            .spawn((
                ArenaScene,
                RigidBody::Static,
                Collider::cuboid(half.x * 2.0, half.y * 2.0, half.z * 2.0),
                Mesh3d(self.meshes.add(Cuboid::new(half.x * 2.0, half.y * 2.0, half.z * 2.0))),
                MeshMaterial3d(mat),
                Transform::from_translation(center),
            ))
            .id()
    }

    /// A box with no collider (decoration).
    fn deco(&mut self, center: Vec3, half: Vec3, mat: Handle<StandardMaterial>) -> Entity {
        self.commands
            .spawn((
                ArenaScene,
                Mesh3d(self.meshes.add(Cuboid::new(half.x * 2.0, half.y * 2.0, half.z * 2.0))),
                MeshMaterial3d(mat),
                Transform::from_translation(center),
            ))
            .id()
    }

    fn sphere(&mut self, center: Vec3, r: f32, mat: Handle<StandardMaterial>) -> Entity {
        self.commands
            .spawn((
                ArenaScene,
                Mesh3d(self.meshes.add(Sphere::new(r).mesh().uv(24, 16))),
                MeshMaterial3d(mat),
                Transform::from_translation(center),
            ))
            .id()
    }

    fn cylinder(&mut self, center: Vec3, r: f32, h: f32, mat: Handle<StandardMaterial>) -> Entity {
        self.commands
            .spawn((ArenaScene, Mesh3d(self.meshes.add(Cylinder::new(r, h))), MeshMaterial3d(mat), Transform::from_translation(center)))
            .id()
    }

    fn solid_cylinder(&mut self, center: Vec3, r: f32, h: f32, mat: Handle<StandardMaterial>) -> Entity {
        let e = self.cylinder(center, r, h, mat);
        self.commands.entity(e).insert((RigidBody::Static, Collider::cylinder(r, h)));
        e
    }
}

/// The room shell: floor, four walls, ceiling, all with colliders (the fly is a ragdoll and the walls are what it hits).
fn room_shell(k: &mut Kit, images: &mut Assets<Image>, floor_tint: Color) {
    let wood = images.add(procedural_image(128, 128, |u, v| {
        let plank = (u * 8.0).floor();
        let n = hash2(plank as u32, (v * 6.0) as u32) * 0.12;
        let grain = ((v * 40.0 + plank * 3.1).sin() * 0.03).abs();
        let edge = if (u * 8.0).fract() < 0.04 { 0.35 } else { 1.0 };
        [(0.52 + n - grain) * edge, (0.36 + n * 0.8 - grain) * edge, (0.22 + n * 0.5) * edge]
    }));
    let wallpaper = images.add(procedural_image(64, 64, |u, v| {
        let stripe = if (u * 6.0).fract() < 0.5 { 0.86 } else { 0.78 };
        let dot = if ((u * 6.0).fract() - 0.25).abs() < 0.04 && (v * 5.0).fract() < 0.3 { 0.08 } else { 0.0 };
        [stripe - dot, stripe * 0.94 - dot, stripe * 0.86]
    }));
    let floor = k.tex_mat(wood, floor_tint, 0.7, Vec2::new(3.0, 2.0));
    let wall = k.tex_mat(wallpaper, Color::srgb(0.9, 0.86, 0.8), 0.95, Vec2::new(4.0, 1.2));
    let ceil = k.mat(Color::srgb(0.93, 0.92, 0.9), 1.0);
    let t = 0.3;
    k.solid(Vec3::new(0.0, -t, 0.0), Vec3::new(RX + t, t, RZ + t), floor);
    k.solid(Vec3::new(0.0, RY + t, 0.0), Vec3::new(RX + t, t, RZ + t), ceil);
    k.solid(Vec3::new(0.0, RY / 2.0, -RZ - t), Vec3::new(RX + t, RY / 2.0 + t, t), wall.clone());
    k.solid(Vec3::new(0.0, RY / 2.0, RZ + t), Vec3::new(RX + t, RY / 2.0 + t, t), wall.clone());
    k.solid(Vec3::new(-RX - t, RY / 2.0, 0.0), Vec3::new(t, RY / 2.0 + t, RZ + t), wall.clone());
    k.solid(Vec3::new(RX + t, RY / 2.0, 0.0), Vec3::new(t, RY / 2.0 + t, RZ + t), wall);
}

fn room_furniture(k: &mut Kit, images: &mut Assets<Image>) {
    let rug_img = images.add(procedural_image(64, 64, |u, v| {
        let d = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
        let ring = ((d * 18.0).sin() * 0.5 + 0.5) * 0.25;
        [0.36 + ring * 0.8, 0.14 + ring * 0.3, 0.16 + ring * 0.25]
    }));
    let rug = k.tex_mat(rug_img, Color::WHITE, 1.0, Vec2::ONE);
    k.deco(Vec3::new(0.0, 0.006, 0.4), Vec3::new(2.4, 0.006, 1.7), rug);
    // couch (matches COLLIDERS[0]: x -3.5..-1.1, y 0..0.62, z -RZ..-RZ+1)
    let fabric = k.mat(Color::srgb(0.32, 0.25, 0.42), 0.95);
    let b = world::room_colliders();
    let (c0, h0) = ((b[0].lo + b[0].hi) / 2.0, (b[0].hi - b[0].lo) / 2.0);
    k.solid(Vec3::new(c0.x, 0.2, c0.z), Vec3::new(h0.x, 0.2, h0.z), fabric.clone());
    k.solid(Vec3::new(c0.x, 0.45, -RZ + 0.12), Vec3::new(h0.x, 0.3, 0.12), fabric.clone());
    k.solid(Vec3::new(b[0].lo.x + 0.12, 0.42, c0.z), Vec3::new(0.12, 0.22, h0.z), fabric);
    // bookshelf: x RX-0.45..RX, z -1.9..-0.1, 2.05 high, with a row of coloured books on each shelf
    let wood = k.mat(Color::srgb(0.36, 0.24, 0.14), 0.8);
    let (c1, h1) = ((b[1].lo + b[1].hi) / 2.0, (b[1].hi - b[1].lo) / 2.0);
    k.solid(c1, h1, wood.clone());
    // the books: one merged mesh with vertex colours, so a whole shelf is a single draw call
    let mut rng = Rng::new(5);
    let (mut pos, mut col, mut nor, mut idx): (Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>, Vec<u32>) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for shelf in 0..5 {
        let y = 0.25 + shelf as f32 * 0.38;
        let mut z = b[1].lo.z + 0.08;
        while z < b[1].hi.z - 0.1 {
            let w = rng.uniform(0.04, 0.09);
            let hgt = rng.uniform(0.2, 0.32);
            let c = Color::hsl(rng.uniform(0.0, 360.0), 0.5, rng.uniform(0.3, 0.5)).to_linear();
            let (cx, cy, cz) = (b[1].lo.x - 0.01, y + hgt / 2.0 - 0.1, z + w / 2.0);
            let (hx, hy, hz) = (0.12, hgt / 2.0, w / 2.0);
            // five visible faces of a box (front -x, top, the two sides)
            let faces: [([f32; 3], [[f32; 3]; 4]); 4] = [
                ([-1.0, 0.0, 0.0], [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]]),
                ([0.0, 1.0, 0.0], [[-hx, hy, -hz], [-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz]]),
                ([0.0, 0.0, 1.0], [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]),
                ([0.0, 0.0, -1.0], [[-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz], [hx, -hy, -hz]]),
            ];
            for (n, quad) in faces {
                let base = pos.len() as u32;
                for q in quad {
                    pos.push([cx + q[0], cy + q[1], cz + q[2]]);
                    col.push([c.red, c.green, c.blue, 1.0]);
                    nor.push(n);
                }
                idx.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            }
            z += w + 0.004;
        }
    }
    let books = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nor)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, col)
        .with_inserted_indices(Indices::U32(idx));
    let book_mat = k.mat(Color::WHITE, 0.9);
    let book_mesh = k.meshes.add(books);
    k.commands.spawn((ArenaScene, Mesh3d(book_mesh), MeshMaterial3d(book_mat), Transform::default()));
    // plant
    let pot = k.mat(Color::srgb(0.6, 0.32, 0.2), 0.9);
    let leaf = k.mat(Color::srgb(0.18, 0.5, 0.2), 0.9);
    let (c2, _) = ((b[2].lo + b[2].hi) / 2.0, 0.0);
    k.solid_cylinder(Vec3::new(c2.x, 0.2, c2.z), 0.28, 0.4, pot);
    for i in 0..7 {
        let a = i as f32 * 0.9;
        k.sphere(Vec3::new(c2.x + a.cos() * 0.18, 0.55 + 0.05 * i as f32, c2.z + a.sin() * 0.18), 0.17, leaf.clone());
    }
}

/// A window on the back wall (decorative) so the room has a visible light source.
fn window(k: &mut Kit) {
    let frame = k.mat(Color::srgb(0.85, 0.85, 0.82), 0.6);
    let glass = k.materials.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.8, 1.0),
        emissive: LinearRgba::rgb(0.5, 0.7, 1.1),
        unlit: true,
        ..default()
    });
    k.deco(Vec3::new(1.8, 1.7, -RZ + 0.02), Vec3::new(0.8, 0.7, 0.02), glass);
    for (c, h) in [
        (Vec3::new(1.8, 2.42, -RZ + 0.04), Vec3::new(0.86, 0.04, 0.04)),
        (Vec3::new(1.8, 0.98, -RZ + 0.04), Vec3::new(0.86, 0.04, 0.04)),
        (Vec3::new(0.95, 1.7, -RZ + 0.04), Vec3::new(0.04, 0.74, 0.04)),
        (Vec3::new(2.65, 1.7, -RZ + 0.04), Vec3::new(0.04, 0.74, 0.04)),
        (Vec3::new(1.8, 1.7, -RZ + 0.04), Vec3::new(0.02, 0.7, 0.04)),
    ] {
        k.deco(c, h, frame.clone());
    }
}

fn fan(k: &mut Kit) {
    let metal = k.mat(Color::srgb(0.55, 0.57, 0.6), 0.4);
    let blade = k.materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.72, 0.78),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.4,
        ..default()
    });
    let c = Vec3::new(FAN3.x, 0.9, FAN3.z);
    k.cylinder(Vec3::new(c.x - 0.12, 0.35, c.z), 0.05, 0.7, metal.clone());
    k.deco(Vec3::new(c.x - 0.12, 0.05, c.z), Vec3::new(0.25, 0.05, 0.25), metal.clone());
    let hub = k.sphere(c, 0.09, metal);
    k.commands.entity(hub).insert(Spin(Vec3::ZERO));
    // four blades on a spinning pivot (rotation about the wind axis, +x)
    let pivot = k.commands.spawn((ArenaScene, Transform::from_translation(c), Visibility::default(), Spin(Vec3::new(18.0, 0.0, 0.0)))).id();
    for i in 0..4 {
        let a = i as f32 * std::f32::consts::FRAC_PI_2;
        let blade_e = k
            .commands
            .spawn((
                Mesh3d(k.meshes.add(Cuboid::new(0.02, 0.5, 0.16))),
                MeshMaterial3d(blade.clone()),
                Transform::from_rotation(Quat::from_rotation_x(a) * Quat::from_rotation_z(0.0))
                    .with_translation(Vec3::new(0.0, a.cos() * 0.28, a.sin() * 0.28))
                    .with_rotation(Quat::from_rotation_x(a)),
            ))
            .id();
        k.commands.entity(pivot).add_child(blade_e);
    }
}

fn flypaper(k: &mut Kit, images: &mut Assets<Image>) {
    let (x0, x1, z0, z1) = PAPER3;
    let img = images.add(procedural_image(64, 64, |u, v| {
        let n = hash2((u * 64.0) as u32, (v * 64.0) as u32) * 0.2;
        [0.9 + n * 0.3, 0.74 + n * 0.3, 0.2 + n * 0.2]
    }));
    let m = k.materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(img),
        perceptual_roughness: 0.15,
        reflectance: 0.9,
        ..default()
    });
    k.deco(Vec3::new((x0 + x1) / 2.0, 0.008, (z0 + z1) / 2.0), Vec3::new((x1 - x0) / 2.0, 0.008, (z1 - z0) / 2.0), m);
}

fn pool(k: &mut Kit) {
    // the whole floor is a pool 0.6 m deep: walls of water, a translucent surface
    let water = k.materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.5, 0.75, 0.55),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.05,
        reflectance: 0.8,
        ..default()
    });
    k.deco(Vec3::new(0.0, WATER3, 0.0), Vec3::new(RX, 0.004, RZ), water);
}

fn lamp(k: &mut Kit, on: bool) {
    let shade = k.mat(Color::srgb(0.2, 0.2, 0.22), 0.6);
    let bulb = k.materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.9, 0.6),
        emissive: if on { LinearRgba::rgb(14.0, 10.0, 4.0) } else { LinearRgba::rgb(0.4, 0.35, 0.2) },
        ..default()
    });
    k.cylinder(Vec3::new(LAMP3.x, RY - 0.25, LAMP3.z), 0.012, 0.5, shade.clone());
    k.sphere(LAMP3, 0.14, bulb);
    k.cylinder(Vec3::new(LAMP3.x, LAMP3.y + 0.2, LAMP3.z), 0.2, 0.12, shade);
    if on {
        k.commands.spawn((
            ArenaScene,
            LampLight,
            PointLight { intensity: 900_000.0, range: 12.0, color: Color::srgb(1.0, 0.85, 0.55), shadow_maps_enabled: false, ..default() },
            Transform::from_translation(LAMP3),
        ));
    }
}

/// The thermo arena: a floor that fades blue (15 C) to red (35 C) across the room.
fn thermo(k: &mut Kit, images: &mut Assets<Image>) {
    let img = images.add(procedural_image(256, 4, |u, _| {
        let t = u;
        [0.15 + 0.8 * t, 0.35 + 0.1 * (1.0 - (2.0 * t - 1.0).abs()), 0.9 - 0.8 * t]
    }));
    let m = k.materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(img),
        perceptual_roughness: 0.9,
        unlit: false,
        ..default()
    });
    k.deco(Vec3::new(0.0, 0.007, 0.0), Vec3::new(THERMO_HALF3 + 0.3, 0.006, RZ - 0.1), m);
}

fn sugar_goal(k: &mut Kit) {
    let sugar = k.materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.95, 0.95),
        emissive: LinearRgba::rgb(0.6, 0.5, 0.55),
        ..default()
    });
    for i in 0..6 {
        k.deco(
            Vec3::new(RX - 0.7 + (i % 3) as f32 * 0.06 - 0.06, 0.03 + (i / 3) as f32 * 0.05, (i % 2) as f32 * 0.07),
            Vec3::splat(0.03),
            sugar.clone(),
        );
    }
}

/// The kitchen: the room's floor is the countertop; a fridge, the fruit bowl, the vinegar trap, the sink and the burner stand on it.
fn kitchen(k: &mut Kit, images: &mut Assets<Image>) {
    let tile = images.add(procedural_image(64, 64, |u, v| {
        let g = if (u * 8.0).fract() < 0.06 || (v * 8.0).fract() < 0.06 { 0.55 } else { 0.9 };
        [g, g * 0.97, g * 0.9]
    }));
    let counter = k.tex_mat(tile, Color::WHITE, 0.4, Vec2::new(4.0, 3.0));
    // retexture the floor: draw the counter just above it
    k.deco(Vec3::new(0.0, 0.004, 0.0), Vec3::new(RX, 0.004, RZ), counter);
    let steel = k.mat(Color::srgb(0.75, 0.77, 0.8), 0.25);
    let b = kit::colliders();
    let fridge = &b[1];
    k.solid((fridge.lo + fridge.hi) / 2.0, (fridge.hi - fridge.lo) / 2.0, steel.clone());
    let dark = k.mat_dark();
    k.deco(Vec3::new(fridge.hi.x + 0.02, 1.2, (fridge.lo.z + fridge.hi.z) / 2.0), Vec3::new(0.02, 0.4, 0.03), dark);
    // fruit bowl
    let ceramic = k.mat(Color::srgb(0.85, 0.8, 0.7), 0.5);
    k.cylinder(kit::BOWL_POS + Vec3::new(0.0, kit::BOWL_H / 2.0, 0.0), kit::BOWL_R, kit::BOWL_H, ceramic);
    // vinegar trap: a glass jar with a funnel mouth
    let glass = k.materials.add(StandardMaterial {
        base_color: Color::srgba(0.8, 0.9, 0.7, 0.35),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.05,
        ..default()
    });
    let jar = k.solid_cylinder(kit::TRAP_POS + Vec3::new(0.0, kit::TRAP_H / 2.0, 0.0), kit::TRAP_R, kit::TRAP_H, glass);
    k.commands.entity(jar).insert(Name::new("vinegar trap"));
    let vinegar = k.materials.add(StandardMaterial { base_color: Color::srgb(0.45, 0.22, 0.1), perceptual_roughness: 0.2, ..default() });
    k.cylinder(kit::TRAP_POS + Vec3::new(0.0, 0.2, 0.0), kit::TRAP_R * 0.92, 0.4, vinegar);
    // the sink: a dark basin with water 0.3 m deep
    let (x0, x1, z0, z1) = kit::SINK;
    let basin = k.mat(Color::srgb(0.25, 0.27, 0.3), 0.4);
    k.deco(Vec3::new((x0 + x1) / 2.0, 0.01, (z0 + z1) / 2.0), Vec3::new((x1 - x0) / 2.0 + 0.1, 0.01, (z1 - z0) / 2.0 + 0.1), basin);
    let water = k.materials.add(StandardMaterial {
        base_color: Color::srgba(0.25, 0.55, 0.8, 0.5),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.05,
        reflectance: 0.8,
        ..default()
    });
    k.deco(Vec3::new((x0 + x1) / 2.0, kit::SINK_WATER, (z0 + z1) / 2.0), Vec3::new((x1 - x0) / 2.0, 0.004, (z1 - z0) / 2.0), water);
    // the burner: a flat disc and a glowing ring
    let iron = k.mat(Color::srgb(0.1, 0.1, 0.11), 0.5);
    k.cylinder(kit::BURNER_POS + Vec3::new(0.0, 0.02, 0.0), kit::BURNER_R + 0.08, 0.04, iron);
    let ring = k.materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.2, 0.05),
        emissive: LinearRgba::rgb(6.0, 1.0, 0.1),
        ..default()
    });
    let e = k
        .commands
        .spawn((
            ArenaScene,
            BurnerGlow,
            Mesh3d(k.meshes.add(Torus::new(kit::BURNER_R * 0.55, kit::BURNER_R * 0.8))),
            MeshMaterial3d(ring),
            Transform::from_translation(kit::BURNER_POS + Vec3::new(0.0, 0.05, 0.0)),
        ))
        .id();
    let _ = e;
    k.commands.spawn((
        ArenaScene,
        PointLight { intensity: 120_000.0, range: 3.0, color: Color::srgb(1.0, 0.4, 0.1), ..default() },
        Transform::from_translation(kit::BURNER_POS + Vec3::new(0.0, 0.3, 0.0)),
    ));
}

impl Kit<'_> {
    fn mat_dark(&mut self) -> Handle<StandardMaterial> {
        self.mat(Color::srgb(0.05, 0.05, 0.06), 0.5)
    }
}

/// One merged mesh of every grass tuft (two crossed quads each): thousands of blades, one draw call.
fn grass_mesh(tufts: &[world::Tuft], seed: u64) -> Mesh {
    let mut rng = Rng::new(seed);
    let (mut pos, mut col, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for t in tufts {
        let a = rng.uniform(0.0, std::f32::consts::PI);
        let w = t.height * 0.35;
        let green = [0.18 * t.shade, 0.42 * t.shade, 0.14 * t.shade, 1.0];
        let tip = [0.3 * t.shade, 0.6 * t.shade, 0.2 * t.shade, 1.0];
        for k in 0..2 {
            let ang = a + k as f32 * std::f32::consts::FRAC_PI_2;
            let d = Vec3::new(ang.cos(), 0.0, ang.sin()) * w;
            let base = t.pos - Vec3::new(0.0, t.height / 2.0, 0.0);
            let b = pos.len() as u32;
            pos.extend([
                (base - d).to_array(),
                (base + d).to_array(),
                (base + d * 0.15 + Vec3::Y * t.height).to_array(),
                (base - d * 0.15 + Vec3::Y * t.height).to_array(),
            ]);
            col.extend([green, green, tip, tip]);
            idx.extend([b, b + 1, b + 2, b, b + 2, b + 3, b, b + 2, b + 1, b, b + 3, b + 2]);
        }
    }
    let n = pos.len();
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, col)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32, 1.0, 0.0]; n])
        .with_inserted_indices(Indices::U32(idx))
}

fn outdoors(k: &mut Kit, images: &mut Assets<Image>, arena: &str) {
    let w = world::spec(arena);
    let sc = world::scenery(arena, 7);
    let ground_img = images.add(procedural_image(128, 128, |u, v| {
        let n = hash2((u * 128.0) as u32, (v * 128.0) as u32) * 0.12;
        [w.ground[0] + n * 0.4, w.ground[1] + n * 0.5, w.ground[2] + n * 0.3]
    }));
    let ground = k.tex_mat(ground_img, Color::WHITE, 1.0, Vec2::splat(60.0));
    let big = 220.0;
    k.solid(Vec3::new(0.0, -0.5, 0.0), Vec3::new(big, 0.5, big), ground);
    let rock = k.mat(Color::srgb(0.45, 0.44, 0.42), 0.95);
    for r in &sc.rocks {
        let rm = k.materials.add(StandardMaterial {
            base_color: Color::srgb(r.shade, r.shade * 0.98, r.shade * 0.94),
            perceptual_roughness: 0.95,
            ..default()
        });
        let e = k
            .commands
            .spawn((
                ArenaScene,
                Mesh3d(k.meshes.add(Sphere::new(1.0).mesh().uv(12, 8))),
                MeshMaterial3d(rm),
                Transform::from_translation(r.pos).with_rotation(Quat::from_rotation_y(r.yaw)).with_scale(r.scale),
            ))
            .id();
        k.commands.entity(e).insert((RigidBody::Static, Collider::sphere(1.0)));
    }
    let _ = rock;
    let grass = k.meshes.add(grass_mesh(&sc.tufts, 21));
    let gm = k.materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        cull_mode: None,
        double_sided: true,
        ..default()
    });
    k.commands.spawn((ArenaScene, Mesh3d(grass), MeshMaterial3d(gm), Transform::default()));
    let bark = k.mat(Color::srgb(0.32, 0.22, 0.14), 1.0);
    let leaves = k.mat(Color::srgb(0.2, 0.45, 0.18), 0.9);
    for t in &sc.trees {
        let trunk = k.solid_cylinder(t.pos + Vec3::new(0.0, t.height / 2.0, 0.0), 0.18, t.height, bark.clone());
        let _ = trunk;
        k.sphere(t.pos + Vec3::new(0.0, t.height + 0.2, 0.0), t.crown, leaves.clone());
        k.sphere(t.pos + Vec3::new(t.crown * 0.5, t.height - 0.1, 0.2), t.crown * 0.7, leaves.clone());
    }
}

/// Spawn the scenery of `arena` and its lights.
#[allow(clippy::too_many_arguments)]
pub fn build_arena(
    commands: Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    arena: &str,
) {
    let mut k = Kit { commands, meshes, materials };
    let outdoor = world::is_outdoor(arena);
    let boxes = match arena {
        "kitchen" => kit::colliders(),
        a if world::is_outdoor(a) => world::tree_colliders(&world::scenery(a, 7).trees),
        _ => world::room_colliders(),
    };
    k.commands.insert_resource(WorldColliders(boxes));
    if outdoor {
        outdoors(&mut k, images, arena);
    } else {
        room_shell(&mut k, images, if arena == "pool" { Color::srgb(0.5, 0.55, 0.6) } else { Color::WHITE });
        window(&mut k);
        match arena {
            "kitchen" => kitchen(&mut k, images),
            _ => room_furniture(&mut k, images),
        }
        // the ceiling light fixture
        let glow = k.materials.add(StandardMaterial { base_color: Color::WHITE, emissive: LinearRgba::rgb(5.0, 4.5, 3.8), ..default() });
        k.cylinder(Vec3::new(0.0, RY - 0.05, 0.0), 0.35, 0.05, glow);
        k.commands.spawn((
            ArenaScene,
            CeilingLight,
            PointLight { intensity: 1_200_000.0, range: 14.0, color: Color::srgb(1.0, 0.93, 0.82), shadow_maps_enabled: true, ..default() },
            Transform::from_xyz(0.0, RY - 0.3, 0.0),
        ));
    }
    match arena {
        "fan" => fan(&mut k),
        "flypaper" => flypaper(&mut k, images),
        "pool" => pool(&mut k),
        "lamp" => lamp(&mut k, true),
        "thermo" => thermo(&mut k, images),
        "escaperoom" => {
            fan(&mut k);
            flypaper(&mut k, images);
            lamp(&mut k, true);
            sugar_goal(&mut k);
        }
        _ => {}
    }
    k.commands.spawn((
        ArenaScene,
        SunLight,
        DirectionalLight {
            illuminance: if outdoor { 14_000.0 } else { 2_500.0 },
            shadow_maps_enabled: true,
            color: Color::srgb(1.0, 0.95, 0.85),
            ..default()
        },
        Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)).looking_to(Vec3::new(-0.3, -0.55, 0.78), Vec3::Y),
    ));
}

fn spawn_initial(
    commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    arena: Res<Arena>,
) {
    build_arena(commands, &mut meshes, &mut materials, &mut images, arena.name());
}

#[allow(clippy::too_many_arguments)]
fn switch_arena(
    mut commands: Commands,
    mut msgs: MessageReader<SwitchArena>,
    mut arena: ResMut<Arena>,
    scene: Query<Entity, With<ArenaScene>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(SwitchArena(i)) = msgs.read().last().copied() else { return };
    arena.0 = i % world::ARENAS.len();
    for e in &scene {
        commands.entity(e).despawn();
    }
    build_arena(commands, &mut meshes, &mut materials, &mut images, arena.name());
}

fn spin(time: Res<Time>, mut q: Query<(&Spin, &mut Transform)>) {
    for (s, mut t) in &mut q {
        t.rotate_local(Quat::from_euler(EulerRot::XYZ, s.0.x * time.delta_secs(), s.0.y * time.delta_secs(), s.0.z * time.delta_secs()));
    }
}

/// The sun, the ceiling light and the clear colour follow the day, the storm and the lightning.
#[allow(clippy::type_complexity)]
fn update_lighting(
    atmo: Res<Atmosphere>,
    arena: Res<Arena>,
    mut sun: Query<(&mut DirectionalLight, &mut Transform), With<SunLight>>,
    mut ceiling: Query<&mut PointLight, (With<CeilingLight>, Without<LampLight>)>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut clear: ResMut<ClearColor>,
) {
    let outdoor = world::is_outdoor(arena.name());
    let boost = 1.0 + 2.5 * atmo.flash;
    for (mut l, mut t) in &mut sun {
        if outdoor {
            let up = atmo.sun_dir.y.max(0.0);
            l.illuminance = 14_000.0 * (0.25 + 0.9 * up.sqrt()) * atmo.day * boost;
            *t = Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)).looking_to(-atmo.sun_dir, Vec3::Y);
        } else {
            l.illuminance = 2_500.0 * boost;
        }
    }
    for mut l in &mut ceiling {
        l.intensity = 1_200_000.0 * boost;
    }
    let w = world::spec(arena.name());
    if outdoor {
        let up = atmo.sun_dir.y.max(0.0);
        let k = (0.45 + 0.55 * up) * atmo.day * boost;
        clear.0 = Color::srgb((w.sky[0] * k).min(1.0), (w.sky[1] * k).min(1.0), (w.sky[2] * k).min(1.0));
        ambient.brightness = 350.0 * atmo.day * boost;
        ambient.color = Color::srgb(0.7, 0.8, 1.0);
    } else {
        clear.0 = Color::srgb(0.08, 0.09, 0.11);
        ambient.brightness = 220.0 * boost;
        ambient.color = Color::srgb(0.85, 0.88, 1.0);
    }
}

pub struct ArenaPlugin;

impl Plugin for ArenaPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Arena>()
            .init_resource::<Atmosphere>()
            .init_resource::<WorldColliders>()
            .add_message::<SwitchArena>()
            .add_systems(Startup, spawn_initial)
            .add_systems(Update, (switch_arena, spin, update_lighting).chain());
    }
}
