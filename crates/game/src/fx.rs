//! Short-lived effects: fire, flame, spray, dust, sparks, ice shards, streaks and lightning bolts. Python: `Game3D.parts`, `puff`, `bolts3`,
//! `shards3`.
//!
//! The particles are simulated on the CPU (a few hundred floats of state each) but DRAWN as one instanced call: every frame they are written
//! into a [`ParticleCloud`] (`gpu.rs`), a single mesh whose vertex shader pulls each billboard from a storage buffer. Nothing spawns an
//! entity per particle.

use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;

use crate::gpu::{ParticleCloud, PointsMaterial, spawn_particle_cloud};
use crate::rng::Rng;

pub const CAPACITY: usize = 6000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Flame,
    Fire,
    Cleaner,
    Freeze,
    Dust,
    Streak,
    Spark,
    Ice,
    Steam,
    /// The blaster's muzzle flash and the pellet's trail (the duel).
    Muzzle,
    Trail,
}

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub born: f32,
    pub life: f32,
    pub size: f32,
    pub kind: Kind,
}

#[derive(Debug, Clone, Copy)]
pub struct Bolt {
    pub from: Vec3,
    pub to: Vec3,
    pub born: f32,
}

#[derive(Resource)]
pub struct Fx {
    pub particles: Vec<Particle>,
    pub bolts: Vec<Bolt>,
    pub rng: Rng,
}

impl Default for Fx {
    fn default() -> Self {
        Self { particles: Vec::new(), bolts: Vec::new(), rng: Rng::new(99) }
    }
}

impl Fx {
    pub fn add(&mut self, p: Particle) {
        if self.particles.len() < CAPACITY {
            self.particles.push(p);
        }
    }

    /// A puff of dust at a point (`puff`).
    pub fn puff(&mut self, pos: Vec3, n: usize, spread: f32, now: f32) {
        for _ in 0..n {
            let v = Vec3::new(self.rng.normal() * spread, self.rng.uniform(0.0, spread * 0.7), self.rng.normal() * spread)
                * 0.004
                * 60.0
                * 0.25;
            let life = self.rng.uniform(0.4, 0.8);
            self.add(Particle { pos, vel: v, born: now, life, size: 0.05, kind: Kind::Dust });
        }
    }

    /// The fireball of a bomb.
    pub fn explosion(&mut self, pos: Vec3, now: f32) {
        self.puff(pos, 24, 7.0, now);
        for _ in 0..140 {
            let d = Vec3::new(self.rng.normal(), self.rng.normal(), self.rng.normal()).normalize_or_zero()
                * self.rng.uniform(0.02, 0.07)
                * 60.0;
            let life = self.rng.uniform(0.25, 0.5);
            let size = self.rng.uniform(0.12, 0.3);
            self.add(Particle { pos, vel: d, born: now, life, size, kind: Kind::Fire });
        }
    }

    /// Ice shards when a frozen fly is smashed.
    pub fn shatter(&mut self, pos: Vec3, now: f32) {
        for _ in 0..60 {
            let d = Vec3::new(self.rng.normal(), self.rng.uniform(0.2, 1.5), self.rng.normal()).normalize_or_zero()
                * self.rng.uniform(0.6, 3.2);
            let life = self.rng.uniform(0.6, 1.4);
            let size = self.rng.uniform(0.02, 0.06);
            self.add(Particle { pos, vel: d, born: now, life, size, kind: Kind::Ice });
        }
    }

    /// A cone of jet particles from a nozzle along `dir` (`_jet`).
    pub fn jet(&mut self, tip: Vec3, dir: Vec3, kind: Kind, now: f32) {
        let (n, spread, speed) = if kind == Kind::Flame { (7, 0.12, (0.05, 0.08)) } else { (5, 0.18, (0.035, 0.055)) };
        for _ in 0..n {
            let j = Vec3::new(self.rng.normal(), self.rng.normal(), self.rng.normal()) * spread;
            let v = (dir + j).normalize_or_zero() * self.rng.uniform(speed.0, speed.1) * 60.0;
            let life = if kind == Kind::Flame { self.rng.uniform(0.22, 0.4) } else { self.rng.uniform(0.4, 0.65) };
            self.add(Particle { pos: tip, vel: v, born: now, life, size: 0.05, kind });
        }
    }

    pub fn bolt(&mut self, from: Vec3, to: Vec3, now: f32) {
        self.bolts.push(Bolt { from, to, born: now });
    }
}

#[derive(Component)]
pub struct FxCloud;

fn spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PointsMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let e = spawn_particle_cloud(&mut commands, &mut meshes, &mut materials, &mut buffers, CAPACITY, 1.6);
    commands.entity(e).insert(FxCloud);
}

fn color(p: &Particle, age: f32) -> ([f32; 3], f32) {
    let k = (age / p.life).clamp(0.0, 1.0);
    match p.kind {
        Kind::Flame => ([1.0, 0.55 - 0.35 * k, 0.12], 1.0 - k),
        Kind::Fire => ([1.0, 0.75 - 0.55 * k, 0.2 - 0.15 * k], 1.3 * (1.0 - k)),
        Kind::Cleaner => ([0.55, 0.85, 1.0], 0.8 * (1.0 - k)),
        Kind::Freeze => ([0.8, 0.95, 1.0], 0.9 * (1.0 - k)),
        Kind::Dust => ([0.75, 0.7, 0.62], 0.55 * (1.0 - k)),
        Kind::Streak => ([0.9, 0.95, 1.0], 0.5 * (1.0 - k)),
        Kind::Spark => ([1.0, 0.95, 0.5], 1.2 * (1.0 - k)),
        Kind::Ice => ([0.7, 0.9, 1.0], 1.0 - 0.6 * k),
        Kind::Steam => ([0.85, 0.85, 0.9], 0.4 * (1.0 - k)),
        Kind::Muzzle => ([0.7, 1.0, 0.5], 0.9 * (1.0 - k)),
        Kind::Trail => ([0.4, 1.0, 0.35], 0.4 * (1.0 - k)),
    }
}

/// Move and age the particles, then hand the survivors to the GPU cloud (one draw call).
pub fn step(time: Res<Time<Virtual>>, mut fx: ResMut<Fx>, mut cloud: Single<&mut ParticleCloud, With<FxCloud>>) {
    let (now, dt) = (time.elapsed_secs(), time.delta_secs());
    let fx = &mut *fx;
    for p in &mut fx.particles {
        p.pos += p.vel * dt;
        match p.kind {
            Kind::Flame => p.vel = p.vel * 0.95f32.powf(dt * 60.0) + Vec3::Y * 0.0015 * 60.0 * dt * 60.0,
            Kind::Cleaner | Kind::Freeze => p.vel = p.vel * 0.93f32.powf(dt * 60.0) - Vec3::Y * 0.0003 * 60.0 * dt * 60.0,
            Kind::Dust | Kind::Steam => p.vel = p.vel * 0.96f32.powf(dt * 60.0) + Vec3::Y * 0.0004 * 60.0 * dt * 60.0,
            Kind::Fire => p.vel = p.vel * 0.9f32.powf(dt * 60.0) + Vec3::Y * 0.001 * 60.0 * dt * 60.0,
            Kind::Ice => {
                p.vel.y -= crate::predators::GRAVITY * dt;
                if p.pos.y < p.size {
                    p.pos.y = p.size;
                    p.vel *= Vec3::new(0.6, -0.3, 0.6);
                }
            }
            Kind::Streak | Kind::Spark | Kind::Muzzle | Kind::Trail => {}
        }
    }
    fx.particles.retain(|p| now - p.born < p.life);
    fx.bolts.retain(|b| now - b.born < 0.25);
    for p in &fx.particles {
        let age = now - p.born;
        let (c, a) = color(p, age);
        let grow = match p.kind {
            Kind::Dust | Kind::Steam | Kind::Flame | Kind::Fire => 1.0 + 1.5 * age / p.life,
            // `muzzle`: 0.06 + 0.1 e, from a start size of 0.05
            Kind::Muzzle => 1.2 + 2.0 * age / p.life,
            _ => 1.0,
        };
        cloud.push(p.pos, p.size * grow, c, a);
    }
}

/// The zapper's bolts as jagged gizmo lines.
fn draw_bolts(time: Res<Time<Virtual>>, fx: Res<Fx>, mut gizmos: Gizmos) {
    let now = time.elapsed_secs();
    for b in &fx.bolts {
        let mut r = Rng::new((b.born * 1000.0) as u64 ^ ((now * 60.0) as u64));
        let n = 9;
        let mut prev = b.from;
        for i in 1..=n {
            let t = i as f32 / n as f32;
            let jitter = if i == n { Vec3::ZERO } else { Vec3::new(r.normal(), r.normal(), r.normal()) * 0.05 };
            let p = b.from.lerp(b.to, t) + jitter;
            gizmos.line(prev, p, Color::srgb(0.75, 0.92, 1.0));
            prev = p;
        }
    }
}

pub struct FxPlugin;

impl Plugin for FxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Fx>()
            .add_systems(Startup, spawn)
            .add_systems(Update, (step, draw_bolts).chain().before(crate::gpu::flush_particles));
    }
}
