//! GPU-driven drawing for anything numerous: neurons, particles and rain.
//!
//! The technique is vertex pulling. One mesh holds N quads (their corner and index are in the vertex position), a custom [`Material`] binds
//! a storage buffer, and the vertex shader reads each quad's record from it. A draw is therefore ONE entity and ONE draw call however many
//! things it shows, and the data the sim produces reaches the GPU as a plain buffer write per frame:
//!
//! * [`PointsMaterial`]: generic glowing billboards (bomb fire, flame, dust, sparks, streaks) from a `Vec<GpuPoint>`.
//! * [`BrainMaterial`]: the whole-brain view; ~140,000 neurons from the static soma buffer plus an activity buffer rewritten each frame
//!   from the sim snapshot (`brainview.rs` does the writing).
//! * [`RainMaterial`]: rain that exists only on the GPU (positions come from a hash of the index and the clock).

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

/// A mesh of `n` camera-facing quads. Each vertex's position is `(corner.x, corner.y, quad index)`; the material's shader turns that into
/// a billboard. The mesh has no real extent, so entities using it must carry `NoFrustumCulling`.
pub fn quad_mesh(n: usize) -> Mesh {
    let corners = [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let mut positions = Vec::with_capacity(n * 4);
    let mut indices = Vec::with_capacity(n * 6);
    for i in 0..n {
        for c in corners {
            positions.push([c[0], c[1], i as f32]);
        }
        let b = (i * 4) as u32;
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_indices(Indices::U32(indices))
}

// --- generic points -------------------------------------------------------------------------------------------------------------
/// One billboard: world position and radius, colour and brightness.
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuPoint {
    pub pos_size: [f32; 4],
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct PointsParams {
    pub count: u32,
    pub size_scale: f32,
    pub time: f32,
    pub softness: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct PointsMaterial {
    #[uniform(0)]
    pub params: PointsParams,
    #[storage(1, read_only)]
    pub points: Handle<ShaderBuffer>,
}

impl Material for PointsMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://kick_the_fly/shaders/points.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://kick_the_fly/shaders/points.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
}

/// A pool of up to `capacity` glowing points drawn in one call. Fill `points` each frame; [`flush_particles`] uploads it.
#[derive(Component)]
pub struct ParticleCloud {
    pub capacity: usize,
    pub points: Vec<GpuPoint>,
    buffer: Handle<ShaderBuffer>,
    material: Handle<PointsMaterial>,
}

impl ParticleCloud {
    pub fn push(&mut self, pos: Vec3, radius: f32, color: [f32; 3], brightness: f32) {
        if self.points.len() < self.capacity {
            self.points.push(GpuPoint { pos_size: [pos.x, pos.y, pos.z, radius], color: [color[0], color[1], color[2], brightness] });
        }
    }
}

/// Spawn a particle cloud entity. Returns it so the caller can find it again.
pub fn spawn_particle_cloud(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<PointsMaterial>,
    buffers: &mut Assets<ShaderBuffer>,
    capacity: usize,
    softness: f32,
) -> Entity {
    let buffer = buffers.add(ShaderBuffer::with_size(capacity * std::mem::size_of::<GpuPoint>(), RenderAssetUsages::default()));
    let material =
        materials.add(PointsMaterial { params: PointsParams { count: 0, size_scale: 1.0, time: 0.0, softness }, points: buffer.clone() });
    commands
        .spawn((
            Mesh3d(meshes.add(quad_mesh(capacity))),
            MeshMaterial3d(material.clone()),
            Transform::default(),
            bevy::camera::visibility::NoFrustumCulling,
            ParticleCloud { capacity, points: Vec::new(), buffer, material },
        ))
        .id()
}

/// Upload every cloud's points for this frame (a plain buffer write; the draw itself does not change).
pub fn flush_particles(
    mut clouds: Query<&mut ParticleCloud>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<PointsMaterial>>,
    time: Res<Time>,
) {
    for mut c in &mut clouds {
        let n = c.points.len();
        if let Some(mut b) = buffers.get_mut(&c.buffer) {
            let bytes: &[u8] = bytemuck::cast_slice(&c.points);
            let mut data = vec![0u8; c.capacity * std::mem::size_of::<GpuPoint>()];
            data[..bytes.len()].copy_from_slice(bytes);
            b.data = Some(data);
        }
        if let Some(mut m) = materials.get_mut(&c.material) {
            m.params.count = n as u32;
            m.params.time = time.elapsed_secs();
        }
        c.points.clear();
    }
}

// --- the brain ------------------------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct BrainParams {
    pub count: u32,
    pub size: f32,
    pub base_glow: f32,
    pub gain: f32,
    pub rot: Vec4,
    /// The tilt (cos, sin) and the shift of the turned, zoomed cloud in the camera's plane.
    pub tilt: Vec4,
    /// Imaging mode: the four stops of the palette (`imaging.LUTS`), and `x` = 1 to colour by them instead of by population.
    pub lut: [Vec4; 4],
    pub mode: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct BrainMaterial {
    #[uniform(0)]
    pub params: BrainParams,
    #[storage(1, read_only)]
    pub soma: Handle<ShaderBuffer>,
    #[storage(2, read_only)]
    pub activity: Handle<ShaderBuffer>,
}

impl Material for BrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://kick_the_fly/shaders/brain.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://kick_the_fly/shaders/brain.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
}

// --- rain -----------------------------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct RainParams {
    pub count: u32,
    pub time: f32,
    pub intensity: f32,
    pub fall_speed: f32,
    pub wind: Vec4,
    pub center: Vec4,
    pub half_extent: Vec4,
    pub color: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct RainMaterial {
    #[uniform(0)]
    pub params: RainParams,
}

impl Material for RainMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://kick_the_fly/shaders/rain.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://kick_the_fly/shaders/rain.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
}

/// Registers the three materials and their shaders and flushes the particle clouds each frame.
pub struct GpuPlugin;

impl Plugin for GpuPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/points.wgsl");
        embedded_asset!(app, "shaders/brain.wgsl");
        embedded_asset!(app, "shaders/rain.wgsl");
        app.add_plugins((
            MaterialPlugin::<PointsMaterial>::default(),
            MaterialPlugin::<BrainMaterial>::default(),
            MaterialPlugin::<RainMaterial>::default(),
        ))
        .add_systems(PostUpdate, flush_particles);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_quad_mesh_carries_each_quads_index_in_its_vertices() {
        let m = quad_mesh(3);
        let pos = m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().as_float3().unwrap();
        assert_eq!(pos.len(), 12);
        assert!(pos[8..12].iter().all(|p| p[2] == 2.0), "the third quad is index 2");
        assert_eq!(m.indices().unwrap().len(), 18);
        assert_eq!(std::mem::size_of::<GpuPoint>(), 32, "two vec4s, std430-compatible");
    }
}
