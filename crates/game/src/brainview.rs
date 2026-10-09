//! The brain on screen: every neuron of the connectome drawn as a glowing point, lit by its own firing, in one draw call.
//! Python: `game/kick_the_fly.py:BrainView` (the CPU-rasterised panel) and `game/gpu_brainview.py` (the optional GPU variant); this is the
//! GPU one only.
//!
//! Data flow, each frame: the newest sim snapshot (per-neuron EMA rates and the spike list) is turned into one `f32` of activity per neuron
//! and written to a storage buffer; the vertex shader (`shaders/brain.wgsl`) reads it next to the static layout buffer. Nothing is
//! rasterised or sorted on the CPU, and the cost on the CPU side is one pass over the activity array.
//!
//! The panel is a second camera on its own render layer with a viewport over the right side of the window. `B` makes it the whole window
//! (drag to turn the brain), `V` cycles solid, see-through, faint, hidden.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::camera::{ClearColorConfig, RenderTarget};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureFormat};
use bevy::render::storage::ShaderBuffer;
use bevy::window::PrimaryWindow;

use crate::app::LatestSnapshot;
use crate::gpu::{BrainMaterial, BrainParams, quad_mesh};
use crate::player::MouseLook;

pub const BRAIN_LAYER: usize = 1;
/// Spikes per step at which a neuron shows as fully lit (30 Hz at the 5 ms step).
const FULL_RATE: f32 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelMode {
    Solid,
    SeeThrough,
    Faint,
    Hidden,
}

impl PanelMode {
    pub fn next(self) -> Self {
        match self {
            PanelMode::Solid => PanelMode::SeeThrough,
            PanelMode::SeeThrough => PanelMode::Faint,
            PanelMode::Faint => PanelMode::Hidden,
            PanelMode::Hidden => PanelMode::Solid,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PanelMode::Solid => "solid",
            PanelMode::SeeThrough => "see-through",
            PanelMode::Faint => "faint",
            PanelMode::Hidden => "hidden",
        }
    }

    fn gain(self) -> f32 {
        match self {
            PanelMode::Solid => 1.0,
            PanelMode::SeeThrough => 0.7,
            PanelMode::Faint => 0.3,
            PanelMode::Hidden => 0.0,
        }
    }
}

#[derive(Resource)]
pub struct BrainView {
    pub mode: PanelMode,
    pub big: bool,
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub auto_rotate: bool,
    pub lit: usize,
    /// While the kill cam plays, these per-neuron rates (spikes per step) are shown instead of the live brain's, and `highlight` neurons pulse.
    pub replay: Option<Vec<f32>>,
    pub highlight: Vec<u32>,
    activity: Vec<f32>,
    built: bool,
}

impl Default for BrainView {
    fn default() -> Self {
        Self {
            mode: PanelMode::Solid,
            big: false,
            yaw: 0.0,
            pitch: 0.0,
            zoom: 0.8,
            auto_rotate: true,
            lit: 0,
            replay: None,
            highlight: Vec::new(),
            activity: Vec::new(),
            built: false,
        }
    }
}

impl BrainView {
    pub fn built(&self) -> bool {
        self.built
    }

    /// Is the cloud drawn at all right now?
    pub fn visible(&self) -> bool {
        self.big || self.mode != PanelMode::Hidden
    }
}

#[derive(Component)]
struct BrainCamera;

/// The UI node that shows the brain camera's picture (the camera renders into an image, so the HUD can sit on top of the panel).
#[derive(Component)]
struct PanelNode;

#[derive(Resource)]
struct PanelImage(Handle<Image>);

#[derive(Component)]
struct BrainCloud {
    material: Handle<BrainMaterial>,
    activity: Handle<ShaderBuffer>,
}

fn setup(mut commands: Commands, mut view: ResMut<BrainView>, mut images: ResMut<Assets<Image>>) {
    view.big = std::env::var_os("KTF_BIG_BRAIN").is_some();
    let image = images.add(Image::new_target_texture(512, 512, TextureFormat::Rgba8UnormSrgb, None));
    commands.insert_resource(PanelImage(image.clone()));
    commands.spawn((
        BrainCamera,
        Camera3d::default(),
        Camera { order: -2, clear_color: ClearColorConfig::Custom(Color::srgb(0.015, 0.02, 0.035)), is_active: true, ..default() },
        RenderTarget::Image(image.clone().into()),
        Projection::Perspective(PerspectiveProjection { fov: 32f32.to_radians(), near: 0.05, far: 50.0, ..default() }),
        Transform::from_xyz(0.0, 0.0, 3.4).looking_at(Vec3::ZERO, Vec3::Y),
        RenderLayers::layer(BRAIN_LAYER),
    ));
    commands.spawn((
        PanelNode,
        ImageNode::new(image),
        Node { position_type: PositionType::Absolute, left: px(0), top: px(0), width: px(300), height: px(300), ..default() },
        GlobalZIndex(400),
    ));
}

/// Build the cloud when the first snapshot with the layout arrives.
fn build_cloud(
    mut commands: Commands,
    mut view: ResMut<BrainView>,
    snap: Res<LatestSnapshot>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BrainMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    if view.built {
        return;
    }
    let Some(info) = snap.0.as_deref().and_then(|s| s.info.as_deref()) else { return };
    let n = info.layout.len();
    if n == 0 {
        return;
    }
    // soma buffer: xyz and the population (-1: not placed, the shader skips it)
    let mut bytes = Vec::with_capacity(n * 16);
    for (p, &pop) in info.layout.iter().zip(&info.population) {
        let placed = p[0].is_finite();
        let rec = if placed { [p[0], p[1], p[2], if pop == 255 { 9.0 } else { f32::from(pop) }] } else { [0.0, 0.0, 0.0, -1.0] };
        for v in rec {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let soma = buffers.add(ShaderBuffer::new(&bytes, RenderAssetUsages::RENDER_WORLD));
    let activity = buffers.add(ShaderBuffer::new(&vec![0u8; n * 4], RenderAssetUsages::default()));
    let material = materials.add(BrainMaterial {
        params: BrainParams { count: n as u32, size: 0.0030, base_glow: 0.012, gain: 0.7, rot: Vec4::new(1.0, 0.0, 1.0, 0.0) },
        soma,
        activity: activity.clone(),
    });
    commands.spawn((
        Mesh3d(meshes.add(quad_mesh(n))),
        MeshMaterial3d(material.clone()),
        Transform::default(),
        NoFrustumCulling,
        RenderLayers::layer(BRAIN_LAYER),
        BrainCloud { material, activity },
    ));
    view.activity = vec![0.0; n];
    view.built = true;
}

/// One pass over the neurons: decay, take the EMA rate, flash the neurons that just spiked, write the buffer.
fn update_activity(
    mut view: ResMut<BrainView>,
    snap: Res<LatestSnapshot>,
    cloud: Option<Single<&BrainCloud>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<BrainMaterial>>,
    time: Res<Time>,
) {
    let (Some(cloud), Some(s)) = (cloud, snap.0.as_deref()) else { return };
    if s.rates.len() != view.activity.len() {
        return;
    }
    let decay = (-time.delta_secs() * 9.0).exp();
    let v = &mut *view;
    if let Some(replay) = v.replay.take() {
        // the kill cam: the recorded rates, no spike flashes, the risers pulsing
        if replay.len() == v.activity.len() {
            for (a, &r) in v.activity.iter_mut().zip(&replay) {
                *a = (r / FULL_RATE).min(1.0) * 0.8;
            }
            let pulse = 0.75 + 0.25 * (time.elapsed_secs() * 6.0).sin();
            for &i in &v.highlight {
                v.activity[i as usize] = pulse;
            }
        }
        v.replay = Some(replay);
    } else {
        for (a, &r) in v.activity.iter_mut().zip(&s.rates) {
            *a = (*a * decay).max((r / FULL_RATE).min(1.0) * 0.8);
        }
        for &i in &s.spikes {
            v.activity[i as usize] = 1.0;
        }
    }
    v.lit = s.active_count;
    if std::env::var_os("KTF_DEBUG").is_some() && s.step % 200 == 0 {
        let mean = v.activity.iter().sum::<f32>() / v.activity.len() as f32;
        let rmean = s.rates.iter().sum::<f32>() / s.rates.len() as f32;
        eprintln!("brainview: mean activity {mean:.4}, mean rate {rmean:.5}, spikes {}", s.spikes.len());
    }
    if let Some(mut b) = buffers.get_mut(&cloud.activity) {
        b.data = Some(v.activity.iter().flat_map(|f| f.to_le_bytes()).collect());
    }
    if let Some(mut m) = materials.get_mut(&cloud.material) {
        m.params.gain = 0.7 * v.mode.gain();
        m.params.base_glow = 0.015 + 0.03 * v.mode.gain();
        let (sn, c) = v.yaw.sin_cos();
        m.params.rot = Vec4::new(c, sn, v.zoom, 0.0);
        m.params.size = 0.0030 * if v.big { 1.3 } else { 1.0 };
    }
}

#[allow(clippy::too_many_arguments)]
fn control(
    act: Res<crate::actions::Actions>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    look: Res<MouseLook>,
    mut view: ResMut<BrainView>,
    time: Res<Time>,
) {
    if act.just("big_view") {
        view.big = !view.big;
    }
    if act.just("panel") {
        view.mode = view.mode.next();
    }
    if view.big && !look.captured && mouse.pressed(MouseButton::Left) {
        view.yaw += motion.delta.x * 0.01;
        view.auto_rotate = false;
    } else if view.auto_rotate {
        view.yaw += time.delta_secs() * 0.12;
    }
    if keys.just_pressed(KeyCode::KeyR) && view.big {
        view.yaw = 0.0;
        view.auto_rotate = true;
    }
}

/// The panel's rectangle: the right third of the window, or all of it in the big view. The picture is rendered at the window's own
/// resolution, resized when the panel changes size.
#[allow(clippy::too_many_arguments)]
fn layout(
    mut cam: Single<&mut Camera, With<BrainCamera>>,
    view: Res<BrainView>,
    window: Single<&Window, With<PrimaryWindow>>,
    panel: Res<PanelImage>,
    mut images: ResMut<Assets<Image>>,
    mut node: Single<(&mut Node, &mut ImageNode, &mut Visibility), With<PanelNode>>,
    hud: Res<crate::hud_canvas::HudLayout>,
) {
    let (lw, lh) = (window.width(), window.height());
    if lw < 2.0 || lh < 2.0 {
        return;
    }
    let shown = view.built && view.mode != PanelMode::Hidden || view.big;
    cam.is_active = shown;
    *node.2 = if shown { Visibility::Inherited } else { Visibility::Hidden };
    // the panel's picture sits where the HUD leaves a hole for it: (play_w + 12, 54), 356 x 193 HUD units (`VIEW_SIZES["panel"]`)
    let sf = window.scale_factor();
    let k = hud.0.scale / sf;
    let (w, h) = if view.big { (lw, lh) } else { (356.0 * k, 193.0 * k) };
    node.0.position_type = PositionType::Absolute;
    node.0.right = Val::Auto;
    if view.big {
        node.0.left = px(0);
        node.0.top = px(0);
    } else {
        node.0.left = px((hud.0.play_w + 12) as f32 * k);
        node.0.top = px(54.0 * k);
    }
    node.0.width = px(w);
    node.0.height = px(h);
    let alpha = if view.big {
        1.0
    } else {
        match view.mode {
            PanelMode::Solid => 1.0,
            PanelMode::SeeThrough => 0.6,
            PanelMode::Faint => 0.28,
            PanelMode::Hidden => 0.0,
        }
    };
    node.1.color = Color::srgba(1.0, 1.0, 1.0, alpha);
    let (pw, ph) = (((w * sf) as u32).clamp(64, 2400), ((h * sf) as u32).clamp(64, 2400));
    if let Some(img) = images.get(&panel.0)
        && (img.width() != pw || img.height() != ph)
        && let Some(mut img) = images.get_mut(&panel.0)
    {
        img.resize(Extent3d { width: pw, height: ph, depth_or_array_layers: 1 });
    }
}

pub struct BrainViewPlugin;

impl Plugin for BrainViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BrainView>()
            .add_systems(Startup, setup)
            .add_systems(Update, (control, build_cloud, update_activity, layout).chain());
    }
}
