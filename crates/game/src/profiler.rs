//! The profiler overlay (F3): FPS, frame time, simulation steps per second, how stale the simulation's snapshot is, and what the GPU is
//! drawing. Python: `core/profiler.py`. Off at every launch.
//!
//! * frame time: the render loop's own (smoothed and worst of the last second), with a bar graph of the last 120 frames.
//! * sim steps/s and sim lag: the sim thread's measured rate against its fixed 200/s, steps it had to drop to catch up, and the age of the
//!   snapshot the renderer is reading (how long ago the sim thread published it).
//! * draws: Bevy batches meshes that share a mesh and a material into one draw, so the honest count is *batches*: the distinct
//!   (mesh, material) pairs among the visible meshes. The neuron cloud, the particles and the rain are each one batch however many things
//!   they draw; the number of points they put on screen is listed beside the batch count. Bevy does not expose the GPU driver's own draw
//!   call counter, so this is derived from what is visible (it ignores shadow-map passes).
//! * GPU: per-pass GPU time from Bevy's render diagnostics when the adapter supports timestamp queries.

use std::collections::{HashSet, VecDeque};

use bevy::diagnostic::{Diagnostic, DiagnosticPath, DiagnosticsStore, EntityCountDiagnosticsPlugin, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;

use crate::app::LatestSnapshot;
use crate::brainview::BrainView;
use crate::config::Settings;
use crate::gpu::{ParticleCloud, PointsMaterial};

#[derive(Resource, Default)]
pub struct ProfilerStats {
    frame_ms: VecDeque<f32>,
    /// CPU time of the named parts of a frame, in ms, smoothed (Python's `PROF` sections).
    pub cpu: Vec<(&'static str, f32)>,
    pub batches: usize,
    pub mesh_entities: usize,
    pub points_on_screen: usize,
}

impl ProfilerStats {
    /// A part of the frame took the time since `t0`.
    pub fn record(&mut self, name: &'static str, t0: std::time::Instant) {
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        match self.cpu.iter_mut().find(|c| c.0 == name) {
            Some(c) => c.1 += (ms - c.1) * 0.05,
            None => self.cpu.push((name, ms)),
        }
    }

    fn cpu_line(&self) -> String {
        self.cpu.iter().map(|(n, v)| format!("{n} {v:.2} ms")).collect::<Vec<_>>().join("   ")
    }
}

#[derive(Component)]
struct ProfilerText;

#[derive(Component)]
struct ProfilerBars;

#[derive(Component)]
struct Bar(usize);

fn setup(mut commands: Commands) {
    commands
        .spawn((
            ProfilerBars,
            Node {
                position_type: PositionType::Absolute,
                top: px(150),
                left: px(12),
                width: px(240),
                height: px(46),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::FlexEnd,
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            GlobalZIndex(600), // over the HUD canvas (500), as Python draws its card last
        ))
        .with_children(|p| {
            for i in 0..120 {
                p.spawn((Bar(i), Node { width: px(2), height: px(2), ..default() }, BackgroundColor(Color::srgb(0.3, 0.9, 0.5))));
            }
        });
    commands.spawn((
        ProfilerText,
        Text::new(""),
        TextFont { font_size: FontSize::Px(14.0), ..default() },
        TextColor(Color::srgb(0.85, 0.95, 0.85)),
        Node { position_type: PositionType::Absolute, top: px(200), left: px(12), display: Display::None, ..default() },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        GlobalZIndex(600),
    ));
}

fn toggle(act: Res<crate::actions::Actions>, mut settings: ResMut<Settings>) {
    if act.just("profiler") {
        settings.show_profiler = !settings.show_profiler;
    }
}

/// Count what is on screen: the batches Bevy will draw (distinct mesh + material among visible meshes) and the points the instanced
/// draws put up.
#[allow(clippy::type_complexity)]
fn survey(
    time: Res<Time<Real>>,
    mut stats: ResMut<ProfilerStats>,
    visible: Query<(&ViewVisibility, &Mesh3d, Option<&MeshMaterial3d<StandardMaterial>>, Option<&MeshMaterial3d<PointsMaterial>>)>,
    clouds: Query<&ParticleCloud>,
    view: Res<BrainView>,
    snap: Res<LatestSnapshot>,
    settings: Res<Settings>,
    mut last_print: Local<f32>,
) {
    stats.frame_ms.push_back(time.delta_secs() * 1000.0);
    while stats.frame_ms.len() > 120 {
        stats.frame_ms.pop_front();
    }
    // KTF_DEBUG=1: the same numbers on the terminal every two seconds
    if std::env::var_os("KTF_DEBUG").is_some() && time.elapsed_secs() - *last_print >= 2.0 {
        *last_print = time.elapsed_secs();
        let mean = stats.frame_ms.iter().sum::<f32>() / stats.frame_ms.len().max(1) as f32;
        eprintln!("profiler: frame {mean:.2} ms ({:.0} fps)   cpu: {}", 1000.0 / mean.max(1e-3), stats.cpu_line());
    }
    if !settings.show_profiler {
        return;
    }
    let mut seen: HashSet<(AssetId<Mesh>, u128)> = HashSet::new();
    let mut n = 0;
    for (vis, mesh, std_mat, pts_mat) in &visible {
        if !vis.get() {
            continue;
        }
        n += 1;
        let m = std_mat.map(|m| m.0.id().to_string()).or_else(|| pts_mat.map(|m| m.0.id().to_string())).unwrap_or_default();
        seen.insert((mesh.0.id(), m.bytes().fold(0u128, |a, b| a.wrapping_mul(131).wrapping_add(u128::from(b)))));
    }
    stats.mesh_entities = n;
    stats.batches = seen.len();
    let neurons = if view.built() && view.visible() { snap.0.as_deref().map_or(0, |s| s.neurons) } else { 0 };
    stats.points_on_screen = neurons + clouds.iter().map(|c| c.points.len()).sum::<usize>();
}

fn show(
    settings: Res<Settings>,
    stats: Res<ProfilerStats>,
    diagnostics: Res<DiagnosticsStore>,
    snap: Res<LatestSnapshot>,
    mut text: Single<(&mut Text, &mut Node), (With<ProfilerText>, Without<ProfilerBars>)>,
    mut bars_node: Single<&mut Node, (With<ProfilerBars>, Without<ProfilerText>)>,
    mut bars: Query<(&Bar, &mut Node), (Without<ProfilerBars>, Without<ProfilerText>)>,
    mut colors: Query<(&Bar, &mut BackgroundColor)>,
) {
    let on = settings.show_profiler;
    let disp = if on { Display::Flex } else { Display::None };
    text.1.display = disp;
    bars_node.display = disp;
    if !on {
        return;
    }
    let fps = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS).and_then(Diagnostic::smoothed).unwrap_or(0.0);
    let ft = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME).and_then(Diagnostic::smoothed).unwrap_or(0.0);
    let worst = stats.frame_ms.iter().cloned().fold(0.0f32, f32::max);
    let entities = diagnostics.get(&EntityCountDiagnosticsPlugin::ENTITY_COUNT).and_then(Diagnostic::value).unwrap_or(0.0);
    let mut s = format!("render  {fps:5.1} fps   frame {ft:5.2} ms   worst {worst:5.2} ms\n");
    if let Some(sn) = snap.0.as_deref() {
        let age_ms = sn.published_at.map_or(0.0, |t| t.elapsed().as_secs_f32() * 1000.0);
        s += &format!(
            "sim     {:5.1} steps/s of 200   x{:.2}   dropped {}   lag {:5.1} ms\n",
            sn.steps_per_s, sn.speed, sn.steps_dropped, age_ms
        );
        s += &format!("brain   {} neurons, {} spiking, {} lit\n", sn.neurons, sn.active_count, sn.active_count);
    } else {
        s += "sim     waiting for the brain\n";
    }
    s += &format!("draws   {} batches ({} visible meshes)   {} entities\n", stats.batches, stats.mesh_entities, entities as u64);
    s += &format!("points  {} drawn by instanced shaders (one draw each)\n", stats.points_on_screen);
    s += &format!("cpu     {}\n", stats.cpu_line());
    let mut gpu: Vec<(String, f64)> = diagnostics
        .iter()
        .filter(|d| d.path().as_str().ends_with("/elapsed_gpu"))
        .filter_map(|d| {
            d.smoothed().map(|v| (d.path().as_str().trim_end_matches("/elapsed_gpu").rsplit('/').next().unwrap_or("").to_string(), v))
        })
        .collect();
    gpu.sort_by(|a, b| b.1.total_cmp(&a.1));
    if gpu.is_empty() {
        s += "gpu     (no timestamp queries on this adapter)";
    } else {
        s += &format!("gpu     {}", gpu.iter().take(4).map(|(n, v)| format!("{n} {v:.2} ms")).collect::<Vec<_>>().join("   "));
    }
    text.0.0 = s;
    // the frame-time bar graph: 0..33 ms, red above 20
    let n = stats.frame_ms.len();
    for (b, mut node) in &mut bars {
        let idx = (b.0 + n).saturating_sub(120);
        let ms = if b.0 + n >= 120 { stats.frame_ms.get(idx).copied().unwrap_or(0.0) } else { 0.0 };
        node.height = px((ms / 33.0 * 44.0).clamp(1.0, 44.0));
    }
    for (b, mut c) in &mut colors {
        let idx = (b.0 + n).saturating_sub(120);
        let ms = if b.0 + n >= 120 { stats.frame_ms.get(idx).copied().unwrap_or(0.0) } else { 0.0 };
        c.0 = if ms > 20.0 {
            Color::srgb(0.95, 0.3, 0.25)
        } else if ms > 17.5 {
            Color::srgb(0.95, 0.8, 0.3)
        } else {
            Color::srgb(0.3, 0.9, 0.5)
        };
    }
}

pub struct ProfilerPlugin;

impl Plugin for ProfilerPlugin {
    fn build(&self, app: &mut App) {
        let _ = DiagnosticPath::new("kick_the_fly");
        app.add_plugins((FrameTimeDiagnosticsPlugin::default(), EntityCountDiagnosticsPlugin::default(), RenderDiagnosticsPlugin))
            .init_resource::<ProfilerStats>()
            .add_systems(Startup, setup)
            .add_systems(Update, (toggle, survey, show).chain());
    }
}
