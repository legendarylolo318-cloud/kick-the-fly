//! The fly's-eye view (F4). Python: `game/fly_eye.py`. VISUAL ONLY: it is not fed to the brain (the brain's looming input is the
//! angular-expansion measure in `senses.rs`), exactly as in the Python game.
//!
//! A second camera sits at the fly's head, looking where the fly faces, with a very wide field of view, and renders into a small texture.
//! A UI material (`shaders/eye.wgsl`) shows that texture as two discs of hexagonal ommatidia, one colour per hexagon, in the fly's
//! spectrum (UV-blue-green, almost no red), darker toward the rim. Everything is GPU: no pixel is read back.

use bevy::asset::embedded_asset;
use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType, TextureFormat};
use bevy::shader::ShaderRef;
use bevy::ui_render::UiMaterialPlugin;
use bevy::ui_render::ui_material::{MaterialNode, UiMaterial};

use crate::fly::{Fly, FlyPart, Part};
use crate::player::PlayerCamera;

const SIZE: u32 = 384;

#[derive(Debug, Clone, Copy, ShaderType)]
pub struct EyeParams {
    pub cells: f32,
    pub time: f32,
    pub fade: f32,
    pub pad: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct EyeMaterial {
    #[uniform(0)]
    pub params: EyeParams,
    #[texture(1)]
    #[sampler(2)]
    pub view: Handle<Image>,
}

impl UiMaterial for EyeMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://kick_the_fly/shaders/eye.wgsl".into()
    }
}

#[derive(Resource, Default)]
pub struct FlyEye {
    pub on: bool,
    fade: f32,
}

#[derive(Component)]
struct EyeCamera;
#[derive(Component)]
struct EyeScreen(Handle<EyeMaterial>);

fn setup(mut eye: ResMut<FlyEye>, mut commands: Commands, mut images: ResMut<Assets<Image>>, mut mats: ResMut<Assets<EyeMaterial>>) {
    eye.on = std::env::var_os("KTF_EYE").is_some();
    let image = images.add(Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba8UnormSrgb, None));
    commands.spawn((
        EyeCamera,
        Camera3d::default(),
        Camera { order: -1, is_active: false, ..default() },
        RenderTarget::Image(image.clone().into()),
        Projection::Perspective(PerspectiveProjection { fov: 130f32.to_radians(), near: 0.02, far: 100.0, ..default() }),
        Transform::default(),
    ));
    let m = mats.add(EyeMaterial { params: EyeParams { cells: 34.0, time: 0.0, fade: 0.0, pad: 0.0 }, view: image });
    commands.spawn((
        EyeScreen(m.clone()),
        MaterialNode(m),
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            top: px(0),
            width: percent(100),
            height: percent(100),
            display: Display::None,
            ..default()
        },
        ZIndex(50),
        BackgroundColor(Color::NONE),
    ));
}

fn update(
    act: Res<crate::actions::Actions>,
    time: Res<Time<Real>>,
    mut eye: ResMut<FlyEye>,
    flies: Query<&Fly>,
    parts: Query<(&FlyPart, &Transform), Without<EyeCamera>>,
    mut cam: Single<(&mut Camera, &mut Transform), (With<EyeCamera>, Without<PlayerCamera>)>,
    mut screen: Single<(&EyeScreen, &mut Node)>,
    mut mats: ResMut<Assets<EyeMaterial>>,
) {
    if act.just("fly_eye") {
        eye.on = !eye.on;
    }
    let target = if eye.on { 1.0 } else { 0.0 };
    eye.fade += (target - eye.fade) * (time.delta_secs() * 8.0).min(1.0);
    let active = eye.fade > 0.01;
    cam.0.is_active = active;
    screen.1.display = if active { Display::Flex } else { Display::None };
    if let Some(mut m) = mats.get_mut(&screen.0.0) {
        m.params.fade = eye.fade;
        m.params.time = time.elapsed_secs();
    }
    // the camera rides the head of the first living fly, looking where the head looks
    if let Some(f) = flies.iter().find(|f| !f.dead())
        && let Ok((_, head)) = parts.get(f.parts[Part::Head.index()])
        && let Ok((_, thx)) = parts.get(f.parts[Part::Thorax.index()])
    {
        let fwd = thx.rotation * Vec3::X;
        cam.1.translation = head.translation + fwd * 0.07 + Vec3::Y * 0.04;
        cam.1.look_to(fwd, Vec3::Y);
    }
}

pub struct FlyEyePlugin;

impl Plugin for FlyEyePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/eye.wgsl");
        app.add_plugins(UiMaterialPlugin::<EyeMaterial>::default())
            .init_resource::<FlyEye>()
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }
}
