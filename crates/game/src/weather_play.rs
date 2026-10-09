//! Rain, gusts and storms in the world (outdoor arenas): the rain itself is drawn on the GPU, the lightning drives the scene's light.
//! The rules are `weather.rs` (ported from `game/weather.py`); what the weather does to a fly's neurons is in `environment.rs`.
//!
//! The rain is one mesh of 6,000 streaks and one draw call; each streak's position is a hash of its index and the clock, evaluated in the
//! vertex shader (`shaders/rain.wgsl`), so no drop exists on the CPU. The box of rain follows the camera.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::prelude::*;

use crate::arena::Atmosphere;
use crate::config::{Arena, Settings};
use crate::feedback::Sfx;
use crate::gpu::{RainMaterial, RainParams, quad_mesh};
use crate::player::PlayerCamera;
use crate::weather::Weather;
use crate::world;

pub const STREAKS: usize = 6000;

#[derive(Resource)]
pub struct WeatherRes(pub Weather);

impl Default for WeatherRes {
    fn default() -> Self {
        Self(Weather::new(3))
    }
}

#[derive(Component)]
struct RainCloud(Handle<RainMaterial>);

fn spawn_rain(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<RainMaterial>>) {
    let material = materials.add(RainMaterial {
        params: RainParams {
            count: STREAKS as u32,
            time: 0.0,
            intensity: 0.0,
            fall_speed: 9.0,
            wind: Vec4::ZERO,
            center: Vec4::new(0.0, 0.0, 0.0, 0.45),
            half_extent: Vec4::new(7.0, 7.0, 7.0, 0.008),
            color: Vec4::new(0.7, 0.78, 0.92, 0.5),
        },
    });
    commands.spawn((RainCloud(material.clone()), Mesh3d(meshes.add(quad_mesh(STREAKS))), MeshMaterial3d(material), Transform::default(), NoFrustumCulling, Visibility::Hidden));
}

/// Advance the weather and publish the sky: how dark, how bright the lightning, where the sun is.
#[allow(clippy::too_many_arguments)]
fn update(
    time: Res<Time<Virtual>>,
    settings: Res<Settings>,
    arena: Res<Arena>,
    mut wx: ResMut<WeatherRes>,
    mut atmo: ResMut<Atmosphere>,
    mut sfx: MessageWriter<Sfx>,
    mut rain: Single<(&RainCloud, &mut Visibility)>,
    mut materials: ResMut<Assets<RainMaterial>>,
    cam: Single<&Transform, With<PlayerCamera>>,
) {
    let outdoor = world::is_outdoor(arena.name());
    let now = time.elapsed_secs();
    let dt = time.delta_secs().clamp(0.0, 0.1);
    if outdoor {
        wx.0.update(dt, settings.rain, settings.gust_hz, settings.storm);
        for _ in 0..wx.0.thunder_due() {
            sfx.write(Sfx::vol("boom", 0.55));
        }
    }
    let day_s = settings.day_night_s;
    let (az, el) = world::sun_now(now, day_s);
    atmo.sun_dir = world::sun_direction(az, el);
    if outdoor {
        let day = 0.12 + 0.88 * world::dusk(el);
        atmo.day = day * (1.0 - wx.0.darkness());
        atmo.flash = wx.0.screen_flash(settings.reduced_flashing);
    } else {
        atmo.day = 1.0;
        atmo.flash = 0.0;
    }
    let (cloud, vis) = &mut *rain;
    let raining = outdoor && wx.0.rain > 0.0;
    **vis = if raining { Visibility::Inherited } else { Visibility::Hidden };
    if let Some(mut m) = materials.get_mut(&cloud.0) {
        let (ws, wd) = wx.0.wind(settings.wind_speed, settings.wind_dir_deg);
        let wv = world::wind_vector(wd, ws);
        m.params.time = now;
        m.params.intensity = wx.0.rain;
        m.params.wind = Vec4::new(wv.x, 0.0, wv.z, 0.0);
        m.params.center = Vec4::new(cam.translation.x, cam.translation.y, cam.translation.z, 0.45);
    }
}

pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WeatherRes>().add_systems(Startup, spawn_rain).add_systems(Update, update);
    }
}
