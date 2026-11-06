//! The window remembers where the player left it: size and (where the window system says) position, written to `config.toml`'s `[window]`
//! a moment after the last resize or move, windowed mode only. Fullscreen is the existing `graphics.fullscreen` setting.

use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowMode, WindowMoved, WindowResized};
use ktf_data::config::WindowGeom;

use crate::menu_ui::UserConfig;

/// Seconds without a resize or move before the new geometry is written.
const SETTLE_S: f32 = 1.0;

/// The saved geometry as `Window` settings for the first frame (size, and the position when one was saved).
pub fn apply_saved(window: &mut Window, geom: Option<WindowGeom>) {
    let Some(g) = geom else { return };
    window.resolution.set(g.width as f32, g.height as f32);
    if let Some((x, y)) = g.pos {
        window.position = bevy::window::WindowPosition::At(IVec2::new(x, y));
    }
}

fn remember(
    mut resized: MessageReader<WindowResized>,
    mut moved: MessageReader<WindowMoved>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut cfg: ResMut<UserConfig>,
    real: Res<Time<Real>>,
    mut changed_at: Local<Option<f32>>,
) {
    let now = real.elapsed_secs();
    if resized.read().count() + moved.read().count() > 0 {
        *changed_at = Some(now);
    }
    let Some(t) = *changed_at else { return };
    if now - t < SETTLE_S {
        return;
    }
    *changed_at = None;
    if !matches!(window.mode, WindowMode::Windowed) {
        return;
    }
    let pos = match window.position {
        bevy::window::WindowPosition::At(p) => Some((p.x, p.y)),
        _ => None,
    };
    let geom = WindowGeom { width: window.resolution.width().round() as u32, height: window.resolution.height().round() as u32, pos };
    if cfg.0.window != Some(geom) && geom.width >= 200 && geom.height >= 200 {
        cfg.0.window = Some(geom);
        cfg.0.dirty = true;
        cfg.0.save(env!("CARGO_PKG_VERSION"));
    }
}

pub struct WindowStatePlugin;

impl Plugin for WindowStatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, remember);
    }
}
