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

/// Whether at least `MIN_VISIBLE` px of the title-bar band of a window at `pos` (physical) lands on one of `monitors` (position, size).
pub fn on_some_monitor(pos: IVec2, width: u32, monitors: &[(IVec2, UVec2)]) -> bool {
    const MIN_VISIBLE: i32 = 64;
    monitors.iter().any(|&(mp, ms)| {
        let (l, r) = (pos.x.max(mp.x), (pos.x + width as i32).min(mp.x + ms.x as i32));
        let top_ok = pos.y >= mp.y - 8 && pos.y < mp.y + ms.y as i32 - MIN_VISIBLE;
        r - l >= MIN_VISIBLE && top_ok
    })
}

/// Once the monitors are known: a saved position that is on none of them (a monitor was unplugged, the layout changed) is dropped and the
/// window is centred, and a saved size bigger than the monitor is shrunk to fit, so the window never opens out of reach.
fn keep_on_screen(monitors: Query<&bevy::window::Monitor>, mut window: Single<&mut Window, With<PrimaryWindow>>, mut done: Local<u8>) {
    if *done > 0 || monitors.is_empty() {
        return;
    }
    *done = 1;
    let all: Vec<(IVec2, UVec2)> = monitors.iter().map(|m| (m.physical_position, m.physical_size())).collect();
    let largest = all.iter().map(|m| m.1).max_by_key(|s| u64::from(s.x) * u64::from(s.y)).unwrap_or(UVec2::new(1280, 720));
    let (w, h) = (window.resolution.physical_width(), window.resolution.physical_height());
    if w > largest.x || h > largest.y {
        let k = window.resolution.scale_factor();
        let (nw, nh) = (w.min(largest.x * 9 / 10), h.min(largest.y * 9 / 10));
        window.resolution.set(nw as f32 / k, nh as f32 / k);
    }
    if let bevy::window::WindowPosition::At(p) = window.position
        && !on_some_monitor(p, window.resolution.physical_width(), &all)
    {
        window.position = bevy::window::WindowPosition::Centered(bevy::window::MonitorSelection::Primary);
    }
}

pub struct WindowStatePlugin;

impl Plugin for WindowStatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (keep_on_screen, remember).chain());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_left_on_an_unplugged_monitor_is_off_screen() {
        let one = [(IVec2::ZERO, UVec2::new(1920, 1080))];
        assert!(on_some_monitor(IVec2::new(100, 100), 1280, &one));
        assert!(on_some_monitor(IVec2::new(-1000, 0), 1280, &one), "partly visible is fine");
        assert!(!on_some_monitor(IVec2::new(2200, 100), 1280, &one), "right of the only monitor");
        assert!(!on_some_monitor(IVec2::new(100, 1060), 1280, &one), "title bar below the bottom edge");
        assert!(!on_some_monitor(IVec2::new(100, -500), 1280, &one), "title bar above the top");
        let two = [(IVec2::ZERO, UVec2::new(1920, 1080)), (IVec2::new(1920, 0), UVec2::new(2560, 1440))];
        assert!(on_some_monitor(IVec2::new(2200, 100), 1280, &two));
    }
}
