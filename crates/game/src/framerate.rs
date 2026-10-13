//! Settings > Graphics > FPS cap and VSync (Python: `clock.tick(cfg["graphics.fps_cap"])` at the end of every frame, and
//! `App(vsync=cfg["graphics.vsync"])`). Bevy by itself presents with vsync and has no cap, so both settings did nothing.
//! The FPS counter itself is drawn with the HUD (`ktf_ui::hud`, `HudState.fps`).

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::window::{PresentMode, PrimaryWindow};

use crate::menu_ui::UserConfig;

/// The frame rate while the window is not focused.
const UNFOCUSED_FPS: i64 = 30;

/// The cap in force: the player's, except behind another window (alt-tab) or minimized, where nobody is watching and the game never
/// draws faster than [`UNFOCUSED_FPS`], even with the cap off. A lower cap the player chose is kept.
pub fn effective_cap(cap: i64, focused: bool) -> i64 {
    if !focused && (cap <= 0 || cap > UNFOCUSED_FPS) { UNFOCUSED_FPS } else { cap }
}

/// How long a frame may take at this cap (0 or less: no cap, as pygame's `Clock.tick(0)`).
pub fn frame_budget(cap: i64) -> Option<Duration> {
    (cap > 0).then(|| Duration::from_secs_f64(1.0 / cap as f64))
}

/// VSync on: wait for the display; off: present at once (the cap alone then sets the pace). Followed live.
fn vsync(cfg: Res<UserConfig>, mut window: Single<&mut Window, With<PrimaryWindow>>) {
    let want = if cfg.0.get_bool("graphics.vsync") { PresentMode::AutoVsync } else { PresentMode::AutoNoVsync };
    if window.present_mode != want {
        window.present_mode = want;
    }
}

/// `Clock.tick`: sleep out the rest of the frame.
fn cap(cfg: Res<UserConfig>, window: Single<&Window, With<PrimaryWindow>>, mut last: Local<Option<Instant>>) {
    let fps_cap = effective_cap(cfg.0.get_i64("graphics.fps_cap"), window.focused);
    if let (Some(budget), Some(t0)) = (frame_budget(fps_cap), *last) {
        // the deadline counts from the last deadline, so a late frame does not make every later one late too
        let deadline = t0 + budget;
        let now = Instant::now();
        if deadline > now {
            std::thread::sleep(deadline - now);
            *last = Some(deadline);
            return;
        }
    }
    *last = Some(Instant::now());
}

pub struct FrameRatePlugin;

impl Plugin for FrameRatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, vsync).add_systems(Last, cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_is_the_frame_budget() {
        assert_eq!(frame_budget(0), None);
        assert_eq!(frame_budget(-5), None);
        assert_eq!(frame_budget(60), Some(Duration::from_secs_f64(1.0 / 60.0)));
        assert!(frame_budget(240).unwrap() < frame_budget(30).unwrap());
    }

    #[test]
    fn an_unfocused_window_is_held_to_30_fps_and_a_focused_one_is_left_alone() {
        assert_eq!(effective_cap(0, true), 0, "unlimited stays unlimited while you play");
        assert_eq!(effective_cap(240, true), 240);
        assert_eq!(effective_cap(0, false), 30, "unlimited is held back behind another window");
        assert_eq!(effective_cap(240, false), 30);
        assert_eq!(effective_cap(30, false), 30);
    }
}
