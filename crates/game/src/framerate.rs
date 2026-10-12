//! Settings > Graphics > FPS cap and VSync (Python: `clock.tick(cfg["graphics.fps_cap"])` at the end of every frame, and
//! `App(vsync=cfg["graphics.vsync"])`). Bevy by itself presents with vsync and has no cap, so both settings did nothing.
//! The FPS counter itself is drawn with the HUD (`ktf_ui::hud`, `HudState.fps`).

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::window::{PresentMode, PrimaryWindow};

use crate::menu_ui::UserConfig;

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
fn cap(cfg: Res<UserConfig>, mut last: Local<Option<Instant>>) {
    if let (Some(budget), Some(t0)) = (frame_budget(cfg.0.get_i64("graphics.fps_cap")), *last) {
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
}
