//! HUD units -> window pixels (`kick3d.compute_layout`). The HUD always fills the window. In "crisp" mode the scale is a whole number
//! whenever the window is at least 760 px tall per step, so text maps to whole pixels; "large" keeps the original 760-unit-tall HUD,
//! smoothly scaled.

/// Width of the brain panel in HUD units (`PANEL_W = W - PLAY_W` = 1280 - 890 in the pygame game, 390 wide).
pub const PANEL_W: i32 = 390;
pub const PANEL_MODES: [(&str, u32); 4] = [("solid", 255), ("see-through", 150), ("faint", 70), ("hidden", 0)];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    pub scale: f32,
    pub hud_w: i32,
    pub hud_h: i32,
    /// The HUD area left of the brain panel.
    pub play_w: i32,
    /// How wide the 3D view is (the whole window unless the panel is solid).
    pub view_w: i32,
}

pub fn compute_layout(wn: i32, hn: i32, crisp: bool, panel_mode: usize, ui_scale: f32) -> Layout {
    let (wn_f, hn_f) = (wn as f32, hn as f32);
    let hidden = PANEL_MODES[panel_mode].0 == "hidden";
    let mut s = if crisp && hn >= 760 { (hn / 760).max(1) as f32 } else { hn_f / 760.0 };
    if ui_scale != 1.0 {
        s *= ui_scale;
        if crisp && s >= 1.0 {
            s = s.round_ties_even().max(1.0);
        }
    }
    let need = if hidden { 900.0 } else { 1280.0 };
    if wn_f / s < need {
        s = wn_f / need;
    }
    if hn_f / s < 760.0 {
        s = hn_f / 760.0;
    }
    let hud_w = ((wn_f / s).round() as i32).max(1);
    let hud_h = ((hn_f / s).round() as i32).max(1);
    let play_w = if hidden { hud_w } else { hud_w - PANEL_W };
    let view_w = if PANEL_MODES[panel_mode].0 == "solid" { play_w } else { hud_w };
    Layout { scale: s, hud_w, hud_h, play_w, view_w }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_python_layout() {
        // 930x906 window, solid panel: scale 0.7266, hud 1280x1247, play area 890 wide (the reference screenshot)
        let l = compute_layout(930, 906, true, 0, 1.0);
        assert_eq!((l.hud_w, l.hud_h, l.play_w, l.view_w), (1280, 1247, 890, 890));
        // 1080p fullscreen, crisp: whole-number scale 1
        let l = compute_layout(1920, 1080, true, 0, 1.0);
        assert_eq!((l.scale, l.hud_w, l.hud_h, l.play_w), (1.0, 1920, 1080, 1530));
        // 4K crisp: 2x
        let l = compute_layout(3840, 2160, true, 0, 1.0);
        assert_eq!((l.scale, l.hud_w, l.hud_h), (2.0, 1920, 1080));
        // large mode: smooth
        let l = compute_layout(1920, 1080, false, 0, 1.0);
        assert!((l.scale - 1080.0 / 760.0).abs() < 1e-6);
        // hidden panel
        assert_eq!(compute_layout(1920, 1080, true, 3, 1.0).play_w, 1920);
    }
}
