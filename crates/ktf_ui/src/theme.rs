//! Colors of `ui/menu.py` (RGB, 0-255).
pub type Color = [u8; 3];
pub const BG: Color = [14, 17, 23];
pub const PANEL: Color = [20, 24, 32];
pub const ROW_HOVER: Color = [30, 35, 46];
pub const BTN: Color = [44, 50, 64];
pub const BTN_HOVER: Color = [62, 70, 90];
pub const BTN_DOWN: Color = [80, 90, 116];
pub const BTN_PRIMARY: Color = [40, 110, 150];
pub const BTN_PRIMARY_HOVER: Color = [52, 136, 182];
pub const BTN_DANGER: Color = [130, 44, 44];
pub const BTN_DANGER_HOVER: Color = [160, 56, 56];
pub const BORDER: Color = [52, 60, 76];
pub const ACCENT: Color = [86, 214, 255];
pub const AMBER: Color = [255, 176, 64];
pub const INK: Color = [240, 243, 248];
pub const TEXT: Color = [205, 212, 224];
pub const LABEL: Color = [130, 142, 160];
pub const DIM: Color = [80, 88, 102];
pub const GOOD: Color = [90, 200, 120];
pub const BAD: Color = [230, 90, 80];

/// Chip colors by label (`TAG_COLORS`); anything else is `LABEL`.
pub fn tag_color(label: &str) -> Color {
    match label {
        "Connectome" | "CONNECTOME" | "REAL" => [60, 170, 220],
        "Game rule" | "GAME RULE" | "RULE" => [220, 150, 50],
        "RESTART" | "3D ONLY" => [110, 118, 136],
        "LITERATURE" => [110, 190, 130],
        "REFUSED" => [200, 70, 60],
        // `lab.install` registers it (`TAG_COLORS.setdefault("MODEL", ...)`) and the Lab pages are the ones that draw it
        "MODEL" => [150, 120, 220],
        _ => LABEL,
    }
}
