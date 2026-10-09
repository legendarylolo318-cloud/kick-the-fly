//! Text the way `pygame.font` set it in the menu: DejaVu Sans (regular for body text, bold for buttons and headings), the same pixel
//! sizes, the same line heights and baselines, whole-pixel advances. `legacy-python` picked the first of segoeui, dejavusans,
//! liberationsans, arial that the OS had; on Windows that was Segoe UI. The Rust game bundles DejaVu Sans so every OS looks like
//! the Linux original (the one listed difference of the menus in PORTING.md).

use std::collections::HashMap;
use std::sync::OnceLock;

use ab_glyph::{Font as _, FontRef, PxScale, ScaleFont};

use crate::canvas::Canvas;
use crate::theme::Color;

static REGULAR: &[u8] = include_bytes!("../../../assets/fonts/FreeSans.otf");
static BOLD: &[u8] = include_bytes!("../../../assets/fonts/FreeSansBold.otf");

/// pygame's hinted per-glyph metrics for the sizes the menus use: `[advance, min_x, max_x]` (`data/advances.json`, written by
/// `legacy-python/make_ui_metrics.py`). FreeType's hinting moves advances by up to a pixel per glyph, which a plain scaler cannot
/// reproduce, and a menu laid out from different widths would not line up with the original.
fn table() -> &'static HashMap<String, HashMap<char, [i32; 3]>> {
    static T: OnceLock<HashMap<String, HashMap<char, [i32; 3]>>> = OnceLock::new();
    T.get_or_init(|| {
        let raw: HashMap<String, HashMap<String, [i32; 3]>> =
            serde_json::from_str(include_str!("../data/advances.json")).expect("advances.json");
        raw.into_iter().map(|(k, v)| (k, v.into_iter().filter_map(|(c, m)| c.chars().next().map(|c| (c, m))).collect())).collect()
    })
}

#[derive(Clone)]
pub struct Font {
    face: FontRef<'static>,
    pub size: i32,
    pub bold: bool,
    pub linesize: i32,
    pub ascent: i32,
    glyphs: Option<&'static HashMap<char, [i32; 3]>>,
}

/// (size, linesize, ascent) pygame reported for the sizes the menus use (`tests/ui_reference/font_metrics.json`).
const METRICS: [(i32, i32, i32); 8] =
    [(14, 17, 13), (16, 20, 15), (22, 27, 20), (38, 46, 35), (18, 22, 17), (20, 24, 18), (28, 34, 26), (48, 58, 44)];

impl Font {
    pub fn new(size: i32, bold: bool) -> Self {
        let face = FontRef::try_from_slice(if bold { BOLD } else { REGULAR }).expect("bundled font");
        let (linesize, ascent) = METRICS
            .iter()
            .find(|m| m.0 == size)
            .map_or(((size as f32 * 1.2).ceil() as i32, (size as f32 * 0.9277).round() as i32), |m| (m.1, m.2));
        let suffix = if bold { "b" } else { "" };
        let glyphs = table().iter().find(|(k, _)| k.ends_with(&format!("@{size}{suffix}")) && (k.ends_with('b') == bold)).map(|(_, v)| v);
        Font { face, size, bold, linesize, ascent, glyphs }
    }

    fn scaled(&self) -> ab_glyph::PxScaleFont<&FontRef<'static>> {
        // pygame's size is the em in pixels; ab_glyph's PxScale is ascent+descent in pixels
        let h = self.face.height_unscaled();
        self.face.as_scaled(PxScale::from(self.size as f32 * h / self.face.units_per_em().unwrap_or(1000.0)))
    }

    /// (advance, min_x, max_x) of a glyph in whole pixels.
    fn metrics(&self, c: char) -> [i32; 3] {
        if let Some(m) = self.glyphs.and_then(|g| g.get(&c)) {
            return *m;
        }
        let s = self.scaled();
        let id = s.glyph_id(c);
        let adv = s.h_advance(id).round() as i32;
        let (min, max) = self.face.outline_glyph(id.with_scale_and_position(s.scale(), ab_glyph::point(0.0, 0.0))).map_or((0, 0), |o| {
            let b = o.px_bounds();
            (b.min.x.floor() as i32, b.max.x.ceil() as i32)
        });
        [adv, min, max]
    }

    /// `font.size(text)[0]`: the pen position after the last advance, or the last glyph's right edge when it reaches further (SDL_ttf).
    pub fn size(&self, text: &str) -> i32 {
        let (mut pen, mut hi, mut lo) = (0, 0, 0);
        for (i, c) in text.chars().enumerate() {
            let [adv, mn, mx] = self.metrics(c);
            if i == 0 {
                lo = mn.min(0);
            }
            hi = hi.max(pen + mx);
            pen += adv;
        }
        pen.max(hi) - lo
    }

    /// `font.render(text, True, color)` blitted with its top-left at (x, y): the image is `linesize` tall, the baseline `ascent` down.
    pub fn draw(&self, cv: &mut Canvas, text: &str, x: i32, y: i32, color: Color) {
        let s = self.scaled();
        let mut pen = x;
        for c in text.chars() {
            let g = s.glyph_id(c).with_scale_and_position(s.scale(), ab_glyph::point(pen as f32, (y + self.ascent) as f32));
            if let Some(o) = self.face.outline_glyph(g) {
                let b = o.px_bounds();
                o.draw(|gx, gy, cov| {
                    let a = (cov * 255.0 + 0.5) as u32;
                    cv.blend(b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32, color, a);
                });
            }
            pen += self.metrics(c)[0];
        }
    }
}
