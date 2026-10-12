//! Text the way `pygame.font` set it in the menu: DejaVu Sans (regular for body text, bold for buttons and headings), the same pixel
//! sizes, the same line heights and baselines, whole-pixel advances. `legacy-python` picked the first of segoeui, dejavusans,
//! liberationsans, arial that the OS had; on Windows that was Segoe UI. The Rust game bundles DejaVu Sans so every OS looks like
//! the Linux original (the one listed difference of the menus in PORTING.md).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

use ab_glyph::{Font as _, FontRef, PxScale, ScaleFont};

use crate::canvas::Canvas;
use crate::theme::Color;

static REGULAR: &[u8] = include_bytes!("../../../assets/fonts/FreeSans.otf");
static BOLD: &[u8] = include_bytes!("../../../assets/fonts/FreeSansBold.otf");
static MONO: &[u8] = include_bytes!("../../../assets/fonts/DejaVuSansMono.ttf");
static FREE_MONO: &[u8] = include_bytes!("../../../assets/fonts/FreeMono.otf");

/// pygame's hinted per-glyph metrics for the sizes the menus and the HUD use: `[advance, min_x, max_x]` per glyph plus the line height and
/// ascent (`data/advances.json`, written by `legacy-python/make_ui_metrics.py`), keyed "<face><size>". FreeType's hinting moves advances
/// by up to a pixel per glyph, which a plain scaler cannot reproduce, and a layout made from different widths would not line up with
/// the original.
struct Table {
    glyphs: HashMap<char, [i32; 3]>,
    linesize: i32,
    ascent: i32,
}

fn tables() -> &'static HashMap<String, Table> {
    static T: OnceLock<HashMap<String, Table>> = OnceLock::new();
    T.get_or_init(|| {
        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(include_str!("../data/advances.json")).expect("advances.json");
        raw.into_iter()
            .map(|(k, v)| {
                let m = v["_metrics"].as_array().expect("metrics");
                let (linesize, ascent) = (m[0].as_i64().unwrap() as i32, m[1].as_i64().unwrap() as i32);
                let glyphs = v
                    .as_object()
                    .unwrap()
                    .iter()
                    .filter(|(c, _)| c.chars().count() == 1)
                    .map(|(c, a)| {
                        let a = a.as_array().unwrap();
                        (
                            c.chars().next().unwrap(),
                            [a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32, a[2].as_i64().unwrap() as i32],
                        )
                    })
                    .collect();
                (k, Table { glyphs, linesize, ascent })
            })
            .collect()
    })
}

/// One rasterised glyph: where its top-left pixel sits relative to the pen on the baseline, its width, and the coverage row by row.
struct Raster {
    x0: i32,
    y0: i32,
    w: usize,
    cov: Vec<f32>,
}

thread_local! {
    /// (font file, pixel size, character) -> its picture.
    static GLYPHS: RefCell<HashMap<(usize, i32, char), Rc<Raster>>> = RefCell::new(HashMap::new());
}

#[derive(Clone)]
pub struct Font {
    face: FontRef<'static>,
    /// Which bundled font file this is (its address), for the glyph cache.
    face_id: usize,
    pub size: i32,
    pub bold: bool,
    pub linesize: i32,
    pub ascent: i32,
    glyphs: Option<&'static HashMap<char, [i32; 3]>>,
}

impl Font {
    fn make(bytes: &'static [u8], key: &str, size: i32, bold: bool) -> Self {
        let face = FontRef::try_from_slice(bytes).expect("bundled font");
        let face_id = bytes.as_ptr() as usize;
        match tables().get(key) {
            Some(t) => Font { face, face_id, size, bold, linesize: t.linesize, ascent: t.ascent, glyphs: Some(&t.glyphs) },
            None => Font {
                face,
                face_id,
                size,
                bold,
                linesize: (size as f32 * 1.2).ceil() as i32,
                ascent: (size as f32 * 0.9277).round() as i32,
                glyphs: None,
            },
        }
    }

    /// FreeSans / FreeSansBold at a pixel size (`SysFont("segoeui,...", size, bold)`).
    pub fn new(size: i32, bold: bool) -> Self {
        Self::make(if bold { BOLD } else { REGULAR }, &format!("{}{size}", if bold { "bold" } else { "sans" }), size, bold)
    }

    /// FreeMono (`SysFont("consolas", size)`): the HUD's small text.
    pub fn free_mono(size: i32) -> Self {
        Self::make(FREE_MONO, &format!("mono{size}"), size, false)
    }

    /// `pygame.font.SysFont("dejavusansmono,...", size)`: share codes, bug reports.
    pub fn mono(size: i32) -> Self {
        let mut f = Self::make(MONO, "none", size, false);
        (f.linesize, f.ascent) = match size {
            12 => (14, 12),
            13 => (16, 13),
            _ => (f.linesize, f.ascent),
        };
        f
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
        self.draw_a(cv, text, x, y, color, 255)
    }

    /// The same with `img.set_alpha(alpha)`.
    pub fn draw_a(&self, cv: &mut Canvas, text: &str, x: i32, y: i32, color: Color, alpha: u32) {
        let mut pen = x;
        let base = y + self.ascent;
        for c in text.chars() {
            let g = self.raster(c);
            for (i, &cov) in g.cov.iter().enumerate() {
                if cov > 0.0 {
                    let a = (cov * alpha as f32 + 0.5) as u32;
                    cv.blend(pen + g.x0 + (i % g.w) as i32, base + g.y0 + (i / g.w) as i32, color, a);
                }
            }
            pen += self.metrics(c)[0];
        }
    }

    /// The coverage of one glyph with its pen at the origin. The pen always sits on whole pixels, so the picture of a glyph is the same
    /// wherever it is drawn: it is outlined and rasterised once per (font, size, character) and kept. The HUD draws a few hundred
    /// glyphs a frame; doing the outline for each of them every frame took most of the frame.
    fn raster(&self, c: char) -> Rc<Raster> {
        let key = (self.face_id, self.size, c);
        if let Some(g) = GLYPHS.with_borrow(|m| m.get(&key).cloned()) {
            return g;
        }
        let s = self.scaled();
        let g = s.glyph_id(c).with_scale_and_position(s.scale(), ab_glyph::point(0.0, 0.0));
        let r = match self.face.outline_glyph(g) {
            Some(o) => {
                let b = o.px_bounds();
                let (w, h) = ((b.width() as usize).max(1), (b.height() as usize).max(1));
                let mut cov = vec![0.0f32; w * h];
                o.draw(|gx, gy, v| {
                    if let Some(px) = cov.get_mut(gy as usize * w + gx as usize) {
                        *px = v;
                    }
                });
                Raster { x0: b.min.x as i32, y0: b.min.y as i32, w, cov }
            }
            None => Raster { x0: 0, y0: 0, w: 1, cov: Vec::new() },
        };
        let r = Rc::new(r);
        GLYPHS.with_borrow_mut(|m| m.insert(key, r.clone()));
        r
    }
}
