//! A software canvas with the handful of drawing calls the pygame menu uses (`pygame.draw.rect` with a border radius, `circle`,
//! alpha fills, clipping, text blits). Like pygame's they are not anti-aliased, so shapes land on the same pixels; text is
//! anti-aliased like `Font.render(..., True, ...)`.

use crate::theme::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Rect { x, y, w, h }
    }
    pub fn right(&self) -> i32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> i32 {
        self.y + self.h
    }
    pub fn centerx(&self) -> i32 {
        self.x + self.w / 2
    }
    pub fn centery(&self) -> i32 {
        self.y + self.h / 2
    }
    pub fn center(&self) -> (i32, i32) {
        (self.centerx(), self.centery())
    }
    /// pygame's `collidepoint`: the right and bottom edges are outside.
    pub fn contains(&self, p: (i32, i32)) -> bool {
        p.0 >= self.x && p.0 < self.right() && p.1 >= self.y && p.1 < self.bottom()
    }
    pub fn intersect(&self, o: &Rect) -> Rect {
        let x = self.x.max(o.x);
        let y = self.y.max(o.y);
        let r = self.right().min(o.right());
        let b = self.bottom().min(o.bottom());
        Rect { x, y, w: (r - x).max(0), h: (b - y).max(0) }
    }
}

pub struct Canvas {
    pub w: i32,
    pub h: i32,
    /// RGBA, straight (non-premultiplied) alpha, row-major.
    pub px: Vec<u8>,
    pub clip: Option<Rect>,
    /// The canvas is shown by a renderer that blends in linear light (Bevy), while pygame blended in sRGB space: translucent pixels get
    /// the alpha that makes the result look the same over a typical scene (`alpha' = 1 - (1 - alpha)^1.9`; the veil's 196 becomes 240).
    pub linear_light: bool,
}

fn light_lut() -> &'static [u8; 256] {
    static L: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        let mut t = [0u8; 256];
        for (a, v) in t.iter_mut().enumerate() {
            *v = (255.0 * (1.0 - (1.0 - a as f32 / 255.0).powf(1.9)) + 0.5) as u8;
        }
        t
    })
}

impl Canvas {
    pub fn new(w: i32, h: i32) -> Self {
        Canvas { w, h, px: vec![0; (w * h * 4) as usize], clip: None, linear_light: false }
    }

    pub fn clear(&mut self) {
        self.px.fill(0);
    }

    fn bounds(&self) -> Rect {
        let full = Rect::new(0, 0, self.w, self.h);
        self.clip.map_or(full, |c| c.intersect(&full))
    }

    /// Source-over one pixel (`alpha` 0-255) honoring the clip.
    #[inline]
    pub fn blend(&mut self, x: i32, y: i32, c: Color, alpha: u32) {
        let b = self.bounds();
        if !b.contains((x, y)) || alpha == 0 {
            return;
        }
        let alpha = if self.linear_light && alpha < 255 { light_lut()[alpha as usize] as u32 } else { alpha };
        let i = ((y * self.w + x) * 4) as usize;
        let d = &mut self.px[i..i + 4];
        if alpha >= 255 {
            d.copy_from_slice(&[c[0], c[1], c[2], 255]);
            return;
        }
        let da = d[3] as u32;
        let out_a = alpha + da * (255 - alpha) / 255;
        if out_a == 0 {
            return;
        }
        for k in 0..3 {
            let num = c[k] as u32 * alpha * 255 + d[k] as u32 * da * (255 - alpha);
            d[k] = (num / (out_a * 255)).min(255) as u8;
        }
        d[3] = out_a as u8;
    }

    /// `surf.blit(veil)`: a whole-canvas fill with alpha (the clip is ignored, as in pygame where the veil is blitted whole).
    pub fn veil(&mut self, c: Color, alpha: u32) {
        let a = alpha.min(255);
        let a = if self.linear_light && a < 255 { light_lut()[a as usize] as u32 } else { a };
        for p in self.px.as_chunks_mut::<4>().0 {
            let da = p[3] as u32;
            let out_a = a + da * (255 - a) / 255;
            if out_a == 0 {
                continue;
            }
            for k in 0..3 {
                p[k] = ((c[k] as u32 * a * 255 + p[k] as u32 * da * (255 - a)) / (out_a * 255)).min(255) as u8;
            }
            p[3] = out_a as u8;
        }
    }

    /// Make a rectangle fully transparent (where the 3D view or the brain image shows through).
    pub fn clear_rect(&mut self, r: Rect) {
        let b = r.intersect(&Rect::new(0, 0, self.w, self.h));
        for y in b.y..b.bottom() {
            let i = ((y * self.w + b.x) * 4) as usize;
            self.px[i..i + (b.w * 4) as usize].fill(0);
        }
    }

    /// Per row, how far a rounded rectangle's edge is pulled in at `dy` rows from its top (pygame draws the corners as quarter circles).
    fn inset(r: i32, dy: i32) -> i32 {
        if dy >= r {
            return 0;
        }
        let yc = (r - dy) as f32 - 0.5;
        let dx = ((r * r) as f32 - yc * yc).max(0.0).sqrt();
        r - dx.round() as i32
    }

    /// `pygame.draw.rect(surf, color, rect, border_radius=r)` (filled).
    pub fn fill_rect(&mut self, rc: Rect, c: Color, radius: i32) {
        self.fill_rect_a(rc, c, 255, radius);
    }

    pub fn fill_rect_a(&mut self, rc: Rect, c: Color, alpha: u32, radius: i32) {
        let r = radius.min(rc.w / 2).min(rc.h / 2).max(0);
        for dy in 0..rc.h {
            let d = if dy < r {
                dy
            } else if dy >= rc.h - r {
                rc.h - 1 - dy
            } else {
                r
            };
            let ins = Self::inset(r, d);
            for x in rc.x + ins..rc.right() - ins {
                self.blend(x, rc.y + dy, c, alpha);
            }
        }
    }

    /// `pygame.draw.rect(surf, color, rect, 1, border_radius=r)`: a one-pixel outline.
    pub fn stroke_rect(&mut self, rc: Rect, c: Color, radius: i32) {
        let r = radius.min(rc.w / 2).min(rc.h / 2).max(0);
        let inner = |dy: i32| -> Option<(i32, i32)> {
            // the filled interior, one pixel smaller on every side with a radius one smaller
            if dy < 1 || dy >= rc.h - 1 {
                return None;
            }
            let ir = (r - 1).max(0);
            let dd = dy - 1;
            let ih = rc.h - 2;
            let d = if dd < ir {
                dd
            } else if dd >= ih - ir {
                ih - 1 - dd
            } else {
                ir
            };
            let ins = Self::inset(ir, d);
            Some((rc.x + 1 + ins, rc.right() - 1 - ins))
        };
        for dy in 0..rc.h {
            let d = if dy < r {
                dy
            } else if dy >= rc.h - r {
                rc.h - 1 - dy
            } else {
                r
            };
            let ins = Self::inset(r, d);
            let (l, rr) = (rc.x + ins, rc.right() - ins);
            match inner(dy) {
                None => (l..rr).for_each(|x| self.blend(x, rc.y + dy, c, 255)),
                Some((il, ir)) => {
                    (l..il.max(l)).for_each(|x| self.blend(x, rc.y + dy, c, 255));
                    (ir.min(rr)..rr).for_each(|x| self.blend(x, rc.y + dy, c, 255));
                }
            }
        }
    }

    /// `pygame.draw.rect(surf, color, rect, width, border_radius=r)` with a given outline width.
    pub fn stroke_rect2(&mut self, rc: Rect, c: Color, width: i32, radius: i32) {
        for k in 0..width {
            self.stroke_rect(Rect::new(rc.x + k, rc.y + k, rc.w - 2 * k, rc.h - 2 * k), c, (radius - k).max(0));
        }
    }

    /// `pygame.draw.circle(surf, color, center, radius)`.
    pub fn circle(&mut self, center: (i32, i32), radius: i32, c: Color) {
        for dy in -radius..=radius {
            let dx = (((radius * radius - dy * dy) as f32).max(0.0).sqrt()) as i32;
            for x in center.0 - dx..=center.0 + dx {
                self.blend(x, center.1 + dy, c, 255);
            }
        }
    }

    /// Write the canvas as a PNG-ready RGB image composited over `bg` (for tests and the reference comparison).
    pub fn to_rgb_over(&self, bg: Color) -> Vec<u8> {
        let mut out = Vec::with_capacity((self.w * self.h * 3) as usize);
        for p in self.px.as_chunks::<4>().0 {
            let a = p[3] as u32;
            for k in 0..3 {
                out.push(((p[k] as u32 * a + bg[k] as u32 * (255 - a)) / 255) as u8);
            }
        }
        out
    }

    // --- anti-aliased shapes (pygame's gfxdraw: filled + aa outline) ---------------------------------------------------------

    /// `aacircle`: a filled circle with a smooth edge.
    pub fn aa_circle(&mut self, c: (f32, f32), r: f32, col: Color) {
        self.aa_circle_a(c, r, col, 255);
    }

    pub fn aa_circle_a(&mut self, c: (f32, f32), r: f32, col: Color, alpha: u32) {
        let r = r.round().max(1.0);
        let (cx, cy) = (c.0.round(), c.1.round());
        for y in (cy - r - 1.0) as i32..=(cy + r + 1.0) as i32 {
            for x in (cx - r - 1.0) as i32..=(cx + r + 1.0) as i32 {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                let cov = (r - d + 0.5).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend(x, y, col, (cov * alpha as f32) as u32);
                }
            }
        }
    }

    /// `pygame.draw.circle(surf, col, c, r, 1)`-like ring, smooth.
    pub fn aa_ring(&mut self, c: (f32, f32), r: f32, col: Color) {
        for y in (c.1 - r - 2.0) as i32..=(c.1 + r + 2.0) as i32 {
            for x in (c.0 - r - 2.0) as i32..=(c.0 + r + 2.0) as i32 {
                let d = ((x as f32 - c.0).powi(2) + (y as f32 - c.1).powi(2)).sqrt();
                let cov = (1.0 - (d - r).abs()).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend(x, y, col, (cov * 255.0) as u32);
                }
            }
        }
    }

    /// `aapoly`: a filled polygon with smooth edges (4x4 sub-pixel coverage).
    pub fn aa_poly(&mut self, pts: &[(f32, f32)], col: Color) {
        self.aa_poly_a(pts, col, 255);
    }

    pub fn aa_poly_a(&mut self, pts: &[(f32, f32)], col: Color, alpha: u32) {
        if pts.len() < 3 {
            return;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in pts {
            x0 = x0.min(p.0);
            y0 = y0.min(p.1);
            x1 = x1.max(p.0);
            y1 = y1.max(p.1);
        }
        const N: i32 = 4;
        for y in y0.floor() as i32..=y1.ceil() as i32 {
            for x in x0.floor() as i32..=x1.ceil() as i32 {
                let mut hit = 0;
                for sy in 0..N {
                    for sx in 0..N {
                        let p = (x as f32 + (sx as f32 + 0.5) / N as f32, y as f32 + (sy as f32 + 0.5) / N as f32);
                        if point_in_poly(p, pts) {
                            hit += 1;
                        }
                    }
                }
                if hit > 0 {
                    self.blend(x, y, col, alpha * hit / (N * N) as u32);
                }
            }
        }
    }

    /// `thick_line`: a line `w` wide with a round end at `b`.
    pub fn thick_line(&mut self, a: (f32, f32), b: (f32, f32), w: f32, col: Color) {
        let d = (b.0 - a.0, b.1 - a.1);
        let l = (d.0 * d.0 + d.1 * d.1).sqrt().max(1.0);
        let n = (-d.1 / l * w / 2.0, d.0 / l * w / 2.0);
        self.aa_poly(&[(a.0 + n.0, a.1 + n.1), (b.0 + n.0, b.1 + n.1), (b.0 - n.0, b.1 - n.1), (a.0 - n.0, a.1 - n.1)], col);
        self.aa_circle(b, w / 2.0, col);
    }

    /// `pygame.draw.line` with width 1 or more, no smoothing (Bresenham).
    pub fn line(&mut self, a: (i32, i32), b: (i32, i32), width: i32, col: Color, alpha: u32) {
        let (mut x, mut y) = a;
        let (dx, dy) = ((b.0 - a.0).abs(), -(b.1 - a.1).abs());
        let (sx, sy) = (if a.0 < b.0 { 1 } else { -1 }, if a.1 < b.1 { 1 } else { -1 });
        let mut err = dx + dy;
        loop {
            for k in 0..width {
                // thick lines grow across the minor axis, as pygame's do
                if dx >= -dy {
                    self.blend(x, y + k - width / 2, col, alpha);
                } else {
                    self.blend(x + k - width / 2, y, col, alpha);
                }
            }
            if (x, y) == b {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// `pygame.draw.aalines(surf, col, False, pts)`: a smooth 1 px polyline.
    pub fn aa_lines(&mut self, pts: &[(f32, f32)], col: Color) {
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let d = (b.0 - a.0, b.1 - a.1);
            let l = (d.0 * d.0 + d.1 * d.1).sqrt().max(1e-3);
            let n = (-d.1 / l * 0.5, d.0 / l * 0.5);
            self.aa_poly(&[(a.0 + n.0, a.1 + n.1), (b.0 + n.0, b.1 + n.1), (b.0 - n.0, b.1 - n.1), (a.0 - n.0, a.1 - n.1)], col);
        }
    }

    /// A filled ellipse (`gfxdraw.filled_ellipse`).
    pub fn ellipse(&mut self, c: (f32, f32), rx: f32, ry: f32, col: Color, alpha: u32) {
        for y in (c.1 - ry - 1.0) as i32..=(c.1 + ry + 1.0) as i32 {
            for x in (c.0 - rx - 1.0) as i32..=(c.0 + rx + 1.0) as i32 {
                let v = ((x as f32 - c.0) / rx).powi(2) + ((y as f32 - c.1) / ry).powi(2);
                let edge = (1.0 - v) * rx.min(ry) * 0.5 + 0.5;
                let cov = edge.clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend(x, y, col, (cov * alpha as f32) as u32);
                }
            }
        }
    }

    /// An ellipse outline, one pixel.
    pub fn ellipse_outline(&mut self, c: (f32, f32), rx: f32, ry: f32, col: Color) {
        let n = ((rx + ry) * 3.0) as i32;
        for k in 0..n {
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            self.blend((c.0 + rx * a.cos()).round() as i32, (c.1 + ry * a.sin()).round() as i32, col, 255);
        }
    }
}

fn point_in_poly(p: (f32, f32), pts: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut j = pts.len() - 1;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[j]);
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
            inside = !inside;
        }
        j = i;
    }
    inside
}
