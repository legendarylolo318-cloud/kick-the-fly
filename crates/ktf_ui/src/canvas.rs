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
}

impl Canvas {
    pub fn new(w: i32, h: i32) -> Self {
        Canvas { w, h, px: vec![0; (w * h * 4) as usize], clip: None }
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
}
