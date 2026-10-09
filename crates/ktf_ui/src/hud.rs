//! The in-game HUD of the 3D game, drawn the way `kick_the_fly.py:_draw_hud` / `kick3d.py:draw_hud3d` drew it: a transparent canvas of
//! `hud_w x hud_h` units over the 3D view, with the status card, health bar, hint line, PAIN / REWARD / MEMORY cards, the tool hotbar with
//! its icons, the crosshair, popups, and the brain panel on the right (title, touch and descending-neuron bars, the whole-brain trace,
//! the reaction log). Everything dynamic comes in as [`HudState`], filled each frame by the game.

use crate::canvas::{Canvas, Rect};
use crate::font::Font;
use crate::layout::Layout;
use crate::theme::*;

pub const PANEL_BG: Color = [13, 16, 22];
pub const CARD: Color = [20, 24, 32];
pub const S_GOOD: Color = [12, 163, 12];
pub const S_WARN: Color = [250, 178, 25];
pub const S_CRIT: Color = [208, 59, 59];
const BORDER_PANEL: Color = [40, 47, 60];

/// The HUD's fonts (`Game.fonts`): FreeMono 13 for dense text, FreeSans for the rest; `k` is 1.25 with Settings > Accessibility > Larger text.
#[derive(Clone)]
pub struct HudFonts {
    pub small: Font,
    pub text: Font,
    pub bold: Font,
    pub head: Font,
    pub title: Font,
    pub big: Font,
}

impl HudFonts {
    pub fn new(larger_text: bool) -> Self {
        let k = if larger_text { 1.25 } else { 1.0 };
        let sz = |n: f32| (n * k).round_ties_even() as i32;
        HudFonts {
            small: Font::free_mono(13),
            text: Font::new(sz(15.0), false),
            bold: Font::new(sz(16.0), true),
            head: Font::new(sz(20.0), true),
            title: Font::new(sz(34.0), true),
            big: Font::new(sz(40.0), false),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolSlot {
    pub name: String,
    pub label: String,
    /// The rebindable key shown in the corner ("1" ... "0").
    pub key: String,
}

#[derive(Debug, Clone, Default)]
pub struct Popup {
    pub x: f32,
    pub y: f32,
    pub text: String,
    pub color: Color,
    /// 0..1 of life used up.
    pub age: f32,
    /// "real" / "rule" chip under the word (when the honesty tags are on).
    pub source: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MotorRow {
    pub label: String,
    /// Fast rate as a multiple of the calm baseline.
    pub level: f32,
    pub thresh: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct LogLine {
    pub age: f32,
    pub text: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct PanelData {
    pub dead: bool,
    pub calm_fx: bool,
    pub neurons: usize,
    /// The big brain view is showing it: the panel image is replaced by a note.
    pub big_view: bool,
    /// head, body, legs, wing: spikes/s per neuron.
    pub touch: [f32; 4],
    pub motor: Vec<MotorRow>,
    pub whole_trace: Vec<f32>,
    pub log: Vec<LogLine>,
    pub steps_per_s: f32,
    pub step_ms: f32,
}

#[derive(Debug, Clone, Default)]
pub struct HudState {
    pub fly_state: String,
    pub hits: u32,
    pub kills: u32,
    pub flies: u32,
    pub focus: u32,
    pub max_flies: u32,
    pub health: f32,
    pub max_health: f32,
    pub immortal: bool,
    pub fly_dead: bool,
    pub arena: String,
    pub arena_status: Vec<String>,
    pub surgery_on: bool,
    pub muted: bool,
    pub saved_msg: Option<String>,
    pub pain: f32,
    pub pain_parts: [f32; 5],
    pub pain_trace: Vec<f32>,
    pub pain_peak: f32,
    pub pain_max_s: f32,
    pub pain_level_name: String,
    pub pain_level: usize,
    pub pain_neurons: u32,
    pub pain_connectome: bool,
    pub reward: f32,
    pub reward_level: f32,
    /// (smell, fear, like)
    pub memory: Vec<(String, f32, f32)>,
    /// The hotbar page's tools, in slot order.
    pub tools: Vec<ToolSlot>,
    /// The tool in hand: its name, and the line above the hotbar ("1  HAND: drag the fly and throw it").
    pub tool_name: String,
    pub tool_line: String,
    pub page: usize,
    pub pages: usize,
    /// The mouse is captured for looking (the "click the room" hint shows when it is not).
    pub look: bool,
    pub free_mouse_key: String,
    pub overlay_open: bool,
    pub menu_open: bool,
    pub dead_player: bool,
    pub crosshair_hot: bool,
    pub hide_hud: bool,
    pub tags_on: bool,
    pub panel: Option<PanelData>,
    pub panel_alpha: u32,
    pub popups: Vec<Popup>,
    /// "PAUSED  x0.5" style label of the time indicator and whether the user (not the menu) paused.
    pub time_label: Option<(String, bool)>,
    /// The H controls overlay.
    pub help_open: bool,
    /// Neurodex toasts, newest last: (text, alpha 0..255).
    pub toasts: Vec<(String, u32)>,
    /// "+N more discovered (Neurodex: D)" under the toasts.
    pub toasts_more: Option<String>,
    /// The Neuron of the Day card, while it is up.
    pub notd: Option<NotdView>,
    /// The first-launch tutorial card.
    pub tutorial: Option<TutorialView>,
}

/// The tutorial's card (`ui/tutorial.py:draw`).
#[derive(Debug, Clone, Default)]
pub struct TutorialView {
    /// "Step 2 of 5" (and "  ·  replay")
    pub step_line: String,
    pub title: String,
    pub body: String,
    pub next_label: String,
    pub skip_label: String,
    /// The accent colour (the loadout palette's "select").
    pub accent: Color,
    /// Outline the brain panel (the brain step), with this extra line width (0 with reduced flashing).
    pub outline_panel: Option<i32>,
}

/// What the launch card shows (`Extras.draw_notd`).
#[derive(Debug, Clone, Default)]
pub struct NotdView {
    pub label: String,
    pub text: String,
    pub cite: String,
    /// "laser", "surgery" or "none"
    pub action: String,
    pub plan_text: String,
    /// 3D with the mouse captured: the buttons cannot be clicked until Tab frees it.
    pub look3d: bool,
}

/// Where the clickable parts of the HUD ended up this frame (HUD units), for the click handler.
#[derive(Debug, Clone, Default)]
pub struct HudHits {
    pub notd_try: Option<Rect>,
    pub notd_off: Option<Rect>,
    pub notd_close: Option<Rect>,
    pub tutorial_next: Option<Rect>,
    pub tutorial_skip: Option<Rect>,
}

/// `draw_source_chip`: a REAL / RULE pill.
pub fn source_chip(cv: &mut Canvas, pos: (i32, i32), source: &str, font: &Font, anchor_topright: bool, alpha: u32) -> Rect {
    let col = if source == "real" { [60, 170, 220] } else { [220, 150, 50] };
    let word = if source == "real" { "REAL" } else { "RULE" };
    let (w, h) = (font.size(word) + 10, font.linesize + 2);
    let r = if anchor_topright { Rect::new(pos.0 - w, pos.1, w, h) } else { Rect::new(pos.0 - w / 2, pos.1, w, h) };
    cv.fill_rect_a(r, col, alpha, 5);
    font.draw_a(cv, word, r.centerx() - font.size(word) / 2, r.centery() - font.linesize / 2, [10, 12, 16], alpha);
    r
}

fn text(cv: &mut Canvas, s: &str, pos: (i32, i32), color: Color, font: &Font, anchor: &str) -> Rect {
    let (w, h) = (font.size(s), font.linesize);
    let (x, y) = match anchor {
        "topleft" => (pos.0, pos.1),
        "topright" => (pos.0 - w, pos.1),
        "midtop" => (pos.0 - w / 2, pos.1),
        "center" => (pos.0 - w / 2, pos.1 - h / 2),
        "midleft" => (pos.0, pos.1 - h / 2),
        _ => (pos.0, pos.1),
    };
    font.draw(cv, s, x, y, color);
    Rect::new(x, y, w, h)
}

/// A card: `pygame.Surface(SRCALPHA)` with a rounded translucent rect, blitted.
fn card(cv: &mut Canvas, r: Rect, rgba: (Color, u32), radius: i32) {
    cv.fill_rect_a(r, rgba.0, rgba.1, radius);
}

/// `Game._bar`
fn bar(cv: &mut Canvas, x: i32, y: i32, w: i32, frac: f32, color: Color, mark: Option<f32>) {
    cv.fill_rect(Rect::new(x, y, w, 8), [30, 36, 48], 4);
    let fw = (w as f32 * frac.clamp(0.0, 1.0)) as i32;
    if fw > 0 {
        cv.fill_rect(Rect::new(x, y, fw.max(4), 8), color, 4);
    }
    if let Some(m) = mark {
        let tx = x + (w as f32 * m.min(1.0)) as i32;
        cv.line((tx, y - 2), (tx, y + 9), 1, INK, 255);
    }
}

/// `draw_icon`: the 18 tool icons.
pub fn draw_icon(cv: &mut Canvas, name: &str, c: (f32, f32), col: Color) {
    let (x, y) = c;
    match name {
        "hand" => {
            cv.aa_circle((x, y + 4.0), 9.0, col);
            for k in 0..4 {
                let k = k as f32;
                cv.thick_line((x - 7.0 + k * 4.5, y), (x - 8.0 + k * 5.0, y - 11.0 + (k - 1.5).abs() * 2.0), 4.0, col);
            }
        }
        "flick" => {
            cv.aa_circle((x - 4.0, y + 3.0), 7.0, col);
            cv.thick_line((x - 2.0, y - 1.0), (x + 10.0, y - 9.0), 4.0, col);
            for k in [-1.0f32, 0.0, 1.0] {
                cv.thick_line((x + 13.0, y - 12.0 + k * 6.0), (x + 18.0, y - 14.0 + k * 8.0), 2.0, col);
            }
        }
        "swatter" => {
            cv.aa_poly(&[(x - 3.0, y - 13.0), (x + 11.0, y - 13.0), (x + 11.0, y + 1.0), (x - 3.0, y + 1.0)], col);
            cv.thick_line((x + 2.0, y + 1.0), (x - 8.0, y + 12.0), 3.0, col);
        }
        "zapper" => {
            cv.ellipse_outline((x, y - 4.0), 9.0, 11.0, col);
            cv.thick_line((x, y + 7.0), (x, y + 14.0), 3.0, col);
            cv.aa_poly(
                &[(x + 1.0, y - 12.0), (x - 5.0, y - 2.0), (x, y - 2.0), (x - 2.0, y + 5.0), (x + 5.0, y - 5.0), (x, y - 5.0)],
                [255, 230, 90],
            );
        }
        "freeze" => {
            for a in [0.0f32, 1.047, 2.094] {
                cv.thick_line((x - 11.0 * a.cos(), y - 11.0 * a.sin()), (x + 11.0 * a.cos(), y + 11.0 * a.sin()), 2.5, [170, 220, 255]);
            }
        }
        "spider" => {
            for sgn in [-1.0f32, 1.0] {
                for k in 0..4 {
                    let a = (k as f32 - 1.5) * 0.45;
                    cv.thick_line((x, y), (x + sgn * 13.0 * a.cos(), y + 9.0 * a.sin() - 3.0 + k as f32), 1.6, col);
                }
            }
            cv.aa_circle((x, y + 1.0), 6.0, col);
            cv.aa_circle((x, y - 6.0), 4.0, col);
        }
        "frog" => {
            cv.aa_circle((x, y + 2.0), 8.0, [70, 140, 58]);
            cv.aa_circle((x - 4.0, y - 5.0), 3.0, [240, 220, 80]);
            cv.aa_circle((x + 4.0, y - 5.0), 3.0, [240, 220, 80]);
            cv.thick_line((x + 6.0, y + 4.0), (x + 14.0, y + 9.0), 3.0, [230, 90, 100]);
        }
        "dragonfly" => {
            cv.thick_line((x - 12.0, y), (x + 8.0, y), 3.0, [38, 116, 190]);
            cv.aa_circle((x + 10.0, y), 4.0, [30, 150, 90]);
            cv.ellipse((x - 2.0, y - 7.0), 9.0, 3.0, [200, 215, 235], 190);
            cv.ellipse((x - 2.0, y + 7.0), 9.0, 3.0, [200, 215, 235], 190);
        }
        "mantis" => {
            cv.thick_line((x - 8.0, y + 6.0), (x, y - 2.0), 4.0, [116, 158, 56]);
            cv.thick_line((x, y - 2.0), (x + 3.0, y - 12.0), 3.0, [116, 158, 56]);
            cv.aa_circle((x + 4.0, y - 14.0), 3.0, [78, 116, 38]);
            cv.thick_line((x + 3.0, y - 10.0), (x + 11.0, y - 6.0), 2.0, [78, 116, 38]);
        }
        "sugar" => {
            cv.aa_poly(&[(x - 8.0, y - 4.0), (x + 2.0, y - 9.0), (x + 11.0, y - 4.0), (x + 1.0, y + 1.0)], [250, 250, 255]);
            cv.aa_poly(&[(x - 8.0, y - 4.0), (x + 1.0, y + 1.0), (x + 1.0, y + 12.0), (x - 8.0, y + 7.0)], [215, 215, 225]);
            cv.aa_poly(&[(x + 1.0, y + 1.0), (x + 11.0, y - 4.0), (x + 11.0, y + 7.0), (x + 1.0, y + 12.0)], [190, 190, 205]);
        }
        "fruit" => {
            cv.aa_circle((x, y + 2.0), 9.0, [214, 64, 48]);
            cv.aa_circle((x - 3.0, y - 1.0), 3.0, [245, 150, 130]);
            cv.thick_line((x, y - 6.0), (x + 2.0, y - 12.0), 2.0, [90, 60, 30]);
            cv.aa_poly(&[(x + 2.0, y - 10.0), (x + 10.0, y - 13.0), (x + 7.0, y - 6.0)], [90, 170, 80]);
        }
        "cleaner" => {
            cv.aa_poly(&[(x - 8.0, y - 6.0), (x + 2.0, y - 6.0), (x + 2.0, y + 13.0), (x - 8.0, y + 13.0)], col);
            cv.aa_poly(&[(x - 6.0, y - 11.0), (x, y - 11.0), (x, y - 6.0), (x - 6.0, y - 6.0)], col);
            for k in 0..3 {
                cv.aa_circle((x + 8.0 + k as f32 * 4.0, y - 12.0 + (k as f32 - 1.0) * 4.0), 2.0, [160, 220, 255]);
            }
        }
        "torch" => {
            cv.aa_poly(&[(x - 12.0, y + 2.0), (x + 2.0, y - 4.0), (x + 5.0, y + 2.0), (x - 9.0, y + 8.0)], col);
            cv.aa_poly(&[(x + 4.0, y - 4.0), (x + 16.0, y - 14.0), (x + 11.0, y - 1.0)], [255, 140, 40]);
            cv.aa_poly(&[(x + 5.0, y - 3.0), (x + 12.0, y - 9.0), (x + 9.0, y - 2.0)], [255, 230, 120]);
        }
        "alcohol" => {
            cv.aa_poly(
                &[
                    (x - 6.0, y + 11.0),
                    (x + 6.0, y + 11.0),
                    (x + 8.0, y + 3.0),
                    (x + 3.0, y - 3.0),
                    (x + 3.0, y - 10.0),
                    (x - 3.0, y - 10.0),
                    (x - 3.0, y - 3.0),
                    (x - 8.0, y + 3.0),
                ],
                col,
            );
            cv.aa_poly(&[(x - 5.0, y + 10.0), (x + 5.0, y + 10.0), (x + 6.0, y + 4.0), (x - 6.0, y + 4.0)], [225, 80, 130]);
            cv.aa_circle((x, y - 1.0), 2.0, [255, 180, 210]);
        }
        "laser" => {
            cv.aa_poly(&[(x - 10.0, y + 8.0), (x + 2.0, y - 4.0), (x + 6.0, y - 2.0), (x - 6.0, y + 10.0)], col);
            cv.aa_circle((x + 4.0, y - 3.0), 3.0, [80, 200, 255]);
            cv.thick_line((x + 5.0, y - 4.0), (x + 14.0, y - 11.0), 2.0, [100, 220, 255]);
        }
        "cva" => {
            cv.aa_circle((x - 4.0, y), 5.0, [255, 180, 100]);
            cv.aa_circle((x + 3.0, y - 2.0), 6.0, [255, 160, 80]);
            cv.aa_circle((x, y + 4.0), 4.0, [255, 200, 120]);
        }
        "decoy" => {
            cv.aa_circle((x, y - 6.0), 3.0, [180, 180, 200]);
            cv.aa_circle((x, y - 1.0), 4.0, [140, 140, 160]);
            cv.ellipse_outline((x, y + 6.0), 5.0, 8.0, [120, 120, 140]);
        }
        _ => {
            cv.aa_circle((x - 2.0, y + 3.0), 10.0, col);
            cv.thick_line((x + 5.0, y - 5.0), (x + 10.0, y - 11.0), 3.0, col);
            cv.aa_circle((x + 11.0, y - 13.0), 3.0, AMBER);
        }
    }
}

/// `Game._text` with the HUD's `tr`-less strings.
pub struct Hud<'a> {
    pub cv: &'a mut Canvas,
    pub lay: Layout,
    pub f: &'a HudFonts,
}

impl Hud<'_> {
    /// `draw_hud3d` (without the overlays that have their own screens): the order is the pygame order.
    pub fn draw(&mut self, st: &HudState, now: f64) -> HudHits {
        let mut hits = HudHits::default();
        let play_w = self.lay.play_w;
        // popups over the fly
        for p in &st.popups {
            let a = (255.0 * (1.0 - p.age.powi(3))).clamp(0.0, 255.0) as u32;
            let w = self.f.big.size(&p.text);
            let x = p.x as i32 - w / 2;
            let y = (p.y - 50.0 * p.age - 20.0) as i32;
            self.f.big.draw_a(self.cv, &p.text, x + 3, y + 3, [120, 16, 22], a);
            self.f.big.draw_a(self.cv, &p.text, x, y, p.color, a);
            if let (Some(src), true) = (&p.source, st.tags_on) {
                source_chip(self.cv, (p.x as i32, y + self.f.big.linesize), src, &self.f.small, false, a);
            }
        }
        if !st.overlay_open && !st.hide_hud {
            let (cx, cy) = (play_w / 2, self.lay.hud_h / 2);
            let col = if st.crosshair_hot { [255, 190, 90] } else { [240, 240, 240] };
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                self.cv.line((cx + dx * 5 + 1, cy + dy * 5 + 1), (cx + dx * 13 + 1, cy + dy * 13 + 1), 3, [0, 0, 0], 120);
                self.cv.line((cx + dx * 5, cy + dy * 5), (cx + dx * 13, cy + dy * 13), 2, col, 255);
            }
        }
        self.toolbar(st);
        self.status(st, now);
        if !st.look && !st.overlay_open && !st.dead_player && !st.menu_open && !st.hide_hud {
            let msg = format!("click the room (or press {}) to look around   ·   Esc: menu", title_case(&st.free_mouse_key));
            let w = self.f.bold.size(&msg);
            let (cx, cy) = (play_w / 2, self.lay.hud_h / 2 + 60);
            let bx = Rect::new(cx - w / 2 - 12, cy - self.f.bold.linesize / 2 - 6, w + 24, self.f.bold.linesize + 12);
            card(self.cv, bx, ([8, 10, 16], 200), 8);
            text(self.cv, &msg, bx.center(), [250, 250, 252], &self.f.bold, "center");
        }
        if st.help_open {
            self.help();
        }
        if let Some(p) = &st.panel {
            self.panel(st, p, now);
        }
        self.toasts(st);
        if let Some(n) = &st.notd {
            hits = self.notd(n);
        }
        if let Some(t) = &st.tutorial {
            let (n, k) = self.tutorial(t, st.panel.as_ref().is_some_and(|p| !p.big_view));
            hits.tutorial_next = Some(n);
            hits.tutorial_skip = Some(k);
        }
        if let Some((label, user)) = &st.time_label {
            let w = self.f.bold.size(label);
            let bx = Rect::new(play_w / 2 - w / 2 - 11, 92 - 5, w + 22, self.f.bold.linesize + 10);
            self.cv.fill_rect(bx, if *user { AMBER } else { ACCENT }, 8);
            text(self.cv, label, bx.center(), [20, 16, 8], &self.f.bold, "center");
        }
        hits
    }

    /// `Extras._banner` for each toast: a green-edged pill at the top of the play area.
    fn toasts(&mut self, st: &HudState) {
        let mut y = 8;
        let all: Vec<(&str, u32, Color)> = st
            .toasts
            .iter()
            .map(|(t, a)| (t.as_str(), *a, [90, 200, 120]))
            .chain(st.toasts_more.iter().map(|t| (t.as_str(), 200, [90, 200, 120])))
            .collect();
        for (t, a, col) in all {
            let w = (self.f.bold.size(t) + 28).min(self.lay.hud_w - 20);
            let cx = self.lay.hud_w.min(890) / 2;
            let r = Rect::new(cx - w / 2, y, w, 28);
            card(self.cv, r, ([14, 18, 26], (a as f32 * 0.9) as u32), 10);
            self.cv.stroke_rect2_a(r, col, a, 2, 10);
            self.f.bold.draw_a(self.cv, t, r.x + 14, r.y + 4, [240, 243, 248], a);
            y += 34;
        }
    }

    /// `Tutorial.draw`: a card at the top of the play area, with Next and Skip buttons; returns their rectangles.
    fn tutorial(&mut self, t: &TutorialView, panel_shown: bool) -> (Rect, Rect) {
        let f = self.f;
        let (w, pad) = (560, 14);
        let lines = wrap_lines(&t.body, &f.text, w - 2 * pad);
        let lh = f.text.linesize;
        let h = 44 + lh * lines.len() as i32 + 56;
        let card_r = Rect::new(self.lay.play_w / 2 - w / 2, 128, w, h);
        card(self.cv, card_r, ([8, 10, 16], 236), 12);
        self.cv.stroke_rect2(card_r, t.accent, 2, 12);
        text(self.cv, &t.step_line, (card_r.x + pad, card_r.y + 8), LABEL, &f.small, "topleft");
        text(self.cv, &t.title, (card_r.x + pad, card_r.y + 26), INK, &f.bold, "topleft");
        let mut y = card_r.y + 50;
        for ln in &lines {
            text(self.cv, ln, (card_r.x + pad, y), TEXT, &f.text, "topleft");
            y += lh;
        }
        let by = card_r.bottom() - 40;
        let mut rects = vec![];
        for (i, (label, skip)) in [(&t.next_label, false), (&t.skip_label, true)].into_iter().enumerate() {
            let r = Rect::new(card_r.x + pad + i as i32 * 190, by, f.small.size(label) + 24, 30);
            self.cv.fill_rect(r, if skip { [44, 50, 64] } else { [40, 110, 150] }, 8);
            text(self.cv, label, r.center(), INK, &f.small, "center");
            rects.push(r);
        }
        if let (Some(grow), true) = (t.outline_panel, panel_shown) {
            let view = Rect::new(self.lay.play_w + 12, 54, 356, 193);
            self.cv.stroke_rect2(Rect::new(view.x - 4, view.y - 4, view.w + 8, view.h + 8), t.accent, 3 + grow, 6);
        }
        (rects[0], rects[1])
    }

    /// `Extras.draw_notd`: the Neuron of the Day card, centred on the play area above where the science card goes.
    fn notd(&mut self, n: &NotdView) -> HudHits {
        let f = self.f;
        let mut hits = HudHits::default();
        let base = Rect::new(self.lay.play_w / 2 - 300, self.lay.hud_h - 250, 600, 104);
        let play_right = base.centerx() * 2 - 16;
        let left = 256.max(base.centerx() - 300);
        let w = 440.max(600.min(play_right - left));
        let r = Rect::new(left, base.y - 172, w, 162);
        card(self.cv, r, ([14, 20, 30], 240), 14);
        self.cv.stroke_rect2(r, ACCENT, 2, 14);
        text(self.cv, "NEURON OF THE DAY", (r.x + 16, r.y + 10), ACCENT, &f.small, "topleft");
        text(self.cv, &n.label, (r.x + 16, r.y + 26), INK, &f.head, "topleft");
        for (i, ln) in wrap_lines(&n.text, &f.text, r.w - 32).iter().take(3).enumerate() {
            text(self.cv, ln, (r.x + 16, r.y + 54 + i as i32 * 18), TEXT, &f.text, "topleft");
        }
        text(self.cv, &n.cite, (r.x + 16, r.bottom() - 50), LABEL, &f.small, "topleft");
        let bx = r.x + 16;
        if n.action != "none" {
            let tr_ = Rect::new(bx, r.bottom() - 30, 92, 24);
            self.cv.fill_rect(tr_, if n.look3d { [44, 50, 64] } else { [40, 110, 150] }, 8);
            text(self.cv, "Try it", tr_.center(), INK, &f.small, "center");
            hits.notd_try = Some(tr_);
            let mut hint = if n.look3d {
                "Tab frees the mouse to click".to_string()
            } else {
                format!("{}.", n.plan_text.split(". ").next().unwrap_or(""))
            };
            let room = r.right() - 178 - (tr_.right() + 10);
            while !hint.is_empty() && f.small.size(&hint) > room {
                hint.pop();
                hint.pop();
            }
            text(self.cv, &hint, (tr_.right() + 10, tr_.y + 5), LABEL, &f.small, "topleft");
        }
        let off = Rect::new(r.right() - 168, r.bottom() - 30, 152, 24);
        self.cv.fill_rect(off, [44, 50, 64], 8);
        text(self.cv, "Don't show again", off.center(), TEXT, &f.small, "center");
        hits.notd_off = Some(off);
        let x = Rect::new(r.right() - 30, r.y + 6, 24, 22);
        text(self.cv, "x", x.center(), LABEL, &f.bold, "center");
        hits.notd_close = Some(x);
        text(self.cv, "LITERATURE", (r.right() - 110, r.y + 12), [120, 200, 140], &f.small, "topleft");
        hits
    }

    /// `Game3D._draw_help`: the CONTROLS panel.
    fn help(&mut self) {
        let f = self.f;
        let panel = Rect::new(135, 80, 620, 64 + 30 * HELP3D.len() as i32);
        card(self.cv, panel, ([18, 21, 28], 255), 16);
        self.cv.stroke_rect(panel, BORDER_PANEL, 16);
        text(self.cv, "CONTROLS", (panel.x + 24, panel.y + 16), INK, &f.head, "topleft");
        for (k, (key, what)) in HELP3D.iter().enumerate() {
            let y = panel.y + 54 + k as i32 * 30;
            text(self.cv, key, (panel.x + 30, y), AMBER, &f.bold, "topleft");
            text(self.cv, what, (panel.x + 190, y), TEXT, &f.text, "topleft");
        }
    }

    fn status(&mut self, st: &HudState, now: f64) {
        let play_w = self.lay.play_w;
        let f = self.f;
        if st.hide_hud {
            let msg = "AUTOPILOT / SPECTATOR   (Y: exit)";
            let w = f.small.size(msg);
            let bx = Rect::new(play_w / 2 - w / 2 - 8, 14 - 3, w + 16, f.small.linesize + 6);
            card(self.cv, bx, ([10, 12, 18], 160), 6);
            text(self.cv, msg, bx.center(), [130, 160, 190], &f.small, "center");
            return;
        }
        card(self.cv, Rect::new(10, 8, 236, 62), ([10, 12, 18], 170), 10);
        text(self.cv, &st.fly_state.to_uppercase(), (22, 12), AMBER, &f.head, "topleft");
        let flies_txt = if st.flies > 1 {
            format!("   flies {}/{} · #{}", st.flies, st.max_flies, st.focus + 1)
        } else {
            format!("   flies 1/{} (N)", st.max_flies)
        };
        text(self.cv, &format!("hits {}   kills {}{flies_txt}", st.hits, st.kills), (22, 42), TEXT, &f.text, "topleft");
        // health
        let (bw, bx, by) = (300, play_w / 2 - 150, 16);
        let frac = st.health / st.max_health;
        let col = if st.immortal {
            [230, 185, 60]
        } else if frac > 0.5 {
            S_GOOD
        } else if frac > 0.25 {
            S_WARN
        } else {
            S_CRIT
        };
        self.cv.fill_rect(Rect::new(bx - 4, by - 4, bw + 8, 26), [10, 12, 18], 13);
        if frac > 0.0 {
            let fw = (bw as f32 * frac) as i32;
            self.cv.fill_rect(Rect::new(bx, by, fw.max(18), 18), col, 9);
            self.cv.fill_rect(Rect::new(bx + 6, by + 3, (fw - 12).max(6), 4), col.map(|c| c.saturating_add(60)), 2);
        }
        let label = if st.fly_dead {
            "DEAD".to_string()
        } else {
            format!("HEALTH {:.0}{}", st.health, if st.immortal { "  IMMORTAL" } else { "" })
        };
        text(self.cv, &label, (play_w / 2, by + 9), INK, &f.bold, "center");
        // the hint line, top right
        let mut bits = vec![format!("arena: {} (E)", st.arena)];
        bits.extend(st.arena_status.iter().cloned());
        if st.surgery_on {
            bits.push("SURGERY ON (O)".into());
        }
        if st.muted {
            bits.push("muted (M)".into());
        }
        bits.push(if st.arena_status.is_empty() { "F11 fullscreen   H help".into() } else { "H help".into() });
        let hint = bits.join("   ");
        let w = f.small.size(&hint);
        let hb = Rect::new(play_w - 14 - w - 7, 44 - 3, w + 14, f.small.linesize + 6);
        card(self.cv, hb, ([10, 12, 18], 170), 6);
        text(self.cv, &hint, hb.center(), if st.surgery_on { AMBER } else { TEXT }, &f.small, "center");
        if let Some(m) = &st.saved_msg {
            text(self.cv, m, (play_w / 2, 66), [170, 230, 190], &f.small, "midtop");
        }
        self.pain_card(st);
        self.reward_card(st);
        self.memory_card(st);
        let _ = now;
    }

    fn pain_card(&mut self, st: &HudState) {
        let f = self.f;
        let (x, y, w, h) = (10, 78, 236, 222);
        card(self.cv, Rect::new(x, y, w, h), ([10, 12, 18], 180), 10);
        let p = st.pain;
        let (word, col) = if p >= 99.0 {
            ("MAXED OUT", S_CRIT)
        } else if p >= 70.0 {
            ("agony", S_CRIT)
        } else if p >= 35.0 {
            ("severe", [236, 131, 90])
        } else if p >= 10.0 {
            ("mild", S_WARN)
        } else {
            ("calm", S_GOOD)
        };
        text(self.cv, "PAIN", (x + 12, y + 8), LABEL, &f.small, "topleft");
        if st.pain_connectome {
            text(self.cv, "CONNECTOME (noci)", (x + 46, y + 8), [50, 220, 100], &f.small, "topleft");
        } else {
            text(self.cv, "ESTIMATE (RULE)", (x + 46, y + 8), DIM, &f.small, "topleft");
        }
        let r = text(self.cv, &format!("{p:.0}"), (x + 12, y + 20), INK, &f.title, "topleft");
        text(self.cv, "/100", (r.right() + 4, r.bottom() - 22), LABEL, &f.small, "topleft");
        text(self.cv, word, (x + w - 12, y + 30), col, &f.bold, "topright");
        let (bx, bw) = (x + 12, w - 24);
        self.cv.fill_rect(Rect::new(bx, y + 64, bw, 10), [30, 36, 48], 5);
        if p > 0.5 {
            self.cv.fill_rect(Rect::new(bx, y + 64, 8.max((bw as f32 * p / 100.0) as i32), 10), col, 5);
        }
        let mut yy = y + 84;
        for (label, v) in ["touch overload", "heat / cold", "chemical", "DN alarm", "body relay"].iter().zip(st.pain_parts) {
            text(self.cv, label, (bx, yy - 3), TEXT, &f.small, "topleft");
            bar(self.cv, bx + 112, yy, bw - 146, v, [170, 176, 188], None);
            text(self.cv, &format!("{:3.0}", v * 100.0), (bx + bw, yy - 3), TEXT, &f.small, "topright");
            yy += 15;
        }
        let tr = &st.pain_trace;
        let (sy, sh) = (y + 162, 22);
        self.cv.fill_rect(Rect::new(bx, sy, bw, sh), [22, 26, 34], 3);
        if tr.len() > 2 {
            let pts: Vec<(f32, f32)> = (0..tr.len())
                .step_by(3)
                .map(|k| (bx as f32 + k as f32 * bw as f32 / 719.0, (sy + sh) as f32 - tr[k] / 100.0 * (sh - 2) as f32))
                .collect();
            if pts.len() > 1 {
                self.cv.aa_lines(&pts, S_CRIT);
            }
        }
        text(self.cv, &format!("peak {:.0}   maxed {:.1}s", st.pain_peak, st.pain_max_s), (bx, y + h - 34), LABEL, &f.small, "topleft");
        text(
            self.cv,
            &format!("P  {} ({} neurons)", st.pain_level_name, group3(st.pain_neurons)),
            (bx, y + h - 18),
            if st.pain_level > 0 { AMBER } else { TEXT },
            &f.small,
            "topleft",
        );
    }

    fn reward_card(&mut self, st: &HudState) {
        let f = self.f;
        let (x, y, w, h) = (10, 308, 236, 64);
        card(self.cv, Rect::new(x, y, w, h), ([10, 12, 18], 180), 10);
        let r = st.reward;
        let (word, col): (&str, Color) = if r >= 70.0 {
            ("bliss", [255, 120, 170])
        } else if r >= 30.0 {
            ("happy", [255, 160, 190])
        } else if r >= 8.0 {
            ("pleased", [220, 190, 200])
        } else {
            ("neutral", LABEL)
        };
        text(self.cv, "REWARD", (x + 12, y + 8), LABEL, &f.small, "topleft");
        text(self.cv, &format!("{r:.0}"), (x + 72, y + 3), INK, &f.head, "topleft");
        text(self.cv, word, (x + w - 12, y + 6), col, &f.bold, "topright");
        self.cv.fill_rect(Rect::new(x + 12, y + 32, w - 24, 10), [30, 36, 48], 5);
        if r > 0.5 {
            self.cv.fill_rect(Rect::new(x + 12, y + 32, 8.max(((w - 24) as f32 * r / 100.0) as i32), 10), col, 5);
        }
        text(self.cv, &format!("PAM dopamine neurons x{:.2} calm", st.reward_level), (x + 12, y + 46), LABEL, &f.small, "topleft");
    }

    fn memory_card(&mut self, st: &HudState) {
        let f = self.f;
        let (x, y, w) = (10, 380, 236);
        let mut rows: Vec<&(String, f32, f32)> = st.memory.iter().filter(|r| r.1.max(r.2) >= 0.02).collect();
        rows.sort_by(|a, b| b.1.max(b.2).total_cmp(&a.1.max(a.2)));
        rows.truncate(4);
        let h = 46 + 16 * rows.len().max(1) as i32;
        card(self.cv, Rect::new(x, y, w, h), ([10, 12, 18], 180), 10);
        text(self.cv, "MEMORY", (x + 12, y + 8), LABEL, &f.small, "topleft");
        text(self.cv, "mushroom body", (x + w - 12, y + 8), DIM, &f.small, "topright");
        let mut yy = y + 28;
        if rows.is_empty() {
            text(self.cv, "nothing learned yet", (x + 12, yy), DIM, &f.small, "topleft");
        }
        for (key, fear, like) in rows {
            let good = like > fear;
            let v = if good { *like } else { *fear };
            text(self.cv, key, (x + 12, yy - 3), TEXT, &f.small, "topleft");
            bar(self.cv, x + 84, yy, w - 150, v, if good { [255, 150, 190] } else { [240, 150, 60] }, None);
            text(self.cv, &format!("{} {v:.2}", if good { "likes" } else { "fears" }), (x + w - 12, yy - 3), TEXT, &f.small, "topright");
            yy += 16;
        }
        text(self.cv, "real synapses · T: train", (x + 12, y + h - 16), DIM, &f.small, "topleft");
    }

    fn toolbar(&mut self, st: &HudState) {
        if st.hide_hud {
            return;
        }
        let f = self.f;
        let floor = self.lay.hud_h - 120;
        let play_w = self.lay.play_w;
        let (bw, gap) = (58, 4);
        let n = st.tools.len() as i32;
        let x0 = (play_w - bw * n - gap * (n - 1)) / 2;
        text(self.cv, &st.tool_line, (play_w / 2, floor + 10), INK, &f.bold, "midtop");
        for (k, t) in st.tools.iter().enumerate() {
            let r = Rect::new(x0 + k as i32 * (bw + gap), floor + 38, bw, 64);
            let on = t.name == st.tool_name;
            card(self.cv, r, (if on { [60, 46, 22] } else { [16, 18, 24] }, if on { 230 } else { 210 }), 10);
            self.cv.stroke_rect2(r, if on { AMBER } else { BORDER_PANEL }, 2, 10);
            draw_icon(self.cv, &t.name, (r.centerx() as f32, (r.y + 24) as f32), if on { AMBER } else { TEXT });
            text(self.cv, &t.key.to_uppercase(), (r.x + 7, r.y + 4), LABEL, &f.small, "topleft");
            let label: String = t.label.chars().take(9).collect();
            text(self.cv, &label, (r.centerx(), r.y + 44), if on { AMBER } else { INK }, &f.small, "midtop");
        }
        if st.pages > 1 {
            let y = floor + 38 + 18;
            for (x, sym) in [(x0 - 30, "<"), (x0 + n * (bw + gap) + 2, ">")] {
                let r = Rect::new(x, y, 28, 28);
                card(self.cv, r, ([16, 18, 24], 210), 8);
                self.cv.stroke_rect(r, BORDER_PANEL, 8);
                text(self.cv, sym, r.center(), INK, &f.bold, "center");
            }
            text(self.cv, &format!("{}/{}", st.page + 1, st.pages), (x0 - 16, y + 34), LABEL, &f.small, "center");
        }
    }

    fn panel(&mut self, st: &HudState, p: &PanelData, now: f64) {
        let f = self.f;
        let lay = self.lay;
        let x0 = lay.play_w;
        let (w, h) = (lay.hud_w, lay.hud_h);
        let pa = st.panel_alpha.min(255);
        self.cv.fill_rect_a(Rect::new(x0, 0, w - x0, h), PANEL_BG, pa, 0);
        self.cv.line((x0, 0), (x0, h), 1, BORDER_PANEL, 255);
        let x = x0 + 12;
        text(self.cv, "THE FLY'S BRAIN", (x, 8), ACCENT, &f.head, "topleft");
        let (status, scol) = if p.dead { ("FLATLINE", S_CRIT) } else { ("LIVE", S_GOOD) };
        let r = text(self.cv, status, (w - 14, 13), scol, &f.bold, "topright");
        let lit = p.dead || p.calm_fx || (now * 2.0) as i64 % 2 != 0;
        self.cv.aa_circle(((r.x - 10) as f32, r.centery() as f32), 4.0, if lit { scol } else { DIM });
        text(self.cv, &format!("MaleCNS v1.0 connectome, {} neurons", group3(p.neurons as u32)), (x, 34), LABEL, &f.small, "topleft");
        let (nw, nh) = (356, 193);
        let view = Rect::new(x, 54, nw, nh);
        if p.big_view {
            self.cv.fill_rect(view, [4, 5, 8], 0);
            text(self.cv, "shown in big view", view.center(), DIM, &f.small, "center");
        } else {
            // the brain image is the 3D side's texture, shown through this hole
            self.cv.clear_rect(view);
            text(self.cv, "B: big view", (x + nw - 6, 54 + nh - 16), LABEL, &f.small, "topright");
        }
        let mut y = 54 + nh + 12;
        let bw = w - x - 12;
        y = self.card_head(x, y, bw, "TOUCH NEURONS", "spikes/s per neuron", 4, pa);
        for (label, hz) in ["head BM/JO", "body SNta", "legs SNpp", "wing WG"].iter().zip(p.touch) {
            text(self.cv, label, (x + 8, y - 3), TEXT, &f.small, "topleft");
            bar(self.cv, x + 104, y, bw - 150, hz / 60.0, [90, 200, 120], None);
            text(self.cv, &format!("{hz:4.0}"), (x + bw - 8, y - 3), TEXT, &f.small, "topright");
            y += 16;
        }
        y += 10;
        y = self.card_head(x, y, bw, "DESCENDING NEURONS", "x calm baseline", p.motor.len() as i32, pa);
        for m in &p.motor {
            let over = m.thresh.is_some_and(|t| m.level > t);
            text(self.cv, &m.label, (x + 8, y - 3), if over { AMBER } else { TEXT }, &f.small, "topleft");
            bar(self.cv, x + 200, y, bw - 246, m.level / 5.0, if over { AMBER } else { ACCENT }, m.thresh.map(|t| t / 5.0));
            text(self.cv, &format!("{:4.1}", m.level), (x + bw - 8, y - 3), TEXT, &f.small, "topright");
            y += 16;
        }
        y += 10;
        y = self.card_head(x, y, bw, "WHOLE BRAIN", "spikes/s, last 12 s", 3, pa);
        let tr = &p.whole_trace;
        let th = 40;
        if tr.len() > 2 {
            let top = tr.iter().cloned().fold(0.0f32, f32::max).max(8.0);
            let pts: Vec<(f32, f32)> = (0..tr.len())
                .step_by(2)
                .map(|k| {
                    ((x + 8) as f32 + k as f32 * (bw - 16) as f32 / (tr.len() - 1) as f32, (y + th) as f32 - tr[k] / top * (th - 4) as f32)
                })
                .collect();
            self.cv.aa_lines(&pts, ACCENT);
            text(self.cv, &format!("{:.1}", tr[tr.len() - 1]), (x + bw - 8, y - 6), TEXT, &f.small, "topright");
        }
        y += th + 12;
        text(self.cv, "REACTIONS", (x, y), LABEL, &f.small, "topleft");
        y += 16;
        let room = ((h - 36 - y) / 15).clamp(1, 4) as usize;
        let start = p.log.len().saturating_sub(room);
        for l in p.log[start..].iter().rev() {
            let col = if l.age < 1.0 {
                AMBER
            } else if l.age < 5.0 {
                TEXT
            } else {
                DIM
            };
            text(self.cv, &format!("{:4.1}s  {}", l.age, l.text), (x, y), col, &f.small, "topleft");
            if let (true, Some(src)) = (st.tags_on, &l.source) {
                source_chip(self.cv, (w - 14, y), src, &f.small, true, 255);
            }
            y += 15;
        }
        text(self.cv, &format!("sim {:4.0} steps/s  {:4.1} ms/step", p.steps_per_s, p.step_ms), (x, h - 18), DIM, &f.small, "topleft");
    }

    fn card_head(&mut self, x: i32, y: i32, w: i32, title: &str, unit: &str, rows: i32, pa: u32) -> i32 {
        let h = 24 + rows * 16;
        self.cv.fill_rect_a(Rect::new(x - 4, y - 4, w + 8, h + 4), CARD, (pa + 40).min(255), 8);
        text(self.cv, title, (x + 4, y), LABEL, &self.f.small, "topleft");
        text(self.cv, unit, (x + w - 6, y), DIM, &self.f.small, "topright");
        y + 22
    }
}

fn wrap_lines(text: &str, font: &Font, width: i32) -> Vec<String> {
    let (mut lines, mut line) = (vec![], String::new());
    for w in text.split_whitespace() {
        let trial = if line.is_empty() { w.to_string() } else { format!("{line} {w}") };
        if font.size(&trial) > width && !line.is_empty() {
            lines.push(std::mem::take(&mut line));
            line = w.to_string();
        } else {
            line = trial;
        }
    }
    lines.push(line);
    lines
}

fn title_case(s: &str) -> String {
    s.split(' ')
        .map(|w| w.chars().next().map(|c| c.to_uppercase().collect::<String>() + &w[c.len_utf8()..]).unwrap_or_default())
        .collect::<Vec<_>>()
        .join(" ")
}

/// `{:,}`
fn group3(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `HELP3D` (kick3d.py): what the H overlay lists.
pub const HELP3D: [(&str, &str); 27] = [
    ("WASD", "walk (Shift sprint, Ctrl crouch)"),
    ("Mouse", "look; left click uses the tool in your hand"),
    ("1-9, 0 / wheel", "pick a hotbar tool; - and = turn the page of a long loadout"),
    ("Q", "loadout editor: choose which tools are on the hotbar"),
    ("` (hold)", "tool wheel: every tool, one gesture away"),
    ("Gamepad", "sticks walk and look, RT/ZR uses, LB/RB cycle the loadout, hold Y (wheel) pick any tool, Start/+ menu"),
    ("Tab", "free the mouse to click the brain panel and menus"),
    ("B", "big live brain view; click a neuron to inspect it"),
    ("O", "brain surgery"),
    ("T", "training: teach it to fear or like a smell (saved)"),
    ("X", "1v1 duel: the fly gets a blaster and can kill you"),
    ("E", "arena: room, fan, flypaper, pool, lamp, escape room, open field, orchard, thermo, kitchen"),
    ("J", "outdoors: call back a fly that flew out of sight"),
    ("P / I", "pain neurons / immortal mode"),
    ("M", "mute"),
    ("F12 / G", "save a screenshot / a GIF of the last 6 s"),
    ("Shift+R", "record a video (MP4 with ffmpeg, else GIF; again to stop)"),
    ("V", "brain panel: solid, see-through, faint, hidden"),
    ("U", "menu size: crisp (whole-pixel scaling) or large"),
    ("F11", "fullscreen"),
    ("N", "spawn another fly, each with its own brain (16; one per CPU core, up to 32, on Numba)"),
    ("F", "cycle focused fly (in photo mode: autofocus)"),
    ("R", "reset to a single fresh fly"),
    ("Z  [  ]  .", "pause time, slower, faster, single step"),
    ("F3", "profiler overlay: sim, physics, render and UI milliseconds, FPS, engine (off at every launch)"),
    (
        "F4",
        "fly's-eye view: the scene as a fly's eyes sample it (wide field, hexagonal ommatidia, UV-blue-green color); visual only, not fed to the brain",
    ),
    ("Esc", "close a panel, or open the menu (settings, save, quit)"),
];
