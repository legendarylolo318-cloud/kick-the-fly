//! The clickable pause menu and Settings screen: `legacy-python/kickthefly/ui/menu.py`, function for function.
//!
//! Immediate mode like the original: each frame the current screen draws its widgets and registers where they are; the next mouse
//! or key event is matched against those regions. Python's lambdas become [`Act`] values (data), so the menu needs no references
//! to the game: [`Menu::handle`] applies what it can itself (tabs, key capture, typing) and returns [`Effect`]s for the host
//! (the game) to carry out.

use std::collections::HashMap;

use ktf_data::config::{self, Config, Setting};
use serde_json::Value;

use crate::canvas::{Canvas, Rect};
use crate::font::Font;
use crate::i18n::tr;
use crate::theme::*;

#[derive(Debug, Clone, PartialEq)]
pub enum UiEvent {
    /// `name` is `pygame.key.name(key)` ("escape", "return", "w", "left shift", "f12"); `ch` the typed character, if any.
    KeyDown {
        name: String,
        ch: Option<char>,
        ctrl: bool,
    },
    MouseDown {
        pos: (i32, i32),
    },
    MouseUp {
        pos: (i32, i32),
    },
    Motion {
        pos: (i32, i32),
    },
    /// Positive scrolls up.
    Wheel {
        y: f32,
    },
    /// A gamepad button or axis pressed while a pad binding waits (`gamepad.capture_binding` already named it).
    PadBinding(String),
}

pub type Key = String;

/// What the host has to do after an event.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    /// `host.menu_action(name)`: "resume", "quit", "save_state", "click_sound", "closed", ...
    Action(String),
    /// A setting changed in `cfg` (`None` key: only apply and save, e.g. when a slider is released). `save` false while dragging.
    SettingChanged { key: Option<String>, save: bool },
    /// Key bindings changed (`host.set_setting("keys", None)`).
    KeysChanged,
}

/// What a button does when clicked (Python's lambdas).
#[derive(Debug, Clone, PartialEq)]
pub enum Act {
    Action(String),
    Set {
        key: String,
        value: Value,
        save: bool,
    },
    /// `host.set_setting(None, None)`
    ApplySave,
    Tab(String),
    Capture(String),
    Back,
    ResetTab,
    RandomSeed(String),
    Nothing,
}

#[derive(Debug, Clone)]
enum EditTarget {
    Setting { key: String, pct: bool, int: bool },
    Text(String),
}

#[derive(Debug, Clone)]
enum Kind {
    Button { click: Act },
    Slider { track: Rect, lo: f64, hi: f64, step: f64, key: String },
    Edit { text: String, target: EditTarget, alpha: bool, limit: usize, compact: bool },
    Label,
}

#[derive(Debug, Clone)]
struct Hit {
    rect: Rect,
    kind: Kind,
    id: String,
    enabled: bool,
    #[allow(dead_code)]
    tip: Option<String>,
    clip: Option<Rect>,
}

#[derive(Debug, Clone)]
struct Drag {
    id: String,
    track: Rect,
    lo: f64,
    hi: f64,
    step: f64,
    key: String,
}

#[derive(Debug, Clone)]
struct Edit {
    id: String,
    text: String,
    target: EditTarget,
    fresh: bool,
    alpha: bool,
    limit: usize,
    compact: bool,
}

#[derive(Clone)]
struct Fonts {
    small: Font,
    text: Font,
    bold: Font,
    head: Font,
    title: Font,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Anchor {
    TopLeft,
    MidLeft,
    Center,
    MidTop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Normal,
    Primary,
    Danger,
}

pub struct Menu {
    pub screen: Option<String>,
    stack: Vec<String>,
    pub tab: String,
    scroll: HashMap<String, f32>,
    hits: Vec<Hit>,
    pressed: Option<String>,
    drag: Option<Drag>,
    edit: Option<Edit>,
    pub capture: Option<String>,
    message: Option<(String, f64, Color)>,
    hover_id: Option<String>,
    hover_since: f64,
    tip: Option<(Rect, String)>,
    clip: Option<Rect>,
    content_h: HashMap<String, i32>,
    fonts: Option<(bool, Fonts)>,
    mouse: (i32, i32),
    now: f64,
    fx: Vec<Effect>,
    /// "Kick the Fly 3.1.1" version shown under PAUSED.
    pub version: String,
    /// Lines of the ENGINE panel (Settings > Brain), set by the host.
    pub engine_lines: Vec<String>,
    /// `Some("connected: Xbox pad")` or `Some("none connected")` for the gamepad block of Settings > Controls (3D game only).
    pub pad_status: Option<String>,
    /// Alpha of the dark veil under the panel: 196 as in pygame, which blends in sRGB space. Bevy blends in linear light, where the same
    /// look needs about 240 (see PORTING.md, GUI parity).
    pub veil_alpha: u32,
    rng: u64,
    text_commits: Vec<(String, String)>,
}

/// `config.PAD_BUTTONS`-aware text of a gamepad binding (`gamepad.pretty`).
pub fn pad_pretty(binding: &str) -> String {
    if binding.is_empty() {
        return "unbound".into();
    }
    let named = match binding {
        "leftx" => "left stick X",
        "lefty" => "left stick Y",
        "rightx" => "right stick X",
        "righty" => "right stick Y",
        "lefttrigger" => "left trigger",
        "righttrigger" => "right trigger",
        "leftshoulder" => "left bumper",
        "rightshoulder" => "right bumper",
        "leftstick" => "left stick click",
        "rightstick" => "right stick click",
        "back" => "back / select / -",
        "start" => "start / +",
        "guide" => "home / guide",
        _ => "",
    };
    if !named.is_empty() {
        named.into()
    } else if config::schema().pad_buttons.iter().any(|b| b == binding) {
        binding.to_uppercase()
    } else {
        binding.into()
    }
}

fn g(x: f64) -> String {
    ktf_data::sharecode::g(x)
}

/// Python's `fmt.format(value)` for the handful of specs the settings use (`{:g}`, `{:.0%}`, `{:.1f}`, `{:.0f} s`, `{:.0f}°`, `{:.1f}m`, ...).
pub fn py_fmt(fmt: &str, v: f64) -> String {
    let (Some(a), Some(b)) = (fmt.find("{:"), fmt.find('}')) else { return fmt.to_string() };
    let spec = &fmt[a + 2..b];
    let body = if spec == "g" {
        g(v)
    } else if let Some(p) = spec.strip_suffix("%") {
        let digits: usize = p.trim_start_matches('.').parse().unwrap_or(0);
        format!("{:.*}%", digits, v * 100.0)
    } else if let Some(p) = spec.strip_suffix('f') {
        let digits: usize = p.trim_start_matches('.').parse().unwrap_or(6);
        format!("{:.*}", digits, v)
    } else {
        g(v)
    };
    format!("{}{}{}", &fmt[..a], body, &fmt[b + 1..])
}

fn num_eq(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if a.is_number() && b.is_number() => x == y,
        _ => a == b,
    }
}

impl Menu {
    pub fn new(version: &str) -> Self {
        Menu {
            screen: None,
            stack: vec![],
            tab: config::TABS[0].into(),
            scroll: HashMap::new(),
            hits: vec![],
            pressed: None,
            drag: None,
            edit: None,
            capture: None,
            message: None,
            hover_id: None,
            hover_since: 0.0,
            tip: None,
            clip: None,
            content_h: HashMap::new(),
            fonts: None,
            mouse: (0, 0),
            now: 0.0,
            fx: vec![],
            version: version.into(),
            engine_lines: vec![],
            pad_status: None,
            veil_alpha: 196,
            rng: 0x9E37_79B9_7F4A_7C15,
            text_commits: vec![],
        }
    }

    fn f(&self) -> &Fonts {
        &self.fonts.as_ref().expect("fonts are made on the first draw").1
    }

    /// Make the fonts for the current `access.larger_text` (`Menu.fonts`).
    fn ensure_fonts(&mut self, big: bool) {
        if self.fonts.as_ref().is_none_or(|f| f.0 != big) {
            let k = if big { 1.25 } else { 1.0 };
            let sz = |n: f64| (n * k).round_ties_even() as i32;
            self.fonts = Some((
                big,
                Fonts {
                    small: Font::new(sz(14.0), false),
                    text: Font::new(sz(16.0), false),
                    bold: Font::new(sz(16.0), true),
                    head: Font::new(sz(22.0), true),
                    title: Font::new(sz(38.0), true),
                },
            ));
        }
    }

    // --- navigation ---------------------------------------------------------------------------------------------------------
    pub fn is_open(&self) -> bool {
        self.screen.is_some()
    }

    pub fn show(&mut self, screen: &str) {
        match &self.screen {
            None => self.stack.clear(),
            Some(s) if s != screen => self.stack.push(s.clone()),
            _ => {}
        }
        self.screen = Some(screen.into());
        self.edit = None;
        self.capture = None;
        self.drag = None;
    }

    pub fn back(&mut self) {
        self.edit = None;
        self.capture = None;
        self.drag = None;
        match self.stack.pop() {
            Some(s) => self.screen = Some(s),
            None => self.close(),
        }
    }

    pub fn close(&mut self) {
        self.screen = None;
        self.stack.clear();
        self.edit = None;
        self.capture = None;
        self.fx.push(Effect::Action("closed".into()));
    }

    pub fn flash(&mut self, text: &str, color: Color, seconds: f64) {
        self.message = Some((text.into(), self.now + seconds, color));
    }

    // --- events -------------------------------------------------------------------------------------------------------------
    fn hit(&self, pos: (i32, i32)) -> Option<&Hit> {
        self.hits.iter().rev().find(|h| h.rect.contains(pos) && h.clip.is_none_or(|c| c.contains(pos)))
    }

    /// Consume an event while the menu is open. `pos` is the mouse in HUD coordinates. Returns (consumed, what the host must do).
    pub fn handle(&mut self, ev: &UiEvent, cfg: &mut Config, three_d: bool) -> (bool, Vec<Effect>) {
        let consumed = self.handle_inner(ev, cfg, three_d);
        (consumed, std::mem::take(&mut self.fx))
    }

    fn handle_inner(&mut self, ev: &UiEvent, cfg: &mut Config, three_d: bool) -> bool {
        if !self.is_open() {
            return false;
        }
        if let (Some(cap), UiEvent::PadBinding(b)) = (&self.capture, ev)
            && let Some(action) = cap.strip_prefix("pad:")
        {
            let action = action.to_string();
            self.bind_pad(cfg, &action, b);
            return true;
        }
        match ev {
            UiEvent::PadBinding(_) => true,
            UiEvent::KeyDown { name, ch, ctrl } => self.key(name, *ch, *ctrl, cfg),
            UiEvent::MouseDown { pos } => {
                if self.edit.is_some() {
                    self.commit_edit(cfg);
                }
                let Some(h) = self.hit(*pos).cloned() else { return true };
                match h.kind {
                    Kind::Slider { track, lo, hi, step, key } => {
                        self.drag = Some(Drag { id: h.id, track, lo, hi, step, key });
                        self.drag_to(pos.0, cfg);
                    }
                    Kind::Edit { text, target, alpha, limit, compact } => {
                        self.edit = Some(Edit { id: h.id, text, target, fresh: true, alpha, limit, compact });
                    }
                    _ => self.pressed = Some(h.id),
                }
                true
            }
            UiEvent::Motion { pos } => {
                if self.drag.is_some() {
                    self.drag_to(pos.0, cfg);
                }
                true
            }
            UiEvent::MouseUp { pos } => {
                if let Some(d) = self.drag.take() {
                    self.fx.push(Effect::SettingChanged { key: None, save: true });
                    let _ = d;
                    return true;
                }
                if let Some(h) = self.hit(*pos).cloned()
                    && Some(&h.id) == self.pressed.as_ref()
                    && h.enabled
                    && let Kind::Button { click } = h.kind
                {
                    self.pressed = None;
                    self.fx.push(Effect::Action("click_sound".into()));
                    self.run(click, cfg, three_d);
                }
                self.pressed = None;
                true
            }
            UiEvent::Wheel { y } => {
                let key = self.scroll_key();
                let limit = self.content_h.get(&key).copied().unwrap_or(0).max(0) as f32;
                let v = self.scroll.get(&key).copied().unwrap_or(0.0) - y * 40.0;
                self.scroll.insert(key, v.clamp(0.0, limit));
                true
            }
        }
    }

    /// Where the widget with this id was drawn last frame (tests and scripted captures click through it).
    pub fn center_of(&self, id: &str) -> Option<(i32, i32)> {
        self.hits.iter().find(|h| h.id == id).map(|h| h.rect.center())
    }

    fn scroll_key(&self) -> String {
        let s = self.screen.clone().unwrap_or_default();
        if s == "settings" { format!("{s}:{}", self.tab) } else { s }
    }

    fn run(&mut self, act: Act, cfg: &mut Config, _three_d: bool) {
        match act {
            Act::Action(a) => self.fx.push(Effect::Action(a)),
            Act::Set { key, value, save } => self.set_setting(cfg, &key, &value, save),
            Act::ApplySave => self.fx.push(Effect::SettingChanged { key: None, save: true }),
            Act::Tab(t) => self.tab = t,
            Act::Capture(c) => self.capture = Some(c),
            Act::Back => self.back(),
            Act::ResetTab => self.reset_tab(cfg),
            Act::RandomSeed(key) => {
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 7;
                self.rng ^= self.rng << 17;
                let v = 1 + (self.rng % (2u64.pow(31) - 2)) as i64;
                self.set_setting(cfg, &key, &Value::from(v), true);
            }
            Act::Nothing => {}
        }
    }

    /// `host.set_setting(key, value)`: validate and store in `cfg`; the host applies it live and saves.
    fn set_setting(&mut self, cfg: &mut Config, key: &str, value: &Value, save: bool) {
        match cfg.set(key, value) {
            Ok(_) => self.fx.push(Effect::SettingChanged { key: Some(key.into()), save }),
            Err(e) => self.flash(&e, BAD, 4.0),
        }
    }

    fn reset_tab(&mut self, cfg: &mut Config) {
        let tab = self.tab.clone();
        let changed = cfg.reset_tab(&tab);
        for k in &changed {
            self.fx.push(if k == "keys" { Effect::KeysChanged } else { Effect::SettingChanged { key: Some(k.clone()), save: false } });
        }
        self.fx.push(Effect::SettingChanged { key: None, save: true });
        let msg = if changed.is_empty() { format!("{tab} already at defaults") } else { format!("{tab} reset to defaults") };
        self.flash(&msg, GOOD, 4.0);
    }

    fn bind_pad(&mut self, cfg: &mut Config, action: &str, binding: &str) {
        self.capture = None;
        let (ok, msg) = cfg.bind_pad(action, binding);
        if ok {
            self.fx.push(Effect::KeysChanged);
        }
        let label = config::schema().pad_actions.iter().find(|(a, _, _)| a == action).map_or("", |(_, l, _)| l.as_str());
        if msg.is_empty() {
            self.flash(&format!("{label}: {}", pad_pretty(binding)), GOOD, 4.0);
        } else {
            self.flash(&msg, AMBER, 4.0);
        }
    }

    fn key(&mut self, name: &str, ch: Option<char>, ctrl: bool, cfg: &mut Config) -> bool {
        if let Some(c) = self.capture.clone() {
            if let Some(action) = c.strip_prefix("pad:") {
                if name == "escape" {
                    self.capture = None;
                } else if name == "backspace" || name == "delete" {
                    let a = action.to_string();
                    self.bind_pad(cfg, &a, "");
                }
                return true;
            }
            self.capture = None;
            if name == "escape" {
                return true;
            }
            let (ok, msg) = cfg.bind(&c, name);
            if ok {
                self.fx.push(Effect::KeysChanged);
            }
            if msg.is_empty() {
                self.flash(&format!("{}: {name}", Config::action_label(&c)), GOOD, 4.0);
            } else {
                self.flash(&msg, AMBER, 4.0);
            }
            return true;
        }
        if self.edit.is_some() {
            match name {
                "return" | "enter" => self.commit_edit(cfg),
                "escape" => self.edit = None,
                "backspace" => {
                    let e = self.edit.as_mut().unwrap();
                    if std::mem::take(&mut e.fresh) {
                        e.text.clear();
                    } else {
                        e.text.pop();
                    }
                }
                "v" if ctrl && self.edit.as_ref().is_some_and(|e| e.alpha) => {
                    // paste: the host puts the clipboard text in `ch`'s place through `paste_text` (only on Ctrl+V in a text box)
                }
                _ => {
                    let e = self.edit.as_mut().unwrap();
                    if let Some(c) = ch.filter(|c| if e.alpha { !c.is_control() } else { "0123456789.-".contains(*c) })
                        && e.text.chars().count() < e.limit
                    {
                        if std::mem::take(&mut e.fresh) {
                            e.text.clear();
                        }
                        e.text.push(c);
                    }
                }
            }
            return true;
        }
        if name == "escape" {
            self.back();
        }
        true
    }

    /// Text pasted with Ctrl+V into an open text box (the host reads the clipboard only then).
    pub fn paste_text(&mut self, pasted: &str) {
        if let Some(e) = self.edit.as_mut().filter(|e| e.alpha) {
            let text: String =
                if e.compact { pasted.split_whitespace().collect() } else { pasted.split_whitespace().collect::<Vec<_>>().join(" ") };
            if !text.is_empty() {
                if std::mem::take(&mut e.fresh) {
                    e.text.clear();
                }
                e.text.push_str(&text);
                e.text = e.text.chars().take(e.limit).collect();
            }
        }
    }

    pub fn is_editing_text(&self) -> bool {
        self.edit.as_ref().is_some_and(|e| e.alpha)
    }

    /// The text being typed in a text box, if any (for the host's own checks).
    pub fn edit_text(&self) -> Option<&str> {
        self.edit.as_ref().map(|e| e.text.as_str())
    }

    /// Committed text of a text box (names, share codes), if one was just confirmed. Hosts with text pages poll this.
    pub fn take_text_commits(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.text_commits)
    }

    fn commit_edit(&mut self, cfg: &mut Config) {
        let Some(e) = self.edit.take() else { return };
        match e.target {
            EditTarget::Text(id) => self.text_commits.push((id, e.text)),
            EditTarget::Setting { key, pct, int } => match e.text.parse::<f64>() {
                Ok(v) => {
                    let v = if pct { v / 100.0 } else { v };
                    let value = if int { Value::from(v.round_ties_even() as i64) } else { Value::from(v) };
                    self.set_setting(cfg, &key, &value, false);
                    self.fx.push(Effect::SettingChanged { key: None, save: true });
                }
                Err(_) => self.flash("not a number", BAD, 4.0),
            },
        }
    }

    fn drag_to(&mut self, x: i32, cfg: &mut Config) {
        let Some(d) = self.drag.clone() else { return };
        let frac = ((x - d.track.x) as f64 / d.track.w.max(1) as f64).clamp(0.0, 1.0);
        let mut v = d.lo + frac * (d.hi - d.lo);
        if d.step != 0.0 {
            v = (v / d.step).round_ties_even() * d.step;
        }
        self.set_setting(cfg, &d.key, &Value::from(v), false);
    }
}

// --- widgets (called while drawing) ----------------------------------------------------------------------------------------------
impl Menu {
    fn fonts_now(&self) -> Fonts {
        self.f().clone()
    }

    /// Adds a hit region. Returns whether the mouse is over it (and inside the current clip); hovering with a tip for 0.35 s shows it.
    fn register(&mut self, rect: Rect, kind: Kind, id: &str, enabled: bool, tip: Option<&str>) -> bool {
        self.hits.push(Hit { rect, kind, id: id.into(), enabled, tip: tip.map(String::from), clip: self.clip });
        let m = self.mouse;
        let over = rect.contains(m) && self.clip.is_none_or(|c| c.contains(m));
        if over && let Some(t) = tip {
            if self.hover_id.as_deref() != Some(id) {
                self.hover_id = Some(id.into());
                self.hover_since = self.now;
            } else if self.now - self.hover_since > 0.35 {
                self.tip = Some((rect, t.into()));
            }
        }
        over
    }

    fn text(&self, cv: &mut Canvas, s: &str, pos: (i32, i32), color: Color, font: &Font, anchor: Anchor) -> Rect {
        let (w, h) = (font.size(s), font.linesize);
        let (x, y) = match anchor {
            Anchor::TopLeft => (pos.0, pos.1),
            Anchor::MidLeft => (pos.0, pos.1 - h / 2),
            Anchor::Center => (pos.0 - w / 2, pos.1 - h / 2),
            Anchor::MidTop => (pos.0 - w / 2, pos.1),
        };
        font.draw(cv, s, x, y, color);
        Rect::new(x, y, w, h)
    }

    /// Word-wrapped text; returns the y below it.
    fn wrapped(&self, cv: &mut Canvas, s: &str, pos: (i32, i32), width: i32, color: Color, font: &Font, max_lines: usize) -> i32 {
        let mut lines: Vec<String> = vec![];
        let mut line = String::new();
        for word in s.split_whitespace() {
            let trial = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
            if font.size(&trial) > width && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                line = word.to_string();
            } else {
                line = trial;
            }
        }
        lines.push(line);
        if lines.len() > max_lines {
            lines.truncate(max_lines);
            let last = lines.last_mut().unwrap();
            last.pop();
            last.push('…');
        }
        let mut y = pos.1;
        for ln in &lines {
            self.text(cv, ln, (pos.0, y), color, font, Anchor::TopLeft);
            y += font.linesize;
        }
        y
    }

    /// A label in at most `max_lines` lines of at most `width` px: word-wrapped, the last line cut with an ellipsis when it still does
    /// not fit (never a smaller font). Returns (lines, cut).
    pub fn fit_lines(font: &Font, s: &str, width: i32, max_lines: i32) -> (Vec<String>, bool) {
        if font.size(s) <= width || s.trim().is_empty() {
            return (vec![s.to_string()], false);
        }
        let mut lines: Vec<String> = vec![];
        let mut line = String::new();
        for word in s.split_whitespace() {
            let trial = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
            if font.size(&trial) > width && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                line = word.to_string();
            } else {
                line = trial;
            }
        }
        lines.push(line);
        let mut cut = lines.len() as i32 > max_lines;
        lines.truncate(max_lines.max(1) as usize);
        let last = lines.last().unwrap().clone();
        if cut || font.size(&last) > width {
            let mut l = last;
            while !l.is_empty() && font.size(&format!("{l}…")) > width {
                l.pop();
            }
            *lines.last_mut().unwrap() = format!("{}…", l.trim_end());
            cut = true;
        }
        (lines, cut)
    }

    /// A button's label centred in its rect, fitted by `fit_lines`. Returns whether it had to be cut.
    fn label(&self, cv: &mut Canvas, rect: Rect, label: &str, color: Color, font: &Font) -> bool {
        let lh = font.linesize;
        let (lines, cut) = Self::fit_lines(font, label, rect.w - 12, ((rect.h - 4) / lh).max(1));
        let mut y = rect.centery() as f32 - (lh * lines.len() as i32) as f32 / 2.0;
        for ln in &lines {
            self.text(cv, ln, (rect.centerx(), (y + lh as f32 / 2.0) as i32), color, font, Anchor::Center);
            y += lh as f32;
        }
        cut
    }

    pub fn label_fits(font: &Font, label: &str, width: i32, height: i32) -> bool {
        !Self::fit_lines(font, label, width - 12, ((height - 4) / font.linesize).max(1)).1
    }

    /// A button width that fits its label on one line in the current font.
    pub fn bw(&self, label: &str, min_w: i32) -> i32 {
        min_w.max(self.f().bold.size(label) + 28)
    }

    #[allow(clippy::too_many_arguments)]
    fn button(
        &mut self,
        cv: &mut Canvas,
        rect: Rect,
        label: &str,
        click: Act,
        id: &str,
        style: Style,
        enabled: bool,
        tip: Option<&str>,
        active: bool,
        font: Option<Font>,
    ) -> bool {
        let font = font.unwrap_or_else(|| self.f().bold.clone());
        let cut_tip;
        let tip = match tip {
            None if !Self::label_fits(&font, label, rect.w, rect.h) => {
                cut_tip = label.to_string();
                Some(cut_tip.as_str())
            }
            t => t,
        };
        let over = self.register(rect, Kind::Button { click }, id, enabled, tip);
        let down = over && self.pressed.as_deref() == Some(id);
        let (base, hover) = match style {
            Style::Primary => (BTN_PRIMARY, BTN_PRIMARY_HOVER),
            Style::Danger => (BTN_DANGER, BTN_DANGER_HOVER),
            Style::Normal => (BTN, BTN_HOVER),
        };
        let mut fill = if !enabled {
            [32, 36, 46]
        } else if down {
            BTN_DOWN
        } else if over {
            hover
        } else {
            base
        };
        if active {
            fill = if over { BTN_PRIMARY_HOVER } else { BTN_PRIMARY };
        }
        cv.fill_rect(rect, fill, 8);
        if (over && enabled) || active {
            cv.stroke_rect(rect, if active { ACCENT } else { [120, 132, 156] }, 8);
        }
        self.label(cv, rect, label, if enabled { INK } else { DIM }, &font);
        over
    }

    fn btn(&mut self, cv: &mut Canvas, rect: Rect, label: &str, click: Act, id: &str) -> bool {
        self.button(cv, rect, label, click, id, Style::Normal, true, None, false, None)
    }

    fn toggle(&mut self, cv: &mut Canvas, rect: Rect, value: bool, key: &str, enabled: bool, tip: &str) {
        let click = Act::Set { key: key.into(), value: Value::Bool(!value), save: true };
        let over = self.register(rect, Kind::Button { click }, key, enabled, Some(tip));
        let track = Rect::new(rect.x, rect.centery() - 11, 44, 22);
        let on_col = if enabled { [40, 150, 200] } else { [40, 70, 90] };
        cv.fill_rect(
            track,
            if value {
                on_col
            } else if over {
                [70, 76, 90]
            } else {
                [50, 56, 68]
            },
            11,
        );
        let knob = (if value { track.right() - 11 } else { track.x + 11 }, track.centery());
        cv.circle(knob, 8, if enabled { INK } else { DIM });
        let f = self.f().small.clone();
        self.text(
            cv,
            &tr(if value { "On" } else { "Off" }),
            (track.right() + 10, track.centery()),
            if enabled { TEXT } else { DIM },
            &f,
            Anchor::MidLeft,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn segmented(
        &mut self,
        cv: &mut Canvas,
        rect: Rect,
        labels: &[String],
        current: usize,
        key: &str,
        options: &[Value],
        enabled: bool,
        tip: &str,
    ) {
        let n = labels.len() as i32;
        let w = rect.w / n;
        let small = self.f().small.clone();
        for (i, lab) in labels.iter().enumerate() {
            let r = Rect::new(rect.x + i as i32 * w, rect.y, w - 4, rect.h);
            let t = if Self::label_fits(&small, lab, r.w, r.h) { tip.to_string() } else { lab.clone() };
            let click = Act::Set { key: key.into(), value: options[i].clone(), save: true };
            let over = self.register(r, Kind::Button { click }, &format!("{key}#{i}"), enabled, Some(&t));
            let on = i == current;
            let mut fill = if on {
                if over { BTN_PRIMARY_HOVER } else { BTN_PRIMARY }
            } else if over && enabled {
                BTN_HOVER
            } else {
                BTN
            };
            if !enabled {
                fill = if on { [40, 60, 74] } else { [32, 36, 46] };
            }
            cv.fill_rect(r, fill, 6);
            if on {
                cv.stroke_rect(r, if enabled { ACCENT } else { DIM }, 6);
            }
            self.label(cv, r, lab, if enabled { INK } else { DIM }, &small);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn slider(&mut self, cv: &mut Canvas, rect: Rect, value: f64, s: &Setting, enabled: bool, tip: &str) {
        let (lo, hi) = (s.lo, s.hi);
        let key = s.key.as_str();
        let bx = Rect::new(rect.right() - 76, rect.y, 76, rect.h);
        let track = Rect::new(rect.x, rect.centery() - 3, rect.w - 92, 6);
        let grab = Rect::new(track.x - 8, rect.y, track.w + 16, rect.h);
        let mut over = false;
        if enabled {
            over = self.register(grab, Kind::Slider { track, lo, hi, step: s.step, key: key.into() }, key, true, Some(tip));
        }
        let frac = if hi == lo { 0.0 } else { (value - lo) / (hi - lo) };
        cv.fill_rect(track, [50, 56, 68], 3);
        cv.fill_rect(
            Rect::new(track.x, track.y, 3.max((track.w as f64 * frac) as i32), track.h),
            if enabled { [40, 150, 200] } else { [40, 70, 90] },
            3,
        );
        let dragging = self.drag.as_ref().is_some_and(|d| d.id == key);
        cv.circle(
            (track.x + (track.w as f64 * frac) as i32, track.centery()),
            if over || dragging { 9 } else { 7 },
            if over || dragging {
                ACCENT
            } else if enabled {
                INK
            } else {
                DIM
            },
        );
        let pct = s.fmt.contains('%');
        let editing = self.edit.as_ref().is_some_and(|e| e.id == key);
        let caret = if (self.now * 2.0) as i64 % 2 != 0 { "|" } else { "" };
        let shown = if editing { format!("{}{caret}", self.edit.as_ref().unwrap().text) } else { py_fmt(&s.fmt, value) };
        if enabled {
            let text = if pct { g(value * 100.0) } else { g(value) };
            self.register(
                bx,
                Kind::Edit {
                    text,
                    target: EditTarget::Setting { key: key.into(), pct, int: false },
                    alpha: false,
                    limit: 12,
                    compact: false,
                },
                key,
                true,
                Some("Click to type a value, Enter to apply."),
            );
        }
        cv.fill_rect(bx, if editing { [10, 12, 18] } else { [30, 34, 44] }, 6);
        cv.stroke_rect(bx, if editing { ACCENT } else { BORDER }, 6);
        let f = self.f().small.clone();
        self.text(cv, &shown, bx.center(), if enabled { INK } else { DIM }, &f, Anchor::Center);
    }

    fn number_field(&mut self, cv: &mut Canvas, rect: Rect, value: i64, key: &str, tip: &str) {
        let editing = self.edit.as_ref().is_some_and(|e| e.id == key);
        self.register(
            rect,
            Kind::Edit {
                text: value.to_string(),
                target: EditTarget::Setting { key: key.into(), pct: false, int: true },
                alpha: false,
                limit: 12,
                compact: false,
            },
            key,
            true,
            Some(tip),
        );
        cv.fill_rect(rect, if editing { [10, 12, 18] } else { [30, 34, 44] }, 6);
        cv.stroke_rect(rect, if editing { ACCENT } else { BORDER }, 6);
        let caret = if (self.now * 2.0) as i64 % 2 != 0 { "|" } else { "" };
        let shown = if editing { format!("{}{caret}", self.edit.as_ref().unwrap().text) } else { value.to_string() };
        let f = self.f().small.clone();
        self.text(cv, &shown, (rect.x + 10, rect.centery()), INK, &f, Anchor::MidLeft);
    }

    /// A one-line text box (names, share codes). Click, type, Enter commits (see [`Menu::take_text_commits`]), Esc cancels.
    pub fn text_field(&mut self, cv: &mut Canvas, rect: Rect, value: &str, id: &str, tip: Option<&str>, limit: usize, compact: bool) {
        let editing = self.edit.as_ref().is_some_and(|e| e.id == id);
        self.register(
            rect,
            Kind::Edit { text: value.into(), target: EditTarget::Text(id.into()), alpha: true, limit, compact },
            id,
            true,
            tip,
        );
        cv.fill_rect(rect, if editing { [10, 12, 18] } else { [30, 34, 44] }, 6);
        cv.stroke_rect(rect, if editing { ACCENT } else { BORDER }, 6);
        let caret = if (self.now * 2.0) as i64 % 2 != 0 { "|" } else { "" };
        let shown = if editing { format!("{}{caret}", self.edit.as_ref().unwrap().text) } else { value.to_string() };
        let f = self.f().small.clone();
        self.text(cv, &shown, (rect.x + 10, rect.centery()), INK, &f, Anchor::MidLeft);
    }

    fn chip(&self, cv: &mut Canvas, pos: (i32, i32), label: &str) -> Rect {
        let f = &self.f().small;
        let up = label.to_uppercase();
        let r = Rect::new(pos.0, pos.1, f.size(&up) + 12, f.linesize + 2);
        cv.fill_rect(r, tag_color(label), 5);
        self.text(cv, &up, r.center(), [12, 14, 18], f, Anchor::Center);
        r
    }
}

// --- drawing ---------------------------------------------------------------------------------------------------------------------
impl Menu {
    /// Draw the open screen onto `cv` (the HUD canvas; the veil, panel, page, flash message and tooltip) and register its hit regions.
    pub fn draw(&mut self, cv: &mut Canvas, cfg: &Config, three_d: bool, mouse: (i32, i32), now: f64) {
        let Some(screen) = self.screen.clone() else { return };
        self.ensure_fonts(cfg.get_bool("access.larger_text"));
        self.hits.clear();
        self.tip = None;
        self.mouse = mouse;
        self.now = now;
        let (w, h) = (cv.w, cv.h);
        cv.clip = None;
        cv.veil([4, 5, 8], self.veil_alpha);
        let screen = if matches!(screen.as_str(), "pause" | "confirm_quit" | "settings") {
            screen
        } else {
            self.screen = Some("pause".into());
            "pause".into()
        };
        let (pw, mut ph) = if matches!(screen.as_str(), "pause" | "confirm_quit") {
            (440, 650.min(h - 20))
        } else {
            (980.min(w - 40), (if h < 1100 { 680 } else { 860 }).min(h - 30))
        };
        if screen == "confirm_quit" {
            ph = 250;
        }
        let rect = Rect::new((w - pw) / 2, (h - ph) / 2, pw, ph);
        cv.fill_rect(rect, PANEL, 16);
        cv.stroke_rect(rect, BORDER, 16);
        match screen.as_str() {
            "pause" => self.page_pause(cv, rect, cfg),
            "confirm_quit" => self.page_confirm_quit(cv, rect),
            _ => self.page_settings(cv, rect, cfg, three_d),
        }
        if let Some((text, until, color)) = self.message.clone()
            && now < until
        {
            let f = self.f().small.clone();
            self.text(cv, &text, (rect.centerx(), rect.bottom() + 10), color, &f, Anchor::MidTop);
        }
        if let Some((_, text)) = self.tip.clone() {
            self.draw_tip(cv, &text);
        }
    }

    fn draw_tip(&self, cv: &mut Canvas, text: &str) {
        let f = &self.f().small;
        let width = 360;
        let mut lines: Vec<String> = vec![];
        let mut line = String::new();
        for word in text.split_whitespace() {
            let trial = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
            if f.size(&trial) > width - 20 && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
                line = word.to_string();
            } else {
                line = trial;
            }
        }
        lines.push(line);
        let lh = f.linesize;
        let mut bx = Rect::new(0, 0, width, 14 + lh * lines.len() as i32);
        bx.x = (self.mouse.0 + 16).min(cv.w - width - 8);
        bx.y = self.mouse.1 + 20;
        if bx.bottom() > cv.h - 8 {
            bx.y = self.mouse.1 - 10 - bx.h;
        }
        cv.fill_rect(bx, [8, 10, 14], 8);
        cv.stroke_rect(bx, ACCENT, 8);
        for (i, ln) in lines.iter().enumerate() {
            self.text(cv, ln, (bx.x + 10, bx.y + 7 + i as i32 * lh), TEXT, f, Anchor::TopLeft);
        }
    }

    fn page_pause(&mut self, cv: &mut Canvas, rect: Rect, cfg: &Config) {
        let fo = self.fonts_now();
        self.text(cv, &tr("PAUSED"), (rect.centerx(), rect.y + 22), INK, &fo.title, Anchor::MidTop);
        let lab = cfg.lab();
        let mode_title = if cfg.pet() {
            "Pet"
        } else if lab {
            "Lab"
        } else {
            "Play"
        };
        let sub = format!("Kick the Fly {}   ·   {mode_title} mode", self.version);
        self.text(cv, &sub, (rect.centerx(), rect.y + 76), LABEL, &fo.small, Anchor::MidTop);
        let challenges = if lab { "Lab tools" } else { "Challenges" };
        let mode_label = format!("Mode: {mode_title}");
        let items: Vec<(String, &str, Style, String)> = vec![
            ("Resume".into(), "resume", Style::Primary, "Back to the fly (Esc).".into()),
            (
                challenges.into(),
                if lab { "lab" } else { "challenges" },
                Style::Normal,
                if lab { "Validation results, assays, repeated trials, data export and protocol files." } else { "Games with a goal and a score." }.into(),
            ),
            ("Neurodex".into(), "neurodex", Style::Normal, "The cell types you have discovered, with what the dataset says about each. Default key D.".into()),
            ("Settings".into(), "settings", Style::Normal, "Graphics, audio, brain, controls and accessibility.".into()),
            (
                "Mic and streamer".into(),
                "live_inputs",
                Style::Normal,
                "Opt-in: sound from your microphone drives the fly's Johnston's organ, and Twitch viewers vote on what happens. Both are off every time the game starts and show a red indicator while on.".into(),
            ),
            (
                "Fly arcade".into(),
                "arcade",
                Style::Normal,
                "A tournament of flies (each brain steers and shoots in 1v1 duels) and fly racing with in-game points. Points are not money.".into(),
            ),
            (
                "Mini-papers".into(),
                "minipapers",
                Style::Normal,
                "Short guided experiments that reproduce a classic fly paper: state a hypothesis, run it on the model and see your result next to what the paper found.".into(),
            ),
            (
                "Share".into(),
                "share",
                Style::Normal,
                "Make a short code for your surgery, loadout, protocol, challenge setup or Lab parameters, or import one and see what it would change first.".into(),
            ),
            (
                "Save State".into(),
                "save_state",
                Style::Normal,
                "Save the whole simulation: every neuron's voltage, the learned synapses, surgery, the room and the flies.".into(),
            ),
            ("Load State".into(), "load_state", Style::Normal, "Go back to a saved moment.".into()),
            (
                mode_label,
                "toggle_mode",
                Style::Normal,
                "Switch between Play (the game), Lab (research tools), and Pet (one persistent fly). Saved in your settings.".into(),
            ),
            ("Quit".into(), "quit", Style::Danger, "Asks first. Training memory is saved.".into()),
        ];
        let lh = fo.text.linesize;
        let (bw, bh, gap) = (300, 34.max(40.min(lh + 10)), 5);
        let mut y = rect.y + 108;
        for (label, action, style, tip) in items {
            let r = Rect::new(rect.centerx() - bw / 2, y, bw, bh);
            self.button(
                cv,
                r,
                &tr(&label),
                Act::Action(action.into()),
                &format!("pause:{action}"),
                style,
                true,
                Some(&tr(&tip)),
                false,
                None,
            );
            y += bh + gap;
        }
    }

    fn page_confirm_quit(&mut self, cv: &mut Canvas, rect: Rect) {
        let fo = self.fonts_now();
        self.text(cv, &tr("Quit Kick the Fly?"), (rect.centerx(), rect.y + 30), INK, &fo.head, Anchor::MidTop);
        let msg = tr("The fly's training memory and your settings are saved.");
        let lines = Self::fit_lines(&fo.small, &msg, rect.w - 48, 3).0;
        for (i, ln) in lines.iter().enumerate() {
            self.text(cv, ln, (rect.centerx(), rect.y + 76 + i as i32 * fo.small.linesize), LABEL, &fo.small, Anchor::MidTop);
        }
        self.button(
            cv,
            Rect::new(rect.centerx() - 170, rect.bottom() - 90, 160, 50),
            &tr("Quit"),
            Act::Action("quit_now".into()),
            "cq:quit",
            Style::Danger,
            true,
            None,
            false,
            None,
        );
        self.btn(cv, Rect::new(rect.centerx() + 10, rect.bottom() - 90, 160, 50), &tr("Cancel"), Act::Back, "cq:cancel");
    }

    fn page_settings(&mut self, cv: &mut Canvas, rect: Rect, cfg: &Config, three_d: bool) {
        let fo = self.fonts_now();
        self.text(cv, "SETTINGS", (rect.x + 24, rect.y + 16), INK, &fo.head, Anchor::TopLeft);
        let tabs = config::TABS;
        let need: Vec<i32> = tabs.iter().map(|t| fo.bold.size(&tr(t)) + 24).collect();
        let spare = ((rect.w - 48) - need.iter().sum::<i32>() - 8 * tabs.len() as i32).max(0) / tabs.len() as i32;
        let mut tx = rect.x + 24;
        for (tab, w) in tabs.iter().zip(&need) {
            self.button(
                cv,
                Rect::new(tx, rect.y + 54, w + spare, 38),
                &tr(tab),
                Act::Tab((*tab).into()),
                &format!("tab:{tab}"),
                Style::Normal,
                true,
                None,
                *tab == self.tab,
                None,
            );
            tx += w + spare + 8;
        }
        let body = Rect::new(rect.x + 16, rect.y + 104, rect.w - 32, rect.h - 104 - 70);
        let key = self.scroll_key();
        let off = self.scroll.get(&key).copied().unwrap_or(0.0) as i32;
        self.clip = Some(body);
        cv.clip = Some(body);
        let mut y = body.y + 6 - off;
        let label_w = 250;
        let ctl_x = body.x + label_w + 150;
        let ctl_w = body.right() - ctl_x - 16;
        let cur_tab = self.tab.clone();
        for s in config::schema().settings.iter().filter(|s| s.tab == cur_tab) {
            if s.only == "linux" && !cfg!(target_os = "linux") {
                continue;
            }
            let on = !(s.only == "3d" && !three_d);
            let mut chips: Vec<&str> = vec![];
            if !s.tag.is_empty() {
                chips.push(&s.tag);
            }
            if s.restart {
                chips.push("RESTART");
            }
            if !on {
                chips.push("3D ONLY");
            }
            let (mut extra, mut own_line) = (0, false);
            let chips_end = (body.x + 12 + fo.text.size(&tr(&s.label)) + 10).max(body.x + label_w)
                + chips.iter().map(|c| fo.small.size(&c.to_uppercase()) + 18).sum::<i32>();
            let cx0 = ctl_x.max(chips_end + 8);
            let labels: Vec<String> = if !s.labels.is_empty() {
                s.labels.clone()
            } else {
                s.options.iter().map(|o| o.as_str().map(String::from).unwrap_or_else(|| o.to_string())).collect()
            };
            if s.kind == "choice" && labels.len() <= 4 {
                let seg = (ctl_x + ctl_w - cx0) / labels.len() as i32 - 4;
                if labels.iter().any(|lb| fo.small.size(lb) > seg - 12) {
                    extra = fo.small.linesize;
                    if labels.iter().any(|lb| Self::fit_lines(&fo.small, lb, seg - 12, 2).1) {
                        own_line = true;
                        extra = 38 + fo.small.linesize;
                    }
                }
            }
            let row = Rect::new(body.x, y, body.w, 44 + extra);
            if row.contains(self.mouse) && body.contains(self.mouse) {
                cv.fill_rect(row, ROW_HOVER, 8);
            }
            let tip = tr(&s.tip) + if on { "" } else { "  (3D game only.)" };
            let ly = if own_line { row.y + 22 } else { row.centery() };
            let lr = self.text(cv, &tr(&s.label), (row.x + 12, ly), if on { TEXT } else { DIM }, &fo.text, Anchor::MidLeft);
            self.register(Rect::new(row.x, row.y, label_w + 150, row.h), Kind::Label, &format!("label:{}", s.key), true, Some(&tip));
            let mut cx = (lr.right() + 10).max(row.x + label_w);
            for chip in &chips {
                cx = self.chip(cv, (cx, ly - 10), chip).right() + 6;
            }
            let ctl = if own_line {
                Rect::new(body.x + 24, row.y + 44, body.w - 40, 30 + fo.small.linesize)
            } else {
                Rect::new(cx0, row.y + 7, ctl_x + ctl_w - cx0, 30 + extra)
            };
            let v = cfg.get(&s.key).clone();
            match s.kind.as_str() {
                "bool" => self.toggle(cv, ctl, v.as_bool().unwrap_or(false), &s.key, on, &tip),
                "choice" => {
                    let i = s.options.iter().position(|o| num_eq(o, &v)).unwrap_or(0);
                    if labels.len() <= 4 {
                        self.segmented(cv, ctl, &labels, i, &s.key, &s.options, on, &tip);
                    } else {
                        let mid = 152.max(labels.iter().map(|lb| fo.bold.size(lb)).max().unwrap_or(0) + 16);
                        let n = s.options.len();
                        let prev = s.options[(i + n - 1) % n].clone();
                        let next = s.options[(i + 1) % n].clone();
                        self.button(
                            cv,
                            Rect::new(ctl.x, ctl.y, 44, ctl.h),
                            "<",
                            Act::Set { key: s.key.clone(), value: prev, save: true },
                            &format!("{}<", s.key),
                            Style::Normal,
                            on,
                            Some(&tip),
                            false,
                            None,
                        );
                        self.text(cv, &labels[i], (ctl.x + 44 + mid / 2, ctl.centery()), INK, &fo.bold, Anchor::Center);
                        self.button(
                            cv,
                            Rect::new(ctl.x + 44 + mid, ctl.y, 44, ctl.h),
                            ">",
                            Act::Set { key: s.key.clone(), value: next, save: true },
                            &format!("{}>", s.key),
                            Style::Normal,
                            on,
                            Some(&tip),
                            false,
                            None,
                        );
                    }
                }
                "float" => self.slider(cv, ctl, v.as_f64().unwrap_or(0.0), s, on, &tip),
                "int" => {
                    self.number_field(cv, Rect::new(ctl.x, ctl.y, 180, ctl.h), v.as_i64().unwrap_or(0), &s.key, &tip);
                    self.button(
                        cv,
                        Rect::new(ctl.x + 192, ctl.y, 110, ctl.h),
                        "Random",
                        Act::RandomSeed(s.key.clone()),
                        &format!("{}:rand", s.key),
                        Style::Normal,
                        true,
                        Some("Pick a new random seed."),
                        false,
                        Some(fo.small.clone()),
                    );
                }
                _ => {}
            }
            y += 48 + extra;
        }
        if self.tab == "Brain" {
            y = self.engine_info(cv, body, y + 4);
        }
        if self.tab == "Controls" {
            y = self.keybinds(cv, body, y + 8, cfg, three_d);
        }
        self.content_h.insert(key.clone(), (y + off - body.bottom() + 12).max(0));
        cv.clip = None;
        self.clip = None;
        let ch = self.content_h[&key];
        if ch > 0 {
            let frac = body.h as f32 / (body.h + ch) as f32;
            let bar_h = 30.max((body.h as f32 * frac) as i32);
            let by = body.y + ((body.h - bar_h) as f32 * (off as f32 / ch.max(1) as f32)) as i32;
            cv.fill_rect(Rect::new(body.right() - 6, by, 4, bar_h), [60, 66, 80], 2);
        }
        let fy = rect.bottom() - 58;
        let reset_w = self.bw("Reset to defaults", 200);
        if self.tab != "Help" {
            let tip = format!("Put every {} setting back to how the game ships.", self.tab);
            let id = format!("reset:{}", self.tab);
            self.button(
                cv,
                Rect::new(rect.x + 24, fy, reset_w, 42),
                "Reset to defaults",
                Act::ResetTab,
                &id,
                Style::Normal,
                true,
                Some(&tip),
                false,
                None,
            );
        }
        self.button(
            cv,
            Rect::new(rect.right() - 164, fy, 140, 42),
            &tr("Back"),
            Act::Back,
            "settings:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
        let note = "Changes apply right away and are saved.";
        let left = rect.x + 24 + if self.tab != "Help" { reset_w + 16 } else { 0 };
        let nw = fo.small.size(note);
        let nx = rect.centerx().max(left + nw / 2);
        if nx + nw / 2 < rect.right() - 176 {
            self.text(cv, note, (nx, fy + 21), LABEL, &fo.small, Anchor::Center);
        }
    }

    fn engine_info(&self, cv: &mut Canvas, body: Rect, mut y: i32) -> i32 {
        let f = self.f().small.clone();
        self.text(cv, "ENGINE", (body.x + 12, y + 8), LABEL, &f, Anchor::MidLeft);
        y += 22;
        for ln in &self.engine_lines {
            self.text(cv, ln, (body.x + 12, y + 8), TEXT, &f, Anchor::MidLeft);
            y += 22;
        }
        y + 4
    }

    fn keybinds(&mut self, cv: &mut Canvas, body: Rect, y: i32, cfg: &Config, three_d: bool) -> i32 {
        let fo = self.fonts_now();
        let sc = config::schema();
        let k = self.text(cv, "KEYS", (body.x + 12, y), LABEL, &fo.small, Anchor::TopLeft);
        let wy = self.wrapped(
            cv,
            "Click a key, then press the new one (Esc cancels). A key that's taken swaps with that action.",
            (k.right() + 16, y),
            body.right() - 16 - k.right() - 16,
            DIM,
            &fo.small,
            2,
        ) + 8;
        let mut y = (y + 26).max(wy);
        let need = sc.actions.iter().chain(sc.pad_actions.iter()).map(|(_, l, _)| fo.small.size(l)).max().unwrap_or(0) + 200;
        let ncol = if need <= (body.w - 24) / 2 { 2 } else { 1 };
        let col_w = (body.w - 24) / ncol;
        let conflicts: Vec<String> = cfg.conflicts().into_iter().flat_map(|(_, a)| a).collect();
        let movement_3d_only = ["forward", "back", "left", "right", "sprint", "crouch", "free_mouse", "duel", "panel", "menu_size"];
        for (i, (action, label, _)) in sc.actions.iter().enumerate() {
            let cx = body.x + 12 + (i as i32 % ncol) * col_w;
            let ry = y + (i as i32 / ncol) * 38;
            let active = self.capture.as_deref() == Some(action.as_str());
            let dim = movement_3d_only.contains(&action.as_str()) && !three_d;
            self.text(cv, label, (cx, ry + 16), if dim { DIM } else { TEXT }, &fo.small, Anchor::MidLeft);
            let key = cfg.key_of(action);
            let name = if active {
                "press a key...".to_string()
            } else if key.is_empty() {
                tr("unbound")
            } else {
                key.to_string()
            };
            self.button(
                cv,
                Rect::new(cx + col_w - 190, ry + 2, 170, 30),
                &name,
                Act::Capture(action.clone()),
                &format!("key:{action}"),
                if conflicts.contains(action) { Style::Danger } else { Style::Normal },
                true,
                Some(&format!("{label}: click, then press a key.")),
                active,
                Some(fo.small.clone()),
            );
        }
        y += ((sc.actions.len() as i32 + ncol - 1) / ncol) * 38 + 10;
        if !three_d {
            return y;
        }
        let status = self.pad_status.clone().unwrap_or_else(|| "none connected".into());
        self.text(cv, &format!("GAMEPAD ({status})"), (body.x + 12, y), LABEL, &fo.small, Anchor::TopLeft);
        let y2 = self.wrapped(
            cv,
            "Click a binding, then press the button or push the stick. Backspace unbinds.",
            (body.x + 12, y + 18),
            body.w - 24,
            DIM,
            &fo.small,
            2,
        ) + 8;
        y = y2;
        for (i, (action, label, _)) in sc.pad_actions.iter().enumerate() {
            let cx = body.x + 12 + (i as i32 % ncol) * col_w;
            let ry = y + (i as i32 / ncol) * 38;
            let cap = format!("pad:{action}");
            let active = self.capture.as_deref() == Some(cap.as_str());
            self.text(cv, label, (cx, ry + 16), TEXT, &fo.small, Anchor::MidLeft);
            let binding = cfg.pad.iter().find(|(a, _)| a == action).map(|(_, b)| b.as_str()).unwrap_or("");
            let name = if active { "press or push...".to_string() } else { pad_pretty(binding) };
            self.button(
                cv,
                Rect::new(cx + col_w - 190, ry + 2, 170, 30),
                &name,
                Act::Capture(cap),
                &format!("pad:{action}"),
                Style::Normal,
                true,
                Some(&format!("{label}: click, then press a gamepad button or push a stick.")),
                active,
                Some(fo.small.clone()),
            );
        }
        y + ((sc.pad_actions.len() as i32 + ncol - 1) / ncol) * 38
    }
}
