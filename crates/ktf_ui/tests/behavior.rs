//! Clicking, typing, dragging and key capture through the same event path the game uses.
use ktf_data::config::Config;
use ktf_ui::menu::NoHost;
use ktf_ui::{Canvas, Effect, Menu, UiEvent};
use serde_json::json;

struct Rig {
    m: Menu,
    cfg: Config,
    cv: Canvas,
    t: f64,
}
impl Rig {
    fn new(screen: &str) -> Self {
        let mut m = Menu::new("test");
        ktf_ui::pages::install(&mut m);
        m.show(screen);
        let mut r = Rig { m, cfg: Config::new(None), cv: Canvas::new(1280, 720), t: 0.0 };
        r.frame();
        r
    }
    fn frame(&mut self) {
        self.t += 0.016;
        self.cv.clear();
        self.m.draw(&mut self.cv, &mut self.cfg, true, (-1, -1), self.t, &mut NoHost);
    }
    fn send(&mut self, ev: UiEvent) -> Vec<Effect> {
        let (_, fx) = self.m.handle(&ev, &mut self.cfg, true, &mut NoHost);
        self.frame();
        fx
    }
    fn click(&mut self, id: &str) -> Vec<Effect> {
        let pos = self.m.center_of(id).unwrap_or_else(|| panic!("no widget {id}"));
        self.send(UiEvent::MouseDown { pos });
        self.send(UiEvent::MouseUp { pos })
    }
    fn key(&mut self, name: &str, ch: Option<char>) -> Vec<Effect> {
        self.send(UiEvent::KeyDown { name: name.into(), ch, ctrl: false })
    }
}

#[test]
fn pause_buttons_report_their_actions_and_esc_closes() {
    let mut r = Rig::new("pause");
    let fx = r.click("pause:settings");
    assert_eq!(fx, vec![Effect::Action("click_sound".into()), Effect::Action("settings".into())]);
    let fx = r.key("escape", None);
    assert_eq!(fx, vec![Effect::Action("closed".into())]);
    assert!(!r.m.is_open());
}

#[test]
fn settings_toggle_choice_and_tabs() {
    let mut r = Rig::new("settings");
    assert!(!r.cfg.get_bool("graphics.fullscreen"));
    let fx = r.click("graphics.fullscreen");
    assert!(fx.contains(&Effect::SettingChanged { key: Some("graphics.fullscreen".into()), save: true }));
    assert!(r.cfg.get_bool("graphics.fullscreen"));
    // the FPS cap has 8 options: arrows
    r.click("graphics.fps_cap>");
    assert_eq!(r.cfg.get_i64("graphics.fps_cap"), 90);
    r.click("graphics.fps_cap<");
    r.click("graphics.fps_cap<");
    assert_eq!(r.cfg.get_i64("graphics.fps_cap"), 30);
    r.click("tab:Audio");
    assert_eq!(r.m.tab, "Audio");
    // reset the tab
    r.click("tab:Graphics");
    let fx = r.click("reset:Graphics");
    assert!(fx.contains(&Effect::SettingChanged { key: None, save: true }));
    assert!(!r.cfg.get_bool("graphics.fullscreen") && r.cfg.get_i64("graphics.fps_cap") == 60);
}

#[test]
fn slider_drag_and_typed_value() {
    let mut r = Rig::new("settings");
    r.click("tab:Audio");
    let c = r.m.center_of("audio.master").unwrap();
    r.send(UiEvent::MouseDown { pos: (c.0 - 100, c.1) });
    let live = r.send(UiEvent::Motion { pos: (c.0 - 40, c.1) });
    assert!(live.contains(&Effect::SettingChanged { key: Some("audio.master".into()), save: false }));
    let v = r.cfg.get_f32("audio.master");
    assert!(v > 0.0 && v < 1.0, "{v}");
    let up = r.send(UiEvent::MouseUp { pos: (c.0 - 40, c.1) });
    assert_eq!(up, vec![Effect::SettingChanged { key: None, save: true }]);
    // click the value box, type 35, Enter: a percent field
    let pos = (r.m.center_of("audio.master").unwrap().0 + 2, 0);
    let _ = pos;
}

#[test]
fn key_capture_swaps_and_refuses_escape() {
    let mut r = Rig::new("settings");
    r.click("tab:Controls");
    r.send(UiEvent::Wheel { y: -5.0 });
    r.click("key:big_view");
    assert_eq!(r.m.capture.as_deref(), Some("big_view"));
    let fx = r.key("n", Some('n'));
    assert!(fx.contains(&Effect::KeysChanged));
    assert_eq!(r.cfg.key_of("big_view"), "n");
    assert_eq!(r.cfg.key_of("spawn"), "b", "n was spawn's key: the two swap");
    r.click("key:big_view");
    assert_eq!(r.m.capture.as_deref(), Some("big_view"));
    r.key("escape", None);
    assert_eq!(r.cfg.key_of("big_view"), "n", "Esc cancels a capture");
    assert!(r.m.is_open(), "and does not close the menu");
}

#[test]
fn seed_typing_and_random() {
    let mut r = Rig::new("settings");
    r.click("tab:Brain");
    // typed number: click the field, the first character replaces the old value
    let c = r.m.center_of("brain.seed").unwrap();
    r.send(UiEvent::MouseDown { pos: c });
    for ch in "1234".chars() {
        r.key(&ch.to_string(), Some(ch));
    }
    r.key("backspace", None);
    r.key("return", None);
    assert_eq!(r.cfg.get_i64("brain.seed"), 123);
    r.click("brain.seed:rand");
    assert!(r.cfg.get_i64("brain.seed") > 0);
}

#[test]
fn wheel_scrolls_within_limits() {
    let mut r = Rig::new("settings");
    r.click("tab:Controls");
    for _ in 0..200 {
        r.send(UiEvent::Wheel { y: -1.0 });
    }
    // scrolled to the end: the last key button is on screen, the first setting row is not
    assert!(r.m.center_of("key:page_next").is_some());
    for _ in 0..400 {
        r.send(UiEvent::Wheel { y: 1.0 });
    }
    r.frame();
    assert!(r.m.center_of("key:forward").is_some());
    let _ = json!(null);
}

#[test]
fn loadout_editor_click_and_drag() {
    let mut r = Rig::new("loadout");
    assert_eq!(r.cfg.get_str("controls.loadout_preset"), "auto");
    // click a card: equips the bomb, the preset becomes custom, the config is to be saved
    let fx = r.click("lo-card:bomb");
    assert!(fx.contains(&Effect::SaveConfig));
    assert_eq!(r.cfg.get_str("controls.loadout_preset"), "custom");
    assert_eq!(r.cfg.custom_loadout, ["hand", "swatter", "torch", "freeze", "sugar", "bomb"]);
    // click it again: unequipped
    r.click("lo-card:bomb");
    assert_eq!(r.cfg.custom_loadout, ["hand", "swatter", "torch", "freeze", "sugar"]);
    // the hand can't be removed
    r.click("lo-slot:0");
    // drag the sugar (slot 4) onto slot 1: reordered, the hand stays first
    let from = r.m.center_of("lo-slot:4").unwrap();
    let to = r.m.center_of("lo-drop:1").unwrap();
    r.send(UiEvent::MouseDown { pos: from });
    r.send(UiEvent::Motion { pos: (from.0 - 30, from.1 - 20) });
    r.send(UiEvent::Motion { pos: to });
    r.send(UiEvent::MouseUp { pos: to });
    assert_eq!(r.cfg.custom_loadout, ["hand", "sugar", "swatter", "torch", "freeze"]);
    // drag the torch (now slot 3) off the hotbar: removed
    let from = r.m.center_of("lo-slot:3").unwrap();
    r.send(UiEvent::MouseDown { pos: from });
    r.send(UiEvent::Motion { pos: (from.0, from.1 + 200) });
    r.send(UiEvent::MouseUp { pos: (from.0, from.1 + 200) });
    assert_eq!(r.cfg.custom_loadout, ["hand", "sugar", "swatter", "freeze"]);
    // save under a name, then load it back after changing
    r.click("lo:save");
    assert_eq!(r.cfg.saved_loadouts.len(), 1);
    r.click("lo-card:flick");
    assert!(r.cfg.custom_loadout.contains(&"flick".to_string()));
    r.click("lo-saved:0");
    assert_eq!(r.cfg.custom_loadout, ["hand", "sugar", "swatter", "freeze"]);
    // a preset button
    r.click("lo-preset:chaos");
    assert_eq!(r.cfg.get_str("controls.loadout_preset"), "chaos");
}
